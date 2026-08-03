// Mantic-Mind — what running a model costs, and whether a node can pay it.
//
// The change this file makes real is SHAPE, not precision:
//
//   before  estimate_inference_vram_mb() -> one number -> nodes_with_available_vram()
//   after   ResourceFootprint{vram, ram, disk} -> evaluate_fit()
//
// Soma's cost is RAM + disk + optional VRAM, and no amount of tuning a VRAM
// scalar expresses that. NodeInfo::disk_free_mb was already collected by the
// health poll and never consulted for placement, which is the same gap seen from
// the other end.

#include "common/footprint.hpp"

#include "common/gguf_metadata.hpp"
#include "common/models.hpp"

#include <algorithm>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

namespace mm {

namespace {

constexpr std::int64_t kMiB = 1024 * 1024;

} // namespace

// ── ResourceFootprint ─────────────────────────────────────────────────────────

bool ResourceFootprint::empty() const {
    return vram_mb <= 0 && ram_mb <= 0 && disk_mb <= 0;
}

std::int64_t ResourceFootprint::total_mb() const {
    return vram_mb + ram_mb + disk_mb;
}

std::int64_t ResourceFootprint::dominant_mb() const {
    return std::max({vram_mb, ram_mb, disk_mb});
}

// ── capacity ──────────────────────────────────────────────────────────────────

HostCapacity capacity_of(const NodeInfo& node) {
    HostCapacity c;
    c.vram_total_mb = node.metrics.gpu_vram_total_mb;
    c.vram_free_mb =
        std::max<std::int64_t>(0, node.metrics.gpu_vram_total_mb - node.metrics.gpu_vram_used_mb);
    c.ram_total_mb = node.metrics.ram_total_mb;
    c.ram_free_mb = std::max<std::int64_t>(0, node.metrics.ram_total_mb - node.metrics.ram_used_mb);
    // Collected by the health poll since it was written, and read here for the
    // first time.
    c.disk_free_mb = node.disk_free_mb;
    return c;
}

// ── fit ───────────────────────────────────────────────────────────────────────

FitQuality evaluate_fit(const ResourceFootprint& footprint,
                        const HostCapacity& capacity,
                        const CapacityPolicy& policy,
                        std::string* out_reason) {
    const auto reject = [&](const std::string& why) {
        if (out_reason != nullptr) *out_reason = why;
        return FitQuality::None;
    };

    // Disk first, and as a HARD constraint. There is nothing to trade it
    // against — you cannot offload disk to RAM the way you can offload weights
    // from VRAM — which is exactly why a single scalar could not express it.
    //
    // Checked even when the footprint asks for NO disk. A node with no room left
    // cannot write a KV checkpoint or spill, so it cannot host anything, whatever
    // the model costs. `disk_free_mb == 0` means "not reported" rather than
    // "full": the field defaults to zero and an older node never sends it, so
    // enforcing against it would exclude every node that predates the field.
    if (capacity.disk_free_mb > 0) {
        const auto free_after = capacity.disk_free_mb - footprint.disk_mb;
        if (free_after < policy.disk_headroom_mb) {
            return reject("needs " + std::to_string(footprint.disk_mb) + " MB disk, node has " +
                          std::to_string(capacity.disk_free_mb) + " MB free (keeping " +
                          std::to_string(policy.disk_headroom_mb) + " MB headroom)");
        }
    }

    if (footprint.ram_mb > 0) {
        const auto free_after = capacity.ram_free_mb - footprint.ram_mb;
        if (free_after < policy.ram_headroom_mb) {
            return reject("needs " + std::to_string(footprint.ram_mb) + " MB RAM, node has " +
                          std::to_string(capacity.ram_free_mb) + " MB free (keeping " +
                          std::to_string(policy.ram_headroom_mb) + " MB headroom)");
        }
    }

    if (footprint.vram_mb <= 0) {
        if (out_reason != nullptr) *out_reason = "fits (no VRAM required)";
        return FitQuality::Native;
    }

    if (capacity.vram_free_mb - footprint.vram_mb >= policy.vram_headroom_mb) {
        if (out_reason != nullptr) *out_reason = "fits in VRAM";
        return FitQuality::Native;
    }

    // Offload: satisfy a VRAM-shaped need partly out of system RAM. Weighted
    // below 1.0 because CPU-offloaded weights are slower and less reliable, so a
    // node that only fits this way must rank below one that fits natively rather
    // than merely differently.
    if (capacity.vram_total_mb < policy.min_gpu_for_offload_mb) {
        return reject("needs " + std::to_string(footprint.vram_mb) +
                      " MB VRAM; node has no GPU large enough to offload against (" +
                      std::to_string(capacity.vram_total_mb) + " MB total)");
    }

    const auto ram_for_offload = static_cast<std::int64_t>(
        static_cast<double>(std::max<std::int64_t>(
            0, capacity.ram_free_mb - footprint.ram_mb - policy.ram_headroom_mb)) *
        policy.ram_offload_weight);
    const auto vram_usable =
        std::max<std::int64_t>(0, capacity.vram_free_mb - policy.vram_headroom_mb);
    if (vram_usable + ram_for_offload >= footprint.vram_mb) {
        if (out_reason != nullptr) {
            *out_reason = "fits by offloading " + std::to_string(footprint.vram_mb - vram_usable) +
                          " MB to system RAM";
        }
        return FitQuality::Offload;
    }

    return reject("needs " + std::to_string(footprint.vram_mb) + " MB VRAM, node has " +
                  std::to_string(capacity.vram_free_mb) + " MB free and cannot offload the rest");
}

double capacity_score(const ResourceFootprint& footprint,
                      const HostCapacity& capacity,
                      const CapacityPolicy& policy) {
    const auto fit = evaluate_fit(footprint, capacity, policy, nullptr);
    if (fit == FitQuality::None) return 0.0;

    // Headroom left AFTER placement, as a fraction of what the node has. Ranking
    // on leftover rather than on raw size is what stops every agent piling onto
    // the largest node and then thrashing it.
    const auto frac = [](std::int64_t left, std::int64_t total) {
        if (total <= 0) return 1.0; // an axis the node does not have is not a constraint
        return std::clamp(static_cast<double>(left) / static_cast<double>(total), 0.0, 1.0);
    };

    const double vram = frac(capacity.vram_free_mb - footprint.vram_mb, capacity.vram_total_mb);
    const double ram = frac(capacity.ram_free_mb - footprint.ram_mb, capacity.ram_total_mb);
    const double disk = capacity.disk_free_mb > 0
                            ? frac(capacity.disk_free_mb - footprint.disk_mb, capacity.disk_free_mb)
                            : 1.0;

    const double base = 0.5 * vram + 0.3 * ram + 0.2 * disk;
    // A native fit always outranks an offloaded one, whatever the headroom: the
    // difference is a performance cliff, not a gradient.
    return fit == FitQuality::Native ? 1.0 + base : base;
}

// ── sizing ────────────────────────────────────────────────────────────────────

std::int64_t measure_model_bytes(const std::string& model_path,
                                 const std::string& models_dir,
                                 std::string* out_resolved_path) {
    const std::string resolved = resolve_model_path_for_metadata(model_path, models_dir);
    const fs::path path = resolved.empty() ? fs::path(model_path) : fs::path(resolved);
    if (out_resolved_path != nullptr) *out_resolved_path = path.string();

    std::error_code ec;
    if (!fs::exists(path, ec) || ec) return 0;

    if (!fs::is_directory(path, ec)) {
        const auto bytes = fs::file_size(path, ec);
        return ec ? 0 : static_cast<std::int64_t>(bytes);
    }

    // The bug this exists to fix: fs::file_size() sets an error_code on a
    // directory, and the caller fell through to a flat 2048 MB. Every
    // multi-shard HF checkpoint and every converted Soma container sized
    // identically — and that single scalar is what placement consumed.
    std::int64_t total = 0;
    for (fs::recursive_directory_iterator
             it(path, fs::directory_options::skip_permission_denied, ec),
         end;
         it != end && !ec;
         it.increment(ec)) {
        std::error_code fe;
        if (!it->is_regular_file(fe) || fe) continue;
        const auto sz = it->file_size(fe);
        if (!fe) total += static_cast<std::int64_t>(sz);
    }
    return total;
}

const char* to_string(FitQuality quality) {
    switch (quality) {
    case FitQuality::None:
        return "none";
    case FitQuality::Offload:
        return "offload";
    case FitQuality::Native:
        return "native";
    }
    return "unknown";
}

std::int64_t bytes_to_mb(std::int64_t bytes) {
    if (bytes <= 0) return 0;
    return std::max<std::int64_t>(1, (bytes + kMiB - 1) / kMiB);
}

} // namespace mm
