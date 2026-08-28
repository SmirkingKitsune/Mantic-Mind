#pragma once

// Mantic-Mind — the resource footprint of running a model, and a node's capacity
// to host one.
//
// Shared by node and control because it crosses that boundary: the node measures
// or reads it, control places against it.
//
// This replaces a single int64_t threaded through the scheduler. The change is
// SHAPE, not precision:
//
//   before  estimate_inference_vram_mb() -> one number -> nodes_with_available_vram()
//   after   ResourceFootprint{vram, ram, disk} -> nodes_with_capacity()
//
// Soma's cost is RAM + disk + optional VRAM. No amount of tuning a VRAM scalar
// expresses that, and `NodeInfo::disk_free_mb` is already collected by the health
// poll and never consulted for placement.

#include <cstdint>
#include <string>

namespace mm {

struct ResourceFootprint {
    std::int64_t vram_mb = 0;
    std::int64_t ram_mb = 0;
    std::int64_t disk_mb = 0;

    bool empty() const;
    std::int64_t total_mb() const;

    /// Legacy single-scalar view for call sites not yet converted. Returns the
    /// largest axis, which is the only honest one-number answer.
    std::int64_t dominant_mb() const;
};

/// A node's free capacity, as the health poll reports it.
struct HostCapacity {
    std::int64_t vram_free_mb = 0;
    std::int64_t vram_total_mb = 0;
    std::int64_t ram_free_mb = 0;
    std::int64_t ram_total_mb = 0;
    std::int64_t disk_free_mb = 0;
};

/// Placement headroom. Preserved from the existing constants in
/// NodeRegistry::nodes_with_available_vram, which were reasonable, plus a disk
/// margin that had nowhere to live before.
struct CapacityPolicy {
    std::int64_t vram_headroom_mb = 1024; ///< keep 1 GiB VRAM free
    std::int64_t ram_headroom_mb = 2048;  ///< keep 2 GiB system RAM free
    std::int64_t disk_headroom_mb = 8192; ///< keep 8 GiB free for KV + spill

    /// CPU-offloaded weights are slower and less reliable, so RAM counts for
    /// less than VRAM when satisfying a VRAM-shaped need.
    double ram_offload_weight = 0.60;
    std::int64_t min_gpu_for_offload_mb = 8192;
};

enum class FitQuality : std::uint8_t {
    None = 0, ///< does not fit
    Offload,  ///< fits only by counting weighted RAM against a VRAM need
    Native,   ///< fits on the axes it actually wants
};

struct NodeInfo;

/// A node's capacity, read from the fields the health poll already collects.
///
/// `disk_free_mb` in particular: it has been collected on every poll and never
/// consulted for placement, because the scalar the scheduler passed around had
/// nowhere to put it.
HostCapacity capacity_of(const NodeInfo& node);

/// Does `footprint` fit in `capacity`?
///
/// Disk is a hard constraint with no offload equivalent — there is nothing to
/// trade it against, which is exactly why a scalar could not express it.
FitQuality evaluate_fit(const ResourceFootprint& footprint,
                        const HostCapacity& capacity,
                        const CapacityPolicy& policy,
                        std::string* out_reason = nullptr);

/// Ranking score for node selection. Higher is better; 0 means no fit.
double capacity_score(const ResourceFootprint& footprint,
                      const HostCapacity& capacity,
                      const CapacityPolicy& policy);

/// Recursive directory sizing for the fallback path.
///
/// The function this replaces calls fs::file_size() on the model path, which
/// sets an error_code on a directory and falls back to a flat kFallbackModelMb =
/// 2048. Every multi-shard HF model and every converted Soma directory sizes
/// identically today. Soma does not use this at all — its footprint comes from
/// `soma plan --json`, measured rather than estimated.
std::int64_t measure_model_bytes(const std::string& model_path,
                                 const std::string& models_dir,
                                 std::string* out_resolved_path = nullptr);

const char* to_string(FitQuality quality);

/// Bytes to whole MB, rounding UP and never to zero for a non-empty model.
/// Rounding down would let a model that does not fit look like it does.
std::int64_t bytes_to_mb(std::int64_t bytes);

} // namespace mm
