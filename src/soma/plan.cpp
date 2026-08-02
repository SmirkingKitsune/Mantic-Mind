// Soma — the plan document and the verdict function.
//
// One document, four consumers: `soma plan --json`, the node's pre-flight sizing,
// the scheduler's max_batch gate, and every API client via
// GET /v1/models/{id}/plan.
//
// Computing it reads HEADERS ONLY and allocates no model memory, which is what
// makes it safe to call during placement on a node that could not host the model.

#include "soma/plan.hpp"

#include "soma/attention_backend.hpp"
#include "soma/quant_format.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace soma {

namespace {

/// Active fraction above which streaming cannot win at any expert size.
///
/// Mixtral is 2-of-8 = 25%: a quarter of every layer fires per token, so
/// streaming reads a quarter of the entire routed set every token no matter how
/// the cache is tuned. Qwen3 is 8-of-128 = 6.3% and DeepSeek-V2-Lite 6-of-64 =
/// 9.4%. The boundary sits between them; 15% is the midpoint of a wide gap
/// rather than a tuned constant.
constexpr float kMaxStreamableActiveFraction = 0.15f;

/// Below this, per-read overhead dominates and streaming is bounded by IOPS
/// rather than bandwidth. Coarse-grained MoE never gets here; the check exists
/// for the opposite extreme (very many very small experts).
constexpr std::uint64_t kMinStreamableExpertBytes = 64ull * 1024;

/// Floor on projected throughput. A model that streams correctly but at 0.2
/// tok/s is not usefully served, and admitting it as `stream` would produce a
/// technically-working deployment nobody can use.
constexpr float kMinProjectedTokS = 1.0f;

std::uint64_t
bytes_for(const ArchIr& arch, std::uint32_t rows, std::uint32_t cols, TensorRole role) {
    const auto& spec = arch.quantization.for_role(role);
    // Row-aware: the effective group is the largest divisor of `cols` not
    // exceeding the requested one, matching quantize_tensor(). A flat
    // element-count calculation disagrees for any tensor narrower than the group.
    return quantized_tensor_bytes(spec.dtype, rows, cols, spec.group ? spec.group : kDefaultGroup);
}

} // namespace

Status compute_plan(const ArchIr& arch, const HostBudget& budget, PlanDocument& out) {
    out = PlanDocument{};
    out.arch_hash = arch.arch_hash;
    out.model_name = arch.source_repo;
    out.schema_version = arch.schema_version;

    const auto d = arch.topology.d_model;
    const auto n_layers = arch.topology.n_layers;
    const auto n_moe = arch.n_moe_layers();
    const auto n_experts = arch.router.n_experts;
    const auto top_k = arch.router.top_k;
    const auto fi = arch.ffn.expert_intermediate;

    if (d == 0 || n_layers == 0) {
        return {StatusCode::InvalidArgument, "plan: topology has a zero dimension"};
    }

    // ── one expert: gate + up + down ─────────────────────────────────────────
    const std::uint64_t expert_bytes = bytes_for(arch, fi, d, TensorRole::ExpertGate) +
                                       bytes_for(arch, fi, d, TensorRole::ExpertUp) +
                                       bytes_for(arch, d, fi, TensorRole::ExpertDown);

    out.total_routed_bytes = static_cast<std::uint64_t>(n_moe) * n_experts * expert_bytes;
    out.bytes_per_token = static_cast<std::uint64_t>(n_moe) * top_k * expert_bytes;
    out.disk_footprint_bytes = out.total_routed_bytes;

    // ── the resident half ────────────────────────────────────────────────────
    const auto hq = arch.attention.n_heads * arch.attention.head_dim;
    const auto hkv = arch.attention.n_kv_heads * arch.attention.head_dim;

    std::uint64_t dense = 0;
    dense += bytes_for(arch, arch.topology.vocab_size, d, TensorRole::Embed);
    if (!arch.topology.tie_word_embeddings) {
        dense += bytes_for(arch, arch.topology.vocab_size, d, TensorRole::Embed);
    }
    for (std::uint32_t l = 0; l < n_layers; ++l) {
        dense += bytes_for(arch, hq, d, TensorRole::AttnProj);
        dense += 2 * bytes_for(arch, hkv, d, TensorRole::AttnProj);
        dense += bytes_for(arch, d, hq, TensorRole::AttnProj);
        dense += 2ull * d * sizeof(float); // input + post-attn norms, always f32
        if (arch.is_moe_layer(l)) {
            dense += static_cast<std::uint64_t>(n_experts) * d * sizeof(float); // router, f32
            if (arch.router.n_shared_experts > 0) {
                const auto si = arch.ffn.shared_intermediate ? arch.ffn.shared_intermediate : fi;
                dense += arch.router.n_shared_experts *
                         (2 * bytes_for(arch, si, d, TensorRole::SharedExpert) +
                          bytes_for(arch, d, si, TensorRole::SharedExpert));
            }
        } else {
            const auto di = arch.ffn.dense_intermediate;
            dense += 2 * bytes_for(arch, di, d, TensorRole::ExpertGate);
            dense += bytes_for(arch, d, di, TensorRole::ExpertDown);
        }
    }
    out.dense_resident_bytes = dense;

    // ── KV, which competes with the expert cache for the SAME RAM ────────────
    //
    // Not a footnote. At 32k context Qwen3's GQA cache wants 3.2 GB of exactly
    // the memory the expert cache wants. Sizing the expert cache first and
    // letting KV take the remainder thrashes on long contexts and presents as an
    // unrelated bug.
    const auto* attn = resolve_attention_backend(arch.attention.family);
    std::uint64_t kv_per_token = 0;
    if (attn != nullptr && attn->kv_bytes_per_token != nullptr) {
        kv_per_token = attn->kv_bytes_per_token(arch);
    }
    out.kv_bytes_at_ctx =
        kv_per_token * budget.ctx_size * std::max<std::uint32_t>(1, budget.kv_slots);

    // ── what is left for experts ─────────────────────────────────────────────
    const std::uint64_t available =
        (budget.ram_free_bytes > 0) ? budget.ram_free_bytes : budget.ram_total_bytes;
    const std::uint64_t committed = out.dense_resident_bytes + out.kv_bytes_at_ctx;
    out.expert_cache_bytes = (available > committed) ? (available - committed) : 0;
    out.pin_bytes = out.expert_cache_bytes / 8; // default: 12.5% of the cache pinned
    out.vram_hot_bytes = 0;                     // v1 is CPU-only; declared, always empty

    out.footprint.ram_bytes = out.dense_resident_bytes + out.kv_bytes_at_ctx +
                              std::min(out.expert_cache_bytes, out.total_routed_bytes);
    out.footprint.disk_bytes = out.disk_footprint_bytes;
    out.footprint.vram_bytes = 0;

    // ── cache-aware concurrency ──────────────────────────────────────────────
    out.cap_per_layer =
        (expert_bytes > 0 && n_moe > 0)
            ? static_cast<std::uint32_t>(out.expert_cache_bytes / (expert_bytes * n_moe))
            : 0;

    // Expected UNIQUE experts across a batch of rows, from the coupon-collector
    // expectation: E * (1 - (1 - k/E)^rows). At rows=1 it is exactly top_k; it
    // saturates toward E as the batch grows, which is precisely why max_batch
    // cannot be a constant.
    const float p =
        (n_experts > 0) ? static_cast<float>(top_k) / static_cast<float>(n_experts) : 0.0f;
    out.expected_unique_experts_per_step = static_cast<float>(top_k);

    out.expert_set_fully_resident = (out.total_routed_bytes <= out.expert_cache_bytes);

    if (out.expert_set_fully_resident) {
        // Nothing to amortize and nothing to thrash: concurrency is bounded by
        // compute, not by the cache.
        out.max_batch = std::max<std::uint32_t>(1, budget.kv_slots);
    } else if (out.cap_per_layer > 0 && top_k > 0) {
        // Grow the batch while the expected unique set still fits the per-layer
        // cap. N sequences x top_k against a small cap thrashes: every step
        // evicts what the next needs, the union degenerates into per-row reads
        // plus eviction overhead, and throughput falls BELOW single-sequence.
        std::uint32_t best = 1;
        for (std::uint32_t rows = 1; rows <= 64; ++rows) {
            const float uniq = static_cast<float>(n_experts) *
                               (1.0f - std::pow(1.0f - p, static_cast<float>(rows)));
            if (uniq > static_cast<float>(out.cap_per_layer)) break;
            best = rows;
            out.expected_unique_experts_per_step = uniq;
        }
        out.max_batch = best;
    } else {
        out.max_batch = 1;
    }

    // ── projected throughput ─────────────────────────────────────────────────
    const double hit_rate = (out.total_routed_bytes > 0)
                                ? std::min(1.0,
                                           static_cast<double>(out.expert_cache_bytes) /
                                               static_cast<double>(out.total_routed_bytes))
                                : 1.0;
    const double miss_bytes = static_cast<double>(out.bytes_per_token) * (1.0 - hit_rate);
    if (out.expert_set_fully_resident || miss_bytes <= 0.0) {
        out.projected_tok_s = 0.0f; // not disk-bound; compute decides, not this plan
    } else if (budget.disk_bandwidth > 0) {
        out.projected_tok_s =
            static_cast<float>(static_cast<double>(budget.disk_bandwidth) / miss_bytes);
    }

    out.prefetch_enabled_layers = 0; // set from pilot_profile once measured

    // ── the verdict ──────────────────────────────────────────────────────────
    const float active_fraction =
        (n_experts > 0) ? static_cast<float>(top_k) / static_cast<float>(n_experts) : 1.0f;

    std::ostringstream why;
    if (n_moe == 0 || n_experts == 0) {
        out.verdict = Verdict::ResidentOnly;
        why << "no routed experts; nothing to stream";
    } else if (out.expert_set_fully_resident) {
        out.verdict = Verdict::ResidentOnly;
        why << "routed set (" << out.total_routed_bytes / (1024 * 1024)
            << " MiB) fits the expert cache (" << out.expert_cache_bytes / (1024 * 1024)
            << " MiB); streaming has nothing to do";
    } else if (active_fraction > kMaxStreamableActiveFraction) {
        // Coarse-grained. It does not fit and streaming cannot win, so the only
        // question left is whether the fallback can hold it at all.
        out.verdict = Verdict::Reject;
        why << "active fraction " << static_cast<int>(active_fraction * 100) << "% exceeds "
            << static_cast<int>(kMaxStreamableActiveFraction * 100)
            << "%: a quarter-scale slice of every layer fires per token, so streaming reads "
            << out.bytes_per_token / (1024 * 1024) << " MiB/token and buys nothing";
    } else if (expert_bytes < kMinStreamableExpertBytes) {
        out.verdict = Verdict::Reject;
        why << "expert size " << expert_bytes << " B is below the " << kMinStreamableExpertBytes
            << " B floor; per-read overhead would dominate";
    } else if (out.projected_tok_s > 0.0f && out.projected_tok_s < kMinProjectedTokS) {
        out.verdict = Verdict::Reject;
        why << "projected " << out.projected_tok_s << " tok/s is below the " << kMinProjectedTokS
            << " floor";
    } else if (hit_rate > 0.5) {
        out.verdict = Verdict::Hybrid;
        why << "active fraction " << active_fraction * 100.0f << "% with " << expert_bytes / 1024
            << " KiB experts; " << static_cast<int>(hit_rate * 100)
            << "% of the routed set is resident";
    } else {
        out.verdict = Verdict::Stream;
        why << "active fraction " << active_fraction * 100.0f << "% with " << expert_bytes / 1024
            << " KiB experts; routed set (" << out.total_routed_bytes / (1024 * 1024)
            << " MiB) exceeds the cache (" << out.expert_cache_bytes / (1024 * 1024) << " MiB)";
    }
    out.verdict_reason = why.str();
    return {};
}

Status serialize_plan(const PlanDocument& plan, std::string& out_json) {
    std::ostringstream o;
    o << "{\n"
      << "  \"arch_hash\": \"" << plan.arch_hash << "\",\n"
      << "  \"model_name\": \"" << plan.model_name << "\",\n"
      << "  \"schema_version\": " << plan.schema_version << ",\n"
      << "  \"footprint\": {\"vram_mb\": " << plan.footprint.vram_bytes / (1024 * 1024)
      << ", \"ram_mb\": " << plan.footprint.ram_bytes / (1024 * 1024)
      << ", \"disk_mb\": " << plan.footprint.disk_bytes / (1024 * 1024) << "},\n"
      << "  \"dense_resident_bytes\": " << plan.dense_resident_bytes << ",\n"
      << "  \"kv_bytes_at_ctx\": " << plan.kv_bytes_at_ctx << ",\n"
      << "  \"expert_cache_bytes\": " << plan.expert_cache_bytes << ",\n"
      << "  \"pin_bytes\": " << plan.pin_bytes << ",\n"
      << "  \"vram_hot_bytes\": " << plan.vram_hot_bytes << ",\n"
      << "  \"total_routed_bytes\": " << plan.total_routed_bytes << ",\n"
      << "  \"cap_per_layer\": " << plan.cap_per_layer << ",\n"
      << "  \"expected_unique_experts_per_step\": " << plan.expected_unique_experts_per_step
      << ",\n"
      << "  \"max_batch\": " << plan.max_batch << ",\n"
      << "  \"expert_set_fully_resident\": " << (plan.expert_set_fully_resident ? "true" : "false")
      << ",\n"
      << "  \"bytes_per_token\": " << plan.bytes_per_token << ",\n"
      << "  \"projected_tok_s\": " << plan.projected_tok_s << ",\n"
      << "  \"prefetch_enabled_layers\": " << plan.prefetch_enabled_layers << ",\n"
      << "  \"verdict\": \"" << to_string(plan.verdict) << "\",\n"
      << "  \"verdict_reason\": \"" << plan.verdict_reason << "\"\n"
      << "}\n";
    out_json = o.str();
    return {};
}

Status compute_plan(const std::string& model_dir, const HostBudget& budget, PlanDocument& out) {
    // Reads arch.json only — no weights, no container payload. This is what makes
    // the call safe on a node that could not host the model.
    (void)model_dir;
    (void)budget;
    (void)out;
    return {StatusCode::Unsupported,
            "directory-based plan lands with the admission converter's arch.json output; "
            "use compute_plan(ArchIr, ...) with an adapted config for now"};
}

} // namespace soma
