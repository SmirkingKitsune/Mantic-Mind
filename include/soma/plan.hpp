#pragma once

// Soma — the plan document.
//
// ONE document, consumed by four callers:
//   `soma plan --json`   operator inspection
//   the node             pre-flight sizing before spawning an engine
//   the scheduler        max_batch gate, cache caps, prefetch policy
//   every API client     GET /v1/models/{id}/plan, and placement
//
// Computing it READS HEADERS ONLY and allocates nothing. That constraint is what
// makes it safe for the scheduler to call during placement, on a node that may
// not have the RAM to load the model at all.

#include "soma/arch_ir.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <string>

namespace soma {

/// The host side of the verdict. The verdict is a property of
/// (model, quantization, host budget) — see schemas/arch-ir.md §8 — so it cannot
/// be computed without this.
struct HostBudget {
    std::uint64_t ram_total_bytes = 0;
    std::uint64_t ram_free_bytes = 0;
    std::uint64_t vram_total_bytes = 0; ///< v1: unused
    std::uint64_t disk_free_bytes = 0;
    std::uint64_t disk_bandwidth = 0; ///< bytes/sec, measured at expert size
    std::uint32_t ctx_size = 4096;
    std::uint32_t kv_slots = 4;
};

/// The footprint, in the shape placement actually needs.
///
/// Soma's cost is RAM + disk + optional VRAM. The existing scheduler estimates a
/// single VRAM scalar from file size; that is a different SHAPE, not a different
/// number, which is why placement is re-worked rather than re-tuned.
struct ResourceFootprint {
    std::uint64_t vram_bytes = 0;
    std::uint64_t ram_bytes = 0;
    std::uint64_t disk_bytes = 0;
};

struct PlanDocument {
    std::string arch_hash;
    std::string model_name;
    std::uint32_t schema_version = kArchIrSchemaVersion;

    ResourceFootprint footprint{};

    std::uint64_t dense_resident_bytes = 0;
    std::uint64_t kv_bytes_at_ctx = 0;    ///< ctx_size × kv_slots
    std::uint64_t expert_cache_bytes = 0; ///< the safe cap after KV is subtracted
    std::uint64_t pin_bytes = 0;
    std::uint64_t vram_hot_bytes = 0; ///< v1: always 0
    std::uint64_t total_routed_bytes = 0;
    std::uint64_t disk_footprint_bytes = 0;

    /// Cache-aware concurrency, derived not configured:
    ///     cap_per_layer / expected_unique_experts_per_step
    std::uint32_t cap_per_layer = 0;
    float expected_unique_experts_per_step = 0.0f;
    std::uint32_t max_batch = 0;
    bool expert_set_fully_resident = false;

    std::uint64_t bytes_per_token = 0;
    float projected_tok_s = 0.0f;

    /// Effective for THIS host — not necessarily the registry's stored value.
    Verdict verdict = Verdict::Reject;
    std::string verdict_reason;

    /// Layers whose measured router-lookahead recall clears the threshold.
    /// Prefetch is enabled per layer; a wrong prefetch evicts something useful,
    /// so a poor-recall layer gets none.
    std::uint32_t prefetch_enabled_layers = 0;
};

/// Reads the container header and arch.json. Allocates no model memory, loads no
/// weights, and is safe to call on a node that could not host the model.
Status compute_plan(const std::string& model_dir, const HostBudget& budget, PlanDocument& out);

/// Same, when the IR is already parsed.
Status compute_plan(const ArchIr& arch, const HostBudget& budget, PlanDocument& out);

/// The stable wire form. This exact JSON is what `soma plan --json` prints and
/// what GET /v1/models/{id}/plan returns — one serializer, so the CLI and the
/// API can never disagree about a footprint.
Status serialize_plan(const PlanDocument& plan, std::string& out_json);

} // namespace soma
