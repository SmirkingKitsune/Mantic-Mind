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

    /// Slowest generation the DEPLOYER will accept, tok/s. Below it the verdict
    /// is Reject.
    ///
    /// A host property, not a model one, and that is the whole argument for it
    /// living here. This was `kMinProjectedTokS`, a constant in plan.cpp, and its
    /// reasoning was sound as written — "a model that streams correctly but at
    /// 0.2 tok/s is not usefully served". What a global constant cannot express
    /// is that the answer depends on who is asking: GLM-5.2 projects 0.087 tok/s
    /// on a 24 GiB workstation, and Colibri served those same 744B weights on
    /// 16-24 GB to someone who found the result useful. For a model of that size
    /// on that hardware, 0.1 tok/s may be the entire point.
    ///
    /// The verdict is already documented as a property of (model, quantization,
    /// host budget). "How slow is too slow" belongs in the third of those, beside
    /// `ram_total_bytes` and `disk_bandwidth`, rather than being decided once for
    /// every deployment that will ever exist.
    ///
    /// 0 means UNSTATED, and resolves to the engine's default of 1.0 — not to
    /// "no floor". A default-constructed budget therefore guards exactly as it
    /// did before this field existed, so nothing admits that did not admit
    /// before; lowering the bar takes a deliberate statement by whoever runs the
    /// host (roadmap D21).
    ///
    /// The sentinel earns its keep in the refusal message, which says whether the
    /// figure was chosen or inherited. "Too slow against a number you picked" and
    /// "too slow against our default" call for different responses.
    float min_tok_s = 0.0f;
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

    /// Topology and per-expert economics, carried so a CONSUMER of the plan does
    /// not need to inspect the architecture IR to learn what it is looking at. Control's
    /// registry denormalizes exactly these for its queries and its TUI, and the
    /// plan is the only view of the model it has.
    std::string attention_family; ///< mha | gqa | mla | mla+dsa
    std::uint32_t n_layers = 0;
    std::uint32_t n_moe_layers = 0;
    std::uint32_t n_experts = 0;
    std::uint32_t top_k = 0;
    std::uint64_t expert_bytes = 0; ///< one expert: gate + up + down
    /// Fraction of the routed set that fires per token — top_k / n_experts. THE
    /// number the verdict turns on: Mixtral moves 5x the bytes of Qwen3 per token
    /// while being 1.35x larger, because it fires 25% of its experts and Qwen3
    /// fires 6.25%.
    double active_fraction = 0.0;

    /// Is there a backend for this attention family at all?
    ///
    /// Distinct from the verdict, which it also forces to Reject. A model
    /// rejected on ECONOMICS may still be worth converting — the verdict is a
    /// property of (model, quantization, host), so a beefier node can reach a
    /// different one from the same container. A model with no backend cannot be
    /// run by any host, ever, so converting it is pure waste. Admission reads
    /// this to decide whether to spend the hours.
    bool arch_supported = true;

    /// Effective for THIS host — not necessarily the registry's stored value.
    Verdict verdict = Verdict::Reject;
    std::string verdict_reason;

    /// What the ECONOMICS alone say, before `arch_supported` is consulted.
    ///
    /// Equal to `verdict` whenever a backend exists, which is every model but
    /// one today. It differs only when the engine cannot run the family at all,
    /// and then it answers the question the forced Reject would otherwise hide:
    /// *would* this model stream, if something could serve it?
    ///
    /// That is not idle curiosity — it is the whole reason to plan an
    /// unsupported architecture. GLM-5.2 is the case: `verdict` is reject
    /// because nothing implements DSA, while `economic_verdict` says whether
    /// implementing it would buy a streamable model or a resident one, which is
    /// what decides whether the backend is worth writing.
    Verdict economic_verdict = Verdict::Reject;

    /// Layers whose measured router-lookahead recall clears the threshold.
    /// Prefetch is enabled per layer; a wrong prefetch evicts something useful,
    /// so a poor-recall layer gets none.
    std::uint32_t prefetch_enabled_layers = 0;
};

/// Reads the container metadata and adapts its copied config.json. Allocates no model memory, loads
/// no weights, and is safe to call on a node that could not host the model. `quant_overlay_json`
/// states a HYPOTHETICAL quantization, in the same shape a container's `container_meta.json` uses —
/// `{"dtype_gate_up", "dtype_down", "group"}` — and is applied AFTER the container's own map, so it
/// wins.
///
/// It exists because the verdict is a property of (model, quantization, host)
/// and, without it, the only quantization askable was the one the model had
/// already been converted at. That made "should I convert this at q4?" a
/// question you could only answer by converting it at q4 — which for a 744B
/// model is hours and hundreds of gigabytes, and inverts the entire reason this
/// function reads headers only.
///
/// Empty means "whatever the model already says", which is every existing
/// caller's behaviour unchanged.
Status compute_plan(const std::string& model_dir,
                    const HostBudget& budget,
                    PlanDocument& out,
                    const std::string& quant_overlay_json = {});

/// Resolve what a model in `model_dir` IS: adapt config.json, apply the
/// container's own recorded quantization, then apply the caller's overlay, then
/// stamp arch_hash.
///
/// Extracted so `plan` and `serve` share it rather than each deciding what a model
/// is. main.cpp says the two "must not be able to disagree", and they did: serve
/// hardcoded `q4_g/q4_g/q6_g @128` and never read container_meta.json at all, so
/// it could only load containers that happened to match that guess — a container
/// converted at any other group was refused with an expert-size mismatch that
/// looked like a corrupt container — and it had no way to express a quantized
/// dense half, which is the one thing `plan --quant-dense` had been describing all
/// along (roadmap D41).
///
/// `quant_overlay_json` is in container_meta.json's shape and goes through the
/// same applier, so an overlay cannot mean something a container could not be.
/// Empty means "whatever the model already says".
Status
resolve_arch(const std::string& model_dir, const std::string& quant_overlay_json, ArchIr& out);

/// Same, when the IR is already parsed.
Status compute_plan(const ArchIr& arch, const HostBudget& budget, PlanDocument& out);

/// The stable wire form. This exact JSON is what `soma plan --json` prints and
/// what GET /v1/models/{id}/plan returns — one serializer, so the CLI and the
/// API can never disagree about a footprint.
Status serialize_plan(const PlanDocument& plan, std::string& out_json);

} // namespace soma
