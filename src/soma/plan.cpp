// Soma — the plan document and the verdict function.
//
// One document, four consumers: `soma plan --json`, the node's pre-flight sizing,
// the scheduler's max_batch gate, and every API client via
// GET /v1/models/{id}/plan.
//
// Computing it reads HEADERS ONLY and allocates no model memory, which is what
// makes it safe to call during placement on a node that could not host the model.

#include "soma/plan.hpp"

#include "soma/arch_ir.hpp"
#include "soma/attention_backend.hpp"
#include "soma/f32_model.hpp"
#include "soma/quant_format.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>

namespace fs = std::filesystem;

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

/// Default floor on projected throughput, when the caller states none.
///
/// The reasoning it was written with still holds — a model that streams correctly
/// but at 0.2 tok/s is not usefully served, and admitting it as `stream` would
/// produce a technically-working deployment nobody can use. What it could not
/// express as a CONSTANT is that "usefully" depends on the deployer: GLM-5.2 at
/// 0.087 tok/s on a workstation is the case Colibri proved useful. It moved to
/// `HostBudget::min_tok_s`, which defaults to exactly this, so the guard is
/// unchanged for anyone who does not deliberately lower it (roadmap D21).
constexpr float kDefaultMinProjectedTokS = 1.0f;

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
    if (arch.schema_version >= kArchIrSchemaVersionV2 &&
        arch.topology.max_position_embeddings > 0 &&
        budget.ctx_size > arch.topology.max_position_embeddings) {
        return {StatusCode::InvalidArgument,
                "requested context " + std::to_string(budget.ctx_size) + " exceeds model maximum " +
                    std::to_string(arch.topology.max_position_embeddings)};
    }

    // ── one expert: gate + up + down ─────────────────────────────────────────
    //
    // At the ROUTED width, which is `d_model` for every family without a latent
    // MoE and narrower for one that has it. This read `d` directly, and was
    // correct for five families by coincidence rather than by argument — none of
    // them projected the residual stream down before routing.
    //
    // The coincidence is expensive to keep: `bytes_per_token` is
    // `n_moe_layers x top_k x expert_bytes` and the verdict is computed from it,
    // so charging an expert at 7168 wide when it is 3584 wide doubles the
    // headline number and refuses a model that streams comfortably. Wrong in the
    // pessimistic direction, which is the direction that looks responsible.
    const auto ew = arch.routed_expert_width();
    const std::uint64_t expert_bytes = bytes_for(arch, fi, ew, TensorRole::ExpertGate) +
                                       bytes_for(arch, fi, ew, TensorRole::ExpertUp) +
                                       bytes_for(arch, ew, fi, TensorRole::ExpertDown);

    out.attention_family = to_string(arch.attention.family);
    out.modality = to_string(arch.modality.modality);
    out.vision_layers = arch.modality.vision_layers;
    out.vision_hidden = arch.modality.vision_hidden;
    out.n_layers = n_layers;
    out.n_moe_layers = n_moe;
    out.n_experts = n_experts;
    out.top_k = top_k;
    out.expert_bytes = expert_bytes;
    out.active_fraction =
        n_experts > 0 ? static_cast<double>(top_k) / static_cast<double>(n_experts) : 0.0;
    out.ctx_size = budget.ctx_size;
    out.max_context = arch.topology.max_position_embeddings;
    out.kv_slots = std::max<std::uint32_t>(1, budget.kv_slots);

    out.total_routed_bytes = static_cast<std::uint64_t>(n_moe) * n_experts * expert_bytes;
    out.bytes_per_token = static_cast<std::uint64_t>(n_moe) * top_k * expert_bytes;
    out.disk_footprint_bytes = out.total_routed_bytes;
    out.speculative_available = arch.speculative.present;
    out.speculative_selected = arch.speculative.present && budget.speculative;
    if (arch.speculative.present) {
        out.speculative_method = "dspark";
        out.speculative_stages = arch.speculative.n_layers;
        out.speculative_trained_block_size = arch.speculative.trained_block_size;
        out.speculative_default_tokens = 7;
        out.speculative_routed_bytes = arch.speculative.routed_bytes;
        out.speculative_resident_bytes = arch.speculative.resident_bytes;
        out.speculative_kv_bytes_per_slot = arch.speculative.kv_bytes_per_sequence;
        out.disk_footprint_bytes += arch.speculative.routed_bytes + arch.speculative.resident_bytes;
    }

    // ── the resident half ────────────────────────────────────────────────────
    std::uint64_t dense = 0;
    dense += bytes_for(arch, arch.topology.vocab_size, d, TensorRole::Embed);
    if (!arch.topology.tie_word_embeddings) {
        dense += bytes_for(arch, arch.topology.vocab_size, d, TensorRole::Embed);
    }
    dense += static_cast<std::uint64_t>(d) * sizeof(float); // final norm, always f32
    // Asked of the BACKEND, not computed here. The formula differs by family
    // and the planner must not know how — a core-side branch on the family is
    // exactly what tools/ci/check_seam.py refuses, and it was right to.
    const auto* attn = resolve_attention_backend(arch.attention.family);
    const bool exact_attn = attn != nullptr && attn->resident_weight_bytes != nullptr;
    if (exact_attn) dense += attn->resident_weight_bytes(arch, &bytes_for);
    const std::uint64_t attn_bytes =
        (!exact_attn && attn != nullptr && attn->weight_bytes_per_layer != nullptr)
            ? attn->weight_bytes_per_layer(arch, &bytes_for)
            : 0;
    for (std::uint32_t l = 0; l < n_layers; ++l) {
        dense += attn_bytes;
        dense += 2ull * d * sizeof(float); // input + post-attn norms, always f32
        if (arch.is_moe_layer(l)) {
            dense += static_cast<std::uint64_t>(n_experts) * d * sizeof(float); // router, f32
            if (arch.ffn.routed_expert_hidden != 0) {
                // A latent MoE's two projections. Dense, read on every token,
                // and NOT small: at 7168 x 3584 each, over 92 MoE layers, they
                // are ~4.7 B parameters — comparable to the whole resident half
                // of a mid-size model. Omitting them because the experts they
                // wrap are streamed would under-count exactly the memory that
                // has to be there before any expert can be.
                // SharedExpert, matching how f32_model binds them: dense FFN tensors
                // the converter keeps at F32. Charging AttnProj here while
                // binding SharedExpert would make the plan and the load
                // disagree about the same bytes.
                dense += 2ull * bytes_for(arch, ew, d, TensorRole::SharedExpert);
                if (arch.ffn.routed_expert_norm) {
                    dense += static_cast<std::uint64_t>(ew) * sizeof(float);
                }
            }
            if (arch.router.n_shared_experts > 0) {
                // `shared_intermediate` ALREADY carries the count.
                //
                // Shared experts are fused into one set of tensors of width
                // `moe_intermediate x n_shared`, and when config.json omits
                // `shared_expert_intermediate_size` the adapter derives exactly
                // that product. Multiplying by `n_shared_experts` here charged it
                // twice — n_shared^2 — so DeepSeek-V2-Lite and Moonlight (2
                // shared) were 2x over on every MoE layer, and it was invisible
                // on GLM-5.2 and Qwen (1 shared) where squaring one is one.
                //
                // Verified against the real tensors: DeepSeek-V2-Lite's
                // shared_experts gate/up/down are each [64,64] at
                // moe_intermediate 32 and n_shared 2 — one fused set, not two.
                const auto si = arch.ffn.shared_intermediate ? arch.ffn.shared_intermediate : fi;
                dense += 2 * bytes_for(arch, si, d, TensorRole::SharedExpert) +
                         bytes_for(arch, d, si, TensorRole::SharedExpert);
            }
        } else {
            // SharedExpert, matching how the loader binds these and how the
            // converter stores them — F32 in dense.safetensors, never quantized.
            // Sizing them with the EXPERT roles charged q4 for tensors that are
            // f32 on disk, so the resident half was under-estimated for every
            // model with a dense layer. Same mis-assignment as the loader's, and
            // it had to be fixed in both or the plan and the load would disagree
            // about the same bytes.
            const auto di = arch.ffn.dense_intermediate;
            dense += 2 * bytes_for(arch, di, d, TensorRole::SharedExpert);
            dense += bytes_for(arch, d, di, TensorRole::SharedExpert);
        }
    }
    if (out.speculative_selected) dense += arch.speculative.resident_bytes;
    out.dense_resident_bytes = dense;

    // ── KV, which competes with the expert cache for the SAME RAM ────────────
    //
    // Not a footnote. At 32k context Qwen3's GQA cache wants 6.4 GB of exactly
    // the memory the expert cache wants — per slot, and `kv_slots` multiplies
    // it. Sizing the expert cache first and letting KV take the remainder
    // thrashes on long contexts and presents as an unrelated bug.
    //
    // 6.4, not the 3.2 this said: that was the fp16 figure, and the cache is
    // fp32 (D45). The ARITHMETIC below was always right — `kv_bytes_per_token`
    // has always multiplied by `sizeof(float)` — so no plan ever came out
    // wrong. What was wrong is the number a reader checks it against, which is
    // how a correct implementation gets "fixed" into a broken one.
    std::uint64_t kv_one_slot = 0;
    if (attn != nullptr && attn->kv_bytes_for_context != nullptr) {
        kv_one_slot = attn->kv_bytes_for_context(arch, budget.ctx_size);
    } else if (attn != nullptr && attn->kv_bytes_per_token != nullptr) {
        kv_one_slot = attn->kv_bytes_per_token(arch) * budget.ctx_size;
    }
    out.kv_bytes_at_ctx = kv_one_slot * std::max<std::uint32_t>(1, budget.kv_slots);
    if (out.speculative_selected) {
        out.speculative_kv_bytes_at_ctx =
            arch.speculative.kv_bytes_per_sequence * std::max<std::uint32_t>(1, budget.kv_slots);
        out.kv_bytes_at_ctx += out.speculative_kv_bytes_at_ctx;
    }

    // ── what is left for experts ─────────────────────────────────────────────
    const std::uint64_t available =
        (budget.ram_free_bytes > 0) ? budget.ram_free_bytes : budget.ram_total_bytes;
    const std::uint64_t committed = out.dense_resident_bytes + out.kv_bytes_at_ctx;
    out.expert_cache_bytes = (available > committed) ? (available - committed) : 0;
    out.pin_bytes = out.expert_cache_bytes / 8; // default: 12.5% of the cache pinned
    out.vram_hot_bytes = 0;                     // v1 is CPU-only; declared, always empty

    const auto effective_routed =
        out.total_routed_bytes + (out.speculative_selected ? out.speculative_routed_bytes : 0);
    const auto effective_moe_layers =
        n_moe + (out.speculative_selected ? out.speculative_stages : 0);
    out.footprint.ram_bytes = out.dense_resident_bytes + out.kv_bytes_at_ctx +
                              std::min(out.expert_cache_bytes, effective_routed);
    out.footprint.disk_bytes = out.disk_footprint_bytes;
    out.footprint.vram_bytes = 0;

    // ── cache-aware concurrency ──────────────────────────────────────────────
    out.cap_per_layer = (expert_bytes > 0 && effective_moe_layers > 0)
                            ? static_cast<std::uint32_t>(out.expert_cache_bytes /
                                                         (expert_bytes * effective_moe_layers))
                            : 0;

    // Expected UNIQUE experts across a batch of rows, from the coupon-collector
    // expectation: E * (1 - (1 - k/E)^rows). At rows=1 it is exactly top_k; it
    // saturates toward E as the batch grows, which is precisely why max_batch
    // cannot be a constant.
    const float p =
        (n_experts > 0) ? static_cast<float>(top_k) / static_cast<float>(n_experts) : 0.0f;
    out.expected_unique_experts_per_step = static_cast<float>(top_k);

    out.expert_set_fully_resident = (effective_routed <= out.expert_cache_bytes);

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

    // A zero or negative budget means "unstated", not "accept anything" — a
    // default-constructed HostBudget must keep the guard, or forgetting to set
    // the field would silently admit every slow model.
    const float min_tok_s = (budget.min_tok_s > 0.0f) ? budget.min_tok_s : kDefaultMinProjectedTokS;

    std::ostringstream why;
    if (arch.schema_version >= kArchIrSchemaVersionV2 && available > 0 && committed > available) {
        out.verdict = Verdict::Reject;
        why << "resident weights plus " << budget.kv_slots << " KV slot(s) at context "
            << budget.ctx_size << " require " << committed / (1024 * 1024)
            << " MiB RAM, exceeding the available " << available / (1024 * 1024) << " MiB";
    } else if (arch.schema_version >= kArchIrSchemaVersionV2 && n_moe > 0 &&
               out.expert_cache_bytes < expert_bytes) {
        out.verdict = Verdict::Reject;
        why << "resident weights plus KV leave " << out.expert_cache_bytes / (1024 * 1024)
            << " MiB for routed experts, less than one live expert ("
            << expert_bytes / (1024 * 1024) << " MiB)";
    } else if (n_moe == 0 || n_experts == 0) {
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
    } else if (out.projected_tok_s > 0.0f && out.projected_tok_s < min_tok_s) {
        out.verdict = Verdict::Reject;
        // Names the floor AND where it came from. A refusal against a default is
        // a different situation from a refusal against a figure the operator
        // chose, and telling them apart is the difference between "raise your
        // tolerance" and "this host is too small".
        why << "projected " << out.projected_tok_s << " tok/s is below the " << min_tok_s
            << " tok/s floor"
            << (budget.min_tok_s > 0.0f ? " requested for this host" : " default");
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

    // ── can this build actually RUN it? ──────────────────────────────────────
    //
    // Asked LAST, and asked of the registry rather than of a table, so it cannot
    // drift from what the engine will really resolve at load. `arch_supported`
    // had a reader in admission and no producer at all before this: it defaulted
    // to true and nothing ever set it, because an unadaptable model failed
    // earlier in adapt_hf_config and never reached a plan.
    //
    // glm_moe_dsa is the case that needed the distinction. Its expert economics
    // are perfectly computable — that is everything above this line — while
    // `resolve_f32_backend` returns nullptr for MlaDsa because serving it through
    // the MLA backend would run it as DENSE attention.
    //
    // The verdict is forced to Reject deliberately, per this field's contract: a
    // verdict is a ROUTING decision, and routing an agent to an engine that
    // cannot execute the model correctly is worse than refusing. The economics
    // above are still reported — total_routed_bytes, active_fraction,
    // bytes_per_token — so "does this model need streaming at all" is answered
    // even though "can we serve it" is no.
    // Captured BEFORE the backend check can overwrite it.
    out.economic_verdict = out.verdict;

    out.arch_supported = (resolve_f32_backend(arch) != nullptr);
    if (!out.arch_supported) {
        out.verdict = Verdict::Reject;
        // The reason distinguishes THIS reject from an economic one. They call
        // for opposite responses: economics can change on a bigger host, a
        // missing backend cannot change on any host.
        out.verdict_reason = std::string("no backend for ") + to_string(arch.attention.family) +
                             " attention in this build; the economics above are computed and "
                             "valid, but nothing can serve this model until one exists";
    }
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
      << "  \"ctx_size\": " << plan.ctx_size << ",\n"
      << "  \"max_context\": " << plan.max_context << ",\n"
      << "  \"kv_slots\": " << plan.kv_slots << ",\n"
      << "  \"kv_bytes_at_ctx\": " << plan.kv_bytes_at_ctx << ",\n"
      << "  \"expert_cache_bytes\": " << plan.expert_cache_bytes << ",\n"
      << "  \"pin_bytes\": " << plan.pin_bytes << ",\n"
      << "  \"vram_hot_bytes\": " << plan.vram_hot_bytes << ",\n"
      << "  \"total_routed_bytes\": " << plan.total_routed_bytes << ",\n"
      << "  \"speculative_available\": " << (plan.speculative_available ? "true" : "false") << ",\n"
      << "  \"speculative_selected\": " << (plan.speculative_selected ? "true" : "false") << ",\n"
      << "  \"speculative_method\": \"" << plan.speculative_method << "\",\n"
      << "  \"speculative_stages\": " << plan.speculative_stages << ",\n"
      << "  \"speculative_trained_block_size\": " << plan.speculative_trained_block_size << ",\n"
      << "  \"speculative_default_tokens\": " << plan.speculative_default_tokens << ",\n"
      << "  \"speculative_routed_bytes\": " << plan.speculative_routed_bytes << ",\n"
      << "  \"speculative_resident_bytes\": " << plan.speculative_resident_bytes << ",\n"
      << "  \"speculative_kv_bytes_per_slot\": " << plan.speculative_kv_bytes_per_slot << ",\n"
      << "  \"speculative_kv_bytes_at_ctx\": " << plan.speculative_kv_bytes_at_ctx << ",\n"
      << "  \"cap_per_layer\": " << plan.cap_per_layer << ",\n"
      << "  \"expected_unique_experts_per_step\": " << plan.expected_unique_experts_per_step
      << ",\n"
      << "  \"max_batch\": " << plan.max_batch << ",\n"
      << "  \"expert_set_fully_resident\": " << (plan.expert_set_fully_resident ? "true" : "false")
      << ",\n"
      << "  \"bytes_per_token\": " << plan.bytes_per_token
      << ",\n"
      // Topology and per-expert economics, so a consumer of the plan does not
      // have to inspect the architecture IR to learn what it is looking at. Control's
      // registry denormalizes exactly these, and the plan is its only view.
      << "  \"attention_family\": \"" << plan.attention_family << "\",\n"
      << "  \"modality\": \"" << plan.modality << "\",\n"
      << "  \"vision_layers\": " << plan.vision_layers << ",\n"
      << "  \"vision_hidden\": " << plan.vision_hidden << ",\n"
      << "  \"n_layers\": " << plan.n_layers << ",\n"
      << "  \"n_moe_layers\": " << plan.n_moe_layers << ",\n"
      << "  \"n_experts\": " << plan.n_experts << ",\n"
      << "  \"top_k\": " << plan.top_k << ",\n"
      << "  \"expert_bytes\": " << plan.expert_bytes << ",\n"
      << "  \"active_fraction\": " << plan.active_fraction << ",\n"
      << "  \"projected_tok_s\": " << plan.projected_tok_s << ",\n"
      << "  \"prefetch_enabled_layers\": " << plan.prefetch_enabled_layers << ",\n"
      << "  \"arch_supported\": " << (plan.arch_supported ? "true" : "false") << ",\n"
      << "  \"economic_verdict\": \"" << to_string(plan.economic_verdict) << "\",\n"
      << "  \"verdict\": \"" << to_string(plan.verdict) << "\",\n"
      << "  \"verdict_reason\": \"" << plan.verdict_reason << "\"\n"
      << "}\n";
    out_json = o.str();
    return {};
}

Status
resolve_arch(const std::string& model_dir, const std::string& quant_overlay_json, ArchIr& arch) {
    // Reads config.json only — no weights, no container payload. That is what
    // makes the call safe on a host that could not possibly load the model, and
    // it is the reason admission can plan before it has anywhere to run.
    //
    // A converted container carries the same config.json beside its payload, so
    // one path serves both an HF checkpoint and a container. Adding a second
    // description file that had to agree with the first is how they drift.
    const fs::path root(model_dir);
    std::ifstream in(root / "config.json", std::ios::binary);
    if (!in) {
        return {StatusCode::NotFound,
                "no config.json in " + model_dir + "; a plan needs the model's architecture"};
    }
    std::string cfg_text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    arch = ArchIr{};
    if (auto st = adapt_hf_config(cfg_text, arch); !st.ok()) return st;

    // A CONTAINER also carries what it was quantized AT, and that is part of the
    // model's identity — arch_hash covers the quant map precisely so that
    // re-admitting the same weights at a different quantization is a different
    // model with its own verdict and its own KV checkpoints.
    //
    // Without this the hash covered a quant map nobody ever populated: every
    // container of a given architecture hashed identically no matter what it was
    // converted at, the registry could not tell two quantizations apart, and a KV
    // checkpoint written under q4_g would replay under q8_0 with nothing
    // detecting it. The field was in the hash and the value never arrived.
    //
    // container_meta.json is not a second description of the architecture — it is
    // the record of a conversion, written by the converter, and it is the only
    // place the quantization exists at all.
    if (std::ifstream meta_in(root / "container_meta.json", std::ios::binary); meta_in) {
        std::string meta_text((std::istreambuf_iterator<char>(meta_in)),
                              std::istreambuf_iterator<char>());
        if (auto st = apply_container_quant(meta_text, arch); !st.ok()) return st;
    }

    // The caller's HYPOTHETICAL map, applied last so it wins over whatever the
    // model was actually converted at. Same function as the container path, so a
    // plan asked at q4_g and a container built at q4_g cannot disagree about
    // what q4_g means — including the rule that gate and up share a dtype
    // because the converter interleaves them into one range.
    //
    // The resulting plan describes a model that may not exist yet. arch_hash
    // covers the quant map, so it differs from the container's — which is the
    // guard that stops a hypothetical being mistaken for a record of what was
    // built.
    if (!quant_overlay_json.empty()) {
        if (auto st = apply_container_quant(quant_overlay_json, arch); !st.ok()) return st;
    }

    // Stamped HERE, not left empty.
    //
    // arch_hash is the model's IDENTITY: the registry keys rows on it, KV
    // checkpoints gate on it, and containers refuse to load across a mismatch.
    // A plan that omitted it left admission with nothing to record a model
    // under, so every unconverted model would have collided on the empty string.
    if (auto st = compute_arch_hash(arch, arch.arch_hash); !st.ok()) return st;
    return {};
}

Status compute_plan(const std::string& model_dir,
                    const HostBudget& budget,
                    PlanDocument& out,
                    const std::string& quant_overlay_json) {
    ArchIr arch;
    if (auto st = resolve_arch(model_dir, quant_overlay_json, arch); !st.ok()) return st;
    return compute_plan(arch, budget, out);
}

} // namespace soma
