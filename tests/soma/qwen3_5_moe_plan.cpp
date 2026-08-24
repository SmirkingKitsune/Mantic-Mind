// Soma — Qwen3.5-MoE adaptation and planning.
//
// The SECOND hybrid family, and every assertion here exists because the obvious
// misreading of it produces a model that runs and is wrong rather than one that
// fails to load:
//
//   * ZERO-BASED `layer_types`, against KDA's one-based lists. The two hybrids
//     state their splits in opposite conventions, and a reader who has just
//     finished the other one is primed for exactly the wrong answer. The
//     `full_attention_interval` FALLBACK is one-based, which is the trap.
//   * VALUE-head recurrent state. q/k project 16 heads and are broadcast to 128
//     before the recurrence; sizing from the key heads under-counts the
//     per-sequence state 8x, and it is the one term that does not grow with
//     context so nothing else catches it.
//   * FUSED output gate. `q_proj` is emitted at double width, so the resident
//     half is a whole extra query matrix per full layer.
//   * PARTIAL rotary. 64 of 256 channels rotate; the general `|rope=` hash term
//     covers theta and scaling only, so the width has to be hashed here.
//   * SHARED expert by WIDTH, not count. The config states no count at all, and
//     both the loader and the planner gate on the count — so the branch simply
//     did not exist before.
//   * ONE convolution over q ++ k ++ v, not three.
//   * CONDITIONAL hashing. Describing this family must not move the arch_hash of
//     any container already admitted.

#include "soma/arch_ir.hpp"
#include "soma/attention_backend.hpp"
#include "soma/plan.hpp"
#include "soma/quant_format.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool check(bool value, const char* expression, int line) {
    if (value) return true;
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    return false;
}

#define CHECK(expr)                                                                                \
    do {                                                                                           \
        if (!check((expr), #expr, __LINE__)) return 1;                                             \
    } while (false)

constexpr std::uint32_t kLayers = 92;
constexpr std::uint32_t kFull = 23;
constexpr std::uint32_t kLinear = 69;

/// The real Qwen/Qwen3.8-2.4T-A95B config.json, with `layer_types` written out
/// in full rather than abbreviated — it is the single most load-bearing field in
/// the file and an abbreviation here would be the test asserting its own guess.
std::string config_with_layer_types(bool include_layer_types) {
    std::string types;
    if (include_layer_types) {
        types = "\"layer_types\":[";
        for (std::uint32_t i = 0; i < kLayers; ++i) {
            // Verbatim from the checkpoint: `full_attention` at 0-based 3, 7, …
            types += ((i + 1) % 4 == 0) ? "\"full_attention\"" : "\"linear_attention\"";
            if (i + 1 < kLayers) types += ",";
        }
        types += "],";
    }
    return std::string(R"json({
  "model_type":"qwen3_5_moe_text",
  "architectures":["Qwen3_5MoeForCausalLM"],
  "attention_bias":false,"attn_output_gate":true,
  "bos_token_id":248044,"eos_token_id":248044,
  "full_attention_interval":4,
  "head_dim":256,"hidden_act":"silu","hidden_size":8192,
  )json") +
           types + R"json(
  "linear_conv_kernel_dim":4,
  "linear_key_head_dim":128,"linear_num_key_heads":16,
  "linear_num_value_heads":128,"linear_value_head_dim":128,
  "max_position_embeddings":262144,
  "moe_intermediate_size":2048,
  "mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":false,
  "num_attention_heads":64,"num_experts":512,"num_experts_per_tok":10,
  "num_hidden_layers":92,"num_key_value_heads":4,
  "output_gate_type":"swish","partial_rotary_factor":0.25,
  "rms_norm_eps":1e-06,
  "rope_parameters":{"partial_rotary_factor":0.25,"rope_theta":10000000,"rope_type":"default"},
  "shared_expert_intermediate_size":2048,
  "tie_word_embeddings":false,"vocab_size":248320
})json";
}

/// A pre-existing family, used to prove the new IR fields are genuinely
/// conditional: adding them must not move any already-admitted arch_hash.
constexpr const char* kQwen3Config = R"json({
  "model_type":"qwen3_moe","hidden_size":2048,"vocab_size":151936,
  "num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":4,
  "head_dim":128,"num_experts":128,"num_experts_per_tok":8,
  "moe_intermediate_size":768,"intermediate_size":6144,"hidden_act":"silu",
  "norm_topk_prob":true,"rope_theta":1000000,"rms_norm_eps":1e-6
})json";

/// An F32 sizer, matching what an unquantized plan hands a backend. Declared
/// here rather than borrowed from plan.cpp because the point is to compare two
/// `resident_weight_bytes` calls against each other, not to reproduce a plan.
std::uint64_t f32_sizer(const soma::ArchIr&,
                        std::uint32_t rows,
                        std::uint32_t cols,
                        soma::TensorRole) {
    return soma::quantized_tensor_bytes(soma::DType::F32, rows, cols, 128);
}

} // namespace

int main() {
    const auto config = config_with_layer_types(true);
    soma::ArchIr arch;
    CHECK(soma::adapt_hf_config(config, arch).ok());

    CHECK(arch.source_model_type == "qwen3_5_moe_text");
    CHECK(arch.topology.n_layers == kLayers);
    CHECK(arch.topology.d_model == 8192);
    CHECK(arch.topology.vocab_size == 248320);
    CHECK(arch.topology.max_position_embeddings == 262144);
    CHECK(arch.topology.eos_token_ids == std::vector<std::uint32_t>{248044});
    CHECK(arch.modality.modality == soma::Modality::Text);

    // ── the hybrid split, ZERO-BASED ─────────────────────────────────────────
    CHECK(arch.attention.family == soma::AttentionFamily::GqaGdn);
    const auto& gdn = arch.attention.gdn;
    CHECK(gdn.layer_kinds.size() == kLayers);
    CHECK(gdn.n_full_layers() == kFull);
    CHECK(gdn.n_linear_layers() == kLinear);
    // Under KDA's one-based reading every one of these flips.
    CHECK(gdn.layer_kinds[0] == soma::AttnLayerKind::Linear);
    CHECK(gdn.layer_kinds[2] == soma::AttnLayerKind::Linear);
    CHECK(gdn.layer_kinds[3] == soma::AttnLayerKind::Full);
    CHECK(gdn.layer_kinds[4] == soma::AttnLayerKind::Linear);
    CHECK(gdn.layer_kinds[90] == soma::AttnLayerKind::Linear);
    CHECK(gdn.layer_kinds[91] == soma::AttnLayerKind::Full);

    // The FALLBACK must agree with the stated list. `full_attention_interval` is
    // applied to a one-based index upstream, so the natural `i % 4 == 0` reading
    // shifts every layer by three — and would agree with nothing.
    {
        soma::ArchIr derived;
        CHECK(soma::adapt_hf_config(config_with_layer_types(false), derived).ok());
        CHECK(derived.attention.gdn.layer_kinds == gdn.layer_kinds);
    }

    // ── the linear half: two head counts, one convolution ────────────────────
    CHECK(gdn.n_k_heads == 16 && gdn.n_v_heads == 128);
    CHECK(gdn.head_k_dim == 128 && gdn.head_v_dim == 128);
    CHECK(gdn.conv_kernel == 4);
    CHECK(gdn.key_dim() == 2048 && gdn.value_dim() == 16384);
    // q ++ k ++ v through ONE depthwise convolution. Three separate ones — KDA's
    // shape — would be 3 x 16384 here and match nothing in the checkpoint.
    CHECK(gdn.conv_width() == 20480);
    // Indexed by the VALUE heads. The key-head reading gives 262144, an 8x
    // under-count of the term that dominates short-context memory.
    CHECK(gdn.recurrent_elems() == 2097152);

    // ── the full half: GQA, gated, partially rotated ─────────────────────────
    CHECK(arch.attention.n_heads == 64 && arch.attention.n_kv_heads == 4);
    CHECK(arch.attention.head_dim == 256);
    CHECK(arch.attention.qk_norm == soma::QkNormKind::PerHead);
    CHECK(arch.attention.fused_output_gate);
    CHECK(!arch.attention.bias);
    // 0.25 x 256. Not the full head, and not zero.
    CHECK(arch.attention.rope.partial_dim == 64);
    // The nested `rope_parameters.rope_theta` wins over any top-level default.
    CHECK(arch.attention.rope.theta == 10000000.0f);
    CHECK(arch.attention.rope.scaling.kind == soma::RopeScalingKind::None);
    CHECK(arch.attention.sliding_window == 0);

    // ── the router ───────────────────────────────────────────────────────────
    CHECK(arch.router.n_experts == 512);
    CHECK(arch.router.top_k == 10);
    CHECK(arch.router.score_fn == soma::ScoreFn::Softmax);
    // FORCED. `Qwen3_5MoeTopKRouter` divides by the top-k sum unconditionally
    // and the config carries no `norm_topk_prob`; reading the absent key as
    // false mis-scales every expert contribution.
    CHECK(arch.router.normalize_topk);
    CHECK(!arch.router.bias_correction);
    // INFERRED from the width. The config states no count, and both the loader
    // and the planner gate the shared expert on this being non-zero.
    CHECK(arch.router.n_shared_experts == 1);

    // ── ffn ──────────────────────────────────────────────────────────────────
    CHECK(arch.ffn.activation == soma::Activation::SwiGlu);
    CHECK(arch.ffn.expert_intermediate == 2048);
    CHECK(arch.ffn.shared_intermediate == 2048);
    CHECK(arch.ffn.shared_expert_gate);
    // No latent MoE: routed experts read the residual stream directly.
    CHECK(arch.ffn.routed_expert_hidden == 0);
    CHECK(arch.routed_expert_width() == 8192);
    CHECK(arch.block_residual.block_size == 0);
    CHECK(arch.hyper_connections.multiplier == 1);

    // Every layer is MoE — there is no dense prefix.
    CHECK(arch.n_moe_layers() == kLayers);
    CHECK(arch.is_moe_layer(0) && arch.is_moe_layer(kLayers - 1));

    // ── the MTP head is declared, and cannot be served ───────────────────────
    CHECK(arch.speculative.method == soma::SpeculativeMethod::Mtp);
    CHECK(arch.speculative.source_declared);
    CHECK(arch.speculative.n_layers == 1);
    CHECK(!arch.speculative.present);

    // ── the plan ─────────────────────────────────────────────────────────────
    soma::HostBudget budget{};
    budget.ram_total_bytes = 512ull << 30;
    budget.ram_free_bytes = 480ull << 30;
    budget.disk_free_bytes = 8ull << 40;
    budget.disk_bandwidth = 2ull << 30;
    budget.ctx_size = 4096;
    budget.kv_slots = 1;
    soma::PlanDocument plan;
    CHECK(soma::compute_plan(arch, budget, plan).ok());
    CHECK(plan.attention_family == std::string("gqa+gdn"));

    // SERVABLE, and this asserted the OPPOSITE until there was evidence — the
    // same sequence DSA and the MlaKda hybrid each went through.
    //
    // `tests/fixtures/tiny/Qwen3.5-MoE-Tiny` is the evidence: an oracle built by
    // transformers' own `modeling_qwen3_5_moe.py`, which Soma matches at
    // 4.58e-06 over 512 teacher-forced positions with 256 greedy tokens exact.
    //
    // The economics below are unchanged by that, which was the point of
    // computing them before a backend existed: bytes per token and the routed
    // set never depended on how attention carries state.
    CHECK(plan.arch_supported);

    // ── expert economics ─────────────────────────────────────────────────────
    // No latent width here, so an expert is the plain 3 x [d_model, 2048].
    const auto expert = 2ull * soma::quantized_tensor_bytes(soma::DType::F32, 2048, 8192, 128) +
                        soma::quantized_tensor_bytes(soma::DType::F32, 8192, 2048, 128);
    CHECK(plan.expert_bytes == expert);
    CHECK(plan.bytes_per_token == std::uint64_t{kLayers} * 10ull * plan.expert_bytes);
    CHECK(plan.total_routed_bytes == std::uint64_t{kLayers} * 512ull * plan.expert_bytes);

    // ── the cache is AFFINE, not linear ──────────────────────────────────────
    const auto* attn = soma::resolve_attention_backend(arch.attention.family);
    CHECK(attn != nullptr);
    // Its OWN backend. GQA's would charge 92 layers of K/V and nothing for the
    // 69 recurrent states.
    CHECK(std::string(attn->name) == "gdn");
    CHECK(attn->family == soma::AttentionFamily::GqaGdn);
    CHECK(attn->kv_bytes_for_context != nullptr);
    // A uniform per-layer average cannot describe this stack, so the planner
    // must be taking the exact path.
    CHECK(attn->resident_weight_bytes != nullptr);
    CHECK(attn->weight_bytes_per_layer == nullptr);

    const auto at0 = attn->kv_bytes_for_context(arch, 0);
    const auto at1k = attn->kv_bytes_for_context(arch, 1024);
    const auto at2k = attn->kv_bytes_for_context(arch, 2048);

    // Constant term: 69 x (128*128*128 recurrent + 20480*3 conv) x f32.
    const std::uint64_t state =
        std::uint64_t{kLinear} * (2097152ull + 20480ull * 3) * sizeof(float);
    CHECK(at0 == state);
    CHECK(at0 == 595771392ull); // 568.2 MiB, present at zero tokens

    CHECK(at2k - at1k == at1k - at0); // affine
    // Growth is 23 layers of TWO planes of 4 kv-heads x 256, not 92 and not 64
    // query heads.
    CHECK(at1k - at0 == std::uint64_t{kFull} * 2ull * (4 * 256) * 1024 * sizeof(float));
    CHECK(attn->kv_bytes_per_token(arch) == std::uint64_t{kFull} * 2ull * (4 * 256) * sizeof(float));
    CHECK(attn->kv_bytes_per_token(arch) == 188416ull);

    // At the stated context the split is what decides whether the model fits.
    const auto at_max = attn->kv_bytes_for_context(arch, 262144);
    const std::uint64_t all_full_reading =
        std::uint64_t{kLayers} * 2ull * (4 * 256) * 262144ull * sizeof(float);
    CHECK(at_max == 49987895296ull);           // 46.55 GiB
    CHECK(all_full_reading == 197568495616ull); // 184.0 GiB
    CHECK(all_full_reading > at_max * 3);       // 3.95x

    // Below the crossover the constant term dominates — the half of "affine"
    // that a linear model gets wrong in the OTHER direction.
    CHECK(attn->kv_bytes_for_context(arch, 512) >
          std::uint64_t{kLayers} * 2ull * (4 * 256) * 512ull * sizeof(float));
    // …and exactly at 1054 tokens the two readings cross.
    CHECK(state == std::uint64_t{kLinear} * 2ull * (4 * 256) * 1054ull * sizeof(float));

    // ── the fused gate is charged, not merely noted ──────────────────────────
    {
        soma::ArchIr ungated = arch;
        ungated.attention.fused_output_gate = false;
        const auto with = attn->resident_weight_bytes(arch, &f32_sizer);
        const auto without = attn->resident_weight_bytes(ungated, &f32_sizer);
        // Exactly one extra [n_heads*head_dim, d_model] matrix per FULL layer —
        // and none on the 69 linear ones, which have no q_proj at all.
        CHECK(with - without ==
              std::uint64_t{kFull} *
                  soma::quantized_tensor_bytes(soma::DType::F32, 64 * 256, 8192, 128));
    }

    // ── refusals: configs that disagree with themselves ──────────────────────
    {
        // 128 value heads over 12 key heads cannot repeat_interleave.
        std::string bad(config);
        const auto at = bad.find("\"linear_num_key_heads\":16");
        CHECK(at != std::string::npos);
        bad.replace(at, 25, "\"linear_num_key_heads\":12");
        soma::ArchIr ignored;
        CHECK(!soma::adapt_hf_config(bad, ignored).ok());
    }
    {
        // A layer kind this engine does not implement must be refused, not
        // quietly rounded to Full — that would size a full cache for a windowed
        // layer.
        std::string bad(config);
        const auto at = bad.find("\"full_attention\"");
        CHECK(at != std::string::npos);
        bad.replace(at, 16, "\"sliding_attn\"  ");
        soma::ArchIr ignored;
        CHECK(!soma::adapt_hf_config(bad, ignored).ok());
    }
    {
        // A layer_types list of the wrong length is a config disagreeing with
        // its own num_hidden_layers.
        std::string bad(config);
        const auto at = bad.find("\"num_hidden_layers\":92");
        CHECK(at != std::string::npos);
        bad.replace(at, 22, "\"num_hidden_layers\":91");
        soma::ArchIr ignored;
        CHECK(!soma::adapt_hf_config(bad, ignored).ok());
    }

    // ── describing this family must not move an existing hash ────────────────
    soma::ArchIr other;
    CHECK(soma::adapt_hf_config(kQwen3Config, other).ok());
    std::string before;
    CHECK(soma::compute_arch_hash(other, before).ok());
    // Every field added for this family, set to something non-default on a model
    // that is not one. All are hashed conditionally, so none may contribute.
    other.attention.gdn.n_k_heads = 7;
    other.attention.gdn.n_v_heads = 21;
    other.attention.gdn.head_k_dim = 11;
    other.attention.gdn.head_v_dim = 13;
    other.attention.gdn.conv_kernel = 5;
    other.attention.gdn.layer_kinds.assign(other.topology.n_layers, soma::AttnLayerKind::Linear);
    other.attention.fused_output_gate = true;
    other.ffn.shared_expert_gate = true;
    std::string after;
    CHECK(soma::compute_arch_hash(other, after).ok());
    CHECK(before == after);

    // …while the same fields DO move the hash of a model that has them.
    std::string h_base, h_split, h_rot, h_gate, h_heads;
    CHECK(soma::compute_arch_hash(arch, h_base).ok());
    {
        soma::ArchIr t = arch;
        t.attention.gdn.layer_kinds[3] = soma::AttnLayerKind::Linear;
        t.attention.gdn.layer_kinds[4] = soma::AttnLayerKind::Full;
        CHECK(soma::compute_arch_hash(t, h_split).ok());
    }
    {
        // The general `|rope=` term hashes theta and scaling only, so without
        // the family's own contribution these two would collide.
        soma::ArchIr t = arch;
        t.attention.rope.partial_dim = 256;
        CHECK(soma::compute_arch_hash(t, h_rot).ok());
    }
    {
        soma::ArchIr t = arch;
        t.attention.fused_output_gate = false;
        CHECK(soma::compute_arch_hash(t, h_gate).ok());
    }
    {
        soma::ArchIr t = arch;
        t.attention.gdn.n_v_heads = 64;
        CHECK(soma::compute_arch_hash(t, h_heads).ok());
    }
    CHECK(h_base != h_split && h_base != h_rot && h_base != h_gate && h_base != h_heads);

    std::cout << "qwen3_5_moe_plan: OK (" << kFull << " full / " << kLinear << " linear, state "
              << at0 << " B, 262144 ctx " << at_max << " B, expert " << plan.expert_bytes
              << " B)\n";
    return 0;
}
