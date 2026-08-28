// Soma — Kimi-K3 adaptation and planning.
//
// The family this pins is a hybrid: 24 of 93 layers are MLA (NoPE, output
// gated), the other 69 are gated delta-rule linear attention carrying a
// constant-size recurrent state. It is also the first family with a LATENT MoE
// — routed experts run in a 3584-wide space, not the 7168-wide residual stream
// — and the first arriving as a multimodal wrapper with the language model
// nested under `text_config`.
//
// Each of those is a way to be confidently wrong rather than loudly broken, so
// each gets an assertion that fails under the plausible misreading:
//
//   * 1-BASED layer lists. A zero-based read shifts every layer by one and the
//     stride (every 4th) is regular enough to look right.
//   * LATENT expert width. Sizing an expert at d_model doubles bytes/token —
//     exactly the number the verdict is computed from.
//   * AFFINE cache growth. 69 layers hold a fixed ~443 MiB; treating the stack
//     as uniformly MLA over-counts 1M context by ~3.8x.
//   * NoPE. The rope slice is still cached; only the rotation is gone.
//   * CONDITIONAL hashing. Describing this family must not move the arch_hash
//     of any container already admitted.

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

// Trimmed from the real moonshotai/Kimi-K3 config.json: the wrapper, the vision
// tower's identifying fields, and the whole language model under `text_config`.
// The layer lists are verbatim, because they are the thing most worth pinning.
constexpr const char* kConfig = R"json({
  "model_type":"kimi_k3",
  "architectures":["KimiK3ForConditionalGeneration"],
  "media_placeholder_token_id":163605,
  "text_config":{
    "model_type":"kimi_linear","hidden_size":7168,"vocab_size":163840,
    "num_hidden_layers":93,"num_attention_heads":96,"num_key_value_heads":96,
    "rms_norm_eps":1e-05,"max_position_embeddings":1048576,
    "eos_token_id":163586,"tie_word_embeddings":false,
    "q_lora_rank":1536,"kv_lora_rank":512,"qk_nope_head_dim":128,
    "qk_rope_head_dim":64,"v_head_dim":128,
    "mla_use_nope":true,"mla_use_output_gate":true,
    "linear_attn_config":{
      "num_heads":96,"head_dim":128,"short_conv_kernel_size":4,
      "gate_lower_bound":-5.0,"use_full_rank_gate":true,
      "full_attn_layers":[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,
        76,80,84,88,92,93],
      "kda_layers":[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29,
        30,31,33,34,35,37,38,39,41,42,43,45,46,47,49,50,51,53,54,55,57,58,59,61,
        62,63,65,66,67,69,70,71,73,74,75,77,78,79,81,82,83,85,86,87,89,90,91]
    },
    "num_experts":896,"num_experts_per_token":16,"num_shared_experts":2,
    "moe_intermediate_size":3072,"routed_expert_hidden_size":3584,
    "latent_moe_use_norm":true,"moe_renormalize":true,
    "moe_router_activation_func":"sigmoid","topk_method":"noaux_tc",
    "num_expert_group":1,"topk_group":1,"routed_scaling_factor":1.0,
    "first_k_dense_replace":1,"moe_layer_freq":1,"intermediate_size":33792,
    "hidden_act":"situ","activation_situ_beta":4.0,
    "activation_situ_linear_beta":25.0,"attn_res_block_size":12
  },
  "vision_config":{
    "vt_num_hidden_layers":27,"vt_hidden_size":1024,"patch_size":14
  }
})json";

// A pre-existing family, used to prove the new IR fields are genuinely
// conditional: adding them must not move any already-admitted arch_hash.
constexpr const char* kQwenConfig = R"json({
  "model_type":"qwen3_moe","hidden_size":2048,"vocab_size":151936,
  "num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":4,
  "head_dim":128,"num_experts":128,"num_experts_per_tok":8,
  "moe_intermediate_size":768,"intermediate_size":6144,"hidden_act":"silu",
  "norm_topk_prob":true,"rope_theta":1000000,"rms_norm_eps":1e-6
})json";

constexpr std::uint32_t kLayers = 93;
constexpr std::uint32_t kFull = 24;
constexpr std::uint32_t kLinear = 69;

} // namespace

int main() {
    soma::ArchIr arch;
    CHECK(soma::adapt_hf_config(kConfig, arch).ok());

    // ── the wrapper was descended into ───────────────────────────────────────
    // Read flat, every one of these is zero and validation reports only
    // "topology has a zero dimension".
    CHECK(arch.source_model_type == "kimi_k3");
    CHECK(arch.topology.n_layers == kLayers);
    CHECK(arch.topology.d_model == 7168);
    CHECK(arch.topology.vocab_size == 163840);
    CHECK(arch.topology.max_position_embeddings == 1048576);
    CHECK(arch.topology.eos_token_ids == std::vector<std::uint32_t>{163586});

    // ── the vision half is DECLARED, not silently dropped ────────────────────
    CHECK(arch.modality.modality == soma::Modality::VisionText);
    CHECK(arch.modality.vision_layers == 27);
    CHECK(arch.modality.vision_hidden == 1024);
    CHECK(arch.modality.vision_patch_size == 14);
    CHECK(arch.modality.media_placeholder_token_id == 163605);

    // ── the hybrid split, ONE-BASED ──────────────────────────────────────────
    CHECK(arch.attention.family == soma::AttentionFamily::MlaKda);
    const auto& kda = arch.attention.kda;
    CHECK(kda.layer_kinds.size() == kLayers);
    CHECK(kda.n_full_layers() == kFull);
    CHECK(kda.n_linear_layers() == kLinear);
    // Under a zero-based reading every one of these four flips.
    CHECK(kda.layer_kinds[0] == soma::AttnLayerKind::Linear);  // 1 in kda_layers
    CHECK(kda.layer_kinds[3] == soma::AttnLayerKind::Full);    // 4 in full_attn
    CHECK(kda.layer_kinds[4] == soma::AttnLayerKind::Linear);  // 5 in kda_layers
    // The tail is where the stride stops being a stride: 92 AND 93 are both
    // full, so the last two layers are adjacent full-attention layers.
    CHECK(kda.layer_kinds[90] == soma::AttnLayerKind::Linear); // 91
    CHECK(kda.layer_kinds[91] == soma::AttnLayerKind::Full);   // 92
    CHECK(kda.layer_kinds[92] == soma::AttnLayerKind::Full);   // 93
    CHECK(kda.n_heads == 96 && kda.head_dim == 128 && kda.conv_kernel == 4);
    CHECK(kda.full_rank_gate);
    CHECK(kda.has_gate_bound && kda.gate_lower_bound == -5.0f);

    // ── MLA half: NoPE keeps the SLICE and drops the ROTATION ────────────────
    const auto& mla = arch.attention.mla;
    CHECK(mla.nope && mla.output_gate);
    CHECK(mla.kv_lora_rank == 512 && mla.q_lora_rank == 1536);
    CHECK(mla.qk_nope_head_dim == 128 && mla.qk_rope_head_dim == 64);
    CHECK(mla.v_head_dim == 128);
    CHECK(arch.attention.head_dim == 192); // nope ++ rope, both halves
    CHECK(arch.attention.rope.theta == 0.0f);
    CHECK(arch.attention.rope.scaling.kind == soma::RopeScalingKind::None);

    // ── the router, in Kimi's spelling ───────────────────────────────────────
    CHECK(arch.router.n_experts == 896);
    CHECK(arch.router.top_k == 16);           // num_experts_per_token
    CHECK(arch.router.n_shared_experts == 2); // num_shared_experts
    CHECK(arch.router.score_fn == soma::ScoreFn::Sigmoid); // moe_router_activation_func
    CHECK(arch.router.normalize_topk);        // moe_renormalize
    CHECK(arch.router.bias_correction);       // noaux_tc
    CHECK(arch.router.n_groups == 1 && arch.router.topk_group == 1);

    // ── ffn: latent MoE, situ, block residual ────────────────────────────────
    CHECK(arch.ffn.activation == soma::Activation::Situ);
    CHECK(arch.ffn.situ_beta == 4.0f && arch.ffn.situ_linear_beta == 25.0f);
    CHECK(arch.ffn.expert_intermediate == 3072);
    CHECK(arch.ffn.routed_expert_hidden == 3584);
    CHECK(arch.ffn.routed_expert_norm);
    CHECK(arch.routed_expert_width() == 3584);
    // Shared experts read the RESIDUAL stream, not the latent space, and their
    // width already carries the count: 3072 x 2.
    CHECK(arch.ffn.shared_intermediate == 6144);
    CHECK(arch.block_residual.block_size == 12);
    CHECK(arch.block_residual.n_blocks(kLayers) == 8);

    // Layer 0 dense, 1..92 MoE.
    CHECK(arch.n_moe_layers() == 92);
    CHECK(!arch.is_moe_layer(0) && arch.is_moe_layer(1));

    // ── the plan ─────────────────────────────────────────────────────────────
    soma::HostBudget budget{};
    budget.ram_total_bytes = 512ull << 30;
    budget.ram_free_bytes = 480ull << 30;
    budget.disk_free_bytes = 4ull << 40;
    budget.disk_bandwidth = 2ull << 30;
    budget.ctx_size = 4096;
    budget.kv_slots = 1;
    soma::PlanDocument plan;
    CHECK(soma::compute_plan(arch, budget, plan).ok());
    CHECK(plan.attention_family == std::string("mla+kda"));

    // SERVABLE. This asserted the opposite for as long as the family had no
    // execution path, and the flip is gated on evidence rather than on the code
    // existing: `tests/fixtures/tiny/Kimi-Linear-Tiny` is a token-exact oracle
    // built by the real `modeling_kimi_linear.py`, and Soma matches it at
    // 2.21e-06 over 512 positions with 256 greedy tokens exact.
    //
    // The economics are unchanged by that, which was the point of computing them
    // before a backend existed: bytes per token and the routed set never
    // depended on how attention carries state.
    CHECK(plan.arch_supported);

    // ── the expert width, which the verdict turns on ─────────────────────────
    //
    // Latent: 3 x [3072, 3584]. The d_model reading is 3 x [3072, 7168] — the
    // same tensors charged at twice the width, and bytes_per_token is linear in
    // it. F32 with no group reduction at these widths, so the ratio is exact.
    const auto naive = 2ull * soma::quantized_tensor_bytes(soma::DType::F32, 3072, 7168, 128) +
                       soma::quantized_tensor_bytes(soma::DType::F32, 7168, 3072, 128);
    const auto latent = 2ull * soma::quantized_tensor_bytes(soma::DType::F32, 3072, 3584, 128) +
                        soma::quantized_tensor_bytes(soma::DType::F32, 3584, 3072, 128);
    CHECK(plan.expert_bytes == latent);
    CHECK(naive == 2 * latent);
    CHECK(plan.bytes_per_token == 92ull * 16ull * plan.expert_bytes);
    CHECK(plan.total_routed_bytes == 92ull * 896ull * plan.expert_bytes);

    // ── the cache is AFFINE, not linear ──────────────────────────────────────
    const auto* attn = soma::resolve_attention_backend(arch.attention.family);
    CHECK(attn != nullptr && attn->kv_bytes_for_context != nullptr);
    // Its own backend, not MLA's — MLA would size 93 layers of latent cache and
    // charge nothing for 69 recurrent states.
    CHECK(std::string(attn->name) == "kda");
    CHECK(attn->family == soma::AttentionFamily::MlaKda);
    // A uniform per-layer average cannot describe this stack, so the planner
    // must be taking the exact path.
    CHECK(attn->resident_weight_bytes != nullptr);
    CHECK(attn->weight_bytes_per_layer == nullptr);

    const auto at0 = attn->kv_bytes_for_context(arch, 0);
    const auto at1k = attn->kv_bytes_for_context(arch, 1024);
    const auto at2k = attn->kv_bytes_for_context(arch, 2048);
    // Constant term: 69 layers x (96*128*128 recurrent + 3*3*96*128 conv) x f32.
    const std::uint64_t state =
        std::uint64_t{kLinear} * (96ull * 128 * 128 + 3ull * 3 * 96 * 128) * sizeof(float);
    CHECK(at0 == state);
    CHECK(at0 == 464633856ull); // ~443 MiB, and it is there at zero tokens
    CHECK(at2k - at1k == at1k - at0); // affine
    // Growth is 24 layers of (512 + 64) latents, NOT 93 and NOT 512.
    CHECK(at1k - at0 == std::uint64_t{kFull} * (512 + 64) * 1024 * sizeof(float));

    const auto at1m = attn->kv_bytes_for_context(arch, 1048576);
    const std::uint64_t all_mla_reading =
        std::uint64_t{kLayers} * (512 + 64) * 1048576ull * sizeof(float);
    CHECK(at1m < all_mla_reading / 3); // ~3.8x, and it decides whether 1M fits

    // Below the crossover the constant term dominates, which is the half of
    // "affine" a linear model gets wrong in the other direction.
    CHECK(attn->kv_bytes_for_context(arch, 512) >
          std::uint64_t{kLayers} * (512 + 64) * 512ull * sizeof(float));

    // ── refusals: a config that disagrees with itself ────────────────────────
    {
        // A zero-based reading of the real list would put "93" out of range;
        // this is that error arriving from the other direction.
        std::string bad(kConfig);
        const auto at = bad.find("[4,8,12");
        CHECK(at != std::string::npos);
        bad.replace(at, 2, "[0");
        soma::ArchIr ignored;
        CHECK(!soma::adapt_hf_config(bad, ignored).ok());
    }
    {
        std::string bad(kConfig);
        const auto at = bad.find("\"linear_attn_config\"");
        CHECK(at != std::string::npos);
        bad.replace(at, 20, "\"linear_attn_unused\"");
        soma::ArchIr ignored;
        CHECK(!soma::adapt_hf_config(bad, ignored).ok());
    }

    // ── describing this family must not move an existing hash ────────────────
    soma::ArchIr qwen;
    CHECK(soma::adapt_hf_config(kQwenConfig, qwen).ok());
    std::string before;
    CHECK(soma::compute_arch_hash(qwen, before).ok());
    // Every field added for the hybrid, set to something non-default on a model
    // that is not one. All are hashed conditionally on family/presence, so none
    // may contribute — otherwise merely upgrading Soma would invalidate every
    // admitted container's arch_hash and every KV checkpoint keyed on it.
    qwen.attention.kda.n_heads = 7;
    qwen.attention.kda.head_dim = 11;
    qwen.attention.kda.layer_kinds.assign(qwen.topology.n_layers, soma::AttnLayerKind::Linear);
    qwen.attention.mla.nope = true;
    qwen.attention.mla.output_gate = true;
    qwen.block_residual.block_size = 5;
    qwen.ffn.situ_beta = 3.0f;
    qwen.ffn.situ_linear_beta = 9.0f;
    std::string after;
    CHECK(soma::compute_arch_hash(qwen, after).ok());
    CHECK(before == after);

    // …while the same fields DO move the hash of a model that has them. A
    // checkpoint written under one layer split must never replay under another.
    std::string h_kimi, h_shifted, h_beta, h_width;
    CHECK(soma::compute_arch_hash(arch, h_kimi).ok());
    {
        soma::ArchIr shifted = arch;
        shifted.attention.kda.layer_kinds[3] = soma::AttnLayerKind::Linear;
        shifted.attention.kda.layer_kinds[4] = soma::AttnLayerKind::Full;
        CHECK(soma::compute_arch_hash(shifted, h_shifted).ok());
    }
    {
        soma::ArchIr tweaked = arch;
        tweaked.ffn.situ_beta = 4.5f;
        CHECK(soma::compute_arch_hash(tweaked, h_beta).ok());
    }
    {
        soma::ArchIr tweaked = arch;
        tweaked.ffn.routed_expert_hidden = 3072;
        CHECK(soma::compute_arch_hash(tweaked, h_width).ok());
    }
    CHECK(h_kimi != h_shifted && h_kimi != h_beta && h_kimi != h_width);

    std::cout << "kimi_k3_plan: OK (" << kFull << " full / " << kLinear << " linear, 1M state "
              << at1m << " bytes, expert " << plan.expert_bytes << " B)\n";
    return 0;
}
