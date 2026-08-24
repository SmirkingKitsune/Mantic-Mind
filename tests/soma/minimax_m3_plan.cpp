// Soma — MiniMax-M3 adaptation and planning.
//
// The block-sparse GQA family, and — like every family before it — every
// assertion here exists because the obvious misreading produces a model that
// runs and is wrong rather than one that fails to load:
//
//   * THE WRAPPER. The language model is nested under `text_config` and the top
//     level carries almost none of the keys the adapter reads. Read flat it does
//     not fail; it succeeds with n_layers 0.
//   * BLOCKS, not keys. A query sees `topk_blocks * block_size` keys — 2048 —
//     and a reader who transcribes DSA's per-key top-k selects 16 of them. Both
//     produce finite logits.
//   * TWO SPELLINGS of the same indexer. The checkpoint ships
//     `sparse_attention_config`; a config re-saved by transformers ships flat
//     `index_*` keys and `layer_types`. Reading one means the fixture and the
//     model it stands for are not the same family, so both are read here and
//     asserted to agree — down to the arch_hash.
//   * `moe_layer_freq` AS A LIST. Every family before this states it as a
//     stride. Read as a scalar the key is simply ignored, the three dense layers
//     become MoE, and `bytes_per_token` gains three layers of routed experts.
//   * THE EXPERT WIDTH IS `intermediate_size` and the DENSE width is not. This
//     is the first family to state them separately, and every earlier one falls
//     back to the shared key.
//   * THE SELECTION BIAS with no `topk_method` to announce it. The config says
//     `use_routing_bias`, names no method, and the bias changes which experts
//     fire on 42% of (token, layer) pairs at fixture scale.
//   * THE INDEXER'S KEY IS CACHED, one head wide. It widens the K plane by
//     `index_head_dim` per position — 30 GiB at this model's stated context —
//     and sizing it per indexer head would overstate that fourfold.
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

constexpr std::uint32_t kLayers = 60;
constexpr std::uint32_t kDense = 3;  ///< leading layers: dense MLP, dense attention
constexpr std::uint32_t kSparse = 57;
constexpr std::uint32_t kDModel = 6144;
constexpr std::uint32_t kHeads = 64;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kIndexDim = 128;
constexpr std::uint32_t kIndexHeads = 4;
constexpr std::uint32_t kExperts = 128;
constexpr std::uint32_t kTopK = 4;
constexpr std::uint32_t kExpertInter = 3072;

/// `[0,0,0,1,1,...]` — the per-layer mask, written out rather than abbreviated.
std::string prefix_mask() {
    std::string s = "[";
    for (std::uint32_t i = 0; i < kLayers; ++i) {
        s += (i < kDense) ? "0" : "1";
        if (i + 1 < kLayers) s += ",";
    }
    return s + "]";
}

/// The text half, shared by both spellings below.
std::string text_body() {
    return R"json(
  "hidden_size":6144,"intermediate_size":3072,"num_hidden_layers":60,
  "num_attention_heads":64,"num_key_value_heads":4,"head_dim":128,
  "vocab_size":200064,"max_position_embeddings":1048576,
  "rms_norm_eps":1e-06,"use_gemma_norm":true,"attention_output_gate":false,
  "rope_theta":5000000,"rotary_dim":64,"partial_rotary_factor":0.5,
  "hidden_act":"swigluoai","use_qk_norm":true,"tie_word_embeddings":false,
  "dense_intermediate_size":12288,"shared_intermediate_size":3072,
  "num_local_experts":128,"num_experts_per_tok":4,"n_shared_experts":1,
  "scoring_func":"sigmoid","use_routing_bias":true,
  "qk_norm_type":"per_head","num_mtp_modules":7,"num_nextn_predict_layers":1,
  "swiglu_alpha":1.702,"swiglu_limit":7.0,"routed_scaling_factor":2.0,
  "eos_token_id":200020,"bos_token_id":200034,
  "architectures":["MiniMaxM3SparseForCausalLM"])json";
}

/// The real `MiniMaxAI/MiniMax-M3` config.json: a multimodal wrapper whose
/// indexer is described by the NESTED `sparse_attention_config`.
std::string wrapper_config() {
    const auto mask = prefix_mask();
    return std::string(R"json({
  "architectures":["MiniMaxM3SparseForConditionalGeneration"],
  "model_type":"minimax_m3_vl",
  "image_token_index":200025,"video_token_index":200026,
  "projector_hidden_size":6144,"torch_dtype":"bfloat16",
  "vision_config":{"hidden_size":1280,"num_attention_heads":16,
    "num_hidden_layers":32,"intermediate_size":5120,"patch_size":14,
    "image_size":2016,"model_type":"clip_vision_model"},
  "text_config":{)json") +
           text_body() + ",\n  \"moe_layer_freq\":" + mask +
           R"json(,
  "sparse_attention_config":{"use_sparse_attention":true,
    "sparse_index_dim":128,"sparse_num_index_heads":4,
    "sparse_topk_blocks":16,"sparse_block_size":128,
    "sparse_score_type":"max","sparse_init_block":0,"sparse_local_block":1,
    "sparse_disable_index_value":)json" +
           mask + ",\n    \"sparse_attention_freq\":" + mask + "}\n  }\n}";
}

/// The SAME model as `transformers` re-saves it: text-only, with the indexer
/// flattened into `index_*` keys and the layer split stated as `layer_types`.
/// This is what the tiny fixture carries.
std::string flat_config() {
    std::string types = "[";
    std::string mlp_types = "[";
    for (std::uint32_t i = 0; i < kLayers; ++i) {
        types += (i < kDense) ? "\"full_attention\"" : "\"minimax_m3_sparse\"";
        mlp_types += (i < kDense) ? "\"dense\"" : "\"sparse\"";
        if (i + 1 < kLayers) {
            types += ",";
            mlp_types += ",";
        }
    }
    types += "]";
    mlp_types += "]";
    return std::string("{\n  \"model_type\":\"minimax_m3_vl_text\",") + text_body() +
           ",\n  \"layer_types\":" + types + ",\n  \"mlp_layer_types\":" + mlp_types +
           R"json(,
  "index_n_heads":4,"index_head_dim":128,"index_block_size":128,
  "index_topk_blocks":16,"index_local_blocks":1
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

std::uint64_t f32_sizer(const soma::ArchIr&,
                        std::uint32_t rows,
                        std::uint32_t cols,
                        soma::TensorRole) {
    return soma::quantized_tensor_bytes(soma::DType::F32, rows, cols, 128);
}

std::uint64_t f32_bytes(std::uint32_t rows, std::uint32_t cols) {
    return soma::quantized_tensor_bytes(soma::DType::F32, rows, cols, 128);
}

} // namespace

int main() {
    soma::ArchIr arch;
    CHECK(soma::adapt_hf_config(wrapper_config(), arch).ok());

    // ── the wrapper, hoisted ─────────────────────────────────────────────────
    CHECK(arch.source_model_type == "minimax_m3_vl");
    CHECK(arch.topology.n_layers == kLayers);
    CHECK(arch.topology.d_model == kDModel);
    CHECK(arch.topology.vocab_size == 200064);
    CHECK(arch.topology.max_position_embeddings == 1048576);
    CHECK(arch.topology.eos_token_ids == std::vector<std::uint32_t>{200020});
    CHECK(!arch.topology.tie_word_embeddings);

    // The tower is DESCRIBED, not served. Recorded through the plain
    // `num_hidden_layers`/`hidden_size` spelling — Kimi-K3's `vt_`-prefixed one
    // is tried first and finds nothing here, and reading only that left this
    // family reporting a 0-layer tower, i.e. claiming a text model.
    CHECK(arch.modality.modality == soma::Modality::VisionText);
    CHECK(arch.modality.vision_layers == 32);
    CHECK(arch.modality.vision_hidden == 1280);
    CHECK(arch.modality.vision_patch_size == 14);
    CHECK(arch.modality.media_placeholder_token_id == 200025);

    // ── the dense prefix, from a LIST-valued moe_layer_freq ──────────────────
    //
    // Read as a scalar this key is skipped entirely and all 60 layers come out
    // MoE. Nothing fails: the planner charges three extra layers of routed
    // experts, which is 2.7 TB of over-count on the quantity the verdict is
    // computed from.
    CHECK(arch.n_moe_layers() == kSparse);
    CHECK(!arch.is_moe_layer(0));
    CHECK(!arch.is_moe_layer(2));
    CHECK(arch.is_moe_layer(3));
    CHECK(arch.is_moe_layer(kLayers - 1));

    // ── attention: GQA, per-head norms, half-rotated ─────────────────────────
    CHECK(arch.attention.family == soma::AttentionFamily::GqaBsa);
    CHECK(arch.attention.n_heads == kHeads && arch.attention.n_kv_heads == kKvHeads);
    CHECK(arch.attention.head_dim == kHeadDim);
    CHECK(arch.attention.qk_norm == soma::QkNormKind::PerHead);
    CHECK(!arch.attention.fused_output_gate); // `attention_output_gate: false`
    CHECK(!arch.attention.bias);
    CHECK(arch.attention.sliding_window == 0);
    // 0.5 x 128, cross-checked against the config's own `rotary_dim: 64`.
    CHECK(arch.attention.rope.partial_dim == 64);
    CHECK(arch.attention.rope.theta == 5000000.0f);
    CHECK(arch.attention.rope.scaling.kind == soma::RopeScalingKind::None);
    // Gemma's convention, and the config states it. A checkpoint written for
    // `(1 + w)` and read as `w` is scaled by roughly nothing.
    CHECK(arch.rms_norm_scale == soma::RmsNormScale::OnePlusWeight);
    CHECK(arch.rms_norm_weight_offset() == 1.0f);

    // ── the indexer ──────────────────────────────────────────────────────────
    const auto& bsa = arch.attention.bsa;
    CHECK(bsa.n_index_heads == kIndexHeads);
    CHECK(bsa.index_head_dim == kIndexDim);
    CHECK(bsa.block_size == 128);
    CHECK(bsa.topk_blocks == 16);
    CHECK(bsa.local_blocks == 1);
    CHECK(bsa.layer_kinds.size() == kLayers);
    CHECK(bsa.n_indexed_layers() == kSparse);
    CHECK(bsa.layer_kinds[0] == soma::IndexerKind::None);
    CHECK(bsa.layer_kinds[2] == soma::IndexerKind::None);
    CHECK(bsa.layer_kinds[3] == soma::IndexerKind::Full);
    CHECK(bsa.layer_kinds[kLayers - 1] == soma::IndexerKind::Full);
    // No IndexShare in this family: every sparse layer owns its weights.
    for (const auto kind : bsa.layer_kinds) CHECK(kind != soma::IndexerKind::Shared);

    // BLOCKS, not keys. 16 x 128 = 2048, and the DSA-shaped misreading gives 16.
    CHECK(bsa.visible_keys(1000000) == 2048);
    // Below the span every block is selected and the sparse path is dense —
    // which is the number any test of this family has to sit above.
    CHECK(bsa.visible_keys(512) == 512);
    CHECK(bsa.visible_keys(2048) == 2048);

    // ── the two spellings are the same model ─────────────────────────────────
    //
    // Not merely "both parse". The FIXTURE carries the flat form and the
    // CHECKPOINT carries the nested one, so anything that distinguished them
    // would mean the oracle graded a different architecture from the one served.
    {
        soma::ArchIr flat;
        CHECK(soma::adapt_hf_config(flat_config(), flat).ok());
        CHECK(flat.attention.family == soma::AttentionFamily::GqaBsa);
        CHECK(flat.attention.bsa.layer_kinds == bsa.layer_kinds);
        CHECK(flat.attention.bsa.n_index_heads == bsa.n_index_heads);
        CHECK(flat.attention.bsa.index_head_dim == bsa.index_head_dim);
        CHECK(flat.attention.bsa.block_size == bsa.block_size);
        CHECK(flat.attention.bsa.topk_blocks == bsa.topk_blocks);
        CHECK(flat.attention.bsa.local_blocks == bsa.local_blocks);
        CHECK(flat.topology.layer_kinds == arch.topology.layer_kinds);
        CHECK(flat.ffn.activation == arch.ffn.activation);
        CHECK(flat.router.normalize_topk == arch.router.normalize_topk);
        // The one legitimate difference: the text-only spelling declares no
        // tower, so it is a text model and the wrapper is not. That difference
        // is IN the hash, deliberately.
        CHECK(flat.modality.modality == soma::Modality::Text);
    }

    // ── the router ───────────────────────────────────────────────────────────
    CHECK(arch.router.n_experts == kExperts);
    CHECK(arch.router.top_k == kTopK);
    CHECK(arch.router.score_fn == soma::ScoreFn::Sigmoid);
    // FROM `use_routing_bias`, with no `topk_method` in the file. Reading only
    // the method leaves this false on a router that biases its selection.
    CHECK(arch.router.bias_correction);
    // FORCED. `MiniMaxM3VLTopKRouter` divides by the top-k sum unconditionally
    // and the config carries no `norm_topk_prob`.
    CHECK(arch.router.normalize_topk);
    CHECK(arch.router.routed_scaling_factor == 2.0f);
    CHECK(arch.router.n_shared_experts == 1);
    CHECK(arch.router.n_groups == 1 && arch.router.topk_group == 1);

    // ── ffn: three widths and a clamped activation ───────────────────────────
    CHECK(arch.ffn.activation == soma::Activation::SwiGluOai);
    CHECK(arch.ffn.swiglu_alpha == 1.702f);
    CHECK(arch.ffn.swiglu_limit == 7.0f);
    // `intermediate_size` is the EXPERT width here. Every earlier family would
    // have read it as the dense one.
    CHECK(arch.ffn.expert_intermediate == kExpertInter);
    CHECK(arch.ffn.dense_intermediate == 12288);
    CHECK(arch.ffn.shared_intermediate == 3072);
    // Ungated, unlike Qwen's: `MiniMaxM3VLSparseMoeBlock` adds the shared branch
    // at full strength.
    CHECK(!arch.ffn.shared_expert_gate);
    CHECK(arch.ffn.routed_expert_hidden == 0);
    CHECK(arch.routed_expert_width() == kDModel);
    CHECK(arch.block_residual.block_size == 0);
    CHECK(arch.hyper_connections.multiplier == 1);

    // ── the MTP head is declared, and cannot be served ───────────────────────
    //
    // `num_mtp_modules` is 7 and `num_nextn_predict_layers` is 1. The LAYER
    // count is what is comparable to GLM-5.2's and Qwen3.5's; recording the
    // module count would make this head look seven times its size.
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
    CHECK(plan.attention_family == std::string("gqa+bsa"));
    CHECK(plan.modality == std::string("vision+text"));
    CHECK(plan.vision_layers == 32);

    // SERVABLE from the start, and that is a claim about evidence:
    // `tests/fixtures/tiny/MiniMax-M3-Tiny` is an oracle built by transformers
    // 5.15.1's own `modeling_minimax_m3_vl.py`, which Soma matches at 1.22e-06
    // over 512 teacher-forced positions with 256 greedy tokens exact — while a
    // DENSE implementation of the same weights differs from it by 7.8e-01 and
    // picks a different argmax at 217 of those 512 positions.
    CHECK(plan.arch_supported);

    // ── expert economics ─────────────────────────────────────────────────────
    const auto expert = 2ull * f32_bytes(kExpertInter, kDModel) + f32_bytes(kDModel, kExpertInter);
    CHECK(plan.expert_bytes == expert);
    CHECK(expert == 226492416ull);
    // 57 MoE layers, not 60. The scalar reading of `moe_layer_freq` gives 60.
    CHECK(plan.bytes_per_token == std::uint64_t{kSparse} * kTopK * expert);
    CHECK(plan.total_routed_bytes == std::uint64_t{kSparse} * kExperts * expert);
    CHECK(plan.active_fraction == static_cast<float>(kTopK) / kExperts);

    // ── the cache: full K/V, plus ONE indexer key per position ───────────────
    const auto* attn = soma::resolve_attention_backend(arch.attention.family);
    CHECK(attn != nullptr);
    CHECK(std::string(attn->name) == "gqa");
    // Deliberately NOT declared: `kv_bytes_for_context` would switch KvCache to
    // its opaque layout, and this family's cache is two ordinary planes.
    CHECK(attn->kv_bytes_for_context == nullptr);
    CHECK(attn->kv_bytes_per_token != nullptr);
    CHECK(attn->resident_weight_bytes != nullptr);

    const std::uint64_t hkv = std::uint64_t{kKvHeads} * kHeadDim; // 512
    // K plane carries `[K | index_k]`; V plane carries V.
    const std::uint64_t per_token = std::uint64_t{kLayers} * (2 * hkv + kIndexDim) * sizeof(float);
    CHECK(attn->kv_bytes_per_token(arch) == per_token);
    CHECK(per_token == 276480ull);
    // The plain-GQA reading, which omits the indexer key entirely.
    const std::uint64_t without_indexer = std::uint64_t{kLayers} * 2 * hkv * sizeof(float);
    CHECK(without_indexer == 245760ull);
    // 30 GiB of difference at the context this model advertises — on the cache,
    // which competes with the expert cache for the same RAM.
    CHECK(per_token * 1048576ull - without_indexer * 1048576ull == 32212254720ull);
    // One key head, not `n_index_heads` of them. The per-head reading would add
    // 4x this much.
    CHECK(per_token - without_indexer == std::uint64_t{kLayers} * kIndexDim * sizeof(float));

    // ── the resident half is NOT uniform across layers ───────────────────────
    const auto per_layer = attn->weight_bytes_per_layer(arch, &f32_sizer);
    CHECK(per_layer == f32_bytes(kHeads * kHeadDim, kDModel) +
                           2 * f32_bytes(hkv, kDModel) +
                           f32_bytes(kDModel, kHeads * kHeadDim));
    const auto indexer_layer = f32_bytes(kIndexHeads * kIndexDim, kDModel) +
                               f32_bytes(kIndexDim, kDModel) + 2ull * kIndexDim * sizeof(float);
    CHECK(attn->resident_weight_bytes(arch, &f32_sizer) ==
          std::uint64_t{kLayers} * per_layer + std::uint64_t{kSparse} * indexer_layer);
    // 855 MiB the per-layer average cannot express: 57 layers own an indexer and
    // 3 do not.
    CHECK(std::uint64_t{kSparse} * indexer_layer == 896590848ull);

    // ── the same descriptor still serves plain GQA exactly as before ─────────
    //
    // `resident_weight_bytes` is new and is PREFERRED over the per-layer average
    // once declared, so a family with no indexer must get the identical number
    // by both routes or every existing GQA plan just moved.
    {
        soma::ArchIr qwen;
        CHECK(soma::adapt_hf_config(kQwen3Config, qwen).ok());
        const auto* q = soma::resolve_attention_backend(qwen.attention.family);
        CHECK(q != nullptr);
        CHECK(q->resident_weight_bytes(qwen, &f32_sizer) ==
              std::uint64_t{qwen.topology.n_layers} * q->weight_bytes_per_layer(qwen, &f32_sizer));
        // …and its cache is untouched by the indexer plane.
        CHECK(q->kv_bytes_per_token(qwen) ==
              std::uint64_t{qwen.topology.n_layers} * 2ull *
                  (qwen.attention.n_kv_heads * qwen.attention.head_dim) * sizeof(float));
    }

    // ── hashing is conditional ───────────────────────────────────────────────
    std::string hash_wrapper, hash_flat, hash_qwen, hash_qwen_again;
    CHECK(soma::compute_arch_hash(arch, hash_wrapper).ok());
    CHECK(!hash_wrapper.empty());
    {
        soma::ArchIr flat;
        CHECK(soma::adapt_hf_config(flat_config(), flat).ok());
        CHECK(soma::compute_arch_hash(flat, hash_flat).ok());
        // DIFFERENT, and that is the point: one declares a vision tower and the
        // other does not, so they are not interchangeable checkpoints even
        // though their text stacks agree.
        CHECK(hash_flat != hash_wrapper);
    }
    {
        soma::ArchIr qwen;
        CHECK(soma::adapt_hf_config(kQwen3Config, qwen).ok());
        CHECK(soma::compute_arch_hash(qwen, hash_qwen).ok());
        CHECK(soma::compute_arch_hash(qwen, hash_qwen_again).ok());
        CHECK(hash_qwen == hash_qwen_again);
        CHECK(hash_qwen != hash_wrapper);
    }

    // Two models differing ONLY in a block-sparse parameter are two models.
    {
        soma::ArchIr other = arch;
        other.attention.bsa.topk_blocks = 8;
        std::string h;
        CHECK(soma::compute_arch_hash(other, h).ok());
        CHECK(h != hash_wrapper);
    }
    {
        soma::ArchIr other = arch;
        other.attention.bsa.local_blocks = 2;
        std::string h;
        CHECK(soma::compute_arch_hash(other, h).ok());
        CHECK(h != hash_wrapper);
    }
    // …and so are two that differ only in the activation's gain.
    {
        soma::ArchIr other = arch;
        other.ffn.swiglu_alpha = 1.0f;
        std::string h;
        CHECK(soma::compute_arch_hash(other, h).ok());
        CHECK(h != hash_wrapper);
    }

    // ── malformed configs are refused, not repaired ──────────────────────────
    {
        // A stack with no indexed layer is plain GQA wearing this family's name,
        // and would be charged for an indexer plane nothing writes.
        soma::ArchIr bad = arch;
        bad.attention.bsa.layer_kinds.assign(kLayers, soma::IndexerKind::None);
        CHECK(!soma::validate_arch_ir(bad).ok());
    }
    {
        // More forced-local blocks than top-k slots cannot be satisfied:
        // upstream forces them THROUGH the score, so the extra ones are silently
        // dropped rather than growing the selection.
        soma::ArchIr bad = arch;
        bad.attention.bsa.local_blocks = bad.attention.bsa.topk_blocks + 1;
        CHECK(!soma::validate_arch_ir(bad).ok());
    }
    {
        // An indexer head narrower than the rotation would truncate the rope
        // table mid-frequency. Refused rather than transcribed.
        soma::ArchIr bad = arch;
        bad.attention.bsa.index_head_dim = 32; // rotary width is 64
        CHECK(!soma::validate_arch_ir(bad).ok());
    }
    {
        // `n_heads` must be a multiple of `index_n_heads`, or some query heads
        // have no indexer head to obey.
        soma::ArchIr bad = arch;
        bad.attention.bsa.n_index_heads = 7;
        CHECK(!soma::validate_arch_ir(bad).ok());
    }
    {
        // A config that states `rotary_dim` and `partial_rotary_factor` and
        // disagrees with itself has two opinions about the one quantity that
        // decides its long-context behaviour.
        std::string cfg = wrapper_config();
        const auto at = cfg.find("\"rotary_dim\":64");
        CHECK(at != std::string::npos);
        cfg.replace(at, std::string("\"rotary_dim\":64").size(), "\"rotary_dim\":96");
        soma::ArchIr bad;
        CHECK(!soma::adapt_hf_config(cfg, bad).ok());
    }

    std::cout << "minimax_m3_plan: OK\n"
              << "  layers        " << kLayers << " (" << kDense << " dense + " << kSparse
              << " block-sparse MoE)\n"
              << "  visible keys  " << bsa.visible_keys(1u << 20) << " of "
              << arch.topology.max_position_embeddings << " (" << bsa.topk_blocks << " x "
              << bsa.block_size << ")\n"
              << "  bytes/token   " << (plan.bytes_per_token >> 20) << " MiB\n"
              << "  routed set    " << (plan.total_routed_bytes >> 30) << " GiB\n"
              << "  kv/token      " << per_token << " B  (" << (per_token - without_indexer)
              << " B of it the indexer's key)\n"
              << "  attn resident " << (attn->resident_weight_bytes(arch, &f32_sizer) >> 20)
              << " MiB\n";
    return 0;
}
