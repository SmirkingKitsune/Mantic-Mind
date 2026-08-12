#pragma once

// Soma — MLA attention backend (Multi-head Latent Attention).
//
// Second architecture through the seam, at G4. Its job at that gate is proving
// the seam carries a second attention FAMILY — architectural correctness, not
// streaming economics.
//
// Reference config this backend is designed against:
//
//   DeepSeek-V2-Lite  27 layers, d_model 2048, 16 heads
//                     kv_lora_rank 512, q_lora_rank null (no Q down-projection)
//                     qk_nope_head_dim 128, qk_rope_head_dim 64, v_head_dim 128
//                     64 routed experts, top-6, 2 shared, moe_intermediate 1408
//                     first_k_dense_replace 1  -> 26 MoE layers, layer 0 dense
//                     softmax scoring, n_group 1, routed_scaling_factor 1.0
//                     rope_theta 1e4 with YaRN scaling (factor 40, mscale 0.707)
//
// The cheapest real MLA + fine-grained-MoE checkpoint that exists, so G4 does
// not require 100B-class hardware.
//
// NOTE it will likely admit as resident-only at q4 (7.2 GB routed set fits in
// RAM). That is expected and fine: the conformance ladder runs regardless of
// verdict, production PLACEMENT is what the verdict gates, and forcing Soma for
// the test is exactly what backend_override: soma exists for.
//
// DEPENDENCY RULE: this header may include core headers. Core headers may not
// include this one, and may not mention "mla". CI enforces it.

#include "soma/arch_backend.hpp"
#include "soma/arch_ir.hpp"
#include "soma/attention_backend.hpp"
#include "soma/f32_model.hpp"
#include "soma/types.hpp"

#include <cstddef>
#include <span>

namespace soma::arch::mla {

/// This backend's KV checkpoint tag. Owned here, not in a core enum.
/// v2: the V plane changed shape. It was the K plane's width for every layer and
/// is now `index_head_dim` for DSA and ABSENT for plain MLA, so a checkpoint
/// written under v1 has a different layout and must be refused rather than
/// replayed into a cache shaped differently. That refusal is the entire reason
/// this id is per-backend and not a global version.
inline constexpr KvFormatId kKvFormat = kv_format_id("soma.kv.mla.latent.v2");

/// Compressed latent + RoPE slice: kv_lora_rank + qk_rope_head_dim elements per
/// token per layer. Note this is independent of head count — the whole point.
///
///   DeepSeek-V2-Lite  512 + 64 = 576 elem/tok/layer × 27 = 31 KB/tok @fp16
///                     vs. uncompressed 16×(128+64) + 16×128 = 5120 -> ~8.9x
///
/// The ratio grows with head count, so it is far larger on full-size V2/V3
/// (128 heads). Compressed KV is the difference between feasible and not for
/// this family, which is why it is carried forward from the prior art rather
/// than treated as an optimization.
std::size_t kv_bytes_per_token(const ArchIr& arch) noexcept;

/// Weight absorption's load-time hook — DELIBERATELY A NO-OP, because absorption
/// turned out not to want load time.
///
/// The absorption itself is real and is in `f32_attention_kv`: `(W_k^T q) . c`
/// rather than `q . (W_k c)`, and one projection of the accumulated latent rather
/// than one per attended key. What it does NOT need is a folded copy of the
/// weights sitting in memory.
///
/// Folding at load means materializing `W_k^T` per layer as fp32 — on GLM-5.2
/// that is `n_heads * qk_nope * kv_lora_rank * 4` bytes per layer, 1.96 GB across
/// the stack, against a plan that fits a 24 GiB host by 3 GB. Transposing per step
/// instead costs `n_heads * qk_nope * kv_lora_rank` element reads per layer per
/// step: 6.3e6, next to the 1.5e8 the absorbed attention already does and the
/// 3.0e10 it replaced. The memory was the scarce thing; the arithmetic was not.
///
/// This declaration previously described the load-time fold in detail and was
/// never defined at all — the same shape as the `attention_backend()` bug in D16,
/// where a declared-never-defined function meant the planner silently used GQA's
/// formula for MLA. Defined here so the description and the code agree, and so a
/// future caller gets a no-op rather than a link error (roadmap D39).
Status prepare_weights(ModelState& model);

StatusCode prefill(const ModelState& model,
                   ExecScratch& exec,
                   SeqBatch& batch,
                   LayerIndex layer,
                   std::span<const float> in,
                   std::span<float> out) noexcept;

/// Latent-space decode against the compressed cache.
///
/// Same signature as the GQA path, and that is the seam working: the core calls
/// this identically, having never learned that the cache holds a latent rather
/// than K and V.
StatusCode decode(const ModelState& model,
                  ExecScratch& exec,
                  SeqBatch& batch,
                  LayerIndex layer,
                  std::span<const float> in,
                  std::span<float> out) noexcept;

StatusCode init_kv_region(const ArchIr& arch, KvRegion& region) noexcept;

/// Partial RoPE: only qk_rope_head_dim of each head carries position, the
/// qk_nope_head_dim remainder does not. Applied to the RoPE slice only.
StatusCode apply_rope(const ArchIr& arch,
                      std::span<float> q,
                      std::span<float> k,
                      std::span<const std::uint32_t> positions) noexcept;

/// YaRN scaling, including the mscale attenuation applied to attention logits.
/// Declared separately because it is not a rope variant so much as a rope
/// variant plus a logit scale, and folding it into apply_rope would hide the
/// second half.
float yarn_mscale(const ArchIr& arch) noexcept;

const AttentionBackend& attention_backend() noexcept;

// ── Router ───────────────────────────────────────────────────────────────────

/// Group-limited top-k with optional pre-selection bias correction and
/// post-selection scaling.
///
/// V2-Lite is the degenerate case (n_group 1, topk_group 1, no bias correction,
/// routed_scaling_factor 1.0) but the general form is implemented from the
/// start: n_group > 1 selects candidate groups by their top expert before
/// selecting experts within them, which is a different algorithm rather than a
/// filter applied afterwards. Retrofitting it would mean rewriting the function
/// that decides which experts fire.
StatusCode route(const ArchIr& arch,
                 std::span<const float> logits_f32,
                 std::uint32_t n_rows,
                 RouterOut& out) noexcept;

StatusCode apply_expert(const ModelState& model,
                        ExecScratch& exec,
                        LayerIndex layer,
                        ExpertId expert,
                        CByteSpan expert_bytes,
                        std::span<const std::uint32_t> row_indices,
                        std::span<const float> row_weights) noexcept;

/// Layer 0 on V2-Lite (first_k_dense_replace 1). Topology::layer_kinds is
/// authoritative; this reads it rather than re-deriving the stride.
StatusCode dense_ffn(const ModelState& model,
                     ExecScratch& exec,
                     LayerIndex layer,
                     std::uint32_t n_rows) noexcept;

/// Two shared experts on V2-Lite, applied to every row unconditionally —
/// unlike the routed experts, they are never streamed and live in the resident
/// half of the static partition.
StatusCode shared_experts(const ModelState& model,
                          ExecScratch& exec,
                          LayerIndex layer,
                          std::uint32_t n_rows) noexcept;

StatusCode apply_norm(const ModelState& model,
                      ExecScratch& exec,
                      const DenseTensor& weight,
                      std::uint32_t n_rows) noexcept;

Status validate(const ArchIr& arch);

const ArchBackend& backend() noexcept;

// ── fp32 reference path ──────────────────────────────────────────────────────

/// MLA's per-layer attention weights. Nothing here is shared with GQA — which is
/// the point of the opaque payload: neither family's tensor list appears in core.
struct F32AttnWeights {
    /// Used when q_lora_rank == 0 (DeepSeek-V2-Lite). Mutually exclusive with
    /// the q_a/q_b pair, which the full V2 uses instead.
    soma::WeightRef q_proj;
    soma::WeightRef q_a_proj, q_b_proj;
    std::span<const float> q_a_norm;

    soma::WeightRef kv_a_proj; ///< kv_a_proj_with_mqa: latent ++ shared rope
    std::span<const float> kv_a_norm;
    soma::WeightRef kv_b_proj; ///< latent -> per-head (K-nope ++ V)
    soma::WeightRef o_proj;

    /// `mlp.gate.e_score_correction_bias` — V3's `noaux_tc` routing bias.
    ///
    /// A ROUTER parameter living in the attention payload, which is only odd
    /// until you notice the payload is the backend's per-layer state and not
    /// specifically its attention weights. Empty on V2 and on dense layers.
    std::span<const float> e_score_bias;

    /// DSA's indexer. Bound only on `full` layers — `shared` layers own NO
    /// indexer tensors at all (57 of GLM-5.2's 78), so absence here is the
    /// architecture rather than a missing weight.
    ///
    /// `wq_b` reads the SAME `q_resid` the main query path produces when query
    /// LoRA is present, or the hidden state for a direct-query MLA variant.
    /// Either way there is no second down-projection.
    struct Indexer {
        soma::WeightRef wq_b;         ///< [n_index_heads*index_head_dim, q_lora_rank or d_model]
        soma::WeightRef wk;           ///< [index_head_dim, d_model] — ONE head, shared
        soma::WeightRef weights_proj; ///< [n_index_heads, d_model] — per-head mixing

        /// A true LayerNorm: mean-centred, with a BIAS. Not the RMSNorm used
        /// everywhere else in this model, and `k_norm.bias` is the only bias in
        /// the entire attention block — which is the tell that it is not an RMS.
        std::span<const float> k_norm_w, k_norm_b;
    };

    bool has_indexer = false;
    Indexer idx;
};

/// One `full` layer's key selection, reused by the `shared` layers after it.
///
/// Lives in `F32Workspace::arch_state`, so the core never sees this type. The
/// entries are KEY POSITIONS, which is what makes sharing sound: a position means
/// the same thing in every layer even though each layer's keys differ.
struct DsaSelection {
    std::uint32_t n_tokens = 0;
    std::uint32_t stride = 0;          ///< allocated top-k slots per query
    std::vector<std::uint32_t> keys;   ///< [n_tokens * stride]
    std::vector<std::uint32_t> counts; ///< selected count per query (<= stride)

    /// True when every query kept every key it was allowed to attend. Recorded
    /// rather than inferred because it is the condition under which DSA is
    /// bit-identical to dense MLA — a test run entirely inside it proves nothing
    /// about the indexer, and the harness needs to be able to SAY so.
    bool dense_equivalent = false;

    std::span<const std::uint32_t> for_query(std::uint32_t t) const noexcept {
        return {keys.data() + static_cast<std::size_t>(t) * stride, counts[t]};
    }
};

/// Compute one `full` layer's selection. Exposed rather than kept internal so the
/// conformance harness can check the SELECTION, not only the logits it leads to.
StatusCode f32_index_select(const ArchIr& arch,
                            const F32AttnWeights& w,
                            const float* x,
                            const float* q_resid,
                            std::uint32_t n_tokens,
                            std::uint32_t pos_base,
                            DsaSelection& out) noexcept;

StatusCode f32_bind_layer(const ArchIr& arch,
                          const soma::LayerBindCtx& ctx,
                          soma::ArchLayerPayload& out) noexcept;

StatusCode f32_attention(const ArchIr& arch,
                         const soma::F32LayerWeights& lw,
                         const float* x,
                         std::uint32_t n_tokens,
                         soma::F32Workspace& ws,
                         float* out) noexcept;

/// The decode-side selection: same scoring, keys read from the cache.
///
/// Separate from `f32_index_select` rather than a flag on it, for the reason the
/// two attention entry points are separate — prefill owns the whole sequence and
/// computes every key; this owns one position per row and must read the rest.
StatusCode f32_index_select_kv(const ArchIr& arch,
                               const F32AttnWeights& w,
                               const float* x,
                               const float* q_resid,
                               std::uint32_t n_rows,
                               LayerIndex layer,
                               const soma::KvRow* rows,
                               DsaSelection& out) noexcept;

/// K is `kv_lora_rank + qk_rope_head_dim` — the compressed latent plus the single
/// shared RoPE segment, independent of head count, which is the point.
///
/// V is `index_head_dim` for DSA and **zero** for plain MLA, because MLA derives
/// V from the latent rather than storing it. Reporting the K width for both, as
/// this did, allocated a full second plane that nothing ever read: 2.94 GB on
/// GLM-5.2 at 4k x 4 slots, and every byte of it dead for DeepSeek-V2-Lite and
/// Moonlight.
///
/// DSA's indexer key is the one thing that legitimately lives there: `k_norm(wk(x))`
/// roped, one vector per token, needed at every later step and impossible to
/// recompute then because it depends on that token's hidden state at that layer.
KvGeometry f32_kv_geometry(const ArchIr& arch) noexcept;

StatusCode f32_attention_kv(const ArchIr& arch,
                            const soma::F32LayerWeights& lw,
                            const float* x,
                            std::uint32_t n_rows,
                            LayerIndex layer,
                            const soma::KvRow* rows,
                            soma::F32Workspace& ws,
                            float* out) noexcept;

StatusCode f32_route(const ArchIr& arch,
                     const soma::F32LayerWeights& lw,
                     const float* logits,
                     std::uint32_t n_tokens,
                     std::uint32_t* out_ids,
                     float* out_weights) noexcept;

const soma::F32Backend& f32_backend() noexcept;

} // namespace soma::arch::mla
