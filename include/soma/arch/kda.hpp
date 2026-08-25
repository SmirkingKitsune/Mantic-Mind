#pragma once

// Soma — hybrid linear/full attention backend (KDA + MLA).
//
// Fourth family through the seam. What it contributes that the first three did
// not is a stack whose layers do not all cost the same thing:
//
//   * a MINORITY of layers are MLA — compressed latent KV, growing per token,
//     here with NoPE and a sigmoid output gate;
//   * the MAJORITY are gated delta-rule linear attention, carrying a
//     `n_heads x head_dim x head_dim` recurrent matrix and a short convolution
//     window, both CONSTANT in context length.
//
// Reference config this backend is designed against:
//
//   Kimi-K3          93 layers, d_model 7168, 96 heads
//                    24 full-attention layers (1-based 4,8,…,88,92,93)
//                    69 KDA layers, head_dim 128, short conv 4, full-rank gate
//                    kv_lora_rank 512, q_lora_rank 1536
//                    qk_nope_head_dim 128, qk_rope_head_dim 64, v_head_dim 128
//                    mla_use_nope, mla_use_output_gate
//                    896 routed experts, top-16, 2 shared, moe_intermediate 3072
//                    routed_expert_hidden_size 3584 (LATENT MoE)
//                    first_k_dense_replace 1 -> 92 MoE layers, layer 0 dense
//                    situ activation, sigmoid + noaux_tc routing
//                    max_position_embeddings 1048576
//
// WHY THE SPLIT IS NOT AN OPTIMIZATION DETAIL. At Kimi-K3's 1M context, sizing
// all 93 layers as MLA asks for ~225 GB of KV per sequence; the real figure is
// ~58 GB, because 69 of those layers hold a fixed ~443 MiB between them no
// matter how long the context runs. A planner that got this wrong would refuse a
// model that fits, which is the failure mode a verdict exists to prevent.
//
// The corollary matters too: the constant term dominates at SHORT context. Below
// roughly 17k tokens this stack wants more per-sequence memory than an all-MLA
// stack of the same shape would, so `kv_bytes_for_context` is genuinely affine
// and not linear.
//
// DEPENDENCY RULE: this header may include core headers. Core headers may not
// include this one, and may not mention "kda". CI enforces it.

#include "soma/arch/mla.hpp"
#include "soma/arch_ir.hpp"
#include "soma/attention_backend.hpp"
#include "soma/f32_model.hpp"
#include "soma/kv_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace soma::arch::kda {

/// This backend's KV checkpoint tag. Owned here, not in a core enum.
///
/// Deliberately NOT shared with `mla`'s. A hybrid checkpoint holds 24 layers of
/// latent cache interleaved with 69 recurrent states; an MLA checkpoint of the
/// same nominal model holds 93 layers of latent cache. Same family name in
/// prose, different bytes, and replaying one as the other resumes a conversation
/// the cache does not describe — fluently, and with nothing to detect it.
inline constexpr KvFormatId kKvFormat = kv_format_id("soma.kv.mla.kda.hybrid.v1");

// ── the opaque cache ─────────────────────────────────────────────────────────
//
// This family's cache cannot be a pair of planes. 24 layers want a latent that
// grows per token; 69 want a fixed matrix and a convolution window that do not.
// `KvCache` already supports an opaque per-family buffer, and selects it exactly
// when a backend supplies `kv_bytes_for_context` — so the byte count below IS
// the allocation, and the layout below is the only thing allowed to interpret it.

/// Byte offsets of one layer's regions within the opaque buffer.
///
/// A full-attention layer uses `latent` only; a linear layer uses `recurrent`
/// and `conv` only. The unused offsets equal `end`, so touching one is a
/// zero-length access rather than an alias onto a neighbour.
struct LayerRegion {
    std::size_t latent = 0;    ///< [ctx][kv_lora_rank + qk_rope_head_dim] f32
    std::size_t recurrent = 0; ///< [n_heads][head_dim][head_dim] f32
    std::size_t conv = 0;      ///< [3][n_heads * head_dim][conv_kernel - 1] f32
    std::size_t end = 0;
};

/// Where `layer`'s regions sit. Walks every preceding layer, because the stride
/// is not uniform — that is the point of the family.
LayerRegion layer_region(const ArchIr& arch, std::uint32_t layer, std::uint32_t context) noexcept;

/// Per-sequence state at a given context: the affine one.
///
/// Derived from `layer_region`, NOT computed alongside it. A byte count and a
/// layout that are separately maintained are a byte count and a layout that will
/// eventually disagree, and the symptom is an out-of-bounds write into the next
/// layer's state — which reads as a model that degrades with depth.
std::uint64_t kv_bytes_for_context(const ArchIr& arch, std::uint32_t context) noexcept;

/// Growth rate only — the full-attention layers' contribution per token.
///
/// The constant term is NOT included, because this function has no context to
/// amortise it over and folding it in at some assumed length is how an estimate
/// becomes confidently wrong. Callers that need the truth call
/// `kv_bytes_for_context`.
std::size_t kv_bytes_per_token(const ArchIr& arch) noexcept;

/// Per-sequence bytes that do NOT grow with context.
std::uint64_t recurrent_state_bytes(const ArchIr& arch) noexcept;

const AttentionBackend& attention_backend() noexcept;

// ── the kernel ───────────────────────────────────────────────────────────────
//
// Kimi Delta Attention: a delta-rule linear attention whose forget gate is
// PER CHANNEL rather than per head. That one difference is the architecture —
// the state decays by a diagonal matrix, not a scalar — and it is why this is
// not GatedDeltaNet with different constants.
//
// Reference semantics, transcribed from `fla.ops.kda`'s `naive_recurrent_kda`
// with the flags `modeling_kimi_linear.py` actually passes
// (`use_qk_l2norm_in_kernel`, `use_gate_in_kernel`, `use_beta_sigmoid_in_kernel`,
// `safe_gate` when a lower bound is configured):
//
//     q, k  <- l2norm per head;  q *= head_dim ** -0.5
//     beta  <- sigmoid(b_proj(x))                              per head
//     g     <- lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))   safe
//              -exp(A_log) * softplus(g_raw + dt_bias)                 otherwise
//     S     <- S * exp(g)                    decay along the KEY axis
//     S     <- S + beta * k (x) (v - S^T k)  delta rule against the DECAYED state
//     o     <- S^T q                         from the UPDATED state
//
// Every one of those four orderings is load-bearing and none is guessable:
// predicting from the pre-decay state, or reading the output before the update,
// produces finite plausible numbers and a different model.

/// Log-space decay for one token. `g_raw`, `dt_bias`, `out`: [n_heads*head_dim].
/// `a_log`: [n_heads] — one scalar per head, broadcast across its channels.
void gate(const ArchIr& arch,
          const float* a_log,
          const float* dt_bias,
          const float* g_raw,
          float* out) noexcept;

/// Causal depthwise short convolution over one token, then SiLU.
///
/// `state` carries the previous `kernel - 1` inputs per channel, oldest first,
/// and is advanced in place. `weight` is [width][kernel] with `weight[c][kernel-1]`
/// multiplying the CURRENT token — PyTorch's conv1d ordering. `bias` may be null.
void short_conv(std::uint32_t width,
                std::uint32_t kernel,
                const float* weight,
                const float* bias,
                const float* x,
                float* state,
                float* out) noexcept;

/// One recurrent step. `state` is [n_heads][head_dim][head_dim], row-major with
/// the KEY axis outer; it is read and written. `q`, `k`, `v`, `g` are
/// [n_heads*head_dim]; `beta_raw` is [n_heads] PRE-sigmoid; `out` is
/// [n_heads*head_dim]. `scratch` is [head_dim], caller-owned.
///
/// `q` and `k` arrive un-normalized: L2 and the `head_dim ** -0.5` scale happen
/// here, because they are part of the operator rather than of the projection.
///
/// `scratch` is a parameter rather than a slice of `out` because the delta rule
/// needs its prediction ALIVE while it reads `v`. Borrowing the output buffer
/// works right up until a caller passes `out == v` to save a copy, and then
/// silently corrupts the value it is correcting toward.
void step(const ArchIr& arch,
          const float* q,
          const float* k,
          const float* v,
          const float* g,
          const float* beta_raw,
          float* state,
          float* scratch,
          float* out) noexcept;

/// RMSNorm per head, then weight, then a sigmoid gate — in that order.
///
/// The order is the whole content of this function. Gating BEFORE the norm (as
/// Mamba's `RMSNormGated` does) is a different operator, and both are plausible
/// readings of the name `FusedRMSNormGated`. `fla` normalizes first.
void gated_rmsnorm(const ArchIr& arch,
                   const float* x,
                   const float* gate_raw,
                   const float* weight,
                   float eps,
                   float* out) noexcept;

// ── the F32 execution path ───────────────────────────────────────────────────
//
// One backend for a stack of two layer kinds. The dispatch cannot live in the
// core: `F32Backend::attention` takes no layer index, so the only place that
// knows whether a given layer is Full or Linear is the payload bound to it.
// That is what `F32HybridWeights::linear` is for.
//
// Full layers are MLA layers, and are delegated to the MLA backend rather than
// reimplemented beside it — including its NoPE and output-gate variants, which
// live there because they are variants OF MLA and the next family to want one
// should find it already done.

/// One layer's attention weights: whichever kind this layer is.
struct F32HybridWeights {
    bool linear = false;

    /// Full layers. A COPY of what `mla::f32_bind_layer` produced — every member
    /// is a non-owning view, so copying costs nothing and avoids a second
    /// transcription of MLA's tensor names and its all-or-nothing checks.
    arch::mla::F32AttnWeights full;

    /// Linear layers.
    soma::WeightRef q_proj, k_proj, v_proj;
    std::span<const float> q_conv_w, k_conv_w, v_conv_w;
    std::span<const float> q_conv_b, k_conv_b, v_conv_b; ///< may be empty
    std::span<const float> a_log;                        ///< [n_heads]
    std::span<const float> dt_bias;                      ///< [n_heads * head_dim]
    soma::WeightRef f_a_proj, f_b_proj;
    soma::WeightRef b_proj;
    soma::WeightRef g_proj;             ///< full-rank output gate
    soma::WeightRef g_a_proj, g_b_proj; ///< low-rank output gate
    std::span<const float> o_norm;      ///< [head_dim]
    soma::WeightRef o_proj;

    // ── block residual, on EVERY layer kind ──────────────────────────────────
    //
    // Precomputed as `norm.weight * proj.weight`, which is the only form the
    // mixing uses: `_apply_attn_res` scores a candidate by dotting its
    // RMS-normalized value against that product. Two [d_model] tensors collapse
    // into one at bind time because nothing ever needs them apart.
    //
    // Empty when `BlockResidualSpec::block_size` is zero, i.e. for an ordinary
    // single-stream residual.
    std::vector<float> attn_res_score, mlp_res_score;

    /// `noaux_tc`'s per-expert selection bias, on both layer kinds.
    ///
    /// A ROUTER tensor in an attention payload, which is odd only until you
    /// notice the payload is this backend's per-layer state rather than
    /// specifically its attention weights — `mla::F32AttnWeights` carries the
    /// same field for the same reason. Bound here for BOTH kinds because every
    /// MoE layer routes, whichever attention it uses.
    std::span<const float> e_score_bias;
};

/// Model-level block-residual weights: the final mixing after the last layer.
struct F32HybridModel {
    std::vector<float> out_res_score; ///< [d_model], same product as above
};

/// Per-forward block-residual state, held in `F32Workspace::arch_state`.
///
/// Two things the core's single `hidden` buffer cannot both be. `prefix` is the
/// running residual sum; `hidden` is what the layer norms and attends over, and
/// the two DIVERGE the moment any mixing happens — which is the whole mechanism.
///
/// `prefix_valid` transcribes the reference's `prefix_sum = None`: at a block
/// boundary the layer does not carry the incoming residual forward, it restarts
/// from its own attention output.
///
/// It is NOT load-bearing, and saying so is more useful than implying otherwise.
/// Zeroing the prefix instead of invalidating it computes the same thing —
/// nothing reads it between the boundary and the next `merge_attention`, where
/// `prefix = branch` and `prefix = 0 + branch` agree. Mutation testing is what
/// established that; an earlier version of this comment claimed the two differed.
/// The flag stays because it mirrors the reference line for line and costs a
/// bool, not because a zeroed prefix would be wrong.
struct BlockResidualState {
    std::uint32_t n_tokens = 0;
    std::uint32_t width = 0;
    std::uint32_t n_blocks = 0; ///< snapshots pushed so far
    bool prefix_valid = false;
    std::vector<float> prefix; ///< [n_tokens][width]
    std::vector<float> stack;  ///< [n_tokens][max_blocks][width]
};

StatusCode f32_route(const ArchIr& arch,
                     const soma::F32LayerWeights& lw,
                     const TokenId* input_tokens,
                     const float* logits,
                     std::uint32_t n_tokens,
                     std::uint32_t* out_ids,
                     float* out_weights) noexcept;

StatusCode f32_bind_model(const ArchIr& arch,
                          const soma::ModelBindCtx& ctx,
                          soma::ArchLayerPayload& out) noexcept;

StatusCode f32_begin_forward(const ArchIr& arch,
                             const soma::ArchLayerPayload& model_payload,
                             const TokenId* tokens,
                             std::uint32_t n_tokens,
                             soma::F32Workspace& ws,
                             float* hidden) noexcept;

StatusCode f32_pre_attention(const ArchIr& arch,
                             const soma::F32LayerWeights& lw,
                             std::uint32_t n_tokens,
                             soma::F32Workspace& ws,
                             float* hidden) noexcept;

StatusCode f32_merge_attention(const ArchIr& arch,
                               const soma::F32LayerWeights& lw,
                               const float* branch,
                               std::uint32_t n_tokens,
                               soma::F32Workspace& ws,
                               float* hidden) noexcept;

StatusCode f32_pre_ffn(const ArchIr& arch,
                       const soma::F32LayerWeights& lw,
                       std::uint32_t n_tokens,
                       soma::F32Workspace& ws,
                       float* hidden) noexcept;

StatusCode f32_merge_ffn(const ArchIr& arch,
                         const soma::F32LayerWeights& lw,
                         const float* branch,
                         std::uint32_t n_tokens,
                         soma::F32Workspace& ws,
                         float* hidden) noexcept;

StatusCode f32_end_forward(const ArchIr& arch,
                           const soma::ArchLayerPayload& model_payload,
                           std::uint32_t n_tokens,
                           soma::F32Workspace& ws,
                           float* hidden) noexcept;

/// Softmax-weighted mix of `prefix` with every snapshot on the stack.
///
/// Exposed for the test: the mechanism is small, entirely made of orderings, and
/// each ordering is independently checkable.
void mix_block_residual(const BlockResidualState& st,
                        std::span<const float> score,
                        float eps,
                        const float* prefix,
                        float* out) noexcept;

StatusCode f32_bind_layer(const ArchIr& arch,
                          const soma::LayerBindCtx& ctx,
                          soma::ArchLayerPayload& out) noexcept;

StatusCode f32_attention(const ArchIr& arch,
                         const soma::F32LayerWeights& lw,
                         const float* x,
                         std::uint32_t n_tokens,
                         soma::F32Workspace& ws,
                         float* out) noexcept;

StatusCode f32_attention_kv(const ArchIr& arch,
                            const soma::F32LayerWeights& lw,
                            const float* x,
                            std::uint32_t n_rows,
                            LayerIndex layer,
                            const soma::KvRow* rows,
                            soma::F32Workspace& ws,
                            float* out) noexcept;

/// Run `n_tokens` consecutive tokens of ONE sequence through a linear layer,
/// advancing `recurrent` and `conv` in place.
///
/// Exposed so a test can drive a layer without a model, and so prefill and
/// decode provably share one implementation: they differ only in where the
/// state comes from, and a second copy of the recurrence is a second thing to
/// keep in agreement.
StatusCode f32_linear_layer(const ArchIr& arch,
                            const F32HybridWeights& w,
                            const float* x,
                            std::uint32_t n_tokens,
                            float* recurrent,
                            float* conv,
                            float* out) noexcept;

const soma::F32Backend& f32_backend() noexcept;

} // namespace soma::arch::kda
