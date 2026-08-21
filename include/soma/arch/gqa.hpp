#pragma once

// Soma — GQA attention backend (also serves MHA as the n_kv_heads == n_heads
// degenerate case).
//
// Covers most open MoE checkpoints: Qwen3-MoE, Mixtral, GPT-OSS, Llama-family.
// Ships first for exactly that reason.
//
// Reference configs this backend is designed against:
//
//   Qwen3-30B-A3B    48 layers, d_model 2048, 32 heads / 4 kv-heads, head_dim 128
//                    qk_norm, rope_theta 1e6, no scaling, no sliding window
//                    128 experts, top-8, moe_intermediate 768, norm_topk_prob
//                    all layers MoE (decoder_sparse_step 1, mlp_only_layers [])
//
//   Mixtral-8x7B     32 layers, d_model 4096, 32 heads / 8 kv-heads, head_dim 128
//                    8 experts, top-2, intermediate 14336
//                    -> 88 MB experts at q4, 25% active fraction: the deliberate
//                       resident-only fixture. This backend must represent it
//                       FAITHFULLY so admission can look at it and say no.
//
// DEPENDENCY RULE: this header may include core headers. Core headers may not
// include this one, and may not mention "gqa". CI enforces it.

#include "soma/arch_ir.hpp"
#include "soma/attention_backend.hpp"
#include "soma/f32_model.hpp"
#include "soma/types.hpp"

#include <cstddef>

namespace soma::arch::gqa {

/// This backend's KV checkpoint tag. Owned here, not in a core enum.
inline constexpr KvFormatId kKvFormat = kv_format_id("soma.kv.gqa.full_kv.v1");

/// Full K and V per token: 2 × n_kv_heads × head_dim elements per layer.
///
///   Qwen3-30B-A3B  2 × 4 × 128 = 1024 elem/tok/layer × 48 = 98 KB/tok @fp16
///   Mixtral-8x7B   2 × 8 × 128 = 2048 elem/tok/layer × 32 = 131 KB/tok @fp16
///
/// At 32k context that is 3.2 GB and 4.3 GB respectively — of the same RAM the
/// expert cache wants. This is why the planner models the competition explicitly
/// instead of sizing the expert cache first and letting KV take the remainder:
/// the latter thrashes on long contexts and presents as an unrelated bug.
std::size_t kv_bytes_per_token(const ArchIr& arch) noexcept;

/// Sliding-window span for a layer, or 0 for full attention.
/// Qwen3-30B-A3B sets sliding_window null; the path exists for families that
/// alternate windowed and full layers.
std::uint32_t window_span(const ArchIr& arch, LayerIndex layer) noexcept;

const AttentionBackend& attention_backend() noexcept;

// ── F32-activation execution path ────────────────────────────────────────────
//
// The `f32_` prefix is historical and is now the only execution path there is.
// It once named the reference half of a planned pair, the other being
// ExecScratch/SeqBatch entry points that no family ever implemented and that have
// since been deleted. What survives runs quantized SIMD kernels against streamed
// experts — "fp32" describes its ACTIVATIONS, not its weights.

/// GQA's per-layer attention weights. Lives HERE, not in F32LayerWeights.
///
/// The core used to hold these fields directly, which made "a layer's attention
/// weights" mean "a GQA layer's attention weights" everywhere. MLA has none of
/// them.
struct F32AttnWeights {
    soma::WeightRef q_proj, k_proj, v_proj, o_proj;
    std::span<const float> q_norm, k_norm; ///< empty when QkNormKind::None
};

StatusCode f32_bind_layer(const ArchIr& arch,
                          const soma::LayerBindCtx& ctx,
                          soma::ArchLayerPayload& out) noexcept;

StatusCode f32_attention(const ArchIr& arch,
                         const soma::F32LayerWeights& w,
                         const float* x,
                         std::uint32_t n_tokens,
                         soma::F32Workspace& ws,
                         float* out) noexcept;

/// The batched-decode variant: every row brings its own KV cache, position and
/// visible length, so rows from different sequences batch with no special case.
StatusCode f32_attention_kv(const ArchIr& arch,
                            const soma::F32LayerWeights& w,
                            const float* x,
                            std::uint32_t n_rows,
                            LayerIndex layer,
                            const soma::KvRow* rows,
                            soma::F32Workspace& ws,
                            float* out) noexcept;

StatusCode f32_route(const ArchIr& arch,
                     const soma::F32LayerWeights& lw,
                     const TokenId* input_tokens,
                     const float* logits,
                     std::uint32_t n_tokens,
                     std::uint32_t* out_ids,
                     float* out_weights) noexcept;

const soma::F32Backend& f32_backend() noexcept;

} // namespace soma::arch::gqa
