#pragma once

// Soma — GQA attention backend (also serves MHA as the n_kv_heads == n_heads
// degenerate case, and GqaBsa as the same attention under a block-sparse key
// selector).
//
// Covers most open MoE checkpoints: Qwen3-MoE, Mixtral, GPT-OSS, Llama-family.
// Ships first for exactly that reason.
//
// THREE families, one backend, and the same argument each time. MHA is GQA at a
// repeat factor of 1. BSA is GQA plus a selector that decides which keys the
// softmax may see — every projection, both norms, the rotation and the score
// loop are identical, and only the visible SET changes. Giving either its own
// file would mean two implementations of one arithmetic kept in agreement by
// hand, which is the failure mode this file has already argued against once.
//
// The precedent is exact: `arch::mla` carries DSA the same way, for the same
// reason. See the family branches below, all of which resolve ONCE per layer
// rather than inside a loop.
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
#include "soma/kv_cache.hpp"
#include "soma/types.hpp"

#include <cstddef>

namespace soma::arch::gqa {

/// This backend's KV checkpoint tag. Owned here, not in a core enum.
///
/// ONE tag for all three families this backend serves, even though a BSA cache
/// is genuinely wider — its K plane carries the indexer key alongside K. That is
/// safe because `format_id` is the COARSE gate and `arch_hash` is the fine one:
/// `compute_arch_hash` emits a `|bsa=` term covering the indexer's dimensions
/// and per-layer map, and `KvCheckpointStore::load` refuses on an arch_hash
/// mismatch before it ever looks at the geometry. A GQA checkpoint therefore
/// cannot be replayed into a BSA cache, or the reverse.
///
/// Same arrangement `arch::mla` uses for MLA and MLA+DSA, whose V planes differ
/// in exactly the same way.
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

    /// The router's per-expert selection bias, when the family has one. Empty
    /// otherwise, which is how `f32_route` decides whether to apply it.
    ///
    /// It lives in the ATTENTION payload because that is the only per-layer
    /// storage a backend owns — `F32Backend::route` is handed `F32LayerWeights`
    /// precisely so a router with parameters beyond the gate matrix can reach
    /// them. Not an attention tensor by any reading, and named plainly so nobody
    /// mistakes it for one.
    std::span<const float> e_score_bias;

    /// The block-sparse indexer. Present only on layers the IR marks `Full`.
    ///
    /// Its key projection produces ONE head, not `n_index_heads`: every indexer
    /// head scores its own query against the same shared key. That asymmetry is
    /// the whole reason this is cheap, and reading `k_proj` at the query width
    /// would both fail the shape check and, if it did not, cache four times the
    /// bytes per token.
    struct Indexer {
        soma::WeightRef q_proj; ///< [n_index_heads * index_head_dim, d_model]
        soma::WeightRef k_proj; ///< [index_head_dim, d_model] — one head
        std::span<const float> q_norm, k_norm; ///< index_head_dim each
    } idx;

    /// Resolved at bind, so the hot path never re-derives it from layer_kinds.
    bool has_indexer = false;
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

/// Both cache planes, in floats per position per layer.
///
/// Non-null for this backend — where it used to be null, meaning "the core's GQA
/// default". The default is still what plain GQA and MHA get; BSA widens the K
/// plane by `index_head_dim` so the indexer's key can be cached beside K.
///
/// UNIFORM across layers, because `KvCache` allocates one geometry for the whole
/// stack and cannot do otherwise. On MiniMax-M3 that means the three
/// non-indexed leading layers own an indexer slot they never write — 3 of 60
/// layers at 128 of 1152 floats, i.e. 0.6% of the cache. Reported as allocated
/// rather than as needed, because the planner's job is to predict the
/// allocation, and a figure 0.6% below what `KvCache::open` actually reserves
/// would be optimistic on exactly the quantity the verdict turns on.
soma::KvGeometry f32_kv_geometry(const ArchIr& arch) noexcept;

/// Exact attention-owned resident bytes for the whole stack.
///
/// Needed because BSA's layers are NOT uniform: the indexed ones own a query
/// projection, a single-head key projection and two norms that the others do
/// not, so no per-layer average is exact. Returns
/// `n_layers * weight_bytes_per_layer` verbatim for GQA and MHA, which have no
/// indexer — the two paths agree to the byte on every family that predates this
/// one.
std::uint64_t resident_weight_bytes(const ArchIr& arch,
                                    AttentionBackend::ByteSizer sizer) noexcept;

const soma::F32Backend& f32_backend() noexcept;

} // namespace soma::arch::gqa
