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

#include "soma/arch_backend.hpp"
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

/// No-op. GQA has no weight-absorption analogue.
///
/// The hook exists in AttentionBackend because MLA needs it; implementing it as
/// a no-op here is the honest way to say so. An interface without the hook would
/// force MLA's absorption into the loader, which is where it would leak into the
/// core.
Status prepare_weights(ModelState& model);

StatusCode prefill(const ModelState& model,
                   ExecScratch& exec,
                   SeqBatch& batch,
                   LayerIndex layer,
                   std::span<const float> in,
                   std::span<float> out) noexcept;

/// Repeat-KV expansion then standard SDPA, per sequence.
///
/// v1 loops sequences inside this call: each attends its own KV region at its
/// own length. The signature takes the whole batch so a paged-KV block table can
/// batch the loop away later without touching AttentionBackend.
StatusCode decode(const ModelState& model,
                  ExecScratch& exec,
                  SeqBatch& batch,
                  LayerIndex layer,
                  std::span<const float> in,
                  std::span<float> out) noexcept;

StatusCode init_kv_region(const ArchIr& arch, KvRegion& region) noexcept;

/// Sliding-window span for a layer, or 0 for full attention.
/// Qwen3-30B-A3B sets sliding_window null; the path exists for families that
/// alternate windowed and full layers.
std::uint32_t window_span(const ArchIr& arch, LayerIndex layer) noexcept;

const AttentionBackend& attention_backend() noexcept;

// ── Router ───────────────────────────────────────────────────────────────────

/// Softmax scoring, optional post-selection renormalization (norm_topk_prob),
/// no grouping, no bias correction.
///
/// Kept separate from the MLA-family router even though both are "softmax
/// top-k": group-limited routing and pre-top-k bias correction are not flags
/// that can be switched off cleanly — they change the selection algorithm, and a
/// single parameterized router would be a chain of branches in the one function
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

StatusCode dense_ffn(const ModelState& model,
                     ExecScratch& exec,
                     LayerIndex layer,
                     std::uint32_t n_rows) noexcept;

/// Qwen3-MoE and Mixtral both have n_shared_experts == 0, so this is a no-op for
/// both reference configs. Declared because GPT-OSS-family GQA MoE does use
/// shared experts, and discovering that after the fact would mean a second
/// backend rather than a filled-in function.
StatusCode shared_experts(const ModelState& model,
                          ExecScratch& exec,
                          LayerIndex layer,
                          std::uint32_t n_rows) noexcept;

StatusCode apply_norm(const ModelState& model,
                      ExecScratch& exec,
                      const DenseTensor& weight,
                      std::uint32_t n_rows) noexcept;

StatusCode apply_rope(const ArchIr& arch,
                      std::span<float> q,
                      std::span<float> k,
                      std::span<const std::uint32_t> positions) noexcept;

/// Admission-time gate. Rejects before conversion spends hours.
Status validate(const ArchIr& arch);

const ArchBackend& backend() noexcept;

// ── G0 fp32 reference path ───────────────────────────────────────────────────
//
// Named distinctly from the streaming-era entry points above: those take
// ExecScratch/SeqBatch and arrive with the scheduler at G3, while these operate
// on a single sequence with no cache and are what the conformance oracle is
// compared against today. Both will coexist — the fp32 path stays as the
// reference the quantized kernels are validated against.

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
                     const float* logits,
                     std::uint32_t n_tokens,
                     std::uint32_t* out_ids,
                     float* out_weights) noexcept;

const soma::F32Backend& f32_backend() noexcept;

} // namespace soma::arch::gqa
