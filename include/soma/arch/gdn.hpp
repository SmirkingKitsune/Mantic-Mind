#pragma once

// Soma — hybrid linear/full attention backend (GDN + GQA).
//
// The FIFTH family through the seam, and the second hybrid. It exists as its own
// backend rather than as a mode of `arch/kda` because the two hybrids share a
// shape and no arithmetic:
//
//                        MlaKda (Kimi-K3)            GqaGdn (Qwen3.5)
//   full layers          MLA, latent KV              GQA, full K and V planes
//   full-layer gate      own d x (h*v_head) tensor   fused into q_proj
//   rotation             none (NoPE)                 first quarter of each head
//   linear decay         per CHANNEL (diagonal)      per HEAD (scalar)
//   linear head count    one, shared by q/k/v        16 for q/k, 128 for v
//   conv                 three, one per projection   one, over q ++ k ++ v
//
// Reference config this backend is designed against:
//
//   Qwen3.8-2.4T-A95B  92 layers, d_model 8192, 64 heads / 4 kv-heads, head_dim 256
//                      23 full-attention layers (0-based 3, 7, … 91)
//                      69 GDN layers: 16 key heads x 128, 128 value heads x 128,
//                                     short conv 4, no conv bias
//                      attn_output_gate, partial_rotary_factor 0.25 -> 64 of 256
//                      rope_theta 1e7, qk-norm per head
//                      512 routed experts, top-10, 1 GATED shared expert,
//                      moe_intermediate 2048, all 92 layers MoE
//                      softmax router, top-k renormalized unconditionally
//                      max_position_embeddings 262144
//                      2.446 T parameters; 4.89 TB of bf16 on disk
//
// WHY THE SPLIT IS THE WHOLE POINT, in this model's own numbers. A full layer
// caches `2 * 4 * 256` floats per token, so 23 of them want 184 KiB/token; the
// 69 linear layers want 568.2 MiB between them and not one byte more however
// long the context runs. Sizing all 92 layers as full attention asks for
// 736 KiB/token — at the stated 262144 context, 184.0 GiB against the real
// 46.55 GiB, a factor of 3.95 on the quantity the verdict turns on.
//
// The corollary matters as much and points the other way: below 1054 tokens
// this stack wants MORE per-sequence memory than an all-full-attention stack of
// the same shape, because the constant term has not yet been paid for. That
// crossover is exact, not approximate — 595771392 / (69 * 8192).
// `kv_bytes_for_context` is genuinely affine, and a planner that treats
// `kv_bytes_per_token * context` as the truth is wrong at both ends — optimistic
// when short, pessimistic when long.
//
// DEPENDENCY RULE: this header may include core headers. Core headers may not
// include this one, and may not mention "gdn". CI enforces it.

#include "soma/arch/gqa.hpp"
#include "soma/arch_ir.hpp"
#include "soma/attention_backend.hpp"
#include "soma/f32_model.hpp"
#include "soma/kv_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace soma::arch::gdn {

/// This backend's KV checkpoint tag. Owned here, not in a core enum.
///
/// Distinct from `gqa`'s and from `kda`'s, and neither is a near miss. A
/// checkpoint of this family holds 23 layers of full K/V interleaved with 69
/// recurrent states; a `gqa` checkpoint of a model with the same head counts
/// holds 92 layers of K/V and nothing recurrent. Same family name in prose,
/// different bytes, and replaying one as the other resumes a conversation the
/// cache does not describe — fluently, with nothing to detect it.
inline constexpr KvFormatId kKvFormat = kv_format_id("soma.kv.gqa.gdn.hybrid.v1");

// ── the opaque cache ─────────────────────────────────────────────────────────
//
// This family's cache cannot be a pair of planes. 23 layers want K and V that
// grow per token; 69 want a fixed matrix and a convolution window that do not.
// `KvCache` already supports an opaque per-family buffer and selects it exactly
// when a backend supplies `kv_bytes_for_context` — so the byte count below IS
// the allocation, and the layout below is the only thing allowed to read it.

/// Byte offsets of one layer's regions within the opaque buffer.
///
/// A full-attention layer uses `k` and `v`; a linear layer uses `recurrent` and
/// `conv`. The unused offsets equal `end`, so touching one is a zero-length
/// access rather than an alias onto a neighbour.
struct LayerRegion {
    std::size_t k = 0;         ///< [ctx][n_kv_heads * head_dim] f32
    std::size_t v = 0;         ///< [ctx][n_kv_heads * head_dim] f32
    std::size_t recurrent = 0; ///< [n_v_heads][head_k_dim][head_v_dim] f32
    std::size_t conv = 0;      ///< [conv_width][conv_kernel - 1] f32
    std::size_t end = 0;
};

/// Where `layer`'s regions sit. Walks every preceding layer, because the stride
/// is not uniform — that is the point of the family.
LayerRegion layer_region(const ArchIr& arch, std::uint32_t layer, std::uint32_t context) noexcept;

/// Per-sequence state at a given context: the affine one.
///
/// Derived from `layer_region`, NOT computed alongside it. A byte count and a
/// layout that are separately maintained will eventually disagree, and the
/// symptom is an out-of-bounds write into the next layer's state — which reads
/// as a model that degrades with depth.
std::uint64_t kv_bytes_for_context(const ArchIr& arch, std::uint32_t context) noexcept;

/// Growth rate only — the full-attention layers' contribution per token.
///
/// The constant term is deliberately NOT folded in: this function has no context
/// to amortise it over, and picking one is how an estimate becomes confidently
/// wrong. Callers that need the truth call `kv_bytes_for_context`.
std::size_t kv_bytes_per_token(const ArchIr& arch) noexcept;

/// Per-sequence bytes that do NOT grow with context: the recurrent states and
/// their convolution windows, plus the layout's alignment padding.
std::uint64_t recurrent_state_bytes(const ArchIr& arch) noexcept;

const AttentionBackend& attention_backend() noexcept;

// ── the kernel ───────────────────────────────────────────────────────────────
//
// Gated DeltaNet. Reference semantics, transcribed from
// `torch_recurrent_gated_delta_rule` in transformers' `modeling_qwen3_5_moe.py`
// with the flag the model actually passes (`use_qk_l2norm_in_kernel=True`):
//
//     q, k  <- l2norm per head (eps 1e-6);  q *= head_k_dim ** -0.5
//     beta  <- sigmoid(b_proj(x))                       per VALUE head
//     g     <- -exp(A_log) * softplus(a_proj(x) + dt_bias)   per VALUE head
//     S     <- S * exp(g)                    ONE SCALAR, whole matrix
//     S     <- S + k (x) (v - S^T k) * beta  delta rule against the DECAYED state
//     o     <- S^T q                         from the UPDATED state
//
// Every one of those orderings is load-bearing and none is guessable: predicting
// from the pre-decay state, or reading the output before the update, produces
// finite plausible numbers and a different model.
//
// The one difference from `arch::kda`'s otherwise identical-looking recurrence
// is the third line. There the decay is a VECTOR applied along the key axis;
// here it is a scalar multiplying the whole state. Reusing that kernel with a
// broadcast vector would compute the same thing and would also quietly accept a
// per-channel gate this family cannot produce, so the two stay separate.

/// Log-space decay for one token. `a_raw`, `out`: [n_v_heads]. `a_log`,
/// `dt_bias`: [n_v_heads] — this family is per HEAD throughout, unlike KDA
/// whose `dt_bias` is per channel.
void gate(const ArchIr& arch,
          const float* a_log,
          const float* dt_bias,
          const float* a_raw,
          float* out) noexcept;

/// Causal depthwise short convolution over ONE token, then SiLU.
///
/// One convolution spanning q ++ k ++ v — `GdnSpec::conv_width()` channels.
/// `state` carries the previous `kernel - 1` inputs per channel, oldest first,
/// and is advanced in place. `weight` is [width][kernel] with
/// `weight[c][kernel-1]` multiplying the CURRENT token, which is PyTorch's
/// conv1d ordering. This family's convolution has no bias — `nn.Conv1d(...,
/// bias=False)` — so there is no bias parameter to forget.
void short_conv(std::uint32_t width,
                std::uint32_t kernel,
                const float* weight,
                const float* x,
                float* state,
                float* out) noexcept;

/// One recurrent step. `state` is [n_v_heads][head_k_dim][head_v_dim],
/// row-major with the KEY axis outer; it is read and written.
///
/// `q` and `k` are [n_k_heads * head_k_dim] and are broadcast to the value head
/// count HERE, so callers pass the un-repeated projections. `v` is
/// [n_v_heads * head_v_dim]; `g` and `beta_raw` are [n_v_heads] with `beta_raw`
/// PRE-sigmoid; `out` is [n_v_heads * head_v_dim]. `scratch` is [head_v_dim],
/// caller-owned.
///
/// `q` and `k` arrive un-normalized: the L2 and the `head_k_dim ** -0.5` scale
/// happen here, because they are part of the operator rather than of the
/// projection.
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

/// RMSNorm per VALUE head, then weight, then a SiLU gate — in that order.
///
/// The order is the whole content of this function. Gating BEFORE the norm (as
/// Mamba's `RMSNormGated` does) is a different operator, and both are plausible
/// readings of the name. `Qwen3_5MoeRMSNormGated` normalizes first and says so
/// in a comment; it also uses SiLU rather than the plain sigmoid that gates this
/// family's FULL-attention layers, which is a second thing not to assume.
void gated_rmsnorm(const ArchIr& arch,
                   const float* x,
                   const float* z,
                   const float* weight,
                   float eps,
                   float* out) noexcept;

// ── the F32 execution path ───────────────────────────────────────────────────
//
// One backend for a stack of two layer kinds. The dispatch cannot live in the
// core: `F32Backend::attention` takes no layer index, so the only place that
// knows whether a given layer is Full or Linear is the payload bound to it.
//
// Full layers are GQA layers and are DELEGATED to the GQA backend rather than
// reimplemented beside it — including its fused output gate and partial
// rotation, which live there because they are variants OF GQA and the next
// family to want one should find them already done.

/// One layer's attention weights: whichever kind this layer is.
struct F32HybridWeights {
    bool linear = false;

    /// Full layers. A COPY of what `gqa::f32_bind_layer` produced — every member
    /// is a non-owning view, so copying costs nothing and avoids a second
    /// transcription of GQA's tensor names and its width checks.
    arch::gqa::F32AttnWeights full;

    /// Linear layers.
    soma::WeightRef in_proj_qkv; ///< [2*key_dim + value_dim, d_model], FUSED
    soma::WeightRef in_proj_z;   ///< [value_dim, d_model]
    soma::WeightRef in_proj_b;   ///< [n_v_heads, d_model]
    soma::WeightRef in_proj_a;   ///< [n_v_heads, d_model]
    std::span<const float> conv_w;  ///< [conv_width][conv_kernel]
    std::span<const float> a_log;   ///< [n_v_heads]
    std::span<const float> dt_bias; ///< [n_v_heads]
    std::span<const float> o_norm;  ///< [head_v_dim]
    soma::WeightRef out_proj;       ///< [d_model, value_dim]
};

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

} // namespace soma::arch::gdn
