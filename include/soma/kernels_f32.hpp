#pragma once

// Soma — fp32 reference kernels.
//
// The G0 path. Correctness first, and deliberately so: these are the numbers
// every later gate is checked against, so a clever kernel that is subtly wrong
// here poisons the whole ladder. Autotuned quantized kernels arrive at G1 and
// are validated against THIS.
//
// Weight layout matches safetensors / torch.nn.Linear: W is [out, in],
// row-major, so y = W @ x reads a contiguous row per output.

#include "soma/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace soma::f32 {

/// y[m] = sum_k W[m*K + k] * x[k]      W:[M,K]  x:[K]  y:[M]
void matvec(std::span<const float> w,
            std::span<const float> x,
            std::uint32_t m,
            std::uint32_t k,
            std::span<float> y) noexcept;

/// Y[t,m] = sum_k W[m*K + k] * X[t*K + k]   over `rows` rows of X.
void matmul(std::span<const float> w,
            std::span<const float> x,
            std::uint32_t rows,
            std::uint32_t m,
            std::uint32_t k,
            std::span<float> y) noexcept;

/// x * rsqrt(mean(x^2) + eps) * weight, in place over `n` elements.
///
/// Matches HF's RMSNorm: the mean is taken in fp32 and the weight is applied
/// after the reciprocal square root, not folded into it.
void rmsnorm(std::span<float> x,
             std::span<const float> weight,
             std::uint32_t n,
             float eps) noexcept;

void rmsnorm_into(std::span<const float> x,
                  std::span<const float> weight,
                  std::uint32_t n,
                  float eps,
                  std::span<float> out) noexcept;

/// In-place softmax over `n`, max-subtracted.
void softmax(std::span<float> x, std::uint32_t n) noexcept;

/// SwiGLU: out = silu(gate) * up, elementwise over `n`.
void swiglu(std::span<const float> gate,
            std::span<const float> up,
            std::uint32_t n,
            std::span<float> out) noexcept;

/// GeGLU (tanh approximation, matching HF's gelu_pytorch_tanh).
void geglu(std::span<const float> gate,
           std::span<const float> up,
           std::uint32_t n,
           std::span<float> out) noexcept;

/// ReLU^2: out = relu(gate)^2 * up.
void relu2_glu(std::span<const float> gate,
               std::span<const float> up,
               std::uint32_t n,
               std::span<float> out) noexcept;

/// SiTU: out = beta*tanh(gate/beta)*sigmoid(gate) * u, where u is `up` itself
/// when `linear_beta` is zero and `linear_beta*tanh(up/linear_beta)` otherwise.
///
/// A saturating SwiGLU: as beta grows, `beta*tanh(g/beta) -> g` and the gate half
/// becomes exactly SiLU, so SwiGLU is its limit rather than an approximation of
/// it. The parameters are model identity, not tuning knobs, and live in FfnSpec.
///
/// `linear_beta == 0` means the linear half is NOT transformed. It does not mean
/// "beta zero", which would collapse `0*tanh(up/0)` and silence the FFN.
void situ_glu(std::span<const float> gate,
              std::span<const float> up,
              std::uint32_t n,
              float beta,
              float linear_beta,
              std::span<float> out) noexcept;

/// NeoX-style rotary embedding — the "rotate half" form HF uses.
///
/// head_dim is split in half: pairs are (i, i + head_dim/2), NOT adjacent
/// elements. The interleaved variant exists in other stacks and produces
/// plausible-but-wrong output if substituted, so the two are never conflated
/// here; `RopeConfig::interleaved` selects.
void rope_neox(std::span<float> vec,
               std::uint32_t n_heads,
               std::uint32_t head_dim,
               std::uint32_t position,
               float theta,
               std::uint32_t rotary_dim) noexcept;

void rope_interleaved(std::span<float> vec,
                      std::uint32_t n_heads,
                      std::uint32_t head_dim,
                      std::uint32_t position,
                      float theta,
                      std::uint32_t rotary_dim) noexcept;

/// Descending top-k by value. Stable on ties by ascending index, which matches
/// torch.topk and matters because a tie broken differently changes WHICH EXPERT
/// FIRES — a semantic divergence, not a numeric one.
void top_k(std::span<const float> values,
           std::uint32_t n,
           std::uint32_t k,
           std::span<std::uint32_t> out_indices,
           std::span<float> out_values) noexcept;

float dot(std::span<const float> a, std::span<const float> b, std::uint32_t n) noexcept;

/// y[i] += alpha * x[i].
///
/// Small enough to look like it belongs inline at its call site, and it did live
/// there — but attention runs it once per (query, key, head), which makes it
/// O(T^2 * heads * head_dim) and one of the two hottest loops in the engine.
/// Routing it through here is what lets the SIMD dispatch live in one place.
void axpy(float alpha, std::span<const float> x, std::uint32_t n, std::span<float> y) noexcept;

} // namespace soma::f32
