#pragma once

// Soma — SIMD quantized dot products.
//
// Promoted into G3 by measurement, not by ambition. The batch union cut expert
// reads 35x on the real checkpoint and bought almost no wall-clock, because
// `scaling_g3` showed reads were only ~7% of runtime: the quantized matvec was
// scalar, and G1's autotuner had already recorded fp32 running 2.2-2.4x FASTER
// than q4_g — backwards for a format that moves a quarter of the bytes. Until
// that inverts, every concurrency measurement measures the kernels instead of
// the design, so the G3 throughput gate cannot be evaluated at all.
//
// Dispatch is at RUNTIME, off CPUID, not at compile time. Two reasons: the
// binary has to run on hosts without AVX2, and aarch64 is a stated target where
// this TU is not compiled at all. Callers never see the choice.
//
// Deliberately NOT in the seam. These are numeric primitives over a byte layout
// that `quant_format.hpp` owns; no architecture's name appears here, and none
// should.

#include "soma/quant.hpp"

#include <cstddef>
#include <cstdint>

namespace soma::simd {

/// Which instruction set the kernels will actually use, decided once at startup.
///
/// ── The determinism boundary, stated explicitly ──────────────────────────────
///
/// A 16-wide accumulator sums the same products in a different order than an
/// 8-wide one, so the engine's output differs BETWEEN TIERS. That was already
/// true of Avx2 vs Scalar; adding Avx512 does not introduce the property, but it
/// makes it worth naming:
///
///   * WITHIN a host, output is bit-identical across runs, thread counts, and
///     batch composition. That is what `determinism: strict` promises and what
///     tests/soma/threading_g3.cpp enforces.
///   * ACROSS hosts of different tiers, it is not, and cannot be without giving
///     up vectorisation entirely.
///
/// The consequence to remember: a KV checkpoint or a cached logit captured on one
/// tier is not bit-comparable against another. Conformance gates use tolerances
/// against the oracle rather than cross-host equality, so nothing today depends
/// on the stronger property.
enum class SimdTier {
    Scalar = 0,
    Avx2,
    Avx512,
};

/// Resolved once, from CPUID plus the OS state-save bits.
///
/// The OS check is not paperwork: a CPU can report a feature while the OS
/// declines to preserve the corresponding register state across a context
/// switch, and using it then corrupts state in a way that surfaces as unrelated
/// nondeterminism much later. AVX-512 needs the opmask and both ZMM halves
/// declared, not just YMM.
SimdTier tier() noexcept;

/// True for any vector tier. Kept as the one-line predicate most call sites want.
bool available() noexcept;

/// One row of a quantized weight against x, dequantizing on the fly.
///
/// `group_xsum`, when non-null, supplies the per-group sum of x — required only
/// by the asymmetric formats, where the group contributes `min * sum(x)`. It is
/// independent of the weight row, so hoisting it out of the row loop removes a
/// full pass over x per output element. Pass null to compute it inline.
float qdot_q4g(const std::byte* p,
               std::uint32_t cols,
               std::uint32_t group,
               const float* x,
               const float* group_xsum) noexcept;

float qdot_q6g(const std::byte* p,
               std::uint32_t cols,
               std::uint32_t group,
               const float* x) noexcept;

float qdot_q8_0(const std::byte* p,
                std::uint32_t cols,
                std::uint32_t group,
                const float* x) noexcept;

float qdot_q4_0(const std::byte* p,
                std::uint32_t cols,
                std::uint32_t group,
                const float* x) noexcept;

// ── fp32 ─────────────────────────────────────────────────────────────────────
//
// Landing after the quantized kernels because the quantized kernels moved the
// bottleneck onto these. With q4_g at 34.7 GF/s and fp32 at 10.8, the attention
// projections, the output head, and the O(T^2) attention inner loop became the
// dominant cost — the same measurement that promoted the quant kernels now
// points here.
//
// `dot` and `axpy` matter more than their size suggests: attention runs both
// once per (query, key, head), so they are O(T^2 * heads * head_dim) while the
// projections are only O(T * d_model^2).

float dot(const float* a, const float* b, std::uint32_t n) noexcept;

/// y[m] = sum_k w[m*k + i] * x[i]   — w is [m, k] row-major.
void matvec(const float* w, const float* x, std::uint32_t m, std::uint32_t k, float* y) noexcept;

/// y[i] += alpha * x[i]. The attention value-accumulation step.
void axpy(float alpha, const float* x, std::uint32_t n, float* y) noexcept;

float sumsq(const float* x, std::uint32_t n) noexcept;

/// Max element, or -inf for n == 0.
float vmax(const float* x, std::uint32_t n) noexcept;

/// y[i] = x[i] * s, then returns nothing. Separate from axpy because the
/// in-place scale in softmax and rmsnorm has no accumulate.
void scale(float* x, float s, std::uint32_t n) noexcept;

// ── per-tier implementations ─────────────────────────────────────────────────
//
// The functions above are DISPATCHERS, compiled at the baseline ISA. These are
// the real kernels, each in a translation unit built with its own -march flags
// and entered only after tier() has confirmed the host supports it.
//
// Declared here rather than hidden behind the dispatchers so tests can measure
// and compare tiers directly on a host that has both — the only way to tell a
// tier that is slower than its predecessor from one that is merely present.

namespace avx2 {
float qdot_q4g(const std::byte*, std::uint32_t, std::uint32_t, const float*, const float*) noexcept;
float qdot_q6g(const std::byte*, std::uint32_t, std::uint32_t, const float*) noexcept;
float qdot_q8_0(const std::byte*, std::uint32_t, std::uint32_t, const float*) noexcept;
float qdot_q4_0(const std::byte*, std::uint32_t, std::uint32_t, const float*) noexcept;

float dot(const float*, const float*, std::uint32_t) noexcept;
void matvec(const float*, const float*, std::uint32_t, std::uint32_t, float*) noexcept;
void axpy(float, const float*, std::uint32_t, float*) noexcept;
float sumsq(const float*, std::uint32_t) noexcept;
float vmax(const float*, std::uint32_t) noexcept;
void scale(float*, float, std::uint32_t) noexcept;
} // namespace avx2

namespace avx512 {
float qdot_q4g(const std::byte*, std::uint32_t, std::uint32_t, const float*, const float*) noexcept;
float qdot_q8_0(const std::byte*, std::uint32_t, std::uint32_t, const float*) noexcept;
float qdot_q4_0(const std::byte*, std::uint32_t, std::uint32_t, const float*) noexcept;

// NOTE: no qdot_q6g here, deliberately. 6-bit fields do not divide a byte, and
// the AVX2 version straightens them with a 4x4 float transpose of 128-bit lanes.
// Widening that needs a cross-lane byte permute (AVX512VBMI) — available on the
// target CPU, but it would make the whole AVX-512 tier depend on a narrower
// feature bit for one projection out of three. Q6_G therefore stays on the AVX2
// kernel even at the AVX-512 tier; the dispatcher routes it there.

float dot(const float*, const float*, std::uint32_t) noexcept;
void matvec(const float*, const float*, std::uint32_t, std::uint32_t, float*) noexcept;
void axpy(float, const float*, std::uint32_t, float*) noexcept;
float sumsq(const float*, std::uint32_t) noexcept;
float vmax(const float*, std::uint32_t) noexcept;
void scale(float*, float, std::uint32_t) noexcept;
} // namespace avx512

} // namespace soma::simd
