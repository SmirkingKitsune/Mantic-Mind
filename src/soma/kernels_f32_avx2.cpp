// Soma — AVX2 + FMA fp32 primitives.
//
// Same contract as the quantized SIMD TU: x86-64 only, compiled with /arch:AVX2
// (or -mavx2 -mfma), reached only behind the runtime CPUID check in
// kernels_quant.cpp. The scalar versions in kernels_f32.cpp remain the reference
// and the fallback.
//
// Every reduction here uses FOUR independent accumulators, not one. A single
// accumulator serialises the whole loop on FMA latency (~4 cycles) instead of
// throughput (~2/cycle), which costs roughly 8x on a modern core and is the
// difference between vectorised and merely-using-vector-registers. The partial
// sums are then combined in a FIXED order, so the result is deterministic —
// different from the scalar order, but identical run to run.

#include "soma/kernels_simd.hpp"

#include <immintrin.h>

namespace soma::simd::avx2 {

namespace {

float hsum(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, _mm256_extractf128_ps(v, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

/// The shared body of dot() and one row of matvec().
float dot_impl(const float* a, const float* b, std::uint32_t n) noexcept {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();

    std::uint32_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    for (; i + 8 <= n; i += 8) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    }

    float acc = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; i < n; ++i)
        acc += a[i] * b[i];
    return acc;
}

} // namespace

float dot(const float* a, const float* b, std::uint32_t n) noexcept {
    return dot_impl(a, b, n);
}

void matvec(const float* w, const float* x, std::uint32_t m, std::uint32_t k, float* y) noexcept {
    // Four output rows at a time. Each x load is then used four times instead of
    // once, which matters because the output head is [151936, 2048]: streaming
    // 1.2 GB of weights past the core makes this loop bandwidth-bound, and
    // reusing x is the only lever that does not require touching fewer weights.
    std::uint32_t row = 0;
    for (; row + 4 <= m; row += 4) {
        const float* w0 = w + static_cast<std::size_t>(row) * k;
        const float* w1 = w0 + k;
        const float* w2 = w1 + k;
        const float* w3 = w2 + k;

        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();

        std::uint32_t i = 0;
        for (; i + 8 <= k; i += 8) {
            const __m256 xv = _mm256_loadu_ps(x + i);
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i), xv, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + i), xv, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + i), xv, s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + i), xv, s3);
        }
        float r0 = hsum(s0), r1 = hsum(s1), r2 = hsum(s2), r3 = hsum(s3);
        for (; i < k; ++i) {
            const float xv = x[i];
            r0 += w0[i] * xv;
            r1 += w1[i] * xv;
            r2 += w2[i] * xv;
            r3 += w3[i] * xv;
        }
        y[row] = r0;
        y[row + 1] = r1;
        y[row + 2] = r2;
        y[row + 3] = r3;
    }
    for (; row < m; ++row) {
        y[row] = dot_impl(w + static_cast<std::size_t>(row) * k, x, k);
    }
}

void axpy(float alpha, const float* x, std::uint32_t n, float* y) noexcept {
    const __m256 va = _mm256_set1_ps(alpha);
    std::uint32_t i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(y + i,
                         _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
        _mm256_storeu_ps(
            y + i + 8, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8)));
        _mm256_storeu_ps(
            y + i + 16,
            _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16)));
        _mm256_storeu_ps(
            y + i + 24,
            _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i,
                         _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
    for (; i < n; ++i)
        y[i] += alpha * x[i];
}

float sumsq(const float* x, std::uint32_t n) noexcept {
    return dot_impl(x, x, n);
}

float vmax(const float* x, std::uint32_t n) noexcept {
    if (n == 0) return -3.402823466e38f;

    std::uint32_t i = 0;
    float best = x[0];
    if (n >= 8) {
        __m256 m0 = _mm256_loadu_ps(x);
        i = 8;
        for (; i + 8 <= n; i += 8)
            m0 = _mm256_max_ps(m0, _mm256_loadu_ps(x + i));

        __m128 lo = _mm_max_ps(_mm256_castps256_ps128(m0), _mm256_extractf128_ps(m0, 1));
        lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
        best = _mm_cvtss_f32(lo);
    }
    for (; i < n; ++i) {
        if (x[i] > best) best = x[i];
    }
    return best;
}

void scale(float* x, float s, std::uint32_t n) noexcept {
    const __m256 vs = _mm256_set1_ps(s);
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vs));
    }
    for (; i < n; ++i)
        x[i] *= s;
}

} // namespace soma::simd::avx2
