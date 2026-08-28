// Soma — AVX-512 kernels, quantized and fp32.
//
// The third dispatch tier. Reached only after tier() == Avx512, which checks the
// feature bits AND that the OS is preserving opmask + both ZMM halves.
//
// Worth doing here specifically because the target is Zen 5, which has a NATIVE
// 512-bit datapath: two 512-bit FMAs per cycle, no frequency penalty, no
// double-pumping. On the earlier Intel parts that gave AVX-512 its reputation,
// wide code could clock the core down far enough to lose to AVX2 — which is why
// the tier is measured against AVX2 in tests/soma/simd_g3.cpp rather than
// assumed faster.
//
// Q6_G is deliberately absent; see the note in kernels_simd.hpp. Its 6-bit
// fields need a cross-lane byte permute to widen, which would make this whole
// tier depend on AVX512VBMI for one projection out of three.

#include "soma/kernels_simd.hpp"
#include "soma/quant_format.hpp"

#include <cstring>
#include <immintrin.h>

namespace soma::simd::avx512 {

namespace {

float load_f32(const std::byte* p) noexcept {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/// 16 packed bytes -> 32 nibble levels, as two 16-wide float vectors.
///
/// The nibble de-interleave stays on 128-bit lanes on purpose. `unpacklo_epi8`
/// at 256 or 512 bits operates WITHIN each 128-bit lane, so the wide version
/// does not produce sequential bytes and would silently permute the weight row.
/// Reusing the proven 128-bit step and widening afterwards costs nothing:
/// `_mm512_cvtepu8_epi32` takes all 16 bytes of an __m128i in one instruction.
void unpack_nibbles(__m128i v, __m512& l0, __m512& l1) noexcept {
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(v, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);

    l0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi))); // 0..15
    l1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(lo, hi))); // 16..31
}

} // namespace

// ── Q4_G — asymmetric, w = min + scale*level ─────────────────────────────────
float qdot_q4g(const std::byte* p,
               std::uint32_t cols,
               std::uint32_t g,
               const float* x,
               const float* group_xsum) noexcept {
    const auto gb = group_bytes(DType::Q4_G, g);
    const bool need_xsum = (group_xsum == nullptr);

    float acc = 0.0f;
    std::uint32_t gi = 0;
    for (std::uint32_t c = 0; c < cols; c += g, ++gi) {
        const float scale = load_f32(p);
        const float minv = load_f32(p + 4);
        const float* xv = x + c;

        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 xs0 = _mm512_setzero_ps(), xs1 = _mm512_setzero_ps();

        std::uint32_t i = 0;
        for (; i + 32 <= g; i += 32) {
            __m512 l0, l1;
            unpack_nibbles(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8 + i / 2)), l0, l1);

            const __m512 x0 = _mm512_loadu_ps(xv + i);
            const __m512 x1 = _mm512_loadu_ps(xv + i + 16);

            a0 = _mm512_fmadd_ps(l0, x0, a0);
            a1 = _mm512_fmadd_ps(l1, x1, a1);
            if (need_xsum) {
                xs0 = _mm512_add_ps(xs0, x0);
                xs1 = _mm512_add_ps(xs1, x1);
            }
        }

        float part = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
        float xsum = need_xsum ? _mm512_reduce_add_ps(_mm512_add_ps(xs0, xs1)) : group_xsum[gi];

        for (; i < g; i += 2) {
            const auto byte = static_cast<std::uint8_t>(p[8 + i / 2]);
            part += static_cast<float>(byte & 0x0F) * xv[i];
            if (need_xsum) xsum += xv[i];
            if (i + 1 < g) {
                part += static_cast<float>(byte >> 4) * xv[i + 1];
                if (need_xsum) xsum += xv[i + 1];
            }
        }

        acc += scale * part + minv * xsum;
        p += gb;
    }
    return acc;
}

// ── Q4_0 — symmetric, level biased by 8 ──────────────────────────────────────
float qdot_q4_0(const std::byte* p, std::uint32_t cols, std::uint32_t g, const float* x) noexcept {
    const auto gb = group_bytes(DType::Q4_0, g);
    const __m512 bias = _mm512_set1_ps(8.0f);

    float acc = 0.0f;
    for (std::uint32_t c = 0; c < cols; c += g) {
        const float scale = load_f32(p);
        const float* xv = x + c;

        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        std::uint32_t i = 0;
        for (; i + 32 <= g; i += 32) {
            __m512 l0, l1;
            unpack_nibbles(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 4 + i / 2)), l0, l1);
            a0 = _mm512_fmadd_ps(_mm512_sub_ps(l0, bias), _mm512_loadu_ps(xv + i), a0);
            a1 = _mm512_fmadd_ps(_mm512_sub_ps(l1, bias), _mm512_loadu_ps(xv + i + 16), a1);
        }

        float part = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
        for (; i < g; i += 2) {
            const auto byte = static_cast<std::uint8_t>(p[4 + i / 2]);
            part += static_cast<float>(static_cast<int>(byte & 0x0F) - 8) * xv[i];
            if (i + 1 < g) {
                part += static_cast<float>(static_cast<int>(byte >> 4) - 8) * xv[i + 1];
            }
        }
        acc += scale * part;
        p += gb;
    }
    return acc;
}

// ── Q8_0 ─────────────────────────────────────────────────────────────────────
float qdot_q8_0(const std::byte* p, std::uint32_t cols, std::uint32_t g, const float* x) noexcept {
    const auto gb = group_bytes(DType::Q8_0, g);

    float acc = 0.0f;
    for (std::uint32_t c = 0; c < cols; c += g) {
        const float scale = load_f32(p);
        const float* xv = x + c;

        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        std::uint32_t i = 0;
        for (; i + 32 <= g; i += 32) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 4 + i));
            const __m512 l0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(v)));
            const __m512 l1 =
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(v, 1)));
            a0 = _mm512_fmadd_ps(l0, _mm512_loadu_ps(xv + i), a0);
            a1 = _mm512_fmadd_ps(l1, _mm512_loadu_ps(xv + i + 16), a1);
        }

        float part = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
        for (; i < g; ++i) {
            part += static_cast<float>(static_cast<std::int8_t>(p[4 + i])) * xv[i];
        }
        acc += scale * part;
        p += gb;
    }
    return acc;
}

// ── fp32 ─────────────────────────────────────────────────────────────────────

namespace {

float dot_impl(const float* a, const float* b, std::uint32_t n) noexcept {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();

    std::uint32_t i = 0;
    for (; i + 64 <= n; i += 64) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), a3);
    }
    for (; i + 16 <= n; i += 16) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), a0);
    }

    float acc = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));

    // Masked tail rather than a scalar loop: one instruction instead of up to
    // fifteen, and it keeps the remainder in the same accumulation width as the
    // body — a scalar tail would give a different answer for the same inputs
    // depending only on whether n happened to be a multiple of 16.
    if (i < n) {
        const auto rem = static_cast<std::uint32_t>(n - i);
        const __mmask16 m = static_cast<__mmask16>((1u << rem) - 1u);
        const __m512 va = _mm512_maskz_loadu_ps(m, a + i);
        const __m512 vb = _mm512_maskz_loadu_ps(m, b + i);
        acc += _mm512_reduce_add_ps(_mm512_mul_ps(va, vb));
    }
    return acc;
}

} // namespace

float dot(const float* a, const float* b, std::uint32_t n) noexcept {
    return dot_impl(a, b, n);
}

void matvec(const float* w, const float* x, std::uint32_t m, std::uint32_t k, float* y) noexcept {
    std::uint32_t row = 0;
    for (; row + 4 <= m; row += 4) {
        const float* w0 = w + static_cast<std::size_t>(row) * k;
        const float* w1 = w0 + k;
        const float* w2 = w1 + k;
        const float* w3 = w2 + k;

        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();

        std::uint32_t i = 0;
        for (; i + 16 <= k; i += 16) {
            const __m512 xv = _mm512_loadu_ps(x + i);
            s0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + i), xv, s0);
            s1 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + i), xv, s1);
            s2 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + i), xv, s2);
            s3 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + i), xv, s3);
        }
        float r0 = _mm512_reduce_add_ps(s0), r1 = _mm512_reduce_add_ps(s1);
        float r2 = _mm512_reduce_add_ps(s2), r3 = _mm512_reduce_add_ps(s3);

        if (i < k) {
            const auto rem = static_cast<std::uint32_t>(k - i);
            const __mmask16 mk = static_cast<__mmask16>((1u << rem) - 1u);
            const __m512 xv = _mm512_maskz_loadu_ps(mk, x + i);
            r0 += _mm512_reduce_add_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(mk, w0 + i), xv));
            r1 += _mm512_reduce_add_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(mk, w1 + i), xv));
            r2 += _mm512_reduce_add_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(mk, w2 + i), xv));
            r3 += _mm512_reduce_add_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(mk, w3 + i), xv));
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
    const __m512 va = _mm512_set1_ps(alpha);
    std::uint32_t i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm512_storeu_ps(y + i,
                         _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
        _mm512_storeu_ps(
            y + i + 16,
            _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(y + i + 16)));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(y + i,
                         _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    }
    if (i < n) {
        const __mmask16 m = static_cast<__mmask16>((1u << (n - i)) - 1u);
        _mm512_mask_storeu_ps(
            y + i,
            m,
            _mm512_fmadd_ps(va, _mm512_maskz_loadu_ps(m, x + i), _mm512_maskz_loadu_ps(m, y + i)));
    }
}

float sumsq(const float* x, std::uint32_t n) noexcept {
    return dot_impl(x, x, n);
}

float vmax(const float* x, std::uint32_t n) noexcept {
    if (n == 0) return -3.402823466e38f;
    std::uint32_t i = 0;
    float best = x[0];
    if (n >= 16) {
        __m512 m0 = _mm512_loadu_ps(x);
        i = 16;
        for (; i + 16 <= n; i += 16)
            m0 = _mm512_max_ps(m0, _mm512_loadu_ps(x + i));
        best = _mm512_reduce_max_ps(m0);
    }
    for (; i < n; ++i) {
        if (x[i] > best) best = x[i];
    }
    return best;
}

void scale(float* x, float s, std::uint32_t n) noexcept {
    const __m512 vs = _mm512_set1_ps(s);
    std::uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), vs));
    }
    if (i < n) {
        const __mmask16 m = static_cast<__mmask16>((1u << (n - i)) - 1u);
        _mm512_mask_storeu_ps(x + i, m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, x + i), vs));
    }
}

} // namespace soma::simd::avx512
