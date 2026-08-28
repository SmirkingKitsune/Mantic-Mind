// Soma — AVX2 + FMA quantized dot products.
//
// Compiled ONLY on x86-64, and only reached after a runtime CPUID check. The
// scalar path in kernels_quant.cpp stays the reference implementation and the
// fallback; these must agree with it to within float reassociation, which
// tests/soma/simd_g3.cpp checks directly rather than inferring from a passing
// forward.
//
// The shape of the win is the same in all four formats: unpack a block of
// sub-byte levels into float lanes, then one FMA per lane against x. The scalar
// version spends most of its time on the unpack — a shift, a mask, a widen and a
// convert PER LEVEL — and that is what vectorizes, not the multiply-add.

#include "soma/kernels_simd.hpp"
#include "soma/quant_format.hpp"

#include <cstring>
#include <immintrin.h>

namespace soma::simd::avx2 {

namespace {

float hsum256(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, _mm256_extractf128_ps(v, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}

float hsum128(__m128 v) noexcept {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 0x55));
    return _mm_cvtss_f32(v);
}

float load_f32(const std::byte* p) noexcept {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/// 16 packed bytes -> 32 nibble levels, in index order, as four float lanes.
///
/// Byte j holds level 2j in its low nibble and level 2j+1 in its high nibble, so
/// the two nibble planes have to be re-interleaved. `unpacklo/hi_epi8` does
/// exactly that: it emits lo[0], hi[0], lo[1], hi[1] ... which is levels 0, 1,
/// 2, 3 in order. Getting this backwards produces a plausible-looking dot
/// product against a permuted weight row.
void unpack_nibbles(__m128i v, __m256& l0, __m256& l1, __m256& l2, __m256& l3) noexcept {
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(v, mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);

    const __m128i a = _mm_unpacklo_epi8(lo, hi); // levels 0..15
    const __m128i b = _mm_unpackhi_epi8(lo, hi); // levels 16..31

    l0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(a));
    l1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(a, 8)));
    l2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b));
    l3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b, 8)));
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

        __m256 vacc = _mm256_setzero_ps();
        __m256 vxs = _mm256_setzero_ps();

        std::uint32_t i = 0;
        for (; i + 32 <= g; i += 32) {
            __m256 l0, l1, l2, l3;
            unpack_nibbles(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8 + i / 2)), l0, l1, l2, l3);

            const __m256 x0 = _mm256_loadu_ps(xv + i);
            const __m256 x1 = _mm256_loadu_ps(xv + i + 8);
            const __m256 x2 = _mm256_loadu_ps(xv + i + 16);
            const __m256 x3 = _mm256_loadu_ps(xv + i + 24);

            vacc = _mm256_fmadd_ps(l0, x0, vacc);
            vacc = _mm256_fmadd_ps(l1, x1, vacc);
            vacc = _mm256_fmadd_ps(l2, x2, vacc);
            vacc = _mm256_fmadd_ps(l3, x3, vacc);

            if (need_xsum) {
                vxs =
                    _mm256_add_ps(vxs, _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3)));
            }
        }

        float part = hsum256(vacc);
        float xsum = need_xsum ? hsum256(vxs) : group_xsum[gi];

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
    const __m256 bias = _mm256_set1_ps(8.0f);

    float acc = 0.0f;
    for (std::uint32_t c = 0; c < cols; c += g) {
        const float scale = load_f32(p);
        const float* xv = x + c;

        __m256 vacc = _mm256_setzero_ps();
        std::uint32_t i = 0;
        for (; i + 32 <= g; i += 32) {
            __m256 l0, l1, l2, l3;
            unpack_nibbles(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 4 + i / 2)), l0, l1, l2, l3);

            vacc = _mm256_fmadd_ps(_mm256_sub_ps(l0, bias), _mm256_loadu_ps(xv + i), vacc);
            vacc = _mm256_fmadd_ps(_mm256_sub_ps(l1, bias), _mm256_loadu_ps(xv + i + 8), vacc);
            vacc = _mm256_fmadd_ps(_mm256_sub_ps(l2, bias), _mm256_loadu_ps(xv + i + 16), vacc);
            vacc = _mm256_fmadd_ps(_mm256_sub_ps(l3, bias), _mm256_loadu_ps(xv + i + 24), vacc);
        }

        float part = hsum256(vacc);
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

        __m256 vacc = _mm256_setzero_ps();
        std::uint32_t i = 0;
        for (; i + 16 <= g; i += 16) {
            const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 4 + i));
            const __m256 l0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
            const __m256 l1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8)));
            vacc = _mm256_fmadd_ps(l0, _mm256_loadu_ps(xv + i), vacc);
            vacc = _mm256_fmadd_ps(l1, _mm256_loadu_ps(xv + i + 8), vacc);
        }

        float part = hsum256(vacc);
        for (; i < g; ++i) {
            part += static_cast<float>(static_cast<std::int8_t>(p[4 + i])) * xv[i];
        }
        acc += scale * part;
        p += gb;
    }
    return acc;
}

// ── Q6_G — 4 levels per 3 bytes ──────────────────────────────────────────────
//
// The awkward one: 6 bits does not divide a byte, so a level's bits are found by
// treating each 3-byte run as a 24-bit word and taking four 6-bit fields. That
// puts level 4j+k in lane j of vector k, i.e. TRANSPOSED relative to x. A 4x4
// float transpose is cheaper than any per-level extraction, so the levels are
// straightened rather than gathering x to match.
float qdot_q6g(const std::byte* p, std::uint32_t cols, std::uint32_t g, const float* x) noexcept {
    const auto gb = group_bytes(DType::Q6_G, g);
    const __m128i shuf = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    const __m128i m6 = _mm_set1_epi32(0x3F);
    const __m128 bias = _mm_set1_ps(32.0f);

    float acc = 0.0f;
    for (std::uint32_t c = 0; c < cols; c += g) {
        const float scale = load_f32(p);
        const float* xv = x + c;
        const std::byte* q = p + 4;

        __m128 vacc = _mm_setzero_ps();
        std::uint32_t i = 0;
        for (; i + 16 <= g; i += 16, q += 12) {
            // Exactly 12 bytes, never 16. A 16-byte load here would run past the
            // final group of the final row of the tensor.
            __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q));
            std::uint32_t hi4 = 0;
            std::memcpy(&hi4, q + 8, 4);
            v = _mm_insert_epi32(v, static_cast<int>(hi4), 2);

            const __m128i d = _mm_shuffle_epi8(v, shuf);
            __m128 f0 = _mm_cvtepi32_ps(_mm_and_si128(d, m6));
            __m128 f1 = _mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi32(d, 6), m6));
            __m128 f2 = _mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi32(d, 12), m6));
            __m128 f3 = _mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi32(d, 18), m6));

            _MM_TRANSPOSE4_PS(f0, f1, f2, f3); // -> levels 0..3, 4..7, 8..11, 12..15

            vacc = _mm_fmadd_ps(_mm_sub_ps(f0, bias), _mm_loadu_ps(xv + i), vacc);
            vacc = _mm_fmadd_ps(_mm_sub_ps(f1, bias), _mm_loadu_ps(xv + i + 4), vacc);
            vacc = _mm_fmadd_ps(_mm_sub_ps(f2, bias), _mm_loadu_ps(xv + i + 8), vacc);
            vacc = _mm_fmadd_ps(_mm_sub_ps(f3, bias), _mm_loadu_ps(xv + i + 12), vacc);
        }

        float part = hsum128(vacc);
        for (; i < g; i += 4, q += 3) {
            const std::uint32_t packed =
                static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[0])) |
                (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[1])) << 8) |
                (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[2])) << 16);
            for (std::uint32_t k = 0; k < 4 && i + k < g; ++k) {
                const int lv = static_cast<int>((packed >> (6 * k)) & 0x3F) - 32;
                part += static_cast<float>(lv) * xv[i + k];
            }
        }
        acc += scale * part;
        p += gb;
    }
    return acc;
}

} // namespace soma::simd::avx2
