// Soma — quantized matvec, fused with dequantization.
//
// The accumulation walks groups and applies each group's scale to a partial dot
// product, so a quantized row is never materialized as fp32. Materializing would
// cost a full row of scratch per call and give back exactly the memory the
// quantization was for.
//
// Correctness first, as at G0. These are the numbers the autotuner will later
// choose between; a clever kernel that is subtly wrong here poisons every
// measurement built on it.

#include "soma/kernels_f32.hpp"
#include "soma/kernels_simd.hpp"
#include "soma/quant_format.hpp"
#include "soma/threading.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(SOMA_HAS_AVX2)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

namespace soma {

namespace {

float get_f32(const std::byte* p) noexcept {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/// One row of a quantized weight against x.
float qdot_row(
    DType dtype, const std::byte* p, std::uint32_t cols, std::uint32_t g, const float* x) noexcept {
    float acc = 0.0f;
    const auto gb = group_bytes(dtype, g);

    for (std::uint32_t c = 0; c < cols; c += g) {
        const float scale = get_f32(p);
        const float* xv = x + c;
        float part = 0.0f;

        switch (dtype) {
        case DType::Q8_0:
            for (std::uint32_t i = 0; i < g; ++i) {
                part += static_cast<float>(static_cast<std::int8_t>(p[4 + i])) * xv[i];
            }
            acc += scale * part;
            break;

        case DType::Q4_0:
            for (std::uint32_t i = 0; i < g; i += 2) {
                const auto byte = static_cast<std::uint8_t>(p[4 + i / 2]);
                part += static_cast<float>(static_cast<int>(byte & 0x0F) - 8) * xv[i];
                if (i + 1 < g) {
                    part += static_cast<float>(static_cast<int>(byte >> 4) - 8) * xv[i + 1];
                }
            }
            acc += scale * part;
            break;

        case DType::Q4_G: {
            // Asymmetric: w = min + scale*level, so the group contributes
            // min * sum(x) + scale * dot(level, x). Folding the min into a
            // separate sum keeps it one pass.
            const float minv = get_f32(p + 4);
            float xsum = 0.0f;
            for (std::uint32_t i = 0; i < g; i += 2) {
                const auto byte = static_cast<std::uint8_t>(p[8 + i / 2]);
                part += static_cast<float>(byte & 0x0F) * xv[i];
                xsum += xv[i];
                if (i + 1 < g) {
                    part += static_cast<float>(byte >> 4) * xv[i + 1];
                    xsum += xv[i + 1];
                }
            }
            acc += scale * part + minv * xsum;
            break;
        }

        case DType::Q6_G: {
            const std::byte* q = p + 4;
            for (std::uint32_t i = 0; i < g; i += 4) {
                const std::uint32_t packed =
                    static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[0])) |
                    (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[1])) << 8) |
                    (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[2])) << 16);
                for (std::uint32_t k = 0; k < 4 && i + k < g; ++k) {
                    const int lv = static_cast<int>((packed >> (6 * k)) & 0x3F) - 32;
                    part += static_cast<float>(lv) * xv[i + k];
                }
                q += 3;
            }
            acc += scale * part;
            break;
        }

        default:
            return 0.0f;
        }
        p += gb;
    }
    return acc;
}

/// Per-group sums of x, for the asymmetric formats.
///
/// `min * sum(x)` over a group does not depend on the weight row, so computing
/// it once per matvec instead of once per output element removes a whole pass
/// over x per row. On a [2048, 2048] projection that is 2047 redundant passes.
constexpr std::size_t kMaxCachedGroups = 256;

} // namespace

void matvec(const WeightRef& w, std::span<const float> x, std::span<float> y) noexcept {
    if (!w.quantized()) {
        f32::matvec(w.f32, x, w.rows, w.cols, y);
        return;
    }
    const auto per_row =
        (static_cast<std::size_t>(w.cols) / w.group) * group_bytes(w.dtype, w.group);
    const std::byte* base = w.bytes.data();

    const bool use_simd = simd::available();

    // Hoisted group sums, when the format needs them and the shape fits. Above
    // the cap the kernel recomputes inline — correct either way, and the cap is
    // ~4x past any projection width in the target models.
    float xsum[kMaxCachedGroups];
    const float* xsum_p = nullptr;
    const auto n_groups = static_cast<std::size_t>(w.cols) / w.group;
    if (w.dtype == DType::Q4_G && n_groups <= kMaxCachedGroups) {
        for (std::size_t gi = 0; gi < n_groups; ++gi) {
            float s = 0.0f;
            const float* xv = x.data() + gi * w.group;
            for (std::uint32_t i = 0; i < w.group; ++i)
                s += xv[i];
            xsum[gi] = s;
        }
        xsum_p = xsum;
    }

    const auto row_of = [&](std::uint32_t r) noexcept {
        const std::byte* p = base + static_cast<std::size_t>(r) * per_row;
        if (use_simd) {
            switch (w.dtype) {
            case DType::Q4_G:
                y[r] = simd::qdot_q4g(p, w.cols, w.group, x.data(), xsum_p);
                return;
            case DType::Q6_G:
                y[r] = simd::qdot_q6g(p, w.cols, w.group, x.data());
                return;
            case DType::Q8_0:
                y[r] = simd::qdot_q8_0(p, w.cols, w.group, x.data());
                return;
            case DType::Q4_0:
                y[r] = simd::qdot_q4_0(p, w.cols, w.group, x.data());
                return;
            default:
                break;
            }
        }
        y[r] = qdot_row(w.dtype, p, w.cols, w.group, x.data());
    };

    // Same partitioning rule as the fp32 path: one output row per thread, whole.
    // The hoisted `xsum` above is computed before the split and then read-only,
    // so sharing it across workers is safe and it is not recomputed per chunk.
    const auto work = static_cast<std::uint64_t>(w.rows) * w.cols;
    if (work >= kParallelMacThreshold && !ThreadPool::in_parallel_region()) {
        const auto rows_per = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(kChunkMacs / std::max<std::uint32_t>(1, w.cols)));
        ThreadPool::global().parallel_for(
            w.rows, rows_per, [&](std::uint32_t begin, std::uint32_t end, std::uint32_t) {
                for (std::uint32_t r = begin; r < end; ++r)
                    row_of(r);
            });
        return;
    }
    for (std::uint32_t r = 0; r < w.rows; ++r)
        row_of(r);
}

void matmul_tiled(const WeightRef& w, const float* x, std::uint32_t n_inputs, float* y) noexcept {
    if (n_inputs == 0 || w.rows == 0) return;
    if (n_inputs == 1) {
        matvec(w, std::span<const float>(x, w.cols), std::span<float>(y, w.rows));
        return;
    }

    const bool use_simd = simd::available();
    const auto per_row =
        w.quantized() ? (static_cast<std::size_t>(w.cols) / w.group) * group_bytes(w.dtype, w.group)
                      : 0;

    // One xsum row per input, since Q4_G's `min * sum(x)` term depends on the
    // input rather than the weight. Hoisted out of the weight loop for the same
    // reason it is hoisted in matvec: it is recomputed otherwise once per output.
    const auto n_groups = w.quantized() ? static_cast<std::size_t>(w.cols) / w.group : 0;
    std::vector<float> xsum;
    const bool cache_xsum = (w.dtype == DType::Q4_G) && n_groups > 0 && n_groups * n_inputs <= 4096;
    if (cache_xsum) {
        xsum.resize(n_groups * n_inputs);
        for (std::uint32_t t = 0; t < n_inputs; ++t) {
            for (std::size_t gi = 0; gi < n_groups; ++gi) {
                float s = 0.0f;
                const float* xv = x + static_cast<std::size_t>(t) * w.cols + gi * w.group;
                for (std::uint32_t i = 0; i < w.group; ++i)
                    s += xv[i];
                xsum[static_cast<std::size_t>(t) * n_groups + gi] = s;
            }
        }
    }

    const auto do_row = [&](std::uint32_t r) noexcept {
        for (std::uint32_t t = 0; t < n_inputs; ++t) {
            const float* xt = x + static_cast<std::size_t>(t) * w.cols;
            float* dst = y + static_cast<std::size_t>(t) * w.rows + r;
            if (!w.quantized()) {
                *dst = f32::dot(w.f32.subspan(static_cast<std::size_t>(r) * w.cols, w.cols),
                                std::span<const float>(xt, w.cols),
                                w.cols);
                continue;
            }
            const std::byte* p = w.bytes.data() + static_cast<std::size_t>(r) * per_row;
            const float* gx =
                cache_xsum ? xsum.data() + static_cast<std::size_t>(t) * n_groups : nullptr;
            if (use_simd) {
                switch (w.dtype) {
                case DType::Q4_G:
                    *dst = simd::qdot_q4g(p, w.cols, w.group, xt, gx);
                    continue;
                case DType::Q6_G:
                    *dst = simd::qdot_q6g(p, w.cols, w.group, xt);
                    continue;
                case DType::Q8_0:
                    *dst = simd::qdot_q8_0(p, w.cols, w.group, xt);
                    continue;
                case DType::Q4_0:
                    *dst = simd::qdot_q4_0(p, w.cols, w.group, xt);
                    continue;
                default:
                    break;
                }
            }
            *dst = qdot_row(w.dtype, p, w.cols, w.group, xt);
        }
    };

    // Parallel over WEIGHT rows: outputs are disjoint across r for every t, so
    // this keeps the bit-identity property the whole threading design rests on.
    const auto work = static_cast<std::uint64_t>(w.rows) * w.cols * n_inputs;
    if (work >= kParallelMacThreshold && !ThreadPool::in_parallel_region()) {
        const auto rows_per = std::max<std::uint32_t>(
            1,
            static_cast<std::uint32_t>(kChunkMacs / std::max<std::uint32_t>(1, w.cols * n_inputs)));
        ThreadPool::global().parallel_for(
            w.rows, rows_per, [&](std::uint32_t b, std::uint32_t e, std::uint32_t) {
                for (std::uint32_t r = b; r < e; ++r)
                    do_row(r);
            });
        return;
    }
    for (std::uint32_t r = 0; r < w.rows; ++r)
        do_row(r);
}

void matmul(const WeightRef& w,
            std::span<const float> x,
            std::uint32_t n_rows,
            std::span<float> y) noexcept {
    for (std::uint32_t i = 0; i < n_rows; ++i) {
        matvec(w,
               x.subspan(static_cast<std::size_t>(i) * w.cols, w.cols),
               y.subspan(static_cast<std::size_t>(i) * w.rows, w.rows));
    }
}

namespace simd {

namespace {

/// AVX2 + FMA, *and* the OS agreeing to preserve YMM.
///
/// The OSXSAVE/XCR0 half is not ceremony. A CPU can advertise AVX2 while the OS
/// declines to save YMM across a context switch; using it then corrupts vector
/// state at arbitrary points, which surfaces as irreproducible numerics far from
/// the cause. Checking is a few instructions, once.
///
/// This detection deliberately lives in a translation unit compiled WITHOUT
/// /arch:AVX2. Putting it beside the kernels would let the compiler emit AVX2
/// into the very function whose job is to decide whether AVX2 may run.
SimdTier detect() noexcept {
#if defined(SOMA_HAS_AVX2)
#if defined(_MSC_VER)
    int r[4]{};
    __cpuid(r, 0);
    if (r[0] < 7) return SimdTier::Scalar;

    __cpuid(r, 1);
    const bool fma = (r[2] & (1 << 12)) != 0;
    const bool osxsave = (r[2] & (1 << 27)) != 0;
    if (!osxsave || !fma) return SimdTier::Scalar;

    const auto xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6u) != 0x6u) return SimdTier::Scalar; // XMM + YMM

    __cpuidex(r, 7, 0);
    const bool avx2 = (r[1] & (1 << 5)) != 0;
    if (!avx2) return SimdTier::Scalar;

#if defined(SOMA_HAS_AVX512)
    // 0xE6 = XMM | YMM | opmask | ZMM_hi256 | hi16_ZMM. All five must be
    // preserved by the OS; checking only the AVX2 pair and then executing
    // 512-bit code is how you corrupt vector state on a kernel that does not
    // save it.
    if ((xcr0 & 0xE6u) == 0xE6u) {
        const bool f = (r[1] & (1 << 16)) != 0;   // AVX512F
        const bool dq = (r[1] & (1 << 17)) != 0;  // AVX512DQ
        const bool bw = (r[1] & (1 << 30)) != 0;  // AVX512BW
        const bool vl = (r[1] & (1u << 31)) != 0; // AVX512VL
        if (f && dq && bw && vl) return SimdTier::Avx512;
    }
#endif
    return SimdTier::Avx2;
#else
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        return SimdTier::Scalar;
    }
#if defined(SOMA_HAS_AVX512)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl")) {
        return SimdTier::Avx512;
    }
#endif
    return SimdTier::Avx2;
#endif
#else
    return SimdTier::Scalar;
#endif
}

} // namespace

/// SOMA_SIMD_TIER = scalar | avx2 | avx512, capped by what the host supports.
///
/// Exists so "is this tier actually worth it?" is an A/B measurement on one
/// binary rather than a rebuild-and-compare across two. The AVX-512 kernels are
/// 1.5-1.7x faster in isolation and made no measurable difference end to end;
/// distinguishing those two facts required being able to flip the tier under a
/// fixed workload, and every future ISA tier will raise the same question.
SimdTier tier_override(SimdTier detected) noexcept {
#if defined(_MSC_VER)
    char* buf = nullptr;
    std::size_t len = 0;
    const bool have = (_dupenv_s(&buf, &len, "SOMA_SIMD_TIER") == 0 && buf != nullptr);
    const std::string v = have ? std::string(buf) : std::string();
    std::free(buf);
#else
    const char* raw = std::getenv("SOMA_SIMD_TIER");
    const std::string v = raw ? std::string(raw) : std::string();
#endif
    if (v == "scalar") return SimdTier::Scalar;
    // Capped, never raised: asking for a tier the CPU lacks must not execute it.
    if (v == "avx2") return std::min(detected, SimdTier::Avx2);
    if (v == "avx512") return detected;
    return detected;
}

SimdTier tier() noexcept {
    static const SimdTier kTier = tier_override(detect());
    return kTier;
}

bool available() noexcept {
    return tier() != SimdTier::Scalar;
}

// ── dispatchers ──────────────────────────────────────────────────────────────
//
// Compiled at the baseline ISA, so calling one is always safe; the branch is on
// a cached value and predicts perfectly.

#if defined(SOMA_HAS_AVX512)
#define SOMA_IS_512 (tier() == SimdTier::Avx512)
#else
#define SOMA_IS_512 false
#endif

#if defined(SOMA_HAS_AVX2)

float qdot_q4g(const std::byte* p,
               std::uint32_t cols,
               std::uint32_t g,
               const float* x,
               const float* gx) noexcept {
    if (SOMA_IS_512) return avx512::qdot_q4g(p, cols, g, x, gx);
    return avx2::qdot_q4g(p, cols, g, x, gx);
}

/// Q6_G has no 512-bit kernel — see kernels_simd.hpp. Routed to AVX2 even at the
/// AVX-512 tier, which is why the tier is not a promise that every format got
/// wider.
float qdot_q6g(const std::byte* p, std::uint32_t cols, std::uint32_t g, const float* x) noexcept {
    return avx2::qdot_q6g(p, cols, g, x);
}

float qdot_q8_0(const std::byte* p, std::uint32_t cols, std::uint32_t g, const float* x) noexcept {
    if (SOMA_IS_512) return avx512::qdot_q8_0(p, cols, g, x);
    return avx2::qdot_q8_0(p, cols, g, x);
}

float qdot_q4_0(const std::byte* p, std::uint32_t cols, std::uint32_t g, const float* x) noexcept {
    if (SOMA_IS_512) return avx512::qdot_q4_0(p, cols, g, x);
    return avx2::qdot_q4_0(p, cols, g, x);
}

// ── fp32 stays on AVX2, even at the AVX-512 tier ─────────────────────────────
//
// This is a MEASURED choice, not an oversight, and it is the more interesting
// result of the AVX-512 work. `simd_g3 --bench` runs both kernels back to back
// on the same data, best-of-5, and reports AVX-512 at 0.97-0.99x of AVX2 for a
// 2048x2048 fp32 matvec — parity, reproducibly, across repeated runs.
//
// (The autotuner's f32 figure swings between ~76 and ~86 GF/s run to run and is
// NOT evidence either way. An earlier version of this comment cited that swing
// as a regression; it was noise, and a single-shot number should not have been
// used to justify a dispatch decision when a stable one was available.)
//
// The reason is what the two kernel families do per byte. The quantized kernels
// spend most of their instructions UNPACKING sub-byte levels — shift, mask,
// widen, convert — which is pure ALU work that doubles with the vector width;
// they gain 1.5-1.7x. An fp32 matvec has no unpacking at all: it streams weights
// and issues one FMA per element, so at these shapes it is already limited by how
// fast the weights arrive, and a wider register does not make memory faster.
//
// So the tier is not a blanket "use the widest thing available" — it is per
// kernel family, decided by measurement. The AVX-512 fp32 kernels are kept and
// still exercised by the test, because this conclusion is a property of THIS
// silicon and its memory system; on a host with more bandwidth per core the
// answer could flip, and the measurement is there to catch that.
float dot(const float* a, const float* b, std::uint32_t n) noexcept {
    return avx2::dot(a, b, n);
}

void matvec(const float* w, const float* x, std::uint32_t m, std::uint32_t k, float* y) noexcept {
    avx2::matvec(w, x, m, k, y);
}

void axpy(float alpha, const float* x, std::uint32_t n, float* y) noexcept {
    avx2::axpy(alpha, x, n, y);
}

float sumsq(const float* x, std::uint32_t n) noexcept {
    return avx2::sumsq(x, n);
}

float vmax(const float* x, std::uint32_t n) noexcept {
    return avx2::vmax(x, n);
}

void scale(float* x, float s, std::uint32_t n) noexcept {
    avx2::scale(x, s, n);
}

#else // !SOMA_HAS_AVX2

// ── no vector tier in this build ─────────────────────────────────────────────
//
// aarch64 and every other non-x86-64 target: src/soma/CMakeLists.txt omits the
// AVX2 translation units, so there is nothing for these to dispatch to. They
// still have to be DEFINED — kernels_simd.hpp is architecture-neutral and
// kernels_f32.cpp names simd::dot() inside its `available()` branch, so the
// symbols are referenced on every host regardless of what got compiled.
//
// Reaching one is a contract violation, not a host-capability question: detect()
// returns Scalar unconditionally in this build, so available() is false and
// every call site in this library already branches to the scalar reference
// (qdot_row here, the plain loops in kernels_f32.cpp). Aborting rather than
// returning 0 is deliberate — these are noexcept and return floats that flow
// straight into logits, so a silent wrong answer would surface as a numerics
// bug a long way from a missing `if (simd::available())`.
//
// Deleting these definitions instead is not an option, and neither is leaving
// them undefined: that is precisely what broke the aarch64 build, as an
// undefined reference to soma::simd::avx2::* out of kernels_quant.cpp.

namespace {

[[noreturn]] void no_simd_tier(const char* name) noexcept {
    std::fprintf(stderr,
                 "soma: simd::%s called on a build with no SIMD tier; "
                 "call sites must check simd::available() first\n",
                 name);
    std::abort();
}

} // namespace

float qdot_q4g(
    const std::byte*, std::uint32_t, std::uint32_t, const float*, const float*) noexcept {
    no_simd_tier("qdot_q4g");
}

float qdot_q6g(const std::byte*, std::uint32_t, std::uint32_t, const float*) noexcept {
    no_simd_tier("qdot_q6g");
}

float qdot_q8_0(const std::byte*, std::uint32_t, std::uint32_t, const float*) noexcept {
    no_simd_tier("qdot_q8_0");
}

float qdot_q4_0(const std::byte*, std::uint32_t, std::uint32_t, const float*) noexcept {
    no_simd_tier("qdot_q4_0");
}

float dot(const float*, const float*, std::uint32_t) noexcept {
    no_simd_tier("dot");
}

void matvec(const float*, const float*, std::uint32_t, std::uint32_t, float*) noexcept {
    no_simd_tier("matvec");
}

void axpy(float, const float*, std::uint32_t, float*) noexcept {
    no_simd_tier("axpy");
}

float sumsq(const float*, std::uint32_t) noexcept {
    no_simd_tier("sumsq");
}

float vmax(const float*, std::uint32_t) noexcept {
    no_simd_tier("vmax");
}

void scale(float*, float, std::uint32_t) noexcept {
    no_simd_tier("scale");
}

#endif // SOMA_HAS_AVX2

#undef SOMA_IS_512

} // namespace simd

void matvec_single_row(const WeightRef& w, std::span<const float> x, std::span<float> y) noexcept {
    // Currently identical to matvec(). Kept as a distinct entry point because
    // Determinism::Strict must pin the kernel family, and once the autotuner can
    // select a different implementation per shape these two stop agreeing by
    // construction. Collapsing them now would mean re-separating them later,
    // after something already depends on the collapse.
    matvec(w, x, y);
}

} // namespace soma
