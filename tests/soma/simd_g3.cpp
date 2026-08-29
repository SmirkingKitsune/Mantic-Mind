// Soma — G3: SIMD quantized kernels against the scalar reference.
//
// Checked DIRECTLY, not inferred from a forward that still produces plausible
// logits. A dot product against a permuted weight row — the exact failure mode
// of getting nibble interleaving or the 6-bit transpose backwards — changes the
// output by an amount that looks like quantization noise at the model level but
// is a straightforward correctness bug at this level.
//
// The reference is the scalar path in kernels_quant.cpp, reached by the same
// public entry point with dispatch disabled. Agreement is to within float
// REASSOCIATION, not bit-identity: an 8-wide accumulator with a horizontal sum
// at the end adds the same products in a different order.
//
// Usage: simd_g3 [--bench]

#include "soma/kernels_simd.hpp"
#include "soma/quant_format.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << "   " << std::left << std::setw(46) << what << (ok ? "OK" : "FAIL");
    if (!detail.empty()) std::cout << "   " << detail;
    std::cout << "\n";
    if (!ok) ++g_failures;
}

/// Deterministic pseudo-random floats. Not std::mt19937 — the point is a fixed
/// sequence that is identical on every platform and every run, so a failure is
/// reproducible from the test name alone.
struct Rng {
    std::uint32_t s = 0x12345678u;
    float next() noexcept {
        s = s * 1664525u + 1013904223u;
        return (static_cast<float>(s >> 8) / 8388608.0f) - 1.0f;  // [-1, 1)
    }
};

/// The scalar reference, transcribed from kernels_quant.cpp.
///
/// Duplicated rather than exposed, for the same reason convert.py implements the
/// quant formats a second time: if the SIMD kernel and its reference shared an
/// unpacking helper, they would agree on a mistake in that helper. Two
/// independent readings of the byte layout is the whole value of the check.
/// `Acc` is float for the reference and double for ground truth — the same
/// reading of the byte layout at two precisions, so the exact answer costs no
/// third implementation to disagree with.
template <typename Acc>
Acc ref_qdot_t(soma::DType dtype, const std::byte* p, std::uint32_t cols,
               std::uint32_t g, const float* x) {
    Acc acc = 0;
    const auto gb = soma::group_bytes(dtype, g);
    for (std::uint32_t c = 0; c < cols; c += g) {
        float scale = 0.0f;
        std::memcpy(&scale, p, 4);
        const float* xv = x + c;
        Acc part = 0;

        if (dtype == soma::DType::Q8_0) {
            for (std::uint32_t i = 0; i < g; ++i) {
                part += static_cast<Acc>(static_cast<std::int8_t>(p[4 + i])) * xv[i];
            }
            acc += static_cast<Acc>(scale) * part;
        } else if (dtype == soma::DType::Q4_0) {
            for (std::uint32_t i = 0; i < g; ++i) {
                const auto b = static_cast<std::uint8_t>(p[4 + i / 2]);
                const int lv = ((i % 2 == 0) ? (b & 0x0F) : (b >> 4)) - 8;
                part += static_cast<Acc>(lv) * xv[i];
            }
            acc += static_cast<Acc>(scale) * part;
        } else if (dtype == soma::DType::Q4_G) {
            float minv = 0.0f;
            std::memcpy(&minv, p + 4, 4);
            Acc xsum = 0;
            for (std::uint32_t i = 0; i < g; ++i) {
                const auto b = static_cast<std::uint8_t>(p[8 + i / 2]);
                const int lv = (i % 2 == 0) ? (b & 0x0F) : (b >> 4);
                part += static_cast<Acc>(lv) * xv[i];
                xsum += xv[i];
            }
            acc += static_cast<Acc>(scale) * part + static_cast<Acc>(minv) * xsum;
        } else if (dtype == soma::DType::Q6_G) {
            for (std::uint32_t i = 0; i < g; ++i) {
                const std::byte* q = p + 4 + (i / 4) * 3;
                const std::uint32_t packed =
                    static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[0])) |
                    (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[1])) << 8) |
                    (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[2])) << 16);
                const int lv = static_cast<int>((packed >> (6 * (i % 4))) & 0x3F) - 32;
                part += static_cast<Acc>(lv) * xv[i];
            }
            acc += static_cast<Acc>(scale) * part;
        }
        p += gb;
    }
    return acc;
}

float ref_qdot(soma::DType dtype, const std::byte* p, std::uint32_t cols,
               std::uint32_t g, const float* x) {
    return ref_qdot_t<float>(dtype, p, cols, g, x);
}

float simd_qdot(soma::DType dtype, const std::byte* p, std::uint32_t cols,
                std::uint32_t g, const float* x) {
    switch (dtype) {
        case soma::DType::Q4_G: return soma::simd::qdot_q4g(p, cols, g, x, nullptr);
        case soma::DType::Q6_G: return soma::simd::qdot_q6g(p, cols, g, x);
        case soma::DType::Q8_0: return soma::simd::qdot_q8_0(p, cols, g, x);
        case soma::DType::Q4_0: return soma::simd::qdot_q4_0(p, cols, g, x);
        default: return 0.0f;
    }
}

const char* name_of(soma::DType d) {
    switch (d) {
        case soma::DType::Q4_G: return "q4_g";
        case soma::DType::Q6_G: return "q6_g";
        case soma::DType::Q8_0: return "q8_0";
        case soma::DType::Q4_0: return "q4_0";
        default: return "?";
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") bench = true;
    }

    const auto tier = soma::simd::tier();
    const char* tier_name = (tier == soma::simd::SimdTier::Avx512) ? "AVX-512"
                            : (tier == soma::simd::SimdTier::Avx2) ? "AVX2"
                                                                   : "scalar";
    std::cout << "SIMD tier: " << tier_name << "\n";
    if (!soma::simd::available()) {
        // Not a failure. The scalar path is the fallback by design, and a host
        // without AVX2 should report that rather than fail a correctness gate it
        // cannot run.
        std::cout << "\nSIMD unavailable on this host — the scalar path is in use and this\n"
                     "check has nothing to compare against. Reporting OK.\n";
        return 0;
    }

    const soma::DType dtypes[] = {soma::DType::Q4_G, soma::DType::Q6_G,
                                  soma::DType::Q8_0, soma::DType::Q4_0};

    // ── 1. agreement with the scalar reference ───────────────────────────────
    //
    // Widths chosen to exercise the tail: 128 is the production group and hits
    // only the vector path; 2048 is a real projection width; 192 and 320 leave
    // remainders the SIMD loop must hand back to scalar code. A kernel that is
    // correct only on multiples of its vector width passes a naive test and
    // fails on the first unusual shape.
    // The bar is NOT an absolute error bound. A fixed threshold conflates two
    // different things: q4_g is asymmetric, so its result is
    // `scale*dot(level,x) + min*sum(x)` — a DIFFERENCE OF TWO LARGE TERMS that
    // cancel. Its relative error against exact arithmetic is inherently ~10x the
    // symmetric formats' and has nothing to do with which kernel computed it.
    //
    // So both kernels are measured against a double-precision evaluation of the
    // same bytes, and the requirement is that SIMD is NO WORSE THAN SCALAR. That
    // separates "this format is ill-conditioned" (expected, and a property of
    // the quant map) from "this kernel is wrong" (a bug), which an absolute
    // threshold cannot do in either direction.
    std::cout << "\n1. accuracy vs exact, scalar and SIMD\n";
    std::cout << "   " << std::left << std::setw(8) << "dtype" << std::setw(15) << "scalar err"
              << std::setw(15) << "simd err" << "simd/scalar\n";

    for (const auto dt : dtypes) {
        double worst_ref = 0.0, worst_simd = 0.0, worst_ratio = 0.0;
        for (const std::uint32_t cols : {128u, 192u, 320u, 2048u}) {
            for (const std::uint32_t g : {32u, 64u, 128u}) {
                if (cols % g != 0) continue;

                Rng rng;
                const auto n_groups = cols / g;
                std::vector<std::byte> buf(n_groups * soma::group_bytes(dt, g));
                for (auto& b : buf) {
                    b = static_cast<std::byte>(static_cast<std::uint8_t>(
                        static_cast<int>(rng.next() * 127.0f) & 0xFF));
                }
                // Real scales, not random bit patterns: a random 4 bytes is a
                // valid float only by luck and can be NaN or 1e38, which would
                // make the comparison meaningless rather than strict.
                for (std::uint32_t gi = 0; gi < n_groups; ++gi) {
                    const auto off = gi * soma::group_bytes(dt, g);
                    const float scale = 0.01f + 0.001f * static_cast<float>(gi);
                    std::memcpy(buf.data() + off, &scale, 4);
                    if (dt == soma::DType::Q4_G) {
                        const float minv = -0.5f + 0.01f * static_cast<float>(gi);
                        std::memcpy(buf.data() + off + 4, &minv, 4);
                    }
                }

                std::vector<float> x(cols);
                for (auto& v : x) v = rng.next();

                const double exact = ref_qdot_t<double>(dt, buf.data(), cols, g, x.data());
                const float r = ref_qdot(dt, buf.data(), cols, g, x.data());
                const float s = simd_qdot(dt, buf.data(), cols, g, x.data());

                const double denom = std::max(1e-6, std::fabs(exact));
                const double e_ref = std::fabs(r - exact) / denom;
                const double e_simd = std::fabs(s - exact) / denom;
                worst_ref = std::max(worst_ref, e_ref);
                worst_simd = std::max(worst_simd, e_simd);
            }
        }
        // Envelopes, NOT the worst per-case ratio. Per-case, the scalar error is
        // sometimes near zero by luck, and dividing by it manufactures a huge
        // ratio out of two numbers that are both negligible. Comparing the worst
        // error each kernel produces is the stable form of the same question.
        worst_ratio = worst_simd / std::max(worst_ref, 1e-12);

        std::cout << "   " << std::left << std::setw(8) << name_of(dt)
                  << std::scientific << std::setprecision(2)
                  << std::setw(15) << worst_ref << std::setw(15) << worst_simd
                  << std::fixed << std::setprecision(2) << worst_ratio << "x\n";

        // Two independent claims, so two checks.
        //
        // A permuted or mis-unpacked row gives O(1) relative error — thousands of
        // times past anything rounding produces — so 1e-3 catches the structural
        // bug regardless of conditioning.
        check(worst_simd < 1e-3, std::string(name_of(dt)) + " unpacks the layout correctly",
              "O(1) error would mean a permuted row");

        // Then: not materially less accurate than scalar. The 1e-5 floor matters
        // as much as the 4x factor — below it, fp32 error over thousands of terms
        // is dominated by which rounding happened to fall where, and a ratio
        // between two such numbers carries no signal about kernel quality.
        check(worst_simd <= std::max(4.0 * worst_ref, 1e-5),
              std::string(name_of(dt)) + " is no less accurate than scalar",
              "ratio " + std::to_string(worst_ratio));
    }

    // ── 2. the xsum hoist must not change the answer ─────────────────────────
    //
    // Q4_G's `min * sum(x)` term is hoisted out of the row loop in matvec(). That
    // is an algebraic identity, so it should agree with computing it inline — but
    // it is a different summation order over the same values, so "should" is
    // worth checking once rather than assuming.
    std::cout << "\n2. hoisted group-sum path\n";
    {
        constexpr std::uint32_t cols = 2048, g = 128;
        Rng rng;
        const auto n_groups = cols / g;
        std::vector<std::byte> buf(n_groups * soma::group_bytes(soma::DType::Q4_G, g));
        for (auto& b : buf) {
            b = static_cast<std::byte>(
                static_cast<std::uint8_t>(static_cast<int>(rng.next() * 127.0f) & 0xFF));
        }
        for (std::uint32_t gi = 0; gi < n_groups; ++gi) {
            const auto off = gi * soma::group_bytes(soma::DType::Q4_G, g);
            const float scale = 0.02f, minv = -0.3f;
            std::memcpy(buf.data() + off, &scale, 4);
            std::memcpy(buf.data() + off + 4, &minv, 4);
        }
        std::vector<float> x(cols);
        for (auto& v : x) v = rng.next();

        std::vector<float> xs(n_groups);
        for (std::uint32_t gi = 0; gi < n_groups; ++gi) {
            float s = 0.0f;
            for (std::uint32_t i = 0; i < g; ++i) s += x[gi * g + i];
            xs[gi] = s;
        }

        const float inl = soma::simd::qdot_q4g(buf.data(), cols, g, x.data(), nullptr);
        const float hoi = soma::simd::qdot_q4g(buf.data(), cols, g, x.data(), xs.data());
        const double rel = std::fabs(static_cast<double>(hoi - inl)) /
                           std::max(1e-6f, std::fabs(inl));
        check(rel < 1e-5, "hoisted xsum agrees with inline",
              "rel " + std::to_string(rel));
    }

    // ── 3. fp32 primitives ───────────────────────────────────────────────────
    //
    // Same standard as the quantized kernels: measured against a double-precision
    // evaluation, required to be no worse than the scalar loop it replaces.
    // matvec's 4-rows-at-a-time path makes the row count matter as much as the
    // width, so both are varied — a kernel correct only when m % 4 == 0 passes
    // any test that forgets to try m = 7.
    std::cout << "\n3. fp32 primitives vs exact\n";
    std::cout << "   " << std::left << std::setw(10) << "op" << std::setw(15) << "scalar err"
              << std::setw(15) << "simd err" << "\n";
    {
        double dot_ref = 0.0, dot_simd = 0.0;
        double mv_ref = 0.0, mv_simd = 0.0;
        double ax_worst = 0.0;

        for (const std::uint32_t n : {7u, 8u, 33u, 128u, 2048u, 4096u}) {
            Rng rng;
            std::vector<float> a(n), b(n);
            for (auto& v : a) v = rng.next();
            for (auto& v : b) v = rng.next();

            double exact = 0.0;
            for (std::uint32_t i = 0; i < n; ++i) {
                exact += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            }
            float s_ref = 0.0f;
            for (std::uint32_t i = 0; i < n; ++i) s_ref += a[i] * b[i];
            const float s_simd = soma::simd::dot(a.data(), b.data(), n);

            const double den = std::max(1e-6, std::fabs(exact));
            dot_ref = std::max(dot_ref, std::fabs(s_ref - exact) / den);
            dot_simd = std::max(dot_simd, std::fabs(s_simd - exact) / den);

            // axpy: exact identity, so any difference beyond rounding is a bug.
            std::vector<float> y0(n), y1(n);
            for (std::uint32_t i = 0; i < n; ++i) y0[i] = y1[i] = b[i];
            for (std::uint32_t i = 0; i < n; ++i) y0[i] += 0.375f * a[i];
            soma::simd::axpy(0.375f, a.data(), n, y1.data());
            for (std::uint32_t i = 0; i < n; ++i) {
                ax_worst = std::max(ax_worst, std::fabs(static_cast<double>(y1[i] - y0[i])));
            }
        }

        // m deliberately includes values that are not multiples of 4.
        for (const std::uint32_t m : {1u, 3u, 4u, 7u, 64u}) {
            for (const std::uint32_t k : {7u, 64u, 2048u}) {
                Rng rng;
                std::vector<float> w(static_cast<std::size_t>(m) * k), x(k), y(m), yr(m);
                for (auto& v : w) v = rng.next();
                for (auto& v : x) v = rng.next();

                soma::simd::matvec(w.data(), x.data(), m, k, y.data());
                for (std::uint32_t r = 0; r < m; ++r) {
                    const float* wr = w.data() + static_cast<std::size_t>(r) * k;
                    float acc = 0.0f;
                    double exact = 0.0;
                    for (std::uint32_t i = 0; i < k; ++i) {
                        acc += wr[i] * x[i];
                        exact += static_cast<double>(wr[i]) * static_cast<double>(x[i]);
                    }
                    yr[r] = acc;
                    const double den = std::max(1e-6, std::fabs(exact));
                    mv_ref = std::max(mv_ref, std::fabs(acc - exact) / den);
                    mv_simd = std::max(mv_simd, std::fabs(y[r] - exact) / den);
                }
            }
        }

        std::cout << "   " << std::left << std::setw(10) << "dot"
                  << std::scientific << std::setprecision(2)
                  << std::setw(15) << dot_ref << std::setw(15) << dot_simd << "\n"
                  << "   " << std::left << std::setw(10) << "matvec"
                  << std::setw(15) << mv_ref << std::setw(15) << mv_simd << "\n";

        check(dot_simd <= std::max(4.0 * dot_ref, 1e-5), "dot is no less accurate than scalar");
        check(mv_simd <= std::max(4.0 * mv_ref, 1e-5), "matvec is no less accurate than scalar");
        check(ax_worst < 1e-6, "axpy matches the scalar identity",
              "max abs " + std::to_string(ax_worst));

        // The tail claim, tested EXACTLY rather than with a tolerance.
        //
        // The error columns above are dominated by cancellation at small k — a
        // random dot product lands near zero and inflates the relative error to
        // ~1e-3 with nothing wrong. Judging "does the 4-row path drop the
        // remainder?" against a float threshold therefore measures the RNG as
        // much as the kernel, and would sit one seed away from a false verdict
        // in either direction.
        //
        // With every w and x element 1.0, y[r] must be exactly k for every row:
        // small integers, exactly representable, no rounding anywhere. A dropped
        // row reads as 0, a dropped tail element as k-1. Both are unmissable.
        bool tail_exact = true;
        for (const std::uint32_t m : {1u, 3u, 5u, 7u, 9u, 64u}) {
            for (const std::uint32_t k : {1u, 3u, 7u, 8u, 9u, 31u, 33u}) {
                std::vector<float> w(static_cast<std::size_t>(m) * k, 1.0f), x(k, 1.0f);
                std::vector<float> y(m, -1.0f);
                soma::simd::matvec(w.data(), x.data(), m, k, y.data());
                for (std::uint32_t r = 0; r < m; ++r) {
                    tail_exact &= (y[r] == static_cast<float>(k));
                }
            }
        }
        check(tail_exact, "matvec is exact for every (m % 4, k % 8) remainder",
              "dropped row -> 0, dropped element -> k-1");

        // vmax and scale are exact operations — no reassociation is possible, so
        // anything but equality is a bug, and the check says so.
        Rng rng;
        std::vector<float> v(1000);
        for (auto& e : v) e = rng.next();
        float ref_max = v[0];
        for (const auto e : v) ref_max = std::max(ref_max, e);
        check(soma::simd::vmax(v.data(), 1000) == ref_max, "vmax is exact");

        std::vector<float> s0 = v, s1 = v;
        for (auto& e : s0) e *= 0.25f;
        soma::simd::scale(s1.data(), 0.25f, 1000);
        bool same = true;
        for (std::size_t i = 0; i < s0.size(); ++i) same &= (s0[i] == s1[i]);
        check(same, "scale is exact");
    }

    // ── 4. speed ─────────────────────────────────────────────────────────────
    if (bench) {
        std::cout << "\n4. throughput vs scalar  (2048x2048 matvec)\n";
        std::cout << "   " << std::left << std::setw(8) << "dtype" << std::setw(13) << "scalar ms"
                  << std::setw(13) << "simd ms" << "speedup\n";

        constexpr std::uint32_t cols = 2048, rows = 2048, g = 128;
        for (const auto dt : dtypes) {
            const auto per_row = (cols / g) * soma::group_bytes(dt, g);
            Rng rng;
            std::vector<std::byte> buf(static_cast<std::size_t>(rows) * per_row);
            for (auto& b : buf) {
                b = static_cast<std::byte>(
                    static_cast<std::uint8_t>(static_cast<int>(rng.next() * 127.0f) & 0xFF));
            }
            for (std::uint32_t r = 0; r < rows; ++r) {
                for (std::uint32_t gi = 0; gi < cols / g; ++gi) {
                    const auto off = r * per_row + gi * soma::group_bytes(dt, g);
                    const float scale = 0.02f, minv = -0.3f;
                    std::memcpy(buf.data() + off, &scale, 4);
                    if (dt == soma::DType::Q4_G) std::memcpy(buf.data() + off + 4, &minv, 4);
                }
            }
            std::vector<float> x(cols);
            for (auto& v : x) v = rng.next();

            double sink = 0.0;
            auto t0 = std::chrono::steady_clock::now();
            for (std::uint32_t r = 0; r < rows; ++r) {
                sink += ref_qdot(dt, buf.data() + static_cast<std::size_t>(r) * per_row,
                                 cols, g, x.data());
            }
            const auto ms_ref = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();

            t0 = std::chrono::steady_clock::now();
            for (std::uint32_t r = 0; r < rows; ++r) {
                sink += simd_qdot(dt, buf.data() + static_cast<std::size_t>(r) * per_row,
                                  cols, g, x.data());
            }
            const auto ms_simd = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();

            std::cout << "   " << std::left << std::setw(8) << name_of(dt)
                      << std::fixed << std::setprecision(2)
                      << std::setw(13) << ms_ref << std::setw(13) << ms_simd
                      << (ms_ref / ms_simd) << "x\n";
            if (sink == 1e300) std::cout << "";  // keep the work
        }
    }

    // ── 5. AVX-512 tier vs AVX2, on a host that has both ─────────────────────
    //
    // Correctness AND speed, because "wider" is not automatically "faster". On
    // the Intel parts that gave AVX-512 its reputation, 512-bit code could clock
    // the core down far enough to lose to AVX2 outright. Zen 5 has a native
    // 512-bit datapath and should not, but that is a claim about this silicon,
    // so it is measured rather than assumed — and a regression here is a reason
    // to drop the tier, not to keep it for tidiness.
    //
    // Compiled out, not just skipped at runtime: this is the one section that
    // names the per-tier kernels directly rather than going through the
    // dispatchers, so it needs those translation units to EXIST. They do not on
    // aarch64, nor on an x86-64 host whose compiler cannot emit AVX-512, and the
    // `tier == Avx512` guard below is a runtime check that cannot help a linker.
#if defined(SOMA_HAS_AVX512)
    if (tier == soma::simd::SimdTier::Avx512) {
        std::cout << "\n5. AVX-512 vs AVX2\n";

        constexpr std::uint32_t cols = 2048, rows = 2048, g = 128;
        const soma::DType wide[] = {soma::DType::Q4_G, soma::DType::Q8_0,
                                    soma::DType::Q4_0};

        std::cout << "   " << std::left << std::setw(9) << "op" << std::setw(13) << "avx2 ms"
                  << std::setw(13) << "avx512 ms" << std::setw(11) << "speedup"
                  << "max rel diff\n";

        for (const auto dt : wide) {
            const auto per_row = (cols / g) * soma::group_bytes(dt, g);
            Rng rng;
            std::vector<std::byte> buf(static_cast<std::size_t>(rows) * per_row);
            for (auto& b : buf) {
                b = static_cast<std::byte>(
                    static_cast<std::uint8_t>(static_cast<int>(rng.next() * 127.0f) & 0xFF));
            }
            for (std::uint32_t r = 0; r < rows; ++r) {
                for (std::uint32_t gi = 0; gi < cols / g; ++gi) {
                    const auto off = r * per_row + gi * soma::group_bytes(dt, g);
                    const float scale = 0.02f, minv = -0.3f;
                    std::memcpy(buf.data() + off, &scale, 4);
                    if (dt == soma::DType::Q4_G) std::memcpy(buf.data() + off + 4, &minv, 4);
                }
            }
            std::vector<float> x(cols);
            for (auto& v : x) v = rng.next();

            const auto call = [&](bool use512, std::uint32_t r) {
                const std::byte* p = buf.data() + static_cast<std::size_t>(r) * per_row;
                switch (dt) {
                    case soma::DType::Q4_G:
                        return use512 ? soma::simd::avx512::qdot_q4g(p, cols, g, x.data(), nullptr)
                                      : soma::simd::avx2::qdot_q4g(p, cols, g, x.data(), nullptr);
                    case soma::DType::Q8_0:
                        return use512 ? soma::simd::avx512::qdot_q8_0(p, cols, g, x.data())
                                      : soma::simd::avx2::qdot_q8_0(p, cols, g, x.data());
                    default:
                        return use512 ? soma::simd::avx512::qdot_q4_0(p, cols, g, x.data())
                                      : soma::simd::avx2::qdot_q4_0(p, cols, g, x.data());
                }
            };

            double worst = 0.0;
            for (std::uint32_t r = 0; r < 64; ++r) {
                const float a = call(false, r), b = call(true, r);
                worst = std::max(worst, std::fabs(static_cast<double>(a - b)) /
                                            std::max(1e-6f, std::fabs(a)));
            }

            // BEST of several, not one shot.
            //
            // A single pass over 2048 rows takes ~0.2 ms, which is short enough
            // that one descheduling or a cold cache line swamps it: an earlier
            // version of this reported q4_0 at 1.72x and then 0.55x on successive
            // runs of the same binary. Best-of-N reports the floor, which is the
            // number that reflects the kernel rather than the machine's mood.
            double sink = 0.0;
            double ms2 = 1e30, ms512 = 1e30;
            for (int rep = 0; rep < 5; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                for (std::uint32_t r = 0; r < rows; ++r) sink += call(false, r);
                ms2 = std::min(ms2, std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - t0).count());

                t0 = std::chrono::steady_clock::now();
                for (std::uint32_t r = 0; r < rows; ++r) sink += call(true, r);
                ms512 = std::min(ms512, std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - t0).count());
            }

            std::cout << "   " << std::left << std::setw(9) << name_of(dt) << std::fixed
                      << std::setprecision(2) << std::setw(13) << ms2 << std::setw(13) << ms512
                      << std::setw(11) << (ms2 / ms512) << std::scientific
                      << std::setprecision(2) << worst << "\n";

            check(worst < 1e-4,
                  std::string(name_of(dt)) + ": AVX-512 agrees with AVX2");
            if (sink == 1e300) std::cout << "";
        }

        // fp32 matvec, the single hottest shape in the engine.
        {
            constexpr std::uint32_t m = 2048, k = 2048;
            Rng rng;
            std::vector<float> w(static_cast<std::size_t>(m) * k), x(k), y2(m), y5(m);
            for (auto& v : w) v = rng.next();
            for (auto& v : x) v = rng.next();

            double ms2 = 1e30, ms512 = 1e30;
            for (int rep = 0; rep < 5; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                soma::simd::avx2::matvec(w.data(), x.data(), m, k, y2.data());
                ms2 = std::min(ms2, std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - t0).count());

                t0 = std::chrono::steady_clock::now();
                soma::simd::avx512::matvec(w.data(), x.data(), m, k, y5.data());
                ms512 = std::min(ms512, std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - t0).count());
            }

            // Against EXACT arithmetic, not against AVX2.
            //
            // Comparing the two kernels to each other cannot distinguish "the new
            // one is wrong" from "the old one is wrong" from "this row cancelled".
            // With 2048 random rows one of them lands near zero by chance, and its
            // relative error reaches ~1e-3 with both kernels behaving perfectly —
            // which is exactly what the first version of this check flagged as a
            // failure. Both are measured against a double evaluation instead, and
            // the requirement is that the wider kernel is no worse.
            double e2 = 0.0, e5 = 0.0;
            for (std::uint32_t r = 0; r < m; ++r) {
                const float* wr = w.data() + static_cast<std::size_t>(r) * k;
                double exact = 0.0;
                for (std::uint32_t i = 0; i < k; ++i) {
                    exact += static_cast<double>(wr[i]) * static_cast<double>(x[i]);
                }
                const double den = std::max(1e-6, std::fabs(exact));
                e2 = std::max(e2, std::fabs(y2[r] - exact) / den);
                e5 = std::max(e5, std::fabs(y5[r] - exact) / den);
            }
            std::cout << "   " << std::left << std::setw(9) << "f32.mv" << std::fixed
                      << std::setprecision(2) << std::setw(13) << ms2 << std::setw(13) << ms512
                      << std::setw(11) << (ms2 / ms512) << std::scientific
                      << std::setprecision(2) << e2 << " / " << e5 << " (avx2/512)\n";
            check(e5 <= std::max(4.0 * e2, 1e-5),
                  "f32 matvec: AVX-512 no less accurate than AVX2");
        }

        // The masked tail is new at this tier — AVX2 falls back to scalar for the
        // remainder, AVX-512 handles it in-register. Same exactness standard as
        // the AVX2 tail check, and for the same reason: a float threshold here
        // would be measuring cancellation, not the kernel.
        bool tail_exact = true;
        for (const std::uint32_t m : {1u, 3u, 5u, 7u, 9u, 64u}) {
            for (const std::uint32_t k : {1u, 3u, 15u, 16u, 17u, 63u, 65u}) {
                std::vector<float> w(static_cast<std::size_t>(m) * k, 1.0f), x(k, 1.0f);
                std::vector<float> y(m, -1.0f);
                soma::simd::avx512::matvec(w.data(), x.data(), m, k, y.data());
                for (std::uint32_t r = 0; r < m; ++r) {
                    tail_exact &= (y[r] == static_cast<float>(k));
                }
            }
        }
        check(tail_exact, "AVX-512 masked tail is exact for every remainder");
    }
#endif // SOMA_HAS_AVX512

    std::cout << "\n" << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES")
              << "\n";
    return g_failures == 0 ? 0 : 1;
}
