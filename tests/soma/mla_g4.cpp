// Soma — G4: MLA's positional path, checked against the reference formulas.
//
// mla.cpp applies RoPE to the rope segment in INTERLEAVED layout and leaves it
// there. The reference (DeepseekV2 apply_rotary_pos_emb) first de-interleaves —
// `view(d/2, 2).transpose(4, 3)` — and then applies the rotate-half form, so its
// output is in half-split layout.
//
// The claim in mla.cpp is that this does not matter: q and k receive the SAME
// permutation, and a dot product is invariant under a permutation applied to
// both operands. That argument is correct on paper and was re-derived twice —
// which is exactly the kind of reasoning that should be checked numerically
// rather than trusted, because if it is wrong the output is finite, plausible,
// and wrong only at positions past zero.
//
// Usage: mla_g4

#include "soma/kernels_f32.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << "   " << std::left << std::setw(52) << what << (ok ? "OK" : "FAIL");
    if (!detail.empty()) std::cout << "   " << detail;
    std::cout << "\n";
    if (!ok) ++g_failures;
}

struct Rng {
    std::uint32_t s = 0x2468ACE1u;
    float next() noexcept {
        s = s * 1664525u + 1013904223u;
        return (static_cast<float>(s >> 8) / 8388608.0f) - 1.0f;
    }
};

/// The reference: de-interleave, then rotate-half against cat(freqs, freqs).
std::vector<float> reference_rope(const std::vector<float>& x, std::uint32_t pos,
                                  const std::vector<float>& inv_freq) {
    const auto d = static_cast<std::uint32_t>(x.size());
    const auto h = d / 2;

    // view(d/2, 2).transpose -> even indices first, then odd.
    std::vector<float> y(d);
    for (std::uint32_t i = 0; i < h; ++i) {
        y[i] = x[2 * i];
        y[h + i] = x[2 * i + 1];
    }

    // emb = cat(freqs, freqs), so cos[i] == cos[h + i].
    std::vector<float> c(d), s(d);
    for (std::uint32_t i = 0; i < h; ++i) {
        const float a = static_cast<float>(pos) * inv_freq[i];
        c[i] = c[h + i] = std::cos(a);
        s[i] = s[h + i] = std::sin(a);
    }

    // out = y*cos + rotate_half(y)*sin,  rotate_half(y) = cat(-y[h:], y[:h])
    std::vector<float> out(d);
    for (std::uint32_t i = 0; i < h; ++i) {
        out[i] = y[i] * c[i] + (-y[h + i]) * s[i];
        out[h + i] = y[h + i] * c[h + i] + y[i] * s[h + i];
    }
    return out;
}

/// What mla.cpp does: rotate the interleaved pairs in place.
std::vector<float> engine_rope(const std::vector<float>& x, std::uint32_t pos,
                               const std::vector<float>& inv_freq) {
    std::vector<float> v = x;
    const auto h = static_cast<std::uint32_t>(x.size()) / 2;
    for (std::uint32_t i = 0; i < h; ++i) {
        const float a = static_cast<float>(pos) * inv_freq[i];
        const float c = std::cos(a), s = std::sin(a);
        const float p = v[2 * i], q = v[2 * i + 1];
        v[2 * i] = p * c - q * s;
        v[2 * i + 1] = q * c + p * s;
    }
    return v;
}

float dot(const std::vector<float>& a, const std::vector<float>& b) {
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) acc += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(acc);
}

}  // namespace

int main() {
    // DeepSeek-V2-Lite's rope segment.
    constexpr std::uint32_t kDim = 8;
    const std::vector<float> inv_freq{1.0f, 0.1f, 0.005125f, 0.000025f};

    std::cout << "1. rotation layouts produce the same q.k\n";
    {
        Rng rng;
        double worst_dot = 0.0, worst_elem = 0.0;
        bool layouts_differ = false;

        // Several (query, key) position PAIRS, not just q and k at the same
        // position: attention pairs a query at t with keys at every j <= t, and
        // an equivalence that only held for t == j would pass a lazier test and
        // fail in the model.
        for (std::uint32_t t = 0; t < 40; ++t) {
            for (std::uint32_t j = 0; j <= t; j += 7) {
                std::vector<float> q(kDim), k(kDim);
                for (auto& v : q) v = rng.next();
                for (auto& v : k) v = rng.next();

                const auto qr = reference_rope(q, t, inv_freq);
                const auto kr = reference_rope(k, j, inv_freq);
                const auto qe = engine_rope(q, t, inv_freq);
                const auto ke = engine_rope(k, j, inv_freq);

                // The layouts really are different — otherwise this test proves
                // nothing about the permutation argument.
                for (std::uint32_t i = 0; i < kDim; ++i) {
                    layouts_differ |= (std::fabs(qr[i] - qe[i]) > 1e-6f);
                    worst_elem = std::max(worst_elem,
                                          std::fabs(static_cast<double>(qr[i] - qe[i])));
                }

                const float dr = dot(qr, kr);
                const float de = dot(qe, ke);
                worst_dot = std::max(worst_dot, std::fabs(static_cast<double>(dr - de)));
            }
        }

        check(layouts_differ, "the two layouts really do differ elementwise",
              "max elem diff " + std::to_string(worst_elem));
        check(worst_dot < 1e-5, "but the dot products agree",
              "max |dot diff| " + std::to_string(worst_dot));
    }

    // The permutation argument only holds if BOTH operands get it. If only the
    // query were rotated in one layout and the key in the other, the dot product
    // would diverge — which is what a partial application of this optimisation
    // would look like.
    std::cout << "\n2. mixing layouts DOES break it (the argument is load-bearing)\n";
    {
        Rng rng;
        std::vector<float> q(kDim), k(kDim);
        for (auto& v : q) v = rng.next();
        for (auto& v : k) v = rng.next();

        const auto qr = reference_rope(q, 11, inv_freq);
        const auto ke = engine_rope(k, 5, inv_freq);
        const auto qe = engine_rope(q, 11, inv_freq);
        const auto kr = reference_rope(k, 5, inv_freq);

        const float matched = dot(qr, kr);
        const float mixed = dot(qr, ke);
        check(std::fabs(matched - mixed) > 1e-3f,
              "one operand in each layout gives a different answer",
              "matched " + std::to_string(matched) + " vs mixed " + std::to_string(mixed));
        (void)qe;
    }

    std::cout << "\n" << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES")
              << "\n";
    return g_failures == 0 ? 0 : 1;
}
