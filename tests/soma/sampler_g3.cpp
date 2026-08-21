// Soma — G3: token sampling.
//
// Two things are being checked, and the second is the one that is easy to skip.
//
//   1. The stages do what they say: greedy, top-k, top-p, min-p, penalties.
//   2. THE DISTRIBUTION IS ACTUALLY RIGHT. A sampler that returns plausible
//      tokens from a subtly wrong distribution passes every structural test —
//      the tokens are all in-vocabulary, high-probability ones come up often —
//      and quietly degrades output quality in a way no assertion catches. So the
//      empirical frequencies are compared against the softmax they claim to be.
//
// Usage: sampler_g3

#include "soma/sampler.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << "   " << std::left << std::setw(50) << what << (ok ? "OK" : "FAIL");
    if (!detail.empty()) std::cout << "   " << detail;
    std::cout << "\n";
    if (!ok) ++g_failures;
}

soma::SamplerState greedy() {
    soma::SamplerState s;
    s.temperature = 0.0f;
    return s;
}

soma::SamplerState plain(std::uint64_t seed) {
    soma::SamplerState s;
    s.temperature = 1.0f;
    s.top_p = 1.0f;
    s.top_k = -1;
    s.min_p = -1.0f;
    s.repeat_penalty = -1.0f;
    s.presence_penalty = 0.0f;
    s.rng_state = seed;
    return s;
}

}  // namespace

int main() {
    soma::SamplerScratch sc;
    const std::vector<soma::TokenId> none;

    // The scheduler relies on this guard before sampling. Without it NaN can
    // quietly turn into token zero in greedy mode, which makes a broken launch
    // look like valid generation.
    const std::vector<float> finite_logits{1.0f, -2.0f, 0.0f};
    const std::vector<float> nan_logits{1.0f, std::numeric_limits<float>::quiet_NaN()};
    const std::vector<float> inf_logits{1.0f, std::numeric_limits<float>::infinity()};
    check(soma::logits_are_finite(finite_logits),
          "finite logits pass the runtime guard");
    check(!soma::logits_are_finite(nan_logits),
          "NaN logits fail the runtime guard");
    check(!soma::logits_are_finite(inf_logits),
          "infinite logits fail the runtime guard");

    // ── 1. greedy ────────────────────────────────────────────────────────────
    std::cout << "1. greedy\n";
    {
        const std::vector<float> lg{0.1f, 3.0f, 0.5f, 2.9f};
        auto s = greedy();
        const auto before = s.rng_state;
        const auto t = soma::sample_token(lg, s, none, sc);
        check(t == 1, "temperature 0 returns the argmax", "got " + std::to_string(t));
        check(s.rng_state == before, "greedy consumes no randomness",
              "so toggling temperature does not shift the stream");

        auto k1 = plain(42);
        k1.top_k = 1;
        check(soma::sample_token(lg, k1, none, sc) == 1, "top_k = 1 is greedy");
    }

    // ── 2. reproducibility ───────────────────────────────────────────────────
    std::cout << "\n2. reproducibility\n";
    {
        const std::vector<float> lg{1.0f, 1.1f, 0.9f, 1.05f, 0.8f};
        const auto draw = [&](std::uint64_t seed) {
            auto s = plain(seed);
            std::vector<soma::TokenId> out;
            for (int i = 0; i < 24; ++i) out.push_back(soma::sample_token(lg, s, none, sc));
            return out;
        };
        const auto a = draw(12345), b = draw(12345), c = draw(999);
        check(a == b, "same seed gives the same sequence");
        check(a != c, "a different seed gives a different sequence",
              "otherwise the seed is being ignored");

        // Near-uniform logits, so a sampler that quietly fell back to argmax
        // would emit one token 24 times.
        std::map<soma::TokenId, int> hist;
        for (const auto t : a) ++hist[t];
        check(hist.size() > 1, "sampling actually varies",
              std::to_string(hist.size()) + " distinct tokens in 24 draws");
    }

    // ── 3. the distribution ──────────────────────────────────────────────────
    //
    // The check that a structural test cannot substitute for.
    std::cout << "\n3. empirical frequency vs softmax\n";
    {
        const std::vector<float> lg{std::log(0.5f), std::log(0.3f), std::log(0.2f)};
        auto s = plain(0xC0FFEE);
        constexpr int kDraws = 200000;
        std::vector<int> count(3, 0);
        for (int i = 0; i < kDraws; ++i) ++count[soma::sample_token(lg, s, none, sc)];

        const float want[3] = {0.5f, 0.3f, 0.2f};
        double worst = 0.0;
        for (int i = 0; i < 3; ++i) {
            const double got = static_cast<double>(count[i]) / kDraws;
            worst = std::max(worst, std::fabs(got - want[i]));
            std::cout << "   token " << i << "  want " << std::fixed << std::setprecision(3)
                      << want[i] << "  got " << got << "\n";
        }
        // 200k draws puts the standard error near 0.001, so 0.01 is ~10 sigma —
        // loose enough never to flake, tight enough that an off-by-one in the
        // cumulative walk (which shifts mass by a whole category) cannot hide.
        check(worst < 0.01, "frequencies match the softmax",
              "max deviation " + std::to_string(worst));
    }

    // ── 4. truncation ────────────────────────────────────────────────────────
    std::cout << "\n4. top-k / top-p / min-p truncate\n";
    {
        const std::vector<float> lg{5.0f, 4.0f, 0.0f, -5.0f, -20.0f};

        auto k2 = plain(7);
        k2.top_k = 2;
        bool only_top2 = true;
        for (int i = 0; i < 400; ++i) {
            const auto t = soma::sample_token(lg, k2, none, sc);
            only_top2 &= (t == 0 || t == 1);
        }
        check(only_top2, "top_k = 2 never returns a third token");

        auto p = plain(7);
        p.top_p = 0.6f;   // token 0 alone is ~0.71 of the mass
        bool only_first = true;
        for (int i = 0; i < 400; ++i) only_first &= (soma::sample_token(lg, p, none, sc) == 0);
        check(only_first, "top_p below the mode's mass keeps only the mode");

        // Degenerate settings must still yield a token. An empty candidate set
        // returning 0 would look like the model emitting padding.
        auto m = plain(7);
        m.min_p = 1.5f;   // above 1.0: nothing can clear the bar
        const auto t = soma::sample_token(lg, m, none, sc);
        check(t == 0, "an impossible min_p still returns the mode, not nothing",
              "got " + std::to_string(t));
    }

    // ── 5. penalties ─────────────────────────────────────────────────────────
    std::cout << "\n5. repetition and presence penalties\n";
    {
        // Token 0 leads by a hair; penalising it must hand the lead to token 1.
        const std::vector<float> lg{2.0f, 1.9f, 0.0f};
        const std::vector<soma::TokenId> hist{0, 0, 0};

        auto g = greedy();
        check(soma::sample_token(lg, g, none, sc) == 0, "unpenalised, token 0 leads");

        auto rp = greedy();
        rp.repeat_penalty = 2.0f;
        check(soma::sample_token(lg, rp, hist, sc) == 1,
              "repeat_penalty demotes a repeated token");

        auto pp = greedy();
        pp.presence_penalty = 0.5f;
        check(soma::sample_token(lg, pp, hist, sc) == 1,
              "presence_penalty demotes a seen token");

        // A negative logit divided by a penalty > 1 moves TOWARD zero, i.e. up —
        // rewarding exactly the token being suppressed. The sign branch exists
        // for this case, so it is the case worth testing.
        const std::vector<float> neg{-1.0f, -1.1f, -9.0f};
        auto rn = greedy();
        rn.repeat_penalty = 4.0f;
        const auto t = soma::sample_token(neg, rn, {&hist[0], 1}, sc);
        check(t == 1, "penalising a NEGATIVE logit still demotes it",
              "got " + std::to_string(t) + " (0 would mean the penalty inverted)");
    }

    std::cout << "\n" << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES")
              << "\n";
    return g_failures == 0 ? 0 : 1;
}
