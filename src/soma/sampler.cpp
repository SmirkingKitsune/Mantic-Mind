// Soma — token sampling. See sampler.hpp for why the RNG is per sequence.

#include "soma/sampler.hpp"

#include <algorithm>
#include <cmath>

namespace soma {

bool logits_are_finite(std::span<const float> logits) noexcept {
    return std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); });
}

TokenId sample_token(std::span<const float> logits,
                     SamplerState& s,
                     std::span<const TokenId> history,
                     SamplerScratch& sc) noexcept {
    const auto n = static_cast<std::uint32_t>(logits.size());
    if (n == 0) return 0;

    // ── 1. penalties, on raw logits ──────────────────────────────────────────
    //
    // Applied before temperature so a penalty means the same thing at every
    // temperature. Scaling first would make `repeat_penalty` a function of
    // `temperature`, which is a surprising coupling to debug.
    sc.prob.assign(logits.begin(), logits.end());
    const bool penalise =
        (s.repeat_penalty > 0.0f && s.repeat_penalty != 1.0f) || s.presence_penalty != 0.0f;
    if (penalise && !history.empty()) {
        for (const auto t : history) {
            if (t >= n) continue;
            float& v = sc.prob[t];
            if (s.repeat_penalty > 0.0f && s.repeat_penalty != 1.0f) {
                // Divide when positive, multiply when negative — dividing a
                // negative logit would REWARD the token it is meant to suppress.
                v = (v > 0.0f) ? (v / s.repeat_penalty) : (v * s.repeat_penalty);
            }
            v -= s.presence_penalty;
        }
    }

    // ── 2. temperature, or greedy ────────────────────────────────────────────
    if (s.temperature <= 0.0f) {
        // Greedy consumes no randomness at all. A caller that flips temperature
        // to 0 mid-stream must not find the RNG stream shifted underneath it.
        std::uint32_t best = 0;
        for (std::uint32_t i = 1; i < n; ++i) {
            if (sc.prob[i] > sc.prob[best]) best = i;
        }
        return static_cast<TokenId>(best);
    }
    const float inv_t = 1.0f / s.temperature;
    for (auto& v : sc.prob)
        v *= inv_t;

    // Candidate set, ordered by logit descending and by token id ascending on
    // ties — the tie rule that makes an equal-probability distribution still
    // sample reproducibly.
    sc.idx.resize(n);
    for (std::uint32_t i = 0; i < n; ++i)
        sc.idx[i] = i;

    const auto keep_k =
        (s.top_k > 0) ? std::min<std::uint32_t>(static_cast<std::uint32_t>(s.top_k), n) : n;
    const auto by_logit = [&](std::uint32_t a, std::uint32_t b) {
        if (sc.prob[a] != sc.prob[b]) return sc.prob[a] > sc.prob[b];
        return a < b;
    };

    // ── 3. top-k ─────────────────────────────────────────────────────────────
    if (keep_k < n) {
        std::partial_sort(sc.idx.begin(), sc.idx.begin() + keep_k, sc.idx.end(), by_logit);
        sc.idx.resize(keep_k);
    } else {
        std::sort(sc.idx.begin(), sc.idx.end(), by_logit);
    }

    // Softmax over the survivors, max-subtracted.
    const float mx = sc.prob[sc.idx.front()];
    float sum = 0.0f;
    std::vector<float> p;
    p.reserve(sc.idx.size());
    for (const auto i : sc.idx) {
        const float e = std::exp(sc.prob[i] - mx);
        p.push_back(e);
        sum += e;
    }
    if (sum <= 0.0f) return static_cast<TokenId>(sc.idx.front());
    for (auto& v : p)
        v /= sum;

    // ── 4. top-p, on the already-sorted survivors ────────────────────────────
    std::size_t keep = p.size();
    if (s.top_p > 0.0f && s.top_p < 1.0f) {
        float acc = 0.0f;
        keep = 0;
        for (std::size_t i = 0; i < p.size(); ++i) {
            acc += p[i];
            ++keep;
            if (acc >= s.top_p) break;
        }
    }

    // ── 5. min-p, relative to the mode ───────────────────────────────────────
    if (s.min_p > 0.0f) {
        const float floor_p = s.min_p * p.front();
        std::size_t m = 0;
        while (m < keep && p[m] >= floor_p)
            ++m;
        // At least one candidate always survives: an empty set has nothing to
        // draw from, and silently returning token 0 would look like the model
        // emitting padding.
        keep = std::max<std::size_t>(1, m);
    }

    // ── 6. renormalise and draw ──────────────────────────────────────────────
    float kept = 0.0f;
    for (std::size_t i = 0; i < keep; ++i)
        kept += p[i];
    if (kept <= 0.0f) return static_cast<TokenId>(sc.idx.front());

    const float r = rng_uniform(s.rng_state) * kept;
    float acc = 0.0f;
    for (std::size_t i = 0; i < keep; ++i) {
        acc += p[i];
        if (r < acc) return static_cast<TokenId>(sc.idx[i]);
    }
    // Reachable only through float rounding when r lands within an ulp of the
    // total; the last candidate is the correct answer there.
    return static_cast<TokenId>(sc.idx[keep - 1]);
}

} // namespace soma
