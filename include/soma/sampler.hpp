#pragma once

// Soma — token sampling.
//
// ── The property that makes sampling safe to batch ───────────────────────────
//
// The RNG state lives in SamplerState, which is PER SEQUENCE. That is not a
// convenience; it is the reason a sampled sequence produces the same tokens
// whether it runs alone or beside seven others.
//
// A single engine-wide RNG would work perfectly in every single-sequence test
// and be wrong the moment two sequences shared a step: each draw would depend on
// how many other sequences happened to draw first, so a sequence's output would
// become a function of its neighbours. That is exactly the failure the
// batched-equals-solo gate exists to catch, and it is invisible until there is
// concurrency.
//
// The PRNG is specified here rather than taken from <random> because
// std::uniform_real_distribution is not required to produce the same values
// across implementations. A seed is part of a request, and a request that
// replays differently on another host is not reproducible.

#include "soma/model.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace soma {

/// splitmix64. Fully specified, seedable, and identical everywhere.
inline std::uint64_t rng_next(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// Uniform in [0, 1), from the top 24 bits — exactly representable in a float,
/// so the conversion introduces no rounding of its own.
inline float rng_uniform(std::uint64_t& state) noexcept {
    return static_cast<float>(rng_next(state) >> 40) * (1.0f / 16777216.0f);
}

/// Scratch reused across calls so the sampler allocates nothing per token.
struct SamplerScratch {
    std::vector<std::uint32_t> idx;
    std::vector<float> prob;
};

/// Pick one token from `logits`.
///
/// Stages apply in this order, and the order is load-bearing:
///
///   1. repeat / presence penalties   on RAW logits, before any scaling, so the
///                                    penalty's magnitude does not depend on
///                                    temperature
///   2. temperature                   <= 0 means GREEDY: argmax, no RNG consumed
///   3. top-k                         keep the k highest
///   4. top-p (nucleus)               keep the smallest prefix reaching p
///   5. min-p                         drop anything below min_p x max_prob
///   6. renormalise and draw
///
/// `history` feeds the penalties; pass the sequence's emitted tokens.
///
/// Ties break toward the LOWER token id at every stage, so a distribution with
/// equal probabilities still samples reproducibly.
TokenId sample_token(std::span<const float> logits,
                     SamplerState& sampler,
                     std::span<const TokenId> history,
                     SamplerScratch& scratch) noexcept;

} // namespace soma
