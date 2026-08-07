#pragma once

// Soma — the fp32 conformance comparison, as a library function.
//
// This is admission ladder STAGE 1: teacher-forced logits against a
// `transformers` oracle, plus greedy token-exactness, on a tiny-random model
// carrying the REAL architecture.
//
// It lives here rather than inside tests/soma/conformance_g0.cpp because it now
// has two callers — that test, and `soma conform`, which the admission pipeline
// runs. Two implementations of one comparison would be two sets of tolerances,
// two oracle parsers, and two opinions about what "passes" means; the ladder's
// whole job is to have one.
//
// What it validates is worth being exact about: the tiny model is RANDOM, so
// this says nothing about the admitted weights. It says the engine implements
// this ARCHITECTURE the way `transformers` does. A real checkpoint can be
// approximately right in ways that hide a bug for weeks — a tiny-random one is
// either exactly right or obviously wrong, which is why the fixture exists.

#include "soma/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace soma {

/// Tolerances for the teacher-forced comparison.
///
/// The engine and torch accumulate the same sums in different orders, so exact
/// bit-equality is not a reasonable bar for fp32 — but the bar still has to be
/// tight enough to catch a wrong RoPE pairing or a mis-scaled expert, both of
/// which move logits far more than reassociation does.
inline constexpr float kConformanceMaxAbsDiff = 2e-3f;
inline constexpr float kConformanceMaxMeanDiff = 2e-4f;

/// A `SOMAORCL` fixture, as written by tools/admission/make_oracle.py.
struct OracleFixture {
    std::uint32_t positions = 0;
    std::uint32_t vocab = 0;
    std::vector<std::int32_t> input_ids;
    std::vector<float> tf_logits;
    std::vector<std::int32_t> greedy_prefix;
    std::vector<std::int32_t> greedy_tokens;
};

/// Read the flat sidecar. `.npz` is a zip of `.npy` members, so reading it from
/// C++ would mean zlib plus an npy parser to move numbers that were never
/// compressible; the Python consumers keep the `.npz` and the engine reads this.
Status read_oracle_fixture(const std::string& path, OracleFixture& out);

struct Fp32ConformanceResult {
    bool loaded = false;
    bool logits_pass = false;
    bool greedy_pass = false;

    /// An architecture with no backend is a GAP IN COVERAGE, not a failure of
    /// the engine — reporting it red would make the ladder permanently red and
    /// therefore ignored. The caller decides whether a gap blocks it.
    bool skipped = false;
    std::string detail;

    float max_abs = 0.0f;
    float mean_abs = 0.0f;

    /// The divergence at position 0, reported because it BISECTS the search.
    ///
    /// At t=0 RoPE is the identity and attention is a softmax over one element,
    /// so a divergence already present at 0 cannot be either — it is projection,
    /// qk-norm, routing, or the expert MLP. One that is clean at 0 and grows
    /// with t is the opposite. That single number is what turned four rounds of
    /// failed reasoning about MLA into one afternoon; see the G4 section.
    float max_abs_pos0 = 0.0f;
    std::uint32_t max_at_pos = 0;

    std::uint32_t first_bad_token = 0;
    std::uint32_t matched_tokens = 0;

    bool passed() const noexcept { return loaded && logits_pass && greedy_pass; }
};

/// Load `model_dir` as fp32, run it against `model_dir/oracle.bin`, and report.
///
/// `model_dir` is a TINY-RANDOM fixture directory — config.json, weights, and
/// the oracle — not a converted container.
Fp32ConformanceResult run_fp32_conformance(const std::string& model_dir);

} // namespace soma
