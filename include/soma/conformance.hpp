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

// ── stage 2: real_logit_kl ───────────────────────────────────────────────────
//
// Stage 1 demands token-exactness on a tiny-RANDOM model, which validates the
// architecture and says nothing about the weights an operator ships. Stage 2 is
// the other half: the REAL checkpoint, quantized, against a bf16 reference pass
// over the same checkpoint.
//
// It cannot demand exactness — a quantized engine does not reproduce a bf16
// reference bit-for-bit — so the bar is DISTRIBUTIONAL. That separation is
// load-bearing: failing stage 2 while stage 1 passes is a QUANTIZATION finding,
// and the remedy is to requantize a role or raise group granularity, not to
// debug a kernel. Reporting them as one failure costs days.

/// Mean and p95 KL, in nats.
///
/// 0.05 mean is a distribution the engine reproduces closely enough that
/// sampling from it is not meaningfully different. The p95 exists because a mean
/// alone hides a handful of catastrophic positions among many good ones, and a
/// model that is confidently wrong twenty times in five hundred is not usable
/// even with a flattering average.
inline constexpr double kRealLogitKlMeanMax = 0.05;
inline constexpr double kRealLogitKlP95Max = 0.25;

struct RealLogitKlResult {
    bool loaded = false;
    bool passed = false;
    /// Distinguished from a failure: no reference fixture is missing EVIDENCE,
    /// not evidence of a bad model, and must never read as a reject.
    bool skipped = false;
    std::string detail;

    std::uint32_t positions = 0;
    std::uint32_t vocab = 0;
    double mean_kl = 0.0;
    double median_kl = 0.0;
    double p95_kl = 0.0;
    double worst_kl = 0.0;
    std::uint32_t worst_at = 0;

    /// How often the engine's argmax equals the reference's. Reported alongside
    /// KL because they fail differently: KL can be small while top-1 drifts on
    /// near-ties, and top-1 can look fine while the tail is badly mis-shaped.
    double top1_agreement_pct = 0.0;

    /// Set when the output is not merely degraded but broken — KL near what a
    /// UNIFORM distribution would score, or top-1 agreement at chance. Worth its
    /// own flag because the remedy differs: a degenerate result usually means a
    /// wrong quant map or a corrupt container, not a granularity that needs
    /// raising.
    bool degenerate = false;

    double cache_hit_rate_pct = 0.0;
    double forward_seconds = 0.0;

    /// Streaming cost on a REAL checkpoint. Not part of the pass rule — a model
    /// can be faithful and slow — but reported because bytes/token is the
    /// quantity the streamable verdict rests on, and this is the only place it
    /// is measured against real weights rather than estimated from headers.
    std::uint64_t cache_misses = 0;
    std::uint64_t cache_evictions = 0;
    std::uint64_t bytes_read = 0;
    /// The batch union: how many distinct experts were actually read versus how
    /// many a per-row loop would have. Same claim as bytes/token, measured twice.
    std::uint64_t unique_expert_reads = 0;
    std::uint64_t naive_expert_reads = 0;
};

/// Run the QUANTIZED container against a bf16 reference fixture.
///
/// `container_dir` is a converted container; the quant map is read from its own
/// `container_meta.json` rather than assumed, because admission accepts a
/// `QuantOverride` and a hardcoded map would silently dequantize with the wrong
/// one — measuring a model nobody is going to run.
///
/// `reference_path` is a `SOMAORCL` file from tools/admission/make_reference.py.
/// Same format as the tiny oracles, so there is one reader.
RealLogitKlResult run_real_logit_kl(const std::string& container_dir,
                                    const std::string& reference_path,
                                    std::uint64_t cache_gib = 8,
                                    std::uint32_t max_positions = 0);

} // namespace soma
