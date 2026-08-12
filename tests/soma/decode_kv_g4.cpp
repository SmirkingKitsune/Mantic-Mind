// Soma — the cached-decode path must agree with the teacher-forced one.
//
// Two entry points compute the same thing by different routes. `f32_attention`
// owns the whole sequence and recomputes every key; `f32_attention_kv` owns one
// position per row and reads the rest from a cache. If they disagree, generation
// diverges from the conformance run that certified the model — and conformance
// cannot catch it, because `generate_greedy_f32` re-runs the PREFILL each step
// and never touches the cached path at all.
//
// That gap is why MLA's cached decode could sit as a stub returning Unsupported
// while every gate stayed green.
//
// The check: feed a prompt through the teacher-forced forward, then feed the same
// prompt one token at a time through the cached forward, and require the logits
// at every position to match. Not just the last — a cache that is wrong only for
// early positions still produces a correct final row when the last token
// dominates, and "the last logit matched" is exactly the observation that would
// hide it.
//
// Runs over every tiny fixture, so GQA is covered by the same assertion as
// MLA and MLA+DSA rather than by a separate one that could drift.
//
// Usage: decode_kv_g4 [fixture_root]

#include "soma/f32_model.hpp"
#include "soma/kv_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

/// fp32 accumulation order differs between the two paths — one matmuls a whole
/// sequence, the other one row — so bit-equality is not the bar. This is the
/// same order of tolerance the conformance gate uses for logits.
constexpr float kTol = 1e-3f;

struct Outcome {
    bool ran = false;
    bool skipped = false;
    bool passed = false;
    float max_abs = 0.0f;
    std::uint32_t worst_pos = 0;
    double decode_ms = 0.0; ///< the cached loop ONLY; the teacher-forced run is shared
    std::string note;
};

Outcome check_fixture(const fs::path& dir, std::uint32_t n_tokens, bool absorb) {
    Outcome r;

    soma::F32Model model;
    if (const auto st = soma::load_f32_model(dir.string(), model); !st.ok()) {
        r.note = st.message();
        r.skipped = st.code() == soma::StatusCode::Unsupported;
        r.ran = !r.skipped;
        return r;
    }

    // MLA's absorbed decode reaches the same numbers by different arithmetic:
    // `(W_k^T q) . c` instead of `q . (W_k c)`, and one projection of the
    // accumulated latent instead of a projection per attended key. Both must
    // agree with the teacher-forced path, which is precisely the A/B
    // `MlaSpec::absorb_weights` was documented for and never used for.
    //
    // Running only the default would leave the OTHER path untested — and the
    // unabsorbed one is the reference the fast one was checked against, so
    // letting it rot would quietly remove the thing that makes absorption
    // believable.
    model.arch.attention.mla.absorb_weights = absorb;
    // From this point on, every failure belongs to a family the engine claims to
    // support and must fail the gate. The original draft treated all early
    // returns as skips, which would have let MLA regress back to Unsupported while
    // the GQA fixtures kept the test green — exactly the gap this test exists to
    // close.
    r.ran = true;

    const auto vocab = model.arch.topology.vocab_size;
    std::vector<soma::TokenId> tokens(n_tokens);
    for (std::uint32_t i = 0; i < n_tokens; ++i) {
        // Deterministic, and deliberately not all the same id: a repeated token
        // makes every position's hidden state similar enough that a cache
        // indexing bug can go unnoticed.
        tokens[i] = static_cast<soma::TokenId>((i * 7 + 3) % vocab);
    }

    soma::F32Workspace ws;
    ws.reserve(model.arch, n_tokens);

    std::vector<float> teacher;
    if (const auto st = soma::forward_f32(model, tokens, ws, teacher); !st.ok()) {
        r.note = std::string("teacher-forced forward failed: ") + st.message();
        return r;
    }

    soma::KvCache kv;
    if (const auto st = kv.open(model.arch, n_tokens); !st.ok()) {
        r.note = std::string("kv cache: ") + st.message();
        return r;
    }

    std::vector<float> step_logits;
    r.passed = true;
    const auto t0 = std::chrono::steady_clock::now();

    for (std::uint32_t p = 0; p < n_tokens; ++p) {
        soma::KvRow row;
        row.k_base = kv.k_at(0, 0);
        row.v_base = kv.v_at(0, 0);
        // FROM the cache, not recomputed. `n_tokens * kv.k_hkv()` was right here
        // only because this test opens the cache at exactly n_tokens — the same
        // reasoning that made the scheduler's copy right for GQA and catastrophic
        // for MLA (roadmap D40). A test that duplicates the formula it exists to
        // protect cannot catch the formula drifting.
        row.k_stride = kv.k_stride();
        row.v_stride = kv.v_stride();
        row.k_hkv = kv.k_hkv();
        row.v_hkv = kv.v_hkv();
        row.pos = p;
        row.len = p + 1; // visible history INCLUDING this position

        const soma::TokenId one[] = {tokens[p]};
        if (const auto st = soma::forward_step_f32(model, one, {&row, 1}, ws, step_logits);
            !st.ok()) {
            r.note = "step " + std::to_string(p) + ": " + st.message();
            r.passed = false;
            return r;
        }

        const float* want = teacher.data() + static_cast<std::size_t>(p) * vocab;
        for (std::uint32_t i = 0; i < vocab; ++i) {
            const float diff = std::fabs(step_logits[i] - want[i]);
            if (diff > r.max_abs) {
                r.max_abs = diff;
                r.worst_pos = p;
            }
        }
    }

    r.decode_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    r.passed = r.max_abs <= kTol;
    return r;
}

} // namespace

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures/tiny");

    // ABOVE GLM-5.2's index_topk of 64, and that is the whole reason for the
    // number. Below it, DSA's top-k selects every key it is allowed to attend and
    // the sparse path is bit-identical to dense — so a shorter run would exercise
    // the cached path without ever exercising the SELECTION in it, and pass for a
    // reason that has nothing to do with what was implemented. The first version
    // of this test used 24 and did exactly that.
    // Overridable so the absorbed-vs-expanded ratio can be measured against
    // CONTEXT LENGTH, which is the axis it scales on: the expanded form costs
    // O(n_sel * lora * H * (nope+vd)) and the absorbed one O(n_sel * H * lora),
    // so their ratio approaches (nope+vd) as the j loop grows and the fixed
    // per-step costs stop dominating.
    const std::uint32_t kTokens =
        (argc > 2) ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 128u;

    std::cout << "cached decode vs teacher-forced forward (" << kTokens << " positions)\n";
    std::cout << std::left << std::setw(34) << "fixture" << std::setw(10) << "result"
              << "detail\n";
    std::cout << std::string(86, '-') << "\n";

    int failures = 0;
    int ran = 0;
    std::vector<fs::path> dirs;
    if (fs::is_directory(root)) {
        for (const auto& e : fs::directory_iterator(root))
            if (e.is_directory()) dirs.push_back(e.path());
    }
    std::sort(dirs.begin(), dirs.end());

    for (const auto& dir : dirs) {
        const auto name = dir.filename().string();
        for (const bool absorb : {false, true}) {
            const auto r = check_fixture(dir, kTokens, absorb);
            const auto label = name + (absorb ? "  [absorbed]" : "  [expanded]");
            std::cout << std::left << std::setw(34) << label;
            if (r.skipped) {
                // Only a family with no adapter is a skip. Once load succeeds, a
                // missing cached path is a regression and is reported as FAIL.
                std::cout << std::setw(10) << "-"
                          << "SKIP: " << r.note << "\n";
                break; // the same skip both ways; saying it twice adds nothing
            }
            ++ran;
            std::cout << std::setw(10) << (r.passed ? "PASS" : "FAIL")
                      << "max|dlogit|=" << std::scientific << std::setprecision(2) << r.max_abs
                      << " @pos" << r.worst_pos << std::defaultfloat << "  " << std::fixed
                      << std::setprecision(0) << r.decode_ms << " ms" << std::defaultfloat;
            if (!r.note.empty()) std::cout << "  " << r.note;
            std::cout << "\n";
            if (!r.passed) ++failures;
        }
    }

    std::cout << std::string(86, '-') << "\n";
    // Zero fixtures exercised is a failure, not a pass. Every assertion above is
    // inside a loop that an empty directory would skip entirely.
    if (ran == 0) {
        std::cout << "FAIL: no fixture exercised the cached path\n";
        return 1;
    }
    std::cout << ran << " exercised, " << failures << " failed\n";
    return failures == 0 ? 0 : 1;
}
