// Soma — G0 conformance: fp32 path vs the transformers oracle.
//
// Gate (docs/roadmap.md):
//   * teacher-forced logits match to fp32 tolerance over >= 512 positions
//   * greedy generation is token-exact for >= 256 tokens
//
// Failing this is `verdict: reject` -> the llama.cpp fallback. These are not
// development checkpoints; they are stages 1 and 2 of the admission ladder.
//
// Usage: conformance_g0 <fixtures_dir> [fixture_name ...]

#include "soma/f32_model.hpp"
#include "soma/kernels_f32.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Tolerances.
//
// The engine and torch accumulate the same sums in different orders, so exact
// bit-equality is not a reasonable bar for fp32 — but the bar still has to be
// tight enough to catch a wrong RoPE pairing or a mis-scaled expert, both of
// which move logits far more than reassociation does.
constexpr float kMaxAbsDiff  = 2e-3f;
constexpr float kMaxMeanDiff = 2e-4f;

struct Oracle {
    std::uint32_t positions = 0;
    std::uint32_t vocab = 0;
    std::vector<std::int32_t> input_ids;
    std::vector<float>        tf_logits;
    std::vector<std::int32_t> greedy_prefix;
    std::vector<std::int32_t> greedy_tokens;
};

bool read_oracle(const fs::path& path, Oracle& out, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { err = "cannot open " + path.string(); return false; }

    char magic[8]{};
    in.read(magic, 8);
    if (std::memcmp(magic, "SOMAORCL", 8) != 0) { err = "bad magic in " + path.string(); return false; }

    std::uint32_t hdr[5]{};
    in.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (hdr[0] != 1) { err = "unsupported oracle version " + std::to_string(hdr[0]); return false; }

    out.positions = hdr[1];
    out.vocab     = hdr[2];
    const std::uint32_t n_prefix = hdr[3];
    const std::uint32_t n_greedy = hdr[4];

    out.input_ids.resize(out.positions);
    out.tf_logits.resize(static_cast<std::size_t>(out.positions) * out.vocab);
    out.greedy_prefix.resize(n_prefix);
    out.greedy_tokens.resize(n_greedy);

    in.read(reinterpret_cast<char*>(out.input_ids.data()),
            static_cast<std::streamsize>(out.input_ids.size() * 4));
    in.read(reinterpret_cast<char*>(out.tf_logits.data()),
            static_cast<std::streamsize>(out.tf_logits.size() * 4));
    in.read(reinterpret_cast<char*>(out.greedy_prefix.data()),
            static_cast<std::streamsize>(out.greedy_prefix.size() * 4));
    in.read(reinterpret_cast<char*>(out.greedy_tokens.data()),
            static_cast<std::streamsize>(out.greedy_tokens.size() * 4));
    if (!in) { err = "short read on " + path.string(); return false; }
    return true;
}

struct Result {
    bool  loaded = false;
    bool  logits_pass = false;
    bool  greedy_pass = false;
    bool  skipped = false;
    std::string detail;
    float max_abs = 0.0f;
    float max_abs_pos0 = 0.0f;
    std::uint32_t max_at_pos = 0;
    float mean_abs = 0.0f;
    std::uint32_t first_bad_token = 0;
    std::uint32_t matched_tokens = 0;
};

Result run_fixture(const fs::path& dir) {
    Result r;

    soma::F32Model model;
    if (auto st = soma::load_f32_model(dir.string(), model); !st.ok()) {
        r.detail = st.message();
        // An unsupported family is a gap in coverage, not a failure of the
        // engine. G4 adds MLA; reporting it as a failure until then would make
        // the suite permanently red and therefore ignored.
        r.skipped = (st.code() == soma::StatusCode::Unsupported);
        return r;
    }
    r.loaded = true;

    Oracle oracle;
    std::string err;
    if (!read_oracle(dir / "oracle.bin", oracle, err)) { r.detail = err; return r; }

    if (oracle.vocab != model.vocab()) {
        r.detail = "oracle vocab " + std::to_string(oracle.vocab) + " != model vocab " +
                   std::to_string(model.vocab());
        return r;
    }

    // ── teacher forcing ──────────────────────────────────────────────────────
    std::vector<soma::TokenId> tokens(oracle.input_ids.begin(), oracle.input_ids.end());
    soma::F32Workspace ws;
    std::vector<float> logits;
    if (auto st = soma::forward_f32(model, tokens, ws, logits); !st.ok()) {
        r.detail = st.message();
        return r;
    }
    if (logits.size() != oracle.tf_logits.size()) {
        r.detail = "logit count mismatch";
        return r;
    }

    double sum_abs = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float d = std::fabs(logits[i] - oracle.tf_logits[i]);
        if (d > r.max_abs) { r.max_abs = d; r.max_at_pos = static_cast<std::uint32_t>(i / oracle.vocab); }
        sum_abs += d;
    }
    r.mean_abs = static_cast<float>(sum_abs / static_cast<double>(logits.size()));

    // Position 0 is a natural bisection point and costs nothing to report.
    //
    // At t=0 RoPE is the identity (angle 0 -> cos 1, sin 0) and attention is
    // trivial (one visible key, softmax of a single element = 1). So a
    // divergence that is ALREADY present at position 0 cannot be RoPE or the
    // attention reduction — it is projection, qk-norm, routing, or the expert
    // MLP. A divergence that is clean at 0 and grows with t is the opposite.
    for (std::uint32_t i = 0; i < oracle.vocab; ++i) {
        r.max_abs_pos0 = std::max(r.max_abs_pos0, std::fabs(logits[i] - oracle.tf_logits[i]));
    }
    r.logits_pass = (r.max_abs <= kMaxAbsDiff) && (r.mean_abs <= kMaxMeanDiff);

    // ── greedy ───────────────────────────────────────────────────────────────
    std::vector<soma::TokenId> prefix(oracle.greedy_prefix.begin(), oracle.greedy_prefix.end());
    std::vector<soma::TokenId> generated;
    const auto want = static_cast<std::uint32_t>(oracle.greedy_tokens.size());
    if (auto st = soma::generate_greedy_f32(model, prefix, want, ws, generated); !st.ok()) {
        r.detail = st.message();
        return r;
    }

    r.greedy_pass = true;
    for (std::uint32_t i = 0; i < want; ++i) {
        if (static_cast<std::int32_t>(generated[i]) != oracle.greedy_tokens[i]) {
            r.greedy_pass = false;
            r.first_bad_token = i;
            break;
        }
        r.matched_tokens = i + 1;
    }
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures/tiny");
    if (!fs::is_directory(root)) {
        std::cerr << "fixtures directory not found: " << root.string() << "\n";
        return 2;
    }

    std::vector<fs::path> fixtures;
    if (argc > 2) {
        for (int i = 2; i < argc; ++i) fixtures.push_back(root / argv[i]);
    } else {
        for (const auto& e : fs::directory_iterator(root)) {
            if (e.is_directory() && fs::exists(e.path() / "oracle.bin")) {
                fixtures.push_back(e.path());
            }
        }
        std::sort(fixtures.begin(), fixtures.end());
    }

    int failures = 0, passed = 0, skipped = 0;
    std::cout << std::left << std::setw(34) << "fixture"
              << std::setw(10) << "logits" << std::setw(10) << "greedy"
              << "detail\n"
              << std::string(86, '-') << "\n";

    for (const auto& dir : fixtures) {
        const Result r = run_fixture(dir);
        std::cout << std::left << std::setw(34) << dir.filename().string();

        if (r.skipped) {
            ++skipped;
            std::cout << std::setw(10) << "-" << std::setw(10) << "-"
                      << "SKIP: " << r.detail << "\n";
            continue;
        }
        if (!r.loaded) {
            ++failures;
            std::cout << std::setw(10) << "ERR" << std::setw(10) << "-" << r.detail << "\n";
            continue;
        }

        std::ostringstream detail;
        detail << "max=" << std::scientific << std::setprecision(2) << r.max_abs
               << "@t" << r.max_at_pos
               << " pos0=" << r.max_abs_pos0
               << " mean=" << r.mean_abs;
        if (!r.greedy_pass) {
            detail << "  greedy diverged at token " << r.first_bad_token;
        }
        if (!r.detail.empty()) detail << "  " << r.detail;

        const bool ok = r.logits_pass && r.greedy_pass;
        ok ? ++passed : ++failures;
        std::cout << std::setw(10) << (r.logits_pass ? "PASS" : "FAIL")
                  << std::setw(10) << (r.greedy_pass ? "PASS" : "FAIL")
                  << detail.str() << "\n";
    }

    std::cout << std::string(86, '-') << "\n"
              << passed << " passed, " << failures << " failed, " << skipped << " skipped\n";

    // A gate that evaluated nothing has not passed.
    //
    // Pointed at the wrong directory this printed "0 passed, 0 failed, 0 skipped"
    // and exited 0 — green, and having checked nothing at all. That is the worst
    // available outcome for a conformance gate, because it looks exactly like
    // success right up until someone moves the fixtures.
    if (passed == 0) {
        std::cout << "\nNo fixture was evaluated. This is a FAILURE, not a pass: either the\n"
                     "fixtures directory is wrong, or every family in it is unsupported.\n";
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
