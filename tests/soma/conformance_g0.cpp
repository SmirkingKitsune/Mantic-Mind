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
#include "soma/conformance.hpp"
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

// The comparison itself lives in the LIBRARY now — `soma conform` runs the same
// stage from the admission pipeline, and two implementations would be two sets
// of tolerances and two opinions about what passes. This file keeps the fixture
// walk and the reporting, which is what a test is for.
using Result = soma::Fp32ConformanceResult;

Result run_fixture(const fs::path& dir) {
    return soma::run_fp32_conformance(dir.string());
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
