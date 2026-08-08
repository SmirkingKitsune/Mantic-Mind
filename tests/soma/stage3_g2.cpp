// Soma — conformance stage 3: logit-KL against an fp16/bf16 reference.
//
// Stages 1 and 2 demand token-exactness on tiny-random models. Stage 3 cannot and
// should not: a real checkpoint runs at a precision the quantized engine does not
// reproduce bit-for-bit, so the bar is DISTRIBUTIONAL.
//
// That separation is load-bearing. Failing stage 3 while 1 and 2 pass is a
// QUANTIZATION finding, not a correctness bug — remediation is requantizing a
// role or raising a group-scale granularity, not debugging a kernel. Reporting
// them as the same failure costs days.
//
// The COMPARISON now lives in soma/conformance.hpp, because `soma conform` runs
// it too and two implementations would mean two thresholds and two opinions
// about what passes. What stays here is the reporting and the remediation
// guidance, which is what a harness is for.
//
// Usage: stage3_g2 <container_dir> <reference_dir> [--cache-gib N] [--positions N]

#include "soma/conformance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: stage3_g2 <container_dir> <reference_dir> "
                     "[--cache-gib N] [--positions N]\n";
        return 2;
    }
    const fs::path cdir(argv[1]);
    const fs::path rdir(argv[2]);
    std::uint64_t cache_gib = 8;
    std::uint32_t want_positions = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "--cache-gib") cache_gib = std::stoull(argv[i + 1]);
        if (std::string(argv[i]) == "--positions") want_positions = std::stoul(argv[i + 1]);
    }

    std::cout << "loading container and streaming forward ...\n" << std::flush;
    const auto r = soma::run_real_logit_kl(cdir.string(), (rdir / "oracle.bin").string(),
                                           cache_gib, want_positions);
    if (r.skipped) {
        std::cerr << "SKIPPED: " << r.detail << "\n";
        return 2;
    }
    if (!r.loaded || r.positions == 0) {
        std::cerr << "failed: " << r.detail << "\n";
        return 2;
    }

    const auto n = r.positions;
    std::cout << std::fixed
              << "  forward       " << std::setprecision(1) << r.forward_seconds << "s ("
              << (r.forward_seconds / n * 1000.0) << " ms/token)\n"
              << "  cache         " << std::setprecision(1) << r.cache_hit_rate_pct
              << "% hit, " << r.cache_misses << " misses, " << r.cache_evictions
              << " evictions, " << (r.bytes_read / 1048576) << " MiB read\n"
              << "  bytes/token   " << (r.bytes_read / n / 1024) << " KiB\n"
              // The batch union, on a real checkpoint. Reported next to
              // bytes/token because they are the same claim measured twice: the
              // union is what makes the read cost per-expert rather than
              // per-row, and the ratio is how far that got.
              << "  expert reads  " << r.unique_expert_reads << " unique of "
              << r.naive_expert_reads << " naive  (" << std::setprecision(1)
              << (static_cast<double>(r.naive_expert_reads) /
                  static_cast<double>(std::max<std::uint64_t>(1, r.unique_expert_reads)))
              << "x union saving)\n\n"
              << "logit-KL(reference || engine), nats\n"
              << "  mean          " << std::setprecision(5) << r.mean_kl << "\n"
              << "  median        " << r.median_kl << "\n"
              << "  p95           " << r.p95_kl << "\n"
              << "  max           " << r.worst_kl << "  (position " << r.worst_at << ")\n"
              << "  top-1 agree   " << std::setprecision(1) << r.top1_agreement_pct << "%\n\n";

    // Explicit precision: the stream is still carrying setprecision(1) from the
    // top-1 line above, which printed the real 0.05/0.25 gate as "0.1/0.2" —
    // looser on the mean, tighter on p95. The comparison was always against the
    // constants, but a gate that misreports its own threshold is a gate nobody
    // can check.
    std::cout << "stage 3: " << (r.passed ? "PASS" : "FAIL") << "  (mean <= "
              << std::setprecision(3) << soma::kRealLogitKlMeanMax
              << ", p95 <= " << soma::kRealLogitKlP95Max << ")\n";

    if (!r.passed) {
        // A failure is only a QUANTIZATION finding if the output is degraded but
        // still related to the reference. Two signatures say otherwise, and
        // calling them "quantization" sends the reader to remediate the quant map
        // while the real fault is elsewhere. run_real_logit_kl decides which; the
        // wording is here, because a remediation hint is a harness concern.
        if (r.degenerate) {
            const double uniform_kl = std::log(static_cast<double>(r.vocab));
            std::cout << "\nThis is NOT a quantization finding.\n\n"
                      << "  mean KL " << std::setprecision(3) << r.mean_kl
                      << " vs ln(vocab) = " << uniform_kl << ", and top-1 agreement "
                      << std::setprecision(1) << r.top1_agreement_pct << "%.\n";
            // Two different faults, and the difference is a hard bound rather
            // than a judgement call: KL(ref||uniform) = ln(vocab) - H(ref), so it
            // can never EXCEED ln(vocab). A mean above that is arithmetically not
            // a flat engine — it is a confident one pointing somewhere else,
            // which is what a shuffled vocab or a mismatched tokenizer looks
            // like. Saying "essentially uniform" there sends the reader to
            // inspect weights that are fine.
            if (r.mean_kl > uniform_kl) {
                std::cout
                    << "  KL above ln(vocab) is not a flat distribution — it cannot be. The\n"
                    << "  engine is confidently placing mass where the reference has almost\n"
                    << "  none, which means the two are not describing the same model: check\n"
                    << "  that the reference was built from THIS checkpoint and that the\n"
                    << "  tokenizer and vocab ordering match before touching the quant map.\n";
            } else {
                std::cout
                    << "  The engine is producing an essentially uniform distribution, which\n"
                    << "  quantization does not do at any precision. Look for weights that are\n"
                    << "  zero or never loaded — check the container's dense tensors before\n"
                    << "  touching the quant map.\n";
            }
        } else {
            std::cout << "\nThis is a QUANTIZATION finding, not a correctness bug: stages 1 and 2\n"
                         "pass token-exact on the same code. Remediation is the quant map — raise\n"
                         "expert_down, or tighten the group — not the kernels.\n";
        }
    }
    return r.passed ? 0 : 1;
}
