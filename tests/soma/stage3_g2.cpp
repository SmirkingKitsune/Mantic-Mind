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
// Usage: stage3_g2 <container_dir> <reference_dir> [--cache-gib N] [--positions N]

#include "soma/expert_store.hpp"
#include "soma/f32_model.hpp"
#include "soma/memory_hierarchy.hpp"
#include "soma/plan.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

/// Mean KL(reference || engine) in nats, over the softmax of each position.
///
/// Asymmetric on purpose and in this direction: it penalizes the engine for
/// putting low probability where the reference puts high probability, which is
/// the failure that matters. The reverse direction would tolerate the engine
/// dropping a mode the reference is confident about.
double logit_kl(const float* ref, const float* eng, std::uint32_t vocab) {
    double rmax = -1e30, emax = -1e30;
    for (std::uint32_t i = 0; i < vocab; ++i) {
        rmax = std::max(rmax, static_cast<double>(ref[i]));
        emax = std::max(emax, static_cast<double>(eng[i]));
    }
    double rsum = 0.0, esum = 0.0;
    for (std::uint32_t i = 0; i < vocab; ++i) {
        rsum += std::exp(static_cast<double>(ref[i]) - rmax);
        esum += std::exp(static_cast<double>(eng[i]) - emax);
    }
    const double rlog = rmax + std::log(rsum);
    const double elog = emax + std::log(esum);

    double kl = 0.0;
    for (std::uint32_t i = 0; i < vocab; ++i) {
        const double lp = static_cast<double>(ref[i]) - rlog;
        const double lq = static_cast<double>(eng[i]) - elog;
        const double p = std::exp(lp);
        if (p > 1e-12) kl += p * (lp - lq);
    }
    return kl;
}

struct Reference {
    std::uint32_t positions = 0, vocab = 0;
    std::vector<std::int32_t> ids;
    std::vector<float> logits;
};

bool read_reference(const fs::path& p, Reference& out, std::string& err) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { err = "cannot open " + p.string(); return false; }
    char magic[8]{};
    in.read(magic, 8);
    if (std::memcmp(magic, "SOMAORCL", 8) != 0) { err = "bad magic"; return false; }
    std::uint32_t hdr[5]{};
    in.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    out.positions = hdr[1];
    out.vocab = hdr[2];
    out.ids.resize(out.positions);
    out.logits.resize(static_cast<std::size_t>(out.positions) * out.vocab);
    in.read(reinterpret_cast<char*>(out.ids.data()),
            static_cast<std::streamsize>(out.ids.size() * 4));
    in.read(reinterpret_cast<char*>(out.logits.data()),
            static_cast<std::streamsize>(out.logits.size() * 4));
    if (!in) { err = "short read"; return false; }
    return true;
}

}  // namespace

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

    Reference ref;
    std::string err;
    if (!read_reference(rdir / "oracle.bin", ref, err)) {
        std::cerr << "reference: " << err << "\n";
        return 2;
    }

    soma::QuantMap qm;
    qm.expert_gate = {soma::DType::Q4_G, 128};
    qm.expert_up = {soma::DType::Q4_G, 128};
    qm.expert_down = {soma::DType::Q6_G, 128};

    std::cout << "loading dense half from container ...\n" << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    soma::F32Model model;
    if (auto st = soma::load_f32_model(cdir.string(), model, qm); !st.ok()) {
        std::cerr << "load failed: " << st.message() << "\n";
        return 2;
    }
    std::cout << "  " << std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count()
              << "s, streamed=" << (model.experts_are_streamed ? "yes" : "no") << "\n";

    soma::ExpertStore store;
    if (auto st = store.open(cdir.string(), model.arch); !st.ok()) {
        std::cerr << "container open failed: " << st.message() << "\n";
        return 2;
    }
    const auto& h = store.header();

    soma::MemoryBudget budget;
    budget.ram_expert_cache_bytes = cache_gib * 1024ull * 1024 * 1024;
    budget.pin_bytes = budget.ram_expert_cache_bytes / 8;

    soma::MemoryHierarchy mem;
    if (auto st = mem.open(model.arch, store, budget); !st.ok()) {
        std::cerr << "hierarchy open failed: " << st.message() << "\n";
        return 2;
    }
    model.streamed_experts = &mem;

    const auto n = want_positions ? std::min(want_positions, ref.positions) : ref.positions;
    if (ref.vocab != model.vocab()) {
        std::cerr << "vocab mismatch: reference " << ref.vocab << " vs model " << model.vocab()
                  << "\n";
        return 2;
    }

    std::cout << "container   " << h.n_layers << "L x " << h.n_experts << "E, expert="
              << (h.expert_bytes / 1024) << " KiB\n"
              << "cache       " << cache_gib << " GiB  (cap_per_layer="
              << mem.cap_per_layer() << ")\n"
              << "positions   " << n << " of " << ref.positions << "\n\n"
              << "streaming forward ...\n" << std::flush;

    std::vector<soma::TokenId> toks(ref.ids.begin(), ref.ids.begin() + n);
    soma::F32Workspace ws;
    std::vector<float> logits;
    t0 = std::chrono::steady_clock::now();
    if (auto st = soma::forward_f32(model, toks, ws, logits); !st.ok()) {
        std::cerr << "forward failed: " << st.message() << "\n";
        return 1;
    }
    const auto secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    // ── KL ───────────────────────────────────────────────────────────────────
    std::vector<double> per_pos(n);
    double sum = 0.0, worst = 0.0;
    std::uint32_t worst_at = 0;
    std::uint32_t top1_agree = 0;
    for (std::uint32_t t = 0; t < n; ++t) {
        const float* r = ref.logits.data() + static_cast<std::size_t>(t) * ref.vocab;
        const float* e = logits.data() + static_cast<std::size_t>(t) * ref.vocab;
        const double kl = logit_kl(r, e, ref.vocab);
        per_pos[t] = kl;
        sum += kl;
        if (kl > worst) { worst = kl; worst_at = t; }

        std::uint32_t ra = 0, ea = 0;
        for (std::uint32_t i = 1; i < ref.vocab; ++i) {
            if (r[i] > r[ra]) ra = i;
            if (e[i] > e[ea]) ea = i;
        }
        if (ra == ea) ++top1_agree;
    }
    const double mean = sum / n;
    std::sort(per_pos.begin(), per_pos.end());
    const double median = per_pos[n / 2];
    const double p95 = per_pos[static_cast<std::size_t>(n * 0.95)];

    const auto cs = mem.stats();
    const double hit_rate = 100.0 * static_cast<double>(cs.hits) /
                            static_cast<double>(std::max<std::uint64_t>(1, cs.hits + cs.misses));

    std::cout << std::fixed
              << "  forward       " << std::setprecision(1) << secs << "s ("
              << (secs / n * 1000.0) << " ms/token)\n"
              << "  cache         " << std::setprecision(1) << hit_rate << "% hit, "
              << cs.misses << " misses, " << cs.evictions << " evictions, "
              << (cs.bytes_read / 1048576) << " MiB read\n"
              << "  bytes/token   " << (cs.bytes_read / n / 1024) << " KiB\n"
              // The batch union, on a real checkpoint. Reported next to
              // bytes/token because they are the same claim measured twice: the
              // union is what makes the read cost per-expert rather than
              // per-row, and the ratio is how far that got.
              << "  expert reads  " << ws.unique_expert_reads << " unique of "
              << ws.naive_expert_reads << " naive  ("
              << std::setprecision(1)
              << (static_cast<double>(ws.naive_expert_reads) /
                  static_cast<double>(std::max<std::uint64_t>(1, ws.unique_expert_reads)))
              << "x union saving)\n\n"
              << "logit-KL(reference || engine), nats\n"
              << "  mean          " << std::setprecision(5) << mean << "\n"
              << "  median        " << median << "\n"
              << "  p95           " << p95 << "\n"
              << "  max           " << worst << "  (position " << worst_at << ")\n"
              << "  top-1 agree   " << std::setprecision(1)
              << (100.0 * top1_agree / n) << "%  (" << top1_agree << "/" << n << ")\n\n";

    // Threshold.
    //
    // 0.05 nats mean is a distribution the engine reproduces closely — for
    // reference, that is roughly the KL between a distribution and itself with
    // one logit perturbed by a few percent. Chosen against the measured
    // quantization error rather than as a round number: G1 showed q4_g moving
    // logits by ~16% relative, and the mean KL that induces is what this bounds.
    constexpr double kMeanKlThreshold = 0.05;
    constexpr double kP95KlThreshold = 0.25;

    const bool pass = (mean <= kMeanKlThreshold) && (p95 <= kP95KlThreshold);
    // Explicit precision: the stream is still carrying setprecision(1) from the
    // top-1 line above, which printed the real 0.05/0.25 gate as "0.1/0.2" —
    // looser on the mean, tighter on p95. The comparison was always against the
    // constants, but a gate that misreports its own threshold is a gate nobody
    // can check.
    std::cout << "stage 3: " << (pass ? "PASS" : "FAIL") << "  (mean <= "
              << std::setprecision(3) << kMeanKlThreshold << ", p95 <= "
              << kP95KlThreshold << ")\n";

    if (!pass) {
        // A failure is only a QUANTIZATION finding if the output is degraded but
        // still related to the reference. Two signatures say otherwise, and
        // calling them "quantization" sends the reader to remediate the quant map
        // while the real fault is elsewhere:
        //
        //   KL ~ ln(vocab)      the engine is emitting a UNIFORM distribution.
        //                       Quantization does not flatten a distribution to
        //                       maximum entropy; zeroed or unloaded weights do.
        //   top-1 agree ~ 0%    the outputs are unrelated, not merely blurred.
        //                       q4_g moves logits ~16% relative, which keeps the
        //                       argmax most of the time.
        const double uniform_kl = std::log(static_cast<double>(ref.vocab));
        const double top1_pct = 100.0 * top1_agree / n;
        const bool degenerate = (mean > 0.8 * uniform_kl) || (top1_pct < 5.0);

        if (degenerate) {
            std::cout << "\nThis is NOT a quantization finding.\n\n"
                      << "  mean KL " << std::setprecision(3) << mean << " vs ln(vocab) = "
                      << uniform_kl << ", and top-1 agreement " << std::setprecision(1)
                      << top1_pct << "%.\n"
                      << "  The engine is producing an essentially uniform distribution, which\n"
                      << "  quantization does not do at any precision. Look for weights that are\n"
                      << "  zero or never loaded — check the container's dense tensors before\n"
                      << "  touching the quant map.\n";
        } else {
            std::cout << "\nThis is a QUANTIZATION finding, not a correctness bug: stages 1 and 2\n"
                         "pass token-exact on the same code. Remediation is the quant map — raise\n"
                         "expert_down, or tighten the group — not the kernels.\n";
        }
    }
    return pass ? 0 : 1;
}
