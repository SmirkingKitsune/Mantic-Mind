// Soma — admission ladder stage 1, shared by the test and by `soma conform`.
//
// Lifted verbatim out of tests/soma/conformance_g0.cpp when the admission
// pipeline gained a second caller. The tolerances, the oracle parser and the
// pass rule are here so there is exactly one of each.

#include "soma/conformance.hpp"

#include "soma/arch_ir.hpp"
#include "soma/expert_store.hpp"
#include "soma/f32_model.hpp"
#include "soma/kernels_f32.hpp"
#include "soma/memory_hierarchy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace soma {

Status read_oracle_fixture(const std::string& path, OracleFixture& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {StatusCode::NotFound, "cannot open " + path};

    char magic[8]{};
    in.read(magic, 8);
    if (std::memcmp(magic, "SOMAORCL", 8) != 0) {
        return {StatusCode::InvalidArgument, "bad magic in " + path};
    }

    std::uint32_t hdr[5]{};
    in.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (hdr[0] != 1) {
        return {StatusCode::VersionMismatch,
                "unsupported oracle version " + std::to_string(hdr[0])};
    }

    out.positions = hdr[1];
    out.vocab = hdr[2];
    out.input_ids.resize(out.positions);
    out.tf_logits.resize(static_cast<std::size_t>(out.positions) * out.vocab);
    out.greedy_prefix.resize(hdr[3]);
    out.greedy_tokens.resize(hdr[4]);

    const auto read_into = [&](auto& v) {
        in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    };
    read_into(out.input_ids);
    read_into(out.tf_logits);
    read_into(out.greedy_prefix);
    read_into(out.greedy_tokens);
    if (!in) return {StatusCode::InvalidArgument, "short read on " + path};
    return {};
}

Fp32ConformanceResult run_fp32_conformance(const std::string& model_dir) {
    Fp32ConformanceResult r;

    F32Model model;
    if (auto st = load_f32_model(model_dir, model); !st.ok()) {
        r.detail = st.message();
        // An unsupported family is a gap in coverage, not an engine failure.
        r.skipped = (st.code() == StatusCode::Unsupported);
        return r;
    }
    r.loaded = true;

    OracleFixture oracle;
    if (auto st = read_oracle_fixture((fs::path(model_dir) / "oracle.bin").string(), oracle);
        !st.ok()) {
        r.detail = st.message();
        return r;
    }

    if (oracle.vocab != model.vocab()) {
        r.detail = "oracle vocab " + std::to_string(oracle.vocab) + " != model vocab " +
                   std::to_string(model.vocab());
        return r;
    }

    // ── teacher forcing ──────────────────────────────────────────────────────
    std::vector<TokenId> tokens(oracle.input_ids.begin(), oracle.input_ids.end());
    F32Workspace ws;
    std::vector<float> logits;
    if (auto st = forward_f32(model, tokens, ws, logits); !st.ok()) {
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
        if (d > r.max_abs) {
            r.max_abs = d;
            r.max_at_pos = static_cast<std::uint32_t>(i / oracle.vocab);
        }
        sum_abs += d;
    }
    r.mean_abs = static_cast<float>(sum_abs / static_cast<double>(logits.size()));

    for (std::uint32_t i = 0; i < oracle.vocab; ++i) {
        r.max_abs_pos0 = std::max(r.max_abs_pos0, std::fabs(logits[i] - oracle.tf_logits[i]));
    }
    r.logits_pass =
        (r.max_abs <= kConformanceMaxAbsDiff) && (r.mean_abs <= kConformanceMaxMeanDiff);

    // ── greedy ───────────────────────────────────────────────────────────────
    std::vector<TokenId> prefix(oracle.greedy_prefix.begin(), oracle.greedy_prefix.end());
    std::vector<TokenId> generated;
    const auto want = static_cast<std::uint32_t>(oracle.greedy_tokens.size());
    if (auto st = generate_greedy_f32(model, prefix, want, ws, generated); !st.ok()) {
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

// ── stage 2 ──────────────────────────────────────────────────────────────────

namespace {

/// Mean KL(reference || engine) in nats over one position's softmax.
///
/// Asymmetric on purpose, and in this direction: it penalizes the engine for
/// putting LOW probability where the reference puts high probability, which is
/// the failure that matters. The reverse direction would happily tolerate an
/// engine that drops a mode the reference is confident about.
///
/// Both sides are log-sum-exp'd from raw logits rather than softmaxed first: at
/// a 50k vocab the naive form underflows the tail, and the tail is exactly where
/// a bad quantization shows up.
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

/// The container's OWN quant map, not an assumed one.
QuantMap container_quant_map(const std::string& container_dir, const QuantMap& fallback) {
    std::ifstream meta(fs::path(container_dir) / "container_meta.json", std::ios::binary);
    if (!meta) return fallback;
    const std::string text((std::istreambuf_iterator<char>(meta)),
                           std::istreambuf_iterator<char>());
    ArchIr arch;
    arch.quantization = fallback;
    if (auto st = apply_container_quant(text, arch); !st.ok()) return fallback;
    return arch.quantization;
}

} // namespace

RealLogitKlResult run_real_logit_kl(const std::string& container_dir,
                                    const std::string& reference_path,
                                    std::uint64_t cache_gib,
                                    std::uint32_t max_positions) {
    RealLogitKlResult r;

    OracleFixture ref;
    if (auto st = read_oracle_fixture(reference_path, ref); !st.ok()) {
        // Absent evidence, not adverse evidence. See the header.
        r.skipped = true;
        r.detail = st.message();
        return r;
    }

    // Defaults only if the container declines to say; the map it was built with
    // is the one it has to be read back with.
    QuantMap fallback;
    fallback.expert_gate = {DType::Q4_G, 128};
    fallback.expert_up = {DType::Q4_G, 128};
    fallback.expert_down = {DType::Q6_G, 128};
    const auto qm = container_quant_map(container_dir, fallback);

    F32Model model;
    if (auto st = load_f32_model(container_dir, model, qm); !st.ok()) {
        r.detail = st.message();
        r.skipped = (st.code() == StatusCode::Unsupported);
        return r;
    }
    r.loaded = true;

    ExpertStore store;
    if (auto st = store.open(container_dir, model.arch); !st.ok()) {
        r.detail = "container open failed: " + st.message();
        return r;
    }

    MemoryBudget budget;
    budget.ram_expert_cache_bytes = cache_gib * 1024ull * 1024 * 1024;
    budget.pin_bytes = budget.ram_expert_cache_bytes / 8;

    MemoryHierarchy mem;
    if (auto st = mem.open(model.arch, store, budget); !st.ok()) {
        r.detail = "hierarchy open failed: " + st.message();
        return r;
    }
    model.streamed_experts = &mem;

    if (ref.vocab != model.vocab()) {
        r.detail = "vocab mismatch: reference " + std::to_string(ref.vocab) + " vs model " +
                   std::to_string(model.vocab());
        return r;
    }
    const std::uint32_t n = max_positions ? std::min(max_positions, ref.positions) : ref.positions;
    if (n == 0) {
        r.skipped = true;
        r.detail = "reference carries no positions";
        return r;
    }
    r.positions = n;
    r.vocab = ref.vocab;

    std::vector<TokenId> toks(ref.input_ids.begin(), ref.input_ids.begin() + n);
    F32Workspace ws;
    std::vector<float> logits;
    const auto t0 = std::chrono::steady_clock::now();
    if (auto st = forward_f32(model, toks, ws, logits); !st.ok()) {
        r.detail = st.message();
        return r;
    }
    r.forward_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::vector<double> per_pos(n);
    double sum = 0.0;
    std::uint32_t top1_agree = 0;
    for (std::uint32_t t = 0; t < n; ++t) {
        const float* rp = ref.tf_logits.data() + static_cast<std::size_t>(t) * ref.vocab;
        const float* ep = logits.data() + static_cast<std::size_t>(t) * ref.vocab;
        const double kl = logit_kl(rp, ep, ref.vocab);
        per_pos[t] = kl;
        sum += kl;
        if (kl > r.worst_kl) {
            r.worst_kl = kl;
            r.worst_at = t;
        }
        std::uint32_t ra = 0, ea = 0;
        for (std::uint32_t i = 1; i < ref.vocab; ++i) {
            if (rp[i] > rp[ra]) ra = i;
            if (ep[i] > ep[ea]) ea = i;
        }
        if (ra == ea) ++top1_agree;
    }
    r.mean_kl = sum / n;
    r.top1_agreement_pct = 100.0 * static_cast<double>(top1_agree) / n;

    std::sort(per_pos.begin(), per_pos.end());
    r.median_kl = per_pos[n / 2];
    r.p95_kl = per_pos[static_cast<std::size_t>(static_cast<double>(n) * 0.95)];

    const auto cs = mem.stats();
    const auto looks = cs.hits + cs.misses;
    r.cache_hit_rate_pct =
        looks ? 100.0 * static_cast<double>(cs.hits) / static_cast<double>(looks) : 0.0;
    r.cache_misses = cs.misses;
    r.cache_evictions = cs.evictions;
    r.bytes_read = cs.bytes_read;
    r.unique_expert_reads = ws.unique_expert_reads;
    r.naive_expert_reads = ws.naive_expert_reads;

    // Broken rather than merely degraded: KL approaching what a UNIFORM
    // distribution would score, or top-1 at chance. Different remedy, so it gets
    // its own flag instead of being folded into "failed".
    const double uniform_kl = std::log(static_cast<double>(ref.vocab));
    r.degenerate = (r.mean_kl > 0.8 * uniform_kl) || (r.top1_agreement_pct < 5.0);

    r.passed = (r.mean_kl <= kRealLogitKlMeanMax) && (r.p95_kl <= kRealLogitKlP95Max);
    return r;
}

} // namespace soma
