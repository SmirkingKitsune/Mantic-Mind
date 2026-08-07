// Soma — admission ladder stage 1, shared by the test and by `soma conform`.
//
// Lifted verbatim out of tests/soma/conformance_g0.cpp when the admission
// pipeline gained a second caller. The tolerances, the oracle parser and the
// pass rule are here so there is exactly one of each.

#include "soma/conformance.hpp"

#include "soma/f32_model.hpp"
#include "soma/kernels_f32.hpp"

#include <algorithm>
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

} // namespace soma
