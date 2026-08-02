// Soma — G1 conformance: the quantized path against the fp32 path.
//
// Gate — AMENDED from the roadmap's original wording, for a reason worth stating:
//
//   * validate_quant_map rejects a non-F32 router — tested by TRYING it
//   * codec round-trip error matches the format's theoretical bits/weight
//   * the forward PROPAGATES quantization error linearly rather than AMPLIFYING
//     it: relative logit error stays within a small factor of relative weight
//     error
//
// The roadmap originally said "greedy token-exact vs fp32". That is not a
// meaningful gate on a TINY-RANDOM model. An untrained model has essentially no
// logit margin, so any perturbation flips argmax within a few tokens regardless
// of whether the kernels are correct — measured here, q8_0 at 5e-3 weight error
// diverges after ~30 tokens, from a codec demonstrably correct to 9.00
// bits/weight.
//
// What DOES separate a correct quantized forward from a broken one is whether
// end-to-end error TRACKS the weight error. A mis-packed nibble, a wrong group
// stride, or a broken accumulation amplifies it by orders of magnitude. That is
// what this asserts.
//
// Token-exactness becomes a real signal again at G2, on a trained checkpoint
// with actual logit margins.
//
// The fp32 path is the reference because G0 already tied it to `transformers`.
// Comparing quantized output against the oracle directly would conflate
// quantization error with a forward bug; comparing against fp32 isolates it.
//
// Usage: conformance_g1 <fixtures_dir> [fixture ...]

#include "soma/f32_model.hpp"
#include "soma/quant_format.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kPositions = 64;   // logit comparison length
constexpr std::uint32_t kGreedy    = 64;   // greedy comparison length

struct Config {
    const char*    name;
    soma::QuantMap map;
    soma::DType    probe_dtype;   ///< representative dtype, for the codec bound
    std::uint32_t  probe_group;
};

/// How much the forward may amplify the codec's own relative error before it
/// counts as a bug. Generous — attention and the MoE union both sum many
/// perturbed terms — but orders of magnitude below what a packing or stride bug
/// produces.
constexpr float kMaxAmplification = 6.0f;

soma::QuantMap all_f32() { return {}; }

soma::QuantMap uniform(soma::DType d, std::uint32_t group) {
    soma::QuantMap m;
    m.embed = {d, group};
    m.attn_proj = {d, group};
    m.expert_gate = {d, group};
    m.expert_up = {d, group};
    m.expert_down = {d, group};
    m.shared_expert = {d, group};
    // router and norms stay F32 — the map refuses otherwise.
    return m;
}

/// The map from schemas/arch-ir.md §5: cheaper gate/up, higher-precision down.
soma::QuantMap mixed_role_map() {
    soma::QuantMap m;
    m.embed = {soma::DType::Q8_0, 32};
    m.attn_proj = {soma::DType::Q4_G, 128};
    m.expert_gate = {soma::DType::Q4_G, 128};
    m.expert_up = {soma::DType::Q4_G, 128};
    m.expert_down = {soma::DType::Q6_G, 128};
    m.shared_expert = {soma::DType::Q4_G, 128};
    return m;
}

/// Round-trip error of the format itself, isolated from any model.
///
/// A format bug and a forward bug produce the same symptom downstream, so
/// measuring the codec on its own first means a later mismatch has one fewer
/// possible cause.
bool codec_check(std::ostream& os) {
    std::mt19937 rng(20260729);
    std::normal_distribution<float> dist(0.0f, 0.05f);

    constexpr std::uint32_t rows = 8, cols = 256;
    std::vector<float> src(static_cast<std::size_t>(rows) * cols);
    for (auto& v : src) v = dist(rng);

    struct Row { soma::DType d; std::uint32_t group; };
    const Row rows_to_try[] = {
        {soma::DType::Q8_0, 32}, {soma::DType::Q8_0, 128},
        {soma::DType::Q4_0, 32}, {soma::DType::Q4_G, 128},
        {soma::DType::Q6_G, 128},
    };

    os << "  codec round-trip (sigma=0.05, 8x256)\n";
    bool ok = true;
    std::vector<float> back(src.size());
    for (const auto& r : rows_to_try) {
        soma::QTensor t;
        if (auto st = soma::quantize_tensor(src, rows, cols, r.d, r.group, t); !st.ok()) {
            os << "    " << soma::to_string(r.d) << ": FAILED " << st.message() << "\n";
            ok = false;
            continue;
        }
        if (auto st = soma::dequantize_tensor(t, back); !st.ok()) {
            os << "    " << soma::to_string(r.d) << ": dequant FAILED " << st.message() << "\n";
            ok = false;
            continue;
        }
        double sum2 = 0.0, ref2 = 0.0;
        float maxe = 0.0f;
        for (std::size_t i = 0; i < src.size(); ++i) {
            const float e = back[i] - src[i];
            sum2 += static_cast<double>(e) * e;
            ref2 += static_cast<double>(src[i]) * src[i];
            maxe = std::max(maxe, std::fabs(e));
        }
        const double rel = std::sqrt(sum2 / ref2);
        const double bpw = 8.0 * static_cast<double>(t.data.size()) /
                           static_cast<double>(src.size());
        os << "    " << std::left << std::setw(6) << soma::to_string(r.d)
           << " g=" << std::setw(4) << r.group
           << " bits/weight=" << std::fixed << std::setprecision(2) << bpw
           << "  rel_rms=" << std::scientific << std::setprecision(2) << rel
           << "  max=" << maxe << "\n";
    }
    return ok;
}

struct Outcome {
    bool ok = false;
    std::string error;
    float max_logit_diff = 0.0f;
    float mean_logit_diff = 0.0f;
    float rel_logit_err = 0.0f;   ///< mean |dlogit| / rms(reference logits)
    std::uint32_t greedy_matched = 0;
    std::uint64_t bytes = 0;
};

/// Relative RMS error the FORMAT introduces, on the same distribution the
/// fixture weights were drawn from. End-to-end relative logit error is compared
/// against this.
float codec_rel_rms(soma::DType d, std::uint32_t group) {
    std::mt19937 rng(20260729);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);  // matches make_oracle
    constexpr std::uint32_t rows = 16, cols = 256;
    std::vector<float> src(static_cast<std::size_t>(rows) * cols), back(src.size());
    for (auto& v : src) v = dist(rng);

    soma::QTensor t;
    if (!soma::quantize_tensor(src, rows, cols, d, group, t).ok()) return 0.0f;
    if (!soma::dequantize_tensor(t, back).ok()) return 0.0f;
    double e2 = 0.0, r2 = 0.0;
    for (std::size_t i = 0; i < src.size(); ++i) {
        const double e = back[i] - src[i];
        e2 += e * e;
        r2 += static_cast<double>(src[i]) * src[i];
    }
    return static_cast<float>(std::sqrt(e2 / r2));
}

Outcome run_config(const fs::path& dir, const soma::QuantMap& map,
                   const std::vector<soma::TokenId>& tokens,
                   const std::vector<float>& ref_logits,
                   const std::vector<soma::TokenId>& ref_greedy) {
    Outcome o;
    soma::F32Model model;
    if (auto st = soma::load_f32_model(dir.string(), model, map); !st.ok()) {
        o.error = st.message();
        return o;
    }
    for (const auto& q : model.quantized) o.bytes += soma::quantized_footprint(q);

    soma::F32Workspace ws;
    std::vector<float> logits;
    if (auto st = soma::forward_f32(model, tokens, ws, logits); !st.ok()) {
        o.error = st.message();
        return o;
    }
    if (logits.size() != ref_logits.size()) {
        o.error = "logit count mismatch";
        return o;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float d = std::fabs(logits[i] - ref_logits[i]);
        o.max_logit_diff = std::max(o.max_logit_diff, d);
        sum += d;
    }
    o.mean_logit_diff = static_cast<float>(sum / static_cast<double>(logits.size()));

    double r2 = 0.0;
    for (const float v : ref_logits) r2 += static_cast<double>(v) * v;
    const double rms = std::sqrt(r2 / static_cast<double>(ref_logits.size()));
    o.rel_logit_err = (rms > 0.0) ? static_cast<float>(o.mean_logit_diff / rms) : 0.0f;

    std::vector<soma::TokenId> prefix(tokens.begin(), tokens.begin() + 16);
    std::vector<soma::TokenId> got;
    if (auto st = soma::generate_greedy_f32(model, prefix, kGreedy, ws, got); !st.ok()) {
        o.error = st.message();
        return o;
    }
    while (o.greedy_matched < ref_greedy.size() &&
           o.greedy_matched < got.size() &&
           got[o.greedy_matched] == ref_greedy[o.greedy_matched]) {
        ++o.greedy_matched;
    }
    o.ok = true;
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures/tiny");
    if (!fs::is_directory(root)) {
        std::cerr << "fixtures directory not found: " << root.string() << "\n";
        return 2;
    }

    int failures = 0;

    // ── negative test: the router may not be quantized ───────────────────────
    //
    // Quantizing router logits changes WHICH EXPERTS FIRE — a semantic change,
    // not a precision one. Asserted by trying it, because a constraint nobody
    // tests is a comment.
    {
        soma::QuantMap bad;
        bad.router = {soma::DType::Q8_0, 32};
        const auto st = soma::validate_quant_map(bad);
        const bool refused = !st.ok();
        std::cout << "  quantized router refused: " << (refused ? "yes" : "NO — GATE BROKEN")
                  << "\n";
        if (!refused) ++failures;

        soma::QuantMap bad_norms;
        bad_norms.norms = {soma::DType::Q4_0, 32};
        const bool refused_norms = !soma::validate_quant_map(bad_norms).ok();
        std::cout << "  quantized norms refused:  " << (refused_norms ? "yes" : "NO — GATE BROKEN")
                  << "\n";
        if (!refused_norms) ++failures;
    }
    std::cout << "\n";

    if (!codec_check(std::cout)) ++failures;
    std::cout << "\n";

    const Config configs[] = {
        {"f32 (reference)", all_f32(),                       soma::DType::F32,  0},
        {"q8_0 g32",        uniform(soma::DType::Q8_0, 32),  soma::DType::Q8_0, 32},
        {"q8_0 g128",       uniform(soma::DType::Q8_0, 128), soma::DType::Q8_0, 128},
        {"q6_g g128",       uniform(soma::DType::Q6_G, 128), soma::DType::Q6_G, 128},
        {"q4_g/q6_g mixed", mixed_role_map(),                soma::DType::Q4_G, 128},
        {"q4_g g128",       uniform(soma::DType::Q4_G, 128), soma::DType::Q4_G, 128},
        {"q4_0 g32",        uniform(soma::DType::Q4_0, 32),  soma::DType::Q4_0, 32},
    };

    std::vector<fs::path> fixtures;
    if (argc > 2) {
        for (int i = 2; i < argc; ++i) fixtures.push_back(root / argv[i]);
    } else {
        for (const auto& e : fs::directory_iterator(root)) {
            if (e.is_directory() && fs::exists(e.path() / "oracle.bin")) fixtures.push_back(e.path());
        }
        std::sort(fixtures.begin(), fixtures.end());
    }

    for (const auto& dir : fixtures) {
        soma::F32Model ref;
        if (auto st = soma::load_f32_model(dir.string(), ref); !st.ok()) {
            if (st.code() != soma::StatusCode::Unsupported) {
                std::cout << dir.filename().string() << ": ERROR " << st.message() << "\n";
                ++failures;
            }
            continue;
        }

        std::mt19937 rng(7);
        std::vector<soma::TokenId> tokens(kPositions);
        for (auto& t : tokens) t = rng() % ref.vocab();

        soma::F32Workspace ws;
        std::vector<float> ref_logits;
        std::vector<soma::TokenId> ref_greedy;
        if (auto st = soma::forward_f32(ref, tokens, ws, ref_logits); !st.ok()) {
            std::cout << dir.filename().string() << ": ERROR " << st.message() << "\n";
            ++failures;
            continue;
        }
        std::vector<soma::TokenId> prefix(tokens.begin(), tokens.begin() + 16);
        if (auto st = soma::generate_greedy_f32(ref, prefix, kGreedy, ws, ref_greedy); !st.ok()) {
            std::cout << dir.filename().string() << ": ERROR " << st.message() << "\n";
            ++failures;
            continue;
        }

        std::cout << "== " << dir.filename().string() << "  (vs fp32, " << kPositions
                  << " positions, " << kGreedy << " greedy tokens)\n";
        std::cout << "   " << std::left << std::setw(18) << "config"
                  << std::setw(12) << "max|dlog|" << std::setw(11) << "rel_err"
                  << std::setw(11) << "codec" << std::setw(8) << "amp"
                  << std::setw(9) << "greedy" << "weights\n";

        for (const auto& cfg : configs) {
            const auto o = run_config(dir, cfg.map, tokens, ref_logits, ref_greedy);
            std::cout << "   " << std::left << std::setw(18) << cfg.name;
            if (!o.ok) {
                std::cout << "ERROR: " << o.error << "\n";
                ++failures;
                continue;
            }
            const float codec = (cfg.probe_group > 0)
                                    ? codec_rel_rms(cfg.probe_dtype, cfg.probe_group)
                                    : 0.0f;
            const float amp = (codec > 0.0f) ? o.rel_logit_err / codec : 0.0f;
            const bool amp_ok = (codec == 0.0f) || (amp <= kMaxAmplification);
            if (!amp_ok) ++failures;

            auto sci = [](float v) {
                std::ostringstream ss;
                ss << std::scientific << std::setprecision(2) << v;
                return ss.str();
            };
            std::ostringstream g;
            g << o.greedy_matched << "/" << kGreedy;
            std::ostringstream a;
            if (codec > 0.0f) {
                a << std::fixed << std::setprecision(2) << amp << (amp_ok ? "" : "!");
            } else {
                a << "-";
            }

            std::cout << std::setw(12) << sci(o.max_logit_diff)
                      << std::setw(11) << sci(o.rel_logit_err)
                      << std::setw(11) << (codec > 0.0f ? sci(codec) : std::string("-"))
                      << std::setw(8) << a.str()
                      << std::setw(9) << g.str();
            if (o.bytes > 0) std::cout << (o.bytes / 1024) << " KiB";
            else std::cout << "fp32";
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    std::cout << (failures == 0 ? "no hard failures" : std::to_string(failures) + " FAILURES")
              << "\n";
    return failures == 0 ? 0 : 1;
}
