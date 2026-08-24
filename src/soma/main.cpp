// Soma — the engine executable.
//
//   soma serve   --model-dir DIR [--port N] [--host H] [--ctx-size N] ...
//   soma plan    --model-dir DIR [--json]
//   soma conform --model-dir DIR [--json]
//
// `plan` exists as a subcommand of the same binary rather than a separate tool
// because the planner it runs is the one the server runs: an operator asking
// "what will this do on this host?" and the engine deciding what to do must not
// be able to disagree.

#include "soma/arch_ir.hpp"
#include "soma/conformance.hpp"
#include "soma/plan.hpp"
#include "soma/quant_format.hpp"
#include "soma/safetensors.hpp"
#include "soma/serve.hpp"
#include "soma/tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int usage() {
    std::cerr << "usage:\n"
                 "  soma serve   --model-dir DIR [--host H] [--port N] [--ctx-size N]\n"
                 "               [--kv-slots N] [--max-batch N]\n"
                 "               [--speculative off|auto|dspark] [--speculative-tokens N]\n"
                 "               [--dspark-confidence-threshold 0..1]\n"
                 "               [--generation-timeout SECONDS]\n"
                 "               [--ram-budget BYTES] [--pin BYTES] [--kv-dir DIR]\n"
                 "               [--quant-dense DTYPE]  quantize the RESIDENT half at load\n"
                 "               [--served-name NAME]\n"
                 "  soma plan    --model-dir DIR [--json]\n"
                 "               [--quant DTYPE] [--expert-down DTYPE] [--quant-dense DTYPE]\n"
                 "               [--group N]\n"
                 "               [--ram SIZE] [--ram-free SIZE] [--disk-bw SIZE] [--ctx N]\n"
                 "               [--kv-slots N]\n"
                 "               [--speculative off|auto|dspark]\n"
                 "               [--min-tok-s RATE]   slowest generation you will accept "
                 "(default 1.0)\n"
                 "               the verdict is a property of (model, quantization, host);\n"
                 "               these ASK about a quantization and a host, and convert nothing\n"
                 "  soma conform --model-dir DIR [--json]\n";
    return 2;
}

// ── conform ──────────────────────────────────────────────────────────────────
//
// The admission ladder, for the stages that can honestly run against a converted
// container and nothing else. The rest are REPORTED AS SKIPPED, with what they
// would need — a `passed` row nobody earned is worse than an absent one, because
// the verdict then looks validated when it was only computed.
//
// Runs here rather than in control for the same reason `plan` does: the codec
// under test is the one the engine uses, and a second implementation in another
// process is how they come to disagree.

/// Theoretical bits per weight for a group-scale format.
///
/// Written out as a formula rather than taken from quantized_tensor_bytes(),
/// which IS the implementation — comparing the implementation against itself
/// would pass through any packing bug that was consistent about its size.
double theoretical_bpw(soma::DType d, std::uint32_t group) {
    const double g = static_cast<double>(group);
    switch (d) {
    case soma::DType::Q8_0:
        return 8.0 + 32.0 / g; // one fp32 scale per group
    case soma::DType::Q4_0:
        return 4.0 + 32.0 / g;
    case soma::DType::Q6_G:
        return 6.0 + 32.0 / g;
    case soma::DType::Q4_G:
        return 4.0 + 64.0 / g; // scale AND min
    default:
        return 0.0;
    }
}

/// Relative-RMS ceiling per format, generous by ~3x against the measured values
/// in docs/roadmap.md's G1 table.
///
/// Generous on purpose: this is not an accuracy metric, it is a PACKING check. A
/// mis-packed nibble, a wrong group stride or a broken accumulation lands at
/// rel_rms near 1.0 or above, so the gap between "correct on unfamiliar weights"
/// and "broken" is two orders of magnitude wide. A tight bound here would fail
/// honest models and catch nothing extra.
double rel_rms_ceiling(soma::DType d) {
    switch (d) {
    case soma::DType::Q8_0:
        return 0.02;
    case soma::DType::Q6_G:
        return 0.08;
    case soma::DType::Q4_G:
    case soma::DType::Q4_0:
        return 0.30;
    default:
        return 1.0;
    }
}

bool parse_dtype_name(const std::string& text, soma::DType& out) {
    for (const auto d : {soma::DType::Q8_0,
                         soma::DType::Q4_0,
                         soma::DType::Q4_G,
                         soma::DType::Q6_G,
                         soma::DType::F32}) {
        if (text == soma::to_string(d)) {
            out = d;
            return true;
        }
    }
    return false;
}

struct StageResult {
    std::string stage;
    std::string status; ///< passed | failed | skipped
    nlohmann::json detail = nlohmann::json::object();
};

/// Quantize the container's own dense weights and measure what came back.
///
/// The container's dense tensors are the right subject: they are F32 on disk, they
/// are THIS model's real weight distribution rather than a synthetic one, and they
/// are bounded — the experts are the gigabytes, and they are already quantized, so
/// there is no fp32 original to compare them against anyway.
StageResult stage_quant_codec(const std::filesystem::path& dir) {
    StageResult r{"quant_codec", "skipped", {}};

    const auto meta_path = dir / "container_meta.json";
    std::ifstream meta_in(meta_path, std::ios::binary);
    if (!meta_in) {
        r.detail["reason"] = "no container_meta.json; this is not a converted container";
        return r;
    }
    nlohmann::json meta;
    try {
        meta_in >> meta;
    } catch (const std::exception& e) {
        r.status = "failed";
        r.detail["reason"] = std::string("container_meta.json is unreadable: ") + e.what();
        return r;
    }

    const auto group = meta.value("group", 128u);
    std::vector<std::pair<std::string, soma::DType>> formats;
    for (const char* key : {"dtype_gate_up", "dtype_down"}) {
        const auto name = meta.value(key, std::string{});
        soma::DType d{};
        if (!name.empty() && parse_dtype_name(name, d)) {
            const auto seen = std::find_if(
                formats.begin(), formats.end(), [&](const auto& p) { return p.second == d; });
            if (seen == formats.end()) formats.emplace_back(name, d);
        }
    }
    if (formats.empty()) {
        r.detail["reason"] = "container declares no quantized formats";
        return r;
    }

    soma::SafeTensors dense;
    if (auto st = dense.open_dir(dir.string()); !st.ok()) {
        r.detail["reason"] = "no dense tensor set: " + st.message();
        return r;
    }

    bool passed = true;
    auto rows_json = nlohmann::json::array();
    std::vector<float> back;

    for (const auto& [name, dtype] : formats) {
        double sum2 = 0.0, ref2 = 0.0, expect_bits = 0.0;
        std::size_t weights = 0, packed_bytes = 0, tensors = 0;
        std::uint32_t group_lo = 0, group_hi = 0;

        for (const auto& tname : dense.names()) {
            const auto* t = dense.find(tname);
            // Rank 2 and F32 only: quantization is defined along `cols`, and a
            // 1-D norm weight has no cols to group.
            if (t == nullptr || t->rank() != 2 || t->dtype != soma::DType::F32) continue;
            const auto src = t->f32();
            if (src.empty()) continue;

            const auto rows = static_cast<std::uint32_t>(t->dim(0));
            const auto cols = static_cast<std::uint32_t>(t->dim(1));
            soma::QTensor q;
            if (!soma::quantize_tensor(src, rows, cols, dtype, group, q).ok()) continue;
            back.resize(src.size());
            if (!soma::dequantize_tensor(q, back).ok()) continue;

            for (std::size_t i = 0; i < src.size(); ++i) {
                const double e = static_cast<double>(back[i]) - src[i];
                sum2 += e * e;
                ref2 += static_cast<double>(src[i]) * src[i];
            }
            weights += src.size();
            packed_bytes += q.data.size();
            // Per tensor, against ITS OWN effective group. quantize_tensor
            // reduces the requested group to the largest divisor of `cols` that
            // fits, so a container whose tensors have different widths uses
            // several groups at once — the fixture's dense weights use 64 and 32.
            // One group in the expectation would fail every tensor that is not
            // the widest, for being correct.
            expect_bits += theoretical_bpw(dtype, q.group) * static_cast<double>(src.size());
            group_lo = (group_lo == 0) ? q.group : std::min(group_lo, q.group);
            group_hi = std::max(group_hi, q.group);
            ++tensors;
        }

        if (weights == 0) {
            r.detail["reason"] = "dense.safetensors holds no 2-D F32 weights";
            return r;
        }

        const double rel = ref2 > 0.0 ? std::sqrt(sum2 / ref2) : 0.0;
        const double bpw = 8.0 * static_cast<double>(packed_bytes) / static_cast<double>(weights);
        const double want_bpw = expect_bits / static_cast<double>(weights);
        const double ceiling = rel_rms_ceiling(dtype);

        const bool bpw_ok = std::fabs(bpw - want_bpw) < 0.02;
        const bool rel_ok = rel <= ceiling;
        passed = passed && bpw_ok && rel_ok;

        rows_json.push_back(nlohmann::json{{"dtype", name},
                                           {"group_min", group_lo},
                                           {"group_max", group_hi},
                                           {"tensors", tensors},
                                           {"weights", weights},
                                           {"bits_per_weight", bpw},
                                           {"bits_per_weight_expected", want_bpw},
                                           {"rel_rms", rel},
                                           {"rel_rms_ceiling", ceiling},
                                           {"passed", bpw_ok && rel_ok}});
    }

    r.status = passed ? "passed" : "failed";
    r.detail = nlohmann::json{{"formats", rows_json}};
    return r;
}

/// Admission ladder stage 1: the engine's forward against a `transformers`
/// oracle, on a tiny-random model carrying THIS architecture.
///
/// The fixture is a SUBDIRECTORY of the container, not the container itself, and
/// the distinction is the point: the oracle is built from RANDOM weights with the
/// real config, so this validates the ARCHITECTURE rather than the admitted
/// checkpoint. A real checkpoint can be approximately right in ways that hide a
/// bug for weeks; a tiny-random one is either exactly right or obviously wrong.
StageResult stage_fp32_tiny_tf(const std::filesystem::path& dir) {
    StageResult r{"fp32_tiny_tf", "skipped", {}};

    const auto fixture = dir / "conformance";
    if (!std::filesystem::exists(fixture / "oracle.bin")) {
        r.detail["reason"] = "no conformance fixture in this container; admission builds one with "
                             "tools/admission/make_oracle.py when transformers is available";
        return r;
    }

    const auto c = soma::run_fp32_conformance(fixture.string());
    if (c.skipped) {
        // No backend for this attention family. A gap in coverage rather than a
        // failure — reporting it red would make the ladder permanently red, and
        // a permanently red ladder is an ignored one.
        r.detail["reason"] = "no fp32 backend for this architecture: " + c.detail;
        return r;
    }

    r.status = c.passed() ? "passed" : "failed";
    r.detail = nlohmann::json{{"logits_pass", c.logits_pass},
                              {"greedy_pass", c.greedy_pass},
                              {"max_abs", c.max_abs},
                              {"max_abs_at_position", c.max_at_pos},
                              // Always reported, because it BISECTS a failure:
                              // clean at 0 and growing with t is positional;
                              // already wrong at 0 is not.
                              {"max_abs_pos0", c.max_abs_pos0},
                              {"mean_abs", c.mean_abs},
                              {"tolerance_max_abs", soma::kConformanceMaxAbsDiff},
                              {"tolerance_mean_abs", soma::kConformanceMaxMeanDiff},
                              {"greedy_tokens_matched", c.matched_tokens}};
    if (!c.detail.empty()) r.detail["error"] = c.detail;
    if (!c.greedy_pass) r.detail["first_bad_token"] = c.first_bad_token;
    return r;
}

/// Ladder stage 2. The one that looks at the weights an operator will ship.
///
/// Stage 1 runs tiny-RANDOM weights and proves the architecture. This runs the
/// container as it will actually be served — same quant map, same streaming
/// expert cache — against a bf16 pass over the real checkpoint. It is the only
/// stage whose failure can mean "this quantization of this model is not good
/// enough", which is what `reject` is supposed to encode.
StageResult stage_real_logit_kl(const std::filesystem::path& dir) {
    StageResult r{"real_logit_kl", "skipped", {}};

    const auto reference = dir / "conformance" / "reference.bin";
    if (!std::filesystem::exists(reference)) {
        r.detail["reason"] = "no bf16 reference in this container; admission builds one with "
                             "tools/admission/make_reference.py when transformers is available";
        return r;
    }

    // Bounded on purpose. The reference carries 512 positions and the streaming
    // forward is the expensive part of admission; the cap keeps a stage that
    // reports a distribution from dominating a pipeline that also has to convert
    // weights. Raising it tightens the p95 estimate, nothing else.
    const auto c = soma::run_real_logit_kl(dir.string(),
                                           reference.string(),
                                           /*cache_gib=*/4,
                                           /*max_positions=*/256);
    if (c.skipped) {
        r.detail["reason"] = c.detail;
        return r;
    }

    r.status = c.passed ? "passed" : "failed";
    r.detail = nlohmann::json{{"mean_kl", c.mean_kl},
                              {"median_kl", c.median_kl},
                              {"p95_kl", c.p95_kl},
                              {"max_kl", c.worst_kl},
                              {"max_kl_at_position", c.worst_at},
                              {"top1_agreement_pct", c.top1_agreement_pct},
                              {"positions", c.positions},
                              {"tolerance_mean_kl", soma::kRealLogitKlMeanMax},
                              {"tolerance_p95_kl", soma::kRealLogitKlP95Max},
                              // Measured on real weights rather than estimated
                              // from headers — the quantity the streamable
                              // verdict rests on.
                              {"bytes_per_token", c.positions ? c.bytes_read / c.positions : 0},
                              {"cache_hit_rate_pct", c.cache_hit_rate_pct}};
    if (!c.detail.empty()) r.detail["error"] = c.detail;
    // Says which remedy applies. A degenerate result is not a quant-map problem,
    // and an operator who requantizes in response to one burns an hour re-running
    // conversion to reach the same place.
    if (!c.passed) {
        r.detail["finding"] = c.degenerate ? "not_quantization" : "quantization";
    }
    return r;
}

StageResult stage_tokenizer_roundtrip(const std::filesystem::path& dir) {
    StageResult r{"tokenizer_roundtrip", "skipped", {}};

    if (std::filesystem::exists(dir / "tokenizer.unsupported")) {
        std::ifstream in(dir / "tokenizer.unsupported");
        std::string why;
        std::getline(in, why);
        // A tokenizer the compiler REFUSED is a known, recorded state — the model
        // serves token ids. Not a conformance failure, and not a silent pass.
        r.detail["reason"] = "tokenizer not compiled: " + why;
        return r;
    }

    soma::CompiledTokenizer tok;
    if (auto st = tok.open((dir / "tokenizer.soma").string()); !st.ok()) {
        r.detail["reason"] = "no compiled tokenizer: " + st.message();
        return r;
    }
    std::vector<soma::TokenizerOracleCase> oracle;
    if (auto st = soma::read_tokenizer_oracle((dir / "tokenizer_oracle.bin").string(), oracle);
        !st.ok()) {
        r.detail["reason"] = st.message();
        return r;
    }

    soma::RoundTripResult rt;
    if (auto st = soma::verify_roundtrip(tok, oracle, rt); !st.ok()) {
        r.status = "failed";
        r.detail["reason"] = st.message();
        return r;
    }
    r.status = rt.clean() ? "passed" : "failed";
    r.detail = nlohmann::json{
        {"cases", rt.cases}, {"encode_ok", rt.encode_ok}, {"decode_ok", rt.decode_ok}};
    if (!rt.first_failure.empty()) r.detail["first_failure"] = rt.first_failure;
    return r;
}

int cmd_conform(int argc, char** argv) {
    std::string dir;
    bool as_json = false;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model-dir" && i + 1 < argc)
            dir = argv[++i];
        else if (a == "--json")
            as_json = true;
    }
    if (dir.empty()) return usage();
    const std::filesystem::path root(dir);

    std::vector<StageResult> stages;
    stages.push_back(stage_tokenizer_roundtrip(root));
    stages.push_back(stage_quant_codec(root));

    // The stages that need something this process does not have, named with what
    // that is. They are recorded as SKIPPED, never as passed.
    stages.push_back(stage_fp32_tiny_tf(root));
    stages.push_back(stage_real_logit_kl(root));
    stages.push_back(
        {"accuracy_floor", "skipped", {{"reason", "no downstream task harness exists yet"}}});

    // A failed stage is a REJECT verdict, not a failed request: the operator
    // asked whether Soma can run this model, and "no, here is why" is an answer.
    bool any_failed = false, any_ran = false;
    for (const auto& s : stages) {
        if (s.status == "failed") any_failed = true;
        if (s.status != "skipped") any_ran = true;
    }

    if (as_json) {
        auto out = nlohmann::json::array();
        for (const auto& s : stages) {
            out.push_back(
                nlohmann::json{{"stage", s.stage}, {"status", s.status}, {"detail", s.detail}});
        }
        std::cout << nlohmann::json{{"stages", out},
                                    {"passed", any_ran && !any_failed},
                                    {"ran", any_ran}}
                         .dump()
                  << "\n";
    } else {
        for (const auto& s : stages) {
            std::cout << "  " << std::left << std::setw(22) << s.stage << std::setw(9) << s.status
                      << s.detail.dump() << "\n";
        }
    }
    // Exit 0 even when a stage fails: the finding is the output. Non-zero is
    // reserved for "could not run", which is a different thing entirely and the
    // caller has to tell them apart.
    return 0;
}

/// Parse "24GiB", "16G", "8192MB", "1073741824".
///
/// Strict: an unparseable size is an ERROR, not a fallback to the default. A
/// typo'd budget that silently plans against 16 GiB answers a question nobody
/// asked, and the answer looks exactly like a real one.
bool parse_size(const std::string& text, std::uint64_t& out) {
    if (text.empty()) return false;
    std::size_t i = 0;
    std::uint64_t n = 0;
    while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
        n = n * 10 + static_cast<std::uint64_t>(text[i] - '0');
        ++i;
    }
    if (i == 0) return false;

    std::string suffix;
    for (; i < text.size(); ++i)
        suffix += static_cast<char>(std::tolower(text[i]));

    // KiB/MiB/GiB/TiB are 1024-based; KB/MB/GB/TB are 1000-based. Both spellings
    // appear in this codebase's own docs, and quietly treating GB as GiB is a 7%
    // error on the one number the verdict divides by.
    std::uint64_t mul = 1;
    if (suffix.empty() || suffix == "b")
        mul = 1;
    else if (suffix == "k" || suffix == "kib")
        mul = 1024ull;
    else if (suffix == "kb")
        mul = 1000ull;
    else if (suffix == "m" || suffix == "mib")
        mul = 1024ull * 1024;
    else if (suffix == "mb")
        mul = 1000ull * 1000;
    else if (suffix == "g" || suffix == "gib")
        mul = 1024ull * 1024 * 1024;
    else if (suffix == "gb")
        mul = 1000ull * 1000 * 1000;
    else if (suffix == "t" || suffix == "tib")
        mul = 1024ull * 1024 * 1024 * 1024;
    else if (suffix == "tb")
        mul = 1000ull * 1000 * 1000 * 1000;
    else
        return false;

    out = n * mul;
    return true;
}

int cmd_plan(int argc, char** argv) {
    std::string dir;
    bool as_json = false;

    // The verdict is a property of (model, quantization, HOST). `--model-dir`
    // varies the first; these vary the other two, which until now were fixed
    // constants — so `plan` could only ever evaluate that function at one point.
    //
    // Both are HYPOTHETICAL. Nothing here converts a weight or reserves a byte:
    // the point of a headers-only planner is to answer "would this be worth
    // converting?" before spending the hours, and for any quantization but the
    // default that question could not be asked at all.
    std::string q_gate_up, q_down, q_dense;
    std::uint32_t q_group = 0;
    std::uint64_t ram_total = 0, ram_free = 0, disk_bw = 0;
    std::uint32_t ctx = 0, kv_slots = 0;
    float min_tok_s = 0.0f; ///< 0 = unstated; compute_plan applies the default
    enum class PlanSpeculation { Off, Auto, Required };
    PlanSpeculation speculation = PlanSpeculation::Off;

    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string{};
        };
        if (a == "--model-dir" && i + 1 < argc)
            dir = next();
        else if (a == "--json")
            as_json = true;
        else if (a == "--quant" && i + 1 < argc)
            q_gate_up = next();
        else if (a == "--expert-down" && i + 1 < argc)
            q_down = next();
        else if (a == "--quant-dense" && i + 1 < argc)
            q_dense = next();
        else if (a == "--group" && i + 1 < argc)
            q_group = static_cast<std::uint32_t>(std::strtoul(next().c_str(), nullptr, 10));
        else if (a == "--ram" && i + 1 < argc) {
            if (!parse_size(next(), ram_total)) {
                std::cerr << "plan: --ram wants a size like 24GiB\n";
                return 2;
            }
        } else if (a == "--ram-free" && i + 1 < argc) {
            if (!parse_size(next(), ram_free)) {
                std::cerr << "plan: --ram-free wants a size like 20GiB\n";
                return 2;
            }
        } else if (a == "--disk-bw" && i + 1 < argc) {
            if (!parse_size(next(), disk_bw)) {
                std::cerr << "plan: --disk-bw wants a size like 3GB (per second)\n";
                return 2;
            }
        } else if (a == "--ctx" && i + 1 < argc) {
            ctx = static_cast<std::uint32_t>(std::strtoul(next().c_str(), nullptr, 10));
        } else if (a == "--kv-slots" && i + 1 < argc) {
            kv_slots = static_cast<std::uint32_t>(std::strtoul(next().c_str(), nullptr, 10));
            if (kv_slots == 0) {
                std::cerr << "plan: --kv-slots wants a positive integer\n";
                return 2;
            }
        } else if (a == "--speculative" && i + 1 < argc) {
            const auto value = next();
            if (value == "off")
                speculation = PlanSpeculation::Off;
            else if (value == "auto")
                speculation = PlanSpeculation::Auto;
            else if (value == "dspark")
                speculation = PlanSpeculation::Required;
            else {
                std::cerr << "plan: --speculative wants off, auto, or dspark\n";
                return 2;
            }
        } else if (a == "--min-tok-s") {
            const auto text = next();
            char* end = nullptr;
            errno = 0;
            min_tok_s = std::strtof(text.c_str(), &end);
            if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
                !std::isfinite(min_tok_s) || !(min_tok_s > 0.0f)) {
                std::cerr << "plan: --min-tok-s wants a positive rate like 0.1\n";
                return 2;
            }
        }
    }
    if (dir.empty()) return usage();

    // Reject a dtype we cannot honour rather than planning at the default and
    // reporting a number for a quantization nobody asked for.
    for (const auto* name : {&q_gate_up, &q_down, &q_dense}) {
        soma::DType parsed{};
        if (!name->empty() && !parse_dtype_name(*name, parsed)) {
            std::cerr << "plan: unknown dtype '" << *name << "'\n";
            return 2;
        }
    }

    // Host budget from the machine this runs on. The verdict is a property of
    // (model, quantization, HOST) — running `plan` on a different box than the
    // one that will serve gives a different and equally correct answer, which
    // is why the registry stores the admission host's verdict and the node
    // re-derives its own.
    soma::HostBudget host;
    host.ram_total_bytes = 16ull << 30;
    host.ram_free_bytes = 8ull << 30;
    host.disk_bandwidth = 1230ull * 1000 * 1000;

    // An EXPLICIT budget is a statement about what the engine may have, so
    // `--ram 24GiB` alone means 24 GiB of budget rather than 24 with half
    // reserved. The 16/8 default keeps modelling a real machine with an OS on
    // it; silently halving a number the operator typed would answer a different
    // question and look identical. `--ram-free` states the split when it
    // matters — it is what compute_plan actually divides by.
    if (ram_total > 0) {
        host.ram_total_bytes = ram_total;
        host.ram_free_bytes = ram_total;
    }
    if (ram_free > 0) host.ram_free_bytes = ram_free;
    if (disk_bw > 0) host.disk_bandwidth = disk_bw;
    if (ctx > 0) host.ctx_size = ctx;
    if (kv_slots > 0) host.kv_slots = kv_slots;
    if (speculation == PlanSpeculation::Required) {
        host.speculative = true;
    } else if (speculation == PlanSpeculation::Auto) {
        // Match serve's conservative auto policy.  Absence, malformed metadata,
        // or an unprofiled draft all mean autoregressive planning; explicit
        // `dspark` remains the way to inspect the auxiliary footprint before a
        // speed profile exists.
        try {
            std::ifstream meta_in(std::filesystem::path(dir) / "container_meta.json");
            nlohmann::json meta;
            if (meta_in && (meta_in >> meta)) {
                host.speculative = meta.value("dspark", std::string{}) == "present" &&
                                   meta.value("dspark_profiled_speedup", 0.0) >= 1.05;
            }
        } catch (const std::exception&) {
            host.speculative = false;
        }
    }
    // Left at 0 when unstated, which compute_plan reads as "use the default" —
    // see HostBudget::min_tok_s. Passing 1.0 here instead would erase the
    // distinction between a floor someone chose and one they inherited.
    host.min_tok_s = min_tok_s;

    soma::PlanDocument doc;
    // Resolve a converted container's copied config.json plus conversion metadata,
    // or adapt an upstream config.json directly. Both are legitimate inputs:
    // an operator asking "what will this do here?" usually has the HF checkpoint,
    // not a container, and refusing them would make `plan` useless exactly when
    // it is most wanted — before conversion.
    // Built in the SHAPE a container_meta.json uses and handed to the same
    // applier, rather than mapping dtype names to roles a second time here.
    // That mapping carries a rule — gate and up must share a dtype, because the
    // converter interleaves them into one range — and a second copy of it would
    // let `plan --quant` describe a container the converter cannot produce.
    std::string overlay;
    if (!q_gate_up.empty() || !q_down.empty() || !q_dense.empty() || q_group > 0) {
        nlohmann::json o = nlohmann::json::object();
        if (!q_gate_up.empty()) o["dtype_gate_up"] = q_gate_up;
        if (!q_down.empty()) o["dtype_down"] = q_down;
        // Embeddings, attention projections and shared experts. One flag rather
        // than three: they are the "resident, not routed" family, and the reason
        // to quantize any of them is the same one.
        if (!q_dense.empty()) o["dtype_dense"] = q_dense;
        if (q_group > 0) o["group"] = q_group;
        overlay = o.dump();
    }

    if (auto st = soma::compute_plan(dir, host, doc, overlay); !st.ok()) {
        std::string cfg_text;
        soma::ArchIr arch;
        std::ifstream in(std::filesystem::path(dir) / "config.json", std::ios::binary);
        if (!in) {
            std::cerr << "plan failed: " << st.message() << "\n";
            return 1;
        }
        cfg_text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        if (auto a = soma::adapt_hf_config(cfg_text, arch); !a.ok()) {
            std::cerr << "plan failed: " << a.message() << "\n";
            return 1;
        }
        // The same overlay. Without this the fallback silently ignores --quant
        // and reports default-quantization numbers for a flag the operator set —
        // which is worse than refusing, because the output looks like an answer.
        if (!overlay.empty()) {
            if (auto q = soma::apply_container_quant(overlay, arch); !q.ok()) {
                std::cerr << "plan failed: " << q.message() << "\n";
                return 1;
            }
        }
        if (auto p = soma::compute_plan(arch, host, doc); !p.ok()) {
            std::cerr << "plan failed: " << p.message() << "\n";
            return 1;
        }
    }
    if (as_json) {
        std::string js;
        (void)soma::serialize_plan(doc, js);
        std::cout << js << "\n";
    } else {
        // Both, when they differ. A reader who sees only "reject" cannot tell
        // whether the model is uneconomic or merely unimplemented, and those
        // lead to completely different next actions — requantize on a bigger
        // host, versus write a backend.
        std::cout << "verdict      " << soma::to_string(doc.verdict) << "\n";
        if (doc.economic_verdict != doc.verdict) {
            std::cout << "economics    " << soma::to_string(doc.economic_verdict)
                      << "   (what the economics alone say; see reason)\n";
        }
        std::cout << "reason       " << doc.verdict_reason << "\n"
                  << "routed       " << (doc.total_routed_bytes >> 20) << " MiB\n"
                  << "bytes/token  " << (doc.bytes_per_token >> 20) << " MiB\n"
                  << "max_batch    " << doc.max_batch << "\n";
        // Only when it is not plain text, and never silently.
        //
        // Soma serves the text stack. Doing that with a vision-capable
        // checkpoint is legitimate; doing it WITHOUT SAYING SO is a model
        // answering about an image it never received, and nothing
        // downstream can tell that apart from a model that simply got the
        // answer wrong.
        if (doc.modality != "text") {
            std::cout << "modality     " << doc.modality
                      << "   (SERVED TEXT-ONLY; the " << doc.vision_layers
                      << "-layer, " << doc.vision_hidden
                      << "-wide vision tower is neither converted nor served)\n";
        }
    }
    return 0;
}

int cmd_serve(int argc, char** argv) {
    soma::ServeConfig cfg;
    if (auto st = soma::parse_serve_config(argc, argv, cfg); !st.ok()) {
        std::cerr << st.message() << "\n";
        return usage();
    }

    soma::ServeServer server;
    if (auto st = server.open(cfg); !st.ok()) {
        std::cerr << "open failed: " << st.message() << "\n";
        return 1;
    }

    // Printed AFTER open() succeeds and the routes are live, so a supervisor
    // that races the log against the port finds the port already accepting.
    // Readiness is still the /health poll — this line is for humans.
    std::cout << "soma serve listening on " << cfg.host << ":" << cfg.port
              << "  model=" << cfg.model_dir
              << "  verdict=" << soma::to_string(server.plan().verdict) << std::endl;

    if (auto st = server.listen(); !st.ok()) {
        std::cerr << st.message() << "\n";
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    if (cmd == "serve") return cmd_serve(argc - 2, argv + 2);
    if (cmd == "plan") return cmd_plan(argc - 2, argv + 2);
    if (cmd == "conform") return cmd_conform(argc - 2, argv + 2);
    if (cmd == "--help" || cmd == "-h") {
        usage();
        return 0;
    }
    std::cerr << "unknown command '" << cmd << "'\n";
    return usage();
}
