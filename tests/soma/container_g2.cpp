// Soma — G2a: container round-trip, and the verdict function against the doc.
//
// Two independent checks:
//
//   1. Every expert read back from the container is byte-identical to the same
//      expert quantized by the ENGINE. convert.py implements the formats a second
//      time, in Python; a divergence in either direction shows up here rather than
//      both sides agreeing on the same mistake.
//
//   2. The verdict function reproduces the worked table in schemas/arch-ir.md §8.
//      The doc makes specific claims — Mixtral resident-only at q4 on 32 GB,
//      Qwen3 flipping to stream at bf16 — and code that disagrees with them means
//      one of the two is wrong.
//
// Usage: container_g2 <fixtures_root>

#include "soma/expert_store.hpp"
#include "soma/f32_model.hpp"
#include "soma/memory_hierarchy.hpp"
#include "soma/plan.hpp"
#include "soma/safetensors.hpp"
#include "soma/quant_format.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t kGiB = 1024ull * 1024 * 1024;

soma::QuantMap container_map() {
    soma::QuantMap m;
    m.expert_gate = {soma::DType::Q4_G, 128};
    m.expert_up = {soma::DType::Q4_G, 128};
    m.expert_down = {soma::DType::Q6_G, 128};
    return m;
}

// ── 1. container round-trip ──────────────────────────────────────────────────

int check_container(const fs::path& fixture, const fs::path& container) {
    soma::F32Model model;
    if (auto st = soma::load_f32_model(fixture.string(), model); !st.ok()) {
        // An unsupported family is a coverage gap, not a container fault. The
        // converter wrote these fine; the C++ IR adapter gains MLA at G4.
        // Counting it as a failure would make the suite permanently red and
        // therefore ignored.
        const bool gap = (st.code() == soma::StatusCode::Unsupported);
        std::cout << (gap ? "   SKIP: " : "   load failed: ") << st.message() << "\n";
        return gap ? 0 : 1;
    }

    // Re-quantize with the ENGINE first, so its IR describes the container's
    // precision. Opening with the all-f32 IR is now correctly refused.
    soma::F32Model qmodel;
    if (auto st = soma::load_f32_model(fixture.string(), qmodel, container_map()); !st.ok()) {
        std::cout << "   quantized load failed: " << st.message() << "\n";
        return 1;
    }

    // Can the CONTAINER be loaded as a model at all?
    //
    // Everything below opens the expert store and compares payload bytes, which
    // says nothing about the dense half — so a container missing its attention
    // weights passed this test for as long as it existed. `convert.py`'s
    // allow-list was GQA-shaped and silently dropped MLA's kv_a/kv_b, the
    // dense-layer MLPs and the noaux_tc router bias; `soma serve` on a DeepSeek
    // container died with "binding attention weights failed at layer 0", and
    // nothing here noticed because nothing here ever tried (roadmap D18).
    //
    // G4 passed on the fp32 SOURCE path, which has the tensors. The container
    // path is the one production serves.
    soma::F32Model from_container;
    if (auto st = soma::load_f32_model(container.string(), from_container, container_map());
        !st.ok()) {
        std::cout << "   CONTAINER WILL NOT LOAD: " << st.message() << "\n"
                  << "   (the round-trip below only checks experts; this is the dense half)\n";
        return 1;
    }

    soma::ExpertStore store;
    if (auto st = store.open(container.string(), qmodel.arch); !st.ok()) {
        std::cout << "   open failed: " << st.message() << "\n";
        return 1;
    }
    const auto& h = store.header();

    const auto d = model.arch.topology.d_model;
    const auto fi = model.arch.ffn.expert_intermediate;

    std::vector<std::byte> buf(h.expert_bytes > 0 ? h.expert_bytes : 1u << 20);
    int mismatches = 0, compared = 0;
    // Separate from `mismatches`, which the byte-comparison summary reports.
    // Sharing one counter made that line print "MISMATCH" when the BYTES were
    // fine and the forward had diverged — a passing check labelled with another
    // check's failure.
    bool forward_diverged = false;
    std::size_t first_bad_byte = 0;

    for (std::uint32_t l = 0; l < h.n_layers && mismatches == 0; ++l) {
        if (!model.arch.is_moe_layer(l)) continue;
        for (std::uint32_t e = 0; e < h.n_experts; ++e) {
            const auto loc = store.locate(l, e);
            if (loc.length == 0) continue;
            if (store.read(l, e, buf) != soma::StatusCode::Ok) {
                std::cout << "   read failed at layer " << l << " expert " << e << "\n";
                return 1;
            }

            // Expected: gate ++ up ++ down, in that order.
            std::vector<std::byte> want;
            for (const auto* ref : {&qmodel.layers[l].expert_gate[e],
                                    &qmodel.layers[l].expert_up[e],
                                    &qmodel.layers[l].expert_down[e]}) {
                if (!ref->quantized()) {
                    std::cout << "   engine did not quantize the expert\n";
                    return 1;
                }
                want.insert(want.end(), ref->bytes.begin(), ref->bytes.end());
            }

            ++compared;
            if (want.size() != loc.length) {
                std::cout << "   layer " << l << " expert " << e << ": container has "
                          << loc.length << " B, engine produced " << want.size() << " B\n";
                ++mismatches;
                break;
            }
            for (std::size_t i = 0; i < want.size(); ++i) {
                if (want[i] != buf[i]) {
                    first_bad_byte = i;
                    ++mismatches;
                    break;
                }
            }
            if (mismatches) {
                std::cout << "   layer " << l << " expert " << e
                          << ": first byte mismatch at offset " << first_bad_byte << "\n";
                break;
            }
        }
    }

    // ── the container's OUTPUT, which nothing had ever checked ───────────────
    //
    // Everything above compares BYTES. `streaming_g2` compares a streamed
    // forward against a resident one and finds them bit-identical — but it loads
    // the model from `tiny/` in BOTH cases and only attaches the container's
    // experts, so the container's dense half has never participated in a forward
    // that anyone checked (roadmap D19).
    //
    // Same tokens, same quantization, same expert bytes (proven byte-identical
    // just above). The ONLY difference is where the dense half came from, so any
    // divergence isolates the container's dense load.
    {
        soma::F32Workspace fws;
        std::vector<soma::TokenId> toks;
        for (std::uint32_t i = 0; i < 24; ++i) toks.push_back((i * 37 + 5) % qmodel.vocab());

        // A container-loaded model has no resident experts by construction —
        // that IS the container. Cache generously: this is an output comparison,
        // not a paging test, and streaming_g2 already covers eviction.
        soma::MemoryHierarchy mem;
        soma::MemoryBudget mb;
        mb.ram_expert_cache_bytes = 256ull * 1024 * 1024;
        if (auto st = mem.open(from_container.arch, store, mb); !st.ok()) {
            std::cout << "   hierarchy open failed: " << st.message() << "\n";
            return 1;
        }
        from_container.streamed_experts = &mem;

        std::vector<float> from_source_logits, from_container_logits;
        const auto a = soma::forward_f32(qmodel, toks, fws, from_source_logits);
        const auto b = soma::forward_f32(from_container, toks, fws, from_container_logits);
        if (!a.ok() || !b.ok()) {
            std::cout << "   forward failed: " << (a.ok() ? b.message() : a.message()) << "\n";
            return 1;
        }
        float worst = 0.0f;
        for (std::size_t i = 0;
             i < std::min(from_source_logits.size(), from_container_logits.size()); ++i) {
            worst = std::max(worst, std::fabs(from_source_logits[i] - from_container_logits[i]));
        }
        std::cout << "   container-loaded forward vs source-loaded: " << std::scientific
                  << std::setprecision(2) << worst << " max|diff|"
                  << (worst == 0.0f ? "  OK" : "  DIVERGES") << std::fixed << "\n";
        // Counted separately from `mismatches`, which the round-trip summary
        // below reports. Sharing the counter made the byte-comparison line print
        // "MISMATCH" for a forward divergence — labelling a passing check with
        // another check's failure.
        forward_diverged = (worst != 0.0f);
    }

    std::uint64_t bw = 0;
    const auto bw_st = store.measure_bandwidth(bw);

    std::cout << "   " << h.n_layers << "L x " << h.n_experts << "E, expert=" << h.expert_bytes
              << " B (gate/up q4_g + down q6_g), " << h.n_shards << " shard(s)\n"
              << "   byte-identical to engine quantization: " << compared << "/" << compared
              << (mismatches == 0 ? "  OK" : "  MISMATCH") << "\n";
    if (bw_st.ok()) {
        std::cout << "   random-read bandwidth at " << (h.expert_bytes / 1024)
                  << " KiB reads: " << std::fixed << std::setprecision(0)
                  << (static_cast<double>(bw) / 1e6) << " MB/s\n";
    }
    (void)d;
    (void)fi;
    return mismatches + (forward_diverged ? 1 : 0);
}

// ── 2. verdict function vs the documented table ──────────────────────────────

struct Expectation {
    const char*   name;
    std::uint32_t n_layers, first_dense, d_model, n_experts, top_k, expert_inter;
    std::uint32_t n_heads, n_kv, head_dim, vocab;
    soma::DType   dtype;
    std::uint64_t ram_gib;
    soma::Verdict expect;
    const char*   note;
};

soma::ArchIr make_arch(const Expectation& x) {
    soma::ArchIr a;
    a.source_repo = x.name;
    a.topology.n_layers = x.n_layers;
    a.topology.d_model = x.d_model;
    a.topology.vocab_size = x.vocab;
    a.topology.layer_kinds.assign(x.n_layers, soma::LayerKind::Moe);
    for (std::uint32_t i = 0; i < x.first_dense && i < x.n_layers; ++i) {
        a.topology.layer_kinds[i] = soma::LayerKind::Dense;
    }
    a.attention.family = (x.n_kv == x.n_heads) ? soma::AttentionFamily::Mha
                                               : soma::AttentionFamily::Gqa;
    a.attention.n_heads = x.n_heads;
    a.attention.n_kv_heads = x.n_kv;
    a.attention.head_dim = x.head_dim;
    a.router.n_experts = x.n_experts;
    a.router.top_k = x.top_k;
    a.ffn.expert_intermediate = x.expert_inter;
    a.ffn.dense_intermediate = x.expert_inter;

    const std::uint32_t group = (x.dtype == soma::DType::F32) ? 0 : 128;
    a.quantization.embed = {x.dtype, group};
    a.quantization.attn_proj = {x.dtype, group};
    a.quantization.expert_gate = {x.dtype, group};
    a.quantization.expert_up = {x.dtype, group};
    a.quantization.expert_down = {x.dtype, group};
    a.quantization.shared_expert = {x.dtype, group};
    return a;
}

int check_verdicts() {
    // Straight from the published configs; see schemas/arch-ir.md §8.
    // bf16 is modelled as f32 here — the ratio that drives the verdict is
    // routed-set-vs-cache, and both are 2x q4_g's expert size in the same
    // direction, so the CLASSIFICATION is unchanged even though the absolute
    // bytes are 2x high.
    const Expectation cases[] = {
        {"Qwen3-30B-A3B @q4_g/32GiB", 48, 0, 2048, 128, 8, 768, 32, 4, 128, 151936,
         soma::DType::Q4_G, 32, soma::Verdict::ResidentOnly,
         "14.5 GB routed set fits; streaming has nothing to do"},

        {"Qwen3-30B-A3B @q4_g/8GiB", 48, 0, 2048, 128, 8, 768, 32, 4, 128, 151936,
         soma::DType::Q4_G, 8, soma::Verdict::Stream,
         "constrained cache -> the streaming path it was built for"},

        {"DeepSeek-V2-Lite @q4_g/32GiB", 27, 1, 2048, 64, 6, 1408, 16, 16, 192, 102400,
         soma::DType::Q4_G, 32, soma::Verdict::ResidentOnly,
         "7.2 GB routed set fits -- expected, and why G4 needs backend_override"},

        {"Mixtral-8x7B @q4_g/32GiB", 32, 0, 4096, 8, 2, 14336, 32, 8, 128, 32000,
         soma::DType::Q4_G, 32, soma::Verdict::ResidentOnly,
         "fits at 22.6 GB -- resident-only via the fits branch"},

        {"Mixtral-8x7B @q4_g/8GiB", 32, 0, 4096, 8, 2, 14336, 32, 8, 128, 32000,
         soma::DType::Q4_G, 8, soma::Verdict::Reject,
         "25% active fraction: does not fit AND cannot stream"},

        {"Mixtral-8x7B @f32/32GiB", 32, 0, 4096, 8, 2, 14336, 32, 8, 128, 32000,
         soma::DType::F32, 32, soma::Verdict::Reject,
         "neither fits nor streams -> fallback with a smaller quantization"},
    };

    std::cout << std::left << std::setw(32) << "case" << std::setw(11) << "routed"
              << std::setw(11) << "cache" << std::setw(10) << "b/token" << std::setw(7) << "batch"
              << std::setw(15) << "verdict" << "expected\n"
              << std::string(104, '-') << "\n";

    int bad = 0;
    for (const auto& x : cases) {
        const auto arch = make_arch(x);
        soma::HostBudget b;
        b.ram_total_bytes = x.ram_gib * kGiB;
        b.ram_free_bytes = x.ram_gib * kGiB;
        b.ctx_size = 4096;
        b.kv_slots = 4;
        b.disk_bandwidth = 3ull * 1000 * 1000 * 1000;  // ~3 GB/s NVMe

        soma::PlanDocument plan;
        if (auto st = soma::compute_plan(arch, b, plan); !st.ok()) {
            std::cout << std::setw(32) << x.name << "ERROR " << st.message() << "\n";
            ++bad;
            continue;
        }
        const bool ok = (plan.verdict == x.expect);
        if (!ok) ++bad;

        auto gb = [](std::uint64_t v) {
            std::ostringstream s;
            s << std::fixed << std::setprecision(1) << (static_cast<double>(v) / 1e9) << "G";
            return s.str();
        };
        std::cout << std::left << std::setw(32) << x.name << std::setw(11)
                  << gb(plan.total_routed_bytes) << std::setw(11) << gb(plan.expert_cache_bytes)
                  << std::setw(10) << gb(plan.bytes_per_token) << std::setw(7) << plan.max_batch
                  << std::setw(15) << soma::to_string(plan.verdict)
                  << soma::to_string(x.expect) << (ok ? "" : "   <-- DISAGREES") << "\n";
        if (!ok) std::cout << "        reason: " << plan.verdict_reason << "\n";
    }
    return bad;
}

}  // namespace

/// DESCRIBABLE is not SERVABLE, and the plan has to say which.
///
/// Before GLM-5.2 those were the same question: a model_type with no adapter
/// failed in adapt_hf_config and never reached a plan, so `arch_supported`
/// defaulted to true and nothing ever set it — a field with a reader in
/// admission and no producer anywhere.
///
/// `glm_moe_dsa` separates them. Its expert half is ordinary and its economics
/// compute exactly like any other MoE; its attention is MLA with a sparse key
/// indexer, and `resolve_f32_backend` returns nullptr for MlaDsa because serving
/// it through the plain MLA backend would run it as DENSE attention — finite,
/// plausible, wrong.
///
/// Both directions are asserted. A check that only proved the refusal would pass
/// just as well if `arch_supported` were hardcoded false.
int check_plan_vs_serve() {
    int bad = 0;
    const auto check = [&](bool ok, const char* what, const std::string& detail = {}) {
        std::cout << "   " << std::left << std::setw(58) << what << (ok ? "OK" : "FAIL");
        if (!detail.empty()) std::cout << "   " << detail;
        std::cout << "\n";
        if (!ok) ++bad;
    };

    // Minimal, and deliberately NOT read from the checkpoint on disk: a test that
    // needs 1.4 TB present is a test that runs on one machine. These are the real
    // GLM-5.2 values, just without the weights.
    const std::string glm = R"({
        "model_type": "glm_moe_dsa", "num_hidden_layers": 78, "hidden_size": 6144,
        "vocab_size": 154880, "first_k_dense_replace": 3, "moe_layer_freq": 1,
        "n_routed_experts": 256, "num_experts_per_tok": 8, "n_shared_experts": 1,
        "moe_intermediate_size": 2048, "intermediate_size": 12288,
        "num_attention_heads": 64, "num_key_value_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 2048, "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64, "v_head_dim": 256,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": true,
        "routed_scaling_factor": 2.5, "rms_norm_eps": 1e-5, "hidden_act": "silu"
    })";

    soma::ArchIr arch;
    const auto st = soma::adapt_hf_config(glm, arch);
    check(st.ok(), "glm_moe_dsa ADAPTS rather than erroring", st.ok() ? "" : st.message());
    if (!st.ok()) return bad;

    check(arch.attention.family == soma::AttentionFamily::MlaDsa,
          "and is classified mla+dsa, not plain mla",
          soma::to_string(arch.attention.family));
    // The expert half is ordinary. If these drift, the economics below are
    // meaningless and the verdict would be wrong for a reason unrelated to DSA.
    check(arch.router.n_experts == 256 && arch.router.top_k == 8 &&
              arch.router.n_shared_experts == 1,
          "the router reads through unchanged (256 experts, top-8, 1 shared)");
    check(arch.n_moe_layers() == 75, "3 dense + 75 MoE layers",
          std::to_string(arch.n_moe_layers()) + " moe");
    check(arch.attention.mla.kv_lora_rank == 512 && arch.attention.mla.v_head_dim == 256,
          "MLA dims survive the DSA classification");

    soma::HostBudget b;
    b.ram_total_bytes = 24ull * kGiB; // Colibri's "comfortable" figure
    b.ram_free_bytes = 24ull * kGiB;
    b.ctx_size = 4096;
    b.kv_slots = 1;
    b.disk_bandwidth = 3ull * 1000 * 1000 * 1000;

    soma::PlanDocument plan;
    const auto pst = soma::compute_plan(arch, b, plan);
    check(pst.ok(), "and it PLANS", pst.ok() ? "" : pst.message());
    if (!pst.ok()) return bad;

    // The economics are the point of planning it at all, and they are computed
    // whether or not a backend exists. active_fraction is quantization- and
    // host-independent — 8/256 — which is what makes it comparable across the
    // table above.
    check(plan.active_fraction > 0.03 && plan.active_fraction < 0.032,
          "economics are computed: active fraction 8/256",
          std::to_string(plan.active_fraction));
    check(plan.total_routed_bytes > 0 && plan.bytes_per_token > 0,
          "routed set and bytes/token are real numbers");

    check(!plan.arch_supported, "arch_supported is FALSE — nothing can serve MlaDsa");
    check(plan.verdict == soma::Verdict::Reject,
          "so the verdict is reject regardless of economics",
          soma::to_string(plan.verdict));
    // The two rejects call for opposite responses — economics can change on a
    // bigger host, a missing backend cannot change on any host — so the reason
    // has to distinguish them.
    check(plan.verdict_reason.find("no backend") != std::string::npos,
          "and says WHY, so it is not read as an economic reject",
          plan.verdict_reason.substr(0, 46));

    // The control. Without it, hardcoding arch_supported=false would pass
    // everything above.
    const std::string olmoe = R"({
        "model_type": "olmoe", "num_hidden_layers": 16, "hidden_size": 2048,
        "vocab_size": 50304, "num_experts": 64, "num_experts_per_tok": 8,
        "intermediate_size": 1024, "num_attention_heads": 16,
        "num_key_value_heads": 16, "hidden_act": "silu"
    })";
    soma::ArchIr ok_arch;
    soma::PlanDocument ok_plan;
    if (soma::adapt_hf_config(olmoe, ok_arch).ok() &&
        soma::compute_plan(ok_arch, b, ok_plan).ok()) {
        check(ok_plan.arch_supported, "and a GQA model still reports arch_supported TRUE");
    } else {
        check(false, "and a GQA model still reports arch_supported TRUE", "control failed to plan");
    }
    return bad;
}

/// The verdict is a function of THREE arguments, and now all three can be varied.
///
/// This document says "the verdict is a property of (model, quantization, host)"
/// in a dozen places, and until `soma plan` grew `--quant` and `--ram` nothing
/// could demonstrate it: the CLI fixed two of the three, so the function could
/// only ever be evaluated at one point. One model reaching all three verdicts is
/// the claim made executable.
int check_verdict_varies_by_host_and_quant() {
    int bad = 0;
    const auto check = [&](bool ok, const char* what, const std::string& detail = {}) {
        std::cout << "   " << std::left << std::setw(58) << what << (ok ? "OK" : "FAIL");
        if (!detail.empty()) std::cout << "   " << detail;
        std::cout << "\n";
        if (!ok) ++bad;
    };

    // OLMoE-1B-7B's real shape, GQA so a backend exists and the verdict is not
    // short-circuited by arch_supported.
    const std::string cfg = R"({
        "model_type": "olmoe", "num_hidden_layers": 16, "hidden_size": 2048,
        "vocab_size": 50304, "num_experts": 64, "num_experts_per_tok": 8,
        "intermediate_size": 1024, "num_attention_heads": 16,
        "num_key_value_heads": 16, "hidden_act": "silu"
    })";

    const auto plan_at = [&](const char* overlay, std::uint64_t ram_gib, soma::PlanDocument& out) {
        soma::ArchIr a;
        if (!soma::adapt_hf_config(cfg, a).ok()) return false;
        if (overlay != nullptr && *overlay != '\0' &&
            !soma::apply_container_quant(overlay, a).ok()) {
            return false;
        }
        soma::HostBudget b;
        b.ram_total_bytes = ram_gib * kGiB;
        b.ram_free_bytes = ram_gib * kGiB;
        b.ctx_size = 4096;
        b.kv_slots = 1;
        b.disk_bandwidth = 3ull * 1000 * 1000 * 1000;
        return soma::compute_plan(a, b, out).ok();
    };

    // Same model, same quantization, only the HOST changes.
    soma::PlanDocument tight, mid, roomy;
    const bool ran = plan_at(R"({"dtype_gate_up":"q4_g","dtype_down":"q4_g","group":128})", 2, tight) &&
                     plan_at(R"({"dtype_gate_up":"q4_g","dtype_down":"q4_g","group":128})", 8, mid) &&
                     plan_at(R"({"dtype_gate_up":"q4_g","dtype_down":"q4_g","group":128})", 64, roomy);
    check(ran, "three hosts plan");
    if (!ran) return bad;

    check(tight.verdict == soma::Verdict::Stream,
          "a 2 GiB host STREAMS it (routed set exceeds the cache)",
          soma::to_string(tight.verdict));
    check(roomy.verdict == soma::Verdict::ResidentOnly,
          "a 64 GiB host says resident-only (streaming buys nothing)",
          soma::to_string(roomy.verdict));
    check(tight.verdict != roomy.verdict,
          "-> the HOST alone moves the verdict, model and quant fixed");

    // Same model, same host, only the QUANTIZATION changes. The routed set has
    // to grow: q8_0 is ~2x q4_g per weight, and that is the quantity the verdict
    // divides by.
    soma::PlanDocument q4, q8;
    const bool ran2 = plan_at(R"({"dtype_gate_up":"q4_g","dtype_down":"q4_g","group":128})", 8, q4) &&
                      plan_at(R"({"dtype_gate_up":"q8_0","dtype_down":"q8_0","group":128})", 8, q8);
    check(ran2, "two quantizations plan");
    if (!ran2) return bad;
    check(q8.total_routed_bytes > q4.total_routed_bytes,
          "-> and the QUANT alone moves the routed set",
          std::to_string(q4.total_routed_bytes >> 20) + " -> " +
              std::to_string(q8.total_routed_bytes >> 20) + " MiB");

    // An overlay must never be able to describe a container the converter cannot
    // produce: gate and up are interleaved into one range, so they share a dtype
    // whatever the caller asked for.
    soma::ArchIr split;
    (void)soma::adapt_hf_config(cfg, split);
    (void)soma::apply_container_quant(R"({"dtype_gate_up":"q8_0","dtype_down":"q4_g"})", split);
    check(split.quantization.expert_gate.dtype == split.quantization.expert_up.dtype,
          "gate and up always share a dtype, whatever was asked for");
    check(split.quantization.expert_down.dtype != split.quantization.expert_gate.dtype,
          "but down is independent, which is the whole point of expert_down");
    return bad;
}

/// `dense_resident_bytes` against the tensors that actually exist.
///
/// The plan's resident half was never checked against anything. It was derived
/// from a formula, the formula was written against GQA, and two errors lived in
/// it undetected:
///
///   * MLA was charged `q + 2*(n_kv_heads x head_dim) + o` — the GQA shape. MLA
///     has no per-head K or V projection at all, which is the entire point of
///     it. 1.66x over.
///   * Shared experts were multiplied by `n_shared_experts` when
///     `shared_intermediate` already carries that factor: n_shared^2. Invisible
///     at n_shared = 1 (GLM-5.2, Qwen) and 2x over at 2 (DeepSeek, Moonlight).
///
/// Both inflated the RESIDENT half, which is what decides whether a model can be
/// hosted at all — so the errors ran in the direction that makes models look
/// unservable, and MLA's whole reason for existing is a smaller resident half.
///
/// Checked against real bytes rather than a second formula. A formula compared
/// to a formula agrees with itself; these fixtures carry the actual tensors, and
/// summing them is the only independent statement available.
int check_dense_sizing(const fs::path& tiny_root) {
    int bad = 0;
    if (!fs::is_directory(tiny_root)) {
        std::cout << "   (no tiny fixtures)\n";
        return 0;
    }

    for (const auto& e : fs::directory_iterator(tiny_root)) {
        if (!e.is_directory()) continue;
        const auto name = e.path().filename().string();
        const auto weights = e.path() / "model.safetensors";
        if (!fs::is_regular_file(weights)) continue;

        soma::ArchIr arch;
        std::ifstream cfg(e.path() / "config.json", std::ios::binary);
        if (!cfg) continue;
        const std::string text((std::istreambuf_iterator<char>(cfg)),
                               std::istreambuf_iterator<char>());
        // An unadapted family is a coverage gap, not a sizing failure. granitemoe
        // has no adapter and must not read as a broken estimate.
        if (!soma::adapt_hf_config(text, arch).ok()) continue;

        soma::SafeTensors st;
        if (!st.open(weights.string()).ok()) continue;

        // Everything that is not a ROUTED expert is resident by definition —
        // the same split the container makes when it separates dense.safetensors
        // from the expert payload.
        std::uint64_t actual = 0;
        for (const auto& tn : st.names()) {
            if (tn.find(".experts.") != std::string::npos) continue;
            const auto* t = st.find(tn);
            if (t == nullptr) continue;
            std::uint64_t n = 1;
            for (std::size_t i = 0; i < t->rank(); ++i) n *= static_cast<std::uint64_t>(t->dim(i));
            actual += n * 4; // fixtures are f32
        }
        if (actual == 0) continue;

        soma::HostBudget b;
        b.ram_total_bytes = 8 * kGiB;
        b.ram_free_bytes = 8 * kGiB;
        soma::PlanDocument plan;
        if (!soma::compute_plan(arch, b, plan).ok()) continue;

        const double ratio =
            static_cast<double>(plan.dense_resident_bytes) / static_cast<double>(actual);
        // 3% covers the norms and the odd f32-vs-quantized rounding; it does not
        // cover a missing tensor family or a squared count, which is what this
        // is for.
        const bool ok = (ratio > 0.97 && ratio < 1.03);
        if (!ok) ++bad;
        std::ostringstream d;
        d << std::fixed << std::setprecision(2) << ratio << "x";
        std::cout << "   " << std::left << std::setw(34) << name << std::setw(9)
                  << soma::to_string(arch.attention.family) << (ok ? "OK" : "FAIL") << "   "
                  << d.str() << "\n";
    }
    return bad;
}

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures");
    int failures = 0;

    std::cout << "container round-trip\n";
    const fs::path cdir = root / "containers";
    if (fs::is_directory(cdir)) {
        for (const auto& e : fs::directory_iterator(cdir)) {
            if (!e.is_directory()) continue;
            const auto name = e.path().filename().string();
            std::cout << "== " << name << "\n";
            failures += check_container(root / "tiny" / name, e.path());
        }
    } else {
        std::cout << "   (no containers; run tools/admission/convert.py)\n";
    }

    std::cout << "\nverdict function vs schemas/arch-ir.md §8\n";
    failures += check_verdicts();

    std::cout << "\ndescribable is not servable (glm_moe_dsa)\n";
    failures += check_plan_vs_serve();

    std::cout << "\nthe verdict varies by host AND quantization\n";
    failures += check_verdict_varies_by_host_and_quant();

    std::cout << "\ndense_resident_bytes vs the tensors that exist\n";
    failures += check_dense_sizing(root / "tiny");

    std::cout << "\n" << (failures == 0 ? "OK" : std::to_string(failures) + " FAILURES") << "\n";
    return failures == 0 ? 0 : 1;
}
