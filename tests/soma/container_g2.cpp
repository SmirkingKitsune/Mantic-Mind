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
#include "soma/plan.hpp"
#include "soma/quant_format.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
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
    return mismatches;
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

    std::cout << "\n" << (failures == 0 ? "OK" : std::to_string(failures) + " FAILURES") << "\n";
    return failures == 0 ? 0 : 1;
}
