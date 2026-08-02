// Soma — G1 autotuner.
//
// Gate:
//   * every candidate for an (op, dtype) is numerically equivalent — checked
//     inside autotune_shapes before any timing is trusted
//   * shapes are derived from the model's IR, and m == 1 is always present
//   * the tuned table round-trips through kernel_choice rows and resolves back
//     to registered implementations
//
// The interesting output is not "it ran" but WHICH kernel won where. The prior
// art measured an int4 single-row path slower than fp32; if that reproduces on
// this hardware it is a finding to record, not a bug to fix.
//
// Usage: autotune_g1 <fixtures_dir> [fixture]

#include "soma/f32_model.hpp"
#include "soma/kernel_registry.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace {

soma::QuantMap mixed_map() {
    soma::QuantMap m;
    m.embed = {soma::DType::Q8_0, 32};
    m.attn_proj = {soma::DType::Q4_G, 128};
    m.expert_gate = {soma::DType::Q4_G, 128};
    m.expert_up = {soma::DType::Q4_G, 128};
    m.expert_down = {soma::DType::Q6_G, 128};
    m.shared_expert = {soma::DType::Q4_G, 128};
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures/tiny");
    const std::string want = (argc > 2) ? argv[2] : "Qwen3-30B-A3B";
    const fs::path dir = root / want;

    if (!fs::is_directory(dir)) {
        std::cerr << "fixture not found: " << dir.string() << "\n";
        return 2;
    }

    soma::F32Model model;
    if (auto st = soma::load_f32_model(dir.string(), model); !st.ok()) {
        std::cerr << "load failed: " << st.message() << "\n";
        return 2;
    }

    // Tiny-fixture dimensions are far too small to time meaningfully — a 64x64
    // matvec is dominated by call overhead. The SHAPES come from the fixture's
    // IR to prove extraction works; the tuning runs at production dimensions,
    // because a kernel choice made at d_model=64 tells you nothing about
    // d_model=2048.
    soma::ArchIr prod = model.arch;
    prod.topology.d_model = 2048;
    prod.topology.vocab_size = 151936;
    prod.attention.n_heads = 32;
    prod.attention.n_kv_heads = 4;
    prod.attention.head_dim = 128;
    prod.ffn.expert_intermediate = 768;
    prod.ffn.dense_intermediate = 0;
    prod.router.n_experts = 128;

    const std::uint32_t batches[] = {1, 32};
    const auto quant = mixed_map();

    const auto fixture_shapes = soma::model_shapes(model.arch, quant, batches);
    const auto shapes = soma::model_shapes(prod, quant, batches);

    std::cout << "shape extraction\n"
              << "  fixture IR  -> " << fixture_shapes.size() << " distinct shapes\n"
              << "  production  -> " << shapes.size() << " distinct shapes\n";

    const bool has_decode =
        std::any_of(shapes.begin(), shapes.end(), [](const auto& s) { return s.m == 1; });
    std::cout << "  m==1 present: " << (has_decode ? "yes" : "NO — GATE BROKEN") << "\n\n";
    int failures = has_decode ? 0 : 1;

    soma::AutotuneOptions opts;
    std::vector<soma::TuneResult> results;
    std::cout << "autotuning " << shapes.size() << " shapes (best-of-"
              << opts.measure_iters << ")...\n\n";

    if (auto st = soma::autotune_shapes(shapes, opts, results); !st.ok()) {
        // A disagreement between candidates is a correctness failure, not a
        // tuning failure, and it must not be reported as a slow kernel.
        std::cout << "AUTOTUNE FAILED: " << st.message() << "\n";
        return 1;
    }

    std::cout << std::left << std::setw(6) << "op" << std::setw(7) << "m"
              << std::setw(9) << "n" << std::setw(8) << "k" << std::setw(7) << "dtype"
              << std::setw(15) << "winner" << std::setw(10) << "GF/s" << "runners-up\n"
              << std::string(104, '-') << "\n";

    int dequant_wins = 0;
    for (const auto& r : results) {
        std::ostringstream others;
        for (const auto& [name, gf] : r.all) {
            if (name == r.impl) continue;
            if (!others.str().empty()) others << "  ";
            others << name << " " << std::fixed << std::setprecision(2) << gf;
        }
        if (r.impl.find("dequant") != std::string::npos) ++dequant_wins;

        std::cout << std::left << std::setw(6) << soma::to_string(r.shape.op)
                  << std::setw(7) << r.shape.m << std::setw(9) << r.shape.n
                  << std::setw(8) << r.shape.k << std::setw(7)
                  << soma::to_string(r.shape.dtype) << std::setw(15) << r.impl
                  << std::setw(10) << [&] {
                         std::ostringstream s;
                         s << std::fixed << std::setprecision(2) << r.gflops;
                         return s.str();
                     }()
                  << others.str() << "\n";
    }

    // ── round-trip through the resolved table ────────────────────────────────
    soma::ResolvedKernels table;
    if (auto st = soma::build_resolved(results, table); !st.ok()) {
        std::cout << "\nresolve failed: " << st.message() << "\n";
        return 1;
    }
    int unresolved = 0;
    for (const auto& r : results) {
        const auto* impl =
            table.resolve(r.shape.op, r.shape.dtype, r.shape.m, r.shape.n, r.shape.k);
        if (impl == nullptr || r.impl != impl->name) ++unresolved;
    }

    // An untuned shape must still resolve — to the default, not to nothing.
    const auto* fallback = table.resolve(soma::KernelOp::Gemv, soma::DType::Q4_G, 1, 7, 13);
    const bool fallback_ok = (fallback != nullptr);

    // EVERY extracted shape must resolve to something callable, including the
    // ones the tuner skipped for want of candidates. A nullptr here is a crash
    // at load, and the gap is invisible unless it is checked explicitly: the
    // tuner simply reports fewer results than shapes.
    int unresolvable = 0;
    for (const auto& s : shapes) {
        if (table.resolve(s.op, s.dtype, s.m, s.n, s.k) == nullptr) {
            std::cout << "  UNRESOLVABLE: " << soma::to_string(s.op) << " m=" << s.m
                      << " n=" << s.n << " k=" << s.k << " "
                      << soma::to_string(s.dtype) << "\n";
            ++unresolvable;
        }
    }
    std::cout << "every extracted shape resolves: "
              << (unresolvable == 0 ? "yes" : std::to_string(unresolvable) + " DO NOT") << "\n";
    failures += unresolvable;

    std::cout << "\nresolved table: " << table.size() << " entries, "
              << (unresolved == 0 ? "all round-trip" : std::to_string(unresolved) + " MISMATCH")
              << "; untuned shape falls back: " << (fallback_ok ? "yes" : "NO") << "\n";
    failures += unresolved + (fallback_ok ? 0 : 1);

    const auto sql = soma::to_registry_rows(results, 1);
    const auto rows = static_cast<int>(std::count(sql.begin(), sql.end(), '\n'));
    std::cout << "kernel_choice rows: " << rows << "\n";
    if (rows != static_cast<int>(results.size())) ++failures;

    std::cout << "\ndequantize-first won " << dequant_wins << " of " << results.size()
              << " shapes\n";

    std::cout << (failures == 0 ? "\nOK\n" : "\n" + std::to_string(failures) + " FAILURES\n");
    return failures == 0 ? 0 : 1;
}
