// Soma — open a real container and report what the verdict depends on.
//
// The bandwidth number here is a G2 gate item and cannot be obtained from the
// tiny fixtures: their experts are 4 KB, where reads are IOPS-bound, while a
// production expert is ~3 MB. Those do not achieve the same bandwidth on the
// same drive, and using the small number would make every verdict pessimistic.
//
// Usage: probe_container <container_dir> <source_config.json>

#include "soma/expert_store.hpp"
#include "soma/plan.hpp"
#include "soma/quant_format.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: probe_container <container_dir> <source_config.json>\n";
        return 2;
    }
    const fs::path cdir(argv[1]);
    const fs::path cfg_path(argv[2]);

    std::ifstream in(cfg_path);
    if (!in) {
        std::cerr << "cannot read " << cfg_path.string() << "\n";
        return 2;
    }
    std::ostringstream ss;
    ss << in.rdbuf();

    soma::ArchIr arch;
    if (auto st = soma::adapt_hf_config(ss.str(), arch); !st.ok()) {
        std::cerr << "adapt failed: " << st.message() << "\n";
        return 2;
    }

    // The IR must describe the container's precision, or the expert-size
    // cross-check refuses the open — which is the point of that check.
    arch.quantization.expert_gate = {soma::DType::Q4_G, 128};
    arch.quantization.expert_up = {soma::DType::Q4_G, 128};
    arch.quantization.expert_down = {soma::DType::Q6_G, 128};

    soma::ExpertStore store;
    if (auto st = store.open(cdir.string(), arch); !st.ok()) {
        std::cerr << "open failed: " << st.message() << "\n";
        return 1;
    }
    const auto& h = store.header();

    std::cout << "container  " << cdir.filename().string() << "\n"
              << "  layers/experts   " << h.n_layers << " x " << h.n_experts << " = "
              << (h.n_layers * h.n_experts) << " experts\n"
              << "  expert bytes     " << h.expert_bytes << " ("
              << std::fixed << std::setprecision(2)
              << (static_cast<double>(h.expert_bytes) / 1e6) << " MB)\n"
              << "  shards           " << h.n_shards << "\n"
              << "  size cross-check PASSED (IR quant map matches on-disk bytes)\n";

    // A single read, verified against the index length.
    std::vector<std::byte> buf(h.expert_bytes);
    if (store.read(0, 0, buf) != soma::StatusCode::Ok) {
        std::cerr << "  read of (0,0) FAILED\n";
        return 1;
    }
    const auto loc = store.locate(0, 0);
    std::cout << "  first expert     shard " << loc.shard << " offset " << loc.offset
              << " len " << loc.length
              << (loc.offset % soma::kDirectIoAlign == 0 ? "  (aligned)" : "  UNALIGNED") << "\n";

    std::uint64_t bw = 0;
    if (auto st = store.measure_bandwidth(bw); st.ok()) {
        std::cout << "  random-read BW   " << std::setprecision(0)
                  << (static_cast<double>(bw) / 1e6) << " MB/s at "
                  << std::setprecision(2) << (static_cast<double>(h.expert_bytes) / 1e6)
                  << " MB reads\n";
    } else {
        std::cout << "  bandwidth probe  " << st.message() << "\n";
        return 1;
    }

    // The plan against a few host budgets, using the measured bandwidth rather
    // than a spec-sheet figure.
    std::cout << "\n" << std::left << std::setw(10) << "RAM" << std::setw(12) << "routed"
              << std::setw(12) << "cache" << std::setw(12) << "b/token" << std::setw(8) << "batch"
              << std::setw(9) << "tok/s" << "verdict\n";
    for (const std::uint64_t gib : {8ull, 16ull, 32ull, 64ull}) {
        soma::HostBudget b;
        b.ram_total_bytes = gib * 1024ull * 1024 * 1024;
        b.ram_free_bytes = b.ram_total_bytes;
        b.ctx_size = 4096;
        b.kv_slots = 4;
        b.disk_bandwidth = bw;

        soma::PlanDocument plan;
        if (auto st = soma::compute_plan(arch, b, plan); !st.ok()) {
            std::cout << gib << " GiB: " << st.message() << "\n";
            continue;
        }
        auto g = [](std::uint64_t v) {
            std::ostringstream o;
            o << std::fixed << std::setprecision(1) << (static_cast<double>(v) / 1e9) << "G";
            return o.str();
        };
        std::cout << std::left << std::setw(10) << (std::to_string(gib) + " GiB")
                  << std::setw(12) << g(plan.total_routed_bytes)
                  << std::setw(12) << g(plan.expert_cache_bytes)
                  << std::setw(12) << g(plan.bytes_per_token)
                  << std::setw(8) << plan.max_batch
                  << std::setw(9) << std::setprecision(3) << plan.projected_tok_s
                  << soma::to_string(plan.verdict) << "\n";
    }
    return 0;
}
