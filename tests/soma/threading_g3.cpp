// Soma — G3: the forward is bit-identical regardless of thread count.
//
// This is the load-bearing claim of the whole threading design, and the only one
// that cannot be checked by looking at the code. Every parallel region partitions
// OUTPUT elements, so each output is produced by one worker in the order a single
// thread would have used. If that holds, results are byte-for-byte independent of
// core count and scheduling.
//
// If it does NOT hold, the failure is quiet and expensive: logits drift by ~1e-7,
// every conformance number becomes a property of the machine that produced it,
// `determinism: strict` is unreachable, and nothing crashes. So it is asserted
// against a real forward rather than argued from the source.
//
// The test re-executes itself with SOMA_THREADS set, because the pool reads it
// once at first use — a single process cannot hold two pool sizes.
//
// Usage: threading_g3 <fixtures_root>            (driver)
//        threading_g3 <fixtures_root> --emit     (child: prints a digest)

#include "soma/f32_model.hpp"
#include "soma/threading.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

/// FNV-1a over the raw logit bytes. A hash rather than a tolerance, because the
/// claim is bit-identity: a comparison with any epsilon at all would pass on
/// precisely the drift this exists to detect.
std::uint64_t digest(const std::vector<float>& v) {
    std::uint64_t h = 1469598103934665603ull;
    const auto* p = reinterpret_cast<const unsigned char*>(v.data());
    for (std::size_t i = 0; i < v.size() * sizeof(float); ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

int emit(const fs::path& root, const std::string& name) {
    soma::QuantMap cmap;
    cmap.expert_gate = {soma::DType::Q4_G, 128};
    cmap.expert_up = {soma::DType::Q4_G, 128};
    cmap.expert_down = {soma::DType::Q6_G, 128};

    soma::F32Model model;
    if (auto st = soma::load_f32_model((root / "tiny" / name).string(), model, cmap); !st.ok()) {
        std::cerr << "load failed: " << st.message() << "\n";
        return 2;
    }

    // Enough positions that the ragged attention chunking actually splits.
    std::vector<soma::TokenId> toks(48);
    for (std::uint32_t i = 0; i < toks.size(); ++i) toks[i] = (i * 37 + 5) % model.vocab();

    soma::F32Workspace ws;
    std::vector<float> logits;
    if (auto st = soma::forward_f32(model, toks, ws, logits); !st.ok()) {
        std::cerr << "forward failed: " << st.message() << "\n";
        return 2;
    }
    std::cout << soma::ThreadPool::global().size() << " " << digest(logits) << "\n";
    return 0;
}

/// Re-run this executable with SOMA_THREADS=`threads`, return "<size> <digest>".
std::string run_child(const std::string& exe, const fs::path& root,
                      const std::string& name, const std::string& threads) {
#if defined(_WIN32)
    _putenv_s("SOMA_THREADS", threads.c_str());
#else
    setenv("SOMA_THREADS", threads.c_str(), 1);
#endif
    std::ostringstream cmd;
    cmd << "\"\"" << exe << "\" \"" << root.string() << "\" " << name << " --emit\"";

    std::string out;
    FILE* p =
#if defined(_WIN32)
        _popen(cmd.str().c_str(), "r");
#else
        popen(cmd.str().c_str(), "r");
#endif
    if (!p) return {};
    char buf[256];
    while (std::fgets(buf, sizeof(buf), p)) out += buf;
#if defined(_WIN32)
    _pclose(p);
#else
    pclose(p);
#endif
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures");
    const std::string name = (argc > 2 && std::string(argv[2]) != "--emit")
                                 ? argv[2]
                                 : std::string("Qwen3-30B-A3B");
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--emit") return emit(root, name);
    }

    std::cout << "host threads available: " << soma::ThreadPool::global().size() << "\n\n";

    struct Row {
        const char* threads;
        std::string result;
    };
    Row rows[] = {{"1", {}}, {"2", {}}, {"3", {}}, {"8", {}}, {"0", {}}};  // 0 = default

    for (auto& r : rows) {
        r.result = run_child(argv[0], root, name,
                             std::string(r.threads) == "0" ? "" : r.threads);
        if (r.result.empty()) {
            std::cerr << "child failed at SOMA_THREADS=" << r.threads << "\n";
            return 2;
        }
    }

    std::cout << std::left << std::setw(16) << "SOMA_THREADS" << std::setw(10) << "workers"
              << "logit digest\n";
    for (const auto& r : rows) {
        std::istringstream in(r.result);
        std::string workers, dg;
        in >> workers >> dg;
        std::cout << std::left << std::setw(16)
                  << (std::string(r.threads) == "0" ? "(default)" : r.threads)
                  << std::setw(10) << workers << dg << "\n";
    }

    // Every digest must match the serial one exactly.
    //
    // Thread counts that do NOT divide the work evenly are the interesting cases:
    // 3 leaves a remainder on almost every loop, and the attention chunking is
    // ragged by construction. A design that only reproduces at powers of two is
    // not deterministic, it is lucky.
    std::string serial;
    {
        std::istringstream in(rows[0].result);
        std::string w;
        in >> w >> serial;
    }
    bool all_match = true;
    std::string worker_counts;
    for (const auto& r : rows) {
        std::istringstream in(r.result);
        std::string w, dg;
        in >> w >> dg;
        worker_counts += w + " ";
        all_match &= (dg == serial);
    }

    std::cout << "\n";
    if (!all_match) {
        std::cout << "FAIL: the forward is NOT bit-identical across thread counts.\n\n"
                     "  Some parallel region is splitting a REDUCTION rather than partitioning\n"
                     "  outputs, so the summation order now depends on how the work was divided.\n"
                     "  Look for a parallel_for whose body accumulates into a shared location,\n"
                     "  or for scratch that two workers write concurrently.\n";
        return 1;
    }

    // A run where every child had one worker proves nothing, and would otherwise
    // report a confident pass.
    if (worker_counts.find('2') == std::string::npos &&
        worker_counts.find('3') == std::string::npos &&
        worker_counts.find('8') == std::string::npos) {
        std::cout << "INCONCLUSIVE: no child ran with more than one worker, so identity\n"
                     "across thread counts was never actually exercised.\n";
        return 2;
    }

    std::cout << "OK: bit-identical across " << worker_counts << "workers.\n";
    return 0;
}
