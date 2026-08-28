// Soma — G3: the STREAMED forward is deterministic, including under prefetch.
//
// This test exists because its absence cost real time. threading_g3 proves the
// forward is bit-identical across thread counts, but it runs a RESIDENT model —
// no ExpertStore, no cache, no loader threads. The entire streaming path was
// therefore unguarded, and a data race in it (concurrent `seekg` + `read` on one
// shared ifstream, which silently returns another expert's bytes) surfaced as an
// odd column in a performance table rather than as a failing test.
//
// The lesson generalises: a determinism gate that skips the concurrent subsystem
// is not a determinism gate. Anything that can hand the engine wrong bytes has to
// be inside the thing being asserted.
//
// Runs against the tiny fixture and its committed container, so it belongs in
// CI. Each configuration runs in a child process, because both the thread pool
// and the SIMD tier latch their settings at first use.
//
// Usage: streamed_determinism_g3 <fixtures_root> [fixture]
//        streamed_determinism_g3 <fixtures_root> <fixture> --emit

#include "soma/expert_store.hpp"
#include "soma/f32_model.hpp"
#include "soma/memory_hierarchy.hpp"

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

#if !defined(_WIN32)
std::string posix_shell_quote(const std::string& value) {
    std::string out = "'";
    for (const char c : value) {
        if (c == '\'')
            out += "'\\''";
        else
            out += c;
    }
    out += '\'';
    return out;
}
#endif

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
    soma::ExpertStore store;
    if (auto st = store.open((root / "containers" / name).string(), model.arch); !st.ok()) {
        std::cerr << "container open failed: " << st.message() << "\n";
        return 2;
    }

    // A cache far smaller than the routed set, so experts are genuinely evicted
    // and re-read. A run whose cache holds everything exercises no concurrency in
    // the loader at all and would pass while proving nothing.
    soma::MemoryHierarchy mem;
    soma::MemoryBudget b;
    b.ram_expert_cache_bytes = store.header().expert_bytes * 4;
    if (auto st = mem.open(model.arch, store, b); !st.ok()) {
        std::cerr << "hierarchy open failed: " << st.message() << "\n";
        return 2;
    }
    model.streamed_experts = &mem;

    std::vector<soma::TokenId> toks(48);
    for (std::uint32_t i = 0; i < toks.size(); ++i) toks[i] = (i * 37 + 5) % model.vocab();

    soma::F32Workspace ws;
    std::vector<float> logits;
    if (auto st = soma::forward_f32(model, toks, ws, logits); !st.ok()) {
        std::cerr << "forward failed: " << st.message() << "\n";
        return 2;
    }
    // The unique-expert count is reported alongside the logit digest because it
    // is a pure function of routing and therefore an independent witness: if it
    // drifts, the routing drifted, and the logits are wrong for a reason that has
    // nothing to do with float reassociation.
    std::cout << digest(logits) << " " << ws.unique_expert_reads << "\n";
    return 0;
}

std::string run_child(const std::string& exe, const fs::path& root, const std::string& name,
                      const char* threads, const char* prefetch) {
#if defined(_WIN32)
    _putenv_s("SOMA_THREADS", threads);
    _putenv_s("SOMA_PREFETCH_DEPTH", prefetch);
#else
    setenv("SOMA_THREADS", threads, 1);
    setenv("SOMA_PREFETCH_DEPTH", prefetch, 1);
#endif
    std::ostringstream cmd;
#if defined(_WIN32)
    cmd << "\"\"" << exe << "\" \"" << root.string() << "\" " << name << " --emit\"";
#else
    cmd << posix_shell_quote(exe) << ' ' << posix_shell_quote(root.string()) << ' '
        << posix_shell_quote(name) << " --emit";
#endif

    std::string out;
#if defined(_WIN32)
    FILE* p = _popen(cmd.str().c_str(), "r");
#else
    FILE* p = popen(cmd.str().c_str(), "r");
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
    const std::string name =
        (argc > 2 && std::string(argv[2]) != "--emit") ? argv[2] : std::string("Qwen3-30B-A3B");
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--emit") return emit(root, name);
    }

    struct Cfg {
        const char* threads;
        const char* prefetch;
        const char* label;
    };
    // Repeated configurations on purpose. A race is not reproducible by
    // definition, so one run of each setting can pass by luck; the same setting
    // twice is the cheapest way to make luck visible.
    const Cfg cfgs[] = {
        {"1", "0", "1 thread,  no prefetch"},
        {"1", "3", "1 thread,  prefetch 3"},
        {"8", "0", "8 threads, no prefetch"},
        {"8", "3", "8 threads, prefetch 3"},
        {"8", "3", "8 threads, prefetch 3 (repeat)"},
        {"0", "3", "all cores, prefetch 3"},
        {"0", "3", "all cores, prefetch 3 (repeat)"},
        {"0", "8", "all cores, prefetch 8"},
    };

    std::cout << std::left << std::setw(34) << "configuration" << std::setw(24) << "logit digest"
              << "unique experts\n";

    std::string first_digest, first_unique;
    bool ok = true;
    for (const auto& c : cfgs) {
        const auto r = run_child(argv[0], root, name, c.threads, c.prefetch);
        if (r.empty()) {
            std::cerr << "child failed: " << c.label << "\n";
            return 2;
        }
        std::istringstream in(r);
        std::string dg, uniq;
        in >> dg >> uniq;
        if (first_digest.empty()) {
            first_digest = dg;
            first_unique = uniq;
        }
        const bool match = (dg == first_digest) && (uniq == first_unique);
        ok &= match;
        std::cout << std::left << std::setw(34) << c.label << std::setw(24) << dg << uniq
                  << (match ? "" : "   <-- DIFFERS") << "\n";
    }

    std::cout << "\n";
    if (!ok) {
        std::cout << "FAIL: the streamed forward is not deterministic.\n\n"
                     "  Identical input produced different logits. Because the expert weights\n"
                     "  arrive through the cache and the loader threads, the first suspects are\n"
                     "  the paths that hand out bytes:\n\n"
                     "    * a shared file position — concurrent positional reads must carry the\n"
                     "      offset in the CALL (pread / ReadFile+OVERLAPPED), never in the handle\n"
                     "    * a slot whose bytes are replaced or freed while a borrow is live\n"
                     "    * an eviction that ignores refs or the in-flight `loading` flag\n\n"
                     "  If `unique experts` also differs, routing itself changed, which means some\n"
                     "  expert's WEIGHTS were wrong — not merely summed in a different order.\n";
        return 1;
    }
    std::cout << "OK: identical across thread counts and prefetch depths.\n";
    return 0;
}
