// Soma — G3: does concurrency actually pay in the disk-bound regime?
//
// The design claims aggregate throughput scales SUPER-LINEARLY in batch size
// when reads dominate: doubling the rows less than doubles the bytes, because
// the union reads each unique expert once regardless of how many rows want it.
//
// Two confounds make the naive version of this measurement lie, and both are
// handled explicitly rather than hoped away:
//
//   1. THE OS PAGE CACHE. After one pass the container is resident in free RAM
//      and every subsequent "disk" read is a memcpy. Wall time then measures
//      the page cache, not the device. So the PRIMARY gate here is BYTES per
//      token — an architectural quantity the page cache cannot touch — and wall
//      time is reported as secondary, labelled, and never gated on.
//
//   2. BUILD TYPE. In a Debug build the fp32 GEMMs are slow enough to swamp any
//      I/O effect, so wall-clock scaling would measure compute. The build type
//      is printed with the results so a Debug number is never mistaken for a
//      performance claim.
//
// Usage: scaling_g3 <container_dir> [--tokens-per-seq N] [--cache-gib N]

#include "soma/expert_store.hpp"
#include "soma/f32_model.hpp"
#include "soma/memory_hierarchy.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Row {
    std::uint32_t batch = 0;
    std::uint64_t bytes = 0;
    std::uint64_t unique = 0;
    std::uint64_t naive = 0;
    double        secs = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: scaling_g3 <container_dir> [--tokens-per-seq N] "
                     "[--cache-gib N]\n";
        return 2;
    }
    const std::string cdir = argv[1];
    std::uint32_t per_seq = 8;
    std::uint64_t cache_gib = 2;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string f = argv[i];
        if (f == "--tokens-per-seq") per_seq = std::stoul(argv[i + 1]);
        if (f == "--cache-gib") cache_gib = std::stoull(argv[i + 1]);
    }

    soma::QuantMap qm;
    qm.expert_gate = {soma::DType::Q4_G, 128};
    qm.expert_up = {soma::DType::Q4_G, 128};
    qm.expert_down = {soma::DType::Q6_G, 128};

    soma::F32Model model;
    if (auto st = soma::load_f32_model(cdir, model, qm); !st.ok()) {
        std::cerr << "load failed: " << st.message() << "\n";
        return 2;
    }
    soma::ExpertStore store;
    if (auto st = store.open(cdir, model.arch); !st.ok()) {
        std::cerr << "container open failed: " << st.message() << "\n";
        return 2;
    }

#ifdef NDEBUG
    const char* build = "Release";
#else
    const char* build = "Debug  (wall-clock below is NOT a performance claim)";
#endif

    std::cout << "container    " << cdir << "\n"
              << "build        " << build << "\n"
              << "cache        " << cache_gib << " GiB\n"
              << "seq length   " << per_seq << " tokens\n\n";

    // Each "sequence" contributes `per_seq` rows to one union forward. That is
    // exactly what the step-major scheduler will do with N concurrent sequences,
    // so batching rows here measures the same thing without needing the
    // scheduler to exist yet.
    std::vector<Row> rows;
    for (const std::uint32_t nseq : {1u, 2u, 4u, 8u}) {
        soma::MemoryHierarchy mem;
        soma::MemoryBudget b;
        b.ram_expert_cache_bytes = cache_gib * 1024ull * 1024 * 1024;
        b.pin_bytes = b.ram_expert_cache_bytes / 8;
        if (auto st = mem.open(model.arch, store, b); !st.ok()) {
            std::cerr << "hierarchy open failed: " << st.message() << "\n";
            return 2;
        }
        model.streamed_experts = &mem;

        // Distinct token ids per sequence, so the sequences route to genuinely
        // different experts. Reusing one prompt N times would make the union
        // look perfect for the wrong reason — every row would select the same
        // experts, which is the best case and not the expected one.
        const auto n_rows = nseq * per_seq;
        std::vector<soma::TokenId> toks(n_rows);
        for (std::uint32_t s = 0; s < nseq; ++s) {
            for (std::uint32_t t = 0; t < per_seq; ++t) {
                toks[s * per_seq + t] =
                    (s * 7919u + t * 131u + 11u) % model.vocab();
            }
        }

        soma::F32Workspace ws;
        std::vector<float> logits;
        const auto t0 = std::chrono::steady_clock::now();
        if (auto st = soma::forward_f32(model, toks, ws, logits); !st.ok()) {
            std::cerr << "forward failed at nseq=" << nseq << ": " << st.message() << "\n";
            return 1;
        }
        const auto secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        rows.push_back({nseq, mem.stats().bytes_read, ws.unique_expert_reads,
                        ws.naive_expert_reads, secs});
        std::cout << "  nseq=" << nseq << " done (" << std::fixed << std::setprecision(1)
                  << secs << "s)\n" << std::flush;
    }

    std::cout << "\n"
              << std::left << std::setw(7) << "nseq" << std::setw(8) << "rows"
              << std::setw(13) << "MiB read" << std::setw(12) << "KiB/token"
              << std::setw(11) << "unique" << std::setw(9) << "union"
              << std::setw(10) << "sec" << "tok/s\n";

    const auto& base = rows.front();
    const double base_per_tok = static_cast<double>(base.bytes) / (base.batch * per_seq);

    for (const auto& r : rows) {
        const auto n_rows = r.batch * per_seq;
        std::cout << std::left << std::setw(7) << r.batch << std::setw(8) << n_rows
                  << std::setw(13) << (r.bytes / 1048576)
                  << std::setw(12) << std::fixed << std::setprecision(1)
                  << (static_cast<double>(r.bytes) / n_rows / 1024.0)
                  << std::setw(11) << r.unique
                  << std::setw(9) << std::setprecision(1)
                  << (static_cast<double>(r.naive) /
                      static_cast<double>(std::max<std::uint64_t>(1, r.unique)))
                  << std::setw(10) << std::setprecision(1) << r.secs
                  << std::setprecision(2) << (n_rows / r.secs) << "\n";
    }

    // ── the gate ─────────────────────────────────────────────────────────────
    const auto& top = rows.back();
    const double top_per_tok =
        static_cast<double>(top.bytes) / (top.batch * per_seq);
    const double io_speedup = base_per_tok / top_per_tok;

    std::cout << "\nbytes/token fell " << std::setprecision(2) << io_speedup
              << "x from nseq=1 to nseq=" << top.batch << "\n";

    // ── the three-part gate ──────────────────────────────────────────────────
    //
    // In the disk-bound regime time is proportional to bytes, so
    //
    //   TP_N / TP_1  =  N * R_1/R_N  =  b_1 / b_N
    //
    // where b is bytes per token. The aggregate-throughput ratio EQUALS the
    // bytes/token reduction, which in turn equals the union ratio
    // (rows*top_k / unique). They are the same number measured two ways.
    //
    // This gate used to ask for SUPER-LINEAR scaling — more than Nx throughput
    // from N sequences. The identity above says that is impossible: a larger
    // batch reads MORE bytes in total (R_N > R_1), just fewer per token, so
    // N*R_1/R_N < N always. No correct implementation could ever have passed it.
    //
    // What replaces it is three separable questions, so a failure says WHICH
    // thing broke.
    constexpr double kDeviceMbPerSec = 1230.0;  // measured at G2, local NVMe
    const double implied_mb_s =
        static_cast<double>(top.bytes) / 1048576.0 / top.secs;
    const double bandwidth_share = implied_mb_s / kDeviceMbPerSec;

    const double tps_1 = static_cast<double>(base.batch * per_seq) / base.secs;
    const double tps_n = static_cast<double>(top.batch * per_seq) / top.secs;
    const double tp_ratio = tps_n / tps_1;

    // How much of the theoretically available speedup actually became speed.
    // This is the part the old wording had no room for, and the part that
    // catches a real regression: the mechanism can work and the regime can be
    // right while unoverlapped I/O keeps saved bytes from becoming saved time.
    const double conversion = tp_ratio / io_speedup;

    std::cout << "aggregate tok/s  " << std::setprecision(2) << tps_1 << " -> " << tps_n
              << "  (" << tp_ratio << "x for " << (top.batch / base.batch)
              << "x the rows)\n"
              << "read bandwidth   " << std::setprecision(0) << implied_mb_s
              << " MB/s of " << kDeviceMbPerSec << " MB/s device ("
              << (bandwidth_share * 100.0) << "% — saturates, >90% means disk-bound)\n\n";

    const bool bytes_win = io_speedup > 1.05;
    const bool read_bound = bandwidth_share > 0.5;
    const bool converts = conversion >= 0.5;

    std::cout << std::setprecision(2)
              << "mechanism   bytes/token falls          " << io_speedup << "x   "
              << (bytes_win ? "PASS" : "FAIL") << "\n"
              << "regime      reads dominate runtime     "
              << std::setprecision(0) << (bandwidth_share * 100.0) << "%   "
              << (read_bound ? "PASS" : "NO — compute-bound") << "\n"
              << "conversion  realised / available       " << std::setprecision(2)
              << tp_ratio << "x of " << io_speedup << "x = "
              << std::setprecision(0) << (conversion * 100.0) << "%   "
              << (converts ? "PASS" : "FAIL") << "\n\n"
              << "G3 throughput gate: "
              << (bytes_win && read_bound && converts
                      ? "PASS"
                      : (bytes_win && !read_bound ? "NOT EVALUABLE on this run" : "FAIL"))
              << "\n";

    if (bytes_win && read_bound && !converts) {
        std::cout << "\nThe union is reducing bytes and reads dominate, but the saving is not\n"
                     "becoming time. That points at OVERLAP, not at the kernels or the union:\n"
                     "reads issued serially inside the expert loop stall the compute that could\n"
                     "have run alongside them. Check that prefetch is enabled and that its depth\n"
                     "is not being clamped to zero by a small cap_per_layer.\n";
    }

    if (!bytes_win) {
        std::cout << "\nbytes/token did not fall. Either the union is not "
                     "deduplicating (check the\nunion column above — near 1.0 "
                     "means it is not), or the cache is large enough that\nevery "
                     "expert was already resident, in which case this is not the "
                     "disk-bound\nregime either. Re-run with a smaller --cache-gib.\n";
    } else if (!read_bound) {
        std::cout << "\nThe union is working — bytes/token fell as designed — but this run "
                     "spent most\nof its time in the kernels, not waiting on reads, so the "
                     "saved bytes could not\nturn into saved time. Super-linear throughput "
                     "is a claim about the disk-bound\nregime and this is not it.\n\n"
                     "  Where the time goes is a question for `soma_autotune_g1`, not for this\n"
                     "  message. Two earlier versions of this text named the specific culprit —\n"
                     "  first the quantized kernels, then the fp32 ones — and both were obsolete\n"
                     "  within a day of being written, because fixing the named bottleneck is\n"
                     "  exactly what this tool provokes. It now reports the REGIME and leaves the\n"
                     "  attribution to the tool that measures it fresh.\n";
    }

    // Exit status tracks the MECHANISM and, when the regime allows it to be
    // judged, the CONVERSION. The regime itself is deliberately excluded: it is
    // a property of the host's disk-to-compute ratio, not of the code, so failing
    // on it would make the verdict depend on which machine ran the tool.
    return (bytes_win && (!read_bound || converts)) ? 0 : 1;
}
