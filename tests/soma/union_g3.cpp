// Soma — G3: the batch-union, and the telemetry text dump.
//
// The union is the central claim of the design: read each unique expert ONCE per
// step and apply it to every row that selected it, so read cost is per-expert
// rather than per-row. This measures whether that is actually happening.
//
// The ratio is the whole point. A ratio near 1.0 means the union is buying
// nothing and something upstream is wrong — that check is worth more than any
// throughput number, because throughput can look fine for unrelated reasons.
//
// The G3 amendment is also here: a MINIMAL TEXT DUMP of tier/heat telemetry must
// exist at G3, four gates before the FTXUI panels. Watching expert-load patterns
// across a batch is the primary instrument for catching cache thrash, so the
// debugging view has to precede the pretty one.
//
// Usage: union_g3 <fixtures_root> [fixture]

#include "soma/expert_store.hpp"
#include "soma/f32_model.hpp"
#include "soma/memory_hierarchy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << "   " << std::left << std::setw(50) << what << (ok ? "OK" : "FAIL");
    if (!detail.empty()) std::cout << "   " << detail;
    std::cout << "\n";
    if (!ok) ++g_failures;
}

/// Minimal tier/heat dump — the G3 amendment.
///
/// Deliberately plain text. Its job is to be readable while debugging cache
/// thrash, not to be pretty; the FTXUI panels arrive at G7 and consume the same
/// numbers through the API.
std::string telemetry_dump(const soma::MemoryHierarchy& mem, std::uint32_t max_layers) {
    const auto occ = mem.occupancy();
    const auto st = mem.stats();
    const auto heat = mem.heat();

    std::ostringstream o;
    const auto total = st.hits + st.misses;
    o << "tier   vram  " << std::setw(6) << occ.vram_experts << " experts  "
      << std::setw(9) << occ.vram_bytes << " B\n"
      << "tier   ram   " << std::setw(6) << occ.ram_experts << " experts  "
      << std::setw(9) << (occ.ram_bytes / 1024) << " KiB / "
      << (occ.ram_capacity_bytes / 1024) << " KiB   (pinned "
      << occ.pinned_experts << ")\n"
      << "tier   disk  " << std::setw(6) << occ.disk_experts << " experts\n"
      << "cache  hit " << std::fixed << std::setprecision(1)
      << (total ? 100.0 * static_cast<double>(st.hits) / static_cast<double>(total) : 0.0)
      << "%  miss " << st.misses << "  evict " << st.evictions
      << "  prefetch " << st.prefetch_hits << " hit / " << st.prefetch_wasted << " wasted\n";

    // Per-layer heat: one cell per expert, shaded by decayed heat. Coarse on
    // purpose — the eye is looking for a layer whose whole row is hot (thrash)
    // or cold (dead), not for individual values.
    static const char* kShades = " .:-=+*#%@";
    float peak = 0.0f;
    for (const auto& c : heat.cells) peak = std::max(peak, c.decayed);

    std::vector<std::vector<float>> grid(heat.n_layers,
                                         std::vector<float>(heat.n_experts, 0.0f));
    for (const auto& c : heat.cells) {
        if (c.layer < heat.n_layers && c.expert < heat.n_experts) {
            grid[c.layer][c.expert] = c.decayed;
        }
    }
    for (std::uint32_t l = 0; l < std::min(heat.n_layers, max_layers); ++l) {
        o << "layer " << std::setw(3) << l << "  ";
        std::uint32_t hot = 0;
        for (std::uint32_t e = 0; e < heat.n_experts; ++e) {
            const float v = grid[l][e];
            if (v > 0.0f) ++hot;
            const int idx = (peak > 0.0f)
                                ? static_cast<int>(9.0f * std::sqrt(v / peak))
                                : 0;
            o << kShades[std::clamp(idx, 0, 9)];
        }
        o << "  " << hot << "/" << heat.n_experts << " touched\n";
    }
    return o.str();
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures");
    const std::string name = (argc > 2) ? argv[2] : "Qwen3-30B-A3B";

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
    const auto& h = store.header();
    const auto top_k = model.arch.router.top_k;
    const auto n_exp = model.arch.router.n_experts;
    const auto n_moe = model.arch.n_moe_layers();

    std::cout << "== " << name << "  " << n_moe << " MoE layers x " << n_exp
              << " experts, top_k=" << top_k << "\n\n";

    // ── 1. CSR correctness ───────────────────────────────────────────────────
    //
    // Checked directly rather than inferred from the forward's output: a union
    // that dropped or duplicated a selection could still produce plausible
    // logits, and the failure would surface much later as an accuracy question.
    std::cout << "1. union CSR structure\n";
    {
        constexpr std::uint32_t rows = 37;
        std::vector<std::uint32_t> ids(static_cast<std::size_t>(rows) * top_k);
        std::vector<float> wts(ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            ids[i] = static_cast<std::uint32_t>((i * 7 + 3) % n_exp);
            wts[i] = 1.0f / static_cast<float>(i + 1);
        }

        soma::F32Workspace ws;
        ws.reserve(model.arch, rows);
        soma::build_expert_union(rows, top_k, n_exp, ids.data(), wts.data(), ws);

        const auto n_unique = ws.union_experts.size();
        check(ws.union_offsets.size() == n_unique + 1, "offsets has n_unique+1 entries");
        check(ws.union_offsets.back() == ids.size(), "offsets cover every selection",
              std::to_string(ws.union_offsets.back()) + "/" + std::to_string(ids.size()));

        bool ascending = true;
        for (std::size_t i = 1; i < n_unique; ++i) {
            ascending &= (ws.union_experts[i] > ws.union_experts[i - 1]);
        }
        check(ascending, "experts emitted in ascending id order (deterministic)");

        // Every (row, expert, weight) triple appears exactly once.
        std::vector<int> seen(ids.size(), 0);
        bool consistent = true;
        for (std::size_t u = 0; u < n_unique; ++u) {
            const auto e = ws.union_experts[u];
            for (auto i = ws.union_offsets[u]; i < ws.union_offsets[u + 1]; ++i) {
                const auto row = ws.union_rows[i];
                bool matched = false;
                for (std::uint32_t s = 0; s < top_k; ++s) {
                    const auto k = static_cast<std::size_t>(row) * top_k + s;
                    if (ids[k] == e && !seen[k] && wts[k] == ws.union_weights[i]) {
                        seen[k] = 1;
                        matched = true;
                        break;
                    }
                }
                consistent &= matched;
            }
        }
        const auto unmatched = std::count(seen.begin(), seen.end(), 0);
        check(consistent && unmatched == 0, "every selection appears exactly once",
              std::to_string(ids.size() - unmatched) + "/" + std::to_string(ids.size()));
    }

    // ── 2. the ratio, across batch sizes ─────────────────────────────────────
    std::cout << "\n2. unique vs naive expert reads\n";
    std::cout << "   " << std::left << std::setw(9) << "rows" << std::setw(11) << "naive"
              << std::setw(11) << "unique" << std::setw(10) << "ratio"
              << std::setw(12) << "misses" << "reads/row\n";

    double prev_ratio = 0.0;
    bool monotone = true;
    for (const std::uint32_t rows : {1u, 4u, 16u, 64u}) {
        soma::MemoryHierarchy mem;
        soma::MemoryBudget b;
        b.ram_expert_cache_bytes = h.expert_bytes * (top_k + 2);
        (void)mem.open(model.arch, store, b);

        soma::F32Model m;
        (void)soma::load_f32_model((root / "tiny" / name).string(), m, cmap);
        m.streamed_experts = &mem;

        std::vector<soma::TokenId> toks(rows);
        for (std::uint32_t i = 0; i < rows; ++i) toks[i] = (i * 41 + 7) % m.vocab();

        soma::F32Workspace ws;
        std::vector<float> logits;
        if (auto st = soma::forward_f32(m, toks, ws, logits); !st.ok()) {
            check(false, "forward at rows=" + std::to_string(rows), st.message());
            continue;
        }
        const double ratio = static_cast<double>(ws.naive_expert_reads) /
                             static_cast<double>(std::max<std::uint64_t>(1, ws.unique_expert_reads));
        if (rows > 1 && ratio < prev_ratio) monotone = false;
        prev_ratio = ratio;

        std::cout << "   " << std::left << std::setw(9) << rows
                  << std::setw(11) << ws.naive_expert_reads
                  << std::setw(11) << ws.unique_expert_reads
                  << std::setw(10) << std::fixed << std::setprecision(2) << ratio
                  << std::setw(12) << mem.stats().misses
                  << std::setprecision(2)
                  << (static_cast<double>(ws.unique_expert_reads) / rows / n_moe) << "\n";
    }
    std::cout << "\n";
    check(prev_ratio > 2.0, "union saves materially at batch 64",
          std::to_string(static_cast<int>(prev_ratio)) + "x fewer reads");
    check(monotone, "ratio is non-decreasing in batch size",
          "more rows -> more sharing");

    // Read the miss column with the ratio, not on its own.
    //
    // A working union drives the CACHE HIT RATE TOWARD ZERO on a single pass,
    // and that is the union succeeding. Before the union, the cache absorbed the
    // duplicate (row, expert) lookups and reported them as hits; now the CSR
    // removes them before the cache is ever consulted, so what remains is
    // cold-first-touch traffic and nothing else. Misses == unique reads exactly.
    //
    // Worth stating because the number moves the wrong way: hit rate dropping
    // from ~97% to ~0% while bytes read falls 20x looks like a regression to
    // anyone reading the hit rate alone. Bytes/token is the metric that matters;
    // hit rate is only meaningful across REPEATED steps, where it measures
    // inter-step reuse rather than intra-step duplication.

    // At one row the union CANNOT help: top_k distinct experts, one row each.
    // A ratio above 1.0 there would mean the router emitted duplicates.
    {
        soma::MemoryHierarchy mem;
        soma::MemoryBudget b;
        b.ram_expert_cache_bytes = h.expert_bytes * (top_k + 2);
        (void)mem.open(model.arch, store, b);
        soma::F32Model m;
        (void)soma::load_f32_model((root / "tiny" / name).string(), m, cmap);
        m.streamed_experts = &mem;
        std::vector<soma::TokenId> one{7};
        soma::F32Workspace ws;
        std::vector<float> logits;
        (void)soma::forward_f32(m, one, ws, logits);
        check(ws.naive_expert_reads == ws.unique_expert_reads,
              "at 1 row the union is exactly a no-op",
              std::to_string(ws.unique_expert_reads) + " reads");
    }

    // ── 3. does the gate's prediction match reality? ─────────────────────────
    //
    // The cache-aware gate sizes max_batch from a coupon-collector expectation:
    //   E * (1 - (1 - k/E)^rows)
    // which assumes experts are chosen UNIFORMLY AT RANDOM. Real routers are not
    // uniform — they concentrate on a hot subset — so the formula should
    // OVERESTIMATE the unique count, making the gate conservative.
    //
    // Conservative is the safe direction and it is still worth measuring, because
    // the unsafe direction is silent: a formula that underestimates would size
    // max_batch above what the cache can hold and thrash exactly where the gate
    // was supposed to prevent it.
    std::cout << "\n3. gate prediction vs measurement\n";
    std::cout << "   " << std::left << std::setw(9) << "rows" << std::setw(13) << "predicted"
              << std::setw(13) << "measured" << "pred/meas\n";

    bool conservative = true;
    for (const std::uint32_t rows : {4u, 16u, 64u}) {
        soma::MemoryHierarchy mem;
        soma::MemoryBudget b;
        b.ram_expert_cache_bytes = h.expert_bytes * (top_k + 2);
        (void)mem.open(model.arch, store, b);
        soma::F32Model m;
        (void)soma::load_f32_model((root / "tiny" / name).string(), m, cmap);
        m.streamed_experts = &mem;

        std::vector<soma::TokenId> toks(rows);
        for (std::uint32_t i = 0; i < rows; ++i) toks[i] = (i * 41 + 7) % m.vocab();
        soma::F32Workspace ws;
        std::vector<float> logits;
        (void)soma::forward_f32(m, toks, ws, logits);

        const double p = static_cast<double>(top_k) / static_cast<double>(n_exp);
        const double per_layer =
            static_cast<double>(n_exp) * (1.0 - std::pow(1.0 - p, static_cast<double>(rows)));
        const double predicted = per_layer * n_moe;
        const double measured = static_cast<double>(ws.unique_expert_reads);
        const double ratio = predicted / std::max(1.0, measured);
        if (ratio < 0.95) conservative = false;

        std::cout << "   " << std::left << std::setw(9) << rows
                  << std::setw(13) << std::fixed << std::setprecision(1) << predicted
                  << std::setw(13) << measured
                  << std::setprecision(2) << ratio << "\n";
    }
    std::cout << "\n";
    check(conservative, "prediction never UNDERestimates unique experts",
          "under-estimating would size max_batch into thrash");

    // ── 4. telemetry text dump (the G3 amendment) ────────────────────────────
    std::cout << "\n4. telemetry text dump\n";
    {
        soma::MemoryHierarchy mem;
        soma::MemoryBudget b;
        b.ram_expert_cache_bytes = h.expert_bytes * (top_k + 2);
        (void)mem.open(model.arch, store, b);

        soma::F32Model m;
        (void)soma::load_f32_model((root / "tiny" / name).string(), m, cmap);
        m.streamed_experts = &mem;

        std::vector<soma::TokenId> toks(64);
        for (std::uint32_t i = 0; i < 64; ++i) toks[i] = (i * 41 + 7) % m.vocab();
        soma::F32Workspace ws;
        std::vector<float> logits;
        (void)soma::forward_f32(m, toks, ws, logits);

        const auto dump = telemetry_dump(mem, 8);
        check(!dump.empty(), "dump produced");
        check(dump.find("tier   ram") != std::string::npos, "reports RAM tier occupancy");
        check(dump.find("tier   vram") != std::string::npos,
              "reports the VRAM tier (declared, empty in v1)");
        check(dump.find("cache  hit") != std::string::npos, "reports cache hit rate");
        check(dump.find("layer   0") != std::string::npos, "reports per-layer heat");
        std::cout << "\n" << dump;
    }

    std::cout << "\n" << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES")
              << "\n";
    return g_failures == 0 ? 0 : 1;
}
