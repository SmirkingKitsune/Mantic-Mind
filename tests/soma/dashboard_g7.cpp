// Mantic-Mind — G7: the brain grid's layout.
//
// The gate says the grid renders a 48x128 model bucketed to <=4096 cells at 2 Hz
// without visible cost. Cost is measured here; CORRECTNESS is the harder half,
// because a grid that is subtly wrong looks exactly like a model that routes
// oddly — and an operator staring at a heat map has no way to tell the two
// apart. So the reduction is checked as arithmetic:
//
//   * counts are CONSERVED — a bucket is the sum of its cells, and a grid whose
//     total drifts is reporting traffic that did not happen
//   * the tier split weights by TRAFFIC, not membership — an expert on disk
//     that never fires costs nothing, and a grid that says otherwise saturates
//   * every cell is covered exactly once, including a ragged last row/column
//   * "fired once" and "never fired" render differently
//
// layout_heat() is deliberately pure — no FTXUI, no HTTP, no clock — so all of
// that is checkable without a terminal.
//
// Usage: dashboard_g7

#include "control/soma_dashboard.hpp"
#include "control/soma_panels.hpp"

#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << "   " << std::left << std::setw(58) << what << (ok ? "OK" : "FAIL");
    if (!detail.empty()) std::cout << "   " << detail;
    std::cout << "\n";
    if (!ok) ++g_failures;
}

/// A frame with a known, non-uniform distribution.
///
/// Deliberately not uniform: a reduction that drops or double-counts cells still
/// conserves the total when every cell holds the same number, so a flat fixture
/// would pass a broken implementation.
mm::SomaHeatView make_frame(std::uint32_t rows, std::uint32_t cols) {
    mm::SomaHeatView h;
    h.rows = rows;
    h.cols = cols;
    h.n_layers = rows;
    h.n_experts = cols;
    h.resolution = "full";
    h.counts.resize(static_cast<std::size_t>(rows) * cols);
    h.tiers.resize(h.counts.size(), 1); // Ram
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < cols; ++c) {
            // Distinct per cell, so a mis-indexed read shows up as a wrong total
            // rather than an equal one.
            h.counts[static_cast<std::size_t>(r) * cols + c] = 1 + r * 7 + c * 3;
        }
    }
    return h;
}

std::uint64_t sum(const std::vector<std::uint64_t>& v) {
    return std::accumulate(v.begin(), v.end(), std::uint64_t{0});
}

} // namespace

/// Draw the whole tab against a synthetic cluster and print it.
///
/// `dashboard_g7 --preview`. Assertions prove the panel says the right words;
/// they say nothing about whether the columns line up or the grid is legible,
/// and a TUI nobody can look at without building a cluster does not get looked at.
int preview() {
    mm::SomaSnapshot snap;
    mm::SomaEngineView a;
    a.id = "eng-7a07d6e9";
    a.backend = "soma";
    a.state = "Ready";
    a.model_path = "/srv/containers/Qwen3-30B-A3B";
    a.node_id = "node-alpha";
    a.vram_usage_mb = 0;
    a.effective_ctx_size = 8192;
    a.agent_ids = {"agent-1", "agent-2"};
    mm::SomaEngineView b;
    b.id = "eng-fea80c20";
    b.backend = "llama-cpp";
    b.state = "Ready";
    b.model_path = "/srv/models/mixtral-8x7b-q4.gguf";
    b.node_id = "node-beta";
    b.vram_usage_mb = 22400;
    b.effective_ctx_size = 4096;
    b.agent_ids = {"agent-3"};
    snap.engines = {a, b};
    snap.tiers.engines = 2;
    snap.tiers.vram_mb = 22400;
    snap.tiers.node_disk_free_mb = 1843200;
    snap.engines_at_ms = 100000;
    snap.heat_at_ms = 99000;
    snap.heat_engine_id = a.id;

    // A plausible routing distribution: a few hot experts per layer rather than
    // uniform traffic, which is what a real router produces and what the grid
    // has to make legible.
    snap.heat = make_frame(48, 128);
    snap.heat.layer_bucket = 1;
    snap.heat.expert_bucket = 1;
    for (std::uint32_t r = 0; r < 48; ++r) {
        for (std::uint32_t c = 0; c < 128; ++c) {
            const auto i = static_cast<std::size_t>(r) * 128 + c;
            const bool hot = ((c * 7 + r * 13) % 17) < 2;
            snap.heat.counts[i] = hot ? 400 + (r * 31 + c * 11) % 600 : (c % 5 == 0 ? 3 : 0);
            snap.heat.tiers[i] = hot ? 1 : 2; // hot in RAM, cold on disk
        }
    }

    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(112), ftxui::Dimension::Fixed(34));
    auto page = ftxui::vbox({
        mm::render_engine_list(snap, 0),
        ftxui::hbox({
            mm::render_brain_grid(snap, 16, 96) | ftxui::flex,
            mm::render_tier_bar(snap) | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, 46),
        }),
        ftxui::filler(),
        mm::render_status_line(snap, 130000),
    });
    ftxui::Render(screen, page);
    std::cout << screen.ToString() << "\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--preview") return preview();
    // ── 1. no reduction needed ───────────────────────────────────────────────
    std::cout << "\n1. a frame that already fits\n";
    {
        const auto h = make_frame(4, 16);
        const auto g = mm::layout_heat(h, 32, 128);
        check(g.rows == 4 && g.cols == 16,
              "passes through at its own size",
              std::to_string(g.rows) + "x" + std::to_string(g.cols));
        check(!g.viewport_reduced, "and is not marked as reduced");
        check(g.total == sum(h.counts),
              "with every count intact",
              std::to_string(g.total) + " == " + std::to_string(sum(h.counts)));
        bool all_present = true;
        for (const auto& c : g.cells)
            if (!c.present) all_present = false;
        check(all_present, "and every cell present");
    }

    // ── 2. the gate's own case: 48x128 ───────────────────────────────────────
    //
    // 6144 cells, above the 4096 cap, so it must reduce — and the reduction has
    // to conserve, because the number an operator reads off this grid is the
    // model's real routing traffic.
    std::cout << "\n2. 48x128 into a terminal-sized viewport\n";
    {
        const auto h = make_frame(48, 128);
        const auto g = mm::layout_heat(h, 24, 64);
        check(g.rows == 24 && g.cols == 64,
              "reduced to the viewport",
              std::to_string(g.rows) + "x" + std::to_string(g.cols));
        check(g.viewport_reduced, "and says so");
        check(g.row_stride == 2 && g.col_stride == 2,
              "with the strides it used",
              std::to_string(g.row_stride) + "x" + std::to_string(g.col_stride));
        check(g.total == sum(h.counts),
              "counts are CONSERVED",
              std::to_string(g.total) + " == " + std::to_string(sum(h.counts)));

        // Every source cell landed in exactly one bucket. Checked by summing the
        // cells rather than trusting `total`, which is accumulated by the same
        // loop and would agree with itself through a double-count.
        std::uint64_t from_cells = 0;
        for (const auto& c : g.cells)
            from_cells += c.count;
        check(from_cells == sum(h.counts),
              "and summing the CELLS agrees",
              std::to_string(from_cells));
    }

    // ── 3. a ragged reduction ────────────────────────────────────────────────
    //
    // 48 rows into 7 does not divide. Flooring the stride would leave a
    // remainder with nowhere to go, and the rows that vanish are the LAST ones —
    // nearest the output, where a routing anomaly is most worth seeing.
    std::cout << "\n3. dimensions that do not divide\n";
    {
        const auto h = make_frame(48, 100);
        const auto g = mm::layout_heat(h, 7, 9);
        check(g.total == sum(h.counts),
              "counts are still conserved",
              std::to_string(g.total) + " == " + std::to_string(sum(h.counts)));
        check(g.rows * g.row_stride >= 48 && g.cols * g.col_stride >= 100,
              "and the grid covers the whole frame",
              std::to_string(g.rows) + "x" + std::to_string(g.row_stride) + ", " +
                  std::to_string(g.cols) + "x" + std::to_string(g.col_stride));
        std::size_t missing = 0;
        for (const auto& c : g.cells)
            if (!c.present) ++missing;
        check(
            missing == 0, "with no empty cell past the edge", std::to_string(missing) + " absent");
    }

    // ── 4. tier reduction weights by TRAFFIC ─────────────────────────────────
    //
    // Defect D2. The reduction used to take the COLDEST TIER PRESENT in a bucket,
    // which ignores counts: a bucket of six experts where one sits on disk and
    // never fires reported Disk, identically to a bucket whose disk-resident
    // expert is the hot one. On a streamed model — where most experts are on disk
    // by construction — the whole grid went one colour the moment any reduction
    // happened, and got MORE uniform the more it reduced. The channel carried
    // nothing exactly where it was needed.
    //
    // An expert on disk that never fires costs nothing. One that fires constantly
    // costs everything. These four cases are the difference.
    std::cout << "\n4. tier reduction weights by traffic\n";
    {
        // Every count equal, so only the tiers vary and the fractions are exact.
        const auto flat = [](std::uint32_t rows, std::uint32_t cols, std::uint64_t each) {
            mm::SomaHeatView h;
            h.rows = rows;
            h.cols = cols;
            h.n_layers = rows;
            h.n_experts = cols;
            h.counts.assign(static_cast<std::size_t>(rows) * cols, each);
            h.tiers.assign(h.counts.size(), 1);
            return h;
        };

        // (a) all resident.
        {
            auto h = flat(2, 4, 10);
            h.tiers.assign(8, 0);
            const auto g = mm::layout_heat(h, 1, 2);
            check(g.cells[0].cold_fraction() == 0.0 && g.cells[1].cold_fraction() == 0.0,
                  "an all-resident frame is 0% cold everywhere");
        }

        // (b) THE REGRESSION. One disk expert in a bucket of four, and it never
        //     fires. Under coldest-wins this bucket was Disk; it costs nothing.
        {
            auto h = flat(2, 4, 10);
            h.tiers = {0, 0, 0, 0, 2, 0, 0, 0}; // the disk cell is (1,0)
            h.counts[4] = 0;                    // ...and it never fired
            const auto g = mm::layout_heat(h, 1, 2);
            check(g.cells[0].cold_fraction() == 0.0,
                  "a disk expert that never fires does not colour its bucket",
                  "cold=" + std::to_string(g.cells[0].cold_fraction()));
        }

        // (c) The same bucket, but the disk expert is the one doing the work.
        {
            auto h = flat(2, 4, 1);
            h.tiers = {0, 0, 0, 0, 2, 0, 0, 0};
            h.counts[4] = 97; // 97 of 100 routes in bucket 0 come from disk
            const auto g = mm::layout_heat(h, 1, 2);
            check(g.cells[0].cold_fraction() > 0.9,
                  "a disk expert carrying the traffic DOES",
                  "cold=" + std::to_string(g.cells[0].cold_fraction()));
        }

        // (d) The saturation property itself, stated as a test rather than as a
        //     hope: a realistically streamed model — most experts on disk, the
        //     hot ones resident — must not reduce to a uniform grid.
        {
            auto h = flat(16, 16, 0);
            for (std::uint32_t r = 0; r < 16; ++r) {
                for (std::uint32_t c = 0; c < 16; ++c) {
                    const auto i = static_cast<std::size_t>(r) * 16 + c;
                    const bool resident = (c % 8 == 0); // 1 in 8 experts cached
                    h.tiers[i] = resident ? 1 : 2;
                    h.counts[i] = resident ? 100 : 1; // and they carry the traffic
                }
            }
            const auto g = mm::layout_heat(h, 4, 4);
            double lo = 1.0, hi = 0.0;
            for (const auto& cell : g.cells) {
                lo = std::min(lo, cell.cold_fraction());
                hi = std::max(hi, cell.cold_fraction());
            }
            check(hi - lo > 0.25,
                  "a streamed model still varies across the grid",
                  "cold_fraction spans " + std::to_string(lo) + " to " + std::to_string(hi));
            // Under the old rule every one of these buckets contained a disk
            // expert, so every one reported Disk and this span was exactly 0.
            check(lo < 0.5,
                  "and the cached regions read as cached",
                  "coldest bucket " + std::to_string(lo));
        }

        // Traffic is conserved across the tier split, or the fractions are
        // fractions of the wrong denominator.
        {
            auto h = flat(4, 4, 7);
            for (std::size_t i = 0; i < h.tiers.size(); ++i)
                h.tiers[i] = static_cast<std::uint8_t>(i % 3);
            const auto g = mm::layout_heat(h, 2, 2);
            bool conserved = true;
            for (const auto& cell : g.cells) {
                if (cell.tier_count[0] + cell.tier_count[1] + cell.tier_count[2] != cell.count) {
                    conserved = false;
                }
            }
            check(conserved, "the per-tier split sums back to the cell's count");
        }
    }

    // ── 5. cold is not the same as never ─────────────────────────────────────
    std::cout << "\n5. intensity\n";
    {
        mm::SomaHeatView h;
        h.rows = 1;
        h.cols = 4;
        h.n_layers = 1;
        h.n_experts = 4;
        h.counts = {0, 1, 500, 1000};
        h.tiers = {1, 1, 1, 1};
        const auto g = mm::layout_heat(h, 8, 8);
        check(g.cells[0].intensity == 0,
              "a cell that never fired is 0",
              std::to_string(g.cells[0].intensity));
        check(g.cells[1].intensity >= 1,
              "a cell that fired ONCE is not 0",
              std::to_string(g.cells[1].intensity));
        check(g.cells[3].intensity > g.cells[1].intensity,
              "and the hottest outranks it",
              std::to_string(g.cells[3].intensity) + " > " + std::to_string(g.cells[1].intensity));
        check(g.hottest == 1000, "with the frame's peak recorded", std::to_string(g.hottest));

        // Relative, not absolute: the same shape at 1000x the traffic must render
        // identically, or the grid is blank early in a run and saturated later.
        mm::SomaHeatView big = h;
        for (auto& c : big.counts)
            c *= 1000;
        const auto gb = mm::layout_heat(big, 8, 8);
        bool same = true;
        for (std::size_t i = 0; i < g.cells.size(); ++i) {
            if (g.cells[i].intensity != gb.cells[i].intensity) same = false;
        }
        check(same, "and 1000x the traffic renders the same shape");
    }

    // ── 6. degenerate input ──────────────────────────────────────────────────
    //
    // A fallback engine has no grid at all. The panel has to get an empty layout
    // rather than a crash or a 1x1 grid that looks like a model with one expert.
    std::cout << "\n6. nothing to draw\n";
    {
        check(mm::layout_heat({}, 24, 64).cells.empty(), "an empty frame lays out to nothing");
        check(mm::layout_heat(make_frame(4, 4), 0, 64).cells.empty(), "a zero-height viewport too");
        mm::SomaHeatView lying;
        lying.rows = 4;
        lying.cols = 4;        // claims 16 cells...
        lying.counts = {1, 2}; // ...and carries 2
        const auto g = mm::layout_heat(lying, 8, 8);
        check(g.cells.empty() || g.total <= 3,
              "and a frame whose arrays disagree does not read past them",
              std::to_string(g.cells.size()) + " cells");
    }

    // ── 7. the cost claim ────────────────────────────────────────────────────
    //
    // "at 2 Hz without visible cost" is a measurement, not an assertion. 48x128
    // is the gate's model; the budget is one frame's layout well under the 500 ms
    // between ticks.
    std::cout << "\n7. cost at the gate's size\n";
    {
        const auto h = make_frame(48, 128);
        constexpr int kFrames = 200;
        const auto t0 = std::chrono::steady_clock::now();
        std::uint64_t sink = 0;
        for (int i = 0; i < kFrames; ++i) {
            const auto g = mm::layout_heat(h, 24, 64);
            sink += g.total; // so the loop cannot be optimized away
        }
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        const double per_frame_ms = static_cast<double>(us) / kFrames / 1000.0;
        std::cout << "   " << kFrames << " layouts of 48x128 -> 24x64: " << std::fixed
                  << std::setprecision(3) << per_frame_ms << " ms/frame\n";
        check(sink > 0, "the loop actually ran");
        // Two orders of magnitude of headroom against the 500 ms tick. A bound
        // this loose still catches the failure that matters — an accidentally
        // quadratic reduction — while not going red on a loaded CI box.
        check(per_frame_ms < 5.0,
              "well inside a 2 Hz tick",
              std::to_string(per_frame_ms) + " ms vs 500 ms");
    }

    // ── 8. the panels, rendered ──────────────────────────────────────────────
    //
    // Drawn to an off-screen buffer and asserted as TEXT. The states worth
    // checking are the ones nobody opens the app in — a fallback engine, an
    // engine that has not routed a token yet, a selection pointing at an engine
    // that just disappeared — and "a human looked at it" checks none of them.
    std::cout << "\n8. the panels\n";
    {
        const auto draw = [](ftxui::Element e, int w, int h) {
            auto screen =
                ftxui::Screen::Create(ftxui::Dimension::Fixed(w), ftxui::Dimension::Fixed(h));
            ftxui::Render(screen, e);
            return screen.ToString();
        };
        const auto has = [](const std::string& hay, const std::string& needle) {
            return hay.find(needle) != std::string::npos;
        };

        mm::SomaSnapshot snap;
        snap.engines_at_ms = 0;
        check(has(draw(mm::render_engine_list(snap, -1), 100, 6), "reachable"),
              "an empty list before the first poll says so");
        snap.engines_at_ms = 1;
        check(has(draw(mm::render_engine_list(snap, -1), 100, 6), "no engines"),
              "and after one, says the cluster is empty");

        mm::SomaEngineView soma_engine;
        soma_engine.id = "eng-soma-1";
        soma_engine.backend = "soma";
        soma_engine.state = "Ready";
        soma_engine.model_path = "/models/Qwen3-30B-A3B";
        soma_engine.node_id = "node-a";
        mm::SomaEngineView fallback;
        fallback.id = "eng-llama-1";
        fallback.backend = "llama-cpp";
        fallback.state = "Ready";
        fallback.model_path = "/models/mixtral.gguf";
        fallback.node_id = "node-b";
        snap.engines = {soma_engine, fallback};

        const auto listing = draw(mm::render_engine_list(snap, 0), 110, 6);
        check(has(listing, "eng-soma-1") && has(listing, "eng-llama-1"), "both engines are listed");
        check(has(listing, "llama-cpp"), "the fallback is a first-class row, not an omission");

        // ── graceful degradation ─────────────────────────────────────────────
        snap.heat_engine_id = "eng-llama-1";
        check(has(draw(mm::render_brain_grid(snap, 8, 32), 70, 8), "fallback"),
              "a fallback engine explains why it has no grid");

        snap.heat_engine_id = "eng-soma-1";
        check(has(draw(mm::render_brain_grid(snap, 8, 32), 70, 8), "no heat frame yet"),
              "a Soma engine with no frame says so rather than drawing an empty one");

        snap.heat_engine_id = "eng-vanished";
        check(has(draw(mm::render_brain_grid(snap, 8, 32), 70, 8), "gone"),
              "a selection pointing at a departed engine says so");

        // ── a real frame ─────────────────────────────────────────────────────
        snap.heat_engine_id = "eng-soma-1";
        snap.heat = make_frame(48, 128);
        snap.heat.layer_bucket = 2;
        snap.heat.expert_bucket = 2;
        snap.heat.tiers.assign(snap.heat.counts.size(), 1);
        snap.heat.tiers[0] = 2; // one disk cell
        const auto grid = draw(mm::render_brain_grid(snap, 12, 40), 90, 20);
        check(has(grid, "48x128 experts"), "the caption carries the MODEL's shape, not the grid's");
        check(has(grid, "engine bucket 2x2") && has(grid, "view "),
              "and reports the engine's bucketing separately from the viewport's");
        check(has(grid, "from disk") && has(grid, "none") && has(grid, "all"),
              "with a legend for the disk-share bands");

        // ── the tier bar ─────────────────────────────────────────────────────
        //
        // The gate line: VRAM present and empty, not hidden. A reader who cannot
        // see the tier assumes the design has two.
        snap.tiers.vram_mb = 0;
        snap.tiers.node_disk_free_mb = 500000;
        const auto tiers = draw(mm::render_tier_bar(snap), 70, 8);
        check(has(tiers, "VRAM"), "the VRAM tier is present at zero");
        check(has(tiers, "CPU-only"), "and says why it is empty rather than looking broken");
        check(has(tiers, "DISK"), "alongside the tiers that do have numbers");

        // ── staleness ────────────────────────────────────────────────────────
        //
        // Per field, because engines and heat are separate requests and one can
        // fail while the other succeeds.
        snap.engines_at_ms = 100000;
        snap.heat_at_ms = 0;
        snap.last_error = "GET heat: HTTP 503";
        const auto status = draw(mm::render_status_line(snap, 130000), 100, 1);
        check(has(status, "30s ago"), "a stale reading reports its age");
        check(has(status, "never"), "a field that never succeeded says never, not 0s");
        check(has(status, "503"), "and the error stays visible");
    }

    std::cout << "\n"
              << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES") << "\n";
    return g_failures == 0 ? 0 : 1;
}
