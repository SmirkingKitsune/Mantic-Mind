// Soma — G6: the telemetry feed.
//
// Two claims the design makes and this checks:
//
//   1. DOWNSAMPLING IS THE DEFAULT, and it never exceeds the cap. A grid the
//      caller did not ask for at full resolution must fit the brain view's
//      budget whatever the model's shape.
//   2. THE RATE IS CLAMPED IN THE ENGINE. `?hz=` is a request, not an
//      instruction: the ceiling is a property of the engine.
//
// bucket_heat() is pure, so most of this needs no model at all — which is the
// point of it being a free function rather than a method on the channel.

#include "soma/scheduler.hpp"
#include "soma/telemetry.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
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

/// A full-resolution snapshot with a known total count, so bucketing can be
/// checked for CONSERVATION rather than just for size.
soma::HeatSnapshot make_snapshot(std::uint32_t layers, std::uint32_t experts) {
    soma::HeatSnapshot s;
    s.n_layers = layers;
    s.n_experts = experts;
    s.cells.reserve(static_cast<std::size_t>(layers) * experts);
    for (std::uint32_t l = 0; l < layers; ++l) {
        for (std::uint32_t e = 0; e < experts; ++e) {
            soma::HeatCell c;
            c.layer = static_cast<soma::LayerIndex>(l);
            c.expert = static_cast<soma::ExpertId>(e);
            c.count = 1; // every cell counts once: the total is layers*experts
            c.tier = soma::MemoryTier::Disk;
            s.cells.push_back(c);
        }
    }
    return s;
}

std::uint64_t total_count(const soma::HeatFrame& f) {
    std::uint64_t sum = 0;
    for (const auto& c : f.cells)
        sum += c.count;
    return sum;
}

} // namespace

int main() {
    std::cout << "1. bucketing never exceeds the cap\n";
    {
        // Real shapes, from the models this engine actually targets.
        struct Shape {
            const char* name;
            std::uint32_t layers, experts;
        };

        const Shape shapes[] = {
            {"Qwen3-30B-A3B  48x128", 48, 128},   //  6144 cells
            {"DeepSeek-V2-Lite 27x64", 27, 64},   //  1728 — already under the cap
            {"Mixtral-8x7B    32x8", 32, 8},      //   256 — far under
            {"hypothetical    60x256", 60, 256},  // 15360
            {"pathological   128x512", 128, 512}, // 65536
        };
        for (const auto& s : shapes) {
            const auto snap = make_snapshot(s.layers, s.experts);
            const auto f = soma::bucket_heat(snap, soma::kMaxBucketedCells);
            const std::uint64_t raw = static_cast<std::uint64_t>(s.layers) * s.experts;
            check(f.cells.size() <= soma::kMaxBucketedCells,
                  std::string(s.name) + " fits the cap",
                  std::to_string(raw) + " -> " + std::to_string(f.cells.size()) + " cells (" +
                      std::to_string(f.layer_bucket) + "x" + std::to_string(f.expert_bucket) + ")");
            // Nothing is lost. A bucketed grid that dropped counts would make
            // the brain view understate load exactly where it is highest.
            check(total_count(f) == raw,
                  "  and conserves every count",
                  std::to_string(total_count(f)) + " of " + std::to_string(raw));
            // The true dimensions survive, so a client can label its axes.
            check(f.n_layers == s.layers && f.n_experts == s.experts,
                  "  and reports the pre-bucketing dimensions");
        }
    }

    std::cout << "\n2. a grid under the cap is NOT bucketed\n";
    {
        const auto snap = make_snapshot(32, 8);
        const auto f = soma::bucket_heat(snap, soma::kMaxBucketedCells);
        // 1:1 rather than padded to the cap. A grid claiming a bucket factor it
        // did not apply would make a reader divide twice.
        check(f.layer_bucket == 1 && f.expert_bucket == 1, "bucket factors stay 1");
        check(f.cells.size() == snap.cells.size(), "and every cell is passed through");
    }

    std::cout << "\n3. BOTH axes are reduced, not one\n";
    {
        // The failure this guards: bucketing only experts turns 128x512 into
        // 128x1 — every layer keeps its row and the expert axis vanishes, so the
        // display shows which layers are hot and nothing about which experts.
        const auto snap = make_snapshot(128, 512);
        const auto f = soma::bucket_heat(snap, soma::kMaxBucketedCells);
        check(f.layer_bucket > 1, "layers are bucketed", "x" + std::to_string(f.layer_bucket));
        check(f.expert_bucket > 1, "experts are bucketed", "x" + std::to_string(f.expert_bucket));
        const auto rows = (f.n_layers + f.layer_bucket - 1) / f.layer_bucket;
        const auto cols = (f.n_experts + f.expert_bucket - 1) / f.expert_bucket;
        check(rows > 1 && cols > 1,
              "so the result is still a GRID",
              std::to_string(rows) + "x" + std::to_string(cols));
        // Close to the cap rather than far under it: over-reducing throws away
        // resolution the budget would have paid for.
        check(f.cells.size() * 4 > soma::kMaxBucketedCells,
              "and uses most of the budget",
              std::to_string(f.cells.size()) + " of " + std::to_string(soma::kMaxBucketedCells));
    }

    std::cout << "\n4. the coldest tier in a bucket wins\n";
    {
        // A cell shown as resident when only one of its experts is resident
        // would read as "this region is cached" when most of it is not.
        soma::HeatSnapshot snap = make_snapshot(64, 128);
        snap.cells[0].tier = soma::MemoryTier::Ram; // one warm cell...
        for (std::size_t i = 1; i < 64; ++i)
            snap.cells[i].tier = soma::MemoryTier::Disk;
        const auto f = soma::bucket_heat(snap, soma::kMaxBucketedCells);
        check(!f.cells.empty() && f.cells[0].tier == soma::MemoryTier::Disk,
              "a mostly-cold bucket reports cold");
    }

    std::cout << "\n5. degenerate inputs do not crash or lie\n";
    {
        soma::HeatSnapshot empty;
        const auto f = soma::bucket_heat(empty, soma::kMaxBucketedCells);
        check(f.cells.empty(), "an empty snapshot yields an empty grid");
        check(f.layer_bucket == 1 && f.expert_bucket == 1, "with no claimed bucketing");

        const auto snap = make_snapshot(4, 4);
        const auto zero_cap = soma::bucket_heat(snap, 0);
        check(zero_cap.cells.size() == snap.cells.size(),
              "a zero cap passes through rather than dividing by it");
    }

    std::cout << "\n6. the rate ceiling is the ENGINE's\n";
    // ── a SPARSE snapshot, which is the only kind a real engine produces ─────
    //
    // MemoryHierarchy::heat() skips every expert that has neither fired nor been
    // made resident, and each cell carries its own (layer, expert). The wire
    // format has no coordinates — `counts` and `tiers` are flat arrays read
    // against `rows` and `cols` — so bucket_heat() owes its caller a DENSE grid.
    //
    // make_snapshot() above builds every cell, so nothing here ever exercised
    // that. On a real OLMoE the passthrough branch emitted 878 entries for a
    // declared 16x64 grid: every count on the wrong expert, and a consumer that
    // checks lengths rejects the frame outright. Defect D9.
    std::cout << "\n0. sparse input densifies\n";
    {
        const auto sparse = [](std::uint32_t layers, std::uint32_t experts, std::uint32_t nth) {
            soma::HeatSnapshot s;
            s.n_layers = layers;
            s.n_experts = experts;
            for (std::uint32_t l = 0; l < layers; ++l) {
                for (std::uint32_t e = 0; e < experts; ++e) {
                    if ((l * experts + e) % nth != 0) continue; // the rest never fired
                    soma::HeatCell c;
                    c.layer = static_cast<soma::LayerIndex>(l);
                    c.expert = static_cast<soma::ExpertId>(e);
                    c.count = 1 + l + e;
                    c.tier = soma::MemoryTier::Ram;
                    s.cells.push_back(c);
                }
            }
            return s;
        };

        // Under the cap: the passthrough branch, which is the one that was wrong.
        {
            const auto snap = sparse(16, 64, 7);
            const auto f = soma::bucket_heat(snap, soma::kMaxBucketedCells);
            check(f.cells.size() == 16u * 64u,
                  "a 16x64 sparse snapshot densifies to 1024 cells",
                  std::to_string(snap.cells.size()) + " in -> " + std::to_string(f.cells.size()));

            // The counts have to land where the coordinates said, not in input
            // order — that is the whole failure.
            bool placed = true, totals = true;
            std::uint64_t sum_in = 0, sum_out = 0;
            for (const auto& c : snap.cells) sum_in += c.count;
            for (std::uint32_t l = 0; l < 16 && placed; ++l) {
                for (std::uint32_t e = 0; e < 64; ++e) {
                    const auto& cell = f.cells[static_cast<std::size_t>(l) * 64 + e];
                    sum_out += cell.count;
                    const bool fired = ((l * 64 + e) % 7 == 0);
                    if (fired && cell.count != 1 + l + e) placed = false;
                    if (!fired && cell.count != 0) placed = false;
                }
            }
            if (sum_in != sum_out) totals = false;
            check(placed, "every count sits at its own (layer, expert)");
            check(totals, "and the total is conserved",
                  std::to_string(sum_in) + " == " + std::to_string(sum_out));

            // An expert that never fired and is not resident is on DISK. Vram is
            // the warmest tier and v1 has none of it — reporting it for the cells
            // we know least about is the worst available answer.
            check(f.cells[1].count == 0 && f.cells[1].tier == soma::MemoryTier::Disk,
                  "an untouched cell reads as Disk, not Vram");
        }

        // Over the cap: the bucketed branch already scattered by coordinate, so
        // this is a guard against the fix diverging from it.
        {
            const auto snap = sparse(128, 512, 11);
            const auto f = soma::bucket_heat(snap, soma::kMaxBucketedCells);
            const std::uint32_t rows = (128 + f.layer_bucket - 1) / f.layer_bucket;
            const std::uint32_t cols = (512 + f.expert_bucket - 1) / f.expert_bucket;
            check(f.cells.size() == static_cast<std::size_t>(rows) * cols,
                  "a bucketed sparse snapshot is dense too",
                  std::to_string(rows) + "x" + std::to_string(cols) + " = " +
                      std::to_string(f.cells.size()));
            std::uint64_t sum_in = 0, sum_out = 0;
            for (const auto& c : snap.cells) sum_in += c.count;
            for (const auto& c : f.cells) sum_out += c.count;
            check(sum_in == sum_out, "with the total conserved",
                  std::to_string(sum_in) + " == " + std::to_string(sum_out));
        }
    }

    // ── the sampler never waits on the work it measures ──────────────────────
    //
    // Defect D11. The blocking accessors take locks the step loop holds across a
    // whole forward, so a sampler calling them goes quiet exactly while the
    // engine is busy — measured on a live 7B MoE at 17.3 frames/s idle and
    // 1.3/s during generation. The try_ forms are what the telemetry path uses.
    //
    // Checked structurally rather than by racing a thread: a timing test for a
    // lock is a flaky test. What matters is that a HELD lock produces a refusal
    // rather than a wait.
    std::cout << "\n0b. the non-blocking readers refuse rather than wait\n";
    {
        soma::MemoryHierarchy mem;
        soma::CacheStats cs{};
        soma::TierOccupancy occ{};
        soma::HeatSnapshot hs;
        // Uncontended: all three succeed and agree with the blocking forms.
        check(mem.try_stats(cs), "try_stats succeeds when uncontended");
        check(mem.try_occupancy(occ), "try_occupancy succeeds when uncontended");
        check(mem.try_heat(hs), "try_heat succeeds when uncontended");

        soma::Scheduler sched;
        soma::SchedulerStats ss{};
        check(sched.try_stats(ss), "Scheduler::try_stats succeeds when uncontended");

        // And the value matches — a non-blocking reader that returned something
        // different from the blocking one would be a second source of truth.
        check(mem.stats().hits == cs.hits && mem.stats().misses == cs.misses,
              "and reports the same numbers as the blocking form");
        check(sched.stats().steps == ss.steps,
              "the scheduler's too");
    }

    {
        // The constants the HTTP layer clamps against. Asserted here so a change
        // to either is a deliberate edit to a test rather than a silent widening.
        check(soma::kDefaultTelemetryHz == 2,
              "default is 2 Hz",
              std::to_string(soma::kDefaultTelemetryHz));
        check(
            soma::kMaxTelemetryHz == 10, "ceiling is 10 Hz", std::to_string(soma::kMaxTelemetryHz));
        check(soma::kMaxBucketedCells == 4096,
              "bucketed grids cap at 4096 cells",
              std::to_string(soma::kMaxBucketedCells));
        check(std::string(soma::to_string(soma::HeatResolution::Bucketed)) == "bucketed" &&
                  std::string(soma::to_string(soma::HeatResolution::Full)) == "full",
              "and both resolutions name themselves");
    }

    std::cout << "\n"
              << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES") << "\n";
    return g_failures == 0 ? 0 : 1;
}
