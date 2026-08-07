// Soma — the internal telemetry channel.
//
// Two constraints are enforced HERE rather than deferred to the transport, and
// both are the reason this class exists at all instead of the HTTP layer just
// polling:
//
//   1. AGGREGATION HAPPENS IN THE ENGINE. Heat counters accumulate in
//      MemoryHierarchy and are sampled at the tick rate. Nothing is emitted per
//      token. A throttle applied at the HTTP layer would still have paid for
//      producing the data — and on a 60k-expert model, producing it per token is
//      orders of magnitude above the chat stream it shares a process with.
//
//   2. DOWNSAMPLING IS THE DEFAULT. Full resolution is an explicit opt-in, so a
//      client cannot ask for 60k cells by accident.

#include "soma/telemetry.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>

namespace soma {

const char* to_string(HeatResolution resolution) noexcept {
    switch (resolution) {
    case HeatResolution::Bucketed:
        return "bucketed";
    case HeatResolution::Full:
        return "full";
    }
    return "unknown";
}

namespace {

std::uint64_t now_ms() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::system_clock::now().time_since_epoch())
                                          .count());
}

} // namespace

/// A sparse snapshot as the dense n_layers x n_experts grid the wire promises.
///
/// Absent cells are count 0 on DISK, which is what their absence means: heat()
/// skips exactly those experts that have neither fired nor been made resident.
/// Defaulting them to Vram would report the warmest tier for the cells we know
/// least about — and in a CPU-only v1 where the VRAM tier is always empty.
std::vector<HeatCell> densify(const HeatSnapshot& snapshot) {
    std::vector<HeatCell> grid(static_cast<std::size_t>(snapshot.n_layers) * snapshot.n_experts);
    for (std::uint32_t l = 0; l < snapshot.n_layers; ++l) {
        for (std::uint32_t e = 0; e < snapshot.n_experts; ++e) {
            auto& cell = grid[static_cast<std::size_t>(l) * snapshot.n_experts + e];
            cell.layer = static_cast<LayerIndex>(l);
            cell.expert = static_cast<ExpertId>(e);
            cell.tier = MemoryTier::Disk;
        }
    }
    for (const auto& src : snapshot.cells) {
        if (src.layer == kInvalidLayer || src.expert == kInvalidExpert) continue;
        if (src.layer >= snapshot.n_layers || src.expert >= snapshot.n_experts) continue;
        grid[static_cast<std::size_t>(src.layer) * snapshot.n_experts + src.expert] = src;
    }
    return grid;
}

HeatFrame bucket_heat(const HeatSnapshot& snapshot, std::uint32_t max_cells) {
    HeatFrame out;
    out.tick_ms = now_ms();
    out.resolution = HeatResolution::Bucketed;
    out.n_layers = snapshot.n_layers;
    out.n_experts = snapshot.n_experts;

    const std::uint64_t total = static_cast<std::uint64_t>(snapshot.n_layers) * snapshot.n_experts;
    if (total == 0 || max_cells == 0 || total <= max_cells) {
        // Already small enough — 1:1, no bucket factor, because a grid claiming
        // one it did not apply would make a reader divide twice.
        //
        // DENSIFIED, not copied. MemoryHierarchy::heat() returns a SPARSE list:
        // it skips every expert that has neither fired nor been made resident,
        // and each cell carries its own (layer, expert). The wire format does
        // not — heat_frame_json emits `counts` and `tiers` as flat arrays
        // alongside `rows` and `cols`, so a consumer reads them as dense and
        // indexes `r * cols + c`.
        //
        // Copying the sparse list through therefore produced a frame that
        // declared 16x64 and carried 878 entries: every count landed on the
        // wrong expert, and a strict consumer rejected the frame outright. The
        // bucketed path below has always scattered by coordinate into a dense
        // grid; only this branch disagreed with it. Measured on a real OLMoE —
        // the fixtures are 4x16 and dense enough that every cell is touched, so
        // nothing ever exercised the gap.
        out.cells = densify(snapshot);
        return out;
    }

    // Split the reduction across BOTH axes rather than collapsing one.
    //
    // A 48x128 grid bucketed only on experts becomes 48x1: every layer keeps its
    // own row and the expert axis vanishes, so the display shows which layers are
    // hot and nothing about which experts. Reducing both keeps the grid a grid,
    // which is the entire point of the brain view.
    const double scale = std::sqrt(static_cast<double>(total) / max_cells);
    std::uint32_t lb = std::max<std::uint32_t>(1, static_cast<std::uint32_t>(std::ceil(scale)));
    std::uint32_t eb = lb;

    // Widen the LONGER axis until it fits. The sqrt is a starting point; integer
    // ceilings can leave the product a few cells over, and growing whichever
    // dimension is currently larger keeps the grid closer to square than
    // repeatedly halving one would.
    while (true) {
        const std::uint64_t r = (snapshot.n_layers + lb - 1) / lb;
        const std::uint64_t c = (snapshot.n_experts + eb - 1) / eb;
        if (r * c <= max_cells) break;
        if (r >= c)
            ++lb;
        else
            ++eb;
    }

    out.layer_bucket = lb;
    out.expert_bucket = eb;
    const std::uint32_t rows = (snapshot.n_layers + lb - 1) / lb;
    const std::uint32_t cols = (snapshot.n_experts + eb - 1) / eb;

    std::vector<HeatCell> grid(static_cast<std::size_t>(rows) * cols);
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < cols; ++c) {
            auto& cell = grid[static_cast<std::size_t>(r) * cols + c];
            cell.layer = static_cast<LayerIndex>(r * lb);
            cell.expert = static_cast<ExpertId>(c * eb);
            // Seeded at Vram so the max() below resolves to the COLDEST tier
            // any source contributes — a cell shown as resident when only one of
            // its experts is would read as "this region is cached" when most of
            // it is not.
            //
            // Corrected to Disk after the scatter if nothing landed here: a
            // bucket with no sources is one whose experts have never fired and
            // are not resident, and leaving it at the seed would report VRAM for
            // the cells we know least about.
            cell.tier = MemoryTier::Vram;
            cell.count = 0;
        }
    }

    for (const auto& src : snapshot.cells) {
        if (src.layer == kInvalidLayer || src.expert == kInvalidExpert) continue;
        const std::uint32_t r = static_cast<std::uint32_t>(src.layer) / lb;
        const std::uint32_t c = static_cast<std::uint32_t>(src.expert) / eb;
        if (r >= rows || c >= cols) continue;
        auto& cell = grid[static_cast<std::size_t>(r) * cols + c];
        cell.count += src.count;
        cell.decayed += src.decayed;
        if (static_cast<int>(src.tier) > static_cast<int>(cell.tier)) cell.tier = src.tier;
    }

    for (auto& cell : grid) {
        if (cell.count == 0 && cell.decayed == 0.0f) cell.tier = MemoryTier::Disk;
    }

    out.cells = std::move(grid);
    return out;
}

// ── TelemetryChannel ──────────────────────────────────────────────────────────

struct TelemetryChannel::Impl {
    const MemoryHierarchy* memory = nullptr;
    const Scheduler* scheduler = nullptr;

    std::atomic<std::uint32_t> hz{kDefaultTelemetryHz};
    std::atomic<HeatResolution> resolution{HeatResolution::Bucketed};
    std::atomic<bool> running{false};

    mutable std::mutex mu; ///< guards the sinks only

    /// Carried across ticks so a contended read can reuse its last value rather
    /// than emit a zero. Touched only by the ticker thread.
    TelemetryFrame last_frame{};
    TelemetrySink on_frame;
    HeatSink on_heat;

    std::thread ticker;
    std::mutex wake_mu;
    std::condition_variable wake;

    void loop() {
        while (running.load()) {
            {
                // Sinks copied out under the lock and called outside it. A sink
                // writes to a socket; holding the lock across that would let one
                // slow client stall the tick for every other.
                TelemetrySink frame_sink;
                HeatSink heat_sink;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    frame_sink = on_frame;
                    heat_sink = on_heat;
                }
                if (frame_sink) {
                    // try_ everywhere, last-good-value on contention. The step
                    // loop holds the scheduler's mutex across a whole forward and
                    // the hierarchy's across expert reads, so the blocking forms
                    // make this thread sample at the rate the engine happens to
                    // be idle — measured at 1.3 frames/s against 17.3 idle.
                    TelemetryFrame f = last_frame;
                    f.tick_ms = now_ms();
                    f.stale = false;
                    if (memory != nullptr) {
                        if (!memory->try_occupancy(f.occupancy)) f.stale = true;
                        if (!memory->try_stats(f.cache)) f.stale = true;
                    }
                    if (scheduler != nullptr && !scheduler->try_stats(f.scheduler)) {
                        f.stale = true;
                    }
                    last_frame = f;
                    frame_sink(f);
                }
                if (heat_sink && memory != nullptr) {
                    // A heat frame is only emitted when it is FRESH. Unlike the
                    // counters above, a grid is what an operator reads spatially:
                    // republishing the previous one at tick rate would animate a
                    // still image, and there is no honest way to shade a cell
                    // "this is last tick's".
                    HeatSnapshot snap;
                    if (!memory->try_heat(snap)) {
                        std::unique_lock<std::mutex> lk(wake_mu);
                        wake.wait_for(
                            lk,
                            std::chrono::milliseconds(
                                1000 / std::clamp<std::uint32_t>(hz.load(), 1, kMaxTelemetryHz)),
                            [&] { return !running.load(); });
                        continue;
                    }
                    heat_sink(resolution.load() == HeatResolution::Full
                                  ? [&] {
                                        HeatFrame full;
                                        full.tick_ms = now_ms();
                                        full.resolution = HeatResolution::Full;
                                        full.n_layers = snap.n_layers;
                                        full.n_experts = snap.n_experts;
                                        full.cells = snap.cells;
                                        return full;
                                    }()
                                  : bucket_heat(snap, kMaxBucketedCells));
                }
            }

            const auto rate = std::clamp<std::uint32_t>(hz.load(), 1, kMaxTelemetryHz);
            std::unique_lock<std::mutex> lk(wake_mu);
            // Waits on a condition rather than sleeping, so close() is immediate
            // rather than up to a full tick late.
            wake.wait_for(
                lk, std::chrono::milliseconds(1000 / rate), [this] { return !running.load(); });
        }
    }
};

TelemetryChannel::TelemetryChannel() : impl_(std::make_unique<Impl>()) {}

TelemetryChannel::~TelemetryChannel() {
    close();
}

Status TelemetryChannel::open(const MemoryHierarchy& memory,
                              const Scheduler& scheduler,
                              std::uint32_t hz) {
    close();
    impl_->memory = &memory;
    impl_->scheduler = &scheduler;
    impl_->hz.store(std::clamp<std::uint32_t>(hz, 1, kMaxTelemetryHz));
    impl_->running.store(true);
    impl_->ticker = std::thread([p = impl_.get()] { p->loop(); });
    return {};
}

void TelemetryChannel::close() {
    if (!impl_->running.exchange(false)) return;
    impl_->wake.notify_all();
    if (impl_->ticker.joinable()) impl_->ticker.join();
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->on_frame = nullptr;
    impl_->on_heat = nullptr;
}

void TelemetryChannel::set_telemetry_sink(TelemetrySink sink) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->on_frame = std::move(sink);
}

void TelemetryChannel::set_heat_sink(HeatSink sink) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->on_heat = std::move(sink);
}

void TelemetryChannel::set_rate(std::uint32_t hz) noexcept {
    // CLAMPED here, not validated at the caller. The rate arrives from a query
    // parameter, and a client asking for 1000 Hz must get 10 rather than an
    // error — the ceiling is a property of the engine, not a mistake by the
    // caller.
    impl_->hz.store(std::clamp<std::uint32_t>(hz, 1, kMaxTelemetryHz));
    impl_->wake.notify_all();
}

void TelemetryChannel::set_heat_resolution(HeatResolution resolution) noexcept {
    impl_->resolution.store(resolution);
}

Status TelemetryChannel::snapshot(TelemetryFrame& out) const {
    out = TelemetryFrame{};
    out.tick_ms = now_ms();
    if (impl_->memory != nullptr) {
        out.occupancy = impl_->memory->occupancy();
        out.cache = impl_->memory->stats();
    }
    if (impl_->scheduler != nullptr) out.scheduler = impl_->scheduler->stats();
    return {};
}

Status TelemetryChannel::snapshot_heat(HeatResolution resolution, HeatFrame& out) const {
    if (impl_->memory == nullptr) {
        // A resident model has no MemoryHierarchy at all, and that is a normal
        // configuration rather than a failure — the grid is simply empty.
        out = HeatFrame{};
        out.tick_ms = now_ms();
        out.resolution = resolution;
        return {};
    }
    const auto snap = impl_->memory->heat();
    if (resolution == HeatResolution::Full) {
        out = HeatFrame{};
        out.tick_ms = now_ms();
        out.resolution = HeatResolution::Full;
        out.n_layers = snap.n_layers;
        out.n_experts = snap.n_experts;
        out.cells = snap.cells;
    } else {
        out = bucket_heat(snap, kMaxBucketedCells);
    }
    return {};
}

Status TelemetryChannel::write_text_dump(std::string& out) const {
    // The G3 instrument, four gates before the polished panels. Watching
    // expert-load patterns across concurrent sequences is how cache thrash gets
    // caught, so the debugging view precedes the pretty one.
    TelemetryFrame f;
    (void)snapshot(f);
    HeatFrame h;
    (void)snapshot_heat(HeatResolution::Bucketed, h);

    std::ostringstream o;
    o << "tier   vram=" << f.occupancy.vram_experts << " ram=" << f.occupancy.ram_experts
      << " disk=" << f.occupancy.disk_experts << " pinned=" << f.occupancy.pinned_experts
      << "  ram=" << (f.occupancy.ram_bytes >> 20) << "/" << (f.occupancy.ram_capacity_bytes >> 20)
      << " MiB\n";

    const auto lookups = f.cache.hits + f.cache.misses;
    o << "cache  hits=" << f.cache.hits << " misses=" << f.cache.misses
      << " evictions=" << f.cache.evictions << " read=" << (f.cache.bytes_read >> 20) << " MiB";
    if (lookups > 0) {
        o << "  hit_rate=" << std::fixed << std::setprecision(1)
          << (100.0 * static_cast<double>(f.cache.hits) / static_cast<double>(lookups)) << "%";
    }
    o << "\n";

    o << "sched  active=" << f.scheduler.active_sequences << " batch=" << f.scheduler.current_batch
      << "/" << f.scheduler.effective_max_batch << " steps=" << f.scheduler.steps
      << " tokens=" << f.scheduler.tokens_out;
    if (f.scheduler.unique_experts_last_step > 0) {
        // THE payoff, in one number. A ratio near 1.0 means the union is buying
        // nothing and something upstream is wrong.
        o << "  union=" << std::fixed << std::setprecision(2)
          << (static_cast<double>(f.scheduler.naive_expert_reads_last_step) /
              f.scheduler.unique_experts_last_step)
          << "x";
    }
    o << "\n";

    o << "heat   " << h.n_layers << " layers x " << h.n_experts << " experts";
    if (h.layer_bucket > 1 || h.expert_bucket > 1) {
        o << " (bucketed " << h.layer_bucket << "x" << h.expert_bucket << ")";
    }
    o << ", " << h.cells.size() << " cells\n";

    std::vector<const HeatCell*> hottest;
    hottest.reserve(h.cells.size());
    for (const auto& c : h.cells) {
        if (c.count > 0) hottest.push_back(&c);
    }
    std::partial_sort(hottest.begin(),
                      hottest.begin() + std::min<std::size_t>(8, hottest.size()),
                      hottest.end(),
                      [](const HeatCell* a, const HeatCell* b) { return a->count > b->count; });
    for (std::size_t i = 0; i < std::min<std::size_t>(8, hottest.size()); ++i) {
        o << "       L" << static_cast<int>(hottest[i]->layer) << " E"
          << static_cast<int>(hottest[i]->expert) << "  " << hottest[i]->count << " ("
          << to_string(hottest[i]->tier) << ")\n";
    }

    out = o.str();
    return {};
}

} // namespace soma
