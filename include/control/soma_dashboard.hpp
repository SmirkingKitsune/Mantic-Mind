#pragma once

// Mantic-Mind — the Soma dashboard's data source and layout.
//
// THE RULE THIS FILE EXISTS TO ENFORCE: every number a Soma panel shows arrives
// over `/v1/*`. No panel reaches into NodeRegistry, AgentScheduler,
// ControlModelRegistry, or an EngineSupervisor — not because in-process access
// would be slower, but because a TUI that reads private state is a second client
// with privileges no other client has, and it stops being evidence that the API
// is complete. P1 says the API is the single control plane; a dashboard that
// quietly bypasses it makes that claim untestable.
//
// So the rule is made MECHANICAL rather than aspirational: this header and
// src/control/soma_dashboard.cpp may not include any of those headers, and
// tools/ci/check_ui_api.py fails the build if they do. The existing TUI already
// mixes direct access and loopback HTTP, so the temptation is live and a comment
// asking nicely would not survive.
//
// The layout half is deliberately PURE — no FTXUI, no HTTP, no clock. A grid
// that maps a heat frame onto terminal cells is arithmetic, and arithmetic that
// can only be checked by looking at a terminal does not get checked.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mm {

/// One engine, as `GET /v1/engines` reports it.
///
/// A view, not a model: these are the fields the wire carries, named as the wire
/// names them. Anything the dashboard wants that is not here is a missing route,
/// which is the point — the panel cannot paper over a gap in the API by reaching
/// around it.
struct SomaEngineView {
    std::string id;
    std::string node_id;
    std::string node_url;
    std::string backend; ///< "soma" | "llama-cpp"
    std::string state;   ///< Idle | Loading | Ready | Suspended | Error | ...
    std::string model_path;
    std::vector<std::string> agent_ids;
    std::uint64_t vram_usage_mb = 0;
    std::uint32_t effective_ctx_size = 0;

    /// Only Soma engines publish heat and telemetry. A fallback engine is a
    /// first-class row that simply has no grid, and the panel says so rather
    /// than rendering an empty one that looks like an idle model.
    bool soma() const noexcept { return backend == "soma"; }
};

/// The cluster-wide totals `GET /v1/engines` returns alongside the list.
struct SomaTierSummary {
    std::uint64_t engines = 0;
    std::uint64_t vram_mb = 0;
    std::uint64_t node_disk_free_mb = 0;
};

/// A heat grid, as `GET /v1/engines/{id}/heat` reports it.
///
/// `rows`/`cols` are the GRID's dimensions, which are the model's only at full
/// resolution. `n_layers`/`n_experts` are the model's real shape, carried so the
/// panel can say "48x128 shown as 24x64" instead of implying the model is small.
struct SomaHeatView {
    std::uint64_t tick_ms = 0;
    std::string resolution; ///< "bucketed" | "full"
    std::uint32_t n_layers = 0;
    std::uint32_t n_experts = 0;
    std::uint32_t layer_bucket = 1;
    std::uint32_t expert_bucket = 1;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<std::uint64_t> counts;
    std::vector<std::uint8_t> tiers; ///< MemoryTier: 0 vram, 1 ram, 2 disk

    bool empty() const noexcept { return rows == 0 || cols == 0 || counts.empty(); }

    bool bucketed() const noexcept { return layer_bucket > 1 || expert_bucket > 1; }
};

/// What the dashboard last managed to read. Every field carries its own staleness
/// rather than the snapshot carrying one for all of them: engines and heat are
/// separate requests and one can fail while the other succeeds, and a panel that
/// blanks both because one timed out is worse than one that says "3s ago".
struct SomaSnapshot {
    std::vector<SomaEngineView> engines;
    SomaTierSummary tiers;

    /// Heat for the SELECTED engine only. Polling every engine's grid would put
    /// the cost of the dashboard on models nobody is looking at.
    std::string heat_engine_id;
    SomaHeatView heat;

    std::int64_t engines_at_ms = 0; ///< 0 = never succeeded
    std::int64_t heat_at_ms = 0;
    std::string last_error; ///< the most recent failure, kept until one succeeds
};

// ── layout ───────────────────────────────────────────────────────────────────

/// One terminal cell of the brain grid.
struct GridCell {
    /// 0..4. Derived from the cell's count relative to the frame's hottest cell,
    /// so a grid is readable whether the model fired 10 times or 10 million.
    std::uint8_t intensity = 0;

    std::uint64_t count = 0;

    /// Routing traffic attributable to each tier — [vram, ram, disk] — summing to
    /// `count`.
    ///
    /// TRAFFIC, not membership, and that distinction is the whole point. This
    /// used to be a single `tier` field holding the coldest tier PRESENT in the
    /// bucket, which ignored counts entirely: a bucket of six experts where one
    /// sits on disk and never fires reported Disk, exactly like a bucket whose
    /// disk-resident expert is the hot one. On a streamed model — where most
    /// experts are on disk by construction — that made the whole grid one colour
    /// the moment any reduction happened, and the more it reduced the more
    /// uniform it got. The channel carried no information precisely when it was
    /// needed most.
    ///
    /// An expert on disk that never fires costs nothing. One that fires
    /// constantly costs everything. Weighting by count is what tells them apart.
    std::uint64_t tier_count[3] = {0, 0, 0};

    bool present = false; ///< false past the edge of a ragged last row

    /// Share of this cell's traffic served from disk, 0 when nothing fired.
    ///
    /// The number the colour channel carries. At full resolution — one expert per
    /// cell — it is 0 or 1 and the rendering degenerates exactly to the old
    /// per-expert tier, which is the property that makes this a generalisation
    /// rather than a different metric.
    double cold_fraction() const noexcept {
        return count > 0 ? static_cast<double>(tier_count[2]) / static_cast<double>(count) : 0.0;
    }
};

/// A brain grid laid out for a viewport.
struct GridLayout {
    std::uint32_t rows = 0, cols = 0;
    std::vector<GridCell> cells; ///< rows*cols, row-major
    std::uint64_t hottest = 0;
    std::uint64_t total = 0;

    /// Set when the viewport is smaller than the frame and cells were combined
    /// HERE rather than by the engine. The engine's own bucketing is reported
    /// separately by SomaHeatView::bucketed() — two different reductions, and a
    /// panel that conflated them would tell the operator the model is a shape it
    /// is not.
    bool viewport_reduced = false;
    std::uint32_t row_stride = 1, col_stride = 1;
};

/// Lay a heat frame out for a viewport of at most `max_rows` x `max_cols`.
///
/// Pure: no FTXUI, no HTTP, no clock. The reduction conserves counts (a bucket's
/// count is the sum of its cells) and takes the COLDEST tier in a bucket, which
/// is the same rule bucket_heat() uses in the engine — a bucket containing one
/// disk-resident expert is a disk read waiting to happen, and reporting the
/// warmest member would hide exactly the cell an operator is looking for.
GridLayout layout_heat(const SomaHeatView& heat, std::uint32_t max_rows, std::uint32_t max_cols);

// ── the poller ───────────────────────────────────────────────────────────────

/// Polls `/v1/*` on a background thread and hands out immutable snapshots.
///
/// Owns no engine state and holds no reference to anything in-process. Its only
/// inputs are a base URL and a token, which is what makes the no-reach-through
/// rule checkable by a script rather than by review.
class SomaDashboard {
public:
    SomaDashboard(std::string base_url, std::string token);
    SomaDashboard(const SomaDashboard&) = delete;
    SomaDashboard& operator=(const SomaDashboard&) = delete;
    ~SomaDashboard();

    /// Start polling. Idempotent.
    void start(std::uint32_t interval_ms = 1000);
    void stop();

    /// Which engine's heat to fetch. Empty means none — the default, so a
    /// dashboard nobody has opened costs one cheap request per tick.
    void select_engine(const std::string& id);
    std::string selected_engine() const;

    /// A consistent copy. Cheap enough to call every render tick.
    SomaSnapshot snapshot() const;

    /// One synchronous poll. Exposed for tests, which must not race a thread to
    /// decide whether a route works.
    void poll_once();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mm
