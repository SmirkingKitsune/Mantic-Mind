#pragma once

// Soma — the internal telemetry channel to the node.
//
// Carries tier occupancy, per-expert routing heat, and scheduler state. Control
// re-publishes all of it on /v1/*, where the FTXUI TUI and any external client
// consume it identically (P1).
//
// Two constraints are decided here rather than deferred to the transport:
//
//   1. AGGREGATION HAPPENS IN THE ENGINE. Heat counters accumulate in
//      MemoryHierarchy and are sampled at the tick rate. Nothing is emitted per
//      token. A brain-grid feed over tens of thousands of experts refreshing per
//      token is orders of magnitude above the chat event stream, and a throttle
//      applied at the HTTP layer would still have paid for producing the data.
//
//   2. DOWNSAMPLING IS THE DEFAULT. Full resolution is an explicit opt-in.

#include "soma/memory_hierarchy.hpp"
#include "soma/scheduler.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace soma {

inline constexpr std::uint32_t kDefaultTelemetryHz = 2;
inline constexpr std::uint32_t kMaxTelemetryHz = 10;

/// Bucketed grids never exceed this. A 48-layer x 128-expert model is 6144
/// cells; a 60-layer x 256-expert model is 15360. Both bucket down to this cap.
inline constexpr std::uint32_t kMaxBucketedCells = 4096;

enum class HeatResolution : std::uint8_t {
    Bucketed = 0, ///< default
    Full,         ///< explicit opt-in; requires the read scope
};

/// One telemetry tick. Flat and cheap to serialize.
struct TelemetryFrame {
    std::uint64_t tick_ms = 0;
    TierOccupancy occupancy{};
    CacheStats cache{};
    SchedulerStats scheduler{};
};

/// The brain grid. `cells` is bucketed unless resolution == Full.
struct HeatFrame {
    std::uint64_t tick_ms = 0;
    HeatResolution resolution = HeatResolution::Bucketed;
    std::uint32_t layer_bucket = 1;  ///< layers aggregated per cell
    std::uint32_t expert_bucket = 1; ///< experts aggregated per cell
    std::uint32_t n_layers = 0;      ///< true dimensions, pre-bucketing
    std::uint32_t n_experts = 0;
    std::vector<HeatCell> cells;
};

using TelemetrySink = std::function<void(const TelemetryFrame&)>;
using HeatSink = std::function<void(const HeatFrame&)>;

/// Samples the engine at a fixed rate and pushes frames to its sinks.
///
/// Owns its own thread. Sampling is lock-light: occupancy and stats are atomic
/// counters, and the heat snapshot is copied under a short lock rather than held
/// across serialization.
class TelemetryChannel {
public:
    TelemetryChannel();
    TelemetryChannel(const TelemetryChannel&) = delete;
    TelemetryChannel& operator=(const TelemetryChannel&) = delete;
    ~TelemetryChannel();

    Status open(const MemoryHierarchy& memory,
                const Scheduler& scheduler,
                std::uint32_t hz = kDefaultTelemetryHz);
    void close();

    void set_telemetry_sink(TelemetrySink sink);
    void set_heat_sink(HeatSink sink);

    /// Clamped to [1, kMaxTelemetryHz].
    void set_rate(std::uint32_t hz) noexcept;
    void set_heat_resolution(HeatResolution resolution) noexcept;

    /// One-shot, for GET /v1/engines/{id}/heat and for the G3 text dump.
    Status snapshot(TelemetryFrame& out) const;
    Status snapshot_heat(HeatResolution resolution, HeatFrame& out) const;

    /// Human-readable tier/heat dump.
    ///
    /// Required by G3 — four gates before the polished FTXUI panels. Watching
    /// expert-load patterns across concurrent sequences is the primary
    /// instrument for catching cache thrash, so the debugging view has to
    /// precede the pretty one.
    Status write_text_dump(std::string& out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Bucket a full-resolution snapshot down to at most kMaxBucketedCells.
HeatFrame bucket_heat(const HeatSnapshot& snapshot, std::uint32_t max_cells);

const char* to_string(HeatResolution resolution) noexcept;

} // namespace soma
