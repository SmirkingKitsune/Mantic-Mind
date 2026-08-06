// Mantic-Mind — the Soma dashboard's data source and layout.
//
// Includes, in full, and deliberately short: an HTTP client, JSON, and the
// standard library. Nothing from src/control or src/node beyond this file's own
// header. tools/ci/check_ui_api.py enforces that, because the rule is only worth
// having if it cannot be quietly broken by someone who needs one number quickly.

#include "control/soma_dashboard.hpp"

#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace mm {

namespace {

/// Parse `GET /v1/engines`. Missing fields default rather than throw: a control
/// that gains a field must not blank a panel built against the older shape.
void parse_engines(const nlohmann::json& j,
                   std::vector<SomaEngineView>& out,
                   SomaTierSummary& tiers) {
    out.clear();
    for (const auto& e : j.value("data", nlohmann::json::array())) {
        SomaEngineView v;
        v.id = e.value("id", std::string{});
        v.node_id = e.value("node_id", std::string{});
        v.node_url = e.value("node_url", std::string{});
        v.backend = e.value("backend", std::string{});
        v.state = e.value("state", std::string{});
        v.model_path = e.value("model_path", std::string{});
        v.vram_usage_mb = e.value("vram_usage_mb", std::uint64_t{0});
        v.effective_ctx_size = e.value("effective_ctx_size", std::uint32_t{0});
        for (const auto& a : e.value("agent_ids", nlohmann::json::array())) {
            if (a.is_string()) v.agent_ids.push_back(a.get<std::string>());
        }
        out.push_back(std::move(v));
    }
    const auto t = j.value("tier_summary", nlohmann::json::object());
    tiers.engines = t.value("engines", std::uint64_t{0});
    tiers.vram_mb = t.value("vram_mb", std::uint64_t{0});
    tiers.node_disk_free_mb = t.value("node_disk_free_mb", std::uint64_t{0});
}

/// Parse `GET /v1/engines/{id}/heat`.
///
/// Returns false on a frame whose arrays disagree with its dimensions. A short
/// `counts` array would otherwise render as a grid whose right-hand columns are
/// permanently cold, which reads as a routing finding rather than a parse bug.
bool parse_heat(const nlohmann::json& j, SomaHeatView& out) {
    SomaHeatView h;
    h.tick_ms = j.value("tick_ms", std::uint64_t{0});
    h.resolution = j.value("resolution", std::string{});
    h.n_layers = j.value("n_layers", std::uint32_t{0});
    h.n_experts = j.value("n_experts", std::uint32_t{0});
    h.layer_bucket = std::max(1u, j.value("layer_bucket", std::uint32_t{1}));
    h.expert_bucket = std::max(1u, j.value("expert_bucket", std::uint32_t{1}));
    h.rows = j.value("rows", std::uint32_t{0});
    h.cols = j.value("cols", std::uint32_t{0});
    for (const auto& c : j.value("counts", nlohmann::json::array())) {
        h.counts.push_back(c.is_number() ? c.get<std::uint64_t>() : 0);
    }
    for (const auto& t : j.value("tiers", nlohmann::json::array())) {
        h.tiers.push_back(t.is_number() ? static_cast<std::uint8_t>(t.get<int>()) : 2);
    }

    const auto want = static_cast<std::size_t>(h.rows) * h.cols;
    if (want == 0 || h.counts.size() != want) return false;
    // A short tier array is survivable — pad with Disk, the coldest, so a
    // missing tier never makes a cell look warmer than it is.
    h.tiers.resize(want, 2);
    out = std::move(h);
    return true;
}

} // namespace

// ── layout ───────────────────────────────────────────────────────────────────

GridLayout layout_heat(const SomaHeatView& heat, std::uint32_t max_rows, std::uint32_t max_cols) {
    GridLayout g;
    if (heat.empty() || max_rows == 0 || max_cols == 0) return g;

    // The frame's dimensions are a CLAIM about its arrays, and this function is
    // public: parse_heat() checks them, and it is not the only caller. Indexing
    // rows*cols into a shorter vector is a read past the end, and the bytes it
    // finds render as plausible heat.
    if (heat.counts.size() < static_cast<std::size_t>(heat.rows) * heat.cols) return g;

    // Ceiling division, so the stride always covers the frame. Flooring leaves a
    // remainder of rows with nowhere to go, and the cells that get dropped are
    // the LAST layers — the ones nearest the output, which is where a routing
    // anomaly is most worth seeing.
    g.row_stride = (heat.rows + max_rows - 1) / max_rows;
    g.col_stride = (heat.cols + max_cols - 1) / max_cols;
    g.viewport_reduced = (g.row_stride > 1 || g.col_stride > 1);

    g.rows = (heat.rows + g.row_stride - 1) / g.row_stride;
    g.cols = (heat.cols + g.col_stride - 1) / g.col_stride;
    g.cells.assign(static_cast<std::size_t>(g.rows) * g.cols, GridCell{});

    // A short tier array is survivable where a short count array is not: the
    // missing entries read as Disk, the coldest, so an absent tier never makes a
    // cell look warmer than it is.
    const auto tier_at = [&](std::size_t i) -> std::uint8_t {
        return i < heat.tiers.size() ? heat.tiers[i] : std::uint8_t{2};
    };

    for (std::uint32_t r = 0; r < heat.rows; ++r) {
        for (std::uint32_t c = 0; c < heat.cols; ++c) {
            const auto src = static_cast<std::size_t>(r) * heat.cols + c;
            const auto dst =
                static_cast<std::size_t>(r / g.row_stride) * g.cols + (c / g.col_stride);
            auto& cell = g.cells[dst];
            cell.present = true;
            // Traffic attributed to the tier the expert lives on, rather than the
            // bucket taking its coldest member's tier. Coldest-wins is the right
            // rule for "is a disk read POSSIBLE here" and the wrong one for "how
            // much of this region is being read from disk" — and the second is
            // the question a heat map is asked. See GridCell::tier_count.
            const auto t = std::min<std::uint8_t>(tier_at(src), 2);
            cell.tier_count[t] += heat.counts[src];
            cell.count += heat.counts[src];
            g.total += heat.counts[src];
        }
    }

    for (const auto& cell : g.cells)
        g.hottest = std::max(g.hottest, cell.count);

    // Intensity is RELATIVE to the frame's hottest cell, not absolute: the same
    // grid has to be readable after ten tokens and after ten million, and an
    // absolute scale is blank for one of those.
    if (g.hottest > 0) {
        for (auto& cell : g.cells) {
            if (cell.count == 0) {
                cell.intensity = 0;
                continue;
            }
            const double f = static_cast<double>(cell.count) / static_cast<double>(g.hottest);
            // 1..4, with any non-zero count reaching at least 1. A cell that
            // fired once must not render identically to one that never fired —
            // "cold" and "never" are different findings.
            cell.intensity = static_cast<std::uint8_t>(1 + std::min(3, static_cast<int>(f * 4.0)));
        }
    }
    return g;
}

// ── the poller ───────────────────────────────────────────────────────────────

struct SomaDashboard::Impl {
    std::string base_url;
    std::string token;

    mutable std::mutex mu;
    SomaSnapshot snap;
    std::string selected;

    std::thread worker;
    std::atomic<bool> running{false};
    std::mutex wake_mu;
    std::condition_variable wake;

    HttpClient client;

    explicit Impl(std::string url, std::string tok)
        : base_url(std::move(url)), token(std::move(tok)), client(base_url) {
        if (!token.empty()) client.set_bearer_token(token);
        // Short, because this runs on a render cadence: a dashboard that blocks
        // for 30 s on an unreachable node stops being a dashboard.
        client.set_timeouts(2, 3, 3);
    }
};

SomaDashboard::SomaDashboard(std::string base_url, std::string token)
    : impl_(std::make_unique<Impl>(std::move(base_url), std::move(token))) {}

SomaDashboard::~SomaDashboard() {
    stop();
}

void SomaDashboard::start(std::uint32_t interval_ms) {
    auto& im = *impl_;
    if (im.running.exchange(true)) return;
    im.worker = std::thread([this, interval_ms] {
        auto& im = *impl_;
        while (im.running.load()) {
            poll_once();
            std::unique_lock<std::mutex> lk(im.wake_mu);
            // Waited on rather than slept, so stop() is immediate. A dashboard
            // that takes a poll interval to shut down makes quitting the TUI
            // feel broken.
            im.wake.wait_for(
                lk, std::chrono::milliseconds(interval_ms), [&] { return !im.running.load(); });
        }
    });
}

void SomaDashboard::stop() {
    auto& im = *impl_;
    if (!im.running.exchange(false)) return;
    im.wake.notify_all();
    if (im.worker.joinable()) im.worker.join();
}

void SomaDashboard::select_engine(const std::string& id) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->selected = id;
}

std::string SomaDashboard::selected_engine() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->selected;
}

SomaSnapshot SomaDashboard::snapshot() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->snap;
}

void SomaDashboard::poll_once() {
    auto& im = *impl_;

    std::string want_heat;
    {
        std::lock_guard<std::mutex> lk(im.mu);
        want_heat = im.selected;
    }

    // Built outside the lock, published under it. The HTTP calls take hundreds
    // of milliseconds against a remote node and the render thread reads this
    // mutex every tick.
    std::vector<SomaEngineView> engines;
    SomaTierSummary tiers{};
    bool engines_ok = false;
    std::string error;

    if (const auto res = im.client.get("/v1/engines"); res.ok()) {
        try {
            parse_engines(nlohmann::json::parse(res.body), engines, tiers);
            engines_ok = true;
        } catch (const std::exception& e) {
            error = std::string("GET /v1/engines: ") + e.what();
        }
    } else {
        error = "GET /v1/engines: HTTP " + std::to_string(res.status);
    }

    SomaHeatView heat;
    bool heat_ok = false;
    if (!want_heat.empty()) {
        // Only for a Soma engine, and only when the engine list says it exists.
        // Asking a fallback for heat produces a 501 every tick, which fills the
        // log with a fact the panel already knows.
        const auto it = std::find_if(
            engines.begin(), engines.end(), [&](const auto& e) { return e.id == want_heat; });
        const bool eligible = engines_ok && it != engines.end() && it->soma();
        if (eligible) {
            if (const auto res = im.client.get("/v1/engines/" + want_heat + "/heat"); res.ok()) {
                try {
                    heat_ok = parse_heat(nlohmann::json::parse(res.body), heat);
                    if (!heat_ok) error = "heat frame dimensions disagree with its arrays";
                } catch (const std::exception& e) {
                    error = std::string("GET heat: ") + e.what();
                }
            } else if (res.status != 501) {
                // 501 is the documented answer from an engine with no telemetry.
                // Not an error, and recording it as one would make every
                // fallback engine look broken.
                error = "GET heat: HTTP " + std::to_string(res.status);
            }
        }
    }

    const auto now = util::now_ms();
    {
        std::lock_guard<std::mutex> lk(im.mu);
        if (engines_ok) {
            im.snap.engines = std::move(engines);
            im.snap.tiers = tiers;
            im.snap.engines_at_ms = now;
        }
        if (heat_ok) {
            im.snap.heat = std::move(heat);
            im.snap.heat_engine_id = want_heat;
            im.snap.heat_at_ms = now;
        } else if (want_heat != im.snap.heat_engine_id) {
            // The selection moved and the new engine has nothing to show. Drop
            // the old grid rather than leaving another engine's heat on screen
            // under this engine's name.
            im.snap.heat = {};
            im.snap.heat_engine_id = want_heat;
            im.snap.heat_at_ms = 0;
        }
        // Cleared only by a clean pass, so a transient failure stays visible
        // long enough to read.
        im.snap.last_error = error;
    }
}

} // namespace mm
