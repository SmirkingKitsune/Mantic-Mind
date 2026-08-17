#pragma once

// Mantic-Mind — the Engines tab: what the cluster is configured to run, and
// which nodes actually run it.
//
// THE SAME RULE AS soma_dashboard.hpp, for the same reason: every value here
// arrives over `/v1/cluster/engines/*`. This panel does not read NodeRegistry
// even though it renders per-node rows, and it does not read EngineConfigStore
// even though it renders the configuration — a TUI that reached into either
// would be a client with privileges no other client has, and P1's claim that
// the API is the single control plane would stop being testable.
//
// The claim is load-bearing HERE more than anywhere: the CLI's `engines setup`
// wizard and this tab must ask for the same decision and enforce the same
// rules. They do, because both are HTTP clients of the same two routes. A tab
// that wrote the store directly would be a second implementation of the
// validation, and the first divergence would be invisible.
//
// tools/ci/check_ui_api.py guards this header and its .cpp.
//
// Rendering is PURE over a snapshot, so the panels can be drawn to an off-screen
// buffer and asserted as text. The states worth testing are the ones nobody
// opens the app in: unconfigured, mid-build, and drifted.

#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mm {

/// One engine on one node, as the conformance route reports it.
struct EnginePanelRuntime {
    std::string engine_id;
    std::string status; ///< resolved | ready | building | error | absent
    std::string version;
    std::string variant;
    std::string last_error;
    bool ready = false;
};

/// One node's engine conformance.
struct EnginePanelNode {
    std::string node_id;
    std::string hostname;
    std::string url;
    bool connected = false;
    std::string state; ///< unconfigured | converging | conforming | drifted | failed
    std::string detail;
    std::string needs_artifact;
    std::uint32_t config_version = 0;
    bool placement_eligible = false;
    std::vector<EnginePanelRuntime> engines;

    /// Live build/transfer progress, straight from NodeActionProgress.
    bool progress_active = false;
    std::string progress_action;
    std::string progress_stage;
    double progress_fraction = -1.0;
};

/// Everything the tab draws, in one consistent read.
struct EngineSnapshot {
    bool reachable = false;  ///< did the last poll succeed?
    bool configured = false; ///< has a policy ever been set?
    std::string error;       ///< why the poll failed, when it did

    std::uint32_t config_version = 0;
    std::string primary_engine;
    std::string backup_engine; ///< empty means NO backup, which is a real choice
    bool share_builds = true;

    std::vector<EnginePanelNode> nodes;
    int conforming = 0;

    /// Engine ids any node reports it can run. Drives the configuration
    /// choices, so a node that grows a third engine appears here without a
    /// code change.
    std::vector<std::string> known_engines;
};

/// Polls `/v1/cluster/engines/{config,conformance}` on its own thread.
///
/// On its own thread for the reason the Soma dashboard states: an HTTP round
/// trip on the render path stalls every other panel.
class EngineDashboard {
public:
    EngineDashboard(std::string base_url, std::string api_token);
    ~EngineDashboard();

    EngineDashboard(const EngineDashboard&) = delete;
    EngineDashboard& operator=(const EngineDashboard&) = delete;

    void start(int interval_ms);
    void stop();
    /// Poll once, now — after a save, so the tab does not show the old version
    /// for up to a full interval.
    void refresh_now();

    EngineSnapshot snapshot() const;

    /// PUT a new configuration. Returns false with `out_error` carrying the
    /// server's message — including the forbidden-key refusal, which is the one
    /// an operator most needs to read verbatim.
    bool save(const std::string& primary,
              const std::string& backup,
              bool share_builds,
              std::string& out_error);

    /// Re-push the current configuration to every node whose version differs.
    bool resync(std::string& out_error);

    /// Broker one artifact. `source_node_id` empty lets control pick a holder.
    bool share(const std::string& fingerprint,
               const std::string& target_node_id,
               const std::string& source_node_id,
               std::string& out_error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ── Pure renderers ────────────────────────────────────────────────────────────

/// The policy panel: primary, backup (or an explicit "none"), share-builds.
ftxui::Element render_engine_config(const EngineSnapshot& snap);

/// Node × conformance, with the drift reason inline.
///
/// `selected` indexes `snap.nodes`; out of range renders without a highlight
/// rather than clamping, so a stale selection after a node disappears is visible
/// instead of silently pointing at its neighbour.
ftxui::Element render_conformance_table(const EngineSnapshot& snap, int selected);

/// Live build/transfer progress across the cluster. Renders an explicit "no
/// activity" rather than an empty box — a quiet cluster and a broken poll look
/// identical otherwise.
ftxui::Element render_engine_activity(const EngineSnapshot& snap);

/// The banner shown on every other tab while no policy exists. Empty element
/// when configured.
ftxui::Element render_unconfigured_banner(const EngineSnapshot& snap);

/// The whole tab: panels plus the configuration form.
///
/// `selected_node` indexes the conformance table. `force_setup` is set by the
/// caller when no configuration exists, so the form opens by itself — the tab
/// does not decide that for itself, because the same banner has to appear on
/// every other tab and one owner of that state is one fewer way for the two to
/// disagree.
ftxui::Component engine_tab(EngineDashboard& dashboard, int& selected_node, bool& force_setup);

} // namespace mm
