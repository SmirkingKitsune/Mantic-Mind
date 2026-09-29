#pragma once

#include "common/models.hpp"
#include <functional>
#include <string>
#include <vector>
#include <deque>
#include <mutex>

namespace mm {

class NodeRegistry;

// FTXUI-based terminal UI for mantic-mind-control.
//
// Ten tabs (keyboard 1-9, 0):
//   1 Nodes        list + health metrics + add/remove actions
//   2 Models       the served-model catalog users pick from
//   3 Activity     scrollable system event log with level filter
//   4 Chat         chat as a user, with any model, switchable per turn
//   5 Users        users, their conversations and memories, curation
//   6 Performance  rolling request latency and throughput metrics
//   7 Voice        a user's voice: proposals, synthesis, local playback
//   8 Soma         engines, the expert-heat brain grid, memory tiers
//   9 Engines      cluster engine policy and per-node conformance
//   0 Admissions   admitting models into the registry
//
// Every tab but Nodes, Activity and Performance is fed ENTIRELY by /v1/*, in its
// own translation unit that cannot reach the references this class holds —
// tools/ci/check_ui_api.py enforces that. All state mutations come from a
// thread-safe event queue drained on the FTXUI loop tick to avoid locking inside
// the renderer.
class ControlUI {
public:
    ControlUI(NodeRegistry& registry,
              std::string control_base_url,
              std::string control_api_token);
    ~ControlUI();

    // Append a log entry (thread-safe).
    enum class LogLevel { Info, Warn, Error };
    void log(LogLevel level, const std::string& message);

    // Trigger a UI refresh after external state change (thread-safe).
    void refresh();

    // Set the pairing key used for PSK-mode auto-pairing (call before run()).
    void set_pairing_key(const std::string& k);

    // Blocks until the user quits.
    void run();

    void quit();

private:
    NodeRegistry& registry_;
    std::string   control_api_token_;
    std::string   control_base_url_;

    struct LogEntry { LogLevel level; std::string message; int64_t timestamp_ms = 0; };
    std::mutex             log_mutex_;
    std::deque<LogEntry>   log_entries_;

    std::string            pairing_key_;

    // Screen callbacks — set inside run(), cleared on exit.
    std::mutex             screen_mutex_;
    std::function<void()>  quit_fn_;
    std::function<void()>  refresh_fn_;
};

} // namespace mm
