#pragma once

#include "common/models.hpp"
#include <functional>
#include <string>
#include <vector>
#include <mutex>

namespace mm {

class NodeRegistry;
class AgentManager;

// FTXUI-based terminal UI for mantic-mind-control.
//
// Five tabs (keyboard 1/2/3/4/5):
//   Tab 1 — Nodes:    list + health metrics + add/remove actions
//   Tab 2 — Agents:   list + full config editor + memories + conversations
//   Tab 3 — Activity: scrollable system event log with level filter
//   Tab 4 — Chat:     in-app agent chat for troubleshooting model behavior
//   Tab 5 — Curation: conversation trees and memory management actions
//
// All state mutations come from a thread-safe event queue drained on the
// FTXUI loop tick to avoid locking inside the renderer.
class ControlUI {
public:
    using LocalChatFallback = std::function<bool(
        const std::string& agent_id,
        const std::string& message,
        std::string* out_text,
        std::string* out_conv_id,
        std::string* out_error)>;

    ControlUI(NodeRegistry& registry,
              AgentManager& agents,
              std::string models_dir,
              std::string control_base_url,
              LocalChatFallback local_chat_fallback = {});
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
    AgentManager& agents_;
    std::string   models_dir_;
    std::string   control_base_url_;
    LocalChatFallback local_chat_fallback_;

    struct LogEntry { LogLevel level; std::string message; int64_t timestamp_ms = 0; };
    std::mutex             log_mutex_;
    std::vector<LogEntry>  log_entries_;

    std::string            pairing_key_;

    // Screen callbacks — set inside run(), cleared on exit.
    std::mutex             screen_mutex_;
    std::function<void()>  quit_fn_;
    std::function<void()>  refresh_fn_;
};

} // namespace mm
