#pragma once

#include "common/models.hpp"
#include <optional>
#include <string>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <functional>
#include <thread>

namespace mm {

// In-progress pairing handshake.  Expires 60 seconds after creation.
struct PendingPair {
    std::string challenge;          // nonce sent by control
    std::string expected_response;  // HMAC(pin_or_psk, challenge)
    std::string pin;                // 6-digit PIN shown in TUI; empty in PSK mode
    int64_t     expiry_ms;          // ms timestamp after which this is invalid
};

struct StreamingTextState {
    SlotId      slot_id;
    AgentId     agent_id;
    std::string content;
    std::string thinking;
    int         tokens_used   = 0;
    std::string finish_reason;
    bool        active        = false;
    int64_t     started_ms    = 0;
    int64_t     updated_ms    = 0;
};

struct LlamaUpdateState {
    bool        running = false;
    std::string status  = "idle"; // idle|running|succeeded|failed
    std::string message;
    int64_t     started_ms  = 0;
    int64_t     finished_ms = 0;
};

struct LlamaRuntimeSummary {
    std::string install_root;
    std::string repo_dir;
    std::string build_dir;
    std::string binary_path;

    std::string installed_commit;
    std::string remote_commit;
    std::string remote_error;
    int64_t     remote_checked_ms = 0;
    bool        update_available  = false;
    std::string update_reason     = "unknown";

    std::string last_log_path;
};

// Thread-safe shared state for the node process.
// Metrics are updated by a background polling thread.
class NodeState {
public:
    NodeState();
    ~NodeState();

    // ── Registration ─────────────────────────────────────────────────────────
    bool        is_registered()  const;
    NodeId      get_node_id()    const;
    void        set_registered(bool v, const NodeId& id = {});
    void        mark_control_contact();
    int64_t     get_last_control_contact_ms() const;
    bool        has_recent_control_contact(int64_t max_age_ms) const;

    // ── Model / agent tracking (deprecated — kept for backwards compat) ─────
    std::string get_loaded_model() const;
    std::string get_active_agent() const;
    void        set_loaded_model(const std::string& model);
    void        set_active_agent(const std::string& agent_id);

    // ── Multi-slot tracking ─────────────────────────────────────────────────
    std::vector<SlotInfo> get_slots() const;
    void                  set_slots(const std::vector<SlotInfo>& slots);

    // ── Health metrics ────────────────────────────────────────────────────────
    NodeHealthMetrics get_metrics() const;
    void              update_metrics(const NodeHealthMetrics& m);

    // ── Diagnostics ───────────────────────────────────────────────────────────
    std::string get_last_error() const;
    void        set_last_error(const std::string& err);
    std::string get_llama_server_path() const;
    void        set_llama_server_path(const std::string& path);

    LlamaUpdateState get_llama_update_state() const;
    bool             start_llama_update(const std::string& message);
    void             finish_llama_update(bool success, const std::string& message);
    void             set_llama_update_message(const std::string& message);
    LlamaRuntimeSummary get_llama_runtime_summary() const;
    void                set_llama_runtime_summary(const LlamaRuntimeSummary& summary);

    // ── Streaming text (for TUI display of generated output) ─────────────────
    StreamingTextState get_streaming_text() const;
    void start_streaming_text(const SlotId& slot_id, const AgentId& agent_id);
    void append_streaming_text(const std::string& delta_content,
                               const std::string& thinking_delta,
                               int tokens_used = 0);
    void finish_streaming_text(const std::string& finish_reason, int tokens_used);
    void clear_streaming_text();

    // ── API keys ──────────────────────────────────────────────────────────────
    void                     add_api_key(const std::string& key);
    void                     remove_api_key(const std::string& key);
    bool                     validate_api_key(const std::string& key) const;
    std::vector<std::string> get_api_keys() const;

    // ── Pairing ───────────────────────────────────────────────────────────────
    void                       set_pending_pair(const PendingPair& p);
    std::optional<PendingPair> get_pending_pair() const; // nullopt if expired
    void                       clear_pending_pair();

    // ── Background metrics polling ────────────────────────────────────────────
    // Polls CPU/RAM/GPU every `interval_ms` milliseconds.
    void start_metrics_poll(int interval_ms = 2000);
    void stop_metrics_poll();

    // Callback fired after each metrics refresh (for UI updates).
    using MetricsCallback = std::function<void(const NodeHealthMetrics&)>;
    void set_metrics_callback(MetricsCallback cb);

private:
    mutable std::mutex           mutex_;
    bool                         registered_   = false;
    NodeId                       node_id_;
    std::string                  loaded_model_;
    std::string                  active_agent_;
    std::vector<SlotInfo>        slots_;
    NodeHealthMetrics            metrics_;
    std::string                  last_error_;
    std::string                  llama_server_path_;
    std::unordered_set<std::string> api_keys_;

    std::optional<PendingPair>   pending_pair_;
    int64_t                      last_control_contact_ms_ = 0;
    LlamaUpdateState             llama_update_;
    LlamaRuntimeSummary          llama_runtime_;
    StreamingTextState           streaming_text_;

    std::atomic<bool>            polling_{false};
    std::thread                  poll_thread_;
    MetricsCallback              metrics_cb_;

    static NodeHealthMetrics sample_metrics(const std::string& llama_server_path = "");
};

} // namespace mm
