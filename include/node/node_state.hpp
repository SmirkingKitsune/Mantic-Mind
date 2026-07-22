#pragma once

#include "common/models.hpp"
#include <optional>
#include <string>
#include <vector>
#include <unordered_set>
#include <condition_variable>
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

    // ── Cluster capabilities (advertised to control) ──────────────────────────
    NodeCapabilities get_capabilities() const;
    void             set_capabilities(const NodeCapabilities& caps);
    LlamaRuntimeStatus get_llama_runtime() const;
    void               set_llama_runtime(const LlamaRuntimeStatus& runtime);

    // Live install/upgrade progress for the node TUI loading bar.
    NodeActionProgress  get_action_progress() const;
    void                set_action_progress(const NodeActionProgress& p);
    void                clear_action_progress(const std::string& operation_id = {});
    bool                request_action_cancel();
    bool                action_cancel_requested(const std::string& operation_id = {}) const;

    // ── Diagnostics ───────────────────────────────────────────────────────────
    std::string get_last_error() const;
    void        set_last_error(const std::string& err);

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
    NodeCapabilities             capabilities_;
    LlamaRuntimeStatus           llama_runtime_;
    NodeActionProgress           action_progress_;
    std::string                  last_error_;
    std::unordered_set<std::string> api_keys_;

    std::optional<PendingPair>   pending_pair_;
    int64_t                      last_control_contact_ms_ = 0;
    StreamingTextState           streaming_text_;

    std::atomic<bool>            polling_{false};
    std::thread                  poll_thread_;
    std::mutex                   poll_mutex_;   // guards poll_cv_ wait predicate
    std::condition_variable      poll_cv_;      // wakes the poll loop on stop
    MetricsCallback              metrics_cb_;

    static NodeHealthMetrics sample_metrics();
};

} // namespace mm
