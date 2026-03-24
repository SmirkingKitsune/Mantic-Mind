#pragma once

#include "common/models.hpp"
#include "common/node_discovery.hpp"
#include <unordered_map>
#include <vector>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <atomic>

namespace mm {

// Tracks all registered nodes and runs background health polling.
class NodeRegistry {
public:
    NodeRegistry();
    ~NodeRegistry();

    // Register a node; returns assigned NodeId.
    NodeId add_node(const std::string& url,
                    const std::string& api_key,
                    const std::string& platform = {});

    void remove_node(const NodeId& id);

    // Returns copy of NodeInfo (throws if not found).
    NodeInfo get_node(const NodeId& id) const;
    std::vector<NodeInfo> list_nodes() const;
    std::optional<NodeInfo> find_node_by_api_key(const std::string& api_key) const;

    // Update the loaded_model field for a node (deprecated — kept for backwards compat).
    void set_node_loaded_model(const NodeId& id, const std::string& model_path);

    // Nodes that have the given model currently loaded (deprecated single-model).
    std::vector<NodeInfo> nodes_with_model(const std::string& model_path) const;
    // Connected nodes (any model / idle).
    std::vector<NodeInfo> available_nodes() const;

    // ── Multi-slot queries ─────────────────────────────────────────────────────
    /// Update node's slot list (from status poll).
    void update_node_slots(const NodeId& id, const std::vector<SlotInfo>& slots);
    /// Update node's stored models and disk info (from status poll).
    void update_node_storage(const NodeId& id,
                             const std::vector<StoredModel>& models,
                             int64_t disk_free_mb);

    /// Nodes that have a model loaded in a ready slot.
    std::vector<NodeInfo> nodes_with_model_loaded(const std::string& model_path) const;
    /// Nodes that have a model file stored on disk.
    std::vector<NodeInfo> nodes_with_model_stored(const std::string& model_path) const;
    /// Nodes that can likely host a model requiring `min_vram_mb`:
    /// - preferred: enough free VRAM
    /// - fallback: VRAM + weighted RAM budget (CPU offload)
    std::vector<NodeInfo> nodes_with_available_vram(int64_t min_vram_mb) const;

    // Callback fired whenever node status changes (health poll results).
    using UpdateCallback = std::function<void(const NodeInfo&)>;
    void set_update_callback(UpdateCallback cb);

    // Start/stop background health polling (every interval_s seconds).
    void start_health_poll(int interval_s = 30);
    void stop_health_poll();

    // ── Discovery ─────────────────────────────────────────────────────────────
    void start_discovery_listen(uint16_t port = 7072);
    void stop_discovery_listen();
    // Returns discovered nodes whose URL is not already registered.
    std::vector<DiscoveredNode> get_discovered_nodes() const;

    // ── Pairing ───────────────────────────────────────────────────────────────
    // Step 1: send pair-request to node; returns the nonce on success, empty on failure.
    // Call this first — it triggers PIN generation and display on the node TUI.
    std::string start_pair(const std::string& url);

    // Step 2: send pair-complete using the nonce from start_pair plus the entered PIN or PSK.
    // Returns the new api_key on success, empty string on failure.
    std::string complete_pair(const std::string& url,
                              const std::string& nonce,
                              const std::string& pin_or_psk);

    // Convenience: does start_pair + complete_pair in one call (used for PSK auto-pairing).
    std::string pair_node(const std::string& url, const std::string& pin_or_psk);

    // Trigger node-side llama.cpp updater script (manual action).
    // Returns true if the node accepted the update job.
    bool request_llama_update(const NodeId& id,
                              bool build = true,
                              std::string* out_message = nullptr);
    bool request_llama_check_update(const NodeId& id,
                                    std::string* out_message = nullptr);

private:
    mutable std::mutex                    mutex_;
    std::unordered_map<NodeId, NodeInfo>  nodes_;
    UpdateCallback                        update_cb_;

    std::atomic<bool> polling_{false};
    std::thread       poll_thread_;

    NodeDiscoveryListener discovery_listener_;

    void poll_all_nodes();
    bool ping_node(NodeInfo& info);
};

} // namespace mm
