#pragma once

#include "common/models.hpp"
#include "common/node_discovery.hpp"
#include <unordered_map>
#include <vector>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <atomic>
#include <unordered_set>

namespace mm {

// Peers that should be nudged to re-check for a vLLM update when the node
// `source_id` just updated: connected, same environment (accelerator | platform
// | arch), and on a different version. Excludes the source. Pure; unit-tested.
std::vector<NodeId> select_vllm_update_peers(const std::vector<NodeInfo>& nodes,
                                             const NodeId& source_id);

NodeConnectionStatus classify_node_reachability(int64_t unreachable_since_ms,
                                                int64_t now_ms,
                                                int64_t offline_after_ms);


// Tracks all registered nodes and runs background health polling.
class NodeRegistry {
public:
    NodeRegistry();
    explicit NodeRegistry(std::string data_dir);
    ~NodeRegistry();

    // Register a node; returns assigned NodeId.
    NodeId add_node(const std::string& url,
                    const std::string& api_key,
                    const std::string& platform = {},
                    bool remember = false,
                    const std::string& hostname = {});

    void set_offline_after_seconds(int seconds);

    void remove_node(const NodeId& id);
    bool forget_node(const NodeId& id);

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

    /// Nodes that have a model loaded in a ready slot.
    std::vector<NodeInfo> nodes_with_model_loaded(const std::string& model_path) const;
    /// Nodes that can likely host a model requiring `min_vram_mb`:
    /// - preferred: enough free VRAM
    /// - fallback: VRAM + weighted RAM budget (CPU offload)
    std::vector<NodeInfo> nodes_with_available_vram(int64_t min_vram_mb) const;
    /// Nodes whose HF cache already holds model_ref (avoids a fresh download).
    std::vector<NodeInfo> nodes_with_model_cached(const std::string& model_ref) const;

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
                              const std::string& pin_or_psk,
                              bool remember = false);

    // Convenience: does start_pair + complete_pair in one call (used for PSK auto-pairing).
    std::string pair_node(const std::string& url,
                          const std::string& pin_or_psk,
                          bool remember = false);

private:
    mutable std::mutex                    mutex_;
    std::unordered_map<NodeId, NodeInfo>  nodes_;
    std::unordered_set<NodeId>            remembered_nodes_;
    std::string                           remembered_nodes_path_;
    UpdateCallback                        update_cb_;
    // Last vLLM runtime version seen per node, used to detect an update and nudge
    // same-environment peers to check (cluster version convergence).
    std::unordered_map<NodeId, std::string> last_vllm_version_;
    std::atomic<int64_t> offline_after_ms_{90000};

    std::atomic<bool>       polling_{false};
    std::thread             poll_thread_;
    std::mutex              poll_mutex_;       // guards poll_cv_ wait predicate
    std::condition_variable poll_cv_;          // wakes the poll loop on stop

    NodeDiscoveryListener discovery_listener_;

    void poll_all_nodes();
    bool ping_node(NodeInfo& info);
    // When node `source_id` just updated its vLLM runtime, POST a check-update to
    // every same-environment peer (see select_vllm_update_peers). Each peer then
    // follows its own policy (prompt/auto/manual). Makes HTTP calls; call without
    // mutex_ held.
    void nudge_environment_peers(const NodeId& source_id);
    void load_remembered_nodes();
    void save_remembered_nodes_unlocked() const;
};

} // namespace mm
