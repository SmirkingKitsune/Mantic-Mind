#pragma once

#include "common/footprint.hpp"
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
    /// Nodes that can host `footprint`, best fit first.
    ///
    /// REPLACES nodes_with_available_vram(int64_t). The policy is unchanged —
    /// same headroom, same 0.60 offload weight, same 8 GiB minimum GPU, and a
    /// native fit still outranks an offloaded one — but it is expressed over
    /// three axes instead of one, and it lives in common/footprint.cpp so the
    /// node and control agree on what "fits" means.
    ///
    /// Soma's cost is RAM + disk + optional VRAM, and no amount of tuning a VRAM
    /// scalar expresses that. `NodeInfo::disk_free_mb` has been collected by the
    /// health poll since it was written and consulted by nothing.
    std::vector<NodeInfo> nodes_with_capacity(const ResourceFootprint& footprint,
                                              const CapacityPolicy& policy = {}) const;

    // Callback fired whenever node status changes (health poll results).
    using UpdateCallback = std::function<void(const NodeInfo&)>;
    void set_update_callback(UpdateCallback cb);

    // ── Cluster engine configuration ──────────────────────────────────────────
    /// Supplies the current cluster engine config. Set by control's startup;
    /// unset means engine conformance is not managed and no node is ever
    /// pushed to.
    ///
    /// A provider rather than a stored copy: the config changes underneath the
    /// registry, and a copy taken at construction would push a stale version
    /// forever.
    using EngineConfigProvider = std::function<std::optional<ClusterEngineConfig>()>;
    void set_engine_config_provider(EngineConfigProvider provider);

    /// Push the configuration to one node now. Used at registration, on an
    /// operator-forced resync, and by the health poll on a version mismatch.
    /// Returns false with `out_error` on a transport failure or a node refusal.
    bool push_engine_config(const NodeId& id,
                            const ClusterEngineConfig& cfg,
                            std::string& out_error);

    /// Push to every connected node whose reported version differs. Called
    /// after a config save so a change propagates immediately rather than
    /// waiting up to one poll interval.
    void push_engine_config_to_all(const ClusterEngineConfig& cfg);

    /// Nodes reporting a conformance state that permits placement. Distinct
    /// from `connected`: a reachable node running the wrong engines is exactly
    /// the node this exists to exclude.
    std::vector<NodeInfo> conforming_nodes() const;

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
    EngineConfigProvider                  engine_config_provider_;
    std::atomic<int64_t> offline_after_ms_{90000};

    std::atomic<bool>       polling_{false};
    std::thread             poll_thread_;
    std::mutex              poll_mutex_;       // guards poll_cv_ wait predicate
    std::condition_variable poll_cv_;          // wakes the poll loop on stop

    NodeDiscoveryListener discovery_listener_;

    void poll_all_nodes();
    /// May placement target this node? Call with mutex_ held.
    ///
    /// Answers true unconditionally when no engine-config provider is set —
    /// see the definition for why gating an unmanaged registry would break it
    /// silently.
    bool placement_allowed_locked(const NodeInfo& n) const;
    bool ping_node(NodeInfo& info);
    void load_remembered_nodes();
    void save_remembered_nodes_unlocked() const;
};

} // namespace mm
