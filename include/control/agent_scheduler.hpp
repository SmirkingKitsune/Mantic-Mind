#pragma once

#include "common/models.hpp"
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mm {

class NodeRegistry;
class ModelDistributor;

/// Result of a successful scheduling decision.
struct ScheduleResult {
    NodeId node_id;
    SlotId slot_id;
};

/// VRAM-aware agent scheduler.
/// Routes agent requests to nodes, managing placements, suspension, and model
/// distribution. Thread-safe — the entire scheduling decision is serialized.
class AgentScheduler {
public:
    AgentScheduler(NodeRegistry& registry,
                   ModelDistributor& distributor,
                   std::string models_dir);

    /// Ensure an agent has a running slot on some node.
    /// Returns {node_id, slot_id} on success, nullopt if no capacity.
    /// May suspend idle agents or distribute models as needed.
    std::optional<ScheduleResult> ensure_agent_running(const AgentConfig& cfg);

    /// Release an agent's placement entirely (e.g. on agent delete).
    void release_agent(const AgentId& agent_id);

    /// Mark an agent as idle (inference complete) — updates LRU timestamp.
    void mark_agent_idle(const AgentId& agent_id);

    /// Mark an agent as active (inference starting).
    void mark_agent_active(const AgentId& agent_id);

    /// Get the current placement for an agent, if any.
    std::optional<AgentPlacement> get_placement(const AgentId& agent_id) const;

    /// List all current placements.
    std::vector<AgentPlacement> list_placements() const;

    /// Last scheduling failure reason for diagnostics.
    std::string last_error() const;

    /// Periodic housekeeping: clean stale placements, delete orphaned models.
    void housekeeping(const std::vector<AgentConfig>& active_agents);

private:
    NodeRegistry&       registry_;
    ModelDistributor&   distributor_;
    std::string         models_dir_;

    mutable std::mutex                              mutex_;
    std::unordered_map<AgentId, AgentPlacement>     placements_;
    std::string                                      last_error_;

    /// Check if a node URL is local (loopback).
    static bool is_local_node(const std::string& node_url);

    /// Estimate VRAM needed for a model (from file size).
    int64_t estimate_vram_mb(const std::string& model_path) const;

    /// Try to find a node that can accommodate a model of the given VRAM.
    /// Returns node info or nullopt.
    std::optional<NodeInfo> find_node_with_vram(int64_t vram_mb,
                                                const NodeId& preferred = {}) const;

    /// Find the LRU idle agent placement on any node, or a specific node.
    std::optional<AgentId> find_lru_idle_agent(const NodeId& on_node = {}) const;

    /// Suspend an agent: call the node's suspend-slot API, update placement.
    bool suspend_agent(const AgentId& agent_id);

    /// Restore an agent on a given node.
    std::optional<SlotId> restore_agent_on_node(const AgentPlacement& placement,
                                                const AgentConfig& cfg,
                                                const NodeId& node_id);

    /// Load a fresh model on a node for an agent.
    std::optional<SlotId> load_agent_on_node(const AgentConfig& cfg,
                                             const NodeId& node_id);

    /// True when a node API failure body indicates slot exhaustion.
    static bool response_indicates_max_slots(const std::string& body);

    /// Free one slot on a node. Prefers suspending known idle placements,
    /// then falls back to unloading a node-reported slot directly.
    bool evict_one_slot_on_node(const NodeId& node_id,
                                const AgentId& preserve_agent = {});

    /// Free multiple slots on a node. max_to_evict:
    ///  - >0 : evict up to that many slots
    ///  - <0 : evict all evictable slots
    bool evict_slots_on_node(const NodeId& node_id,
                             const AgentId& preserve_agent,
                             int max_to_evict);
};

} // namespace mm
