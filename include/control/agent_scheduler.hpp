#pragma once

#include "common/models.hpp"
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mm {

class NodeRegistry;

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
    std::string         models_dir_;

    // schedule_mutex_ serializes scheduling decisions end-to-end — these can
    // include slow node HTTP calls and multi-GB model uploads. state_mutex_
    // guards the placement map and last_error_ only, so read APIs
    // (get_placement, list_placements) and idle/active marks never block
    // behind a scheduling operation in progress.
    std::mutex                                      schedule_mutex_;
    mutable std::mutex                              state_mutex_;
    std::unordered_map<AgentId, AgentPlacement>     placements_;
    std::string                                      last_error_;

    /// Copy a placement out of the map (state_mutex_).
    std::optional<AgentPlacement> find_placement_copy(const AgentId& id) const;

    /// Insert or overwrite a placement (state_mutex_).
    void store_placement(const AgentPlacement& p);

    /// Remove a placement; returns false if absent (state_mutex_).
    bool erase_placement_entry(const AgentId& id);

    /// Apply fn to an existing placement; returns false if absent (state_mutex_).
    template <typename Fn>
    bool mutate_placement(const AgentId& id, Fn&& fn) {
        std::lock_guard<std::mutex> g(state_mutex_);
        auto it = placements_.find(id);
        if (it == placements_.end()) return false;
        fn(it->second);
        return true;
    }

    void set_last_error(const std::string& err);

    /// Check if a node URL is local (loopback).
    static bool is_local_node(const std::string& node_url);

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
