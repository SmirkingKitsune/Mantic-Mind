#pragma once

#include "common/models.hpp"

#include <mutex>
#include <condition_variable>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mm {

class NodeRegistry;

struct ScheduleResult {
    NodeId node_id;
    SlotId slot_id;
};

/// VRAM-aware scheduler for llama.cpp agents.
class AgentScheduler {
public:
    AgentScheduler(NodeRegistry& registry, std::string models_dir);

    std::optional<ScheduleResult> ensure_agent_running(const AgentConfig& cfg);
    void release_agent(const AgentId& agent_id);
    void mark_agent_idle(const AgentId& agent_id);
    void mark_agent_active(const AgentId& agent_id);
    std::optional<AgentPlacement> get_placement(const AgentId& agent_id) const;
    std::vector<AgentPlacement> list_placements() const;
    std::string last_error() const;
    void housekeeping(const std::vector<AgentConfig>& active_agents);

private:
    NodeRegistry& registry_;
    std::string models_dir_;

    // Serialize competing schedules for the same agent without holding a
    // scheduler mutex during node operations or callbacks.
    std::mutex coordination_mutex_;
    std::condition_variable coordination_cv_;
    std::unordered_set<AgentId> scheduling_agents_;
    mutable std::mutex state_mutex_;
    std::unordered_map<AgentId, AgentPlacement> placements_;
    std::string last_error_;

    std::optional<AgentPlacement> find_placement_copy(const AgentId& id) const;
    void store_placement(const AgentPlacement& placement);
    bool erase_placement_entry(const AgentId& id);

    template <typename Fn>
    bool mutate_placement(const AgentId& id, Fn&& fn) {
        std::lock_guard<std::mutex> guard(state_mutex_);
        const auto it = placements_.find(id);
        if (it == placements_.end()) return false;
        fn(it->second);
        return true;
    }

    void set_last_error(const std::string& error);
    void begin_agent_schedule(const AgentId& id);
    void end_agent_schedule(const AgentId& id);
    void detach_placement_best_effort(const AgentPlacement& placement,
                                      const AgentId& agent_id,
                                      const std::string& reason);
    std::vector<AgentId> lru_idle_agents(const NodeId& on_node = {}) const;
    bool suspend_agent(const AgentId& agent_id);
    std::optional<SlotId> restore_agent_on_node(const AgentPlacement& placement,
                                                const AgentConfig& cfg,
                                                const NodeId& node_id);
    std::optional<SlotId> load_agent_on_node(const AgentConfig& cfg,
                                             const NodeId& node_id);
    static bool response_indicates_capacity_pressure(const std::string& body);
    bool evict_slots_on_node(const NodeId& node_id,
                             const AgentId& preserve_agent,
                             int max_to_evict);
};

} // namespace mm
