#pragma once

#include "common/models.hpp"
#include "soma/routing.hpp"

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mm {

class NodeRegistry;
class ControlModelRegistry;

struct ScheduleResult {
    NodeId node_id;
    SlotId slot_id;
};

/// Placement engine. Backend-agnostic and verdict-driven.
///
/// No longer "VRAM-aware scheduler for llama.cpp agents": it resolves WHICH
/// engine serves an agent before placing it, and sends that engine's id to the
/// node instead of the literal "llama-cpp" it used to hardcode at three call
/// sites.
class AgentScheduler {
public:
    AgentScheduler(NodeRegistry& registry, std::string models_dir);

    /// Which local engine serves this agent, and why. PURE — no placement, no
    /// I/O — so `GET /v1/agents/{id}` can show the decision without causing one.
    ///
    /// Returns an empty id for agents that own no local slot at all ("api").
    struct BackendRouting {
        std::string engine_id; ///< "soma" | "llama-cpp" | "" for API-backed
        std::string reason;    ///< one line, suitable for an API field and a log
    };

    /// The pure form. `record` is the admission evidence; a default-constructed
    /// one means "nothing admitted this model", which routes to the fallback.
    static BackendRouting resolve_backend(const AgentConfig& cfg,
                                          const soma::AdmissionRecord& record = {});

    /// The same decision, with the record looked up in the model registry.
    /// Falls back to the pure form with no evidence when no registry is set.
    BackendRouting resolve_backend_for(const AgentConfig& cfg) const;

    /// Optional. Without it every `auto` agent routes to the fallback, since
    /// absence of a record is not evidence of admissibility.
    void set_model_registry(const ControlModelRegistry* registry);

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
    const ControlModelRegistry* models_ = nullptr;
    std::string models_dir_;

    // Scheduling can include node HTTP calls and large model transfers. Keep it
    // serialized without blocking read-only placement queries.
    std::mutex schedule_mutex_;
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
