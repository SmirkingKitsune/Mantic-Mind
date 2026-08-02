#pragma once

// Mantic-Mind — verdict-driven, backend-agnostic placement.
//
// REPLACES AgentScheduler, whose own doc comment read "VRAM-aware scheduler for
// llama.cpp agents" and whose first act (agent_scheduler.cpp:304) is:
//
//     if (!is_llama_backend(cfg.inference_backend)) { release_agent(...); return nullopt; }
//
// The placement ALGORITHM is preserved almost intact — it is good, and it was
// arrived at by fixing real problems:
//
//     existing placement -> suspended restore -> preferred node -> shared engine
//     -> capacity fit -> evict LRU-idle + retry
//
// What changes is everything around it:
//
//   1. BACKEND SELECTION comes first, from the registry verdict, and is
//      recorded with a reason.
//   2. ResourceFootprint{vram, ram, disk} replaces the single vram_needed
//      scalar, and disk becomes a real constraint.
//   3. Capacity pressure is a structured error code, not six English substrings.
//   4. The two-mutex split is KEPT verbatim: schedule_mutex_ serializes whole
//      scheduling operations including multi-GB transfers, state_mutex_ guards
//      the placement map so GET /v1/placements never blocks behind one. That was
//      right and is why read-only queries stay fast.

#include "common/footprint.hpp"
#include "common/models.hpp"

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
    std::string engine_id;
};

/// Why an agent landed on the engine it did. Surfaced on GET /v1/placements —
/// "which backend" is far less useful than "which backend, and why".
enum class BackendReason : std::uint8_t {
    Verdict = 0,       ///< registry verdict selected it
    NoAdmissionRecord, ///< no record -> fallback
    OperatorOverride,  ///< AgentConfig::backend_override
    VerdictReject,     ///< failed conformance -> fallback
    ResidentOnly,      ///< fits, streaming buys nothing -> fallback
    RemoteApi,         ///< inference_backend == "api"
};

struct BackendDecision {
    std::string engine_id; ///< "soma" | "llama-cpp"
    BackendReason reason = BackendReason::NoAdmissionRecord;
    std::string detail;
    ResourceFootprint footprint{};
    bool from_plan_document = false; ///< true = measured, not estimated
};

/// Structured pressure signal.
///
/// Replaces response_indicates_capacity_pressure()'s substring match over
/// "max slots reached" / "no available ports" / "out of memory" / … — which a
/// new engine would have had to reproduce verbatim to earn an evict-and-retry.
enum class PlacementFailure : std::uint8_t {
    None = 0,
    CapacityPressure,
    NoConnectedNode,
    ModelNotAdmitted,
    ModelTransferFailed,
    EngineUnavailable,
    Rejected,
};

class PlacementEngine {
public:
    PlacementEngine(NodeRegistry& registry, ControlModelRegistry& models, std::string models_dir);

    /// Selects a backend, then places. Backend selection is a separate,
    /// independently testable step — it is pure given (config, registry) and
    /// deserves to be exercised without a node in the loop.
    std::optional<ScheduleResult> ensure_agent_running(const AgentConfig& cfg);

    /// Pure. What ensure_agent_running would choose, and why.
    /// Also serves GET /v1/agents/{id} so a client can see the decision without
    /// causing a placement.
    BackendDecision select_backend(const AgentConfig& cfg) const;

    void release_agent(const AgentId& agent_id);
    void mark_agent_idle(const AgentId& agent_id);
    void mark_agent_active(const AgentId& agent_id);

    std::optional<AgentPlacement> get_placement(const AgentId& agent_id) const;
    std::vector<AgentPlacement> list_placements() const;

    /// Operator-visible suspend/restore.
    ///
    /// Both exist today but only as internal scheduler decisions reachable via
    /// the node API — there is no /v1/* route for either. Promoted here because
    /// P1 says a capability the system has is a capability the API exposes.
    bool suspend_agent(const AgentId& agent_id);
    bool restore_agent(const AgentId& agent_id);

    PlacementFailure last_failure() const;
    std::string last_error() const;

    void housekeeping(const std::vector<AgentConfig>& active_agents);

private:
    NodeRegistry& registry_;
    ControlModelRegistry& models_;
    std::string models_dir_;
    CapacityPolicy policy_{};

    /// Serializes whole scheduling operations, including node HTTP calls and
    /// multi-GB model transfers. Kept from AgentScheduler.
    std::mutex schedule_mutex_;

    /// Guards placements_ / last_error_ only. Read-only queries never block
    /// behind an in-flight transfer. Kept from AgentScheduler.
    mutable std::mutex state_mutex_;

    std::unordered_map<AgentId, AgentPlacement> placements_;
    std::string last_error_;
    PlacementFailure last_failure_ = PlacementFailure::None;

    std::optional<AgentPlacement> find_placement_copy(const AgentId& id) const;
    void store_placement(const AgentPlacement& placement);
    bool erase_placement_entry(const AgentId& id);
    void set_failure(PlacementFailure failure, const std::string& error);

    /// Identity of "the thing currently running", so a config change forces a
    /// reload. Gains `engine_id` and `arch_hash`: the same weights served by two
    /// different engines are not the same engine, and a requantization changes
    /// arch_hash without changing any launch flag.
    std::string engine_fingerprint(const AgentConfig& cfg, const BackendDecision& decision) const;

    std::vector<AgentId> lru_idle_agents(const NodeId& on_node = {}) const;

    std::optional<SlotId>
    load_on_node(const AgentConfig& cfg, const BackendDecision& decision, const NodeId& node_id);
    std::optional<SlotId> restore_on_node(const AgentPlacement& placement,
                                          const AgentConfig& cfg,
                                          const BackendDecision& decision,
                                          const NodeId& node_id);

    /// Now footprint-aware in three dimensions. Reads NodeInfo::disk_free_mb,
    /// which the health poll has always collected and placement has never used.
    std::vector<NodeInfo> candidate_nodes(const ResourceFootprint& footprint) const;

    bool evict_on_node(const NodeId& node_id, const AgentId& preserve_agent, int max_to_evict);
};

const char* to_string(BackendReason reason);
const char* to_string(PlacementFailure failure);

} // namespace mm
