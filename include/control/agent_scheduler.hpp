#pragma once

#include "common/footprint.hpp"
#include "common/models.hpp"
#include "soma/routing.hpp"

#include <functional>
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

/// Why a placement did not happen.
///
/// Every value here corresponds to a real `set_failure()` site in
/// agent_scheduler.cpp — the taxonomy was read OUT of the code, not designed for
/// it. A deleted design header proposed a similar enum with values like
/// `Rejected` and `EngineUnavailable` that matched nothing the scheduler
/// actually does (roadmap D46/D63); copying that list would have produced codes
/// no caller could ever observe.
///
/// The gap this closes: placement failure was a bare English string, so
/// "no node is conforming" and "every node is full" — opposite operator actions
/// — differed only by wording, and the only way to tell them apart was to match
/// prose (roadmap D64).
enum class PlacementFailure : std::uint8_t {
    None = 0,            ///< no failure recorded
    EngineConfigMissing, ///< the cluster has no engine policy yet
    NoLocalBackend,      ///< API-backed agent: it owns no node slot by design
    NoEligibleNode,      ///< nothing passed the connected + conforming filter
    NoCapacity,          ///< eligible nodes exist; none could take this model
    ModelTransferFailed, ///< the model could not be put on the target node
    NodeRejected,        ///< the node answered with an HTTP error
    NodeUnreachable,     ///< the request to the node threw
    NodeProtocolError,   ///< the node answered OK with no slot id
};

const char* to_string(PlacementFailure failure) noexcept;

/// Might retrying, with nobody changing anything, plausibly work?
///
/// Deliberately biased toward `true` where it is arguable. A false "retryable"
/// costs a client one wasted poll; a false "not retryable" makes it give up on
/// a placement that would have succeeded. `NoEligibleNode` is the interesting
/// case and it is retryable: a node that was offline or mid-convergence rejoins
/// on its own, with no operator involved.
///
/// Only the two that genuinely require a human are `false` — the cluster has no
/// engine configuration, or the agent is configured to own no slot at all.
bool placement_failure_retryable(PlacementFailure failure) noexcept;

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

    /// WHERE an agent's model lives, as opposed to what the agent calls it.
    ///
    /// `AgentConfig::model_path` is an IDENTITY — it is what the operator typed
    /// and what the registry is keyed on. It is not a location: an admitted
    /// model's container directory carries its quantization
    /// (`<name>-q4_g-q6_g-g128`), so the two strings differ, and handing a node
    /// the identity makes it look for a directory that does not exist.
    ///
    /// Public because it is exactly what a test needs to assert, and because the
    /// distinction is worth being able to state from outside.
    std::string model_location(const AgentConfig& cfg) const;

    /// Optional. Without it every `auto` agent routes to the fallback, since
    /// absence of a record is not evidence of admissibility.
    void set_model_registry(const ControlModelRegistry* registry);

    /// Where an agent landed, and where it stopped being — for the audit trail.
    ///
    /// A CALLBACK PAIR rather than a second registry pointer, for two reasons.
    /// The scheduler holds `ControlModelRegistry` as `const*` because it reads
    /// verdicts and must not be able to mutate the model tables; recording an
    /// audit row is a different concern that happens to live in the same
    /// database. And a callback lets the owner decide what "record" means —
    /// tests substitute a vector.
    ///
    /// Both fire with NO scheduler lock held. `schedule_mutex_` serializes whole
    /// scheduling operations including multi-GB transfers, and a synchronous
    /// SQLite insert has no business inside it; a callback invoked under a lock
    /// is also exactly the shape that killed the node in D56.
    struct PlacementAudit {
        std::function<void(const AgentId&,
                           const NodeId&,
                           const SlotId&,
                           const std::string& backend,
                           const std::string& backend_reason,
                           const ResourceFootprint&)>
            placed;
        std::function<void(const AgentId&)> released;
    };

    void set_placement_audit(PlacementAudit audit);

    /// Gate placement on the cluster having an engine configuration.
    ///
    /// A predicate rather than a stored flag: the configuration arrives while
    /// the scheduler is already running (first-run setup happens after
    /// startup), and a bool captured at construction would keep refusing after
    /// the operator had configured it.
    ///
    /// Unset leaves the gate OPEN, which is what every existing test and any
    /// embedding without a config store needs. Control sets it at startup, so
    /// the gate is closed exactly where a cluster head is present to answer it.
    using EngineConfigReadyFn = std::function<bool()>;
    void set_engine_config_gate(EngineConfigReadyFn ready);

    std::optional<ScheduleResult> ensure_agent_running(const AgentConfig& cfg);
    void release_agent(const AgentId& agent_id);

    /// Suspend an agent's placement: its KV is checkpointed and its slot freed,
    /// but the placement is remembered so a later ensure_agent_running() can
    /// restore it rather than reloading from scratch.
    ///
    /// PUBLIC because P1 says there are no internal-only capabilities, and this
    /// was one. The scheduler has always been able to do it — eviction under
    /// capacity pressure calls it — and the node API has always exposed
    /// /api/node/suspend-slot, but no /v1/* route could reach it, so an operator
    /// with the whole API could not do a thing the scheduler does on its own.
    /// A design header that nothing compiled documented exactly this gap and
    /// called for promoting it. Because it compiled, nothing could fail, so the
    /// note sat there being right for as long as anyone cared to read it. That
    /// header is deleted (roadmap D46); the lesson it cost is that a design
    /// stated where no build can check it is a design that does not exist.
    ///
    /// Returns false when the agent has no live placement to suspend.
    bool suspend_agent(const AgentId& agent_id);
    void mark_agent_idle(const AgentId& agent_id);
    void mark_agent_active(const AgentId& agent_id);
    std::optional<AgentPlacement> get_placement(const AgentId& agent_id) const;
    std::vector<AgentPlacement> list_placements() const;
    std::string last_error() const;

    /// The same failure, as a code. Set with the message and cleared with it, so
    /// the two can never disagree about whether a placement failed.
    PlacementFailure last_failure() const;
    void housekeeping(const std::vector<AgentConfig>& active_agents);

    /// Does this engine refusal mean "no capacity right now", i.e. evict and retry?
    ///
    /// Pure, and public for the same reason `model_location` is: it decides
    /// whether a failed placement gets a second chance, and that decision
    /// deserves to be asserted directly rather than inferred from the outcome of
    /// a placement. Reaching it through `ensure_agent_running` would need a node
    /// that can be made to refuse on demand, which tests the harness more than
    /// the rule.
    ///
    /// A structured code is AUTHORITATIVE — see the definition.
    static bool response_indicates_capacity_pressure(const std::string& body);

private:
    NodeRegistry& registry_;
    const ControlModelRegistry* models_ = nullptr;
    PlacementAudit audit_;
    std::string models_dir_;
    bool engine_config_required_ = false;
    EngineConfigReadyFn engine_config_ready_;

    // Scheduling can include node HTTP calls and large model transfers. Keep it
    // serialized without blocking read-only placement queries.
    std::mutex schedule_mutex_;
    mutable std::mutex state_mutex_;
    std::unordered_map<AgentId, AgentPlacement> placements_;
    std::string last_error_;
    PlacementFailure last_failure_ = PlacementFailure::None;

    /// Audit events queued during a scheduling operation, flushed once the
    /// scheduling mutex is released. See PlacementAudit.
    struct PendingAudit {
        bool placed = false; ///< false = released
        AgentId agent_id;
        NodeId node_id;
        SlotId slot_id;
        std::string backend;
        std::string backend_reason;
        ResourceFootprint footprint;
    };

    void flush_audit(std::vector<PendingAudit>& pending) const;

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

    /// Record a failure and its code together. THE reason they are one call:
    /// two setters would let a caller set one without the other, and a code that
    /// disagrees with its message is worse than no code.
    void set_failure(PlacementFailure failure, const std::string& error);

    /// Clears the code as well. Named for what it is used for — the success
    /// path calls it with {} to reset both.
    void set_last_error(const std::string& error);
    void detach_placement_best_effort(const AgentPlacement& placement,
                                      const AgentId& agent_id,
                                      const std::string& reason);
    std::vector<AgentId> lru_idle_agents(const NodeId& on_node = {}) const;
    std::optional<SlotId> restore_agent_on_node(const AgentPlacement& placement,
                                                const AgentConfig& cfg,
                                                const NodeId& node_id);
    std::optional<SlotId> load_agent_on_node(const AgentConfig& cfg,
                                             const NodeId& node_id);
    bool evict_slots_on_node(const NodeId& node_id,
                             const AgentId& preserve_agent,
                             int max_to_evict);
};

} // namespace mm
