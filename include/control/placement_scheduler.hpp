#pragma once

// Mantic-Mind — where served models run.
//
// Places a HOLDER — a served model — on a node, and keeps every request that is
// using that placement from having it evicted underneath it. Placements are
// keyed by holder, so every user who picks a served model shares its one
// placement; this used to be keyed by agent, which made the same model loaded
// for two agents two separate things to schedule.
//
// ── Leases, not flags ─────────────────────────────────────────────────────────
//
// A request CLAIMS a placement with acquire() and holds a Lease until it is
// done. The claim is taken while the schedule lock is still held, so there is no
// window between "placed" and "in use" in which eviction can take it — the old
// ensure-then-mark_active pair had exactly that window. The count is a count
// because many users share one placement: "busy" must stay true until the last
// of them finishes.
//
// ── Per-call results ──────────────────────────────────────────────────────────
//
// Every placement call returns its own failure code and message. The old
// scheduler kept ONE last_error for the whole process and callers read it after
// the call returned, so two concurrent placements could report each other's
// failures — rare with one agent per model, routine with many users per model.

#include "common/footprint.hpp"
#include "common/models.hpp"
#include "common/served_model.hpp"
#include "soma/routing.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mm {

class NodeRegistry;
class ControlModelRegistry;

/// Why a placement did not happen.
///
/// Every value here corresponds to a real failure site in
/// placement_scheduler.cpp — the taxonomy was read OUT of the code, not designed
/// for it. A deleted design header proposed a similar enum with values like
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
    NoLocalBackend,      ///< API-backed model: it owns no node slot by design
    NoEligibleNode,      ///< nothing passed the connected + conforming filter
    NoCapacity,          ///< eligible nodes exist; none could take this model
    ModelTransferFailed, ///< the model could not be put on the target node
    NodeRejected,        ///< the node answered with an HTTP error
    NodeUnreachable,     ///< the request to the node threw
    NodeProtocolError,   ///< the node answered OK with no slot id
    /// The model's launch settings changed, and its current placement cannot be
    /// replaced while requests are still using it. Clears on its own once they
    /// finish — the old code detached the engine out from under them instead.
    Draining,
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
/// engine configuration, or the model is configured to own no slot at all.
bool placement_failure_retryable(PlacementFailure failure) noexcept;

/// Is this worth WAITING on, a second at a time, inside one request? Narrower
/// than retryable: capacity freeing up, a node rejoining, or a placement
/// draining are things that resolve in seconds. A node that rejected the load
/// is not going to change its answer while the user watches.
bool placement_failure_transient(PlacementFailure failure) noexcept;

class PlacementScheduler {
public:
    PlacementScheduler(NodeRegistry& registry, std::string models_dir);

    /// Which local engine serves this model, and why. PURE — no placement, no
    /// I/O — so `GET /v1/served-models/{id}` can show the decision without
    /// causing one.
    ///
    /// Returns an empty id for models that own no local slot at all ("api").
    struct BackendRouting {
        std::string engine_id; ///< "soma" | "llama-cpp" | "vllm" | "" for API-backed
        std::string reason;    ///< one line, suitable for an API field and a log
    };

    /// The pure form. `record` is the admission evidence; a default-constructed
    /// one means "nothing admitted this model", which routes to the fallback.
    static BackendRouting resolve_backend(const ServedModel& model,
                                          const soma::AdmissionRecord& record = {});

    /// The same decision, with the record looked up in the model registry.
    /// Falls back to the pure form with no evidence when no registry is set.
    BackendRouting resolve_backend_for(const ServedModel& model) const;

    /// WHERE a model's bytes are, as opposed to what the model is called.
    ///
    /// `ServedModel::model_path` is an IDENTITY — it is what the operator typed
    /// and what the registry is keyed on. It is not a location: an admitted
    /// model's container directory carries its quantization
    /// (`<name>-q4_g-q6_g-g128`), so the two strings differ, and handing a node
    /// the identity makes it look for a directory that does not exist.
    std::string model_location(const ServedModel& model) const;

    /// What a Soma-routed model actually costs the host it lands on: RAM and
    /// disk, no VRAM (Soma v1 is CPU-only; roadmap D62).
    ResourceFootprint soma_footprint(const ServedModel& model) const;

    /// What `model` costs a SPECIFIC node — the same footprint, plus the disk
    /// the container needs only if that node does not already hold it.
    ResourceFootprint footprint_for_node(const ServedModel& model,
                                         const ResourceFootprint& base,
                                         const NodeInfo& node) const;

    /// The cache id the node would store this model under, or "" when control
    /// holds no local bytes for it.
    std::string model_cache_id(const ServedModel& model) const;

    /// The model name to put in an inference request to `engine_id`.
    ///
    /// ONE rule, stated once. vLLM serves under the name it was launched with
    /// (`engine_model_name` when set); every other engine is sent model_path.
    /// This was re-derived at three call sites and missing at three others,
    /// which sent vLLM a name it did not know for compaction and memory calls.
    static std::string request_model_name(const ServedModel& model, const std::string& engine_id);

    /// Optional. Without it every `auto` model routes to the fallback, since
    /// absence of a record is not evidence of admissibility.
    void set_model_registry(const ControlModelRegistry* registry);

    /// Where a holder landed, and where it stopped being — for the audit trail.
    ///
    /// A CALLBACK PAIR rather than a second registry pointer: the scheduler
    /// holds ControlModelRegistry as `const*` because it reads verdicts and must
    /// not be able to mutate the model tables, and a callback lets the owner
    /// decide what "record" means — tests substitute a vector.
    ///
    /// Both fire with NO scheduler lock held. The schedule lock serializes whole
    /// placements including multi-GB transfers, and a callback invoked under a
    /// lock is exactly the shape that killed the node in D56.
    struct PlacementAudit {
        std::function<void(const HolderId&,
                           const NodeId&,
                           const SlotId&,
                           const std::string& backend,
                           const std::string& backend_reason,
                           const ResourceFootprint&)>
            placed;
        std::function<void(const HolderId&)> released;
    };

    void set_placement_audit(PlacementAudit audit);

    /// Gate placement on the cluster having an engine configuration. A
    /// predicate rather than a stored flag, because the configuration arrives
    /// while the scheduler is already running. Unset leaves the gate OPEN.
    using EngineConfigReadyFn = std::function<bool()>;
    void set_engine_config_gate(EngineConfigReadyFn ready);

    /// Supplies the live cluster engine order and the launch profile applied to
    /// each candidate.
    using EngineConfigProvider = std::function<std::optional<ClusterEngineConfig>()>;
    void set_engine_config_provider(EngineConfigProvider provider);

    /// One request's claim on a placement. Move-only; releasing it (explicitly
    /// or by destruction) gives the claim back. The scheduler must outlive it.
    class Lease {
    public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        ~Lease();

        explicit operator bool() const noexcept { return owner_ != nullptr; }

        const HolderId& holder_id() const noexcept { return holder_id_; }
        const NodeId& node_id() const noexcept { return node_id_; }
        const SlotId& slot_id() const noexcept { return slot_id_; }
        const std::string& engine_id() const noexcept { return engine_id_; }
        /// What to put in the inference request's `model` field for this
        /// placement's engine — see request_model_name().
        const std::string& request_model() const noexcept { return request_model_; }

        /// Give the claim back now rather than at destruction. Idempotent.
        void release() noexcept;

    private:
        friend class PlacementScheduler;
        PlacementScheduler* owner_ = nullptr;
        HolderId holder_id_;
        std::uint64_t epoch_ = 0;
        NodeId node_id_;
        SlotId slot_id_;
        std::string engine_id_;
        std::string request_model_;
    };

    struct Outcome {
        std::optional<Lease> lease; ///< acquire() only, on success
        NodeId node_id;             ///< on success
        SlotId slot_id;             ///< on success
        std::string engine_id;      ///< on success
        PlacementFailure failure = PlacementFailure::None;
        std::string error;

        bool ok() const noexcept { return failure == PlacementFailure::None && !slot_id.empty(); }
    };

    /// Place `model` (or reuse its placement) and CLAIM it for one request.
    ///
    /// `wait` > 0 retries TRANSIENT failures (placement_failure_transient) once
    /// a second until it elapses, then reports the last one.
    Outcome acquire(const ServedModel& model,
                    std::chrono::milliseconds wait = std::chrono::milliseconds{0});

    /// Place (or reuse) without claiming. The explicit restore route.
    Outcome ensure_running(const ServedModel& model);

    enum class SuspendResult : std::uint8_t { Suspended, NotPlaced, Busy, Failed };

    /// Checkpoint a placement's KV and free its slot, remembering the placement
    /// so a later acquisition restores rather than reloads. `Busy` while any
    /// request — on this holder or any holder sharing its slot — is in flight.
    SuspendResult suspend(const HolderId& holder_id);

    enum class ReleaseResult : std::uint8_t { Released, NotPlaced, Busy };

    /// Forget a placement and detach it from its node. `Busy` while requests
    /// are in flight, unless `force`, which the operator can ask for.
    ReleaseResult release(const HolderId& holder_id, bool force = false);

    std::optional<Placement> get_placement(const HolderId& holder_id) const;
    std::vector<Placement> list_placements() const;

    /// `live` is every holder that still exists. A placement whose holder is
    /// gone is released — unless requests are still using it — and surviving
    /// Ray groups are reconciled against `live`.
    void housekeeping(const std::vector<ServedModel>& live);

    /// Does this engine refusal mean "no capacity right now", i.e. evict and
    /// retry? A structured code is AUTHORITATIVE — see the definition.
    static bool response_indicates_capacity_pressure(const std::string& body);

private:
    /// One placement call's failure, carried by reference through the helpers.
    /// Local to the call on purpose: see "Per-call results" above.
    struct Attempt {
        PlacementFailure failure = PlacementFailure::None;
        std::string error;

        void fail(PlacementFailure f, std::string message) {
            failure = f;
            error = std::move(message);
        }
    };

    struct Claimed {
        NodeId node_id;
        SlotId slot_id;
        std::string engine_id;
        std::uint64_t epoch = 0;
    };

    NodeRegistry& registry_;
    const ControlModelRegistry* models_ = nullptr;
    PlacementAudit audit_;
    std::string models_dir_;
    bool engine_config_required_ = false;
    EngineConfigReadyFn engine_config_ready_;
    EngineConfigProvider engine_config_provider_;

    // Scheduling can include node HTTP calls and large model transfers. Keep it
    // serialized without blocking read-only placement queries.
    std::mutex schedule_mutex_;
    mutable std::mutex state_mutex_;
    std::unordered_map<HolderId, Placement> placements_;
    std::uint64_t next_epoch_ = 1;

    /// Audit events queued during a scheduling operation, flushed once the
    /// scheduling mutex is released. See PlacementAudit.
    struct PendingAudit {
        bool placed = false; ///< false = released
        HolderId holder_id;
        NodeId node_id;
        SlotId slot_id;
        std::string backend;
        std::string backend_reason;
        ResourceFootprint footprint;
    };

    void flush_audit(std::vector<PendingAudit>& pending) const;

    std::optional<Claimed> place(const ServedModel& model, bool claim, Attempt& attempt);
    void release_claim(const HolderId& holder_id, std::uint64_t epoch) noexcept;
    /// suspend() for callers already holding the schedule lock — eviction,
    /// which runs inside place().
    SuspendResult suspend_locked(const HolderId& holder_id);

    std::optional<Placement> find_placement_copy(const HolderId& id) const;
    /// Store a placement that REPLACES whatever was there: a fresh epoch.
    std::uint64_t store_new_placement(Placement placement);
    /// Update a placement in place: same epoch.
    void store_placement(const Placement& placement);
    bool erase_placement_entry(const HolderId& id);

    template <typename Fn>
    bool mutate_placement(const HolderId& id, Fn&& fn) {
        std::lock_guard<std::mutex> guard(state_mutex_);
        const auto it = placements_.find(id);
        if (it == placements_.end()) return false;
        fn(it->second);
        return true;
    }

    void detach_placement_best_effort(const Placement& placement,
                                      const HolderId& holder_id,
                                      const std::string& reason);
    std::vector<BackendRouting> routing_candidates(const ServedModel& model) const;
    void stop_ray_group_best_effort(const Placement& placement, const std::string& reason) const;
    std::vector<HolderId> lru_idle_holders(const NodeId& on_node = {}) const;
    std::optional<SlotId> restore_on_node(const Placement& placement,
                                          const ServedModel& model,
                                          const NodeId& node_id,
                                          const BackendRouting& routing,
                                          const std::optional<VllmEngineConfig>& vllm,
                                          Attempt& attempt);
    std::optional<SlotId> load_on_node(const ServedModel& model,
                                       const NodeId& node_id,
                                       const BackendRouting& routing,
                                       const std::optional<VllmEngineConfig>& vllm,
                                       Attempt& attempt,
                                       const std::string& ray_address = {},
                                       const std::string& host_ip = {},
                                       const std::vector<int>& gpu_devices = {});
    bool evict_slots_on_node(const NodeId& node_id,
                             const HolderId& preserve_holder,
                             int max_to_evict);
    bool evict_holder_for_capacity(const HolderId& holder_id);
};


} // namespace mm
