#pragma once

// Mantic-Mind — the node's pool of engine subprocesses.
//
// REPLACES SlotManager.
//
// The old Slot hardcoded `unique_ptr<RuntimeProcess> process` plus a manager-wide
// `llama_server_path_`, so a node could run exactly one engine kind. An Engine
// here holds an EngineDescriptor* and gets its argv, readiness probe, client,
// KV backend, and capability probe from it.
//
// What is deliberately KEPT from SlotManager, because it was right:
//   * multi-agent sharing of one live process, gated on launch compatibility
//   * RAII request leases (active_requests, so a busy engine is not evicted)
//   * port allocation with a bind probe
//   * suspended engines not counting against capacity
//
// What changes beyond descriptor-driving:
//   * a watchdog — nothing polled the child after Ready, so a crashed engine
//     advertised Ready until a request happened to fail
//   * ResourceFootprint instead of a single vram_usage_mb scalar
//   * per-sequence KV, where the backend supports it

#include "common/models.hpp"
#include "node/engine_descriptor.hpp"
#include "node/engine_process.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mm {

class EngineClient;

enum class EngineOpStatus { Ok, NotFound, Busy, Unsupported, Failed };

struct EngineOpResult {
    EngineOpStatus status = EngineOpStatus::Failed;
    std::string message;
    std::string error_code; ///< structured; see EngineError
    std::string kv_checkpoint_path;

    bool ok() const { return status == EngineOpStatus::Ok; }
};

struct DetachResult {
    EngineOpStatus status = EngineOpStatus::Failed;
    std::string message;
    int remaining_agents = 0;
    bool unloaded = false;

    bool ok() const { return status == EngineOpStatus::Ok; }
};

/// Live per-sequence state, for GET /v1/engines/{id}/slots.
///
/// Nothing like this exists today: the node keeps one active_requests counter
/// per slot and llama-server assigns its own internal slots invisibly. Soma has
/// genuine per-sequence state and it is worth exposing — a stalled sequence and
/// a saturated batch look identical from a request counter.
struct SequenceInfo {
    std::uint32_t index = 0;
    std::string agent_id;
    std::uint32_t position = 0;
    std::uint32_t kv_tokens = 0;
    bool prefilling = false;
    bool suspended = false;
    std::string determinism; ///< batched | strict
};

class EngineSupervisor {
public:
    using LogCallback = EngineProcess::LogCallback;

    /// RAII borrow of a ready engine. Holds off eviction for its lifetime.
    class Lease {
    public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        ~Lease();

        explicit operator bool() const { return client_ != nullptr; }

        EngineClient* get() const { return client_; }

        const SlotId& slot_id() const { return slot_id_; }

    private:
        friend class EngineSupervisor;
        Lease(EngineSupervisor* owner, SlotId slot_id, EngineClient* client);
        void reset();

        EngineSupervisor* owner_ = nullptr;
        SlotId slot_id_;
        EngineClient* client_ = nullptr;
    };

    EngineSupervisor(std::uint16_t port_range_start, std::uint16_t port_range_end, int max_slots);
    ~EngineSupervisor();

    void set_log_callback(LogCallback cb);

    /// Attach to a compatible ready engine, or start a new one.
    ///
    /// `engine_id` selects the descriptor. Unknown ids fail with Unsupported and
    /// a message naming EngineRegistry::ids() — accurate by construction, unlike
    /// the two hand-maintained `supported_backends: ["llama-cpp"]` literals it
    /// replaces.
    SlotId load(const std::string& engine_id,
                const EngineLoadRequest& request,
                const AgentId& agent_id = {});

    EngineOpResult unload(const SlotId& slot_id);
    DetachResult detach_agent(const SlotId& slot_id, const AgentId& agent_id);
    EngineOpResult unload_all(bool force = true);

    /// Persist KV and stop the process.
    ///
    /// Refuses (Unsupported) when the engine has live sequences beyond index 0
    /// and its KV backend reports supports_multi_sequence() == false — today
    /// this silently saves sequence 0 and discards the rest.
    EngineOpResult suspend(const SlotId& slot_id);

    SlotId restore(const std::string& engine_id,
                   const EngineLoadRequest& request,
                   const std::string& kv_checkpoint_path,
                   const AgentId& agent_id = {});

    Lease acquire(const SlotId& slot_id);
    bool touch(const SlotId& slot_id);
    std::optional<SlotId> find_by_agent(const AgentId& agent_id) const;

    std::vector<SlotInfo> slots() const;
    std::optional<SlotInfo> find(const SlotId& slot_id) const;
    std::vector<SequenceInfo> sequences(const SlotId& slot_id) const;

    /// Where an engine is listening, and what it publishes. Empty url when the
    /// slot is unknown or has no live process; empty paths when the engine has
    /// no telemetry surface.
    struct EngineEndpoint {
        std::string base_url;
        std::string telemetry_path;
        std::string heat_path;

        bool ok() const { return !base_url.empty(); }
    };

    EngineEndpoint endpoint(const SlotId& slot_id) const;

    int available_slot_count() const;
    int max_slots() const;
    ResourceFootprint total_footprint() const;
    std::string last_error() const;

    void set_kv_checkpoint_dir(const std::string& dir);
    void set_models_dir(const std::string& dir);

#ifdef MM_TESTING
    /// A Ready engine with no process behind it.
    ///
    /// Test-only, and gated at compile time so it cannot be reached in a shipped
    /// binary. The alternative is spawning a real engine for tests about slot
    /// bookkeeping — leases, detach, suspend refusals — which are the parts that
    /// have nothing to do with inference.
    SlotId add_ready_test_engine(const std::string& engine_id = "llama-cpp",
                                 std::string model_path = "model.gguf",
                                 AgentId agent_id = {},
                                 RuntimeSettings settings = {},
                                 std::string mmproj_path = {});

    /// A Suspended record — one whose process is already stopped.
    ///
    /// Needed because suspend() cannot be used to produce one in a test: a test
    /// engine has no live process, so its KV save fails and the suspend is
    /// correctly refused. Constructing the end state directly is honest; making
    /// suspend() succeed without a checkpoint would not be.
    SlotId add_suspended_test_engine(const std::string& engine_id = "llama-cpp",
                                     std::string model_path = "model.gguf",
                                     AgentId agent_id = {},
                                     RuntimeSettings settings = {});
#endif

    /// Per-engine provisioning/health, keyed by engine id. Generalizes the
    /// single LlamaRuntimeStatus field on NodeState.
    void set_runtime_status(const RuntimeStatus& status);
    std::vector<RuntimeStatus> runtime_statuses() const;
    bool runtime_ready(const std::string& engine_id) const;

private:
    struct Engine {
        SlotId id;
        const EngineDescriptor* descriptor = nullptr;
        std::uint16_t port = 0;

        std::string model_path;
        std::string mmproj_path;
        std::vector<AgentId> agents;
        RuntimeSettings launch_settings;
        std::string kv_checkpoint_path;

        std::unique_ptr<EngineProcess> process;
        std::unique_ptr<EngineClient> client;

        int active_requests = 0;
        ResourceFootprint footprint{};
        std::int64_t last_active_ms = 0;
        SlotState state = SlotState::Empty;
        int effective_ctx_size = 0;
        /// Reported as SlotInfo::backend when no descriptor is registered.
        /// NOT behind #ifdef MM_TESTING on purpose: a conditional member gives
        /// the library and a test-instrumented build different layouts for the
        /// same struct, and one string per engine is a cheap price for not
        /// having that hazard at all.
        std::string fallback_backend_id;
    };

    std::string kv_checkpoint_dir_ = "data/kv_cache";
    std::string models_dir_;
    std::uint16_t port_range_start_;
    std::uint16_t port_range_end_;
    int max_slots_;
    LogCallback log_cb_;

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Engine>> engines_;
    std::set<std::uint16_t> used_ports_;
    std::vector<RuntimeStatus> runtime_statuses_;
    std::string last_error_;
    int pending_loads_ = 0;

    void release_request(const SlotId& slot_id);
    void on_engine_crash(const SlotId& slot_id, int exit_code, const std::string& detail);
    std::optional<SlotId> try_attach(const std::string& engine_id,
                                     const EngineLoadRequest& request,
                                     const AgentId& agent_id);
    std::optional<std::uint16_t> allocate_port();
    void release_port(std::uint16_t port);
    SlotInfo make_slot_info(const Engine& engine) const;
};

} // namespace mm
