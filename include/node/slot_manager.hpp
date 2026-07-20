#pragma once

#include "common/models.hpp"
#include "common/runtime_client.hpp"
#include "node/runtime_process.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mm {

enum class SlotOperationStatus {
    Ok,
    NotFound,
    Busy,
    Failed
};

struct SlotOperationResult {
    SlotOperationStatus status = SlotOperationStatus::Failed;
    std::string message;
    std::string kv_cache_path;

    bool ok() const { return status == SlotOperationStatus::Ok; }
};

struct DetachResult {
    SlotOperationStatus status = SlotOperationStatus::Failed;
    std::string message;
    int  remaining_agents = 0;
    bool unloaded = false;

    bool ok() const { return status == SlotOperationStatus::Ok; }
};

/// Manages a pool of concurrent vLLM engine processes (slots).
/// Each slot is an independent `vllm serve` subprocess with its own port.
/// Engines serving the same model with compatible launch settings are shared
/// by multiple agents; idle engines can sleep (weights to host RAM) and wake.
class SlotManager {
public:
    using LogCallback = RuntimeProcess::LogCallback;

    class SlotLease {
    public:
        SlotLease() = default;
        SlotLease(const SlotLease&) = delete;
        SlotLease& operator=(const SlotLease&) = delete;
        SlotLease(SlotLease&& other) noexcept;
        SlotLease& operator=(SlotLease&& other) noexcept;
        ~SlotLease();

        explicit operator bool() const { return client_ != nullptr; }
        RuntimeClient* get() const { return client_; }
        const SlotId& slot_id() const { return slot_id_; }

    private:
        friend class SlotManager;
        SlotLease(SlotManager* manager, SlotId slot_id, RuntimeClient* client);

        void reset();

        SlotManager* manager_ = nullptr;
        SlotId slot_id_;
        RuntimeClient* client_ = nullptr;
    };

    SlotManager(uint16_t port_range_start,
                uint16_t port_range_end,
                int max_slots,
                std::string vllm_server_path = "vllm",
                double vllm_gpu_budget = 0.90);

    ~SlotManager();

    /// Set a callback invoked for engine stdout/stderr output.
    void set_log_callback(LogCallback cb);

    /// Load a model into a new engine slot, optionally assigning it to an agent.
    /// Returns the slot ID, or empty string on failure. When a ready slot
    /// already serves the same model with compatible launch settings, the agent
    /// is attached to it (or a compatible sleeping engine is woken) and that
    /// slot's ID is returned instead of spawning a new engine.
    SlotId load_model(const std::string& model_path,
                      VllmSettings vllm_settings,
                      const AgentId& agent_id = {},
                      bool vision_enabled = false);

    /// Load a GGUF model into a new llama.cpp engine slot. Mirrors load_model
    /// for the llama-server backend: attaches to a compatible ready llama slot
    /// when one exists, otherwise spawns a fresh llama-server. llama slots do
    /// not draw on the vLLM GPU-fraction budget; their footprint is estimated
    /// from the on-disk GGUF size for node VRAM accounting.
    SlotId load_model_llama(const std::string& model_path,
                            const std::string& mmproj_path,
                            RuntimeSettings settings,
                            const AgentId& agent_id = {});
    SlotId load_model_llama(const std::string& model_path,
                            RuntimeSettings settings,
                            const AgentId& agent_id = {}) {
        return load_model_llama(model_path, {}, std::move(settings), agent_id);
    }

    /// Unload a slot — stops its engine process, frees the port.
    SlotOperationResult unload_slot(const SlotId& slot_id);

    /// Detach an agent from a slot. When the last agent detaches and the slot
    /// has no in-flight requests, the slot is unloaded as well.
    DetachResult detach_agent(const SlotId& slot_id, const AgentId& agent_id);

    /// Suspend a slot via vLLM sleep mode: /sleep level=1 offloads weights to
    /// host RAM, frees the GPU, and keeps the process alive for a fast wake.
    /// Falls back to stopping the process when the sleep request fails.
    SlotOperationResult suspend_slot(const SlotId& slot_id);

    /// Restore a suspended agent: attaches to a compatible running engine or
    /// wakes a compatible sleeping one; otherwise starts a fresh engine.
    /// Returns the slot ID, or empty string on failure.
    SlotId restore_slot(const std::string& model_path,
                        const VllmSettings& vllm_settings,
                        const AgentId& agent_id = {},
                        bool vision_enabled = false);

    /// Restore a suspended llama.cpp agent. Starts a fresh llama-server (with
    /// the KV-cache slot endpoints enabled) and, when kv_cache_path names an
    /// existing saved cache, replays it via /slots/0?action=restore so context
    /// survives the suspend. A missing/failed cache degrades to a cold start.
    SlotId restore_slot_llama(const std::string& model_path,
                              const std::string& mmproj_path,
                              const RuntimeSettings& settings,
                              const std::string& kv_cache_path,
                              const AgentId& agent_id = {});
    SlotId restore_slot_llama(const std::string& model_path,
                              const RuntimeSettings& settings,
                              const std::string& kv_cache_path,
                              const AgentId& agent_id = {}) {
        return restore_slot_llama(model_path, {}, settings, kv_cache_path, agent_id);
    }

    /// Gracefully stop all slots. force=true is intended for shutdown cleanup.
    SlotOperationResult unload_all(bool force = true);

    /// Find the slot assigned to a given agent.
    std::optional<SlotId> find_slot_by_agent(const AgentId& agent_id) const;

    /// Acquire a temporary inference lease for a ready slot.
    SlotLease acquire_slot(const SlotId& slot_id);

    /// Update last-active timestamp for a slot when it is selected for work.
    bool touch_slot(const SlotId& slot_id);

    /// Get info for all slots.
    std::vector<SlotInfo> get_slot_info() const;

    /// Get info for a specific slot.
    std::optional<SlotInfo> find_slot(const SlotId& slot_id) const;

    /// Number of slots available (max_slots - active_slots).
    int available_slot_count() const;
    int max_slots() const { return max_slots_; }

    /// Sum of estimated VRAM usage across all active slots.
    int64_t total_vram_usage() const;

    /// Total GPU memory fraction vLLM slots on this node may claim together.
    double vllm_gpu_budget() const;

    /// GPU memory fraction currently claimed by active vLLM slots, including
    /// loads in flight.
    double vllm_gpu_fraction_used() const;

    /// Total GPU VRAM reported by node metrics; used to convert vLLM
    /// utilization fractions into MB estimates.
    void set_gpu_vram_total_mb(int64_t mb);

    /// Scrape /metrics from every ready vLLM engine and cache the load
    /// signals on the slots. Intended for a periodic background caller; the
    /// HTTP requests run without holding the slot mutex.
    void refresh_vllm_metrics();

    /// Last load/restore failure reason for diagnostics.
    std::string last_error() const;

#ifdef MM_TESTING
    /// Test-only helper for exercising lease behavior without launching an engine.
    SlotId add_ready_test_slot(std::string model_path = "Qwen/Qwen3-8B",
                               AgentId agent_id = {},
                               double gpu_mem_fraction = 0.0);

    /// Test-only: put a slot into the sleeping-suspended state.
    bool mark_test_slot_sleeping(const SlotId& slot_id);

    /// Test-only: add a ready llama.cpp slot (backend=LlamaCpp) without spawning
    /// llama-server, for exercising backend-tagged slot info and suspension.
    SlotId add_ready_test_slot_llama(std::string model_path = "model.gguf",
                                     AgentId agent_id = {},
                                     RuntimeSettings settings = {},
                                     std::string mmproj_path = {});
#endif

    void        set_vllm_server_path(const std::string& path);
    std::string vllm_server_path() const;

    /// llama.cpp engine executable (llama-server) and where KV-cache slot saves
    /// land. models_dir lets the VRAM estimate resolve a transferred GGUF by
    /// basename when the model_path is not directly present. Set once at startup.
    void        set_llama_server_path(const std::string& path);
    std::string llama_server_path() const;
    void        set_kv_cache_dir(const std::string& dir);
    void        set_models_dir(const std::string& dir);

private:
    struct Slot {
        SlotId                              id;
        uint16_t                            port       = 0;
        std::string                         model_path;
        std::string                         mmproj_path;
        bool                                vision_enabled = false;
        // Every creation path stamps this explicitly; the default matches the
        // branch default runtime.
        EngineBackend                       backend    = EngineBackend::LlamaCpp;
        std::vector<AgentId>                agents;          // attached agents
        VllmSettings                        launch_settings; // vLLM engine identity for sharing
        RuntimeSettings                     llama_launch_settings; // llama.cpp engine identity
        std::string                         kv_cache_path;   // llama.cpp: saved KV on suspend
        std::unique_ptr<RuntimeProcess> process;
        std::unique_ptr<RuntimeClient>     client;
        int                                 active_requests = 0;
        int64_t                             vram_usage_mb  = 0;
        double                              gpu_mem_fraction = 0.0;
        int64_t                             last_active_ms = 0;
        SlotState                           state          = SlotState::Empty;
        bool                                sleeping       = false; // sleep mode: process alive, GPU freed
        // vLLM engine load from the last /metrics scrape.
        bool                                engine_metrics_valid = false;
        int                                 num_requests_running = 0;
        int                                 num_requests_waiting = 0;
        double                              kv_cache_usage       = 0.0;
        int                                 effective_ctx_size = 0;
    };

    std::string        vllm_server_path_;
    std::string        llama_server_path_ = "llama-server";
    std::string        kv_cache_dir_      = "data/kv_cache";
    std::string        models_dir_;
    uint16_t           port_range_start_;
    uint16_t           port_range_end_;
    int                max_slots_;
    double             vllm_gpu_budget_;
    LogCallback        log_cb_;

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::set<uint16_t>                 used_ports_;
    std::string                        last_error_;
    // Loads in flight (process starting outside the lock). Counted against
    // max_slots_ so concurrent loads cannot oversubscribe the node.
    int                                pending_loads_ = 0;
    // GPU fraction reserved by vLLM loads in flight, counted against
    // vllm_gpu_budget_ so concurrent loads cannot oversubscribe the GPU.
    double                             pending_vllm_fraction_ = 0.0;
    int64_t                            gpu_vram_total_mb_ = 0;

    void release_slot_request(const SlotId& slot_id);

    /// GPU fraction held by non-suspended vLLM slots plus loads in flight.
    /// Caller must hold mutex_.
    double vllm_fraction_allocated_locked() const;

    /// Reserve a GPU fraction for a vLLM load. Returns the granted slice
    /// (clamped to the remaining budget), or nullopt when the budget is
    /// exhausted (sets last_error_). Caller must hold mutex_.
    std::optional<double> reserve_vllm_fraction_locked(double requested);

    /// Find a ready vLLM slot serving the same model with compatible launch
    /// settings, or nullptr. Caller must hold mutex_.
    Slot* find_compatible_vllm_slot_locked(const std::string& model_path,
                                           const VllmSettings& settings);

    /// Attach an agent to a ready compatible llama.cpp slot if one exists.
    /// Returns the slot ID on success, nullopt to spawn a fresh engine. Unlike
    /// vLLM there is no sleeping llama process to wake — suspend stops it — so
    /// this only shares live engines. Takes mutex_ internally.
    std::optional<SlotId> try_attach_llama(const std::string& model_path,
                                           const std::string& mmproj_path,
                                           const RuntimeSettings& settings,
                                           const AgentId& agent_id);

    /// Delete a slot's saved KV-cache file, if any. Caller must hold mutex_.
    void remove_kv_cache_file_locked(const Slot& slot) const;

    /// Attach to a ready compatible engine, or wake a sleeping one and attach.
    /// Returns the slot ID on success; nullopt means the caller should proceed
    /// with a fresh engine start. Takes mutex_ internally (the wake HTTP call
    /// runs unlocked).
    std::optional<SlotId> try_attach_or_wake(const std::string& model_path,
                                             const VllmSettings& settings,
                                             const AgentId& agent_id,
                                             bool vision_enabled);

    /// Attach an agent to a slot if not already attached. Caller must hold mutex_.
    static void attach_agent_locked(Slot& slot, const AgentId& agent_id);

    /// Drop an agent from suspended slot records; erase records left with no
    /// agents. Called when the agent comes back up via attach or restore.
    /// Caller must hold mutex_.
    void remove_agent_from_suspended_locked(const AgentId& agent_id);

    /// Try to find and claim an available port in the configured range.
    std::optional<uint16_t> allocate_port();
    void release_port(uint16_t port);

    /// Test-bind a TCP socket to check if a port is free.
    static bool test_port_available(uint16_t port);

    SlotInfo make_slot_info(const Slot& s) const;
};

} // namespace mm
