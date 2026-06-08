#pragma once

#include "common/models.hpp"
#include "common/llama_cpp_client.hpp"
#include "node/llama_server_process.hpp"

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

/// Manages a pool of concurrent llama-server processes (slots).
/// Each slot is an independent llama-server subprocess with its own port.
class SlotManager {
public:
    using LogCallback = LlamaServerProcess::LogCallback;

    class SlotLease {
    public:
        SlotLease() = default;
        SlotLease(const SlotLease&) = delete;
        SlotLease& operator=(const SlotLease&) = delete;
        SlotLease(SlotLease&& other) noexcept;
        SlotLease& operator=(SlotLease&& other) noexcept;
        ~SlotLease();

        explicit operator bool() const { return client_ != nullptr; }
        LlamaCppClient* get() const { return client_; }
        const SlotId& slot_id() const { return slot_id_; }

    private:
        friend class SlotManager;
        SlotLease(SlotManager* manager, SlotId slot_id, LlamaCppClient* client);

        void reset();

        SlotManager* manager_ = nullptr;
        SlotId slot_id_;
        LlamaCppClient* client_ = nullptr;
    };

    SlotManager(std::string llama_server_path,
                uint16_t port_range_start,
                uint16_t port_range_end,
                int max_slots,
                std::string kv_cache_dir);

    ~SlotManager();

    /// Set a callback invoked for llama-server stdout/stderr output.
    void set_log_callback(LogCallback cb);

    /// Load a model into a new slot, optionally assigning it to an agent.
    /// Returns the slot ID, or empty string on failure.
    /// On start failure, retries with halved ctx_size (min 2048, up to 3 retries).
    SlotId load_model(const std::string& model_path,
                      LlamaSettings settings,
                      const AgentId& agent_id = {});

    /// Unload a slot — stops its llama-server process, frees the port.
    SlotOperationResult unload_slot(const SlotId& slot_id);

    /// Suspend a slot: save KV cache, stop process.
    /// Returns the KV cache file path, or empty string on failure.
    SlotOperationResult suspend_slot(const SlotId& slot_id);

    /// Restore a suspended slot: start a new process, load model, restore KV cache.
    /// Returns the new slot ID, or empty string on failure.
    SlotId restore_slot(const std::string& model_path,
                        const LlamaSettings& settings,
                        const std::string& kv_cache_path,
                        const AgentId& agent_id = {});

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

    /// Last load/restore failure reason for diagnostics.
    std::string last_error() const;

#ifdef MM_TESTING
    /// Test-only helper for exercising lease behavior without launching llama-server.
    SlotId add_ready_test_slot(std::string model_path = "test-model.gguf",
                               AgentId agent_id = {});
#endif

    /// Runtime override for llama-server executable path.
    void        set_llama_server_path(const std::string& path);
    std::string llama_server_path() const;

private:
    struct Slot {
        SlotId                              id;
        uint16_t                            port       = 0;
        std::string                         model_path;
        AgentId                             assigned_agent;
        std::unique_ptr<LlamaServerProcess> process;
        std::unique_ptr<LlamaCppClient>     client;
        int                                 active_requests = 0;
        int64_t                             vram_usage_mb  = 0;
        int64_t                             last_active_ms = 0;
        SlotState                           state          = SlotState::Empty;
        std::string                         kv_cache_path;
        int                                 effective_ctx_size = 0;
    };

    std::string        llama_server_path_;
    uint16_t           port_range_start_;
    uint16_t           port_range_end_;
    int                max_slots_;
    std::string        kv_cache_dir_;
    LogCallback        log_cb_;

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::set<uint16_t>                 used_ports_;
    std::string                        last_error_;

    void release_slot_request(const SlotId& slot_id);

    /// Try to find and claim an available port in the configured range.
    std::optional<uint16_t> allocate_port();
    void release_port(uint16_t port);

    /// Rough memory estimate from file size and effective llama-server context.
    static int64_t estimate_vram_mb(const std::string& model_path,
                                    const LlamaSettings& settings);

    /// Test-bind a TCP socket to check if a port is free.
    static bool test_port_available(uint16_t port);

    SlotInfo make_slot_info(const Slot& s) const;
};

} // namespace mm
