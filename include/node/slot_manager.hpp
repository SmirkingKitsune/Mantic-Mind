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

/// Manages a pool of concurrent llama-server processes (slots).
/// Each slot is an independent llama-server subprocess with its own port.
class SlotManager {
public:
    using LogCallback = LlamaServerProcess::LogCallback;

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
    bool unload_slot(const SlotId& slot_id);

    /// Suspend a slot: save KV cache, stop process.
    /// Returns the KV cache file path, or empty string on failure.
    std::string suspend_slot(const SlotId& slot_id);

    /// Restore a suspended slot: start a new process, load model, restore KV cache.
    /// Returns the new slot ID, or empty string on failure.
    SlotId restore_slot(const std::string& model_path,
                        const LlamaSettings& settings,
                        const std::string& kv_cache_path,
                        const AgentId& agent_id = {});

    /// Gracefully stop all slots.
    void unload_all();

    /// Find the slot assigned to a given agent.
    std::optional<SlotId> find_slot_by_agent(const AgentId& agent_id) const;

    /// Get the LlamaCppClient for a specific slot (for inference).
    LlamaCppClient* get_client(const SlotId& slot_id);

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

    /// Try to find and claim an available port in the configured range.
    std::optional<uint16_t> allocate_port();
    void release_port(uint16_t port);

    /// Rough VRAM estimate from file size.
    static int64_t estimate_vram_mb(const std::string& model_path);

    /// Test-bind a TCP socket to check if a port is free.
    static bool test_port_available(uint16_t port);

    SlotInfo make_slot_info(const Slot& s) const;
};

} // namespace mm
