#pragma once

#include "common/models.hpp"
#include "common/runtime_client.hpp"
#include "node/runtime_process.hpp"

#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
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
    int remaining_agents = 0;
    bool unloaded = false;

    bool ok() const { return status == SlotOperationStatus::Ok; }
};

/// Manages the node's pool of llama-server processes. Compatible agents share
/// a live process; suspension persists the llama.cpp KV cache and stops it.
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

    SlotManager(uint16_t port_range_start, uint16_t port_range_end, int max_slots);
    ~SlotManager();

    void set_log_callback(LogCallback cb);

    /// Attach to a compatible ready llama.cpp slot, or start a new one.
    SlotId load_model(const std::string& model_path,
                      const std::string& mmproj_path,
                      RuntimeSettings settings,
                      const AgentId& agent_id = {});
    SlotId load_model(const std::string& model_path,
                      RuntimeSettings settings,
                      const AgentId& agent_id = {}) {
        return load_model(model_path, {}, std::move(settings), agent_id);
    }

    SlotOperationResult unload_slot(const SlotId& slot_id);
    DetachResult detach_agent(const SlotId& slot_id, const AgentId& agent_id);

    /// Save the slot's KV cache and stop its llama-server process.
    SlotOperationResult suspend_slot(const SlotId& slot_id);

    /// Start a llama-server and replay a saved KV cache when it is available.
    SlotId restore_slot(const std::string& model_path,
                        const std::string& mmproj_path,
                        const RuntimeSettings& settings,
                        const std::string& kv_cache_path,
                        const AgentId& agent_id = {});
    SlotId restore_slot(const std::string& model_path,
                        const RuntimeSettings& settings,
                        const std::string& kv_cache_path,
                        const AgentId& agent_id = {}) {
        return restore_slot(model_path, {}, settings, kv_cache_path, agent_id);
    }

    SlotOperationResult unload_all(bool force = true);
    // Rejects new loads/leases and cancels pending runtime startups. The host
    // calls this before joining API/queue workers; unload_all() performs the
    // destructive drain later in shutdown.
    void request_shutdown();
    void reset_shutdown();
    std::optional<SlotId> find_slot_by_agent(const AgentId& agent_id) const;
    SlotLease acquire_slot(const SlotId& slot_id);
    bool touch_slot(const SlotId& slot_id);
    std::vector<SlotInfo> get_slot_info() const;
    std::optional<SlotInfo> find_slot(const SlotId& slot_id) const;
    int available_slot_count() const;
    int max_slots() const { return max_slots_; }
    int64_t total_vram_usage() const;
    std::string last_error() const;

#ifdef MM_TESTING
    SlotId add_ready_test_slot(std::string model_path = "model.gguf",
                               AgentId agent_id = {},
                               RuntimeSettings settings = {},
                               std::string mmproj_path = {},
                               std::string runtime_base_url =
                                   "http://127.0.0.1:0");
#endif

    void set_llama_server_path(const std::string& path);
    std::string llama_server_path() const;
    void set_kv_cache_dir(const std::string& dir);
    void set_models_dir(const std::string& dir);

private:
    struct Slot {
        SlotId id;
        uint16_t port = 0;
        std::string model_path;
        std::string mmproj_path;
        std::vector<AgentId> agents;
        RuntimeSettings launch_settings;
        std::string kv_cache_path;
        std::unique_ptr<RuntimeProcess> process;
        std::unique_ptr<RuntimeClient> client;
        int active_requests = 0;
        int64_t vram_usage_mb = 0;
        int64_t last_active_ms = 0;
        SlotState state = SlotState::Empty;
        int effective_ctx_size = 0;
    };

    std::string llama_server_path_ = "llama-server";
    std::string kv_cache_dir_ = "data/kv_cache";
    std::string models_dir_;
    uint16_t port_range_start_;
    uint16_t port_range_end_;
    int max_slots_;
    LogCallback log_cb_;

    mutable std::mutex mutex_;
    std::condition_variable lease_cv_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::set<uint16_t> used_ports_;
    std::string last_error_;
    int pending_loads_ = 0;
    std::atomic<bool> unloading_all_{false};
    std::atomic<bool> shutdown_requested_{false};

    void release_slot_request(const SlotId& slot_id);
    std::optional<SlotId> try_attach(const std::string& model_path,
                                     const std::string& mmproj_path,
                                     const RuntimeSettings& settings,
                                     const AgentId& agent_id);
    void remove_kv_cache_file_locked(const Slot& slot) const;
    static void attach_agent_locked(Slot& slot, const AgentId& agent_id);
    void remove_agent_from_suspended_locked(const AgentId& agent_id);
    std::optional<uint16_t> allocate_port();
    void release_port(uint16_t port);
    static bool test_port_available(uint16_t port);
    SlotInfo make_slot_info(const Slot& slot) const;
};

} // namespace mm
