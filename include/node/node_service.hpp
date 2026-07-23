#pragma once

#include "common/node_inference.hpp"
#include "node/slot_manager.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace mm {

class ModelStore;
class NodeState;

struct NodeManagedModelSnapshot {
    std::string id;
    std::string load_path;
    int64_t     size_bytes = 0;
    int64_t     last_used_ms = 0;
    bool        pinned = false;
};

// Typed equivalent of GET /api/node/status. This is deliberately independent
// of JSON and HTTP so an embedded control process can consume it directly.
struct NodeStatusSnapshot {
    NodeId           node_id;
    std::string      hostname;
    bool             registered = false;
    int64_t          last_control_contact_ms = 0;
    NodeHealthMetrics health;
    NodeCapabilities capabilities;
    LlamaRuntimeStatus llama_runtime;
    NodeActionProgress action_progress;
    std::string      last_error;

    std::vector<SlotInfo> slots;
    int              max_slots = 0;
    int              slot_in_use = 0;
    int              slot_available = 0;
    int              slot_ready = 0;
    int              slot_loading = 0;
    int              slot_suspending = 0;
    int              slot_suspended = 0;
    int              slot_error = 0;

    std::string      loaded_model;
    AgentId          active_agent;
    std::string      llama_server_path;
    int64_t          disk_free_mb = 0;

    std::vector<NodeManagedModelSnapshot> managed_models;
    int64_t          model_cache_free_bytes = -1;
};

// Resolve a model already available to the combined process. Unlike the HTTP
// transfer path, this never copies the file into ModelStore. If model_id names
// an existing managed model, its managed load path is preferred and its LRU /
// sticky pin metadata is refreshed.
struct NodePrepareLocalModelRequest {
    std::string model_ref;
    std::string models_dir;
    std::string model_id;
    bool        pin = false;
};

struct NodePreparedLocalModel {
    NodeServiceStatus status = NodeServiceStatus::Failed;
    std::string       error;
    std::string       model_id;
    std::string       load_path;
    int64_t           size_bytes = 0;
    bool              managed = false;
    bool              pin_applied = false;

    bool ok() const { return status == NodeServiceStatus::Ok; }
};

struct NodeLoadModelRequest {
    std::string    model_path;
    std::string    mmproj_path;
    bool           vision_enabled = false;
    RuntimeSettings runtime_settings;
    AgentId        agent_id;
    std::string    backend = "llama-cpp";
    std::string    model_id;
    std::string    mmproj_model_id;
    bool           pin = false;
};

struct NodeRestoreModelRequest : NodeLoadModelRequest {
    std::string kv_cache_path;
};

struct NodeLoadModelResult {
    NodeServiceStatus status = NodeServiceStatus::Failed;
    std::string       error;
    SlotId            slot_id;
    std::string       model_path;
    int               effective_ctx_size = 0;

    bool ok() const { return status == NodeServiceStatus::Ok; }
};

struct NodeUnloadModelRequest {
    // Empty unloads all slots. force=false preserves the public API's busy
    // protection; shutdown callers may opt into force=true explicitly.
    SlotId slot_id;
    bool   force = false;
};

struct NodeDetachAgentRequest {
    SlotId  slot_id;
    AgentId agent_id;
};

struct NodeSuspendSlotRequest {
    SlotId slot_id;
};

struct NodeRuntimeOperationResult {
    NodeServiceStatus status = NodeServiceStatus::Failed;
    std::string       error;
    LlamaRuntimeStatus runtime;

    bool ok() const { return status == NodeServiceStatus::Ok; }
};

// In-process node boundary used by an all-in-one control. All operations are
// synchronous: callers own worker-thread policy, cancellation, and shutdown
// ordering. The service does not own NodeState, SlotManager, or ModelStore.
class NodeService final {
public:
    using InferenceChunkCallback = std::function<void(const InferenceChunk&)>;
    using InferenceErrorCallback = std::function<void(const std::string&)>;
    using RuntimeLogsProvider = std::function<std::vector<std::string>(int tail)>;
    using LlamaProvisionCallback = std::function<LlamaRuntimeStatus()>;
    using LlamaUpdateCallback =
        std::function<LlamaRuntimeStatus(const std::string& accelerator)>;
    using LlamaSwitchCallback =
        std::function<LlamaRuntimeStatus(const std::string& variant)>;
    using LlamaCheckUpdateCallback = std::function<LlamaRuntimeStatus()>;
    using LlamaDiagnoseCallback = std::function<LlamaRuntimeStatus()>;
    using LlamaRecoveryCallback =
        std::function<LlamaRuntimeStatus(const std::string& action,
                                         const std::string& variant)>;

    NodeService(NodeState& state,
                SlotManager& slot_manager,
                ModelStore* model_store = nullptr);

    void set_model_store(ModelStore* model_store);

    NodeStatusSnapshot snapshot() const;
    NodePreparedLocalModel prepare_local_model(
        const NodePrepareLocalModelRequest& request);

    NodeLoadModelResult load_model(const NodeLoadModelRequest& request);
    SlotOperationResult unload_model(const NodeUnloadModelRequest& request = {});
    DetachResult detach_agent(const NodeDetachAgentRequest& request);
    SlotOperationResult suspend_slot(const NodeSuspendSlotRequest& request);
    NodeLoadModelResult restore_slot(const NodeRestoreModelRequest& request);

    NodeInferResult infer(const NodeInferRequest& request,
                          InferenceChunkCallback chunk_callback = {},
                          InferenceErrorCallback error_callback = {});

    void set_runtime_logs_provider(RuntimeLogsProvider provider);
    std::vector<std::string> runtime_logs(int tail) const;
    bool cancel_action();

    void set_llama_provision_callback(LlamaProvisionCallback callback);
    void set_llama_update_callback(LlamaUpdateCallback callback);
    void set_llama_switch_callback(LlamaSwitchCallback callback);
    void set_llama_check_update_callback(LlamaCheckUpdateCallback callback);
    void set_llama_diagnose_callback(LlamaDiagnoseCallback callback);
    void set_llama_recovery_callback(LlamaRecoveryCallback callback);

    NodeRuntimeOperationResult provision_llama_runtime();
    NodeRuntimeOperationResult update_llama_runtime(
        const std::string& accelerator = {});
    NodeRuntimeOperationResult switch_llama_runtime(const std::string& variant);
    NodeRuntimeOperationResult check_llama_runtime_update();
    NodeRuntimeOperationResult diagnose_llama_runtime();
    NodeRuntimeOperationResult recover_llama_runtime(
        const std::string& action,
        const std::string& variant = {});

private:
    NodeState&   state_;
    SlotManager& slot_manager_;

    mutable std::mutex callback_mutex_;
    ModelStore* model_store_ = nullptr;
    RuntimeLogsProvider runtime_logs_provider_;
    LlamaProvisionCallback llama_provision_callback_;
    LlamaUpdateCallback llama_update_callback_;
    LlamaSwitchCallback llama_switch_callback_;
    LlamaCheckUpdateCallback llama_check_update_callback_;
    LlamaDiagnoseCallback llama_diagnose_callback_;
    LlamaRecoveryCallback llama_recovery_callback_;

    void refresh_slot_state();
    std::string managed_load_path(const std::string& model_id,
                                  bool pin,
                                  bool* managed,
                                  bool* pin_applied);
    NodeLoadModelResult validate_load_request(const NodeLoadModelRequest& request,
                                              std::string* model_path,
                                              std::string* mmproj_path);
    void touch_managed_models(const std::string& model_id,
                              const std::string& mmproj_model_id,
                              bool pin);
    NodeRuntimeOperationResult runtime_callback_unavailable(const char* operation) const;
    NodeRuntimeOperationResult apply_runtime_result(LlamaRuntimeStatus status,
                                                    bool conflict_on_error,
                                                    bool fail_on_disabled = true);
    NodeRuntimeOperationResult runtime_callback_exception(
        const char* operation,
        const std::string& detail);
};

} // namespace mm
