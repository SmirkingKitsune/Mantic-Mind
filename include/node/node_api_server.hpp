#pragma once

#include "common/models.hpp"
#include "node/node_service.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <cstdint>
#include <thread>
#include <vector>

namespace mm {

class NodeState;
class SlotManager;
class HttpServer;
class ModelStore;
struct SseInferCtx;

// Hosts the node REST API.
// Most endpoints require "Authorization: Bearer <node-api-key>".
// The /pair-request and /pair-complete endpoints are unauthenticated.
class NodeApiServer {
public:
    // Preferred lifecycle-host constructor: the caller owns the typed service
    // and this class remains a wire/authentication adapter.
    NodeApiServer(NodeService& service,
                  NodeState& state,
                  SlotManager& slot_mgr,
                  std::string control_url = {},
                  std::string pairing_key = {});

    // Compatibility constructor retained for existing embedders.
    NodeApiServer(NodeState& state,
                  SlotManager& slot_mgr,
                  std::string control_url = {},
                  std::string pairing_key = {});
    ~NodeApiServer();

    // Registers all routes and starts listening.  Blocks until stop() called.
    bool listen(uint16_t port);
    void stop();

    using RuntimeLogsProvider = std::function<std::vector<std::string>(int tail)>;
    using RememberApiKeyCallback = std::function<void(const std::string& key)>;
    using LlamaProvisionCallback = std::function<LlamaRuntimeStatus()>;
    // accelerator is empty for the current target, or an explicit release
    // alternative selected from LlamaRuntimeStatus.
    using LlamaUpdateCallback = std::function<LlamaRuntimeStatus(const std::string& accelerator)>;
    using LlamaSwitchCallback = std::function<LlamaRuntimeStatus(const std::string& variant)>;
    using LlamaCheckUpdateCallback = std::function<LlamaRuntimeStatus()>;
    using LlamaDiagnoseCallback = std::function<LlamaRuntimeStatus()>;
    using LlamaRecoveryCallback =
        std::function<LlamaRuntimeStatus(const std::string& action,
                                         const std::string& variant)>;
    void set_runtime_logs_provider(RuntimeLogsProvider provider);
    void set_remember_api_key_callback(RememberApiKeyCallback callback);
    void set_llama_provision_callback(LlamaProvisionCallback callback);
    void set_llama_update_callback(LlamaUpdateCallback callback);
    void set_llama_switch_callback(LlamaSwitchCallback callback);
    void set_llama_check_update_callback(LlamaCheckUpdateCallback callback);
    void set_llama_diagnose_callback(LlamaDiagnoseCallback callback);
    void set_llama_recovery_callback(LlamaRecoveryCallback callback);
    // Local model cache: control-transferred models + LRU eviction. Optional;
    // when unset the model transfer/receive endpoints report unavailable.
    void set_model_store(ModelStore* store);

private:
    NodeState&     state_;
    SlotManager&   slot_mgr_;
    // Typed service boundary shared with the embedded-node transport.  The
    // HTTP server is intentionally limited to authentication, wire parsing,
    // and response/status translation for operations represented here.
    std::unique_ptr<NodeService> owned_service_;
    NodeService&   service_;
    ModelStore*    model_store_ = nullptr;
    std::string    control_url_;
    std::string    pairing_key_;
    std::unique_ptr<HttpServer> server_;
    RememberApiKeyCallback remember_api_key_cb_;

    struct InferenceTask {
        std::thread worker;
        std::shared_ptr<std::atomic<bool>> finished;
        std::weak_ptr<SseInferCtx> context;
    };
    std::atomic<bool> stopping_{false};
    std::mutex inference_tasks_mutex_;
    std::vector<InferenceTask> inference_tasks_;

    void register_routes();
    bool check_auth(const std::string& auth_header);
    bool launch_inference_task(const std::shared_ptr<SseInferCtx>& context,
                               std::function<void()> work);
    void cancel_and_join_inference_tasks();
};

} // namespace mm
