#pragma once

#include "common/engine_config.hpp"
#include "common/models.hpp"
#include <functional>
#include <memory>
#include <string>
#include <cstdint>
#include <vector>

namespace mm {

class NodeState;
class EngineSupervisor;
class HttpServer;
class ModelStore;
class NodeEngineManager;

// Hosts the node REST API.
// Most endpoints require "Authorization: Bearer <node-api-key>".
// The /pair-request and /pair-complete endpoints are unauthenticated.
class NodeApiServer {
public:
    NodeApiServer(NodeState& state,
                  EngineSupervisor& engines,
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

    // ── Cluster engine configuration ──────────────────────────────────────────
    // The manager answers "what do I run and am I conforming"; the callback is
    // how a pushed config is APPLIED, because applying it also updates NodeState
    // and starts a background provision that the server must not own.
    //
    // Both optional: unset, the engine routes report unavailable rather than
    // half-working. A node that cannot apply a config should say so to the
    // master that pushed it, not accept it and quietly do nothing.
    void set_engine_manager(NodeEngineManager* manager);
    using EngineConfigCallback = std::function<void(const ClusterEngineConfig&)>;
    void set_engine_config_callback(EngineConfigCallback callback);

private:
    NodeState&        state_;
    EngineSupervisor& engines_;
    ModelStore*    model_store_ = nullptr;
    NodeEngineManager* engine_manager_ = nullptr;
    EngineConfigCallback engine_config_cb_;
    std::string    control_url_;
    std::string    pairing_key_;
    std::unique_ptr<HttpServer> server_;
    RuntimeLogsProvider runtime_logs_provider_;
    RememberApiKeyCallback remember_api_key_cb_;
    LlamaProvisionCallback llama_provision_cb_;
    LlamaUpdateCallback llama_update_cb_;
    LlamaSwitchCallback llama_switch_cb_;
    LlamaCheckUpdateCallback llama_check_update_cb_;
    LlamaDiagnoseCallback llama_diagnose_cb_;
    LlamaRecoveryCallback llama_recovery_cb_;

    void register_routes();
    bool check_auth(const std::string& auth_header);
};

} // namespace mm
