#pragma once

#include "common/models.hpp"
#include <functional>
#include <memory>
#include <string>
#include <cstdint>
#include <vector>

namespace mm {

class NodeState;
class SlotManager;
class HttpServer;
class ModelStore;

// Hosts the node REST API.
// Most endpoints require "Authorization: Bearer <node-api-key>".
// The /pair-request and /pair-complete endpoints are unauthenticated.
class NodeApiServer {
public:
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
    using VllmProvisionCallback = std::function<VllmRuntimeStatus()>;
    // User-approved update: force a managed upgrade to the target version.
    using VllmUpdateCallback = std::function<VllmRuntimeStatus()>;
    // On-demand online probe for a newer build; never installs.
    using VllmCheckUpdateCallback = std::function<VllmRuntimeStatus()>;
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
    void set_vllm_provision_callback(VllmProvisionCallback callback);
    void set_vllm_update_callback(VllmUpdateCallback callback);
    void set_vllm_check_update_callback(VllmCheckUpdateCallback callback);
    void set_llama_provision_callback(LlamaProvisionCallback callback);
    void set_llama_update_callback(LlamaUpdateCallback callback);
    void set_llama_switch_callback(LlamaSwitchCallback callback);
    void set_llama_check_update_callback(LlamaCheckUpdateCallback callback);
    void set_llama_diagnose_callback(LlamaDiagnoseCallback callback);
    void set_llama_recovery_callback(LlamaRecoveryCallback callback);
    // Ray CLI config for the multi-node engine-group endpoints.
    void set_ray_config(std::string ray_path, uint16_t ray_port);
    // HF model-cache config: the `hf` CLI and the resolved hub cache directory.
    void set_hf_config(std::string hf_cli_path, std::string hf_hub_cache_dir);
    // Local model cache: control-transferred models + LRU eviction. Optional;
    // when unset the model transfer/receive endpoints report unavailable.
    void set_model_store(ModelStore* store);

private:
    NodeState&     state_;
    SlotManager&   slot_mgr_;
    ModelStore*    model_store_ = nullptr;
    std::string    control_url_;
    std::string    pairing_key_;
    std::string    ray_path_ = "ray";
    uint16_t       ray_port_ = 6379;
    std::string    hf_cli_path_ = "hf";
    std::string    hf_hub_cache_dir_;
    std::unique_ptr<HttpServer> server_;
    RuntimeLogsProvider runtime_logs_provider_;
    RememberApiKeyCallback remember_api_key_cb_;
    VllmProvisionCallback vllm_provision_cb_;
    VllmUpdateCallback vllm_update_cb_;
    VllmCheckUpdateCallback vllm_check_update_cb_;
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
