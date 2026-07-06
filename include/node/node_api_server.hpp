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
    void set_runtime_logs_provider(RuntimeLogsProvider provider);
    void set_remember_api_key_callback(RememberApiKeyCallback callback);
    void set_vllm_provision_callback(VllmProvisionCallback callback);
    void set_vllm_update_callback(VllmUpdateCallback callback);
    void set_vllm_check_update_callback(VllmCheckUpdateCallback callback);
    // Ray CLI config for the multi-node engine-group endpoints.
    void set_ray_config(std::string ray_path, uint16_t ray_port);
    // HF model-cache config: the `hf` CLI and the resolved hub cache directory.
    void set_hf_config(std::string hf_cli_path, std::string hf_hub_cache_dir);

private:
    NodeState&     state_;
    SlotManager&   slot_mgr_;
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

    void register_routes();
    bool check_auth(const std::string& auth_header);
};

} // namespace mm
