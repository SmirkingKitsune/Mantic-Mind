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
class ModelStorage;
class ModelPuller;
class HttpServer;
class LlamaRuntimeManager;

// Hosts the node REST API.
// Most endpoints require "Authorization: Bearer <node-api-key>".
// The /pair-request and /pair-complete endpoints are unauthenticated.
class NodeApiServer {
public:
    NodeApiServer(NodeState& state,
                  SlotManager& slot_mgr,
                  ModelStorage& model_storage,
                  LlamaRuntimeManager& runtime_mgr,
                  std::string control_url = {},
                  std::string pairing_key = {});
    ~NodeApiServer();

    // Registers all routes and starts listening.  Blocks until stop() called.
    bool listen(uint16_t port);
    void stop();

    using RuntimeLogsProvider = std::function<std::vector<std::string>(int tail)>;
    void set_runtime_logs_provider(RuntimeLogsProvider provider);

private:
    NodeState&     state_;
    SlotManager&   slot_mgr_;
    ModelStorage&  model_storage_;
    std::unique_ptr<ModelPuller> model_puller_;
    LlamaRuntimeManager& runtime_mgr_;
    std::string    control_url_;
    std::string    pairing_key_;
    std::unique_ptr<HttpServer> server_;
    RuntimeLogsProvider runtime_logs_provider_;

    void register_routes();
    bool check_auth(const std::string& auth_header);
};

} // namespace mm
