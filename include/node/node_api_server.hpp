#pragma once

#include "common/models.hpp"
#include <memory>
#include <string>
#include <cstdint>

namespace mm {

class NodeState;
class SlotManager;
class ModelStorage;
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
                  std::string pairing_key = {});
    ~NodeApiServer();

    // Registers all routes and starts listening.  Blocks until stop() called.
    bool listen(uint16_t port);
    void stop();

private:
    NodeState&     state_;
    SlotManager&   slot_mgr_;
    ModelStorage&  model_storage_;
    LlamaRuntimeManager& runtime_mgr_;
    std::string    pairing_key_;
    std::unique_ptr<HttpServer> server_;

    void register_routes();
    bool check_auth(const std::string& auth_header);
};

} // namespace mm
