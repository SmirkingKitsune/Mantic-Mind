#pragma once

#include "control/node_operations.hpp"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

namespace mm {

class NodeService;

// Transport-neutral control-to-node adapter used by mantic-mind-aio.  It
// deliberately owns neither NodeService nor its runtime/slot dependencies;
// the AIO lifecycle host must keep those objects alive until control work has
// drained.
class LocalNodeOperations final : public NodeOperations {
public:
    LocalNodeOperations(NodeService& service,
                        std::string models_dir,
                        std::string runtime_network_policy = "prompt");

    bool embedded() const override { return true; }
    std::string endpoint() const override { return {}; }

    NodeOperationResult health() override;
    NodeOperationResult status() override;
    NodeOperationResult logs(int tail) override;
    NodeOperationResult cancel_action() override;

    std::optional<PreparedNodeModel> prepare_model(
        const std::string& source_path,
        const std::string& model_id,
        bool pin,
        bool force,
        std::string* error) override;

    NodeOperationResult load_model(const nlohmann::json& request) override;
    NodeOperationResult unload_model(const nlohmann::json& request) override;
    NodeOperationResult detach_agent(const nlohmann::json& request) override;
    NodeOperationResult suspend_slot(const nlohmann::json& request) override;
    NodeOperationResult restore_slot(const nlohmann::json& request) override;

    NodeOperationResult llama_runtime() override;
    NodeOperationResult llama_provision(const nlohmann::json& request) override;
    NodeOperationResult llama_update(const nlohmann::json& request) override;
    NodeOperationResult llama_check_update(
        const nlohmann::json& request = nlohmann::json::object()) override;
    NodeOperationResult llama_switch(const nlohmann::json& request) override;
    NodeOperationResult llama_diagnose() override;
    NodeOperationResult llama_recover(const nlohmann::json& request) override;

    NodeInferResult infer(
        const NodeInferRequest& request,
        InferenceChunkCallback chunk_cb = {}) override;
    bool stream_infer(const nlohmann::json& request,
                      SseLineCallback line_cb,
                      int* out_status = nullptr,
                      std::string* out_body = nullptr) override;

    NodeOperationResult pair_request(const nlohmann::json& request) override;
    NodeOperationResult pair_complete(const nlohmann::json& request) override;

    const std::string& runtime_network_policy() const {
        return runtime_network_policy_;
    }
    void set_mutation_callback(std::function<void()> callback);
    void request_shutdown() override;

private:
    NodeService& service_;
    std::string models_dir_;
    std::string runtime_network_policy_;
    std::function<void()> mutation_callback_;
    std::atomic<bool> shutdown_requested_{false};
    // Serializes runtime mutations/checks at the adapter boundary.  A queued
    // control request re-checks shutdown only after acquiring this mutex, so
    // it cannot begin after AIO teardown has canceled an earlier action.
    std::mutex runtime_operation_mutex_;

    NodeOperationResult require_network_consent(
        const nlohmann::json* request,
        const char* operation) const;
    NodeOperationResult runtime_canceled() const;
    void notify_mutation() const;
};

} // namespace mm
