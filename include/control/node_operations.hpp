#pragma once

#include "common/http_client.hpp"
#include "common/node_inference.hpp"

#include <functional>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace mm {

// Transport-neutral result returned by node operations.  The status values
// intentionally mirror the existing node HTTP contract so callers can retain
// their retry/error behaviour while embedded nodes avoid sockets entirely.
struct NodeOperationResult {
    int status = 0;
    nlohmann::json body = nlohmann::json::object();
    std::string raw_body;

    bool ok() const { return status >= 200 && status < 300; }
    std::string error_message() const;

    static NodeOperationResult from_http(const HttpResponse& response);
    static NodeOperationResult success(nlohmann::json body = nlohmann::json::object(),
                                       int status = 200);
    static NodeOperationResult failure(int status, std::string error);
};

struct PreparedNodeModel {
    std::string load_path;
    std::string model_id;
};

// All control-to-node traffic is expressed through this interface.  Remote
// nodes use HttpNodeOperations; the AIO binary registers a LocalNodeOperations
// implementation backed by NodeService.
class NodeOperations {
public:
    // Returning false from a chunk callback requests cancellation. Incremental
    // callbacks never contain is_done; the returned NodeInferResult is the
    // single typed terminal outcome.
    using InferenceChunkCallback = std::function<bool(const InferenceChunk&)>;
    using SseLineCallback = std::function<bool(const std::string&)>;

    virtual ~NodeOperations() = default;

    virtual bool embedded() const = 0;
    virtual std::string endpoint() const = 0;

    // Abort transport work owned by this adapter. Hosts call this before
    // joining listener/queue workers so a silent remote endpoint cannot hold
    // shutdown until a long HTTP timeout expires.
    virtual void request_shutdown() {}

    virtual NodeOperationResult health() = 0;
    virtual NodeOperationResult status() = 0;
    virtual NodeOperationResult logs(int tail) = 0;
    virtual NodeOperationResult cancel_action() = 0;

    // For an embedded node this validates and returns the canonical source
    // path.  A remote implementation performs the existing manifest-aware
    // cache query/upload and returns the remote load path.
    virtual std::optional<PreparedNodeModel> prepare_model(
        const std::string& source_path,
        const std::string& model_id,
        bool pin,
        bool force,
        std::string* error) = 0;

    virtual NodeOperationResult load_model(const nlohmann::json& request) = 0;
    virtual NodeOperationResult unload_model(const nlohmann::json& request) = 0;
    virtual NodeOperationResult detach_agent(const nlohmann::json& request) = 0;
    virtual NodeOperationResult suspend_slot(const nlohmann::json& request) = 0;
    virtual NodeOperationResult restore_slot(const nlohmann::json& request) = 0;

    virtual NodeOperationResult llama_runtime() = 0;
    virtual NodeOperationResult llama_provision(const nlohmann::json& request) = 0;
    virtual NodeOperationResult llama_update(const nlohmann::json& request) = 0;
    virtual NodeOperationResult llama_check_update(
        const nlohmann::json& request = nlohmann::json::object()) = 0;
    virtual NodeOperationResult llama_switch(const nlohmann::json& request) = 0;
    virtual NodeOperationResult llama_diagnose() = 0;
    virtual NodeOperationResult llama_recover(const nlohmann::json& request) = 0;

    virtual NodeInferResult infer(
        const NodeInferRequest& request,
        InferenceChunkCallback chunk_cb = {}) = 0;

    // Legacy REST/SSE compatibility surface used by node API proxies. Control
    // scheduling and runtime clients use infer() above.
    virtual bool stream_infer(const nlohmann::json& request,
                              SseLineCallback line_cb,
                              int* out_status = nullptr,
                              std::string* out_body = nullptr) = 0;

    // Pairing is deliberately remote-only.  Embedded implementations return
    // HTTP-style 409 failures.
    virtual NodeOperationResult pair_request(const nlohmann::json& request) = 0;
    virtual NodeOperationResult pair_complete(const nlohmann::json& request) = 0;
};

class HttpNodeOperations final : public NodeOperations {
public:
    HttpNodeOperations(std::string base_url, std::string api_key);

    bool embedded() const override { return false; }
    std::string endpoint() const override { return base_url_; }
    void request_shutdown() override;

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

private:
    HttpClient client(int read_timeout_s = 30, int write_timeout_s = 10) const;
    NodeOperationResult get(const std::string& path) const;
    NodeOperationResult post(const std::string& path,
                             const nlohmann::json& request) const;

    std::string base_url_;
    std::string api_key_;
    std::atomic<bool> shutdown_requested_{false};
};

using NodeOperationsPtr = std::shared_ptr<NodeOperations>;

} // namespace mm
