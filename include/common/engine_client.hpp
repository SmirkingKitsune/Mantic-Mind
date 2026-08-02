#pragma once

// Mantic-Mind — HTTP client to an engine subprocess.
//
// REPLACES RuntimeClient.
//
// The single most consequential change is that stream_complete() becomes
// VIRTUAL. It is non-virtual today, which is why control cannot subclass its way
// to a streaming proxy and instead calls
//     node_cli.stream_post("/api/node/infer", ...)
// inline at control_api_server.cpp:1949, with its own retry loop, bypassing the
// abstraction entirely. NodeProxyRuntimeClient exists but can only override
// complete(). Making the streaming path virtual removes the reason for the
// bypass rather than documenting it.
//
// Also moved behind the interface: /tokenize and GET /props, which are
// llama.cpp-specific and were called from SlotManager directly.

#include "common/models.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <string>

namespace mm {

/// Structured refusal from an engine.
///
/// AgentScheduler::response_indicates_capacity_pressure currently substring-
/// matches six English phrases ("max slots reached", "out of memory", …) against
/// the node's error body. A new engine would have to reproduce those literals
/// verbatim to earn an evict-and-retry. Both engines emit codes instead.
struct EngineError {
    std::string code; ///< capacity_pressure | model_not_found | unsupported_content | internal
    std::string message;
    int http_status = 0;

    bool is_capacity_pressure() const;
};

class EngineClient {
public:
    explicit EngineClient(std::string base_url, std::string api_key = {});
    virtual ~EngineClient() = default;

    EngineClient(const EngineClient&) = delete;
    EngineClient& operator=(const EngineClient&) = delete;

    using ChunkCallback = std::function<void(const InferenceChunk&)>;
    using ErrorCallback = std::function<void(const EngineError&)>;

    virtual Message complete(const InferenceRequest& req) = 0;

    /// Virtual — this is the change that matters. Exactly one is_done chunk is
    /// always delivered, on every path including error.
    virtual void stream_complete(const InferenceRequest& req,
                                 ChunkCallback chunk_cb,
                                 ErrorCallback error_cb) = 0;

    virtual bool health_check() = 0;

    /// Optional. Default returns false; llama.cpp implements it via /tokenize.
    virtual bool count_tokens(const std::string& text, int& out_tokens);

    /// Optional capability probe. llama.cpp reads GET /props modalities.vision.
    virtual bool query_capabilities(std::string& out_json);

    const std::string& base_url() const;

protected:
    std::string base_url_;
    std::string api_key_;
};

/// llama-server: OpenAI /v1/chat/completions, /tokenize, /props, /health.
class LlamaEngineClient final : public EngineClient {
public:
    explicit LlamaEngineClient(std::string base_url, std::string api_key = {});

    Message complete(const InferenceRequest& req) override;
    void stream_complete(const InferenceRequest&, ChunkCallback, ErrorCallback) override;
    bool health_check() override;
    bool count_tokens(const std::string& text, int& out_tokens) override;
    bool query_capabilities(std::string& out_json) override;
};

/// Soma: the same OpenAI surface, plus /internal/telemetry and /internal/plan.
///
/// Rejects image content parts with 422 rather than dropping them, and surfaces
/// determinism as a request field.
class SomaEngineClient final : public EngineClient {
public:
    explicit SomaEngineClient(std::string base_url, std::string api_key = {});

    Message complete(const InferenceRequest& req) override;
    void stream_complete(const InferenceRequest&, ChunkCallback, ErrorCallback) override;
    bool health_check() override;

    /// Terse frames: tier occupancy, cache stats, scheduler state. The node
    /// forwards these to control, which re-publishes on /v1/engines/{id}/telemetry.
    using TelemetryCallback = std::function<void(const std::string& frame_json)>;
    void stream_telemetry(TelemetryCallback cb, std::atomic<bool>& stop_flag);

    /// GET /internal/plan — the plan document for the loaded model.
    bool fetch_plan(std::string& out_json);
};

} // namespace mm
