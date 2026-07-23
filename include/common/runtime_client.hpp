#pragma once

#include "common/models.hpp"

#include <atomic>
#include <functional>
#include <string>

namespace mm {

// OpenAI-compatible HTTP client for llama-server and remote API backends.
// All calls block on the calling thread.
class RuntimeClient {
public:
    // base_url: "http://127.0.0.1:8080"
    explicit RuntimeClient(std::string base_url,
                            std::string api_key = {},
                            std::string chat_completions_path = "/v1/chat/completions");
    virtual ~RuntimeClient() = default;

    using CancelCheck = std::function<bool()>;

    // Non-streaming: waits for the full response.
    virtual Message complete(const InferenceRequest& req,
                             CancelCheck cancel_requested = {});

    // Streaming: chunk_cb fires for each parsed SSE event.
    using ChunkCallback = std::function<void(const InferenceChunk&)>;
    using ErrorCallback = std::function<void(const std::string&)>;
    void stream_complete(const InferenceRequest& req,
                         ChunkCallback chunk_cb,
                         ErrorCallback error_cb,
                         CancelCheck cancel_requested = {});

    int  count_tokens(const std::string& text);
    bool load_model(const std::string& model_path, const RuntimeSettings& settings);
    bool is_model_loaded() const;
    bool health_check();

private:
    std::string         base_url_;
    std::string         api_key_;
    std::string         chat_completions_path_;
    std::string         host_;
    int                 port_ = 8080;
    std::atomic<bool>   model_loaded_{false};
};

} // namespace mm
