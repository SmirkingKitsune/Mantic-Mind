#pragma once

#include "common/models.hpp"
#include <functional>
#include <string>
#include <atomic>

namespace mm {

// HTTP client for a running llama-server instance (OpenAI-compatible API).
// All calls block on the calling thread.
// stream_complete() fires chunk_cb for each SSE event.
class LlamaCppClient {
public:
    // base_url: "http://127.0.0.1:8080"
    explicit LlamaCppClient(std::string base_url);
    virtual ~LlamaCppClient() = default;

    // ── Inference ─────────────────────────────────────────────────────────────
    // Non-streaming — waits for the full response.
    virtual Message complete(const InferenceRequest& req);

    // Streaming — chunk_cb fired for each parsed SSE event.
    // is_done=true on the final chunk.  error_cb fires on transport failure.
    using ChunkCallback = std::function<void(const InferenceChunk&)>;
    using ErrorCallback = std::function<void(const std::string&)>;
    void stream_complete(const InferenceRequest& req,
                         ChunkCallback chunk_cb,
                         ErrorCallback error_cb);

    // ── Utility ───────────────────────────────────────────────────────────────
    int  count_tokens(const std::string& text);
    bool load_model(const std::string& model_path, const LlamaSettings& settings);
    bool is_model_loaded() const;
    bool health_check();

private:
    std::string         base_url_;
    std::string         host_;
    int                 port_         = 8080;
    std::atomic<bool>   model_loaded_{false};

    // Parse a single "data: {...}" SSE line into an InferenceChunk.
    // Returns an empty (no-op) chunk on parse failure.
    InferenceChunk parse_sse_line(const std::string& data_json);

    // State-machine extraction of <think>…</think> from a streaming content delta.
    // buf  : persistent buffer across calls (partial tag accumulation)
    // in_think : persistent bool across calls
    // Returns content to emit to delta_content;
    // sets out_thinking to content to emit to thinking_delta.
    static std::string extract_thinking(const std::string& raw,
                                        std::string& buf,
                                        bool& in_think,
                                        std::string& out_thinking);
};

} // namespace mm
