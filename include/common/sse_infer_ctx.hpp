#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>

namespace mm {

// Shared context between an LLM worker thread and an SSE content provider.
// Used by both node_api_server and control_api_server.
struct SseInferCtx {
    std::mutex              mx;
    std::condition_variable cv;
    std::deque<std::string> lines; // pre-formatted "data: ...\n\n" payloads
    bool                    done = false;
};

} // namespace mm
