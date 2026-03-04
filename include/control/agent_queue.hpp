#pragma once

#include "common/models.hpp"
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <memory>

namespace mm {

struct InferenceJob {
    JobId       job_id;
    AgentId     agent_id;
    std::string user_message;
    ConvId      conversation_id;  // empty = use active conversation

    using ChunkCallback = std::function<void(const InferenceChunk&)>;
    using DoneCallback  = std::function<void(const ConvId&, bool success)>;
    ChunkCallback chunk_cb;
    DoneCallback  done_cb;

    // Full inference pipeline.  If set, worker_loop calls process_fn() and
    // ignores chunk_cb / done_cb directly (process_fn closes them itself).
    std::function<void()> process_fn;
};

// Per-agent FIFO worker queue.  Creates one worker thread per agent on demand.
// Concurrent requests to the same agent are serialised.
class AgentQueue {
public:
    AgentQueue();
    ~AgentQueue();

    // Post a job for an agent.  Returns immediately.
    void enqueue(InferenceJob job);

    // Drain all pending jobs and join all worker threads.
    void shutdown();

private:
    struct AgentWorker {
        std::queue<InferenceJob>  queue;
        std::mutex                mutex;
        std::condition_variable   cv;
        std::thread               thread;
        bool                      stop = false;
    };

    std::mutex mutex_;
    std::unordered_map<AgentId, std::shared_ptr<AgentWorker>> workers_;

    std::shared_ptr<AgentWorker> get_or_create_worker(const AgentId& id);
    static void worker_loop(std::shared_ptr<AgentWorker> w);
};

} // namespace mm
