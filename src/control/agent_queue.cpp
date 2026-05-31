#include "control/agent_queue.hpp"
#include "common/logger.hpp"

#include <exception>

namespace mm {

AgentQueue::AgentQueue()  = default;
AgentQueue::~AgentQueue() { shutdown(); }

// ── Worker management ──────────────────────────────────────────────────────────

std::shared_ptr<AgentQueue::AgentWorker>
AgentQueue::get_or_create_worker(const AgentId& id) {
    // Caller must hold mutex_.
    auto it = workers_.find(id);
    if (it != workers_.end()) return it->second;

    auto w = std::make_shared<AgentWorker>();
    w->thread = std::thread(&AgentQueue::worker_loop, w);
    workers_[id] = w;
    return w;
}

void AgentQueue::enqueue(InferenceJob job) {
    std::shared_ptr<AgentWorker> w;
    {
        std::lock_guard<std::mutex> g(mutex_);
        w = get_or_create_worker(job.agent_id);
    }
    {
        std::lock_guard<std::mutex> wg(w->mutex);
        w->queue.push(std::move(job));
    }
    w->cv.notify_one();
}

void AgentQueue::shutdown() {
    std::vector<std::shared_ptr<AgentWorker>> snapshot;
    {
        std::lock_guard<std::mutex> g(mutex_);
        for (auto& [_, w] : workers_) snapshot.push_back(w);
    }

    for (auto& w : snapshot) {
        {
            std::lock_guard<std::mutex> wg(w->mutex);
            w->stop = true;
        }
        w->cv.notify_all();
        if (w->thread.joinable()) w->thread.join();
    }

    std::lock_guard<std::mutex> g(mutex_);
    workers_.clear();
}

// ── Worker loop ────────────────────────────────────────────────────────────────

/*static*/ void AgentQueue::worker_loop(std::shared_ptr<AgentWorker> w) {
    while (true) {
        InferenceJob job;
        {
            std::unique_lock<std::mutex> lk(w->mutex);
            w->cv.wait(lk, [&] { return w->stop || !w->queue.empty(); });
            if (w->stop && w->queue.empty()) break;
            job = std::move(w->queue.front());
            w->queue.pop();
        }

        // process_fn encapsulates the full inference pipeline.
        try {
            if (job.process_fn) {
                job.process_fn();
            } else if (job.done_cb) {
                job.done_cb(job.conversation_id, false);
            }
        } catch (const std::exception& e) {
            MM_ERROR("AgentQueue worker job {} for agent {} failed: {}",
                     job.job_id, job.agent_id, e.what());
            if (job.done_cb) {
                try { job.done_cb(job.conversation_id, false); }
                catch (...) {}
            }
        } catch (...) {
            MM_ERROR("AgentQueue worker job {} for agent {} failed with unknown exception",
                     job.job_id, job.agent_id);
            if (job.done_cb) {
                try { job.done_cb(job.conversation_id, false); }
                catch (...) {}
            }
        }
    }
}

} // namespace mm
