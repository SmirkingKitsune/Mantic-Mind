#include "control/conversation_queue.hpp"

#include "common/logger.hpp"

#include <algorithm>
#include <exception>

namespace mm {

ConversationQueue::ConversationQueue(std::size_t workers) {
    const std::size_t n = std::max<std::size_t>(1, workers);
    threads_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) threads_.emplace_back([this] { worker_loop(); });
}

ConversationQueue::~ConversationQueue() {
    shutdown();
}

bool ConversationQueue::enqueue(const std::string& key, std::function<void()> job,
                                std::function<void()> on_reject) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!stopping_) {
            auto& strand = strands_[key];
            strand.jobs.push_back(std::move(job));
            // A key already running, or already waiting its turn, is picked up
            // again when that finishes; queueing it twice would let two workers
            // run one conversation at once.
            if (!strand.running && !strand.queued) {
                strand.queued = true;
                ready_.push_back(key);
            }
            cv_.notify_one();
            return true;
        }
    }
    if (on_reject) {
        try {
            on_reject();
        } catch (...) {
        }
    }
    return false;
}

void ConversationQueue::worker_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_.wait(lk, [this] { return stopping_ || !ready_.empty(); });
        if (ready_.empty()) {
            if (stopping_) return;
            continue;
        }
        const std::string key = std::move(ready_.front());
        ready_.pop_front();
        auto it = strands_.find(key);
        if (it == strands_.end() || it->second.jobs.empty()) {
            if (it != strands_.end()) strands_.erase(it);
            continue;
        }
        auto job = std::move(it->second.jobs.front());
        it->second.jobs.pop_front();
        it->second.queued = false;
        it->second.running = true;

        lk.unlock();
        try {
            job();
        } catch (const std::exception& e) {
            MM_ERROR("ConversationQueue: job for '{}' failed: {}", key, e.what());
        } catch (...) {
            MM_ERROR("ConversationQueue: job for '{}' failed with an unknown exception", key);
        }
        lk.lock();

        it = strands_.find(key);
        if (it == strands_.end()) continue;
        it->second.running = false;
        if (it->second.jobs.empty()) {
            strands_.erase(it);
        } else {
            // To the BACK: a conversation with a backlog takes turns with the
            // others rather than holding a worker until it is empty.
            it->second.queued = true;
            ready_.push_back(key);
            cv_.notify_one();
        }
    }
}

void ConversationQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopping_ && threads_.empty()) return;
        stopping_ = true;
    }
    cv_.notify_all();
    // Workers keep draining ready_ until it is empty, then exit.
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
}

std::size_t ConversationQueue::pending(const std::string& key) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = strands_.find(key);
    return it == strands_.end() ? 0 : it->second.jobs.size();
}

std::size_t ConversationQueue::active_keys() const {
    std::lock_guard<std::mutex> lk(mu_);
    return strands_.size();
}

} // namespace mm
