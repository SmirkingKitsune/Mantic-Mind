#pragma once

// Mantic-Mind — work ordered per key, run on a bounded pool.
//
// Replaces the per-agent queue, which gave every agent its own thread for the
// life of the process and serialized everything that agent did. With users,
// ordering is a property of a CONVERSATION — two turns of one conversation must
// not overlap, since each is built on the one before — while a user's other
// conversations, and everyone else's, can run at the same time. Keying a thread
// per conversation would be unbounded; this keeps one STRAND per key and runs
// strands on a fixed pool.
//
// ── The shape ─────────────────────────────────────────────────────────────────
//
//   * Jobs with the same key run one at a time, in submission order.
//   * Jobs with different keys run in parallel, up to the pool size.
//   * A key with nothing left to run holds no state: its strand is erased.
//   * After a key's job finishes, the key goes to the BACK of the ready queue,
//     so one busy conversation cannot starve the others.
//
// shutdown() stops intake, runs what is already queued, and joins. A job
// submitted after that is refused through its `on_reject`, never silently lost.

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mm {

class ConversationQueue {
public:
    /// `workers` is clamped to at least 1.
    explicit ConversationQueue(std::size_t workers);
    ~ConversationQueue();

    ConversationQueue(const ConversationQueue&) = delete;
    ConversationQueue& operator=(const ConversationQueue&) = delete;

    /// Queue `job` behind every earlier job with the same `key`. Returns false —
    /// after calling `on_reject`, if given — once shutdown has begun. A job that
    /// throws is logged and does not take its worker or its strand down with it.
    bool enqueue(const std::string& key, std::function<void()> job,
                 std::function<void()> on_reject = {});

    /// Stop intake, run everything already queued, join the workers. Idempotent.
    void shutdown();

    /// Jobs queued (not yet started) under `key`.
    std::size_t pending(const std::string& key) const;
    /// Keys with work queued or running.
    std::size_t active_keys() const;
    std::size_t workers() const noexcept { return threads_.size(); }

private:
    struct Strand {
        std::deque<std::function<void()>> jobs;
        bool running = false; ///< a worker is executing this key's job
        bool queued = false;  ///< the key is in ready_
    };

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<std::string, Strand> strands_;
    std::deque<std::string> ready_;
    std::vector<std::thread> threads_;
    bool stopping_ = false;

    void worker_loop();
};

} // namespace mm
