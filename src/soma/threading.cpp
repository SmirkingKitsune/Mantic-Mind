// Soma — thread pool implementation.
//
// Spin-then-sleep, because the access pattern is bursty and both extremes are
// wrong on their own: pure spinning burns sixteen cores while the engine waits
// on a socket, and pure condition-variable sleeping pays ~10 us of wake latency
// on regions that run for 30 us. Workers spin briefly after a region ends —
// long enough to catch the next one in a tight sequence of layers — then sleep.

#include "soma/threading.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace soma {

namespace {

/// One spin iteration's worth of backing off.
inline void cpu_relax() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

/// Set on pool workers AND on the calling thread for the duration of a region,
/// so the nesting check catches both. The caller participates in its own
/// parallel_for, and an inner parallel_for reached from the caller's chunk must
/// flatten exactly as one reached from a worker's chunk.
thread_local bool t_in_region = false;

std::uint32_t desired_threads() {
    // Env override, per the repo convention that any config key can be set from
    // the environment. Also the escape hatch for pinning a benchmark to one core.
#if defined(_MSC_VER)
    char* buf = nullptr;
    std::size_t len = 0;
    const bool have = (_dupenv_s(&buf, &len, "SOMA_THREADS") == 0 && buf != nullptr);
    const std::string v = have ? std::string(buf) : std::string();
    std::free(buf);
#else
    const char* raw = std::getenv("SOMA_THREADS");
    const std::string v = raw ? std::string(raw) : std::string();
#endif
    if (!v.empty()) {
        const long n = std::strtol(v.c_str(), nullptr, 10);
        if (n > 0) return static_cast<std::uint32_t>(std::min<long>(n, 256));
    }
    const auto hw = std::thread::hardware_concurrency();
    return std::clamp<std::uint32_t>(hw ? hw : 1u, 1u, 256u);
}

} // namespace

struct ThreadPool::Impl {
    std::uint32_t n_threads = 1;
    std::vector<std::thread> workers;

    // The current region. Published under `mu`, read after observing `epoch`.
    const std::function<void(std::uint32_t, std::uint32_t, std::uint32_t)>* fn = nullptr;
    std::uint32_t total = 0;
    std::uint32_t chunk = 0;

    std::atomic<std::uint64_t> epoch{0}; ///< bumped once per region
    std::atomic<std::uint32_t> next{0};  ///< claimed chunk cursor

    /// Workers that have finished this region. The FIXED TEAM is the point: all
    /// n_threads-1 workers acknowledge every region, whether or not they got a
    /// chunk, and parallel_for waits for all of them.
    ///
    /// The first version instead counted ELEMENTS completed and returned as soon
    /// as they reached n. That let a straggler still inside drain() — between its
    /// epoch check and its next.fetch_add — observe the NEXT region's freshly
    /// reset `next` and `total`. It then processed another region's chunk against
    /// stale bookkeeping, the counters desynchronised, and the caller spun on a
    /// condition that could no longer become true. It hung at 16 threads after
    /// running clean at 1, 2, 3, 4 and 8, which is exactly how this class of bug
    /// presents: correct-looking, and load-dependent.
    ///
    /// Waiting on the team makes setup for region N+1 provably later than every
    /// worker's exit from region N, so there is no window to race.
    std::atomic<std::uint32_t> arrived{0};
    std::atomic<bool> stopping{false};

    std::mutex mu;
    std::condition_variable cv;

    /// Claim and run chunks until the region is exhausted.
    void drain(std::uint32_t worker) {
        for (;;) {
            const std::uint32_t begin = next.fetch_add(chunk, std::memory_order_relaxed);
            if (begin >= total) return;
            const std::uint32_t end = std::min(begin + chunk, total);
            (*fn)(begin, end, worker);
        }
    }

    void worker_loop(std::uint32_t worker) {
        t_in_region = true; // a worker is always inside the pool
        std::uint64_t seen = 0;
        for (;;) {
            // Spin first: in a 48-layer forward the next region is microseconds
            // away, and sleeping through that gap is most of the overhead.
            // PAUSE, not yield, for the first phase. yield() is a syscall that
            // enters the scheduler; with 32 workers polling it, the scheduler
            // becomes the bottleneck and the pool spends more time arbitrating
            // than computing. _mm_pause is a few cycles and hints the core to
            // release SMT resources to its sibling — which matters here, because
            // 32 workers on 16 physical cores means every pair shares one.
            bool got = false;
            for (int spin = 0; spin < 20000; ++spin) {
                if (epoch.load(std::memory_order_acquire) != seen) {
                    got = true;
                    break;
                }
                if (stopping.load(std::memory_order_acquire)) return;
                cpu_relax();
                if ((spin & 1023) == 1023) std::this_thread::yield();
            }
            if (!got) {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] {
                    return stopping.load(std::memory_order_acquire) ||
                           epoch.load(std::memory_order_acquire) != seen;
                });
            }
            if (stopping.load(std::memory_order_acquire)) return;

            seen = epoch.load(std::memory_order_acquire);
            drain(worker);
            arrived.fetch_add(1, std::memory_order_release);
        }
    }
};

ThreadPool::ThreadPool() : impl_(new Impl) {
    impl_->n_threads = desired_threads();
    // n_threads - 1 spawned: the caller is participant 0.
    impl_->workers.reserve(impl_->n_threads - 1);
    for (std::uint32_t i = 1; i < impl_->n_threads; ++i) {
        impl_->workers.emplace_back([this, i] { impl_->worker_loop(i); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->stopping.store(true, std::memory_order_release);
        impl_->epoch.fetch_add(1, std::memory_order_release);
    }
    impl_->cv.notify_all();
    for (auto& t : impl_->workers) {
        if (t.joinable()) t.join();
    }
    delete impl_;
}

ThreadPool& ThreadPool::global() {
    static ThreadPool pool;
    return pool;
}

std::uint32_t ThreadPool::size() const noexcept {
    return impl_->n_threads;
}

bool ThreadPool::in_parallel_region() noexcept {
    return t_in_region;
}

void ThreadPool::parallel_for(
    std::uint32_t n,
    std::uint32_t min_chunk,
    const std::function<void(std::uint32_t, std::uint32_t, std::uint32_t)>& fn) {
    if (n == 0) return;

    const std::uint32_t mc = std::max(1u, min_chunk);
    if (impl_->n_threads == 1 || t_in_region || n < mc * 2) {
        fn(0, n, 0);
        return;
    }

    // Chunk for load balance, not just for splitting: several chunks per worker
    // lets a thread that finishes early pick up more, which matters because the
    // attention rows are ragged (row t attends over t+1 keys, so the last row
    // costs n_tokens times the first).
    const std::uint32_t target = impl_->n_threads * 4;
    std::uint32_t chunk = std::max(mc, (n + target - 1) / target);

    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->fn = &fn;
        impl_->total = n;
        impl_->chunk = chunk;
        impl_->next.store(0, std::memory_order_relaxed);
        impl_->arrived.store(0, std::memory_order_relaxed);
        impl_->epoch.fetch_add(1, std::memory_order_release);
    }
    impl_->cv.notify_all();

    t_in_region = true;
    impl_->drain(0);
    t_in_region = false;

    // Wait for the whole TEAM, not for the element count. See `arrived`.
    const std::uint32_t team = impl_->n_threads - 1;
    int spin = 0;
    while (impl_->arrived.load(std::memory_order_acquire) < team) {
        cpu_relax();
        if (++spin > 8192) {
            std::this_thread::yield();
            spin = 0;
        }
    }
}

} // namespace soma
