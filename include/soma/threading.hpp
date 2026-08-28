#pragma once

// Soma — the forward's thread pool.
//
// The last kernel pass left reads at 28% of wall time with sixteen cores idle,
// so this is the largest remaining factor before the G3 throughput gate can be
// evaluated at all.
//
// ── The rule that shapes the whole design ────────────────────────────────────
//
// EVERY parallel region here partitions OUTPUT elements, never input ranges.
// Each output is computed start to finish by exactly one worker, in the same
// order a single thread would have used, so results are BIT-IDENTICAL to the
// serial path regardless of thread count or scheduling.
//
// That is not a nicety. Splitting a reduction across threads and combining the
// partials would make output depend on how many cores the host has and on how
// the OS happened to schedule — which would put `determinism: strict` out of
// reach permanently, break the streamed-vs-resident bit-identity check, and make
// every conformance number a function of the machine that produced it. Read-only
// sharing of inputs is free; only the write side needs partitioning.
//
// The consequence, stated plainly: reductions over a single output (a dot
// product, an rmsnorm sum) are NOT parallelised. Only loops with many
// independent outputs are.

#include <cstdint>
#include <functional>

namespace soma {

/// Fixed-size pool, created once and shared by the whole process.
class ThreadPool {
public:
    /// Workers = SOMA_THREADS if set, else hardware_concurrency(), clamped to
    /// [1, 256]. One of them is the CALLING thread, which participates rather
    /// than blocking — with short regions, handing all work to others and idling
    /// wastes the core the caller is already holding.
    static ThreadPool& global();

    /// Total participating threads, including the caller. 1 means serial.
    std::uint32_t size() const noexcept;

    /// Split [0, n) across workers and run `fn(begin, end, worker)` on each part.
    /// Blocks until all parts finish.
    ///
    /// `worker` is a dense index in [0, size()) — the hook for per-thread
    /// scratch, which is what lets a loop with shared scratch buffers be
    /// parallelised without turning them into a race.
    ///
    /// Runs SERIALLY, on this thread, when any of these hold:
    ///   * n < min_chunk * 2      — not enough work to cover the sync cost
    ///   * size() == 1
    ///   * already inside a parallel region (see below)
    ///
    /// NESTING IS FLATTENED, not forbidden. `matvec` parallelises internally and
    /// is also called from inside already-parallel loops; if both levels spawned,
    /// the pool would oversubscribe by size()^2. An inner call therefore detects
    /// it is on a worker and runs serially, so a hot loop can be parallelised at
    /// whichever level is coarsest without auditing every callee.
    void parallel_for(std::uint32_t n,
                      std::uint32_t min_chunk,
                      const std::function<void(std::uint32_t, std::uint32_t, std::uint32_t)>& fn);

    /// True when called on a pool worker (including inside a parallel_for body).
    static bool in_parallel_region() noexcept;

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    ThreadPool();
    ~ThreadPool();
    struct Impl;
    Impl* impl_;
};

/// Total work below this many multiply-accumulates is left serial.
///
/// Waking workers and rejoining costs a few microseconds; a region that only
/// takes a few microseconds to run therefore costs more to distribute than to
/// do. An expert's [768, 2048] projection is 1.57 M MACs and sits just above.
inline constexpr std::uint64_t kParallelMacThreshold = 1u << 20;

/// Target work PER CHUNK — a different question from the one above, and
/// conflating them is a mistake worth naming because it silently disables the
/// whole pool.
///
/// The first version of this code derived `min_chunk` from
/// kParallelMacThreshold, giving 512 rows per chunk at k = 2048. parallel_for
/// requires n >= min_chunk * 2 before it will split, so every matvec with fewer
/// than 1024 output rows ran SERIALLY — which is all of the expert projections,
/// i.e. most of the engine. It still passed every test, still produced identical
/// numbers, and delivered 1.2x on 32 cores.
///
/// A chunk wants to be a few tens of microseconds: long enough to dwarf the
/// dispatch, short enough that several land per worker for load balancing.
/// ~128 K MACs is roughly 7 us at the measured 35 GF/s.
inline constexpr std::uint64_t kChunkMacs = 1u << 17;

} // namespace soma
