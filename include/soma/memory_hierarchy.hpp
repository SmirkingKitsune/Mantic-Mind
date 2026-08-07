#pragma once

// Soma — the tiered memory manager.
//
// VRAM, RAM, and disk are one managed hierarchy. The static partition splits the
// weight graph in two and everything here is bookkeeping around that split:
//
//   dense / resident   attention projections, shared experts, embeddings, norms,
//                      router weights. Loaded once, never evicted.
//   routed / streamed  routed expert weights. Live on disk, page through here.
//
// v1 is CPU-only: MemoryTier::Vram is reported everywhere and always empty.

#include "soma/arch_ir.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace soma {

class ExpertStore;

struct TierOccupancy {
    std::uint32_t vram_experts = 0;
    std::uint32_t ram_experts = 0;
    std::uint32_t disk_experts = 0;
    std::uint32_t pinned_experts = 0;

    std::uint64_t vram_bytes = 0;
    std::uint64_t ram_bytes = 0;
    std::uint64_t ram_capacity_bytes = 0;
};

struct CacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t prefetch_hits = 0;
    std::uint64_t prefetch_wasted = 0; ///< prefetched, then evicted unused
    std::uint64_t bytes_read = 0;
    std::uint64_t io_wait_ns = 0;
};

/// One cell of the heat map / brain grid.
struct HeatCell {
    LayerIndex layer = kInvalidLayer;
    ExpertId expert = kInvalidExpert;
    MemoryTier tier = MemoryTier::Disk;
    std::uint64_t count = 0;
    float decayed = 0.0f;
};

struct HeatSnapshot {
    std::uint32_t n_layers = 0;
    std::uint32_t n_experts = 0;
    std::vector<HeatCell> cells;
};

struct MemoryBudget {
    std::uint64_t ram_expert_cache_bytes = 0;
    std::uint64_t vram_hot_bytes = 0; ///< v1: always 0
    std::uint64_t pin_bytes = 0;
    std::uint32_t readahead_depth = 0;
    std::uint32_t load_pool_threads = 0;
};

/// Manages expert residency across tiers: per-layer LRU, pinned hot store, and
/// the OS page cache as a free L2.
class MemoryHierarchy {
public:
    /// An RAII borrow of a resident expert. The expert cannot be evicted while
    /// any ExpertRef to it is alive.
    ///
    /// This is the one place in the memory manager where a subtle bug is both
    /// easy to write and catastrophic under concurrency — evicting bytes another
    /// thread is mid-GEMM over. It is therefore expressed in the type system
    /// rather than in a convention, and there is deliberately no way to get at
    /// expert bytes without holding one.
    class ExpertRef {
    public:
        ExpertRef() noexcept = default;
        ExpertRef(const ExpertRef&) = delete;
        ExpertRef& operator=(const ExpertRef&) = delete;
        ExpertRef(ExpertRef&& other) noexcept;
        ExpertRef& operator=(ExpertRef&& other) noexcept;
        ~ExpertRef();

        explicit operator bool() const noexcept;
        CByteSpan bytes() const noexcept;
        MemoryTier tier() const noexcept;

    private:
        friend class MemoryHierarchy;
        ExpertRef(MemoryHierarchy* owner, LayerIndex layer, ExpertId expert) noexcept;
        void release() noexcept;

        MemoryHierarchy* owner_ = nullptr;
        LayerIndex layer_ = kInvalidLayer;
        ExpertId expert_ = kInvalidExpert;
    };

    MemoryHierarchy();
    MemoryHierarchy(const MemoryHierarchy&) = delete;
    MemoryHierarchy& operator=(const MemoryHierarchy&) = delete;
    ~MemoryHierarchy();

    Status open(const ArchIr& arch, ExpertStore& store, const MemoryBudget& budget);
    void close();

    /// Blocks on I/O if the expert is not resident. Returns a falsy ref on
    /// failure; callers must check.
    ExpertRef acquire(LayerIndex layer, ExpertId expert) noexcept;

    /// Non-blocking, best-effort. Drives router-lookahead prefetch.
    ///
    /// Enabled PER LAYER, only above the recall threshold measured at admission
    /// (pilot_profile). A wrong prefetch is worse than none — it evicts
    /// something useful — so a layer whose lookahead recall is poor gets none.
    void prefetch(LayerIndex layer, std::span<const ExpertId> experts) noexcept;

    /// Queue experts for the background loader and return immediately.
    ///
    /// A DIFFERENT KIND OF PREFETCH from the one above, and the distinction is
    /// the reason this is a separate entry point rather than a flag.
    ///
    ///   `prefetch()`       SPECULATIVE. Guesses which experts the next step will
    ///                      route to, from router lookahead. Can be wrong, so it
    ///                      is gated per layer on measured recall.
    ///   `prefetch_ahead()` CERTAIN. The caller has already built this step's
    ///                      expert union, so these experts WILL be read. There is
    ///                      no recall question and no per-layer gate: the only
    ///                      thing being decided is *when* the read happens, not
    ///                      whether it was needed.
    ///
    /// Because the reads are certain, the risk is purely one of ordering — queue
    /// too far ahead against a small cache and a prefetched expert is evicted
    /// before its turn. Callers bound the depth against `cap_per_layer()`.
    void prefetch_ahead(LayerIndex layer, std::span<const ExpertId> experts) noexcept;

    /// Block until the background loader has drained. Testing and shutdown only.
    void drain_prefetch() noexcept;

    /// Ask ahead of a step whether this set is likely to fit without thrashing.
    /// Feeds the scheduler's cache-aware admission gate.
    bool would_thrash(std::uint32_t unique_experts_per_layer) const noexcept;

    /// Per-layer prefetch gate, set from measured router-lookahead recall
    /// (pilot_profile). Default is OFF for every layer: prefetch has to earn its
    /// place, because a wrong prefetch evicts something that was going to be used.
    void set_prefetch_enabled(LayerIndex layer, bool enabled) noexcept;
    bool prefetch_enabled(LayerIndex layer) const noexcept;

    std::uint32_t cap_per_layer() const noexcept;

    void pin(LayerIndex layer, ExpertId expert) noexcept;
    void unpin(LayerIndex layer, ExpertId expert) noexcept;

    /// Pin the hottest experts at startup from the persisted histogram, so the
    /// cache is not cold on first run.
    Status apply_heat_bootstrap(const HeatSnapshot& heat);

    TierOccupancy occupancy() const noexcept;
    CacheStats stats() const noexcept;

    /// Aggregated snapshot. Heat counters are accumulated HERE and sampled at
    /// the telemetry tick rate — never emitted per token. That is what makes the
    /// throttle undefeatable by a careless client rather than merely advisory.
    HeatSnapshot heat() const;

    /// Non-blocking reads for the telemetry path — see Scheduler::try_stats.
    ///
    /// The hierarchy's mutex is held across expert reads and evictions, which is
    /// most of what a streamed model does. A sampler that waits for it samples
    /// nothing while the model works. Each returns false without touching `out`
    /// when the lock was busy.
    bool try_occupancy(TierOccupancy& out) const noexcept;
    bool try_stats(CacheStats& out) const noexcept;
    bool try_heat(HeatSnapshot& out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace soma
