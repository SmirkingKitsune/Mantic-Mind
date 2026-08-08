// Soma — the tiered memory manager.
//
// v1 is CPU-only: MemoryTier::Vram is reported on every surface and always
// empty. The tier exists in the enum, the telemetry, and the brain grid from day
// one so that adding GPU residency later is an implementation rather than a
// migration through every format and route.
//
// The one invariant that matters more than any policy here: an expert with a live
// ExpertRef CANNOT be evicted. Everything else is tuning; that one is
// correctness, and violating it means evicting bytes another thread is mid-GEMM
// over.

#include "soma/memory_hierarchy.hpp"

#include "soma/expert_store.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace soma {

namespace {

/// Exponential decay applied to heat on each sample, so a routing pattern that
/// stops being used loses its pin rather than holding it forever.
constexpr float kHeatDecay = 0.995f;

std::uint64_t slot_key(LayerIndex layer, ExpertId expert) noexcept {
    return (static_cast<std::uint64_t>(layer) << 32) | expert;
}

} // namespace

struct MemoryHierarchy::Impl {
    ExpertStore* store = nullptr;
    ArchIr arch;
    MemoryBudget budget{};

    std::uint32_t n_layers = 0;
    std::uint32_t n_experts = 0;
    std::uint64_t expert_bytes = 0;
    std::uint32_t cap_per_layer = 0;

    struct Slot {
        std::vector<std::byte> bytes;
        std::uint64_t last_used = 0;
        std::uint32_t refs = 0;
        bool pinned = false;
        bool resident = false;
        /// A read for this slot is in flight, on a loader thread or a caller.
        ///
        /// Without it, an acquire that arrives while the loader is mid-read finds
        /// the slot non-resident and starts a SECOND read of the same bytes. Both
        /// complete correctly — fetch_locked already handles the double-insert —
        /// but the whole point of prefetching is to do the I/O once and earlier,
        /// and doing it twice concurrently is worse than not prefetching at all.
        bool loading = false;
        std::uint64_t count = 0;
        float decayed = 0.0f;
    };

    mutable std::mutex mutex;
    std::vector<Slot> slots; // n_layers * n_experts, flat
    std::uint64_t clock = 0;
    std::uint64_t resident_bytes = 0;
    std::uint32_t resident_count = 0;
    std::uint32_t pinned_count = 0;

    CacheStats stats{};

    // Prefetch bookkeeping. `prefetched` marks slots brought in speculatively so
    // an eviction can be attributed as wasted rather than merely counted.
    std::vector<bool> prefetched;
    std::vector<bool> prefetch_layer_enabled;

    // ── background loader ────────────────────────────────────────────────────
    //
    // What turns "read earlier" into "read concurrently". prefetch_ahead() only
    // enqueues; these threads do the reading while the caller is applying the
    // expert it already has.
    std::deque<std::pair<LayerIndex, ExpertId>> queue;
    std::condition_variable queue_cv; ///< work available
    std::condition_variable slot_cv;  ///< a `loading` cleared
    std::vector<std::thread> loaders;
    std::uint32_t in_flight = 0;
    bool shutting_down = false;

    /// Both readers of each of these — the blocking one and the try_ one — call
    /// these. Two copies of the body would be two things to keep in step, and the
    /// non-blocking path is the one nobody watches.
    ///
    /// Callers hold `mutex`.
    TierOccupancy occupancy_locked() const noexcept {
        TierOccupancy o;
        o.vram_experts = 0; // v1: declared, always empty
        o.vram_bytes = 0;
        o.ram_experts = resident_count;
        o.ram_bytes = resident_bytes;
        o.ram_capacity_bytes = budget.ram_expert_cache_bytes;
        o.pinned_experts = pinned_count;
        o.disk_experts = static_cast<std::uint32_t>(slots.size()) - resident_count;
        return o;
    }

    HeatSnapshot heat_locked() const {
        HeatSnapshot snap;
        snap.n_layers = n_layers;
        snap.n_experts = n_experts;
        snap.cells.reserve(slots.size());
        for (std::uint32_t l = 0; l < n_layers; ++l) {
            for (std::uint32_t e = 0; e < n_experts; ++e) {
                const auto& sl = slots[idx(l, e)];
                if (sl.count == 0 && !sl.resident) continue;
                HeatCell c;
                c.layer = l;
                c.expert = e;
                c.tier = sl.resident ? MemoryTier::Ram : MemoryTier::Disk;
                c.count = sl.count;
                c.decayed = sl.decayed;
                snap.cells.push_back(c);
            }
        }
        return snap;
    }

    std::size_t idx(LayerIndex layer, ExpertId expert) const noexcept {
        return static_cast<std::size_t>(layer) * n_experts + expert;
    }

    /// Evict LRU until `need` bytes fit. Never touches a pinned slot or one with
    /// a live reference.
    bool make_room_locked(std::uint64_t need) {
        if (budget.ram_expert_cache_bytes == 0) return true;
        while (resident_bytes + need > budget.ram_expert_cache_bytes) {
            std::size_t victim = slots.size();
            std::uint64_t oldest = ~0ull;
            for (std::size_t i = 0; i < slots.size(); ++i) {
                const auto& s = slots[i];
                if (!s.resident || s.pinned || s.refs > 0) continue;
                if (s.last_used < oldest) {
                    oldest = s.last_used;
                    victim = i;
                }
            }
            if (victim == slots.size()) {
                // Everything resident is pinned or borrowed. Refusing is correct:
                // the alternative is evicting bytes in use.
                return false;
            }
            auto& v = slots[victim];
            resident_bytes -= v.bytes.size();
            v.bytes.clear();
            v.bytes.shrink_to_fit();
            v.resident = false;
            --resident_count;
            ++stats.evictions;
            if (prefetched[victim]) {
                ++stats.prefetch_wasted;
                prefetched[victim] = false;
            }
        }
        return true;
    }

    StatusCode fetch_locked(std::size_t i,
                            LayerIndex layer,
                            ExpertId expert,
                            std::unique_lock<std::mutex>& lk) {
        const auto loc = store->locate(layer, expert);
        if (loc.length == 0) return StatusCode::NotFound;

        if (!make_room_locked(loc.length)) return StatusCode::CapacityPressure;

        std::vector<std::byte> buf(loc.length);
        // Read with the lock RELEASED: the I/O is the slow part, and holding the
        // cache mutex across it would serialize every other thread's hits behind
        // one miss — exactly the stall the load pool exists to avoid.
        //
        // `loading` is published BEFORE unlocking, so anyone arriving during the
        // window waits for this read instead of starting a redundant one.
        slots[i].loading = true;
        lk.unlock();
        const auto rc = store->read(layer, expert, buf);
        lk.lock();
        slots[i].loading = false;
        slot_cv.notify_all();

        if (rc != StatusCode::Ok) return rc;

        auto& s = slots[i];
        if (s.resident) {
            // Another thread won the race. Its copy is equally valid and already
            // accounted for, so drop ours rather than double-counting bytes.
            return StatusCode::Ok;
        }
        resident_bytes += buf.size();
        s.bytes = std::move(buf);
        s.resident = true;
        ++resident_count;
        stats.bytes_read += s.bytes.size();
        return StatusCode::Ok;
    }
};

// ── ExpertRef ────────────────────────────────────────────────────────────────

MemoryHierarchy::ExpertRef::ExpertRef(MemoryHierarchy* owner,
                                      LayerIndex layer,
                                      ExpertId expert) noexcept
    : owner_(owner), layer_(layer), expert_(expert) {}

MemoryHierarchy::ExpertRef::ExpertRef(ExpertRef&& other) noexcept
    : owner_(other.owner_), layer_(other.layer_), expert_(other.expert_) {
    other.owner_ = nullptr;
}

MemoryHierarchy::ExpertRef& MemoryHierarchy::ExpertRef::operator=(ExpertRef&& other) noexcept {
    if (this != &other) {
        release();
        owner_ = other.owner_;
        layer_ = other.layer_;
        expert_ = other.expert_;
        other.owner_ = nullptr;
    }
    return *this;
}

MemoryHierarchy::ExpertRef::~ExpertRef() {
    release();
}

void MemoryHierarchy::ExpertRef::release() noexcept {
    if (owner_ == nullptr) return;
    auto& impl = *owner_->impl_;
    std::lock_guard<std::mutex> g(impl.mutex);
    auto& s = impl.slots[impl.idx(layer_, expert_)];
    if (s.refs > 0) --s.refs;
    owner_ = nullptr;
}

MemoryHierarchy::ExpertRef::operator bool() const noexcept {
    return owner_ != nullptr;
}

CByteSpan MemoryHierarchy::ExpertRef::bytes() const noexcept {
    if (owner_ == nullptr) return {};
    auto& impl = *owner_->impl_;
    // No lock: the slot is pinned by this reference for its whole lifetime, so
    // neither the buffer nor its address can change underneath us. That is the
    // property the RAII type exists to provide.
    const auto& s = impl.slots[impl.idx(layer_, expert_)];
    return CByteSpan(s.bytes.data(), s.bytes.size());
}

MemoryTier MemoryHierarchy::ExpertRef::tier() const noexcept {
    if (owner_ == nullptr) return MemoryTier::Disk;
    auto& impl = *owner_->impl_;
    const auto& s = impl.slots[impl.idx(layer_, expert_)];
    return s.resident ? MemoryTier::Ram : MemoryTier::Disk;
}

// ── MemoryHierarchy ──────────────────────────────────────────────────────────

MemoryHierarchy::MemoryHierarchy() : impl_(std::make_unique<Impl>()) {}

MemoryHierarchy::~MemoryHierarchy() {
    close();
}

void MemoryHierarchy::close() {
    // Stop the loaders before the Impl they reference is replaced. Rebuilding
    // impl_ with threads still running against the old one is a use-after-free
    // that would surface only under load, which is the worst way to find it.
    if (impl_) {
        {
            std::lock_guard<std::mutex> lk(impl_->mutex);
            impl_->shutting_down = true;
        }
        impl_->queue_cv.notify_all();
        for (auto& t : impl_->loaders) {
            if (t.joinable()) t.join();
        }
        impl_->loaders.clear();
    }
    impl_ = std::make_unique<Impl>();
}

Status MemoryHierarchy::open(const ArchIr& arch, ExpertStore& store, const MemoryBudget& budget) {
    close();
    auto& impl = *impl_;
    impl.store = &store;
    impl.arch = arch;
    impl.budget = budget;
    impl.n_layers = store.header().n_layers;
    impl.n_experts = store.header().n_experts;
    impl.expert_bytes = store.header().expert_bytes;

    if (impl.n_layers == 0 || impl.n_experts == 0) {
        return {StatusCode::InvalidArgument, "container reports no experts"};
    }

    const std::size_t n = static_cast<std::size_t>(impl.n_layers) * impl.n_experts;
    impl.slots.assign(n, Impl::Slot{});
    impl.prefetched.assign(n, false);
    impl.prefetch_layer_enabled.assign(impl.n_layers, false);

    const auto n_moe = std::max<std::uint32_t>(1, arch.n_moe_layers());
    impl.cap_per_layer = (impl.expert_bytes > 0)
                             ? static_cast<std::uint32_t>(budget.ram_expert_cache_bytes /
                                                          (impl.expert_bytes * n_moe))
                             : 0;

    // Loader threads. Default 2, not one per core: these threads do no
    // arithmetic, they wait on the device, and past the point where the queue
    // stays non-empty more of them only add contention on the cache mutex and
    // seek pressure on the disk.
    const auto n_loaders = (budget.load_pool_threads > 0) ? budget.load_pool_threads : 2u;
    impl.loaders.reserve(n_loaders);
    for (std::uint32_t t = 0; t < n_loaders; ++t) {
        impl.loaders.emplace_back([p = impl_.get()] {
            std::unique_lock<std::mutex> lk(p->mutex);
            for (;;) {
                p->queue_cv.wait(lk, [&] { return p->shutting_down || !p->queue.empty(); });
                if (p->shutting_down) return;

                const auto [layer, expert] = p->queue.front();
                p->queue.pop_front();

                const auto i = p->idx(layer, expert);
                if (i >= p->slots.size()) continue;
                // Re-checked under the lock: the caller may have reached this
                // expert first, or an earlier duplicate entry already loaded it.
                if (p->slots[i].resident || p->slots[i].loading) continue;

                ++p->in_flight;
                if (p->fetch_locked(i, layer, expert, lk) == StatusCode::Ok) {
                    p->prefetched[i] = true;
                    // Touch the LRU clock, or the prefetch is self-defeating.
                    //
                    // fetch_locked does not set last_used — on the acquire path
                    // the caller has already done it. A prefetched slot therefore
                    // kept whatever timestamp it had (0, if never used), which
                    // made it the OLDEST entry in the cache and so the first
                    // victim of the very next make_room. Experts were being
                    // fetched and evicted before the loop reached them, and the
                    // measurement showed it plainly: 21% MORE bytes read with
                    // prefetch enabled, and a net slowdown.
                    //
                    // A prefetched expert is about to be used, so it belongs at
                    // the MRU end, exactly like one that just was.
                    p->slots[i].last_used = ++p->clock;
                }
                --p->in_flight;
                // Wakes drain_prefetch(), which waits on this same cv for the
                // queue to empty AND the in-flight count to reach zero.
                p->queue_cv.notify_all();
            }
        });
    }
    return {};
}

std::uint32_t MemoryHierarchy::cap_per_layer() const noexcept {
    return impl_->cap_per_layer;
}

MemoryHierarchy::ExpertRef MemoryHierarchy::acquire(LayerIndex layer, ExpertId expert) noexcept {
    auto& impl = *impl_;
    std::unique_lock<std::mutex> lk(impl.mutex);

    const auto i = impl.idx(layer, expert);
    if (i >= impl.slots.size()) return {};
    auto& s = impl.slots[i];

    // The reference is taken BEFORE any I/O, so a concurrent make_room cannot
    // choose this slot as a victim while the read is in flight.
    ++s.refs;
    s.last_used = ++impl.clock;
    ++s.count;
    s.decayed = s.decayed * kHeatDecay + 1.0f;

    // A loader is already reading exactly these bytes: wait for it rather than
    // racing it. This is where prefetching actually pays when the estimate of
    // "how far ahead" was slightly short — the caller still waits, but for the
    // remainder of a read that started earlier, not for a whole new one.
    // io_wait_ns: the time the CALLER spends blocked on bytes.
    //
    // Declared in CacheStats from the beginning and never populated, which is
    // not a cosmetic gap — without it, "reads are ~44% of wall time" is an
    // inference from throughput rather than a measurement, and nothing can say
    // whether prefetching is covering the reads or merely running alongside
    // them. That gap is how a stale claim about serial reads survived long
    // enough to have a gate written around it (see G9).
    //
    // Both blocking sites are counted, and they mean different things:
    //   * this wait  — a read was ALREADY in flight, so prefetch fired but not
    //                  far enough ahead. Time here is a depth problem.
    //   * the miss   — nothing was in flight at all. Time there is a coverage
    //                  problem, and no amount of extra depth fixes it.
    // Summing them into one counter would hide exactly the distinction that
    // decides what to do next, so they are timed together and attributed apart.
    const auto wait_t0 = std::chrono::steady_clock::now();
    bool waited_on_inflight = false;
    while (s.loading) {
        waited_on_inflight = true;
        impl.slot_cv.wait(lk);
    }
    if (waited_on_inflight) {
        impl.stats.io_wait_ns +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - wait_t0)
                                           .count());
        ++impl.stats.inflight_waits;
    }

    if (s.resident) {
        ++impl.stats.hits;
        if (impl.prefetched[i]) {
            ++impl.stats.prefetch_hits;
            impl.prefetched[i] = false;
        }
        return ExpertRef(this, layer, expert);
    }

    ++impl.stats.misses;
    const auto miss_t0 = std::chrono::steady_clock::now();
    const auto rc = impl.fetch_locked(i, layer, expert, lk);
    const auto miss_ns =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - miss_t0)
                                       .count());
    // Counted in both: io_wait_ns is the total a caller spent blocked, and
    // miss_wait_ns is the part of it prefetching never had a chance at.
    impl.stats.io_wait_ns += miss_ns;
    impl.stats.miss_wait_ns += miss_ns;
    if (rc != StatusCode::Ok) {
        if (s.refs > 0) --s.refs;
        return {};
    }
    return ExpertRef(this, layer, expert);
}

void MemoryHierarchy::prefetch(LayerIndex layer, std::span<const ExpertId> experts) noexcept {
    auto& impl = *impl_;
    if (layer >= impl.prefetch_layer_enabled.size()) return;

    // Enabled PER LAYER, from measured router-lookahead recall. A layer whose
    // recall is poor gets none: a wrong prefetch is worse than no prefetch,
    // because it evicts something that was going to be used.
    if (!impl.prefetch_layer_enabled[layer]) return;

    std::unique_lock<std::mutex> lk(impl.mutex);
    for (const auto e : experts) {
        const auto i = impl.idx(layer, e);
        if (i >= impl.slots.size()) continue;
        auto& s = impl.slots[i];
        if (s.resident) continue;
        if (impl.fetch_locked(i, layer, e, lk) == StatusCode::Ok) {
            impl.prefetched[i] = true;
        } else {
            // Best-effort by contract. Failing to prefetch is not an error; it
            // just means the miss happens later, on the acquire that needs it.
            break;
        }
    }
}

void MemoryHierarchy::prefetch_ahead(LayerIndex layer, std::span<const ExpertId> experts) noexcept {
    auto& impl = *impl_;
    if (impl.loaders.empty()) return; // no pool: prefetching would just block

    {
        std::lock_guard<std::mutex> lk(impl.mutex);
        for (const auto e : experts) {
            const auto i = impl.idx(layer, e);
            if (i >= impl.slots.size()) continue;
            const auto& s = impl.slots[i];
            if (s.resident || s.loading) continue;

            // Not deduplicated against the queue itself. A duplicate entry costs
            // one wasted pop — the loader re-checks residency under the lock and
            // drops it — whereas scanning the queue on every enqueue would put an
            // O(depth) walk on the caller's critical path to save nothing.
            impl.queue.emplace_back(layer, e);
        }
    }
    impl.queue_cv.notify_all();
}

void MemoryHierarchy::drain_prefetch() noexcept {
    auto& impl = *impl_;
    std::unique_lock<std::mutex> lk(impl.mutex);
    impl.queue_cv.wait(lk, [&] { return impl.queue.empty() && impl.in_flight == 0; });
}

void MemoryHierarchy::set_prefetch_enabled(LayerIndex layer, bool enabled) noexcept {
    auto& impl = *impl_;
    if (layer < impl.prefetch_layer_enabled.size()) {
        impl.prefetch_layer_enabled[layer] = enabled;
    }
}

bool MemoryHierarchy::prefetch_enabled(LayerIndex layer) const noexcept {
    const auto& impl = *impl_;
    return layer < impl.prefetch_layer_enabled.size() && impl.prefetch_layer_enabled[layer];
}

bool MemoryHierarchy::would_thrash(std::uint32_t unique_experts_per_layer) const noexcept {
    const auto& impl = *impl_;
    if (impl.cap_per_layer == 0) return false; // unbounded cache
    return unique_experts_per_layer > impl.cap_per_layer;
}

void MemoryHierarchy::pin(LayerIndex layer, ExpertId expert) noexcept {
    auto& impl = *impl_;
    std::unique_lock<std::mutex> lk(impl.mutex);
    const auto i = impl.idx(layer, expert);
    if (i >= impl.slots.size()) return;
    auto& s = impl.slots[i];
    if (!s.resident) {
        if (impl.fetch_locked(i, layer, expert, lk) != StatusCode::Ok) return;
    }
    if (!s.pinned) {
        s.pinned = true;
        ++impl.pinned_count;
    }
}

void MemoryHierarchy::unpin(LayerIndex layer, ExpertId expert) noexcept {
    auto& impl = *impl_;
    std::lock_guard<std::mutex> g(impl.mutex);
    const auto i = impl.idx(layer, expert);
    if (i >= impl.slots.size()) return;
    if (impl.slots[i].pinned) {
        impl.slots[i].pinned = false;
        --impl.pinned_count;
    }
}

Status MemoryHierarchy::apply_heat_bootstrap(const HeatSnapshot& heat) {
    auto& impl = *impl_;

    // Pin the hottest experts up to the pin budget, so the cache is not cold on
    // first run. Sorted by decayed heat rather than raw count: a pattern that has
    // stopped being used should lose its pin.
    std::vector<const HeatCell*> ranked;
    ranked.reserve(heat.cells.size());
    for (const auto& c : heat.cells)
        ranked.push_back(&c);
    std::sort(ranked.begin(), ranked.end(), [](const HeatCell* a, const HeatCell* b) {
        return a->decayed > b->decayed;
    });

    std::uint64_t pinned_bytes = 0;
    std::uint32_t pinned = 0;
    for (const auto* c : ranked) {
        if (impl.budget.pin_bytes > 0 && pinned_bytes + impl.expert_bytes > impl.budget.pin_bytes) {
            break;
        }
        pin(c->layer, c->expert);
        {
            std::lock_guard<std::mutex> g(impl.mutex);
            const auto i = impl.idx(c->layer, c->expert);
            if (i < impl.slots.size() && impl.slots[i].pinned) {
                impl.slots[i].decayed = c->decayed;
                impl.slots[i].count = c->count;
                pinned_bytes += impl.expert_bytes;
                ++pinned;
            }
        }
    }
    if (pinned == 0 && !ranked.empty()) {
        return {StatusCode::CapacityPressure,
                "heat bootstrap pinned nothing; pin budget is smaller than one expert"};
    }
    return {};
}

TierOccupancy MemoryHierarchy::occupancy() const noexcept {
    const auto& impl = *impl_;
    std::lock_guard<std::mutex> g(impl.mutex);
    return impl.occupancy_locked();
}

CacheStats MemoryHierarchy::stats() const noexcept {
    const auto& impl = *impl_;
    std::lock_guard<std::mutex> g(impl.mutex);
    return impl.stats;
}

// ── the non-blocking readers ─────────────────────────────────────────────────
//
// The telemetry sampler uses these and never the blocking forms. This mutex is
// held across expert reads and evictions — most of what a streamed model does —
// so a sampler that waits for it samples nothing while the model works. See
// Scheduler::try_stats for the measurement that made this necessary.

bool MemoryHierarchy::try_occupancy(TierOccupancy& out) const noexcept {
    const auto& impl = *impl_;
    std::unique_lock<std::mutex> g(impl.mutex, std::try_to_lock);
    if (!g.owns_lock()) return false;
    out = impl.occupancy_locked();
    return true;
}

bool MemoryHierarchy::try_stats(CacheStats& out) const noexcept {
    const auto& impl = *impl_;
    std::unique_lock<std::mutex> g(impl.mutex, std::try_to_lock);
    if (!g.owns_lock()) return false;
    out = impl.stats;
    return true;
}

bool MemoryHierarchy::try_heat(HeatSnapshot& out) const {
    const auto& impl = *impl_;
    std::unique_lock<std::mutex> g(impl.mutex, std::try_to_lock);
    if (!g.owns_lock()) return false;
    out = impl.heat_locked();
    return true;
}

HeatSnapshot MemoryHierarchy::heat() const {
    const auto& impl = *impl_;
    std::lock_guard<std::mutex> g(impl.mutex);
    return impl.heat_locked();
}

} // namespace soma
