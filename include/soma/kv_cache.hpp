#pragma once

// Soma — per-sequence KV storage, and the row view the batched forward consumes.
//
// This is the enabler for everything else in G3. Until now the forward was
// teacher-forced: it recomputed attention over the whole prefix on every call,
// which is fine for conformance and useless for serving. Batching one decode row
// from each of N sequences requires each row to attend over ITS OWN history, so
// the history has to be stored per sequence rather than recomputed per call.
//
// Layout is [layer][position][backend cache width] — position-major within a
// layer, because a decode row appends exactly one position and then reads the
// contiguous run [0, len). For GQA the width is kv_head * head_dim; MLA supplies
// its compressed latent width through the resolved backend.

#include "soma/arch_ir.hpp"
#include "soma/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace soma {

/// Reverse journal for bounded speculative KV writes.
///
/// Each capture is tagged with its verification-row ordinal. Rolling back rows
/// N..end in reverse order restores the exact state produced by rows 0..N-1,
/// even when several rows overwrite the same circular-window or compressor
/// carry bytes. This is deliberately byte-oriented: the backend owns the cache
/// layout, while the core owns transaction lifetime.
class KvTransaction {
public:
    void begin(std::uint32_t max_rows);
    void capture(std::uint32_t row, std::byte* destination, std::size_t bytes);
    void rollback_from(std::uint32_t first_row) noexcept;
    void clear() noexcept;
    bool active() const noexcept { return active_; }
    std::size_t journal_bytes() const noexcept { return saved_.size(); }

private:
    struct Entry {
        std::uint32_t row = 0;
        std::byte* destination = nullptr;
        std::size_t offset = 0;
        std::size_t bytes = 0;
    };
    std::vector<Entry> entries_;
    std::vector<std::byte> saved_;
    std::uint32_t max_rows_ = 0;
    bool active_ = false;
};

/// Floats per position per layer, per plane.
///
/// TWO numbers because the planes are not the same size for every family, and
/// assuming they were cost 2.94 GB on GLM-5.2 at 4k context. GQA stores per-head
/// K and V, so both planes are `n_kv_heads * head_dim`. MLA stores a compressed
/// latent and DERIVES V from it, so its V plane held nothing at all — and the
/// cache allocated it anyway, at the K plane's width, for every layer.
///
/// `v_floats == 0` is a real answer meaning "this family stores no second plane",
/// not a missing value. DSA is why MLA's is not always zero: its indexer key must
/// be cached, and the otherwise-dead plane is where it goes.
struct KvGeometry {
    std::uint32_t k_floats = 0;
    std::uint32_t v_floats = 0;
};

/// One sequence's K/V across all layers. Sized once at admission.
class KvCache {
public:
    Status open(const ArchIr& arch, std::uint32_t max_ctx);

    /// Positions currently stored. Also the next write index.
    std::uint32_t length() const noexcept { return length_; }

    std::uint32_t capacity() const noexcept { return max_ctx_; }

    void reset() noexcept {
        transaction_.clear();
        length_ = 0;
        if (!opaque_.empty()) std::fill(opaque_.begin(), opaque_.end(), std::byte{0});
    }

    /// Advance after a step has written its row. Separate from the write itself
    /// so a batched forward can fill every row's slot and only then commit —
    /// otherwise a row would attend over a neighbour's half-written position.
    void commit(std::uint32_t n = 1) noexcept { length_ += n; }

    /// Start a speculative write set without copying the live cache. Backends
    /// journal only bytes they mutate through KvRow::transaction.
    Status begin_tentative(std::uint32_t max_rows);
    Status commit_tentative_prefix(std::uint32_t accepted_rows);
    void abort_tentative() noexcept;
    KvTransaction* transaction() noexcept {
        return transaction_.active() ? &transaction_ : nullptr;
    }

    float* k_at(std::uint32_t layer, std::uint32_t pos) noexcept {
        return k_.data() + (static_cast<std::size_t>(layer) * max_ctx_ + pos) * k_hkv_;
    }

    /// Null when this family stores no second plane. A caller that does not check
    /// is a caller writing where nothing was allocated — which is why the DSA
    /// indexer, the only user, reaches it through `has_indexer`.
    float* v_at(std::uint32_t layer, std::uint32_t pos) noexcept {
        if (v_hkv_ == 0) return nullptr;
        return v_.data() + (static_cast<std::size_t>(layer) * max_ctx_ + pos) * v_hkv_;
    }

    std::uint64_t bytes() const noexcept {
        return opaque_.empty() ? (k_.size() + v_.size()) * sizeof(float) : opaque_.size();
    }

    bool is_opaque() const noexcept { return !opaque_.empty(); }
    std::byte* opaque_data() noexcept { return opaque_.data(); }
    const std::byte* opaque_data() const noexcept { return opaque_.data(); }
    std::size_t opaque_size() const noexcept { return opaque_.size(); }

    /// Floats per position, per plane. Checkpointing needs both to walk live
    /// positions layer by layer, and they are no longer the same number.
    std::uint32_t k_hkv() const noexcept { return k_hkv_; }

    std::uint32_t v_hkv() const noexcept { return v_hkv_; }

    /// Floats per layer in each plane — exactly `KvRow::k_stride` and
    /// `KvRow::v_stride`.
    ///
    /// Exposed so a caller building a KvRow takes the geometry FROM the buffer it
    /// is about to point into, rather than recomputing it. Recomputing is how
    /// `n_kv_heads * head_dim` ended up in the scheduler as well: correct for GQA,
    /// and for MLA a width that has nothing to do with what was allocated
    /// (roadmap D40).
    std::size_t k_stride() const noexcept { return static_cast<std::size_t>(max_ctx_) * k_hkv_; }

    std::size_t v_stride() const noexcept { return static_cast<std::size_t>(max_ctx_) * v_hkv_; }

    std::uint32_t n_layers() const noexcept {
        if (is_opaque()) return n_layers_;
        return k_hkv_ && max_ctx_ ? static_cast<std::uint32_t>(
                                        k_.size() / (static_cast<std::size_t>(max_ctx_) * k_hkv_))
                                  : 0;
    }

    /// Set the live length directly. Restore-only: `commit` is the forward's
    /// path and asserts nothing about where the data came from.
    Status set_length(std::uint32_t n) noexcept {
        if (n > max_ctx_) return {StatusCode::InvalidArgument, "restored length exceeds capacity"};
        length_ = n;
        return {};
    }

private:
    std::vector<float> k_, v_;
    std::vector<std::byte> opaque_;
    std::uint32_t max_ctx_ = 0;
    std::uint32_t k_hkv_ = 0; ///< backend K width in floats
    std::uint32_t v_hkv_ = 0; ///< backend V width; 0 when the family stores none
    std::uint32_t length_ = 0;
    std::uint32_t n_layers_ = 0;
    KvTransaction transaction_{};
};

/// What one row of a batched forward needs in order to attend.
///
/// A ROW, not a sequence — that distinction is the scheduler's whole premise.
/// Rows from different sequences, at different positions, with different history
/// lengths, sit side by side in one forward; only this struct differs between
/// them. Prefill rows and decode rows are the same type.
struct KvRow {
    float* k_base = nullptr;  ///< this sequence's K, layer 0
    float* v_base = nullptr;  ///< null when the family stores no V plane
    std::size_t k_stride = 0; ///< floats per layer (max_ctx * k_hkv)
    std::size_t v_stride = 0;
    std::uint32_t k_hkv = 0;
    std::uint32_t v_hkv = 0; ///< 0 when the family stores no V plane
    std::uint32_t pos = 0;   ///< this row's absolute position (RoPE + write slot)
    std::uint32_t len = 0;   ///< positions visible to it, INCLUDING pos
    std::byte* opaque_base = nullptr;
    std::size_t opaque_bytes = 0;
    std::uint32_t max_ctx = 0;
    KvTransaction* transaction = nullptr;
    std::uint32_t transaction_row = 0;

    // Named per plane rather than sharing one `stride`/`hkv` pair. The shared
    // pair was correct only while both planes had the same shape, and it made the
    // asymmetry inexpressible — so MLA's empty V plane was allocated at the K
    // plane's width for every layer. Renaming forces every caller to say which
    // plane it means instead of inheriting an assumption.
    float* k_at(std::uint32_t layer, std::uint32_t p) const noexcept {
        return k_base + static_cast<std::size_t>(layer) * k_stride +
               static_cast<std::size_t>(p) * k_hkv;
    }

    float* v_at(std::uint32_t layer, std::uint32_t p) const noexcept {
        if (v_base == nullptr || v_hkv == 0) return nullptr;
        return v_base + static_cast<std::size_t>(layer) * v_stride +
               static_cast<std::size_t>(p) * v_hkv;
    }
};

} // namespace soma
