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

#include <cstdint>
#include <vector>

namespace soma {

/// One sequence's K/V across all layers. Sized once at admission.
class KvCache {
public:
    Status open(const ArchIr& arch, std::uint32_t max_ctx);

    /// Positions currently stored. Also the next write index.
    std::uint32_t length() const noexcept { return length_; }

    std::uint32_t capacity() const noexcept { return max_ctx_; }

    void reset() noexcept { length_ = 0; }

    /// Advance after a step has written its row. Separate from the write itself
    /// so a batched forward can fill every row's slot and only then commit —
    /// otherwise a row would attend over a neighbour's half-written position.
    void commit(std::uint32_t n = 1) noexcept { length_ += n; }

    float* k_at(std::uint32_t layer, std::uint32_t pos) noexcept {
        return k_.data() + (static_cast<std::size_t>(layer) * max_ctx_ + pos) * hkv_;
    }

    float* v_at(std::uint32_t layer, std::uint32_t pos) noexcept {
        return v_.data() + (static_cast<std::size_t>(layer) * max_ctx_ + pos) * hkv_;
    }

    std::uint64_t bytes() const noexcept { return (k_.size() + v_.size()) * sizeof(float); }

    /// Floats per layer in ONE of the two planes. Checkpointing needs this to
    /// walk live positions layer by layer.
    std::uint32_t hkv() const noexcept { return hkv_; }

    std::uint32_t n_layers() const noexcept {
        return hkv_ && max_ctx_ ? static_cast<std::uint32_t>(
                                      k_.size() / (static_cast<std::size_t>(max_ctx_) * hkv_))
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
    std::uint32_t max_ctx_ = 0;
    std::uint32_t hkv_ = 0; ///< backend cache width in floats, per plane
    std::uint32_t length_ = 0;
};

/// What one row of a batched forward needs in order to attend.
///
/// A ROW, not a sequence — that distinction is the scheduler's whole premise.
/// Rows from different sequences, at different positions, with different history
/// lengths, sit side by side in one forward; only this struct differs between
/// them. Prefill rows and decode rows are the same type.
struct KvRow {
    float* k_base = nullptr; ///< this sequence's K, layer 0
    float* v_base = nullptr;
    std::uint32_t stride = 0; ///< floats per layer (max_ctx * hkv)
    std::uint32_t hkv = 0;
    std::uint32_t pos = 0; ///< this row's absolute position (RoPE + write slot)
    std::uint32_t len = 0; ///< positions visible to it, INCLUDING pos

    float* k_at(std::uint32_t layer, std::uint32_t p) const noexcept {
        return k_base + static_cast<std::size_t>(layer) * stride +
               static_cast<std::size_t>(p) * hkv;
    }

    float* v_at(std::uint32_t layer, std::uint32_t p) const noexcept {
        return v_base + static_cast<std::size_t>(layer) * stride +
               static_cast<std::size_t>(p) * hkv;
    }
};

} // namespace soma
