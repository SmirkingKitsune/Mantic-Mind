#pragma once

// Soma — what a sequence's cached positions were built from, beyond their token
// ids.
//
// A KV checkpoint carries the token ids occupying its cached positions so a
// restore can be CHECKED against the prompt it is attached to (kv_checkpoint.hpp
// explains why v1, which did not, was unsafe). That check rests on an assumption
// the engine has always been able to make: that the token ids DETERMINE the
// hidden state.
//
// A supplied embedding breaks it. Two different images occupy the same
// placeholder positions and produce byte-identical token arrays, so the prefix
// check passes, the cache is attached, and the model answers fluently about a
// picture nobody sent — the exact failure the token array exists to prevent,
// one layer down and invisible to it.
//
// So a checkpoint records a digest of the embeddings that were supplied for its
// cached positions, and a restore recomputes it over the new request's
// embeddings. Text-only sequences supply none, digest stays empty, and both
// sides compare equal — the check costs nothing when nothing is at stake.
//
// DIGESTED OVER THE ROWS, NOT THE SOURCE IMAGE. Digesting the source bytes would
// be weaker in the direction that matters: one JPEG through two preprocessing
// paths gives different rows and an identical source digest, so a cache built by
// one would validate against the other. The rows are what actually reached the
// forward.

#include "soma/types.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace soma {

/// A rolling digest over (position, row) pairs, in ascending position order.
///
/// A CHAIN rather than a stream — `acc = H(acc || position || row)` — so the
/// accumulator IS the digest. That is what lets a checkpoint restore one and
/// keep folding: a streaming hash would have to persist its internal state,
/// and a cross-process resume has only what is on disk.
struct MediaDigest {
    std::array<std::uint8_t, 32> bytes{};

    /// No embeddings were supplied. The all-zero value is also the chain's
    /// starting state, so "absorbed nothing" and "empty" are the same thing
    /// rather than two states that have to be kept in agreement.
    ///
    /// A real chain landing on all-zero is a 2^-256 event, and its only
    /// consequence would be a spurious cold start — the safe direction.
    bool empty() const noexcept;

    /// Lowercase hex, for log lines and test failure messages.
    std::string hex() const;

    friend bool operator==(const MediaDigest&, const MediaDigest&) noexcept = default;
};

/// Fold one supplied row into the running digest.
///
/// `position` is folded in alongside the values because the same row at a
/// different prompt position is a different context, and a digest that could not
/// tell them apart would accept a cache whose image had moved.
void media_digest_fold(MediaDigest& acc,
                       std::uint32_t position,
                       std::span<const float> row) noexcept;

/// The digest of every supplied row at a position BELOW `upto`.
///
/// A pure function of its inputs, which is the property the whole scheme rests
/// on: a save records what it folded in incrementally, a restore recomputes it in
/// one pass over the new request, and the two must agree exactly.
///
/// `positions` must be ascending and `values` holds `positions.size() * d_model`
/// floats row-major in the same order. A malformed pair yields an empty digest
/// rather than a partial one — the caller validates the shape (see
/// `validate_prompt_embeddings`) and a digest is not the place to report it.
MediaDigest media_digest_prefix(std::span<const std::uint32_t> positions,
                                std::span<const float> values,
                                std::uint32_t d_model,
                                std::uint32_t upto) noexcept;

/// Pre-computed hidden-state rows supplied for specific prompt positions.
///
/// The rows REPLACE the embedding-table lookup at those positions. The token ids
/// underneath them stay real — a placeholder id the model's own template emits —
/// so everything downstream that reads the token array (the attention backends'
/// begin-forward hook among them) keeps working unchanged.
struct PromptEmbeddings {
    /// Prompt indices carrying a supplied row. Strictly ascending.
    std::vector<std::uint32_t> positions;
    /// `positions.size() * d_model` floats, row-major, in `positions` order.
    std::vector<float> values;

    bool empty() const noexcept { return positions.empty(); }
};

/// Reject a malformed pair at the boundary rather than deep in a forward.
///
/// Ascending-and-unique is not fussiness: the batch builder looks positions up
/// by binary search and the digest folds them in order, so an unsorted list
/// would silently override the wrong row and digest to something neither side
/// could reproduce.
Status validate_prompt_embeddings(const PromptEmbeddings& e,
                                  std::uint32_t d_model,
                                  std::uint32_t prompt_len);

} // namespace soma
