#pragma once

// Soma — KV persistence. ONE format, THREE callers:
//
//   1. warm conversation reopen    reopen with zero re-prefill
//   2. scheduler preemption        evict under memory pressure, re-admit later
//   3. cluster slot suspend/restore  stop the process, resume byte-identically
//
// These are the same operation, so they get the same format. Unifying them is
// not a convenience — it is the reason preemption is nearly free.
//
// Note what is NOT inherited from the fallback: llama.cpp's
// POST /slots/0?action=save hardcodes sequence 0, so a --parallel > 1 slot only
// ever checkpoints its first sequence. That is a latent data-loss bug in the
// current system. Checkpoints here are PER SEQUENCE.

#include "soma/attention_backend.hpp"
#include "soma/kv_cache.hpp"
#include "soma/model.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace soma {

/// v3 carries the SAMPLER's stream position and the penalty history.
///
/// v2 restored a cache and nothing else, so a resumed sequence drew from a fresh
/// RNG stream and with no repetition history. The tokens were correct and the
/// CONTINUATION was not: the same prompt resumed twice produced different text,
/// and a model that had already said something was free to say it again
/// immediately. Neither is visible as an error — they look like the model being
/// inconsistent.
///
/// v2 files refuse to load rather than being reinterpreted, as v1 did before
/// them.
///
/// v2 carries the TOKEN IDS the cached positions correspond to.
///
/// v1 did not, and that made every checkpoint unsafe to replay: a cache of
/// length L can be attached to any prompt, and if the first L tokens differ the
/// attention reads a context nobody asked for and the output is quietly wrong.
/// Nothing detects it. Carrying the ids costs 4 bytes per position against a
/// payload of `L x n_kv x n_layers x 2 x 4` — for a 24-layer model with 128 kv
/// channels, 0.016% — and turns "trust the caller" into a checkable prefix.
///
/// v1 files refuse to load rather than being reinterpreted, which is the same
/// rule every other on-disk format here follows.
inline constexpr std::uint32_t kKvCheckpointVersion = 3;
inline constexpr std::uint32_t kKvCheckpointVersionSpeculative = 4;

struct KvCheckpointHeader {
    std::uint32_t version = kKvCheckpointVersion;
    std::string arch_hash;

    /// Guards against replaying a checkpoint into an engine of a different
    /// attention family — different cache shape, silently wrong output rather
    /// than a clean failure. The value is backend-owned; see attention_backend.hpp.
    KvFormatId format_id = kKvFormatInvalid;

    std::uint32_t length_tokens = 0;
    std::uint32_t d_model = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t written_at_ms = 0;

    /// The sampler's stream position. Restoring it is what makes a resumed
    /// conversation continue rather than start a new one that happens to share
    /// a cache.
    std::uint64_t rng_state = 0;
    std::uint32_t n_emitted = 0;
    std::uint64_t auxiliary_bytes = 0;

    /// Offsets into the file, COMPUTED by the parser rather than stored. They
    /// are arithmetic from the fixed fields, which is what lets stat() read a
    /// bounded prefix of a multi-gigabyte checkpoint and still know where
    /// everything is.
    std::size_t tokens_at = 0;
    std::size_t emitted_at = 0;
    std::size_t payload_at = 0;
    std::size_t auxiliary_at = 0;
};

/// Everything a resume needs beyond the cache itself.
///
/// A struct rather than three more parameters: the set grows as the engine gains
/// per-sequence state, and a five-argument save() is where the wrong two get
/// swapped.
struct SeqPersistState {
    /// The ids occupying the cached positions, in order. Not metadata — it is
    /// what makes a restore checkable against the prompt it is attached to.
    std::vector<TokenId> tokens;

    /// What the sequence has produced. Feeds the repetition and presence
    /// penalties, which is why it is distinct from `tokens`: the last emitted
    /// token was sampled but never fed back, so it has no cached position.
    std::vector<TokenId> emitted;

    /// splitmix64 stream position. NOT the sampling parameters — temperature,
    /// top_p and the rest ride the request, and a checkpoint that restored them
    /// would make a client's change silently ineffective after a resume.
    std::uint64_t rng_state = 0;
    /// Optional backend-owned per-sequence state. Version 4 checkpoints append
    /// it after the ordinary target KV payload; v3 remains byte-identical.
    std::vector<std::byte> auxiliary;
};

/// Parse a checkpoint header out of raw bytes.
///
/// Exposed as a free function because the header is a WIRE FORMAT between two
/// binaries, not an implementation detail of the store. The node has to read it
/// before spawning an engine — that is the whole point of a pre-spawn resume
/// check — and a second parser living in the node is how the two would drift.
/// The store's own load()/stat()/sweep() go through this same function.
/// The offsets land in `out` and are computed arithmetically, so neither the
/// token arrays nor the payload need to be present in `data`. That is what lets
/// stat() read a bounded prefix of a multi-gigabyte checkpoint.
Status parse_kv_checkpoint_header(const std::byte* data, std::size_t size, KvCheckpointHeader& out);

/// Read just the header from a file. Reads a bounded prefix, never the payload —
/// a checkpoint is gigabytes and the caller only wants to know whether it is
/// replayable here.
Status read_kv_checkpoint_header(const std::string& path, KvCheckpointHeader& out);

/// The extension this format uses on disk. Backend-owned, like format_id.
const char* kv_checkpoint_extension() noexcept;

/// Persists and restores per-sequence KV regions.
///
/// Every load is gated on BOTH arch_hash and format_id. A mismatch returns
/// ArchMismatch or VersionMismatch rather than reading the bytes — a stale
/// resume against the wrong architecture is otherwise a very confusing bug
/// report, and the failure mode is degraded output rather than a crash.
class KvCheckpointStore {
public:
    KvCheckpointStore();
    KvCheckpointStore(const KvCheckpointStore&) = delete;
    KvCheckpointStore& operator=(const KvCheckpointStore&) = delete;
    ~KvCheckpointStore();

    Status open(const std::string& checkpoint_dir, const ArchIr& arch);
    void close();

    /// The pair, against the scheduler's KvCache.
    ///
    /// There were once two, this and a `SeqState` overload for an engine that was
    /// never built; the other has been deleted. One on-disk format is still the
    /// premise of this file: a checkpoint written by the scheduler must be
    /// loadable by warm reopen and by cluster slot restore without translation.
    ///
    /// `state.tokens` must be exactly the ids occupying the cached positions, in
    /// order — load() returns them so the caller can prove the checkpoint is a
    /// prefix of the prompt it is about to be attached to.
    Status save(const std::string& key, const KvCache& kv, const SeqPersistState& state);
    Status load(const std::string& key, KvCache& kv, SeqPersistState& out);

    Status stat(const std::string& key, KvCheckpointHeader& out) const;
    bool exists(const std::string& key) const noexcept;
    Status remove(const std::string& key);

    /// Drop checkpoints whose arch_hash no longer matches, and those older than
    /// max_age_ms. Called at open and on a timer, so a requantization does not
    /// leave unreadable files accumulating forever.
    Status sweep(std::uint64_t max_age_ms, std::uint32_t& out_removed);

    std::uint64_t total_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace soma
