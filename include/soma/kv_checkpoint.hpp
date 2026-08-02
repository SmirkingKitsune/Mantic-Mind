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

namespace soma {

inline constexpr std::uint32_t kKvCheckpointVersion = 1;

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
};

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

    Status save(const std::string& key, const SeqState& seq);
    Status load(const std::string& key, SeqState& seq);

    /// The G3 pair, against the fp32 path's KvCache.
    ///
    /// Same format, same header, same gating as the SeqState overloads — only the
    /// in-memory type differs. Keeping one on-disk format across both is the
    /// whole premise of this file: a checkpoint written by the scheduler must be
    /// loadable by warm reopen and by cluster slot restore without translation.
    Status save(const std::string& key, const KvCache& kv);
    Status load(const std::string& key, KvCache& kv);

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
