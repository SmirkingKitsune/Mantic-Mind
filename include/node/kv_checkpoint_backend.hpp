#pragma once

// Mantic-Mind — KV checkpoint save/restore, behind an interface.
//
// Today this is two hardcoded calls in SlotManager:
//     POST http://127.0.0.1:{port}/slots/0?action=save    (slot_manager.cpp:322)
//     POST http://127.0.0.1:{port}/slots/0?action=restore (slot_manager.cpp:446)
//
// Both hardcode SEQUENCE 0. A slot launched with --parallel > 1 therefore only
// ever checkpoints its first sequence — a latent data-loss bug in the current
// system, and one the rebuild must not carry into an engine whose whole point is
// running many sequences at once.
//
// Three callers, one mechanism (docs/architecture.md §6):
//   warm conversation reopen | scheduler preemption | cluster slot suspend

#include "common/models.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mm {

struct KvCheckpointInfo {
    std::string path;
    std::string arch_hash;       ///< empty for the fallback
    std::uint32_t format_id = 0; ///< backend-owned; guards cross-family replay
    std::uint32_t sequence = 0;
    std::uint32_t length_tokens = 0;
    std::int64_t bytes = 0;
    std::int64_t written_at_ms = 0;
};

class KvCheckpointBackend {
public:
    virtual ~KvCheckpointBackend() = default;

    virtual const char* engine_id() const = 0;

    /// Whether this engine can checkpoint sequences beyond index 0.
    ///
    /// False for llama.cpp. The supervisor consults it before advertising
    /// suspend on a multi-sequence slot, so the current silent partial save
    /// becomes an explicit refusal.
    virtual bool supports_multi_sequence() const = 0;

    /// Save one sequence. `sequence` is ignored by backends that report
    /// supports_multi_sequence() == false.
    virtual bool save(const std::string& base_url,
                      const std::string& api_key,
                      std::uint32_t sequence,
                      const std::string& checkpoint_path,
                      std::string& out_error) = 0;

    virtual bool restore(const std::string& base_url,
                         const std::string& api_key,
                         std::uint32_t sequence,
                         const std::string& checkpoint_path,
                         std::string& out_error) = 0;

    /// Read the header without loading the payload. Used to reject a
    /// cross-architecture resume BEFORE spawning an engine for it, which is the
    /// difference between a clean error and a confusing one.
    virtual bool
    stat(const std::string& checkpoint_path, KvCheckpointInfo& out, std::string& out_error) = 0;

    virtual bool remove(const std::string& checkpoint_path, std::string& out_error) = 0;
};

/// llama.cpp: POST /slots/{n}?action=save|restore.
///
/// supports_multi_sequence() returns false — the wire protocol addresses a slot
/// index, but llama-server's mapping from our sequences to its internal slots is
/// not ours to assume, and pretending otherwise is what produced the seq-0 bug.
class LlamaKvBackend final : public KvCheckpointBackend {
public:
    const char* engine_id() const override;
    bool supports_multi_sequence() const override;
    bool save(const std::string&,
              const std::string&,
              std::uint32_t,
              const std::string&,
              std::string&) override;
    bool restore(const std::string&,
                 const std::string&,
                 std::uint32_t,
                 const std::string&,
                 std::string&) override;
    bool stat(const std::string&, KvCheckpointInfo&, std::string&) override;
    bool remove(const std::string&, std::string&) override;
};

/// Soma: per-sequence, versioned, arch_hash- and format_id-gated.
class SomaKvBackend final : public KvCheckpointBackend {
public:
    const char* engine_id() const override;
    bool supports_multi_sequence() const override; ///< true
    bool save(const std::string&,
              const std::string&,
              std::uint32_t,
              const std::string&,
              std::string&) override;
    bool restore(const std::string&,
                 const std::string&,
                 std::uint32_t,
                 const std::string&,
                 std::string&) override;
    bool stat(const std::string&, KvCheckpointInfo&, std::string&) override;
    bool remove(const std::string&, std::string&) override;
};

} // namespace mm
