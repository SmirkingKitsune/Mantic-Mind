#pragma once

// Soma — fundamental types shared by the invariant core and every backend.
//
// This header is the root of the seam: it may be included from anywhere,
// including include/soma/arch/. It must never grow an architecture-specific
// concept.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace soma {

using LayerIndex = std::uint32_t;
using ExpertId = std::uint32_t;
using TokenId = std::uint32_t;
using SeqId = std::uint64_t;
using ByteSpan = std::span<std::byte>;
using CByteSpan = std::span<const std::byte>;

inline constexpr LayerIndex kInvalidLayer = ~LayerIndex{0};
inline constexpr ExpertId kInvalidExpert = ~ExpertId{0};
inline constexpr SeqId kInvalidSeq = ~SeqId{0};

/// Expert reads are issued O_DIRECT where the platform supports it, so every
/// offset and length in the container is aligned to this boundary.
inline constexpr std::size_t kDirectIoAlign = 4096;

enum class StatusCode : std::uint8_t {
    Ok = 0,
    InvalidArgument,
    NotFound,
    OutOfMemory,
    IoError,
    Unsupported,
    VersionMismatch,  ///< on-disk schema_version does not match this build
    ArchMismatch,     ///< arch_hash does not match the loaded model
    CapacityPressure, ///< admission refused; maps to the node's structured error
    Cancelled,
    Internal,
};

/// Error type for **cold paths only** — load, admission, checkpoint I/O, config.
/// May allocate.
///
/// The hot path (step, forward, expert acquire) returns `StatusCode` directly:
/// it fits in a register and cannot allocate. That split is deliberate and is
/// the reason two error types exist rather than one.
class Status {
public:
    /// Default-constructed is Ok. There is deliberately no `static Status ok()`
    /// factory — it would collide with the `ok()` predicate below, and the
    /// default constructor already says the same thing.
    Status() noexcept = default;
    Status(StatusCode code, std::string message);

    bool ok() const noexcept;
    StatusCode code() const noexcept;
    const std::string& message() const noexcept;

    explicit operator bool() const noexcept; ///< true when ok()

private:
    StatusCode code_ = StatusCode::Ok;
    std::string message_;
};

/// Where a tensor currently lives.
///
/// `Vram` is declared from day one and reported on every telemetry surface, but
/// is always zero-occupancy in v1 (CPU-only). Adding GPU residency later is then
/// an implementation, not a schema migration across every format and route.
enum class MemoryTier : std::uint8_t {
    Vram = 0,
    Ram = 1,
    Disk = 2,
};

const char* to_string(MemoryTier tier) noexcept;
const char* to_string(StatusCode code) noexcept;

/// Verdict — decides Soma vs. the llama.cpp fallback.
///
/// NOT a property of the model alone: it is a property of
/// (model, quantization, host budget). The registry stores the admission-host
/// verdict; `soma plan --json` re-derives it for the actual target node.
/// See schemas/arch-ir.md §8.
enum class Verdict : std::uint8_t {
    Stream = 0,   ///< → Soma
    Hybrid,       ///< → Soma
    ResidentOnly, ///< → fallback: it fits, streaming buys nothing
    Reject,       ///< → fallback: failed conformance stage 1 or 2
};

const char* to_string(Verdict verdict) noexcept;
bool verdict_selects_soma(Verdict verdict) noexcept;

} // namespace soma
