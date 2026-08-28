#include "soma/types.hpp"

#include <utility>

namespace soma {

Status::Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

bool Status::ok() const noexcept {
    return code_ == StatusCode::Ok;
}

StatusCode Status::code() const noexcept {
    return code_;
}

const std::string& Status::message() const noexcept {
    return message_;
}

Status::operator bool() const noexcept {
    return ok();
}

const char* to_string(MemoryTier tier) noexcept {
    switch (tier) {
    case MemoryTier::Vram:
        return "vram";
    case MemoryTier::Ram:
        return "ram";
    case MemoryTier::Disk:
        return "disk";
    }
    return "unknown";
}

const char* to_string(StatusCode code) noexcept {
    switch (code) {
    case StatusCode::Ok:
        return "ok";
    case StatusCode::InvalidArgument:
        return "invalid_argument";
    case StatusCode::NotFound:
        return "not_found";
    case StatusCode::OutOfMemory:
        return "out_of_memory";
    case StatusCode::IoError:
        return "io_error";
    case StatusCode::Unsupported:
        return "unsupported";
    case StatusCode::VersionMismatch:
        return "version_mismatch";
    case StatusCode::ArchMismatch:
        return "arch_mismatch";
    case StatusCode::CapacityPressure:
        return "capacity_pressure";
    case StatusCode::Cancelled:
        return "cancelled";
    case StatusCode::Internal:
        return "internal";
    }
    return "unknown";
}

const char* to_string(Verdict verdict) noexcept {
    switch (verdict) {
    case Verdict::Stream:
        return "stream";
    case Verdict::Hybrid:
        return "hybrid";
    case Verdict::ResidentOnly:
        return "resident-only";
    case Verdict::Reject:
        return "reject";
    }
    return "unknown";
}

bool verdict_selects_soma(Verdict verdict) noexcept {
    // resident-only and reject both route to the llama.cpp fallback. See
    // docs/architecture.md §8.2 — the first because streaming buys nothing, the
    // second because conformance failed.
    return verdict == Verdict::Stream || verdict == Verdict::Hybrid;
}

} // namespace soma
