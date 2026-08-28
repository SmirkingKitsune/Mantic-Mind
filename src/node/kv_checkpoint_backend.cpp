// Mantic-Mind — KV checkpoint save/restore, behind an interface.
//
// What this replaces is two hardcoded calls in SlotManager:
//     POST http://127.0.0.1:{port}/slots/0?action=save    (slot_manager.cpp:322)
//     POST http://127.0.0.1:{port}/slots/0?action=restore (slot_manager.cpp:446)
//
// Both hardcode SEQUENCE 0, so a slot launched with --parallel > 1 only ever
// checkpoints its first sequence and silently discards the rest. The wire
// protocol is unchanged for llama.cpp — it is the right protocol, and it is the
// only one llama-server speaks — but supports_multi_sequence() now REPORTS the
// limitation, which is what lets EngineSupervisor::suspend refuse the case
// instead of performing a partial save that looks like a complete one.
//
// The asymmetry between the two backends below is real and worth reading as
// design rather than as an omission: llama-server can be told to serialise its
// live slot state at any moment, and `soma serve` cannot, because it holds no
// sequence between requests. See SomaKvBackend::save.

#include "node/kv_checkpoint_backend.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"
#include "soma/kv_checkpoint.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace mm {

namespace {

/// llama-server addresses checkpoints by BASENAME, relative to its own
/// --slot-save-path. The node deals in absolute paths, so every call splits one.
/// The directory half is not discarded silently: a mismatch against the launch
/// argument would write the file somewhere the node will never look for it, and
/// that presents later as "the checkpoint vanished".
std::string basename_of(const std::string& path) {
    return fs::path(path).filename().string();
}

httplib::Client engine_client(const std::string& base_url, const std::string& api_key) {
    httplib::Client cli(base_url);
    cli.set_connection_timeout(5);
    // Serialising a multi-gigabyte KV region is not fast, and this call runs on
    // the suspend path where the alternative to waiting is losing the context.
    cli.set_read_timeout(120);
    cli.set_write_timeout(30);
    if (!api_key.empty()) cli.set_bearer_token_auth(api_key);
    return cli;
}

bool file_stat(const std::string& path, KvCheckpointInfo& out, std::string& out_error) {
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        out_error = "no checkpoint at " + path;
        return false;
    }
    const auto bytes = fs::file_size(path, ec);
    if (ec) {
        out_error = "cannot size " + path;
        return false;
    }
    if (bytes == 0) {
        // A zero-byte checkpoint is the shape a crash mid-write leaves behind.
        // Restoring from it fails deep inside the engine; refusing here is the
        // difference between a clear error and a confusing one.
        out_error = "empty checkpoint " + path;
        return false;
    }
    out.path = path;
    out.bytes = static_cast<std::int64_t>(bytes);

    const auto written = fs::last_write_time(path, ec);
    if (!ec) {
        const auto sys = std::chrono::clock_cast<std::chrono::system_clock>(written);
        out.written_at_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(sys.time_since_epoch()).count();
    }
    return true;
}

bool file_remove(const std::string& path, std::string& out_error) {
    std::error_code ec;
    if (!fs::exists(path, ec)) return true; // idempotent: gone is gone
    if (!fs::remove(path, ec) || ec) {
        out_error = "cannot remove " + path + ": " + ec.message();
        return false;
    }
    return true;
}

} // namespace

// ── LlamaKvBackend ────────────────────────────────────────────────────────────

const char* LlamaKvBackend::engine_id() const {
    return "llama-cpp";
}

const char* LlamaKvBackend::file_extension() const {
    return ".kvbin";
}

bool LlamaKvBackend::supports_multi_sequence() const {
    // The wire protocol addresses a slot index, so `false` here is not a claim
    // that llama-server cannot serialise slot 1. It is a claim that OUR mapping
    // from sequences to its internal slots is not ours to assume — and assuming
    // it is exactly what produced the seq-0 bug. Reporting false makes the
    // supervisor refuse a multi-sequence suspend rather than half-perform one.
    return false;
}

bool LlamaKvBackend::save(const std::string& base_url,
                          const std::string& api_key,
                          std::uint32_t sequence,
                          const std::string& checkpoint_path,
                          std::string& out_error) {
    if (sequence != 0) {
        out_error = "llama-server checkpoints are limited to sequence 0; asked for " +
                    std::to_string(sequence);
        return false;
    }
    auto cli = engine_client(base_url, api_key);
    const auto body = nlohmann::json{{"filename", basename_of(checkpoint_path)}}.dump();
    auto res = cli.Post("/slots/0?action=save", body, "application/json");
    if (!res) {
        out_error = "no response from " + base_url;
        return false;
    }
    if (res->status != 200) {
        out_error = "HTTP " + std::to_string(res->status) + ": " + res->body;
        return false;
    }
    // llama-server reports success before the file is necessarily visible to us
    // if --slot-save-path points somewhere else. Verify, so a misconfigured
    // launch fails here rather than at the restore that happens hours later.
    std::error_code ec;
    if (!fs::exists(checkpoint_path, ec)) {
        out_error = "llama-server reported a save but nothing appeared at " + checkpoint_path +
                    " (is --slot-save-path pointing at the node's kv dir?)";
        return false;
    }
    return true;
}

bool LlamaKvBackend::restore(const std::string& base_url,
                             const std::string& api_key,
                             std::uint32_t sequence,
                             const std::string& checkpoint_path,
                             std::string& out_error) {
    if (sequence != 0) {
        out_error = "llama-server checkpoints are limited to sequence 0; asked for " +
                    std::to_string(sequence);
        return false;
    }
    std::error_code ec;
    if (!fs::exists(checkpoint_path, ec)) {
        out_error = "no checkpoint at " + checkpoint_path;
        return false;
    }
    auto cli = engine_client(base_url, api_key);
    const auto body = nlohmann::json{{"filename", basename_of(checkpoint_path)}}.dump();
    auto res = cli.Post("/slots/0?action=restore", body, "application/json");
    if (!res) {
        out_error = "no response from " + base_url;
        return false;
    }
    if (res->status != 200) {
        out_error = "HTTP " + std::to_string(res->status) + ": " + res->body;
        return false;
    }
    return true;
}

bool LlamaKvBackend::stat(const std::string& checkpoint_path,
                          KvCheckpointInfo& out,
                          std::string& out_error) {
    // No header parse. llama-server's session blob is versioned by llama.cpp and
    // validated on its own restore, and inventing a reader for a format we do not
    // own would be a decoder that goes stale on their next bump. `arch_hash` stays
    // empty, which the interface documents — the pre-spawn check llama.cpp
    // supports is existence and non-emptiness, and claiming more would be false
    // confidence.
    if (!file_stat(checkpoint_path, out, out_error)) return false;
    out.sequence = 0;
    out.format_id = 0;
    return true;
}

bool LlamaKvBackend::remove(const std::string& checkpoint_path, std::string& out_error) {
    return file_remove(checkpoint_path, out_error);
}

// ── SomaKvBackend ─────────────────────────────────────────────────────────────

const char* SomaKvBackend::engine_id() const {
    return "soma";
}

const char* SomaKvBackend::file_extension() const {
    // NOT kv_checkpoint_extension(). That is the extension of ONE sequence's KV,
    // which the engine writes and reads. The node's artifact is a manifest naming
    // every session in the engine — which is what supports_multi_sequence()
    // returning true has to mean if it is to mean anything.
    return ".somasession";
}

bool SomaKvBackend::supports_multi_sequence() const {
    return true;
}

bool SomaKvBackend::save(const std::string& base_url,
                         const std::string& api_key,
                         std::uint32_t,
                         const std::string& checkpoint_path,
                         std::string& out_error) {
    // `sequence` is ignored on purpose, and this is the substantive difference
    // from the fallback. llama.cpp is asked for one sequence and silently gives
    // you sequence 0; Soma is asked for the engine and writes all of them.
    auto cli = engine_client(base_url, api_key);
    const auto body = nlohmann::json{{"path", checkpoint_path}}.dump();
    auto res = cli.Post("/internal/kv/save", body, "application/json");
    if (!res) {
        out_error = "no response from " + base_url;
        return false;
    }
    if (res->status != 200) {
        out_error = "HTTP " + std::to_string(res->status) + ": " + res->body;
        return false;
    }
    std::error_code ec;
    if (!fs::exists(checkpoint_path, ec)) {
        out_error = "engine reported a save but nothing appeared at " + checkpoint_path;
        return false;
    }
    return true;
}

bool SomaKvBackend::restore(const std::string& base_url,
                            const std::string& api_key,
                            std::uint32_t,
                            const std::string& checkpoint_path,
                            std::string& out_error) {
    std::error_code ec;
    if (!fs::exists(checkpoint_path, ec)) {
        out_error = "no manifest at " + checkpoint_path;
        return false;
    }
    auto cli = engine_client(base_url, api_key);
    const auto body = nlohmann::json{{"path", checkpoint_path}}.dump();
    auto res = cli.Post("/internal/kv/restore", body, "application/json");
    if (!res) {
        out_error = "no response from " + base_url;
        return false;
    }
    if (res->status != 200) {
        out_error = "HTTP " + std::to_string(res->status) + ": " + res->body;
        return false;
    }
    return true;
}

bool SomaKvBackend::stat(const std::string& checkpoint_path,
                         KvCheckpointInfo& out,
                         std::string& out_error) {
    // Runs BEFORE an engine is spawned — the whole point, since rejecting a
    // cross-architecture resume after a 60-second model load is the confusing
    // version of that error. The manifest carries arch_hash for exactly this.
    if (!file_stat(checkpoint_path, out, out_error)) return false;

    std::ifstream in(checkpoint_path, std::ios::binary);
    nlohmann::json m;
    try {
        in >> m;
    } catch (const std::exception& e) {
        out_error = checkpoint_path + ": not a Soma session manifest (" + e.what() + ")";
        return false;
    }
    if (m.value("engine", std::string{}) != "soma") {
        out_error = checkpoint_path + ": manifest is not a Soma one";
        return false;
    }
    out.arch_hash = m.value("arch_hash", std::string{});
    out.length_tokens = m.value("total_tokens", 0u);
    out.sequence = static_cast<std::uint32_t>(m.value("sessions", nlohmann::json::array()).size());
    if (const auto ms = m.value("written_at_ms", std::uint64_t{0}); ms != 0) {
        out.written_at_ms = static_cast<std::int64_t>(ms);
    }
    return true;
}

bool SomaKvBackend::stat_sequence(const std::string& checkpoint_path,
                                  KvCheckpointInfo& out,
                                  std::string& out_error) {
    // One session's KV, read through the codec both binaries link. Separate from
    // stat() because they answer different questions: this one is "is this cache
    // replayable here", stat() is "what does this suspended engine contain".
    if (!file_stat(checkpoint_path, out, out_error)) return false;

    soma::KvCheckpointHeader header;
    if (auto st = soma::read_kv_checkpoint_header(checkpoint_path, header); !st.ok()) {
        out_error = st.message();
        return false;
    }
    out.arch_hash = header.arch_hash;
    out.format_id = header.format_id;
    out.length_tokens = header.length_tokens;
    if (header.written_at_ms != 0) {
        out.written_at_ms = static_cast<std::int64_t>(header.written_at_ms);
    }
    return true;
}

bool SomaKvBackend::remove(const std::string& checkpoint_path, std::string& out_error) {
    return file_remove(checkpoint_path, out_error);
}

} // namespace mm
