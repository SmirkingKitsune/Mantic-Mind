#pragma once

// Mantic-Mind — the master's engine configuration, persisted.
//
// The cluster's engine policy has exactly one home, and this is it. Nodes hold
// a copy of what they were last told; only this is authoritative.
//
// ── Why a JSON file and not the .toml ─────────────────────────────────────────
//
// ConfigFile is a read-only TOML SUBSET with no writer (common/config_file.hpp),
// so a configurator that wrote back to mantic-mind-control.toml would have to
// grow a TOML emitter and would destroy the file's comments on every save. The
// remembered-nodes journal in NodeRegistry already establishes the alternative:
// operator-mutable state lives as JSON under data_dir, and hand-edited
// configuration stays in the .toml. This follows that split.
//
// ── Absence is the unconfigured signal ────────────────────────────────────────
//
// No file means no engine policy, which is what forces first-run setup. That
// makes a TRUNCATED file dangerous in a way the remembered-nodes journal is not:
// a half-written config reads as "never configured" and would silently re-run
// setup on a cluster that had already been configured. So the write is atomic —
// temp file, flush, rename — and a parse failure is reported rather than
// swallowed into a default.

#include "common/engine_config.hpp"

#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace mm {

class EngineConfigStore {
public:
    /// `data_dir` empty means memory-only: no load, no save, and configured()
    /// still answers honestly. Used by tests that care about the rules and not
    /// about the filesystem.
    explicit EngineConfigStore(std::string data_dir);

    /// Reads data/engine_config.json if present. Returns false ONLY on a
    /// present-but-unreadable file, with `out_error` populated — a missing file
    /// is the expected first-run state and returns true with configured()
    /// false.
    bool load(std::string& out_error);

    /// Has an engine policy ever been set?
    bool configured() const;

    ClusterEngineConfig get() const;

    /// The current version, cheaply. This is compared against every node's
    /// reported version on every health poll, so it does not copy the config.
    std::uint32_t version() const;

    /// Validate, bump `version`, stamp `updated_at_ms`/`updated_by`, persist,
    /// then fire the change callback.
    ///
    /// The version is assigned HERE and the caller's value is ignored: a client
    /// that echoed back a stale version would otherwise be able to make the
    /// cluster converge backwards. Returns false with `out_error` on validation
    /// or write failure, and in that case nothing is stored and no callback
    /// fires — a rejected write must not leave nodes chasing a config that was
    /// never accepted.
    bool save(const ClusterEngineConfig& cfg,
              const std::vector<std::string>& known_engine_ids,
              const std::string& updated_by,
              std::string& out_error);

    /// Fired after a successful save, with the stored config. Registered by
    /// control's startup so a change propagates to nodes immediately rather
    /// than waiting up to one health-poll interval.
    ///
    /// Invoked WITHOUT the internal lock held, so a callback may call back into
    /// the store (and may block on HTTP) without deadlocking.
    using ChangeCallback = std::function<void(const ClusterEngineConfig&)>;
    void set_change_callback(ChangeCallback cb);

    /// A sane starting point for the setup menu: the given primary, llama.cpp as
    /// backup when it is not itself the primary, and a default spec for each.
    /// Pure — it stores nothing.
    static ClusterEngineConfig default_for(const std::string& primary_engine);

    std::string path() const;

private:
    mutable std::mutex mutex_;
    std::string path_;
    ClusterEngineConfig config_;
    bool configured_ = false;
    ChangeCallback on_change_;

    bool write_locked(std::string& out_error) const;
};

} // namespace mm
