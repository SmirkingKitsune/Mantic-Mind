#include "control/engine_config_store.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace mm {

namespace fs = std::filesystem;

EngineConfigStore::EngineConfigStore(std::string data_dir) {
    if (!data_dir.empty()) {
        path_ = (fs::path(data_dir) / "engine_config.json").string();
    }
}

std::string EngineConfigStore::path() const {
    return path_;
}

bool EngineConfigStore::load(std::string& out_error) {
    out_error.clear();
    std::lock_guard<std::mutex> g(mutex_);
    if (path_.empty()) return true;

    std::error_code ec;
    if (!fs::exists(path_, ec)) {
        // First run. Not an error — it is the signal that forces setup.
        MM_INFO("EngineConfigStore: no engine configuration at {} (first run)", path_);
        return true;
    }

    std::ifstream in(path_);
    if (!in.is_open()) {
        out_error = "cannot open " + path_;
        return false;
    }
    std::stringstream buf;
    buf << in.rdbuf();

    try {
        const auto j = nlohmann::json::parse(buf.str());
        ClusterEngineConfig cfg = j.get<ClusterEngineConfig>();
        // Validated WITHOUT a registry: this process may not have one yet, and
        // a stored config that fails structural validation should be reported
        // rather than silently replaced with a default the operator never
        // chose.
        std::string verr;
        if (!validate_engine_config(cfg, {}, verr)) {
            out_error = "stored engine configuration is invalid: " + verr;
            return false;
        }
        config_ = std::move(cfg);
        configured_ = true;
        MM_INFO("EngineConfigStore: loaded v{} — primary='{}' backup='{}'",
                config_.version,
                config_.primary_engine,
                config_.backup_engine.empty() ? "(none)" : config_.backup_engine);
        return true;
    } catch (const std::exception& e) {
        out_error = std::string("cannot parse ") + path_ + ": " + e.what();
        return false;
    }
}

bool EngineConfigStore::configured() const {
    std::lock_guard<std::mutex> g(mutex_);
    return configured_;
}

ClusterEngineConfig EngineConfigStore::get() const {
    std::lock_guard<std::mutex> g(mutex_);
    return config_;
}

std::uint32_t EngineConfigStore::version() const {
    std::lock_guard<std::mutex> g(mutex_);
    return configured_ ? config_.version : 0u;
}

void EngineConfigStore::set_change_callback(ChangeCallback cb) {
    std::lock_guard<std::mutex> g(mutex_);
    on_change_ = std::move(cb);
}

bool EngineConfigStore::save(const ClusterEngineConfig& cfg,
                             const std::vector<std::string>& known_engine_ids,
                             const std::string& updated_by,
                             std::string& out_error) {
    out_error.clear();

    ClusterEngineConfig stored;
    ChangeCallback cb;
    {
        std::lock_guard<std::mutex> g(mutex_);

        if (!validate_engine_config(cfg, known_engine_ids, out_error)) return false;

        stored = cfg;
        // Assigned here, never taken from the caller. A client echoing a stale
        // version back would otherwise be able to move the cluster backwards,
        // and every node compares on this number alone.
        stored.version = config_.version + 1;
        stored.updated_at_ms = util::now_ms();
        stored.updated_by = updated_by;

        const ClusterEngineConfig previous = config_;
        const bool was_configured = configured_;
        config_ = stored;
        configured_ = true;

        if (!write_locked(out_error)) {
            // Roll back so memory and disk cannot disagree. A store that
            // reports a version it failed to persist would hand nodes a config
            // that vanishes on restart.
            config_ = previous;
            configured_ = was_configured;
            return false;
        }
        cb = on_change_;
    }

    MM_INFO("EngineConfigStore: saved v{} by '{}' — primary='{}' backup='{}' share_builds={}",
            stored.version,
            updated_by,
            stored.primary_engine,
            stored.backup_engine.empty() ? "(none)" : stored.backup_engine,
            stored.share_builds);

    // Outside the lock: this fans out to every node over HTTP.
    if (cb) cb(stored);
    return true;
}

bool EngineConfigStore::write_locked(std::string& out_error) const {
    if (path_.empty()) return true; // memory-only

    std::error_code ec;
    const fs::path path(path_);
    if (path.has_parent_path()) fs::create_directories(path.parent_path(), ec);

    // Temp-then-rename. A truncated file would parse as absent, which reads as
    // "never configured" and would re-run first-run setup on a configured
    // cluster.
    const fs::path temp = path.string() + ".tmp";
    {
        std::ofstream out(temp, std::ios::trunc | std::ios::binary);
        if (!out.is_open()) {
            out_error = "cannot write " + temp.string();
            return false;
        }
        const nlohmann::json root = config_;
        out << root.dump(2) << '\n';
        out.flush();
        if (!out.good()) {
            out_error = "write failed for " + temp.string();
            return false;
        }
    }

    fs::rename(temp, path, ec);
    if (ec) {
        // Windows will not rename onto an existing file on every filesystem;
        // fall back to replace, then report honestly if that fails too.
        std::error_code rec;
        fs::remove(path, rec);
        fs::rename(temp, path, ec);
        if (ec) {
            out_error = "cannot replace " + path_ + ": " + ec.message();
            fs::remove(temp, rec);
            return false;
        }
    }
    return true;
}

ClusterEngineConfig EngineConfigStore::default_for(const std::string& primary_engine) {
    ClusterEngineConfig cfg;
    cfg.primary_engine = primary_engine;
    // The backup is a DEFAULT, not a floor. Setup offers clearing it, and an
    // empty value survives every later save.
    cfg.backup_engine =
        (primary_engine == kDefaultBackupEngine) ? std::string{} : kDefaultBackupEngine;

    EngineSpec primary;
    primary.engine_id = primary_engine;
    if (primary_engine == "vllm") primary.vllm = VllmEngineConfig{};
    cfg.engines.push_back(primary);

    if (!cfg.backup_engine.empty()) {
        EngineSpec backup;
        backup.engine_id = cfg.backup_engine;
        if (backup.engine_id == "vllm") backup.vllm = VllmEngineConfig{};
        cfg.engines.push_back(backup);
    }
    return cfg;
}

} // namespace mm
