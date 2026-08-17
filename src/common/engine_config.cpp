#include "common/engine_config.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace mm {

namespace {

bool one_of(const std::string& value, std::initializer_list<const char*> allowed) {
    return std::any_of(allowed.begin(), allowed.end(), [&](const char* a) { return value == a; });
}

} // namespace

std::string EngineArtifact::fingerprint() const {
    // Fixed field order and a separator that cannot appear in any field, so the
    // string can be parsed back. `variant` is legitimately empty for an engine
    // with no build variants and keeps its slot rather than collapsing, or
    // "soma|1|linux|x86_64|" and "soma|1|linux||x86_64" would be confusable.
    return engine_id + "|" + version + "|" + platform + "|" + arch + "|" + variant;
}

bool EngineArtifact::valid() const noexcept {
    return !engine_id.empty() && !version.empty() && !platform.empty() && !arch.empty();
}

bool parse_engine_fingerprint(const std::string& fingerprint, EngineArtifact& out) {
    std::vector<std::string> parts;
    std::string cur;
    for (const char c : fingerprint) {
        if (c == '|') {
            parts.push_back(cur);
            cur.clear();
        } else
            cur.push_back(c);
    }
    parts.push_back(cur);
    if (parts.size() != 5) return false;

    EngineArtifact a;
    a.engine_id = parts[0];
    a.version = parts[1];
    a.platform = parts[2];
    a.arch = parts[3];
    a.variant = parts[4];
    if (!a.valid()) return false;
    out = a;
    return true;
}

const EngineSpec* ClusterEngineConfig::find(const std::string& engine_id) const noexcept {
    for (const auto& s : engines)
        if (s.engine_id == engine_id) return &s;
    return nullptr;
}

std::vector<std::string> ClusterEngineConfig::required_engines() const {
    std::vector<std::string> out;
    if (!primary_engine.empty()) out.push_back(primary_engine);
    // Skipped when empty, which is the whole mechanism behind "the backup is
    // optional": a node provisions required_engines() and nothing else, so an
    // empty backup is not a llama.cpp build that is merely unused — it is a
    // llama.cpp build that never happens.
    if (!backup_engine.empty()) out.push_back(backup_engine);
    return out;
}

bool conformance_permits_placement(const EngineConformance& c) noexcept {
    return c.state == EngineConformanceState::Conforming;
}

const std::vector<std::string>& forbidden_config_keys() {
    // Per-machine facts. Present here, they would be a cluster-wide claim about
    // hardware the master cannot see. See engine_config.hpp's invariant.
    static const std::vector<std::string> keys = {
        "accelerator",
        "cuda_arch",
        "cuda_architecture",
        "executable",
        "executable_path",
        "provision_dir",
        "llama_server_path",
        "soma_path",
        "variant",
    };
    return keys;
}

bool validate_engine_config(const ClusterEngineConfig& cfg,
                            const std::vector<std::string>& known_engine_ids,
                            std::string& out_error) {
    out_error.clear();

    if (cfg.primary_engine.empty()) {
        out_error = "primary_engine is required";
        return false;
    }
    if (!cfg.backup_engine.empty() && cfg.backup_engine == cfg.primary_engine) {
        out_error = "backup_engine must differ from primary_engine ('" + cfg.primary_engine +
                    "'); leave it empty for no backup";
        return false;
    }

    std::unordered_set<std::string> seen;
    for (const auto& s : cfg.engines) {
        if (s.engine_id.empty()) {
            out_error = "an engine spec has an empty engine_id";
            return false;
        }
        if (!seen.insert(s.engine_id).second) {
            out_error = "duplicate engine spec for '" + s.engine_id + "'";
            return false;
        }
        if (!one_of(s.install_method, {"auto", "release", "source", "path"})) {
            out_error = "engine '" + s.engine_id + "': invalid install_method '" +
                        s.install_method + "' (expected auto|release|source|path)";
            return false;
        }
        if (!one_of(s.update_policy, {"prompt", "auto", "manual"})) {
            out_error = "engine '" + s.engine_id + "': invalid update_policy '" + s.update_policy +
                        "' (expected prompt|auto|manual)";
            return false;
        }
        if (s.build_jobs < 0) {
            out_error = "engine '" + s.engine_id + "': build_jobs must be >= 0";
            return false;
        }
        if (s.update_check_interval_hours < 0) {
            out_error = "engine '" + s.engine_id + "': update_check_interval_hours must be >= 0";
            return false;
        }
        // Checked only when the caller supplied a registry. An unknown id is a
        // configuration the cluster cannot satisfy on ANY node, so it is worth
        // refusing at the write rather than discovering per node.
        if (!known_engine_ids.empty() &&
            std::find(known_engine_ids.begin(), known_engine_ids.end(), s.engine_id) ==
                known_engine_ids.end()) {
            out_error = "unknown engine '" + s.engine_id + "'";
            return false;
        }
    }

    // Named-but-unspecified is the failure that would otherwise surface as a
    // node silently provisioning defaults nobody chose.
    for (const auto& id : cfg.required_engines()) {
        if (cfg.find(id) == nullptr) {
            out_error = "engine '" + id + "' is named but has no spec";
            return false;
        }
    }
    return true;
}

const char* to_string(EngineConformanceState state) noexcept {
    switch (state) {
    case EngineConformanceState::Unconfigured:
        return "unconfigured";
    case EngineConformanceState::Converging:
        return "converging";
    case EngineConformanceState::Conforming:
        return "conforming";
    case EngineConformanceState::Drifted:
        return "drifted";
    case EngineConformanceState::Failed:
        return "failed";
    }
    return "unconfigured";
}

bool parse_conformance_state(const std::string& s, EngineConformanceState& out) noexcept {
    if (s == "unconfigured") {
        out = EngineConformanceState::Unconfigured;
        return true;
    }
    if (s == "converging") {
        out = EngineConformanceState::Converging;
        return true;
    }
    if (s == "conforming") {
        out = EngineConformanceState::Conforming;
        return true;
    }
    if (s == "drifted") {
        out = EngineConformanceState::Drifted;
        return true;
    }
    if (s == "failed") {
        out = EngineConformanceState::Failed;
        return true;
    }
    return false;
}

void to_json(nlohmann::json& j, const EngineSpec& s) {
    j = nlohmann::json{{"engine_id", s.engine_id},
                       {"version", s.version},
                       {"install_method", s.install_method},
                       {"update_policy", s.update_policy},
                       {"update_check", s.update_check},
                       {"update_check_interval_hours", s.update_check_interval_hours},
                       {"cmake_args", s.cmake_args},
                       {"build_jobs", s.build_jobs}};
}

void from_json(const nlohmann::json& j, EngineSpec& s) {
    if (j.contains("engine_id")) j.at("engine_id").get_to(s.engine_id);
    if (j.contains("version")) j.at("version").get_to(s.version);
    if (j.contains("install_method")) j.at("install_method").get_to(s.install_method);
    if (j.contains("update_policy")) j.at("update_policy").get_to(s.update_policy);
    if (j.contains("update_check")) j.at("update_check").get_to(s.update_check);
    if (j.contains("update_check_interval_hours"))
        j.at("update_check_interval_hours").get_to(s.update_check_interval_hours);
    if (j.contains("cmake_args")) j.at("cmake_args").get_to(s.cmake_args);
    if (j.contains("build_jobs")) j.at("build_jobs").get_to(s.build_jobs);
}

void to_json(nlohmann::json& j, const ClusterEngineConfig& c) {
    j = nlohmann::json{{"version", c.version},
                       {"primary_engine", c.primary_engine},
                       {"backup_engine", c.backup_engine},
                       {"engines", c.engines},
                       {"share_builds", c.share_builds},
                       {"updated_at_ms", c.updated_at_ms},
                       {"updated_by", c.updated_by}};
}

void from_json(const nlohmann::json& j, ClusterEngineConfig& c) {
    // Refuse a per-machine key ANYWHERE in the document — top level or inside a
    // spec. Ignoring it is the worse failure: the write is accepted, the
    // operator believes the cluster was told, and every node drifts against a
    // setting that was silently discarded.
    const auto& forbidden = forbidden_config_keys();
    const auto reject = [&](const nlohmann::json& obj, const char* where) {
        if (!obj.is_object()) return;
        for (const auto& key : forbidden) {
            if (obj.contains(key)) {
                throw std::invalid_argument(
                    std::string("'") + key + "' may not appear in the cluster engine config" +
                    where + ": it is a per-machine fact each node resolves from its own hardware");
            }
        }
    };
    reject(j, "");
    if (j.contains("engines") && j.at("engines").is_array()) {
        for (const auto& e : j.at("engines"))
            reject(e, " (in an engine spec)");
    }

    if (j.contains("version")) j.at("version").get_to(c.version);
    if (j.contains("primary_engine")) j.at("primary_engine").get_to(c.primary_engine);
    if (j.contains("backup_engine")) j.at("backup_engine").get_to(c.backup_engine);
    if (j.contains("engines")) j.at("engines").get_to(c.engines);
    if (j.contains("share_builds")) j.at("share_builds").get_to(c.share_builds);
    if (j.contains("updated_at_ms")) j.at("updated_at_ms").get_to(c.updated_at_ms);
    if (j.contains("updated_by")) j.at("updated_by").get_to(c.updated_by);
}

void to_json(nlohmann::json& j, const EngineConformance& c) {
    j = nlohmann::json{{"state", to_string(c.state)},
                       {"config_version", c.config_version},
                       {"detail", c.detail},
                       {"needs_artifact", c.needs_artifact}};
}

void to_json(nlohmann::json& j, const RuntimeStatus& s) {
    j = nlohmann::json{{"engine_id", s.engine_id},
                       {"status", s.status},
                       {"executable_path", s.executable_path},
                       {"version", s.version},
                       {"variant", s.variant},
                       {"last_error", s.last_error},
                       {"ready", s.ready}};
}

void from_json(const nlohmann::json& j, RuntimeStatus& s) {
    if (j.contains("engine_id")) j.at("engine_id").get_to(s.engine_id);
    if (j.contains("status")) j.at("status").get_to(s.status);
    if (j.contains("executable_path")) j.at("executable_path").get_to(s.executable_path);
    if (j.contains("version")) j.at("version").get_to(s.version);
    if (j.contains("variant")) j.at("variant").get_to(s.variant);
    if (j.contains("last_error")) j.at("last_error").get_to(s.last_error);
    if (j.contains("ready")) j.at("ready").get_to(s.ready);
}

void from_json(const nlohmann::json& j, EngineConformance& c) {
    if (j.contains("state") && j.at("state").is_string()) {
        // An unrecognized state leaves the default (Unconfigured), which stops
        // placement. A node speaking a vocabulary this build does not know is
        // exactly the node not to schedule onto.
        parse_conformance_state(j.at("state").get<std::string>(), c.state);
    }
    if (j.contains("config_version")) j.at("config_version").get_to(c.config_version);
    if (j.contains("detail")) j.at("detail").get_to(c.detail);
    if (j.contains("needs_artifact")) j.at("needs_artifact").get_to(c.needs_artifact);
}

} // namespace mm
