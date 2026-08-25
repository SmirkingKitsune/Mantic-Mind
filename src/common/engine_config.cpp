#include "common/engine_config.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace mm {

namespace {

bool one_of(const std::string& value, std::initializer_list<const char*> allowed) {
    return std::any_of(allowed.begin(), allowed.end(), [&](const char* a) { return value == a; });
}

bool arg_matches_flag(const std::string& raw, const std::string& flag) {
    const auto first = raw.find_first_not_of(" \t");
    if (first == std::string::npos) return false;
    const std::string arg = raw.substr(first);
    return arg == flag || arg.rfind(flag + "=", 0) == 0 ||
           arg.rfind(flag + " ", 0) == 0;
}

const std::vector<std::string>& managed_vllm_flags() {
    static const std::vector<std::string> flags = {
        "--model", "--host", "--port", "--served-model-name",
        "--max-model-len", "--max-num-seqs", "--max-num-batched-tokens",
        "--tensor-parallel-size", "--pipeline-parallel-size",
        "--distributed-executor-backend", "--gpu-memory-utilization",
        "--gpu_memory_utilization", "--dtype", "--quantization",
        "--trust-remote-code", "--enable-prefix-caching",
        "--no-enable-prefix-caching", "--enable-auto-tool-choice",
        "--no-enable-auto-tool-choice", "--enable-sleep-mode",
        "--no-enable-sleep-mode", "--tool-call-parser",
    };
    return flags;
}

} // namespace

bool vllm_launch_compatible(const VllmEngineConfig& a,
                            const VllmEngineConfig& b) noexcept {
    return a.max_model_len == b.max_model_len &&
           a.max_num_seqs == b.max_num_seqs &&
           a.max_num_batched_tokens == b.max_num_batched_tokens &&
           a.tensor_parallel_size == b.tensor_parallel_size &&
           a.pipeline_parallel_size == b.pipeline_parallel_size &&
           a.gpu_memory_utilization == b.gpu_memory_utilization &&
           a.dtype == b.dtype && a.quantization == b.quantization &&
           a.trust_remote_code == b.trust_remote_code &&
           a.enable_prefix_caching == b.enable_prefix_caching &&
           a.enable_auto_tool_choice == b.enable_auto_tool_choice &&
           a.enable_sleep_mode == b.enable_sleep_mode &&
           a.tool_call_parser == b.tool_call_parser &&
           a.extra_args == b.extra_args && a.ray_mode == b.ray_mode &&
           a.allow_experimental_gloo == b.allow_experimental_gloo;
}

VllmEngineConfig effective_vllm_config(const EngineSpec& spec) {
    return spec.vllm.value_or(VllmEngineConfig{});
}

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
        const bool install_ok = s.engine_id == "vllm"
            ? one_of(s.install_method, {"auto", "wheel", "source", "path"})
            : one_of(s.install_method, {"auto", "release", "source", "path"});
        if (!install_ok) {
            out_error = "engine '" + s.engine_id + "': invalid install_method '" +
                        s.install_method + "' (expected " +
                        (s.engine_id == "vllm" ? "auto|wheel|source|path" :
                                                  "auto|release|source|path") + ")";
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
        if (s.engine_id != "vllm" && s.vllm.has_value()) {
            out_error = "engine '" + s.engine_id +
                        "': vllm profile is only valid for engine_id 'vllm'";
            return false;
        }
        if (s.engine_id == "vllm") {
            const auto v = effective_vllm_config(s);
            if (v.max_model_len <= 0 || v.max_num_seqs <= 0) {
                out_error = "engine 'vllm': max_model_len and max_num_seqs must be > 0";
                return false;
            }
            if (v.max_num_batched_tokens != -1 && v.max_num_batched_tokens <= 0) {
                out_error = "engine 'vllm': max_num_batched_tokens must be -1 or > 0";
                return false;
            }
            if (v.tensor_parallel_size < 1 || v.pipeline_parallel_size < 1) {
                out_error = "engine 'vllm': tensor and pipeline parallel sizes must be >= 1";
                return false;
            }
            if (!std::isfinite(v.gpu_memory_utilization) ||
                v.gpu_memory_utilization <= 0.0 || v.gpu_memory_utilization > 1.0) {
                out_error = "engine 'vllm': gpu_memory_utilization must be in (0, 1]";
                return false;
            }
            if (v.dtype.empty()) {
                out_error = "engine 'vllm': dtype must not be empty";
                return false;
            }
            if (v.ray_mode != "automatic") {
                out_error = "engine 'vllm': ray.mode must be 'automatic'";
                return false;
            }
            for (const auto& arg : v.extra_args) {
                for (const auto& flag : managed_vllm_flags()) {
                    if (arg_matches_flag(arg, flag)) {
                        out_error = "engine 'vllm': extra_args may not override managed flag '" +
                                    flag + "'";
                        return false;
                    }
                }
            }
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
    if (s.vllm.has_value()) j["vllm"] = *s.vllm;
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
    if (j.contains("vllm") && !j.at("vllm").is_null())
        s.vllm = j.at("vllm").get<VllmEngineConfig>();
}

void to_json(nlohmann::json& j, const VllmEngineConfig& s) {
    j = nlohmann::json{
        {"max_model_len", s.max_model_len},
        {"max_num_seqs", s.max_num_seqs},
        {"max_num_batched_tokens", s.max_num_batched_tokens},
        {"tensor_parallel_size", s.tensor_parallel_size},
        {"pipeline_parallel_size", s.pipeline_parallel_size},
        {"gpu_memory_utilization", s.gpu_memory_utilization},
        {"dtype", s.dtype}, {"quantization", s.quantization},
        {"trust_remote_code", s.trust_remote_code},
        {"enable_prefix_caching", s.enable_prefix_caching},
        {"enable_auto_tool_choice", s.enable_auto_tool_choice},
        {"enable_sleep_mode", s.enable_sleep_mode},
        {"tool_call_parser", s.tool_call_parser},
        {"extra_args", s.extra_args},
        {"ray", {{"mode", s.ray_mode},
                 {"allow_experimental_gloo", s.allow_experimental_gloo}}},
    };
}

void from_json(const nlohmann::json& j, VllmEngineConfig& s) {
    if (j.contains("max_model_len")) j.at("max_model_len").get_to(s.max_model_len);
    if (j.contains("max_num_seqs")) j.at("max_num_seqs").get_to(s.max_num_seqs);
    if (j.contains("max_num_batched_tokens"))
        j.at("max_num_batched_tokens").get_to(s.max_num_batched_tokens);
    if (j.contains("tensor_parallel_size"))
        j.at("tensor_parallel_size").get_to(s.tensor_parallel_size);
    if (j.contains("pipeline_parallel_size"))
        j.at("pipeline_parallel_size").get_to(s.pipeline_parallel_size);
    if (j.contains("gpu_memory_utilization"))
        j.at("gpu_memory_utilization").get_to(s.gpu_memory_utilization);
    if (j.contains("dtype")) j.at("dtype").get_to(s.dtype);
    if (j.contains("quantization")) j.at("quantization").get_to(s.quantization);
    if (j.contains("trust_remote_code"))
        j.at("trust_remote_code").get_to(s.trust_remote_code);
    if (j.contains("enable_prefix_caching"))
        j.at("enable_prefix_caching").get_to(s.enable_prefix_caching);
    if (j.contains("enable_auto_tool_choice"))
        j.at("enable_auto_tool_choice").get_to(s.enable_auto_tool_choice);
    if (j.contains("enable_sleep_mode"))
        j.at("enable_sleep_mode").get_to(s.enable_sleep_mode);
    if (j.contains("tool_call_parser"))
        j.at("tool_call_parser").get_to(s.tool_call_parser);
    if (j.contains("extra_args")) j.at("extra_args").get_to(s.extra_args);
    if (j.contains("ray") && j.at("ray").is_object()) {
        const auto& ray = j.at("ray");
        if (ray.contains("mode")) ray.at("mode").get_to(s.ray_mode);
        if (ray.contains("allow_experimental_gloo"))
            ray.at("allow_experimental_gloo").get_to(s.allow_experimental_gloo);
    }
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
