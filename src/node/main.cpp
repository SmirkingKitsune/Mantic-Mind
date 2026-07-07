#include "common/config_file.hpp"
#include "common/cli_repl.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/node_discovery.hpp"
#include "common/util.hpp"
#include "node/node_api_server.hpp"
#include "node/node_config.hpp"
#include "node/hf_cache.hpp"
#include "node/model_store.hpp"
#include "node/node_state.hpp"
#include "node/node_ui.hpp"
#include "node/singleton_lock.hpp"
#include "node/slot_manager.hpp"
#include "node/vllm_provisioner.hpp"
#include "node/vllm_runtime.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

// ── Config loading ─────────────────────────────────────────────────────────────
// Priority: config file < environment variables.

static mm::NodeConfig load_config(std::string* loaded_cfg_path = nullptr,
                                  const std::string& cfg_name = "mantic-mind.toml") {
    mm::NodeConfig cfg;
    mm::ConfigFile file;

    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;
    std::vector<std::string> seen;
    auto add_candidate = [&](const fs::path& p) {
        if (p.empty()) return;
        auto s = p.lexically_normal().string();
        if (std::find(seen.begin(), seen.end(), s) != seen.end()) return;
        seen.push_back(s);
        candidates.push_back(p);
    };

    const char* node_cfg_env = std::getenv("MM_NODE_CONFIG_FILE");
    if (node_cfg_env && *node_cfg_env) add_candidate(fs::path(node_cfg_env));
    const char* cfg_env = std::getenv("MM_CONFIG_FILE");
    if (cfg_env && *cfg_env) add_candidate(fs::path(cfg_env));

    std::error_code ec;
    fs::path dir = fs::current_path(ec);
    if (!ec) {
        for (int i = 0; i < 10; ++i) {
            add_candidate(dir / cfg_name);
            if (!dir.has_parent_path()) break;
            auto parent = dir.parent_path();
            if (parent == dir) break;
            dir = parent;
        }
    }

    std::string loaded_from;
    for (const auto& c : candidates) {
        if (file.load(c.string())) {
            loaded_from = c.string();
            break;
        }
    }

    if (loaded_cfg_path) *loaded_cfg_path = loaded_from;

    if (!loaded_from.empty()) {
        cfg.control_url       = file.get("control_url",       "");
        cfg.control_api_key   = file.get("control_api_key",   "");
        cfg.vllm_server_path  = file.get("vllm_server_path",  "vllm");
        cfg.vllm_auto_provision = file.get_bool("vllm_auto_provision", true);
        cfg.vllm_provision_dir = file.get("vllm_provision_dir", "");
        cfg.vllm_install_method = file.get("vllm_install_method", "auto");
        cfg.vllm_version = file.get("vllm_version", "latest");
        cfg.vllm_python_path = file.get("vllm_python_path", "");
        cfg.vllm_update_policy = file.get("vllm_update_policy", "prompt");
        cfg.vllm_update_check = file.get_bool("vllm_update_check", true);
        cfg.vllm_update_check_interval_hours =
            file.get_int("vllm_update_check_interval_hours", 24);
        cfg.models_dir        = file.get("models_dir",        "models");
        cfg.data_dir          = file.get("data_dir",          "data");
        cfg.log_file          = file.get("log_file", "logs/mantic-mind.log");
        cfg.listen_port = static_cast<uint16_t>(
            file.get_int("listen_port", 7070));
        cfg.runtime_port_range_start = static_cast<uint16_t>(
            file.get_int("runtime_port_range_start", 8080));
        cfg.runtime_port_range_end = static_cast<uint16_t>(
            file.get_int("runtime_port_range_end", cfg.runtime_port_range_start + 10));
        cfg.max_slots    = file.get_int("max_slots", 4);
        cfg.vllm_gpu_budget = static_cast<double>(
            file.get_float("vllm_gpu_budget", 0.90f));
        cfg.comm_backends   = file.get("comm_backends", "");
        if (file.has("supports_ray")) {
            cfg.supports_ray     = file.get_bool("supports_ray", false);
            cfg.supports_ray_set = true;
        }
        cfg.node_gpu_count    = file.get_int("node_gpu_count", 0);
        cfg.interconnect_gbps = static_cast<double>(
            file.get_float("interconnect_gbps", 0.0f));
        cfg.ray_path = file.get("ray_path", "ray");
        cfg.ray_port = static_cast<uint16_t>(file.get_int("ray_port", 6379));
        cfg.hf_cli_path = file.get("hf_cli_path", "hf");
        cfg.hf_cache_dir = file.get("hf_cache_dir", "");
        cfg.kv_cache_dir = file.get("kv_cache_dir", "data/kv_cache");
        cfg.model_cache_min_free_mb = static_cast<int64_t>(
            file.get_int("model_cache_min_free_mb",
                         static_cast<int>(cfg.model_cache_min_free_mb)));
        cfg.model_cache_clear_on_shutdown =
            file.get_bool("model_cache_clear_on_shutdown", cfg.model_cache_clear_on_shutdown);
        cfg.pairing_key    = file.get("pairing_key",    "");
        cfg.discovery_port = static_cast<uint16_t>(
            file.get_int("discovery_port", 7072));
    }

    // Environment variables override file values.
    auto env = [](const char* name, const std::string& cur) -> std::string {
        const char* v = std::getenv(name);
        if (!v) return cur;
        std::string s(v);
        return s.empty() ? cur : s;
    };
    auto env_port = [](const char* name, uint16_t cur) -> uint16_t {
        const char* v = std::getenv(name);
        if (!v) return cur;
        try { return static_cast<uint16_t>(std::stoi(v)); } catch (...) { return cur; }
    };
    auto env_bool = [](const char* name, bool cur) -> bool {
        const char* v = std::getenv(name);
        if (!v) return cur;
        const std::string s = mm::util::to_lower(mm::util::trim(v));
        if (s == "1" || s == "true" || s == "yes") return true;
        if (s == "0" || s == "false" || s == "no") return false;
        return cur;
    };

    cfg.control_url       = env("MM_CONTROL_URL",     cfg.control_url);
    cfg.control_api_key   = env("MM_CONTROL_API_KEY", cfg.control_api_key);
    cfg.vllm_server_path  = env("MM_VLLM_PATH",       cfg.vllm_server_path);
    cfg.vllm_auto_provision = env_bool("MM_VLLM_AUTO_PROVISION", cfg.vllm_auto_provision);
    cfg.vllm_provision_dir = env("MM_VLLM_PROVISION_DIR", cfg.vllm_provision_dir);
    cfg.vllm_install_method = env("MM_VLLM_INSTALL_METHOD", cfg.vllm_install_method);
    cfg.vllm_version = env("MM_VLLM_VERSION", cfg.vllm_version);
    cfg.vllm_python_path = env("MM_VLLM_PYTHON_PATH", cfg.vllm_python_path);
    cfg.vllm_update_policy = env("MM_VLLM_UPDATE_POLICY", cfg.vllm_update_policy);
    cfg.vllm_update_check = env_bool("MM_VLLM_UPDATE_CHECK", cfg.vllm_update_check);
    cfg.models_dir        = env("MM_MODELS_DIR",      cfg.models_dir);
    cfg.data_dir          = env("MM_DATA_DIR",        cfg.data_dir);
    cfg.log_file          = env("MM_LOG_FILE",        cfg.log_file);
    cfg.kv_cache_dir      = env("MM_KV_CACHE_DIR",   cfg.kv_cache_dir);
    cfg.pairing_key       = env("MM_PAIRING_KEY",     cfg.pairing_key);
    cfg.listen_port       = env_port("MM_LISTEN_PORT",    cfg.listen_port);
    cfg.discovery_port    = env_port("MM_DISCOVERY_PORT", cfg.discovery_port);

    auto env_int = [](const char* name, int cur) -> int {
        const char* v = std::getenv(name);
        if (!v) return cur;
        try { return std::stoi(v); } catch (...) { return cur; }
    };
    cfg.runtime_port_range_start = env_port("MM_RUNTIME_PORT_RANGE_START", cfg.runtime_port_range_start);
    cfg.runtime_port_range_end   = env_port("MM_RUNTIME_PORT_RANGE_END",   cfg.runtime_port_range_end);
    cfg.max_slots              = env_int("MM_MAX_SLOTS", cfg.max_slots);
    cfg.vllm_update_check_interval_hours =
        env_int("MM_VLLM_UPDATE_CHECK_INTERVAL_HOURS", cfg.vllm_update_check_interval_hours);
    cfg.model_cache_min_free_mb = static_cast<int64_t>(
        env_int("MM_MODEL_CACHE_MIN_FREE_MB",
                static_cast<int>(cfg.model_cache_min_free_mb)));
    cfg.model_cache_clear_on_shutdown =
        env_bool("MM_MODEL_CACHE_CLEAR_ON_SHUTDOWN", cfg.model_cache_clear_on_shutdown);

    auto env_double = [](const char* name, double cur) -> double {
        const char* v = std::getenv(name);
        if (!v) return cur;
        try { return std::stod(v); } catch (...) { return cur; }
    };
    cfg.vllm_gpu_budget = env_double("MM_VLLM_GPU_BUDGET", cfg.vllm_gpu_budget);
    cfg.comm_backends   = env("MM_COMM_BACKENDS", cfg.comm_backends);
    cfg.node_gpu_count  = env_int("MM_NODE_GPU_COUNT", cfg.node_gpu_count);
    cfg.interconnect_gbps = env_double("MM_INTERCONNECT_GBPS", cfg.interconnect_gbps);
    cfg.ray_path = env("MM_RAY_PATH", cfg.ray_path);
    cfg.ray_port = env_port("MM_RAY_PORT", cfg.ray_port);
    cfg.hf_cli_path = env("MM_HF_CLI_PATH", cfg.hf_cli_path);
    cfg.hf_cache_dir = env("MM_HF_CACHE_DIR", cfg.hf_cache_dir);
    if (const char* sr = std::getenv("MM_SUPPORTS_RAY")) {
        const std::string v = mm::util::to_lower(sr);
        cfg.supports_ray     = (v == "1" || v == "true" || v == "yes");
        cfg.supports_ray_set = true;
    }

    // Keep node startup resilient if env/config accidentally provide empty strings.
    if (cfg.vllm_server_path.empty()) cfg.vllm_server_path = "vllm";
    if (cfg.vllm_install_method.empty()) cfg.vllm_install_method = "auto";
    cfg.vllm_install_method = mm::normalize_vllm_install_method(cfg.vllm_install_method);
    if (cfg.vllm_version.empty()) cfg.vllm_version = "latest";
    {
        const std::string policy = mm::util::to_lower(mm::util::trim(cfg.vllm_update_policy));
        cfg.vllm_update_policy =
            (policy == "auto" || policy == "manual") ? policy : "prompt";
    }
    if (cfg.vllm_update_check_interval_hours < 1) cfg.vllm_update_check_interval_hours = 24;
    if (cfg.models_dir.empty()) cfg.models_dir = "models";
    if (cfg.model_cache_min_free_mb < 0) cfg.model_cache_min_free_mb = 0;
    if (cfg.data_dir.empty()) cfg.data_dir = "data";
    if (cfg.vllm_provision_dir.empty())
        cfg.vllm_provision_dir =
            (std::filesystem::path(cfg.data_dir) / "runtimes" / "vllm").string();
    if (cfg.kv_cache_dir.empty()) cfg.kv_cache_dir = "data/kv_cache";
    if (cfg.log_file.empty()) cfg.log_file = "logs/mantic-mind.log";
    if (cfg.ray_path.empty()) cfg.ray_path = "ray";
    if (cfg.hf_cli_path.empty()) cfg.hf_cli_path = "hf";
    if (cfg.vllm_gpu_budget <= 0.0 || cfg.vllm_gpu_budget > 1.0) cfg.vllm_gpu_budget = 0.90;

    return cfg;
}

namespace {

// Run a command, capturing trimmed stdout (first line). Empty on failure.
std::string capture_command_line(const std::string& cmd) {
#ifdef _WIN32
    FILE* f = _popen((cmd + " 2>nul").c_str(), "r");
#else
    FILE* f = ::popen((cmd + " 2>/dev/null").c_str(), "r");
#endif
    if (!f) return {};
    std::string out;
    char buf[512];
    while (fgets(buf, static_cast<int>(sizeof(buf)), f)) out += buf;
#ifdef _WIN32
    _pclose(f);
#else
    ::pclose(f);
#endif
    // Return the first non-empty line.
    for (auto& line : mm::util::split(out, '\n')) {
        const std::string t = mm::util::trim(line);
        if (!t.empty()) return t;
    }
    return {};
}

// Count GPUs visible via nvidia-smi (one CSV line per GPU).
int detect_gpu_count() {
    const std::string cmd =
        "nvidia-smi --query-gpu=index --format=csv,noheader,nounits";
#ifdef _WIN32
    FILE* f = _popen((cmd + " 2>nul").c_str(), "r");
#else
    FILE* f = ::popen((cmd + " 2>/dev/null").c_str(), "r");
#endif
    if (!f) return 0;
    int count = 0;
    char buf[256];
    while (fgets(buf, static_cast<int>(sizeof(buf)), f)) {
        if (!mm::util::trim(buf).empty()) ++count;
    }
#ifdef _WIN32
    _pclose(f);
#else
    ::pclose(f);
#endif
    return count;
}

// Build the capability block this node advertises. Detection fills the gaps
// the config left blank.
mm::NodeCapabilities detect_node_capabilities(const mm::NodeConfig& cfg,
                                              bool gpu_backend_available) {
    mm::NodeCapabilities caps;

#if defined(__aarch64__) || defined(_M_ARM64)
    caps.arch = "aarch64";
#elif defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
    caps.arch = "x86_64";
#else
    caps.arch = "";
#endif

    const std::string platform = mm::current_vllm_platform();
    const bool is_linux = platform == "linux";

    caps.gpu_count = cfg.node_gpu_count > 0 ? cfg.node_gpu_count : detect_gpu_count();
    caps.interconnect_gbps = cfg.interconnect_gbps;

    // Ray multi-node membership: Linux/WSL only in practice. Config wins.
    caps.supports_ray = cfg.supports_ray_set ? cfg.supports_ray : is_linux;

    // Comm backends: explicit CSV, else inferred. NCCL has no Windows build
    // yet (NVIDIA/nccl#1922 pending), so native Windows gets Gloo only; Linux
    // with a CUDA GPU gets NCCL + Gloo; CPU-only gets Gloo.
    if (!cfg.comm_backends.empty()) {
        for (auto& b : mm::util::split(cfg.comm_backends, ',')) {
            const std::string t = mm::util::to_lower(mm::util::trim(b));
            if (!t.empty()) caps.comm_backends.push_back(t);
        }
    } else {
        if (is_linux && gpu_backend_available && caps.gpu_count > 0)
            caps.comm_backends = {"nccl", "gloo"};
        else
            caps.comm_backends = {"gloo"};
    }

    // vLLM build fingerprint: `vllm --version` (e.g. "0.6.3").
    const std::string version_cmd =
        mm::util::trim(cfg.vllm_server_path).empty()
            ? std::string{}
            : "\"" + mm::util::trim(cfg.vllm_server_path) + "\" --version";
    if (!version_cmd.empty()) {
        const std::string v = capture_command_line(version_cmd);
        // Keep only a version-looking token if the line has extra words.
        if (!v.empty()) {
            for (auto& tok : mm::util::split(v, ' ')) {
                const std::string t = mm::util::trim(tok);
                if (!t.empty() && (std::isdigit(static_cast<unsigned char>(t[0])) ||
                                   t.find('.') != std::string::npos)) {
                    caps.vllm_version = t;
                    break;
                }
            }
            if (caps.vllm_version.empty()) caps.vllm_version = v;
        }
    }

    return caps;
}

std::filesystem::path remembered_api_keys_file(const mm::NodeConfig& cfg) {
    return std::filesystem::path(cfg.data_dir) / "api_keys.json";
}

std::vector<std::string> load_remembered_api_keys(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) return {};

    std::vector<std::string> out;
    try {
        auto root = nlohmann::json::parse(in);
        const auto& keys_json = root.is_array() ? root : root.at("keys");
        for (const auto& item : keys_json) {
            std::string key = item.get<std::string>();
            key = mm::util::trim(key);
            if (key.empty()) continue;
            if (std::find(out.begin(), out.end(), key) == out.end()) out.push_back(std::move(key));
        }
    } catch (const std::exception& e) {
        MM_WARN("Failed to load remembered API keys from {}: {}", path.string(), e.what());
    }
    return out;
}

bool save_remembered_api_keys(const std::filesystem::path& path,
                              const std::vector<std::string>& keys) {
    std::vector<std::string> unique;
    for (const auto& key : keys) {
        const std::string trimmed = mm::util::trim(key);
        if (trimmed.empty()) continue;
        if (std::find(unique.begin(), unique.end(), trimmed) == unique.end()) unique.push_back(trimmed);
    }

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) return false;

    out << nlohmann::json{{"version", 1}, {"keys", unique}}.dump(2) << '\n';
    return true;
}

} // namespace

// ── Registration helper ────────────────────────────────────────────────────────

static bool try_register(const mm::NodeConfig& cfg,
                          mm::NodeState& state,
                          const std::string& api_key) {
    const char* self_env = std::getenv("MM_SELF_URL");
    std::string self_url = self_env
        ? std::string(self_env)
        : "http://127.0.0.1:" + std::to_string(cfg.listen_port);

    const std::string platform = mm::current_vllm_platform();

    mm::HttpClient ctrl(cfg.control_url);
    if (!cfg.control_api_key.empty())
        ctrl.set_bearer_token(cfg.control_api_key);

    nlohmann::json body = {
        {"node_url", self_url},
        {"api_key",  api_key},
        {"platform", platform},
        {"gpu_info", ""}
    };

    MM_INFO("Attempting registration with control at {} ...", cfg.control_url);
    auto resp = ctrl.post("/api/control/register-node", body);
    if (!resp.ok()) {
        MM_WARN("Registration failed (HTTP {}): {}", resp.status, resp.body);
        return false;
    }

    try {
        auto j = nlohmann::json::parse(resp.body);
        if (j.value("accepted", false)) {
            std::string node_id = j.value("node_id", std::string{});
            state.set_registered(true, node_id);
            state.mark_control_contact();
            MM_INFO("Registered with control — node_id: {}", node_id);
            return true;
        }
        MM_WARN("Control rejected registration: {}", resp.body);
    } catch (const std::exception& e) {
        MM_WARN("Registration response parse error: {}", e.what());
    }
    return false;
}

enum class NodeRunMode { Tui, Cli };
enum class CliOutputMode { Text, Json };

struct NodeMainArgs {
    NodeRunMode mode = NodeRunMode::Tui;
    CliOutputMode output = CliOutputMode::Text;
    bool show_help = false;
    std::string error;
};

static NodeMainArgs parse_node_main_args(int argc, char** argv) {
    NodeMainArgs out;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i] ? argv[i] : "";
        if (arg == "--help" || arg == "-h") {
            out.show_help = true;
            continue;
        }
        if (arg == "--mode") {
            if (i + 1 >= argc) {
                out.error = "--mode requires a value: tui or cli";
                return out;
            }
            std::string value = mm::util::to_lower(argv[++i] ? argv[i] : "");
            if (value == "tui") out.mode = NodeRunMode::Tui;
            else if (value == "cli") out.mode = NodeRunMode::Cli;
            else out.error = "invalid --mode value '" + value + "' (expected tui|cli)";
            if (!out.error.empty()) return out;
            continue;
        }
        if (arg == "--output") {
            if (i + 1 >= argc) {
                out.error = "--output requires a value: text or json";
                return out;
            }
            std::string value = mm::util::to_lower(argv[++i] ? argv[i] : "");
            if (value == "text") out.output = CliOutputMode::Text;
            else if (value == "json") out.output = CliOutputMode::Json;
            else out.error = "invalid --output value '" + value + "' (expected text|json)";
            if (!out.error.empty()) return out;
            continue;
        }
        if (mm::util::starts_with(arg, "--mode=")) {
            std::string value = mm::util::to_lower(arg.substr(std::string("--mode=").size()));
            if (value == "tui") out.mode = NodeRunMode::Tui;
            else if (value == "cli") out.mode = NodeRunMode::Cli;
            else out.error = "invalid --mode value '" + value + "' (expected tui|cli)";
            if (!out.error.empty()) return out;
            continue;
        }
        if (mm::util::starts_with(arg, "--output=")) {
            std::string value = mm::util::to_lower(arg.substr(std::string("--output=").size()));
            if (value == "text") out.output = CliOutputMode::Text;
            else if (value == "json") out.output = CliOutputMode::Json;
            else out.error = "invalid --output value '" + value + "' (expected text|json)";
            if (!out.error.empty()) return out;
            continue;
        }
        out.error = "unknown argument: " + arg;
        return out;
    }
    return out;
}

static void print_node_usage() {
    std::cout
        << "Usage: mantic-mind [--mode tui|cli] [--output text|json] [--help]\n\n"
        << "Modes:\n"
        << "  tui  Default FTXUI terminal interface.\n"
        << "  cli  Interactive REPL suitable for terminal assistants.\n\n"
        << "Output:\n"
        << "  text Default human-readable CLI output.\n"
        << "  json Structured CLI output for automation.\n\n"
        << "CLI commands:\n"
        << "  status\n"
        << "  metrics\n"
        << "  slots\n"
        << "  models list\n"
        << "  api-key list\n"
        << "  api-key add <key>\n"
        << "  api-key remove <key>\n"
        << "  pair status\n"
        << "  logs tail [n]\n"
        << "  help\n"
        << "  quit\n";
}

class CliPrinter {
public:
    explicit CliPrinter(std::string prompt) : prompt_(std::move(prompt)) {}

    void print_prompt() {
        std::lock_guard<std::mutex> lk(mu_);
        prompt_visible_ = true;
        std::cout << prompt_ << std::flush;
    }

    void line(const std::string& text) {
        std::lock_guard<std::mutex> lk(mu_);
        if (prompt_visible_) std::cout << '\r';
        prompt_visible_ = false;
        std::cout << text << '\n';
    }

    void block(const std::string& text) {
        std::lock_guard<std::mutex> lk(mu_);
        if (prompt_visible_) std::cout << '\r';
        prompt_visible_ = false;
        std::cout << text;
        if (text.empty() || text.back() != '\n') std::cout << '\n';
    }

private:
    std::mutex mu_;
    std::string prompt_;
    bool prompt_visible_ = false;
};

static std::atomic<bool>* g_node_cli_stop = nullptr;

static void node_cli_signal_handler(int /*signal*/) {
    if (g_node_cli_stop) g_node_cli_stop->store(true);
}

static std::string summarize_http_error(const mm::HttpResponse& r) {
    std::string msg = "HTTP " + std::to_string(r.status);
    const std::string body = mm::util::trim(r.body);
    if (!body.empty()) msg += ": " + body;
    return msg;
}

static void run_node_cli(uint16_t listen_port,
                         const std::string& initial_key,
                         CliOutputMode output_mode,
                         std::atomic<bool>& stop_flag) {
    CliPrinter printer("mm-node> ");
    printer.line("CLI mode active. Type 'help' for commands.");

    mm::HttpClient self("http://127.0.0.1:" + std::to_string(listen_port));
    self.set_bearer_token(initial_key);
    const bool json_mode = output_mode == CliOutputMode::Json;

    auto print_help = [&]() {
        printer.line("Commands: status | metrics | slots | models list | "
                     "api-key list|add|remove | pair status | logs tail [n] | help | quit");
    };

    auto parse_or_wrap = [&](const std::string& body) -> nlohmann::json {
        try {
            return nlohmann::json::parse(body);
        } catch (...) {
            return nlohmann::json{{"raw", body}};
        }
    };

    auto emit_result = [&](bool ok, const std::string& command, const nlohmann::json& data, const std::string& error) {
        if (json_mode) {
            nlohmann::json out{{"ok", ok}, {"command", command}};
            if (ok) out["data"] = data;
            else out["error"] = error;
            printer.block(out.dump());
            return;
        }
        if (!ok) {
            printer.line("error: " + error);
            return;
        }
        if (data.is_null() || data.empty()) printer.line("ok");
        else if (data.is_string()) printer.line(data.get<std::string>());
        else printer.block(data.dump(2));
    };

    auto emit_http_result = [&](const std::string& command, const mm::HttpResponse& r) {
        if (r.ok()) emit_result(true, command, parse_or_wrap(r.body), "");
        else emit_result(false, command, nlohmann::json::object(), summarize_http_error(r));
    };

    while (!stop_flag.load()) {
        printer.print_prompt();
        std::string line;
        if (!std::getline(std::cin, line)) {
            printer.line("");
            break;
        }

        std::vector<std::string> tokens;
        std::string parse_error;
        if (!mm::cli::tokenize_command_line(line, &tokens, &parse_error)) {
            printer.line("error: " + parse_error);
            continue;
        }
        if (tokens.empty()) continue;

        const std::string cmd0 = mm::util::to_lower(tokens[0]);
        if (cmd0 == "quit" || cmd0 == "exit") break;
        if (cmd0 == "help") {
            print_help();
            continue;
        }

        if (cmd0 == "status") {
            auto r = self.get("/api/node/status");
            if (!r.ok()) {
                emit_result(false, "status", nlohmann::json::object(), summarize_http_error(r));
                continue;
            }
            auto j = parse_or_wrap(r.body);
            if (json_mode) {
                emit_result(true, "status", j, "");
            } else {
                std::string line_out =
                    "node_id=" + j.value("node_id", std::string{}) +
                    " slots=" + std::to_string(j.value("slot_in_use", 0)) + "/" + std::to_string(j.value("max_slots", 0));
                printer.line(line_out);
            }
            continue;
        }

        if (cmd0 == "metrics") {
            auto r = self.get("/api/node/health");
            if (!r.ok()) {
                emit_result(false, "metrics", nlohmann::json::object(), summarize_http_error(r));
                continue;
            }
            auto j = parse_or_wrap(r.body);
            if (json_mode) emit_result(true, "metrics", j, "");
            else {
                printer.line(
                    "cpu=" + std::to_string(j.value("cpu_percent", 0.0f)) +
                    "% ram=" + std::to_string(j.value("ram_percent", 0.0f)) +
                    "% (" + std::to_string(j.value("ram_used_mb", 0LL)) + "/" + std::to_string(j.value("ram_total_mb", 0LL)) + " MB)" +
                    " gpu=" + std::to_string(j.value("gpu_percent", 0.0f)) +
                    "% vram=" + std::to_string(j.value("gpu_vram_used_mb", 0LL)) + "/" + std::to_string(j.value("gpu_vram_total_mb", 0LL)) + " MB" +
                    " disk_free=" + std::to_string(j.value("disk_free_mb", 0LL)) + " MB" +
                    " gpu_backend=" + (j.value("gpu_backend_available", false) ? "yes" : "no"));
            }
            continue;
        }

        if (cmd0 == "slots") {
            auto r = self.get("/api/node/status");
            if (!r.ok()) {
                emit_result(false, "slots", nlohmann::json::object(), summarize_http_error(r));
                continue;
            }
            auto j = parse_or_wrap(r.body);
            if (json_mode) {
                emit_result(true, "slots", j.value("slots", nlohmann::json::array()), "");
                continue;
            }
            auto slots = j.value("slots", nlohmann::json::array());
            if (!slots.is_array() || slots.empty()) {
                printer.line("(no slots)");
                continue;
            }
            for (const auto& s : slots) {
                printer.line(
                    s.value("id", std::string{}) +
                    " state=" + s.value("state", std::string{}) +
                    " port=" + std::to_string(s.value("port", 0)) +
                    " model=" + (s.value("model_path", std::string{}).empty() ? "-" : s.value("model_path", std::string{})) +
                    " agent=" + (s.value("assigned_agent", std::string{}).empty() ? "-" : s.value("assigned_agent", std::string{})) +
                    " vram_mb=" + std::to_string(s.value("vram_usage_mb", 0LL)));
            }
            continue;
        }

        if (cmd0 == "models") {
            if (tokens.size() < 2 || mm::util::to_lower(tokens[1]) != "list") {
                printer.line("usage: models list");
                continue;
            }
            auto r = self.get("/api/node/models");
            if (!r.ok()) {
                emit_result(false, "models list", nlohmann::json::object(), summarize_http_error(r));
                continue;
            }
            auto j = parse_or_wrap(r.body);
            if (json_mode) {
                emit_result(true, "models list", j, "");
                continue;
            }
            auto models = j.value("models", nlohmann::json::array());
            if (!models.is_array() || models.empty()) {
                printer.line("(no models)");
                continue;
            }
            for (const auto& m : models) {
                printer.line(m.value("model_path", std::string{}) +
                             " size_bytes=" + std::to_string(m.value("size_bytes", 0LL)) +
                             " shards=" + std::to_string(m.value("shard_count", 0)));
            }
            continue;
        }

        if (cmd0 == "api-key") {
            if (tokens.size() < 2) {
                printer.line("usage: api-key list | api-key add <key> | api-key remove <key>");
                continue;
            }
            const std::string sub = mm::util::to_lower(tokens[1]);
            if (sub == "list") {
                emit_http_result("api-key list", self.get("/api/node/api-keys"));
                continue;
            }
            if (sub == "add") {
                if (tokens.size() < 3) {
                    printer.line("usage: api-key add <key>");
                    continue;
                }
                auto r = self.post("/api/node/api-keys", nlohmann::json{{"key", tokens[2]}});
                emit_http_result("api-key add", r);
                continue;
            }
            if (sub == "remove") {
                if (tokens.size() < 3) {
                    printer.line("usage: api-key remove <key>");
                    continue;
                }
                auto r = self.del("/api/node/api-keys/" + tokens[2]);
                emit_http_result("api-key remove", r);
                continue;
            }
            printer.line("error: unknown api-key subcommand");
            continue;
        }

        if (cmd0 == "pair") {
            if (tokens.size() >= 2 && mm::util::to_lower(tokens[1]) == "status") {
                emit_http_result("pair status", self.get("/api/node/pair-status"));
                continue;
            }
            printer.line("usage: pair status");
            continue;
        }

        if (cmd0 == "logs") {
            if (tokens.size() < 2 || mm::util::to_lower(tokens[1]) != "tail") {
                printer.line("usage: logs tail [n]");
                continue;
            }
            int n = 20;
            if (tokens.size() >= 3) {
                try {
                    n = std::stoi(tokens[2]);
                } catch (...) {
                    printer.line("error: n must be an integer");
                    continue;
                }
                if (n < 1) n = 1;
            }

            emit_http_result("logs tail", self.get("/api/node/logs?tail=" + std::to_string(n)));
            continue;
        }

        printer.line("error: unknown command. Type 'help'.");
    }
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    const NodeMainArgs args = parse_node_main_args(argc, argv);
    if (!args.error.empty()) {
        std::cerr << "ERROR: " << args.error << "\n\n";
        print_node_usage();
        return 1;
    }
    if (args.show_help) {
        print_node_usage();
        return 0;
    }

    // ── Singleton lock ────────────────────────────────────────────────────────
    auto instance_lock = mm::SingletonLock::try_acquire();
    if (!instance_lock) {
        std::cerr << "ERROR: Another mantic-mind node instance is already running.\n";
        return 1;
    }

    // Try config file first, then override with env vars.
    std::string cfg_path;
    auto cfg = load_config(&cfg_path);

    namespace fs = std::filesystem;
    {
        std::error_code ec;
        fs::create_directories(fs::path(cfg.log_file).parent_path(), ec);
        fs::create_directories(cfg.kv_cache_dir, ec);
    }

    mm::init_logger(
        cfg.log_file,
        "mm-node",
        args.mode == NodeRunMode::Cli ? spdlog::level::off : spdlog::level::info,
        spdlog::level::trace);
    MM_INFO("mantic-mind node starting — listen port {}, slots {}, port range {}-{}",
            cfg.listen_port, cfg.max_slots,
            cfg.runtime_port_range_start, cfg.runtime_port_range_end);
    MM_INFO("Node config source: {}",
            cfg_path.empty() ? "(defaults/env only; no config file found)" : cfg_path);
    MM_INFO("Node config — control_url='{}', models_dir='{}'",
            cfg.control_url, cfg.models_dir);
    MM_INFO("Node vLLM path: {}", cfg.vllm_server_path);

    // ── NodeState ─────────────────────────────────────────────────────────────

    mm::NodeState state;
    const std::string local_node_id = "local-" + mm::util::generate_uuid();
    state.set_registered(false, local_node_id);
    MM_INFO("Local node identity: {}", local_node_id);

    // Start metrics polling before provisioning so the GPU-backend probe
    // (nvidia-smi) has populated gpu_backend_available before we pick the
    // accelerator-correct vLLM build variant.
    state.start_metrics_poll(2000);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    mm::VllmProvisionConfig vllm_provision_cfg;
    vllm_provision_cfg.requested_executable = cfg.vllm_server_path;
    vllm_provision_cfg.provision_dir = cfg.vllm_provision_dir;
    vllm_provision_cfg.auto_provision = cfg.vllm_auto_provision;
    vllm_provision_cfg.install_method = cfg.vllm_install_method;
    vllm_provision_cfg.version = cfg.vllm_version;
    vllm_provision_cfg.python_path = cfg.vllm_python_path;
    vllm_provision_cfg.accelerator = mm::detect_vllm_accelerator(
        mm::current_vllm_platform(), mm::current_vllm_arch(),
        state.get_metrics().gpu_backend_available, mm::detect_rocm_present());
    mm::VllmProvisioner vllm_provisioner(vllm_provision_cfg);
    auto vllm_runtime = vllm_provisioner.ensure_runtime();
    state.set_vllm_runtime(vllm_runtime);
    if (mm::vllm_runtime_usable(vllm_runtime)) {
        cfg.vllm_server_path = vllm_runtime.executable_path;
        MM_INFO("Using vLLM executable: {} (accelerator={})",
                cfg.vllm_server_path,
                vllm_runtime.accelerator.empty() ? "?" : vllm_runtime.accelerator);
    } else if (!vllm_runtime.last_error.empty()) {
        state.set_last_error(vllm_runtime.last_error);
        MM_WARN("vLLM runtime is not ready: {}", vllm_runtime.last_error);
    }
    MM_INFO("vLLM update policy: {} (checks {})",
            cfg.vllm_update_policy, cfg.vllm_update_check ? "enabled" : "disabled");

    const auto remembered_keys_path = remembered_api_keys_file(cfg);
    std::vector<std::string> remembered_api_keys =
        load_remembered_api_keys(remembered_keys_path);

    // Prefer configured API key; otherwise reuse a remembered pairing key; fall back to generated.
    const char* key_env = std::getenv("MM_API_KEY");
    std::string initial_key = (key_env && *key_env)
        ? std::string(key_env)
        : (!remembered_api_keys.empty() ? remembered_api_keys.front()
                                        : mm::util::generate_api_key());
    state.add_api_key(initial_key);
    for (const auto& key : remembered_api_keys) {
        state.add_api_key(key);
    }
    MM_INFO("Node API key: {}", initial_key);
    if (!remembered_api_keys.empty()) {
        MM_INFO("Loaded {} remembered node API key(s)", remembered_api_keys.size());
    }

    // ── SlotManager ───────────────────────────────────────────────────────────

    mm::SlotManager slot_mgr(cfg.runtime_port_range_start,
                             cfg.runtime_port_range_end,
                             cfg.max_slots,
                             cfg.vllm_server_path,
                             cfg.vllm_gpu_budget);
    slot_mgr.set_gpu_vram_total_mb(state.get_metrics().gpu_vram_total_mb);

    // ── Local model cache ─────────────────────────────────────────────────────
    // Control transfers models into models_dir; the store keeps an LRU
    // use-queue, evicts unpinned models under disk pressure, and clears
    // unpinned models on shutdown. Pinned models — those control marked this
    // node preferred for — are retained.
    mm::ModelStore model_store(cfg.models_dir, cfg.model_cache_min_free_mb);
    MM_INFO("Model cache: root={} min_free={}MB clear_on_shutdown={}",
            model_store.root(), cfg.model_cache_min_free_mb,
            cfg.model_cache_clear_on_shutdown);

    // Advertise cluster capabilities (arch, GPUs, comm backends, vLLM build)
    // for multi-node engine-group planning on control.
    {
        auto caps = detect_node_capabilities(
            cfg, state.get_metrics().gpu_backend_available);
        state.set_capabilities(caps);
        MM_INFO("Node capabilities: arch={} gpus={} ray={} backends=[{}] vllm={}",
                caps.arch.empty() ? "?" : caps.arch, caps.gpu_count,
                caps.supports_ray, mm::util::join(caps.comm_backends, ","),
                caps.vllm_version.empty() ? "?" : caps.vllm_version);
    }

    // Periodically scrape running vLLM engines' /metrics so node status
    // reports per-engine load (running/waiting requests, KV cache usage).
    std::atomic<bool> stop_engine_metrics{false};
    std::thread engine_metrics_thread([&slot_mgr, &model_store, &stop_engine_metrics]() {
        while (!stop_engine_metrics) {
            slot_mgr.refresh_vllm_metrics();
            // Relieve disk pressure: evict LRU unpinned models that are not
            // backing a live slot.
            std::set<std::string> in_use;
            for (const auto& s : slot_mgr.get_slot_info())
                if (!s.model_path.empty()) in_use.insert(s.model_path);
            for (const auto& id : model_store.enforce_min_free(in_use))
                MM_INFO("Model cache: disk pressure evicted {}", id);
            for (int i = 0; i < 25 && !stop_engine_metrics; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // ── vLLM runtime management (provision / update) ───────────────────────────
    // Apply a runtime status produced by the provisioner: on a usable runtime,
    // rebind the engine executable path and refresh advertised capabilities;
    // always mirror the status (and any error) into NodeState. Serialized so the
    // TUI key, the REST endpoints, and the background thread can't race on
    // cfg.vllm_server_path.
    std::mutex vllm_apply_mutex;
    auto apply_runtime_result =
        [&](mm::VllmRuntimeStatus runtime) -> mm::VllmRuntimeStatus {
        std::lock_guard<std::mutex> lk(vllm_apply_mutex);
        if (mm::vllm_runtime_usable(runtime)) {
            cfg.vllm_server_path = runtime.executable_path;
            slot_mgr.set_vllm_server_path(cfg.vllm_server_path);
            auto caps = detect_node_capabilities(
                cfg, state.get_metrics().gpu_backend_available);
            state.set_capabilities(caps);
        }
        if (!runtime.last_error.empty()) {
            state.set_last_error(runtime.last_error);
        } else if (mm::vllm_runtime_usable(runtime)) {
            state.set_last_error("");
        }
        state.set_vllm_runtime(runtime);
        return runtime;
    };

    // Perform the user-approved update. If the upgrade install fails, fall back
    // to the still-present install so the node keeps serving, while surfacing
    // the update failure as the node error.
    auto do_vllm_update = [&]() -> mm::VllmRuntimeStatus {
        auto updated = vllm_provisioner.update_runtime();
        if (updated.status == "failed") {
            const std::string upd_err = updated.last_error;
            auto fallback = vllm_provisioner.ensure_runtime();
            if (mm::vllm_runtime_usable(fallback)) {
                fallback.last_error = "vLLM update failed: " + upd_err;
                MM_WARN("vLLM update failed, kept existing runtime: {}", upd_err);
                return apply_runtime_result(fallback);
            }
        }
        return apply_runtime_result(updated);
    };

    // Background update checker: probes upstream on an interval and, under the
    // "auto" policy, applies updates without prompting. "manual" skips checks
    // entirely; "prompt" leaves update_available set for the TUI / REST to act on.
    std::atomic<bool> stop_update_check{false};
    std::thread update_check_thread;
    if (cfg.vllm_update_check && cfg.vllm_update_policy != "manual") {
        update_check_thread = std::thread([&]() {
            // Let startup/registration settle before the first network probe.
            for (int i = 0; i < 50 && !stop_update_check; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            while (!stop_update_check) {
                if (mm::vllm_runtime_usable(state.get_vllm_runtime())) {
                    auto st = vllm_provisioner.check_for_update();
                    state.set_vllm_runtime(st);
                    if (st.update_available && cfg.vllm_update_policy == "auto") {
                        MM_INFO("vLLM auto-update policy: updating {} -> {}",
                                st.version, st.latest_version);
                        do_vllm_update();
                    }
                }
                const int64_t interval_ms =
                    static_cast<int64_t>(cfg.vllm_update_check_interval_hours) * 3600 * 1000;
                for (int64_t slept = 0; slept < interval_ms && !stop_update_check;
                     slept += 200)
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
    }

    // On-demand update worker for the TUI Update button/prompt: runs the (slow)
    // install off the UI thread on a tracked thread that shutdown joins, so the
    // update never outlives the objects it captures. One at a time.
    std::atomic<bool> manual_update_running{false};
    std::thread manual_update_thread;
    auto request_manual_update = [&]() {
        bool expected = false;
        if (!manual_update_running.compare_exchange_strong(expected, true)) return;
        if (manual_update_thread.joinable()) manual_update_thread.join();
        manual_update_thread = std::thread([&]() {
            do_vllm_update();
            manual_update_running = false;
        });
    };

    std::mutex runtime_log_mutex;
    std::deque<std::string> runtime_logs;
    constexpr std::size_t kMaxRuntimeLogLines = 4000;
    auto append_runtime_log = [&](const std::string& entry) {
        std::lock_guard<std::mutex> lk(runtime_log_mutex);
        if (runtime_logs.size() >= kMaxRuntimeLogLines) runtime_logs.pop_front();
        runtime_logs.push_back(entry);
    };

    mm::NodeUI* ui_ptr = nullptr;
    slot_mgr.set_log_callback([&](const std::string& line, bool is_stderr) {
        const std::string out = is_stderr ? "[stderr] " + line : line;
        append_runtime_log(out);
        if (ui_ptr) ui_ptr->append_log(out);
    });

    // Stream vLLM install/upgrade output into the runtime log and TUI, and mirror
    // install progress into NodeState so the UI can render a loading bar.
    vllm_provisioner.set_log_sink([&](const std::string& line, bool is_stderr) {
        const std::string out = is_stderr ? "[stderr] " + line : line;
        append_runtime_log(out);
        if (ui_ptr) ui_ptr->append_log(out);
    });
    vllm_provisioner.set_progress_sink([&](const mm::VllmInstallProgress& p) {
        state.set_vllm_install_progress(p);
    });

    // ── API server ────────────────────────────────────────────────────────────

    mm::NodeApiServer api_server(state, slot_mgr,
                                 cfg.control_url, cfg.pairing_key);
    api_server.set_ray_config(cfg.ray_path, cfg.ray_port);
    api_server.set_model_store(&model_store);
    // Resolve the HF hub cache dir once (config override → HF_HUB_CACHE →
    // HF_HOME/hub → ~/.cache/huggingface/hub) and share it with the API server
    // (status scan + pre-fetch). When configured, also point the vLLM child at
    // it so a cluster can share one cache (e.g. an NFS export on the NAS).
    {
        auto getenv_str = [](const char* k) -> std::string {
            const char* v = std::getenv(k);
            return v ? std::string(v) : std::string{};
        };
#ifdef _WIN32
        const std::string home = getenv_str("USERPROFILE");
#else
        const std::string home = getenv_str("HOME");
#endif
        const std::string hub_dir = mm::resolve_hf_hub_cache_dir(
            cfg.hf_cache_dir, getenv_str("HF_HUB_CACHE"), getenv_str("HF_HOME"), home);
        api_server.set_hf_config(cfg.hf_cli_path, hub_dir);
        if (!cfg.hf_cache_dir.empty()) {
#ifdef _WIN32
            _putenv_s("HF_HUB_CACHE", cfg.hf_cache_dir.c_str());
#else
            ::setenv("HF_HUB_CACHE", cfg.hf_cache_dir.c_str(), 1);
#endif
            MM_INFO("HF hub cache pinned to {} (shared with vLLM child)", cfg.hf_cache_dir);
        }
        MM_INFO("HF hub cache dir: {}", hub_dir.empty() ? "(default)" : hub_dir);
    }
    api_server.set_vllm_provision_callback([&]() {
        auto runtime = apply_runtime_result(vllm_provisioner.ensure_runtime());
        if (mm::vllm_runtime_usable(runtime))
            MM_INFO("vLLM runtime reprovisioned: {}", cfg.vllm_server_path);
        else if (!runtime.last_error.empty())
            MM_WARN("vLLM runtime reprovision failed: {}", runtime.last_error);
        return runtime;
    });
    // User-approved update (TUI Update button or POST .../provision {"update":true}).
    api_server.set_vllm_update_callback([&]() {
        MM_INFO("vLLM update requested by user");
        return do_vllm_update();
    });
    // On-demand online update check (POST .../check-update).
    api_server.set_vllm_check_update_callback([&]() {
        auto st = vllm_provisioner.check_for_update();
        state.set_vllm_runtime(st);
        return st;
    });
    api_server.set_remember_api_key_callback([&](const std::string& key) {
        remembered_api_keys.push_back(key);
        if (save_remembered_api_keys(remembered_keys_path, remembered_api_keys)) {
            MM_INFO("Remembered paired API key at {}", remembered_keys_path.string());
        } else {
            MM_WARN("Failed to remember paired API key at {}", remembered_keys_path.string());
        }
    });
    api_server.set_runtime_logs_provider([&](int tail) {
        if (tail < 1) tail = 1;
        std::vector<std::string> out;
        std::lock_guard<std::mutex> lk(runtime_log_mutex);
        const int total = static_cast<int>(runtime_logs.size());
        const int start = std::max(0, total - tail);
        out.reserve(static_cast<std::size_t>(total - start));
        for (int i = start; i < total; ++i) {
            out.push_back(runtime_logs[static_cast<std::size_t>(i)]);
        }
        return out;
    });

    std::thread api_thread([&]() {
        if (!api_server.listen(cfg.listen_port))
            MM_ERROR("NodeApiServer failed to bind on port {}", cfg.listen_port);
    });

    // Give the server a moment to bind before we register / broadcast.
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // ── UDP discovery broadcaster ─────────────────────────────────────────────
    const char* self_env = std::getenv("MM_SELF_URL");
    std::string broadcast_url = self_env
        ? std::string(self_env)
        : "http://127.0.0.1:" + std::to_string(cfg.listen_port);
    std::string broadcast_id = mm::util::generate_uuid();

    // Warn when we advertise a loopback host: control on another machine (or in
    // a different network namespace, e.g. WSL/Hyper-V) cannot reach it. The
    // control-side listener now substitutes the packet's source IP, but setting
    // MM_SELF_URL to a routable address is the correct fix.
    {
        const auto [adv_host, adv_port] = mm::util::parse_url(broadcast_url);
        (void)adv_port;
        const std::string h = mm::util::to_lower(adv_host);
        const bool loopback = h == "localhost" || h == "::1" || h == "::" ||
                              h == "0.0.0.0" || h.rfind("127.", 0) == 0;
        if (loopback)
            MM_WARN("Discovery broadcasting loopback URL {} — set MM_SELF_URL to a "
                    "routable address for multi-host clusters", broadcast_url);
    }

    mm::NodeDiscoveryBroadcaster broadcaster;
    broadcaster.start(broadcast_url, broadcast_id, cfg.discovery_port);
    MM_INFO("Discovery broadcaster started on port {} (id={})",
            cfg.discovery_port, broadcast_id);

    // ── UI ────────────────────────────────────────────────────────────────────

    auto forget_remembered_pairings = [&](std::string* out_message) -> bool {
        const std::size_t count = remembered_api_keys.size();
        remembered_api_keys.clear();
        if (!save_remembered_api_keys(remembered_keys_path, remembered_api_keys)) {
            if (out_message) *out_message = "failed to write " + remembered_keys_path.string();
            return false;
        }

        if (out_message) {
            *out_message = count == 0
                ? "no remembered pairing keys"
                : "forgot " + std::to_string(count) + " remembered pairing key(s)";
        }
        MM_INFO("Forgot remembered pairing keys at {}", remembered_keys_path.string());
        return true;
    };

    // ── Reconnection loop ─────────────────────────────────────────────────────
    // Retries registration with control every 30 s until success or shutdown.

    std::atomic<bool> stop_reconnect{false};
    std::thread reconnect_thread([&]() {
        constexpr int kActivePollMs = 5000;
        constexpr int64_t kControlStaleMs = 70000; // > default control health poll interval.

        if (cfg.control_url.empty()) {
            MM_INFO("MM_CONTROL_URL not set — registration disabled; using pair/contact detection only");
        }

        // Brief delay so the server finishes starting.
        for (int i = 0; i < 5 && !stop_reconnect; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

        while (!stop_reconnect) {
            if (state.is_registered()) {
                if (!cfg.control_url.empty()) {
                    mm::HttpClient ctrl(cfg.control_url);
                    if (!cfg.control_api_key.empty())
                        ctrl.set_bearer_token(cfg.control_api_key);
                    auto ping = ctrl.get("/v1/nodes");
                    if (!ping.ok()) {
                        MM_WARN("Control ping failed (HTTP {}) — marking node as disconnected", ping.status);
                        state.set_registered(false);
                    } else {
                        state.mark_control_contact();
                    }
                } else if (!state.has_recent_control_contact(kControlStaleMs)) {
                    MM_WARN("No control traffic observed for {}s — marking node as disconnected",
                            static_cast<int>(kControlStaleMs / 1000));
                    state.set_registered(false);
                }

                for (int i = 0; i < (kActivePollMs / 100) && !stop_reconnect; ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            if (cfg.control_url.empty()) {
                if (state.has_recent_control_contact(kControlStaleMs)) {
                    state.set_registered(true);
                    continue;
                }
                for (int i = 0; i < (kActivePollMs / 100) && !stop_reconnect; ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            if (try_register(cfg, state, initial_key)) {
                for (int i = 0; i < (kActivePollMs / 100) && !stop_reconnect; ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            MM_INFO("Will retry registration in 30 s ...");
            for (int i = 0; i < 300 && !stop_reconnect; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // ── Run UI (blocks until q / Esc) ─────────────────────────────────────────

    if (args.mode == NodeRunMode::Tui) {
        mm::NodeUI ui(
            state,
            cfg.listen_port,
            forget_remembered_pairings,
            request_manual_update);
        ui_ptr = &ui;
        ui.run();
        ui_ptr = nullptr;
        MM_INFO("UI exited - shutting down");
    } else {
        std::atomic<bool> stop_cli{false};
        g_node_cli_stop = &stop_cli;
        auto old_int = std::signal(SIGINT, node_cli_signal_handler);
#ifdef SIGTERM
        auto old_term = std::signal(SIGTERM, node_cli_signal_handler);
#endif
        run_node_cli(cfg.listen_port, initial_key, args.output, stop_cli);
        g_node_cli_stop = nullptr;
        std::signal(SIGINT, old_int);
#ifdef SIGTERM
        std::signal(SIGTERM, old_term);
#endif
        MM_INFO("CLI exited - shutting down");
    }

    // ── Graceful shutdown ─────────────────────────────────────────────────────

    stop_reconnect = true;
    reconnect_thread.join();

    broadcaster.stop();

    api_server.stop();
    if (api_thread.joinable()) api_thread.join();

    stop_engine_metrics = true;
    if (engine_metrics_thread.joinable()) engine_metrics_thread.join();

    stop_update_check = true;
    if (update_check_thread.joinable()) update_check_thread.join();
    if (manual_update_thread.joinable()) manual_update_thread.join();

    state.stop_metrics_poll();
    slot_mgr.unload_all();

    // With all slots down nothing is in use; drop every unpinned model so the
    // node keeps only the models control pinned to it.
    if (cfg.model_cache_clear_on_shutdown) {
        const auto cleared = model_store.clear_unpinned({});
        if (!cleared.empty())
            MM_INFO("Model cache: cleared {} unpinned model(s) on shutdown",
                    cleared.size());
    }

    MM_INFO("mantic-mind node stopped");
    return 0;
}
