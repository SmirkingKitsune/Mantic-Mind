#include "common/config_file.hpp"
#include "common/cli_repl.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/node_discovery.hpp"
#include "common/util.hpp"
#include "node/node_api_server.hpp"
#include "node/node_config.hpp"
#include "node/model_store.hpp"
#include "node/node_state.hpp"
#include "node/node_ui.hpp"
#include "node/singleton_lock.hpp"
#include "node/slot_manager.hpp"
#include "node/llama_cpp_provisioner.hpp"
#include "node/llama_runtime.hpp"

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
        cfg.llama_server_path = file.get("llama_server_path", "llama-server");
        cfg.llama_auto_provision = file.get_bool("llama_auto_provision", true);
        cfg.llama_provision_dir = file.get("llama_provision_dir", "");
        cfg.llama_install_method = file.get("llama_install_method", "auto");
        cfg.llama_version = file.get("llama_version", "latest");
        cfg.llama_accelerator = file.get("llama_accelerator", "");
        cfg.llama_cuda_arch = file.get("llama_cuda_arch", "");
        cfg.llama_build_jobs = file.get_int("llama_build_jobs", 0);
        cfg.llama_update_policy = file.get("llama_update_policy", "prompt");
        cfg.llama_update_check = file.get_bool("llama_update_check", true);
        cfg.llama_update_check_interval_hours =
            file.get_int("llama_update_check_interval_hours", 24);
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
        cfg.node_gpu_count    = file.get_int("node_gpu_count", 0);
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
    cfg.llama_server_path = env("MM_LLAMA_PATH", cfg.llama_server_path);
    cfg.llama_auto_provision = env_bool("MM_LLAMA_AUTO_PROVISION", cfg.llama_auto_provision);
    cfg.llama_provision_dir = env("MM_LLAMA_PROVISION_DIR", cfg.llama_provision_dir);
    cfg.llama_install_method = env("MM_LLAMA_INSTALL_METHOD", cfg.llama_install_method);
    cfg.llama_version = env("MM_LLAMA_VERSION", cfg.llama_version);
    cfg.llama_accelerator = env("MM_LLAMA_ACCELERATOR", cfg.llama_accelerator);
    cfg.llama_cuda_arch = env("MM_LLAMA_CUDA_ARCH", cfg.llama_cuda_arch);
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
    cfg.llama_build_jobs = env_int("MM_LLAMA_BUILD_JOBS", cfg.llama_build_jobs);
    cfg.llama_update_policy = env("MM_LLAMA_UPDATE_POLICY", cfg.llama_update_policy);
    cfg.llama_update_check = env_bool("MM_LLAMA_UPDATE_CHECK", cfg.llama_update_check);
    cfg.llama_update_check_interval_hours =
        env_int("MM_LLAMA_UPDATE_CHECK_INTERVAL_HOURS",
                cfg.llama_update_check_interval_hours);
    cfg.model_cache_min_free_mb = static_cast<int64_t>(
        env_int("MM_MODEL_CACHE_MIN_FREE_MB",
                static_cast<int>(cfg.model_cache_min_free_mb)));
    cfg.model_cache_clear_on_shutdown =
        env_bool("MM_MODEL_CACHE_CLEAR_ON_SHUTDOWN", cfg.model_cache_clear_on_shutdown);

    cfg.node_gpu_count  = env_int("MM_NODE_GPU_COUNT", cfg.node_gpu_count);

    // Keep node startup resilient if env/config accidentally provide empty strings.
    {
        const std::string policy = mm::util::to_lower(mm::util::trim(cfg.llama_update_policy));
        cfg.llama_update_policy =
            (policy == "auto" || policy == "manual") ? policy : "prompt";
    }
    if (cfg.llama_update_check_interval_hours < 1)
        cfg.llama_update_check_interval_hours = 24;
    if (cfg.llama_build_jobs < 0) cfg.llama_build_jobs = 0;
    if (cfg.models_dir.empty()) cfg.models_dir = "models";
    if (cfg.model_cache_min_free_mb < 0) cfg.model_cache_min_free_mb = 0;
    if (cfg.data_dir.empty()) cfg.data_dir = "data";
    if (cfg.llama_server_path.empty()) cfg.llama_server_path = "llama-server";
    if (cfg.llama_install_method.empty()) cfg.llama_install_method = "auto";
    cfg.llama_install_method = mm::normalize_llama_install_method(cfg.llama_install_method);
    if (cfg.llama_version.empty()) cfg.llama_version = "latest";
    if (cfg.llama_provision_dir.empty())
        cfg.llama_provision_dir =
            (std::filesystem::path(cfg.data_dir) / "runtimes" / "llama.cpp").string();
    if (cfg.kv_cache_dir.empty()) cfg.kv_cache_dir = "data/kv_cache";
    if (cfg.log_file.empty()) cfg.log_file = "logs/mantic-mind.log";

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

// Resolve every visible NVIDIA GPU's compute capability to the CMake form
// (8.9 -> 89, 12.1 -> 121). Passing an explicit list avoids fragile "native"
// architecture probing in older CMake versions and WSL toolchains.
std::string detect_cuda_architectures() {
    const std::string cmd =
        "nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits";
#ifdef _WIN32
    FILE* f = _popen((cmd + " 2>nul").c_str(), "r");
#else
    FILE* f = ::popen((cmd + " 2>/dev/null").c_str(), "r");
#endif
    if (!f) return {};
    std::set<std::string> capabilities;
    char buf[256];
    while (fgets(buf, static_cast<int>(sizeof(buf)), f)) {
        std::string value = mm::util::trim(buf);
        value.erase(std::remove(value.begin(), value.end(), '.'), value.end());
        if (!value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
                return std::isdigit(ch) != 0;
            }))
            capabilities.insert(value);
    }
#ifdef _WIN32
    _pclose(f);
#else
    ::pclose(f);
#endif
    std::vector<std::string> ordered(capabilities.begin(), capabilities.end());
    return mm::util::join(ordered, ";");
}

// Build the capability block this node advertises. Detection fills the gaps
// the config left blank. llama_version is the resolved llama.cpp build
// fingerprint (from the llama provisioner), "" while unknown/provisioning.
mm::NodeCapabilities detect_node_capabilities(const mm::NodeConfig& cfg,
                                              const std::string& llama_version = {}) {
    mm::NodeCapabilities caps;

#if defined(__aarch64__) || defined(_M_ARM64)
    caps.arch = "aarch64";
#elif defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
    caps.arch = "x86_64";
#else
    caps.arch = "";
#endif

    caps.gpu_count = cfg.node_gpu_count > 0 ? cfg.node_gpu_count : detect_gpu_count();

    // llama.cpp build fingerprint (RPC groups need matching builds; advertised
    // for planning even while supports_llama_rpc stays a future capability).
    caps.llama_cpp_version = llama_version;

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

    const std::string platform = mm::current_runtime_platform();

    mm::HttpClient ctrl(cfg.control_url);
    if (!cfg.control_api_key.empty())
        ctrl.set_bearer_token(cfg.control_api_key);

    nlohmann::json body = {
        {"node_url", self_url},
        {"api_key",  api_key},
        {"platform", platform},
        {"hostname", mm::util::hostname()},
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
    MM_INFO("Node llama-server path: {}", cfg.llama_server_path);

    // ── NodeState ─────────────────────────────────────────────────────────────

    mm::NodeState state;
    const std::string local_node_id = "local-" + mm::util::generate_uuid();
    state.set_registered(false, local_node_id);
    MM_INFO("Local node identity: {}", local_node_id);

    // Start metrics polling before provisioning so the GPU-backend probe
    // (nvidia-smi) has populated gpu_backend_available before we pick the
    // accelerator-correct llama.cpp build variant.
    state.start_metrics_poll(2000);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

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
                             cfg.max_slots);
    slot_mgr.set_llama_server_path(cfg.llama_server_path);
    slot_mgr.set_kv_cache_dir(cfg.kv_cache_dir);
    slot_mgr.set_models_dir(cfg.models_dir);

    // ── Local model cache ─────────────────────────────────────────────────────
    // Control transfers models into models_dir; the store keeps an LRU
    // use-queue, evicts unpinned models under disk pressure, and clears
    // unpinned models on shutdown. Pinned models — those control marked this
    // node preferred for — are retained.
    mm::ModelStore model_store(cfg.models_dir, cfg.model_cache_min_free_mb);
    MM_INFO("Model cache: root={} min_free={}MB clear_on_shutdown={}",
            model_store.root(), cfg.model_cache_min_free_mb,
            cfg.model_cache_clear_on_shutdown);

    // Advertise hardware and llama.cpp runtime capabilities to control.
    {
        const auto llama_rt = state.get_llama_runtime();
        auto caps = detect_node_capabilities(
            cfg, mm::llama_runtime_usable(llama_rt) ? llama_rt.version : std::string{});
        state.set_capabilities(caps);
        MM_INFO("Node capabilities: arch={} gpus={} llama={}",
                caps.arch.empty() ? "?" : caps.arch, caps.gpu_count,
                caps.llama_cpp_version.empty() ? "?" : caps.llama_cpp_version);
    }

    // Periodically relieve disk pressure without evicting a live model.
    std::atomic<bool> stop_model_housekeeping{false};
    std::thread model_housekeeping_thread([&slot_mgr, &model_store, &stop_model_housekeeping]() {
        while (!stop_model_housekeeping) {
            // Relieve disk pressure: evict LRU unpinned models that are not
            // backing a live slot.
            std::set<std::string> in_use;
            for (const auto& s : slot_mgr.get_slot_info()) {
                if (!s.model_path.empty()) in_use.insert(s.model_path);
                if (!s.mmproj_path.empty()) in_use.insert(s.mmproj_path);
            }
            for (const auto& id : model_store.enforce_min_free(in_use))
                MM_INFO("Model cache: disk pressure evicted {}", id);
            for (int i = 0; i < 25 && !stop_model_housekeeping; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // ── llama.cpp runtime management (provision in the background) ─────────────
    // Release retrieval or its source-build fallback runs on a background
    // thread, so node startup never blocks on it. The fast
    // resolve-from-PATH and already-provisioned paths complete near-instantly
    // inside ensure_runtime(). A shutdown flag drives the provisioner's cancel
    // check so a download/build in flight aborts promptly.
    mm::LlamaProvisionConfig llama_provision_cfg;
    llama_provision_cfg.requested_executable = cfg.llama_server_path;
    llama_provision_cfg.provision_dir = cfg.llama_provision_dir;
    llama_provision_cfg.auto_provision = cfg.llama_auto_provision;
    llama_provision_cfg.install_method = cfg.llama_install_method;
    llama_provision_cfg.version = cfg.llama_version;
    llama_provision_cfg.cuda_arch = cfg.llama_cuda_arch;
    llama_provision_cfg.cmake_args = cfg.llama_cmake_args;
    llama_provision_cfg.build_jobs = cfg.llama_build_jobs;
    llama_provision_cfg.accelerator_explicit = !cfg.llama_accelerator.empty();
    llama_provision_cfg.accelerator = !cfg.llama_accelerator.empty()
        ? cfg.llama_accelerator
        : mm::detect_llama_accelerator(mm::current_runtime_platform(), mm::current_runtime_arch(),
                                       state.get_metrics().gpu_backend_available,
                                       mm::detect_rocm_present());
    if (llama_provision_cfg.accelerator == "cuda" &&
        llama_provision_cfg.cuda_arch.empty()) {
        llama_provision_cfg.cuda_arch = detect_cuda_architectures();
        if (!llama_provision_cfg.cuda_arch.empty())
            MM_INFO("Detected CUDA compute architecture(s) for llama.cpp: {}",
                    llama_provision_cfg.cuda_arch);
    }
    mm::LlamaCppProvisioner llama_provisioner(llama_provision_cfg);
    state.set_llama_runtime(llama_provisioner.status());
    MM_INFO("llama.cpp update policy: {} (checks {})",
            cfg.llama_update_policy, cfg.llama_update_check ? "enabled" : "disabled");
    llama_provisioner.set_log_sink([](const std::string& line, bool is_stderr) {
        if (is_stderr) MM_WARN("[llama] {}", line); else MM_INFO("[llama] {}", line);
    });
    std::atomic<bool> llama_provision_stop{false};
    llama_provisioner.set_progress_sink([&](const mm::RuntimeInstallProgress& p) {
        if (!p.active) {
            state.clear_action_progress("llama-runtime");
            return;
        }
        mm::NodeActionProgress action;
        action.active = true;
        action.operation_id = "llama-runtime";
        action.kind = "runtime";
        action.action = p.stage.find("Building") != std::string::npos
            ? "Compiling llama.cpp runtime"
            : "Provisioning llama.cpp runtime";
        const auto progress_runtime = llama_provisioner.status();
        action.target = progress_runtime.accelerator.empty()
            ? std::string{"llama-server"}
            : "llama-server (" + progress_runtime.accelerator + ")";
        action.stage = p.stage;
        action.detail = p.last_line;
        action.step = p.step;
        action.total_steps = p.total_steps;
        action.fraction = p.fraction;
        action.cancelable = true;
        state.set_action_progress(action);
    });
    llama_provisioner.set_cancel_check([&]() {
        return llama_provision_stop.load()
            || state.action_cancel_requested("llama-runtime");
    });

    // ── llama.cpp runtime management (provision / update) ─────────────────────
    // Apply a runtime status produced by the provisioner and refresh advertised
    // capabilities. Serialized so UI, API, and background work cannot race.
    std::mutex runtime_apply_mutex;
    std::mutex llama_operation_mutex;
    auto apply_llama_result =
        [&](mm::LlamaRuntimeStatus runtime) -> mm::LlamaRuntimeStatus {
        std::lock_guard<std::mutex> lk(runtime_apply_mutex);
        if (mm::llama_runtime_usable(runtime)) {
            cfg.llama_server_path = runtime.executable_path;
            slot_mgr.set_llama_server_path(cfg.llama_server_path);
            auto caps = detect_node_capabilities(cfg, runtime.version);
            state.set_capabilities(caps);
        }
        if (!runtime.last_error.empty()) {
            state.set_last_error(runtime.last_error);
        } else if (mm::llama_runtime_usable(runtime)) {
            state.set_last_error("");
        }
        state.set_llama_runtime(runtime);
        return runtime;
    };

    std::thread llama_provision_thread([&]() {
        std::lock_guard<std::mutex> op_lk(llama_operation_mutex);
        auto runtime = apply_llama_result(llama_provisioner.ensure_runtime());
        if (mm::llama_runtime_usable(runtime)) {
            MM_INFO("Using llama-server executable: {} (accelerator={})",
                    runtime.executable_path, runtime.accelerator);
        } else if (!runtime.last_error.empty()) {
            MM_WARN("llama.cpp runtime unavailable: {}", runtime.last_error);
        }
    });

    auto ensure_llama_runtime = [&]() -> mm::LlamaRuntimeStatus {
        std::lock_guard<std::mutex> op_lk(llama_operation_mutex);
        return apply_llama_result(llama_provisioner.ensure_runtime());
    };

    auto do_llama_update = [&](const std::string& accelerator) -> mm::LlamaRuntimeStatus {
        std::lock_guard<std::mutex> op_lk(llama_operation_mutex);
        auto updated = llama_provisioner.update_runtime(accelerator);
        if (updated.status == "failed") {
            const std::string update_error = updated.last_error;
            auto fallback = llama_provisioner.ensure_runtime();
            if (mm::llama_runtime_usable(fallback)) {
                fallback.last_error = "llama.cpp update failed: " + update_error;
                fallback.troubleshooting = updated.troubleshooting;
                MM_WARN("llama.cpp update failed, kept existing runtime: {}", update_error);
                return apply_llama_result(fallback);
            }
        }
        return apply_llama_result(updated);
    };

    auto do_llama_switch = [&](const std::string& variant) -> mm::LlamaRuntimeStatus {
        std::lock_guard<std::mutex> op_lk(llama_operation_mutex);
        auto switched = llama_provisioner.switch_runtime(variant);
        if (switched.status == "failed") {
            const std::string switch_error = switched.last_error;
            auto fallback = llama_provisioner.ensure_runtime();
            if (mm::llama_runtime_usable(fallback)) {
                fallback.last_error = "llama.cpp engine switch failed: " + switch_error;
                fallback.troubleshooting = switched.troubleshooting;
                MM_WARN("llama.cpp engine switch failed, kept existing runtime: {}",
                        switch_error);
                return apply_llama_result(fallback);
            }
        }
        return apply_llama_result(switched);
    };

    auto do_llama_recovery = [&](const std::string& action,
                                 const std::string& variant) -> mm::LlamaRuntimeStatus {
        std::lock_guard<std::mutex> op_lk(llama_operation_mutex);
        MM_INFO("llama.cpp troubleshooting action requested: {}{}", action,
                variant.empty() ? std::string{} : " (" + variant + ")");
        auto runtime = action == "diagnose"
            ? llama_provisioner.diagnose_environment()
            : llama_provisioner.recover_runtime(action, variant);
        if (action == "target" && runtime.status == "failed") {
            const auto failed = runtime;
            auto fallback = llama_provisioner.ensure_runtime();
            if (mm::llama_runtime_usable(fallback)) {
                fallback.last_error =
                    "llama.cpp target install failed; kept active " +
                    (fallback.variant.empty() ? fallback.accelerator
                                              : fallback.variant) +
                    " runtime: " + failed.last_error;
                fallback.build_log_path = failed.build_log_path;
                fallback.troubleshooting = failed.troubleshooting;
                MM_WARN("llama.cpp target install failed, kept active runtime: {}",
                        failed.last_error);
                return apply_llama_result(fallback);
            }
        }
        return apply_llama_result(runtime);
    };

    auto check_llama_update = [&]() -> mm::LlamaRuntimeStatus {
        std::lock_guard<std::mutex> op_lk(llama_operation_mutex);
        auto status = llama_provisioner.check_for_update();
        state.set_llama_runtime(status);
        return status;
    };

    std::atomic<bool> stop_llama_update_check{false};
    std::thread llama_update_check_thread;
    if (cfg.llama_update_check && cfg.llama_update_policy != "manual") {
        llama_update_check_thread = std::thread([&]() {
            for (int i = 0; i < 50 && !stop_llama_update_check; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            while (!stop_llama_update_check) {
                if (mm::llama_runtime_usable(state.get_llama_runtime())) {
                    auto st = check_llama_update();
                    if (st.update_available && cfg.llama_update_policy == "auto") {
                        if (st.update_action == "unavailable") {
                            MM_WARN("llama.cpp auto-update skipped: {}", st.update_warning);
                        } else {
                            MM_INFO("llama.cpp auto-update policy: updating {} -> {} ({})",
                                    st.version, st.latest_version,
                                    st.update_action.empty() ? "release/fallback" : st.update_action);
                            do_llama_update({});
                        }
                    }
                }
                const int64_t interval_ms =
                    static_cast<int64_t>(cfg.llama_update_check_interval_hours) * 3600 * 1000;
                for (int64_t slept = 0;
                     slept < interval_ms && !stop_llama_update_check;
                     slept += 200)
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
    }

    // On-demand update worker for the TUI Update button/prompt: runs the (slow)
    // install off the UI thread on a tracked thread that shutdown joins, so the
    // update never outlives the objects it captures. One at a time.
    std::atomic<bool> manual_llama_update_running{false};
    std::thread manual_llama_update_thread;
    auto request_manual_llama_update = [&](std::string accelerator) {
        bool expected = false;
        if (!manual_llama_update_running.compare_exchange_strong(expected, true)) return;
        if (manual_llama_update_thread.joinable()) manual_llama_update_thread.join();
        manual_llama_update_thread = std::thread(
            [&, accelerator = std::move(accelerator)]() {
                do_llama_update(accelerator);
                manual_llama_update_running = false;
            });
    };
    auto request_manual_llama_switch = [&](std::string variant) {
        bool expected = false;
        if (!manual_llama_update_running.compare_exchange_strong(expected, true)) return;
        if (manual_llama_update_thread.joinable()) manual_llama_update_thread.join();
        manual_llama_update_thread = std::thread(
            [&, variant = std::move(variant)]() {
                do_llama_switch(variant);
                manual_llama_update_running = false;
            });
    };
    std::atomic<bool> manual_llama_recovery_running{false};
    std::thread manual_llama_recovery_thread;
    auto request_manual_llama_recovery = [&](std::string action,
                                             std::string variant) {
        bool expected = false;
        if (!manual_llama_recovery_running.compare_exchange_strong(expected, true)) return;
        if (manual_llama_recovery_thread.joinable())
            manual_llama_recovery_thread.join();
        manual_llama_recovery_thread = std::thread(
            [&, action = std::move(action), variant = std::move(variant)]() {
                do_llama_recovery(action, variant);
                manual_llama_recovery_running = false;
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

    // ── API server ────────────────────────────────────────────────────────────

    mm::NodeApiServer api_server(state, slot_mgr,
                                 cfg.control_url, cfg.pairing_key);
    api_server.set_model_store(&model_store);
    api_server.set_llama_provision_callback([&]() {
        MM_INFO("llama.cpp provisioning requested by user");
        return ensure_llama_runtime();
    });
    api_server.set_llama_update_callback([&](const std::string& accelerator) {
        MM_INFO("llama.cpp update requested by user (accelerator={})",
                accelerator.empty() ? "current" : accelerator);
        return do_llama_update(accelerator);
    });
    api_server.set_llama_switch_callback([&](const std::string& variant) {
        MM_INFO("llama.cpp engine switch requested by user (variant={})", variant);
        return do_llama_switch(variant);
    });
    api_server.set_llama_check_update_callback([&]() {
        return check_llama_update();
    });
    api_server.set_llama_diagnose_callback([&]() {
        return do_llama_recovery("diagnose", {});
    });
    api_server.set_llama_recovery_callback(
        [&](const std::string& action, const std::string& variant) {
            return do_llama_recovery(action, variant);
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
    broadcaster.start(broadcast_url, broadcast_id, mm::util::hostname(), cfg.discovery_port);
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
            request_manual_llama_update,
            request_manual_llama_switch,
            request_manual_llama_recovery);
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

    stop_model_housekeeping = true;
    if (model_housekeeping_thread.joinable()) model_housekeeping_thread.join();

    stop_llama_update_check = true;
    llama_provision_stop = true;
    state.request_action_cancel();
    if (llama_provision_thread.joinable()) llama_provision_thread.join();
    if (llama_update_check_thread.joinable()) llama_update_check_thread.join();
    if (manual_llama_update_thread.joinable()) {
        state.request_action_cancel();
        manual_llama_update_thread.join();
    }
    if (manual_llama_recovery_thread.joinable()) {
        state.request_action_cancel();
        manual_llama_recovery_thread.join();
    }

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
