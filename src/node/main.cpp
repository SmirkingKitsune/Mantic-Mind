#include "common/config_file.hpp"
#include "common/cli_repl.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/model_catalog.hpp"
#include "common/node_discovery.hpp"
#include "common/util.hpp"
#include "node/model_storage.hpp"
#include "node/node_api_server.hpp"
#include "node/node_config.hpp"
#include "node/node_state.hpp"
#include "node/node_ui.hpp"
#include "node/llama_runtime_manager.hpp"
#include "node/singleton_lock.hpp"
#include "node/slot_manager.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
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
        cfg.models_dir        = file.get("models_dir",        "models");
        cfg.data_dir          = file.get("data_dir",          "data");
        cfg.log_file          = file.get("log_file", "logs/mantic-mind.log");
        cfg.listen_port = static_cast<uint16_t>(
            file.get_int("listen_port", 7070));
        cfg.llama_port  = static_cast<uint16_t>(
            file.get_int("llama_port",  8080));
        cfg.llama_port_range_start = static_cast<uint16_t>(
            file.get_int("llama_port_range_start", cfg.llama_port));
        cfg.llama_port_range_end = static_cast<uint16_t>(
            file.get_int("llama_port_range_end", cfg.llama_port_range_start + 10));
        cfg.max_slots    = file.get_int("max_slots", 4);
        cfg.kv_cache_dir = file.get("kv_cache_dir", "data/kv_cache");
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

    cfg.control_url       = env("MM_CONTROL_URL",     cfg.control_url);
    cfg.control_api_key   = env("MM_CONTROL_API_KEY", cfg.control_api_key);
    cfg.llama_server_path = env("MM_LLAMA_PATH",      cfg.llama_server_path);
    cfg.models_dir        = env("MM_MODELS_DIR",      cfg.models_dir);
    cfg.data_dir          = env("MM_DATA_DIR",        cfg.data_dir);
    cfg.log_file          = env("MM_LOG_FILE",        cfg.log_file);
    cfg.kv_cache_dir      = env("MM_KV_CACHE_DIR",   cfg.kv_cache_dir);
    cfg.pairing_key       = env("MM_PAIRING_KEY",     cfg.pairing_key);
    cfg.listen_port       = env_port("MM_LISTEN_PORT",    cfg.listen_port);
    cfg.llama_port        = env_port("MM_LLAMA_PORT",     cfg.llama_port);
    cfg.discovery_port    = env_port("MM_DISCOVERY_PORT", cfg.discovery_port);

    auto env_int = [](const char* name, int cur) -> int {
        const char* v = std::getenv(name);
        if (!v) return cur;
        try { return std::stoi(v); } catch (...) { return cur; }
    };
    cfg.llama_port_range_start = env_port("MM_LLAMA_PORT_RANGE_START", cfg.llama_port_range_start);
    cfg.llama_port_range_end   = env_port("MM_LLAMA_PORT_RANGE_END",   cfg.llama_port_range_end);
    cfg.max_slots              = env_int("MM_MAX_SLOTS", cfg.max_slots);

    // Keep node startup resilient if env/config accidentally provide empty strings.
    if (cfg.llama_server_path.empty()) cfg.llama_server_path = "llama-server";
    if (cfg.models_dir.empty()) cfg.models_dir = "models";
    if (cfg.data_dir.empty()) cfg.data_dir = "data";
    if (cfg.kv_cache_dir.empty()) cfg.kv_cache_dir = "data/kv_cache";
    if (cfg.log_file.empty()) cfg.log_file = "logs/mantic-mind.log";

    return cfg;
}

namespace {

std::filesystem::path resolve_llama_cache_file(const mm::NodeConfig& cfg) {
    const char* env_cache = std::getenv("MM_LLAMA_PATH_CACHE_FILE");
    if (env_cache && *env_cache) return std::filesystem::path(env_cache);
    return std::filesystem::path(cfg.data_dir) / "llama_server_path.txt";
}

bool is_default_llama_path(const std::string& path_in) {
    std::string s = mm::util::to_lower(mm::util::trim(path_in));
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') ||
         (s.front() == '\'' && s.back() == '\''))) {
        s = mm::util::to_lower(mm::util::trim(s.substr(1, s.size() - 2)));
    }
    return s.empty() || s == "llama-server" || s == "llama-server.exe";
}

bool looks_path_like(const std::string& s) {
    return s.find('\\') != std::string::npos ||
           s.find('/')  != std::string::npos ||
           s.find(':')  != std::string::npos;
}

bool file_exists_regular(const std::string& p) {
    if (p.empty()) return false;
    std::error_code ec;
    return std::filesystem::exists(p, ec) && std::filesystem::is_regular_file(p, ec);
}

std::string load_cached_llama_path(const std::filesystem::path& cache_file) {
    std::ifstream in(cache_file);
    if (!in) return {};
    std::string line;
    std::getline(in, line);
    line = mm::util::trim(line);
    if (line.size() >= 2 &&
        ((line.front() == '"' && line.back() == '"') ||
         (line.front() == '\'' && line.back() == '\''))) {
        line = mm::util::trim(line.substr(1, line.size() - 2));
    }
    return line;
}

bool save_cached_llama_path(const std::filesystem::path& cache_file,
                            const std::string& path) {
    if (path.empty()) return false;
    std::error_code ec;
    std::filesystem::create_directories(cache_file.parent_path(), ec);
    std::ofstream out(cache_file, std::ios::out | std::ios::trunc);
    if (!out) return false;
    out << path << '\n';
    return true;
}

#ifdef _WIN32
void set_process_env_cache_file(const std::filesystem::path& p) {
    _putenv_s("MM_LLAMA_PATH_CACHE_FILE", p.string().c_str());
}
void set_process_env_llama_path(const std::string& p) {
    if (p.empty()) return;
    _putenv_s("MM_LLAMA_PATH", p.c_str());
}
#else
void set_process_env_cache_file(const std::filesystem::path& p) {
    setenv("MM_LLAMA_PATH_CACHE_FILE", p.string().c_str(), 1);
}
void set_process_env_llama_path(const std::string& p) {
    if (p.empty()) return;
    setenv("MM_LLAMA_PATH", p.c_str(), 1);
}
#endif

} // namespace

// ── Registration helper ────────────────────────────────────────────────────────

static bool try_register(const mm::NodeConfig& cfg,
                          mm::NodeState& state,
                          const std::string& api_key) {
    const char* self_env = std::getenv("MM_SELF_URL");
    std::string self_url = self_env
        ? std::string(self_env)
        : "http://127.0.0.1:" + std::to_string(cfg.listen_port);

#ifdef _WIN32
    const std::string platform = "windows";
#else
    const std::string platform = "linux";
#endif

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
        << "  models pull <model_filename>\n"
        << "  models delete <model_filename>\n"
        << "  llama check-update\n"
        << "  llama update [build(true|false)] [force(true|false)]\n"
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
        printer.line("Commands: status | metrics | slots | models list | models pull <file> | "
                     "models delete <file> | llama check-update | llama update [build] [force] | "
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
                    " slots=" + std::to_string(j.value("slot_in_use", 0)) + "/" + std::to_string(j.value("max_slots", 0)) +
                    " stored_models=" + std::to_string(static_cast<int>(j.value("stored_models", nlohmann::json::array()).size())) +
                    " llama_path=\"" + j.value("llama_server_path", std::string{}) + "\"" +
                    " update_available=" + (j.value("llama_update_available", false) ? "yes" : "no");
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
            if (tokens.size() < 2) {
                printer.line("usage: models list | models pull <model_filename> | models delete <model_filename>");
                continue;
            }
            const std::string sub = mm::util::to_lower(tokens[1]);
            if (sub == "list") {
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
            if (sub == "pull") {
                if (tokens.size() < 3) {
                    printer.line("usage: models pull <model_filename>");
                    continue;
                }
                const std::string filename = mm::canonical_model_filename(tokens[2]);
                if (!mm::is_safe_model_filename(filename)) {
                    printer.line("error: invalid model filename");
                    continue;
                }
                auto r = self.post("/api/node/models/pull",
                                   nlohmann::json{{"model_filename", filename}, {"force", false}});
                emit_http_result("models pull", r);
                continue;
            }
            if (sub == "delete") {
                if (tokens.size() < 3) {
                    printer.line("usage: models delete <model_filename>");
                    continue;
                }
                const std::string filename = mm::canonical_model_filename(tokens[2]);
                if (!mm::is_safe_model_filename(filename)) {
                    printer.line("error: invalid model filename");
                    continue;
                }
                auto r = self.del("/api/node/models/" + filename);
                emit_http_result("models delete", r);
                continue;
            }
            printer.line("error: unknown models subcommand");
            continue;
        }

        if (cmd0 == "llama") {
            if (tokens.size() < 2) {
                printer.line("usage: llama check-update | llama update [build] [force]");
                continue;
            }
            const std::string sub = mm::util::to_lower(tokens[1]);
            if (sub == "check-update") {
                auto r = self.post("/api/node/llama/check-update", nlohmann::json::object());
                emit_http_result("llama check-update", r);
                continue;
            }
            if (sub == "update") {
                bool build = true;
                bool force = false;
                if (tokens.size() >= 3 && !mm::cli::parse_bool_token(tokens[2], &build)) {
                    printer.line("error: build must be true|false");
                    continue;
                }
                if (tokens.size() >= 4 && !mm::cli::parse_bool_token(tokens[3], &force)) {
                    printer.line("error: force must be true|false");
                    continue;
                }
                auto r = self.post("/api/node/llama/update",
                                   nlohmann::json{{"build", build}, {"force", force}});
                emit_http_result("llama update", r);
                continue;
            }
            printer.line("error: unknown llama subcommand");
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
    auto llama_cache_file = resolve_llama_cache_file(cfg);
    {
        std::error_code ec;
        fs::create_directories(fs::path(cfg.log_file).parent_path(), ec);
        fs::create_directories(cfg.kv_cache_dir, ec);
        fs::create_directories(llama_cache_file.parent_path(), ec);
    }
    if (!(std::getenv("MM_LLAMA_PATH_CACHE_FILE") && *std::getenv("MM_LLAMA_PATH_CACHE_FILE"))) {
        set_process_env_cache_file(llama_cache_file);
    }

    // If no explicit MM_LLAMA_PATH is set, restore cached executable path when config
    // still points at default launcher name or an invalid explicit path.
    if (!(std::getenv("MM_LLAMA_PATH") && *std::getenv("MM_LLAMA_PATH"))) {
        std::string cached = load_cached_llama_path(llama_cache_file);
        bool should_apply = is_default_llama_path(cfg.llama_server_path);
        if (!should_apply && looks_path_like(cfg.llama_server_path)
            && !file_exists_regular(cfg.llama_server_path)) {
            should_apply = true;
        }
        if (should_apply && !cached.empty() && file_exists_regular(cached)) {
            cfg.llama_server_path = cached;
        }
    }

    mm::init_logger(
        cfg.log_file,
        "mm-node",
        args.mode == NodeRunMode::Cli ? spdlog::level::off : spdlog::level::info,
        spdlog::level::trace);
    MM_INFO("mantic-mind node starting — listen port {}, slots {}, port range {}-{}",
            cfg.listen_port, cfg.max_slots,
            cfg.llama_port_range_start, cfg.llama_port_range_end);
    MM_INFO("Node config source: {}",
            cfg_path.empty() ? "(defaults/env only; no config file found)" : cfg_path);
    MM_INFO("Node config — control_url='{}', llama_server_path='{}', models_dir='{}'",
            cfg.control_url, cfg.llama_server_path, cfg.models_dir);
    MM_INFO("Node llama path cache file: {}", llama_cache_file.string());
    if (looks_path_like(cfg.llama_server_path) && file_exists_regular(cfg.llama_server_path)) {
        if (!save_cached_llama_path(llama_cache_file, cfg.llama_server_path)) {
            MM_WARN("Failed to persist llama path cache: {}", llama_cache_file.string());
        }
    }

    // ── NodeState ─────────────────────────────────────────────────────────────

    mm::NodeState state;
    const std::string local_node_id = "local-" + mm::util::generate_uuid();
    state.set_registered(false, local_node_id);
    state.set_llama_server_path(cfg.llama_server_path);
    MM_INFO("Local node identity: {}", local_node_id);

    // Prefer configured API key; fall back to env var; fall back to generated.
    const char* key_env = std::getenv("MM_API_KEY");
    std::string initial_key = (key_env && *key_env)
        ? std::string(key_env)
        : mm::util::generate_api_key();
    state.add_api_key(initial_key);
    MM_INFO("Node API key: {}", initial_key);

    state.start_metrics_poll(2000);

    // Brief wait for first metrics sample, then warn if GPU detected but no CUDA backend.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    {
        auto m = state.get_metrics();
        if (m.gpu_vram_total_mb > 0 && !m.gpu_backend_available) {
            MM_WARN("NVIDIA GPU detected ({} MB VRAM) but llama-server lacks CUDA backend. "
                    "Models will load on CPU. Rebuild llama-server with -DGGML_CUDA=ON.",
                    static_cast<long long>(m.gpu_vram_total_mb));
        }
    }

    // ── SlotManager + ModelStorage ────────────────────────────────────────────

    mm::SlotManager slot_mgr(cfg.llama_server_path,
                             cfg.llama_port_range_start,
                             cfg.llama_port_range_end,
                             cfg.max_slots,
                             cfg.kv_cache_dir);

    mm::ModelStorage model_storage(cfg.models_dir);
    state.set_storage(model_storage.list_models(), model_storage.free_space_mb());

    const char* repo_url_env = std::getenv("MM_LLAMA_REPO_URL");
    const char* install_root_env = std::getenv("MM_LLAMA_INSTALL_ROOT");
    const char* updater_log_dir_env = std::getenv("MM_LLAMA_UPDATER_LOG_DIR");

    mm::LlamaRuntimeManager::Options runtime_opts;
    runtime_opts.repo_url = (repo_url_env && *repo_url_env)
        ? std::string(repo_url_env)
        : "https://github.com/ggml-org/llama.cpp.git";
    runtime_opts.install_root = (install_root_env && *install_root_env)
        ? fs::path(install_root_env)
        : (fs::path(cfg.data_dir) / "llama.cpp");
    runtime_opts.metadata_file = fs::path(cfg.data_dir) / "llama_runtime.json";
    runtime_opts.log_dir = (updater_log_dir_env && *updater_log_dir_env)
        ? fs::path(updater_log_dir_env)
        : (fs::path(cfg.log_file).parent_path() / "llama-updater");
    mm::LlamaRuntimeManager llama_runtime(state, runtime_opts);

    llama_runtime.set_binary_ready_callback([&](const std::string& out_path) {
        if (out_path.empty()) return;
        slot_mgr.set_llama_server_path(out_path);
        state.set_llama_server_path(out_path);
        set_process_env_llama_path(out_path);
        if (!save_cached_llama_path(llama_cache_file, out_path)) {
            MM_WARN("Failed to persist llama path cache: {}", llama_cache_file.string());
        }
        MM_INFO("Applied updated llama-server path: {}", out_path);
    });

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
    llama_runtime.set_log_callback([&](const std::string& line) {
        const std::string out = "[llama-updater] " + line;
        append_runtime_log(out);
        if (ui_ptr) ui_ptr->append_log(out);
    });

    // ── API server ────────────────────────────────────────────────────────────

    mm::NodeApiServer api_server(state, slot_mgr, model_storage, llama_runtime,
                                 cfg.control_url, cfg.pairing_key);
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
    if (args.mode == NodeRunMode::Tui) {
        std::string msg;
        llama_runtime.check_update(&msg, /*force_remote=*/false);
        MM_INFO("llama runtime check: {}", msg);
    }

    // ── UDP discovery broadcaster ─────────────────────────────────────────────
    const char* self_env = std::getenv("MM_SELF_URL");
    std::string broadcast_url = self_env
        ? std::string(self_env)
        : "http://127.0.0.1:" + std::to_string(cfg.listen_port);
    std::string broadcast_id = mm::util::generate_uuid();

    mm::NodeDiscoveryBroadcaster broadcaster;
    broadcaster.start(broadcast_url, broadcast_id, cfg.discovery_port);
    MM_INFO("Discovery broadcaster started on port {} (id={})",
            cfg.discovery_port, broadcast_id);

    // ── UI ────────────────────────────────────────────────────────────────────

    auto request_llama_update = [&]() {
        mm::HttpClient self("http://127.0.0.1:" + std::to_string(cfg.listen_port));
        self.set_bearer_token(initial_key);
        auto r = self.post("/api/node/llama/update",
                           nlohmann::json{{"build", true}, {"force", false}});
        if (!r.ok()) {
            MM_WARN("Node UI/CLI update request failed (HTTP {}): {}", r.status, r.body);
            return;
        }
        MM_INFO("Node UI/CLI update request accepted: {}", r.body);
    };
    auto request_model_pull = [&](const std::string& model_filename, std::string* out_message) -> bool {
        const std::string filename = mm::canonical_model_filename(model_filename);
        if (!mm::is_safe_model_filename(filename)) {
            if (out_message) *out_message = "invalid model filename";
            return false;
        }

        mm::HttpClient self("http://127.0.0.1:" + std::to_string(cfg.listen_port));
        self.set_bearer_token(initial_key);
        auto r = self.post("/api/node/models/pull",
                           nlohmann::json{{"model_filename", filename}, {"force", false}});
        if (!r.ok()) {
            std::string msg = "pull failed (HTTP " + std::to_string(r.status) + ")";
            try {
                auto j = nlohmann::json::parse(r.body);
                std::string err = j.value("error", std::string{});
                if (!err.empty()) msg += ": " + err;
            } catch (...) {}
            if (out_message) *out_message = msg;
            return false;
        }

        if (out_message) *out_message = "pull requested: " + filename;
        return true;
    };
    auto request_model_delete = [&](const std::string& model_filename, std::string* out_message) -> bool {
        const std::string filename = mm::canonical_model_filename(model_filename);
        if (!mm::is_safe_model_filename(filename)) {
            if (out_message) *out_message = "invalid model filename";
            return false;
        }

        mm::HttpClient self("http://127.0.0.1:" + std::to_string(cfg.listen_port));
        self.set_bearer_token(initial_key);
        auto r = self.del("/api/node/models/" + filename);
        if (!r.ok()) {
            std::string msg = "delete failed (HTTP " + std::to_string(r.status) + ")";
            try {
                auto j = nlohmann::json::parse(r.body);
                std::string err = j.value("error", std::string{});
                if (!err.empty()) msg += ": " + err;
            } catch (...) {}
            if (out_message) *out_message = msg;
            return false;
        }

        if (out_message) *out_message = "deleted " + filename;
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
            request_llama_update,
            request_model_pull,
            request_model_delete);
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

    state.stop_metrics_poll();
    slot_mgr.unload_all();

    MM_INFO("mantic-mind node stopped");
    return 0;
}
