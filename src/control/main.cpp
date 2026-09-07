#include "common/config_file.hpp"
#include "common/cli_repl.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "control/control_config.hpp"
#include "control/agent_manager.hpp"
#include "control/model_registry.hpp"
#include "control/node_registry.hpp"
#include "control/agent_scheduler.hpp"
#include "control/agent_queue.hpp"
#include "control/control_api_server.hpp"
#include "control/control_ui.hpp"
#include "control/engine_config_store.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <functional>
#include <memory>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/file.h>
#  include <unistd.h>
#endif
// ── Config loading ─────────────────────────────────────────────────────────────
// Priority: config file < environment variables.

static mm::ControlConfig load_config(
    std::string* loaded_cfg_path = nullptr,
    const std::string& cfg_name = "mantic-mind-control.toml") {
    mm::ControlConfig cfg;
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

    const char* control_cfg_env = std::getenv("MM_CONTROL_CONFIG_FILE");
    if (control_cfg_env && *control_cfg_env) add_candidate(fs::path(control_cfg_env));
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
        cfg.listen_port = static_cast<uint16_t>(
            file.get_int("listen_port", static_cast<int>(cfg.listen_port)));
        cfg.openai_compat_port = static_cast<uint16_t>(
            file.get_int("openai_compat_port", static_cast<int>(cfg.openai_compat_port)));
        cfg.data_dir    = file.get("data_dir",  cfg.data_dir);
        cfg.admission_python      = file.get("admission_python",      cfg.admission_python);
        cfg.admission_tools_dir   = file.get("admission_tools_dir",   cfg.admission_tools_dir);
        cfg.admission_soma_path   = file.get("soma_path",             cfg.admission_soma_path);
        cfg.containers_dir        = file.get("containers_dir",        cfg.containers_dir);
        cfg.sources_dir           = file.get("sources_dir",           cfg.sources_dir);
        cfg.admission_allow_pickle =
            file.get_bool("admission_allow_pickle", cfg.admission_allow_pickle);
        cfg.admission_max_concurrent =
            file.get_int("admission_max_concurrent", cfg.admission_max_concurrent);
        cfg.admission_quant       = file.get("admission_quant",       cfg.admission_quant);
        cfg.admission_expert_down = file.get("admission_expert_down", cfg.admission_expert_down);
        cfg.log_file    = file.get("log_file",   cfg.log_file);
        cfg.node_health_poll_interval_s = static_cast<uint32_t>(
            file.get_int("node_health_poll_interval_s",
                         static_cast<int>(cfg.node_health_poll_interval_s)));
        cfg.node_offline_after_s = static_cast<uint32_t>(
            file.get_int("node_offline_after_s",
                         static_cast<int>(cfg.node_offline_after_s)));
        cfg.models_dir     = file.get("models_dir",     cfg.models_dir);
        cfg.external_api_token = file.get("external_api_token", cfg.external_api_token);
        cfg.tts.enabled = file.get_bool("tts_enabled", cfg.tts.enabled);
        cfg.tts.service_url = file.get("tts_service_url", cfg.tts.service_url);
        cfg.tts.service_command = file.get("tts_service_command", cfg.tts.service_command);
        cfg.tts.cache_dir = file.get("tts_cache_dir", cfg.tts.cache_dir);
        cfg.tts.voice_design_model_id =
            file.get("tts_voice_design_model_id", cfg.tts.voice_design_model_id);
        cfg.tts.clone_model_id = file.get("tts_clone_model_id", cfg.tts.clone_model_id);
        cfg.tts.custom_voice_model_id =
            file.get("tts_custom_voice_model_id", cfg.tts.custom_voice_model_id);
        cfg.tts.cache_ttl_ms = file.get_int(
            "tts_cache_ttl_ms",
            static_cast<int>(cfg.tts.cache_ttl_ms));
        cfg.tts.timeout_s = file.get_int("tts_timeout_s", cfg.tts.timeout_s);
        cfg.pairing_key    = file.get("pairing_key",    "");
        cfg.discovery_port = static_cast<uint16_t>(
            file.get_int("discovery_port", static_cast<int>(cfg.discovery_port)));
    }

    // Environment variables override file values.
    auto env = [](const char* name, const std::string& cur) -> std::string {
        const char* v = std::getenv(name);
        if (!v) return cur;
        std::string s(v);
        return s.empty() ? cur : s;
    };
    auto env_int = [](const char* name, int cur) -> int {
        const char* v = std::getenv(name);
        if (!v) return cur;
        try { return std::stoi(v); } catch (...) { return cur; }
    };
    auto env_bool = [](const char* name, bool cur) -> bool {
        const char* v = std::getenv(name);
        if (!v) return cur;
        std::string s = mm::util::to_lower(mm::util::trim(v));
        if (s == "true" || s == "yes" || s == "1" || s == "on") return true;
        if (s == "false" || s == "no" || s == "0" || s == "off") return false;
        return cur;
    };

    cfg.listen_port = static_cast<uint16_t>(
        env_int("MM_CONTROL_PORT", static_cast<int>(cfg.listen_port)));
    cfg.openai_compat_port = static_cast<uint16_t>(
        env_int("MM_OPENAI_COMPAT_PORT", static_cast<int>(cfg.openai_compat_port)));
    cfg.data_dir    = env("MM_DATA_DIR",    cfg.data_dir);
    cfg.admission_python      = env("MM_ADMISSION_PYTHON", cfg.admission_python);
    cfg.admission_tools_dir   = env("MM_ADMISSION_TOOLS",  cfg.admission_tools_dir);
    cfg.admission_soma_path   = env("MM_SOMA_PATH",        cfg.admission_soma_path);
    cfg.containers_dir        = env("MM_CONTAINERS_DIR",   cfg.containers_dir);
    cfg.sources_dir           = env("MM_SOURCES_DIR",      cfg.sources_dir);
    cfg.admission_allow_pickle =
        env_bool("MM_ADMISSION_ALLOW_PICKLE", cfg.admission_allow_pickle);
    // env_int already keeps the current value on an unparseable string, so a
    // garbage knob does not stop a cluster head from booting.
    cfg.admission_max_concurrent =
        env_int("MM_ADMISSION_MAX_CONCURRENT", cfg.admission_max_concurrent);
    cfg.admission_quant       = env("MM_ADMISSION_QUANT",  cfg.admission_quant);
    cfg.log_file    = env("MM_LOG_FILE",    cfg.log_file);
    cfg.models_dir  = env("MM_MODELS_DIR",  cfg.models_dir);
    cfg.external_api_token =
        env("MM_CONTROL_EXTERNAL_API_TOKEN", cfg.external_api_token);
    cfg.tts.enabled = env_bool("MM_TTS_ENABLED", cfg.tts.enabled);
    cfg.tts.service_url = env("MM_TTS_SERVICE_URL", cfg.tts.service_url);
    cfg.tts.service_command = env("MM_TTS_SERVICE_COMMAND", cfg.tts.service_command);
    cfg.tts.cache_dir = env("MM_TTS_CACHE_DIR", cfg.tts.cache_dir);
    cfg.tts.voice_design_model_id =
        env("MM_TTS_VOICE_DESIGN_MODEL_ID", cfg.tts.voice_design_model_id);
    cfg.tts.clone_model_id = env("MM_TTS_CLONE_MODEL_ID", cfg.tts.clone_model_id);
    cfg.tts.custom_voice_model_id =
        env("MM_TTS_CUSTOM_VOICE_MODEL_ID", cfg.tts.custom_voice_model_id);
    cfg.tts.cache_ttl_ms = env_int(
        "MM_TTS_CACHE_TTL_MS",
        static_cast<int>(cfg.tts.cache_ttl_ms));
    cfg.tts.timeout_s = env_int("MM_TTS_TIMEOUT_S", cfg.tts.timeout_s);
    cfg.pairing_key = env("MM_PAIRING_KEY", cfg.pairing_key);
    cfg.node_health_poll_interval_s = static_cast<uint32_t>(
        env_int("MM_POLL_INTERVAL_S",
                static_cast<int>(cfg.node_health_poll_interval_s)));
    cfg.node_offline_after_s = static_cast<uint32_t>(
        env_int("MM_NODE_OFFLINE_AFTER_S",
                static_cast<int>(cfg.node_offline_after_s)));
    cfg.discovery_port = static_cast<uint16_t>(
        env_int("MM_DISCOVERY_PORT", static_cast<int>(cfg.discovery_port)));

    return cfg;
}
// ── main ──────────────────────────────────────────────────────────────────────

namespace {

class ProcessSingletonLock {
public:
    ~ProcessSingletonLock() {
#ifdef _WIN32
        if (handle_) {
            ReleaseMutex(handle_);
            CloseHandle(handle_);
        }
#else
        if (fd_ >= 0) {
            flock(fd_, LOCK_UN);
            close(fd_);
        }
#endif
    }

    ProcessSingletonLock(const ProcessSingletonLock&) = delete;
    ProcessSingletonLock& operator=(const ProcessSingletonLock&) = delete;

    static std::unique_ptr<ProcessSingletonLock> try_acquire(const std::string& data_dir,
                                                             uint16_t port) {
        const std::string key = data_dir + "|" + std::to_string(port);
        const std::string suffix = std::to_string(std::hash<std::string>{}(key));

#ifdef _WIN32
        const std::string mutex_name = "Global\\mantic-mind-control-" + suffix;
        HANDLE handle = CreateMutexA(nullptr, TRUE, mutex_name.c_str());
        if (!handle || GetLastError() == ERROR_ALREADY_EXISTS) {
            if (handle) CloseHandle(handle);
            return nullptr;
        }

        auto lock = std::unique_ptr<ProcessSingletonLock>(new ProcessSingletonLock());
        lock->handle_ = handle;
        return lock;
#else
        const std::string lock_path = "/tmp/mantic-mind-control-" + suffix + ".lock";
        int fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd < 0) return nullptr;
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            close(fd);
            return nullptr;
        }

        auto lock = std::unique_ptr<ProcessSingletonLock>(new ProcessSingletonLock());
        lock->fd_ = fd;
        return lock;
#endif
    }

private:
    ProcessSingletonLock() = default;

#ifdef _WIN32
    HANDLE handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

} // namespace
enum class ControlRunMode { Tui, Cli };
enum class CliOutputMode { Text, Json };

struct ControlMainArgs {
    ControlRunMode mode = ControlRunMode::Tui;
    CliOutputMode output = CliOutputMode::Text;
    bool show_help = false;
    std::string error;
};

static ControlMainArgs parse_control_main_args(int argc, char** argv) {
    ControlMainArgs out;
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
            if (value == "tui") out.mode = ControlRunMode::Tui;
            else if (value == "cli") out.mode = ControlRunMode::Cli;
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
            if (value == "tui") out.mode = ControlRunMode::Tui;
            else if (value == "cli") out.mode = ControlRunMode::Cli;
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

static void print_control_usage() {
    std::cout << "Usage: mantic-mind-control [--mode tui|cli] [--output text|json] [--help]\n\n"
              << "Modes:\n"
              << "  tui  Default FTXUI terminal interface.\n"
              << "  cli  Interactive REPL suitable for terminal assistants.\n\n"
              << "Output:\n"
              << "  text Default human-readable CLI output.\n"
              << "  json Structured CLI output for automation.\n\n"
              << "CLI commands:\n"
              << "  nodes list\n"
              << "  nodes discovered\n"
              << "  nodes add <url> <api_key> [platform] [remember]\n"
              << "  nodes remove <node_id>\n"
              << "  nodes forget <node_id>\n"
              << "  nodes pair start <url>\n"
              << "  nodes pair complete <url> <nonce> <pin_or_psk> [remember]\n"
              << "  nodes pair psk <url> [psk] [remember]\n"
              << "  models list\n"
              << "  models show|plan|conformance|heat <model_id>\n"
              << "  models admit <source> [expert_gate] [expert_down] [group]\n"
              << "  models admissions\n"
              << "  models cancel <operation_id>\n"
              << "  models register <json>\n"
              << "  models reprofile <model_id>\n"
              << "  models verdict <model_id> <stream|hybrid|resident_only|reject> [reason]\n"
              << "  models delete <model_id>\n"
              << "  tokens list|create|delete ...\n"
              << "  performance [reset]\n"
              << "  engines show\n"
              << "  engines running\n"
              << "  engines heat|slots <engine_id>\n"
              << "  engines conform\n"
              << "  engines setup\n"
              << "  engines set primary <engine> [backup <engine>|backup none] [vllm-* <value>]\n"
              << "    --vllm-install-method auto|wheel|source|path  --vllm-version <version>\n"
              << "    --vllm-tp <gpus-per-node>  --vllm-pp <nodes>\n"
              << "    --vllm-experimental-gloo true|false (unsupported upstream)\n"
              << "  engines ray\n"
              << "  engines resync\n"
              << "  engines share <fingerprint> <target_node_id> [source_node_id]\n"
              << "  agents list|show|create|update|delete ...\n"
              << "  agents suspend|restore|release <agent_id>\n"
              << "  placements\n"
              << "  chat send <agent_id> <message> [conversation_id]\n"
              << "  curation conv ...\n"
              << "  curation mem ...\n"
              << "  curation local list|create|update|delete <agent_id> <conv_id> ...\n"
              << "  curation propose <agent_id>\n"
              << "  curation apply <agent_id> <json>\n"
              << "  voice show|proposals|propose <agent_id>\n"
              << "  voice approve|reject|sample <agent_id> <proposal_id>\n"
              << "  activity tail [n]\n"
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

    void fragment(const std::string& text) {
        std::lock_guard<std::mutex> lk(mu_);
        if (prompt_visible_) std::cout << '\r';
        prompt_visible_ = false;
        std::cout << text << std::flush;
    }

private:
    std::mutex mu_;
    std::string prompt_;
    bool prompt_visible_ = false;
};

static std::atomic<bool>* g_control_cli_stop = nullptr;

static void control_cli_signal_handler(int /*signal*/) {
    if (g_control_cli_stop) g_control_cli_stop->store(true);
}

static std::string summarize_http_error(const mm::HttpResponse& r) {
    std::string msg = "HTTP " + std::to_string(r.status);
    const std::string body = mm::util::trim(r.body);
    if (!body.empty()) msg += ": " + body;
    return msg;
}

static void run_control_cli(uint16_t listen_port,
                            CliOutputMode output_mode,
                            std::atomic<bool>& stop_flag) {
    CliPrinter printer("mm-control> ");
    printer.line("CLI mode active. Type 'help' for commands.");
    mm::HttpClient self("http://127.0.0.1:" + std::to_string(listen_port));

    const bool json_mode = output_mode == CliOutputMode::Json;

    auto print_help = [&]() {
        printer.line("Use --help for the full command list.");
        printer.line("Top-level groups: nodes, models, agents, chat, curation, activity, help, quit");
    };

    auto pretty_body = [&](const std::string& body) -> std::string {
        try {
            auto j = nlohmann::json::parse(body);
            return j.dump(2);
        } catch (...) {
            return body;
        }
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
        if (ok) {
            if (data.is_null() || data.empty()) printer.line("ok");
            else if (data.is_string()) printer.line(data.get<std::string>());
            else printer.block(data.dump(2));
        } else {
            printer.line("error: " + error);
        }
    };

    auto emit_http_result = [&](const std::string& command, const mm::HttpResponse& r) {
        if (r.ok()) emit_result(true, command, parse_or_wrap(r.body), "");
        else emit_result(false, command, nlohmann::json::object(), summarize_http_error(r));
    };

    // ── engine configuration ──────────────────────────────────────────────────
    //
    // Everything below drives the same /v1/cluster/engines/* routes the TUI
    // uses, so the rules have one implementation rather than a CLI copy that
    // drifts. `engines setup` is the CLI half of forced first-run
    // configuration: a headless deployment must be able to reach the same
    // decision the TUI's modal asks for, or `--mode cli` would be a way to run
    // a cluster that can never place anything.

    auto engine_config_missing = [&]() -> bool {
        const auto r = self.get("/v1/cluster/engines/config");
        if (!r.ok()) return false; // unreachable/unsupported: not our call to make
        try {
            return !nlohmann::json::parse(r.body).value("configured", false);
        } catch (...) {
            return false;
        }
    };

    // Engine ids any connected node reports it can run. The setup prompt offers
    // these rather than a hardcoded pair, so a node that grows a third engine
    // shows up here without an edit.
    auto known_engine_ids = [&]() -> std::vector<std::string> {
        std::vector<std::string> ids;
        const auto r = self.get("/v1/nodes");
        if (r.ok()) try {
            const auto j = nlohmann::json::parse(r.body);
            const auto& arr = j.contains("data") ? j.at("data") : j;
            for (const auto& n : arr) {
                if (!n.contains("engines")) continue;
                for (const auto& e : n.at("engines")) {
                    const auto id = e.value("engine_id", std::string{});
                    if (!id.empty() &&
                        std::find(ids.begin(), ids.end(), id) == ids.end())
                        ids.push_back(id);
                }
            }
        } catch (...) {
        }
        for (const std::string builtin : {"soma", "llama-cpp", "vllm"}) {
            if (std::find(ids.begin(), ids.end(), builtin) == ids.end())
                ids.push_back(builtin);
        }
        return ids;
    };

    auto put_engine_config = [&](const std::string& primary,
                                 const std::string& backup,
                                 const std::optional<mm::VllmEngineConfig>& vllm = std::nullopt,
                                 const std::string& vllm_install_method = {},
                                 const std::string& vllm_version = {}) {
        std::optional<mm::ClusterEngineConfig> existing;
        const auto current = self.get("/v1/cluster/engines/config");
        if (current.ok()) try {
            const auto root = nlohmann::json::parse(current.body);
            if (root.value("configured", false))
                existing = root.at("config").get<mm::ClusterEngineConfig>();
        } catch (...) {
        }
        nlohmann::json engines = nlohmann::json::array();
        auto spec = [&](const std::string& id) {
            nlohmann::json out{{"engine_id", id}};
            if (existing) {
                if (const auto* previous = existing->find(id)) out = *previous;
            }
            if (id == "vllm") {
                if (!vllm_install_method.empty())
                    out["install_method"] = vllm_install_method;
                else if (!out.contains("install_method"))
                    out["install_method"] = "auto";
                if (!vllm_version.empty()) out["version"] = vllm_version;
                out["vllm"] = vllm.value_or(mm::VllmEngineConfig{});
            }
            return out;
        };
        engines.push_back(spec(primary));
        if (!backup.empty()) engines.push_back(spec(backup));
        const nlohmann::json body{{"primary_engine", primary},
                                  {"backup_engine", backup},
                                  {"engines", engines},
                                  {"share_builds", existing
                                      ? existing->share_builds : true}};
        emit_http_result("engines set", self.put("/v1/cluster/engines/config", body));
    };

    auto run_engine_setup = [&]() {
        printer.line("");
        printer.line("  No cluster engine configuration exists.");
        printer.line("  Nothing can be placed until one is set — every node is waiting to be");
        printer.line("  told what to run.");
        printer.line("");

        auto ids = known_engine_ids();
        if (ids.empty()) {
            // Not an error: on a fresh install no node has registered yet.
            // Offering the two engines this build ships is better than refusing
            // to proceed until a node appears.
            ids = {"soma", "llama-cpp", "vllm"};
            printer.line("  (no node has reported its engines yet; offering the built-in set)");
        }
        printer.line("  Available engines: " + mm::util::join(ids, ", "));
        printer.line("");

        std::string primary;
        while (primary.empty()) {
            printer.line("  Primary engine? (" + mm::util::join(ids, "/") + ")");
            printer.print_prompt();
            std::string answer;
            if (!std::getline(std::cin, answer)) return;
            answer = mm::util::trim(answer);
            if (answer.empty()) continue;
            if (std::find(ids.begin(), ids.end(), answer) == ids.end()) {
                printer.line("  '" + answer + "' is not one of: " + mm::util::join(ids, ", "));
                continue;
            }
            primary = answer;
        }

        // llama.cpp is the DEFAULT backup, not a requirement. Offering "none"
        // explicitly is the point: a Soma-only cluster should not compile a
        // llama-server it will never launch, and that has to be a choice the
        // operator can actually make here.
        std::string backup = (primary == mm::kDefaultBackupEngine)
                                 ? std::string{}
                                 : std::string(mm::kDefaultBackupEngine);
        printer.line("");
        printer.line(backup.empty()
                         ? "  Backup engine? (none available — primary is the default backup)"
                         : "  Backup engine? [" + backup + "] — enter an engine, or 'none'");
        if (!backup.empty()) {
            printer.print_prompt();
            std::string answer;
            if (std::getline(std::cin, answer)) {
                answer = mm::util::trim(answer);
                if (mm::util::to_lower(answer) == "none") backup.clear();
                else if (!answer.empty()) backup = answer;
            }
        }

        std::optional<mm::VllmEngineConfig> vllm;
        std::string vllm_install_method;
        std::string vllm_version;
        if (primary == "vllm" || backup == "vllm") {
            mm::VllmEngineConfig profile;
            auto prompt_int = [&](const std::string& label, int current) {
                printer.line("  " + label + " [" + std::to_string(current) + "]");
                printer.print_prompt();
                std::string answer;
                if (!std::getline(std::cin, answer)) return current;
                answer = mm::util::trim(answer);
                if (answer.empty()) return current;
                try { return std::stoi(answer); } catch (...) { return current; }
            };
            auto prompt_string = [&](const std::string& label, std::string current) {
                printer.line("  " + label + " [" +
                             (current.empty() ? std::string{"none"} : current) + "]");
                printer.print_prompt();
                std::string answer;
                if (!std::getline(std::cin, answer)) return current;
                answer = mm::util::trim(answer);
                return answer.empty() ? current : answer;
            };
            auto prompt_bool = [&](const std::string& label, bool current) {
                printer.line("  " + label + (current ? " [yes]" : " [no]"));
                printer.print_prompt();
                std::string answer;
                if (!std::getline(std::cin, answer)) return current;
                answer = mm::util::to_lower(mm::util::trim(answer));
                if (answer.empty()) return current;
                return answer == "yes" || answer == "y" || answer == "true" ||
                       answer == "1" || answer == "on";
            };
            vllm_install_method = prompt_string(
                "vLLM install method (auto/wheel/source/path)", "auto");
            vllm_version = prompt_string("vLLM version", "latest");
            profile.max_model_len = prompt_int("vLLM max model length", profile.max_model_len);
            profile.max_num_seqs = prompt_int("vLLM max sequences", profile.max_num_seqs);
            profile.max_num_batched_tokens = prompt_int(
                "vLLM max batched tokens (-1 = automatic)",
                profile.max_num_batched_tokens);
            profile.tensor_parallel_size = prompt_int(
                "vLLM tensor parallel GPUs per node", profile.tensor_parallel_size);
            profile.pipeline_parallel_size = prompt_int(
                "vLLM pipeline parallel nodes", profile.pipeline_parallel_size);
            printer.line("  vLLM GPU memory utilization [" +
                         std::to_string(profile.gpu_memory_utilization) + "]");
            printer.print_prompt();
            std::string gpu_answer;
            if (std::getline(std::cin, gpu_answer)) {
                gpu_answer = mm::util::trim(gpu_answer);
                if (!gpu_answer.empty()) {
                    try { profile.gpu_memory_utilization = std::stod(gpu_answer); }
                    catch (...) {}
                }
            }
            profile.dtype = prompt_string("vLLM dtype", profile.dtype);
            profile.quantization = prompt_string(
                "vLLM quantization (blank keeps none)", profile.quantization);
            profile.trust_remote_code = prompt_bool(
                "Trust remote model code?", profile.trust_remote_code);
            profile.enable_prefix_caching = prompt_bool(
                "Enable prefix caching?", profile.enable_prefix_caching);
            profile.enable_auto_tool_choice = prompt_bool(
                "Enable automatic tool choice?", profile.enable_auto_tool_choice);
            profile.enable_sleep_mode = prompt_bool(
                "Enable sleep mode?", profile.enable_sleep_mode);
            profile.tool_call_parser = prompt_string(
                "Tool-call parser (blank keeps none)", profile.tool_call_parser);
            const auto extras = prompt_string(
                "Additional vLLM args, comma separated (blank keeps none)", {});
            for (const auto& raw : mm::util::split(extras, ',')) {
                const auto arg = mm::util::trim(raw);
                if (!arg.empty()) profile.extra_args.push_back(arg);
            }
            if (profile.pipeline_parallel_size > 1) {
                profile.allow_experimental_gloo = prompt_bool(
                    "EXPERIMENTAL: allow Gloo when NCCL is unavailable?",
                    profile.allow_experimental_gloo);
            }
            vllm = std::move(profile);
        }

        printer.line("");
        printer.line("  primary: " + primary);
        printer.line("  backup:  " + (backup.empty() ? "(none)" : backup));
        put_engine_config(primary, backup, vllm,
                          vllm_install_method, vllm_version);
    };

    // Forced on entry, exactly as the TUI opens its Engines tab modally. The
    // API stays up either way — this blocks the operator, not the process, so
    // an automated deployment can still PUT the configuration from outside.
    if (engine_config_missing()) run_engine_setup();

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

        if (cmd0 == "engines") {
            if (tokens.size() < 2) {
                printer.line("usage: engines show|conform|ray|setup|set|resync|share|running|"
                             "heat|slots|provision|check-update|switch|diagnose|recover ...");
                continue;
            }
            const std::string sub = mm::util::to_lower(tokens[1]);
            if (sub == "show") {
                emit_http_result("engines show", self.get("/v1/cluster/engines/config"));
                continue;
            }
            if (sub == "conform") {
                emit_http_result("engines conform",
                                 self.get("/v1/cluster/engines/conformance"));
                continue;
            }
            if (sub == "ray") {
                emit_http_result("engines ray", self.get("/v1/cluster/engines/ray"));
                continue;
            }
            if (sub == "setup") {
                run_engine_setup();
                continue;
            }
            if (sub == "resync") {
                emit_http_result("engines resync",
                                 self.post("/v1/cluster/engines/resync", nlohmann::json::object()));
                continue;
            }
            if (sub == "set") {
                // engines set primary <id> [backup <id>|backup none]
                std::string primary, backup;
                bool have_backup = false;
                mm::VllmEngineConfig profile;
                std::optional<mm::ClusterEngineConfig> current_cluster;
                const auto current_response =
                    self.get("/v1/cluster/engines/config");
                if (current_response.ok()) try {
                    const auto root = nlohmann::json::parse(current_response.body);
                    if (root.value("configured", false)) {
                        current_cluster =
                            root.at("config").get<mm::ClusterEngineConfig>();
                        if (const auto* spec = current_cluster->find("vllm"))
                            profile = mm::effective_vllm_config(*spec);
                    }
                } catch (...) {
                }
                bool vllm_option = false;
                std::string vllm_install_method;
                std::string vllm_version;
                auto parse_bool = [](const std::string& raw) {
                    const auto v = mm::util::to_lower(mm::util::trim(raw));
                    if (v == "1" || v == "true" || v == "yes" || v == "on")
                        return true;
                    if (v == "0" || v == "false" || v == "no" || v == "off")
                        return false;
                    throw std::invalid_argument("expected true|false");
                };
                bool option_error = false;
                std::string option_error_detail;
                try {
                for (std::size_t i = 2; i + 1 < tokens.size(); i += 2) {
                    std::string key = mm::util::to_lower(tokens[i]);
                    if (key.rfind("--", 0) == 0) key.erase(0, 2);
                    if (key == "primary") primary = tokens[i + 1];
                    else if (key == "backup") {
                        have_backup = true;
                        backup = mm::util::to_lower(tokens[i + 1]) == "none" ? std::string{}
                                                                            : tokens[i + 1];
                    }
                    else if (key == "vllm-install-method") {
                        vllm_install_method = tokens[i + 1];
                    } else if (key == "vllm-version") {
                        vllm_version = tokens[i + 1];
                    }
                    else if (key == "vllm-max-model-len") {
                        profile.max_model_len = std::stoi(tokens[i + 1]); vllm_option = true;
                    } else if (key == "vllm-max-num-seqs") {
                        profile.max_num_seqs = std::stoi(tokens[i + 1]); vllm_option = true;
                    } else if (key == "vllm-max-num-batched-tokens") {
                        profile.max_num_batched_tokens = std::stoi(tokens[i + 1]); vllm_option = true;
                    } else if (key == "vllm-tp") {
                        profile.tensor_parallel_size = std::stoi(tokens[i + 1]); vllm_option = true;
                    } else if (key == "vllm-pp") {
                        profile.pipeline_parallel_size = std::stoi(tokens[i + 1]); vllm_option = true;
                    } else if (key == "vllm-gpu-memory-utilization") {
                        profile.gpu_memory_utilization = std::stod(tokens[i + 1]); vllm_option = true;
                    } else if (key == "vllm-dtype") {
                        profile.dtype = tokens[i + 1]; vllm_option = true;
                    } else if (key == "vllm-quantization") {
                        profile.quantization = tokens[i + 1]; vllm_option = true;
                    } else if (key == "vllm-trust-remote-code") {
                        profile.trust_remote_code = parse_bool(tokens[i + 1]); vllm_option = true;
                    } else if (key == "vllm-prefix-caching") {
                        profile.enable_prefix_caching = parse_bool(tokens[i + 1]); vllm_option = true;
                    } else if (key == "vllm-auto-tool-choice") {
                        profile.enable_auto_tool_choice = parse_bool(tokens[i + 1]); vllm_option = true;
                    } else if (key == "vllm-sleep-mode") {
                        profile.enable_sleep_mode = parse_bool(tokens[i + 1]); vllm_option = true;
                    } else if (key == "vllm-tool-call-parser") {
                        profile.tool_call_parser = tokens[i + 1]; vllm_option = true;
                    } else if (key == "vllm-extra-arg") {
                        profile.extra_args.push_back(tokens[i + 1]); vllm_option = true;
                    } else if (key == "vllm-experimental-gloo") {
                        profile.allow_experimental_gloo = parse_bool(tokens[i + 1]); vllm_option = true;
                    }
                }
                } catch (const std::exception& e) {
                    option_error = true;
                    option_error_detail = e.what();
                }
                if (option_error) {
                    printer.line("invalid engines set option: " + option_error_detail);
                    continue;
                }
                if (primary.empty()) {
                    printer.line("usage: engines set primary <engine> [backup <engine>|backup none]");
                    continue;
                }
                // Unstated backup keeps the current one rather than clearing
                // it: `engines set primary soma` should not silently drop a
                // configured fallback. Clearing takes the explicit word.
                if (!have_backup) {
                    if (current_cluster) backup = current_cluster->backup_engine;
                }
                if (backup == primary) backup.clear();
                if ((primary == "vllm" || backup == "vllm") && !vllm_option) {
                    if (current_cluster)
                        if (const auto* spec = current_cluster->find("vllm"))
                            profile = mm::effective_vllm_config(*spec);
                }
                put_engine_config(primary, backup,
                                  (primary == "vllm" || backup == "vllm")
                                      ? std::optional<mm::VllmEngineConfig>(profile)
                                      : std::nullopt,
                                  vllm_install_method, vllm_version);
                continue;
            }
            // Live engine PROCESSES, as opposed to `engines show`, which is the
            // cluster's engine POLICY. Two different resources one word apart —
            // see the /v1/engines vs /v1/cluster/engines split.
            if (sub == "running") {
                emit_http_result("engines running", self.get("/v1/engines"));
                continue;
            }
            if (sub == "heat" || sub == "slots") {
                if (tokens.size() < 3) {
                    printer.line("usage: engines " + sub + " <engine_id>");
                    continue;
                }
                emit_http_result("engines " + sub,
                                 self.get("/v1/engines/" + tokens[2] + "/" + sub));
                continue;
            }
            if (sub == "share") {
                if (tokens.size() < 4) {
                    printer.line("usage: engines share <fingerprint> <target_node_id> [source_node_id]");
                    continue;
                }
                nlohmann::json body{{"fingerprint", tokens[2]},
                                    {"target_node_id", tokens[3]}};
                if (tokens.size() > 4) body["source_node_id"] = tokens[4];
                emit_http_result("engines share",
                                 self.post("/v1/cluster/engines/share", body));
                continue;
            }
            // Per-node engine actions. The verb an operator reaches for when a
            // node reports `failed` and `engines resync` did nothing — which it
            // will, because resync skips any node already at the current config
            // version and a node bumps that on ACCEPTING a config, not on
            // conforming to one.
            //
            // The node answers 202 and works in the background, so these print a
            // start receipt rather than an outcome. `engines conform` is where
            // the outcome shows up.
            if (sub == "provision" || sub == "check-update" || sub == "diagnose" ||
                sub == "switch" || sub == "recover") {
                if (tokens.size() < 3) {
                    printer.line("usage: engines " + sub + " <node_id>" +
                                 (sub == "switch" ? " <variant>"
                                  : sub == "recover"
                                      ? " [retry|target|compile-anyway|release <variant>]"
                                      : ""));
                    continue;
                }
                nlohmann::json body = nlohmann::json::object();
                if (sub == "switch") {
                    if (tokens.size() < 4) {
                        printer.line("usage: engines switch <node_id> <variant>");
                        continue;
                    }
                    body["variant"] = tokens[3];
                }
                if (sub == "recover") {
                    // Defaulted, because retry is the one an operator wants from
                    // a REPL; the report-driven actions need the node's
                    // troubleshooting matrix in front of you to choose between.
                    body["action"] = tokens.size() > 3 ? tokens[3] : std::string("retry");
                    if (tokens.size() > 4) body["variant"] = tokens[4];
                }
                emit_http_result(
                    "engines " + sub,
                    self.post("/v1/cluster/engines/nodes/" + tokens[2] + "/" + sub, body));
                continue;
            }
            printer.line("usage: engines show|conform|setup|set|resync|share|running|"
                         "heat|slots|provision|check-update|switch|diagnose|recover ...");
            continue;
        }

        if (cmd0 == "nodes") {
            if (tokens.size() < 2) {
                printer.line("usage: nodes list|discovered|add|remove|forget|pair ...");
                continue;
            }
            const std::string sub = mm::util::to_lower(tokens[1]);
            if (sub == "list") {
                emit_http_result("nodes list", self.get("/v1/nodes"));
                continue;
            }
            if (sub == "discovered") {
                emit_http_result("nodes discovered", self.get("/v1/nodes/discovered"));
                continue;
            }
            if (sub == "add") {
                if (tokens.size() < 4) {
                    printer.line("usage: nodes add <url> <api_key> [platform] [remember]");
                    continue;
                }
                const std::string platform = tokens.size() >= 5 ? tokens[4] : "";
                bool remember = false;
                if (tokens.size() >= 6 && !mm::cli::parse_bool_token(tokens[5], &remember)) {
                    printer.line("error: remember must be true|false");
                    continue;
                }
                emit_http_result(
                    "nodes add",
                    self.post("/v1/nodes",
                              nlohmann::json{{"url", tokens[2]},
                                             {"api_key", tokens[3]},
                                             {"platform", platform},
                                             {"remember", remember}}));
                continue;
            }
            if (sub == "remove") {
                if (tokens.size() < 3) {
                    printer.line("usage: nodes remove <node_id>");
                    continue;
                }
                emit_http_result("nodes remove", self.del("/v1/nodes/" + tokens[2]));
                continue;
            }
            if (sub == "forget") {
                if (tokens.size() < 3) {
                    printer.line("usage: nodes forget <node_id>");
                    continue;
                }
                emit_http_result("nodes forget", self.post("/v1/nodes/" + tokens[2] + "/forget",
                                                           nlohmann::json::object()));
                continue;
            }
            if (sub == "pair") {
                if (tokens.size() < 4) {
                    printer.line("usage: nodes pair start|complete|psk ...");
                    continue;
                }
                const std::string pair_sub = mm::util::to_lower(tokens[2]);
                if (pair_sub == "start") {
                    emit_http_result("nodes pair start",
                                     self.post("/v1/nodes/pair/start", nlohmann::json{{"url", tokens[3]}}));
                    continue;
                }
                if (pair_sub == "complete") {
                    if (tokens.size() < 6) {
                        printer.line("usage: nodes pair complete <url> <nonce> <pin_or_psk> [remember]");
                        continue;
                    }
                    bool remember = false;
                    if (tokens.size() >= 7 && !mm::cli::parse_bool_token(tokens[6], &remember)) {
                        printer.line("error: remember must be true|false");
                        continue;
                    }
                    emit_http_result(
                        "nodes pair complete",
                        self.post("/v1/nodes/pair/complete",
                                  nlohmann::json{{"url", tokens[3]},
                                                 {"nonce", tokens[4]},
                                                 {"pin_or_psk", tokens[5]},
                                                 {"remember", remember}}));
                    continue;
                }
                if (pair_sub == "psk") {
                    bool remember = false;
                    if (tokens.size() >= 6 && !mm::cli::parse_bool_token(tokens[5], &remember)) {
                        printer.line("error: remember must be true|false");
                        continue;
                    }
                    nlohmann::json body{{"url", tokens[3]}};
                    if (tokens.size() >= 5) body["psk"] = tokens[4];
                    body["remember"] = remember;
                    emit_http_result("nodes pair psk", self.post("/v1/nodes/pair/psk", body));
                    continue;
                }
                printer.line("error: unknown nodes pair subcommand");
                continue;
            }
            printer.line("error: unknown nodes subcommand");
            continue;
        }

        if (cmd0 == "models") {
            const std::string sub =
                tokens.size() < 2 ? std::string{"list"} : mm::util::to_lower(tokens[1]);
            if (sub == "list") {
                emit_http_result("models list", self.get("/v1/models"));
                continue;
            }
            if (sub == "admissions") {
                emit_http_result("models admissions", self.get("/v1/models/admissions"));
                continue;
            }
            // The registry's administrative half. All four are `operator` on the
            // API and had no CLI form, so a headless deployment could admit a
            // model and then neither correct its verdict nor remove it.
            if (sub == "show") {
                if (tokens.size() < 3) {
                    printer.line("usage: models show <model_id>");
                    continue;
                }
                emit_http_result("models show", self.get("/v1/models/" + tokens[2]));
                continue;
            }
            if (sub == "plan" || sub == "conformance" || sub == "heat") {
                if (tokens.size() < 3) {
                    printer.line("usage: models " + sub + " <model_id>");
                    continue;
                }
                emit_http_result("models " + sub, self.get("/v1/models/" + tokens[2] + "/" + sub));
                continue;
            }
            if (sub == "delete") {
                if (tokens.size() < 3) {
                    printer.line("usage: models delete <model_id>");
                    continue;
                }
                emit_http_result("models delete", self.del("/v1/models/" + tokens[2]));
                continue;
            }
            if (sub == "reprofile") {
                if (tokens.size() < 3) {
                    printer.line("usage: models reprofile <model_id>");
                    continue;
                }
                // SSE like admit, and handled the same way: read the operation
                // id off the first frame and let `models admissions` watch it.
                // Re-profiling re-derives a verdict from the SAME bytes — it
                // never requantizes, so arch_hash and every KV checkpoint
                // written against it survive.
                std::string op_id;
                int status = 0;
                std::string error_body;
                self.stream_post(
                    "/v1/models/" + tokens[2] + "/reprofile",
                    nlohmann::json::object(),
                    mm::HttpClient::capture_first_field("operation_id", op_id),
                    &status,
                    &error_body);
                if (!op_id.empty())
                    emit_result(
                        true,
                        "models reprofile",
                        nlohmann::json{{"operation_id", op_id}, {"watch", "models admissions"}},
                        "");
                else
                    emit_result(false,
                                "models reprofile",
                                nlohmann::json::object(),
                                error_body.empty() ? "reprofile reported no operation id"
                                                   : error_body);
                continue;
            }
            if (sub == "verdict") {
                if (tokens.size() < 4) {
                    printer.line("usage: models verdict <model_id> "
                                 "<stream|hybrid|resident_only|reject> [reason]");
                    continue;
                }
                nlohmann::json body{{"verdict", tokens[3]}};
                if (tokens.size() > 4) body["reason"] = mm::cli::join_tokens(tokens, 4);
                emit_http_result("models verdict",
                                 self.put("/v1/models/" + tokens[2] + "/verdict", body));
                continue;
            }
            if (sub == "register") {
                if (tokens.size() < 3) {
                    printer.line("usage: models register <json>");
                    continue;
                }
                try {
                    emit_http_result(
                        "models register",
                        self.post("/v1/models",
                                  nlohmann::json::parse(mm::cli::join_tokens(tokens, 2))));
                } catch (const std::exception& e) {
                    printer.line(std::string("error: invalid JSON: ") + e.what());
                }
                continue;
            }
            if (sub == "cancel") {
                if (tokens.size() < 3) {
                    printer.line("usage: models cancel <operation_id>");
                    continue;
                }
                emit_http_result("models cancel",
                                 self.post("/v1/models/admissions/" + tokens[2] + "/cancel",
                                           nlohmann::json::object()));
                continue;
            }
            if (sub == "admit") {
                if (tokens.size() < 3) {
                    printer.line("usage: models admit <source> [expert_gate] [expert_down] [group]");
                    continue;
                }
                nlohmann::json body{{"source", tokens[2]}};
                nlohmann::json quant = nlohmann::json::object();
                if (tokens.size() > 3 && !tokens[3].empty()) quant["expert_gate"] = tokens[3];
                if (tokens.size() > 4 && !tokens[4].empty()) quant["expert_down"] = tokens[4];
                if (tokens.size() > 5) {
                    try { quant["group"] = std::stoi(tokens[5]); } catch (...) {}
                }
                if (!quant.empty()) body["quantization"] = quant;

                // The route answers ONLY as a stream, and this admission runs
                // for hours. So the id is read off the first frame and the
                // stream dropped — control logs "client disconnected;
                // conversion continues" and the worker is detached precisely so
                // it outlives the request. `models admissions` is how you watch
                // it afterwards, which is also what makes this survive closing
                // the REPL.
                std::string op_id;
                int status = 0;
                std::string error_body;
                const bool connected = self.stream_post(
                    "/v1/models/admit", body,
                    mm::HttpClient::capture_first_field("operation_id", op_id),
                    &status, &error_body);

                if (!op_id.empty()) {
                    emit_result(true, "models admit",
                                nlohmann::json{{"operation_id", op_id},
                                               {"source", tokens[2]},
                                               {"watch", "models admissions"}},
                                "");
                } else {
                    std::string why = error_body;
                    try {
                        const auto j = nlohmann::json::parse(error_body);
                        if (j.contains("error")) why = j["error"].get<std::string>();
                    } catch (...) {
                    }
                    if (why.empty())
                        why = connected ? "admission started but reported no operation id"
                                        : "cannot reach control (HTTP " +
                                              std::to_string(status) + ")";
                    emit_result(false, "models admit", nlohmann::json::object(), why);
                }
                continue;
            }
            printer.line("usage: models list|show|plan|conformance|heat|admit|admissions|"
                         "cancel|register|reprofile|verdict|delete ...");
            continue;
        }

        if (cmd0 == "agents") {
            if (tokens.size() < 2) {
                printer.line("usage: agents list|show|create|update|delete|backend|"
                             "suspend|restore|release ...");
                continue;
            }
            const std::string sub = mm::util::to_lower(tokens[1]);
            if (sub == "list") {
                emit_http_result("agents list", self.get("/v1/agents"));
                continue;
            }
            if (sub == "show") {
                if (tokens.size() < 3) {
                    printer.line("usage: agents show <agent_id>");
                    continue;
                }
                auto r = self.get("/v1/agents/" + tokens[2]);
                emit_http_result("agents show", r);
                continue;
            }
            if (sub == "create") {
                if (tokens.size() < 3) {
                    printer.line("usage: agents create <json>");
                    continue;
                }
                auto payload = mm::cli::join_tokens(tokens, 2);
                try {
                    auto j = nlohmann::json::parse(payload);
                    auto r = self.post("/v1/agents", j);
                    emit_http_result("agents create", r);
                } catch (const std::exception& e) {
                    printer.line(std::string("error: invalid JSON: ") + e.what());
                }
                continue;
            }
            if (sub == "update") {
                if (tokens.size() < 4) {
                    printer.line("usage: agents update <agent_id> <json>");
                    continue;
                }
                auto payload = mm::cli::join_tokens(tokens, 3);
                try {
                    auto j = nlohmann::json::parse(payload);
                    auto r = self.put("/v1/agents/" + tokens[2], j);
                    emit_http_result("agents update", r);
                } catch (const std::exception& e) {
                    printer.line(std::string("error: invalid JSON: ") + e.what());
                }
                continue;
            }
            // Which ENGINE serves this agent — the one place an operator can
            // overrule a verdict. `operator` on the API, and it had no CLI form,
            // so overruling a verdict was a TUI-only act.
            if (sub == "backend") {
                if (tokens.size() < 4) {
                    printer.line("usage: agents backend <agent_id> <auto|soma|fallback>");
                    continue;
                }
                emit_http_result("agents backend",
                                 self.put("/v1/agents/" + tokens[2] + "/backend",
                                          nlohmann::json{{"backend_override", tokens[3]}}));
                continue;
            }
            // Placement lifecycle. These three had no /v1 route at all until the
            // parity audit: the scheduler could do them and the node API exposed
            // them, and no client could ask for them.
            if (sub == "suspend" || sub == "restore" || sub == "release") {
                if (tokens.size() < 3) {
                    printer.line("usage: agents " + sub + " <agent_id>");
                    continue;
                }
                emit_http_result("agents " + sub,
                                 self.post("/v1/agents/" + tokens[2] + "/" + sub,
                                           nlohmann::json::object()));
                continue;
            }
            if (sub == "delete") {
                if (tokens.size() < 3) {
                    printer.line("usage: agents delete <agent_id>");
                    continue;
                }
                auto r = self.del("/v1/agents/" + tokens[2]);
                emit_http_result("agents delete", r);
                continue;
            }
            printer.line("error: unknown agents subcommand");
            continue;
        }

        if (cmd0 == "chat") {
            if (tokens.size() < 4 || mm::util::to_lower(tokens[1]) != "send") {
                printer.line("usage: chat send <agent_id> <message> [conversation_id]");
                continue;
            }
            const std::string conv_hint = tokens.size() >= 5 ? tokens[4] : "";
            const std::string path = "/v1/agents/" + tokens[2] + "/chat";
            nlohmann::json body = {{"message", tokens[3]}};
            if (!conv_hint.empty()) body["conversation_id"] = conv_hint;
            bool done_seen = false;
            bool done_success = false;
            std::string done_error;
            std::string done_conv_id;
            bool printed_delta = false;
            int stream_status = 0;
            std::string stream_body;

            bool stream_ok = self.stream_post(path, body, [&](const std::string& data) -> bool {
                if (data == "[DONE]") return true;
                nlohmann::json j;
                try {
                    j = nlohmann::json::parse(data);
                } catch (...) {
                    return true;
                }
                const std::string type = j.value("type", std::string{});
                if (json_mode) {
                    nlohmann::json event{{"event", type}};
                    if (type == "delta") event["content"] = j.value("content", std::string{});
                    else if (type == "thinking") event["content"] = j.value("content", std::string{});
                    else if (type == "tool_call") {
                        event["name"] = j.value("name", std::string{});
                        event["arguments"] = j.value("arguments", std::string{});
                    } else if (type == "done") {
                        event["conv_id"] = j.value("conv_id", std::string{});
                        event["success"] = j.value("success", false);
                        if (j.contains("error")) event["error"] = j["error"];
                    }
                    printer.block(event.dump());
                } else {
                    if (type == "thinking") printer.line("[thinking] " + j.value("content", std::string{}));
                    else if (type == "delta") {
                        printer.fragment(j.value("content", std::string{}));
                        printed_delta = true;
                    } else if (type == "tool_call") {
                        printer.line("");
                        printer.line("[tool_call] " + j.value("name", std::string{}) +
                                     " args=" + j.value("arguments", std::string{}));
                    }
                }
                if (type == "done") {
                    done_seen = true;
                    done_success = j.value("success", false);
                    done_conv_id = j.value("conv_id", std::string{});
                    done_error = j.value("error", std::string{});
                }
                return true;
            }, &stream_status, &stream_body);

            if (!json_mode && printed_delta) printer.line("");
            const bool success = stream_ok && done_seen && done_success;
            if (success) {
                emit_result(true, "chat send", nlohmann::json{{"conv_id", done_conv_id}}, "");
            } else {
                std::string error;
                if (done_seen && !done_success) error = done_error.empty() ? "chat failed" : done_error;
                else if (!stream_ok) error = stream_status > 0 ? summarize_http_error({stream_status, stream_body}) : "stream connection failed";
                else error = "chat stream ended without done event";
                emit_result(false, "chat send", nlohmann::json::object(), error);
            }
            continue;
        }

        if (cmd0 == "curation") {
            if (tokens.size() < 3) {
                printer.line("usage: curation conv|mem|local|propose|apply ...");
                continue;
            }
            const std::string group = mm::util::to_lower(tokens[1]);
            const std::string sub = mm::util::to_lower(tokens[2]);

            // ── proposals ─────────────────────────────────────────────────────
            // The review half of curation: the model proposes edits, an operator
            // applies them. Both routes existed with no CLI form, so the
            // propose/review loop was TUI-only in practice.
            if (group == "propose") {
                emit_http_result("curation propose",
                                 self.post("/v1/agents/" + tokens[2] + "/curation/proposals",
                                           nlohmann::json::object()));
                continue;
            }
            if (group == "apply") {
                if (tokens.size() < 4) {
                    printer.line("usage: curation apply <agent_id> <json>");
                    continue;
                }
                try {
                    emit_http_result(
                        "curation apply",
                        self.post("/v1/agents/" + tokens[2] + "/curation/apply",
                                  nlohmann::json::parse(mm::cli::join_tokens(tokens, 3))));
                } catch (const std::exception& e) {
                    printer.line(std::string("error: invalid JSON: ") + e.what());
                }
                continue;
            }

            // ── conversation-local memories ───────────────────────────────────
            // Distinct from the agent-wide memories `curation mem` reaches:
            // these are scoped to one conversation, and all four verbs were
            // unreachable outside the TUI.
            if (group == "local") {
                if (tokens.size() < 5) {
                    printer.line("usage: curation local list|create|update|delete "
                                 "<agent_id> <conv_id> [memory_id] [json]");
                    continue;
                }
                const std::string base =
                    "/v1/agents/" + tokens[3] + "/conversations/" + tokens[4] + "/local-memories";
                if (sub == "list") {
                    emit_http_result("curation local list", self.get(base));
                    continue;
                }
                if (sub == "create") {
                    if (tokens.size() < 6) {
                        printer.line("usage: curation local create <agent_id> <conv_id> <json>");
                        continue;
                    }
                    try {
                        emit_http_result(
                            "curation local create",
                            self.post(base,
                                      nlohmann::json::parse(mm::cli::join_tokens(tokens, 5))));
                    } catch (const std::exception& e) {
                        printer.line(std::string("error: invalid JSON: ") + e.what());
                    }
                    continue;
                }
                if (sub == "update") {
                    if (tokens.size() < 7) {
                        printer.line("usage: curation local update <agent_id> <conv_id> "
                                     "<memory_id> <json>");
                        continue;
                    }
                    try {
                        emit_http_result(
                            "curation local update",
                            self.put(base + "/" + tokens[5],
                                     nlohmann::json::parse(mm::cli::join_tokens(tokens, 6))));
                    } catch (const std::exception& e) {
                        printer.line(std::string("error: invalid JSON: ") + e.what());
                    }
                    continue;
                }
                if (sub == "delete") {
                    if (tokens.size() < 6) {
                        printer.line("usage: curation local delete <agent_id> <conv_id> "
                                     "<memory_id>");
                        continue;
                    }
                    emit_http_result("curation local delete", self.del(base + "/" + tokens[5]));
                    continue;
                }
                printer.line("usage: curation local list|create|update|delete ...");
                continue;
            }

            if (group == "conv") {
                if (sub == "list") {
                    if (tokens.size() < 4) {
                        printer.line("usage: curation conv list <agent_id>");
                        continue;
                    }
                    auto r = self.get("/v1/agents/" + tokens[3] + "/conversations");
                    emit_http_result("curation conv list", r);
                    continue;
                }
                if (sub == "create") {
                    if (tokens.size() < 5) {
                        printer.line("usage: curation conv create <agent_id> <json>");
                        continue;
                    }
                    try {
                        auto j = nlohmann::json::parse(mm::cli::join_tokens(tokens, 4));
                        auto r = self.post("/v1/agents/" + tokens[3] + "/conversations", j);
                        emit_http_result("curation conv create", r);
                    } catch (const std::exception& e) {
                        printer.line(std::string("error: invalid JSON: ") + e.what());
                    }
                    continue;
                }
                if (sub == "activate" || sub == "delete" || sub == "compact") {
                    if (tokens.size() < 5) {
                        printer.line("usage: curation conv activate|delete|compact <agent_id> <conv_id>");
                        continue;
                    }
                    const std::string base = "/v1/agents/" + tokens[3] + "/conversations/" + tokens[4];
                    mm::HttpResponse r;
                    if (sub == "activate") r = self.post(base + "/activate", nlohmann::json::object());
                    else if (sub == "compact") r = self.post(base + "/compact", nlohmann::json::object());
                    else r = self.del(base);
                    emit_http_result("curation conv " + sub, r);
                    continue;
                }
            }

            if (group == "mem") {
                if (sub == "list") {
                    if (tokens.size() < 4) {
                        printer.line("usage: curation mem list <agent_id>");
                        continue;
                    }
                    auto r = self.get("/v1/agents/" + tokens[3] + "/memories");
                    emit_http_result("curation mem list", r);
                    continue;
                }
                if (sub == "delete") {
                    if (tokens.size() < 5) {
                        printer.line("usage: curation mem delete <agent_id> <memory_id>");
                        continue;
                    }
                    auto r = self.del("/v1/agents/" + tokens[3] + "/memories/" + tokens[4]);
                    emit_http_result("curation mem delete", r);
                    continue;
                }
                if (sub == "extract") {
                    if (tokens.size() < 5) {
                        printer.line("usage: curation mem extract <agent_id> <json>");
                        continue;
                    }
                    try {
                        auto j = nlohmann::json::parse(mm::cli::join_tokens(tokens, 4));
                        auto r = self.post("/v1/agents/" + tokens[3] + "/memories/extract", j);
                        emit_http_result("curation mem extract", r);
                    } catch (const std::exception& e) {
                        printer.line(std::string("error: invalid JSON: ") + e.what());
                    }
                    continue;
                }
            }

            printer.line("error: unknown curation command");
            continue;
        }

        // GET /v1/placements had no CLI reader at all — the one route that says
        // where every agent actually is.
        if (cmd0 == "placements") {
            emit_http_result("placements", self.get("/v1/placements"));
            continue;
        }

        // ── voice ─────────────────────────────────────────────────────────────
        //
        // Voice-design proposals: the model drafts a voice, an operator listens
        // and approves or rejects. Approve/reject are `operator` on the API and
        // had no CLI form, so a headless deployment could create proposals it
        // could never act on.
        //
        // `sample` renders audio to the server's cache and returns its id; the
        // audio itself is fetched by a route this REPL exempts, because a WAV is
        // not something a terminal can usefully render.
        if (cmd0 == "voice") {
            const std::string sub =
                tokens.size() < 2 ? std::string{} : mm::util::to_lower(tokens[1]);
            if (sub == "show" && tokens.size() > 2) {
                emit_http_result("voice show", self.get("/v1/agents/" + tokens[2] + "/voice"));
                continue;
            }
            if (sub == "proposals" && tokens.size() > 2) {
                emit_http_result("voice proposals",
                                 self.get("/v1/agents/" + tokens[2] + "/voice/proposals"));
                continue;
            }
            if (sub == "propose" && tokens.size() > 2) {
                emit_http_result("voice propose",
                                 self.post("/v1/agents/" + tokens[2] + "/voice/proposals",
                                           nlohmann::json::object()));
                continue;
            }
            if ((sub == "approve" || sub == "reject" || sub == "sample") && tokens.size() > 3) {
                emit_http_result("voice " + sub,
                                 self.post("/v1/agents/" + tokens[2] + "/voice/proposals/" +
                                               tokens[3] + "/" + sub,
                                           nlohmann::json::object()));
                continue;
            }
            printer.line("usage: voice show|proposals|propose <agent_id>  |  "
                         "voice approve|reject|sample <agent_id> <proposal_id>");
            continue;
        }

        // ── tokens ────────────────────────────────────────────────────────────
        //
        // The scoped-credential surface, and the sharpest of the coverage gaps:
        // with no CLI form, a headless deployment could not mint its FIRST
        // token. Bootstrapping required either the legacy flat
        // `external_api_token` from config or hand-rolled HTTP — so the scoped
        // auth system was, in practice, unreachable on exactly the deployments
        // it was designed for.
        //
        // The plaintext is returned ONCE at creation and never persisted (only
        // sha256 is stored), so `tokens create` prints the only copy that will
        // ever exist. Said in the output rather than assumed.
        if (cmd0 == "tokens") {
            const std::string sub =
                tokens.size() < 2 ? std::string{"list"} : mm::util::to_lower(tokens[1]);
            if (sub == "list") {
                emit_http_result("tokens list", self.get("/v1/tokens"));
                continue;
            }
            if (sub == "create") {
                if (tokens.size() < 3) {
                    printer.line("usage: tokens create <label> [read|chat|operator,...]");
                    continue;
                }
                nlohmann::json body{{"label", tokens[2]}};
                // Defaults to `read` rather than to everything: a credential
                // whose scope nobody chose should be the harmless one.
                body["scopes"] = tokens.size() > 3 ? tokens[3] : std::string{"read"};
                const auto r = self.post("/v1/tokens", body);
                if (r.ok()) printer.line("  the plaintext token is shown ONCE — store it now");
                emit_http_result("tokens create", r);
                continue;
            }
            if (sub == "delete") {
                if (tokens.size() < 3) {
                    printer.line("usage: tokens delete <token_id>");
                    continue;
                }
                emit_http_result("tokens delete", self.del("/v1/tokens/" + tokens[2]));
                continue;
            }
            printer.line("usage: tokens list|create|delete ...");
            continue;
        }

        // ── performance ───────────────────────────────────────────────────────
        if (cmd0 == "performance") {
            if (tokens.size() > 1 && mm::util::to_lower(tokens[1]) == "reset") {
                emit_http_result("performance reset", self.del("/v1/performance"));
                continue;
            }
            emit_http_result("performance", self.get("/v1/performance"));
            continue;
        }

        if (cmd0 == "activity") {
            if (tokens.size() < 2 || mm::util::to_lower(tokens[1]) != "tail") {
                printer.line("usage: activity tail [n] [level]");
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
            std::string path = "/v1/activity?tail=" + std::to_string(n);
            if (tokens.size() >= 4) path += "&level=" + tokens[3];
            auto r = self.get(path);
            if (!r.ok()) {
                emit_result(false, "activity tail", nlohmann::json::object(), summarize_http_error(r));
                continue;
            }
            if (json_mode) {
                emit_result(true, "activity tail", parse_or_wrap(r.body), "");
            } else {
                nlohmann::json j = parse_or_wrap(r.body);
                if (!j.is_object() || !j.contains("entries") || !j["entries"].is_array()) {
                    printer.block(pretty_body(r.body));
                    continue;
                }
                for (const auto& e : j["entries"]) {
                    const int level = e.value("level", 0);
                    const std::string tag = level == 2 ? "[error]" : level == 1 ? "[warn]" : "[info]";
                    printer.line(tag + " " + e.value("message", std::string{}));
                }
            }
            continue;
        }

        printer.line("error: unknown command. Type 'help'.");
    }
}

int main(int argc, char** argv) {
    const ControlMainArgs args = parse_control_main_args(argc, argv);
    if (!args.error.empty()) {
        std::cerr << "ERROR: " << args.error << "\n\n";
        print_control_usage();
        return 1;
    }
    if (args.show_help) {
        print_control_usage();
        return 0;
    }

    std::string cfg_path;
    auto cfg = load_config(&cfg_path);

    std::error_code data_ec;
    std::filesystem::path data_dir_abs =
        std::filesystem::absolute(cfg.data_dir, data_ec);
    const std::string lock_data_dir =
        data_ec ? cfg.data_dir : data_dir_abs.lexically_normal().string();
    auto instance_lock =
        ProcessSingletonLock::try_acquire(lock_data_dir, cfg.listen_port);
    if (!instance_lock) {
        std::fprintf(stderr,
                     "Another mantic-mind-control instance appears to be running for data_dir='%s' and port=%u.\n",
                     cfg.data_dir.c_str(),
                     static_cast<unsigned>(cfg.listen_port));
        return 1;
    }

    if (cfg.openai_compat_port != 0 && cfg.openai_compat_port == cfg.listen_port) {
        std::fprintf(stderr,
                     "openai_compat_port must differ from listen_port, or be 0 to disable it.\n");
        return 1;
    }

    // Ensure models directory exists.
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(cfg.models_dir, ec);
    }

    // Disable console logging — the TUI owns the terminal.
    mm::init_logger(
        cfg.log_file,
        "mm-control",
        spdlog::level::off,
        spdlog::level::trace);

    MM_INFO("mantic-mind-control starting on port {}", cfg.listen_port);
    if (cfg.openai_compat_port != 0) {
        MM_INFO("OpenAI-compatible API starting on port {}", cfg.openai_compat_port);
    }
    MM_INFO("Control config source: {}",
            cfg_path.empty() ? "(defaults/env only; no config file found)" : cfg_path);

    // ── Core services ─────────────────────────────────────────────────────────

    // ── control.db ────────────────────────────────────────────────────────────
    //
    // The first control-wide database in this system. Without an admission
    // record, select_backend() routes every agent to the fallback — absence of a
    // record is not evidence of admissibility — so this is what makes Soma
    // reachable at all.
    mm::ControlModelRegistry model_registry;
    {
        std::string registry_error;
        if (!model_registry.open(cfg.data_dir, registry_error)) {
            // Non-fatal. A control that cannot open its registry still serves
            // llama.cpp agents correctly; refusing to start would turn a routing
            // limitation into an outage.
            MM_WARN("Model registry unavailable ({}); every agent will route to the "
                    "fallback engine", registry_error);
        } else {
            mm::AdmissionTools tools;
            tools.python = cfg.admission_python;
            tools.tools_dir = cfg.admission_tools_dir;
            tools.soma_path = cfg.admission_soma_path;
            tools.containers_dir = cfg.containers_dir;
            tools.sources_dir = cfg.sources_dir;
            tools.allow_pickle = cfg.admission_allow_pickle;
            tools.quant = cfg.admission_quant;
            tools.expert_down = cfg.admission_expert_down;
            model_registry.set_tools(tools);
            model_registry.set_max_concurrent_admissions(
                static_cast<std::size_t>(std::max(1, cfg.admission_max_concurrent)));
            MM_INFO("Model registry: {} models, schema v{}",
                    model_registry.list().size(), model_registry.schema_version());
        }
    }

    mm::AgentManager agents(cfg.data_dir);
    agents.load_all();

    mm::NodeRegistry      registry(cfg.data_dir);
    mm::AgentScheduler    scheduler(registry, cfg.models_dir);
    registry.set_offline_after_seconds(static_cast<int>(cfg.node_offline_after_s));
    mm::AgentQueue        queue;
    // Both sides of the routing decision see the same registry: the scheduler
    // reads it to choose an engine, the API serves and edits it. Two lookups
    // against one table rather than a cached copy that can disagree with itself.
    scheduler.set_model_registry(&model_registry);

    // The audit trail. `placement_history` and its writer shipped with no caller
    // and no reader — a table created on every start for a history nothing
    // recorded (roadmap D60). Wired here rather than inside the scheduler
    // because the scheduler holds the registry as const on purpose: it reads
    // verdicts and must not be able to write model rows.
    scheduler.set_placement_audit({
        [&model_registry](const mm::AgentId& agent_id,
                          const mm::NodeId& node_id,
                          const mm::SlotId& slot_id,
                          const std::string& backend,
                          const std::string& backend_reason,
                          const mm::ResourceFootprint& footprint) {
            model_registry.record_placement(
                agent_id, node_id, slot_id, backend, backend_reason, footprint);
        },
        [&model_registry](const mm::AgentId& agent_id) {
            model_registry.mark_placement_released(agent_id);
        },
    });

    // ── the master's engine policy ────────────────────────────────────────────
    //
    // What the cluster runs, owned here and pushed to nodes. Until it exists,
    // placement refuses and first-run setup is forced — see the gate below.
    mm::EngineConfigStore engine_config(cfg.data_dir);
    {
        std::string load_error;
        if (!engine_config.load(load_error)) {
            // A present-but-unreadable configuration is fatal to CONFIGURATION,
            // not to the process: control still serves reads and the setup
            // surface. Starting with a silently empty policy would be worse —
            // it would re-run setup on a cluster that already had one and then
            // push a config the operator did not write.
            MM_ERROR("Engine configuration unusable: {}", load_error);
            std::fprintf(stderr, "\n  Engine configuration at %s is unusable:\n    %s\n"
                                 "  Fix or remove it, then restart.\n\n",
                         engine_config.path().c_str(), load_error.c_str());
            return 1;
        }
    }
    // The health poll converges nodes on this; the callback makes a change
    // propagate immediately instead of waiting up to one poll interval.
    registry.set_engine_config_provider(
        [&engine_config]() -> std::optional<mm::ClusterEngineConfig> {
            if (!engine_config.configured()) return std::nullopt;
            return engine_config.get();
        });
    engine_config.set_change_callback(
        [&registry](const mm::ClusterEngineConfig& c) { registry.push_engine_config_to_all(c); });
    scheduler.set_engine_config_gate([&engine_config]() { return engine_config.configured(); });
    scheduler.set_engine_config_provider(
        [&engine_config]() -> std::optional<mm::ClusterEngineConfig> {
            if (!engine_config.configured()) return std::nullopt;
            return engine_config.get();
        });

    mm::ControlApiServer  api_server(
        agents, queue, registry, scheduler,
        cfg.data_dir, cfg.models_dir, cfg.external_api_token, cfg.tts);
    api_server.set_model_registry(&model_registry);
    api_server.set_engine_config_store(&engine_config);
    api_server.cleanup_expired_tts_cache();
    mm::ControlUI         ui(
        registry,
        agents,
        scheduler,
        cfg.models_dir,
        "http://127.0.0.1:" + std::to_string(cfg.listen_port),
        cfg.external_api_token,
        [&api_server](const std::string& agent_id,
                      const std::string& message,
                      std::string* out_text,
                      std::string* out_conv_id,
                      std::string* out_error) -> bool {
            auto res = api_server.chat_local(agent_id, message);
            if (out_text) {
                std::string text;
                for (const auto& c : res.chunks) {
                    if (!c.delta_content.empty()) text += c.delta_content;
                }
                *out_text = std::move(text);
            }
            if (out_conv_id) *out_conv_id = res.conv_id;
            if (out_error) *out_error = res.error;
            return res.success;
        });

    api_server.set_log_callback([&](int level, const std::string& message) {
        auto ll = level == 2 ? mm::ControlUI::LogLevel::Error
                : level == 1 ? mm::ControlUI::LogLevel::Warn
                :              mm::ControlUI::LogLevel::Info;
        ui.log(ll, message);
    });

    registry.set_update_callback([&](const mm::NodeInfo& n) {
        if (n.id.empty()) {
            ui.refresh();
            return;
        }
        MM_INFO("Node {} -> {} ({})",
                n.id, mm::to_string(n.health), n.connected ? "up" : "down");
        const std::string msg =
            "Node " + n.id.substr(0, 8) + "... -> " +
            mm::to_string(n.health) +
            (n.connected ? " [up]" : " [down]");
        api_server.publish_activity(0, msg);
    });
    registry.start_health_poll(
        static_cast<int>(cfg.node_health_poll_interval_s));
    registry.start_discovery_listen(cfg.discovery_port);

    ui.set_pairing_key(cfg.pairing_key);

    // ── Housekeeping thread (every 5 min) ─────────────────────────────────────

    std::atomic<bool> stop_housekeeping{false};
    std::thread housekeeping_thread([&]() {
        while (!stop_housekeeping) {
            // Sleep 5 minutes in small increments for responsive shutdown.
            for (int i = 0; i < 300 && !stop_housekeeping; ++i)
                std::this_thread::sleep_for(std::chrono::seconds(1));

            if (!stop_housekeeping) {
                MM_INFO("Running scheduler housekeeping");
                scheduler.housekeeping(agents.list_agents());
                api_server.cleanup_expired_tts_cache();
            }
        }
    });

    // ── API server on background thread ───────────────────────────────────────

    std::thread server_thread([&] {
        MM_INFO("API server listening on 0.0.0.0:{}", cfg.listen_port);
        api_server.publish_activity(0, "API server listening on port " + std::to_string(cfg.listen_port));
        if (!api_server.listen(cfg.listen_port)) {
            MM_ERROR("Server failed on port {}", cfg.listen_port);
            api_server.publish_activity(2, "Server failed to start on port " + std::to_string(cfg.listen_port));
            ui.quit();
            if (g_control_cli_stop) g_control_cli_stop->store(true);
        }
    });

    std::thread openai_server_thread;
    if (cfg.openai_compat_port != 0) {
        openai_server_thread = std::thread([&] {
            MM_INFO("OpenAI-compatible API listening on 0.0.0.0:{}", cfg.openai_compat_port);
            api_server.publish_activity(
                0,
                "OpenAI-compatible API listening on port " +
                    std::to_string(cfg.openai_compat_port));
            if (!api_server.listen_openai_compat(cfg.openai_compat_port)) {
                MM_ERROR("OpenAI-compatible API failed on port {}", cfg.openai_compat_port);
                api_server.publish_activity(
                    2,
                    "OpenAI-compatible API failed to start on port " +
                        std::to_string(cfg.openai_compat_port));
                ui.quit();
                if (g_control_cli_stop) g_control_cli_stop->store(true);
            }
        });
    }

    // ── TUI on main thread (blocks until user quits) ──────────────────────────

    if (args.mode == ControlRunMode::Tui) {
        ui.run();
        MM_INFO("UI exited - shutting down");
    } else {
        std::atomic<bool> stop_cli{false};
        g_control_cli_stop = &stop_cli;
        auto old_int = std::signal(SIGINT, control_cli_signal_handler);
#ifdef SIGTERM
        auto old_term = std::signal(SIGTERM, control_cli_signal_handler);
#endif
        run_control_cli(cfg.listen_port, args.output, stop_cli);
        g_control_cli_stop = nullptr;
        std::signal(SIGINT, old_int);
#ifdef SIGTERM
        std::signal(SIGTERM, old_term);
#endif
        MM_INFO("CLI exited - shutting down");
    }

    // ── Graceful shutdown ─────────────────────────────────────────────────────

    stop_housekeeping = true;
    if (housekeeping_thread.joinable()) housekeeping_thread.join();

    api_server.stop();
    api_server.stop_openai_compat();
    registry.stop_discovery_listen();
    registry.stop_health_poll();
    queue.shutdown();
    if (server_thread.joinable()) server_thread.join();
    if (openai_server_thread.joinable()) openai_server_thread.join();
    MM_INFO("mantic-mind-control stopped");

    return 0;
}
