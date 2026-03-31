#include "common/config_file.hpp"
#include "common/cli_repl.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "control/control_config.hpp"
#include "control/agent_manager.hpp"
#include "control/node_registry.hpp"
#include "control/model_distributor.hpp"
#include "control/agent_scheduler.hpp"
#include "control/model_router.hpp"
#include "control/agent_queue.hpp"
#include "control/control_api_server.hpp"
#include "control/control_ui.hpp"

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
        cfg.data_dir    = file.get("data_dir",  cfg.data_dir);
        cfg.log_file    = file.get("log_file",   cfg.log_file);
        cfg.node_health_poll_interval_s = static_cast<uint32_t>(
            file.get_int("node_health_poll_interval_s",
                         static_cast<int>(cfg.node_health_poll_interval_s)));
        cfg.models_dir     = file.get("models_dir",     cfg.models_dir);
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

    cfg.listen_port = static_cast<uint16_t>(
        env_int("MM_CONTROL_PORT", static_cast<int>(cfg.listen_port)));
    cfg.data_dir    = env("MM_DATA_DIR",    cfg.data_dir);
    cfg.log_file    = env("MM_LOG_FILE",    cfg.log_file);
    cfg.models_dir  = env("MM_MODELS_DIR",  cfg.models_dir);
    cfg.pairing_key = env("MM_PAIRING_KEY", cfg.pairing_key);
    cfg.node_health_poll_interval_s = static_cast<uint32_t>(
        env_int("MM_POLL_INTERVAL_S",
                static_cast<int>(cfg.node_health_poll_interval_s)));
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
    std::cout
        << "Usage: mantic-mind-control [--mode tui|cli] [--output text|json] [--help]\n\n"
        << "Modes:\n"
        << "  tui  Default FTXUI terminal interface.\n"
        << "  cli  Interactive REPL suitable for terminal assistants.\n\n"
        << "Output:\n"
        << "  text Default human-readable CLI output.\n"
        << "  json Structured CLI output for automation.\n\n"
        << "CLI commands:\n"
        << "  nodes list\n"
        << "  nodes discovered\n"
        << "  nodes add <url> <api_key> [platform]\n"
        << "  nodes remove <node_id>\n"
        << "  nodes pair start <url>\n"
        << "  nodes pair complete <url> <nonce> <pin_or_psk>\n"
        << "  nodes pair psk <url> [psk]\n"
        << "  nodes llama check <node_id>\n"
        << "  nodes llama update <node_id> [build(true|false)] [force(true|false)]\n"
        << "  models list\n"
        << "  node-models list <node_id>\n"
        << "  node-models pull <node_id> <model_filename> [force(true|false)]\n"
        << "  node-models delete <node_id> <model_filename>\n"
        << "  agents list|show|create|update|delete ...\n"
        << "  chat send <agent_id> <message> [conversation_id]\n"
        << "  curation conv ...\n"
        << "  curation mem ...\n"
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
        printer.line("Top-level groups: nodes, models, node-models, agents, chat, curation, activity, help, quit");
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

        if (cmd0 == "nodes") {
            if (tokens.size() < 2) {
                printer.line("usage: nodes list|discovered|add|remove|pair|llama ...");
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
                    printer.line("usage: nodes add <url> <api_key> [platform]");
                    continue;
                }
                const std::string platform = tokens.size() >= 5 ? tokens[4] : "";
                emit_http_result(
                    "nodes add",
                    self.post("/v1/nodes",
                              nlohmann::json{{"url", tokens[2]}, {"api_key", tokens[3]}, {"platform", platform}}));
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
                        printer.line("usage: nodes pair complete <url> <nonce> <pin_or_psk>");
                        continue;
                    }
                    emit_http_result(
                        "nodes pair complete",
                        self.post("/v1/nodes/pair/complete",
                                  nlohmann::json{{"url", tokens[3]}, {"nonce", tokens[4]}, {"pin_or_psk", tokens[5]}}));
                    continue;
                }
                if (pair_sub == "psk") {
                    nlohmann::json body{{"url", tokens[3]}};
                    if (tokens.size() >= 5) body["psk"] = tokens[4];
                    emit_http_result("nodes pair psk", self.post("/v1/nodes/pair/psk", body));
                    continue;
                }
                printer.line("error: unknown nodes pair subcommand");
                continue;
            }
            if (sub == "llama") {
                if (tokens.size() < 4) {
                    printer.line("usage: nodes llama check|update <node_id> [build] [force]");
                    continue;
                }
                const std::string llama_sub = mm::util::to_lower(tokens[2]);
                if (llama_sub == "check") {
                    emit_http_result("nodes llama check",
                                     self.post("/v1/nodes/" + tokens[3] + "/llama/check-update", nlohmann::json::object()));
                    continue;
                }
                if (llama_sub == "update") {
                    bool build = true;
                    if (tokens.size() >= 5 && !mm::cli::parse_bool_token(tokens[4], &build)) {
                        printer.line("error: build must be true|false");
                        continue;
                    }
                    bool force = false;
                    if (tokens.size() >= 6 && !mm::cli::parse_bool_token(tokens[5], &force)) {
                        printer.line("error: force must be true|false");
                        continue;
                    }
                    emit_http_result(
                        "nodes llama update",
                        self.post("/v1/nodes/" + tokens[3] + "/llama/update",
                                  nlohmann::json{{"build", build}, {"force", force}}));
                    continue;
                }
                printer.line("error: unknown nodes llama subcommand");
                continue;
            }
            printer.line("error: unknown nodes subcommand");
            continue;
        }

        if (cmd0 == "models") {
            if (tokens.size() >= 2 && mm::util::to_lower(tokens[1]) != "list") {
                printer.line("usage: models [list]");
                continue;
            }
            emit_http_result("models list", self.get("/v1/models"));
            continue;
        }

        if (cmd0 == "node-models") {
            if (tokens.size() < 3) {
                printer.line("usage: node-models list|pull|delete|refresh ...");
                continue;
            }
            const std::string sub = mm::util::to_lower(tokens[1]);
            const std::string node_id = tokens[2];
            if (sub == "list" || sub == "refresh") {
                auto r = self.get("/v1/nodes/" + node_id + "/models");
                emit_http_result("node-models " + sub, r);
                continue;
            }
            if (sub == "pull") {
                if (tokens.size() < 4) {
                    printer.line("usage: node-models pull <node_id> <model_filename> [force]");
                    continue;
                }
                bool force = false;
                if (tokens.size() >= 5 && !mm::cli::parse_bool_token(tokens[4], &force)) {
                    printer.line("error: force must be true|false");
                    continue;
                }
                auto r = self.post("/v1/nodes/" + node_id + "/models/pull",
                                   nlohmann::json{{"model_filename", tokens[3]}, {"force", force}});
                emit_http_result("node-models pull", r);
                continue;
            }
            if (sub == "delete") {
                if (tokens.size() < 4) {
                    printer.line("usage: node-models delete <node_id> <model_filename>");
                    continue;
                }
                auto r = self.del("/v1/nodes/" + node_id + "/models/" + tokens[3]);
                emit_http_result("node-models delete", r);
                continue;
            }
            printer.line("error: unknown node-models subcommand");
            continue;
        }

        if (cmd0 == "agents") {
            if (tokens.size() < 2) {
                printer.line("usage: agents list|show|create|update|delete ...");
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
                printer.line("usage: curation conv|mem ...");
                continue;
            }
            const std::string group = mm::util::to_lower(tokens[1]);
            const std::string sub = mm::util::to_lower(tokens[2]);

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
    MM_INFO("Control config source: {}",
            cfg_path.empty() ? "(defaults/env only; no config file found)" : cfg_path);

    // ── Core services ─────────────────────────────────────────────────────────

    mm::AgentManager agents(cfg.data_dir);
    agents.load_all();

    mm::NodeRegistry      registry;
    mm::ModelDistributor  distributor(registry, cfg.models_dir);
    mm::AgentScheduler    scheduler(registry, distributor, cfg.models_dir);
    mm::ModelRouter       router(scheduler);
    mm::AgentQueue        queue;
    mm::ControlApiServer  api_server(agents, queue, registry, router, scheduler, cfg.models_dir);
    mm::ControlUI         ui(
        registry,
        agents,
        cfg.models_dir,
        "http://127.0.0.1:" + std::to_string(cfg.listen_port),
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
    registry.stop_discovery_listen();
    registry.stop_health_poll();
    queue.shutdown();
    if (server_thread.joinable()) server_thread.join();
    MM_INFO("mantic-mind-control stopped");

    return 0;
}
