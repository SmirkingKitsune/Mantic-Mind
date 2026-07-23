#include "common/config_file.hpp"
#include "common/cli_repl.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "control/control_config.hpp"
#include "control/agent_manager.hpp"
#include "control/node_registry.hpp"
#include "control/agent_scheduler.hpp"
#include "control/agent_queue.hpp"
#include "control/control_api_server.hpp"
#include "control/control_host.hpp"
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
        << "  nodes add <url> <api_key> [platform] [remember]\n"
        << "  nodes remove <node_id>\n"
        << "  nodes forget <node_id>\n"
        << "  nodes pair start <url>\n"
        << "  nodes pair complete <url> <nonce> <pin_or_psk> [remember]\n"
        << "  nodes pair psk <url> [psk] [remember]\n"
        << "  models list\n"
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
            if (tokens.size() >= 2 && mm::util::to_lower(tokens[1]) != "list") {
                printer.line("usage: models [list]");
                continue;
            }
            emit_http_result("models list", self.get("/v1/models"));
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

    mm::ControlHost::Options host_options;
    host_options.config = cfg;
    host_options.bind_host = "0.0.0.0";
    host_options.enable_remote_nodes = true;
    host_options.enable_discovery = true;
    host_options.allow_legacy_environment = true;
    mm::ControlHost host(std::move(host_options));
    std::string host_error;
    if (!host.acquire_singleton_lock(&host_error)) {
        std::fprintf(stderr, "ERROR: %s.\n", host_error.c_str());
        return 1;
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

    if (!host.initialize(&host_error)) {
        std::fprintf(stderr, "ERROR: %s.\n", host_error.c_str());
        return 1;
    }

    // ── Core services ─────────────────────────────────────────────────────────

    auto& agents = host.agents();
    auto& registry = host.registry();
    auto& scheduler = host.scheduler();
    auto& api_server = host.api();
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
        },
        [&host] { host.request_shutdown(); });

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
    ui.set_pairing_key(cfg.pairing_key);

    // ── Housekeeping thread (every 5 min) ─────────────────────────────────────

    // ── API server on background thread ───────────────────────────────────────

    host.set_failure_callback([&ui](const std::string&) {
        ui.quit();
        if (g_control_cli_stop) g_control_cli_stop->store(true);
    });
    if (!host.start(&host_error)) {
        std::fprintf(stderr, "ERROR: %s.\n", host_error.c_str());
        return 1;
    }
    MM_INFO("API server listening on 0.0.0.0:{}", cfg.listen_port);
    api_server.publish_activity(
        0, "API server listening on port " + std::to_string(cfg.listen_port));
    if (cfg.openai_compat_port != 0) {
        MM_INFO("OpenAI-compatible API listening on 0.0.0.0:{}",
                cfg.openai_compat_port);
        api_server.publish_activity(
            0, "OpenAI-compatible API listening on port " +
                   std::to_string(cfg.openai_compat_port));
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

    host.stop();
    MM_INFO("mantic-mind-control stopped");

    return 0;
}
