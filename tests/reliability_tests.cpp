#include "common/inference_response_parser.hpp"
#include "common/agent.hpp"
#include "common/agent_db.hpp"
#include "common/config_file.hpp"
#include "common/conversation_manager.hpp"
#include "common/http_client.hpp"
#include "common/runtime_client.hpp"
#include "common/memory_manager.hpp"
#include "common/trace_provenance.hpp"
#include "common/tool_executor.hpp"
#include "common/util.hpp"
#include "control/agent_manager.hpp"
#include "control/agent_queue.hpp"
#include "control/agent_scheduler.hpp"
#include "control/control_api_server.hpp"
#include "control/node_registry.hpp"
#include "control/performance_tracker.hpp"
#include "control/tts_service_client.hpp"
#include "common/gguf_metadata.hpp"
#include "common/inference_sizing.hpp"
#include "control/agent_config_validator.hpp"
#include "node/runtime_process.hpp"
#include "node/slot_manager.hpp"
#include "node/llama_runtime.hpp"
#include "node/llama_cpp_provisioner.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <httplib.h>
#include <SQLiteCpp/SQLiteCpp.h>

namespace {

class FixedSummaryRuntimeClient : public mm::RuntimeClient {
public:
    FixedSummaryRuntimeClient() : mm::RuntimeClient("http://127.0.0.1:1") {}

    mm::Message complete(const mm::InferenceRequest& req) override {
        last_model = req.model;

        mm::Message summary;
        summary.role = mm::MessageRole::Assistant;
        summary.content = "Summary: original source conversation kept the launch thermal review context.";
        summary.token_count = 12;
        summary.timestamp_ms = mm::util::now_ms();
        return summary;
    }

    std::string last_model;
};

const mm::TraceEvent* find_trace_event(const std::vector<mm::TraceEvent>& events,
                                       const std::string& title) {
    for (const auto& event : events) {
        if (event.title == title) return &event;
    }
    return nullptr;
}

bool check(bool condition, const char* expression, int line) {
    if (condition) return true;
    std::cerr << "CHECK failed at line " << line << ": " << expression << "\n";
    return false;
}

#define CHECK(expr) do { if (!check((expr), #expr, __LINE__)) return false; } while (0)

std::filesystem::path temp_test_dir(const std::string& name) {
    return std::filesystem::temp_directory_path()
        / ("mantic-mind-" + name + "-" + mm::util::generate_uuid());
}

// Windows reserves shifting blocks of high ports (Hyper-V/WSL excluded port
// ranges, see `netsh int ipv4 show excludedportrange`); a hardcoded test port
// can land inside one after a reboot and the server under test silently fails
// to bind. Probe for a port we can actually bind. Socket headers/WSA init come
// with <httplib.h>.
uint16_t find_free_test_port() {
    static uint16_t next_candidate = 42800;
    for (int p = next_candidate; p < 65000; ++p) {
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(static_cast<uint16_t>(p));
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
#ifdef _WIN32
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (s == INVALID_SOCKET) continue;
        const bool ok =
            bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
        closesocket(s);
#else
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) continue;
        int opt = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        const bool ok =
            ::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
        ::close(s);
#endif
        if (ok) {
            next_candidate = static_cast<uint16_t>(p + 1);
            return static_cast<uint16_t>(p);
        }
    }
    return 0;
}

bool remove_tree(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    if (ec) {
        std::cerr << "cleanup failed for " << dir << ": " << ec.message() << "\n";
        return false;
    }
    return true;
}

class ScopedCurrentPath {
public:
    explicit ScopedCurrentPath(const std::filesystem::path& path)
        : original_(std::filesystem::current_path()) {
        std::filesystem::current_path(path);
    }

    ScopedCurrentPath(const ScopedCurrentPath&) = delete;
    ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

    ~ScopedCurrentPath() {
        std::error_code ec;
        std::filesystem::current_path(original_, ec);
    }

private:
    std::filesystem::path original_;
};

bool wait_for_test_server(const std::string& url) {
    mm::HttpClient client(url);
    for (int i = 0; i < 80; ++i) {
        if (client.get("/api/node/health").ok()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    return false;
}

bool wait_for_registered_node(mm::NodeRegistry& registry,
                              const mm::NodeId& node_id) {
    for (int i = 0; i < 80; ++i) {
        try {
            if (registry.get_node(node_id).connected) return true;
        } catch (...) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    return false;
}

template <typename Predicate>
bool wait_for_node_snapshot(mm::NodeRegistry& registry,
                            const mm::NodeId& node_id,
                            int64_t not_before_ms,
                            Predicate&& predicate) {
    for (int i = 0; i < 160; ++i) {
        try {
            const auto node = registry.get_node(node_id);
            if (node.connected && node.last_seen_ms >= not_before_ms &&
                predicate(node)) {
                return true;
            }
        } catch (...) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    return false;
}

nlohmann::json test_curation_proposal(const std::string& id,
                                      const std::string& action,
                                      const std::string& target_type,
                                      const std::string& target_id,
                                      const std::string& conversation_id,
                                      nlohmann::json current,
                                      nlohmann::json proposed) {
    return nlohmann::json{
        {"id", id},
        {"action", action},
        {"target_type", target_type},
        {"target_id", target_id},
        {"conversation_id", conversation_id},
        {"current", std::move(current)},
        {"proposed", std::move(proposed)},
        {"rationale", "test proposal"},
        {"dedupe_key", action + ":" + target_type + ":" + target_id + ":" + conversation_id}
    };
}

bool test_non_stream_parser_preserves_text() {
    const std::string body =
        R"({"choices":[{"message":{"content":"abcdefghijklmnopqrstuvwxyz"}}],)"
        R"("usage":{"completion_tokens":7}})";
    auto parsed = mm::inference::parse_openai_chat_completion(body, 123);
    CHECK(parsed.has_value());
    CHECK(parsed->content == "abcdefghijklmnopqrstuvwxyz");
    CHECK(parsed->thinking_text.empty());
    CHECK(parsed->token_count == 7);
    return true;
}

bool test_non_stream_parser_extracts_thinking() {
    const std::string body =
        R"({"choices":[{"message":{"content":"before <think>hidden</think> after"}}]})";
    std::string error;
    auto parsed = mm::inference::parse_openai_chat_completion(body, 123, &error);
    if (!parsed) std::cerr << "parse error: " << error << "\n";
    CHECK(parsed.has_value());
    CHECK(parsed->content == "before  after");
    CHECK(parsed->thinking_text == "hidden");
    return true;
}

bool test_stream_tool_call_indices() {
    const std::string first =
        R"json({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"alpha","arguments":"{\"a\""}},{"index":1,"id":"call_b","function":{"name":"beta","arguments":"{\"b\""}}]}}]})json";
    const std::string second =
        R"json({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}},{"index":1,"function":{"arguments":":2}"}}]}}]})json";

    std::map<int, mm::ToolCall> calls;
    for (const auto& payload : {first, second}) {
        std::string error;
        auto parsed = mm::inference::parse_openai_sse_delta(payload, &error);
        if (!parsed) std::cerr << "parse error: " << error << "\n";
        CHECK(parsed.has_value());
        CHECK(error.empty());
        for (const auto& delta : parsed->tool_calls) {
            auto& call = calls[delta.index];
            if (!delta.call.id.empty()) call.id = delta.call.id;
            if (!delta.call.function_name.empty()) call.function_name = delta.call.function_name;
            call.arguments_json += delta.call.arguments_json;
        }
    }

    CHECK(calls.size() == 2);
    CHECK(calls[0].id == "call_a");
    CHECK(calls[0].function_name == "alpha");
    CHECK(calls[0].arguments_json == R"({"a":1})");
    CHECK(calls[1].id == "call_b");
    CHECK(calls[1].function_name == "beta");
    CHECK(calls[1].arguments_json == R"({"b":2})");
    return true;
}

bool test_agent_queue_survives_throwing_job() {
    mm::AgentQueue queue;
    std::mutex mutex;
    std::condition_variable cv;
    bool completed = false;
    bool failure_reported = false;

    mm::InferenceJob failing;
    failing.job_id = "throwing-job";
    failing.agent_id = "agent-a";
    failing.conversation_id = "conv-a";
    failing.done_cb = [&](const mm::ConvId& conv_id, bool success) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            failure_reported = (conv_id == "conv-a" && !success);
        }
        cv.notify_one();
    };
    failing.process_fn = [] {
        throw std::runtime_error("intentional test failure");
    };
    queue.enqueue(std::move(failing));

    mm::InferenceJob succeeding;
    succeeding.job_id = "succeeding-job";
    succeeding.agent_id = "agent-a";
    succeeding.process_fn = [&] {
        {
            std::lock_guard<std::mutex> lock(mutex);
            completed = true;
        }
        cv.notify_one();
    };
    queue.enqueue(std::move(succeeding));

    std::unique_lock<std::mutex> lock(mutex);
    CHECK(cv.wait_for(lock, std::chrono::seconds(3), [&] {
        return completed && failure_reported;
    }));
    lock.unlock();
    queue.shutdown();
    return true;
}

bool test_agent_manager_rejects_duplicates_and_defers_cleanup_until_handles_release() {
    auto dir = temp_test_dir("agents");
    std::filesystem::create_directories(dir);
    const auto agent_dir = dir / "agents" / "agent-a";

    mm::AgentManager manager(dir.string());
    mm::AgentConfig cfg;
    cfg.id = "agent-a";
    cfg.name = "Agent A";
    cfg.model_path = "model.gguf";

    CHECK(manager.create_agent(cfg) == "agent-a");
    auto held = manager.get_agent("agent-a");
    CHECK(static_cast<bool>(held));

    bool duplicate_rejected = false;
    try {
        manager.create_agent(cfg);
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    CHECK(duplicate_rejected);

    CHECK(manager.delete_agent("agent-a"));
    CHECK(held->get_id() == "agent-a");
    CHECK(std::filesystem::exists(agent_dir));

    held.reset();
    CHECK(!std::filesystem::exists(agent_dir));

    CHECK(remove_tree(dir));
    return true;
}

bool test_agent_api_settings_round_trip_without_key_persistence() {
    auto dir = temp_test_dir("agent-api-settings");
    std::filesystem::create_directories(dir);

    {
        mm::AgentManager manager(dir.string());
        mm::AgentConfig cfg;
        cfg.id = "api-agent";
        cfg.name = "API Agent";
        cfg.model_path = "frontier-test-model";
        cfg.inference_backend = "api";
        cfg.api_settings.base_url = "https://api.openai.com";
        cfg.api_settings.chat_completions_path = "/v1/chat/completions";
        cfg.api_settings.api_key = "secret-not-persisted";
        cfg.api_settings.api_key_env = "MANTIC_TEST_API_KEY";
        cfg.served_model_name = "api-agent-alias";
        cfg.runtime_settings.top_k = 20;
        cfg.runtime_settings.min_p = 0.0f;
        cfg.runtime_settings.presence_penalty = 1.5f;
        cfg.runtime_settings.repeat_penalty = 1.0f;

        CHECK(manager.create_agent(cfg) == "api-agent");
        auto agent = manager.get_agent("api-agent");
        CHECK(static_cast<bool>(agent));

        auto live = agent->get_config();
        CHECK(live.inference_backend == "api");
        CHECK(live.api_settings.api_key == "secret-not-persisted");

        nlohmann::json live_json = live;
        const std::string live_dump = live_json.dump();
        CHECK(live_dump.find("secret-not-persisted") == std::string::npos);
        CHECK(live_json["api_settings"]["api_key_configured"] == true);
        CHECK(live_json["api_settings"]["api_key_env"] == "MANTIC_TEST_API_KEY");

        mm::AgentDB persisted("api-agent", dir.string());
        auto loaded = persisted.load_config();
        CHECK(loaded.inference_backend == "api");
        CHECK(loaded.model_path == "frontier-test-model");
        CHECK(loaded.served_model_name == "api-agent-alias");
        CHECK(loaded.api_settings.base_url == "https://api.openai.com");
        CHECK(loaded.api_settings.chat_completions_path == "/v1/chat/completions");
        CHECK(loaded.api_settings.api_key.empty());
        CHECK(loaded.api_settings.api_key_env == "MANTIC_TEST_API_KEY");
        CHECK(loaded.runtime_settings.top_k == 20);
        CHECK(loaded.runtime_settings.min_p == 0.0f);
        CHECK(loaded.runtime_settings.presence_penalty == 1.5f);
        CHECK(loaded.runtime_settings.repeat_penalty == 1.0f);

        nlohmann::json loaded_json = loaded;
        CHECK(loaded_json.dump().find("secret-not-persisted") == std::string::npos);
        CHECK(loaded_json["api_settings"]["api_key_configured"] == false);
    }

    CHECK(remove_tree(dir));
    return true;
}

bool test_served_model_name_legacy_compatibility() {
    nlohmann::json legacy_json = {
        {"id", "legacy-json-agent"},
        {"name", "Legacy JSON Agent"},
        {"model_path", "legacy.gguf"},
        {"vllm_settings", {{"served_model_name", "legacy-json-alias"}}}
    };
    const auto from_legacy_json = legacy_json.get<mm::AgentConfig>();
    CHECK(from_legacy_json.served_model_name == "legacy-json-alias");

    auto dir = temp_test_dir("legacy-model-alias");
    std::filesystem::create_directories(dir);
    {
        mm::AgentDB db("legacy-db-agent", dir.string());
        mm::AgentConfig cfg;
        cfg.id = "legacy-db-agent";
        cfg.name = "Legacy DB Agent";
        cfg.model_path = "legacy.gguf";
        cfg.served_model_name = "modern-alias";
        db.save_config(cfg);
    }

    const auto db_path = dir / "agents" / "legacy-db-agent" / "agent.db";
    {
        SQLite::Database db(db_path.string(), SQLite::OPEN_READWRITE);
        db.exec(R"sql(
            UPDATE agent_config
               SET served_model_name = '',
                   vllm_settings_json = '{"served_model_name":"legacy-db-alias"}'
             WHERE id = 'legacy-db-agent'
        )sql");
    }
    {
        mm::AgentDB db("legacy-db-agent", dir.string());
        const auto loaded = db.load_config();
        CHECK(loaded.served_model_name == "legacy-db-alias");
    }

    CHECK(remove_tree(dir));
    return true;
}

bool test_slot_manager_not_found_statuses() {
    auto dir = temp_test_dir("slots");
    mm::SlotManager slots(46100, 46101, 1);

    auto unload = slots.unload_slot("missing-slot");
    CHECK(unload.status == mm::SlotOperationStatus::NotFound);

    auto suspend = slots.suspend_slot("missing-slot");
    CHECK(suspend.status == mm::SlotOperationStatus::NotFound);

    auto unload_all = slots.unload_all(false);
    CHECK(unload_all.status == mm::SlotOperationStatus::Ok);

    CHECK(remove_tree(dir));
    return true;
}

bool test_slot_lease_blocks_unload_and_suspend_while_busy() {
    auto dir = temp_test_dir("lease-busy");
    mm::SlotManager slots(46110, 46111, 1);
    const auto slot_id = slots.add_ready_test_slot("test-model.gguf", "agent-a");

    {
        auto inference_lease = slots.acquire_slot(slot_id);
        CHECK(static_cast<bool>(inference_lease));

        auto unload = slots.unload_slot(slot_id);
        CHECK(unload.status == mm::SlotOperationStatus::Busy);

        auto suspend = slots.suspend_slot(slot_id);
        CHECK(suspend.status == mm::SlotOperationStatus::Busy);
    }

    auto unload_after_release = slots.unload_slot(slot_id);
    CHECK(unload_after_release.status == mm::SlotOperationStatus::Ok);

    CHECK(remove_tree(dir));
    return true;
}

bool test_node_action_progress_json_round_trip() {
    mm::NodeActionProgress p;
    p.active = true;
    p.operation_id = "op-1";
    p.kind = "model_receive";
    p.action = "Downloading model";
    p.target = "Qwen/Qwen3-8B";
    p.stage = "receiving";
    p.detail = "model.safetensors";
    p.step = 2;
    p.total_steps = 4;
    p.bytes_done = 128;
    p.bytes_total = 256;
    p.fraction = 0.5;
    p.cancelable = true;
    p.cancel_requested = true;
    p.last_error = "canceled";

    nlohmann::json j = p;
    auto parsed = j.get<mm::NodeActionProgress>();
    CHECK(parsed.active);
    CHECK(parsed.operation_id == "op-1");
    CHECK(parsed.kind == "model_receive");
    CHECK(parsed.action == "Downloading model");
    CHECK(parsed.target == "Qwen/Qwen3-8B");
    CHECK(parsed.stage == "receiving");
    CHECK(parsed.detail == "model.safetensors");
    CHECK(parsed.step == 2);
    CHECK(parsed.total_steps == 4);
    CHECK(parsed.bytes_done == 128);
    CHECK(parsed.bytes_total == 256);
    CHECK(parsed.fraction > 0.49 && parsed.fraction < 0.51);
    CHECK(parsed.cancelable);
    CHECK(parsed.cancel_requested);
    CHECK(parsed.last_error == "canceled");

    mm::NodeInfo n;
    n.id = "node-a";
    n.url = "http://127.0.0.1:1";
    n.action_progress = p;
    auto node = nlohmann::json(n).get<mm::NodeInfo>();
    CHECK(node.action_progress.operation_id == "op-1");
    CHECK(node.action_progress.cancel_requested);
    return true;
}

bool test_scheduler_skips_failed_node_current_attempt() {
    bool ok = true;
#define RECORD(expr) do { if (!(expr)) { std::cerr << "CHECK failed at line " << __LINE__ << ": " << #expr << "\n"; ok = false; } } while (0)

    const uint16_t bad_port = find_free_test_port();
    const uint16_t good_port = find_free_test_port();
    RECORD(bad_port != 0);
    RECORD(good_port != 0);

    httplib::Server bad_server;
    httplib::Server good_server;
    std::atomic<int> bad_loads{0};
    std::atomic<int> good_loads{0};

    auto install_node_routes = [](httplib::Server& server,
                                  std::atomic<int>& load_calls,
                                  bool load_ok) {
        server.Get("/api/node/health", [](const httplib::Request&, httplib::Response& res) {
            mm::NodeHealthMetrics h;
            h.cpu_percent = 2.0f;
            h.ram_percent = 10.0f;
            h.gpu_percent = 0.0f;
            h.gpu_vram_total_mb = 24576;
            h.gpu_vram_used_mb = 0;
            h.gpu_backend_available = true;
            res.set_content(nlohmann::json(h).dump(), "application/json");
        });
        server.Get("/api/node/status", [](const httplib::Request&, httplib::Response& res) {
            nlohmann::json body = {
                {"loaded_model", ""},
                {"slots", nlohmann::json::array()},
                {"max_slots", 1},
                {"slot_in_use", 0},
                {"slot_available", 1},
                {"slot_ready", 0},
                {"slot_loading", 0},
                {"slot_suspending", 0},
                {"slot_suspended", 0},
                {"slot_error", 0}
            };
            res.set_content(body.dump(), "application/json");
        });
        server.Post("/api/node/load-model", [&load_calls, load_ok](
            const httplib::Request&, httplib::Response& res) {
            ++load_calls;
            if (!load_ok) {
                res.status = 503;
                res.set_content(nlohmann::json{
                    {"error", "llama.cpp runtime not ready"},
                    {"detail", "synthetic runtime failure"}
                }.dump(), "application/json");
                return;
            }
            res.set_content(nlohmann::json{
                {"status", "loaded"},
                {"slot_id", "slot-good"},
                {"effective_ctx_size", 4096}
            }.dump(), "application/json");
        });
    };

    install_node_routes(bad_server, bad_loads, false);
    install_node_routes(good_server, good_loads, true);

    std::atomic<bool> bad_listen_ok{false}, good_listen_ok{false};
    std::thread bad_thread([&] {
        bad_listen_ok = bad_server.listen("127.0.0.1", bad_port);
    });
    std::thread good_thread([&] {
        good_listen_ok = good_server.listen("127.0.0.1", good_port);
    });

    const std::string bad_url = "http://127.0.0.1:" + std::to_string(bad_port);
    const std::string good_url = "http://127.0.0.1:" + std::to_string(good_port);
    auto wait_for_server = [](const std::string& url) {
        mm::HttpClient client(url);
        for (int i = 0; i < 50; ++i) {
            if (client.get("/api/node/health").ok()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return false;
    };
    RECORD(wait_for_server(bad_url));
    RECORD(wait_for_server(good_url));

    auto dir = temp_test_dir("scheduler-failover");
    std::filesystem::create_directories(dir / "models");
    mm::NodeRegistry registry(dir.string());
    const auto bad_id = registry.add_node(bad_url, "bad-node-secret", "test", false);
    const auto good_id = registry.add_node(good_url, "good-node-secret", "test", false);
    registry.start_health_poll(1);
    for (int i = 0; i < 50; ++i) {
        const auto nodes = registry.list_nodes();
        const int connected = static_cast<int>(std::count_if(
            nodes.begin(), nodes.end(), [](const mm::NodeInfo& n) { return n.connected; }));
        if (connected == 2) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    mm::AgentScheduler scheduler(registry, (dir / "models").string());
    mm::AgentConfig cfg;
    cfg.id = "agent-a";
    cfg.name = "Agent A";
    cfg.model_path = "Qwen/Qwen3-8B";
    cfg.preferred_node_id = bad_id;

    auto retired = cfg;
    retired.id = "retired-agent";
    retired.inference_backend = "vllm";
    RECORD(!scheduler.ensure_agent_running(retired).has_value());
    RECORD(scheduler.last_error().find("supports llama-cpp only") !=
           std::string::npos);
    RECORD(bad_loads.load() == 0);
    RECORD(good_loads.load() == 0);

    auto scheduled = scheduler.ensure_agent_running(cfg);
    RECORD(scheduled.has_value());
    if (scheduled) {
        RECORD(scheduled->node_id == good_id);
        RECORD(scheduled->slot_id == "slot-good");
    }
    RECORD(bad_loads.load() >= 1);
    RECORD(good_loads.load() >= 1);

    registry.stop_health_poll();
    bad_server.stop();
    good_server.stop();
    if (bad_thread.joinable()) bad_thread.join();
    if (good_thread.joinable()) good_thread.join();
    RECORD(bad_listen_ok);
    RECORD(good_listen_ok);
    RECORD(remove_tree(dir));

#undef RECORD
    return ok;
}

bool test_scheduler_transfers_existing_relative_models_with_unique_cache_ids() {
    const auto dir = temp_test_dir("scheduler-local-identities");
    std::filesystem::create_directories(dir / "a");
    std::filesystem::create_directories(dir / "b");
    std::filesystem::create_directories(dir / "models");
    {
        std::ofstream(dir / "a" / "model.gguf", std::ios::binary) << "AAAA";
        std::ofstream(dir / "b" / "model.gguf", std::ios::binary) << "BBBB";
        std::ofstream(dir / "models" / "bare.gguf", std::ios::binary) << "CCCC";
    }
    const auto shared_timestamp = std::filesystem::file_time_type::clock::now()
                                - std::chrono::seconds(5);
    std::filesystem::last_write_time(dir / "a" / "model.gguf", shared_timestamp);
    std::filesystem::last_write_time(dir / "b" / "model.gguf", shared_timestamp);

    const uint16_t port = find_free_test_port();
    CHECK(port != 0);
    const std::string url = "http://127.0.0.1:" + std::to_string(port);

    httplib::Server server;
    std::mutex requests_mutex;
    std::vector<std::string> received_ids;
    std::map<std::string, nlohmann::json> load_bodies;
    server.Get("/api/node/health", [](const httplib::Request&,
                                      httplib::Response& res) {
        mm::NodeHealthMetrics health;
        health.gpu_vram_total_mb = 131072;
        health.gpu_backend_available = true;
        res.set_content(nlohmann::json(health).dump(), "application/json");
    });
    server.Get("/api/node/status", [](const httplib::Request&,
                                      httplib::Response& res) {
        res.set_content(nlohmann::json{
            {"slots", nlohmann::json::array()},
            {"max_slots", 8},
            {"slot_available", 8}
        }.dump(), "application/json");
    });
    server.Get("/api/node/models/local", [](const httplib::Request&,
                                             httplib::Response& res) {
        res.set_content(nlohmann::json{{"present", false}}.dump(),
                        "application/json");
    });
    server.Post("/api/node/models/receive", [&](const httplib::Request& req,
                                                 httplib::Response& res) {
        const std::string model_id = req.get_header_value("X-MM-Model-Id");
        const std::string relative = req.get_header_value("X-MM-Rel-Path");
        {
            std::lock_guard<std::mutex> lock(requests_mutex);
            received_ids.push_back(model_id);
        }
        const std::string stored =
            "/node/cache/" + model_id + "/" + relative;
        res.set_content(nlohmann::json{
            {"stored_path", stored}, {"load_path", stored}
        }.dump(), "application/json");
    });
    server.Post("/api/node/load-model", [&](const httplib::Request& req,
                                             httplib::Response& res) {
        const auto body = nlohmann::json::parse(req.body);
        const std::string agent_id = body.value("agent_id", std::string{});
        {
            std::lock_guard<std::mutex> lock(requests_mutex);
            load_bodies[agent_id] = body;
        }
        res.set_content(nlohmann::json{
            {"status", "loaded"}, {"slot_id", "slot-" + agent_id}
        }.dump(), "application/json");
    });

    std::atomic<bool> listen_ok{false};
    std::thread server_thread([&] {
        listen_ok = server.listen("127.0.0.1", port);
    });

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    RECORD(wait_for_test_server(url));
    mm::NodeRegistry registry((dir / "registry").string());
    const auto node_id = registry.add_node(url, "local-identity-secret", "test");
    registry.start_health_poll(1);
    RECORD(wait_for_registered_node(registry, node_id));

    mm::AgentScheduler scheduler(registry, (dir / "models").string());
    {
        ScopedCurrentPath current_path(dir);
        RECORD(mm::util::model_ref_is_local_path("a/model.gguf"));
        RECORD(mm::util::model_ref_is_local_path("b/model.gguf"));
        RECORD(mm::util::model_ref_is_local_path("bare.gguf"));

        auto schedule = [&](const std::string& agent_id,
                            const std::string& model_ref) {
            mm::AgentConfig cfg;
            cfg.id = agent_id;
            cfg.name = agent_id;
            cfg.model_path = model_ref;
            cfg.preferred_node_id = node_id;
            return scheduler.ensure_agent_running(cfg);
        };
        RECORD(schedule("relative-a", "a/model.gguf").has_value());
        RECORD(schedule("relative-b", "b/model.gguf").has_value());
        RECORD(schedule("bare", "bare.gguf").has_value());
    }

    std::vector<std::string> ids;
    std::map<std::string, nlohmann::json> bodies;
    {
        std::lock_guard<std::mutex> lock(requests_mutex);
        ids = received_ids;
        bodies = load_bodies;
    }
    RECORD(ids.size() == 3);
    if (ids.size() == 3) {
        RECORD(!ids[0].empty());
        RECORD(!ids[1].empty());
        RECORD(ids[0] != ids[1]);
        if (bodies.count("relative-a") == 1) {
            RECORD(bodies.at("relative-a").value("model_id", std::string{})
                   == ids[0]);
        }
        if (bodies.count("relative-b") == 1) {
            RECORD(bodies.at("relative-b").value("model_id", std::string{})
                   == ids[1]);
        }
        if (bodies.count("bare") == 1) {
            RECORD(bodies.at("bare").value("model_id", std::string{}) == ids[2]);
        }
    }
    for (const std::string agent_id : {"relative-a", "relative-b", "bare"}) {
        RECORD(bodies.count(agent_id) == 1);
        if (bodies.count(agent_id) == 1) {
            const auto& body = bodies.at(agent_id);
            RECORD(body.contains("model_id"));
            RECORD(body.value("model_path", std::string{}).rfind("/node/cache/", 0)
                   == 0);
        }
    }

    registry.stop_health_poll();
    server.stop();
    if (server_thread.joinable()) server_thread.join();
    RECORD(listen_ok);
    RECORD(remove_tree(dir));

#undef RECORD
    return ok;
}

bool test_scheduler_eviction_skips_unsuspendable_shared_slot() {
    const auto dir = temp_test_dir("scheduler-eviction-candidates");
    std::filesystem::create_directories(dir / "models");
    const uint16_t port = find_free_test_port();
    CHECK(port != 0);
    const std::string url = "http://127.0.0.1:" + std::to_string(port);

    httplib::Server server;
    std::atomic<bool> standalone_suspended{false};
    std::atomic<int> standalone_suspend_calls{0};
    std::atomic<int> shared_suspend_calls{0};
    std::atomic<int> unload_calls{0};
    std::atomic<int> new_load_calls{0};
    server.Get("/api/node/health", [](const httplib::Request&,
                                      httplib::Response& res) {
        mm::NodeHealthMetrics health;
        health.gpu_vram_total_mb = 131072;
        health.gpu_backend_available = true;
        res.set_content(nlohmann::json(health).dump(), "application/json");
    });
    server.Get("/api/node/status", [](const httplib::Request&,
                                      httplib::Response& res) {
        // Placements are intentionally the only eviction candidates. This
        // makes the test fail if the scheduler gives up on the first idle
        // placement instead of trying the next one.
        res.set_content(nlohmann::json{
            {"slots", nlohmann::json::array()},
            {"max_slots", 4},
            {"slot_available", 0}
        }.dump(), "application/json");
    });
    server.Post("/api/node/load-model", [&](const httplib::Request& req,
                                             httplib::Response& res) {
        const auto body = nlohmann::json::parse(req.body);
        const std::string agent_id = body.value("agent_id", std::string{});
        if (agent_id == "new-agent") {
            ++new_load_calls;
            if (!standalone_suspended.load()) {
                res.status = 503;
                res.set_content(
                    nlohmann::json{{"error", "max slots reached"}}.dump(),
                    "application/json");
                return;
            }
            res.set_content(nlohmann::json{
                {"status", "loaded"}, {"slot_id", "slot-new"}
            }.dump(), "application/json");
            return;
        }

        const std::string slot_id = agent_id == "standalone-idle"
            ? "slot-standalone"
            : "slot-shared";
        res.set_content(nlohmann::json{
            {"status", "loaded"}, {"slot_id", slot_id}
        }.dump(), "application/json");
    });
    server.Post("/api/node/suspend-slot", [&](const httplib::Request& req,
                                               httplib::Response& res) {
        const auto body = nlohmann::json::parse(req.body);
        const std::string slot_id = body.value("slot_id", std::string{});
        if (slot_id == "slot-standalone") {
            ++standalone_suspend_calls;
            standalone_suspended = true;
            res.set_content(nlohmann::json{
                {"status", "suspended"}, {"kv_cache_path", "standalone.kvbin"}
            }.dump(), "application/json");
            return;
        }
        ++shared_suspend_calls;
        res.status = 409;
        res.set_content(nlohmann::json{{"error", "shared slot is active"}}.dump(),
                        "application/json");
    });
    server.Post("/api/node/unload-model", [&](const httplib::Request&,
                                               httplib::Response& res) {
        ++unload_calls;
        res.status = 409;
        res.set_content(nlohmann::json{{"error", "unexpected unload"}}.dump(),
                        "application/json");
    });
    server.Post("/api/node/detach-agent", [](const httplib::Request&,
                                              httplib::Response& res) {
        res.set_content(nlohmann::json{{"status", "detached"}}.dump(),
                        "application/json");
    });

    std::atomic<bool> listen_ok{false};
    std::thread server_thread([&] {
        listen_ok = server.listen("127.0.0.1", port);
    });

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    RECORD(wait_for_test_server(url));
    mm::NodeRegistry registry((dir / "registry").string());
    const auto node_id = registry.add_node(url, "eviction-secret", "test");
    registry.start_health_poll(1);
    RECORD(wait_for_registered_node(registry, node_id));
    mm::AgentScheduler scheduler(registry, (dir / "models").string());

    auto config = [&](const std::string& id) {
        mm::AgentConfig cfg;
        cfg.id = id;
        cfg.name = id;
        cfg.model_path = "org/shared-model";
        cfg.preferred_node_id = node_id;
        return cfg;
    };

    RECORD(scheduler.ensure_agent_running(config("shared-idle")).has_value());
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    RECORD(scheduler.ensure_agent_running(config("shared-active")).has_value());
    scheduler.mark_agent_active("shared-active");
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    RECORD(scheduler.ensure_agent_running(config("standalone-idle")).has_value());
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    const auto scheduled = scheduler.ensure_agent_running(config("new-agent"));
    RECORD(scheduled.has_value());
    if (scheduled) RECORD(scheduled->slot_id == "slot-new");
    RECORD(new_load_calls.load() >= 2);
    RECORD(shared_suspend_calls.load() == 0);
    RECORD(standalone_suspend_calls.load() == 1);
    RECORD(unload_calls.load() == 0);

    const auto shared_idle = scheduler.get_placement("shared-idle");
    const auto shared_active = scheduler.get_placement("shared-active");
    const auto standalone = scheduler.get_placement("standalone-idle");
    RECORD(shared_idle.has_value() && !shared_idle->suspended);
    RECORD(shared_active.has_value() && !shared_active->suspended);
    RECORD(standalone.has_value() && standalone->suspended);

    registry.stop_health_poll();
    server.stop();
    if (server_thread.joinable()) server_thread.join();
    RECORD(listen_ok);
    RECORD(remove_tree(dir));

#undef RECORD
    return ok;
}

bool test_scheduler_backend_change_releases_local_placement() {
    const auto dir = temp_test_dir("scheduler-backend-change");
    std::filesystem::create_directories(dir / "models");
    const uint16_t port = find_free_test_port();
    CHECK(port != 0);
    const std::string url = "http://127.0.0.1:" + std::to_string(port);

    httplib::Server server;
    std::atomic<int> detach_calls{0};
    std::mutex detach_mutex;
    nlohmann::json detach_body;
    server.Get("/api/node/health", [](const httplib::Request&,
                                      httplib::Response& res) {
        mm::NodeHealthMetrics health;
        health.gpu_vram_total_mb = 131072;
        health.gpu_backend_available = true;
        res.set_content(nlohmann::json(health).dump(), "application/json");
    });
    server.Get("/api/node/status", [](const httplib::Request&,
                                      httplib::Response& res) {
        res.set_content(nlohmann::json{
            {"slots", nlohmann::json::array()},
            {"max_slots", 2},
            {"slot_available", 2}
        }.dump(), "application/json");
    });
    server.Post("/api/node/load-model", [](const httplib::Request&,
                                            httplib::Response& res) {
        res.set_content(nlohmann::json{
            {"status", "loaded"}, {"slot_id", "slot-local"}
        }.dump(), "application/json");
    });
    server.Post("/api/node/detach-agent", [&](const httplib::Request& req,
                                               httplib::Response& res) {
        ++detach_calls;
        {
            std::lock_guard<std::mutex> lock(detach_mutex);
            detach_body = nlohmann::json::parse(req.body);
        }
        res.set_content(nlohmann::json{{"status", "detached"}}.dump(),
                        "application/json");
    });

    std::atomic<bool> listen_ok{false};
    std::thread server_thread([&] {
        listen_ok = server.listen("127.0.0.1", port);
    });

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    RECORD(wait_for_test_server(url));
    mm::NodeRegistry registry((dir / "registry").string());
    const auto node_id = registry.add_node(url, "backend-change-secret", "test");
    registry.start_health_poll(1);
    RECORD(wait_for_registered_node(registry, node_id));
    mm::AgentScheduler scheduler(registry, (dir / "models").string());

    mm::AgentConfig cfg;
    cfg.id = "backend-agent";
    cfg.name = "Backend Agent";
    cfg.model_path = "org/model";
    cfg.preferred_node_id = node_id;
    RECORD(scheduler.ensure_agent_running(cfg).has_value());
    RECORD(scheduler.get_placement(cfg.id).has_value());

    auto api_cfg = cfg;
    api_cfg.inference_backend = "api";
    RECORD(!scheduler.ensure_agent_running(api_cfg).has_value());
    RECORD(!scheduler.get_placement(cfg.id).has_value());
    RECORD(detach_calls.load() == 1);
    {
        std::lock_guard<std::mutex> lock(detach_mutex);
        RECORD(detach_body.value("slot_id", std::string{}) == "slot-local");
        RECORD(detach_body.value("agent_id", std::string{}) == cfg.id);
    }
    RECORD(scheduler.last_error().find("supports llama-cpp only") !=
           std::string::npos);

    registry.stop_health_poll();
    server.stop();
    if (server_thread.joinable()) server_thread.join();
    RECORD(listen_ok);
    RECORD(remove_tree(dir));

#undef RECORD
    return ok;
}

bool test_scheduler_reconciles_ready_absent_and_suspended_snapshots() {
    const auto dir = temp_test_dir("scheduler-placement-reconciliation");
    std::filesystem::create_directories(dir / "models");
    const uint16_t port = find_free_test_port();
    CHECK(port != 0);
    const std::string url = "http://127.0.0.1:" + std::to_string(port);

    httplib::Server server;
    // 0: no reported slot, 1: attached Ready, 2: absent again,
    // 3: attached Suspended with a node-local KV path, 4: status unavailable.
    std::atomic<int> snapshot_phase{0};
    std::atomic<int> load_calls{0};
    std::atomic<int> detach_calls{0};
    std::atomic<int> restore_calls{0};
    std::mutex requests_mutex;
    std::vector<nlohmann::json> detach_bodies;
    nlohmann::json restore_body;

    server.Get("/api/node/health", [](const httplib::Request&,
                                      httplib::Response& res) {
        mm::NodeHealthMetrics health;
        health.gpu_vram_total_mb = 131072;
        health.gpu_backend_available = true;
        res.set_content(nlohmann::json(health).dump(), "application/json");
    });
    server.Get("/api/node/status", [&](const httplib::Request&,
                                       httplib::Response& res) {
        const int phase = snapshot_phase.load();
        if (phase == 4) {
            res.status = 503;
            return;
        }
        nlohmann::json slots = nlohmann::json::array();
        if (phase == 1 || phase == 3) {
            mm::SlotInfo slot;
            slot.id = phase == 1 ? "slot-initial" : "slot-rescheduled";
            slot.model_path = "org/model";
            slot.assigned_agent = "reconcile-agent";
            slot.agent_ids = {"reconcile-agent"};
            slot.state = phase == 1 ? mm::SlotState::Ready
                                    : mm::SlotState::Suspended;
            if (phase == 3) slot.kv_cache_path = "reported-cache.kvbin";
            slots.push_back(slot);
        }
        res.set_content(nlohmann::json{
            {"slots", std::move(slots)},
            {"max_slots", 2},
            {"slot_available", phase == 1 ? 1 : 2}
        }.dump(), "application/json");
    });
    server.Post("/api/node/load-model", [&](const httplib::Request&,
                                             httplib::Response& res) {
        const int call = ++load_calls;
        const std::string slot_id = call == 1 ? "slot-initial"
                                               : "slot-rescheduled";
        res.set_content(nlohmann::json{
            {"status", "loaded"}, {"slot_id", slot_id}
        }.dump(), "application/json");
    });
    server.Post("/api/node/detach-agent", [&](const httplib::Request& req,
                                               httplib::Response& res) {
        ++detach_calls;
        {
            std::lock_guard<std::mutex> lock(requests_mutex);
            detach_bodies.push_back(nlohmann::json::parse(req.body));
        }
        res.set_content(nlohmann::json{{"status", "detached"}}.dump(),
                        "application/json");
    });
    server.Post("/api/node/restore-slot", [&](const httplib::Request& req,
                                               httplib::Response& res) {
        ++restore_calls;
        {
            std::lock_guard<std::mutex> lock(requests_mutex);
            restore_body = nlohmann::json::parse(req.body);
        }
        res.set_content(nlohmann::json{
            {"status", "restored"}, {"slot_id", "slot-restored"}
        }.dump(), "application/json");
    });

    std::atomic<bool> listen_ok{false};
    std::thread server_thread([&] {
        listen_ok = server.listen("127.0.0.1", port);
    });

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    RECORD(wait_for_test_server(url));
    mm::NodeRegistry registry((dir / "registry").string());
    const auto node_id = registry.add_node(url, "reconcile-secret", "test");
    registry.start_health_poll(1);
    RECORD(wait_for_registered_node(registry, node_id));
    mm::AgentScheduler scheduler(registry, (dir / "models").string());

    mm::AgentConfig cfg;
    cfg.id = "reconcile-agent";
    cfg.name = "Reconcile Agent";
    cfg.model_path = "org/model";
    cfg.preferred_node_id = node_id;

    const auto initial = scheduler.ensure_agent_running(cfg);
    RECORD(initial.has_value());
    if (initial) RECORD(initial->slot_id == "slot-initial");
    RECORD(load_calls.load() == 1);
    const auto initial_placement = scheduler.get_placement(cfg.id);
    RECORD(initial_placement.has_value());

    // A newer health sample does not make an older slot snapshot authoritative.
    // If status polling fails, retain the placement created after that snapshot.
    const int64_t initial_slot_snapshot =
        registry.get_node(node_id).slot_snapshot_at_ms;
    snapshot_phase = 4;
    const int64_t initial_placed_at = initial_placement
        ? initial_placement->placed_at_ms : 0;
    RECORD(wait_for_node_snapshot(
        registry, node_id, initial_placed_at,
        [&](const mm::NodeInfo& node) {
            return node.last_seen_ms > initial_placed_at &&
                   node.slot_snapshot_at_ms == initial_slot_snapshot;
        }));
    const auto health_only = scheduler.ensure_agent_running(cfg);
    RECORD(health_only.has_value());
    if (health_only) RECORD(health_only->slot_id == "slot-initial");
    RECORD(load_calls.load() == 1);
    RECORD(detach_calls.load() == 0);

    // A newer connected snapshot confirms the same attachment. The scheduler
    // must return it without issuing another load request.
    snapshot_phase = 1;
    const int64_t health_only_seen = registry.get_node(node_id).last_seen_ms;
    RECORD(wait_for_node_snapshot(
        registry, node_id, health_only_seen + 1,
        [](const mm::NodeInfo& node) {
            return node.slots.size() == 1 &&
                   node.slots[0].id == "slot-initial" &&
                   node.slots[0].state == mm::SlotState::Ready;
        }));
    const auto ready_snapshot_seen = registry.get_node(node_id).last_seen_ms;
    const auto still_ready = scheduler.ensure_agent_running(cfg);
    RECORD(still_ready.has_value());
    if (still_ready) RECORD(still_ready->slot_id == "slot-initial");
    RECORD(load_calls.load() == 1);
    RECORD(detach_calls.load() == 0);
    RECORD(restore_calls.load() == 0);

    // A still-newer snapshot that no longer contains the attachment disproves
    // the cached placement. It must be detached and loaded again.
    snapshot_phase = 2;
    RECORD(wait_for_node_snapshot(
        registry, node_id, ready_snapshot_seen + 1,
        [](const mm::NodeInfo& node) { return node.slots.empty(); }));
    const auto absent_snapshot_seen = registry.get_node(node_id).last_seen_ms;
    const auto rescheduled = scheduler.ensure_agent_running(cfg);
    RECORD(rescheduled.has_value());
    if (rescheduled) RECORD(rescheduled->slot_id == "slot-rescheduled");
    RECORD(load_calls.load() == 2);
    RECORD(detach_calls.load() == 1);
    const auto rescheduled_placement = scheduler.get_placement(cfg.id);
    RECORD(rescheduled_placement.has_value());
    {
        std::lock_guard<std::mutex> lock(requests_mutex);
        RECORD(detach_bodies.size() == 1);
        if (detach_bodies.size() == 1) {
            RECORD(detach_bodies[0].value("slot_id", std::string{}) ==
                   "slot-initial");
            RECORD(detach_bodies[0].value("agent_id", std::string{}) == cfg.id);
        }
    }

    // If the node reports that the rescheduled attachment is Suspended, retain
    // it as a suspended placement and restore with the node-reported KV path.
    snapshot_phase = 3;
    const int64_t rescheduled_placed_at = rescheduled_placement
        ? rescheduled_placement->placed_at_ms : 0;
    RECORD(wait_for_node_snapshot(
        registry, node_id,
        std::max(absent_snapshot_seen + 1, rescheduled_placed_at),
        [](const mm::NodeInfo& node) {
            return node.slots.size() == 1 &&
                   node.slots[0].id == "slot-rescheduled" &&
                   node.slots[0].state == mm::SlotState::Suspended &&
                   node.slots[0].kv_cache_path == "reported-cache.kvbin";
        }));
    const auto restored = scheduler.ensure_agent_running(cfg);
    RECORD(restored.has_value());
    if (restored) RECORD(restored->slot_id == "slot-restored");
    RECORD(load_calls.load() == 2);
    RECORD(detach_calls.load() == 1);
    RECORD(restore_calls.load() == 1);
    {
        std::lock_guard<std::mutex> lock(requests_mutex);
        RECORD(restore_body.value("kv_cache_path", std::string{}) ==
               "reported-cache.kvbin");
        RECORD(restore_body.value("agent_id", std::string{}) == cfg.id);
    }
    const auto final_placement = scheduler.get_placement(cfg.id);
    RECORD(final_placement.has_value());
    if (final_placement) {
        RECORD(final_placement->slot_id == "slot-restored");
        RECORD(!final_placement->suspended);
        RECORD(final_placement->kv_cache_node_path.empty());
    }

    registry.stop_health_poll();
    server.stop();
    if (server_thread.joinable()) server_thread.join();
    RECORD(listen_ok);
    RECORD(remove_tree(dir));

#undef RECORD
    return ok;
}

bool test_control_api_external_token_gate() {
    auto dir = temp_test_dir("control-auth");
    std::filesystem::create_directories(dir);
    std::filesystem::create_directories(dir / "models");

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    {
        mm::AgentManager agents(dir.string());
        mm::AgentConfig cfg;
        cfg.id = "agent-a";
        cfg.name = "Agent A";
        cfg.model_path = "model.gguf";
        agents.create_agent(cfg);

        mm::AgentQueue queue;
        mm::NodeRegistry registry(dir.string());
        mm::AgentScheduler scheduler(registry, (dir / "models").string());
        mm::ControlApiServer api(
            agents, queue, registry, scheduler,
            dir.string(), (dir / "models").string(), "control-secret");

        const uint16_t port = find_free_test_port();
        CHECK(port != 0);
        const std::string base_url = "http://127.0.0.1:" + std::to_string(port);
        std::atomic<bool> listen_returned{false};
        std::atomic<bool> listen_ok{false};
        std::thread server_thread([&] {
            listen_ok = api.listen(port);
            listen_returned = true;
        });

        mm::HttpClient client(base_url);
        bool server_ready = false;
        for (int i = 0; i < 50; ++i) {
            auto resp = client.get("/v1/nodes");
            if (resp.status != 0) {
                server_ready = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        RECORD(server_ready);

        // mm::HttpClient opens a fresh connection per request; rapid
        // sequential connect/close cycles on Windows loopback occasionally
        // fail at the transport level (status == 0). Retry those.
        auto with_retry = [](auto&& request) {
            mm::HttpResponse resp;
            for (int attempt = 0; attempt < 8; ++attempt) {
                resp = request();
                if (resp.status != 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return resp;
        };

        auto expect_error = [&](const mm::HttpResponse& resp,
                                int expected_status,
                                const std::string& expected_text,
                                int call_line) {
            if (resp.status != expected_status ||
                resp.body.find(expected_text) == std::string::npos) {
                std::cerr << "expect_error (call at line " << call_line
                          << "): got status=" << resp.status
                          << " body=" << resp.body.substr(0, 200)
                          << " | expected status=" << expected_status
                          << " containing '" << expected_text << "'\n";
            }
            RECORD(resp.status == expected_status);
            RECORD(resp.body.find(expected_text) != std::string::npos);
        };
#define EXPECT_ERROR(resp, status, text) expect_error((resp), (status), (text), __LINE__)

        auto missing = with_retry([&] { return client.get("/v1/nodes"); });
        EXPECT_ERROR(missing, 401, "missing bearer token");

        client.set_bearer_token("wrong-secret");
        auto invalid = with_retry([&] { return client.get("/v1/nodes"); });
        EXPECT_ERROR(invalid, 403, "invalid bearer token");

        registry.add_node("http://127.0.0.1:1", "node-secret", "test", false);

        client.set_bearer_token("node-secret");
        auto node_token_on_external = with_retry([&] { return client.get("/v1/nodes"); });
        EXPECT_ERROR(node_token_on_external, 403, "invalid bearer token");

        client.set_bearer_token("control-secret");
        auto valid = with_retry([&] { return client.get("/v1/nodes"); });
        RECORD(valid.status == 200);
        auto valid_models = with_retry([&] { return client.get("/v1/models"); });
        RECORD(valid_models.status == 200);
        RECORD(valid_models.body.find("agent:agent-a") != std::string::npos);
        auto valid_voice = with_retry([&] { return client.get("/v1/agents/agent-a/voice"); });
        RECORD(valid_voice.status == 200);

        const auto png_path = dir / "attachment-test.png";
        {
            std::ofstream png(png_path, std::ios::binary);
            const unsigned char signature[] =
                {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
            png.write(reinterpret_cast<const char*>(signature), sizeof(signature));
        }

        mm::HttpClient missing_attachment_client(base_url);
        auto missing_upload = with_retry([&] {
            return missing_attachment_client.post_file(
                "/v1/agents/agent-a/attachments", png_path.string(),
                {{"X-Filename", "test.png"}}, "image/png");
        });
        EXPECT_ERROR(missing_upload, 401, "missing bearer token");

        auto mismatched_upload = with_retry([&] {
            return client.post_file(
                "/v1/agents/agent-a/attachments", png_path.string(),
                {{"X-Filename", "test.jpg"}}, "image/jpeg");
        });
        EXPECT_ERROR(mismatched_upload, 415, "signature does not match");

        auto upload = with_retry([&] {
            return client.post_file(
                "/v1/agents/agent-a/attachments", png_path.string(),
                {{"X-Filename", "../test.png"}}, "image/png");
        });
        RECORD(upload.status == 201);
        const auto uploaded_attachment = nlohmann::json::parse(upload.body);
        RECORD(!uploaded_attachment.contains("relative_path"));
        RECORD(uploaded_attachment["original_filename"] == "test.png");
        const std::string uploaded_attachment_id =
            uploaded_attachment["id"].get<std::string>();

        auto download = with_retry([&] {
            return client.get("/v1/agents/agent-a/attachments/" + uploaded_attachment_id);
        });
        RECORD(download.status == 200);
        RECORD(download.body.size() == 8);

        mm::HttpClient missing_voice_client(base_url);
        auto missing_voice = with_retry(
            [&] { return missing_voice_client.get("/v1/agents/agent-a/voice"); });
        EXPECT_ERROR(missing_voice, 401, "missing bearer token");

        struct StreamAttempt {
            bool ok = false;
            int status = 0;
            std::string body;
            std::vector<std::string> events;
        };
        auto stream_chat = [&](const std::string& token,
                               const nlohmann::json& body) {
            mm::HttpClient stream_client(base_url);
            if (!token.empty()) stream_client.set_bearer_token(token);
            StreamAttempt attempt;
            for (int retry = 0; retry < 3; ++retry) {
                attempt = StreamAttempt{};
                attempt.ok = stream_client.stream_post(
                    "/v1/agents/agent-a/chat",
                    body,
                    [&](const std::string& event) {
                        attempt.events.push_back(event);
                        return true;
                    },
                    &attempt.status,
                    &attempt.body);
                // status stays 0 only on transport failure; retry those.
                if (attempt.ok || attempt.status != 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return attempt;
        };

        auto missing_chat = stream_chat("", nlohmann::json{{"message", "hello"}});
        RECORD(!missing_chat.ok);
        RECORD(missing_chat.status == 401);
        RECORD(missing_chat.body.find("missing bearer token") != std::string::npos);
        RECORD(missing_chat.events.empty());

        auto invalid_chat = stream_chat("wrong-secret", nlohmann::json{{"message", "hello"}});
        RECORD(!invalid_chat.ok);
        RECORD(invalid_chat.status == 403);
        RECORD(invalid_chat.body.find("invalid bearer token") != std::string::npos);
        RECORD(invalid_chat.events.empty());

        auto node_chat = stream_chat("node-secret", nlohmann::json{{"message", "hello"}});
        RECORD(!node_chat.ok);
        RECORD(node_chat.status == 403);
        RECORD(node_chat.body.find("invalid bearer token") != std::string::npos);
        RECORD(node_chat.events.empty());

        auto valid_chat_route = stream_chat("control-secret", nlohmann::json{{"message", ""}});
        RECORD(!valid_chat_route.ok);
        RECORD(valid_chat_route.status == 400);
        RECORD(valid_chat_route.body.find("message required") != std::string::npos);

        auto disabled_vision_chat = stream_chat(
            "control-secret",
            nlohmann::json{{"message", "describe"},
                           {"attachment_ids", {uploaded_attachment_id}}});
        RECORD(!disabled_vision_chat.ok);
        RECORD(disabled_vision_chat.status == 422);
        RECORD(disabled_vision_chat.body.find("does not accept images") !=
               std::string::npos);

        auto delete_pending_attachment = with_retry([&] {
            return client.del("/v1/agents/agent-a/attachments/" +
                              uploaded_attachment_id);
        });
        RECORD(delete_pending_attachment.status == 204);

        mm::HttpClient missing_mutator(base_url);
        auto missing_create_conversation = with_retry([&] {
            return missing_mutator.post(
                "/v1/agents/agent-a/conversations",
                nlohmann::json{{"title", "Blocked"}, {"set_active", true}});
        });
        EXPECT_ERROR(missing_create_conversation, 401, "missing bearer token");

        mm::HttpClient node_mutator(base_url);
        node_mutator.set_bearer_token("node-secret");
        auto node_create_conversation = with_retry([&] {
            return node_mutator.post(
                "/v1/agents/agent-a/conversations",
                nlohmann::json{{"title", "Blocked"}, {"set_active", true}});
        });
        EXPECT_ERROR(node_create_conversation, 403, "invalid bearer token");

        auto create_conversation = with_retry([&] {
            return client.post(
                "/v1/agents/agent-a/conversations",
                nlohmann::json{{"title", "Original title"}, {"set_active", true}});
        });
        RECORD(create_conversation.status == 201);
        auto conversation_body = nlohmann::json::parse(create_conversation.body);
        const std::string conversation_id =
            conversation_body["conversation"]["id"].get<std::string>();

        auto referenced_upload = with_retry([&] {
            return client.post_file(
                "/v1/agents/agent-a/attachments", png_path.string(),
                {{"X-Filename", "referenced.png"}}, "image/png");
        });
        RECORD(referenced_upload.status == 201);
        const std::string referenced_attachment_id =
            nlohmann::json::parse(referenced_upload.body)["id"].get<std::string>();
        auto too_many_images = stream_chat(
            "control-secret",
            nlohmann::json{{"message", "too many"},
                           {"attachment_ids", std::vector<std::string>(
                               9, referenced_attachment_id)}});
        RECORD(!too_many_images.ok);
        RECORD(too_many_images.status == 400);
        RECORD(too_many_images.body.find("at most 8 images") != std::string::npos);
        auto agent_for_attachment = agents.get_agent("agent-a");
        RECORD(agent_for_attachment != nullptr);
        if (agent_for_attachment) {
            mm::Message image_message;
            image_message.role = mm::MessageRole::User;
            image_message.content = "persist this image";
            image_message.content_parts = {
                mm::MessageContentPart{"text", "persist this image", {}, {}, {}},
                mm::MessageContentPart{"image_attachment", {},
                                       referenced_attachment_id, {}, "image/png"}
            };
            agent_for_attachment->db().append_message(conversation_id, image_message, 0);
        }
        auto delete_referenced_attachment = with_retry([&] {
            return client.del("/v1/agents/agent-a/attachments/" +
                              referenced_attachment_id);
        });
        EXPECT_ERROR(delete_referenced_attachment, 409, "referenced by a message");

        mm::HttpClient missing_put(base_url);
        auto missing_rename = with_retry([&] {
            return missing_put.put(
                "/v1/agents/agent-a/conversations/" + conversation_id,
                nlohmann::json{{"title", "Blocked rename"}});
        });
        EXPECT_ERROR(missing_rename, 401, "missing bearer token");

        auto rename = with_retry([&] {
            return client.put(
                "/v1/agents/agent-a/conversations/" + conversation_id,
                nlohmann::json{{"title", "Renamed by authorized client"}});
        });
        RECORD(rename.status == 200);
        auto renamed_body = nlohmann::json::parse(rename.body);
        RECORD(renamed_body["title"] == "Renamed by authorized client");

        mm::HttpClient invalid_memory_client(base_url);
        invalid_memory_client.set_bearer_token("wrong-secret");
        auto invalid_create_memory = with_retry([&] {
            return invalid_memory_client.post(
                "/v1/agents/agent-a/memories",
                nlohmann::json{{"content", "Blocked memory"}, {"source_conv_id", conversation_id}});
        });
        EXPECT_ERROR(invalid_create_memory, 403, "invalid bearer token");

        auto create_memory = with_retry([&] {
            return client.post(
                "/v1/agents/agent-a/memories",
                nlohmann::json{{"content", "Authorized memory"}, {"source_conv_id", conversation_id}});
        });
        RECORD(create_memory.status == 201);
        auto memory_body = nlohmann::json::parse(create_memory.body);
        const std::string memory_id = memory_body["id"].get<std::string>();

        mm::HttpClient missing_delete(base_url);
        auto missing_delete_memory = with_retry(
            [&] { return missing_delete.del("/v1/agents/agent-a/memories/" + memory_id); });
        EXPECT_ERROR(missing_delete_memory, 401, "missing bearer token");

        auto delete_memory = with_retry(
            [&] { return client.del("/v1/agents/agent-a/memories/" + memory_id); });
        RECORD(delete_memory.status == 200);

        if (agent_for_attachment) {
            agent_for_attachment->db().delete_conversation(conversation_id);
        }
        auto attachment_after_conversation_delete = with_retry([&] {
            return client.get("/v1/agents/agent-a/attachments/" +
                              referenced_attachment_id);
        });
        EXPECT_ERROR(attachment_after_conversation_delete, 404, "attachment not found");

        api.stop();
        if (server_thread.joinable()) server_thread.join();
        RECORD(listen_ok);
        RECORD(listen_returned);
        queue.shutdown();
    }
    RECORD(remove_tree(dir));
#undef EXPECT_ERROR
#undef RECORD
    return ok;
}

bool test_openai_compat_api_listener_and_model_catalog() {
    auto dir = temp_test_dir("openai-compat");
    std::filesystem::create_directories(dir);
    std::filesystem::create_directories(dir / "models");

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    {
        mm::AgentManager agents(dir.string());
        mm::AgentConfig cfg;
        cfg.id = "agent-a";
        cfg.name = "agent-a-name";
        cfg.model_path = "model-a.gguf";
        cfg.served_model_name = "served-agent-a";
        agents.create_agent(cfg);

        mm::AgentQueue queue;
        mm::NodeRegistry registry(dir.string());
        mm::AgentScheduler scheduler(registry, (dir / "models").string());
        mm::ControlApiServer api(
            agents, queue, registry, scheduler,
            dir.string(), (dir / "models").string(), "control-secret");

        const uint16_t port = find_free_test_port();
        CHECK(port != 0);
        const std::string base_url = "http://127.0.0.1:" + std::to_string(port);
        std::atomic<bool> listen_returned{false};
        std::atomic<bool> listen_ok{false};
        std::thread server_thread([&] {
            listen_ok = api.listen_openai_compat(port);
            listen_returned = true;
        });

        mm::HttpClient client(base_url);
        auto with_retry = [](auto&& request) {
            mm::HttpResponse resp;
            for (int attempt = 0; attempt < 3; ++attempt) {
                resp = request();
                if (resp.status != 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return resp;
        };

        bool server_ready = false;
        for (int i = 0; i < 50; ++i) {
            auto resp = client.get("/v1/models");
            if (resp.status != 0) {
                server_ready = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        RECORD(server_ready);

        auto missing = with_retry([&] { return client.get("/v1/models"); });
        RECORD(missing.status == 401);
        RECORD(missing.body.find("missing bearer token") != std::string::npos);
        RECORD(missing.body.find("\"type\":\"authentication_error\"") != std::string::npos);

        client.set_bearer_token("control-secret");
        auto models = with_retry([&] { return client.get("/v1/models"); });
        RECORD(models.status == 200);
        auto models_body = nlohmann::json::parse(models.body);
        RECORD(models_body["object"] == "list");
        RECORD(models_body["data"].size() == 1);
        RECORD(models_body["data"][0]["id"] == "agent:agent-a");
        RECORD(models_body["data"][0]["metadata"]["vision_enabled"] == false);

        auto model = with_retry([&] { return client.get("/v1/models/served-agent-a"); });
        RECORD(model.status == 200);
        auto model_body = nlohmann::json::parse(model.body);
        RECORD(model_body["id"] == "agent:agent-a");
        RECORD(model_body["metadata"]["agent_id"] == "agent-a");

        auto missing_model = with_retry([&] {
            return client.post(
                "/v1/chat/completions",
                nlohmann::json{
                    {"model", "missing-model"},
                    {"messages", nlohmann::json::array({{
                        {"role", "user"},
                        {"content", "hello"}
                    }})}
                });
        });
        RECORD(missing_model.status == 404);
        RECORD(missing_model.body.find("use agent:{agent_id}") != std::string::npos);

        auto invalid_messages = with_retry([&] {
            return client.post(
                "/v1/chat/completions",
                nlohmann::json{
                    {"model", "agent:agent-a"},
                    {"messages", nlohmann::json::array()}
                });
        });
        RECORD(invalid_messages.status == 400);
        RECORD(invalid_messages.body.find("messages must not be empty") != std::string::npos);

        auto remote_image = with_retry([&] {
            return client.post(
                "/v1/chat/completions",
                nlohmann::json{
                    {"model", "agent:agent-a"},
                    {"messages", nlohmann::json::array({{
                        {"role", "user"},
                        {"content", nlohmann::json::array({
                            {{"type", "text"}, {"text", "describe"}},
                            {{"type", "image_url"},
                             {"image_url", {{"url", "https://example.test/image.png"}}}}
                        })}
                    }})}
                });
        });
        RECORD(remote_image.status == 400);
        RECORD(remote_image.body.find("not supported") != std::string::npos);

        auto non_user_image = with_retry([&] {
            return client.post(
                "/v1/chat/completions",
                nlohmann::json{
                    {"model", "agent:agent-a"},
                    {"messages", nlohmann::json::array({{
                        {"role", "assistant"},
                        {"content", nlohmann::json::array({
                            {{"type", "image_url"},
                             {"image_url", {{"url", "data:image/png;base64,iVBORw0KGgo="}}}}
                        })}
                    }})}
                });
        });
        RECORD(non_user_image.status == 400);
        RECORD(non_user_image.body.find("only on user messages") != std::string::npos);

        auto disabled_image = with_retry([&] {
            return client.post(
                "/v1/chat/completions",
                nlohmann::json{
                    {"model", "agent:agent-a"},
                    {"messages", nlohmann::json::array({{
                        {"role", "user"},
                        {"content", nlohmann::json::array({
                            {{"type", "text"}, {"text", "describe"}},
                            {{"type", "image_url"},
                             {"image_url", {{"url", "data:image/png;base64,iVBORw0KGgo="}}}}
                        })}
                    }})}
                });
        });
        RECORD(disabled_image.status == 422);
        RECORD(disabled_image.body.find("does not accept images") != std::string::npos);

        api.stop_openai_compat();
        if (server_thread.joinable()) server_thread.join();
        RECORD(listen_ok);
        RECORD(listen_returned);
        queue.shutdown();
    }
    RECORD(remove_tree(dir));
#undef RECORD
    return ok;
}

bool test_control_api_agent_api_mode_chat() {
    auto dir = temp_test_dir("control-api-agent-mode");
    std::filesystem::create_directories(dir);
    std::filesystem::create_directories(dir / "models");

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    const uint16_t backend_port = find_free_test_port();
    CHECK(backend_port != 0);
    const std::string backend_url = "http://127.0.0.1:" + std::to_string(backend_port);

    httplib::Server backend;
    std::mutex captured_mx;
    std::string captured_auth;
    nlohmann::json captured_body;
    int captured_requests = 0;

    backend.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"ok", true}}.dump(), "application/json");
    });
    backend.Post("/v1/chat/completions",
        [&](const httplib::Request& req, httplib::Response& res) {
            {
                std::lock_guard<std::mutex> lock(captured_mx);
                captured_auth = req.get_header_value("Authorization");
                captured_body = nlohmann::json::parse(req.body);
                ++captured_requests;
            }
            const std::string body =
                "data: {\"choices\":[{\"delta\":{\"content\":\"frontier \"}}]}\n\n"
                "data: {\"choices\":[{\"delta\":{\"content\":\"reply\"}}]}\n\n"
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],"
                "\"usage\":{\"completion_tokens\":2}}\n\n"
                "data: [DONE]\n\n";
            res.set_content(body, "text/event-stream");
        });

    std::atomic<bool> backend_listen_returned{false};
    std::atomic<bool> backend_listen_ok{false};
    std::thread backend_thread([&] {
        backend_listen_ok = backend.listen("127.0.0.1", backend_port);
        backend_listen_returned = true;
    });

    mm::HttpClient backend_client(backend_url);
    bool backend_ready = false;
    for (int i = 0; i < 50; ++i) {
        auto resp = backend_client.get("/health");
        if (resp.status == 200) {
            backend_ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    RECORD(backend_ready);

    {
        mm::AgentManager agents(dir.string());
        mm::AgentQueue queue;
        mm::NodeRegistry registry(dir.string());
        mm::AgentScheduler scheduler(registry, (dir / "models").string());
        mm::ControlApiServer api(
            agents, queue, registry, scheduler,
            dir.string(), (dir / "models").string(), "");

        const uint16_t port = find_free_test_port();
        CHECK(port != 0);
        const std::string base_url = "http://127.0.0.1:" + std::to_string(port);
        std::atomic<bool> listen_returned{false};
        std::atomic<bool> listen_ok{false};
        std::thread server_thread([&] {
            listen_ok = api.listen(port);
            listen_returned = true;
        });

        mm::HttpClient client(base_url);
        bool server_ready = false;
        for (int i = 0; i < 50; ++i) {
            auto resp = client.get("/v1/agents");
            if (resp.status != 0) {
                server_ready = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        RECORD(server_ready);

        auto with_retry = [](auto&& request) {
            mm::HttpResponse resp;
            for (int attempt = 0; attempt < 8; ++attempt) {
                resp = request();
                if (resp.status != 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return resp;
        };

        auto create = with_retry([&] {
            return client.post(
                "/v1/agents",
                nlohmann::json{
                    {"id", "api-agent"},
                    {"name", "API Agent"},
                    {"inference_backend", "api"},
                    {"model_path", "frontier-test-model"},
                    {"vision_settings", {{"enabled", true}, {"mmproj_path", ""}}},
                    {"memories_enabled", false},
                    {"tools_enabled", false},
                    {"runtime_settings", {
                        {"top_k", 20},
                        {"min_p", 0.0},
                        {"presence_penalty", 1.5},
                        {"repeat_penalty", 1.0}
                    }},
                    {"api_settings", {
                        {"base_url", backend_url},
                        {"chat_completions_path", "/v1/chat/completions"},
                        {"api_key", "test-secret"},
                        {"api_key_env", ""}
                    }}
                });
        });
        RECORD(create.status == 201);
        RECORD(create.body.find("test-secret") == std::string::npos);
        if (create.status == 201) {
            auto body = nlohmann::json::parse(create.body);
            RECORD(body["inference_backend"] == "api");
            RECORD(body["api_settings"]["api_key_configured"] == true);
            RECORD(body["node_compatibility"]["backend"] == "api");
            RECORD(body["node_compatibility"]["requires_node"] == false);
        }

        auto agent_resp = with_retry([&] { return client.get("/v1/agents/api-agent"); });
        RECORD(agent_resp.status == 200);
        RECORD(agent_resp.body.find("test-secret") == std::string::npos);
        if (agent_resp.status == 200) {
            auto body = nlohmann::json::parse(agent_resp.body);
            RECORD(body["status"] == "api");
            RECORD(body["inference_backend"] == "api");
            RECORD(body["node_compatibility"]["backend"] == "api");
            RECORD(body["node_compatibility"]["requires_node"] == false);
        }

        struct StreamAttempt {
            bool ok = false;
            int status = 0;
            std::string body;
            std::vector<std::string> events;
        };
        auto stream_chat = [&](const nlohmann::json& request_body) {
            StreamAttempt attempt;
            for (int retry = 0; retry < 3; ++retry) {
                attempt = StreamAttempt{};
                attempt.ok = client.stream_post(
                    "/v1/agents/api-agent/chat",
                    request_body,
                    [&](const std::string& event) {
                        attempt.events.push_back(event);
                        return true;
                    },
                    &attempt.status,
                    &attempt.body);
                if (attempt.ok || attempt.status != 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return attempt;
        };

        auto chat = stream_chat(nlohmann::json{{"message", "hello from test"}});
        RECORD(chat.ok);
        RECORD(chat.status == 200);

        std::string combined_delta;
        bool saw_done = false;
        for (const auto& event : chat.events) {
            if (event == "[DONE]") continue;
            auto j = nlohmann::json::parse(event);
            const std::string type = j.value("type", std::string{});
            if (type == "delta") {
                combined_delta += j.value("content", std::string{});
            } else if (type == "done") {
                saw_done = j.value("success", false);
            }
        }
        RECORD(combined_delta == "frontier reply");
        RECORD(saw_done);

        std::string auth;
        nlohmann::json sent;
        int request_count = 0;
        {
            std::lock_guard<std::mutex> lock(captured_mx);
            auth = captured_auth;
            sent = captured_body;
            request_count = captured_requests;
        }
        RECORD(request_count == 1);
        RECORD(auth == "Bearer test-secret");
        RECORD(sent["model"] == "frontier-test-model");
        RECORD(sent["stream"] == true);
        RECORD(sent["top_k"] == 20);
        RECORD(sent["min_p"] == 0.0);
        RECORD(sent["presence_penalty"] == 1.5);
        RECORD(sent["repeat_penalty"] == 1.0);
        RECORD(sent.dump().find("hello from test") != std::string::npos);

        const auto image_path = dir / "api-vision.png";
        {
            std::ofstream image(image_path, std::ios::binary);
            const unsigned char signature[] =
                {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
            image.write(reinterpret_cast<const char*>(signature), sizeof(signature));
        }
        const auto upload = with_retry([&] {
            return client.post_file(
                "/v1/agents/api-agent/attachments", image_path.string(),
                {{"X-Filename", "api-vision.png"}}, "image/png");
        });
        RECORD(upload.status == 201);
        const std::string attachment_id =
            nlohmann::json::parse(upload.body)["id"].get<std::string>();

        auto vision_chat = stream_chat(
            nlohmann::json{{"message", "describe this image"},
                           {"attachment_ids", {attachment_id}}});
        RECORD(vision_chat.ok);
        RECORD(vision_chat.status == 200);
        {
            std::lock_guard<std::mutex> lock(captured_mx);
            sent = captured_body;
            request_count = captured_requests;
        }
        RECORD(request_count == 2);
        const auto image_parts = sent["messages"].back()["content"];
        RECORD(image_parts.is_array());
        RECORD(image_parts.size() == 2);
        RECORD(image_parts[0]["type"] == "text");
        RECORD(image_parts[0]["text"] == "describe this image");
        RECORD(image_parts[1]["type"] == "image_url");
        RECORD(image_parts[1]["image_url"]["url"] ==
               "data:image/png;base64,iVBORw0KGgo=");

        auto followup = stream_chat(
            nlohmann::json{{"message", "what image did I send?"}});
        RECORD(followup.ok);
        RECORD(followup.status == 200);
        {
            std::lock_guard<std::mutex> lock(captured_mx);
            sent = captured_body;
            request_count = captured_requests;
        }
        RECORD(request_count == 3);
        bool retained_image = false;
        for (const auto& request_message : sent["messages"]) {
            if (!request_message.contains("content") ||
                !request_message["content"].is_array()) continue;
            for (const auto& part : request_message["content"]) {
                if (part.value("type", std::string{}) == "image_url" &&
                    part["image_url"].value("url", std::string{}) ==
                        "data:image/png;base64,iVBORw0KGgo=") {
                    retained_image = true;
                }
            }
        }
        RECORD(retained_image);

        api.stop();
        if (server_thread.joinable()) server_thread.join();
        RECORD(listen_ok);
        RECORD(listen_returned);
        queue.shutdown();
    }

    backend.stop();
    if (backend_thread.joinable()) backend_thread.join();
    RECORD(backend_listen_ok);
    RECORD(backend_listen_returned);
    RECORD(remove_tree(dir));
#undef RECORD
    return ok;
}

bool test_agent_voice_db_and_cache_lifecycle() {
    auto dir = temp_test_dir("agent-voice-db");
    std::filesystem::create_directories(dir);
    {
        mm::AgentDB db("agent-a", dir.string());

        mm::VoiceDesignProposal proposal;
        proposal.id = "proposal-a";
        proposal.agent_id = "agent-a";
        proposal.display_name = "Analyst Voice";
        proposal.language = "English";
        proposal.voice_description = "Clear, calm, original synthetic narrator voice.";
        proposal.sample_text = "Here is a concise operational update.";
        proposal.rationale = "Fits the agent role.";
        proposal.status = "pending";
        db.save_voice_proposal(proposal);

        auto loaded = db.get_voice_proposal("proposal-a");
        CHECK(loaded.has_value());
        CHECK(loaded->display_name == "Analyst Voice");
        CHECK(db.list_voice_proposals().size() == 1);
        db.update_voice_proposal_status("proposal-a", "sampled");
        loaded = db.get_voice_proposal("proposal-a");
        CHECK(loaded.has_value());
        CHECK(loaded->status == "sampled");

        mm::AgentVoiceProfile profile_a;
        profile_a.id = "profile-a";
        profile_a.agent_id = "agent-a";
        profile_a.display_name = "Voice A";
        profile_a.voice_description = "First voice.";
        profile_a.sample_text = "Sample A";
        profile_a.voice_clone_prompt_path = "prompt-a.pkl";
        profile_a.active = true;
        db.save_voice_profile(profile_a);
        auto active = db.get_active_voice_profile();
        CHECK(active.has_value());
        CHECK(active->id == "profile-a");

        mm::AgentVoiceProfile profile_b = profile_a;
        profile_b.id = "profile-b";
        profile_b.display_name = "Voice B";
        db.save_voice_profile(profile_b);
        active = db.get_active_voice_profile();
        CHECK(active.has_value());
        CHECK(active->id == "profile-b");
        auto old_profile = db.get_voice_profile("profile-a");
        CHECK(old_profile.has_value());
        CHECK(!old_profile->active);

        mm::TtsSynthesisResult cache;
        cache.cache_id = "cache-a";
        cache.agent_id = "agent-a";
        cache.voice_profile_id = "profile-b";
        cache.conversation_id = "conv-a";
        cache.message_index = 3;
        cache.text_hash = "hash-a";
        cache.audio_path = "speech-a.wav";
        cache.expires_at_ms = mm::util::now_ms() + 60000;
        db.save_tts_cache_entry(cache);
        auto found = db.find_tts_cache_entry("profile-b", "hash-a", "conv-a", 3);
        CHECK(found.has_value());
        CHECK(found->cached);
        CHECK(found->cache_id == "cache-a");

        mm::TtsSynthesisResult expired = cache;
        expired.cache_id = "cache-expired";
        expired.text_hash = "hash-expired";
        expired.expires_at_ms = mm::util::now_ms() - 1;
        db.save_tts_cache_entry(expired);
        auto removed = db.delete_expired_tts_cache_entries(mm::util::now_ms());
        CHECK(removed.size() == 1);
        CHECK(removed[0].cache_id == "cache-expired");
        CHECK(!db.get_tts_cache_entry("cache-expired").has_value());
    }

    CHECK(remove_tree(dir));
    return true;
}

bool test_tts_service_client_fake_sidecar_paths() {
    const uint16_t port = find_free_test_port();
    CHECK(port != 0);
    httplib::Server server;

    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"ok", true}}.dump(), "application/json");
    });

    server.Post("/voice-design", [](const httplib::Request& req, httplib::Response& res) {
        auto body = nlohmann::json::parse(req.body);
        res.set_content(nlohmann::json{
            {"ok", true},
            {"audio_path", body.value("output_audio_path", std::string{})},
            {"voice_clone_prompt_path", body.value("output_prompt_path", std::string{})},
            {"sample_rate", 24000},
            {"duration_ms", 500}
        }.dump(), "application/json");
    });

    server.Post("/synthesize", [](const httplib::Request& req, httplib::Response& res) {
        auto body = nlohmann::json::parse(req.body);
        if (body.value("text", std::string{}) == "fail") {
            res.status = 500;
            res.set_content(nlohmann::json{{"ok", false}, {"error", "synthetic failure"}}.dump(),
                            "application/json");
            return;
        }
        res.set_content(nlohmann::json{
            {"ok", true},
            {"audio_path", body.value("output_audio_path", std::string{})},
            {"sample_rate", 24000},
            {"duration_ms", 650}
        }.dump(), "application/json");
    });

    std::atomic<bool> listen_returned{false};
    std::atomic<bool> listen_ok{false};
    std::thread server_thread([&] {
        listen_ok = server.listen("127.0.0.1", port);
        listen_returned = true;
    });

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    mm::TtsServiceConfig config;
    config.enabled = true;
    config.service_url = "http://127.0.0.1:" + std::to_string(port);
    mm::TtsServiceClient client(config);

    bool ready = false;
    std::string health_error;
    for (int i = 0; i < 50; ++i) {
        if (client.health(&health_error)) {
            ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    RECORD(ready);

    mm::VoiceDesignProposal proposal;
    proposal.sample_text = "Preview text.";
    proposal.language = "English";
    proposal.voice_description = "Original clear voice.";
    auto sample = client.generate_voice_sample(proposal, "preview.wav", "prompt.pkl");
    RECORD(sample.ok);
    RECORD(sample.status == 200);
    RECORD(sample.audio_path == "preview.wav");
    RECORD(sample.voice_clone_prompt_path == "prompt.pkl");
    RECORD(sample.sample_rate == 24000);

    mm::AgentVoiceProfile profile;
    profile.id = "profile-a";
    profile.language = "English";
    profile.voice_clone_prompt_path = "prompt.pkl";

    mm::TtsSynthesisRequest request;
    request.text = "Speak.";
    request.format = "wav";
    auto speech = client.synthesize(request, profile, "speech.wav");
    RECORD(speech.ok);
    RECORD(speech.audio_path == "speech.wav");
    RECORD(speech.duration_ms == 650);

    request.text = "fail";
    auto failed = client.synthesize(request, profile, "failed.wav");
    RECORD(!failed.ok);
    RECORD(failed.status == 500);
    RECORD(failed.error.find("synthetic failure") != std::string::npos);

    server.stop();
    if (server_thread.joinable()) server_thread.join();
    RECORD(listen_ok);
    RECORD(listen_returned);
#undef RECORD
    return ok;
}

bool test_control_api_tts_routes_disabled() {
    auto dir = temp_test_dir("control-tts-disabled");
    std::filesystem::create_directories(dir);
    std::filesystem::create_directories(dir / "models");

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    {
        mm::AgentManager agents(dir.string());
        mm::AgentConfig cfg;
        cfg.id = "agent-a";
        cfg.name = "Agent A";
        cfg.model_path = "model.gguf";
        agents.create_agent(cfg);

        mm::AgentQueue queue;
        mm::NodeRegistry registry(dir.string());
        mm::AgentScheduler scheduler(registry, (dir / "models").string());
        mm::ControlApiServer api(
            agents, queue, registry, scheduler,
            dir.string(), (dir / "models").string());

        const uint16_t port = find_free_test_port();
        CHECK(port != 0);
        std::atomic<bool> listen_returned{false};
        std::atomic<bool> listen_ok{false};
        std::thread server_thread([&] {
            listen_ok = api.listen(port);
            listen_returned = true;
        });

        mm::HttpClient client("http://127.0.0.1:" + std::to_string(port));
        bool server_ready = false;
        for (int i = 0; i < 50; ++i) {
            auto resp = client.get("/v1/nodes");
            if (resp.status != 0) {
                server_ready = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        RECORD(server_ready);

    auto create = client.post(
        "/v1/agents/agent-a/voice/proposals",
        nlohmann::json{
            {"display_name", "Agent A Voice"},
            {"language", "English"},
            {"voice_description", "A clear original synthetic assistant voice."},
            {"sample_text", "This is a local voice preview."},
            {"rationale", "It matches the assistant role."}
        });
    RECORD(create.status == 201);
    auto create_body = nlohmann::json::parse(create.body);
    const std::string proposal_id = create_body["proposal"]["id"].get<std::string>();

    auto state = client.get("/v1/agents/agent-a/voice");
    RECORD(state.status == 200);
    auto state_body = nlohmann::json::parse(state.body);
    RECORD(state_body["tts_enabled"] == false);
    RECORD(state_body["proposals"].size() == 1);

    auto sample = client.post(
        "/v1/agents/agent-a/voice/proposals/" + proposal_id + "/sample",
        nlohmann::json::object());
    RECORD(sample.status == 503);
    RECORD(sample.body.find("disabled") != std::string::npos);

    auto speech = client.post(
        "/v1/agents/agent-a/speech",
        nlohmann::json{{"text", "Speak this message."}});
    RECORD(speech.status == 503);
    RECORD(speech.body.find("disabled") != std::string::npos);

    auto compat = client.post(
        "/v1/audio/speech",
        nlohmann::json{{"voice", "agent:agent-a"}, {"input", "Speak this message."}});
    RECORD(compat.status == 503);
    RECORD(compat.body.find("disabled") != std::string::npos);

        api.stop();
        if (server_thread.joinable()) server_thread.join();
        RECORD(listen_ok);
        RECORD(listen_returned);
        queue.shutdown();
    }
    RECORD(remove_tree(dir));
#undef RECORD
    return ok;
}

bool test_control_api_curation_routes() {
    auto dir = temp_test_dir("control-curation-routes");
    std::filesystem::create_directories(dir);
    std::filesystem::create_directories(dir / "models");

    mm::AgentManager agents(dir.string());
    mm::AgentConfig cfg;
    cfg.id = "agent-a";
    cfg.name = "Agent A";
    cfg.model_path = "model.gguf";
    agents.create_agent(cfg);

    mm::AgentQueue queue;
    mm::NodeRegistry registry(dir.string());
    mm::AgentScheduler scheduler(registry, (dir / "models").string());
    mm::ControlApiServer api(
        agents, queue, registry, scheduler,
        dir.string(), (dir / "models").string());

    const uint16_t port = find_free_test_port();
    CHECK(port != 0);
    std::atomic<bool> listen_returned{false};
    std::atomic<bool> listen_ok{false};
    std::thread server_thread([&] {
        listen_ok = api.listen(port);
        listen_returned = true;
    });

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    mm::HttpClient client("http://127.0.0.1:" + std::to_string(port));
    bool server_ready = false;
    for (int i = 0; i < 50; ++i) {
        auto resp = client.get("/v1/nodes");
        if (resp.status != 0) {
            server_ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    RECORD(server_ready);

    auto create_active = client.post(
        "/v1/agents/agent-a/conversations",
        nlohmann::json{{"title", "Original title"}, {"set_active", true}});
    RECORD(create_active.status == 201);
    auto active_body = nlohmann::json::parse(create_active.body);
    const std::string active_id = active_body["conversation"]["id"].get<std::string>();
    RECORD(active_body["conversation"]["is_active"].get<bool>());

    auto rename = client.put(
        "/v1/agents/agent-a/conversations/" + active_id,
        nlohmann::json{{"title", "Renamed curation title"}});
    RECORD(rename.status == 200);
    auto renamed_body = nlohmann::json::parse(rename.body);
    RECORD(renamed_body["title"] == "Renamed curation title");

    auto create_global = client.post(
        "/v1/agents/agent-a/memories",
        nlohmann::json{
            {"content", "  Durable global summary  "},
            {"source_conv_id", active_id},
            {"importance", 0.75}
        });
    RECORD(create_global.status == 201);
    auto global_body = nlohmann::json::parse(create_global.body);
    const std::string global_id = global_body["id"].get<std::string>();
    RECORD(global_body["content"] == "Durable global summary");
    RECORD(global_body["source_conv_id"] == active_id);

    auto create_global_delete = client.post(
        "/v1/agents/agent-a/memories",
        nlohmann::json{
            {"content", "Temporary global summary"},
            {"source_conv_id", active_id},
            {"importance", 0.25}
        });
    RECORD(create_global_delete.status == 201);
    auto global_delete_body = nlohmann::json::parse(create_global_delete.body);
    const std::string global_delete_id = global_delete_body["id"].get<std::string>();

    auto create_local = client.post(
        "/v1/agents/agent-a/conversations/" + active_id + "/local-memories",
        nlohmann::json{{"content", "Local note"}});
    RECORD(create_local.status == 201);
    auto local_body = nlohmann::json::parse(create_local.body);
    const std::string local_id = local_body["id"].get<std::string>();
    RECORD(local_body["conversation_id"] == active_id);

    auto update_local = client.put(
        "/v1/agents/agent-a/conversations/" + active_id + "/local-memories/" + local_id,
        nlohmann::json{{"content", "Updated local note"}});
    RECORD(update_local.status == 200);
    auto updated_local_body = nlohmann::json::parse(update_local.body);
    RECORD(updated_local_body["content"] == "Updated local note");

    auto create_local_delete = client.post(
        "/v1/agents/agent-a/conversations/" + active_id + "/local-memories",
        nlohmann::json{{"content", "Temporary local note"}});
    RECORD(create_local_delete.status == 201);
    auto local_delete_body = nlohmann::json::parse(create_local_delete.body);
    const std::string local_delete_id = local_delete_body["id"].get<std::string>();

    auto proposals = client.post(
        "/v1/agents/agent-a/curation/proposals",
        nlohmann::json{{"conversation_id", active_id}});
    RECORD(proposals.status == 200);
    auto proposals_body = nlohmann::json::parse(proposals.body);
    RECORD(proposals_body.contains("proposals"));

    auto create_batch_delete_conv = client.post(
        "/v1/agents/agent-a/conversations",
        nlohmann::json{{"title", "Batch delete conversation"}, {"set_active", false}});
    RECORD(create_batch_delete_conv.status == 201);
    auto batch_delete_conv_body = nlohmann::json::parse(create_batch_delete_conv.body);
    const std::string batch_delete_conv_id =
        batch_delete_conv_body["conversation"]["id"].get<std::string>();

    nlohmann::json apply_proposals = nlohmann::json::array({
        test_curation_proposal(
            "proposal-rename",
            "rename_conversation",
            "conversation",
            active_id,
            active_id,
            renamed_body,
            nlohmann::json{{"title", "Batch applied title"}}),
        test_curation_proposal(
            "proposal-update-local",
            "update_local_memory",
            "local_memory",
            local_id,
            active_id,
            updated_local_body,
            nlohmann::json{{"content", "  Batch updated local note  "}}),
        test_curation_proposal(
            "proposal-create-local",
            "create_local_memory",
            "local_memory",
            "",
            active_id,
            nlohmann::json::object(),
            nlohmann::json{{"content", "Batch created local note"}}),
        test_curation_proposal(
            "proposal-delete-local",
            "delete_local_memory",
            "local_memory",
            local_delete_id,
            active_id,
            local_delete_body,
            nlohmann::json::object()),
        test_curation_proposal(
            "proposal-update-global",
            "update_global_memory",
            "global_memory",
            global_id,
            active_id,
            global_body,
            nlohmann::json{
                {"content", "Batch updated global summary"},
                {"importance", 0.9}
            }),
        test_curation_proposal(
            "proposal-create-global",
            "create_global_memory",
            "global_memory",
            "",
            active_id,
            nlohmann::json::object(),
            nlohmann::json{
                {"content", "Batch created global summary"},
                {"importance", 0.4}
            }),
        test_curation_proposal(
            "proposal-delete-global",
            "delete_global_memory",
            "global_memory",
            global_delete_id,
            active_id,
            global_delete_body,
            nlohmann::json::object()),
        test_curation_proposal(
            "proposal-delete-conversation",
            "delete_conversation",
            "conversation",
            batch_delete_conv_id,
            batch_delete_conv_id,
            batch_delete_conv_body["conversation"],
            nlohmann::json::object())
    });

    auto apply = client.post(
        "/v1/agents/agent-a/curation/proposals/apply",
        nlohmann::json{{"proposals", apply_proposals}});
    RECORD(apply.status == 200);
    auto apply_body = nlohmann::json::parse(apply.body);
    RECORD(apply_body["status"] == "applied");
    RECORD(apply_body["applied_count"].get<int>() == 8);
    RECORD(apply_body["results"].is_array());
    RECORD(apply_body["results"].size() == 8);

    auto applied_conversation = client.get("/v1/agents/agent-a/conversations/" + active_id);
    RECORD(applied_conversation.status == 200);
    auto applied_conversation_body = nlohmann::json::parse(applied_conversation.body);
    RECORD(applied_conversation_body["title"] == "Batch applied title");

    auto applied_local = client.get(
        "/v1/agents/agent-a/conversations/" + active_id + "/local-memories");
    RECORD(applied_local.status == 200);
    auto applied_local_body = nlohmann::json::parse(applied_local.body);
    bool saw_updated_local = false;
    bool saw_created_local = false;
    bool saw_deleted_local = false;
    for (const auto& item : applied_local_body) {
        const std::string item_id = item["id"].get<std::string>();
        const std::string content = item["content"].get<std::string>();
        if (item_id == local_id && content == "Batch updated local note") {
            saw_updated_local = true;
        }
        if (content == "Batch created local note") {
            saw_created_local = true;
        }
        if (item_id == local_delete_id) {
            saw_deleted_local = true;
        }
    }
    RECORD(saw_updated_local);
    RECORD(saw_created_local);
    RECORD(!saw_deleted_local);

    auto applied_global = client.get("/v1/agents/agent-a/memories");
    RECORD(applied_global.status == 200);
    auto applied_global_body = nlohmann::json::parse(applied_global.body);
    bool saw_updated_global = false;
    bool saw_created_global = false;
    bool saw_deleted_global = false;
    for (const auto& item : applied_global_body) {
        const std::string item_id = item["id"].get<std::string>();
        const std::string content = item["content"].get<std::string>();
        if (item_id == global_id && content == "Batch updated global summary") {
            const double importance = item["importance"].get<double>();
            saw_updated_global = importance > 0.89 && importance < 0.91;
        }
        if (content == "Batch created global summary" &&
            item["source_conv_id"].get<std::string>() == active_id) {
            saw_created_global = true;
        }
        if (item_id == global_delete_id) {
            saw_deleted_global = true;
        }
    }
    RECORD(saw_updated_global);
    RECORD(saw_created_global);
    RECORD(!saw_deleted_global);

    auto deleted_batch_conv = client.get(
        "/v1/agents/agent-a/conversations/" + batch_delete_conv_id);
    RECORD(deleted_batch_conv.status == 404);

    auto create_invalid_target_conv = client.post(
        "/v1/agents/agent-a/conversations",
        nlohmann::json{{"title", "Invalid target container"}, {"set_active", false}});
    RECORD(create_invalid_target_conv.status == 201);
    auto invalid_target_conv_body = nlohmann::json::parse(create_invalid_target_conv.body);
    const std::string invalid_target_conv_id =
        invalid_target_conv_body["conversation"]["id"].get<std::string>();

    nlohmann::json invalid_apply_proposals = nlohmann::json::array({
        test_curation_proposal(
            "proposal-should-not-rename",
            "rename_conversation",
            "conversation",
            active_id,
            active_id,
            applied_conversation_body,
            nlohmann::json{{"title", "Should not apply"}}),
        test_curation_proposal(
            "proposal-invalid-local-target",
            "update_local_memory",
            "local_memory",
            local_id,
            invalid_target_conv_id,
            nlohmann::json::object(),
            nlohmann::json{{"content", "Should not apply"}})
    });
    auto invalid_apply = client.post(
        "/v1/agents/agent-a/curation/apply",
        nlohmann::json{{"proposals", invalid_apply_proposals}});
    RECORD(invalid_apply.status == 400);
    auto invalid_apply_body = nlohmann::json::parse(invalid_apply.body);
    RECORD(invalid_apply_body["error"] == "invalid curation proposal");
    RECORD(invalid_apply_body["index"].get<int>() == 1);
    RECORD(invalid_apply_body["proposal_id"] == "proposal-invalid-local-target");
    RECORD(invalid_apply_body["reason"].get<std::string>().find("does not belong") !=
           std::string::npos);

    auto conversation_after_invalid =
        client.get("/v1/agents/agent-a/conversations/" + active_id);
    RECORD(conversation_after_invalid.status == 200);
    auto conversation_after_invalid_body =
        nlohmann::json::parse(conversation_after_invalid.body);
    RECORD(conversation_after_invalid_body["title"] == "Batch applied title");

    auto local_after_invalid = client.get(
        "/v1/agents/agent-a/conversations/" + active_id + "/local-memories");
    RECORD(local_after_invalid.status == 200);
    auto local_after_invalid_body = nlohmann::json::parse(local_after_invalid.body);
    bool local_preserved_after_invalid = false;
    for (const auto& item : local_after_invalid_body) {
        if (item["id"].get<std::string>() == local_id &&
            item["content"].get<std::string>() == "Batch updated local note") {
            local_preserved_after_invalid = true;
        }
    }
    RECORD(local_preserved_after_invalid);

    auto delete_invalid_target_conv =
        client.del("/v1/agents/agent-a/conversations/" + invalid_target_conv_id);
    RECORD(delete_invalid_target_conv.status == 200);

    auto delete_active = client.del("/v1/agents/agent-a/conversations/" + active_id);
    RECORD(delete_active.status == 409);

    auto create_inactive = client.post(
        "/v1/agents/agent-a/conversations",
        nlohmann::json{{"title", "Inactive cleanup"}, {"set_active", false}});
    RECORD(create_inactive.status == 201);
    auto inactive_body = nlohmann::json::parse(create_inactive.body);
    const std::string inactive_id = inactive_body["conversation"]["id"].get<std::string>();

    auto delete_inactive = client.del("/v1/agents/agent-a/conversations/" + inactive_id);
    RECORD(delete_inactive.status == 200);

    auto delete_local = client.del(
        "/v1/agents/agent-a/conversations/" + active_id + "/local-memories/" + local_id);
    RECORD(delete_local.status == 200);

    api.stop();
    if (server_thread.joinable()) server_thread.join();
    RECORD(listen_ok);
    RECORD(listen_returned);
    queue.shutdown();
    RECORD(agents.delete_agent("agent-a"));
    RECORD(remove_tree(dir));
#undef RECORD
    return ok;
}

bool test_global_memory_origin_tool_and_context_metadata() {
    auto dir = temp_test_dir("global-memory-origin");
    std::filesystem::create_directories(dir);

    {
        mm::AgentDB db(dir.string());
        mm::ConvId conv_id = db.create_conversation("Origin chain");

        mm::Message user_msg;
        user_msg.role = mm::MessageRole::User;
        user_msg.content = "We decided the launch checklist needs a thermal review.";
        user_msg.timestamp_ms = mm::util::now_ms();
        db.append_message(conv_id, user_msg, 0);

        mm::LocalMemory local;
        local.id = "local-1";
        local.conversation_id = conv_id;
        local.content = "Thermal review is locked to this conversation.";
        db.add_local_memory(local);

        mm::Memory global;
        global.id = "global-1";
        global.content = "Origin chain covers launch checklist thermal-review decisions.";
        global.source_conv_id = conv_id;
        global.importance = 0.9f;
        global.created_at_ms = mm::util::now_ms();
        db.add_memory(global);

        const std::string formatted = mm::MemoryManager::format_memories_for_context({global});
        CHECK(formatted.find("global-1") != std::string::npos);
        CHECK(formatted.find(conv_id) != std::string::npos);
        CHECK(formatted.find("get_global_memory_origin") != std::string::npos);

        mm::ToolExecutor tools(db);
        mm::ToolCall call;
        call.id = "call-1";
        call.function_name = "get_global_memory_origin";
        call.arguments_json = R"({"memory_id":"global-1"})";

        mm::Message result = tools.execute_tool(call, conv_id);
        auto parsed = nlohmann::json::parse(result.content);
        CHECK(parsed["memory"]["id"] == "global-1");
        CHECK(parsed["memory"]["source_conv_id"] == conv_id);
        CHECK(parsed["origin"]["conversation_id"] == conv_id);
        CHECK(parsed["origin"]["local_memories"].size() == 1);
        CHECK(parsed["origin"]["messages"].size() == 1);
        CHECK(parsed["origin"]["messages"][0]["content_preview"]
                  .get<std::string>()
                  .find("thermal review") != std::string::npos);
    }

    CHECK(remove_tree(dir));
    return true;
}

bool test_message_trace_events_round_trip() {
    auto dir = temp_test_dir("trace-events");
    std::filesystem::create_directories(dir);

    {
        mm::AgentDB db("agent-a", dir.string());
        mm::ConvId conv_id = db.create_conversation("Trace events");

        mm::TraceEvent event;
        event.id = "trace-1";
        event.type = "global-memory";
        event.category = "global-memory";
        event.title = "Global memory reviewed";
        event.detail = "Project uses SvelteKit.";
        event.source_id = "global-1";
        event.timestamp_ms = mm::util::now_ms();
        event.sequence = 0;
        event.metadata = {{"importance", 0.8}};

        mm::Message msg;
        msg.role = mm::MessageRole::Assistant;
        msg.content = "I inspected the existing UI.";
        msg.timestamp_ms = mm::util::now_ms();
        msg.trace_events.push_back(event);
        db.append_message(conv_id, msg, 0);

        auto loaded = db.load_conversation(conv_id);
        CHECK(loaded.has_value());
        CHECK(loaded->messages.size() == 1);
        CHECK(loaded->messages[0].trace_events.size() == 1);
        CHECK(loaded->messages[0].trace_events[0].title == "Global memory reviewed");
        CHECK(loaded->messages[0].trace_events[0].metadata["importance"] == 0.8);

        nlohmann::json serialized = loaded->messages[0];
        CHECK(serialized["trace_events"].size() == 1);
        CHECK(serialized["trace_events"][0]["source_id"] == "global-1");
    }

    CHECK(remove_tree(dir));
    return true;
}

bool test_compaction_followup_trace_provenance_survives() {
    auto dir = temp_test_dir("compaction-trace-provenance");
    std::filesystem::create_directories(dir);

    {
        mm::AgentDB db("agent-a", dir.string());
        mm::AgentConfig cfg;
        cfg.id = "agent-a";
        cfg.name = "Agent A";
        cfg.model_path = "model.gguf";
        cfg.system_prompt = "Use memory provenance carefully.";
        cfg.memories_enabled = true;
        cfg.runtime_settings.ctx_size = 128;

        const mm::ConvId source_conv_id = db.create_conversation("Launch review");
        db.set_active_conversation(source_conv_id);

        for (int i = 0; i < 6; ++i) {
            mm::Message msg;
            msg.role = (i % 2 == 0) ? mm::MessageRole::User : mm::MessageRole::Assistant;
            msg.content = i == 0
                ? "The source conversation decided the launch checklist needs a thermal review."
                : "Conversation turn " + std::to_string(i);
            msg.token_count = 18;
            msg.timestamp_ms = mm::util::now_ms();
            db.append_message(source_conv_id, msg, i);
        }

        mm::LocalMemory local;
        local.id = "local-launch-review";
        local.conversation_id = source_conv_id;
        local.content = "Thermal review remains the local follow-up detail.";
        db.add_local_memory(local);

        mm::Memory global;
        global.id = "global-launch-review";
        global.content = "Launch review origin covers the thermal checklist decision.";
        global.source_conv_id = source_conv_id;
        global.importance = 0.95f;
        global.created_at_ms = mm::util::now_ms();
        db.add_memory(global);

        FixedSummaryRuntimeClient runtime;
        mm::ConversationManager conv_mgr(db, runtime);
        const mm::ConvId continued_conv_id = conv_mgr.force_compact(source_conv_id, cfg);
        CHECK(!continued_conv_id.empty());
        CHECK(continued_conv_id != source_conv_id);
        CHECK(runtime.last_model == "model.gguf");

        auto source = db.load_conversation(source_conv_id);
        auto continued = db.load_conversation(continued_conv_id);
        CHECK(source.has_value());
        CHECK(continued.has_value());
        CHECK(!source->is_active);
        CHECK(continued->is_active);
        CHECK(continued->parent_conv_id == source_conv_id);
        CHECK(continued->compaction_summary.find("thermal review") != std::string::npos);

        CHECK(db.list_local_memories(source_conv_id).empty());
        auto continued_local = db.list_local_memories(continued_conv_id);
        CHECK(continued_local.size() == 1);
        CHECK(continued_local[0].id == "local-launch-review");
        CHECK(continued_local[0].conversation_id == continued_conv_id);

        std::vector<mm::TraceEvent> trace_events =
            mm::build_context_trace_events(db, continued_conv_id, db.list_memories());
        CHECK(trace_events.size() == 4);

        const mm::TraceEvent* parent_trace =
            find_trace_event(trace_events, "Parent conversation accessed");
        CHECK(parent_trace != nullptr);
        CHECK(parent_trace->source_id == source_conv_id);
        CHECK(parent_trace->metadata["conversation_id"] == continued_conv_id);

        const mm::TraceEvent* summary_trace =
            find_trace_event(trace_events, "Compaction summary reviewed");
        CHECK(summary_trace != nullptr);
        CHECK(summary_trace->source_id == source_conv_id);
        CHECK(summary_trace->metadata["conversation_id"] == continued_conv_id);
        CHECK(summary_trace->metadata["parent_conv_id"] == source_conv_id);

        const mm::TraceEvent* local_trace =
            find_trace_event(trace_events, "Conversation-local memory reviewed");
        CHECK(local_trace != nullptr);
        CHECK(local_trace->source_id == "local-launch-review");
        CHECK(local_trace->metadata["conversation_id"] == continued_conv_id);

        const mm::TraceEvent* global_trace =
            find_trace_event(trace_events, "Global memory reviewed");
        CHECK(global_trace != nullptr);
        CHECK(global_trace->source_id == "global-launch-review");
        CHECK(global_trace->metadata["source_conv_id"] == source_conv_id);

        mm::Message followup;
        followup.role = mm::MessageRole::Assistant;
        followup.content = "The thermal review context is still available.";
        followup.timestamp_ms = mm::util::now_ms();
        followup.trace_events = trace_events;
        db.append_message(continued_conv_id, followup, 4);

        mm::ToolExecutor tools(db);
        mm::ToolCall local_call;
        local_call.id = "call-local";
        local_call.function_name = "list_local_memories";
        local_call.arguments_json = "{}";
        mm::Message local_result = tools.execute_tool(local_call, continued_conv_id);
        auto local_result_json = nlohmann::json::parse(local_result.content);
        CHECK(local_result_json["count"] == 1);
        CHECK(local_result_json["memories"][0]["id"] == "local-launch-review");
        auto local_tool_trace = mm::build_tool_access_trace(
            local_call, local_result, continued_conv_id);
        CHECK(local_tool_trace.has_value());
        local_result.trace_events.push_back(*local_tool_trace);
        db.append_message(continued_conv_id, local_result, 5);

        mm::ToolCall origin_call;
        origin_call.id = "call-origin";
        origin_call.function_name = "get_global_memory_origin";
        origin_call.arguments_json = R"({"memory_id":"global-launch-review"})";
        mm::Message origin_result = tools.execute_tool(origin_call, continued_conv_id);
        auto origin_json = nlohmann::json::parse(origin_result.content);
        CHECK(origin_json["memory"]["source_conv_id"] == source_conv_id);
        CHECK(origin_json["origin"]["conversation_id"] == source_conv_id);
        auto origin_tool_trace = mm::build_tool_access_trace(
            origin_call, origin_result, continued_conv_id);
        CHECK(origin_tool_trace.has_value());
        CHECK(origin_tool_trace->source_id == "global-launch-review");
        origin_result.trace_events.push_back(*origin_tool_trace);
        db.append_message(continued_conv_id, origin_result, 6);

        auto reloaded = db.load_conversation(continued_conv_id);
        CHECK(reloaded.has_value());
        CHECK(reloaded->messages.size() == 7);
        CHECK(reloaded->messages[4].trace_events.size() == 4);

        const auto& persisted_followup_traces = reloaded->messages[4].trace_events;
        const mm::TraceEvent* persisted_summary =
            find_trace_event(persisted_followup_traces, "Compaction summary reviewed");
        CHECK(persisted_summary != nullptr);
        CHECK(persisted_summary->source_id == source_conv_id);
        CHECK(persisted_summary->metadata["conversation_id"] == continued_conv_id);

        const mm::TraceEvent* persisted_local =
            find_trace_event(persisted_followup_traces, "Conversation-local memory reviewed");
        CHECK(persisted_local != nullptr);
        CHECK(persisted_local->source_id == "local-launch-review");
        CHECK(persisted_local->metadata["conversation_id"] == continued_conv_id);

        const mm::TraceEvent* persisted_global =
            find_trace_event(persisted_followup_traces, "Global memory reviewed");
        CHECK(persisted_global != nullptr);
        CHECK(persisted_global->source_id == "global-launch-review");
        CHECK(persisted_global->metadata["source_conv_id"] == source_conv_id);

        CHECK(reloaded->messages[5].trace_events.size() == 1);
        CHECK(reloaded->messages[5].trace_events[0].title ==
              "Conversation-local memories listed");
        CHECK(reloaded->messages[5].trace_events[0].source_id == continued_conv_id);

        CHECK(reloaded->messages[6].trace_events.size() == 1);
        CHECK(reloaded->messages[6].trace_events[0].title ==
              "Global memory origin accessed");
        CHECK(reloaded->messages[6].trace_events[0].source_id == "global-launch-review");
    }

    CHECK(remove_tree(dir));
    return true;
}

bool test_config_and_url_parsing_edge_cases() {
    auto dir = temp_test_dir("config-hash");
    std::filesystem::create_directories(dir);
    const auto cfg_path = dir / "test.toml";
    {
        std::ofstream f(cfg_path);
        f << "# full-line comment\n";
        f << "token = \"abc#123\" # trailing comment\n";
        f << "plain = value # comment\n";
        f << "single = 'x#y'\n";
    }
    mm::ConfigFile cfg;
    CHECK(cfg.load(cfg_path.string()));
    CHECK(cfg.get("token", "") == "abc#123");
    CHECK(cfg.get("plain", "") == "value");
    CHECK(cfg.get("single", "") == "x#y");

    CHECK(mm::util::parse_url("https://example.com") ==
          std::make_pair(std::string("example.com"), 443));
    CHECK(mm::util::parse_url("http://example.com:7070/path") ==
          std::make_pair(std::string("example.com"), 7070));
    CHECK(mm::util::parse_url("http://[::1]:9090") ==
          std::make_pair(std::string("::1"), 9090));
    CHECK(mm::util::parse_url("http://[::1]/x") ==
          std::make_pair(std::string("::1"), 80));
    CHECK(mm::util::parse_url("https://example.com:notaport") ==
          std::make_pair(std::string("example.com"), 443));

    CHECK(remove_tree(dir));
    return true;
}

} // namespace

// ── llama.cpp backend ─────────────────────────────────────────────────────────

bool test_llama_server_args() {
    auto has = [](const std::vector<std::string>& a, const std::string& f) {
        return std::find(a.begin(), a.end(), f) != a.end();
    };
    auto val_after = [](const std::vector<std::string>& a, const std::string& f) {
        auto it = std::find(a.begin(), a.end(), f);
        return (it != a.end() && std::next(it) != a.end()) ? *std::next(it) : std::string{};
    };
    auto count = [](const std::vector<std::string>& a, const std::string& f) {
        return static_cast<int>(std::count(a.begin(), a.end(), f));
    };

    mm::RuntimeSettings s;
    s.ctx_size = 4096;
    s.parallel = 2;            // server context = ctx_size * parallel = 8192
    s.n_gpu_layers = -1;
    s.flash_attn = true;
    s.batch_size = 512;
    s.ubatch_size = 256;
    const auto args = mm::build_llama_server_args("/models/m.gguf", s, 8081, "data/kv");

    CHECK(val_after(args, "--model") == "/models/m.gguf");
    CHECK(val_after(args, "--port") == "8081");
    CHECK(val_after(args, "--ctx-size") == "8192");
    CHECK(val_after(args, "--gpu-layers") == "-1");
    CHECK(has(args, "--flash-attn"));
    CHECK(val_after(args, "--flash-attn") == "on");   // valued form, not bare
    CHECK(val_after(args, "--batch-size") == "512");
    CHECK(val_after(args, "--ubatch-size") == "256");
    CHECK(val_after(args, "--parallel") == "2");
    CHECK(val_after(args, "--slot-save-path") == "data/kv");

    // extra_args override discipline: a user --ctx-size wins (no default added),
    // and -fa suppresses the default --flash-attn injection.
    mm::RuntimeSettings s2;
    s2.ctx_size = 2048;
    s2.parallel = 1;
    s2.flash_attn = true;
    s2.extra_args = {"--ctx-size", "1234", "-fa"};
    const auto args2 = mm::build_llama_server_args("m.gguf", s2, 8080, "");
    CHECK(count(args2, "--ctx-size") == 1);      // only the user's, no injected default
    CHECK(val_after(args2, "--ctx-size") == "1234");
    CHECK(count(args2, "--flash-attn") == 0);    // suppressed by -fa
    CHECK(!has(args2, "--slot-save-path"));       // omitted when empty

    const auto vision_args = mm::build_llama_server_args(
        "m.gguf", "mmproj-vision.gguf", s, 8082, "");
    CHECK(count(vision_args, "--mmproj") == 1);
    CHECK(val_after(vision_args, "--mmproj") == "mmproj-vision.gguf");
    return true;
}

bool test_vision_config_attachment_and_message_round_trip() {
    const auto dir = temp_test_dir("vision-db");
    {
        mm::AgentDB db("vision-agent", dir.string());

        mm::AgentConfig cfg;
        cfg.id = "vision-agent";
        cfg.name = "Vision Agent";
        cfg.model_path = "model.gguf";
        cfg.vision_settings.enabled = true;
        cfg.vision_settings.mmproj_path = "mmproj-model.gguf";
        db.save_config(cfg);
        const auto loaded_cfg = db.load_config();
        CHECK(loaded_cfg.vision_settings.enabled);
        CHECK(loaded_cfg.vision_settings.mmproj_path == "mmproj-model.gguf");

        mm::ImageAttachment attachment;
        attachment.id = "image-one";
        attachment.original_filename = "sample.png";
        attachment.mime_type = "image/png";
        attachment.relative_path = "attachments/image-one.png";
        attachment.size_bytes = 8;
        attachment.created_at_ms = mm::util::now_ms();
        attachment.expires_at_ms = attachment.created_at_ms + 60'000;
        const auto attachment_path = db.attachment_file_path(attachment);
        CHECK(!attachment_path.empty());
        {
            std::ofstream image(attachment_path, std::ios::binary);
            const unsigned char png_signature[] =
                {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
            image.write(reinterpret_cast<const char*>(png_signature),
                        sizeof(png_signature));
        }
        db.save_attachment(attachment);
        const auto public_json = nlohmann::json(attachment);
        CHECK(!public_json.contains("relative_path"));

        mm::ImageAttachment traversal = attachment;
        traversal.id = "outside";
        traversal.relative_path = "../outside.png";
        bool rejected_traversal = false;
        try {
            db.save_attachment(traversal);
        } catch (const std::invalid_argument&) {
            rejected_traversal = true;
        }
        CHECK(rejected_traversal);

        const auto conv_id = db.create_conversation("vision ordering");
        mm::Message old_user;
        old_user.role = mm::MessageRole::User;
        old_user.content = "older context that should be summarized";
        old_user.token_count = 3000;
        db.append_message(conv_id, old_user, 0);
        mm::Message old_assistant;
        old_assistant.role = mm::MessageRole::Assistant;
        old_assistant.content = "older response";
        old_assistant.token_count = 10;
        db.append_message(conv_id, old_assistant, 1);

        mm::Message message;
        message.role = mm::MessageRole::User;
        message.content = "before after";
        message.token_count = 2 + 2048;
        message.content_parts = {
            mm::MessageContentPart{"text", "before", {}, {}, {}},
            mm::MessageContentPart{"image_attachment", {}, attachment.id, {}, "image/png"},
            mm::MessageContentPart{"text", "after", {}, {}, {}}
        };
        db.append_message(conv_id, message, 2);
        mm::Message recent_assistant;
        recent_assistant.role = mm::MessageRole::Assistant;
        recent_assistant.content = "recent response";
        recent_assistant.token_count = 10;
        db.append_message(conv_id, recent_assistant, 3);

        const auto messages = db.load_messages(conv_id);
        CHECK(messages.size() == 4);
        CHECK(messages[2].content == "before after");
        CHECK(messages[2].content_parts.size() == 3);
        CHECK(messages[2].content_parts[0].text == "before");
        CHECK(messages[2].content_parts[1].type == "image_attachment");
        CHECK(messages[2].content_parts[1].attachment_id == attachment.id);
        CHECK(messages[2].content_parts[2].text == "after");
        CHECK(db.get_attachment(attachment.id)->expires_at_ms == 0);

        bool referenced = false;
        CHECK(!db.delete_attachment(attachment.id, &referenced));
        CHECK(referenced);

        FixedSummaryRuntimeClient summary_runtime;
        mm::ConversationManager conversation_manager(db, summary_runtime);
        cfg.runtime_settings.ctx_size = 4096;
        cfg.runtime_settings.max_tokens = 1024;
        const auto compacted_id = conversation_manager.force_compact(conv_id, cfg);
        CHECK(compacted_id != conv_id);
        const auto compacted_messages = db.load_messages(compacted_id);
        CHECK(compacted_messages.size() == 2);
        CHECK(compacted_messages[0].content_parts.size() == 3);
        CHECK(compacted_messages[0].content_parts[1].attachment_id == attachment.id);

        db.delete_conversation(conv_id);
        CHECK(db.get_attachment(attachment.id).has_value());
        CHECK(std::filesystem::exists(attachment_path));
        db.delete_conversation(compacted_id);
        CHECK(!db.get_attachment(attachment.id).has_value());
        CHECK(!std::filesystem::exists(attachment_path));

        mm::ImageAttachment expired;
        expired.id = "expired";
        expired.original_filename = "expired.jpg";
        expired.mime_type = "image/jpeg";
        expired.relative_path = "attachments/expired.jpg";
        expired.size_bytes = 3;
        expired.created_at_ms = mm::util::now_ms() - 1000;
        expired.expires_at_ms = mm::util::now_ms() - 1;
        const auto expired_path = db.attachment_file_path(expired);
        {
            std::ofstream image(expired_path, std::ios::binary);
            const unsigned char jpeg_signature[] = {0xff, 0xd8, 0xff};
            image.write(reinterpret_cast<const char*>(jpeg_signature),
                        sizeof(jpeg_signature));
        }
        db.save_attachment(expired);
        const auto removed = db.delete_expired_unreferenced_attachments(mm::util::now_ms());
        CHECK(removed.size() == 1);
        CHECK(removed[0].id == expired.id);
        CHECK(!std::filesystem::exists(expired_path));
    }
    CHECK(remove_tree(dir));
    return true;
}

bool test_vision_profile_validation_and_suggestions() {
    const auto dir = temp_test_dir("vision-validation");
    std::filesystem::create_directories(dir);
    const auto model_path = dir / "model.gguf";
    const auto projector_path = dir / "MMPROJ-Vision.GGUF";
    {
        std::ofstream model(model_path, std::ios::binary);
        model << "model";
        std::ofstream projector(projector_path, std::ios::binary);
        projector << "projector";
    }

    const auto suggestions = mm::suggest_mmproj_files(model_path.string());
    CHECK(suggestions.size() == 1);
    CHECK(std::filesystem::path(suggestions[0]).filename() == projector_path.filename());

    mm::AgentConfig cfg;
    cfg.name = "Vision";
    cfg.model_path = model_path.string();
    cfg.inference_backend = "llama-cpp";
    cfg.vision_settings.enabled = true;
    CHECK(!mm::validate_agent_config(cfg, nullptr, "", nullptr).ok());

    cfg.vision_settings.mmproj_path = projector_path.string();
    CHECK(mm::validate_agent_config(cfg, nullptr, "", nullptr).ok());

    cfg.runtime_settings.extra_args = {"--mmproj-url=https://invalid.example/mmproj"};
    CHECK(!mm::validate_agent_config(cfg, nullptr, "", nullptr).ok());
    cfg.runtime_settings.extra_args.clear();

    cfg.vision_settings.mmproj_path = (dir / "projector.gguf").string();
    const auto unconventional = mm::validate_agent_config(cfg, nullptr, "", nullptr);
    CHECK(unconventional.ok());
    CHECK(std::any_of(unconventional.issues.begin(), unconventional.issues.end(),
                      [](const mm::ValidationIssue& issue) {
                          return issue.field == "vision_settings.mmproj_path" &&
                                 issue.severity == mm::ValidationSeverity::Warning;
                      }));

    cfg.inference_backend = "vllm";
    const auto retired_backend = mm::validate_agent_config(cfg, nullptr, "", nullptr);
    CHECK(!retired_backend.ok());
    CHECK(std::any_of(retired_backend.issues.begin(), retired_backend.issues.end(),
                      [](const mm::ValidationIssue& issue) {
                          return issue.field == "inference_backend" &&
                                 issue.severity == mm::ValidationSeverity::Error;
                      }));

    cfg.inference_backend = "api";
    cfg.api_settings.base_url = "https://example.test";
    cfg.api_settings.chat_completions_path = "/v1/chat/completions";
    cfg.vision_settings.mmproj_path = projector_path.string();
    CHECK(!mm::validate_agent_config(cfg, nullptr, "", nullptr).ok());
    cfg.vision_settings.mmproj_path.clear();
    CHECK(mm::validate_agent_config(cfg, nullptr, "", nullptr).ok());

    CHECK(remove_tree(dir));
    return true;
}

bool test_vision_slot_projector_isolation_and_json() {
    mm::SlotManager slots(46250, 46253, 4);
    slots.set_llama_server_path("missing-llama-server");
    mm::RuntimeSettings settings;
    const auto first = slots.add_ready_test_slot(
        "model.gguf", "agent-a", settings, "mmproj-a.gguf");
    CHECK(!first.empty());

    const auto shared = slots.load_model(
        "model.gguf", "mmproj-a.gguf", settings, "agent-b");
    CHECK(shared == first);

    const auto different = slots.load_model(
        "model.gguf", "mmproj-b.gguf", settings, "agent-c");
    CHECK(different.empty());
    const auto info = slots.find_slot(first);
    CHECK(info.has_value());
    CHECK(info->backend == "llama-cpp");
    CHECK(info->vision_enabled);
    CHECK(info->mmproj_path == "mmproj-a.gguf");
    CHECK(std::find(info->agent_ids.begin(), info->agent_ids.end(), "agent-b") !=
          info->agent_ids.end());
    CHECK(std::find(info->agent_ids.begin(), info->agent_ids.end(), "agent-c") ==
          info->agent_ids.end());

    const auto slot_json = nlohmann::json(*info);
    CHECK(slot_json["vision_enabled"] == true);
    CHECK(slot_json["mmproj_path"] == "mmproj-a.gguf");

    mm::Message ordered;
    ordered.role = mm::MessageRole::User;
    ordered.content_parts = {
        mm::MessageContentPart{"text", "look", {}, {}, {}},
        mm::MessageContentPart{"image_url", {}, {},
                               "data:image/png;base64,iVBORw0KGgo=", "image/png"},
        mm::MessageContentPart{"text", "closely", {}, {}, {}}
    };
    const auto round_trip = nlohmann::json(ordered).get<mm::Message>();
    CHECK(round_trip.content_parts.size() == 3);
    CHECK(round_trip.content_parts[1].image_url ==
          "data:image/png;base64,iVBORw0KGgo=");
    return true;
}

bool test_llama_model_path_normalization() {
#ifdef _WIN32
    CHECK(mm::normalize_llama_model_path("/mnt/y/models/m.gguf") == "Y:\\models\\m.gguf");
    CHECK(mm::normalize_llama_model_path("\"C:\\a\\b.gguf\"") == "C:\\a\\b.gguf");
#else
    CHECK(mm::normalize_llama_model_path("  /models/m.gguf  ") == "/models/m.gguf");
#endif
    return true;
}

bool test_llama_accelerator_detection() {
    CHECK(mm::detect_llama_accelerator("linux", "x86_64", true, false) == "cuda");
    CHECK(mm::detect_llama_accelerator("linux", "x86_64", false, true) == "rocm");
    CHECK(mm::detect_llama_accelerator("linux", "x86_64", false, false) == "cpu");
    CHECK(mm::detect_llama_accelerator("macos", "aarch64", false, false) == "metal");
    // llama.cpp builds CUDA natively on Windows — no separate "windows" variant.
    CHECK(mm::detect_llama_accelerator("windows", "x86_64", true, false) == "cuda");
    return true;
}

bool test_llama_launch_compatible() {
    mm::RuntimeSettings a;
    a.ctx_size = 4096;
    a.parallel = 2;
    a.temperature = 0.7f;
    mm::RuntimeSettings b = a;
    b.temperature = 0.2f;   // generation params do not gate engine sharing
    b.max_tokens = 99;
    CHECK(mm::llama_launch_compatible(a, b));
    b.ctx_size = 8192;      // launch-time identity differs
    CHECK(!mm::llama_launch_compatible(a, b));
    return true;
}

bool test_llama_backend_validation_and_gguf_routing() {
    mm::AgentConfig cfg;
    cfg.name = "a";
    cfg.model_path = "/models/m.gguf";
    cfg.inference_backend = "llama-cpp";
    auto r = mm::validate_agent_config(cfg, nullptr, "", nullptr);
    CHECK(r.ok());

    // Legacy vLLM profiles must fail explicitly instead of silently routing to
    // llama.cpp with incompatible model/settings semantics.
    mm::AgentConfig retired = cfg;
    retired.inference_backend = "vllm";
    const auto retired_result =
        mm::validate_agent_config(retired, nullptr, "", nullptr);
    CHECK(!retired_result.ok());
    CHECK(std::any_of(retired_result.issues.begin(), retired_result.issues.end(),
                      [](const mm::ValidationIssue& issue) {
                          return issue.field == "inference_backend" &&
                                 issue.severity == mm::ValidationSeverity::Error &&
                                 issue.message.find("not available") != std::string::npos;
                      }));

    // llama-cpp with an HF repo id warns (needs a local GGUF).
    mm::AgentConfig h = cfg;
    h.model_path = "Qwen/Qwen3-8B";
    auto rh = mm::validate_agent_config(h, nullptr, "", nullptr);
    bool hf_warn = false;
    for (const auto& i : rh.issues)
        if (i.field == "model_path" && i.severity == mm::ValidationSeverity::Warning)
            hf_warn = true;
    CHECK(hf_warn);

    // An unknown backend is a hard error.
    mm::AgentConfig bad = cfg;
    bad.inference_backend = "tensorrt";
    CHECK(!mm::validate_agent_config(bad, nullptr, "", nullptr).ok());

    // The legacy "llama.cpp"/"llama" spellings normalize to the llama backend.
    mm::AgentConfig legacy = cfg;
    legacy.inference_backend = "llama.cpp";
    CHECK(mm::validate_agent_config(legacy, nullptr, "", nullptr).ok());
    return true;
}

bool test_llama_install_plan_and_method() {
    CHECK(mm::normalize_llama_install_method("SOURCE") == "source");
    CHECK(mm::normalize_llama_install_method("release") == "release");
    CHECK(mm::normalize_llama_install_method("garbage") == "auto");
    CHECK(mm::normalize_llama_install_method("") == "auto");

    mm::LlamaProvisionConfig cfg;
    cfg.platform = "linux";
    cfg.arch = "x86_64";
    cfg.accelerator = "cuda";
    cfg.cuda_arch = "121";          // DGX Spark GB10 sm_121
    cfg.install_method = "source";
    cfg.provision_dir = "data/llama-plan-test";
    const auto plan = mm::build_llama_install_plan(cfg, false);
    CHECK(!plan.empty());

    std::string joined;
    for (const auto& step : plan)
        for (const auto& a : step.argv) joined += a + " ";
    CHECK(joined.find("clone") != std::string::npos);
    CHECK(joined.find("-DGGML_CUDA=ON") != std::string::npos);
    CHECK(joined.find("-DCMAKE_CUDA_ARCHITECTURES=121a-real") != std::string::npos);
    CHECK(joined.find("CMAKE_CUDA_FLAGS=-arch") == std::string::npos);
    CHECK(joined.find("sm_$cuda_probe_arch") != std::string::npos);
    CHECK(joined.find("CUDA compiler/assembler smoke test") != std::string::npos);
    CHECK(joined.find("CMakeCache.txt") != std::string::npos);
    CHECK(joined.find("CMakeFiles") != std::string::npos);
    CHECK(joined.find("linux-x64-cuda") != std::string::npos); // isolated CMake cache
    CHECK(joined.find("nvcc") != std::string::npos);            // CUDA preflight
    CHECK(joined.find("--parallel 2") != std::string::npos);    // conservative default
    CHECK(joined.find("llama-server") != std::string::npos);  // build target

    // Auto starts with an official release lookup, not a source build.
    cfg.install_method = "auto";
    const auto auto_plan = mm::build_llama_install_plan(cfg, false);
    CHECK(auto_plan.size() == 1);
    CHECK(auto_plan.front().argv.front() == "python3");
    std::string auto_joined;
    for (const auto& a : auto_plan.front().argv) auto_joined += a + " ";
    CHECK(auto_joined.find("api.github.com/repos/ggml-org/llama.cpp") != std::string::npos);
    CHECK(auto_joined.find("bin-ubuntu-cuda") != std::string::npos);
    CHECK(auto_joined.find("git clone") == std::string::npos);

    // Current Windows releases need the base/server archive plus the selected
    // backend, and CUDA needs the matching runtime DLL archive as well.
    cfg.platform = "windows";
    const auto windows_plan = mm::build_llama_install_plan(cfg, false);
    CHECK(windows_plan.size() == 1);
    CHECK(windows_plan.front().argv.front() == "powershell");
    std::string windows_joined;
    for (const auto& a : windows_plan.front().argv) windows_joined += a + " ";
    CHECK(windows_joined.find("bin-win-cpu") != std::string::npos);
    CHECK(windows_joined.find("bin-win-cuda") != std::string::npos);
    CHECK(windows_joined.find("cudart-llama-bin-win-cuda") != std::string::npos);
    CHECK(windows_joined.find("nvidia-smi") != std::string::npos);
    return true;
}

bool test_llama_provisioner_disabled_and_cancel() {
    auto dir = temp_test_dir("llama-prov");
    // auto-provision disabled + missing executable => "disabled", no exe.
    mm::LlamaProvisionConfig cfg;
    cfg.requested_executable = "definitely-not-a-real-llama-server-xyz";
    cfg.provision_dir = (dir / "prov").string();
    cfg.auto_provision = false;
    cfg.platform = "linux";
    cfg.arch = "x86_64";
    cfg.accelerator = "cpu";
    mm::LlamaCppProvisioner prov(cfg);
    const auto st = prov.ensure_runtime();
    CHECK(st.status == "disabled");
    CHECK(st.executable_path.empty());

    // A cancel check that trips before the first step yields a canceled failure
    // and never runs a command.
    mm::LlamaProvisionConfig cfg2 = cfg;
    cfg2.auto_provision = true;
    bool ran = false;
    mm::LlamaCommandRunner runner;
    runner.run = [&](const std::vector<std::string>&, const std::filesystem::path&,
                     const mm::StreamLineCallback&, const mm::CancelCheckCallback&,
                     std::string*) { ran = true; return 0; };
    runner.capture_first_line = [](const std::vector<std::string>&,
                                   const std::filesystem::path&) { return std::string{}; };
    mm::LlamaCppProvisioner prov2(cfg2, runner);
    prov2.set_cancel_check([] { return true; });
    const auto st2 = prov2.ensure_runtime();
    CHECK(st2.status == "failed");
    CHECK(st2.last_error.find("canceled") != std::string::npos);
    CHECK(!ran);

    CHECK(remove_tree(dir));
    return true;
}

bool test_llama_path_resolution_respects_accelerator() {
    auto dir = temp_test_dir("llama-path-accelerator");
    mm::LlamaProvisionConfig cfg;
    cfg.requested_executable = "llama-server";
    cfg.provision_dir = (dir / "cuda").string();
    cfg.auto_provision = true;
    cfg.install_method = "release";
    cfg.version = "b2000";
    cfg.platform = "windows";
    cfg.arch = "x86_64";
    cfg.accelerator = "cuda";
    cfg.accelerator_explicit = true;

    const std::string winget =
        "C:/Users/test/AppData/Local/Microsoft/WinGet/Packages/ggml.llamacpp/llama-server.exe";
    bool ran_managed_install = false;
    bool fail_managed_install = false;
    mm::LlamaCommandRunner runner;
    runner.resolve_executable = [winget](const std::string&) { return winget; };
    runner.run = [&](const std::vector<std::string>&,
                     const std::filesystem::path&,
                     const mm::StreamLineCallback&,
                     const mm::CancelCheckCallback&,
                     std::string* error) {
        ran_managed_install = true;
        if (fail_managed_install) {
            if (error) *error = "simulated target install failure";
            return 1;
        }
        const auto executable = std::filesystem::path(cfg.provision_dir) /
            "release" / "bin" / "llama-server.exe";
        std::filesystem::create_directories(executable.parent_path());
        std::ofstream(executable) << "fake managed CUDA llama-server";
        return 0;
    };
    runner.capture_first_line = [](const std::vector<std::string>& argv,
                                   const std::filesystem::path&) {
        return !argv.empty() && argv.front().find("WinGet") != std::string::npos
            ? std::string{"'--version' is not recognized as a valid option"}
            : std::string{"llama.cpp version: 2000 (managedcuda)"};
    };
    runner.fetch_latest = [](const mm::LlamaProvisionConfig&) {
        return std::string{"b2000"};
    };
    runner.fetch_release_assets = [](const mm::LlamaProvisionConfig&,
                                     const std::string&) {
        return std::vector<std::string>{};
    };

    mm::LlamaCppProvisioner cuda(cfg, runner);
    const auto managed = cuda.ensure_runtime();
    CHECK(ran_managed_install);
    CHECK(managed.status == "ready");
    CHECK(managed.managed);
    CHECK(managed.method == "release");
    CHECK(managed.accelerator == "cuda");
    CHECK(managed.variant == "cuda");
    CHECK(managed.target_method == "release");
    CHECK(managed.target_accelerator == "cuda");
    CHECK(managed.target_variant == "cuda");
    CHECK(!managed.target_mismatch);
    CHECK(!managed.available_variants.empty());
    CHECK(managed.executable_path.find("WinGet") == std::string::npos);

    // The same generic PATH result must not block managed update operations.
    ran_managed_install = false;
    const auto updated = cuda.update_runtime();
    CHECK(ran_managed_install);
    CHECK(updated.status == "ready");
    CHECK(updated.managed);
    CHECK(updated.last_error.empty());

    // The generic PATH executable must not shadow the active managed CUDA
    // selection on restart.
    ran_managed_install = false;
    mm::LlamaCppProvisioner restarted(cfg, runner);
    const auto restored = restarted.ensure_runtime();
    CHECK(!ran_managed_install);
    CHECK(restored.status == "ready");
    CHECK(restored.managed);
    CHECK(restored.accelerator == "cuda");
    CHECK(restored.executable_path == managed.executable_path);
    CHECK(restored.build_log_path == updated.build_log_path);

    // A generic PATH build remains valid for CPU nodes.
    auto cpu_cfg = cfg;
    cpu_cfg.provision_dir = (dir / "cpu").string();
    cpu_cfg.accelerator = "cpu";
    ran_managed_install = false;
    mm::LlamaCppProvisioner cpu(cpu_cfg, runner);
    const auto cpu_status = cpu.ensure_runtime();
    CHECK(!ran_managed_install);
    CHECK(cpu_status.status == "resolved");
    CHECK(!cpu_status.managed);
    CHECK(cpu_status.method == "path");
    CHECK(cpu_status.executable_path == winget);
    CHECK(cpu_status.version.empty());

    // Disabling auto-provision or naming an explicit path is an intentional
    // user override even for an accelerator node.
    auto path_only_cfg = cfg;
    path_only_cfg.provision_dir = (dir / "path-only").string();
    path_only_cfg.auto_provision = false;
    mm::LlamaCppProvisioner path_only(path_only_cfg, runner);
    CHECK(path_only.ensure_runtime().executable_path == winget);

    auto explicit_cfg = cfg;
    explicit_cfg.provision_dir = (dir / "explicit").string();
    explicit_cfg.requested_executable = "C:/custom/llama-server.exe";
    mm::LlamaCppProvisioner explicit_runtime(explicit_cfg, runner);
    const auto explicit_status = explicit_runtime.ensure_runtime();
    CHECK(explicit_status.status == "resolved");
    CHECK(explicit_status.executable_path == winget);
    CHECK(explicit_status.version.empty());

    // Backend changes are available even when no version update is pending.
    ran_managed_install = false;
    const auto switched = restarted.switch_runtime("vulkan");
    CHECK(ran_managed_install);
    CHECK(switched.status == "ready");
    CHECK(switched.managed);
    CHECK(switched.accelerator == "vulkan");
    CHECK(switched.variant == "vulkan");
    CHECK(switched.target_accelerator == "cuda");
    CHECK(switched.target_variant == "cuda");
    CHECK(switched.target_mismatch);
    CHECK(switched.target_mismatch_reason.find("targets cuda") != std::string::npos);
    nlohmann::json switched_json = switched;
    const auto switched_round_trip = switched_json.get<mm::LlamaRuntimeStatus>();
    CHECK(switched_round_trip.variant == "vulkan");
    CHECK(switched_round_trip.target_accelerator == "cuda");
    CHECK(switched_round_trip.target_mismatch);
    CHECK(switched_round_trip.available_variants.size() ==
          switched.available_variants.size());

    ran_managed_install = false;
    mm::LlamaCppProvisioner switched_restart(cfg, runner);
    const auto switched_restored = switched_restart.ensure_runtime();
    CHECK(!ran_managed_install);
    CHECK(switched_restored.accelerator == "vulkan");
    CHECK(switched_restored.variant == "vulkan");
    CHECK(switched_restored.target_accelerator == "cuda");
    CHECK(switched_restored.target_mismatch);

    // A failed target attempt does not replace the persisted fallback marker.
    fail_managed_install = true;
    const auto failed_target = switched_restart.recover_runtime("target");
    CHECK(failed_target.status == "failed");
    CHECK(failed_target.last_error.find("simulated target install failure") !=
          std::string::npos);
    fail_managed_install = false;
    const auto after_failed_target = switched_restart.ensure_runtime();
    CHECK(after_failed_target.accelerator == "vulkan");
    CHECK(after_failed_target.target_mismatch);

    // The explicit configured target no longer silently displaces a working
    // fallback at startup. A deliberate target recovery installs it instead.
    ran_managed_install = false;
    const auto target = switched_restart.recover_runtime("target");
    CHECK(ran_managed_install);
    CHECK(target.status == "ready");
    CHECK(target.accelerator == "cuda");
    CHECK(target.target_accelerator == "cuda");
    CHECK(!target.target_mismatch);

    CHECK(remove_tree(dir));
    return true;
}

bool test_llama_auto_release_then_source_fallback() {
    auto dir = temp_test_dir("llama-auto-fallback");
    mm::LlamaProvisionConfig cfg;
    cfg.requested_executable = "definitely-not-a-real-llama-server-auto";
    cfg.provision_dir = (dir / "prov").string();
    cfg.auto_provision = true;
    cfg.install_method = "auto";
    cfg.platform = "linux";
    cfg.arch = "x86_64";
    cfg.accelerator = "cuda";
    cfg.cuda_arch = "120";

    // Seed more than the retention limit; the new attempt should prune the
    // oldest managed-runtime transcripts before creating its own.
    const auto logs_dir = dir / "prov" / "logs";
    std::filesystem::create_directories(logs_dir);
    for (int i = 0; i < 21; ++i) {
        std::ofstream(logs_dir / ("llama-build-old-" + std::to_string(i) + ".log"))
            << "old log";
    }

    bool saw_release = false;
    bool saw_source = false;
    mm::LlamaCommandRunner runner;
    runner.run = [&](const std::vector<std::string>& argv,
                     const std::filesystem::path&,
                     const mm::StreamLineCallback&,
                     const mm::CancelCheckCallback&,
                     std::string* error) {
        if (!argv.empty() && argv.front() == "python3") {
            saw_release = true;
            if (error) *error = "no matching Linux CUDA release";
            return 1;
        }
        saw_source = true;
        const auto build_it = std::find(argv.begin(), argv.end(), "--build");
        if (build_it != argv.end() && std::next(build_it) != argv.end()) {
            const auto exe = std::filesystem::path(*std::next(build_it)) / "bin"
                / "llama-server";
            std::filesystem::create_directories(exe.parent_path());
            std::ofstream(exe) << "fake llama-server";
        }
        return 0;
    };
    runner.capture_first_line = [](const std::vector<std::string>&,
                                   const std::filesystem::path&) {
        return std::string{"version b9999"};
    };

    mm::LlamaCppProvisioner provisioner(cfg, runner);
    const auto status = provisioner.ensure_runtime();
    CHECK(saw_release);
    CHECK(saw_source);
    CHECK(status.status == "ready");
    CHECK(status.method == "source");
    CHECK(status.cuda_architecture == "120a-real");
    CHECK(status.target_cuda_architecture == "120a-real");
    CHECK(!status.target_mismatch);
    auto stale_architecture = status;
    stale_architecture.cuda_architecture = "120";
    CHECK(mm::llama_runtime_target_mismatch_reason(stale_architecture).find(
              "targets 120a-real") != std::string::npos);
    CHECK(status.executable_path.find("llama.cpp-src") != std::string::npos);
    CHECK(!status.build_log_path.empty());
    CHECK(std::filesystem::exists(status.build_log_path));
    {
        std::ifstream log(status.build_log_path);
        const std::string text((std::istreambuf_iterator<char>(log)),
                               std::istreambuf_iterator<char>());
        CHECK(text.find("operation: install") != std::string::npos);
        CHECK(text.find("command:") != std::string::npos);
        CHECK(text.find("no matching Linux CUDA release") != std::string::npos);
        CHECK(text.find("result: success") != std::string::npos);
    }
    size_t retained_logs = 0;
    for (const auto& entry : std::filesystem::directory_iterator(logs_dir)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && name.rfind("llama-build-", 0) == 0 &&
            entry.path().extension() == ".log") {
            ++retained_logs;
        }
    }
    CHECK(retained_logs == 20);

    mm::LlamaCppProvisioner restarted(cfg, runner);
    const auto restored = restarted.ensure_runtime();
    CHECK(restored.cuda_architecture == "120a-real");
    CHECK(restored.target_cuda_architecture == "120a-real");
    CHECK(!restored.target_mismatch);

    CHECK(remove_tree(dir));
    return true;
}

bool test_llama_update_release_decision() {
    auto dir = temp_test_dir("llama-update-decision");
    mm::LlamaProvisionConfig cfg;
    cfg.requested_executable = "definitely-not-a-real-llama-server-update";
    cfg.provision_dir = (dir / "prov").string();
    cfg.auto_provision = false;
    cfg.install_method = "auto";
    cfg.platform = "linux";
    cfg.arch = "x86_64";
    cfg.accelerator = "cuda";

    auto source_cfg = cfg;
    source_cfg.install_method = "source";
    const auto managed = mm::managed_llama_executable_path(source_cfg);
    std::filesystem::create_directories(managed.parent_path());
    std::ofstream(managed) << "fake llama-server";

    const std::vector<std::string> assets{
        "llama-b1001-bin-ubuntu-x64.tar.gz",
        "llama-b1001-bin-ubuntu-vulkan-x64.tar.gz",
    };
    const auto available = mm::llama_release_accelerators(assets, cfg);
    CHECK(available == std::vector<std::string>({"vulkan", "cpu"}));

    mm::LlamaCommandRunner runner;
    runner.run = [&](const std::vector<std::string>& argv,
                     const std::filesystem::path&,
                     const mm::StreamLineCallback&,
                     const mm::CancelCheckCallback&,
                     std::string*) {
        if (!argv.empty() && argv.front() == "python3") {
            const auto executable = dir / "prov" / "release" / "bin" / "llama-server";
            std::filesystem::create_directories(executable.parent_path());
            std::ofstream(executable) << "fake Vulkan llama-server";
        }
        return 0;
    };
    runner.capture_first_line = [](const std::vector<std::string>&,
                                   const std::filesystem::path&) {
        return std::string{"llama.cpp version: 1000 (deadbeef)"};
    };
    std::string latest = "b1000";
    runner.fetch_latest = [&](const mm::LlamaProvisionConfig&) {
        return latest;
    };
    runner.fetch_release_assets = [assets](const mm::LlamaProvisionConfig&,
                                           const std::string&) { return assets; };

    mm::LlamaCppProvisioner provisioner(cfg, runner);
    CHECK(provisioner.ensure_runtime().status == "ready");
    CHECK(!provisioner.check_for_update().update_available);
    latest = "b1001";
    const auto update = provisioner.check_for_update();
    CHECK(update.update_available);
    CHECK(update.update_action == "compile");
    CHECK(!update.update_release_available);
    CHECK(update.update_release_alternatives ==
          std::vector<std::string>({"vulkan", "cpu"}));
    CHECK(update.update_warning.find("compile llama-server from source") != std::string::npos);
    CHECK(!update.available_variants.empty());

    nlohmann::json encoded = update;
    const auto decoded = encoded.get<mm::LlamaRuntimeStatus>();
    CHECK(decoded.update_action == "compile");
    CHECK(decoded.update_release_alternatives == update.update_release_alternatives);
    CHECK(decoded.available_variants.size() == update.available_variants.size());

    const auto switched = provisioner.update_runtime("vulkan");
    CHECK(switched.status == "ready");
    CHECK(switched.method == "release");
    CHECK(switched.accelerator == "vulkan");
    CHECK(std::filesystem::exists(dir / "prov" / "active-runtime.json"));

    mm::LlamaCppProvisioner restarted(cfg, runner);
    const auto restored = restarted.ensure_runtime();
    CHECK(restored.status == "ready");
    CHECK(restored.method == "release");
    CHECK(restored.accelerator == "vulkan");
    const auto restored_update = restarted.check_for_update();
    CHECK(restored_update.update_action == "release");
    CHECK(restored_update.update_release_available);
    CHECK(restarted.update_runtime().accelerator == "vulkan");

    CHECK(remove_tree(dir));
    return true;
}

bool test_llama_runtime_variant_matrix() {
    mm::LlamaProvisionConfig cfg;
    cfg.platform = "windows";
    cfg.arch = "x86_64";
    cfg.accelerator = "cuda";
    const std::vector<std::string> windows_assets{
        "llama-b2000-bin-win-cpu-x64.zip",
        "llama-b2000-bin-win-cuda-12.4-x64.zip",
        "llama-b2000-bin-win-cuda-13.3-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.3-x64.zip",
        "llama-b2000-bin-win-vulkan-x64.zip",
        "llama-b2000-bin-win-openvino-2026.2.1-x64.zip",
    };
    const auto windows = mm::llama_runtime_variants(windows_assets, cfg);
    auto find = [](const auto& variants, const std::string& id) {
        return std::find_if(variants.begin(), variants.end(),
                            [&](const auto& v) { return v.id == id; });
    };
    CHECK(find(windows, "cuda-12") != windows.end());
    CHECK(find(windows, "cuda-12")->release_available);
    CHECK(find(windows, "cuda-13")->release_available);
    CHECK(find(windows, "vulkan")->release_available);
    CHECK(find(windows, "openvino")->release_available);
    CHECK(find(windows, "cpu")->release_available);
    CHECK(!find(windows, "sycl-fp32")->release_available); // backend DLL is not a server bundle
    CHECK(find(windows, "sycl-fp32")->source_supported);
    CHECK(!find(windows, "metal")->platform_supported);

    cfg.platform = "linux";
    cfg.arch = "s390x";
    const auto s390x = mm::llama_runtime_variants(
        {"llama-b2000-bin-ubuntu-s390x.tar.gz"}, cfg);
    CHECK(find(s390x, "cpu")->release_available);
    CHECK(!find(s390x, "vulkan")->platform_supported);
    CHECK(!find(s390x, "cuda-13")->platform_supported);

    cfg.platform = "macos";
    cfg.arch = "apple-silicon";
    const auto apple = mm::llama_runtime_variants(
        {"llama-b2000-bin-macos-arm64.tar.gz"}, cfg);
    CHECK(find(apple, "metal")->release_available);
    CHECK(find(apple, "cpu")->release_available);
    return true;
}

bool test_llama_failure_diagnostics_and_recovery() {
    auto dir = temp_test_dir("llama-diagnostics");
    mm::LlamaProvisionConfig cfg;
    cfg.requested_executable = "definitely-not-a-real-llama-server-diagnostics";
    cfg.provision_dir = (dir / "prov").string();
    cfg.auto_provision = true;
    cfg.install_method = "auto";
    cfg.platform = "linux";
    cfg.arch = "x86_64";
    cfg.accelerator = "cuda";

    const std::vector<std::string> assets{
        "llama-b2000-bin-ubuntu-x64.tar.gz",
        "llama-b2000-bin-ubuntu-vulkan-x64.tar.gz",
    };
    mm::LlamaCommandRunner runner;
    std::string recovery_mode;
    std::vector<std::vector<std::string>> recovery_commands;
    runner.run = [&](const std::vector<std::string>& argv,
                    const std::filesystem::path&,
                    const mm::StreamLineCallback&,
                    const mm::CancelCheckCallback&,
                    std::string* error) {
        recovery_commands.push_back(argv);
        if (recovery_mode == "release") {
            const auto executable = dir / "prov" / "release" / "bin" / "llama-server";
            std::filesystem::create_directories(executable.parent_path());
            std::ofstream(executable) << "fake release llama-server";
            return 0;
        }
        if (recovery_mode == "compile") {
            const auto build = std::find(argv.begin(), argv.end(), "--build");
            if (build != argv.end() && std::next(build) != argv.end()) {
                const auto executable = std::filesystem::path(*std::next(build)) /
                    "bin" / "llama-server";
                std::filesystem::create_directories(executable.parent_path());
                std::ofstream(executable) << "fake compiled llama-server";
            }
            return 0;
        }
        if (error) *error = "simulated compiler failure";
        return 1;
    };
    runner.capture_output = [](const std::vector<std::string>& argv,
                               const std::filesystem::path&) {
        std::string joined;
        for (const auto& arg : argv) joined += arg + " ";
        if (joined.find("kernel/osrelease") != std::string::npos) return std::string{"WSL"};
        if (joined.find("MemAvailable") != std::string::npos) return std::string{"8.00 GiB free"};
        if (!argv.empty()) {
            const std::string tool = argv.back();
            if (tool == "git" || tool == "cmake" || tool == "c++" ||
                tool == "nvidia-smi")
                return std::string{"/usr/bin/"} + tool;
            if (tool == "nvcc") return std::string{};
        }
        return std::string{};
    };
    runner.capture_first_line = [](const std::vector<std::string>&,
                                   const std::filesystem::path&) {
        return std::string{};
    };
    runner.fetch_latest = [](const mm::LlamaProvisionConfig&) {
        return std::string{"b2000"};
    };
    runner.fetch_release_assets = [assets](const mm::LlamaProvisionConfig&,
                                           const std::string&) { return assets; };

    mm::LlamaCppProvisioner provisioner(cfg, runner);
    const auto failed = provisioner.ensure_runtime();
    CHECK(failed.status == "failed");
    CHECK(failed.troubleshooting.required);
    CHECK(failed.troubleshooting.platform == "linux");
    CHECK(failed.troubleshooting.architecture == "x64");
    CHECK(failed.troubleshooting.can_override_checks);
    CHECK(!failed.troubleshooting.fingerprint.empty());
    CHECK(std::any_of(failed.troubleshooting.checks.begin(),
                      failed.troubleshooting.checks.end(), [](const auto& check) {
        return check.id == "cuda-toolkit" && check.status == "fail" && check.blocking;
    }));
    CHECK(std::any_of(failed.troubleshooting.checks.begin(),
                      failed.troubleshooting.checks.end(), [](const auto& check) {
        return check.id == "cuda-driver-toolkit-mismatch" && check.status == "fail";
    }));
    CHECK(std::count_if(failed.troubleshooting.variants.begin(),
                        failed.troubleshooting.variants.end(), [](const auto& variant) {
        return variant.release_available;
    }) == 2);
    CHECK(!failed.build_log_path.empty());
    CHECK(std::filesystem::exists(failed.build_log_path));
    {
        std::ifstream log(failed.build_log_path);
        const std::string text((std::istreambuf_iterator<char>(log)),
                               std::istreambuf_iterator<char>());
        CHECK(text.find("simulated compiler failure") != std::string::npos);
        CHECK(text.find("result: failed") != std::string::npos);
        CHECK(text.find("llama.cpp troubleshooting report") != std::string::npos);
    }
    const std::string report_text = mm::format_llama_troubleshooting_report(
        failed.troubleshooting, failed.build_log_path);
    CHECK(report_text.find("Full build log: " + failed.build_log_path) !=
          std::string::npos);
    CHECK(report_text.find("simulated compiler failure") != std::string::npos);

    nlohmann::json encoded = failed;
    const auto decoded = encoded.get<mm::LlamaRuntimeStatus>();
    CHECK(decoded.build_log_path == failed.build_log_path);
    CHECK(decoded.troubleshooting.summary == failed.troubleshooting.summary);
    CHECK(decoded.troubleshooting.checks.size() == failed.troubleshooting.checks.size());

    auto normal_source = cfg;
    normal_source.install_method = "source";
    auto normal_plan = mm::build_llama_install_plan(normal_source, false);
    CHECK(std::any_of(normal_plan.begin(), normal_plan.end(), [](const auto& step) {
        return step.label == "Checking source-build prerequisites";
    }));
    normal_source.bypass_environment_checks = true;
    auto bypass_plan = mm::build_llama_install_plan(normal_source, false);
    CHECK(std::none_of(bypass_plan.begin(), bypass_plan.end(), [](const auto& step) {
        return step.label == "Checking source-build prerequisites";
    }));

    recovery_mode = "release";
    recovery_commands.clear();
    const auto released = provisioner.recover_runtime("release", "cpu");
    CHECK(released.status == "ready");
    CHECK(released.method == "release");
    CHECK(released.accelerator == "cpu");

    recovery_mode = "compile";
    recovery_commands.clear();
    const auto compiled = provisioner.recover_runtime("compile-anyway");
    CHECK(compiled.status == "ready");
    CHECK(compiled.method == "source");
    CHECK(std::none_of(recovery_commands.begin(), recovery_commands.end(),
                       [](const auto& argv) {
        return std::find(argv.begin(), argv.end(), "mantic-mind-llama-preflight") != argv.end();
    }));

    CHECK(remove_tree(dir));
    return true;
}

bool test_llama_nvcc_architecture_preflight_and_diagnostics() {
    auto dir = temp_test_dir("llama-nvcc-architecture");
    mm::LlamaProvisionConfig cfg;
    cfg.requested_executable = "definitely-not-a-real-llama-server-nvcc";
    cfg.provision_dir = (dir / "prov").string();
    cfg.auto_provision = true;
    cfg.install_method = "source";
    cfg.platform = "linux";
    cfg.arch = "x86_64";
    cfg.accelerator = "cuda";
    cfg.cuda_arch = "120";

    const auto plan = mm::build_llama_install_plan(cfg, false);
    const auto preflight = std::find_if(plan.begin(), plan.end(), [](const auto& step) {
        return step.label == "Checking source-build prerequisites";
    });
    CHECK(preflight != plan.end());
    std::string preflight_text;
    for (const auto& arg : preflight->argv) preflight_text += arg + " ";
    CHECK(preflight_text.find("--list-gpu-arch") != std::string::npos);
    CHECK(preflight_text.find("CUDA Toolkit 12.8 or newer") != std::string::npos);
    CHECK(preflight->argv[preflight->argv.size() - 2] == "120a");
    CHECK(preflight->argv.back() == "120a");
    const auto configure = std::find_if(plan.begin(), plan.end(), [](const auto& step) {
        return step.label == "Configuring llama.cpp (CMake)";
    });
    CHECK(configure != plan.end());
    CHECK(std::find(configure->argv.begin(), configure->argv.end(),
                    "-DCMAKE_CUDA_ARCHITECTURES=120a-real") !=
          configure->argv.end());
    CHECK(std::none_of(configure->argv.begin(), configure->argv.end(),
                       [](const auto& arg) {
        return arg.rfind("-DCMAKE_CUDA_FLAGS=-arch", 0) == 0;
    }));

    bool supports_120 = false;
    mm::LlamaCommandRunner runner;
    runner.run = [](const std::vector<std::string>&,
                    const std::filesystem::path&,
                    const mm::StreamLineCallback&,
                    const mm::CancelCheckCallback&,
                    std::string* error) {
        if (error) *error = "Selected NVCC /usr/bin/nvcc does not support compute_120";
        return 3;
    };
    runner.capture_output = [&](const std::vector<std::string>& argv,
                                const std::filesystem::path&) {
        std::string joined;
        for (const auto& arg : argv) joined += arg + " ";
        if (joined.find("kernel/osrelease") != std::string::npos)
            return std::string{"WSL"};
        if (joined.find("MemAvailable") != std::string::npos)
            return std::string{"8.00 GiB free"};
        if (joined.find("CUDACXX") != std::string::npos &&
            joined.find("command -v nvcc") != std::string::npos)
            return std::string{"/usr/bin/nvcc"};
        if (argv.size() == 2 && argv[0] == "/usr/bin/nvcc" && argv[1] == "--version")
            return std::string{"Cuda compilation tools, release 11.5, V11.5.119"};
        if (argv.size() == 2 && argv[0] == "/usr/bin/nvcc" &&
            argv[1] == "--list-gpu-arch")
            return supports_120 ? std::string{"compute_50\ncompute_90\ncompute_120\ncompute_120a"}
                                : std::string{"compute_50\ncompute_90"};
        if (!argv.empty()) {
            const std::string tool = argv.back();
            if (tool == "git" || tool == "cmake" || tool == "c++" ||
                tool == "nvidia-smi")
                return std::string{"/usr/bin/"} + tool;
        }
        return std::string{};
    };
    runner.capture_first_line = [](const std::vector<std::string>&,
                                   const std::filesystem::path&) {
        return std::string{};
    };
    runner.fetch_latest = [](const mm::LlamaProvisionConfig&) {
        return std::string{"b2000"};
    };
    runner.fetch_release_assets = [](const mm::LlamaProvisionConfig&,
                                     const std::string&) {
        return std::vector<std::string>{};
    };

    mm::LlamaCppProvisioner provisioner(cfg, runner);
    const auto failed = provisioner.ensure_runtime();
    const auto incompatible = std::find_if(
        failed.troubleshooting.checks.begin(), failed.troubleshooting.checks.end(),
        [](const auto& check) { return check.id == "cuda-architecture"; });
    CHECK(incompatible != failed.troubleshooting.checks.end());
    CHECK(incompatible->status == "fail");
    CHECK(incompatible->blocking);
    CHECK(incompatible->detected.find("/usr/bin/nvcc") != std::string::npos);
    CHECK(incompatible->detected.find("release 11.5") != std::string::npos);
    CHECK(incompatible->required.find("CUDA Toolkit 12.8") != std::string::npos);
    CHECK(incompatible->remediation.find("nvidia-smi") != std::string::npos);

    supports_120 = true;
    const auto refreshed = provisioner.diagnose_environment();
    const auto compatible = std::find_if(
        refreshed.troubleshooting.checks.begin(), refreshed.troubleshooting.checks.end(),
        [](const auto& check) { return check.id == "cuda-architecture"; });
    CHECK(compatible != refreshed.troubleshooting.checks.end());
    CHECK(compatible->status == "pass");
    CHECK(!compatible->blocking);

    // CUDA 13 removes sm_52. Older CMake compiler-identification paths can
    // still try that default before target CUDA_ARCHITECTURES take effect;
    // diagnose it explicitly even when NVCC advertises sm_120 support.
    mm::LlamaCommandRunner compiler_id_runner = runner;
    compiler_id_runner.run = [](const std::vector<std::string>&,
                                const std::filesystem::path&,
                                const mm::StreamLineCallback&,
                                const mm::CancelCheckCallback&,
                                std::string* error) {
        if (error) {
            *error =
                "ptxas -arch=sm_52 tmp/CMakeCUDACompilerId.ptx\n"
                "ptxas fatal : Value 'sm_52' is not defined for option 'gpu-name'\n"
                "/usr/share/cmake/Modules/CMakeDetermineCUDACompiler.cmake";
        }
        return 1;
    };
    mm::LlamaCppProvisioner compiler_id_provisioner(cfg, compiler_id_runner);
    const auto compiler_id_failed = compiler_id_provisioner.ensure_runtime();
    const auto compiler_id_check = std::find_if(
        compiler_id_failed.troubleshooting.checks.begin(),
        compiler_id_failed.troubleshooting.checks.end(),
        [](const auto& check) { return check.id == "cuda-cmake-compiler-id"; });
    CHECK(compiler_id_check != compiler_id_failed.troubleshooting.checks.end());
    CHECK(compiler_id_check->status == "fail");
    CHECK(compiler_id_check->blocking);
    CHECK(compiler_id_check->detected.find("sm_120a") != std::string::npos);

    // A baseline sm_120 override can pass a trivial architecture probe but
    // still breaks llama.cpp's architecture-specific Blackwell FP4 kernels.
    mm::LlamaCommandRunner blackwell_runner = runner;
    blackwell_runner.run = [](const std::vector<std::string>&,
                              const std::filesystem::path&,
                              const mm::StreamLineCallback&,
                              const mm::CancelCheckCallback&,
                              std::string* error) {
        if (error) {
            *error =
                "ptxas mmq-instance-mxfp4.compute_120.ptx; "
                "Feature '.kind::mxf4' not supported on .target 'sm_120'; "
                "Instruction 'mma with block scale' not supported on .target 'sm_120'";
        }
        return 1;
    };
    mm::LlamaCppProvisioner blackwell_provisioner(cfg, blackwell_runner);
    const auto blackwell_failed = blackwell_provisioner.ensure_runtime();
    const auto blackwell_check = std::find_if(
        blackwell_failed.troubleshooting.checks.begin(),
        blackwell_failed.troubleshooting.checks.end(),
        [](const auto& check) {
            return check.id == "cuda-blackwell-feature-target";
        });
    CHECK(blackwell_check != blackwell_failed.troubleshooting.checks.end());
    CHECK(blackwell_check->status == "fail");
    CHECK(blackwell_check->blocking);
    CHECK(blackwell_check->required.find("sm_120a") != std::string::npos);
    CHECK(blackwell_check->remediation.find("120a-real") != std::string::npos);

    CHECK(remove_tree(dir));
    return true;
}

bool test_llama_slot_info_backend_and_suspend() {
    auto dir = temp_test_dir("llama-slot");
    mm::SlotManager slots(46170, 46173, 2);
    slots.set_llama_server_path("missing-llama");
    slots.set_kv_cache_dir((dir / "kv").string());

    mm::RuntimeSettings s;
    s.ctx_size = 2048;
    s.parallel = 1;
    const auto id = slots.add_ready_test_slot("m.gguf", "agent-l", s);

    auto info = slots.find_slot(id);
    CHECK(info.has_value());
    CHECK(info->backend == "llama-cpp");
    // Backend survives the SlotInfo JSON round-trip.
    nlohmann::json j = *info;
    CHECK(j.get<mm::SlotInfo>().backend == "llama-cpp");

    // A test slot has no live engine (port 0), so suspend skips the KV save and
    // still transitions to Suspended with an empty cache path.
    auto susp = slots.suspend_slot(id);
    CHECK(susp.status == mm::SlotOperationStatus::Ok);
    CHECK(susp.kv_cache_path.empty());
    auto after = slots.find_slot(id);
    CHECK(after.has_value());
    CHECK(after->state == mm::SlotState::Suspended);

    CHECK(remove_tree(dir));
    return true;
}

bool test_runtime_client_health_empty_body_ok() {
    // OpenAI-compatible runtimes may answer /health with an empty 200 body;
    // health_check must treat that as healthy.
    const uint16_t port = find_free_test_port();
    CHECK(port != 0);
    httplib::Server srv;
    srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
    });
    std::atomic<bool> listen_ok{false};
    std::thread th([&] { listen_ok = srv.listen("127.0.0.1", port); });
    const std::string url = "http://127.0.0.1:" + std::to_string(port);
    bool reachable = false;
    for (int i = 0; i < 50 && !reachable; ++i) {
        mm::HttpClient probe(url);
        if (probe.get("/health").ok()) reachable = true;
        else std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    bool healthy = false;
    if (reachable) {
        mm::RuntimeClient client(url);
        healthy = client.health_check();
    }
    srv.stop();
    th.join();
    CHECK(reachable);
    CHECK(healthy);
    return true;
}

bool test_llama_default_backend_and_slot_sharing() {
    // llama.cpp is the default runtime on this branch.
    mm::AgentConfig fresh;
    CHECK(fresh.inference_backend == "llama-cpp");
    CHECK(mm::is_llama_backend(""));
    CHECK(mm::is_llama_backend("llama-cpp"));
    CHECK(mm::is_llama_backend("llama.cpp"));
    CHECK(mm::is_llama_backend("llama"));
    CHECK(!mm::is_llama_backend("vllm"));
    CHECK(!mm::is_llama_backend("api"));

    // A slot payload without a backend field parses as the default runtime.
    auto parsed = nlohmann::json{{"id", "s1"}}.get<mm::SlotInfo>();
    CHECK(parsed.backend == "llama-cpp");

    // An unspecified agent backend validates as llama.cpp.
    mm::AgentConfig cfg;
    cfg.name = "d";
    cfg.model_path = "/models/m.gguf";
    cfg.inference_backend = "";
    CHECK(mm::validate_agent_config(cfg, nullptr, "", nullptr).ok());

    // Compatible agents share one ready process. A launch-setting mismatch
    // cannot attach; the attempted new process then fails on the fake path.
    auto dir = temp_test_dir("llama-sharing");
    mm::SlotManager slots(46180, 46183, 4);
    slots.set_llama_server_path("missing-llama");
    mm::RuntimeSettings settings;
    settings.ctx_size = 4096;
    const auto llama_id = slots.add_ready_test_slot("m.gguf", "agent-a", settings);
    CHECK(slots.load_model("m.gguf", settings, "agent-b") == llama_id);
    auto llama_info = slots.find_slot(llama_id);
    CHECK(llama_info.has_value());
    CHECK(llama_info->backend == "llama-cpp");
    CHECK(llama_info->assigned_agent == "agent-a");
    CHECK(llama_info->agent_ids.size() == 2);
    CHECK(std::find(llama_info->agent_ids.begin(), llama_info->agent_ids.end(),
                    "agent-b") != llama_info->agent_ids.end());

    auto incompatible = settings;
    incompatible.ctx_size = 8192;
    CHECK(slots.load_model("m.gguf", incompatible, "agent-c").empty());
    llama_info = slots.find_slot(llama_id);
    CHECK(llama_info.has_value());
    CHECK(std::find(llama_info->agent_ids.begin(), llama_info->agent_ids.end(),
                    "agent-c") == llama_info->agent_ids.end());

    const auto first_detach = slots.detach_agent(llama_id, "agent-b");
    CHECK(first_detach.ok());
    CHECK(first_detach.remaining_agents == 1);
    CHECK(!first_detach.unloaded);
    const auto last_detach = slots.detach_agent(llama_id, "agent-a");
    CHECK(last_detach.ok());
    CHECK(last_detach.unloaded);
    CHECK(!slots.find_slot(llama_id).has_value());

    CHECK(remove_tree(dir));
    return true;
}

bool test_llama_restore_attaches_and_cleans_suspended_record() {
    auto dir = temp_test_dir("llama-restore-attach");
    mm::SlotManager slots(46190, 46193, 4);
    mm::RuntimeSettings settings;
    settings.ctx_size = 4096;

    const auto suspended_id =
        slots.add_ready_test_slot("m.gguf", "agent-b", settings);
    CHECK(slots.suspend_slot(suspended_id).ok());
    CHECK(slots.find_slot(suspended_id)->state == mm::SlotState::Suspended);

    const auto ready_id = slots.add_ready_test_slot("m.gguf", "agent-a", settings);
    const auto restored = slots.restore_slot("m.gguf", settings, "", "agent-b");
    CHECK(restored == ready_id);
    CHECK(!slots.find_slot(suspended_id).has_value());
    CHECK(slots.find_slot_by_agent("agent-b") == ready_id);

    const auto ready = slots.find_slot(ready_id);
    CHECK(ready.has_value());
    CHECK(ready->agent_ids.size() == 2);
    CHECK(ready->backend == "llama-cpp");

    CHECK(remove_tree(dir));
    return true;
}

bool test_node_reachability_and_json_compatibility() {
    CHECK(mm::classify_node_reachability(1000, 90999, 90000) ==
          mm::NodeConnectionStatus::Unreachable);
    CHECK(mm::classify_node_reachability(1000, 91000, 90000) ==
          mm::NodeConnectionStatus::Offline);
    CHECK(mm::classify_node_reachability(1000, 1001, 0) ==
          mm::NodeConnectionStatus::Offline);
    CHECK(mm::classify_node_reachability(0, 91000, 90000) ==
          mm::NodeConnectionStatus::Unreachable);

    mm::NodeInfo source;
    source.id = "node-1";
    source.hostname = "workstation";
    source.url = "http://127.0.0.1:7070";
    source.health = mm::NodeHealthStatus::Degraded;
    source.connection_status = mm::NodeConnectionStatus::Offline;
    source.last_seen_ms = 1000;
    source.unreachable_since_ms = 2000;
    source.metrics_sampled_at_ms = 900;
    source.consecutive_failures = 4;

    const nlohmann::json encoded = source;
    const auto decoded = encoded.get<mm::NodeInfo>();
    CHECK(decoded.id == source.id);
    CHECK(decoded.hostname == source.hostname);
    CHECK(decoded.connection_status == mm::NodeConnectionStatus::Offline);
    CHECK(decoded.health == mm::NodeHealthStatus::Degraded);
    CHECK(decoded.last_seen_ms == 1000);
    CHECK(decoded.unreachable_since_ms == 2000);
    CHECK(decoded.metrics_sampled_at_ms == 900);
    CHECK(decoded.consecutive_failures == 4);

    const nlohmann::json legacy = {
        {"id", "legacy-node"},
        {"url", "http://127.0.0.1:7071"},
        {"connected", true},
    };
    const auto legacy_node = legacy.get<mm::NodeInfo>();
    CHECK(legacy_node.connection_status == mm::NodeConnectionStatus::Online);
    return true;
}

bool test_performance_tracker_capacity_aggregation_and_clear() {
    mm::PerformanceTracker tracker(2);

    mm::PerformanceSample first;
    first.request_id = "a";
    first.total_ms = 100;
    first.time_to_first_token_ms = 20;
    first.input_tokens = 3;
    first.output_tokens = 8;
    first.success = true;
    tracker.record(first);

    mm::PerformanceSample second;
    second.request_id = "b";
    second.total_ms = 300;
    second.input_tokens = 2;
    second.success = false;
    second.error = "failed";
    tracker.record(second);

    mm::PerformanceSample third;
    third.request_id = "c";
    third.total_ms = 500;
    third.time_to_first_token_ms = 100;
    third.input_tokens = 7;
    third.output_tokens = 20;
    third.image_count = 2;
    third.decoded_image_bytes = 8192;
    third.vision_routing = true;
    third.projector_basename = "mmproj-test.gguf";
    third.success = true;
    tracker.record(third);

    const auto snapshot = tracker.snapshot(10);
    CHECK(snapshot.at("session").get<bool>());
    CHECK(snapshot.at("samples").size() == 2);
    CHECK(snapshot.at("samples").at(0).at("request_id") == "b");
    CHECK(snapshot.at("samples").at(1).at("request_id") == "c");
    CHECK(snapshot.at("aggregate").at("requests") == 2);
    CHECK(snapshot.at("aggregate").at("successful") == 1);
    CHECK(snapshot.at("aggregate").at("failed") == 1);
    CHECK(snapshot.at("aggregate").at("input_tokens") == 9);
    CHECK(snapshot.at("aggregate").at("total_ms").at("p50") == 400.0);
    CHECK(snapshot.at("samples").at(1).at("output_tokens_per_second") == 50.0);
    CHECK(snapshot.at("samples").at(1).at("image_count") == 2);
    CHECK(snapshot.at("samples").at(1).at("decoded_image_bytes") == 8192);
    CHECK(snapshot.at("samples").at(1).at("vision_routing") == true);
    CHECK(snapshot.at("samples").at(1).at("projector_basename") ==
          "mmproj-test.gguf");

    tracker.clear();
    CHECK(tracker.snapshot(10).at("samples").empty());
    return true;
}

bool test_inference_sizing_estimate() {
    // Unknown model path falls back to a positive estimate (never zero), and the
    // effective server context honors ctx_size * parallel.
    mm::RuntimeSettings s;
    s.ctx_size = 4096;
    s.parallel = 3;
    CHECK(mm::effective_llama_server_ctx_tokens(s) == 12288);
    CHECK(mm::effective_llama_parallel(s) == 3);
    CHECK(mm::estimate_inference_vram_mb("does-not-exist.gguf", s, "") > 0);
    return true;
}

int main(int argc, char** argv) {
    struct TestCase {
        const char* name;
        bool (*fn)();
    };

    const TestCase tests[] = {
        {"non_stream_parser_preserves_text", test_non_stream_parser_preserves_text},
        {"non_stream_parser_extracts_thinking", test_non_stream_parser_extracts_thinking},
        {"stream_tool_call_indices", test_stream_tool_call_indices},
        {"agent_queue_survives_throwing_job", test_agent_queue_survives_throwing_job},
        {"agent_manager_rejects_duplicates_and_defers_cleanup_until_handles_release",
         test_agent_manager_rejects_duplicates_and_defers_cleanup_until_handles_release},
        {"agent_api_settings_round_trip_without_key_persistence",
         test_agent_api_settings_round_trip_without_key_persistence},
        {"served_model_name_legacy_compatibility",
         test_served_model_name_legacy_compatibility},
        {"slot_manager_not_found_statuses", test_slot_manager_not_found_statuses},
        {"slot_lease_blocks_unload_and_suspend_while_busy",
         test_slot_lease_blocks_unload_and_suspend_while_busy},
        {"node_action_progress_json_round_trip",
         test_node_action_progress_json_round_trip},
        {"scheduler_skips_failed_node_current_attempt",
         test_scheduler_skips_failed_node_current_attempt},
        {"scheduler_transfers_existing_relative_models_with_unique_cache_ids",
         test_scheduler_transfers_existing_relative_models_with_unique_cache_ids},
        {"scheduler_eviction_skips_unsuspendable_shared_slot",
         test_scheduler_eviction_skips_unsuspendable_shared_slot},
        {"scheduler_backend_change_releases_local_placement",
         test_scheduler_backend_change_releases_local_placement},
        {"scheduler_reconciles_ready_absent_and_suspended_snapshots",
         test_scheduler_reconciles_ready_absent_and_suspended_snapshots},
        {"control_api_external_token_gate", test_control_api_external_token_gate},
        {"openai_compat_api_listener_and_model_catalog",
         test_openai_compat_api_listener_and_model_catalog},
        {"control_api_agent_api_mode_chat",
         test_control_api_agent_api_mode_chat},
        {"agent_voice_db_and_cache_lifecycle",
         test_agent_voice_db_and_cache_lifecycle},
        {"tts_service_client_fake_sidecar_paths",
         test_tts_service_client_fake_sidecar_paths},
        {"control_api_tts_routes_disabled", test_control_api_tts_routes_disabled},
        {"control_api_curation_routes", test_control_api_curation_routes},
        {"global_memory_origin_tool_and_context_metadata",
         test_global_memory_origin_tool_and_context_metadata},
        {"message_trace_events_round_trip", test_message_trace_events_round_trip},
        {"compaction_followup_trace_provenance_survives",
         test_compaction_followup_trace_provenance_survives},
        {"config_and_url_parsing_edge_cases", test_config_and_url_parsing_edge_cases},
        {"llama_server_args", test_llama_server_args},
        {"vision_config_attachment_and_message_round_trip",
         test_vision_config_attachment_and_message_round_trip},
        {"vision_profile_validation_and_suggestions",
         test_vision_profile_validation_and_suggestions},
        {"vision_slot_projector_isolation_and_json",
         test_vision_slot_projector_isolation_and_json},
        {"llama_model_path_normalization", test_llama_model_path_normalization},
        {"llama_accelerator_detection", test_llama_accelerator_detection},
        {"llama_launch_compatible", test_llama_launch_compatible},
        {"llama_backend_validation_and_gguf_routing",
         test_llama_backend_validation_and_gguf_routing},
        {"llama_install_plan_and_method", test_llama_install_plan_and_method},
        {"llama_provisioner_disabled_and_cancel",
         test_llama_provisioner_disabled_and_cancel},
        {"llama_path_resolution_respects_accelerator",
         test_llama_path_resolution_respects_accelerator},
        {"llama_auto_release_then_source_fallback",
         test_llama_auto_release_then_source_fallback},
        {"llama_update_release_decision",
         test_llama_update_release_decision},
        {"llama_runtime_variant_matrix",
         test_llama_runtime_variant_matrix},
        {"llama_failure_diagnostics_and_recovery",
         test_llama_failure_diagnostics_and_recovery},
        {"llama_nvcc_architecture_preflight_and_diagnostics",
         test_llama_nvcc_architecture_preflight_and_diagnostics},
        {"llama_slot_info_backend_and_suspend",
         test_llama_slot_info_backend_and_suspend},
        {"runtime_client_health_empty_body_ok",
         test_runtime_client_health_empty_body_ok},
        {"llama_default_backend_and_slot_sharing",
         test_llama_default_backend_and_slot_sharing},
        {"llama_restore_attaches_and_cleans_suspended_record",
         test_llama_restore_attaches_and_cleans_suspended_record},
        {"node_reachability_and_json_compatibility",
         test_node_reachability_and_json_compatibility},
        {"performance_tracker_capacity_aggregation_and_clear",
         test_performance_tracker_capacity_aggregation_and_clear},
        {"inference_sizing_estimate", test_inference_sizing_estimate},
    };

    const std::string filter = argc > 1 ? argv[1] : std::string{};
    bool ran_any = false;
    for (const auto& test : tests) {
        if (!filter.empty() && std::string(test.name).find(filter) == std::string::npos) {
            continue;
        }
        ran_any = true;
        if (!test.fn()) {
            std::cerr << "FAILED: " << test.name << "\n";
            return 1;
        }
        std::cout << "PASSED: " << test.name << "\n";
    }
    if (!filter.empty() && !ran_any) {
        std::cerr << "No tests matched filter: " << filter << "\n";
        return 1;
    }
    return 0;
}
