#include "common/inference_response_parser.hpp"
#include "common/agent.hpp"
#include "common/agent_db.hpp"
#include "common/conversation_manager.hpp"
#include "common/http_client.hpp"
#include "common/llama_cpp_client.hpp"
#include "common/memory_manager.hpp"
#include "common/trace_provenance.hpp"
#include "common/tool_executor.hpp"
#include "common/util.hpp"
#include "control/agent_manager.hpp"
#include "control/agent_queue.hpp"
#include "control/agent_scheduler.hpp"
#include "control/control_api_server.hpp"
#include "control/model_distributor.hpp"
#include "control/model_router.hpp"
#include "control/node_registry.hpp"
#include "node/llama_server_process.hpp"
#include "node/slot_manager.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

class FixedSummaryLlamaClient : public mm::LlamaCppClient {
public:
    FixedSummaryLlamaClient() : mm::LlamaCppClient("http://127.0.0.1:1") {}

    mm::Message complete(const mm::InferenceRequest&) override {
        mm::Message summary;
        summary.role = mm::MessageRole::Assistant;
        summary.content = "Summary: original source conversation kept the launch thermal review context.";
        summary.token_count = 12;
        summary.timestamp_ms = mm::util::now_ms();
        return summary;
    }
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

bool remove_tree(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    if (ec) {
        std::cerr << "cleanup failed for " << dir << ": " << ec.message() << "\n";
        return false;
    }
    return true;
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

bool test_slot_manager_not_found_statuses() {
    auto dir = temp_test_dir("slots");
    mm::SlotManager slots("missing-llama-server", 46100, 46101, 1, dir.string());

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
    mm::SlotManager slots("missing-llama-server", 46110, 46111, 1, dir.string());
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

bool has_arg_pair(const std::vector<std::string>& args,
                  const std::string& flag,
                  const std::string& value) {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == flag && args[i + 1] == value) return true;
    }
    return false;
}

bool has_arg(const std::vector<std::string>& args, const std::string& flag) {
    for (const auto& arg : args) {
        if (arg == flag) return true;
    }
    return false;
}

bool test_llama_settings_throughput_fields_round_trip() {
    mm::LlamaSettings settings;
    settings.ctx_size = 8192;
    settings.n_threads = 8;
    settings.n_threads_http = 4;
    settings.parallel = 6;
    settings.batch_size = 1024;
    settings.ubatch_size = 256;

    nlohmann::json serialized = settings;
    CHECK(serialized["n_threads_http"] == 4);
    CHECK(serialized["parallel"] == 6);
    CHECK(serialized["batch_size"] == 1024);
    CHECK(serialized["ubatch_size"] == 256);

    auto parsed = serialized.get<mm::LlamaSettings>();
    CHECK(parsed.n_threads_http == 4);
    CHECK(parsed.parallel == 6);
    CHECK(parsed.batch_size == 1024);
    CHECK(parsed.ubatch_size == 256);

    auto dir = temp_test_dir("llama-settings-throughput");
    std::filesystem::create_directories(dir);
    {
        mm::AgentDB db("agent-a", dir.string());
        mm::AgentConfig cfg;
        cfg.id = "agent-a";
        cfg.name = "Agent A";
        cfg.model_path = "model.gguf";
        cfg.llama_settings = settings;
        db.save_config(cfg);

        auto loaded = db.load_config();
        CHECK(loaded.llama_settings.n_threads_http == 4);
        CHECK(loaded.llama_settings.parallel == 6);
        CHECK(loaded.llama_settings.batch_size == 1024);
        CHECK(loaded.llama_settings.ubatch_size == 256);
    }

    CHECK(remove_tree(dir));
    return true;
}

bool test_llama_server_args_throughput_tunables() {
    mm::LlamaSettings defaults;
    auto default_args = mm::build_llama_server_args_for_test("model.gguf", defaults, 48123);
    CHECK(has_arg_pair(default_args, "--ctx-size", "4096"));
    CHECK(!has_arg(default_args, "--parallel"));
    CHECK(!has_arg(default_args, "--batch-size"));
    CHECK(!has_arg(default_args, "--ubatch"));
    CHECK(!has_arg(default_args, "--threads-http"));

    mm::LlamaSettings tuned;
    tuned.ctx_size = 2048;
    tuned.n_threads = 8;
    tuned.n_threads_http = 4;
    tuned.parallel = 4;
    tuned.batch_size = 1024;
    tuned.ubatch_size = 256;
    auto tuned_args = mm::build_llama_server_args_for_test("model.gguf", tuned, 48124);
    CHECK(has_arg_pair(tuned_args, "--ctx-size", "8192"));
    CHECK(has_arg_pair(tuned_args, "--threads", "8"));
    CHECK(has_arg_pair(tuned_args, "--threads-http", "4"));
    CHECK(has_arg_pair(tuned_args, "--parallel", "4"));
    CHECK(has_arg_pair(tuned_args, "--batch-size", "1024"));
    CHECK(has_arg_pair(tuned_args, "--ubatch", "256"));

    mm::LlamaSettings overridden;
    overridden.ctx_size = 1024;
    overridden.parallel = 1;
    overridden.batch_size = 512;
    overridden.extra_args = {"--parallel=8", "--batch-size", "2048"};
    auto override_args = mm::build_llama_server_args_for_test("model.gguf", overridden, 48125);
    CHECK(has_arg_pair(override_args, "--ctx-size", "8192"));
    CHECK(!has_arg_pair(override_args, "--parallel", "1"));
    CHECK(!has_arg_pair(override_args, "--batch-size", "512"));
    CHECK(has_arg(override_args, "--parallel=8"));
    CHECK(has_arg_pair(override_args, "--batch-size", "2048"));

    mm::LlamaSettings explicit_ctx;
    explicit_ctx.ctx_size = 4096;
    explicit_ctx.parallel = 8;
    explicit_ctx.extra_args = {"--ctx-size=32768"};
    auto explicit_ctx_args = mm::build_llama_server_args_for_test("model.gguf", explicit_ctx, 48126);
    CHECK(!has_arg_pair(explicit_ctx_args, "--ctx-size", "32768"));
    CHECK(has_arg(explicit_ctx_args, "--ctx-size=32768"));

    return true;
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
        mm::ModelDistributor distributor(registry, (dir / "models").string());
        mm::AgentScheduler scheduler(registry, distributor, (dir / "models").string());
        mm::ModelRouter router(scheduler);
        mm::ControlApiServer api(
            agents, queue, registry, router, scheduler,
            (dir / "models").string(), "control-secret");

        const uint16_t port = 49287;
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

        auto expect_error = [&](const mm::HttpResponse& resp,
                                int expected_status,
                                const std::string& expected_text) {
            RECORD(resp.status == expected_status);
            RECORD(resp.body.find(expected_text) != std::string::npos);
        };

        auto missing = client.get("/v1/nodes");
        expect_error(missing, 401, "missing bearer token");

        client.set_bearer_token("wrong-secret");
        auto invalid = client.get("/v1/nodes");
        expect_error(invalid, 403, "invalid bearer token");

        registry.add_node("http://127.0.0.1:1", "node-secret", "test", false);

        client.set_bearer_token("node-secret");
        auto node_token_on_external = client.get("/v1/nodes");
        expect_error(node_token_on_external, 403, "invalid bearer token");

        client.set_bearer_token("control-secret");
        auto valid = client.get("/v1/nodes");
        RECORD(valid.status == 200);

        mm::HttpClient external_on_internal(base_url);
        external_on_internal.set_bearer_token("control-secret");
        auto external_token_internal = external_on_internal.get("/api/control/models");
        expect_error(external_token_internal, 401, "invalid node api key");

        mm::HttpClient node_client(base_url);
        node_client.set_bearer_token("node-secret");
        auto node_internal = node_client.get("/api/control/models");
        RECORD(node_internal.status == 200);

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
            attempt.ok = stream_client.stream_post(
                "/v1/agents/agent-a/chat",
                body,
                [&](const std::string& event) {
                    attempt.events.push_back(event);
                    return true;
                },
                &attempt.status,
                &attempt.body);
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

        mm::HttpClient missing_mutator(base_url);
        auto missing_create_conversation = missing_mutator.post(
            "/v1/agents/agent-a/conversations",
            nlohmann::json{{"title", "Blocked"}, {"set_active", true}});
        expect_error(missing_create_conversation, 401, "missing bearer token");

        mm::HttpClient node_mutator(base_url);
        node_mutator.set_bearer_token("node-secret");
        auto node_create_conversation = node_mutator.post(
            "/v1/agents/agent-a/conversations",
            nlohmann::json{{"title", "Blocked"}, {"set_active", true}});
        expect_error(node_create_conversation, 403, "invalid bearer token");

        auto create_conversation = client.post(
            "/v1/agents/agent-a/conversations",
            nlohmann::json{{"title", "Original title"}, {"set_active", true}});
        RECORD(create_conversation.status == 201);
        auto conversation_body = nlohmann::json::parse(create_conversation.body);
        const std::string conversation_id =
            conversation_body["conversation"]["id"].get<std::string>();

        mm::HttpClient missing_put(base_url);
        auto missing_rename = missing_put.put(
            "/v1/agents/agent-a/conversations/" + conversation_id,
            nlohmann::json{{"title", "Blocked rename"}});
        expect_error(missing_rename, 401, "missing bearer token");

        auto rename = client.put(
            "/v1/agents/agent-a/conversations/" + conversation_id,
            nlohmann::json{{"title", "Renamed by authorized client"}});
        RECORD(rename.status == 200);
        auto renamed_body = nlohmann::json::parse(rename.body);
        RECORD(renamed_body["title"] == "Renamed by authorized client");

        mm::HttpClient invalid_memory_client(base_url);
        invalid_memory_client.set_bearer_token("wrong-secret");
        auto invalid_create_memory = invalid_memory_client.post(
            "/v1/agents/agent-a/memories",
            nlohmann::json{{"content", "Blocked memory"}, {"source_conv_id", conversation_id}});
        expect_error(invalid_create_memory, 403, "invalid bearer token");

        auto create_memory = client.post(
            "/v1/agents/agent-a/memories",
            nlohmann::json{{"content", "Authorized memory"}, {"source_conv_id", conversation_id}});
        RECORD(create_memory.status == 201);
        auto memory_body = nlohmann::json::parse(create_memory.body);
        const std::string memory_id = memory_body["id"].get<std::string>();

        mm::HttpClient missing_delete(base_url);
        auto missing_delete_memory = missing_delete.del(
            "/v1/agents/agent-a/memories/" + memory_id);
        expect_error(missing_delete_memory, 401, "missing bearer token");

        auto delete_memory = client.del("/v1/agents/agent-a/memories/" + memory_id);
        RECORD(delete_memory.status == 200);

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
    mm::ModelDistributor distributor(registry, (dir / "models").string());
    mm::AgentScheduler scheduler(registry, distributor, (dir / "models").string());
    mm::ModelRouter router(scheduler);
    mm::ControlApiServer api(
        agents, queue, registry, router, scheduler,
        (dir / "models").string());

    const uint16_t port = 49288;
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
        cfg.llama_settings.ctx_size = 128;

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

        FixedSummaryLlamaClient llama;
        mm::ConversationManager conv_mgr(db, llama);
        const mm::ConvId continued_conv_id = conv_mgr.force_compact(source_conv_id, cfg);
        CHECK(!continued_conv_id.empty());
        CHECK(continued_conv_id != source_conv_id);

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

} // namespace

int main() {
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
        {"slot_manager_not_found_statuses", test_slot_manager_not_found_statuses},
        {"slot_lease_blocks_unload_and_suspend_while_busy",
         test_slot_lease_blocks_unload_and_suspend_while_busy},
        {"llama_settings_throughput_fields_round_trip",
         test_llama_settings_throughput_fields_round_trip},
        {"llama_server_args_throughput_tunables",
         test_llama_server_args_throughput_tunables},
        {"control_api_external_token_gate", test_control_api_external_token_gate},
        {"control_api_curation_routes", test_control_api_curation_routes},
        {"global_memory_origin_tool_and_context_metadata",
         test_global_memory_origin_tool_and_context_metadata},
        {"message_trace_events_round_trip", test_message_trace_events_round_trip},
        {"compaction_followup_trace_provenance_survives",
         test_compaction_followup_trace_provenance_survives},
    };

    for (const auto& test : tests) {
        if (!test.fn()) {
            std::cerr << "FAILED: " << test.name << "\n";
            return 1;
        }
        std::cout << "PASSED: " << test.name << "\n";
    }
    return 0;
}
