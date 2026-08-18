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
#include "common/engine_capabilities.hpp"
#include "common/engine_client.hpp"
#include "control/agent_scheduler.hpp"
#include "control/engine_config_store.hpp"
#include "node/node_state.hpp"
#include "node/engine_manager.hpp"
#include "node/engine_provisioner.hpp"
#include "node/node_ui.hpp"
#include "node/engine_descriptor.hpp"
#include "control/control_api_server.hpp"
#include "common/pairing.hpp"
#include "control/model_registry.hpp"
#include "control/route_scope.hpp"
#include "control/node_registry.hpp"
#include "control/performance_tracker.hpp"
#include "control/tts_service_client.hpp"
#include "common/gguf_metadata.hpp"
#include "common/inference_sizing.hpp"
#include "control/agent_config_validator.hpp"
// The requantization gate reaches into the ENGINE's types: arch_hash is
// computed there, and the KV store is what enforces the invalidation.
#include "soma/arch_ir.hpp"
#include "soma/kv_cache.hpp"
#include "soma/kv_checkpoint.hpp"
#include "node/runtime_process.hpp"
#include "node/engine_supervisor.hpp"
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

/// The Windows-loopback transport flake, and the ONE budget for it.
///
/// mm::HttpClient opens a fresh connection per request, and rapid sequential
/// connect/close cycles on Windows loopback occasionally fail before the server
/// is reached at all: `status` stays 0 and no response is parsed. It is a real
/// property of the environment, not of the code under test, so every request in
/// this file retries it.
///
/// Stated once because it used to be stated twice, with different numbers:
/// `with_retry` allowed 8 flat 50 ms attempts (400 ms) and the SSE helper allowed
/// 3 (150 ms). The SSE path is the one that carries the auth negatives, so the
/// thinner budget sat under the assertions where a transport failure is hardest
/// to tell from a wrong verdict — `status == 403` fails identically whether
/// authorization returned 200 or the request never arrived. That is exactly how
/// it presented: one intermittent failure, at the auth assertion, in the suite's
/// most alarming test.
///
/// Backoff rather than a flat interval: the failure is a socket in TIME_WAIT or a
/// listener mid-accept, and both clear on a timescale that a fixed short retry
/// re-hits every time.
constexpr int kTransportRetries = 6;
inline void transport_backoff(int attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50 * (attempt + 1)));
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
/// A port no OTHER TEST PROCESS will also pick.
///
/// The probe below binds, closes, and returns the number, so the caller binds it
/// a moment later — a window in which anything may take it. Within one process
/// the monotonic counter makes that harmless. ACROSS processes it was not: every
/// process started at 42800 and marched upward in lockstep, so two of them
/// racing picked the same ports in the same order.
///
/// The failure that produces is worse than a refused connection. The loser's
/// `listen()` fails, the winner's server answers the readiness poll, and the test
/// proceeds to assert against A DIFFERENT PROCESS'S SERVER — which replies with
/// plausible, wrong statuses. That is what D1 had been reporting as an
/// intermittent transport flake: measured here at roughly 80% failure with eight
/// concurrent copies, and the failing checks were content assertions, not
/// connection errors.
///
/// An atomic directory per port is the cross-process lease. The process keeps
/// every lease until exit, so another copy of this test cannot select the same
/// port during the probe-to-listen gap. A hard-killed process can leave stale
/// leases, but the 22k-port range makes those skips harmless and normal teardown
/// removes them. Unrelated processes do not honor the lease, so callers still
/// assert that THEIR listen succeeded; see RECORD(listen_ok) in each test.
struct TestPortLeases {
    std::filesystem::path root =
        std::filesystem::temp_directory_path() / "mantic-mind-test-port-leases";
    std::vector<std::filesystem::path> held;

    TestPortLeases() {
        std::error_code ec;
        std::filesystem::create_directories(root, ec);
    }

    ~TestPortLeases() {
        std::error_code ec;
        for (const auto& lease : held) {
            std::filesystem::remove(lease, ec);
            ec.clear();
        }
        std::filesystem::remove(root, ec); // succeeds only for the last process
    }
};

uint16_t find_free_test_port() {
    static uint16_t next_candidate = 42800;
    static TestPortLeases leases;
    for (int p = next_candidate; p < 65000; ++p) {
        std::error_code lease_ec;
        const auto lease = leases.root / ("port-" + std::to_string(p));
        if (!std::filesystem::create_directory(lease, lease_ec)) continue;

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(static_cast<uint16_t>(p));
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        bool ok = false;
#ifdef _WIN32
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (s != INVALID_SOCKET) {
            ok = bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
            closesocket(s);
        }
#else
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s >= 0) {
            int opt = 1;
            setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            ok = ::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
            ::close(s);
        }
#endif
        if (ok) {
            leases.held.push_back(lease);
            next_candidate = static_cast<uint16_t>(p + 1);
            return static_cast<uint16_t>(p);
        }
        std::filesystem::remove(lease, lease_ec);
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

bool test_engine_supervisor_not_found_statuses() {
    auto dir = temp_test_dir("slots");
    mm::EngineSupervisor slots(46100, 46101, 1);

    auto unload = slots.unload("missing-slot");
    CHECK(unload.status == mm::EngineOpStatus::NotFound);

    auto suspend = slots.suspend("missing-slot");
    CHECK(suspend.status == mm::EngineOpStatus::NotFound);

    auto unload_all = slots.unload_all(false);
    CHECK(unload_all.status == mm::EngineOpStatus::Ok);

    CHECK(remove_tree(dir));
    return true;
}

bool test_slot_lease_blocks_unload_and_suspend_while_busy() {
    auto dir = temp_test_dir("lease-busy");
    mm::EngineSupervisor slots(46110, 46111, 1);
    const auto slot_id = slots.add_ready_test_engine("llama-cpp", "test-model.gguf", "agent-a");

    {
        auto inference_lease = slots.acquire(slot_id);
        CHECK(static_cast<bool>(inference_lease));

        auto unload = slots.unload(slot_id);
        CHECK(unload.status == mm::EngineOpStatus::Busy);

        auto suspend = slots.suspend(slot_id);
        CHECK(suspend.status == mm::EngineOpStatus::Busy);
    }

    auto unload_after_release = slots.unload(slot_id);
    CHECK(unload_after_release.status == mm::EngineOpStatus::Ok);

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

// ── Cluster engine configuration ─────────────────────────────────────────────

bool test_engine_config_validation_and_round_trip() {
    // A config is only meaningful with a primary.
    mm::ClusterEngineConfig empty;
    std::string err;
    CHECK(!mm::validate_engine_config(empty, {}, err));
    CHECK(err.find("primary_engine") != std::string::npos);

    // Named but unspecified: the failure that would otherwise surface as a node
    // provisioning defaults nobody chose.
    mm::ClusterEngineConfig no_spec;
    no_spec.primary_engine = "soma";
    CHECK(!mm::validate_engine_config(no_spec, {}, err));
    CHECK(err.find("no spec") != std::string::npos);

    auto cfg = mm::EngineConfigStore::default_for("soma");
    CHECK(cfg.primary_engine == "soma");
    CHECK(cfg.backup_engine == "llama-cpp");
    CHECK(mm::validate_engine_config(cfg, {}, err));

    // Backup == primary is a configuration mistake with a silent failure mode
    // (one engine, reported as two), so it is refused rather than deduplicated.
    auto same = cfg;
    same.backup_engine = "soma";
    CHECK(!mm::validate_engine_config(same, {}, err));
    CHECK(err.find("differ") != std::string::npos);

    // An engine no node can run is refused at the write, not discovered per node.
    CHECK(!mm::validate_engine_config(cfg, {"soma"}, err));
    CHECK(err.find("unknown engine") != std::string::npos);
    CHECK(mm::validate_engine_config(cfg, {"soma", "llama-cpp"}, err));

    // THE claim this whole feature rests on: an empty backup is a real
    // configuration, and required_engines() is what makes it mean "never
    // provision llama.cpp" rather than "provision it and don't use it".
    auto solo = mm::EngineConfigStore::default_for("soma");
    solo.backup_engine.clear();
    solo.engines.erase(std::remove_if(solo.engines.begin(), solo.engines.end(),
                                      [](const mm::EngineSpec& s) {
                                          return s.engine_id == "llama-cpp";
                                      }),
                       solo.engines.end());
    CHECK(mm::validate_engine_config(solo, {}, err));
    CHECK(solo.required_engines().size() == 1);
    CHECK(solo.required_engines()[0] == "soma");

    // llama.cpp as primary gets no backup, rather than itself as one.
    auto llama_primary = mm::EngineConfigStore::default_for("llama-cpp");
    CHECK(llama_primary.backup_engine.empty());
    CHECK(mm::validate_engine_config(llama_primary, {}, err));

    cfg.version = 7;
    cfg.updated_by = "tui";
    const nlohmann::json j = cfg;
    const auto parsed = j.get<mm::ClusterEngineConfig>();
    CHECK(parsed.version == 7);
    CHECK(parsed.primary_engine == "soma");
    CHECK(parsed.backup_engine == "llama-cpp");
    CHECK(parsed.updated_by == "tui");
    CHECK(parsed.engines.size() == cfg.engines.size());
    return true;
}

bool test_engine_config_rejects_per_machine_keys() {
    // The invariant, asserted rather than only documented. A cluster config that
    // carried an accelerator would make every heterogeneous cluster permanently
    // non-conforming, and SILENTLY ignoring the key is the worse failure: the
    // write is accepted and the operator believes the cluster was told.
    for (const auto& key : mm::forbidden_config_keys()) {
        nlohmann::json top{{"primary_engine", "soma"}};
        top[key] = "cuda";
        bool threw = false;
        try {
            (void)top.get<mm::ClusterEngineConfig>();
        } catch (const std::exception& e) {
            threw = true;
            // The message must name the field; "invalid config" would send the
            // operator looking in the wrong place.
            CHECK(std::string(e.what()).find(key) != std::string::npos);
        }
        CHECK(threw);

        // And nested inside a spec, which is where an operator would most
        // naturally try to put it.
        nlohmann::json spec{{"engine_id", "llama-cpp"}};
        spec[key] = "cuda";
        nlohmann::json nested{{"primary_engine", "llama-cpp"},
                              {"engines", nlohmann::json::array({spec})}};
        bool nested_threw = false;
        try {
            (void)nested.get<mm::ClusterEngineConfig>();
        } catch (const std::exception&) {
            nested_threw = true;
        }
        CHECK(nested_threw);
    }

    // A well-formed config still parses — the guard must not be a blanket
    // refusal of anything it does not recognise.
    const nlohmann::json ok{{"primary_engine", "soma"},
                            {"engines", nlohmann::json::array(
                                            {nlohmann::json{{"engine_id", "soma"}}})}};
    const auto parsed = ok.get<mm::ClusterEngineConfig>();
    CHECK(parsed.primary_engine == "soma");
    return true;
}

bool test_engine_artifact_fingerprint_is_exact() {
    mm::EngineArtifact base;
    base.engine_id = "llama-cpp";
    base.version = "b4321";
    base.platform = "linux";
    base.arch = "x86_64";
    base.variant = "cuda-12";
    CHECK(base.valid());

    const std::string fp = base.fingerprint();

    // Every identity field must move the fingerprint. A field that did not
    // would let a share match on a binary that cannot run: an x86_64 build on
    // aarch64, or cuda-12 on a cuda-13 host.
    const std::vector<std::function<void(mm::EngineArtifact&)>> mutations = {
        [](mm::EngineArtifact& a) { a.engine_id = "soma"; },
        [](mm::EngineArtifact& a) { a.version = "b4322"; },
        [](mm::EngineArtifact& a) { a.platform = "windows"; },
        [](mm::EngineArtifact& a) { a.arch = "aarch64"; },
        [](mm::EngineArtifact& a) { a.variant = "cuda-13"; },
    };
    for (const auto& mutate : mutations) {
        auto other = base;
        mutate(other);
        CHECK(other.fingerprint() != fp);
    }

    // sha256 is deliberately NOT part of it: two independent builds of the same
    // source at the same version are the same NEED even when their bytes differ.
    auto rehashed = base;
    rehashed.sha256 = "deadbeef";
    CHECK(rehashed.fingerprint() == fp);

    mm::EngineArtifact parsed;
    CHECK(mm::parse_engine_fingerprint(fp, parsed));
    CHECK(parsed.engine_id == base.engine_id);
    CHECK(parsed.version == base.version);
    CHECK(parsed.platform == base.platform);
    CHECK(parsed.arch == base.arch);
    CHECK(parsed.variant == base.variant);

    // An engine with no build variants keeps its empty slot, so the round trip
    // survives and "soma|1|linux|x86_64|" cannot be confused with a 4-field form.
    mm::EngineArtifact no_variant;
    no_variant.engine_id = "soma";
    no_variant.version = "1";
    no_variant.platform = "linux";
    no_variant.arch = "x86_64";
    mm::EngineArtifact rt;
    CHECK(mm::parse_engine_fingerprint(no_variant.fingerprint(), rt));
    CHECK(rt.variant.empty());
    CHECK(rt.engine_id == "soma");

    mm::EngineArtifact junk;
    CHECK(!mm::parse_engine_fingerprint("llama-cpp|b1|linux", junk));
    CHECK(!mm::parse_engine_fingerprint("|||| ", junk));
    return true;
}

bool test_engine_config_store_persists_and_bumps_version() {
    auto dir = temp_test_dir("engine-config-store");
    std::filesystem::create_directories(dir);

    mm::EngineConfigStore store(dir.string());
    std::string err;
    CHECK(store.load(err));
    // Absence of the file IS the unconfigured signal — the load succeeds and
    // reports nothing configured, because that is what forces first-run setup.
    CHECK(err.empty());
    CHECK(!store.configured());
    CHECK(store.version() == 0);

    // A rejected write must leave nothing behind: nodes that started chasing a
    // config the store refused would be converging on something that does not
    // exist.
    mm::ClusterEngineConfig bad;
    CHECK(!store.save(bad, {}, "test", err));
    CHECK(!store.configured());
    CHECK(!std::filesystem::exists(dir / "engine_config.json"));

    int callbacks = 0;
    std::uint32_t seen_version = 0;
    store.set_change_callback([&](const mm::ClusterEngineConfig& c) {
        ++callbacks;
        seen_version = c.version;
    });

    auto cfg = mm::EngineConfigStore::default_for("soma");
    // The caller's version is IGNORED — a client echoing a stale one back could
    // otherwise walk the cluster backwards, and every node compares on it.
    cfg.version = 999;
    CHECK(store.save(cfg, {}, "test", err));
    CHECK(store.configured());
    CHECK(store.version() == 1);
    CHECK(callbacks == 1);
    CHECK(seen_version == 1);
    CHECK(store.get().updated_by == "test");
    CHECK(store.get().updated_at_ms > 0);

    cfg.share_builds = false;
    CHECK(store.save(cfg, {}, "test2", err));
    CHECK(store.version() == 2);
    CHECK(callbacks == 2);

    // Reopened from disk: the version survives, so a restart does not re-push
    // v1 to a cluster already at v2.
    mm::EngineConfigStore reopened(dir.string());
    CHECK(reopened.load(err));
    CHECK(reopened.configured());
    CHECK(reopened.version() == 2);
    CHECK(reopened.get().primary_engine == "soma");
    CHECK(!reopened.get().share_builds);

    // A corrupt file is reported, not silently replaced with a default the
    // operator never chose — that would re-run setup on a configured cluster.
    {
        std::ofstream out(dir / "engine_config.json", std::ios::trunc);
        out << "{ not json";
    }
    mm::EngineConfigStore broken(dir.string());
    std::string load_err;
    CHECK(!broken.load(load_err));
    CHECK(!load_err.empty());
    CHECK(!broken.configured());

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return true;
}

bool test_placement_refused_until_engine_config_exists() {
    auto dir = temp_test_dir("engine-gate");
    std::filesystem::create_directories(dir / "models");

    mm::NodeRegistry registry(dir.string());
    mm::AgentScheduler scheduler(registry, (dir / "models").string());
    mm::EngineConfigStore store(dir.string());
    std::string err;
    CHECK(store.load(err));
    scheduler.set_engine_config_gate([&store]() { return store.configured(); });

    mm::AgentConfig cfg;
    cfg.id = "agent-gate";
    cfg.name = "Gate";
    cfg.model_path = "Qwen/Qwen3-8B";

    // No configuration: refused, and the message names the fix. "No available
    // nodes" would send an operator to inspect nodes that are all healthy.
    CHECK(!scheduler.ensure_agent_running(cfg).has_value());
    const std::string refusal = scheduler.last_error();
    CHECK(refusal.find("engine configuration") != std::string::npos);
    CHECK(refusal.find("/v1/cluster/engines/config") != std::string::npos);
    // Structurally, not by prose. This assertion used to exist only as the
    // string match above, which is the defect D64 names: reword the message and
    // the test still passes while every client breaks.
    CHECK(scheduler.last_failure() == mm::PlacementFailure::EngineConfigMissing);
    // An operator must act. Retrying this forever is the wrong behaviour and a
    // client can now know that without reading English.
    CHECK(!mm::placement_failure_retryable(scheduler.last_failure()));

    // Configured: the gate opens. It still finds no node — there are none
    // registered — but the refusal is no longer the configuration one, which is
    // the distinction being asserted.
    CHECK(store.save(mm::EngineConfigStore::default_for("llama-cpp"), {}, "test", err));
    CHECK(!scheduler.ensure_agent_running(cfg).has_value());
    CHECK(scheduler.last_error().find("engine configuration required") == std::string::npos);
    // The SAME failure the old code called "no capacity: no connected node
    // could load this model" — which conflated an unconfigured cluster with a
    // full one. Nothing is eligible here; nothing is full.
    CHECK(scheduler.last_failure() == mm::PlacementFailure::NoEligibleNode);
    CHECK(mm::placement_failure_retryable(scheduler.last_failure()));

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return true;
}

bool test_conformance_gates_placement_candidates() {
    auto dir = temp_test_dir("engine-conformance");
    std::filesystem::create_directories(dir);
    mm::NodeRegistry registry(dir.string());

    const auto id = registry.add_node("http://127.0.0.1:1", "secret", "linux", false, "n1");

    // With no engine-config provider nothing is MANAGING conformance, so the
    // gate stays open. Closing it here would mean a registry used without a
    // config store silently places nowhere, and the symptom would point at
    // healthy nodes.
    CHECK(registry.available_nodes().empty()); // not connected yet
    CHECK(registry.conforming_nodes().empty());

    mm::ClusterEngineConfig cfg = mm::EngineConfigStore::default_for("soma");
    cfg.version = 3;
    registry.set_engine_config_provider(
        [&cfg]() -> std::optional<mm::ClusterEngineConfig> { return cfg; });

    // conformance_permits_placement is the single predicate both filters use,
    // so assert its contract directly across every state.
    mm::EngineConformance c;
    for (const auto st : {mm::EngineConformanceState::Unconfigured,
                          mm::EngineConformanceState::Converging,
                          mm::EngineConformanceState::Drifted,
                          mm::EngineConformanceState::Failed}) {
        c.state = st;
        CHECK(!mm::conformance_permits_placement(c));
    }
    c.state = mm::EngineConformanceState::Conforming;
    CHECK(mm::conformance_permits_placement(c));

    // Round-trip through the wire form the node actually sends.
    c.config_version = 3;
    c.detail = "running 2 configured engine(s)";
    const auto parsed = nlohmann::json(c).get<mm::EngineConformance>();
    CHECK(parsed.state == mm::EngineConformanceState::Conforming);
    CHECK(parsed.config_version == 3);
    CHECK(parsed.detail == c.detail);

    // An unrecognised state leaves the default, which STOPS placement. A node
    // speaking a vocabulary this build does not know is the node not to
    // schedule onto.
    const auto unknown =
        nlohmann::json{{"state", "quantum"}}.get<mm::EngineConformance>();
    CHECK(unknown.state == mm::EngineConformanceState::Unconfigured);
    CHECK(!mm::conformance_permits_placement(unknown));

    (void)id;
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return true;
}

bool test_engine_digest_and_package_grants_are_one_shot() {
    mm::NodeState state;

    // Unbrokered transfers are refused: an empty expected digest is what the
    // receive route turns into a 409.
    CHECK(state.take_expected_engine_digest("llama-cpp|b1|linux|x86_64|cpu").empty());

    state.expect_engine_digest("llama-cpp|b1|linux|x86_64|cpu", "abc123");
    CHECK(state.take_expected_engine_digest("llama-cpp|b1|linux|x86_64|cpu") == "abc123");
    // Consumed. A grant that outlived its transfer would authorize a later
    // unbrokered push of the same fingerprint — exactly what it exists to
    // refuse.
    CHECK(state.take_expected_engine_digest("llama-cpp|b1|linux|x86_64|cpu").empty());

    state.set_prepared_engine_package("tok-1", "/tmp/pkg.tar.gz");
    CHECK(state.take_prepared_engine_package("tok-1") == "/tmp/pkg.tar.gz");
    CHECK(state.take_prepared_engine_package("tok-1").empty());
    CHECK(state.take_prepared_engine_package("never-issued").empty());
    return true;
}

bool test_admission_variant_is_the_collision_key() {
    // The identity rule, asserted without converting anything. Two refs that
    // denote one model must produce ONE container directory — that is what makes
    // the variant usable as an in-flight collision key, and it is why the key is
    // derived by the same function that picks the write path rather than by a
    // second copy of the rule.
    mm::AdmissionTools tools;
    tools.quant = "q4_g";
    tools.expert_down = "q6_g";
    tools.group = 128;

    const auto from_repo = mm::admission_variant("Qwen/Qwen3-30B-A3B", true, tools);
    const auto from_dir =
        mm::admission_variant("/models/Qwen3-30B-A3B", false, tools);
    CHECK(from_repo == from_dir);
    CHECK(from_repo == "Qwen3-30B-A3B-q4_g-q6_g-g128");

    // A revision suffix names the same model directory. (Whether it SHOULD is a
    // separate question — see the note in the roadmap — but the two halves of
    // the system must at least agree, which is what this pins.)
    CHECK(mm::admission_variant("Qwen/Qwen3-30B-A3B@main", true, tools) == from_repo);

    // Quantization is part of the identity: the same weights at two quants are
    // two containers, and must NOT collide.
    auto other = tools;
    other.expert_down = "q4_g";
    CHECK(mm::admission_variant("Qwen/Qwen3-30B-A3B", true, other) != from_repo);
    auto grouped = tools;
    grouped.group = 64;
    CHECK(mm::admission_variant("Qwen/Qwen3-30B-A3B", true, grouped) != from_repo);

    // The fetch destination shares the name half, so sources/<name> and
    // containers/<name>-... cannot disagree about which model a ref denotes.
    CHECK(mm::admission_source_name("Qwen/Qwen3-30B-A3B", true) == "Qwen3-30B-A3B");
    CHECK(mm::admission_source_name("/models/Qwen3-30B-A3B", false) == "Qwen3-30B-A3B");
    CHECK(mm::admission_source_name("Qwen/Qwen3-30B-A3B@main", true) == "Qwen3-30B-A3B");
    return true;
}

bool test_concurrent_admission_of_one_model_joins_not_duplicates() {
    // The case nothing had ever run. Both existing admission tests drive exactly
    // one operation, so two in flight — the shape that corrupts a container —
    // had never happened in the suite.
    const char* soma_path = std::getenv("MM_TEST_SOMA_PATH");
    const char* container_dir = std::getenv("MM_TEST_CONTAINER_DIR");
    if (soma_path == nullptr || container_dir == nullptr) {
        std::cout << "  (skipped: MM_TEST_SOMA_PATH / MM_TEST_CONTAINER_DIR unset)\n";
        return true;
    }

    auto dir = temp_test_dir("admission-concurrency");
    mm::ControlModelRegistry reg;
    std::string err;
    CHECK(reg.open(dir.string(), err));

    mm::AdmissionTools tools;
    tools.soma_path = soma_path;
    tools.containers_dir = (dir / "containers").string();
    tools.sources_dir = (dir / "sources").string();
    reg.set_tools(tools);

    // Default is 1, and 0 must not be honoured — a cap of zero parks every
    // admission on a gate nothing can open, which reads as "pause" and behaves
    // as "hang forever".
    CHECK(reg.max_concurrent_admissions() == 1);
    reg.set_max_concurrent_admissions(0);
    CHECK(reg.max_concurrent_admissions() == 1);
    reg.set_max_concurrent_admissions(2);
    CHECK(reg.max_concurrent_admissions() == 2);
    reg.set_max_concurrent_admissions(1);

    // Two admissions of ONE source. The second must JOIN the first rather than
    // start a second convert.py writing the same directory.
    std::atomic<int> frames_a{0};
    std::atomic<int> frames_b{0};
    const std::string source = container_dir;

    const auto id_a = reg.admit(source, [&frames_a](const mm::AdmissionProgress&) {
        ++frames_a;
    }, err);
    CHECK(!id_a.empty());
    const auto id_b = reg.admit(source, [&frames_b](const mm::AdmissionProgress&) {
        ++frames_b;
    }, err);
    CHECK(!id_b.empty());

    // THE assertion: one operation, not two. Before the fix these were distinct
    // ids and both threads ran.
    CHECK(id_a == id_b);

    // The joiner gets a replayed frame immediately rather than waiting for the
    // next stage boundary, which mid-convert can be twenty minutes away.
    CHECK(frames_b.load() >= 1);

    // And the registry lists ONE operation for the pair.
    int matching = 0;
    for (const auto& op : reg.operations())
        if (op.source_ref == source) ++matching;
    CHECK(matching == 1);

    reg.cancel(id_a);
    for (int i = 0; i < 100; ++i) {
        const auto op = reg.operation(id_a);
        if (op && op->done) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    const auto finished = reg.operation(id_a);
    CHECK(finished.has_value());
    // Exactly one terminal frame on every path — including cancelled-while-
    // queued, which used to publish nothing at all and left a watcher waiting
    // forever for an operation that had already stopped.
    CHECK(finished->done);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return true;
}

bool test_desired_artifact_names_what_a_node_lacks() {
    // The distinction this exists to hold: what a node HAS versus what it
    // WANTS. `needs_artifact` was filled from installed_artifact(), which
    // requires a working runtime — and is consulted only when provisioning
    // FAILED. So a node that could not build asked for help by naming the build
    // it did not have, and named nothing: the share path had no reachable
    // trigger at all.
    mm::SomaEngineProvisioner soma("soma");
    mm::EngineSpec soma_spec;
    soma_spec.engine_id = "soma";

    // Soma ships with the node, so there is no peer that could supply one.
    // Naming an artifact would send control hunting a source that cannot exist.
    CHECK(!soma.desired_artifact(soma_spec).has_value());
    CHECK(!soma.installed_artifact().has_value());
    CHECK(!soma.shareable());

    mm::LlamaEngineProvisioner llama("llama-server", {});
    mm::EngineSpec llama_spec;
    llama_spec.engine_id = "llama-cpp";

    // Nothing resolved yet: no version and no variant, so the node cannot name
    // what it needs and says so rather than emitting a fingerprint with blank
    // fields that would match no source ever.
    llama_spec.version = "latest";
    CHECK(!llama.desired_artifact(llama_spec).has_value());

    // A pinned version is knowable immediately — but a variant still is not,
    // and every real llama install advertises an accelerator. A fingerprint
    // with a blank variant 404s every time, which is worse than admitting
    // ignorance.
    llama_spec.version = "b4321";
    CHECK(!llama.desired_artifact(llama_spec).has_value());

    // Nothing is installed, so it is not a source for anyone.
    CHECK(!llama.installed_artifact().has_value());
    CHECK(!llama.shareable());
    return true;
}

namespace {

/// A provisioner that fails the only way the two real ones can't be made to on
/// demand. Every other method is inert; `ensure` is the one the manager calls.
class ThrowingProvisioner final : public mm::EngineProvisioner {
public:
    explicit ThrowingProvisioner(std::string id) : id_(std::move(id)) {}

    const std::string& engine_id() const override { return id_; }
    mm::RuntimeStatus ensure(const mm::EngineSpec&) override {
        ++calls;
        throw std::runtime_error("provisioner exploded");
    }
    mm::RuntimeStatus check_for_update(const mm::EngineSpec&) override { return status(); }
    mm::RuntimeStatus update(const mm::EngineSpec&, const std::string&) override {
        return status();
    }
    mm::RuntimeStatus status() const override {
        mm::RuntimeStatus s;
        s.engine_id = id_;
        s.status = "absent";
        return s;
    }
    std::optional<mm::EngineArtifact> installed_artifact() const override { return std::nullopt; }
    std::optional<mm::EngineArtifact> desired_artifact(const mm::EngineSpec&) const override {
        return std::nullopt;
    }
    bool shareable() const override { return false; }
    bool package(const std::string&, std::string& err) override {
        err = "not shareable";
        return false;
    }
    bool install_package(const std::string&,
                         const mm::EngineArtifact&,
                         std::string& err) override {
        err = "not installable";
        return false;
    }
    std::string executable_path() const override { return {}; }

    int calls = 0;

private:
    std::string id_;
};

} // namespace

bool test_provisioner_exception_fails_the_engine_not_the_node() {
    // The guarantee behind D56: whatever a provisioner does, the node survives
    // it and says what happened. The reentrant lock was one way to throw; the
    // point of the boundary is that it does not have to be the last one.
    // Soma stays REAL and stays primary, and is planted where it will actually
    // resolve, because the second half of the claim is that a backup blowing up
    // does not stop the primary from coming up. With one catch around the whole
    // loop it would have — and a primary that failed for its own reasons would
    // have hidden the difference.
    const std::string soma_name = "mm-soma-fixture-" + mm::util::generate_uuid();
#ifdef _WIN32
    const std::filesystem::path planted =
        std::filesystem::path(mm::util::executable_dir()) / (soma_name + ".exe");
#else
    const std::filesystem::path planted =
        std::filesystem::path(mm::util::executable_dir()) / soma_name;
#endif
    {
        std::ofstream out(planted, std::ios::binary | std::ios::trunc);
        out << "not a real engine";
    }

    mm::EngineManagerPaths paths;
    paths.llama_provision_dir = temp_test_dir("engine-manager-throw").string();
    paths.soma_executable = soma_name;
    mm::NodeEngineManager manager(paths);

    auto owned = std::make_unique<ThrowingProvisioner>("llama-cpp");
    auto* thrower = owned.get();
    manager.set_provisioner("llama-cpp", std::move(owned));

    mm::ClusterEngineConfig cfg;
    cfg.version = 1;
    cfg.primary_engine = "soma";
    cfg.backup_engine = "llama-cpp";
    mm::EngineSpec soma_spec;
    soma_spec.engine_id = "soma";
    mm::EngineSpec llama_spec;
    llama_spec.engine_id = "llama-cpp";
    cfg.engines = {soma_spec, llama_spec};

    manager.apply(cfg); // must return, not terminate
    CHECK(thrower->calls == 1);

    const auto conf = manager.conformance();
    // Failed, not Converging: the application finished, and a node stuck
    // reporting "wait" forever reads healthier than one reporting a fault.
    CHECK(conf.state == mm::EngineConformanceState::Failed);
    CHECK(conf.detail.find("provisioner exploded") != std::string::npos);

    // And the per-engine row says error, not absent. Absent means nothing
    // tried; something tried and blew up, and an operator reading "absent"
    // would go looking for a missing install rather than a crash.
    bool saw_llama = false, soma_ready = false;
    for (const auto& s : manager.engine_statuses()) {
        if (s.engine_id == "soma") soma_ready = s.ready;
        if (s.engine_id != "llama-cpp") continue;
        saw_llama = true;
        CHECK(s.status == "error");
        CHECK(!s.ready);
        CHECK(s.last_error.find("provisioner exploded") != std::string::npos);
    }
    CHECK(saw_llama);
    CHECK(soma_ready);

    std::error_code ec;
    std::filesystem::remove(planted, ec);
    return true;
}

bool test_soma_resolves_beside_the_node_binary() {
    // D58: the node looked for `soma` on PATH alone, while the documented rule —
    // and the deployment layout — is that soma ships BESIDE the node binary. In
    // any build tree those are different directories, so the engine built with
    // the node was invisible to it and every Soma-primary config reported the
    // engine absent.
    const std::string dir = mm::util::executable_dir();
    CHECK(!dir.empty());

    // Written beside the RUNNING test binary, because that is the only way to
    // assert "beside the executable" without hardcoding a build layout — which
    // is the very thing this fix exists to avoid.
    const std::string name = "mm-soma-fixture-" + mm::util::generate_uuid();
#ifdef _WIN32
    const std::filesystem::path planted = std::filesystem::path(dir) / (name + ".exe");
#else
    const std::filesystem::path planted = std::filesystem::path(dir) / name;
#endif
    {
        std::ofstream out(planted, std::ios::binary | std::ios::trunc);
        out << "not a real engine";
    }

    mm::EngineSpec spec;
    spec.engine_id = "soma";

    // The bare name resolves to the sibling, and to its full path — a status
    // that echoed the request back would pass a weaker test while telling the
    // supervisor nothing it did not already have.
    mm::SomaEngineProvisioner found(name);
    const auto ok = found.ensure(spec);
    CHECK(ok.ready);
    CHECK(ok.status == "ready");
    CHECK(ok.executable_path == planted.string());

    std::error_code ec;
    std::filesystem::remove(planted, ec);

    // Gone from both places now, and the failure has to NAME where it looked.
    // "not found, check the install" is a dead end when the operator's next
    // question is which of the two directories was empty.
    mm::SomaEngineProvisioner missing(name);
    const auto absent = missing.ensure(spec);
    CHECK(!absent.ready);
    CHECK(absent.status == "absent");
    CHECK(absent.last_error.find(dir) != std::string::npos);
    return true;
}

bool test_provisioning_progress_sink_may_read_status() {
    // The node crash: a cluster config naming llama-cpp killed the node inside
    // one health poll, with nothing in the log, on every fresh install (D56).
    //
    // The shape is a lock held across a callback. LlamaEngineProvisioner held
    // one mutex for the whole of ensure(); ensure() reports progress by calling
    // the progress sink; the node's sink asks the engine manager for llama
    // status, which comes straight back into the same object and takes the same
    // mutex on the same thread. MSVC does not hang on that — it THROWS
    // `resource deadlock would occur` — out of a worker thread with no handler,
    // into std::terminate, which is abort().
    //
    // So the assertion is not "no deadlock", which a passing test cannot
    // distinguish from a test that never reached the callback. It is: the sink
    // FIRED, it read status from inside the callback, and ensure() returned.
    // The middle one is what makes the other two mean anything.
    auto dir = temp_test_dir("provisioner-reentrancy");
    std::filesystem::create_directories(dir);

    // Injected wholesale: reaching the progress callback at all requires
    // getting into a managed install, and a test that downloaded llama.cpp to
    // get there would be a network test that fails for its own reasons.
    mm::LlamaCommandRunner runner;
    runner.resolve_executable = [](const std::string&) { return std::string{}; };
    runner.capture_output = [](const std::vector<std::string>&,
                               const std::filesystem::path&) { return std::string{}; };
    runner.capture_first_line = [](const std::vector<std::string>&,
                                   const std::filesystem::path&) { return std::string{}; };
    runner.fetch_latest = [](const mm::LlamaProvisionConfig&) { return std::string{"b4321"}; };
    runner.fetch_release_assets = [](const mm::LlamaProvisionConfig&, const std::string&) {
        return std::vector<std::string>{};
    };
    runner.run = [](const std::vector<std::string>&,
                    const std::filesystem::path&,
                    const mm::StreamLineCallback&,
                    const mm::CancelCheckCallback&,
                    std::string*) { return 0; };

    mm::LlamaEngineProvisioner llama("llama-server", dir.string(), runner);

    int progress_frames = 0;
    int reads_from_inside_the_callback = 0;
    llama.set_progress_sink([&](const mm::RuntimeInstallProgress& p) {
        if (!p.active) return;
        ++progress_frames;
        // Exactly what src/node/main.cpp's progress sink does — it reads the
        // accelerator to label the operation. Reading it here is the whole
        // test; before the fix this threw.
        const auto s = llama.llama_status();
        (void)s.accelerator;
        // …and the generic accessors, which the API thread and the TUI reach
        // through on the same object while a build runs.
        (void)llama.status();
        (void)llama.executable_path();
        ++reads_from_inside_the_callback;
    });

    mm::EngineSpec spec;
    spec.engine_id = "llama-cpp";
    spec.version = "latest";
    spec.install_method = "auto";

    bool threw = false;
    try {
        llama.ensure(spec);
    } catch (...) {
        threw = true;
    }

    CHECK(!threw);
    // The install cannot SUCCEED here — no bytes were ever downloaded — and
    // that is fine. It has to reach the progress callback, which is a
    // different claim, and the one this test is about.
    CHECK(progress_frames > 0);
    CHECK(reads_from_inside_the_callback == progress_frames);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return true;
}

bool test_node_modal_ladder() {
    using mm::NodeModal;
    using mm::NodeModalInputs;

    // Everything available at once. The precedence has to be asserted from the
    // TOP down, because that is the property the old code could not state: it
    // was five scattered "if X then Y = false" statements plus a second,
    // differently-ordered ladder in the event handler, agreeing only by luck.
    NodeModalInputs all;
    all.progress_active = true;
    all.engine_switch_available = true;
    all.engine_variants_listed = true;
    all.can_troubleshoot = true;
    all.troubleshoot_unacknowledged = true;
    all.can_install_target = true;
    all.target_unacknowledged = true;
    all.can_update = true;
    all.update_unacknowledged = true;

    // Progress outranks everything, from any current modal.
    for (const auto cur : {NodeModal::None, NodeModal::EngineSwitch, NodeModal::Troubleshoot,
                           NodeModal::Target, NodeModal::Update})
        CHECK(mm::resolve_node_modal(all, cur) == NodeModal::Progress);

    auto no_progress = all;
    no_progress.progress_active = false;

    // EngineSwitch next — but ONLY when it is already open. It has no
    // auto-open, so from None the ladder falls through to Troubleshoot even
    // with everything else available.
    CHECK(mm::resolve_node_modal(no_progress, NodeModal::EngineSwitch) ==
          NodeModal::EngineSwitch);
    CHECK(mm::resolve_node_modal(no_progress, NodeModal::None) == NodeModal::Troubleshoot);

    // Then Troubleshoot > Target > Update, each asserted by removing the one
    // above it rather than by trusting the order of the branches.
    auto below_troubleshoot = no_progress;
    below_troubleshoot.can_troubleshoot = false;
    CHECK(mm::resolve_node_modal(below_troubleshoot, NodeModal::None) == NodeModal::Target);

    auto below_target = below_troubleshoot;
    below_target.can_install_target = false;
    CHECK(mm::resolve_node_modal(below_target, NodeModal::None) == NodeModal::Update);

    auto below_update = below_target;
    below_update.can_update = false;
    CHECK(mm::resolve_node_modal(below_update, NodeModal::None) == NodeModal::None);

    // Acknowledgement closes an auto-opening prompt — and NOT while it is the
    // current modal, or acknowledging from inside would close it mid-read.
    NodeModalInputs update_only;
    update_only.can_update = true;
    update_only.update_unacknowledged = false;
    CHECK(mm::resolve_node_modal(update_only, NodeModal::None) == NodeModal::None);
    CHECK(mm::resolve_node_modal(update_only, NodeModal::Update) == NodeModal::Update);
    update_only.update_unacknowledged = true;
    CHECK(mm::resolve_node_modal(update_only, NodeModal::None) == NodeModal::Update);

    // "Can" gates the prompt regardless of stickiness: a runtime that stops
    // offering an update must close the prompt, not keep it pinned open.
    update_only.can_update = false;
    CHECK(mm::resolve_node_modal(update_only, NodeModal::Update) == NodeModal::None);

    // EngineSwitch closes when its variant list empties — the modal is a menu,
    // and a menu with nothing in it is not a thing to show.
    NodeModalInputs engine_only;
    engine_only.engine_switch_available = true;
    engine_only.engine_variants_listed = true;
    CHECK(mm::resolve_node_modal(engine_only, NodeModal::EngineSwitch) ==
          NodeModal::EngineSwitch);
    engine_only.engine_variants_listed = false;
    CHECK(mm::resolve_node_modal(engine_only, NodeModal::EngineSwitch) == NodeModal::None);
    engine_only.engine_variants_listed = true;
    engine_only.engine_switch_available = false;
    CHECK(mm::resolve_node_modal(engine_only, NodeModal::EngineSwitch) == NodeModal::None);

    // Nothing available is None from every starting point — no modal can pin
    // itself open against its own preconditions.
    NodeModalInputs none;
    for (const auto cur : {NodeModal::None, NodeModal::Progress, NodeModal::EngineSwitch,
                           NodeModal::Troubleshoot, NodeModal::Target, NodeModal::Update})
        CHECK(mm::resolve_node_modal(none, cur) == NodeModal::None);

    CHECK(std::string(mm::to_string(NodeModal::EngineSwitch)) == "engine-switch");
    CHECK(std::string(mm::to_string(NodeModal::None)) == "none");
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
    // Asserted STRUCTURALLY: an agent with no node-local backend resolves to an
    // empty engine id. The previous version matched the phrase "supports
    // llama-cpp only", which made the test a hostage of the error prose — the
    // same mistake the capacity-pressure substring matcher made.
    RECORD(mm::AgentScheduler::resolve_backend(retired).engine_id.empty());
    RECORD(bad_loads.load() == 0);
    RECORD(good_loads.load() == 0);

    // And the agent that DOES own a slot routes to a real engine. `auto` with no
    // admission record is the fallback, which is policy rather than a placeholder:
    // absence of a record is not evidence of admissibility.
    RECORD(mm::AgentScheduler::resolve_backend(cfg).engine_id == "llama-cpp");

    // An operator override to Soma is REFUSED while no record admits these
    // weights — forcing the streaming engine onto a model nothing has passed
    // through conformance is the same bet as overriding a `reject`, with less
    // evidence behind it. Once the model registry lands, an admitted model with
    // a stream verdict is what makes this resolve to "soma".
    auto forced = cfg;
    forced.backend_override = "soma";
    RECORD(mm::AgentScheduler::resolve_backend(forced).engine_id == "llama-cpp");
    RECORD(mm::AgentScheduler::resolve_backend(forced).reason.find("override_refused") !=
           std::string::npos);

    auto forced_fallback = cfg;
    forced_fallback.backend_override = "fallback";
    RECORD(mm::AgentScheduler::resolve_backend(forced_fallback).engine_id == "llama-cpp");

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

bool test_soma_footprint_is_ram_shaped_not_vram_shaped() {
    // D62: every demand figure was VRAM, produced by llama.cpp's estimator, for
    // BOTH engines. Soma v1 is CPU-only, and evaluate_fit() refuses to offload
    // against a host with less than min_gpu_for_offload_mb (8 GiB) of GPU — so a
    // Soma agent could not be placed on a GPU-less node at all, for VRAM it
    // would never have touched. That is a rejection, not merely a bad estimate.
    mm::ResourceFootprint soma;
    soma.ram_mb = 8192; // a resident half, no VRAM — the shape soma_footprint returns

    mm::HostCapacity cpu_only;
    cpu_only.vram_total_mb = 0; // no GPU whatsoever
    cpu_only.vram_free_mb = 0;
    cpu_only.ram_total_mb = 65536;
    cpu_only.ram_free_mb = 64000;
    cpu_only.disk_free_mb = 500000;

    const mm::CapacityPolicy policy;
    std::string reason;
    // A zero-VRAM ask short-circuits to Native: RAM and disk are still checked,
    // the GPU question simply never arises.
    CHECK(mm::evaluate_fit(soma, cpu_only, policy, &reason) == mm::FitQuality::Native);

    // The same host, asked the OLD way — a llama-shaped VRAM figure for a model
    // whose real cost is RAM. This is what placement used to compute for Soma,
    // and it is refused outright: not "ranked lower", refused.
    mm::ResourceFootprint vram_shaped;
    vram_shaped.vram_mb = 8192;
    CHECK(mm::evaluate_fit(vram_shaped, cpu_only, policy, &reason) == mm::FitQuality::None);
    CHECK(reason.find("no GPU large enough") != std::string::npos);

    // RAM is still a real constraint in the new shape — this is not "Soma fits
    // everywhere". A resident half larger than the host's free RAM is refused,
    // which is the whole point of moving the demand onto the right axis.
    mm::ResourceFootprint too_big;
    too_big.ram_mb = 63000; // leaves under the 2 GiB RAM headroom
    CHECK(mm::evaluate_fit(too_big, cpu_only, policy, &reason) == mm::FitQuality::None);
    CHECK(reason.find("MB RAM") != std::string::npos);

    // And the producer, not just the rule it feeds. Whatever the registry knows
    // or does not know about a model, the Soma footprint never asks for VRAM —
    // that is the invariant, and the one the old code broke for every agent.
    const auto dir = temp_test_dir("soma-footprint");
    std::filesystem::create_directories(dir / "models");
    {
        mm::NodeRegistry registry((dir / "registry").string());
        mm::AgentScheduler scheduler(registry, (dir / "models").string());
        mm::AgentConfig cfg;
        cfg.id = "soma-agent";
        cfg.model_path = "org/model";
        const auto fp = scheduler.soma_footprint(cfg);
        CHECK(fp.vram_mb == 0);
        // No registry and no local bytes, so the resident half is unknown and
        // reported as nothing rather than guessed — under-charging is the safe
        // direction, since the node re-derives the real plan before loading.
        CHECK(fp.ram_mb == 0);
        CHECK(fp.disk_mb == 0);
    }
    CHECK(remove_tree(dir));
    return true;
}

bool test_placement_failure_codes_separate_eligibility_from_capacity() {
    // The pair D64 exists for. "No node is conforming" and "every node is full"
    // called for opposite operator actions and produced the SAME sentence, so
    // the only way to tell them apart was to match prose.
    //
    // The eligibility half is covered by the engine-config gate test with no
    // nodes registered. This is the other half, which needs a node that is
    // connected and conforming and simply has no room — otherwise the two codes
    // could be swapped and nothing would notice.
    const auto dir = temp_test_dir("placement-failure-codes");
    std::filesystem::create_directories(dir / "models");
    const uint16_t port = find_free_test_port();
    CHECK(port != 0);
    const std::string url = "http://127.0.0.1:" + std::to_string(port);

    httplib::Server server;
    server.Get("/api/node/health", [](const httplib::Request&, httplib::Response& res) {
        mm::NodeHealthMetrics health;
        // Connected and healthy, with nowhere near enough room for any model.
        // This is what separates "not eligible" from "no capacity": the node
        // passes every filter except the one about space.
        health.gpu_vram_total_mb = 1;
        health.gpu_vram_used_mb = 1;
        health.gpu_backend_available = true;
        health.disk_free_mb = 1;
        res.set_content(nlohmann::json(health).dump(), "application/json");
    });
    server.Get("/api/node/status", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"slots", nlohmann::json::array()},
                                       {"max_slots", 1},
                                       {"slot_available", 1}}
                            .dump(),
                        "application/json");
    });

    std::atomic<bool> listen_ok{false};
    std::thread server_thread([&] { listen_ok = server.listen("127.0.0.1", port); });

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    RECORD(wait_for_test_server(url));
    mm::NodeRegistry registry((dir / "registry").string());
    const auto node_id = registry.add_node(url, "codes-secret", "test");
    registry.start_health_poll(1);
    RECORD(wait_for_registered_node(registry, node_id));
    mm::AgentScheduler scheduler(registry, (dir / "models").string());

    mm::AgentConfig cfg;
    cfg.id = "no-room-agent";
    cfg.name = "No Room";
    cfg.model_path = "org/model";

    RECORD(!scheduler.ensure_agent_running(cfg).has_value());
    // A node IS eligible — it is connected and no engine-config gate is set —
    // so this must not report NoEligibleNode.
    RECORD(!registry.available_nodes().empty());
    RECORD(scheduler.last_failure() == mm::PlacementFailure::NoCapacity);
    RECORD(mm::placement_failure_retryable(scheduler.last_failure()));
    // The prose is still there and still useful; it is simply no longer the
    // only thing carrying the answer.
    RECORD(scheduler.last_error().find("no capacity") != std::string::npos);

    // The wire spellings are part of the contract: clients branch on these, so
    // renaming one is an API change and should break a test.
    RECORD(std::string(mm::to_string(mm::PlacementFailure::None)) == "none");
    RECORD(std::string(mm::to_string(mm::PlacementFailure::NoEligibleNode)) == "no_eligible_node");
    RECORD(std::string(mm::to_string(mm::PlacementFailure::NoCapacity)) == "no_capacity");
    RECORD(std::string(mm::to_string(mm::PlacementFailure::EngineConfigMissing)) ==
           "engine_config_missing");
    // Only the two that genuinely need a human are non-retryable.
    RECORD(!mm::placement_failure_retryable(mm::PlacementFailure::EngineConfigMissing));
    RECORD(!mm::placement_failure_retryable(mm::PlacementFailure::NoLocalBackend));
    RECORD(mm::placement_failure_retryable(mm::PlacementFailure::NodeUnreachable));
    RECORD(mm::placement_failure_retryable(mm::PlacementFailure::ModelTransferFailed));

    registry.stop_health_poll();
    server.stop();
    if (server_thread.joinable()) server_thread.join();
    RECORD(listen_ok);
    RECORD(remove_tree(dir));

#undef RECORD
    return ok;
}

bool test_scheduler_audits_placement_and_release() {
    // The wiring D60 was missing: nothing ever CALLED record_placement. A
    // registry round-trip proves the table works and says nothing about whether
    // a placement reaches it, which is the half that was actually broken.
    const auto dir = temp_test_dir("scheduler-audit");
    std::filesystem::create_directories(dir / "models");
    const uint16_t port = find_free_test_port();
    CHECK(port != 0);
    const std::string url = "http://127.0.0.1:" + std::to_string(port);

    httplib::Server server;
    server.Get("/api/node/health", [](const httplib::Request&, httplib::Response& res) {
        mm::NodeHealthMetrics health;
        health.gpu_vram_total_mb = 131072;
        health.gpu_backend_available = true;
        res.set_content(nlohmann::json(health).dump(), "application/json");
    });
    server.Get("/api/node/status", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"slots", nlohmann::json::array()},
                                       {"max_slots", 2},
                                       {"slot_available", 2}}
                            .dump(),
                        "application/json");
    });
    server.Post("/api/node/load-model", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"status", "loaded"}, {"slot_id", "slot-audit"}}.dump(),
                        "application/json");
    });
    server.Post("/api/node/detach-agent", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"status", "detached"}}.dump(), "application/json");
    });

    std::atomic<bool> listen_ok{false};
    std::thread server_thread([&] { listen_ok = server.listen("127.0.0.1", port); });

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    RECORD(wait_for_test_server(url));
    mm::NodeRegistry registry((dir / "registry").string());
    const auto node_id = registry.add_node(url, "audit-secret", "test");
    registry.start_health_poll(1);
    RECORD(wait_for_registered_node(registry, node_id));
    mm::AgentScheduler scheduler(registry, (dir / "models").string());

    struct Placed {
        mm::AgentId agent_id;
        mm::NodeId node_id;
        mm::SlotId slot_id;
        std::string backend;
        std::string reason;
    };

    std::mutex audit_mutex;
    std::vector<Placed> placed;
    std::vector<mm::AgentId> released;
    scheduler.set_placement_audit({[&](const mm::AgentId& a,
                                       const mm::NodeId& n,
                                       const mm::SlotId& sl,
                                       const std::string& backend,
                                       const std::string& reason,
                                       const mm::ResourceFootprint&) {
                                       std::lock_guard<std::mutex> lk(audit_mutex);
                                       placed.push_back({a, n, sl, backend, reason});
                                   },
                                   [&](const mm::AgentId& a) {
                                       std::lock_guard<std::mutex> lk(audit_mutex);
                                       released.push_back(a);
                                   }});

    mm::AgentConfig cfg;
    cfg.id = "audited-agent";
    cfg.name = "Audited";
    cfg.model_path = "org/model";
    cfg.preferred_node_id = node_id;

    RECORD(scheduler.ensure_agent_running(cfg).has_value());
    {
        std::lock_guard<std::mutex> lk(audit_mutex);
        RECORD(placed.size() == 1);
        if (placed.size() == 1) {
            RECORD(placed[0].agent_id == cfg.id);
            RECORD(placed[0].node_id == node_id);
            RECORD(placed[0].slot_id == "slot-audit");
            // The engine the scheduler ACTED on, and its reason string — not a
            // reconstruction. Nothing is admitted here, so it routes to the
            // fallback and says so.
            RECORD(placed[0].backend == "llama-cpp");
            RECORD(placed[0].reason.find("no_admission_record") != std::string::npos);
        }
        RECORD(released.empty());
    }

    // A second ensure on an UNCHANGED placement is a refresh, not a placement.
    // store_placement() is called on that path too, which is exactly why the
    // audit hangs off the two publish sites rather than off the store.
    RECORD(scheduler.ensure_agent_running(cfg).has_value());
    {
        std::lock_guard<std::mutex> lk(audit_mutex);
        RECORD(placed.size() == 1);
    }

    scheduler.release_agent(cfg.id);
    {
        std::lock_guard<std::mutex> lk(audit_mutex);
        RECORD(released.size() == 1);
        if (released.size() == 1) RECORD(released[0] == cfg.id);
    }

    registry.stop_health_poll();
    server.stop();
    if (server_thread.joinable()) server_thread.join();
    RECORD(listen_ok);
    RECORD(remove_tree(dir));

#undef RECORD
    return ok;
}

bool test_placement_history_records_and_closes_rows() {
    // D60: the table, its index and record_placement() all shipped with no
    // caller and no query. The subtle half is the CLOSE rule — an agent that was
    // placed, released, and placed again has two rows, and stamping "the row for
    // this agent" would close the older one a second time and lose the first
    // placement's duration entirely.
    auto dir = temp_test_dir("placement-history");
    // Scoped so the SQLite handle is closed before the directory is removed —
    // on Windows an open file cannot be deleted, and the failure surfaces as a
    // cleanup error that says nothing about the behaviour under test.
    {
        mm::ControlModelRegistry reg;
        std::string err;
        CHECK(reg.open(dir.string(), err));

        CHECK(reg.placement_history("agent-1").empty());

        mm::ResourceFootprint first;
        first.vram_mb = 4096;
        reg.record_placement(
            "agent-1", "node-a", "slot-1", "llama-cpp", "llama-cpp (no_admission_record)", first);

        auto rows = reg.placement_history("agent-1");
        CHECK(rows.size() == 1);
        CHECK(rows[0].node_id == "node-a");
        CHECK(rows[0].backend == "llama-cpp");
        CHECK(rows[0].backend_reason == "llama-cpp (no_admission_record)");
        CHECK(rows[0].vram_mb == 4096);
        // Open until something closes it — and 0 rather than a null, so a renderer
        // does not need a separate presence check to ask the same question.
        CHECK(rows[0].open());
        CHECK(rows[0].released_at_ms == 0);

        reg.mark_placement_released("agent-1");
        rows = reg.placement_history("agent-1");
        CHECK(rows.size() == 1);
        CHECK(!rows[0].open());
        const std::int64_t first_released = rows[0].released_at_ms;
        CHECK(first_released > 0);

        // Placed again, on a different engine. Two rows now, newest first.
        mm::ResourceFootprint second;
        second.ram_mb = 19840;
        second.disk_mb = 14848;
        reg.record_placement(
            "agent-1", "node-b", "slot-7", "soma", "soma (verdict=stream)", second);
        rows = reg.placement_history("agent-1");
        CHECK(rows.size() == 2);
        CHECK(rows[0].backend == "soma"); // newest first
        CHECK(rows[0].ram_mb == 19840);
        CHECK(rows[0].disk_mb == 14848);
        CHECK(rows[0].open());
        CHECK(rows[1].backend == "llama-cpp");
        CHECK(!rows[1].open());

        // THE RULE: closing again must close the NEW row and leave the old one's
        // timestamp untouched.
        reg.mark_placement_released("agent-1");
        rows = reg.placement_history("agent-1");
        CHECK(rows.size() == 2);
        CHECK(!rows[0].open());
        CHECK(rows[1].released_at_ms == first_released);

        // Idempotent: a release for an agent with no open row is a no-op, because a
        // release can legitimately arrive for a placement this process never saw.
        reg.mark_placement_released("agent-1");
        reg.mark_placement_released("agent-never-placed");
        rows = reg.placement_history("agent-1");
        CHECK(rows.size() == 2);
        CHECK(rows[1].released_at_ms == first_released);

        // Other agents are not touched, and the limit is honoured.
        CHECK(reg.placement_history("agent-2").empty());
        CHECK(reg.placement_history("agent-1", 1).size() == 1);
        CHECK(reg.placement_history("agent-1", 0).empty());
    }

    CHECK(remove_tree(dir));
    return true;
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
    // Structural again: an "api" agent owns no node-local slot, so it resolves
    // to no engine at all — which is a different statement from "this branch
    // only supports llama-cpp", and the one that stays true now that it does not.
    RECORD(mm::AgentScheduler::resolve_backend(api_cfg).engine_id.empty());
    RECORD(scheduler.last_error().find("no node-local backend") != std::string::npos);

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

        // Is the server that answered OURS?
        //
        // `listen()` blocks while serving, so `listen_ok` cannot be read until
        // teardown — which is where it IS asserted, ~350 lines below. That is too
        // late to be useful: if another process holds this port, our bind fails,
        // the readiness poll above is satisfied by THEIR server, and every
        // assertion in between fails against a stranger returning plausible wrong
        // statuses. The teardown check then reports the cause after twenty
        // confusing symptoms.
        //
        // `listen_returned` is readable now and says the same thing early: our
        // listen() has only returned if it FAILED, because a serving one does not
        // return. A ready server plus a returned listen means the responder is
        // not ours.
        RECORD(!listen_returned);

        // Transport retries: see kTransportRetries. One budget for the whole
        // file — this test is where two of them diverging did damage.
        auto with_retry = [](auto&& request) {
            mm::HttpResponse resp;
            for (int attempt = 0; attempt < kTransportRetries; ++attempt) {
                resp = request();
                if (resp.status != 0) break;
                transport_backoff(attempt);
            }
            return resp;
        };

        auto expect_error = [&](const mm::HttpResponse& resp,
                                int expected_status,
                                const std::string& expected_text,
                                int call_line) {
            // Same ambiguity `reached_server` exists to remove, on the non-SSE
            // path: status stays 0 when the request never arrived, and `0 !=
            // 401` fails exactly the way a broken auth gate does. This half of
            // the file was left without the guard when D1 was fixed, so the
            // recurrence reported a bare status mismatch and said nothing about
            // which of the two findings it was. Checked first, and loudly.
            if (resp.status == 0) {
                std::cerr << "  TRANSPORT FAILURE at line " << call_line << " after "
                          << kTransportRetries << " attempts: status=0\n"
                          << "  (the assertion below is about to fail for a reason that\n"
                          << "   has nothing to do with authorization)\n";
            }
            RECORD(resp.status != 0);
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
        // /v1/models is the ADMISSION REGISTRY now, not agents wearing model
        // costumes — the agents catalog lives on the :9091 OpenAI-compat
        // listener where it belongs. This server has no registry attached, so
        // the route answers 503; that still proves the token gate let it
        // through, which is what this test is about. The old assertion (that the
        // body contained "agent:agent-a") is the behaviour being retired.
        auto valid_models = with_retry([&] { return client.get("/v1/models"); });
        RECORD(valid_models.status == 503);
        RECORD(valid_models.body.find("registry") != std::string::npos);
        RECORD(valid_models.body.find("agent:agent-a") == std::string::npos);
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
            int retries = 0;
        };
        auto stream_chat = [&](const std::string& token,
                               const nlohmann::json& body) {
            mm::HttpClient stream_client(base_url);
            if (!token.empty()) stream_client.set_bearer_token(token);
            StreamAttempt attempt;
            for (int retry = 0; retry < kTransportRetries; ++retry) {
                attempt = StreamAttempt{};
                attempt.retries = retry;
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
                transport_backoff(retry);
            }
            return attempt;
        };

        // A transport failure and a wrong status code are DIFFERENT findings,
        // and the assertions below cannot tell them apart: status stays 0 when
        // the request never reached the server, and `0 == 403` fails exactly the
        // way a broken auth gate does. That ambiguity cost a diagnosis once —
        // one intermittent failure at `status == 403` that said nothing about
        // whether authorization had been consulted at all.
        //
        // Checked first, separately, and loudly.
        const auto reached_server = [&](const StreamAttempt& a, const char* what) {
            if (a.status != 0) return;
            std::cerr << "  TRANSPORT FAILURE on " << what << " after " << (a.retries + 1)
                      << " attempts: status=0, body=\"" << a.body.substr(0, 120) << "\"\n"
                      << "  (the auth assertions below are about to fail for a reason that\n"
                      << "   has nothing to do with auth)\n";
        };

        auto missing_chat = stream_chat("", nlohmann::json{{"message", "hello"}});
        reached_server(missing_chat, "missing_chat");
        RECORD(missing_chat.status != 0);
        RECORD(!missing_chat.ok);
        RECORD(missing_chat.status == 401);
        RECORD(missing_chat.body.find("missing bearer token") != std::string::npos);
        RECORD(missing_chat.events.empty());

        auto invalid_chat = stream_chat("wrong-secret", nlohmann::json{{"message", "hello"}});
        reached_server(invalid_chat, "invalid_chat");
        RECORD(invalid_chat.status != 0);
        RECORD(!invalid_chat.ok);
        RECORD(invalid_chat.status == 403);
        RECORD(invalid_chat.body.find("invalid bearer token") != std::string::npos);
        RECORD(invalid_chat.events.empty());

        auto node_chat = stream_chat("node-secret", nlohmann::json{{"message", "hello"}});
        reached_server(node_chat, "node_chat");
        RECORD(node_chat.status != 0);
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
            for (int attempt = 0; attempt < kTransportRetries; ++attempt) {
                resp = request();
                if (resp.status != 0) break;
                transport_backoff(attempt);
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
            for (int attempt = 0; attempt < kTransportRetries; ++attempt) {
                resp = request();
                if (resp.status != 0) break;
                transport_backoff(attempt);
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
            // Same budget as everywhere else; see kTransportRetries. This loop
            // was the third copy of the policy and the second one with 3.
            for (int retry = 0; retry < kTransportRetries; ++retry) {
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
                transport_backoff(retry);
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
    // Sharing is decided by the DESCRIPTOR's launch_compatible predicate now, so
    // the registry has to hold one. The executable is deliberately bogus: a load
    // that cannot attach must try to spawn and fail, which is the case under test.
    mm::EngineRegistry::instance().register_engine(
        mm::make_llama_descriptor("missing-llama-server"));

    mm::EngineSupervisor slots(46250, 46253, 4);
    slots.set_models_dir("missing-llama-server");
    mm::RuntimeSettings settings;
    const auto first = slots.add_ready_test_engine("llama-cpp",
        "model.gguf", "agent-a", settings, "mmproj-a.gguf");
    CHECK(!first.empty());

    mm::EngineLoadRequest same_req;
    same_req.model_path = "model.gguf";
    same_req.mmproj_path = "mmproj-a.gguf";
    same_req.settings = settings;
    const auto shared = slots.load("llama-cpp", same_req, "agent-b");
    CHECK(shared == first);

    mm::EngineLoadRequest other_req = same_req;
    other_req.mmproj_path = "mmproj-b.gguf";
    const auto different = slots.load("llama-cpp", other_req, "agent-c");
    CHECK(different.empty());
    const auto info = slots.find(first);
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
    mm::EngineSupervisor slots(46170, 46173, 2);
    slots.set_models_dir("missing-llama");
    slots.set_kv_checkpoint_dir((dir / "kv").string());

    mm::RuntimeSettings s;
    s.ctx_size = 2048;
    s.parallel = 1;
    const auto id = slots.add_ready_test_engine("llama-cpp", "m.gguf", "agent-l", s);

    auto info = slots.find(id);
    CHECK(info.has_value());
    CHECK(info->backend == "llama-cpp");
    // Backend survives the SlotInfo JSON round-trip.
    nlohmann::json j = *info;
    CHECK(j.get<mm::SlotInfo>().backend == "llama-cpp");

    // A test engine has no live process, so the KV save cannot succeed — and a
    // suspend whose checkpoint was not written now FAILS rather than reporting
    // success with an empty path. SlotManager reported Ok here and dropped the
    // context silently; that is the behaviour the rebuild does not carry forward.
    // The engine is left Ready, not killed.
    auto susp = slots.suspend(id);
    CHECK(susp.status == mm::EngineOpStatus::Failed);
    auto after = slots.find(id);
    CHECK(after.has_value());
    CHECK(after->state == mm::SlotState::Ready);

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
    // Every other server in this file asserts its own listen; this one did not,
    // so a stolen port would have been probed, answered by a stranger, and
    // reported as a health-check result.
    CHECK(listen_ok);
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
    mm::EngineSupervisor slots(46180, 46183, 4);
    slots.set_models_dir("missing-llama");
    mm::RuntimeSettings settings;
    settings.ctx_size = 4096;
    mm::EngineRegistry::instance().register_engine(
        mm::make_llama_descriptor("missing-llama"));
    const auto llama_id = slots.add_ready_test_engine("llama-cpp", "m.gguf", "agent-a", settings);
    mm::EngineLoadRequest compat_req;
    compat_req.model_path = "m.gguf";
    compat_req.settings = settings;
    CHECK(slots.load("llama-cpp", compat_req, "agent-b") == llama_id);
    auto llama_info = slots.find(llama_id);
    CHECK(llama_info.has_value());
    CHECK(llama_info->backend == "llama-cpp");
    CHECK(llama_info->assigned_agent == "agent-a");
    CHECK(llama_info->agent_ids.size() == 2);
    CHECK(std::find(llama_info->agent_ids.begin(), llama_info->agent_ids.end(),
                    "agent-b") != llama_info->agent_ids.end());

    auto incompatible = settings;
    incompatible.ctx_size = 8192;
    mm::EngineLoadRequest incompat_req;
    incompat_req.model_path = "m.gguf";
    incompat_req.settings = incompatible;
    CHECK(slots.load("llama-cpp", incompat_req, "agent-c").empty());
    llama_info = slots.find(llama_id);
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
    CHECK(!slots.find(llama_id).has_value());

    CHECK(remove_tree(dir));
    return true;
}

bool test_llama_restore_attaches_and_cleans_suspended_record() {
    auto dir = temp_test_dir("llama-restore-attach");
    mm::EngineSupervisor slots(46190, 46193, 4);
    mm::RuntimeSettings settings;
    settings.ctx_size = 4096;

    // Constructed directly rather than via suspend(): a test engine has no live
    // process, so its KV save fails and the suspend is correctly refused. What
    // this test is about is what RESTORE does with a suspended record.
    const auto suspended_id =
        slots.add_suspended_test_engine("llama-cpp", "m.gguf", "agent-b", settings);
    CHECK(slots.find(suspended_id)->state == mm::SlotState::Suspended);

    const auto ready_id = slots.add_ready_test_engine("llama-cpp", "m.gguf", "agent-a", settings);
    mm::EngineLoadRequest restore_req;
    restore_req.model_path = "m.gguf";
    restore_req.settings = settings;
    const auto restored = slots.restore("llama-cpp", restore_req, "", "agent-b");
    CHECK(restored == ready_id);
    CHECK(!slots.find(suspended_id).has_value());
    CHECK(slots.find_by_agent("agent-b") == ready_id);

    const auto ready = slots.find(ready_id);
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

bool test_multi_shard_directory_sizes_correctly() {
    // The bug: fs::file_size() sets an error_code on a DIRECTORY, and the caller
    // fell through to a flat 2048 MB. Every multi-shard HF checkpoint and every
    // converted Soma container therefore reported the same size — and that single
    // number is what placement consumed.
    auto dir = temp_test_dir("multi-shard");
    const auto model = dir / "model";
    std::filesystem::create_directories(model / "nested");

    const auto write_blob = [](const std::filesystem::path& p, std::size_t bytes) {
        std::ofstream f(p, std::ios::binary);
        const std::vector<char> chunk(bytes, '\0');
        f.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    };
    write_blob(model / "shard-00001.safetensors", 3 * 1024 * 1024);
    write_blob(model / "shard-00002.safetensors", 5 * 1024 * 1024);
    write_blob(model / "nested" / "extra.bin", 1024 * 1024);

    // Recursive, and it finds the nested file too.
    const auto bytes = mm::measure_model_bytes(model.string(), "", nullptr);
    CHECK(bytes == 9 * 1024 * 1024);

    // A single file still works, and a missing path reports nothing rather than
    // a plausible-looking constant.
    write_blob(dir / "single.gguf", 2 * 1024 * 1024);
    CHECK(mm::measure_model_bytes((dir / "single.gguf").string(), "", nullptr) ==
          2 * 1024 * 1024);
    CHECK(mm::measure_model_bytes((dir / "absent.gguf").string(), "", nullptr) == 0);

    // Two DIFFERENT directories must not size identically, which is the whole
    // point — the old path returned 2048 MB for both.
    const auto other = dir / "other";
    std::filesystem::create_directories(other);
    write_blob(other / "shard-00001.safetensors", 7 * 1024 * 1024);
    CHECK(mm::measure_model_bytes(model.string(), "", nullptr) !=
          mm::measure_model_bytes(other.string(), "", nullptr));

    CHECK(mm::bytes_to_mb(0) == 0);
    CHECK(mm::bytes_to_mb(1) == 1);            // rounds UP, never to zero
    CHECK(mm::bytes_to_mb(9 * 1024 * 1024) == 9);

    CHECK(remove_tree(dir));
    return true;
}

bool test_model_registry_makes_soma_routable() {
    auto dir = temp_test_dir("control-db");
    mm::ControlModelRegistry reg;
    std::string err;
    CHECK(reg.open(dir.string(), err));
    // v2 added the conformance table's third state. Asserted rather than
    // ranged: a migration that did not run is the failure this catches.
    CHECK(reg.schema_version() == 2);
    CHECK(reg.list().empty());
    CHECK(std::filesystem::exists(dir / "control.db"));

    mm::NodeRegistry nodes((dir / "nodes").string());
    mm::AgentScheduler scheduler(nodes, (dir / "models").string());

    mm::AgentConfig cfg;
    cfg.id = "agent-soma";
    cfg.name = "Soma Agent";
    cfg.model_path = "Qwen/Qwen3-30B-A3B";

    // ── before: nothing admitted it, so nothing routes to Soma ───────────────
    scheduler.set_model_registry(&reg);
    CHECK(scheduler.resolve_backend_for(cfg).engine_id == "llama-cpp");
    // And with no record, the agent's own path IS the location — this is the
    // fallback's GGUF, and it must keep passing through untouched.
    CHECK(scheduler.model_location(cfg) == cfg.model_path);
    auto forced = cfg;
    forced.backend_override = "soma";
    CHECK(scheduler.resolve_backend_for(forced).engine_id == "llama-cpp");

    // ── admit it with a stream verdict ───────────────────────────────────────
    mm::AdmittedModel m;
    m.arch_hash = std::string(64, 'a');
    m.name = "Qwen3-30B-A3B";
    // The quantization suffix is the POINT, not decoration. With
    // `model_dir == "/containers/" + name` the record's location and the agent's
    // model_path are the same string, and every assertion below passes whichever
    // one the scheduler happens to use — which is exactly how defect D7 hid.
    m.model_dir = "/containers/Qwen3-30B-A3B-q4_g-q6_g-g128";
    m.attention_family = "gqa";
    m.n_layers = 48; m.n_moe_layers = 48; m.n_experts = 128; m.top_k = 8;
    m.bytes_per_token = 1098ll * 1024 * 1024;
    m.total_routed_bytes = 17ll * 1024 * 1024 * 1024;
    m.active_fraction = 0.0625;
    m.verdict = mm::ModelVerdict::Stream;
    CHECK(reg.upsert(m, err));
    CHECK(m.id > 0);

    // The agent's model_path is "Qwen/Qwen3-30B-A3B" and the record's name is
    // "Qwen3-30B-A3B": resolution compares the trailing component, so an agent
    // configured with an HF-style ref matches a record admitted from a directory.
    CHECK(reg.resolve("Qwen/Qwen3-30B-A3B").has_value());
    CHECK(reg.resolve("/some/where/qwen3-30b-a3b/").has_value());   // case + trailing slash
    CHECK(!reg.resolve("Mistral-7B").has_value());
    CHECK(!reg.resolve("").has_value());

    // ── the model's LOCATION comes from the record, not the agent ────────────
    //
    // Defect D7: placement resolved the record for `arch_hash` and `verdict` and
    // threw `model_dir` away, then handed the node the agent's model_path. The
    // node resolves what it is handed against its own models_dir and found no
    // such directory — `model file not found on this node: OLMoE-1B-7B-0924`
    // while `OLMoE-1B-7B-0924-q4_g-q6_g-g128` sat right there.
    CHECK(scheduler.model_location(cfg) == "/containers/Qwen3-30B-A3B-q4_g-q6_g-g128");
    CHECK(scheduler.model_location(cfg) != cfg.model_path);

    // A model nobody admitted still passes its own path through.
    auto unknown = cfg;
    unknown.model_path = "/models/mixtral-8x7b-q4.gguf";
    CHECK(scheduler.model_location(unknown) == "/models/mixtral-8x7b-q4.gguf");

    // ── after: the SAME agent config now routes to Soma, unprompted ──────────
    CHECK(scheduler.resolve_backend_for(cfg).engine_id == "soma");
    CHECK(scheduler.resolve_backend_for(cfg).reason.find("verdict") != std::string::npos);

    // An explicit fallback override still wins: it can only be more conservative.
    auto pinned = cfg;
    pinned.backend_override = "fallback";
    CHECK(scheduler.resolve_backend_for(pinned).engine_id == "llama-cpp");

    // resident-only means it fits and streaming buys nothing → fallback.
    CHECK(reg.set_verdict(m.id, mm::ModelVerdict::ResidentOnly, "fits in RAM", err));
    CHECK(scheduler.resolve_backend_for(cfg).engine_id == "llama-cpp");
    // ...but an operator may override THAT, because it is an economics call.
    CHECK(scheduler.resolve_backend_for(forced).engine_id == "soma");

    // reject means it FAILED CONFORMANCE, and the override is refused. This is
    // the one asymmetry in the policy: no config flag turns "produces wrong
    // tokens" into "serve it anyway".
    CHECK(reg.set_verdict(m.id, mm::ModelVerdict::Reject, "stage 2 divergence", err));
    CHECK(scheduler.resolve_backend_for(cfg).engine_id == "llama-cpp");
    CHECK(scheduler.resolve_backend_for(forced).engine_id == "llama-cpp");
    CHECK(scheduler.resolve_backend_for(forced).reason.find("refused") != std::string::npos);

    // An unparseable verdict must not become a licence to stream.
    CHECK(mm::parse_verdict("nonsense") == mm::ModelVerdict::Reject);
    CHECK(mm::parse_verdict("STREAM") == mm::ModelVerdict::Stream);
    CHECK(mm::verdict_selects_soma(mm::ModelVerdict::Hybrid));
    CHECK(!mm::verdict_selects_soma(mm::ModelVerdict::ResidentOnly));

    // ── persistence, and re-admission updating rather than duplicating ───────
    CHECK(reg.set_verdict(m.id, mm::ModelVerdict::Stream, "re-profiled", err));
    mm::AdmittedModel again = m;
    again.name = "Qwen3-30B-A3B (requantized labels)";
    again.verdict = mm::ModelVerdict::Hybrid;
    CHECK(reg.upsert(again, err));
    CHECK(again.id == m.id);          // same arch_hash is the same model
    CHECK(reg.list().size() == 1);

    reg.close();
    mm::ControlModelRegistry reopened;
    CHECK(reopened.open(dir.string(), err));
    const auto rows = reopened.list();
    CHECK(rows.size() == 1);
    CHECK(rows[0].verdict == mm::ModelVerdict::Hybrid);
    CHECK(rows[0].arch_hash == m.arch_hash);
    CHECK(rows[0].top_k == 8);

    // Removal, and a scheduler with no registry at all.
    CHECK(reopened.remove(rows[0].id, err));
    CHECK(reopened.list().empty());
    CHECK(!reopened.remove(rows[0].id, err));   // gone stays gone, and says so

    mm::AgentScheduler bare(nodes, (dir / "models").string());
    CHECK(bare.resolve_backend_for(cfg).engine_id == "llama-cpp");

    reopened.close();
    CHECK(remove_tree(dir));
    return true;
}

bool test_admission_pipeline_runs_and_reports() {
    // Drives the REAL pipeline against the real `soma` binary — no mock. What is
    // under test is orchestration: staged progress, a terminal frame, cancel, and
    // the registry row at the end. Conversion is skipped by pointing at a
    // container that already exists, which is exactly what reprofile() does.
    const char* soma_path = std::getenv("MM_TEST_SOMA_PATH");
    const char* model_dir = std::getenv("MM_TEST_MODEL_DIR");
    if (soma_path == nullptr || model_dir == nullptr) {
        // Skipped rather than silently passing: CTest passes both, and a
        // developer running the binary by hand should be told why this is quiet.
        std::cout << "  (skipped: MM_TEST_SOMA_PATH / MM_TEST_MODEL_DIR unset)\n";
        return true;
    }

    auto dir = temp_test_dir("admission");
    mm::ControlModelRegistry reg;
    std::string err;
    CHECK(reg.open(dir.string(), err));

    mm::AdmissionTools tools;
    tools.soma_path = soma_path;
    tools.containers_dir = (dir / "containers").string();
    reg.set_tools(tools);

    // ── a source that does not exist fails BEFORE an operation exists ────────
    CHECK(reg.admit((dir / "nope").string(), nullptr, err).empty());
    CHECK(!err.empty());
    CHECK(reg.operations().empty());   // nothing was started, so nothing is listed

    // ── the real thing ───────────────────────────────────────────────────────
    std::mutex m;
    std::condition_variable cv;
    std::vector<mm::AdmissionProgress> frames;
    bool finished = false;

    const auto op = reg.admit_container(model_dir, [&](const mm::AdmissionProgress& p) {
        std::lock_guard<std::mutex> lk(m);
        frames.push_back(p);
        if (p.done) { finished = true; cv.notify_all(); }
    }, err);
    CHECK(!op.empty());

    {
        std::unique_lock<std::mutex> lk(m);
        CHECK(cv.wait_for(lk, std::chrono::seconds(120), [&] { return finished; }));
    }

    CHECK(!frames.empty());
    const auto& last = frames.back();
    CHECK(last.done);
    CHECK(last.operation_id == op);
    // A terminal frame ALWAYS arrives. A stream that just goes quiet is
    // indistinguishable from a network fault, which is why `done` is a field and
    // not the absence of further frames.
    CHECK(last.finished_at_ms >= last.started_at_ms);

    // The tiny fixture is a raw HF checkpoint, not a converted container, so it
    // carries no arch_hash — and without an identity there is nothing to key a
    // row on. Refused with that reason rather than recorded under an empty hash,
    // which would collide with every other unconverted model.
    if (!last.last_error.empty()) {
        CHECK(last.last_error.find("arch_hash") != std::string::npos);
        CHECK(reg.list().empty());
    } else {
        CHECK(last.model_id > 0);
        const auto admitted = reg.find_by_id(last.model_id);
        CHECK(admitted.has_value());
        // Straight from `soma plan --json`, which is why those fields were added
        // to the plan document: control has no other view of the model.
        CHECK(admitted->n_experts > 0);
        CHECK(admitted->top_k > 0);
        CHECK(admitted->active_fraction > 0.0);
        CHECK(!admitted->attention_family.empty());
    }

    // Progress is STAGED, and the operation is retrievable after it ends — an
    // SSE connection will not survive a real conversion, so a client that
    // reconnects has to be able to find out how it went.
    bool saw_profile = false, saw_finalize = false;
    for (const auto& f : frames) {
        if (f.stage == "profile") saw_profile = true;
        if (f.stage == "finalize") saw_finalize = true;
    }
    CHECK(saw_profile);
    CHECK(saw_finalize);
    CHECK(reg.operation(op).has_value());
    CHECK(reg.operation(op)->done);
    CHECK(reg.operations().size() == 1);
    CHECK(!reg.operation("no-such-operation").has_value());

    // Cancelling a finished operation is refused rather than silently accepted:
    // "too late" and "never existed" are both false, but only one is confusing.
    CHECK(!reg.cancel(op));
    CHECK(!reg.cancel("no-such-operation"));

    // A late watcher still gets the outcome, immediately, and is not registered
    // as a sink for a stream that is over.
    mm::AdmissionProgress snapshot;
    CHECK(reg.attach_sink(op, [](const mm::AdmissionProgress&) {}, snapshot));
    CHECK(snapshot.done);
    CHECK(snapshot.operation_id == op);

    reg.close();
    CHECK(remove_tree(dir));
    return true;
}

bool test_admission_fetch_stage() {
    // The fetch stage is the only one that touches the network, so it is the
    // only one that cannot be driven against the real thing here. What IS under
    // test is everything on this side of the subprocess: the repo-id rule, the
    // progress parsing, the resolved path, and the two ways a fetch can exit 0
    // while having produced nothing.
    //
    // A stub tools/ directory does that. convert.py is stubbed too, so the run
    // reaches the REAL `soma plan --json` on a real container and the whole
    // six-stage path is exercised end to end.

    // ── the repo-id rule, on its own ─────────────────────────────────────────
    //
    // Checked directly rather than only through a failed download: this is what
    // stands between an operator-scoped `source` field and a write outside
    // sources_dir, and "the download failed" is not evidence that it held.
    std::string why;
    CHECK(mm::valid_repo_id("gpt2", why));
    CHECK(mm::valid_repo_id("Qwen/Qwen3-30B-A3B", why));
    CHECK(mm::valid_repo_id("org/model@refs/pr/1", why));
    CHECK(mm::valid_repo_id("a.b/c-d_e", why));
    CHECK(!mm::valid_repo_id("", why));
    CHECK(!mm::valid_repo_id("../../etc/passwd", why));
    CHECK(!mm::valid_repo_id("org/../../x", why));
    CHECK(!mm::valid_repo_id("a/b/c", why));       // more than one component
    CHECK(!mm::valid_repo_id("C:\\weights", why)); // a Windows path is not a repo id
    CHECK(!mm::valid_repo_id("org/model@../evil", why));
    CHECK(!mm::valid_repo_id("-leading-dash/x", why));

    const char* python = std::getenv("MM_TEST_PYTHON");
    const char* soma_path = std::getenv("MM_TEST_SOMA_PATH");
    // The CONTAINER, not the raw checkpoint: the stubbed convert stands in for
    // the real one, so what it produces has to be what the real one produces.
    // Otherwise the conformance stage downstream has no container to read, and
    // skips for a reason the test invented rather than the pipeline.
    const char* model_dir = std::getenv("MM_TEST_CONTAINER_DIR");
    if (python == nullptr || soma_path == nullptr || model_dir == nullptr) {
        std::cout << "  (stage skipped: MM_TEST_PYTHON / SOMA_PATH / CONTAINER_DIR unset)\n";
        return true;
    }

    auto dir = temp_test_dir("admission-fetch");
    const auto tools_dir = dir / "tools";
    std::filesystem::create_directories(tools_dir);

    // The stub. Emits exactly the line protocol fetch.py promises, writes a
    // directory, and can be told to fail in each of the interesting ways.
    {
        std::ofstream f(tools_dir / "fetch.py", std::ios::binary);
        f << "import os, sys\n"
             "repo = sys.argv[1]\n"
             "out = sys.argv[sys.argv.index('--out') + 1]\n"
             "mode = os.environ.get('MM_STUB_FETCH', 'ok')\n"
             "if mode == 'fail':\n"
             "    print('cannot read %s: 401 Unauthorized' % repo, flush=True)\n"
             "    sys.exit(3)\n"
             "print('manifest 3 4096', flush=True)\n"
             "print('progress 1024 4096', flush=True)\n"
             "print('progress 4096 4096', flush=True)\n"
             // Exits 0 having produced no directory.
             "if mode == 'silent':\n"
             "    sys.exit(0)\n"
             "os.makedirs(out, exist_ok=True)\n"
             // A real config, copied from the fixture: what a fetch produces has
             // to be plannable, because the architecture check runs on it before
             // conversion is allowed to start.
             "import shutil\n"
             "shutil.copy(os.path.join(os.environ['MM_STUB_CONTAINER'], 'config.json'), out)\n"
             "print('resolved ' + os.path.abspath(out), flush=True)\n";
    }
    // Conversion, stubbed: copy the real fixture so `soma plan` has something
    // true to read. Stubbing the planner too would leave nothing under test.
    {
        std::ofstream f(tools_dir / "convert.py", std::ios::binary);
        f << "import shutil, sys, os\n"
             "src = os.environ['MM_STUB_CONTAINER']\n"
             "out = sys.argv[sys.argv.index('--out') + 1]\n"
             "shutil.rmtree(out, ignore_errors=True)\n"
             "shutil.copytree(src, out)\n"
             "print('    layer 1/1  0.00 GB', flush=True)\n";
        std::ofstream t(tools_dir / "compile_tokenizer.py", std::ios::binary);
        t << "print('stub tokenizer', flush=True)\n";
        // Stands in for the real oracle builder, which needs torch and must not
        // become a dependency of this suite. It writes the directory SHAPE
        // make_oracle.py produces — <out>/<model-name>/ — so the pipeline's
        // lift-into-place step is exercised rather than silently skipped by a
        // missing script.
        std::ofstream o(tools_dir / "make_oracle.py", std::ios::binary);
        o << "import os, sys\n"
             "out = sys.argv[sys.argv.index('--out') + 1]\n"
             "d = os.path.join(out, 'stub-fixture')\n"
             "os.makedirs(d, exist_ok=True)\n"
             "open(os.path.join(d, 'oracle.bin'), 'wb').write(b'SOMAORCL')\n"
             "print('stub oracle', flush=True)\n";
        // Same reasoning for the bf16 reference, which needs torch AND enough
        // RAM to hold a real checkpoint. It writes <out>/oracle.bin flat, which
        // is the shape make_reference.py produces, so the pipeline's
        // rename-to-reference.bin step is exercised rather than skipped by a
        // missing script.
        std::ofstream mr(tools_dir / "make_reference.py", std::ios::binary);
        mr << "import os, sys\n"
              "out = sys.argv[sys.argv.index('--out') + 1]\n"
              "os.makedirs(out, exist_ok=True)\n"
              "open(os.path.join(out, 'oracle.bin'), 'wb').write(b'SOMAORCL')\n"
              "print('stub reference', flush=True)\n";
    }

    mm::ControlModelRegistry reg;
    std::string err;
    CHECK(reg.open(dir.string(), err));

    mm::AdmissionTools tools;
    tools.python = python;
    tools.soma_path = soma_path;
    tools.tools_dir = tools_dir.string();
    tools.containers_dir = (dir / "containers").string();
    tools.sources_dir = (dir / "sources").string();
    reg.set_tools(tools);

#ifdef _WIN32
    _putenv_s("MM_STUB_CONTAINER", model_dir);
#else
    setenv("MM_STUB_CONTAINER", model_dir, 1);
#endif

    // ── a source that is neither a directory nor a repo id ───────────────────
    //
    // Refused before an operation exists, so a typo is a 400 rather than an
    // operation that appears to start and dies a second later.
    CHECK(reg.admit("org/../../escape", nullptr, err).empty());
    CHECK(!err.empty());
    CHECK(reg.operations().empty());

    struct Run {
        std::vector<mm::AdmissionProgress> frames;
        std::mutex m;
        std::condition_variable cv;
        bool finished = false;
    };
    const auto drive = [&](const char* mode, const std::string& source) {
#ifdef _WIN32
        _putenv_s("MM_STUB_FETCH", mode);
#else
        setenv("MM_STUB_FETCH", mode, 1);
#endif
        auto run = std::make_shared<Run>();
        std::string e;
        const auto id = reg.admit(source, [run](const mm::AdmissionProgress& p) {
            std::lock_guard<std::mutex> lk(run->m);
            run->frames.push_back(p);
            if (p.done) {
                run->finished = true;
                run->cv.notify_all();
            }
        }, e);
        if (!id.empty()) {
            std::unique_lock<std::mutex> lk(run->m);
            (void)run->cv.wait_for(lk, std::chrono::seconds(120), [&] { return run->finished; });
        }
        return run;
    };

    // ── the happy path ───────────────────────────────────────────────────────
    const auto good = drive("ok", "fake-org/Qwen3-30B-A3B");
    CHECK(!good->frames.empty());
    CHECK(good->finished);

    // Every stage, in order, and `step` consistent with `total_steps`. Those two
    // used to be written independently: a container admission advertised 2 total
    // steps and then emitted step 5, which renders as 250%.
    //  builds the tiny-random conformance fixture that turns ladder
    // stage 1 from "skipped" into an answer.
    // "reference" is the bf16 pass over the REAL checkpoint — the counterpart to
    // "oracle", and the stage that turns ladder stage 2 from "skipped" into an
    // answer. Both are here because a pipeline that silently stopped running
    // either would leave the ladder reporting fewer stages while still saying
    // "no failures", which reads as a pass.
    const std::vector<std::string> want{"fetch",     "convert",     "tokenize",
                                        "oracle",    "reference",   "profile",
                                        "conformance", "finalize"};
    std::size_t at = 0;
    std::int64_t peak_bytes = 0, peak_total = 0;
    for (const auto& f : good->frames) {
        CHECK(f.step >= 1 && f.step <= f.total_steps);
        CHECK(f.total_steps == static_cast<int>(want.size()));
        if (at < want.size() && f.stage == want[at]) ++at;
        peak_bytes = std::max(peak_bytes, f.bytes_done);
        peak_total = std::max(peak_total, f.bytes_total);
    }
    CHECK(at == want.size()); // all six seen, in order

    // The byte counters are the reason fetch is worth a stage of its own: it is
    // the only one whose remaining time a client can estimate.
    CHECK(peak_bytes == 4096);
    CHECK(peak_total == 4096);

    const auto& done = good->frames.back();
    CHECK(done.done);
    CHECK(done.last_error.empty());
    CHECK(done.model_id > 0);

    const auto admitted = reg.find_by_id(done.model_id);
    CHECK(admitted.has_value());
    // The trailing component, so `fake-org/Qwen3-30B-A3B` and a local directory
    // of the same name are ONE model rather than two rows for one set of weights.
    CHECK(admitted->name.find("Qwen3-30B-A3B") != std::string::npos);
    CHECK(!admitted->arch_hash.empty());
    // The fetched directory is where it was told to put it, not somewhere the
    // repo id chose.
    CHECK(std::filesystem::exists(dir / "sources" / "Qwen3-30B-A3B" / "config.json"));

    // ── a fetch that fails ───────────────────────────────────────────────────
    const auto failed = drive("fail", "fake-org/unauthorized-model");
    CHECK(!failed->frames.empty());
    CHECK(failed->frames.back().done);
    CHECK(!failed->frames.back().last_error.empty());
    CHECK(failed->frames.back().last_error.find("fetch.py") != std::string::npos);
    CHECK(failed->frames.back().model_id == 0);

    // ── a fetch that exits 0 having produced nothing ─────────────────────────
    //
    // The failure this guards is a confusing one: conversion would be handed a
    // path that is not there and the error would name convert.py.
    const auto silent = drive("silent", "fake-org/silent-model");
    CHECK(!silent->frames.empty());
    CHECK(silent->frames.back().done);
    CHECK(silent->frames.back().last_error.find("no directory") != std::string::npos);

    // One record from three admissions: two of them never got far enough to
    // write one, and neither left a half-row behind.
    CHECK(reg.list().size() == 1);

    // ── the ladder ran, and said what it did not do ──────────────────────────
    //
    // `soma conform` is invoked for real here — the stubs cover fetch and
    // convert, not this. What matters is that the SKIPPED stages are recorded as
    // skipped: a pipeline that wrote `passed` rows for stages needing a
    // transformers oracle would hand every model a verdict that looks validated.
    {
        const auto stages = reg.conformance(done.model_id);
        CHECK(!stages.empty());
        std::size_t skipped = 0, passed = 0;
        bool saw_reason = false;
        for (const auto& s : stages) {
            CHECK(s.status == "passed" || s.status == "failed" || s.status == "skipped");
            CHECK(s.passed == (s.status == "passed"));
            if (s.status == "skipped") {
                ++skipped;
                // A skip without a reason is indistinguishable from a stage
                // nobody wrote.
                if (s.detail.find("reason") != std::string::npos) saw_reason = true;
            }
            if (s.status == "passed") ++passed;
        }
        CHECK(skipped >= 3); // fp32_tiny_tf, real_logit_kl, accuracy_floor
        CHECK(passed >= 1);  // quant_codec, against the real container
        CHECK(saw_reason);
    }

    // ── an unsupported architecture fails BEFORE conversion ─────────────────
    //
    // The G8 gate line, and the difference between failing in 200 ms and failing
    // after six hours. `adapt_hf_config`'s table IS the registry of architectures
    // this engine understands; an unknown `model_type` stops there, and a
    // container built from a config Soma cannot parse is gigabytes nothing can
    // read.
    //
    // Proven by the stub convert never running: it writes a container, so if the
    // check happened afterwards the directory would exist.
    {
        const auto weird = dir / "weird-arch";
        std::filesystem::create_directories(weird);
        {
            std::ofstream f(weird / "config.json", std::ios::binary);
            f << R"({"model_type":"nonexistent_moe","num_hidden_layers":4,"hidden_size":64})";
        }
        auto run = std::make_shared<Run>();
        std::string e;
        const auto id = reg.admit(weird.string(), [run](const mm::AdmissionProgress& p) {
            std::lock_guard<std::mutex> lk(run->m);
            run->frames.push_back(p);
            if (p.done) {
                run->finished = true;
                run->cv.notify_all();
            }
        }, e);
        CHECK(!id.empty()); // it is a real directory, so the operation starts
        {
            std::unique_lock<std::mutex> lk(run->m);
            (void)run->cv.wait_for(lk, std::chrono::seconds(60), [&] { return run->finished; });
        }
        CHECK(run->finished);
        const auto& fin = run->frames.back();
        CHECK(!fin.last_error.empty());
        CHECK(fin.last_error.find("not supported") != std::string::npos);
        CHECK(fin.model_id == 0);

        // Conversion never started. The stub convert copies a whole container, so
        // its absence is what proves the check ran first rather than after.
        CHECK(!std::filesystem::exists(dir / "containers" / "weird-arch"));
        bool converted = false;
        for (const auto& f : run->frames) {
            if (f.stage == "tokenize") converted = true;
        }
        CHECK(!converted);
    }

    // ── a failed stage rejects the model rather than the request ─────────────
    //
    // The operator asked whether Soma can run this; "no, and here is the stage
    // that says so" is an answer. A rejected model is a successfully admitted
    // RECORD meaning "route this to the fallback", so the admission must SUCCEED
    // and the verdict must be reject — not the other way around.
    {
        const auto broken = dir / "broken-container";
        std::filesystem::copy(model_dir, broken,
                              std::filesystem::copy_options::recursive |
                                  std::filesystem::copy_options::overwrite_existing);
        // A tokenizer paired with ANOTHER model's oracle. Realistic — it is what
        // a mismatched compile step produces — and it fails one conformance stage
        // while leaving the container plannable, which is the combination this
        // case needs.
        //
        // Corrupting container_meta.json would not do: the planner reads it now,
        // because the quantization is part of arch_hash, so an unreadable one
        // fails the admission outright rather than one stage of the ladder.
        const auto tok_dir = std::filesystem::path(model_dir).parent_path().parent_path() /
                             "tokenizers";
        if (!std::filesystem::exists(tok_dir / "Qwen3-30B-A3B" / "tokenizer.soma") ||
            !std::filesystem::exists(tok_dir / "OLMoE-1B-7B-0924" / "tokenizer_oracle.bin")) {
            std::cout << "  (verdict-downgrade case skipped: tokenizer fixtures not found)\n";
            reg.close();
            CHECK(remove_tree(dir));
            return true;
        }
        std::filesystem::copy_file(tok_dir / "Qwen3-30B-A3B" / "tokenizer.soma",
                                   broken / "tokenizer.soma");
        std::filesystem::copy_file(tok_dir / "OLMoE-1B-7B-0924" / "tokenizer_oracle.bin",
                                   broken / "tokenizer_oracle.bin");

        auto run = std::make_shared<Run>();
        std::string e;
        const auto id = reg.admit_container(broken.string(), [run](const mm::AdmissionProgress& p) {
            std::lock_guard<std::mutex> lk(run->m);
            run->frames.push_back(p);
            if (p.done) {
                run->finished = true;
                run->cv.notify_all();
            }
        }, e);
        CHECK(!id.empty());
        {
            std::unique_lock<std::mutex> lk(run->m);
            (void)run->cv.wait_for(lk, std::chrono::seconds(120), [&] { return run->finished; });
        }
        CHECK(run->finished);
        const auto& fin = run->frames.back();
        CHECK(fin.last_error.empty());  // the REQUEST succeeded
        CHECK(fin.model_id > 0);

        const auto rec = reg.find_by_id(fin.model_id);
        CHECK(rec.has_value());
        CHECK(rec->verdict == mm::ModelVerdict::Reject);
        CHECK(rec->verdict_reason.find("conformance") != std::string::npos);
        CHECK(!mm::verdict_selects_soma(rec->verdict)); // so it routes to the fallback

        bool saw_failed = false;
        for (const auto& s : reg.conformance(fin.model_id)) {
            if (s.stage == "tokenizer_roundtrip" && s.status == "failed") saw_failed = true;
        }
        CHECK(saw_failed);
    }

    reg.close();
    CHECK(remove_tree(dir));
    return true;
}

bool test_requantization_is_a_new_admission() {
    // The G8 gate line, and it has two halves that pull opposite ways:
    //
    //   re-admitting at a different quantization MUST produce a new arch_hash
    //   and invalidate KV checkpoints;
    //   re-PROFILING must NOT.
    //
    // Getting either one alone is easy. A hash over the whole container makes
    // the first true and the second false — every reprofile would orphan every
    // checkpoint. A hash over the architecture only makes the second true and
    // the first false — two quantizations share one identity and a checkpoint
    // written under one replays under the other, fluently and wrongly.

    // ── the hash covers the quantization, all of it ──────────────────────────
    //
    // Checked on the IR directly rather than only through the pipeline: this is
    // the property everything else here depends on, and a pipeline test that
    // happened to pass would not say which field made it pass.
    soma::ArchIr ir;
    ir.schema_version = soma::kArchIrSchemaVersion;
    ir.attention.family = soma::AttentionFamily::Gqa;
    ir.topology.n_layers = 4;
    ir.topology.d_model = 64;
    ir.topology.vocab_size = 512;
    ir.topology.layer_kinds.assign(4, soma::LayerKind::Moe);
    ir.attention.n_heads = 4;
    ir.attention.n_kv_heads = 2;
    ir.attention.head_dim = 16;
    ir.router.n_experts = 16;
    ir.router.top_k = 2;
    ir.quantization.expert_gate = {soma::DType::Q4_G, 128};
    ir.quantization.expert_up = {soma::DType::Q4_G, 128};
    ir.quantization.expert_down = {soma::DType::Q6_G, 128};

    std::string base;
    CHECK(soma::compute_arch_hash(ir, base).ok());
    CHECK(base.size() == 64);

    {
        // Same dtypes, different GROUP. These dequantize to different weights,
        // and before this they hashed identically — so a KV checkpoint written
        // at group 128 would replay under group 64 with nothing detecting it.
        auto other = ir;
        other.quantization.expert_gate.group = 64;
        std::string h;
        CHECK(soma::compute_arch_hash(other, h).ok());
        CHECK(h != base);
    }
    {
        auto other = ir;
        other.quantization.expert_down.dtype = soma::DType::Q8_0;
        std::string h;
        CHECK(soma::compute_arch_hash(other, h).ok());
        CHECK(h != base);
    }
    {
        // A role the original hash did not cover at all.
        auto other = ir;
        other.quantization.shared_expert = {soma::DType::Q8_0, 32};
        std::string h;
        CHECK(soma::compute_arch_hash(other, h).ok());
        CHECK(h != base);
    }
    {
        // And the other half: a MEASUREMENT changing must not move the hash.
        // Economics is deliberately outside it — re-profiling on a faster disk
        // would otherwise orphan every checkpoint written before the upgrade.
        auto other = ir;
        other.economics.measured_disk_bw = 4ull * 1000 * 1000 * 1000;
        other.economics.expert_bytes = 999999;
        other.economics.measured_at_host = "some-other-box";
        std::string h;
        CHECK(soma::compute_arch_hash(other, h).ok());
        CHECK(h == base);
    }

    // ── and the pipeline honours it ──────────────────────────────────────────
    const char* python = std::getenv("MM_TEST_PYTHON");
    const char* soma_path = std::getenv("MM_TEST_SOMA_PATH");
    const char* container_dir = std::getenv("MM_TEST_CONTAINER_DIR");
    if (python == nullptr || soma_path == nullptr || container_dir == nullptr) {
        std::cout << "  (pipeline half skipped: MM_TEST_PYTHON / SOMA_PATH / CONTAINER_DIR unset)\n";
        return true;
    }

    auto dir = temp_test_dir("requant");
    const auto tools_dir = dir / "tools";
    std::filesystem::create_directories(tools_dir);

    // A stub convert that RESPECTS --quant, so the two admissions differ in the
    // way a real conversion would. It copies the fixture and rewrites the
    // container's declared quantization; that is exactly the field `soma plan`
    // reads to build the IR the hash is taken over.
    {
        std::ofstream f(tools_dir / "convert.py", std::ios::binary);
        f << "import json, os, shutil, sys\n"
             "src = os.environ['MM_STUB_CONTAINER']\n"
             "out = sys.argv[sys.argv.index('--out') + 1]\n"
             "quant = sys.argv[sys.argv.index('--quant') + 1]\n"
             "down = sys.argv[sys.argv.index('--expert-down') + 1]\n"
             "group = int(sys.argv[sys.argv.index('--group') + 1])\n"
             "shutil.rmtree(out, ignore_errors=True)\n"
             "shutil.copytree(src, out)\n"
             "p = os.path.join(out, 'container_meta.json')\n"
             "m = json.load(open(p))\n"
             "m['dtype_gate_up'] = quant\n"
             "m['dtype_down'] = down\n"
             "m['group'] = group\n"
             "json.dump(m, open(p, 'w'))\n"
             "print('    layer 1/1  0.00 GB', flush=True)\n";
        std::ofstream t(tools_dir / "compile_tokenizer.py", std::ios::binary);
        t << "print('stub tokenizer', flush=True)\n";
        // Stands in for the real oracle builder, which needs torch and must not
        // become a dependency of this suite. It writes the directory SHAPE
        // make_oracle.py produces — <out>/<model-name>/ — so the pipeline's
        // lift-into-place step is exercised rather than silently skipped by a
        // missing script.
        std::ofstream o(tools_dir / "make_oracle.py", std::ios::binary);
        o << "import os, sys\n"
             "out = sys.argv[sys.argv.index('--out') + 1]\n"
             "d = os.path.join(out, 'stub-fixture')\n"
             "os.makedirs(d, exist_ok=True)\n"
             "open(os.path.join(d, 'oracle.bin'), 'wb').write(b'SOMAORCL')\n"
             "print('stub oracle', flush=True)\n";
        // Same reasoning for the bf16 reference, which needs torch AND enough
        // RAM to hold a real checkpoint. It writes <out>/oracle.bin flat, which
        // is the shape make_reference.py produces, so the pipeline's
        // rename-to-reference.bin step is exercised rather than skipped by a
        // missing script.
        std::ofstream mr(tools_dir / "make_reference.py", std::ios::binary);
        mr << "import os, sys\n"
              "out = sys.argv[sys.argv.index('--out') + 1]\n"
              "os.makedirs(out, exist_ok=True)\n"
              "open(os.path.join(out, 'oracle.bin'), 'wb').write(b'SOMAORCL')\n"
              "print('stub reference', flush=True)\n";
    }

    mm::ControlModelRegistry reg;
    std::string err;
    CHECK(reg.open(dir.string(), err));

    mm::AdmissionTools tools;
    tools.python = python;
    tools.soma_path = soma_path;
    tools.tools_dir = tools_dir.string();
    tools.containers_dir = (dir / "containers").string();
    tools.sources_dir = (dir / "sources").string();
    reg.set_tools(tools);

#ifdef _WIN32
    _putenv_s("MM_STUB_CONTAINER", container_dir);
#else
    setenv("MM_STUB_CONTAINER", container_dir, 1);
#endif

    // The source is a raw config the architecture check can read.
    const auto source = dir / "weights";
    std::filesystem::create_directories(source);
    std::filesystem::copy_file(std::filesystem::path(container_dir) / "config.json",
                               source / "config.json");

    struct Run {
        std::vector<mm::AdmissionProgress> frames;
        std::mutex m;
        std::condition_variable cv;
        bool finished = false;
    };
    const auto admit_at = [&](const mm::ControlModelRegistry::QuantOverride& q) {
        auto run = std::make_shared<Run>();
        std::string e;
        const auto id = reg.admit(source.string(), q, [run](const mm::AdmissionProgress& p) {
            std::lock_guard<std::mutex> lk(run->m);
            run->frames.push_back(p);
            if (p.done) {
                run->finished = true;
                run->cv.notify_all();
            }
        }, e);
        if (!id.empty()) {
            std::unique_lock<std::mutex> lk(run->m);
            (void)run->cv.wait_for(lk, std::chrono::seconds(120), [&] { return run->finished; });
        }
        return run;
    };

    const auto first = admit_at({"q4_g", "q6_g", 128});
    CHECK(first->finished);
    CHECK(first->frames.back().last_error.empty());
    const auto id_a = first->frames.back().model_id;
    CHECK(id_a > 0);

    const auto second = admit_at({"q8_0", "q8_0", 32});
    CHECK(second->finished);
    CHECK(second->frames.back().last_error.empty());
    const auto id_b = second->frames.back().model_id;
    CHECK(id_b > 0);

    // TWO rows, not one updated in place. The registry keys on arch_hash, so
    // this is the observable form of "a different quantization is a different
    // model".
    CHECK(id_a != id_b);
    CHECK(reg.list().size() == 2);

    const auto a = reg.find_by_id(id_a);
    const auto b = reg.find_by_id(id_b);
    CHECK(a.has_value() && b.has_value());
    CHECK(!a->arch_hash.empty());
    CHECK(a->arch_hash != b->arch_hash);

    // Separate containers. Sharing one directory meant the second conversion
    // overwrote the first, leaving row A describing bytes that were no longer
    // its quantization — with nothing to detect it.
    CHECK(a->model_dir != b->model_dir);
    CHECK(std::filesystem::exists(a->model_dir));
    CHECK(std::filesystem::exists(b->model_dir));

    // ── the checkpoint half ──────────────────────────────────────────────────
    //
    // "Invalidates KV checkpoints" is the consequence that matters, and it is
    // checked against the real store rather than inferred from the hashes being
    // different. A resume gated on a hash nobody compares is not a gate.
    {
        auto ckpt_dir = dir / "kv";
        // The same shape both sides, differing ONLY in arch_hash. That is the
        // point: the cache geometry is identical, so nothing about the bytes
        // would stop the load — the hash is the only thing standing between a
        // q4_g checkpoint and a q8_0 engine.
        auto arch_a = ir;
        arch_a.arch_hash = a->arch_hash;
        auto arch_b = ir;
        arch_b.arch_hash = b->arch_hash;

        soma::KvCheckpointStore store_a;
        CHECK(store_a.open(ckpt_dir.string(), arch_a).ok());
        soma::KvCache kv;
        CHECK(kv.open(arch_a, 64).ok());
        CHECK(kv.set_length(4).ok());
        soma::SeqPersistState st;
        st.tokens = {1, 2, 3, 4};
        CHECK(store_a.save("conv-1", kv, st).ok());
        store_a.close();

        soma::KvCheckpointStore store_b;
        CHECK(store_b.open(ckpt_dir.string(), arch_b).ok());
        soma::KvCache kv2;
        CHECK(kv2.open(arch_b, 64).ok());
        soma::SeqPersistState out;
        const auto load = store_b.load("conv-1", kv2, out);
        CHECK(!load.ok());
        CHECK(load.code() == soma::StatusCode::ArchMismatch);
    }

    // ── and re-profiling does NOT ────────────────────────────────────────────
    //
    // The other half of the gate. A reprofile re-derives a verdict from the same
    // bytes; if it moved the hash it would orphan every checkpoint written
    // against the model, for a request that asked only for a fresh number.
    {
        auto run = std::make_shared<Run>();
        std::string e;
        const auto id = reg.reprofile(id_a, [run](const mm::AdmissionProgress& p) {
            std::lock_guard<std::mutex> lk(run->m);
            run->frames.push_back(p);
            if (p.done) {
                run->finished = true;
                run->cv.notify_all();
            }
        }, e);
        CHECK(!id.empty());
        {
            std::unique_lock<std::mutex> lk(run->m);
            (void)run->cv.wait_for(lk, std::chrono::seconds(120), [&] { return run->finished; });
        }
        CHECK(run->finished);
        CHECK(run->frames.back().last_error.empty());

        const auto again = reg.find_by_id(id_a);
        CHECK(again.has_value());
        CHECK(again->arch_hash == a->arch_hash);
        CHECK(again->model_dir == a->model_dir);
        // Still two rows: the reprofile updated one, it did not add a third.
        CHECK(reg.list().size() == 2);
    }

    // ── which one an agent gets ──────────────────────────────────────────────
    //
    // Two rows now match the same name, so "the first row the scan reaches" is
    // not an answer — it is whichever the b-tree yielded, and it would change
    // under an unrelated insert. An exact arch_hash is unambiguous and stays so.
    {
        const auto by_hash = reg.resolve(a->arch_hash);
        CHECK(by_hash.has_value());
        CHECK(by_hash->id == id_a);
        const auto by_hash_b = reg.resolve(b->arch_hash);
        CHECK(by_hash_b.has_value());
        CHECK(by_hash_b->id == id_b);

        // By name it is deterministic rather than arbitrary: repeated calls
        // agree, which is the property that was missing.
        const auto once = reg.resolve("weights");
        const auto twice = reg.resolve("weights");
        CHECK(once.has_value() && twice.has_value());
        CHECK(once->id == twice->id);
    }

    reg.close();
    CHECK(remove_tree(dir));
    return true;
}

bool test_scope_negatives_over_http() {
    // The predicate is asserted elsewhere; this asserts the WIRE. A scope table
    // that is right in a unit test and unreached by the middleware protects
    // nothing, and the failure mode is silent — every request succeeds.
    auto dir = temp_test_dir("scope-http");
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

        // The registry has to be attached: without one, has_api_tokens() is
        // false and — with no legacy token either — auth is off entirely, so
        // the scoped path is never reached. That is the exact shape of the bug
        // this test would otherwise miss.
        mm::ControlModelRegistry models;
        std::string err;
        RECORD(models.open(dir.string(), err));

        const auto read_token = models.create_api_token(
            "dashboard", static_cast<mm::ScopeSet>(mm::Scope::Read), err);
        const auto chat_token = models.create_api_token(
            "assistant", static_cast<mm::ScopeSet>(mm::Scope::Chat), err);
        const auto op_token = models.create_api_token(
            "ops", static_cast<mm::ScopeSet>(mm::Scope::Operator), err);
        RECORD(!read_token.empty() && !chat_token.empty() && !op_token.empty());

        // No legacy token: auth is on purely because api_token has rows.
        mm::ControlApiServer api(agents, queue, registry, scheduler, dir.string(),
                                 (dir / "models").string(), /*external_api_token=*/"");
        api.set_model_registry(&models);

        const uint16_t port = find_free_test_port();
        CHECK(port != 0);
        const std::string base_url = "http://127.0.0.1:" + std::to_string(port);
        std::atomic<bool> listen_ok{false};
        std::thread server_thread([&] { listen_ok = api.listen(port); });

        mm::HttpClient probe(base_url);
        bool ready = false;
        for (int i = 0; i < 50 && !ready; ++i) {
            if (probe.get("/v1/nodes").status != 0) ready = true;
            else std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        RECORD(ready);

        auto with_retry = [](auto&& request) {
            mm::HttpResponse resp;
            for (int attempt = 0; attempt < kTransportRetries; ++attempt) {
                resp = request();
                if (resp.status != 0) break;
                transport_backoff(attempt);
            }
            return resp;
        };
        auto as = [&](const std::string& token) {
            auto c = std::make_shared<mm::HttpClient>(base_url);
            c->set_bearer_token(token);
            return c;
        };

        // ── read: GETs yes, everything else no ───────────────────────────────
        auto reader = as(read_token);
        RECORD(with_retry([&] { return reader->get("/v1/nodes"); }).status == 200);
        RECORD(with_retry([&] { return reader->get("/v1/agents"); }).status == 200);
        RECORD(with_retry([&] { return reader->get("/v1/models"); }).status == 200);

        auto reader_admits = with_retry([&] {
            return reader->post("/v1/models/admit", nlohmann::json{{"source", "/nope"}});
        });
        RECORD(reader_admits.status == 403);
        // The body names WHICH scope was needed and which were held. "403
        // forbidden" with no detail sends an operator to read source to find out
        // what to grant.
        RECORD(reader_admits.body.find("insufficient scope") != std::string::npos);
        RECORD(reader_admits.body.find("operator") != std::string::npos);
        RECORD(reader_admits.body.find("read") != std::string::npos);

        RECORD(with_retry([&] { return reader->del("/v1/agents/agent-a"); }).status == 403);
        RECORD(with_retry([&] {
                   return reader->post("/v1/agents/agent-a/conversations",
                                       nlohmann::json{{"title", "nope"}});
               }).status == 403);
        // Still there: the refusal was a refusal, not a 403 after the fact.
        RECORD(agents.get_agent("agent-a") != nullptr);

        // ── chat: conversations yes, admission and deletes no ────────────────
        auto chatter = as(chat_token);
        // chat implies read.
        RECORD(with_retry([&] { return chatter->get("/v1/agents"); }).status == 200);
        auto chat_creates = with_retry([&] {
            return chatter->post("/v1/agents/agent-a/conversations",
                                 nlohmann::json{{"title", "Allowed"}, {"set_active", true}});
        });
        RECORD(chat_creates.status == 200 || chat_creates.status == 201);

        auto chat_admits = with_retry([&] {
            return chatter->post("/v1/models/admit", nlohmann::json{{"source", "/nope"}});
        });
        RECORD(chat_admits.status == 403);
        RECORD(chat_admits.body.find("insufficient scope") != std::string::npos);
        RECORD(with_retry([&] { return chatter->del("/v1/agents/agent-a"); }).status == 403);
        RECORD(with_retry([&] { return chatter->get("/v1/tokens"); }).status == 403);

        // ── operator: everything, including the things that refused above ────
        auto op = as(op_token);
        RECORD(with_retry([&] { return op->get("/v1/agents"); }).status == 200);
        RECORD(with_retry([&] { return op->get("/v1/tokens"); }).status == 200);
        auto op_admits = with_retry([&] {
            return op->post("/v1/models/admit", nlohmann::json{{"source", "/nope"}});
        });
        // 400, not 403: authorization PASSED and the request was then rejected on
        // its merits. That distinction is the whole point — a 403 here would mean
        // the operator scope was not being honoured.
        RECORD(op_admits.status == 400);
        RECORD(op_admits.body.find("not found") != std::string::npos);

        // ── a revoked token stops working immediately ────────────────────────
        const auto rows = models.list_api_tokens();
        std::int64_t chat_id = 0;
        for (const auto& t : rows) {
            if (t.label == "assistant") chat_id = t.id;
        }
        RECORD(chat_id != 0);
        RECORD(models.revoke_api_token(chat_id, err));
        auto revoked = with_retry([&] { return chatter->get("/v1/agents"); });
        RECORD(revoked.status == 403);
        RECORD(revoked.body.find("invalid bearer token") != std::string::npos);

        // ── an unknown token is 403, a missing one is 401 ────────────────────
        RECORD(with_retry([&] { return as("never-minted")->get("/v1/agents"); }).status == 403);
        RECORD(with_retry([&] { return probe.get("/v1/agents"); }).status == 401);

        // Deleting the agent as operator — the mutation the reader and chatter
        // were both refused — proves the route works and the refusals were about
        // the credential rather than the request.
        RECORD(with_retry([&] { return op->del("/v1/agents/agent-a"); }).status == 200);

        api.stop();
        if (server_thread.joinable()) server_thread.join();
        RECORD(listen_ok.load());
        models.close();
    }

    RECORD(remove_tree(dir));
#undef RECORD
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// capacity_pressure: the signal that earns a failed placement a second chance.
//
// This is a G8 criterion that had NO coverage at all — the code was implemented
// across both engines, the node proxy and the scheduler, and nothing asserted
// any of it. What makes that worth fixing is not the happy path, which is one
// string compare, but the PRECEDENCE rule underneath it: a structured code is
// authoritative, so an engine that says `model_not_found` in prose containing
// "out of memory" must NOT be retried. Swap the two blocks in the definition
// and every obvious test still passes.
//
// Also pinned here: the wire shape the node actually emits, which is neither of
// the shapes you would guess. See below.
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// Images are refused by the ENGINE's capability, not only the agent's profile.
//
// The G8 criterion is "image content parts return 422, not a dropped part", and
// it was half-implemented in a way that read as done: all four gates tested
// `vision_settings.enabled` — the operator's INTENT — and none tested whether
// the engine on the far end could accept an image at all. So an agent with
// vision switched on, serving a model that earned a streamable verdict, sent
// image parts to Soma, which is text-only by construction (roadmap D12).
//
// That failure is the bad kind: not a crash, a silently text-only answer to a
// question about a picture.
// ─────────────────────────────────────────────────────────────────────────────
bool test_images_refused_by_engine_capability() {
    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    // ── the capability table ─────────────────────────────────────────────────
    RECORD(mm::engine_supports_vision("llama-cpp"));
    RECORD(!mm::engine_supports_vision("soma"));
    // Unknown engines refuse. A new engine that has said nothing about vision
    // gets a clear refusal the operator can act on; the other default silently
    // drops the image and answers as though it had been read.
    RECORD(!mm::engine_supports_vision("vllm"));
    RECORD(!mm::engine_supports_vision(""));

    // ── the rule ─────────────────────────────────────────────────────────────
    const auto refusal = [](bool enabled, const char* engine, const char* why = "") {
        return mm::image_refusal_for(enabled, engine, why);
    };

    // Profile off is the whole answer, whatever the engine. Naming an engine
    // here would misdirect: the operator's own setting is what to change.
    RECORD(refusal(false, "llama-cpp") == "this agent profile does not accept images");
    RECORD(refusal(false, "soma") == "this agent profile does not accept images");

    // Profile on + an engine that can: allowed. This is the case that must keep
    // working — a capability check that refuses everything would also "fix" D12.
    RECORD(refusal(true, "llama-cpp").empty());

    // THE DEFECT. Profile on, engine cannot, and this used to be allowed.
    const auto blocked = refusal(true, "soma", "soma (verdict=hybrid)");
    RECORD(!blocked.empty());
    RECORD(blocked.find("soma") != std::string::npos);
    RECORD(blocked.find("does not accept images") != std::string::npos);
    // The reason already names the engine, so the message must not ALSO quote
    // the id — that produced "the 'soma' engine ... (soma (verdict=hybrid))",
    // saying it twice inside nested parens (D15's residue).
    RECORD(blocked.find("'soma'") == std::string::npos);
    RECORD(blocked.find("((") == std::string::npos);
    // With no reason to carry the name, the id is quoted — otherwise the message
    // would not say WHICH engine at all.
    RECORD(refusal(true, "soma").find("'soma'") != std::string::npos);
    // The routing reason rides along, so the message says why THAT engine is
    // serving. Without it the text reads as a profile problem and the operator
    // goes to switch on a setting that is already on.
    RECORD(blocked.find("verdict=hybrid") != std::string::npos);
    // An unknown engine is refused on the same rule, not a separate branch.
    RECORD(!refusal(true, "vllm").empty());

    // API-backed agents own no node-local engine; the remote provider's own
    // capabilities govern and control has no business refusing on its behalf.
    RECORD(refusal(true, "").empty());

    // ── the descriptors read the SAME table ──────────────────────────────────
    // This is what keeps the two sides from drifting. Before, `supports_vision`
    // was a literal in each descriptor and control could not see it at all;
    // asserting the node's view against the shared function is what makes the
    // single-source claim true rather than merely intended.
    RECORD(mm::make_soma_descriptor("soma").supports_vision ==
           mm::engine_supports_vision("soma"));
    RECORD(mm::make_llama_descriptor("llama-server").supports_vision ==
           mm::engine_supports_vision("llama-cpp"));

#undef RECORD
    return ok;
}

bool test_capacity_pressure_is_structured() {
    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    const auto pressure = &mm::AgentScheduler::response_indicates_capacity_pressure;

    // ── 1. the two structured shapes, because there are two producers ────────
    // `soma serve` and the node's inference proxy nest it under `error`.
    RECORD(pressure(R"({"error":{"code":"capacity_pressure","message":"no free sequence"}})"));
    // The node's LOAD handlers keep `error` as a human string and carry the code
    // beside it, so existing clients are not broken by the addition. `error` is
    // therefore a STRING here, not an object — a parser that only checks the
    // nested shape misses this entirely.
    RECORD(pressure(
        R"({"error":"failed to load model","code":"capacity_pressure","engine":"soma"})"));

    // ── 2. a structured code is AUTHORITATIVE ────────────────────────────────
    // The whole point of the rewrite. The engine has decided this is not
    // capacity; reading its prose for a contradicting hint would undo that, and
    // an evict-and-retry here would destroy a healthy resident agent to make
    // room for a model that is never going to be found.
    RECORD(!pressure(
        R"({"error":{"code":"model_not_found","message":"ran out of memory looking"}})"));
    RECORD(!pressure(
        R"({"error":"out of memory","code":"model_not_found","engine":"llama-cpp"})"));
    // Same rule, stated for a code nobody has defined yet: unrecognised is not
    // capacity. A future engine inventing `quota_exceeded` gets a hard failure
    // rather than an eviction storm, until someone decides otherwise.
    RECORD(!pressure(R"({"error":{"code":"quota_exceeded","message":"insufficient vram"}})"));

    // ── 3. the legacy substring fallback ─────────────────────────────────────
    // Retained deliberately for a node that predates the code. It is reachable
    // ONLY when no structured code was found — never as a tiebreak.
    RECORD(pressure("max slots reached"));
    RECORD(pressure("llama-server: out of memory"));
    RECORD(pressure("INSUFFICIENT VRAM"));           // matched case-insensitively
    RECORD(pressure(R"({"error":"no available ports"})")); // JSON, but no code
    RECORD(!pressure("model architecture is not supported"));
    RECORD(!pressure(""));

    // ── 4. an unparseable body is not a guess ────────────────────────────────
    // Truncated JSON falls through to the substring match rather than throwing,
    // and answers on the prose alone.
    RECORD(pressure(R"({"error":"out of memory while loa)"));
    // A truncated body whose only capacity evidence is the CODE stays false —
    // the six legacy phrases are llama.cpp's prose, and "capacity_pressure" is
    // deliberately not among them. So a half-written structured body is not
    // silently rescued by the matcher underneath it, which is the right answer:
    // a response that arrived incomplete has not told us what went wrong.
    RECORD(!pressure(R"({"error":{"code":"capacity_pre)"));
    RECORD(!pressure("<html><body>502 Bad Gateway</body></html>"));

    // ── 5. the same refusal, through EngineClient's parser ───────────────────
    // Two parsers exist for one code and they cover different shapes: this one
    // reads only the NESTED form, so the node's load-handler shape above leaves
    // its code empty — and is rescued by the status-derived fallback, since the
    // node answers 503 for capacity. That recovery is load-bearing and entirely
    // implicit, so it is pinned here: if either the node's status or this
    // fallback changes, this is the assertion that objects.
    //
    // Stated plainly, because the section reads like it guards something live:
    // `EngineError::is_capacity_pressure()` has NO production consumer today.
    // The scheduler uses its own matcher; these assertions describe the path
    // EngineClient is being built toward. They are worth having early — the
    // shape divergence above is exactly the kind that is free to fix now and
    // expensive to discover the first time something depends on it — but they
    // are not evidence that this parser is in the eviction path. It is not.
    const auto through_client = [](int status, const std::string& body) {
        return mm::EngineError::parse("HTTP " + std::to_string(status) + ": " + body);
    };
    RECORD(through_client(503, R"({"error":"failed to load model","code":"capacity_pressure"})")
               .is_capacity_pressure());
    RECORD(through_client(503, "").is_capacity_pressure());
    RECORD(through_client(500, R"({"error":{"code":"capacity_pressure"}})")
               .is_capacity_pressure());
    // 500 with no code is internal, not capacity: retrying it evicts a live
    // agent to re-run a request that will fail again the same way.
    RECORD(!through_client(500, R"({"error":"segfault in expert routing"})")
                .is_capacity_pressure());
    RECORD(!through_client(404, R"({"error":"no such model"})").is_capacity_pressure());

#undef RECORD
    return ok;
}

bool test_engine_telemetry_republication() {
    // Control's half of the chain, against a node that answers. What is under
    // test is the ROUTING: finding which node holds an engine, passing the
    // upstream status through rather than flattening it, and distinguishing
    // "no such engine" from "the node is unreachable".
    auto dir = temp_test_dir("engine-telemetry");
    std::filesystem::create_directories(dir / "models");

    bool ok = true;
    auto record = [&](bool condition, const char* expression, int line) {
        if (!check(condition, expression, line)) ok = false;
    };
#define RECORD(expr) record((expr), #expr, __LINE__)

    {
        // A stand-in node exposing the two proxied GETs.
        httplib::Server node;
        std::atomic<int> heat_calls{0};
        std::string last_heat_query;
        std::mutex query_mutex;

        node.Get("/api/node/engines/:slot/heat",
                 [&](const httplib::Request& req, httplib::Response& res) {
                     const auto slot = req.path_params.at("slot");
                     // The 501 lives INSIDE the parameterised handler, exactly as
                     // it does on the real node: the descriptor is consulted per
                     // slot, not routed around. A second, more specific httplib
                     // route would never be reached — `:slot` is registered first
                     // and matches everything.
                     if (slot == "no-telemetry") {
                         res.status = 501;
                         res.set_content(
                             nlohmann::json{{"error", "this engine publishes no heat map"}}
                                 .dump(),
                             "application/json");
                         return;
                     }
                     ++heat_calls;
                     {
                         std::lock_guard<std::mutex> lk(query_mutex);
                         last_heat_query = req.get_param_value("resolution");
                     }
                     res.set_content(nlohmann::json{{"slot", slot},
                                                    {"resolution", "bucketed"},
                                                    {"n_layers", 4},
                                                    {"n_experts", 16}}
                                         .dump(),
                                     "application/json");
                 });
        node.Get("/api/node/engines/:slot/sequences",
                 [&](const httplib::Request&, httplib::Response& res) {
                     res.set_content(
                         nlohmann::json{{"sequences", nlohmann::json::array()}}.dump(),
                         "application/json");
                 });

        // Health and status too: `connected` and the slot list come from the
        // POLL, not from a setter. Faking them would test a registry state the
        // real system never reaches this way.
        node.Get("/api/node/health", [&](const httplib::Request&, httplib::Response& res) {
            res.set_content(nlohmann::json{{"cpu_percent", 10.0},
                                           {"ram_percent", 20.0},
                                           {"ram_total_mb", 65536},
                                           {"ram_used_mb", 8192},
                                           {"gpu_vram_total_mb", 24576},
                                           {"gpu_vram_used_mb", 1024}}
                                .dump(),
                            "application/json");
        });
        node.Get("/api/node/status", [&](const httplib::Request&, httplib::Response& res) {
            nlohmann::json soma_slot = {{"id", "slot-soma"},
                                        {"backend", "soma"},
                                        {"state", "ready"},
                                        {"model_path", "container/Qwen3-30B-A3B"},
                                        {"vram_usage_mb", 512},
                                        {"agent_ids", nlohmann::json::array()}};
            nlohmann::json llama_slot = {{"id", "no-telemetry"},
                                         {"backend", "llama-cpp"},
                                         {"state", "ready"},
                                         {"model_path", "m.gguf"},
                                         {"agent_ids", nlohmann::json::array()}};
            res.set_content(nlohmann::json{{"slots", {soma_slot, llama_slot}},
                                           {"max_slots", 4},
                                           {"slot_available", 2},
                                           {"disk_free_mb", 100000}}
                                .dump(),
                            "application/json");
        });

        const uint16_t node_port = find_free_test_port();
        RECORD(node_port != 0);
        std::thread node_thread([&] { node.listen("127.0.0.1", node_port); });
        for (int i = 0; i < 50 && !node.is_running(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        mm::AgentManager agents(dir.string());
        mm::AgentQueue queue;
        mm::NodeRegistry registry(dir.string());
        mm::AgentScheduler scheduler(registry, (dir / "models").string());
        mm::ControlApiServer api(agents, queue, registry, scheduler, dir.string(),
                                 (dir / "models").string(), "");

        const auto node_url = "http://127.0.0.1:" + std::to_string(node_port);
        RECORD(wait_for_test_server(node_url));
        const auto node_id = registry.add_node(node_url, "", "engine-host", false);
        RECORD(!node_id.empty());
        // The slot list is what makes an engine FINDABLE: control discovers which
        // node holds it rather than making the caller supply one, because the
        // answer changes on every eviction. It arrives through the health poll,
        // which is the path the real system uses.
        registry.start_health_poll(1);
        RECORD(wait_for_registered_node(registry, node_id));
        RECORD(wait_for_node_snapshot(registry, node_id, 0, [](const mm::NodeInfo& n) {
            return n.slots.size() == 2;
        }));

        const uint16_t port = find_free_test_port();
        RECORD(port != 0);
        const std::string base = "http://127.0.0.1:" + std::to_string(port);
        std::thread api_thread([&] { api.listen(port); });
        mm::HttpClient client(base);
        for (int i = 0; i < 50; ++i) {
            if (client.get("/v1/engines").status != 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        auto with_retry = [](auto&& request) {
            mm::HttpResponse resp;
            for (int attempt = 0; attempt < kTransportRetries; ++attempt) {
                resp = request();
                if (resp.status != 0) break;
                transport_backoff(attempt);
            }
            return resp;
        };

        // ── the cluster view ─────────────────────────────────────────────────
        auto engines = with_retry([&] { return client.get("/v1/engines"); });
        RECORD(engines.status == 200);
        {
            const auto j = nlohmann::json::parse(engines.body, nullptr, false);
            RECORD(!j.is_discarded());
            RECORD(j.value("data", nlohmann::json::array()).size() == 2);
            // The node is reported WITH the engine, so a client never has to
            // resolve it separately.
            RECORD(engines.body.find("\"node_id\"") != std::string::npos);
            RECORD(engines.body.find("\"backend\":\"soma\"") != std::string::npos);
            RECORD(j["tier_summary"].value("vram_mb", 0) == 512);
        }

        // ── the proxied GETs ─────────────────────────────────────────────────
        auto heat = with_retry([&] { return client.get("/v1/engines/slot-soma/heat"); });
        RECORD(heat.status == 200);
        RECORD(heat.body.find("\"n_experts\":16") != std::string::npos);
        RECORD(heat_calls.load() >= 1);
        {
            std::lock_guard<std::mutex> lk(query_mutex);
            // Bucketed by default all the way down: the opt-in is explicit at
            // every hop rather than defaulted back to full by a middle layer.
            RECORD(last_heat_query.empty());
        }

        auto full = with_retry([&] {
            return client.get("/v1/engines/slot-soma/heat?resolution=full");
        });
        RECORD(full.status == 200);
        {
            std::lock_guard<std::mutex> lk(query_mutex);
            RECORD(last_heat_query == "full");
        }

        auto slots = with_retry([&] { return client.get("/v1/engines/slot-soma/slots"); });
        RECORD(slots.status == 200);
        RECORD(slots.body.find("sequences") != std::string::npos);

        // ── the refusals ─────────────────────────────────────────────────────
        auto unknown = with_retry([&] { return client.get("/v1/engines/nope/heat"); });
        RECORD(unknown.status == 404);
        RECORD(unknown.body.find("no such engine") != std::string::npos);

        // 501 survives the hop rather than becoming a generic control error.
        auto unsupported = with_retry([&] {
            return client.get("/v1/engines/no-telemetry/heat");
        });
        RECORD(unsupported.status == 501);
        RECORD(unsupported.body.find("publishes no heat map") != std::string::npos);

        // A node that has gone away is 502, not 500 — an operator chasing this
        // should be pointed at the node rather than at control.
        node.stop();
        if (node_thread.joinable()) node_thread.join();
        auto gone = with_retry([&] { return client.get("/v1/engines/slot-soma/heat"); });
        RECORD(gone.status == 502);
        RECORD(gone.body.find("node unreachable") != std::string::npos);
        RECORD(gone.body.find(node_id) != std::string::npos);

        registry.stop_health_poll();
        api.stop();
        if (api_thread.joinable()) api_thread.join();
    }

    RECORD(remove_tree(dir));
#undef RECORD
    return ok;
}

bool test_route_scopes_and_token_store() {
    // ── the table is exhaustive, and the check that says so actually works ───
    //
    // A coverage check that cannot fail is decoration. This asserts BOTH
    // directions: the real table covers a realistic route set, and a route
    // absent from it is reported rather than defaulted.
    std::vector<std::string> missing;
    CHECK(mm::require_complete_coverage(
        {"GET /v1/agents", "POST /v1/agents/:id/chat", "POST /v1/models/admit",
         "PUT /v1/agents/:id/backend", "DELETE /v1/models/:id", "GET /v1/tokens"},
        missing));
    CHECK(missing.empty());

    CHECK(!mm::require_complete_coverage({"GET /v1/agents", "POST /v1/not/in/the/table"},
                                         missing));
    CHECK(missing.size() == 1);
    CHECK(missing[0] == "POST /v1/not/in/the/table");

    // Every entry has a method and a pattern — an empty one would match nothing
    // and silently fall through to the restrictive default.
    for (const auto& rs : mm::route_scope_table()) {
        CHECK(rs.method != nullptr && *rs.method != '\0');
        CHECK(rs.pattern != nullptr && *rs.pattern != '\0');
    }

    // ── the implication order ────────────────────────────────────────────────
    const auto read = static_cast<mm::ScopeSet>(mm::Scope::Read);
    const auto chat = static_cast<mm::ScopeSet>(mm::Scope::Chat);
    const auto oper = static_cast<mm::ScopeSet>(mm::Scope::Operator);

    CHECK(mm::scope_permits(read, mm::Scope::Read));
    CHECK(!mm::scope_permits(read, mm::Scope::Chat));
    CHECK(!mm::scope_permits(read, mm::Scope::Operator));
    // chat implies read: sending a message you cannot then fetch is not a
    // coherent permission.
    CHECK(mm::scope_permits(chat, mm::Scope::Read));
    CHECK(mm::scope_permits(chat, mm::Scope::Chat));
    CHECK(!mm::scope_permits(chat, mm::Scope::Operator));
    // operator implies everything.
    CHECK(mm::scope_permits(oper, mm::Scope::Read));
    CHECK(mm::scope_permits(oper, mm::Scope::Chat));
    CHECK(mm::scope_permits(oper, mm::Scope::Operator));
    CHECK(!mm::scope_permits(mm::kScopeNone, mm::Scope::Read));

    // ── parsing rejects rather than drops ────────────────────────────────────
    mm::ScopeSet parsed = mm::kScopeNone;
    CHECK(mm::parse_scopes("read,operator", parsed));
    CHECK(parsed == (read | oper));
    CHECK(mm::parse_scopes(" READ , chat ", parsed));   // case and space tolerant
    CHECK(parsed == (read | chat));
    // "opereator" must be REPORTED. Silently ignoring it would mint a token that
    // looks right in the request and cannot do what it was minted for.
    CHECK(!mm::parse_scopes("read,opereator", parsed));
    CHECK(parsed == read);
    CHECK(mm::format_scopes(read | oper) == "read,operator");

    // ── the token store ──────────────────────────────────────────────────────
    auto dir = temp_test_dir("scopes");
    mm::ControlModelRegistry reg;
    std::string err;
    CHECK(reg.open(dir.string(), err));
    CHECK(!reg.has_api_tokens());

    // A token with no scopes can do nothing; minting one is refused rather than
    // producing a credential that fails every request mysteriously.
    CHECK(reg.create_api_token("empty", mm::kScopeNone, err).empty());

    const auto reader = reg.create_api_token("dashboard", read, err);
    CHECK(!reader.empty());
    CHECK(reg.has_api_tokens());
    const auto admin = reg.create_api_token("ops", oper, err);
    CHECK(!admin.empty());
    CHECK(reader != admin);
    CHECK(reader.size() >= 32);   // CSPRNG material, not a counter

    // The token is stored HASHED. A leaked backup must not hand over working
    // credentials, so the secret appears in no row.
    mm::ApiToken row;
    CHECK(reg.find_api_token(mm::pairing::sha256_hex(reader), row));
    CHECK(row.label == "dashboard");
    CHECK(row.scopes == read);
    CHECK(row.token_sha256 != reader);
    CHECK(!reg.find_api_token(reader, row));            // the raw token is not a key
    CHECK(!reg.find_api_token(mm::pairing::sha256_hex("guessed"), row));

    CHECK(reg.list_api_tokens().size() == 2);
    for (const auto& t : reg.list_api_tokens()) CHECK(t.token_sha256.size() == 64);

    // Revoked, not deleted: the row is the audit trail, and a deleted token
    // cannot answer "what was this credential allowed to do".
    const auto reader_id = reg.list_api_tokens().front().id;
    CHECK(reg.revoke_api_token(reader_id, err));
    CHECK(!reg.find_api_token(mm::pairing::sha256_hex(reader), row));
    CHECK(reg.list_api_tokens().size() == 2);
    CHECK(!reg.revoke_api_token(reader_id, err));       // already revoked
    CHECK(reg.has_api_tokens());                        // the operator token remains

    reg.close();
    CHECK(remove_tree(dir));
    return true;
}

bool test_capacity_fit_across_three_axes() {
    mm::CapacityPolicy policy;   // defaults are the constants this replaced

    mm::ResourceFootprint model;
    model.vram_mb = 8000;

    // A node with plenty of everything: native fit.
    mm::HostCapacity roomy;
    roomy.vram_total_mb = 24576; roomy.vram_free_mb = 20000;
    roomy.ram_total_mb = 65536;  roomy.ram_free_mb = 40000;
    roomy.disk_free_mb = 500000;
    CHECK(mm::evaluate_fit(model, roomy, policy) == mm::FitQuality::Native);

    // Not enough VRAM but a big GPU and spare RAM: offload, and it must rank
    // BELOW a native fit however much headroom it has left.
    mm::HostCapacity offloadable = roomy;
    offloadable.vram_free_mb = 4000;
    CHECK(mm::evaluate_fit(model, offloadable, policy) == mm::FitQuality::Offload);
    CHECK(mm::capacity_score(model, roomy, policy) >
          mm::capacity_score(model, offloadable, policy));

    // A small GPU cannot offload against: hybrid loads need a real one.
    mm::HostCapacity tiny_gpu = roomy;
    tiny_gpu.vram_total_mb = 4096; tiny_gpu.vram_free_mb = 4000;
    CHECK(mm::evaluate_fit(model, tiny_gpu, policy) == mm::FitQuality::None);

    // DISK. Collected by the health poll since it was written and consulted by
    // nothing until now: a node with no room cannot write a KV checkpoint or
    // spill, so it cannot host anything however much VRAM it has.
    mm::HostCapacity no_disk = roomy;
    no_disk.disk_free_mb = 512;
    std::string why;
    CHECK(mm::evaluate_fit(model, no_disk, policy, &why) == mm::FitQuality::None);
    CHECK(why.find("disk") != std::string::npos);

    // Zero means NOT REPORTED, not full — the field defaults to zero and an
    // older node never sends it. Enforcing against that would exclude every node
    // predating the field.
    mm::HostCapacity silent = roomy;
    silent.disk_free_mb = 0;
    CHECK(mm::evaluate_fit(model, silent, policy) == mm::FitQuality::Native);

    // A streaming footprint — RAM + disk, no VRAM — fits when the disk is there.
    mm::ResourceFootprint streamed;
    streamed.ram_mb = 4000;
    streamed.disk_mb = 400000;
    CHECK(mm::evaluate_fit(streamed, roomy, policy) == mm::FitQuality::Native);

    // And is refused when it is not, with no offload path: there is nothing to
    // trade disk against, which is exactly why one scalar could not carry it.
    // 495000 leaves 5000 MB, under the 8192 MB headroom.
    streamed.disk_mb = 495000;
    std::string disk_why;
    CHECK(mm::evaluate_fit(streamed, roomy, policy, &disk_why) == mm::FitQuality::None);
    CHECK(disk_why.find("disk") != std::string::npos);
    // The same node still takes the VRAM-shaped model, so the refusal is about
    // the axis that ran out rather than about the node.
    CHECK(mm::evaluate_fit(model, roomy, policy) == mm::FitQuality::Native);

    // A no-VRAM footprint fits natively; the old scalar had no way to say this.
    mm::ResourceFootprint cpu_only;
    cpu_only.ram_mb = 8000;
    CHECK(mm::evaluate_fit(cpu_only, roomy, policy) == mm::FitQuality::Native);
    CHECK(!cpu_only.empty());
    CHECK(cpu_only.dominant_mb() == 8000);
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
        {"engine_supervisor_not_found_statuses", test_engine_supervisor_not_found_statuses},
        {"slot_lease_blocks_unload_and_suspend_while_busy",
         test_slot_lease_blocks_unload_and_suspend_while_busy},
        {"node_action_progress_json_round_trip",
         test_node_action_progress_json_round_trip},
        {"engine_config_validation_and_round_trip",
         test_engine_config_validation_and_round_trip},
        {"engine_config_rejects_per_machine_keys",
         test_engine_config_rejects_per_machine_keys},
        {"engine_artifact_fingerprint_is_exact",
         test_engine_artifact_fingerprint_is_exact},
        {"engine_config_store_persists_and_bumps_version",
         test_engine_config_store_persists_and_bumps_version},
        {"placement_refused_until_engine_config_exists",
         test_placement_refused_until_engine_config_exists},
        {"conformance_gates_placement_candidates",
         test_conformance_gates_placement_candidates},
        {"admission_variant_is_the_collision_key",
         test_admission_variant_is_the_collision_key},
        {"concurrent_admission_of_one_model_joins_not_duplicates",
         test_concurrent_admission_of_one_model_joins_not_duplicates},
        {"desired_artifact_names_what_a_node_lacks",
         test_desired_artifact_names_what_a_node_lacks},
        {"provisioning_progress_sink_may_read_status",
         test_provisioning_progress_sink_may_read_status},
        {"provisioner_exception_fails_the_engine_not_the_node",
         test_provisioner_exception_fails_the_engine_not_the_node},
        {"soma_resolves_beside_the_node_binary",
         test_soma_resolves_beside_the_node_binary},
        {"node_modal_ladder", test_node_modal_ladder},
        {"engine_digest_and_package_grants_are_one_shot",
         test_engine_digest_and_package_grants_are_one_shot},
        {"scheduler_skips_failed_node_current_attempt",
         test_scheduler_skips_failed_node_current_attempt},
        {"scheduler_transfers_existing_relative_models_with_unique_cache_ids",
         test_scheduler_transfers_existing_relative_models_with_unique_cache_ids},
        {"scheduler_eviction_skips_unsuspendable_shared_slot",
         test_scheduler_eviction_skips_unsuspendable_shared_slot},
        {"soma_footprint_is_ram_shaped_not_vram_shaped",
         test_soma_footprint_is_ram_shaped_not_vram_shaped},
        {"placement_failure_codes_separate_eligibility_from_capacity",
         test_placement_failure_codes_separate_eligibility_from_capacity},
        {"scheduler_audits_placement_and_release",
         test_scheduler_audits_placement_and_release},
        {"placement_history_records_and_closes_rows",
         test_placement_history_records_and_closes_rows},
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
        {"multi_shard_directory_sizes_correctly", test_multi_shard_directory_sizes_correctly},
        {"capacity_fit_across_three_axes", test_capacity_fit_across_three_axes},
        {"model_registry_makes_soma_routable", test_model_registry_makes_soma_routable},
        {"admission_pipeline_runs_and_reports", test_admission_pipeline_runs_and_reports},
        {"admission_fetch_stage", test_admission_fetch_stage},
        {"requantization_is_a_new_admission", test_requantization_is_a_new_admission},
        {"route_scopes_and_token_store", test_route_scopes_and_token_store},
        {"scope_negatives_over_http", test_scope_negatives_over_http},
        {"images_refused_by_engine_capability", test_images_refused_by_engine_capability},
        {"capacity_pressure_is_structured", test_capacity_pressure_is_structured},
        {"engine_telemetry_republication", test_engine_telemetry_republication},
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
