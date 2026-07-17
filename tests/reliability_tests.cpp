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
#include "control/engine_group_planner.hpp"
#include "control/control_api_server.hpp"
#include "control/node_registry.hpp"
#include "control/performance_tracker.hpp"
#include "control/tts_service_client.hpp"
#include "common/gguf_metadata.hpp"
#include "common/inference_sizing.hpp"
#include "control/agent_config_validator.hpp"
#include "node/runtime_process.hpp"
#include "node/slot_manager.hpp"
#include "node/vllm_provisioner.hpp"
#include "node/vllm_runtime.hpp"
#include "node/llama_runtime.hpp"
#include "node/llama_cpp_provisioner.hpp"
#include "node/ray_orchestration.hpp"
#include "node/hf_cache.hpp"

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

bool test_vllm_slot_info_and_suspend_without_kv_cache() {
    auto dir = temp_test_dir("vllm-slot");
    mm::SlotManager slots(46120, 46121, 1, "missing-vllm");
    const auto slot_id = slots.add_ready_test_slot("Qwen/Qwen3-8B",
                                                   "agent-v");

    auto info = slots.find_slot(slot_id);
    CHECK(info.has_value());

    nlohmann::json serialized = *info;
    auto parsed = serialized.get<mm::SlotInfo>();
    CHECK(parsed.model_path == "Qwen/Qwen3-8B");

    auto suspend = slots.suspend_slot(slot_id);
    CHECK(suspend.status == mm::SlotOperationStatus::Ok);
    CHECK(suspend.kv_cache_path.empty());

    auto suspended = slots.find_slot(slot_id);
    CHECK(suspended.has_value());
    CHECK(suspended->state == mm::SlotState::Suspended);
    CHECK(suspended->kv_cache_path.empty());

    CHECK(remove_tree(dir));
    return true;
}

bool test_vllm_gpu_budget_accounting() {
    auto dir = temp_test_dir("vllm-budget");
    mm::SlotManager slots(46130, 46133, 4, "missing-vllm", 0.80);
    CHECK(slots.vllm_gpu_budget() == 0.80);
    CHECK(slots.vllm_gpu_fraction_used() == 0.0);

    // An active vLLM slot holding 0.78 leaves less than the minimum useful
    // slice (0.05) of the 0.80 budget — the next load must be rejected
    // before any process spawn is attempted.
    const auto holder_id = slots.add_ready_test_slot("Qwen/Qwen3-8B",
                                                     "agent-holder",
                                                     0.78);
    CHECK(slots.vllm_gpu_fraction_used() == 0.78);

    auto rejected = slots.load_model("Qwen/Qwen3-4B",
                                     mm::VllmSettings{},
                                     "agent-rejected");
    CHECK(rejected.empty());
    CHECK(slots.last_error().find("insufficient GPU memory budget")
          != std::string::npos);
    CHECK(slots.vllm_gpu_fraction_used() == 0.78);

    // The fraction is informational on the SlotInfo and survives JSON.
    auto info = slots.find_slot(holder_id);
    CHECK(info.has_value());
    nlohmann::json serialized = *info;
    auto parsed = serialized.get<mm::SlotInfo>();
    CHECK(parsed.gpu_mem_fraction == 0.78);

    // Suspending the holder releases its claim on the budget.
    auto suspend = slots.suspend_slot(holder_id);
    CHECK(suspend.status == mm::SlotOperationStatus::Ok);
    CHECK(slots.vllm_gpu_fraction_used() == 0.0);

    // A failed load (missing vLLM executable) must release its pending
    // reservation rather than leak budget.
    auto failed = slots.load_model("Qwen/Qwen3-4B",
                                   mm::VllmSettings{},
                                   "agent-failed");
    CHECK(failed.empty());
    CHECK(slots.vllm_gpu_fraction_used() == 0.0);

    CHECK(remove_tree(dir));
    return true;
}

bool test_vllm_shared_slot_attach_and_detach() {
    auto dir = temp_test_dir("vllm-shared");
    mm::SlotManager slots(46140, 46143, 4, "missing-vllm", 0.90);

    // Agent A's engine is running (test slots carry default launch settings).
    const auto engine_id = slots.add_ready_test_slot("Qwen/Qwen3-8B",
                                                     "agent-a",
                                                     0.45);

    // Agent B requests the same model with compatible launch settings —
    // attaches to the running engine instead of spawning a second process
    // (a real spawn of "missing-vllm" would fail).
    auto shared_id = slots.load_model("Qwen/Qwen3-8B",
                                      mm::VllmSettings{},
                                      "agent-b");
    CHECK(shared_id == engine_id);
    CHECK(slots.find_slot_by_agent("agent-a").has_value());
    CHECK(slots.find_slot_by_agent("agent-b").has_value());
    CHECK(slots.vllm_gpu_fraction_used() == 0.45);  // no new engine

    auto info = slots.find_slot(engine_id);
    CHECK(info.has_value());
    CHECK(info->agent_ids.size() == 2);
    CHECK(info->assigned_agent == "agent-a");

    nlohmann::json serialized = *info;
    auto parsed = serialized.get<mm::SlotInfo>();
    CHECK(parsed.agent_ids.size() == 2);

    // Mismatched launch settings must NOT attach — the fresh spawn then
    // fails on the missing executable, proving no sharing happened.
    mm::VllmSettings other;
    other.max_model_len = 8192;
    auto fresh = slots.load_model("Qwen/Qwen3-8B", other, "agent-c");
    CHECK(fresh.empty());

    // Detaching one agent keeps the engine up for the other.
    auto d1 = slots.detach_agent(engine_id, "agent-b");
    CHECK(d1.ok());
    CHECK(d1.remaining_agents == 1);
    CHECK(!d1.unloaded);
    CHECK(!slots.find_slot_by_agent("agent-b").has_value());

    // Detaching the last agent unloads the engine.
    auto d2 = slots.detach_agent(engine_id, "agent-a");
    CHECK(d2.ok());
    CHECK(d2.remaining_agents == 0);
    CHECK(d2.unloaded);
    CHECK(!slots.find_slot(engine_id).has_value());

    CHECK(remove_tree(dir));
    return true;
}

bool test_vllm_restore_attaches_and_cleans_suspended_record() {
    auto dir = temp_test_dir("vllm-restore-attach");
    mm::SlotManager slots(46150, 46153, 4, "missing-vllm", 0.90);

    // Agent B's old engine was suspended (vLLM: no KV cache survives).
    const auto old_id = slots.add_ready_test_slot("Qwen/Qwen3-8B",
                                                  "agent-b",
                                                  0.45);
    CHECK(slots.suspend_slot(old_id).ok());

    // Meanwhile agent A's compatible engine is running.
    const auto engine_id = slots.add_ready_test_slot("Qwen/Qwen3-8B",
                                                     "agent-a",
                                                     0.45);

    // Restoring agent B attaches to A's engine and drops B's stale
    // suspended record.
    auto restored = slots.restore_slot("Qwen/Qwen3-8B",
                                       mm::VllmSettings{},
                                       "agent-b");
    CHECK(restored == engine_id);
    CHECK(!slots.find_slot(old_id).has_value());

    auto info = slots.find_slot(engine_id);
    CHECK(info.has_value());
    CHECK(info->agent_ids.size() == 2);

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

bool test_vllm_sleep_fallback_and_dead_wake_cleanup() {
    auto dir = temp_test_dir("vllm-sleep");
    mm::SlotManager slots(46160, 46163, 4, "missing-vllm", 0.90);

    // Suspending an unreachable engine falls back to stopping it: state is
    // Suspended but not sleeping.
    const auto a = slots.add_ready_test_slot("Qwen/Qwen3-8B", "agent-a", 0.45);
    auto susp = slots.suspend_slot(a);
    CHECK(susp.ok());
    auto info = slots.find_slot(a);
    CHECK(info.has_value());
    CHECK(info->state == mm::SlotState::Suspended);
    CHECK(!info->sleeping);

    // A sleeping engine whose process died: the wake fails, the dead record
    // is discarded, and the load falls through to a fresh start (which then
    // fails on the missing executable).
    const auto b = slots.add_ready_test_slot("Qwen/Qwen3-8B", "agent-b", 0.45);
    CHECK(slots.mark_test_slot_sleeping(b));
    auto sleeping_info = slots.find_slot(b);
    CHECK(sleeping_info.has_value());
    CHECK(sleeping_info->sleeping);
    nlohmann::json serialized = *sleeping_info;
    CHECK(serialized.value("sleeping", false));
    CHECK(serialized.get<mm::SlotInfo>().sleeping);

    auto loaded = slots.load_model("Qwen/Qwen3-8B", mm::VllmSettings{},
                                   "agent-c");
    CHECK(loaded.empty());                   // fresh spawn fails (missing vllm)
    CHECK(!slots.find_slot(b).has_value());  // dead sleeping record discarded
    CHECK(slots.vllm_gpu_fraction_used() == 0.0);

    // Sleep mode flag lands in the engine args; disabling removes it.
    mm::VllmSettings vs;
    auto args = mm::build_vllm_server_args("Qwen/Qwen3-8B", vs, 8000);
    CHECK(has_arg(args, "--enable-sleep-mode"));
    vs.enable_sleep_mode = false;
    args = mm::build_vllm_server_args("Qwen/Qwen3-8B", vs, 8000);
    CHECK(!has_arg(args, "--enable-sleep-mode"));

    CHECK(remove_tree(dir));
    return true;
}

bool test_vllm_metrics_parse_and_slot_info_round_trip() {
    const std::string text =
        "# HELP vllm:num_requests_running Number of requests on GPU.\n"
        "# TYPE vllm:num_requests_running gauge\n"
        "vllm:num_requests_running{model_name=\"Qwen/Qwen3-8B\"} 2.0\n"
        "vllm:num_requests_waiting{model_name=\"Qwen/Qwen3-8B\"} 5.0\n"
        "vllm:gpu_cache_usage_perc{model_name=\"Qwen/Qwen3-8B\"} 0.42\n"
        "some_other_metric 7\n";
    auto m = mm::parse_vllm_metrics_text(text);
    CHECK(m.valid);
    CHECK(m.num_requests_running == 2);
    CHECK(m.num_requests_waiting == 5);
    CHECK(m.kv_cache_usage > 0.41 && m.kv_cache_usage < 0.43);

    // vLLM v1 renamed the cache metric.
    auto m_v1 = mm::parse_vllm_metrics_text(
        "vllm:kv_cache_usage_perc{model_name=\"x\"} 0.9\n");
    CHECK(m_v1.valid);
    CHECK(m_v1.kv_cache_usage > 0.89);

    auto m_empty = mm::parse_vllm_metrics_text("# comments only\n");
    CHECK(!m_empty.valid);

    // Engine load survives the SlotInfo JSON contract node → control.
    mm::SlotInfo si;
    si.engine_metrics_valid = true;
    si.num_requests_running = 3;
    si.num_requests_waiting = 1;
    si.kv_cache_usage = 0.5;
    nlohmann::json j = si;
    auto parsed = j.get<mm::SlotInfo>();
    CHECK(parsed.engine_metrics_valid);
    CHECK(parsed.num_requests_running == 3);
    CHECK(parsed.num_requests_waiting == 1);
    CHECK(parsed.kv_cache_usage > 0.49 && parsed.kv_cache_usage < 0.51);

    return true;
}

namespace {
mm::NodeInfo make_capable_node(const std::string& id,
                               const std::string& arch,
                               const std::string& vllm_version,
                               int gpu_count,
                               std::vector<std::string> backends,
                               bool supports_ray,
                               double interconnect_gbps = 0.0) {
    mm::NodeInfo n;
    n.id = id;
    n.url = "http://" + id + ":7070";
    n.connected = true;
    n.capabilities.arch = arch;
    n.capabilities.vllm_version = vllm_version;
    n.capabilities.gpu_count = gpu_count;
    n.capabilities.comm_backends = std::move(backends);
    n.capabilities.supports_ray = supports_ray;
    n.capabilities.interconnect_gbps = interconnect_gbps;
    return n;
}
} // namespace

bool test_engine_group_planner() {
    using mm::EngineGroupRequest;

    // Two DGX Sparks (aarch64, 1 GPU each, NCCL over 200GbE) form the natural
    // NCCL pipeline-parallel group for a 2-GPU model.
    auto spark1 = make_capable_node("spark-1", "aarch64", "0.6.3", 1,
                                    {"nccl", "gloo"}, true, 200.0);
    auto spark2 = make_capable_node("spark-2", "aarch64", "0.6.3", 1,
                                    {"nccl", "gloo"}, true, 200.0);
    auto nas    = make_capable_node("nas", "x86_64", "0.6.3", 1,
                                    {"nccl", "gloo"}, true, 10.0);

    {
        EngineGroupRequest req{"big-model", 2};
        auto best = mm::best_engine_group(req, {spark1, spark2, nas});
        CHECK(best.has_value());
        CHECK(best->comm_backend == "nccl");
        CHECK(best->pipeline_parallel_size == 2);
        CHECK(best->tensor_parallel_size == 1);
        CHECK(best->world_size == 2);
        // Cross-arch pairing (spark + nas) is never formed: different
        // fingerprints can't share an engine, so the two Sparks are chosen.
        CHECK(best->nodes.size() == 2);
        CHECK((best->nodes[0] == "spark-1" || best->nodes[0] == "spark-2"));
        CHECK((best->nodes[1] == "spark-1" || best->nodes[1] == "spark-2"));
    }

    // Spark 2 unavailable: a single remaining Spark can't reach world_size 2.
    {
        EngineGroupRequest req{"big-model", 2};
        auto best = mm::best_engine_group(req, {spark1, nas});
        CHECK(!best.has_value());
    }

    // A model that fits on one node yields a single-node (pp=1) plan, so the
    // caller routes it normally rather than forming a group.
    {
        EngineGroupRequest req{"small-model", 1};
        auto best = mm::best_engine_group(req, {spark1, spark2});
        CHECK(best.has_value());
        CHECK(best->pipeline_parallel_size == 1);
        CHECK(!best->spans_nodes());
    }

    // Gloo last resort: a pair where one member lacks NCCL forms a Gloo group
    // only because no all-NCCL pair exists for that fingerprint.
    {
        auto win1 = make_capable_node("win-1", "x86_64", "0.6.3", 1,
                                      {"gloo"}, true, 1.0);
        auto win2 = make_capable_node("win-2", "x86_64", "0.6.3", 1,
                                      {"nccl", "gloo"}, true, 1.0);
        EngineGroupRequest req{"big-model", 2};
        auto best = mm::best_engine_group(req, {win1, win2});
        CHECK(best.has_value());
        CHECK(best->comm_backend == "gloo");
    }

    // NCCL pair outranks a Gloo pair for the same need.
    {
        auto a = make_capable_node("a", "x86_64", "1.0", 1, {"nccl", "gloo"}, true, 100.0);
        auto b = make_capable_node("b", "x86_64", "1.0", 1, {"nccl", "gloo"}, true, 100.0);
        auto c = make_capable_node("c", "aarch64", "1.0", 1, {"gloo"}, true, 1.0);
        auto d = make_capable_node("d", "aarch64", "1.0", 1, {"gloo"}, true, 1.0);
        EngineGroupRequest req{"big-model", 2};
        auto ranked = mm::plan_engine_groups(req, {a, b, c, d});
        CHECK(ranked.size() >= 2);
        CHECK(ranked.front().valid);
        CHECK(ranked.front().comm_backend == "nccl");  // NCCL pool wins
    }

    // A multi-node need where a member lacks Ray is rejected.
    {
        auto a = make_capable_node("a", "x86_64", "1.0", 1, {"nccl", "gloo"}, true);
        auto b = make_capable_node("b", "x86_64", "1.0", 1, {"nccl", "gloo"}, false);
        EngineGroupRequest req{"big-model", 2};
        auto best = mm::best_engine_group(req, {a, b});
        CHECK(!best.has_value());
    }

    // Nodes with no advertised capabilities (older build) are ignored.
    {
        mm::NodeInfo bare;
        bare.id = "legacy";
        bare.connected = true;
        EngineGroupRequest req{"big-model", 1};
        auto best = mm::best_engine_group(req, {bare});
        CHECK(!best.has_value());
    }

    // Capability block survives the node-status JSON round trip.
    {
        nlohmann::json j = spark1.capabilities;
        auto parsed = j.get<mm::NodeCapabilities>();
        CHECK(parsed.arch == "aarch64");
        CHECK(parsed.vllm_version == "0.6.3");
        CHECK(parsed.gpu_count == 1);
        CHECK(parsed.has_comm_backend("nccl"));
        CHECK(parsed.supports_ray);
        CHECK(parsed.interconnect_gbps > 199.0);
    }

    return true;
}

bool test_engine_group_launch_helpers() {
    // Head address derivation: host comes from the node's API URL, port from
    // the head's advertised GCS port.
    CHECK(mm::derive_ray_head_address("http://192.168.68.82:7070", 6379)
          == "192.168.68.82:6379");
    CHECK(mm::derive_ray_head_address("https://spark1.local:7070/", 6380)
          == "spark1.local:6380");
    CHECK(mm::derive_ray_head_address("http://10.0.0.5", 6379)
          == "10.0.0.5:6379");
    CHECK(mm::derive_ray_head_address("http://[::1]:7070", 6379)
          == "[::1]:6379");
    CHECK(mm::derive_ray_head_address("", 6379).empty());
    CHECK(mm::derive_ray_head_address("http://", 6379).empty());

    // The planned split overrides the agent's own tp/pp ask; everything else
    // in the settings is preserved (it keys engine-attach compatibility).
    mm::EngineGroupCandidate group;
    group.tensor_parallel_size   = 2;
    group.pipeline_parallel_size = 3;
    mm::VllmSettings s;
    s.tensor_parallel_size   = 1;
    s.pipeline_parallel_size = 6;
    s.max_model_len          = 32768;
    s.dtype                  = "bfloat16";
    auto effective = mm::apply_group_plan(group, s);
    CHECK(effective.tensor_parallel_size == 2);
    CHECK(effective.pipeline_parallel_size == 3);
    CHECK(effective.max_model_len == 32768);
    CHECK(effective.dtype == "bfloat16");

    return true;
}

bool test_hf_cache_helpers() {
    // Repo-id classification: "org/name" yes, local paths no.
    CHECK(mm::util::is_hf_repo_id("Qwen/Qwen2.5-0.5B-Instruct"));
    CHECK(mm::util::is_hf_repo_id("meta-llama/Llama-3.1-8B"));
    CHECK(!mm::util::is_hf_repo_id("/models/local-dir"));
    CHECK(!mm::util::is_hf_repo_id("./rel/path"));
    CHECK(!mm::util::is_hf_repo_id("C:\\models\\x"));
    CHECK(!mm::util::is_hf_repo_id("bare-name"));
    CHECK(!mm::util::is_hf_repo_id("a/b/c"));

    // Cache dirname <-> repo id mapping ('/' encoded as '--', literal '-' kept).
    CHECK(mm::hf_repo_id_from_cache_dirname("models--Qwen--Qwen2.5-0.5B-Instruct")
          == "Qwen/Qwen2.5-0.5B-Instruct");
    CHECK(mm::hf_repo_id_from_cache_dirname("models--meta-llama--Llama-3.1-8B")
          == "meta-llama/Llama-3.1-8B");
    CHECK(mm::hf_repo_id_from_cache_dirname("datasets--foo--bar").empty());

    // Cache scan over a fake hub directory.
    auto dir = temp_test_dir("hf-hub");
    std::filesystem::create_directories(dir / "models--Qwen--Qwen2.5-0.5B-Instruct");
    std::filesystem::create_directories(dir / "models--meta-llama--Llama-3.1-8B");
    std::filesystem::create_directories(dir / "datasets--foo--bar");  // ignored
    {
        std::ofstream(dir / "version.txt") << "1";  // stray file ignored
    }
    auto found = mm::scan_hf_cache_models(dir.string());
    CHECK(found.size() == 2);
    CHECK(found[0] == "Qwen/Qwen2.5-0.5B-Instruct");          // sorted
    CHECK(found[1] == "meta-llama/Llama-3.1-8B");
    CHECK(mm::scan_hf_cache_models((dir / "does-not-exist").string()).empty());

    // Hub-dir resolution precedence.
    CHECK(mm::resolve_hf_hub_cache_dir("/cfg", "/env-hub", "/home-hf", "/home") == "/cfg");
    CHECK(mm::resolve_hf_hub_cache_dir("", "/env-hub", "/home-hf", "/home") == "/env-hub");
    {
        auto r = mm::resolve_hf_hub_cache_dir("", "", "/home-hf", "/home");
        CHECK(r == (std::filesystem::path("/home-hf") / "hub").string());
    }
    {
        auto r = mm::resolve_hf_hub_cache_dir("", "", "", "/home");
        CHECK(r == (std::filesystem::path("/home") / ".cache" / "huggingface" / "hub").string());
    }

    // Download args.
    auto args = mm::build_hf_download_args("Qwen/Qwen2.5-0.5B-Instruct", "/cache");
    CHECK(args.size() >= 2 && args[0] == "download");
    CHECK(args[1] == "Qwen/Qwen2.5-0.5B-Instruct");
    CHECK(has_arg_pair(args, "--cache-dir", "/cache"));
    auto args_no_cache = mm::build_hf_download_args("Qwen/Qwen2.5-0.5B-Instruct", "");
    CHECK(!has_arg(args_no_cache, "--cache-dir"));

    // cached_models survives the node-status JSON round trip.
    mm::NodeInfo n;
    n.cached_models = {"Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.1-8B"};
    n.vllm_runtime.status = "ready";
    n.vllm_runtime.platform = "linux";
    n.vllm_runtime.method = "auto";
    n.vllm_runtime.source_repo = mm::kOfficialVllmRepoUrl;
    n.vllm_runtime.version = "vllm 9.9.9-test";
    n.vllm_runtime.managed = true;
    n.vllm_runtime.executable_path = "/tmp/vllm/bin/vllm";
    n.vllm_runtime.accelerator = "cuda";
    n.vllm_runtime.latest_version = "9.9.10";
    n.vllm_runtime.update_available = true;
    nlohmann::json j = n;
    auto parsed = j.get<mm::NodeInfo>();
    CHECK(parsed.cached_models.size() == 2);
    CHECK(parsed.cached_models[0] == "Qwen/Qwen2.5-0.5B-Instruct");
    CHECK(parsed.vllm_runtime.status == "ready");
    CHECK(parsed.vllm_runtime.managed);
    CHECK(parsed.vllm_runtime.executable_path == "/tmp/vllm/bin/vllm");
    CHECK(parsed.vllm_runtime.accelerator == "cuda");
    CHECK(parsed.vllm_runtime.latest_version == "9.9.10");
    CHECK(parsed.vllm_runtime.update_available);

    // Pre-fetch is Linux-gated; on Windows it refuses cleanly.
#ifdef _WIN32
    CHECK(!mm::hf_prefetch_supported());
    std::string err;
    CHECK(!mm::hf_download("hf", "Qwen/Qwen2.5-0.5B-Instruct", "", &err));
    CHECK(!err.empty());
#else
    CHECK(mm::hf_prefetch_supported());
#endif

    CHECK(remove_tree(dir));
    return true;
}

bool test_ray_start_args() {
    // Head node: binds the GCS port and advertises its GPUs.
    {
        mm::RayStartConfig cfg;
        cfg.role = mm::RayRole::Head;
        cfg.port = 6379;
        cfg.num_gpus = 2;
        auto args = mm::build_ray_start_args(cfg);
        CHECK(!args.empty() && args.front() == "start");
        CHECK(has_arg(args, "--head"));
        CHECK(has_arg(args, "--port=6379"));
        CHECK(has_arg(args, "--num-gpus=2"));
        CHECK(has_arg(args, "--disable-usage-stats"));
    }

    // Worker node: connects to the head's address, no --head.
    {
        mm::RayStartConfig cfg;
        cfg.role = mm::RayRole::Worker;
        cfg.head_address = "10.0.0.1:6379";
        cfg.num_gpus = 1;
        auto args = mm::build_ray_start_args(cfg);
        CHECK(has_arg(args, "--address=10.0.0.1:6379"));
        CHECK(has_arg(args, "--num-gpus=1"));
        CHECK(!has_arg(args, "--head"));
    }

    // num_gpus 0 means "let Ray auto-detect" — flag omitted.
    {
        mm::RayStartConfig cfg;
        cfg.role = mm::RayRole::Head;
        auto args = mm::build_ray_start_args(cfg);
        for (const auto& a : args) CHECK(a.rfind("--num-gpus", 0) != 0);
    }

    // Live orchestration is Linux-only; on Windows it must refuse cleanly.
#ifdef _WIN32
    CHECK(!mm::ray_supported());
    std::string err;
    mm::RayStartConfig cfg;
    CHECK(!mm::ray_start(cfg, &err));
    CHECK(!err.empty());
#else
    CHECK(mm::ray_supported());
#endif
    return true;
}

bool test_vllm_runtime_defaults_and_args() {
    mm::VllmSettings settings;
    settings.max_model_len = 16384;
    settings.max_num_seqs = 2;
    settings.max_num_batched_tokens = 8192;
    settings.tensor_parallel_size = 2;
    settings.pipeline_parallel_size = 2;
    settings.gpu_memory_utilization = 0.8;
    settings.served_model_name = "agent-qwen";
    settings.trust_remote_code = true;
    settings.enable_prefix_caching = true;
    settings.enable_auto_tool_choice = true;
    settings.tool_call_parser = "hermes";
    settings.extra_args = {"--gpu-memory-utilization", "0.7",
                           "--disable-log-requests"};

    auto args = mm::build_vllm_server_args("Qwen/Qwen3-8B", settings, 8123);
    CHECK(args.size() >= 2);
    CHECK(args[0] == "serve");
    CHECK(args[1] == "Qwen/Qwen3-8B");
    CHECK(has_arg_pair(args, "--host", "127.0.0.1"));
    CHECK(has_arg_pair(args, "--port", "8123"));
    CHECK(has_arg_pair(args, "--max-model-len", "16384"));
    CHECK(has_arg_pair(args, "--max-num-seqs", "2"));
    CHECK(has_arg_pair(args, "--max-num-batched-tokens", "8192"));
    CHECK(has_arg_pair(args, "--tensor-parallel-size", "2"));
    CHECK(has_arg_pair(args, "--pipeline-parallel-size", "2"));
    CHECK(!has_arg_pair(args, "--gpu-memory-utilization", "0.8"));
    CHECK(has_arg_pair(args, "--gpu-memory-utilization", "0.7"));
    CHECK(has_arg_pair(args, "--served-model-name", "agent-qwen"));
    CHECK(has_arg(args, "--trust-remote-code"));
    CHECK(has_arg(args, "--enable-prefix-caching"));
    CHECK(has_arg(args, "--enable-auto-tool-choice"));
    CHECK(has_arg_pair(args, "--tool-call-parser", "hermes"));
    CHECK(has_arg(args, "--disable-log-requests"));

    // GGUF detection (drives the experimental-support advisory at launch).
    CHECK(mm::model_ref_is_gguf("/models/model.GGUF"));
    CHECK(!mm::model_ref_is_gguf("/models/model.safetensors"));
    CHECK(!mm::model_ref_is_gguf(""));

#ifdef _WIN32
    CHECK(mm::default_vllm_repo_url_for_platform() == mm::kWindowsVllmRepoUrl);
    CHECK(mm::default_vllm_branch_for_platform() == mm::kWindowsVllmBranch);
#elif defined(__APPLE__) && defined(__aarch64__)
    CHECK(mm::default_vllm_repo_url_for_platform() == mm::kMetalVllmRepoUrl);
    CHECK(mm::default_vllm_branch_for_platform() == "main");
#else
    CHECK(mm::default_vllm_repo_url_for_platform() == mm::kOfficialVllmRepoUrl);
    CHECK(mm::default_vllm_branch_for_platform() == "main");
#endif

    return true;
}

bool test_vllm_provisioner_helpers() {
    CHECK(mm::normalize_vllm_install_method("wheel") == "wheel");
    CHECK(mm::normalize_vllm_install_method("source") == "source");
    CHECK(mm::normalize_vllm_install_method("bogus") == "auto");

    CHECK(mm::is_apple_silicon_environment("macos", "aarch64"));
    CHECK(mm::is_apple_silicon_environment("macos", "arm64"));
    CHECK(!mm::is_apple_silicon_environment("macos", "x86_64"));
    CHECK(mm::default_vllm_repo_url_for_environment("windows", "x86_64")
          == mm::kWindowsVllmRepoUrl);
    CHECK(mm::default_vllm_repo_url_for_environment("linux", "x86_64")
          == mm::kOfficialVllmRepoUrl);
    CHECK(mm::default_vllm_repo_url_for_environment("macos", "aarch64")
          == mm::kMetalVllmRepoUrl);

    // True when any step's joined argv contains `needle`.
    auto plan_has = [](const std::vector<mm::VllmInstallStep>& plan,
                       const std::string& needle) {
        for (const auto& s : plan) {
            std::string joined;
            for (const auto& a : s.argv) { joined += a; joined += ' '; }
            if (joined.find(needle) != std::string::npos) return true;
        }
        return false;
    };

    auto dir = temp_test_dir("vllm-provisioner");
    mm::VllmProvisionConfig linux_cfg;
    linux_cfg.requested_executable = "definitely-missing-vllm-for-test";
    linux_cfg.provision_dir = dir.string();
    linux_cfg.platform = "linux";
    linux_cfg.arch = "x86_64";
    linux_cfg.install_method = "auto";
    linux_cfg.package_manager = "uv";  // deterministic plan (don't probe host PATH)

    auto linux_plan = mm::build_vllm_install_plan(linux_cfg);
    CHECK(plan_has(linux_plan, "uv"));
    CHECK(plan_has(linux_plan, "pip"));
    CHECK(plan_has(linux_plan, "install"));
    CHECK(plan_has(linux_plan, "--torch-backend=auto"));
    CHECK(plan_has(linux_plan, "vllm"));
    CHECK(mm::managed_vllm_executable_path(linux_cfg).string().find("bin") !=
          std::string::npos);

    mm::VllmProvisionConfig mac_cfg = linux_cfg;
    mac_cfg.platform = "macos";
    mac_cfg.arch = "aarch64";
    auto mac_plan = mm::build_vllm_install_plan(mac_cfg);
    CHECK(plan_has(mac_plan, "vllm-project/vllm-metal"));
    CHECK(plan_has(mac_plan, "install.sh"));

    bool command_ran = false;
    mm::VllmCommandRunner runner;
    runner.run = [&](const std::vector<std::string>&,
                     const std::filesystem::path&,
                     const mm::StreamLineCallback&,
                     const mm::CancelCheckCallback&,
                     std::string*) {
        command_ran = true;
        const auto exe = mm::managed_vllm_executable_path(linux_cfg);
        std::filesystem::create_directories(exe.parent_path());
        std::ofstream(exe) << "fake vllm";
        return 0;
    };
    runner.capture_first_line = [&](const std::vector<std::string>& args,
                                    const std::filesystem::path&) {
        if (!args.empty() && args.front() == mm::managed_vllm_executable_path(linux_cfg).string())
            return std::string{"vllm 9.9.9-test"};
        return std::string{};
    };
    mm::VllmProvisioner provisioner(linux_cfg, runner);
    auto ready = provisioner.ensure_runtime();
    CHECK(command_ran);
    CHECK(ready.status == "ready");
    CHECK(ready.managed);
    CHECK(ready.executable_path == mm::managed_vllm_executable_path(linux_cfg).string());
    CHECK(ready.version == "vllm 9.9.9-test");
    CHECK(mm::vllm_runtime_usable(ready));

    mm::VllmProvisionConfig disabled_cfg = linux_cfg;
    disabled_cfg.provision_dir = (dir / "disabled").string();
    disabled_cfg.auto_provision = false;
    bool disabled_ran = false;
    mm::VllmCommandRunner disabled_runner;
    disabled_runner.run = [&](const std::vector<std::string>&,
                              const std::filesystem::path&,
                              const mm::StreamLineCallback&,
                              const mm::CancelCheckCallback&,
                              std::string*) {
        disabled_ran = true;
        return 0;
    };
    disabled_runner.capture_first_line = [](const std::vector<std::string>&,
                                            const std::filesystem::path&) {
        return std::string{};
    };
    mm::VllmProvisioner disabled(disabled_cfg, disabled_runner);
    auto disabled_status = disabled.ensure_runtime();
    CHECK(!disabled_ran);
    CHECK(disabled_status.status == "disabled");
    CHECK(disabled_status.last_error.find("auto-provisioning is disabled") !=
          std::string::npos);

    mm::VllmProvisionConfig locked_cfg = linux_cfg;
    locked_cfg.provision_dir = (dir / "locked").string();
    std::filesystem::create_directories(
        std::filesystem::path(locked_cfg.provision_dir) / ".provision.lock");
    bool locked_ran = false;
    mm::VllmCommandRunner locked_runner;
    locked_runner.run = [&](const std::vector<std::string>&,
                            const std::filesystem::path&,
                            const mm::StreamLineCallback&,
                            const mm::CancelCheckCallback&,
                            std::string*) {
        locked_ran = true;
        return 0;
    };
    locked_runner.capture_first_line = [](const std::vector<std::string>&,
                                          const std::filesystem::path&) {
        return std::string{};
    };
    mm::VllmProvisioner locked(locked_cfg, locked_runner);
    auto locked_status = locked.ensure_runtime();
    CHECK(!locked_ran);
    CHECK(locked_status.status == "failed");
    CHECK(locked_status.last_error.find("already active") != std::string::npos);

    mm::VllmProvisionConfig unsupported_mac = linux_cfg;
    unsupported_mac.platform = "macos";
    unsupported_mac.arch = "x86_64";
    mm::VllmProvisioner mac_provisioner(unsupported_mac, disabled_runner);
    auto mac_status = mac_provisioner.ensure_runtime();
    CHECK(mac_status.status == "failed");
    CHECK(mac_status.last_error.find("Apple Silicon") != std::string::npos);

    CHECK(remove_tree(dir));
    return true;
}

bool test_vllm_update_flow() {
    // Version comparison is numeric, tolerant of tags/suffixes, and never
    // reports a spurious ordering for unparseable input.
    CHECK(mm::compare_vllm_versions("0.6.3", "0.6.4") < 0);
    CHECK(mm::compare_vllm_versions("0.6.4", "0.6.3") > 0);
    CHECK(mm::compare_vllm_versions("0.6.3", "0.6.3") == 0);
    CHECK(mm::compare_vllm_versions("v0.6.10", "v0.6.9") > 0);        // numeric, not lexical
    CHECK(mm::compare_vllm_versions("vllm 0.6.3.dev1+gabc", "0.6.3") == 0);
    CHECK(mm::compare_vllm_versions("1.0", "0.9.9") > 0);
    CHECK(mm::compare_vllm_versions("garbage", "0.6.3") == 0);

    // Accelerator-correct build variant per environment.
    CHECK(mm::detect_vllm_accelerator("macos", "aarch64", false, false) == "metal");
    CHECK(mm::detect_vllm_accelerator("windows", "x86_64", true, false) == "windows");
    CHECK(mm::detect_vllm_accelerator("linux", "x86_64", true, false) == "cuda");
    CHECK(mm::detect_vllm_accelerator("linux", "x86_64", false, true) == "rocm");
    CHECK(mm::detect_vllm_accelerator("linux", "x86_64", false, false) == "cpu");

    // The Linux plan pins the torch backend from the accelerator.
    auto plan_has = [](const std::vector<mm::VllmInstallStep>& plan,
                       const std::string& needle) {
        for (const auto& s : plan) {
            std::string joined;
            for (const auto& a : s.argv) { joined += a; joined += ' '; }
            if (joined.find(needle) != std::string::npos) return true;
        }
        return false;
    };

    mm::VllmProvisionConfig lc;
    lc.requested_executable = "definitely-missing-vllm-for-test";
    lc.provision_dir = temp_test_dir("vllm-update-script").string();
    lc.platform = "linux";
    lc.arch = "x86_64";
    lc.accelerator = "cpu";
    lc.package_manager = "uv";
    CHECK(plan_has(mm::build_vllm_install_plan(lc, false), "--torch-backend=cpu"));
    mm::VllmProvisionConfig lc_cuda = lc;
    lc_cuda.accelerator = "cuda";
    CHECK(plan_has(mm::build_vllm_install_plan(lc_cuda, false), "--torch-backend=auto"));

    // An upgrade run forces a wheel reinstall on Windows; a fresh install does not.
    mm::VllmProvisionConfig wc = lc;
    wc.platform = "windows";
    wc.package_manager = "pip";
    CHECK(plan_has(mm::build_vllm_install_plan(wc, /*upgrade=*/true), "--force-reinstall"));
    CHECK(!plan_has(mm::build_vllm_install_plan(wc, /*upgrade=*/false), "--force-reinstall"));
    CHECK(remove_tree(std::filesystem::path(lc.provision_dir)));

    // check_for_update() / update_runtime() with a fully injected runner.
    auto dir = temp_test_dir("vllm-update-run");
    mm::VllmProvisionConfig cfg;
    cfg.requested_executable = "definitely-missing-vllm-for-test";
    cfg.provision_dir = dir.string();
    cfg.platform = "linux";
    cfg.arch = "x86_64";
    cfg.accelerator = "cuda";
    cfg.package_manager = "pip";  // deterministic step count

    int run_count = 0;
    std::string latest = "0.6.4";
    mm::VllmCommandRunner runner;
    runner.run = [&](const std::vector<std::string>&,
                     const std::filesystem::path&,
                     const mm::StreamLineCallback&,
                     const mm::CancelCheckCallback&,
                     std::string*) {
        ++run_count;
        const auto exe = mm::managed_vllm_executable_path(cfg);
        std::filesystem::create_directories(exe.parent_path());
        std::ofstream(exe) << "fake vllm";
        return 0;
    };
    runner.capture_first_line = [&](const std::vector<std::string>& args,
                                    const std::filesystem::path&) {
        if (!args.empty() && args.front() == mm::managed_vllm_executable_path(cfg).string())
            return std::string{"0.6.3"};
        return std::string{};
    };
    runner.fetch_latest = [&](const mm::VllmProvisionConfig&) { return latest; };

    mm::VllmProvisioner prov(cfg, runner);
    auto ready = prov.ensure_runtime();
    CHECK(ready.status == "ready");
    const int after_ensure = run_count;
    CHECK(after_ensure >= 1);        // ran the install plan's step(s)
    CHECK(ready.version == "0.6.3");
    CHECK(ready.accelerator == "cuda");

    // Newer upstream -> update_available.
    auto checked = prov.check_for_update();
    CHECK(checked.latest_version == "0.6.4");
    CHECK(checked.update_available);

    // Equal upstream -> no update.
    latest = "0.6.3";
    CHECK(!prov.check_for_update().update_available);

    // Unknown upstream (offline) -> no spurious update.
    latest = "";
    auto offline = prov.check_for_update();
    CHECK(offline.latest_version.empty());
    CHECK(!offline.update_available);

    // update_runtime() forces a reinstall even though a managed exe exists.
    latest = "0.6.4";
    auto updated = prov.update_runtime();
    CHECK(updated.status == "ready");
    CHECK(run_count > after_ensure);   // installer re-ran (forced)
    CHECK(!updated.update_available);

    CHECK(remove_tree(dir));
    return true;
}

bool test_vllm_install_progress_parser() {
    auto approx = [](double v, double target) { return v > target - 0.02 && v < target + 0.02; };

    CHECK(approx(mm::parse_vllm_install_fraction("Downloading torch (2.1 GB) ... 45%"), 0.45));
    CHECK(approx(mm::parse_vllm_install_fraction("vllm  12.3/45.6 MB"), 12.3 / 45.6));
    CHECK(mm::parse_vllm_install_fraction("100%") >= 0.999);
    // No parseable progress -> indeterminate (<0).
    CHECK(mm::parse_vllm_install_fraction("Collecting vllm") < 0.0);
    CHECK(mm::parse_vllm_install_fraction("") < 0.0);
    // A path with slashes must not be mistaken for a ratio.
    CHECK(mm::parse_vllm_install_fraction("saving to /root/vllm-windows.whl") < 0.0);
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

bool test_vllm_provision_cancel() {
    auto dir = temp_test_dir("vllm-cancel");
    mm::VllmProvisionConfig cfg;
    cfg.requested_executable = "definitely-missing-vllm-for-test";
    cfg.provision_dir = dir.string();
    cfg.platform = "linux";
    cfg.arch = "x86_64";
    cfg.accelerator = "cpu";
    cfg.package_manager = "pip";

    bool command_ran = false;
    mm::VllmCommandRunner runner;
    runner.run = [&](const std::vector<std::string>&,
                     const std::filesystem::path&,
                     const mm::StreamLineCallback&,
                     const mm::CancelCheckCallback&,
                     std::string*) {
        command_ran = true;
        return 0;
    };
    runner.capture_first_line = [](const std::vector<std::string>&,
                                   const std::filesystem::path&) {
        return std::string{};
    };

    mm::VllmProvisioner provisioner(cfg, runner);
    provisioner.set_cancel_check([] { return true; });
    auto status = provisioner.ensure_runtime();
    CHECK(status.status == "failed");
    CHECK(status.last_error.find("canceled") != std::string::npos);
    CHECK(!command_ran);
    CHECK(remove_tree(dir));
    return true;
}

bool test_vllm_peer_selection() {
    auto mk = [](const std::string& id, bool conn, const std::string& accel,
                 const std::string& plat, const std::string& arch, const std::string& ver) {
        mm::NodeInfo n;
        n.id = id;
        n.connected = conn;
        n.vllm_runtime.accelerator = accel;
        n.vllm_runtime.platform = plat;
        n.vllm_runtime.version = ver;
        n.capabilities.arch = arch;
        return n;
    };

    const std::vector<mm::NodeInfo> nodes = {
        mk("src",       true,  "cuda", "linux", "x86_64", "0.6.4"),  // just updated
        mk("same_old",  true,  "cuda", "linux", "x86_64", "0.6.3"),  // same env, behind -> nudge
        mk("same_cur",  true,  "cuda", "linux", "x86_64", "0.6.4"),  // same env, converged -> skip
        mk("diff_acc",  true,  "rocm", "linux", "x86_64", "0.6.3"),  // different accelerator -> skip
        mk("diff_arch", true,  "cuda", "linux", "aarch64","0.6.3"),  // different arch -> skip
        mk("offline",   false, "cuda", "linux", "x86_64", "0.6.3"),  // disconnected -> skip
    };

    const auto peers = mm::select_vllm_update_peers(nodes, "src");
    CHECK(peers.size() == 1);
    CHECK(!peers.empty() && peers.front() == "same_old");
    // Unknown source id -> no peers.
    CHECK(mm::select_vllm_update_peers(nodes, "does-not-exist").empty());
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
            mm::VllmRuntimeStatus rt;
            rt.status = "ready";
            rt.managed = true;
            rt.executable_path = "fake-vllm";
            nlohmann::json body = {
                {"loaded_model", ""},
                {"slots", nlohmann::json::array()},
                {"cached_models", nlohmann::json::array()},
                {"max_slots", 1},
                {"slot_in_use", 0},
                {"slot_available", 1},
                {"slot_ready", 0},
                {"slot_loading", 0},
                {"slot_suspending", 0},
                {"slot_suspended", 0},
                {"slot_error", 0},
                {"vllm_runtime", rt},
                {"vllm_gpu_budget", 0.90},
                {"vllm_gpu_fraction_used", 0.0}
            };
            res.set_content(body.dump(), "application/json");
        });
        server.Post("/api/node/load-model", [&load_calls, load_ok](
            const httplib::Request&, httplib::Response& res) {
            ++load_calls;
            if (!load_ok) {
                res.status = 503;
                res.set_content(nlohmann::json{
                    {"error", "vllm runtime not ready"},
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
        cfg.vllm_settings.served_model_name = "served-agent-a";
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
        auto stream_chat = [&] {
            StreamAttempt attempt;
            for (int retry = 0; retry < 3; ++retry) {
                attempt = StreamAttempt{};
                attempt.ok = client.stream_post(
                    "/v1/agents/api-agent/chat",
                    nlohmann::json{{"message", "hello from test"}},
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

        auto chat = stream_chat();
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

bool test_tts_service_client_vllm_backend_paths() {
    const uint16_t port = find_free_test_port();
    CHECK(port != 0);
    httplib::Server server;
    std::mutex requests_mutex;
    std::vector<nlohmann::json> requests;

    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content("", "text/plain");
    });

    server.Post("/v1/audio/speech", [&](const httplib::Request& req, httplib::Response& res) {
        auto body = nlohmann::json::parse(req.body);
        {
            std::lock_guard<std::mutex> lock(requests_mutex);
            requests.push_back(body);
        }
        if (body.value("input", std::string{}) == "fail") {
            res.status = 500;
            res.set_content(
                nlohmann::json{{"error", {{"message", "vllm failure"}}}}.dump(),
                "application/json");
            return;
        }
        res.set_content(std::string{"RIFFfake-wav"}, "audio/wav");
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

    auto dir = temp_test_dir("tts-vllm");
    std::filesystem::create_directories(dir);

    mm::TtsServiceConfig config;
    config.enabled = true;
    config.backend = "vllm";
    config.vllm_base_url = "http://127.0.0.1:" + std::to_string(port);
    config.vllm_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";
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
    RECORD(client.provider_name() == "qwen3-tts-vllm");
    RECORD(client.default_synthesis_model_id() == config.vllm_model_id);

    const auto preview_path = (dir / "preview.wav").string();
    const auto descriptor_path = (dir / "voice.voice.json").string();

    mm::VoiceDesignProposal proposal;
    proposal.display_name = "Analyst Voice";
    proposal.sample_text = "Preview text.";
    proposal.language = "English";
    proposal.voice_description = "Original clear voice.";
    auto sample = client.generate_voice_sample(proposal, preview_path, descriptor_path);
    RECORD(sample.ok);
    RECORD(sample.status == 200);
    RECORD(sample.audio_path == preview_path);
    RECORD(sample.voice_clone_prompt_path == descriptor_path);
    RECORD(std::filesystem::is_regular_file(preview_path));
    RECORD(std::filesystem::is_regular_file(descriptor_path));

    nlohmann::json descriptor;
    {
        std::ifstream in(descriptor_path, std::ios::binary);
        in >> descriptor;
    }
    RECORD(descriptor["provider"] == "qwen3-tts-vllm");
    RECORD(descriptor["model"] == config.vllm_model_id);
    RECORD(descriptor["voice_description"] == "Original clear voice.");

    mm::AgentVoiceProfile profile;
    profile.id = "profile-a";
    profile.display_name = "Analyst Voice";
    profile.language = "English";
    profile.voice_description = "Should be overridden by descriptor.";
    profile.voice_clone_prompt_path = descriptor_path;

    const auto speech_path = (dir / "speech.wav").string();
    mm::TtsSynthesisRequest request;
    request.text = "Speak.";
    request.format = "wav";
    auto speech = client.synthesize(request, profile, speech_path);
    RECORD(speech.ok);
    RECORD(speech.audio_path == speech_path);
    RECORD(std::filesystem::is_regular_file(speech_path));
    {
        std::ifstream in(speech_path, std::ios::binary);
        const std::string audio((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
        RECORD(audio == "RIFFfake-wav");
    }

    request.text = "fail";
    auto failed = client.synthesize(request, profile, (dir / "failed.wav").string());
    RECORD(!failed.ok);
    RECORD(failed.status == 500);
    RECORD(failed.error.find("vllm failure") != std::string::npos);

    {
        std::lock_guard<std::mutex> lock(requests_mutex);
        RECORD(requests.size() == 3);
        if (requests.size() >= 2) {
            RECORD(requests[0]["model"] == config.vllm_model_id);
            RECORD(requests[0]["input"] == "Preview text.");
            RECORD(requests[0]["voice"] == "Original clear voice.");
            RECORD(requests[1]["input"] == "Speak.");
            RECORD(requests[1]["voice"] == "Original clear voice.");
            RECORD(requests[1]["response_format"] == "wav");
        }
    }

    server.stop();
    if (server_thread.joinable()) server_thread.join();
    RECORD(listen_ok);
    RECORD(listen_returned);
    RECORD(remove_tree(dir));
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

    // GGUF served through vLLM should warn (routing hint), still valid overall.
    mm::AgentConfig v = cfg;
    v.inference_backend = "vllm";
    auto rv = mm::validate_agent_config(v, nullptr, "", nullptr);
    bool vllm_gguf_warn = false;
    for (const auto& i : rv.issues)
        if (i.field == "inference_backend" && i.severity == mm::ValidationSeverity::Warning)
            vllm_gguf_warn = true;
    CHECK(vllm_gguf_warn);

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
    mm::SlotManager slots(46170, 46173, 2, "missing-vllm");
    slots.set_llama_server_path("missing-llama");
    slots.set_kv_cache_dir((dir / "kv").string());

    mm::RuntimeSettings s;
    s.ctx_size = 2048;
    s.parallel = 1;
    const auto id = slots.add_ready_test_slot_llama("m.gguf", "agent-l", s);

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
    // Regression: vLLM answers /health with an empty 200 body; health_check must
    // treat that (and a {"status":"ok"} body) as healthy.
    const uint16_t port = find_free_test_port();
    CHECK(port != 0);
    httplib::Server srv;
    srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;  // empty body, vLLM-style
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

bool test_llama_default_backend_and_isolation() {
    // llama.cpp is the default runtime on this branch.
    mm::AgentConfig fresh;
    CHECK(fresh.inference_backend == "llama-cpp");
    CHECK(mm::engine_backend_from_string("") == mm::EngineBackend::LlamaCpp);
    CHECK(mm::engine_backend_from_string("vllm") == mm::EngineBackend::Vllm);
    CHECK(mm::engine_backend_from_string("llama.cpp") == mm::EngineBackend::LlamaCpp);

    // A slot payload without a backend field parses as the default runtime.
    auto parsed = nlohmann::json{{"id", "s1"}}.get<mm::SlotInfo>();
    CHECK(parsed.backend == "llama-cpp");

    // An unspecified agent backend validates as llama-cpp (no vLLM-side errors).
    mm::AgentConfig cfg;
    cfg.name = "d";
    cfg.model_path = "/models/m.gguf";
    cfg.inference_backend = "";
    CHECK(mm::validate_agent_config(cfg, nullptr, "", nullptr).ok());

    // Cross-backend isolation: a vLLM load must never attach to a llama slot
    // serving the same model (default settings on both sides would otherwise
    // pass vllm_launch_compatible), and vice versa. With no real engines the
    // load falls through to a spawn that fails fast, returning empty.
    auto dir = temp_test_dir("backend-isolation");
    mm::SlotManager slots(46180, 46183, 4, "missing-vllm");
    slots.set_llama_server_path("missing-llama");
    const auto llama_id = slots.add_ready_test_slot_llama("m.gguf", "agent-l");
    CHECK(slots.load_model("m.gguf", mm::VllmSettings{}, "agent-v").empty());
    auto llama_info = slots.find_slot(llama_id);
    CHECK(llama_info.has_value());
    CHECK(std::find(llama_info->agent_ids.begin(), llama_info->agent_ids.end(),
                    "agent-v") == llama_info->agent_ids.end());

    const auto vllm_id = slots.add_ready_test_slot("v.gguf", "agent-v2");
    CHECK(slots.load_model_llama("v.gguf", mm::RuntimeSettings{}, "agent-l2").empty());
    auto vllm_info = slots.find_slot(vllm_id);
    CHECK(vllm_info.has_value());
    CHECK(vllm_info->backend == "vllm");
    CHECK(std::find(vllm_info->agent_ids.begin(), vllm_info->agent_ids.end(),
                    "agent-l2") == vllm_info->agent_ids.end());

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
        {"slot_manager_not_found_statuses", test_slot_manager_not_found_statuses},
        {"slot_lease_blocks_unload_and_suspend_while_busy",
         test_slot_lease_blocks_unload_and_suspend_while_busy},
        {"vllm_slot_info_and_suspend_without_kv_cache",
         test_vllm_slot_info_and_suspend_without_kv_cache},
        {"vllm_gpu_budget_accounting",
         test_vllm_gpu_budget_accounting},
        {"vllm_shared_slot_attach_and_detach",
         test_vllm_shared_slot_attach_and_detach},
        {"vllm_restore_attaches_and_cleans_suspended_record",
         test_vllm_restore_attaches_and_cleans_suspended_record},
        {"vllm_sleep_fallback_and_dead_wake_cleanup",
         test_vllm_sleep_fallback_and_dead_wake_cleanup},
        {"vllm_metrics_parse_and_slot_info_round_trip",
         test_vllm_metrics_parse_and_slot_info_round_trip},
        {"engine_group_planner",
         test_engine_group_planner},
        {"engine_group_launch_helpers",
         test_engine_group_launch_helpers},
        {"ray_start_args",
         test_ray_start_args},
        {"hf_cache_helpers",
         test_hf_cache_helpers},
        {"vllm_runtime_defaults_and_args",
         test_vllm_runtime_defaults_and_args},
        {"vllm_provisioner_helpers",
         test_vllm_provisioner_helpers},
        {"vllm_update_flow",
         test_vllm_update_flow},
        {"vllm_install_progress_parser",
         test_vllm_install_progress_parser},
        {"node_action_progress_json_round_trip",
         test_node_action_progress_json_round_trip},
        {"vllm_provision_cancel",
         test_vllm_provision_cancel},
        {"vllm_peer_selection",
         test_vllm_peer_selection},
        {"scheduler_skips_failed_node_current_attempt",
         test_scheduler_skips_failed_node_current_attempt},
        {"control_api_external_token_gate", test_control_api_external_token_gate},
        {"openai_compat_api_listener_and_model_catalog",
         test_openai_compat_api_listener_and_model_catalog},
        {"control_api_agent_api_mode_chat",
         test_control_api_agent_api_mode_chat},
        {"agent_voice_db_and_cache_lifecycle",
         test_agent_voice_db_and_cache_lifecycle},
        {"tts_service_client_fake_sidecar_paths",
         test_tts_service_client_fake_sidecar_paths},
        {"tts_service_client_vllm_backend_paths",
         test_tts_service_client_vllm_backend_paths},
        {"control_api_tts_routes_disabled", test_control_api_tts_routes_disabled},
        {"control_api_curation_routes", test_control_api_curation_routes},
        {"global_memory_origin_tool_and_context_metadata",
         test_global_memory_origin_tool_and_context_metadata},
        {"message_trace_events_round_trip", test_message_trace_events_round_trip},
        {"compaction_followup_trace_provenance_survives",
         test_compaction_followup_trace_provenance_survives},
        {"config_and_url_parsing_edge_cases", test_config_and_url_parsing_edge_cases},
        {"llama_server_args", test_llama_server_args},
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
        {"llama_default_backend_and_isolation",
         test_llama_default_backend_and_isolation},
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
