#include "common/inference_response_parser.hpp"
#include "common/agent.hpp"
#include "common/agent_db.hpp"
#include "common/config_file.hpp"
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
#include "control/engine_group_planner.hpp"
#include "control/control_api_server.hpp"
#include "control/node_registry.hpp"
#include "control/tts_service_client.hpp"
#include "node/llama_server_process.hpp"
#include "node/slot_manager.hpp"
#include "node/vllm_runtime.hpp"
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
    nlohmann::json j = n;
    auto parsed = j.get<mm::NodeInfo>();
    CHECK(parsed.cached_models.size() == 2);
    CHECK(parsed.cached_models[0] == "Qwen/Qwen2.5-0.5B-Instruct");

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

#ifdef _WIN32
    CHECK(mm::default_vllm_repo_url_for_platform() == mm::kWindowsVllmRepoUrl);
    CHECK(mm::default_vllm_branch_for_platform() == mm::kWindowsVllmBranch);
#else
    CHECK(mm::default_vllm_repo_url_for_platform() == mm::kOfficialVllmRepoUrl);
    CHECK(mm::default_vllm_branch_for_platform() == "main");
#endif

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
        mm::AgentScheduler scheduler(registry, (dir / "models").string());
        mm::ControlApiServer api(
            agents, queue, registry, scheduler,
            dir.string(), (dir / "models").string(), "control-secret");

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

        // mm::HttpClient opens a fresh connection per request; rapid
        // sequential connect/close cycles on Windows loopback occasionally
        // fail at the transport level (status == 0). Retry those.
        auto with_retry = [](auto&& request) {
            mm::HttpResponse resp;
            for (int attempt = 0; attempt < 3; ++attempt) {
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
    const uint16_t port = 49290;
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
    const uint16_t port = 49291;
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

        const uint16_t port = 49289;
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
        {"ray_start_args",
         test_ray_start_args},
        {"hf_cache_helpers",
         test_hf_cache_helpers},
        {"vllm_runtime_defaults_and_args",
         test_vllm_runtime_defaults_and_args},
        {"control_api_external_token_gate", test_control_api_external_token_gate},
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
