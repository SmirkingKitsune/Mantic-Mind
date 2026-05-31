#include "common/inference_response_parser.hpp"
#include "common/agent.hpp"
#include "common/util.hpp"
#include "control/agent_manager.hpp"
#include "control/agent_queue.hpp"
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

namespace {

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
