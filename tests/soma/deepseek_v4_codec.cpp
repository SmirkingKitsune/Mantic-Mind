#include "soma/arch/compressed_sparse.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

namespace {

bool check(bool value, const char* expression, int line) {
    if (value) return true;
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    return false;
}

#define CHECK(expr)                                                                                \
    do {                                                                                           \
        if (!check((expr), #expr, __LINE__)) return 1;                                             \
    } while (false)

} // namespace

int main(int argc, char** argv) {
    const auto& codec = soma::arch::compressed_sparse::prompt_codec();
    soma::PromptCodecState state;
    std::string prompt;

    ordered_json direct{
        {"messages",
         ordered_json::array({ordered_json{{"role", "system"}, {"content", "Be concise."}},
                              ordered_json{{"role", "user"}, {"content", "你好 123!"}}})}};
    CHECK(codec.encode(direct.dump(), prompt, state).ok());
    CHECK(!state.mode);
    CHECK(prompt == "<｜begin▁of▁sentence｜>Be concise.<｜User｜>你好 123!"
                    "<｜Assistant｜></think>");

    json thinking = direct;
    thinking["reasoning_effort"] = "high";
    CHECK(codec.encode(thinking.dump(), prompt, state).ok());
    CHECK(state.mode);
    CHECK(prompt.starts_with("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"));
    CHECK(prompt.ends_with("<｜Assistant｜><think>"));

    thinking["reasoning_effort"] = "max";
    CHECK(codec.encode(thinking.dump(), prompt, state).ok());
    CHECK(prompt.starts_with(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum"));
    thinking["reasoning_effort"] = "low";
    CHECK(codec.encode(thinking.dump(), prompt, state).ok());
    CHECK(prompt.starts_with("<｜begin▁of▁sentence｜>Be concise."));

    thinking["reasoning_effort"] = "medium";
    CHECK(!codec.encode(thinking.dump(), prompt, state).ok());
    direct["tool_choice"] = "required";
    CHECK(!codec.encode(direct.dump(), prompt, state).ok());
    direct["tool_choice"] = ordered_json{{"type", "function"}, {"function", "weather"}};
    CHECK(!codec.encode(direct.dump(), prompt, state).ok());

    // Official repository golden #2, adapted only by selecting the API's
    // explicit low-effort reasoning mode. Earlier reasoning is dropped while
    // the final assistant turn keeps its <think> payload byte-for-byte.
    ordered_json official_history{
        {"messages",
         ordered_json::array(
             {ordered_json{{"role", "system"},
                           {"content", "You are a helpful assistant."}},
              ordered_json{{"role", "user"}, {"content", "Hello"}},
              ordered_json{{"role", "assistant"},
                           {"reasoning_content",
                            "The user said hello, I should greet back."},
                           {"content", "Hi there! How can I help you?"}},
              ordered_json{{"role", "user"},
                           {"content", "What is the capital of France?"}},
              ordered_json{{"role", "assistant"},
                           {"reasoning_content",
                            "The user asks about the capital of France. It is Paris."},
                           {"content", "The capital of France is Paris."}}})},
        {"reasoning_effort", "low"}};
    CHECK(codec.encode(official_history.dump(), prompt, state).ok());
    CHECK(prompt ==
          "<｜begin▁of▁sentence｜>You are a helpful assistant."
          "<｜User｜>Hello<｜Assistant｜></think>Hi there! How can I help you?"
          "<｜end▁of▁sentence｜><｜User｜>What is the capital of France?"
          "<｜Assistant｜><think>The user asks about the capital of France. It is Paris."
          "</think>The capital of France is Paris.<｜end▁of▁sentence｜>");

    const ordered_json tools = ordered_json::array(
        {ordered_json{{"type", "function"},
              {"function",
               ordered_json{{"name", "weather"},
                    {"description", "Get weather"},
                    {"parameters",
                     ordered_json{
                         {"type", "object"},
                         {"properties",
                          ordered_json{{"city", ordered_json{{"type", "string"}}}}}}}}}}});
    ordered_json with_tools{
        {"messages",
         ordered_json::array({ordered_json{{"role", "user"}, {"content", "Weather?"}}})},
        {"tools", tools},
        {"tool_choice", "auto"},
        {"response_format", ordered_json{{"type", "json_object"}}}};
    CHECK(codec.encode(with_tools.dump(), prompt, state).ok());
    CHECK(prompt.find("## Tools") != std::string::npos);
    CHECK(prompt.find("<｜DSML｜invoke name=\"$TOOL_NAME\">") != std::string::npos);
    CHECK(prompt.find("{\"name\": \"weather\", \"description\": \"Get weather\", "
                      "\"parameters\": {\"type\": \"object\", \"properties\": "
                      "{\"city\": {\"type\": \"string\"}}}}") != std::string::npos);
    CHECK(prompt.find("## Response Format:") != std::string::npos);
    CHECK(prompt.find("invoke tool calls.\n\n\n## Response Format:") != std::string::npos);

    with_tools["tool_choice"] = "none";
    CHECK(codec.encode(with_tools.dump(), prompt, state).ok());
    CHECK(prompt.find("## Tools") == std::string::npos);

    // Tool results are rendered in the assistant call order even when the API
    // returns them out of order, while a following user text block keeps its
    // place after the results.
    ordered_json result_history{
        {"messages",
         ordered_json::array(
             {ordered_json{{"role", "user"}, {"content", "Run both"}},
              ordered_json{{"role", "assistant"},
                           {"content", ""},
                           {"tool_calls",
                            ordered_json::array(
                                {ordered_json{{"id", "call_a"},
                                              {"type", "function"},
                                              {"function",
                                               ordered_json{{"name", "weather"},
                                                            {"arguments", "{\"city\":\"A\"}"}}}},
                                 ordered_json{{"id", "call_b"},
                                              {"type", "function"},
                                              {"function",
                                               ordered_json{{"name", "weather"},
                                                            {"arguments", "{\"city\":\"B\"}"}}}}})}},
              ordered_json{{"role", "tool"},
                           {"tool_call_id", "call_b"},
                           {"content", "result B"}},
              ordered_json{{"role", "tool"},
                           {"tool_call_id", "call_a"},
                           {"content", "result A"}},
              ordered_json{{"role", "user"}, {"content", "Summarize"}}})},
        {"tools", tools},
        {"reasoning_effort", "low"}};
    CHECK(codec.encode(result_history.dump(), prompt, state).ok());
    const auto result_a = prompt.find("<tool_result>result A</tool_result>");
    const auto result_b = prompt.find("<tool_result>result B</tool_result>");
    const auto summarize = prompt.find("Summarize");
    CHECK(result_a != std::string::npos && result_a < result_b && result_b < summarize);

    soma::PromptMessage message;
    soma::PromptCodecState direct_state{false};
    CHECK(codec.parse("Unicode ✓", direct_state, true, true, message).ok());
    CHECK(message.content == "Unicode ✓");
    CHECK(message.reasoning_content == "");
    CHECK(message.tool_calls.empty());

    soma::PromptCodecState thinking_state{true};
    CHECK(codec.parse("reasoning 漢", thinking_state, false, false, message).ok());
    CHECK(message.reasoning_content == "reasoning 漢");
    CHECK(message.content == "");
    CHECK(message.tool_calls.empty());

    CHECK(codec.parse("reasoning 漢字</think>answer", thinking_state, true, true, message).ok());
    CHECK(message.reasoning_content == "reasoning 漢字");
    CHECK(message.content == "answer");

    const std::string completion =
        "Checking</think>Done\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"weather\">\n"
        "<｜DSML｜parameter name=\"city\" string=\"true\">東京</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    CHECK(codec.parse(completion, thinking_state, true, true, message).ok());
    CHECK(message.content == "Done");
    CHECK(message.reasoning_content == "Checking");
    CHECK(message.tool_calls.size() == 1);
    CHECK(message.tool_calls[0].name == "weather");
    CHECK(message.tool_calls[0].arguments == "{\"city\": \"東京\", \"days\": 3}");
    CHECK(json::parse(message.tool_calls[0].arguments)["city"] == "東京");
    const auto stable_id = message.tool_calls[0].id;
    CHECK(codec.parse(completion, thinking_state, true, true, message).ok());
    CHECK(message.tool_calls[0].id == stable_id);

    CHECK(!codec.parse("unfinished", thinking_state, true, false, message).ok());
    CHECK(!codec.parse("x\n\n<｜DSML｜tool_calls>\nnot-an-invoke",
                       direct_state,
                       true,
                       true,
                       message)
               .ok());

    CHECK(codec.parse("abc\n\n<｜DSML｜tool_", direct_state, false, false, message).ok());
    CHECK(message.content == "abc");

    ordered_json no_args{
        {"messages",
         ordered_json::array(
             {ordered_json{{"role", "user"}, {"content", "Ping"}},
              ordered_json{{"role", "assistant"},
                           {"content", ""},
                           {"tool_calls",
                            ordered_json::array({ordered_json{
                                {"type", "function"},
                                {"function",
                                 ordered_json{{"name", "ping"}, {"arguments", "{}"}}}}})}}})},
        {"reasoning_effort", "low"}};
    CHECK(codec.encode(no_args.dump(), prompt, state).ok());
    CHECK(prompt.find("<｜DSML｜invoke name=\"ping\">\n\n</｜DSML｜invoke>") !=
          std::string::npos);

    // Byte-exact acceptance probe against the pinned repository's unmodified
    // case 1. CTest passes the checked-in authoritative pair; an operator may
    // also point this executable at the downloaded repository's encoding dir.
    if (argc > 1) {
        const std::filesystem::path root(argv[1]);
        std::ifstream input(root / "tests" / "test_input_1.json", std::ios::binary);
        std::ifstream golden(root / "tests" / "test_output_1.txt", std::ios::binary);
        CHECK(input.good());
        CHECK(golden.good());
        ordered_json official = ordered_json::parse(input);
        official["reasoning_effort"] = "low";
        CHECK(codec.encode(official.dump(), prompt, state).ok());
        std::ostringstream expected;
        expected << golden.rdbuf();
        auto expected_bytes = expected.str();
        // The checked-in text fixture has the repository's conventional final
        // LF; upstream's golden intentionally ends at the EOS token.
        if (!expected_bytes.empty() && expected_bytes.back() == '\n') expected_bytes.pop_back();
        CHECK(prompt == expected_bytes);
    }

    std::cout << "deepseek_v4_codec: OK\n";
    return 0;
}
