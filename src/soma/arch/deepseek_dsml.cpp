#include "soma/arch/compressed_sparse.hpp"

#include <nlohmann/json.hpp>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

namespace soma::arch::compressed_sparse {
namespace {

constexpr std::string_view kBos = "<｜begin▁of▁sentence｜>";
constexpr std::string_view kEos = "<｜end▁of▁sentence｜>";
constexpr std::string_view kUser = "<｜User｜>";
constexpr std::string_view kAssistant = "<｜Assistant｜>";
constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kDsml = "｜DSML｜";
constexpr std::string_view kToolStart = "\n\n<｜DSML｜tool_calls>\n";
constexpr std::string_view kToolEnd = "</｜DSML｜tool_calls>";
constexpr std::string_view kInvokeStart = "<｜DSML｜invoke name=\"";
constexpr std::string_view kInvokeEnd = "</｜DSML｜invoke>";
constexpr std::string_view kParamStart = "<｜DSML｜parameter name=\"";
constexpr std::string_view kParamEnd = "</｜DSML｜parameter>";

constexpr std::string_view kToolsTemplate = R"(## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$TOOL_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
)";

constexpr std::string_view kHigh =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to "
    "resolve the root cause, rigorously stress-testing your logic against all potential paths, "
    "edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate "
    "step, considered alternative, and rejected hypothesis to ensure absolutely no assumption "
    "is left unchecked.\n\n";

constexpr std::string_view kMax =
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: "
    "exhaustively decompose the problem into its most fundamental components, trace every causal "
    "chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple "
    "angles and are certain that no assumption remains unchecked and no error remains "
    "undiscovered.\n\n";

Status bad(std::string message) {
    return {StatusCode::InvalidArgument, std::move(message)};
}

Status protocol(std::string message) {
    return {StatusCode::InvalidArgument, "DSML protocol error: " + std::move(message)};
}

Status content_text(const ordered_json& content, std::string& out) {
    out.clear();
    if (content.is_null()) return {};
    if (content.is_string()) {
        out = content.get<std::string>();
        return {};
    }
    if (!content.is_array()) return bad("message content must be string, null, or an array");
    bool first = true;
    for (const auto& part : content) {
        if (!part.is_object() || part.value("type", std::string{}) != "text") {
            return {StatusCode::Unsupported, "only text content parts are supported"};
        }
        if (!first) out += "\n\n";
        out += part.value("text", std::string{});
        first = false;
    }
    return {};
}

// Match Python json.dumps(..., ensure_ascii=False), which is what the official
// encoder uses. nlohmann's compact dump omits spaces after commas and colons;
// both those bytes and object-key order affect the prompt token stream.
std::string official_json(const ordered_json& value) {
    if (value.is_object()) {
        std::string out = "{";
        bool first = true;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (!first) out += ", ";
            out += ordered_json(it.key()).dump(
                -1, ' ', false, ordered_json::error_handler_t::strict);
            out += ": ";
            out += official_json(*it);
            first = false;
        }
        return out + "}";
    }
    if (value.is_array()) {
        std::string out = "[";
        for (std::size_t i = 0; i < value.size(); ++i) {
            if (i) out += ", ";
            out += official_json(value[i]);
        }
        return out + "]";
    }
    return value.dump(-1, ' ', false, ordered_json::error_handler_t::strict);
}

std::string render_tools(const ordered_json& tools) {
    std::string schemas;
    for (std::size_t i = 0; i < tools.size(); ++i) {
        if (i) schemas += '\n';
        schemas += official_json(tools[i].at("function"));
    }
    std::string rendered(kToolsTemplate);
    const auto at = rendered.find("{tool_schemas}");
    rendered.replace(at, std::string("{tool_schemas}").size(), schemas);
    return rendered;
}

Status render_call(const ordered_json& call, std::string& out) {
    if (!call.is_object() || !call.contains("function") || !call["function"].is_object()) {
        return bad("assistant tool_calls entries require function objects");
    }
    const auto& fn = call["function"];
    if (!fn.contains("name") || !fn["name"].is_string()) {
        return bad("assistant tool call function.name must be a string");
    }
    const auto name = fn["name"].get<std::string>();
    const auto raw = fn.value("arguments", std::string{"{}"});
    ordered_json args;
    try {
        args = ordered_json::parse(raw);
        if (!args.is_object()) args = ordered_json{{"arguments", raw}};
    } catch (...) {
        args = ordered_json{{"arguments", raw}};
    }

    out = std::string(kInvokeStart) + name + "\">\n";
    bool first = true;
    for (auto it = args.begin(); it != args.end(); ++it) {
        if (!first) out += '\n';
        out += std::string(kParamStart) + it.key() + "\" string=\"";
        if (it->is_string()) {
            out += "true\">" + it->get<std::string>();
        } else {
            out += "false\">" + official_json(*it);
        }
        out += kParamEnd;
        first = false;
    }
    // The official template has one newline before and one after the
    // `{arguments}` substitution.  For a no-argument function that deliberately
    // becomes a blank line rather than collapsing to a single newline.
    out += '\n';
    out += kInvokeEnd;
    return {};
}

Status preprocess(const ordered_json& request,
                  ordered_json& messages,
                  bool include_tools) {
    messages = request.at("messages");
    if (messages.empty()) return bad("messages[] must not be empty");

    const bool have_tools = include_tools && request.contains("tools") &&
                            !request["tools"].is_null() && !request["tools"].empty();
    if (have_tools && !request["tools"].is_array()) return bad("tools must be an array");

    const bool have_format = request.contains("response_format") &&
                             !request["response_format"].is_null();
    if (have_tools || have_format) {
        std::size_t target = messages.size();
        for (std::size_t i = 0; i < messages.size(); ++i) {
            const auto role = messages[i].value("role", std::string{});
            if (role == "system") {
                target = i;
                break;
            }
        }
        if (target == messages.size()) {
            for (std::size_t i = 0; i < messages.size(); ++i) {
                if (messages[i].value("role", std::string{}) == "developer") {
                    target = i;
                    break;
                }
            }
        }
        if (target == messages.size()) {
            messages.insert(
                messages.begin(), ordered_json{{"role", "system"}, {"content", ""}});
            target = 0;
        }
        if (have_tools) messages[target]["tools"] = request["tools"];
        if (have_format) messages[target]["response_format"] = request["response_format"];
    }

    ordered_json merged = ordered_json::array();
    for (auto msg : messages) {
        if (!msg.is_object()) return bad("messages entries must be objects");
        const auto role = msg.value("role", std::string{});
        if (role == "tool") {
            std::string text;
            if (auto st = content_text(msg.value("content", ordered_json{}), text); !st.ok())
                return st;
            ordered_json block{{"type", "tool_result"},
                               {"tool_use_id", msg.value("tool_call_id", std::string{})},
                               {"content", text}};
            if (!merged.empty() && merged.back().value("role", std::string{}) == "user" &&
                merged.back().contains("content_blocks")) {
                merged.back()["content_blocks"].push_back(std::move(block));
            } else {
                merged.push_back(ordered_json{
                    {"role", "user"},
                    {"content_blocks", ordered_json::array({std::move(block)})}});
            }
            continue;
        }
        if (role == "user") {
            std::string text;
            if (auto st = content_text(msg.value("content", ordered_json{}), text); !st.ok())
                return st;
            ordered_json block{{"type", "text"}, {"text", text}};
            if (!merged.empty() && merged.back().value("role", std::string{}) == "user" &&
                merged.back().contains("content_blocks") && !merged.back().contains("task")) {
                merged.back()["content_blocks"].push_back(std::move(block));
            } else {
                msg["content"] = text;
                msg["content_blocks"] = ordered_json::array({std::move(block)});
                merged.push_back(std::move(msg));
            }
            continue;
        }
        merged.push_back(std::move(msg));
    }

    std::map<std::string, std::size_t> call_order;
    for (auto& msg : merged) {
        const auto role = msg.value("role", std::string{});
        if (role == "assistant" && msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
            call_order.clear();
            for (std::size_t i = 0; i < msg["tool_calls"].size(); ++i) {
                const auto id = msg["tool_calls"][i].value("id", std::string{});
                if (!id.empty()) call_order[id] = i;
            }
        } else if (role == "user" && msg.contains("content_blocks") && !call_order.empty()) {
            auto& blocks = msg["content_blocks"];
            // Reorder only tool-result slots. Treating text and tool blocks as
            // mutually equivalent inside one std::stable_sort comparator while
            // still ordering two tool blocks violates strict weak ordering.
            // It usually looked right, but its behavior was formally undefined
            // for a user message that also carried text after tool results.
            std::vector<ordered_json> tool_results;
            for (const auto& block : blocks) {
                if (block.value("type", std::string{}) == "tool_result")
                    tool_results.push_back(block);
            }
            std::stable_sort(
                tool_results.begin(),
                tool_results.end(),
                [&](const ordered_json& a, const ordered_json& b) {
                const auto ai = call_order.find(a.value("tool_use_id", std::string{}));
                const auto bi = call_order.find(b.value("tool_use_id", std::string{}));
                const auto av = ai == call_order.end() ? call_order.size() : ai->second;
                const auto bv = bi == call_order.end() ? call_order.size() : bi->second;
                return av < bv;
                });
            std::size_t result = 0;
            for (auto& block : blocks) {
                if (block.value("type", std::string{}) == "tool_result")
                    block = std::move(tool_results[result++]);
            }
        }
    }
    messages = std::move(merged);
    return {};
}

Status encode(const ordered_json& request,
              std::string& prompt,
              PromptCodecState& state) noexcept {
    try {
        if (!request.contains("messages") || !request["messages"].is_array()) {
            return bad("messages[] is required");
        }

        bool include_tools = true;
        if (request.contains("tool_choice")) {
            if (!request["tool_choice"].is_string()) {
                return bad("tool_choice required/named forcing is not defined by the model protocol");
            }
            const auto choice = request["tool_choice"].get<std::string>();
            if (choice == "none") include_tools = false;
            else if (choice != "auto") {
                return bad("tool_choice must be omitted, 'auto', or 'none'");
            }
        }

        std::string effort;
        state.mode = request.contains("reasoning_effort");
        if (state.mode) {
            if (!request["reasoning_effort"].is_string()) {
                return bad("reasoning_effort must be one of: low, high, max");
            }
            effort = request["reasoning_effort"].get<std::string>();
            if (effort != "low" && effort != "high" && effort != "max") {
                return bad("reasoning_effort must be one of: low, high, max");
            }
        }

        ordered_json messages;
        if (auto st = preprocess(request, messages, include_tools); !st.ok()) return st;

        std::size_t last_user = messages.size();
        bool any_tools = false;
        for (std::size_t i = 0; i < messages.size(); ++i) {
            const auto role = messages[i].value("role", std::string{});
            if (role == "user" || role == "developer") last_user = i;
            any_tools |= messages[i].contains("tools") && !messages[i]["tools"].empty();
        }
        const bool drop_thinking = !any_tools;

        prompt.assign(kBos);
        if (state.mode && effort == "high") prompt += kHigh;
        if (state.mode && effort == "max") prompt += kMax;

        for (std::size_t i = 0; i < messages.size(); ++i) {
            const auto& msg = messages[i];
            const auto role = msg.value("role", std::string{});
            if (state.mode && drop_thinking && role == "developer" && i < last_user) continue;

            std::string content;
            if (auto st = content_text(msg.value("content", ordered_json{}), content); !st.ok())
                return st;

            if (role == "system") {
                prompt += content;
                if (msg.contains("tools") && !msg["tools"].empty()) {
                    prompt += "\n\n" + render_tools(msg["tools"]);
                }
                if (msg.contains("response_format") && !msg["response_format"].is_null()) {
                    prompt += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following "
                              "schema to reply:\n";
                    prompt += official_json(msg["response_format"]);
                }
            } else if (role == "developer") {
                if (content.empty()) return bad("developer messages require content");
                prompt += kUser;
                prompt += content;
                if (msg.contains("tools") && !msg["tools"].empty()) {
                    prompt += "\n\n" + render_tools(msg["tools"]);
                }
                if (msg.contains("response_format") && !msg["response_format"].is_null()) {
                    prompt += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following "
                              "schema to reply:\n";
                    prompt += official_json(msg["response_format"]);
                }
            } else if (role == "user") {
                prompt += kUser;
                if (msg.contains("content_blocks")) {
                    bool first = true;
                    for (const auto& block : msg["content_blocks"]) {
                        if (!first) prompt += "\n\n";
                        if (block.value("type", std::string{}) == "text") {
                            prompt += block.value("text", std::string{});
                        } else if (block.value("type", std::string{}) == "tool_result") {
                            prompt += "<tool_result>" + block.value("content", std::string{}) +
                                      "</tool_result>";
                        }
                        first = false;
                    }
                } else {
                    prompt += content;
                }
            } else if (role == "assistant") {
                if (state.mode && (!drop_thinking || i > last_user)) {
                    prompt += msg.value("reasoning_content", std::string{});
                    prompt += kThinkClose;
                }
                prompt += content;
                if (msg.contains("tool_calls") && !msg["tool_calls"].empty()) {
                    prompt += "\n\n<｜DSML｜tool_calls>\n";
                    for (std::size_t c = 0; c < msg["tool_calls"].size(); ++c) {
                        if (c) prompt += '\n';
                        std::string call;
                        if (auto st = render_call(msg["tool_calls"][c], call); !st.ok()) return st;
                        prompt += call;
                    }
                    prompt += "\n</｜DSML｜tool_calls>";
                }
                prompt += kEos;
            } else {
                return bad("unsupported message role '" + role + "'");
            }

            if (i + 1 < messages.size()) {
                const auto next = messages[i + 1].value("role", std::string{});
                if (next != "assistant" && next != "latest_reminder") continue;
            }
            if (role == "user" || role == "developer") {
                prompt += kAssistant;
                if (!state.mode) {
                    prompt += kThinkClose;
                } else if (!drop_thinking || i >= last_user) {
                    prompt += kThinkOpen;
                } else {
                    prompt += kThinkClose;
                }
            }
        }
        return {};
    } catch (const std::exception& e) {
        return bad(e.what());
    }
}

std::string stable_before_marker(std::string_view text, std::string_view marker) {
    std::size_t hold = 0;
    const auto max = std::min(text.size(), marker.size() - 1);
    for (std::size_t n = 1; n <= max; ++n) {
        if (text.substr(text.size() - n) == marker.substr(0, n)) hold = n;
    }
    return std::string(text.substr(0, text.size() - hold));
}

std::string call_id(std::string_view completion, std::size_t ordinal) {
    std::string material(completion);
    material.push_back('\0');
    material += std::to_string(ordinal);
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(reinterpret_cast<const unsigned char*>(material.data()), material.size(), digest.data());
    std::ostringstream out;
    out << "call_" << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < 12; ++i) out << std::setw(2) << unsigned(digest[i]);
    return out.str();
}

Status parse_calls(std::string_view text, std::string_view completion, json& calls) {
    calls = json::array();
    std::size_t pos = 0;
    while (pos < text.size()) {
        if (text.substr(pos, kToolEnd.size()) == kToolEnd) {
            pos += kToolEnd.size();
            if (pos != text.size()) return protocol("unexpected content after tool calls");
            return {};
        }
        if (text.substr(pos, kInvokeStart.size()) != kInvokeStart) {
            return protocol("expected an invoke element");
        }
        pos += kInvokeStart.size();
        const auto name_end = text.find("\">\n", pos);
        if (name_end == std::string_view::npos) return protocol("malformed invoke name");
        const std::string name(text.substr(pos, name_end - pos));
        if (name.empty()) return protocol("empty invoke name");
        pos = name_end + 3;

        ordered_json args = ordered_json::object();
        while (text.substr(pos, kParamStart.size()) == kParamStart) {
            pos += kParamStart.size();
            const auto attr = text.find("\" string=\"", pos);
            if (attr == std::string_view::npos) return protocol("malformed parameter name");
            const std::string key(text.substr(pos, attr - pos));
            pos = attr + std::string_view("\" string=\"").size();
            if (text.substr(pos, 6) == "true\">") {
                pos += 6;
                const auto end = text.find(kParamEnd, pos);
                if (end == std::string_view::npos) return protocol("unterminated string parameter");
                if (args.contains(key)) return protocol("duplicate parameter '" + key + "'");
                args[key] = std::string(text.substr(pos, end - pos));
                pos = end + kParamEnd.size();
            } else if (text.substr(pos, 7) == "false\">") {
                pos += 7;
                const auto end = text.find(kParamEnd, pos);
                if (end == std::string_view::npos) return protocol("unterminated JSON parameter");
                if (args.contains(key)) return protocol("duplicate parameter '" + key + "'");
                try {
                    args[key] = ordered_json::parse(text.substr(pos, end - pos));
                } catch (...) {
                    return protocol("invalid JSON parameter '" + key + "'");
                }
                pos = end + kParamEnd.size();
            } else {
                return protocol("parameter string attribute must be true or false");
            }
            if (text.substr(pos, 1) == "\n") ++pos;
        }
        if (text.substr(pos, kInvokeEnd.size()) != kInvokeEnd) {
            return protocol("unterminated invoke element");
        }
        pos += kInvokeEnd.size();
        if (text.substr(pos, 1) == "\n") ++pos;

        json call;
        call["id"] = call_id(completion, calls.size());
        call["type"] = "function";
        call["function"] = json{{"name", name}, {"arguments", official_json(args)}};
        calls.push_back(std::move(call));
    }
    return protocol("missing tool_calls closing element");
}

bool contains_reserved(std::string_view text) {
    return text.find(kBos) != std::string_view::npos ||
           text.find(kEos) != std::string_view::npos ||
           text.find(kThinkOpen) != std::string_view::npos ||
           text.find(kThinkClose) != std::string_view::npos ||
           text.find(kDsml) != std::string_view::npos;
}

Status parse(std::string_view completion,
             const PromptCodecState& state,
             bool final,
             bool ended_by_stop,
             json& message) noexcept {
    try {
        message = json{{"role", "assistant"},
                       {"content", ""},
                       {"reasoning_content", ""},
                       {"tool_calls", json::array()}};
        std::size_t pos = 0;
        if (state.mode) {
            const auto close = completion.find(kThinkClose);
            if (close == std::string_view::npos) {
                if (final) return protocol("thinking completion is missing </think>");
                message["reasoning_content"] = stable_before_marker(completion, kThinkClose);
                return {};
            }
            const auto reasoning = completion.substr(0, close);
            if (contains_reserved(reasoning)) return protocol("special token in reasoning content");
            message["reasoning_content"] = reasoning;
            pos = close + kThinkClose.size();
        }

        const auto body = completion.substr(pos);
        const auto tool = body.find(kToolStart);
        if (tool == std::string_view::npos) {
            if (final) {
                if (contains_reserved(body)) return protocol("unexpected special token in content");
                message["content"] = body;
            } else {
                message["content"] = stable_before_marker(body, kToolStart);
            }
            (void)ended_by_stop;
            return {};
        }

        const auto content = body.substr(0, tool);
        if (contains_reserved(content)) return protocol("special token in content");
        message["content"] = content;
        if (!final) {
            // Tool calls are emitted only after the complete block validates.
            // This prevents a malformed invocation from becoming a fabricated
            // OpenAI call merely because its prefix looked plausible.
            const auto rest = body.substr(tool + kToolStart.size());
            if (rest.find(kToolEnd) == std::string_view::npos) return {};
        }
        const auto call_text = body.substr(tool + kToolStart.size());
        if (auto st = parse_calls(call_text, completion, message["tool_calls"]); !st.ok()) {
            if (!final) {
                message["tool_calls"] = json::array();
                return {};
            }
            return st;
        }
        return {};
    } catch (const std::exception& e) {
        return protocol(e.what());
    }
}

} // namespace

const soma::PromptCodec& prompt_codec() noexcept {
    static const soma::PromptCodec codec{"dsml-v4", &encode, &parse};
    return codec;
}

} // namespace soma::arch::compressed_sparse
