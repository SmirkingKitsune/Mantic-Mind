#include "common/inference_response_parser.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>

namespace mm::inference {
namespace {

std::string json_value_to_string(const nlohmann::json& value) {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_null()) return {};
    return value.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::optional<ToolCall> parse_tool_call_object(const nlohmann::json& tc) {
    ToolCall out;
    if (tc.contains("id") && !tc["id"].is_null()) {
        out.id = tc["id"].get<std::string>();
    }
    if (tc.contains("function") && tc["function"].is_object()) {
        const auto& fn = tc["function"];
        if (fn.contains("name") && !fn["name"].is_null()) {
            out.function_name = fn["name"].get<std::string>();
        }
        if (fn.contains("arguments")) {
            out.arguments_json = json_value_to_string(fn["arguments"]);
        }
    }
    if (out.id.empty() && out.function_name.empty() && out.arguments_json.empty()) {
        return std::nullopt;
    }
    return out;
}

void set_error(std::string* error, const std::string& value) {
    if (error) *error = value;
}

} // namespace

ThinkingParts ThinkingExtractor::append(const std::string& raw) {
    buffer_ += raw;

    ThinkingParts out;
    static const std::string kOpen = "<think>";
    static const std::string kClose = "</think>";

    bool progress = true;
    while (progress) {
        progress = false;
        if (!in_think_) {
            auto pos = buffer_.find(kOpen);
            if (pos != std::string::npos) {
                out.content += buffer_.substr(0, pos);
                buffer_ = buffer_.substr(pos + kOpen.size());
                in_think_ = true;
                progress = true;
            }
        } else {
            auto pos = buffer_.find(kClose);
            if (pos != std::string::npos) {
                out.thinking += buffer_.substr(0, pos);
                buffer_ = buffer_.substr(pos + kClose.size());
                in_think_ = false;
                progress = true;
            }
        }
    }

    const size_t guard = std::max(kOpen.size(), kClose.size()) - 1;
    if (buffer_.size() > guard) {
        const std::string emit = buffer_.substr(0, buffer_.size() - guard);
        buffer_ = buffer_.substr(buffer_.size() - guard);
        if (in_think_) out.thinking += emit;
        else out.content += emit;
    }

    return out;
}

ThinkingParts ThinkingExtractor::flush() {
    ThinkingParts out;
    if (in_think_) out.thinking = buffer_;
    else out.content = buffer_;
    buffer_.clear();
    in_think_ = false;
    return out;
}

std::optional<OpenAISseDelta> parse_openai_sse_delta(const std::string& data_json,
                                                     std::string* error) {
    try {
        auto j = nlohmann::json::parse(data_json);
        OpenAISseDelta out;

        if (j.contains("usage") && j["usage"].is_object()) {
            out.tokens_used = j["usage"].value("completion_tokens", 0);
        }
        if (!j.contains("choices") || j["choices"].empty()) {
            return out;
        }

        const auto& choice = j["choices"][0];
        const auto delta = choice.value("delta", nlohmann::json::object());

        if (delta.contains("content") && !delta["content"].is_null()) {
            out.content = delta["content"].get<std::string>();
        }

        if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
            int fallback_index = 0;
            for (const auto& tc_json : delta["tool_calls"]) {
                auto call = parse_tool_call_object(tc_json);
                if (!call) {
                    ++fallback_index;
                    continue;
                }

                IndexedToolCallDelta tc;
                tc.index = tc_json.value("index", fallback_index);
                tc.call = std::move(*call);
                out.tool_calls.push_back(std::move(tc));
                ++fallback_index;
            }
        }

        if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) {
            out.finish_reason = choice["finish_reason"].get<std::string>();
        }

        return out;
    } catch (const std::exception& e) {
        set_error(error, e.what());
        return std::nullopt;
    }
}

std::optional<Message> parse_openai_chat_completion(const std::string& body,
                                                    int64_t timestamp_ms,
                                                    std::string* error) {
    try {
        auto j = nlohmann::json::parse(body);
        if (!j.contains("choices") || j["choices"].empty()) {
            set_error(error, "missing choices");
            return std::nullopt;
        }

        const auto& msg = j["choices"][0]["message"];
        Message out;
        out.role = MessageRole::Assistant;
        out.timestamp_ms = timestamp_ms;
        if (j.contains("usage") && j["usage"].is_object()) {
            out.token_count = j["usage"].value("completion_tokens", 0);
        }

        ThinkingExtractor extractor;
        auto parts = extractor.append(msg.value("content", std::string{}));
        auto tail = extractor.flush();
        out.content = parts.content + tail.content;
        out.thinking_text = parts.thinking + tail.thinking;

        if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
            for (const auto& tc_json : msg["tool_calls"]) {
                auto parsed = parse_tool_call_object(tc_json);
                if (parsed && !parsed->function_name.empty()) {
                    out.tool_calls.push_back(std::move(*parsed));
                }
            }
        }

        return out;
    } catch (const std::exception& e) {
        set_error(error, e.what());
        return std::nullopt;
    }
}

} // namespace mm::inference
