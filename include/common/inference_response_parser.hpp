#pragma once

#include "common/models.hpp"

#include <optional>
#include <string>
#include <vector>

namespace mm::inference {

struct ThinkingParts {
    std::string content;
    std::string thinking;
};

class ThinkingExtractor {
public:
    ThinkingParts append(const std::string& raw);
    ThinkingParts flush();

private:
    std::string buffer_;
    bool in_think_ = false;
};

struct IndexedToolCallDelta {
    int index = 0;
    ToolCall call;
};

struct OpenAISseDelta {
    std::string content;
    std::vector<IndexedToolCallDelta> tool_calls;
    std::string finish_reason;
    int tokens_used = 0;
};

std::optional<OpenAISseDelta> parse_openai_sse_delta(const std::string& data_json,
                                                     std::string* error = nullptr);

std::optional<Message> parse_openai_chat_completion(const std::string& body,
                                                    int64_t timestamp_ms,
                                                    std::string* error = nullptr);

} // namespace mm::inference
