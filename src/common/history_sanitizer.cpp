#include "common/history_sanitizer.hpp"

#include <algorithm>
#include <map>
#include <utility>

namespace mm {

namespace {

bool is_image_part(const MessageContentPart& part) {
    return part.type == "image_attachment" || part.type == "image_url";
}

void append_block(std::string& dst, const std::string& text) {
    if (text.empty()) return;
    if (!dst.empty()) dst += "\n\n";
    dst += text;
}

/// The message as text: its parts in order, each image replaced by a
/// placeholder. `content` is rebuilt too, because the request builder sends
/// plain `content` whenever no part is an image.
void strip_images(Message& m, const AttachmentNamer& name_of) {
    if (std::none_of(m.content_parts.begin(), m.content_parts.end(), is_image_part)) return;
    std::string text;
    bool had_text_part = false;
    for (const auto& part : m.content_parts) {
        if (is_image_part(part)) {
            std::string name;
            if (part.type == "image_attachment" && name_of) name = name_of(part.attachment_id);
            if (!text.empty()) text += "\n";
            text += image_placeholder(name);
        } else if (!part.text.empty()) {
            had_text_part = true;
            if (!text.empty()) text += "\n";
            text += part.text;
        }
    }
    // A turn stored with its words only in `content` must not lose them.
    if (!had_text_part && !m.content.empty()) text = m.content + "\n" + text;
    m.content = std::move(text);
    m.content_parts.clear();
}

std::string describe_call(const ToolCall& call) {
    return call.function_name + "(" + (call.arguments_json.empty() ? "{}" : call.arguments_json) +
           ")";
}

/// Folds each tool round into one assistant message. A round starts at an
/// assistant message carrying tool calls and runs through the tool results and
/// assistant replies that follow, up to the next user or system message.
std::vector<Message> fold_tool_rounds(std::vector<Message> messages) {
    std::vector<Message> out;
    out.reserve(messages.size());
    for (std::size_t i = 0; i < messages.size();) {
        Message& m = messages[i];
        const bool starts_round = m.role == MessageRole::Assistant && !m.tool_calls.empty();
        if (!starts_round) {
            // A tool result with no call before it (history that began mid-round)
            // is still text the model should see.
            if (m.role == MessageRole::Tool) {
                Message note;
                note.role = MessageRole::Assistant;
                note.content = "[tool result] " + m.content;
                note.model_id = m.model_id;
                note.sequence_num = m.sequence_num;
                note.timestamp_ms = m.timestamp_ms;
                out.push_back(std::move(note));
            } else {
                out.push_back(std::move(m));
            }
            ++i;
            continue;
        }

        Message folded;
        folded.role = MessageRole::Assistant;
        folded.model_id = m.model_id;
        folded.sequence_num = m.sequence_num;
        folded.timestamp_ms = m.timestamp_ms;
        folded.token_count = 0;

        std::map<std::string, std::string> results; // tool_call_id -> content
        std::vector<ToolCall> calls;
        std::size_t j = i;
        for (; j < messages.size(); ++j) {
            const Message& r = messages[j];
            if (r.role == MessageRole::User || r.role == MessageRole::System) break;
            folded.token_count += r.token_count;
            if (r.role == MessageRole::Tool) {
                results[r.tool_call_id] = r.content;
                continue;
            }
            // An assistant message inside the round: flush the calls made so far
            // with whatever results have arrived, then its own text.
            if (!calls.empty()) {
                for (const auto& call : calls) {
                    std::string line = "[called " + describe_call(call);
                    const auto hit = results.find(call.id);
                    if (hit != results.end()) line += " -> " + hit->second;
                    append_block(folded.content, line + "]");
                }
                calls.clear();
            }
            append_block(folded.content, r.content);
            calls = r.tool_calls;
            // The LAST assistant message in the round is the one that answered;
            // it is the model a reader should see credited.
            if (!r.model_id.empty()) folded.model_id = r.model_id;
        }
        for (const auto& call : calls) {
            std::string line = "[called " + describe_call(call);
            const auto hit = results.find(call.id);
            if (hit != results.end()) line += " -> " + hit->second;
            append_block(folded.content, line + "]");
        }
        out.push_back(std::move(folded));
        i = j;
    }
    return out;
}

} // namespace

std::string image_placeholder(const std::string& name) {
    return name.empty() ? "[image omitted]" : "[image omitted: " + name + "]";
}

std::vector<Message> sanitize_history(std::vector<Message> messages,
                                      const ModelCapabilities& caps,
                                      const AttachmentNamer& name_of) {
    if (!caps.vision) {
        for (auto& m : messages) strip_images(m, name_of);
    }
    if (!caps.tools) messages = fold_tool_rounds(std::move(messages));
    return messages;
}

} // namespace mm
