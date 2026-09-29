#include "common/conversation_context.hpp"

#include "common/util.hpp"

#include <algorithm>
#include <iterator>
#include <utility>

namespace mm {

namespace {

constexpr const char* kFoldInstructions =
    "You maintain the running summary of a conversation, so that it can continue after its "
    "older messages are no longer shown. Merge the new messages into the summary. Keep every "
    "fact, decision, name, number, preference, open question and commitment; drop pleasantries "
    "and repetition. Refer to the participants as the user and the assistant. Output only the "
    "updated summary.";

constexpr const char* kCondenseInstructions =
    "Shorten this summary of a conversation. Keep every fact, decision, name, number, "
    "preference, open question and commitment; drop what is least likely to matter later. "
    "Output only the shortened summary.";

/// Tokens a chat template adds around each message, as an estimate.
constexpr int kPerMessageOverhead = 8;

/// The least room a chunk may have; below it, compaction cannot make progress.
constexpr int kMinChunkTokens = 64;

void append_block(std::string& dst, const std::string& text) {
    if (text.empty()) return;
    if (!dst.empty()) dst += "\n\n";
    dst += text;
}

bool is_image_part(const MessageContentPart& part) {
    return part.type == "image_attachment" || part.type == "image_url";
}

/// The text a message sends: its content (or, lacking one, its text parts)
/// and its tool calls.
std::size_t sent_bytes(const Message& m) {
    std::size_t bytes = m.content.size();
    if (m.content.empty()) {
        for (const auto& part : m.content_parts)
            if (!is_image_part(part)) bytes += part.text.size();
    }
    for (const auto& call : m.tool_calls) bytes += call.function_name.size() + call.arguments_json.size();
    return bytes;
}

// UTF-8-safe cuts: never split a multi-byte sequence.
std::string safe_prefix(const std::string& text, std::size_t bytes) {
    if (bytes >= text.size()) return text;
    while (bytes > 0 && (static_cast<unsigned char>(text[bytes]) & 0xC0) == 0x80) --bytes;
    return text.substr(0, bytes);
}

std::string safe_suffix(const std::string& text, std::size_t bytes) {
    if (bytes >= text.size()) return text;
    std::size_t start = text.size() - bytes;
    while (start < text.size() && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) ++start;
    return text.substr(start);
}

/// `text` cut to about `tokens`, keeping its beginning and its end.
std::string truncate_to_tokens(const std::string& text, int tokens) {
    const std::size_t max_bytes = static_cast<std::size_t>(std::max(tokens, 1)) * 4;
    if (text.size() <= max_bytes) return text;
    const std::string marker =
        "\n[... " + std::to_string(text.size() - max_bytes) + " bytes omitted ...]\n";
    if (max_bytes <= marker.size() + 16) return safe_prefix(text, max_bytes);
    const std::size_t keep = max_bytes - marker.size();
    const std::string head = safe_prefix(text, keep * 2 / 3);
    return head + marker + safe_suffix(text, keep - head.size());
}

/// One message as a line of the summarizer's input. Images are named, not sent:
/// the summarizer reads text.
std::string render_for_summary(const Message& m) {
    std::string text;
    if (m.content_parts.empty()) {
        text = m.content;
    } else {
        bool had_text = false;
        for (const auto& part : m.content_parts) {
            if (is_image_part(part)) {
                if (!text.empty()) text += " ";
                text += "[image]";
            } else if (!part.text.empty()) {
                had_text = true;
                if (!text.empty()) text += " ";
                text += part.text;
            }
        }
        if (!had_text && !m.content.empty()) text = m.content + " " + text;
    }
    switch (m.role) {
        case MessageRole::User:
            return "User: " + text;
        case MessageRole::Assistant: {
            std::string line = "Assistant: " + text;
            for (const auto& call : m.tool_calls)
                line += " [called " + call.function_name + "(" + call.arguments_json + ")]";
            return line;
        }
        case MessageRole::Tool:
            return "Tool result: " + text;
        case MessageRole::System:
            return "System: " + text;
    }
    return text;
}

std::string fold_user_text(const std::string& summary, const std::vector<std::string>& lines) {
    std::string text = "Summary so far:\n";
    text += summary.empty() ? "(none yet)" : summary;
    text += "\n\nNew messages to fold in:\n";
    for (const auto& line : lines) {
        text += line;
        text += "\n";
    }
    return text;
}

std::vector<Message> two_messages(const std::string& system, const std::string& user) {
    std::vector<Message> prompt(2);
    prompt[0].role = MessageRole::System;
    prompt[0].content = system;
    prompt[1].role = MessageRole::User;
    prompt[1].content = user;
    return prompt;
}

std::string format_global_memories(const std::vector<Memory>& memories, bool tools_offered) {
    if (memories.empty()) return {};
    std::string text = "## What you remember about this user\n"
                       "From earlier conversations, most important first.";
    if (tools_offered)
        text += " Each has an id; get_global_memory_origin shows the conversation it came from.";
    text += "\n";
    for (const auto& memory : memories) {
        text += "- ";
        if (tools_offered) text += "[" + memory.id + "] ";
        text += memory.content + "\n";
    }
    return text;
}

std::string format_local_memories(const std::vector<LocalMemory>& memories, bool tools_offered) {
    if (memories.empty()) return {};
    std::string text = "## Your notes for this conversation\n";
    for (const auto& memory : memories) {
        text += "- ";
        if (tools_offered) text += "[" + memory.id + "] ";
        text += memory.content + "\n";
    }
    return text;
}

} // namespace

// ── Estimates ─────────────────────────────────────────────────────────────────

int prompt_budget(const RuntimeSettings& settings) {
    const int ctx = settings.ctx_size;
    const int reserve = settings.max_tokens > 0 && ctx > 1 ? std::min(settings.max_tokens, ctx - 1) : 0;
    return std::max(1, ctx - reserve);
}

int estimate_tokens(const std::string& text) {
    return static_cast<int>((text.size() + 3) / 4);
}

int image_count(const Message& message) {
    return static_cast<int>(
        std::count_if(message.content_parts.begin(), message.content_parts.end(), is_image_part));
}

int estimate_tokens(const Message& message, bool vision) {
    // The recorded count of an assistant message covers its reasoning too, which
    // is never sent back; without this a reasoning model's history would look
    // several times its real size and compact far too early.
    int recorded = message.token_count;
    if (!message.thinking_text.empty())
        recorded = std::max(0, recorded - estimate_tokens(message.thinking_text));
    const int text = static_cast<int>((sent_bytes(message) + 3) / 4);
    int tokens = std::max(recorded, text);
    if (vision) tokens += kImageTokenEstimate * image_count(message);
    return tokens;
}

// ── The context ───────────────────────────────────────────────────────────────

std::string summary_section(const std::string& summary) {
    if (util::trim(summary).empty()) return {};
    return "## Earlier in this conversation\n"
           "The messages before this point are no longer shown. This is a summary of them:\n" +
           summary;
}

std::string build_system_text(const UserProfile& user,
                              const Conversation& conversation,
                              const std::vector<Memory>& global_memories,
                              const std::vector<LocalMemory>& local_memories,
                              bool tools_offered) {
    std::string text;
    append_block(text, util::trim(user.custom_instructions));
    append_block(text, util::trim(conversation.system_prompt));
    if (user.memories_enabled) {
        append_block(text, format_global_memories(global_memories, tools_offered));
        append_block(text, format_local_memories(local_memories, tools_offered));
    }
    append_block(text, summary_section(conversation.summary));
    return text;
}

std::vector<Message> build_turn_context(const std::string& system_text,
                                        std::vector<Message> tail,
                                        const ModelCapabilities& caps,
                                        const AttachmentNamer& name_of) {
    std::vector<Message> out;
    out.reserve(tail.size() + 1);
    if (!system_text.empty()) {
        Message system;
        system.role = MessageRole::System;
        system.content = system_text;
        out.push_back(std::move(system));
    }
    auto readable = sanitize_history(std::move(tail), caps, name_of);
    out.insert(out.end(), std::make_move_iterator(readable.begin()),
               std::make_move_iterator(readable.end()));
    return out;
}

// ── Rolling compaction ────────────────────────────────────────────────────────

const char* to_string(CompactionResult::Status status) {
    switch (status) {
        case CompactionResult::Status::NotNeeded: return "not_needed";
        case CompactionResult::Status::Compacted: return "compacted";
        case CompactionResult::Status::Failed: return "failed";
        case CompactionResult::Status::Overflow: return "context_overflow";
    }
    return "unknown";
}

RollingCompactor::RollingCompactor(Summarize summarize,
                                   SaveSummary save,
                                   AttachmentBytes bytes_of,
                                   CompactionPolicy policy)
    : summarize_(std::move(summarize)),
      save_(std::move(save)),
      bytes_of_(std::move(bytes_of)),
      policy_(policy) {}

int RollingCompactor::summary_cap(int budget) {
    return std::clamp(budget / 4, 256, 2048);
}

CompactionResult RollingCompactor::run(const CompactionRequest& request) const {
    CompactionResult result;
    result.summary = request.summary;
    result.summary_through_seq = request.summary_through_seq;

    const int budget = std::max(1, request.budget);
    const auto& messages = request.messages;
    const std::size_t n = messages.size();

    // Suffix sums: what the prompt costs from message i onwards.
    std::vector<long long> tail_tokens(n + 1, 0);
    std::vector<int> tail_images(n + 1, 0);
    std::vector<std::int64_t> tail_bytes(n + 1, 0);
    for (std::size_t i = n; i-- > 0;) {
        tail_tokens[i] = tail_tokens[i + 1] + estimate_tokens(messages[i], request.vision);
        int images = 0;
        std::int64_t bytes = 0;
        if (request.vision) {
            for (const auto& part : messages[i].content_parts) {
                if (!is_image_part(part)) continue;
                ++images;
                if (bytes_of_ && part.type == "image_attachment") bytes += bytes_of_(part.attachment_id);
            }
        }
        tail_images[i] = tail_images[i + 1] + images;
        tail_bytes[i] = tail_bytes[i + 1] + bytes;
    }
    const auto prompt_with = [&](const std::string& summary, std::size_t from) {
        const long long total = static_cast<long long>(request.base_system_tokens) +
                                estimate_tokens(summary_section(summary)) + tail_tokens[from];
        return static_cast<int>(std::min<long long>(total, 1LL << 30));
    };

    result.prompt_tokens = prompt_with(request.summary, 0);
    const bool image_pressure =
        request.vision &&
        (tail_images[0] >= static_cast<int>(policy_.max_images * policy_.trigger) ||
         tail_bytes[0] >= static_cast<std::int64_t>(policy_.max_image_bytes * policy_.trigger));
    const bool needed = request.force || image_pressure ||
                        result.prompt_tokens >= static_cast<int>(budget * policy_.trigger);
    if (!needed) return result;

    // ── What stays: the longest tail that starts at a user turn and fits ──────
    std::vector<std::size_t> user_starts;
    for (std::size_t i = 0; i < n; ++i)
        if (messages[i].role == MessageRole::User) user_starts.push_back(i);
    std::size_t tail_start = user_starts.empty() ? n : user_starts.back();
    const long long keep = static_cast<long long>(budget * policy_.keep_tail);
    // A manual compaction keeps only the latest turn: asked to compact, it
    // compacts, rather than finding that everything still fits.
    for (std::size_t k = request.force ? 0 : user_starts.size(); k-- > 1;) {
        const std::size_t u = user_starts[k - 1];
        const bool fits = tail_tokens[u] <= keep &&
                          (!request.vision || (tail_images[u] <= policy_.max_images &&
                                               tail_bytes[u] <= policy_.max_image_bytes));
        if (!fits) break;
        tail_start = u;
    }

    if (tail_start == 0) {
        // Nothing before the tail: there is nothing to fold.
        if (result.prompt_tokens > budget) {
            result.status = CompactionResult::Status::Overflow;
            result.error = "the latest turn alone does not fit this model's context window";
        }
        return result;
    }

    // ── Folding ──────────────────────────────────────────────────────────────
    const int cap = summary_cap(budget);
    // What a summarizer prompt may hold: the window, less the summary it
    // writes, less the slack for the template and for estimating by bytes.
    const int call_room = budget - cap - policy_.margin - 2 * kPerMessageOverhead;
    const int fold_fixed = estimate_tokens(kFoldInstructions) + estimate_tokens(fold_user_text({}, {}));

    std::string summary = request.summary;
    int watermark = request.summary_through_seq;

    const auto fail = [&](std::string error) {
        result.status = CompactionResult::Status::Failed;
        result.error = std::move(error);
        result.summary = summary;
        result.summary_through_seq = watermark;
        result.prompt_tokens = prompt_with(summary, 0);
        return result;
    };
    const auto call = [&](const std::string& system, const std::string& user)
        -> std::optional<std::string> {
        ++result.summarizer_calls;
        auto out = summarize_ ? summarize_(two_messages(system, user), cap) : std::nullopt;
        if (!out || util::trim(*out).empty()) return std::nullopt;
        return util::trim(*out);
    };
    // A summary too big for this model to extend — one written by a model with
    // a larger window — is condensed first, in a call of its own.
    const auto condense = [&]() -> bool {
        const int room = call_room - estimate_tokens(kCondenseInstructions);
        if (room < kMinChunkTokens) return false;
        auto shorter = call(kCondenseInstructions, truncate_to_tokens(summary, room));
        if (!shorter) return false;
        summary = std::move(*shorter);
        return !save_ || save_(summary, watermark);
    };

    if (call_room - fold_fixed < kMinChunkTokens)
        return fail("this model's context window is too small to compact in");
    if (estimate_tokens(summary) > cap && !condense())
        return fail("could not condense the existing summary for this model");

    std::size_t i = 0;
    while (i < tail_start) {
        int room = call_room - fold_fixed - estimate_tokens(summary);
        if (room < kMinChunkTokens) {
            if (!condense()) return fail("could not condense the summary to make room");
            room = call_room - fold_fixed - estimate_tokens(summary);
            if (room < kMinChunkTokens) return fail("the summary leaves no room to fold into");
        }
        std::vector<std::string> lines;
        int used = 0;
        int last_seq = watermark;
        while (i < tail_start) {
            std::string line = render_for_summary(messages[i]);
            const int cost = estimate_tokens(line) + 1;
            if (!lines.empty() && used + cost > room) break;
            if (lines.empty() && cost > room) line = truncate_to_tokens(line, room - 1);
            used += estimate_tokens(line) + 1;
            lines.push_back(std::move(line));
            last_seq = messages[i].sequence_num;
            ++i;
        }
        auto folded = call(kFoldInstructions, fold_user_text(summary, lines));
        if (!folded) return fail("the summarizer failed; progress up to the last saved chunk is kept");
        summary = std::move(*folded);
        watermark = last_seq;
        if (save_ && !save_(summary, watermark)) return fail("the store refused the summary");
    }

    result.summary = summary;
    result.summary_through_seq = watermark;
    result.prompt_tokens = prompt_with(summary, tail_start);
    if (result.prompt_tokens > budget) {
        result.status = CompactionResult::Status::Overflow;
        result.error = "the conversation does not fit this model's context window even compacted";
        return result;
    }
    result.status = CompactionResult::Status::Compacted;
    return result;
}

} // namespace mm
