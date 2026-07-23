#include "common/conversation_manager.hpp"
#include "common/agent_db.hpp"
#include "common/runtime_client.hpp"
#include "common/util.hpp"
#include "common/logger.hpp"

#include <algorithm>
#include <sstream>

namespace mm {

static constexpr double kCompactionThreshold  = 0.80;
static constexpr int    kCompactionKeepRecent = 4;
static constexpr int    kImageTokenEstimate   = 2048;
static constexpr int    kMaxContextImages     = 8;
static constexpr int64_t kMaxContextImageBytes = 400LL * 1024 * 1024;

bool cancellation_requested(
    const ConversationManager::CancelCheck& cancel_requested) {
    if (!cancel_requested) return false;
    try { return cancel_requested(); } catch (...) { return true; }
}

struct ImageUsage {
    int count = 0;
    int64_t bytes = 0;
};

ImageUsage image_usage(AgentDB& db, const std::vector<Message>& messages) {
    ImageUsage usage;
    for (const auto& message : messages) {
        for (const auto& part : message.content_parts) {
            if (part.type != "image_attachment") continue;
            ++usage.count;
            if (auto attachment = db.get_attachment(part.attachment_id))
                usage.bytes += attachment->size_bytes;
        }
    }
    return usage;
}

ConversationManager::ConversationManager(AgentDB& db, RuntimeClient& runtime)
    : db_(db), runtime_(runtime) {}

// ── Context building ──────────────────────────────────────────────────────────
std::vector<Message> ConversationManager::build_context(
    const ConvId& conv_id,
    const AgentConfig& cfg,
    const std::vector<Memory>& memories) const
{
    std::vector<Message> ctx;

    // ── 1. System message ─────────────────────────────────────────────────────
    std::string system_text = cfg.system_prompt;
    if (cfg.memories_enabled && !memories.empty())
        system_text += "\n\n" + format_memories(memories);

    // ── 1b. Inject local memories ─────────────────────────────────────────────
    if (cfg.memories_enabled) {
        auto local_mems = db_.list_local_memories(conv_id);
        if (!local_mems.empty()) {
            system_text += "\n\n## Your notes for this conversation\n";
            for (const auto& lm : local_mems) {
                system_text += "- [" + lm.id + "] " + lm.content + "\n";
            }
        }
    }

    if (!system_text.empty()) {
        Message sys;
        sys.role    = MessageRole::System;
        sys.content = system_text;
        ctx.push_back(sys);
    }

    // ── 2. Conversation messages ──────────────────────────────────────────────
    auto conv = db_.load_conversation(conv_id);
    if (!conv) return ctx;

    // If this is a compacted continuation, prepend the summary as a
    // synthetic assistant message so the model has context.
    if (!conv->compaction_summary.empty()) {
        Message summary_msg;
        summary_msg.role    = MessageRole::Assistant;
        summary_msg.content = "[Previous conversation summary]\n" + conv->compaction_summary;
        ctx.push_back(summary_msg);
    }

    for (auto& m : conv->messages)
        ctx.push_back(m);

    return ctx;
}

// ── Compaction ────────────────────────────────────────────────────────────────
ConvId ConversationManager::maybe_compact(const ConvId& conv_id,
                                           const AgentConfig& cfg,
                                           CancelCheck cancel_requested) {
    if (cancellation_requested(cancel_requested)) return conv_id;
    int total   = db_.get_total_tokens(conv_id);
    int ctx_sz  = cfg.runtime_settings.ctx_size;
    // Reserve the configured completion allowance before deciding how much
    // history can remain. Long-reasoning profiles commonly allow 16K-32K
    // output tokens; comparing history against the whole context window can
    // otherwise admit a prompt that leaves no room for the response.
    const int completion_reserve = cfg.runtime_settings.max_tokens > 0 && ctx_sz > 1
        ? std::min(cfg.runtime_settings.max_tokens, ctx_sz - 1)
        : 0;
    const int prompt_budget = std::max(1, ctx_sz - completion_reserve);
    double ratio = static_cast<double>(total) / prompt_budget;
    const auto images = image_usage(db_, db_.load_messages(conv_id));
    const bool image_pressure =
        images.count >= static_cast<int>(kMaxContextImages * kCompactionThreshold) ||
        images.bytes >= static_cast<int64_t>(kMaxContextImageBytes * kCompactionThreshold);

    if (ratio >= kCompactionThreshold || image_pressure) {
        MM_INFO("Conversation {} at {:.0f}% context — compacting", conv_id, ratio * 100);
        return compact_conversation(conv_id, cfg, std::move(cancel_requested));
    }
    return conv_id;
}

ConvId ConversationManager::force_compact(const ConvId& conv_id,
                                          const AgentConfig& cfg,
                                          CancelCheck cancel_requested) {
    if (cancellation_requested(cancel_requested)) return conv_id;
    return compact_conversation(conv_id, cfg, std::move(cancel_requested));
}

ConvId ConversationManager::compact_conversation(const ConvId& conv_id,
                                                   const AgentConfig& cfg,
                                                   CancelCheck cancel_requested) {
    if (cancellation_requested(cancel_requested)) return conv_id;
    auto conv = db_.load_conversation(conv_id);
    if (!conv) return conv_id;

    auto& msgs = conv->messages;
    int keep_start = std::max(0, static_cast<int>(msgs.size()) - kCompactionKeepRecent);
    const auto all_images = image_usage(db_, msgs);

    if (all_images.count > 0) {
        std::vector<int> user_starts;
        for (int i = 0; i < static_cast<int>(msgs.size()); ++i)
            if (msgs[i].role == MessageRole::User) user_starts.push_back(i);

        if (!user_starts.empty()) {
            keep_start = user_starts.back();
            int retained_images = 0;
            int64_t retained_bytes = 0;
            int retained_tokens = 0;
            const int completion_reserve =
                cfg.runtime_settings.max_tokens > 0 &&
                cfg.runtime_settings.ctx_size > 1
                    ? std::min(cfg.runtime_settings.max_tokens,
                               cfg.runtime_settings.ctx_size - 1)
                    : 0;
            const int prompt_budget = std::max(
                1, cfg.runtime_settings.ctx_size - completion_reserve);
            const int token_budget = std::max(1, static_cast<int>(
                prompt_budget * kCompactionThreshold));
            for (int turn = static_cast<int>(user_starts.size()) - 1; turn >= 0; --turn) {
                const int begin = user_starts[turn];
                const int end = turn + 1 < static_cast<int>(user_starts.size())
                    ? user_starts[turn + 1] : static_cast<int>(msgs.size());
                std::vector<Message> turn_messages(msgs.begin() + begin, msgs.begin() + end);
                const auto turn_images = image_usage(db_, turn_messages);
                int turn_tokens = 0;
                for (const auto& message : turn_messages)
                    turn_tokens += std::max(0, message.token_count);
                turn_tokens = std::max(turn_tokens, turn_images.count * kImageTokenEstimate);
                const bool fits = retained_images + turn_images.count <= kMaxContextImages &&
                    retained_bytes + turn_images.bytes <= kMaxContextImageBytes &&
                    retained_tokens + turn_tokens <= token_budget;
                if (!fits && turn != static_cast<int>(user_starts.size()) - 1) break;
                keep_start = begin;
                retained_images += turn_images.count;
                retained_bytes += turn_images.bytes;
                retained_tokens += turn_tokens;
            }
        }
    }

    if (keep_start <= 0) return conv_id;

    auto old_msgs = std::vector<Message>(msgs.begin(), msgs.begin() + keep_start);
    auto recent   = std::vector<Message>(msgs.begin() + keep_start, msgs.end());

    // ── Summarise old messages ────────────────────────────────────────────────
    std::ostringstream prompt;
    prompt << "You are summarising a conversation. Produce a concise summary "
              "that preserves all important facts, decisions, and context "
              "so the conversation can continue naturally. Output only the summary.\n\n";
    for (auto& m : old_msgs) {
        if (m.role == MessageRole::System) continue;
        prompt << (m.role == MessageRole::User ? "User" : "Assistant")
               << ": " << m.content << "\n";
        const auto omitted = image_usage(db_, std::vector<Message>{m}).count;
        if (omitted > 0)
            prompt << "[" << omitted << " image attachment(s) omitted from summary input]\n";
    }

    InferenceRequest req;
    req.model = cfg.model_path;
    req.settings = cfg.runtime_settings;
    req.messages = {{ .role = MessageRole::User, .content = prompt.str() }};

    Message summary = runtime_.complete(req, cancel_requested);
    if (cancellation_requested(cancel_requested)) return conv_id;
    if (summary.content.empty()) {
        MM_WARN("Compaction summary is empty — skipping compaction");
        return conv_id;
    }

    // ── Create new conversation ───────────────────────────────────────────────
    ConvId new_id = db_.create_conversation(conv->title + " (continued)");
    Conversation new_conv;
    new_conv.id                 = new_id;
    new_conv.agent_id           = conv->agent_id;
    new_conv.title              = conv->title + " (continued)";
    new_conv.is_active          = true;
    new_conv.compaction_summary = summary.content;
    new_conv.parent_conv_id     = conv_id;
    new_conv.created_at_ms      = util::now_ms();
    new_conv.updated_at_ms      = util::now_ms();
    db_.save_conversation(new_conv);

    // Re-append the recent messages into the new conversation.
    for (int i = 0; i < static_cast<int>(recent.size()); ++i)
        db_.append_message(new_id, recent[i], i);

    // Transfer local memories from old conversation to new one.
    db_.transfer_local_memories(conv_id, new_id);

    // Mark old conversation inactive.
    Conversation old_conv = *conv;
    old_conv.is_active = false;
    db_.save_conversation(old_conv);

    db_.set_active_conversation(new_id);
    MM_INFO("Compaction done: {} → {}", conv_id, new_id);
    return new_id;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
std::string ConversationManager::format_memories(const std::vector<Memory>& memories) {
    return MemoryManager::format_memories_for_context(memories);
}

} // namespace mm
