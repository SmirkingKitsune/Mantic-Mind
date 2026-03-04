#include "common/conversation_manager.hpp"
#include "common/agent_db.hpp"
#include "common/llama_cpp_client.hpp"
#include "common/util.hpp"
#include "common/logger.hpp"

#include <algorithm>
#include <sstream>

namespace mm {

static constexpr double kCompactionThreshold  = 0.80;
static constexpr int    kCompactionKeepRecent = 4;

ConversationManager::ConversationManager(AgentDB& db, LlamaCppClient& llama)
    : db_(db), llama_(llama) {}

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
                                           const AgentConfig& cfg) {
    int total   = db_.get_total_tokens(conv_id);
    int ctx_sz  = cfg.llama_settings.ctx_size;
    double ratio = ctx_sz > 0 ? static_cast<double>(total) / ctx_sz : 0.0;

    if (ratio >= kCompactionThreshold) {
        MM_INFO("Conversation {} at {:.0f}% context — compacting", conv_id, ratio * 100);
        return compact_conversation(conv_id, cfg);
    }
    return conv_id;
}

ConvId ConversationManager::force_compact(const ConvId& conv_id,
                                          const AgentConfig& cfg) {
    return compact_conversation(conv_id, cfg);
}

ConvId ConversationManager::compact_conversation(const ConvId& conv_id,
                                                   const AgentConfig& cfg) {
    auto conv = db_.load_conversation(conv_id);
    if (!conv) return conv_id;

    auto& msgs = conv->messages;
    const int keep_recent = kCompactionKeepRecent;

    // Nothing to compact if fewer messages than the keep threshold.
    if (static_cast<int>(msgs.size()) <= keep_recent)
        return conv_id;

    auto old_msgs = std::vector<Message>(msgs.begin(),
                                         msgs.end() - keep_recent);
    auto recent   = std::vector<Message>(msgs.end() - keep_recent, msgs.end());

    // ── Summarise old messages ────────────────────────────────────────────────
    std::ostringstream prompt;
    prompt << "You are summarising a conversation. Produce a concise summary "
              "that preserves all important facts, decisions, and context "
              "so the conversation can continue naturally. Output only the summary.\n\n";
    for (auto& m : old_msgs) {
        if (m.role == MessageRole::System) continue;
        prompt << (m.role == MessageRole::User ? "User" : "Assistant")
               << ": " << m.content << "\n";
    }

    InferenceRequest req;
    req.settings = cfg.llama_settings;
    req.messages = {{ .role = MessageRole::User, .content = prompt.str() }};

    Message summary = llama_.complete(req);
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
