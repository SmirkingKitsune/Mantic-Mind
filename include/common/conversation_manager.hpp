#pragma once

#include "common/models.hpp"
#include "common/memory_manager.hpp"
#include <vector>
#include <string>

namespace mm {

class AgentDB;
class LlamaCppClient;

// Manages context building and automatic compaction for a single agent.
//
// Compaction fires when total_tokens >= 0.80 * ctx_size.
// Strategy: summarise all messages except the 4 most recent, create a new
// Conversation with the summary as first user message, mark old conv inactive.
class ConversationManager {
public:
    ConversationManager(AgentDB& db, LlamaCppClient& llama);

    // Build the message list to send to the model:
    //   system prompt → memories injection → compaction_summary (if any) → messages
    std::vector<Message> build_context(const ConvId& conv_id,
                                       const AgentConfig& cfg,
                                       const std::vector<Memory>& memories) const;

    // Check token usage; compact if >= 80 % of ctx_size.
    // Returns the (possibly new) active ConvId after potential compaction.
    ConvId maybe_compact(const ConvId& conv_id, const AgentConfig& cfg);
    // Compact immediately, regardless of token threshold.
    ConvId force_compact(const ConvId& conv_id, const AgentConfig& cfg);

private:
    AgentDB&        db_;
    LlamaCppClient& llama_;

    // Summarise old messages and return a new active ConvId.
    ConvId compact_conversation(const ConvId& conv_id, const AgentConfig& cfg);

    static std::string format_memories(const std::vector<Memory>& memories);
};

} // namespace mm
