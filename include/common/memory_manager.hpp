#pragma once

#include "common/models.hpp"
#include <vector>
#include <string>

namespace mm {

class AgentDB;
class LlamaCppClient;

// Extracts memories from completed conversations and retrieves relevant
// memories to inject into context.
class MemoryManager {
public:
    MemoryManager(AgentDB& db, LlamaCppClient& llama);

    // Prompt the model to extract facts from a conversation; persist results.
    // Designed to run asynchronously after conversation ends.
    void extract_and_store_memories(const ConvId& conv_id, const AgentConfig& cfg);
    // Prompt the model to extract memories from a curated message subset.
    void extract_and_store_memories_from_messages(const ConvId& conv_id,
                                                  const std::vector<Message>& messages,
                                                  const AgentConfig& cfg);

    // Retrieve memories ranked by importance * recency score (up to max).
    std::vector<Memory> get_relevant_memories(const AgentConfig& cfg,
                                               int max = 20) const;

    // Format memory list into a string for system-prompt injection.
    static std::string format_memories_for_context(const std::vector<Memory>& memories);

private:
    AgentDB&        db_;
    LlamaCppClient& llama_;
};

} // namespace mm
