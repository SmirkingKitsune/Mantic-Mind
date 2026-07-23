#pragma once

#include "common/models.hpp"
#include <string>
#include <vector>
#include <functional>

namespace mm {

class AgentDB;
class RuntimeClient;

// Extracts memories from completed conversations and retrieves relevant
// memories to inject into context.
class MemoryManager {
public:
    using CancelCheck = std::function<bool()>;

    MemoryManager(AgentDB& db, RuntimeClient& runtime);

    // Prompt the model to extract facts from a conversation; persist results.
    // Designed to run asynchronously after conversation ends.
    void extract_and_store_memories(const ConvId& conv_id,
                                    const AgentConfig& cfg,
                                    CancelCheck cancel_requested = {});
    // Prompt the model to extract memories from a curated message subset.
    void extract_and_store_memories_from_messages(const ConvId& conv_id,
                                                  const std::vector<Message>& messages,
                                                  const AgentConfig& cfg,
                                                  CancelCheck cancel_requested = {});

    // Retrieve global memories relevant to the current conversation, using
    // model-assisted selection over the highest-ranked candidates.
    std::vector<Memory> get_relevant_memories(const ConvId& conv_id,
                                              const AgentConfig& cfg,
                                              int max_candidates = 40,
                                              int max_selected = 8,
                                              CancelCheck cancel_requested = {}) const;

    // Format memory list into a string for system-prompt injection.
    static std::string format_memories_for_context(const std::vector<Memory>& memories);

private:
    AgentDB&        db_;
    RuntimeClient& runtime_;
};

} // namespace mm
