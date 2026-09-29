#pragma once

#include "common/models.hpp"

#include <string>
#include <vector>

namespace mm {

class UserDB;
class RuntimeClient;

/// Memory selection and extraction for one user, answered by one model.
///
/// Which model is the CALLER's decision — the turn's for relevance, the
/// request's or the conversation's for extraction — and arrives here as a
/// client, the name its engine knows the model by, and its settings. A memory
/// manager used to read all three off an agent, which is how extraction came to
/// run on whatever model the agent happened to have.
class MemoryManager {
public:
    MemoryManager(UserDB& db,
                  RuntimeClient& runtime,
                  std::string request_model,
                  RuntimeSettings settings);

    /// Store global memories the model extracts from `messages`, each traced to
    /// `conv_id`. Returns how many were stored, or -1 when the model failed or
    /// answered with something that is not the JSON asked for.
    int extract_and_store_memories_from_messages(const ConvId& conv_id,
                                                 const std::vector<Message>& messages);

    /// The global memories that matter to the next reply in `conv_id`, chosen by
    /// the model from the highest-ranked candidates. `instructions` is the text
    /// the reply will be written under (the user's and the conversation's), so
    /// the choice can weigh it. Falls back to the top few candidates when the
    /// model cannot answer.
    std::vector<Memory> get_relevant_memories(const ConvId& conv_id,
                                              const std::string& instructions,
                                              int max_candidates = 40,
                                              int max_selected = 8) const;

private:
    UserDB& db_;
    RuntimeClient& runtime_;
    std::string request_model_;
    RuntimeSettings settings_;
};

} // namespace mm
