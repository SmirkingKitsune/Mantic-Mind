#pragma once

// Mantic-Mind — who is asking.
//
// A user owns conversations, the local memories inside each one, and the global
// memories that carry across all of them. A user does NOT own a model: each
// conversation names one from the served-model catalog, a turn may switch it,
// and every assistant message records which model wrote it.
//
// This is the persona half of what used to be an agent, plus the store an agent
// kept to itself. An agent welded both to a single model, so changing the model
// meant changing the agent — and nothing it had said could be answered by any
// other model.

#include "common/models.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

namespace mm {

struct UserProfile {
    /// Immutable (util::is_valid_user_id). It names the user's directory and
    /// token bindings key on it, so a rename would have to move both at once —
    /// and a binding that missed the move would reach nobody, or somebody new.
    UserId id;
    std::string display_name;
    /// Leads the system message of every conversation this user has.
    std::string custom_instructions;
    /// A served-model id, used when neither the turn nor its conversation names
    /// one. Empty: every turn must.
    std::string default_model;
    /// Global and local memories are injected into context, and a conversation
    /// that stops being active is reviewed for global ones.
    bool memories_enabled = true;
    /// The memory tools. Offered only when memories are on too, and only to a
    /// model whose engine can call tools.
    bool tools_enabled = false;
    std::int64_t created_at_ms = 0;
    std::int64_t updated_at_ms = 0;
};

void to_json(nlohmann::json& j, const UserProfile& p);

/// Lenient: every field optional, wrong types ignored. For reading what the API
/// wrote; input from a client goes through apply_profile_patch.
void from_json(const nlohmann::json& j, UserProfile& p);

/// Apply the fields present in `patch` to `p`. Refuses what is not a profile
/// field or has the wrong type rather than dropping it — a misspelled key that
/// silently did nothing would read as "the server ignored my setting". `id` may
/// appear only if it equals p.id. The timestamps are the server's and are
/// ignored, so a profile read with GET can be sent back with PUT unchanged.
/// Returns false with `out_error` naming the field.
bool apply_profile_patch(const nlohmann::json& patch, UserProfile& p, std::string& out_error);

} // namespace mm
