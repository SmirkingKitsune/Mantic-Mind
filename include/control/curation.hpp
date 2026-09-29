#pragma once

// Mantic-Mind — curation: proposed edits to a user's conversations and
// memories, reviewed by a person, then applied.
//
// Proposals are built from rules (a generic title, an empty note, whitespace)
// rather than by a model, so reviewing them costs nothing and they are the same
// every time. Applying is two steps — every proposal is validated before any is
// applied — so a batch with one bad entry changes nothing.
//
// Pure over a UserDB: no server, no model. The routes that expose it live in
// user_routes.cpp.

#include "common/models.hpp"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>

namespace mm {

class UserDB;

/// Proposals for one conversation — `conversation_id`, else the active one,
/// else the newest — and for the user's memories. At most `max_items`.
/// nullopt when `conversation_id` names no conversation.
std::optional<nlohmann::json> build_curation_proposals(UserDB& db,
                                                       const ConvId& conversation_id,
                                                       int max_items);

/// One proposal, validated against the store as it is now.
struct CurationApplyPlan {
    std::string proposal_id;
    std::string action;
    std::string target_type;
    std::string target_id;
    ConvId conversation_id;
    ConvId source_conv_id;
    std::string title;
    std::string content;
    float importance = 0.5f;
};

/// Throws std::invalid_argument naming what is wrong with the proposal.
CurationApplyPlan validate_curation_apply_proposal(UserDB& db, const nlohmann::json& proposal);

/// Applies a validated plan and describes the result.
nlohmann::json apply_curation_plan(UserDB& db, const CurationApplyPlan& plan);

} // namespace mm
