#include "control/curation.hpp"

#include "common/user_db.hpp"
#include "common/util.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace mm {

namespace {

std::string preview_text(const std::string& text, std::size_t max_len) {
    std::string out = util::trim(text);
    if (out.size() <= max_len) return out;
    return out.substr(0, max_len);
}

std::string first_user_preview(const Conversation& conv, std::size_t max_len) {
    for (const auto& msg : conv.messages) {
        if (msg.role == MessageRole::User && !util::trim(msg.content).empty())
            return preview_text(msg.content, max_len);
    }
    for (const auto& msg : conv.messages) {
        if (!util::trim(msg.content).empty()) return preview_text(msg.content, max_len);
    }
    return {};
}

nlohmann::json curation_proposal(const std::string& action,
                                 const std::string& target_type,
                                 const std::string& target_id,
                                 const ConvId& conversation_id,
                                 nlohmann::json current,
                                 nlohmann::json proposed,
                                 const std::string& rationale) {
    const std::string seed = action + ":" + target_type + ":" + target_id + ":" + conversation_id;
    return nlohmann::json{{"id", util::generate_uuid()},
                          {"action", action},
                          {"target_type", target_type},
                          {"target_id", target_id},
                          {"conversation_id", conversation_id},
                          {"current", std::move(current)},
                          {"proposed", std::move(proposed)},
                          {"rationale", rationale},
                          {"dedupe_key", seed}};
}

std::string json_string_field(const nlohmann::json& j, const char* key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return {};
    if (!it->is_string()) throw std::invalid_argument(std::string(key) + " must be a string");
    return it->get<std::string>();
}

std::string required_json_string_field(const nlohmann::json& j, const char* key) {
    const std::string value = json_string_field(j, key);
    if (value.empty()) throw std::invalid_argument(std::string(key) + " is required");
    return value;
}

float json_float_field(const nlohmann::json& j, const char* key, float fallback) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return fallback;
    if (!it->is_number()) throw std::invalid_argument(std::string(key) + " must be a number");
    return it->get<float>();
}

const nlohmann::json& proposal_payload(const nlohmann::json& proposal, const char* key) {
    auto it = proposal.find(key);
    if (it == proposal.end() || !it->is_object())
        throw std::invalid_argument(std::string(key) + " must be an object");
    return *it;
}

void require_target_type(const CurationApplyPlan& plan, const std::string& expected) {
    if (plan.target_type != expected)
        throw std::invalid_argument(plan.action + " requires target_type '" + expected + "'");
}

} // namespace

std::optional<nlohmann::json> build_curation_proposals(UserDB& db,
                                                       const ConvId& conversation_id,
                                                       int max_items) {
    max_items = std::clamp(max_items, 1, 50);
    nlohmann::json proposals = nlohmann::json::array();
    const auto add = [&](nlohmann::json proposal) {
        if (static_cast<int>(proposals.size()) < max_items) proposals.push_back(std::move(proposal));
    };

    const std::vector<Conversation> convs = db.list_conversations();
    std::optional<Conversation> focus;
    if (!conversation_id.empty()) {
        focus = db.load_conversation(conversation_id);
        if (!focus) return std::nullopt;
    } else {
        for (const auto& item : convs) {
            if (item.is_active) {
                focus = db.load_conversation(item.id);
                break;
            }
        }
        if (!focus && !convs.empty()) focus = db.load_conversation(convs.front().id);
    }

    if (focus) {
        const std::string title = util::trim(focus->title);
        const std::string prompt_preview = first_user_preview(*focus, 56);
        if (!prompt_preview.empty() &&
            (title.empty() || title == "New Conversation" || title == "Untitled conversation")) {
            add(curation_proposal("rename_conversation", "conversation", focus->id, focus->id,
                                  nlohmann::json{{"title", focus->title}},
                                  nlohmann::json{{"title", prompt_preview}},
                                  "The conversation has a generic title; the first user prompt "
                                  "gives it a clearer label."));
        }

        const auto local_mems = db.list_local_memories(focus->id);
        if (local_mems.empty() && !prompt_preview.empty()) {
            add(curation_proposal("create_local_memory", "local_memory", "", focus->id,
                                  nlohmann::json::object(),
                                  nlohmann::json{{"content", prompt_preview}},
                                  "No local memory exists for the selected conversation; this "
                                  "captures the main prompt."));
        }
        for (const auto& mem : local_mems) {
            const std::string trimmed = util::trim(mem.content);
            if (trimmed.empty()) {
                add(curation_proposal("delete_local_memory", "local_memory", mem.id, focus->id,
                                      nlohmann::json(mem), nlohmann::json::object(),
                                      "This local memory is empty."));
            } else if (trimmed != mem.content) {
                add(curation_proposal("update_local_memory", "local_memory", mem.id, focus->id,
                                      nlohmann::json(mem), nlohmann::json{{"content", trimmed}},
                                      "This local memory has leading or trailing whitespace."));
            }
        }
    }

    for (const auto& conv_meta : convs) {
        if (static_cast<int>(proposals.size()) >= max_items) break;
        if (conv_meta.is_active || db.message_count(conv_meta.id) > 0) continue;
        add(curation_proposal("delete_conversation", "conversation", conv_meta.id, conv_meta.id,
                              nlohmann::json(conv_meta), nlohmann::json::object(),
                              "This inactive conversation has no messages."));
    }

    const auto memories = db.list_memories();
    if (memories.empty() && focus) {
        const std::string prompt_preview = first_user_preview(*focus, 96);
        if (!prompt_preview.empty()) {
            add(curation_proposal(
                "create_global_memory", "global_memory", "", focus->id, nlohmann::json::object(),
                nlohmann::json{{"content", "Conversation '" + focus->title + "' covers: " + prompt_preview},
                               {"importance", 0.5}},
                "No global summary exists yet; this creates a traceable summary for future "
                "conversations."));
        }
    }
    for (const auto& mem : memories) {
        if (static_cast<int>(proposals.size()) >= max_items) break;
        const std::string trimmed = util::trim(mem.content);
        if (trimmed.empty()) {
            add(curation_proposal("delete_global_memory", "global_memory", mem.id,
                                  mem.source_conv_id, nlohmann::json(mem), nlohmann::json::object(),
                                  "This global memory is empty."));
        } else if (trimmed != mem.content) {
            add(curation_proposal("update_global_memory", "global_memory", mem.id,
                                  mem.source_conv_id, nlohmann::json(mem),
                                  nlohmann::json{{"content", trimmed}, {"importance", mem.importance}},
                                  "This global memory has leading or trailing whitespace."));
        }
    }
    return proposals;
}

CurationApplyPlan validate_curation_apply_proposal(UserDB& db, const nlohmann::json& proposal) {
    if (!proposal.is_object()) throw std::invalid_argument("proposal must be an object");

    CurationApplyPlan plan;
    plan.proposal_id = required_json_string_field(proposal, "id");
    plan.action = required_json_string_field(proposal, "action");
    plan.target_type = required_json_string_field(proposal, "target_type");
    plan.target_id = json_string_field(proposal, "target_id");
    plan.conversation_id = json_string_field(proposal, "conversation_id");

    const nlohmann::json& proposed = proposal_payload(proposal, "proposed");

    if (plan.action == "rename_conversation") {
        require_target_type(plan, "conversation");
        ConvId conv_id = plan.conversation_id.empty() ? plan.target_id : plan.conversation_id;
        if (conv_id.empty()) throw std::invalid_argument("conversation target is required");
        if (!plan.target_id.empty() && plan.target_id != conv_id)
            throw std::invalid_argument("target_id must match conversation_id for rename_conversation");
        if (!db.conversation_exists(conv_id)) throw std::invalid_argument("conversation target not found");
        plan.target_id = conv_id;
        plan.conversation_id = conv_id;
        plan.title = util::trim(required_json_string_field(proposed, "title"));
        if (plan.title.empty()) throw std::invalid_argument("proposed.title is required");
        return plan;
    }

    if (plan.action == "delete_conversation") {
        require_target_type(plan, "conversation");
        if (plan.target_id.empty()) throw std::invalid_argument("target_id is required");
        if (!plan.conversation_id.empty() && plan.conversation_id != plan.target_id)
            throw std::invalid_argument("conversation_id must match target_id for delete_conversation");
        if (!db.conversation_exists(plan.target_id))
            throw std::invalid_argument("conversation target not found");
        if (db.is_conversation_active(plan.target_id))
            throw std::invalid_argument(
                "cannot delete active conversation; activate another conversation first");
        plan.conversation_id = plan.target_id;
        return plan;
    }

    if (plan.action == "create_local_memory") {
        require_target_type(plan, "local_memory");
        if (!plan.target_id.empty())
            throw std::invalid_argument("create_local_memory target_id must be empty");
        if (plan.conversation_id.empty()) throw std::invalid_argument("conversation_id is required");
        if (!db.conversation_exists(plan.conversation_id))
            throw std::invalid_argument("conversation target not found");
        plan.content = util::trim(required_json_string_field(proposed, "content"));
        if (plan.content.empty()) throw std::invalid_argument("proposed.content is required");
        return plan;
    }

    if (plan.action == "update_local_memory" || plan.action == "delete_local_memory") {
        require_target_type(plan, "local_memory");
        if (plan.target_id.empty()) throw std::invalid_argument("target_id is required");
        if (plan.conversation_id.empty()) throw std::invalid_argument("conversation_id is required");
        if (!db.conversation_exists(plan.conversation_id))
            throw std::invalid_argument("conversation target not found");
        auto existing = db.get_local_memory(plan.target_id);
        if (!existing) throw std::invalid_argument("local memory target not found");
        if (existing->conversation_id != plan.conversation_id)
            throw std::invalid_argument("local memory target does not belong to proposal conversation");
        if (plan.action == "update_local_memory") {
            plan.content = util::trim(required_json_string_field(proposed, "content"));
            if (plan.content.empty()) throw std::invalid_argument("proposed.content is required");
        }
        return plan;
    }

    if (plan.action == "create_global_memory") {
        require_target_type(plan, "global_memory");
        if (!plan.target_id.empty())
            throw std::invalid_argument("create_global_memory target_id must be empty");
        plan.source_conv_id = json_string_field(proposed, "source_conv_id");
        if (plan.source_conv_id.empty()) plan.source_conv_id = plan.conversation_id;
        if (!plan.source_conv_id.empty() && !db.conversation_exists(plan.source_conv_id))
            throw std::invalid_argument("source conversation target not found");
        plan.content = util::trim(required_json_string_field(proposed, "content"));
        if (plan.content.empty()) throw std::invalid_argument("proposed.content is required");
        plan.importance = json_float_field(proposed, "importance", 0.5f);
        return plan;
    }

    if (plan.action == "update_global_memory" || plan.action == "delete_global_memory") {
        require_target_type(plan, "global_memory");
        if (plan.target_id.empty()) throw std::invalid_argument("target_id is required");
        auto existing = db.get_memory(plan.target_id);
        if (!existing) throw std::invalid_argument("global memory target not found");
        if (!plan.conversation_id.empty() && existing->source_conv_id != plan.conversation_id)
            throw std::invalid_argument("global memory target source conversation does not match proposal");
        plan.source_conv_id = existing->source_conv_id;
        if (plan.action == "update_global_memory") {
            plan.content = proposed.contains("content")
                               ? util::trim(json_string_field(proposed, "content"))
                               : existing->content;
            if (plan.content.empty()) throw std::invalid_argument("proposed.content is required");
            plan.importance = json_float_field(proposed, "importance", existing->importance);
        }
        return plan;
    }

    throw std::invalid_argument("unsupported curation proposal action: " + plan.action);
}

nlohmann::json apply_curation_plan(UserDB& db, const CurationApplyPlan& plan) {
    nlohmann::json result = {{"proposal_id", plan.proposal_id},
                             {"action", plan.action},
                             {"target_type", plan.target_type},
                             {"target_id", plan.target_id},
                             {"status", "applied"}};

    if (plan.action == "rename_conversation") {
        ConversationPatch patch;
        patch.title = plan.title;
        db.update_conversation(plan.conversation_id, patch);
        auto saved = db.get_conversation(plan.conversation_id);
        result["target_id"] = plan.conversation_id;
        result["conversation"] = saved ? nlohmann::json(*saved) : nlohmann::json::object();
        return result;
    }
    if (plan.action == "delete_conversation") {
        db.delete_conversation(plan.target_id);
        result["deleted"] = true;
        return result;
    }
    if (plan.action == "create_local_memory") {
        LocalMemory mem;
        mem.id = util::generate_uuid();
        mem.conversation_id = plan.conversation_id;
        mem.content = plan.content;
        db.add_local_memory(mem);
        auto saved = db.get_local_memory(mem.id);
        result["target_id"] = mem.id;
        result["local_memory"] = saved ? nlohmann::json(*saved) : nlohmann::json(mem);
        return result;
    }
    if (plan.action == "update_local_memory") {
        auto existing = db.get_local_memory(plan.target_id);
        LocalMemory mem = existing ? *existing : LocalMemory{};
        mem.id = plan.target_id;
        mem.conversation_id = plan.conversation_id;
        mem.content = plan.content;
        db.update_local_memory(mem);
        auto saved = db.get_local_memory(plan.target_id);
        result["local_memory"] = saved ? nlohmann::json(*saved) : nlohmann::json(mem);
        return result;
    }
    if (plan.action == "delete_local_memory") {
        db.delete_local_memory(plan.target_id);
        result["deleted"] = true;
        return result;
    }
    if (plan.action == "create_global_memory") {
        Memory mem;
        mem.id = util::generate_uuid();
        mem.content = plan.content;
        mem.source_conv_id = plan.source_conv_id;
        mem.importance = plan.importance;
        db.add_memory(mem);
        auto saved = db.get_memory(mem.id);
        result["target_id"] = mem.id;
        result["global_memory"] = saved ? nlohmann::json(*saved) : nlohmann::json(mem);
        return result;
    }
    if (plan.action == "update_global_memory") {
        auto existing = db.get_memory(plan.target_id);
        Memory mem = existing ? *existing : Memory{};
        mem.id = plan.target_id;
        mem.content = plan.content;
        mem.source_conv_id = plan.source_conv_id;
        mem.importance = plan.importance;
        db.update_memory(mem);
        auto saved = db.get_memory(plan.target_id);
        result["global_memory"] = saved ? nlohmann::json(*saved) : nlohmann::json(mem);
        return result;
    }
    if (plan.action == "delete_global_memory") {
        db.delete_memory(plan.target_id);
        result["deleted"] = true;
        return result;
    }
    throw std::logic_error("validated curation proposal action is not implemented");
}

} // namespace mm
