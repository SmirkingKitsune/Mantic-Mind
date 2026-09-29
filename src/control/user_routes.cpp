// Mantic-Mind — /v1/users/…: users, their conversations and memories, their
// chat, attachments and voice. And /v1/audio/speech, which speaks as a user.
//
// A user owns everything under their id; a model is chosen per conversation
// (and per turn) from the served-model catalog. See docs/external-api.md §2c.
//
// Work that touches a conversation's history — a turn, compaction, extraction,
// deletion — runs on that conversation's strand of `conversations_`, so it can
// never interleave with a turn in flight on the same conversation.

#include "control/control_api_server.hpp"

#include "common/engine_capabilities.hpp"
#include "common/http_server.hpp"
#include "common/logger.hpp"
#include "common/sse_infer_ctx.hpp"
#include "common/user_db.hpp"
#include "common/util.hpp"
#include "control/api_support.hpp"
#include "control/conversation_queue.hpp"
#include "control/curation.hpp"
#include "control/model_registry.hpp"
#include "control/placement_scheduler.hpp"
#include "control/served_model_catalog.hpp"
#include "control/user_manager.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>

namespace mm {

using namespace api;

namespace {

nlohmann::json list_json(nlohmann::json data) {
    return nlohmann::json{{"object", "list"}, {"data", std::move(data)}};
}

/// A user profile, with what a client usually wants next to it.
nlohmann::json user_json(User& user) {
    nlohmann::json j = user.profile();
    const auto active = user.db().get_active_conversation_id();
    j["active_conversation_id"] = active ? nlohmann::json(*active) : nlohmann::json(nullptr);
    return j;
}

/// Parses `settings` from a request body into `out`. Range problems are
/// refused, with every bad field named.
bool read_settings(const nlohmann::json& body, SamplingOverrides& out, std::string& error) {
    if (!body.contains("settings")) return true;
    if (!parse_sampling_overrides(body.at("settings"), out, error)) return false;
    const auto problems = sampling_override_problems(out);
    if (problems.empty()) return true;
    error.clear();
    for (const auto& p : problems) error += (error.empty() ? "" : "; ") + p.field + ": " + p.message;
    return false;
}

} // namespace

std::string ControlApiServer::bound_user_of(const httplib::Request& req) const {
    return scopes_.authorize(req).bound_user_id;
}

void ControlApiServer::register_user_routes() {
    using httplib::Request;
    using httplib::Response;

    // The user a route names, or a 404 written for the caller.
    const auto user_or_404 = [this](const Request& req, Response& res) -> std::shared_ptr<User> {
        auto user = users_.get_user(req.path_params.at("id"));
        if (!user) set_json_error_code(res, 404, "user not found", "user_not_found");
        return user;
    };
    // A model id a client names for a conversation or a default must be one
    // the catalog has. Empty is allowed: it defers to the next choice.
    const auto check_model_id = [this](const std::string& id, std::string& error) {
        if (id.empty()) return true;
        if (served_models_ == nullptr) {
            error = "the served-model catalog is not available";
            return false;
        }
        if (!served_models_->find(id)) {
            error = "model not found: " + id + "; GET /v1/served-models lists them";
            return false;
        }
        return true;
    };

    // ── Users ────────────────────────────────────────────────────────────────

    server_->Get("/v1/users", [this](const Request& req, Response& res) {
        // A token bound to a user sees that user and no other.
        const std::string bound = bound_user_of(req);
        nlohmann::json data = nlohmann::json::array();
        for (const auto& profile : users_.list_users()) {
            if (!bound.empty() && profile.id != bound) continue;
            if (auto user = users_.get_user(profile.id)) data.push_back(user_json(*user));
        }
        res.set_content(list_json(std::move(data)).dump(), "application/json");
    });

    server_->Post("/v1/users", [this, check_model_id](const Request& req, Response& res) {
        nlohmann::json body;
        try {
            body = parse_request_json(req);
        } catch (const std::exception& e) {
            return set_json_error(res, 400, "invalid request", e.what());
        }
        UserProfile profile;
        std::string id;
        std::string display_name;
        try {
            id = util::trim(body.value("id", std::string{}));
            display_name = util::trim(body.value("display_name", std::string{}));
        } catch (const std::exception& e) {
            return set_json_error(res, 400, "invalid request", e.what());
        }
        if (id.empty()) id = util::slugify(display_name, 64);
        if (id.empty())
            return set_json_error(res, 400, "id or display_name is required");
        profile.id = id;
        nlohmann::json patch = body;
        patch.erase("id");
        std::string error;
        if (!apply_profile_patch(patch, profile, error)) return set_json_error(res, 400, error);
        if (!check_model_id(profile.default_model, error)) return set_json_error(res, 400, error);
        switch (users_.create_user(profile, error)) {
            case UserManager::CreateResult::Created: break;
            case UserManager::CreateResult::Invalid: return set_json_error(res, 400, error);
            case UserManager::CreateResult::Exists:
                return set_json_error_code(res, 409, error, "user_exists");
        }
        publish_activity(0, "Created user " + profile.id);
        res.status = 201;
        res.set_content(user_json(*users_.get_user(profile.id)).dump(), "application/json");
    });

    server_->Get("/v1/users/:id", [user_or_404](const Request& req, Response& res) {
        if (auto user = user_or_404(req, res)) res.set_content(user_json(*user).dump(), "application/json");
    });

    server_->Put("/v1/users/:id", [this, user_or_404, check_model_id](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        nlohmann::json body;
        try {
            body = parse_request_json(req);
        } catch (const std::exception& e) {
            return set_json_error(res, 400, "invalid request", e.what());
        }
        UserProfile profile = user->profile();
        std::string error;
        if (!apply_profile_patch(body, profile, error)) return set_json_error(res, 400, error);
        if (profile.default_model != user->profile().default_model &&
            !check_model_id(profile.default_model, error))
            return set_json_error(res, 400, error);
        users_.update_profile(user->id(), profile);
        res.set_content(user_json(*user).dump(), "application/json");
    });

    server_->Delete("/v1/users/:id", [this](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        if (!users_.delete_user(id)) return set_json_error_code(res, 404, "user not found", "user_not_found");
        // A later user created under the same id must not inherit this one's
        // credentials, so its bound tokens go with it.
        const int revoked = models_ != nullptr ? models_->revoke_tokens_for_user(id) : 0;
        if (!util::trim(tts_.config().cache_dir).empty()) {
            std::error_code ec;
            std::filesystem::remove_all(user_tts_cache_dir(tts_.config(), data_dir_, id), ec);
        }
        publish_activity(0, "Deleted user " + id + " (" + std::to_string(revoked) + " token(s) revoked)");
        res.set_content(nlohmann::json{{"status", "deleted"}, {"tokens_revoked", revoked}}.dump(),
                        "application/json");
    });

    // ── Conversations ────────────────────────────────────────────────────────

    server_->Get("/v1/users/:id/conversations", [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        nlohmann::json data = nlohmann::json::array();
        for (const auto& c : user->db().list_conversations()) data.push_back(nlohmann::json(c));
        res.set_content(list_json(std::move(data)).dump(), "application/json");
    });

    server_->Post("/v1/users/:id/conversations",
                  [this, user_or_404, check_model_id](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        nlohmann::json body;
        try {
            body = parse_request_json(req);
        } catch (const std::exception& e) {
            return set_json_error(res, 400, "invalid request", e.what());
        }
        NewConversation spec;
        try {
            spec.title = util::trim(body.value("title", std::string{}));
            spec.model_id = util::to_lower(util::trim(body.value("model", std::string{})));
            spec.system_prompt = body.value("system_prompt", std::string{});
            spec.parent_conv_id = body.value("parent_conv_id", std::string{});
            spec.set_active = body.value("set_active", body.value("activate", false));
        } catch (const std::exception& e) {
            return set_json_error(res, 400, "invalid request", e.what());
        }
        std::string error;
        if (!read_settings(body, spec.settings, error)) return set_json_error(res, 400, error);
        if (!check_model_id(spec.model_id, error)) return set_json_error_code(res, 404, error, "model_not_found");
        if (!spec.parent_conv_id.empty() && !user->db().conversation_exists(spec.parent_conv_id))
            return set_json_error(res, 400, "parent conversation not found");

        const auto previous = user->db().get_active_conversation_id();
        const ConvId cid = user->db().create_conversation(spec);
        if (cid.empty()) return set_json_error(res, 400, "failed to create conversation");
        if (spec.set_active && previous && *previous != cid) queue_global_recall(user->id(), *previous);
        res.status = 201;
        res.set_content(nlohmann::json{{"conversation", *user->db().get_conversation(cid)}}.dump(),
                        "application/json");
    });

    server_->Get("/v1/users/:id/conversations/:cid", [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        auto conv = user->db().load_conversation(req.path_params.at("cid"));
        if (!conv) return set_json_error_code(res, 404, "conversation not found", "conversation_not_found");
        res.set_content(nlohmann::json(*conv).dump(), "application/json");
    });

    server_->Put("/v1/users/:id/conversations/:cid",
                 [user_or_404, check_model_id](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        const std::string cid = req.path_params.at("cid");
        if (!user->db().conversation_exists(cid))
            return set_json_error_code(res, 404, "conversation not found", "conversation_not_found");
        nlohmann::json body;
        try {
            body = parse_request_json(req);
        } catch (const std::exception& e) {
            return set_json_error(res, 400, "invalid request", e.what());
        }
        ConversationPatch patch;
        std::string error;
        for (auto it = body.begin(); it != body.end(); ++it) {
            const std::string& key = it.key();
            if (key == "title") {
                if (!it->is_string()) return set_json_error(res, 400, "title must be a string");
                const std::string title = util::trim(it->get<std::string>());
                if (title.empty()) return set_json_error(res, 400, "title cannot be empty");
                patch.title = title;
            } else if (key == "model") {
                if (!it->is_string()) return set_json_error(res, 400, "model must be a string");
                const std::string model = util::to_lower(util::trim(it->get<std::string>()));
                if (!check_model_id(model, error)) return set_json_error_code(res, 404, error, "model_not_found");
                patch.model_id = model; // "" clears it: the user's default applies
            } else if (key == "system_prompt") {
                if (!it->is_string()) return set_json_error(res, 400, "system_prompt must be a string");
                patch.system_prompt = it->get<std::string>();
            } else if (key == "settings") {
                SamplingOverrides settings;
                if (!read_settings(body, settings, error)) return set_json_error(res, 400, error);
                patch.settings = settings;
            } else {
                return set_json_error(res, 400,
                                      key + " is not a conversation field (expected title, model, "
                                            "system_prompt or settings)");
            }
        }
        user->db().update_conversation(cid, patch);
        res.set_content(nlohmann::json(*user->db().get_conversation(cid)).dump(), "application/json");
    });

    server_->Post("/v1/users/:id/conversations/:cid/activate",
                  [this, user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        const std::string cid = req.path_params.at("cid");
        const auto previous = user->db().get_active_conversation_id();
        if (!user->db().set_active_conversation(cid))
            return set_json_error_code(res, 404, "conversation not found", "conversation_not_found");
        if (previous && *previous != cid) queue_global_recall(user->id(), *previous);
        res.set_content(nlohmann::json{{"status", "ok"}, {"active_conversation_id", cid}}.dump(),
                        "application/json");
    });

    server_->Post("/v1/users/:id/conversations/:cid/compact",
                  [this, user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        const std::string cid = req.path_params.at("cid");
        if (!user->db().conversation_exists(cid))
            return set_json_error_code(res, 404, "conversation not found", "conversation_not_found");
        std::string model;
        try {
            model = parse_request_json(req).value("model", std::string{});
        } catch (const std::exception& e) {
            return set_json_error(res, 400, "invalid request", e.what());
        }
        const auto outcome = compact_now(user->id(), cid, model);
        if (outcome.status != 200) return set_json_error_code(res, outcome.status, outcome.error, outcome.code);
        const auto conv = user->db().get_conversation(cid);
        res.set_content(nlohmann::json{{"conversation_id", cid},
                                       {"status", outcome.state},
                                       {"summary_through_seq", conv ? conv->summary_through_seq : -1},
                                       {"summary_model_id", conv ? conv->summary_model_id : std::string{}}}
                            .dump(),
                        "application/json");
    });

    server_->Delete("/v1/users/:id/conversations/:cid", [this, user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        const std::string cid = req.path_params.at("cid");
        if (!user->db().conversation_exists(cid))
            return set_json_error_code(res, 404, "conversation not found", "conversation_not_found");
        if (user->db().is_conversation_active(cid))
            return set_json_error(res, 409,
                                  "cannot delete the active conversation; activate another first");
        // On the conversation's strand: a turn in flight finishes first rather
        // than writing into a conversation that no longer exists.
        bool deleted = false;
        if (!run_on_conversation(user->id(), cid, [&] { deleted = user->db().delete_conversation(cid); }))
            return set_json_error(res, 503, "the server is shutting down");
        if (!deleted) return set_json_error_code(res, 404, "conversation not found", "conversation_not_found");
        res.set_content(R"({"status":"deleted"})", "application/json");
    });

    // ── Local memories (per conversation) ────────────────────────────────────

    server_->Get("/v1/users/:id/conversations/:cid/local-memories",
                 [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        const std::string cid = req.path_params.at("cid");
        if (!user->db().conversation_exists(cid))
            return set_json_error_code(res, 404, "conversation not found", "conversation_not_found");
        nlohmann::json data = nlohmann::json::array();
        for (const auto& m : user->db().list_local_memories(cid)) data.push_back(nlohmann::json(m));
        res.set_content(list_json(std::move(data)).dump(), "application/json");
    });

    server_->Post("/v1/users/:id/conversations/:cid/local-memories",
                  [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        const std::string cid = req.path_params.at("cid");
        if (!user->db().conversation_exists(cid))
            return set_json_error_code(res, 404, "conversation not found", "conversation_not_found");
        try {
            const std::string content = util::trim(parse_request_json(req).value("content", std::string{}));
            if (content.empty()) return set_json_error(res, 400, "content is required");
            LocalMemory mem;
            mem.id = util::generate_uuid();
            mem.conversation_id = cid;
            mem.content = content;
            user->db().add_local_memory(mem);
            const auto saved = user->db().get_local_memory(mem.id);
            res.status = 201;
            res.set_content(nlohmann::json(saved ? *saved : mem).dump(), "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid request", e.what());
        }
    });

    server_->Put("/v1/users/:id/conversations/:cid/local-memories/:mid",
                 [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        const std::string cid = req.path_params.at("cid");
        auto existing = user->db().get_local_memory(req.path_params.at("mid"));
        if (!existing || existing->conversation_id != cid)
            return set_json_error(res, 404, "local memory not found");
        try {
            const std::string content = util::trim(parse_request_json(req).value("content", std::string{}));
            if (content.empty()) return set_json_error(res, 400, "content is required");
            existing->content = content;
            user->db().update_local_memory(*existing);
            res.set_content(nlohmann::json(*user->db().get_local_memory(existing->id)).dump(),
                            "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid request", e.what());
        }
    });

    server_->Delete("/v1/users/:id/conversations/:cid/local-memories/:mid",
                    [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        auto existing = user->db().get_local_memory(req.path_params.at("mid"));
        if (!existing || existing->conversation_id != req.path_params.at("cid"))
            return set_json_error(res, 404, "local memory not found");
        user->db().delete_local_memory(existing->id);
        res.set_content(R"({"status":"deleted"})", "application/json");
    });

    // ── Global memories ──────────────────────────────────────────────────────

    server_->Get("/v1/users/:id/memories", [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        nlohmann::json data = nlohmann::json::array();
        for (const auto& m : user->db().list_memories()) data.push_back(nlohmann::json(m));
        res.set_content(list_json(std::move(data)).dump(), "application/json");
    });

    server_->Post("/v1/users/:id/memories", [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        try {
            const auto j = parse_request_json(req);
            const std::string content = util::trim(j.value("content", std::string{}));
            if (content.empty()) return set_json_error(res, 400, "content is required");
            const std::string source = j.value("source_conv_id", std::string{});
            if (!source.empty() && !user->db().conversation_exists(source))
                return set_json_error(res, 400, "source conversation not found");
            Memory mem;
            mem.id = util::generate_uuid();
            mem.content = content;
            mem.source_conv_id = source;
            mem.importance = std::clamp(j.value("importance", 0.5f), 0.0f, 1.0f);
            user->db().add_memory(mem);
            const auto saved = user->db().get_memory(mem.id);
            res.status = 201;
            res.set_content(nlohmann::json(saved ? *saved : mem).dump(), "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid request", e.what());
        }
    });

    server_->Put("/v1/users/:id/memories/:mid", [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        auto existing = user->db().get_memory(req.path_params.at("mid"));
        if (!existing) return set_json_error(res, 404, "memory not found");
        try {
            const auto j = parse_request_json(req);
            existing->content = util::trim(j.value("content", existing->content));
            existing->importance = std::clamp(j.value("importance", existing->importance), 0.0f, 1.0f);
            if (existing->content.empty()) return set_json_error(res, 400, "content is required");
            user->db().update_memory(*existing);
            res.set_content(nlohmann::json(*user->db().get_memory(existing->id)).dump(), "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid request", e.what());
        }
    });

    server_->Delete("/v1/users/:id/memories/:mid", [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        if (!user->db().get_memory(req.path_params.at("mid")))
            return set_json_error(res, 404, "memory not found");
        user->db().delete_memory(req.path_params.at("mid"));
        res.set_content(R"({"status":"deleted"})", "application/json");
    });

    // Extraction from a range of one conversation's messages, by a model: the
    // request's, else the conversation's, else the user's default. Queued on
    // the conversation's strand; the answer says what was queued.
    server_->Post("/v1/users/:id/memories/extract", [this, user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        ConvId conv_id;
        std::string requested_model;
        int start_index = -1;
        int end_index = -1;
        int context_before = 2;
        try {
            const auto j = parse_request_json(req);
            conv_id = j.value("conversation_id", std::string{});
            requested_model = j.value("model", std::string{});
            start_index = j.value("start_index", -1);
            end_index = j.value("end_index", -1);
            context_before = std::clamp(j.value("context_before", 2), 0, 20);
        } catch (const std::exception& e) {
            return set_json_error(res, 400, "invalid request", e.what());
        }
        if (conv_id.empty()) return set_json_error(res, 400, "conversation_id is required");
        if (start_index < 0 || end_index < start_index)
            return set_json_error(res, 400, "invalid message range");
        const auto conv = user->db().get_conversation(conv_id);
        if (!conv) return set_json_error_code(res, 404, "conversation not found", "conversation_not_found");
        const int count = user->db().message_count(conv_id);
        if (count == 0) return set_json_error(res, 400, "conversation has no messages");
        if (end_index >= count) return set_json_error(res, 400, "message range out of bounds");

        const auto choice = resolve_turn_model(
            requested_model, conv->model_id, user->profile().default_model,
            [this](const std::string& id) -> std::optional<CatalogEntry> {
                if (served_models_ == nullptr) return std::nullopt;
                return served_models_->find(id);
            });
        if (choice.problem != TurnModel::Problem::None) {
            const int status = choice.problem == TurnModel::Problem::Required ? 400
                             : choice.problem == TurnModel::Problem::NotFound ? 404 : 409;
            return set_json_error_code(res, status, choice.error, to_string(choice.problem));
        }
        // Message indices are sequence numbers: dense and 0-based per conversation.
        const int begin_index = std::max(0, start_index - context_before);
        const UserId user_id = user->id();
        const ServedModel model = choice.entry->model;
        queue_extraction(user_id, conv_id, begin_index, end_index, model);
        res.set_content(nlohmann::json{{"status", "queued"},
                                       {"conversation_id", conv_id},
                                       {"model_id", model.id},
                                       {"selected_count", end_index - begin_index + 1}}
                            .dump(),
                        "application/json");
    });

    // ── Curation ─────────────────────────────────────────────────────────────

    server_->Post("/v1/users/:id/curation/proposals", [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        try {
            const auto body = parse_request_json(req);
            auto proposals = build_curation_proposals(
                user->db(), body.value("conversation_id", std::string{}), body.value("max_items", 12));
            if (!proposals)
                return set_json_error_code(res, 404, "conversation not found", "conversation_not_found");
            res.set_content(nlohmann::json{{"proposals", *proposals}}.dump(), "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid request", e.what());
        }
    });

    server_->Post("/v1/users/:id/curation/apply", [this, user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        nlohmann::json body;
        try {
            body = util::trim(req.body).empty() ? nlohmann::json::object() : nlohmann::json::parse(req.body);
        } catch (const std::exception& e) {
            return set_json_error(res, 400, "invalid request", e.what());
        }
        const nlohmann::json* proposals = nullptr;
        if (body.is_array()) proposals = &body;
        else if (body.contains("proposals")) proposals = &body.at("proposals");
        else if (body.contains("approved_proposals")) proposals = &body.at("approved_proposals");
        if (!proposals || !proposals->is_array()) return set_json_error(res, 400, "proposals must be an array");
        if (proposals->empty()) return set_json_error(res, 400, "at least one proposal is required");

        // Every proposal is validated before any is applied: a batch with one
        // bad entry changes nothing.
        std::vector<CurationApplyPlan> plans;
        plans.reserve(proposals->size());
        for (std::size_t i = 0; i < proposals->size(); ++i) {
            const auto& proposal = proposals->at(i);
            try {
                plans.push_back(validate_curation_apply_proposal(user->db(), proposal));
            } catch (const std::exception& e) {
                nlohmann::json rejection = {{"error", "invalid curation proposal"},
                                            {"index", i},
                                            {"reason", e.what()}};
                if (proposal.is_object() && proposal.contains("id") && proposal.at("id").is_string())
                    rejection["proposal_id"] = proposal.at("id").get<std::string>();
                res.status = 400;
                res.set_content(rejection.dump(), "application/json");
                return;
            }
        }
        nlohmann::json results = nlohmann::json::array();
        for (const auto& plan : plans) {
            // A conversation's own history changes on its strand, never under a
            // turn in flight.
            if (plan.action == "delete_conversation") {
                nlohmann::json applied;
                run_on_conversation(user->id(), plan.target_id,
                                    [&] { applied = apply_curation_plan(user->db(), plan); });
                results.push_back(std::move(applied));
            } else {
                results.push_back(apply_curation_plan(user->db(), plan));
            }
        }
        res.set_content(nlohmann::json{{"status", "applied"},
                                       {"applied_count", results.size()},
                                       {"results", results}}
                            .dump(),
                        "application/json");
    });

    // ── Chat (SSE) ───────────────────────────────────────────────────────────

    server_->Post("/v1/users/:id/chat", [this, user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;

        ChatTurnRequest turn;
        turn.user_id = user->id();
        std::vector<std::string> attachment_ids;
        std::string conversation_hint;
        try {
            const auto j = parse_request_json(req);
            turn.message = j.value("message", std::string{});
            if (j.contains("attachment_ids")) {
                if (!j.at("attachment_ids").is_array())
                    throw std::invalid_argument("attachment_ids must be an array");
                attachment_ids = j.at("attachment_ids").get<std::vector<std::string>>();
            }
            conversation_hint = j.value("conversation_id", std::string{});
            turn.model = j.value("model", std::string{});
            turn.max_tokens_override = j.value("max_tokens", 0);
            if (turn.max_tokens_override < 0) throw std::invalid_argument("max_tokens must not be negative");
            std::string error;
            if (!read_settings(j, turn.settings, error)) throw std::invalid_argument(error);
        } catch (const std::exception& e) {
            return set_json_error(res, 400, "invalid chat request", e.what());
        }
        if (turn.message.empty() && attachment_ids.empty())
            return set_json_error(res, 400, "message required unless attachment_ids are supplied");

        if (conversation_hint.empty()) {
            turn.conv_id = user->db().get_or_create_active_conversation();
        } else if (!user->db().conversation_exists(conversation_hint)) {
            return set_json_error_code(res, 404, "conversation not found", "conversation_not_found");
        } else {
            turn.conv_id = conversation_hint;
        }
        try {
            turn.parts = prepare_attachment_turn(user->db(), turn.message, attachment_ids).parts;
        } catch (const std::exception& e) {
            return set_json_error(res, 400, "invalid attachments", e.what());
        }

        // Refused before the stream opens, where a status code can still say
        // so: no model to answer with, or images to an engine that cannot see.
        {
            const auto conv = user->db().get_conversation(turn.conv_id);
            const auto choice = resolve_turn_model(
                turn.model, conv ? conv->model_id : std::string{}, user->profile().default_model,
                [this](const std::string& id) -> std::optional<CatalogEntry> {
                    if (served_models_ == nullptr) return std::nullopt;
                    return served_models_->find(id);
                });
            if (choice.problem != TurnModel::Problem::None) {
                const int status = choice.problem == TurnModel::Problem::Required ? 400
                                 : choice.problem == TurnModel::Problem::NotFound ? 404 : 409;
                return set_json_error_code(res, status, choice.error, to_string(choice.problem));
            }
            if (!attachment_ids.empty()) {
                const ServedModel& model = choice.entry->model;
                std::string engine_id, reason;
                if (model.node_local()) {
                    const auto routing = scheduler_.resolve_backend_for(model);
                    engine_id = routing.engine_id;
                    reason = routing.reason;
                }
                const auto refusal = image_refusal_for(model.vision_settings.enabled, engine_id, reason);
                if (!refusal.empty()) return set_json_error_code(res, 422, refusal, "images_unsupported");
            }
        }

        auto ctx = std::make_shared<SseInferCtx>();
        const auto push = [ctx](std::string payload, bool done) {
            std::lock_guard<std::mutex> lk(ctx->mx);
            ctx->lines.push_back(std::move(payload));
            if (done) ctx->done = true;
            ctx->cv.notify_one();
        };
        auto chunk_fn = [push](const InferenceChunk& chunk) {
            std::string payload;
            if (!chunk.tool_result_json.empty()) {
                payload = chunk.tool_result_json;
            } else if (!chunk.thinking_delta.empty()) {
                payload = nlohmann::json{{"type", "thinking"}, {"content", chunk.thinking_delta}}.dump();
            } else if (chunk.tool_call_delta) {
                const auto& tc = *chunk.tool_call_delta;
                payload = nlohmann::json{{"type", "tool_call"},
                                         {"id", tc.id},
                                         {"name", tc.function_name},
                                         {"arguments", tc.arguments_json}}
                              .dump();
            } else if (!chunk.delta_content.empty()) {
                payload = nlohmann::json{{"type", "delta"}, {"content", chunk.delta_content}}.dump();
            }
            if (!payload.empty()) push("data: " + payload + "\n\n", false);
        };
        auto done_sent = std::make_shared<std::atomic_bool>(false);
        auto done_fn = [push, done_sent](const ConvId& conv_id, const TurnOutcome& outcome) {
            if (done_sent->exchange(true)) return;
            nlohmann::json payload = {{"type", "done"},
                                      {"conv_id", conv_id},
                                      {"success", outcome.ok},
                                      {"model_id", outcome.model_id}};
            if (!outcome.ok) {
                payload["error"] = outcome.error;
                payload["code"] = outcome.code;
            }
            push("data: " + payload.dump() + "\n\n", true);
        };

        const bool queued = conversations_->enqueue(
            queue_key(turn.user_id, turn.conv_id),
            [this, turn, chunk_fn, done_fn]() mutable {
                try {
                    run_chat_turn(turn, chunk_fn, done_fn);
                } catch (const std::exception& e) {
                    MM_ERROR("chat turn for user '{}' failed: {}", turn.user_id, e.what());
                    done_fn(turn.conv_id, TurnOutcome{false, e.what(), "internal_error", turn.model});
                }
            },
            [done_fn, conv_id = turn.conv_id, model = turn.model] {
                done_fn(conv_id, TurnOutcome{false, "the server is shutting down", "shutting_down", model});
            });
        (void)queued;

        res.set_chunked_content_provider("text/event-stream",
                                         [ctx](size_t /*offset*/, httplib::DataSink& sink) -> bool {
            std::unique_lock<std::mutex> lk(ctx->mx);
            ctx->cv.wait(lk, [&] { return !ctx->lines.empty() || ctx->done; });
            while (!ctx->lines.empty()) {
                std::string payload = std::move(ctx->lines.front());
                ctx->lines.pop_front();
                lk.unlock();
                if (!sink.write(payload.data(), payload.size())) return false;
                lk.lock();
            }
            if (ctx->done && ctx->lines.empty()) {
                lk.unlock();
                const std::string fin = "data: [DONE]\n\n";
                sink.write(fin.data(), fin.size());
                sink.done();
            }
            return true;
        });
    });

    // ── Attachments ──────────────────────────────────────────────────────────
    // Uploads stream straight to the user's attachments/; SQLite stores
    // metadata and relative paths only.
    server_->PostUpload("/v1/users/:id/attachments",
        [this](const httplib::Request& req, httplib::Response& res, const HttpServer::UploadPump& pump) {
        auto user = users_.get_user(req.path_params.at("id"));
        if (!user) return set_json_error_code(res, 404, "user not found", "user_not_found");

        const std::string mime = normalized_image_mime(req.get_header_value("Content-Type"));
        if (mime != "image/jpeg" && mime != "image/png")
            return set_json_error(res, 415, "only image/jpeg and image/png uploads are supported");
        if (req.has_header("Content-Length")) {
            try {
                if (std::stoll(req.get_header_value("Content-Length")) > kMaxImageBytes)
                    return set_json_error(res, 413, "image exceeds the 50 MiB limit");
            } catch (...) {
                return set_json_error(res, 400, "invalid Content-Length");
            }
        }

        UserDB& db = user->db();
        ImageAttachment attachment;
        attachment.id = util::generate_uuid();
        attachment.original_filename = safe_attachment_filename(req.get_header_value("X-Filename"), mime);
        attachment.mime_type = mime;
        attachment.relative_path = "attachments/" + attachment.id + (mime == "image/png" ? ".png" : ".jpg");
        attachment.created_at_ms = util::now_ms();
        attachment.expires_at_ms = attachment.created_at_ms + kPendingAttachmentTtlMs;

        const std::filesystem::path destination(db.attachment_file_path(attachment));
        const std::filesystem::path temporary = destination.string() + ".part";
        std::filesystem::create_directories(destination.parent_path());
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) return set_json_error(res, 500, "cannot create managed attachment");
        std::vector<unsigned char> signature;
        signature.reserve(8);
        std::int64_t received = 0;
        bool too_large = false;
        bool write_failed = false;
        const bool pumped = pump([&](const char* data, std::size_t length) {
            if (received + static_cast<std::int64_t>(length) > kMaxImageBytes) {
                too_large = true;
                return false;
            }
            for (std::size_t i = 0; i < length && signature.size() < 8; ++i)
                signature.push_back(static_cast<unsigned char>(data[i]));
            output.write(data, static_cast<std::streamsize>(length));
            if (!output) {
                write_failed = true;
                return false;
            }
            received += static_cast<std::int64_t>(length);
            return true;
        });
        output.close();

        const auto remove_temp = [&] {
            std::error_code ec;
            std::filesystem::remove(temporary, ec);
        };
        if (too_large) {
            remove_temp();
            return set_json_error(res, 413, "image exceeds the 50 MiB limit");
        }
        if (!pumped || write_failed || !output || received == 0) {
            remove_temp();
            return set_json_error(res, 400, "image upload was interrupted or empty");
        }
        if (detected_image_mime(signature) != mime) {
            remove_temp();
            return set_json_error(res, 415, "image signature does not match the declared MIME type");
        }
        attachment.size_bytes = received;
        std::error_code rename_ec;
        std::filesystem::rename(temporary, destination, rename_ec);
        if (rename_ec) {
            remove_temp();
            return set_json_error(res, 500, "failed to commit managed attachment");
        }
        try {
            db.save_attachment(attachment);
        } catch (const std::exception& e) {
            std::error_code remove_ec;
            std::filesystem::remove(destination, remove_ec);
            return set_json_error(res, 500, "failed to persist attachment metadata", e.what());
        }
        res.status = 201;
        res.set_content(nlohmann::json(attachment).dump(), "application/json");
    });

    server_->Get("/v1/users/:id/attachments/:attachment_id", [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        auto attachment = user->db().get_attachment(req.path_params.at("attachment_id"));
        if (!attachment) return set_json_error(res, 404, "attachment not found");
        if (!set_file_response(res, user->db().attachment_file_path(*attachment), attachment->mime_type))
            set_json_error(res, 404, "attachment file not found");
    });

    server_->Delete("/v1/users/:id/attachments/:attachment_id", [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        bool referenced = false;
        if (!user->db().delete_attachment(req.path_params.at("attachment_id"), &referenced)) {
            if (referenced) return set_json_error(res, 409, "attachment is referenced by a message");
            return set_json_error(res, 404, "attachment not found");
        }
        res.status = 204;
    });

    // ── Voice ────────────────────────────────────────────────────────────────

    const auto voice_request_mentions_real_person = [](const VoiceDesignProposal& p) {
        const std::string text = util::to_lower(p.display_name + "\n" + p.voice_description + "\n" + p.rationale);
        return text.find("celebrity") != std::string::npos ||
               text.find("impersonate") != std::string::npos ||
               text.find("real person") != std::string::npos ||
               text.find("sound like ") != std::string::npos ||
               text.find("sounds like ") != std::string::npos;
    };

    const auto synthesize_speech = [this](const UserId& user_id,
                                          TtsSynthesisRequest request,
                                          TtsSynthesisResult* out,
                                          std::string* error,
                                          int* status) -> bool {
        const auto fail = [&](int code, const std::string& why) {
            if (status) *status = code;
            if (error) *error = why;
            return false;
        };
        auto user = users_.get_user(user_id);
        if (!user) return fail(404, "user not found");
        request.user_id = user_id;
        request.text = util::trim(request.text);
        request.format = util::to_lower(util::trim(request.format.empty() ? "wav" : request.format));
        if (request.text.empty()) return fail(400, "text is required");
        if (!is_supported_tts_format(request.format))
            return fail(400, "only wav speech output is supported in this runtime");
        if (!tts_.enabled()) return fail(503, "TTS service is disabled");

        UserDB& db = user->db();
        const std::optional<VoiceProfile> profile = request.voice_profile_id.empty()
                                                        ? db.get_active_voice_profile()
                                                        : db.get_voice_profile(request.voice_profile_id);
        if (!profile) {
            return fail(request.voice_profile_id.empty() ? 409 : 404,
                        request.voice_profile_id.empty() ? "user has no active voice profile"
                                                         : "voice profile not found");
        }
        if (profile->voice_clone_prompt_path.empty() || !path_file_exists(profile->voice_clone_prompt_path))
            return fail(409, "active voice profile is missing its clone-prompt artifact");

        const std::string language = request.language.empty() ? profile->language : request.language;
        const std::string text_hash =
            sha256_hex(profile->id + "\n" + request.format + "\n" + language + "\n" + request.text);
        if (request.use_cache) {
            auto cached = db.find_tts_cache_entry(profile->id, text_hash, request.conversation_id,
                                                  request.message_index);
            if (cached && path_file_exists(cached->audio_path)) {
                if (out) *out = *cached;
                return true;
            }
        }

        const TtsCacheId cache_id = util::generate_uuid();
        const auto cache_dir = user_tts_cache_dir(tts_.config(), data_dir_, user_id);
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);
        if (ec) return fail(500, "failed to create TTS cache directory: " + ec.message());
        const auto audio_path = (cache_dir / (cache_id + ".wav")).string();
        auto svc = tts_.synthesize(request, *profile, audio_path);
        if (!svc.ok) {
            remove_file_quietly(audio_path);
            return fail(svc.status == 0 ? 503 : svc.status,
                        svc.error.empty() ? "TTS synthesis failed" : svc.error);
        }

        TtsSynthesisResult result;
        result.cache_id = cache_id;
        result.user_id = user_id;
        result.voice_profile_id = profile->id;
        result.conversation_id = request.conversation_id;
        result.message_index = request.message_index;
        result.text_hash = text_hash;
        result.audio_path = svc.audio_path.empty() ? audio_path : svc.audio_path;
        result.mime_type = tts_mime_type_for_format(request.format);
        result.format = request.format.empty() ? "wav" : request.format;
        result.sample_rate = svc.sample_rate;
        result.duration_ms = svc.duration_ms;
        result.cached = false;
        result.created_at_ms = util::now_ms();
        result.expires_at_ms = result.created_at_ms + std::max<std::int64_t>(1, tts_.config().cache_ttl_ms);
        db.save_tts_cache_entry(result);
        if (out) *out = result;
        return true;
    };

    server_->Get("/v1/users/:id/voice", [this, user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        UserDB& db = user->db();
        const auto active = db.get_active_voice_profile();
        const nlohmann::json body = {{"active_profile", active ? nlohmann::json(*active) : nlohmann::json(nullptr)},
                                     {"profiles", db.list_voice_profiles()},
                                     {"proposals", db.list_voice_proposals()},
                                     {"tts_enabled", tts_.enabled()},
                                     {"provider", tts_.provider_name()},
                                     {"backend", tts_.backend()}};
        res.set_content(body.dump(), "application/json");
    });

    server_->Get("/v1/users/:id/voice/proposals", [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        res.set_content(nlohmann::json{{"proposals", user->db().list_voice_proposals()}}.dump(),
                        "application/json");
    });

    server_->Post("/v1/users/:id/voice/proposals",
                  [this, user_or_404, voice_request_mentions_real_person](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        try {
            const auto j = parse_request_json(req);
            VoiceDesignProposal proposal;
            const bool explicit_design = !util::trim(j.value("voice_description", std::string{})).empty() ||
                                         !util::trim(j.value("sample_text", std::string{})).empty();
            if (!explicit_design) {
                std::string infer_error;
                auto inferred = infer_voice_design(user->id(), j.value("model", std::string{}), infer_error);
                if (!inferred) return set_json_error(res, 503, "voice proposal inference failed", infer_error);
                proposal = *inferred;
            }
            const UserProfile profile = user->profile();
            proposal.id = util::generate_uuid();
            proposal.user_id = user->id();
            proposal.display_name = util::trim(j.value(
                "display_name",
                proposal.display_name.empty() ? profile.display_name + " Voice" : proposal.display_name));
            proposal.language = util::trim(j.value("language", proposal.language));
            if (proposal.language.empty()) proposal.language = "Auto";
            proposal.voice_description = util::trim(j.value("voice_description", proposal.voice_description));
            proposal.sample_text = util::trim(j.value("sample_text", proposal.sample_text));
            proposal.rationale = util::trim(j.value("rationale", proposal.rationale));
            proposal.status = "pending";
            proposal.provider = tts_.provider_name();
            proposal.voice_design_model_id = j.value("voice_design_model_id", tts_.config().voice_design_model_id);
            proposal.clone_model_id = j.value("clone_model_id", tts_.default_synthesis_model_id());
            proposal.created_at_ms = util::now_ms();
            proposal.updated_at_ms = proposal.created_at_ms;
            if (proposal.voice_description.empty() || proposal.sample_text.empty())
                return set_json_error(res, 400, "voice_description and sample_text are required");
            if (voice_request_mentions_real_person(proposal))
                return set_json_error(res, 400, "voice proposals must describe an original synthetic voice");
            user->db().save_voice_proposal(proposal);
            res.status = 201;
            res.set_content(nlohmann::json{{"proposal", proposal}}.dump(), "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid voice proposal request", e.what());
        }
    });

    server_->Post("/v1/users/:id/voice/proposals/:proposal_id/sample",
                  [this, user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        const std::string proposal_id = req.path_params.at("proposal_id");
        auto proposal = user->db().get_voice_proposal(proposal_id);
        if (!proposal) return set_json_error(res, 404, "voice proposal not found");
        if (!tts_.enabled()) return set_json_error(res, 503, "TTS service is disabled");

        const auto proposal_dir = user_tts_dir(data_dir_, user->id(), "proposals");
        std::error_code ec;
        std::filesystem::create_directories(proposal_dir, ec);
        if (ec) return set_json_error(res, 500, "failed to create TTS proposal directory", ec.message());
        const auto preview_path = (proposal_dir / (proposal_id + ".wav")).string();
        const auto prompt_path = (proposal_dir / (proposal_id + ".prompt.pkl")).string();
        auto svc = tts_.generate_voice_sample(*proposal, preview_path, prompt_path);
        if (!svc.ok) {
            user->db().update_voice_proposal_status(
                proposal_id, "error", svc.error.empty() ? "voice sample generation failed" : svc.error);
            return set_json_error(res, svc.status == 0 ? 503 : svc.status, "voice sample generation failed",
                                  svc.error);
        }
        proposal->preview_audio_path = svc.audio_path.empty() ? preview_path : svc.audio_path;
        proposal->voice_clone_prompt_path =
            svc.voice_clone_prompt_path.empty() ? prompt_path : svc.voice_clone_prompt_path;
        proposal->status = "sampled";
        proposal->error.clear();
        proposal->updated_at_ms = util::now_ms();
        user->db().save_voice_proposal(*proposal);
        res.set_content(nlohmann::json{{"proposal", *proposal},
                                       {"preview_url", "/v1/users/" + user->id() + "/voice/proposals/" +
                                                           proposal_id + "/sample"}}
                            .dump(),
                        "application/json");
    });

    server_->Get("/v1/users/:id/voice/proposals/:proposal_id/sample",
                 [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        auto proposal = user->db().get_voice_proposal(req.path_params.at("proposal_id"));
        if (!proposal || proposal->preview_audio_path.empty() || !path_file_exists(proposal->preview_audio_path))
            return set_json_error(res, 404, "voice proposal sample not found");
        if (!set_file_response(res, proposal->preview_audio_path, "audio/wav"))
            set_json_error(res, 404, "voice proposal sample not found");
    });

    server_->Post("/v1/users/:id/voice/proposals/:proposal_id/approve",
                  [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        const std::string proposal_id = req.path_params.at("proposal_id");
        auto proposal = user->db().get_voice_proposal(proposal_id);
        if (!proposal) return set_json_error(res, 404, "voice proposal not found");
        if (proposal->voice_clone_prompt_path.empty() || !path_file_exists(proposal->voice_clone_prompt_path))
            return set_json_error(res, 400, "proposal must be sampled before approval");

        VoiceProfile profile;
        profile.id = util::generate_uuid();
        profile.user_id = user->id();
        profile.display_name = proposal->display_name;
        profile.language = proposal->language;
        profile.voice_description = proposal->voice_description;
        profile.sample_text = proposal->sample_text;
        profile.rationale = proposal->rationale;
        profile.provider = proposal->provider;
        profile.voice_design_model_id = proposal->voice_design_model_id;
        profile.clone_model_id = proposal->clone_model_id;
        profile.reference_audio_path = proposal->preview_audio_path;
        profile.voice_clone_prompt_path = proposal->voice_clone_prompt_path;
        profile.approved_from_proposal_id = proposal->id;
        profile.active = true;
        profile.created_at_ms = util::now_ms();
        profile.updated_at_ms = profile.created_at_ms;
        user->db().save_voice_profile(profile);
        user->db().update_voice_proposal_status(proposal_id, "approved");
        proposal->status = "approved";
        proposal->error.clear();
        proposal->updated_at_ms = util::now_ms();
        res.set_content(nlohmann::json{{"profile", profile}, {"proposal", *proposal}}.dump(), "application/json");
    });

    server_->Post("/v1/users/:id/voice/proposals/:proposal_id/reject",
                  [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        const std::string proposal_id = req.path_params.at("proposal_id");
        auto proposal = user->db().get_voice_proposal(proposal_id);
        if (!proposal) return set_json_error(res, 404, "voice proposal not found");
        user->db().update_voice_proposal_status(proposal_id, "rejected");
        proposal->status = "rejected";
        proposal->updated_at_ms = util::now_ms();
        res.set_content(nlohmann::json{{"proposal", *proposal}}.dump(), "application/json");
    });

    server_->Post("/v1/users/:id/speech", [synthesize_speech](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        try {
            TtsSynthesisRequest request = parse_request_json(req).get<TtsSynthesisRequest>();
            TtsSynthesisResult result;
            std::string error;
            int status = 500;
            if (!synthesize_speech(id, request, &result, &error, &status))
                return set_json_error(res, status, error.empty() ? "speech synthesis failed" : error);
            res.set_content(nlohmann::json{{"result", result},
                                           {"audio_url", "/v1/users/" + id + "/speech/cache/" + result.cache_id}}
                                .dump(),
                            "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid speech synthesis request", e.what());
        }
    });

    server_->Get("/v1/users/:id/speech/cache/:cache_id", [user_or_404](const Request& req, Response& res) {
        auto user = user_or_404(req, res);
        if (!user) return;
        auto entry = user->db().get_tts_cache_entry(req.path_params.at("cache_id"));
        if (!entry || !path_file_exists(entry->audio_path))
            return set_json_error(res, 404, "speech cache entry not found");
        if (!set_file_response(res, entry->audio_path, entry->mime_type))
            set_json_error(res, 404, "speech audio file not found");
    });

    // OpenAI's speech route, speaking as a user: `voice` is `user:{id}`, and a
    // token bound to a user may speak only as that user.
    server_->Post("/v1/audio/speech", [this, synthesize_speech](const Request& req, Response& res) {
        try {
            const auto j = parse_request_json(req);
            const std::string voice = util::trim(j.value("voice", std::string{}));
            static const std::string prefix = "user:";
            if (voice.rfind(prefix, 0) != 0 || voice.size() <= prefix.size())
                return set_json_error(res, 400, "voice must be of the form user:{user_id}");
            const std::string user_id = voice.substr(prefix.size());
            const std::string bound = bound_user_of(req);
            if (!bound.empty() && bound != user_id)
                return set_json_error_code(res, 403,
                                           "this token is bound to user '" + bound +
                                               "' and cannot speak as '" + user_id + "'",
                                           "user_binding");

            TtsSynthesisRequest request = j.get<TtsSynthesisRequest>();
            request.user_id = user_id;
            if (request.format.empty()) request.format = j.value("response_format", std::string{"wav"});
            TtsSynthesisResult result;
            std::string error;
            int status = 500;
            if (!synthesize_speech(user_id, request, &result, &error, &status))
                return set_json_error(res, status, error.empty() ? "speech synthesis failed" : error);
            if (!set_file_response(res, result.audio_path, result.mime_type))
                set_json_error(res, 404, "speech audio file not found");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid audio speech request", e.what());
        }
    });
}

} // namespace mm
