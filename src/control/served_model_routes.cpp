// Mantic-Mind — /v1/served-models: the catalog users pick a model from.
//
// Its own translation unit so the served-model surface is readable in one place
// rather than as another few hundred lines of control_api_server.cpp. The
// handlers are ControlApiServer members and share its state.

#include "control/control_api_server.hpp"

#include "common/http_server.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "control/model_registry.hpp"
#include "control/node_registry.hpp"
#include "control/placement_scheduler.hpp"
#include "control/served_model_catalog.hpp"
#include "control/served_model_validator.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>

namespace mm {

namespace {

void json_error(httplib::Response& res, int status, const std::string& message,
                nlohmann::json extra = nlohmann::json::object()) {
    res.status = status;
    extra["error"] = message;
    res.set_content(extra.dump(), "application/json");
}

nlohmann::json entry_json(const CatalogEntry& e) {
    nlohmann::json j = e.model;
    j["source"] = to_string(e.source);
    j["available"] = e.available;
    if (!e.unavailable_reason.empty()) j["unavailable_reason"] = e.unavailable_reason;
    // 0 = none. Always present, for the reason placement_history's
    // released_at_ms is 0 rather than null: a key that is sometimes absent
    // makes every client write a presence check before the question it has.
    j["admission_id"] = e.admission_id;
    if (!e.arch_hash.empty()) j["arch_hash"] = e.arch_hash;
    return j;
}

std::string placement_status(const std::optional<Placement>& placement, const ServedModel& model) {
    if (placement) {
        if (placement->suspended) return "suspended";
        return placement->in_flight > 0 ? "active" : "idle";
    }
    return model.node_local() ? "unplaced" : "api";
}

/// A served model from a request body. Every field optional; the handler sets
/// the id where the route owns it.
bool parse_served_model(const httplib::Request& req, ServedModel& out, std::string& error) {
    try {
        const auto j = nlohmann::json::parse(req.body.empty() ? std::string("{}") : req.body);
        if (!j.is_object()) {
            error = "body must be a JSON object";
            return false;
        }
        out = j.get<ServedModel>();
        return true;
    } catch (const std::exception& e) {
        error = std::string("invalid served model: ") + e.what();
        return false;
    }
}

} // namespace

void ControlApiServer::register_served_model_routes() {
    using httplib::Request;
    using httplib::Response;

    const auto catalog_ready = [this](Response& res) {
        if (served_models_ != nullptr) return true;
        // 503, not an empty list: "no models are served" and "this control has
        // no catalog" are different facts, and only one is something to fix.
        json_error(res, 503, "the served-model catalog is not available");
        return false;
    };

    // Routing and placement, beside the entry. Routing is pure — asking cannot
    // start an engine — which is what makes it safe on a read route.
    const auto describe = [this](const CatalogEntry& e) {
        auto j = entry_json(e);
        const auto routing = scheduler_.resolve_backend_for(e.model);
        j["backend"] = routing.engine_id;
        j["backend_reason"] = routing.reason;
        const auto placement = scheduler_.get_placement(e.model.id);
        if (placement) j["placement"] = *placement;
        j["status"] = placement_status(placement, e.model);
        return j;
    };

    const auto validation_failed = [](Response& res, const ValidationResult& v) {
        res.status = 400;
        nlohmann::json body = {{"error", "validation_failed"}, {"issues", v.issues}};
        if (v.model_info) body["model_info"] = *v.model_info;
        res.set_content(body.dump(), "application/json");
    };

    // ── GET /v1/served-models ────────────────────────────────────────────────
    server_->Get("/v1/served-models", [this, catalog_ready, describe](const Request&,
                                                                     Response& res) {
        if (!catalog_ready(res)) return;
        nlohmann::json data = nlohmann::json::array();
        for (const auto& e : served_models_->list()) data.push_back(describe(e));
        res.set_content(nlohmann::json{{"object", "list"}, {"data", data}}.dump(),
                        "application/json");
    });

    // ── GET /v1/served-models/:id ────────────────────────────────────────────
    server_->Get("/v1/served-models/:id", [this, catalog_ready, describe](const Request& req,
                                                                         Response& res) {
        if (!catalog_ready(res)) return;
        const auto entry = served_models_->find(req.path_params.at("id"));
        if (!entry) {
            json_error(res, 404, "no such served model");
            return;
        }
        auto j = describe(*entry);

        // Where it could go. A fixed probe rather than the model's real
        // footprint: this answers "is any connected node able to take a model
        // at all", and the real ladder runs at placement.
        if (entry->model.node_local()) {
            ResourceFootprint probe;
            probe.vram_mb = 2048;
            int connected = 0;
            for (const auto& n : registry_.list_nodes())
                if (n.connected) ++connected;
            j["node_compatibility"] = {
                {"estimated_vram_mb", probe.vram_mb},
                {"compatible_nodes", static_cast<int>(registry_.nodes_with_capacity(probe).size())},
                {"total_nodes", connected}};
        } else {
            j["node_compatibility"] = {{"backend", "api"},
                                       {"requires_node", false},
                                       {"compatible_nodes", 0},
                                       {"total_nodes", 0}};
        }

        // Where it has run, newest first.
        nlohmann::json history = nlohmann::json::array();
        if (models_ != nullptr) {
            for (const auto& e : models_->placement_history(entry->model.id)) {
                history.push_back(
                    {{"node_id", e.node_id},
                     {"slot_id", e.slot_id},
                     {"backend", e.backend},
                     {"backend_reason", e.backend_reason},
                     {"footprint", {{"vram_mb", e.vram_mb}, {"ram_mb", e.ram_mb}, {"disk_mb", e.disk_mb}}},
                     {"placed_at_ms", e.placed_at_ms},
                     {"released_at_ms", e.released_at_ms},
                     {"open", e.open()}});
            }
        }
        j["placement_history"] = std::move(history);
        res.set_content(j.dump(), "application/json");
    });

    // ── POST /v1/served-models ───────────────────────────────────────────────
    server_->Post("/v1/served-models", [this, catalog_ready, describe, validation_failed](
                                           const Request& req, Response& res) {
        if (!catalog_ready(res)) return;
        ServedModel model;
        std::string error;
        if (!parse_served_model(req, model, error)) {
            json_error(res, 400, error);
            return;
        }
        if (model.id.empty()) {
            // Named after what a person would call it, made unique.
            std::string hint = model.display_name;
            if (util::trim(hint).empty())
                hint = std::filesystem::path(model.model_path).filename().string();
            if (util::trim(hint).empty()) hint = model.model_path;
            model.id = served_models_->unique_id(hint);
            if (model.id.empty()) {
                json_error(res, 400,
                           "could not derive an id; give the model an `id` of [a-z0-9._-]");
                return;
            }
        }
        const auto validation = validate_served_model(model, &registry_, models_dir_, models_);
        if (!validation.ok()) {
            validation_failed(res, validation);
            return;
        }
        switch (served_models_->create(model, error)) {
        case ServedModelCatalog::PutResult::Created:
            break;
        case ServedModelCatalog::PutResult::Conflict:
            json_error(res, 409, error);
            return;
        default:
            json_error(res, 400, error);
            return;
        }
        const auto created = served_models_->find(model.id);
        if (!created) {
            json_error(res, 500, "created, then not found");
            return;
        }
        auto j = describe(*created);
        j["issues"] = validation.issues; // warnings, when there are any
        publish_activity(0, "Served model created: " + model.id);
        res.status = 201;
        res.set_content(j.dump(), "application/json");
    });

    // ── PUT /v1/served-models/:id ────────────────────────────────────────────
    //
    // Replaces the entry. On an ADMITTED entry this writes the operator's
    // override under the same id, so conversations that named it keep working.
    server_->Put("/v1/served-models/:id", [this, catalog_ready, describe, validation_failed](
                                              const Request& req, Response& res) {
        if (!catalog_ready(res)) return;
        const std::string id = req.path_params.at("id");
        ServedModel model;
        std::string error;
        if (!parse_served_model(req, model, error)) {
            json_error(res, 400, error);
            return;
        }
        if (!model.id.empty() && model.id != id) {
            // Ids are immutable: conversations and placement history name them.
            json_error(res, 400, "a served model's id cannot be changed; create a new one");
            return;
        }
        model.id = id;
        const auto validation = validate_served_model(model, &registry_, models_dir_, models_);
        if (!validation.ok()) {
            validation_failed(res, validation);
            return;
        }
        switch (served_models_->update(id, model, error)) {
        case ServedModelCatalog::PutResult::Updated:
            break;
        case ServedModelCatalog::PutResult::NotFound:
            json_error(res, 404, error);
            return;
        default:
            json_error(res, 400, error);
            return;
        }
        // A changed LAUNCH setting is handled at the next acquisition, which
        // replaces the placement once its requests drain. Nothing to do here.
        const auto updated = served_models_->find(id);
        auto j = updated ? describe(*updated) : nlohmann::json::object();
        j["issues"] = validation.issues;
        publish_activity(0, "Served model updated: " + id);
        res.set_content(j.dump(), "application/json");
    });

    // ── DELETE /v1/served-models/:id ─────────────────────────────────────────
    server_->Delete("/v1/served-models/:id", [this, catalog_ready](const Request& req,
                                                                   Response& res) {
        if (!catalog_ready(res)) return;
        const std::string id = req.path_params.at("id");
        std::string error;
        switch (served_models_->remove(id, error)) {
        case ServedModelCatalog::RemoveResult::Removed:
            break;
        case ServedModelCatalog::RemoveResult::NotFound:
            json_error(res, 404, error);
            return;
        case ServedModelCatalog::RemoveResult::NotExplicit:
            json_error(res, 409, error);
            return;
        case ServedModelCatalog::RemoveResult::Failed:
            json_error(res, 500, error);
            return;
        }
        // Not forced: a request still streaming finishes, and housekeeping
        // collects the placement once nothing is using it.
        scheduler_.release(id);
        publish_activity(0, "Served model deleted: " + id);
        res.set_content(nlohmann::json{{"status", "deleted"}, {"served_model_id", id}}.dump(),
                        "application/json");
    });

    // ── PUT /v1/served-models/:id/backend ────────────────────────────────────
    //
    // The operator's engine override. Returns the DECISION, not just the
    // setting: asking for `soma` on a model nothing has admitted is refused,
    // and an operator told only "saved" would reasonably believe it took.
    server_->Put("/v1/served-models/:id/backend", [this, catalog_ready](const Request& req,
                                                                        Response& res) {
        if (!catalog_ready(res)) return;
        const std::string id = req.path_params.at("id");
        std::string value;
        try {
            value = util::to_lower(util::trim(
                nlohmann::json::parse(req.body).value("backend_override", std::string{})));
        } catch (const std::exception& e) {
            json_error(res, 400, e.what());
            return;
        }
        if (value != "auto" && value != "soma" && value != "fallback") {
            json_error(res, 400, "backend_override must be auto, soma or fallback",
                       {{"got", value}});
            return;
        }
        auto entry = served_models_->find(id);
        if (!entry) {
            json_error(res, 404, "no such served model");
            return;
        }
        ServedModel model = entry->model;
        model.backend_override = value;
        std::string error;
        if (served_models_->update(id, model, error) != ServedModelCatalog::PutResult::Updated) {
            json_error(res, 400, error);
            return;
        }
        const auto routing = scheduler_.resolve_backend_for(model);
        res.set_content(nlohmann::json{{"served_model_id", id},
                                       {"backend_override", value},
                                       {"backend", routing.engine_id},
                                       {"backend_reason", routing.reason}}
                            .dump(),
                        "application/json");
    });

    // ── placement lifecycle ──────────────────────────────────────────────────
    server_->Post("/v1/served-models/:id/suspend", [this, catalog_ready](const Request& req,
                                                                         Response& res) {
        if (!catalog_ready(res)) return;
        const std::string id = req.path_params.at("id");
        if (!served_models_->find(id)) {
            json_error(res, 404, "no such served model");
            return;
        }
        switch (scheduler_.suspend(id)) {
        case PlacementScheduler::SuspendResult::Suspended:
            publish_activity(0, "Suspended served model " + id);
            res.set_content(
                nlohmann::json{{"status", "suspended"}, {"served_model_id", id}}.dump(),
                "application/json");
            return;
        case PlacementScheduler::SuspendResult::NotPlaced:
            json_error(res, 409, "no live placement to suspend", {{"served_model_id", id}});
            return;
        case PlacementScheduler::SuspendResult::Busy:
            json_error(res, 409, "the placement is serving a request",
                       {{"served_model_id", id}, {"code", "busy"}});
            return;
        case PlacementScheduler::SuspendResult::Failed:
            json_error(res, 502, "the node refused to suspend the placement",
                       {{"served_model_id", id}});
            return;
        }
    });

    server_->Post("/v1/served-models/:id/restore", [this, catalog_ready](const Request& req,
                                                                         Response& res) {
        if (!catalog_ready(res)) return;
        const std::string id = req.path_params.at("id");
        const auto entry = served_models_->find(id);
        if (!entry) {
            json_error(res, 404, "no such served model");
            return;
        }
        if (!entry->available) {
            json_error(res, 409, entry->unavailable_reason,
                       {{"served_model_id", id}, {"code", "model_unavailable"}});
            return;
        }
        // Restore IS placement: its first step is the existing or suspended
        // placement, so a suspended model resumes from its checkpoint.
        const auto placed = scheduler_.ensure_running(entry->model);
        if (!placed.ok()) {
            json_error(res, 409, placed.error,
                       {{"served_model_id", id},
                        {"code", to_string(placed.failure)},
                        {"retryable", placement_failure_retryable(placed.failure)}});
            return;
        }
        publish_activity(0, "Restored served model " + id + " on node " + placed.node_id);
        res.set_content(nlohmann::json{{"status", "restored"},
                                       {"served_model_id", id},
                                       {"node_id", placed.node_id},
                                       {"slot_id", placed.slot_id},
                                       {"engine_id", placed.engine_id}}
                            .dump(),
                        "application/json");
    });

    server_->Post("/v1/served-models/:id/release", [this, catalog_ready](const Request& req,
                                                                         Response& res) {
        if (!catalog_ready(res)) return;
        const std::string id = req.path_params.at("id");
        if (!served_models_->find(id)) {
            json_error(res, 404, "no such served model");
            return;
        }
        // Idempotent: releasing what holds nothing is not an error. `force`
        // releases even mid-request, which ends that request.
        const bool force = req.get_param_value("force") == "true";
        if (scheduler_.release(id, force) == PlacementScheduler::ReleaseResult::Busy) {
            json_error(res, 409,
                       "the placement is serving a request; retry, or release with ?force=true",
                       {{"served_model_id", id}, {"code", "busy"}});
            return;
        }
        publish_activity(0, "Released served model " + id);
        res.set_content(nlohmann::json{{"status", "released"}, {"served_model_id", id}}.dump(),
                        "application/json");
    });
}

} // namespace mm
