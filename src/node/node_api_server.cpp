#include "node/node_api_server.hpp"
#include "node/llama_runtime_manager.hpp"
#include "node/node_state.hpp"
#include "node/slot_manager.hpp"
#include "node/model_storage.hpp"
#include "node/model_puller.hpp"
#include "common/http_server.hpp"
#include "common/llama_cpp_client.hpp"
#include "common/model_catalog.hpp"
#include "common/models.hpp"
#include "common/logger.hpp"
#include "common/pairing.hpp"
#include "common/util.hpp"
#include "common/sse_infer_ctx.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

namespace mm {
namespace fs = std::filesystem;

namespace {

std::string sanitize_path_for_runtime(std::string p) {
    p = mm::util::trim(p);
    if (p.size() >= 2) {
        if ((p.front() == '"' && p.back() == '"') ||
            (p.front() == '\'' && p.back() == '\'')) {
            p = mm::util::trim(p.substr(1, p.size() - 2));
        }
    }
    p.erase(std::remove_if(p.begin(), p.end(), [](unsigned char ch) {
        return ch == '\r' || ch == '\n' || ch == '\t';
    }), p.end());
    return p;
}

std::string safe_json_dump(const nlohmann::json& j) {
    try {
        return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    } catch (...) {
        return R"({"type":"error","message":"failed to serialize JSON"})";
    }
}

nlohmann::json runtime_status_json(const LlamaRuntimeStatus& s) {
    return nlohmann::json{
        {"running", s.running},
        {"status", s.status},
        {"message", s.message},
        {"started_ms", s.started_ms},
        {"finished_ms", s.finished_ms},
        {"repo_url", s.repo_url},
        {"install_root", s.install_root},
        {"repo_dir", s.repo_dir},
        {"build_dir", s.build_dir},
        {"binary_path", s.binary_path},
        {"installed_commit", s.installed_commit},
        {"remote_commit", s.remote_commit},
        {"remote_error", s.remote_error},
        {"remote_checked_ms", s.remote_checked_ms},
        {"update_available", s.update_available},
        {"update_reason", s.update_reason},
        {"last_job_id", s.last_job_id},
        {"last_log_path", s.last_log_path},
        {"last_log_tail", s.last_log_tail}
    };
}

std::string resolve_load_model_path(const std::string& model_ref,
                                    const ModelStorage& storage) {
    const std::string cleaned = sanitize_path_for_runtime(model_ref);
    if (cleaned.empty()) return {};

    std::error_code ec;
    fs::path direct(cleaned);
    if (fs::exists(direct, ec) && fs::is_regular_file(direct, ec)) {
        return direct.string();
    }

    const std::string canonical = canonical_model_filename(cleaned);
    if (is_safe_model_filename(canonical)) {
        fs::path in_models_dir(storage.model_path(canonical));
        ec.clear();
        if (fs::exists(in_models_dir, ec) && fs::is_regular_file(in_models_dir, ec)) {
            return in_models_dir.string();
        }
    }
    return {};
}

} // namespace

NodeApiServer::NodeApiServer(NodeState& state,
                             SlotManager& slot_mgr,
                             ModelStorage& model_storage,
                             LlamaRuntimeManager& runtime_mgr,
                             std::string control_url,
                             std::string pairing_key)
    : state_(state)
    , slot_mgr_(slot_mgr)
    , model_storage_(model_storage)
    , model_puller_(std::make_unique<ModelPuller>(model_storage_, state_, control_url))
    , runtime_mgr_(runtime_mgr)
    , control_url_(std::move(control_url))
    , pairing_key_(std::move(pairing_key))
    , server_(std::make_unique<HttpServer>())
{}

NodeApiServer::~NodeApiServer() { stop(); }

bool NodeApiServer::listen(uint16_t port) {
    register_routes();
    MM_INFO("NodeApiServer listening on port {}", port);
    return server_->listen("0.0.0.0", port);
}

void NodeApiServer::stop() { server_->stop(); }

// ── Auth check ────────────────────────────────────────────────────────────────
bool NodeApiServer::check_auth(const std::string& auth_header) {
    static const std::string kBearer = "Bearer ";
    if (auth_header.rfind(kBearer, 0) != 0) return false;
    if (!state_.validate_api_key(auth_header.substr(kBearer.size()))) return false;
    state_.mark_control_contact();
    return true;
}

// ── Routes ────────────────────────────────────────────────────────────────────
void NodeApiServer::register_routes() {
    using namespace httplib;

    // ── GET /api/node/health ──────────────────────────────────────────────────
    server_->Get("/api/node/health", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        auto m = state_.get_metrics();
        nlohmann::json j = m;
        j["status"] = "ok";
        res.set_content(j.dump(), "application/json");
    });

    // ── GET /api/node/status ──────────────────────────────────────────────────
    server_->Get("/api/node/status", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        auto slots = slot_mgr_.get_slot_info();
        auto stored = model_storage_.list_models();
        const int64_t disk_free_mb = model_storage_.free_space_mb();
        state_.set_storage(stored, disk_free_mb);
        const int max_slots = static_cast<int>(slot_mgr_.max_slots());

        int ready_slots = 0;
        int loading_slots = 0;
        int suspending_slots = 0;
        int suspended_slots = 0;
        int error_slots = 0;
        for (const auto& s : slots) {
            switch (s.state) {
                case SlotState::Ready:      ++ready_slots; break;
                case SlotState::Loading:    ++loading_slots; break;
                case SlotState::Suspending: ++suspending_slots; break;
                case SlotState::Suspended:  ++suspended_slots; break;
                case SlotState::Error:      ++error_slots; break;
                case SlotState::Empty:
                default:
                    break;
            }
        }
        const int in_use_slots = ready_slots + loading_slots + suspending_slots + error_slots;
        const int available_slots = std::max(0, max_slots - in_use_slots);

        nlohmann::json j;
        j["node_id"]       = state_.get_node_id();
        j["slots"]         = slots;
        j["stored_models"] = stored;
        j["disk_free_mb"]  = disk_free_mb;
        j["health"]        = state_.get_metrics();
        j["max_slots"]     = max_slots;
        j["slot_in_use"]   = in_use_slots;
        j["slot_available"] = available_slots;
        j["slot_ready"]    = ready_slots;
        j["slot_loading"]  = loading_slots;
        j["slot_suspending"] = suspending_slots;
        j["slot_suspended"] = suspended_slots;
        j["slot_error"]    = error_slots;
        j["llama_server_path"] = slot_mgr_.llama_server_path();
        j["llama_server_path_state"] = state_.get_llama_server_path();

        auto upd = state_.get_llama_update_state();
        j["llama_update_running"]     = upd.running;
        j["llama_update_status"]      = upd.status;
        j["llama_update_message"]     = upd.message;
        j["llama_update_started_ms"]  = upd.started_ms;
        j["llama_update_finished_ms"] = upd.finished_ms;

        auto rt = state_.get_llama_runtime_summary();
        j["llama_install_root"]      = rt.install_root;
        j["llama_repo_dir"]          = rt.repo_dir;
        j["llama_build_dir"]         = rt.build_dir;
        j["llama_binary_path"]       = rt.binary_path;
        j["llama_installed_commit"]  = rt.installed_commit;
        j["llama_remote_commit"]     = rt.remote_commit;
        j["llama_remote_error"]      = rt.remote_error;
        j["llama_remote_checked_ms"] = rt.remote_checked_ms;
        j["llama_update_available"]  = rt.update_available;
        j["llama_update_reason"]     = rt.update_reason;
        j["llama_update_log_path"]   = rt.last_log_path;

        // Backwards compat: set single-model fields from first ready slot.
        std::string first_model;
        std::string first_agent;
        for (const auto& s : slots) {
            if (s.state == SlotState::Ready) {
                first_model = s.model_path;
                first_agent = s.assigned_agent;
                break;
            }
        }
        j["loaded_model"]  = first_model;
        j["active_agent"]  = first_agent;

        res.set_content(j.dump(), "application/json");
    });

    // ── POST /api/node/llama/check-update ─────────────────────────────────────
    server_->Post("/api/node/llama/check-update", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }

        std::string message;
        const bool ok = runtime_mgr_.check_update(&message, /*force_remote=*/true);
        auto status = runtime_mgr_.get_status(80);
        auto out = runtime_status_json(status);
        out["ok"] = ok;
        out["message"] = message;
        res.set_content(out.dump(), "application/json");
    });

    // ── POST /api/node/llama/update ───────────────────────────────────────────
    server_->Post("/api/node/llama/update", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }

        bool build = true;
        bool force = false;
        if (!req.body.empty()) {
            try {
                auto j = nlohmann::json::parse(req.body);
                build = j.value("build", true);
                force = j.value("force", false);
            } catch (const std::exception&) {
            }
        }

        std::string job_id;
        std::string message;
        bool accepted = runtime_mgr_.start_update(build, force, &job_id, &message);
        if (!accepted) res.status = 409;

        auto out = runtime_status_json(runtime_mgr_.get_status(80));
        out["accepted"] = accepted;
        out["job_id"] = job_id;
        out["message"] = message;
        res.set_content(out.dump(), "application/json");
    });

    // ── GET /api/node/llama/status ────────────────────────────────────────────
    server_->Get("/api/node/llama/status", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        auto out = runtime_status_json(runtime_mgr_.get_status(120));
        res.set_content(out.dump(), "application/json");
    });

    // ── GET /api/node/llama/jobs/:job_id/log ──────────────────────────────────
    server_->Get(R"(/api/node/llama/jobs/([^/]+)/log)", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }

        std::string job_id;
        if (!req.matches.empty() && req.matches.size() > 1) {
            job_id = req.matches[1].str();
        }
        if (job_id.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"job_id required"})", "application/json");
            return;
        }

        size_t offset = 0;
        size_t limit = 200;
        if (req.has_param("offset")) {
            try { offset = static_cast<size_t>(std::stoull(req.get_param_value("offset"))); } catch (...) {}
        }
        if (req.has_param("limit")) {
            try { limit = static_cast<size_t>(std::stoull(req.get_param_value("limit"))); } catch (...) {}
        }
        if (limit > 1000) limit = 1000;

        auto chunk = runtime_mgr_.read_job_log(job_id, offset, limit);
        if (!chunk.found) {
            res.status = 404;
            res.set_content(R"({"error":"job log not found"})", "application/json");
            return;
        }

        nlohmann::json out{
            {"job_id", chunk.job_id},
            {"log_path", chunk.log_path},
            {"offset", chunk.offset},
            {"next_offset", chunk.next_offset},
            {"lines", chunk.lines}
        };
        res.set_content(out.dump(), "application/json");
    });

    // ── GET /api/node/models ──────────────────────────────────────────────────
    server_->Get("/api/node/models", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        auto models = model_storage_.list_models();
        state_.set_storage(models, model_storage_.free_space_mb());
        res.set_content(nlohmann::json{{"models", models}}.dump(), "application/json");
    });

    // ── GET /api/node/storage ─────────────────────────────────────────────────
    server_->Get("/api/node/storage", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        auto stored = model_storage_.list_models();
        auto disk_free = model_storage_.free_space_mb();
        state_.set_storage(stored, disk_free);
        nlohmann::json j;
        j["disk_free_mb"]  = disk_free;
        j["stored_models"] = stored;
        res.set_content(j.dump(), "application/json");
    });

    // ── POST /api/node/load-model ─────────────────────────────────────────────
    server_->Post("/api/node/load-model", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string model_ref  = j.value("model_path", std::string{});
            std::string agent_id   = j.value("agent_id",   std::string{});
            LlamaSettings settings;
            if (j.contains("settings")) settings = j["settings"].get<LlamaSettings>();

            // Ensure runtime slot manager path follows node state path.
            auto desired_llama = sanitize_path_for_runtime(state_.get_llama_server_path());
            auto current_llama = sanitize_path_for_runtime(slot_mgr_.llama_server_path());
            if (!desired_llama.empty() && desired_llama != current_llama) {
                slot_mgr_.set_llama_server_path(desired_llama);
                current_llama = desired_llama;
                MM_INFO("NodeApiServer: applied llama_server_path override before load-model: {}",
                        desired_llama);
            }

            if (model_ref.empty()) {
                res.status = 400;
                state_.set_last_error("model_path required");
                res.set_content(R"({"error":"model_path required"})", "application/json");
                return;
            }

            const std::string canonical = canonical_model_filename(model_ref);
            nlohmann::json pull_result;
            std::string model_path = resolve_load_model_path(model_ref, model_storage_);
            if (model_path.empty() && is_safe_model_filename(canonical) && model_puller_) {
                model_puller_->pull_model(canonical, /*force=*/false, &pull_result);
                model_path = resolve_load_model_path(canonical, model_storage_);
                state_.set_storage(model_storage_.list_models(), model_storage_.free_space_mb());
            }
            if (model_path.empty()) {
                res.status = 404;
                nlohmann::json err = {
                    {"error", "model file not found on node"},
                    {"model_path", model_ref},
                    {"model_filename", canonical},
                    {"detail", "model missing locally; pull from control failed or unavailable"}
                };
                if (!pull_result.is_null() && !pull_result.empty()) err["transfer"] = pull_result;
                state_.set_last_error("model file not found on node");
                res.set_content(err.dump(), "application/json");
                return;
            }

            auto slot_id = slot_mgr_.load_model(model_path, settings, agent_id);
            if (slot_id.empty()) {
                res.status = 500;
                nlohmann::json err = {{"error", "failed to load model"}};
                auto detail = slot_mgr_.last_error();
                err["llama_server_path"] = slot_mgr_.llama_server_path();
                err["model_path"] = model_path;
                state_.set_last_error(detail.empty() ? "failed to load model" : detail);
                if (!detail.empty()) err["detail"] = detail;
                if (!pull_result.is_null() && !pull_result.empty()) err["transfer"] = pull_result;
                res.set_content(err.dump(), "application/json");
            } else {
                // Update NodeState for UI
                auto slots_info = slot_mgr_.get_slot_info();
                state_.set_slots(slots_info);
                state_.set_loaded_model(model_path);
                if (!agent_id.empty()) state_.set_active_agent(agent_id);
                state_.set_last_error("");

                // Find the effective_ctx_size for the newly loaded slot.
                int effective_ctx = 0;
                for (const auto& si : slots_info) {
                    if (si.id == slot_id) {
                        effective_ctx = si.effective_ctx_size;
                        break;
                    }
                }
                res.set_content(
                    nlohmann::json{{"status","loaded"},
                                   {"slot_id", slot_id},
                                   {"effective_ctx_size", effective_ctx},
                                   {"model_path", model_path},
                                   {"model_filename", canonical},
                                   {"transfer", pull_result}}.dump(),
                    "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 400;
            state_.set_last_error(e.what());
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── POST /api/node/unload-model ───────────────────────────────────────────
    server_->Post("/api/node/unload-model", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string slot_id = j.value("slot_id", std::string{});

            if (slot_id.empty()) {
                // Backwards compat: unload all if no slot_id given
                slot_mgr_.unload_all();
                state_.set_loaded_model("");
                state_.set_active_agent("");
                state_.set_slots({});
                state_.set_last_error("");
            } else {
                slot_mgr_.unload_slot(slot_id);
                state_.set_slots(slot_mgr_.get_slot_info());
            }
            res.set_content(R"({"status":"unloaded"})", "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── POST /api/node/suspend-slot ───────────────────────────────────────────
    server_->Post("/api/node/suspend-slot", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string slot_id = j.value("slot_id", std::string{});
            if (slot_id.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"slot_id required"})", "application/json");
                return;
            }

            auto cache_path = slot_mgr_.suspend_slot(slot_id);
            state_.set_slots(slot_mgr_.get_slot_info());

            nlohmann::json resp;
            resp["status"] = "suspended";
            resp["kv_cache_path"] = cache_path;
            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── POST /api/node/restore-slot ───────────────────────────────────────────
    server_->Post("/api/node/restore-slot", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string model_path    = j.value("model_path",    std::string{});
            std::string kv_cache_path = j.value("kv_cache_path", std::string{});
            std::string agent_id      = j.value("agent_id",      std::string{});
            LlamaSettings settings;
            if (j.contains("settings")) settings = j["settings"].get<LlamaSettings>();

            if (model_path.empty()) {
                res.status = 400;
                state_.set_last_error("model_path required");
                res.set_content(R"({"error":"model_path required"})", "application/json");
                return;
            }

            auto slot_id = slot_mgr_.restore_slot(model_path, settings, kv_cache_path, agent_id);
            state_.set_slots(slot_mgr_.get_slot_info());

            if (slot_id.empty()) {
                res.status = 500;
                nlohmann::json err = {{"error", "failed to restore slot"}};
                auto detail = slot_mgr_.last_error();
                state_.set_last_error(detail.empty() ? "failed to restore slot" : detail);
                if (!detail.empty()) err["detail"] = detail;
                res.set_content(err.dump(), "application/json");
            } else {
                state_.set_loaded_model(model_path);
                if (!agent_id.empty()) state_.set_active_agent(agent_id);
                state_.set_last_error("");
                res.set_content(
                    nlohmann::json{{"status","restored"},{"slot_id", slot_id}}.dump(),
                    "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 400;
            state_.set_last_error(e.what());
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── POST /api/node/upload-model ───────────────────────────────────────────
    server_->Post("/api/node/upload-model", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }

        std::string filename    = req.get_header_value("X-Model-Filename");
        std::string sha256      = req.get_header_value("X-Model-SHA256");

        if (filename.empty() || !is_safe_model_filename(canonical_model_filename(filename))) {
            res.status = 400;
            res.set_content(R"({"error":"valid X-Model-Filename header required"})", "application/json");
            return;
        }
        filename = canonical_model_filename(filename);

        // Write the raw body as a chunk at offset 0 (or the specified offset)
        std::string offset_s = req.get_header_value("X-Chunk-Offset");
        int64_t offset = 0;
        if (!offset_s.empty()) {
            try { offset = std::stoll(offset_s); } catch (...) {}
        }

        bool ok = model_storage_.write_chunk(filename,
                                             req.body.data(),
                                             req.body.size(),
                                             offset);
        if (!ok) {
            res.status = 500;
            res.set_content(R"({"error":"failed to write chunk"})", "application/json");
            return;
        }

        // If this is the last chunk (or a single-chunk upload), verify hash.
        bool verified = false;
        bool is_last_chunk = req.get_header_value("X-Last-Chunk") == "true";
        if (is_last_chunk && !sha256.empty()) {
            verified = model_storage_.verify_hash(filename, sha256);
        }

        nlohmann::json resp;
        resp["status"] = "written";
        resp["verified"] = verified;
        state_.set_storage(model_storage_.list_models(), model_storage_.free_space_mb());
        res.set_content(resp.dump(), "application/json");
    });

    // ── POST /api/node/models/pull ────────────────────────────────────────────
    server_->Post("/api/node/models/pull", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string model_filename = canonical_model_filename(
                j.value("model_filename", std::string{}));
            bool force = j.value("force", false);
            if (!is_safe_model_filename(model_filename)) {
                res.status = 400;
                res.set_content(R"({"error":"valid model_filename required"})", "application/json");
                return;
            }

            nlohmann::json pull_result;
            const bool ok = model_puller_ && model_puller_->pull_model(model_filename, force, &pull_result);
            state_.set_storage(model_storage_.list_models(), model_storage_.free_space_mb());
            if (!ok) {
                res.status = 502;
                if (pull_result.is_null()) {
                    pull_result = nlohmann::json{
                        {"status", "failed"},
                        {"error", "model pull failed"}
                    };
                }
                res.set_content(pull_result.dump(), "application/json");
                return;
            }

            if (!pull_result.contains("status")) pull_result["status"] = "downloaded";
            res.set_content(pull_result.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── DELETE /api/node/models/:filename ─────────────────────────────────────
    server_->Delete("/api/node/models/:filename", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        std::string filename = canonical_model_filename(req.path_params.at("filename"));
        if (!is_safe_model_filename(filename)) {
            res.status = 400;
            res.set_content(R"({"error":"invalid filename"})", "application/json");
            return;
        }
        if (model_storage_.delete_model(filename)) {
            state_.set_storage(model_storage_.list_models(), model_storage_.free_space_mb());
            res.set_content(R"({"status":"deleted"})", "application/json");
        } else {
            res.status = 404;
            res.set_content(R"({"error":"model not found"})", "application/json");
        }
    });

    // ── POST /api/node/api-keys ───────────────────────────────────────────────
    server_->Post("/api/node/api-keys", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string key = j.value("key", std::string{});
            if (key.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"key required"})", "application/json");
                return;
            }
            state_.add_api_key(key);
            res.set_content(R"({"status":"added"})", "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── DELETE /api/node/api-keys/{key} ───────────────────────────────────────
    server_->Delete("/api/node/api-keys/:key", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        std::string key = req.path_params.at("key");
        state_.remove_api_key(key);
        res.set_content(R"({"status":"removed"})", "application/json");
    });

    // ── POST /api/node/pair-request  (unauthenticated) ───────────────────────
    server_->Post("/api/node/pair-request", [this](const Request& req, Response& res) {
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string challenge = j.value("challenge", std::string{});
            if (challenge.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"challenge required"})", "application/json");
                return;
            }

            PendingPair pp;
            pp.challenge  = challenge;
            pp.expiry_ms  = mm::util::now_ms() + 60000; // 60 s

            std::string mode;
            if (!pairing_key_.empty()) {
                pp.expected_response = mm::pairing::hmac_sha256_hex(pairing_key_, challenge);
                mode = "psk";
            } else {
                pp.pin               = mm::pairing::generate_pin();
                pp.expected_response = mm::pairing::hmac_sha256_hex(pp.pin, challenge);
                mode = "pin";
            }

            state_.set_pending_pair(pp);
            MM_INFO("NodeApiServer: pair-request accepted (mode={})", mode);

            res.set_content(
                nlohmann::json{{"accepted", true}, {"mode", mode}}.dump(),
                "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── POST /api/node/pair-complete  (unauthenticated) ──────────────────────
    server_->Post("/api/node/pair-complete", [this](const Request& req, Response& res) {
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string challenge = j.value("challenge", std::string{});
            std::string response  = j.value("response",  std::string{});

            auto pp = state_.get_pending_pair();
            if (!pp || pp->challenge != challenge || pp->expected_response != response) {
                res.status = 200;
                res.set_content(
                    R"({"accepted":false,"error":"invalid response"})",
                    "application/json");
                MM_WARN("NodeApiServer: pair-complete rejected");
                return;
            }

            std::string new_key = mm::util::generate_api_key();
            state_.add_api_key(new_key);
            state_.clear_pending_pair();
            state_.mark_control_contact();
            // Pairing means control has authenticated this node; reflect that in the TUI.
            state_.set_registered(true);
            MM_INFO("NodeApiServer: pair-complete accepted — new key issued");

            res.set_content(
                nlohmann::json{{"accepted", true}, {"api_key", new_key}}.dump(),
                "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── POST /api/node/infer  (SSE streaming) ─────────────────────────────────
    server_->Post("/api/node/infer", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }

        // Parse the InferenceRequest + slot_id
        InferenceRequest infer_req;
        std::string slot_id;
        try {
            auto j = nlohmann::json::parse(req.body);
            infer_req = j.get<InferenceRequest>();
            slot_id = j.value("slot_id", std::string{});
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        auto slots = slot_mgr_.get_slot_info();

        // Resolve selected slot.
        std::string selected_slot_id = slot_id;
        if (selected_slot_id.empty()) {
            // Backwards compat: use first ready slot.
            for (const auto& s : slots) {
                if (s.state == SlotState::Ready) {
                    selected_slot_id = s.id;
                    break;
                }
            }
        }

        // Find the right LlamaCppClient.
        LlamaCppClient* client = nullptr;
        if (!selected_slot_id.empty()) {
            client = slot_mgr_.get_client(selected_slot_id);
        }

        if (!client) {
            res.status = 503;
            res.set_content(R"({"error":"no ready slot available"})", "application/json");
            return;
        }

        // Shared context between LLM worker and SSE provider
        auto ctx = std::make_shared<SseInferCtx>();

        // Look up agent for this slot and start streaming text tracking.
        std::string agent_for_slot;
        std::string model_for_slot;
        for (const auto& s : slots) {
            if (s.id == selected_slot_id) {
                agent_for_slot = s.assigned_agent;
                model_for_slot = s.model_path;
                break;
            }
        }
        slot_mgr_.touch_slot(selected_slot_id);
        state_.set_slots(slot_mgr_.get_slot_info());
        if (!model_for_slot.empty()) state_.set_loaded_model(model_for_slot);
        state_.start_streaming_text(selected_slot_id, agent_for_slot);
        if (!agent_for_slot.empty()) state_.set_active_agent(agent_for_slot);

        // Fire the LLM call on a background thread
        std::thread([this, client, infer_req, ctx]() {
            auto emit_line = [ctx](const std::string& payload, bool done) {
                std::lock_guard<std::mutex> lk(ctx->mx);
                ctx->lines.push_back(payload);
                if (done) ctx->done = true;
                ctx->cv.notify_one();
            };

            try {
                if (infer_req.stream) {
                    client->stream_complete(infer_req,
                        [this, emit_line](const InferenceChunk& c) {
                            nlohmann::json j;
                            if (!c.thinking_delta.empty()) {
                                j = {{"type","thinking"},{"content", c.thinking_delta}};
                                state_.append_streaming_text("", c.thinking_delta);
                            } else if (!c.delta_content.empty()) {
                                j = {{"type","delta"},{"content", c.delta_content}};
                                state_.append_streaming_text(c.delta_content, "");
                            } else if (c.tool_call_delta) {
                                auto& tc = *c.tool_call_delta;
                                j = {{"type","tool_call"},{"name", tc.function_name},
                                     {"arguments", tc.arguments_json}};
                            } else if (c.is_done) {
                                j = {{"type","done"},
                                     {"tokens_used", c.tokens_used},
                                     {"finish_reason", c.finish_reason}};
                                state_.finish_streaming_text(c.finish_reason, c.tokens_used);
                            }
                            if (!j.is_null()) {
                                emit_line("data: " + safe_json_dump(j) + "\n\n", c.is_done);
                            } else if (c.is_done) {
                                emit_line("data: " + safe_json_dump(nlohmann::json{
                                    {"type", "done"},
                                    {"tokens_used", c.tokens_used},
                                    {"finish_reason", c.finish_reason}
                                }) + "\n\n", true);
                                state_.finish_streaming_text(c.finish_reason, c.tokens_used);
                            }
                        },
                        [this, emit_line](const std::string& err) {
                            state_.finish_streaming_text("error", 0);
                            emit_line("data: " + safe_json_dump(nlohmann::json{
                                {"type","error"},
                                {"message", err}
                            }) + "\n\n", true);
                        }
                    );
                } else {
                    Message full = client->complete(infer_req);
                    if (full.role != MessageRole::Assistant) {
                        state_.finish_streaming_text("error", 0);
                        emit_line("data: " + safe_json_dump(nlohmann::json{
                            {"type", "error"},
                            {"message", "non-stream inference failed"}
                        }) + "\n\n", true);
                        return;
                    }
                    if (!full.thinking_text.empty()) {
                        state_.append_streaming_text("", full.thinking_text);
                        emit_line("data: " + safe_json_dump(nlohmann::json{
                            {"type", "thinking"},
                            {"content", full.thinking_text}
                        }) + "\n\n", false);
                    }
                    if (!full.content.empty()) {
                        state_.append_streaming_text(full.content, "");
                        emit_line("data: " + safe_json_dump(nlohmann::json{
                            {"type", "delta"},
                            {"content", full.content}
                        }) + "\n\n", false);
                    }
                    for (const auto& tc : full.tool_calls) {
                        emit_line("data: " + safe_json_dump(nlohmann::json{
                            {"type", "tool_call"},
                            {"name", tc.function_name},
                            {"arguments", tc.arguments_json}
                        }) + "\n\n", false);
                    }
                    const std::string finish_reason = full.content.empty() && full.tool_calls.empty()
                        ? "empty"
                        : "stop";
                    state_.finish_streaming_text(finish_reason, full.token_count);
                    emit_line("data: " + safe_json_dump(nlohmann::json{
                        {"type", "done"},
                        {"tokens_used", full.token_count},
                        {"finish_reason", finish_reason}
                    }) + "\n\n", true);
                }
            } catch (const std::exception& e) {
                MM_ERROR("NodeApiServer infer worker exception: {}", e.what());
                state_.set_last_error(std::string("infer worker exception: ") + e.what());
                state_.finish_streaming_text("error", 0);
                emit_line("data: " + safe_json_dump(nlohmann::json{
                    {"type","error"},
                    {"message", std::string("infer worker exception: ") + e.what()}
                }) + "\n\n", true);
            } catch (...) {
                MM_ERROR("NodeApiServer infer worker exception: unknown");
                state_.set_last_error("infer worker exception: unknown");
                state_.finish_streaming_text("error", 0);
                emit_line("data: " + safe_json_dump(nlohmann::json{
                    {"type","error"},
                    {"message", "infer worker exception: unknown"}
                }) + "\n\n", true);
            }

            std::lock_guard<std::mutex> lk(ctx->mx);
            if (!ctx->done) {
                ctx->done = true;
                ctx->cv.notify_one();
            }
        }).detach();

        // Stream SSE to client via chunked content provider
        res.set_chunked_content_provider("text/event-stream",
            [ctx](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::unique_lock<std::mutex> lk(ctx->mx);
                ctx->cv.wait(lk, [&]{ return !ctx->lines.empty() || ctx->done; });

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
                    return false;
                }
                return true;
            }
        );
    });
}

} // namespace mm
