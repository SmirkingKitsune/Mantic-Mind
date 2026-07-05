#include "node/node_api_server.hpp"
#include "node/node_state.hpp"
#include "node/slot_manager.hpp"
#include "node/ray_orchestration.hpp"
#include "node/hf_cache.hpp"
#include "common/http_server.hpp"
#include "common/llama_cpp_client.hpp"
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
#include <utility>

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

} // namespace

NodeApiServer::NodeApiServer(NodeState& state,
                             SlotManager& slot_mgr,
                             std::string control_url,
                             std::string pairing_key)
    : state_(state)
    , slot_mgr_(slot_mgr)
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

void NodeApiServer::set_runtime_logs_provider(RuntimeLogsProvider provider) {
    runtime_logs_provider_ = std::move(provider);
}

void NodeApiServer::set_remember_api_key_callback(RememberApiKeyCallback callback) {
    remember_api_key_cb_ = std::move(callback);
}

void NodeApiServer::set_vllm_provision_callback(VllmProvisionCallback callback) {
    vllm_provision_cb_ = std::move(callback);
}

void NodeApiServer::set_ray_config(std::string ray_path, uint16_t ray_port) {
    ray_path_ = std::move(ray_path);
    ray_port_ = ray_port;
}

void NodeApiServer::set_hf_config(std::string hf_cli_path,
                                  std::string hf_hub_cache_dir) {
    hf_cli_path_ = std::move(hf_cli_path);
    hf_hub_cache_dir_ = std::move(hf_hub_cache_dir);
}

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

        const auto metrics = state_.get_metrics();
        // Keep the slot manager's VRAM total fresh so vLLM utilization
        // fractions convert to accurate MB estimates.
        slot_mgr_.set_gpu_vram_total_mb(metrics.gpu_vram_total_mb);

        nlohmann::json j;
        j["node_id"]       = state_.get_node_id();
        j["slots"]         = slots;
        j["cached_models"] = scan_hf_cache_models(hf_hub_cache_dir_);
        j["disk_free_mb"]  = metrics.disk_free_mb;
        j["health"]        = metrics;
        j["capabilities"]  = state_.get_capabilities();
        j["max_slots"]     = max_slots;
        j["slot_in_use"]   = in_use_slots;
        j["slot_available"] = available_slots;
        j["slot_ready"]    = ready_slots;
        j["slot_loading"]  = loading_slots;
        j["slot_suspending"] = suspending_slots;
        j["slot_suspended"] = suspended_slots;
        j["slot_error"]    = error_slots;
        j["vllm_server_path"] = slot_mgr_.vllm_server_path();
        j["vllm_runtime"] = state_.get_vllm_runtime();
        j["vllm_gpu_budget"] = slot_mgr_.vllm_gpu_budget();
        j["vllm_gpu_fraction_used"] = slot_mgr_.vllm_gpu_fraction_used();

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

    // ── GET /api/node/logs ───────────────────────────────────────────────────
    server_->Get("/api/node/logs", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }

        int tail = 20;
        if (req.has_param("tail")) {
            try {
                tail = std::stoi(req.get_param_value("tail"));
            } catch (...) {
                res.status = 400;
                res.set_content(R"({"error":"tail must be an integer"})", "application/json");
                return;
            }
        }
        if (tail < 1) tail = 1;
        if (tail > 5000) tail = 5000;

        std::vector<std::string> lines;
        if (runtime_logs_provider_) lines = runtime_logs_provider_(tail);
        res.set_content(nlohmann::json{{"lines", lines}}.dump(), "application/json");
    });

    server_->Get("/api/node/runtime/vllm", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        res.set_content(nlohmann::json{{"vllm_runtime", state_.get_vllm_runtime()}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/runtime/vllm/provision", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (!vllm_provision_cb_) {
            res.status = 501;
            res.set_content(R"({"error":"vLLM provisioning is not configured"})",
                            "application/json");
            return;
        }
        auto runtime = vllm_provision_cb_();
        state_.set_vllm_runtime(runtime);
        if (!runtime.last_error.empty()) state_.set_last_error(runtime.last_error);
        res.status = (runtime.status == "failed" || runtime.status == "disabled") ? 500 : 200;
        res.set_content(nlohmann::json{{"vllm_runtime", runtime}}.dump(),
                        "application/json");
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
            VllmSettings vllm_settings;
            if (j.contains("vllm_settings")) vllm_settings = j["vllm_settings"].get<VllmSettings>();

            if (model_ref.empty()) {
                res.status = 400;
                state_.set_last_error("model_path required");
                res.set_content(R"({"error":"model_path required"})", "application/json");
                return;
            }

            // vLLM resolves the model from an HF id / cache / local dir.
            const std::string model_path = sanitize_path_for_runtime(model_ref);

            auto slot_id = slot_mgr_.load_model(model_path, vllm_settings, agent_id);
            if (slot_id.empty()) {
                res.status = 500;
                nlohmann::json err = {{"error", "failed to load model"}};
                auto detail = slot_mgr_.last_error();
                err["vllm_server_path"] = slot_mgr_.vllm_server_path();
                err["vllm_runtime"] = state_.get_vllm_runtime();
                err["model_path"] = model_path;
                state_.set_last_error(detail.empty() ? "failed to load model" : detail);
                if (!detail.empty()) err["detail"] = detail;
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
                                   {"model_path", model_path}}.dump(),
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
                auto unload = slot_mgr_.unload_all(false);
                if (!unload.ok()) {
                    res.status = unload.status == SlotOperationStatus::Busy ? 409 : 500;
                    res.set_content(nlohmann::json{{"error", unload.message}}.dump(),
                                    "application/json");
                    return;
                }
                state_.set_loaded_model("");
                state_.set_active_agent("");
                state_.set_slots({});
                state_.set_last_error("");
            } else {
                auto unload = slot_mgr_.unload_slot(slot_id);
                state_.set_slots(slot_mgr_.get_slot_info());
                if (!unload.ok()) {
                    res.status = unload.status == SlotOperationStatus::NotFound ? 404
                               : unload.status == SlotOperationStatus::Busy ? 409
                               : 500;
                    res.set_content(nlohmann::json{{"error", unload.message}}.dump(),
                                    "application/json");
                    return;
                }
            }
            res.set_content(R"({"status":"unloaded"})", "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── POST /api/node/ray/start ──────────────────────────────────────────────
    // Join (or head) a Ray cluster for a multi-node vLLM engine group. Body:
    //   {"role":"head"|"worker", "head_address":"ip:port"}  (worker only)
    // Gated to Linux nodes — returns 501 on platforms without Ray support.
    server_->Post("/api/node/ray/start", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (!ray_supported()) {
            res.status = 501;
            res.set_content(
                R"({"error":"Ray multi-node orchestration is only supported on Linux nodes"})",
                "application/json");
            return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string role = j.value("role", std::string{"head"});

            RayStartConfig rcfg;
            rcfg.ray_path = ray_path_;
            rcfg.port     = ray_port_;
            rcfg.num_gpus = state_.get_capabilities().gpu_count;
            if (role == "worker") {
                rcfg.role = RayRole::Worker;
                rcfg.head_address = j.value("head_address", std::string{});
                if (rcfg.head_address.empty()) {
                    res.status = 400;
                    res.set_content(R"({"error":"head_address required for worker role"})",
                                    "application/json");
                    return;
                }
            } else {
                rcfg.role = RayRole::Head;
            }

            std::string err;
            if (!ray_start(rcfg, &err)) {
                res.status = 500;
                res.set_content(nlohmann::json{{"error", err}}.dump(),
                                "application/json");
                return;
            }
            res.set_content(nlohmann::json{{"status", "ray-started"},
                                           {"role", role},
                                           {"port", ray_port_}}.dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── POST /api/node/ray/stop ───────────────────────────────────────────────
    server_->Post("/api/node/ray/stop", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (!ray_supported()) {
            res.status = 501;
            res.set_content(
                R"({"error":"Ray multi-node orchestration is only supported on Linux nodes"})",
                "application/json");
            return;
        }
        std::string err;
        if (!ray_stop(ray_path_, &err)) {
            res.status = 500;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }
        res.set_content(R"({"status":"ray-stopped"})", "application/json");
    });

    // ── POST /api/node/models/pull ────────────────────────────────────────────
    // Pre-fetch an HF model into this node's cache out-of-band, so the multi-GB
    // download does not happen inside the load-model health-timeout window.
    // Gated to Linux (hf_prefetch_supported); Windows nodes let vLLM download
    // lazily at load time instead.
    server_->Post("/api/node/models/pull", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string model_ref = j.value("model_ref",
                                                   j.value("model_path", std::string{}));
            if (model_ref.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"model_ref required"})", "application/json");
                return;
            }
            if (!hf_prefetch_supported()) {
                res.status = 501;
                res.set_content(
                    R"({"error":"HF pre-fetch is only supported on Linux nodes; vLLM downloads at load time"})",
                    "application/json");
                return;
            }
            std::string err;
            if (!hf_download(hf_cli_path_, model_ref, hf_hub_cache_dir_, &err)) {
                res.status = 500;
                res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
                return;
            }
            res.set_content(nlohmann::json{{"status", "pulled"},
                                           {"model_ref", model_ref}}.dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── POST /api/node/detach-agent ───────────────────────────────────────────
    // Removes an agent from a slot. The slot is unloaded once its last agent
    // detaches (deferred while inference is draining).
    server_->Post("/api/node/detach-agent", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string slot_id  = j.value("slot_id",  std::string{});
            std::string agent_id = j.value("agent_id", std::string{});
            if (slot_id.empty() || agent_id.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"slot_id and agent_id required"})",
                                "application/json");
                return;
            }

            auto detach = slot_mgr_.detach_agent(slot_id, agent_id);
            state_.set_slots(slot_mgr_.get_slot_info());
            if (!detach.ok()) {
                res.status = detach.status == SlotOperationStatus::NotFound ? 404 : 500;
                res.set_content(nlohmann::json{{"error", detach.message}}.dump(),
                                "application/json");
                return;
            }

            res.set_content(
                nlohmann::json{{"status", "detached"},
                               {"remaining_agents", detach.remaining_agents},
                               {"unloaded", detach.unloaded}}.dump(),
                "application/json");
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

            auto suspend = slot_mgr_.suspend_slot(slot_id);
            state_.set_slots(slot_mgr_.get_slot_info());
            if (!suspend.ok()) {
                res.status = suspend.status == SlotOperationStatus::NotFound ? 404
                           : suspend.status == SlotOperationStatus::Busy ? 409
                           : 500;
                res.set_content(nlohmann::json{{"error", suspend.message}}.dump(),
                                "application/json");
                return;
            }

            nlohmann::json resp;
            resp["status"] = "suspended";
            resp["kv_cache_path"] = suspend.kv_cache_path;
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
            std::string model_path = sanitize_path_for_runtime(
                j.value("model_path", std::string{}));
            std::string agent_id   = j.value("agent_id", std::string{});
            VllmSettings vllm_settings;
            if (j.contains("vllm_settings")) vllm_settings = j["vllm_settings"].get<VllmSettings>();

            if (model_path.empty()) {
                res.status = 400;
                state_.set_last_error("model_path required");
                res.set_content(R"({"error":"model_path required"})", "application/json");
                return;
            }

            auto slot_id = slot_mgr_.restore_slot(model_path, vllm_settings, agent_id);
            state_.set_slots(slot_mgr_.get_slot_info());

            if (slot_id.empty()) {
                res.status = 500;
                nlohmann::json err = {{"error", "failed to restore slot"}};
                auto detail = slot_mgr_.last_error();
                err["vllm_server_path"] = slot_mgr_.vllm_server_path();
                err["vllm_runtime"] = state_.get_vllm_runtime();
                state_.set_last_error(detail.empty() ? "failed to restore slot" : detail);
                if (!detail.empty()) err["detail"] = detail;
                res.set_content(err.dump(), "application/json");
            } else {
                state_.set_loaded_model(model_path);
                if (!agent_id.empty()) state_.set_active_agent(agent_id);
                state_.set_last_error("");
                res.set_content(
                    nlohmann::json{{"status","restored"},
                                   {"slot_id", slot_id}}.dump(),
                    "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 400;
            state_.set_last_error(e.what());
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── GET /api/node/api-keys ────────────────────────────────────────────────
    server_->Get("/api/node/api-keys", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        auto keys = state_.get_api_keys();
        res.set_content(nlohmann::json{{"keys", keys}}.dump(), "application/json");
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

    // ── GET /api/node/pair-status ─────────────────────────────────────────────
    server_->Get("/api/node/pair-status", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        auto p = state_.get_pending_pair();
        nlohmann::json j;
        j["pending"] = static_cast<bool>(p);
        if (p) {
            j["mode"] = p->pin.empty() ? "psk" : "pin";
            j["expires_ms"] = p->expiry_ms;
            j["challenge"] = p->challenge;
            if (!p->pin.empty()) j["pin"] = p->pin;
        }
        res.set_content(j.dump(), "application/json");
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
            const bool remember = j.value("remember", false);

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
            if (remember && remember_api_key_cb_) remember_api_key_cb_(new_key);
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

        auto slot_lease = selected_slot_id.empty()
            ? SlotManager::SlotLease{}
            : slot_mgr_.acquire_slot(selected_slot_id);

        if (!slot_lease) {
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
        std::thread([this,
                     infer_req,
                     ctx,
                     slot_lease = std::move(slot_lease)]() mutable {
            LlamaCppClient* client = slot_lease.get();
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
                                j = {{"type","tool_call"},{"id", tc.id},
                                     {"name", tc.function_name},
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
                            {"id", tc.id},
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
                    sink.done();
                    return true;
                }
                return true;
            }
        );
    });
}

} // namespace mm
