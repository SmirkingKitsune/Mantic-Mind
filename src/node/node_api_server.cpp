#include "node/node_api_server.hpp"
#include "node/node_state.hpp"
#include "node/slot_manager.hpp"
#include "node/ray_orchestration.hpp"
#include "node/hf_cache.hpp"
#include "node/model_store.hpp"
#include "common/http_server.hpp"
#include "common/runtime_client.hpp"
#include "common/models.hpp"
#include "common/logger.hpp"
#include "common/pairing.hpp"
#include "common/util.hpp"
#include "common/sse_infer_ctx.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <utility>

namespace mm {
namespace fs = std::filesystem;

namespace {

// True when a model reference is a filesystem path (contains a separator or a
// Windows drive prefix) rather than a bare model name or HF repo id.
bool looks_like_fs_path(const std::string& s) {
    if (s.find('/') != std::string::npos || s.find('\\') != std::string::npos) return true;
    if (s.size() >= 2 &&
        std::isalpha(static_cast<unsigned char>(s[0])) && s[1] == ':') return true;
    return false;
}

// A vLLM runtime is usable for launching engines once it resolves to an
// executable (found on PATH or a completed managed install).
bool node_vllm_runtime_ready(const VllmRuntimeStatus& rt) {
    return (rt.status == "resolved" || rt.status == "ready") && !rt.executable_path.empty();
}

// Load paths currently backing a live slot — models the store must never
// evict out from under a running engine.
std::set<std::string> loaded_model_paths(SlotManager& mgr) {
    std::set<std::string> paths;
    for (const auto& s : mgr.get_slot_info())
        if (!s.model_path.empty()) paths.insert(s.model_path);
    return paths;
}

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
    // Streamed model uploads far exceed cpp-httplib's 100 MB default body cap;
    // raise it so /api/node/models/receive can accept multi-GB model files.
    server_->set_payload_max_length(std::size_t{1} << 42);  // 4 TiB
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

void NodeApiServer::set_vllm_update_callback(VllmUpdateCallback callback) {
    vllm_update_cb_ = std::move(callback);
}

void NodeApiServer::set_vllm_check_update_callback(VllmCheckUpdateCallback callback) {
    vllm_check_update_cb_ = std::move(callback);
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

void NodeApiServer::set_model_store(ModelStore* store) {
    model_store_ = store;
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
        if (model_store_) {
            nlohmann::json managed = nlohmann::json::array();
            for (const auto& m : model_store_->list())
                managed.push_back({{"id", m.id},
                                   {"size_bytes", m.size_bytes},
                                   {"pinned", m.pinned},
                                   {"last_used_ms", m.last_used_ms}});
            j["managed_models"] = managed;
            j["model_cache_free_bytes"] = model_store_->free_bytes();
        }
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
        j["vllm_install_progress"] = state_.get_vllm_install_progress();
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
        // Optional {"update": true} approves a managed upgrade to the target
        // version; the default (no body / update=false) ensures a runtime exists.
        bool want_update = false;
        if (!req.body.empty()) {
            try {
                const auto j = nlohmann::json::parse(req.body);
                want_update = j.value("update", false);
            } catch (...) {
                res.status = 400;
                res.set_content(R"({"error":"invalid JSON body"})", "application/json");
                return;
            }
        }
        auto* cb = want_update ? &vllm_update_cb_ : &vllm_provision_cb_;
        if (!*cb) {
            res.status = 501;
            res.set_content(want_update
                                ? R"({"error":"vLLM update is not configured"})"
                                : R"({"error":"vLLM provisioning is not configured"})",
                            "application/json");
            return;
        }
        auto runtime = (*cb)();
        state_.set_vllm_runtime(runtime);
        if (!runtime.last_error.empty()) state_.set_last_error(runtime.last_error);
        res.status = (runtime.status == "failed" || runtime.status == "disabled") ? 500 : 200;
        res.set_content(nlohmann::json{{"vllm_runtime", runtime}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/runtime/vllm/check-update", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (!vllm_check_update_cb_) {
            res.status = 501;
            res.set_content(R"({"error":"vLLM update checking is not configured"})",
                            "application/json");
            return;
        }
        auto runtime = vllm_check_update_cb_();
        state_.set_vllm_runtime(runtime);
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
            // Optional: identity + pin flag for a control-transferred model, so
            // the node can refresh the LRU use-queue and pinned state on load.
            const std::string model_id = j.value("model_id", std::string{});
            const bool model_pin       = j.value("pin", false);
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

            // Fail fast rather than spawn an engine we know will die: without a
            // usable runtime the launch reports a misleading "did not become
            // healthy within 600s" instead of the real cause.
            if (const auto rt = state_.get_vllm_runtime(); !node_vllm_runtime_ready(rt)) {
                res.status = 503;
                const std::string msg = rt.last_error.empty()
                    ? "vLLM runtime is not ready on this node (status=" + rt.status + ")"
                    : "vLLM runtime is not ready on this node: " + rt.last_error;
                state_.set_last_error(msg);
                res.set_content(nlohmann::json{{"error", "vllm runtime not ready"},
                                               {"detail", msg},
                                               {"vllm_runtime", rt}}.dump(),
                                "application/json");
                return;
            }

            // A path-like ref that is not an HF repo id must exist on this node.
            // This catches a control-side local/UNC/Windows path (e.g. N:\...)
            // handed to a node that cannot resolve it, with a clear error
            // instead of a 10-minute vLLM startup timeout.
            if (!mm::util::is_hf_repo_id(model_path) && looks_like_fs_path(model_path)) {
                std::error_code ec;
                if (!fs::exists(model_path, ec)) {
                    res.status = 400;
                    const std::string msg =
                        "model path not found on this node: " + model_path +
                        " (control must transfer the model here, or use an HF repo "
                        "id or a node-local path)";
                    state_.set_last_error(msg);
                    res.set_content(nlohmann::json{{"error", "model not found on node"},
                                                   {"detail", msg},
                                                   {"model_path", model_path}}.dump(),
                                    "application/json");
                    return;
                }
            }

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

                // Refresh the local cache's use-queue and pin state for a
                // transferred model. Pin is STICKY: a load from a preferred
                // agent pins the model, but a later load from a non-preferred
                // agent (pin=false) must not clear a pin another agent set —
                // otherwise the model would become evictable despite a standing
                // preference. Unpinning is a separate, explicit lifecycle event.
                if (model_store_ && !model_id.empty()) {
                    if (model_pin) model_store_->set_pinned(model_id, true);
                    model_store_->touch(model_id);
                }

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

    // ── POST /api/node/models/receive ─────────────────────────────────────────
    // Accept one file of a control-transferred model, streamed as the raw
    // request body (no in-memory buffering). Metadata rides in headers:
    //   X-MM-Model-Id  stable model id (identical on control and node)
    //   X-MM-Rel-Path  this file's path within the model
    //   X-MM-Size      this file's byte length (used to make room); optional
    //   X-MM-Pin       "true" to pin the model on this node (survives shutdown)
    // Before writing, LRU-evicts unpinned models if disk is tight. Returns
    // {status, model_id, load_path, evicted[]} once the file is in place.
    server_->PostUpload("/api/node/models/receive",
        [this](const httplib::Request& req, httplib::Response& res,
               const HttpServer::UploadPump& pump) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401;
            res.set_content(R"({"error":"unauthorized"})", "application/json");
            return;
        }
        if (!model_store_) {
            res.status = 501;
            res.set_content(R"({"error":"model store is not configured on this node"})",
                            "application/json");
            return;
        }
        const std::string id  = util::trim(req.get_header_value("X-MM-Model-Id"));
        const std::string rel = util::trim(req.get_header_value("X-MM-Rel-Path"));
        const bool pin =
            util::to_lower(util::trim(req.get_header_value("X-MM-Pin"))) == "true";
        int64_t size = 0;
        if (req.has_header("X-MM-Size")) {
            try { size = std::stoll(req.get_header_value("X-MM-Size")); } catch (...) {}
        }
        if (id.empty() || rel.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"X-MM-Model-Id and X-MM-Rel-Path required"})",
                            "application/json");
            return;
        }

        // Reserve this model for the whole (possibly multi-file, multi-request)
        // transfer so neither make_room_for here nor the background disk-pressure
        // thread can evict files we have already received for it. The reservation
        // is refreshed on every file and self-expires if the transfer is
        // abandoned. 1h matches the upload write-timeout ceiling.
        model_store_->reserve(id, 3600LL * 1000);

        const auto in_use = loaded_model_paths(slot_mgr_);
        std::vector<std::string> evicted = model_store_->make_room_for(size, in_use);

        auto slot = model_store_->begin_file(id, rel);
        if (!slot.ok) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", slot.error}}.dump(), "application/json");
            return;
        }

        std::ofstream out(slot.temp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            res.status = 500;
            res.set_content(R"({"error":"cannot open destination file for writing"})",
                            "application/json");
            return;
        }
        bool write_ok = true;
        const bool pumped = pump([&](const char* data, size_t len) -> bool {
            out.write(data, static_cast<std::streamsize>(len));
            if (!out) { write_ok = false; return false; }
            return true;
        });
        out.close();
        // close() flushes the final buffered block; a full-disk ENOSPC can
        // surface only here, so re-check the stream before declaring success.
        if (!out) write_ok = false;
        if (!pumped || !write_ok) {
            std::error_code ec;
            fs::remove(slot.temp_path, ec);
            res.status = 500;
            res.set_content(R"({"error":"upload interrupted or write failed"})",
                            "application/json");
            return;
        }

        std::string commit_err;
        if (!model_store_->commit_file(slot, &commit_err)) {
            res.status = 500;
            res.set_content(nlohmann::json{{"error", commit_err}}.dump(), "application/json");
            return;
        }
        model_store_->register_model(id, pin);
        if (!evicted.empty())
            MM_INFO("ModelStore: evicted {} model(s) to receive {}", evicted.size(), id);

        res.set_content(nlohmann::json{{"status", "stored"},
                                       {"model_id", id},
                                       {"load_path", model_store_->load_path(id)},
                                       {"evicted", evicted}}.dump(),
                        "application/json");
    });

    // ── GET /api/node/models/local ────────────────────────────────────────────
    // Report the node-local managed model cache. With ?id=<model_id> it answers
    // whether that model is already present (size + load path), so control can
    // skip a transfer the node already holds.
    server_->Get("/api/node/models/local", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (!model_store_) {
            res.set_content(nlohmann::json{{"available", false},
                                           {"models", nlohmann::json::array()}}.dump(),
                            "application/json");
            return;
        }
        if (req.has_param("id")) {
            const std::string id = req.get_param_value("id");
            auto m = model_store_->get(id);
            nlohmann::json j;
            j["available"] = true;
            j["present"]   = static_cast<bool>(m);
            if (m) {
                j["size_bytes"] = m->size_bytes;
                j["pinned"]     = m->pinned;
                j["load_path"]  = model_store_->load_path(id);
            }
            res.set_content(j.dump(), "application/json");
            return;
        }
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& m : model_store_->list()) {
            arr.push_back({{"id", m.id},
                           {"size_bytes", m.size_bytes},
                           {"last_used_ms", m.last_used_ms},
                           {"pinned", m.pinned},
                           {"load_path", model_store_->load_path(m.id)}});
        }
        res.set_content(nlohmann::json{{"available", true},
                                       {"root", model_store_->root()},
                                       {"free_bytes", model_store_->free_bytes()},
                                       {"min_free_mb", model_store_->min_free_mb()},
                                       {"models", arr}}.dump(),
                        "application/json");
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
            RuntimeClient* client = slot_lease.get();
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
