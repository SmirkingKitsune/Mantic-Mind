#include "node/node_api_server.hpp"
#include "node/engine_descriptor.hpp"
#include "node/engine_manager.hpp"
#include "node/node_state.hpp"
#include "node/engine_supervisor.hpp"
#include "node/model_store.hpp"
#include "node/ray_orchestration.hpp"
#include "common/engine_client.hpp"
#include "common/http_client.hpp"
#include "common/http_server.hpp"
#include "common/runtime_client.hpp"
#include "common/models.hpp"
#include "common/logger.hpp"
#include "common/pairing.hpp"
#include "common/util.hpp"
#include "common/sse_infer_ctx.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <utility>

namespace mm {

/// Upstream read timeout for a proxied SSE stream.
///
/// Long and FINITE. `set_read_timeout(0, 0)` reads as zero seconds in
/// cpp-httplib rather than as "no limit", which made every proxied telemetry
/// stream close instantly with a clean, empty 200 — defect D10. Infinity is not
/// the right answer either: a wedged engine should eventually release the socket.
/// An hour is far beyond any tick interval and far short of forever.
constexpr std::chrono::seconds kSseUpstreamReadTimeout{3600};
namespace fs = std::filesystem;

namespace {

// A llama.cpp runtime is usable for launching engines once it resolves to an
// executable (found on PATH or a completed managed install).
bool node_llama_runtime_ready(const LlamaRuntimeStatus& rt) {
    return (rt.status == "resolved" || rt.status == "ready") && !rt.executable_path.empty();
}

// True when :id names the engine these llama-shaped actions apply to.
// Otherwise writes a 400 naming the engine asked for and the reason.
//
// A refusal rather than a silent no-op: "switch variant" on Soma is not a
// request that should quietly succeed, and a 200 would tell the operator a
// switch happened. The engine id is echoed so the message is about what they
// asked for rather than about llama.cpp.
bool require_llama_engine(const httplib::Request& req, httplib::Response& res) {
    const auto it = req.path_params.find("id");
    const std::string id = it == req.path_params.end() ? std::string{} : it->second;
    if (id == "llama-cpp") return true;

    res.status = 400;
    res.set_content(
        nlohmann::json{
            {"error", "engine '" + id +
                          "' has no release variants or troubleshooting matrix; this action "
                          "exists only for llama-cpp"},
            {"engine", id},
            {"engines", EngineRegistry::instance().ids()}}
            .dump(),
        "application/json");
    return false;
}

// Load paths currently backing a live slot — models the store must never
// evict out from under a running engine.
std::set<std::string> loaded_model_paths(EngineSupervisor& engines) {
    std::set<std::string> paths;
    for (const auto& s : engines.slots()) {
        if (!s.model_path.empty()) paths.insert(s.model_path);
        if (!s.mmproj_path.empty()) paths.insert(s.mmproj_path);
    }
    return paths;
}

/// Classify a slot failure into a STRUCTURED code for control.
///
/// What this ends: AgentScheduler decided whether to evict-and-retry by
/// substring-matching six English phrases against this body
/// (agent_scheduler.cpp:904). Every engine had to reproduce those literals
/// verbatim to earn a retry, and rewording any of them would have broken
/// eviction with nothing to catch it. The phrases are still produced — they are
/// the human-readable detail — but the DECISION now rides a code.
std::string engine_error_code_for(const std::string& detail) {
    const auto lower = mm::util::to_lower(detail);
    const auto has = [&](const char* needle) {
        return lower.find(needle) != std::string::npos;
    };
    if (has("max slots reached") || has("max active slots reached") ||
        has("no available ports") || has("out of memory") || has("insufficient memory") ||
        has("insufficient vram") || has("already reserved")) {
        return "capacity_pressure";
    }
    if (has("not found") || has("no such file") || has("does not exist")) {
        return "model_not_found";
    }
    return "internal";
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
                             EngineSupervisor& engines,
                             std::string control_url,
                             std::string pairing_key)
    : state_(state)
    , engines_(engines)
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

void NodeApiServer::set_llama_provision_callback(LlamaProvisionCallback callback) {
    llama_provision_cb_ = std::move(callback);
}

void NodeApiServer::set_llama_update_callback(LlamaUpdateCallback callback) {
    llama_update_cb_ = std::move(callback);
}

void NodeApiServer::set_llama_switch_callback(LlamaSwitchCallback callback) {
    llama_switch_cb_ = std::move(callback);
}

void NodeApiServer::set_llama_check_update_callback(LlamaCheckUpdateCallback callback) {
    llama_check_update_cb_ = std::move(callback);
}

void NodeApiServer::set_llama_diagnose_callback(LlamaDiagnoseCallback callback) {
    llama_diagnose_cb_ = std::move(callback);
}

void NodeApiServer::set_llama_recovery_callback(LlamaRecoveryCallback callback) {
    llama_recovery_cb_ = std::move(callback);
}

void NodeApiServer::set_model_store(ModelStore* store) {
    model_store_ = store;
}

void NodeApiServer::set_engine_manager(NodeEngineManager* manager) {
    engine_manager_ = manager;
}

void NodeApiServer::set_engine_config_callback(EngineConfigCallback callback) {
    engine_config_cb_ = std::move(callback);
}

void NodeApiServer::set_ray_controller(RayController* controller) {
    ray_controller_ = controller;
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
        auto slots = engines_.slots();
        const int max_slots = static_cast<int>(engines_.max_slots());

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
        nlohmann::json j;
        j["node_id"]       = state_.get_node_id();
        j["hostname"]      = mm::util::hostname();
        j["slots"]         = slots;
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
        // Which models this node already holds, by cache id.
        //
        // Ids only, not sizes: control knows the size of anything it would send
        // and the list rides every 30s health poll, so the useful signal is
        // membership. Control charges a placement for the model's disk footprint
        // only when the target does NOT appear to hold it — before this it could
        // not ask, so it charged nothing and disk demand was invisible to
        // placement (roadmap D65).
        //
        // A HINT with a poll's worth of age on it, and safe in both directions:
        // a stale "resident" means control under-charges and the transfer path's
        // own make_room_for() evicts to fit, and a stale "absent" means it
        // over-charges and prefers a different node. Neither can place a model
        // somewhere it cannot go.
        if (model_store_ != nullptr) {
            nlohmann::json ids = nlohmann::json::array();
            for (const auto& m : model_store_->list())
                ids.push_back(m.id);
            j["local_model_ids"] = std::move(ids);
        } else {
            // Absent rather than empty would be indistinguishable from "this
            // node holds nothing", and those are different claims.
            j["local_model_ids"] = nlohmann::json::array();
            j["local_model_cache"] = false;
        }

        // Engines are DATA now, so health advertises the whole registry rather
        // than one runtime's path. `llama_server_path` is retained because
        // control's UI reads it, but it comes from the descriptor.
        j["engines"] = EngineRegistry::instance().ids();
        // The convergence signal control polls on. Reported even when engine
        // management is unconfigured — version 0 and `unconfigured` are exactly
        // what tells control to push, and omitting them would leave a node that
        // has never been told looking identical to one that is up to date.
        if (engine_manager_ != nullptr) {
            j["engine_config_version"] = engine_manager_->current_config().version;
            j["engine_conformance"] = engine_manager_->conformance();
            j["engine_runtimes"] = engine_manager_->engine_statuses();
        } else {
            j["engine_config_version"] = 0;
            EngineConformance none;
            none.detail = "engine management is not configured on this node";
            j["engine_conformance"] = none;
            j["engine_runtimes"] = nlohmann::json::array();
        }
        if (const auto* llama = EngineRegistry::instance().find("llama-cpp")) {
            j["llama_server_path"] = llama->build_launch({}).executable;
        }
        j["llama_runtime"] = state_.get_llama_runtime();
        j["action_progress"] = state_.get_action_progress();
        j["ray"] = ray_controller_ != nullptr
            ? nlohmann::json(ray_controller_->status())
            : nlohmann::json(RayRuntimeStatus{});

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
    server_->Post("/api/node/actions/cancel", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401;
            return;
        }
        const bool accepted = state_.request_action_cancel();
        if (!accepted) {
            res.status = 409;
            res.set_content(R"({"error":"no cancelable action is active"})", "application/json");
            return;
        }
        res.set_content(R"({"cancel_requested":true})", "application/json");
    });

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

    // ── Engines ───────────────────────────────────────────────────────────────
    //
    // These replace six /api/node/runtime/llama/* routes. The old paths named
    // one engine in the URL, so a second engine could not be provisioned,
    // switched, or diagnosed without a second family of routes spelling its
    // name — which is how the node ended up with exactly one engine it could
    // manage.
    //
    // The llama-only ACTIONS survive under :id and refuse a non-llama engine by
    // name. That is not a limitation being papered over: release variants and a
    // troubleshooting matrix are things llama.cpp has and Soma does not.

    // GET /api/node/engines — everything control needs to render this node's
    // engine state, in one round trip.
    server_->Get("/api/node/engines", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (engine_manager_ == nullptr) {
            res.status = 501;
            res.set_content(R"({"error":"engine management is not configured on this node"})",
                            "application/json");
            return;
        }
        nlohmann::json artifacts = nlohmann::json::array();
        for (const auto& a : engine_manager_->shareable_artifacts()) {
            artifacts.push_back(nlohmann::json{{"engine_id", a.engine_id},
                                               {"version", a.version},
                                               {"platform", a.platform},
                                               {"arch", a.arch},
                                               {"variant", a.variant},
                                               {"fingerprint", a.fingerprint()}});
        }
        const auto cluster_cfg = engine_manager_->current_config();
        res.set_content(nlohmann::json{{"engines", engine_manager_->engine_statuses()},
                                       {"conformance", engine_manager_->conformance()},
                                       {"engine_config_version", cluster_cfg.version},
                                       {"engine_config", cluster_cfg},
                                       {"shareable_artifacts", artifacts},
                                       // The llama detail the generic status
                                       // cannot carry; the TUI's troubleshooting
                                       // wizard reads it.
                                       {"llama_runtime", state_.get_llama_runtime()}}
                            .dump(),
                        "application/json");
    });

    // POST /api/node/engine-config — the master telling this node what to run.
    server_->Post("/api/node/engine-config", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (engine_manager_ == nullptr || !engine_config_cb_) {
            res.status = 501;
            res.set_content(R"({"error":"engine management is not configured on this node"})",
                            "application/json");
            return;
        }
        ClusterEngineConfig cfg;
        try {
            cfg = nlohmann::json::parse(req.body).get<ClusterEngineConfig>();
        } catch (const std::exception& e) {
            // Includes the forbidden-key refusal from from_json. Reported to
            // the master verbatim: it pushed a config this node will not run,
            // and it needs the reason rather than a generic 400.
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        // Validated against THIS node's registry, so a config naming an engine
        // this build cannot run is refused at the push instead of surfacing
        // later as an unexplained conformance failure.
        std::string verr;
        if (!validate_engine_config(cfg, EngineRegistry::instance().ids(), verr)) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", verr}}.dump(), "application/json");
            return;
        }

        MM_INFO("Applying cluster engine config v{} from control", cfg.version);
        engine_config_cb_(cfg);

        // Answers immediately with `converging`; provisioning runs in the
        // background because a source build takes minutes and control's push
        // must not block on it.
        res.set_content(nlohmann::json{{"accepted", true},
                                       {"engine_config_version", cfg.version},
                                       {"conformance", engine_manager_->conformance()}}
                            .dump(),
                        "application/json");
    });

    // POST /api/node/engines/:id/provision
    server_->Post("/api/node/engines/:id/provision", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (!require_llama_engine(req, res)) return;
        bool want_update = false;
        std::string accelerator;
        if (!req.body.empty()) {
            try {
                const auto j = nlohmann::json::parse(req.body);
                want_update = j.value("update", false);
                accelerator = j.value("accelerator", std::string{});
                if (!accelerator.empty() && !want_update) {
                    res.status = 400;
                    res.set_content(
                        R"({"error":"accelerator is only valid with update=true"})",
                        "application/json");
                    return;
                }
            } catch (...) {
                res.status = 400;
                res.set_content(R"({"error":"invalid JSON body"})", "application/json");
                return;
            }
        }
        if ((want_update && !llama_update_cb_) || (!want_update && !llama_provision_cb_)) {
            res.status = 501;
            res.set_content(want_update
                                ? R"({"error":"llama.cpp update is not configured"})"
                                : R"({"error":"llama.cpp provisioning is not configured"})",
                            "application/json");
            return;
        }
        auto runtime = want_update ? llama_update_cb_(accelerator)
                                   : llama_provision_cb_();
        state_.set_llama_runtime(runtime);
        if (!runtime.last_error.empty()) state_.set_last_error(runtime.last_error);
        if (runtime.status == "failed" || runtime.status == "disabled") res.status = 500;
        else if (want_update && !runtime.last_error.empty()) res.status = 409;
        else res.status = 200;
        res.set_content(nlohmann::json{{"llama_runtime", runtime}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/engines/:id/check-update", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (!require_llama_engine(req, res)) return;
        if (!llama_check_update_cb_) {
            res.status = 501;
            res.set_content(R"({"error":"llama.cpp update checking is not configured"})",
                            "application/json");
            return;
        }
        auto runtime = llama_check_update_cb_();
        state_.set_llama_runtime(runtime);
        res.set_content(nlohmann::json{{"llama_runtime", runtime}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/engines/:id/switch", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (!require_llama_engine(req, res)) return;
        if (!llama_switch_cb_) {
            res.status = 501;
            res.set_content(R"({"error":"llama.cpp engine switching is not configured"})",
                            "application/json");
            return;
        }
        std::string variant;
        try {
            const auto j = nlohmann::json::parse(req.body);
            variant = j.value("variant", std::string{});
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid JSON body"})", "application/json");
            return;
        }
        if (variant.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"variant is required"})", "application/json");
            return;
        }
        auto runtime = llama_switch_cb_(variant);
        state_.set_llama_runtime(runtime);
        if (!runtime.last_error.empty()) state_.set_last_error(runtime.last_error);
        if (runtime.status == "failed" || runtime.status == "disabled") res.status = 500;
        else if (!runtime.last_error.empty()) res.status = 409;
        else res.status = 200;
        res.set_content(nlohmann::json{{"llama_runtime", runtime}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/engines/:id/diagnose", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (!require_llama_engine(req, res)) return;
        if (!llama_diagnose_cb_) {
            res.status = 501;
            res.set_content(R"({"error":"llama.cpp diagnostics are not configured"})",
                            "application/json");
            return;
        }
        auto runtime = llama_diagnose_cb_();
        state_.set_llama_runtime(runtime);
        res.set_content(nlohmann::json{{"llama_runtime", runtime}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/engines/:id/recover", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (!require_llama_engine(req, res)) return;
        if (!llama_recovery_cb_) {
            res.status = 501;
            res.set_content(R"({"error":"llama.cpp recovery is not configured"})",
                            "application/json");
            return;
        }
        std::string action;
        std::string variant;
        try {
            const auto j = nlohmann::json::parse(req.body);
            action = j.value("action", std::string{});
            variant = j.value("variant", std::string{});
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid JSON body"})", "application/json");
            return;
        }
        const std::set<std::string> allowed{
            "retry", "target", "compile-anyway", "release"};
        if (!allowed.count(action) || (action == "release" && variant.empty())) {
            res.status = 400;
            res.set_content(
                R"({"error":"action must be retry, target, compile-anyway, or release; release requires variant"})",
                "application/json");
            return;
        }
        auto runtime = llama_recovery_cb_(action, variant);
        state_.set_llama_runtime(runtime);
        if (!runtime.last_error.empty()) state_.set_last_error(runtime.last_error);
        if (runtime.status == "failed") res.status = 500;
        else if (!runtime.last_error.empty()) res.status = 409;
        else res.status = 200;
        res.set_content(nlohmann::json{{"llama_runtime", runtime}}.dump(),
                        "application/json");
    });

    server_->Get("/api/node/ray/status", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (ray_controller_ == nullptr) {
            res.status = 501;
            res.set_content(R"({"error":"Ray orchestration is not configured"})",
                            "application/json");
            return;
        }
        res.set_content(nlohmann::json(ray_controller_->status()).dump(),
                        "application/json");
    });

    server_->Post("/api/node/vllm/preflight", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            const auto j = nlohmann::json::parse(req.body);
            const std::string model = j.value("model_path", std::string{});
            if (model.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"model_path required"})", "application/json");
                return;
            }
            if (!util::is_hf_repo_id(model)) {
                const std::filesystem::path path(model);
                std::error_code ec;
                if (!path.is_absolute() || !std::filesystem::exists(path, ec)) {
                    res.status = 409;
                    res.set_content(
                        nlohmann::json{{"error", "shared vLLM model path is not accessible"},
                                       {"model_path", model}}
                            .dump(),
                        "application/json");
                    return;
                }
            }
            res.set_content(nlohmann::json{{"accessible", true},
                                           {"model_path", model}}.dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    server_->Post("/api/node/ray/start", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (ray_controller_ == nullptr) {
            res.status = 501;
            res.set_content(R"({"error":"Ray orchestration is not configured"})",
                            "application/json");
            return;
        }
        try {
            const auto j = nlohmann::json::parse(req.body);
            RayStartConfig cfg;
            cfg.group_id = j.value("group_id", std::string{});
            cfg.agent_id = j.value("agent_id", std::string{});
            cfg.role = j.value("role", std::string{"head"});
            cfg.head_address = j.value("head_address", std::string{});
            cfg.node_ip = j.value("node_ip", std::string{});
            cfg.transport = j.value("transport", std::string{"nccl"});
            cfg.num_gpus = j.value("num_gpus", 0);
            if (cfg.transport != "nccl" && cfg.transport != "gloo") {
                res.status = 400;
                res.set_content(R"({"error":"transport must be nccl or gloo"})",
                                "application/json");
                return;
            }
            const auto current = ray_controller_->status();
            if (current.group_id.empty()) {
                const auto slots = engines_.slots();
                const bool occupied = std::any_of(
                    slots.begin(), slots.end(), [](const SlotInfo& slot) {
                        return slot.state != SlotState::Empty;
                    });
                if (occupied) {
                    res.status = 409;
                    res.set_content(
                        R"({"error":"Ray requires an idle node with no engine slots","code":"capacity_pressure"})",
                        "application/json");
                    return;
                }
                const int available_gpus = state_.get_capabilities().gpu_count;
                if (cfg.num_gpus < 1 || cfg.num_gpus > available_gpus) {
                    res.status = 409;
                    res.set_content(
                        nlohmann::json{{"error", "invalid Ray GPU reservation"},
                                       {"code", "capacity_pressure"},
                                       {"requested_gpus", cfg.num_gpus},
                                       {"available_gpus", available_gpus}}
                            .dump(),
                        "application/json");
                    return;
                }
            }
            std::string error;
            if (!ray_controller_->start(cfg, error)) {
                const bool conflict = error.find("reserved") != std::string::npos ||
                    error.find("already bound") != std::string::npos ||
                    error.find("already in progress") != std::string::npos;
                res.status = conflict ? 409 : 500;
                res.set_content(nlohmann::json{{"error", error},
                                               {"ray", ray_controller_->status()}}.dump(),
                                "application/json");
                return;
            }
            res.set_content(nlohmann::json(ray_controller_->status()).dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    server_->Post("/api/node/ray/stop", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (ray_controller_ == nullptr) {
            res.status = 501;
            res.set_content(R"({"error":"Ray orchestration is not configured"})",
                            "application/json");
            return;
        }
        try {
            const auto j = nlohmann::json::parse(req.body);
            std::string error;
            if (!ray_controller_->stop(j.value("group_id", std::string{}), error)) {
                res.status = error.find("ownership") != std::string::npos ? 409 : 500;
                res.set_content(nlohmann::json{{"error", error},
                                               {"ray", ray_controller_->status()}}.dump(),
                                "application/json");
                return;
            }
            res.set_content(nlohmann::json(ray_controller_->status()).dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    server_->Post("/api/node/load-model", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string model_ref  = j.value("model_path", std::string{});
            std::string mmproj_ref = j.value("mmproj_path", std::string{});
            const bool vision_enabled = j.value("vision_enabled", !mmproj_ref.empty());
            std::string agent_id   = j.value("agent_id",   std::string{});
            // Optional: identity + pin flag for a control-transferred model, so
            // the node can refresh the LRU use-queue and pinned state on load.
            const std::string model_id = j.value("model_id", std::string{});
            const std::string mmproj_model_id =
                j.value("mmproj_model_id", std::string{});
            const bool model_pin       = j.value("pin", false);
            // llama.cpp is the default runtime: a payload without an explicit
            // backend loads through llama-server. Control always sends the
            // field, so this default only decides hand-written requests.
            const std::string backend = mm::util::to_lower(mm::util::trim(
                j.value("backend", std::string{"llama-cpp"})));

            // Ray group ownership is a node-wide exclusive reservation. The
            // only process admitted while it is active is the group's vLLM
            // server on the head, for the same agent. Workers run Ray only.
            if (ray_controller_ != nullptr) {
                const auto ray = ray_controller_->status();
                const bool owned_head_load = ray.active() && backend == "vllm" &&
                    ray.role == "head" && !agent_id.empty() &&
                    agent_id == ray.agent_id;
                if (!ray.group_id.empty() && !owned_head_load) {
                    res.status = 409;
                    res.set_content(
                        nlohmann::json{{"error", "node is reserved by Ray group '" +
                                                       ray.group_id + "'"},
                                       {"code", "capacity_pressure"},
                                       {"ray", ray}}
                            .dump(),
                        "application/json");
                    return;
                }
            }
            RuntimeSettings runtime_settings;
            if (j.contains("runtime_settings")) runtime_settings = j["runtime_settings"].get<RuntimeSettings>();
            const std::string served_model_name =
                j.value("served_model_name", std::string{});

            // A registry lookup, not a string literal. The `supported_backends`
            // list in the body is now the registry's ACTUAL contents, so it is
            // accurate by construction — the two hand-maintained copies of this
            // check (here and in the restore handler below) could not be.
            if (mm::EngineRegistry::instance().find(backend) == nullptr) {
                res.status = 400;
                const std::string msg = "unsupported node inference backend: " + backend;
                state_.set_last_error(msg);
                res.set_content(
                    nlohmann::json{{"error", msg},
                                   {"supported_backends", mm::EngineRegistry::instance().ids()}}
                        .dump(),
                    "application/json");
                return;
            }

            if (model_ref.empty()) {
                res.status = 400;
                state_.set_last_error("model_path required");
                res.set_content(R"({"error":"model_path required"})", "application/json");
                return;
            }

            const std::string model_path = sanitize_path_for_runtime(model_ref);
            const std::string mmproj_path = sanitize_path_for_runtime(mmproj_ref);

            // ── engine-specific preconditions ────────────────────────────────
            //
            // Everything from here to the dispatch below used to run for EVERY
            // backend: the projector rules, a llama-runtime readiness check, and
            // `is_regular_file(model_path)`. All three are llama.cpp's. A Soma
            // container is a directory, so the third refused every Soma load
            // twenty lines before the dispatch that would have routed it away
            // from llama.cpp. Defect D8.
            //
            // Vision is llama.cpp's alone — Soma answers 422 for image parts by
            // contract — so the mmproj rules stay gated on the backend here
            // rather than becoming a descriptor field no other engine would set.
            const bool is_llama = (backend == "llama-cpp");

            if (is_llama && vision_enabled && mmproj_path.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"vision-enabled llama-cpp load requires mmproj_path"})",
                                "application/json");
                return;
            }
            if (is_llama && !vision_enabled && !mmproj_path.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"llama.cpp mmproj_path requires vision_enabled=true"})",
                                "application/json");
                return;
            }
            if (is_llama && !mmproj_path.empty()) {
                std::error_code ec;
                if (!fs::is_regular_file(mmproj_path, ec)) {
                    res.status = 400;
                    const std::string msg = "projector path not found on this node: " + mmproj_path;
                    state_.set_last_error(msg);
                    res.set_content(nlohmann::json{{"error", "projector not found on node"},
                                                   {"detail", msg}}.dump(),
                                    "application/json");
                    return;
                }
            }

            // Fail fast rather than spawn an engine without a usable runtime.
            // This is llama.cpp's PROVISIONED llama-server binary; Soma's runtime
            // is the `soma` executable the descriptor already knows how to launch,
            // and it has no equivalent readiness state to consult.
            if (is_llama)
            if (const auto rt = state_.get_llama_runtime(); !node_llama_runtime_ready(rt)) {
                res.status = 503;
                const std::string msg = rt.last_error.empty()
                    ? "llama.cpp runtime is not ready on this node (status=" + rt.status + ")"
                    : "llama.cpp runtime is not ready on this node: " + rt.last_error;
                state_.set_last_error(msg);
                res.set_content(nlohmann::json{{"error", "llama runtime not ready"},
                                               {"detail", msg},
                                               {"llama_runtime", rt}}.dump(),
                                "application/json");
                return;
            }

            // What a loadable model reference IS belongs to the engine, not to
            // this handler: a GGUF file for llama.cpp, a converted container
            // directory for Soma. The descriptor was already the registry of
            // per-engine knowledge — this is one more row in it rather than one
            // more branch here.
            if (const auto* desc = mm::EngineRegistry::instance().find(backend);
                desc != nullptr && desc->validate_model_ref) {
                std::string detail;
                if (!desc->validate_model_ref(model_path, detail)) {
                    res.status = 400;
                    state_.set_last_error(detail);
                    res.set_content(nlohmann::json{{"error", "model not found on node"},
                                                   {"detail", detail},
                                                   {"engine", backend},
                                                   {"model_path", model_path}}.dump(),
                                    "application/json");
                    return;
                }
            }
            // The engine id travels with the request now. Which engine to launch,
            // what argv it takes and how to probe it are the descriptor's; this
            // handler's job is to pass the id along, not to know any of that.
            EngineLoadRequest load_req;
            load_req.model_path = model_path;
            load_req.mmproj_path = mmproj_path;
            load_req.settings = runtime_settings;
            load_req.served_model_name = served_model_name.empty()
                ? model_ref : served_model_name;
            if (backend == "vllm") {
                if (engine_manager_ == nullptr) {
                    res.status = 503;
                    res.set_content(R"({"error":"engine manager unavailable"})",
                                    "application/json");
                    return;
                }
                const auto cluster = engine_manager_->current_config();
                const EngineSpec* spec = cluster.find("vllm");
                if (spec == nullptr) {
                    res.status = 409;
                    res.set_content(
                        R"({"error":"vllm is not in the applied cluster engine configuration"})",
                        "application/json");
                    return;
                }
                load_req.vllm = effective_vllm_config(*spec);
                load_req.ray_address = j.value("ray_address", std::string{});
                load_req.host_ip = j.value("host_ip", std::string{});
                if (j.contains("gpu_devices") && j.at("gpu_devices").is_array())
                    load_req.gpu_devices = j.at("gpu_devices").get<std::vector<int>>();
                const int wanted = load_req.vllm->tensor_parallel_size;
                const int available = state_.get_capabilities().gpu_count;
                std::set<int> reserved;
                for (const auto& slot : engines_.slots()) {
                    if (slot.backend != "vllm" || slot.state == SlotState::Empty ||
                        slot.state == SlotState::Error) continue;
                    reserved.insert(slot.gpu_devices.begin(), slot.gpu_devices.end());
                }
                if (load_req.gpu_devices.empty()) {
                    for (int index = 0;
                         index < available &&
                         static_cast<int>(load_req.gpu_devices.size()) < wanted;
                         ++index) {
                        if (reserved.count(index) == 0)
                            load_req.gpu_devices.push_back(index);
                    }
                    if (static_cast<int>(load_req.gpu_devices.size()) < wanted) {
                        res.status = 503;
                        res.set_content(
                            nlohmann::json{{"error", "insufficient GPUs for vLLM"},
                                           {"code", "capacity_pressure"},
                                           {"required_gpus", wanted},
                                           {"available_gpus", available}}
                                .dump(),
                            "application/json");
                        return;
                    }
                } else {
                    std::set<int> selected(load_req.gpu_devices.begin(),
                                           load_req.gpu_devices.end());
                    const bool valid = static_cast<int>(selected.size()) == wanted &&
                        selected.size() == load_req.gpu_devices.size() &&
                        std::all_of(selected.begin(), selected.end(), [&](const int index) {
                            return index >= 0 && index < available &&
                                   reserved.count(index) == 0;
                        });
                    if (!valid) {
                        res.status = 409;
                        res.set_content(
                            nlohmann::json{{"error", "invalid or unavailable vLLM GPU reservation"},
                                           {"code", "capacity_pressure"},
                                           {"required_gpus", wanted},
                                           {"available_gpus", available},
                                           {"requested_devices", load_req.gpu_devices}}
                                .dump(),
                            "application/json");
                        return;
                    }
                }
            }
            auto slot_id = engines_.load(backend, load_req, agent_id);
            if (slot_id.empty()) {
                nlohmann::json err = {{"error", "failed to load model"}};
                auto detail = engines_.last_error();
                const auto code = engine_error_code_for(detail);
                // 503 for capacity, so the status alone already says "try again
                // elsewhere"; the code says it unambiguously.
                res.status = code == "capacity_pressure" ? 503 : 500;
                err["code"] = code;
                err["engine"] = backend;
                err["llama_runtime"] = state_.get_llama_runtime();
                err["model_path"] = model_path;
                state_.set_last_error(detail.empty() ? "failed to load model" : detail);
                if (!detail.empty()) err["detail"] = detail;
                res.set_content(err.dump(), "application/json");
            } else {
                // Update NodeState for UI
                auto slots_info = engines_.slots();
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
                if (model_store_) {
                    for (const auto& id : {model_id, mmproj_model_id}) {
                        if (id.empty()) continue;
                        if (model_pin) model_store_->set_pinned(id, true);
                        model_store_->touch(id);
                        if (id == model_id && mmproj_model_id == model_id) break;
                    }
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
                auto unload = engines_.unload_all(false);
                if (!unload.ok()) {
                    res.status = unload.status == EngineOpStatus::Busy ? 409 : 500;
                    res.set_content(nlohmann::json{{"error", unload.message}}.dump(),
                                    "application/json");
                    return;
                }
                state_.set_loaded_model("");
                state_.set_active_agent("");
                state_.set_slots({});
                state_.set_last_error("");
            } else {
                auto unload = engines_.unload(slot_id);
                state_.set_slots(engines_.slots());
                if (!unload.ok()) {
                    res.status = unload.status == EngineOpStatus::NotFound ? 404
                               : unload.status == EngineOpStatus::Busy ? 409
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

    // ── Engine artifact sharing ───────────────────────────────────────────────
    //
    // A node that has built an engine serves it to a node that needs the same
    // one. Bytes go NODE TO NODE; control brokers — it decides who shares with
    // whom, mints and revokes the credential, and never stores the artifact.
    //
    // Three rules, enforced below rather than only documented:
    //
    //  1. The expected digest comes from CONTROL, out of band from the bytes.
    //     A digest supplied by the sending node would let a compromised source
    //     define its own checksum, and the verification would be theatre.
    //  2. Fingerprint equality is EXACT across (engine, version, platform,
    //     arch, variant). A near-match is a wrong binary, not a close one — an
    //     x86_64 CUDA-12 build on an aarch64 host does not run.
    //  3. The receiving node re-checks the artifact against its OWN
    //     environment. Control's view of this node's platform is a report from
    //     this node; the only authority on what executes here is here.

    // POST /api/node/engines/prepare — package the artifact and report its
    // digest, WITHOUT sending it anywhere.
    //
    // Phase one of two, and the split exists for one reason: only the holder of
    // an artifact can hash it, but the receiver must learn the expected digest
    // from CONTROL rather than from the sender. So control asks here, relays
    // the answer to the target over its own authenticated channel, and only
    // then triggers the push.
    //
    // What that buys, stated precisely: a transfer credential that leaked
    // cannot be used to push DIFFERENT bytes, because the digest the target
    // will accept was fixed by an authenticated peer before the credential was
    // ever used. It does NOT make a compromised source safe — a source that
    // lies here supplies both the artifact and the hash. Signed builds would be
    // the answer to that, and this is not one.
    server_->Post("/api/node/engines/prepare", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        if (engine_manager_ == nullptr) {
            res.status = 501;
            res.set_content(R"({"error":"engine management is not configured on this node"})",
                            "application/json");
            return;
        }
        std::string fingerprint;
        try {
            fingerprint = nlohmann::json::parse(req.body).value("fingerprint", std::string{});
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        // Must be an artifact this node actually holds and can vouch for.
        // shareable_artifacts() excludes PATH-resolved binaries: this node
        // cannot say what build one of those is, and shipping it to a peer
        // under a fingerprint it invented is worse than a rebuild.
        EngineArtifact artifact;
        bool found = false;
        for (const auto& a : engine_manager_->shareable_artifacts()) {
            if (a.fingerprint() == fingerprint) { artifact = a; found = true; break; }
        }
        if (!found) {
            res.status = 404;
            res.set_content(
                nlohmann::json{{"error", "this node has no shareable artifact " + fingerprint}}
                    .dump(),
                "application/json");
            return;
        }

        const std::string operation_id = "engine-share-" + mm::util::generate_uuid();
        NodeActionProgress progress;
        progress.active = true;
        progress.operation_id = operation_id;
        progress.kind = "engine_share";
        progress.action = "Packaging engine build";
        progress.target = fingerprint;
        progress.stage = "packaging";
        progress.fraction = -1.0;
        state_.set_action_progress(progress);

        std::error_code ec;
        const std::string token = mm::util::generate_uuid();
        const fs::path package =
            fs::temp_directory_path(ec) / ("mm-engine-" + token + ".tar.gz");
        std::string err;
        if (!engine_manager_->package(artifact.engine_id, package.string(), err)) {
            state_.clear_action_progress(operation_id);
            res.status = 500;
            res.set_content(nlohmann::json{{"error", "packaging failed: " + err}}.dump(),
                            "application/json");
            return;
        }
        const std::string digest = pairing::sha256_file_hex(package.string(), &err);
        state_.clear_action_progress(operation_id);
        if (digest.empty()) {
            fs::remove(package, ec);
            res.status = 500;
            res.set_content(nlohmann::json{{"error", "cannot hash package: " + err}}.dump(),
                            "application/json");
            return;
        }

        int64_t size = 0;
        {
            std::error_code sec;
            size = static_cast<int64_t>(fs::file_size(package, sec));
            if (sec) size = 0;
        }
        state_.set_prepared_engine_package(token, package.string());
        res.set_content(nlohmann::json{{"prepared", true},
                                       {"fingerprint", fingerprint},
                                       {"sha256", digest},
                                       {"package_token", token},
                                       {"size_bytes", size}}
                            .dump(),
                        "application/json");
    });

    // POST /api/node/engines/share — phase two: send the package prepared
    // above. Takes the token rather than a fingerprint so the bytes pushed are
    // provably the bytes hashed; re-packaging here would let the artifact
    // change between the digest control relayed and what arrives.
    server_->Post("/api/node/engines/share", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        std::string token, target_url, transfer_key, fingerprint;
        try {
            const auto j = nlohmann::json::parse(req.body);
            token = j.value("package_token", std::string{});
            target_url = j.value("target_url", std::string{});
            transfer_key = j.value("transfer_key", std::string{});
            fingerprint = j.value("fingerprint", std::string{});
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        if (token.empty() || target_url.empty() || fingerprint.empty()) {
            res.status = 400;
            res.set_content(
                R"({"error":"package_token, target_url and fingerprint are required"})",
                "application/json");
            return;
        }

        // Consumed: the package is deleted after one send either way, so a
        // repeated call cannot re-push bytes whose brokered authorization has
        // already been spent.
        const std::string package = state_.take_prepared_engine_package(token);
        if (package.empty()) {
            res.status = 409;
            res.set_content(R"({"error":"unknown or already-used package_token"})",
                            "application/json");
            return;
        }

        const std::string operation_id = "engine-share-" + token;
        NodeActionProgress progress;
        progress.active = true;
        progress.operation_id = operation_id;
        progress.kind = "engine_share";
        progress.action = "Sharing engine build";
        progress.target = fingerprint;
        progress.stage = "uploading";
        progress.fraction = -1.0;
        state_.set_action_progress(progress);

        HttpClient peer(target_url);
        if (!transfer_key.empty()) peer.set_bearer_token(transfer_key);
        // An engine package is hundreds of megabytes; the default 10 s write
        // timeout would abort every share on anything but a LAN.
        peer.set_timeouts(10, 300, 3600);
        const auto upload =
            peer.post_file("/api/node/engines/receive", package,
                           {{"X-MM-Engine-Fingerprint", fingerprint}});

        std::error_code ec;
        fs::remove(package, ec);
        state_.clear_action_progress(operation_id);

        if (!upload.ok()) {
            res.status = 502;
            res.set_content(nlohmann::json{{"error", "peer refused the artifact"},
                                           {"peer_status", upload.status},
                                           {"peer_body", upload.body}}
                                .dump(),
                            "application/json");
            return;
        }
        MM_INFO("Shared engine artifact {} to {}", fingerprint, target_url);
        res.set_content(nlohmann::json{{"shared", true}, {"fingerprint", fingerprint}}.dump(),
                        "application/json");
    });

    // POST /api/node/engines/receive — accept a peer's package, streamed to
    // disk without buffering, then verified and installed.
    //   X-MM-Engine-Fingerprint  what this claims to be
    //   X-MM-Engine-Sha256       the sender's digest (informational)
    // The digest that AUTHORIZES the install is the one control recorded when
    // it brokered the transfer, matched below.
    server_->PostUpload("/api/node/engines/receive",
        [this](const httplib::Request& req, httplib::Response& res,
               const HttpServer::UploadPump& pump) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401;
            res.set_content(R"({"error":"unauthorized"})", "application/json");
            return;
        }
        if (engine_manager_ == nullptr) {
            res.status = 501;
            res.set_content(R"({"error":"engine management is not configured on this node"})",
                            "application/json");
            return;
        }

        const std::string fingerprint =
            util::trim(req.get_header_value("X-MM-Engine-Fingerprint"));
        EngineArtifact artifact;
        if (fingerprint.empty() || !parse_engine_fingerprint(fingerprint, artifact)) {
            res.status = 400;
            res.set_content(R"({"error":"X-MM-Engine-Fingerprint required and must be well-formed"})",
                            "application/json");
            return;
        }

        // The expected digest, relayed by control from the source's prepare
        // step before the transfer credential was ever handed out. Absent means
        // nothing brokered this, and an unbrokered artifact is refused rather
        // than trusted on the sender's word.
        const std::string expected = state_.take_expected_engine_digest(fingerprint);
        if (expected.empty()) {
            res.status = 409;
            res.set_content(
                nlohmann::json{
                    {"error", "no brokered transfer is expected for " + fingerprint +
                                  "; control must authorize a share before it is sent"}}
                    .dump(),
                "application/json");
            return;
        }

        const std::string operation_id = "engine-receive-" + mm::util::generate_uuid();
        NodeActionProgress progress;
        progress.active = true;
        progress.operation_id = operation_id;
        progress.kind = "engine_receive";
        progress.action = "Receiving engine build";
        progress.target = fingerprint;
        progress.stage = "receiving";
        progress.fraction = -1.0;
        state_.set_action_progress(progress);

        std::error_code ec;
        const fs::path temp =
            fs::temp_directory_path(ec) / ("mm-engine-in-" + mm::util::generate_uuid() + ".tar.gz");
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) {
            state_.clear_action_progress(operation_id);
            res.status = 500;
            res.set_content(R"({"error":"cannot open destination file for writing"})",
                            "application/json");
            return;
        }
        bool write_ok = true;
        int64_t received = 0;
        const bool pumped = pump([&](const char* data, size_t len) -> bool {
            out.write(data, static_cast<std::streamsize>(len));
            if (!out) { write_ok = false; return false; }
            received += static_cast<int64_t>(len);
            progress.bytes_done = received;
            state_.set_action_progress(progress);
            return true;
        });
        out.close();
        if (!out) write_ok = false;

        if (!pumped || !write_ok) {
            fs::remove(temp, ec);
            state_.clear_action_progress(operation_id);
            res.status = 500;
            res.set_content(R"({"error":"upload interrupted or write failed"})",
                            "application/json");
            return;
        }

        progress.stage = "verifying";
        state_.set_action_progress(progress);
        std::string hash_err;
        const std::string actual = pairing::sha256_file_hex(temp.string(), &hash_err);
        if (actual.empty() || actual != expected) {
            // Deleted, not quarantined: a package whose bytes do not match what
            // control authorized has no legitimate use on this node, and
            // keeping it invites someone to install it manually.
            fs::remove(temp, ec);
            state_.clear_action_progress(operation_id);
            MM_WARN("Refused engine artifact {}: digest mismatch (expected {}, got {})",
                    fingerprint, expected, actual.empty() ? "<unreadable>" : actual);
            res.status = 409;
            res.set_content(nlohmann::json{{"error", "digest mismatch; artifact refused"},
                                           {"expected_sha256", expected}}
                                .dump(),
                            "application/json");
            return;
        }

        progress.stage = "installing";
        state_.set_action_progress(progress);
        artifact.sha256 = actual;
        std::string install_err;
        const bool installed =
            engine_manager_->install_package(temp.string(), artifact, install_err);
        fs::remove(temp, ec);
        state_.clear_action_progress(operation_id);

        if (!installed) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", install_err}}.dump(), "application/json");
            return;
        }
        MM_INFO("Installed engine artifact {} received from a peer", fingerprint);
        res.set_content(nlohmann::json{{"installed", true},
                                       {"fingerprint", fingerprint},
                                       {"conformance", engine_manager_->conformance()}}
                            .dump(),
                        "application/json");
    });

    // POST /api/node/engines/expect — control announcing an inbound artifact.
    //
    // Split from the share call because the two hit DIFFERENT nodes: control
    // tells the RECEIVER what digest to expect, then tells the SENDER to push.
    // Without this the receiver would have to trust the sender's own digest.
    server_->Post("/api/node/engines/expect", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            const auto j = nlohmann::json::parse(req.body);
            const auto fingerprint = j.value("fingerprint", std::string{});
            const auto sha256 = j.value("sha256", std::string{});
            if (fingerprint.empty() || sha256.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"fingerprint and sha256 are required"})",
                                "application/json");
                return;
            }
            state_.expect_engine_digest(fingerprint, sha256);
            res.set_content(R"({"expecting":true})", "application/json");
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

        const std::string operation_id =
            "receive-" + id + "-" + mm::util::generate_uuid();
        int64_t received = 0;
        auto publish_receive_progress =
            [&](const std::string& stage, const std::string& detail) {
            NodeActionProgress progress;
            progress.active = true;
            progress.operation_id = operation_id;
            progress.kind = "model_receive";
            progress.action = "Downloading model";
            progress.target = id;
            progress.stage = stage;
            progress.detail = detail;
            progress.bytes_done = received;
            progress.bytes_total = size;
            progress.fraction = size > 0
                ? std::min(1.0, static_cast<double>(received) / static_cast<double>(size))
                : -1.0;
            progress.cancelable = true;
            state_.set_action_progress(progress);
        };
        publish_receive_progress("preparing cache", rel);

        // Reserve this model for the whole (possibly multi-file, multi-request)
        // transfer so neither make_room_for here nor the background disk-pressure
        // thread can evict files we have already received for it. The reservation
        // is refreshed on every file and self-expires if the transfer is
        // abandoned. 1h matches the upload write-timeout ceiling.
        model_store_->reserve(id, 3600LL * 1000);

        const auto in_use = loaded_model_paths(engines_);
        std::vector<std::string> evicted = model_store_->make_room_for(size, in_use);

        auto slot = model_store_->begin_file(id, rel);
        if (!slot.ok) {
            state_.clear_action_progress(operation_id);
            res.status = 400;
            res.set_content(nlohmann::json{{"error", slot.error}}.dump(), "application/json");
            return;
        }

        std::ofstream out(slot.temp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            state_.clear_action_progress(operation_id);
            res.status = 500;
            res.set_content(R"({"error":"cannot open destination file for writing"})",
                            "application/json");
            return;
        }
        bool write_ok = true;
        bool canceled = false;
        publish_receive_progress("receiving", rel);
        const bool pumped = pump([&](const char* data, size_t len) -> bool {
            if (state_.action_cancel_requested(operation_id)) {
                canceled = true;
                return false;
            }
            out.write(data, static_cast<std::streamsize>(len));
            if (!out) { write_ok = false; return false; }
            received += static_cast<int64_t>(len);
            publish_receive_progress("receiving", rel);
            if (state_.action_cancel_requested(operation_id)) {
                canceled = true;
                return false;
            }
            return true;
        });
        out.close();
        // close() flushes the final buffered block; a full-disk ENOSPC can
        // surface only here, so re-check the stream before declaring success.
        if (!out) write_ok = false;
        if (canceled) {
            std::error_code ec;
            fs::remove(slot.temp_path, ec);
            state_.set_last_error("model download canceled: " + id);
            state_.clear_action_progress(operation_id);
            res.status = 499;
            res.set_content(nlohmann::json{{"error", "model receive canceled"},
                                           {"model_id", id}}.dump(),
                            "application/json");
            return;
        }
        if (!pumped || !write_ok) {
            std::error_code ec;
            fs::remove(slot.temp_path, ec);
            state_.clear_action_progress(operation_id);
            res.status = 500;
            res.set_content(R"({"error":"upload interrupted or write failed"})",
                            "application/json");
            return;
        }

        std::string commit_err;
        if (!model_store_->commit_file(slot, &commit_err)) {
            state_.clear_action_progress(operation_id);
            res.status = 500;
            res.set_content(nlohmann::json{{"error", commit_err}}.dump(), "application/json");
            return;
        }
        model_store_->register_model(id, pin);
        if (!evicted.empty())
            MM_INFO("ModelStore: evicted {} model(s) to receive {}", evicted.size(), id);
        state_.clear_action_progress(operation_id);

        res.set_content(nlohmann::json{{"status", "stored"},
                                       {"model_id", id},
                                       {"load_path", model_store_->load_path(id)},
                                       {"stored_path", slot.final_path},
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

            auto detach = engines_.detach_agent(slot_id, agent_id);
            state_.set_slots(engines_.slots());
            if (!detach.ok()) {
                res.status = detach.status == EngineOpStatus::NotFound ? 404 : 500;
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

            auto suspend = engines_.suspend(slot_id);
            state_.set_slots(engines_.slots());
            if (!suspend.ok()) {
                res.status = suspend.status == EngineOpStatus::NotFound ? 404
                           : suspend.status == EngineOpStatus::Busy ? 409
                           : 500;
                res.set_content(nlohmann::json{{"error", suspend.message}}.dump(),
                                "application/json");
                return;
            }

            nlohmann::json resp;
            resp["status"] = "suspended";
            // Field name unchanged on the wire; control reads it.
            resp["kv_cache_path"] = suspend.kv_checkpoint_path;
            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── engine telemetry, proxied ───────────────────────────────────────────
    //
    // The node forwards whatever the descriptor names and never learns that Soma
    // calls it /internal/telemetry. An engine with no telemetry surface —
    // llama.cpp — answers 501 here rather than 404ing halfway down the chain,
    // because "this engine does not publish that" and "no such slot" are
    // different facts and only one of them is the caller's mistake.
    server_->Get("/api/node/engines/:slot/heat", [this](const Request& req, Response& res) {
        const auto ep = engines_.endpoint(req.path_params.at("slot"));
        if (!ep.ok()) { res.status = 404; return; }
        if (ep.heat_path.empty()) {
            res.status = 501;
            res.set_content(
                nlohmann::json{{"error", "this engine publishes no heat map"}}.dump(),
                "application/json");
            return;
        }
        httplib::Client cli(ep.base_url);
        cli.set_read_timeout(10);
        const auto query = req.has_param("resolution")
                               ? "?resolution=" + req.get_param_value("resolution")
                               : std::string{};
        auto upstream = cli.Get((ep.heat_path + query).c_str());
        if (!upstream) { res.status = 502; return; }
        res.status = upstream->status;
        res.set_content(upstream->body, "application/json");
    });

    server_->Get("/api/node/engines/:slot/sequences", [this](const Request& req, Response& res) {
        const auto slot = req.path_params.at("slot");
        if (!engines_.find(slot)) { res.status = 404; return; }
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& s : engines_.sequences(slot)) {
            arr.push_back(nlohmann::json{{"index", s.index},
                                         {"agent_id", s.agent_id},
                                         {"position", s.position},
                                         {"kv_tokens", s.kv_tokens},
                                         {"prefilling", s.prefilling},
                                         {"suspended", s.suspended},
                                         {"determinism", s.determinism}});
        }
        res.set_content(nlohmann::json{{"slot_id", slot}, {"sequences", arr}}.dump(),
                        "application/json");
    });

    server_->Get("/api/node/engines/:slot/telemetry", [this](const Request& req, Response& res) {
        const auto ep = engines_.endpoint(req.path_params.at("slot"));
        if (!ep.ok()) { res.status = 404; return; }
        if (ep.telemetry_path.empty()) {
            res.status = 501;
            res.set_content(
                nlohmann::json{{"error", "this engine publishes no telemetry"}}.dump(),
                "application/json");
            return;
        }
        std::string query;
        if (req.has_param("hz")) query += "?hz=" + req.get_param_value("hz");
        if (req.has_param("resolution")) {
            query += (query.empty() ? "?" : "&");
            query += "resolution=" + req.get_param_value("resolution");
        }

        // Streamed through rather than buffered: the upstream feed never ends,
        // so anything that waits for a complete body waits forever. The engine's
        // own clamp applies — the node forwards `hz` and does not second-guess a
        // ceiling that belongs to the engine.
        res.set_chunked_content_provider(
            "text/event-stream",
            [ep, query](std::size_t, httplib::DataSink& sink) {
                httplib::Client cli(ep.base_url);
                // cpp-httplib reads this as ZERO seconds, not "no limit": set_read_timeout(0, 0)
                // makes every read time out immediately. On an SSE proxy that presents as a
                // clean, empty 200 — headers go out, the inner Get fails at once, and the
                // client sees a stream that ends without a single frame. Measured against a
                // live engine that was streaming perfectly one hop upstream.
                //
                // A long FINITE timeout instead of trying to express infinity: a wedged
                // engine should eventually release the socket rather than pin it forever.
                cli.set_read_timeout(kSseUpstreamReadTimeout);
                const bool ok = cli.Get(
                    (ep.telemetry_path + query).c_str(),
                    [&](const char* data, std::size_t len) {
                        // Returning false when the downstream write fails is what
                        // tears the upstream connection down: without it a
                        // departed client leaves the engine feeding a socket
                        // nobody reads.
                        return sink.write(data, len);
                    });
                if (!ok) sink.done();
                return false; // one pass; the inner Get blocks for the stream's life
            });
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
            std::string mmproj_path = sanitize_path_for_runtime(
                j.value("mmproj_path", std::string{}));
            const bool vision_enabled = j.value("vision_enabled", !mmproj_path.empty());
            std::string agent_id   = j.value("agent_id", std::string{});
            const std::string backend = mm::util::to_lower(mm::util::trim(
                j.value("backend", std::string{"llama-cpp"})));
            if (ray_controller_ != nullptr) {
                const auto ray = ray_controller_->status();
                if (!ray.group_id.empty()) {
                    const auto slots = engines_.slots();
                    const bool sleeping_owned_vllm = std::any_of(
                        slots.begin(), slots.end(), [&](const SlotInfo& slot) {
                            const bool attached = slot.assigned_agent == agent_id ||
                                std::find(slot.agent_ids.begin(), slot.agent_ids.end(),
                                          agent_id) != slot.agent_ids.end();
                            return slot.backend == "vllm" &&
                                   slot.state == SlotState::Suspended && attached;
                        });
                    const bool owned_restore = ray.active() && ray.role == "head" &&
                        backend == "vllm" && !agent_id.empty() &&
                        agent_id == ray.agent_id && sleeping_owned_vllm;
                    if (!owned_restore) {
                        res.status = 409;
                        res.set_content(
                            nlohmann::json{{"error", "node is reserved by Ray group '" +
                                                           ray.group_id + "'"},
                                           {"code", "capacity_pressure"},
                                           {"ray", ray}}
                                .dump(),
                            "application/json");
                        return;
                    }
                }
            }
            const std::string kv_cache_path = j.value("kv_cache_path", std::string{});
            const std::string model_id = j.value("model_id", std::string{});
            const std::string mmproj_model_id =
                j.value("mmproj_model_id", std::string{});
            const bool model_pin = j.value("pin", false);
            RuntimeSettings runtime_settings;
            if (j.contains("runtime_settings")) runtime_settings = j["runtime_settings"].get<RuntimeSettings>();

            // The second of the two copies. Now the same registry lookup, so the
            // pair cannot drift — which is exactly how the first one would have
            // been updated and this one missed.
            if (mm::EngineRegistry::instance().find(backend) == nullptr) {
                res.status = 400;
                const std::string msg = "unsupported node inference backend: " + backend;
                state_.set_last_error(msg);
                res.set_content(
                    nlohmann::json{{"error", msg},
                                   {"supported_backends", mm::EngineRegistry::instance().ids()}}
                        .dump(),
                    "application/json");
                return;
            }

            // Same gating as load-model, and for the same reason: these are
            // llama.cpp's preconditions. Ungated they made a suspended Soma slot
            // unrestorable — the second half of defect D8, in the handler that
            // brings a slot BACK.
            const bool is_llama = (backend == "llama-cpp");

            if (model_path.empty()) {
                res.status = 400;
                state_.set_last_error("model_path required");
                res.set_content(R"({"error":"model_path required"})", "application/json");
                return;
            }

            if (is_llama)
            if (const auto rt = state_.get_llama_runtime(); !node_llama_runtime_ready(rt)) {
                res.status = 503;
                const std::string msg = rt.last_error.empty()
                    ? "llama.cpp runtime is not ready on this node (status=" + rt.status + ")"
                    : "llama.cpp runtime is not ready on this node: " + rt.last_error;
                state_.set_last_error(msg);
                res.set_content(nlohmann::json{{"error", "llama runtime not ready"},
                                               {"detail", msg},
                                               {"llama_runtime", rt}}.dump(),
                                "application/json");
                return;
            }

            if (is_llama && vision_enabled && mmproj_path.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"vision-enabled llama-cpp restore requires mmproj_path"})",
                                "application/json");
                return;
            }
            if (is_llama && !vision_enabled && !mmproj_path.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"llama.cpp mmproj_path requires vision_enabled=true"})",
                                "application/json");
                return;
            }
            if (is_llama && !mmproj_path.empty()) {
                std::error_code ec;
                if (!fs::is_regular_file(mmproj_path, ec)) {
                    res.status = 400;
                    res.set_content(nlohmann::json{{"error", "projector not found on node"},
                                                   {"detail", "projector path not found on this node: " + mmproj_path}}.dump(),
                                    "application/json");
                    return;
                }
            }
            if (const auto* desc = mm::EngineRegistry::instance().find(backend);
                desc != nullptr && desc->validate_model_ref) {
                std::string detail;
                if (!desc->validate_model_ref(model_path, detail)) {
                    res.status = 400;
                    state_.set_last_error(detail);
                    res.set_content(nlohmann::json{{"error", "model not found on node"},
                                                   {"detail", detail},
                                                   {"engine", backend},
                                                   {"model_path", model_path}}.dump(),
                                    "application/json");
                    return;
                }
            }

            EngineLoadRequest restore_req;
            restore_req.model_path = model_path;
            restore_req.mmproj_path = mmproj_path;
            restore_req.settings = runtime_settings;
            restore_req.served_model_name = j.value("served_model_name", model_path);
            if (backend == "vllm") {
                if (engine_manager_ == nullptr) {
                    res.status = 503;
                    res.set_content(R"({"error":"engine manager unavailable"})",
                                    "application/json");
                    return;
                }
                const auto cluster = engine_manager_->current_config();
                const auto* spec = cluster.find("vllm");
                if (spec == nullptr) {
                    res.status = 409;
                    res.set_content(
                        R"({"error":"vllm is not in the applied cluster engine configuration"})",
                        "application/json");
                    return;
                }
                restore_req.vllm = effective_vllm_config(*spec);
            }
            auto slot_id = engines_.restore(backend, restore_req, kv_cache_path, agent_id);
            state_.set_slots(engines_.slots());

            if (slot_id.empty()) {
                nlohmann::json err = {{"error", "failed to restore slot"}};
                auto detail = engines_.last_error();
                const auto code = engine_error_code_for(detail);
                res.status = code == "capacity_pressure" ? 503 : 500;
                err["code"] = code;
                err["engine"] = backend;
                err["llama_runtime"] = state_.get_llama_runtime();
                state_.set_last_error(detail.empty() ? "failed to restore slot" : detail);
                if (!detail.empty()) err["detail"] = detail;
                res.set_content(err.dump(), "application/json");
            } else {
                state_.set_loaded_model(model_path);
                if (!agent_id.empty()) state_.set_active_agent(agent_id);
                state_.set_last_error("");
                if (model_store_) {
                    for (const auto& id : {model_id, mmproj_model_id}) {
                        if (id.empty()) continue;
                        if (model_pin) model_store_->set_pinned(id, true);
                        model_store_->touch(id);
                        if (id == model_id && mmproj_model_id == model_id) break;
                    }
                }
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
                nlohmann::json{{"accepted", true}, {"api_key", new_key},
                               {"hostname", mm::util::hostname()}}.dump(),
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

        auto slots = engines_.slots();

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
            ? EngineSupervisor::Lease{}
            : engines_.acquire(selected_slot_id);

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
        engines_.touch(selected_slot_id);
        state_.set_slots(engines_.slots());
        if (!model_for_slot.empty()) state_.set_loaded_model(model_for_slot);
        state_.start_streaming_text(selected_slot_id, agent_for_slot);
        if (!agent_for_slot.empty()) state_.set_active_agent(agent_for_slot);

        // Fire the LLM call on a background thread
        std::thread([this,
                     infer_req,
                     ctx,
                     slot_lease = std::move(slot_lease)]() mutable {
            EngineClient* client = slot_lease.get();
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
                            // EVERY field the chunk carries, not the first one
                            // that matches.
                            //
                            // This was an if/else-if priority chain, so a chunk
                            // carrying both thinking_delta and delta_content
                            // silently dropped one, and tool_result_json was
                            // never emitted at all — it had no branch. Both are
                            // documented in mantic-mind-integration.md as faults
                            // the rebuild must not inherit, and a dropped delta
                            // presents as the model omitting a word rather than
                            // as a transport bug.
                            if (!c.thinking_delta.empty()) {
                                state_.append_streaming_text("", c.thinking_delta);
                                emit_line("data: " + safe_json_dump(nlohmann::json{
                                    {"type","thinking"},{"content", c.thinking_delta}
                                }) + "\n\n", false);
                            }
                            if (!c.delta_content.empty()) {
                                state_.append_streaming_text(c.delta_content, "");
                                emit_line("data: " + safe_json_dump(nlohmann::json{
                                    {"type","delta"},{"content", c.delta_content}
                                }) + "\n\n", false);
                            }
                            if (c.tool_call_delta) {
                                const auto& tc = *c.tool_call_delta;
                                emit_line("data: " + safe_json_dump(nlohmann::json{
                                    {"type","tool_call"},{"id", tc.id},
                                    {"name", tc.function_name},
                                    {"arguments", tc.arguments_json}
                                }) + "\n\n", false);
                            }
                            if (!c.tool_result_json.empty()) {
                                emit_line("data: " + safe_json_dump(nlohmann::json{
                                    {"type","tool_result"},{"result", c.tool_result_json}
                                }) + "\n\n", false);
                            }
                            // Always last, and always sent: `done` is what closes
                            // the stream, so it cannot be conditional on any of
                            // the above having matched.
                            if (c.is_done) {
                                state_.finish_streaming_text(c.finish_reason, c.tokens_used);
                                emit_line("data: " + safe_json_dump(nlohmann::json{
                                    {"type", "done"},
                                    {"tokens_used", c.tokens_used},
                                    {"finish_reason", c.finish_reason}
                                }) + "\n\n", true);
                            }
                        },
                        [this, emit_line](const EngineError& err) {
                            state_.finish_streaming_text("error", 0);
                            // The structured code rides through to control, which
                            // no longer has to read the message to decide whether
                            // this was capacity.
                            emit_line("data: " + safe_json_dump(nlohmann::json{
                                {"type","error"},
                                {"code", err.code},
                                {"message", err.message}
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
