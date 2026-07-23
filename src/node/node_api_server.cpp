#include "node/node_api_server.hpp"
#include "node/node_state.hpp"
#include "node/slot_manager.hpp"
#include "node/model_store.hpp"
#include "common/http_server.hpp"
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
#include <set>
#include <string>
#include <thread>
#include <utility>

namespace mm {
namespace fs = std::filesystem;

namespace {

// Load paths currently backing a live slot — models the store must never
// evict out from under a running engine.
std::set<std::string> loaded_model_paths(SlotManager& mgr) {
    std::set<std::string> paths;
    for (const auto& s : mgr.get_slot_info()) {
        if (!s.model_path.empty()) paths.insert(s.model_path);
        if (!s.mmproj_path.empty()) paths.insert(s.mmproj_path);
    }
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

NodeApiServer::NodeApiServer(NodeService& service,
                             NodeState& state,
                             SlotManager& slot_mgr,
                             std::string control_url,
                             std::string pairing_key)
    : state_(state)
    , slot_mgr_(slot_mgr)
    , service_(service)
    , control_url_(std::move(control_url))
    , pairing_key_(std::move(pairing_key))
    , server_(std::make_unique<HttpServer>())
{}

NodeApiServer::NodeApiServer(NodeState& state,
                             SlotManager& slot_mgr,
                             std::string control_url,
                             std::string pairing_key)
    : state_(state)
    , slot_mgr_(slot_mgr)
    , owned_service_(std::make_unique<NodeService>(state, slot_mgr))
    , service_(*owned_service_)
    , control_url_(std::move(control_url))
    , pairing_key_(std::move(pairing_key))
    , server_(std::make_unique<HttpServer>())
{}

NodeApiServer::~NodeApiServer() { stop(); }

bool NodeApiServer::listen(uint16_t port) {
    stopping_ = false;
    // Streamed model uploads far exceed cpp-httplib's 100 MB default body cap;
    // raise it so /api/node/models/receive can accept multi-GB model files.
    server_->set_payload_max_length(std::size_t{1} << 42);  // 4 TiB
    register_routes();
    MM_INFO("NodeApiServer listening on port {}", port);
    return server_->listen("0.0.0.0", port);
}

void NodeApiServer::stop() {
    stopping_ = true;
    server_->stop();
    cancel_and_join_inference_tasks();
}

bool NodeApiServer::launch_inference_task(
    const std::shared_ptr<SseInferCtx>& context,
    std::function<void()> work) {
    std::vector<std::thread> completed;
    bool launched = false;

    {
        std::lock_guard<std::mutex> lock(inference_tasks_mutex_);

        // Join completed workers opportunistically so a long-running node does
        // not retain one native thread handle for every inference request.
        auto it = inference_tasks_.begin();
        while (it != inference_tasks_.end()) {
            if (it->finished && it->finished->load()) {
                if (it->worker.joinable()) completed.push_back(std::move(it->worker));
                it = inference_tasks_.erase(it);
            } else {
                ++it;
            }
        }

        if (!stopping_) {
            auto finished = std::make_shared<std::atomic<bool>>(false);
            inference_tasks_.push_back(InferenceTask{});
            auto& task = inference_tasks_.back();
            task.finished = finished;
            task.context = context;
            try {
                task.worker = std::thread(
                    [this, context, work = std::move(work), finished]() mutable {
                        std::string unhandled_error;
                        try {
                            work();
                        } catch (const std::exception& e) {
                            MM_ERROR("NodeApiServer unhandled infer task exception: {}", e.what());
                            unhandled_error =
                                std::string("unhandled infer task exception: ") + e.what();
                        } catch (...) {
                            MM_ERROR("NodeApiServer unhandled infer task exception: unknown");
                            unhandled_error = "unhandled infer task exception: unknown";
                        }
                        if (!unhandled_error.empty()) {
                            state_.set_last_error(unhandled_error);
                            bool terminal_emitted = false;
                            try {
                                std::lock_guard<std::mutex> context_lock(context->mx);
                                if (!context->done) {
                                    if (!context->canceled) {
                                        context->lines.push_back(
                                            "data: " + safe_json_dump(nlohmann::json{
                                                {"type", "error"},
                                                {"message", unhandled_error}
                                            }) + "\n\n");
                                        terminal_emitted = true;
                                    }
                                    context->done = true;
                                    context->cv.notify_all();
                                }
                            } catch (...) {
                                context->canceled = true;
                                context->cv.notify_all();
                            }
                            state_.finish_streaming_text(
                                terminal_emitted ? "error" : "canceled", 0);
                        }
                        finished->store(true);
                    });
                launched = true;
            } catch (...) {
                inference_tasks_.pop_back();
            }
        }
    }

    for (auto& worker : completed) {
        if (worker.joinable()) worker.join();
    }
    return launched;
}

void NodeApiServer::cancel_and_join_inference_tasks() {
    std::vector<std::thread> workers;
    std::vector<std::shared_ptr<SseInferCtx>> contexts;

    {
        std::lock_guard<std::mutex> lock(inference_tasks_mutex_);
        workers.reserve(inference_tasks_.size());
        contexts.reserve(inference_tasks_.size());
        for (auto& task : inference_tasks_) {
            if (auto context = task.context.lock()) contexts.push_back(std::move(context));
            if (task.worker.joinable()) workers.push_back(std::move(task.worker));
        }
        inference_tasks_.clear();
    }

    for (const auto& context : contexts) {
        context->canceled = true;
        context->cv.notify_all();
    }
    for (auto& worker : workers) {
        if (worker.joinable()) worker.join();
    }
}

void NodeApiServer::set_runtime_logs_provider(RuntimeLogsProvider provider) {
    service_.set_runtime_logs_provider(std::move(provider));
}

void NodeApiServer::set_remember_api_key_callback(RememberApiKeyCallback callback) {
    remember_api_key_cb_ = std::move(callback);
}

void NodeApiServer::set_llama_provision_callback(LlamaProvisionCallback callback) {
    service_.set_llama_provision_callback(std::move(callback));
}

void NodeApiServer::set_llama_update_callback(LlamaUpdateCallback callback) {
    service_.set_llama_update_callback(std::move(callback));
}

void NodeApiServer::set_llama_switch_callback(LlamaSwitchCallback callback) {
    service_.set_llama_switch_callback(std::move(callback));
}

void NodeApiServer::set_llama_check_update_callback(LlamaCheckUpdateCallback callback) {
    service_.set_llama_check_update_callback(std::move(callback));
}

void NodeApiServer::set_llama_diagnose_callback(LlamaDiagnoseCallback callback) {
    service_.set_llama_diagnose_callback(std::move(callback));
}

void NodeApiServer::set_llama_recovery_callback(LlamaRecoveryCallback callback) {
    service_.set_llama_recovery_callback(std::move(callback));
}

void NodeApiServer::set_model_store(ModelStore* store) {
    model_store_ = store;
    service_.set_model_store(store);
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
        const auto snapshot = service_.snapshot();
        nlohmann::json j = snapshot.health;
        j["status"] = "ok";
        res.set_content(j.dump(), "application/json");
    });

    // ── GET /api/node/status ──────────────────────────────────────────────────
    server_->Get("/api/node/status", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        const auto snapshot = service_.snapshot();
        nlohmann::json j;
        j["node_id"]       = snapshot.node_id;
        j["hostname"]      = snapshot.hostname;
        j["slots"]         = snapshot.slots;
        if (model_store_) {
            nlohmann::json managed = nlohmann::json::array();
            for (const auto& m : snapshot.managed_models)
                managed.push_back({{"id", m.id},
                                   {"size_bytes", m.size_bytes},
                                   {"pinned", m.pinned},
                                   {"last_used_ms", m.last_used_ms}});
            j["managed_models"] = managed;
            j["model_cache_free_bytes"] = snapshot.model_cache_free_bytes;
        }
        j["disk_free_mb"]  = snapshot.disk_free_mb;
        j["health"]        = snapshot.health;
        j["capabilities"]  = snapshot.capabilities;
        j["max_slots"]     = snapshot.max_slots;
        j["slot_in_use"]   = snapshot.slot_in_use;
        j["slot_available"] = snapshot.slot_available;
        j["slot_ready"]    = snapshot.slot_ready;
        j["slot_loading"]  = snapshot.slot_loading;
        j["slot_suspending"] = snapshot.slot_suspending;
        j["slot_suspended"] = snapshot.slot_suspended;
        j["slot_error"]    = snapshot.slot_error;
        j["llama_server_path"] = snapshot.llama_server_path;
        j["llama_runtime"] = snapshot.llama_runtime;
        j["action_progress"] = snapshot.action_progress;
        j["loaded_model"]  = snapshot.loaded_model;
        j["active_agent"]  = snapshot.active_agent;

        res.set_content(j.dump(), "application/json");
    });

    // ── GET /api/node/logs ───────────────────────────────────────────────────
    server_->Post("/api/node/actions/cancel", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401;
            return;
        }
        const bool accepted = service_.cancel_action();
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

        const std::vector<std::string> lines = service_.runtime_logs(tail);
        res.set_content(nlohmann::json{{"lines", lines}}.dump(), "application/json");
    });

    // ── POST /api/node/load-model ─────────────────────────────────────────────
    server_->Get("/api/node/runtime/llama", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        res.set_content(nlohmann::json{{"llama_runtime", service_.snapshot().llama_runtime}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/runtime/llama/provision", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
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
        const auto result = want_update
            ? service_.update_llama_runtime(accelerator)
            : service_.provision_llama_runtime();
        if (result.status == NodeServiceStatus::Unavailable) {
            res.status = 501;
            res.set_content(want_update
                                ? R"({"error":"llama.cpp update is not configured"})"
                                : R"({"error":"llama.cpp provisioning is not configured"})",
                            "application/json");
            return;
        }
        if (result.status == NodeServiceStatus::Failed) res.status = 500;
        else if (result.status == NodeServiceStatus::Conflict) res.status = 409;
        else res.status = 200;
        res.set_content(nlohmann::json{{"llama_runtime", result.runtime}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/runtime/llama/check-update", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        const auto result = service_.check_llama_runtime_update();
        if (result.status == NodeServiceStatus::Unavailable) {
            res.status = 501;
            res.set_content(R"({"error":"llama.cpp update checking is not configured"})",
                            "application/json");
            return;
        }
        res.set_content(nlohmann::json{{"llama_runtime", result.runtime}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/runtime/llama/switch", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
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
        const auto result = service_.switch_llama_runtime(variant);
        if (result.status == NodeServiceStatus::Unavailable) {
            res.status = 501;
            res.set_content(R"({"error":"llama.cpp engine switching is not configured"})",
                            "application/json");
            return;
        }
        if (result.status == NodeServiceStatus::Failed) res.status = 500;
        else if (result.status == NodeServiceStatus::Conflict) res.status = 409;
        else res.status = 200;
        res.set_content(nlohmann::json{{"llama_runtime", result.runtime}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/runtime/llama/diagnose", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        const auto result = service_.diagnose_llama_runtime();
        if (result.status == NodeServiceStatus::Unavailable) {
            res.status = 501;
            res.set_content(R"({"error":"llama.cpp diagnostics are not configured"})",
                            "application/json");
            return;
        }
        res.set_content(nlohmann::json{{"llama_runtime", result.runtime}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/runtime/llama/recover", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
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
        const auto result = service_.recover_llama_runtime(action, variant);
        if (result.status == NodeServiceStatus::Unavailable) {
            res.status = 501;
            res.set_content(R"({"error":"llama.cpp recovery is not configured"})",
                            "application/json");
            return;
        }
        if (result.status == NodeServiceStatus::Failed) res.status = 500;
        else if (result.status == NodeServiceStatus::Conflict) res.status = 409;
        else res.status = 200;
        res.set_content(nlohmann::json{{"llama_runtime", result.runtime}}.dump(),
                        "application/json");
    });

    server_->Post("/api/node/load-model", [this](const Request& req, Response& res) {
        if (!check_auth(req.get_header_value("Authorization"))) {
            res.status = 401; return;
        }
        try {
            const auto j = nlohmann::json::parse(req.body);
            NodeLoadModelRequest request;
            request.model_path = j.value("model_path", std::string{});
            request.mmproj_path = j.value("mmproj_path", std::string{});
            request.vision_enabled =
                j.value("vision_enabled", !request.mmproj_path.empty());
            request.agent_id = j.value("agent_id", std::string{});
            request.model_id = j.value("model_id", std::string{});
            request.mmproj_model_id = j.value("mmproj_model_id", std::string{});
            request.pin = j.value("pin", false);
            request.backend = j.value("backend", std::string{"llama-cpp"});
            if (j.contains("runtime_settings")) {
                request.runtime_settings = j["runtime_settings"].get<RuntimeSettings>();
            }

            if (request.model_path.empty()) {
                res.status = 400;
                state_.set_last_error("model_path required");
                res.set_content(
                    R"({"error":"model_path required","code":"invalid_argument"})",
                    "application/json");
                return;
            }

            const auto result = service_.load_model(request);
            if (!result.ok()) {
                if (result.status == NodeServiceStatus::UnsupportedBackend) {
                    res.status = 400;
                    res.set_content(nlohmann::json{{"error", result.error},
                                                   {"code", "unsupported_backend"},
                                                   {"supported_backends", {"llama-cpp"}}}.dump(),
                                    "application/json");
                } else if (result.status == NodeServiceStatus::RuntimeUnavailable) {
                    res.status = 503;
                    res.set_content(nlohmann::json{
                        {"error", "llama runtime not ready"},
                        {"code", "runtime_unavailable"},
                        {"detail", result.error},
                        {"llama_runtime", service_.snapshot().llama_runtime}
                    }.dump(), "application/json");
                } else if (result.status == NodeServiceStatus::ModelNotFound) {
                    res.status = 400;
                    const std::string model_path = sanitize_path_for_runtime(request.model_path);
                    if (result.error.rfind("projector path", 0) == 0) {
                        res.set_content(nlohmann::json{
                            {"error", "projector not found on node"},
                            {"code", "model_not_found"},
                            {"detail", result.error}
                        }.dump(), "application/json");
                    } else {
                        const std::string detail = mm::util::is_hf_repo_id(model_path)
                            ? "llama.cpp requires a node-local GGUF file; Hugging Face repository IDs are not loadable"
                            : result.error;
                        res.set_content(nlohmann::json{
                            {"error", "model not found on node"},
                            {"code", "model_not_found"},
                            {"detail", detail},
                            {"model_path", model_path}
                        }.dump(), "application/json");
                    }
                } else if (result.status == NodeServiceStatus::InvalidArgument) {
                    res.status = 400;
                    res.set_content(nlohmann::json{{"error", result.error},
                                                   {"code", "invalid_argument"}}.dump(),
                                    "application/json");
                } else {
                    res.status = 500;
                    const auto snapshot = service_.snapshot();
                    nlohmann::json error = {
                        {"error", "failed to load model"},
                        {"code", to_string(result.status)},
                        {"llama_server_path", snapshot.llama_server_path},
                        {"llama_runtime", snapshot.llama_runtime},
                        {"model_path", sanitize_path_for_runtime(request.model_path)}
                    };
                    if (!result.error.empty()) error["detail"] = result.error;
                    res.set_content(error.dump(), "application/json");
                }
                return;
            }
            res.set_content(nlohmann::json{
                {"status", "loaded"},
                {"slot_id", result.slot_id},
                {"effective_ctx_size", result.effective_ctx_size},
                {"model_path", result.model_path}
            }.dump(), "application/json");
            return;

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

            const auto unload = service_.unload_model({slot_id, false});
            if (!unload.ok()) {
                res.status = unload.status == SlotOperationStatus::NotFound ? 404
                           : unload.status == SlotOperationStatus::Busy ? 409
                           : 500;
                res.set_content(nlohmann::json{{"error", unload.message}}.dump(),
                                "application/json");
                return;
            }
            res.set_content(R"({"status":"unloaded"})", "application/json");
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

        const auto in_use = loaded_model_paths(slot_mgr_);
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

            const auto detach = service_.detach_agent({slot_id, agent_id});
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

            const auto suspend = service_.suspend_slot({slot_id});
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
            const auto j = nlohmann::json::parse(req.body);
            NodeRestoreModelRequest request;
            request.model_path = j.value("model_path", std::string{});
            request.mmproj_path = j.value("mmproj_path", std::string{});
            request.vision_enabled =
                j.value("vision_enabled", !request.mmproj_path.empty());
            request.agent_id = j.value("agent_id", std::string{});
            request.backend = j.value("backend", std::string{"llama-cpp"});
            request.kv_cache_path = j.value("kv_cache_path", std::string{});
            request.model_id = j.value("model_id", std::string{});
            request.mmproj_model_id = j.value("mmproj_model_id", std::string{});
            request.pin = j.value("pin", false);
            if (j.contains("runtime_settings")) {
                request.runtime_settings = j["runtime_settings"].get<RuntimeSettings>();
            }

            if (request.model_path.empty()) {
                res.status = 400;
                state_.set_last_error("model_path required");
                res.set_content(R"({"error":"model_path required"})", "application/json");
                return;
            }

            const auto result = service_.restore_slot(request);
            if (!result.ok()) {
                if (result.status == NodeServiceStatus::UnsupportedBackend) {
                    res.status = 400;
                    res.set_content(nlohmann::json{{"error", result.error},
                                                   {"supported_backends", {"llama-cpp"}}}.dump(),
                                    "application/json");
                } else if (result.status == NodeServiceStatus::RuntimeUnavailable) {
                    res.status = 503;
                    res.set_content(nlohmann::json{
                        {"error", "llama runtime not ready"},
                        {"detail", result.error},
                        {"llama_runtime", service_.snapshot().llama_runtime}
                    }.dump(), "application/json");
                } else if (result.status == NodeServiceStatus::ModelNotFound) {
                    res.status = 400;
                    const std::string model_path = sanitize_path_for_runtime(request.model_path);
                    if (result.error.rfind("projector path", 0) == 0) {
                        res.set_content(nlohmann::json{
                            {"error", "projector not found on node"},
                            {"detail", result.error}
                        }.dump(), "application/json");
                    } else {
                        const std::string detail = mm::util::is_hf_repo_id(model_path)
                            ? "llama.cpp requires a node-local GGUF file; Hugging Face repository IDs are not loadable"
                            : "model file not found on this node during restore: " + model_path;
                        res.set_content(nlohmann::json{
                            {"error", "model not found on node"},
                            {"detail", detail},
                            {"model_path", model_path}
                        }.dump(), "application/json");
                    }
                } else if (result.status == NodeServiceStatus::InvalidArgument) {
                    res.status = 400;
                    std::string error = result.error;
                    if (error == "vision-enabled llama-cpp load requires mmproj_path") {
                        error = "vision-enabled llama-cpp restore requires mmproj_path";
                    }
                    res.set_content(nlohmann::json{{"error", error}}.dump(),
                                    "application/json");
                } else {
                    res.status = 500;
                    const auto snapshot = service_.snapshot();
                    nlohmann::json error = {
                        {"error", "failed to restore slot"},
                        {"llama_server_path", snapshot.llama_server_path},
                        {"llama_runtime", snapshot.llama_runtime}
                    };
                    if (!result.error.empty()) error["detail"] = result.error;
                    res.set_content(error.dump(), "application/json");
                }
                return;
            }
            res.set_content(nlohmann::json{{"status", "restored"},
                                           {"slot_id", result.slot_id}}.dump(),
                            "application/json");
            return;

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

        const auto slots = service_.snapshot().slots;

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

        const bool selected_ready = std::any_of(
            slots.begin(), slots.end(), [&](const SlotInfo& slot) {
                return slot.id == selected_slot_id && slot.state == SlotState::Ready;
            });
        if (!selected_ready) {
            res.status = 503;
            res.set_content(R"({"error":"no ready slot available"})", "application/json");
            return;
        }

        // Shared context between LLM worker and SSE provider
        auto ctx = std::make_shared<SseInferCtx>();

        const bool launched = launch_inference_task(
            ctx,
            [this, infer_req, selected_slot_id, ctx]() {
            auto emit_line = [ctx](const std::string& payload, bool terminal) -> bool {
                std::lock_guard<std::mutex> lk(ctx->mx);
                // The first typed terminal event wins. The runtime transport
                // can report an error and subsequently flush a done chunk; do
                // not expose both to the node protocol.
                if (ctx->canceled || ctx->done) return false;
                ctx->lines.push_back(payload);
                if (terminal) ctx->done = true;
                ctx->cv.notify_all();
                return true;
            };

            const NodeInferRequest service_request{
                infer_req,
                selected_slot_id,
                [ctx] { return ctx->canceled.load(); }
            };
            const auto inference_result = service_.infer(
                service_request,
                [emit_line](const InferenceChunk& chunk) {
                    if (!chunk.thinking_delta.empty()) {
                        if (!emit_line("data: " + safe_json_dump({
                                {"type", "thinking"},
                                {"content", chunk.thinking_delta}}) + "\n\n",
                                false)) return;
                    }
                    if (!chunk.delta_content.empty()) {
                        if (!emit_line("data: " + safe_json_dump({
                                {"type", "delta"},
                                {"content", chunk.delta_content}}) + "\n\n",
                                false)) return;
                    }
                    if (chunk.tool_call_delta) {
                        const auto& tool_call = *chunk.tool_call_delta;
                        if (!emit_line("data: " + safe_json_dump({
                                {"type", "tool_call"},
                                {"id", tool_call.id},
                                {"name", tool_call.function_name},
                                {"arguments", tool_call.arguments_json}}) +
                                "\n\n", false)) return;
                    }
                    if (!chunk.tool_result_json.empty()) {
                        if (!emit_line("data: " + safe_json_dump({
                                {"type", "tool_result"},
                                {"content", chunk.tool_result_json}}) + "\n\n",
                                false)) return;
                    }
                    // A backend may combine its last data with the terminal
                    // accounting chunk. Emit every field above, then exactly
                    // one terminal result instead of losing it to an else-if.
                    if (chunk.is_done) {
                        emit_line("data: " + safe_json_dump({
                            {"type", "done"},
                            {"tokens_used", chunk.tokens_used},
                            {"finish_reason", chunk.finish_reason}}) + "\n\n",
                            true);
                    }
                },
                [emit_line, stream = infer_req.stream](const std::string& error) {
                    emit_line("data: " + safe_json_dump(nlohmann::json{
                        {"type", "error"},
                        {"message", stream ? error : "non-stream inference failed"}
                    }) + "\n\n", true);
                });

            if (!inference_result.ok() &&
                inference_result.status != NodeServiceStatus::Canceled) {
                const std::string message = infer_req.stream
                    ? (inference_result.error.empty()
                           ? "inference failed"
                           : inference_result.error)
                    : "non-stream inference failed";
                emit_line("data: " + safe_json_dump(nlohmann::json{
                    {"type", "error"}, {"message", message}
                }) + "\n\n", true);
            }

            const std::string missing_terminal_payload =
                "data: " + safe_json_dump(nlohmann::json{
                    {"type", "error"},
                    {"message", "inference ended without a terminal result"}
                }) + "\n\n";
            bool fallback_terminal = false;
            bool canceled_without_terminal = false;
            {
                std::lock_guard<std::mutex> lk(ctx->mx);
                if (!ctx->done) {
                    canceled_without_terminal = ctx->canceled;
                    if (!canceled_without_terminal) {
                        ctx->lines.push_back(missing_terminal_payload);
                        fallback_terminal = true;
                    }
                    ctx->done = true;
                    ctx->cv.notify_all();
                }
            }
            if (fallback_terminal) {
                state_.finish_streaming_text("error", 0);
            } else if (canceled_without_terminal) {
                state_.finish_streaming_text("canceled", 0);
            }
        });

        if (!launched) {
            state_.finish_streaming_text("canceled", 0);
            res.status = 503;
            res.set_content(R"({"error":"node is stopping"})", "application/json");
            return;
        }

        // Stream SSE to client via chunked content provider. A failed sink or
        // resource release marks the context canceled; NodeService propagates
        // that flag to the active llama-server request.
        res.set_chunked_content_provider("text/event-stream",
            [ctx](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::unique_lock<std::mutex> lk(ctx->mx);
                ctx->cv.wait(lk, [&] {
                    return !ctx->lines.empty() || ctx->done || ctx->canceled;
                });

                if (ctx->canceled) return false;

                while (!ctx->lines.empty()) {
                    std::string payload = std::move(ctx->lines.front());
                    ctx->lines.pop_front();
                    lk.unlock();
                    if (!sink.write(payload.data(), payload.size())) {
                        ctx->canceled = true;
                        ctx->cv.notify_all();
                        return false;
                    }
                    lk.lock();
                    if (ctx->canceled) return false;
                }

                if (ctx->done && ctx->lines.empty()) {
                    lk.unlock();
                    const std::string fin = "data: [DONE]\n\n";
                    if (!sink.write(fin.data(), fin.size())) {
                        ctx->canceled = true;
                        ctx->cv.notify_all();
                        return false;
                    }
                    sink.done();
                    return true;
                }
                return true;
            },
            [ctx](bool success) {
                if (!success) {
                    ctx->canceled = true;
                    ctx->cv.notify_all();
                }
            }
        );
    });
}

} // namespace mm
