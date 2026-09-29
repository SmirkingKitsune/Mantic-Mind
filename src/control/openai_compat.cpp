// Mantic-Mind — the OpenAI-compatible listener (:9091), stateless.
//
// A client that already owns its conversation — its history, its tools, its
// sampling — sends an ordinary chat completion naming a served model, and the
// request reaches the engine placed for that model UNCHANGED. Nothing is
// stored: no user, no conversation, no attachment. That is what an OpenAI
// client means by /v1/chat/completions, and what a harness that must own its
// workers' context needs (Code-Aid's ADR-001 rejected Mantic-Mind for lacking
// exactly this).
//
// This listener used to list AGENTS as models and flatten the client's whole
// message array into one user turn in that agent's own conversation, with the
// agent's history and memories prepended — and it dropped tool calls on the way
// out. None of that survives.
//
// The only rewrite is the model name: in, from the served-model id to what the
// engine or remote provider calls it; out, back to the served-model id.

#include "control/control_api_server.hpp"

#include "common/engine_capabilities.hpp"
#include "common/http_client.hpp"
#include "common/http_server.hpp"
#include "common/logger.hpp"
#include "common/sse_infer_ctx.hpp"
#include "common/util.hpp"
#include "control/conversation_queue.hpp"
#include "control/engine_config_store.hpp"
#include "control/node_registry.hpp"
#include "control/placement_scheduler.hpp"
#include "control/served_model_catalog.hpp"
#include "control/token_binding.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <memory>
#include <string>

namespace mm {

namespace {

void set_openai_error(httplib::Response& res,
                      int status,
                      const std::string& message,
                      const std::string& type = "invalid_request_error",
                      const std::string& code = {}) {
    res.status = status;
    res.set_content(nlohmann::json{{"error",
                                    {{"message", message},
                                     {"type", type},
                                     {"param", nullptr},
                                     {"code", code.empty() ? nlohmann::json(nullptr)
                                                           : nlohmann::json(code)}}}}
                        .dump(),
                    "application/json");
}

bool messages_have_images(const nlohmann::json& messages) {
    for (const auto& m : messages) {
        if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
        for (const auto& part : m["content"]) {
            const auto type = part.is_object() ? part.value("type", std::string{}) : std::string{};
            if (type == "image_url" || type == "input_image") return true;
        }
    }
    return false;
}

/// A history carrying tool calls needs an engine that can re-render them —
/// Soma refuses those with a 422 just as it refuses `tools`.
bool history_has_tool_calls(const nlohmann::json& messages) {
    for (const auto& m : messages) {
        if (!m.is_object()) continue;
        if (m.value("role", std::string{}) == "tool") return true;
        if (m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty())
            return true;
    }
    return false;
}

/// `payload` with its `model` field set to `model_id`; unchanged when it is
/// not a JSON object (`[DONE]`) or carries no model.
std::string with_model(const std::string& payload, const std::string& model_id) {
    auto j = nlohmann::json::parse(payload, nullptr, /*allow_exceptions=*/false);
    if (!j.is_object() || !j.contains("model")) return payload;
    j["model"] = model_id;
    return j.dump();
}

nlohmann::json model_entry(const CatalogEntry& e) {
    return {{"id", e.model.id},
            {"object", "model"},
            {"created", e.model.created_at_ms / 1000},
            {"owned_by", "mantic-mind"},
            {"metadata",
             {{"display_name", e.model.display_name},
              {"source", to_string(e.source)},
              {"inference_backend", e.model.inference_backend},
              {"vision", e.model.vision_settings.enabled},
              {"tool_calling", e.model.tool_calling},
              {"context_window", e.model.runtime_settings.ctx_size}}}};
}

} // namespace

bool ControlApiServer::authorize_openai_compat_request(const httplib::Request& req,
                                                       httplib::Response& res) const {
    // The SAME authorizer as the main port, so a scoped token means the same
    // thing on both. This listener used to honour only the legacy flat token,
    // and was wide open whenever that was unset — even with scoped tokens
    // issued, which is the opposite of what an operator issuing them meant.
    //
    // Authentication and scope both: authorize() only establishes WHO is
    // asking, and stopping there would let a read-only token send completions.
    const auto result = scopes_.authorize(req);
    if (!result.allowed) {
        set_openai_error(res, result.status == 0 ? 401 : result.status,
                         result.error.empty() ? "unauthorized" : result.error,
                         result.status == 403 ? "permission_error" : "authentication_error");
        return false;
    }
    const Scope required = scopes_.required_scope(req.method, req.path);
    if (!scope_permits(result.granted, required)) {
        set_openai_error(res, 403,
                         std::string("insufficient scope: requires ") + to_string(required) +
                             ", token grants " + format_scopes(result.granted),
                         "permission_error", "insufficient_scope");
        return false;
    }
    // And the same user binding: one policy, whichever port a token is shown to.
    const auto binding = binding_permits(result.bound_user_id, req.method, req.path);
    if (!binding.allowed) {
        set_openai_error(res, 403, binding.error, "permission_error", "user_binding");
        return false;
    }
    return true;
}

void ControlApiServer::register_openai_compat_routes() {
    using httplib::Request;
    using httplib::Response;

    // Only what can be served now: an OpenAI client lists models to pick one.
    openai_server_->Get("/v1/models", [this](const Request&, Response& res) {
        nlohmann::json data = nlohmann::json::array();
        if (served_models_ != nullptr) {
            for (const auto& e : served_models_->list())
                if (e.available) data.push_back(model_entry(e));
        }
        res.set_content(nlohmann::json{{"object", "list"}, {"data", data}}.dump(),
                        "application/json");
    });

    // `:id`, not `:model`: the scope table is keyed by the exact pattern, and
    // this is the same GET /v1/models/:id the main port scopes as `read`.
    openai_server_->Get("/v1/models/:id", [this](const Request& req, Response& res) {
        const auto entry = served_models_ != nullptr
                               ? served_models_->find(req.path_params.at("id"))
                               : std::nullopt;
        if (!entry) {
            set_openai_error(res, 404, "model not found: " + req.path_params.at("id"),
                             "invalid_request_error", "model_not_found");
            return;
        }
        res.set_content(model_entry(*entry).dump(), "application/json");
    });

    openai_server_->Post("/v1/chat/completions", [this](const Request& req, Response& res) {
        if (served_models_ == nullptr) {
            set_openai_error(res, 503, "the served-model catalog is not available",
                             "server_error");
            return;
        }
        nlohmann::json body;
        try {
            body = req.body.empty() ? nlohmann::json::object() : nlohmann::json::parse(req.body);
            if (!body.is_object()) throw std::invalid_argument("request body must be a JSON object");
        } catch (const std::exception& e) {
            set_openai_error(res, 400, e.what());
            return;
        }
        const std::string model_id = util::trim(body.value("model", std::string{}));
        if (model_id.empty()) {
            set_openai_error(res, 400, "model is required");
            return;
        }
        if (!body.contains("messages") || !body["messages"].is_array() ||
            body["messages"].empty()) {
            set_openai_error(res, 400, "messages must be a non-empty array");
            return;
        }
        // Several choices would have to be fanned out and merged per engine,
        // and silently returning one would look like the engine ignored `n`.
        if (body.contains("n") && body["n"].is_number_integer() && body["n"].get<int>() > 1) {
            set_openai_error(res, 400, "n > 1 is not supported; send separate requests");
            return;
        }

        const auto entry = served_models_->find(model_id);
        if (!entry) {
            set_openai_error(res, 404,
                             "model not found: " + model_id + "; GET /v1/models lists what is served",
                             "invalid_request_error", "model_not_found");
            return;
        }
        if (!entry->available) {
            set_openai_error(res, 409, entry->unavailable_reason, "invalid_request_error",
                             "model_unavailable");
            return;
        }
        const ServedModel model = entry->model;
        const bool stream = body.value("stream", false);
        const bool wants_images = messages_have_images(body["messages"]);
        const bool wants_tools =
            (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty()) ||
            history_has_tool_calls(body["messages"]);

        // Refused BEFORE anything is sent: an engine that cannot see an image
        // answers as though it had, and one that cannot render tools either
        // 422s or answers in prose about functions it never saw.
        const auto refusal = [&](const std::string& engine_id,
                                 const std::string& reason) -> std::string {
            if (wants_images) {
                auto r = image_refusal_for(model.vision_settings.enabled, engine_id, reason);
                if (!r.empty()) return r;
            }
            if (wants_tools) {
                if (!model.tool_calling) return "this model is not configured to call tools";
                if (!engine_id.empty() && !engine_supports_tools(engine_id, vllm_tool_choice()))
                    return "the engine serving this model cannot call tools: " + reason;
            }
            return {};
        };

        std::optional<PlacementScheduler::Lease> lease;
        std::string target_url;
        std::string target_path;
        std::string bearer;
        if (model.node_local()) {
            const auto routing = scheduler_.resolve_backend_for(model);
            if (auto r = refusal(routing.engine_id, routing.reason); !r.empty()) {
                set_openai_error(res, 422, r);
                return;
            }
            auto outcome = scheduler_.acquire(model, chat_queue_wait());
            if (!outcome.ok()) {
                set_openai_error(res, placement_failure_retryable(outcome.failure) ? 503 : 409,
                                 outcome.error, "server_error", to_string(outcome.failure));
                return;
            }
            lease = std::move(*outcome.lease);
            // The engine actually placed can be the backup, so ask again.
            if (auto r = refusal(lease->engine_id(), "placed on " + lease->engine_id());
                !r.empty()) {
                set_openai_error(res, 422, r);
                return;
            }
            NodeInfo node;
            try {
                node = registry_.get_node(lease->node_id());
            } catch (const std::exception& e) {
                set_openai_error(res, 503, std::string("node lookup failed: ") + e.what(),
                                 "server_error");
                return;
            }
            target_url = node.url;
            target_path = "/api/node/engines/" + lease->slot_id() + "/chat/completions";
            bearer = node.api_key;
            body["model"] = lease->request_model();
        } else {
            if (auto r = refusal({}, {}); !r.empty()) {
                set_openai_error(res, 422, r);
                return;
            }
            target_url = model.api_settings.base_url;
            target_path = model.api_settings.chat_completions_path;
            bearer = resolve_api_key(model.api_settings);
            body["model"] = model.model_path;
        }

        const auto make_client = [target_url, bearer] {
            HttpClient client(target_url);
            if (!bearer.empty()) client.set_bearer_token(bearer);
            // A generation can run for minutes; a connection should not.
            client.set_timeouts(10, 3600, 60);
            return client;
        };

        if (!stream) {
            auto client = make_client();
            const auto upstream = client.post(target_path, body);
            lease.reset();
            if (upstream.status == 0) {
                set_openai_error(res, 502, "the model's engine did not answer", "server_error");
                return;
            }
            res.status = upstream.status;
            res.set_content(upstream.ok() ? with_model(upstream.body, model.id) : upstream.body,
                            "application/json");
            return;
        }

        // ── streaming ────────────────────────────────────────────────────────
        //
        // The upstream request runs on a pool worker, holding the placement,
        // while the provider below relays its frames — rewritten only in
        // `model`. The handler waits for the first frame first, so a request
        // the engine refuses outright is answered with the engine's own status
        // rather than a 200 wrapped around an error.
        struct StreamCtx {
            SseInferCtx sse;
            bool started = false;
            bool saw_done = false;
            int status = 0;
            std::string error_body;
        };
        auto ctx = std::make_shared<StreamCtx>();
        auto held = std::make_shared<std::optional<PlacementScheduler::Lease>>(std::move(lease));
        const std::string served_id = model.id;
        const auto finish = [ctx] {
            std::lock_guard<std::mutex> lk(ctx->sse.mx);
            ctx->sse.done = true;
            ctx->sse.cv.notify_all();
        };
        const bool queued = openai_queue().enqueue(
            "openai:" + util::generate_uuid(),
            [ctx, held, body, target_path, served_id, make_client, finish] {
                auto client = make_client();
                int status = 0;
                std::string error_body;
                client.stream_post(
                    target_path, body,
                    [ctx, served_id](const std::string& payload) {
                        std::lock_guard<std::mutex> lk(ctx->sse.mx);
                        if (payload == "[DONE]") ctx->saw_done = true;
                        ctx->sse.lines.push_back("data: " + with_model(payload, served_id) + "\n\n");
                        ctx->started = true;
                        ctx->sse.cv.notify_all();
                        return true;
                    },
                    &status, &error_body);
                held->reset(); // the placement is free before the client hears the end
                {
                    std::lock_guard<std::mutex> lk(ctx->sse.mx);
                    ctx->status = status;
                    ctx->error_body = std::move(error_body);
                }
                finish();
            },
            finish);
        if (!queued) {
            set_openai_error(res, 503, "the server is shutting down", "server_error");
            return;
        }

        std::unique_lock<std::mutex> lk(ctx->sse.mx);
        ctx->sse.cv.wait(lk, [&] { return ctx->started || ctx->sse.done; });
        if (!ctx->started) {
            res.status = ctx->status != 0 ? ctx->status : 502;
            res.set_content(ctx->error_body.empty()
                                ? std::string(R"({"error":{"message":"the model's engine did not answer","type":"server_error"}})")
                                : ctx->error_body,
                            "application/json");
            return;
        }
        lk.unlock();

        res.set_chunked_content_provider(
            "text/event-stream", [ctx](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::unique_lock<std::mutex> guard(ctx->sse.mx);
                ctx->sse.cv.wait(guard, [&] { return !ctx->sse.lines.empty() || ctx->sse.done; });
                while (!ctx->sse.lines.empty()) {
                    std::string payload = std::move(ctx->sse.lines.front());
                    ctx->sse.lines.pop_front();
                    guard.unlock();
                    if (!sink.write(payload.data(), payload.size())) return false;
                    guard.lock();
                }
                if (ctx->sse.done && ctx->sse.lines.empty()) {
                    const bool saw_done = ctx->saw_done;
                    guard.unlock();
                    if (!saw_done) {
                        static const std::string fin = "data: [DONE]\n\n";
                        sink.write(fin.data(), fin.size());
                    }
                    sink.done();
                }
                return true;
            });
    });
}

bool ControlApiServer::vllm_tool_choice() const {
    if (engine_config_ == nullptr || !engine_config_->configured()) return false;
    const auto cluster = engine_config_->get();
    const auto* spec = cluster.find("vllm");
    return spec != nullptr && effective_vllm_config(*spec).enable_auto_tool_choice;
}

ConversationQueue& ControlApiServer::openai_queue() {
    std::call_once(openai_queue_once_, [this] {
        // A bounded pool: each streaming completion holds a worker for as long
        // as it streams, and a flood of them should queue, not spawn threads.
        openai_queue_ = std::make_unique<ConversationQueue>(openai_workers_);
    });
    return *openai_queue_;
}

} // namespace mm
