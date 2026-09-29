#include "common/engine_capabilities.hpp"
#include "control/control_api_server.hpp"
#include "control/model_registry.hpp"
#include "control/node_registry.hpp"
#include "control/conversation_queue.hpp"
#include "control/placement_scheduler.hpp"
#include "control/served_model_catalog.hpp"
#include "control/api_support.hpp"
#include "control/user_manager.hpp"
#include "common/pairing.hpp"
#include "control/served_model_validator.hpp"
#include "control/engine_config_store.hpp"
#include "control/engine_group_planner.hpp"
#include "control/tts_service_client.hpp"
#include "common/http_server.hpp"
#include "common/http_client.hpp"
#include "common/conversation_context.hpp"
#include "common/runtime_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "common/sse_infer_ctx.hpp"
#include <deque>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <string>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mm {

/// Upstream read timeout for a proxied SSE stream.
///
/// Long and FINITE. `set_read_timeout(0, 0)` reads as zero seconds in
/// cpp-httplib rather than as "no limit", which made every proxied telemetry
/// stream close instantly with a clean, empty 200 — defect D10. Infinity is not
/// the right answer either: a wedged engine should eventually release the socket.
/// An hour is far beyond any tick interval and far short of forever.
constexpr std::chrono::seconds kSseUpstreamReadTimeout{3600};

namespace {

/// One admitted model as JSON. The verdict is labelled with the scope it applies
/// to, because a reader who takes it as "what this node will do" will eventually
/// be wrong: the verdict is a property of (model, quantization, host budget).
nlohmann::json admitted_model_json(const AdmittedModel& m) {
    return nlohmann::json{
        {"id", m.id},
        {"object", "model"},
        {"arch_hash", m.arch_hash},
        {"name", m.name},
        {"source_repo", m.source_repo},
        {"source_revision", m.source_revision},
        {"model_dir", m.model_dir},
        {"attention_family", m.attention_family},
        {"n_layers", m.n_layers},
        {"n_moe_layers", m.n_moe_layers},
        {"n_experts", m.n_experts},
        {"top_k", m.top_k},
        {"expert_bytes", m.expert_bytes},
        {"bytes_per_token", m.bytes_per_token},
        {"total_routed_bytes", m.total_routed_bytes},
        {"active_fraction", m.active_fraction},
        {"verdict", to_string(m.verdict)},
        {"verdict_scope", "admission-host"},
        {"verdict_reason", m.verdict_reason},
        {"selects_soma", verdict_selects_soma(m.verdict)},
        {"admitted_at_ms", m.admitted_at_ms},
        {"profiled_at_ms", m.profiled_at_ms},
    };
}

/// The conformance ladder as an array, for the two routes that publish it.
///
/// One builder because two would drift, and the thing they would drift about is
/// which field a client should read. `status` is that field — `passed` stays for
/// clients written against the original shape, but a boolean cannot express
/// "did not run", and that is what most of this ladder says on a serving host.
nlohmann::json conformance_json(const ControlModelRegistry& models, std::int64_t model_id) {
    auto stages = nlohmann::json::array();
    for (const auto& c : models.conformance(model_id)) {
        stages.push_back(nlohmann::json{{"stage", c.stage},
                                        {"status", c.status},
                                        {"passed", c.passed},
                                        {"detail", c.detail},
                                        {"ran_at_ms", c.ran_at_ms}});
    }
    return stages;
}

nlohmann::json admission_progress_json(const AdmissionProgress& p) {
    return nlohmann::json{
        {"operation_id", p.operation_id},
        {"stage", p.stage},
        {"detail", p.detail},
        {"step", p.step},
        {"total_steps", p.total_steps},
        {"fraction", p.fraction},
        // Populated by the fetch stage, which is the only one whose remaining
        // time a client can estimate. Zero elsewhere, and a client should read
        // it as "not reported" rather than "nothing to do".
        {"bytes_done", p.bytes_done},
        {"bytes_total", p.bytes_total},
        {"cancelable", p.cancelable},
        {"done", p.done},
        {"canceled", p.canceled},
        {"error", p.last_error},
        {"model_id", p.model_id},
        {"source", p.source_ref},
        {"started_at_ms", p.started_at_ms},
        {"finished_at_ms", p.finished_at_ms},
    };
}

/// Drain an operation's frames to the wire until it ends.
///
/// A heartbeat every few seconds, because conversion produces no output for
/// long stretches while it writes a shard — and a silent connection is one that
/// proxies and clients time out. An SSE comment is invisible to a parser and
/// keeps the socket alive, which is exactly what is wanted.
void stream_admission(httplib::Response& res,
                      ControlModelRegistry* models,
                      const std::string& op_id,
                      const std::shared_ptr<SseInferCtx>& ctx) {
    res.set_chunked_content_provider(
        "text/event-stream", [ctx, models, op_id](std::size_t, httplib::DataSink& sink) {
            std::unique_lock<std::mutex> lk(ctx->mx);
            const bool have = ctx->cv.wait_for(lk, std::chrono::seconds(5),
                                               [&] { return !ctx->lines.empty() || ctx->done; });
            if (!have && !ctx->done) {
                lk.unlock();
                static const std::string beat = ": keepalive\n\n";
                return sink.write(beat.data(), beat.size());
            }
            std::deque<std::string> batch;
            batch.swap(ctx->lines);
            const bool finished = ctx->done;
            lk.unlock();

            for (const auto& line : batch) {
                if (!sink.write(line.data(), line.size())) {
                    // The client hung up. The admission KEEPS RUNNING: hours of
                    // conversion must not be discarded because a browser tab
                    // closed, and /v1/models/admissions/{op} exists to rejoin it.
                    MM_INFO("admission {}: client disconnected; conversion continues", op_id);
                    (void)models;
                    return false;
                }
            }
            if (finished && batch.empty()) {
                sink.done();
                return false;
            }
            return true;
        });
}

std::int64_t parse_model_id(const std::string& raw) {
    try {
        return std::stoll(raw);
    } catch (...) {
        return -1;   // never matches a row; the caller answers 404
    }
}
}  // namespace

using namespace api;

namespace {

/// The largest :9091 request body: a completion carrying the most image bytes
/// a request may, base64-encoded, plus room for the text around them.
constexpr std::size_t kOpenAiJsonCeiling =
    static_cast<std::size_t>((kMaxContextImageBytes * 4 + 2) / 3) + (16ULL * 1024 * 1024);

} // namespace

// ── Constructor / destructor ───────────────────────────────────────────────────

std::chrono::milliseconds ControlApiServer::chat_queue_wait() {
    std::int64_t wait_ms = 180000;
    if (const char* env_wait = std::getenv("MM_CHAT_QUEUE_WAIT_MS")) {
        try {
            const std::int64_t parsed = std::stoll(env_wait);
            if (parsed > 0) wait_ms = parsed;
        } catch (...) {
        }
    }
    return std::chrono::milliseconds{wait_ms};
}

std::optional<ServedModel> ControlApiServer::served_model_for_holder(
    const HolderId& holder_id) const {
    if (served_models_ != nullptr) {
        if (auto entry = served_models_->find(holder_id)) return entry->model;
    }
    return std::nullopt;
}

ControlApiServer::ControlApiServer(UserManager& users,
                                   NodeRegistry& registry,
                                   PlacementScheduler& scheduler,
                                   std::string data_dir,
                                   std::string models_dir,
                                   std::string external_api_token,
                                   TtsServiceConfig tts_config,
                                   std::size_t chat_workers)
    : users_(users)
    , registry_(registry)
    , scheduler_(scheduler)
    , data_dir_(std::move(data_dir))
    , models_dir_(std::move(models_dir))
    , external_api_token_(std::move(external_api_token))
    , tts_(std::move(tts_config))
    , server_(std::make_unique<HttpServer>())
    , openai_server_(std::make_unique<HttpServer>())
    , conversations_(std::make_unique<ConversationQueue>(chat_workers))
{
    // Configured HERE, not only from set_model_registry(): the legacy token is a
    // constructor argument and the registry is optional, so deferring this left
    // auth entirely off for any caller that never attached a registry.
    scopes_.configure(nullptr, external_api_token_);
}

ControlApiServer::~ControlApiServer() { stop(); }

bool ControlApiServer::listen(uint16_t port) {
    server_->set_payload_max_length(static_cast<std::size_t>(kMaxImageBytes) +
                                    (2ULL * 1024 * 1024));
    server_->SetPreRoutingHandler([this](const httplib::Request& req,
                                         httplib::Response& res) {
        return authorize_external_request(req, res);
    });
    register_routes();
    register_served_model_routes();
    register_user_routes();

    // Every registered handler must have a scope entry, checked HERE against the
    // actual registrations rather than against a second list someone maintains.
    // A route with no entry is a startup failure: defaulting it to `read` would
    // silently under-protect a new mutation, and defaulting to `operator` would
    // silently break a new GET.
    //
    // /api/control/* is excluded — node registration has its own credential and
    // predates scopes entirely.
    {
        std::vector<std::string> scoped;
        for (const auto& route : server_->registered_routes()) {
            if (route.find(" /v1/") != std::string::npos) scoped.push_back(route);
        }
        std::vector<std::string> missing;
        if (!require_complete_coverage(scoped, missing)) {
            for (const auto& m : missing) {
                MM_ERROR("route has no scope entry: {}", m);
            }
            MM_ERROR("{} route(s) are unscoped; refusing to listen. Add them to "
                     "route_scope_table() in src/control/route_scope.cpp.",
                     missing.size());
            return false;
        }
    }

    if (!scopes_.auth_enabled()) {
        // Said out loud, once. A system that is open because nobody configured a
        // token should announce it rather than look configured.
        MM_WARN("API authorization is DISABLED: no external_api_token and no rows in "
                "api_token. Every /v1/* route is open to anyone who can reach this port.");
    }

    MM_INFO("ControlApiServer listening on port {}", port);
    return server_->listen("0.0.0.0", port);
}

bool ControlApiServer::listen_openai_compat(uint16_t port) {
    openai_server_->set_payload_max_length(kOpenAiJsonCeiling);
    openai_server_->SetPreRoutingHandler([this](const httplib::Request& req,
                                                httplib::Response& res) {
        return authorize_openai_compat_request(req, res);
    });
    register_openai_compat_routes();
    // The same refusal as the main port: a route with no scope entry is a
    // startup failure. This listener had no such check, so its routes could
    // drift out of route_scope_table() unnoticed.
    {
        std::vector<std::string> scoped;
        for (const auto& route : openai_server_->registered_routes()) {
            if (route.find(" /v1/") != std::string::npos) scoped.push_back(route);
        }
        std::vector<std::string> missing;
        if (!require_complete_coverage(scoped, missing)) {
            for (const auto& m : missing) MM_ERROR("OpenAI-compat route has no scope entry: {}", m);
            return false;
        }
    }
    MM_INFO("OpenAI-compatible API listening on port {}", port);
    return openai_server_->listen("0.0.0.0", port);
}

void ControlApiServer::stop() {
    server_->stop();
    stop_openai_compat();
    // After the listeners, so nothing new arrives: in-flight turns and streams
    // finish and release their placements before the scheduler can go away.
    if (conversations_) conversations_->shutdown();
    if (openai_queue_) openai_queue_->shutdown();
}

void ControlApiServer::stop_openai_compat() { openai_server_->stop(); }

void ControlApiServer::cleanup_expired_tts_cache() {
    const int64_t now = util::now_ms();
    int removed = 0;
    for (const auto& profile : users_.list_users()) {
        auto user = users_.get_user(profile.id);
        if (!user) continue;
        for (const auto& entry : user->db().delete_expired_tts_cache_entries(now)) {
            remove_file_quietly(entry.audio_path);
            ++removed;
        }
        removed += static_cast<int>(user->db().delete_expired_unreferenced_attachments(now).size());
    }
    if (removed > 0) {
        activity_log(0, "Expired managed cache entries cleaned: " + std::to_string(removed));
    }
}

void ControlApiServer::set_log_callback(LogCallback cb) { log_cb_ = std::move(cb); }

void ControlApiServer::publish_activity(int level, const std::string& message) {
    nlohmann::json entry = {
        {"timestamp_ms", util::now_ms()},
        {"level", level},
        {"message", message}
    };
    {
        std::lock_guard<std::mutex> lk(activity_mutex_);
        if (activity_entries_.size() >= kMaxActivityEntries) activity_entries_.pop_front();
        activity_entries_.push_back(entry);
    }
    if (log_cb_) log_cb_(level, message);
}

void ControlApiServer::activity_log(int level, const std::string& message) {
    publish_activity(level, message);
}

std::optional<NodeInfo> ControlApiServer::find_engine_node(const SlotId& engine_id) const {
    for (const auto& node : registry_.list_nodes()) {
        if (!node.connected) continue;
        for (const auto& slot : node.slots) {
            if (slot.id == engine_id) return node;
        }
    }
    return std::nullopt;
}

void ControlApiServer::proxy_engine_get(const SlotId& engine_id,
                                        const std::string& suffix,
                                        const std::string& query,
                                        httplib::Response& res) const {
    const auto node = find_engine_node(engine_id);
    if (!node) {
        res.status = 404;
        res.set_content(nlohmann::json{{"error", "no such engine"}}.dump(), "application/json");
        return;
    }
    httplib::Client cli(node->url);
    cli.set_read_timeout(15);
    if (!node->api_key.empty()) cli.set_bearer_token_auth(node->api_key);
    auto upstream = cli.Get(("/api/node/engines/" + engine_id + suffix + query).c_str());
    if (!upstream) {
        // 502, not 500: the failure is upstream, and an operator chasing it
        // should be pointed at the node rather than at control.
        res.status = 502;
        res.set_content(nlohmann::json{{"error", "node unreachable"}, {"node_id", node->id}}.dump(),
                        "application/json");
        return;
    }
    // Status passed through: a 501 from an engine with no telemetry surface must
    // stay a 501 rather than becoming a generic control-side error.
    res.status = upstream->status;
    res.set_content(upstream->body, "application/json");
}

void ControlApiServer::proxy_node_engine_action(const NodeId& node_id,
                                                const std::string& engine_id,
                                                const std::string& action,
                                                const std::string& body,
                                                httplib::Response& res) const {
    NodeInfo node;
    try {
        node = registry_.get_node(node_id);
    } catch (const std::exception&) {
        res.status = 404;
        res.set_content(nlohmann::json{{"error", "no such node"}, {"node_id", node_id}}.dump(),
                        "application/json");
        return;
    }
    // Refused BEFORE the round trip, and with the node's name in it. A
    // disconnected node is the ordinary reason one of these fails, and
    // "connection refused" bubbling up as a 502 tells an operator less than
    // this does about a node control already knows is down.
    if (!node.connected) {
        res.status = 409;
        res.set_content(nlohmann::json{{"error", "node is not connected"},
                                       {"node_id", node_id},
                                       {"hostname", node.hostname}}
                            .dump(),
                        "application/json");
        return;
    }

    httplib::Client cli(node.url);
    // 30 s is generous for a handler that only starts a worker, and deliberately
    // NOT sized for a build: if this ever needs minutes, the node has gone back
    // to running the work inline and that is the bug to fix, not the timeout.
    cli.set_read_timeout(30);
    if (!node.api_key.empty()) cli.set_bearer_token_auth(node.api_key);
    auto upstream = cli.Post(("/api/node/engines/" + engine_id + "/" + action).c_str(),
                             body.empty() ? std::string("{}") : body,
                             "application/json");
    if (!upstream) {
        res.status = 502;
        res.set_content(nlohmann::json{{"error", "node unreachable"},
                                       {"node_id", node_id},
                                       {"hostname", node.hostname}}
                            .dump(),
                        "application/json");
        return;
    }
    res.status = upstream->status;
    res.set_content(upstream->body, "application/json");
}

bool ControlApiServer::authorize_external_request(const httplib::Request& req,
                                                  httplib::Response& res) const {
    // external_api_token_ is only for public control clients. Internal node
    // routes under /api/control/* keep their separate registered-node auth.
    if (req.path.rfind("/v1/", 0) != 0 && req.path != "/v1") return true;

    // One middleware, REPLACED rather than layered. The flat-token comparison
    // that lived here is now the legacy path inside ScopeAuthorizer, matched
    // before the token table and granted every scope — so existing deployments
    // keep working unchanged while a scoped token can be narrower.
    return scopes_.authorize_or_reject(req, res);
}

// ── register_routes ───────────────────────────────────────────────────────────

void ControlApiServer::register_routes() {
    using namespace httplib;

    auto extract_bearer = [](const std::string& auth_header) -> std::string {
        static const std::string kBearer = "Bearer ";
        if (auth_header.rfind(kBearer, 0) != 0) return {};
        return auth_header.substr(kBearer.size());
    };

    // ── POST /api/control/register-node ───────────────────────────────────────
    server_->Post("/api/control/register-node", [this, extract_bearer](const Request& req, Response& res) {
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string node_url = j.value("node_url", "");
            std::string api_key  = j.value("api_key", "");
            std::string platform = j.value("platform", "");
            std::string hostname = j.value("hostname", "");

            if (node_url.empty() || api_key.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"node_url and api_key required"})",
                                "application/json");
                return;
            }

            // A node may only self-register if it is already known (re-announce
            // after restart / address change, matched by its api_key) or if the
            // request carries the control's external bearer token. New nodes
            // must be added through pairing or POST /v1/nodes.
            const bool known_node = registry_.find_node_by_api_key(api_key).has_value();
            const bool bearer_ok = !external_api_token_.empty() &&
                extract_bearer(req.get_header_value("Authorization")) == external_api_token_;
            if (!known_node && !bearer_ok) {
                MM_WARN("register-node rejected for {}: unknown api_key and no valid bearer token",
                        node_url);
                res.status = 401;
                res.set_content(
                    R"({"error":"unauthorized: pair this node first or send the control bearer token"})",
                    "application/json");
                return;
            }

            NodeId node_id = registry_.add_node(node_url, api_key, platform, false, hostname);
            MM_INFO("Node registered: {} @ {}", node_id, node_url);

            res.set_content(
                nlohmann::json{{"node_id", node_id}, {"accepted", true}}.dump(),
                "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── GET /v1/nodes ─────────────────────────────────────────────────────────
    server_->Get("/v1/nodes/discovered", [this](const Request& /*req*/, Response& res) {
        auto discovered = registry_.get_discovered_nodes();
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& d : discovered) {
            arr.push_back(nlohmann::json{
                {"url", d.url},
                {"node_id", d.node_id},
                {"hostname", d.hostname},
                {"last_seen_ms", d.last_seen_ms}
            });
        }
        res.set_content(arr.dump(), "application/json");
    });

    server_->Post("/v1/nodes", [this](const Request& req, Response& res) {
        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string url = util::trim(j.value("url", std::string{}));
            const std::string api_key = util::trim(j.value("api_key", std::string{}));
            const std::string platform = util::trim(j.value("platform", std::string{}));
            const bool remember = j.value("remember", false);
            const std::string hostname = util::trim(j.value("hostname", std::string{}));
            if (url.empty() || api_key.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"url and api_key required"})", "application/json");
                return;
            }
            const NodeId node_id = registry_.add_node(url, api_key, platform, remember, hostname);
            publish_activity(0, "Node added via API: " + node_id + " @ " + url +
                                (remember ? " (remembered)" : ""));
            res.status = 201;
            res.set_content(nlohmann::json{{"node_id", node_id}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    server_->Delete("/v1/nodes/:id", [this](const Request& req, Response& res) {
        const std::string node_id = req.path_params.at("id");
        try {
            registry_.get_node(node_id);
        } catch (...) {
            res.status = 404;
            res.set_content(R"({"error":"node not found"})", "application/json");
            return;
        }
        registry_.remove_node(node_id);
        publish_activity(0, "Node removed via API: " + node_id);
        res.set_content(R"({"status":"removed"})", "application/json");
    });

    server_->Post("/v1/nodes/:id/forget", [this](const Request& req, Response& res) {
        const std::string node_id = req.path_params.at("id");
        try {
            registry_.get_node(node_id);
        } catch (...) {
            res.status = 404;
            res.set_content(R"({"error":"node not found"})", "application/json");
            return;
        }
        const bool changed = registry_.forget_node(node_id);
        publish_activity(0, "Node forget requested via API: " + node_id);
        res.set_content(nlohmann::json{{"status", "forgotten"}, {"changed", changed}}.dump(),
                        "application/json");
    });

    server_->Post("/v1/nodes/pair/start", [this](const Request& req, Response& res) {
        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string url = util::trim(j.value("url", std::string{}));
            if (url.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"url required"})", "application/json");
                return;
            }
            std::string nonce = registry_.start_pair(url);
            if (nonce.empty()) {
                res.status = 502;
                res.set_content(R"({"error":"pair start failed"})", "application/json");
                return;
            }
            publish_activity(0, "Pair start requested for " + url);
            res.set_content(nlohmann::json{{"nonce", nonce}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    server_->Post("/v1/nodes/pair/complete", [this](const Request& req, Response& res) {
        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string url = util::trim(j.value("url", std::string{}));
            const std::string nonce = util::trim(j.value("nonce", std::string{}));
            const std::string pin_or_psk = util::trim(j.value("pin_or_psk", std::string{}));
            const bool remember = j.value("remember", false);
            if (url.empty() || nonce.empty() || pin_or_psk.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"url, nonce, and pin_or_psk required"})", "application/json");
                return;
            }
            std::string key = registry_.complete_pair(url, nonce, pin_or_psk, remember);
            if (key.empty()) {
                res.status = 502;
                res.set_content(R"({"error":"pair complete failed"})", "application/json");
                return;
            }
            publish_activity(0, "Pair complete accepted for " + url +
                                (remember ? " (remembered)" : ""));
            res.set_content(nlohmann::json{{"api_key", key}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    server_->Post("/v1/nodes/pair/psk", [this](const Request& req, Response& res) {
        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string url = util::trim(j.value("url", std::string{}));
            std::string psk = util::trim(j.value("psk", std::string{}));
            const bool remember = j.value("remember", false);
            if (url.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"url required"})", "application/json");
                return;
            }
            if (psk.empty()) {
                const char* env_psk = std::getenv("MM_PAIRING_KEY");
                if (env_psk && *env_psk) psk = env_psk;
            }
            if (psk.empty()) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "psk required (or set MM_PAIRING_KEY)"}}.dump(),
                                "application/json");
                return;
            }
            std::string key = registry_.pair_node(url, psk, remember);
            if (key.empty()) {
                res.status = 502;
                res.set_content(R"({"error":"psk pair failed"})", "application/json");
                return;
            }
            publish_activity(0, "PSK pair accepted for " + url +
                                (remember ? " (remembered)" : ""));
            res.set_content(nlohmann::json{{"api_key", key}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    server_->Get("/v1/nodes", [this](const Request& /*req*/, Response& res) {
        auto nodes = registry_.list_nodes();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& n : nodes) arr.push_back(nlohmann::json(n));
        res.set_content(arr.dump(), "application/json");
    });

    server_->Get("/v1/activity", [this](const Request& req, Response& res) {
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

        int level_filter = -1;
        if (req.has_param("level")) {
            std::string raw = util::to_lower(util::trim(req.get_param_value("level")));
            if (raw == "info" || raw == "0") level_filter = 0;
            else if (raw == "warn" || raw == "warning" || raw == "1") level_filter = 1;
            else if (raw == "error" || raw == "2") level_filter = 2;
            else {
                res.status = 400;
                res.set_content(R"({"error":"level must be info|warn|error|0|1|2"})", "application/json");
                return;
            }
        }

        std::vector<nlohmann::json> filtered;
        {
            std::lock_guard<std::mutex> lk(activity_mutex_);
            filtered.reserve(activity_entries_.size());
            for (const auto& entry : activity_entries_) {
                if (level_filter >= 0) {
                    const int lv = entry.value("level", 0);
                    if (lv != level_filter) continue;
                }
                filtered.push_back(entry);
            }
        }
        const int total = static_cast<int>(filtered.size());
        const int start = std::max(0, total - tail);
        nlohmann::json out = nlohmann::json::array();
        for (int i = start; i < total; ++i) out.push_back(filtered[static_cast<std::size_t>(i)]);
        res.set_content(nlohmann::json{{"entries", out}}.dump(), "application/json");
    });

    server_->Get("/v1/performance", [this](const Request& req, Response& res) {
        int tail = 200;
        if (req.has_param("tail")) {
            try { tail = std::stoi(req.get_param_value("tail")); }
            catch (...) {
                res.status = 400;
                res.set_content(R"({"error":"tail must be an integer"})", "application/json");
                return;
            }
        }
        tail = std::clamp(tail, 1, 2000);
        res.set_content(performance_.snapshot(static_cast<std::size_t>(tail)).dump(),
                        "application/json");
    });

    server_->Delete("/v1/performance", [this](const Request&, Response& res) {
        performance_.clear();
        res.set_content(R"({"cleared":true})", "application/json");
    });

    // ── /v1/tokens — the scope store ────────────────────────────────────────
    //
    // Operator-only, necessarily: a credential that can mint credentials is the
    // most powerful one there is.
    server_->Get("/v1/tokens", [this](const Request& /*req*/, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& t : models_->list_api_tokens()) {
            // The HASH, never the token. There is nothing to leak here, and the
            // hash is what correlates a row with a log line.
            arr.push_back(nlohmann::json{{"id", t.id},
                                         {"label", t.label},
                                         {"scopes", format_scopes(t.scopes)},
                                         {"token_sha256", t.token_sha256},
                                         {"user_id", t.user_id.empty() ? nlohmann::json(nullptr)
                                                                       : nlohmann::json(t.user_id)},
                                         {"created_at_ms", t.created_at_ms},
                                         {"last_used_at_ms", t.last_used_at_ms},
                                         {"revoked", t.revoked}});
        }
        res.set_content(nlohmann::json{{"data", arr}}.dump(), "application/json");
    });

    server_->Post("/v1/tokens", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        std::string label, scopes_csv, user_id;
        try {
            const auto j = nlohmann::json::parse(req.body);
            label = j.value("label", std::string{});
            scopes_csv = j.value("scopes", std::string{});
            user_id = util::trim(j.value("user_id", std::string{}));
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        ScopeSet scopes = kScopeNone;
        if (!parse_scopes(scopes_csv, scopes) || scopes == kScopeNone) {
            // An unrecognised scope name is REFUSED rather than dropped. Silently
            // ignoring "opereator" would mint a token that looks right in the
            // request and cannot do the thing it was minted for.
            res.status = 400;
            res.set_content(
                nlohmann::json{{"error", "scopes must be a comma-separated subset of "
                                         "read,chat,operator"},
                               {"got", scopes_csv}}
                    .dump(),
                "application/json");
            return;
        }
        // A binding must name a user that exists: a token bound to nobody would
        // reach nothing, and one bound to a name nobody holds yet would become a
        // key to whoever is created under it.
        if (!user_id.empty() && !users_.get_user(user_id)) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", "user not found: " + user_id}}.dump(),
                            "application/json");
            return;
        }
        std::string err;
        const auto secret = models_->create_api_token(label, scopes, err, user_id);
        if (secret.empty()) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }
        res.status = 201;
        // SHOWN ONCE. Only the hash is stored, so this response is the only place
        // the secret exists; a caller that loses it mints another.
        res.set_content(nlohmann::json{{"token", secret},
                                       {"label", label},
                                       {"scopes", format_scopes(scopes)},
                                       {"user_id", user_id.empty() ? nlohmann::json(nullptr)
                                                                   : nlohmann::json(user_id)},
                                       {"note", "stored hashed; this is the only time it is shown"}}
                            .dump(),
                        "application/json");
    });

    server_->Delete("/v1/tokens/:id", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        std::string err;
        if (!models_->revoke_api_token(parse_model_id(req.path_params.at("id")), err)) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }
        // Revoked, not deleted: the row is the audit trail.
        res.set_content(nlohmann::json{{"status", "revoked"}}.dump(), "application/json");
    });

    // ── /v1/models — the ADMISSION REGISTRY ─────────────────────────────────
    //
    // Reclaimed. This route used to return agents wearing model costumes, with
    // an `openai_compat_note` pointing at the other port for the real thing. The
    // catalog a client picks a model from is /v1/served-models (and GET
    // /v1/models on the :9091 listener); here /v1/models means what it says.
    server_->Get("/v1/models", [this](const Request& /*req*/, Response& res) {
        if (models_ == nullptr) {
            res.status = 503;
            res.set_content(nlohmann::json{{"error", "model registry is not open"}}.dump(),
                            "application/json");
            return;
        }
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& m : models_->list()) arr.push_back(admitted_model_json(m));
        res.set_content(nlohmann::json{{"object", "list"}, {"data", arr}}.dump(),
                        "application/json");
    });

    // Documented since the API surface was written, and until D10-era testing
    // nobody called it: the rows came back inside GET /v1/models/{id} and this
    // path 404'd. Registered rather than struck from the docs, because a client
    // written against the document should work — and because a caller who wants
    // only the ladder should not have to fetch the whole record to read it.
    server_->Get("/v1/models/:id/conformance", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        const auto model = models_->find_by_id(parse_model_id(req.path_params.at("id")));
        if (!model) { res.status = 404; return; }
        res.set_content(nlohmann::json{{"model_id", model->id},
                                       {"arch_hash", model->arch_hash},
                                       {"conformance", conformance_json(*models_, model->id)}}
                            .dump(),
                        "application/json");
    });

    server_->Get("/v1/models/:id", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        const auto model = models_->find_by_id(parse_model_id(req.path_params.at("id")));
        if (!model) { res.status = 404; return; }
        auto j = admitted_model_json(*model);
        j["conformance"] = conformance_json(*models_, model->id);
        res.set_content(j.dump(), "application/json");
    });

    // Register an ALREADY-CONVERTED model. The conversion pipeline is offline
    // (tools/admission/), and a model converted there still has to become
    // evidence here — without a record, every served model routes to the fallback.
    server_->Post("/v1/models", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        AdmittedModel m;
        try {
            const auto j = nlohmann::json::parse(req.body);
            m.arch_hash = j.value("arch_hash", std::string{});
            m.name = j.value("name", std::string{});
            m.source_repo = j.value("source_repo", std::string{});
            m.source_revision = j.value("source_revision", std::string{});
            m.model_dir = j.value("model_dir", std::string{});
            m.attention_family = j.value("attention_family", std::string{"gqa"});
            m.n_layers = j.value("n_layers", 0u);
            m.n_moe_layers = j.value("n_moe_layers", 0u);
            m.n_experts = j.value("n_experts", 0u);
            m.top_k = j.value("top_k", 0u);
            m.expert_bytes = j.value("expert_bytes", std::int64_t{0});
            m.bytes_per_token = j.value("bytes_per_token", std::int64_t{0});
            m.total_routed_bytes = j.value("total_routed_bytes", std::int64_t{0});
            m.active_fraction = j.value("active_fraction", 0.0);
            m.verdict = parse_verdict(j.value("verdict", std::string{"reject"}));
            m.verdict_basis = j.contains("verdict_basis") ? j["verdict_basis"].dump()
                                                          : std::string("{}");
            m.verdict_reason = j.value("verdict_reason", std::string{});
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        std::string err;
        if (!models_->upsert(m, err)) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }
        res.status = 201;
        res.set_content(admitted_model_json(m).dump(), "application/json");
    });

    // The operator override of a computed verdict.
    server_->Put("/v1/models/:id/verdict", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        std::string verdict_text, reason;
        try {
            const auto j = nlohmann::json::parse(req.body);
            verdict_text = j.value("verdict", std::string{});
            reason = j.value("reason", std::string{});
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        // Validated against the spelling, not silently coerced: parse_verdict()
        // defaults to `reject`, so a typo would otherwise quietly become the
        // most restrictive verdict and read as the engine refusing the model.
        static const std::vector<std::string> kValid = {"stream", "hybrid", "resident-only",
                                                        "reject"};
        if (std::find(kValid.begin(), kValid.end(), util::to_lower(util::trim(verdict_text))) ==
            kValid.end()) {
            res.status = 400;
            res.set_content(
                nlohmann::json{{"error", "verdict must be stream, hybrid, resident-only or reject"},
                               {"got", verdict_text}}
                    .dump(),
                "application/json");
            return;
        }
        std::string err;
        const auto id = parse_model_id(req.path_params.at("id"));
        if (!models_->set_verdict(id, parse_verdict(verdict_text), reason, err)) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }
        const auto model = models_->find_by_id(id);
        res.set_content(model ? admitted_model_json(*model).dump() : std::string("{}"),
                        "application/json");
    });

    server_->Delete("/v1/models/:id", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        std::string err;
        if (!models_->remove(parse_model_id(req.path_params.at("id")), err)) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }
        res.set_content(nlohmann::json{{"status", "deleted"}}.dump(), "application/json");
    });

    server_->Get("/v1/models/:id/plan", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        std::string body, err;
        if (!models_->plan_for_host(parse_model_id(req.path_params.at("id")), {}, body, err)) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }
        res.set_content(body, "application/json");
    });

    // ── POST /v1/models/admit ───────────────────────────────────────────────
    //
    // Convert, compile the tokenizer, plan, record — as SSE, because it runs for
    // hours and an operation you cannot watch is an operation you cannot trust.
    // The response is a stream of AdmissionProgress frames ending in one with
    // `done: true`; `last_error` distinguishes failure from success there,
    // because a stream that simply goes quiet is indistinguishable from a
    // network fault.
    server_->Post("/v1/models/admit", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        std::string source;
        ControlModelRegistry::QuantOverride quant;
        try {
            const auto body = nlohmann::json::parse(req.body);
            source = body.value("source", std::string{});
            // Per request, because the same weights at two quantizations are two
            // different admissions with two different verdicts — the premise the
            // registry keys on, and one that cannot be exercised at all if the
            // quantization is a config file the operator has to edit and restart
            // to change.
            const auto q = body.value("quantization", nlohmann::json::object());
            quant.quant = q.value("expert_gate", std::string{});
            quant.expert_down = q.value("expert_down", std::string{});
            quant.group = q.value("group", 0);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        auto ctx = std::make_shared<SseInferCtx>();
        std::string err;
        // The sink is registered BEFORE admit() returns, so the first frames
        // cannot be produced before anyone is listening for them.
        const auto op_id = models_->admit(source, quant, [ctx](const AdmissionProgress& p) {
            std::lock_guard<std::mutex> lk(ctx->mx);
            ctx->lines.push_back("data: " + admission_progress_json(p).dump() + "\n\n");
            if (p.done) ctx->done = true;
            ctx->cv.notify_one();
        }, err);

        if (op_id.empty()) {
            // A bad source is a 400 on the REQUEST, not an error frame inside a
            // 200 stream: nothing started, so there is nothing to stream about.
            res.status = 400;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }
        stream_admission(res, models_, op_id, ctx);
    });

    // Watch an admission that is already running, or find out how one ended.
    server_->Get("/v1/models/admissions", [this](const Request& /*req*/, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& p : models_->operations()) arr.push_back(admission_progress_json(p));
        res.set_content(nlohmann::json{{"data", arr}}.dump(), "application/json");
    });

    server_->Get("/v1/models/admissions/:op", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        const std::string op_id = req.path_params.at("op");
        auto ctx = std::make_shared<SseInferCtx>();
        AdmissionProgress current;
        if (!models_->attach_sink(op_id, [ctx](const AdmissionProgress& p) {
                std::lock_guard<std::mutex> lk(ctx->mx);
                ctx->lines.push_back("data: " + admission_progress_json(p).dump() + "\n\n");
                if (p.done) ctx->done = true;
                ctx->cv.notify_one();
            }, current)) {
            res.status = 404;
            return;
        }
        {
            // The current state first, so a client joining late is not left
            // waiting for the next update to learn where things stand — and so a
            // client joining after the end learns the outcome immediately.
            std::lock_guard<std::mutex> lk(ctx->mx);
            ctx->lines.push_back("data: " + admission_progress_json(current).dump() + "\n\n");
            if (current.done) ctx->done = true;
        }
        stream_admission(res, models_, op_id, ctx);
    });

    server_->Post("/v1/models/admissions/:op/cancel", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        const bool ok = models_->cancel(req.path_params.at("op"));
        if (!ok) {
            // 409, not 404: the operation may exist and simply be past the point
            // of cancelling, which is a different thing from never having existed.
            res.status = 409;
            res.set_content(
                nlohmann::json{{"error", "unknown operation, or it already finished"}}.dump(),
                "application/json");
            return;
        }
        res.set_content(nlohmann::json{{"status", "canceling"}}.dump(), "application/json");
    });

    server_->Post("/v1/models/:id/reprofile", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        auto ctx = std::make_shared<SseInferCtx>();
        std::string err;
        const auto op_id = models_->reprofile(
            parse_model_id(req.path_params.at("id")),
            [ctx](const AdmissionProgress& p) {
                std::lock_guard<std::mutex> lk(ctx->mx);
                ctx->lines.push_back("data: " + admission_progress_json(p).dump() + "\n\n");
                if (p.done) ctx->done = true;
                ctx->cv.notify_one();
            },
            err);
        if (op_id.empty()) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }
        stream_admission(res, models_, op_id, ctx);
    });

    server_->Get("/v1/models/:id/heat", [this](const Request& req, Response& res) {
        if (models_ == nullptr) { res.status = 503; return; }
        // Bucketed by DEFAULT; `?resolution=full` is an explicit opt-in. A model
        // with 60k experts is not something a client should get by accident.
        const bool bucketed = req.get_param_value("resolution") != "full";
        std::string body;
        if (!models_->heat(parse_model_id(req.path_params.at("id")), bucketed, body)) {
            res.status = 404;
            return;
        }
        res.set_content(body, "application/json");
    });

    // ── /v1/engines — the cluster's live engines ────────────────────────────
    //
    // An "engine" cluster-wide is a (node, slot) pair, so the id is the slot id
    // and the node is discovered rather than supplied. That keeps a client from
    // having to know which node holds what — which changes under it on every
    // eviction.
    server_->Get("/v1/engines", [this](const Request& /*req*/, Response& res) {
        nlohmann::json arr = nlohmann::json::array();
        ResourceFootprint cluster{};
        for (const auto& node : registry_.list_nodes()) {
            if (!node.connected) continue;
            for (const auto& slot : node.slots) {
                arr.push_back(nlohmann::json{{"id", slot.id},
                                             {"node_id", node.id},
                                             {"node_url", node.url},
                                             {"backend", slot.backend},
                                             {"state", to_string(slot.state)},
                                             {"model_path", slot.model_path},
                                             {"served_model_ids", slot.holder_ids},
                                             {"vram_usage_mb", slot.vram_usage_mb},
                                             {"effective_ctx_size", slot.effective_ctx_size}});
                cluster.vram_mb += slot.vram_usage_mb;
            }
            cluster.disk_mb += node.disk_free_mb;
        }
        res.set_content(nlohmann::json{{"data", arr},
                                       {"tier_summary",
                                        {{"engines", arr.size()},
                                         {"vram_mb", cluster.vram_mb},
                                         {"node_disk_free_mb", cluster.disk_mb}}}}
                            .dump(),
                        "application/json");
    });

    // ── Cluster engine configuration ──────────────────────────────────────────
    //
    // Namespaced under /v1/cluster/ because /v1/engines above is already taken
    // and means something else: a live engine PROCESS on some node. These are
    // about engine KINDS and what the cluster is configured to run. Two
    // resources one word apart is how a client ends up reading slot state and
    // believing it is policy.

    server_->Get("/v1/cluster/engines/config", [this](const Request&, Response& res) {
        if (engine_config_ == nullptr) {
            res.status = 503;
            res.set_content(R"({"error":"engine configuration is not available"})",
                            "application/json");
            return;
        }
        // 200 with configured=false, not 404. The absence of a configuration is
        // a state of the cluster the client needs to READ — it is what drives
        // first-run setup — and a 404 would read as "this control does not have
        // this feature".
        res.set_content(nlohmann::json{{"configured", engine_config_->configured()},
                                       {"config", engine_config_->get()}}
                            .dump(),
                        "application/json");
    });

    server_->Put("/v1/cluster/engines/config", [this](const Request& req, Response& res) {
        if (engine_config_ == nullptr) {
            res.status = 503;
            res.set_content(R"({"error":"engine configuration is not available"})",
                            "application/json");
            return;
        }
        ClusterEngineConfig cfg;
        try {
            cfg = nlohmann::json::parse(req.body).get<ClusterEngineConfig>();
        } catch (const std::exception& e) {
            // Carries the forbidden-key refusal verbatim. An operator who set
            // `accelerator` needs to be told the field is per-machine, not
            // handed a generic parse error.
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        // Built-ins remain configurable before the first node registers. Node
        // reports may add future engine ids to this set.
        // cluster of Soma-only nodes cannot be configured for an engine none of
        // them has. Empty means no node has reported yet, and validation then
        // skips the id check rather than refusing every write until a node
        // appears.
        std::vector<std::string> known{"soma", "llama-cpp", "vllm"};
        for (const auto& n : registry_.list_nodes()) {
            for (const auto& e : n.engines) {
                if (std::find(known.begin(), known.end(), e.engine_id) == known.end())
                    known.push_back(e.engine_id);
            }
        }

        std::string err;
        if (!engine_config_->save(cfg, known, "api", err)) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }
        publish_activity(0, std::string("Cluster engine configuration updated to v") +
                                std::to_string(engine_config_->version()));
        res.set_content(nlohmann::json{{"configured", true},
                                       {"config", engine_config_->get()}}
                            .dump(),
                        "application/json");
    });

    server_->Get("/v1/cluster/engines/ray", [this](const Request&, Response& res) {
        if (engine_config_ == nullptr) {
            res.status = 503;
            res.set_content(R"({"error":"engine configuration is not available"})",
                            "application/json");
            return;
        }
        const auto cfg = engine_config_->get();
        const bool configured = cfg.primary_engine == "vllm" ||
                                cfg.backup_engine == "vllm";
        VllmEngineConfig profile;
        if (const auto* spec = cfg.find("vllm")) profile = effective_vllm_config(*spec);
        int eligible = 0;
        int active = 0;
        nlohmann::json eligible_nodes = nlohmann::json::array();
        nlohmann::json groups = nlohmann::json::array();
        std::unordered_map<std::string, nlohmann::json> group_rows;
        for (const auto& node : registry_.list_nodes()) {
            const bool linux = util::to_lower(node.platform) == "linux";
            const bool ready = std::any_of(
                node.engines.begin(), node.engines.end(), [](const RuntimeStatus& runtime) {
                    return runtime.engine_id == "vllm" && runtime.ready;
                });
            if (node.connected && linux && ready && node.capabilities.supports_ray &&
                node.capabilities.gpu_count >= profile.tensor_parallel_size &&
                node.engine_config_version == cfg.version && node.ray.group_id.empty())
            {
                ++eligible;
                eligible_nodes.push_back(nlohmann::json{
                    {"node_id", node.id},
                    {"gpu_count", node.capabilities.gpu_count},
                    {"architecture", node.capabilities.arch},
                    {"transports", node.capabilities.comm_backends}});
            }
            if (!node.ray.group_id.empty()) {
                auto [it, inserted] = group_rows.emplace(
                    node.ray.group_id,
                    nlohmann::json{{"group_id", node.ray.group_id},
                                   {"served_model_id", node.ray.holder_id},
                                   {"state", node.ray.state},
                                   {"transport", node.ray.transport},
                                   {"members", nlohmann::json::array()}});
                auto& group = it->second;
                if (node.ray.active()) group["state"] = "active";
                group["members"].push_back(nlohmann::json{
                    {"node_id", node.id}, {"role", node.ray.role},
                    {"state", node.ray.state},
                    {"head_address", node.ray.head_address},
                    {"reserved_gpus", node.ray.reserved_gpus}});
            }
        }
        for (auto& [_, group] : group_rows) {
            if (group.value("state", std::string{}) == "active") ++active;
            groups.push_back(std::move(group));
        }
        const bool required = configured && profile.pipeline_parallel_size > 1;
        std::vector<EngineGroupCandidate> eligible_groups;
        if (required) {
            EngineGroupRequest group_request;
            group_request.model_ref = "status/probe"; // valid HF-shaped id; no I/O
            group_request.tensor_parallel_size = profile.tensor_parallel_size;
            group_request.pipeline_parallel_size = profile.pipeline_parallel_size;
            group_request.config_version = cfg.version;
            group_request.allow_experimental_gloo = profile.allow_experimental_gloo;
            eligible_groups = plan_engine_groups(group_request, registry_.list_nodes());
        }
        std::string state = "hidden";
        std::string detail = "vLLM is not selected as primary or backup";
        if (configured && !required) {
            state = "inactive";
            detail = "pipeline parallelism is 1; Ray is not needed";
        } else if (required && active > 0) {
            state = "active";
            detail = std::to_string(active) + " Ray group(s) active";
        } else if (required && !eligible_groups.empty()) {
            state = "ready";
            detail = std::to_string(eligible_groups.size()) +
                     " compatible Ray placement(s) available on demand";
        } else if (required) {
            state = "unavailable";
            detail = "need " + std::to_string(profile.pipeline_parallel_size) +
                     " idle compatible Linux nodes; " + std::to_string(eligible) +
                     " currently eligible";
        }
        res.set_content(nlohmann::json{{"configured", configured},
                                       {"required", required},
                                       {"state", state},
                                       {"detail", detail},
                                       {"tensor_parallel_size", profile.tensor_parallel_size},
                                       {"pipeline_parallel_size", profile.pipeline_parallel_size},
                                       {"eligible_node_count", eligible},
                                       {"eligible_nodes", eligible_nodes},
                                       {"eligible_group_count", eligible_groups.size()},
                                       {"active_groups", active},
                                       {"groups", groups},
                                       {"allow_experimental_gloo",
                                        profile.allow_experimental_gloo}}
                            .dump(),
                        "application/json");
    });

    server_->Get("/v1/cluster/engines/conformance", [this](const Request&, Response& res) {
        if (engine_config_ == nullptr) {
            res.status = 503;
            res.set_content(R"({"error":"engine configuration is not available"})",
                            "application/json");
            return;
        }
        const auto cfg = engine_config_->get();
        nlohmann::json nodes = nlohmann::json::array();
        int conforming = 0;
        for (const auto& n : registry_.list_nodes()) {
            const bool ok = conformance_permits_placement(n.conformance);
            const bool wire_ok = node_wire_compatible(n);
            if (ok) ++conforming;
            nodes.push_back(nlohmann::json{
                {"node_id", n.id},
                {"hostname", n.hostname},
                {"url", n.url},
                {"connected", n.connected},
                {"conformance", n.conformance},
                {"engine_config_version", n.engine_config_version},
                {"engines", n.engines},
                {"wire_protocol", n.wire_protocol},
                // Stated per node, because "why is nothing scheduling here" is
                // the question this route exists to answer — and a node that
                // speaks an older wire protocol is one answer to it.
                {"placement_eligible", n.connected && ok && wire_ok},
                {"placement_blocked_reason",
                 !n.connected ? std::string("disconnected")
                 : !wire_ok   ? "node speaks wire protocol " + std::to_string(n.wire_protocol) +
                                    "; control speaks " + std::to_string(kNodeWireProtocol)
                 : !ok        ? std::string("engine configuration not conforming")
                              : std::string{}},
                {"action_progress", n.action_progress}});
        }
        res.set_content(nlohmann::json{{"config_version", cfg.version},
                                       {"primary_engine", cfg.primary_engine},
                                       {"backup_engine", cfg.backup_engine},
                                       {"share_builds", cfg.share_builds},
                                       {"nodes", nodes},
                                       {"conforming", conforming},
                                       {"total", nodes.size()}}
                            .dump(),
                        "application/json");
    });

    server_->Post("/v1/cluster/engines/resync", [this](const Request&, Response& res) {
        if (engine_config_ == nullptr || !engine_config_->configured()) {
            res.status = 409;
            res.set_content(R"({"error":"no cluster engine configuration to resync"})",
                            "application/json");
            return;
        }
        // Forces a push to every node whose version differs, without waiting for
        // the next health poll. The operator-visible half of convergence.
        const auto cfg = engine_config_->get();
        registry_.push_engine_config_to_all(cfg);
        res.set_content(nlohmann::json{{"resynced", true}, {"config_version", cfg.version}}.dump(),
                        "application/json");
    });

    // POST /v1/cluster/engines/share — broker one artifact between two nodes.
    //
    // Control decides WHO shares with WHOM, relays the digest, mints and
    // revokes the credential, and never touches the bytes. The sequence is not
    // rearrangeable:
    //
    //   1. ask the SOURCE to package and hash — it does not send anything
    //   2. announce that digest to the TARGET, over control's own channel
    //   3. mint a transfer credential into the TARGET
    //   4. tell the SOURCE to push the package it already hashed
    //   5. revoke the credential — on success AND on failure
    //
    // Steps 1-3 before 4 are the security argument, and the guarantee is worth
    // stating exactly: the digest the target will accept is fixed by an
    // AUTHENTICATED peer before the transfer credential exists, so a credential
    // that leaks cannot be used to push different bytes. It does NOT make a
    // compromised source safe — a source that lies in step 1 supplies both the
    // artifact and the hash that validates it. Signed builds would answer that;
    // this is not one, and pretending otherwise would be the theatre this
    // ordering exists to avoid.
    server_->Post("/v1/cluster/engines/share", [this](const Request& req, Response& res) {
        std::string fingerprint, source_id, target_id;
        try {
            const auto j = nlohmann::json::parse(req.body);
            fingerprint = j.value("fingerprint", std::string{});
            source_id = j.value("source_node_id", std::string{});
            target_id = j.value("target_node_id", std::string{});
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        if (fingerprint.empty() || target_id.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"fingerprint and target_node_id are required"})",
                            "application/json");
            return;
        }

        NodeInfo source{}, target{};
        bool have_source = false, have_target = false;
        for (const auto& n : registry_.list_nodes()) {
            if (n.id == target_id) { target = n; have_target = true; }
            if (!source_id.empty() && n.id == source_id) { source = n; have_source = true; }
        }
        if (!have_target) {
            res.status = 404;
            res.set_content(R"({"error":"unknown target_node_id"})", "application/json");
            return;
        }
        // With no source named, control picks one that advertises the exact
        // fingerprint. Exact: a near-match is a wrong binary.
        if (!have_source) {
            for (const auto& n : registry_.list_nodes()) {
                if (n.id == target_id || !n.connected) continue;
                for (const auto& e : n.engines) {
                    EngineArtifact a;
                    a.engine_id = e.engine_id;
                    a.version = e.version;
                    a.platform = n.platform;
                    a.arch = n.capabilities.arch;
                    a.variant = e.variant;
                    if (e.ready && a.fingerprint() == fingerprint) {
                        source = n;
                        have_source = true;
                        break;
                    }
                }
                if (have_source) break;
            }
        }
        if (!have_source) {
            res.status = 404;
            res.set_content(
                nlohmann::json{{"error", "no connected node advertises " + fingerprint}}.dump(),
                "application/json");
            return;
        }

        // ── 1. package and hash at the source ─────────────────────────────────
        HttpClient source_client(source.url);
        if (!source.api_key.empty()) source_client.set_bearer_token(source.api_key);
        // Packaging a CUDA build is a multi-hundred-megabyte tar; the default
        // 30 s read would time out on the prepare call alone.
        source_client.set_timeouts(10, 3600, 3600);

        const auto prepared = source_client.post("/api/node/engines/prepare",
                                                 nlohmann::json{{"fingerprint", fingerprint}});
        if (!prepared.ok()) {
            res.status = 502;
            res.set_content(nlohmann::json{{"error", "source could not package the artifact"},
                                           {"source_node_id", source.id},
                                           {"source_status", prepared.status},
                                           {"source_body", prepared.body}}
                                .dump(),
                            "application/json");
            return;
        }
        std::string digest, package_token;
        try {
            const auto pj = nlohmann::json::parse(prepared.body);
            digest = pj.value("sha256", std::string{});
            package_token = pj.value("package_token", std::string{});
        } catch (...) {
        }
        if (digest.empty() || package_token.empty()) {
            res.status = 502;
            res.set_content(R"({"error":"source returned no digest for the artifact"})",
                            "application/json");
            return;
        }

        // ── 2. tell the target what to expect, BEFORE any credential exists ───
        HttpClient target_client(target.url);
        if (!target.api_key.empty()) target_client.set_bearer_token(target.api_key);
        if (!target_client
                 .post("/api/node/engines/expect",
                       nlohmann::json{{"fingerprint", fingerprint}, {"sha256", digest}})
                 .ok()) {
            res.status = 502;
            res.set_content(R"({"error":"target node would not record the expected digest"})",
                            "application/json");
            return;
        }

        // ── 3. mint a scoped credential into the target ───────────────────────
        // Scoped to one node and revoked below. A leaked long-lived node key is
        // a far worse outcome than a rebuild.
        const std::string transfer_key = "mmxfer-" + pairing::generate_nonce(24);
        auto revoke = [&]() {
            HttpClient c(target.url);
            if (!target.api_key.empty()) c.set_bearer_token(target.api_key);
            c.del("/api/node/api-keys/" + transfer_key);
        };

        if (!target_client.post("/api/node/api-keys",
                                nlohmann::json{{"key", transfer_key}}).ok()) {
            res.status = 502;
            res.set_content(R"({"error":"target node refused the transfer credential"})",
                            "application/json");
            return;
        }

        // ── 4. push, then 5. revoke unconditionally ───────────────────────────
        const auto push = source_client.post("/api/node/engines/share",
                                             nlohmann::json{{"package_token", package_token},
                                                            {"fingerprint", fingerprint},
                                                            {"target_url", target.url},
                                                            {"transfer_key", transfer_key}});
        revoke();

        if (!push.ok()) {
            res.status = 502;
            res.set_content(nlohmann::json{{"error", "share failed"},
                                           {"source_node_id", source.id},
                                           {"source_status", push.status},
                                           {"source_body", push.body}}
                                .dump(),
                            "application/json");
            return;
        }
        publish_activity(0, "Shared engine artifact " + fingerprint + " from " + source.id +
                                " to " + target.id);
        res.set_content(nlohmann::json{{"shared", true},
                                       {"fingerprint", fingerprint},
                                       {"source_node_id", source.id},
                                       {"target_node_id", target.id}}
                            .dump(),
                        "application/json");
    });

    // ── Per-node engine actions ──────────────────────────────────────────────
    //
    // The gap these close: the node has had provision / check-update / switch /
    // diagnose / recover since the engine work landed, and NOTHING on control
    // ever called them. The wizard that drives them lived only in the node's own
    // TUI, wired in-process, so recovering a node meant opening a session on
    // that machine — on a cluster head whose entire premise is that you do not
    // have to.
    //
    // Worse, the one lever control DID offer was misleading. "Resync nodes"
    // skips any node already at the current config version, and a node bumps its
    // version when it ACCEPTS a config, not when it conforms to one. A node that
    // accepted v3 and then failed its build sits at v3 forever: resync reports
    // success and does nothing. These routes are the lever that actually moves
    // it.
    //
    // Addressed by node id because that is the unit an operator is looking at in
    // the conformance table, and because a node with no working engine has no
    // slot to name.
    for (const std::string action : {"provision", "check-update", "switch",
                                     "diagnose", "recover"}) {
        server_->Post("/v1/cluster/engines/nodes/:node_id/" + action,
                      [this, action](const Request& req, Response& res) {
                          // Engine id is a body field with a default rather than
                          // a path segment: every one of these actions is
                          // llama-cpp-only today, and the node says so by name
                          // when it is not. Putting it in the path would invite
                          // a caller to believe otherwise.
                          std::string engine_id = "llama-cpp";
                          if (!req.body.empty()) {
                              try {
                                  const auto j = nlohmann::json::parse(req.body);
                                  engine_id = j.value("engine_id", engine_id);
                              } catch (const std::exception& e) {
                                  res.status = 400;
                                  res.set_content(
                                      nlohmann::json{{"error", std::string("invalid JSON body: ") +
                                                                   e.what()}}
                                          .dump(),
                                      "application/json");
                                  return;
                              }
                          }
                          const auto& node_id = req.path_params.at("node_id");
                          proxy_node_engine_action(node_id, engine_id, action, req.body, res);
                          // Logged on the ACCEPTANCE only. A 409 or a 404 is not
                          // work the cluster did, and an activity line for one
                          // would read as a build that started and vanished.
                          if (res.status == 202)
                              publish_activity(0, "Engine " + action + " started on node " +
                                                      node_id + " (" + engine_id + ")");
                      });
    }

    server_->Get("/v1/engines/:id/slots", [this](const Request& req, Response& res) {
        proxy_engine_get(req.path_params.at("id"), "/sequences", {}, res);
    });

    server_->Get("/v1/engines/:id/heat", [this](const Request& req, Response& res) {
        // Bucketed by DEFAULT all the way down. The opt-in is explicit at every
        // hop rather than defaulted back to full by a middle layer.
        const auto query = req.get_param_value("resolution") == "full"
                               ? std::string("?resolution=full")
                               : std::string{};
        proxy_engine_get(req.path_params.at("id"), "/heat", query, res);
    });

    server_->Get("/v1/engines/:id/telemetry", [this](const Request& req, Response& res) {
        const auto found = find_engine_node(req.path_params.at("id"));
        if (!found) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", "no such engine"}}.dump(),
                            "application/json");
            return;
        }
        // `hz` is forwarded, not re-clamped here. The ceiling is the ENGINE's,
        // and a second clamp at this layer would be a second number to keep in
        // step with it.
        std::string query;
        if (req.has_param("hz")) query += "?hz=" + req.get_param_value("hz");
        if (req.get_param_value("resolution") == "full") {
            query += (query.empty() ? "?" : "&");
            query += "resolution=full";
        }
        const auto url = found->url;
        const auto key = found->api_key;
        const auto path = "/api/node/engines/" + req.path_params.at("id") + "/telemetry" + query;

        res.set_chunked_content_provider(
            "text/event-stream", [url, key, path](std::size_t, httplib::DataSink& sink) {
                httplib::Client cli(url);
                // cpp-httplib reads this as ZERO seconds, not "no limit": set_read_timeout(0, 0)
                // makes every read time out immediately. On an SSE proxy that presents as a
                // clean, empty 200 — headers go out, the inner Get fails at once, and the
                // client sees a stream that ends without a single frame. Measured against a
                // live engine that was streaming perfectly one hop upstream.
                //
                // A long FINITE timeout instead of trying to express infinity: a wedged
                // engine should eventually release the socket rather than pin it forever.
                cli.set_read_timeout(kSseUpstreamReadTimeout);
                if (!key.empty()) cli.set_bearer_token_auth(key);
                const bool ok = cli.Get(path.c_str(), [&](const char* data, std::size_t len) {
                    return sink.write(data, len);
                });
                if (!ok) sink.done();
                return false;
            });
    });

    // ── GET /v1/placements ──────────────────────────────────────────────────
    server_->Get("/v1/placements", [this](const Request& /*req*/, Response& res) {
        auto placements = scheduler_.list_placements();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& p : placements) {
            auto entry = nlohmann::json(p);
            // WHICH engine, and WHY. An operator could previously only infer the
            // routing decision from behaviour; the reason string is the same one
            // the scheduler acted on, not a reconstruction.
            if (const auto model = served_model_for_holder(p.holder_id)) {
                const auto routing = scheduler_.resolve_backend_for(*model);
                entry["backend"] = routing.engine_id;
                entry["backend_reason"] = routing.reason;
            }
            arr.push_back(std::move(entry));
        }
        res.set_content(arr.dump(), "application/json");
    });
}

} // namespace mm
