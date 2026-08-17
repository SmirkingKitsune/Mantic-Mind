#pragma once

#include "common/models.hpp"
#include "control/route_scope.hpp"
#include "control/performance_tracker.hpp"
#include "control/tts_service_client.hpp"
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <cstdint>
#include <vector>

namespace httplib { struct Request; struct Response; }

namespace mm {

class AgentManager;
class AgentQueue;
class NodeRegistry;
class AgentScheduler;
class HttpServer;
class EngineConfigStore;

// Hosts the external REST + SSE API for mantic-mind-control (see plan §REST API).
// Also hosts the node registration endpoint (/api/control/register-node).
class ControlModelRegistry;

class ControlApiServer {
public:
    struct LocalChatResult {
        bool                     success = false;
        std::string              conv_id;
        std::string              error;
        std::vector<InferenceChunk> chunks;
    };

    ControlApiServer(AgentManager& agents,
                     AgentQueue& queue,
                     NodeRegistry& registry,
                     AgentScheduler& scheduler,
                     std::string data_dir,
                     std::string models_dir,
                     std::string external_api_token = {},
                     TtsServiceConfig tts_config = {});
    ~ControlApiServer();

    bool listen(uint16_t port);
    bool listen_openai_compat(uint16_t port);
    void stop();
    void stop_openai_compat();
    void cleanup_expired_tts_cache();

    // Activity logging callback — 0=Info, 1=Warn, 2=Error.
    using LogCallback = std::function<void(int level, const std::string& message)>;
    void set_log_callback(LogCallback cb);
    void publish_activity(int level, const std::string& message);

    // In-process chat path for local tooling when loopback HTTP is unavailable.
    // max_tokens_override != 0 replaces the agent's configured max_tokens for this request.
    LocalChatResult chat_local(const AgentId& agent_id,
                               const std::string& message,
                               const ConvId& conv_id_hint = {},
                               int max_tokens_override = 0,
                               const std::vector<std::string>& attachment_ids = {});

    /// The admission registry backing /v1/models. Optional; when unset those
    /// routes answer 503.
    void set_model_registry(ControlModelRegistry* registry) {
        models_ = registry;
        // The authorizer reads api_token from the same database, so it is
        // configured here rather than by a second call a caller could forget.
        scopes_.configure(registry, external_api_token_);
    }

    /// The cluster engine configuration backing /v1/cluster/engines/*.
    /// Optional; when unset those routes answer 503 rather than reporting an
    /// unconfigured cluster, because "nobody has configured this" and "this
    /// build has no configurator" are different facts.
    void set_engine_config_store(EngineConfigStore* store) { engine_config_ = store; }

private:
    /// Why this agent cannot be sent an image, or empty if it can.
    ///
    /// TWO conditions, and only one of them used to be checked. The profile's
    /// `vision_settings.enabled` is the operator's intent; whether the ENGINE
    /// that will serve the agent can accept an image is a separate fact, and an
    /// agent with vision switched on whose model earned a streamable verdict
    /// routes to Soma, which is text-only. Nothing caught that, so the image
    /// part travelled to an engine that would never look at it (roadmap D12).
    ///
    /// One function because there are four call sites — the OpenAI-compat route,
    /// the SSE chat route, its attachment path, and the local chat helper — and
    /// four copies of a two-part rule is how the first part came to be checked
    /// everywhere and the second part nowhere.
    std::string image_refusal(const AgentConfig& cfg) const;

    AgentManager&   agents_;
    AgentQueue&     queue_;
    NodeRegistry&   registry_;
    AgentScheduler& scheduler_;
    /// Optional. When absent every /v1/models route answers 503 rather than
    /// pretending an empty registry, because "no models admitted" and "the
    /// registry never opened" are different facts and only one is actionable.
    ControlModelRegistry* models_ = nullptr;
    EngineConfigStore*    engine_config_ = nullptr;
    /// Scoped authorization for /v1/*. Replaces the flat-token comparison that
    /// used to live in authorize_external_request().
    ScopeAuthorizer scopes_;
    std::string     data_dir_;
    std::string     models_dir_;
    std::string     external_api_token_;
    TtsServiceClient tts_;
    std::unique_ptr<HttpServer> server_;
    PerformanceTracker performance_;
    std::unique_ptr<HttpServer> openai_server_;
    LogCallback     log_cb_;
    mutable std::mutex activity_mutex_;
    std::deque<nlohmann::json> activity_entries_;
    static constexpr std::size_t kMaxActivityEntries = 4000;

    using ChunkCb = std::function<void(const InferenceChunk&)>;
    using DoneCb  = std::function<void(const ConvId&, bool, const std::string&)>;

    void register_routes();
    void register_openai_compat_routes();
    bool authorize_external_request(const httplib::Request& req,
                                    httplib::Response& res) const;
    bool authorize_openai_compat_request(const httplib::Request& req,
                                         httplib::Response& res) const;

    /// Which node holds an engine (slot). Discovered rather than supplied,
    /// because the answer changes on every eviction and a client should not have
    /// to track it.
    std::optional<NodeInfo> find_engine_node(const SlotId& engine_id) const;

    /// Forward a GET to the node that holds `engine_id`. One helper because
    /// heat and slots differ only in path and query.
    void proxy_engine_get(const SlotId& engine_id,
                          const std::string& suffix,
                          const std::string& query,
                          httplib::Response& res) const;

    // Runs on the AgentQueue worker thread: builds context, routes to node,
    // proxies SSE, persists messages, fires callbacks.
    void handle_chat(const AgentId& agent_id,
                     const std::string& message,
                     const ConvId& conv_id_hint,
                     ChunkCb chunk_cb,
                     DoneCb done_cb,
                     int max_tokens_override = 0,
                     std::vector<MessageContentPart> content_parts = {});

    // Queue a global recall job for a conversation being deactivated.
    // Runs as an internal inference round where the agent reviews local
    // memories and decides what to persist as global memories.
    void queue_global_recall(const AgentId& agent_id, const ConvId& conv_id);

    void activity_log(int level, const std::string& message);
};

} // namespace mm
