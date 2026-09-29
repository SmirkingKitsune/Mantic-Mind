#pragma once

#include "common/models.hpp"
#include "common/served_model.hpp"
#include "control/route_scope.hpp"
#include "control/performance_tracker.hpp"
#include "control/tts_service_client.hpp"
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <cstdint>
#include <vector>

namespace httplib { struct Request; struct Response; }

namespace mm {

class UserManager;
class NodeRegistry;
class PlacementScheduler;
class HttpServer;
class EngineConfigStore;
class ControlModelRegistry;
class ServedModelCatalog;
class ConversationQueue;

// Hosts the external REST + SSE API for mantic-mind-control (see
// docs/external-api.md), and the node registration endpoint
// (/api/control/register-node).
//
// Its member functions live in several translation units, by surface:
//   control_api_server.cpp   nodes, cluster, admissions, tokens, telemetry
//   served_model_routes.cpp  /v1/served-models
//   user_routes.cpp          /v1/users/…, /v1/audio/speech
//   chat_pipeline.cpp        one user turn, recall, extraction, compaction
//   openai_compat.cpp        the stateless :9091 listener
class ControlApiServer {
public:
    struct LocalChatResult {
        bool                     success = false;
        std::string              conv_id;
        std::string              model_id;
        std::string              error;
        std::string              code;
        std::vector<InferenceChunk> chunks;
    };

    ControlApiServer(UserManager& users,
                     NodeRegistry& registry,
                     PlacementScheduler& scheduler,
                     std::string data_dir,
                     std::string models_dir,
                     std::string external_api_token = {},
                     TtsServiceConfig tts_config = {},
                     std::size_t chat_workers = 16);
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

    /// In-process chat for local tooling (the CLI and the TUI): the same turn
    /// as POST /v1/users/:id/chat, on the same per-conversation queue, so a
    /// local turn cannot overlap an HTTP one on the same conversation.
    /// `model` empty: the conversation's, else the user's default.
    /// `max_tokens_override` != 0 replaces the model's max_tokens for this turn.
    LocalChatResult chat_local(const UserId& user_id,
                               const std::string& message,
                               const ConvId& conv_id = {},
                               const std::string& model = {},
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

    /// The served-model catalog backing /v1/served-models and every user turn.
    /// Optional; when unset those routes answer 503 and a turn fails with
    /// `catalog_unavailable`.
    void set_served_model_catalog(ServedModelCatalog* catalog) { served_models_ = catalog; }

private:
    /// A served model bound for one piece of work: the placement it holds (for
    /// a node model), a client to call it through, the name its engine knows it
    /// by, and what the engine it landed on can take. Defined in
    /// chat_pipeline.cpp; only the pipeline binds.
    struct BoundModel;

    /// Place (node) or connect (API) a served model. The lease the result holds
    /// keeps the placement from being evicted until the result is destroyed.
    /// Waits up to `wait` for capacity that is only temporarily missing. On
    /// failure, `error` says why and `code` is the placement failure (or
    /// `node_unavailable`).
    std::unique_ptr<BoundModel> bind_model(const ServedModel& model,
                                           std::chrono::milliseconds wait,
                                           std::string& error,
                                           std::string& code);

    /// One user turn, as the chat route and chat_local submit it. The
    /// conversation is resolved (or created) BEFORE the turn is queued, so the
    /// queue key is known and two first turns cannot create two conversations.
    struct ChatTurnRequest {
        UserId user_id;
        ConvId conv_id;
        std::string message;
        std::vector<MessageContentPart> parts;
        std::string model;            ///< the request's; empty defers
        SamplingOverrides settings;   ///< the request's; over the conversation's
        int max_tokens_override = 0;
    };
    struct TurnOutcome {
        bool ok = false;
        std::string error;
        std::string code;     ///< machine-readable, on failure
        std::string model_id; ///< the model that answered (or would have)
    };
    using ChunkCb = std::function<void(const InferenceChunk&)>;
    using TurnDoneCb = std::function<void(const ConvId&, const TurnOutcome&)>;

    /// Runs on the conversation's strand of `conversations_`.
    void run_chat_turn(const ChatTurnRequest& turn, ChunkCb chunk_cb, TurnDoneCb done_cb);

    /// The strand a conversation's work runs on.
    static std::string queue_key(const UserId& user_id, const ConvId& conv_id);

    /// Run `fn` on the conversation's strand and wait for it. False when the
    /// queue is shutting down and `fn` never ran.
    bool run_on_conversation(const UserId& user_id,
                             const ConvId& conv_id,
                             std::function<void()> fn);

    /// Review a conversation that stopped being active for global memories, on
    /// its own strand. With a model that can call tools the memory tools do the
    /// work; with one that cannot (Soma), the model answers in JSON instead.
    void queue_global_recall(const UserId& user_id, const ConvId& conv_id);
    void run_global_recall(const UserId& user_id, const ConvId& conv_id);

    /// Extract global memories from messages [begin_seq, end_seq] of a
    /// conversation with `model`, on the conversation's strand.
    void queue_extraction(const UserId& user_id,
                          const ConvId& conv_id,
                          int begin_seq,
                          int end_seq,
                          const ServedModel& model);

    struct CompactOutcome {
        int status = 200;
        std::string state;  ///< CompactionResult status, or the failure
        std::string error;
        std::string code;
        std::string model_id;
    };
    /// A forced fold of one conversation, on its strand. `model` empty: the
    /// conversation's, else the user's default.
    CompactOutcome compact_now(const UserId& user_id,
                               const ConvId& conv_id,
                               const std::string& model);

    /// Model-designed voice proposal for a user. `model` empty: the user's
    /// default. nullopt with `error` on failure.
    std::optional<VoiceDesignProposal> infer_voice_design(const UserId& user_id,
                                                          const std::string& model,
                                                          std::string& error);

    /// The catalog entry a model id resolves to, for a turn or a background job.
    std::optional<ServedModel> available_model(const std::string& id, std::string& error,
                                               std::string& code) const;

    /// The served model a placement holder stands for, when it still exists.
    std::optional<ServedModel> served_model_for_holder(const HolderId& holder_id) const;

    UserManager&    users_;
    NodeRegistry&   registry_;
    PlacementScheduler& scheduler_;
    /// Optional. When absent every /v1/models route answers 503 rather than
    /// pretending an empty registry, because "no models admitted" and "the
    /// registry never opened" are different facts and only one is actionable.
    ControlModelRegistry* models_ = nullptr;
    ServedModelCatalog*   served_models_ = nullptr;
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
    /// Every user's work: turns, recall, extraction, compaction, deletes.
    /// One strand per conversation on a bounded pool (chat_worker_threads).
    std::unique_ptr<ConversationQueue> conversations_;
    LogCallback     log_cb_;
    mutable std::mutex activity_mutex_;
    std::deque<nlohmann::json> activity_entries_;
    static constexpr std::size_t kMaxActivityEntries = 4000;

    void register_routes();
    void register_served_model_routes(); ///< served_model_routes.cpp
    void register_user_routes();         ///< user_routes.cpp
    void register_openai_compat_routes(); ///< openai_compat.cpp

    /// How long a request waits for placement capacity that is only
    /// temporarily missing. MM_CHAT_QUEUE_WAIT_MS overrides three minutes.
    static std::chrono::milliseconds chat_queue_wait();

    /// Whether the cluster's vLLM launches with auto tool choice — the one
    /// engine where tool support is configuration rather than software.
    bool vllm_tool_choice() const;

    /// The pool :9091's streaming completions run on, created on first use.
    ConversationQueue& openai_queue();
    std::once_flag openai_queue_once_;
    std::unique_ptr<ConversationQueue> openai_queue_;
    std::size_t openai_workers_ = 16;
    bool authorize_external_request(const httplib::Request& req,
                                    httplib::Response& res) const;
    bool authorize_openai_compat_request(const httplib::Request& req,
                                         httplib::Response& res) const;

    /// The user a request's token is bound to; empty for an unbound token, the
    /// legacy token, or auth-off.
    std::string bound_user_of(const httplib::Request& req) const;

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

    /// Forward one engine ACTION to a named node.
    ///
    /// Addressed by NODE, not by engine: provisioning is what a node does when
    /// it has no engine yet, so `find_engine_node` — which searches live slots —
    /// would fail on precisely the node an operator needs to fix. That
    /// asymmetry is why this is a second helper rather than a verb on the
    /// first.
    ///
    /// The node answers 202/409 and returns immediately (see
    /// NodeApiServer's llama action callbacks), so this holds no connection
    /// across a build and needs no long timeout. Status and body are passed
    /// through verbatim: a 400 saying "this action exists only for llama-cpp"
    /// is the node's sentence to say, and rewording it here would give an
    /// operator a second vocabulary for one fault.
    void proxy_node_engine_action(const NodeId& node_id,
                                  const std::string& engine_id,
                                  const std::string& action,
                                  const std::string& body,
                                  httplib::Response& res) const;

    void activity_log(int level, const std::string& message);
};

} // namespace mm
