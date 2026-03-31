#pragma once

#include "common/models.hpp"
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <cstdint>

namespace mm {

class AgentManager;
class AgentQueue;
class NodeRegistry;
class ModelRouter;
class AgentScheduler;
class HttpServer;

// Hosts the external REST + SSE API for mantic-mind-control (see plan §REST API).
// Also hosts the node registration endpoint (/api/control/register-node).
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
                     ModelRouter& router,
                     AgentScheduler& scheduler,
                     std::string models_dir);
    ~ControlApiServer();

    bool listen(uint16_t port);
    void stop();

    // Activity logging callback — 0=Info, 1=Warn, 2=Error.
    using LogCallback = std::function<void(int level, const std::string& message)>;
    void set_log_callback(LogCallback cb);
    void publish_activity(int level, const std::string& message);

    // In-process chat path for local tooling when loopback HTTP is unavailable.
    // max_tokens_override != 0 replaces the agent's configured max_tokens for this request.
    LocalChatResult chat_local(const AgentId& agent_id,
                               const std::string& message,
                               const ConvId& conv_id_hint = {},
                               int max_tokens_override = 0);

private:
    AgentManager&   agents_;
    AgentQueue&     queue_;
    NodeRegistry&   registry_;
    ModelRouter&    router_;
    AgentScheduler& scheduler_;
    std::string     models_dir_;
    std::unique_ptr<HttpServer> server_;
    LogCallback     log_cb_;
    mutable std::mutex activity_mutex_;
    std::deque<nlohmann::json> activity_entries_;
    static constexpr std::size_t kMaxActivityEntries = 4000;

    using ChunkCb = std::function<void(const InferenceChunk&)>;
    using DoneCb  = std::function<void(const ConvId&, bool, const std::string&)>;

    void register_routes();

    // Runs on the AgentQueue worker thread: builds context, routes to node,
    // proxies SSE, persists messages, fires callbacks.
    void handle_chat(const AgentId& agent_id,
                     const std::string& message,
                     const ConvId& conv_id_hint,
                     ChunkCb chunk_cb,
                     DoneCb done_cb,
                     int max_tokens_override = 0);

    // Queue a global recall job for a conversation being deactivated.
    // Runs as an internal inference round where the agent reviews local
    // memories and decides what to persist as global memories.
    void queue_global_recall(const AgentId& agent_id, const ConvId& conv_id);

    void activity_log(int level, const std::string& message);
};

} // namespace mm
