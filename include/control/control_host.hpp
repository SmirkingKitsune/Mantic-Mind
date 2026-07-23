#pragma once

#include "control/control_config.hpp"

#include <functional>
#include <memory>
#include <string>

namespace mm {

class AgentManager;
class AgentQueue;
class AgentScheduler;
class ControlApiServer;
class NodeRegistry;

// Owns the transport-neutral control service graph and its background
// lifecycle. Presentation (TUI/CLI) deliberately stays outside this class so
// the same host can be composed by the standalone and AIO entry points.
class ControlHost final {
public:
    struct Options {
        ControlConfig config;
        std::string bind_host = "0.0.0.0";
        bool enable_remote_nodes = true;
        bool enable_discovery = true;
        bool allow_legacy_environment = true;
    };

    explicit ControlHost(Options options);
    ~ControlHost();

    ControlHost(const ControlHost&) = delete;
    ControlHost& operator=(const ControlHost&) = delete;

    // AIO acquires the control lock before the node lock, then initializes
    // both roles. Standalone callers may call initialize() directly; it
    // acquires this lock automatically.
    bool acquire_singleton_lock(std::string* error = nullptr);
    bool initialize(std::string* error = nullptr);
    bool start(std::string* error = nullptr);

    // Immediately closes ingress and asks queued/active work to cancel. stop()
    // completes the drain, joins workers/listeners, destroys the graph, and
    // releases the singleton lock. Every shutdown operation is idempotent.
    void request_shutdown();
    void stop();

    bool initialized() const;
    bool running() const;
    bool listener_failed() const;
    void set_failure_callback(std::function<void(const std::string&)> callback);

    AgentManager& agents();
    NodeRegistry& registry();
    AgentScheduler& scheduler();
    AgentQueue& queue();
    ControlApiServer& api();

    const ControlConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mm
