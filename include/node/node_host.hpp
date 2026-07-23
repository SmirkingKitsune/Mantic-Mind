#pragma once

#include "common/models.hpp"
#include "node/node_config.hpp"

#include <memory>
#include <string>

namespace mm {

class ModelStore;
class NodeService;
class NodeState;
class SlotManager;

// Owns the process-local node service graph shared by the standalone node and
// AIO. Network listeners, registration/discovery, and presentation remain in
// their respective shells.
class NodeHost final {
public:
    struct Options {
        NodeConfig config;
        NodeId node_id;
        bool registered = false;
        bool mark_control_contact = false;

        // Standalone nodes manage a transferred-model cache. AIO points slots
        // directly at control-owned model paths and must leave this false.
        bool manage_model_cache = false;

        int metrics_interval_ms = 2000;
        int initial_metrics_timeout_ms = 5000;
        std::string singleton_lock_name = "mantic-mind-node";
    };

    explicit NodeHost(Options options);
    ~NodeHost();

    NodeHost(const NodeHost&) = delete;
    NodeHost& operator=(const NodeHost&) = delete;

    // Two-phase locking lets AIO retain control-lock -> node-lock ordering.
    // initialize() acquires the node lock automatically for standalone use.
    bool acquire_singleton_lock(std::string* error = nullptr);
    bool initialize(std::string* error = nullptr);

    // request_shutdown() is non-destructive and may be called while users are
    // draining. Before stop(), callers must stop/join NodeApiServer inference
    // or drain LocalNodeOperations/ControlHost; stop() force-unloads slots.
    void request_shutdown();
    void stop();

    bool initialized() const;
    bool initial_metrics_ready() const;

    NodeState& state();
    SlotManager& slots();
    NodeService& service();
    ModelStore* model_store();
    const NodeConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mm
