#pragma once

#include "common/models.hpp"
#include <optional>
#include <string>

namespace mm {

class AgentScheduler;

/// Thin adapter that delegates to AgentScheduler.
/// Kept for backwards compatibility with any residual callers.
class ModelRouter {
public:
    explicit ModelRouter(AgentScheduler& scheduler);

    /// Route an agent config through the scheduler.
    /// Returns {node_id, slot_id} on success.
    struct RouteResult {
        NodeId node_id;
        SlotId slot_id;
    };
    std::optional<RouteResult> route(const AgentConfig& cfg);

    /// Legacy overload — constructs a minimal AgentConfig and delegates.
    std::optional<NodeId> route(const std::string& model_path,
                                const LlamaSettings& settings,
                                const NodeId& preferred_node_id = {});

private:
    AgentScheduler& scheduler_;
};

} // namespace mm
