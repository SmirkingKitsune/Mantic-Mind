#include "control/model_router.hpp"
#include "control/agent_scheduler.hpp"
#include "common/logger.hpp"

namespace mm {

ModelRouter::ModelRouter(AgentScheduler& scheduler) : scheduler_(scheduler) {}

std::optional<ModelRouter::RouteResult> ModelRouter::route(const AgentConfig& cfg) {
    auto result = scheduler_.ensure_agent_running(cfg);
    if (!result) return std::nullopt;
    return RouteResult{result->node_id, result->slot_id};
}

std::optional<NodeId> ModelRouter::route(const std::string& model_path,
                                          const LlamaSettings& settings,
                                          const NodeId& preferred_node_id) {
    // Build a minimal AgentConfig for the legacy interface.
    AgentConfig cfg;
    cfg.id                = "legacy-route";
    cfg.model_path        = model_path;
    cfg.llama_settings    = settings;
    cfg.preferred_node_id = preferred_node_id;

    auto result = scheduler_.ensure_agent_running(cfg);
    if (!result) return std::nullopt;
    return result->node_id;
}

} // namespace mm
