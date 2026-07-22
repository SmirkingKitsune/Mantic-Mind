#pragma once

#include "common/models.hpp"

#include <string>

namespace mm {

class NodeRegistry;

AgentValidationResult validate_agent_config(const AgentConfig& cfg,
                                            const NodeRegistry* registry = nullptr);

} // namespace mm
