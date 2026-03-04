#pragma once

#include "common/models.hpp"

#include <string>

namespace mm {

class NodeRegistry;

AgentValidationResult validate_agent_config(const AgentConfig& cfg,
                                            const NodeRegistry* registry,
                                            const std::string& models_dir,
                                            const ModelCapabilityInfo* precomputed_model_info = nullptr);

} // namespace mm
