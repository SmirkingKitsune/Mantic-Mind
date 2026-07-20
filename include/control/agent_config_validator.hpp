#pragma once

#include "common/models.hpp"

#include <string>
#include <vector>

namespace mm {

class NodeRegistry;

AgentValidationResult validate_agent_config(const AgentConfig& cfg,
                                            const NodeRegistry* registry,
                                            const std::string& models_dir,
                                            const ModelCapabilityInfo* precomputed_model_info = nullptr);

// Case-insensitive adjacent mmproj-*.gguf candidates for profile-editor hints.
// Candidates are suggestions only; callers must never auto-select one.
std::vector<std::string> suggest_mmproj_files(const std::string& model_path);

} // namespace mm
