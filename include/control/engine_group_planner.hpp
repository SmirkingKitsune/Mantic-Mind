#pragma once

#include "common/engine_config.hpp"
#include "common/models.hpp"

#include <optional>
#include <string>
#include <vector>

namespace mm {

struct EngineGroupRequest {
    std::string model_ref;
    int tensor_parallel_size = 1;
    int pipeline_parallel_size = 1;
    std::uint32_t config_version = 0;
    bool allow_experimental_gloo = false;
};

struct EngineGroupCandidate {
    std::vector<NodeId> nodes; ///< head first
    std::string transport;
    std::string runtime_fingerprint;
    double score = 0.0;
    bool experimental = false;
    std::string reason;
};

bool distributed_vllm_model_ref_supported(const std::string& model_ref,
                                          std::string& reason);
std::vector<EngineGroupCandidate> plan_engine_groups(
    const EngineGroupRequest& request,
    const std::vector<NodeInfo>& nodes);
std::optional<EngineGroupCandidate> best_engine_group(
    const EngineGroupRequest& request,
    const std::vector<NodeInfo>& nodes);

} // namespace mm
