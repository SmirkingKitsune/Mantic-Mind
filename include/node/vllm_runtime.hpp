#pragma once

#include "common/engine_config.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mm {

/// Pure vLLM CLI adapter; the process supervisor owns spawning and readiness.
std::vector<std::string> build_vllm_server_args(
    const std::string& model_ref,
    const std::string& served_model_name,
    const VllmEngineConfig& settings,
    std::uint16_t port);

} // namespace mm
