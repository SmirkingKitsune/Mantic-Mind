#pragma once

#include "common/models.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mm {

inline constexpr const char* kOfficialVllmRepoUrl =
    "https://github.com/vllm-project/vllm";
inline constexpr const char* kWindowsVllmRepoUrl =
    "https://github.com/SystemPanic/vllm-windows";
inline constexpr const char* kWindowsVllmBranch = "vllm-for-windows";

std::string default_vllm_repo_url_for_platform();
std::string default_vllm_branch_for_platform();

std::vector<std::string> build_vllm_server_args(const std::string& model_ref,
                                                const VllmSettings& settings,
                                                uint16_t port);

/// Load signals scraped from a vLLM engine's Prometheus /metrics endpoint.
struct VllmEngineMetrics {
    int    num_requests_running = 0;
    int    num_requests_waiting = 0;
    double kv_cache_usage       = 0.0;  // 0.0–1.0
    bool   valid                = false;
};

/// Parse Prometheus text exposition from vLLM's /metrics. Handles both the
/// legacy gpu_cache_usage_perc and the v1 kv_cache_usage_perc metric names.
VllmEngineMetrics parse_vllm_metrics_text(const std::string& text);

} // namespace mm
