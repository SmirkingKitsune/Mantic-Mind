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
inline constexpr const char* kMetalVllmRepoUrl =
    "https://github.com/vllm-project/vllm-metal";

std::string current_vllm_platform();
std::string current_vllm_arch();
bool is_apple_silicon_environment(const std::string& platform,
                                  const std::string& arch);

// The accelerator-correct vLLM build variant for this environment:
//   metal   — Apple Silicon macOS
//   windows — Windows (SystemPanic build; accelerator baked into that wheel)
//   cuda    — Linux with an NVIDIA/CUDA GPU
//   rocm    — Linux with an AMD/ROCm GPU
//   cpu     — Linux with no GPU backend
std::string detect_vllm_accelerator(const std::string& platform,
                                    const std::string& arch,
                                    bool has_cuda,
                                    bool has_rocm);

// Best-effort ROCm probe (POSIX only): true when an AMD ROCm stack is present
// (/opt/rocm exists or `rocminfo` is on PATH). Always false off POSIX.
bool detect_rocm_present();

// Compare two vLLM version strings by their leading dotted-numeric core
// (e.g. "0.6.3.dev1+g abc" -> {0,6,3}). Returns <0 if a<b, 0 if equal, >0 if
// a>b. Unparseable input compares equal so an unknown version never reports a
// spurious update.
int compare_vllm_versions(const std::string& a, const std::string& b);

std::string default_vllm_repo_url_for_environment(const std::string& platform,
                                                  const std::string& arch);
std::string default_vllm_branch_for_environment(const std::string& platform,
                                                const std::string& arch);
std::string default_vllm_repo_url_for_platform();
std::string default_vllm_branch_for_platform();

std::vector<std::string> build_vllm_server_args(const std::string& model_ref,
                                                const VllmSettings& settings,
                                                uint16_t port);

// True when a model reference points at a GGUF file (case-insensitive .gguf).
// vLLM's GGUF support is experimental and architecture-limited, so callers use
// this to attach a diagnostic advisory at launch. Pure.
bool model_ref_is_gguf(const std::string& model_ref);

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
