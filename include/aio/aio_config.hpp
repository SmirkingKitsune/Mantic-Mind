#pragma once

#include "control/control_config.hpp"
#include "node/node_config.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mm {

// Controls whether the AIO process may perform runtime-management operations
// that need network access (for example, resolving or downloading llama.cpp).
// Prompt is deliberately the default so first-run provisioning is not silent.
enum class AioRuntimeNetworkPolicy {
    Prompt,
    Allow,
    Deny,
};

std::string_view to_string(AioRuntimeNetworkPolicy policy) noexcept;
std::optional<AioRuntimeNetworkPolicy> aio_runtime_network_policy_from_string(
    std::string_view value);

struct AioSharedConfig {
    std::string data_dir = "data";
    std::string models_dir = "models";
    std::string log_file = "logs/mantic-mind-aio.log";
};

// The AIO control server is the only network-facing application server. The
// same host is used for both the control and optional OpenAI-compatible ports.
struct AioControlConfig {
    std::string bind_host = "127.0.0.1";
    uint16_t listen_port = 9090;
    uint16_t openai_compat_port = 9091;
    uint32_t node_health_poll_interval_s = 30;
    uint32_t node_offline_after_s = 90;
    std::string external_api_token;
    TtsServiceConfig tts;
};

// Only non-network NodeConfig settings are represented here. The embedded node
// has no bind address, API port, advertised URL, API key, registration, or
// discovery surface.
struct AioNodeConfig {
    std::string llama_server_path = "llama-server";
    bool llama_auto_provision = true;
    std::string llama_provision_dir;
    std::string llama_install_method = "auto";
    std::string llama_version = "latest";
    std::string llama_accelerator;
    std::string llama_cuda_arch;
    std::vector<std::string> llama_cmake_args;
    int llama_build_jobs = 0;
    std::string llama_update_policy = "prompt";
    bool llama_update_check = true;
    int llama_update_check_interval_hours = 24;
    AioRuntimeNetworkPolicy runtime_network_policy =
        AioRuntimeNetworkPolicy::Prompt;

    int node_gpu_count = 0;
    uint16_t runtime_port_range_start = 8080;
    uint16_t runtime_port_range_end = 8090;
    int max_slots = 4;

    std::string kv_cache_dir;
    int64_t model_cache_min_free_mb = 10240;
    bool model_cache_clear_on_shutdown = true;
};

struct AioClusterConfig {
    bool enabled = false;
    bool discovery_enabled = false;
    uint16_t discovery_port = 7072;
    std::string pairing_key;
};

struct AioConfig {
    AioSharedConfig shared;
    AioControlConfig control;
    AioNodeConfig node;
    AioClusterConfig cluster;
};

struct AioConfigIssue {
    // File path, environment variable, or "validation".
    std::string source;
    // Canonical dotted key when applicable, such as "control.listen_port".
    std::string key;
    std::string message;
};

struct AioConfigLoadOptions {
    // The caller maps --config to this field. If supplied, a missing or invalid
    // file is an error and lower-precedence sources are not consulted.
    std::optional<std::filesystem::path> explicit_path;
    // Empty means std::filesystem::current_path().
    std::filesystem::path search_start;
    std::size_t upward_search_limit = std::numeric_limits<std::size_t>::max();
    bool use_environment = true;
};

struct AioConfigLoadResult {
    AioConfig config;
    // Empty means defaults/environment only; no file was found in the upward
    // search. Explicit and MM_AIO_CONFIG_FILE paths are always reported here.
    std::filesystem::path source_path;
    std::vector<AioConfigIssue> issues;

    bool ok() const noexcept { return issues.empty(); }
};

// File precedence is:
//   options.explicit_path > MM_AIO_CONFIG_FILE > upward mantic-mind-aio.toml
// File values are then overridden by MM_AIO_<SECTION>_<KEY> variables.
AioConfigLoadResult load_aio_config(const AioConfigLoadOptions& options = {});

std::vector<AioConfigIssue> validate_aio_config(const AioConfig& config);

// These materializers apply the namespaced AIO settings to the legacy role
// configs consumed by the extracted services.
ControlConfig make_control_config(const AioConfig& config);
NodeConfig make_node_config(const AioConfig& config);

// Public for startup validation/tests and useful to callers deciding whether a
// configured listener requires credentials. Accepts host text or an HTTP URL.
bool aio_host_is_loopback(std::string_view host_or_url);

} // namespace mm
