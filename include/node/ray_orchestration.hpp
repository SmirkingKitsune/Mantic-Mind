#pragma once

#include "common/models.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mm {

struct RayStartConfig {
    std::string group_id;
    std::string agent_id;
    std::string role = "head";
    std::string head_address;
    std::string node_ip;
    std::string transport = "nccl";
    int num_gpus = 0;
};

std::vector<std::string> build_ray_start_args(const RayStartConfig& cfg,
                                              std::uint16_t port);
bool ray_supported() noexcept;

class RayController {
public:
    RayController(std::string executable = "ray", std::uint16_t port = 6379);

    /// Idempotent for the same group/role. A different owner is refused.
    bool start(const RayStartConfig& cfg, std::string& error);
    /// Only the owner may stop an active group. Empty group is never accepted.
    bool stop(const std::string& group_id, std::string& error);
    RayRuntimeStatus status() const;

    /// Update discovery after a managed vLLM environment is provisioned. The
    /// active owner cannot be changed underneath a running group.
    bool set_executable_if_idle(std::string executable);

private:
    std::string executable_;
    std::uint16_t port_;
    mutable std::mutex mutex_;
    RayRuntimeStatus status_;
};

} // namespace mm
