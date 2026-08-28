#include "node/ray_orchestration.hpp"

#include "common/process_exec.hpp"
#include "common/util.hpp"

namespace mm {

std::vector<std::string> build_ray_start_args(const RayStartConfig& cfg,
                                              std::uint16_t port) {
    std::vector<std::string> args{"start"};
    if (cfg.role == "head") {
        args.emplace_back("--head");
        args.emplace_back("--port=" + std::to_string(port));
    } else {
        args.emplace_back("--address=" + cfg.head_address);
    }
    if (!cfg.node_ip.empty())
        args.emplace_back("--node-ip-address=" + cfg.node_ip);
    if (cfg.num_gpus > 0)
        args.emplace_back("--num-gpus=" + std::to_string(cfg.num_gpus));
    args.emplace_back("--disable-usage-stats");
    return args;
}

bool ray_supported() noexcept {
#if defined(_WIN32) || defined(__APPLE__)
    return false;
#else
    return true;
#endif
}

RayController::RayController(std::string executable, std::uint16_t port)
    : executable_(executable.empty() ? "ray" : std::move(executable)),
      port_(port == 0 ? 6379 : port) {}

bool RayController::start(const RayStartConfig& cfg, std::string& error) {
    error.clear();
    if (!ray_supported()) {
        error = "Ray multi-node orchestration is supported only on Linux nodes";
        return false;
    }
    if (cfg.group_id.empty() || (cfg.role != "head" && cfg.role != "worker")) {
        error = "group_id and role=head|worker are required";
        return false;
    }
    if (cfg.role == "worker" && cfg.head_address.empty()) {
        error = "head_address is required for a Ray worker";
        return false;
    }

    {
        std::lock_guard<std::mutex> g(mutex_);
        if (!status_.group_id.empty()) {
            const std::string expected_head = cfg.role == "head"
                ? cfg.node_ip + ":" + std::to_string(port_)
                : cfg.head_address;
            const bool same_start = status_.group_id == cfg.group_id &&
                status_.agent_id == cfg.agent_id && status_.role == cfg.role &&
                status_.head_address == expected_head &&
                status_.transport == cfg.transport &&
                status_.reserved_gpus == cfg.num_gpus;
            if (same_start && status_.active()) return true;
            if (same_start && status_.state == "starting") {
                error = "Ray group start is already in progress";
                return false;
            }
            if (same_start && status_.state == "error") {
                error = status_.last_error.empty()
                    ? "Ray group start previously failed" : status_.last_error;
                return false;
            }
            error = status_.group_id == cfg.group_id
                ? "Ray group id is already bound to different start parameters"
                : "node is reserved by Ray group '" + status_.group_id + "'";
            return false;
        }
        status_.state = "starting";
        status_.group_id = cfg.group_id;
        status_.agent_id = cfg.agent_id;
        status_.role = cfg.role;
        status_.head_address = cfg.role == "head"
            ? cfg.node_ip + ":" + std::to_string(port_) : cfg.head_address;
        status_.transport = cfg.transport;
        status_.reserved_gpus = cfg.num_gpus;
        status_.last_error.clear();
    }

    auto args = build_ray_start_args(cfg, port_);
    args.insert(args.begin(), executable_);
    std::string command_error;
    const int rc = run_streamed_command(args, {}, {}, &command_error);
    std::lock_guard<std::mutex> g(mutex_);
    if (rc != 0) {
        status_.state = "error";
        status_.last_error = command_error.empty()
            ? "ray start exited with status " + std::to_string(rc) : command_error;
        error = status_.last_error;
        return false;
    }
    status_.state = "active";
    return true;
}

bool RayController::stop(const std::string& group_id, std::string& error) {
    error.clear();
    {
        std::lock_guard<std::mutex> g(mutex_);
        if (group_id.empty()) {
            error = "group_id is required";
            return false;
        }
        if (status_.group_id.empty()) return true;
        if (status_.group_id != group_id) {
            error = "Ray group ownership mismatch: active='" + status_.group_id + "'";
            return false;
        }
    }
    std::string command_error;
    const int rc = run_streamed_command(
        {executable_, "stop", "--force"}, {}, {}, &command_error);
    std::lock_guard<std::mutex> g(mutex_);
    if (rc != 0) {
        status_.state = "error";
        status_.last_error = command_error.empty()
            ? "ray stop exited with status " + std::to_string(rc) : command_error;
        error = status_.last_error;
        return false;
    }
    status_ = RayRuntimeStatus{};
    return true;
}

RayRuntimeStatus RayController::status() const {
    std::lock_guard<std::mutex> g(mutex_);
    return status_;
}

bool RayController::set_executable_if_idle(std::string executable) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!status_.group_id.empty() || executable.empty()) return false;
    executable_ = std::move(executable);
    return true;
}

} // namespace mm
