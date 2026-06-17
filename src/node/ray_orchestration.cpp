#include "node/ray_orchestration.hpp"

#include "common/logger.hpp"

#include <string>
#include <vector>

#ifndef _WIN32
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace mm {

std::vector<std::string> build_ray_start_args(const RayStartConfig& cfg) {
    std::vector<std::string> args;
    args.push_back("start");

    if (cfg.role == RayRole::Head) {
        args.push_back("--head");
        args.push_back("--port=" + std::to_string(cfg.port > 0 ? cfg.port : 6379));
    } else {
        // Workers attach to the head's GCS address ("ip:port").
        args.push_back("--address=" + cfg.head_address);
    }

    if (cfg.num_gpus > 0) {
        args.push_back("--num-gpus=" + std::to_string(cfg.num_gpus));
    }

    // `ray start` returns once the node has joined; the daemons stay up. These
    // flags keep it non-interactive and quiet on a headless node.
    args.push_back("--disable-usage-stats");
    return args;
}

bool ray_supported() {
#ifdef _WIN32
    return false;
#else
    return true;
#endif
}

namespace {

#ifndef _WIN32
// Run a command to completion without a shell (no injection surface) and
// return true on exit code 0.
bool run_blocking(const std::string& exe,
                  const std::vector<std::string>& args,
                  std::string* error_out) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(exe.c_str()));
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    pid_t pid = ::fork();
    if (pid < 0) {
        if (error_out) *error_out = "fork failed";
        return false;
    }
    if (pid == 0) {
        ::execvp(exe.c_str(), argv.data());
        ::_exit(127); // execvp only returns on failure
    }

    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        if (error_out) *error_out = "waitpid failed";
        return false;
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return true;
    if (error_out) {
        *error_out = "'" + exe + " " + (args.empty() ? "" : args.front()) +
                     "' exited with status " +
                     std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    }
    return false;
}
#endif

} // namespace

bool ray_start(const RayStartConfig& cfg, std::string* error_out) {
    if (!ray_supported()) {
        if (error_out)
            *error_out = "Ray multi-node orchestration is only supported on "
                         "Linux nodes";
        return false;
    }
#ifdef _WIN32
    (void)cfg;
    return false; // unreachable: ray_supported() is false on Windows
#else
    const auto args = build_ray_start_args(cfg);
    MM_INFO("RayOrchestration: {} ray cluster (head_address={}, num_gpus={})",
            cfg.role == RayRole::Head ? "starting head of" : "joining",
            cfg.role == RayRole::Head ? "self" : cfg.head_address, cfg.num_gpus);
    const bool ok = run_blocking(cfg.ray_path, args, error_out);
    if (!ok && error_out) {
        MM_WARN("RayOrchestration: ray start failed: {}", *error_out);
    }
    return ok;
#endif
}

bool ray_stop(const std::string& ray_path, std::string* error_out) {
    if (!ray_supported()) {
        if (error_out)
            *error_out = "Ray multi-node orchestration is only supported on "
                         "Linux nodes";
        return false;
    }
#ifdef _WIN32
    (void)ray_path;
    return false;
#else
    return run_blocking(ray_path, {"stop"}, error_out);
#endif
}

} // namespace mm
