#pragma once

#include "common/models.hpp"

#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace mm {

struct VllmProvisionConfig {
    std::string requested_executable = "vllm";
    std::string provision_dir;
    bool        auto_provision = true;
    std::string install_method = "auto"; // auto|wheel|source
    std::string version = "latest";
    std::string python_path;
    // Accelerator-correct build variant: cuda|rocm|cpu|metal|windows. Empty =>
    // let the install fall back to torch-backend auto-detection.
    std::string accelerator;

    // Test hooks; empty means use the build host.
    std::string platform;
    std::string arch;
};

struct VllmCommandRunner {
    using RunFn = std::function<int(const std::vector<std::string>&,
                                    const std::filesystem::path&,
                                    std::string*)>;
    using CaptureFn = std::function<std::string(const std::vector<std::string>&,
                                                const std::filesystem::path&)>;
    // Return the newest vLLM version available upstream for this environment, or
    // "" when it cannot be determined (offline, tool missing, etc.).
    using FetchLatestFn = std::function<std::string(const VllmProvisionConfig&)>;
    RunFn run;
    CaptureFn capture_first_line;
    FetchLatestFn fetch_latest;
};

std::string normalize_vllm_install_method(const std::string& method);
std::filesystem::path managed_vllm_executable_path(const VllmProvisionConfig& cfg);
std::string build_vllm_provision_script(const VllmProvisionConfig& cfg, bool upgrade = false);
std::string resolve_vllm_executable(const std::string& executable);
bool vllm_runtime_usable(const VllmRuntimeStatus& status);

class VllmProvisioner {
public:
    explicit VllmProvisioner(VllmProvisionConfig cfg,
                             VllmCommandRunner runner = {});

    VllmRuntimeStatus ensure_runtime();
    // Query upstream for the newest available build and update latest_version /
    // update_available on the cached status. Never installs anything.
    VllmRuntimeStatus check_for_update();
    // Force a managed (re)install/upgrade to the target version — the user-
    // approved update action. Refuses when vLLM is resolved from PATH (the node
    // does not own that environment).
    VllmRuntimeStatus update_runtime();
    VllmRuntimeStatus status() const;
    VllmProvisionConfig config() const;

private:
    VllmProvisionConfig cfg_;
    VllmCommandRunner runner_;
    mutable std::mutex status_mutex_;
    VllmRuntimeStatus status_;

    VllmRuntimeStatus make_base_status() const;
    std::string capture_version(const std::string& executable) const;
    int run_script(const std::filesystem::path& script_path, std::string* error);
    // Lock, write the install script (upgrade=true adds --upgrade / git pull),
    // run it, and validate the managed executable. Shared by ensure_runtime()
    // (bootstrap) and update_runtime() (forced upgrade).
    VllmRuntimeStatus run_managed_install(bool upgrade);
    void set_status(const VllmRuntimeStatus& status);
};

} // namespace mm
