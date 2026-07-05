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
    RunFn run;
    CaptureFn capture_first_line;
};

std::string normalize_vllm_install_method(const std::string& method);
std::filesystem::path managed_vllm_executable_path(const VllmProvisionConfig& cfg);
std::string build_vllm_provision_script(const VllmProvisionConfig& cfg);
std::string resolve_vllm_executable(const std::string& executable);
bool vllm_runtime_usable(const VllmRuntimeStatus& status);

class VllmProvisioner {
public:
    explicit VllmProvisioner(VllmProvisionConfig cfg,
                             VllmCommandRunner runner = {});

    VllmRuntimeStatus ensure_runtime();
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
    void set_status(const VllmRuntimeStatus& status);
};

} // namespace mm
