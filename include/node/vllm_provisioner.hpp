#pragma once

#include "common/models.hpp"
#include "node/process_exec.hpp"

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
    // Installer backend: "uv" | "pip" | "" (auto: uv when present, else pip).
    std::string package_manager;

    // Test hooks; empty means use the build host.
    std::string platform;
    std::string arch;
};

// One command in a vLLM install/upgrade plan. Steps are executed in order,
// each streaming its output live to the UI. No script file is written to disk.
struct VllmInstallStep {
    std::string              label;   // shown as the loading-bar stage
    std::vector<std::string> argv;    // executable + args (no shell)
    std::filesystem::path    cwd;     // working directory ("" = provision root)
    bool                     allow_failure = false;  // e.g. best-effort git pull
};

struct VllmCommandRunner {
    // Run a command to completion, streaming each output line to `on_line` as it
    // arrives. Returns the exit code (non-zero => failure). This is the seam
    // tests inject to avoid spawning real processes.
    using RunFn = std::function<int(const std::vector<std::string>& argv,
                                    const std::filesystem::path& cwd,
                                    const StreamLineCallback& on_line,
                                    const CancelCheckCallback& cancel_requested,
                                    std::string* error)>;
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
// Build the ordered, in-program install/upgrade plan for this environment.
std::vector<VllmInstallStep> build_vllm_install_plan(const VllmProvisionConfig& cfg,
                                                     bool upgrade = false);
// Extract a 0..1 progress fraction from a pip/uv/git output line, or <0 when the
// line carries no parseable progress. Pure; unit-tested.
double parse_vllm_install_fraction(const std::string& line);
std::string resolve_vllm_executable(const std::string& executable);
bool vllm_runtime_usable(const VllmRuntimeStatus& status);

class VllmProvisioner {
public:
    explicit VllmProvisioner(VllmProvisionConfig cfg,
                             VllmCommandRunner runner = {});

    // Sinks for visible installs: streamed output lines and coarse+fine
    // progress. Both optional; set before ensure_runtime()/update_runtime().
    using LogSink = std::function<void(const std::string& line, bool is_stderr)>;
    using ProgressSink = std::function<void(const VllmInstallProgress&)>;
    using CancelCheck = CancelCheckCallback;
    void set_log_sink(LogSink sink);
    void set_progress_sink(ProgressSink sink);
    void set_cancel_check(CancelCheck check);

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
    LogSink log_sink_;
    ProgressSink progress_sink_;
    CancelCheck cancel_check_;
    mutable std::mutex status_mutex_;
    VllmRuntimeStatus status_;

    VllmRuntimeStatus make_base_status() const;
    std::string capture_version(const std::string& executable) const;
    // Lock, run the install plan step-by-step with live streaming + progress
    // (upgrade=true adds --upgrade / git pull), then validate the managed
    // executable. Shared by ensure_runtime() (bootstrap) and update_runtime().
    VllmRuntimeStatus run_managed_install(bool upgrade);
    void emit_progress(const VllmInstallProgress& p);
    void set_status(const VllmRuntimeStatus& status);
};

} // namespace mm
