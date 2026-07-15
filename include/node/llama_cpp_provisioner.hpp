#pragma once

#include "common/models.hpp"
#include "node/process_exec.hpp"

#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace mm {

// Node-managed llama.cpp (llama-server) runtime, the llama analog of
// VllmProvisioner. It resolves an existing llama-server from PATH / a managed
// install, then prefers an environment-matched official GitHub release. A
// source build is the final fallback for combinations without a published
// binary (notably Linux CUDA and DGX Spark's aarch64 + sm_121). Install steps
// stream their output live and report progress through the shared
// VllmInstallProgress shape.
struct LlamaProvisionConfig {
    std::string requested_executable = "llama-server";
    std::string provision_dir;
    bool        auto_provision = true;
    std::string install_method = "auto"; // auto|release|source
    std::string version = "latest";      // git ref / release tag, or "latest"
    // Accelerator-correct build: cuda|rocm|vulkan|metal|cpu. Empty => detected.
    std::string accelerator;
    // True when the accelerator came from config rather than hardware probing.
    // Explicit config may intentionally supersede a previously selected update
    // alternative on the next restart.
    bool accelerator_explicit = false;
    // CUDA compute capability for the source build, e.g. "121" for the DGX Spark
    // GB10 (sm_121, requires CUDA 13+). "" => caller auto-detects when possible.
    std::string cuda_arch;
    // Extra -D flags appended verbatim to the CMake configure step.
    std::vector<std::string> cmake_args;
    // Concurrent compiler jobs for source builds. 0 selects a conservative
    // managed default (2 for CUDA/ROCm, 4 otherwise) to avoid WSL OOM failures.
    int build_jobs = 0;

    // Test hooks; empty means use the build host.
    std::string platform; // windows|linux|macos
    std::string arch;      // x86_64|aarch64|arm64
};

// One command in a llama.cpp install/build plan. Steps run in order, each
// streaming output live. No script file is written to disk.
struct LlamaInstallStep {
    std::string              label;
    std::vector<std::string> argv;
    std::filesystem::path    cwd;
    bool                     allow_failure = false;
};

struct LlamaCommandRunner {
    using RunFn = std::function<int(const std::vector<std::string>& argv,
                                    const std::filesystem::path& cwd,
                                    const StreamLineCallback& on_line,
                                    const CancelCheckCallback& cancel_requested,
                                    std::string* error)>;
    using CaptureFn = std::function<std::string(const std::vector<std::string>&,
                                                const std::filesystem::path&)>;
    // Newest llama.cpp release tag available upstream, or "" when unknown.
    using FetchLatestFn = std::function<std::string(const LlamaProvisionConfig&)>;
    // Asset names attached to a release tag. Kept separate from fetch_latest so
    // tests and offline callers can make the update decision deterministically.
    using FetchReleaseAssetsFn =
        std::function<std::vector<std::string>(const LlamaProvisionConfig&,
                                               const std::string& tag)>;
    RunFn run;
    CaptureFn capture_first_line;
    FetchLatestFn fetch_latest;
    FetchReleaseAssetsFn fetch_release_assets;
};

std::string normalize_llama_install_method(const std::string& method);
std::filesystem::path managed_llama_executable_path(const LlamaProvisionConfig& cfg);
// Build one ordered install attempt for this environment. auto returns its
// preferred release attempt; LlamaCppProvisioner runs the source fallback only
// if that attempt cannot produce a usable executable. Pure and unit-tested (no
// process is spawned).
std::vector<LlamaInstallStep> build_llama_install_plan(const LlamaProvisionConfig& cfg,
                                                       bool upgrade = false);
// Return accelerator variants that have an official release for cfg's OS/arch.
// The list is ordered for UI display and does not include source-build options.
std::vector<std::string> llama_release_accelerators(
    const std::vector<std::string>& asset_names,
    const LlamaProvisionConfig& cfg);
std::string resolve_llama_executable(const std::string& executable);
bool llama_runtime_usable(const LlamaRuntimeStatus& status);

class LlamaCppProvisioner {
public:
    explicit LlamaCppProvisioner(LlamaProvisionConfig cfg,
                                 LlamaCommandRunner runner = {});

    using LogSink = std::function<void(const std::string& line, bool is_stderr)>;
    using ProgressSink = std::function<void(const VllmInstallProgress&)>;
    using CancelCheck = CancelCheckCallback;
    void set_log_sink(LogSink sink);
    void set_progress_sink(ProgressSink sink);
    void set_cancel_check(CancelCheck check);

    LlamaRuntimeStatus ensure_runtime();
    LlamaRuntimeStatus check_for_update();
    // Empty override keeps the configured accelerator. A non-empty override is
    // an explicit user choice from the update prompt and uses a release asset.
    LlamaRuntimeStatus update_runtime(const std::string& accelerator_override = {});
    LlamaRuntimeStatus status() const;
    LlamaProvisionConfig config() const;

private:
    LlamaProvisionConfig cfg_;
    LlamaCommandRunner runner_;
    LogSink log_sink_;
    ProgressSink progress_sink_;
    CancelCheck cancel_check_;
    mutable std::mutex status_mutex_;
    LlamaRuntimeStatus status_;

    LlamaRuntimeStatus make_base_status() const;
    std::string capture_version(const std::string& executable) const;
    LlamaRuntimeStatus run_managed_install(bool upgrade,
                                           const std::string& accelerator_override = {});
    void emit_progress(const VllmInstallProgress& p);
    void set_status(const LlamaRuntimeStatus& status);
};

} // namespace mm
