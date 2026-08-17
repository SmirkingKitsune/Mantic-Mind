#include "node/engine_provisioner.hpp"

#include "common/logger.hpp"
#include "common/pairing.hpp"
#include "common/util.hpp"
#include "node/llama_cpp_provisioner.hpp"
#include "node/llama_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <set>

namespace mm {

namespace fs = std::filesystem;

// EngineArtifact's implementation moved with its declaration, to
// src/common/engine_config.cpp.

// ── EngineProvisioner ─────────────────────────────────────────────────────────

EngineProvisioner::~EngineProvisioner() = default;

void EngineProvisioner::set_log_sink(LogSink sink) {
    log_sink_ = std::move(sink);
}

void EngineProvisioner::set_progress_sink(ProgressSink sink) {
    progress_sink_ = std::move(sink);
}

void EngineProvisioner::set_cancel_check(CancelCheck check) {
    cancel_check_ = std::move(check);
}

namespace {

/// Every visible NVIDIA GPU's compute capability in CMake form (8.9 -> 89).
///
/// Moved here from node/main.cpp. It is llama.cpp source-build detail, and it
/// belongs with the provisioner that consumes it rather than in the node's
/// entry point — the cluster config cannot state it (per-machine), so something
/// on this side has to discover it.
std::string detect_cuda_architectures() {
    const std::string cmd = "nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits";
#ifdef _WIN32
    FILE* f = _popen((cmd + " 2>nul").c_str(), "r");
#else
    FILE* f = ::popen((cmd + " 2>/dev/null").c_str(), "r");
#endif
    if (!f) return {};
    std::set<std::string> capabilities;
    char buf[256];
    while (fgets(buf, static_cast<int>(sizeof(buf)), f)) {
        std::string value = mm::util::trim(buf);
        value.erase(std::remove(value.begin(), value.end(), '.'), value.end());
        if (!value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
                return std::isdigit(ch) != 0;
            }))
            capabilities.insert(value);
    }
#ifdef _WIN32
    _pclose(f);
#else
    ::pclose(f);
#endif
    std::vector<std::string> ordered(capabilities.begin(), capabilities.end());
    return mm::util::join(ordered, ";");
}

/// tar is the packaging tool because every target ships one: bsdtar has been in
/// Windows since 1803, and it is standard on Linux and macOS. Adding a zip
/// dependency to carry files between two machines that both already have this
/// would be a new third-party surface for no capability.
bool run_tar(const std::vector<std::string>& argv,
             const fs::path& cwd,
             const StreamLineCallback& on_line,
             std::string& err) {
    std::string launch_error;
    const int rc = run_streamed_command(argv, cwd, on_line, &launch_error);
    if (rc == -1) {
        err = "cannot run tar: " + launch_error;
        return false;
    }
    if (rc != 0) {
        err = "tar exited " + std::to_string(rc);
        return false;
    }
    return true;
}

RuntimeStatus from_llama(const LlamaRuntimeStatus& s) {
    RuntimeStatus out;
    out.engine_id = "llama-cpp";
    out.executable_path = s.executable_path;
    out.version = s.version;
    out.variant = s.variant.empty() ? s.accelerator : s.variant;
    out.last_error = s.last_error;
    out.ready = llama_runtime_usable(s);

    // Mapped rather than passed through: LlamaRuntimeStatus::status has its own
    // vocabulary (resolved|provisioning|ready|failed|disabled) and RuntimeStatus
    // documents another (resolved|ready|building|error|absent). Two enums
    // sharing four spellings and disagreeing on the fifth is how a UI ends up
    // rendering a state nobody defined.
    if (!s.last_error.empty() && !out.ready)
        out.status = "error";
    else if (out.ready)
        out.status = "ready";
    else if (s.status == "provisioning")
        out.status = "building";
    else if (s.status == "disabled")
        out.status = "absent";
    else if (s.status == "resolved")
        out.status = "resolved";
    else
        out.status = "absent";
    return out;
}

} // namespace

// ── LlamaEngineProvisioner ────────────────────────────────────────────────────

struct LlamaEngineProvisioner::Impl {
    std::string id = "llama-cpp";
    std::string requested_executable;
    std::string provision_dir;

    mutable std::mutex mutex_;
    std::unique_ptr<LlamaCppProvisioner> prov;
    LlamaRuntimeStatus last;
    /// True once a managed install produced the active runtime. A binary merely
    /// found on PATH is NOT shareable: this node cannot say what build it is,
    /// and shipping an unidentified binary to a peer under a fingerprint this
    /// node invented is worse than making the peer resolve its own.
    bool managed = false;

    /// Turn cluster intent into a machine-specific plan. THE local-resolution
    /// step: everything added here is something the master could not have known.
    LlamaProvisionConfig resolve_local(const EngineSpec& spec, bool gpu_available) const {
        LlamaProvisionConfig cfg;
        cfg.requested_executable = requested_executable;
        cfg.provision_dir = provision_dir;
        cfg.auto_provision = true; // naming the engine in cluster config IS the opt-in
        cfg.install_method = normalize_llama_install_method(spec.install_method);
        cfg.version = spec.version;
        cfg.cmake_args = spec.cmake_args;
        cfg.build_jobs = spec.build_jobs;

        cfg.accelerator = detect_llama_accelerator(current_runtime_platform(),
                                                   current_runtime_arch(),
                                                   gpu_available,
                                                   detect_rocm_present());
        // Never explicit: the cluster config cannot carry an accelerator, so a
        // node's accelerator is always detected. Saying otherwise would make the
        // provisioner treat a probe result as an operator's choice and keep
        // offering to "reinstall the target".
        cfg.accelerator_explicit = false;
        if (cfg.accelerator == "cuda") cfg.cuda_arch = detect_cuda_architectures();
        return cfg;
    }
};

LlamaEngineProvisioner::LlamaEngineProvisioner(std::string requested_executable,
                                               std::string provision_dir)
    : impl_(std::make_unique<Impl>()) {
    impl_->requested_executable = std::move(requested_executable);
    impl_->provision_dir = std::move(provision_dir);
    impl_->last.status = "disabled";
}

LlamaEngineProvisioner::~LlamaEngineProvisioner() = default;

const std::string& LlamaEngineProvisioner::engine_id() const {
    return impl_->id;
}

RuntimeStatus LlamaEngineProvisioner::ensure(const EngineSpec& spec) {
    std::lock_guard<std::mutex> g(impl_->mutex_);

    // Rebuilt per call so a cluster config change (a new version, a switch from
    // release to source) takes effect without restarting the node.
    auto cfg = impl_->resolve_local(spec, /*gpu_available=*/true);
    impl_->prov = std::make_unique<LlamaCppProvisioner>(cfg);
    if (log_sink_) impl_->prov->set_log_sink(log_sink_);
    if (progress_sink_) impl_->prov->set_progress_sink(progress_sink_);
    if (cancel_check_) impl_->prov->set_cancel_check(cancel_check_);

    impl_->last = impl_->prov->ensure_runtime();
    impl_->managed = impl_->last.managed;
    return from_llama(impl_->last);
}

RuntimeStatus LlamaEngineProvisioner::check_for_update(const EngineSpec& spec) {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    if (!impl_->prov) {
        auto cfg = impl_->resolve_local(spec, true);
        impl_->prov = std::make_unique<LlamaCppProvisioner>(cfg);
    }
    impl_->last = impl_->prov->check_for_update();
    return from_llama(impl_->last);
}

RuntimeStatus LlamaEngineProvisioner::update(const EngineSpec& spec,
                                             const std::string& variant_override) {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    if (!impl_->prov) {
        auto cfg = impl_->resolve_local(spec, true);
        impl_->prov = std::make_unique<LlamaCppProvisioner>(cfg);
        if (log_sink_) impl_->prov->set_log_sink(log_sink_);
        if (progress_sink_) impl_->prov->set_progress_sink(progress_sink_);
        if (cancel_check_) impl_->prov->set_cancel_check(cancel_check_);
    }
    impl_->last = impl_->prov->update_runtime(variant_override);
    if (!llama_runtime_usable(impl_->last)) {
        // An update that produced nothing usable must not leave the node with
        // no engine: fall back to whatever was already installed.
        impl_->last = impl_->prov->ensure_runtime();
    }
    impl_->managed = impl_->last.managed;
    return from_llama(impl_->last);
}

RuntimeStatus LlamaEngineProvisioner::status() const {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    return from_llama(impl_->last);
}

LlamaRuntimeStatus LlamaEngineProvisioner::llama_status() const {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    return impl_->last;
}

std::string LlamaEngineProvisioner::executable_path() const {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    return impl_->last.executable_path;
}

RuntimeStatus LlamaEngineProvisioner::switch_variant(const std::string& variant) {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    if (!impl_->prov) {
        RuntimeStatus s = from_llama(impl_->last);
        s.last_error = "llama.cpp is not provisioned on this node";
        return s;
    }
    impl_->last = impl_->prov->switch_runtime(variant);
    if (!llama_runtime_usable(impl_->last)) impl_->last = impl_->prov->ensure_runtime();
    impl_->managed = impl_->last.managed;
    return from_llama(impl_->last);
}

RuntimeStatus LlamaEngineProvisioner::diagnose() {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    if (!impl_->prov) {
        RuntimeStatus s = from_llama(impl_->last);
        s.last_error = "llama.cpp is not provisioned on this node";
        return s;
    }
    impl_->last = impl_->prov->diagnose_environment();
    return from_llama(impl_->last);
}

RuntimeStatus LlamaEngineProvisioner::recover(const std::string& action,
                                              const std::string& variant) {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    if (!impl_->prov) {
        RuntimeStatus s = from_llama(impl_->last);
        s.last_error = "llama.cpp is not provisioned on this node";
        return s;
    }
    impl_->last = impl_->prov->recover_runtime(action, variant);
    impl_->managed = impl_->last.managed;
    return from_llama(impl_->last);
}

std::optional<EngineArtifact> LlamaEngineProvisioner::installed_artifact() const {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    if (!llama_runtime_usable(impl_->last)) return std::nullopt;
    if (!impl_->managed) return std::nullopt; // see Impl::managed
    if (impl_->last.version.empty()) return std::nullopt;

    EngineArtifact a;
    a.engine_id = impl_->id;
    a.version = impl_->last.version;
    a.platform = current_runtime_platform();
    a.arch = current_runtime_arch();
    a.variant = impl_->last.variant.empty() ? impl_->last.accelerator : impl_->last.variant;
    if (!a.valid()) return std::nullopt;
    return a;
}

bool LlamaEngineProvisioner::shareable() const {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    return impl_->managed && llama_runtime_usable(impl_->last);
}

bool LlamaEngineProvisioner::package(const std::string& out_path, std::string& err) {
    err.clear();
    std::string exe;
    {
        std::lock_guard<std::mutex> g(impl_->mutex_);
        if (!impl_->managed || !llama_runtime_usable(impl_->last)) {
            err = "no managed llama.cpp runtime on this node to package";
            return false;
        }
        exe = impl_->last.executable_path;
    }

    std::error_code ec;
    const fs::path exe_path(exe);
    const fs::path root = exe_path.parent_path();
    if (root.empty() || !fs::is_directory(root, ec)) {
        err = "installed runtime has no directory to package: " + exe;
        return false;
    }

    const fs::path out(out_path);
    if (out.has_parent_path()) fs::create_directories(out.parent_path(), ec);

    // The whole install directory, not just the executable: a CUDA build is a
    // binary plus its ggml/backend shared libraries, and a peer that received
    // only the exe would get a runtime that fails to start at load time rather
    // than at transfer time.
    const auto on_line = [this](const std::string& line, bool is_stderr) {
        if (log_sink_) log_sink_(line, is_stderr);
    };
    return run_tar({"tar", "-czf", out.string(), "-C", root.string(), "."}, root, on_line, err);
}

bool LlamaEngineProvisioner::install_package(const std::string& package_path,
                                             const EngineArtifact& artifact,
                                             std::string& err) {
    err.clear();
    if (artifact.engine_id != impl_->id) {
        err = "package is for engine '" + artifact.engine_id + "', not llama-cpp";
        return false;
    }
    // The receiving node checks the artifact against ITS OWN environment. A
    // fingerprint match brokered by control is necessary and not sufficient —
    // control's view of this node's platform is a report from this node, and the
    // only authority on what will actually execute here is here.
    if (artifact.platform != current_runtime_platform() ||
        artifact.arch != current_runtime_arch()) {
        err = "package targets " + artifact.platform + "/" + artifact.arch + ", this node is " +
              current_runtime_platform() + "/" + current_runtime_arch();
        return false;
    }

    LlamaProvisionConfig cfg;
    cfg.requested_executable = impl_->requested_executable;
    cfg.provision_dir = impl_->provision_dir;
    const fs::path target = managed_llama_executable_path(cfg).parent_path();
    if (target.empty()) {
        err = "no managed install directory configured";
        return false;
    }

    std::error_code ec;
    fs::create_directories(target, ec);
    if (ec) {
        err = "cannot create " + target.string() + ": " + ec.message();
        return false;
    }

    const auto on_line = [this](const std::string& line, bool is_stderr) {
        if (log_sink_) log_sink_(line, is_stderr);
    };
    if (!run_tar({"tar", "-xzf", package_path, "-C", target.string()}, target, on_line, err))
        return false;

    MM_INFO(
        "llama.cpp: installed shared artifact {} into {}", artifact.fingerprint(), target.string());
    return true;
}

// ── SomaEngineProvisioner ─────────────────────────────────────────────────────

struct SomaEngineProvisioner::Impl {
    std::string id = "soma";
    std::string requested_executable;
    mutable std::mutex mutex_;
    RuntimeStatus status;
};

SomaEngineProvisioner::SomaEngineProvisioner(std::string requested_executable)
    : impl_(std::make_unique<Impl>()) {
    impl_->requested_executable = std::move(requested_executable);
    impl_->status.engine_id = "soma";
    impl_->status.status = "absent";
}

SomaEngineProvisioner::~SomaEngineProvisioner() = default;

const std::string& SomaEngineProvisioner::engine_id() const {
    return impl_->id;
}

RuntimeStatus SomaEngineProvisioner::ensure(const EngineSpec& /*spec*/) {
    std::lock_guard<std::mutex> g(impl_->mutex_);

    // Resolution, not acquisition. Soma is built from this repository and ships
    // beside the node binary, so "provisioning" it means finding it. The spec's
    // version/install_method are deliberately unread: reporting a version this
    // node cannot install would invite a conformance failure nobody can fix
    // from the cluster config.
    const std::string resolved = resolve_llama_executable(impl_->requested_executable);
    impl_->status.engine_id = impl_->id;
    if (resolved.empty()) {
        impl_->status.status = "absent";
        impl_->status.ready = false;
        impl_->status.executable_path.clear();
        impl_->status.last_error =
            "soma executable '" + impl_->requested_executable +
            "' not found; it is built with the node — check MM_SOMA_PATH or the install";
    } else {
        impl_->status.status = "ready";
        impl_->status.ready = true;
        impl_->status.executable_path = resolved;
        impl_->status.last_error.clear();
    }
    return impl_->status;
}

RuntimeStatus SomaEngineProvisioner::check_for_update(const EngineSpec& spec) {
    // Soma updates with the node. There is nothing to check that would not be a
    // claim about a release channel that does not exist.
    return ensure(spec);
}

RuntimeStatus SomaEngineProvisioner::update(const EngineSpec& spec,
                                            const std::string& /*variant_override*/) {
    return ensure(spec);
}

RuntimeStatus SomaEngineProvisioner::status() const {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    return impl_->status;
}

std::string SomaEngineProvisioner::executable_path() const {
    std::lock_guard<std::mutex> g(impl_->mutex_);
    return impl_->status.executable_path;
}

std::optional<EngineArtifact> SomaEngineProvisioner::installed_artifact() const {
    // Nothing to advertise: Soma is not acquired, so it is never the answer to
    // another node's need.
    return std::nullopt;
}

bool SomaEngineProvisioner::shareable() const {
    return false;
}

bool SomaEngineProvisioner::package(const std::string& /*out_path*/, std::string& err) {
    err = "soma ships with the node and is not shared between nodes";
    return false;
}

bool SomaEngineProvisioner::install_package(const std::string& /*package_path*/,
                                            const EngineArtifact& /*artifact*/,
                                            std::string& err) {
    err = "soma ships with the node and cannot be installed from a peer";
    return false;
}

} // namespace mm
