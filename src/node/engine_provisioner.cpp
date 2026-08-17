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
#include <vector>

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

/// The engine sitting beside this node's own binary, if it is there.
///
/// Consulted BEFORE PATH, and that order is the point. Soma is built from this
/// repository and versioned with the node; a `soma` that happens to be on PATH
/// belongs to some other install, and preferring it would pair this node with
/// an engine it was never built against. An operator who names a path outright
/// still wins over both — that case never reaches here.
///
/// Bare-name-only by construction: a request containing a separator is already
/// an answer, and joining it onto the executable's directory would invent a
/// path nobody asked for.
std::string resolve_beside_executable(const std::string& requested) {
    if (requested.empty()) return {};
    if (requested.find('/') != std::string::npos || requested.find('\\') != std::string::npos)
        return {};

    const std::string dir = util::executable_dir();
    if (dir.empty()) return {};

    std::error_code ec;
    std::vector<std::string> names{requested};
#ifdef _WIN32
    names.push_back(requested + ".exe");
#endif
    for (const auto& name : names) {
        const fs::path candidate = fs::path(dir) / name;
        if (fs::is_regular_file(candidate, ec)) return candidate.string();
    }
    return {};
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
    LlamaCommandRunner runner;

    /// Serializes long operations against each other — ensure, update, switch,
    /// diagnose, recover. Held for the whole operation, and never taken by an
    /// accessor, so a status read never waits on a build.
    std::mutex op_mutex_;

    /// Guards the fields below. Held only long enough to read or write one, and
    /// NEVER across a call into LlamaCppProvisioner: those call back out through
    /// the progress and log sinks, the node's progress sink asks the engine
    /// manager for llama status, and that lands right back here. One mutex for
    /// both jobs meant a single thread locked it twice — which MSVC reports by
    /// THROWING `resource deadlock would occur` rather than hanging, out of a
    /// worker thread with no handler. See the header (D56).
    ///
    /// `shared_ptr`, not `unique_ptr`: an operation runs with only `op_mutex_`
    /// held, so it needs a handle that provably outlives the call rather than
    /// one whose safety has to be re-argued from who holds what.
    mutable std::mutex state_mutex_;
    std::shared_ptr<LlamaCppProvisioner> prov;
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

    /// Build a provisioner for `spec`, wire the sinks, publish it, and return a
    /// handle the caller can use with no lock of ours held.
    ///
    /// Callers hold `op_mutex_` and must NOT hold `state_mutex_` when they run
    /// the operation — that is the invariant this whole file is arranged around.
    std::shared_ptr<LlamaCppProvisioner> rebuild(const EngineSpec& spec,
                                                 const EngineProvisioner::LogSink& log,
                                                 const EngineProvisioner::ProgressSink& progress,
                                                 const EngineProvisioner::CancelCheck& cancel) {
        // Rebuilt per call so a cluster config change (a new version, a switch
        // from release to source) takes effect without restarting the node.
        auto built = std::make_shared<LlamaCppProvisioner>(resolve_local(spec, true), runner);
        if (log) built->set_log_sink(log);
        if (progress) built->set_progress_sink(progress);
        if (cancel) built->set_cancel_check(cancel);
        std::lock_guard<std::mutex> g(state_mutex_);
        prov = built;
        return built;
    }

    /// The provisioner an earlier operation left behind, or null.
    std::shared_ptr<LlamaCppProvisioner> current() const {
        std::lock_guard<std::mutex> g(state_mutex_);
        return prov;
    }

    /// Record the outcome of an operation. Separate from running it, because
    /// running it must happen with no lock held.
    void settle(const LlamaRuntimeStatus& s, bool track_managed) {
        std::lock_guard<std::mutex> g(state_mutex_);
        last = s;
        if (track_managed) managed = s.managed;
    }
};

LlamaEngineProvisioner::LlamaEngineProvisioner(std::string requested_executable,
                                               std::string provision_dir,
                                               LlamaCommandRunner runner)
    : impl_(std::make_unique<Impl>()) {
    impl_->requested_executable = std::move(requested_executable);
    impl_->provision_dir = std::move(provision_dir);
    impl_->runner = std::move(runner);
    impl_->last.status = "disabled";
}

LlamaEngineProvisioner::~LlamaEngineProvisioner() = default;

const std::string& LlamaEngineProvisioner::engine_id() const {
    return impl_->id;
}

RuntimeStatus LlamaEngineProvisioner::ensure(const EngineSpec& spec) {
    std::lock_guard<std::mutex> op(impl_->op_mutex_);

    auto prov = impl_->rebuild(spec, log_sink_, progress_sink_, cancel_check_);

    // Run it with NO lock of ours held. ensure_runtime() reports progress by
    // calling back through the sinks, the node's sink asks the engine manager
    // for llama status, and that comes straight back into this object — so a
    // lock held here is a lock this thread takes twice (D56). The shared_ptr is
    // what makes the unlocked call safe.
    const LlamaRuntimeStatus s = prov->ensure_runtime();
    impl_->settle(s, /*track_managed=*/true);
    return from_llama(s);
}

RuntimeStatus LlamaEngineProvisioner::check_for_update(const EngineSpec& spec) {
    std::lock_guard<std::mutex> op(impl_->op_mutex_);

    auto prov = impl_->current();
    // Sinks wired here too. They were not before, so an update check's log and
    // progress went nowhere on a node that had never provisioned — the one
    // case where the operator most needs to see it.
    if (!prov) prov = impl_->rebuild(spec, log_sink_, progress_sink_, cancel_check_);

    const LlamaRuntimeStatus s = prov->check_for_update();
    impl_->settle(s, /*track_managed=*/false);
    return from_llama(s);
}

RuntimeStatus LlamaEngineProvisioner::update(const EngineSpec& spec,
                                             const std::string& variant_override) {
    std::lock_guard<std::mutex> op(impl_->op_mutex_);

    auto prov = impl_->current();
    if (!prov) prov = impl_->rebuild(spec, log_sink_, progress_sink_, cancel_check_);

    LlamaRuntimeStatus s = prov->update_runtime(variant_override);
    if (!llama_runtime_usable(s)) {
        // An update that produced nothing usable must not leave the node with
        // no engine: fall back to whatever was already installed.
        s = prov->ensure_runtime();
    }
    impl_->settle(s, /*track_managed=*/true);
    return from_llama(s);
}

RuntimeStatus LlamaEngineProvisioner::status() const {
    std::lock_guard<std::mutex> g(impl_->state_mutex_);
    return from_llama(impl_->last);
}

LlamaRuntimeStatus LlamaEngineProvisioner::llama_status() const {
    std::lock_guard<std::mutex> g(impl_->state_mutex_);
    return impl_->last;
}

std::string LlamaEngineProvisioner::executable_path() const {
    std::lock_guard<std::mutex> g(impl_->state_mutex_);
    return impl_->last.executable_path;
}

RuntimeStatus LlamaEngineProvisioner::switch_variant(const std::string& variant) {
    std::lock_guard<std::mutex> op(impl_->op_mutex_);

    auto prov = impl_->current();
    if (!prov) {
        RuntimeStatus s = status();
        s.last_error = "llama.cpp is not provisioned on this node";
        return s;
    }
    LlamaRuntimeStatus s = prov->switch_runtime(variant);
    if (!llama_runtime_usable(s)) s = prov->ensure_runtime();
    impl_->settle(s, /*track_managed=*/true);
    return from_llama(s);
}

RuntimeStatus LlamaEngineProvisioner::diagnose() {
    std::lock_guard<std::mutex> op(impl_->op_mutex_);

    auto prov = impl_->current();
    if (!prov) {
        RuntimeStatus s = status();
        s.last_error = "llama.cpp is not provisioned on this node";
        return s;
    }
    const LlamaRuntimeStatus s = prov->diagnose_environment();
    impl_->settle(s, /*track_managed=*/false);
    return from_llama(s);
}

RuntimeStatus LlamaEngineProvisioner::recover(const std::string& action,
                                              const std::string& variant) {
    std::lock_guard<std::mutex> op(impl_->op_mutex_);

    auto prov = impl_->current();
    if (!prov) {
        RuntimeStatus s = status();
        s.last_error = "llama.cpp is not provisioned on this node";
        return s;
    }
    const LlamaRuntimeStatus s = prov->recover_runtime(action, variant);
    impl_->settle(s, /*track_managed=*/true);
    return from_llama(s);
}

std::optional<EngineArtifact> LlamaEngineProvisioner::installed_artifact() const {
    std::lock_guard<std::mutex> g(impl_->state_mutex_);
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

std::optional<EngineArtifact>
LlamaEngineProvisioner::desired_artifact(const EngineSpec& spec) const {
    std::lock_guard<std::mutex> g(impl_->state_mutex_);

    // The version this node is trying to reach. A pinned spec names it
    // outright; "latest" is only knowable once an update check has resolved it,
    // and `target_variant`/`latest_version` are where that lands.
    std::string version = spec.version;
    if (version.empty() || version == "latest") version = impl_->last.latest_version;
    if (version.empty()) return std::nullopt;

    // The variant this node RESOLVED for itself — the local half of the
    // intent/resolution split. `target_*` is what it wants rather than what it
    // fell back to, which matters here: a node running a Vulkan fallback while
    // targeting CUDA needs the CUDA build shared to it, not another Vulkan one.
    std::string variant = impl_->last.target_variant;
    if (variant.empty()) variant = impl_->last.target_accelerator;
    if (variant.empty()) variant = impl_->last.variant;
    if (variant.empty()) variant = impl_->last.accelerator;
    // Required for llama.cpp specifically, even though EngineArtifact permits
    // an empty variant for engines that have none. Every real llama install
    // advertises an accelerator, so a fingerprint with a blank one matches no
    // source that will ever exist — a request that 404s every time is worse
    // than admitting this node cannot yet name what it needs.
    if (variant.empty()) return std::nullopt;

    EngineArtifact a;
    a.engine_id = impl_->id;
    a.version = version;
    a.platform = current_runtime_platform();
    a.arch = current_runtime_arch();
    a.variant = variant;
    if (!a.valid()) return std::nullopt;
    return a;
}

bool LlamaEngineProvisioner::shareable() const {
    std::lock_guard<std::mutex> g(impl_->state_mutex_);
    return impl_->managed && llama_runtime_usable(impl_->last);
}

bool LlamaEngineProvisioner::package(const std::string& out_path, std::string& err) {
    err.clear();
    std::string exe;
    {
        std::lock_guard<std::mutex> g(impl_->state_mutex_);
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
    //
    // "Beside the node binary" was the documented claim and PATH was the only
    // place actually searched, so in every build tree — soma in src/soma/, the
    // node in src/node/ — the engine that ships with the node was invisible to
    // it, and the node reported `absent` for a binary sitting one directory
    // away (D58). Both are searched now, sibling first.
    std::string resolved = resolve_beside_executable(impl_->requested_executable);
    if (resolved.empty()) resolved = resolve_llama_executable(impl_->requested_executable);
    impl_->status.engine_id = impl_->id;
    if (resolved.empty()) {
        const std::string dir = util::executable_dir();
        impl_->status.status = "absent";
        impl_->status.ready = false;
        impl_->status.executable_path.clear();
        // Names WHERE it looked. "Not found" plus a suggestion to check the
        // install is a dead end when the operator's next question is which of
        // the two places was empty.
        impl_->status.last_error =
            "soma executable '" + impl_->requested_executable + "' not found beside the node (" +
            (dir.empty() ? "install directory unknown" : dir) +
            ") or on PATH; it is built with the node — check MM_SOMA_PATH or the install";
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

std::optional<EngineArtifact>
SomaEngineProvisioner::desired_artifact(const EngineSpec& /*spec*/) const {
    // Nothing to want from a peer: Soma ships with the node, so a missing one
    // is a broken install rather than a build this cluster could hand over.
    // Naming an artifact here would send control looking for a source that
    // cannot exist.
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
