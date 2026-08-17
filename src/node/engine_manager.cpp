#include "node/engine_manager.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"

#include <atomic>
#include <map>
#include <mutex>
#include <thread>

namespace mm {

struct NodeEngineManager::Impl {
    EngineManagerPaths paths;

    mutable std::mutex mutex;
    std::map<std::string, std::unique_ptr<EngineProvisioner>> provisioners;
    ClusterEngineConfig config;
    bool have_config = false;

    /// Set while apply() runs. Conformance reads it to answer Converging rather
    /// than Failed for an engine that is mid-build — a build in progress is not
    /// a build that failed, and reporting it as one would light up every
    /// operator surface on a healthy first run.
    std::atomic<bool> applying{false};
    /// Populated by apply() when it cannot acquire an engine locally and a peer
    /// might have it. Read by conformance() into `needs_artifact`.
    std::string needs_artifact;
    std::string apply_error;

    EngineProvisioner::LogSink log_sink;
    EngineProvisioner::ProgressSink progress_sink;
    EngineProvisioner::CancelCheck cancel_check;
    EngineResolvedCallback resolved_cb;

    std::thread worker;
    std::atomic<bool> stopping{false};

    EngineProvisioner* find(const std::string& id) const {
        auto it = provisioners.find(id);
        return it == provisioners.end() ? nullptr : it->second.get();
    }
};

NodeEngineManager::NodeEngineManager(EngineManagerPaths paths) : impl_(std::make_unique<Impl>()) {
    impl_->paths = std::move(paths);

    // Every engine this node COULD acquire is constructed up front; what it
    // actually provisions is decided by apply(). Constructing a provisioner
    // costs nothing and resolves nothing — LlamaEngineProvisioner does not
    // touch the filesystem until ensure().
    impl_->provisioners.emplace(
        "llama-cpp",
        std::make_unique<LlamaEngineProvisioner>(impl_->paths.llama_executable,
                                                 impl_->paths.llama_provision_dir));
    impl_->provisioners.emplace(
        "soma", std::make_unique<SomaEngineProvisioner>(impl_->paths.soma_executable));
}

NodeEngineManager::~NodeEngineManager() {
    shutdown();
}

void NodeEngineManager::shutdown() {
    impl_->stopping.store(true);
    std::thread worker;
    {
        std::lock_guard<std::mutex> g(impl_->mutex);
        worker = std::move(impl_->worker);
    }
    if (worker.joinable()) worker.join();
}

void NodeEngineManager::set_log_sink(LogSink sink) {
    std::lock_guard<std::mutex> g(impl_->mutex);
    impl_->log_sink = sink;
    for (auto& [id, p] : impl_->provisioners)
        p->set_log_sink(sink);
}

void NodeEngineManager::set_progress_sink(ProgressSink sink) {
    std::lock_guard<std::mutex> g(impl_->mutex);
    impl_->progress_sink = sink;
    for (auto& [id, p] : impl_->provisioners)
        p->set_progress_sink(sink);
}

void NodeEngineManager::set_cancel_check(CancelCheck check) {
    std::lock_guard<std::mutex> g(impl_->mutex);
    impl_->cancel_check = check;
    for (auto& [id, p] : impl_->provisioners)
        p->set_cancel_check(check);
}

void NodeEngineManager::set_engine_resolved_callback(EngineResolvedCallback cb) {
    std::lock_guard<std::mutex> g(impl_->mutex);
    impl_->resolved_cb = std::move(cb);
}

EngineProvisioner* NodeEngineManager::provisioner(const std::string& engine_id) const {
    std::lock_guard<std::mutex> g(impl_->mutex);
    return impl_->find(engine_id);
}

ClusterEngineConfig NodeEngineManager::current_config() const {
    std::lock_guard<std::mutex> g(impl_->mutex);
    return impl_->config;
}

void NodeEngineManager::apply(const ClusterEngineConfig& cfg) {
    std::vector<std::pair<std::string, const EngineSpec*>> work;
    EngineResolvedCallback resolved_cb;
    {
        std::lock_guard<std::mutex> g(impl_->mutex);
        impl_->config = cfg;
        impl_->have_config = true;
        impl_->needs_artifact.clear();
        impl_->apply_error.clear();
        resolved_cb = impl_->resolved_cb;
    }
    impl_->applying.store(true);

    MM_INFO("Engine config v{}: primary='{}' backup='{}' share_builds={}",
            cfg.version,
            cfg.primary_engine,
            cfg.backup_engine.empty() ? "(none)" : cfg.backup_engine,
            cfg.share_builds);

    std::string first_error;
    std::string needs;

    for (const auto& id : cfg.required_engines()) {
        if (impl_->stopping.load()) break;

        EngineProvisioner* p = provisioner(id);
        if (p == nullptr) {
            // Named an engine this node has no acquisition story for. Reported
            // rather than skipped: the cluster asked for something this build
            // cannot supply, and silence would render as healthy.
            if (first_error.empty())
                first_error = "no provisioner for engine '" + id + "' on this node";
            continue;
        }
        const EngineSpec* spec = cfg.find(id);
        if (spec == nullptr) {
            if (first_error.empty()) first_error = "engine '" + id + "' has no spec";
            continue;
        }

        const RuntimeStatus status = p->ensure(*spec);
        if (status.ready) {
            if (resolved_cb && !status.executable_path.empty())
                resolved_cb(id, status.executable_path);
            MM_INFO("Engine '{}' ready: {} (variant={}, version={})",
                    id,
                    status.executable_path,
                    status.variant.empty() ? "-" : status.variant,
                    status.version.empty() ? "-" : status.version);
            continue;
        }

        if (first_error.empty()) {
            first_error = "engine '" + id +
                          "': " + (status.last_error.empty() ? "not ready" : status.last_error);
        }
        // Only a shareable engine can be satisfied by a peer. Asking control to
        // find a Soma artifact would send it looking for something no node ever
        // advertises.
        // What this node NEEDS, not what it has. Asking installed_artifact()
        // here was empty by construction: it requires a working runtime, and
        // this branch is reached only when provisioning failed — so a node that
        // could not build asked for help by naming the build it did not have,
        // and named nothing. `desired_artifact` answers the other question.
        //
        // Soma returns nullopt: it ships with the node, so no peer has one to
        // give and asking control to look would be a search that cannot succeed.
        if (needs.empty() && cfg.share_builds) {
            if (auto a = p->desired_artifact(*spec)) needs = a->fingerprint();
        }
        MM_WARN("Engine '{}' not ready: {}", id, status.last_error);
    }

    {
        std::lock_guard<std::mutex> g(impl_->mutex);
        impl_->apply_error = first_error;
        impl_->needs_artifact = needs;
    }
    impl_->applying.store(false);
}

void NodeEngineManager::apply_async(const ClusterEngineConfig& cfg) {
    std::thread previous;
    {
        std::lock_guard<std::mutex> g(impl_->mutex);
        previous = std::move(impl_->worker);
    }
    // Joined before starting the next one: two concurrent applications would
    // race on the same provisioner and could leave a half-installed runtime
    // that neither call believes it owns.
    if (previous.joinable()) previous.join();

    std::thread next([this, cfg]() { apply(cfg); });
    {
        std::lock_guard<std::mutex> g(impl_->mutex);
        impl_->worker = std::move(next);
    }
}

std::vector<RuntimeStatus> NodeEngineManager::engine_statuses() const {
    std::lock_guard<std::mutex> g(impl_->mutex);
    std::vector<RuntimeStatus> out;
    out.reserve(impl_->provisioners.size());
    for (const auto& [id, p] : impl_->provisioners) {
        RuntimeStatus s = p->status();
        if (s.engine_id.empty()) s.engine_id = id;
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<EngineArtifact> NodeEngineManager::shareable_artifacts() const {
    std::lock_guard<std::mutex> g(impl_->mutex);
    std::vector<EngineArtifact> out;
    for (const auto& [id, p] : impl_->provisioners) {
        if (!p->shareable()) continue;
        if (auto a = p->installed_artifact()) out.push_back(*a);
    }
    return out;
}

LlamaRuntimeStatus NodeEngineManager::llama_status() const {
    std::lock_guard<std::mutex> g(impl_->mutex);
    auto* p = impl_->find("llama-cpp");
    if (p == nullptr) return {};
    return static_cast<LlamaEngineProvisioner*>(p)->llama_status();
}

EngineConformance NodeEngineManager::conformance() const {
    EngineConformance c;
    std::lock_guard<std::mutex> g(impl_->mutex);

    if (!impl_->have_config) {
        c.state = EngineConformanceState::Unconfigured;
        c.detail = "no cluster engine configuration has reached this node";
        return c;
    }

    c.config_version = impl_->config.version;
    c.needs_artifact = impl_->needs_artifact;

    const auto required = impl_->config.required_engines();
    if (required.empty()) {
        // A config that names nothing is not a conforming node — it is a config
        // that failed validation somewhere upstream, and serving on it would
        // mean serving with whatever happened to be lying around.
        c.state = EngineConformanceState::Failed;
        c.detail = "cluster configuration names no engines";
        return c;
    }

    std::vector<std::string> missing;
    for (const auto& id : required) {
        auto* p = impl_->find(id);
        if (p == nullptr) {
            missing.push_back(id + " (unsupported here)");
            continue;
        }
        const RuntimeStatus s = p->status();
        if (!s.ready) missing.push_back(id + (s.last_error.empty() ? "" : ": " + s.last_error));
    }

    if (missing.empty()) {
        c.state = EngineConformanceState::Conforming;
        c.detail = "running " + std::to_string(required.size()) +
                   " configured engine(s) at config v" + std::to_string(c.config_version);
        return c;
    }

    // Mid-application is Converging, not Failed. The distinction is the whole
    // reason both states exist: one is "wait", the other is "act".
    if (impl_->applying.load()) {
        c.state = EngineConformanceState::Converging;
        c.detail = "applying config v" + std::to_string(c.config_version) +
                   "; pending: " + util::join(missing, ", ");
        return c;
    }

    c.state = EngineConformanceState::Failed;
    c.detail = impl_->apply_error.empty() ? ("not ready: " + util::join(missing, ", "))
                                          : impl_->apply_error;
    return c;
}

bool NodeEngineManager::package(const std::string& engine_id,
                                const std::string& out_path,
                                std::string& err) {
    EngineProvisioner* p = provisioner(engine_id);
    if (p == nullptr) {
        err = "no provisioner for engine '" + engine_id + "'";
        return false;
    }
    return p->package(out_path, err);
}

bool NodeEngineManager::install_package(const std::string& package_path,
                                        const EngineArtifact& artifact,
                                        std::string& err) {
    EngineProvisioner* p = provisioner(artifact.engine_id);
    if (p == nullptr) {
        err = "no provisioner for engine '" + artifact.engine_id + "'";
        return false;
    }
    if (!p->install_package(package_path, artifact, err)) return false;

    // Re-resolve so the descriptor picks up the newly installed binary; without
    // this the node holds an installed engine it still reports as absent.
    const ClusterEngineConfig cfg = current_config();
    if (const EngineSpec* spec = cfg.find(artifact.engine_id)) {
        const RuntimeStatus s = p->ensure(*spec);
        EngineResolvedCallback cb;
        {
            std::lock_guard<std::mutex> g(impl_->mutex);
            cb = impl_->resolved_cb;
        }
        if (s.ready && cb && !s.executable_path.empty()) cb(artifact.engine_id, s.executable_path);
    }
    return true;
}

} // namespace mm
