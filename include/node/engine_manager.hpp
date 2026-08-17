#pragma once

// Mantic-Mind — the node's side of the cluster engine configuration.
//
// Owns every EngineProvisioner, applies what the master sent, and answers the
// one question control actually asks: does this node run what the cluster said
// it should?
//
// ── What this replaces ────────────────────────────────────────────────────────
//
// ~120 lines in node/main.cpp: a LlamaProvisionConfig assembled inline, a
// provisioner constructed as a local, three sinks, an apply-result lambda under
// its own mutex, a provisioning thread, and four more lambdas for the API
// routes. All of it llama.cpp-shaped, all of it unconditional. A second engine
// had nowhere to go, and the node had no way to be told it did not want the
// first one.
//
// ── The provisioning set ──────────────────────────────────────────────────────
//
// EXACTLY ClusterEngineConfig::required_engines(), which is primary plus a
// possibly-empty backup. Nothing else is provisioned — that is what makes the
// backup engine optional in fact rather than only in the schema. An engine the
// node knows how to acquire but the cluster did not name is left alone.

#include "common/engine_config.hpp"
#include "node/engine_provisioner.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mm {

/// Where this node keeps things. Deployment settings, not cluster policy: the
/// master says what to install, this says where it lands.
struct EngineManagerPaths {
    std::string llama_executable = "llama-server";
    std::string llama_provision_dir;
    std::string soma_executable = "soma";
};

class NodeEngineManager {
public:
    explicit NodeEngineManager(EngineManagerPaths paths);
    ~NodeEngineManager();

    NodeEngineManager(const NodeEngineManager&) = delete;
    NodeEngineManager& operator=(const NodeEngineManager&) = delete;

    using LogSink = EngineProvisioner::LogSink;
    using ProgressSink = EngineProvisioner::ProgressSink;
    using CancelCheck = EngineProvisioner::CancelCheck;
    void set_log_sink(LogSink sink);
    void set_progress_sink(ProgressSink sink);
    void set_cancel_check(CancelCheck check);

    /// Fired whenever an engine's executable resolves or changes, so the caller
    /// can re-register the EngineDescriptor. Passing the id and path rather
    /// than touching EngineRegistry here keeps this class testable without the
    /// singleton, and keeps descriptor construction where the descriptors live.
    using EngineResolvedCallback =
        std::function<void(const std::string& engine_id, const std::string& executable)>;
    void set_engine_resolved_callback(EngineResolvedCallback cb);

    /// Apply a cluster configuration. BLOCKING — a source build takes minutes;
    /// callers run it on a background thread (see apply_async).
    ///
    /// Idempotent: applying the same version twice re-verifies rather than
    /// re-installing, because ensure() on an already-satisfied engine resolves
    /// and returns.
    void apply(const ClusterEngineConfig& cfg);

    /// apply() on a detached worker, replacing any in-flight application. Used
    /// by the API route, which must answer before a build finishes.
    void apply_async(const ClusterEngineConfig& cfg);

    /// The last configuration applied or being applied. Version 0 = none yet.
    ClusterEngineConfig current_config() const;

    /// Recomputed from live provisioner status on every call rather than cached
    /// at apply time: a runtime that fails after a successful apply is drift,
    /// and a snapshot taken at apply time would keep reporting the moment it
    /// succeeded.
    EngineConformance conformance() const;

    /// Status of every engine this node knows how to provision — including ones
    /// the cluster did not ask for, which report `absent`. Control renders this,
    /// and "we could run this but were not asked to" is worth seeing.
    std::vector<RuntimeStatus> engine_statuses() const;

    /// Artifacts this node could serve to a peer. Only managed, verified
    /// installs appear; a PATH-resolved binary never does.
    std::vector<EngineArtifact> shareable_artifacts() const;

    /// nullptr when this node has no provisioner for `engine_id`.
    EngineProvisioner* provisioner(const std::string& engine_id) const;

    /// Replace the provisioner for one engine, wiring it to the current sinks.
    ///
    /// A test seam, and named as one. The property it exists to assert is that a
    /// provisioner which THROWS leaves this node reporting a failed engine
    /// rather than taking the process down — and that is not assertable against
    /// the two real provisioners, which would need a network and a build
    /// toolchain to be made to fail. It was not hypothetical: an exception out
    /// of llama.cpp provisioning killed the node on every config push (D56).
    ///
    /// Not for the API or the TUI: which engines a node can acquire is a
    /// property of the build, not something the cluster or an operator sets.
    void set_provisioner(const std::string& engine_id, std::unique_ptr<EngineProvisioner> p);

    /// The llama-specific view, for the routes and TUI that need release
    /// variants and the troubleshooting report. Default-constructed when llama
    /// is not among this node's provisioners.
    LlamaRuntimeStatus llama_status() const;

    /// Package the named engine for transfer to a peer.
    bool package(const std::string& engine_id, const std::string& out_path, std::string& err);

    /// Install a peer's package. The caller must have verified the digest
    /// already; this checks the fingerprint against what the node needs.
    bool install_package(const std::string& package_path,
                         const EngineArtifact& artifact,
                         std::string& err);

    /// Stop any in-flight application and wait for the worker to exit.
    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mm
