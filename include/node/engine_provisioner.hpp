#pragma once

// Mantic-Mind — getting an engine onto this node.
//
// The counterpart of EngineDescriptor. That answers "how do I RUN this engine";
// this answers "how do I HAVE it in the first place" — and, before this existed,
// only one engine had an answer at all.
//
// What this generalizes: LlamaCppProvisioner is a good 2200-line implementation
// of llama.cpp acquisition, and it is llama.cpp's alone. The node's main()
// instantiated it directly, wired fifteen lambdas around it, and provisioned it
// UNCONDITIONALLY — on a Soma-only cluster, a node still resolved, downloaded,
// or compiled a llama.cpp it would never launch. There was no seam at which a
// second engine could have an acquisition story, and no way for the cluster to
// say it did not want the first one.
//
// The llama implementation here is a WRAPPER, not a rewrite. Every install
// plan, release-asset matrix, accelerator fallback, and troubleshooting probe
// stays in llama_cpp_provisioner.cpp where it is already unit-tested.
//
// ── Where "local resolution" physically happens ───────────────────────────────
//
// ClusterEngineConfig carries no accelerator and no paths, by invariant. Each
// provisioner takes the cluster's INTENT (EngineSpec) and resolves it against
// hardware only this node can see. LlamaEngineProvisioner::resolve_local() is
// that step, and it is the reason the same cluster config produces a CUDA build
// on one node and a Metal one on another without either being non-conforming.

#include "common/engine_config.hpp"
#include "common/models.hpp"
#include "common/process_exec.hpp"
#include "node/engine_descriptor.hpp"
// For LlamaCommandRunner, which LlamaEngineProvisioner forwards. The wrapper
// already exposed LlamaRuntimeStatus, so this pulls in no dependency the seam
// did not already have.
#include "node/llama_cpp_provisioner.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mm {

// EngineArtifact and parse_engine_fingerprint moved to
// common/engine_config.hpp: control brokers transfers, so it has to build and
// compare fingerprints too, and a second definition on that side is exactly the
// drift this file's seam argument is about. Reachable from here unchanged.

/// How one engine is acquired and kept current on this node.
///
/// Implementations own their own status and must be safe to call from the API
/// thread, the UI thread, and a background provisioning thread.
class EngineProvisioner {
public:
    using LogSink = std::function<void(const std::string& line, bool is_stderr)>;
    using ProgressSink = std::function<void(const RuntimeInstallProgress&)>;
    using CancelCheck = CancelCheckCallback;

    virtual ~EngineProvisioner();

    virtual const std::string& engine_id() const = 0;

    /// Bring the engine to a usable state for `spec`, resolving anything
    /// machine-specific locally. Blocking; may take minutes for a source build.
    virtual RuntimeStatus ensure(const EngineSpec& spec) = 0;

    virtual RuntimeStatus check_for_update(const EngineSpec& spec) = 0;

    /// `variant_override` empty keeps the locally resolved variant.
    virtual RuntimeStatus update(const EngineSpec& spec, const std::string& variant_override) = 0;

    virtual RuntimeStatus status() const = 0;

    /// What is installed, if anything is. `nullopt` means nothing shareable —
    /// which is also the honest answer for an engine that was resolved from PATH
    /// rather than acquired, because this node cannot vouch for what that binary
    /// is or package it as a known build.
    virtual std::optional<EngineArtifact> installed_artifact() const = 0;

    /// What this node WANTS for `spec` — as opposed to what it has.
    ///
    /// The two are different questions and conflating them broke the share
    /// path: `needs_artifact` was filled from `installed_artifact()`, which is
    /// empty in exactly the case a peer is needed, because provisioning had
    /// just failed. A node that cannot build asked for help by naming the build
    /// it did not have, and so named nothing.
    ///
    /// `nullopt` when the identity cannot be pinned down — most often because
    /// the spec says "latest" and no update check has resolved what that is.
    /// A fingerprint with an empty version would match nothing and is worse
    /// than admitting the node cannot name what it needs.
    virtual std::optional<EngineArtifact> desired_artifact(const EngineSpec& spec) const = 0;

    /// Can this engine be packaged and sent to a peer at all? False for an
    /// engine that ships with the node and for a PATH-resolved binary.
    ///
    /// About SENDING. Whether a node can RECEIVE one is `desired_artifact()`
    /// answering — a node with a failed build is not shareable and is precisely
    /// the one that needs a share.
    virtual bool shareable() const = 0;

    /// Package the installed engine into `out_path`. Returns false with `err`
    /// when nothing is installed, when !shareable(), or on an I/O failure.
    virtual bool package(const std::string& out_path, std::string& err) = 0;

    /// Install a package received from a peer. The caller has ALREADY verified
    /// the digest against control's copy; an implementation must still refuse a
    /// package whose fingerprint does not match what this node needs.
    virtual bool install_package(const std::string& package_path,
                                 const EngineArtifact& artifact,
                                 std::string& err) = 0;

    /// Executable path once resolved; empty while unresolved. The node
    /// re-registers the EngineDescriptor with this whenever it changes.
    virtual std::string executable_path() const = 0;

    void set_log_sink(LogSink sink);
    void set_progress_sink(ProgressSink sink);
    void set_cancel_check(CancelCheck check);

protected:
    LogSink log_sink_;
    ProgressSink progress_sink_;
    CancelCheck cancel_check_;
};

/// llama.cpp — wraps LlamaCppProvisioner without reimplementing it.
///
/// ── Two mutexes, and why one was a crash ──────────────────────────────────────
///
/// A provisioning call runs for minutes and reports progress by CALLING BACK
/// through `progress_sink_`. The node's sink asks the engine manager for the
/// current llama status, which comes straight back here. So a single mutex held
/// across `ensure()` was locked twice by one thread — and MSVC's std::mutex does
/// not hang on that, it THROWS `resource deadlock would occur`, out of a worker
/// thread with no handler, into std::terminate. Every cluster config naming
/// llama-cpp killed the node within one health poll (D56).
///
/// So: `op` serializes long operations against each other and is held across
/// them; `state` guards the fields and is never held across a call into
/// LlamaCppProvisioner. The rule is the general one — a lock is not held across
/// a callback into code that does not belong to it — and it also removes a real
/// two-thread inversion, because the accessors run manager→state while the sink
/// ran state→manager.
class LlamaEngineProvisioner final : public EngineProvisioner {
public:
    /// `requested_executable` and `provision_dir` are NODE deployment settings
    /// (MM_LLAMA_PATH and friends), not cluster policy. They say where this
    /// machine keeps things; the cluster says what it wants kept there.
    ///
    /// `runner` is the same seam LlamaCppProvisioner already exposes, forwarded
    /// so this wrapper can be exercised without a network: the reentrancy above
    /// is only reachable once provisioning starts EMITTING progress, and a test
    /// that had to download llama.cpp to get there would not be a test.
    LlamaEngineProvisioner(std::string requested_executable,
                           std::string provision_dir,
                           LlamaCommandRunner runner = {});
    ~LlamaEngineProvisioner() override;

    const std::string& engine_id() const override;
    RuntimeStatus ensure(const EngineSpec& spec) override;
    RuntimeStatus check_for_update(const EngineSpec& spec) override;
    RuntimeStatus update(const EngineSpec& spec, const std::string& variant_override) override;
    RuntimeStatus status() const override;
    std::optional<EngineArtifact> installed_artifact() const override;
    std::optional<EngineArtifact> desired_artifact(const EngineSpec& spec) const override;
    bool shareable() const override;
    bool package(const std::string& out_path, std::string& err) override;
    bool install_package(const std::string& package_path,
                         const EngineArtifact& artifact,
                         std::string& err) override;
    std::string executable_path() const override;

    /// The llama-shaped detail the generic RuntimeStatus cannot carry: release
    /// variants, CUDA architecture, target mismatch, and the troubleshooting
    /// report. Kept llama-specific deliberately — generalizing a release-asset
    /// matrix would invent structure for engines that have none.
    LlamaRuntimeStatus llama_status() const;

    /// Wizard actions that exist only for llama.cpp.
    RuntimeStatus switch_variant(const std::string& variant);
    RuntimeStatus diagnose();
    RuntimeStatus recover(const std::string& action, const std::string& variant);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Soma — resolves the in-tree binary; there is nothing to build.
///
/// Soma is compiled as part of this repository and ships beside the node, so
/// this reports `ready` or `absent` and says so. Modelling a download it does
/// not do would produce a provisioning UI that can only ever fail, and a
/// conformance state that blames the cluster config for a missing build
/// artifact.
class SomaEngineProvisioner final : public EngineProvisioner {
public:
    explicit SomaEngineProvisioner(std::string requested_executable);
    ~SomaEngineProvisioner() override;

    const std::string& engine_id() const override;
    RuntimeStatus ensure(const EngineSpec& spec) override;
    RuntimeStatus check_for_update(const EngineSpec& spec) override;
    RuntimeStatus update(const EngineSpec& spec, const std::string& variant_override) override;
    RuntimeStatus status() const override;
    std::optional<EngineArtifact> installed_artifact() const override;
    std::optional<EngineArtifact> desired_artifact(const EngineSpec& spec) const override;
    bool shareable() const override;
    bool package(const std::string& out_path, std::string& err) override;
    bool install_package(const std::string& package_path,
                         const EngineArtifact& artifact,
                         std::string& err) override;
    std::string executable_path() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mm
