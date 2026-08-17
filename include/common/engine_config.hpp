#pragma once

// Mantic-Mind — the cluster engine configuration.
//
// WHAT THE CLUSTER IS SUPPOSED TO RUN, stated once, by the master.
//
// Before this, every engine setting lived in NodeConfig and was read from a
// node-local .toml: which llama.cpp to build, at what version, by what method,
// and whether to provision one at all. Control could observe the RESULT
// (NodeInfo::llama_runtime) and do nothing about it — there was no control->node
// provisioning route anywhere. Ten nodes could run ten different builds and the
// cluster head would render all ten as healthy, because nothing anywhere stated
// what they were supposed to be running.
//
// This type is that statement. It lives in common/ rather than control/ for the
// same reason engine_capabilities.hpp does: control OWNS it, nodes CONSUME it,
// and a copy on each side is how the two drift.
//
// ── The invariant ─────────────────────────────────────────────────────────────
//
// This carries NO accelerator, NO cuda_arch, and NO executable path, and the
// omission is load-bearing rather than an oversight. A Metal Mac and a CUDA box
// cannot share those values, so a cluster config that carried them would make
// every heterogeneous cluster permanently non-conforming — the operator would
// state a fact that most of the cluster is structurally unable to satisfy.
//
// The split is: master states INTENT (which engines, which version, by what
// method, updated how), and each node RESOLVES that intent against hardware it
// alone can see. LlamaEngineProvisioner is where the resolution physically
// happens.
//
// The invariant is enforced, not merely documented: from_json REFUSES a
// forbidden key rather than ignoring it (see kForbiddenConfigKeys). Tolerating
// it would be worse than rejecting it — an operator who sets `accelerator`,
// sees the config accepted, and then watches every node drift has been told a
// lie by the system that accepted the write.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mm {

/// The default backup engine when an operator does not choose otherwise.
///
/// llama.cpp is the BACKUP, not the floor: `ClusterEngineConfig::backup_engine`
/// may be empty, and an empty value is a real configuration rather than an
/// unset one. A Soma-only cluster should not pay for a llama.cpp build it will
/// never launch, and the node provisions nothing that the cluster config does
/// not name.
inline constexpr const char* kDefaultBackupEngine = "llama-cpp";

/// One engine's policy. Every field here is a property of the CLUSTER's intent.
/// Anything that varies per machine is deliberately absent — see the header
/// comment.
struct EngineSpec {
    std::string engine_id;          ///< "soma" | "llama-cpp"; validated against EngineRegistry ids
    std::string version = "latest"; ///< git ref / release tag, or "latest"
    std::string install_method = "auto";  ///< auto|release|source|path
    std::string update_policy = "prompt"; ///< prompt|auto|manual
    bool update_check = true;
    int update_check_interval_hours = 24;

    /// Extra -D flags for a source build. Cluster-uniform by definition: a flag
    /// that only makes sense on one machine belongs to that machine's detection,
    /// not to the cluster's policy.
    std::vector<std::string> cmake_args;

    /// 0 = the node picks a conservative accelerator-aware default. A cluster
    /// may state a number when its members are known to be alike; it is intent,
    /// not detection, which is why it is allowed to live here.
    int build_jobs = 0;
};

/// The master's engine policy for the whole cluster.
struct ClusterEngineConfig {
    /// Monotonic, bumped on every save. THE convergence signal: a node reports
    /// the version it last applied, control compares, and a mismatch on the
    /// existing health poll is what triggers a push. No new polling thread, and
    /// a node that was offline during a change converges on its next successful
    /// poll rather than staying silently stale.
    std::uint32_t version = 0;

    std::string primary_engine; ///< required; must have a matching spec

    /// Empty means NO backup, which is a supported configuration and not a
    /// missing value. Defaults to kDefaultBackupEngine at first-run setup, and
    /// is explicitly clearable from there.
    std::string backup_engine;

    std::vector<EngineSpec> engines;

    /// May a node that has built an engine serve it to a node that needs the
    /// same one? Off means every node builds alone.
    bool share_builds = true;

    std::int64_t updated_at_ms = 0;
    std::string updated_by;

    /// The spec for `engine_id`, or nullptr. Does not synthesize a default: an
    /// engine named without a spec is a validation failure, and inventing one
    /// here would hide it.
    const EngineSpec* find(const std::string& engine_id) const noexcept;

    /// primary + backup, in that order, skipping an empty backup. THE list a
    /// node provisions — nothing outside it is provisioned, which is what makes
    /// the backup optional in practice rather than only on paper.
    std::vector<std::string> required_engines() const;
};

/// The identity of one built engine — and therefore what makes two nodes' needs
/// the same need.
///
/// All five fields are part of the identity. A build differing in ANY of them is
/// a different binary, not a near-match: an x86_64 CUDA-12 artifact on an
/// aarch64 host does not run, and a cuda-12 build on a cuda-13 host is the exact
/// mismatch the release matrix exists to prevent. Sharing compares the whole
/// fingerprint and refuses anything else.
///
/// Shared rather than node-local because control BROKERS transfers: it builds
/// fingerprints from what nodes report to find a source for a need, so a second
/// definition on that side would be two spellings of one identity.
struct EngineArtifact {
    std::string engine_id;
    std::string version;  ///< resolved build id, never "latest"
    std::string platform; ///< windows|linux|macos
    std::string arch;     ///< x86_64|aarch64|arm64
    std::string variant;  ///< cuda-12|vulkan|cpu|... ; "" when the engine has none

    /// Digest over the packaged artifact. On the RECEIVING side this is never
    /// taken from the sending node: control relays the digest the source
    /// attested before the transfer credential existed.
    std::string sha256;

    /// The shareable key. Excludes sha256: two independent builds of the same
    /// source at the same version are the same NEED even when their bytes
    /// differ, and it is the need that has to match before a transfer is worth
    /// brokering.
    std::string fingerprint() const;

    bool valid() const noexcept;
};

bool parse_engine_fingerprint(const std::string& fingerprint, EngineArtifact& out);

/// Per-engine provisioning/health status, keyed by engine id.
///
/// Generalizes LlamaRuntimeStatus, which was a single scalar on NodeState and
/// therefore the only engine whose health control could see at all.
///
/// It lives here rather than in node/engine_descriptor.hpp because control now
/// holds a vector of these per node: an engine status the node alone could name
/// is the exact shape of the problem the cluster configurator exists to fix.
/// The llama-shaped extras — release variants, CUDA architecture, the
/// troubleshooting report — stay in LlamaRuntimeStatus, because generalizing a
/// release-asset matrix would invent structure for engines that have none.
struct RuntimeStatus {
    std::string engine_id;
    std::string status; ///< resolved | ready | building | error | absent
    std::string executable_path;
    std::string version;
    std::string variant;
    std::string last_error;
    bool ready = false;
};

/// How a node stands relative to the cluster config.
///
/// `Drifted` and `Failed` are distinct on purpose. Failed is "I tried and could
/// not" — a build error, an unavailable release, a refused artifact. Drifted is
/// "I am running something other than what was asked", which includes the node
/// that has not yet been told. Both stop placement; only one is a fault.
enum class EngineConformanceState : std::uint8_t {
    Unconfigured = 0, ///< no cluster config has reached this node yet
    Converging,       ///< applying: provisioning, building, or awaiting an artifact
    Conforming,       ///< every required engine is ready at the configured version
    Drifted,          ///< running something other than the configured set
    Failed,           ///< could not satisfy the config; `detail` says why
};

struct EngineConformance {
    EngineConformanceState state = EngineConformanceState::Unconfigured;

    /// Which cluster config version this reflects. Compared against
    /// ClusterEngineConfig::version to decide whether to push.
    std::uint32_t config_version = 0;

    /// Human-readable, and required to be non-empty for Drifted and Failed. A
    /// state that stops placement without saying why sends the operator to the
    /// logs of the wrong machine.
    std::string detail;

    /// Artifact fingerprint this node is waiting on, or empty. Set while
    /// Converging when the node cannot build or fetch an engine itself; control
    /// reads it to find a node that already has one.
    std::string needs_artifact;
};

/// Whether a node in this state may receive new placements.
///
/// One predicate, because the alternative is the same comparison written at
/// four call sites in the scheduler, which is how the fourth one gets missed.
bool conformance_permits_placement(const EngineConformance& c) noexcept;

/// Config keys that must never appear. Rejected by from_json rather than
/// ignored — see the header comment.
const std::vector<std::string>& forbidden_config_keys();

/// Structural validation. Returns false and populates `out_error` on:
///   * empty primary_engine
///   * primary or backup named with no matching spec
///   * backup_engine == primary_engine
///   * a spec with an empty engine_id, or two specs sharing one
///   * an install_method / update_policy outside its enumerated set
///   * negative build_jobs or update_check_interval_hours
///
/// `known_engine_ids` is the registry's ids(). Empty means "do not check ids" —
/// control validates against what a node can actually run, and a caller with no
/// registry (a unit test, a config file read before startup) should not be
/// forced to invent one.
bool validate_engine_config(const ClusterEngineConfig& cfg,
                            const std::vector<std::string>& known_engine_ids,
                            std::string& out_error);

const char* to_string(EngineConformanceState state) noexcept;
bool parse_conformance_state(const std::string& s, EngineConformanceState& out) noexcept;

void to_json(nlohmann::json& j, const EngineSpec& s);
void from_json(const nlohmann::json& j, EngineSpec& s);
void to_json(nlohmann::json& j, const ClusterEngineConfig& c);
/// Throws std::invalid_argument when a forbidden key is present, naming it.
void from_json(const nlohmann::json& j, ClusterEngineConfig& c);
void to_json(nlohmann::json& j, const EngineConformance& c);
void from_json(const nlohmann::json& j, EngineConformance& c);
void to_json(nlohmann::json& j, const RuntimeStatus& s);
void from_json(const nlohmann::json& j, RuntimeStatus& s);

} // namespace mm
