#pragma once

// Mantic-Mind — control-side model registry and admission orchestration.
//
// Owns {data_dir}/control.db, the FIRST control-wide database in this system.
// Until now the only SQLite was per-agent (data/agents/{id}/agent.db), with
// remembered nodes in nodes.json and node model state in a JSON journal.
//
// Schema: schemas/registry/001_init.sql
// Migrations follow AgentDB::run_migrations() exactly — schema_migrations table,
// one `if (!has_version(N)) { Transaction; DDL; INSERT N; commit; }` block per
// version. That pattern works and there is no reason to invent a second one.

#include "common/footprint.hpp"
#include "common/models.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace mm {

enum class ModelVerdict : std::uint8_t { Stream, Hybrid, ResidentOnly, Reject };

struct AdmittedModel {
    std::int64_t id = 0;
    std::string arch_hash;
    std::string name;
    std::string source_repo;
    std::string source_revision;
    std::string model_dir;

    std::string attention_family;
    std::uint32_t n_layers = 0;
    std::uint32_t n_moe_layers = 0;
    std::uint32_t n_experts = 0;
    std::uint32_t top_k = 0;

    std::int64_t expert_bytes = 0;
    std::int64_t bytes_per_token = 0;
    std::int64_t total_routed_bytes = 0;
    double active_fraction = 0.0;

    /// The ADMISSION-HOST verdict.
    ///
    /// The verdict is a property of (model, quantization, host budget), not of
    /// the model — see schemas/arch-ir.md §8. Qwen3-30B-A3B is resident-only at
    /// q4 on a 32 GB host and stream at bf16. Placement therefore re-derives it
    /// from the target node's plan document; this field is the default and the
    /// answer to "what did admission conclude".
    ModelVerdict verdict = ModelVerdict::Reject;
    std::string verdict_basis; ///< JSON: the host assumptions behind it
    std::string verdict_reason;

    std::int64_t admitted_at_ms = 0;
    std::int64_t profiled_at_ms = 0;
};

struct ConformanceEntry {
    std::string stage;

    /// `passed` | `failed` | `skipped`.
    ///
    /// A third state, because a boolean cannot say "did not run" and that is the
    /// most common answer here: `fp32_tiny_tf` and `real_logit_kl` need a
    /// transformers oracle for THIS model, which the serving host does not have.
    /// Recording those as failures would reject every model; recording them as
    /// passes would make the verdict look validated when it was only computed.
    std::string status = "skipped";

    bool passed = false;
    std::string detail; ///< JSON: metrics, first divergence, thresholds, or why it was skipped
    std::int64_t ran_at_ms = 0;
};

/// Progress for the admission SSE stream. Conversion + profiling runs for hours,
/// so this is the difference between a usable operation and an opaque one.
struct AdmissionProgress {
    std::string operation_id;
    std::string stage; ///< fetch | convert | tokenize | conformance | profile | finalize
    std::string detail;
    int step = 0;
    int total_steps = 0;
    std::int64_t bytes_done = 0;
    std::int64_t bytes_total = 0;
    double fraction = 0.0;
    bool cancelable = true;
    std::string last_error;

    /// Terminal states. `done` without `last_error` is success; the two are
    /// separate because a client needs to stop waiting either way, and a stream
    /// that just goes quiet is indistinguishable from a network fault.
    bool done = false;
    bool canceled = false;
    std::int64_t model_id = 0; ///< set on success
    std::string source_ref;
    std::int64_t started_at_ms = 0;
    std::int64_t finished_at_ms = 0;
};

using AdmissionProgressSink = std::function<void(const AdmissionProgress&)>;

/// One in-flight (or finished) admission. Opaque here on purpose: it owns a
/// thread, a cancel flag and a sink list, none of which is anyone else's
/// business. Defined in model_registry.cpp.
struct AdmissionOperation;

/// Where the offline tools live, and how to run them.
///
/// Admission is orchestration, not reimplementation: the converter and the
/// tokenizer compiler are Python and stay Python, because they read HF
/// checkpoints and that ecosystem is theirs. Control runs them as subprocesses
/// and streams their output. `tools/admission/` is NEVER a runtime dependency —
/// it is a dependency of ADMITTING, which happens once per model.
struct AdmissionTools {
    std::string python = "python"; ///< MM_ADMISSION_PYTHON
    std::string tools_dir = "tools/admission";
    std::string soma_path = "soma"; ///< for `soma plan --json`
    std::string containers_dir = "data/containers";

    /// Where fetched repos land. Separate from containers_dir because these are
    /// the ORIGINAL weights: conversion reads them and never writes them, and an
    /// operator reclaiming disk wants to delete one without touching the other.
    std::string sources_dir = "data/sources"; ///< MM_SOURCES_DIR

    /// Permit `.bin` weights when a repo publishes no safetensors. Off by
    /// default: converting a pickle executes code from the repo, and that is a
    /// decision an operator makes per model rather than a default they inherit.
    bool allow_pickle = false;

    /// Quantization for the converted container. Part of the verdict's identity:
    /// the same weights at a different quant are a different admission.
    std::string quant = "q4_g";
    std::string expert_down = "q6_g";
    int group = 128;
};

/// Is `ref` a HuggingFace repo id this pipeline is willing to fetch?
///
/// Accepts `name`, `org/name`, and either with an `@revision` suffix. Validated
/// rather than trusted, because the id becomes a DIRECTORY NAME under
/// `sources_dir`: `../../etc` is a legal-looking string and an illegal path.
/// Exposed as a free function so the rule can be tested for itself rather than
/// only through a download that fails.
bool valid_repo_id(const std::string& ref, std::string& out_why);

/// The model name that both the container variant and the fetch destination are
/// built from.
///
/// The trailing component either way — a repo id `Qwen/Qwen3-30B-A3B` and a
/// directory `.../Qwen3-30B-A3B` are the same model and must produce the same
/// container, or admitting one after the other silently makes two. Shared so
/// `sources/<name>` and `containers/<name>-…` cannot disagree about which model
/// a ref denotes.
std::string admission_source_name(const std::string& source, bool needs_fetch);

/// The container directory name an admission of `source` will write:
/// `<name>-<quant>-<expert_down>-g<group>`.
///
/// Extracted because it is the COLLISION KEY as well as the write path, and the
/// two must not be able to disagree. run_admission derived it inline; adding a
/// second derivation for the in-flight check would have been the same defect one
/// layer up — a guard watching a different directory from the one being written
/// is a guard that passes while the corruption happens.
///
/// Pure, so the identity rule ("these two refs are one model") can be asserted
/// without converting anything.
std::string
admission_variant(const std::string& source, bool needs_fetch, const AdmissionTools& tools);

/// One row of api_token. The token itself is NEVER stored — only its SHA-256 —
/// so a leaked database backup does not hand over working credentials.
struct ApiToken {
    std::int64_t id = 0;
    std::string token_sha256;
    std::string label;
    std::uint8_t scopes = 0; ///< ScopeSet; see control/route_scope.hpp
    std::int64_t created_at_ms = 0;
    std::int64_t last_used_at_ms = 0;
    bool revoked = false;
};

/// One row of `placement_history` — where an agent ran, on what engine, and why.
///
/// The reason is the same composed string the scheduler acted on
/// (`BackendDecision::explain()`), not a reconstruction, which is the whole
/// point of recording it at placement time: after the fact the admission record
/// may have changed and the decision would no longer be re-derivable.
struct PlacementHistoryEntry {
    NodeId node_id;
    SlotId slot_id;
    std::string backend;
    std::string backend_reason;
    std::int64_t vram_mb = 0;
    std::int64_t ram_mb = 0;
    std::int64_t disk_mb = 0;
    std::int64_t placed_at_ms = 0;

    /// 0 while the placement is still live. The column is nullable and this is
    /// not; zero reads as "still open" everywhere it is rendered, and an epoch
    /// timestamp of 0 is not a plausible real value.
    std::int64_t released_at_ms = 0;

    bool open() const noexcept { return released_at_ms == 0; }
};

class ControlModelRegistry {
public:
    ControlModelRegistry();
    ~ControlModelRegistry();

    /// Opens {data_dir}/control.db, creating it and running pending migrations.
    bool open(const std::string& data_dir, std::string& out_error);
    void close();

    void set_tools(const AdmissionTools& tools);

    /// How many admissions may RUN at once; the rest wait and report `queued`.
    ///
    /// One by default. A conversion spawns Python and moves tens to hundreds of
    /// gigabytes, so two on one box do not go twice as fast — they contend for
    /// the same disk and RAM. Before this there was no limit at all: every
    /// `admit()` detached a thread, so N requests meant N conversions.
    ///
    /// Clamped to at least 1, because 0 would deadlock every admission on a gate
    /// nothing can open.
    void set_max_concurrent_admissions(std::size_t n);
    std::size_t max_concurrent_admissions() const;
    AdmissionTools tools() const;

    /// Every admission this process has run, newest first. Survives the SSE
    /// stream disconnecting — a client that loses its connection mid-conversion
    /// must be able to find out how it ended.
    std::vector<AdmissionProgress> operations() const;
    std::optional<AdmissionProgress> operation(const std::string& id) const;

    std::uint32_t schema_version() const;

    // ── queries ──────────────────────────────────────────────────────────────
    std::vector<AdmittedModel> list() const;
    std::optional<AdmittedModel> find_by_id(std::int64_t id) const;
    std::optional<AdmittedModel> find_by_arch_hash(const std::string& arch_hash) const;

    /// Resolve an agent's model_path to an admission record.
    ///
    /// nullopt routes to the FALLBACK. Absence of a record is not evidence of
    /// admissibility, and defaulting the other way would send unvalidated models
    /// to the engine with the stricter requirements.
    std::optional<AdmittedModel> resolve(const std::string& model_ref) const;

    std::vector<ConformanceEntry> conformance(std::int64_t id) const;

    /// Replace this model's conformance rows with `stages`.
    ///
    /// Replace rather than append: a reprofile re-runs the ladder against the
    /// same weights, and its answer supersedes the old one rather than
    /// accumulating a history nobody reads.
    bool record_conformance(std::int64_t id,
                            const std::vector<ConformanceEntry>& stages,
                            std::string& out_error);

    /// Persisted routing histogram. `bucketed` caps the result at the telemetry
    /// grid size so a careless client cannot ask for 60k rows by accident.
    bool heat(std::int64_t id, bool bucketed, std::string& out_json) const;

    // ── admission ────────────────────────────────────────────────────────────
    /// Long-running: fetch, convert, compile the tokenizer, run the conformance
    /// ladder, profile, write registry rows, compute a verdict. Runs on its own
    /// thread and reports through the sink.
    ///
    /// Requires the `operator` scope. This is why scopes exist at all: hours of
    /// CPU and tens of GB of disk must not sit behind the same token that lets a
    /// client send a chat message.
    std::string
    admit(const std::string& source_ref, AdmissionProgressSink sink, std::string& out_error);

    /// Same, at a quantization other than the deployment's default.
    ///
    /// Per REQUEST rather than per deployment, because the same weights at two
    /// quantizations are two different admissions with two different verdicts —
    /// that is the premise the registry keys on, and it cannot be exercised at
    /// all if the quantization is a config file the operator has to edit and
    /// restart to change.
    ///
    /// Empty fields fall back to the deployment default, so a caller changing
    /// only the group does not have to restate the dtypes.
    struct QuantOverride {
        std::string quant;       ///< expert gate/up dtype, e.g. "q4_g"
        std::string expert_down; ///< expert down dtype, e.g. "q6_g"
        int group = 0;           ///< 0 = leave the default
    };

    std::string admit(const std::string& source_ref,
                      const QuantOverride& quant,
                      AdmissionProgressSink sink,
                      std::string& out_error);

    /// Admit a container that has ALREADY been converted.
    ///
    /// Same pipeline minus conversion: plan, then record. This is the path for a
    /// model converted by hand with tools/admission/convert.py, and it is what
    /// reprofile() runs — re-deriving a verdict should not rewrite gigabytes to
    /// arrive at the same bytes.
    std::string admit_container(const std::string& container_dir,
                                AdmissionProgressSink sink,
                                std::string& out_error);

    /// Write (or update) a record for an ALREADY-CONVERTED model.
    ///
    /// The write primitive admit() ends with, exposed on its own because the
    /// conversion pipeline is offline (tools/admission/) and a model converted
    /// there still has to become evidence here. Keyed on arch_hash, which is the
    /// model's identity — re-registering the same weights updates rather than
    /// duplicating, and requantized weights are a different row because they
    /// have a different verdict.
    ///
    /// `model.id` is filled in on success.
    bool upsert(AdmittedModel& model, std::string& out_error);

    // ── api tokens ───────────────────────────────────────────────────────────
    //
    // Admission is why scopes exist: hours of CPU and tens of GB of disk must not
    // sit behind the token that lets a client send a chat message.

    /// Mint a token. Returns the SECRET, which is shown once and never stored —
    /// only its hash is. A caller that loses it mints another.
    std::string
    create_api_token(const std::string& label, std::uint8_t scopes, std::string& out_error);

    /// By hash, and only if not revoked. Touches last_used_at.
    bool find_api_token(const std::string& token_sha256, ApiToken& out) const;

    std::vector<ApiToken> list_api_tokens() const;
    bool revoke_api_token(std::int64_t id, std::string& out_error);

    /// Whether ANY usable token exists. Together with the legacy config token,
    /// this is what decides whether auth is on at all.
    bool has_api_tokens() const;

    /// Record which backend an agent actually got, and why.
    ///
    /// Rare, causally interesting, and otherwise unrecoverable after the fact.
    /// Never fatal: losing an audit row must not fail a placement that worked.
    void record_placement(const AgentId& agent_id,
                          const NodeId& node_id,
                          const SlotId& slot_id,
                          const std::string& backend,
                          const std::string& backend_reason,
                          const ResourceFootprint& footprint);

    /// Close the agent's most recent open row. Idempotent: an agent with no open
    /// row is a no-op, not an error, because a release can legitimately arrive
    /// for a placement this process never recorded (control restarted).
    void mark_placement_released(const AgentId& agent_id);

    /// The agent's placement history, newest first, at most `limit` rows.
    ///
    /// A READER is what makes the writer worth having. The table, its index and
    /// `record_placement()` all shipped with no caller and no query — a schema
    /// created on every start for a history nothing recorded and nothing could
    /// read (roadmap D60). Wiring only the writer would have been worse: an
    /// unbounded write-only table is a capability with no way to reach it, which
    /// is the thing P1 exists to forbid.
    std::vector<PlacementHistoryEntry> placement_history(const AgentId& agent_id,
                                                         int limit = 20) const;

    std::string reprofile(std::int64_t id, AdmissionProgressSink sink, std::string& out_error);
    /// Watch an operation that is already running.
    ///
    /// The SSE route for an admission that started before this client connected.
    /// `out_current` always receives the latest snapshot, so a caller learns the
    /// outcome of an operation that has already finished rather than attaching
    /// to a stream that will never speak again. Returns false for an unknown id.
    bool attach_sink(const std::string& operation_id,
                     AdmissionProgressSink sink,
                     AdmissionProgress& out_current);

    bool cancel(const std::string& operation_id);
    bool remove(std::int64_t id, std::string& out_error);

    /// Operator override of a computed verdict. Recorded with a reason so the
    /// override is visible rather than mysterious.
    bool set_verdict(std::int64_t id,
                     ModelVerdict verdict,
                     const std::string& reason,
                     std::string& out_error);

    /// The plan document for this model against a specific node's capacity.
    /// Served verbatim by GET /v1/models/{id}/plan.
    bool plan_for_host(std::int64_t id,
                       const HostCapacity& capacity,
                       std::string& out_json,
                       std::string& out_error) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    /// The staged pipeline, on the operation's own thread.
    ///
    /// `container_is_ready` skips conversion: reprofile() re-derives a verdict
    /// for a container that already exists, and rewriting gigabytes to reach the
    /// same bytes is not what "re-profile" should mean.
    /// Register an operation and start its thread. Shared by admit(),
    /// admit_container() and reprofile(), which differ only in whether
    /// conversion runs and what they validate first.
    std::string start_operation(const std::string& source,
                                bool container_is_ready,
                                const QuantOverride& quant,
                                AdmissionProgressSink sink);
    std::mutex& ops_mu_ref() const;

    void run_admission(std::shared_ptr<AdmissionOperation> op,
                       const std::string& source,
                       const AdmissionTools& tools,
                       bool container_is_ready = false);
};

ModelVerdict parse_verdict(const std::string& text);
const char* to_string(ModelVerdict verdict);

/// True for Stream and Hybrid. ResidentOnly and Reject go to the fallback.
bool verdict_selects_soma(ModelVerdict verdict);

} // namespace mm
