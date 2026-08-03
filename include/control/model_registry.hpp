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
    bool passed = false;
    std::string detail;
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
};

using AdmissionProgressSink = std::function<void(const AdmissionProgress&)>;

class ControlModelRegistry {
public:
    ControlModelRegistry();
    ~ControlModelRegistry();

    /// Opens {data_dir}/control.db, creating it and running pending migrations.
    bool open(const std::string& data_dir, std::string& out_error);
    void close();

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

    std::string reprofile(std::int64_t id, AdmissionProgressSink sink, std::string& out_error);
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
};

ModelVerdict parse_verdict(const std::string& text);
const char* to_string(ModelVerdict verdict);

/// True for Stream and Hybrid. ResidentOnly and Reject go to the fallback.
bool verdict_selects_soma(ModelVerdict verdict);

} // namespace mm
