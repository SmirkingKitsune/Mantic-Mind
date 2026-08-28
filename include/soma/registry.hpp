#pragma once

// Soma — the model registry (SQLite, control-side).
//
// This is the first control-wide database in Mantic-Mind: until now the only
// SQLite was per-agent (data/agents/{id}/agent.db), with node state in a JSON
// journal and remembered nodes in nodes.json.
//
// Schema and migration mechanism: schemas/registry/001_init.sql

#include "soma/arch_ir.hpp"
#include "soma/kernels.hpp"
#include "soma/memory_hierarchy.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace soma {

inline constexpr std::uint32_t kRegistrySchemaVersion = 1;

struct ModelRecord {
    std::int64_t id = 0;
    std::string arch_hash;
    std::string name;
    std::string source_repo;
    std::string source_revision;
    std::string model_dir;

    AttentionFamily attention_family = AttentionFamily::Unknown;
    std::uint32_t n_layers = 0;
    std::uint32_t n_moe_layers = 0;
    std::uint32_t n_experts = 0;
    std::uint32_t top_k = 0;

    Economics economics{};

    /// The ADMISSION-HOST verdict. Consumers wanting the effective verdict for a
    /// particular node must read the plan, not this field.
    Verdict verdict = Verdict::Reject;
    std::string verdict_basis; ///< JSON: the host assumptions behind it
    std::string verdict_reason;

    std::uint64_t admitted_at_ms = 0;
    std::uint64_t profiled_at_ms = 0;
};

enum class ConformanceStage : std::uint8_t {
    Fp32TinyTeacherForced = 0,
    QuantTinyGreedy,
    RealLogitKl,
    AccuracyFloor,
};

struct ConformanceResult {
    ConformanceStage stage = ConformanceStage::Fp32TinyTeacherForced;
    bool passed = false;
    std::string detail; ///< JSON: metrics, first divergence, thresholds
    std::uint64_t ran_at_ms = 0;
};

struct PilotProfileRow {
    LayerIndex layer = kInvalidLayer;
    float recall_at_k = 0.0f;
    std::uint32_t samples = 0;
};

/// API token scopes. Mantic-Mind has had exactly one flat bearer token with no
/// scope mechanism; this is the first.
///
/// Admission starts hours-long, resource-consuming conversion and profiling. It
/// must not sit behind the same token that lets a client send a chat message.
enum class Scope : std::uint8_t {
    Read = 1 << 0,     ///< all GETs, telemetry SSE
    Chat = 1 << 1,     ///< agent chat, conversations, memories, attachments
    Operator = 1 << 2, ///< admit, reprofile, delete, override, suspend/restore
};

using ScopeSet = std::uint8_t;

inline constexpr ScopeSet kAllScopes = static_cast<ScopeSet>(Scope::Read) |
                                       static_cast<ScopeSet>(Scope::Chat) |
                                       static_cast<ScopeSet>(Scope::Operator);

struct TokenRecord {
    std::int64_t id = 0;
    std::string label;
    ScopeSet scopes = 0;
    std::uint64_t created_at_ms = 0;
    std::uint64_t last_used_at_ms = 0;
    bool revoked = false;
};

class Registry {
public:
    Registry();
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;
    ~Registry();

    /// Opens (creating if absent) and runs pending migrations.
    Status open(const std::string& db_path);
    void close();

    std::uint32_t schema_version() const noexcept;

    // ── models ───────────────────────────────────────────────────────────────
    Status upsert_model(const ModelRecord& record, std::int64_t& out_id);
    std::optional<ModelRecord> find_by_arch_hash(const std::string& arch_hash) const;
    std::optional<ModelRecord> find_by_id(std::int64_t id) const;

    /// Resolve a model reference (name, repo id, or directory) to a record.
    ///
    /// Returns nullopt when there is no admission record — which routes to the
    /// FALLBACK. Absence of a record is not evidence of admissibility.
    std::optional<ModelRecord> resolve(const std::string& model_ref) const;

    std::vector<ModelRecord> list_models() const;
    Status delete_model(std::int64_t id);
    Status set_verdict(std::int64_t id, Verdict verdict, const std::string& reason);

    // ── profiling artifacts ──────────────────────────────────────────────────
    Status put_heat(std::int64_t id, const HeatSnapshot& heat);
    Status get_heat(std::int64_t id, HeatSnapshot& out) const;

    Status put_kernel_choices(std::int64_t id, std::span<const KernelChoice> choices);
    Status get_kernel_choices(std::int64_t id, std::vector<KernelChoice>& out) const;

    Status put_pilot_profile(std::int64_t id, std::span<const PilotProfileRow> rows);
    Status get_pilot_profile(std::int64_t id, std::vector<PilotProfileRow>& out) const;

    Status put_conformance(std::int64_t id, const ConformanceResult& result);
    Status get_conformance(std::int64_t id, std::vector<ConformanceResult>& out) const;

    // ── tokens ───────────────────────────────────────────────────────────────
    /// Stores only the SHA-256 of the token; the plaintext is returned to the
    /// caller once and never persisted.
    Status create_token(const std::string& label,
                        ScopeSet scopes,
                        std::string& out_plaintext,
                        std::int64_t& out_id);
    std::optional<TokenRecord> authenticate(const std::string& bearer) const;
    Status revoke_token(std::int64_t id);
    std::vector<TokenRecord> list_tokens() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool scope_permits(ScopeSet held, Scope required) noexcept;
const char* to_string(Scope scope) noexcept;
const char* to_string(ConformanceStage stage) noexcept;
Status parse_scopes(const std::string& csv, ScopeSet& out);
std::string format_scopes(ScopeSet scopes);

} // namespace soma
