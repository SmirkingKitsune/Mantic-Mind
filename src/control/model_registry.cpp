// Mantic-Mind — the control-side model registry.
//
// Owns {data_dir}/control.db, the FIRST control-wide database in this system:
// until now the only SQLite was per-agent (data/agents/{id}/agent.db), with
// remembered nodes in nodes.json and node model state in a JSON journal.
//
// What it is FOR, in one line: without an admission record, select_backend()
// routes every agent to the fallback, because absence of a record is not
// evidence of admissibility. This table is what lets a model be evidence.
//
// Schema in schemas/registry/001_init.sql. Migrations follow
// AgentDB::run_migrations() exactly — a schema_migrations table and one
// `if (!has_version(N)) { Transaction; DDL; INSERT N; commit; }` block per
// version. That pattern works; there was no reason to invent a second one.

#include "control/model_registry.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"

#include <SQLiteCpp/SQLiteCpp.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <mutex>

namespace fs = std::filesystem;

namespace mm {

namespace {

/// A model_ref may be a path, a HF-style "org/name", or a bare name. Compared
/// case-insensitively on the trailing component so an agent configured with
/// "Qwen/Qwen3-30B-A3B" resolves against a record admitted from a local
/// directory ending in the same name.
std::string ref_key(const std::string& ref) {
    auto trimmed = util::trim(ref);
    if (trimmed.empty()) return {};
    // Strip a trailing separator so ".../Qwen3-30B-A3B/" and ".../Qwen3-30B-A3B"
    // are the same model rather than one match and one miss.
    while (trimmed.size() > 1 && (trimmed.back() == '/' || trimmed.back() == '\\')) {
        trimmed.pop_back();
    }
    const auto slash = trimmed.find_last_of("/\\");
    if (slash != std::string::npos) trimmed = trimmed.substr(slash + 1);
    return util::to_lower(trimmed);
}

} // namespace

ModelVerdict parse_verdict(const std::string& text) {
    const auto v = util::to_lower(util::trim(text));
    if (v == "stream") return ModelVerdict::Stream;
    if (v == "hybrid") return ModelVerdict::Hybrid;
    if (v == "resident-only" || v == "resident_only") return ModelVerdict::ResidentOnly;
    // Reject is the default for anything unrecognised, deliberately: an
    // unparseable verdict must not become a licence to stream.
    return ModelVerdict::Reject;
}

const char* to_string(ModelVerdict verdict) {
    switch (verdict) {
    case ModelVerdict::Stream:
        return "stream";
    case ModelVerdict::Hybrid:
        return "hybrid";
    case ModelVerdict::ResidentOnly:
        return "resident-only";
    case ModelVerdict::Reject:
        return "reject";
    }
    return "reject";
}

bool verdict_selects_soma(ModelVerdict verdict) {
    return verdict == ModelVerdict::Stream || verdict == ModelVerdict::Hybrid;
}

struct ControlModelRegistry::Impl {
    mutable std::mutex mu;
    std::unique_ptr<SQLite::Database> db;
    std::string path;

    bool has_version(std::uint32_t v) const {
        SQLite::Statement q(*db, "SELECT 1 FROM schema_migrations WHERE version = ? LIMIT 1");
        q.bind(1, static_cast<int>(v));
        return q.executeStep();
    }

    void run_migrations() {
        db->exec("PRAGMA journal_mode = WAL");
        db->exec("PRAGMA foreign_keys = ON");
        db->exec(R"(
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    INTEGER NOT NULL PRIMARY KEY,
                applied_at INTEGER NOT NULL
                           DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000)
            ))");

        if (!has_version(1)) {
            SQLite::Transaction tx(*db);
            // Kept byte-for-byte in step with schemas/registry/001_init.sql. The
            // .sql file is the reviewable artifact and the reference for
            // `sqlite3 < 001_init.sql`; this is what actually runs. A migration
            // that has been applied is never edited — the next one is added.
            db->exec(R"(
                CREATE TABLE IF NOT EXISTS model (
                    id                   INTEGER PRIMARY KEY,
                    arch_hash            TEXT    NOT NULL UNIQUE,
                    name                 TEXT    NOT NULL,
                    source_repo          TEXT,
                    source_revision      TEXT,
                    model_dir            TEXT    NOT NULL,
                    schema_version       INTEGER NOT NULL,
                    arch_json            TEXT    NOT NULL,
                    attention_family     TEXT    NOT NULL,
                    n_layers             INTEGER NOT NULL,
                    n_moe_layers         INTEGER NOT NULL,
                    n_experts            INTEGER NOT NULL,
                    top_k                INTEGER NOT NULL,
                    expert_bytes         INTEGER NOT NULL,
                    bytes_per_token      INTEGER NOT NULL,
                    total_routed_bytes   INTEGER NOT NULL,
                    dense_resident_bytes INTEGER NOT NULL,
                    active_fraction      REAL    NOT NULL,
                    measured_disk_bw     INTEGER,
                    verdict              TEXT    NOT NULL
                                         CHECK (verdict IN ('stream','hybrid',
                                                            'resident-only','reject')),
                    verdict_basis        TEXT    NOT NULL,
                    verdict_reason       TEXT,
                    admitted_at          INTEGER NOT NULL,
                    profiled_at          INTEGER
                ))");
            db->exec("CREATE INDEX IF NOT EXISTS idx_model_verdict ON model(verdict)");
            db->exec("CREATE INDEX IF NOT EXISTS idx_model_name ON model(name)");

            db->exec(R"(
                CREATE TABLE IF NOT EXISTS expert_heat (
                    model_id   INTEGER NOT NULL REFERENCES model(id) ON DELETE CASCADE,
                    layer      INTEGER NOT NULL,
                    expert     INTEGER NOT NULL,
                    count      INTEGER NOT NULL DEFAULT 0,
                    decayed    REAL    NOT NULL DEFAULT 0.0,
                    tier       TEXT    NOT NULL DEFAULT 'disk'
                               CHECK (tier IN ('vram','ram','disk')),
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (model_id, layer, expert)
                ))");
            db->exec("CREATE INDEX IF NOT EXISTS idx_heat_rank "
                     "ON expert_heat(model_id, decayed DESC)");

            db->exec(R"(
                CREATE TABLE IF NOT EXISTS kernel_choice (
                    model_id INTEGER NOT NULL REFERENCES model(id) ON DELETE CASCADE,
                    op       TEXT    NOT NULL,
                    m        INTEGER NOT NULL,
                    n        INTEGER NOT NULL,
                    k        INTEGER NOT NULL,
                    dtype    TEXT    NOT NULL,
                    impl     TEXT    NOT NULL,
                    gflops   REAL    NOT NULL,
                    PRIMARY KEY (model_id, op, m, n, k, dtype)
                ))");

            db->exec(R"(
                CREATE TABLE IF NOT EXISTS pilot_profile (
                    model_id    INTEGER NOT NULL REFERENCES model(id) ON DELETE CASCADE,
                    layer       INTEGER NOT NULL,
                    recall_at_k REAL    NOT NULL,
                    samples     INTEGER NOT NULL,
                    PRIMARY KEY (model_id, layer)
                ))");

            db->exec(R"(
                CREATE TABLE IF NOT EXISTS conformance (
                    model_id INTEGER NOT NULL REFERENCES model(id) ON DELETE CASCADE,
                    stage    TEXT    NOT NULL
                             CHECK (stage IN ('fp32_tiny_tf','quant_tiny_greedy',
                                              'real_logit_kl','accuracy_floor')),
                    passed   INTEGER NOT NULL,
                    detail   TEXT,
                    ran_at   INTEGER NOT NULL,
                    PRIMARY KEY (model_id, stage)
                ))");

            db->exec(R"(
                CREATE TABLE IF NOT EXISTS api_token (
                    id           INTEGER PRIMARY KEY,
                    token_sha256 TEXT    NOT NULL UNIQUE,
                    label        TEXT    NOT NULL,
                    scopes       TEXT    NOT NULL,
                    created_at   INTEGER NOT NULL,
                    last_used_at INTEGER,
                    revoked_at   INTEGER
                ))");
            db->exec("CREATE INDEX IF NOT EXISTS idx_token_active "
                     "ON api_token(token_sha256) WHERE revoked_at IS NULL");

            db->exec(R"(
                CREATE TABLE IF NOT EXISTS placement_history (
                    id             INTEGER PRIMARY KEY,
                    agent_id       TEXT    NOT NULL,
                    node_id        TEXT    NOT NULL,
                    slot_id        TEXT    NOT NULL,
                    model_id       INTEGER REFERENCES model(id) ON DELETE SET NULL,
                    backend        TEXT    NOT NULL,
                    backend_reason TEXT    NOT NULL,
                    footprint_json TEXT    NOT NULL,
                    placed_at      INTEGER NOT NULL,
                    released_at    INTEGER
                ))");
            db->exec("CREATE INDEX IF NOT EXISTS idx_placement_agent "
                     "ON placement_history(agent_id, placed_at DESC)");

            db->exec("INSERT OR IGNORE INTO schema_migrations(version) VALUES (1)");
            tx.commit();
        }
    }

    static AdmittedModel read_row(SQLite::Statement& q) {
        AdmittedModel m;
        m.id = q.getColumn("id").getInt64();
        m.arch_hash = q.getColumn("arch_hash").getText();
        m.name = q.getColumn("name").getText();
        m.source_repo = q.getColumn("source_repo").getText();
        m.source_revision = q.getColumn("source_revision").getText();
        m.model_dir = q.getColumn("model_dir").getText();
        m.attention_family = q.getColumn("attention_family").getText();
        m.n_layers = static_cast<std::uint32_t>(q.getColumn("n_layers").getInt());
        m.n_moe_layers = static_cast<std::uint32_t>(q.getColumn("n_moe_layers").getInt());
        m.n_experts = static_cast<std::uint32_t>(q.getColumn("n_experts").getInt());
        m.top_k = static_cast<std::uint32_t>(q.getColumn("top_k").getInt());
        m.expert_bytes = q.getColumn("expert_bytes").getInt64();
        m.bytes_per_token = q.getColumn("bytes_per_token").getInt64();
        m.total_routed_bytes = q.getColumn("total_routed_bytes").getInt64();
        m.active_fraction = q.getColumn("active_fraction").getDouble();
        m.verdict = parse_verdict(q.getColumn("verdict").getText());
        m.verdict_basis = q.getColumn("verdict_basis").getText();
        m.verdict_reason = q.getColumn("verdict_reason").getText();
        m.admitted_at_ms = q.getColumn("admitted_at").getInt64();
        m.profiled_at_ms = q.getColumn("profiled_at").getInt64();
        return m;
    }
};

ControlModelRegistry::ControlModelRegistry() : impl_(std::make_unique<Impl>()) {}

ControlModelRegistry::~ControlModelRegistry() = default;

bool ControlModelRegistry::open(const std::string& data_dir, std::string& out_error) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::error_code ec;
    fs::create_directories(data_dir, ec);
    impl_->path = (fs::path(data_dir) / "control.db").string();
    try {
        impl_->db = std::make_unique<SQLite::Database>(
            impl_->path, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
        impl_->run_migrations();
    } catch (const std::exception& e) {
        out_error = std::string("cannot open ") + impl_->path + ": " + e.what();
        impl_->db.reset();
        return false;
    }
    MM_INFO("ControlModelRegistry: opened {}", impl_->path);
    return true;
}

void ControlModelRegistry::close() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->db.reset();
}

std::uint32_t ControlModelRegistry::schema_version() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) return 0;
    try {
        SQLite::Statement q(*impl_->db, "SELECT MAX(version) FROM schema_migrations");
        if (q.executeStep() && !q.getColumn(0).isNull()) {
            return static_cast<std::uint32_t>(q.getColumn(0).getInt());
        }
    } catch (const std::exception& e) {
        MM_WARN("ControlModelRegistry::schema_version: {}", e.what());
    }
    return 0;
}

// ── queries ───────────────────────────────────────────────────────────────────

std::vector<AdmittedModel> ControlModelRegistry::list() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<AdmittedModel> out;
    if (!impl_->db) return out;
    try {
        SQLite::Statement q(*impl_->db, "SELECT * FROM model ORDER BY name, id");
        while (q.executeStep())
            out.push_back(Impl::read_row(q));
    } catch (const std::exception& e) {
        MM_WARN("ControlModelRegistry::list: {}", e.what());
    }
    return out;
}

std::optional<AdmittedModel> ControlModelRegistry::find_by_id(std::int64_t id) const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) return std::nullopt;
    try {
        SQLite::Statement q(*impl_->db, "SELECT * FROM model WHERE id = ? LIMIT 1");
        q.bind(1, id);
        if (q.executeStep()) return Impl::read_row(q);
    } catch (const std::exception& e) {
        MM_WARN("ControlModelRegistry::find_by_id: {}", e.what());
    }
    return std::nullopt;
}

std::optional<AdmittedModel>
ControlModelRegistry::find_by_arch_hash(const std::string& arch_hash) const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db || arch_hash.empty()) return std::nullopt;
    try {
        SQLite::Statement q(*impl_->db, "SELECT * FROM model WHERE arch_hash = ? LIMIT 1");
        q.bind(1, arch_hash);
        if (q.executeStep()) return Impl::read_row(q);
    } catch (const std::exception& e) {
        MM_WARN("ControlModelRegistry::find_by_arch_hash: {}", e.what());
    }
    return std::nullopt;
}

std::optional<AdmittedModel> ControlModelRegistry::resolve(const std::string& model_ref) const {
    const auto key = ref_key(model_ref);
    if (key.empty()) return std::nullopt;

    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) return std::nullopt;
    try {
        // An exact arch_hash match first: it is the only identity that cannot be
        // coincidental.
        SQLite::Statement byhash(*impl_->db, "SELECT * FROM model WHERE arch_hash = ? LIMIT 1");
        byhash.bind(1, util::trim(model_ref));
        if (byhash.executeStep()) return Impl::read_row(byhash);

        // Then the trailing name component of `name` or `model_dir`. Scanned in
        // C++ rather than matched in SQL because the comparison strips
        // separators and case, and encoding that in a LIKE would be a second,
        // subtly different implementation of ref_key().
        SQLite::Statement q(*impl_->db, "SELECT * FROM model");
        while (q.executeStep()) {
            auto m = Impl::read_row(q);
            if (ref_key(m.name) == key || ref_key(m.model_dir) == key) return m;
        }
    } catch (const std::exception& e) {
        MM_WARN("ControlModelRegistry::resolve: {}", e.what());
    }
    return std::nullopt;
}

std::vector<ConformanceEntry> ControlModelRegistry::conformance(std::int64_t id) const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<ConformanceEntry> out;
    if (!impl_->db) return out;
    try {
        SQLite::Statement q(*impl_->db,
                            "SELECT stage, passed, detail, ran_at FROM conformance "
                            "WHERE model_id = ? ORDER BY stage");
        q.bind(1, id);
        while (q.executeStep()) {
            ConformanceEntry e;
            e.stage = q.getColumn(0).getText();
            e.passed = q.getColumn(1).getInt() != 0;
            e.detail = q.getColumn(2).getText();
            e.ran_at_ms = q.getColumn(3).getInt64();
            out.push_back(std::move(e));
        }
    } catch (const std::exception& e) {
        MM_WARN("ControlModelRegistry::conformance: {}", e.what());
    }
    return out;
}

bool ControlModelRegistry::heat(std::int64_t id, bool bucketed, std::string& out_json) const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) return false;
    // 4096 is the brain grid's default cell count. The cap is applied in SQL so
    // a careless client cannot ask for 60k rows and get them: aggregating after
    // the fact would still have paid for the transfer.
    constexpr int kBucketedLimit = 4096;
    try {
        nlohmann::json j;
        j["model_id"] = id;
        j["bucketed"] = bucketed;
        auto& arr = j["experts"] = nlohmann::json::array();

        SQLite::Statement q(*impl_->db,
                            bucketed
                                ? "SELECT layer, expert, count, decayed, tier FROM expert_heat "
                                  "WHERE model_id = ? ORDER BY decayed DESC LIMIT ?"
                                : "SELECT layer, expert, count, decayed, tier FROM expert_heat "
                                  "WHERE model_id = ? ORDER BY layer, expert");
        q.bind(1, id);
        if (bucketed) q.bind(2, kBucketedLimit);
        while (q.executeStep()) {
            arr.push_back(nlohmann::json{{"layer", q.getColumn(0).getInt()},
                                         {"expert", q.getColumn(1).getInt()},
                                         {"count", q.getColumn(2).getInt64()},
                                         {"decayed", q.getColumn(3).getDouble()},
                                         {"tier", q.getColumn(4).getText()}});
        }
        j["truncated"] = bucketed && arr.size() == static_cast<std::size_t>(kBucketedLimit);
        out_json = j.dump();
        return true;
    } catch (const std::exception& e) {
        MM_WARN("ControlModelRegistry::heat: {}", e.what());
        return false;
    }
}

// ── writes ────────────────────────────────────────────────────────────────────

bool ControlModelRegistry::upsert(AdmittedModel& model, std::string& out_error) {
    if (model.arch_hash.empty()) {
        out_error = "arch_hash is required: it is the model's identity";
        return false;
    }
    if (model.name.empty()) model.name = ref_key(model.model_dir);

    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) {
        out_error = "registry is not open";
        return false;
    }
    try {
        SQLite::Transaction tx(*impl_->db);
        SQLite::Statement q(*impl_->db, R"(
            INSERT INTO model
                (arch_hash, name, source_repo, source_revision, model_dir,
                 schema_version, arch_json, attention_family,
                 n_layers, n_moe_layers, n_experts, top_k,
                 expert_bytes, bytes_per_token, total_routed_bytes,
                 dense_resident_bytes, active_fraction,
                 verdict, verdict_basis, verdict_reason, admitted_at, profiled_at)
            VALUES
                (:arch_hash,:name,:source_repo,:source_revision,:model_dir,
                 1,'{}',:attention_family,
                 :n_layers,:n_moe_layers,:n_experts,:top_k,
                 :expert_bytes,:bytes_per_token,:total_routed_bytes,
                 0,:active_fraction,
                 :verdict,:verdict_basis,:verdict_reason,:admitted_at,:profiled_at)
            ON CONFLICT(arch_hash) DO UPDATE SET
                name               = excluded.name,
                source_repo        = excluded.source_repo,
                source_revision    = excluded.source_revision,
                model_dir          = excluded.model_dir,
                attention_family   = excluded.attention_family,
                n_layers           = excluded.n_layers,
                n_moe_layers       = excluded.n_moe_layers,
                n_experts          = excluded.n_experts,
                top_k              = excluded.top_k,
                expert_bytes       = excluded.expert_bytes,
                bytes_per_token    = excluded.bytes_per_token,
                total_routed_bytes = excluded.total_routed_bytes,
                active_fraction    = excluded.active_fraction,
                verdict            = excluded.verdict,
                verdict_basis      = excluded.verdict_basis,
                verdict_reason     = excluded.verdict_reason,
                profiled_at        = excluded.profiled_at
        )");
        const auto now = util::now_ms();
        q.bind(":arch_hash", model.arch_hash);
        q.bind(":name", model.name);
        q.bind(":source_repo", model.source_repo);
        q.bind(":source_revision", model.source_revision);
        q.bind(":model_dir", model.model_dir);
        q.bind(":attention_family",
               model.attention_family.empty() ? std::string("gqa") : model.attention_family);
        q.bind(":n_layers", static_cast<int>(model.n_layers));
        q.bind(":n_moe_layers", static_cast<int>(model.n_moe_layers));
        q.bind(":n_experts", static_cast<int>(model.n_experts));
        q.bind(":top_k", static_cast<int>(model.top_k));
        q.bind(":expert_bytes", model.expert_bytes);
        q.bind(":bytes_per_token", model.bytes_per_token);
        q.bind(":total_routed_bytes", model.total_routed_bytes);
        q.bind(":active_fraction", model.active_fraction);
        q.bind(":verdict", std::string(to_string(model.verdict)));
        q.bind(":verdict_basis",
               model.verdict_basis.empty() ? std::string("{}") : model.verdict_basis);
        q.bind(":verdict_reason", model.verdict_reason);
        q.bind(":admitted_at", model.admitted_at_ms > 0 ? model.admitted_at_ms : now);
        q.bind(":profiled_at", model.profiled_at_ms);
        q.exec();
        tx.commit();

        SQLite::Statement idq(*impl_->db, "SELECT id FROM model WHERE arch_hash = ?");
        idq.bind(1, model.arch_hash);
        if (idq.executeStep()) model.id = idq.getColumn(0).getInt64();
        return true;
    } catch (const std::exception& e) {
        out_error = e.what();
        MM_ERROR("ControlModelRegistry::upsert: {}", out_error);
        return false;
    }
}

bool ControlModelRegistry::set_verdict(std::int64_t id,
                                       ModelVerdict verdict,
                                       const std::string& reason,
                                       std::string& out_error) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) {
        out_error = "registry is not open";
        return false;
    }
    try {
        SQLite::Statement q(*impl_->db,
                            "UPDATE model SET verdict = ?, verdict_reason = ? WHERE id = ?");
        q.bind(1, std::string(to_string(verdict)));
        // Recorded with a reason so an override is visible rather than
        // mysterious: a verdict that differs from what admission computed is a
        // decision someone made, and the record should say so.
        q.bind(2, reason.empty() ? std::string("operator override") : reason);
        q.bind(3, id);
        if (q.exec() == 0) {
            out_error = "no model with id " + std::to_string(id);
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        out_error = e.what();
        return false;
    }
}

bool ControlModelRegistry::remove(std::int64_t id, std::string& out_error) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) {
        out_error = "registry is not open";
        return false;
    }
    try {
        // ON DELETE CASCADE clears heat, kernel_choice, pilot_profile and
        // conformance; placement_history keeps its rows with a null model_id,
        // because the history of a decision outlives the thing decided about.
        SQLite::Statement q(*impl_->db, "DELETE FROM model WHERE id = ?");
        q.bind(1, id);
        if (q.exec() == 0) {
            out_error = "no model with id " + std::to_string(id);
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        out_error = e.what();
        return false;
    }
}

void ControlModelRegistry::record_placement(const AgentId& agent_id,
                                            const NodeId& node_id,
                                            const SlotId& slot_id,
                                            const std::string& backend,
                                            const std::string& backend_reason,
                                            const ResourceFootprint& footprint) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) return;
    try {
        SQLite::Statement q(*impl_->db, R"(
            INSERT INTO placement_history
                (agent_id, node_id, slot_id, backend, backend_reason,
                 footprint_json, placed_at)
            VALUES (?,?,?,?,?,?,?)
        )");
        q.bind(1, agent_id);
        q.bind(2, node_id);
        q.bind(3, slot_id);
        q.bind(4, backend);
        q.bind(5, backend_reason);
        q.bind(6,
               nlohmann::json{{"vram_mb", footprint.vram_mb},
                              {"ram_mb", footprint.ram_mb},
                              {"disk_mb", footprint.disk_mb}}
                   .dump());
        q.bind(7, util::now_ms());
        q.exec();
    } catch (const std::exception& e) {
        // Never fatal: losing an audit row must not fail a placement that
        // otherwise succeeded.
        MM_WARN("ControlModelRegistry::record_placement: {}", e.what());
    }
}

// ── not implemented, and saying so ────────────────────────────────────────────

std::string ControlModelRegistry::admit(const std::string& source_ref,
                                        AdmissionProgressSink sink,
                                        std::string& out_error) {
    (void)source_ref;
    (void)sink;
    // The conversion pipeline is tools/admission/ and lands with the
    // self-service gate. What exists now is everything DOWNSTREAM of it: the
    // record, the verdict, and the routing that reads them — which is what makes
    // an already-converted model routable via upsert().
    out_error = "in-process admission is not implemented; convert with "
                "tools/admission/convert.py and register the result with POST /v1/models";
    return {};
}

std::string ControlModelRegistry::reprofile(std::int64_t id,
                                            AdmissionProgressSink sink,
                                            std::string& out_error) {
    (void)id;
    (void)sink;
    out_error = "reprofiling is not implemented; it shares the admission pipeline";
    return {};
}

bool ControlModelRegistry::cancel(const std::string& operation_id) {
    (void)operation_id;
    return false;
}

bool ControlModelRegistry::plan_for_host(std::int64_t id,
                                         const HostCapacity& capacity,
                                         std::string& out_json,
                                         std::string& out_error) const {
    (void)capacity;
    // The plan is `soma plan --json` on the TARGET node, because the verdict is a
    // property of (model, quantization, host budget). Control cannot compute it
    // from a row — that is the whole point of the distinction — so this returns
    // the stored admission-host verdict and says which one it is.
    const auto model = find_by_id(id);
    if (!model) {
        out_error = "no model with id " + std::to_string(id);
        return false;
    }
    out_json = nlohmann::json{{"model_id", model->id},
                              {"arch_hash", model->arch_hash},
                              {"verdict", to_string(model->verdict)},
                              {"verdict_basis", model->verdict_basis},
                              {"verdict_reason", model->verdict_reason},
                              {"bytes_per_token", model->bytes_per_token},
                              {"total_routed_bytes", model->total_routed_bytes},
                              {"active_fraction", model->active_fraction},
                              {"scope", "admission-host"},
                              {"note",
                               "the effective verdict is re-derived per node; "
                               "see GET /v1/placements"}}
                   .dump();
    return true;
}

} // namespace mm
