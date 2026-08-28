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

#include "control/route_scope.hpp"

#include "common/logger.hpp"
#include "common/pairing.hpp"
#include "common/process_exec.hpp"
#include "common/util.hpp"

#include <SQLiteCpp/SQLiteCpp.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

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

/// The stages this run will actually go through, in order.
///
/// One list rather than hardcoded step numbers, so `step` and `total_steps`
/// cannot disagree. They did: a container admission advertised 2 total steps and
/// then emitted steps 3, 4 and 5, which a progress bar reads as 250%.
std::vector<std::string> admission_stages(bool container_is_ready, bool needs_fetch) {
    if (container_is_ready) return {"profile", "conformance", "finalize"};
    std::vector<std::string> s;
    if (needs_fetch) s.push_back("fetch");
    s.insert(s.end(), {"convert", "tokenize", "oracle", "reference", "profile", "conformance", "finalize"});
    return s;
}

/// One component of a repo id: starts alphanumeric, then alphanumeric `.`, `-`, `_`.
bool valid_repo_component(const std::string& c) {
    if (c.empty() || !std::isalnum(static_cast<unsigned char>(c.front()))) return false;
    return std::all_of(c.begin(), c.end(), [](unsigned char ch) {
        return std::isalnum(ch) != 0 || ch == '.' || ch == '-' || ch == '_';
    });
}

} // namespace

std::string admission_source_name(const std::string& source, bool needs_fetch) {
    // The trailing component either way — a repo id `Qwen/Qwen3-30B-A3B` and a
    // directory `.../Qwen3-30B-A3B` are the same model and must produce the same
    // container, or admitting one after the other silently makes two.
    if (!needs_fetch) {
        auto trimmed = util::trim(source);
        while (trimmed.size() > 1 && (trimmed.back() == '/' || trimmed.back() == '\\')) {
            trimmed.pop_back();
        }
        return fs::path(trimmed).filename().string();
    }
    auto id = source;
    if (const auto at = id.rfind('@'); at != std::string::npos) id = id.substr(0, at);
    const auto slash = id.find_last_of('/');
    return (slash == std::string::npos) ? id : id.substr(slash + 1);
}

std::string admission_variant(const std::string& source,
                              bool needs_fetch,
                              const AdmissionTools& tools) {
    const auto name = admission_source_name(source, needs_fetch);
    // The QUANTIZATION is part of the directory name, not just the record.
    //
    // Without it, admitting the same weights at q4_g and again at q6_g wrote both
    // to `containers/<name>` — the second overwrote the first, and the first's
    // registry row then pointed at bytes that were no longer the quantization it
    // described. The verdict, the expert_bytes, the KV format: all recorded
    // against a container that had been replaced underneath them, with nothing
    // to detect it.
    return name + "-" + tools.quant + "-" + tools.expert_down + "-g" +
           std::to_string(tools.group);
}

bool valid_repo_id(const std::string& ref, std::string& out_why) {
    auto id = util::trim(ref);
    if (id.empty()) {
        out_why = "empty";
        return false;
    }

    // `@revision` first: it is the only place `/` is allowed to appear freely,
    // since a branch name may contain one.
    if (const auto at = id.rfind('@'); at != std::string::npos) {
        const auto rev = id.substr(at + 1);
        id = id.substr(0, at);
        if (rev.empty() || rev.find("..") != std::string::npos ||
            !std::isalnum(static_cast<unsigned char>(rev.front())) ||
            !std::all_of(rev.begin(), rev.end(), [](unsigned char ch) {
                return std::isalnum(ch) != 0 || ch == '.' || ch == '-' || ch == '_' || ch == '/';
            })) {
            out_why = "bad revision after '@'";
            return false;
        }
    }

    // Rejected explicitly rather than left to the component rule, because these
    // are the strings that would escape sources_dir and the error should say so.
    if (id.find('\\') != std::string::npos || id.find("..") != std::string::npos) {
        out_why = "a repo id becomes a directory name; '..' and '\\' are not allowed";
        return false;
    }

    const auto slash = id.find('/');
    if (slash == std::string::npos) {
        if (!valid_repo_component(id)) {
            out_why = "not a repo id";
            return false;
        }
        return true;
    }
    if (id.find('/', slash + 1) != std::string::npos) {
        out_why = "a repo id has at most one '/'";
        return false;
    }
    if (!valid_repo_component(id.substr(0, slash)) ||
        !valid_repo_component(id.substr(slash + 1))) {
        out_why = "not a repo id";
        return false;
    }
    return true;
}

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

/// One running (or finished) admission.
///
/// Kept after it finishes, deliberately: conversion runs for hours, an SSE
/// connection will not survive that reliably, and a client that reconnects must
/// be able to find out how its operation ended rather than guess.
struct AdmissionOperation {
    AdmissionProgress progress;
    std::vector<AdmissionProgressSink> sinks;
    std::atomic<bool> cancel{false};
    std::thread worker;

    /// The container directory this operation will write, and therefore what it
    /// collides with. Empty for an operation that writes no container.
    std::string variant;
    /// Still holding a concurrency slot? Cleared when the worker finishes, so a
    /// completed operation stops counting against the cap without being erased
    /// from the history the API serves.
    bool occupying_slot = false;
};

struct ControlModelRegistry::Impl {
    mutable std::mutex mu;
    std::unique_ptr<SQLite::Database> db;
    std::string path;

    AdmissionTools tools;

    mutable std::mutex ops_mu;
    std::map<std::string, std::shared_ptr<AdmissionOperation>> ops;
    std::vector<std::string> op_order; ///< newest last

    /// How many admissions may RUN at once. Everything past this waits in
    /// `admission_gate` and reports `queued` until a slot frees.
    ///
    /// One by default, and that is a deliberate choice rather than a
    /// placeholder. A conversion spawns Python and moves tens to hundreds of
    /// gigabytes; two at once on one box do not go twice as fast, they contend
    /// for the same disk and the same RAM and can take each other out. Admission
    /// is a once-per-model operation, so serializing costs latency nobody is
    /// waiting on interactively — and the operation already had a `queued` state
    /// it never actually used.
    ///
    /// Raiseable for a host that really can run two, which is why it is a field
    /// rather than a constant.
    std::size_t max_concurrent_admissions = 1;
    std::condition_variable admission_gate;

    /// Operations counted against `max_concurrent_admissions`. Call with ops_mu.
    std::size_t running_admissions_locked() const {
        std::size_t n = 0;
        for (const auto& [id, op] : ops)
            if (op && op->occupying_slot) ++n;
        return n;
    }

    /// The live operation writing `variant`, or nullptr. Call with ops_mu.
    ///
    /// Keyed on the container directory rather than the source string, because
    /// that is what actually collides: two different-looking refs for one model
    /// resolve to one directory, and two `convert.py` processes interleaving
    /// writes into it produce a corrupt container with a registry row calling it
    /// good.
    std::shared_ptr<AdmissionOperation> live_for_variant_locked(const std::string& variant) const {
        if (variant.empty()) return nullptr;
        for (const auto& [id, op] : ops) {
            if (!op || op->variant != variant) continue;
            if (op->progress.done) continue;
            return op;
        }
        return nullptr;
    }

    /// Publish a progress update to every attached sink.
    ///
    /// Sinks are copied out under the lock and called OUTSIDE it: a sink writes
    /// to a socket, and a slow client holding ops_mu would stall the conversion
    /// it is watching.
    void publish(const std::shared_ptr<AdmissionOperation>& op, const AdmissionProgress& p);

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

        if (!has_version(2)) {
            SQLite::Transaction tx(*db);
            // The conformance table gains a THIRD state and two stage names.
            //
            // `passed` as a boolean cannot say "did not run", and the difference
            // matters more than either value: a stage that needs a transformers
            // oracle this host does not have is not a failure, and recording it
            // as one would reject every model. Recording it as a pass would be
            // worse — the verdict would look validated when it was only computed.
            //
            // SQLite cannot alter a CHECK constraint, so the table is rebuilt.
            // Existing rows carry forward with status derived from `passed`,
            // which is the honest reading of what they meant when written.
            db->exec(R"(
                CREATE TABLE conformance_v2 (
                    model_id INTEGER NOT NULL REFERENCES model(id) ON DELETE CASCADE,
                    stage    TEXT    NOT NULL
                             CHECK (stage IN ('fp32_tiny_tf',
                                              'quant_tiny_greedy',
                                              'real_logit_kl',
                                              'accuracy_floor',
                                              'tokenizer_roundtrip',
                                              'quant_codec')),
                    status   TEXT    NOT NULL
                             CHECK (status IN ('passed', 'failed', 'skipped')),
                    passed   INTEGER NOT NULL,
                    detail   TEXT,
                    ran_at   INTEGER NOT NULL,
                    PRIMARY KEY (model_id, stage)
                ))");
            db->exec("INSERT INTO conformance_v2(model_id, stage, status, passed, detail, ran_at) "
                     "SELECT model_id, stage, CASE WHEN passed THEN 'passed' ELSE 'failed' END, "
                     "       passed, detail, ran_at FROM conformance");
            db->exec("DROP TABLE conformance");
            db->exec("ALTER TABLE conformance_v2 RENAME TO conformance");

            db->exec("INSERT OR IGNORE INTO schema_migrations(version) VALUES (2)");
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

void ControlModelRegistry::Impl::publish(const std::shared_ptr<AdmissionOperation>& op,
                                         const AdmissionProgress& p) {
    // Sinks are copied out under the lock and called OUTSIDE it: a sink writes to
    // a socket, and a slow client holding ops_mu would stall the conversion it is
    // watching.
    std::vector<AdmissionProgressSink> sinks;
    {
        std::lock_guard<std::mutex> lk(ops_mu);
        op->progress = p;
        sinks = op->sinks;
    }
    for (const auto& s : sinks) {
        try {
            if (s) s(p);
        } catch (const std::exception& e) {
            MM_WARN("admission sink threw: {}", e.what());
        }
    }
}

ControlModelRegistry::ControlModelRegistry() : impl_(std::make_unique<Impl>()) {}

ControlModelRegistry::~ControlModelRegistry() {
    // Admission operations outlive the HTTP request that started them, but they
    // must not outlive their registry. Every worker captures `this` and may use
    // the gate, progress map, or database until its final instruction.
    std::vector<std::shared_ptr<AdmissionOperation>> operations;
    {
        std::lock_guard<std::mutex> lk(impl_->ops_mu);
        operations.reserve(impl_->ops.size());
        for (const auto& [id, op] : impl_->ops) {
            if (!op) continue;
            op->cancel.store(true);
            operations.push_back(op);
        }
    }

    // Release workers that have not claimed an admission slot yet. Running
    // workers observe the same cancel flag in run_streamed_command().
    impl_->admission_gate.notify_all();

    // Join outside ops_mu: each worker takes that mutex while publishing its
    // terminal state and releasing its slot.
    for (const auto& op : operations) {
        if (op->worker.joinable()) op->worker.join();
    }
}

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
        // Re-admitting the same weights at a second quantization gives TWO rows
        // matching one name, so "the first row the scan reaches" is not an
        // answer — it is whichever the b-tree happened to yield, and it would
        // change under an unrelated insert. Ranked instead:
        //
        //   1. a verdict that selects Soma, because a row that routes to the
        //      fallback is not what an agent asking for this model wants when a
        //      streamable variant of the same weights exists;
        //   2. more recently profiled, so a fresh admission supersedes a stale
        //      one rather than losing to it on row order.
        //
        // An operator who wants a SPECIFIC variant passes its arch_hash, which
        // the exact match above already handles and which is the only identity
        // that cannot be ambiguous.
        std::optional<AdmittedModel> best;
        const auto better = [](const AdmittedModel& a, const AdmittedModel& b) {
            const bool as = verdict_selects_soma(a.verdict);
            const bool bs = verdict_selects_soma(b.verdict);
            if (as != bs) return as;
            return a.profiled_at_ms > b.profiled_at_ms;
        };

        SQLite::Statement q(*impl_->db, "SELECT * FROM model");
        while (q.executeStep()) {
            auto m = Impl::read_row(q);
            if (ref_key(m.name) == key || ref_key(m.model_dir) == key) {
                if (!best || better(m, *best)) best = std::move(m);
            }
        }
        if (best) return best;
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
                            "SELECT stage, status, passed, detail, ran_at FROM conformance "
                            "WHERE model_id = ? ORDER BY stage");
        q.bind(1, id);
        while (q.executeStep()) {
            ConformanceEntry e;
            e.stage = q.getColumn(0).getText();
            e.status = q.getColumn(1).getText();
            e.passed = q.getColumn(2).getInt() != 0;
            e.detail = q.getColumn(3).getText();
            e.ran_at_ms = q.getColumn(4).getInt64();
            out.push_back(std::move(e));
        }
    } catch (const std::exception& e) {
        MM_WARN("ControlModelRegistry::conformance: {}", e.what());
    }
    return out;
}

bool ControlModelRegistry::record_conformance(std::int64_t id,
                                              const std::vector<ConformanceEntry>& stages,
                                              std::string& out_error) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) {
        out_error = "registry is not open";
        return false;
    }
    try {
        SQLite::Transaction tx(*impl_->db);
        for (const auto& e : stages) {
            // REPLACE, not INSERT: a reprofile re-runs the ladder against the
            // same weights, and the row it produces supersedes the old one rather
            // than accumulating a history nobody reads.
            SQLite::Statement q(*impl_->db,
                                "INSERT OR REPLACE INTO conformance"
                                "(model_id, stage, status, passed, detail, ran_at) "
                                "VALUES (?, ?, ?, ?, ?, ?)");
            q.bind(1, id);
            q.bind(2, e.stage);
            q.bind(3, e.status);
            q.bind(4, e.status == "passed" ? 1 : 0);
            q.bind(5, e.detail);
            q.bind(6, e.ran_at_ms != 0 ? e.ran_at_ms : util::now_ms());
            q.exec();
        }
        tx.commit();
    } catch (const std::exception& ex) {
        out_error = ex.what();
        MM_WARN("ControlModelRegistry::record_conformance: {}", out_error);
        return false;
    }
    return true;
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

void ControlModelRegistry::mark_placement_released(const AgentId& agent_id) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) return;
    try {
        // The most recent OPEN row only. An agent placed, released, and placed
        // again has two rows; stamping by agent alone would close the older one
        // a second time and lose the first placement's duration.
        SQLite::Statement q(*impl_->db, R"(
            UPDATE placement_history SET released_at = ?
             WHERE id = (SELECT id FROM placement_history
                          WHERE agent_id = ? AND released_at IS NULL
                          ORDER BY placed_at DESC LIMIT 1)
        )");
        q.bind(1, util::now_ms());
        q.bind(2, agent_id);
        q.exec();
    } catch (const std::exception& e) {
        // Same rule as the insert: an audit row must never fail the operation
        // it is describing.
        MM_WARN("ControlModelRegistry::mark_placement_released: {}", e.what());
    }
}

std::vector<PlacementHistoryEntry> ControlModelRegistry::placement_history(const AgentId& agent_id,
                                                                           int limit) const {
    std::vector<PlacementHistoryEntry> out;
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) return out;
    if (limit <= 0) return out;
    try {
        SQLite::Statement q(*impl_->db, R"(
            SELECT node_id, slot_id, backend, backend_reason, footprint_json,
                   placed_at, released_at
              FROM placement_history
             WHERE agent_id = ?
             ORDER BY placed_at DESC
             LIMIT ?
        )");
        q.bind(1, agent_id);
        q.bind(2, limit);
        while (q.executeStep()) {
            PlacementHistoryEntry e;
            e.node_id = q.getColumn(0).getString();
            e.slot_id = q.getColumn(1).getString();
            e.backend = q.getColumn(2).getString();
            e.backend_reason = q.getColumn(3).getString();
            // Stored as JSON rather than three columns because the footprint's
            // shape is owned by common/footprint.hpp and has already gained an
            // axis once. A malformed blob yields zeroes rather than throwing:
            // losing one row's numbers must not blank the whole history.
            const std::string fp = q.getColumn(4).getString();
            if (!fp.empty()) {
                const auto j = nlohmann::json::parse(fp, nullptr, /*allow_exceptions=*/false);
                if (j.is_object()) {
                    e.vram_mb = j.value("vram_mb", std::int64_t{0});
                    e.ram_mb = j.value("ram_mb", std::int64_t{0});
                    e.disk_mb = j.value("disk_mb", std::int64_t{0});
                }
            }
            e.placed_at_ms = q.getColumn(5).getInt64();
            e.released_at_ms = q.getColumn(6).isNull() ? 0 : q.getColumn(6).getInt64();
            out.push_back(std::move(e));
        }
    } catch (const std::exception& e) {
        MM_WARN("ControlModelRegistry::placement_history: {}", e.what());
    }
    return out;
}

// ── api tokens ────────────────────────────────────────────────────────────────

std::string ControlModelRegistry::create_api_token(const std::string& label,
                                                   std::uint8_t scopes,
                                                   std::string& out_error) {
    if (scopes == 0) {
        out_error = "a token with no scopes can do nothing; refusing to mint one";
        return {};
    }
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) {
        out_error = "registry is not open";
        return {};
    }
    // 32 bytes of CSPRNG. Returned once and never stored — only the hash goes to
    // the table, so this is the only moment the secret exists anywhere we
    // control, and losing it means minting another rather than recovering it.
    const auto secret = pairing::generate_nonce(32);
    try {
        SQLite::Statement q(*impl_->db,
                            "INSERT INTO api_token(token_sha256, label, scopes, created_at) "
                            "VALUES (?,?,?,?)");
        q.bind(1, pairing::sha256_hex(secret));
        q.bind(2, label.empty() ? std::string("unnamed") : label);
        q.bind(3, format_scopes(scopes));
        q.bind(4, util::now_ms());
        q.exec();
    } catch (const std::exception& e) {
        out_error = e.what();
        return {};
    }
    MM_INFO("ControlModelRegistry: minted api token '{}' with scopes [{}]",
            label, format_scopes(scopes));
    return secret;
}

bool ControlModelRegistry::find_api_token(const std::string& token_sha256, ApiToken& out) const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db || token_sha256.empty()) return false;
    try {
        SQLite::Statement q(*impl_->db,
                            "SELECT id, token_sha256, label, scopes, created_at, last_used_at "
                            "FROM api_token WHERE token_sha256 = ? AND revoked_at IS NULL "
                            "LIMIT 1");
        q.bind(1, token_sha256);
        if (!q.executeStep()) return false;
        out.id = q.getColumn(0).getInt64();
        out.token_sha256 = q.getColumn(1).getText();
        out.label = q.getColumn(2).getText();
        (void)parse_scopes(q.getColumn(3).getText(), out.scopes);
        out.created_at_ms = q.getColumn(4).getInt64();
        out.last_used_at_ms = q.getColumn(5).getInt64();
        out.revoked = false;

        // Best-effort: a failed last_used update must not fail the request that
        // was otherwise authorized.
        try {
            SQLite::Statement t(*impl_->db, "UPDATE api_token SET last_used_at = ? WHERE id = ?");
            t.bind(1, util::now_ms());
            t.bind(2, out.id);
            t.exec();
        } catch (const std::exception&) {
        }
        return true;
    } catch (const std::exception& e) {
        MM_WARN("ControlModelRegistry::find_api_token: {}", e.what());
        return false;
    }
}

std::vector<ApiToken> ControlModelRegistry::list_api_tokens() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<ApiToken> out;
    if (!impl_->db) return out;
    try {
        SQLite::Statement q(*impl_->db,
                            "SELECT id, token_sha256, label, scopes, created_at, last_used_at, "
                            "revoked_at FROM api_token ORDER BY id");
        while (q.executeStep()) {
            ApiToken t;
            t.id = q.getColumn(0).getInt64();
            // The hash is listed, not the token: there is nothing to leak here,
            // and it is what an operator needs to correlate a row with a log line.
            t.token_sha256 = q.getColumn(1).getText();
            t.label = q.getColumn(2).getText();
            (void)parse_scopes(q.getColumn(3).getText(), t.scopes);
            t.created_at_ms = q.getColumn(4).getInt64();
            t.last_used_at_ms = q.getColumn(5).getInt64();
            t.revoked = !q.getColumn(6).isNull();
            out.push_back(std::move(t));
        }
    } catch (const std::exception& e) {
        MM_WARN("ControlModelRegistry::list_api_tokens: {}", e.what());
    }
    return out;
}

bool ControlModelRegistry::revoke_api_token(std::int64_t id, std::string& out_error) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) {
        out_error = "registry is not open";
        return false;
    }
    try {
        // Revoked, not deleted: the row is the audit trail, and a deleted token
        // cannot answer "what was this credential allowed to do".
        SQLite::Statement q(*impl_->db,
                            "UPDATE api_token SET revoked_at = ? WHERE id = ? AND "
                            "revoked_at IS NULL");
        q.bind(1, util::now_ms());
        q.bind(2, id);
        if (q.exec() == 0) {
            out_error = "no active token with id " + std::to_string(id);
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        out_error = e.what();
        return false;
    }
}

bool ControlModelRegistry::has_api_tokens() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->db) return false;
    try {
        SQLite::Statement q(*impl_->db,
                            "SELECT 1 FROM api_token WHERE revoked_at IS NULL LIMIT 1");
        return q.executeStep();
    } catch (const std::exception&) {
        return false;
    }
}

// ── admission ─────────────────────────────────────────────────────────────────

void ControlModelRegistry::set_tools(const AdmissionTools& tools) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->tools = tools;
}

void ControlModelRegistry::set_max_concurrent_admissions(std::size_t n) {
    {
        std::lock_guard<std::mutex> lk(impl_->ops_mu);
        // Zero would park every admission on a gate nothing can open — a
        // configuration that reads as "pause admissions" and behaves as "hang
        // them forever".
        impl_->max_concurrent_admissions = std::max<std::size_t>(1, n);
    }
    // Raising the cap must release whoever is already waiting, or the new value
    // takes effect only after the next admission happens to finish.
    impl_->admission_gate.notify_all();
}

std::size_t ControlModelRegistry::max_concurrent_admissions() const {
    std::lock_guard<std::mutex> lk(impl_->ops_mu);
    return impl_->max_concurrent_admissions;
}

AdmissionTools ControlModelRegistry::tools() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->tools;
}

std::vector<AdmissionProgress> ControlModelRegistry::operations() const {
    std::lock_guard<std::mutex> lk(impl_->ops_mu);
    std::vector<AdmissionProgress> out;
    out.reserve(impl_->op_order.size());
    for (auto it = impl_->op_order.rbegin(); it != impl_->op_order.rend(); ++it) {
        if (auto found = impl_->ops.find(*it); found != impl_->ops.end()) {
            out.push_back(found->second->progress);
        }
    }
    return out;
}

std::optional<AdmissionProgress>
ControlModelRegistry::operation(const std::string& id) const {
    std::lock_guard<std::mutex> lk(impl_->ops_mu);
    if (auto it = impl_->ops.find(id); it != impl_->ops.end()) return it->second->progress;
    return std::nullopt;
}

bool ControlModelRegistry::attach_sink(const std::string& operation_id,
                                       AdmissionProgressSink sink,
                                       AdmissionProgress& out_current) {
    std::lock_guard<std::mutex> lk(impl_->ops_mu);
    auto it = impl_->ops.find(operation_id);
    if (it == impl_->ops.end()) return false;
    out_current = it->second->progress;
    // A finished operation gets no sink: there will never be another update, and
    // registering one would leave the caller waiting on a stream that is over.
    // It still gets the terminal snapshot above, which is the answer it wanted.
    if (!out_current.done) it->second->sinks.push_back(std::move(sink));
    return true;
}

bool ControlModelRegistry::cancel(const std::string& operation_id) {
    {
        std::lock_guard<std::mutex> lk(impl_->ops_mu);
        auto it = impl_->ops.find(operation_id);
        if (it == impl_->ops.end()) return false;
        if (it->second->progress.done) return false; // too late is not a failure to report
        it->second->cancel.store(true);
    }
    // Wakes an operation still WAITING for a concurrency slot. Without this a
    // queued admission would sit on the gate until some other operation
    // finished, long after it was told to stop.
    impl_->admission_gate.notify_all();
    return true;
}

std::string ControlModelRegistry::admit(const std::string& source_ref,
                                        AdmissionProgressSink sink,
                                        std::string& out_error) {
    return admit(source_ref, QuantOverride{}, std::move(sink), out_error);
}

std::string ControlModelRegistry::admit(const std::string& source_ref,
                                        const QuantOverride& quant,
                                        AdmissionProgressSink sink,
                                        std::string& out_error) {
    if (!impl_->db) {
        out_error = "registry is not open";
        return {};
    }
    const auto source = util::trim(source_ref);
    std::error_code ec;
    if (source.empty()) {
        out_error = "no source given";
        return {};
    }
    // A local directory is used as-is; anything else has to be a repo id worth
    // fetching. Both are checked BEFORE the operation exists, so a typo is a 400
    // rather than an operation that appears to start and fails a second later.
    if (fs::exists(source, ec)) {
        return start_operation(source, /*container_is_ready=*/false, quant, std::move(sink));
    }
    std::string why;
    if (!valid_repo_id(source, why)) {
        // Both halves, because "not found" alone sends the operator looking for a
        // typo in a path when they meant a repo id, and vice versa.
        out_error = "source not found: no directory at '" + source +
                    "', and it is not a usable repo id (" + why + ")";
        return {};
    }
    return start_operation(source, /*container_is_ready=*/false, quant, std::move(sink));
}

std::string ControlModelRegistry::admit_container(const std::string& container_dir,
                                                 AdmissionProgressSink sink,
                                                 std::string& out_error) {
    if (!impl_->db) {
        out_error = "registry is not open";
        return {};
    }
    const auto dir = util::trim(container_dir);
    std::error_code ec;
    if (dir.empty() || !fs::exists(dir, ec)) {
        out_error = "container directory not found: " + dir;
        return {};
    }
    return start_operation(dir, /*container_is_ready=*/true, QuantOverride{}, std::move(sink));
}

std::string ControlModelRegistry::reprofile(std::int64_t id,
                                            AdmissionProgressSink sink,
                                            std::string& out_error) {
    const auto model = find_by_id(id);
    if (!model) {
        out_error = "no model with id " + std::to_string(id);
        return {};
    }
    if (model->model_dir.empty()) {
        out_error = "model has no container directory to re-plan";
        return {};
    }
    // Conversion is skipped: the container is what conversion produces, and
    // re-running it would rewrite gigabytes to reach the same bytes. What gets
    // re-derived is the verdict, which is the part that goes stale — a changed
    // host budget changes it without any weight changing.
    // No override: re-profiling must not change the bytes. That is the half of
    // the gate that is easy to get wrong in the other direction — a reprofile
    // that requantized would produce a new arch_hash and orphan every KV
    // checkpoint written against the old one, for a request that asked only for
    // a fresh verdict.
    return start_operation(model->model_dir, /*container_is_ready=*/true, QuantOverride{},
                           std::move(sink));
}

std::string ControlModelRegistry::start_operation(const std::string& source,
                                                 bool container_is_ready,
                                                 const QuantOverride& quant,
                                                 AdmissionProgressSink sink) {
    // Applied to the COPY the operation runs with, not to the registry's tools:
    // two admissions of the same model at different quantizations can be in
    // flight at once, and a shared field would let the second rewrite the
    // first's conversion arguments mid-run.
    auto tools_copy = tools();
    if (!quant.quant.empty()) tools_copy.quant = quant.quant;
    if (!quant.expert_down.empty()) tools_copy.expert_down = quant.expert_down;
    if (quant.group > 0) tools_copy.group = quant.group;
    const auto id = util::generate_uuid();

    std::error_code ec;
    const bool needs_fetch = !container_is_ready && !fs::exists(source, ec);
    const auto stages = admission_stages(container_is_ready, needs_fetch);

    // What this operation will WRITE, and therefore what it collides with.
    // Container admissions and reprofiles target an existing directory and are
    // not conversions, so they carry no variant and cannot collide this way.
    const std::string variant =
        container_is_ready ? std::string{} : admission_variant(source, needs_fetch, tools_copy);

    auto op = std::make_shared<AdmissionOperation>();
    op->progress.operation_id = id;
    op->progress.stage = stages.front();
    op->progress.detail = "queued";
    op->progress.total_steps = static_cast<int>(stages.size());
    op->progress.source_ref = source;
    op->progress.started_at_ms = util::now_ms();
    op->variant = variant;
    // The sink is attached BEFORE the thread starts, so the first frames cannot
    // be produced before anyone is listening for them.
    if (sink) op->sinks.push_back(std::move(sink));
    // ── Duplicate in flight ───────────────────────────────────────────────────
    //
    // JOIN it rather than refuse. The caller asked for this model to be admitted
    // and it is being admitted; handing back the running operation's id — with
    // their sink attached — gives them the progress stream they wanted, and
    // gives the second caller of a double-clicked button the right answer
    // instead of an error.
    //
    // Refusing would also be defensible. Starting a second is not, and that is
    // what happened: two convert.py processes interleaved writes into one
    // directory, and the loser's registry row described bytes the winner had
    // replaced.
    {
        std::shared_ptr<AdmissionOperation> live;
        AdmissionProgress replay;
        std::vector<AdmissionProgressSink> joined;
        {
            std::lock_guard<std::mutex> lk(ops_mu_ref());
            live = impl_->live_for_variant_locked(variant);
            if (live) {
                for (auto& s : op->sinks)
                    if (s) {
                        live->sinks.push_back(s);
                        joined.push_back(std::move(s));
                    }
                replay = live->progress;
            } else {
                impl_->ops[id] = op;
                impl_->op_order.push_back(id);
            }
        }
        if (live) {
            const auto joined_id = replay.operation_id;
            // Replayed OUTSIDE the lock, following publish()'s rule: a sink
            // writes to a socket, and a slow client holding ops_mu would stall
            // the conversion it just joined. The replay exists because a joiner
            // arriving mid-convert would otherwise see nothing until the next
            // stage boundary, which can be twenty minutes away.
            for (const auto& s : joined)
                if (s) s(replay);
            MM_INFO("admission: joined in-flight operation {} for '{}' (variant '{}')",
                    joined_id, source, variant);
            return joined_id;
        }
    }

    op->worker = std::thread([this, op, source, tools_copy, container_is_ready] {
        // ── Concurrency gate ──────────────────────────────────────────────────
        //
        // Waited on HERE rather than in start_operation, so the caller's request
        // returns immediately with an operation id and the wait shows up as
        // `queued` in the progress stream. Blocking the HTTP thread until a slot
        // freed would turn a queued admission into a hung request.
        {
            std::unique_lock<std::mutex> lk(ops_mu_ref());
            impl_->admission_gate.wait(lk, [this, &op] {
                return op->cancel.load() ||
                       impl_->running_admissions_locked() < impl_->max_concurrent_admissions;
            });
            // Claimed under the same lock the predicate was evaluated under, so
            // two waiters cannot both see a free slot and both take it.
            if (!op->cancel.load()) op->occupying_slot = true;
        }

        if (op->cancel.load()) {
            // Cancelled while still QUEUED, so run_admission never ran and never
            // published a terminal frame. Exactly one `done` is delivered on
            // every path — the API's stated guarantee — and a watcher that got
            // none would wait forever for an operation that had already stopped.
            auto progress = op->progress;
            progress.detail = "canceled before it started";
            progress.last_error = "canceled";
            progress.canceled = true;
            progress.done = true;
            progress.cancelable = false;
            progress.finished_at_ms = util::now_ms();
            impl_->publish(op, progress);
        } else {
            run_admission(op, source, tools_copy, container_is_ready);
        }

        // Released on EVERY path — cancel included — or one stuck admission
        // would wedge every queued one behind it forever.
        {
            std::lock_guard<std::mutex> lk(ops_mu_ref());
            op->occupying_slot = false;
        }
        impl_->admission_gate.notify_all();
    });
    // Kept joinable by the registry. The operation still outlives the request —
    // nothing joins here — but registry destruction can now cancel and join it
    // before releasing the state captured by the worker.
    return id;
}

std::mutex& ControlModelRegistry::ops_mu_ref() const { return impl_->ops_mu; }

void ControlModelRegistry::run_admission(std::shared_ptr<AdmissionOperation> op,
                                         const std::string& source,
                                         const AdmissionTools& tools,
                                         bool container_is_ready) {
    auto progress = op->progress;

    std::error_code fs_ec;
    const bool needs_fetch = !container_is_ready && !fs::exists(source, fs_ec);
    const auto stages = admission_stages(container_is_ready, needs_fetch);

    // `step` is derived from the stage name, never written by hand. The two used
    // to be independent and drifted immediately.
    const auto emit = [&](const std::string& stage, const std::string& detail, double fraction) {
        const auto at = std::find(stages.begin(), stages.end(), stage);
        progress.stage = stage;
        progress.step = static_cast<int>(std::distance(stages.begin(), at)) + 1;
        progress.detail = detail;
        progress.fraction = fraction;
        impl_->publish(op, progress);
    };
    const auto fail = [&](const std::string& why) {
        progress.last_error = why;
        progress.done = true;
        progress.cancelable = false;
        progress.finished_at_ms = util::now_ms();
        impl_->publish(op, progress);
        MM_ERROR("admission {}: {}", progress.operation_id, why);
    };
    const auto canceled = [&] { return op->cancel.load(); };

    const fs::path tools_dir(tools.tools_dir);

    // Derived by the shared helper, which start_operation() also uses as the
    // in-flight collision key. Deriving it twice is how a guard ends up watching
    // a different directory from the one being written.
    const auto variant = admission_variant(source, needs_fetch, tools);
    // The fetch destination shares the same name rule, so the two cannot
    // disagree about which model this is.
    const auto name = admission_source_name(source, needs_fetch);
    const auto container = (fs::path(tools.containers_dir) / variant).string();

    // What conversion actually reads. For a local source that is the source; for
    // a repo id it is whatever fetch.py resolves to, which is NOT assumed to be
    // the directory we asked for.
    std::string local_source = source;

    // ── 1. fetch ─────────────────────────────────────────────────────────────
    //
    // The only stage that touches the network. Everything after it works on a
    // local directory, which is why this is a separate stage rather than a mode
    // of conversion.
    if (needs_fetch) {
        const auto dest = (fs::path(tools.sources_dir) / name).string();
        emit("fetch", "fetching " + source + " -> " + dest, 0.01);
        std::error_code ec;
        fs::create_directories(tools.sources_dir, ec);

        std::vector<std::string> argv{tools.python, (tools_dir / "fetch.py").string(), source,
                                      "--out", dest};
        if (tools.allow_pickle) argv.push_back("--allow-pickle");

        std::string err, resolved;
        std::int64_t seen = 0, expect = 0;
        const int frc = run_streamed_command(
            argv, fs::current_path(),
            [&](const std::string& line, bool) {
                const auto text = util::trim(line);
                if (text.empty()) return;
                // `manifest <files> <bytes>` and `progress <done> <total>` are the
                // machine-readable lines; everything else is detail for the
                // operator and is forwarded unchanged.
                if (text.rfind("resolved ", 0) == 0) {
                    resolved = util::trim(text.substr(9));
                    return;
                }
                if (text.rfind("manifest ", 0) == 0) {
                    std::istringstream in(text.substr(9));
                    long long files = 0, bytes = 0;
                    if (in >> files >> bytes) expect = bytes;
                    return;
                }
                if (text.rfind("progress ", 0) == 0) {
                    std::istringstream in(text.substr(9));
                    long long done = 0, total = 0;
                    if (in >> done >> total) {
                        seen = done;
                        if (total > 0) expect = total;
                        progress.bytes_done = seen;
                        progress.bytes_total = expect;
                        // The fetch owns 0.01..0.35. It is frequently the longest
                        // stage in wall-clock and the only one whose remaining
                        // time a client can actually estimate.
                        const double f = expect > 0 ? static_cast<double>(seen) /
                                                          static_cast<double>(expect)
                                                    : 0.0;
                        emit("fetch", util::bytes_label(seen) + " / " + util::bytes_label(expect),
                             0.01 + 0.34 * f);
                    }
                    return;
                }
                emit("fetch", text, progress.fraction);
            },
            canceled, &err);
        if (canceled()) {
            progress.canceled = true;
            fail("canceled during fetch");
            return;
        }
        if (frc != 0) {
            fail("fetch.py failed (exit " + std::to_string(frc) + ")" +
                 (err.empty() ? "" : ": " + err));
            return;
        }
        if (resolved.empty() || !fs::exists(resolved, ec)) {
            // fetch.py exiting 0 without a directory would send conversion at a
            // path that is not there, and the error would name convert.py.
            fail("fetch reported success but produced no directory");
            return;
        }
        local_source = resolved;
        progress.bytes_done = progress.bytes_total = 0;
        emit("fetch", "fetched " + util::bytes_label(seen), 0.35);
    }

    // ── 2. convert ───────────────────────────────────────────────────────────
    //
    // Preceded by an architecture check, which is the difference between failing
    // in 200 ms and failing in six hours. `soma plan` on the SOURCE reads
    // config.json and allocates nothing; if there is no backend for the attention
    // family, no host can run this model and the container would be gigabytes
    // nothing can read.
    //
    // Only `arch_supported` short-circuits. A verdict of reject on ECONOMICS does
    // not: the verdict is a property of (model, quantization, host), so a node
    // with more RAM can reach a different one from the same container — throwing
    // away the conversion because THIS host said no would be a category error.
    std::string plan_json;
    bool arch_unsupported = false;
    if (!container_is_ready) {
        std::string probe, err;
        const int arc = run_streamed_command(
            {tools.soma_path, "plan", "--model-dir", local_source, "--json"}, fs::current_path(),
            [&](const std::string& line, bool is_stderr) {
                if (!is_stderr) probe += line;
            },
            canceled, &err);
        if (canceled()) {
            progress.canceled = true;
            fail("canceled during the architecture check");
            return;
        }
        if (arc != 0) {
            // `plan` refusing the source is the reachable form of "no backend":
            // adapt_hf_config's table IS the registry of architectures this
            // engine understands, and an unknown `model_type` stops there. A
            // container built from a config Soma cannot parse is gigabytes
            // nothing can read, so this fails the admission rather than
            // discovering it after conversion.
            fail("this architecture is not supported: " +
                 (err.empty() ? "soma plan exited " + std::to_string(arc) : util::trim(err)));
            return;
        }
        try {
            const auto j = nlohmann::json::parse(probe);
            // Absent means an older `soma` that does not report it. Treated as
            // supported, so a version skew degrades to the previous behaviour —
            // convert and find out — rather than refusing every model.
            if (!j.value("arch_supported", true)) {
                // Parsed, but there is no forward for its attention family. Not
                // an error: it is a REJECT record saying "route this to the
                // fallback", which is a successful admission. Conversion is
                // skipped because no host will ever read the container.
                arch_unsupported = true;
                plan_json = probe;
                emit("convert",
                     "no backend for attention family '" +
                         j.value("attention_family", std::string{"unknown"}) +
                         "'; skipping conversion",
                     0.80);
            }
        } catch (const std::exception&) {
            // Exited 0 with unreadable output. Let convert run and produce the
            // error; its message names the file.
        }
    }

    if (!container_is_ready && !arch_unsupported) {
        const double convert_lo = needs_fetch ? 0.35 : 0.05;
        emit("convert", "converting " + name + " -> " + container, convert_lo);
        std::error_code ec;
        fs::create_directories(tools.containers_dir, ec);

        std::string err;
        const int rc = run_streamed_command(
            {tools.python, (tools_dir / "convert.py").string(), local_source, "--out", container,
             "--quant", tools.quant, "--expert-down", tools.expert_down, "--group",
             std::to_string(tools.group)},
            fs::current_path(),
            [&](const std::string& line, bool) {
                // convert.py prints "    layer 12/48  3.40 GB" per layer with
                // flush=True. Parsed rather than counted, so the fraction tracks
                // the model's real shape instead of an assumed one.
                const auto pos = line.find("layer ");
                if (pos != std::string::npos) {
                    const auto slash = line.find('/', pos);
                    if (slash != std::string::npos) {
                        try {
                            const int done = std::stoi(line.substr(pos + 6, slash - pos - 6));
                            const int total = std::stoi(line.substr(slash + 1));
                            if (total > 0) {
                                // Conversion is the long pole; it owns 5%..70%.
                                emit("convert", util::trim(line),
                                     0.05 + 0.65 * static_cast<double>(done) / total);
                                return;
                            }
                        } catch (...) {
                            // Not the line we thought; fall through and just log it.
                        }
                    }
                }
                if (!util::trim(line).empty()) emit("convert", util::trim(line), progress.fraction);
            },
            canceled, &err);
        if (canceled()) {
            progress.canceled = true;
            fail("canceled during conversion");
            return;
        }
        if (rc != 0) {
            fail("convert.py failed (exit " + std::to_string(rc) + ")" +
                 (err.empty() ? "" : ": " + err));
            return;
        }

        // ── 2. tokenizer ─────────────────────────────────────────────────────
        emit("tokenize", "compiling tokenizer", 0.72);
        const int trc = run_streamed_command(
            // `--out` is a DIRECTORY. It was being handed
            // `<container>/tokenizer.soma`, so the script created a directory of
            // that name and wrote tokenizer.soma inside it — the engine looks for
            // a FILE at that path, found a directory, and every model admitted
            // through this pipeline served raw token ids. Nothing failed; the
            // tokenizer was simply never there.
            //
            // The container directory is also where tokenizer_oracle.bin belongs,
            // which is what the conformance stage checks the tokenizer against.
            {tools.python, (tools_dir / "compile_tokenizer.py").string(), local_source, "--out",
             container},
            fs::current_path(),
            [&](const std::string& line, bool) {
                if (!util::trim(line).empty()) emit("tokenize", util::trim(line), 0.72);
            },
            canceled, &err);
        if (canceled()) {
            progress.canceled = true;
            fail("canceled during tokenizer compilation");
            return;
        }
        if (trc != 0) {
            // NOT fatal. A container without a compiled tokenizer serves token
            // ids rather than text, which is degraded but honest — and refusing
            // the whole admission over it would discard hours of conversion.
            MM_WARN("admission {}: tokenizer compilation failed (exit {}); container is "
                    "usable but will not detokenize", progress.operation_id, trc);
            emit("tokenize", "tokenizer compilation failed; continuing without it", 0.72);
        }
    }

    // ── 2b. the conformance oracle ───────────────────────────────────────────
    //
    // A tiny-random model carrying THIS architecture, plus the logits
    // `transformers` produces for it. That is what turns ladder stage 1 from
    // "skipped, needs an oracle" into an answer.
    //
    // Built from the SOURCE, not the container: make_oracle.py shrinks the real
    // config and keeps every semantic field, so the fixture validates the
    // architecture rather than the admitted weights. Random weights are the
    // point — a real checkpoint can be approximately right in ways that hide a
    // bug for weeks.
    //
    // NOT fatal when it fails, for the same reason the tokenizer compile is not:
    // the model is still admissible and still routable, and discarding hours of
    // conversion over a missing fixture would be the wrong trade. The ladder
    // then reports stage 1 as skipped, which is the honest result.
    if (!container_is_ready) {
        emit("oracle", "building the conformance fixture", 0.74);
        std::string err;
        const auto fixture = (fs::path(container) / "conformance").string();
        const int orc = run_streamed_command(
            {tools.python, (tools_dir / "make_oracle.py").string(), local_source, "--out",
             (fs::path(container) / "conformance-build").string()},
            fs::current_path(),
            [&](const std::string& line, bool) {
                if (!util::trim(line).empty()) emit("oracle", util::trim(line), 0.76);
            },
            canceled, &err);
        if (canceled()) {
            progress.canceled = true;
            fail("canceled while building the oracle");
            return;
        }
        if (orc == 0) {
            // make_oracle.py writes <out>/<model-name>/; lift it to a fixed path
            // so `soma conform` does not have to guess the model's name.
            std::error_code ec;
            fs::remove_all(fixture, ec);
            const fs::path built((fs::path(container) / "conformance-build"));
            for (const auto& e : fs::directory_iterator(built, ec)) {
                if (e.is_directory(ec)) {
                    fs::rename(e.path(), fixture, ec);
                    break;
                }
            }
            fs::remove_all(built, ec);
            emit("oracle", ec ? "built, but could not be placed: " + ec.message() : "built",
                 0.78);
        } else {
            MM_WARN("admission {}: make_oracle.py exited {}; ladder stage 1 will report "
                    "skipped", progress.operation_id, orc);
            emit("oracle", "not built (exit " + std::to_string(orc) +
                               "); conformance stage 1 will report skipped", 0.78);
        }
    }

    // The bf16 REFERENCE, for ladder stage 2.
    //
    // The counterpart to the oracle above, and the opposite trade: that fixture
    // is tiny-random and validates the architecture; this is a forward pass over
    // the REAL checkpoint at bf16, and it is the only evidence that this
    // quantization of these weights is faithful. Without it `reject` is
    // unreachable for a real model — the verdict exists and nothing can produce
    // it.
    //
    // Also not fatal, on the same reasoning. The cost is real: loading the source
    // at bf16 is the slowest thing in admission, and on a host that cannot hold
    // it this will fail rather than swap the machine to death. Stage 2 then
    // reports skipped, which distinguishes "no evidence" from "bad model" — and
    // only one of those should ever read as a reject.
    if (!container_is_ready) {
        emit("reference", "bf16 reference pass over the real checkpoint", 0.79);
        std::string err;
        const auto ref_build = (fs::path(container) / "reference-build").string();
        const int rc = run_streamed_command(
            {tools.python, (tools_dir / "make_reference.py").string(), local_source, "--out",
             ref_build},
            fs::current_path(),
            [&](const std::string& line, bool) {
                if (!util::trim(line).empty()) emit("reference", util::trim(line), 0.80);
            },
            canceled, &err);
        if (canceled()) {
            progress.canceled = true;
            fail("canceled while building the reference");
            return;
        }
        std::error_code ec;
        const fs::path produced = fs::path(ref_build) / "oracle.bin";
        if (rc == 0 && fs::exists(produced, ec)) {
            // Beside the stage-1 fixture, under a name that says which it is.
            // Both are SOMAORCL and a single `oracle.bin` for two stages is how
            // one would silently be scored against the other's model.
            const fs::path dest = fs::path(container) / "conformance" / "reference.bin";
            fs::create_directories(dest.parent_path(), ec);
            fs::remove(dest, ec);
            fs::rename(produced, dest, ec);
            if (ec) fs::copy_file(produced, dest, fs::copy_options::overwrite_existing, ec);
            emit("reference", ec ? "built, but could not be placed: " + ec.message() : "built",
                 0.81);
        } else {
            MM_WARN("admission {}: make_reference.py exited {}; ladder stage 2 will report "
                    "skipped", progress.operation_id, rc);
            emit("reference", "not built (exit " + std::to_string(rc) +
                                  "); conformance stage 2 will report skipped", 0.81);
        }
        fs::remove_all(ref_build, ec);
    }

    // ── 3. plan ──────────────────────────────────────────────────────────────
    //
    // `soma plan --json` is the verdict. It reads headers only and allocates
    // nothing, which is why it is safe to run on control rather than requiring a
    // node that could host the model.
    // Already answered when the architecture check ran: re-planning the same
    // source would produce the same document and the same verdict.
    emit("profile", arch_unsupported ? "already planned; no backend for this architecture"
                                     : "planning",
         0.80);
    if (!arch_unsupported) {
        std::string err;
        const int prc = run_streamed_command(
            {tools.soma_path, "plan", "--model-dir", container_is_ready ? local_source : container,
             "--json"},
            fs::current_path(),
            [&](const std::string& line, bool is_stderr) {
                if (!is_stderr) plan_json += line;
            },
            canceled, &err);
        if (canceled()) {
            progress.canceled = true;
            fail("canceled during planning");
            return;
        }
        if (prc != 0) {
            fail("soma plan failed (exit " + std::to_string(prc) + ")" +
                 (err.empty() ? "" : ": " + err));
            return;
        }
    }

    // ── 4. conformance ───────────────────────────────────────────────────────
    //
    // `soma conform --json`, for the same reason `plan` is a subcommand: the
    // codec under test is the one the engine uses, and a second implementation
    // living in control is how the two come to disagree.
    //
    // Stages that need a transformers oracle for THIS model do not run here and
    // are recorded as SKIPPED with what they would need — never as passed.
    emit("conformance",
         arch_unsupported ? "not run; there is no container to check" : "running the ladder", 0.90);
    std::vector<ConformanceEntry> conf;
    bool conformance_failed = false;
    if (!arch_unsupported) {
        std::string conform_json, err;
        const int crc = run_streamed_command(
            {tools.soma_path, "conform", "--model-dir", container_is_ready ? local_source : container,
             "--json"},
            fs::current_path(),
            [&](const std::string& line, bool is_stderr) {
                if (!is_stderr) conform_json += line;
            },
            canceled, &err);
        if (canceled()) {
            progress.canceled = true;
            fail("canceled during conformance");
            return;
        }
        if (crc != 0) {
            // Could-not-run, which is different from a finding. `conform` exits 0
            // when a stage fails precisely so these stay distinguishable.
            MM_WARN("admission {}: soma conform exited {}; the ladder was not run",
                    progress.operation_id, crc);
            emit("conformance", "could not run the ladder (exit " + std::to_string(crc) + ")", 0.92);
        } else {
            try {
                const auto j = nlohmann::json::parse(conform_json);
                const auto now = util::now_ms();
                for (const auto& s : j.value("stages", nlohmann::json::array())) {
                    ConformanceEntry e;
                    e.stage = s.value("stage", std::string{});
                    e.status = s.value("status", std::string{"skipped"});
                    e.passed = (e.status == "passed");
                    e.detail = s.value("detail", nlohmann::json::object()).dump();
                    e.ran_at_ms = now;
                    if (e.stage.empty()) continue;
                    if (e.status == "failed") conformance_failed = true;
                    conf.push_back(std::move(e));
                }
                std::size_t ran = 0;
                for (const auto& e : conf)
                    if (e.status != "skipped") ++ran;
                emit("conformance",
                     std::to_string(ran) + " of " + std::to_string(conf.size()) +
                         " stages ran; " + (conformance_failed ? "FAILED" : "no failures"),
                     0.92);
            } catch (const std::exception& e) {
                MM_WARN("admission {}: could not parse conform output: {}", progress.operation_id,
                        e.what());
                emit("conformance", "conform produced unreadable output", 0.92);
            }
        }
    }

    // ── 5. finalize ──────────────────────────────────────────────────────────
    emit("finalize", "recording", 0.95);
    // Keyed on arch_hash below, so a reprofile updates the row it came from
    // rather than creating a second one for the same weights.
    AdmittedModel m;
    m.model_dir = (container_is_ready || arch_unsupported) ? local_source : container;
    m.name = name;
    m.source_repo = container_is_ready ? std::string{} : source;
    try {
        const auto j = nlohmann::json::parse(plan_json);
        m.arch_hash = j.value("arch_hash", std::string{});
        m.attention_family = j.value("attention_family", std::string{"gqa"});
        m.n_layers = j.value("n_layers", 0u);
        m.n_moe_layers = j.value("n_moe_layers", 0u);
        m.n_experts = j.value("n_experts", 0u);
        m.top_k = j.value("top_k", 0u);
        m.expert_bytes = j.value("expert_bytes", std::int64_t{0});
        m.bytes_per_token = j.value("bytes_per_token", std::int64_t{0});
        m.total_routed_bytes = j.value("total_routed_bytes", std::int64_t{0});
        m.active_fraction = j.value("active_fraction", 0.0);
        m.verdict = parse_verdict(j.value("verdict", std::string{"reject"}));
        m.verdict_reason = j.value("verdict_reason", std::string{});
        if (const auto planned = j.value("model_name", std::string{}); !planned.empty()) {
            m.name = planned;
        }
        m.verdict_basis = j.dump();
        m.profiled_at_ms = util::now_ms();
    } catch (const std::exception& e) {
        fail(std::string("could not parse `soma plan --json` output: ") + e.what());
        return;
    }
    if (m.arch_hash.empty()) {
        // Without it there is no identity to key the row on, and re-admitting
        // would duplicate rather than update.
        fail("plan produced no arch_hash; the container may predate the field");
        return;
    }

    // A failed conformance stage is a REJECT verdict, not a failed request.
    //
    // The operator asked whether Soma can run this model; "no, and here is the
    // stage that says so" is an answer, and a rejected model is a successfully
    // admitted RECORD meaning "route this to the fallback". Applied after the
    // plan so the plan's own reasoning survives in verdict_basis — the two
    // reasons are different findings and neither should erase the other.
    if (conformance_failed) {
        m.verdict = ModelVerdict::Reject;
        m.verdict_reason = m.verdict_reason.empty()
                               ? std::string("conformance failed")
                               : "conformance failed (plan said: " + m.verdict_reason + ")";
    }

    std::string err;
    if (!upsert(m, err)) {
        fail("could not record the admission: " + err);
        return;
    }
    if (!conf.empty() && !record_conformance(m.id, conf, err)) {
        // Not fatal: the model is admitted and routable, and losing the ladder's
        // detail is worse reported than pretended away.
        MM_WARN("admission {}: could not record conformance rows: {}", progress.operation_id, err);
    }

    progress.model_id = m.id;
    progress.done = true;
    progress.cancelable = false;
    progress.finished_at_ms = util::now_ms();
    emit("finalize", "admitted as model " + std::to_string(m.id) + " (" +
                            to_string(m.verdict) + ")", 1.0);
    MM_INFO("admission {}: {} admitted as model {} with verdict {}", progress.operation_id,
            m.name, m.id, to_string(m.verdict));
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
