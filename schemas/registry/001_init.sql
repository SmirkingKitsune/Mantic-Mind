-- Soma control-side registry — migration 001
--
-- Target: {data_dir}/control.db
--
-- This is the FIRST control-wide database in Mantic-Mind. Until now the only
-- SQLite in the system was per-agent ({data_dir}/agents/{id}/agent.db); node
-- state lived in a JSON journal and remembered nodes in nodes.json.
--
-- Migration mechanism follows AgentDB::run_migrations() exactly:
--   if (!has_version(N)) { SQLite::Transaction tx(*db_); <DDL>;
--                          INSERT OR IGNORE INTO schema_migrations(version)
--                          VALUES (N); tx.commit(); }
-- Each NNN_*.sql file is one such block. Never edit an applied migration;
-- add the next one.
--
-- Conventions inherited from agent_db.cpp:
--   * timestamps are INTEGER milliseconds since epoch
--   * WAL mode, foreign keys ON
--   * TEXT columns hold JSON where the shape is open-ended

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER NOT NULL PRIMARY KEY,
    applied_at INTEGER NOT NULL
               DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- model — the admission record. `verdict` drives backend selection.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model (
    id                   INTEGER PRIMARY KEY,
    arch_hash            TEXT    NOT NULL UNIQUE,
    name                 TEXT    NOT NULL,
    source_repo          TEXT,
    source_revision      TEXT,
    model_dir            TEXT    NOT NULL,   -- converted container, control-relative
    schema_version       INTEGER NOT NULL,
    arch_json            TEXT    NOT NULL,   -- the full arch.json, verbatim

    -- denormalized from arch.json for cheap queries and the TUI
    attention_family     TEXT    NOT NULL,   -- mha | gqa | mla | mla+dsa
    n_layers             INTEGER NOT NULL,
    n_moe_layers         INTEGER NOT NULL,
    n_experts            INTEGER NOT NULL,
    top_k                INTEGER NOT NULL,

    -- economics (arch.json §7); NOT covered by arch_hash
    expert_bytes         INTEGER NOT NULL,
    bytes_per_token      INTEGER NOT NULL,
    total_routed_bytes   INTEGER NOT NULL,
    dense_resident_bytes INTEGER NOT NULL,
    active_fraction      REAL    NOT NULL,
    measured_disk_bw     INTEGER,            -- bytes/sec at THIS model's expert size

    -- the verdict, and the host assumptions that produced it
    verdict              TEXT    NOT NULL
                         CHECK (verdict IN ('stream','hybrid','resident-only','reject')),
    verdict_basis        TEXT    NOT NULL,   -- JSON {ram_budget, ctx, quant, host}
    verdict_reason       TEXT,               -- human-readable, surfaced on /v1/placements

    admitted_at          INTEGER NOT NULL,
    profiled_at          INTEGER
);

-- The verdict stored here is the ADMISSION-HOST verdict. `soma plan --json`
-- re-derives it against the actual target node's budget, because the verdict is
-- a property of (model, quantization, host) and not of the model alone. See
-- schemas/arch-ir.md §8. Consumers wanting the effective verdict must read
-- /v1/placements, not this column.

CREATE INDEX IF NOT EXISTS idx_model_verdict ON model(verdict);
CREATE INDEX IF NOT EXISTS idx_model_name    ON model(name);

-- ─────────────────────────────────────────────────────────────────────────────
-- expert_heat — routing histogram. Bootstrapped at admission over a calibration
-- corpus, updated with exponential decay during serving. Feeds the startup pin,
-- GET /v1/models/{id}/heat, and the brain grid.
-- ─────────────────────────────────────────────────────────────────────────────
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
);

-- Ranked pin selection at startup reads this index.
CREATE INDEX IF NOT EXISTS idx_heat_rank ON expert_heat(model_id, decayed DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- kernel_choice — autotuner output. Codegen'd into the per-model specialization
-- header as a static dispatch table; never consulted at runtime.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kernel_choice (
    model_id INTEGER NOT NULL REFERENCES model(id) ON DELETE CASCADE,
    op       TEXT    NOT NULL,    -- gemm | gemv | moe_expert | attn_qk | attn_av
    m        INTEGER NOT NULL,
    n        INTEGER NOT NULL,
    k        INTEGER NOT NULL,
    dtype    TEXT    NOT NULL,
    impl     TEXT    NOT NULL,    -- chosen implementation id
    gflops   REAL    NOT NULL,
    PRIMARY KEY (model_id, op, m, n, k, dtype)
);

-- Which kernel wins at a given shape is empirical, not derivable: int4 single-row
-- measured SLOWER than f32 in the prior art. The (m,n,k) key exists because the
-- winner changes with batch size, which is also why greedy output is not
-- byte-stable across batch sizes (docs/architecture.md §10).

-- ─────────────────────────────────────────────────────────────────────────────
-- pilot_profile — per-layer router-lookahead recall. Prefetch is enabled per
-- layer only above threshold; a wrong prefetch is worse than none because it
-- evicts something useful.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pilot_profile (
    model_id    INTEGER NOT NULL REFERENCES model(id) ON DELETE CASCADE,
    layer       INTEGER NOT NULL,
    recall_at_k REAL    NOT NULL,
    samples     INTEGER NOT NULL,
    PRIMARY KEY (model_id, layer)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- conformance — the admission gate. Stages 1-2 failing means reject → fallback.
-- Stage 3 failing with 1-2 passing is a QUANTIZATION finding, not a correctness
-- bug: different remediation. The distinction is why `stage` is a column and not
-- a boolean.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conformance (
    model_id INTEGER NOT NULL REFERENCES model(id) ON DELETE CASCADE,
    stage    TEXT    NOT NULL
             CHECK (stage IN ('fp32_tiny_tf',
                              'quant_tiny_greedy',
                              'real_logit_kl',
                              'accuracy_floor')),
    passed   INTEGER NOT NULL,
    detail   TEXT,                -- JSON: metrics, first divergence, thresholds
    ran_at   INTEGER NOT NULL,
    PRIMARY KEY (model_id, stage)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- api_token — the scope store. Mantic-Mind has had exactly one flat bearer token
-- with no scope mechanism; this table introduces the first one.
--
-- Scopes:
--   read      all GETs, telemetry SSE
--   chat      agent chat, conversations, memories, attachments
--   operator  admission, reprofile, delete, placement override, suspend/restore
--
-- Admission kicks off hours-long, resource-consuming conversion and profiling.
-- It must not sit behind the same token that lets a client send a chat message.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_token (
    id           INTEGER PRIMARY KEY,
    token_sha256 TEXT    NOT NULL UNIQUE,   -- never store the token itself
    label        TEXT    NOT NULL,
    scopes       TEXT    NOT NULL,          -- comma-separated subset of read,chat,operator
    created_at   INTEGER NOT NULL,
    last_used_at INTEGER,
    revoked_at   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_token_active
    ON api_token(token_sha256) WHERE revoked_at IS NULL;

-- The legacy ControlConfig::external_api_token is grandfathered at startup as a
-- synthetic all-scopes token so existing deployments keep working. It is NOT
-- inserted here — it is matched in the middleware before the table lookup, so
-- that rotating it in config does not leave a stale row.

-- ─────────────────────────────────────────────────────────────────────────────
-- placement_history — why a given agent landed on a given backend. Bounded by
-- retention sweep, not by trigger.
--
-- PerformanceTracker is deliberately in-memory and non-persistent; this table is
-- not a second copy of it. It records BACKEND-SELECTION DECISIONS, which are
-- rare, causally interesting, and otherwise unrecoverable after the fact.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS placement_history (
    id             INTEGER PRIMARY KEY,
    agent_id       TEXT    NOT NULL,
    node_id        TEXT    NOT NULL,
    slot_id        TEXT    NOT NULL,
    model_id       INTEGER REFERENCES model(id) ON DELETE SET NULL,
    backend        TEXT    NOT NULL,        -- soma | llama-cpp
    backend_reason TEXT    NOT NULL,        -- verdict | override | no-record | fallback-forced
    footprint_json TEXT    NOT NULL,        -- {vram_mb, ram_mb, disk_mb}
    placed_at      INTEGER NOT NULL,
    released_at    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_placement_agent ON placement_history(agent_id, placed_at DESC);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);
