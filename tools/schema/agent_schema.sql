-- ─────────────────────────────────────────────────────────────────────────────
-- Mantic-Mind: per-agent SQLite schema
-- Applied once per database file via run_migrations().
-- WAL mode and foreign-key enforcement are set at connection open time.
-- ─────────────────────────────────────────────────────────────────────────────

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Migration version tracking ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER NOT NULL PRIMARY KEY,
    applied_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s', 'now') AS INTEGER) * 1000)
);

-- ── Agent configuration (exactly 1 row per database) ─────────────────────────
CREATE TABLE IF NOT EXISTS agent_config (
    id                TEXT    NOT NULL PRIMARY KEY,
    name              TEXT    NOT NULL,
    model_path        TEXT    NOT NULL DEFAULT '',
    system_prompt     TEXT    NOT NULL DEFAULT '',

    -- LlamaSettings columns
    ctx_size          INTEGER NOT NULL DEFAULT 4096,
    n_gpu_layers      INTEGER NOT NULL DEFAULT -1,
    n_threads         INTEGER NOT NULL DEFAULT -1,
    temperature       REAL    NOT NULL DEFAULT 0.7,
    top_p             REAL    NOT NULL DEFAULT 0.9,
    max_tokens        INTEGER NOT NULL DEFAULT 1024,
    flash_attn        INTEGER NOT NULL DEFAULT 1,   -- stored as bool (0/1)
    extra_args_json   TEXT    NOT NULL DEFAULT '[]', -- JSON array of strings

    -- Feature flags
    reasoning_enabled INTEGER NOT NULL DEFAULT 0,
    memories_enabled  INTEGER NOT NULL DEFAULT 1,
    tools_enabled     INTEGER NOT NULL DEFAULT 0,

    preferred_node_id TEXT    NOT NULL DEFAULT '',

    created_at_ms     INTEGER NOT NULL DEFAULT (CAST(strftime('%s', 'now') AS INTEGER) * 1000),
    updated_at_ms     INTEGER NOT NULL DEFAULT (CAST(strftime('%s', 'now') AS INTEGER) * 1000)
);

-- ── Conversations ─────────────────────────────────────────────────────────────
-- Exactly one conversation has is_active = 1 at any time.
-- Compacted conversations keep their messages for history but is_active = 0.
CREATE TABLE IF NOT EXISTS conversations (
    id                  TEXT    NOT NULL PRIMARY KEY,
    title               TEXT    NOT NULL DEFAULT '',
    total_tokens        INTEGER NOT NULL DEFAULT 0,
    is_active           INTEGER NOT NULL DEFAULT 0, -- bool (0/1)
    compaction_summary  TEXT    NOT NULL DEFAULT '',
    parent_conv_id      TEXT    REFERENCES conversations(id) ON DELETE SET NULL,
    created_at_ms       INTEGER NOT NULL DEFAULT (CAST(strftime('%s', 'now') AS INTEGER) * 1000),
    updated_at_ms       INTEGER NOT NULL DEFAULT (CAST(strftime('%s', 'now') AS INTEGER) * 1000)
);

CREATE INDEX IF NOT EXISTS idx_conversations_active
    ON conversations(is_active);
CREATE INDEX IF NOT EXISTS idx_conversations_created
    ON conversations(created_at_ms DESC);

-- ── Messages ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL
                    CHECK(role IN ('system', 'user', 'assistant', 'tool')),
    content         TEXT    NOT NULL DEFAULT '',
    tool_calls_json TEXT    NOT NULL DEFAULT '[]', -- JSON array of ToolCall
    tool_call_id    TEXT    NOT NULL DEFAULT '',   -- non-empty for role='tool'
    thinking_text   TEXT    NOT NULL DEFAULT '',   -- from <think>…</think>
    token_count     INTEGER NOT NULL DEFAULT 0,
    sequence_num    INTEGER NOT NULL DEFAULT 0,
    timestamp_ms    INTEGER NOT NULL DEFAULT (CAST(strftime('%s', 'now') AS INTEGER) * 1000)
);

CREATE INDEX IF NOT EXISTS idx_messages_conv_seq
    ON messages(conversation_id, sequence_num);

-- ── Memories ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memories (
    id             TEXT NOT NULL PRIMARY KEY,
    content        TEXT NOT NULL,
    source_conv_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    importance     REAL NOT NULL DEFAULT 0.5,
    created_at_ms  INTEGER NOT NULL DEFAULT (CAST(strftime('%s', 'now') AS INTEGER) * 1000)
);

CREATE INDEX IF NOT EXISTS idx_memories_rank
    ON memories(importance DESC, created_at_ms DESC);

-- ── Seed migration marker ─────────────────────────────────────────────────────
INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);
