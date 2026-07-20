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
    inference_backend TEXT    NOT NULL DEFAULT 'vllm',

    -- RuntimeSettings columns
    ctx_size          INTEGER NOT NULL DEFAULT 4096,
    n_gpu_layers      INTEGER NOT NULL DEFAULT -1,
    n_threads         INTEGER NOT NULL DEFAULT -1,
    n_threads_http    INTEGER NOT NULL DEFAULT -1,
    parallel          INTEGER NOT NULL DEFAULT 1,
    batch_size        INTEGER NOT NULL DEFAULT -1,
    ubatch_size       INTEGER NOT NULL DEFAULT -1,
    temperature       REAL    NOT NULL DEFAULT 0.7,
    top_p             REAL    NOT NULL DEFAULT 0.9,
    top_k             INTEGER NOT NULL DEFAULT -1,
    min_p             REAL    NOT NULL DEFAULT -1.0,
    presence_penalty  REAL    NOT NULL DEFAULT 0.0,
    repeat_penalty    REAL    NOT NULL DEFAULT -1.0,
    max_tokens        INTEGER NOT NULL DEFAULT 1024,
    flash_attn        INTEGER NOT NULL DEFAULT 1,   -- stored as bool (0/1)
    extra_args_json   TEXT    NOT NULL DEFAULT '[]', -- JSON array of strings
    vllm_settings_json TEXT   NOT NULL DEFAULT '{}',
    api_settings_json TEXT    NOT NULL DEFAULT '{}',

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
    trace_events_json TEXT   NOT NULL DEFAULT '[]',
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

-- Voice design / TTS metadata. Audio and prompt artifacts live under the
-- agent data directory; SQLite stores paths and lifecycle state.
CREATE TABLE IF NOT EXISTS agent_voice_proposals (
    id                       TEXT    NOT NULL PRIMARY KEY,
    display_name             TEXT    NOT NULL DEFAULT '',
    language                 TEXT    NOT NULL DEFAULT 'Auto',
    voice_description        TEXT    NOT NULL DEFAULT '',
    sample_text              TEXT    NOT NULL DEFAULT '',
    rationale                TEXT    NOT NULL DEFAULT '',
    status                   TEXT    NOT NULL DEFAULT 'pending',
    provider                 TEXT    NOT NULL DEFAULT 'qwen3-tts',
    voice_design_model_id    TEXT    NOT NULL DEFAULT '',
    clone_model_id           TEXT    NOT NULL DEFAULT '',
    preview_audio_path       TEXT    NOT NULL DEFAULT '',
    voice_clone_prompt_path  TEXT    NOT NULL DEFAULT '',
    error                    TEXT    NOT NULL DEFAULT '',
    created_at_ms            INTEGER NOT NULL,
    updated_at_ms            INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_voice_profiles (
    id                        TEXT    NOT NULL PRIMARY KEY,
    display_name              TEXT    NOT NULL DEFAULT '',
    language                  TEXT    NOT NULL DEFAULT 'Auto',
    voice_description         TEXT    NOT NULL DEFAULT '',
    sample_text               TEXT    NOT NULL DEFAULT '',
    rationale                 TEXT    NOT NULL DEFAULT '',
    provider                  TEXT    NOT NULL DEFAULT 'qwen3-tts',
    voice_design_model_id     TEXT    NOT NULL DEFAULT '',
    clone_model_id            TEXT    NOT NULL DEFAULT '',
    reference_audio_path      TEXT    NOT NULL DEFAULT '',
    voice_clone_prompt_path   TEXT    NOT NULL DEFAULT '',
    approved_from_proposal_id TEXT    NOT NULL DEFAULT '',
    active                    INTEGER NOT NULL DEFAULT 0,
    created_at_ms             INTEGER NOT NULL,
    updated_at_ms             INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tts_audio_cache (
    id               TEXT    NOT NULL PRIMARY KEY,
    voice_profile_id TEXT    NOT NULL DEFAULT '',
    conversation_id  TEXT    NOT NULL DEFAULT '',
    message_index    INTEGER NOT NULL DEFAULT -1,
    text_hash        TEXT    NOT NULL DEFAULT '',
    audio_path       TEXT    NOT NULL DEFAULT '',
    mime_type        TEXT    NOT NULL DEFAULT 'audio/wav',
    format           TEXT    NOT NULL DEFAULT 'wav',
    sample_rate      INTEGER NOT NULL DEFAULT 0,
    duration_ms      INTEGER NOT NULL DEFAULT 0,
    created_at_ms    INTEGER NOT NULL,
    expires_at_ms    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_voice_proposals_status
    ON agent_voice_proposals(status, created_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_voice_profiles_active
    ON agent_voice_profiles(active, updated_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_tts_cache_lookup
    ON tts_audio_cache(voice_profile_id, text_hash, conversation_id, message_index);
CREATE INDEX IF NOT EXISTS idx_tts_cache_expiry
    ON tts_audio_cache(expires_at_ms);

-- ── Seed migration marker ─────────────────────────────────────────────────────
INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);
INSERT OR IGNORE INTO schema_migrations(version) VALUES (5);
INSERT OR IGNORE INTO schema_migrations(version) VALUES (6);
INSERT OR IGNORE INTO schema_migrations(version) VALUES (8);
