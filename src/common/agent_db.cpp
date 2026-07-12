#include "common/agent_db.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <SQLiteCpp/SQLiteCpp.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace mm {

//
namespace {

// nlohmann ADL picks up mm::to_json / mm::from_json automatically.
std::string serialize(const std::vector<ToolCall>& tcs) {
    return nlohmann::json(tcs).dump();
}
std::vector<ToolCall> deserialize_tool_calls(const std::string& s) {
    try { return nlohmann::json::parse(s).get<std::vector<ToolCall>>(); }
    catch (const std::exception& e) { MM_DEBUG("deserialize_tool_calls: {}", e.what()); return {}; }
}

std::string serialize(const std::vector<TraceEvent>& events) {
    return nlohmann::json(events).dump();
}
std::vector<TraceEvent> deserialize_trace_events(const std::string& s) {
    try { return nlohmann::json::parse(s).get<std::vector<TraceEvent>>(); }
    catch (const std::exception& e) { MM_DEBUG("deserialize_trace_events: {}", e.what()); return {}; }
}

std::string serialize(const std::vector<std::string>& v) {
    return nlohmann::json(v).dump();
}
std::vector<std::string> deserialize_strings(const std::string& s) {
    try { return nlohmann::json::parse(s).get<std::vector<std::string>>(); }
    catch (const std::exception& e) { MM_DEBUG("deserialize_strings: {}", e.what()); return {}; }
}

std::string serialize(const VllmSettings& s) {
    return nlohmann::json(s).dump();
}
VllmSettings deserialize_vllm_settings(const std::string& s) {
    try { return nlohmann::json::parse(s).get<VllmSettings>(); }
    catch (const std::exception& e) { MM_DEBUG("deserialize_vllm_settings: {}", e.what()); return {}; }
}

std::string serialize(const ApiSettings& s) {
    ApiSettings persisted = s;
    persisted.api_key.clear();
    return nlohmann::json(persisted).dump();
}
ApiSettings deserialize_api_settings(const std::string& s) {
    try { return nlohmann::json::parse(s).get<ApiSettings>(); }
    catch (const std::exception& e) { MM_DEBUG("deserialize_api_settings: {}", e.what()); return {}; }
}

std::string normalize_inference_backend(std::string backend) {
    backend = util::to_lower(util::trim(backend));
    // llama.cpp is the default runtime on this branch: unspecified resolves to
    // it, and the legacy llama spellings canonicalize onto the wire form
    // ("llama-cpp"). "vllm" and "api" pass through unchanged.
    if (backend.empty()) return "llama-cpp";
    if (backend == "llama.cpp" || backend == "llama" || backend == "llama-cpp")
        return "llama-cpp";
    return backend;
}

bool is_transient_sqlite_lock(const SQLite::Exception& e) {
    const std::string msg = util::to_lower(e.what());
    return msg.find("locked") != std::string::npos ||
           msg.find("busy") != std::string::npos;
}

template <typename Fn>
void run_with_sqlite_retry(const char* operation, Fn&& fn) {
    constexpr int kMaxAttempts = 5;
    constexpr auto kRetryDelay = std::chrono::milliseconds(150);

    for (int attempt = 1; attempt <= kMaxAttempts; ++attempt) {
        try {
            fn();
            return;
        } catch (const SQLite::Exception& e) {
            if (!is_transient_sqlite_lock(e) || attempt == kMaxAttempts) {
                throw;
            }
            MM_WARN("AgentDB {} hit a transient SQLite lock (attempt {}/{}): {}",
                    operation, attempt, kMaxAttempts, e.what());
            std::this_thread::sleep_for(kRetryDelay);
        }
    }
}

Message row_to_message(SQLite::Statement& q) {
    Message m;
    m.role          = message_role_from_string(q.getColumn("role").getText());
    m.content       = q.getColumn("content").getText();
    m.tool_calls    = deserialize_tool_calls(q.getColumn("tool_calls_json").getText());
    m.tool_call_id  = q.getColumn("tool_call_id").getText();
    m.thinking_text = q.getColumn("thinking_text").getText();
    m.token_count   = q.getColumn("token_count").getInt();
    m.timestamp_ms  = q.getColumn("timestamp_ms").getInt64();
    m.trace_events  = deserialize_trace_events(q.getColumn("trace_events_json").getText());
    return m;
}

AgentVoiceProfile row_to_voice_profile(SQLite::Statement& q, const AgentId& agent_id) {
    AgentVoiceProfile p;
    p.id                        = q.getColumn("id").getText();
    p.agent_id                  = agent_id;
    p.display_name              = q.getColumn("display_name").getText();
    p.language                  = q.getColumn("language").getText();
    p.voice_description         = q.getColumn("voice_description").getText();
    p.sample_text               = q.getColumn("sample_text").getText();
    p.rationale                 = q.getColumn("rationale").getText();
    p.provider                  = q.getColumn("provider").getText();
    p.voice_design_model_id     = q.getColumn("voice_design_model_id").getText();
    p.clone_model_id            = q.getColumn("clone_model_id").getText();
    p.reference_audio_path      = q.getColumn("reference_audio_path").getText();
    p.voice_clone_prompt_path   = q.getColumn("voice_clone_prompt_path").getText();
    p.approved_from_proposal_id = q.getColumn("approved_from_proposal_id").getText();
    p.active                    = q.getColumn("active").getInt() != 0;
    p.created_at_ms             = q.getColumn("created_at_ms").getInt64();
    p.updated_at_ms             = q.getColumn("updated_at_ms").getInt64();
    return p;
}

VoiceDesignProposal row_to_voice_proposal(SQLite::Statement& q, const AgentId& agent_id) {
    VoiceDesignProposal p;
    p.id                    = q.getColumn("id").getText();
    p.agent_id              = agent_id;
    p.display_name          = q.getColumn("display_name").getText();
    p.language              = q.getColumn("language").getText();
    p.voice_description     = q.getColumn("voice_description").getText();
    p.sample_text           = q.getColumn("sample_text").getText();
    p.rationale             = q.getColumn("rationale").getText();
    p.status                = q.getColumn("status").getText();
    p.provider              = q.getColumn("provider").getText();
    p.voice_design_model_id = q.getColumn("voice_design_model_id").getText();
    p.clone_model_id        = q.getColumn("clone_model_id").getText();
    p.preview_audio_path    = q.getColumn("preview_audio_path").getText();
    p.voice_clone_prompt_path = q.getColumn("voice_clone_prompt_path").getText();
    p.error                 = q.getColumn("error").getText();
    p.created_at_ms         = q.getColumn("created_at_ms").getInt64();
    p.updated_at_ms         = q.getColumn("updated_at_ms").getInt64();
    return p;
}

TtsSynthesisResult row_to_tts_cache(SQLite::Statement& q, const AgentId& agent_id) {
    TtsSynthesisResult r;
    r.cache_id         = q.getColumn("id").getText();
    r.agent_id         = agent_id;
    r.voice_profile_id = q.getColumn("voice_profile_id").getText();
    r.conversation_id  = q.getColumn("conversation_id").getText();
    r.message_index    = q.getColumn("message_index").getInt();
    r.text_hash        = q.getColumn("text_hash").getText();
    r.audio_path       = q.getColumn("audio_path").getText();
    r.mime_type        = q.getColumn("mime_type").getText();
    r.format           = q.getColumn("format").getText();
    r.sample_rate      = q.getColumn("sample_rate").getInt();
    r.duration_ms      = q.getColumn("duration_ms").getInt();
    r.created_at_ms    = q.getColumn("created_at_ms").getInt64();
    r.expires_at_ms    = q.getColumn("expires_at_ms").getInt64();
    return r;
}

} // namespace

//
AgentDB::AgentDB(const AgentId& agent_id, const std::string& data_dir)
    : agent_id_(agent_id)
{
    namespace fs = std::filesystem;
    fs::path dir = fs::path(data_dir) / "agents" / agent_id;
    fs::create_directories(dir);

    db_ = std::make_unique<SQLite::Database>(
        (dir / "agent.db").string(),
        SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE
    );

    run_with_sqlite_retry("init_pragmas", [&] {
        db_->exec("PRAGMA journal_mode=WAL");
        db_->exec("PRAGMA foreign_keys=ON");
        db_->exec("PRAGMA synchronous=NORMAL");
        db_->exec("PRAGMA cache_size=-8000"); // 8 MB page cache
        db_->exec("PRAGMA busy_timeout=5000");
    });

    run_with_sqlite_retry("run_migrations", [&] {
        run_migrations();
    });
}

AgentDB::~AgentDB() = default;

void AgentDB::run_migrations() {
    db_->exec(R"(
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INTEGER NOT NULL PRIMARY KEY,
            applied_at INTEGER NOT NULL
                DEFAULT (CAST(strftime('%s','now') AS INTEGER)*1000)
        )
    )");

    auto has_version = [this](int version) {
        SQLite::Statement q(*db_,
            "SELECT 1 FROM schema_migrations WHERE version=?");
        q.bind(1, version);
        return q.executeStep();
    };
    auto has_column = [this](const std::string& table, const std::string& column) {
        SQLite::Statement q(*db_, "PRAGMA table_info(" + table + ")");
        while (q.executeStep()) {
            if (std::string(q.getColumn("name").getText()) == column) return true;
        }
        return false;
    };

    //
    if (!has_version(1)) {
        SQLite::Transaction tx(*db_);

        db_->exec(R"(
            CREATE TABLE IF NOT EXISTS agent_config (
                id                TEXT    NOT NULL PRIMARY KEY,
                name              TEXT    NOT NULL,
                model_path        TEXT    NOT NULL DEFAULT '',
                system_prompt     TEXT    NOT NULL DEFAULT '',
                inference_backend TEXT    NOT NULL DEFAULT 'llama-cpp',
                ctx_size          INTEGER NOT NULL DEFAULT 4096,
                n_gpu_layers      INTEGER NOT NULL DEFAULT -1,
                n_threads         INTEGER NOT NULL DEFAULT -1,
                n_threads_http    INTEGER NOT NULL DEFAULT -1,
                parallel          INTEGER NOT NULL DEFAULT 1,
                batch_size        INTEGER NOT NULL DEFAULT -1,
                ubatch_size       INTEGER NOT NULL DEFAULT -1,
                temperature       REAL    NOT NULL DEFAULT 0.7,
                top_p             REAL    NOT NULL DEFAULT 0.9,
                max_tokens        INTEGER NOT NULL DEFAULT 1024,
                flash_attn        INTEGER NOT NULL DEFAULT 1,
                extra_args_json   TEXT    NOT NULL DEFAULT '[]',
                vllm_settings_json TEXT   NOT NULL DEFAULT '{}',
                api_settings_json TEXT    NOT NULL DEFAULT '{}',
                reasoning_enabled INTEGER NOT NULL DEFAULT 0,
                memories_enabled  INTEGER NOT NULL DEFAULT 1,
                tools_enabled     INTEGER NOT NULL DEFAULT 0,
                preferred_node_id TEXT    NOT NULL DEFAULT '',
                created_at_ms     INTEGER NOT NULL
                    DEFAULT (CAST(strftime('%s','now') AS INTEGER)*1000),
                updated_at_ms     INTEGER NOT NULL
                    DEFAULT (CAST(strftime('%s','now') AS INTEGER)*1000)
            )
        )");

        db_->exec(R"(
            CREATE TABLE IF NOT EXISTS conversations (
                id                 TEXT    NOT NULL PRIMARY KEY,
                title              TEXT    NOT NULL DEFAULT '',
                total_tokens       INTEGER NOT NULL DEFAULT 0,
                is_active          INTEGER NOT NULL DEFAULT 0,
                compaction_summary TEXT    NOT NULL DEFAULT '',
                parent_conv_id     TEXT
                    REFERENCES conversations(id) ON DELETE SET NULL,
                created_at_ms      INTEGER NOT NULL
                    DEFAULT (CAST(strftime('%s','now') AS INTEGER)*1000),
                updated_at_ms      INTEGER NOT NULL
                    DEFAULT (CAST(strftime('%s','now') AS INTEGER)*1000)
            )
        )");

        db_->exec(R"(
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT    NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL DEFAULT '',
                tool_calls_json TEXT    NOT NULL DEFAULT '[]',
                tool_call_id    TEXT    NOT NULL DEFAULT '',
                thinking_text   TEXT    NOT NULL DEFAULT '',
                token_count     INTEGER NOT NULL DEFAULT 0,
                sequence_num    INTEGER NOT NULL DEFAULT 0,
                trace_events_json TEXT   NOT NULL DEFAULT '[]',
                timestamp_ms    INTEGER NOT NULL
                    DEFAULT (CAST(strftime('%s','now') AS INTEGER)*1000)
            )
        )");

        db_->exec(R"(
            CREATE TABLE IF NOT EXISTS memories (
                id             TEXT NOT NULL PRIMARY KEY,
                content        TEXT NOT NULL,
                source_conv_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
                importance     REAL NOT NULL DEFAULT 0.5,
                created_at_ms  INTEGER NOT NULL
                    DEFAULT (CAST(strftime('%s','now') AS INTEGER)*1000)
            )
        )");

        db_->exec("CREATE INDEX IF NOT EXISTS idx_conv_active   ON conversations(is_active)");
        db_->exec("CREATE INDEX IF NOT EXISTS idx_conv_created  ON conversations(created_at_ms DESC)");
        db_->exec("CREATE INDEX IF NOT EXISTS idx_msg_conv_seq  ON messages(conversation_id, sequence_num)");
        db_->exec("CREATE INDEX IF NOT EXISTS idx_mem_rank      ON memories(importance DESC, created_at_ms DESC)");

        db_->exec("INSERT OR IGNORE INTO schema_migrations(version) VALUES (1)");
        tx.commit();
    }

    //
    if (!has_version(2)) {
        SQLite::Transaction tx(*db_);

        // Create local_memories table
        db_->exec(R"(
            CREATE TABLE IF NOT EXISTS local_memories (
                id              TEXT    PRIMARY KEY,
                conversation_id TEXT    NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                content         TEXT    NOT NULL,
                created_at_ms   INTEGER NOT NULL,
                updated_at_ms   INTEGER NOT NULL
            )
        )");
        db_->exec("CREATE INDEX IF NOT EXISTS idx_local_mem_conv ON local_memories(conversation_id)");

        //
        db_->exec(R"(
            CREATE TABLE IF NOT EXISTS global_memories (
                id              TEXT    PRIMARY KEY,
                content         TEXT    NOT NULL,
                source_conv_id  TEXT    REFERENCES conversations(id) ON DELETE SET NULL,
                importance      REAL    NOT NULL DEFAULT 0.5,
                created_at_ms   INTEGER NOT NULL,
                updated_at_ms   INTEGER NOT NULL
            )
        )");

        // Copy existing data if the old table exists.
        // The check statement must be finalized before DROP TABLE, otherwise
        // the open cursor on sqlite_master causes SQLITE_LOCKED.
        bool has_old_memories = false;
        {
            SQLite::Statement check(*db_,
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'");
            has_old_memories = check.executeStep();
        }
        if (has_old_memories) {
            db_->exec(R"(
                INSERT OR IGNORE INTO global_memories
                    (id, content, source_conv_id, importance, created_at_ms, updated_at_ms)
                SELECT id, content, source_conv_id, importance, created_at_ms, created_at_ms
                FROM memories
            )");
            db_->exec("DROP TABLE memories");
        }

        // Drop old index if it exists, create new one
        db_->exec("DROP INDEX IF EXISTS idx_mem_rank");
        db_->exec("CREATE INDEX IF NOT EXISTS idx_global_mem_rank ON global_memories(importance DESC, created_at_ms DESC)");

        db_->exec("INSERT OR IGNORE INTO schema_migrations(version) VALUES (2)");
        tx.commit();
    }

    if (!has_version(3)) {
        SQLite::Transaction tx(*db_);
        bool has_trace_events_json = false;
        SQLite::Statement columns(*db_, "PRAGMA table_info(messages)");
        while (columns.executeStep()) {
            if (std::string(columns.getColumn("name").getText()) == "trace_events_json") {
                has_trace_events_json = true;
                break;
            }
        }
        if (!has_trace_events_json) {
            db_->exec("ALTER TABLE messages ADD COLUMN trace_events_json TEXT NOT NULL DEFAULT '[]'");
        }
        db_->exec("INSERT OR IGNORE INTO schema_migrations(version) VALUES (3)");
        tx.commit();
    }

    if (!has_version(4)) {
        SQLite::Transaction tx(*db_);
        if (!has_column("agent_config", "n_threads_http")) {
            db_->exec("ALTER TABLE agent_config ADD COLUMN n_threads_http INTEGER NOT NULL DEFAULT -1");
        }
        if (!has_column("agent_config", "parallel")) {
            db_->exec("ALTER TABLE agent_config ADD COLUMN parallel INTEGER NOT NULL DEFAULT 1");
        }
        if (!has_column("agent_config", "batch_size")) {
            db_->exec("ALTER TABLE agent_config ADD COLUMN batch_size INTEGER NOT NULL DEFAULT -1");
        }
        if (!has_column("agent_config", "ubatch_size")) {
            db_->exec("ALTER TABLE agent_config ADD COLUMN ubatch_size INTEGER NOT NULL DEFAULT -1");
        }
        db_->exec("INSERT OR IGNORE INTO schema_migrations(version) VALUES (4)");
        tx.commit();
    }

    if (!has_version(5)) {
        SQLite::Transaction tx(*db_);
        db_->exec(R"(
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
            )
        )");
        db_->exec(R"(
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
            )
        )");
        db_->exec(R"(
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
            )
        )");
        db_->exec("CREATE INDEX IF NOT EXISTS idx_voice_proposals_status ON agent_voice_proposals(status, created_at_ms DESC)");
        db_->exec("CREATE INDEX IF NOT EXISTS idx_voice_profiles_active ON agent_voice_profiles(active, updated_at_ms DESC)");
        db_->exec("CREATE INDEX IF NOT EXISTS idx_tts_cache_lookup ON tts_audio_cache(voice_profile_id, text_hash, conversation_id, message_index)");
        db_->exec("CREATE INDEX IF NOT EXISTS idx_tts_cache_expiry ON tts_audio_cache(expires_at_ms)");
        db_->exec("INSERT OR IGNORE INTO schema_migrations(version) VALUES (5)");
        tx.commit();
    }

    if (!has_version(6)) {
        SQLite::Transaction tx(*db_);
        // DBs that predate this column are from the pre-vLLM llama.cpp era, so
        // the backfill default matches this branch's default runtime. Agents
        // created under the vLLM line store 'vllm' explicitly and are untouched.
        if (!has_column("agent_config", "inference_backend")) {
            db_->exec("ALTER TABLE agent_config ADD COLUMN inference_backend TEXT NOT NULL DEFAULT 'llama-cpp'");
        }
        if (!has_column("agent_config", "vllm_settings_json")) {
            db_->exec("ALTER TABLE agent_config ADD COLUMN vllm_settings_json TEXT NOT NULL DEFAULT '{}'");
        }
        db_->exec("INSERT OR IGNORE INTO schema_migrations(version) VALUES (6)");
        tx.commit();
    }

    if (!has_version(7)) {
        SQLite::Transaction tx(*db_);
        if (!has_column("agent_config", "api_settings_json")) {
            db_->exec("ALTER TABLE agent_config ADD COLUMN api_settings_json TEXT NOT NULL DEFAULT '{}'");
        }
        db_->exec("INSERT OR IGNORE INTO schema_migrations(version) VALUES (7)");
        tx.commit();
    }
}

//
void AgentDB::save_config(const AgentConfig& cfg) {
    std::lock_guard g(mutex_);
    const auto& s = cfg.runtime_settings;
    const auto extra_args_json = serialize(s.extra_args);
    const auto vllm_settings_json = serialize(cfg.vllm_settings);
    const auto api_settings_json = serialize(cfg.api_settings);
    const auto inference_backend = normalize_inference_backend(cfg.inference_backend);

    run_with_sqlite_retry("save_config", [&] {
        SQLite::Statement q(*db_, R"(
            INSERT INTO agent_config
                (id, name, model_path, system_prompt,
                 inference_backend,
                 ctx_size, n_gpu_layers, n_threads, n_threads_http,
                 parallel, batch_size, ubatch_size, temperature, top_p,
                 max_tokens, flash_attn, extra_args_json, vllm_settings_json,
                 api_settings_json,
                 reasoning_enabled, memories_enabled, tools_enabled,
                 preferred_node_id, updated_at_ms)
            VALUES
                (:id,:name,:model_path,:system_prompt,
                 :inference_backend,
                 :ctx_size,:n_gpu_layers,:n_threads,:n_threads_http,
                 :parallel,:batch_size,:ubatch_size,:temperature,:top_p,
                 :max_tokens,:flash_attn,:extra_args_json,:vllm_settings_json,
                 :api_settings_json,
                 :reasoning,:memories,:tools,
                 :preferred_node_id,:now)
            ON CONFLICT(id) DO UPDATE SET
                name              = excluded.name,
                model_path        = excluded.model_path,
                system_prompt     = excluded.system_prompt,
                inference_backend = excluded.inference_backend,
                ctx_size          = excluded.ctx_size,
                n_gpu_layers      = excluded.n_gpu_layers,
                n_threads         = excluded.n_threads,
                n_threads_http    = excluded.n_threads_http,
                parallel          = excluded.parallel,
                batch_size        = excluded.batch_size,
                ubatch_size       = excluded.ubatch_size,
                temperature       = excluded.temperature,
                top_p             = excluded.top_p,
                max_tokens        = excluded.max_tokens,
                flash_attn        = excluded.flash_attn,
                extra_args_json   = excluded.extra_args_json,
                vllm_settings_json = excluded.vllm_settings_json,
                api_settings_json = excluded.api_settings_json,
                reasoning_enabled = excluded.reasoning_enabled,
                memories_enabled  = excluded.memories_enabled,
                tools_enabled     = excluded.tools_enabled,
                preferred_node_id = excluded.preferred_node_id,
                updated_at_ms     = excluded.updated_at_ms
        )");

        q.bind(":id",                cfg.id);
        q.bind(":name",              cfg.name);
        q.bind(":model_path",        cfg.model_path);
        q.bind(":system_prompt",     cfg.system_prompt);
        q.bind(":inference_backend", inference_backend);
        q.bind(":ctx_size",          s.ctx_size);
        q.bind(":n_gpu_layers",      s.n_gpu_layers);
        q.bind(":n_threads",         s.n_threads);
        q.bind(":n_threads_http",    s.n_threads_http);
        q.bind(":parallel",          s.parallel);
        q.bind(":batch_size",        s.batch_size);
        q.bind(":ubatch_size",       s.ubatch_size);
        q.bind(":temperature",       static_cast<double>(s.temperature));
        q.bind(":top_p",             static_cast<double>(s.top_p));
        q.bind(":max_tokens",        s.max_tokens);
        q.bind(":flash_attn",        s.flash_attn ? 1 : 0);
        q.bind(":extra_args_json",   extra_args_json);
        q.bind(":vllm_settings_json", vllm_settings_json);
        q.bind(":api_settings_json", api_settings_json);
        q.bind(":reasoning",         cfg.reasoning_enabled ? 1 : 0);
        q.bind(":memories",          cfg.memories_enabled ? 1 : 0);
        q.bind(":tools",             cfg.tools_enabled ? 1 : 0);
        q.bind(":preferred_node_id", cfg.preferred_node_id);
        q.bind(":now",               util::now_ms());
        q.exec();
    });
}

AgentConfig AgentDB::load_config() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM agent_config WHERE id=? LIMIT 1");
    q.bind(1, agent_id_);
    if (!q.executeStep()) return {};

    AgentConfig cfg;
    cfg.id           = q.getColumn("id").getText();
    cfg.name         = q.getColumn("name").getText();
    cfg.model_path   = q.getColumn("model_path").getText();
    cfg.system_prompt= q.getColumn("system_prompt").getText();
    cfg.inference_backend = normalize_inference_backend(
        q.getColumn("inference_backend").getText());

    auto& s = cfg.runtime_settings;
    s.ctx_size      = q.getColumn("ctx_size").getInt();
    s.n_gpu_layers  = q.getColumn("n_gpu_layers").getInt();
    s.n_threads     = q.getColumn("n_threads").getInt();
    s.n_threads_http = q.getColumn("n_threads_http").getInt();
    s.parallel      = q.getColumn("parallel").getInt();
    s.batch_size    = q.getColumn("batch_size").getInt();
    s.ubatch_size   = q.getColumn("ubatch_size").getInt();
    s.temperature   = static_cast<float>(q.getColumn("temperature").getDouble());
    s.top_p         = static_cast<float>(q.getColumn("top_p").getDouble());
    s.max_tokens    = q.getColumn("max_tokens").getInt();
    s.flash_attn    = q.getColumn("flash_attn").getInt() != 0;
    s.extra_args    = deserialize_strings(q.getColumn("extra_args_json").getText());
    cfg.vllm_settings =
        deserialize_vllm_settings(q.getColumn("vllm_settings_json").getText());
    cfg.api_settings =
        deserialize_api_settings(q.getColumn("api_settings_json").getText());

    cfg.reasoning_enabled  = q.getColumn("reasoning_enabled").getInt() != 0;
    cfg.memories_enabled   = q.getColumn("memories_enabled").getInt()  != 0;
    cfg.tools_enabled      = q.getColumn("tools_enabled").getInt()     != 0;
    cfg.preferred_node_id  = q.getColumn("preferred_node_id").getText();
    return cfg;
}

//
ConvId AgentDB::create_conversation(const std::string& title) {
    return create_conversation(title, "");
}

ConvId AgentDB::create_conversation(const std::string& title,
                                    const ConvId& parent_conv_id) {
    std::lock_guard g(mutex_);
    if (!parent_conv_id.empty()) {
        SQLite::Statement exists_q(*db_,
            "SELECT 1 FROM conversations WHERE id=? LIMIT 1");
        exists_q.bind(1, parent_conv_id);
        if (!exists_q.executeStep()) {
            return {};
        }
    }

    ConvId id = util::generate_uuid();
    int64_t now = util::now_ms();

    SQLite::Statement q(*db_, R"(
        INSERT INTO conversations(id, title, is_active, parent_conv_id, created_at_ms, updated_at_ms)
        VALUES (?,?,0,?,?,?)
    )");
    q.bind(1, id);
    q.bind(2, title.empty() ? "New Conversation" : title);
    if (parent_conv_id.empty()) q.bind(3);
    else                        q.bind(3, parent_conv_id);
    q.bind(4, now);
    q.bind(5, now);
    q.exec();
    return id;
}

void AgentDB::save_conversation(const Conversation& conv) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        INSERT INTO conversations
            (id, title, total_tokens, is_active, compaction_summary,
             parent_conv_id, created_at_ms, updated_at_ms)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            title              = excluded.title,
            total_tokens       = excluded.total_tokens,
            is_active          = excluded.is_active,
            compaction_summary = excluded.compaction_summary,
            parent_conv_id     = excluded.parent_conv_id,
            updated_at_ms      = excluded.updated_at_ms
    )");
    q.bind(1, conv.id);
    q.bind(2, conv.title);
    q.bind(3, conv.total_tokens);
    q.bind(4, conv.is_active ? 1 : 0);
    q.bind(5, conv.compaction_summary);
    if (conv.parent_conv_id.empty()) q.bind(6); // NULL
    else                             q.bind(6, conv.parent_conv_id);
    q.bind(7, conv.created_at_ms ? conv.created_at_ms : util::now_ms());
    q.bind(8, util::now_ms());
    q.exec();
}

std::optional<Conversation> AgentDB::load_conversation(const ConvId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM conversations WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return std::nullopt;

    Conversation conv;
    conv.id                 = q.getColumn("id").getText();
    conv.title              = q.getColumn("title").getText();
    conv.total_tokens       = q.getColumn("total_tokens").getInt();
    conv.is_active          = q.getColumn("is_active").getInt() != 0;
    conv.compaction_summary = q.getColumn("compaction_summary").getText();
    auto pc = q.getColumn("parent_conv_id");
    if (!pc.isNull()) conv.parent_conv_id = pc.getText();
    conv.created_at_ms      = q.getColumn("created_at_ms").getInt64();
    conv.updated_at_ms      = q.getColumn("updated_at_ms").getInt64();

    // Load messages
    SQLite::Statement mq(*db_,
        "SELECT * FROM messages WHERE conversation_id=? ORDER BY sequence_num");
    mq.bind(1, id);
    while (mq.executeStep()) conv.messages.push_back(row_to_message(mq));

    return conv;
}

std::vector<Conversation> AgentDB::list_conversations() const {
    std::lock_guard g(mutex_);
    // Returns metadata only (no messages) for list views.
    SQLite::Statement q(*db_,
        "SELECT * FROM conversations ORDER BY created_at_ms DESC");
    std::vector<Conversation> out;
    while (q.executeStep()) {
        Conversation c;
        c.id                 = q.getColumn("id").getText();
        c.title              = q.getColumn("title").getText();
        c.total_tokens       = q.getColumn("total_tokens").getInt();
        c.is_active          = q.getColumn("is_active").getInt() != 0;
        c.compaction_summary = q.getColumn("compaction_summary").getText();
        auto pc = q.getColumn("parent_conv_id");
        if (!pc.isNull()) c.parent_conv_id = pc.getText();
        c.created_at_ms      = q.getColumn("created_at_ms").getInt64();
        c.updated_at_ms      = q.getColumn("updated_at_ms").getInt64();
        out.push_back(c);
    }
    return out;
}

bool AgentDB::conversation_exists(const ConvId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT 1 FROM conversations WHERE id=? LIMIT 1");
    q.bind(1, id);
    return q.executeStep();
}

bool AgentDB::is_conversation_active(const ConvId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT is_active FROM conversations WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return false;
    return q.getColumn(0).getInt() != 0;
}

void AgentDB::rename_conversation(const ConvId& id, const std::string& title) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "UPDATE conversations SET title=?, updated_at_ms=? WHERE id=?");
    q.bind(1, title.empty() ? "Untitled conversation" : title);
    q.bind(2, util::now_ms());
    q.bind(3, id);
    q.exec();
}

void AgentDB::delete_conversation(const ConvId& id) {
    std::lock_guard g(mutex_);
    // FK CASCADE removes messages automatically.
    SQLite::Statement q(*db_, "DELETE FROM conversations WHERE id=?");
    q.bind(1, id);
    q.exec();
}

void AgentDB::set_active_conversation(const ConvId& id) {
    std::lock_guard g(mutex_);
    SQLite::Transaction tx(*db_);
    db_->exec("UPDATE conversations SET is_active=0");
    SQLite::Statement q(*db_,
        "UPDATE conversations SET is_active=1, updated_at_ms=? WHERE id=?");
    q.bind(1, util::now_ms());
    q.bind(2, id);
    q.exec();
    tx.commit();
}

std::optional<ConvId> AgentDB::get_active_conversation_id() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT id FROM conversations WHERE is_active=1 LIMIT 1");
    if (!q.executeStep()) return std::nullopt;
    return std::string(q.getColumn(0).getText());
}

//
void AgentDB::append_message(const ConvId& conv_id,
                              const Message& msg,
                              int sequence_num) {
    std::lock_guard g(mutex_);
    int64_t ts = msg.timestamp_ms ? msg.timestamp_ms : util::now_ms();

    SQLite::Transaction tx(*db_);

    SQLite::Statement q(*db_, R"(
        INSERT INTO messages
            (conversation_id, role, content, tool_calls_json,
             tool_call_id, thinking_text, token_count, sequence_num,
             trace_events_json, timestamp_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    )");
    q.bind(1, conv_id);
    q.bind(2, to_string(msg.role));
    q.bind(3, msg.content);
    q.bind(4, serialize(msg.tool_calls));
    q.bind(5, msg.tool_call_id);
    q.bind(6, msg.thinking_text);
    q.bind(7, msg.token_count);
    q.bind(8, sequence_num);
    q.bind(9, serialize(msg.trace_events));
    q.bind(10, ts);
    q.exec();

    // Update running token total on the conversation.
    SQLite::Statement uq(*db_, R"(
        UPDATE conversations
        SET total_tokens = total_tokens + ?,
            updated_at_ms = ?
        WHERE id = ?
    )");
    uq.bind(1, msg.token_count);
    uq.bind(2, ts);
    uq.bind(3, conv_id);
    uq.exec();

    tx.commit();
}

std::vector<Message> AgentDB::load_messages(const ConvId& conv_id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM messages WHERE conversation_id=? ORDER BY sequence_num");
    q.bind(1, conv_id);
    std::vector<Message> out;
    while (q.executeStep()) out.push_back(row_to_message(q));
    return out;
}

int AgentDB::get_total_tokens(const ConvId& conv_id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT total_tokens FROM conversations WHERE id=? LIMIT 1");
    q.bind(1, conv_id);
    if (!q.executeStep()) return 0;
    return q.getColumn(0).getInt();
}

//
void AgentDB::add_memory(const Memory& mem) {
    std::lock_guard g(mutex_);
    int64_t now = util::now_ms();
    SQLite::Statement q(*db_, R"(
        INSERT INTO global_memories(id, content, source_conv_id, importance, created_at_ms, updated_at_ms)
        VALUES (?,?,?,?,?,?)
    )");
    q.bind(1, mem.id.empty() ? util::generate_uuid() : mem.id);
    q.bind(2, mem.content);
    if (mem.source_conv_id.empty()) q.bind(3); // NULL
    else                            q.bind(3, mem.source_conv_id);
    q.bind(4, static_cast<double>(mem.importance));
    q.bind(5, mem.created_at_ms ? mem.created_at_ms : now);
    q.bind(6, mem.updated_at_ms ? mem.updated_at_ms : now);
    q.exec();
}

void AgentDB::update_memory(const Memory& mem) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        UPDATE global_memories SET content=?, importance=?, updated_at_ms=? WHERE id=?
    )");
    q.bind(1, mem.content);
    q.bind(2, static_cast<double>(mem.importance));
    q.bind(3, util::now_ms());
    q.bind(4, mem.id);
    q.exec();
}

void AgentDB::delete_memory(const MemoryId& id) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "DELETE FROM global_memories WHERE id=?");
    q.bind(1, id);
    q.exec();
}

std::optional<Memory> AgentDB::get_memory(const MemoryId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM global_memories WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return std::nullopt;

    Memory m;
    m.id             = q.getColumn("id").getText();
    m.agent_id       = agent_id_;
    m.content        = q.getColumn("content").getText();
    auto sc = q.getColumn("source_conv_id");
    if (!sc.isNull()) m.source_conv_id = sc.getText();
    m.importance     = static_cast<float>(q.getColumn("importance").getDouble());
    m.created_at_ms  = q.getColumn("created_at_ms").getInt64();
    m.updated_at_ms  = q.getColumn("updated_at_ms").getInt64();
    return m;
}

std::vector<Memory> AgentDB::list_memories() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM global_memories ORDER BY importance DESC, created_at_ms DESC");
    std::vector<Memory> out;
    while (q.executeStep()) {
        Memory m;
        m.id             = q.getColumn("id").getText();
        m.agent_id       = agent_id_;
        m.content        = q.getColumn("content").getText();
        auto sc = q.getColumn("source_conv_id");
        if (!sc.isNull()) m.source_conv_id = sc.getText();
        m.importance     = static_cast<float>(q.getColumn("importance").getDouble());
        m.created_at_ms  = q.getColumn("created_at_ms").getInt64();
        m.updated_at_ms  = q.getColumn("updated_at_ms").getInt64();
        out.push_back(m);
    }
    return out;
}

std::vector<Memory> AgentDB::search_memories(const std::string& query, int limit) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        SELECT * FROM global_memories
        WHERE content LIKE ?
        ORDER BY importance DESC, created_at_ms DESC
        LIMIT ?
    )");
    std::string pattern = "%" + query + "%";
    q.bind(1, pattern);
    q.bind(2, limit);
    std::vector<Memory> out;
    while (q.executeStep()) {
        Memory m;
        m.id             = q.getColumn("id").getText();
        m.agent_id       = agent_id_;
        m.content        = q.getColumn("content").getText();
        auto sc = q.getColumn("source_conv_id");
        if (!sc.isNull()) m.source_conv_id = sc.getText();
        m.importance     = static_cast<float>(q.getColumn("importance").getDouble());
        m.created_at_ms  = q.getColumn("created_at_ms").getInt64();
        m.updated_at_ms  = q.getColumn("updated_at_ms").getInt64();
        out.push_back(m);
    }
    return out;
}

//
void AgentDB::add_local_memory(const LocalMemory& mem) {
    std::lock_guard g(mutex_);
    int64_t now = util::now_ms();
    SQLite::Statement q(*db_, R"(
        INSERT INTO local_memories(id, conversation_id, content, created_at_ms, updated_at_ms)
        VALUES (?,?,?,?,?)
    )");
    q.bind(1, mem.id.empty() ? util::generate_uuid() : mem.id);
    q.bind(2, mem.conversation_id);
    q.bind(3, mem.content);
    q.bind(4, mem.created_at_ms ? mem.created_at_ms : now);
    q.bind(5, mem.updated_at_ms ? mem.updated_at_ms : now);
    q.exec();
}

void AgentDB::update_local_memory(const LocalMemory& mem) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        UPDATE local_memories SET content=?, updated_at_ms=? WHERE id=?
    )");
    q.bind(1, mem.content);
    q.bind(2, util::now_ms());
    q.bind(3, mem.id);
    q.exec();
}

void AgentDB::delete_local_memory(const std::string& id) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "DELETE FROM local_memories WHERE id=?");
    q.bind(1, id);
    q.exec();
}

std::optional<LocalMemory> AgentDB::get_local_memory(const std::string& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM local_memories WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return std::nullopt;

    LocalMemory m;
    m.id              = q.getColumn("id").getText();
    m.conversation_id = q.getColumn("conversation_id").getText();
    m.content         = q.getColumn("content").getText();
    m.created_at_ms   = q.getColumn("created_at_ms").getInt64();
    m.updated_at_ms   = q.getColumn("updated_at_ms").getInt64();
    return m;
}

std::vector<LocalMemory> AgentDB::list_local_memories(const ConvId& conv_id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM local_memories WHERE conversation_id=? ORDER BY created_at_ms");
    q.bind(1, conv_id);
    std::vector<LocalMemory> out;
    while (q.executeStep()) {
        LocalMemory m;
        m.id              = q.getColumn("id").getText();
        m.conversation_id = q.getColumn("conversation_id").getText();
        m.content         = q.getColumn("content").getText();
        m.created_at_ms   = q.getColumn("created_at_ms").getInt64();
        m.updated_at_ms   = q.getColumn("updated_at_ms").getInt64();
        out.push_back(m);
    }
    return out;
}

void AgentDB::transfer_local_memories(const ConvId& from, const ConvId& to) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        UPDATE local_memories SET conversation_id=?, updated_at_ms=?
        WHERE conversation_id=?
    )");
    q.bind(1, to);
    q.bind(2, util::now_ms());
    q.bind(3, from);
    q.exec();
}

void AgentDB::save_voice_proposal(const VoiceDesignProposal& proposal) {
    std::lock_guard g(mutex_);
    const int64_t now = util::now_ms();
    SQLite::Statement q(*db_, R"(
        INSERT INTO agent_voice_proposals
            (id, display_name, language, voice_description, sample_text,
             rationale, status, provider, voice_design_model_id, clone_model_id,
             preview_audio_path, voice_clone_prompt_path, error,
             created_at_ms, updated_at_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            display_name = excluded.display_name,
            language = excluded.language,
            voice_description = excluded.voice_description,
            sample_text = excluded.sample_text,
            rationale = excluded.rationale,
            status = excluded.status,
            provider = excluded.provider,
            voice_design_model_id = excluded.voice_design_model_id,
            clone_model_id = excluded.clone_model_id,
            preview_audio_path = excluded.preview_audio_path,
            voice_clone_prompt_path = excluded.voice_clone_prompt_path,
            error = excluded.error,
            updated_at_ms = excluded.updated_at_ms
    )");
    q.bind(1, proposal.id.empty() ? util::generate_uuid() : proposal.id);
    q.bind(2, proposal.display_name);
    q.bind(3, proposal.language.empty() ? "Auto" : proposal.language);
    q.bind(4, proposal.voice_description);
    q.bind(5, proposal.sample_text);
    q.bind(6, proposal.rationale);
    q.bind(7, proposal.status.empty() ? "pending" : proposal.status);
    q.bind(8, proposal.provider.empty() ? "qwen3-tts" : proposal.provider);
    q.bind(9, proposal.voice_design_model_id);
    q.bind(10, proposal.clone_model_id);
    q.bind(11, proposal.preview_audio_path);
    q.bind(12, proposal.voice_clone_prompt_path);
    q.bind(13, proposal.error);
    q.bind(14, proposal.created_at_ms ? proposal.created_at_ms : now);
    q.bind(15, proposal.updated_at_ms ? proposal.updated_at_ms : now);
    q.exec();
}

std::optional<VoiceDesignProposal> AgentDB::get_voice_proposal(const VoiceProposalId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM agent_voice_proposals WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return std::nullopt;
    return row_to_voice_proposal(q, agent_id_);
}

std::vector<VoiceDesignProposal> AgentDB::list_voice_proposals() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM agent_voice_proposals ORDER BY created_at_ms DESC");
    std::vector<VoiceDesignProposal> out;
    while (q.executeStep()) out.push_back(row_to_voice_proposal(q, agent_id_));
    return out;
}

void AgentDB::update_voice_proposal_status(const VoiceProposalId& id,
                                           const std::string& status,
                                           const std::string& error) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        UPDATE agent_voice_proposals
        SET status=?, error=?, updated_at_ms=?
        WHERE id=?
    )");
    q.bind(1, status);
    q.bind(2, error);
    q.bind(3, util::now_ms());
    q.bind(4, id);
    q.exec();
}

void AgentDB::save_voice_profile(const AgentVoiceProfile& profile) {
    std::lock_guard g(mutex_);
    const int64_t now = util::now_ms();
    SQLite::Transaction tx(*db_);
    if (profile.active) {
        SQLite::Statement clear(*db_,
            "UPDATE agent_voice_profiles SET active=0, updated_at_ms=? WHERE active=1");
        clear.bind(1, now);
        clear.exec();
    }
    SQLite::Statement q(*db_, R"(
        INSERT INTO agent_voice_profiles
            (id, display_name, language, voice_description, sample_text,
             rationale, provider, voice_design_model_id, clone_model_id,
             reference_audio_path, voice_clone_prompt_path,
             approved_from_proposal_id, active, created_at_ms, updated_at_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            display_name = excluded.display_name,
            language = excluded.language,
            voice_description = excluded.voice_description,
            sample_text = excluded.sample_text,
            rationale = excluded.rationale,
            provider = excluded.provider,
            voice_design_model_id = excluded.voice_design_model_id,
            clone_model_id = excluded.clone_model_id,
            reference_audio_path = excluded.reference_audio_path,
            voice_clone_prompt_path = excluded.voice_clone_prompt_path,
            approved_from_proposal_id = excluded.approved_from_proposal_id,
            active = excluded.active,
            updated_at_ms = excluded.updated_at_ms
    )");
    q.bind(1, profile.id.empty() ? util::generate_uuid() : profile.id);
    q.bind(2, profile.display_name);
    q.bind(3, profile.language.empty() ? "Auto" : profile.language);
    q.bind(4, profile.voice_description);
    q.bind(5, profile.sample_text);
    q.bind(6, profile.rationale);
    q.bind(7, profile.provider.empty() ? "qwen3-tts" : profile.provider);
    q.bind(8, profile.voice_design_model_id);
    q.bind(9, profile.clone_model_id);
    q.bind(10, profile.reference_audio_path);
    q.bind(11, profile.voice_clone_prompt_path);
    q.bind(12, profile.approved_from_proposal_id);
    q.bind(13, profile.active ? 1 : 0);
    q.bind(14, profile.created_at_ms ? profile.created_at_ms : now);
    q.bind(15, profile.updated_at_ms ? profile.updated_at_ms : now);
    q.exec();
    tx.commit();
}

std::optional<AgentVoiceProfile> AgentDB::get_voice_profile(const VoiceProfileId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM agent_voice_profiles WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return std::nullopt;
    return row_to_voice_profile(q, agent_id_);
}

std::optional<AgentVoiceProfile> AgentDB::get_active_voice_profile() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        SELECT * FROM agent_voice_profiles
        WHERE active=1
        ORDER BY updated_at_ms DESC
        LIMIT 1
    )");
    if (!q.executeStep()) return std::nullopt;
    return row_to_voice_profile(q, agent_id_);
}

std::vector<AgentVoiceProfile> AgentDB::list_voice_profiles() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM agent_voice_profiles ORDER BY active DESC, updated_at_ms DESC");
    std::vector<AgentVoiceProfile> out;
    while (q.executeStep()) out.push_back(row_to_voice_profile(q, agent_id_));
    return out;
}

void AgentDB::set_active_voice_profile(const VoiceProfileId& id) {
    std::lock_guard g(mutex_);
    const int64_t now = util::now_ms();
    SQLite::Transaction tx(*db_);
    SQLite::Statement clear(*db_,
        "UPDATE agent_voice_profiles SET active=0, updated_at_ms=? WHERE active=1");
    clear.bind(1, now);
    clear.exec();
    SQLite::Statement set(*db_,
        "UPDATE agent_voice_profiles SET active=1, updated_at_ms=? WHERE id=?");
    set.bind(1, now);
    set.bind(2, id);
    set.exec();
    tx.commit();
}

void AgentDB::save_tts_cache_entry(const TtsSynthesisResult& entry) {
    std::lock_guard g(mutex_);
    const int64_t now = util::now_ms();
    SQLite::Statement q(*db_, R"(
        INSERT INTO tts_audio_cache
            (id, voice_profile_id, conversation_id, message_index, text_hash,
             audio_path, mime_type, format, sample_rate, duration_ms,
             created_at_ms, expires_at_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            voice_profile_id = excluded.voice_profile_id,
            conversation_id = excluded.conversation_id,
            message_index = excluded.message_index,
            text_hash = excluded.text_hash,
            audio_path = excluded.audio_path,
            mime_type = excluded.mime_type,
            format = excluded.format,
            sample_rate = excluded.sample_rate,
            duration_ms = excluded.duration_ms,
            created_at_ms = excluded.created_at_ms,
            expires_at_ms = excluded.expires_at_ms
    )");
    q.bind(1, entry.cache_id.empty() ? util::generate_uuid() : entry.cache_id);
    q.bind(2, entry.voice_profile_id);
    q.bind(3, entry.conversation_id);
    q.bind(4, entry.message_index);
    q.bind(5, entry.text_hash);
    q.bind(6, entry.audio_path);
    q.bind(7, entry.mime_type.empty() ? "audio/wav" : entry.mime_type);
    q.bind(8, entry.format.empty() ? "wav" : entry.format);
    q.bind(9, entry.sample_rate);
    q.bind(10, entry.duration_ms);
    q.bind(11, entry.created_at_ms ? entry.created_at_ms : now);
    q.bind(12, entry.expires_at_ms);
    q.exec();
}

std::optional<TtsSynthesisResult> AgentDB::find_tts_cache_entry(
    const VoiceProfileId& profile_id,
    const std::string& text_hash,
    const ConvId& conversation_id,
    int message_index) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        SELECT * FROM tts_audio_cache
        WHERE voice_profile_id=?
          AND text_hash=?
          AND conversation_id=?
          AND message_index=?
          AND expires_at_ms>?
        ORDER BY created_at_ms DESC
        LIMIT 1
    )");
    q.bind(1, profile_id);
    q.bind(2, text_hash);
    q.bind(3, conversation_id);
    q.bind(4, message_index);
    q.bind(5, util::now_ms());
    if (!q.executeStep()) return std::nullopt;
    auto result = row_to_tts_cache(q, agent_id_);
    result.cached = true;
    return result;
}

std::optional<TtsSynthesisResult> AgentDB::get_tts_cache_entry(const TtsCacheId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM tts_audio_cache WHERE id=? AND expires_at_ms>? LIMIT 1");
    q.bind(1, id);
    q.bind(2, util::now_ms());
    if (!q.executeStep()) return std::nullopt;
    auto result = row_to_tts_cache(q, agent_id_);
    result.cached = true;
    return result;
}

std::vector<TtsSynthesisResult> AgentDB::delete_expired_tts_cache_entries(int64_t now_ms) {
    std::lock_guard g(mutex_);
    std::vector<TtsSynthesisResult> expired;
    {
        SQLite::Statement q(*db_,
            "SELECT * FROM tts_audio_cache WHERE expires_at_ms<=?");
        q.bind(1, now_ms);
        while (q.executeStep()) expired.push_back(row_to_tts_cache(q, agent_id_));
    }
    SQLite::Statement del(*db_,
        "DELETE FROM tts_audio_cache WHERE expires_at_ms<=?");
    del.bind(1, now_ms);
    del.exec();
    return expired;
}

} // namespace mm
