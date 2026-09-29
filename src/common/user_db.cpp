#include "common/user_db.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <SQLiteCpp/SQLiteCpp.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <thread>

namespace mm {

namespace {

// nlohmann ADL picks up mm::to_json / mm::from_json.
std::string serialize(const std::vector<ToolCall>& calls) { return nlohmann::json(calls).dump(); }
std::vector<ToolCall> deserialize_tool_calls(const std::string& s) {
    try { return nlohmann::json::parse(s).get<std::vector<ToolCall>>(); }
    catch (const std::exception& e) { MM_DEBUG("deserialize_tool_calls: {}", e.what()); return {}; }
}

std::string serialize(const std::vector<TraceEvent>& events) { return nlohmann::json(events).dump(); }
std::vector<TraceEvent> deserialize_trace_events(const std::string& s) {
    try { return nlohmann::json::parse(s).get<std::vector<TraceEvent>>(); }
    catch (const std::exception& e) { MM_DEBUG("deserialize_trace_events: {}", e.what()); return {}; }
}

std::string serialize(const SamplingOverrides& o) { return nlohmann::json(o).dump(); }
SamplingOverrides deserialize_settings(const std::string& s) {
    try { return nlohmann::json::parse(s).get<SamplingOverrides>(); }
    catch (const std::exception& e) { MM_DEBUG("deserialize_settings: {}", e.what()); return {}; }
}

bool is_transient_sqlite_lock(const SQLite::Exception& e) {
    const std::string msg = util::to_lower(e.what());
    return msg.find("locked") != std::string::npos || msg.find("busy") != std::string::npos;
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
            if (!is_transient_sqlite_lock(e) || attempt == kMaxAttempts) throw;
            MM_WARN("UserDB {} hit a transient SQLite lock (attempt {}/{}): {}",
                    operation, attempt, kMaxAttempts, e.what());
            std::this_thread::sleep_for(kRetryDelay);
        }
    }
}

Message row_to_message(SQLite::Statement& q) {
    Message m;
    m.storage_id    = q.getColumn("id").getInt64();
    m.sequence_num  = q.getColumn("sequence_num").getInt();
    m.role          = message_role_from_string(q.getColumn("role").getText());
    m.content       = q.getColumn("content").getText();
    m.tool_calls    = deserialize_tool_calls(q.getColumn("tool_calls_json").getText());
    m.tool_call_id  = q.getColumn("tool_call_id").getText();
    m.thinking_text = q.getColumn("thinking_text").getText();
    m.token_count   = q.getColumn("token_count").getInt();
    m.model_id      = q.getColumn("model_id").getText();
    m.timestamp_ms  = q.getColumn("timestamp_ms").getInt64();
    m.trace_events  = deserialize_trace_events(q.getColumn("trace_events_json").getText());
    return m;
}

void load_message_parts(SQLite::Database& db, Message& m) {
    if (m.storage_id <= 0) return;
    SQLite::Statement q(db, R"(
        SELECT type, text, attachment_id
        FROM message_content_parts
        WHERE message_id=? ORDER BY ordinal
    )");
    q.bind(1, m.storage_id);
    while (q.executeStep()) {
        MessageContentPart part;
        part.type = q.getColumn("type").getText();
        part.text = q.getColumn("text").getText();
        part.attachment_id = q.getColumn("attachment_id").getText();
        m.content_parts.push_back(std::move(part));
    }
}

Conversation row_to_conversation(SQLite::Statement& q) {
    Conversation c;
    c.id                  = q.getColumn("id").getText();
    c.title               = q.getColumn("title").getText();
    c.total_tokens        = q.getColumn("total_tokens").getInt();
    c.is_active           = q.getColumn("is_active").getInt() != 0;
    c.model_id            = q.getColumn("model_id").getText();
    c.system_prompt       = q.getColumn("system_prompt").getText();
    c.settings            = deserialize_settings(q.getColumn("settings_json").getText());
    c.summary             = q.getColumn("summary").getText();
    c.summary_through_seq = q.getColumn("summary_through_seq").getInt();
    c.summary_model_id    = q.getColumn("summary_model_id").getText();
    const auto parent = q.getColumn("parent_conv_id");
    if (!parent.isNull()) c.parent_conv_id = parent.getText();
    c.created_at_ms       = q.getColumn("created_at_ms").getInt64();
    c.updated_at_ms       = q.getColumn("updated_at_ms").getInt64();
    return c;
}

ImageAttachment row_to_attachment(SQLite::Statement& q) {
    ImageAttachment a;
    a.id = q.getColumn("id").getText();
    a.original_filename = q.getColumn("original_filename").getText();
    a.mime_type = q.getColumn("mime_type").getText();
    a.relative_path = q.getColumn("relative_path").getText();
    a.size_bytes = q.getColumn("size_bytes").getInt64();
    a.created_at_ms = q.getColumn("created_at_ms").getInt64();
    a.expires_at_ms = q.getColumn("expires_at_ms").getInt64();
    return a;
}

Memory row_to_memory(SQLite::Statement& q) {
    Memory m;
    m.id            = q.getColumn("id").getText();
    m.content       = q.getColumn("content").getText();
    const auto source = q.getColumn("source_conv_id");
    if (!source.isNull()) m.source_conv_id = source.getText();
    m.importance    = static_cast<float>(q.getColumn("importance").getDouble());
    m.created_at_ms = q.getColumn("created_at_ms").getInt64();
    m.updated_at_ms = q.getColumn("updated_at_ms").getInt64();
    return m;
}

VoiceProfile row_to_voice_profile(SQLite::Statement& q, const UserId& user_id) {
    VoiceProfile p;
    p.id                        = q.getColumn("id").getText();
    p.user_id                   = user_id;
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

VoiceDesignProposal row_to_voice_proposal(SQLite::Statement& q, const UserId& user_id) {
    VoiceDesignProposal p;
    p.id                      = q.getColumn("id").getText();
    p.user_id                 = user_id;
    p.display_name            = q.getColumn("display_name").getText();
    p.language                = q.getColumn("language").getText();
    p.voice_description       = q.getColumn("voice_description").getText();
    p.sample_text             = q.getColumn("sample_text").getText();
    p.rationale               = q.getColumn("rationale").getText();
    p.status                  = q.getColumn("status").getText();
    p.provider                = q.getColumn("provider").getText();
    p.voice_design_model_id   = q.getColumn("voice_design_model_id").getText();
    p.clone_model_id          = q.getColumn("clone_model_id").getText();
    p.preview_audio_path      = q.getColumn("preview_audio_path").getText();
    p.voice_clone_prompt_path = q.getColumn("voice_clone_prompt_path").getText();
    p.error                   = q.getColumn("error").getText();
    p.created_at_ms           = q.getColumn("created_at_ms").getInt64();
    p.updated_at_ms           = q.getColumn("updated_at_ms").getInt64();
    return p;
}

TtsSynthesisResult row_to_tts_cache(SQLite::Statement& q, const UserId& user_id) {
    TtsSynthesisResult r;
    r.cache_id         = q.getColumn("id").getText();
    r.user_id          = user_id;
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

LocalMemory row_to_local_memory(SQLite::Statement& q) {
    LocalMemory m;
    m.id              = q.getColumn("id").getText();
    m.conversation_id = q.getColumn("conversation_id").getText();
    m.content         = q.getColumn("content").getText();
    m.created_at_ms   = q.getColumn("created_at_ms").getInt64();
    m.updated_at_ms   = q.getColumn("updated_at_ms").getInt64();
    return m;
}

} // namespace

UserDB::UserDB(const UserId& user_id, const std::string& data_dir) : user_id_(user_id) {
    namespace fs = std::filesystem;
    // The id becomes a path component; refuse anything that could climb out of
    // users/ or name a device, rather than trust every caller to have checked.
    if (!util::is_valid_user_id(user_id))
        throw std::invalid_argument("invalid user id '" + user_id + "'");
    const fs::path dir = fs::path(data_dir) / "users" / user_id;
    fs::create_directories(dir);
    user_dir_ = dir.lexically_normal().string();

    db_ = std::make_unique<SQLite::Database>((dir / "user.db").string(),
                                             SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);

    run_with_sqlite_retry("init_pragmas", [&] {
        db_->exec("PRAGMA journal_mode=WAL");
        db_->exec("PRAGMA foreign_keys=ON");
        db_->exec("PRAGMA synchronous=NORMAL");
        db_->exec("PRAGMA cache_size=-8000"); // 8 MB page cache
        db_->exec("PRAGMA busy_timeout=5000");
    });
    run_with_sqlite_retry("run_migrations", [&] { run_migrations(); });

    // Startup reconciliation, scoped to the managed attachments directory:
    // expire abandoned uploads, and remove files no row describes (including
    // interrupted .part uploads).
    delete_expired_unreferenced_attachments(util::now_ms());
    const fs::path attachments_dir = fs::path(user_dir_) / "attachments";
    std::error_code dir_ec;
    fs::create_directories(attachments_dir, dir_ec);
    std::set<std::string> known_paths;
    {
        std::lock_guard g(mutex_);
        SQLite::Statement q(*db_, "SELECT relative_path FROM attachments");
        while (q.executeStep())
            known_paths.insert(fs::path(q.getColumn(0).getText()).lexically_normal().generic_string());
    }
    for (auto it = fs::recursive_directory_iterator(attachments_dir, dir_ec);
         !dir_ec && it != fs::recursive_directory_iterator(); it.increment(dir_ec)) {
        std::error_code file_ec;
        if (!it->is_regular_file(file_ec)) continue;
        const auto rel =
            fs::relative(it->path(), fs::path(user_dir_), file_ec).lexically_normal().generic_string();
        if (!file_ec && !known_paths.count(rel)) fs::remove(it->path(), file_ec);
    }
}

UserDB::~UserDB() = default;

void UserDB::run_migrations() {
    db_->exec(R"(
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INTEGER NOT NULL PRIMARY KEY,
            applied_at INTEGER NOT NULL
                DEFAULT (CAST(strftime('%s','now') AS INTEGER)*1000)
        )
    )");
    const auto has_version = [this](int version) {
        SQLite::Statement q(*db_, "SELECT 1 FROM schema_migrations WHERE version=?");
        q.bind(1, version);
        return q.executeStep();
    };

    if (!has_version(1)) {
        SQLite::Transaction tx(*db_);
        db_->exec(R"(
            CREATE TABLE user_profile (
                id                  TEXT    NOT NULL PRIMARY KEY,
                display_name        TEXT    NOT NULL DEFAULT '',
                custom_instructions TEXT    NOT NULL DEFAULT '',
                default_model       TEXT    NOT NULL DEFAULT '',
                memories_enabled    INTEGER NOT NULL DEFAULT 1,
                tools_enabled       INTEGER NOT NULL DEFAULT 0,
                created_at_ms       INTEGER NOT NULL,
                updated_at_ms       INTEGER NOT NULL
            )
        )");
        db_->exec(R"(
            CREATE TABLE conversations (
                id                  TEXT    NOT NULL PRIMARY KEY,
                title               TEXT    NOT NULL DEFAULT '',
                total_tokens        INTEGER NOT NULL DEFAULT 0,
                is_active           INTEGER NOT NULL DEFAULT 0,
                model_id            TEXT    NOT NULL DEFAULT '',
                system_prompt       TEXT    NOT NULL DEFAULT '',
                settings_json       TEXT    NOT NULL DEFAULT '{}',
                summary             TEXT    NOT NULL DEFAULT '',
                summary_through_seq INTEGER NOT NULL DEFAULT -1,
                summary_model_id    TEXT    NOT NULL DEFAULT '',
                parent_conv_id      TEXT
                    REFERENCES conversations(id) ON DELETE SET NULL,
                created_at_ms       INTEGER NOT NULL,
                updated_at_ms       INTEGER NOT NULL
            )
        )");
        // At most one active conversation. A partial unique index, so the store
        // refuses a second one instead of every writer having to clear the
        // others first — and two first turns racing cannot both create one.
        db_->exec("CREATE UNIQUE INDEX idx_conv_one_active ON conversations(is_active) "
                  "WHERE is_active = 1");
        db_->exec("CREATE INDEX idx_conv_created ON conversations(created_at_ms DESC)");
        db_->exec(R"(
            CREATE TABLE messages (
                id                INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                conversation_id   TEXT    NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                sequence_num      INTEGER NOT NULL,
                role              TEXT    NOT NULL,
                content           TEXT    NOT NULL DEFAULT '',
                tool_calls_json   TEXT    NOT NULL DEFAULT '[]',
                tool_call_id      TEXT    NOT NULL DEFAULT '',
                thinking_text     TEXT    NOT NULL DEFAULT '',
                token_count       INTEGER NOT NULL DEFAULT 0,
                model_id          TEXT    NOT NULL DEFAULT '',
                trace_events_json TEXT    NOT NULL DEFAULT '[]',
                timestamp_ms      INTEGER NOT NULL,
                UNIQUE (conversation_id, sequence_num)
            )
        )");
        db_->exec(R"(
            CREATE TABLE attachments (
                id                TEXT    NOT NULL PRIMARY KEY,
                original_filename TEXT    NOT NULL DEFAULT '',
                mime_type         TEXT    NOT NULL,
                relative_path     TEXT    NOT NULL,
                size_bytes        INTEGER NOT NULL,
                created_at_ms     INTEGER NOT NULL,
                expires_at_ms     INTEGER NOT NULL DEFAULT 0
            )
        )");
        db_->exec(R"(
            CREATE TABLE message_content_parts (
                message_id    INTEGER NOT NULL
                    REFERENCES messages(id) ON DELETE CASCADE,
                ordinal       INTEGER NOT NULL,
                type          TEXT    NOT NULL,
                text          TEXT    NOT NULL DEFAULT '',
                attachment_id TEXT
                    REFERENCES attachments(id) ON DELETE RESTRICT,
                PRIMARY KEY (message_id, ordinal)
            )
        )");
        db_->exec("CREATE INDEX idx_parts_attachment ON message_content_parts(attachment_id)");
        db_->exec("CREATE INDEX idx_attachments_expiry ON attachments(expires_at_ms)");
        db_->exec(R"(
            CREATE TABLE local_memories (
                id              TEXT    NOT NULL PRIMARY KEY,
                conversation_id TEXT    NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                content         TEXT    NOT NULL,
                created_at_ms   INTEGER NOT NULL,
                updated_at_ms   INTEGER NOT NULL
            )
        )");
        db_->exec("CREATE INDEX idx_local_mem_conv ON local_memories(conversation_id)");
        db_->exec(R"(
            CREATE TABLE global_memories (
                id             TEXT    NOT NULL PRIMARY KEY,
                content        TEXT    NOT NULL,
                source_conv_id TEXT
                    REFERENCES conversations(id) ON DELETE SET NULL,
                importance     REAL    NOT NULL DEFAULT 0.5,
                created_at_ms  INTEGER NOT NULL,
                updated_at_ms  INTEGER NOT NULL
            )
        )");
        db_->exec("CREATE INDEX idx_global_mem_rank "
                  "ON global_memories(importance DESC, created_at_ms DESC)");
        db_->exec(R"(
            CREATE TABLE voice_proposals (
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
            CREATE TABLE voice_profiles (
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
            CREATE TABLE tts_audio_cache (
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
        db_->exec("CREATE INDEX idx_voice_proposals_status "
                  "ON voice_proposals(status, created_at_ms DESC)");
        db_->exec("CREATE INDEX idx_voice_profiles_active "
                  "ON voice_profiles(active, updated_at_ms DESC)");
        db_->exec("CREATE INDEX idx_tts_cache_lookup "
                  "ON tts_audio_cache(voice_profile_id, text_hash, conversation_id, message_index)");
        db_->exec("CREATE INDEX idx_tts_cache_expiry ON tts_audio_cache(expires_at_ms)");
        db_->exec("INSERT INTO schema_migrations(version) VALUES (1)");
        tx.commit();
    }
}

// ── Profile ───────────────────────────────────────────────────────────────────

void UserDB::save_profile(const UserProfile& profile) {
    if (profile.id != user_id_)
        throw std::invalid_argument("profile '" + profile.id + "' saved to the store of '" +
                                    user_id_ + "'");
    std::lock_guard g(mutex_);
    const std::int64_t now = util::now_ms();
    run_with_sqlite_retry("save_profile", [&] {
        SQLite::Statement q(*db_, R"(
            INSERT INTO user_profile
                (id, display_name, custom_instructions, default_model,
                 memories_enabled, tools_enabled, created_at_ms, updated_at_ms)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                display_name        = excluded.display_name,
                custom_instructions = excluded.custom_instructions,
                default_model       = excluded.default_model,
                memories_enabled    = excluded.memories_enabled,
                tools_enabled       = excluded.tools_enabled,
                updated_at_ms       = excluded.updated_at_ms
        )");
        q.bind(1, profile.id);
        q.bind(2, profile.display_name);
        q.bind(3, profile.custom_instructions);
        q.bind(4, profile.default_model);
        q.bind(5, profile.memories_enabled ? 1 : 0);
        q.bind(6, profile.tools_enabled ? 1 : 0);
        q.bind(7, profile.created_at_ms ? profile.created_at_ms : now);
        q.bind(8, now);
        q.exec();
    });
}

std::optional<UserProfile> UserDB::load_profile() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT * FROM user_profile WHERE id=? LIMIT 1");
    q.bind(1, user_id_);
    if (!q.executeStep()) return std::nullopt;
    UserProfile p;
    p.id                  = q.getColumn("id").getText();
    p.display_name        = q.getColumn("display_name").getText();
    p.custom_instructions = q.getColumn("custom_instructions").getText();
    p.default_model       = q.getColumn("default_model").getText();
    p.memories_enabled    = q.getColumn("memories_enabled").getInt() != 0;
    p.tools_enabled       = q.getColumn("tools_enabled").getInt() != 0;
    p.created_at_ms       = q.getColumn("created_at_ms").getInt64();
    p.updated_at_ms       = q.getColumn("updated_at_ms").getInt64();
    return p;
}

// ── Conversations ─────────────────────────────────────────────────────────────

ConvId UserDB::create_conversation(const NewConversation& spec) {
    std::lock_guard g(mutex_);
    SQLite::Transaction tx(*db_);
    if (!spec.parent_conv_id.empty()) {
        SQLite::Statement exists(*db_, "SELECT 1 FROM conversations WHERE id=? LIMIT 1");
        exists.bind(1, spec.parent_conv_id);
        if (!exists.executeStep()) return {};
    }
    const ConvId id = util::generate_uuid();
    const std::int64_t now = util::now_ms();
    if (spec.set_active) db_->exec("UPDATE conversations SET is_active=0 WHERE is_active=1");
    SQLite::Statement q(*db_, R"(
        INSERT INTO conversations
            (id, title, is_active, model_id, system_prompt, settings_json,
             parent_conv_id, created_at_ms, updated_at_ms)
        VALUES (?,?,?,?,?,?,?,?,?)
    )");
    q.bind(1, id);
    q.bind(2, spec.title.empty() ? std::string("New Conversation") : spec.title);
    q.bind(3, spec.set_active ? 1 : 0);
    q.bind(4, spec.model_id);
    q.bind(5, spec.system_prompt);
    q.bind(6, serialize(spec.settings));
    if (spec.parent_conv_id.empty()) q.bind(7);
    else                             q.bind(7, spec.parent_conv_id);
    q.bind(8, now);
    q.bind(9, now);
    q.exec();
    tx.commit();
    return id;
}

ConvId UserDB::get_or_create_active_conversation() {
    std::lock_guard g(mutex_);
    SQLite::Transaction tx(*db_);
    {
        SQLite::Statement active(*db_, "SELECT id FROM conversations WHERE is_active=1 LIMIT 1");
        if (active.executeStep()) return active.getColumn(0).getText();
    }
    const ConvId id = util::generate_uuid();
    const std::int64_t now = util::now_ms();
    SQLite::Statement q(*db_, R"(
        INSERT INTO conversations(id, title, is_active, created_at_ms, updated_at_ms)
        VALUES (?, 'New Conversation', 1, ?, ?)
    )");
    q.bind(1, id);
    q.bind(2, now);
    q.bind(3, now);
    q.exec();
    tx.commit();
    return id;
}

std::optional<Conversation> UserDB::get_conversation_locked(const ConvId& id) const {
    SQLite::Statement q(*db_, "SELECT * FROM conversations WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return std::nullopt;
    return row_to_conversation(q);
}

std::optional<Conversation> UserDB::get_conversation(const ConvId& id) const {
    std::lock_guard g(mutex_);
    return get_conversation_locked(id);
}

std::optional<Conversation> UserDB::load_conversation(const ConvId& id) const {
    std::lock_guard g(mutex_);
    auto conv = get_conversation_locked(id);
    if (!conv) return std::nullopt;
    SQLite::Statement mq(*db_,
        "SELECT * FROM messages WHERE conversation_id=? ORDER BY sequence_num");
    mq.bind(1, id);
    while (mq.executeStep()) {
        auto message = row_to_message(mq);
        load_message_parts(*db_, message);
        conv->messages.push_back(std::move(message));
    }
    return conv;
}

std::vector<Conversation> UserDB::list_conversations() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT * FROM conversations ORDER BY created_at_ms DESC, id");
    std::vector<Conversation> out;
    while (q.executeStep()) out.push_back(row_to_conversation(q));
    return out;
}

bool UserDB::conversation_exists(const ConvId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT 1 FROM conversations WHERE id=? LIMIT 1");
    q.bind(1, id);
    return q.executeStep();
}

bool UserDB::is_conversation_active(const ConvId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT is_active FROM conversations WHERE id=? LIMIT 1");
    q.bind(1, id);
    return q.executeStep() && q.getColumn(0).getInt() != 0;
}

bool UserDB::update_conversation(const ConvId& id, const ConversationPatch& patch) {
    std::lock_guard g(mutex_);
    auto conv = get_conversation_locked(id);
    if (!conv) return false;
    if (patch.title) conv->title = patch.title->empty() ? "Untitled conversation" : *patch.title;
    if (patch.model_id) conv->model_id = *patch.model_id;
    if (patch.system_prompt) conv->system_prompt = *patch.system_prompt;
    if (patch.settings) conv->settings = *patch.settings;
    SQLite::Statement q(*db_, R"(
        UPDATE conversations
        SET title=?, model_id=?, system_prompt=?, settings_json=?, updated_at_ms=?
        WHERE id=?
    )");
    q.bind(1, conv->title);
    q.bind(2, conv->model_id);
    q.bind(3, conv->system_prompt);
    q.bind(4, serialize(conv->settings));
    q.bind(5, util::now_ms());
    q.bind(6, id);
    return q.exec() > 0;
}

bool UserDB::set_active_conversation(const ConvId& id) {
    std::lock_guard g(mutex_);
    SQLite::Transaction tx(*db_);
    {
        SQLite::Statement exists(*db_, "SELECT 1 FROM conversations WHERE id=? LIMIT 1");
        exists.bind(1, id);
        if (!exists.executeStep()) return false;
    }
    // Cleared first: the partial unique index refuses a second active row.
    SQLite::Statement clear(*db_, "UPDATE conversations SET is_active=0 WHERE is_active=1 AND id<>?");
    clear.bind(1, id);
    clear.exec();
    SQLite::Statement set(*db_, "UPDATE conversations SET is_active=1, updated_at_ms=? WHERE id=?");
    set.bind(1, util::now_ms());
    set.bind(2, id);
    set.exec();
    tx.commit();
    return true;
}

std::optional<ConvId> UserDB::get_active_conversation_id() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT id FROM conversations WHERE is_active=1 LIMIT 1");
    if (!q.executeStep()) return std::nullopt;
    return std::string(q.getColumn(0).getText());
}

bool UserDB::delete_conversation(const ConvId& id) {
    std::lock_guard g(mutex_);
    SQLite::Transaction tx(*db_);
    SQLite::Statement q(*db_, "DELETE FROM conversations WHERE id=?");
    q.bind(1, id);
    if (q.exec() == 0) return false;

    // Attachments only this conversation referenced are orphans now. Uploads
    // still pending (expires_at_ms > 0) wait for their normal expiry.
    std::vector<ImageAttachment> orphans;
    {
        SQLite::Statement orphan_q(*db_, R"(
            SELECT * FROM attachments a
            WHERE a.expires_at_ms=0 AND NOT EXISTS (
                SELECT 1 FROM message_content_parts p WHERE p.attachment_id=a.id)
        )");
        while (orphan_q.executeStep()) orphans.push_back(row_to_attachment(orphan_q));
    }
    for (const auto& attachment : orphans) {
        SQLite::Statement dq(*db_, "DELETE FROM attachments WHERE id=?");
        dq.bind(1, attachment.id);
        dq.exec();
    }
    tx.commit();
    // Files after the rows are gone, so a failed commit cannot leave rows
    // pointing at deleted files.
    for (const auto& attachment : orphans) {
        const auto path = attachment_file_path(attachment);
        std::error_code ec;
        if (!path.empty()) std::filesystem::remove(path, ec);
    }
    return true;
}

bool UserDB::save_summary(const ConvId& id,
                          const std::string& summary,
                          int through_seq,
                          const std::string& model_id) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        UPDATE conversations
        SET summary=?, summary_through_seq=?, summary_model_id=?, updated_at_ms=?
        WHERE id=? AND summary_through_seq<=?
    )");
    q.bind(1, summary);
    q.bind(2, through_seq);
    q.bind(3, model_id);
    q.bind(4, util::now_ms());
    q.bind(5, id);
    q.bind(6, through_seq);
    return q.exec() > 0;
}

// ── Messages ──────────────────────────────────────────────────────────────────

int UserDB::append_message(const ConvId& conv_id, const Message& msg) {
    std::lock_guard g(mutex_);
    const std::int64_t ts = msg.timestamp_ms ? msg.timestamp_ms : util::now_ms();
    SQLite::Transaction tx(*db_);
    {
        SQLite::Statement exists(*db_, "SELECT 1 FROM conversations WHERE id=? LIMIT 1");
        exists.bind(1, conv_id);
        if (!exists.executeStep()) return -1;
    }
    int seq = 0;
    {
        SQLite::Statement next(*db_,
            "SELECT COALESCE(MAX(sequence_num), -1) + 1 FROM messages WHERE conversation_id=?");
        next.bind(1, conv_id);
        if (next.executeStep()) seq = next.getColumn(0).getInt();
    }

    SQLite::Statement q(*db_, R"(
        INSERT INTO messages
            (conversation_id, sequence_num, role, content, tool_calls_json,
             tool_call_id, thinking_text, token_count, model_id,
             trace_events_json, timestamp_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    )");
    q.bind(1, conv_id);
    q.bind(2, seq);
    q.bind(3, to_string(msg.role));
    q.bind(4, msg.content);
    q.bind(5, serialize(msg.tool_calls));
    q.bind(6, msg.tool_call_id);
    q.bind(7, msg.thinking_text);
    q.bind(8, msg.token_count);
    q.bind(9, msg.model_id);
    q.bind(10, serialize(msg.trace_events));
    q.bind(11, ts);
    q.exec();

    const std::int64_t message_id = db_->getLastInsertRowid();
    std::vector<MessageContentPart> parts = msg.content_parts;
    if (parts.empty() && !msg.content.empty()) {
        MessageContentPart text_part;
        text_part.type = "text";
        text_part.text = msg.content;
        parts.push_back(std::move(text_part));
    }
    for (std::size_t ordinal = 0; ordinal < parts.size(); ++ordinal) {
        const auto& part = parts[ordinal];
        SQLite::Statement pq(*db_, R"(
            INSERT INTO message_content_parts(message_id, ordinal, type, text, attachment_id)
            VALUES (?,?,?,?,?)
        )");
        pq.bind(1, message_id);
        pq.bind(2, static_cast<int>(ordinal));
        pq.bind(3, part.type.empty() ? std::string("text") : part.type);
        pq.bind(4, part.text);
        if (part.attachment_id.empty()) pq.bind(5);
        else                            pq.bind(5, part.attachment_id);
        pq.exec();
        if (!part.attachment_id.empty()) {
            // Referenced now: no longer a pending upload that may expire.
            SQLite::Statement aq(*db_, "UPDATE attachments SET expires_at_ms=0 WHERE id=?");
            aq.bind(1, part.attachment_id);
            aq.exec();
        }
    }

    SQLite::Statement uq(*db_, R"(
        UPDATE conversations
        SET total_tokens = total_tokens + ?, updated_at_ms = ?
        WHERE id = ?
    )");
    uq.bind(1, msg.token_count);
    uq.bind(2, ts);
    uq.bind(3, conv_id);
    uq.exec();

    tx.commit();
    return seq;
}

std::vector<Message> UserDB::load_messages(const ConvId& conv_id, int after_seq) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        SELECT * FROM messages
        WHERE conversation_id=? AND sequence_num>?
        ORDER BY sequence_num
    )");
    q.bind(1, conv_id);
    q.bind(2, after_seq);
    std::vector<Message> out;
    while (q.executeStep()) {
        auto message = row_to_message(q);
        load_message_parts(*db_, message);
        out.push_back(std::move(message));
    }
    return out;
}

int UserDB::get_total_tokens(const ConvId& conv_id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT total_tokens FROM conversations WHERE id=? LIMIT 1");
    q.bind(1, conv_id);
    return q.executeStep() ? q.getColumn(0).getInt() : 0;
}

int UserDB::message_count(const ConvId& conv_id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT COUNT(*) FROM messages WHERE conversation_id=?");
    q.bind(1, conv_id);
    return q.executeStep() ? q.getColumn(0).getInt() : 0;
}

// ── Attachments ───────────────────────────────────────────────────────────────

std::string UserDB::attachment_directory() const {
    const auto dir = std::filesystem::path(user_dir_) / "attachments";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir.lexically_normal().string();
}

std::string UserDB::attachment_file_path(const ImageAttachment& attachment) const {
    if (attachment.relative_path.empty()) return {};
    const auto root = (std::filesystem::path(user_dir_) / "attachments").lexically_normal();
    const auto full =
        (std::filesystem::path(user_dir_) / attachment.relative_path).lexically_normal();
    // Inside attachments/, component by component — a stored "../" must not
    // reach anything else.
    auto rit = root.begin();
    auto fit = full.begin();
    for (; rit != root.end(); ++rit, ++fit) {
        if (fit == full.end() || *fit != *rit) return {};
    }
    return full.string();
}

void UserDB::save_attachment(const ImageAttachment& attachment) {
    if (attachment.id.empty() || attachment.relative_path.empty() ||
        attachment.mime_type.empty() || attachment.size_bytes < 0 ||
        attachment_file_path(attachment).empty()) {
        throw std::invalid_argument("invalid attachment metadata");
    }
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        INSERT INTO attachments
            (id, original_filename, mime_type, relative_path, size_bytes,
             created_at_ms, expires_at_ms)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            original_filename=excluded.original_filename,
            mime_type=excluded.mime_type,
            relative_path=excluded.relative_path,
            size_bytes=excluded.size_bytes,
            created_at_ms=excluded.created_at_ms,
            expires_at_ms=excluded.expires_at_ms
    )");
    q.bind(1, attachment.id);
    q.bind(2, attachment.original_filename);
    q.bind(3, attachment.mime_type);
    q.bind(4, attachment.relative_path);
    q.bind(5, attachment.size_bytes);
    q.bind(6, attachment.created_at_ms);
    q.bind(7, attachment.expires_at_ms);
    q.exec();
}

std::optional<ImageAttachment> UserDB::get_attachment(const std::string& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT * FROM attachments WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return std::nullopt;
    return row_to_attachment(q);
}

bool UserDB::attachment_is_referenced(const std::string& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT 1 FROM message_content_parts WHERE attachment_id=? LIMIT 1");
    q.bind(1, id);
    return q.executeStep();
}

bool UserDB::delete_attachment(const std::string& id, bool* referenced) {
    std::lock_guard g(mutex_);
    if (referenced) *referenced = false;
    ImageAttachment attachment;
    {
        SQLite::Statement get_q(*db_, "SELECT * FROM attachments WHERE id=? LIMIT 1");
        get_q.bind(1, id);
        if (!get_q.executeStep()) return false;
        attachment = row_to_attachment(get_q);
    }
    SQLite::Statement ref_q(*db_, "SELECT 1 FROM message_content_parts WHERE attachment_id=? LIMIT 1");
    ref_q.bind(1, id);
    if (ref_q.executeStep()) {
        if (referenced) *referenced = true;
        return false;
    }
    SQLite::Statement delete_q(*db_, "DELETE FROM attachments WHERE id=?");
    delete_q.bind(1, id);
    delete_q.exec();
    const auto path = attachment_file_path(attachment);
    if (!path.empty()) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    return true;
}

std::vector<ImageAttachment> UserDB::delete_expired_unreferenced_attachments(std::int64_t now_ms) {
    std::lock_guard g(mutex_);
    std::vector<ImageAttachment> removed;
    {
        SQLite::Statement q(*db_, R"(
            SELECT * FROM attachments a
            WHERE a.expires_at_ms>0 AND a.expires_at_ms<=? AND NOT EXISTS (
                SELECT 1 FROM message_content_parts p WHERE p.attachment_id=a.id)
        )");
        q.bind(1, now_ms);
        while (q.executeStep()) removed.push_back(row_to_attachment(q));
    }
    for (const auto& attachment : removed) {
        SQLite::Statement dq(*db_, "DELETE FROM attachments WHERE id=?");
        dq.bind(1, attachment.id);
        dq.exec();
        const auto path = attachment_file_path(attachment);
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
    return removed;
}

// ── Global memories ───────────────────────────────────────────────────────────

void UserDB::add_memory(const Memory& mem) {
    std::lock_guard g(mutex_);
    const std::int64_t now = util::now_ms();
    SQLite::Statement q(*db_, R"(
        INSERT INTO global_memories
            (id, content, source_conv_id, importance, created_at_ms, updated_at_ms)
        VALUES (?,?,?,?,?,?)
    )");
    q.bind(1, mem.id.empty() ? util::generate_uuid() : mem.id);
    q.bind(2, mem.content);
    if (mem.source_conv_id.empty()) q.bind(3);
    else                            q.bind(3, mem.source_conv_id);
    q.bind(4, static_cast<double>(mem.importance));
    q.bind(5, mem.created_at_ms ? mem.created_at_ms : now);
    q.bind(6, mem.updated_at_ms ? mem.updated_at_ms : now);
    q.exec();
}

void UserDB::update_memory(const Memory& mem) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "UPDATE global_memories SET content=?, importance=?, updated_at_ms=? WHERE id=?");
    q.bind(1, mem.content);
    q.bind(2, static_cast<double>(mem.importance));
    q.bind(3, util::now_ms());
    q.bind(4, mem.id);
    q.exec();
}

void UserDB::delete_memory(const MemoryId& id) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "DELETE FROM global_memories WHERE id=?");
    q.bind(1, id);
    q.exec();
}

std::optional<Memory> UserDB::get_memory(const MemoryId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT * FROM global_memories WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return std::nullopt;
    return row_to_memory(q);
}

std::vector<Memory> UserDB::list_memories() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM global_memories ORDER BY importance DESC, created_at_ms DESC");
    std::vector<Memory> out;
    while (q.executeStep()) out.push_back(row_to_memory(q));
    return out;
}

std::vector<Memory> UserDB::search_memories(const std::string& query, int limit) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        SELECT * FROM global_memories
        WHERE content LIKE ?
        ORDER BY importance DESC, created_at_ms DESC
        LIMIT ?
    )");
    q.bind(1, "%" + query + "%");
    q.bind(2, limit);
    std::vector<Memory> out;
    while (q.executeStep()) out.push_back(row_to_memory(q));
    return out;
}

// ── Local memories ────────────────────────────────────────────────────────────

void UserDB::add_local_memory(const LocalMemory& mem) {
    std::lock_guard g(mutex_);
    const std::int64_t now = util::now_ms();
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

void UserDB::update_local_memory(const LocalMemory& mem) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "UPDATE local_memories SET content=?, updated_at_ms=? WHERE id=?");
    q.bind(1, mem.content);
    q.bind(2, util::now_ms());
    q.bind(3, mem.id);
    q.exec();
}

void UserDB::delete_local_memory(const std::string& id) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "DELETE FROM local_memories WHERE id=?");
    q.bind(1, id);
    q.exec();
}

std::optional<LocalMemory> UserDB::get_local_memory(const std::string& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT * FROM local_memories WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return std::nullopt;
    return row_to_local_memory(q);
}

std::vector<LocalMemory> UserDB::list_local_memories(const ConvId& conv_id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM local_memories WHERE conversation_id=? ORDER BY created_at_ms, id");
    q.bind(1, conv_id);
    std::vector<LocalMemory> out;
    while (q.executeStep()) out.push_back(row_to_local_memory(q));
    return out;
}

// ── Voice ─────────────────────────────────────────────────────────────────────

void UserDB::save_voice_proposal(const VoiceDesignProposal& proposal) {
    std::lock_guard g(mutex_);
    const std::int64_t now = util::now_ms();
    SQLite::Statement q(*db_, R"(
        INSERT INTO voice_proposals
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
    q.bind(3, proposal.language.empty() ? std::string("Auto") : proposal.language);
    q.bind(4, proposal.voice_description);
    q.bind(5, proposal.sample_text);
    q.bind(6, proposal.rationale);
    q.bind(7, proposal.status.empty() ? std::string("pending") : proposal.status);
    q.bind(8, proposal.provider.empty() ? std::string("qwen3-tts") : proposal.provider);
    q.bind(9, proposal.voice_design_model_id);
    q.bind(10, proposal.clone_model_id);
    q.bind(11, proposal.preview_audio_path);
    q.bind(12, proposal.voice_clone_prompt_path);
    q.bind(13, proposal.error);
    q.bind(14, proposal.created_at_ms ? proposal.created_at_ms : now);
    q.bind(15, proposal.updated_at_ms ? proposal.updated_at_ms : now);
    q.exec();
}

std::optional<VoiceDesignProposal> UserDB::get_voice_proposal(const VoiceProposalId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT * FROM voice_proposals WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return std::nullopt;
    return row_to_voice_proposal(q, user_id_);
}

std::vector<VoiceDesignProposal> UserDB::list_voice_proposals() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT * FROM voice_proposals ORDER BY created_at_ms DESC");
    std::vector<VoiceDesignProposal> out;
    while (q.executeStep()) out.push_back(row_to_voice_proposal(q, user_id_));
    return out;
}

void UserDB::update_voice_proposal_status(const VoiceProposalId& id,
                                          const std::string& status,
                                          const std::string& error) {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "UPDATE voice_proposals SET status=?, error=?, updated_at_ms=? WHERE id=?");
    q.bind(1, status);
    q.bind(2, error);
    q.bind(3, util::now_ms());
    q.bind(4, id);
    q.exec();
}

void UserDB::save_voice_profile(const VoiceProfile& profile) {
    std::lock_guard g(mutex_);
    const std::int64_t now = util::now_ms();
    SQLite::Transaction tx(*db_);
    if (profile.active) {
        SQLite::Statement clear(*db_,
            "UPDATE voice_profiles SET active=0, updated_at_ms=? WHERE active=1");
        clear.bind(1, now);
        clear.exec();
    }
    SQLite::Statement q(*db_, R"(
        INSERT INTO voice_profiles
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
    q.bind(3, profile.language.empty() ? std::string("Auto") : profile.language);
    q.bind(4, profile.voice_description);
    q.bind(5, profile.sample_text);
    q.bind(6, profile.rationale);
    q.bind(7, profile.provider.empty() ? std::string("qwen3-tts") : profile.provider);
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

std::optional<VoiceProfile> UserDB::get_voice_profile(const VoiceProfileId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, "SELECT * FROM voice_profiles WHERE id=? LIMIT 1");
    q.bind(1, id);
    if (!q.executeStep()) return std::nullopt;
    return row_to_voice_profile(q, user_id_);
}

std::optional<VoiceProfile> UserDB::get_active_voice_profile() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM voice_profiles WHERE active=1 ORDER BY updated_at_ms DESC LIMIT 1");
    if (!q.executeStep()) return std::nullopt;
    return row_to_voice_profile(q, user_id_);
}

std::vector<VoiceProfile> UserDB::list_voice_profiles() const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM voice_profiles ORDER BY active DESC, updated_at_ms DESC");
    std::vector<VoiceProfile> out;
    while (q.executeStep()) out.push_back(row_to_voice_profile(q, user_id_));
    return out;
}

void UserDB::set_active_voice_profile(const VoiceProfileId& id) {
    std::lock_guard g(mutex_);
    const std::int64_t now = util::now_ms();
    SQLite::Transaction tx(*db_);
    SQLite::Statement clear(*db_,
        "UPDATE voice_profiles SET active=0, updated_at_ms=? WHERE active=1");
    clear.bind(1, now);
    clear.exec();
    SQLite::Statement set(*db_, "UPDATE voice_profiles SET active=1, updated_at_ms=? WHERE id=?");
    set.bind(1, now);
    set.bind(2, id);
    set.exec();
    tx.commit();
}

void UserDB::save_tts_cache_entry(const TtsSynthesisResult& entry) {
    std::lock_guard g(mutex_);
    const std::int64_t now = util::now_ms();
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
    q.bind(7, entry.mime_type.empty() ? std::string("audio/wav") : entry.mime_type);
    q.bind(8, entry.format.empty() ? std::string("wav") : entry.format);
    q.bind(9, entry.sample_rate);
    q.bind(10, entry.duration_ms);
    q.bind(11, entry.created_at_ms ? entry.created_at_ms : now);
    q.bind(12, entry.expires_at_ms);
    q.exec();
}

std::optional<TtsSynthesisResult> UserDB::find_tts_cache_entry(const VoiceProfileId& profile_id,
                                                               const std::string& text_hash,
                                                               const ConvId& conversation_id,
                                                               int message_index) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_, R"(
        SELECT * FROM tts_audio_cache
        WHERE voice_profile_id=? AND text_hash=? AND conversation_id=?
          AND message_index=? AND expires_at_ms>?
        ORDER BY created_at_ms DESC
        LIMIT 1
    )");
    q.bind(1, profile_id);
    q.bind(2, text_hash);
    q.bind(3, conversation_id);
    q.bind(4, message_index);
    q.bind(5, util::now_ms());
    if (!q.executeStep()) return std::nullopt;
    auto result = row_to_tts_cache(q, user_id_);
    result.cached = true;
    return result;
}

std::optional<TtsSynthesisResult> UserDB::get_tts_cache_entry(const TtsCacheId& id) const {
    std::lock_guard g(mutex_);
    SQLite::Statement q(*db_,
        "SELECT * FROM tts_audio_cache WHERE id=? AND expires_at_ms>? LIMIT 1");
    q.bind(1, id);
    q.bind(2, util::now_ms());
    if (!q.executeStep()) return std::nullopt;
    auto result = row_to_tts_cache(q, user_id_);
    result.cached = true;
    return result;
}

std::vector<TtsSynthesisResult> UserDB::delete_expired_tts_cache_entries(std::int64_t now_ms) {
    std::lock_guard g(mutex_);
    std::vector<TtsSynthesisResult> expired;
    {
        SQLite::Statement q(*db_, "SELECT * FROM tts_audio_cache WHERE expires_at_ms<=?");
        q.bind(1, now_ms);
        while (q.executeStep()) expired.push_back(row_to_tts_cache(q, user_id_));
    }
    SQLite::Statement del(*db_, "DELETE FROM tts_audio_cache WHERE expires_at_ms<=?");
    del.bind(1, now_ms);
    del.exec();
    return expired;
}

} // namespace mm
