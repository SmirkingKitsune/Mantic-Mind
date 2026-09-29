#pragma once

// Mantic-Mind — one user's store: {data_dir}/users/{id}/user.db (SQLite, WAL),
// with managed attachments beside it.
//
// It holds the profile, conversations and their messages, per-conversation
// (local) memories, the user's global memories, and their voice. It holds NO
// model: a
// conversation names a served model by id, and each assistant message records
// the one that wrote it.
//
// A fresh schema, not an agent.db migration: agent data is never imported.
// Compared with an agent's store —
//   - a conversation carries its model, its own system prompt and sampling
//     overrides, and its rolling summary with the watermark it covers;
//     compaction folds in place and never forks a new conversation
//   - a message carries its model and a sequence number the STORE assigns,
//     unique within its conversation
//   - at most one conversation is active, enforced by a partial unique index
//     rather than by every caller remembering to clear the others first
//
// Every public method is thread-safe (one connection, one mutex).

#include "common/models.hpp"
#include "common/user.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace SQLite { class Database; }

namespace mm {

/// What a new conversation starts with. At namespace scope, not nested in
/// UserDB: a nested type with member initializers cannot be a default argument
/// of its enclosing class's own functions (CWG 1397), which GCC and Clang
/// enforce.
struct NewConversation {
    std::string title;
    std::string model_id;
    std::string system_prompt;
    SamplingOverrides settings;
    ConvId parent_conv_id;
    bool set_active = false;
};

/// A change to a conversation: only the fields that are set change.
struct ConversationPatch {
    std::optional<std::string> title;
    std::optional<std::string> model_id;
    std::optional<std::string> system_prompt;
    std::optional<SamplingOverrides> settings;
};

class UserDB {
public:
    UserDB(const UserId& user_id, const std::string& data_dir = "data");
    ~UserDB();

    UserDB(const UserDB&) = delete;
    UserDB& operator=(const UserDB&) = delete;

    const UserId& user_id() const noexcept { return user_id_; }
    /// {data_dir}/users/{id}: the database, and attachments/ under it.
    const std::string& directory() const noexcept { return user_dir_; }

    // ── Profile (one row) ─────────────────────────────────────────────────────
    /// Insert or update. Keeps the stored created_at_ms. Throws
    /// std::invalid_argument when profile.id is not this store's user.
    void save_profile(const UserProfile& profile);
    std::optional<UserProfile> load_profile() const;

    // ── Conversations ─────────────────────────────────────────────────────────
    /// Empty when parent_conv_id names no conversation of this user.
    ConvId create_conversation(const NewConversation& spec = {});

    /// The active conversation, created and activated when there is none —
    /// atomically, so two first turns arriving together share one.
    ConvId get_or_create_active_conversation();

    /// With messages.
    std::optional<Conversation> load_conversation(const ConvId& id) const;
    /// Metadata only.
    std::optional<Conversation> get_conversation(const ConvId& id) const;
    /// Metadata only, newest first.
    std::vector<Conversation> list_conversations() const;
    bool conversation_exists(const ConvId& id) const;
    bool is_conversation_active(const ConvId& id) const;

    /// False when there is no such conversation.
    bool update_conversation(const ConvId& id, const ConversationPatch& patch);

    /// False when there is no such conversation; the previous one is left active.
    bool set_active_conversation(const ConvId& id);
    std::optional<ConvId> get_active_conversation_id() const;

    /// Messages and local memories go with it (cascade), and so do attachments
    /// nothing else references. False when there is no such conversation.
    bool delete_conversation(const ConvId& id);

    /// Compaction progress: `summary` covers every message with a sequence
    /// number up to and including `through_seq`. Saved after each chunk so a
    /// failure part-way resumes where it stopped. Refuses to move the watermark
    /// backwards — a stale writer must not undo a newer fold. False when there
    /// is no such conversation or the watermark would regress.
    bool save_summary(const ConvId& id,
                      const std::string& summary,
                      int through_seq,
                      const std::string& model_id);

    // ── Messages ──────────────────────────────────────────────────────────────
    /// Appends with the next sequence number, assigned here in the same
    /// transaction, and returns it; -1 when there is no such conversation.
    int append_message(const ConvId& conv_id, const Message& msg);
    /// Messages with a sequence number above `after_seq`, in order.
    std::vector<Message> load_messages(const ConvId& conv_id, int after_seq = -1) const;
    int get_total_tokens(const ConvId& conv_id) const;
    int message_count(const ConvId& conv_id) const;

    // ── Attachments ───────────────────────────────────────────────────────────
    // Files live under attachment_directory(); rows hold relative paths only.
    std::string attachment_directory() const;
    std::string attachment_file_path(const ImageAttachment& attachment) const;
    void save_attachment(const ImageAttachment& attachment);
    std::optional<ImageAttachment> get_attachment(const std::string& id) const;
    bool attachment_is_referenced(const std::string& id) const;
    bool delete_attachment(const std::string& id, bool* referenced = nullptr);
    std::vector<ImageAttachment> delete_expired_unreferenced_attachments(std::int64_t now_ms);

    // ── Global memories ───────────────────────────────────────────────────────
    void add_memory(const Memory& mem);
    void update_memory(const Memory& mem);
    void delete_memory(const MemoryId& id);
    std::optional<Memory> get_memory(const MemoryId& id) const;
    /// Ranked by importance, then newest.
    std::vector<Memory> list_memories() const;
    std::vector<Memory> search_memories(const std::string& query, int limit = 20) const;

    // ── Local memories (per conversation) ─────────────────────────────────────
    void add_local_memory(const LocalMemory& mem);
    void update_local_memory(const LocalMemory& mem);
    void delete_local_memory(const std::string& id);
    std::optional<LocalMemory> get_local_memory(const std::string& id) const;
    std::vector<LocalMemory> list_local_memories(const ConvId& conv_id) const;

    // ── Voice ─────────────────────────────────────────────────────────────────
    // Audio and clone-prompt artifacts are files under the user's tts/; rows
    // hold metadata and paths.
    void save_voice_proposal(const VoiceDesignProposal& proposal);
    std::optional<VoiceDesignProposal> get_voice_proposal(const VoiceProposalId& id) const;
    std::vector<VoiceDesignProposal> list_voice_proposals() const;
    void update_voice_proposal_status(const VoiceProposalId& id,
                                      const std::string& status,
                                      const std::string& error = {});

    /// Saving an active profile deactivates the others, in one transaction.
    void save_voice_profile(const VoiceProfile& profile);
    std::optional<VoiceProfile> get_voice_profile(const VoiceProfileId& id) const;
    std::optional<VoiceProfile> get_active_voice_profile() const;
    std::vector<VoiceProfile> list_voice_profiles() const;
    void set_active_voice_profile(const VoiceProfileId& id);

    void save_tts_cache_entry(const TtsSynthesisResult& entry);
    std::optional<TtsSynthesisResult> find_tts_cache_entry(const VoiceProfileId& profile_id,
                                                           const std::string& text_hash,
                                                           const ConvId& conversation_id = {},
                                                           int message_index = -1) const;
    std::optional<TtsSynthesisResult> get_tts_cache_entry(const TtsCacheId& id) const;
    std::vector<TtsSynthesisResult> delete_expired_tts_cache_entries(std::int64_t now_ms);

private:
    UserId user_id_;
    std::string user_dir_;
    std::unique_ptr<SQLite::Database> db_;
    mutable std::mutex mutex_;

    void run_migrations();
    std::optional<Conversation> get_conversation_locked(const ConvId& id) const;
};

} // namespace mm
