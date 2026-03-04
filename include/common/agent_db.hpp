#pragma once

#include "common/models.hpp"
#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <memory>

// Forward-declare SQLite::Database to avoid exposing SQLiteCpp in this header.
namespace SQLite { class Database; }

namespace mm {

// Per-agent SQLite database (WAL mode).
// File location: {data_dir}/agents/{agent_id}/agent.db
// All public methods are thread-safe via an internal mutex.
class AgentDB {
public:
    AgentDB(const AgentId& agent_id, const std::string& data_dir = "data");
    ~AgentDB();

    // ── Config (1 row) ────────────────────────────────────────────────────────
    void        save_config(const AgentConfig& cfg);
    AgentConfig load_config() const;

    // ── Conversations ─────────────────────────────────────────────────────────
    ConvId                    create_conversation(const std::string& title = "");
    ConvId                    create_conversation(const std::string& title,
                                                  const ConvId& parent_conv_id);
    void                      save_conversation(const Conversation& conv);
    std::optional<Conversation> load_conversation(const ConvId& id) const;
    std::vector<Conversation> list_conversations() const;
    bool                      conversation_exists(const ConvId& id) const;
    bool                      is_conversation_active(const ConvId& id) const;
    void                      delete_conversation(const ConvId& id);
    void                      set_active_conversation(const ConvId& id);
    std::optional<ConvId>     get_active_conversation_id() const;

    // ── Messages ──────────────────────────────────────────────────────────────
    void               append_message(const ConvId& conv_id,
                                      const Message& msg,
                                      int sequence_num);
    std::vector<Message> load_messages(const ConvId& conv_id) const;
    int                get_total_tokens(const ConvId& conv_id) const;

    // ── Global memories ─────────────────────────────────────────────────────
    void                       add_memory(const Memory& mem);
    void                       update_memory(const Memory& mem);
    void                       delete_memory(const MemoryId& id);
    std::optional<Memory>      get_memory(const MemoryId& id) const;
    std::vector<Memory>        list_memories() const;
    // Returns up to `limit` memories ranked by importance DESC, created_at DESC.
    std::vector<Memory>        search_memories(const std::string& query, int limit = 20) const;

    // ── Local memories (per-conversation) ────────────────────────────────────
    void                           add_local_memory(const LocalMemory& mem);
    void                           update_local_memory(const LocalMemory& mem);
    void                           delete_local_memory(const std::string& id);
    std::optional<LocalMemory>     get_local_memory(const std::string& id) const;
    std::vector<LocalMemory>       list_local_memories(const ConvId& conv_id) const;
    void                           transfer_local_memories(const ConvId& from, const ConvId& to);

private:
    AgentId                       agent_id_;
    std::unique_ptr<SQLite::Database> db_;
    mutable std::mutex            mutex_;

    void run_migrations();
};

} // namespace mm
