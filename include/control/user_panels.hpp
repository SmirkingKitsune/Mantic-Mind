#pragma once

// Mantic-Mind — the Users, Chat and Voice tabs.
//
// A user owns conversations, their notes, global memories and a voice; a model
// is picked per conversation from the served-model catalog (the Models tab).
// These three tabs show one user at a time — the same user on all three, so
// choosing someone in Users is choosing whom Chat talks as.
//
// Same rule as every panel written since the Soma tab: every value arrives over
// `/v1/*`, and every action is a `/v1/*` call. Nothing here touches
// UserManager or a user's store, even though it renders their contents — a TUI
// that reached into them would be a client with privileges no other client
// has, and would skip the per-conversation ordering the API enforces.
// tools/ci/check_ui_api.py guards this header and its .cpp.

#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mm {

class AudioPlayer;

struct UserView {
    std::string id;
    std::string display_name;
    std::string default_model;
    std::string custom_instructions;
    std::string active_conversation_id;
    bool memories_enabled = true;
    bool tools_enabled = false;
};

struct ConversationView {
    std::string id;
    std::string title;
    std::string model_id;       ///< empty: the user's default applies
    std::string summary_model_id;
    bool is_active = false;
    int total_tokens = 0;
    int summary_through_seq = -1;
};

struct MemoryView {
    std::string id;
    std::string content;
    std::string source_conv_id;
    float importance = 0.5f;
};

/// A note taken in one conversation (a local memory).
struct NoteView {
    std::string id;
    std::string content;
};

/// One message as the transcript shows it.
struct TranscriptLine {
    std::string role;     ///< user | assistant | tool
    std::string model_id; ///< who answered, for assistant lines
    std::string text;
};

struct UserSnapshot {
    bool reachable = false;
    std::string error;
    std::vector<UserView> users;
    /// Available served-model ids — what a conversation can be switched to.
    std::vector<std::string> models;
    /// The user the rest describes; empty when there is none.
    std::string user_id;
    std::vector<ConversationView> conversations;
    std::vector<MemoryView> memories;
    /// The conversation the transcript and the notes are of.
    std::string conversation_id;
    std::vector<TranscriptLine> transcript;
    std::vector<NoteView> notes;
    /// GET /v1/users/{id}/voice, verbatim.
    nlohmann::json voice = nlohmann::json::object();
};

/// Polls `/v1/users` (and the focused user's conversations, memories, voice
/// and one transcript) on its own thread, and performs the tabs' actions.
class UserDashboard {
public:
    UserDashboard(std::string base_url, std::string api_token);
    ~UserDashboard();

    UserDashboard(const UserDashboard&) = delete;
    UserDashboard& operator=(const UserDashboard&) = delete;

    void start(int interval_ms);
    void stop();
    void refresh_now();

    UserSnapshot snapshot() const;

    /// Whom, and which conversation, the poll describes in detail. Shared by
    /// the three tabs.
    void focus(const std::string& user_id, const std::string& conversation_id);
    std::string focused_user() const;
    std::string focused_conversation() const;

    // ── Actions ──────────────────────────────────────────────────────────────
    // Synchronous HTTP: call them off the UI thread. False with `error` naming
    // the server's reason.
    bool create_user(const nlohmann::json& body, std::string& error);
    bool update_user(const std::string& id, const nlohmann::json& body, std::string& error);
    bool delete_user(const std::string& id, std::string& error);
    bool create_conversation(const std::string& user_id, const nlohmann::json& body,
                             std::string& error);
    bool update_conversation(const std::string& user_id, const std::string& conv_id,
                             const nlohmann::json& body, std::string& error);
    /// `activate`, `compact` or `delete`.
    bool conversation_action(const std::string& user_id, const std::string& conv_id,
                             const std::string& verb, std::string& error);
    bool add_memory(const std::string& user_id, const nlohmann::json& body, std::string& error);
    bool update_memory(const std::string& user_id, const std::string& memory_id,
                       const nlohmann::json& body, std::string& error);
    bool delete_memory(const std::string& user_id, const std::string& memory_id,
                       std::string& error);
    /// Queue extraction of global memories from a range of one conversation's
    /// messages (POST .../memories/extract).
    bool extract_memories(const std::string& user_id, const nlohmann::json& body,
                          std::string& error);
    // A conversation's notes.
    bool add_note(const std::string& user_id, const std::string& conv_id,
                  const std::string& content, std::string& error);
    bool update_note(const std::string& user_id, const std::string& conv_id,
                     const std::string& note_id, const std::string& content, std::string& error);
    bool delete_note(const std::string& user_id, const std::string& conv_id,
                     const std::string& note_id, std::string& error);
    bool propose_curation(const std::string& user_id, const std::string& conv_id,
                          nlohmann::json& proposals, std::string& error);
    bool apply_curation(const std::string& user_id, const nlohmann::json& proposals,
                        std::string& error);

    struct ChatResult {
        bool ok = false;
        std::string conv_id;
        std::string model_id;
        std::string error;
        std::string code;
    };
    /// One turn over SSE. `model` empty: the conversation's, else the user's
    /// default. `image_path`, when set, is uploaded and attached first.
    /// `on_delta` receives the answer as it streams; `cancel` aborts it.
    ChatResult chat(const std::string& user_id, const std::string& conv_id,
                    const std::string& message, const std::string& model,
                    const std::string& image_path,
                    const std::function<void(const std::string&)>& on_delta,
                    const std::atomic<bool>* cancel);

    bool voice_propose(const std::string& user_id, const nlohmann::json& body, std::string& error);
    /// `sample`, `approve` or `reject`; `response` is the server's answer.
    bool voice_proposal_action(const std::string& user_id, const std::string& proposal_id,
                               const std::string& verb, nlohmann::json& response,
                               std::string& error);
    bool speak(const std::string& user_id, const std::string& text, nlohmann::json& response,
               std::string& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// The transcript's lines from a conversation document as the API returns it.
/// Pure, so it can be asserted.
std::vector<TranscriptLine> transcript_of(const nlohmann::json& conversation);

// ── Pure renderers ────────────────────────────────────────────────────────────

ftxui::Element render_user_list(const UserSnapshot& snap, int selected, bool focused);
ftxui::Element render_conversation_list(const UserSnapshot& snap, int selected, bool focused);
ftxui::Element render_memory_list(const UserSnapshot& snap, int selected, bool focused);
ftxui::Element render_note_list(const UserSnapshot& snap, int selected, bool focused);
/// Each assistant line carries the model that wrote it; `streaming` is the
/// answer arriving now, shown after the stored lines.
ftxui::Element render_transcript(const std::vector<TranscriptLine>& lines,
                                 const std::string& streaming);

// ── The tabs ──────────────────────────────────────────────────────────────────

ftxui::Component users_tab(UserDashboard& dashboard);
/// `shutting_down` aborts a stream in flight when the TUI exits.
ftxui::Component chat_tab(UserDashboard& dashboard, const std::atomic<bool>& shutting_down);
ftxui::Component voice_tab(UserDashboard& dashboard, AudioPlayer& player);

} // namespace mm
