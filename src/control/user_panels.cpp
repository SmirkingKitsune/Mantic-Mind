#include "control/user_panels.hpp"

#include "common/http_client.hpp"
#include "common/util.hpp"
#include "control/audio_player.hpp"
#include "control/model_panels.hpp" // describe_error_body: one reading of an error body

#include <ftxui/component/event.hpp>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <optional>
#include <thread>

namespace mm {

using namespace ftxui;

namespace {

std::string truncate(const std::string& s, std::size_t n) {
    if (s.size() <= n) return s;
    return n > 1 ? s.substr(0, n - 1) + "…" : s.substr(0, n);
}

std::string one_line(std::string s) {
    std::replace(s.begin(), s.end(), '\n', ' ');
    std::replace(s.begin(), s.end(), '\r', ' ');
    return s;
}

std::string str(const nlohmann::json& j, const char* key) {
    if (!j.is_object() || !j.contains(key) || !j.at(key).is_string()) return {};
    return j.at(key).get<std::string>();
}

UserView parse_user(const nlohmann::json& j) {
    UserView u;
    u.id = str(j, "id");
    u.display_name = str(j, "display_name");
    u.default_model = str(j, "default_model");
    u.custom_instructions = str(j, "custom_instructions");
    u.active_conversation_id = str(j, "active_conversation_id");
    u.memories_enabled = j.value("memories_enabled", true);
    u.tools_enabled = j.value("tools_enabled", false);
    return u;
}

ConversationView parse_conversation(const nlohmann::json& j) {
    ConversationView c;
    c.id = str(j, "id");
    c.title = str(j, "title");
    c.model_id = str(j, "model_id");
    c.summary_model_id = str(j, "summary_model_id");
    c.is_active = j.value("is_active", false);
    c.total_tokens = j.value("total_tokens", 0);
    c.summary_through_seq = j.value("summary_through_seq", -1);
    return c;
}

MemoryView parse_memory(const nlohmann::json& j) {
    MemoryView m;
    m.id = str(j, "id");
    m.content = str(j, "content");
    m.source_conv_id = str(j, "source_conv_id");
    m.importance = j.value("importance", 0.5f);
    return m;
}

NoteView parse_note(const nlohmann::json& j) {
    NoteView n;
    n.id = str(j, "id");
    n.content = str(j, "content");
    return n;
}

/// A conversation's notes: its local memories.
std::string notes_path(const std::string& user_id, const std::string& conv_id) {
    return "/v1/users/" + user_id + "/conversations/" + conv_id + "/local-memories";
}

std::string image_mime_for(const std::string& path) {
    const std::string ext = util::to_lower(std::filesystem::path(path).extension().string());
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    return {};
}

/// Column chrome shared by the three lists.
Element column(const std::string& title, Elements rows, bool focused) {
    if (rows.empty()) rows.push_back(text("  (none)") | dim);
    return window(text(" " + title + " ") | (focused ? bold : dim), vbox(std::move(rows)) | yframe);
}

} // namespace

// ── Transcript ────────────────────────────────────────────────────────────────

std::vector<TranscriptLine> transcript_of(const nlohmann::json& conversation) {
    std::vector<TranscriptLine> out;
    if (!conversation.is_object()) return out;
    for (const auto& m : conversation.value("messages", nlohmann::json::array())) {
        TranscriptLine line;
        line.role = str(m, "role");
        line.model_id = str(m, "model_id");
        line.text = str(m, "content");
        if (line.role == "user" && m.contains("content_parts")) {
            int images = 0;
            for (const auto& part : m["content_parts"])
                if (str(part, "type") != "text") ++images;
            if (images > 0) line.text += " [" + std::to_string(images) + " image(s)]";
        }
        if (line.role == "assistant" && m.contains("tool_calls")) {
            for (const auto& call : m["tool_calls"])
                line.text += (line.text.empty() ? "" : " ") + std::string("[called ") +
                             str(call, "function_name") + "]";
        }
        if (line.role == "tool") line.text = "[tool result] " + line.text;
        if (line.role == "system") continue;
        out.push_back(std::move(line));
    }
    return out;
}

// ── UserDashboard ─────────────────────────────────────────────────────────────

struct UserDashboard::Impl {
    std::string base_url;
    std::string api_token;

    mutable std::mutex mutex;
    UserSnapshot snap;
    std::string focus_user;
    std::string focus_conversation;

    std::atomic<bool> running{false};
    std::thread worker;
    std::mutex wake_mutex;
    std::condition_variable wake;
    std::atomic<bool> wake_now{false};

    HttpClient client() const {
        HttpClient c(base_url);
        if (!api_token.empty()) c.set_bearer_token(api_token);
        return c;
    }

    void poll() {
        UserSnapshot next;
        std::string user, conv;
        {
            std::lock_guard<std::mutex> g(mutex);
            user = focus_user;
            conv = focus_conversation;
        }
        auto c = client();
        const auto users = c.get("/v1/users");
        if (!users.ok()) {
            next.error = "GET /v1/users → " + describe_error_body(users.status, users.body);
        } else {
            next.reachable = true;
            try {
                for (const auto& u : nlohmann::json::parse(users.body).value("data", nlohmann::json::array()))
                    next.users.push_back(parse_user(u));
            } catch (const std::exception& e) {
                next.error = std::string("users parse: ") + e.what();
            }
        }
        if (const auto models = c.get("/v1/served-models"); models.ok()) {
            try {
                for (const auto& m : nlohmann::json::parse(models.body).value("data", nlohmann::json::array()))
                    if (m.value("available", false) && m.contains("id")) next.models.push_back(str(m, "id"));
            } catch (...) {
            }
        }

        // Keep a focus that still exists; otherwise the first user, and their
        // active conversation.
        const auto has_user = [&](const std::string& id) {
            return std::any_of(next.users.begin(), next.users.end(),
                               [&](const UserView& u) { return u.id == id; });
        };
        if (user.empty() || !has_user(user)) {
            user = next.users.empty() ? std::string{} : next.users.front().id;
            conv.clear();
        }
        next.user_id = user;
        if (!user.empty()) {
            if (const auto r = c.get("/v1/users/" + user + "/conversations"); r.ok()) {
                try {
                    for (const auto& j : nlohmann::json::parse(r.body).value("data", nlohmann::json::array()))
                        next.conversations.push_back(parse_conversation(j));
                } catch (...) {
                }
            }
            const bool conv_exists = std::any_of(next.conversations.begin(), next.conversations.end(),
                                                 [&](const ConversationView& cv) { return cv.id == conv; });
            if (conv.empty() || !conv_exists) {
                conv.clear();
                for (const auto& cv : next.conversations)
                    if (cv.is_active) conv = cv.id;
                if (conv.empty() && !next.conversations.empty()) conv = next.conversations.front().id;
            }
            if (const auto r = c.get("/v1/users/" + user + "/memories"); r.ok()) {
                try {
                    for (const auto& j : nlohmann::json::parse(r.body).value("data", nlohmann::json::array()))
                        next.memories.push_back(parse_memory(j));
                } catch (...) {
                }
            }
            if (const auto r = c.get("/v1/users/" + user + "/voice"); r.ok()) {
                next.voice = nlohmann::json::parse(r.body, nullptr, false);
                if (next.voice.is_discarded()) next.voice = nlohmann::json::object();
            }
            if (!conv.empty()) {
                if (const auto r = c.get("/v1/users/" + user + "/conversations/" + conv); r.ok())
                    next.transcript = transcript_of(nlohmann::json::parse(r.body, nullptr, false));
                if (const auto r = c.get(notes_path(user, conv)); r.ok()) {
                    try {
                        for (const auto& j : nlohmann::json::parse(r.body).value("data", nlohmann::json::array()))
                            next.notes.push_back(parse_note(j));
                    } catch (...) {
                    }
                }
            }
        }
        next.conversation_id = conv;

        std::lock_guard<std::mutex> g(mutex);
        // A focus the user changed while this poll ran wins over what it found.
        if (focus_user == user || focus_user.empty() || !has_user(focus_user)) {
            focus_user = user;
            if (focus_conversation.empty() || focus_conversation == conv || next.conversations.empty() ||
                !std::any_of(next.conversations.begin(), next.conversations.end(),
                             [&](const ConversationView& cv) { return cv.id == focus_conversation; }))
                focus_conversation = conv;
        }
        snap = std::move(next);
    }

    bool call(const HttpResponse& r, std::string& error) {
        if (r.ok()) {
            error.clear();
            wake_now.store(true);
            wake.notify_all();
            return true;
        }
        error = describe_error_body(r.status, r.body);
        return false;
    }
};

UserDashboard::UserDashboard(std::string base_url, std::string api_token)
    : impl_(std::make_unique<Impl>()) {
    impl_->base_url = std::move(base_url);
    impl_->api_token = std::move(api_token);
}

UserDashboard::~UserDashboard() { stop(); }

void UserDashboard::start(int interval_ms) {
    if (impl_->running.exchange(true)) return;
    impl_->worker = std::thread([this, interval_ms]() {
        while (impl_->running.load()) {
            impl_->poll();
            std::unique_lock<std::mutex> lk(impl_->wake_mutex);
            impl_->wake.wait_for(lk, std::chrono::milliseconds(interval_ms), [this] {
                return !impl_->running.load() || impl_->wake_now.exchange(false);
            });
        }
    });
}

void UserDashboard::stop() {
    if (!impl_->running.exchange(false)) return;
    impl_->wake.notify_all();
    if (impl_->worker.joinable()) impl_->worker.join();
}

void UserDashboard::refresh_now() {
    impl_->wake_now.store(true);
    impl_->wake.notify_all();
}

UserSnapshot UserDashboard::snapshot() const {
    std::lock_guard<std::mutex> g(impl_->mutex);
    return impl_->snap;
}

void UserDashboard::focus(const std::string& user_id, const std::string& conversation_id) {
    {
        std::lock_guard<std::mutex> g(impl_->mutex);
        impl_->focus_user = user_id;
        impl_->focus_conversation = conversation_id;
    }
    refresh_now();
}

std::string UserDashboard::focused_user() const {
    std::lock_guard<std::mutex> g(impl_->mutex);
    return impl_->focus_user;
}

std::string UserDashboard::focused_conversation() const {
    std::lock_guard<std::mutex> g(impl_->mutex);
    return impl_->focus_conversation;
}

bool UserDashboard::create_user(const nlohmann::json& body, std::string& error) {
    return impl_->call(impl_->client().post("/v1/users", body), error);
}

bool UserDashboard::update_user(const std::string& id, const nlohmann::json& body, std::string& error) {
    return impl_->call(impl_->client().put("/v1/users/" + id, body), error);
}

bool UserDashboard::delete_user(const std::string& id, std::string& error) {
    return impl_->call(impl_->client().del("/v1/users/" + id), error);
}

bool UserDashboard::create_conversation(const std::string& user_id, const nlohmann::json& body,
                                        std::string& error) {
    return impl_->call(impl_->client().post("/v1/users/" + user_id + "/conversations", body), error);
}

bool UserDashboard::update_conversation(const std::string& user_id, const std::string& conv_id,
                                        const nlohmann::json& body, std::string& error) {
    return impl_->call(impl_->client().put("/v1/users/" + user_id + "/conversations/" + conv_id, body),
                       error);
}

bool UserDashboard::conversation_action(const std::string& user_id, const std::string& conv_id,
                                        const std::string& verb, std::string& error) {
    const std::string base = "/v1/users/" + user_id + "/conversations/" + conv_id;
    auto c = impl_->client();
    if (verb == "delete") return impl_->call(c.del(base), error);
    if (verb != "activate" && verb != "compact") {
        error = "unknown action " + verb;
        return false;
    }
    // Compaction runs the model, possibly after placing it: minutes, not ms.
    if (verb == "compact") c.set_timeouts(10, 3600, 60);
    return impl_->call(c.post(base + "/" + verb, nlohmann::json::object()), error);
}

bool UserDashboard::add_memory(const std::string& user_id, const nlohmann::json& body,
                               std::string& error) {
    return impl_->call(impl_->client().post("/v1/users/" + user_id + "/memories", body), error);
}

bool UserDashboard::update_memory(const std::string& user_id, const std::string& memory_id,
                                  const nlohmann::json& body, std::string& error) {
    return impl_->call(impl_->client().put("/v1/users/" + user_id + "/memories/" + memory_id, body), error);
}

bool UserDashboard::delete_memory(const std::string& user_id, const std::string& memory_id,
                                  std::string& error) {
    return impl_->call(impl_->client().del("/v1/users/" + user_id + "/memories/" + memory_id), error);
}

bool UserDashboard::extract_memories(const std::string& user_id, const nlohmann::json& body,
                                     std::string& error) {
    return impl_->call(impl_->client().post("/v1/users/" + user_id + "/memories/extract", body), error);
}

bool UserDashboard::add_note(const std::string& user_id, const std::string& conv_id,
                             const std::string& content, std::string& error) {
    return impl_->call(impl_->client().post(notes_path(user_id, conv_id), nlohmann::json{{"content", content}}),
                       error);
}

bool UserDashboard::update_note(const std::string& user_id, const std::string& conv_id,
                                const std::string& note_id, const std::string& content,
                                std::string& error) {
    return impl_->call(impl_->client().put(notes_path(user_id, conv_id) + "/" + note_id,
                                           nlohmann::json{{"content", content}}),
                       error);
}

bool UserDashboard::delete_note(const std::string& user_id, const std::string& conv_id,
                                const std::string& note_id, std::string& error) {
    return impl_->call(impl_->client().del(notes_path(user_id, conv_id) + "/" + note_id), error);
}

bool UserDashboard::propose_curation(const std::string& user_id, const std::string& conv_id,
                                     nlohmann::json& proposals, std::string& error) {
    nlohmann::json body = nlohmann::json::object();
    if (!conv_id.empty()) body["conversation_id"] = conv_id;
    const auto r = impl_->client().post("/v1/users/" + user_id + "/curation/proposals", body);
    if (!impl_->call(r, error)) return false;
    proposals = nlohmann::json::parse(r.body, nullptr, false).value("proposals", nlohmann::json::array());
    return true;
}

bool UserDashboard::apply_curation(const std::string& user_id, const nlohmann::json& proposals,
                                   std::string& error) {
    return impl_->call(impl_->client().post("/v1/users/" + user_id + "/curation/apply",
                                            nlohmann::json{{"proposals", proposals}}),
                       error);
}

UserDashboard::ChatResult UserDashboard::chat(const std::string& user_id,
                                              const std::string& conv_id,
                                              const std::string& message,
                                              const std::string& model,
                                              const std::string& image_path,
                                              const std::function<void(const std::string&)>& on_delta,
                                              const std::atomic<bool>* cancel) {
    ChatResult result;
    auto c = impl_->client();
    nlohmann::json body{{"message", message}};
    if (!conv_id.empty()) body["conversation_id"] = conv_id;
    if (!model.empty()) body["model"] = model;
    if (!util::trim(image_path).empty()) {
        const std::string path = util::trim(image_path);
        const std::string mime = image_mime_for(path);
        if (mime.empty()) {
            result.error = "only .png, .jpg and .jpeg images can be attached";
            return result;
        }
        const auto upload = c.post_file("/v1/users/" + user_id + "/attachments", path,
                                        {{"X-Filename", std::filesystem::path(path).filename().string()}},
                                        mime);
        if (!upload.ok()) {
            result.error = "upload failed: " + describe_error_body(upload.status, upload.body);
            return result;
        }
        const auto attachment = nlohmann::json::parse(upload.body, nullptr, false);
        body["attachment_ids"] = nlohmann::json::array({str(attachment, "id")});
    }

    bool done_seen = false;
    int status = 0;
    std::string error_body;
    c.stream_post("/v1/users/" + user_id + "/chat", body,
                  [&](const std::string& data) -> bool {
                      if (data == "[DONE]") return true;
                      const auto j = nlohmann::json::parse(data, nullptr, false);
                      if (!j.is_object()) return true;
                      const std::string type = str(j, "type");
                      if (type == "delta" && on_delta) on_delta(str(j, "content"));
                      if (type == "done") {
                          done_seen = true;
                          result.ok = j.value("success", false);
                          result.conv_id = str(j, "conv_id");
                          result.model_id = str(j, "model_id");
                          result.error = str(j, "error");
                          result.code = str(j, "code");
                      }
                      return cancel == nullptr || !cancel->load();
                  },
                  &status, &error_body);
    if (!done_seen && result.error.empty()) {
        result.error = status > 0 ? describe_error_body(status, error_body)
                                  : std::string("the chat stream ended without an answer");
    }
    impl_->wake_now.store(true);
    impl_->wake.notify_all();
    return result;
}

bool UserDashboard::voice_propose(const std::string& user_id, const nlohmann::json& body,
                                  std::string& error) {
    auto c = impl_->client();
    c.set_timeouts(10, 600, 60); // an inferred proposal runs the model
    return impl_->call(c.post("/v1/users/" + user_id + "/voice/proposals", body), error);
}

bool UserDashboard::voice_proposal_action(const std::string& user_id,
                                          const std::string& proposal_id,
                                          const std::string& verb,
                                          nlohmann::json& response,
                                          std::string& error) {
    if (verb != "sample" && verb != "approve" && verb != "reject") {
        error = "unknown action " + verb;
        return false;
    }
    auto c = impl_->client();
    c.set_timeouts(10, 600, 60);
    const auto r = c.post("/v1/users/" + user_id + "/voice/proposals/" + proposal_id + "/" + verb,
                          nlohmann::json::object());
    if (!impl_->call(r, error)) return false;
    response = nlohmann::json::parse(r.body, nullptr, false);
    return true;
}

bool UserDashboard::speak(const std::string& user_id, const std::string& text,
                          nlohmann::json& response, std::string& error) {
    auto c = impl_->client();
    c.set_timeouts(10, 600, 60);
    const auto r = c.post("/v1/users/" + user_id + "/speech",
                          nlohmann::json{{"text", text}, {"format", "wav"}});
    if (!impl_->call(r, error)) return false;
    response = nlohmann::json::parse(r.body, nullptr, false);
    return true;
}

// ── Renderers ─────────────────────────────────────────────────────────────────

Element render_user_list(const UserSnapshot& snap, int selected, bool focused) {
    Elements rows;
    for (std::size_t i = 0; i < snap.users.size(); ++i) {
        const auto& u = snap.users[i];
        Element row = hbox({text(u.id == snap.user_id ? "● " : "  ") | color(Color::Green),
                            text(truncate(u.id, 18)) | bold, text(" "),
                            text(truncate(u.default_model.empty() ? "(no default)" : u.default_model, 20)) | dim});
        if (static_cast<int>(i) == selected && focused) row = std::move(row) | inverted;
        rows.push_back(std::move(row));
    }
    if (!snap.reachable) rows.push_back(text("  " + snap.error) | color(Color::Red));
    return column("USERS", std::move(rows), focused);
}

Element render_conversation_list(const UserSnapshot& snap, int selected, bool focused) {
    Elements rows;
    for (std::size_t i = 0; i < snap.conversations.size(); ++i) {
        const auto& c = snap.conversations[i];
        Element row = hbox({text(c.is_active ? "▶ " : "  ") | color(Color::Cyan),
                            text(truncate(one_line(c.title), 26)),
                            filler(),
                            text(" " + truncate(c.model_id.empty() ? "(default)" : c.model_id, 16)) | dim,
                            text(c.summary_through_seq >= 0 ? " Σ" : "  ") | color(Color::Yellow)});
        if (c.id == snap.conversation_id) row = std::move(row) | bold;
        if (static_cast<int>(i) == selected && focused) row = std::move(row) | inverted;
        rows.push_back(std::move(row));
    }
    return column("CONVERSATIONS", std::move(rows), focused);
}

Element render_memory_list(const UserSnapshot& snap, int selected, bool focused) {
    Elements rows;
    for (std::size_t i = 0; i < snap.memories.size(); ++i) {
        const auto& m = snap.memories[i];
        char imp[16];
        std::snprintf(imp, sizeof(imp), "%.2f ", m.importance);
        Element row = hbox({text(imp) | dim, paragraph(truncate(one_line(m.content), 120))});
        if (static_cast<int>(i) == selected && focused) row = std::move(row) | inverted;
        rows.push_back(std::move(row));
    }
    return column("GLOBAL MEMORIES", std::move(rows), focused);
}

Element render_note_list(const UserSnapshot& snap, int selected, bool focused) {
    Elements rows;
    for (std::size_t i = 0; i < snap.notes.size(); ++i) {
        Element row = paragraph(" " + truncate(one_line(snap.notes[i].content), 160));
        if (static_cast<int>(i) == selected && focused) row = std::move(row) | inverted;
        rows.push_back(std::move(row));
    }
    return column("NOTES · this conversation", std::move(rows), focused);
}

Element render_transcript(const std::vector<TranscriptLine>& lines, const std::string& streaming) {
    Elements rows;
    for (const auto& line : lines) {
        if (line.role == "user") {
            rows.push_back(hbox({text("you  ") | bold | color(Color::Cyan), paragraph(line.text)}));
        } else if (line.role == "assistant") {
            rows.push_back(hbox({text(truncate(line.model_id.empty() ? "model" : line.model_id, 18)) |
                                     color(Color::Green) | size(WIDTH, EQUAL, 19),
                                 paragraph(line.text)}));
        } else {
            rows.push_back(paragraph("  " + line.text) | dim);
        }
    }
    if (!streaming.empty())
        rows.push_back(hbox({text("…    ") | color(Color::Yellow), paragraph(streaming)}));
    if (rows.empty()) rows.push_back(text("  (no messages yet)") | dim);
    return vbox(std::move(rows)) | focusPositionRelative(0, 1) | yframe;
}

// ── Shared tab plumbing ───────────────────────────────────────────────────────

namespace {

/// Status line and one-at-a-time background actions, as every tab needs.
struct Runner {
    std::mutex mu;
    std::string status;
    bool ok = true;
    std::atomic<bool> busy{false};
    std::thread worker;

    void set(std::string s, bool good) {
        std::lock_guard<std::mutex> lk(mu);
        status = std::move(s);
        ok = good;
    }
    std::pair<std::string, bool> get() {
        std::lock_guard<std::mutex> lk(mu);
        return {status, ok};
    }
    /// Actions can place a model or run it, so they run off the UI thread —
    /// one at a time: a second press while one runs is ignored, not queued.
    void run(std::function<void()> fn) {
        if (busy.exchange(true)) return;
        if (worker.joinable()) worker.join();
        worker = std::thread([this, fn = std::move(fn)] {
            fn();
            busy.store(false);
        });
    }
    Element render() {
        const auto [s, good] = get();
        if (s.empty() && !busy.load()) return text("");
        return text(" " + s + (busy.load() ? "  (working…)" : "")) | color(good ? Color::Green : Color::Red);
    }
    ~Runner() {
        if (worker.joinable()) worker.join();
    }
};

/// A small form: labelled fields, text or on/off, typed into Inputs that are
/// bound once and never move.
struct Form {
    struct Field {
        std::string key;
        std::string label;
        bool flag = false; ///< true/false field
        std::string value;
        bool used = false;
    };
    static constexpr std::size_t kSlots = 6;
    std::vector<Field> fields = std::vector<Field>(kSlots);
    bool open = false;
    std::string title;
    std::function<void(const nlohmann::json&)> submit;
    /// Keys sent as JSON numbers rather than strings; blank leaves them out.
    std::vector<std::string> numeric;

    void begin(std::string heading, std::vector<Field> spec, std::function<void(const nlohmann::json&)> on_submit,
               std::vector<std::string> numeric_keys = {}) {
        for (std::size_t i = 0; i < kSlots; ++i) {
            fields[i] = i < spec.size() ? spec[i] : Field{};
            fields[i].used = i < spec.size();
        }
        title = std::move(heading);
        submit = std::move(on_submit);
        numeric = std::move(numeric_keys);
        open = true;
    }
    /// The fields as a JSON body; false with `error` for a bad on/off value or
    /// number.
    bool body(nlohmann::json& out, std::string& error) const {
        out = nlohmann::json::object();
        for (const auto& f : fields) {
            if (!f.used) continue;
            const std::string v = util::trim(f.value);
            if (std::find(numeric.begin(), numeric.end(), f.key) != numeric.end()) {
                if (v.empty()) continue;
                try {
                    std::size_t used = 0;
                    const long long whole = std::stoll(v, &used);
                    if (used == v.size()) {
                        out[f.key] = whole;
                        continue;
                    }
                    const double real = std::stod(v, &used);
                    if (used == v.size()) {
                        out[f.key] = real;
                        continue;
                    }
                } catch (const std::exception&) {
                }
                error = f.label + ": expected a number";
                return false;
            }
            if (f.flag) {
                const std::string b = util::to_lower(v);
                if (b == "true" || b == "yes" || b == "on" || b == "1") out[f.key] = true;
                else if (b == "false" || b == "no" || b == "off" || b == "0") out[f.key] = false;
                else {
                    error = f.label + ": expected true or false";
                    return false;
                }
            } else {
                out[f.key] = v;
            }
        }
        return true;
    }
};

Components form_inputs(const std::shared_ptr<Form>& form) {
    Components inputs;
    for (std::size_t i = 0; i < Form::kSlots; ++i) {
        InputOption opt;
        opt.multiline = false;
        inputs.push_back(Maybe(Input(&form->fields[i].value, opt), [form, i] { return form->fields[i].used; }));
    }
    return inputs;
}

Element render_form(const Form& form, const Components& inputs, const Component& save, const Component& close) {
    Elements rows{text(" " + form.title) | bold, separator()};
    for (std::size_t i = 0; i < Form::kSlots; ++i) {
        if (!form.fields[i].used) continue;
        rows.push_back(hbox({text(" " + form.fields[i].label) | dim | size(WIDTH, EQUAL, 24),
                             inputs[i]->Render() | flex}));
    }
    rows.push_back(hbox({save->Render(), text("  "), close->Render()}));
    return vbox(std::move(rows)) | border;
}

int clamp_index(int i, std::size_t n) {
    if (n == 0) return 0;
    return std::clamp(i, 0, static_cast<int>(n) - 1);
}

} // namespace

// ── Users ─────────────────────────────────────────────────────────────────────

Component users_tab(UserDashboard& dashboard) {
    struct State {
        int column = 0; ///< 0 users, 1 conversations, 2 notes, 3 memories
        int user_sel = 0;
        int conv_sel = 0;
        int note_sel = 0;
        int mem_sel = 0;
        std::string pending_delete;
        nlohmann::json proposals = nlohmann::json::array();
        std::shared_ptr<Form> form = std::make_shared<Form>();
        Runner runner;
    };
    auto st = std::make_shared<State>();
    auto inputs = form_inputs(st->form);
    auto save = Button("Save", [st] {
        nlohmann::json body;
        std::string error;
        if (!st->form->body(body, error)) {
            st->runner.set(error, false);
            return;
        }
        auto submit = st->form->submit;
        st->runner.run([st, submit, body] { submit(body); });
    }, ButtonOption::Simple());
    auto close = Button("Close", [st] { st->form->open = false; }, ButtonOption::Simple());
    Components rows = inputs;
    rows.push_back(Container::Horizontal({save, close}));
    auto form_container = Maybe(Container::Vertical(rows), &st->form->open);

    const auto selected_user = [&dashboard, st]() -> std::optional<UserView> {
        const auto snap = dashboard.snapshot();
        if (snap.users.empty()) return std::nullopt;
        return snap.users[static_cast<std::size_t>(clamp_index(st->user_sel, snap.users.size()))];
    };
    const auto selected_conversation = [&dashboard, st]() -> std::optional<ConversationView> {
        const auto snap = dashboard.snapshot();
        if (snap.conversations.empty()) return std::nullopt;
        return snap.conversations[static_cast<std::size_t>(clamp_index(st->conv_sel, snap.conversations.size()))];
    };
    const auto done = [st](bool ok, const std::string& what, const std::string& error) {
        st->runner.set(ok ? what : error, ok);
        if (ok) st->form->open = false;
    };

    auto renderer = Renderer(form_container, [&dashboard, st, inputs, save, close]() -> Element {
        const auto snap = dashboard.snapshot();
        // Follow the shared focus: another tab may have moved it.
        for (std::size_t i = 0; i < snap.users.size(); ++i)
            if (snap.users[i].id == snap.user_id && st->column != 0) st->user_sel = static_cast<int>(i);
        st->user_sel = clamp_index(st->user_sel, snap.users.size());
        st->conv_sel = clamp_index(st->conv_sel, snap.conversations.size());
        st->note_sel = clamp_index(st->note_sel, snap.notes.size());
        st->mem_sel = clamp_index(st->mem_sel, snap.memories.size());

        Element profile = text("");
        if (!snap.users.empty()) {
            const auto& u = snap.users[static_cast<std::size_t>(st->user_sel)];
            profile = window(text(" PROFILE "), vbox({
                hbox({text(" id       ") | dim, text(u.id)}),
                hbox({text(" name     ") | dim, text(u.display_name)}),
                hbox({text(" default  ") | dim, text(u.default_model.empty() ? "(none)" : u.default_model)}),
                hbox({text(" memories ") | dim, text(u.memories_enabled ? "on" : "off"),
                      text("   tools ") | dim, text(u.tools_enabled ? "on" : "off")}),
                paragraph(" " + truncate(one_line(u.custom_instructions), 240)) | dim,
            }));
        }
        Elements body{
            hbox({render_user_list(snap, st->user_sel, st->column == 0) | size(WIDTH, EQUAL, 44),
                  vbox({render_conversation_list(snap, st->conv_sel, st->column == 1) | flex,
                        render_note_list(snap, st->note_sel, st->column == 2) | size(HEIGHT, LESS_THAN, 12)}) |
                      flex,
                  vbox({profile, render_memory_list(snap, st->mem_sel, st->column == 3) | flex}) |
                      size(WIDTH, EQUAL, 56)}) |
                flex,
            text(" ←/→ column · ↑/↓ select · n new · e edit · x delete (twice) · a activate · "
                 "c compact · m extract memories · p propose curation · y apply proposals") | dim,
        };
        if (!st->proposals.empty())
            body.push_back(text(" " + std::to_string(st->proposals.size()) +
                                " curation proposal(s) ready — y applies them all") | color(Color::Yellow));
        body.push_back(st->runner.render());
        if (st->form->open) body.push_back(render_form(*st->form, inputs, save, close));
        return vbox(std::move(body));
    });

    return CatchEvent(renderer, [&dashboard, st, selected_user, selected_conversation, done](const Event& ev) {
        if (st->form->open) return false;
        const auto snap = dashboard.snapshot();
        const auto refocus = [&] {
            const auto u = selected_user();
            const auto c = selected_conversation();
            dashboard.focus(u ? u->id : std::string{}, st->column >= 1 && c ? c->id : std::string{});
        };
        if (ev == Event::ArrowLeft || ev == Event::Character('h')) {
            st->column = std::max(0, st->column - 1);
            st->pending_delete.clear();
            return true;
        }
        if (ev == Event::ArrowRight || ev == Event::Character('l')) {
            st->column = std::min(3, st->column + 1);
            st->pending_delete.clear();
            return true;
        }
        const int delta = (ev == Event::ArrowDown || ev == Event::Character('j'))  ? 1
                          : (ev == Event::ArrowUp || ev == Event::Character('k')) ? -1
                                                                                  : 0;
        if (delta != 0) {
            st->pending_delete.clear();
            if (st->column == 0) {
                st->user_sel = clamp_index(st->user_sel + delta, snap.users.size());
                st->conv_sel = 0;
                st->proposals = nlohmann::json::array();
                const auto u = selected_user();
                dashboard.focus(u ? u->id : std::string{}, {});
            } else if (st->column == 1) {
                st->conv_sel = clamp_index(st->conv_sel + delta, snap.conversations.size());
                st->note_sel = 0;
                refocus();
            } else if (st->column == 2) {
                st->note_sel = clamp_index(st->note_sel + delta, snap.notes.size());
            } else {
                st->mem_sel = clamp_index(st->mem_sel + delta, snap.memories.size());
            }
            return true;
        }

        const auto user = selected_user();
        if (ev == Event::Character('n')) {
            if (st->column == 0) {
                st->form->begin("New user",
                                {{"id", "id (lowercase)", false, "", true},
                                 {"display_name", "display name", false, "", true},
                                 {"default_model", "default model", false,
                                  snap.models.empty() ? "" : snap.models.front(), true},
                                 {"custom_instructions", "instructions", false, "", true},
                                 {"memories_enabled", "memories (true/false)", true, "true", true},
                                 {"tools_enabled", "tools (true/false)", true, "false", true}},
                                [&dashboard, done](const nlohmann::json& body) {
                                    std::string err;
                                    done(dashboard.create_user(body, err), "created " + body.value("id", std::string{}), err);
                                });
            } else if (st->column == 1 && user) {
                const std::string uid = user->id;
                st->form->begin("New conversation for " + uid,
                                {{"title", "title", false, "", true},
                                 {"model", "model (blank = default)", false, "", true},
                                 {"system_prompt", "system prompt", false, "", true},
                                 {"set_active", "make active (true/false)", true, "true", true}},
                                [&dashboard, done, uid](const nlohmann::json& body) {
                                    std::string err;
                                    done(dashboard.create_conversation(uid, body, err), "conversation created", err);
                                });
            } else if (st->column == 2 && user && !snap.conversation_id.empty()) {
                const std::string uid = user->id, cid = snap.conversation_id;
                st->form->begin("New note in this conversation", {{"content", "note", false, "", true}},
                                [&dashboard, done, uid, cid](const nlohmann::json& body) {
                                    std::string err;
                                    done(dashboard.add_note(uid, cid, body.value("content", std::string{}), err),
                                         "note added", err);
                                });
            } else if (st->column == 3 && user) {
                const std::string uid = user->id;
                st->form->begin("New global memory for " + uid,
                                {{"content", "memory", false, "", true},
                                 {"importance", "importance (0-1)", false, "0.5", true}},
                                [&dashboard, done, uid](const nlohmann::json& body) {
                                    std::string err;
                                    done(dashboard.add_memory(uid, body, err), "memory added", err);
                                },
                                {"importance"});
            }
            return true;
        }
        if (ev == Event::Character('e')) {
            if (st->column == 0 && user) {
                const std::string uid = user->id;
                st->form->begin("Edit " + uid,
                                {{"display_name", "display name", false, user->display_name, true},
                                 {"default_model", "default model", false, user->default_model, true},
                                 {"custom_instructions", "instructions", false, user->custom_instructions, true},
                                 {"memories_enabled", "memories (true/false)", true,
                                  user->memories_enabled ? "true" : "false", true},
                                 {"tools_enabled", "tools (true/false)", true, user->tools_enabled ? "true" : "false",
                                  true}},
                                [&dashboard, done, uid](const nlohmann::json& body) {
                                    std::string err;
                                    done(dashboard.update_user(uid, body, err), "saved " + uid, err);
                                });
            } else if (st->column == 1 && user) {
                if (const auto conv = selected_conversation()) {
                    const std::string uid = user->id;
                    const std::string cid = conv->id;
                    st->form->begin("Edit conversation",
                                    {{"title", "title", false, conv->title, true},
                                     {"model", "model (blank = default)", false, conv->model_id, true}},
                                    [&dashboard, done, uid, cid](const nlohmann::json& body) {
                                        std::string err;
                                        done(dashboard.update_conversation(uid, cid, body, err), "saved", err);
                                    });
                }
            } else if (st->column == 2 && user && !snap.notes.empty() && !snap.conversation_id.empty()) {
                const auto& note = snap.notes[static_cast<std::size_t>(clamp_index(st->note_sel, snap.notes.size()))];
                const std::string uid = user->id, cid = snap.conversation_id, nid = note.id;
                st->form->begin("Edit note", {{"content", "note", false, note.content, true}},
                                [&dashboard, done, uid, cid, nid](const nlohmann::json& body) {
                                    std::string err;
                                    done(dashboard.update_note(uid, cid, nid, body.value("content", std::string{}), err),
                                         "note saved", err);
                                });
            } else if (st->column == 3 && user && !snap.memories.empty()) {
                const auto& mem = snap.memories[static_cast<std::size_t>(clamp_index(st->mem_sel, snap.memories.size()))];
                char importance[16];
                std::snprintf(importance, sizeof(importance), "%.2f", mem.importance);
                const std::string uid = user->id, mid = mem.id;
                st->form->begin("Edit global memory",
                                {{"content", "memory", false, mem.content, true},
                                 {"importance", "importance (0-1)", false, importance, true}},
                                [&dashboard, done, uid, mid](const nlohmann::json& body) {
                                    std::string err;
                                    done(dashboard.update_memory(uid, mid, body, err), "memory saved", err);
                                },
                                {"importance"});
            }
            return true;
        }
        if (ev == Event::Character('x') && user) {
            std::string target, label;
            std::function<bool(std::string&)> act;
            if (st->column == 0) {
                target = "user:" + user->id;
                label = "user " + user->id + " and everything they own";
                const std::string uid = user->id;
                act = [&dashboard, uid](std::string& err) { return dashboard.delete_user(uid, err); };
            } else if (st->column == 1) {
                if (const auto conv = selected_conversation()) {
                    target = "conv:" + conv->id;
                    label = "conversation \"" + conv->title + "\"";
                    const std::string uid = user->id, cid = conv->id;
                    act = [&dashboard, uid, cid](std::string& err) {
                        return dashboard.conversation_action(uid, cid, "delete", err);
                    };
                }
            } else if (st->column == 2) {
                if (!snap.notes.empty() && !snap.conversation_id.empty()) {
                    const auto& note = snap.notes[static_cast<std::size_t>(clamp_index(st->note_sel, snap.notes.size()))];
                    target = "note:" + note.id;
                    label = "note \"" + truncate(one_line(note.content), 40) + "\"";
                    const std::string uid = user->id, cid = snap.conversation_id, nid = note.id;
                    act = [&dashboard, uid, cid, nid](std::string& err) {
                        return dashboard.delete_note(uid, cid, nid, err);
                    };
                }
            } else if (!snap.memories.empty()) {
                const auto& mem = snap.memories[static_cast<std::size_t>(clamp_index(st->mem_sel, snap.memories.size()))];
                target = "mem:" + mem.id;
                label = "memory \"" + truncate(one_line(mem.content), 40) + "\"";
                const std::string uid = user->id, mid = mem.id;
                act = [&dashboard, uid, mid](std::string& err) { return dashboard.delete_memory(uid, mid, err); };
            }
            if (!act) return true;
            // Two presses: none of these can be undone from the API.
            if (st->pending_delete != target) {
                st->pending_delete = target;
                st->runner.set("press x again to delete " + label, false);
                return true;
            }
            st->pending_delete.clear();
            st->runner.run([st, act, label] {
                std::string err;
                if (act(err)) st->runner.set("deleted " + label, true);
                else st->runner.set(err, false);
            });
            return true;
        }
        if ((ev == Event::Character('a') || ev == Event::Character('c')) && user) {
            if (const auto conv = selected_conversation()) {
                const std::string verb = ev == Event::Character('a') ? "activate" : "compact";
                const std::string uid = user->id, cid = conv->id;
                st->runner.set(verb + "…", true);
                st->runner.run([st, &dashboard, verb, uid, cid] {
                    std::string err;
                    if (dashboard.conversation_action(uid, cid, verb, err)) st->runner.set(verb + ": done", true);
                    else st->runner.set(err, false);
                });
            }
            return true;
        }
        // Extraction reviews a range of the focused conversation for global
        // memories, with a model: the one named here, else the conversation's.
        if (ev == Event::Character('m') && user && !snap.conversation_id.empty()) {
            const std::string uid = user->id, cid = snap.conversation_id;
            const int last = std::max(0, static_cast<int>(snap.transcript.size()) - 1);
            st->form->begin("Extract global memories from this conversation",
                            {{"start_index", "first message #", false, "0", true},
                             {"end_index", "last message #", false, std::to_string(last), true},
                             {"context_before", "context before", false, "2", true},
                             {"model", "model (blank = the conversation's)", false, "", true}},
                            [&dashboard, done, uid, cid](const nlohmann::json& body) {
                                nlohmann::json request = body;
                                request["conversation_id"] = cid;
                                std::string err;
                                done(dashboard.extract_memories(uid, request, err), "extraction queued", err);
                            },
                            {"start_index", "end_index", "context_before"});
            return true;
        }
        if (ev == Event::Character('p') && user) {
            const auto conv = selected_conversation();
            const std::string uid = user->id, cid = conv ? conv->id : std::string{};
            st->runner.run([st, &dashboard, uid, cid] {
                nlohmann::json proposals;
                std::string err;
                if (dashboard.propose_curation(uid, cid, proposals, err)) {
                    st->proposals = proposals;
                    st->runner.set(std::to_string(proposals.size()) + " proposal(s)", true);
                } else {
                    st->runner.set(err, false);
                }
            });
            return true;
        }
        if (ev == Event::Character('y') && user && !st->proposals.empty()) {
            const std::string uid = user->id;
            const nlohmann::json proposals = st->proposals;
            st->runner.run([st, &dashboard, uid, proposals] {
                std::string err;
                if (dashboard.apply_curation(uid, proposals, err)) {
                    st->proposals = nlohmann::json::array();
                    st->runner.set("applied " + std::to_string(proposals.size()) + " proposal(s)", true);
                } else {
                    st->runner.set(err, false);
                }
            });
            return true;
        }
        return false;
    });
}

// ── Chat ──────────────────────────────────────────────────────────────────────

Component chat_tab(UserDashboard& dashboard, const std::atomic<bool>& shutting_down) {
    struct State {
        std::string input;
        std::string image_path;
        int model_index = 0; ///< 0: the conversation's (or the user's default)
        std::mutex stream_mu;
        std::string streaming;
        Runner runner;
    };
    auto st = std::make_shared<State>();

    const auto chosen_model = [&dashboard, st]() -> std::string {
        const auto snap = dashboard.snapshot();
        if (st->model_index <= 0 || st->model_index > static_cast<int>(snap.models.size())) return {};
        return snap.models[static_cast<std::size_t>(st->model_index - 1)];
    };
    const auto send = [&dashboard, st, chosen_model, &shutting_down] {
        const std::string message = util::trim(st->input);
        const std::string image = util::trim(st->image_path);
        if (message.empty() && image.empty()) return;
        const std::string user = dashboard.focused_user();
        if (user.empty()) {
            st->runner.set("no user — create one in the Users tab", false);
            return;
        }
        const std::string conv = dashboard.focused_conversation();
        const std::string model = chosen_model();
        st->input.clear();
        st->image_path.clear();
        st->runner.set("thinking…", true);
        st->runner.run([st, &dashboard, &shutting_down, user, conv, model, message, image] {
            const auto result = dashboard.chat(
                user, conv, message, model, image,
                [st](const std::string& delta) {
                    std::lock_guard<std::mutex> lk(st->stream_mu);
                    st->streaming += delta;
                },
                &shutting_down);
            {
                std::lock_guard<std::mutex> lk(st->stream_mu);
                st->streaming.clear();
            }
            if (result.ok) {
                st->runner.set("answered by " + result.model_id, true);
                dashboard.focus(user, result.conv_id);
            } else {
                st->runner.set(result.error + (result.code.empty() ? "" : "  [" + result.code + "]"), false);
            }
        });
    };

    InputOption message_opt;
    message_opt.multiline = false;
    message_opt.placeholder = "Message — Enter sends";
    message_opt.on_enter = send;
    auto message_input = Input(&st->input, message_opt);
    InputOption image_opt;
    image_opt.multiline = false;
    image_opt.placeholder = "Image path to attach (optional, .png/.jpg)";
    auto image_input = Input(&st->image_path, image_opt);

    const auto step_conversation = [&dashboard](int delta) {
        const auto snap = dashboard.snapshot();
        if (snap.conversations.empty()) return;
        int index = 0;
        for (std::size_t i = 0; i < snap.conversations.size(); ++i)
            if (snap.conversations[i].id == snap.conversation_id) index = static_cast<int>(i);
        index = clamp_index(index + delta, snap.conversations.size());
        dashboard.focus(snap.user_id, snap.conversations[static_cast<std::size_t>(index)].id);
    };
    const auto step_user = [&dashboard](int delta) {
        const auto snap = dashboard.snapshot();
        if (snap.users.empty()) return;
        int index = 0;
        for (std::size_t i = 0; i < snap.users.size(); ++i)
            if (snap.users[i].id == snap.user_id) index = static_cast<int>(i);
        index = clamp_index(index + delta, snap.users.size());
        dashboard.focus(snap.users[static_cast<std::size_t>(index)].id, {});
    };
    auto prev_user = Button("◄ user", [step_user] { step_user(-1); }, ButtonOption::Simple());
    auto next_user = Button("user ►", [step_user] { step_user(1); }, ButtonOption::Simple());
    auto prev_conv = Button("◄ conv", [step_conversation] { step_conversation(-1); }, ButtonOption::Simple());
    auto next_conv = Button("conv ►", [step_conversation] { step_conversation(1); }, ButtonOption::Simple());
    auto new_conv = Button("new conv", [&dashboard, st] {
        const std::string user = dashboard.focused_user();
        if (user.empty()) return;
        st->runner.run([st, &dashboard, user] {
            std::string err;
            if (dashboard.create_conversation(user, nlohmann::json{{"set_active", true}}, err)) {
                dashboard.focus(user, {});
                st->runner.set("new conversation", true);
            } else {
                st->runner.set(err, false);
            }
        });
    }, ButtonOption::Simple());
    auto cycle_model = Button("model ▸", [&dashboard, st] {
        const auto snap = dashboard.snapshot();
        st->model_index = (st->model_index + 1) % (static_cast<int>(snap.models.size()) + 1);
    }, ButtonOption::Simple());
    auto send_button = Button("Send", send, ButtonOption::Simple());
    auto controls = Container::Horizontal({prev_user, next_user, prev_conv, next_conv, new_conv, cycle_model, send_button});
    auto container = Container::Vertical({message_input, image_input, controls});

    return Renderer(container, [&dashboard, st, chosen_model, message_input, image_input, controls]() -> Element {
        const auto snap = dashboard.snapshot();
        std::string streaming;
        {
            std::lock_guard<std::mutex> lk(st->stream_mu);
            streaming = st->streaming;
        }
        std::string conv_label = "(new conversation)";
        std::string conv_model;
        for (const auto& c : snap.conversations) {
            if (c.id != snap.conversation_id) continue;
            conv_label = c.title;
            conv_model = c.model_id;
        }
        std::string default_model;
        for (const auto& u : snap.users)
            if (u.id == snap.user_id) default_model = u.default_model;
        const std::string model = chosen_model();
        const std::string model_label =
            !model.empty() ? model + " (switches this conversation)"
                           : (!conv_model.empty() ? conv_model
                                                  : (default_model.empty() ? "(none — pick one)" : default_model + " (user default)"));
        return vbox({
            hbox({text(" user ") | dim, text(snap.user_id.empty() ? "(none)" : snap.user_id) | bold,
                  text("   conversation ") | dim, text(truncate(one_line(conv_label), 40)),
                  text("   model ") | dim, text(model_label) | color(Color::Green)}),
            separator(),
            render_transcript(snap.transcript, streaming) | flex,
            separator(),
            hbox({text(" > ") | bold, message_input->Render() | flex}),
            hbox({text(" 📎 ") | dim, image_input->Render() | flex}),
            controls->Render(),
            st->runner.render(),
        });
    });
}

// ── Voice ─────────────────────────────────────────────────────────────────────

Component voice_tab(UserDashboard& dashboard, AudioPlayer& player) {
    struct State {
        std::string description;
        std::string sample_text;
        std::string speech_text;
        int proposal_sel = 0;
        std::vector<std::string> proposal_entries;
        Runner runner;
    };
    auto st = std::make_shared<State>();

    const auto selected_proposal = [&dashboard, st]() -> nlohmann::json {
        const auto snap = dashboard.snapshot();
        const auto proposals = snap.voice.value("proposals", nlohmann::json::array());
        if (proposals.empty()) return nlohmann::json::object();
        return proposals[static_cast<std::size_t>(clamp_index(st->proposal_sel, proposals.size()))];
    };

    InputOption multi;
    multi.multiline = true;
    multi.placeholder = "Describe an original synthetic voice (blank: the model proposes one)";
    auto description_input = Input(&st->description, multi);
    InputOption sample_opt;
    sample_opt.multiline = true;
    sample_opt.placeholder = "Text for the voice preview";
    auto sample_input = Input(&st->sample_text, sample_opt);
    InputOption speech_opt;
    speech_opt.multiline = true;
    speech_opt.placeholder = "Text to speak with the approved voice";
    auto speech_input = Input(&st->speech_text, speech_opt);
    auto proposal_menu = Menu(&st->proposal_entries, &st->proposal_sel, MenuOption::Vertical());

    auto propose = Button("Propose", [&dashboard, st] {
        const std::string user = dashboard.focused_user();
        if (user.empty()) return;
        nlohmann::json body = nlohmann::json::object();
        if (!util::trim(st->description).empty()) body["voice_description"] = st->description;
        if (!util::trim(st->sample_text).empty()) body["sample_text"] = st->sample_text;
        st->runner.run([st, &dashboard, user, body] {
            std::string err;
            if (dashboard.voice_propose(user, body, err)) st->runner.set("proposal created", true);
            else st->runner.set(err, false);
        });
    }, ButtonOption::Simple());
    const auto decide = [&dashboard, st, selected_proposal, &player](const std::string& verb) {
        const std::string user = dashboard.focused_user();
        const std::string id = selected_proposal().value("id", std::string{});
        if (user.empty() || id.empty()) return;
        st->runner.run([st, &dashboard, &player, user, id, verb] {
            nlohmann::json response;
            std::string err;
            if (!dashboard.voice_proposal_action(user, id, verb, response, err)) {
                st->runner.set(err, false);
                return;
            }
            if (verb == "sample") {
                const std::string path =
                    response.value("proposal", nlohmann::json::object()).value("preview_audio_path", std::string{});
                if (!path.empty() && player.load(path)) player.play();
            }
            st->runner.set(verb + ": done", true);
        });
    };
    auto sample = Button("Sample", [decide] { decide("sample"); }, ButtonOption::Simple());
    auto approve = Button("Approve", [decide] { decide("approve"); }, ButtonOption::Simple());
    auto reject = Button("Reject", [decide] { decide("reject"); }, ButtonOption::Simple());
    auto speak = Button("Speak", [&dashboard, st, &player] {
        const std::string user = dashboard.focused_user();
        const std::string text_value = st->speech_text;
        if (user.empty() || util::trim(text_value).empty()) return;
        st->runner.run([st, &dashboard, &player, user, text_value] {
            nlohmann::json response;
            std::string err;
            if (!dashboard.speak(user, text_value, response, err)) {
                st->runner.set(err, false);
                return;
            }
            const std::string path =
                response.value("result", nlohmann::json::object()).value("audio_path", std::string{});
            if (!path.empty() && player.load(path)) player.play();
            st->runner.set("spoken", true);
        });
    }, ButtonOption::Simple());
    auto play = Button("Play", [&player] { player.play(); }, ButtonOption::Simple());
    auto pause = Button("Pause", [&player] { player.pause(); }, ButtonOption::Simple());
    auto stop = Button("Stop", [&player] { player.stop(); }, ButtonOption::Simple());
    auto buttons = Container::Horizontal({propose, sample, approve, reject, speak, play, pause, stop});
    auto container = Container::Vertical({proposal_menu, description_input, sample_input, speech_input, buttons});

    return Renderer(container, [&dashboard, st, &player, proposal_menu, description_input, sample_input,
                                speech_input, buttons]() -> Element {
        const auto snap = dashboard.snapshot();
        const auto proposals = snap.voice.value("proposals", nlohmann::json::array());
        st->proposal_entries.clear();
        for (const auto& p : proposals)
            st->proposal_entries.push_back(p.value("display_name", std::string{"(unnamed)"}) + "  [" +
                                           p.value("status", std::string{}) + "]");
        st->proposal_sel = clamp_index(st->proposal_sel, st->proposal_entries.size());
        const auto active = snap.voice.value("active_profile", nlohmann::json());
        const auto audio = player.status();
        return vbox({
            hbox({text(" user ") | dim, text(snap.user_id.empty() ? "(none)" : snap.user_id) | bold,
                  text("   active voice ") | dim,
                  text(active.is_object() ? active.value("display_name", std::string{}) : std::string("(none)")),
                  filler(),
                  text(snap.voice.value("tts_enabled", false) ? "TTS enabled " : "TTS disabled ") |
                      color(snap.voice.value("tts_enabled", false) ? Color::Green : Color::Red)}),
            separator(),
            window(text(" PROPOSALS "), st->proposal_entries.empty()
                                            ? text("  (none — Propose creates one)") | dim
                                            : proposal_menu->Render() | yframe) |
                size(HEIGHT, LESS_THAN, 12),
            window(text(" DESIGN "), vbox({description_input->Render(), separator(), sample_input->Render()})),
            window(text(" SPEAK "), speech_input->Render()),
            buttons->Render(),
            text(audio.loaded ? " ♪ " + std::filesystem::path(audio.path).filename().string() +
                                    (audio.playing ? " (playing)" : "")
                              : " ♪ (nothing loaded)") | dim,
            st->runner.render(),
        });
    });
}

} // namespace mm
