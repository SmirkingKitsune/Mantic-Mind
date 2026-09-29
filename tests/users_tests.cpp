// mm_users_tests — the user layer, without a server.
//
// Its own executable for the reason mm_models_tests is: mm_reliability_tests
// stops at its first failure, and under WSL that is always admission_fetch_stage,
// so anything registered after it never runs on GCC. Everything here is pure or
// SQLite-only and must be checked on both toolchains.

#include "common/conversation_context.hpp"
#include "common/history_sanitizer.hpp"
#include "common/user.hpp"
#include "common/user_db.hpp"
#include "common/util.hpp"
#include "control/conversation_queue.hpp"
#include "control/served_model_catalog.hpp"
#include "control/token_binding.hpp"
#include "control/user_manager.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

bool check(bool ok, const char* expr, int line) {
    if (!ok) std::cerr << "CHECK failed at line " << line << ": " << expr << "\n";
    return ok;
}

#define CHECK(expr) do { if (!check((expr), #expr, __LINE__)) return false; } while (0)

std::filesystem::path temp_test_dir(const std::string& name) {
    return std::filesystem::temp_directory_path() /
           ("mantic-mind-" + name + "-" + mm::util::generate_uuid());
}

bool remove_tree(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return !ec;
}

mm::Message text_message(mm::MessageRole role, const std::string& text, int tokens = 0) {
    mm::Message m;
    m.role = role;
    m.content = text;
    m.token_count = tokens;
    return m;
}

mm::MessageContentPart image_part(const std::string& attachment_id) {
    mm::MessageContentPart p;
    p.type = "image_attachment";
    p.attachment_id = attachment_id;
    p.mime_type = "image/png";
    return p;
}

mm::MessageContentPart text_part(const std::string& text) {
    mm::MessageContentPart p;
    p.type = "text";
    p.text = text;
    return p;
}

mm::ToolCall tool_call(const std::string& id, const std::string& name, const std::string& args) {
    mm::ToolCall c;
    c.id = id;
    c.function_name = name;
    c.arguments_json = args;
    return c;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// ── ids and profiles ──────────────────────────────────────────────────────────

bool test_user_ids() {
    CHECK(mm::util::is_valid_user_id("alice"));
    CHECK(mm::util::is_valid_user_id("a"));
    CHECK(mm::util::is_valid_user_id("bob-2"));
    CHECK(mm::util::is_valid_user_id("j.smith_1"));
    CHECK(mm::util::is_valid_user_id("console")); // only the exact device stem is reserved
    CHECK(mm::util::is_valid_user_id("com10"));
    // Lowercase: Windows paths ignore case, so these would share a directory.
    CHECK(!mm::util::is_valid_user_id("Alice"));
    CHECK(!mm::util::is_valid_user_id(""));
    CHECK(!mm::util::is_valid_user_id(std::string(65, 'a')));
    CHECK(mm::util::is_valid_user_id(std::string(64, 'a')));
    // Windows drops a trailing dot: "alice." would be the directory "alice".
    CHECK(!mm::util::is_valid_user_id("alice."));
    CHECK(!mm::util::is_valid_user_id("-alice"));
    CHECK(!mm::util::is_valid_user_id("alice_"));
    CHECK(!mm::util::is_valid_user_id("../x"));
    CHECK(!mm::util::is_valid_user_id("a/b"));
    CHECK(!mm::util::is_valid_user_id("has space"));
    // Device names, with or without an extension.
    for (const char* device : {"con", "prn", "aux", "nul", "com1", "lpt9", "com0", "con.txt",
                               "nul.x.y", "lpt1.log"}) {
        if (!check(!mm::util::is_valid_user_id(device), device, __LINE__)) return false;
    }
    return true;
}

bool test_profile_patch() {
    mm::UserProfile p;
    p.id = "alice";
    p.display_name = "Alice";
    p.created_at_ms = 5;
    std::string err;

    CHECK(mm::apply_profile_patch(nlohmann::json{{"display_name", "Al"},
                                                 {"custom_instructions", "Be brief."},
                                                 {"default_model", "qwen3"},
                                                 {"memories_enabled", false},
                                                 {"tools_enabled", true}},
                                  p, err));
    CHECK(p.display_name == "Al");
    CHECK(p.custom_instructions == "Be brief.");
    CHECK(p.default_model == "qwen3");
    CHECK(!p.memories_enabled);
    CHECK(p.tools_enabled);

    // A GET body sent back unchanged is accepted: same id, server timestamps ignored.
    nlohmann::json round_trip = p;
    round_trip["created_at_ms"] = 999;
    CHECK(mm::apply_profile_patch(round_trip, p, err));
    CHECK(p.created_at_ms == 5);

    // Refused, and the profile is left exactly as it was.
    const mm::UserProfile before = p;
    CHECK(!mm::apply_profile_patch(nlohmann::json{{"display_name", "X"}, {"colour", "red"}}, p, err));
    CHECK(contains(err, "colour"));
    CHECK(p.display_name == before.display_name);
    CHECK(!mm::apply_profile_patch(nlohmann::json{{"tools_enabled", "yes"}}, p, err));
    CHECK(contains(err, "tools_enabled"));
    CHECK(!mm::apply_profile_patch(nlohmann::json{{"id", "bob"}}, p, err));
    CHECK(contains(err, "cannot be changed"));
    CHECK(!mm::apply_profile_patch(nlohmann::json::array(), p, err));
    CHECK(p.id == "alice" && p.display_name == before.display_name);

    // Lenient reading, for clients of the API.
    mm::UserProfile read;
    mm::from_json(nlohmann::json{{"id", "bob"}, {"tools_enabled", "not a bool"}}, read);
    CHECK(read.id == "bob");
    CHECK(!read.tools_enabled);
    return true;
}

// ── the store ─────────────────────────────────────────────────────────────────

bool test_user_db_round_trip() {
    const auto dir = temp_test_dir("user-db");
    {
        bool threw = false;
        try { mm::UserDB bad("../escape", dir.string()); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);

        mm::UserDB db("alice", dir.string());
        CHECK(std::filesystem::exists(dir / "users" / "alice" / "user.db"));
        CHECK(!db.load_profile());

        mm::UserProfile p;
        p.id = "alice";
        p.display_name = "Alice";
        p.default_model = "qwen3";
        db.save_profile(p);
        auto loaded = db.load_profile();
        CHECK(loaded && loaded->display_name == "Alice" && loaded->default_model == "qwen3");
        CHECK(loaded->created_at_ms > 0);
        const auto created = loaded->created_at_ms;
        p.display_name = "Alice B";
        p.created_at_ms = 1; // the stored one is kept
        db.save_profile(p);
        loaded = db.load_profile();
        CHECK(loaded && loaded->display_name == "Alice B" && loaded->created_at_ms == created);
        mm::UserProfile other;
        other.id = "bob";
        bool refused = false;
        try { db.save_profile(other); } catch (const std::invalid_argument&) { refused = true; }
        CHECK(refused);

        // A conversation carries its model, prompt and sampling.
        mm::NewConversation spec;
        spec.title = "Trip";
        spec.model_id = "qwen3";
        spec.system_prompt = "Plan trips.";
        spec.settings.temperature = 0.3f;
        spec.settings.max_tokens = 256;
        const auto conv = db.create_conversation(spec);
        CHECK(!conv.empty());
        auto meta = db.get_conversation(conv);
        CHECK(meta && meta->title == "Trip" && meta->model_id == "qwen3");
        CHECK(meta->system_prompt == "Plan trips.");
        CHECK(meta->settings.temperature && *meta->settings.temperature == 0.3f);
        CHECK(meta->settings.max_tokens && *meta->settings.max_tokens == 256);
        CHECK(!meta->settings.top_p);
        CHECK(meta->summary_through_seq == -1);
        CHECK(!meta->is_active);

        mm::NewConversation orphan;
        orphan.parent_conv_id = "no-such-conversation";
        CHECK(db.create_conversation(orphan).empty());

        // The store assigns sequence numbers; messages keep their model.
        auto user = text_message(mm::MessageRole::User, "Where to?", 3);
        CHECK(db.append_message(conv, user) == 0);
        auto reply = text_message(mm::MessageRole::Assistant, "Lisbon.", 2);
        reply.model_id = "qwen3";
        CHECK(db.append_message(conv, reply) == 1);
        // An image must be uploaded before a message can reference it.
        mm::ImageAttachment att;
        att.id = "att-1";
        att.mime_type = "image/png";
        att.relative_path = "attachments/att-1.png";
        att.size_bytes = 4;
        att.created_at_ms = mm::util::now_ms();
        att.expires_at_ms = att.created_at_ms + 60000; // pending until referenced
        db.save_attachment(att);
        auto with_image = text_message(mm::MessageRole::User, "And this?", 3);
        with_image.content_parts = {text_part("And this?"), image_part("att-1")};
        CHECK(db.append_message(conv, with_image) == 2);
        CHECK(db.get_attachment("att-1")->expires_at_ms == 0); // referenced: no longer expires
        CHECK(db.attachment_is_referenced("att-1"));
        CHECK(db.append_message("no-such-conversation", user) == -1);

        const auto all = db.load_messages(conv);
        CHECK(all.size() == 3);
        CHECK(all[0].sequence_num == 0 && all[1].sequence_num == 1 && all[2].sequence_num == 2);
        CHECK(all[1].model_id == "qwen3");
        CHECK(all[0].model_id.empty());
        CHECK(all[2].content_parts.size() == 2 && all[2].content_parts[1].attachment_id == "att-1");
        const auto after = db.load_messages(conv, 0);
        CHECK(after.size() == 2 && after.front().sequence_num == 1);
        CHECK(db.get_total_tokens(conv) == 8);
        const auto full = db.load_conversation(conv);
        CHECK(full && full->messages.size() == 3);

        // JSON carries the user-era fields.
        const nlohmann::json j = *full;
        CHECK(j["model_id"] == "qwen3");
        CHECK(j["settings"]["temperature"].get<float>() == 0.3f);
        CHECK(!j["settings"].contains("top_p"));
        CHECK(j["messages"][1]["model_id"] == "qwen3");
        CHECK(j["messages"][2]["sequence_num"] == 2);
        const auto back = j.get<mm::Conversation>();
        CHECK(back.settings.max_tokens && *back.settings.max_tokens == 256);
        CHECK(back.messages[1].model_id == "qwen3");

        // A patch changes only what it names.
        mm::ConversationPatch patch;
        patch.model_id = "llama";
        CHECK(db.update_conversation(conv, patch));
        meta = db.get_conversation(conv);
        CHECK(meta && meta->model_id == "llama" && meta->title == "Trip");
        CHECK(meta->settings.temperature.has_value());
        CHECK(!db.update_conversation("no-such-conversation", patch));

        // The watermark only moves forward.
        CHECK(db.save_summary(conv, "They chose Lisbon.", 1, "qwen3"));
        CHECK(db.save_summary(conv, "They chose Lisbon (shorter).", 1, "llama"));
        CHECK(!db.save_summary(conv, "stale", 0, "qwen3"));
        meta = db.get_conversation(conv);
        CHECK(meta && meta->summary == "They chose Lisbon (shorter)." && meta->summary_through_seq == 1);
        CHECK(meta->summary_model_id == "llama");

        // Memories, and what deleting a conversation takes with it.
        mm::LocalMemory note;
        note.conversation_id = conv;
        note.content = "prefers trains";
        db.add_local_memory(note);
        CHECK(db.list_local_memories(conv).size() == 1);
        mm::Memory memory;
        memory.id = "m1";
        memory.content = "lives in Porto";
        memory.source_conv_id = conv;
        memory.importance = 0.9f;
        db.add_memory(memory);
        CHECK(db.search_memories("Porto").size() == 1);
        CHECK(db.delete_conversation(conv));
        CHECK(!db.delete_conversation(conv));
        CHECK(!db.conversation_exists(conv));
        CHECK(db.list_local_memories(conv).empty());
        CHECK(!db.get_attachment("att-1")); // referenced only by that conversation
        const auto kept = db.get_memory("m1");
        CHECK(kept && kept->source_conv_id.empty()); // global memories outlive their source
    }
    CHECK(remove_tree(dir));
    return true;
}

bool test_one_active_conversation() {
    const auto dir = temp_test_dir("user-active");
    {
        mm::UserDB db("alice", dir.string());
        CHECK(!db.get_active_conversation_id());
        mm::NewConversation active;
        active.set_active = true;
        const auto a = db.create_conversation(active);
        const auto b = db.create_conversation(active);
        CHECK(db.get_active_conversation_id() == b);
        CHECK(!db.is_conversation_active(a));
        CHECK(db.set_active_conversation(a));
        CHECK(db.is_conversation_active(a) && !db.is_conversation_active(b));
        // An unknown id changes nothing.
        CHECK(!db.set_active_conversation("no-such-conversation"));
        CHECK(db.get_active_conversation_id() == a);
        CHECK(db.get_or_create_active_conversation() == a);
        int active_count = 0;
        for (const auto& c : db.list_conversations()) active_count += c.is_active ? 1 : 0;
        CHECK(active_count == 1);
    }
    CHECK(remove_tree(dir));
    return true;
}

bool test_active_conversation_under_concurrency() {
    const auto dir = temp_test_dir("user-race");
    {
        mm::UserDB db("alice", dir.string());
        std::vector<std::thread> threads;
        std::vector<mm::ConvId> seen(16);
        for (int i = 0; i < 16; ++i)
            threads.emplace_back([&, i] { seen[i] = db.get_or_create_active_conversation(); });
        for (auto& t : threads) t.join();
        CHECK(!seen[0].empty());
        CHECK(std::all_of(seen.begin(), seen.end(), [&](const mm::ConvId& id) { return id == seen[0]; }));
        CHECK(db.list_conversations().size() == 1);

        // The store's sequence numbers stay dense and unique under contention.
        threads.clear();
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&db, &seen] {
                for (int i = 0; i < 25; ++i)
                    db.append_message(seen[0], text_message(mm::MessageRole::User, "x"));
            });
        }
        for (auto& t : threads) t.join();
        const auto messages = db.load_messages(seen[0]);
        CHECK(messages.size() == 200);
        for (int i = 0; i < 200; ++i) CHECK(messages[static_cast<std::size_t>(i)].sequence_num == i);
    }
    CHECK(remove_tree(dir));
    return true;
}

// ── the manager ───────────────────────────────────────────────────────────────

bool test_user_manager() {
    const auto dir = temp_test_dir("user-manager");
    {
        mm::UserManager users(dir.string());
        users.load_all();
        CHECK(!users.legacy_agent_data_present());
        std::string err;

        mm::UserProfile alice;
        alice.id = "alice";
        CHECK(users.create_user(alice, err) == mm::UserManager::CreateResult::Created);
        CHECK(users.create_user(alice, err) == mm::UserManager::CreateResult::Exists);
        mm::UserProfile shouting = alice;
        shouting.id = "ALICE";
        CHECK(users.create_user(shouting, err) == mm::UserManager::CreateResult::Invalid);
        mm::UserProfile device;
        device.id = "con";
        CHECK(users.create_user(device, err) == mm::UserManager::CreateResult::Invalid);
        CHECK(contains(err, "device"));
        mm::UserProfile bob;
        bob.id = "bob";
        bob.display_name = "Bob";
        CHECK(users.create_user(bob, err) == mm::UserManager::CreateResult::Created);

        const auto listed = users.list_users();
        CHECK(listed.size() == 2 && listed[0].id == "alice" && listed[1].id == "bob");
        CHECK(listed[0].display_name == "alice"); // defaulted to the id
        CHECK(listed[0].created_at_ms > 0);

        auto handle = users.get_user("alice");
        CHECK(handle && handle->id() == "alice");
        auto updated = handle->profile();
        updated.default_model = "qwen3";
        CHECK(users.update_profile("alice", updated));
        CHECK(users.get_user("alice")->profile().default_model == "qwen3");
        CHECK(handle->db().load_profile()->default_model == "qwen3");
        updated.id = "mallory";
        CHECK(!users.update_profile("alice", updated));
        CHECK(!users.update_profile("nobody", updated));

        // Deferred delete: the store stays usable to a request that holds it,
        // and the id cannot be reused until that request lets go.
        const auto conv = handle->db().get_or_create_active_conversation();
        CHECK(users.delete_user("alice"));
        CHECK(!users.delete_user("alice"));
        CHECK(!users.get_user("alice"));
        CHECK(std::filesystem::exists(dir / "users" / "alice"));
        CHECK(handle->db().append_message(conv, text_message(mm::MessageRole::User, "still here")) == 0);
        CHECK(users.create_user(alice, err) == mm::UserManager::CreateResult::Exists);
        CHECK(contains(err, "still being removed"));
        handle.reset();
        CHECK(!std::filesystem::exists(dir / "users" / "alice"));
        CHECK(users.create_user(alice, err) == mm::UserManager::CreateResult::Created);
        CHECK(users.get_user("alice")->db().list_conversations().empty()); // a new user, not the old one
    }
    CHECK(remove_tree(dir));
    return true;
}

bool test_user_manager_load_all() {
    const auto dir = temp_test_dir("user-load");
    {
        {
            mm::UserManager users(dir.string());
            std::string err;
            mm::UserProfile carol;
            carol.id = "carol";
            carol.custom_instructions = "Answer in French.";
            CHECK(users.create_user(carol, err) == mm::UserManager::CreateResult::Created);
        }
        // A store whose profile was never written (interrupted creation) ...
        { mm::UserDB half("dave", dir.string()); }
        // ... a directory that is not a user id, and agent data from before.
        std::filesystem::create_directories(dir / "users" / "Not A User");
        std::filesystem::create_directories(dir / "agents" / "old-agent");

        mm::UserManager users(dir.string());
        users.load_all();
        CHECK(users.legacy_agent_data_present());
        const auto listed = users.list_users();
        CHECK(listed.size() == 2);
        CHECK(listed[0].id == "carol" && listed[0].custom_instructions == "Answer in French.");
        CHECK(listed[1].id == "dave" && listed[1].display_name == "dave");
        // Agent data is left exactly where it was.
        CHECK(std::filesystem::exists(dir / "agents" / "old-agent"));
    }
    CHECK(remove_tree(dir));
    return true;
}

// ── the queue ─────────────────────────────────────────────────────────────────

bool test_queue_order_and_parallelism() {
    // Order within a key.
    {
        mm::ConversationQueue queue(4);
        std::mutex mu;
        std::vector<int> order;
        for (int i = 0; i < 50; ++i) {
            queue.enqueue("alice\x1f" "c1", [&, i] {
                std::lock_guard<std::mutex> lk(mu);
                order.push_back(i);
            });
        }
        queue.shutdown();
        CHECK(order.size() == 50);
        for (int i = 0; i < 50; ++i) CHECK(order[static_cast<std::size_t>(i)] == i);
        CHECK(queue.active_keys() == 0); // strands are erased once empty
    }

    // Different keys run at once, up to the pool size and no further.
    const auto peak_with = [](std::size_t workers, int keys) {
        mm::ConversationQueue queue(workers);
        std::mutex mu;
        std::condition_variable cv;
        int running = 0;
        int peak = 0;
        bool release = false;
        for (int k = 0; k < keys; ++k) {
            queue.enqueue("key-" + std::to_string(k), [&] {
                std::unique_lock<std::mutex> lk(mu);
                ++running;
                peak = std::max(peak, running);
                cv.notify_all();
                cv.wait_for(lk, std::chrono::seconds(5), [&] { return release; });
                --running;
            });
        }
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait_for(lk, std::chrono::seconds(5),
                        [&] { return running == std::min<int>(static_cast<int>(workers), keys); });
            // Give an over-eager pool the chance to exceed its size.
            cv.wait_for(lk, std::chrono::milliseconds(100), [] { return false; });
            release = true;
        }
        cv.notify_all();
        queue.shutdown();
        return peak;
    };
    CHECK(peak_with(4, 4) == 4);
    CHECK(peak_with(2, 4) == 2);

    // A job that throws does not take its strand down; shutdown drains, then refuses.
    {
        mm::ConversationQueue queue(1);
        std::atomic<int> ran{0};
        queue.enqueue("k", [] { throw std::runtime_error("boom"); });
        queue.enqueue("k", [&] { ++ran; });
        queue.enqueue("other", [&] { ++ran; });
        queue.shutdown();
        CHECK(ran == 2);
        bool rejected = false;
        CHECK(!queue.enqueue("k", [&] { ++ran; }, [&] { rejected = true; }));
        CHECK(rejected);
        CHECK(ran == 2);
    }
    return true;
}

// ── history for the model that answers ────────────────────────────────────────

bool test_sanitizer_matrix() {
    std::vector<mm::Message> history;
    auto question = text_message(mm::MessageRole::User, "What is in this photo?");
    question.content_parts = {text_part("What is in this photo?"), image_part("att-cat")};
    history.push_back(question);
    auto call = text_message(mm::MessageRole::Assistant, "Let me note that.");
    call.tool_calls = {tool_call("c1", "save_local_memory", R"({"content":"cat"})")};
    call.model_id = "vision-model";
    history.push_back(call);
    auto result = text_message(mm::MessageRole::Tool, R"({"ok":true})");
    result.tool_call_id = "c1";
    history.push_back(result);
    auto answer = text_message(mm::MessageRole::Assistant, "A cat.");
    answer.model_id = "vision-model";
    history.push_back(answer);
    history.push_back(text_message(mm::MessageRole::User, "Thanks"));

    const mm::AttachmentNamer name_of = [](const std::string& id) {
        return id == "att-cat" ? std::string("cat.png") : std::string{};
    };

    // Both capabilities: unchanged.
    auto same = mm::sanitize_history(history, {true, true}, name_of);
    CHECK(same.size() == 5);
    CHECK(same[0].content_parts.size() == 2);
    CHECK(same[1].tool_calls.size() == 1);

    // No vision: the image becomes a placeholder naming its file.
    auto no_vision = mm::sanitize_history(history, {false, true}, name_of);
    CHECK(no_vision.size() == 5);
    CHECK(no_vision[0].content_parts.empty());
    CHECK(no_vision[0].content == "What is in this photo?\n[image omitted: cat.png]");
    CHECK(no_vision[1].tool_calls.size() == 1);

    // No tools: the round folds into ONE assistant message, calls and results written out.
    auto no_tools = mm::sanitize_history(history, {true, false}, name_of);
    CHECK(no_tools.size() == 3);
    CHECK(no_tools[0].role == mm::MessageRole::User && no_tools[0].content_parts.size() == 2);
    CHECK(no_tools[1].role == mm::MessageRole::Assistant);
    CHECK(no_tools[1].tool_calls.empty());
    CHECK(no_tools[1].model_id == "vision-model");
    CHECK(no_tools[1].content ==
          "Let me note that.\n\n[called save_local_memory({\"content\":\"cat\"}) -> {\"ok\":true}]"
          "\n\nA cat.");
    CHECK(no_tools[2].content == "Thanks");

    // Neither: both rewrites, no tool role left anywhere.
    auto neither = mm::sanitize_history(history, {false, false}, name_of);
    CHECK(neither.size() == 3);
    CHECK(neither[0].content_parts.empty() && contains(neither[0].content, "[image omitted: cat.png]"));
    CHECK(std::none_of(neither.begin(), neither.end(), [](const mm::Message& m) {
        return m.role == mm::MessageRole::Tool || !m.tool_calls.empty();
    }));

    // A result whose call came before the history starts is kept as text.
    std::vector<mm::Message> mid_round = {result, text_message(mm::MessageRole::User, "next")};
    auto kept = mm::sanitize_history(mid_round, {true, false});
    CHECK(kept.size() == 2 && kept[0].role == mm::MessageRole::Assistant);
    CHECK(contains(kept[0].content, "[tool result]"));

    // An unnamed image, and words stored only in content.
    auto bare = text_message(mm::MessageRole::User, "look");
    bare.content_parts = {image_part("att-unknown")};
    auto stripped = mm::sanitize_history({bare}, {false, true}, name_of);
    CHECK(stripped[0].content == "look\n[image omitted]");
    return true;
}

// ── which model answers ───────────────────────────────────────────────────────

bool test_turn_model_resolution() {
    std::map<std::string, mm::CatalogEntry> catalog;
    for (const char* id : {"qwen3", "llama", "off"}) {
        mm::CatalogEntry e;
        e.model.id = id;
        e.available = std::string(id) != "off";
        if (!e.available) e.unavailable_reason = "disabled by the operator";
        catalog[id] = e;
    }
    const mm::CatalogLookup find = [&](const std::string& id) -> std::optional<mm::CatalogEntry> {
        const auto it = catalog.find(id);
        if (it == catalog.end()) return std::nullopt;
        return it->second;
    };
    using P = mm::TurnModel::Problem;

    // Request over conversation over the user's default.
    auto t = mm::resolve_turn_model("llama", "qwen3", "off", find);
    CHECK(t.problem == P::None && t.model_id == "llama" && t.sticky);
    t = mm::resolve_turn_model("", "qwen3", "llama", find);
    CHECK(t.problem == P::None && t.model_id == "qwen3" && !t.sticky);
    t = mm::resolve_turn_model("", "", "llama", find);
    CHECK(t.problem == P::None && t.model_id == "llama" && !t.sticky);
    // Naming the model the conversation already has changes nothing.
    t = mm::resolve_turn_model("qwen3", "qwen3", "", find);
    CHECK(t.problem == P::None && !t.sticky);
    // Trimmed and lowercased.
    t = mm::resolve_turn_model("  LLAMA ", "qwen3", "", find);
    CHECK(t.problem == P::None && t.model_id == "llama" && t.sticky);
    CHECK(t.entry && t.entry->model.id == "llama");

    t = mm::resolve_turn_model("", "", "", find);
    CHECK(t.problem == P::Required && std::string(mm::to_string(t.problem)) == "model_required");
    t = mm::resolve_turn_model("gone", "qwen3", "", find);
    CHECK(t.problem == P::NotFound && t.model_id == "gone" && !t.sticky);
    CHECK(std::string(mm::to_string(t.problem)) == "model_not_found");
    t = mm::resolve_turn_model("off", "qwen3", "", find);
    CHECK(t.problem == P::Unavailable && !t.sticky && contains(t.error, "disabled"));
    // A conversation whose model has gone is told so, not silently moved.
    t = mm::resolve_turn_model("", "gone", "llama", find);
    CHECK(t.problem == P::NotFound && t.model_id == "gone");
    return true;
}

// ── token binding ─────────────────────────────────────────────────────────────

bool test_binding_matrix() {
    // Unbound: binding never narrows anything.
    CHECK(mm::binding_permits("", "DELETE", "/v1/users/bob").allowed);
    CHECK(mm::binding_permits("", "POST", "/v1/tokens").allowed);

    const auto allowed = [](const char* method, const char* path) {
        return mm::binding_permits("alice", method, path).allowed;
    };
    // Its own user: scopes apply unchanged.
    CHECK(allowed("GET", "/v1/users/alice"));
    CHECK(allowed("PUT", "/v1/users/alice"));
    CHECK(allowed("POST", "/v1/users/alice/chat"));
    CHECK(allowed("DELETE", "/v1/users/alice/conversations/c1"));
    // Anyone else's: refused, prefix tricks included.
    CHECK(!allowed("GET", "/v1/users/bob"));
    CHECK(!allowed("GET", "/v1/users/bob/conversations"));
    CHECK(!allowed("POST", "/v1/users/alice2/chat"));
    CHECK(!allowed("GET", "/v1/users/alic"));
    CHECK(!allowed("GET", "/v1/users/ALICE"));
    // The user list is filtered by its handler; creating users is refused.
    CHECK(allowed("GET", "/v1/users"));
    CHECK(!allowed("POST", "/v1/users"));
    // What records or reaches every user.
    CHECK(!allowed("GET", "/v1/tokens"));
    CHECK(!allowed("POST", "/v1/tokens"));
    CHECK(!allowed("DELETE", "/v1/tokens/7"));
    CHECK(!allowed("GET", "/v1/activity"));
    CHECK(!allowed("GET", "/v1/performance"));
    CHECK(!allowed("DELETE", "/v1/performance"));
    // Shared state: readable, not changeable.
    CHECK(allowed("GET", "/v1/served-models"));
    CHECK(allowed("GET", "/v1/nodes"));
    CHECK(!allowed("POST", "/v1/served-models"));
    CHECK(!allowed("PUT", "/v1/served-models/qwen3"));
    CHECK(!allowed("POST", "/v1/served-models/qwen3/suspend"));
    // Stateless completions and speech (the handler checks the voice).
    CHECK(allowed("POST", "/v1/chat/completions"));
    CHECK(allowed("POST", "/v1/audio/speech"));
    const auto refused = mm::binding_permits("alice", "GET", "/v1/users/bob");
    CHECK(contains(refused.error, "alice") && contains(refused.error, "bob"));
    return true;
}

// ── the context ───────────────────────────────────────────────────────────────

bool test_context_order() {
    mm::UserProfile user;
    user.id = "alice";
    user.custom_instructions = "Be kind.";
    mm::Conversation conv;
    conv.system_prompt = "Plan trips.";
    conv.summary = "They chose Lisbon.";
    mm::Memory global;
    global.id = "g1";
    global.content = "lives in Porto";
    mm::LocalMemory local;
    local.id = "l1";
    local.content = "prefers trains";

    const auto text = mm::build_system_text(user, conv, {global}, {local}, false);
    const auto at = [&](const std::string& s) { return text.find(s); };
    CHECK(at("Be kind.") != std::string::npos);
    CHECK(at("Be kind.") < at("Plan trips."));
    CHECK(at("Plan trips.") < at("lives in Porto"));
    CHECK(at("lives in Porto") < at("prefers trains"));
    CHECK(at("prefers trains") < at("They chose Lisbon."));
    CHECK(at("[g1]") == std::string::npos); // ids only when there are tools to use them
    CHECK(contains(mm::build_system_text(user, conv, {global}, {local}, true), "[g1]"));

    user.memories_enabled = false;
    const auto quiet = mm::build_system_text(user, conv, {global}, {local}, true);
    CHECK(!contains(quiet, "Porto") && !contains(quiet, "trains"));
    CHECK(contains(quiet, "They chose Lisbon.")); // the summary is the conversation, not a memory

    auto context = mm::build_turn_context(text, {text_message(mm::MessageRole::User, "hi")}, {});
    CHECK(context.size() == 2 && context[0].role == mm::MessageRole::System);
    CHECK(mm::build_turn_context({}, {text_message(mm::MessageRole::User, "hi")}, {}).size() == 1);

    // The budget and the estimates the compactor runs on.
    mm::RuntimeSettings settings;
    settings.ctx_size = 4096;
    settings.max_tokens = 1024;
    CHECK(mm::prompt_budget(settings) == 3072);
    settings.max_tokens = 99999;
    CHECK(mm::prompt_budget(settings) == 1);
    auto reasoning = text_message(mm::MessageRole::Assistant, std::string(400, 'a'), 900);
    reasoning.thinking_text = std::string(3200, 't'); // 800 tokens never sent back
    CHECK(mm::estimate_tokens(reasoning, false) == 100);
    auto pictured = text_message(mm::MessageRole::User, "abcd");
    pictured.content_parts = {text_part("abcd"), image_part("a")};
    CHECK(mm::estimate_tokens(pictured, false) == 1);
    CHECK(mm::estimate_tokens(pictured, true) == 1 + mm::kImageTokenEstimate);
    return true;
}

// ── rolling compaction ────────────────────────────────────────────────────────

/// A summarizer that checks every call fits the model, and records what it saw.
struct FakeSummarizer {
    int budget = 0;
    int calls = 0;
    int fail_on_call = 0; ///< 1-based; 0 never
    bool over_budget = false;
    std::vector<std::string> prompts;

    mm::Summarize fn() {
        return [this](const std::vector<mm::Message>& prompt, int max_tokens)
                   -> std::optional<std::string> {
            ++calls;
            int tokens = 0;
            for (const auto& m : prompt) tokens += mm::estimate_tokens(m.content);
            if (tokens + max_tokens > budget) over_budget = true;
            prompts.push_back(prompt.size() > 1 ? prompt[1].content : std::string{});
            if (calls == fail_on_call) return std::nullopt;
            return "summary after call " + std::to_string(calls);
        };
    }
};

/// Stored state a SaveSummary writes to, as the real store would.
struct SavedSummary {
    std::string summary;
    int through = -1;
    std::vector<int> watermarks;

    mm::SaveSummary fn() {
        return [this](const std::string& s, int seq) {
            if (seq < through) return false; // the store refuses a regression
            summary = s;
            through = seq;
            watermarks.push_back(seq);
            return true;
        };
    }
};

std::vector<mm::Message> long_conversation(int turns, std::size_t chars_per_message) {
    std::vector<mm::Message> out;
    int seq = 0;
    for (int t = 0; t < turns; ++t) {
        auto q = text_message(mm::MessageRole::User, "Q" + std::to_string(t) + std::string(chars_per_message, 'q'));
        q.sequence_num = seq++;
        out.push_back(q);
        auto a = text_message(mm::MessageRole::Assistant, "A" + std::to_string(t) + std::string(chars_per_message, 'a'));
        a.sequence_num = seq++;
        out.push_back(a);
    }
    auto latest = text_message(mm::MessageRole::User, "latest question");
    latest.sequence_num = seq;
    out.push_back(latest);
    return out;
}

bool test_compaction_not_needed_and_overflow() {
    FakeSummarizer fake;
    fake.budget = 3072;
    SavedSummary saved;
    mm::RollingCompactor compactor(fake.fn(), saved.fn());

    mm::CompactionRequest small;
    small.messages = long_conversation(2, 40);
    small.budget = 3072;
    small.base_system_tokens = 50;
    auto r = compactor.run(small);
    CHECK(r.status == mm::CompactionResult::Status::NotNeeded);
    CHECK(fake.calls == 0);
    CHECK(r.prompt_tokens > 50 && r.prompt_tokens < 3072);

    // A forced compaction of a single turn has nothing to fold.
    mm::CompactionRequest single;
    single.messages = {text_message(mm::MessageRole::User, "hi")};
    single.messages[0].sequence_num = 0;
    single.budget = 3072;
    single.force = true;
    r = compactor.run(single);
    CHECK(r.status == mm::CompactionResult::Status::NotNeeded && fake.calls == 0);

    // The latest turn alone does not fit: nothing can help.
    mm::CompactionRequest huge;
    huge.messages = {text_message(mm::MessageRole::User, std::string(20000, 'x'))};
    huge.messages[0].sequence_num = 0;
    huge.budget = 3072;
    r = compactor.run(huge);
    CHECK(r.status == mm::CompactionResult::Status::Overflow);
    CHECK(std::string(mm::to_string(r.status)) == "context_overflow");
    CHECK(fake.calls == 0);
    return true;
}

bool test_compaction_chunks_within_budget() {
    FakeSummarizer fake;
    fake.budget = 3072;
    SavedSummary saved;
    mm::RollingCompactor compactor(fake.fn(), saved.fn());

    mm::CompactionRequest req;
    req.messages = long_conversation(20, 1600); // ~40 messages of ~400 tokens
    req.budget = 3072;
    req.base_system_tokens = 100;
    const auto r = compactor.run(req);
    CHECK(r.status == mm::CompactionResult::Status::Compacted);
    CHECK(!fake.over_budget);            // every call fitted the model
    CHECK(fake.calls >= 3);              // several chunks, not one giant prompt
    CHECK(r.summarizer_calls == fake.calls);
    CHECK(saved.watermarks.size() == static_cast<std::size_t>(fake.calls));
    CHECK(std::is_sorted(saved.watermarks.begin(), saved.watermarks.end()));
    CHECK(r.summary_through_seq == saved.through);
    // The tail starts at a user turn, right after the watermark.
    const auto first_kept = std::find_if(req.messages.begin(), req.messages.end(),
        [&](const mm::Message& m) { return m.sequence_num == r.summary_through_seq + 1; });
    CHECK(first_kept != req.messages.end() && first_kept->role == mm::MessageRole::User);
    CHECK(r.prompt_tokens <= req.budget);
    // Chunks are consecutive: each call carries the summary so far.
    CHECK(contains(fake.prompts[1], "summary after call 1"));
    return true;
}

bool test_compaction_resumes_after_failure() {
    FakeSummarizer failing;
    failing.budget = 3072;
    failing.fail_on_call = 2;
    SavedSummary saved;
    const auto messages = long_conversation(20, 1600);

    mm::CompactionRequest req;
    req.messages = messages;
    req.budget = 3072;
    auto r = mm::RollingCompactor(failing.fn(), saved.fn()).run(req);
    CHECK(r.status == mm::CompactionResult::Status::Failed);
    CHECK(saved.watermarks.size() == 1); // the first chunk's progress is kept
    const int after_first = saved.through;
    CHECK(after_first >= 0);

    // Resume from what was saved: only the messages past the watermark.
    FakeSummarizer working;
    working.budget = 3072;
    mm::CompactionRequest resume;
    resume.summary = saved.summary;
    resume.summary_through_seq = saved.through;
    for (const auto& m : messages)
        if (m.sequence_num > saved.through) resume.messages.push_back(m);
    resume.budget = 3072;
    r = mm::RollingCompactor(working.fn(), saved.fn()).run(resume);
    CHECK(r.status == mm::CompactionResult::Status::Compacted);
    CHECK(!working.over_budget);
    CHECK(saved.through > after_first);
    // The first resumed call extends the summary the failed run saved.
    CHECK(contains(working.prompts[0], "summary after call 1"));
    return true;
}

bool test_compaction_truncates_one_oversized_message() {
    FakeSummarizer fake;
    fake.budget = 3072;
    SavedSummary saved;
    std::vector<mm::Message> messages;
    auto giant = text_message(mm::MessageRole::User, "START" + std::string(40000, 'g') + "END");
    giant.sequence_num = 0;
    messages.push_back(giant);
    auto reply = text_message(mm::MessageRole::Assistant, "ok");
    reply.sequence_num = 1;
    messages.push_back(reply);
    auto latest = text_message(mm::MessageRole::User, "and now?");
    latest.sequence_num = 2;
    messages.push_back(latest);

    mm::CompactionRequest req;
    req.messages = messages;
    req.budget = 3072;
    const auto r = mm::RollingCompactor(fake.fn(), saved.fn()).run(req);
    CHECK(r.status == mm::CompactionResult::Status::Compacted);
    CHECK(!fake.over_budget);
    CHECK(contains(fake.prompts[0], "bytes omitted"));
    // Its beginning and its end both survive the cut.
    CHECK(contains(fake.prompts[0], "START") && contains(fake.prompts[0], "END"));
    CHECK(r.summary_through_seq == 1);
    return true;
}

bool test_compaction_small_model_after_large() {
    // A summary a large model wrote, and a small model taking over.
    FakeSummarizer fake;
    fake.budget = 1536; // ctx 2048, max_tokens 512
    SavedSummary saved;
    saved.through = 9;
    mm::CompactionRequest req;
    req.summary = std::string(12000, 's'); // ~3000 tokens
    req.summary_through_seq = 9;
    int seq = 10;
    for (int t = 0; t < 4; ++t) {
        auto q = text_message(mm::MessageRole::User, std::string(800, 'q'));
        q.sequence_num = seq++;
        req.messages.push_back(q);
        auto a = text_message(mm::MessageRole::Assistant, std::string(800, 'a'));
        a.sequence_num = seq++;
        req.messages.push_back(a);
    }
    req.budget = 1536;
    const auto r = mm::RollingCompactor(fake.fn(), saved.fn()).run(req);
    CHECK(r.status == mm::CompactionResult::Status::Compacted);
    CHECK(!fake.over_budget);
    CHECK(fake.calls >= 2);
    // The first call condenses the old summary on its own.
    CHECK(!contains(fake.prompts[0], "New messages to fold in"));
    CHECK(contains(fake.prompts[1], "New messages to fold in"));
    CHECK(saved.watermarks.front() == 9); // condensing keeps the watermark
    CHECK(r.summary_through_seq > 9);
    CHECK(r.prompt_tokens <= req.budget);
    return true;
}

bool test_compaction_image_pressure() {
    // A vision model: images near the per-request limit force a fold even
    // though the text is short, and what stays holds no more than the limit.
    FakeSummarizer fake;
    fake.budget = 1'000'000;
    SavedSummary saved;
    mm::CompactionRequest req;
    int seq = 0;
    for (int t = 0; t < 3; ++t) {
        auto q = text_message(mm::MessageRole::User, "look");
        q.content_parts = {text_part("look"), image_part("i" + std::to_string(t) + "a"),
                           image_part("i" + std::to_string(t) + "b"),
                           image_part("i" + std::to_string(t) + "c")};
        q.sequence_num = seq++;
        req.messages.push_back(q);
        auto a = text_message(mm::MessageRole::Assistant, "seen");
        a.sequence_num = seq++;
        req.messages.push_back(a);
    }
    req.budget = 1'000'000;
    req.vision = true;
    const auto r = mm::RollingCompactor(fake.fn(), saved.fn()).run(req);
    CHECK(r.status == mm::CompactionResult::Status::Compacted);
    int kept_images = 0;
    for (const auto& m : req.messages)
        if (m.sequence_num > r.summary_through_seq) kept_images += mm::image_count(m);
    CHECK(kept_images > 0 && kept_images <= mm::kMaxContextImages);
    // The summarizer is sent no image, only its mention.
    CHECK(contains(fake.prompts[0], "[image]"));

    // Without vision the same history is text, and nothing is folded.
    FakeSummarizer idle;
    idle.budget = 1'000'000;
    req.vision = false;
    const auto text_only = mm::RollingCompactor(idle.fn(), saved.fn()).run(req);
    CHECK(text_only.status == mm::CompactionResult::Status::NotNeeded && idle.calls == 0);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    struct Test {
        const char* name;
        bool (*fn)();
    };
    const Test tests[] = {
        {"user_ids", test_user_ids},
        {"profile_patch", test_profile_patch},
        {"user_db_round_trip", test_user_db_round_trip},
        {"one_active_conversation", test_one_active_conversation},
        {"active_conversation_under_concurrency", test_active_conversation_under_concurrency},
        {"user_manager", test_user_manager},
        {"user_manager_load_all", test_user_manager_load_all},
        {"queue_order_and_parallelism", test_queue_order_and_parallelism},
        {"sanitizer_matrix", test_sanitizer_matrix},
        {"turn_model_resolution", test_turn_model_resolution},
        {"binding_matrix", test_binding_matrix},
        {"context_order", test_context_order},
        {"compaction_not_needed_and_overflow", test_compaction_not_needed_and_overflow},
        {"compaction_chunks_within_budget", test_compaction_chunks_within_budget},
        {"compaction_resumes_after_failure", test_compaction_resumes_after_failure},
        {"compaction_truncates_one_oversized_message", test_compaction_truncates_one_oversized_message},
        {"compaction_small_model_after_large", test_compaction_small_model_after_large},
        {"compaction_image_pressure", test_compaction_image_pressure},
    };
    const std::string filter = argc > 1 ? argv[1] : std::string{};
    int failed = 0;
    bool ran_any = false;
    // Every test runs, so one red test cannot hide the rest.
    for (const auto& t : tests) {
        if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos) continue;
        ran_any = true;
        bool ok = false;
        try {
            ok = t.fn();
        } catch (const std::exception& e) {
            std::cerr << "exception: " << e.what() << "\n";
        }
        if (ok) {
            std::cout << "PASSED: " << t.name << "\n";
        } else {
            std::cerr << "FAILED: " << t.name << "\n";
            ++failed;
        }
    }
    if (!filter.empty() && !ran_any) {
        std::cerr << "No tests matched filter: " << filter << "\n";
        return 1;
    }
    return failed == 0 ? 0 : 1;
}
