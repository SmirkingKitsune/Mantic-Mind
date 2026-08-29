// Soma — `soma serve`: the OpenAI-compatible HTTP surface the node supervises.
//
// This is the boundary that makes Soma a peer of the llama.cpp fallback rather
// than a library the node has to learn about. The node launches a process,
// polls GET /health until it reports ready, and then speaks the same protocol it
// already speaks to llama-server.
//
// Two things are deliberately NOT inherited from the fallback path:
//
//   * Readiness is an HTTP poll, not a stdout sentinel. RuntimeProcess already
//     works this way and it is the right shape — a sentinel is a line-buffering
//     bug waiting to happen on Windows.
//   * Capacity pressure is a STRUCTURED CODE. The existing scheduler detects it
//     by substring-matching six English phrases against the node's error body
//     (agent_scheduler.cpp:904), so a new engine would have to emit those exact
//     literals to earn an evict-and-retry. This emits
//     {"error":{"code":"capacity_pressure"}} instead.

#include "soma/serve.hpp"

#include "soma/expert_store.hpp"
#include "soma/f32_model.hpp"
#include "soma/kv_checkpoint.hpp"
#include "soma/memory_hierarchy.hpp"
#include "soma/plan.hpp"
#include "soma/prompt_codec.hpp"
#include "soma/scheduler.hpp"
#include "soma/telemetry.hpp"
#include "soma/tokenizer.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace soma {

const char* to_string(ServeError error) noexcept {
    switch (error) {
    case ServeError::None:
        return "none";
    case ServeError::BadRequest:
        return "bad_request";
    case ServeError::NotFound:
        return "not_found";
    case ServeError::UnsupportedContent:
        return "unsupported_content";
    case ServeError::CapacityPressure:
        return "capacity_pressure";
    case ServeError::ProtocolError:
        return "protocol_error";
    case ServeError::Internal:
        return "internal";
    }
    return "unknown";
}

int http_status_for(ServeError error) noexcept {
    switch (error) {
    case ServeError::None:
        return 200;
    case ServeError::BadRequest:
        return 400;
    case ServeError::NotFound:
        return 404;
    case ServeError::UnsupportedContent:
        return 422;
    case ServeError::CapacityPressure:
        return 503;
    case ServeError::ProtocolError:
        return 502;
    case ServeError::Internal:
        return 500;
    }
    return 500;
}

namespace {

std::string env_or(const char* key, const std::string& fallback) {
#if defined(_MSC_VER)
    char* buf = nullptr;
    std::size_t len = 0;
    const bool have = (_dupenv_s(&buf, &len, key) == 0 && buf != nullptr);
    std::string v = have ? std::string(buf) : fallback;
    std::free(buf);
    return v;
#else
    const char* raw = std::getenv(key);
    return raw ? std::string(raw) : fallback;
#endif
}

std::string error_body(ServeError kind, const std::string& message) {
    json j;
    j["error"]["code"] = to_string(kind);
    j["error"]["message"] = message;
    j["error"]["type"] = "invalid_request_error";
    return j.dump();
}

/// Checkpoint key for a conversation.
///
/// Sanitised because the key becomes a FILENAME and arrives from a client. A
/// conversation id of "../../etc/passwd" must not decide where the engine writes.
std::string session_key(const std::string& conversation) {
    std::string safe;
    safe.reserve(conversation.size() + 8);
    for (const char c : conversation) {
        const bool allowed = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                             (c >= '0' && c <= '9') || c == '-' || c == '_';
        safe.push_back(allowed ? c : '_');
    }
    if (safe.size() > 96) safe.resize(96);
    return "conv-" + safe;
}

/// A telemetry tick as JSON. Flat on purpose: this is emitted at up to 10 Hz to
/// every watcher, and a nested shape would cost more to build than to send.
json telemetry_frame_json(const TelemetryFrame& f) {
    const auto lookups = f.cache.hits + f.cache.misses;
    return json{
        {"tick_ms", f.tick_ms},
        // True when a field was carried over because its lock was held by the
        // work being measured. A client that plots these should know which
        // samples are repeats rather than a plateau.
        {"stale", f.stale},
        {"tier",
         {{"vram_experts", f.occupancy.vram_experts},
          {"ram_experts", f.occupancy.ram_experts},
          {"disk_experts", f.occupancy.disk_experts},
          {"pinned_experts", f.occupancy.pinned_experts},
          {"ram_bytes", f.occupancy.ram_bytes},
          {"ram_capacity_bytes", f.occupancy.ram_capacity_bytes}}},
        {"cache",
         {{"hits", f.cache.hits},
          {"misses", f.cache.misses},
          {"evictions", f.cache.evictions},
          {"prefetch_hits", f.cache.prefetch_hits},
          {"prefetch_wasted", f.cache.prefetch_wasted},
          {"bytes_read", f.cache.bytes_read},
          {"hit_rate",
           lookups > 0 ? static_cast<double>(f.cache.hits) / static_cast<double>(lookups) : 0.0}}},
        {"scheduler",
         {{"active_sequences", f.scheduler.active_sequences},
          {"queued_sequences", f.scheduler.queued_sequences},
          {"current_batch", f.scheduler.current_batch},
          {"max_batch_seen", f.scheduler.max_batch_seen},
          {"effective_max_batch", f.scheduler.effective_max_batch},
          {"prefill_rows", f.scheduler.prefill_rows_last_step},
          {"decode_rows", f.scheduler.decode_rows_last_step},
          // The payoff, on the wire. A ratio near 1.0 means the
          // union is buying nothing and something upstream is wrong.
          {"unique_experts", f.scheduler.unique_experts_last_step},
          {"naive_expert_reads", f.scheduler.naive_expert_reads_last_step},
          {"steps", f.scheduler.steps},
          {"tokens_out", f.scheduler.tokens_out},
          {"preemptions", f.scheduler.preemptions},
          {"speculative_draft_tokens", f.scheduler.speculative_draft_tokens},
          {"speculative_accepted_tokens", f.scheduler.speculative_accepted_tokens},
          {"speculative_verifications", f.scheduler.speculative_verifications},
          {"speculative_fallback_steps", f.scheduler.speculative_fallback_steps},
          {"speculative_acceptance_rate",
           f.scheduler.speculative_draft_tokens > 0
               ? static_cast<double>(f.scheduler.speculative_accepted_tokens) /
                     static_cast<double>(f.scheduler.speculative_draft_tokens)
               : 0.0}}},
    };
}

/// The brain grid. Cells are emitted as flat parallel arrays rather than an
/// array of objects: at 4096 cells the object form is roughly 6x the bytes for
/// the same information, and this goes out on every tick.
json heat_frame_json(const HeatFrame& h) {
    json counts = json::array();
    json tiers = json::array();
    counts.get_ref<json::array_t&>().reserve(h.cells.size());
    tiers.get_ref<json::array_t&>().reserve(h.cells.size());
    for (const auto& c : h.cells) {
        counts.push_back(c.count);
        tiers.push_back(static_cast<int>(c.tier));
    }
    return json{
        {"tick_ms", h.tick_ms},
        {"resolution", to_string(h.resolution)},
        {"n_layers", h.n_layers},
        {"n_experts", h.n_experts},
        {"layer_bucket", h.layer_bucket},
        {"expert_bucket", h.expert_bucket},
        {"rows", h.layer_bucket ? (h.n_layers + h.layer_bucket - 1) / h.layer_bucket : 0},
        {"cols", h.expert_bucket ? (h.n_experts + h.expert_bucket - 1) / h.expert_bucket : 0},
        {"counts", std::move(counts)},
        {"tiers", std::move(tiers)}};
}

/// One message's role, or a refusal naming the role that was sent.
Status role_of(const json& m, MessageRole& out, ServeError& err) {
    const auto role = m.value("role", "user");
    if (role == "system" || role == "developer") {
        // OpenAI renamed `system` to `developer` and every chat template still
        // spells it `system`. Mapping is right; refusing would reject a request
        // that is valid against the API this endpoint claims to implement.
        out = MessageRole::System;
    } else if (role == "user") {
        out = MessageRole::User;
    } else if (role == "assistant") {
        out = MessageRole::Assistant;
    } else if (role == "tool" || role == "function") {
        out = MessageRole::Tool;
    } else {
        err = ServeError::BadRequest;
        return {StatusCode::InvalidArgument, "unknown message role '" + role + "'"};
    }
    return {};
}

/// The text of one message, refusing what a text-only engine cannot represent.
Status content_of(const json& m, std::string& out, ServeError& err) {
    out.clear();
    const auto it = m.find("content");
    if (it == m.end() || it->is_null()) return {};
    if (it->is_string()) {
        out = it->get<std::string>();
        return {};
    }
    if (!it->is_array()) {
        err = ServeError::BadRequest;
        return {StatusCode::InvalidArgument, "message content must be string or array"};
    }
    for (const auto& part : *it) {
        const auto type = part.value("type", "");
        if (type != "text") {
            err = ServeError::UnsupportedContent;
            return {StatusCode::Unsupported,
                    "content part '" + type + "' is not supported; this engine is text-only"};
        }
        out += part.value("text", "");
    }
    return {};
}

/// Build the prompt from the container's compiled chat template.
///
/// REFUSES what the template cannot express rather than dropping it. A request
/// carrying `tools` gets a 422 naming them, because the compiled scaffold covers
/// conversation framing and not tool-call encoding — and a model told nothing
/// about the tools it was asked to use answers fluently, in prose, about
/// functions it never saw.
Status build_chat_prompt(const CompiledTokenizer& tokenizer,
                         const json& body,
                         std::vector<TokenId>& out,
                         ServeError& err) {
    if (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty()) {
        err = ServeError::UnsupportedContent;
        return {StatusCode::Unsupported,
                "tools are not supported by the compiled chat template; it covers "
                "conversation framing, not tool-call encoding"};
    }

    std::vector<std::string> contents;
    std::vector<MessageRole> roles;
    for (const auto& m : body["messages"]) {
        if (m.contains("tool_calls") && !m["tool_calls"].empty()) {
            err = ServeError::UnsupportedContent;
            return {StatusCode::Unsupported,
                    "an assistant message carrying tool_calls cannot be re-rendered "
                    "by the compiled chat template"};
        }
        MessageRole role{};
        if (auto st = role_of(m, role, err); !st.ok()) return st;
        std::string text;
        if (auto st = content_of(m, text, err); !st.ok()) return st;
        // `reasoning_content` is the other spelling of a thinking block, and
        // admission checked that this template renders the two identically
        // before setting the flag that says so. Folding it back into the
        // content is what lets ONE channel carry both.
        if (tokenizer.chat_template().has(chat_flag::kAssistantSplitsThink) &&
            role == MessageRole::Assistant && m.contains("reasoning_content") &&
            m["reasoning_content"].is_string() &&
            text.find("</think>") == std::string::npos) {
            text = "<think>" + m["reasoning_content"].get<std::string>() + "</think>" + text;
        }
        roles.push_back(role);
        contents.push_back(std::move(text));
    }

    std::vector<ChatMessage> messages;
    messages.reserve(roles.size());
    for (std::size_t i = 0; i < roles.size(); ++i) {
        messages.push_back(ChatMessage{roles[i], contents[i], {}});
    }

    ChatOptions options;
    options.add_generation_prompt = body.value("add_generation_prompt", true);
    options.enable_thinking = body.value("enable_thinking", true);
    if (body.contains("clear_thinking") && body["clear_thinking"].is_boolean()) {
        options.clear_thinking_set = true;
        options.clear_thinking = body["clear_thinking"].get<bool>();
    }
    if (body.contains("reasoning_effort") && body["reasoning_effort"].is_string()) {
        options.reasoning_effort = body["reasoning_effort"].get<std::string>();
    }

    if (auto st = tokenizer.apply_chat_template(messages, options, out); !st.ok()) {
        // Unsupported here is the caller asking for an option this model's
        // template does not have. That is a request problem, not a server one,
        // and answering 500 would send them looking in the wrong place.
        err = (st.code() == StatusCode::Unsupported) ? ServeError::UnsupportedContent
                                                     : ServeError::BadRequest;
        return st;
    }
    return {};
}

/// Flatten OpenAI `messages` into a prompt, for a container with no compiled
/// chat template.
///
/// NOT a lesser version of the same thing. `user: hi` / `assistant:` is not what
/// any of these models was trained on, and a model served this way answers
/// fluently to a question framed differently from every one it saw in training.
/// It stays because it is better than a 500 and because it is VISIBLY different
/// rather than subtly so — the alternative to an honest fallback is a template
/// this compiler approximated, which is the failure that looks fine.
///
/// Image content parts are REFUSED with 422 rather than dropped. Silently
/// discarding them is the failure mode worth designing out: the request
/// succeeds, the answer ignores the picture, and nothing in the response says
/// why.
Status flatten_messages(const json& msgs, std::string& out, ServeError& err) {
    for (const auto& m : msgs) {
        const auto role = m.value("role", "user");
        const auto& c = m.at("content");
        if (c.is_string()) {
            out += role + ": " + c.get<std::string>() + "\n";
            continue;
        }
        if (!c.is_array()) {
            err = ServeError::BadRequest;
            return {StatusCode::InvalidArgument, "message content must be string or array"};
        }
        for (const auto& part : c) {
            const auto type = part.value("type", "");
            if (type == "text") {
                out += role + ": " + part.value("text", "") + "\n";
            } else {
                err = ServeError::UnsupportedContent;
                return {StatusCode::Unsupported,
                        "content part '" + type + "' is not supported; this engine is text-only"};
            }
        }
    }
    out += "assistant:";
    return {};
}

} // namespace

struct ServeServer::Impl {
    ServeConfig cfg;
    PlanDocument plan_doc;

    F32Model model;
    ExpertStore store;
    MemoryHierarchy memory;
    KvCheckpointStore checkpoints;
    CompiledTokenizer tokenizer;
    bool have_tokenizer = false;
    const PromptCodec* prompt_codec = nullptr;
    bool speculative_selected = false;

    Scheduler sched;

    /// Guards `sessions` and `waiters` only — NOT the scheduler.
    ///
    /// LOCK ORDER, and the one rule this file has to keep: the scheduler's own
    /// lock is taken first, always. Its token and finish callbacks fire from
    /// inside step() and take state_mu, so any path that holds state_mu while
    /// calling into the Scheduler inverts the order and deadlocks. Every call
    /// site below releases state_mu before touching `sched`.
    std::mutex state_mu;

    httplib::Server http;
    std::atomic<bool> is_ready{false};

    bool sched_open = false;

    // ── the shared step loop ─────────────────────────────────────────────────
    //
    // One thread drives step(); request threads admit and wait. That is what
    // makes the batch union reachable over HTTP: its payoff comes from
    // CONCURRENT sequences sharing a step, and a mutex held across a whole turn
    // gives it exactly one sequence to union.
    std::thread stepper;
    std::mutex work_mu;
    std::condition_variable work_cv;
    std::atomic<bool> stop_stepping{false};

    /// One in-flight turn. Owned by the request thread, reached by the callbacks.
    struct Waiter {
        std::mutex mu;
        std::condition_variable cv;

        /// Incremental decode. Per-turn because it carries a partial codepoint
        /// across tokens — the state that makes streaming exact rather than a
        /// repeated full decode.
        std::unique_ptr<CompiledTokenizer::Streamer> stream;

        std::string acc;
        std::vector<TokenId> token_ids;
        bool done = false;
        FinishReason finish_reason = FinishReason::Length;
        Status error;
        std::function<void(const std::string&)> on_delta;
    };

    std::map<SeqId, std::shared_ptr<Waiter>> waiters;

    /// conversation key -> the sequence holding its KV.
    ///
    /// The whole point: a sequence now outlives the request that created it.
    /// Without this every turn re-prefills the entire conversation, which is
    /// exactly the cost warm reopen exists to remove — and there is nothing live
    /// for the node to checkpoint at suspend time either, so cluster
    /// suspend/restore has nothing to save.
    struct Session {
        SeqId seq = 0;
        std::uint64_t last_used = 0;
    };

    std::map<std::string, Session> sessions;
    std::uint64_t session_clock = 0;

    /// Open ONCE, at ServeServer::open().
    ///
    /// It used to be re-opened per request, which discarded every sequence and
    /// made sessions impossible before the question was even asked.
    Status open_scheduler() {
        SchedulerConfig sc;
        sc.kv_slots = std::max(1u, cfg.kv_slots);
        sc.ctx_size = cfg.ctx_size;
        sc.max_batch = cfg.max_batch;
        sc.enable_speculation = speculative_selected;
        sc.speculative_tokens = cfg.speculative_tokens;
        sc.speculative_confidence_threshold = cfg.speculative_confidence_threshold;
        if (auto st = sched.open_f32(model, memory_ptr(), sc, &checkpoints); !st.ok()) return st;

        // Set ONCE, dispatching by SeqId. Re-registering per request was fine
        // when one turn ran at a time and is not: the last writer would own every
        // sequence's tokens.
        sched.set_token_callback([this](SeqId id, TokenId t, bool) { on_token(id, t); });
        sched.set_finish_callback(
            [this](SeqId id, FinishReason reason) { finish(id, {}, reason); });
        sched.set_error_callback([this](SeqId id, StatusCode code, const char* what) {
            finish(id, Status{code, what ? what : "sequence failed"}, FinishReason::Length);
        });

        sched_open = true;
        stop_stepping.store(false);
        stepper = std::thread([this] { step_loop(); });
        return {};
    }

    void close_scheduler() {
        stop_stepping.store(true);
        work_cv.notify_all();
        if (stepper.joinable()) stepper.join();
        sched_open = false;
        // Once the driver is gone nothing will ever finish a waiting turn, so
        // release them now. Without this a request in flight at shutdown blocks
        // on its deadline — ten minutes of a connection that is never going to
        // be answered.
        fail_all({StatusCode::Cancelled, "engine is shutting down"});
    }

    /// The driver. Steps while there is work, sleeps when there is not.
    void step_loop() {
        while (!stop_stepping.load()) {
            if (sched.idle()) {
                std::unique_lock<std::mutex> lk(work_mu);
                // Timed, not indefinite: a sequence admitted between the idle()
                // check and this wait would otherwise sit until the next admit
                // happened to arrive.
                work_cv.wait_for(lk, std::chrono::milliseconds(5));
                continue;
            }
            if (auto st = sched.step(); !st.ok()) {
                // step() re-queues the rows it failed on, so retrying forever
                // would spin on a permanent fault while every caller waits on a
                // response that is not coming. Fail the in-flight turns instead.
                fail_all(st);
            }
        }
    }

    void on_token(SeqId id, TokenId t) {
        std::shared_ptr<Waiter> w;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            if (auto it = waiters.find(id); it != waiters.end()) w = it->second;
        }
        if (!w) return; // a session with no turn in flight

        std::string delta;
        {
            std::lock_guard<std::mutex> lk(w->mu);
            w->token_ids.push_back(t);
            if (w->stream) {
                if (!w->stream->push(t, delta).ok()) return;
            } else {
                delta = std::to_string(t) + " ";
            }
            if (delta.empty()) return; // the codepoint is not finished yet
            w->acc += delta;
        }
        // Outside the waiter's lock: on_delta writes to a socket, and the
        // scheduler's lock is still held further up the stack. Holding a third
        // lock across a network write is how a slow client stalls every sequence
        // in the batch.
        if (w->on_delta) w->on_delta(delta);
    }

    void finish(SeqId id, Status error, FinishReason reason) {
        std::shared_ptr<Waiter> w;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            if (auto it = waiters.find(id); it != waiters.end()) w = it->second;
        }
        if (!w) return;

        // Release whatever the streamer is still holding. A turn that ends
        // mid-codepoint — a truncated byte-fallback sequence at max_tokens — has
        // no continuation coming, and those bytes belong in `acc` or the streamed
        // text and the returned text disagree.
        std::string tail;
        {
            std::lock_guard<std::mutex> lk(w->mu);
            if (w->stream) (void)w->stream->flush(tail);
            w->acc += tail;
        }
        // Before `done`, and outside the lock for the same reason as on_token:
        // once done is set the request thread returns and stops reading, so a
        // tail sent afterwards would be sent to nobody.
        if (!tail.empty() && w->on_delta) w->on_delta(tail);

        {
            std::lock_guard<std::mutex> lk(w->mu);
            if (!error.ok()) w->error = std::move(error);
            w->finish_reason = reason;
            w->done = true;
        }
        w->cv.notify_all();
    }

    void fail_all(const Status& error) {
        std::vector<std::shared_ptr<Waiter>> all;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            for (auto& [_, w] : waiters)
                all.push_back(w);
        }
        for (auto& w : all) {
            {
                std::lock_guard<std::mutex> lk(w->mu);
                w->error = error;
                w->done = true;
            }
            w->cv.notify_all();
        }
    }

    /// Retire the least-recently-used session to free a KV slot.
    ///
    /// A KV slot is real memory, so the scheduler REFUSES rather than silently
    /// evicting; deciding whose context to drop is a policy question and belongs
    /// here, where the sessions have names.
    ///
    /// A session with a turn IN FLIGHT is never the victim — evicting it would
    /// cancel a sequence someone is waiting on, and the wait would time out
    /// rather than fail.
    bool evict_lru_session(const std::string& except) {
        SeqId victim_seq = 0;
        std::string victim_key;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            std::uint64_t oldest = 0;
            for (const auto& [key, s] : sessions) {
                if (key == except) continue;
                if (waiters.count(s.seq) != 0) continue;
                if (victim_key.empty() || s.last_used < oldest) {
                    victim_key = key;
                    victim_seq = s.seq;
                    oldest = s.last_used;
                }
            }
            if (victim_key.empty()) return false;
            sessions.erase(victim_key);
        }
        // Outside state_mu: cancel() takes the scheduler's lock.
        (void)sched.cancel(victim_seq);
        return true;
    }

    /// Admit one turn and WAIT for it. The step loop belongs to nobody.
    ///
    /// This used to hold a mutex across the whole turn, which meant the batch
    /// union — the mechanism the entire engine is built around — only ever had
    /// one sequence to union over HTTP. Concurrent requests now land as
    /// concurrent rows in one forward, and each expert is read once for all of
    /// them regardless of which conversation asked.
    Status generate(const std::string& prompt,
                    std::uint32_t max_tokens,
                    const SamplerState& sampler,
                    const std::string& conversation,
                    const std::function<void(const std::string&)>& on_delta,
                    std::string& out_text,
                    std::vector<TokenId>* out_token_ids,
                    FinishReason& out_finish_reason,
                    // Already-assembled prompt ids, when the caller built them
                    // from a compiled chat template. `prompt` is then only the
                    // human-readable echo; re-encoding it would throw away the
                    // framing the template put there and is exactly the
                    // round-trip-through-text this design avoids.
                    const std::vector<TokenId>* prompt_ids = nullptr) {
        std::vector<TokenId> ids;
        if (prompt_ids != nullptr) {
            ids = *prompt_ids;
        } else if (have_tokenizer) {
            if (auto st = tokenizer.encode(prompt, ids); !st.ok()) return st;
        } else {
            // No tokenizer: fall back to BYTES, one token per byte, folded into
            // the vocabulary. Not a tokenizer and not pretending to be one — but
            // it makes prompt length track the prompt, which the previous
            // fallback (a single token 0 for every request, however long) did
            // not. A session cannot be exercised at all when every prompt
            // encodes identically.
            const auto vocab = model.vocab();
            ids.reserve(prompt.size());
            for (const unsigned char c : prompt) {
                ids.push_back(static_cast<TokenId>(vocab > 0 ? c % vocab : 0u));
            }
        }
        if (ids.empty()) ids.push_back(0);

        if (!sched_open) return {StatusCode::InvalidArgument, "scheduler is not open"};

        SeqId id = 0;
        bool warm = false;

        // ── continue an existing session ─────────────────────────────────────
        //
        // The session is LOOKED UP under state_mu and the scheduler is called
        // after releasing it. Holding state_mu across extend() would invert the
        // lock order against the callbacks and deadlock against the step loop.
        SeqId existing = 0;
        {
            std::lock_guard<std::mutex> lk(state_mu);
            if (!conversation.empty()) {
                if (auto it = sessions.find(conversation); it != sessions.end()) {
                    existing = it->second.seq;
                }
            }
        }
        if (existing != 0) {
            if (sched.extend(existing, ids, max_tokens).ok()) {
                id = existing;
                warm = true;
            } else {
                // The prompt is not an extension of what the cache holds — an
                // edited earlier turn, or a client reusing a key. Retire the
                // sequence and start cold: correct output costs a re-prefill,
                // a mismatched cache costs correctness.
                (void)sched.cancel(existing);
                std::lock_guard<std::mutex> lk(state_mu);
                if (auto it = sessions.find(conversation);
                    it != sessions.end() && it->second.seq == existing) {
                    sessions.erase(it);
                }
            }
        }

        // ── or start a new one ───────────────────────────────────────────────
        AdmitRejection why{};
        if (!warm) {
            SeqRequest req;
            req.prompt = ids;
            req.max_tokens = max_tokens;
            req.sampler = sampler;
            // Model-owned chat protocols also own their special-token boundary.
            // Existing generic families retain their byte-identical generation
            // behavior; V4's codec stops on EOS before it can enter protocol
            // parsing or public content.
            //
            // A prompt assembled from the model's OWN chat template earns the
            // same treatment, and only now does it mean anything. A flattened
            // `user:` / `assistant:` transcript is not a framing the model was
            // trained to end, so it rambled to max_tokens and stopping on EOS
            // would have changed nothing; a correctly framed turn ends with the
            // tokens `eos_token_id` names, and running past them is the model
            // answering its own next question.
            if (prompt_codec != nullptr || prompt_ids != nullptr) {
                req.stop_token_ids = model.arch.topology.eos_token_ids;
            }
            // Cross-PROCESS warm reopen: if this conversation was checkpointed
            // before the engine was stopped, attach that cache. Same mechanism,
            // one tier up — see kv_checkpoint.hpp's "one format, three callers".
            if (!conversation.empty()) req.resume_key = session_key(conversation);

            SeqRequest retry = req;
            auto st = sched.admit(std::move(req), id, why);
            if (!st.ok() && why == AdmitRejection::NoKvSlot && evict_lru_session(conversation)) {
                st = sched.admit(std::move(retry), id, why);
            }
            if (!st.ok()) return st;
        }

        // Register the waiter BEFORE waking the stepper, or a fast turn can
        // finish before anyone is listening for it.
        //
        // Decoding happens in the callback, incrementally. It used to re-decode
        // the whole emitted prefix each step and send the suffix — O(n^2) in tokens,
        // inside the scheduler's lock, so the cost fell on every OTHER sequence in
        // the batch as well. It was also wrong at the boundary it looked right at:
        // a codepoint split across two tokens produced a delta that was a bare
        // lead byte, which is not text.
        auto waiter = std::make_shared<Waiter>();
        waiter->on_delta = on_delta;
        if (have_tokenizer) {
            waiter->stream = std::make_unique<CompiledTokenizer::Streamer>(tokenizer);
        }
        {
            std::lock_guard<std::mutex> lk(state_mu);
            waiters[id] = waiter;
            if (!conversation.empty()) sessions[conversation] = Session{id, ++session_clock};
        }
        work_cv.notify_one();

        // Bounded, and generously: the deadline exists so a lost finish signal
        // presents as a failed request rather than a hung connection, not as a
        // generation limit. A step-loop fault reaches the waiter through
        // fail_all() long before this fires.
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(cfg.generation_timeout_seconds);
        Status result;
        {
            std::unique_lock<std::mutex> lk(waiter->mu);
            const bool finished = waiter->cv.wait_until(lk, deadline, [&] { return waiter->done; });
            if (!finished) {
                result = {StatusCode::Internal,
                          "generation did not complete within " +
                              std::to_string(cfg.generation_timeout_seconds) + " seconds"};
            } else {
                result = waiter->error;
            }
            out_text = waiter->acc;
            if (out_token_ids != nullptr) *out_token_ids = waiter->token_ids;
            out_finish_reason = waiter->finish_reason;
        }

        {
            std::lock_guard<std::mutex> lk(state_mu);
            if (auto it = waiters.find(id); it != waiters.end() && it->second == waiter) {
                waiters.erase(it);
            }
        }

        // A turn with no conversation key has nobody to come back to it, so its
        // sequence is retired here. Retaining finished sequences is what makes a
        // session warm; doing it for stateless turns as well would hold a KV slot
        // per request until the pool was exhausted and every later request was
        // refused with NoKvSlot.
        if (conversation.empty()) (void)sched.cancel(id);
        return result;
    }

    MemoryHierarchy* memory_ptr() { return model.experts_are_streamed ? &memory : nullptr; }

    // ── telemetry ────────────────────────────────────────────────────────────
    //
    // ONE channel sampling the engine, fanned out to N watchers — not one
    // sampler per connection. The whole premise is that aggregation costs the
    // engine once regardless of how many clients are looking; a per-connection
    // sampler would make that cost linear in watchers, which is the thing the
    // design says it avoids.
    TelemetryChannel telemetry;

    struct TelemetryFeed {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> frames;
        bool closed = false;
        std::uint32_t hz = kDefaultTelemetryHz;
        HeatResolution resolution = HeatResolution::Bucketed;
    };

    std::mutex feeds_mu;
    std::vector<std::weak_ptr<TelemetryFeed>> feeds;

    /// Frames a watcher may fall behind by before the oldest are dropped.
    ///
    /// OLD frames go, not new ones: telemetry is a sample of NOW, and a stale
    /// frame delivered late is worse than a gap. Without a bound the engine
    /// would spend memory on a client that has already stopped reading.
    static constexpr std::size_t kMaxQueuedFrames = 32;

    void attach_feed(const std::shared_ptr<TelemetryFeed>& feed) {
        std::lock_guard<std::mutex> lk(feeds_mu);
        feeds.push_back(feed);
        retune_locked();
    }

    void detach_feed(const std::shared_ptr<TelemetryFeed>& feed) {
        std::lock_guard<std::mutex> lk(feeds_mu);
        feeds.erase(std::remove_if(feeds.begin(),
                                   feeds.end(),
                                   [&](const std::weak_ptr<TelemetryFeed>& weak) {
                                       auto f = weak.lock();
                                       return !f || f == feed;
                                   }),
                    feeds.end());
        retune_locked();
    }

    /// The channel samples ONCE, so it must sample at least as often and as
    /// finely as its most demanding watcher. Recomputed on every attach and
    /// detach so a departing client's higher rate does not persist.
    void retune_locked() {
        std::uint32_t fastest = kDefaultTelemetryHz;
        bool want_full = false;
        for (const auto& weak : feeds) {
            if (auto f = weak.lock()) {
                fastest = std::max(fastest, f->hz);
                want_full |= (f->resolution == HeatResolution::Full);
            }
        }
        telemetry.set_rate(fastest);
        telemetry.set_heat_resolution(want_full ? HeatResolution::Full : HeatResolution::Bucketed);
    }

    void broadcast(const std::string& payload) {
        std::vector<std::shared_ptr<TelemetryFeed>> live;
        {
            std::lock_guard<std::mutex> lk(feeds_mu);
            for (const auto& weak : feeds) {
                if (auto f = weak.lock()) live.push_back(std::move(f));
            }
        }
        for (auto& f : live) {
            {
                std::lock_guard<std::mutex> lk(f->mu);
                if (f->frames.size() >= kMaxQueuedFrames) f->frames.pop_front();
                f->frames.push_back(payload);
            }
            f->cv.notify_one();
        }
    }

    void close_feeds() {
        std::vector<std::shared_ptr<TelemetryFeed>> live;
        {
            std::lock_guard<std::mutex> lk(feeds_mu);
            for (const auto& weak : feeds) {
                if (auto f = weak.lock()) live.push_back(std::move(f));
            }
            feeds.clear();
        }
        for (auto& f : live) {
            {
                std::lock_guard<std::mutex> lk(f->mu);
                f->closed = true;
            }
            f->cv.notify_all();
        }
    }

    // ── suspend / restore ────────────────────────────────────────────────────
    //
    // The node asks for ONE artifact, and this engine holds N sessions. That is
    // the difference the KvCheckpointBackend interface calls
    // supports_multi_sequence(): llama.cpp answers false and saves sequence 0,
    // silently losing the rest; Soma writes every session and a manifest naming
    // them, so a restore brings back the whole engine rather than one agent's
    // context.

    /// A snapshot of the session table, taken under state_mu and used after it
    /// is released. Every scheduler call below needs the lock released first.
    std::vector<std::pair<std::string, Session>> snapshot_sessions() {
        std::lock_guard<std::mutex> lk(state_mu);
        return {sessions.begin(), sessions.end()};
    }

    Status save_sessions(const std::string& manifest_path, std::uint32_t& out_saved) {
        out_saved = 0;
        if (cfg.checkpoint_dir.empty()) {
            return {StatusCode::InvalidArgument, "engine was launched without --kv-dir"};
        }
        const auto snap = snapshot_sessions();

        json m;
        m["version"] = 1;
        m["engine"] = "soma";
        m["arch_hash"] = model.arch.arch_hash;
        m["written_at_ms"] =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                           std::chrono::system_clock::now().time_since_epoch())
                                           .count());
        auto& arr = m["sessions"] = json::array();

        std::uint32_t total_tokens = 0;
        for (const auto& [conversation, session] : snap) {
            const auto key = session_key(conversation);
            if (auto st = sched.checkpoint(session.seq, key); !st.ok()) {
                // One unsaveable session does not invalidate the others. Recorded
                // rather than dropped: a manifest that silently omits a session
                // is indistinguishable from one that never had it.
                m["skipped"].push_back(
                    json{{"conversation", conversation}, {"reason", st.message()}});
                continue;
            }
            std::vector<TokenId> toks;
            (void)sched.sequence_tokens(session.seq, toks);
            arr.push_back(
                json{{"conversation", conversation}, {"key", key}, {"tokens", toks.size()}});
            total_tokens += static_cast<std::uint32_t>(toks.size());
            ++out_saved;
        }
        m["total_tokens"] = total_tokens;

        std::ofstream out(manifest_path, std::ios::binary | std::ios::trunc);
        if (!out) return {StatusCode::IoError, "cannot write " + manifest_path};
        const auto text = m.dump(2);
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!out) return {StatusCode::IoError, "short write to " + manifest_path};
        return {};
    }

    /// Live per-sequence state — what EngineSupervisor::sequences() has had
    /// nothing to report. A stalled sequence and a saturated batch look identical
    /// from a request counter; they do not look identical here.
    json sessions_json() {
        const auto snap = snapshot_sessions();
        const auto st = sched.stats();
        json j;
        j["kv_slots"] = cfg.kv_slots;
        j["effective_max_batch"] = st.effective_max_batch;
        j["active_sequences"] = st.active_sequences;
        j["current_batch"] = st.current_batch;
        // The high-water mark, so a caller asking "did it ever batch?" reads a
        // fact instead of racing a sampler.
        j["max_batch_seen"] = st.max_batch_seen;
        // The payoff, made visible on the wire: unique experts read per step vs.
        // rows x top_k. A ratio near 1.0 means the union is buying nothing.
        j["unique_experts_last_step"] = st.unique_experts_last_step;
        j["naive_expert_reads_last_step"] = st.naive_expert_reads_last_step;
        auto& arr = j["sessions"] = json::array();
        for (const auto& [conversation, session] : snap) {
            std::vector<TokenId> toks;
            (void)sched.sequence_tokens(session.seq, toks);
            arr.push_back(json{{"conversation", conversation},
                               {"sequence", session.seq},
                               {"kv_tokens", toks.size()},
                               {"last_used", session.last_used}});
        }
        return j;
    }

    Status restore_sessions(const std::string& manifest_path, std::uint32_t& out_restored) {
        out_restored = 0;

        std::ifstream in(manifest_path, std::ios::binary);
        if (!in) return {StatusCode::NotFound, "no manifest at " + manifest_path};
        json m;
        try {
            in >> m;
        } catch (const std::exception& e) {
            return {StatusCode::InvalidArgument, std::string("bad manifest: ") + e.what()};
        }
        if (m.value("arch_hash", std::string{}) != model.arch.arch_hash) {
            // Refused, not read. The KV of a different architecture has the wrong
            // cache shape; every byte would load and the output would be quietly
            // degraded, which is the single most confusing bug this could produce.
            return {StatusCode::ArchMismatch,
                    "manifest arch_hash " + m.value("arch_hash", std::string{}) +
                        " != " + model.arch.arch_hash};
        }
        if (!sched_open) return {StatusCode::InvalidArgument, "scheduler is not open"};

        // The sessions are re-created lazily: the next request for a conversation
        // passes its key as resume_key and admit() attaches the cache after
        // checking it is a prefix. Restoring here would mean prefilling nothing
        // and holding KV slots for conversations that may never come back.
        for (const auto& s : m.value("sessions", json::array())) {
            const auto key = s.value("key", std::string{});
            if (key.empty() || !checkpoints.exists(key)) continue;
            ++out_restored;
        }
        return {};
    }
};

ServeServer::ServeServer() : impl_(std::make_unique<Impl>()) {}

ServeServer::~ServeServer() {
    stop();
}

Status ServeServer::open(const ServeConfig& config) {
    auto& im = *impl_;
    im.cfg = config;
    if (config.model_dir.empty()) {
        return {StatusCode::InvalidArgument, "--model-dir is required"};
    }

    // What this model IS, resolved the same way `soma plan` resolves it.
    //
    // This used to be three hardcoded lines — q4_g/q4_g/q6_g at group 128 — which
    // is not a fact about the model but a guess about the converter's defaults.
    // Two consequences, both real: a container converted at any other group was
    // refused with "container experts are N B but the IR's quantization map
    // implies M B", which reads as a corrupt container rather than a wrong
    // assumption; and the resident half could only ever be F32, so the
    // configuration `plan --quant-dense q4_g` blesses as `stream` was not
    // reachable through `serve` at all. main.cpp says the planner and the server
    // "must not be able to disagree" (roadmap D41).
    //
    // `--quant-dense` is expressed as an overlay in container_meta.json's shape
    // and applied by the same function the converter's own record goes through, so
    // there is no second dtype-to-role mapping to drift.
    std::string overlay;
    if (!config.quant_dense.empty()) {
        DType dense{};
        if (!parse_dtype(config.quant_dense, dense)) {
            return {StatusCode::InvalidArgument,
                    "unknown --quant-dense dtype '" + config.quant_dense + "'"};
        }
        if (dense != DType::F32 && !is_quantized(dense)) {
            return {StatusCode::Unsupported,
                    "--quant-dense " + config.quant_dense +
                        " is not implemented by the resident-weight loader"};
        }
        overlay = json{{"dtype_dense", config.quant_dense}}.dump();
    }
    ArchIr resolved;
    if (auto st = resolve_arch(config.model_dir, overlay, resolved); !st.ok()) return st;

    if (config.speculative == SpeculativeMode::Required && !resolved.speculative.present) {
        return {StatusCode::Unsupported,
                "--speculative dspark requires a DSpark-capable converted container"};
    }
    im.speculative_selected =
        config.speculative == SpeculativeMode::Required ||
        (config.speculative == SpeculativeMode::Auto && resolved.speculative.present &&
         resolved.speculative.profiled_speedup >= 1.05f);

    HostBudget host;
    host.ram_total_bytes = config.ram_budget_bytes ? config.ram_budget_bytes * 2 : (8ull << 30);
    host.ram_free_bytes = config.ram_budget_bytes ? config.ram_budget_bytes : (4ull << 30);
    host.ctx_size = config.ctx_size;
    host.kv_slots = config.kv_slots;
    host.speculative = im.speculative_selected;
    if (auto st = compute_plan(resolved, host, im.plan_doc); !st.ok()) return st;
    const auto available = host.ram_free_bytes ? host.ram_free_bytes : host.ram_total_bytes;
    const auto committed = im.plan_doc.dense_resident_bytes + im.plan_doc.kv_bytes_at_ctx;
    if (resolved.schema_version >= kArchIrSchemaVersionV2 && available > 0 &&
        committed > available) {
        return {StatusCode::CapacityPressure, im.plan_doc.verdict_reason};
    }
    if (auto st = load_f32_model(config.model_dir, im.model, resolved.quantization); !st.ok()) {
        return st;
    }
    if (const auto* backend = resolve_f32_backend(im.model.arch)) {
        im.prompt_codec = backend->prompt_codec;
    }

    // A container directory streams; a plain checkpoint is resident. Both are
    // legitimate residency modes — `resident-only` is a verdict, not a failure —
    // so the server supports either without a flag.
    if (im.model.experts_are_streamed) {
        if (resolved.schema_version >= kArchIrSchemaVersionV2 &&
            im.plan_doc.expert_cache_bytes < im.plan_doc.expert_bytes) {
            return {StatusCode::CapacityPressure, im.plan_doc.verdict_reason};
        }
        if (auto st = im.store.open(config.model_dir, im.model.arch); !st.ok()) return st;
        MemoryBudget b;
        // Schema v2 requires exact admission: compute_plan reserves resident
        // weights and every selected KV slot before deriving this remainder.
        // Keep v1's established cache interpretation unchanged for old
        // containers, whose planner contract predates that hard admission gate.
        b.ram_expert_cache_bytes =
            resolved.schema_version >= kArchIrSchemaVersionV2
                ? im.plan_doc.expert_cache_bytes
                : (config.ram_budget_bytes ? config.ram_budget_bytes : (2ull << 30));
        if (im.speculative_selected && resolved.topology.n_layers > 0) {
            b.ram_expert_cache_bytes = b.ram_expert_cache_bytes * resolved.topology.n_layers /
                                       (resolved.topology.n_layers + resolved.speculative.n_layers);
        }
        b.pin_bytes = config.pin_bytes;
        if (auto st = im.memory.open(im.model.arch, im.store, b); !st.ok()) return st;
        im.model.streamed_experts = &im.memory;
    }

    if (im.speculative_selected) {
        const auto* speculative = resolve_speculative_backend(im.model.arch);
        if (speculative == nullptr || speculative->bind_model == nullptr ||
            speculative->start_runtime == nullptr) {
            return {StatusCode::Unsupported,
                    "the selected speculative method has no backend in this build"};
        }
        im.model.speculative_backend = speculative;
        if (const auto rc = speculative->bind_model(im.model, config.model_dir);
            rc != StatusCode::Ok) {
            return {rc, "binding speculative model weights failed"};
        }
        const auto draft_cache = im.plan_doc.expert_cache_bytes * resolved.speculative.n_layers /
                                 (resolved.topology.n_layers + resolved.speculative.n_layers);
        if (const auto rc = speculative->start_runtime(im.model, config.model_dir, draft_cache);
            rc != StatusCode::Ok) {
            return {rc, "starting speculative model runtime failed"};
        }
    }

    const auto tok = fs::path(config.model_dir) / "tokenizer.soma";
    if (fs::exists(tok)) {
        im.have_tokenizer = im.tokenizer.open(tok.string()).ok();
    }

    if (!config.checkpoint_dir.empty()) {
        (void)im.checkpoints.open(config.checkpoint_dir, im.model.arch);
    }

    // Opened ONCE, here — not per request. Re-opening it per request discarded
    // every sequence, which made a session impossible before the question of
    // sessions was even asked.
    if (auto st = im.open_scheduler(); !st.ok()) return st;

    // ── telemetry ────────────────────────────────────────────────────────────
    //
    // Started with the engine, not with the first watcher: the channel is what
    // SAMPLES, and sampling has to be running for a snapshot route to have
    // anything to return. It ticks at the default rate with no sinks attached,
    // which costs one atomic read per tick.
    (void)im.telemetry.open(im.memory, im.sched, config.telemetry_hz);
    im.telemetry.set_telemetry_sink([this](const TelemetryFrame& f) {
        impl_->broadcast("event: telemetry\ndata: " + telemetry_frame_json(f).dump() + "\n\n");
    });
    im.telemetry.set_heat_sink([this](const HeatFrame& h) {
        // A named event, so a client that only wants the cheap frame can ignore
        // the grid without parsing it — which is most of the payload.
        impl_->broadcast("event: heat\ndata: " + heat_frame_json(h).dump() + "\n\n");
    });

    const auto served = config.served_model_name.empty()
                            ? fs::path(config.model_dir).filename().string()
                            : config.served_model_name;

    // ── routes ───────────────────────────────────────────────────────────────

    im.http.Get("/health", [this, served](const httplib::Request&, httplib::Response& res) {
        json j;
        j["status"] = impl_->is_ready.load() ? "ok" : "loading";
        j["model"] = served;
        j["engine"] = "soma";
        j["streamed"] = impl_->model.experts_are_streamed;
        j["verdict"] = to_string(impl_->plan_doc.verdict);
        res.status = impl_->is_ready.load() ? 200 : 503;
        res.set_content(j.dump(), "application/json");
    });

    // The footprint source. `estimate_inference_vram_mb()` guesses from file
    // size; this is the planner's own answer, from the model actually loaded, so
    // the node reports what the engine is doing rather than what a size heuristic
    // predicted it would do.
    im.http.Get("/internal/plan", [this](const httplib::Request&, httplib::Response& res) {
        std::string body;
        if (auto st = serialize_plan(impl_->plan_doc, body); !st.ok()) {
            res.status = 500;
            res.set_content(
                json{{"error", {{"code", "internal"}, {"message", st.message()}}}}.dump(),
                "application/json");
            return;
        }
        res.set_content(body, "application/json");
    });

    // ── suspend / restore ────────────────────────────────────────────────────
    // The node's KvCheckpointBackend speaks to these. `path` is chosen by the
    // node and is absolute; the engine writes exactly there so the node's record
    // and the file on disk cannot disagree.
    im.http.Post("/internal/kv/save", [this](const httplib::Request& req, httplib::Response& res) {
        std::string path;
        try {
            path = json::parse(req.body).value("path", std::string{});
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(error_body(ServeError::BadRequest, e.what()), "application/json");
            return;
        }
        if (path.empty()) {
            res.status = 400;
            res.set_content(error_body(ServeError::BadRequest, "path is required"),
                            "application/json");
            return;
        }
        std::uint32_t saved = 0;
        if (auto st = impl_->save_sessions(path, saved); !st.ok()) {
            res.status = http_status_for(ServeError::Internal);
            res.set_content(error_body(ServeError::Internal, st.message()), "application/json");
            return;
        }
        res.set_content(json{{"saved", saved}, {"path", path}}.dump(), "application/json");
    });

    im.http.Post(
        "/internal/kv/restore", [this](const httplib::Request& req, httplib::Response& res) {
            std::string path;
            try {
                path = json::parse(req.body).value("path", std::string{});
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(error_body(ServeError::BadRequest, e.what()), "application/json");
                return;
            }
            std::uint32_t restored = 0;
            if (auto st = impl_->restore_sessions(path, restored); !st.ok()) {
                const auto kind = st.code() == StatusCode::NotFound ? ServeError::NotFound
                                                                    : ServeError::BadRequest;
                res.status = http_status_for(kind);
                res.set_content(error_body(kind, st.message()), "application/json");
                return;
            }
            res.set_content(json{{"restored", restored}}.dump(), "application/json");
        });

    im.http.Get("/internal/sessions", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(impl_->sessions_json().dump(), "application/json");
    });

    // ── GET /internal/telemetry ─────────────────────────────────────────────
    //
    // The tier/heat feed the node forwards and control re-publishes. Two things
    // are decided in the ENGINE rather than at the transport:
    //
    //   * the rate. `?hz=` is CLAMPED to [1, 10] rather than rejected — the
    //     ceiling is a property of the engine, not a mistake by the caller.
    //   * the resolution. Bucketed by default; `?resolution=full` is an explicit
    //     opt-in, so a client cannot ask for a 60k-cell grid by accident.
    //
    // Aggregation happens in MemoryHierarchy and is SAMPLED at the tick rate.
    // Nothing is emitted per token, so a client asking for maximum telemetry
    // cannot make the chat path pay for it.
    im.http.Get("/internal/telemetry", [this](const httplib::Request& req, httplib::Response& res) {
        std::uint32_t hz = kDefaultTelemetryHz;
        if (req.has_param("hz")) {
            try {
                hz = static_cast<std::uint32_t>(std::stoul(req.get_param_value("hz")));
            } catch (const std::exception&) {
                hz = kDefaultTelemetryHz;
            }
        }
        const auto resolution = req.get_param_value("resolution") == "full"
                                    ? HeatResolution::Full
                                    : HeatResolution::Bucketed;

        auto feed = std::make_shared<Impl::TelemetryFeed>();
        feed->hz = std::clamp<std::uint32_t>(hz, 1, kMaxTelemetryHz);
        feed->resolution = resolution;
        impl_->attach_feed(feed);

        res.set_chunked_content_provider("text/event-stream",
                                         [this, feed](std::size_t, httplib::DataSink& sink) {
                                             std::unique_lock<std::mutex> lk(feed->mu);
                                             feed->cv.wait_for(lk, std::chrono::seconds(5), [&] {
                                                 return !feed->frames.empty() || feed->closed;
                                             });
                                             if (feed->closed) {
                                                 lk.unlock();
                                                 impl_->detach_feed(feed);
                                                 sink.done();
                                                 return false;
                                             }
                                             std::deque<std::string> batch;
                                             batch.swap(feed->frames);
                                             lk.unlock();

                                             if (batch.empty()) {
                                                 // A comment frame: invisible to an SSE parser, and
                                                 // it keeps proxies from closing a connection that
                                                 // is merely idle because the engine has nothing to
                                                 // report.
                                                 static const std::string beat = ": keepalive\n\n";
                                                 return sink.write(beat.data(), beat.size());
                                             }
                                             for (const auto& line : batch) {
                                                 if (!sink.write(line.data(), line.size())) {
                                                     impl_->detach_feed(feed);
                                                     return false;
                                                 }
                                             }
                                             return true;
                                         });
    });

    // Non-streaming snapshot, for a client that wants one look rather than a
    // feed — and for the G3 text dump, which is the same data in the shape a
    // human reads.
    im.http.Get("/internal/heat", [this](const httplib::Request& req, httplib::Response& res) {
        const auto resolution = req.get_param_value("resolution") == "full"
                                    ? HeatResolution::Full
                                    : HeatResolution::Bucketed;
        HeatFrame frame;
        (void)impl_->telemetry.snapshot_heat(resolution, frame);
        res.set_content(heat_frame_json(frame).dump(), "application/json");
    });

    im.http.Get("/internal/telemetry/dump",
                [this](const httplib::Request&, httplib::Response& res) {
                    std::string text;
                    (void)impl_->telemetry.write_text_dump(text);
                    res.set_content(text, "text/plain");
                });

    im.http.Get("/v1/models", [served](const httplib::Request&, httplib::Response& res) {
        json j;
        j["object"] = "list";
        j["data"] = json::array({json{{"id", served}, {"object", "model"}, {"owned_by", "soma"}}});
        res.set_content(j.dump(), "application/json");
    });

    im.http.Post(
        "/v1/chat/completions",
        [this, served](const httplib::Request& req, httplib::Response& res) {
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception& e) {
                res.status = http_status_for(ServeError::BadRequest);
                res.set_content(error_body(ServeError::BadRequest, e.what()), "application/json");
                return;
            }
            if (!body.contains("messages") || !body["messages"].is_array()) {
                res.status = 400;
                res.set_content(error_body(ServeError::BadRequest, "messages[] is required"),
                                "application/json");
                return;
            }

            // Three ways to a prompt, in descending order of fidelity.
            //
            // A compiled chat template is the model's OWN framing, resolved to
            // ids at admission and graded there against the ids its real Jinja
            // template produces. `flatten_messages` is the fallback for a
            // container that has none, and it is not a lesser version of the
            // same thing — `user: hi\nassistant:` is not what any of these
            // models was trained on. It stays because a served model with the
            // wrong framing is better than a 500, and because it is visibly
            // different rather than subtly so.
            std::string prompt;
            std::vector<TokenId> prompt_ids;
            bool templated = false;
            PromptCodecState codec_state;
            ServeError err = ServeError::None;
            if (impl_->prompt_codec) {
                if (auto st = impl_->prompt_codec->encode(req.body, prompt, codec_state);
                    !st.ok()) {
                    err = st.code() == StatusCode::Unsupported ? ServeError::UnsupportedContent
                                                               : ServeError::BadRequest;
                    res.status = http_status_for(err);
                    res.set_content(error_body(err, st.message()), "application/json");
                    return;
                }
            } else if (impl_->have_tokenizer && impl_->tokenizer.chat_template().present) {
                if (auto st = build_chat_prompt(impl_->tokenizer, body, prompt_ids, err);
                    !st.ok()) {
                    res.status = http_status_for(err);
                    res.set_content(error_body(err, st.message()), "application/json");
                    return;
                }
                templated = true;
                // The echo, for logs and for the token-count fields. Never fed
                // back to the tokenizer — `prompt_ids` is what generates.
                if (!impl_->tokenizer.decode(prompt_ids, prompt).ok()) prompt.clear();
            } else {
                if (auto st = flatten_messages(body["messages"], prompt, err); !st.ok()) {
                    res.status = http_status_for(err);
                    res.set_content(error_body(err, st.message()), "application/json");
                    return;
                }
            }

            SamplerState sampler;
            sampler.temperature = body.value("temperature", 0.7f);
            sampler.top_p = body.value("top_p", 0.9f);
            sampler.rng_state = body.value("seed", 0ull);
            const auto max_tokens = body.value("max_tokens", 64u);
            const bool stream = body.value("stream", false);
            const bool return_token_ids = body.value("soma_return_token_ids", false);

            // The conversation key. Body field first, header second: the body is what
            // an OpenAI-shaped client can set, the header is what a proxy can add
            // without rewriting a payload. Absent means stateless — every turn cold,
            // which is the old behaviour and stays the default.
            std::string conversation = body.value("conversation", std::string{});
            if (conversation.empty() && req.has_header("X-Conversation-Id")) {
                conversation = req.get_header_value("X-Conversation-Id");
            }

            if (!stream) {
                std::string text;
                std::vector<TokenId> token_ids;
                FinishReason finish = FinishReason::Length;
                if (auto st = impl_->generate(prompt,
                                              max_tokens,
                                              sampler,
                                              conversation,
                                              nullptr,
                                              text,
                                              return_token_ids ? &token_ids : nullptr,
                                              finish,
                                              templated ? &prompt_ids : nullptr);
                    !st.ok()) {
                    const auto kind = (st.code() == StatusCode::CapacityPressure)
                                          ? ServeError::CapacityPressure
                                          : ServeError::Internal;
                    res.status = http_status_for(kind);
                    res.set_content(error_body(kind, st.message()), "application/json");
                    return;
                }
                json out;
                out["object"] = "chat.completion";
                out["model"] = served;
                json message{{"role", "assistant"}, {"content", text}};
                std::string finish_name = finish == FinishReason::Stop ? "stop" : "length";
                if (impl_->prompt_codec) {
                    PromptMessage parsed_message;
                    if (auto st = impl_->prompt_codec->parse(
                            text, codec_state, true, finish == FinishReason::Stop, parsed_message);
                        !st.ok()) {
                        res.status = http_status_for(ServeError::ProtocolError);
                        res.set_content(error_body(ServeError::ProtocolError, st.message()),
                                        "application/json");
                        return;
                    }
                    json calls = json::array();
                    for (const auto& call : parsed_message.tool_calls) {
                        calls.push_back(
                            json{{"id", call.id},
                                 {"type", "function"},
                                 {"function",
                                  json{{"name", call.name}, {"arguments", call.arguments}}}});
                    }
                    message =
                        json{{"role", "assistant"},
                             {"content", std::move(parsed_message.content)},
                             {"reasoning_content", std::move(parsed_message.reasoning_content)},
                             {"tool_calls", std::move(calls)}};
                    if (!parsed_message.tool_calls.empty()) {
                        finish_name = "tool_calls";
                    }
                }
                out["choices"] = json::array({json{{"index", 0},
                                                   {"message", std::move(message)},
                                                   {"finish_reason", finish_name}}});
                if (return_token_ids) out["soma_token_ids"] = std::move(token_ids);
                res.set_content(out.dump(), "application/json");
                return;
            }

            // SSE. Deltas are emitted as they are produced rather than buffered and
            // chunked at the end — a "streaming" endpoint that streams only after
            // the answer is complete is the thing clients notice immediately.
            res.set_chunked_content_provider(
                "text/event-stream",
                // Captured BY VALUE: the provider outlives this handler, and
                // `prompt_ids` is the prompt for a templated request the way
                // `prompt` is for every other one.
                [this, prompt, prompt_ids, templated, max_tokens, sampler, served,
                 conversation, codec_state](
                    std::size_t, httplib::DataSink& sink) {
                    auto send = [&](const json& j) {
                        const auto s = "data: " + j.dump() + "\n\n";
                        return sink.write(s.data(), s.size());
                    };
                    auto send_delta = [&](json delta) {
                        json chunk;
                        chunk["object"] = "chat.completion.chunk";
                        chunk["model"] = served;
                        chunk["choices"] =
                            json::array({json{{"index", 0}, {"delta", std::move(delta)}}});
                        (void)send(chunk);
                    };

                    std::string text;
                    std::string raw;
                    std::size_t reasoning_sent = 0;
                    std::size_t content_sent = 0;
                    bool calls_sent = false;
                    Status codec_error;

                    const auto emit_parsed = [&](bool final, bool stopped) -> Status {
                        PromptMessage message;
                        if (auto parsed = impl_->prompt_codec->parse(
                                raw, codec_state, final, stopped, message);
                            !parsed.ok())
                            return parsed;

                        const auto& reasoning = message.reasoning_content;
                        const auto& content = message.content;
                        if (reasoning.size() < reasoning_sent || content.size() < content_sent) {
                            return {StatusCode::Internal,
                                    "prompt codec produced a non-monotonic streaming prefix"};
                        }
                        if (reasoning.size() > reasoning_sent) {
                            send_delta(
                                json{{"reasoning_content", reasoning.substr(reasoning_sent)}});
                            reasoning_sent = reasoning.size();
                        }
                        if (content.size() > content_sent) {
                            send_delta(json{{"content", content.substr(content_sent)}});
                            content_sent = content.size();
                        }
                        if (!calls_sent && !message.tool_calls.empty()) {
                            json deltas = json::array();
                            for (std::size_t i = 0; i < message.tool_calls.size(); ++i) {
                                const auto& call = message.tool_calls[i];
                                deltas.push_back(json{
                                    {"index", i},
                                    {"id", call.id},
                                    {"type", "function"},
                                    {"function",
                                     json{{"name", call.name}, {"arguments", call.arguments}}}});
                            }
                            send_delta(json{{"tool_calls", std::move(deltas)}});
                            calls_sent = true;
                        }
                        return {};
                    };

                    FinishReason finish = FinishReason::Length;
                    const auto st = impl_->generate(
                        prompt,
                        max_tokens,
                        sampler,
                        conversation,
                        [&](const std::string& delta) {
                            if (impl_->prompt_codec) {
                                raw += delta;
                                if (codec_error.ok()) codec_error = emit_parsed(false, false);
                            } else {
                                send_delta(json{{"content", delta}});
                            }
                        },
                        text,
                        nullptr,
                        finish,
                        templated ? &prompt_ids : nullptr);
                    if (!st.ok()) {
                        json e;
                        e["error"]["code"] = to_string(st.code() == StatusCode::CapacityPressure
                                                           ? ServeError::CapacityPressure
                                                           : ServeError::Internal);
                        e["error"]["message"] = st.message();
                        (void)send(e);
                    } else if (impl_->prompt_codec) {
                        raw = text;
                        if (codec_error.ok()) {
                            codec_error = emit_parsed(true, finish == FinishReason::Stop);
                        }
                        if (!codec_error.ok()) {
                            json e;
                            e["error"]["code"] = to_string(ServeError::ProtocolError);
                            e["error"]["message"] = codec_error.message();
                            (void)send(e);
                        }
                    }
                    if (st.ok() && codec_error.ok()) {
                        std::string reason = finish == FinishReason::Stop ? "stop" : "length";
                        if (calls_sent) reason = "tool_calls";
                        json chunk;
                        chunk["object"] = "chat.completion.chunk";
                        chunk["model"] = served;
                        chunk["choices"] = json::array({json{
                            {"index", 0}, {"delta", json::object()}, {"finish_reason", reason}}});
                        (void)send(chunk);
                    }
                    const std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size());
                    sink.done();
                    return true;
                });
        });

    im.is_ready.store(true);
    return {};
}

Status ServeServer::listen() {
    auto& im = *impl_;
    if (!im.http.listen(im.cfg.host.c_str(), im.cfg.port)) {
        return {StatusCode::IoError,
                "cannot bind " + im.cfg.host + ":" + std::to_string(im.cfg.port)};
    }
    return {};
}

void ServeServer::stop() {
    if (impl_) {
        impl_->is_ready.store(false);
        impl_->http.stop();
        // Watchers first: they hold a condition variable the sampler notifies,
        // and closing the channel underneath a waiting reader would leave it
        // parked until its keepalive timeout.
        impl_->close_feeds();
        impl_->telemetry.close();
        // The step thread outlives the HTTP server by design — an in-flight turn
        // should reach its waiter — but it must not outlive this object. Joined
        // before Impl's members start being destroyed under it.
        impl_->close_scheduler();
    }
}

bool ServeServer::ready() const noexcept {
    return impl_->is_ready.load();
}

const PlanDocument& ServeServer::plan() const noexcept {
    return impl_->plan_doc;
}

const ServeConfig& ServeServer::config() const noexcept {
    return impl_->cfg;
}

Status parse_serve_config(int argc, const char* const* argv, ServeConfig& out) {
    // Env first, CLI second, so CLI wins. Both are supported because the node
    // launches with argv while a container image is configured with env, and
    // making one of them second-class means one caller has to work around it.
    out.host = env_or("SOMA_HOST", out.host);
    out.model_dir = env_or("SOMA_MODEL_DIR", out.model_dir);
    out.checkpoint_dir = env_or("SOMA_KV_DIR", out.checkpoint_dir);
    out.served_model_name = env_or("SOMA_SERVED_NAME", out.served_model_name);
    if (const auto p = env_or("SOMA_PORT", ""); !p.empty()) {
        out.port = static_cast<std::uint16_t>(std::stoul(p));
    }
    if (const auto c = env_or("SOMA_CTX_SIZE", ""); !c.empty()) {
        out.ctx_size = static_cast<std::uint32_t>(std::stoul(c));
    }
    if (const auto r = env_or("SOMA_RAM_BUDGET", ""); !r.empty()) {
        out.ram_budget_bytes = std::stoull(r);
    }
    if (const auto s = env_or("SOMA_KV_SLOTS", ""); !s.empty()) {
        out.kv_slots = static_cast<std::uint32_t>(std::stoul(s));
    }
    if (const auto b = env_or("SOMA_MAX_BATCH", ""); !b.empty()) {
        out.max_batch = static_cast<std::uint32_t>(std::stoul(b));
    }
    if (const auto t = env_or("SOMA_GENERATION_TIMEOUT", ""); !t.empty()) {
        out.generation_timeout_seconds = static_cast<std::uint32_t>(std::stoul(t));
    }
    out.quant_dense = env_or("SOMA_QUANT_DENSE", out.quant_dense);
    const auto parse_speculative = [](const std::string& value, SpeculativeMode& mode) {
        if (value == "off")
            mode = SpeculativeMode::Off;
        else if (value == "auto")
            mode = SpeculativeMode::Auto;
        else if (value == "dspark")
            mode = SpeculativeMode::Required;
        else
            return false;
        return true;
    };
    if (const auto s = env_or("SOMA_SPECULATIVE", "");
        !s.empty() && !parse_speculative(s, out.speculative)) {
        return {StatusCode::InvalidArgument, "SOMA_SPECULATIVE wants off, auto, or dspark"};
    }
    if (const auto n = env_or("SOMA_SPECULATIVE_TOKENS", ""); !n.empty())
        out.speculative_tokens = static_cast<std::uint32_t>(std::stoul(n));
    if (const auto c = env_or("SOMA_DSPARK_CONFIDENCE_THRESHOLD", ""); !c.empty())
        out.speculative_confidence_threshold = std::stof(c);

    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&](std::string& dst) {
            if (i + 1 < argc) dst = argv[++i];
        };
        if (a == "--host")
            next(out.host);
        else if (a == "--model-dir")
            next(out.model_dir);
        else if (a == "--kv-dir")
            next(out.checkpoint_dir);
        else if (a == "--served-name")
            next(out.served_model_name);
        else if (a == "--port" && i + 1 < argc)
            out.port = static_cast<std::uint16_t>(std::stoul(argv[++i]));
        else if (a == "--ctx-size" && i + 1 < argc)
            out.ctx_size = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (a == "--kv-slots" && i + 1 < argc)
            out.kv_slots = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (a == "--max-batch" && i + 1 < argc)
            out.max_batch = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (a == "--speculative" && i + 1 < argc) {
            if (!parse_speculative(argv[++i], out.speculative)) {
                return {StatusCode::InvalidArgument, "--speculative wants off, auto, or dspark"};
            }
        } else if (a == "--speculative-tokens" && i + 1 < argc)
            out.speculative_tokens = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (a == "--dspark-confidence-threshold" && i + 1 < argc)
            out.speculative_confidence_threshold = std::stof(argv[++i]);
        else if (a == "--generation-timeout" && i + 1 < argc)
            out.generation_timeout_seconds = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (a == "--ram-budget" && i + 1 < argc)
            out.ram_budget_bytes = std::stoull(argv[++i]);
        else if (a == "--pin" && i + 1 < argc)
            out.pin_bytes = std::stoull(argv[++i]);
        else if (a == "--quant-dense") {
            if (i + 1 >= argc) {
                return {StatusCode::InvalidArgument, "--quant-dense requires a dtype"};
            }
            out.quant_dense = argv[++i];
        }
    }
    if (out.model_dir.empty()) {
        return {StatusCode::InvalidArgument, "--model-dir (or SOMA_MODEL_DIR) is required"};
    }
    if (out.ctx_size == 0 || out.kv_slots == 0 || out.generation_timeout_seconds == 0 ||
        out.speculative_tokens == 0) {
        return {StatusCode::InvalidArgument,
                "context, KV slots, generation timeout, and speculative tokens must be positive"};
    }
    if (!std::isfinite(out.speculative_confidence_threshold) ||
        out.speculative_confidence_threshold < 0.0f ||
        out.speculative_confidence_threshold > 1.0f) {
        return {StatusCode::InvalidArgument,
                "--dspark-confidence-threshold must be between 0 and 1"};
    }
    return {};
}

} // namespace soma
