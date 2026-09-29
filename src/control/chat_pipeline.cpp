// Mantic-Mind — one user turn, and the model work around it: global recall,
// memory extraction, compaction, voice design.
//
// A turn, in order:
//   1. the model: the request's, else the conversation's, else the user's
//      default (resolve_turn_model). A model the REQUEST names sticks — it is
//      saved on the conversation.
//   2. bind it: place it on a node (waiting out transient capacity) or connect
//      to its API. The lease is held for the whole turn.
//   3. refuse what the engine it landed on cannot take — images to a text-only
//      engine — before anything is stored.
//   4. store the user's message; pick relevant global memories with the model.
//   5. fold history into the rolling summary if the prompt nears the model's
//      window (conversation_context.hpp). Fails the turn with context_overflow
//      if even that is not enough.
//   6. infer, streaming; run the memory tools the model calls, if tools were
//      offered — and only then. Every assistant message records the model.
//
// Everything here runs on the conversation's strand of `conversations_`, so two
// turns of one conversation never overlap, while other conversations run.

#include "control/control_api_server.hpp"

#include "common/conversation_context.hpp"
#include "common/engine_capabilities.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/memory_manager.hpp"
#include "common/tool_executor.hpp"
#include "common/trace_provenance.hpp"
#include "common/user_db.hpp"
#include "common/util.hpp"
#include "control/api_support.hpp"
#include "control/conversation_queue.hpp"
#include "control/node_registry.hpp"
#include "control/placement_scheduler.hpp"
#include "control/served_model_catalog.hpp"
#include "control/user_manager.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <future>
#include <thread>

namespace mm {

using namespace api;

struct ControlApiServer::BoundModel {
    ServedModel model;
    bool api = false;
    std::string engine_id; ///< empty for an API model
    ModelCapabilities caps;
    std::optional<PlacementScheduler::Lease> lease;
    NodeInfo node;
    SlotId slot_id;
    std::string request_model;
    std::unique_ptr<RuntimeClient> client;
};

namespace {

bool has_image_parts(const std::vector<MessageContentPart>& parts) {
    return std::any_of(parts.begin(), parts.end(), [](const MessageContentPart& p) {
        return p.type == "image_attachment" || p.type == "image_url";
    });
}

std::string join_instructions(const UserProfile& user, const Conversation& conv) {
    std::string text = util::trim(user.custom_instructions);
    const std::string own = util::trim(conv.system_prompt);
    if (!own.empty()) text += (text.empty() ? "" : "\n\n") + own;
    return text;
}

/// The newest messages that fit `max_tokens`, oldest first, starting at a user
/// turn where one is available.
std::vector<Message> newest_that_fit(std::vector<Message> messages, int max_tokens, bool vision) {
    int used = 0;
    std::size_t start = messages.size();
    while (start > 0) {
        const int cost = estimate_tokens(messages[start - 1], vision);
        if (used + cost > max_tokens) break;
        used += cost;
        --start;
    }
    while (start < messages.size() && messages[start].role != MessageRole::User) ++start;
    messages.erase(messages.begin(), messages.begin() + static_cast<std::ptrdiff_t>(start));
    return messages;
}

/// The model's reply to a summarizer prompt, or nullopt when it failed.
std::optional<std::string> complete_text(RuntimeClient& client,
                                         const std::string& request_model,
                                         const RuntimeSettings& base,
                                         const std::vector<Message>& prompt,
                                         int max_tokens,
                                         float temperature) {
    InferenceRequest req;
    req.model = request_model;
    req.messages = prompt;
    req.settings = base;
    req.settings.max_tokens = max_tokens;
    req.settings.temperature = temperature;
    req.stream = false;
    const Message out = client.complete(req);
    if (out.role != MessageRole::Assistant || util::trim(out.content).empty()) return std::nullopt;
    return out.content;
}

} // namespace

// ── Binding a model ───────────────────────────────────────────────────────────

std::string ControlApiServer::queue_key(const UserId& user_id, const ConvId& conv_id) {
    return user_id + '\x1f' + conv_id;
}

bool ControlApiServer::run_on_conversation(const UserId& user_id,
                                           const ConvId& conv_id,
                                           std::function<void()> fn) {
    auto finished = std::make_shared<std::promise<bool>>();
    auto ran = finished->get_future();
    const bool queued = conversations_->enqueue(
        queue_key(user_id, conv_id),
        [fn = std::move(fn), finished] {
            try {
                fn();
            } catch (const std::exception& e) {
                MM_ERROR("conversation job failed: {}", e.what());
            }
            finished->set_value(true);
        },
        [finished] { finished->set_value(false); });
    return queued && ran.get();
}

std::optional<ServedModel> ControlApiServer::available_model(const std::string& id,
                                                             std::string& error,
                                                             std::string& code) const {
    if (served_models_ == nullptr) {
        error = "the served-model catalog is not available";
        code = "catalog_unavailable";
        return std::nullopt;
    }
    const auto entry = served_models_->find(id);
    if (!entry) {
        error = "model not found: " + id;
        code = "model_not_found";
        return std::nullopt;
    }
    if (!entry->available) {
        error = "model " + id + " is unavailable: " + entry->unavailable_reason;
        code = "model_unavailable";
        return std::nullopt;
    }
    return entry->model;
}

std::unique_ptr<ControlApiServer::BoundModel> ControlApiServer::bind_model(
    const ServedModel& model,
    std::chrono::milliseconds wait,
    std::string& error,
    std::string& code) {
    auto bound = std::make_unique<BoundModel>();
    bound->model = model;
    if (!model.node_local()) {
        if (model.inference_backend != "api") {
            error = "model " + model.id + " has no backend (inference_backend '" +
                    model.inference_backend + "')";
            code = "model_unavailable";
            return nullptr;
        }
        bound->api = true;
        bound->client = make_api_runtime_client(model.api_settings);
        bound->request_model = model.model_path;
        bound->caps = capabilities_for(model, {}, vllm_tool_choice());
        return bound;
    }

    auto outcome = scheduler_.acquire(model, wait);
    if (!outcome.ok()) {
        error = outcome.error.empty() ? std::string("no node available or model failed to load")
                                      : outcome.error;
        if (placement_failure_transient(outcome.failure))
            error += " (timed out waiting for available node)";
        code = to_string(outcome.failure);
        return nullptr;
    }
    bound->lease = std::move(*outcome.lease);
    try {
        bound->node = registry_.get_node(bound->lease->node_id());
    } catch (const std::exception& e) {
        error = std::string("node lookup failed: ") + e.what();
        code = "node_unavailable";
        return nullptr;
    }
    bound->slot_id = bound->lease->slot_id();
    bound->engine_id = bound->lease->engine_id();
    bound->request_model = bound->lease->request_model();
    bound->client = std::make_unique<NodeProxyRuntimeClient>(bound->node.url, bound->node.api_key,
                                                             bound->slot_id);
    bound->caps = capabilities_for(model, bound->engine_id, vllm_tool_choice());
    return bound;
}

// ── One turn ──────────────────────────────────────────────────────────────────

void ControlApiServer::run_chat_turn(const ChatTurnRequest& turn,
                                     ChunkCb chunk_cb,
                                     TurnDoneCb done_cb) {
    const int64_t request_started_ms = util::now_ms();
    TurnOutcome outcome;
    const auto fail = [&](const std::string& error, const std::string& code) {
        outcome.ok = false;
        outcome.error = error;
        outcome.code = code;
        activity_log(2, "Chat failed: user='" + turn.user_id + "' model='" + outcome.model_id +
                            "' reason='" + error + "'");
        done_cb(turn.conv_id, outcome);
    };

    auto user = users_.get_user(turn.user_id);
    if (!user) return fail("user not found", "user_not_found");
    UserDB& db = user->db();
    const UserProfile profile = user->profile();
    auto conv = db.get_conversation(turn.conv_id);
    if (!conv) return fail("conversation not found", "conversation_not_found");
    if (served_models_ == nullptr)
        return fail("the served-model catalog is not available", "catalog_unavailable");

    // 1. The model.
    const auto choice = resolve_turn_model(
        turn.model, conv->model_id, profile.default_model,
        [this](const std::string& id) { return served_models_->find(id); });
    outcome.model_id = choice.model_id;
    if (choice.problem != TurnModel::Problem::None) return fail(choice.error, to_string(choice.problem));
    if (choice.sticky) {
        ConversationPatch patch;
        patch.model_id = choice.model_id;
        db.update_conversation(turn.conv_id, patch);
        conv->model_id = choice.model_id;
    }
    const ServedModel& model = choice.entry->model;
    RuntimeSettings settings =
        apply_overrides(model.runtime_settings, merge_overrides(conv->settings, turn.settings));
    if (turn.max_tokens_override > 0) settings.max_tokens = turn.max_tokens_override;

    PerformanceSample perf;
    perf.request_id = util::generate_uuid();
    perf.served_model_id = model.id;
    perf.user_id = turn.user_id;
    perf.model = model.model_path;
    perf.started_at_ms = request_started_ms;
    if (!model.vision_settings.mmproj_path.empty())
        perf.projector_basename = std::filesystem::path(model.vision_settings.mmproj_path).filename().string();

    // 2. Bind: the lease is held for the whole turn — compaction, memory
    //    selection and every tool round — so the placement cannot be evicted
    //    while this turn is using it.
    std::string bind_error, bind_code;
    auto bound = bind_model(model, chat_queue_wait(), bind_error, bind_code);
    if (!bound) return fail(bind_error, bind_code);
    perf.backend = bound->api ? std::string("api") : bound->engine_id;
    activity_log(0, "Chat routed: user='" + turn.user_id + "' model='" + model.id + "' -> " +
                        (bound->api ? "api=" + model.api_settings.base_url
                                    : "node=" + bound->node.id.substr(0, 8) + " slot=" +
                                          bound->slot_id.substr(0, 8)));

    // 3. What this engine cannot take is refused before anything is stored.
    if (has_image_parts(turn.parts) && !bound->caps.vision) {
        return fail(image_refusal_for(model.vision_settings.enabled, bound->engine_id,
                                      bound->api ? std::string{} : "placed on " + bound->engine_id),
                    "images_unsupported");
    }

    // 4. The user's message.
    Message user_msg;
    user_msg.role = MessageRole::User;
    user_msg.content = turn.message;
    user_msg.content_parts = turn.parts;
    if (user_msg.content_parts.empty() && !turn.message.empty()) {
        MessageContentPart text_part;
        text_part.type = "text";
        text_part.text = turn.message;
        user_msg.content_parts.push_back(std::move(text_part));
    }
    user_msg.token_count = estimate_tokens(turn.message);
    user_msg.timestamp_ms = util::now_ms();
    if (db.append_message(turn.conv_id, user_msg) < 0)
        return fail("conversation not found", "conversation_not_found");

    const bool memories_on = profile.memories_enabled;
    const bool tools_offered = profile.tools_enabled && memories_on && bound->caps.tools;
    RuntimeClient& runtime = *bound->client;

    std::vector<Memory> memories;
    if (memories_on) {
        MemoryManager mem_mgr(db, runtime, bound->request_model, settings);
        memories = mem_mgr.get_relevant_memories(turn.conv_id, join_instructions(profile, *conv));
    }

    // 5. Keep the prompt inside this model's window.
    const AttachmentNamer name_of = [&db](const std::string& id) {
        auto a = db.get_attachment(id);
        return a ? a->original_filename : std::string{};
    };
    {
        Conversation without_summary = *conv;
        without_summary.summary.clear();
        const auto locals = memories_on ? db.list_local_memories(turn.conv_id)
                                        : std::vector<LocalMemory>{};
        CompactionRequest creq;
        creq.summary = conv->summary;
        creq.summary_through_seq = conv->summary_through_seq;
        creq.messages = db.load_messages(turn.conv_id, conv->summary_through_seq);
        creq.budget = prompt_budget(settings);
        creq.base_system_tokens = estimate_tokens(
            build_system_text(profile, without_summary, memories, locals, tools_offered));
        creq.vision = bound->caps.vision;
        const RollingCompactor compactor(
            [&](const std::vector<Message>& prompt, int max_tokens) {
                return complete_text(runtime, bound->request_model, settings, prompt, max_tokens, 0.2f);
            },
            [&](const std::string& summary, int through) {
                return db.save_summary(turn.conv_id, summary, through, model.id);
            },
            [&db](const std::string& id) -> std::int64_t {
                auto a = db.get_attachment(id);
                return a ? a->size_bytes : 0;
            });
        const auto folded = compactor.run(creq);
        if (folded.status == CompactionResult::Status::Overflow)
            return fail(folded.error, "context_overflow");
        if (folded.status == CompactionResult::Status::Failed) {
            if (folded.prompt_tokens > creq.budget)
                return fail("compaction failed: " + folded.error, "compaction_failed");
            MM_WARN("compaction failed for conversation {} but the prompt still fits: {}",
                    turn.conv_id, folded.error);
        }
        if (folded.status == CompactionResult::Status::Compacted) {
            activity_log(0, "Conversation compacted: user='" + turn.user_id + "' conv='" +
                                turn.conv_id.substr(0, 8) + "' through=" +
                                std::to_string(folded.summary_through_seq) +
                                " calls=" + std::to_string(folded.summarizer_calls));
        }
    }

    // 6. Infer, running the memory tools the model calls.
    ToolExecutor tool_exec(db);
    static constexpr int kMaxToolRounds = 10;
    bool ok = true;
    std::string failure_reason;
    std::string final_assistant_content;
    int final_total_tokens = 0;
    const int64_t infer_start_ms = util::now_ms();
    bool force_prefill_safe = false;
    perf.queue_ms = infer_start_ms - request_started_ms;
    int64_t first_token_at_ms = 0;

    for (int tool_round = 0; tool_round < kMaxToolRounds; ++tool_round) {
        // Rebuilt every round: the tools may have changed the notes.
        conv = db.get_conversation(turn.conv_id);
        if (!conv) return fail("conversation was deleted mid-turn", "conversation_not_found");
        const auto locals = memories_on ? db.list_local_memories(turn.conv_id)
                                        : std::vector<LocalMemory>{};
        std::vector<TraceEvent> context_trace_events =
            build_context_trace_events(db, turn.conv_id, memories);
        std::vector<Message> context_msgs = build_turn_context(
            build_system_text(profile, *conv, memories, locals, tools_offered),
            db.load_messages(turn.conv_id, conv->summary_through_seq), bound->caps, name_of);

        try {
            int context_images = 0;
            int64_t context_image_bytes = 0;
            hydrate_attachment_parts(db, context_msgs, &context_images, &context_image_bytes);
            if (tool_round == 0) {
                perf.image_count = context_images;
                perf.decoded_image_bytes = context_image_bytes;
                perf.vision_routing = context_images > 0;
            }
        } catch (const std::exception& e) {
            ok = false;
            failure_reason = e.what();
            break;
        }
        if (tool_round == 0) {
            for (const auto& context_message : context_msgs)
                perf.input_tokens += estimate_tokens(context_message, bound->caps.vision);
        }

        InferenceRequest infer_req;
        infer_req.model = bound->request_model;
        infer_req.messages = context_msgs;
        infer_req.settings = settings;
        infer_req.stream = true;
        if (tools_offered) infer_req.tools = ToolExecutor::local_tool_catalog();

        std::string assistant_content;
        std::string thinking_content;
        int total_tokens = 0;
        std::string finish_reason;
        bool infer_done_seen = false;
        bool stream_had_error = false;
        std::string stream_error_message;
        int infer_http_status = 0;
        std::string infer_http_body;
        std::vector<ToolCall> accumulated_tool_calls;

        const auto reset_infer_state = [&]() {
            assistant_content.clear();
            thinking_content.clear();
            total_tokens = 0;
            finish_reason.clear();
            infer_done_seen = false;
            stream_had_error = false;
            stream_error_message.clear();
            infer_http_status = 0;
            infer_http_body.clear();
            accumulated_tool_calls.clear();
        };
        const auto note_first_token = [&](const InferenceChunk& chunk) {
            if (first_token_at_ms == 0 &&
                (!chunk.delta_content.empty() || !chunk.thinking_delta.empty() || chunk.tool_call_delta))
                first_token_at_ms = util::now_ms();
        };

        const auto node_stream_cb = [&](const std::string& data) -> bool {
            if (data == "[DONE]") return true;
            try {
                auto j = nlohmann::json::parse(data);
                const std::string type = j.value("type", "");
                InferenceChunk chunk;
                if (type == "thinking") {
                    chunk.thinking_delta = j.value("content", "");
                    thinking_content += chunk.thinking_delta;
                } else if (type == "delta") {
                    chunk.delta_content = j.value("content", "");
                    assistant_content += chunk.delta_content;
                } else if (type == "tool_call") {
                    ToolCall tc;
                    tc.id = j.value("id", util::generate_uuid());
                    tc.function_name = j.value("name", "");
                    tc.arguments_json = j.value("arguments", "");
                    accumulated_tool_calls.push_back(tc);
                    chunk.tool_call_delta = tc;
                } else if (type == "error") {
                    stream_had_error = true;
                    stream_error_message = j.value("message", std::string{});
                } else if (type == "done") {
                    infer_done_seen = true;
                    total_tokens = j.value("tokens_used", 0);
                    finish_reason = j.value("finish_reason", "");
                    return true;
                }
                note_first_token(chunk);
                chunk_cb(chunk);
            } catch (const std::exception& e) {
                MM_WARN("chat turn: SSE parse error: {}", e.what());
            }
            return true;
        };

        const auto run_infer_once = [&](bool stream_mode) -> bool {
            InferenceRequest req_to_send = infer_req;
            req_to_send.stream = stream_mode;
            if (force_prefill_safe) enforce_prefill_safe_tail(req_to_send.messages);
            if (bound->api) {
                if (stream_mode) {
                    bool api_ok = true;
                    runtime.stream_complete(
                        req_to_send,
                        [&](const InferenceChunk& chunk) {
                            if (!chunk.is_done) note_first_token(chunk);
                            if (chunk.is_done) {
                                infer_done_seen = true;
                                total_tokens = chunk.tokens_used;
                                finish_reason = chunk.finish_reason;
                                return;
                            }
                            thinking_content += chunk.thinking_delta;
                            assistant_content += chunk.delta_content;
                            if (chunk.tool_call_delta) accumulated_tool_calls.push_back(*chunk.tool_call_delta);
                            chunk_cb(chunk);
                        },
                        [&](const std::string& error) {
                            api_ok = false;
                            stream_had_error = true;
                            stream_error_message = error;
                            infer_http_body = error;
                        });
                    return api_ok;
                }
                const Message response = runtime.complete(req_to_send);
                if (response.role != MessageRole::Assistant) {
                    stream_had_error = true;
                    stream_error_message = "API chat completion failed";
                    infer_http_body = stream_error_message;
                    return false;
                }
                thinking_content = response.thinking_text;
                assistant_content = response.content;
                total_tokens = response.token_count;
                accumulated_tool_calls = response.tool_calls;
                infer_done_seen = true;
                InferenceChunk replay;
                replay.thinking_delta = thinking_content;
                replay.delta_content = assistant_content;
                note_first_token(replay);
                if (!thinking_content.empty()) {
                    InferenceChunk c;
                    c.thinking_delta = thinking_content;
                    chunk_cb(c);
                }
                if (!assistant_content.empty()) {
                    InferenceChunk c;
                    c.delta_content = assistant_content;
                    chunk_cb(c);
                }
                for (const auto& tc : accumulated_tool_calls) {
                    InferenceChunk c;
                    c.tool_call_delta = tc;
                    chunk_cb(c);
                }
                return true;
            }
            HttpClient node_cli(bound->node.url);
            node_cli.set_bearer_token(bound->node.api_key);
            nlohmann::json infer_body = nlohmann::json(req_to_send);
            infer_body["slot_id"] = bound->slot_id;
            return node_cli.stream_post("/api/node/infer", infer_body, node_stream_cb,
                                        &infer_http_status, &infer_http_body);
        };

        const auto finalize_attempt = [&](bool transport_ok) -> bool {
            const std::string infer_label = bound->api ? "api infer" : "node infer";
            bool ok_now = transport_ok;
            if (!ok_now) {
                if (infer_done_seen && !stream_had_error) {
                    MM_WARN("chat turn: {} transport ended non-successfully after done event;"
                            " accepting completion (user='{}', conv='{}')",
                            infer_label, turn.user_id, turn.conv_id);
                    ok_now = true;
                } else if (infer_http_status > 0) {
                    std::string body = util::trim(infer_http_body);
                    if (body.size() > 400) body = body.substr(0, 400) + "...";
                    failure_reason = infer_label + " failed (HTTP " + std::to_string(infer_http_status) + ")";
                    if (!body.empty()) failure_reason += ": " + body;
                } else {
                    failure_reason = infer_label + " stream failed (connection error or non-2xx)";
                }
            }
            if (stream_had_error) {
                ok_now = false;
                failure_reason = stream_error_message.empty() ? infer_label + " reported stream error"
                                                              : stream_error_message;
            }
            return ok_now;
        };

        static constexpr int kMaxStreamAttempts = 3;
        bool stream_ok = false;
        for (int attempt = 1; attempt <= kMaxStreamAttempts; ++attempt) {
            reset_infer_state();
            stream_ok = finalize_attempt(run_infer_once(true));
            if (stream_ok) break;
            if (contains_prefill_thinking_incompatibility(failure_reason) ||
                contains_prefill_thinking_incompatibility(infer_http_body) ||
                contains_prefill_thinking_incompatibility(stream_error_message)) {
                force_prefill_safe = true;
            }
            if (attempt < kMaxStreamAttempts)
                std::this_thread::sleep_for(std::chrono::milliseconds(250 * attempt));
        }
        if (!stream_ok) {
            reset_infer_state();
            if (finalize_attempt(run_infer_once(false))) {
                MM_WARN("chat turn: recovered with non-stream infer fallback (user='{}')", turn.user_id);
                stream_ok = true;
            }
        }
        if (!stream_ok) {
            ok = false;
            break;
        }

        final_total_tokens += total_tokens;

        Message asst_msg;
        asst_msg.role = MessageRole::Assistant;
        asst_msg.content = assistant_content;
        asst_msg.thinking_text = thinking_content;
        asst_msg.tool_calls = tools_offered ? accumulated_tool_calls : std::vector<ToolCall>{};
        asst_msg.token_count = total_tokens;
        asst_msg.timestamp_ms = util::now_ms();
        asst_msg.trace_events = std::move(context_trace_events);
        asst_msg.model_id = model.id;
        db.append_message(turn.conv_id, asst_msg);

        // Only tools that were OFFERED run: a model that emits a call to one it
        // was never given does not get to execute it.
        const bool wants_tools =
            tools_offered &&
            std::any_of(accumulated_tool_calls.begin(), accumulated_tool_calls.end(),
                        [&](const ToolCall& tc) { return tool_exec.is_memory_tool(tc.function_name); });
        if (!wants_tools) {
            final_assistant_content = assistant_content;
            break;
        }

        for (const auto& tc : accumulated_tool_calls) {
            Message tool_result;
            if (tool_exec.is_memory_tool(tc.function_name)) {
                tool_result = tool_exec.execute_tool(tc, turn.conv_id);
            } else {
                tool_result.role = MessageRole::Tool;
                tool_result.tool_call_id = tc.id;
                tool_result.content = nlohmann::json{{"error", "tool not available"}}.dump();
                tool_result.timestamp_ms = util::now_ms();
            }
            if (auto trace = build_tool_access_trace(tc, tool_result, turn.conv_id))
                tool_result.trace_events.push_back(*trace);
            db.append_message(turn.conv_id, tool_result);

            InferenceChunk tool_chunk;
            tool_chunk.tool_result_json = nlohmann::json{{"type", "tool_result"},
                                                         {"tool_call_id", tc.id},
                                                         {"function_name", tc.function_name},
                                                         {"result", tool_result.content}}
                                              .dump();
            chunk_cb(tool_chunk);
        }
    }

    const int64_t duration_ms = util::now_ms() - infer_start_ms;
    {
        char buf[512];
        if (ok) {
            std::snprintf(buf, sizeof(buf),
                          "Chat complete: user='%s' model='%s' tokens=%d content=%zuB duration=%lldms "
                          "images=%d vision=%s",
                          turn.user_id.c_str(), model.id.c_str(), final_total_tokens,
                          final_assistant_content.size(), static_cast<long long>(duration_ms),
                          perf.image_count, perf.vision_routing ? "true" : "false");
            activity_log(0, buf);
        } else {
            std::snprintf(buf, sizeof(buf),
                          "Chat error: user='%s' model='%s' backend=%s reason='%s' duration=%lldms",
                          turn.user_id.c_str(), model.id.c_str(), perf.backend.c_str(),
                          failure_reason.substr(0, 80).c_str(), static_cast<long long>(duration_ms));
            activity_log(2, buf);
        }
    }

    // Give the placement back before signalling completion, so a client that
    // immediately sends another turn does not find it still busy.
    const NodeId perf_node_id = bound->api ? NodeId{} : bound->node.id;
    bound.reset();

    outcome.ok = ok;
    outcome.error = failure_reason;
    if (!ok) outcome.code = "inference_failed";
    done_cb(turn.conv_id, outcome);

    const int64_t completed_at_ms = util::now_ms();
    perf.node_id = perf_node_id;
    perf.time_to_first_token_ms = first_token_at_ms > 0 ? first_token_at_ms - infer_start_ms : -1;
    perf.total_ms = completed_at_ms - infer_start_ms;
    perf.output_tokens = final_total_tokens;
    perf.success = ok;
    perf.error = failure_reason;
    performance_.record(std::move(perf));
}

ControlApiServer::LocalChatResult ControlApiServer::chat_local(
    const UserId& user_id,
    const std::string& message,
    const ConvId& conv_id,
    const std::string& model,
    int max_tokens_override,
    const std::vector<std::string>& attachment_ids) {
    LocalChatResult result;
    auto user = users_.get_user(user_id);
    if (!user) {
        result.error = "user not found";
        result.code = "user_not_found";
        return result;
    }
    ChatTurnRequest turn;
    turn.user_id = user_id;
    turn.message = message;
    turn.model = model;
    turn.max_tokens_override = max_tokens_override;
    if (conv_id.empty()) {
        turn.conv_id = user->db().get_or_create_active_conversation();
    } else if (!user->db().conversation_exists(conv_id)) {
        result.error = "conversation not found";
        result.code = "conversation_not_found";
        return result;
    } else {
        turn.conv_id = conv_id;
    }
    try {
        turn.parts = prepare_attachment_turn(user->db(), message, attachment_ids).parts;
    } catch (const std::exception& e) {
        result.error = e.what();
        result.code = "invalid_request";
        return result;
    }

    // Through the queue, like every turn: a local turn must not overlap an HTTP
    // one on the same conversation.
    const bool ran = run_on_conversation(user_id, turn.conv_id, [&] {
        run_chat_turn(
            turn, [&result](const InferenceChunk& chunk) { result.chunks.push_back(chunk); },
            [&result](const ConvId& id, const TurnOutcome& outcome) {
                result.conv_id = id;
                result.success = outcome.ok;
                result.error = outcome.error;
                result.code = outcome.code;
                result.model_id = outcome.model_id;
            });
    });
    if (!ran) {
        result.error = "the server is shutting down";
        result.code = "shutting_down";
    }
    return result;
}

// ── Global recall ─────────────────────────────────────────────────────────────

void ControlApiServer::queue_global_recall(const UserId& user_id, const ConvId& conv_id) {
    auto user = users_.get_user(user_id);
    if (!user || !user->profile().memories_enabled) return;
    activity_log(0, "Global recall queued: user='" + user_id + "' conv='" + conv_id.substr(0, 8) + "'");
    conversations_->enqueue(queue_key(user_id, conv_id),
                            [this, user_id, conv_id] { run_global_recall(user_id, conv_id); });
}

void ControlApiServer::run_global_recall(const UserId& user_id, const ConvId& conv_id) {
    auto user = users_.get_user(user_id);
    if (!user) return;
    const UserProfile profile = user->profile();
    if (!profile.memories_enabled) return;
    UserDB& db = user->db();
    const auto conv = db.get_conversation(conv_id);
    if (!conv) return;
    const auto messages = db.load_messages(conv_id, conv->summary_through_seq);
    if (messages.empty() && conv->summary.empty()) return;

    // The model that last answered this conversation, else its own, else the
    // user's default. None: nothing to recall with, which is not an error.
    std::string model_id;
    for (auto it = messages.rbegin(); it != messages.rend() && model_id.empty(); ++it)
        if (it->role == MessageRole::Assistant) model_id = it->model_id;
    if (model_id.empty()) model_id = conv->model_id;
    if (model_id.empty()) model_id = profile.default_model;
    if (model_id.empty()) {
        activity_log(1, "Global recall skipped: user='" + user_id + "' has no model to recall with");
        return;
    }
    std::string error, code;
    const auto model = available_model(model_id, error, code);
    if (!model) {
        activity_log(1, "Global recall skipped: user='" + user_id + "' " + error);
        return;
    }
    auto bound = bind_model(*model, chat_queue_wait(), error, code);
    if (!bound) {
        activity_log(1, "Global recall failed: user='" + user_id + "' " + error);
        return;
    }
    const RuntimeSettings settings = apply_overrides(model->runtime_settings, conv->settings);
    RuntimeClient& runtime = *bound->client;
    ToolExecutor tool_exec(db);
    const bool with_tools = bound->caps.tools;

    std::string system_text =
        "You are reviewing a completed conversation to maintain global summaries for this "
        "user's future conversations. Save only durable context that is genuinely useful "
        "and not already captured by an existing global summary. Each summary should explain "
        "what the source conversation contains and why it may matter later.";
    if (with_tools) {
        system_text += " Use the tools to save, update or delete global memories.";
    } else {
        system_text +=
            "\n\nAnswer ONLY with a JSON object of this shape, and nothing else:\n"
            "{\"memories\": [{\"content\": \"...\", \"importance\": 0.5}]}\n"
            "Use an empty array when there is nothing worth saving.";
    }
    const auto globals = db.list_memories();
    if (!globals.empty()) {
        system_text += "\n\n## Existing global summaries\n";
        for (const auto& gm : globals) {
            system_text += "- ";
            if (with_tools) system_text += "[" + gm.id + "] ";
            system_text += gm.content + "\n";
        }
    }
    const auto locals = db.list_local_memories(conv_id);
    if (!locals.empty()) {
        system_text += "\n\n## Notes taken in this conversation\n";
        for (const auto& lm : locals) system_text += "- " + lm.content + "\n";
    }
    if (!util::trim(conv->summary).empty())
        system_text += "\n\n## Earlier in the conversation (summary)\n" + conv->summary;

    // Bounded: the newest messages that fit this model, not the whole history.
    const int reply_cap = std::min(std::max(settings.max_tokens, 256), 1024);
    RuntimeSettings recall_settings = settings;
    recall_settings.max_tokens = reply_cap;
    recall_settings.temperature = 0.2f;
    const int room = prompt_budget(recall_settings) - estimate_tokens(system_text) - 128;
    const auto tail = newest_that_fit(messages, std::max(room, 0), false);
    std::vector<Message> recall_ctx = build_turn_context(
        system_text, tail, ModelCapabilities{false, with_tools},
        [&db](const std::string& id) {
            auto a = db.get_attachment(id);
            return a ? a->original_filename : std::string{};
        });
    Message ask;
    ask.role = MessageRole::User;
    ask.content = "Review this conversation and save what is worth remembering.";
    recall_ctx.push_back(ask);

    if (!with_tools) {
        const auto answer = complete_text(runtime, bound->request_model, recall_settings,
                                          recall_ctx, reply_cap, 0.2f);
        bound.reset();
        if (!answer) {
            activity_log(1, "Global recall failed: user='" + user_id + "' model did not answer");
            return;
        }
        int saved = 0;
        try {
            const auto parsed = nlohmann::json::parse(extract_json_object_text(*answer));
            for (const auto& item : parsed.value("memories", nlohmann::json::array())) {
                if (!item.is_object() || !item.contains("content") || !item["content"].is_string())
                    continue;
                const std::string content = util::trim(item["content"].get<std::string>());
                if (content.empty()) continue;
                Memory mem;
                mem.id = util::generate_uuid();
                mem.content = content;
                mem.source_conv_id = conv_id;
                mem.importance = std::clamp(
                    item.contains("importance") && item["importance"].is_number()
                        ? item["importance"].get<float>() : 0.5f,
                    0.0f, 1.0f);
                db.add_memory(mem);
                ++saved;
            }
        } catch (const std::exception& e) {
            activity_log(1, "Global recall: user='" + user_id + "' unreadable answer: " + e.what());
            return;
        }
        activity_log(0, "Global recall complete: user='" + user_id + "' saved=" + std::to_string(saved));
        return;
    }

    static constexpr int kMaxRecallRounds = 5;
    for (int round = 0; round < kMaxRecallRounds; ++round) {
        InferenceRequest req;
        req.model = bound->request_model;
        req.messages = recall_ctx;
        req.settings = recall_settings;
        req.tools = tool_exec.get_all_tool_definitions();
        req.stream = false;
        const Message response = runtime.complete(req);
        if (response.role != MessageRole::Assistant) break;
        recall_ctx.push_back(response);
        const bool calls_memory_tools =
            std::any_of(response.tool_calls.begin(), response.tool_calls.end(),
                        [&](const ToolCall& tc) { return tool_exec.is_memory_tool(tc.function_name); });
        if (!calls_memory_tools) break;
        for (const auto& tc : response.tool_calls) {
            Message tool_result;
            if (tool_exec.is_memory_tool(tc.function_name)) {
                tool_result = tool_exec.execute_tool(tc, conv_id);
            } else {
                tool_result.role = MessageRole::Tool;
                tool_result.tool_call_id = tc.id;
                tool_result.content = nlohmann::json{{"error", "tool not available"}}.dump();
                tool_result.timestamp_ms = util::now_ms();
            }
            recall_ctx.push_back(tool_result);
        }
    }
    bound.reset();
    activity_log(0, "Global recall complete: user='" + user_id + "'");
}

// ── Extraction ────────────────────────────────────────────────────────────────

void ControlApiServer::queue_extraction(const UserId& user_id,
                                        const ConvId& conv_id,
                                        int begin_seq,
                                        int end_seq,
                                        const ServedModel& model) {
    conversations_->enqueue(queue_key(user_id, conv_id), [this, user_id, conv_id, begin_seq, end_seq, model] {
        auto user = users_.get_user(user_id);
        if (!user) return;
        auto messages = user->db().load_messages(conv_id, begin_seq - 1);
        messages.erase(std::remove_if(messages.begin(), messages.end(),
                                      [&](const Message& m) { return m.sequence_num > end_seq; }),
                       messages.end());
        std::string error, code;
        auto bound = bind_model(model, chat_queue_wait(), error, code);
        if (!bound) {
            activity_log(1, "Memory extraction failed: user='" + user_id + "' " + error);
            return;
        }
        const auto conv = user->db().get_conversation(conv_id);
        const RuntimeSettings settings =
            apply_overrides(model.runtime_settings, conv ? conv->settings : SamplingOverrides{});
        MemoryManager extractor(user->db(), *bound->client, bound->request_model, settings);
        const int stored = extractor.extract_and_store_memories_from_messages(conv_id, messages);
        activity_log(stored < 0 ? 1 : 0, "Memory extraction: user='" + user_id + "' model='" + model.id +
                                             "' stored=" + std::to_string(stored));
    });
}

// ── Manual compaction ─────────────────────────────────────────────────────────

ControlApiServer::CompactOutcome ControlApiServer::compact_now(const UserId& user_id,
                                                               const ConvId& conv_id,
                                                               const std::string& requested_model) {
    CompactOutcome out;
    const bool ran = run_on_conversation(user_id, conv_id, [&] {
        auto user = users_.get_user(user_id);
        if (!user) {
            out = {404, "failed", "user not found", "user_not_found", {}};
            return;
        }
        UserDB& db = user->db();
        auto conv = db.get_conversation(conv_id);
        if (!conv) {
            out = {404, "failed", "conversation not found", "conversation_not_found", {}};
            return;
        }
        const auto choice = resolve_turn_model(
            requested_model, conv->model_id, user->profile().default_model,
            [this](const std::string& id) -> std::optional<CatalogEntry> {
                if (served_models_ == nullptr) return std::nullopt;
                return served_models_->find(id);
            });
        out.model_id = choice.model_id;
        if (choice.problem != TurnModel::Problem::None) {
            out.status = choice.problem == TurnModel::Problem::Required ? 400
                       : choice.problem == TurnModel::Problem::NotFound ? 404 : 409;
            out.state = "failed";
            out.error = choice.error;
            out.code = to_string(choice.problem);
            return;
        }
        const ServedModel& model = choice.entry->model;
        std::string error, code;
        auto bound = bind_model(model, chat_queue_wait(), error, code);
        if (!bound) {
            out = {503, "failed", error, code, model.id};
            return;
        }
        const RuntimeSettings settings = apply_overrides(model.runtime_settings, conv->settings);
        const UserProfile profile = user->profile();
        Conversation without_summary = *conv;
        without_summary.summary.clear();
        CompactionRequest creq;
        creq.summary = conv->summary;
        creq.summary_through_seq = conv->summary_through_seq;
        creq.messages = db.load_messages(conv_id, conv->summary_through_seq);
        creq.budget = prompt_budget(settings);
        creq.base_system_tokens = estimate_tokens(build_system_text(
            profile, without_summary, {},
            profile.memories_enabled ? db.list_local_memories(conv_id) : std::vector<LocalMemory>{},
            false));
        creq.vision = bound->caps.vision;
        creq.force = true;
        RuntimeClient& runtime = *bound->client;
        const RollingCompactor compactor(
            [&](const std::vector<Message>& prompt, int max_tokens) {
                return complete_text(runtime, bound->request_model, settings, prompt, max_tokens, 0.2f);
            },
            [&](const std::string& summary, int through) {
                return db.save_summary(conv_id, summary, through, model.id);
            });
        const auto result = compactor.run(creq);
        out.state = to_string(result.status);
        if (result.status == CompactionResult::Status::Failed) {
            out.status = 502;
            out.error = result.error;
            out.code = "compaction_failed";
        }
    });
    if (!ran) out = {503, "failed", "the server is shutting down", "shutting_down", {}};
    return out;
}

// ── Voice design ──────────────────────────────────────────────────────────────

std::optional<VoiceDesignProposal> ControlApiServer::infer_voice_design(const UserId& user_id,
                                                                        const std::string& requested,
                                                                        std::string& error) {
    auto user = users_.get_user(user_id);
    if (!user) {
        error = "user not found";
        return std::nullopt;
    }
    const UserProfile profile = user->profile();
    const std::string model_id =
        util::to_lower(util::trim(requested.empty() ? profile.default_model : requested));
    if (model_id.empty()) {
        error = "no model: name one in the request or give the user a default_model";
        return std::nullopt;
    }
    std::string code;
    const auto model = available_model(model_id, error, code);
    if (!model) return std::nullopt;
    auto bound = bind_model(*model, chat_queue_wait(), error, code);
    if (!bound) return std::nullopt;
    try {
        std::vector<Message> messages(2);
        messages[0].role = MessageRole::System;
        messages[0].content =
            "You are designing a synthetic TTS voice for an AI assistant that speaks with this "
            "user. Return only a strict JSON object with these string fields: display_name, "
            "language, voice_description, sample_text, rationale. Do not imitate, reference, or "
            "request the voice of any real person, celebrity, private person, or copyrighted "
            "character. Design an original voice that suits the user's instructions. Keep "
            "sample_text under 280 characters.";
        messages[1].role = MessageRole::User;
        messages[1].content = "User: " + profile.display_name + "\n\nTheir instructions for the "
                              "assistant:\n" +
                              (util::trim(profile.custom_instructions).empty()
                                   ? std::string("(none)")
                                   : profile.custom_instructions) +
                              "\n\nDesign the voice now.";
        const RuntimeSettings& s = model->runtime_settings;
        const auto answer = complete_text(*bound->client, bound->request_model, s, messages,
                                          std::min(std::max(s.max_tokens, 256), 768), s.temperature);
        bound.reset();
        if (!answer) {
            error = "the model did not answer";
            return std::nullopt;
        }
        const auto parsed = nlohmann::json::parse(extract_json_object_text(*answer));
        VoiceDesignProposal proposal;
        proposal.display_name = util::trim(parsed.value("display_name", std::string{}));
        proposal.language = util::trim(parsed.value("language", std::string{"Auto"}));
        proposal.voice_description = util::trim(parsed.value("voice_description", std::string{}));
        proposal.sample_text = util::trim(parsed.value("sample_text", std::string{}));
        proposal.rationale = util::trim(parsed.value("rationale", std::string{}));
        return proposal;
    } catch (const std::exception& e) {
        error = e.what();
        return std::nullopt;
    }
}

} // namespace mm
