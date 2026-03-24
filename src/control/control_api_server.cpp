#include "control/control_api_server.hpp"
#include "control/agent_manager.hpp"
#include "control/agent_queue.hpp"
#include "control/node_registry.hpp"
#include "control/model_router.hpp"
#include "control/agent_scheduler.hpp"
#include "control/agent_config_validator.hpp"
#include "common/agent.hpp"
#include "common/agent_db.hpp"
#include "common/http_server.hpp"
#include "common/http_client.hpp"
#include "common/memory_manager.hpp"
#include "common/conversation_manager.hpp"
#include "common/tool_executor.hpp"
#include "common/llama_cpp_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "common/sse_infer_ctx.hpp"
#include "common/gguf_metadata.hpp"
#include "common/model_catalog.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <string>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <unordered_map>

namespace mm {

// ── Constructor / destructor ───────────────────────────────────────────────────

ControlApiServer::ControlApiServer(AgentManager& agents,
                                   AgentQueue& queue,
                                   NodeRegistry& registry,
                                   ModelRouter& router,
                                   AgentScheduler& scheduler,
                                   std::string models_dir)
    : agents_(agents)
    , queue_(queue)
    , registry_(registry)
    , router_(router)
    , scheduler_(scheduler)
    , models_dir_(std::move(models_dir))
    , server_(std::make_unique<HttpServer>())
{}

ControlApiServer::~ControlApiServer() { stop(); }

bool ControlApiServer::listen(uint16_t port) {
    register_routes();
    MM_INFO("ControlApiServer listening on port {}", port);
    return server_->listen("0.0.0.0", port);
}
void ControlApiServer::stop() { server_->stop(); }

void ControlApiServer::set_log_callback(LogCallback cb) { log_cb_ = std::move(cb); }

void ControlApiServer::activity_log(int level, const std::string& message) {
    if (log_cb_) log_cb_(level, message);
}

ControlApiServer::LocalChatResult ControlApiServer::chat_local(
    const AgentId& agent_id,
    const std::string& message,
    const ConvId& conv_id_hint,
    int max_tokens_override) {
    LocalChatResult result;

    auto chunk_cb = [&result](const InferenceChunk& chunk) {
        result.chunks.push_back(chunk);
    };
    auto done_cb = [&result](const ConvId& conv_id, bool success, const std::string& error) {
        result.conv_id = conv_id;
        result.success = success;
        result.error = error;
    };

    handle_chat(agent_id, message, conv_id_hint,
                std::move(chunk_cb), std::move(done_cb), max_tokens_override);
    return result;
}

// ── SSE context shared between worker and provider ────────────────────────────

namespace {

// Routes LLM calls through the Node API (/api/node/infer) instead of talking
// directly to llama-server — necessary because llama-server is bound to
// localhost on the node machine and not reachable from the control process.
class NodeProxyLlamaClient : public LlamaCppClient {
public:
    NodeProxyLlamaClient(std::string node_url, std::string api_key, std::string slot_id = {})
        : LlamaCppClient(node_url)
        , node_url_(std::move(node_url))
        , api_key_(std::move(api_key))
        , slot_id_(std::move(slot_id))
    {}

    Message complete(const InferenceRequest& req) override {
        HttpClient cli(node_url_);
        cli.set_bearer_token(api_key_);

        Message result;
        result.role         = MessageRole::Assistant;
        result.timestamp_ms = util::now_ms();

        InferenceRequest req_copy = req;
        req_copy.stream = req.stream;

        nlohmann::json body = nlohmann::json(req_copy);
        if (!slot_id_.empty()) body["slot_id"] = slot_id_;

        bool ok = cli.stream_post("/api/node/infer",
            body,
            [&result](const std::string& data) -> bool {
                if (data == "[DONE]") return true;
                try {
                    auto j = nlohmann::json::parse(data);
                    std::string type = j.value("type", "");
                    if (type == "delta") {
                        result.content += j.value("content", "");
                    } else if (type == "thinking") {
                        result.thinking_text += j.value("content", "");
                    } else if (type == "tool_call") {
                        ToolCall tc;
                        tc.function_name  = j.value("name", "");
                        tc.arguments_json = j.value("arguments", "");
                        result.tool_calls.push_back(tc);
                    } else if (type == "done") {
                        result.token_count = j.value("tokens_used", 0);
                    }
                } catch (const std::exception& e) {
                    MM_WARN("NodeProxyLlamaClient::complete: parse error: {}", e.what());
                }
                return true;
            });

        if (!ok) {
            MM_WARN("NodeProxyLlamaClient::complete: stream failed to {}", node_url_);
        }
        return result;
    }

private:
    std::string node_url_;
    std::string api_key_;
    std::string slot_id_;
};

bool contains_prefill_thinking_incompatibility(const std::string& text) {
    const std::string lowered = util::to_lower(text);
    return lowered.find("assistant response prefill is incompatible with enable_thinking")
        != std::string::npos;
}

void enforce_prefill_safe_tail(std::vector<Message>& messages) {
    if (messages.empty()) return;
    if (messages.back().role != MessageRole::Assistant) return;

    Message m;
    m.role = MessageRole::User;
    m.content = "Continue with the next response based on the latest context.";
    m.timestamp_ms = util::now_ms();
    messages.push_back(std::move(m));
}

} // namespace

// ── handle_chat ───────────────────────────────────────────────────────────────
// Runs on the AgentQueue worker thread for the given agent.

void ControlApiServer::handle_chat(const AgentId& agent_id,
                                    const std::string& message,
                                    const ConvId& conv_id_hint,
                                    ChunkCb chunk_cb,
                                    DoneCb done_cb,
                                    int max_tokens_override) {
    // 1. Look up agent
    Agent* agent = agents_.get_agent(agent_id);
    if (!agent) {
        MM_WARN("handle_chat: agent '{}' not found", agent_id);
        done_cb({}, false, "agent not found");
        return;
    }

    AgentConfig cfg = agent->get_config();
    AgentDB& db = agent->db();

    activity_log(0, "Chat started: agent='" + agent_id + "' model='" + cfg.model_path + "'");

    // 2. Get or create active conversation
    ConvId conv_id = conv_id_hint;
    if (conv_id.empty()) {
        auto active = db.get_active_conversation_id();
        if (active) {
            conv_id = *active;
        } else {
            conv_id = db.create_conversation();
            db.set_active_conversation(conv_id);
        }
    }

    // 3. Append user message
    auto prior_msgs = db.load_messages(conv_id);
    int seq = static_cast<int>(prior_msgs.size());

    Message user_msg;
    user_msg.role         = MessageRole::User;
    user_msg.content      = message;
    user_msg.timestamp_ms = util::now_ms();
    db.append_message(conv_id, user_msg, seq);

    // 4. Route to a node via AgentScheduler. If capacity is temporarily unavailable,
    // wait and retry so queued prompts can run when the next node/slot opens up.
    int64_t wait_budget_ms = 180000; // default: 3 minutes
    if (const char* env_wait = std::getenv("MM_CHAT_QUEUE_WAIT_MS")) {
        try {
            int64_t parsed = std::stoll(env_wait);
            if (parsed > 0) wait_budget_ms = parsed;
        } catch (...) {}
    }
    constexpr int64_t kRetrySleepMs = 1000;

    std::optional<ScheduleResult> schedule_result;
    std::string sched_err;
    const int64_t wait_started_ms = util::now_ms();
    int wait_attempt = 0;
    while (true) {
        schedule_result = scheduler_.ensure_agent_running(cfg);
        if (schedule_result) break;

        sched_err = scheduler_.last_error();
        std::string err_l = util::to_lower(sched_err);
        bool queueable =
            sched_err.empty() ||
            err_l.find("no capacity") != std::string::npos ||
            err_l.find("no connected node") != std::string::npos ||
            err_l.find("no node available") != std::string::npos;

        if (!queueable) break;

        int64_t waited_ms = util::now_ms() - wait_started_ms;
        if (waited_ms >= wait_budget_ms) {
            if (!sched_err.empty()) {
                sched_err += " (timed out waiting for available node)";
            } else {
                sched_err = "timed out waiting for available node";
            }
            break;
        }

        if (wait_attempt == 0 || (wait_attempt % 15) == 0) {
            MM_INFO("handle_chat: waiting for available node for agent '{}' (waited {}s / {}s)",
                    agent_id,
                    static_cast<int>(waited_ms / 1000),
                    static_cast<int>(wait_budget_ms / 1000));
            if (wait_attempt == 0)
                activity_log(1, "Chat queued: agent='" + agent_id + "' waiting for node");
        }
        ++wait_attempt;
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetrySleepMs));
    }

    if (!schedule_result) {
        MM_WARN("handle_chat: no node available for agent '{}': {}",
                agent_id, sched_err);
        if (sched_err.empty()) {
            sched_err = "no node available or model failed to load";
        }
        activity_log(2, "Chat failed: agent='" + agent_id + "' reason='" + sched_err + "'");
        done_cb(conv_id, false, sched_err);
        return;
    }

    scheduler_.mark_agent_active(agent_id);

    NodeInfo node;
    try { node = registry_.get_node(schedule_result->node_id); }
    catch (const std::exception& e) {
        scheduler_.mark_agent_idle(agent_id);
        MM_WARN("handle_chat: node '{}' disappeared after routing: {}",
                schedule_result->node_id, e.what());
        done_cb(conv_id, false, std::string("node lookup failed: ") + e.what());
        return;
    }

    SlotId active_slot_id = schedule_result->slot_id;

    activity_log(0, "Chat routed: agent='" + agent_id + "' -> node=" +
                 schedule_result->node_id.substr(0, 8) + " slot=" +
                 active_slot_id.substr(0, 8));

    // 5. Real LLM client routed through the node API.
    //    (llama-server is bound to localhost on the node machine; control
    //     cannot reach it directly — all LLM calls go via /api/node/infer.)
    NodeProxyLlamaClient real_llama(node.url, node.api_key, active_slot_id);

    // 6. Compact the conversation if it is approaching context limits.
    //    may_compact() creates a new conversation with a summary if >80% full.
    ConversationManager conv_mgr(db, real_llama);
    conv_id = conv_mgr.maybe_compact(conv_id, cfg);

    // 7. Retrieve memories and build the full context to send to the LLM.
    std::vector<Memory> memories;
    if (cfg.memories_enabled) {
        MemoryManager mem_mgr(db, real_llama);
        memories = mem_mgr.get_relevant_memories(conv_id, cfg);
    }

    // 8. Set up tool executor for memory tools
    ToolExecutor tool_exec(db);

    // 9. Tool call loop — infer, execute tools, re-infer until done or max rounds
    static constexpr int kMaxToolRounds = 10;
    bool ok = true;
    std::string failure_reason;
    std::string final_assistant_content;
    std::string final_thinking_content;
    int         final_total_tokens = 0;
    const int64_t infer_start_ms = util::now_ms();
    bool        force_prefill_safe = false;

    for (int tool_round = 0; tool_round < kMaxToolRounds; ++tool_round) {
        // Build context fresh each round (local memories may have changed)
        std::vector<Message> context_msgs = conv_mgr.build_context(conv_id, cfg, memories);

        // Build inference request
        InferenceRequest infer_req;
        infer_req.model    = cfg.model_path;
        infer_req.messages = context_msgs;
        infer_req.settings = cfg.llama_settings;
        infer_req.stream   = true;

        // Tools are only advertised when explicitly enabled.
        if (cfg.tools_enabled && cfg.memories_enabled) {
            infer_req.tools = ToolExecutor::local_tool_catalog();
        }

        // Per-request max_tokens override
        if (max_tokens_override != 0) {
            infer_req.settings.max_tokens = max_tokens_override;
        }

        // Stream from node; parse SSE events and forward via chunk_cb
        std::string assistant_content;
        std::string thinking_content;
        int         total_tokens  = 0;
        std::string finish_reason;
        bool        infer_done_seen = false;
        bool        stream_had_error = false;
        std::string stream_error_message;
        int         infer_http_status = 0;
        std::string infer_http_body;
        std::vector<ToolCall> accumulated_tool_calls;

        HttpClient node_cli(node.url);
        node_cli.set_bearer_token(node.api_key);

        auto reset_infer_state = [&]() {
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

        auto stream_cb = [&](const std::string& data) -> bool {
                if (data == "[DONE]") return true;
                try {
                    auto j = nlohmann::json::parse(data);
                    std::string type = j.value("type", "");

                    InferenceChunk chunk;
                    if (type == "thinking") {
                        chunk.thinking_delta = j.value("content", "");
                        thinking_content += chunk.thinking_delta;
                    } else if (type == "delta") {
                        chunk.delta_content = j.value("content", "");
                        assistant_content  += chunk.delta_content;
                    } else if (type == "tool_call") {
                        ToolCall tc;
                        tc.id             = j.value("id", util::generate_uuid());
                        tc.function_name  = j.value("name", "");
                        tc.arguments_json = j.value("arguments", "");
                        accumulated_tool_calls.push_back(tc);
                        chunk.tool_call_delta = tc;
                    } else if (type == "error") {
                        stream_had_error = true;
                        stream_error_message = j.value("message", std::string{});
                    } else if (type == "done") {
                        infer_done_seen = true;
                        total_tokens  = j.value("tokens_used", 0);
                        finish_reason = j.value("finish_reason", "");
                        return true;
                    }
                    chunk_cb(chunk);
                } catch (const std::exception& e) {
                    MM_WARN("handle_chat: SSE parse error: {}", e.what());
                }
                return true;
        };

        auto run_infer_once = [&](bool stream_mode) -> bool {
            InferenceRequest req_to_send = infer_req;
            req_to_send.stream = stream_mode;
            if (force_prefill_safe) {
                enforce_prefill_safe_tail(req_to_send.messages);
            }
            nlohmann::json infer_body = nlohmann::json(req_to_send);
            infer_body["slot_id"] = active_slot_id;
            return node_cli.stream_post(
                "/api/node/infer",
                infer_body,
                stream_cb,
                &infer_http_status,
                &infer_http_body
            );
        };

        auto finalize_attempt = [&](bool transport_ok) -> bool {
            bool ok_now = transport_ok;
            if (!ok_now) {
                if (infer_done_seen && !stream_had_error) {
                    MM_WARN("handle_chat: node infer transport ended non-successfully after done event;"
                            " accepting completion (agent='{}', conv='{}')",
                            agent_id, conv_id);
                    ok_now = true;
                } else if (infer_http_status > 0) {
                    std::string body = util::trim(infer_http_body);
                    if (body.size() > 400) body = body.substr(0, 400) + "...";
                    failure_reason = "node infer failed (HTTP " + std::to_string(infer_http_status) + ")";
                    if (!body.empty()) failure_reason += ": " + body;
                } else {
                    failure_reason = "node infer stream failed (connection error or non-2xx)";
                }
            }
            if (stream_had_error) {
                ok_now = false;
                MM_WARN("handle_chat: node infer stream error for agent '{}': {}",
                        agent_id, stream_error_message);
                failure_reason = stream_error_message.empty()
                    ? "node reported infer stream error" : stream_error_message;
            }
            return ok_now;
        };

        static constexpr int kMaxStreamAttempts = 3;
        bool stream_ok = false;
        for (int attempt = 1; attempt <= kMaxStreamAttempts; ++attempt) {
            reset_infer_state();
            const bool transport_ok = run_infer_once(true);
            stream_ok = finalize_attempt(transport_ok);
            if (stream_ok) break;

            if (contains_prefill_thinking_incompatibility(failure_reason)
                || contains_prefill_thinking_incompatibility(infer_http_body)
                || contains_prefill_thinking_incompatibility(stream_error_message)) {
                force_prefill_safe = true;
            }

            if (attempt < kMaxStreamAttempts) {
                const int backoff_ms = 250 * attempt;
                MM_WARN("handle_chat: retrying stream infer for agent '{}' in {} ms (attempt {}/{})",
                        agent_id, backoff_ms, attempt + 1, kMaxStreamAttempts);
                std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            }
        }

        if (!stream_ok) {
            reset_infer_state();
            const bool non_stream_transport_ok = run_infer_once(false);
            const bool non_stream_ok = finalize_attempt(non_stream_transport_ok);
            if (non_stream_ok) {
                MM_WARN("handle_chat: recovered with non-stream infer fallback (agent='{}')", agent_id);
                stream_ok = true;
            }
        }

        if (!stream_ok) {
            ok = false;
            break;
        }

        final_total_tokens += total_tokens;

        // Persist assistant message (with any tool calls)
        Message asst_msg;
        asst_msg.role          = MessageRole::Assistant;
        asst_msg.content       = assistant_content;
        asst_msg.thinking_text = thinking_content;
        asst_msg.tool_calls    = accumulated_tool_calls;
        asst_msg.token_count   = total_tokens;
        asst_msg.timestamp_ms  = util::now_ms();
        db.append_message(conv_id, asst_msg, seq + 1);
        ++seq;

        // Check for memory tool calls that need execution
        bool has_memory_tools = false;
        for (const auto& tc : accumulated_tool_calls) {
            if (tool_exec.is_memory_tool(tc.function_name)) {
                has_memory_tools = true;
                break;
            }
        }

        if (!has_memory_tools || accumulated_tool_calls.empty()) {
            // No tool calls or no memory tools — final response
            final_assistant_content = assistant_content;
            final_thinking_content  = thinking_content;
            break;
        }

        // Execute each tool call and persist results
        for (const auto& tc : accumulated_tool_calls) {
            Message tool_result;
            if (tool_exec.is_memory_tool(tc.function_name)) {
                tool_result = tool_exec.execute_tool(tc, conv_id);
            } else {
                // Non-memory tool call — return error (not handled server-side)
                tool_result.role         = MessageRole::Tool;
                tool_result.tool_call_id = tc.id;
                tool_result.content      = nlohmann::json{{"error", "tool not available"}}.dump();
                tool_result.timestamp_ms = util::now_ms();
            }

            db.append_message(conv_id, tool_result, seq + 1);
            ++seq;

            // Emit tool_result SSE event to client
            InferenceChunk tool_chunk;
            tool_chunk.tool_result_json = nlohmann::json{
                {"type", "tool_result"},
                {"tool_call_id", tc.id},
                {"function_name", tc.function_name},
                {"result", tool_result.content}
            }.dump();
            chunk_cb(tool_chunk);
        }

        // Loop: re-infer with updated context (tool results now in messages)
        MM_DEBUG("handle_chat: tool round {} complete, re-inferring (agent='{}')",
                 tool_round + 1, agent_id);
    }

    // Log completion or failure
    {
        int64_t duration_ms = util::now_ms() - infer_start_ms;
        if (ok) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "Chat complete: agent='%s' tokens=%d content=%zuB duration=%lldms",
                     agent_id.c_str(), final_total_tokens,
                     final_assistant_content.size(), static_cast<long long>(duration_ms));
            activity_log(0, buf);
        } else {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "Chat error: agent='%s' node=%s slot=%s reason='%s' duration=%lldms",
                     agent_id.c_str(),
                     schedule_result->node_id.substr(0, 8).c_str(),
                     active_slot_id.substr(0, 8).c_str(),
                     failure_reason.substr(0, 80).c_str(),
                     static_cast<long long>(duration_ms));
            activity_log(2, buf);
        }
    }

    if (!ok) {
        MM_WARN("handle_chat: skipping assistant persistence for failed stream (agent='{}', conv='{}')",
                agent_id, conv_id);
    }

    // 10. Mark agent idle in scheduler.
    scheduler_.mark_agent_idle(agent_id);

    // 11. Signal completion to the SSE client.
    done_cb(conv_id, ok, failure_reason);
}

// ── queue_global_recall ───────────────────────────────────────────────────────
// Queues an internal inference job where the agent reviews the old conversation
// and creates global memories from local notes + conversation content.

void ControlApiServer::queue_global_recall(const AgentId& agent_id,
                                           const ConvId& conv_id) {
    Agent* agent = agents_.get_agent(agent_id);
    if (!agent) return;

    AgentConfig cfg = agent->get_config();
    if (!cfg.memories_enabled) return;

    activity_log(0, "Global recall queued: agent='" + agent_id +
                 "' conv='" + conv_id.substr(0, 8) + "'");

    InferenceJob recall_job;
    recall_job.job_id   = util::generate_uuid();
    recall_job.agent_id = agent_id;
    recall_job.process_fn = [this, agent_id, conv_id]() {
        Agent* a = agents_.get_agent(agent_id);
        if (!a) return;
        AgentConfig acfg = a->get_config();
        if (!acfg.memories_enabled) return;

        activity_log(0, "Global recall starting: agent='" + agent_id + "'");

        // Schedule agent on a node
        auto sched_result = scheduler_.ensure_agent_running(acfg);
        if (!sched_result) {
            MM_WARN("Global recall: no node for agent '{}': {}",
                    agent_id, scheduler_.last_error());
            activity_log(1, "Global recall failed: agent='" + agent_id + "' no node");
            return;
        }

        scheduler_.mark_agent_active(agent_id);

        NodeInfo ni;
        try { ni = registry_.get_node(sched_result->node_id); }
        catch (const std::exception& e) {
            scheduler_.mark_agent_idle(agent_id);
            MM_WARN("Global recall: node lookup failed: {}", e.what());
            return;
        }

        AgentDB& db = a->db();
        ToolExecutor tool_exec(db);

        // Build recall context: system prompt + conversation local memories
        auto local_mems = db.list_local_memories(conv_id);
        auto global_mems = db.list_memories();
        auto conv = db.load_conversation(conv_id);

        std::string system_text =
            "You are reviewing a completed conversation to extract important information "
            "worth remembering across future conversations.\n\n"
            "Review the conversation below and use the available tools to create global memories "
            "for any important facts, preferences, decisions, or context that would be useful "
            "in future conversations. Be selective — only save information that is genuinely "
            "important and not already captured in existing global memories.\n\n"
            "You can also update or delete existing global memories if the conversation "
            "revealed new information that changes them.";

        if (!global_mems.empty()) {
            system_text += "\n\n## Existing global memories\n";
            for (const auto& gm : global_mems) {
                system_text += "- [" + gm.id + "] (importance=" +
                               std::to_string(gm.importance) + ") " + gm.content + "\n";
            }
        }

        if (!local_mems.empty()) {
            system_text += "\n\n## Local notes from this conversation\n";
            for (const auto& lm : local_mems) {
                system_text += "- [" + lm.id + "] " + lm.content + "\n";
            }
        }

        std::vector<Message> recall_ctx;
        Message sys;
        sys.role    = MessageRole::System;
        sys.content = system_text;
        recall_ctx.push_back(sys);

        // Add conversation messages as context
        if (conv) {
            if (!conv->compaction_summary.empty()) {
                Message summary_msg;
                summary_msg.role    = MessageRole::Assistant;
                summary_msg.content = "[Previous conversation summary]\n" + conv->compaction_summary;
                recall_ctx.push_back(summary_msg);
            }
            for (const auto& m : conv->messages) {
                recall_ctx.push_back(m);
            }
        }

        // Add a user prompt to trigger the recall
        Message recall_prompt;
        recall_prompt.role    = MessageRole::User;
        recall_prompt.content = "Please review this conversation and save any important "
                                "information as global memories.";
        recall_ctx.push_back(recall_prompt);

        // Run tool call loop (internal, not streamed to client)
        NodeProxyLlamaClient recall_llama(ni.url, ni.api_key, sched_result->slot_id);
        static constexpr int kMaxRecallRounds = 5;

        for (int round = 0; round < kMaxRecallRounds; ++round) {
            InferenceRequest req;
            req.model    = acfg.model_path;
            req.messages = recall_ctx;
            req.settings = acfg.llama_settings;
            req.tools    = tool_exec.get_all_tool_definitions();
            req.stream   = false;

            Message response = recall_llama.complete(req);

            // Add assistant response to context
            recall_ctx.push_back(response);

            // Check for tool calls
            bool has_tools = false;
            for (const auto& tc : response.tool_calls) {
                if (tool_exec.is_memory_tool(tc.function_name)) {
                    has_tools = true;
                    break;
                }
            }

            if (!has_tools || response.tool_calls.empty()) break;

            // Execute tool calls
            for (const auto& tc : response.tool_calls) {
                Message tool_result;
                if (tool_exec.is_memory_tool(tc.function_name)) {
                    tool_result = tool_exec.execute_tool(tc, conv_id);
                } else {
                    tool_result.role         = MessageRole::Tool;
                    tool_result.tool_call_id = tc.id;
                    tool_result.content      = nlohmann::json{{"error", "tool not available"}}.dump();
                    tool_result.timestamp_ms = util::now_ms();
                }
                recall_ctx.push_back(tool_result);
            }
        }

        scheduler_.mark_agent_idle(agent_id);
        activity_log(0, "Global recall complete: agent='" + agent_id + "'");
    };
    queue_.enqueue(std::move(recall_job));
}

// ── register_routes ───────────────────────────────────────────────────────────

void ControlApiServer::register_routes() {
    using namespace httplib;

    // Helper: compute node_compatibility info for an agent config.
    auto compute_node_compat = [this](const AgentConfig& cfg) -> nlohmann::json {
        int64_t estimated_vram = 2048; // conservative default
        const std::string resolved = resolve_model_path_for_metadata(cfg.model_path, models_dir_);
        std::error_code ec;
        auto sz = std::filesystem::file_size(resolved.empty() ? cfg.model_path : resolved, ec);
        if (!ec) {
            estimated_vram = static_cast<int64_t>(
                static_cast<double>(sz) * 1.2 / (1024.0 * 1024.0));
        }

        auto compatible = registry_.nodes_with_available_vram(estimated_vram);
        auto all_nodes  = registry_.list_nodes();
        int total_connected = 0;
        for (const auto& n : all_nodes) {
            if (n.connected) ++total_connected;
        }

        return nlohmann::json{
            {"estimated_vram_mb",  estimated_vram},
            {"compatible_nodes",   static_cast<int>(compatible.size())},
            {"total_nodes",        total_connected}
        };
    };

    auto extract_bearer = [](const std::string& auth_header) -> std::string {
        static const std::string kBearer = "Bearer ";
        if (auth_header.rfind(kBearer, 0) != 0) return {};
        return auth_header.substr(kBearer.size());
    };

    auto require_node_auth = [this, extract_bearer](const Request& req, Response& res) -> bool {
        const std::string token = extract_bearer(req.get_header_value("Authorization"));
        if (token.empty()) {
            res.status = 401;
            res.set_content(R"({"error":"missing bearer token"})", "application/json");
            return false;
        }
        if (!registry_.find_node_by_api_key(token).has_value()) {
            res.status = 401;
            res.set_content(R"({"error":"invalid node api key"})", "application/json");
            return false;
        }
        return true;
    };

    struct ControlCatalogSnapshot {
        std::vector<StoredModel> models;
        std::unordered_map<std::string, StoredModel> by_filename;
        std::unordered_map<std::string, std::string> path_by_filename;
    };

    auto build_control_catalog = [this](bool include_hash) -> ControlCatalogSnapshot {
        ControlCatalogSnapshot out;

        auto add_file = [&](const std::string& file_path,
                            const std::string& source_tag) {
            auto meta = inspect_model_file(file_path, include_hash);
            if (!meta) return;
            const std::string filename = canonical_model_filename(meta->model_path);
            if (!is_safe_model_filename(filename)) return;

            auto pit = out.path_by_filename.find(filename);
            if (pit != out.path_by_filename.end()) {
                if (pit->second != file_path) {
                    MM_WARN("Control catalog: duplicate filename '{}' from {}; keeping {}",
                            filename, source_tag, pit->second);
                }
                return;
            }

            meta->model_path = filename;
            out.path_by_filename[filename] = file_path;
            out.by_filename[filename] = *meta;
        };

        std::error_code ec;
        if (!models_dir_.empty() && std::filesystem::exists(models_dir_, ec)) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(models_dir_, ec)) {
                if (!entry.is_regular_file()) continue;
                add_file(entry.path().string(), "models_dir");
            }
        }

        const auto agent_cfgs = agents_.list_agents();
        for (const auto& cfg : agent_cfgs) {
            const std::string resolved = resolve_model_path_for_metadata(cfg.model_path, models_dir_);
            if (resolved.empty()) continue;
            add_file(resolved, "agent:" + cfg.id);
        }

        out.models.reserve(out.by_filename.size());
        for (const auto& [_, m] : out.by_filename) out.models.push_back(m);
        std::sort(out.models.begin(), out.models.end(),
                  [](const StoredModel& a, const StoredModel& b) {
                      return a.model_path < b.model_path;
                  });
        return out;
    };

    // ── POST /api/control/register-node ───────────────────────────────────────
    server_->Post("/api/control/register-node", [this](const Request& req, Response& res) {
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string node_url = j.value("node_url", "");
            std::string api_key  = j.value("api_key", "");
            std::string platform = j.value("platform", "");

            if (node_url.empty() || api_key.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"node_url and api_key required"})",
                                "application/json");
                return;
            }

            NodeId node_id = registry_.add_node(node_url, api_key, platform);
            MM_INFO("Node registered: {} @ {}", node_id, node_url);

            res.set_content(
                nlohmann::json{{"node_id", node_id}, {"accepted", true}}.dump(),
                "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── GET /v1/nodes ─────────────────────────────────────────────────────────
    // ── GET /api/control/models  (node-authenticated) ─────────────────────────
    server_->Get("/api/control/models", [this, require_node_auth, build_control_catalog](const Request& req, Response& res) {
        if (!require_node_auth(req, res)) return;
        auto catalog = build_control_catalog(/*include_hash=*/true);
        res.set_content(nlohmann::json{{"models", catalog.models}}.dump(), "application/json");
    });

    // ── GET /api/control/models/:filename/content  (node-authenticated) ───────
    server_->Get("/api/control/models/:filename/content",
                 [this, require_node_auth, build_control_catalog](const Request& req, Response& res) {
        if (!require_node_auth(req, res)) return;

        std::string filename = canonical_model_filename(req.path_params.at("filename"));
        if (!is_safe_model_filename(filename)) {
            res.status = 400;
            res.set_content(R"({"error":"invalid filename"})", "application/json");
            return;
        }

        auto catalog = build_control_catalog(/*include_hash=*/true);
        auto mit = catalog.by_filename.find(filename);
        if (mit == catalog.by_filename.end()) {
            res.status = 404;
            res.set_content(R"({"error":"model not found"})", "application/json");
            return;
        }
        auto pit = catalog.path_by_filename.find(filename);
        if (pit == catalog.path_by_filename.end()) {
            res.status = 404;
            res.set_content(R"({"error":"model file path could not be resolved"})", "application/json");
            return;
        }

        auto file = std::make_shared<std::ifstream>(pit->second, std::ios::binary);
        if (!file->is_open()) {
            res.status = 404;
            res.set_content(R"({"error":"model file could not be opened"})", "application/json");
            return;
        }

        res.set_header("X-Model-Filename", filename);
        res.set_header("X-Model-SHA256", mit->second.sha256);
        res.set_header("X-Model-Size-Bytes", std::to_string(mit->second.size_bytes));
        res.set_chunked_content_provider(
            "application/octet-stream",
            [file](size_t /*offset*/, DataSink& sink) -> bool {
                char buf[64 * 1024];
                file->read(buf, static_cast<std::streamsize>(sizeof(buf)));
                std::streamsize n = file->gcount();
                if (n > 0) {
                    return sink.write(buf, static_cast<size_t>(n));
                }
                return false;
            });
    });

    server_->Get("/v1/nodes", [this](const Request& /*req*/, Response& res) {
        auto nodes = registry_.list_nodes();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& n : nodes) arr.push_back(nlohmann::json(n));
        res.set_content(arr.dump(), "application/json");
    });

    // ── GET /v1/agents ────────────────────────────────────────────────────────
    // ── GET /v1/models ────────────────────────────────────────────────────────
    server_->Get("/v1/models", [this, build_control_catalog](const Request& /*req*/, Response& res) {
        auto catalog = build_control_catalog(/*include_hash=*/false);
        res.set_content(nlohmann::json{{"models", catalog.models}}.dump(), "application/json");
    });

    // ── GET /v1/nodes/:id/models ──────────────────────────────────────────────
    server_->Get("/v1/nodes/:id/models", [this](const Request& req, Response& res) {
        const std::string node_id = req.path_params.at("id");
        NodeInfo node;
        try {
            node = registry_.get_node(node_id);
        } catch (...) {
            res.status = 404;
            res.set_content(R"({"error":"node not found"})", "application/json");
            return;
        }

        auto [host, port] = util::parse_url(node.url);
        httplib::Client quick_cli(host, port);
        quick_cli.set_connection_timeout(1);
        quick_cli.set_read_timeout(2);
        quick_cli.set_write_timeout(2);
        httplib::Headers headers = {{"Authorization", "Bearer " + node.api_key}};
        auto storage = quick_cli.Get("/api/node/storage", headers);
        if (storage && storage->status >= 200 && storage->status < 300) {
            res.set_content(storage->body, "application/json");
            return;
        }

        nlohmann::json body = {
            {"stored_models", node.stored_models},
            {"disk_free_mb", node.disk_free_mb},
            {"warning", "node storage endpoint unavailable; returning last known snapshot"}
        };
        res.status = 200;
        res.set_content(body.dump(), "application/json");
    });

    // ── POST /v1/nodes/:id/models/pull ────────────────────────────────────────
    server_->Post("/v1/nodes/:id/models/pull", [this](const Request& req, Response& res) {
        const std::string node_id = req.path_params.at("id");
        NodeInfo node;
        try {
            node = registry_.get_node(node_id);
        } catch (...) {
            res.status = 404;
            res.set_content(R"({"error":"node not found"})", "application/json");
            return;
        }

        try {
            auto j = nlohmann::json::parse(req.body);
            std::string model_filename = canonical_model_filename(
                j.value("model_filename", std::string{}));
            bool force = j.value("force", false);
            if (!is_safe_model_filename(model_filename)) {
                res.status = 400;
                res.set_content(R"({"error":"valid model_filename required"})", "application/json");
                return;
            }

            HttpClient cli(node.url);
            cli.set_bearer_token(node.api_key);
            auto pull = cli.post("/api/node/models/pull",
                                 nlohmann::json{{"model_filename", model_filename},
                                                {"force", force}});
            res.status = pull.status == 0 ? 502 : pull.status;
            res.set_content(pull.body.empty()
                                ? nlohmann::json{{"error", "node pull request failed"}}.dump()
                                : pull.body,
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // ── DELETE /v1/nodes/:id/models/:filename ─────────────────────────────────
    server_->Delete("/v1/nodes/:id/models/:filename", [this](const Request& req, Response& res) {
        const std::string node_id = req.path_params.at("id");
        std::string filename = canonical_model_filename(req.path_params.at("filename"));
        if (!is_safe_model_filename(filename)) {
            res.status = 400;
            res.set_content(R"({"error":"invalid filename"})", "application/json");
            return;
        }

        NodeInfo node;
        try {
            node = registry_.get_node(node_id);
        } catch (...) {
            res.status = 404;
            res.set_content(R"({"error":"node not found"})", "application/json");
            return;
        }

        HttpClient cli(node.url);
        cli.set_bearer_token(node.api_key);
        auto del = cli.del("/api/node/models/" + filename);
        res.status = del.status == 0 ? 502 : del.status;
        res.set_content(del.body.empty()
                            ? nlohmann::json{{"error", "node delete request failed"}}.dump()
                            : del.body,
                        "application/json");
    });

    server_->Get("/v1/agents", [this](const Request& /*req*/, Response& res) {
        auto configs = agents_.list_agents();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& c : configs) {
            nlohmann::json j = c;
            auto placement = scheduler_.get_placement(c.id);
            if (placement) {
                j["placement"] = *placement;
                j["status"] = placement->suspended
                    ? "suspended"
                    : (placement->is_active ? "active" : "idle");
            } else {
                j["status"] = "unplaced";
            }
            arr.push_back(std::move(j));
        }
        res.set_content(arr.dump(), "application/json");
    });

    // ── POST /v1/agents ───────────────────────────────────────────────────────
    server_->Post("/v1/agents", [this, compute_node_compat](const Request& req, Response& res) {
        try {
            AgentConfig cfg = nlohmann::json::parse(req.body).get<AgentConfig>();
            auto validation = validate_agent_config(cfg, &registry_, models_dir_);
            if (!validation.ok()) {
                nlohmann::json body = {
                    {"error", "validation_failed"},
                    {"issues", validation.issues}
                };
                if (validation.model_info) body["model_info"] = *validation.model_info;
                res.status = 400;
                res.set_content(body.dump(), "application/json");
                return;
            }
            AgentId id = agents_.create_agent(cfg);
            Agent* a = agents_.get_agent(id);
            nlohmann::json resp = a ? nlohmann::json(a->get_config())
                                    : nlohmann::json{{"id", id}};
            if (a) {
                resp["node_compatibility"] = compute_node_compat(a->get_config());
            }
            res.status = 201;
            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── GET /v1/agents/:id ────────────────────────────────────────────────────
    server_->Get("/v1/agents/:id", [this, compute_node_compat](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        auto cfg = a->get_config();
        nlohmann::json j = cfg;
        // Include placement info if available.
        auto placement = scheduler_.get_placement(id);
        if (placement) {
            j["placement"] = *placement;
            j["status"] = placement->suspended
                ? "suspended"
                : (placement->is_active ? "active" : "idle");
        } else {
            j["status"] = "unplaced";
        }
        j["node_compatibility"] = compute_node_compat(cfg);
        res.set_content(j.dump(), "application/json");
    });

    // ── PUT /v1/agents/:id ────────────────────────────────────────────────────
    server_->Put("/v1/agents/:id", [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        try {
            AgentConfig cfg = nlohmann::json::parse(req.body).get<AgentConfig>();
            // If body omits "id", keep the current ID (no rename).
            if (cfg.id.empty()) cfg.id = id;
            auto validation = validate_agent_config(cfg, &registry_, models_dir_);
            if (!validation.ok()) {
                nlohmann::json body = {
                    {"error", "validation_failed"},
                    {"issues", validation.issues}
                };
                if (validation.model_info) body["model_info"] = *validation.model_info;
                res.status = 400;
                res.set_content(body.dump(), "application/json");
                return;
            }
            AgentId final_id = agents_.update_agent(id, cfg);
            if (final_id.empty()) { res.status = 404; return; }
            Agent* a = agents_.get_agent(final_id);
            if (!a) { res.status = 404; return; }
            res.set_content(nlohmann::json(a->get_config()).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── DELETE /v1/agents/:id ─────────────────────────────────────────────────
    server_->Delete("/v1/agents/:id", [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        scheduler_.release_agent(id);
        if (!agents_.delete_agent(id)) { res.status = 404; return; }
        res.set_content(R"({"status":"deleted"})", "application/json");
    });

    // ── POST /v1/agents/:id/chat  (SSE streaming) ─────────────────────────────
    server_->Post("/v1/agents/:id/chat", [this](const Request& req, Response& res) {
        std::string agent_id = req.path_params.at("id");
        if (!agents_.get_agent(agent_id)) { res.status = 404; return; }

        std::string message;
        ConvId      conv_id_hint;
        int         max_tokens_override = 0;
        try {
            auto j      = nlohmann::json::parse(req.body);
            message     = j.value("message", "");
            conv_id_hint = j.value("conversation_id", "");
            max_tokens_override = j.value("max_tokens", 0);
        } catch (const std::exception& e) {
            MM_WARN("POST /v1/agents/:id/chat: invalid JSON: {}", e.what());
            res.status = 400;
            res.set_content(R"({"error":"invalid JSON"})", "application/json");
            return;
        }
        if (message.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"message required"})", "application/json");
            return;
        }

        // Shared context between worker thread and SSE content provider.
        auto ctx = std::make_shared<SseInferCtx>();

        // Chunk callback: forward InferenceChunks as SSE events to client.
        auto chunk_fn = [ctx](const InferenceChunk& chunk) {
            std::string payload;
            if (!chunk.tool_result_json.empty()) {
                payload = "data: " + chunk.tool_result_json + "\n\n";
            } else if (!chunk.thinking_delta.empty()) {
                payload = "data: " +
                    nlohmann::json{{"type", "thinking"},
                                   {"content", chunk.thinking_delta}}.dump() + "\n\n";
            } else if (chunk.tool_call_delta) {
                auto& tc = *chunk.tool_call_delta;
                payload = "data: " +
                    nlohmann::json{{"type",      "tool_call"},
                                   {"name",      tc.function_name},
                                   {"arguments", tc.arguments_json}}.dump() + "\n\n";
            } else if (!chunk.delta_content.empty()) {
                payload = "data: " +
                    nlohmann::json{{"type",    "delta"},
                                   {"content", chunk.delta_content}}.dump() + "\n\n";
            }
            if (!payload.empty()) {
                std::lock_guard<std::mutex> lk(ctx->mx);
                ctx->lines.push_back(std::move(payload));
                ctx->cv.notify_one();
            }
        };

        // Done callback: emit done event and signal provider to stop.
        auto done_fn = [ctx](const ConvId& conv_id, bool success, const std::string& error) {
            nlohmann::json done_payload = {
                {"type", "done"},
                {"conv_id", conv_id},
                {"success", success},
            };
            if (!success && !error.empty()) {
                done_payload["error"] = error;
            }
            std::string payload = "data: " + done_payload.dump() + "\n\n";
            std::lock_guard<std::mutex> lk(ctx->mx);
            ctx->lines.push_back(std::move(payload));
            ctx->done = true;
            ctx->cv.notify_one();
        };

        // Enqueue job — worker thread calls handle_chat via process_fn.
        InferenceJob job;
        job.job_id   = util::generate_uuid();
        job.agent_id = agent_id;
        job.process_fn = [this, agent_id, message, conv_id_hint,
                          max_tokens_override, chunk_fn, done_fn]() mutable {
            handle_chat(agent_id, message, conv_id_hint,
                        std::move(chunk_fn), std::move(done_fn),
                        max_tokens_override);
        };
        queue_.enqueue(std::move(job));

        // SSE content provider: drain lines from SseInferCtx.
        res.set_chunked_content_provider("text/event-stream",
            [ctx](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::unique_lock<std::mutex> lk(ctx->mx);
                ctx->cv.wait(lk, [&] {
                    return !ctx->lines.empty() || ctx->done;
                });

                while (!ctx->lines.empty()) {
                    std::string payload = std::move(ctx->lines.front());
                    ctx->lines.pop_front();
                    lk.unlock();
                    if (!sink.write(payload.data(), payload.size())) return false;
                    lk.lock();
                }

                if (ctx->done && ctx->lines.empty()) {
                    lk.unlock();
                    const std::string fin = "data: [DONE]\n\n";
                    sink.write(fin.data(), fin.size());
                    return false;
                }
                return true;
            });
    });

    // ── GET /v1/placements ──────────────────────────────────────────────────
    server_->Get("/v1/placements", [this](const Request& /*req*/, Response& res) {
        auto placements = scheduler_.list_placements();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& p : placements) arr.push_back(nlohmann::json(p));
        res.set_content(arr.dump(), "application/json");
    });

    // ── GET /v1/agents/:id/conversations ──────────────────────────────────────
    server_->Get("/v1/agents/:id/conversations",
                 [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        auto convs = a->db().list_conversations();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& c : convs) arr.push_back(nlohmann::json(c));
        res.set_content(arr.dump(), "application/json");
    });

    // ── POST /v1/agents/:id/conversations ─────────────────────────────────────
    server_->Post("/v1/agents/:id/conversations",
                  [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        try {
            nlohmann::json j = nlohmann::json::object();
            if (!util::trim(req.body).empty()) {
                j = nlohmann::json::parse(req.body);
            }

            const std::string title = j.value("title", std::string{});
            const std::string parent_conv_id = j.value("parent_conv_id", std::string{});
            const bool set_active = j.value("set_active", false);

            if (!parent_conv_id.empty() && !a->db().conversation_exists(parent_conv_id)) {
                res.status = 400;
                res.set_content(
                    nlohmann::json{{"error", "parent conversation not found"}}.dump(),
                    "application/json");
                return;
            }

            ConvId cid = a->db().create_conversation(title, parent_conv_id);
            if (cid.empty()) {
                res.status = 400;
                res.set_content(
                    nlohmann::json{{"error", "failed to create conversation"}}.dump(),
                    "application/json");
                return;
            }

            if (set_active) {
                // Trigger global recall on the previously active conversation
                auto prev_active = a->db().get_active_conversation_id();
                if (prev_active && *prev_active != cid) {
                    queue_global_recall(id, *prev_active);
                }
                a->db().set_active_conversation(cid);
            }

            auto conv_opt = a->db().load_conversation(cid);
            if (!conv_opt) {
                res.status = 500;
                res.set_content(
                    nlohmann::json{{"error", "conversation was created but could not be loaded"}}.dump(),
                    "application/json");
                return;
            }

            res.status = 201;
            res.set_content(nlohmann::json{{"conversation", *conv_opt}}.dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── GET /v1/agents/:id/conversations/:cid ─────────────────────────────────
    server_->Get("/v1/agents/:id/conversations/:cid",
                 [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        auto conv_opt = a->db().load_conversation(cid);
        if (!conv_opt) { res.status = 404; return; }
        res.set_content(nlohmann::json(*conv_opt).dump(), "application/json");
    });

    // ── POST /v1/agents/:id/conversations/:cid/activate ──────────────────────
    server_->Post("/v1/agents/:id/conversations/:cid/activate",
                  [this](const Request& req, Response& res) {
        (void)req.body;
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }

        // Trigger global recall on the previously active conversation
        auto prev_active = a->db().get_active_conversation_id();
        if (prev_active && *prev_active != cid) {
            queue_global_recall(id, *prev_active);
        }

        a->db().set_active_conversation(cid);
        res.set_content(
            nlohmann::json{{"status", "ok"}, {"active_conversation_id", cid}}.dump(),
            "application/json");
    });

    // ── POST /v1/agents/:id/conversations/:cid/compact ───────────────────────
    server_->Post("/v1/agents/:id/conversations/:cid/compact",
                  [this](const Request& req, Response& res) {
        (void)req.body;
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");

        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }

        auto cfg = a->get_config();
        auto sched_result = scheduler_.ensure_agent_running(cfg);
        if (!sched_result) {
            std::string err = scheduler_.last_error();
            if (err.empty()) err = "no node available or model failed to load";
            res.status = 503;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }

        scheduler_.mark_agent_active(id);
        auto mark_idle = [&]() { scheduler_.mark_agent_idle(id); };

        try {
            NodeInfo node = registry_.get_node(sched_result->node_id);
            NodeProxyLlamaClient real_llama(node.url, node.api_key, sched_result->slot_id);
            ConversationManager conv_mgr(a->db(), real_llama);
            ConvId new_id = conv_mgr.force_compact(cid, cfg);
            mark_idle();

            res.set_content(
                nlohmann::json{
                    {"status", "ok"},
                    {"old_conversation_id", cid},
                    {"new_conversation_id", new_id}
                }.dump(),
                "application/json");
        } catch (const std::exception& e) {
            mark_idle();
            res.status = 500;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── DELETE /v1/agents/:id/conversations/:cid ──────────────────────────────
    server_->Delete("/v1/agents/:id/conversations/:cid",
                    [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }
        if (a->db().is_conversation_active(cid)) {
            res.status = 409;
            res.set_content(
                nlohmann::json{
                    {"error", "cannot delete active conversation; activate another conversation first"}
                }.dump(),
                "application/json");
            return;
        }
        a->db().delete_conversation(cid);
        res.set_content(R"({"status":"deleted"})", "application/json");
    });

    // ── GET /v1/agents/:id/conversations/:cid/local-memories ─────────────────
    server_->Get("/v1/agents/:id/conversations/:cid/local-memories",
                 [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }

        auto mems = a->db().list_local_memories(cid);
        nlohmann::json arr = nlohmann::json::array();
        for (auto& m : mems) arr.push_back(nlohmann::json(m));
        res.set_content(arr.dump(), "application/json");
    });

    // ── POST /v1/agents/:id/memories/extract ──────────────────────────────────
    server_->Post("/v1/agents/:id/memories/extract",
                  [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string conv_id = j.value("conversation_id", std::string{});
            const int start_index = j.value("start_index", -1);
            const int end_index = j.value("end_index", -1);
            int context_before = j.value("context_before", 2);
            context_before = std::clamp(context_before, 0, 20);

            if (conv_id.empty()) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "conversation_id is required"}}.dump(),
                                "application/json");
                return;
            }
            if (start_index < 0 || end_index < 0 || start_index > end_index) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "invalid message range"}}.dump(),
                                "application/json");
                return;
            }

            auto conv_opt = a->db().load_conversation(conv_id);
            if (!conv_opt) { res.status = 404; return; }

            const auto& msgs = conv_opt->messages;
            if (msgs.empty()) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "conversation has no messages"}}.dump(),
                                "application/json");
                return;
            }
            if (end_index >= static_cast<int>(msgs.size())) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "message range out of bounds"}}.dump(),
                                "application/json");
                return;
            }

            const int begin_index = std::max(0, start_index - context_before);
            std::vector<Message> selected(
                msgs.begin() + begin_index,
                msgs.begin() + end_index + 1);

            InferenceJob job;
            job.job_id   = util::generate_uuid();
            job.agent_id = id;
            job.process_fn = [this, id, conv_id, selected]() {
                Agent* a_local = agents_.get_agent(id);
                if (!a_local) return;

                AgentConfig cfg = a_local->get_config();
                auto sched_result = scheduler_.ensure_agent_running(cfg);
                if (!sched_result) {
                    MM_WARN("Manual memory extraction: no node for agent '{}': {}",
                            id, scheduler_.last_error());
                    return;
                }

                scheduler_.mark_agent_active(id);
                auto mark_idle = [&]() { scheduler_.mark_agent_idle(id); };

                try {
                    NodeInfo node = registry_.get_node(sched_result->node_id);
                    NodeProxyLlamaClient ext_llama(node.url, node.api_key, sched_result->slot_id);
                    MemoryManager mem_mgr(a_local->db(), ext_llama);
                    mem_mgr.extract_and_store_memories_from_messages(conv_id, selected, cfg);
                    mark_idle();
                } catch (const std::exception& e) {
                    mark_idle();
                    MM_WARN("Manual memory extraction failed for '{}': {}", id, e.what());
                }
            };
            queue_.enqueue(std::move(job));

            res.set_content(
                nlohmann::json{
                    {"status", "queued"},
                    {"conversation_id", conv_id},
                    {"selected_count", end_index - start_index + 1}
                }.dump(),
                "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── GET /v1/agents/:id/memories ───────────────────────────────────────────
    server_->Get("/v1/agents/:id/memories",
                 [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        auto mems = a->db().list_memories();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& m : mems) arr.push_back(nlohmann::json(m));
        res.set_content(arr.dump(), "application/json");
    });

    // ── PUT /v1/agents/:id/memories/:mid ──────────────────────────────────────
    server_->Put("/v1/agents/:id/memories/:mid",
                 [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string mid = req.path_params.at("mid");
        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        try {
            auto j = nlohmann::json::parse(req.body);
            Memory mem;
            mem.id         = mid;
            mem.agent_id   = id;
            mem.content    = j.value("content", "");
            mem.importance = j.value("importance", 0.5f);
            a->db().update_memory(mem);
            res.set_content(nlohmann::json(mem).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── DELETE /v1/agents/:id/memories/:mid ───────────────────────────────────
    server_->Delete("/v1/agents/:id/memories/:mid",
                    [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string mid = req.path_params.at("mid");
        Agent* a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        a->db().delete_memory(mid);
        res.set_content(R"({"status":"deleted"})", "application/json");
    });
}

} // namespace mm
