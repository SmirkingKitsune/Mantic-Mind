#include "common/runtime_client.hpp"
#include "common/inference_response_parser.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>

namespace mm {

namespace {

// ── OpenAI request building ───────────────────────────────────────────────────
nlohmann::json message_to_openai(const Message& m) {
    nlohmann::json j;
    j["role"]    = to_string(m.role);
    if (m.content_parts.empty()) {
        j["content"] = m.content;
    } else {
        auto& content = j["content"] = nlohmann::json::array();
        for (const auto& part : m.content_parts) {
            if (part.type == "text") {
                content.push_back({{"type", "text"}, {"text", part.text}});
            } else if (part.type == "image_url" && !part.image_url.empty()) {
                content.push_back({{"type", "image_url"},
                                   {"image_url", {{"url", part.image_url}}}});
            }
        }
    }
    if (!m.tool_calls.empty()) {
        auto& arr = j["tool_calls"] = nlohmann::json::array();
        for (auto& tc : m.tool_calls)
            arr.push_back({ {"id", tc.id}, {"type", "function"},
                {"function", {{"name",        tc.function_name},
                              {"arguments",   tc.arguments_json}}} });
    }
    if (!m.tool_call_id.empty()) j["tool_call_id"] = m.tool_call_id;
    return j;
}

void append_context_line(std::string& dst, const std::string& line) {
    const std::string trimmed = util::trim(line);
    if (trimmed.empty()) return;
    if (!dst.empty()) dst += "\n\n";
    dst += trimmed;
}

// Some chat templates (e.g. strict Jinja variants) require user/assistant
// alternation after an optional system message.
std::vector<Message> normalize_for_strict_chat_template(const std::vector<Message>& messages) {
    std::string system_context;
    std::vector<Message> out;
    out.reserve(messages.size());

    auto push_turn = [&](MessageRole role, const std::string& content) {
        std::string text = util::trim(content);
        if (text.empty()) return;

        if (role != MessageRole::User && role != MessageRole::Assistant) {
            append_context_line(system_context, text);
            return;
        }

        if (out.empty()) {
            if (role == MessageRole::Assistant) {
                append_context_line(system_context, "[assistant context]\n" + text);
                return;
            }
            Message m;
            m.role = MessageRole::User;
            m.content = text;
            out.push_back(std::move(m));
            return;
        }

        if (out.back().role == role) {
            out.back().content += "\n\n" + text;
            return;
        }

        Message m;
        m.role = role;
        m.content = text;
        out.push_back(std::move(m));
    };

    for (const auto& m : messages) {
        switch (m.role) {
            case MessageRole::System:
                append_context_line(system_context, m.content);
                break;
            case MessageRole::User:
                push_turn(MessageRole::User, m.content);
                break;
            case MessageRole::Assistant:
                if (!m.content.empty()) {
                    push_turn(MessageRole::Assistant, m.content);
                } else if (!m.tool_calls.empty()) {
                    nlohmann::json tc = nlohmann::json::array();
                    for (const auto& t : m.tool_calls) {
                        tc.push_back({
                            {"id", t.id},
                            {"name", t.function_name},
                            {"arguments", t.arguments_json},
                        });
                    }
                    push_turn(MessageRole::Assistant, "[tool calls]\n" + tc.dump());
                }
                break;
            case MessageRole::Tool:
                append_context_line(system_context, "[tool result]\n" + m.content);
                break;
        }
    }

    if (out.empty()) {
        Message m;
        m.role = MessageRole::User;
        m.content = system_context.empty()
            ? "Hello."
            : ("Use the following context:\n\n" + system_context + "\n\nRespond to the latest user request.");
        system_context.clear();
        out.push_back(std::move(m));
    }

    if (out.front().role != MessageRole::User) {
        Message lead;
        lead.role = MessageRole::User;
        lead.content = "Please continue.";
        out.insert(out.begin(), std::move(lead));
    }

    // Avoid ending on an assistant turn. Some providers treat a trailing
    // assistant message as response prefill and reject it when thinking mode
    // is enabled.
    if (!out.empty() && out.back().role == MessageRole::Assistant) {
        Message tail;
        tail.role = MessageRole::User;
        tail.content = "Continue with the next response based on the latest context.";
        out.push_back(std::move(tail));
    }

    std::vector<Message> normalized;
    if (!system_context.empty()) {
        Message sys;
        sys.role = MessageRole::System;
        sys.content = system_context;
        normalized.push_back(std::move(sys));
    }
    normalized.insert(normalized.end(), out.begin(), out.end());
    return normalized;
}

nlohmann::json build_request(const InferenceRequest& req, bool stream) {
    nlohmann::json body;
    if (!req.model.empty()) body["model"] = req.model;

    const bool has_multimodal = std::any_of(
        req.messages.begin(), req.messages.end(), [](const Message& message) {
            return std::any_of(message.content_parts.begin(), message.content_parts.end(),
                               [](const MessageContentPart& part) {
                                   return part.type == "image_url" ||
                                          part.type == "image_attachment";
                               });
        });
    auto normalized_messages = has_multimodal
        ? req.messages : normalize_for_strict_chat_template(req.messages);
    auto& msgs = body["messages"] = nlohmann::json::array();
    for (auto& m : normalized_messages) msgs.push_back(message_to_openai(m));

    const auto& s = req.settings;
    body["temperature"] = static_cast<double>(s.temperature);
    body["top_p"]       = static_cast<double>(s.top_p);
    if (s.top_k >= 0) body["top_k"] = s.top_k;
    if (s.min_p >= 0.0f) body["min_p"] = static_cast<double>(s.min_p);
    if (s.presence_penalty != 0.0f) {
        body["presence_penalty"] = static_cast<double>(s.presence_penalty);
    }
    // llama.cpp names this sampler repeat_penalty. It remains optional so
    // OpenAI-compatible providers that do not expose the extension never see it.
    if (s.repeat_penalty > 0.0f) {
        body["repeat_penalty"] = static_cast<double>(s.repeat_penalty);
    }
    body["max_tokens"]  = s.max_tokens;
    body["stream"]      = stream;
    if (stream) body["stream_options"] = { {"include_usage", true} };

    if (!req.tools.empty()) {
        auto& tools = body["tools"] = nlohmann::json::array();
        for (auto& t : req.tools)
            tools.push_back({ {"type", "function"},
                {"function", {{"name",        t.name},
                              {"description", t.description},
                              {"parameters",  t.parameters_schema}}} });
    }
    return body;
}

// Read timeout between bytes during inference. Slow cluster hardware can take
// a long time over prompt processing before the first token arrives, so the
// default is deliberately large (the gap between streamed tokens is what this
// bounds, not total generation time).
int infer_read_timeout_seconds() {
    constexpr int kDefaultSeconds = 3600;
    if (const char* env = std::getenv("MM_INFER_READ_TIMEOUT_S")) {
        try {
            int parsed = std::stoi(env);
            if (parsed > 0) return parsed;
        } catch (...) {
        }
    }
    return kDefaultSeconds;
}

std::string client_base_url(const std::string& base_url) {
    std::string url = util::trim(base_url);
    if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
        return url;
    }
    return "http://" + url;
}

httplib::Headers auth_headers(const std::string& api_key) {
    if (api_key.empty()) return {};
    return {{"Authorization", "Bearer " + api_key}};
}

// ── Tool call delta accumulator ───────────────────────────────────────────────
httplib::Client make_client(const std::string& base_url) {
    httplib::Client cli(client_base_url(base_url));
    cli.set_connection_timeout(10);
    cli.set_read_timeout(infer_read_timeout_seconds());
    cli.set_write_timeout(30);
    return cli;
}

} // namespace

// ── Construction ──────────────────────────────────────────────────────────────
RuntimeClient::RuntimeClient(std::string base_url,
                               std::string api_key,
                               std::string chat_completions_path)
    : base_url_(std::move(base_url))
    , api_key_(std::move(api_key))
    , chat_completions_path_(std::move(chat_completions_path))
{
    auto [h, p] = util::parse_url(base_url_);
    host_ = h;
    port_ = p;
    if (chat_completions_path_.empty()) {
        chat_completions_path_ = "/v1/chat/completions";
    }
}

// ── <think>…</think> extraction ───────────────────────────────────────────────
// Buffer-search approach: find complete tags in the accumulated buffer.
// Keeps the last (tag.size()-1) bytes in buf to handle tags split across chunks.
// ── parse_sse_line ────────────────────────────────────────────────────────────
// ── complete (non-streaming) ──────────────────────────────────────────────────
Message RuntimeClient::complete(const InferenceRequest& req) {
    auto body = build_request(req, false);
    auto cli  = make_client(base_url_);

    std::string body_str;
    try {
        body_str = body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    } catch (const std::exception& e) {
        MM_ERROR("RuntimeClient::complete request serialization error: {}", e.what());
        return {};
    }

    auto res = cli.Post(chat_completions_path_,
                        auth_headers(api_key_),
                        body_str,
                        "application/json");
    if (!res) {
        MM_ERROR("RuntimeClient::complete: connection failed to {}", base_url_);
        return {};
    }
    if (res->status != 200) {
        MM_ERROR("RuntimeClient::complete HTTP {}: {}", res->status, res->body);
        return {};
    }
    std::string parse_error;
    auto parsed = inference::parse_openai_chat_completion(res->body, util::now_ms(), &parse_error);
    if (!parsed) {
        MM_ERROR("RuntimeClient::complete parse error: {}", parse_error);
        return {};
    }
    return *parsed;
}

// ── stream_complete ───────────────────────────────────────────────────────────
void RuntimeClient::stream_complete(const InferenceRequest& req,
                                     ChunkCallback chunk_cb,
                                     ErrorCallback error_cb) {
    auto body = build_request(req, true);
    auto cli  = make_client(base_url_);

    std::string  sse_buf;
    std::string  raw_body;
    inference::ThinkingExtractor thinking;
    int          total_tokens = 0;
    std::string  finish_reason;
    bool         done_sent   = false;
    bool         callback_failed = false;
    std::string  callback_error;

    // Tool call fragments accumulate here until [DONE]
    std::map<int, ToolCall> tc_acc;

    auto safe_error = [&](const std::string& msg) {
        try { error_cb(msg); }
        catch (const std::exception& e) {
            MM_ERROR("RuntimeClient::stream_complete error callback threw: {}", e.what());
        } catch (...) {
            MM_ERROR("RuntimeClient::stream_complete error callback threw unknown exception");
        }
    };

    auto safe_chunk = [&](const InferenceChunk& c) -> bool {
        if (callback_failed) return false;
        try {
            chunk_cb(c);
            return true;
        } catch (const std::exception& e) {
            callback_failed = true;
            callback_error = std::string("chunk callback exception: ") + e.what();
            return false;
        } catch (...) {
            callback_failed = true;
            callback_error = "chunk callback exception: unknown";
            return false;
        }
    };

    auto flush_done = [&]() {
        if (done_sent) return;
        done_sent = true;

        auto tail = thinking.flush();
        if (!tail.thinking.empty()) {
            InferenceChunk c; c.thinking_delta = tail.thinking;
            if (!safe_chunk(c)) return;
        }
        if (!tail.content.empty()) {
            InferenceChunk c; c.delta_content = tail.content;
            if (!safe_chunk(c)) return;
        }

        for (auto& [_, tc] : tc_acc) {
            InferenceChunk c; c.tool_call_delta = tc;
            if (!safe_chunk(c)) return;
        }
        InferenceChunk d;
        d.is_done      = true;
        d.tokens_used  = total_tokens;
        d.finish_reason = finish_reason;
        safe_chunk(d);
    };

    auto content_recv = [&](const char* data, size_t len) -> bool {
        constexpr size_t kMaxErrBody = 64 * 1024;
        if (raw_body.size() < kMaxErrBody) {
            size_t keep = std::min(len, kMaxErrBody - raw_body.size());
            raw_body.append(data, keep);
        }
        sse_buf.append(data, len);
        for (auto& payload : util::drain_sse_lines(sse_buf)) {
            if (payload == "[DONE]") { flush_done(); return true; }

            std::string parse_error;
            auto raw = inference::parse_openai_sse_delta(payload, &parse_error);
            if (!raw) {
                MM_DEBUG("SSE parse error: {}", parse_error);
                continue;
            }
            if (!raw->finish_reason.empty()) finish_reason = raw->finish_reason;
            if (raw->tokens_used > 0) total_tokens = raw->tokens_used;

            for (const auto& tc_delta : raw->tool_calls) {
                auto& acc = tc_acc[tc_delta.index];
                const auto& tc = tc_delta.call;
                if (!tc.id.empty())            acc.id            = tc.id;
                if (!tc.function_name.empty()) acc.function_name = tc.function_name;
                acc.arguments_json += tc.arguments_json;
            }

            if (!raw->content.empty()) {
                auto parts = thinking.append(raw->content);
                if (!parts.thinking.empty()) {
                    InferenceChunk c; c.thinking_delta = parts.thinking;
                    if (!safe_chunk(c)) return false;
                }
                if (!parts.content.empty()) {
                    InferenceChunk c; c.delta_content = parts.content;
                    if (!safe_chunk(c)) return false;
                }
            }
        }
        return true;
    };

    std::string body_str;
    try {
        body_str = body.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    } catch (const std::exception& e) {
        safe_error(std::string("request serialization error: ") + e.what());
        return;
    }
    auto res = cli.Post(
        chat_completions_path_,
        auth_headers(api_key_),
        body_str,
        "application/json",
        content_recv
    );

    if (callback_failed) {
        safe_error(callback_error);
    } else if (!res) {
        safe_error("Connection failed: " + base_url_);
    } else if (res->status != 200) {
        std::string err_body = res->body.empty() ? raw_body : res->body;
        err_body = util::trim(err_body);
        if (err_body.size() > 1000) {
            err_body = err_body.substr(0, 1000) + "...";
        }
        if (err_body.empty()) {
            safe_error("HTTP " + std::to_string(res->status));
        } else {
            safe_error("HTTP " + std::to_string(res->status) + ": " + err_body);
        }
    }
    flush_done(); // ensure done is always sent even on error path
}

// ── count_tokens ──────────────────────────────────────────────────────────────
int RuntimeClient::count_tokens(const std::string& text) {
    auto cli = make_client(base_url_);
    auto res = cli.Post("/tokenize",
                        nlohmann::json{{"content", text}}.dump(),
                        "application/json");
    if (!res || res->status != 200) return 0;
    try {
        auto j = nlohmann::json::parse(res->body);
        if (j.contains("tokens")) return static_cast<int>(j["tokens"].size());
    } catch (const std::exception& e) {
        MM_WARN("RuntimeClient::count_tokens parse error: {}", e.what());
    }
    return 0;
}

// ── load_model / health ───────────────────────────────────────────────────────
bool RuntimeClient::load_model(const std::string& /*model_path*/,
                                 const RuntimeSettings& /*settings*/) {
    // The runtime loads its model at startup; we just verify it's responding.
    return health_check();
}

bool RuntimeClient::is_model_loaded() const { return model_loaded_.load(); }

bool RuntimeClient::health_check() {
    auto cli = make_client(base_url_);
    cli.set_connection_timeout(3);
    cli.set_read_timeout(5);
    auto res = cli.Get("/health");
    if (!res || res->status != 200) { model_loaded_ = false; return false; }
    // llama-server and other OpenAI-compatible servers may answer with an empty
    // body or {"status":"ok"}. Treat a
    // 200 as healthy unless the body carries an explicit non-ok status.
    if (util::trim(res->body).empty()) { model_loaded_ = true; return true; }
    try {
        auto j = nlohmann::json::parse(res->body);
        bool ok = j.value("status", std::string{"ok"}) == "ok";
        model_loaded_ = ok;
        return ok;
    } catch (const std::exception& e) {
        // A 200 with an unparseable/non-JSON body still indicates the server is
        // up; be lenient rather than reporting a spurious unhealthy state.
        MM_DEBUG("RuntimeClient::health_check non-JSON 200 body: {}", e.what());
        model_loaded_ = true;
        return true;
    }
}

} // namespace mm
