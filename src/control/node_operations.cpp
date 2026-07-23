#include "control/node_operations.hpp"

#include "common/util.hpp"

#include <atomic>
#include <filesystem>
#include <system_error>

namespace mm {

namespace {

NodeServiceStatus status_from_code(const std::string& code,
                                   NodeServiceStatus fallback) {
    const std::string normalized = util::to_lower(util::trim(code));
    if (normalized == "ok") return NodeServiceStatus::Ok;
    if (normalized == "invalid_argument") return NodeServiceStatus::InvalidArgument;
    if (normalized == "unsupported_backend") return NodeServiceStatus::UnsupportedBackend;
    if (normalized == "runtime_unavailable") return NodeServiceStatus::RuntimeUnavailable;
    if (normalized == "model_not_found") return NodeServiceStatus::ModelNotFound;
    if (normalized == "capacity_exhausted") return NodeServiceStatus::CapacityExhausted;
    if (normalized == "slot_not_found") return NodeServiceStatus::SlotNotFound;
    if (normalized == "busy") return NodeServiceStatus::Busy;
    if (normalized == "conflict") return NodeServiceStatus::Conflict;
    if (normalized == "canceled" || normalized == "cancelled") {
        return NodeServiceStatus::Canceled;
    }
    if (normalized == "unavailable") return NodeServiceStatus::Unavailable;
    if (normalized == "failed") return NodeServiceStatus::Failed;
    return fallback;
}

NodeServiceStatus status_from_http(int status) {
    switch (status) {
        case 400: return NodeServiceStatus::InvalidArgument;
        case 404: return NodeServiceStatus::SlotNotFound;
        case 409: return NodeServiceStatus::Conflict;
        case 429: return NodeServiceStatus::CapacityExhausted;
        case 499: return NodeServiceStatus::Canceled;
        case 501: return NodeServiceStatus::RuntimeUnavailable;
        case 502:
        case 503:
        case 504:
        case 0:   return NodeServiceStatus::Unavailable;
        default:  return NodeServiceStatus::Failed;
    }
}

std::string response_error_message(const std::string& body) {
    if (body.empty()) return {};
    try {
        const auto value = nlohmann::json::parse(body);
        if (value.is_object()) {
            const std::string error = value.value("error", std::string{});
            if (!error.empty()) return error;
            const std::string message = value.value("message", std::string{});
            if (!message.empty()) return message;
        }
    } catch (...) {
    }
    return util::trim(body);
}

NodeServiceStatus response_error_status(int http_status,
                                        const std::string& body) {
    const NodeServiceStatus fallback = status_from_http(http_status);
    if (body.empty()) return fallback;
    try {
        const auto value = nlohmann::json::parse(body);
        if (value.is_object()) {
            return status_from_code(value.value("code", std::string{}), fallback);
        }
    } catch (...) {
    }
    return fallback;
}

} // namespace

std::string NodeOperationResult::error_message() const {
    if (body.is_object()) {
        const auto it = body.find("error");
        if (it != body.end() && it->is_string()) return it->get<std::string>();
        const auto message = body.find("message");
        if (message != body.end() && message->is_string()) {
            return message->get<std::string>();
        }
    }
    return raw_body;
}

NodeOperationResult NodeOperationResult::from_http(const HttpResponse& response) {
    NodeOperationResult result;
    result.status = response.status;
    result.raw_body = response.body;
    if (!response.body.empty()) {
        try {
            result.body = nlohmann::json::parse(response.body);
        } catch (...) {
            result.body = nlohmann::json::object();
        }
    }
    return result;
}

NodeOperationResult NodeOperationResult::success(nlohmann::json value, int code) {
    NodeOperationResult result;
    result.status = code;
    result.body = std::move(value);
    result.raw_body = result.body.dump();
    return result;
}

NodeOperationResult NodeOperationResult::failure(int code, std::string error) {
    return success(nlohmann::json{{"error", std::move(error)}}, code);
}

HttpNodeOperations::HttpNodeOperations(std::string base_url, std::string api_key)
    : base_url_(std::move(base_url)), api_key_(std::move(api_key)) {}

void HttpNodeOperations::request_shutdown() {
    shutdown_requested_ = true;
}

HttpClient HttpNodeOperations::client(int read_timeout_s, int write_timeout_s) const {
    HttpClient result(base_url_);
    result.set_bearer_token(api_key_);
    result.set_timeouts(10, read_timeout_s, write_timeout_s);
    return result;
}

NodeOperationResult HttpNodeOperations::get(const std::string& path) const {
    auto cli = client();
    return NodeOperationResult::from_http(cli.get(
        path, [this] { return shutdown_requested_.load(); }));
}

NodeOperationResult HttpNodeOperations::post(
    const std::string& path, const nlohmann::json& request) const {
    auto cli = client();
    return NodeOperationResult::from_http(cli.post(
        path, request, [this] { return shutdown_requested_.load(); }));
}

NodeOperationResult HttpNodeOperations::health() { return get("/api/node/health"); }
NodeOperationResult HttpNodeOperations::status() { return get("/api/node/status"); }
NodeOperationResult HttpNodeOperations::logs(int tail) {
    return get("/api/node/logs?tail=" + std::to_string(tail));
}
NodeOperationResult HttpNodeOperations::cancel_action() {
    return post("/api/node/actions/cancel", nlohmann::json::object());
}

std::optional<PreparedNodeModel> HttpNodeOperations::prepare_model(
    const std::string& source_path,
    const std::string& model_id,
    bool pin,
    bool force,
    std::string* error) {
    namespace fs = std::filesystem;

    struct FileEntry {
        std::string absolute_path;
        std::string relative_path;
        int64_t size = 0;
    };

    std::vector<FileEntry> files;
    int64_t total_size = 0;
    std::error_code ec;
    const fs::path root(source_path);
    if (fs::is_regular_file(root, ec)) {
        const auto size = static_cast<int64_t>(fs::file_size(root, ec));
        if (ec) {
            if (error) *error = "failed to inspect model file: " + ec.message();
            return std::nullopt;
        }
        files.push_back({root.string(), root.filename().string(), size});
        total_size += size;
    } else if (fs::is_directory(root, ec)) {
        for (auto it = fs::recursive_directory_iterator(root, ec);
             !ec && it != fs::recursive_directory_iterator();
             it.increment(ec)) {
            std::error_code file_ec;
            if (!it->is_regular_file(file_ec)) continue;
            const auto relative = fs::relative(it->path(), root, file_ec);
            const auto size = static_cast<int64_t>(it->file_size(file_ec));
            if (file_ec) continue;
            files.push_back({it->path().string(), relative.generic_string(), size});
            total_size += size;
        }
    } else {
        if (error) *error = "model path is neither a file nor a directory: " + source_path;
        return std::nullopt;
    }

    if (files.empty()) {
        if (error) *error = "no files found for model: " + source_path;
        return std::nullopt;
    }

    auto cli = client(3600, 3600);
    if (!force) {
        const auto query = NodeOperationResult::from_http(
            cli.get("/api/node/models/local?id=" + model_id,
                    [this] { return shutdown_requested_.load(); }));
        if (query.ok() && query.body.value("present", false) &&
            query.body.value("size_bytes", static_cast<int64_t>(-1)) == total_size) {
            const std::string load_path = query.body.value("load_path", std::string{});
            if (!load_path.empty()) return PreparedNodeModel{load_path, model_id};
        }
    }

    std::string load_path;
    for (const auto& file : files) {
        const std::vector<std::pair<std::string, std::string>> headers = {
            {"X-MM-Model-Id", model_id},
            {"X-MM-Rel-Path", file.relative_path},
            {"X-MM-Size", std::to_string(file.size)},
            {"X-MM-Pin", pin ? "true" : "false"},
        };
        const auto response = NodeOperationResult::from_http(cli.post_file(
            "/api/node/models/receive", file.absolute_path, headers,
            "application/octet-stream",
            [this] { return shutdown_requested_.load(); }));
        if (!response.ok()) {
            if (error) {
                *error = "receive failed for " + file.relative_path + " (status " +
                         std::to_string(response.status) + "): " + response.error_message();
            }
            return std::nullopt;
        }
        load_path = response.body.value("load_path", load_path);
        if (files.size() == 1) {
            load_path = response.body.value("stored_path", load_path);
        }
    }

    if (load_path.empty()) {
        if (error) *error = "node did not report a load path after transfer";
        return std::nullopt;
    }
    return PreparedNodeModel{load_path, model_id};
}

NodeOperationResult HttpNodeOperations::load_model(const nlohmann::json& request) {
    return post("/api/node/load-model", request);
}
NodeOperationResult HttpNodeOperations::unload_model(const nlohmann::json& request) {
    return post("/api/node/unload-model", request);
}
NodeOperationResult HttpNodeOperations::detach_agent(const nlohmann::json& request) {
    return post("/api/node/detach-agent", request);
}
NodeOperationResult HttpNodeOperations::suspend_slot(const nlohmann::json& request) {
    return post("/api/node/suspend-slot", request);
}
NodeOperationResult HttpNodeOperations::restore_slot(const nlohmann::json& request) {
    return post("/api/node/restore-slot", request);
}
NodeOperationResult HttpNodeOperations::llama_runtime() {
    return get("/api/node/runtime/llama");
}
NodeOperationResult HttpNodeOperations::llama_provision(const nlohmann::json& request) {
    auto cli = client(3600, 3600);
    return NodeOperationResult::from_http(
        cli.post("/api/node/runtime/llama/provision", request,
                 [this] { return shutdown_requested_.load(); }));
}
NodeOperationResult HttpNodeOperations::llama_update(const nlohmann::json& request) {
    auto body = request;
    body["update"] = true;
    auto cli = client(3600, 3600);
    return NodeOperationResult::from_http(
        cli.post("/api/node/runtime/llama/provision", body,
                 [this] { return shutdown_requested_.load(); }));
}
NodeOperationResult HttpNodeOperations::llama_check_update(
    const nlohmann::json& request) {
    auto cli = client(300, 300);
    return NodeOperationResult::from_http(
        cli.post("/api/node/runtime/llama/check-update", request,
                 [this] { return shutdown_requested_.load(); }));
}
NodeOperationResult HttpNodeOperations::llama_switch(const nlohmann::json& request) {
    auto cli = client(3600, 3600);
    return NodeOperationResult::from_http(
        cli.post("/api/node/runtime/llama/switch", request,
                 [this] { return shutdown_requested_.load(); }));
}
NodeOperationResult HttpNodeOperations::llama_diagnose() {
    return post("/api/node/runtime/llama/diagnose", nlohmann::json::object());
}
NodeOperationResult HttpNodeOperations::llama_recover(const nlohmann::json& request) {
    auto cli = client(3600, 3600);
    return NodeOperationResult::from_http(
        cli.post("/api/node/runtime/llama/recover", request,
                 [this] { return shutdown_requested_.load(); }));
}

NodeInferResult HttpNodeOperations::infer(
    const NodeInferRequest& request,
    InferenceChunkCallback chunk_cb) {
    NodeInferResult result;
    result.slot_id = request.slot_id;
    result.message.role = MessageRole::Assistant;
    result.message.timestamp_ms = util::now_ms();

    nlohmann::json body;
    try {
        body = nlohmann::json(request.request);
        if (!request.slot_id.empty()) body["slot_id"] = request.slot_id;
    } catch (const std::exception& exception) {
        result.status = NodeServiceStatus::InvalidArgument;
        result.error = std::string("inference request serialization failed: ") +
                       exception.what();
        return result;
    }

    std::atomic<bool> consumer_canceled{false};
    bool terminal_seen = false;
    bool callback_failed = false;
    bool parse_failed = false;
    std::string callback_error;
    std::string parse_error;
    const auto caller_cancel = request.cancel_requested;
    const auto canceled = [&] {
        if (shutdown_requested_.load()) return true;
        if (consumer_canceled.load()) return true;
        if (!caller_cancel) return false;
        try { return caller_cancel(); } catch (...) { return true; }
    };

    auto deliver = [&](InferenceChunk chunk) -> bool {
        if (!chunk.delta_content.empty()) {
            result.message.content += chunk.delta_content;
        }
        if (!chunk.thinking_delta.empty()) {
            result.message.thinking_text += chunk.thinking_delta;
        }
        if (chunk.tool_call_delta) {
            result.message.tool_calls.push_back(*chunk.tool_call_delta);
        }
        if (!chunk_cb) return true;
        try {
            if (chunk_cb(chunk)) return true;
            consumer_canceled = true;
            return false;
        } catch (const std::exception& exception) {
            callback_failed = true;
            callback_error = std::string("inference chunk callback failed: ") +
                             exception.what();
        } catch (...) {
            callback_failed = true;
            callback_error = "inference chunk callback failed";
        }
        consumer_canceled = true;
        return false;
    };

    int http_status = 0;
    std::string response_body;
    auto cli = client(3600, 3600);
    const bool transport_ok = cli.stream_post(
        "/api/node/infer",
        body,
        [&](const std::string& payload) -> bool {
            if (terminal_seen) return true;
            if (canceled()) return false;
            if (payload == "[DONE]") return true;

            nlohmann::json event;
            try {
                event = nlohmann::json::parse(payload);
            } catch (const std::exception& exception) {
                parse_failed = true;
                parse_error = std::string("invalid node inference event: ") +
                              exception.what();
                consumer_canceled = true;
                return false;
            }

            const std::string type = event.value("type", std::string{});
            if (type == "done") {
                terminal_seen = true;
                result.status = NodeServiceStatus::Ok;
                result.tokens_used = event.value("tokens_used", 0);
                result.finish_reason = event.value("finish_reason", std::string{});
                result.message.token_count = result.tokens_used;
                return true;
            }
            if (type == "error") {
                terminal_seen = true;
                result.status = status_from_code(
                    event.value("code", std::string{}),
                    NodeServiceStatus::Failed);
                result.error = event.value("message", std::string{});
                if (result.error.empty()) {
                    result.error = event.value("error", std::string{"inference failed"});
                }
                return true;
            }

            InferenceChunk chunk;
            if (type == "delta" || type == "token") {
                chunk.delta_content = event.value("content", std::string{});
            } else if (type == "thinking") {
                chunk.thinking_delta = event.value("content", std::string{});
            } else if (type == "tool_call") {
                ToolCall tool_call;
                tool_call.id = event.value("id", std::string{});
                tool_call.function_name = event.value("name", std::string{});
                tool_call.arguments_json = event.value("arguments", std::string{});
                chunk.tool_call_delta = std::move(tool_call);
            } else if (type == "tool_result") {
                chunk.tool_result_json = event.value("content", std::string{});
            } else {
                // Ignore additive event types from newer nodes.
                return true;
            }
            return deliver(std::move(chunk));
        },
        &http_status,
        &response_body,
        canceled);

    if (callback_failed) {
        result.status = NodeServiceStatus::Failed;
        result.error = std::move(callback_error);
        return result;
    }
    if (parse_failed) {
        result.status = NodeServiceStatus::Failed;
        result.error = std::move(parse_error);
        return result;
    }
    if (canceled() && !terminal_seen) {
        result.status = NodeServiceStatus::Canceled;
        result.error = "inference canceled";
        return result;
    }
    if (terminal_seen) return result;

    if (!transport_ok) {
        result.status = response_error_status(http_status, response_body);
        result.error = response_error_message(response_body);
        if (result.error.empty()) {
            result.error = http_status > 0
                ? "node inference failed with HTTP status " +
                      std::to_string(http_status)
                : "node inference transport failed";
        }
        return result;
    }

    result.status = NodeServiceStatus::Failed;
    result.error = "node inference ended without a terminal result";
    return result;
}

bool HttpNodeOperations::stream_infer(const nlohmann::json& request,
                                      SseLineCallback line_cb,
                                      int* out_status,
                                      std::string* out_body) {
    if (!line_cb) {
        if (out_status) *out_status = 400;
        if (out_body) *out_body = R"({"error":"stream callback required"})";
        return false;
    }
    auto cli = client(3600, 3600);
    bool terminal_seen = false;
    bool sentinel_seen = false;
    auto guarded_callback =
        [callback = std::move(line_cb),
         &terminal_seen,
         &sentinel_seen](const std::string& line) mutable {
            if (sentinel_seen) return true;
            if (line == "[DONE]") {
                sentinel_seen = true;
                try {
                    return callback && callback(line);
                } catch (...) {
                    return false;
                }
            }

            // Older node versions could report an error and then flush a done
            // event, or leak another data chunk after a terminal result. Keep
            // the transport sentinel, but expose only the first typed terminal
            // event and nothing after it.
            if (terminal_seen) return true;

            bool terminal = false;
            try {
                const auto event = nlohmann::json::parse(line);
                const auto type = event.value("type", std::string{});
                terminal = type == "done" || type == "error";
            } catch (...) {
                // Preserve malformed pre-terminal payloads for compatibility;
                // the consumer owns validation of non-contract extensions.
            }

            bool keep_going = false;
            try {
                keep_going = callback && callback(line);
            } catch (...) {
                keep_going = false;
            }
            if (!keep_going) return false;
            if (terminal) terminal_seen = true;
            return true;
        };
    return cli.stream_post(
        "/api/node/infer", request, std::move(guarded_callback), out_status,
        out_body, [this] { return shutdown_requested_.load(); });
}

NodeOperationResult HttpNodeOperations::pair_request(const nlohmann::json& request) {
    // Pairing endpoints are intentionally unauthenticated; use a fresh client.
    HttpClient unauthenticated(base_url_);
    return NodeOperationResult::from_http(
        unauthenticated.post("/api/node/pair-request", request,
                             [this] { return shutdown_requested_.load(); }));
}
NodeOperationResult HttpNodeOperations::pair_complete(const nlohmann::json& request) {
    HttpClient unauthenticated(base_url_);
    return NodeOperationResult::from_http(
        unauthenticated.post("/api/node/pair-complete", request,
                             [this] { return shutdown_requested_.load(); }));
}

} // namespace mm
