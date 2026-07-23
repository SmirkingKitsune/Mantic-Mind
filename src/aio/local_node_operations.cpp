#include "aio/local_node_operations.hpp"

#include "common/util.hpp"
#include "node/node_service.hpp"

#include <algorithm>
#include <exception>
#include <mutex>
#include <utility>

namespace mm {

namespace {

const char* service_status_code(NodeServiceStatus status) {
    switch (status) {
        case NodeServiceStatus::Ok:                 return "ok";
        case NodeServiceStatus::InvalidArgument:    return "invalid_argument";
        case NodeServiceStatus::UnsupportedBackend: return "unsupported_backend";
        case NodeServiceStatus::RuntimeUnavailable: return "runtime_unavailable";
        case NodeServiceStatus::ModelNotFound:       return "model_not_found";
        case NodeServiceStatus::CapacityExhausted:   return "capacity_exhausted";
        case NodeServiceStatus::SlotNotFound:        return "slot_not_found";
        case NodeServiceStatus::Busy:                return "busy";
        case NodeServiceStatus::Conflict:            return "conflict";
        case NodeServiceStatus::Canceled:            return "canceled";
        case NodeServiceStatus::Unavailable:         return "unavailable";
        case NodeServiceStatus::Failed:              return "failed";
    }
    return "failed";
}

int service_http_status(NodeServiceStatus status) {
    switch (status) {
        case NodeServiceStatus::Ok:                 return 200;
        case NodeServiceStatus::InvalidArgument:
        case NodeServiceStatus::UnsupportedBackend: return 400;
        case NodeServiceStatus::ModelNotFound:
        case NodeServiceStatus::SlotNotFound:        return 404;
        case NodeServiceStatus::Busy:
        case NodeServiceStatus::Conflict:
        case NodeServiceStatus::Canceled:            return 409;
        case NodeServiceStatus::RuntimeUnavailable:
        case NodeServiceStatus::CapacityExhausted:
        case NodeServiceStatus::Unavailable:         return 503;
        case NodeServiceStatus::Failed:              return 500;
    }
    return 500;
}

int runtime_http_status(NodeServiceStatus status) {
    if (status == NodeServiceStatus::Unavailable) return 501;
    return service_http_status(status);
}

NodeOperationResult service_failure(NodeServiceStatus status,
                                    const std::string& error) {
    nlohmann::json body{
        {"error", error.empty() ? service_status_code(status) : error},
        {"code", service_status_code(status)},
    };
    return NodeOperationResult::success(std::move(body), service_http_status(status));
}

NodeOperationResult exception_failure(const std::exception& exception) {
    return NodeOperationResult::failure(400, exception.what());
}

NodeOperationResult runtime_result(NodeRuntimeOperationResult result) {
    nlohmann::json body{{"llama_runtime", result.runtime}};
    if (!result.error.empty()) body["error"] = result.error;
    body["code"] = service_status_code(result.status);
    return NodeOperationResult::success(std::move(body),
                                        runtime_http_status(result.status));
}

int slot_http_status(SlotOperationStatus status) {
    switch (status) {
        case SlotOperationStatus::Ok:       return 200;
        case SlotOperationStatus::NotFound: return 404;
        case SlotOperationStatus::Busy:     return 409;
        case SlotOperationStatus::Failed:   return 500;
    }
    return 500;
}

NodeLoadModelRequest parse_load_request(const nlohmann::json& request) {
    NodeLoadModelRequest parsed;
    parsed.model_path = request.value("model_path", std::string{});
    parsed.mmproj_path = request.value("mmproj_path", std::string{});
    parsed.vision_enabled = request.value(
        "vision_enabled", !parsed.mmproj_path.empty());
    parsed.agent_id = request.value("agent_id", std::string{});
    parsed.backend = request.value("backend", std::string{"llama-cpp"});
    parsed.model_id = request.value("model_id", std::string{});
    parsed.mmproj_model_id = request.value("mmproj_model_id", std::string{});
    parsed.pin = request.value("pin", false);
    if (request.contains("runtime_settings")) {
        parsed.runtime_settings = request.at("runtime_settings").get<RuntimeSettings>();
    }
    return parsed;
}

nlohmann::json status_json(const NodeStatusSnapshot& snapshot) {
    nlohmann::json managed = nlohmann::json::array();
    for (const auto& model : snapshot.managed_models) {
        managed.push_back({
            {"id", model.id},
            {"size_bytes", model.size_bytes},
            {"pinned", model.pinned},
            {"last_used_ms", model.last_used_ms},
        });
    }

    return {
        {"node_id", snapshot.node_id},
        {"hostname", snapshot.hostname},
        {"registered", snapshot.registered},
        {"last_control_contact_ms", snapshot.last_control_contact_ms},
        {"slots", snapshot.slots},
        {"managed_models", std::move(managed)},
        {"model_cache_free_bytes", snapshot.model_cache_free_bytes},
        {"disk_free_mb", snapshot.disk_free_mb},
        {"health", snapshot.health},
        {"capabilities", snapshot.capabilities},
        {"max_slots", snapshot.max_slots},
        {"slot_in_use", snapshot.slot_in_use},
        {"slot_available", snapshot.slot_available},
        {"slot_ready", snapshot.slot_ready},
        {"slot_loading", snapshot.slot_loading},
        {"slot_suspending", snapshot.slot_suspending},
        {"slot_suspended", snapshot.slot_suspended},
        {"slot_error", snapshot.slot_error},
        {"llama_server_path", snapshot.llama_server_path},
        {"llama_runtime", snapshot.llama_runtime},
        {"action_progress", snapshot.action_progress},
        {"loaded_model", snapshot.loaded_model},
        {"active_agent", snapshot.active_agent},
        {"last_error", snapshot.last_error},
    };
}

struct StreamState {
    mutable std::mutex mutex;
    bool terminal_emitted = false;
    bool consumer_canceled = false;
};

bool emit_stream_payload(StreamState& state,
                         const NodeOperations::SseLineCallback& callback,
                         const nlohmann::json& payload,
                         bool terminal) {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.consumer_canceled || state.terminal_emitted) return false;

    bool keep_going = false;
    try {
        keep_going = callback && callback(payload.dump());
    } catch (...) {
        keep_going = false;
    }
    if (terminal) state.terminal_emitted = true;
    if (!keep_going) state.consumer_canceled = true;
    return keep_going;
}

bool stream_is_terminal(const StreamState& state) {
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.terminal_emitted;
}

bool stream_was_canceled(const StreamState& state) {
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.consumer_canceled;
}

} // namespace

LocalNodeOperations::LocalNodeOperations(NodeService& service,
                                         std::string models_dir,
                                         std::string runtime_network_policy)
    : service_(service)
    , models_dir_(std::move(models_dir))
    , runtime_network_policy_(util::to_lower(
          util::trim(std::move(runtime_network_policy)))) {
    if (runtime_network_policy_ != "auto" &&
        runtime_network_policy_ != "offline" &&
        runtime_network_policy_ != "prompt") {
        runtime_network_policy_ = "prompt";
    }
}

void LocalNodeOperations::set_mutation_callback(std::function<void()> callback) {
    mutation_callback_ = std::move(callback);
}

void LocalNodeOperations::request_shutdown() {
    shutdown_requested_ = true;
}

NodeOperationResult LocalNodeOperations::runtime_canceled() const {
    return NodeOperationResult::failure(
        409, "llama runtime operation canceled because the embedded node is shutting down");
}

void LocalNodeOperations::notify_mutation() const {
    if (mutation_callback_) mutation_callback_();
}

NodeOperationResult LocalNodeOperations::health() {
    try {
        nlohmann::json body = service_.snapshot().health;
        body["status"] = "ok";
        return NodeOperationResult::success(std::move(body));
    } catch (const std::exception& exception) {
        return NodeOperationResult::failure(500, exception.what());
    }
}

NodeOperationResult LocalNodeOperations::status() {
    try {
        return NodeOperationResult::success(status_json(service_.snapshot()));
    } catch (const std::exception& exception) {
        return NodeOperationResult::failure(500, exception.what());
    }
}

NodeOperationResult LocalNodeOperations::logs(int tail) {
    try {
        return NodeOperationResult::success({
            {"lines", service_.runtime_logs(std::clamp(tail, 1, 5000))},
        });
    } catch (const std::exception& exception) {
        return NodeOperationResult::failure(500, exception.what());
    }
}

NodeOperationResult LocalNodeOperations::cancel_action() {
    try {
        if (!service_.cancel_action()) {
            return NodeOperationResult::failure(
                409, "no cancelable action is active");
        }
        notify_mutation();
        return NodeOperationResult::success({{"cancel_requested", true}});
    } catch (const std::exception& exception) {
        return NodeOperationResult::failure(500, exception.what());
    }
}

std::optional<PreparedNodeModel> LocalNodeOperations::prepare_model(
    const std::string& source_path,
    const std::string& model_id,
    bool pin,
    bool force,
    std::string* error) {
    (void)force;
    try {
        const auto prepared = service_.prepare_local_model({
            source_path,
            models_dir_,
            model_id,
            pin,
        });
        if (!prepared.ok()) {
            if (error) *error = prepared.error;
            return std::nullopt;
        }
        if (error) error->clear();
        return PreparedNodeModel{prepared.load_path, prepared.model_id};
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return std::nullopt;
    }
}

NodeOperationResult LocalNodeOperations::load_model(
    const nlohmann::json& request) {
    try {
        const auto result = service_.load_model(parse_load_request(request));
        if (!result.ok()) return service_failure(result.status, result.error);
        notify_mutation();
        return NodeOperationResult::success({
            {"status", "loaded"},
            {"slot_id", result.slot_id},
            {"effective_ctx_size", result.effective_ctx_size},
            {"model_path", result.model_path},
        });
    } catch (const std::exception& exception) {
        return exception_failure(exception);
    }
}

NodeOperationResult LocalNodeOperations::unload_model(
    const nlohmann::json& request) {
    try {
        NodeUnloadModelRequest parsed;
        parsed.slot_id = request.value("slot_id", std::string{});
        parsed.force = request.value("force", false);
        const auto result = service_.unload_model(parsed);
        if (!result.ok()) {
            return NodeOperationResult::failure(
                slot_http_status(result.status), result.message);
        }
        notify_mutation();
        return NodeOperationResult::success({{"status", "unloaded"}});
    } catch (const std::exception& exception) {
        return exception_failure(exception);
    }
}

NodeOperationResult LocalNodeOperations::detach_agent(
    const nlohmann::json& request) {
    try {
        const auto result = service_.detach_agent({
            request.value("slot_id", std::string{}),
            request.value("agent_id", std::string{}),
        });
        if (!result.ok()) {
            return NodeOperationResult::failure(
                slot_http_status(result.status), result.message);
        }
        notify_mutation();
        return NodeOperationResult::success({
            {"status", "detached"},
            {"remaining_agents", result.remaining_agents},
            {"unloaded", result.unloaded},
        });
    } catch (const std::exception& exception) {
        return exception_failure(exception);
    }
}

NodeOperationResult LocalNodeOperations::suspend_slot(
    const nlohmann::json& request) {
    try {
        const auto result = service_.suspend_slot({
            request.value("slot_id", std::string{}),
        });
        if (!result.ok()) {
            return NodeOperationResult::failure(
                slot_http_status(result.status), result.message);
        }
        notify_mutation();
        return NodeOperationResult::success({
            {"status", "suspended"},
            {"kv_cache_path", result.kv_cache_path},
        });
    } catch (const std::exception& exception) {
        return exception_failure(exception);
    }
}

NodeOperationResult LocalNodeOperations::restore_slot(
    const nlohmann::json& request) {
    try {
        const auto load = parse_load_request(request);
        NodeRestoreModelRequest parsed;
        static_cast<NodeLoadModelRequest&>(parsed) = load;
        parsed.kv_cache_path = request.value("kv_cache_path", std::string{});
        const auto result = service_.restore_slot(parsed);
        if (!result.ok()) return service_failure(result.status, result.error);
        notify_mutation();
        return NodeOperationResult::success({
            {"status", "restored"},
            {"slot_id", result.slot_id},
            {"effective_ctx_size", result.effective_ctx_size},
            {"model_path", result.model_path},
        });
    } catch (const std::exception& exception) {
        return exception_failure(exception);
    }
}

NodeOperationResult LocalNodeOperations::llama_runtime() {
    try {
        return NodeOperationResult::success({
            {"llama_runtime", service_.snapshot().llama_runtime},
        });
    } catch (const std::exception& exception) {
        return NodeOperationResult::failure(500, exception.what());
    }
}

NodeOperationResult LocalNodeOperations::require_network_consent(
    const nlohmann::json* request,
    const char* operation) const {
    if (runtime_network_policy_ == "auto") {
        return NodeOperationResult::success();
    }

    nlohmann::json body{
        {"error", runtime_network_policy_ == "offline"
             ? std::string(operation) + " is disabled by offline runtime policy"
             : std::string(operation) + " requires explicit network consent"},
        {"runtime_network_policy", runtime_network_policy_},
        {"requires_network_consent", runtime_network_policy_ == "prompt"},
    };
    if (runtime_network_policy_ == "offline") {
        return NodeOperationResult::success(std::move(body), 403);
    }

    if (request && request->value("allow_network", false)) {
        return NodeOperationResult::success();
    }
    return NodeOperationResult::success(std::move(body), 403);
}

NodeOperationResult LocalNodeOperations::llama_provision(
    const nlohmann::json& request) {
    try {
        std::unique_lock<std::mutex> operation_lock(runtime_operation_mutex_);
        if (shutdown_requested_) return runtime_canceled();
        auto consent = require_network_consent(&request, "llama runtime provisioning");
        if (!consent.ok()) return consent;

        const bool update = request.value("update", false);
        const std::string accelerator =
            request.value("accelerator", std::string{});
        if (!update && !accelerator.empty()) {
            return NodeOperationResult::failure(
                400, "accelerator is only valid with update=true");
        }
        auto result = runtime_result(update
            ? service_.update_llama_runtime(accelerator)
            : service_.provision_llama_runtime());
        if (shutdown_requested_) return runtime_canceled();
        notify_mutation();
        return result;
    } catch (const std::exception& exception) {
        return exception_failure(exception);
    }
}

NodeOperationResult LocalNodeOperations::llama_update(
    const nlohmann::json& request) {
    try {
        std::unique_lock<std::mutex> operation_lock(runtime_operation_mutex_);
        if (shutdown_requested_) return runtime_canceled();
        auto consent = require_network_consent(&request, "llama runtime update");
        if (!consent.ok()) return consent;
        auto result = runtime_result(service_.update_llama_runtime(
            request.value("accelerator", std::string{})));
        if (shutdown_requested_) return runtime_canceled();
        notify_mutation();
        return result;
    } catch (const std::exception& exception) {
        return exception_failure(exception);
    }
}

NodeOperationResult LocalNodeOperations::llama_check_update(
    const nlohmann::json& request) {
    try {
        std::unique_lock<std::mutex> operation_lock(runtime_operation_mutex_);
        if (shutdown_requested_) return runtime_canceled();
        auto consent = require_network_consent(&request, "llama runtime update check");
        if (!consent.ok()) return consent;
        auto result = runtime_result(service_.check_llama_runtime_update());
        if (shutdown_requested_) return runtime_canceled();
        notify_mutation();
        return result;
    } catch (const std::exception& exception) {
        return NodeOperationResult::failure(500, exception.what());
    }
}

NodeOperationResult LocalNodeOperations::llama_switch(
    const nlohmann::json& request) {
    try {
        std::unique_lock<std::mutex> operation_lock(runtime_operation_mutex_);
        if (shutdown_requested_) return runtime_canceled();
        auto consent = require_network_consent(&request, "llama runtime switching");
        if (!consent.ok()) return consent;
        auto result = runtime_result(service_.switch_llama_runtime(
            request.value("variant", std::string{})));
        if (shutdown_requested_) return runtime_canceled();
        notify_mutation();
        return result;
    } catch (const std::exception& exception) {
        return exception_failure(exception);
    }
}

NodeOperationResult LocalNodeOperations::llama_diagnose() {
    try {
        std::unique_lock<std::mutex> operation_lock(runtime_operation_mutex_);
        if (shutdown_requested_) return runtime_canceled();
        // Diagnostics inspect local state and do not grant network access.
        auto result = runtime_result(service_.diagnose_llama_runtime());
        if (shutdown_requested_) return runtime_canceled();
        notify_mutation();
        return result;
    } catch (const std::exception& exception) {
        return NodeOperationResult::failure(500, exception.what());
    }
}

NodeOperationResult LocalNodeOperations::llama_recover(
    const nlohmann::json& request) {
    try {
        std::unique_lock<std::mutex> operation_lock(runtime_operation_mutex_);
        if (shutdown_requested_) return runtime_canceled();
        const std::string action = util::to_lower(util::trim(
            request.value("action", std::string{})));
        // Recovery retries may enter the managed release/source installer, so
        // every recovery action stays behind the network-consent boundary.
        auto consent = require_network_consent(&request, "llama runtime recovery");
        if (!consent.ok()) return consent;
        auto result = runtime_result(service_.recover_llama_runtime(
            action, request.value("variant", std::string{})));
        if (shutdown_requested_) return runtime_canceled();
        notify_mutation();
        return result;
    } catch (const std::exception& exception) {
        return exception_failure(exception);
    }
}

NodeInferResult LocalNodeOperations::infer(
    const NodeInferRequest& request,
    InferenceChunkCallback chunk_cb) {
    std::atomic<bool> consumer_canceled{false};
    bool callback_failed = false;
    std::string callback_error;
    const auto caller_cancel = request.cancel_requested;

    NodeInferRequest local_request = request;
    local_request.cancel_requested = [this, caller_cancel, &consumer_canceled] {
        return shutdown_requested_.load() || consumer_canceled.load() ||
               (caller_cancel && caller_cancel());
    };

    NodeInferResult result;
    try {
        result = service_.infer(
            local_request,
            [&](const InferenceChunk& service_chunk) {
                if (consumer_canceled.load()) return;

                // NodeService's runtime callback may combine final content and
                // accounting in one chunk. Preserve its data, but keep the
                // terminal outcome exclusively in NodeInferResult.
                const bool has_incremental_data =
                    !service_chunk.delta_content.empty() ||
                    !service_chunk.thinking_delta.empty() ||
                    service_chunk.tool_call_delta.has_value() ||
                    !service_chunk.tool_result_json.empty();
                if (!has_incremental_data || !chunk_cb) return;

                InferenceChunk chunk = service_chunk;
                chunk.is_done = false;
                chunk.tokens_used = 0;
                chunk.finish_reason.clear();
                try {
                    if (!chunk_cb(chunk)) consumer_canceled = true;
                } catch (const std::exception& exception) {
                    callback_failed = true;
                    callback_error = std::string("inference chunk callback failed: ") +
                                     exception.what();
                    consumer_canceled = true;
                } catch (...) {
                    callback_failed = true;
                    callback_error = "inference chunk callback failed";
                    consumer_canceled = true;
                }
            });
    } catch (const std::exception& exception) {
        result.status = NodeServiceStatus::Failed;
        result.error = std::string("inference exception: ") + exception.what();
        return result;
    } catch (...) {
        result.status = NodeServiceStatus::Failed;
        result.error = "inference exception: unknown";
        return result;
    }

    if (callback_failed) {
        result.status = NodeServiceStatus::Failed;
        result.error = std::move(callback_error);
    } else if (consumer_canceled.load() && result.status != NodeServiceStatus::Canceled) {
        result.status = NodeServiceStatus::Canceled;
        result.error = "inference canceled";
    }
    return result;
}

bool LocalNodeOperations::stream_infer(const nlohmann::json& request,
                                       SseLineCallback line_cb,
                                       int* out_status,
                                       std::string* out_body) {
    if (out_status) *out_status = 0;
    if (out_body) out_body->clear();
    if (!line_cb) {
        if (out_status) *out_status = 400;
        if (out_body) *out_body = R"({"error":"stream callback required"})";
        return false;
    }

    StreamState stream;
    NodeInferRequest parsed;
    try {
        parsed.request = request.get<InferenceRequest>();
        parsed.slot_id = request.value("slot_id", std::string{});
        parsed.cancel_requested = [this, &stream] {
            return shutdown_requested_.load() || stream_was_canceled(stream);
        };
    } catch (const std::exception& exception) {
        if (out_status) *out_status = 400;
        if (out_body) {
            *out_body = nlohmann::json{{"error", exception.what()}}.dump();
        }
        return false;
    }

    NodeInferResult result;
    try {
        result = infer(
            parsed,
            [&](const InferenceChunk& chunk) -> bool {
                if (stream_was_canceled(stream) || stream_is_terminal(stream)) {
                    return false;
                }

                if (!chunk.thinking_delta.empty() &&
                    !emit_stream_payload(stream, line_cb,
                        {{"type", "thinking"},
                         {"content", chunk.thinking_delta}}, false)) {
                    return false;
                }
                if (!chunk.delta_content.empty() &&
                    !emit_stream_payload(stream, line_cb,
                        {{"type", "delta"},
                         {"content", chunk.delta_content}}, false)) {
                    return false;
                }
                if (chunk.tool_call_delta) {
                    const auto& tool = *chunk.tool_call_delta;
                    if (!emit_stream_payload(stream, line_cb,
                            {{"type", "tool_call"},
                             {"id", tool.id},
                             {"name", tool.function_name},
                             {"arguments", tool.arguments_json}}, false)) {
                        return false;
                    }
                }
                if (!chunk.tool_result_json.empty() &&
                    !emit_stream_payload(stream, line_cb,
                        {{"type", "tool_result"},
                         {"content", chunk.tool_result_json}}, false)) {
                    return false;
                }
                if (chunk.is_done) {
                    emit_stream_payload(stream, line_cb,
                        {{"type", "done"},
                         {"tokens_used", chunk.tokens_used},
                         {"finish_reason", chunk.finish_reason}}, true);
                }
                return !stream_was_canceled(stream);
            });
    } catch (const std::exception& exception) {
        if (!stream_was_canceled(stream)) {
            emit_stream_payload(stream, line_cb,
                {{"type", "error"}, {"message", exception.what()}}, true);
        }
        if (out_status) *out_status = 500;
        if (out_body) {
            *out_body = nlohmann::json{{"error", exception.what()}}.dump();
        }
        return false;
    }

    if (stream_was_canceled(stream)) {
        if (out_body) *out_body = R"({"error":"stream canceled by consumer"})";
        return false;
    }

    if (!stream_is_terminal(stream)) {
        if (result.ok()) {
            emit_stream_payload(stream, line_cb,
                {{"type", "done"},
                 {"tokens_used", result.tokens_used},
                 {"finish_reason", result.finish_reason}}, true);
        } else {
            emit_stream_payload(stream, line_cb,
                {{"type", "error"}, {"message", result.error}}, true);
        }
    }

    if (stream_was_canceled(stream)) {
        if (out_body) *out_body = R"({"error":"stream canceled by consumer"})";
        return false;
    }

    bool sentinel_accepted = true;
    try {
        sentinel_accepted = line_cb("[DONE]");
    } catch (...) {
        sentinel_accepted = false;
    }
    if (!sentinel_accepted) {
        if (out_body) *out_body = R"({"error":"stream canceled by consumer"})";
        return false;
    }

    const int status = service_http_status(result.status);
    if (out_status) *out_status = status;
    if (!result.ok() && out_body) {
        *out_body = nlohmann::json{
            {"error", result.error},
            {"code", service_status_code(result.status)},
        }.dump();
    }
    return result.ok();
}

NodeOperationResult LocalNodeOperations::pair_request(
    const nlohmann::json& request) {
    (void)request;
    return NodeOperationResult::failure(409, "embedded node cannot be paired");
}

NodeOperationResult LocalNodeOperations::pair_complete(
    const nlohmann::json& request) {
    (void)request;
    return NodeOperationResult::failure(409, "embedded node cannot be paired");
}

} // namespace mm
