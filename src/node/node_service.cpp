#include "node/node_service.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"
#include "node/model_store.hpp"
#include "node/node_state.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <system_error>
#include <utility>

namespace mm {

namespace {

namespace fs = std::filesystem;

bool llama_runtime_ready(const LlamaRuntimeStatus& runtime) {
    return (runtime.status == "resolved" || runtime.status == "ready") &&
           !runtime.executable_path.empty();
}

std::string clean_runtime_path(std::string path) {
    path = util::trim(path);
    if (path.size() >= 2 &&
        ((path.front() == '"' && path.back() == '"') ||
         (path.front() == '\'' && path.back() == '\''))) {
        path = util::trim(path.substr(1, path.size() - 2));
    }
    path.erase(std::remove_if(path.begin(), path.end(), [](unsigned char ch) {
        return ch == '\r' || ch == '\n' || ch == '\t';
    }), path.end());
    return path;
}

NodeServiceStatus classify_slot_failure(const std::string& error) {
    const std::string lower = util::to_lower(error);
    if (lower.find("startup canceled") != std::string::npos ||
        lower.find("shutting down") != std::string::npos ||
        lower.find("is unloading") != std::string::npos) {
        return NodeServiceStatus::Canceled;
    }
    if (lower.find("max slots") != std::string::npos ||
        lower.find("no available ports") != std::string::npos) {
        return NodeServiceStatus::CapacityExhausted;
    }
    return NodeServiceStatus::Failed;
}

SlotOperationResult invalid_slot_operation(std::string message) {
    return {SlotOperationStatus::Failed, std::move(message), {}};
}

DetachResult invalid_detach(std::string message) {
    return {SlotOperationStatus::Failed, std::move(message), 0, false};
}

void notify_inference_error(
    const std::function<void(const std::string&)>& callback,
    const std::string& error) noexcept {
    if (!callback) return;
    try {
        callback(error);
    } catch (const std::exception& exception) {
        MM_ERROR("NodeService inference error callback threw: {}", exception.what());
    } catch (...) {
        MM_ERROR("NodeService inference error callback threw an unknown exception");
    }
}

} // namespace

NodeService::NodeService(NodeState& state,
                         SlotManager& slot_manager,
                         ModelStore* model_store)
    : state_(state)
    , slot_manager_(slot_manager)
    , model_store_(model_store) {}

void NodeService::set_model_store(ModelStore* model_store) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    model_store_ = model_store;
}

NodeStatusSnapshot NodeService::snapshot() const {
    NodeStatusSnapshot result;
    result.node_id = state_.get_node_id();
    result.hostname = util::hostname();
    result.registered = state_.is_registered();
    result.last_control_contact_ms = state_.get_last_control_contact_ms();
    result.health = state_.get_metrics();
    result.capabilities = state_.get_capabilities();
    result.llama_runtime = state_.get_llama_runtime();
    result.action_progress = state_.get_action_progress();
    result.last_error = state_.get_last_error();
    result.slots = slot_manager_.get_slot_info();
    result.max_slots = slot_manager_.max_slots();
    result.llama_server_path = slot_manager_.llama_server_path();
    result.disk_free_mb = result.health.disk_free_mb;

    for (const auto& slot : result.slots) {
        switch (slot.state) {
            case SlotState::Ready:      ++result.slot_ready; break;
            case SlotState::Loading:    ++result.slot_loading; break;
            case SlotState::Suspending: ++result.slot_suspending; break;
            case SlotState::Suspended:  ++result.slot_suspended; break;
            case SlotState::Error:      ++result.slot_error; break;
            case SlotState::Empty:
            default:
                break;
        }
        if (result.loaded_model.empty() && slot.state == SlotState::Ready) {
            result.loaded_model = slot.model_path;
            result.active_agent = slot.assigned_agent;
        }
    }

    result.slot_in_use = result.slot_ready + result.slot_loading +
                         result.slot_suspending + result.slot_error;
    result.slot_available = std::max(0, result.max_slots - result.slot_in_use);

    ModelStore* store = nullptr;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        store = model_store_;
    }
    if (store) {
        result.model_cache_free_bytes = store->free_bytes();
        for (const auto& model : store->list()) {
            result.managed_models.push_back(NodeManagedModelSnapshot{
                model.id,
                store->load_path(model.id),
                model.size_bytes,
                model.last_used_ms,
                model.pinned,
            });
        }
    }
    return result;
}

std::string NodeService::managed_load_path(const std::string& model_id,
                                           bool pin,
                                           bool* managed,
                                           bool* pin_applied) {
    if (managed) *managed = false;
    if (pin_applied) *pin_applied = false;
    if (model_id.empty()) return {};

    ModelStore* store = nullptr;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        store = model_store_;
    }
    if (!store || !store->get(model_id)) return {};

    const std::string path = store->load_path(model_id);
    std::error_code ec;
    if (path.empty() || !fs::is_regular_file(path, ec)) return {};

    if (pin) {
        store->set_pinned(model_id, true);
        if (pin_applied) *pin_applied = true;
    }
    store->touch(model_id);
    if (managed) *managed = true;
    return path;
}

NodePreparedLocalModel NodeService::prepare_local_model(
    const NodePrepareLocalModelRequest& request) {
    NodePreparedLocalModel result;
    result.model_id = util::trim(request.model_id);

    if (!result.model_id.empty()) {
        result.load_path = managed_load_path(
            result.model_id, request.pin, &result.managed, &result.pin_applied);
        if (!result.load_path.empty()) {
            if (ModelStore* store = [&]() {
                    std::lock_guard<std::mutex> lock(callback_mutex_);
                    return model_store_;
                }()) {
                if (const auto model = store->get(result.model_id)) {
                    result.size_bytes = model->size_bytes;
                }
            }
            result.status = NodeServiceStatus::Ok;
            return result;
        }
    }

    const std::string model_ref = clean_runtime_path(request.model_ref);
    if (model_ref.empty()) {
        result.status = NodeServiceStatus::InvalidArgument;
        result.error = "model_ref required";
        return result;
    }

    auto resolved = util::resolve_existing_local_model_path(model_ref, request.models_dir);
    if (!resolved) {
        ModelStore* store = nullptr;
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            store = model_store_;
        }
        if (store && store->root() != request.models_dir) {
            resolved = util::resolve_existing_local_model_path(model_ref, store->root());
        }
    }
    if (!resolved) {
        result.status = NodeServiceStatus::ModelNotFound;
        result.error = "model file not found locally: " + model_ref;
        return result;
    }

    result.load_path = *resolved;
    if (result.model_id.empty()) result.model_id = util::model_id_from_ref(result.load_path);

    std::error_code ec;
    result.size_bytes = static_cast<int64_t>(fs::file_size(result.load_path, ec));
    if (ec) result.size_bytes = 0;
    result.status = NodeServiceStatus::Ok;
    return result;
}

NodeLoadModelResult NodeService::validate_load_request(
    const NodeLoadModelRequest& request,
    std::string* model_path,
    std::string* mmproj_path) {
    NodeLoadModelResult result;

    const std::string backend = util::to_lower(util::trim(request.backend));
    if (!is_llama_backend(backend)) {
        result.status = NodeServiceStatus::UnsupportedBackend;
        result.error = "unsupported node inference backend: " + backend;
        return result;
    }

    *model_path = clean_runtime_path(request.model_path);
    *mmproj_path = clean_runtime_path(request.mmproj_path);

    if (model_path->empty() && !request.model_id.empty()) {
        *model_path = managed_load_path(request.model_id, request.pin, nullptr, nullptr);
    }
    if (mmproj_path->empty() && !request.mmproj_model_id.empty()) {
        *mmproj_path = managed_load_path(
            request.mmproj_model_id, request.pin, nullptr, nullptr);
    }

    if (model_path->empty()) {
        result.status = NodeServiceStatus::InvalidArgument;
        result.error = "model_path required";
        return result;
    }
    if (request.vision_enabled && mmproj_path->empty()) {
        result.status = NodeServiceStatus::InvalidArgument;
        result.error = "vision-enabled llama-cpp load requires mmproj_path";
        return result;
    }
    if (!request.vision_enabled && !mmproj_path->empty()) {
        result.status = NodeServiceStatus::InvalidArgument;
        result.error = "llama.cpp mmproj_path requires vision_enabled=true";
        return result;
    }

    const auto runtime = state_.get_llama_runtime();
    if (!llama_runtime_ready(runtime)) {
        result.status = NodeServiceStatus::RuntimeUnavailable;
        result.error = runtime.last_error.empty()
            ? "llama.cpp runtime is not ready on this node (status=" +
                  runtime.status + ")"
            : "llama.cpp runtime is not ready on this node: " + runtime.last_error;
        return result;
    }

    std::error_code ec;
    if (!fs::is_regular_file(*model_path, ec)) {
        result.status = NodeServiceStatus::ModelNotFound;
        result.error = "model file not found on this node: " + *model_path;
        return result;
    }
    if (!mmproj_path->empty()) {
        ec.clear();
        if (!fs::is_regular_file(*mmproj_path, ec)) {
            result.status = NodeServiceStatus::ModelNotFound;
            result.error = "projector path not found on this node: " + *mmproj_path;
            return result;
        }
    }

    result.status = NodeServiceStatus::Ok;
    return result;
}

void NodeService::touch_managed_models(const std::string& model_id,
                                       const std::string& mmproj_model_id,
                                       bool pin) {
    ModelStore* store = nullptr;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        store = model_store_;
    }
    if (!store) return;

    std::vector<std::string> ids;
    if (!model_id.empty()) ids.push_back(model_id);
    if (!mmproj_model_id.empty() && mmproj_model_id != model_id) {
        ids.push_back(mmproj_model_id);
    }
    for (const auto& id : ids) {
        if (!store->get(id)) continue;
        if (pin) store->set_pinned(id, true);
        store->touch(id);
    }
}

void NodeService::refresh_slot_state() {
    const auto slots = slot_manager_.get_slot_info();
    state_.set_slots(slots);

    // Keep the legacy single-slot NodeState fields synchronized for the
    // standalone TUI while the typed snapshot remains the source of truth for
    // multi-slot callers.
    std::string loaded_model;
    AgentId active_agent;
    for (const auto& slot : slots) {
        if (slot.state != SlotState::Ready) continue;
        loaded_model = slot.model_path;
        active_agent = slot.assigned_agent;
        break;
    }
    state_.set_loaded_model(loaded_model);
    state_.set_active_agent(active_agent);
}

NodeLoadModelResult NodeService::load_model(const NodeLoadModelRequest& request) {
    std::string model_path;
    std::string mmproj_path;
    auto result = validate_load_request(request, &model_path, &mmproj_path);
    if (!result.ok()) {
        state_.set_last_error(result.error);
        return result;
    }

    result.slot_id = slot_manager_.load_model(
        model_path, mmproj_path, request.runtime_settings, request.agent_id);
    if (result.slot_id.empty()) {
        result.error = slot_manager_.last_error();
        if (result.error.empty()) result.error = "failed to load model";
        result.status = classify_slot_failure(result.error);
        state_.set_last_error(result.error);
        return result;
    }

    result.status = NodeServiceStatus::Ok;
    result.model_path = model_path;
    if (const auto slot = slot_manager_.find_slot(result.slot_id)) {
        result.effective_ctx_size = slot->effective_ctx_size;
    }
    refresh_slot_state();
    state_.set_loaded_model(model_path);
    if (!request.agent_id.empty()) state_.set_active_agent(request.agent_id);
    state_.set_last_error("");
    touch_managed_models(request.model_id, request.mmproj_model_id, request.pin);
    return result;
}

SlotOperationResult NodeService::unload_model(const NodeUnloadModelRequest& request) {
    SlotOperationResult result = request.slot_id.empty()
        ? slot_manager_.unload_all(request.force)
        : slot_manager_.unload_slot(request.slot_id);
    if (result.ok()) {
        refresh_slot_state();
        state_.set_last_error("");
    } else {
        state_.set_last_error(result.message);
    }
    return result;
}

DetachResult NodeService::detach_agent(const NodeDetachAgentRequest& request) {
    if (request.slot_id.empty() || request.agent_id.empty()) {
        auto result = invalid_detach("slot_id and agent_id required");
        state_.set_last_error(result.message);
        return result;
    }
    auto result = slot_manager_.detach_agent(request.slot_id, request.agent_id);
    if (result.ok()) {
        refresh_slot_state();
        state_.set_last_error("");
    } else {
        state_.set_last_error(result.message);
    }
    return result;
}

SlotOperationResult NodeService::suspend_slot(const NodeSuspendSlotRequest& request) {
    if (request.slot_id.empty()) {
        auto result = invalid_slot_operation("slot_id required");
        state_.set_last_error(result.message);
        return result;
    }
    auto result = slot_manager_.suspend_slot(request.slot_id);
    if (result.ok()) {
        refresh_slot_state();
        state_.set_last_error("");
    } else {
        state_.set_last_error(result.message);
    }
    return result;
}

NodeLoadModelResult NodeService::restore_slot(const NodeRestoreModelRequest& request) {
    std::string model_path;
    std::string mmproj_path;
    auto result = validate_load_request(request, &model_path, &mmproj_path);
    if (!result.ok()) {
        state_.set_last_error(result.error);
        return result;
    }

    result.slot_id = slot_manager_.restore_slot(
        model_path,
        mmproj_path,
        request.runtime_settings,
        clean_runtime_path(request.kv_cache_path),
        request.agent_id);
    if (result.slot_id.empty()) {
        result.error = slot_manager_.last_error();
        if (result.error.empty()) result.error = "failed to restore slot";
        result.status = classify_slot_failure(result.error);
        state_.set_last_error(result.error);
        return result;
    }

    result.status = NodeServiceStatus::Ok;
    result.model_path = model_path;
    if (const auto slot = slot_manager_.find_slot(result.slot_id)) {
        result.effective_ctx_size = slot->effective_ctx_size;
    }
    refresh_slot_state();
    state_.set_loaded_model(model_path);
    if (!request.agent_id.empty()) state_.set_active_agent(request.agent_id);
    state_.set_last_error("");
    touch_managed_models(request.model_id, request.mmproj_model_id, request.pin);
    return result;
}

NodeInferResult NodeService::infer(const NodeInferRequest& request,
                                   InferenceChunkCallback chunk_callback,
                                   InferenceErrorCallback error_callback) {
    NodeInferResult result;
    result.message.role = MessageRole::Assistant;
    result.message.timestamp_ms = util::now_ms();

    auto slots = slot_manager_.get_slot_info();
    result.slot_id = request.slot_id;
    if (result.slot_id.empty()) {
        for (const auto& slot : slots) {
            if (slot.state == SlotState::Ready) {
                result.slot_id = slot.id;
                break;
            }
        }
    }

    auto lease = result.slot_id.empty()
        ? SlotManager::SlotLease{}
        : slot_manager_.acquire_slot(result.slot_id);
    if (!lease) {
        result.status = result.slot_id.empty()
            ? NodeServiceStatus::Unavailable
            : NodeServiceStatus::SlotNotFound;
        result.error = "no ready slot available";
        state_.set_last_error(result.error);
        notify_inference_error(error_callback, result.error);
        return result;
    }

    AgentId agent_id;
    std::string model_path;
    for (const auto& slot : slots) {
        if (slot.id != result.slot_id) continue;
        agent_id = slot.assigned_agent;
        model_path = slot.model_path;
        break;
    }
    slot_manager_.touch_slot(result.slot_id);
    refresh_slot_state();
    if (!model_path.empty()) state_.set_loaded_model(model_path);
    if (!agent_id.empty()) state_.set_active_agent(agent_id);
    state_.start_streaming_text(result.slot_id, agent_id);

    RuntimeClient* client = lease.get();
    try {
        if (request.cancel_requested && request.cancel_requested()) {
            result.status = NodeServiceStatus::Canceled;
            result.error = "inference canceled";
            state_.finish_streaming_text("canceled", 0);
            notify_inference_error(error_callback, result.error);
            return result;
        }
        if (request.request.stream) {
            bool had_error = false;
            client->stream_complete(
                request.request,
                [&](const InferenceChunk& chunk) {
                    if (!chunk.thinking_delta.empty()) {
                        result.message.thinking_text += chunk.thinking_delta;
                        state_.append_streaming_text("", chunk.thinking_delta);
                    }
                    if (!chunk.delta_content.empty()) {
                        result.message.content += chunk.delta_content;
                        state_.append_streaming_text(chunk.delta_content, "");
                    }
                    if (chunk.tool_call_delta) {
                        result.message.tool_calls.push_back(*chunk.tool_call_delta);
                    }
                    if (chunk.is_done) {
                        result.tokens_used = chunk.tokens_used;
                        result.finish_reason = chunk.finish_reason;
                        result.message.token_count = chunk.tokens_used;
                        state_.finish_streaming_text(
                            had_error ? "error" : chunk.finish_reason,
                            chunk.tokens_used);
                    }
                    if (chunk_callback) chunk_callback(chunk);
                },
                [&](const std::string& error) {
                    had_error = true;
                    result.error = error;
                    state_.finish_streaming_text("error", result.tokens_used);
                    notify_inference_error(error_callback, error);
                },
                request.cancel_requested);

            if (request.cancel_requested && request.cancel_requested()) {
                result.status = NodeServiceStatus::Canceled;
                result.error = "inference canceled";
                state_.finish_streaming_text("canceled", result.tokens_used);
                notify_inference_error(error_callback, result.error);
                return result;
            }

            if (had_error) {
                result.status = NodeServiceStatus::Failed;
                state_.set_last_error(result.error);
                return result;
            }
            result.status = NodeServiceStatus::Ok;
        } else {
            // Use the streaming runtime transport even for the node protocol's
            // aggregated mode. This keeps disconnect/shutdown cancellation
            // effective and yields one implementation of backend error
            // handling, while the adapter still emits the legacy aggregate
            // sequence after completion.
            bool had_error = false;
            client->stream_complete(
                request.request,
                [&](const InferenceChunk& chunk) {
                    result.message.thinking_text += chunk.thinking_delta;
                    result.message.content += chunk.delta_content;
                    if (chunk.tool_call_delta) {
                        result.message.tool_calls.push_back(*chunk.tool_call_delta);
                    }
                    if (chunk.is_done) {
                        result.tokens_used = chunk.tokens_used;
                        result.finish_reason = chunk.finish_reason;
                        result.message.token_count = chunk.tokens_used;
                    }
                },
                [&](const std::string& error) {
                    had_error = true;
                    result.error = error;
                },
                request.cancel_requested);

            if (request.cancel_requested && request.cancel_requested()) {
                result.status = NodeServiceStatus::Canceled;
                result.error = "inference canceled";
                state_.finish_streaming_text("canceled", result.tokens_used);
                notify_inference_error(error_callback, result.error);
                return result;
            }
            if (had_error) {
                result.status = NodeServiceStatus::Failed;
                if (result.error.empty()) result.error = "non-stream inference failed";
                state_.finish_streaming_text("error", result.tokens_used);
                state_.set_last_error(result.error);
                notify_inference_error(error_callback, result.error);
                return result;
            }
            if (result.finish_reason.empty()) {
                result.finish_reason = result.message.content.empty() &&
                                       result.message.tool_calls.empty()
                    ? "empty"
                    : "stop";
            }
            if (!result.message.thinking_text.empty()) {
                state_.append_streaming_text("", result.message.thinking_text);
            }
            if (!result.message.content.empty()) {
                state_.append_streaming_text(result.message.content, "");
            }
            state_.finish_streaming_text(result.finish_reason, result.tokens_used);

            if (chunk_callback) {
                if (!result.message.thinking_text.empty()) {
                    InferenceChunk chunk;
                    chunk.thinking_delta = result.message.thinking_text;
                    chunk_callback(chunk);
                }
                if (!result.message.content.empty()) {
                    InferenceChunk chunk;
                    chunk.delta_content = result.message.content;
                    chunk_callback(chunk);
                }
                for (const auto& tool_call : result.message.tool_calls) {
                    InferenceChunk chunk;
                    chunk.tool_call_delta = tool_call;
                    chunk_callback(chunk);
                }
                InferenceChunk done;
                done.is_done = true;
                done.tokens_used = result.tokens_used;
                done.finish_reason = result.finish_reason;
                chunk_callback(done);
            }
            result.status = NodeServiceStatus::Ok;
        }
    } catch (const std::exception& exception) {
        result.status = NodeServiceStatus::Failed;
        result.error = std::string("inference exception: ") + exception.what();
        state_.finish_streaming_text("error", result.tokens_used);
        state_.set_last_error(result.error);
        notify_inference_error(error_callback, result.error);
        return result;
    } catch (...) {
        result.status = NodeServiceStatus::Failed;
        result.error = "inference exception: unknown";
        state_.finish_streaming_text("error", result.tokens_used);
        state_.set_last_error(result.error);
        notify_inference_error(error_callback, result.error);
        return result;
    }

    state_.set_last_error("");
    return result;
}

void NodeService::set_runtime_logs_provider(RuntimeLogsProvider provider) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    runtime_logs_provider_ = std::move(provider);
}

std::vector<std::string> NodeService::runtime_logs(int tail) const {
    RuntimeLogsProvider provider;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        provider = runtime_logs_provider_;
    }
    if (!provider) return {};
    return provider(std::clamp(tail, 1, 5000));
}

bool NodeService::cancel_action() {
    return state_.request_action_cancel();
}

void NodeService::set_llama_provision_callback(LlamaProvisionCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    llama_provision_callback_ = std::move(callback);
}

void NodeService::set_llama_update_callback(LlamaUpdateCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    llama_update_callback_ = std::move(callback);
}

void NodeService::set_llama_switch_callback(LlamaSwitchCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    llama_switch_callback_ = std::move(callback);
}

void NodeService::set_llama_check_update_callback(LlamaCheckUpdateCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    llama_check_update_callback_ = std::move(callback);
}

void NodeService::set_llama_diagnose_callback(LlamaDiagnoseCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    llama_diagnose_callback_ = std::move(callback);
}

void NodeService::set_llama_recovery_callback(LlamaRecoveryCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    llama_recovery_callback_ = std::move(callback);
}

NodeRuntimeOperationResult NodeService::runtime_callback_unavailable(
    const char* operation) const {
    NodeRuntimeOperationResult result;
    result.status = NodeServiceStatus::Unavailable;
    result.error = std::string(operation) + " callback is not configured";
    result.runtime = state_.get_llama_runtime();
    return result;
}

NodeRuntimeOperationResult NodeService::apply_runtime_result(
    LlamaRuntimeStatus status,
    bool conflict_on_error,
    bool fail_on_disabled) {
    state_.set_llama_runtime(status);
    state_.set_last_error(status.last_error);

    NodeRuntimeOperationResult result;
    result.runtime = std::move(status);
    result.error = result.runtime.last_error;
    if (result.runtime.status == "failed" ||
        (fail_on_disabled && result.runtime.status == "disabled")) {
        result.status = NodeServiceStatus::Failed;
    } else if (conflict_on_error && !result.runtime.last_error.empty()) {
        result.status = NodeServiceStatus::Conflict;
    } else {
        result.status = NodeServiceStatus::Ok;
    }
    return result;
}

NodeRuntimeOperationResult NodeService::runtime_callback_exception(
    const char* operation,
    const std::string& detail) {
    auto runtime = state_.get_llama_runtime();
    runtime.status = "failed";
    runtime.last_error = std::string(operation) + " exception: " + detail;
    return apply_runtime_result(std::move(runtime), false);
}

NodeRuntimeOperationResult NodeService::provision_llama_runtime() {
    LlamaProvisionCallback callback;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        callback = llama_provision_callback_;
    }
    if (!callback) return runtime_callback_unavailable("llama provision");
    try {
        return apply_runtime_result(callback(), false);
    } catch (const std::exception& exception) {
        return runtime_callback_exception("llama provision", exception.what());
    } catch (...) {
        return runtime_callback_exception("llama provision", "unknown");
    }
}

NodeRuntimeOperationResult NodeService::update_llama_runtime(
    const std::string& accelerator) {
    LlamaUpdateCallback callback;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        callback = llama_update_callback_;
    }
    if (!callback) return runtime_callback_unavailable("llama update");
    try {
        return apply_runtime_result(callback(accelerator), true);
    } catch (const std::exception& exception) {
        return runtime_callback_exception("llama update", exception.what());
    } catch (...) {
        return runtime_callback_exception("llama update", "unknown");
    }
}

NodeRuntimeOperationResult NodeService::switch_llama_runtime(const std::string& variant) {
    if (util::trim(variant).empty()) {
        return {NodeServiceStatus::InvalidArgument,
                "variant is required",
                state_.get_llama_runtime()};
    }
    LlamaSwitchCallback callback;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        callback = llama_switch_callback_;
    }
    if (!callback) return runtime_callback_unavailable("llama switch");
    try {
        return apply_runtime_result(callback(variant), true);
    } catch (const std::exception& exception) {
        return runtime_callback_exception("llama switch", exception.what());
    } catch (...) {
        return runtime_callback_exception("llama switch", "unknown");
    }
}

NodeRuntimeOperationResult NodeService::check_llama_runtime_update() {
    LlamaCheckUpdateCallback callback;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        callback = llama_check_update_callback_;
    }
    if (!callback) return runtime_callback_unavailable("llama update check");
    try {
        return apply_runtime_result(callback(), false, false);
    } catch (const std::exception& exception) {
        return runtime_callback_exception("llama update check", exception.what());
    } catch (...) {
        return runtime_callback_exception("llama update check", "unknown");
    }
}

NodeRuntimeOperationResult NodeService::diagnose_llama_runtime() {
    LlamaDiagnoseCallback callback;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        callback = llama_diagnose_callback_;
    }
    if (!callback) return runtime_callback_unavailable("llama diagnose");
    try {
        return apply_runtime_result(callback(), false, false);
    } catch (const std::exception& exception) {
        return runtime_callback_exception("llama diagnose", exception.what());
    } catch (...) {
        return runtime_callback_exception("llama diagnose", "unknown");
    }
}

NodeRuntimeOperationResult NodeService::recover_llama_runtime(
    const std::string& action,
    const std::string& variant) {
    const std::string normalized_action = util::to_lower(util::trim(action));
    const bool valid_action = normalized_action == "retry" ||
                              normalized_action == "target" ||
                              normalized_action == "compile-anyway" ||
                              normalized_action == "release";
    if (!valid_action ||
        (normalized_action == "release" && util::trim(variant).empty())) {
        return {NodeServiceStatus::InvalidArgument,
                "action must be retry, target, compile-anyway, or release; "
                "release requires variant",
                state_.get_llama_runtime()};
    }
    LlamaRecoveryCallback callback;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        callback = llama_recovery_callback_;
    }
    if (!callback) return runtime_callback_unavailable("llama recovery");
    try {
        return apply_runtime_result(callback(normalized_action, variant), true, false);
    } catch (const std::exception& exception) {
        return runtime_callback_exception("llama recovery", exception.what());
    } catch (...) {
        return runtime_callback_exception("llama recovery", "unknown");
    }
}

} // namespace mm
