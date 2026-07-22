#include "control/agent_config_validator.hpp"

#include "common/util.hpp"
#include "control/node_registry.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace mm {

namespace {

void add_issue(AgentValidationResult& result,
               ValidationSeverity severity,
               std::string field,
               std::string message) {
    result.issues.push_back({severity, std::move(field), std::move(message)});
}

bool is_blank(const std::string& value) {
    return util::trim(value).empty();
}

std::string normalized_backend(const AgentConfig& cfg) {
    std::string backend = util::to_lower(util::trim(cfg.inference_backend));
    if (backend.empty()) return "vllm";
    if (backend == "llama.cpp" || backend == "llama" || backend == "llama-cpp") {
        return "llama-cpp";
    }
    return backend;
}

} // namespace

AgentValidationResult validate_agent_config(const AgentConfig& cfg,
                                            const NodeRegistry* registry) {
    AgentValidationResult result;

    if (is_blank(cfg.name)) {
        add_issue(result, ValidationSeverity::Error, "name", "Name is required.");
    }
    if (is_blank(cfg.model_path)) {
        add_issue(result, ValidationSeverity::Error, "model_path",
                  "Model path is required.");
    }

    const std::string backend = normalized_backend(cfg);
    if (backend == "llama-cpp") {
        add_issue(result,
                  ValidationSeverity::Error,
                  "inference_backend",
                  "The 'llama-cpp' backend is not available in this runtime. "
                  "Use 'vllm' for local inference or 'api' for a remote "
                  "OpenAI-compatible service.");
    } else if (backend != "vllm" && backend != "api") {
        add_issue(result,
                  ValidationSeverity::Error,
                  "inference_backend",
                  "inference_backend must be 'vllm' or 'api'.");
    }

    if (!cfg.id.empty() && !util::is_valid_agent_id(cfg.id)) {
        add_issue(result,
                  ValidationSeverity::Error,
                  "id",
                  "Agent ID must be 1-128 characters, start with an "
                  "alphanumeric character, and use only [a-zA-Z0-9_-].");
    }

    const auto& runtime = cfg.runtime_settings;
    if (runtime.ctx_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.ctx_size",
                  "ctx_size must be greater than 0.");
    }
    if (runtime.max_tokens == -1) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "runtime_settings.max_tokens",
                  "max_tokens is -1 (unlimited generation). Responses may run "
                  "indefinitely until stopped.");
    } else if (runtime.max_tokens <= 0) {
        add_issue(result,
                  ValidationSeverity::Error,
                  "runtime_settings.max_tokens",
                  "max_tokens must be greater than 0, or -1 for unlimited "
                  "generation.");
    }
    if (runtime.top_p <= 0.0f || runtime.top_p > 1.0f ||
        !std::isfinite(runtime.top_p)) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.top_p",
                  "top_p must be within (0, 1].");
    }
    if (runtime.temperature < 0.0f || !std::isfinite(runtime.temperature)) {
        add_issue(result, ValidationSeverity::Error,
                  "runtime_settings.temperature",
                  "temperature must be a finite value >= 0.");
    }
    if (runtime.top_k < -1) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.top_k",
                  "top_k must be -1 (runtime default), 0 (disabled), or greater.");
    }
    if ((runtime.min_p < 0.0f && runtime.min_p != -1.0f) ||
        runtime.min_p > 1.0f || !std::isfinite(runtime.min_p)) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.min_p",
                  "min_p must be -1 (runtime default) or within [0, 1].");
    }
    if (runtime.presence_penalty < -2.0f ||
        runtime.presence_penalty > 2.0f ||
        !std::isfinite(runtime.presence_penalty)) {
        add_issue(result, ValidationSeverity::Error,
                  "runtime_settings.presence_penalty",
                  "presence_penalty must be within [-2, 2].");
    }
    if ((runtime.repeat_penalty <= 0.0f && runtime.repeat_penalty != -1.0f) ||
        !std::isfinite(runtime.repeat_penalty)) {
        add_issue(result, ValidationSeverity::Error,
                  "runtime_settings.repeat_penalty",
                  "repeat_penalty must be -1 (runtime default) or greater than 0.");
    }

    if (backend == "vllm") {
        const auto& vllm = cfg.vllm_settings;
        if (vllm.max_model_len <= 0) {
            add_issue(result, ValidationSeverity::Error,
                      "vllm_settings.max_model_len",
                      "max_model_len must be greater than 0.");
        }
        if (vllm.max_num_seqs <= 0) {
            add_issue(result, ValidationSeverity::Error,
                      "vllm_settings.max_num_seqs",
                      "max_num_seqs must be greater than 0.");
        }
        if (vllm.max_num_batched_tokens != -1 &&
            vllm.max_num_batched_tokens <= 0) {
            add_issue(result, ValidationSeverity::Error,
                      "vllm_settings.max_num_batched_tokens",
                      "max_num_batched_tokens must be greater than 0, or -1 "
                      "for the vLLM default.");
        }
        if (vllm.tensor_parallel_size <= 0) {
            add_issue(result, ValidationSeverity::Error,
                      "vllm_settings.tensor_parallel_size",
                      "tensor_parallel_size must be greater than 0.");
        }
        if (vllm.pipeline_parallel_size <= 0) {
            add_issue(result, ValidationSeverity::Error,
                      "vllm_settings.pipeline_parallel_size",
                      "pipeline_parallel_size must be greater than 0.");
        }
        if (vllm.gpu_memory_utilization <= 0.0 ||
            vllm.gpu_memory_utilization > 1.0 ||
            !std::isfinite(vllm.gpu_memory_utilization)) {
            add_issue(result, ValidationSeverity::Error,
                      "vllm_settings.gpu_memory_utilization",
                      "gpu_memory_utilization must be within (0, 1].");
        }

        if (vllm.max_model_len > 131072) {
            add_issue(result, ValidationSeverity::Warning,
                      "vllm_settings.max_model_len",
                      "max_model_len is extremely large and may fail to load "
                      "or perform poorly.");
        } else if (vllm.max_model_len > 65536) {
            add_issue(result, ValidationSeverity::Warning,
                      "vllm_settings.max_model_len",
                      "max_model_len is very large and may be slow or require "
                      "substantial memory.");
        }

        if (!cfg.preferred_node_id.empty() && registry) {
            const auto nodes = registry->list_nodes();
            const auto node = std::find_if(
                nodes.begin(), nodes.end(), [&](const NodeInfo& candidate) {
                    return candidate.id == cfg.preferred_node_id;
                });
            if (node == nodes.end()) {
                add_issue(result, ValidationSeverity::Warning,
                          "preferred_node_id",
                          "Preferred node was not found among registered nodes.");
            } else if (!node->connected) {
                add_issue(result, ValidationSeverity::Warning,
                          "preferred_node_id",
                          "Preferred node is currently disconnected.");
            }
        }
    } else if (backend == "api") {
        if (is_blank(cfg.api_settings.base_url)) {
            add_issue(result, ValidationSeverity::Error,
                      "api_settings.base_url", "API base_url is required.");
        }
        if (is_blank(cfg.api_settings.chat_completions_path)) {
            add_issue(result, ValidationSeverity::Error,
                      "api_settings.chat_completions_path",
                      "chat_completions_path is required.");
        } else if (cfg.api_settings.chat_completions_path.front() != '/') {
            add_issue(result, ValidationSeverity::Error,
                      "api_settings.chat_completions_path",
                      "chat_completions_path must start with '/'.");
        }
        if (!cfg.preferred_node_id.empty()) {
            add_issue(result, ValidationSeverity::Warning, "preferred_node_id",
                      "Preferred node is ignored for API-backed agents.");
        }
    }

    if (cfg.tools_enabled && !cfg.memories_enabled) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "tools_enabled",
                  "Tools are enabled, but no executable tools are currently "
                  "available unless Memories is also enabled.");
    }

    return result;
}

} // namespace mm
