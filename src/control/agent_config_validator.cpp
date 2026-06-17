#include "control/agent_config_validator.hpp"

#include "common/util.hpp"
#include "control/node_registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mm {

namespace {

void add_issue(AgentValidationResult& result,
               ValidationSeverity severity,
               std::string field,
               std::string message) {
    result.issues.push_back({severity, std::move(field), std::move(message)});
}

bool is_blank(const std::string& s) {
    return util::trim(s).empty();
}

} // namespace

AgentValidationResult validate_agent_config(const AgentConfig& cfg,
                                            const NodeRegistry* registry,
                                            const std::string& models_dir,
                                            const ModelCapabilityInfo* precomputed_model_info) {
    AgentValidationResult result;

    // models_dir and precomputed_model_info are no longer used now that model
    // capability inspection (GGUF metadata) has been removed; the config is
    // always vLLM and model_info is always empty.
    (void)models_dir;
    (void)precomputed_model_info;

    if (is_blank(cfg.name)) {
        add_issue(result, ValidationSeverity::Error, "name", "Name is required.");
    }
    if (is_blank(cfg.model_path)) {
        add_issue(result, ValidationSeverity::Error, "model_path", "Model path is required.");
    }
    if (!cfg.id.empty() && !util::is_valid_agent_id(cfg.id)) {
        add_issue(result,
                  ValidationSeverity::Error,
                  "id",
                  "Agent ID must be 1-128 characters, start with an alphanumeric character, and use only [a-zA-Z0-9_-].");
    }
    if (cfg.llama_settings.ctx_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "llama_settings.ctx_size", "ctx_size must be greater than 0.");
    }
    if (cfg.llama_settings.max_tokens == -1) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "llama_settings.max_tokens",
                  "max_tokens is -1 (unlimited generation). Responses may run indefinitely until stopped.");
    } else if (cfg.llama_settings.max_tokens <= 0) {
        add_issue(result,
                  ValidationSeverity::Error,
                  "llama_settings.max_tokens",
                  "max_tokens must be greater than 0, or -1 for unlimited generation.");
    }
    if (cfg.llama_settings.top_p <= 0.0f || cfg.llama_settings.top_p > 1.0f || !std::isfinite(cfg.llama_settings.top_p)) {
        add_issue(result, ValidationSeverity::Error, "llama_settings.top_p", "top_p must be within (0, 1].");
    }
    if (cfg.llama_settings.temperature < 0.0f || !std::isfinite(cfg.llama_settings.temperature)) {
        add_issue(result, ValidationSeverity::Error, "llama_settings.temperature", "temperature must be a finite value >= 0.");
    }
    if (cfg.llama_settings.n_gpu_layers < -1) {
        add_issue(result, ValidationSeverity::Error, "llama_settings.n_gpu_layers", "n_gpu_layers must be -1 or greater.");
    }
    if (cfg.llama_settings.n_threads < -1) {
        add_issue(result, ValidationSeverity::Error, "llama_settings.n_threads", "n_threads must be -1 or greater.");
    }
    if (cfg.llama_settings.n_threads_http < -1) {
        add_issue(result, ValidationSeverity::Error, "llama_settings.n_threads_http", "n_threads_http must be -1 or greater.");
    }
    if (cfg.llama_settings.parallel <= 0) {
        add_issue(result, ValidationSeverity::Error, "llama_settings.parallel", "parallel must be greater than 0.");
    }
    if (cfg.llama_settings.batch_size != -1 && cfg.llama_settings.batch_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "llama_settings.batch_size", "batch_size must be greater than 0, or -1 for the llama-server default.");
    }
    if (cfg.llama_settings.ubatch_size != -1 && cfg.llama_settings.ubatch_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "llama_settings.ubatch_size", "ubatch_size must be greater than 0, or -1 for the llama-server default.");
    }
    if (cfg.vllm_settings.max_model_len <= 0) {
        add_issue(result, ValidationSeverity::Error, "vllm_settings.max_model_len", "max_model_len must be greater than 0.");
    }
    if (cfg.vllm_settings.max_num_seqs <= 0) {
        add_issue(result, ValidationSeverity::Error, "vllm_settings.max_num_seqs", "max_num_seqs must be greater than 0.");
    }
    if (cfg.vllm_settings.max_num_batched_tokens != -1 &&
        cfg.vllm_settings.max_num_batched_tokens <= 0) {
        add_issue(result, ValidationSeverity::Error, "vllm_settings.max_num_batched_tokens", "max_num_batched_tokens must be greater than 0, or -1 for the vLLM default.");
    }
    if (cfg.vllm_settings.tensor_parallel_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "vllm_settings.tensor_parallel_size", "tensor_parallel_size must be greater than 0.");
    }
    if (cfg.vllm_settings.pipeline_parallel_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "vllm_settings.pipeline_parallel_size", "pipeline_parallel_size must be greater than 0.");
    }
    if (cfg.vllm_settings.gpu_memory_utilization <= 0.0 ||
        cfg.vllm_settings.gpu_memory_utilization > 1.0 ||
        !std::isfinite(cfg.vllm_settings.gpu_memory_utilization)) {
        add_issue(result, ValidationSeverity::Error, "vllm_settings.gpu_memory_utilization", "gpu_memory_utilization must be within (0, 1].");
    }
    if (cfg.tools_enabled && !cfg.memories_enabled) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "tools_enabled",
                  "Tools are enabled, but no executable tools are currently available unless Memories is also enabled.");
    }

    if (!cfg.preferred_node_id.empty() && registry) {
        const auto nodes = registry->list_nodes();
        auto it = std::find_if(nodes.begin(), nodes.end(), [&](const NodeInfo& node) {
            return node.id == cfg.preferred_node_id;
        });
        if (it == nodes.end()) {
            add_issue(result,
                      ValidationSeverity::Warning,
                      "preferred_node_id",
                      "Preferred node was not found among registered nodes.");
        } else if (!it->connected) {
            add_issue(result,
                      ValidationSeverity::Warning,
                      "preferred_node_id",
                      "Preferred node is currently disconnected.");
        }
    }

    if (cfg.vllm_settings.max_model_len > 131072) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "vllm_settings.max_model_len",
                  "max_model_len is extremely large and may fail to load or perform poorly.");
    } else if (cfg.vllm_settings.max_model_len > 65536) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "vllm_settings.max_model_len",
                  "max_model_len is very large and may be slow or require substantial memory.");
    }

    return result;
}

} // namespace mm
