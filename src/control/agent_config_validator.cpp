#include "control/agent_config_validator.hpp"

#include "common/gguf_metadata.hpp"
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

// Case-insensitive ".gguf" suffix test. Local to control so we don't pull in the
// node-only vllm_runtime header just for the classifier.
bool looks_like_gguf(const std::string& model_path) {
    const std::string p = util::to_lower(util::trim(model_path));
    return p.size() >= 5 && p.compare(p.size() - 5, 5, ".gguf") == 0;
}

std::string normalized_backend(const AgentConfig& cfg) {
    std::string backend = util::to_lower(util::trim(cfg.inference_backend));
    if (backend.empty()) return "llama-cpp";  // unspecified => default runtime
    if (backend == "llama.cpp" || backend == "llama" || backend == "llama-cpp")
        return "llama-cpp";
    return backend;
}

} // namespace

AgentValidationResult validate_agent_config(const AgentConfig& cfg,
                                            const NodeRegistry* registry,
                                            const std::string& models_dir,
                                            const ModelCapabilityInfo* precomputed_model_info) {
    AgentValidationResult result;

    if (is_blank(cfg.name)) {
        add_issue(result, ValidationSeverity::Error, "name", "Name is required.");
    }
    if (is_blank(cfg.model_path)) {
        add_issue(result, ValidationSeverity::Error, "model_path", "Model path is required.");
    }
    const std::string backend = normalized_backend(cfg);
    if (backend != "vllm" && backend != "api" && backend != "llama-cpp") {
        add_issue(result,
                  ValidationSeverity::Error,
                  "inference_backend",
                  "inference_backend must be 'vllm', 'llama-cpp', or 'api'.");
    }
    const bool is_local_backend = (backend == "vllm" || backend == "llama-cpp");
    if (!cfg.id.empty() && !util::is_valid_agent_id(cfg.id)) {
        add_issue(result,
                  ValidationSeverity::Error,
                  "id",
                  "Agent ID must be 1-128 characters, start with an alphanumeric character, and use only [a-zA-Z0-9_-].");
    }
    if (cfg.runtime_settings.ctx_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.ctx_size", "ctx_size must be greater than 0.");
    }
    if (cfg.runtime_settings.max_tokens == -1) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "runtime_settings.max_tokens",
                  "max_tokens is -1 (unlimited generation). Responses may run indefinitely until stopped.");
    } else if (cfg.runtime_settings.max_tokens <= 0) {
        add_issue(result,
                  ValidationSeverity::Error,
                  "runtime_settings.max_tokens",
                  "max_tokens must be greater than 0, or -1 for unlimited generation.");
    }
    if (cfg.runtime_settings.top_p <= 0.0f || cfg.runtime_settings.top_p > 1.0f || !std::isfinite(cfg.runtime_settings.top_p)) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.top_p", "top_p must be within (0, 1].");
    }
    if (cfg.runtime_settings.temperature < 0.0f || !std::isfinite(cfg.runtime_settings.temperature)) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.temperature", "temperature must be a finite value >= 0.");
    }
    if (cfg.runtime_settings.top_k < -1) {
        add_issue(result,
                  ValidationSeverity::Error,
                  "runtime_settings.top_k",
                  "top_k must be -1 (runtime default), 0 (disabled), or greater.");
    }
    if ((cfg.runtime_settings.min_p < 0.0f && cfg.runtime_settings.min_p != -1.0f) ||
        cfg.runtime_settings.min_p > 1.0f ||
        !std::isfinite(cfg.runtime_settings.min_p)) {
        add_issue(result,
                  ValidationSeverity::Error,
                  "runtime_settings.min_p",
                  "min_p must be -1 (runtime default) or within [0, 1].");
    }
    if (cfg.runtime_settings.presence_penalty < -2.0f ||
        cfg.runtime_settings.presence_penalty > 2.0f ||
        !std::isfinite(cfg.runtime_settings.presence_penalty)) {
        add_issue(result,
                  ValidationSeverity::Error,
                  "runtime_settings.presence_penalty",
                  "presence_penalty must be within [-2, 2].");
    }
    if ((cfg.runtime_settings.repeat_penalty <= 0.0f &&
         cfg.runtime_settings.repeat_penalty != -1.0f) ||
        !std::isfinite(cfg.runtime_settings.repeat_penalty)) {
        add_issue(result,
                  ValidationSeverity::Error,
                  "runtime_settings.repeat_penalty",
                  "repeat_penalty must be -1 (runtime default) or greater than 0.");
    }
    if (cfg.runtime_settings.n_gpu_layers < -1) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.n_gpu_layers", "n_gpu_layers must be -1 or greater.");
    }
    if (cfg.runtime_settings.n_threads < -1) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.n_threads", "n_threads must be -1 or greater.");
    }
    if (cfg.runtime_settings.n_threads_http < -1) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.n_threads_http", "n_threads_http must be -1 or greater.");
    }
    if (cfg.runtime_settings.parallel <= 0) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.parallel", "parallel must be greater than 0.");
    }
    if (cfg.runtime_settings.batch_size != -1 && cfg.runtime_settings.batch_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.batch_size", "batch_size must be greater than 0, or -1 for the runtime default.");
    }
    if (cfg.runtime_settings.ubatch_size != -1 && cfg.runtime_settings.ubatch_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.ubatch_size", "ubatch_size must be greater than 0, or -1 for the runtime default.");
    }
    if (backend == "vllm") {
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
        // A GGUF served through vLLM is experimental and architecture-limited;
        // llama.cpp is the reliable backend for it.
        if (looks_like_gguf(cfg.model_path)) {
            add_issue(result,
                      ValidationSeverity::Warning,
                      "inference_backend",
                      "model_path is a GGUF file but inference_backend is 'vllm'. "
                      "vLLM's GGUF support is experimental; consider 'llama-cpp'.");
        }
    } else if (backend == "llama-cpp") {
        // llama.cpp loads a local GGUF file (or a directory of split GGUF
        // shards). An HF repo id would need an out-of-band pull we do not do yet.
        if (util::is_hf_repo_id(cfg.model_path)) {
            add_issue(result,
                      ValidationSeverity::Warning,
                      "model_path",
                      "model_path looks like a Hugging Face repo id. The llama.cpp "
                      "backend loads a local GGUF file; transfer or point at a "
                      "node-local .gguf path.");
        } else if (!looks_like_gguf(cfg.model_path)) {
            add_issue(result,
                      ValidationSeverity::Warning,
                      "model_path",
                      "model_path does not end in .gguf. The llama.cpp backend "
                      "expects a GGUF model file.");
        }
        // Inspect GGUF metadata (best effort) to sanity-check the requested
        // context against the model's trained context length.
        ModelCapabilityInfo model_info =
            precomputed_model_info ? *precomputed_model_info
                                   : inspect_model_capabilities(cfg.model_path, models_dir);
        if (model_info.metadata_found && model_info.n_ctx_train > 0 &&
            cfg.runtime_settings.ctx_size > model_info.n_ctx_train) {
            add_issue(result,
                      ValidationSeverity::Warning,
                      "runtime_settings.ctx_size",
                      "ctx_size (" + std::to_string(cfg.runtime_settings.ctx_size) +
                      ") exceeds the model's trained context length (" +
                      std::to_string(model_info.n_ctx_train) +
                      "); the model may degrade or fail to load.");
        }
        if (model_info.metadata_found || !model_info.warnings.empty())
            result.model_info = model_info;
    } else if (backend == "api") {
        if (is_blank(cfg.api_settings.base_url)) {
            add_issue(result, ValidationSeverity::Error, "api_settings.base_url", "API base_url is required.");
        }
        if (is_blank(cfg.api_settings.chat_completions_path)) {
            add_issue(result,
                      ValidationSeverity::Error,
                      "api_settings.chat_completions_path",
                      "chat_completions_path is required.");
        } else if (cfg.api_settings.chat_completions_path.front() != '/') {
            add_issue(result,
                      ValidationSeverity::Error,
                      "api_settings.chat_completions_path",
                      "chat_completions_path must start with '/'.");
        }
        if (!cfg.preferred_node_id.empty()) {
            add_issue(result,
                      ValidationSeverity::Warning,
                      "preferred_node_id",
                      "Preferred node is ignored for API-backed agents.");
        }
    }
    if (cfg.tools_enabled && !cfg.memories_enabled) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "tools_enabled",
                  "Tools are enabled, but no executable tools are currently available unless Memories is also enabled.");
    }

    if (is_local_backend && !cfg.preferred_node_id.empty() && registry) {
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

    if (backend == "vllm" && cfg.vllm_settings.max_model_len > 131072) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "vllm_settings.max_model_len",
                  "max_model_len is extremely large and may fail to load or perform poorly.");
    } else if (backend == "vllm" && cfg.vllm_settings.max_model_len > 65536) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "vllm_settings.max_model_len",
                  "max_model_len is very large and may be slow or require substantial memory.");
    }

    return result;
}

} // namespace mm
