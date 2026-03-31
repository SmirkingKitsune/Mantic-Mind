#include "control/agent_config_validator.hpp"

#include "common/gguf_metadata.hpp"
#include "common/util.hpp"
#include "control/node_registry.hpp"

#include <algorithm>
#include <cmath>

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

    ModelCapabilityInfo model_info;
    if (precomputed_model_info) {
        model_info = *precomputed_model_info;
    } else if (!is_blank(cfg.model_path)) {
        model_info = inspect_model_capabilities(cfg.model_path, models_dir);
    }

    if (!is_blank(cfg.model_path)) {
        result.model_info = model_info;

        if (model_info.metadata_found && model_info.n_ctx_train > 0 &&
            cfg.llama_settings.ctx_size > model_info.n_ctx_train) {
            add_issue(result,
                      ValidationSeverity::Error,
                      "llama_settings.ctx_size",
                      "ctx_size " + std::to_string(cfg.llama_settings.ctx_size) +
                          " exceeds model training context " + std::to_string(model_info.n_ctx_train) +
                          ". Lower ctx_size to <= " + std::to_string(model_info.n_ctx_train) + ".");
        } else if (!model_info.metadata_found || model_info.n_ctx_train <= 0) {
            add_issue(result,
                      ValidationSeverity::Warning,
                      "model_path",
                      "Model metadata could not verify the maximum context size for this model.");
        }

        for (const auto& warning : model_info.warnings) {
            add_issue(result, ValidationSeverity::Warning, "model_path", warning);
        }

        if (cfg.tools_enabled && !cfg.memories_enabled) {
            add_issue(result,
                      ValidationSeverity::Warning,
                      "tools_enabled",
                      "Tools are enabled, but no executable tools are currently available unless Memories is also enabled.");
        }

        if (cfg.tools_enabled &&
            model_info.metadata_found &&
            !model_info.supports_tool_calls &&
            !model_info.used_filename_heuristics) {
            add_issue(result,
                      ValidationSeverity::Warning,
                      "tools_enabled",
                      "GGUF metadata did not advertise tool-call support for this model. Some models can still use tools if their prompt template supports them, so verify with a quick test chat.");
        }

        if (cfg.reasoning_enabled && !model_info.supports_reasoning) {
            add_issue(result,
                      ValidationSeverity::Warning,
                      "reasoning_enabled",
                      "The selected model does not appear to advertise a reasoning chat template.");
        }
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

    if (cfg.llama_settings.ctx_size > 131072) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "llama_settings.ctx_size",
                  "ctx_size is extremely large and may fail to load or perform poorly.");
    } else if (cfg.llama_settings.ctx_size > 65536) {
        add_issue(result,
                  ValidationSeverity::Warning,
                  "llama_settings.ctx_size",
                  "ctx_size is very large and may be slow or require substantial memory.");
    }

    return result;
}

} // namespace mm
