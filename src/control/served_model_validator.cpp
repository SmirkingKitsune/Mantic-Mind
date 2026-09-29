#include "control/served_model_validator.hpp"

#include "common/gguf_metadata.hpp"
#include "common/util.hpp"
#include "control/model_registry.hpp"
#include "control/node_registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>

namespace mm {

namespace {

void add_issue(ValidationResult& result,
               ValidationSeverity severity,
               std::string field,
               std::string message) {
    result.issues.push_back({severity, std::move(field), std::move(message)});
}

bool is_blank(const std::string& s) {
    return util::trim(s).empty();
}

// Case-insensitive ".gguf" suffix test kept local to the control process.
bool looks_like_gguf(const std::string& model_path) {
    const std::string p = util::to_lower(util::trim(model_path));
    return p.size() >= 5 && p.compare(p.size() - 5, 5, ".gguf") == 0;
}

bool has_authoritative_mmproj_flag(const RuntimeSettings& settings) {
    for (const auto& raw : settings.extra_args) {
        const std::string arg = util::to_lower(util::trim(raw));
        for (const std::string flag : {"--mmproj", "-mm", "--mmproj-url"}) {
            if (arg == flag || arg.rfind(flag + "=", 0) == 0 ||
                arg.rfind(flag + " ", 0) == 0) {
                return true;
            }
        }
    }
    return false;
}

void validate_runtime_settings(ValidationResult& result, const RuntimeSettings& rs) {
    if (rs.ctx_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.ctx_size",
                  "ctx_size must be greater than 0.");
    }
    // The sampling ranges have ONE home, shared with conversation and request
    // overrides, so a default and an override cannot be held to different rules.
    for (const auto& problem : sampling_override_problems(sampling_of(rs))) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings." + problem.field,
                  problem.message);
    }
    if (rs.max_tokens == -1) {
        add_issue(result, ValidationSeverity::Warning, "runtime_settings.max_tokens",
                  "max_tokens is -1 (unlimited generation). Responses may run indefinitely "
                  "until stopped.");
    }
    if (rs.n_gpu_layers < -1) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.n_gpu_layers",
                  "n_gpu_layers must be -1 or greater.");
    }
    if (rs.n_threads < -1) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.n_threads",
                  "n_threads must be -1 or greater.");
    }
    if (rs.n_threads_http < -1) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.n_threads_http",
                  "n_threads_http must be -1 or greater.");
    }
    if (rs.parallel <= 0) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.parallel",
                  "parallel must be greater than 0.");
    }
    if (rs.batch_size != -1 && rs.batch_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.batch_size",
                  "batch_size must be greater than 0, or -1 for the runtime default.");
    }
    if (rs.ubatch_size != -1 && rs.ubatch_size <= 0) {
        add_issue(result, ValidationSeverity::Error, "runtime_settings.ubatch_size",
                  "ubatch_size must be greater than 0, or -1 for the runtime default.");
    }
}

} // namespace

std::vector<std::string> suggest_mmproj_files(const std::string& model_path) {
    namespace fs = std::filesystem;
    std::vector<std::string> suggestions;
    const fs::path parent = fs::path(util::trim(model_path)).parent_path();
    std::error_code ec;
    if (parent.empty() || !fs::is_directory(parent, ec)) return suggestions;
    for (const auto& entry : fs::directory_iterator(
             parent, fs::directory_options::skip_permission_denied, ec)) {
        std::error_code file_ec;
        if (!entry.is_regular_file(file_ec)) continue;
        const std::string basename = util::to_lower(entry.path().filename().string());
        if (basename.rfind("mmproj-", 0) == 0 &&
            basename.size() >= 5 && basename.ends_with(".gguf")) {
            suggestions.push_back(entry.path().lexically_normal().string());
        }
    }
    std::sort(suggestions.begin(), suggestions.end(), [](const auto& a, const auto& b) {
        return util::to_lower(a) < util::to_lower(b);
    });
    return suggestions;
}

ValidationResult validate_served_model(const ServedModel& model,
                                       const NodeRegistry* registry,
                                       const std::string& models_dir,
                                       const ControlModelRegistry* admissions,
                                       const ModelCapabilityInfo* precomputed_model_info) {
    ValidationResult result;

    if (!model.id.empty() && !util::is_valid_served_model_id(model.id)) {
        add_issue(result, ValidationSeverity::Error, "id",
                  "Served-model id must be 1-128 characters of [a-z0-9._-], starting with a "
                  "lowercase letter or digit.");
    }
    if (is_blank(model.model_path)) {
        add_issue(result, ValidationSeverity::Error, "model_path", "Model path is required.");
    }

    const std::string backend = model.inference_backend;
    if (backend != "node" && backend != "api") {
        add_issue(result, ValidationSeverity::Error, "inference_backend",
                  "inference_backend must be 'node' (served on a cluster node) or 'api' (a "
                  "remote OpenAI-compatible provider).");
    }
    if (model.backend_override != "auto" && model.backend_override != "soma" &&
        model.backend_override != "fallback") {
        add_issue(result, ValidationSeverity::Error, "backend_override",
                  "backend_override must be 'auto', 'soma' or 'fallback'.");
    }

    const bool node_local = backend == "node";
    const std::string mmproj_path = util::trim(model.vision_settings.mmproj_path);
    if (node_local) {
        if (model.vision_settings.enabled && mmproj_path.empty()) {
            add_issue(result, ValidationSeverity::Error, "vision_settings.mmproj_path",
                      "Vision-enabled node models require an explicit GGUF projector path.");
        }
        if (!mmproj_path.empty()) {
            if (!looks_like_gguf(mmproj_path)) {
                add_issue(result, ValidationSeverity::Error, "vision_settings.mmproj_path",
                          "The llama.cpp vision projector must end in .gguf.");
            } else {
                const std::string basename =
                    util::to_lower(std::filesystem::path(mmproj_path).filename().string());
                if (basename.rfind("mmproj-", 0) != 0) {
                    add_issue(result, ValidationSeverity::Warning,
                              "vision_settings.mmproj_path",
                              "Projector filename does not use the conventional mmproj-*.gguf "
                              "form; verify it matches the model.");
                }
            }
        }
        if (has_authoritative_mmproj_flag(model.runtime_settings)) {
            add_issue(result, ValidationSeverity::Error, "runtime_settings.extra_args",
                      "Projector flags (--mmproj, -mm, --mmproj-url) are controlled by "
                      "vision_settings.mmproj_path and cannot appear in extra_args.");
        }
    } else if (!mmproj_path.empty()) {
        add_issue(result, ValidationSeverity::Error, "vision_settings.mmproj_path",
                  "API-backed models must leave mmproj_path empty; the remote model owns "
                  "image support.");
    }

    validate_runtime_settings(result, model.runtime_settings);

    if (node_local) {
        // An ADMITTED reference is a container, not a GGUF file: the checks
        // below would warn about the one shape of model_path that is correct.
        const bool admitted =
            admissions != nullptr && admissions->resolve(model.model_path).has_value();
        if (!admitted) {
            // A node loads a node-local GGUF file for llama.cpp. An HF repo id
            // would need an out-of-band download this path does not perform.
            const auto local_model =
                util::resolve_existing_local_model_path(model.model_path, models_dir);
            if (!local_model && util::is_hf_repo_id(model.model_path)) {
                add_issue(result, ValidationSeverity::Warning, "model_path",
                          "model_path looks like a Hugging Face repo id. llama.cpp loads a "
                          "local GGUF file; transfer or point at a node-local .gguf path, or "
                          "admit the model so Soma can serve it.");
            } else if (!looks_like_gguf(model.model_path)) {
                add_issue(result, ValidationSeverity::Warning, "model_path",
                          "model_path does not end in .gguf and no admission matches it. "
                          "llama.cpp expects a GGUF model file.");
            }
            // Inspect GGUF metadata (best effort) to sanity-check the requested
            // context against the model's trained context length.
            ModelCapabilityInfo model_info =
                precomputed_model_info ? *precomputed_model_info
                                       : inspect_model_capabilities(model.model_path, models_dir);
            if (model_info.metadata_found && model_info.n_ctx_train > 0 &&
                model.runtime_settings.ctx_size > model_info.n_ctx_train) {
                add_issue(result, ValidationSeverity::Warning, "runtime_settings.ctx_size",
                          "ctx_size (" + std::to_string(model.runtime_settings.ctx_size) +
                              ") exceeds the model's trained context length (" +
                              std::to_string(model_info.n_ctx_train) +
                              "); the model may degrade or fail to load.");
            }
            if (model_info.metadata_found || !model_info.warnings.empty())
                result.model_info = model_info;
        }
    } else if (backend == "api") {
        if (is_blank(model.api_settings.base_url)) {
            add_issue(result, ValidationSeverity::Error, "api_settings.base_url",
                      "API base_url is required.");
        }
        if (is_blank(model.api_settings.chat_completions_path)) {
            add_issue(result, ValidationSeverity::Error, "api_settings.chat_completions_path",
                      "chat_completions_path is required.");
        } else if (model.api_settings.chat_completions_path.front() != '/') {
            add_issue(result, ValidationSeverity::Error, "api_settings.chat_completions_path",
                      "chat_completions_path must start with '/'.");
        }
        if (!model.preferred_node_id.empty()) {
            add_issue(result, ValidationSeverity::Warning, "preferred_node_id",
                      "Preferred node is ignored for API-backed models.");
        }
    }

    if (node_local && !model.preferred_node_id.empty() && registry) {
        const auto nodes = registry->list_nodes();
        auto it = std::find_if(nodes.begin(), nodes.end(), [&](const NodeInfo& node) {
            return node.id == model.preferred_node_id;
        });
        if (it == nodes.end()) {
            add_issue(result, ValidationSeverity::Warning, "preferred_node_id",
                      "Preferred node was not found among registered nodes.");
        } else if (!it->connected) {
            add_issue(result, ValidationSeverity::Warning, "preferred_node_id",
                      "Preferred node is currently disconnected.");
        }
    }

    return result;
}

} // namespace mm
