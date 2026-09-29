#pragma once

// Mantic-Mind — a model a user can pick.
//
// A served model is the operator's answer to "how is this model served": which
// weights, which engine policy, which launch settings, which default sampling.
// It belongs to nobody. Users choose one per conversation (and may change their
// choice between turns), so the same served model is shared by every user who
// picks it, and one conversation may be answered by several.
//
// This is the MODEL HALF of what used to be an agent. An agent welded that half
// to a persona and a private database, so changing an agent's model changed the
// agent, and a conversation could never be answered by more than one model.
// Everything here was on AgentConfig; nothing here is about who is asking.
//
// ── Where a served model comes from ──────────────────────────────────────────
//
// Explicit rows are written by an operator and stored in control.db. Admitted
// models that Soma can serve are listed WITHOUT a row: ServedModelCatalog
// computes them from the admission registry on every read, so an admission can
// never be listed after it is deleted, and nothing has to be kept in sync.

#include "common/engine_capabilities.hpp"
#include "common/models.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mm {

using ServedModelId = std::string;

struct ServedModel {
    /// Public and lowercase (util::is_valid_served_model_id): the string a
    /// client sends as `model`, and the placement holder id.
    ServedModelId id;
    std::string display_name;

    /// IDENTITY of the weights, not their location. Resolved against the
    /// admission registry exactly as an agent's model_path always was: an
    /// admitted model's container directory carries its quantization, so the
    /// identity and the location differ (roadmap D7).
    std::string model_path;

    /// `node`: served on a cluster node, with the engine chosen by routing
    /// (Soma, llama.cpp, vLLM). `api`: a remote OpenAI-compatible provider.
    ///
    /// Agents called the node case "llama-cpp", which stopped being true the
    /// day Soma could serve it — the value names WHERE a request goes, not
    /// which engine answers it.
    std::string inference_backend = "node";

    /// auto | soma | fallback. Persistent on the served model, for the reason
    /// soma::BackendOverride gives.
    std::string backend_override = "auto";

    /// What the engine calls this model when it is not model_path — vLLM's
    /// --served-model-name. Empty means model_path.
    std::string engine_model_name;

    /// Launch settings (engine identity) and the DEFAULT sampling. A
    /// conversation or request may override sampling (SamplingOverrides); none
    /// may override launch settings, because those decide which engine process
    /// serves the request.
    RuntimeSettings runtime_settings;
    ApiSettings api_settings; ///< api_key lives in process memory only
    VisionSettings vision_settings;

    /// The operator's statement that this model can call tools. Necessary, not
    /// sufficient: the engine must also be able to (engine_supports_tools).
    bool tool_calling = true;

    NodeId preferred_node_id;

    /// Disabled entries stay in the catalog, so conversations that name them
    /// fail with a clear `model_unavailable` rather than an unknown id.
    bool enabled = true;

    std::int64_t created_at_ms = 0;
    std::int64_t updated_at_ms = 0;

    /// Exactly `node`. Anything unrecognised is NOT node-local, so a value
    /// nothing validated (an agent's retired "vllm", say) is refused as having
    /// no local backend rather than placed on a guess.
    bool node_local() const noexcept { return inference_backend == "node"; }
};

// SamplingOverrides itself, and its JSON, are in common/models.hpp: a
// Conversation carries one.

/// `base` with every set field of `o` applied. Launch fields are untouched.
RuntimeSettings apply_overrides(RuntimeSettings base, const SamplingOverrides& o);

/// Field by field, `higher` wins where it is set: request over conversation.
SamplingOverrides merge_overrides(const SamplingOverrides& lower,
                                  const SamplingOverrides& higher);

/// Parse overrides from a JSON object, rejecting what cannot be a sampling
/// setting rather than dropping it: a misspelled key that silently did nothing
/// would read as "the model ignored my temperature". Returns false with
/// `out_error` naming the field.
bool parse_sampling_overrides(const nlohmann::json& j,
                              SamplingOverrides& out,
                              std::string& out_error);

struct SamplingProblem {
    std::string field;   ///< "temperature", "top_p", ...
    std::string message;
};

/// Range problems, one per bad field; empty when every set field is in range.
///
/// THE sampling ranges. validate_served_model checks a model's defaults by
/// running them through here (sampling_of), so a conversation override and a
/// model default cannot be held to different rules.
std::vector<SamplingProblem> sampling_override_problems(const SamplingOverrides& o);

/// Every sampling field of `s`, set. Used to validate defaults with the same
/// rules as overrides.
SamplingOverrides sampling_of(const RuntimeSettings& s);

/// The key an API-backed model authenticates with: the one held in process
/// memory, else the environment variable `api_key_env` names, else empty.
std::string resolve_api_key(const ApiSettings& settings);

/// A served model's capabilities on `engine_id`: the engine routing chose, or
/// the one a placement landed on. Each needs the model's own setting AND an
/// engine that can honour it. An API-backed model has no engine here — its
/// provider governs — so `engine_id` is ignored for it.
ModelCapabilities capabilities_for(const ServedModel& m,
                                   const std::string& engine_id,
                                   bool vllm_auto_tool_choice);

// ── JSON ──────────────────────────────────────────────────────────────────────

/// `api_settings.api_key` is never emitted (ApiSettings serializes only whether
/// one is configured).
void to_json(nlohmann::json& j, const ServedModel& m);

/// Every field optional; absent fields keep their defaults. Accepts
/// `api_settings.api_key` for process-local use.
void from_json(const nlohmann::json& j, ServedModel& m);

} // namespace mm
