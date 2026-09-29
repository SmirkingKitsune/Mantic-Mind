#include "common/served_model.hpp"

#include "common/util.hpp"

#include <cmath>
#include <cstdlib>

namespace mm {

RuntimeSettings apply_overrides(RuntimeSettings base, const SamplingOverrides& o) {
    if (o.temperature) base.temperature = *o.temperature;
    if (o.top_p) base.top_p = *o.top_p;
    if (o.top_k) base.top_k = *o.top_k;
    if (o.min_p) base.min_p = *o.min_p;
    if (o.presence_penalty) base.presence_penalty = *o.presence_penalty;
    if (o.repeat_penalty) base.repeat_penalty = *o.repeat_penalty;
    if (o.max_tokens) base.max_tokens = *o.max_tokens;
    return base;
}

SamplingOverrides merge_overrides(const SamplingOverrides& lower,
                                  const SamplingOverrides& higher) {
    SamplingOverrides out = lower;
    if (higher.temperature) out.temperature = higher.temperature;
    if (higher.top_p) out.top_p = higher.top_p;
    if (higher.top_k) out.top_k = higher.top_k;
    if (higher.min_p) out.min_p = higher.min_p;
    if (higher.presence_penalty) out.presence_penalty = higher.presence_penalty;
    if (higher.repeat_penalty) out.repeat_penalty = higher.repeat_penalty;
    if (higher.max_tokens) out.max_tokens = higher.max_tokens;
    return out;
}

bool parse_sampling_overrides(const nlohmann::json& j,
                              SamplingOverrides& out,
                              std::string& out_error) {
    out = {};
    if (j.is_null()) return true;
    if (!j.is_object()) {
        out_error = "settings must be an object";
        return false;
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& key = it.key();
        const auto& v = it.value();
        // An explicit null clears a stored override; it is not an error.
        if (v.is_null()) continue;

        const auto real = [&](std::optional<float>& dst) {
            if (!v.is_number()) {
                out_error = "settings." + key + " must be a number";
                return false;
            }
            dst = v.get<float>();
            return true;
        };
        const auto integer = [&](std::optional<int>& dst) {
            if (!v.is_number_integer()) {
                out_error = "settings." + key + " must be an integer";
                return false;
            }
            dst = v.get<int>();
            return true;
        };

        bool ok = false;
        if (key == "temperature") ok = real(out.temperature);
        else if (key == "top_p") ok = real(out.top_p);
        else if (key == "top_k") ok = integer(out.top_k);
        else if (key == "min_p") ok = real(out.min_p);
        else if (key == "presence_penalty") ok = real(out.presence_penalty);
        else if (key == "repeat_penalty") ok = real(out.repeat_penalty);
        else if (key == "max_tokens") ok = integer(out.max_tokens);
        else {
            out_error = "settings." + key +
                        " is not a sampling setting (expected temperature, top_p, top_k, "
                        "min_p, presence_penalty, repeat_penalty or max_tokens)";
            return false;
        }
        if (!ok) return false;
    }
    return true;
}

std::vector<SamplingProblem> sampling_override_problems(const SamplingOverrides& o) {
    std::vector<SamplingProblem> out;
    const auto bad = [&](const char* field, const char* message) {
        out.push_back({field, message});
    };
    if (o.temperature && (*o.temperature < 0.0f || !std::isfinite(*o.temperature)))
        bad("temperature", "temperature must be a finite value >= 0.");
    if (o.top_p && (*o.top_p <= 0.0f || *o.top_p > 1.0f || !std::isfinite(*o.top_p)))
        bad("top_p", "top_p must be within (0, 1].");
    if (o.top_k && *o.top_k < -1)
        bad("top_k", "top_k must be -1 (runtime default), 0 (disabled), or greater.");
    if (o.min_p &&
        ((*o.min_p < 0.0f && *o.min_p != -1.0f) || *o.min_p > 1.0f || !std::isfinite(*o.min_p)))
        bad("min_p", "min_p must be -1 (runtime default) or within [0, 1].");
    if (o.presence_penalty && (*o.presence_penalty < -2.0f || *o.presence_penalty > 2.0f ||
                               !std::isfinite(*o.presence_penalty)))
        bad("presence_penalty", "presence_penalty must be within [-2, 2].");
    if (o.repeat_penalty && ((*o.repeat_penalty <= 0.0f && *o.repeat_penalty != -1.0f) ||
                             !std::isfinite(*o.repeat_penalty)))
        bad("repeat_penalty", "repeat_penalty must be -1 (runtime default) or greater than 0.");
    if (o.max_tokens && *o.max_tokens <= 0 && *o.max_tokens != -1)
        bad("max_tokens", "max_tokens must be greater than 0, or -1 for unlimited generation.");
    return out;
}

SamplingOverrides sampling_of(const RuntimeSettings& s) {
    SamplingOverrides o;
    o.temperature = s.temperature;
    o.top_p = s.top_p;
    o.top_k = s.top_k;
    o.min_p = s.min_p;
    o.presence_penalty = s.presence_penalty;
    o.repeat_penalty = s.repeat_penalty;
    o.max_tokens = s.max_tokens;
    return o;
}

std::string resolve_api_key(const ApiSettings& settings) {
    if (!settings.api_key.empty()) return settings.api_key;
    const std::string env_name = util::trim(settings.api_key_env);
    if (env_name.empty()) return {};
    const char* value = std::getenv(env_name.c_str());
    return value ? std::string(value) : std::string{};
}

ModelCapabilities capabilities_for(const ServedModel& m,
                                   const std::string& engine_id,
                                   bool vllm_auto_tool_choice) {
    ModelCapabilities caps;
    caps.vision = m.vision_settings.enabled;
    caps.tools = m.tool_calling;
    if (m.node_local()) {
        caps.vision = caps.vision && engine_supports_vision(engine_id);
        caps.tools = caps.tools && engine_supports_tools(engine_id, vllm_auto_tool_choice);
    }
    return caps;
}

void to_json(nlohmann::json& j, const ServedModel& m) {
    j = {{"id", m.id},
         {"display_name", m.display_name},
         {"model_path", m.model_path},
         {"inference_backend", m.inference_backend},
         {"backend_override", m.backend_override},
         {"engine_model_name", m.engine_model_name},
         {"runtime_settings", m.runtime_settings},
         {"api_settings", m.api_settings},
         {"vision_settings", m.vision_settings},
         {"tool_calling", m.tool_calling},
         {"preferred_node_id", m.preferred_node_id},
         {"enabled", m.enabled},
         {"created_at_ms", m.created_at_ms},
         {"updated_at_ms", m.updated_at_ms}};
}

void from_json(const nlohmann::json& j, ServedModel& m) {
    if (j.contains("id")) j.at("id").get_to(m.id);
    if (j.contains("display_name")) j.at("display_name").get_to(m.display_name);
    if (j.contains("model_path")) j.at("model_path").get_to(m.model_path);
    if (j.contains("inference_backend")) j.at("inference_backend").get_to(m.inference_backend);
    if (j.contains("backend_override")) j.at("backend_override").get_to(m.backend_override);
    if (j.contains("engine_model_name")) j.at("engine_model_name").get_to(m.engine_model_name);
    if (j.contains("runtime_settings")) j.at("runtime_settings").get_to(m.runtime_settings);
    if (j.contains("api_settings")) j.at("api_settings").get_to(m.api_settings);
    if (j.contains("vision_settings")) j.at("vision_settings").get_to(m.vision_settings);
    if (j.contains("tool_calling")) j.at("tool_calling").get_to(m.tool_calling);
    if (j.contains("preferred_node_id")) j.at("preferred_node_id").get_to(m.preferred_node_id);
    if (j.contains("enabled")) j.at("enabled").get_to(m.enabled);
    if (j.contains("created_at_ms")) j.at("created_at_ms").get_to(m.created_at_ms);
    if (j.contains("updated_at_ms")) j.at("updated_at_ms").get_to(m.updated_at_ms);
}

} // namespace mm
