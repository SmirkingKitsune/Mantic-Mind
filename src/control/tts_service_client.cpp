#include "control/tts_service_client.hpp"

#include "common/http_client.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace mm {

namespace {

TtsServiceResponse disabled_response() {
    return TtsServiceResponse{
        false,
        503,
        "TTS service is disabled",
        {},
        {},
        0,
        0
    };
}

TtsServiceResponse parse_response(int status, const std::string& body) {
    TtsServiceResponse out;
    out.status = status;
    if (status == 0) {
        out.error = "TTS service is unreachable";
        return out;
    }

    if (body.empty()) {
        out.ok = status >= 200 && status < 300;
        if (!out.ok) out.error = "TTS service returned HTTP " + std::to_string(status);
        return out;
    }

    try {
        const auto j = nlohmann::json::parse(body);
        out.ok = j.value("ok", status >= 200 && status < 300);
        out.error = j.value("error", std::string{});
        out.audio_path = j.value("audio_path", std::string{});
        out.voice_clone_prompt_path = j.value("voice_clone_prompt_path", std::string{});
        out.sample_rate = j.value("sample_rate", 0);
        out.duration_ms = j.value("duration_ms", 0);
        if (!out.ok && out.error.empty()) {
            out.error = "TTS service returned HTTP " + std::to_string(status);
        }
    } catch (const std::exception& e) {
        out.ok = false;
        out.error = std::string("invalid TTS service response: ") + e.what();
    }
    return out;
}

TtsServiceResponse post_json(const TtsServiceConfig& config,
                             const std::string& path,
                             const nlohmann::json& body) {
    HttpClient client(config.service_url);
    int status = 0;
    std::string response_body;
    const bool ok = client.stream_post(
        path,
        body,
        [](const std::string&) { return true; },
        &status,
        &response_body);

    auto parsed = parse_response(status, response_body);
    if (!ok && parsed.error.empty()) {
        parsed.ok = false;
        parsed.error = status == 0
            ? "TTS service is unreachable"
            : "TTS service returned HTTP " + std::to_string(status);
    }
    return parsed;
}

} // namespace

TtsServiceClient::TtsServiceClient(TtsServiceConfig config)
    : config_(std::move(config)) {}

const TtsServiceConfig& TtsServiceClient::config() const { return config_; }

bool TtsServiceClient::enabled() const { return config_.enabled; }

std::string TtsServiceClient::backend() const { return "sidecar"; }

std::string TtsServiceClient::provider_name() const { return "qwen3-tts"; }

std::string TtsServiceClient::default_synthesis_model_id() const { return config_.clone_model_id; }

bool TtsServiceClient::health(std::string* error) const {
    if (!config_.enabled) {
        if (error) *error = "TTS service is disabled";
        return false;
    }
    HttpClient client(config_.service_url);
    client.set_timeouts(10, std::max(config_.timeout_s, 1), 10);
    auto res = client.get("/health");
    if (!res.ok()) {
        if (error) {
            *error = res.status == 0
                ? "TTS service is unreachable"
                : "TTS service returned HTTP " + std::to_string(res.status);
        }
        return false;
    }

    try {
        const auto j = nlohmann::json::parse(res.body);
        if (!j.value("ok", true)) {
            if (error) *error = j.value("error", std::string{"TTS service is unhealthy"});
            return false;
        }
    } catch (const std::exception& e) {
        if (error) *error = std::string("invalid TTS health response: ") + e.what();
        return false;
    }
    return true;
}

TtsServiceResponse TtsServiceClient::generate_voice_sample(
    const VoiceDesignProposal& proposal,
    const std::string& output_audio_path,
    const std::string& output_prompt_path) const {
    if (!config_.enabled) return disabled_response();

    nlohmann::json body = {
        {"text", proposal.sample_text},
        {"language", proposal.language.empty() ? "Auto" : proposal.language},
        {"instruct", proposal.voice_description},
        {"voice_description", proposal.voice_description},
        {"output_audio_path", output_audio_path},
        {"output_prompt_path", output_prompt_path},
        {"voice_design_model_id", proposal.voice_design_model_id.empty()
            ? config_.voice_design_model_id
            : proposal.voice_design_model_id},
        {"clone_model_id", proposal.clone_model_id.empty()
            ? config_.clone_model_id
            : proposal.clone_model_id}
    };

    return post_json(config_, "/voice-design", body);
}

TtsServiceResponse TtsServiceClient::synthesize(
    const TtsSynthesisRequest& request,
    const AgentVoiceProfile& profile,
    const std::string& output_audio_path) const {
    if (!config_.enabled) return disabled_response();

    nlohmann::json body = {
        {"text", request.text},
        {"language", request.language.empty() ? profile.language : request.language},
        {"voice_clone_prompt_path", profile.voice_clone_prompt_path},
        {"output_audio_path", output_audio_path},
        {"format", request.format.empty() ? "wav" : request.format},
        {"clone_model_id", profile.clone_model_id.empty()
            ? config_.clone_model_id
            : profile.clone_model_id}
    };

    return post_json(config_, "/synthesize", body);
}

} // namespace mm
