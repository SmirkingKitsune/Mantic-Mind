#include "control/tts_service_client.hpp"

#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <utility>

namespace mm {

namespace {

std::string normalized_backend(const TtsServiceConfig& config) {
    std::string backend = util::to_lower(util::trim(config.backend));
    if (backend.empty()) backend = "sidecar";
    return backend;
}

bool is_sidecar_backend(const TtsServiceConfig& config) {
    const std::string backend = normalized_backend(config);
    return backend == "sidecar" || backend == "qwen3-tts";
}

bool is_vllm_backend(const TtsServiceConfig& config) {
    return normalized_backend(config) == "vllm";
}

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

TtsServiceResponse unsupported_backend_response(const TtsServiceConfig& config) {
    TtsServiceResponse out;
    out.status = 400;
    out.error = "unsupported TTS backend: " + normalized_backend(config);
    return out;
}

std::string normalized_path(const std::string& path, const std::string& fallback) {
    std::string p = util::trim(path);
    if (p.empty()) p = fallback;
    if (!p.empty() && p.front() != '/') p.insert(p.begin(), '/');
    return p;
}

std::string resolved_vllm_model_id(const TtsServiceConfig& config) {
    std::string model = util::trim(config.vllm_model_id);
    if (!model.empty()) return model;
    model = util::trim(config.custom_voice_model_id);
    if (!model.empty()) return model;
    return util::trim(config.clone_model_id);
}

std::string resolved_vllm_api_key(const TtsServiceConfig& config) {
    std::string api_key = util::trim(config.vllm_api_key);
    if (!api_key.empty()) return api_key;
    const std::string env_name = util::trim(config.vllm_api_key_env);
    if (env_name.empty()) return {};
    const char* value = std::getenv(env_name.c_str());
    return value ? util::trim(value) : std::string{};
}

bool ensure_parent_dir(const std::string& path, std::string* error) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) return true;
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (!ec) return true;
    if (error) *error = ec.message();
    return false;
}

bool write_file(const std::string& path, const std::string& data, std::string* error) {
    if (!ensure_parent_dir(path, error)) return false;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (error) *error = "failed to open " + path;
        return false;
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out.good()) {
        if (error) *error = "failed to write " + path;
        return false;
    }
    return true;
}

std::optional<nlohmann::json> load_json_file(const std::string& path) {
    if (path.empty()) return std::nullopt;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return std::nullopt;
    try {
        nlohmann::json j;
        in >> j;
        if (!j.is_object()) return std::nullopt;
        return j;
    } catch (...) {
        return std::nullopt;
    }
}

std::string first_non_empty(std::initializer_list<std::string> values) {
    for (const auto& value : values) {
        std::string trimmed = util::trim(value);
        if (!trimmed.empty()) return trimmed;
    }
    return {};
}

std::optional<std::string> decode_base64(const std::string& input, std::string* error) {
    static const std::array<int, 256> table = [] {
        std::array<int, 256> t{};
        t.fill(-1);
        for (int i = 0; i < 26; ++i) {
            t[static_cast<unsigned char>('A' + i)] = i;
            t[static_cast<unsigned char>('a' + i)] = 26 + i;
        }
        for (int i = 0; i < 10; ++i) {
            t[static_cast<unsigned char>('0' + i)] = 52 + i;
        }
        t[static_cast<unsigned char>('+')] = 62;
        t[static_cast<unsigned char>('/')] = 63;
        return t;
    }();

    std::string out;
    int val = 0;
    int valb = -8;
    for (unsigned char c : input) {
        if (std::isspace(c)) continue;
        if (c == '=') break;
        const int decoded = table[c];
        if (decoded < 0) {
            if (error) *error = "invalid base64 audio payload";
            return std::nullopt;
        }
        val = (val << 6) + decoded;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

std::optional<std::string> json_string_at(const nlohmann::json& j, const char* key) {
    auto it = j.find(key);
    if (it != j.end() && it->is_string()) return it->get<std::string>();
    return std::nullopt;
}

std::optional<std::string> find_audio_base64(const nlohmann::json& j) {
    for (const char* key : {"audio_base64", "b64_json", "audio"}) {
        auto value = json_string_at(j, key);
        if (value && !value->empty()) return value;
    }
    auto data_it = j.find("data");
    if (data_it == j.end()) return std::nullopt;
    if (data_it->is_string()) return data_it->get<std::string>();
    if (data_it->is_array() && !data_it->empty() && data_it->front().is_object()) {
        for (const char* key : {"b64_json", "audio_base64", "audio"}) {
            auto value = json_string_at(data_it->front(), key);
            if (value && !value->empty()) return value;
        }
    }
    return std::nullopt;
}

std::string json_error_message(const nlohmann::json& j) {
    auto error_it = j.find("error");
    if (error_it == j.end()) return {};
    if (error_it->is_string()) return error_it->get<std::string>();
    if (error_it->is_object()) {
        if (auto message = json_string_at(*error_it, "message")) return *message;
        if (auto type = json_string_at(*error_it, "type")) return *type;
    }
    return error_it->dump();
}

TtsServiceResponse written_audio_response(int status,
                                          const std::string& output_audio_path,
                                          int sample_rate = 0,
                                          int duration_ms = 0) {
    TtsServiceResponse out;
    out.ok = true;
    out.status = status;
    out.audio_path = output_audio_path;
    out.sample_rate = sample_rate;
    out.duration_ms = duration_ms;
    return out;
}

TtsServiceResponse parse_vllm_speech_response(int status,
                                              const std::string& body,
                                              const std::string& output_audio_path) {
    TtsServiceResponse out;
    out.status = status;
    if (status == 0) {
        out.error = "vLLM TTS service is unreachable";
        return out;
    }

    const bool http_ok = status >= 200 && status < 300;
    if (body.empty()) {
        out.ok = false;
        out.error = http_ok
            ? "vLLM TTS service returned an empty audio response"
            : "vLLM TTS service returned HTTP " + std::to_string(status);
        return out;
    }

    try {
        const auto j = nlohmann::json::parse(body);
        const std::string response_error = json_error_message(j);
        if (!http_ok || (!response_error.empty() && !j.value("ok", http_ok))) {
            out.ok = false;
            out.error = response_error.empty()
                ? "vLLM TTS service returned HTTP " + std::to_string(status)
                : response_error;
            return out;
        }

        if (auto audio_path = json_string_at(j, "audio_path")) {
            out.ok = true;
            out.audio_path = *audio_path;
            out.sample_rate = j.value("sample_rate", 0);
            out.duration_ms = j.value("duration_ms", 0);
            return out;
        }

        if (auto encoded = find_audio_base64(j)) {
            std::string decode_error;
            auto decoded = decode_base64(*encoded, &decode_error);
            if (!decoded) {
                out.ok = false;
                out.error = decode_error;
                return out;
            }
            std::string write_error;
            if (!write_file(output_audio_path, *decoded, &write_error)) {
                out.ok = false;
                out.status = 500;
                out.error = write_error;
                return out;
            }
            return written_audio_response(status,
                                          output_audio_path,
                                          j.value("sample_rate", 0),
                                          j.value("duration_ms", 0));
        }

        out.ok = false;
        out.error = "vLLM TTS response did not include audio";
        return out;
    } catch (const std::exception&) {
        if (!http_ok) {
            out.ok = false;
            out.error = "vLLM TTS service returned HTTP " + std::to_string(status);
            return out;
        }

        std::string write_error;
        if (!write_file(output_audio_path, body, &write_error)) {
            out.ok = false;
            out.status = 500;
            out.error = write_error;
            return out;
        }
        return written_audio_response(status, output_audio_path);
    }
}

nlohmann::json build_vllm_speech_body(const TtsServiceConfig& config,
                                      const std::string& text,
                                      const std::string& format,
                                      const std::string& voice_description,
                                      const std::string& display_name) {
    const std::string voice = first_non_empty({
        voice_description,
        display_name,
        std::string{"default"}
    });

    return nlohmann::json{
        {"model", resolved_vllm_model_id(config)},
        {"input", text},
        {"voice", voice},
        {"response_format", format.empty() ? "wav" : format}
    };
}

TtsServiceResponse post_vllm_speech(const TtsServiceConfig& config,
                                    const nlohmann::json& body,
                                    const std::string& output_audio_path) {
    HttpClient client(config.vllm_base_url);
    client.set_timeouts(10, std::max(config.timeout_s, 1), 30);
    const std::string api_key = resolved_vllm_api_key(config);
    if (!api_key.empty()) client.set_bearer_token(api_key);

    auto res = client.post(normalized_path(config.vllm_speech_path, "/v1/audio/speech"), body);
    return parse_vllm_speech_response(res.status, res.body, output_audio_path);
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

std::string TtsServiceClient::backend() const { return normalized_backend(config_); }

std::string TtsServiceClient::provider_name() const {
    return is_vllm_backend(config_) ? "qwen3-tts-vllm" : "qwen3-tts";
}

std::string TtsServiceClient::default_synthesis_model_id() const {
    return is_vllm_backend(config_) ? resolved_vllm_model_id(config_) : config_.clone_model_id;
}

bool TtsServiceClient::health(std::string* error) const {
    if (!config_.enabled) {
        if (error) *error = "TTS service is disabled";
        return false;
    }
    if (!is_sidecar_backend(config_) && !is_vllm_backend(config_)) {
        if (error) *error = "unsupported TTS backend: " + normalized_backend(config_);
        return false;
    }

    HttpClient client(is_vllm_backend(config_) ? config_.vllm_base_url : config_.service_url);
    client.set_timeouts(10, std::max(config_.timeout_s, 1), 10);
    const std::string api_key = resolved_vllm_api_key(config_);
    if (is_vllm_backend(config_) && !api_key.empty()) client.set_bearer_token(api_key);
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
        if (is_vllm_backend(config_) && util::trim(res.body).empty()) return true;
        const auto j = nlohmann::json::parse(res.body);
        if (!j.value("ok", true)) {
            if (error) *error = j.value("error", std::string{"TTS service is unhealthy"});
            return false;
        }
    } catch (const std::exception& e) {
        if (is_vllm_backend(config_)) return true;
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

    if (is_vllm_backend(config_)) {
        const std::string format = "wav";
        auto body = build_vllm_speech_body(config_,
                                           proposal.sample_text,
                                           format,
                                           proposal.voice_description,
                                           proposal.display_name);
        auto svc = post_vllm_speech(config_, body, output_audio_path);
        if (!svc.ok) return svc;

        const nlohmann::json descriptor = {
            {"backend", "vllm"},
            {"provider", "qwen3-tts-vllm"},
            {"model", body.value("model", std::string{})},
            {"display_name", proposal.display_name},
            {"language", proposal.language.empty() ? "Auto" : proposal.language},
            {"voice_description", proposal.voice_description},
            {"sample_text", proposal.sample_text},
            {"created_at_ms", util::now_ms()}
        };
        std::string write_error;
        if (!write_file(output_prompt_path, descriptor.dump(2), &write_error)) {
            TtsServiceResponse out;
            out.status = 500;
            out.error = write_error;
            return out;
        }
        if (svc.audio_path.empty()) svc.audio_path = output_audio_path;
        svc.voice_clone_prompt_path = output_prompt_path;
        return svc;
    }
    if (!is_sidecar_backend(config_)) return unsupported_backend_response(config_);

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

    if (is_vllm_backend(config_)) {
        std::string voice_description = profile.voice_description;
        std::string display_name = profile.display_name;
        if (auto descriptor = load_json_file(profile.voice_clone_prompt_path)) {
            voice_description = first_non_empty({
                descriptor->value("voice_description", std::string{}),
                voice_description
            });
            display_name = first_non_empty({
                descriptor->value("display_name", std::string{}),
                display_name
            });
        }
        const std::string format = request.format.empty() ? "wav" : request.format;
        auto body = build_vllm_speech_body(config_,
                                           request.text,
                                           format,
                                           voice_description,
                                           display_name);
        return post_vllm_speech(config_, body, output_audio_path);
    }
    if (!is_sidecar_backend(config_)) return unsupported_backend_response(config_);

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
