#pragma once

#include "common/models.hpp"

#include <optional>
#include <string>

namespace mm {

struct TtsServiceConfig {
    bool        enabled = false;
    std::string backend = "sidecar"; // sidecar|vllm
    std::string service_url = "http://127.0.0.1:9188";
    std::string service_command;
    std::string cache_dir;
    std::string voice_design_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign";
    std::string clone_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base";
    std::string custom_voice_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";
    std::string vllm_base_url = "http://127.0.0.1:8000";
    std::string vllm_speech_path = "/v1/audio/speech";
    std::string vllm_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";
    std::string vllm_api_key;
    std::string vllm_api_key_env = "MM_TTS_VLLM_API_KEY";
    int64_t     cache_ttl_ms = 24LL * 60LL * 60LL * 1000LL;
    int         timeout_s = 120;
};

struct TtsServiceResponse {
    bool        ok = false;
    int         status = 0;
    std::string error;
    std::string audio_path;
    std::string voice_clone_prompt_path;
    int         sample_rate = 0;
    int         duration_ms = 0;
};

class TtsServiceClient {
public:
    explicit TtsServiceClient(TtsServiceConfig config = {});

    const TtsServiceConfig& config() const;
    bool enabled() const;
    std::string backend() const;
    std::string provider_name() const;
    std::string default_synthesis_model_id() const;

    bool health(std::string* error = nullptr) const;

    TtsServiceResponse generate_voice_sample(const VoiceDesignProposal& proposal,
                                             const std::string& output_audio_path,
                                             const std::string& output_prompt_path) const;

    TtsServiceResponse synthesize(const TtsSynthesisRequest& request,
                                  const AgentVoiceProfile& profile,
                                  const std::string& output_audio_path) const;

private:
    TtsServiceConfig config_;
};

} // namespace mm
