#pragma once

#include "control/tts_service_client.hpp"

#include <string>
#include <cstdint>

namespace mm {

struct ControlConfig {
    uint16_t    listen_port = 9090;
    uint16_t    openai_compat_port = 9091; // 0 disables the compatibility listener
    std::string data_dir    = "data";
    std::string log_file    = "logs/mantic-mind-control.log";
    uint32_t    node_health_poll_interval_s = 30;

    // Model distribution
    std::string models_dir  = "models";

    // Optional bearer token required by external /v1/* client routes.
    std::string external_api_token;

    // Optional Qwen3-TTS backend. Disabled by default; when enabled but
    // unreachable, public TTS routes return 503 JSON errors.
    TtsServiceConfig tts;

    // Pairing / discovery
    std::string pairing_key;           // must match node's pairing_key for PSK mode
    uint16_t    discovery_port = 7072;
};

} // namespace mm
