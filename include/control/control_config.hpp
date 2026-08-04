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
    uint32_t    node_offline_after_s = 90;

    // Model distribution
    std::string models_dir  = "models";

    // ── admission ─────────────────────────────────────────────────────────────
    // Converting a model runs the Python tools in tools/admission/ as
    // subprocesses. They are a dependency of ADMITTING, which happens once per
    // model — never of serving.
    std::string admission_python     = "python";           // MM_ADMISSION_PYTHON
    std::string admission_tools_dir  = "tools/admission";  // MM_ADMISSION_TOOLS
    std::string admission_soma_path  = "soma";             // MM_SOMA_PATH
    std::string containers_dir       = "data/containers";  // MM_CONTAINERS_DIR
    std::string admission_quant      = "q4_g";             // MM_ADMISSION_QUANT
    std::string admission_expert_down = "q6_g";

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
