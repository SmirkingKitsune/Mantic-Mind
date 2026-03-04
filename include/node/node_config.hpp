#pragma once

#include <string>
#include <cstdint>

namespace mm {

struct NodeConfig {
    // Network
    std::string control_url;        // e.g. "http://localhost:9090"
    std::string control_api_key;    // bearer token for control registration
    uint16_t    listen_port = 7070;

    // Subprocess
    std::string llama_server_path;          // full path to llama-server binary
    uint16_t    llama_port = 8080;          // deprecated: single-slot port
    uint16_t    llama_port_range_start = 8080;
    uint16_t    llama_port_range_end   = 8090;
    int         max_slots              = 4;

    // Storage
    std::string models_dir    = "models";  // directory scanned for .gguf files
    std::string data_dir      = "data";
    std::string kv_cache_dir  = "data/kv_cache";

    // Logging
    std::string log_file = "logs/mantic-mind.log";

    // Pairing / discovery
    std::string pairing_key;           // empty = PIN mode; non-empty = PSK mode
    uint16_t    discovery_port = 7072;
};

} // namespace mm
