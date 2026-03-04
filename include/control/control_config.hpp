#pragma once

#include <string>
#include <cstdint>

namespace mm {

struct ControlConfig {
    uint16_t    listen_port = 9090;
    std::string data_dir    = "data";
    std::string log_file    = "logs/mantic-mind-control.log";
    uint32_t    node_health_poll_interval_s = 30;

    // Model distribution
    std::string models_dir  = "models";

    // Pairing / discovery
    std::string pairing_key;           // must match node's pairing_key for PSK mode
    uint16_t    discovery_port = 7072;
};

} // namespace mm
