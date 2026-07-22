#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace mm {

struct NodeConfig {
    // Network
    std::string control_url;        // e.g. "http://localhost:9090"
    std::string control_api_key;    // bearer token for control registration
    uint16_t    listen_port = 7070;

    // llama.cpp (llama-server) runtime. Auto provisioning prefers a matching
    // official release and builds from source only as a fallback.
    std::string llama_server_path = "llama-server"; // llama-server executable
    bool        llama_auto_provision = true;
    std::string llama_provision_dir;
    std::string llama_install_method = "auto"; // auto|release|source
    std::string llama_version = "latest";
    std::string llama_accelerator;             // cuda|rocm|hip|vulkan|openvino|sycl-*|metal|cpu; "" = auto
    std::string llama_cuda_arch;               // e.g. "121" for DGX Spark GB10; "" = auto
    std::vector<std::string> llama_cmake_args; // extra -D flags for the source build
    int         llama_build_jobs = 0;          // 0 = conservative accelerator-aware default
    // Runtime update management. Missing runtimes are bootstrapped according to
    // llama_auto_provision; these govern updating an existing install.
    std::string llama_update_policy = "prompt";       // prompt|auto|manual
    bool        llama_update_check = true;
    int         llama_update_check_interval_hours = 24;

    // Hardware capability override; zero lets startup detect visible GPUs.
    int         node_gpu_count = 0;        // 0 = auto (nvidia-smi)
    uint16_t    runtime_port_range_start = 8080;
    uint16_t    runtime_port_range_end   = 8090;
    int         max_slots              = 4;

    // Storage
    std::string models_dir    = "models";  // local model cache root (control transfers land here)
    std::string data_dir      = "data";
    std::string kv_cache_dir  = "data/kv_cache";

    // Local model cache management. Control transfers models into models_dir;
    // the node keeps an LRU use-queue and evicts unpinned models under disk
    // pressure. Pinned models (control marked this node preferred) are kept.
    int64_t     model_cache_min_free_mb = 10240;      // evict LRU when free < this (0 = off)
    bool        model_cache_clear_on_shutdown = true; // drop unpinned models on exit

    // Logging
    std::string log_file = "logs/mantic-mind.log";

    // Pairing / discovery
    std::string pairing_key;           // empty = PIN mode; non-empty = PSK mode
    uint16_t    discovery_port = 7072;
};

} // namespace mm
