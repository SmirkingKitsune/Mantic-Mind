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

    // Subprocess — vLLM peer backend. llama.cpp is the default runtime on this
    // branch (vLLM-as-default is preserved on the vLLM-runtime branch), so a
    // vLLM already on PATH is used but a missing one is NOT auto-installed
    // unless vllm_auto_provision is enabled.
    std::string vllm_server_path = "vllm";  // vLLM CLI executable or wrapper
    double      vllm_gpu_budget  = 0.90;    // total GPU fraction all vLLM slots may claim
    bool        vllm_auto_provision = false;
    std::string vllm_provision_dir;
    std::string vllm_install_method = "auto"; // auto|wheel|source
    std::string vllm_version = "latest";
    std::string vllm_python_path;

    // llama.cpp (llama-server) peer backend. Serves GGUF models and accelerators
    // without a vLLM build (e.g. DGX Spark). Provisioning mirrors vLLM's.
    std::string llama_server_path = "llama-server"; // llama-server executable
    bool        llama_auto_provision = true;
    std::string llama_provision_dir;
    std::string llama_install_method = "auto"; // auto|release|source
    std::string llama_version = "latest";
    std::string llama_accelerator;             // cuda|rocm|vulkan|metal|cpu; "" = auto
    std::string llama_cuda_arch;               // e.g. "121" for DGX Spark GB10; "" = default
    std::vector<std::string> llama_cmake_args; // extra -D flags for the source build
    // Runtime update management. Missing runtimes are always bootstrapped (per
    // vllm_auto_provision); these govern *updating* an existing install.
    std::string vllm_update_policy = "prompt";        // prompt|auto|manual
    bool        vllm_update_check  = true;            // run periodic online checks
    int         vllm_update_check_interval_hours = 24;

    // ── Cluster capabilities (multi-node vLLM engine groups) ─────────────────
    // Empty/auto values are filled by runtime detection at startup. Set these
    // to override what this node advertises to control.
    std::string comm_backends;             // CSV e.g. "nccl,gloo"; "" = auto-detect
    bool        supports_ray = false;      // multi-node Ray group membership
    bool        supports_ray_set = false;  // true when supports_ray came from config
    int         node_gpu_count = 0;        // 0 = auto (nvidia-smi)
    double      interconnect_gbps = 0.0;   // node-to-node link bandwidth hint
    std::string ray_path = "ray";          // Ray CLI executable or PATH name
    uint16_t    ray_port = 6379;           // head GCS port for Ray clusters
    std::string hf_cli_path = "hf";        // Hugging Face CLI for model pre-fetch
    std::string hf_cache_dir;              // HF hub cache dir override; "" = auto (HF_HOME/default)
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
