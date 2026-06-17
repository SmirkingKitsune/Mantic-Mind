#pragma once

#include <string>
#include <vector>

namespace mm {

// Node-side Ray cluster orchestration for multi-node vLLM engine groups.
//
// A spanning vLLM engine runs as one `vllm serve --pipeline-parallel-size N`
// on a *head* node that sits atop a Ray cluster; the other group members join
// that cluster as Ray *workers* and contribute their GPUs. Control decides the
// group membership (see control/engine_group_planner) and then drives this:
//   1. head node:   ray_start({Head, port, num_gpus})
//   2. each worker: ray_start({Worker, head_address, num_gpus})
//   3. head node:   a normal load-model with tensor/pipeline parallel set —
//                   vLLM discovers the Ray cluster automatically.
//
// SCAFFOLDING: the live launch is gated to Linux (ray_supported()). Ray's
// multi-node transport relies on NCCL/Gloo which do not run on Windows, so on
// Windows these calls return a clear "unsupported" error rather than spawning.
// The argument builder is pure and unit-tested; the exec path needs a real
// Linux multi-node cluster to validate end to end.

enum class RayRole { Head, Worker };

struct RayStartConfig {
    RayRole     role     = RayRole::Head;
    std::string ray_path = "ray";   // executable or PATH name
    std::string head_address;       // worker: "ip:port" of the head's GCS
    int         port     = 6379;    // head: GCS port to bind
    int         num_gpus = 0;       // 0 = let Ray auto-detect
};

// Build the `ray start ...` argument vector (excludes the executable itself,
// mirroring build_vllm_server_args). Pure — no process is spawned.
std::vector<std::string> build_ray_start_args(const RayStartConfig& cfg);

// True when this platform can host a Ray cluster member (Linux only today).
bool ray_supported();

// Start (or join) a Ray cluster per cfg. Blocks until `ray start` returns —
// the Ray daemons keep running in the background afterward. Returns false and
// sets *error_out on failure or when ray_supported() is false.
bool ray_start(const RayStartConfig& cfg, std::string* error_out);

// Tear down this node's Ray daemons (`ray stop`).
bool ray_stop(const std::string& ray_path, std::string* error_out);

} // namespace mm
