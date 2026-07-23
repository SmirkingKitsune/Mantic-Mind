#pragma once

#include "common/models.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mm {

inline constexpr const char* kLlamaCppRepoUrl =
    "https://github.com/ggml-org/llama.cpp";

// Generic host probes shared by llama.cpp runtime selection and provisioning.
std::string current_runtime_platform();
std::string current_runtime_arch();
bool detect_rocm_present();

// NVIDIA host probes shared by standalone and AIO node lifecycles. Parsing is
// exposed separately so platform-independent tests can cover multi-GPU,
// duplicate architecture, and unified-memory fallback behavior without
// requiring nvidia-smi on the test host.
int parse_nvidia_gpu_count(std::string_view output);
std::string parse_cuda_architectures(std::string_view output);
int detect_nvidia_gpu_count();
std::string detect_cuda_architectures();

// Configured values are authoritative. Otherwise use the full nvidia-smi
// count, falling back to one when the metrics probe observed an NVIDIA backend
// whose unified-memory VRAM fields could not be parsed.
int resolve_node_gpu_count(int configured_count,
                           int detected_nvidia_count,
                           bool nvidia_backend_observed);

// Translate a model path for the node's runtime OS. When the node runs Windows
// and control handed it a WSL mount path (/mnt/y/foo.gguf), map it to Y:\foo.gguf
// and vice-versa on POSIX. Also strips wrapping quotes. Pure aside from an
// existence probe on the POSIX translation.
std::string normalize_llama_model_path(const std::string& p);

// Build the `llama-server` argument vector (excludes the executable itself).
// Pure and unit-tested. llama-server hosts
// one shared context of ctx_size*parallel tokens. User-supplied
// RuntimeSettings::extra_args may override tuning defaults, but node-managed
// model/projector, host, port, and KV-save arguments are rejected. When
// slot_save_path is non-empty, `--slot-save-path <dir>` is added so the engine's
// /slots/{id}?action=save|restore KV endpoints are available for suspension.
std::vector<std::string> build_llama_server_args(const std::string& model_path,
                                                 const std::string& mmproj_path,
                                                 const RuntimeSettings& settings,
                                                 uint16_t port,
                                                 const std::string& slot_save_path = {});

inline std::vector<std::string> build_llama_server_args(
    const std::string& model_path,
    const RuntimeSettings& settings,
    uint16_t port,
    const std::string& slot_save_path = {}) {
    return build_llama_server_args(model_path, {}, settings, port, slot_save_path);
}

// Accelerator-correct llama.cpp build variant for this environment:
//   cuda   — NVIDIA/CUDA GPU present (Linux or Windows; includes DGX Spark GB10)
//   rocm   — AMD/ROCm GPU present (Linux)
//   metal  — Apple Silicon macOS
//   vulkan — a GPU is present but neither CUDA nor ROCm (cross-vendor fallback)
//   cpu    — no GPU backend
// llama.cpp builds natively on Windows with CUDA, so there is no separate
// "windows" variant. Pure.
std::string detect_llama_accelerator(const std::string& platform,
                                     const std::string& arch,
                                     bool has_cuda,
                                     bool has_rocm);

} // namespace mm
