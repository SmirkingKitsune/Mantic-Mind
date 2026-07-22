#pragma once

#include "common/models.hpp"

#include <cstdint>
#include <string>

namespace mm {

// Rough VRAM/host-memory estimate for a llama.cpp (llama-server) load. Anchored
// to the on-disk GGUF weight size plus a context-scaled KV cache term rather
// than filename parameter guesses, so it stays backend-honest without parsing
// architecture tensors. Used by the node and scheduler to size llama.cpp slots.
struct InferenceMemoryEstimate {
    int64_t model_weight_mb = 0;
    int64_t context_cache_mb = 0;
    int64_t overhead_mb = 0;
    int64_t total_vram_mb = 0;
    int     effective_ctx_tokens = 0;
    int     parallel = 1;
    std::string source_path;
};

// llama-server sizes one shared context of ctx_size*parallel tokens. These
// resolve the effective values, honoring any --parallel / --ctx-size overrides
// the caller placed in RuntimeSettings::extra_args.
int effective_llama_parallel(const RuntimeSettings& settings);
int effective_llama_server_ctx_tokens(const RuntimeSettings& settings);

InferenceMemoryEstimate estimate_inference_memory(const std::string& model_path,
                                                  const RuntimeSettings& settings,
                                                  const std::string& models_dir = {});

int64_t estimate_inference_vram_mb(const std::string& model_path,
                                   const RuntimeSettings& settings,
                                   const std::string& models_dir = {});

} // namespace mm
