#pragma once

#include "common/models.hpp"

#include <cstdint>
#include <string>

namespace mm {

struct InferenceMemoryEstimate {
    int64_t model_weight_mb = 0;
    int64_t context_cache_mb = 0;
    int64_t overhead_mb = 0;
    int64_t total_vram_mb = 0;
    int     effective_ctx_tokens = 0;
    int     parallel = 1;
    std::string source_path;
};

int effective_llama_parallel(const LlamaSettings& settings);
int effective_llama_server_ctx_tokens(const LlamaSettings& settings);

InferenceMemoryEstimate estimate_inference_memory(const std::string& model_path,
                                                  const LlamaSettings& settings,
                                                  const std::string& models_dir = {});

int64_t estimate_inference_vram_mb(const std::string& model_path,
                                   const LlamaSettings& settings,
                                   const std::string& models_dir = {});

} // namespace mm
