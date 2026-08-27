#include "common/inference_sizing.hpp"

#include "common/footprint.hpp"
#include "common/gguf_metadata.hpp"
#include "common/util.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <optional>
#include <vector>

namespace fs = std::filesystem;

namespace mm {

namespace {

constexpr int64_t kFallbackModelMb = 2048;

bool arg_matches_flag(const std::string& raw, const std::string& flag) {
    const std::string arg = mm::util::trim(raw);
    return arg == flag ||
           arg.rfind(flag + "=", 0) == 0 ||
           arg.rfind(flag + " ", 0) == 0;
}

std::optional<std::string> find_flag_value(const std::vector<std::string>& extra,
                                           std::initializer_list<const char*> flags) {
    for (size_t i = 0; i < extra.size(); ++i) {
        const std::string arg = mm::util::trim(extra[i]);
        for (const char* flag_raw : flags) {
            const std::string flag(flag_raw);
            if (arg == flag) {
                if (i + 1 < extra.size()) return mm::util::trim(extra[i + 1]);
                return std::nullopt;
            }
            if (arg.rfind(flag + "=", 0) == 0) {
                return mm::util::trim(arg.substr(flag.size() + 1));
            }
            if (arg.rfind(flag + " ", 0) == 0) {
                return mm::util::trim(arg.substr(flag.size() + 1));
            }
        }
    }
    return std::nullopt;
}

bool has_any_flag(const std::vector<std::string>& extra,
                  std::initializer_list<const char*> flags) {
    for (const auto& arg : extra) {
        for (const char* flag : flags) {
            if (arg_matches_flag(arg, flag)) return true;
        }
    }
    return false;
}

std::optional<int> parse_positive_int(const std::optional<std::string>& value) {
    if (!value) return std::nullopt;
    try {
        size_t idx = 0;
        int parsed = std::stoi(*value, &idx);
        if (idx != value->size()) return std::nullopt;
        if (parsed <= 0) return std::nullopt;
        return parsed;
    } catch (...) {
        return std::nullopt;
    }
}

int64_t ceil_to_int64(double value) {
    if (value <= 0.0) return 0;
    return static_cast<int64_t>(std::ceil(value));
}

int64_t model_file_size_mb(const std::string& model_path,
                           const std::string& models_dir,
                           std::string* out_source_path) {
    // measure_model_bytes() sizes a DIRECTORY recursively.
    //
    // This used to call fs::file_size() on the path, which sets an error_code on
    // a directory and fell through to a flat 2048 MB — so every multi-shard HF
    // checkpoint and every converted Soma container reported the same size, and
    // that single number is what the scheduler placed against. The fallback below
    // is now reached only when the path genuinely cannot be measured (missing, or
    // unreadable), which is the case it was always meant for.
    const auto bytes = measure_model_bytes(model_path, models_dir, out_source_path);
    if (bytes <= 0) return kFallbackModelMb;
    return bytes_to_mb(bytes);
}

} // namespace

int effective_llama_parallel(const RuntimeSettings& settings) {
    int parallel = settings.parallel > 0 ? settings.parallel : 1;
    if (auto parsed = parse_positive_int(find_flag_value(settings.extra_args, {"--parallel"}))) {
        parallel = *parsed;
    }
    return std::max(1, parallel);
}

int effective_llama_server_ctx_tokens(const RuntimeSettings& settings) {
    if (auto parsed = parse_positive_int(find_flag_value(settings.extra_args,
                                                         {"--ctx-size", "-c"}))) {
        return *parsed;
    }

    const int ctx_size = settings.ctx_size > 0 ? settings.ctx_size : 4096;
    const int parallel = effective_llama_parallel(settings);
    const int64_t server_ctx =
        static_cast<int64_t>(ctx_size) * static_cast<int64_t>(parallel);
    if (server_ctx > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(server_ctx);
}

InferenceMemoryEstimate estimate_inference_memory(const std::string& model_path,
                                                  const RuntimeSettings& settings,
                                                  const std::string& models_dir) {
    InferenceMemoryEstimate estimate;
    estimate.parallel = effective_llama_parallel(settings);
    estimate.effective_ctx_tokens = effective_llama_server_ctx_tokens(settings);
    estimate.model_weight_mb = model_file_size_mb(model_path, models_dir,
                                                  &estimate.source_path);

    const double ctx_scale =
        static_cast<double>(std::max(1, estimate.effective_ctx_tokens)) / 2048.0;

    // KV/cache memory grows with total server context. Without architecture-level
    // GGUF tensors this stays intentionally approximate, anchored to on-disk
    // GGUF weight size rather than filename parameter guesses.
    estimate.context_cache_mb =
        ceil_to_int64(static_cast<double>(estimate.model_weight_mb) * 0.20 * ctx_scale);

    if (has_any_flag(settings.extra_args, {"--cache-type-k", "--cache-type-v"})) {
        estimate.context_cache_mb =
            ceil_to_int64(static_cast<double>(estimate.context_cache_mb) * 0.75);
    }

    estimate.overhead_mb =
        std::max<int64_t>(256, ceil_to_int64(static_cast<double>(
            estimate.model_weight_mb + estimate.context_cache_mb) * 0.10));

    estimate.total_vram_mb =
        estimate.model_weight_mb + estimate.context_cache_mb + estimate.overhead_mb;
    return estimate;
}

int64_t estimate_inference_vram_mb(const std::string& model_path,
                                   const RuntimeSettings& settings,
                                   const std::string& models_dir) {
    return estimate_inference_memory(model_path, settings, models_dir).total_vram_mb;
}

} // namespace mm
