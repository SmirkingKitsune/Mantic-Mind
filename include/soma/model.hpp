#pragma once

// Soma — per-sequence runtime policy shared by the scheduler and sampler.

#include <cstdint>

namespace soma {
/// Reproducibility mode. See docs/architecture.md §10.
///
/// `Batched` tokens remain the argmax of a VALID forward, so quality holds —
/// but the exact stream can depend on who else is on the server, because
/// quantized integer kernels round differently at different shapes.
enum class Determinism : std::uint8_t {
    Batched = 0, ///< default: batch freely
    Strict,      ///< serialized single-row path, single-row kernel family
};

struct SamplerState {
    float temperature = 0.7f;
    float top_p = 0.9f;
    std::int32_t top_k = -1;
    float min_p = -1.0f;
    float presence_penalty = 0.0f;
    float repeat_penalty = -1.0f;
    std::uint64_t rng_state = 0;
};

} // namespace soma
