#pragma once

// Soma — quantized kernel family and the per-model static dispatch table.
//
// Which kernel wins at a given shape is EMPIRICAL, not derivable. The prior art
// measured an int4 single-row path SLOWER than f32 on the same hardware. So the
// autotuner runs over the model's actual shape set at admission and its output
// is codegen'd into a static table — there is no runtime search and no runtime
// heuristic.

#include "soma/quant.hpp"
#include "soma/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace soma {

enum class KernelOp : std::uint8_t {
    Gemm = 0,  ///< batched: m > 1
    Gemv,      ///< single-row: m == 1
    MoeExpert, ///< fused gate/up/down over a CSR row list
    AttnQk,
    AttnAv,
};

struct KernelShape {
    KernelOp op = KernelOp::Gemm;
    std::uint32_t m = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;
    DType dtype = DType::F32;
};

/// One autotuner decision, mirroring a `kernel_choice` registry row.
struct KernelChoice {
    KernelShape shape{};
    const char* impl = nullptr;
    float gflops = 0.0f;
};

/// Resolved once at load from the specialization header. Lookup is a hash probe,
/// never a search.
struct KernelTable {
    std::span<const KernelChoice> choices;

    const KernelChoice* find(const KernelShape& shape) const noexcept;

    /// The single-row family, used unconditionally under Determinism::Strict.
    ///
    /// Strict mode does not merely set batch=1: it also pins kernel selection,
    /// because a batched kernel invoked with m==1 may still round differently
    /// from the dedicated single-row path. Both halves are required for
    /// bit-exact replay.
    const KernelChoice* find_single_row(const KernelShape& shape) const noexcept;
};

/// Autotune over a shape set. Admission-time only — never called while serving.
Status autotune(std::span<const KernelShape> shapes, std::span<KernelChoice> out);

const char* to_string(KernelOp op) noexcept;

} // namespace soma
