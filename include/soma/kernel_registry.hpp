#pragma once

// Soma — the kernel implementation registry and autotuner.
//
// Which kernel wins at a given shape is EMPIRICAL, not derivable. The prior art
// measured an int4 single-row path SLOWER than fp32 on the same hardware — a
// result that is obvious in hindsight (the dequantize cost dominates when there
// is no batch to amortize it over) and completely invisible from the source.
//
// So the autotuner runs over the model's ACTUAL shape set at admission and its
// output is codegen'd into a static table. There is no runtime search and no
// runtime heuristic: each weight resolves its implementation once, at load, and
// the hot path makes one indirect call.
//
// An autotuner with one candidate per shape is theatre. These implementations
// are genuinely different strategies whose winner depends on shape:
//
//   scalar        straightforward accumulate; shortest dependency chain to write,
//                 longest to execute
//   unroll4       four independent accumulators, so FMA latency overlaps instead
//                 of serializing. Wins when k is large enough to amortize the
//                 tail.
//   dequant_f32   materialize the row as fp32, then run the tight fp32 loop.
//                 Pays a scratch write per row and wins only when the fp32 inner
//                 loop is enough faster to cover it — which is exactly the
//                 hypothesis the prior art's finding raises.

#include "soma/arch_ir.hpp"
#include "soma/kernels.hpp"
#include "soma/quant_format.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace soma {

/// One row-major weight against one input vector. `scratch` is at least `cols`
/// floats, and only the dequantize-first strategies touch it.
using QMatvecFn = void (*)(const QTensor& w, const float* x, float* y, float* scratch) noexcept;

using F32MatvecFn = void (*)(
    const float* w, const float* x, std::uint32_t rows, std::uint32_t cols, float* y) noexcept;

struct KernelImpl {
    const char* name = nullptr;
    KernelOp op = KernelOp::Gemv;
    DType dtype = DType::F32;
    QMatvecFn q_fn = nullptr;
    F32MatvecFn f32_fn = nullptr;
};

/// Every candidate for this (op, dtype). Empty when unsupported.
std::span<const KernelImpl> candidates(KernelOp op, DType dtype) noexcept;

/// The implementation used before autotuning has run, and the fallback when a
/// shape has no measured entry. Never null for a supported dtype.
const KernelImpl* default_impl(KernelOp op, DType dtype) noexcept;

const KernelImpl* impl_by_name(KernelOp op, DType dtype, std::string_view name) noexcept;

// ── Shape extraction ─────────────────────────────────────────────────────────

/// Every distinct (op, m, n, k, dtype) a model actually executes.
///
/// Derived from the IR rather than observed at runtime, so admission can tune
/// before the model has ever served a request. `batch_sizes` is the set of row
/// counts to tune for — 1 is the decode path and must always be present, because
/// it is the one shape where dequantize cost has nothing to amortize against.
std::vector<KernelShape>
model_shapes(const ArchIr& arch, const QuantMap& quant, std::span<const std::uint32_t> batch_sizes);

// ── Autotune ─────────────────────────────────────────────────────────────────

struct AutotuneOptions {
    std::uint32_t warmup_iters = 3;
    std::uint32_t measure_iters = 7;
    double min_seconds_per_measure = 0.002;

    /// Best-of-N rather than mean. The question is "how fast can this go",
    /// and the minimum rejects scheduler noise that the mean absorbs.
    bool take_minimum = true;
};

struct TuneResult {
    KernelShape shape{};
    std::string impl;
    float gflops = 0.0f;

    /// Every candidate's score, so a surprising winner can be checked rather
    /// than trusted. This is where "int4 was slower than fp32" would show up.
    std::vector<std::pair<std::string, float>> all;
};

Status autotune_shapes(std::span<const KernelShape> shapes,
                       const AutotuneOptions& opts,
                       std::vector<TuneResult>& out);

/// Resolved table. Lookup is a hash probe; there is no search at runtime.
class ResolvedKernels {
public:
    void set(const KernelShape& shape, const KernelImpl* impl);
    const KernelImpl* find(const KernelShape& shape) const noexcept;

    /// Resolve for a weight, falling back to the default when the shape was not
    /// tuned. Called once per weight at load.
    const KernelImpl* resolve(
        KernelOp op, DType dtype, std::uint32_t m, std::uint32_t n, std::uint32_t k) const noexcept;

    std::size_t size() const noexcept { return entries_.size(); }

private:
    struct Entry {
        KernelShape shape{};
        const KernelImpl* impl = nullptr;
    };

    std::vector<Entry> entries_;
};

Status build_resolved(std::span<const TuneResult> results, ResolvedKernels& out);

/// Serialize to the registry's `kernel_choice` rows.
std::string to_registry_rows(std::span<const TuneResult> results, std::int64_t model_id);

} // namespace soma
