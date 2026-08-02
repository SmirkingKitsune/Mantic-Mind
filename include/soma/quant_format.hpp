#pragma once

// Soma — group-scale quantized tensor formats.
//
// ⚠ THESE ARE SOMA'S OWN FORMATS. They are NOT bit-compatible with llama.cpp's
// K-quants, GGUF, or anything else. That is why they are spelled Q4_G / Q6_G
// rather than Q4_K / Q6_K: a name collision here would eventually have someone
// assume a llama.cpp tensor could be loaded, and the failure would be silent.
//
// Soma writes its own container at admission (schemas/arch-ir.md §5), so there
// is nothing to be gained from bit-compatibility and a real cost to implying it.
//
// Layout — every format is a flat sequence of independent groups, so a single
// weight row can be dequantized without touching its neighbours:
//
//   Q8_0   f32 scale                | G int8   symmetric      4 + G     bytes
//   Q4_0   f32 scale                | G int4   symmetric      4 + G/2   bytes
//   Q4_G   f32 scale, f32 min       | G uint4  asymmetric     8 + G/2   bytes
//   Q6_G   f32 scale                | G int6   symmetric      4 + 3G/4  bytes
//
// Symmetric formats centre on zero and carry one scale; asymmetric adds a min so
// a group whose values do not straddle zero keeps its full resolution. Q6_G
// exists for the expert down-projection, which is consistently more sensitive
// than gate/up — a per-role choice the QuantMap makes explicit.

#include "soma/quant.hpp"
#include "soma/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace soma {

inline constexpr std::uint32_t kDefaultGroup = 128;

/// Bytes one group occupies. 0 when the dtype is unquantized or unsupported.
std::size_t group_bytes(DType dtype, std::uint32_t group) noexcept;

/// Total bytes for `n` elements at a group size that divides `n`.
///
/// Prefer quantized_tensor_bytes() for a real 2-D weight: this overload cannot
/// know the row width, so it cannot apply the group reduction that
/// quantize_tensor() performs when cols < group.
std::size_t quantized_bytes(DType dtype, std::size_t n, std::uint32_t group) noexcept;

/// Total bytes for a [rows, cols] weight quantized along `cols`.
///
/// Applies the SAME group reduction quantize_tensor() does — the effective group
/// is the largest divisor of `cols` not exceeding `group`. Computing the size from
/// a flat element count instead silently disagrees for any tensor narrower than
/// the requested group, and the disagreement propagates into the plan's footprint
/// and the container's expert-size check. That was observed: a container reporting
/// 4352 B/expert against a computed 3904 B, on a fixture whose cols (64, 32) both
/// fall below the requested group of 128.
std::size_t quantized_tensor_bytes(DType dtype,
                                   std::uint32_t rows,
                                   std::uint32_t cols,
                                   std::uint32_t group) noexcept;

/// The group actually used for a given row width.
std::uint32_t effective_group(std::uint32_t cols, std::uint32_t group) noexcept;

/// True when this dtype is a group-scale quantized format.
bool is_quantized(DType dtype) noexcept;

/// A quantized 2-D weight, row-major [rows, cols], quantized along `cols`.
///
/// Quantizing along the contraction dimension is what makes a fused
/// dequantize-and-accumulate matvec possible: one group of scales covers a
/// contiguous run of the dot product.
struct QTensor {
    DType dtype = DType::F32;
    std::uint32_t group = kDefaultGroup;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<std::byte> data;

    std::size_t bytes_per_row() const noexcept;

    bool empty() const noexcept { return data.empty(); }
};

/// Quantize a [rows, cols] fp32 weight.
///
/// When `cols % group != 0` the group is reduced to the largest divisor of cols
/// that does not exceed the request, rather than padding. Padding would put
/// invented zeros inside a real group's scale computation and quietly widen its
/// range.
Status quantize_tensor(std::span<const float> src,
                       std::uint32_t rows,
                       std::uint32_t cols,
                       DType dtype,
                       std::uint32_t group,
                       QTensor& out);

Status dequantize_tensor(const QTensor& src, std::span<float> out);

/// Reference to a weight in whichever precision it was loaded at.
///
/// This is the whole reason G1 does not fork the forward: one entry point, two
/// precisions, and the quantized path is validated against the fp32 path through
/// the SAME code. A separate quantized forward would let the two drift, and the
/// drift is exactly what the gate is supposed to detect.
struct WeightRef {
    // A pure VIEW, owning nothing.
    //
    // It used to hold `const QTensor*`, which silently assumed every quantized
    // weight was backed by an owning tensor. A STREAMED expert is not: it is raw
    // bytes borrowed from the expert cache for the lifetime of an ExpertRef.
    // Describing the weight by (bytes, dtype, group, shape) instead makes
    // resident and streamed weights the same thing to every kernel — which is
    // what lets one forward serve both without a second code path.
    std::span<const float> f32; ///< set when unquantized
    CByteSpan bytes;            ///< set when quantized
    DType dtype = DType::F32;
    std::uint32_t group = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;

    bool empty() const noexcept { return f32.empty() && bytes.empty(); }

    bool quantized() const noexcept { return !bytes.empty(); }

    static WeightRef
    from_f32(std::span<const float> data, std::uint32_t rows, std::uint32_t cols) noexcept;
    static WeightRef from_q(const QTensor& t) noexcept;

    /// Non-owning quantized view — the streaming path's entry point. The caller
    /// must keep the backing bytes alive, which is exactly what the ExpertRef
    /// returned alongside it does.
    static WeightRef from_quantized_bytes(CByteSpan data,
                                          DType dtype,
                                          std::uint32_t group,
                                          std::uint32_t rows,
                                          std::uint32_t cols) noexcept;
};

std::size_t quantized_footprint(const QTensor& t) noexcept;

// ── Dispatching kernels ──────────────────────────────────────────────────────
//
// y = W @ x, with W in whichever precision it was loaded at. Quantized paths
// fuse dequantization into the accumulation rather than materializing the row:
// materializing would cost a full fp32 row of scratch per call and defeat the
// point of storing it small.

void matvec(const WeightRef& w, std::span<const float> x, std::span<float> y) noexcept;

void matmul(const WeightRef& w,
            std::span<const float> x,
            std::uint32_t n_rows,
            std::span<float> y) noexcept;

/// Weights-outer, inputs-inner: `y[t*w.rows + r] = dot(w_row_r, x + t*w.cols)`.
///
/// Same arithmetic as calling matvec() once per input, and the same numbers —
/// each output is still one dot product accumulated in one order, so this is
/// bit-identical, not merely close.
///
/// The difference is which operand gets reused. matmul() walks inputs outermost,
/// so the entire weight matrix streams past the core once PER INPUT ROW; against
/// a 2.9 MB expert that is 2.9 MB of traffic per token. This loop holds one
/// weight row in L1 and applies it to all `n_inputs` inputs before moving on, so
/// the weights stream once per TILE.
///
/// That is the whole fix for the locality the batch union gave up: expert-major
/// order was reading each expert once from disk and then re-reading it from
/// memory once per row.
void matmul_tiled(const WeightRef& w, const float* x, std::uint32_t n_inputs, float* y) noexcept;

/// Single-row kernel family, used unconditionally under Determinism::Strict.
///
/// Distinct from matvec() called with one row: a batched kernel invoked at m == 1
/// may still reassociate differently from the dedicated path, and bit-exact
/// replay needs both halves pinned. See docs/architecture.md §10.
void matvec_single_row(const WeightRef& w, std::span<const float> x, std::span<float> y) noexcept;

} // namespace soma
