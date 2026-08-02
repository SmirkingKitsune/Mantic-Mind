// Soma — kernel implementations and the registry that chooses between them.
//
// Correctness is not negotiable across implementations: every candidate for a
// given (op, dtype) must produce the SAME numbers, or the autotuner is selecting
// on speed while silently varying output. The G1 test asserts that before it
// asserts anything about timing.
//
// "The same numbers" means agreement to the autotuner's 1e-4 relative tolerance,
// not bit-identity, and the distinction became load-bearing when the SIMD
// candidates landed: a vector accumulator sums the same products in a different
// order. Bit-identity would have excluded the fastest kernels for a difference
// smaller than the quantization they are decoding. What the rule actually
// protects against is a candidate computing a DIFFERENT FUNCTION — a permuted
// row, a dropped tail — and 1e-4 catches that with orders of magnitude to spare.

#include "soma/kernel_registry.hpp"

#include "soma/kernels_f32.hpp"
#include "soma/kernels_simd.hpp"

#include <array>
#include <cstring>
#include <sstream>

namespace soma {

namespace {

float get_f32(const std::byte* p) noexcept {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// ── fp32 ─────────────────────────────────────────────────────────────────────

void f32_scalar(
    const float* w, const float* x, std::uint32_t rows, std::uint32_t cols, float* y) noexcept {
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* wr = w + static_cast<std::size_t>(r) * cols;
        float acc = 0.0f;
        for (std::uint32_t i = 0; i < cols; ++i)
            acc += wr[i] * x[i];
        y[r] = acc;
    }
}

void f32_unroll4(
    const float* w, const float* x, std::uint32_t rows, std::uint32_t cols, float* y) noexcept {
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* wr = w + static_cast<std::size_t>(r) * cols;
        // Four independent accumulators so consecutive FMAs do not serialize on
        // each other's latency. The partial sums are combined in a fixed order,
        // so the result is deterministic — but it is NOT the same association as
        // f32_scalar, which is why the correctness check uses a tolerance rather
        // than bit-equality.
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        std::uint32_t i = 0;
        for (; i + 4 <= cols; i += 4) {
            a0 += wr[i] * x[i];
            a1 += wr[i + 1] * x[i + 1];
            a2 += wr[i + 2] * x[i + 2];
            a3 += wr[i + 3] * x[i + 3];
        }
        float tail = 0.0f;
        for (; i < cols; ++i)
            tail += wr[i] * x[i];
        y[r] = (a0 + a1) + (a2 + a3) + tail;
    }
}

/// The fp32 SIMD path, registered for exactly the reason the quantized one is.
///
/// When the quantized kernels were vectorised, this table went on reporting
/// `q4_g.fused` at 4.5 GF/s — the registry's private copy — while the engine ran
/// the new kernel at 35. The fp32 side had the same split, so it is closed at the
/// same time rather than left as a second instance of a mistake already made
/// once here.
void f32_simd(
    const float* w, const float* x, std::uint32_t rows, std::uint32_t cols, float* y) noexcept {
    if (!simd::available()) {
        f32_unroll4(w, x, rows, cols, y);
        return;
    }
    simd::matvec(w, x, rows, cols, y);
}

// ── quantized: fused dequantize-and-accumulate ───────────────────────────────

template <DType D>
float qdot_fused(const std::byte* p, std::uint32_t cols, std::uint32_t g, const float* x) noexcept {
    float acc = 0.0f;
    const auto gb = group_bytes(D, g);
    for (std::uint32_t c = 0; c < cols; c += g) {
        const float scale = get_f32(p);
        const float* xv = x + c;
        float part = 0.0f;

        if constexpr (D == DType::Q8_0) {
            for (std::uint32_t i = 0; i < g; ++i) {
                part += static_cast<float>(static_cast<std::int8_t>(p[4 + i])) * xv[i];
            }
            acc += scale * part;
        } else if constexpr (D == DType::Q4_0) {
            for (std::uint32_t i = 0; i < g; i += 2) {
                const auto b = static_cast<std::uint8_t>(p[4 + i / 2]);
                part += static_cast<float>(static_cast<int>(b & 0x0F) - 8) * xv[i];
                part += static_cast<float>(static_cast<int>(b >> 4) - 8) * xv[i + 1];
            }
            acc += scale * part;
        } else if constexpr (D == DType::Q4_G) {
            const float minv = get_f32(p + 4);
            float xsum = 0.0f;
            for (std::uint32_t i = 0; i < g; i += 2) {
                const auto b = static_cast<std::uint8_t>(p[8 + i / 2]);
                part += static_cast<float>(b & 0x0F) * xv[i];
                part += static_cast<float>(b >> 4) * xv[i + 1];
                xsum += xv[i] + xv[i + 1];
            }
            acc += scale * part + minv * xsum;
        } else if constexpr (D == DType::Q6_G) {
            const std::byte* q = p + 4;
            for (std::uint32_t i = 0; i < g; i += 4) {
                const std::uint32_t packed =
                    static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[0])) |
                    (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[1])) << 8) |
                    (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[2])) << 16);
                part += static_cast<float>(static_cast<int>(packed & 0x3F) - 32) * xv[i];
                part += static_cast<float>(static_cast<int>((packed >> 6) & 0x3F) - 32) * xv[i + 1];
                part +=
                    static_cast<float>(static_cast<int>((packed >> 12) & 0x3F) - 32) * xv[i + 2];
                part +=
                    static_cast<float>(static_cast<int>((packed >> 18) & 0x3F) - 32) * xv[i + 3];
                q += 3;
            }
            acc += scale * part;
        }
        p += gb;
    }
    return acc;
}

template <DType D>
void q_fused(const QTensor& w, const float* x, float* y, float*) noexcept {
    const auto per_row = w.bytes_per_row();
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        y[r] = qdot_fused<D>(
            w.data.data() + static_cast<std::size_t>(r) * per_row, w.cols, w.group, x);
    }
}

/// The SIMD path, entered through the same registry as everything else.
///
/// It has to be a CANDIDATE, not a special case bolted beside the registry. The
/// autotuner's whole purpose is to record which kernel the engine should run for
/// a shape; if the fast path were reachable only from `matvec` and invisible
/// here, `kernel_choice` would faithfully describe code that never executes —
/// and the fp32-beats-q4_g reading that sent us here in the first place would
/// still be on the record, still wrong, still believed.
///
/// Falls back to the scalar kernel when the host lacks AVX2, so the entry is
/// always valid to select and never has to be conditionally registered.
template <DType D>
void q_simd(const QTensor& w, const float* x, float* y, float* scratch) noexcept {
    if (!simd::available()) {
        q_fused<D>(w, x, y, scratch);
        return;
    }
    const auto per_row = w.bytes_per_row();
    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const std::byte* p = w.data.data() + static_cast<std::size_t>(r) * per_row;
        if constexpr (D == DType::Q4_G) {
            y[r] = simd::qdot_q4g(p, w.cols, w.group, x, nullptr);
        } else if constexpr (D == DType::Q6_G) {
            y[r] = simd::qdot_q6g(p, w.cols, w.group, x);
        } else if constexpr (D == DType::Q8_0) {
            y[r] = simd::qdot_q8_0(p, w.cols, w.group, x);
        } else if constexpr (D == DType::Q4_0) {
            y[r] = simd::qdot_q4_0(p, w.cols, w.group, x);
        }
    }
}

/// Dequantize the row to scratch, then run the fp32 loop.
///
/// The strategy the prior art's finding is about: it pays a full fp32 row write
/// per output, and wins only if the fp32 inner loop is enough faster to cover
/// that. At m == 1 there is nothing to amortize the write against.
template <DType D>
void q_dequant_f32(const QTensor& w, const float* x, float* y, float* scratch) noexcept {
    const auto per_row = w.bytes_per_row();
    const auto g = w.group;
    const auto gb = group_bytes(D, g);

    for (std::uint32_t r = 0; r < w.rows; ++r) {
        const std::byte* p = w.data.data() + static_cast<std::size_t>(r) * per_row;
        for (std::uint32_t c = 0; c < w.cols; c += g) {
            const float scale = get_f32(p);
            float* d = scratch + c;
            if constexpr (D == DType::Q8_0) {
                for (std::uint32_t i = 0; i < g; ++i) {
                    d[i] = scale * static_cast<float>(static_cast<std::int8_t>(p[4 + i]));
                }
            } else if constexpr (D == DType::Q4_0) {
                for (std::uint32_t i = 0; i < g; i += 2) {
                    const auto b = static_cast<std::uint8_t>(p[4 + i / 2]);
                    d[i] = scale * static_cast<float>(static_cast<int>(b & 0x0F) - 8);
                    d[i + 1] = scale * static_cast<float>(static_cast<int>(b >> 4) - 8);
                }
            } else if constexpr (D == DType::Q4_G) {
                const float minv = get_f32(p + 4);
                for (std::uint32_t i = 0; i < g; i += 2) {
                    const auto b = static_cast<std::uint8_t>(p[8 + i / 2]);
                    d[i] = minv + scale * static_cast<float>(b & 0x0F);
                    d[i + 1] = minv + scale * static_cast<float>(b >> 4);
                }
            } else if constexpr (D == DType::Q6_G) {
                const std::byte* q = p + 4;
                for (std::uint32_t i = 0; i < g; i += 4) {
                    const std::uint32_t packed =
                        static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[0])) |
                        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[1])) << 8) |
                        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(q[2])) << 16);
                    for (std::uint32_t kk = 0; kk < 4; ++kk) {
                        d[i + kk] = scale * static_cast<float>(
                                                static_cast<int>((packed >> (6 * kk)) & 0x3F) - 32);
                    }
                    q += 3;
                }
            }
            p += gb;
        }
        float acc = 0.0f;
        for (std::uint32_t i = 0; i < w.cols; ++i)
            acc += scratch[i] * x[i];
        y[r] = acc;
    }
}

// ── registry ─────────────────────────────────────────────────────────────────

const std::array<KernelImpl, 21> kImpls{{
    {"f32.simd", KernelOp::Gemv, DType::F32, nullptr, &f32_simd},
    {"f32.scalar", KernelOp::Gemv, DType::F32, nullptr, &f32_scalar},
    {"f32.unroll4", KernelOp::Gemv, DType::F32, nullptr, &f32_unroll4},

    {"q8_0.simd", KernelOp::Gemv, DType::Q8_0, &q_simd<DType::Q8_0>, nullptr},
    {"q8_0.fused", KernelOp::Gemv, DType::Q8_0, &q_fused<DType::Q8_0>, nullptr},
    {"q8_0.dequant", KernelOp::Gemv, DType::Q8_0, &q_dequant_f32<DType::Q8_0>, nullptr},

    {"q4_0.simd", KernelOp::Gemv, DType::Q4_0, &q_simd<DType::Q4_0>, nullptr},
    {"q4_0.fused", KernelOp::Gemv, DType::Q4_0, &q_fused<DType::Q4_0>, nullptr},
    {"q4_0.dequant", KernelOp::Gemv, DType::Q4_0, &q_dequant_f32<DType::Q4_0>, nullptr},

    {"q4_g.simd", KernelOp::Gemv, DType::Q4_G, &q_simd<DType::Q4_G>, nullptr},
    {"q4_g.fused", KernelOp::Gemv, DType::Q4_G, &q_fused<DType::Q4_G>, nullptr},
    {"q4_g.dequant", KernelOp::Gemv, DType::Q4_G, &q_dequant_f32<DType::Q4_G>, nullptr},

    {"q6_g.simd", KernelOp::Gemv, DType::Q6_G, &q_simd<DType::Q6_G>, nullptr},
    {"q6_g.fused", KernelOp::Gemv, DType::Q6_G, &q_fused<DType::Q6_G>, nullptr},
    {"q6_g.dequant", KernelOp::Gemv, DType::Q6_G, &q_dequant_f32<DType::Q6_G>, nullptr},

    // Gemm entries alias the Gemv implementations: batching is expressed by
    // calling per row, so the CHOICE can still differ between m == 1 and m > 1
    // even though the code does not. That difference is the whole point of
    // keying the table on m.
    {"f32.simd", KernelOp::Gemm, DType::F32, nullptr, &f32_simd},
    {"f32.scalar", KernelOp::Gemm, DType::F32, nullptr, &f32_scalar},
    {"f32.unroll4", KernelOp::Gemm, DType::F32, nullptr, &f32_unroll4},
    {"q4_g.simd", KernelOp::Gemm, DType::Q4_G, &q_simd<DType::Q4_G>, nullptr},
    {"q4_g.fused", KernelOp::Gemm, DType::Q4_G, &q_fused<DType::Q4_G>, nullptr},
    {"q4_g.dequant", KernelOp::Gemm, DType::Q4_G, &q_dequant_f32<DType::Q4_G>, nullptr},
}};

} // namespace

std::span<const KernelImpl> candidates(KernelOp op, DType dtype) noexcept {
    // Contiguous runs per (op, dtype) in kImpls, so this is a scan of a fixed
    // 14-entry table at load time only.
    std::size_t first = kImpls.size();
    std::size_t count = 0;
    for (std::size_t i = 0; i < kImpls.size(); ++i) {
        if (kImpls[i].op == op && kImpls[i].dtype == dtype) {
            if (first == kImpls.size()) first = i;
            ++count;
        } else if (count > 0) {
            break;
        }
    }
    if (count == 0) return {};
    return {kImpls.data() + first, count};
}

const KernelImpl* default_impl(KernelOp op, DType dtype) noexcept {
    const auto c = candidates(op, dtype);
    return c.empty() ? nullptr : &c.front();
}

const KernelImpl* impl_by_name(KernelOp op, DType dtype, std::string_view name) noexcept {
    for (const auto& k : candidates(op, dtype)) {
        if (name == k.name) return &k;
    }
    return nullptr;
}

// ── resolved table ───────────────────────────────────────────────────────────

namespace {

bool same_shape(const KernelShape& a, const KernelShape& b) noexcept {
    return a.op == b.op && a.m == b.m && a.n == b.n && a.k == b.k && a.dtype == b.dtype;
}

} // namespace

void ResolvedKernels::set(const KernelShape& shape, const KernelImpl* impl) {
    for (auto& e : entries_) {
        if (same_shape(e.shape, shape)) {
            e.impl = impl;
            return;
        }
    }
    entries_.push_back({shape, impl});
}

const KernelImpl* ResolvedKernels::find(const KernelShape& shape) const noexcept {
    for (const auto& e : entries_) {
        if (same_shape(e.shape, shape)) return e.impl;
    }
    return nullptr;
}

const KernelImpl* ResolvedKernels::resolve(
    KernelOp op, DType dtype, std::uint32_t m, std::uint32_t n, std::uint32_t k) const noexcept {
    KernelShape s{op, m, n, k, dtype};
    if (const auto* hit = find(s)) return hit;

    // A shape the autotuner never saw falls back to the default rather than
    // failing: an untuned model must still run, just not optimally.
    if (const auto* d = default_impl(op, dtype)) return d;

    // Gemm falls back to Gemv for the same dtype.
    //
    // Batching is expressed by calling the per-row kernel repeatedly, so the two
    // ops share implementations; the registry only lists Gemm candidates where a
    // DIFFERENT choice is plausible. Without this, a dtype with no Gemm entry
    // resolved to nullptr — which the autotune test caught as 12 extracted
    // shapes producing 11 tuned results.
    if (op == KernelOp::Gemm) return default_impl(KernelOp::Gemv, dtype);
    return nullptr;
}

Status build_resolved(std::span<const TuneResult> results, ResolvedKernels& out) {
    for (const auto& r : results) {
        const auto* impl = impl_by_name(r.shape.op, r.shape.dtype, r.impl);
        if (impl == nullptr) {
            return {StatusCode::NotFound,
                    "tuned implementation '" + r.impl +
                        "' is not registered in this build; "
                        "the kernel_choice rows were produced by a different binary"};
        }
        out.set(r.shape, impl);
    }
    return {};
}

std::string to_registry_rows(std::span<const TuneResult> results, std::int64_t model_id) {
    std::ostringstream o;
    for (const auto& r : results) {
        o << "INSERT OR REPLACE INTO kernel_choice"
             " (model_id, op, m, n, k, dtype, impl, gflops) VALUES ("
          << model_id << ", '" << to_string(r.shape.op) << "', " << r.shape.m << ", " << r.shape.n
          << ", " << r.shape.k << ", '" << to_string(r.shape.dtype) << "', '" << r.impl << "', "
          << r.gflops << ");\n";
    }
    return o.str();
}

const char* to_string(KernelOp op) noexcept {
    switch (op) {
    case KernelOp::Gemm:
        return "gemm";
    case KernelOp::Gemv:
        return "gemv";
    case KernelOp::MoeExpert:
        return "moe_expert";
    case KernelOp::AttnQk:
        return "attn_qk";
    case KernelOp::AttnAv:
        return "attn_av";
    }
    return "unknown";
}

} // namespace soma
