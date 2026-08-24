#include "soma/kernels_f32.hpp"

#include "soma/kernels_simd.hpp"
#include "soma/threading.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace soma::f32 {

float dot(std::span<const float> a, std::span<const float> b, std::uint32_t n) noexcept {
    if (simd::available()) return simd::dot(a.data(), b.data(), n);
    float acc = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i)
        acc += a[i] * b[i];
    return acc;
}

void matvec(std::span<const float> w,
            std::span<const float> x,
            std::uint32_t m,
            std::uint32_t k,
            std::span<float> y) noexcept {
    const bool vec = simd::available();

    // Output rows are independent, so splitting them is bit-identical to serial:
    // every y[row] is still produced by one thread accumulating in one order.
    // Splitting `k` instead would be a cross-thread reduction and would make the
    // result depend on the core count.
    const auto work = static_cast<std::uint64_t>(m) * k;
    if (work >= kParallelMacThreshold && !ThreadPool::in_parallel_region()) {
        const auto rows_per = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(kChunkMacs / std::max<std::uint32_t>(1, k)));
        ThreadPool::global().parallel_for(
            m, rows_per, [&](std::uint32_t begin, std::uint32_t end, std::uint32_t) {
                const auto rows = end - begin;
                const float* wr = w.data() + static_cast<std::size_t>(begin) * k;
                if (vec) {
                    simd::matvec(wr, x.data(), rows, k, y.data() + begin);
                } else {
                    for (std::uint32_t r = 0; r < rows; ++r) {
                        float acc = 0.0f;
                        const float* wrow = wr + static_cast<std::size_t>(r) * k;
                        for (std::uint32_t i = 0; i < k; ++i)
                            acc += wrow[i] * x[i];
                        y[begin + r] = acc;
                    }
                }
            });
        return;
    }

    if (vec) {
        simd::matvec(w.data(), x.data(), m, k, y.data());
        return;
    }
    for (std::uint32_t row = 0; row < m; ++row) {
        float acc = 0.0f;
        const float* wr = w.data() + static_cast<std::size_t>(row) * k;
        for (std::uint32_t i = 0; i < k; ++i)
            acc += wr[i] * x[i];
        y[row] = acc;
    }
}

void axpy(float alpha, std::span<const float> x, std::uint32_t n, std::span<float> y) noexcept {
    if (simd::available()) {
        simd::axpy(alpha, x.data(), n, y.data());
        return;
    }
    for (std::uint32_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}

void matmul(std::span<const float> w,
            std::span<const float> x,
            std::uint32_t rows,
            std::uint32_t m,
            std::uint32_t k,
            std::span<float> y) noexcept {
    for (std::uint32_t t = 0; t < rows; ++t) {
        matvec(w,
               x.subspan(static_cast<std::size_t>(t) * k, k),
               m,
               k,
               y.subspan(static_cast<std::size_t>(t) * m, m));
    }
}

void rmsnorm_into(std::span<const float> x,
                  std::span<const float> weight,
                  std::uint32_t n,
                  float eps,
                  std::span<float> out,
                  float weight_offset) noexcept {
    float sumsq = 0.0f;
    if (simd::available()) {
        sumsq = simd::sumsq(x.data(), n);
    } else {
        for (std::uint32_t i = 0; i < n; ++i)
            sumsq += x[i] * x[i];
    }
    const float scale = 1.0f / std::sqrt(sumsq / static_cast<float>(n) + eps);
    // The elementwise tail stays scalar: it is a three-operand multiply with no
    // reduction, so the compiler already vectorises it and an intrinsic version
    // would only be a second place for the expression to drift from HF's.
    for (std::uint32_t i = 0; i < n; ++i)
        out[i] = x[i] * scale * (weight[i] + weight_offset);
}

void rmsnorm(std::span<float> x,
             std::span<const float> weight,
             std::uint32_t n,
             float eps,
             float weight_offset) noexcept {
    rmsnorm_into(x, weight, n, eps, x, weight_offset);
}

void softmax(std::span<float> x, std::uint32_t n) noexcept {
    if (n == 0) return;

    // The max and the final scale vectorise; the exp does NOT, deliberately. A
    // polynomial exp approximation would be several times faster and is the
    // obvious next step, but it changes the VALUES rather than their summation
    // order — that is a conformance change, and softmax feeds both the attention
    // weights and the router, where the router's output decides WHICH EXPERTS
    // FIRE. Trading accuracy there for speed needs its own measurement against
    // the oracle, not a quiet substitution inside a SIMD pass.
    const float mx =
        simd::available() ? simd::vmax(x.data(), n) : *std::max_element(x.begin(), x.begin() + n);
    float sum = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    const float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    if (simd::available()) {
        simd::scale(x.data(), inv, n);
    } else {
        for (std::uint32_t i = 0; i < n; ++i)
            x[i] *= inv;
    }
}

void swiglu(std::span<const float> gate,
            std::span<const float> up,
            std::uint32_t n,
            std::span<float> out) noexcept {
    for (std::uint32_t i = 0; i < n; ++i) {
        const float g = gate[i];
        out[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
}

void geglu(std::span<const float> gate,
           std::span<const float> up,
           std::uint32_t n,
           std::span<float> out) noexcept {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    for (std::uint32_t i = 0; i < n; ++i) {
        const float g = gate[i];
        const float inner = kSqrt2OverPi * (g + 0.044715f * g * g * g);
        out[i] = 0.5f * g * (1.0f + std::tanh(inner)) * up[i];
    }
}

void relu2_glu(std::span<const float> gate,
               std::span<const float> up,
               std::uint32_t n,
               std::span<float> out) noexcept {
    for (std::uint32_t i = 0; i < n; ++i) {
        const float g = std::max(0.0f, gate[i]);
        out[i] = g * g * up[i];
    }
}

void situ_glu(std::span<const float> gate,
              std::span<const float> up,
              std::uint32_t n,
              float beta,
              float linear_beta,
              std::span<float> out) noexcept {
    // Guarded rather than trusted: a zero beta would divide, and the IR's own
    // rule is that an unstated beta means one.
    const float b = beta != 0.0f ? beta : 1.0f;
    const bool clamp_linear = linear_beta != 0.0f;
    for (std::uint32_t i = 0; i < n; ++i) {
        const float g = gate[i];
        const float a = b * std::tanh(g / b) * (1.0f / (1.0f + std::exp(-g)));
        const float u = clamp_linear ? linear_beta * std::tanh(up[i] / linear_beta) : up[i];
        out[i] = a * u;
    }
}

void rope_neox(std::span<float> vec,
               std::uint32_t n_heads,
               std::uint32_t head_dim,
               std::uint32_t position,
               float theta,
               std::uint32_t rotary_dim) noexcept {
    const std::uint32_t rd = (rotary_dim == 0 || rotary_dim > head_dim) ? head_dim : rotary_dim;
    const std::uint32_t half = rd / 2;
    if (half == 0) return;

    // pow/cos/sin depend only on i, not on the head — so the original loop
    // computed each of them n_heads times over. Hoisting is not a SIMD change and
    // not an approximation: the identical values are computed once instead of 32
    // times, and three transcendentals per element dwarf the rotation itself.
    //
    // Bounded stack, no allocation: this is on the per-token path, and a heap
    // allocation here would cost more than the trig it is caching.
    constexpr std::uint32_t kMaxHalf = 256;
    float cs[kMaxHalf], sn[kMaxHalf];
    const bool cached = (half <= kMaxHalf);
    if (cached) {
        for (std::uint32_t i = 0; i < half; ++i) {
            const float inv_freq =
                1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(rd));
            const float angle = static_cast<float>(position) * inv_freq;
            cs[i] = std::cos(angle);
            sn[i] = std::sin(angle);
        }
    }

    for (std::uint32_t h = 0; h < n_heads; ++h) {
        float* v = vec.data() + static_cast<std::size_t>(h) * head_dim;
        for (std::uint32_t i = 0; i < half; ++i) {
            float c, s;
            if (cached) {
                c = cs[i];
                s = sn[i];
            } else {
                const float inv_freq =
                    1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(rd));
                const float angle = static_cast<float>(position) * inv_freq;
                c = std::cos(angle);
                s = std::sin(angle);
            }
            // Pairs are (i, i + rd/2) — the "rotate half" form. Pairing adjacent
            // elements instead is the interleaved variant and yields plausible
            // but wrong output.
            const float a = v[i];
            const float b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = b * c + a * s;
        }
    }
}

void rope_interleaved(std::span<float> vec,
                      std::uint32_t n_heads,
                      std::uint32_t head_dim,
                      std::uint32_t position,
                      float theta,
                      std::uint32_t rotary_dim) noexcept {
    const std::uint32_t rd = (rotary_dim == 0 || rotary_dim > head_dim) ? head_dim : rotary_dim;
    const std::uint32_t pairs = rd / 2;
    for (std::uint32_t h = 0; h < n_heads; ++h) {
        float* v = vec.data() + static_cast<std::size_t>(h) * head_dim;
        for (std::uint32_t i = 0; i < pairs; ++i) {
            const float inv_freq =
                1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(rd));
            const float angle = static_cast<float>(position) * inv_freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float a = v[2 * i];
            const float b = v[2 * i + 1];
            v[2 * i] = a * c - b * s;
            v[2 * i + 1] = b * c + a * s;
        }
    }
}

void top_k(std::span<const float> values,
           std::uint32_t n,
           std::uint32_t k,
           std::span<std::uint32_t> out_indices,
           std::span<float> out_values) noexcept {
    const std::uint32_t kk = std::min(k, n);

    // Index-sorted rather than a heap: torch.topk breaks ties toward the lower
    // index, and a tie broken the other way changes WHICH EXPERT FIRES. That is
    // a semantic divergence from the oracle, not a numeric one, and it would
    // show up as a token mismatch with no numeric drift to explain it.
    static thread_local std::vector<std::uint32_t> order;
    order.resize(n);
    std::iota(order.begin(), order.end(), 0u);
    std::partial_sort(
        order.begin(), order.begin() + kk, order.end(), [&](std::uint32_t a, std::uint32_t b) {
            if (values[a] != values[b]) return values[a] > values[b];
            return a < b;
        });
    for (std::uint32_t i = 0; i < kk; ++i) {
        out_indices[i] = order[i];
        out_values[i] = values[order[i]];
    }
}

} // namespace soma::f32
