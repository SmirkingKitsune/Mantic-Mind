#include "soma/arch/compressed_sparse.hpp"

#include "soma/kernels_f32.hpp"
#include "soma/quant_format.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <vector>

namespace soma::arch::compressed_sparse {
namespace {

struct CompressorWeights {
    std::span<const float> ape;
    WeightRef wkv;
    WeightRef wgate;
    std::span<const float> norm;
};

struct IndexerWeights {
    WeightRef wq_b;
    WeightRef weights_proj;
    CompressorWeights compressor;
};

struct LayerPayload {
    std::span<const float> attn_sink;
    WeightRef wq_a, wq_b, wkv, wo_a, wo_b;
    std::span<const float> q_norm, kv_norm;
    CompressorWeights compressor;
    IndexerWeights indexer;

    std::span<const float> hc_attn_fn, hc_attn_base, hc_attn_scale;
    std::span<const float> hc_ffn_fn, hc_ffn_base, hc_ffn_scale;

    std::span<const float> route_bias;
    std::span<const float> tid2eid;
};

struct ModelPayload {
    std::span<const float> hc_head_fn, hc_head_base, hc_head_scale;
};

struct ForwardState {
    std::uint32_t n_tokens = 0;
    std::uint32_t d_model = 0;
    std::uint32_t hc = 0;
    std::vector<float> streams; // [T,hc,d]
    std::vector<float> post;    // [T,hc]
    std::vector<float> comb;    // [T,hc,hc]
};

template <class T>
void destroy(void* p) {
    delete static_cast<T*>(p);
}

float sigmoid(float x) noexcept {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

float bf16_round(float x) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
    const std::uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    bits &= 0xffff0000u;
    return std::bit_cast<float>(bits);
}

float round_nearest_even_nonnegative(float x) noexcept {
    const float lo = std::floor(x);
    const float fraction = x - lo;
    if (fraction < 0.5f) return lo;
    if (fraction > 0.5f) return lo + 1.0f;
    return std::fmod(lo, 2.0f) == 0.0f ? lo : lo + 1.0f;
}

float fp8_e4m3_round(float x) noexcept {
    if (!std::isfinite(x)) return std::copysign(448.0f, x);
    const float sign = std::signbit(x) ? -1.0f : 1.0f;
    float a = std::min(std::abs(x), 448.0f);
    if (a == 0.0f) return x;
    constexpr float kSubStep = 1.0f / 512.0f;
    // TileLang's cast to E4M3 follows the hardware conversion mode: round to
    // nearest, ties to even.  Spell it out rather than inheriting a process-wide
    // floating-point rounding mode which another library could have changed.
    if (a < 1.0f / 64.0f)
        return sign * round_nearest_even_nonnegative(a / kSubStep) * kSubStep;
    const int e = static_cast<int>(std::floor(std::log2(a)));
    const float step = std::ldexp(1.0f, e - 3);
    a = round_nearest_even_nonnegative(a / step) * step;
    return sign * std::min(a, 448.0f);
}

void fake_fp8(std::span<float> x, std::uint32_t group = 64) noexcept {
    for (std::size_t off = 0; off < x.size(); off += group) {
        const auto n = std::min<std::size_t>(group, x.size() - off);
        float amax = 1e-4f;
        for (std::size_t i = 0; i < n; ++i) amax = std::max(amax, std::abs(x[off + i]));
        const float scale = std::exp2(std::ceil(std::log2(amax / 448.0f)));
        for (std::size_t i = 0; i < n; ++i)
            x[off + i] = bf16_round(fp8_e4m3_round(x[off + i] / scale) * scale);
    }
}

void fake_fp4(std::span<float> x, std::uint32_t group = 32) noexcept {
    static constexpr std::array<float, 8> kLevels{0.0f, 0.5f, 1.0f, 1.5f,
                                                  2.0f, 3.0f, 4.0f, 6.0f};
    for (std::size_t off = 0; off < x.size(); off += group) {
        const auto n = std::min<std::size_t>(group, x.size() - off);
        float amax = std::ldexp(6.0f, -126);
        for (std::size_t i = 0; i < n; ++i) amax = std::max(amax, std::abs(x[off + i]));
        const float scale = std::exp2(std::ceil(std::log2(amax / 6.0f)));
        for (std::size_t i = 0; i < n; ++i) {
            const float a = std::abs(x[off + i] / scale);
            std::size_t best = 0;
            for (std::size_t level = 1; level < kLevels.size(); ++level) {
                const float candidate = std::abs(kLevels[level] - a);
                const float incumbent = std::abs(kLevels[best] - a);
                if (candidate < incumbent ||
                    (candidate == incumbent && (level & 1u) == 0u))
                    best = level;
            }
            x[off + i] =
                bf16_round(std::copysign(kLevels[best] * scale, x[off + i]));
        }
    }
}

void hadamard(std::span<float> x) noexcept {
    const std::size_t n = x.size();
    for (std::size_t width = 1; width < n; width *= 2) {
        for (std::size_t base = 0; base < n; base += 2 * width) {
            for (std::size_t i = 0; i < width; ++i) {
                const float a = x[base + i], b = x[base + width + i];
                x[base + i] = a + b;
                x[base + width + i] = a - b;
            }
        }
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(n));
    for (float& v : x) v = bf16_round(v * scale);
}

void rmsnorm_unweighted(std::span<float> x, float eps) noexcept {
    double ss = 0.0;
    for (const float v : x) ss += static_cast<double>(v) * v;
    const float scale = 1.0f / std::sqrt(static_cast<float>(ss / x.size()) + eps);
    for (float& v : x) v *= scale;
}

void rope(std::span<float> x,
          std::uint32_t position,
          float base,
          float factor,
          std::uint32_t original,
          float beta_fast,
          float beta_slow,
          bool inverse = false) noexcept {
    const auto dim = static_cast<std::uint32_t>(x.size());
    auto corr_dim = [&](float rotations) {
        return dim * std::log(original / (rotations * 2.0f * std::numbers::pi_v<float>)) /
               (2.0f * std::log(base));
    };
    int low = 0, high = -1;
    if (original > 0) {
        low = std::max(0, static_cast<int>(std::floor(corr_dim(beta_fast))));
        high = std::min(static_cast<int>(dim) - 1,
                        static_cast<int>(std::ceil(corr_dim(beta_slow))));
    }
    for (std::uint32_t i = 0; i < dim / 2; ++i) {
        float freq = std::pow(base, -2.0f * i / dim);
        if (original > 0) {
            const float ramp = std::clamp((static_cast<float>(i) - low) /
                                              std::max(0.001f, static_cast<float>(high - low)),
                                          0.0f,
                                          1.0f);
            const float smooth = 1.0f - ramp;
            freq = freq / factor * (1.0f - smooth) + freq * smooth;
        }
        float angle = static_cast<float>(position) * freq;
        if (inverse) angle = -angle;
        const float c = std::cos(angle), s = std::sin(angle);
        const float a = x[2 * i], b = x[2 * i + 1];
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

StatusCode bind_compressor(const ArchIr&,
                           const LayerBindCtx& ctx,
                           const std::string& prefix,
                           CompressorWeights& out) noexcept {
    if (!bind_layer_f32(ctx, (prefix + "ape").c_str(), out.ape).ok() ||
        !bind_layer_weight(ctx,
                           (prefix + "wkv.weight").c_str(),
                           TensorRole::AttnProj,
                           out.wkv)
             .ok() ||
        !bind_layer_weight(ctx,
                           (prefix + "wgate.weight").c_str(),
                           TensorRole::AttnProj,
                           out.wgate)
             .ok() ||
        !bind_layer_f32(ctx, (prefix + "norm.weight").c_str(), out.norm).ok())
        return StatusCode::NotFound;
    return StatusCode::Ok;
}

StatusCode bind_layer(const ArchIr& arch,
                      const LayerBindCtx& ctx,
                      ArchLayerPayload& out) noexcept {
    auto* p = new LayerPayload();
    const auto fail = [&]() {
        delete p;
        return StatusCode::NotFound;
    };
    if (!bind_layer_f32(ctx, "self_attn.attn_sink", p->attn_sink).ok() ||
        !bind_layer_weight(ctx, "self_attn.wq_a.weight", TensorRole::AttnProj, p->wq_a).ok() ||
        !bind_layer_weight(ctx, "self_attn.wq_b.weight", TensorRole::AttnProj, p->wq_b).ok() ||
        !bind_layer_f32(ctx, "self_attn.q_norm.weight", p->q_norm).ok() ||
        !bind_layer_weight(ctx, "self_attn.wkv.weight", TensorRole::AttnProj, p->wkv).ok() ||
        !bind_layer_f32(ctx, "self_attn.kv_norm.weight", p->kv_norm).ok() ||
        !bind_layer_weight(ctx, "self_attn.wo_a.weight", TensorRole::AttnProj, p->wo_a).ok() ||
        !bind_layer_weight(ctx, "self_attn.wo_b.weight", TensorRole::AttnProj, p->wo_b).ok())
        return fail();
    if (bind_compressor(arch, ctx, "self_attn.compressor.", p->compressor) != StatusCode::Ok)
        return fail();
    const auto ratio = arch.attention.compressed.compress_ratios[ctx.layer];
    if (ratio == 4) {
        if (!bind_layer_weight(ctx,
                               "self_attn.indexer.wq_b.weight",
                               TensorRole::AttnProj,
                               p->indexer.wq_b)
                 .ok() ||
            !bind_layer_weight(ctx,
                               "self_attn.indexer.weights_proj.weight",
                               TensorRole::AttnProj,
                               p->indexer.weights_proj)
                 .ok() ||
            bind_compressor(arch,
                            ctx,
                            "self_attn.indexer.compressor.",
                            p->indexer.compressor) != StatusCode::Ok)
            return fail();
    }
    if (!bind_layer_f32(ctx, "hc_attn_fn", p->hc_attn_fn).ok() ||
        !bind_layer_f32(ctx, "hc_attn_base", p->hc_attn_base).ok() ||
        !bind_layer_f32(ctx, "hc_attn_scale", p->hc_attn_scale).ok() ||
        !bind_layer_f32(ctx, "hc_ffn_fn", p->hc_ffn_fn).ok() ||
        !bind_layer_f32(ctx, "hc_ffn_base", p->hc_ffn_base).ok() ||
        !bind_layer_f32(ctx, "hc_ffn_scale", p->hc_ffn_scale).ok())
        return fail();
    if (ctx.layer < arch.router.n_hash_layers) {
        if (!bind_layer_f32(ctx, "ffn.gate.tid2eid", p->tid2eid).ok()) return fail();
    } else if (!bind_layer_f32(ctx, "ffn.gate.bias", p->route_bias).ok()) {
        return fail();
    }
    out.adopt(p, &destroy<LayerPayload>);
    return StatusCode::Ok;
}

StatusCode bind_model(const ArchIr&,
                      const ModelBindCtx& ctx,
                      ArchLayerPayload& out) noexcept {
    auto* p = new ModelPayload();
    if (!bind_model_f32(ctx, "model.hc_head_fn", p->hc_head_fn).ok() ||
        !bind_model_f32(ctx, "model.hc_head_base", p->hc_head_base).ok() ||
        !bind_model_f32(ctx, "model.hc_head_scale", p->hc_head_scale).ok()) {
        delete p;
        return StatusCode::NotFound;
    }
    out.adopt(p, &destroy<ModelPayload>);
    return StatusCode::Ok;
}

StatusCode begin_forward(const ArchIr& arch,
                         const ArchLayerPayload&,
                         const TokenId*,
                         std::uint32_t n_tokens,
                         F32Workspace& ws,
                         float* hidden) noexcept {
    auto* st = new ForwardState();
    st->n_tokens = n_tokens;
    st->d_model = arch.topology.d_model;
    st->hc = arch.hyper_connections.multiplier;
    st->streams.resize(static_cast<std::size_t>(n_tokens) * st->hc * st->d_model);
    st->post.resize(static_cast<std::size_t>(n_tokens) * st->hc);
    st->comb.resize(static_cast<std::size_t>(n_tokens) * st->hc * st->hc);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        for (std::uint32_t h = 0; h < st->hc; ++h) {
            std::copy_n(hidden + static_cast<std::size_t>(t) * st->d_model,
                        st->d_model,
                        st->streams.data() +
                            (static_cast<std::size_t>(t) * st->hc + h) * st->d_model);
        }
    }
    ws.arch_state.adopt(st, &destroy<ForwardState>);
    return StatusCode::Ok;
}

StatusCode hc_pre(const ArchIr& arch,
                  std::span<const float> fn,
                  std::span<const float> base,
                  std::span<const float> scale,
                  std::uint32_t n_tokens,
                  F32Workspace& ws,
                  float* hidden) noexcept {
    auto* st = ws.arch_state.as<ForwardState>();
    if (st == nullptr || st->n_tokens != n_tokens || scale.size() < 3) return StatusCode::Internal;
    const auto d = st->d_model, hc = st->hc;
    const auto flat = hc * d;
    const auto mix = (2 + hc) * hc;
    if (fn.size() != static_cast<std::size_t>(mix) * flat || base.size() != mix)
        return StatusCode::InvalidArgument;

    std::vector<float> mixes(mix);
    std::vector<float> matrix(hc * hc);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        const float* streams = st->streams.data() + static_cast<std::size_t>(t) * flat;
        double ss = 0.0;
        for (std::uint32_t i = 0; i < flat; ++i) ss += static_cast<double>(streams[i]) * streams[i];
        const float rsqrt = 1.0f / std::sqrt(static_cast<float>(ss / flat) + arch.rms_norm_eps);
        for (std::uint32_t m = 0; m < mix; ++m) {
            double acc = 0.0;
            const float* row = fn.data() + static_cast<std::size_t>(m) * flat;
            for (std::uint32_t i = 0; i < flat; ++i) acc += static_cast<double>(row[i]) * streams[i];
            mixes[m] = static_cast<float>(acc) * rsqrt;
        }

        float* post = st->post.data() + static_cast<std::size_t>(t) * hc;
        float* comb = st->comb.data() + static_cast<std::size_t>(t) * hc * hc;
        for (std::uint32_t h = 0; h < hc; ++h) {
            const float pre = sigmoid(mixes[h] * scale[0] + base[h]) +
                              arch.hyper_connections.eps;
            post[h] = 2.0f * sigmoid(mixes[hc + h] * scale[1] + base[hc + h]);
            const float* src = streams + static_cast<std::size_t>(h) * d;
            float* dst = hidden + static_cast<std::size_t>(t) * d;
            for (std::uint32_t i = 0; i < d; ++i) dst[i] += (h == 0 ? 0.0f : pre * src[i]);
            if (h == 0)
                for (std::uint32_t i = 0; i < d; ++i) dst[i] = pre * src[i];
        }

        for (std::uint32_t r = 0; r < hc; ++r) {
            float mx = -std::numeric_limits<float>::infinity();
            for (std::uint32_t c = 0; c < hc; ++c) {
                const auto k = r * hc + c;
                matrix[k] = mixes[2 * hc + k] * scale[2] + base[2 * hc + k];
                mx = std::max(mx, matrix[k]);
            }
            float sum = 0.0f;
            for (std::uint32_t c = 0; c < hc; ++c) sum += std::exp(matrix[r * hc + c] - mx);
            for (std::uint32_t c = 0; c < hc; ++c)
                matrix[r * hc + c] = std::exp(matrix[r * hc + c] - mx) / sum +
                                      arch.hyper_connections.eps;
        }
        const auto normalize_cols = [&]() {
            for (std::uint32_t c = 0; c < hc; ++c) {
                float sum = arch.hyper_connections.eps;
                for (std::uint32_t r = 0; r < hc; ++r) sum += matrix[r * hc + c];
                for (std::uint32_t r = 0; r < hc; ++r) matrix[r * hc + c] /= sum;
            }
        };
        const auto normalize_rows = [&]() {
            for (std::uint32_t r = 0; r < hc; ++r) {
                float sum = arch.hyper_connections.eps;
                for (std::uint32_t c = 0; c < hc; ++c) sum += matrix[r * hc + c];
                for (std::uint32_t c = 0; c < hc; ++c) matrix[r * hc + c] /= sum;
            }
        };
        normalize_cols();
        for (std::uint32_t i = 1; i < arch.hyper_connections.sinkhorn_iters; ++i) {
            normalize_rows();
            normalize_cols();
        }
        std::copy(matrix.begin(), matrix.end(), comb);
    }
    return StatusCode::Ok;
}

StatusCode hc_merge(const ArchIr&,
                    const float* branch,
                    std::uint32_t n_tokens,
                    F32Workspace& ws,
                    float* hidden) noexcept {
    auto* st = ws.arch_state.as<ForwardState>();
    if (st == nullptr || st->n_tokens != n_tokens) return StatusCode::Internal;
    const auto d = st->d_model, hc = st->hc;
    std::vector<float> next(st->streams.size());
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        const float* old = st->streams.data() + static_cast<std::size_t>(t) * hc * d;
        const float* post = st->post.data() + static_cast<std::size_t>(t) * hc;
        const float* comb = st->comb.data() + static_cast<std::size_t>(t) * hc * hc;
        for (std::uint32_t o = 0; o < hc; ++o) {
            float* dst = next.data() + (static_cast<std::size_t>(t) * hc + o) * d;
            for (std::uint32_t k = 0; k < d; ++k) {
                double v = static_cast<double>(post[o]) * branch[static_cast<std::size_t>(t) * d + k];
                for (std::uint32_t i = 0; i < hc; ++i)
                    v += static_cast<double>(comb[o * hc + i]) * old[static_cast<std::size_t>(i) * d + k];
                dst[k] = static_cast<float>(v);
            }
        }
        std::copy_n(next.data() + static_cast<std::size_t>(t) * hc * d,
                    d,
                    hidden + static_cast<std::size_t>(t) * d);
    }
    st->streams.swap(next);
    return StatusCode::Ok;
}

StatusCode pre_attention(const ArchIr& arch,
                         const F32LayerWeights& w,
                         std::uint32_t n,
                         F32Workspace& ws,
                         float* hidden) noexcept {
    const auto* p = w.attn.as<LayerPayload>();
    if (p == nullptr) return StatusCode::Internal;
    const auto rc = hc_pre(arch, p->hc_attn_fn, p->hc_attn_base, p->hc_attn_scale, n, ws, hidden);
    if (rc == StatusCode::Ok)
        ws.sink(ws.current_layer,
                "hc_attn_pre",
                hidden,
                static_cast<std::size_t>(n) * arch.topology.d_model);
    return rc;
}

StatusCode merge_attention(const ArchIr& arch,
                           const F32LayerWeights&,
                           const float* branch,
                           std::uint32_t n,
                           F32Workspace& ws,
                           float* hidden) noexcept {
    const auto rc = hc_merge(arch, branch, n, ws, hidden);
    if (rc == StatusCode::Ok) {
        const auto* st = ws.arch_state.as<ForwardState>();
        if (st != nullptr)
            ws.sink(ws.current_layer, "hc_attn_streams", st->streams.data(), st->streams.size());
    }
    return rc;
}

StatusCode pre_ffn(const ArchIr& arch,
                   const F32LayerWeights& w,
                   std::uint32_t n,
                   F32Workspace& ws,
                   float* hidden) noexcept {
    const auto* p = w.attn.as<LayerPayload>();
    if (p == nullptr) return StatusCode::Internal;
    const auto rc = hc_pre(arch, p->hc_ffn_fn, p->hc_ffn_base, p->hc_ffn_scale, n, ws, hidden);
    if (rc == StatusCode::Ok)
        ws.sink(ws.current_layer,
                "hc_ffn_pre",
                hidden,
                static_cast<std::size_t>(n) * arch.topology.d_model);
    return rc;
}

StatusCode merge_ffn(const ArchIr& arch,
                     const F32LayerWeights&,
                     const float* branch,
                     std::uint32_t n,
                     F32Workspace& ws,
                     float* hidden) noexcept {
    const auto rc = hc_merge(arch, branch, n, ws, hidden);
    if (rc == StatusCode::Ok) {
        const auto* st = ws.arch_state.as<ForwardState>();
        if (st != nullptr)
            ws.sink(ws.current_layer, "hc_ffn_streams", st->streams.data(), st->streams.size());
    }
    return rc;
}

StatusCode export_layer_hidden(const ArchIr& arch,
                               LayerIndex,
                               std::uint32_t n,
                               const F32Workspace& ws,
                               const float*,
                               float* out) noexcept {
    const auto* st = ws.arch_state.as<ForwardState>();
    if (st == nullptr || out == nullptr || st->n_tokens != n ||
        st->d_model != arch.topology.d_model ||
        st->hc != arch.hyper_connections.multiplier) {
        return StatusCode::InvalidArgument;
    }
    const auto d = st->d_model, hc = st->hc;
    for (std::uint32_t t = 0; t < n; ++t) {
        const float* streams = st->streams.data() +
                               static_cast<std::size_t>(t) * hc * d;
        float* dst = out + static_cast<std::size_t>(t) * d;
        for (std::uint32_t k = 0; k < d; ++k) {
            double sum = 0.0;
            for (std::uint32_t h = 0; h < hc; ++h)
                sum += streams[static_cast<std::size_t>(h) * d + k];
            dst[k] = static_cast<float>(sum / hc);
        }
    }
    return StatusCode::Ok;
}

StatusCode end_forward(const ArchIr& arch,
                       const ArchLayerPayload& model_payload,
                       std::uint32_t n,
                       F32Workspace& ws,
                       float* hidden) noexcept {
    const auto* p = model_payload.as<ModelPayload>();
    auto* st = ws.arch_state.as<ForwardState>();
    if (p == nullptr || st == nullptr || p->hc_head_scale.empty()) return StatusCode::Internal;
    const auto d = st->d_model, hc = st->hc, flat = hc * d;
    if (p->hc_head_fn.size() != static_cast<std::size_t>(hc) * flat ||
        p->hc_head_base.size() != hc)
        return StatusCode::InvalidArgument;
    for (std::uint32_t t = 0; t < n; ++t) {
        const float* streams = st->streams.data() + static_cast<std::size_t>(t) * flat;
        double ss = 0.0;
        for (std::uint32_t i = 0; i < flat; ++i) ss += static_cast<double>(streams[i]) * streams[i];
        const float rsqrt = 1.0f / std::sqrt(static_cast<float>(ss / flat) + arch.rms_norm_eps);
        std::fill_n(hidden + static_cast<std::size_t>(t) * d, d, 0.0f);
        for (std::uint32_t h = 0; h < hc; ++h) {
            double mix = 0.0;
            const float* row = p->hc_head_fn.data() + static_cast<std::size_t>(h) * flat;
            for (std::uint32_t i = 0; i < flat; ++i) mix += static_cast<double>(row[i]) * streams[i];
            const float pre = sigmoid(static_cast<float>(mix) * rsqrt * p->hc_head_scale[0] +
                                      p->hc_head_base[h]) +
                              arch.hyper_connections.eps;
            for (std::uint32_t k = 0; k < d; ++k)
                hidden[static_cast<std::size_t>(t) * d + k] +=
                    pre * streams[static_cast<std::size_t>(h) * d + k];
        }
    }
    return StatusCode::Ok;
}

StatusCode route(const ArchIr& arch,
                 const F32LayerWeights& w,
                 const TokenId* input_tokens,
                 const float* logits,
                 std::uint32_t n_tokens,
                 std::uint32_t* out_ids,
                 float* out_weights) noexcept {
    const auto* p = w.attn.as<LayerPayload>();
    if (p == nullptr || input_tokens == nullptr) return StatusCode::Internal;
    const auto E = arch.router.n_experts, K = arch.router.top_k;
    std::vector<float> original(E), selected(E);
    std::vector<std::uint32_t> ids(K);
    std::vector<float> values(K);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        for (std::uint32_t e = 0; e < E; ++e) {
            original[e] = std::sqrt(std::log1p(std::exp(-std::abs(logits[t * E + e]))) +
                                    std::max(logits[t * E + e], 0.0f));
            selected[e] = original[e] + (p->route_bias.empty() ? 0.0f : p->route_bias[e]);
        }
        if (!p->tid2eid.empty()) {
            const auto id = input_tokens[t];
            if (id >= arch.topology.vocab_size) return StatusCode::InvalidArgument;
            for (std::uint32_t k = 0; k < K; ++k)
                ids[k] = static_cast<std::uint32_t>(p->tid2eid[static_cast<std::size_t>(id) * K + k]);
        } else {
            f32::top_k(selected, E, K, ids, values);
        }
        float sum = 0.0f;
        for (std::uint32_t k = 0; k < K; ++k) sum += original[ids[k]];
        for (std::uint32_t k = 0; k < K; ++k) {
            out_ids[static_cast<std::size_t>(t) * K + k] = ids[k];
            out_weights[static_cast<std::size_t>(t) * K + k] =
                original[ids[k]] / std::max(sum, std::numeric_limits<float>::min()) *
                arch.router.routed_scaling_factor;
        }
    }
    return StatusCode::Ok;
}

std::vector<float> compress_sequence(const ArchIr& arch,
                                     const CompressorWeights& w,
                                     std::uint32_t ratio,
                                     std::uint32_t head_dim,
                                     bool rotate,
                                     const float* x,
                                     std::uint32_t n_tokens) {
    const auto d = arch.topology.d_model;
    const auto coff = ratio == 4 ? 2u : 1u;
    std::vector<float> values(static_cast<std::size_t>(n_tokens) * coff * head_dim);
    std::vector<float> scores(values.size());
    matmul(w.wkv, std::span<const float>(x, static_cast<std::size_t>(n_tokens) * d), n_tokens, values);
    matmul(w.wgate,
           std::span<const float>(x, static_cast<std::size_t>(n_tokens) * d),
           n_tokens,
           scores);
    const auto count = n_tokens / ratio;
    std::vector<float> out(static_cast<std::size_t>(count) * head_dim);
    std::vector<float> logits(2 * ratio);
    for (std::uint32_t b = 0; b < count; ++b) {
        float* dst = out.data() + static_cast<std::size_t>(b) * head_dim;
        for (std::uint32_t k = 0; k < head_dim; ++k) {
            std::uint32_t nc = 0;
            if (coff == 2 && b > 0) {
                for (std::uint32_t r = 0; r < ratio; ++r) {
                    const auto tok = (b - 1) * ratio + r;
                    logits[nc++] = scores[static_cast<std::size_t>(tok) * 2 * head_dim + k] +
                                   w.ape[static_cast<std::size_t>(r) * 2 * head_dim + k];
                }
            }
            const auto column = (coff == 2 ? head_dim : 0u) + k;
            for (std::uint32_t r = 0; r < ratio; ++r) {
                const auto tok = b * ratio + r;
                logits[nc++] = scores[static_cast<std::size_t>(tok) * coff * head_dim + column] +
                               w.ape[static_cast<std::size_t>(r) * coff * head_dim + column];
            }
            const float mx = *std::max_element(logits.begin(), logits.begin() + nc);
            float denom = 0.0f;
            for (std::uint32_t i = 0; i < nc; ++i) denom += std::exp(logits[i] - mx);
            double acc = 0.0;
            std::uint32_t i = 0;
            if (coff == 2 && b > 0) {
                for (std::uint32_t r = 0; r < ratio; ++r, ++i) {
                    const auto tok = (b - 1) * ratio + r;
                    acc += std::exp(logits[i] - mx) / denom *
                           values[static_cast<std::size_t>(tok) * 2 * head_dim + k];
                }
            }
            for (std::uint32_t r = 0; r < ratio; ++r, ++i) {
                const auto tok = b * ratio + r;
                acc += std::exp(logits[i] - mx) / denom *
                       values[static_cast<std::size_t>(tok) * coff * head_dim + column];
            }
            dst[k] = static_cast<float>(acc);
        }
        f32::rmsnorm(std::span<float>(dst, head_dim), w.norm, head_dim, arch.rms_norm_eps);
        const auto rd = arch.attention.compressed.rope_head_dim;
        rope(std::span<float>(dst + head_dim - rd, rd),
             b * ratio,
             arch.attention.compressed.compress_rope_theta,
             arch.attention.rope.scaling.factor,
             arch.attention.rope.scaling.original_max_position,
             arch.attention.rope.scaling.beta_fast,
             arch.attention.rope.scaling.beta_slow);
        if (rotate) {
            if (arch.attention.compressed.semantic_fp4_quant_dequant) {
                hadamard(std::span<float>(dst, head_dim));
                fake_fp4(std::span<float>(dst, head_dim));
            }
        } else {
            if (arch.attention.compressed.semantic_fp8_quant_dequant) {
                fake_fp8(std::span<float>(dst, head_dim - rd));
                for (std::uint32_t k = head_dim - rd; k < head_dim; ++k)
                    dst[k] = bf16_round(dst[k]);
            }
        }
    }
    return out;
}

StatusCode attention(const ArchIr& arch,
                     const F32LayerWeights& lw,
                     const float* x,
                     std::uint32_t n_tokens,
                     F32Workspace& ws,
                     float* out) noexcept {
    const auto* p = lw.attn.as<LayerPayload>();
    if (p == nullptr) return StatusCode::Internal;
    const auto& c = arch.attention.compressed;
    const auto d = arch.topology.d_model, H = arch.attention.n_heads;
    const auto D = arch.attention.head_dim, rd = c.rope_head_dim;
    const auto qr = c.q_lora_rank, ratio = c.compress_ratios[ws.current_layer];
    const auto original = arch.attention.rope.scaling.original_max_position;
    const auto factor = arch.attention.rope.scaling.factor;

    std::vector<float> qlow(static_cast<std::size_t>(n_tokens) * qr);
    matmul(p->wq_a, std::span<const float>(x, static_cast<std::size_t>(n_tokens) * d), n_tokens, qlow);
    for (std::uint32_t t = 0; t < n_tokens; ++t)
        f32::rmsnorm(std::span<float>(qlow).subspan(static_cast<std::size_t>(t) * qr, qr),
                     p->q_norm,
                     qr,
                     arch.rms_norm_eps);
    matmul(p->wq_b, qlow, n_tokens, ws.q);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        for (std::uint32_t h = 0; h < H; ++h) {
            float* q = ws.q.data() + (static_cast<std::size_t>(t) * H + h) * D;
            rmsnorm_unweighted(std::span<float>(q, D), arch.rms_norm_eps);
            rope(std::span<float>(q + D - rd, rd),
                 t,
                 c.compress_rope_theta,
                 factor,
                 original,
                 arch.attention.rope.scaling.beta_fast,
                 arch.attention.rope.scaling.beta_slow);
        }
    }
    matmul(p->wkv, std::span<const float>(x, static_cast<std::size_t>(n_tokens) * d), n_tokens, ws.k);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        float* kv = ws.k.data() + static_cast<std::size_t>(t) * D;
        f32::rmsnorm(std::span<float>(kv, D), p->kv_norm, D, arch.rms_norm_eps);
        rope(std::span<float>(kv + D - rd, rd),
             t,
             c.compress_rope_theta,
             factor,
             original,
             arch.attention.rope.scaling.beta_fast,
             arch.attention.rope.scaling.beta_slow);
        if (c.semantic_fp8_quant_dequant) {
            fake_fp8(std::span<float>(kv, D - rd));
            for (std::uint32_t k = D - rd; k < D; ++k) kv[k] = bf16_round(kv[k]);
        }
    }

    auto compressed =
        compress_sequence(arch, p->compressor, ratio, D, false, x, n_tokens);
    if (!compressed.empty())
        ws.sink(ws.current_layer, "compressor_kv", compressed.data(), compressed.size());
    std::vector<std::vector<std::uint32_t>> chosen(n_tokens);
    if (ratio == 4) {
        const auto IH = c.index_n_heads, ID = c.index_head_dim;
        auto ikv = compress_sequence(arch, p->indexer.compressor, ratio, ID, true, x, n_tokens);
        if (!ikv.empty()) ws.sink(ws.current_layer, "indexer_kv", ikv.data(), ikv.size());
        std::vector<float> iq(static_cast<std::size_t>(n_tokens) * IH * ID);
        std::vector<float> iw(static_cast<std::size_t>(n_tokens) * IH);
        matmul(p->indexer.wq_b, qlow, n_tokens, iq);
        matmul(p->indexer.weights_proj,
               std::span<const float>(x, static_cast<std::size_t>(n_tokens) * d),
               n_tokens,
               iw);
        const float weight_scale = 1.0f / std::sqrt(static_cast<float>(ID * IH));
        std::vector<float> scores;
        std::vector<std::uint32_t> ids;
        std::vector<float> vals;
        for (std::uint32_t t = 0; t < n_tokens; ++t) {
            for (std::uint32_t h = 0; h < IH; ++h) {
                float* q = iq.data() + (static_cast<std::size_t>(t) * IH + h) * ID;
                rope(std::span<float>(q + ID - rd, rd),
                     t,
                     c.compress_rope_theta,
                     factor,
                     original,
                     arch.attention.rope.scaling.beta_fast,
                     arch.attention.rope.scaling.beta_slow);
                if (c.semantic_fp4_quant_dequant) {
                    hadamard(std::span<float>(q, ID));
                    fake_fp4(std::span<float>(q, ID));
                }
            }
            const auto eligible = (t + 1) / ratio;
            if (eligible == 0) continue;
            scores.assign(eligible, 0.0f);
            for (std::uint32_t k = 0; k < eligible; ++k) {
                double score = 0.0;
                for (std::uint32_t h = 0; h < IH; ++h) {
                    const float* q = iq.data() + (static_cast<std::size_t>(t) * IH + h) * ID;
                    const float dot = f32::dot(std::span<const float>(q, ID),
                                               std::span<const float>(ikv.data() +
                                                                          static_cast<std::size_t>(k) * ID,
                                                                      ID),
                                               ID);
                    score += std::max(0.0f, dot) * iw[static_cast<std::size_t>(t) * IH + h] *
                             weight_scale;
                }
                scores[k] = static_cast<float>(score);
            }
            const auto take = std::min<std::uint32_t>(c.index_topk, eligible);
            ids.resize(take);
            vals.resize(take);
            f32::top_k(scores, eligible, take, ids, vals);
            chosen[t] = ids;
        }
    }

    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<std::uint32_t> keys;
    std::vector<float> logits;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        keys.clear();
        const auto first = t + 1 > arch.attention.sliding_window
                               ? t + 1 - arch.attention.sliding_window
                               : 0;
        for (std::uint32_t k = first; k <= t; ++k) keys.push_back(k);
        const auto eligible = (t + 1) / ratio;
        if (ratio == 4) {
            for (const auto k : chosen[t]) keys.push_back(n_tokens + k);
        } else {
            for (std::uint32_t k = 0; k < eligible; ++k) keys.push_back(n_tokens + k);
        }
        logits.resize(keys.size());
        for (std::uint32_t h = 0; h < H; ++h) {
            const float* q = ws.q.data() + (static_cast<std::size_t>(t) * H + h) * D;
            float* o = ws.attn_heads.data() + (static_cast<std::size_t>(t) * H + h) * D;
            std::fill_n(o, D, 0.0f);
            float mx = p->attn_sink[h];
            for (std::size_t i = 0; i < keys.size(); ++i) {
                const auto k = keys[i];
                const float* kv = k < n_tokens ? ws.k.data() + static_cast<std::size_t>(k) * D
                                               : compressed.data() +
                                                     static_cast<std::size_t>(k - n_tokens) * D;
                logits[i] = f32::dot(std::span<const float>(q, D),
                                     std::span<const float>(kv, D),
                                     D) *
                            softmax_scale;
                mx = std::max(mx, logits[i]);
            }
            float denom = std::exp(p->attn_sink[h] - mx);
            for (const float v : logits) denom += std::exp(v - mx);
            for (std::size_t i = 0; i < keys.size(); ++i) {
                const auto k = keys[i];
                const float* kv = k < n_tokens ? ws.k.data() + static_cast<std::size_t>(k) * D
                                               : compressed.data() +
                                                     static_cast<std::size_t>(k - n_tokens) * D;
                f32::axpy(std::exp(logits[i] - mx) / denom,
                          std::span<const float>(kv, D),
                          D,
                          std::span<float>(o, D));
            }
            rope(std::span<float>(o + D - rd, rd),
                 t,
                 c.compress_rope_theta,
                 factor,
                 original,
                 arch.attention.rope.scaling.beta_fast,
                 arch.attention.rope.scaling.beta_slow,
                 true);
        }
    }

    const auto G = c.o_groups, R = c.o_lora_rank, heads_per_group = H / G;
    std::vector<float> low(static_cast<std::size_t>(n_tokens) * G * R);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        for (std::uint32_t g = 0; g < G; ++g) {
            const auto wg = row_block(p->wo_a, g * R, R);
            matvec(wg,
                   std::span<const float>(ws.attn_heads)
                       .subspan((static_cast<std::size_t>(t) * H + g * heads_per_group) * D,
                                heads_per_group * D),
                   std::span<float>(low).subspan((static_cast<std::size_t>(t) * G + g) * R, R));
        }
    }
    matmul(p->wo_b, low, n_tokens, std::span<float>(out, static_cast<std::size_t>(n_tokens) * d));
    return StatusCode::Ok;
}

std::size_t align64(std::size_t n) noexcept { return (n + 63u) & ~std::size_t{63u}; }

struct LayerRegion {
    std::size_t window = 0;
    std::size_t compressed = 0;
    std::size_t index = 0;
    std::size_t state_values = 0;
    std::size_t state_scores = 0;
    std::size_t index_state_values = 0;
    std::size_t index_state_scores = 0;
    std::size_t end = 0;
};

LayerRegion region_for(const ArchIr& arch,
                       std::uint32_t layer,
                       std::uint32_t ctx,
                       std::size_t base) noexcept {
    const auto& c = arch.attention.compressed;
    const auto D = arch.attention.head_dim, ID = c.index_head_dim;
    const auto ratio = c.compress_ratios[layer], coff = ratio == 4 ? 2u : 1u;
    LayerRegion r;
    std::size_t at = align64(base);
    r.window = at;
    at = align64(at + static_cast<std::size_t>(arch.attention.sliding_window) * D * 2);
    r.compressed = at;
    at = align64(at + static_cast<std::size_t>((ctx + ratio - 1) / ratio) * D * 2);
    if (ratio == 4) {
        r.index = at;
        at = align64(at + static_cast<std::size_t>((ctx + 3) / 4) * ID * 2);
    }
    const auto state_floats = static_cast<std::size_t>(coff) * ratio * coff * D;
    r.state_values = at;
    at = align64(at + state_floats * sizeof(float));
    r.state_scores = at;
    at = align64(at + state_floats * sizeof(float));
    if (ratio == 4) {
        const auto idx_floats = static_cast<std::size_t>(2) * 4 * 2 * ID;
        r.index_state_values = at;
        at = align64(at + idx_floats * sizeof(float));
        r.index_state_scores = at;
        at = align64(at + idx_floats * sizeof(float));
    }
    r.end = at;
    return r;
}

LayerRegion layer_region(const ArchIr& arch, std::uint32_t layer, std::uint32_t ctx) noexcept {
    std::size_t at = 0;
    LayerRegion r;
    for (std::uint32_t l = 0; l <= layer; ++l) {
        r = region_for(arch, l, ctx, at);
        at = r.end;
    }
    return r;
}

std::uint64_t kv_bytes_for_context(const ArchIr& arch, std::uint32_t ctx) noexcept {
    std::size_t at = 0;
    for (std::uint32_t l = 0; l < arch.topology.n_layers; ++l)
        at = region_for(arch, l, ctx, at).end;
    return at;
}

std::size_t kv_bytes_per_token(const ArchIr& arch) noexcept {
    const auto a = kv_bytes_for_context(arch, 65536);
    const auto b = kv_bytes_for_context(arch, 65535);
    return static_cast<std::size_t>(a - b);
}

Status serialize_kv(const ArchIr& arch,
                    std::span<const std::byte> source,
                    std::uint32_t source_context,
                    std::uint32_t length,
                    std::vector<std::byte>& payload) {
    if (length > source_context) {
        return {StatusCode::InvalidArgument, "opaque KV length exceeds source context"};
    }
    try {
        payload.clear();
        const auto& c = arch.attention.compressed;
        const auto D = arch.attention.head_dim, ID = c.index_head_dim;
        const auto W = arch.attention.sliding_window;
        const auto append = [&](std::size_t at, std::size_t bytes) -> bool {
            if (at > source.size() || bytes > source.size() - at) return false;
            payload.insert(payload.end(), source.begin() + at, source.begin() + at + bytes);
            return true;
        };
        const auto append_live_carry = [&](std::size_t values_at,
                                           std::size_t scores_at,
                                           std::uint32_t ratio,
                                           std::uint32_t coff,
                                           std::uint32_t hd) -> bool {
            const auto complete = length / ratio;
            const auto partial = length % ratio;
            const auto row_bytes = static_cast<std::size_t>(coff) * hd * sizeof(float);
            if (coff == 1) {
                const auto bytes = static_cast<std::size_t>(partial) * row_bytes;
                return bytes == 0 || (append(values_at, bytes) && append(scores_at, bytes));
            }
            // Ratio-4 compression overlaps adjacent groups. Once at least one
            // group has completed, its four rows remain live until the next
            // group completes; only the populated prefix of the current group
            // is additionally live.
            if (complete > 0) {
                const auto bytes = static_cast<std::size_t>(ratio) * row_bytes;
                if (!append(values_at, bytes) || !append(scores_at, bytes)) return false;
            }
            if (partial > 0) {
                const auto offset = static_cast<std::size_t>(ratio) * row_bytes;
                const auto bytes = static_cast<std::size_t>(partial) * row_bytes;
                if (!append(values_at + offset, bytes) ||
                    !append(scores_at + offset, bytes)) return false;
            }
            return true;
        };
        for (std::uint32_t layer = 0; layer < arch.topology.n_layers; ++layer) {
            const auto reg = layer_region(arch, layer, source_context);
            if (reg.end > source.size()) {
                return {StatusCode::InvalidArgument, "opaque KV source geometry is truncated"};
            }
            const auto ratio = c.compress_ratios[layer], coff = ratio == 4 ? 2u : 1u;
            const auto live_window = std::min(length, W);
            const auto first_window = length - live_window;
            for (std::uint32_t pos = first_window; pos < length; ++pos) {
                if (!append(reg.window + static_cast<std::size_t>(pos % W) * D * 2,
                            static_cast<std::size_t>(D) * 2)) {
                    return {StatusCode::InvalidArgument, "opaque KV window is truncated"};
                }
            }
            const auto completed = length / ratio;
            if (!append(reg.compressed,
                        static_cast<std::size_t>(completed) * D * 2)) {
                return {StatusCode::InvalidArgument, "opaque compressed history is truncated"};
            }
            if (ratio == 4 &&
                !append(reg.index, static_cast<std::size_t>(completed) * ID * 2)) {
                return {StatusCode::InvalidArgument, "opaque index history is truncated"};
            }
            if (!append_live_carry(reg.state_values,
                                   reg.state_scores,
                                   ratio,
                                   coff,
                                   D)) {
                return {StatusCode::InvalidArgument, "opaque compressor carry is truncated"};
            }
            if (ratio == 4 &&
                !append_live_carry(reg.index_state_values,
                                   reg.index_state_scores,
                                   4,
                                   2,
                                   ID)) {
                return {StatusCode::InvalidArgument,
                        "opaque index compressor carry is truncated"};
            }
        }
        return {};
    } catch (const std::bad_alloc&) {
        return {StatusCode::CapacityPressure, "could not allocate opaque KV checkpoint payload"};
    }
}

Status restore_kv(const ArchIr& arch,
                  std::span<const std::byte> payload,
                  std::uint32_t length,
                  std::span<std::byte> destination,
                  std::uint32_t destination_context) {
    if (length > destination_context) {
        return {StatusCode::InvalidArgument, "opaque KV length exceeds destination context"};
    }
    std::fill(destination.begin(), destination.end(), std::byte{0});
    const auto& c = arch.attention.compressed;
    const auto D = arch.attention.head_dim, ID = c.index_head_dim;
    const auto W = arch.attention.sliding_window;
    std::size_t cursor = 0;
    const auto take = [&](std::size_t at, std::size_t bytes) -> bool {
        if (cursor > payload.size() || bytes > payload.size() - cursor ||
            at > destination.size() || bytes > destination.size() - at) return false;
        std::copy_n(payload.data() + cursor, bytes, destination.data() + at);
        cursor += bytes;
        return true;
    };
    const auto take_live_carry = [&](std::size_t values_at,
                                     std::size_t scores_at,
                                     std::uint32_t ratio,
                                     std::uint32_t coff,
                                     std::uint32_t hd) -> bool {
        const auto rows = static_cast<std::size_t>(coff) * ratio;
        const auto row_floats = static_cast<std::size_t>(coff) * hd;
        std::fill_n(reinterpret_cast<float*>(destination.data() + values_at),
                    rows * row_floats,
                    0.0f);
        std::fill_n(reinterpret_cast<float*>(destination.data() + scores_at),
                    rows * row_floats,
                    -std::numeric_limits<float>::infinity());
        const auto complete = length / ratio;
        const auto partial = length % ratio;
        const auto row_bytes = row_floats * sizeof(float);
        if (coff == 1) {
            const auto bytes = static_cast<std::size_t>(partial) * row_bytes;
            return bytes == 0 || (take(values_at, bytes) && take(scores_at, bytes));
        }
        if (complete > 0) {
            const auto bytes = static_cast<std::size_t>(ratio) * row_bytes;
            if (!take(values_at, bytes) || !take(scores_at, bytes)) return false;
        }
        if (partial > 0) {
            const auto offset = static_cast<std::size_t>(ratio) * row_bytes;
            const auto bytes = static_cast<std::size_t>(partial) * row_bytes;
            if (!take(values_at + offset, bytes) ||
                !take(scores_at + offset, bytes)) return false;
        }
        return true;
    };
    for (std::uint32_t layer = 0; layer < arch.topology.n_layers; ++layer) {
        const auto reg = layer_region(arch, layer, destination_context);
        if (reg.end > destination.size()) {
            return {StatusCode::InvalidArgument, "opaque KV destination geometry is truncated"};
        }
        const auto ratio = c.compress_ratios[layer], coff = ratio == 4 ? 2u : 1u;
        const auto live_window = std::min(length, W);
        const auto first_window = length - live_window;
        for (std::uint32_t pos = first_window; pos < length; ++pos) {
            if (!take(reg.window + static_cast<std::size_t>(pos % W) * D * 2,
                      static_cast<std::size_t>(D) * 2)) {
                return {StatusCode::InvalidArgument, "opaque KV window payload is truncated"};
            }
        }
        const auto completed = length / ratio;
        if (!take(reg.compressed, static_cast<std::size_t>(completed) * D * 2)) {
            return {StatusCode::InvalidArgument,
                    "opaque compressed-history payload is truncated"};
        }
        if (ratio == 4 && !take(reg.index, static_cast<std::size_t>(completed) * ID * 2)) {
            return {StatusCode::InvalidArgument, "opaque index-history payload is truncated"};
        }
        if (!take_live_carry(reg.state_values,
                             reg.state_scores,
                             ratio,
                             coff,
                             D)) {
            return {StatusCode::InvalidArgument,
                    "opaque compressor-carry payload is truncated"};
        }
        if (ratio == 4 &&
            !take_live_carry(reg.index_state_values,
                             reg.index_state_scores,
                             4,
                             2,
                             ID)) {
            return {StatusCode::InvalidArgument,
                    "opaque index-carry payload is truncated"};
        }
    }
    if (cursor != payload.size()) {
        return {StatusCode::InvalidArgument, "opaque KV payload has trailing bytes"};
    }
    return {};
}

std::uint16_t to_bf16(float x) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

float from_bf16(std::uint16_t x) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(x) << 16);
}

void store_bf16(std::byte* dst, std::span<const float> src) noexcept {
    auto* p = reinterpret_cast<std::uint16_t*>(dst);
    for (std::size_t i = 0; i < src.size(); ++i) p[i] = to_bf16(src[i]);
}

void load_bf16(const std::byte* src, std::span<float> dst) noexcept {
    const auto* p = reinterpret_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < dst.size(); ++i) dst[i] = from_bf16(p[i]);
}

bool update_compressor(const ArchIr& arch,
                       const CompressorWeights& w,
                       std::uint32_t ratio,
                       std::uint32_t hd,
                       bool rotate,
                       const float* x,
                       std::uint32_t pos,
                       float* state_values,
                       float* state_scores,
                       std::byte* cache) {
    const auto d = arch.topology.d_model, coff = ratio == 4 ? 2u : 1u;
    std::vector<float> value(coff * hd), score(coff * hd);
    matvec(w.wkv, std::span<const float>(x, d), value);
    matvec(w.wgate, std::span<const float>(x, d), score);
    const auto row = (coff == 2 ? ratio : 0u) + pos % ratio;
    for (std::uint32_t k = 0; k < coff * hd; ++k) {
        state_values[static_cast<std::size_t>(row) * coff * hd + k] = value[k];
        state_scores[static_cast<std::size_t>(row) * coff * hd + k] =
            score[k] + w.ape[static_cast<std::size_t>(pos % ratio) * coff * hd + k];
    }
    if ((pos + 1) % ratio != 0) return false;

    std::vector<float> out(hd), logits(2 * ratio);
    for (std::uint32_t k = 0; k < hd; ++k) {
        std::uint32_t nc = 0;
        if (coff == 2) {
            for (std::uint32_t r = 0; r < ratio; ++r)
                logits[nc++] = state_scores[static_cast<std::size_t>(r) * 2 * hd + k];
        }
        const auto col = (coff == 2 ? hd : 0u) + k;
        for (std::uint32_t r = 0; r < ratio; ++r)
            logits[nc++] = state_scores[static_cast<std::size_t>((coff == 2 ? ratio : 0u) + r) *
                                             coff * hd +
                                         col];
        const float mx = *std::max_element(logits.begin(), logits.begin() + nc);
        float sum = 0.0f;
        for (std::uint32_t i = 0; i < nc; ++i) sum += std::exp(logits[i] - mx);
        double acc = 0.0;
        std::uint32_t i = 0;
        if (coff == 2) {
            for (std::uint32_t r = 0; r < ratio; ++r, ++i)
                acc += std::exp(logits[i] - mx) / sum *
                       state_values[static_cast<std::size_t>(r) * 2 * hd + k];
        }
        for (std::uint32_t r = 0; r < ratio; ++r, ++i)
            acc += std::exp(logits[i] - mx) / sum *
                   state_values[static_cast<std::size_t>((coff == 2 ? ratio : 0u) + r) * coff * hd +
                                col];
        out[k] = static_cast<float>(acc);
    }
    f32::rmsnorm(out, w.norm, hd, arch.rms_norm_eps);
    const auto rd = arch.attention.compressed.rope_head_dim;
    rope(std::span<float>(out).subspan(hd - rd, rd),
         pos + 1 - ratio,
         arch.attention.compressed.compress_rope_theta,
         arch.attention.rope.scaling.factor,
         arch.attention.rope.scaling.original_max_position,
         arch.attention.rope.scaling.beta_fast,
         arch.attention.rope.scaling.beta_slow);
    if (rotate) {
        if (arch.attention.compressed.semantic_fp4_quant_dequant) {
            hadamard(out);
            fake_fp4(out);
        }
    } else {
        if (arch.attention.compressed.semantic_fp8_quant_dequant)
            fake_fp8(std::span<float>(out).first(hd - rd));
    }
    store_bf16(cache + static_cast<std::size_t>(pos / ratio) * hd * 2, out);
    if (coff == 2) {
        const auto bytes = static_cast<std::size_t>(ratio) * 2 * hd * sizeof(float);
        std::copy_n(reinterpret_cast<const std::byte*>(state_values +
                                                       static_cast<std::size_t>(ratio) * 2 * hd),
                    bytes,
                    reinterpret_cast<std::byte*>(state_values));
        std::copy_n(reinterpret_cast<const std::byte*>(state_scores +
                                                       static_cast<std::size_t>(ratio) * 2 * hd),
                    bytes,
                    reinterpret_cast<std::byte*>(state_scores));
    }
    return true;
}

StatusCode attention_kv(const ArchIr& arch,
                        const F32LayerWeights& lw,
                        const float* x,
                        std::uint32_t n_rows,
                        LayerIndex layer,
                        const KvRow* rows,
                        F32Workspace&,
                        float* out) noexcept {
    const auto* p = lw.attn.as<LayerPayload>();
    if (p == nullptr) return StatusCode::Internal;
    const auto& c = arch.attention.compressed;
    const auto d = arch.topology.d_model, H = arch.attention.n_heads;
    const auto D = arch.attention.head_dim, rd = c.rope_head_dim, qr = c.q_lora_rank;
    const auto ratio = c.compress_ratios[layer];
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (std::uint32_t rix = 0; rix < n_rows; ++rix) {
        const auto& row = rows[rix];
        if (row.opaque_base == nullptr || row.max_ctx == 0 || row.pos >= row.max_ctx)
            return StatusCode::InvalidArgument;
        const auto reg = layer_region(arch, layer, row.max_ctx);
        if (reg.end > row.opaque_bytes) return StatusCode::InvalidArgument;
        std::byte* base = row.opaque_base;
        const float* xr = x + static_cast<std::size_t>(rix) * d;
        const auto journal = [&](std::size_t at, std::size_t bytes) {
            if (row.transaction != nullptr)
                row.transaction->capture(row.transaction_row, base + at, bytes);
        };

        // Score carry uses -inf as the identity for positions that have not
        // entered a compression window. A zero-filled byte arena would give
        // those absent positions finite softmax mass on the first ratio-4
        // boundary and disagree with the whole-sequence path immediately.
        if (row.pos == 0) {
            const auto coff = ratio == 4 ? 2u : 1u;
            const auto state_floats = static_cast<std::size_t>(coff) * ratio * coff * D;
            journal(reg.state_values, state_floats * sizeof(float));
            journal(reg.state_scores, state_floats * sizeof(float));
            std::fill_n(reinterpret_cast<float*>(base + reg.state_values), state_floats, 0.0f);
            std::fill_n(reinterpret_cast<float*>(base + reg.state_scores),
                        state_floats,
                        -std::numeric_limits<float>::infinity());
            if (ratio == 4) {
                const auto index_floats = static_cast<std::size_t>(2) * 4 * 2 * c.index_head_dim;
                journal(reg.index_state_values, index_floats * sizeof(float));
                journal(reg.index_state_scores, index_floats * sizeof(float));
                std::fill_n(reinterpret_cast<float*>(base + reg.index_state_values),
                            index_floats,
                            0.0f);
                std::fill_n(reinterpret_cast<float*>(base + reg.index_state_scores),
                            index_floats,
                            -std::numeric_limits<float>::infinity());
            }
        }

        std::vector<float> qlow(qr), q(static_cast<std::size_t>(H) * D), kv(D);
        matvec(p->wq_a, std::span<const float>(xr, d), qlow);
        f32::rmsnorm(qlow, p->q_norm, qr, arch.rms_norm_eps);
        matvec(p->wq_b, qlow, q);
        for (std::uint32_t h = 0; h < H; ++h) {
            float* qh = q.data() + static_cast<std::size_t>(h) * D;
            rmsnorm_unweighted(std::span<float>(qh, D), arch.rms_norm_eps);
            rope(std::span<float>(qh + D - rd, rd),
                 row.pos,
                 c.compress_rope_theta,
                 arch.attention.rope.scaling.factor,
                 arch.attention.rope.scaling.original_max_position,
                 arch.attention.rope.scaling.beta_fast,
                 arch.attention.rope.scaling.beta_slow);
        }
        matvec(p->wkv, std::span<const float>(xr, d), kv);
        f32::rmsnorm(kv, p->kv_norm, D, arch.rms_norm_eps);
        rope(std::span<float>(kv).subspan(D - rd, rd),
             row.pos,
             c.compress_rope_theta,
             arch.attention.rope.scaling.factor,
             arch.attention.rope.scaling.original_max_position,
             arch.attention.rope.scaling.beta_fast,
             arch.attention.rope.scaling.beta_slow);
        if (c.semantic_fp8_quant_dequant) fake_fp8(std::span<float>(kv).first(D - rd));
        const auto window_at = reg.window +
                               static_cast<std::size_t>(row.pos %
                                                        arch.attention.sliding_window) * D * 2;
        journal(window_at, static_cast<std::size_t>(D) * 2);
        store_bf16(base + window_at, kv);
        const auto coff = ratio == 4 ? 2u : 1u;
        const auto state_floats = static_cast<std::size_t>(coff) * ratio * coff * D;
        journal(reg.state_values, state_floats * sizeof(float));
        journal(reg.state_scores, state_floats * sizeof(float));
        if ((row.pos + 1) % ratio == 0) {
            journal(reg.compressed + static_cast<std::size_t>(row.pos / ratio) * D * 2,
                    static_cast<std::size_t>(D) * 2);
        }
        update_compressor(arch,
                          p->compressor,
                          ratio,
                          D,
                          false,
                          xr,
                          row.pos,
                          reinterpret_cast<float*>(base + reg.state_values),
                          reinterpret_cast<float*>(base + reg.state_scores),
                          base + reg.compressed);

        std::vector<std::uint32_t> chosen;
        if (ratio == 4) {
            const auto IH = c.index_n_heads, ID = c.index_head_dim;
            const auto index_state_floats = static_cast<std::size_t>(2) * 4 * 2 * ID;
            journal(reg.index_state_values, index_state_floats * sizeof(float));
            journal(reg.index_state_scores, index_state_floats * sizeof(float));
            if ((row.pos + 1) % ratio == 0) {
                journal(reg.index + static_cast<std::size_t>(row.pos / ratio) * ID * 2,
                        static_cast<std::size_t>(ID) * 2);
            }
            update_compressor(arch,
                              p->indexer.compressor,
                              ratio,
                              ID,
                              true,
                              xr,
                              row.pos,
                              reinterpret_cast<float*>(base + reg.index_state_values),
                              reinterpret_cast<float*>(base + reg.index_state_scores),
                              base + reg.index);
            const auto eligible = (row.pos + 1) / 4;
            if (eligible > 0) {
                std::vector<float> iq(static_cast<std::size_t>(IH) * ID), iw(IH), ikv(ID), scores(eligible);
                matvec(p->indexer.wq_b, qlow, iq);
                matvec(p->indexer.weights_proj, std::span<const float>(xr, d), iw);
                const float scale = 1.0f / std::sqrt(static_cast<float>(ID * IH));
                for (std::uint32_t h = 0; h < IH; ++h) {
                    float* qh = iq.data() + static_cast<std::size_t>(h) * ID;
                    rope(std::span<float>(qh + ID - rd, rd),
                         row.pos,
                         c.compress_rope_theta,
                         arch.attention.rope.scaling.factor,
                         arch.attention.rope.scaling.original_max_position,
                         arch.attention.rope.scaling.beta_fast,
                         arch.attention.rope.scaling.beta_slow);
                    if (c.semantic_fp4_quant_dequant) {
                        hadamard(std::span<float>(qh, ID));
                        fake_fp4(std::span<float>(qh, ID));
                    }
                }
                for (std::uint32_t k = 0; k < eligible; ++k) {
                    load_bf16(base + reg.index + static_cast<std::size_t>(k) * ID * 2, ikv);
                    double score = 0.0;
                    for (std::uint32_t h = 0; h < IH; ++h)
                        score += std::max(0.0f,
                                          f32::dot(std::span<const float>(iq.data() +
                                                                              static_cast<std::size_t>(h) * ID,
                                                                          ID),
                                                   ikv,
                                                   ID)) *
                                 iw[h] * scale;
                    scores[k] = static_cast<float>(score);
                }
                const auto take = std::min<std::uint32_t>(eligible, c.index_topk);
                chosen.resize(take);
                std::vector<float> vals(take);
                f32::top_k(scores, eligible, take, chosen, vals);
            }
        }

        std::vector<float> heads(static_cast<std::size_t>(H) * D), key(D), logits;
        const auto win_first = row.pos + 1 > arch.attention.sliding_window
                                   ? row.pos + 1 - arch.attention.sliding_window
                                   : 0;
        const auto nwin = row.pos - win_first + 1;
        const auto ncomp = ratio == 4 ? static_cast<std::uint32_t>(chosen.size())
                                     : (row.pos + 1) / ratio;
        logits.resize(nwin + ncomp);
        for (std::uint32_t h = 0; h < H; ++h) {
            const float* qh = q.data() + static_cast<std::size_t>(h) * D;
            float* oh = heads.data() + static_cast<std::size_t>(h) * D;
            std::fill_n(oh, D, 0.0f);
            float mx = p->attn_sink[h];
            std::uint32_t at = 0;
            for (std::uint32_t pos = win_first; pos <= row.pos; ++pos, ++at) {
                load_bf16(base + reg.window +
                              static_cast<std::size_t>(pos % arch.attention.sliding_window) * D * 2,
                          key);
                logits[at] = f32::dot(std::span<const float>(qh, D), key, D) * attn_scale;
                mx = std::max(mx, logits[at]);
            }
            for (std::uint32_t k = 0; k < ncomp; ++k, ++at) {
                const auto idx = ratio == 4 ? chosen[k] : k;
                load_bf16(base + reg.compressed + static_cast<std::size_t>(idx) * D * 2, key);
                logits[at] = f32::dot(std::span<const float>(qh, D), key, D) * attn_scale;
                mx = std::max(mx, logits[at]);
            }
            float denom = std::exp(p->attn_sink[h] - mx);
            for (const float v : logits) denom += std::exp(v - mx);
            at = 0;
            for (std::uint32_t pos = win_first; pos <= row.pos; ++pos, ++at) {
                load_bf16(base + reg.window +
                              static_cast<std::size_t>(pos % arch.attention.sliding_window) * D * 2,
                          key);
                f32::axpy(std::exp(logits[at] - mx) / denom, key, D, std::span<float>(oh, D));
            }
            for (std::uint32_t k = 0; k < ncomp; ++k, ++at) {
                const auto idx = ratio == 4 ? chosen[k] : k;
                load_bf16(base + reg.compressed + static_cast<std::size_t>(idx) * D * 2, key);
                f32::axpy(std::exp(logits[at] - mx) / denom, key, D, std::span<float>(oh, D));
            }
            rope(std::span<float>(oh + D - rd, rd),
                 row.pos,
                 c.compress_rope_theta,
                 arch.attention.rope.scaling.factor,
                 arch.attention.rope.scaling.original_max_position,
                 arch.attention.rope.scaling.beta_fast,
                 arch.attention.rope.scaling.beta_slow,
                 true);
        }
        const auto G = c.o_groups, R = c.o_lora_rank, hpg = H / G;
        std::vector<float> low(static_cast<std::size_t>(G) * R);
        for (std::uint32_t g = 0; g < G; ++g) {
            const auto wg = row_block(p->wo_a, g * R, R);
            matvec(wg,
                   std::span<const float>(heads).subspan(static_cast<std::size_t>(g) * hpg * D,
                                                        hpg * D),
                   std::span<float>(low).subspan(static_cast<std::size_t>(g) * R, R));
        }
        matvec(p->wo_b,
               low,
               std::span<float>(out + static_cast<std::size_t>(rix) * d, d));
    }
    return StatusCode::Ok;
}

std::uint64_t resident_weight_bytes(const ArchIr& arch,
                                    AttentionBackend::ByteSizer sizer) noexcept {
    const auto& c = arch.attention.compressed;
    const auto d = arch.topology.d_model, H = arch.attention.n_heads;
    const auto D = arch.attention.head_dim, qr = c.q_lora_rank;
    const auto G = c.o_groups, R = c.o_lora_rank;
    std::uint64_t total = 0;
    for (std::uint32_t l = 0; l < arch.topology.n_layers; ++l) {
        const auto ratio = c.compress_ratios[l], coff = ratio == 4 ? 2u : 1u;
        total += sizer(arch, qr, d, TensorRole::AttnProj);
        total += sizer(arch, H * D, qr, TensorRole::AttnProj);
        total += sizer(arch, D, d, TensorRole::AttnProj);
        total += sizer(arch, G * R, (H / G) * D, TensorRole::AttnProj);
        total += sizer(arch, d, G * R, TensorRole::AttnProj);
        total += static_cast<std::uint64_t>(H + qr + D) * sizeof(float);
        total += 2 * sizer(arch, coff * D, d, TensorRole::AttnProj);
        total += static_cast<std::uint64_t>(ratio) * coff * D * sizeof(float);
        total += static_cast<std::uint64_t>(D) * sizeof(float);
        if (ratio == 4) {
            total += sizer(arch, c.index_n_heads * c.index_head_dim, qr, TensorRole::AttnProj);
            total += sizer(arch, c.index_n_heads, d, TensorRole::AttnProj);
            total += 2 * sizer(arch, 2 * c.index_head_dim, d, TensorRole::AttnProj);
            total += static_cast<std::uint64_t>(4) * 2 * c.index_head_dim * sizeof(float);
            total += static_cast<std::uint64_t>(c.index_head_dim) * sizeof(float);
        }
        const auto hc = arch.hyper_connections.multiplier;
        const auto mix = (2 + hc) * hc;
        total += 2ull * (static_cast<std::uint64_t>(mix) * hc * d + mix + 3) * sizeof(float);
        total += l < arch.router.n_hash_layers
                     ? static_cast<std::uint64_t>(arch.topology.vocab_size) * arch.router.top_k *
                           sizeof(std::uint32_t)
                     : static_cast<std::uint64_t>(arch.router.n_experts) * sizeof(float);
    }
    // Model-level stream collapse is owned once, outside the layer stack.
    const auto hc = arch.hyper_connections.multiplier;
    total += (static_cast<std::uint64_t>(hc) * hc * d + hc + 1) * sizeof(float);
    return total;
}

std::uint64_t weight_bytes_per_layer(const ArchIr& arch,
                                     AttentionBackend::ByteSizer sizer) noexcept {
    const auto total = resident_weight_bytes(arch, sizer);
    return (total + arch.topology.n_layers - 1) / arch.topology.n_layers;
}

} // namespace

const soma::F32Backend& f32_backend() noexcept {
    static const soma::F32Backend backend = [] {
        soma::F32Backend b{};
        b.name = "compressed-sparse-reference";
        b.prompt_codec = &prompt_codec();
        b.bind_layer = &bind_layer;
        b.bind_model = &bind_model;
        b.begin_forward = &begin_forward;
        b.pre_attention = &pre_attention;
        b.merge_attention = &merge_attention;
        b.pre_ffn = &pre_ffn;
        b.merge_ffn = &merge_ffn;
        b.export_layer_hidden = &export_layer_hidden;
        b.end_forward = &end_forward;
        b.attention = &attention;
        b.attention_kv = &attention_kv;
        b.route = &route;
        return b;
    }();
    return backend;
}

const soma::AttentionBackend& attention_backend() noexcept {
    static const soma::AttentionBackend backend = [] {
        soma::AttentionBackend b{};
        b.name = "compressed-sparse";
        b.family = AttentionFamily::CompressedSparse;
        b.persist_format_id = kv_format_id("compressed-sparse-bf16-v1");
        b.kv_bytes_per_token = &kv_bytes_per_token;
        b.kv_bytes_for_context = &kv_bytes_for_context;
        b.serialize_kv = &serialize_kv;
        b.restore_kv = &restore_kv;
        b.weight_bytes_per_layer = &weight_bytes_per_layer;
        b.resident_weight_bytes = &resident_weight_bytes;
        return b;
    }();
    return backend;
}

} // namespace soma::arch::compressed_sparse
