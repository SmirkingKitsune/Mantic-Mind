#include "soma/arch/deepseek_dspark.hpp"

#include "soma/expert_store.hpp"
#include "soma/kernels_f32.hpp"
#include "soma/memory_hierarchy.hpp"
#include "soma/quant_format.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

namespace soma::arch::deepseek_dspark {
namespace {

template <class T>
void destroy(void* p) { delete static_cast<T*>(p); }

struct Layer {
    std::span<const float> input_norm, post_norm;
    std::span<const float> sink, q_norm, kv_norm;
    WeightRef wq_a, wq_b, wkv, wo_a, wo_b;
    std::span<const float> hc_attn_fn, hc_attn_base, hc_attn_scale;
    std::span<const float> hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    std::span<const float> router, route_bias;
    WeightRef shared_gate, shared_up, shared_down;
};

struct Payload {
    WeightRef main_proj, markov_w1, markov_w2, confidence;
    std::span<const float> main_norm, norm;
    std::span<const float> hc_head_fn, hc_head_base, hc_head_scale;
    std::vector<Layer> layers;
    ExpertStore store;
    MemoryHierarchy memory;
    ArchIr draft_arch;
    std::uint32_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
    bool runtime_open = false;
};

struct State {
    std::uint32_t length = 0;
    std::uint32_t width = 0;
    std::vector<std::uint16_t> context_kv; // [stage,128,head_dim], BF16
};

struct HcState {
    std::uint32_t rows = 0, d = 0, hc = 0;
    std::vector<float> streams, post, comb;
};

float sigmoid(float x) noexcept {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

std::uint16_t to_bf16(float x) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(x);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

float from_bf16(std::uint16_t x) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(x) << 16);
}

float round_even(float x) noexcept {
    const float lo = std::floor(x), f = x - lo;
    if (f < 0.5f) return lo;
    if (f > 0.5f) return lo + 1.0f;
    return std::fmod(lo, 2.0f) == 0.0f ? lo : lo + 1.0f;
}

float fp8_round(float x) noexcept {
    if (!std::isfinite(x)) return std::copysign(448.0f, x);
    const float sign = std::signbit(x) ? -1.0f : 1.0f;
    float a = std::min(std::abs(x), 448.0f);
    if (a == 0.0f) return x;
    if (a < 1.0f / 64.0f) return sign * round_even(a * 512.0f) / 512.0f;
    const int e = static_cast<int>(std::floor(std::log2(a)));
    const float step = std::ldexp(1.0f, e - 3);
    return sign * std::min(round_even(a / step) * step, 448.0f);
}

void bf16_round(std::span<float> x) noexcept {
    for (float& v : x) v = from_bf16(to_bf16(v));
}

void fake_fp8(std::span<float> x, std::uint32_t group = 64) noexcept {
    for (std::size_t at = 0; at < x.size(); at += group) {
        const auto n = std::min<std::size_t>(group, x.size() - at);
        float amax = 1e-4f;
        for (std::size_t i = 0; i < n; ++i) amax = std::max(amax, std::abs(x[at + i]));
        const float scale = std::exp2(std::ceil(std::log2(amax / 448.0f)));
        for (std::size_t i = 0; i < n; ++i)
            x[at + i] = from_bf16(to_bf16(fp8_round(x[at + i] / scale) * scale));
    }
}

void semantic_matvec(const F32Model& model, const WeightRef& weight,
                     std::span<const float> input, std::span<float> output) {
    std::vector<float> qinput(input.begin(), input.end());
    if (model.arch.attention.compressed.semantic_fp8_quant_dequant)
        fake_fp8(qinput, 128);
    matvec(weight, qinput, output);
    bf16_round(output);
}

void semantic_matmul(const F32Model& model, const WeightRef& weight,
                     std::span<const float> input, std::uint32_t rows,
                     std::span<float> output) {
    std::vector<float> qinput(input.begin(), input.end());
    if (model.arch.attention.compressed.semantic_fp8_quant_dequant) {
        const auto width = weight.cols;
        for (std::uint32_t row = 0; row < rows; ++row)
            fake_fp8(std::span<float>(qinput).subspan(
                         static_cast<std::size_t>(row) * width, width), 128);
    }
    matmul(weight, qinput, rows, output);
    bf16_round(output);
}

void rms_unweighted(std::span<float> x, float eps) noexcept {
    double ss = 0.0;
    for (const float v : x) ss += static_cast<double>(v) * v;
    const float s = 1.0f / std::sqrt(static_cast<float>(ss / x.size()) + eps);
    for (float& v : x) v *= s;
}

void rope(std::span<float> x, std::uint32_t pos, const ArchIr& a, bool inverse = false) noexcept {
    const auto dim = static_cast<std::uint32_t>(x.size());
    // Every official DSpark layer has compress_ratio == 0. model.py therefore
    // selects base rope_theta and explicitly disables YaRN for both its target
    // KV projection and five draft queries.
    const float base = a.attention.rope.theta;
    for (std::uint32_t i = 0; i < dim / 2; ++i) {
        float freq = std::pow(base, -2.0f * i / dim);
        float angle = static_cast<float>(pos) * freq;
        if (inverse) angle = -angle;
        const float c = std::cos(angle), s = std::sin(angle);
        const float u = x[2 * i], v = x[2 * i + 1];
        x[2 * i] = u * c - v * s;
        x[2 * i + 1] = u * s + v * c;
    }
}

bool bind_f32(const F32Model& model, const std::string& name, std::span<const float>& out) {
    const auto* tv = model.weights.find(name);
    if (tv == nullptr || tv->dtype != DType::F32) return false;
    out = tv->f32();
    return true;
}

bool bind_weight(F32Model& model, const std::string& name, WeightRef& out) {
    ModelBindCtx ctx{&model.weights, &model.quant_map, &model.quantized};
    return bind_model_weight(ctx, name.c_str(), TensorRole::DraftHead, out).ok();
}

StatusCode bind_model(F32Model& model, const std::string&) noexcept {
    auto* p = new Payload();
    p->layers.resize(model.arch.speculative.n_layers);
    const auto fail = [&]() { delete p; return StatusCode::NotFound; };
    if (!bind_weight(model, "model.dspark.main_proj.weight", p->main_proj) ||
        !bind_f32(model, "model.dspark.main_norm.weight", p->main_norm) ||
        !bind_f32(model, "model.dspark.norm.weight", p->norm) ||
        !bind_weight(model, "model.dspark.markov_w1.weight", p->markov_w1) ||
        !bind_weight(model, "model.dspark.markov_w2.weight", p->markov_w2) ||
        !bind_weight(model, "model.dspark.confidence_proj.weight", p->confidence) ||
        !bind_f32(model, "model.dspark.hc_head_fn", p->hc_head_fn) ||
        !bind_f32(model, "model.dspark.hc_head_base", p->hc_head_base) ||
        !bind_f32(model, "model.dspark.hc_head_scale", p->hc_head_scale)) return fail();

    for (std::uint32_t stage = 0; stage < p->layers.size(); ++stage) {
        auto& l = p->layers[stage];
        const auto q = "model.dspark.layers." + std::to_string(stage) + ".";
        if (!bind_f32(model, q + "input_layernorm.weight", l.input_norm) ||
            !bind_f32(model, q + "post_attention_layernorm.weight", l.post_norm) ||
            !bind_f32(model, q + "self_attn.attn_sink", l.sink) ||
            !bind_weight(model, q + "self_attn.wq_a.weight", l.wq_a) ||
            !bind_weight(model, q + "self_attn.wq_b.weight", l.wq_b) ||
            !bind_f32(model, q + "self_attn.q_norm.weight", l.q_norm) ||
            !bind_weight(model, q + "self_attn.wkv.weight", l.wkv) ||
            !bind_f32(model, q + "self_attn.kv_norm.weight", l.kv_norm) ||
            !bind_weight(model, q + "self_attn.wo_a.weight", l.wo_a) ||
            !bind_weight(model, q + "self_attn.wo_b.weight", l.wo_b) ||
            !bind_f32(model, q + "hc_attn_fn", l.hc_attn_fn) ||
            !bind_f32(model, q + "hc_attn_base", l.hc_attn_base) ||
            !bind_f32(model, q + "hc_attn_scale", l.hc_attn_scale) ||
            !bind_f32(model, q + "hc_ffn_fn", l.hc_ffn_fn) ||
            !bind_f32(model, q + "hc_ffn_base", l.hc_ffn_base) ||
            !bind_f32(model, q + "hc_ffn_scale", l.hc_ffn_scale) ||
            !bind_f32(model, q + "ffn.gate.weight", l.router) ||
            !bind_f32(model, q + "ffn.gate.bias", l.route_bias) ||
            !bind_weight(model, q + "ffn.shared_experts.gate_proj.weight", l.shared_gate) ||
            !bind_weight(model, q + "ffn.shared_experts.up_proj.weight", l.shared_up) ||
            !bind_weight(model, q + "ffn.shared_experts.down_proj.weight", l.shared_down))
            return fail();
    }
    p->draft_arch = model.arch;
    p->draft_arch.topology.n_layers = model.arch.speculative.n_layers;
    p->draft_arch.topology.layer_kinds.assign(p->draft_arch.topology.n_layers, LayerKind::Moe);
    p->draft_arch.router.n_hash_layers = 0;
    p->draft_arch.speculative.present = false;
    p->draft_arch.arch_hash.clear(); // auxiliary container is gated by size/shape
    const auto fi = model.arch.ffn.expert_intermediate, d = model.d_model();
    const auto sz = [&](std::uint32_t rows, std::uint32_t cols, TensorRole role) {
        const auto& s = model.quant_map.for_role(role);
        return static_cast<std::uint32_t>(quantized_tensor_bytes(
            s.dtype, rows, cols, s.group ? s.group : kDefaultGroup));
    };
    p->gate_bytes = sz(fi, d, TensorRole::ExpertGate);
    p->up_bytes = sz(fi, d, TensorRole::ExpertUp);
    p->down_bytes = sz(d, fi, TensorRole::ExpertDown);
    model.speculative_payload.adopt(p, &destroy<Payload>);
    return StatusCode::Ok;
}

StatusCode start_runtime(F32Model& model,
                         const std::string& dir,
                         std::uint64_t cache_bytes) noexcept {
    auto* p = model.speculative_payload.as<Payload>();
    if (p == nullptr) return StatusCode::Internal;
    if (auto st = p->store.open_indexed(dir, p->draft_arch, "soma.dspark", "dspark-experts-");
        !st.ok()) return st.code();
    MemoryBudget b;
    b.ram_expert_cache_bytes = cache_bytes;
    if (auto st = p->memory.open(p->draft_arch, p->store, b); !st.ok()) return st.code();
    p->runtime_open = true;
    return StatusCode::Ok;
}

StatusCode open_sequence(const F32Model& model,
                         std::uint32_t,
                         ArchLayerPayload& out) noexcept {
    auto* s = new State();
    s->width = model.arch.attention.head_dim;
    try {
        s->context_kv.assign(static_cast<std::size_t>(model.arch.speculative.n_layers) *
                                 model.arch.attention.sliding_window * s->width,
                             0);
    } catch (...) {
        delete s;
        return StatusCode::CapacityPressure;
    }
    out.adopt(s, &destroy<State>);
    return StatusCode::Ok;
}

void hc_begin(const ArchIr& a, const float* hidden, std::uint32_t rows, HcState& s) {
    s.rows = rows; s.d = a.topology.d_model; s.hc = a.hyper_connections.multiplier;
    s.streams.resize(static_cast<std::size_t>(rows) * s.hc * s.d);
    s.post.resize(static_cast<std::size_t>(rows) * s.hc);
    s.comb.resize(static_cast<std::size_t>(rows) * s.hc * s.hc);
    for (std::uint32_t t = 0; t < rows; ++t)
        for (std::uint32_t h = 0; h < s.hc; ++h)
            std::copy_n(hidden + static_cast<std::size_t>(t) * s.d, s.d,
                        s.streams.data() + (static_cast<std::size_t>(t) * s.hc + h) * s.d);
}

bool hc_pre(const ArchIr& a, std::span<const float> fn, std::span<const float> base,
            std::span<const float> scale, HcState& s, float* hidden) {
    const auto d = s.d, hc = s.hc, flat = hc * d, mix = (2 + hc) * hc;
    if (scale.size() < 3 || fn.size() != static_cast<std::size_t>(mix) * flat ||
        base.size() != mix) return false;
    std::vector<float> z(mix), matrix(hc * hc);
    for (std::uint32_t t = 0; t < s.rows; ++t) {
        const float* streams = s.streams.data() + static_cast<std::size_t>(t) * flat;
        double ss = 0.0;
        for (std::uint32_t i = 0; i < flat; ++i) ss += static_cast<double>(streams[i]) * streams[i];
        const float rs = 1.0f / std::sqrt(static_cast<float>(ss / flat) + a.rms_norm_eps);
        for (std::uint32_t m = 0; m < mix; ++m) {
            double v = 0.0;
            const float* row = fn.data() + static_cast<std::size_t>(m) * flat;
            for (std::uint32_t i = 0; i < flat; ++i) v += static_cast<double>(row[i]) * streams[i];
            z[m] = static_cast<float>(v) * rs;
        }
        float* post = s.post.data() + static_cast<std::size_t>(t) * hc;
        float* comb = s.comb.data() + static_cast<std::size_t>(t) * hc * hc;
        std::fill_n(hidden + static_cast<std::size_t>(t) * d, d, 0.0f);
        for (std::uint32_t h = 0; h < hc; ++h) {
            const float pre = sigmoid(z[h] * scale[0] + base[h]) + a.hyper_connections.eps;
            post[h] = 2.0f * sigmoid(z[hc + h] * scale[1] + base[hc + h]);
            for (std::uint32_t k = 0; k < d; ++k)
                hidden[static_cast<std::size_t>(t) * d + k] +=
                    pre * streams[static_cast<std::size_t>(h) * d + k];
        }
        bf16_round(std::span<float>(hidden + static_cast<std::size_t>(t) * d, d));
        for (std::uint32_t r = 0; r < hc; ++r) {
            float mx = -std::numeric_limits<float>::infinity();
            for (std::uint32_t c = 0; c < hc; ++c) {
                const auto k = r * hc + c;
                matrix[k] = z[2 * hc + k] * scale[2] + base[2 * hc + k];
                mx = std::max(mx, matrix[k]);
            }
            float sum = 0.0f;
            for (std::uint32_t c = 0; c < hc; ++c) sum += std::exp(matrix[r * hc + c] - mx);
            for (std::uint32_t c = 0; c < hc; ++c)
                matrix[r * hc + c] = std::exp(matrix[r * hc + c] - mx) / sum +
                                      a.hyper_connections.eps;
        }
        const auto cols = [&]() {
            for (std::uint32_t c = 0; c < hc; ++c) {
                float sum = a.hyper_connections.eps;
                for (std::uint32_t r = 0; r < hc; ++r) sum += matrix[r * hc + c];
                for (std::uint32_t r = 0; r < hc; ++r) matrix[r * hc + c] /= sum;
            }
        };
        const auto rows = [&]() {
            for (std::uint32_t r = 0; r < hc; ++r) {
                float sum = a.hyper_connections.eps;
                for (std::uint32_t c = 0; c < hc; ++c) sum += matrix[r * hc + c];
                for (std::uint32_t c = 0; c < hc; ++c) matrix[r * hc + c] /= sum;
            }
        };
        cols();
        for (std::uint32_t i = 1; i < a.hyper_connections.sinkhorn_iters; ++i) { rows(); cols(); }
        std::copy(matrix.begin(), matrix.end(), comb);
    }
    return true;
}

void hc_merge(const float* branch, HcState& s, float* hidden) {
    std::vector<float> next(s.streams.size());
    for (std::uint32_t t = 0; t < s.rows; ++t) {
        const float* old = s.streams.data() + static_cast<std::size_t>(t) * s.hc * s.d;
        const float* post = s.post.data() + static_cast<std::size_t>(t) * s.hc;
        const float* comb = s.comb.data() + static_cast<std::size_t>(t) * s.hc * s.hc;
        for (std::uint32_t o = 0; o < s.hc; ++o) {
            float* dst = next.data() + (static_cast<std::size_t>(t) * s.hc + o) * s.d;
            for (std::uint32_t k = 0; k < s.d; ++k) {
                double v = static_cast<double>(post[o]) * branch[static_cast<std::size_t>(t) * s.d + k];
                for (std::uint32_t i = 0; i < s.hc; ++i)
                    // model.py broadcasts residual on the first HC axis and
                    // reduces that axis: output[o] = sum_i comb[i,o] * old[i].
                    v += static_cast<double>(comb[i * s.hc + o]) *
                         old[static_cast<std::size_t>(i) * s.d + k];
                dst[k] = from_bf16(to_bf16(static_cast<float>(v)));
            }
        }
        std::copy_n(next.data() + static_cast<std::size_t>(t) * s.hc * s.d, s.d,
                    hidden + static_cast<std::size_t>(t) * s.d);
    }
    s.streams.swap(next);
}

void project_kv(const F32Model& model, const Layer& l, const float* x,
                 std::uint32_t pos, std::span<float> kv) {
    semantic_matvec(model, l.wkv, std::span<const float>(x, model.d_model()), kv);
    f32::rmsnorm(kv, l.kv_norm, static_cast<std::uint32_t>(kv.size()), model.arch.rms_norm_eps);
    bf16_round(kv);
    const auto rd = model.arch.attention.compressed.rope_head_dim;
    rope(kv.subspan(kv.size() - rd, rd), pos, model.arch);
    bf16_round(kv.subspan(kv.size() - rd, rd));
    if (model.arch.attention.compressed.semantic_fp8_quant_dequant)
        fake_fp8(kv.first(kv.size() - rd));
}

StatusCode observe_target(const F32Model& model, const ArchLayerPayload& payload,
                          ArchLayerPayload& state, const HiddenStateTaps& taps,
                          std::uint32_t first, std::uint32_t count,
                          std::uint32_t first_pos) noexcept {
    const auto* p = payload.as<Payload>();
    auto* s = state.as<State>();
    if (p == nullptr || s == nullptr || taps.layers.size() != p->layers.size() ||
        first + count > taps.n_rows) return StatusCode::InvalidArgument;
    const auto d = model.d_model(), naux = static_cast<std::uint32_t>(taps.layers.size());
    std::vector<float> cat(static_cast<std::size_t>(naux) * d), main(d), kv(model.arch.attention.head_dim);
    for (std::uint32_t r = 0; r < count; ++r) {
        for (std::uint32_t a = 0; a < naux; ++a) {
            const auto src = taps.layer(a);
            std::copy_n(src.data() + static_cast<std::size_t>(first + r) * d, d,
                        cat.data() + static_cast<std::size_t>(a) * d);
        }
        semantic_matvec(model, p->main_proj, cat, main);
        f32::rmsnorm(main, p->main_norm, d, model.arch.rms_norm_eps);
        bf16_round(main);
        const auto pos = first_pos + r;
        for (std::uint32_t stage = 0; stage < p->layers.size(); ++stage) {
            project_kv(model, p->layers[stage], main.data(), pos, kv);
            auto* dst = s->context_kv.data() +
                        (static_cast<std::size_t>(stage) * model.arch.attention.sliding_window +
                         pos % model.arch.attention.sliding_window) * model.arch.attention.head_dim;
            for (std::uint32_t k = 0; k < kv.size(); ++k) dst[k] = to_bf16(kv[k]);
        }
        s->length = std::max(s->length, pos + 1);
    }
    return StatusCode::Ok;
}

void draft_attention(const F32Model& model, const Layer& l, const State& s,
                     std::uint32_t stage, std::uint32_t start,
                     const float* x, std::uint32_t rows, float* out,
                     F32Workspace::Sink sink) {
    const auto& a = model.arch;
    const auto d = model.d_model(), H = a.attention.n_heads, D = a.attention.head_dim;
    const auto rd = a.attention.compressed.rope_head_dim, qr = a.attention.compressed.q_lora_rank;
    std::vector<float> qlow(static_cast<std::size_t>(rows) * qr), q(static_cast<std::size_t>(rows) * H * D);
    std::vector<float> qkv(static_cast<std::size_t>(rows) * D);
    semantic_matmul(model, l.wq_a,
                    std::span<const float>(x, static_cast<std::size_t>(rows) * d),
                    rows, qlow);
    sink(stage, "q_a", qlow.data(), qlow.size());
    for (std::uint32_t t = 0; t < rows; ++t)
        f32::rmsnorm(std::span<float>(qlow).subspan(static_cast<std::size_t>(t) * qr, qr),
                     l.q_norm, qr, a.rms_norm_eps);
    bf16_round(qlow);
    sink(stage, "q_norm", qlow.data(), qlow.size());
    semantic_matmul(model, l.wq_b, qlow, rows, q);
    sink(stage, "q_b", q.data(), q.size());
    for (std::uint32_t t = 0; t < rows; ++t) {
        for (std::uint32_t h = 0; h < H; ++h) {
            auto qh = std::span<float>(q).subspan((static_cast<std::size_t>(t) * H + h) * D, D);
            rms_unweighted(qh, a.rms_norm_eps);
            bf16_round(qh);
            rope(qh.subspan(D - rd, rd), start + t, a);
            bf16_round(qh.subspan(D - rd, rd));
        }
        project_kv(model, l, x + static_cast<std::size_t>(t) * d, start + t,
                   std::span<float>(qkv).subspan(static_cast<std::size_t>(t) * D, D));
    }
    sink(stage, "sparse_q", q.data(), q.size());
    sink(stage, "draft_kv", qkv.data(), qkv.size());
    std::vector<float> heads(static_cast<std::size_t>(rows) * H * D);
    std::vector<float> sparse_heads(heads.size()), key(D), logits;
    const auto W = a.attention.sliding_window;
    const auto ctx_first = start > W ? start - W : 0;
    const auto nctx = start - ctx_first;
    logits.resize(nctx + rows);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (std::uint32_t t = 0; t < rows; ++t) {
        for (std::uint32_t h = 0; h < H; ++h) {
            const float* qh = q.data() + (static_cast<std::size_t>(t) * H + h) * D;
            float* oh = heads.data() + (static_cast<std::size_t>(t) * H + h) * D;
            std::fill_n(oh, D, 0.0f);
            float mx = l.sink[h];
            std::uint32_t at = 0;
            for (std::uint32_t pos = ctx_first; pos < start; ++pos, ++at) {
                const auto* src = s.context_kv.data() +
                    (static_cast<std::size_t>(stage) * W + pos % W) * D;
                for (std::uint32_t k = 0; k < D; ++k) key[k] = from_bf16(src[k]);
                logits[at] = f32::dot(std::span<const float>(qh, D), key, D) * scale;
                mx = std::max(mx, logits[at]);
            }
            for (std::uint32_t j = 0; j < rows; ++j, ++at) {
                const float* k = qkv.data() + static_cast<std::size_t>(j) * D;
                logits[at] = f32::dot(std::span<const float>(qh, D), std::span<const float>(k, D), D) * scale;
                mx = std::max(mx, logits[at]);
            }
            float denom = std::exp(l.sink[h] - mx);
            for (const float v : logits) denom += std::exp(v - mx);
            at = 0;
            for (std::uint32_t pos = ctx_first; pos < start; ++pos, ++at) {
                const auto* src = s.context_kv.data() +
                    (static_cast<std::size_t>(stage) * W + pos % W) * D;
                for (std::uint32_t k = 0; k < D; ++k) key[k] = from_bf16(src[k]);
                f32::axpy(std::exp(logits[at] - mx) / denom, key, D, std::span<float>(oh, D));
            }
            for (std::uint32_t j = 0; j < rows; ++j, ++at)
                f32::axpy(std::exp(logits[at] - mx) / denom,
                          std::span<const float>(qkv).subspan(static_cast<std::size_t>(j) * D, D),
                          D, std::span<float>(oh, D));
            bf16_round(std::span<float>(oh, D));
            std::copy_n(oh, D,
                        sparse_heads.data() + (static_cast<std::size_t>(t) * H + h) * D);
            rope(std::span<float>(oh + D - rd, rd), start + t, a, true);
            bf16_round(std::span<float>(oh, D));
        }
    }
    sink(stage, "sparse_out", sparse_heads.data(), sparse_heads.size());
    const auto G = a.attention.compressed.o_groups, R = a.attention.compressed.o_lora_rank;
    const auto hpg = H / G;
    std::vector<float> low(static_cast<std::size_t>(rows) * G * R);
    for (std::uint32_t t = 0; t < rows; ++t)
        for (std::uint32_t g = 0; g < G; ++g)
            matvec(row_block(l.wo_a, g * R, R),
                   std::span<const float>(heads).subspan(
                       (static_cast<std::size_t>(t) * H + g * hpg) * D, hpg * D),
                   std::span<float>(low).subspan((static_cast<std::size_t>(t) * G + g) * R, R));
    bf16_round(low);
    semantic_matmul(model, l.wo_b, low, rows,
                    std::span<float>(out, static_cast<std::size_t>(rows) * d));
}

bool expert(const F32Model& model, Payload& p, std::uint32_t stage, ExpertId id,
            const float* x, float weight, float* out) {
    auto pin = p.memory.acquire(stage, id);
    if (!pin) return false;
    const auto bytes = pin.bytes();
    if (bytes.size() < static_cast<std::size_t>(p.gate_bytes) + p.up_bytes + p.down_bytes) return false;
    const auto& q = model.quant_map;
    const auto fi = model.arch.ffn.expert_intermediate, d = model.d_model();
    const auto make = [&](std::size_t at, std::size_t n, TensorRole role,
                          std::uint32_t r, std::uint32_t c) {
        const auto& spec = q.for_role(role);
        return WeightRef::from_quantized_bytes(bytes.subspan(at, n), spec.dtype,
                                               spec.group ? spec.group : kDefaultGroup, r, c);
    };
    const auto gate = make(0, p.gate_bytes, TensorRole::ExpertGate, fi, d);
    const auto up = make(p.gate_bytes, p.up_bytes, TensorRole::ExpertUp, fi, d);
    const auto down = make(p.gate_bytes + p.up_bytes, p.down_bytes,
                           TensorRole::ExpertDown, d, fi);
    std::vector<float> g(fi), u(fi), a(fi), y(d);
    semantic_matvec(model, gate, std::span<const float>(x, d), g);
    semantic_matvec(model, up, std::span<const float>(x, d), u);
    const float limit = model.arch.ffn.swiglu_limit;
    if (limit > 0.0f) for (std::uint32_t i = 0; i < fi; ++i) {
        g[i] = std::min(g[i], limit); u[i] = std::clamp(u[i], -limit, limit);
    }
    f32::swiglu(g, u, fi, a);
    bf16_round(a);
    semantic_matvec(model, down, a, y);
    for (std::uint32_t i = 0; i < d; ++i) out[i] += weight * y[i];
    return true;
}

bool draft_ffn(const F32Model& model, Payload& p, const Layer& l,
               std::uint32_t stage, const float* x, std::uint32_t rows,
               float* out, F32Workspace::Sink sink) {
    const auto d = model.d_model(), E = model.arch.router.n_experts;
    const auto K = model.arch.router.top_k, fi = model.arch.ffn.expert_intermediate;
    std::vector<float> logits(E), original(E), selected(E), vals(K);
    std::vector<std::uint32_t> ids(K);
    std::vector<float> route_ids(static_cast<std::size_t>(rows) * K);
    std::vector<float> route_weights(static_cast<std::size_t>(rows) * K);
    std::vector<float> g(model.arch.ffn.shared_intermediate),
                       u(model.arch.ffn.shared_intermediate),
                       a(model.arch.ffn.shared_intermediate), y(d);
    for (std::uint32_t t = 0; t < rows; ++t) {
        const float* xr = x + static_cast<std::size_t>(t) * d;
        float* yr = out + static_cast<std::size_t>(t) * d;
        std::fill_n(yr, d, 0.0f);
        semantic_matvec(model, l.shared_gate, std::span<const float>(xr, d), g);
        semantic_matvec(model, l.shared_up, std::span<const float>(xr, d), u);
        f32::swiglu(g, u, static_cast<std::uint32_t>(g.size()), a);
        bf16_round(a);
        semantic_matvec(model, l.shared_down, a, y);
        std::copy(y.begin(), y.end(), yr);
        f32::matvec(l.router, std::span<const float>(xr, d), E, d, logits);
        for (std::uint32_t e = 0; e < E; ++e) {
            original[e] = std::sqrt(std::log1p(std::exp(-std::abs(logits[e]))) +
                                    std::max(logits[e], 0.0f));
            selected[e] = original[e] + l.route_bias[e];
        }
        f32::top_k(selected, E, K, ids, vals);
        float sum = 0.0f;
        for (const auto id : ids) sum += original[id];
        for (std::uint32_t slot = 0; slot < ids.size(); ++slot) {
            const auto id = ids[slot];
            const float w = original[id] / std::max(sum, std::numeric_limits<float>::min()) *
                            model.arch.router.routed_scaling_factor;
            route_ids[static_cast<std::size_t>(t) * K + slot] = static_cast<float>(id);
            route_weights[static_cast<std::size_t>(t) * K + slot] = w;
            if (!expert(model, p, stage, id, xr, w, yr)) return false;
        }
        bf16_round(std::span<float>(yr, d));
    }
    sink(stage, "router_ids", route_ids.data(), route_ids.size());
    sink(stage, "router_weights", route_weights.data(), route_weights.size());
    (void)fi;
    return true;
}

void hc_head(const F32Model& model, const Payload& p, const HcState& s, float* hidden) {
    const auto d = model.d_model(), hc = model.arch.hyper_connections.multiplier, flat = hc * d;
    for (std::uint32_t t = 0; t < s.rows; ++t) {
        const float* streams = s.streams.data() + static_cast<std::size_t>(t) * flat;
        double ss = 0.0;
        for (std::uint32_t i = 0; i < flat; ++i) ss += static_cast<double>(streams[i]) * streams[i];
        const float rs = 1.0f / std::sqrt(static_cast<float>(ss / flat) + model.arch.rms_norm_eps);
        float* dst = hidden + static_cast<std::size_t>(t) * d;
        std::fill_n(dst, d, 0.0f);
        for (std::uint32_t h = 0; h < hc; ++h) {
            double v = 0.0;
            const float* row = p.hc_head_fn.data() + static_cast<std::size_t>(h) * flat;
            for (std::uint32_t i = 0; i < flat; ++i) v += static_cast<double>(row[i]) * streams[i];
            const float w = sigmoid(static_cast<float>(v) * rs * p.hc_head_scale[0] +
                                    p.hc_head_base[h]) + model.arch.hyper_connections.eps;
            for (std::uint32_t k = 0; k < d; ++k) dst[k] += w * streams[static_cast<std::size_t>(h) * d + k];
        }
        bf16_round(std::span<float>(dst, d));
    }
}

StatusCode propose(const F32Model& model, const ArchLayerPayload& payload,
                   ArchLayerPayload& state, TokenId anchor, std::uint32_t max_tokens,
                   float threshold, SpeculativeProposal& out) noexcept {
    auto* p = payload.as<Payload>(); auto* s = state.as<State>();
    if (p == nullptr || s == nullptr || !p->runtime_open || max_tokens == 0) return StatusCode::InvalidArgument;
    // The checkpoint was trained with one fixed query block. The scheduler's
    // generic runtime cap may be larger, but executing extra noise positions
    // would no longer be the official DSpark graph.
    const auto B = std::min(max_tokens, model.arch.speculative.trained_block_size);
    const auto d = model.d_model(), V = model.vocab();
    std::vector<TokenId> ids(B, model.arch.speculative.noise_token_id); ids[0] = anchor;
    std::vector<float> hidden(static_cast<std::size_t>(B) * d);
    for (std::uint32_t t = 0; t < B; ++t) {
        auto row = row_block(model.embed, ids[t], 1);
        if (dequantize(row, std::span<float>(hidden).subspan(static_cast<std::size_t>(t) * d, d)).code() != StatusCode::Ok)
            return StatusCode::Internal;
        bf16_round(std::span<float>(hidden).subspan(static_cast<std::size_t>(t) * d, d));
    }
    out.sink(0, "input_embed", hidden.data(), hidden.size());
    HcState hc; hc_begin(model.arch, hidden.data(), B, hc);
    std::vector<float> normed(hidden.size()), branch(hidden.size());
    for (std::uint32_t stage = 0; stage < p->layers.size(); ++stage) {
        const auto& l = p->layers[stage];
        if (!hc_pre(model.arch, l.hc_attn_fn, l.hc_attn_base, l.hc_attn_scale, hc, hidden.data()))
            return StatusCode::InvalidArgument;
        for (std::uint32_t t = 0; t < B; ++t)
            f32::rmsnorm_into(std::span<const float>(hidden).subspan(static_cast<std::size_t>(t) * d, d),
                              l.input_norm, d, model.arch.rms_norm_eps,
                               std::span<float>(normed).subspan(static_cast<std::size_t>(t) * d, d));
        bf16_round(normed);
        out.sink(stage, "attn_norm", normed.data(), normed.size());
        draft_attention(model, l, *s, stage, s->length, normed.data(), B, branch.data(),
                        out.sink);
        out.sink(stage, "attn_branch", branch.data(), branch.size());
        hc_merge(branch.data(), hc, hidden.data());
        out.sink(stage, "attn_streams", hc.streams.data(), hc.streams.size());
        if (!hc_pre(model.arch, l.hc_ffn_fn, l.hc_ffn_base, l.hc_ffn_scale, hc, hidden.data()))
            return StatusCode::InvalidArgument;
        for (std::uint32_t t = 0; t < B; ++t)
            f32::rmsnorm_into(std::span<const float>(hidden).subspan(static_cast<std::size_t>(t) * d, d),
                              l.post_norm, d, model.arch.rms_norm_eps,
                               std::span<float>(normed).subspan(static_cast<std::size_t>(t) * d, d));
        bf16_round(normed);
        out.sink(stage, "ffn_norm", normed.data(), normed.size());
        if (!draft_ffn(model, *p, l, stage, normed.data(), B, branch.data(), out.sink))
            return StatusCode::IoError;
        out.sink(stage, "ffn_branch", branch.data(), branch.size());
        hc_merge(branch.data(), hc, hidden.data());
        out.sink(stage, "stage_streams", hc.streams.data(), hc.streams.size());
    }
    hc_head(model, *p, hc, hidden.data());
    const auto final_stage = static_cast<std::uint32_t>(p->layers.size() - 1);
    out.sink(final_stage, "head_hidden", hidden.data(), hidden.size());
    out.tokens.clear(); out.confidence.clear();
    out.tokens.reserve(B); out.confidence.reserve(B);
    std::vector<float> head(d), logits(V), embed(model.arch.speculative.markov_rank), bias(V);
    std::vector<float> base_logits(static_cast<std::size_t>(B) * V);
    std::vector<float> final_logits(static_cast<std::size_t>(B) * V);
    std::vector<float> confidence_logits(B);
    TokenId prev = anchor;
    for (std::uint32_t t = 0; t < B; ++t) {
        f32::rmsnorm_into(std::span<const float>(hidden).subspan(static_cast<std::size_t>(t) * d, d),
                          p->norm, d, model.arch.rms_norm_eps, head);
        bf16_round(head);
        matvec(model.out_head, head, logits);
        std::copy(logits.begin(), logits.end(),
                  base_logits.begin() + static_cast<std::size_t>(t) * V);
        if (dequantize(row_block(p->markov_w1, prev, 1), embed).code() != StatusCode::Ok)
            return StatusCode::Internal;
        matvec(p->markov_w2, embed, bias);
        for (std::uint32_t v = 0; v < V; ++v) logits[v] += bias[v];
        std::copy(logits.begin(), logits.end(),
                  final_logits.begin() + static_cast<std::size_t>(t) * V);
        TokenId best = 0;
        for (TokenId v = 1; v < V; ++v) if (logits[v] > logits[best]) best = v;
        std::vector<float> conf_in(d + embed.size());
        std::copy_n(hidden.data() + static_cast<std::size_t>(t) * d, d, conf_in.data());
        std::copy(embed.begin(), embed.end(), conf_in.begin() + d);
        float conf_logit = 0.0f;
        matvec(p->confidence, conf_in, std::span<float>(&conf_logit, 1));
        confidence_logits[t] = conf_logit;
        const float confidence = sigmoid(conf_logit);
        if (threshold > 0.0f && confidence < threshold) break;
        out.tokens.push_back(best); out.confidence.push_back(confidence); prev = best;
    }
    out.sink(final_stage, "base_logits", base_logits.data(), base_logits.size());
    out.sink(final_stage, "final_logits", final_logits.data(), final_logits.size());
    out.sink(final_stage, "confidence", confidence_logits.data(), confidence_logits.size());
    return StatusCode::Ok;
}

Status serialize_state(const F32Model& model, const ArchLayerPayload& state,
                       std::vector<std::byte>& out) {
    const auto* s = state.as<State>();
    if (s == nullptr) return {StatusCode::InvalidArgument, "missing speculative state"};
    const auto W = model.arch.attention.sliding_window;
    const auto live = std::min(s->length, W), width = s->width;
    const auto stages = model.arch.speculative.n_layers;
    out.resize(sizeof(std::uint32_t) * 2 +
               static_cast<std::size_t>(live) * stages * width * 2);
    std::memcpy(out.data(), &s->length, sizeof(s->length));
    std::memcpy(out.data() + sizeof(s->length), &live, sizeof(live));
    auto* dst = reinterpret_cast<std::uint16_t*>(out.data() + sizeof(std::uint32_t) * 2);
    for (std::uint32_t stage = 0; stage < stages; ++stage)
        for (std::uint32_t pos = s->length - live; pos < s->length; ++pos) {
            const auto* src = s->context_kv.data() +
                              (static_cast<std::size_t>(stage) * W + pos % W) * width;
            std::copy_n(src, width, dst); dst += width;
        }
    return {};
}

Status restore_state(const F32Model& model, std::span<const std::byte> payload,
                     ArchLayerPayload& state) {
    auto* s = state.as<State>();
    if (s == nullptr || payload.size() < sizeof(std::uint32_t) * 2)
        return {StatusCode::InvalidArgument, "truncated speculative state"};
    std::uint32_t length = 0, live = 0;
    std::memcpy(&length, payload.data(), sizeof(length));
    std::memcpy(&live, payload.data() + sizeof(length), sizeof(live));
    const auto W = model.arch.attention.sliding_window, width = s->width;
    const auto stages = model.arch.speculative.n_layers;
    const auto want = sizeof(std::uint32_t) * 2 +
                      static_cast<std::size_t>(live) * stages * width * 2;
    if (live > W || live > length || payload.size() != want)
        return {StatusCode::InvalidArgument, "invalid speculative state geometry"};
    std::fill(s->context_kv.begin(), s->context_kv.end(), 0);
    const auto* src = reinterpret_cast<const std::uint16_t*>(payload.data() + sizeof(std::uint32_t) * 2);
    for (std::uint32_t stage = 0; stage < stages; ++stage)
        for (std::uint32_t pos = length - live; pos < length; ++pos) {
            auto* dst = s->context_kv.data() +
                        (static_cast<std::size_t>(stage) * W + pos % W) * width;
            std::copy_n(src, width, dst); src += width;
        }
    s->length = length;
    return {};
}

} // namespace

const SpeculativeBackend& backend() noexcept {
    static const SpeculativeBackend b = [] {
        SpeculativeBackend x;
        x.name = "dspark";
        x.bind_model = &bind_model;
        x.start_runtime = &start_runtime;
        x.open_sequence = &open_sequence;
        x.observe_target = &observe_target;
        x.propose = &propose;
        x.serialize_state = &serialize_state;
        x.restore_state = &restore_state;
        return x;
    }();
    return b;
}

} // namespace soma::arch::deepseek_dspark
