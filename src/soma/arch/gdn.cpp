// Soma — hybrid linear/full attention backend (GDN + GQA). See soma/arch/gdn.hpp.

#include "soma/arch/gdn.hpp"

#include "soma/kernels_f32.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace soma::arch::gdn {
namespace {

std::size_t align64(std::size_t n) noexcept {
    return (n + 63u) & ~std::size_t{63u};
}

inline float sigmoidf(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float siluf(float x) noexcept {
    return x * sigmoidf(x);
}

/// log(1 + e^x), guarded. The naive form overflows for large x and loses the
/// whole value to rounding for very negative x; both are reachable because
/// `a_raw` is an unbounded projection output.
inline float softplusf(float x) noexcept {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

/// Elements ONE full-attention layer caches per token, per plane.
///
/// `n_kv_heads * head_dim`, and the grouping is the whole reason this is not
/// `n_heads * head_dim`: Qwen3.5 projects 64 query heads over 4 key/value heads,
/// so charging the query count would be 16x over on the dominant term.
std::uint64_t plane_floats_per_token(const ArchIr& arch) noexcept {
    return static_cast<std::uint64_t>(arch.attention.n_kv_heads) * arch.attention.head_dim;
}

/// Floats in ONE linear layer's recurrent state.
///
/// Indexed by the VALUE head count, not the key head count. q and k are
/// broadcast up to `n_v_heads` before the recurrence runs, so the state is
/// `n_v_heads x head_k_dim x head_v_dim` — 128 x 128 x 128 on the reference
/// config, 2.1 M floats, 8 MiB per layer. Reading `n_k_heads` instead gives
/// 1 MiB, an 8x under-count in the optimistic direction on a term that is
/// otherwise invisible because it does not grow with context.
std::uint64_t recurrent_floats(const ArchIr& arch) noexcept {
    return arch.attention.gdn.recurrent_elems();
}

/// Floats in ONE linear layer's convolution window.
///
/// One convolution spanning q ++ k ++ v — `2 * key_dim + value_dim` channels —
/// not three separate ones. `kda` has three because Kimi convolves each
/// projection separately; transcribing that shape here would allocate three
/// windows of the wrong width. The window carries `kernel - 1` positions: the
/// current token is an input, not state.
std::uint64_t conv_floats(const ArchIr& arch) noexcept {
    const auto& g = arch.attention.gdn;
    if (g.conv_kernel < 2) return 0;
    return static_cast<std::uint64_t>(g.conv_width()) * (g.conv_kernel - 1);
}

bool layer_is_linear(const ArchIr& arch, std::uint32_t layer) noexcept {
    const auto& g = arch.attention.gdn;
    return layer < g.layer_kinds.size() && g.layer_kinds[layer] == AttnLayerKind::Linear;
}

LayerRegion region_for(const ArchIr& arch,
                       std::uint32_t layer,
                       std::uint32_t context,
                       std::size_t base) noexcept {
    LayerRegion r;
    std::size_t at = align64(base);
    if (layer_is_linear(arch, layer)) {
        r.recurrent = at;
        at = align64(at + static_cast<std::size_t>(recurrent_floats(arch)) * sizeof(float));
        r.conv = at;
        at = align64(at + static_cast<std::size_t>(conv_floats(arch)) * sizeof(float));
        r.k = at; // zero-length: this layer caches no tokens
        r.v = at;
    } else {
        const auto plane =
            static_cast<std::size_t>(plane_floats_per_token(arch)) * context * sizeof(float);
        r.k = at;
        at = align64(at + plane);
        r.v = at;
        at = align64(at + plane);
        r.recurrent = at; // zero-length: this layer carries no recurrent state
        r.conv = at;
    }
    r.end = at;
    return r;
}

std::uint64_t full_layer_weight_bytes(const ArchIr& arch,
                                      AttentionBackend::ByteSizer sizer) noexcept {
    const auto d = arch.topology.d_model;
    const auto& a = arch.attention;
    const auto hq = a.n_heads * a.head_dim;
    const auto hkv = a.n_kv_heads * a.head_dim;

    // q_proj is DOUBLE WIDTH when the output gate is fused into it. That is not
    // a rounding correction: on the reference config it is a second
    // 8192 x 16384 matrix on each of 23 layers. Charging the plain width would
    // under-report the resident half by 6.2 GB at bf16 — and would do so while
    // reporting a model that "just fits".
    const auto q_rows = a.fused_output_gate ? 2u * hq : hq;

    std::uint64_t bytes = sizer(arch, q_rows, d, TensorRole::AttnProj) +
                          2ull * sizer(arch, hkv, d, TensorRole::AttnProj) +
                          sizer(arch, d, hq, TensorRole::AttnProj);

    // q_norm/k_norm are f32 on disk and per head — one `head_dim` vector each,
    // shared across heads. Small, and listed so the total is complete rather
    // than approximately complete.
    if (a.qk_norm == QkNormKind::PerHead) {
        bytes += 2ull * a.head_dim * sizeof(float);
    } else if (a.qk_norm == QkNormKind::FullWidth) {
        bytes += static_cast<std::uint64_t>(hq) * sizeof(float) +
                 static_cast<std::uint64_t>(hkv) * sizeof(float);
    }
    return bytes;
}

std::uint64_t linear_layer_weight_bytes(const ArchIr& arch,
                                        AttentionBackend::ByteSizer sizer) noexcept {
    const auto d = arch.topology.d_model;
    const auto& g = arch.attention.gdn;
    const auto vdim = g.value_dim();

    std::uint64_t bytes = 0;

    // One FUSED q/k/v projection, at `2 * key_dim + value_dim`. Splitting it
    // into three `sizer` calls would give a different answer under a grouped
    // quantization whose group does not divide the individual widths, so the
    // shape on disk is the shape charged.
    bytes += sizer(arch, g.conv_width(), d, TensorRole::AttnProj); // in_proj_qkv
    bytes += sizer(arch, vdim, d, TensorRole::AttnProj);           // in_proj_z

    // beta and the gate input are `n_v_heads` wide — one scalar per head, not
    // per channel. A [128, 8192] matrix each, which is genuinely negligible and
    // is here so that "negligible" is a measured claim rather than an omission.
    bytes += 2ull * sizer(arch, g.n_v_heads, d, TensorRole::AttnProj);

    // Depthwise short convolution: `conv_width x conv_kernel`, f32, NO BIAS.
    // `nn.Conv1d(..., bias=False)` in both Qwen3-Next and Qwen3.5, so unlike
    // `kda` there is no optional bias term to account for.
    bytes += static_cast<std::uint64_t>(g.conv_width()) * g.conv_kernel * sizeof(float);

    // A_log and dt_bias are both [n_v_heads] — bare nn.Parameters, f32.
    // `kda`'s dt_bias is per CHANNEL and its A_log per head; here they are both
    // per head, because this family's whole gate is per head.
    bytes += 2ull * g.n_v_heads * sizeof(float);

    bytes += static_cast<std::uint64_t>(g.head_v_dim) * sizeof(float); // norm.weight
    bytes += sizer(arch, d, vdim, TensorRole::AttnProj);               // out_proj
    return bytes;
}

/// Exact, not averaged.
///
/// `weight_bytes_per_layer` cannot express this family. A linear layer here
/// carries roughly `(2*key_dim + 2*value_dim) * d` of projection against a full
/// layer's `(3*n_heads + 2*n_kv_heads) * head_dim * d`, and the split is 69/23
/// rather than uniform — so an average is a number describing no layer in the
/// model. Leaving `weight_bytes_per_layer` null is what makes the planner take
/// this path instead of multiplying a fiction by 92.
std::uint64_t resident_weight_bytes(const ArchIr& arch,
                                    AttentionBackend::ByteSizer sizer) noexcept {
    const auto& g = arch.attention.gdn;
    return static_cast<std::uint64_t>(g.n_full_layers()) * full_layer_weight_bytes(arch, sizer) +
           static_cast<std::uint64_t>(g.n_linear_layers()) * linear_layer_weight_bytes(arch, sizer);
}

} // namespace

// ── cache geometry ───────────────────────────────────────────────────────────

LayerRegion layer_region(const ArchIr& arch, std::uint32_t layer, std::uint32_t context) noexcept {
    std::size_t at = 0;
    LayerRegion r;
    for (std::uint32_t l = 0; l <= layer && l < arch.topology.n_layers; ++l) {
        r = region_for(arch, l, context, at);
        at = r.end;
    }
    return r;
}

std::uint64_t kv_bytes_for_context(const ArchIr& arch, std::uint32_t context) noexcept {
    std::size_t at = 0;
    for (std::uint32_t l = 0; l < arch.topology.n_layers; ++l)
        at = region_for(arch, l, context, at).end;
    return at;
}

std::uint64_t recurrent_state_bytes(const ArchIr& arch) noexcept {
    // The layout AT ZERO CONTEXT is exactly the part that does not grow: every
    // K/V plane is zero-length there and every recurrent region is full size.
    // Asking the layout rather than re-deriving the sum keeps the alignment
    // padding consistent with what is actually allocated.
    return kv_bytes_for_context(arch, 0);
}

std::size_t kv_bytes_per_token(const ArchIr& arch) noexcept {
    // Analytic, NOT differenced off the layout, for the same reason `kda` says
    // so: one token moves a full layer by `2 * n_kv_heads * head_dim * 4` bytes,
    // and on a small test configuration that is under the 64-byte alignment
    // quantum — two adjacent contexts round to the same padded size and the
    // measured rate comes out zero. A cache that grows for free.
    //
    // Alignment padding is a bounded per-layer CONSTANT, never more than 63
    // bytes per layer however long the context runs, so the slope below is the
    // layout's true asymptotic growth. The padding lives in
    // `kv_bytes_for_context`, which remains the only figure anything allocates
    // against.
    const auto n_full = static_cast<std::size_t>(arch.attention.gdn.n_full_layers());
    return n_full * 2ull * static_cast<std::size_t>(plane_floats_per_token(arch)) * sizeof(float);
}

// ── the kernel ───────────────────────────────────────────────────────────────

void gate(const ArchIr& arch,
          const float* a_log,
          const float* dt_bias,
          const float* a_raw,
          float* out) noexcept {
    const auto& g = arch.attention.gdn;
    for (std::uint32_t h = 0; h < g.n_v_heads; ++h) {
        // `-exp(A_log) * softplus(a + dt_bias)`, strictly negative, so `exp(g)`
        // below is a contraction. Everything here is per head — A_log, dt_bias
        // and the projection output alike — which is what makes this family's
        // decay a scalar.
        out[h] = -std::exp(a_log[h]) * softplusf(a_raw[h] + dt_bias[h]);
    }
}

void short_conv(std::uint32_t width,
                std::uint32_t kernel,
                const float* weight,
                const float* x,
                float* state,
                float* out) noexcept {
    const auto carried = kernel > 0 ? kernel - 1 : 0u;
    for (std::uint32_t c = 0; c < width; ++c) {
        const float* w = weight + static_cast<std::size_t>(c) * kernel;
        float* s = state + static_cast<std::size_t>(c) * carried;

        // Oldest first, so `s[i]` is the input `carried - i` steps back and the
        // weight that multiplies it is `w[i]`. The current token takes the LAST
        // tap. Reversing either indexing convolves the window backwards, which
        // for a kernel of 4 is a plausible-looking smear over the wrong three
        // positions.
        float acc = 0.0f;
        for (std::uint32_t i = 0; i < carried; ++i)
            acc += s[i] * w[i];
        acc += x[c] * w[carried];

        // Advance: drop the oldest, append the current input (NOT the output —
        // the convolution is over the projection, and feeding its own activated
        // result back would make this an IIR filter).
        for (std::uint32_t i = 0; i + 1 < carried; ++i)
            s[i] = s[i + 1];
        if (carried > 0) s[carried - 1] = x[c];

        out[c] = siluf(acc);
    }
}

void step(const ArchIr& arch,
          const float* q,
          const float* k,
          const float* v,
          const float* g,
          const float* beta_raw,
          float* state,
          float* scratch,
          float* out) noexcept {
    const auto& spec = arch.attention.gdn;
    const auto dk = spec.head_k_dim;
    const auto dv = spec.head_v_dim;
    const auto repeat = spec.n_v_heads / spec.n_k_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dk));

    for (std::uint32_t h = 0; h < spec.n_v_heads; ++h) {
        // The broadcast, done by INDEXING rather than by materializing a
        // repeated buffer: value head `h` reads key head `h / repeat`. That is
        // `repeat_interleave` — consecutive value heads share a key head — and
        // NOT `repeat`, which would tile the whole key block and pair head 1
        // with the wrong projection.
        const auto kh = h / repeat;
        const float* qh = q + static_cast<std::size_t>(kh) * dk;
        const float* kh_ptr = k + static_cast<std::size_t>(kh) * dk;
        const float* vh = v + static_cast<std::size_t>(h) * dv;
        float* sh = state + static_cast<std::size_t>(h) * dk * dv;
        float* oh = out + static_cast<std::size_t>(h) * dv;

        // L2 per head, eps INSIDE the square root's argument — matching
        // `rsqrt(sum(x*x) + eps)` rather than `x / (norm + eps)`. The two differ
        // by more than the epsilon suggests when the norm is small.
        //
        // Normalizing the un-repeated projection is bit-identical to
        // normalizing after the broadcast: every copy of a key head holds the
        // same values, so the reduction is the same reduction.
        float qn = 0.0f, kn = 0.0f;
        for (std::uint32_t i = 0; i < dk; ++i) {
            qn += qh[i] * qh[i];
            kn += kh_ptr[i] * kh_ptr[i];
        }
        const float qinv = 1.0f / std::sqrt(qn + 1e-6f);
        const float kinv = 1.0f / std::sqrt(kn + 1e-6f);

        const float decay = std::exp(g[h]);
        const float beta = sigmoidf(beta_raw[h]);

        // 1. Decay the whole state by one scalar.
        const auto n = static_cast<std::size_t>(dk) * dv;
        for (std::size_t i = 0; i < n; ++i)
            sh[i] *= decay;

        // 2. Predict from the DECAYED state: kv_mem = S^T k.
        std::fill_n(scratch, dv, 0.0f);
        for (std::uint32_t i = 0; i < dk; ++i) {
            const float ki = kh_ptr[i] * kinv;
            if (ki == 0.0f) continue;
            const float* row = sh + static_cast<std::size_t>(i) * dv;
            for (std::uint32_t j = 0; j < dv; ++j)
                scratch[j] += row[j] * ki;
        }

        // 3. Correct toward v, scaled by beta, and write the outer product back.
        for (std::uint32_t j = 0; j < dv; ++j)
            scratch[j] = (vh[j] - scratch[j]) * beta;
        for (std::uint32_t i = 0; i < dk; ++i) {
            const float ki = kh_ptr[i] * kinv;
            if (ki == 0.0f) continue;
            float* row = sh + static_cast<std::size_t>(i) * dv;
            for (std::uint32_t j = 0; j < dv; ++j)
                row[j] += ki * scratch[j];
        }

        // 4. Read out from the UPDATED state.
        std::fill_n(oh, dv, 0.0f);
        for (std::uint32_t i = 0; i < dk; ++i) {
            const float qi = qh[i] * qinv * scale;
            if (qi == 0.0f) continue;
            const float* row = sh + static_cast<std::size_t>(i) * dv;
            for (std::uint32_t j = 0; j < dv; ++j)
                oh[j] += row[j] * qi;
        }
    }
}

void gated_rmsnorm(const ArchIr& arch,
                   const float* x,
                   const float* z,
                   const float* weight,
                   float eps,
                   float* out) noexcept {
    const auto& g = arch.attention.gdn;
    const auto dv = g.head_v_dim;
    for (std::uint32_t h = 0; h < g.n_v_heads; ++h) {
        const auto off = static_cast<std::size_t>(h) * dv;
        float sum = 0.0f;
        for (std::uint32_t i = 0; i < dv; ++i)
            sum += x[off + i] * x[off + i];
        const float inv = 1.0f / std::sqrt(sum / static_cast<float>(dv) + eps);
        for (std::uint32_t i = 0; i < dv; ++i) {
            // Norm, then WEIGHT, then the gate. `weight` is one [head_v_dim]
            // vector shared by every head, not one per head.
            out[off + i] = x[off + i] * inv * weight[i] * siluf(z[off + i]);
        }
    }
}

const AttentionBackend& attention_backend() noexcept {
    static const AttentionBackend kBackend = [] {
        AttentionBackend b{};
        b.name = "gdn";
        b.family = AttentionFamily::GqaGdn;
        b.persist_format_id = kKvFormat;
        b.kv_bytes_per_token = &kv_bytes_per_token;
        b.kv_bytes_for_context = &kv_bytes_for_context;
        b.resident_weight_bytes = &resident_weight_bytes;
        // `weight_bytes_per_layer` stays null on purpose — see
        // `resident_weight_bytes`. So do the execution members: this descriptor
        // answers sizing and persistence questions only, and running the family
        // is `resolve_f32_backend`'s answer to give.
        return b;
    }();
    return kBackend;
}

// ── the F32 execution path ───────────────────────────────────────────────────

namespace {

/// A view of ONE layer's K and V planes, shaped as the plane-based KvRow that
/// GQA's cached decode expects.
///
/// The regions this family allocates are per layer and contiguous, so the view
/// carries bases pointing at that layer's planes and is addressed as layer 0 —
/// rather than a zero stride, which would make `k_at` ignore its argument and
/// silently return the same plane if a caller ever passed a real index.
soma::KvRow full_view(const ArchIr& arch, const soma::KvRow& src, LayerIndex layer) noexcept {
    const auto region = layer_region(arch, layer, src.max_ctx);
    const auto width = arch.attention.n_kv_heads * arch.attention.head_dim;
    soma::KvRow v{};
    v.k_base = reinterpret_cast<float*>(src.opaque_base + region.k);
    v.k_stride = static_cast<std::size_t>(src.max_ctx) * width;
    v.k_hkv = width;
    // Unlike MLA, this family stores a REAL V plane: GQA derives nothing.
    v.v_base = reinterpret_cast<float*>(src.opaque_base + region.v);
    v.v_stride = static_cast<std::size_t>(src.max_ctx) * width;
    v.v_hkv = width;
    v.pos = src.pos;
    v.len = src.len;
    v.max_ctx = src.max_ctx;
    return v;
}

/// GQA's entry points read only `lw.attn`, so a delegation needs nothing else —
/// and building a full copy would mean copying its expert vectors on every layer
/// of every forward.
soma::F32LayerWeights borrow_full(const F32HybridWeights& w) {
    soma::F32LayerWeights lw;
    // A NON-OWNING adopt: the payload belongs to the hybrid weights and outlives
    // this call. The no-op deleter is what says so.
    lw.attn.adopt(const_cast<arch::gqa::F32AttnWeights*>(&w.full), [](void*) {});
    return lw;
}

} // namespace

StatusCode f32_bind_layer(const ArchIr& arch,
                          const soma::LayerBindCtx& ctx,
                          soma::ArchLayerPayload& out) noexcept {
    auto* w = new F32HybridWeights();
    out.adopt(w, [](void* p) { delete static_cast<F32HybridWeights*>(p); });

    if (ctx.layer >= arch.attention.gdn.layer_kinds.size()) return StatusCode::InvalidArgument;
    w->linear = layer_is_linear(arch, ctx.layer);

    if (!w->linear) {
        // Delegated, not re-transcribed. GQA's bind already knows this family's
        // full-layer tensor names and now enforces the fused gate's doubled
        // `q_proj` width. The payload it produces holds only non-owning views,
        // so lifting a copy out of it is free.
        soma::ArchLayerPayload tmp;
        if (const auto rc = arch::gqa::f32_bind_layer(arch, ctx, tmp); rc != StatusCode::Ok)
            return rc;
        if (tmp.empty()) return StatusCode::NotFound;
        w->full = *tmp.as<arch::gqa::F32AttnWeights>();
        return StatusCode::Ok;
    }

    using soma::TensorRole;
    // `linear_attn.`, not `self_attn.` — the two layer kinds live under
    // different module names in this checkpoint, which is what makes a
    // mis-classified layer fail loudly at bind time instead of quietly running
    // the wrong operator.
    if (!soma::bind_layer_weight(
             ctx, "linear_attn.in_proj_qkv.weight", TensorRole::AttnProj, w->in_proj_qkv)
             .ok() ||
        !soma::bind_layer_weight(
             ctx, "linear_attn.in_proj_z.weight", TensorRole::AttnProj, w->in_proj_z)
             .ok() ||
        !soma::bind_layer_weight(
             ctx, "linear_attn.in_proj_b.weight", TensorRole::AttnProj, w->in_proj_b)
             .ok() ||
        !soma::bind_layer_weight(
             ctx, "linear_attn.in_proj_a.weight", TensorRole::AttnProj, w->in_proj_a)
             .ok() ||
        !soma::bind_layer_weight(
             ctx, "linear_attn.out_proj.weight", TensorRole::AttnProj, w->out_proj)
             .ok()) {
        return StatusCode::NotFound;
    }

    // The convolution and the state-space parameters are f32 on disk. `A_log`
    // and `dt_bias` are bare nn.Parameters, so they carry no ".weight" suffix —
    // binding them as though they did fails with "tensor not found" on a
    // checkpoint that has them.
    if (!soma::bind_layer_f32(ctx, "linear_attn.conv1d.weight", w->conv_w).ok() ||
        !soma::bind_layer_f32(ctx, "linear_attn.A_log", w->a_log).ok() ||
        !soma::bind_layer_f32(ctx, "linear_attn.dt_bias", w->dt_bias).ok() ||
        !soma::bind_layer_f32(ctx, "linear_attn.norm.weight", w->o_norm).ok()) {
        return StatusCode::NotFound;
    }

    // Shapes, checked against the IR rather than trusted. A conv weight of the
    // wrong width would read off the end of the tensor for the last channels;
    // an `a_log` of the wrong length would silently gate the wrong heads.
    const auto& g = arch.attention.gdn;
    if (w->conv_w.size() != static_cast<std::size_t>(g.conv_width()) * g.conv_kernel ||
        w->a_log.size() != g.n_v_heads || w->dt_bias.size() != g.n_v_heads ||
        w->o_norm.size() != g.head_v_dim) {
        return StatusCode::InvalidArgument;
    }
    return StatusCode::Ok;
}

StatusCode f32_linear_layer(const ArchIr& arch,
                            const F32HybridWeights& w,
                            const float* x,
                            std::uint32_t n_tokens,
                            float* recurrent,
                            float* conv,
                            float* out) noexcept {
    const auto& g = arch.attention.gdn;
    const auto d = arch.topology.d_model;
    const auto cw = g.conv_width();
    const auto kd = g.key_dim();
    const auto vd = g.value_dim();
    const auto nv = g.n_v_heads;
    const std::span<const float> xs(x, static_cast<std::size_t>(n_tokens) * d);

    // Projections batch over the whole span; only the convolution and the
    // recurrence are inherently sequential. Splitting it here is what keeps
    // prefill off a per-token matvec.
    std::vector<float> qkv(static_cast<std::size_t>(n_tokens) * cw);
    std::vector<float> z(static_cast<std::size_t>(n_tokens) * vd);
    std::vector<float> braw(static_cast<std::size_t>(n_tokens) * nv);
    std::vector<float> araw(static_cast<std::size_t>(n_tokens) * nv);
    soma::matmul(w.in_proj_qkv, xs, n_tokens, qkv);
    soma::matmul(w.in_proj_z, xs, n_tokens, z);
    soma::matmul(w.in_proj_b, xs, n_tokens, braw);
    soma::matmul(w.in_proj_a, xs, n_tokens, araw);

    std::vector<float> mixed(cw);
    std::vector<float> gvec(nv);
    std::vector<float> scratch(g.head_v_dim);
    std::vector<float> core(static_cast<std::size_t>(n_tokens) * vd);

    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        short_conv(cw,
                   g.conv_kernel,
                   w.conv_w.data(),
                   qkv.data() + static_cast<std::size_t>(t) * cw,
                   conv,
                   mixed.data());
        gate(arch,
             w.a_log.data(),
             w.dt_bias.data(),
             araw.data() + static_cast<std::size_t>(t) * nv,
             gvec.data());
        // q ++ k ++ v, in that order, out of the single convolved buffer. The
        // two key-width slices come first and the value slice is the remainder —
        // which is only obvious until you notice `key_dim != value_dim` here, so
        // a symmetric three-way split would be wrong by construction.
        step(arch,
             mixed.data(),
             mixed.data() + kd,
             mixed.data() + 2 * kd,
             gvec.data(),
             braw.data() + static_cast<std::size_t>(t) * nv,
             recurrent,
             scratch.data(),
             core.data() + static_cast<std::size_t>(t) * vd);
    }

    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        const auto off = static_cast<std::size_t>(t) * vd;
        gated_rmsnorm(arch,
                      core.data() + off,
                      z.data() + off,
                      w.o_norm.data(),
                      arch.rms_norm_eps,
                      core.data() + off);
    }

    soma::matmul(
        w.out_proj, core, n_tokens, std::span<float>(out, static_cast<std::size_t>(n_tokens) * d));
    return StatusCode::Ok;
}

StatusCode f32_attention(const ArchIr& arch,
                         const soma::F32LayerWeights& lw,
                         const float* x,
                         std::uint32_t n_tokens,
                         soma::F32Workspace& ws,
                         float* out) noexcept {
    if (lw.attn.empty()) return StatusCode::InvalidArgument;
    const auto& w = *lw.attn.as<F32HybridWeights>();
    if (!w.linear) {
        const auto borrowed = borrow_full(w);
        return arch::gqa::f32_attention(arch, borrowed, x, n_tokens, ws, out);
    }
    // Cacheless: the sequence starts from a zero state, which is what "no cache"
    // MEANS for a recurrent layer. Carrying a leftover state here would make the
    // uncached path depend on whatever ran before it.
    std::vector<float> recurrent(static_cast<std::size_t>(recurrent_floats(arch)), 0.0f);
    std::vector<float> conv(static_cast<std::size_t>(conv_floats(arch)), 0.0f);
    return f32_linear_layer(arch, w, x, n_tokens, recurrent.data(), conv.data(), out);
}

StatusCode f32_attention_kv(const ArchIr& arch,
                            const soma::F32LayerWeights& lw,
                            const float* x,
                            std::uint32_t n_rows,
                            LayerIndex layer,
                            const soma::KvRow* rows,
                            soma::F32Workspace& ws,
                            float* out) noexcept {
    if (lw.attn.empty() || rows == nullptr) return StatusCode::InvalidArgument;
    const auto& w = *lw.attn.as<F32HybridWeights>();
    const auto d = arch.topology.d_model;

    if (!w.linear) {
        // Rows are re-pointed at this layer's planes and the delegated call is
        // told layer 0, because within that view there IS only one layer.
        std::vector<soma::KvRow> views(n_rows);
        for (std::uint32_t r = 0; r < n_rows; ++r) {
            if (rows[r].opaque_base == nullptr) return StatusCode::InvalidArgument;
            views[r] = full_view(arch, rows[r], layer);
        }
        const auto borrowed = borrow_full(w);
        return arch::gqa::f32_attention_kv(
            arch, borrowed, x, n_rows, /*layer=*/0, views.data(), ws, out);
    }

    // Each row is its OWN sequence with its own state, so they cannot be batched
    // through one recurrence the way a cache-attending family batches its rows.
    for (std::uint32_t r = 0; r < n_rows; ++r) {
        const auto& row = rows[r];
        if (row.opaque_base == nullptr) return StatusCode::InvalidArgument;
        const auto region = layer_region(arch, layer, row.max_ctx);
        if (region.end > row.opaque_bytes) return StatusCode::InvalidArgument;
        auto* recurrent = reinterpret_cast<float*>(row.opaque_base + region.recurrent);
        auto* conv = reinterpret_cast<float*>(row.opaque_base + region.conv);
        const auto rc = f32_linear_layer(arch,
                                         w,
                                         x + static_cast<std::size_t>(r) * d,
                                         1,
                                         recurrent,
                                         conv,
                                         out + static_cast<std::size_t>(r) * d);
        if (rc != StatusCode::Ok) return rc;
    }
    return StatusCode::Ok;
}

const soma::F32Backend& f32_backend() noexcept {
    static const soma::F32Backend kBackend = [] {
        soma::F32Backend b{};
        b.name = "gdn";
        b.bind_layer = &f32_bind_layer;
        b.attention = &f32_attention;
        b.attention_kv = &f32_attention_kv;

        // GQA's router, pointed at DIRECTLY, and that is safe here for a reason
        // worth stating because the analogous line in `arch/kda` was a defect.
        //
        // `kda` pointed `route` at `mla::f32_route`, which recovers a selection
        // bias by casting `lw.attn` to `mla::F32AttnWeights` — and a hybrid
        // layer's payload is a `F32HybridWeights`. Undefined behaviour, silently
        // dropping the bias.
        //
        // `gqa::f32_route` reads NO payload at all: it takes `lw` and
        // immediately discards it, because this router has no parameters beyond
        // the gate matrix the caller already applied. Qwen3.5 scores with
        // softmax over every expert, takes top-k, and renormalizes — which is
        // that function exactly, with `normalize_topk` forced true by the
        // adapter. So there is nothing to mis-cast and nothing to re-transcribe.
        b.route = &arch::gqa::f32_route;

        // No block-residual hooks: this family has an ordinary single-stream
        // residual, so the core's plain `hidden += branch` is the model.
        //
        // No kv_geometry either — this family's cache is opaque, so there are no
        // planes to describe. KvCache takes the opaque path because the
        // attention backend supplies kv_bytes_for_context, and never asks.
        return b;
    }();
    return kBackend;
}

} // namespace soma::arch::gdn
