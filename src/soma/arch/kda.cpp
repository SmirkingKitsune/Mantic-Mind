// Soma — hybrid linear/full attention backend (KDA + MLA).
//
// See include/soma/arch/kda.hpp for the family, the reference config, the
// transcribed kernel semantics, and why the per-sequence cost is affine rather
// than linear in context.

#include "soma/arch/kda.hpp"

#include "soma/kernels_f32.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace soma::arch::kda {
namespace {

std::size_t align64(std::size_t n) noexcept {
    return (n + 63u) & ~std::size_t{63u};
}

/// Elements one full-attention layer caches per token.
///
/// The compressed latent plus the shared RoPE slice — `kv_lora_rank +
/// qk_rope_head_dim`, independent of head count, which is the point of the
/// family. NoPE does not narrow this: the slice is still projected, still
/// concatenated into every key, and still cached. Only the rotation is absent.
/// Subtracting it here because "there is no RoPE" would under-size every full
/// layer's cache by 64 floats per token.
std::uint64_t latent_floats_per_token(const ArchIr& arch) noexcept {
    const auto& m = arch.attention.mla;
    return static_cast<std::uint64_t>(m.kv_lora_rank) + m.qk_rope_head_dim;
}

std::uint64_t recurrent_floats(const ArchIr& arch) noexcept {
    const auto& k = arch.attention.kda;
    // The delta-rule state is a MATRIX per head, not a vector: head_dim x
    // head_dim. On Kimi-K3 that is 96 x 128 x 128 = 1.57 M floats per layer, and
    // reading it as `n_heads * head_dim` would under-count by 128x — which
    // would present as a model that plans comfortably and then exhausts RAM at
    // the first sequence.
    return static_cast<std::uint64_t>(k.n_heads) * k.head_dim * k.head_dim;
}

std::uint64_t conv_floats(const ArchIr& arch) noexcept {
    const auto& k = arch.attention.kda;
    if (k.conv_kernel < 2) return 0;
    // Three windows — q, k and v each own a convolution — of `kernel - 1`
    // positions. The current token is an input rather than state, which is why
    // it is not `kernel`.
    return 3ull * k.n_heads * k.head_dim * (k.conv_kernel - 1);
}

LayerRegion region_for(const ArchIr& arch,
                       std::uint32_t layer,
                       std::uint32_t context,
                       std::size_t base) noexcept {
    const auto& kda = arch.attention.kda;
    const bool linear =
        layer < kda.layer_kinds.size() && kda.layer_kinds[layer] == AttnLayerKind::Linear;

    LayerRegion r;
    std::size_t at = align64(base);
    if (linear) {
        r.recurrent = at;
        at = align64(at + static_cast<std::size_t>(recurrent_floats(arch)) * sizeof(float));
        r.conv = at;
        at = align64(at + static_cast<std::size_t>(conv_floats(arch)) * sizeof(float));
        r.latent = at; // zero-length: this layer caches no tokens
    } else {
        r.latent = at;
        at = align64(at + static_cast<std::size_t>(latent_floats_per_token(arch)) * context *
                              sizeof(float));
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
    const auto& m = a.mla;
    const auto qk = m.qk_nope_head_dim + m.qk_rope_head_dim;

    std::uint64_t bytes = 0;

    // Same two shapes as plain MLA — the compression is unchanged by the hybrid.
    if (m.q_lora_rank > 0) {
        bytes += sizer(arch, m.q_lora_rank, d, TensorRole::AttnProj);
        bytes += sizer(arch, a.n_heads * qk, m.q_lora_rank, TensorRole::AttnProj);
        bytes += static_cast<std::uint64_t>(m.q_lora_rank) * sizeof(float); // q_a_layernorm
    } else {
        bytes += sizer(arch, a.n_heads * qk, d, TensorRole::AttnProj);
    }

    bytes += sizer(arch, m.kv_lora_rank + m.qk_rope_head_dim, d, TensorRole::AttnProj);
    bytes += sizer(arch,
                   a.n_heads * (m.qk_nope_head_dim + m.v_head_dim),
                   m.kv_lora_rank,
                   TensorRole::AttnProj);
    bytes += static_cast<std::uint64_t>(m.kv_lora_rank) * sizeof(float); // kv_a_layernorm

    // Sized on v_head_dim, not head_dim: they coincide here (both 128) and
    // differ on other MLA models, so assuming either is right half the time.
    bytes += sizer(arch, d, a.n_heads * m.v_head_dim, TensorRole::AttnProj);

    // The output gate is a FULL projection, not a scalar. At 7168 x 12288 it is
    // 88 M parameters per full layer — about a fifth of the layer's attention
    // weights — so treating `mla_use_output_gate` as a behavioural flag with no
    // footprint would under-count the resident half by that much on 24 layers.
    if (m.output_gate) {
        bytes += sizer(arch, a.n_heads * m.v_head_dim, d, TensorRole::AttnProj);
    }
    return bytes;
}

std::uint64_t linear_layer_weight_bytes(const ArchIr& arch,
                                        AttentionBackend::ByteSizer sizer) noexcept {
    const auto d = arch.topology.d_model;
    const auto& k = arch.attention.kda;
    const auto proj = static_cast<std::uint32_t>(k.n_heads * k.head_dim);

    std::uint64_t bytes = 0;

    // q, k and v are all full-width here: this family does not group its keys,
    // so there is no n_kv_heads reduction to apply. Borrowing GQA's
    // `2 * n_kv_heads * head_dim` would be the same category of error the seam
    // was created to stop.
    bytes += 3ull * sizer(arch, proj, d, TensorRole::AttnProj);

    // Depthwise short convolutions, one per projection, plus their state-space
    // parameters. Small individually and f32 on disk; grouped here so the list
    // is complete rather than approximately complete.
    bytes += 3ull * static_cast<std::uint64_t>(proj) * k.conv_kernel * sizeof(float);
    bytes += static_cast<std::uint64_t>(k.n_heads) * sizeof(float); // A_log
    bytes += static_cast<std::uint64_t>(proj) * sizeof(float);      // dt_bias

    // The forget gate is always low-rank through head_dim.
    bytes += sizer(arch, k.head_dim, d, TensorRole::AttnProj);    // f_a_proj
    bytes += sizer(arch, proj, k.head_dim, TensorRole::AttnProj); // f_b_proj
    bytes += sizer(arch, k.n_heads, d, TensorRole::AttnProj);     // b_proj (beta)

    // The OUTPUT gate is not. Full-rank is d x proj; low-rank is the same pair
    // of shapes as the forget gate — a 56x difference in this tensor on
    // Kimi-K3, which is why `full_rank_gate` is an IR field and not a default.
    if (k.full_rank_gate) {
        bytes += sizer(arch, proj, d, TensorRole::AttnProj);
    } else {
        bytes += sizer(arch, k.head_dim, d, TensorRole::AttnProj);
        bytes += sizer(arch, proj, k.head_dim, TensorRole::AttnProj);
    }

    bytes += static_cast<std::uint64_t>(k.head_dim) * sizeof(float); // o_norm
    bytes += sizer(arch, d, proj, TensorRole::AttnProj);             // o_proj
    return bytes;
}

/// Exact, not averaged.
///
/// `weight_bytes_per_layer` cannot express this family: a full layer and a
/// linear layer differ by roughly a factor of two, and the split is 24/69 rather
/// than uniform. MLA amortises its DSA indexer over the stack because the
/// indexer is a small correction on top of layers that are otherwise identical;
/// here the two layer types share no tensor at all, so an average would be a
/// number describing no layer in the model.
std::uint64_t resident_weight_bytes(const ArchIr& arch,
                                    AttentionBackend::ByteSizer sizer) noexcept {
    const auto& k = arch.attention.kda;
    const auto n_full = static_cast<std::uint64_t>(k.n_full_layers());
    const auto n_linear = static_cast<std::uint64_t>(k.n_linear_layers());
    return n_full * full_layer_weight_bytes(arch, sizer) +
           n_linear * linear_layer_weight_bytes(arch, sizer);
}

inline float sigmoidf(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

/// log(1 + e^x), guarded. The naive form overflows for large x and loses the
/// whole value to rounding for very negative x; both are reachable here because
/// `g_raw` is an unbounded projection output.
inline float softplusf(float x) noexcept {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
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
    // The layout at zero context IS the part that does not grow — every latent
    // region is zero-length there and every recurrent one is full size. Asking
    // the layout rather than re-deriving it keeps the padding consistent with
    // what was actually allocated.
    return kv_bytes_for_context(arch, 0);
}

std::size_t kv_bytes_per_token(const ArchIr& arch) noexcept {
    // Analytic, NOT differenced off the layout — and the difference between
    // those two is the reason this comment exists.
    //
    // Differencing adjacent contexts is the obvious way to keep a rate honest
    // against its layout, and it is what the compressed-sparse backend does. It
    // works there because one token moves that layout by kilobytes. Here a
    // full-attention layer grows by `(kv_lora_rank + qk_rope_head_dim) * 4`
    // bytes per token, which on a small configuration is under the 64-byte
    // alignment quantum — so two adjacent contexts round to the SAME padded
    // size and the measured rate comes out as zero. A cache that grows for free.
    //
    // Alignment padding is a bounded per-layer CONSTANT, not a rate: it never
    // exceeds 63 bytes per layer however long the context runs. So the slope
    // below is the layout's true asymptotic growth, and the padding lives where
    // it belongs — in `kv_bytes_for_context`, which remains the only figure
    // anything allocates against.
    const auto n_full = static_cast<std::size_t>(arch.attention.kda.n_full_layers());
    return n_full * static_cast<std::size_t>(latent_floats_per_token(arch)) * sizeof(float);
}

// ── the kernel ───────────────────────────────────────────────────────────────

void gate(const ArchIr& arch,
          const float* a_log,
          const float* dt_bias,
          const float* g_raw,
          float* out) noexcept {
    const auto& k = arch.attention.kda;
    for (std::uint32_t h = 0; h < k.n_heads; ++h) {
        // ONE scalar per head, broadcast across that head's channels — while
        // dt_bias and g_raw are per channel. Indexing a_log per channel would
        // read past the end of a [n_heads] tensor; indexing dt_bias per head
        // would apply one channel's bias to all of them.
        const float a = std::exp(a_log[h]);
        for (std::uint32_t c = 0; c < k.head_dim; ++c) {
            const auto i = static_cast<std::size_t>(h) * k.head_dim + c;
            const float x = g_raw[i] + (dt_bias != nullptr ? dt_bias[i] : 0.0f);
            // Two genuinely different gates, and the config chooses.
            //
            // Safe: `lower_bound * sigmoid(exp(A_log) * x)` lands in
            // (lower_bound, 0), so the decay `exp(g)` cannot fall below
            // exp(lower_bound) — the state is bounded away from annihilation.
            // Kimi-K3 sets -5.0, i.e. a floor of ~0.0067 per step.
            //
            // Unbounded: `-exp(A_log) * softplus(x)` can decay arbitrarily hard.
            // Substituting one for the other keeps every shape and every sign
            // and changes how long the model remembers.
            out[i] = k.has_gate_bound ? k.gate_lower_bound * sigmoidf(a * x) : -a * softplusf(x);
        }
    }
}

void short_conv(std::uint32_t width,
                std::uint32_t kernel,
                const float* weight,
                const float* bias,
                const float* x,
                float* state,
                float* out) noexcept {
    if (kernel == 0) return;
    const std::uint32_t carried = kernel - 1;
    for (std::uint32_t c = 0; c < width; ++c) {
        float* win = state + static_cast<std::size_t>(c) * carried;
        const float* w = weight + static_cast<std::size_t>(c) * kernel;
        // `weight[kernel-1]` multiplies the CURRENT token: PyTorch's conv1d
        // computes out[t] = sum_j w[j] * x[t - (kernel-1) + j], so the window
        // runs oldest-first and the last tap is the newest input. Reversing it
        // is a different filter that is equally finite.
        float acc = bias != nullptr ? bias[c] : 0.0f;
        for (std::uint32_t j = 0; j < carried; ++j)
            acc += w[j] * win[j];
        acc += w[carried] * x[c];
        // Shift after accumulating, never before: the window must hold the
        // inputs BEFORE this token while it is being read.
        for (std::uint32_t j = 0; j + 1 < carried; ++j)
            win[j] = win[j + 1];
        if (carried > 0) win[carried - 1] = x[c];
        out[c] = acc * sigmoidf(acc); // SiLU
    }
}

void step(const ArchIr& arch,
          const float* q,
          const float* k_in,
          const float* v,
          const float* g,
          const float* beta_raw,
          float* state,
          float* scratch,
          float* out) noexcept {
    const auto& cfg = arch.attention.kda;
    const std::uint32_t H = cfg.n_heads, D = cfg.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (std::uint32_t h = 0; h < H; ++h) {
        const std::size_t off = static_cast<std::size_t>(h) * D;
        float* S = state + static_cast<std::size_t>(h) * D * D; // [key][value]
        const float* qh = q + off;
        const float* kh = k_in + off;
        const float* vh = v + off;
        const float* gh = g + off;

        // L2 per head, then the scale. Both belong to the operator rather than
        // to the projection — `use_qk_l2norm_in_kernel` is why the checkpoint
        // has no separate norm weight to bind.
        float qn = 0.0f, kn = 0.0f;
        for (std::uint32_t i = 0; i < D; ++i) {
            qn += qh[i] * qh[i];
            kn += kh[i] * kh[i];
        }
        qn = 1.0f / std::sqrt(qn + 1e-6f);
        kn = 1.0f / std::sqrt(kn + 1e-6f);
        const float beta = sigmoidf(beta_raw[h]);

        // 1. Decay along the KEY axis, by a DIAGONAL matrix. This is the whole
        //    of what makes it KDA rather than GatedDeltaNet: one factor per
        //    channel, not one per head.
        // 2. Predict from the DECAYED state. Predicting first — from S before
        //    the decay — is the plausible reordering, and it is a different
        //    operator that also converges.
        //
        // Fused into one pass over S, so the state is read once. `pred` is
        // accumulated against the decayed values as they are written.
        float* const pred = scratch;
        for (std::uint32_t j = 0; j < D; ++j)
            pred[j] = 0.0f;
        for (std::uint32_t i = 0; i < D; ++i) {
            const float decay = std::exp(gh[i]);
            const float ki = kh[i] * kn;
            float* row = S + static_cast<std::size_t>(i) * D;
            for (std::uint32_t j = 0; j < D; ++j) {
                row[j] *= decay;
                pred[j] += ki * row[j];
            }
        }
        // 3. Delta rule: correct the state by what it got wrong about v.
        for (std::uint32_t i = 0; i < D; ++i) {
            const float bk = beta * kh[i] * kn;
            if (bk == 0.0f) continue;
            float* row = S + static_cast<std::size_t>(i) * D;
            for (std::uint32_t j = 0; j < D; ++j)
                row[j] += bk * (vh[j] - pred[j]);
        }
        // 4. Read out from the UPDATED state. Reading before the update loses
        //    the current token entirely — and only for the current token, so it
        //    survives every aggregate check and fails on exact-match ones.
        float* const oh = out + off;
        for (std::uint32_t j = 0; j < D; ++j)
            oh[j] = 0.0f;
        for (std::uint32_t i = 0; i < D; ++i) {
            const float qi = qh[i] * qn * scale;
            if (qi == 0.0f) continue;
            const float* row = S + static_cast<std::size_t>(i) * D;
            for (std::uint32_t j = 0; j < D; ++j)
                oh[j] += qi * row[j];
        }
    }
}

void gated_rmsnorm(const ArchIr& arch,
                   const float* x,
                   const float* gate_raw,
                   const float* weight,
                   float eps,
                   float* out) noexcept {
    const auto& cfg = arch.attention.kda;
    const std::uint32_t H = cfg.n_heads, D = cfg.head_dim;
    for (std::uint32_t h = 0; h < H; ++h) {
        const std::size_t off = static_cast<std::size_t>(h) * D;
        // Per HEAD, not over the flattened projection: the norm weight is
        // [head_dim]. Normalising across all heads together would couple them.
        float ss = 0.0f;
        for (std::uint32_t i = 0; i < D; ++i)
            ss += x[off + i] * x[off + i];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(D) + eps);
        for (std::uint32_t i = 0; i < D; ++i) {
            // normalize -> weight -> gate. `fla` gates AFTER the norm; Mamba's
            // similarly-named RMSNormGated multiplies the gate in BEFORE it, so
            // the name alone does not settle this.
            out[off + i] = x[off + i] * inv * weight[i] * sigmoidf(gate_raw[off + i]);
        }
    }
}

// ── the F32 execution path ───────────────────────────────────────────────────

namespace {

/// Precompute `norm.weight * proj.weight` — the only form the mixing needs.
StatusCode bind_res_score(const soma::LayerBindCtx& ctx,
                          const char* norm_suffix,
                          const char* proj_suffix,
                          std::uint32_t d,
                          std::vector<float>& out) noexcept {
    std::span<const float> nw, pw;
    if (!soma::bind_layer_f32(ctx, norm_suffix, nw).ok() ||
        !soma::bind_layer_f32(ctx, proj_suffix, pw).ok()) {
        return StatusCode::NotFound;
    }
    if (nw.size() < d || pw.size() < d) return StatusCode::InvalidArgument;
    out.resize(d);
    for (std::uint32_t i = 0; i < d; ++i)
        out[i] = nw[i] * pw[i];
    return StatusCode::Ok;
}

bool layer_is_linear(const ArchIr& arch, LayerIndex layer) noexcept {
    const auto& k = arch.attention.kda;
    return layer < k.layer_kinds.size() && k.layer_kinds[layer] == AttnLayerKind::Linear;
}

/// A view of ONE layer's latent region, shaped as the plane-based KvRow that
/// MLA's cached decode expects.
///
/// The regions this family allocates are per layer and contiguous, so the view
/// carries a base pointing at that layer's plane and is addressed as layer 0 —
/// rather than a zero stride, which would make `k_at` ignore its argument and
/// silently return the same plane if a caller ever passed a real index.
soma::KvRow latent_view(const ArchIr& arch, const soma::KvRow& src, LayerIndex layer) noexcept {
    const auto region = layer_region(arch, layer, src.max_ctx);
    const auto width = static_cast<std::uint32_t>(latent_floats_per_token(arch));
    soma::KvRow v{};
    v.k_base = reinterpret_cast<float*>(src.opaque_base + region.latent);
    v.k_stride = static_cast<std::size_t>(src.max_ctx) * width;
    v.k_hkv = width;
    v.v_base = nullptr; // plain MLA derives V from the latent; there is no plane
    v.v_stride = 0;
    v.v_hkv = 0;
    v.pos = src.pos;
    v.len = src.len;
    v.max_ctx = src.max_ctx;
    return v;
}

/// MLA's entry points read only `lw.attn`, so a delegation needs nothing else —
/// and building a full copy would mean copying its expert vectors on every
/// layer of every forward.
soma::F32LayerWeights borrow_full(const F32HybridWeights& w) {
    soma::F32LayerWeights lw;
    // A NON-OWNING adopt: the payload belongs to the hybrid weights and outlives
    // this call. The no-op deleter is what says so.
    lw.attn.adopt(const_cast<arch::mla::F32AttnWeights*>(&w.full), [](void*) {});
    return lw;
}

} // namespace

StatusCode f32_bind_layer(const ArchIr& arch,
                          const soma::LayerBindCtx& ctx,
                          soma::ArchLayerPayload& out) noexcept {
    auto* w = new F32HybridWeights();
    out.adopt(w, [](void* p) { delete static_cast<F32HybridWeights*>(p); });

    if (ctx.layer >= arch.attention.kda.layer_kinds.size()) return StatusCode::InvalidArgument;
    w->linear = layer_is_linear(arch, ctx.layer);

    // Router and block-residual tensors live on EVERY layer, both kinds, and are
    // bound before the attention split because they have nothing to do with it.
    //
    // The bias is optional because dense layers have none — but when a MoE layer
    // that should have one does not, the router runs unbiased and picks
    // different experts, fluently. That is what a hardcoded "mlp." block name
    // did here before the oracle caught it.
    {
        const auto bias_name = arch.naming.moe_block + ".gate.e_score_correction_bias";
        (void)soma::bind_layer_f32(ctx, bias_name.c_str(), w->e_score_bias, /*optional=*/true);
    }

    if (arch.block_residual.block_size != 0) {
        const auto d = arch.topology.d_model;
        if (const auto rc = bind_res_score(ctx,
                                           "self_attention_res_norm.weight",
                                           "self_attention_res_proj.weight",
                                           d,
                                           w->attn_res_score);
            rc != StatusCode::Ok)
            return rc;
        if (const auto rc = bind_res_score(
                ctx, "mlp_res_norm.weight", "mlp_res_proj.weight", d, w->mlp_res_score);
            rc != StatusCode::Ok)
            return rc;
    }

    if (!w->linear) {
        // Delegated, not re-transcribed. MLA's bind already knows which shapes
        // this family's full layers have, enforces its own all-or-nothing rules,
        // and now binds the output gate too. The payload it produces holds only
        // non-owning views, so lifting a copy out of it is free.
        soma::ArchLayerPayload tmp;
        if (const auto rc = arch::mla::f32_bind_layer(arch, ctx, tmp); rc != StatusCode::Ok)
            return rc;
        if (tmp.empty()) return StatusCode::NotFound;
        w->full = *tmp.as<arch::mla::F32AttnWeights>();
        return StatusCode::Ok;
    }

    using soma::TensorRole;
    const auto& k = arch.attention.kda;

    if (!soma::bind_layer_weight(ctx, "self_attn.q_proj.weight", TensorRole::AttnProj, w->q_proj)
             .ok() ||
        !soma::bind_layer_weight(ctx, "self_attn.k_proj.weight", TensorRole::AttnProj, w->k_proj)
             .ok() ||
        !soma::bind_layer_weight(ctx, "self_attn.v_proj.weight", TensorRole::AttnProj, w->v_proj)
             .ok() ||
        !soma::bind_layer_weight(
             ctx, "self_attn.f_a_proj.weight", TensorRole::AttnProj, w->f_a_proj)
             .ok() ||
        !soma::bind_layer_weight(
             ctx, "self_attn.f_b_proj.weight", TensorRole::AttnProj, w->f_b_proj)
             .ok() ||
        !soma::bind_layer_weight(ctx, "self_attn.b_proj.weight", TensorRole::AttnProj, w->b_proj)
             .ok() ||
        !soma::bind_layer_weight(ctx, "self_attn.o_proj.weight", TensorRole::AttnProj, w->o_proj)
             .ok()) {
        return StatusCode::NotFound;
    }

    // The convolutions and the state-space parameters are f32 on disk. A_log and
    // dt_bias are bare nn.Parameters, so they carry no ".weight" suffix — binding
    // them as though they did fails with "tensor not found" on a checkpoint that
    // has them.
    if (!soma::bind_layer_f32(ctx, "self_attn.q_conv1d.weight", w->q_conv_w).ok() ||
        !soma::bind_layer_f32(ctx, "self_attn.k_conv1d.weight", w->k_conv_w).ok() ||
        !soma::bind_layer_f32(ctx, "self_attn.v_conv1d.weight", w->v_conv_w).ok() ||
        !soma::bind_layer_f32(ctx, "self_attn.A_log", w->a_log).ok() ||
        !soma::bind_layer_f32(ctx, "self_attn.dt_bias", w->dt_bias).ok() ||
        !soma::bind_layer_f32(ctx, "self_attn.o_norm.weight", w->o_norm).ok()) {
        return StatusCode::NotFound;
    }
    // Genuinely optional: ShortConvolution can be built with or without one, and
    // both are real checkpoints. Absent means no bias, not a broken bind.
    (void)soma::bind_layer_f32(ctx, "self_attn.q_conv1d.bias", w->q_conv_b, /*optional=*/true);
    (void)soma::bind_layer_f32(ctx, "self_attn.k_conv1d.bias", w->k_conv_b, /*optional=*/true);
    (void)soma::bind_layer_f32(ctx, "self_attn.v_conv1d.bias", w->v_conv_b, /*optional=*/true);

    // The output gate's rank is architecture, not a fallback to try in turn: a
    // full-rank checkpoint has no g_a/g_b and a low-rank one has no g_proj, so
    // asking for the wrong one is a missing tensor rather than a missing option.
    if (k.full_rank_gate) {
        if (!soma::bind_layer_weight(
                 ctx, "self_attn.g_proj.weight", TensorRole::AttnProj, w->g_proj)
                 .ok())
            return StatusCode::NotFound;
    } else {
        if (!soma::bind_layer_weight(
                 ctx, "self_attn.g_a_proj.weight", TensorRole::AttnProj, w->g_a_proj)
                 .ok() ||
            !soma::bind_layer_weight(
                 ctx, "self_attn.g_b_proj.weight", TensorRole::AttnProj, w->g_b_proj)
                 .ok())
            return StatusCode::NotFound;
    }
    return StatusCode::Ok;
}

StatusCode f32_route(const ArchIr& arch,
                     const soma::F32LayerWeights& lw,
                     const TokenId*,
                     const float* logits,
                     std::uint32_t n_tokens,
                     std::uint32_t* out_ids,
                     float* out_weights) noexcept {
    // Kimi's MoE gate IS DeepSeek-V3's — its own source says so, mapping its
    // parameter names onto DeepSeek's — so the scoring lives in the MLA backend
    // and is reached through the span-taking entry point.
    //
    // NOT by pointing `b.route` straight at `mla::f32_route`, which is what this
    // did first. That function recovers the bias with
    // `lw.attn.as<F32AttnWeights>()`, and a hybrid layer's payload is a
    // `F32HybridWeights`. The cast is undefined behaviour; in practice it read a
    // garbage span length, failed the size check, and dropped the bias — so the
    // router silently ran without `noaux_tc` on every layer.
    if (lw.attn.empty()) return StatusCode::InvalidArgument;
    const auto& w = *lw.attn.as<F32HybridWeights>();
    return arch::mla::f32_route_with_bias(
        arch, w.e_score_bias, logits, n_tokens, out_ids, out_weights);
}

StatusCode f32_linear_layer(const ArchIr& arch,
                            const F32HybridWeights& w,
                            const float* x,
                            std::uint32_t n_tokens,
                            float* recurrent,
                            float* conv,
                            float* out) noexcept {
    const auto& k = arch.attention.kda;
    const auto d = arch.topology.d_model;
    const auto proj = static_cast<std::uint32_t>(k.n_heads) * k.head_dim;
    const auto carried = k.conv_kernel > 0 ? k.conv_kernel - 1 : 0u;
    const std::span<const float> xs(x, static_cast<std::size_t>(n_tokens) * d);

    // Projections batch over the whole span; only the convolution and the
    // recurrence are inherently sequential. Splitting it here is what keeps
    // prefill off a per-token matvec.
    std::vector<float> qr(static_cast<std::size_t>(n_tokens) * proj);
    std::vector<float> kr(qr.size()), vr(qr.size()), graw(qr.size()), gate_raw(qr.size());
    std::vector<float> fa(static_cast<std::size_t>(n_tokens) * k.head_dim);
    std::vector<float> braw(static_cast<std::size_t>(n_tokens) * k.n_heads);
    soma::matmul(w.q_proj, xs, n_tokens, qr);
    soma::matmul(w.k_proj, xs, n_tokens, kr);
    soma::matmul(w.v_proj, xs, n_tokens, vr);
    soma::matmul(w.f_a_proj, xs, n_tokens, fa);
    soma::matmul(w.f_b_proj, fa, n_tokens, graw);
    soma::matmul(w.b_proj, xs, n_tokens, braw);
    if (k.full_rank_gate) {
        soma::matmul(w.g_proj, xs, n_tokens, gate_raw);
    } else {
        std::vector<float> ga(static_cast<std::size_t>(n_tokens) * k.head_dim);
        soma::matmul(w.g_a_proj, xs, n_tokens, ga);
        soma::matmul(w.g_b_proj, ga, n_tokens, gate_raw);
    }

    // Three windows in one region, in the order q, k, v.
    float* const conv_q = conv;
    float* const conv_k = conv + static_cast<std::size_t>(proj) * carried;
    float* const conv_v = conv + 2 * static_cast<std::size_t>(proj) * carried;

    std::vector<float> qc(proj), kc(proj), vc(proj), gbuf(proj), obuf(proj);
    std::vector<float> scratch(k.head_dim);
    std::vector<float> normed(static_cast<std::size_t>(n_tokens) * proj);

    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        const auto off = static_cast<std::size_t>(t) * proj;
        short_conv(proj,
                   k.conv_kernel,
                   w.q_conv_w.data(),
                   w.q_conv_b.empty() ? nullptr : w.q_conv_b.data(),
                   qr.data() + off,
                   conv_q,
                   qc.data());
        short_conv(proj,
                   k.conv_kernel,
                   w.k_conv_w.data(),
                   w.k_conv_b.empty() ? nullptr : w.k_conv_b.data(),
                   kr.data() + off,
                   conv_k,
                   kc.data());
        short_conv(proj,
                   k.conv_kernel,
                   w.v_conv_w.data(),
                   w.v_conv_b.empty() ? nullptr : w.v_conv_b.data(),
                   vr.data() + off,
                   conv_v,
                   vc.data());
        gate(arch, w.a_log.data(), w.dt_bias.data(), graw.data() + off, gbuf.data());
        step(arch,
             qc.data(),
             kc.data(),
             vc.data(),
             gbuf.data(),
             braw.data() + static_cast<std::size_t>(t) * k.n_heads,
             recurrent,
             scratch.data(),
             obuf.data());
        gated_rmsnorm(arch,
                      obuf.data(),
                      gate_raw.data() + off,
                      w.o_norm.data(),
                      arch.rms_norm_eps,
                      normed.data() + off);
    }

    soma::matmul(
        w.o_proj, normed, n_tokens, std::span<float>(out, static_cast<std::size_t>(n_tokens) * d));
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
        return arch::mla::f32_attention(arch, borrowed, x, n_tokens, ws, out);
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
        // Rows are re-pointed at this layer's latent plane and the delegated call
        // is told layer 0, because within that view there IS only one layer.
        std::vector<soma::KvRow> views(n_rows);
        for (std::uint32_t r = 0; r < n_rows; ++r) {
            if (rows[r].opaque_base == nullptr) return StatusCode::InvalidArgument;
            views[r] = latent_view(arch, rows[r], layer);
        }
        const auto borrowed = borrow_full(w);
        return arch::mla::f32_attention_kv(
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

// ── block residual ───────────────────────────────────────────────────────────
//
// Every `block_size`-th layer pushes a copy of the residual stream onto a
// per-token stack; each layer then mixes over that stack with learned softmax
// scores. This is NOT `HyperConnectionSpec` — that is Sinkhorn-normalized mixing
// over a widened residual — and folding the two together would run one model as
// the other.
//
// The subtlety worth stating: the candidates are SCORED normalized and COMBINED
// unnormalized. `_apply_attn_res` RMS-normalizes each candidate only to compute
// its score, then averages the raw vectors under those weights. Combining the
// normalized ones instead is a plausible reading that discards every candidate's
// magnitude.

namespace {

BlockResidualState& state_for(const ArchIr& arch, std::uint32_t n_tokens, soma::F32Workspace& ws) {
    auto* st = ws.arch_state.as<BlockResidualState>();
    if (st == nullptr) {
        st = new BlockResidualState();
        ws.arch_state.adopt(st, [](void* p) { delete static_cast<BlockResidualState*>(p); });
    }
    const auto d = arch.topology.d_model;
    const auto max_blocks = arch.block_residual.n_blocks(arch.topology.n_layers);
    if (st->n_tokens != n_tokens || st->width != d) {
        st->n_tokens = n_tokens;
        st->width = d;
        st->prefix.assign(static_cast<std::size_t>(n_tokens) * d, 0.0f);
        st->stack.assign(static_cast<std::size_t>(n_tokens) * max_blocks * d, 0.0f);
    }
    return *st;
}

} // namespace

void mix_block_residual(const BlockResidualState& st,
                        std::span<const float> score,
                        float eps,
                        const float* prefix,
                        float* out) noexcept {
    const auto d = st.width;
    const auto B = st.n_blocks;
    const auto max_blocks =
        st.stack.empty() || st.n_tokens == 0
            ? 0u
            : static_cast<std::uint32_t>(st.stack.size() /
                                         (static_cast<std::size_t>(st.n_tokens) * d));
    std::vector<float> scores(static_cast<std::size_t>(B) + 1);
    // ALIAS-SAFE by construction, because one caller genuinely aliases: the
    // final mix passes `hidden` as both the prefix candidate and the output.
    //
    // Accumulating straight into `out` there is silently wrong rather than
    // loud: the destination is zeroed, partially summed, and then read back as
    // the prefix candidate, so the mix folds in a fraction of its own running
    // total. With three equal-weighted candidates {v, 0, prefix} that turns
    // v/3 into 4v/9 — a finite, plausible, entirely wrong number.
    std::vector<float> blend(d);
    for (std::uint32_t tok = 0; tok < st.n_tokens; ++tok) {
        const auto tbase = static_cast<std::size_t>(tok) * max_blocks * d;
        // The candidate list is the stack THEN the prefix, in that order. Order
        // is invisible to the result — softmax over a set — but keeping it the
        // reference's makes a tap comparison meaningful.
        const auto candidate = [&](std::uint32_t ci) -> const float* {
            return ci < B ? st.stack.data() + tbase + static_cast<std::size_t>(ci) * d
                          : prefix + static_cast<std::size_t>(tok) * d;
        };
        float best = -3.0e38f;
        for (std::uint32_t ci = 0; ci <= B; ++ci) {
            const float* v = candidate(ci);
            float ss = 0.0f;
            for (std::uint32_t i = 0; i < d; ++i)
                ss += v[i] * v[i];
            const float inv = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);
            float acc = 0.0f;
            for (std::uint32_t i = 0; i < d; ++i)
                acc += v[i] * inv * score[i];
            scores[ci] = acc;
            if (acc > best) best = acc;
        }
        float sum = 0.0f;
        for (std::uint32_t ci = 0; ci <= B; ++ci) {
            scores[ci] = std::exp(scores[ci] - best);
            sum += scores[ci];
        }
        const float norm = sum > 0.0f ? 1.0f / sum : 0.0f;
        std::fill(blend.begin(), blend.end(), 0.0f);
        for (std::uint32_t ci = 0; ci <= B; ++ci) {
            // The RAW candidate, not the normalized one used for scoring.
            const float* v = candidate(ci);
            const float p = scores[ci] * norm;
            for (std::uint32_t i = 0; i < d; ++i)
                blend[i] += p * v[i];
        }
        // Written only once every candidate has been read.
        std::copy_n(blend.data(), d, out + static_cast<std::size_t>(tok) * d);
    }
}

StatusCode f32_bind_model(const ArchIr& arch,
                          const soma::ModelBindCtx& ctx,
                          soma::ArchLayerPayload& out) noexcept {
    auto* m = new F32HybridModel();
    out.adopt(m, [](void* p) { delete static_cast<F32HybridModel*>(p); });
    if (arch.block_residual.block_size == 0) return StatusCode::Ok;

    const auto d = arch.topology.d_model;
    std::span<const float> nw, pw;
    if (!soma::bind_model_f32(ctx, "model.output_attn_res_norm.weight", nw).ok() ||
        !soma::bind_model_f32(ctx, "model.output_attn_res_proj.weight", pw).ok()) {
        return StatusCode::NotFound;
    }
    if (nw.size() < d || pw.size() < d) return StatusCode::InvalidArgument;
    m->out_res_score.resize(d);
    for (std::uint32_t i = 0; i < d; ++i)
        m->out_res_score[i] = nw[i] * pw[i];
    return StatusCode::Ok;
}

StatusCode f32_begin_forward(const ArchIr& arch,
                             const soma::ArchLayerPayload&,
                             const TokenId*,
                             std::uint32_t n_tokens,
                             soma::F32Workspace& ws,
                             float*) noexcept {
    if (arch.block_residual.block_size == 0) return StatusCode::Ok;
    auto& st = state_for(arch, n_tokens, ws);
    // A stack left over from a previous prompt has the wrong contents and would
    // be mixed in silently. `reset_arch_state` exists for this, but the counters
    // are cleared here too so a reused workspace cannot carry them.
    st.n_blocks = 0;
    st.prefix_valid = false;
    return StatusCode::Ok;
}

StatusCode f32_pre_attention(const ArchIr& arch,
                             const soma::F32LayerWeights& lw,
                             std::uint32_t n_tokens,
                             soma::F32Workspace& ws,
                             float* hidden) noexcept {
    if (arch.block_residual.block_size == 0) return StatusCode::Ok;
    if (lw.attn.empty()) return StatusCode::InvalidArgument;
    const auto& w = *lw.attn.as<F32HybridWeights>();
    auto& st = state_for(arch, n_tokens, ws);
    const auto d = arch.topology.d_model;
    const auto span = static_cast<std::size_t>(n_tokens) * d;

    // The residual stream ENTERING this layer becomes the prefix sum.
    std::copy_n(hidden, span, st.prefix.begin());
    st.prefix_valid = true;

    // Guarded on an empty stack, unlike the pre-FFN mix below: at layer 0 there
    // is nothing to mix with, and the reference skips it rather than mixing a
    // candidate list of one (which would be the identity anyway, but only
    // because softmax over one element is 1).
    if (st.n_blocks > 0) {
        mix_block_residual(st, w.attn_res_score, arch.rms_norm_eps, st.prefix.data(), hidden);
    }

    // Push AFTER mixing and BEFORE attention, then drop the prefix: at a block
    // boundary this layer does not carry the incoming residual forward, it
    // restarts from its own attention output.
    if (ws.current_layer % arch.block_residual.block_size == 0) {
        const auto max_blocks = arch.block_residual.n_blocks(arch.topology.n_layers);
        if (st.n_blocks >= max_blocks) return StatusCode::InvalidArgument;
        for (std::uint32_t tok = 0; tok < n_tokens; ++tok) {
            std::copy_n(st.prefix.data() + static_cast<std::size_t>(tok) * d,
                        d,
                        st.stack.data() +
                            (static_cast<std::size_t>(tok) * max_blocks + st.n_blocks) * d);
        }
        ++st.n_blocks;
        st.prefix_valid = false;
    }
    return StatusCode::Ok;
}

StatusCode f32_merge_attention(const ArchIr& arch,
                               const soma::F32LayerWeights&,
                               const float* branch,
                               std::uint32_t n_tokens,
                               soma::F32Workspace& ws,
                               float* hidden) noexcept {
    const auto d = arch.topology.d_model;
    const auto span = static_cast<std::size_t>(n_tokens) * d;
    if (arch.block_residual.block_size == 0) {
        for (std::size_t i = 0; i < span; ++i)
            hidden[i] += branch[i];
        return StatusCode::Ok;
    }
    auto& st = state_for(arch, n_tokens, ws);
    if (st.prefix_valid) {
        for (std::size_t i = 0; i < span; ++i)
            st.prefix[i] += branch[i];
    } else {
        std::copy_n(branch, span, st.prefix.begin());
        st.prefix_valid = true;
    }
    // `hidden` carries the prefix out of this hook; the pre-FFN mix replaces it.
    std::copy_n(st.prefix.data(), span, hidden);
    return StatusCode::Ok;
}

StatusCode f32_pre_ffn(const ArchIr& arch,
                       const soma::F32LayerWeights& lw,
                       std::uint32_t n_tokens,
                       soma::F32Workspace& ws,
                       float* hidden) noexcept {
    if (arch.block_residual.block_size == 0) return StatusCode::Ok;
    if (lw.attn.empty()) return StatusCode::InvalidArgument;
    const auto& w = *lw.attn.as<F32HybridWeights>();
    auto& st = state_for(arch, n_tokens, ws);
    // UNGUARDED, unlike the pre-attention mix: layer 0 pushed its snapshot above,
    // so the stack is never empty by the time control reaches here.
    mix_block_residual(st, w.mlp_res_score, arch.rms_norm_eps, st.prefix.data(), hidden);
    return StatusCode::Ok;
}

StatusCode f32_merge_ffn(const ArchIr& arch,
                         const soma::F32LayerWeights&,
                         const float* branch,
                         std::uint32_t n_tokens,
                         soma::F32Workspace& ws,
                         float* hidden) noexcept {
    const auto d = arch.topology.d_model;
    const auto span = static_cast<std::size_t>(n_tokens) * d;
    if (arch.block_residual.block_size == 0) {
        for (std::size_t i = 0; i < span; ++i)
            hidden[i] += branch[i];
        return StatusCode::Ok;
    }
    auto& st = state_for(arch, n_tokens, ws);
    for (std::size_t i = 0; i < span; ++i)
        st.prefix[i] += branch[i];
    std::copy_n(st.prefix.data(), span, hidden);
    return StatusCode::Ok;
}

StatusCode f32_end_forward(const ArchIr& arch,
                           const soma::ArchLayerPayload& model_payload,
                           std::uint32_t n_tokens,
                           soma::F32Workspace& ws,
                           float* hidden) noexcept {
    if (arch.block_residual.block_size == 0) return StatusCode::Ok;
    if (model_payload.empty()) return StatusCode::InvalidArgument;
    const auto& m = *model_payload.as<F32HybridModel>();
    auto& st = state_for(arch, n_tokens, ws);
    // The last mix, before the model's output norm. `hidden` is the final prefix
    // sum, which is exactly the candidate the reference passes here.
    mix_block_residual(st, m.out_res_score, arch.rms_norm_eps, hidden, hidden);
    return StatusCode::Ok;
}

const soma::F32Backend& f32_backend() noexcept {
    static const soma::F32Backend kBackend = [] {
        soma::F32Backend b{};
        b.name = "kda";
        b.bind_layer = &f32_bind_layer;
        b.bind_model = &f32_bind_model;
        b.attention = &f32_attention;
        b.attention_kv = &f32_attention_kv;
        // The block-residual hooks. All six are set unconditionally and each
        // returns early when `block_size` is zero — the merge pair falling back
        // to the plain `hidden += branch` the core would have done. Setting them
        // conditionally would mean the backend's behaviour depended on which IR
        // it was first asked about, since the descriptor is a function-local
        // static built once.
        b.begin_forward = &f32_begin_forward;
        b.pre_attention = &f32_pre_attention;
        b.merge_attention = &f32_merge_attention;
        b.pre_ffn = &f32_pre_ffn;
        b.merge_ffn = &f32_merge_ffn;
        b.end_forward = &f32_end_forward;
        // Our own thin wrapper, NOT `mla::f32_route` directly: the scoring is
        // shared but the payload type is not. See f32_route above.
        b.route = &f32_route;
        // No kv_geometry: this family's cache is opaque, so there are no planes
        // to describe. KvCache takes the opaque path because the attention
        // backend supplies kv_bytes_for_context, and never asks.
        return b;
    }();
    return kBackend;
}

const AttentionBackend& attention_backend() noexcept {
    static const AttentionBackend kBackend = [] {
        AttentionBackend b{};
        b.name = "kda";
        b.family = AttentionFamily::MlaKda;
        b.persist_format_id = kKvFormat;
        b.kv_bytes_per_token = &kv_bytes_per_token;
        b.kv_bytes_for_context = &kv_bytes_for_context;
        b.resident_weight_bytes = &resident_weight_bytes;
        // Execution members stay null, and `weight_bytes_per_layer` does too:
        // this stack has no meaningful per-layer average, and leaving it null is
        // what makes the planner take the exact path above rather than silently
        // multiplying a fiction by 93.
        return b;
    }();
    return kBackend;
}

} // namespace soma::arch::kda
