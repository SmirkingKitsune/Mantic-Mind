// Soma — MLA (Multi-head Latent Attention), fp32 reference path.
//
// The second architecture through the seam. Designed against DeepSeek-V2-Lite,
// whose differences from GQA are structural rather than parametric:
//
//   * K and V are not projected per head. One low-rank latent of width
//     `kv_lora_rank` is projected from the hidden state, normalised, and then
//     expanded by kv_b_proj into every head's K-nope and V. The KV cache holds
//     the LATENT, which is the whole point of MLA — it is `kv_lora_rank + rope`
//     wide instead of `n_kv_heads * head_dim`.
//   * A query head is two segments with different jobs: `qk_nope_head_dim`
//     carries content and is never rotated, `qk_rope_head_dim` carries position
//     and is. GQA rotates the whole head.
//   * The value head is a DIFFERENT WIDTH from the query head (`v_head_dim` vs
//     `qk_nope + qk_rope`). Every GQA implementation assumes one head_dim for
//     all three.
//   * The RoPE segment is shared across heads, MQA-style — one k_pe per token,
//     broadcast.
//
// One of only two translation units permitted to name an architecture (the
// other is arch_registry.cpp). tools/ci/check_seam.py enforces that.

#include "soma/arch/mla.hpp"

#include "soma/f32_model.hpp"
#include "soma/kernels_f32.hpp"
#include "soma/threading.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace soma::arch::mla {

namespace {

constexpr float kPi = 3.14159265358979323846f;

/// YaRN's attention-temperature correction.
///
/// `yarn_get_mscale(scale, mscale) = 0.1 * mscale * ln(scale) + 1`, and 1 when
/// the scale is not actually scaling. It appears TWICE in the reference with
/// different meanings, which is the trap: once squared on the softmax scale, and
/// once as a ratio applied to cos/sin. Conflating them changes the attention
/// temperature by a few percent — plausible output, wrong model.
float yarn_mscale(float scale, float mscale) noexcept {
    if (scale <= 1.0f) return 1.0f;
    return 0.1f * mscale * std::log(scale) + 1.0f;
}

float yarn_correction_dim(float rotations, std::uint32_t dim, float base, float max_pos) noexcept {
    return (static_cast<float>(dim) * std::log(max_pos / (rotations * 2.0f * kPi))) /
           (2.0f * std::log(base));
}

/// Per-dimension inverse frequencies under YaRN.
///
/// Low frequencies are interpolated (divided by the scaling factor), high ones
/// are extrapolated unchanged, and a linear ramp between the two correction
/// bounds blends them. Using plain RoPE here would be finite, plausible and
/// wrong at every position past the original context.
void yarn_inv_freq(const RopeConfig& rope, std::uint32_t dim, std::vector<float>& out) {
    const auto half = dim / 2;
    out.resize(half);

    const float base = (rope.theta > 1.0f) ? rope.theta : 10000.0f;

    // Plain RoPE unless the model actually asks for YaRN.
    //
    // This ran UNCONDITIONALLY, which was wrong twice over: it applied YaRN's
    // interpolation to models that never requested it, and on degenerate inputs
    // it produced NaN — `log(max_pos / (rotations * 2pi))` goes to +/-inf when a
    // beta is zero or the base is 1, and `high - low` then evaluates inf - inf.
    // The NaN reached the rope taps while every projection before it was clean,
    // which is what pointed here.
    const bool yarn = (rope.scaling.kind == RopeScalingKind::Yarn) &&
                      (rope.scaling.factor > 1.0f) && (rope.scaling.beta_fast > 0.0f) &&
                      (rope.scaling.beta_slow > 0.0f);

    if (!yarn) {
        for (std::uint32_t i = 0; i < half; ++i) {
            const float e = static_cast<float>(2 * i) / static_cast<float>(dim);
            out[i] = 1.0f / std::pow(base, e);
        }
        return;
    }

    const float factor = rope.scaling.factor;
    const float orig_max = (rope.scaling.original_max_position > 0)
                               ? static_cast<float>(rope.scaling.original_max_position)
                               : 4096.0f;

    float low = std::floor(yarn_correction_dim(rope.scaling.beta_fast, dim, base, orig_max));
    float high = std::ceil(yarn_correction_dim(rope.scaling.beta_slow, dim, base, orig_max));
    low = std::max(low, 0.0f);
    high = std::min(high, static_cast<float>(dim) - 1.0f);
    if (!(high > low)) high = low + 0.001f; // also catches high < low and NaN

    for (std::uint32_t i = 0; i < half; ++i) {
        const float e = static_cast<float>(2 * i) / static_cast<float>(dim);
        const float extra = 1.0f / std::pow(base, e); // no interpolation
        const float inter = extra / factor;           // full interpolation
        // ramp = 0 at `low` and below, 1 at `high` and above.
        const float ramp = std::clamp((static_cast<float>(i) - low) / (high - low), 0.0f, 1.0f);
        const float mask = 1.0f - ramp;
        out[i] = inter * (1.0f - mask) + extra * mask;
    }
}

/// Rotate a rope segment in place, at `pos`, in INTERLEAVED pairing.
///
/// Pairs are `(v[2i], v[2i+1])`, and this is CONFIRMED rather than assumed:
/// transformers' native DeepseekV2 rotates via
///
///     view_as_complex(x.reshape(..., -1, 2)) * freqs_cis
///
/// which treats adjacent elements as (real, imag) — the complex multiply expands
/// to exactly the recurrence below. The `q_pe_rot` / `k_pe_rot` taps agree with
/// it to 2.4e-07.
///
/// The rotate-half form (`(v[i], v[i + dim/2])`) is the other convention in
/// circulation and belongs to DeepSeek's original `trust_remote_code`. It was
/// tried and measured worse; the tap now says why that comparison was
/// uninformative — final-error A/B ranks two wrong answers when a third thing is
/// also wrong.
void rope_at(float* v,
             std::uint32_t dim,
             std::uint32_t pos,
             const std::vector<float>& inv_freq,
             float cs_scale) noexcept {
    const auto half = dim / 2;
    for (std::uint32_t i = 0; i < half; ++i) {
        const float angle = static_cast<float>(pos) * inv_freq[i];
        const float c = std::cos(angle) * cs_scale;
        const float s = std::sin(angle) * cs_scale;
        const float a = v[2 * i];
        const float b = v[2 * i + 1];
        v[2 * i] = a * c - b * s;
        v[2 * i + 1] = b * c + a * s;
    }
}

} // namespace

StatusCode f32_bind_layer(const ArchIr& arch,
                          const soma::LayerBindCtx& ctx,
                          soma::ArchLayerPayload& out) noexcept {
    auto* w = new F32AttnWeights();
    out.adopt(w, [](void* p) { delete static_cast<F32AttnWeights*>(p); });

    using soma::TensorRole;
    const auto& m = arch.attention.mla;

    // q_lora_rank == 0 means the query is NOT down-projected. V2-Lite is built
    // that way and the full V2 is not, so both shapes have to be bound from the
    // same code rather than assumed.
    if (m.q_lora_rank == 0) {
        if (!soma::bind_layer_weight(
                 ctx, "self_attn.q_proj.weight", TensorRole::AttnProj, w->q_proj)
                 .ok()) {
            return StatusCode::NotFound;
        }
    } else {
        if (!soma::bind_layer_weight(
                 ctx, "self_attn.q_a_proj.weight", TensorRole::AttnProj, w->q_a_proj)
                 .ok() ||
            !soma::bind_layer_weight(
                 ctx, "self_attn.q_b_proj.weight", TensorRole::AttnProj, w->q_b_proj)
                 .ok() ||
            !soma::bind_layer_f32(ctx, "self_attn.q_a_layernorm.weight", w->q_a_norm).ok()) {
            return StatusCode::NotFound;
        }
    }

    if (!soma::bind_layer_weight(
             ctx, "self_attn.kv_a_proj_with_mqa.weight", TensorRole::AttnProj, w->kv_a_proj)
             .ok() ||
        !soma::bind_layer_f32(ctx, "self_attn.kv_a_layernorm.weight", w->kv_a_norm).ok() ||
        !soma::bind_layer_weight(
             ctx, "self_attn.kv_b_proj.weight", TensorRole::AttnProj, w->kv_b_proj)
             .ok() ||
        !soma::bind_layer_weight(ctx, "self_attn.o_proj.weight", TensorRole::AttnProj, w->o_proj)
             .ok()) {
        return StatusCode::NotFound;
    }

    // Optional: present only on V3's MoE layers. Absent on V2 entirely and on
    // any dense layer, so a missing tensor here is normal rather than an error.
    (void)soma::bind_layer_f32(ctx,
                               "mlp.gate.e_score_correction_bias",
                               w->e_score_bias,
                               /*optional=*/true);
    return StatusCode::Ok;
}

StatusCode f32_attention(const ArchIr& arch,
                         const soma::F32LayerWeights& lw,
                         const float* x,
                         std::uint32_t n_tokens,
                         soma::F32Workspace& ws,
                         float* out) noexcept {
    if (lw.attn.empty()) return StatusCode::InvalidArgument;
    const auto& w = *lw.attn.as<F32AttnWeights>();

    const auto& m = arch.attention.mla;
    const auto d = arch.topology.d_model;
    const auto H = arch.attention.n_heads;
    const auto nope = m.qk_nope_head_dim;
    const auto rope_d = m.qk_rope_head_dim;
    const auto qk = nope + rope_d; // one query/key head
    const auto vd = m.v_head_dim;  // a DIFFERENT width
    const auto lora = m.kv_lora_rank;
    const auto eps = arch.rms_norm_eps;

    // ── the softmax scale, with YaRN's correction squared into it ────────────
    float scale = 1.0f / std::sqrt(static_cast<float>(qk));
    const auto& rope = arch.attention.rope;
    if (rope.scaling.kind == RopeScalingKind::Yarn) {
        // The attention factor, as transformers' GENERIC yarn init computes it —
        // which is what the native DeepseekV2 uses, and it is NOT the same as
        // DeepSeek's original remote code:
        //
        //   remote code   scale *= mscale(factor, mscale_all_dim)^2
        //   generic       attention_factor = mscale(f, mscale) / mscale(f, mscale_all_dim)
        //
        // With `mscale == mscale_all_dim == 0.707` the ratio is exactly 1 and the
        // correction VANISHES, where the squared form gives 1.58962. Applying the
        // wrong one leaves every projection and both rope outputs exact and only
        // the attention weights wrong — a sharper softmax over correct scores —
        // which is precisely what the taps showed.
        const float num = yarn_mscale(rope.scaling.factor, rope.scaling.mscale);
        const float den = yarn_mscale(rope.scaling.factor, rope.scaling.mscale_all_dim);
        if (den != 0.0f) scale *= num / den;
    }
    // The OTHER use of the same function: a ratio folded into cos/sin. Equal
    // mscale and mscale_all_dim make it exactly 1, which is the V2-Lite case —
    // but the two are independent config keys and a model that sets them apart
    // needs both.
    float cs_scale = 1.0f;
    if (rope.scaling.kind == RopeScalingKind::Yarn) {
        const float num = yarn_mscale(rope.scaling.factor, rope.scaling.mscale);
        const float den = yarn_mscale(rope.scaling.factor, rope.scaling.mscale_all_dim);
        if (den != 0.0f) cs_scale = num / den;
    }

    std::vector<float> inv_freq;
    yarn_inv_freq(rope, rope_d, inv_freq);

    // SOMA_MLA_PROBE prints the two YaRN quantities that are easy to get wrong
    // and impossible to see in the output. Kept because it already earned its
    // place: it ruled the rope parse OUT as the cause of the G0 gap by matching
    // hand-computed reference values to nine digits, which no amount of reading
    // the code was going to establish.
    if (std::getenv("SOMA_MLA_PROBE") != nullptr) {
        std::fprintf(stderr, "[mla] qk=%u scale=%.9f cs=%.9f inv_freq=", qk, scale, cs_scale);
        for (const auto f : inv_freq)
            std::fprintf(stderr, " %.9f", f);
        std::fprintf(stderr, "\n");
    }

    const std::span<const float> xs(x, static_cast<std::size_t>(n_tokens) * d);

    // ── queries ──────────────────────────────────────────────────────────────
    std::vector<float> q(static_cast<std::size_t>(n_tokens) * H * qk);
    if (m.q_lora_rank == 0) {
        soma::matmul(w.q_proj, xs, n_tokens, q);
    } else {
        std::vector<float> qa(static_cast<std::size_t>(n_tokens) * m.q_lora_rank);
        soma::matmul(w.q_a_proj, xs, n_tokens, qa);
        for (std::uint32_t t = 0; t < n_tokens; ++t) {
            f32::rmsnorm(std::span<float>(qa).subspan(static_cast<std::size_t>(t) * m.q_lora_rank,
                                                      m.q_lora_rank),
                         w.q_a_norm,
                         m.q_lora_rank,
                         eps);
        }
        soma::matmul(w.q_b_proj, qa, n_tokens, q);
    }

    // Taps chosen to land on MODULE BOUNDARIES, so the Python side can hook the
    // corresponding submodule directly and the two are comparable without either
    // side reshaping. A tap in the middle of a fused step would be cheap here and
    // unmatchable there.
    ws.sink(ws.current_layer, "q_proj", q.data(), q.size());

    // ── the compressed latent, and the shared rope segment ───────────────────
    std::vector<float> ckv(static_cast<std::size_t>(n_tokens) * (lora + rope_d));
    soma::matmul(w.kv_a_proj, xs, n_tokens, ckv);
    ws.sink(ws.current_layer, "kv_a_proj", ckv.data(), ckv.size());

    std::vector<float> latent(static_cast<std::size_t>(n_tokens) * lora);
    std::vector<float> k_pe(static_cast<std::size_t>(n_tokens) * rope_d);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        const float* src = ckv.data() + static_cast<std::size_t>(t) * (lora + rope_d);
        std::copy_n(src, lora, latent.data() + static_cast<std::size_t>(t) * lora);
        std::copy_n(src + lora, rope_d, k_pe.data() + static_cast<std::size_t>(t) * rope_d);

        // The latent is normalised BEFORE expansion. Normalising after would be
        // a different function entirely.
        f32::rmsnorm(std::span<float>(latent).subspan(static_cast<std::size_t>(t) * lora, lora),
                     w.kv_a_norm,
                     lora,
                     eps);
        rope_at(k_pe.data() + static_cast<std::size_t>(t) * rope_d, rope_d, t, inv_freq, cs_scale);
    }

    ws.sink(ws.current_layer, "kv_a_layernorm", latent.data(), latent.size());

    // kv_b expands the latent into every head's K-nope ++ V.
    std::vector<float> kv(static_cast<std::size_t>(n_tokens) * H * (nope + vd));
    soma::matmul(w.kv_b_proj, latent, n_tokens, kv);
    ws.sink(ws.current_layer, "kv_b_proj", kv.data(), kv.size());

    // Rotate each head's query rope segment.
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        for (std::uint32_t h = 0; h < H; ++h) {
            float* qh = q.data() + (static_cast<std::size_t>(t) * H + h) * qk;
            rope_at(qh + nope, rope_d, t, inv_freq, cs_scale);
        }
    }

    // The taps that module boundaries cannot reach.
    //
    // q_pe and k_pe after rotation are intermediate tensors — no torch module
    // produces them — so the reference side monkey-patches apply_rotary_pos_emb
    // rather than hooking. They are the last unobserved step before the scores,
    // and the one the interleaved-vs-rotate-half question actually turns on:
    // comparing them directly settles empirically what A/B-ing the final error
    // could only rank.
    //
    // The q gather is strided, so it is guarded on the sink being live. In
    // production this is one branch that always predicts.
    ws.sink(ws.current_layer, "k_pe_rot", k_pe.data(), k_pe.size());
    if (ws.sink) {
        // Emitted HEAD-MAJOR — [h][t][rope] — to match torch's
        // [batch, n_heads, seq, rope], not the engine's own [t][h] layout.
        //
        // The first version emitted token-major and reported max|diff| = 1.708
        // against an exact k_pe. That looked like a bug in the q rope path and
        // was a bug in the TAP: k_pe has a single head, so its two layouts
        // coincide, which is precisely why it matched and q did not. A tap whose
        // layout disagrees with the reference manufactures a divergence that no
        // amount of staring at the rotation code will explain.
        std::vector<float> qpe(static_cast<std::size_t>(n_tokens) * H * rope_d);
        for (std::uint32_t h = 0; h < H; ++h) {
            for (std::uint32_t t = 0; t < n_tokens; ++t) {
                const float* src = q.data() + (static_cast<std::size_t>(t) * H + h) * qk + nope;
                std::copy_n(src,
                            rope_d,
                            qpe.data() + (static_cast<std::size_t>(h) * n_tokens + t) * rope_d);
            }
        }
        ws.sink(ws.current_layer, "q_pe_rot", qpe.data(), qpe.size());
    }

    // ── attention ────────────────────────────────────────────────────────────
    std::vector<float> heads(static_cast<std::size_t>(n_tokens) * H * vd, 0.0f);
    std::vector<float> scores(n_tokens);

    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        for (std::uint32_t h = 0; h < H; ++h) {
            const float* qh = q.data() + (static_cast<std::size_t>(t) * H + h) * qk;

            for (std::uint32_t j = 0; j <= t; ++j) {
                // K is assembled per (head, key) rather than materialised: nope
                // from this head's slice of kv_b's output, rope from the single
                // shared segment.
                const float* k_nope =
                    kv.data() + (static_cast<std::size_t>(j) * H + h) * (nope + vd);
                const float* kpe = k_pe.data() + static_cast<std::size_t>(j) * rope_d;

                float acc = f32::dot(
                    std::span<const float>(qh, nope), std::span<const float>(k_nope, nope), nope);
                acc += f32::dot(std::span<const float>(qh + nope, rope_d),
                                std::span<const float>(kpe, rope_d),
                                rope_d);
                scores[j] = acc * scale;
            }
            f32::softmax(std::span<float>(scores.data(), t + 1), t + 1);

            float* dst = heads.data() + (static_cast<std::size_t>(t) * H + h) * vd;
            for (std::uint32_t j = 0; j <= t; ++j) {
                const float* vv =
                    kv.data() + (static_cast<std::size_t>(j) * H + h) * (nope + vd) + nope;
                f32::axpy(scores[j], std::span<const float>(vv, vd), vd, std::span<float>(dst, vd));
            }
        }
    }

    // The last tap before o_proj. If every tap above matches and this one does
    // not, the fault is in the rope/score/softmax/value math itself rather than
    // in any projection — which is the one region the module-boundary taps
    // cannot subdivide.
    ws.sink(ws.current_layer, "o_proj_in", heads.data(), heads.size());

    soma::matmul(
        w.o_proj, heads, n_tokens, std::span<float>(out, static_cast<std::size_t>(n_tokens) * d));
    return StatusCode::Ok;
}

StatusCode f32_attention_kv(const ArchIr& arch,
                            const soma::F32LayerWeights& lw,
                            const float* x,
                            std::uint32_t n_rows,
                            LayerIndex layer,
                            const KvRow* rows,
                            soma::F32Workspace& ws,
                            float* out) noexcept {
    (void)arch;
    (void)lw;
    (void)x;
    (void)n_rows;
    (void)layer;
    (void)rows;
    (void)ws;
    (void)out;
    // The cached path needs a KvRow whose stride is the LATENT width, not
    // n_kv_heads * head_dim — MLA's cache is a different shape, which is the
    // entire reason the architecture exists. Wiring that is the next step;
    // returning Unsupported keeps the scheduler honest rather than letting it
    // attend over a mis-shaped buffer.
    return StatusCode::Unsupported;
}

StatusCode f32_route(const ArchIr& arch,
                     const soma::F32LayerWeights& lw,
                     const float* logits,
                     std::uint32_t n_tokens,
                     std::uint32_t* out_ids,
                     float* out_weights) noexcept {
    // DeepSeek's own router rather than a borrowed one.
    //
    // Reusing the GQA backend's would work numerically for V2-Lite — softmax
    // scoring, plain top-k — and would silently drop `routed_scaling_factor`,
    // which is 1.0 there and 16.0 on the full V2. A shared router that happens
    // to agree on the small model and diverges on the large one is the worst
    // available outcome, so the two stay separate.
    const auto E = arch.router.n_experts;
    const auto K = arch.router.top_k;
    const float rs =
        (arch.router.routed_scaling_factor > 0.0f) ? arch.router.routed_scaling_factor : 1.0f;

    // V3's `noaux_tc`: sigmoid scoring, a per-expert bias that participates in
    // SELECTION ONLY, and group-limited top-k. V2 is the degenerate case of the
    // same code — softmax, no bias, one group — so there is one router rather
    // than two that must be kept in agreement.
    const bool sigmoid = (arch.router.score_fn == ScoreFn::Sigmoid);
    const auto* bias = (lw.attn.empty() || lw.attn.as<F32AttnWeights>()->e_score_bias.size() != E)
                           ? nullptr
                           : lw.attn.as<F32AttnWeights>()->e_score_bias.data();

    const auto n_grp = std::max<std::uint32_t>(1, arch.router.n_groups);
    const auto topk_grp = std::max<std::uint32_t>(1, arch.router.topk_group);
    const auto per_grp = (n_grp > 0) ? E / n_grp : E;

    std::vector<float> probs(E), sel(E), grp(n_grp);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        std::copy_n(logits + static_cast<std::size_t>(t) * E, E, probs.begin());
        if (sigmoid) {
            for (auto& v : probs)
                v = 1.0f / (1.0f + std::exp(-v));
        } else {
            f32::softmax(probs, E);
        }

        // Selection scores: biased if the model has a bias, otherwise the scores
        // themselves. The WEIGHTS below always come from `probs` — using the
        // biased values there would scale every expert's contribution by a term
        // that exists only to steer load balancing.
        std::copy(probs.begin(), probs.end(), sel.begin());
        if (bias != nullptr) {
            for (std::uint32_t e = 0; e < E; ++e)
                sel[e] += bias[e];
        }

        if (n_grp > 1) {
            // A group's score is the sum of its TOP TWO experts, not its best or
            // its mean. Groups outside the top `topk_group` are masked out
            // entirely before expert selection — a different algorithm from
            // "pick top-k then filter", which would select from the wrong pool.
            for (std::uint32_t g = 0; g < n_grp; ++g) {
                float b0 = -1e30f, b1 = -1e30f;
                for (std::uint32_t i = 0; i < per_grp; ++i) {
                    const float v = sel[g * per_grp + i];
                    if (v > b0) {
                        b1 = b0;
                        b0 = v;
                    } else if (v > b1) {
                        b1 = v;
                    }
                }
                grp[g] = b0 + ((per_grp > 1) ? b1 : 0.0f);
            }
            std::vector<std::uint32_t> gi(n_grp);
            std::vector<float> gv(n_grp);
            f32::top_k(grp, n_grp, topk_grp, gi, gv);

            std::vector<bool> keep(n_grp, false);
            for (std::uint32_t i = 0; i < topk_grp; ++i)
                keep[gi[i]] = true;
            for (std::uint32_t g = 0; g < n_grp; ++g) {
                if (keep[g]) continue;
                for (std::uint32_t i = 0; i < per_grp; ++i)
                    sel[g * per_grp + i] = -1e30f;
            }
        }

        const auto slot = static_cast<std::size_t>(t) * K;
        f32::top_k(sel,
                   E,
                   K,
                   std::span<std::uint32_t>(out_ids + slot, K),
                   std::span<float>(out_weights + slot, K));
        // Re-read the UNBIASED score for each chosen expert.
        for (std::uint32_t s = 0; s < K; ++s) {
            out_weights[slot + s] = probs[out_ids[slot + s]];
        }

        if (arch.router.normalize_topk) {
            float sum = 0.0f;
            for (std::uint32_t s = 0; s < K; ++s)
                sum += out_weights[slot + s];
            if (sum > 0.0f) {
                for (std::uint32_t s = 0; s < K; ++s)
                    out_weights[slot + s] /= sum;
            }
        }
        if (rs != 1.0f) {
            for (std::uint32_t s = 0; s < K; ++s)
                out_weights[slot + s] *= rs;
        }
    }
    return StatusCode::Ok;
}

const soma::F32Backend& f32_backend() noexcept {
    static const soma::F32Backend kBackend{
        "mla", &f32_bind_layer, &f32_attention, &f32_attention_kv, &f32_route};
    return kBackend;
}

} // namespace soma::arch::mla
