// Soma — MLA (Multi-head Latent Attention), F32-activation path.
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

/// The OTHER convention, in the same attention block.
///
/// `rotate_half` above: pairs (i, i + R/2), which is what `apply_rotary_pos_emb`
/// computes as `v*cos + rotate_half(v)*sin` with `cos = cat(freqs, freqs)`. The
/// MLA path a few lines up rotates INTERLEAVED pairs off the identical
/// frequencies, and GLM-5.2's indexer uses this one — the reference documents the
/// split explicitly ("the indexer uses NON-interleaved (half-split) RoPE — unlike
/// the main MLA attention") and it is worth restating here, because the two
/// differ only in which elements pair up. Choosing wrong leaves every norm,
/// projection and score finite and plausible, and reorders the top-k.
///
/// Only the leading `dim` elements rotate; anything past that is the pass-through
/// remainder (index_head_dim 32 vs qk_rope_head_dim 8 on the fixture) and must be
/// left alone.
void rope_half_split(float* v,
                     std::uint32_t dim,
                     std::uint32_t pos,
                     const std::vector<float>& inv_freq) noexcept {
    const auto half = dim / 2;
    for (std::uint32_t i = 0; i < half; ++i) {
        const float angle = static_cast<float>(pos) * inv_freq[i];
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float a = v[i];
        const float b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = b * c + a * s;
    }
}

/// True LayerNorm — mean-centred, with a shift.
///
/// Written out rather than routed through `f32::rmsnorm` because they are not the
/// same function: RMSNorm skips the mean and has no bias. This model uses RMSNorm
/// everywhere EXCEPT here, and `indexer.k_norm.bias` — the only bias in the
/// attention block — is the evidence that the exception is real.
void layernorm(float* v,
               std::uint32_t n,
               std::span<const float> weight,
               std::span<const float> bias) noexcept {
    constexpr float kEps = 1e-6f; ///< nn.LayerNorm(head_dim, eps=1e-6)
    float mean = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i)
        mean += v[i];
    mean /= static_cast<float>(n);
    float var = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i) {
        const float dv = v[i] - mean;
        var += dv * dv;
    }
    var /= static_cast<float>(n);
    const float inv = 1.0f / std::sqrt(var + kEps);
    for (std::uint32_t i = 0; i < n; ++i) {
        v[i] = (v[i] - mean) * inv * weight[i] + bias[i];
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

    // ── DSA indexer, on `full` layers only ───────────────────────────────────
    //
    // Which layers own one is read from the IR's per-layer map rather than
    // re-derived from index_topk_freq + index_skip_topk_offset. Those two are in
    // the config and they do reproduce GLM-5.2's pattern, but a stride plus an
    // offset is a re-derivation waiting to disagree with the weights — the same
    // argument Topology::layer_kinds settled, and the reason DsaSpec carries the
    // map at all.
    //
    // ALL-OR-NOTHING is enforced. A `full` layer missing one of its five tensors
    // is a broken conversion, and the alternative — quietly running it as
    // `shared` — produces a model that loads and attends to the wrong keys.
    const auto& dsa = arch.attention.dsa;
    if (arch.attention.family == AttentionFamily::MlaDsa) {
        if (ctx.layer >= dsa.layer_kinds.size()) return StatusCode::InvalidArgument;
        if (dsa.layer_kinds[ctx.layer] == IndexerKind::Full) {
            using soma::TensorRole;
            if (!soma::bind_layer_weight(
                     ctx, "self_attn.indexer.wq_b.weight", TensorRole::AttnProj, w->idx.wq_b)
                     .ok() ||
                !soma::bind_layer_weight(
                     ctx, "self_attn.indexer.wk.weight", TensorRole::AttnProj, w->idx.wk)
                     .ok() ||
                !soma::bind_layer_weight(ctx,
                                         "self_attn.indexer.weights_proj.weight",
                                         TensorRole::AttnProj,
                                         w->idx.weights_proj)
                     .ok() ||
                !soma::bind_layer_f32(ctx, "self_attn.indexer.k_norm.weight", w->idx.k_norm_w)
                     .ok() ||
                !soma::bind_layer_f32(ctx, "self_attn.indexer.k_norm.bias", w->idx.k_norm_b).ok()) {
                return StatusCode::NotFound;
            }
            w->has_indexer = true;
        }
    }
    return StatusCode::Ok;
}

StatusCode f32_index_select(const ArchIr& arch,
                            const F32AttnWeights& w,
                            const float* x,
                            const float* q_resid,
                            std::uint32_t n_tokens,
                            std::uint32_t pos_base,
                            DsaSelection& out) noexcept {
    if (!w.has_indexer) return StatusCode::InvalidArgument;

    const auto& m = arch.attention.mla;
    const auto& dsa = arch.attention.dsa;
    const auto d = arch.topology.d_model;
    const auto H = dsa.n_index_heads;
    const auto D = dsa.index_head_dim;
    const auto R = m.qk_rope_head_dim; ///< rotated prefix; D - R passes through
    if (H == 0 || D == 0 || R > D) return StatusCode::InvalidArgument;

    const std::span<const float> xs(x, static_cast<std::size_t>(n_tokens) * d);
    const std::span<const float> qr =
        m.q_lora_rank > 0
            ? std::span<const float>(q_resid, static_cast<std::size_t>(n_tokens) * m.q_lora_rank)
            : xs;

    // q from the SHARED residual, k from the hidden states. One k per token: the
    // indexer is single-headed on the key side and mixes per-head on the query
    // side instead, which is where its 2.9x comes from.
    std::vector<float> iq(static_cast<std::size_t>(n_tokens) * H * D);
    std::vector<float> ik(static_cast<std::size_t>(n_tokens) * D);
    std::vector<float> mix(static_cast<std::size_t>(n_tokens) * H);
    soma::matmul(w.idx.wq_b, qr, n_tokens, iq);
    soma::matmul(w.idx.wk, xs, n_tokens, ik);
    soma::matmul(w.idx.weights_proj, xs, n_tokens, mix);

    std::vector<float> inv_freq;
    yarn_inv_freq(arch.attention.rope, R, inv_freq);

    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        // LayerNorm, NOT RMSNorm: mean-centred, unit-variance, then scale AND
        // shift. Substituting the RMSNorm used everywhere else in this model
        // leaves the keys plausible and the ordering wrong, which is invisible in
        // every metric except the selection itself.
        layernorm(ik.data() + static_cast<std::size_t>(t) * D, D, w.idx.k_norm_w, w.idx.k_norm_b);

        // Half-split RoPE — pairs (i, i+R/2) — where the MLA path above rotates
        // INTERLEAVED pairs (2i, 2i+1) off the same frequencies. Both conventions
        // in one attention block, and the reference calls it out precisely
        // because nothing downstream distinguishes them: wrong pairing yields
        // finite, plausible scores and a differently-ordered top-k.
        rope_half_split(ik.data() + static_cast<std::size_t>(t) * D, R, pos_base + t, inv_freq);
        for (std::uint32_t h = 0; h < H; ++h) {
            rope_half_split(
                iq.data() + (static_cast<std::size_t>(t) * H + h) * D, R, pos_base + t, inv_freq);
        }
    }

    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
    const float head_scale = 1.0f / std::sqrt(static_cast<float>(H));
    const auto n_keys = pos_base + n_tokens;
    const auto want =
        dsa.index_topk == 0 ? n_keys : std::min<std::uint32_t>(dsa.index_topk, n_keys);

    out.n_tokens = n_tokens;
    out.stride = want;
    out.keys.assign(static_cast<std::size_t>(n_tokens) * want, 0u);
    out.counts.assign(n_tokens, 0u);
    out.dense_equivalent = true;

    std::vector<float> score;
    std::vector<std::uint32_t> order;
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        const auto avail = pos_base + t + 1; // causal: keys 0..pos_base+t
        score.assign(avail, 0.0f);

        // KEY-OUTER, head-inner. The transposed nesting reads each key ONCE and
        // applies every head to it, where head-outer walks the whole key array
        // once per head — H passes over a buffer that grows with context. On
        // GLM-5.2 at 32k that is 32 passes over 16.8 MB rather than one.
        //
        // Bit-identical, not merely equivalent: `score[j]` still accumulates over
        // h in ascending order, and each `dot` is the same call on the same
        // operands. That matters more here than in most loops — this score feeds
        // a top-k, and a reordering that perturbed the low bits could flip a
        // near-tie and change which keys the model attends to.
        for (std::uint32_t j = 0; j < avail; ++j) {
            const float* kj = ik.data() + static_cast<std::size_t>(j) * D;
            float acc = 0.0f;
            for (std::uint32_t h = 0; h < H; ++h) {
                const float* qh = iq.data() + (static_cast<std::size_t>(t) * H + h) * D;
                const float wgt = mix[static_cast<std::size_t>(t) * H + h] * head_scale;
                const float dot =
                    f32::dot(std::span<const float>(qh, D), std::span<const float>(kj, D), D) *
                    softmax_scale;
                // ReLU BEFORE the head mix, not after. Clamping the mixed score
                // instead would let a negative head cancel a positive one, and
                // the whole point of the non-linearity is that it cannot.
                acc += wgt * (dot > 0.0f ? dot : 0.0f);
            }
            score[j] = acc;
        }

        const auto keep = std::min<std::uint32_t>(want, avail);
        order.resize(avail);
        for (std::uint32_t j = 0; j < avail; ++j)
            order[j] = j;

        // Keep ties deterministic by index. torch.topk does not promise the same
        // tie-break, which is why the oracle preserves the real 32 index heads:
        // shrinking that count manufactured exact-zero ties at the top-k cut and
        // made token-exactness grade an implementation detail rather than DSA.
        std::partial_sort(order.begin(),
                          order.begin() + keep,
                          order.end(),
                          [&](std::uint32_t a, std::uint32_t b) {
                              if (score[a] != score[b]) return score[a] > score[b];
                              return a < b;
                          });

        std::sort(order.begin(), order.begin() + keep);
        std::copy_n(order.begin(), keep, out.keys.begin() + static_cast<std::size_t>(t) * want);
        out.counts[t] = keep;
        if (keep < avail) out.dense_equivalent = false;
    }
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
    // MLA's two LATENT norms do NOT use the model's rms_norm_eps.
    //
    // `arch_ir.hpp` says the epsilon "applies to attention q/k norms, layer norms,
    // and the output norm alike". That is false for this family, in BOTH of its
    // reference implementations:
    //
    //   q_a_layernorm  = RMSNorm(config.q_lora_rank)                  <- default eps
    //   kv_a_layernorm = RMSNorm(config.kv_lora_rank)                 <- default eps
    //   input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    //
    // and the class default is 1e-6. DeepSeek-V2-Lite hid this for the whole life
    // of the backend because its rms_norm_eps IS 1e-6, so the two agreed by
    // coincidence. Moonlight and GLM-5.2 set 1e-5, and Moonlight's conformance
    // error sat at 7.25e-05 — seventy times every other fixture, under the
    // threshold, and unexplained — which is what this line was costing before
    // GLM-5.2 made it fail outright.
    constexpr float kLatentNormEps = 1e-6f;
    const auto eps = kLatentNormEps;

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
    // Hoisted out of the branch below because DSA's indexer reads THIS tensor:
    // its query is `wq_b(q_resid)` off the same normalised residual the main
    // query path uses, not a projection of its own.
    std::vector<float> qa;
    if (m.q_lora_rank == 0) {
        soma::matmul(w.q_proj, xs, n_tokens, q);
    } else {
        qa.assign(static_cast<std::size_t>(n_tokens) * m.q_lora_rank, 0.0f);
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

    // ── DSA: which keys may this query attend to? ────────────────────────────
    //
    // A `full` layer computes the selection and publishes it; the `shared` layers
    // after it reuse that one. IndexShare is not an optimisation applied on top of
    // a working layer — 57 of GLM-5.2's 78 layers own no indexer weights and
    // CANNOT produce a selection, so a missing one is a hard error rather than a
    // reason to fall back to dense.
    const DsaSelection* sel = nullptr;
    if (arch.attention.family == AttentionFamily::MlaDsa) {
        if (w.has_indexer) {
            auto* fresh = new DsaSelection();
            ws.arch_state.adopt(fresh, [](void* p) { delete static_cast<DsaSelection*>(p); });
            if (const auto rc =
                    f32_index_select(arch, w, x, qa.data(), n_tokens, /*pos_base=*/0, *fresh);
                rc != StatusCode::Ok) {
                return rc;
            }
            sel = fresh;
        } else {
            sel = ws.arch_state.as<DsaSelection>();
            // Not a fallback to dense. A shared layer with nothing to share from
            // means the layer map and the weights disagree, and attending to
            // everything would be a different model that still produces text.
            if (sel == nullptr) return StatusCode::InvalidArgument;
        }
        if (sel->n_tokens != n_tokens) return StatusCode::InvalidArgument;
    }

    // ── attention ────────────────────────────────────────────────────────────
    std::vector<float> heads(static_cast<std::size_t>(n_tokens) * H * vd, 0.0f);
    std::vector<float> scores(n_tokens);

    // The dense key list, built once. With no selection every query attends
    // 0..t, which is both the non-DSA path and — when the context is shorter than
    // index_topk — exactly what DSA itself degenerates to.
    std::vector<std::uint32_t> dense_keys(n_tokens);
    for (std::uint32_t j = 0; j < n_tokens; ++j)
        dense_keys[j] = j;

    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        const std::span<const std::uint32_t> keys =
            sel != nullptr ? sel->for_query(t)
                           : std::span<const std::uint32_t>(dense_keys.data(), t + 1);
        const auto n_sel = static_cast<std::uint32_t>(keys.size());
        if (n_sel == 0) return StatusCode::InvalidArgument;

        for (std::uint32_t h = 0; h < H; ++h) {
            const float* qh = q.data() + (static_cast<std::size_t>(t) * H + h) * qk;

            for (std::uint32_t i = 0; i < n_sel; ++i) {
                const auto j = keys[i];
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
                scores[i] = acc * scale;
            }
            // Over the SELECTED keys only. This is the whole of what DSA changes
            // about attention: the unselected keys are not scored down, they are
            // absent from the normalisation — which is why a selection that is
            // merely reordered still produces a different distribution.
            f32::softmax(std::span<float>(scores.data(), n_sel), n_sel);

            float* dst = heads.data() + (static_cast<std::size_t>(t) * H + h) * vd;
            for (std::uint32_t i = 0; i < n_sel; ++i) {
                const float* vv =
                    kv.data() + (static_cast<std::size_t>(keys[i]) * H + h) * (nope + vd) + nope;
                f32::axpy(scores[i], std::span<const float>(vv, vd), vd, std::span<float>(dst, vd));
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

StatusCode f32_index_select_kv(const ArchIr& arch,
                               const F32AttnWeights& w,
                               const float* x,
                               const float* q_resid,
                               std::uint32_t n_rows,
                               LayerIndex layer,
                               const KvRow* rows,
                               DsaSelection& out) noexcept {
    // The indexer's KEYS have to be cached, and noticing that corrected the
    // scoping note on this page.
    //
    // `k = k_norm(wk(hidden))` depends on the hidden state of a past token AT
    // THIS LAYER. At step t those hidden states are gone — that is what a KV
    // cache is for — so the key cannot be recomputed and must have been stored.
    // The scope said DSA adds nothing per-sequence because the INDICES are
    // recomputed every step, which is true and was the wrong noun: the indices
    // are per-step, the keys behind them are per-sequence.
    //
    // They live in the second cache plane, which MLA leaves empty.
    if (!w.has_indexer) return StatusCode::InvalidArgument;

    const auto& m = arch.attention.mla;
    const auto& dsa = arch.attention.dsa;
    const auto d = arch.topology.d_model;
    const auto H = dsa.n_index_heads;
    const auto D = dsa.index_head_dim;
    const auto R = m.qk_rope_head_dim;
    if (H == 0 || D == 0 || R > D) return StatusCode::InvalidArgument;

    const std::span<const float> xs(x, static_cast<std::size_t>(n_rows) * d);
    const std::span<const float> qr =
        m.q_lora_rank > 0
            ? std::span<const float>(q_resid, static_cast<std::size_t>(n_rows) * m.q_lora_rank)
            : xs;

    std::vector<float> iq(static_cast<std::size_t>(n_rows) * H * D);
    std::vector<float> ik(static_cast<std::size_t>(n_rows) * D);
    std::vector<float> mix(static_cast<std::size_t>(n_rows) * H);
    soma::matmul(w.idx.wq_b, qr, n_rows, iq);
    soma::matmul(w.idx.wk, xs, n_rows, ik);
    soma::matmul(w.idx.weights_proj, xs, n_rows, mix);

    std::vector<float> inv_freq;
    yarn_inv_freq(arch.attention.rope, R, inv_freq);

    // This step's key into the cache, before anything reads — same ordering rule
    // as the latent above, and for the same reason.
    for (std::uint32_t r = 0; r < n_rows; ++r) {
        const auto p = rows[r].pos;
        float* slot = rows[r].v_at(layer, p);
        std::copy_n(ik.data() + static_cast<std::size_t>(r) * D, D, slot);
        layernorm(slot, D, w.idx.k_norm_w, w.idx.k_norm_b);
        rope_half_split(slot, R, p, inv_freq);
        for (std::uint32_t h = 0; h < H; ++h) {
            rope_half_split(iq.data() + (static_cast<std::size_t>(r) * H + h) * D, R, p, inv_freq);
        }
    }

    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
    const float head_scale = 1.0f / std::sqrt(static_cast<float>(H));

    std::uint32_t widest = 1;
    for (std::uint32_t r = 0; r < n_rows; ++r) {
        const auto avail = rows[r].len;
        widest = std::max(widest, dsa.index_topk == 0 ? avail : std::min(dsa.index_topk, avail));
    }

    out.n_tokens = n_rows;
    out.stride = widest;
    out.keys.assign(static_cast<std::size_t>(n_rows) * widest, 0u);
    out.counts.assign(n_rows, 0u);
    out.dense_equivalent = true;

    std::vector<float> score;
    std::vector<std::uint32_t> order;
    for (std::uint32_t r = 0; r < n_rows; ++r) {
        const auto avail = rows[r].len;
        score.assign(avail, 0.0f);
        // KEY-OUTER, head-inner — see the prefill path for why, and why it is
        // bit-identical rather than merely equivalent.
        //
        // Parallel over KEYS, which the transposed nesting is what makes possible:
        // each iteration writes exactly one `score[j]` and reads shared,
        // immutable q/weights, so there is nothing to synchronise. Head-outer
        // could not be split this way — every head accumulates into every score.
        //
        // Keys are the right axis for DECODE specifically: `avail` grows with the
        // conversation while `n_rows` is usually 1, so parallelising over rows
        // would leave the work single-threaded exactly when it is largest.
        const float* qrow = iq.data() + static_cast<std::size_t>(r) * H * D;
        const float* wrow = mix.data() + static_cast<std::size_t>(r) * H;
        float* out_score = score.data();
        const auto* row = &rows[r];
        const auto score_keys = [&](std::uint32_t j0, std::uint32_t j1) {
            for (std::uint32_t j = j0; j < j1; ++j) {
                const float* kj = row->v_at(layer, j);
                float acc = 0.0f;
                for (std::uint32_t h = 0; h < H; ++h) {
                    const float dot =
                        f32::dot(std::span<const float>(qrow + static_cast<std::size_t>(h) * D, D),
                                 std::span<const float>(kj, D),
                                 D) *
                        softmax_scale;
                    acc += (wrow[h] * head_scale) * (dot > 0.0f ? dot : 0.0f);
                }
                out_score[j] = acc;
            }
        };

        // Below this, dispatching costs more than it saves.
        //
        // Measured, not assumed, and the first version had no threshold and was
        // SLOWER than the serial loop it replaced at 256 and 512 keys — a
        // "parallel" optimisation that lost to sequential code at the sizes most
        // requests actually use. The scoring is one dot product per (key, head);
        // when there are few keys, the per-chunk handoff dominates it.
        //
        // One implementation, called two ways, so the serial and parallel paths
        // cannot compute different things.
        constexpr std::uint32_t kParallelKeyFloor = 1024;
        if (avail < kParallelKeyFloor) {
            score_keys(0, avail);
        } else {
            ThreadPool::global().parallel_for(
                avail, 256, [&](std::uint32_t j0, std::uint32_t j1, std::uint32_t) {
                    score_keys(j0, j1);
                });
        }

        const auto keep =
            std::min<std::uint32_t>(dsa.index_topk == 0 ? avail : dsa.index_topk, avail);
        order.resize(avail);
        for (std::uint32_t j = 0; j < avail; ++j)
            order[j] = j;
        std::partial_sort(order.begin(),
                          order.begin() + keep,
                          order.end(),
                          [&](std::uint32_t a, std::uint32_t b) {
                              if (score[a] != score[b]) return score[a] > score[b];
                              return a < b;
                          });
        std::sort(order.begin(), order.begin() + keep);
        std::copy_n(order.begin(), keep, out.keys.begin() + static_cast<std::size_t>(r) * widest);
        out.counts[r] = keep;
        if (keep < avail) out.dense_equivalent = false;
    }
    return StatusCode::Ok;
}

KvGeometry f32_kv_geometry(const ArchIr& arch) noexcept {
    const auto& m = arch.attention.mla;
    KvGeometry g;
    g.k_floats = m.kv_lora_rank + m.qk_rope_head_dim;
    // ZERO for plain MLA. V is expanded from the latent on demand, so there has
    // never been anything to store; the plane was allocated because one width
    // described both. DSA is the exception and the only one — its indexer key
    // cannot be recomputed from a later step's hidden state, so it must be kept.
    g.v_floats =
        (arch.attention.family == AttentionFamily::MlaDsa) ? arch.attention.dsa.index_head_dim : 0u;
    return g;
}

StatusCode f32_attention_kv(const ArchIr& arch,
                            const soma::F32LayerWeights& lw,
                            const float* x,
                            std::uint32_t n_rows,
                            LayerIndex layer,
                            const KvRow* rows,
                            soma::F32Workspace& ws,
                            float* out) noexcept {
    // Decode against the compressed cache. The same arithmetic as f32_attention;
    // what differs is entirely where the keys come from and that each row carries
    // its own position and its own visible length.
    if (lw.attn.empty()) return StatusCode::InvalidArgument;
    const auto& w = *lw.attn.as<F32AttnWeights>();

    const auto& m = arch.attention.mla;
    const auto d = arch.topology.d_model;
    const auto H = arch.attention.n_heads;
    const auto nope = m.qk_nope_head_dim;
    const auto rope_d = m.qk_rope_head_dim;
    const auto qk = nope + rope_d;
    const auto vd = m.v_head_dim;
    const auto lora = m.kv_lora_rank;
    const bool is_dsa = arch.attention.family == AttentionFamily::MlaDsa;

    // Deliberately the same constant, and the same reason, as the prefill path:
    // the latent norms take the RMSNorm class default, not the model's
    // rms_norm_eps (roadmap D29).
    constexpr float kLatentNormEps = 1e-6f;

    float scale = 1.0f / std::sqrt(static_cast<float>(qk));
    const auto& rope = arch.attention.rope;
    float cs_scale = 1.0f;
    if (rope.scaling.kind == RopeScalingKind::Yarn) {
        const float num = yarn_mscale(rope.scaling.factor, rope.scaling.mscale);
        const float den = yarn_mscale(rope.scaling.factor, rope.scaling.mscale_all_dim);
        if (den != 0.0f) {
            scale *= num / den;
            cs_scale = num / den;
        }
    }
    std::vector<float> inv_freq;
    yarn_inv_freq(rope, rope_d, inv_freq);

    const std::span<const float> xs(x, static_cast<std::size_t>(n_rows) * d);

    // ── queries, and the residual the indexer shares ─────────────────────────
    std::vector<float> q(static_cast<std::size_t>(n_rows) * H * qk);
    std::vector<float> qa;
    if (m.q_lora_rank == 0) {
        soma::matmul(w.q_proj, xs, n_rows, q);
    } else {
        qa.assign(static_cast<std::size_t>(n_rows) * m.q_lora_rank, 0.0f);
        soma::matmul(w.q_a_proj, xs, n_rows, qa);
        for (std::uint32_t r = 0; r < n_rows; ++r) {
            f32::rmsnorm(std::span<float>(qa).subspan(static_cast<std::size_t>(r) * m.q_lora_rank,
                                                      m.q_lora_rank),
                         w.q_a_norm,
                         m.q_lora_rank,
                         kLatentNormEps);
        }
        soma::matmul(w.q_b_proj, qa, n_rows, q);
    }

    std::vector<float> ckv(static_cast<std::size_t>(n_rows) * (lora + rope_d));
    soma::matmul(w.kv_a_proj, xs, n_rows, ckv);

    // ── write this step's latent into every row's own cache ──────────────────
    //
    // Before any row READS, so a row attending over its own position sees the
    // value it just produced.
    for (std::uint32_t r = 0; r < n_rows; ++r) {
        const auto p = rows[r].pos;
        float* slot = rows[r].k_at(layer, p);
        float* src = ckv.data() + static_cast<std::size_t>(r) * (lora + rope_d);

        std::copy_n(src, lora, slot);
        f32::rmsnorm(std::span<float>(slot, lora), w.kv_a_norm, lora, kLatentNormEps);
        std::copy_n(src + lora, rope_d, slot + lora);
        rope_at(slot + lora, rope_d, p, inv_freq, cs_scale);

        for (std::uint32_t h = 0; h < H; ++h) {
            rope_at(q.data() + (static_cast<std::size_t>(r) * H + h) * qk + nope,
                    rope_d,
                    p,
                    inv_freq,
                    cs_scale);
        }
    }

    // ── DSA: this step's key selection, per row ──────────────────────────────
    const DsaSelection* sel = nullptr;
    if (is_dsa) {
        if (w.has_indexer) {
            auto* fresh = new DsaSelection();
            ws.arch_state.adopt(fresh, [](void* p) { delete static_cast<DsaSelection*>(p); });
            if (const auto rc =
                    f32_index_select_kv(arch, w, x, qa.data(), n_rows, layer, rows, *fresh);
                rc != StatusCode::Ok) {
                return rc;
            }
            sel = fresh;
        } else {
            sel = ws.arch_state.as<DsaSelection>();
            if (sel == nullptr) return StatusCode::InvalidArgument;
        }
        if (sel->n_tokens != n_rows) return StatusCode::InvalidArgument;
    }

    // ── attention ────────────────────────────────────────────────────────────
    //
    // The cache holds latents, so the keys and values this needs do not exist
    // until kv_b expands them. Expanding ONLY the keys this row will actually
    // attend to is where DSA pays for itself here: at GLM-5.2's index_topk of
    // 2048 against a 32k context that is a sixteenth of the work, and the saving
    // grows with context rather than shrinking.
    //
    // ABSORPTION, which is what `MlaSpec::absorb_weights` has always described and
    // nothing implemented. The identity is two lines:
    //
    //   q_nope . (W_k c_j)        =  (W_k^T q_nope) . c_j
    //   sum_j a_j (W_v c_j)       =  W_v (sum_j a_j c_j)
    //
    // Both move kv_b to the side of the expression that does NOT depend on j. The
    // unabsorbed form expands every attended latent into per-head K and V — the
    // full-size K and V that MLA exists to avoid storing — and then throws them
    // away. Absorbed, kv_b is touched once per head per step regardless of how
    // many keys are attended, and everything inside the j loop is a `lora`-wide
    // dot or axpy.
    //
    // Per layer per step, GLM-5.2 at index_topk 2048:
    //   unabsorbed  n_sel * lora * H * (nope+vd)  = 3.0e10
    //   absorbed    H*lora*(nope+vd) + 2*n_sel*H*lora = 1.5e8      (~200x)
    //
    // `absorb_weights` selects between them, which is exactly the A/B the flag was
    // documented for. The unabsorbed path STAYS — it is the reference the fast one
    // is checked against, and `soma_decode_kv_g4` runs both.
    const bool absorb = m.absorb_weights;

    std::vector<float> heads(static_cast<std::size_t>(n_rows) * H * vd, 0.0f);
    std::vector<float> latents;
    std::vector<float> kv;
    std::vector<float> scores;
    std::vector<std::uint32_t> keys;

    // Absorbed-only scratch.
    const auto kvb_row = nope + vd; ///< kv_b rows belonging to one head
    std::vector<float> wk_head;     ///< [nope, lora], this head's K up-projection
    std::vector<float> q_lat;       ///< [lora], W_k^T q_nope
    std::vector<float> z_lat;       ///< [lora], sum_j a_j c_j
    if (absorb) {
        wk_head.assign(static_cast<std::size_t>(nope) * lora, 0.0f);
        q_lat.assign(lora, 0.0f);
        z_lat.assign(lora, 0.0f);
        if (w.kv_b_proj.rows != H * kvb_row || w.kv_b_proj.cols != lora) {
            return StatusCode::InvalidArgument;
        }
    }

    for (std::uint32_t r = 0; r < n_rows; ++r) {
        const auto& kvr = rows[r];
        if (kvr.len == 0) return StatusCode::InvalidArgument;

        keys.clear();
        if (sel != nullptr) {
            const auto picked = sel->for_query(r);
            keys.assign(picked.begin(), picked.end());
        } else {
            keys.resize(kvr.len);
            for (std::uint32_t j = 0; j < kvr.len; ++j)
                keys[j] = j;
        }
        const auto n_sel = static_cast<std::uint32_t>(keys.size());
        if (n_sel == 0) return StatusCode::InvalidArgument;

        if (!absorb) {
            latents.resize(static_cast<std::size_t>(n_sel) * lora);
            for (std::uint32_t i = 0; i < n_sel; ++i) {
                std::copy_n(kvr.k_at(layer, keys[i]),
                            lora,
                            latents.data() + static_cast<std::size_t>(i) * lora);
            }
            kv.assign(static_cast<std::size_t>(n_sel) * H * kvb_row, 0.0f);
            soma::matmul(w.kv_b_proj, latents, n_sel, kv);
        }

        scores.resize(n_sel);
        for (std::uint32_t h = 0; h < H; ++h) {
            const float* qh = q.data() + (static_cast<std::size_t>(r) * H + h) * qk;

            if (absorb) {
                // W_k^T q_nope, as a row-scaled accumulation. The transpose is why
                // this needs the weight materialized rather than fused: a kernel
                // walks ROWS, and the sum here runs down COLUMNS. Once per head
                // per step, against a j loop that may be thousands long.
                const auto k_rows = soma::row_block(w.kv_b_proj, h * kvb_row, nope);
                if (k_rows.empty()) return StatusCode::InvalidArgument;
                if (const auto st = soma::dequantize(k_rows, wk_head); !st.ok()) {
                    return st.code();
                }
                std::fill(q_lat.begin(), q_lat.end(), 0.0f);
                for (std::uint32_t i = 0; i < nope; ++i) {
                    f32::axpy(qh[i],
                              std::span<const float>(
                                  wk_head.data() + static_cast<std::size_t>(i) * lora, lora),
                              lora,
                              std::span<float>(q_lat));
                }
            }

            for (std::uint32_t i = 0; i < n_sel; ++i) {
                const float* c_j = kvr.k_at(layer, keys[i]);
                const float* kpe = c_j + lora;
                float acc;
                if (absorb) {
                    acc = f32::dot(
                        std::span<const float>(q_lat), std::span<const float>(c_j, lora), lora);
                } else {
                    const float* k_nope =
                        kv.data() + (static_cast<std::size_t>(i) * H + h) * kvb_row;
                    acc = f32::dot(std::span<const float>(qh, nope),
                                   std::span<const float>(k_nope, nope),
                                   nope);
                }
                acc += f32::dot(std::span<const float>(qh + nope, rope_d),
                                std::span<const float>(kpe, rope_d),
                                rope_d);
                scores[i] = acc * scale;
            }
            f32::softmax(std::span<float>(scores.data(), n_sel), n_sel);

            float* dst = heads.data() + (static_cast<std::size_t>(r) * H + h) * vd;
            if (absorb) {
                // Accumulate in LATENT space, then project once. The unabsorbed
                // loop below does `n_sel` axpys of width vd against expanded
                // values; this does `n_sel` of width lora against cached ones and
                // one matvec at the end.
                std::fill(z_lat.begin(), z_lat.end(), 0.0f);
                for (std::uint32_t i = 0; i < n_sel; ++i) {
                    f32::axpy(scores[i],
                              std::span<const float>(kvr.k_at(layer, keys[i]), lora),
                              lora,
                              std::span<float>(z_lat));
                }
                const auto v_rows = soma::row_block(w.kv_b_proj, h * kvb_row + nope, vd);
                if (v_rows.empty()) return StatusCode::InvalidArgument;
                soma::matvec(v_rows, std::span<const float>(z_lat), std::span<float>(dst, vd));
            } else {
                for (std::uint32_t i = 0; i < n_sel; ++i) {
                    const float* vv =
                        kv.data() + (static_cast<std::size_t>(i) * H + h) * kvb_row + nope;
                    f32::axpy(
                        scores[i], std::span<const float>(vv, vd), vd, std::span<float>(dst, vd));
                }
            }
        }
    }

    soma::matmul(
        w.o_proj, heads, n_rows, std::span<float>(out, static_cast<std::size_t>(n_rows) * d));
    return StatusCode::Ok;
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
    static const soma::F32Backend kBackend = [] {
        soma::F32Backend b{};
        b.name = "mla";
        b.bind_layer = &f32_bind_layer;
        b.attention = &f32_attention;
        b.kv_geometry = &f32_kv_geometry;
        b.attention_kv = &f32_attention_kv;
        b.route = &f32_route;
        return b;
    }();
    return kBackend;
}

// ── sizing ───────────────────────────────────────────────────────────────────
//
// `attention_backend()` was DECLARED in mla.hpp and never defined, so
// `resolve_attention_backend` had to return nullptr for this family and the
// planner sized MLA with GQA's formula. Two consequences, both silent:
// kv_bytes_at_ctx came out ZERO for every MLA model, and the resident half was
// inflated by the K/V projections MLA does not have.

std::size_t kv_bytes_per_token(const ArchIr& arch) noexcept {
    // The whole reason MLA exists. The cache holds a `kv_lora_rank`-wide latent
    // plus ONE shared RoPE segment per token per layer — not per-head K and V.
    // Against GQA's `2 * n_kv_heads * head_dim` that is roughly two orders of
    // magnitude less on comparable configs, which is why it can afford a long
    // context beside a big expert cache.
    //
    // fp32, matching the GQA implementation and the cache itself. There is no
    // configured KV dtype for the planner to scale by — see the note in
    // gqa.cpp's kv_bytes_per_token (D45).
    //
    // BOTH PLANES, each at its own width — which for plain MLA means the K plane
    // and nothing else.
    //
    // This counted one plane at a time when two were always allocated (2x under,
    // in the optimistic direction, like D26), then two when the second was often
    // empty. Asking the geometry removes the arithmetic entirely: there is one
    // description of what the cache holds and both the allocation and this
    // estimate read it.
    //
    // GLM-5.2 at 4096 x 4 slots: the V plane falls from 2.94 GB to 653 MB, and for
    // DeepSeek-V2-Lite and Moonlight it disappears.
    const auto g = f32_kv_geometry(arch);
    const std::size_t per_layer =
        (static_cast<std::size_t>(g.k_floats) + g.v_floats) * sizeof(float);
    return per_layer * arch.topology.n_layers;
}

std::uint64_t weight_bytes_per_layer(const ArchIr& arch,
                                     AttentionBackend::ByteSizer sizer) noexcept {
    // Shapes verified against tests/fixtures/tiny/DeepSeek-V2-Lite, which
    // carries the real tensors: q_proj [96,64], kv_a_proj_with_mqa [40,64],
    // kv_b_proj [128,32], o_proj [64,64] at d=64, n_heads=4, kv_lora=32,
    // qk_nope=16, qk_rope=8, v_head=16.
    const auto d = arch.topology.d_model;
    const auto& a = arch.attention;
    const auto& m = a.mla;
    const auto qk = m.qk_nope_head_dim + m.qk_rope_head_dim;

    std::uint64_t bytes = 0;

    // q_lora_rank is absent (0) on V2-Lite, which projects Q directly; the full
    // V2 and GLM-5.2 compress it through two matrices. Both shapes are real and
    // the difference is large, so it branches rather than assuming either.
    if (m.q_lora_rank > 0) {
        bytes += sizer(arch, m.q_lora_rank, d, TensorRole::AttnProj);
        bytes += sizer(arch, a.n_heads * qk, m.q_lora_rank, TensorRole::AttnProj);
        bytes += static_cast<std::uint64_t>(m.q_lora_rank) * sizeof(float); // q_a_layernorm
    } else {
        bytes += sizer(arch, a.n_heads * qk, d, TensorRole::AttnProj);
    }

    // One down-projection carrying the latent plus the shared RoPE slice, and
    // one up-projection back to per-head nope ++ value. There is no per-head K
    // or V projection anywhere in this family, which is the entire point of it
    // and precisely what the GQA formula used to invent.
    bytes += sizer(arch, m.kv_lora_rank + m.qk_rope_head_dim, d, TensorRole::AttnProj);
    bytes += sizer(arch,
                   a.n_heads * (m.qk_nope_head_dim + m.v_head_dim),
                   m.kv_lora_rank,
                   TensorRole::AttnProj);
    bytes += static_cast<std::uint64_t>(m.kv_lora_rank) * sizeof(float); // kv_a_layernorm

    // o_proj is sized on v_head_dim, NOT head_dim: a query head is nope ++ rope
    // and the value head is a different width entirely. They coincide on
    // GLM-5.2 (both 256) and differ on V2-Lite (16 vs 24), so assuming either
    // is right half the time.
    bytes += sizer(arch, d, a.n_heads * m.v_head_dim, TensorRole::AttnProj);

    // ── DSA's indexer, AMORTISED over the stack ──────────────────────────────
    //
    // Only `Full` layers carry one — 21 of GLM-5.2's 78 — and this function
    // returns a PER-LAYER figure that the planner multiplies by n_layers. So the
    // honest thing it can express is the average, which gives the correct TOTAL
    // even though no individual layer costs exactly this.
    //
    // Without it the resident half came out at 0.90x of the real tensors: an
    // under-count, so the model looked more servable than it is, which is the
    // worse direction (roadmap D22). Shapes read from the committed fixture
    // rather than inferred — at hidden 64, q_lora 48, 4 index heads of 32:
    // wq_b (128, 48), wk (32, 64), weights_proj (4, 64), k_norm (32,) twice.
    const auto& dsa = a.dsa;
    const auto n_layers = static_cast<std::uint64_t>(arch.topology.n_layers);
    const auto n_full = static_cast<std::uint64_t>(dsa.n_full_layers());
    if (n_full > 0 && n_layers > 0 && dsa.index_head_dim > 0) {
        const std::uint64_t heads_dim =
            static_cast<std::uint64_t>(dsa.n_index_heads) * dsa.index_head_dim;
        std::uint64_t idx = 0;
        // wq_b projects the Q latent when there is one, the hidden state when not.
        idx += sizer(arch,
                     static_cast<std::uint32_t>(heads_dim),
                     m.q_lora_rank > 0 ? m.q_lora_rank : d,
                     TensorRole::AttnProj);
        // ONE shared K across index heads, MQA-style — not one per head.
        idx += sizer(arch, dsa.index_head_dim, d, TensorRole::AttnProj);
        idx += sizer(arch, dsa.n_index_heads, d, TensorRole::AttnProj);
        // k_norm weight AND bias: rank-1, always f32, and the only bias in this
        // attention block.
        idx += 2ull * dsa.index_head_dim * sizeof(float);

        bytes += (idx * n_full + n_layers - 1) / n_layers; // round up
    }
    return bytes;
}

const AttentionBackend& attention_backend() noexcept {
    static const AttentionBackend kBackend = [] {
        AttentionBackend b{};
        b.name = "mla";
        b.family = AttentionFamily::Mla;
        b.persist_format_id = kKvFormat;
        b.kv_bytes_per_token = &kv_bytes_per_token;
        b.weight_bytes_per_layer = &weight_bytes_per_layer;
        // Execution members stay null. This instance exists for the planner's
        // DESCRIPTION questions; whether MLA can actually be run is answered by
        // resolve_f32_backend, which has its own opinion and its own reasons.
        return b;
    }();
    return kBackend;
}

} // namespace soma::arch::mla
