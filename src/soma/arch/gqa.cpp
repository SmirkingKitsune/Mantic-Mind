// Soma — GQA/MHA attention and softmax-family routing, fp32 reference path.
//
// One of only two translation units permitted to name an architecture (the
// other is arch_registry.cpp). tools/ci/check_seam.py enforces that.
//
// Designed against two real configs whose differences are NOT visible in a
// boolean flag:
//
//   OLMoE-1B-7B     16/16 MHA, q_norm [n_heads*head_dim] over the whole
//                   projection, norm_topk_prob false
//   Qwen3-30B-A3B   32/4 GQA, head_dim independent of hidden_size,
//                   q_norm [head_dim] applied per head, norm_topk_prob true
//
// MHA is handled here as the n_kv_heads == n_heads case rather than as its own
// backend, because the only thing that changes is the repeat factor.

#include "soma/arch/gqa.hpp"

#include "soma/f32_model.hpp"
#include "soma/kernels_f32.hpp"
#include "soma/threading.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace soma::arch::gqa {

namespace {

/// Q/K normalization, in whichever of the two forms this family uses.
///
/// PerHead normalizes each head's head_dim slice independently against a
/// head_dim-wide weight; FullWidth normalizes the entire n_heads*head_dim
/// projection against a weight of that size. Applying one where the other is
/// meant produces finite, plausible logits that are simply wrong — which is why
/// the IR carries QkNormKind rather than a bool, and why the loader
/// cross-checks the tensor's actual length against it.
void apply_qk_norm(QkNormKind kind,
                   std::span<float> vec,
                   std::span<const float> weight,
                   std::uint32_t n_heads,
                   std::uint32_t head_dim,
                   float eps) noexcept {
    if (kind == QkNormKind::None || weight.empty()) return;

    if (kind == QkNormKind::FullWidth) {
        f32::rmsnorm(vec, weight, n_heads * head_dim, eps);
        return;
    }
    for (std::uint32_t h = 0; h < n_heads; ++h) {
        f32::rmsnorm(
            vec.subspan(static_cast<std::size_t>(h) * head_dim, head_dim), weight, head_dim, eps);
    }
}

} // namespace

StatusCode f32_bind_layer(const ArchIr& arch,
                          const soma::LayerBindCtx& ctx,
                          soma::ArchLayerPayload& out) noexcept {
    auto* w = new F32AttnWeights();
    out.adopt(w, [](void* p) { delete static_cast<F32AttnWeights*>(p); });

    using soma::TensorRole;
    if (!soma::bind_layer_weight(ctx, "self_attn.q_proj.weight", TensorRole::AttnProj, w->q_proj)
             .ok() ||
        !soma::bind_layer_weight(ctx, "self_attn.k_proj.weight", TensorRole::AttnProj, w->k_proj)
             .ok() ||
        !soma::bind_layer_weight(ctx, "self_attn.v_proj.weight", TensorRole::AttnProj, w->v_proj)
             .ok() ||
        !soma::bind_layer_weight(ctx, "self_attn.o_proj.weight", TensorRole::AttnProj, w->o_proj)
             .ok()) {
        return StatusCode::NotFound;
    }

    if (arch.attention.qk_norm != QkNormKind::None) {
        if (!soma::bind_layer_f32(ctx, "self_attn.q_norm.weight", w->q_norm).ok() ||
            !soma::bind_layer_f32(ctx, "self_attn.k_norm.weight", w->k_norm).ok()) {
            return StatusCode::NotFound;
        }
        // The IR says which form this family uses; the tensor says what is
        // actually there. Disagreement means the adapter's family table is
        // wrong, and that is worth failing loudly at load rather than producing
        // plausible logits.
        const auto want = (arch.attention.qk_norm == QkNormKind::PerHead)
                              ? arch.attention.head_dim
                              : arch.attention.n_heads * arch.attention.head_dim;
        if (w->q_norm.size() != want) return StatusCode::InvalidArgument;
    }
    return StatusCode::Ok;
}

StatusCode f32_attention_kv(const ArchIr& arch,
                            const soma::F32LayerWeights& w,
                            const float* x,
                            std::uint32_t n_rows,
                            LayerIndex layer,
                            const KvRow* rows,
                            soma::F32Workspace& ws,
                            float* out) noexcept {
    // The batched-decode path. Same arithmetic as f32_attention; the difference
    // is entirely in WHERE the keys and values come from.
    //
    // Each row carries its own cache, its own position and its own visible
    // length, so rows from different sequences — at different positions, with
    // different history — batch together with no special cases. That is what
    // makes "decode rows and prefill rows are just rows" true rather than
    // aspirational.
    const auto d = arch.topology.d_model;
    const auto H = arch.attention.n_heads;
    const auto KV = arch.attention.n_kv_heads;
    const auto hd = arch.attention.head_dim;
    const auto hq = H * hd;
    const auto hkv = KV * hd;
    const auto group = H / KV;
    const auto eps = arch.rms_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // The payload this backend stored at load. Only this file knows the type.
    if (w.attn.empty()) return StatusCode::InvalidArgument;
    const auto& aw = *w.attn.as<F32AttnWeights>();

    const std::span<const float> xs(x, static_cast<std::size_t>(n_rows) * d);
    soma::matmul(aw.q_proj, xs, n_rows, ws.q);
    soma::matmul(aw.k_proj, xs, n_rows, ws.k);
    soma::matmul(aw.v_proj, xs, n_rows, ws.v);

    for (std::uint32_t r = 0; r < n_rows; ++r) {
        apply_qk_norm(arch.attention.qk_norm,
                      std::span<float>(ws.q).subspan(static_cast<std::size_t>(r) * hq, hq),
                      aw.q_norm,
                      H,
                      hd,
                      eps);
        apply_qk_norm(arch.attention.qk_norm,
                      std::span<float>(ws.k).subspan(static_cast<std::size_t>(r) * hkv, hkv),
                      aw.k_norm,
                      KV,
                      hd,
                      eps);

        const auto& rope = arch.attention.rope;
        const auto rotary = (rope.partial_dim > 0) ? rope.partial_dim : hd;
        auto q_r = std::span<float>(ws.q).subspan(static_cast<std::size_t>(r) * hq, hq);
        auto k_r = std::span<float>(ws.k).subspan(static_cast<std::size_t>(r) * hkv, hkv);
        // RoPE at the row's ABSOLUTE position, not its index in the batch. Using
        // the batch index would rotate every row as if it were at position 0..n,
        // which produces finite, plausible, wrong output.
        const auto p = rows[r].pos;
        if (rope.interleaved) {
            f32::rope_interleaved(q_r, H, hd, p, rope.theta, rotary);
            f32::rope_interleaved(k_r, KV, hd, p, rope.theta, rotary);
        } else {
            f32::rope_neox(q_r, H, hd, p, rope.theta, rotary);
            f32::rope_neox(k_r, KV, hd, p, rope.theta, rotary);
        }

        // Append this row's K/V to its own cache before any row reads, so a row
        // that attends over its own position sees the value it just produced.
        std::copy_n(k_r.data(), hkv, rows[r].k_at(layer, p));
        std::copy_n(ws.v.data() + static_cast<std::size_t>(r) * hkv, hkv, rows[r].v_at(layer, p));
    }

    const auto window = arch.attention.sliding_window;
    const std::uint32_t n_workers = ThreadPool::global().size();
    std::uint32_t longest = 1;
    for (std::uint32_t r = 0; r < n_rows; ++r)
        longest = std::max(longest, rows[r].len);
    ws.ensure_score_scratch(n_workers, longest);

    ThreadPool::global().parallel_for(
        n_rows, 1, [&](std::uint32_t r_begin, std::uint32_t r_end, std::uint32_t worker) {
            float* scores = ws.worker_scores(worker, longest);
            for (std::uint32_t r = r_begin; r < r_end; ++r) {
                const auto& kv = rows[r];
                const std::uint32_t hi = kv.len; // exclusive
                const std::uint32_t lo = (window > 0 && hi > window) ? (hi - window) : 0u;

                for (std::uint32_t h = 0; h < H; ++h) {
                    const std::uint32_t kvh = h / group;
                    const float* qv = ws.q.data() + static_cast<std::size_t>(r) * hq +
                                      static_cast<std::size_t>(h) * hd;

                    for (std::uint32_t j = lo; j < hi; ++j) {
                        const float* kvec = kv.k_at(layer, j) + kvh * hd;
                        scores[j - lo] = f32::dot(std::span<const float>(qv, hd),
                                                  std::span<const float>(kvec, hd),
                                                  hd) *
                                         scale;
                    }
                    const std::uint32_t span_len = hi - lo;
                    f32::softmax(std::span<float>(scores, span_len), span_len);

                    float* dst = ws.attn_heads.data() + static_cast<std::size_t>(r) * hq +
                                 static_cast<std::size_t>(h) * hd;
                    std::fill_n(dst, hd, 0.0f);
                    for (std::uint32_t j = lo; j < hi; ++j) {
                        const float* vvec = kv.v_at(layer, j) + kvh * hd;
                        f32::axpy(scores[j - lo],
                                  std::span<const float>(vvec, hd),
                                  hd,
                                  std::span<float>(dst, hd));
                    }
                }
            }
        });

    soma::matmul(aw.o_proj,
                 ws.attn_heads,
                 n_rows,
                 std::span<float>(out, static_cast<std::size_t>(n_rows) * d));
    return StatusCode::Ok;
}

StatusCode f32_attention(const ArchIr& arch,
                         const soma::F32LayerWeights& w,
                         const float* x,
                         std::uint32_t n_tokens,
                         soma::F32Workspace& ws,
                         float* out) noexcept {
    const auto d = arch.topology.d_model;
    const auto H = arch.attention.n_heads;
    const auto KV = arch.attention.n_kv_heads;
    const auto hd = arch.attention.head_dim;
    const auto hq = H * hd;
    const auto hkv = KV * hd;
    const auto group = H / KV; // 1 for MHA
    const auto eps = arch.rms_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    if (w.attn.empty()) return StatusCode::InvalidArgument;
    const auto& aw = *w.attn.as<F32AttnWeights>();

    const std::span<const float> xs(x, static_cast<std::size_t>(n_tokens) * d);
    soma::matmul(aw.q_proj, xs, n_tokens, ws.q);
    soma::matmul(aw.k_proj, xs, n_tokens, ws.k);
    soma::matmul(aw.v_proj, xs, n_tokens, ws.v);

    // Norm before RoPE, matching HF for both reference families.
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        apply_qk_norm(arch.attention.qk_norm,
                      std::span<float>(ws.q).subspan(static_cast<std::size_t>(t) * hq, hq),
                      aw.q_norm,
                      H,
                      hd,
                      eps);
        apply_qk_norm(arch.attention.qk_norm,
                      std::span<float>(ws.k).subspan(static_cast<std::size_t>(t) * hkv, hkv),
                      aw.k_norm,
                      KV,
                      hd,
                      eps);

        const auto& rope = arch.attention.rope;
        const auto rotary = (rope.partial_dim > 0) ? rope.partial_dim : hd;
        auto q_t = std::span<float>(ws.q).subspan(static_cast<std::size_t>(t) * hq, hq);
        auto k_t = std::span<float>(ws.k).subspan(static_cast<std::size_t>(t) * hkv, hkv);
        if (rope.interleaved) {
            f32::rope_interleaved(q_t, H, hd, t, rope.theta, rotary);
            f32::rope_interleaved(k_t, KV, hd, t, rope.theta, rotary);
        } else {
            f32::rope_neox(q_t, H, hd, t, rope.theta, rotary);
            f32::rope_neox(k_t, KV, hd, t, rope.theta, rotary);
        }
    }

    const auto window = arch.attention.sliding_window;

    // Parallel over QUERY POSITIONS. Each t writes its own slice of attn_heads
    // and reads only q/k/v, so the split is bit-identical to serial — the same
    // property that lets matvec split by output row.
    //
    // Deliberately NOT parallel over j (the key axis): that is the reduction, and
    // splitting it would make the score sum depend on the core count.
    //
    // The chunking is uneven on purpose. Row t attends over t+1 keys, so cost
    // grows linearly along the range and an equal split would leave the thread
    // holding the last rows running while the rest idle. parallel_for hands out
    // several small chunks per worker rather than one big one, which turns the
    // ragged tail into ordinary load balancing.
    const std::uint32_t n_workers = ThreadPool::global().size();
    ws.ensure_score_scratch(n_workers, n_tokens);

    ThreadPool::global().parallel_for(
        n_tokens, 1, [&](std::uint32_t t_begin, std::uint32_t t_end, std::uint32_t worker) {
            float* scores = ws.worker_scores(worker, n_tokens);
            for (std::uint32_t t = t_begin; t < t_end; ++t) {
                for (std::uint32_t h = 0; h < H; ++h) {
                    const std::uint32_t kvh = h / group;
                    const float* qv = ws.q.data() + static_cast<std::size_t>(t) * hq +
                                      static_cast<std::size_t>(h) * hd;

                    // Causal, optionally windowed. `lo` is the first visible position.
                    const std::uint32_t lo = (window > 0 && t + 1 > window) ? (t + 1 - window) : 0u;

                    for (std::uint32_t j = lo; j <= t; ++j) {
                        const float* kv = ws.k.data() + static_cast<std::size_t>(j) * hkv +
                                          static_cast<std::size_t>(kvh) * hd;
                        scores[j - lo] = f32::dot(std::span<const float>(qv, hd),
                                                  std::span<const float>(kv, hd),
                                                  hd) *
                                         scale;
                    }
                    const std::uint32_t span_len = t - lo + 1;
                    f32::softmax(std::span<float>(scores, span_len), span_len);

                    float* dst = ws.attn_heads.data() + static_cast<std::size_t>(t) * hq +
                                 static_cast<std::size_t>(h) * hd;
                    std::fill_n(dst, hd, 0.0f);
                    for (std::uint32_t j = lo; j <= t; ++j) {
                        const float p = scores[j - lo];
                        const float* vv = ws.v.data() + static_cast<std::size_t>(j) * hkv +
                                          static_cast<std::size_t>(kvh) * hd;
                        // Runs once per (query, key, head), so it is O(T^2 * heads
                        // * head_dim) — the same order as the score dot above, and
                        // together they dominate the whole forward at long context.
                        // Both go through soma::f32 rather than a hand-written loop
                        // here so the SIMD dispatch happens in one place.
                        f32::axpy(p, std::span<const float>(vv, hd), hd, std::span<float>(dst, hd));
                    }
                }
            }
        });

    soma::matmul(aw.o_proj,
                 ws.attn_heads,
                 n_tokens,
                 std::span<float>(out, static_cast<std::size_t>(n_tokens) * d));
    return StatusCode::Ok;
}

StatusCode f32_route(const ArchIr& arch,
                     const soma::F32LayerWeights& lw,
                     const float* logits,
                     std::uint32_t n_tokens,
                     std::uint32_t* out_ids,
                     float* out_weights) noexcept {
    // Unused here on purpose: this family's router has no parameters beyond the
    // gate matrix, which the caller has already applied. The argument exists
    // because DeepSeek-V3's does.
    (void)lw;
    const auto E = arch.router.n_experts;
    const auto K = arch.router.top_k;

    // Scored over ALL experts first, then top-k — not top-k on raw logits.
    // The order is not interchangeable: softmax is monotonic so the SELECTION
    // matches either way, but the retained WEIGHTS do not, and the difference
    // silently rescales every expert contribution.
    static thread_local std::vector<float> probs;
    probs.resize(E);

    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        const float* row = logits + static_cast<std::size_t>(t) * E;
        std::copy_n(row, E, probs.begin());

        if (arch.router.score_fn == ScoreFn::Sigmoid) {
            for (std::uint32_t e = 0; e < E; ++e) {
                probs[e] = 1.0f / (1.0f + std::exp(-probs[e]));
            }
        } else {
            f32::softmax(probs, E);
        }

        const auto slot = static_cast<std::size_t>(t) * K;
        f32::top_k(probs,
                   E,
                   K,
                   std::span<std::uint32_t>(out_ids + slot, K),
                   std::span<float>(out_weights + slot, K));

        if (arch.router.normalize_topk) {
            float sum = 0.0f;
            for (std::uint32_t s = 0; s < K; ++s)
                sum += out_weights[slot + s];
            // HF divides without an epsilon guard; matching that exactly matters
            // more than defending against a sum this path cannot produce.
            if (sum != 0.0f) {
                for (std::uint32_t s = 0; s < K; ++s)
                    out_weights[slot + s] /= sum;
            }
        }
        if (arch.router.routed_scaling_factor != 1.0f) {
            for (std::uint32_t s = 0; s < K; ++s) {
                out_weights[slot + s] *= arch.router.routed_scaling_factor;
            }
        }
    }
    return StatusCode::Ok;
}

const soma::F32Backend& f32_backend() noexcept {
    // Named rather than positional. Adding `kv_floats_per_layer` to F32Backend
    // shifted every later member, and the compiler caught it only because the
    // types happened to disagree — an aggregate initialiser that still lined up
    // by type would have bound the wrong pointers in silence.
    //
    // `kv_floats_per_layer` is deliberately left null: null means the GQA default,
    // and this IS the GQA backend, so stating it would be a second copy of the
    // same formula waiting to disagree with the first.
    static const soma::F32Backend kBackend = [] {
        soma::F32Backend b{};
        b.name = "gqa";
        b.bind_layer = &f32_bind_layer;
        b.attention = &f32_attention;
        b.attention_kv = &f32_attention_kv;
        b.route = &f32_route;
        return b;
    }();
    return kBackend;
}

// ── Descriptors ──────────────────────────────────────────────────────────────
//
// The streaming-era hot-path members (prefill/decode/apply_expert/...) arrive
// with the scheduler at G3 and are null here. The members that are meaningful
// TODAY are real: kv_bytes_per_token feeds the planner, validate is the
// admission gate, prepare_weights is genuinely a no-op for this family.
//
// Left null rather than stubbed-with-Ok so that calling one before G3 fails
// visibly instead of silently returning success.

std::size_t kv_bytes_per_token(const ArchIr& arch) noexcept {
    // Full K and V per token, per layer: 2 * n_kv_heads * head_dim.
    //
    // fp32 here because that is the G0 cache dtype; the planner scales this by
    // the configured KV dtype. Worth stating because GQA's KV competes directly
    // with the expert cache for RAM (docs/architecture.md §2.2).
    const std::size_t per_layer =
        2ull * arch.attention.n_kv_heads * arch.attention.head_dim * sizeof(float);
    return per_layer * arch.topology.n_layers;
}

std::uint64_t weight_bytes_per_layer(const ArchIr& arch,
                                     AttentionBackend::ByteSizer sizer) noexcept {
    // q + k + v + o. MHA is the n_kv_heads == n_heads case of this, which is why
    // one function serves both and the collapse happens in the adapter.
    const auto d = arch.topology.d_model;
    const auto& a = arch.attention;
    const auto hq = a.n_heads * a.head_dim;
    const auto hkv = a.n_kv_heads * a.head_dim;
    return sizer(arch, hq, d, TensorRole::AttnProj) +
           2 * sizer(arch, hkv, d, TensorRole::AttnProj) + sizer(arch, d, hq, TensorRole::AttnProj);
}

Status prepare_weights(ModelState& model) {
    // No weight absorption in this family. The hook exists because MLA needs
    // it; saying so explicitly is better than the interface not having it.
    (void)model;
    return {};
}

std::uint32_t window_span(const ArchIr& arch, LayerIndex layer) noexcept {
    (void)layer; // families that alternate windowed and full layers override this
    return arch.attention.sliding_window;
}

Status validate(const ArchIr& arch) {
    if (arch.attention.family != AttentionFamily::Gqa &&
        arch.attention.family != AttentionFamily::Mha) {
        return {StatusCode::Unsupported,
                std::string("gqa backend cannot execute attention family ") +
                    to_string(arch.attention.family)};
    }
    if (arch.attention.n_kv_heads == 0 || arch.attention.n_heads % arch.attention.n_kv_heads != 0) {
        return {StatusCode::InvalidArgument, "n_heads is not a multiple of n_kv_heads"};
    }
    if (arch.attention.rope.partial_dim > arch.attention.head_dim) {
        return {StatusCode::InvalidArgument, "rope partial_dim exceeds head_dim"};
    }
    if (arch.router.n_groups > 1) {
        return {StatusCode::Unsupported,
                "group-limited routing is an MLA-family router; this backend does not "
                "implement it, and silently ignoring n_group would change which experts fire"};
    }
    if (arch.router.bias_correction) {
        return {StatusCode::Unsupported,
                "pre-top-k bias correction (noaux_tc) is not implemented by this backend"};
    }
    return {};
}

const AttentionBackend& attention_backend() noexcept {
    static const AttentionBackend kBackend = [] {
        AttentionBackend b{};
        b.name = "gqa";
        b.family = AttentionFamily::Gqa;
        b.persist_format_id = kKvFormat;
        b.kv_bytes_per_token = &kv_bytes_per_token;
        b.weight_bytes_per_layer = &weight_bytes_per_layer;
        b.prepare_weights = &prepare_weights;
        return b;
    }();
    return kBackend;
}

const ArchBackend& backend() noexcept {
    static const ArchBackend kBackend = [] {
        ArchBackend b{};
        b.name = "gqa";
        b.attention = &attention_backend();
        b.validate = &validate;
        return b;
    }();
    return kBackend;
}

} // namespace soma::arch::gqa
