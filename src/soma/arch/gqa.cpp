// Soma — GQA/MHA attention and softmax-family routing, F32-activation path.
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
//   MiniMax-M3     64/4 GQA, head_dim 128, q/k_norm [head_dim] under Gemma's
//                  `(1 + w)` convention, only the first 64 of each head
//                  rotated, and a block-sparse key selector on 57 of 60 layers
//                  -- 128 experts, top-4, sigmoid + selection bias, one shared
//                  expert, and a clamped SwiGLU-OAI activation
//
// MHA is handled here as the n_kv_heads == n_heads case rather than as its own
// backend, because the only thing that changes is the repeat factor. BSA is
// handled here for the stronger version of the same argument: it changes which
// keys the softmax sees and nothing else at all.
//
// The family branches resolve ONCE per call, never per key -- `keep` is looked
// up per (row, head) and the inner loops walk contiguous runs.

#include "soma/arch/gqa.hpp"

#include "soma/f32_model.hpp"
#include "soma/kernels_f32.hpp"
#include "soma/threading.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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
/// `offset` is `ArchIr::rms_norm_weight_offset()` — the q/k norms are ordinary
/// RMSNorms of whatever convention the model uses, so they follow it exactly as
/// the layer norms do. Passing 0 unconditionally would scale a one-plus
/// checkpoint's queries by a weight centred on ZERO.
void apply_qk_norm(QkNormKind kind,
                   std::span<float> vec,
                   std::span<const float> weight,
                   std::uint32_t n_heads,
                   std::uint32_t head_dim,
                   float eps,
                   float offset) noexcept {
    if (kind == QkNormKind::None || weight.empty()) return;

    if (kind == QkNormKind::FullWidth) {
        f32::rmsnorm(vec, weight, n_heads * head_dim, eps, offset);
        return;
    }
    for (std::uint32_t h = 0; h < n_heads; ++h) {
        f32::rmsnorm(vec.subspan(static_cast<std::size_t>(h) * head_dim, head_dim),
                     weight,
                     head_dim,
                     eps,
                     offset);
    }
}

/// Project queries, splitting off a fused output gate when the family has one.
///
/// Without the gate this is one matmul into `ws.q` and nothing else. With it,
/// `q_proj` is emitted at `n_heads x 2 * head_dim` and each HEAD's slice is
/// `[query | gate]` — upstream views the projection as `[..., n_heads,
/// 2 * head_dim]` and chunks the last axis.
///
/// The de-interleave is the whole content of this function and the reason it
/// exists rather than being written twice. Reading the projection as two
/// contiguous halves — all queries then all gates — is the plausible
/// alternative, it produces correctly-shaped buffers, and it pairs every head's
/// output with a different head's gate. Finite, fluent, and not the model.
void project_q(const ArchIr& arch,
               const soma::WeightRef& q_proj,
               std::span<const float> xs,
               std::uint32_t n_rows,
               soma::F32Workspace& ws) noexcept {
    if (!arch.attention.fused_output_gate) {
        soma::matmul(q_proj, xs, n_rows, ws.q);
        return;
    }
    const auto H = arch.attention.n_heads;
    const auto hd = arch.attention.head_dim;
    const auto hq = H * hd;

    soma::matmul(q_proj, xs, n_rows, ws.q_raw);
    for (std::uint32_t r = 0; r < n_rows; ++r) {
        const float* src = ws.q_raw.data() + static_cast<std::size_t>(r) * 2 * hq;
        float* q = ws.q.data() + static_cast<std::size_t>(r) * hq;
        float* g = ws.attn_gate.data() + static_cast<std::size_t>(r) * hq;
        for (std::uint32_t h = 0; h < H; ++h) {
            const float* head = src + static_cast<std::size_t>(h) * 2 * hd;
            std::copy_n(head, hd, q + static_cast<std::size_t>(h) * hd);
            std::copy_n(head + hd, hd, g + static_cast<std::size_t>(h) * hd);
        }
    }
}

/// `attn_heads *= sigmoid(gate)`, in place, before the output projection.
///
/// AFTER attention and BEFORE `o_proj`, which is the only ordering that is the
/// model: gating the o_proj OUTPUT would apply an `n_heads * head_dim`-wide mask
/// to a `d_model`-wide vector, and gating the queries instead would change what
/// attention attends to rather than how much of it survives.
void apply_output_gate(const ArchIr& arch,
                       std::uint32_t n_rows,
                       soma::F32Workspace& ws) noexcept {
    if (!arch.attention.fused_output_gate) return;
    const auto hq = arch.attention.n_heads * arch.attention.head_dim;
    const auto span = static_cast<std::size_t>(n_rows) * hq;
    for (std::size_t i = 0; i < span; ++i) {
        ws.attn_heads[i] *= 1.0f / (1.0f + std::exp(-ws.attn_gate[i]));
    }
}

    // ── block-sparse key selection ───────────────────────────────────────────────
    //
    // The "Lightning Indexer". A small scoring branch decides, per query and per GQA
    // group, which BLOCKS of keys the softmax below is allowed to see. Everything
    // else about attention is unchanged — same projections, same norms, same
    // rotation, same score loop — which is why this lives here rather than in a
    // backend of its own.
    //
    // Transcribed from `MiniMaxM3VLIndexer.forward` and `build_block_mask` in
    // transformers 5.15.1, and the four places a plausible reading differs from
    // theirs are marked below. Each of them produces a model that runs.

    /// Positive and negative infinity, named because both are load-bearing VALUES
    /// here rather than error states: `-inf` marks a block no key can reach and is
    /// what `topk` sorts to the end, and `+inf` is how a forced-local block wins a
    /// slot rather than being appended to the selection.
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    constexpr float kPosInf = std::numeric_limits<float>::infinity();

    /// Per-forward scratch for the indexer, carried in `F32Workspace::arch_state`.
    ///
    /// The same opaque idiom `arch::mla` uses for its DSA selection, and for the same
    /// reason: this outlives one layer but not one forward, the core must not learn
    /// what is in it, and a `std::vector` allocated in the middle of a 60-layer
    /// forward is exactly the sort of thing that only looks free.
    ///
    /// Unlike DSA's, this is scratch and NOT shared state. Every indexed layer
    /// recomputes its own selection from its own weights — this family has no
    /// IndexShare — so a stale payload from a previous prompt can only be the wrong
    /// SIZE, never the wrong contents. It is resized and refilled on every use
    /// regardless.
    struct BsaScratch {
        std::vector<float> q;          ///< [rows, n_index_heads * index_head_dim]
        std::vector<float> k;          ///< [rows, index_head_dim] — one head
        std::vector<float> block;      ///< [n_workers, n_blocks] block scores
        std::vector<float> topv;       ///< [n_workers, n_blocks]
        std::vector<std::uint32_t> topi; ///< [n_workers, n_blocks]
        std::vector<std::uint8_t> keep;  ///< [rows, n_index_heads, n_blocks]
        std::vector<std::uint32_t> pos;  ///< [rows] absolute positions
        std::uint32_t n_blocks = 0;

        /// Row `r`, indexer head `h`: one byte per key block, 1 = visible.
        std::uint8_t* row(std::uint32_t r, std::uint32_t h, std::uint32_t heads) noexcept {
            return keep.data() +
                   ((static_cast<std::size_t>(r) * heads) + h) * n_blocks;
        }
        const std::uint8_t* row(std::uint32_t r, std::uint32_t h, std::uint32_t heads) const noexcept {
            return keep.data() +
                   ((static_cast<std::size_t>(r) * heads) + h) * n_blocks;
        }
    };

    BsaScratch& bsa_scratch(soma::F32Workspace& ws) {
        if (ws.arch_state.empty()) {
            ws.arch_state.adopt(new BsaScratch(), [](void* p) { delete static_cast<BsaScratch*>(p); });
        }
        return *ws.arch_state.as<BsaScratch>();
    }

    /// Project, normalize and rotate the indexer's queries and this row's key.
    ///
    /// `positions[r]` is the ABSOLUTE position of row `r`, which for the cached path
    /// is `KvRow::pos` and not the row's index in the batch — the same distinction
    /// the main attention makes, and wrong here in the same way: finite, plausible,
    /// and rotated as if every row were at the head of the sequence.
    void bsa_project(const ArchIr& arch,
                     const F32AttnWeights& aw,
                     std::span<const float> xs,
                     std::uint32_t n_rows,
                     const std::uint32_t* positions,
                     BsaScratch& sc) noexcept {
        const auto& b = arch.attention.bsa;
        const auto H = b.n_index_heads;
        const auto D = b.index_head_dim;
        const auto eps = arch.rms_norm_eps;
        const auto offset = arch.rms_norm_weight_offset();

        sc.q.resize(static_cast<std::size_t>(n_rows) * H * D);
        sc.k.resize(static_cast<std::size_t>(n_rows) * D);
        soma::matmul(aw.idx.q_proj, xs, n_rows, sc.q);
        soma::matmul(aw.idx.k_proj, xs, n_rows, sc.k);

        // The rotation is the MAIN attention's slice, not a slice of the indexer's
        // own head. Upstream passes `cos[..., :self.head_dim]` into the shared
        // `apply_rotary_pos_emb`, which then rotates `cos.shape[-1]` channels — and
        // `cos` is only `partial_dim` wide to begin with, so slicing it at the wider
        // index_head_dim is a no-op. `validate_arch_ir` refuses the narrower case
        // rather than transcribing what that slice would do.
        const auto& rope = arch.attention.rope;
        const auto rotary = (rope.partial_dim > 0) ? rope.partial_dim : D;

        for (std::uint32_t r = 0; r < n_rows; ++r) {
            auto qr = std::span<float>(sc.q).subspan(static_cast<std::size_t>(r) * H * D,
                                                     static_cast<std::size_t>(H) * D);
            auto kr = std::span<float>(sc.k).subspan(static_cast<std::size_t>(r) * D, D);
            // PER HEAD, both of them. `q_norm` and `k_norm` are `RMSNorm(head_dim)`
            // applied to a `[..., heads, head_dim]` view, so the key's single head
            // normalizes over D exactly as each of the query's H heads does.
            for (std::uint32_t h = 0; h < H; ++h) {
                f32::rmsnorm(qr.subspan(static_cast<std::size_t>(h) * D, D),
                             aw.idx.q_norm,
                             D,
                             eps,
                             offset);
            }
            f32::rmsnorm(kr, aw.idx.k_norm, D, eps, offset);

            const auto p = positions[r];
            if (rope.interleaved) {
                f32::rope_interleaved(qr, H, D, p, rope.theta, rotary);
                f32::rope_interleaved(kr, 1, D, p, rope.theta, rotary);
            } else {
                f32::rope_neox(qr, H, D, p, rope.theta, rotary);
                f32::rope_neox(kr, 1, D, p, rope.theta, rotary);
            }
        }
    }

    /// Walk the key positions one query may attend, in ascending order.
    ///
    /// `keep == nullptr` is DENSE and is the whole of what plain GQA and MHA do
    /// here: the visible set is `[lo, hi)` and there is no mask to consult. The
    /// branch resolves once per (row, head), never per key, so the families that
    /// predate block sparsity walk exactly the loop they always did.
    ///
    /// The sparse walk iterates BLOCKS and then the keys inside each, rather
    /// than every key with a per-key mask test. That is not a micro-optimization
    /// — it is what makes the family pay off. A per-key test still touches every
    /// cached position, so the loop would stay O(context) and block sparsity
    /// would buy nothing but a smaller softmax.
    ///
    /// Ascending, and both callers rely on it: the scores are written in this
    /// order and read back in the same one, so a block walk that visited blocks
    /// out of order would pair every score with the wrong value vector.
    /// `top_k` returns indices by descending SCORE, which is precisely why the
    /// mask is a per-block flag here rather than the packed index list upstream
    /// hands its kernel.
    template <class Fn>
    void visit_visible(const std::uint8_t* keep,
                       std::uint32_t lo,
                       std::uint32_t hi,
                       std::uint32_t block_size,
                       Fn&& fn) noexcept {
        if (keep == nullptr) {
            for (std::uint32_t j = lo; j < hi; ++j) fn(j);
            return;
        }
        const std::uint32_t n_blocks = (hi + block_size - 1) / block_size;
        for (std::uint32_t b = 0; b < n_blocks; ++b) {
            if (keep[b] == 0) continue;
            const auto begin = std::max(lo, b * block_size);
            const auto end = std::min(hi, (b + 1) * block_size);
            for (std::uint32_t j = begin; j < end; ++j) fn(j);
        }
    }

    /// Fill `sc.keep` for one row: which key blocks this query may attend.
    ///
    /// `key_at(j)` yields the cached indexer key for absolute position `j`, so the
    /// same routine serves the whole-sequence path (keys in `sc.k`) and the cached
    /// path (keys in the KV cache's tail) without either learning about the other.
    template <class KeyAt>
    void bsa_select_row(const ArchIr& arch,
                        const BsaScratch& sc,
                        std::uint32_t r,
                        std::uint32_t pos,
                        std::uint32_t lo,
                        std::uint32_t hi,
                        KeyAt key_at,
                        float* block_scores,
                        float* top_values,
                        std::uint32_t* top_indices,
                        std::uint8_t* keep_base) noexcept {
        const auto& b = arch.attention.bsa;
        const auto H = b.n_index_heads;
        const auto D = b.index_head_dim;
        const auto bs = b.block_size;

        // Blocks are anchored to ABSOLUTE key slots, not to the visible range: block
        // `n` covers positions `[n*bs, (n+1)*bs)` whatever `lo` is. Deriving them
        // from `j - lo` instead would shift every boundary once a sliding window or
        // a chunked prefill made `lo` non-zero, and the selection would no longer
        // agree with the one the same prompt got at full length.
        const std::uint32_t n_blocks = (hi + bs - 1) / bs;

        for (std::uint32_t h = 0; h < H; ++h) {
            const float* qv = sc.q.data() +
                              (static_cast<std::size_t>(r) * H + h) * D;
            std::fill_n(block_scores, n_blocks, kNegInf);

            // MAX-pooled, not summed or averaged. A block is worth attending if its
            // BEST key is — one strong key carries the `block_size - 1` around it,
            // which is the whole premise of scoring at block granularity.
            for (std::uint32_t j = lo; j < hi; ++j) {
                const float s = f32::dot(std::span<const float>(qv, D),
                                         std::span<const float>(key_at(j), D),
                                         D);
                const auto blk = j / bs;
                if (s > block_scores[blk]) block_scores[blk] = s;
            }

            // The local guarantee, applied by OVERWRITING the score with +inf rather
            // than by adding these blocks to the selection afterwards.
            //
            // That is upstream's mechanism and it is not an implementation detail:
            // forcing them through the score means they consume top-k slots, so a
            // query sees `topk_blocks` blocks and not `topk_blocks + local_blocks`.
            // Appending them instead would attend to the query's own block twice —
            // and the deployment kernel this format feeds reads the list
            // sequentially, so a repeat is a double count rather than a no-op.
            for (std::uint32_t i = 0; i < b.local_blocks; ++i) {
                const auto qb = pos / bs;
                const auto blk = (qb >= i) ? (qb - i) : 0u;
                if (blk < n_blocks) block_scores[blk] = kPosInf;
            }

            const auto k = std::min(b.topk_blocks, n_blocks);
            f32::top_k(std::span<const float>(block_scores, n_blocks),
                       n_blocks,
                       k,
                       std::span<std::uint32_t>(top_indices, k),
                       std::span<float>(top_values, k));

            std::uint8_t* keep = keep_base + (static_cast<std::size_t>(r) * H + h) * sc.n_blocks;
            std::fill_n(keep, sc.n_blocks, std::uint8_t{0});
            for (std::uint32_t i = 0; i < k; ++i) {
                // A `-inf` slot is a block no visible key reaches: future, or empty
                // because the prefix is shorter than the block grid. Upstream tags
                // these `-1` and the kernel skips them. Keeping them instead would
                // let a query attend past its own position.
                if (top_values[i] != kNegInf) keep[top_indices[i]] = 1;
            }
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

    // The IR says whether the output gate rides inside `q_proj`; the tensor says
    // how wide it actually is. Checked here because the disagreement is
    // otherwise silent in the worst way: at the plain width the de-interleave
    // reads past the projection, and at the double width without the flag every
    // head's query is really half query and half gate. Both run.
    {
        const auto hq = arch.attention.n_heads * arch.attention.head_dim;
        const auto want = arch.attention.fused_output_gate ? 2u * hq : hq;
        if (w->q_proj.rows != want) return StatusCode::InvalidArgument;
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

    // The router's selection bias. OPTIONAL and absent for every family this
    // backend served before MiniMax-M3, which is why it is bound here rather
    // than required: a checkpoint without one simply routes on its scores.
    //
    // Bound from the MoE block's name, so a family that spells the block
    // `block_sparse_moe` finds it where the router itself lives. Hardcoding
    // `mlp.` is a defect this codebase has already paid for once, in
    // `arch::mla::f32_bind_layer` against Kimi's block name -- the bias silently
    // did not load and the router chose different experts, fluently.
    {
        const auto bias_name = arch.naming.moe_block + ".gate.e_score_correction_bias";
        (void)soma::bind_layer_f32(ctx, bias_name.c_str(), w->e_score_bias, /*optional=*/true);
    }

    // ── the block-sparse indexer, on indexed layers only ─────────────────────
    //
    // Read from `bsa.layer_kinds` rather than from a stride, for the reason the
    // adapter gives: MiniMax-M3's three leading layers own no indexer tensors at
    // all, and a layer the IR called indexed without the weights to back it
    // would fail here -- or, worse, bind nothing and select over garbage.
    if (arch.attention.family == AttentionFamily::GqaBsa) {
        const auto& b = arch.attention.bsa;
        if (ctx.layer >= b.layer_kinds.size()) return StatusCode::InvalidArgument;
        if (b.layer_kinds[ctx.layer] == IndexerKind::Full) {
            using soma::TensorRole;
            if (!soma::bind_layer_weight(
                     ctx, "self_attn.indexer.q_proj.weight", TensorRole::AttnProj, w->idx.q_proj)
                     .ok() ||
                !soma::bind_layer_weight(
                     ctx, "self_attn.indexer.k_proj.weight", TensorRole::AttnProj, w->idx.k_proj)
                     .ok() ||
                !soma::bind_layer_f32(ctx, "self_attn.indexer.q_norm.weight", w->idx.q_norm)
                     .ok() ||
                !soma::bind_layer_f32(ctx, "self_attn.indexer.k_norm.weight", w->idx.k_norm)
                     .ok()) {
                return StatusCode::NotFound;
            }
            // The key projection is SINGLE-HEADED and the query projection is
            // not. Checked rather than assumed, because the two plausible
            // mistakes are silent in opposite directions: a key projection read
            // at the query width would cache `n_index_heads` times the bytes per
            // token, and a query projection read at the key width would score
            // every GQA group against head 0's queries.
            if (w->idx.q_proj.rows != b.n_index_heads * b.index_head_dim ||
                w->idx.k_proj.rows != b.index_head_dim ||
                w->idx.q_norm.size() != b.index_head_dim ||
                w->idx.k_norm.size() != b.index_head_dim) {
                return StatusCode::InvalidArgument;
            }
            w->has_indexer = true;
        }
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
    project_q(arch, aw.q_proj, xs, n_rows, ws);
    soma::matmul(aw.k_proj, xs, n_rows, ws.k);
    soma::matmul(aw.v_proj, xs, n_rows, ws.v);

    for (std::uint32_t r = 0; r < n_rows; ++r) {
        apply_qk_norm(arch.attention.qk_norm,
                      std::span<float>(ws.q).subspan(static_cast<std::size_t>(r) * hq, hq),
                      aw.q_norm,
                      H,
                      hd,
                      eps,
                      arch.rms_norm_weight_offset());
        apply_qk_norm(arch.attention.qk_norm,
                      std::span<float>(ws.k).subspan(static_cast<std::size_t>(r) * hkv, hkv),
                      aw.k_norm,
                      KV,
                      hd,
                      eps,
                      arch.rms_norm_weight_offset());

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
        std::copy_n(k_r.data(), hkv, rows[r].k_at(layer, p)); // K occupies the plane's head
        std::copy_n(ws.v.data() + static_cast<std::size_t>(r) * hkv, hkv, rows[r].v_at(layer, p));
    }

    const auto window = arch.attention.sliding_window;
    const std::uint32_t n_workers = ThreadPool::global().size();
    std::uint32_t longest = 1;
    for (std::uint32_t r = 0; r < n_rows; ++r)
        longest = std::max(longest, rows[r].len);
    ws.ensure_score_scratch(n_workers, longest);

    // ── block-sparse selection, when this layer owns an indexer ──────────────
    //
    // The indexer's own key is appended to the cache FIRST, for exactly the
    // reason K and V are above: the query's own block is forced visible, so a
    // query that could not see the key it just produced would score its own
    // block against a stale slot.
    //
    // It rides in the K plane's TAIL rather than in a plane of its own.
    // `KvCache` allocates two planes and this family uses both, so there is no
    // free one to borrow the way MLA+DSA borrows its empty V -- and a third
    // plane would be a core change for one family. See `f32_kv_geometry`.
    const auto& bsa = arch.attention.bsa;
    const bool sparse = aw.has_indexer;
    const auto idx_group = sparse ? (H / bsa.n_index_heads) : 1u;
    BsaScratch* sc = nullptr;
    if (sparse) {
        sc = &bsa_scratch(ws);
        sc->pos.resize(n_rows);
        for (std::uint32_t r = 0; r < n_rows; ++r)
            sc->pos[r] = rows[r].pos;
        bsa_project(arch, aw, xs, n_rows, sc->pos.data(), *sc);

        const auto D = bsa.index_head_dim;
        for (std::uint32_t r = 0; r < n_rows; ++r) {
            std::copy_n(sc->k.data() + static_cast<std::size_t>(r) * D,
                        D,
                        rows[r].k_at(layer, rows[r].pos) + hkv);
        }

        sc->n_blocks = (longest + bsa.block_size - 1) / bsa.block_size;
        sc->keep.assign(static_cast<std::size_t>(n_rows) * bsa.n_index_heads * sc->n_blocks, 0);
        sc->block.resize(static_cast<std::size_t>(n_workers) * sc->n_blocks);
        sc->topv.resize(static_cast<std::size_t>(n_workers) * sc->n_blocks);
        sc->topi.resize(static_cast<std::size_t>(n_workers) * sc->n_blocks);

        ThreadPool::global().parallel_for(
            n_rows, 1, [&](std::uint32_t r_begin, std::uint32_t r_end, std::uint32_t worker) {
                const auto off = static_cast<std::size_t>(worker) * sc->n_blocks;
                for (std::uint32_t r = r_begin; r < r_end; ++r) {
                    const auto& kv = rows[r];
                    const std::uint32_t hi = kv.len;
                    const std::uint32_t lo = (window > 0 && hi > window) ? (hi - window) : 0u;
                    bsa_select_row(
                        arch,
                        *sc,
                        r,
                        kv.pos,
                        lo,
                        hi,
                        [&](std::uint32_t j) { return kv.k_at(layer, j) + hkv; },
                        sc->block.data() + off,
                        sc->topv.data() + off,
                        sc->topi.data() + off,
                        sc->keep.data());
                }
            });
    }

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
                    // One selection per GQA GROUP: indexer head `h / idx_group`
                    // is the one that scored this query head's keys. Null when
                    // the layer has no indexer, which is what collapses the two
                    // walks below into the dense ones they used to be.
                    const std::uint8_t* keep =
                        sparse ? sc->row(r, h / idx_group, bsa.n_index_heads) : nullptr;

                    std::uint32_t n = 0;
                    visit_visible(keep, lo, hi, bsa.block_size, [&](std::uint32_t j) {
                        const float* kvec = kv.k_at(layer, j) + kvh * hd;
                        scores[n++] = f32::dot(std::span<const float>(qv, hd),
                                               std::span<const float>(kvec, hd),
                                               hd) *
                                      scale;
                    });
                    f32::softmax(std::span<float>(scores, n), n);

                    float* dst = ws.attn_heads.data() + static_cast<std::size_t>(r) * hq +
                                 static_cast<std::size_t>(h) * hd;
                    std::fill_n(dst, hd, 0.0f);
                    n = 0;
                    visit_visible(keep, lo, hi, bsa.block_size, [&](std::uint32_t j) {
                        const float* vvec = kv.v_at(layer, j) + kvh * hd;
                        f32::axpy(scores[n++],
                                  std::span<const float>(vvec, hd),
                                  hd,
                                  std::span<float>(dst, hd));
                    });
                }
            }
        });

    apply_output_gate(arch, n_rows, ws);
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
    project_q(arch, aw.q_proj, xs, n_tokens, ws);
    soma::matmul(aw.k_proj, xs, n_tokens, ws.k);
    soma::matmul(aw.v_proj, xs, n_tokens, ws.v);

    // Norm before RoPE, matching HF for both reference families.
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
        apply_qk_norm(arch.attention.qk_norm,
                      std::span<float>(ws.q).subspan(static_cast<std::size_t>(t) * hq, hq),
                      aw.q_norm,
                      H,
                      hd,
                      eps,
                      arch.rms_norm_weight_offset());
        apply_qk_norm(arch.attention.qk_norm,
                      std::span<float>(ws.k).subspan(static_cast<std::size_t>(t) * hkv, hkv),
                      aw.k_norm,
                      KV,
                      hd,
                      eps,
                      arch.rms_norm_weight_offset());

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

    // ── block-sparse selection ──────────────────────────────────────────────
    //
    // The whole-sequence twin of the cached path above. Every indexer key is in
    // `sc->k` rather than in a cache, because this path owns the entire prefix
    // and there is nothing to append to.
    const auto& bsa = arch.attention.bsa;
    const bool sparse = aw.has_indexer;
    const auto idx_group = sparse ? (H / bsa.n_index_heads) : 1u;
    const std::uint32_t n_workers = ThreadPool::global().size();
    BsaScratch* sc = nullptr;
    if (sparse) {
        sc = &bsa_scratch(ws);
        sc->pos.resize(n_tokens);
        for (std::uint32_t t = 0; t < n_tokens; ++t)
            sc->pos[t] = t;
        bsa_project(arch, aw, xs, n_tokens, sc->pos.data(), *sc);

        const auto D = bsa.index_head_dim;
        sc->n_blocks = (n_tokens + bsa.block_size - 1) / bsa.block_size;
        sc->keep.assign(static_cast<std::size_t>(n_tokens) * bsa.n_index_heads * sc->n_blocks, 0);
        sc->block.resize(static_cast<std::size_t>(n_workers) * sc->n_blocks);
        sc->topv.resize(static_cast<std::size_t>(n_workers) * sc->n_blocks);
        sc->topi.resize(static_cast<std::size_t>(n_workers) * sc->n_blocks);

        ThreadPool::global().parallel_for(
            n_tokens, 1, [&](std::uint32_t t_begin, std::uint32_t t_end, std::uint32_t worker) {
                const auto off = static_cast<std::size_t>(worker) * sc->n_blocks;
                for (std::uint32_t t = t_begin; t < t_end; ++t) {
                    const std::uint32_t hi = t + 1;
                    const std::uint32_t lo = (window > 0 && hi > window) ? (hi - window) : 0u;
                    bsa_select_row(
                        arch,
                        *sc,
                        t,
                        t,
                        lo,
                        hi,
                        [&](std::uint32_t j) {
                            return sc->k.data() + static_cast<std::size_t>(j) * D;
                        },
                        sc->block.data() + off,
                        sc->topv.data() + off,
                        sc->topi.data() + off,
                        sc->keep.data());
                }
            });
    }

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
    ws.ensure_score_scratch(n_workers, n_tokens);

    ThreadPool::global().parallel_for(
        n_tokens, 1, [&](std::uint32_t t_begin, std::uint32_t t_end, std::uint32_t worker) {
            float* scores = ws.worker_scores(worker, n_tokens);
            for (std::uint32_t t = t_begin; t < t_end; ++t) {
                // Causal, optionally windowed. `lo` is the first visible position.
                const std::uint32_t hi = t + 1;
                const std::uint32_t lo = (window > 0 && hi > window) ? (hi - window) : 0u;
                for (std::uint32_t h = 0; h < H; ++h) {
                    const std::uint32_t kvh = h / group;
                    const float* qv = ws.q.data() + static_cast<std::size_t>(t) * hq +
                                      static_cast<std::size_t>(h) * hd;
                    const std::uint8_t* keep =
                        sparse ? sc->row(t, h / idx_group, bsa.n_index_heads) : nullptr;

                    std::uint32_t n = 0;
                    visit_visible(keep, lo, hi, bsa.block_size, [&](std::uint32_t j) {
                        const float* kv = ws.k.data() + static_cast<std::size_t>(j) * hkv +
                                          static_cast<std::size_t>(kvh) * hd;
                        scores[n++] = f32::dot(std::span<const float>(qv, hd),
                                               std::span<const float>(kv, hd),
                                               hd) *
                                      scale;
                    });
                    f32::softmax(std::span<float>(scores, n), n);

                    float* dst = ws.attn_heads.data() + static_cast<std::size_t>(t) * hq +
                                 static_cast<std::size_t>(h) * hd;
                    std::fill_n(dst, hd, 0.0f);
                    n = 0;
                    visit_visible(keep, lo, hi, bsa.block_size, [&](std::uint32_t j) {
                        const float* vv = ws.v.data() + static_cast<std::size_t>(j) * hkv +
                                          static_cast<std::size_t>(kvh) * hd;
                        // Runs once per (query, VISIBLE key, head). Dense that is
                        // O(T^2 * heads * head_dim) — the same order as the score
                        // dot above, and together they dominate the whole forward
                        // at long context. Block-sparse it is the same expression
                        // with T replaced by `BsaSpec::visible_keys(T)`, which is
                        // the entire point of the family. Both go through soma::f32
                        // rather than a hand-written loop so the SIMD dispatch
                        // happens in one place.
                        f32::axpy(scores[n++],
                                  std::span<const float>(vv, hd),
                                  hd,
                                  std::span<float>(dst, hd));
                    });
                }
            }
        });

    apply_output_gate(arch, n_tokens, ws);
    soma::matmul(aw.o_proj,
                 ws.attn_heads,
                 n_tokens,
                 std::span<float>(out, static_cast<std::size_t>(n_tokens) * d));
    return StatusCode::Ok;
}

StatusCode f32_route(const ArchIr& arch,
                     const soma::F32LayerWeights& lw,
                     const TokenId*,
                     const float* logits,
                     std::uint32_t n_tokens,
                     std::uint32_t* out_ids,
                     float* out_weights) noexcept {
    const auto E = arch.router.n_experts;
    const auto K = arch.router.top_k;

    // The per-expert SELECTION bias, when the layer carries one.
    //
    // This argument used to be `(void)lw` with a note that this family's router
    // has no parameters beyond the gate matrix. That was true of Qwen3-MoE,
    // OLMoE and Mixtral and it is not true of MiniMax-M3, whose
    // `MiniMaxM3VLTopKRouter` adds `e_score_correction_bias` to the sigmoid
    // scores before its top-k. Reading the bias is optional; IGNORING it is not
    // — a bias that exists to steer load balancing across 128 experts changes
    // which four of them fire on every token, and nothing downstream can tell.
    //
    // Driven off the TENSOR rather than off `router.bias_correction`, so a
    // checkpoint that ships no such tensor routes exactly as it did before this
    // existed. Same shape as `arch::mla::f32_route`.
    const auto* p = lw.attn.empty() ? nullptr : lw.attn.as<F32AttnWeights>();
    const float* bias =
        (p == nullptr || p->e_score_bias.size() != E) ? nullptr : p->e_score_bias.data();

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
        if (bias == nullptr) {
            f32::top_k(probs,
                       E,
                       K,
                       std::span<std::uint32_t>(out_ids + slot, K),
                       std::span<float>(out_weights + slot, K));
        } else {
            // Selection on the BIASED scores, weights from the UNBIASED ones.
            // Not interchangeable: the bias exists to steer which experts fire
            // and has no business scaling what they contribute, and carrying it
            // into the weights would re-scale every expert by a load-balancing
            // term. Upstream is explicit —
            // `top_k_weights = routing_weights.gather(1, top_k_index)`.
            static thread_local std::vector<float> sel;
            sel.resize(E);
            for (std::uint32_t e = 0; e < E; ++e)
                sel[e] = probs[e] + bias[e];
            f32::top_k(sel,
                       E,
                       K,
                       std::span<std::uint32_t>(out_ids + slot, K),
                       std::span<float>(out_weights + slot, K));
            for (std::uint32_t s = 0; s < K; ++s)
                out_weights[slot + s] = probs[out_ids[slot + s]];
        }

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
    // Named rather than positional. Adding `kv_geometry` to F32Backend
    // shifted every later member, and the compiler caught it only because the
    // types happened to disagree — an aggregate initialiser that still lined up
    // by type would have bound the wrong pointers in silence.
    //
    // `kv_geometry` used to be left null on the argument that null means the
    // GQA default and this IS the GQA backend, so stating it would be a second
    // copy of one formula waiting to disagree with the first.
    //
    // The argument was right and its premise stopped being true. This backend
    // now serves a family whose K plane is NOT `n_kv_heads * head_dim`, so the
    // core's default is no longer the same formula — and the second copy is
    // gone rather than duplicated: `kv_bytes_per_token` is derived from
    // `f32_kv_geometry` too, which is what the null used to buy.
    static const soma::F32Backend kBackend = [] {
        soma::F32Backend b{};
        b.name = "gqa";
        b.bind_layer = &f32_bind_layer;
        b.attention = &f32_attention;
        b.kv_geometry = &f32_kv_geometry;
        b.attention_kv = &f32_attention_kv;
        b.route = &f32_route;
        return b;
    }();
    return kBackend;
}

// ── Descriptors ──────────────────────────────────────────────────────────────
//
// `F32Backend` is the execution descriptor. `AttentionBackend` carries only the
// sizing and persistence properties the planner and checkpoint store need.

soma::KvGeometry f32_kv_geometry(const ArchIr& arch) noexcept {
    const auto hkv = arch.attention.n_kv_heads * arch.attention.head_dim;
    // The indexer's key rides in the K plane's TAIL: `[K | index_k]` per
    // position. Both planes are in use on this family, so there is no empty one
    // to borrow — which is the difference from MLA+DSA, whose V plane holds
    // nothing and became the natural home for exactly this.
    //
    // ONE indexer key per position, not `n_index_heads` of them. `k_proj` is
    // `d_model -> index_head_dim` and every indexer head scores against that
    // same key; sizing it per head would over-allocate this plane fourfold on
    // the reference checkpoint.
    const auto index = (arch.attention.family == AttentionFamily::GqaBsa)
                           ? arch.attention.bsa.index_head_dim
                           : 0u;
    return soma::KvGeometry{hkv + index, hkv};
}

std::size_t kv_bytes_per_token(const ArchIr& arch) noexcept {
    // Full K and V per token, per layer: 2 * n_kv_heads * head_dim.
    //
    // fp32, full stop. This used to add "the planner scales this by the
    // configured KV dtype", and there is no such configuration: KvCache holds
    // `std::vector<float>` (kv_cache.hpp), plan.cpp multiplies this figure by
    // context and slots and by nothing else, and no flag, env var or container
    // field selects a KV dtype. The sentence described an fp16 cache that was
    // never built, and it mattered because the numbers quoted alongside it were
    // fp16 numbers — half the RAM this actually asks for, in the optimistic
    // direction (D45).
    //
    // Worth stating at all because GQA's KV competes directly with the expert
    // cache for the same RAM (docs/architecture.md §2.2).
    //
    // Derived from `f32_kv_geometry` rather than restating `2 * n_kv_heads *
    // head_dim`, so the figure the planner predicts and the buffer
    // `KvCache::open` reserves cannot drift. They are the same expression now,
    // which they were not while this hardcoded GQA's two equal planes: block
    // sparsity widens the K plane and nothing here would have noticed.
    const auto geom = f32_kv_geometry(arch);
    const std::size_t per_layer =
        (static_cast<std::size_t>(geom.k_floats) + geom.v_floats) * sizeof(float);
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

std::uint64_t resident_weight_bytes(const ArchIr& arch,
                                    AttentionBackend::ByteSizer sizer) noexcept {
    const auto n_layers = static_cast<std::uint64_t>(arch.topology.n_layers);
    std::uint64_t total = n_layers * weight_bytes_per_layer(arch, sizer);
    if (arch.attention.family != AttentionFamily::GqaBsa) {
        // Byte-identical to what the planner computed before this function
        // existed, which is the point: GQA and MHA have uniform layers and an
        // average over them IS exact.
        return total;
    }

    // BSA's layers are not uniform, so an average over them is not exact and a
    // per-layer figure would have to be wrong for two thirds of the stack in one
    // direction or the other. MiniMax-M3's three leading layers own no indexer;
    // its other 57 own a query projection, a single-head key projection and two
    // norms apiece.
    //
    // The norms are counted at f32 unconditionally, matching how they are bound
    // — `QuantMap::norms` is not a role the converter quantizes.
    const auto& b = arch.attention.bsa;
    const auto d = arch.topology.d_model;
    const auto per_indexed =
        sizer(arch, b.n_index_heads * b.index_head_dim, d, TensorRole::AttnProj) +
        sizer(arch, b.index_head_dim, d, TensorRole::AttnProj) +
        2ull * b.index_head_dim * sizeof(float);
    total += static_cast<std::uint64_t>(b.n_indexed_layers()) * per_indexed;
    return total;
}

std::uint32_t window_span(const ArchIr& arch, LayerIndex layer) noexcept {
    (void)layer; // families that alternate windowed and full layers override this
    return arch.attention.sliding_window;
}

const AttentionBackend& attention_backend() noexcept {
    static const AttentionBackend kBackend = [] {
        AttentionBackend b{};
        b.name = "gqa";
        b.family = AttentionFamily::Gqa;
        b.persist_format_id = kKvFormat;
        b.kv_bytes_per_token = &kv_bytes_per_token;
        b.weight_bytes_per_layer = &weight_bytes_per_layer;
        // Preferred over the per-layer average when both are set. Deliberately
        // NOT `kv_bytes_for_context`, which would be equally exact here and
        // would also switch `KvCache` to its OPAQUE layout — a byte blob whose
        // geometry the backend owns end to end. This family's cache is two
        // ordinary planes and the core can address it, so declaring that
        // function would trade a working plane-based checkpoint path for
        // nothing.
        b.resident_weight_bytes = &resident_weight_bytes;
        return b;
    }();
    return kBackend;
}

} // namespace soma::arch::gqa
