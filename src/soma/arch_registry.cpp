// Soma — the architecture resolver.
//
// THE single core translation unit permitted to include soma/arch/ and to
// switch on AttentionFamily. tools/ci/check_seam.py allow-lists exactly this
// file and the backends themselves.
//
// Everything here runs ONCE per model load. A switch on family anywhere in a
// loop is a seam violation regardless of which file it appears in — the point
// of resolving to a descriptor is that the hot path never asks the question
// again.

#include "soma/arch/compressed_sparse.hpp"
#include "soma/arch/deepseek_dspark.hpp"
#include "soma/arch/gdn.hpp"
#include "soma/arch/gqa.hpp"
#include "soma/arch/kda.hpp"
#include "soma/arch/mla.hpp"
#include "soma/f32_model.hpp"

namespace soma {

const F32Backend* resolve_f32_backend(const ArchIr& arch) noexcept {
    switch (arch.attention.family) {
    case AttentionFamily::Mha:
    case AttentionFamily::Gqa:
        // MHA is the n_kv_heads == n_heads case of the same backend: the
        // only difference is a repeat factor of 1. Giving it a separate
        // implementation would mean two code paths to keep in agreement for
        // no behavioural gain.
        return &arch::gqa::f32_backend();

    case AttentionFamily::GqaBsa:
        // The SAME backend, for the same reason MHA shares it: block sparsity
        // changes which keys the softmax sees and nothing else. Every
        // projection, both qk-norms, the partial rotation and the score loop are
        // GQA's, and `arch::gqa` branches on the family internally to consult a
        // per-block visibility mask before the dot product.
        //
        // The precedent is `MlaDsa` two cases down, and so is the hazard: this
        // line pointing at a family with no selector would run MiniMax-M3 as
        // DENSE attention over 57 of its 60 layers. Finite, fluent, exact below
        // 2048 tokens where top-k selects everything anyway, and a different
        // model beyond that -- which is the worst shape a defect can take,
        // because a short-prompt smoke test passes.
        //
        // It says `gqa` from the start rather than nullptr, and that is a claim
        // about evidence rather than confidence: `tests/fixtures/tiny/MiniMax-M3-Tiny`
        // is an oracle built by transformers 5.15.1's own
        // `modeling_minimax_m3_vl.py` -- pure torch on the CPU, no Triton and no
        // CUDA -- and its indexer, block pooling and top-k are shrunk so that
        // the fixture's 512 positions sit well past `topk_blocks * block_size`.
        // Below that threshold the sparse path is bit-identical to dense and the
        // fixture would have graded nothing.
        return &arch::gqa::f32_backend();

    case AttentionFamily::Mla:
        return &arch::mla::f32_backend();

    case AttentionFamily::MlaDsa:
        // DSA is MLA plus a sparse key indexer, so it IS the MLA backend — which
        // branches on the family internally to select keys before the softmax.
        //
        // This returned nullptr until the indexer existed, and that was the right
        // answer while it did not: sharing the backend then would have run GLM-5.2
        // as dense attention — finite, plausible, and not the model that was
        // asked for. `arch_supported` is derived from this function rather than
        // declared, so flipping this line is what makes the plan's verdict change.
        return &arch::mla::f32_backend();

    case AttentionFamily::CompressedSparse:
        return &arch::compressed_sparse::f32_backend();

    case AttentionFamily::MlaKda:
        // SERVABLE, and this line moved only when there was evidence for it.
        //
        // It returned nullptr through the whole of the implementation work, on
        // the grounds that every check the family had was internal — invariants,
        // hand-computed traces, three-way agreement between prefill, streaming
        // and cached decode — and all of those would pass on an engine that is
        // self-consistently wrong. The bar was the one DSA had to clear:
        // token-exact against a reference oracle.
        //
        // `tests/fixtures/tiny/Kimi-Linear-Tiny` is that oracle, built by the
        // real `modeling_kimi_linear.py` against `fla`. Soma matches it at
        // max 2.21e-06 over 512 teacher-forced positions with 256 greedy tokens
        // exact — the same order as the seven families already through this
        // switch.
        //
        // It was worth waiting for. The oracle immediately found two defects
        // that every internal test had passed:
        //
        //   * `mla::f32_bind_layer` looked for the router's selection bias under
        //     a hardcoded "mlp." block. Kimi's MoE block is `block_sparse_moe`,
        //     the bind is optional, so the bias silently did not load and the
        //     router chose different experts — fluently.
        //   * this backend pointed `route` straight at `mla::f32_route`, which
        //     recovers that bias by casting the layer payload to
        //     `mla::F32AttnWeights`. A hybrid layer's payload is a
        //     `F32HybridWeights`. Undefined behaviour that read a garbage span
        //     length, failed a size check, and dropped the bias.
        //
        // Both moved the first MoE layer's output by 4.9e-02 against an input
        // that agreed to 7e-08, and neither is the kind of thing an invariant
        // written by the same author was ever going to catch.
        return &arch::kda::f32_backend();

    case AttentionFamily::GqaGdn:
        // SERVABLE, and — as with MlaKda — this line moved only once there was
        // evidence for it.
        //
        // There was never a near-enough backend to borrow. Routing this to `gqa`
        // would run 69 of 92 layers as dense softmax attention over a cache they
        // do not have; routing it to `kda` would apply a per-channel decay to a
        // per-head gate and index the recurrent state by the wrong head count.
        // Both produce finite, fluent, different models.
        //
        // `tests/fixtures/tiny/Qwen3.5-MoE-Tiny` is the oracle, built by
        // transformers' own `modeling_qwen3_5_moe.py` — pure torch on the CPU,
        // with no `fla` and no CUDA, which is what makes this family cheaper to
        // grade than the last hybrid was. Soma matches it at 4.58e-06 over 512
        // teacher-forced positions with 256 greedy tokens exact.
        //
        // It was worth waiting for. The oracle found a defect on the first run
        // that no internal invariant would ever have caught:
        // `Qwen3_5MoeRMSNorm` applies `x_hat * (1 + weight)`, not `x_hat *
        // weight`. Every norm in the model — layer norms, q/k norms, the final
        // norm, but NOT the gated norm inside the linear block, which is a
        // different class using the plain form — was scaled by a weight centred
        // on zero instead of on one. Nothing failed to load, every shape agreed,
        // and the logits were wrong from layer 0 at 1.9e+00.
        return &arch::gdn::f32_backend();

    case AttentionFamily::Unknown:
        return nullptr;
    }
    return nullptr;
}

const SpeculativeBackend* resolve_speculative_backend(const ArchIr& arch) noexcept {
    if (!arch.speculative.present) return nullptr;
    switch (arch.speculative.method) {
    case SpeculativeMethod::DSpark:
        return &arch::deepseek_dspark::backend();
    case SpeculativeMethod::Mtp:
        // Unreachable in practice — the guard above returns on `!present`, and
        // nothing sets `present` for a plain MTP head. Spelled out rather than
        // left to the trailing `return nullptr` so that the day something DOES
        // implement one, this switch is where the compiler points.
        return nullptr;
    case SpeculativeMethod::None:
        return nullptr;
    }
    return nullptr;
}

const AttentionBackend* resolve_attention_backend(AttentionFamily family) noexcept {
    switch (family) {
    case AttentionFamily::Mha:
    case AttentionFamily::Gqa:
        return &arch::gqa::attention_backend();
    case AttentionFamily::GqaBsa:
        // GQA's sizing plus the indexer's, which is why this is not simply the
        // `Gqa` case falling through.
        //
        // Two corrections ride on it, in opposite directions, and both land on
        // the quantity the verdict is computed from. The K plane is WIDER than
        // GQA's by `index_head_dim` per position, because the indexer's key must
        // be cached; and the resident half is larger by a query projection, a
        // key projection and two norms on every indexed layer. Sharing GQA's
        // descriptor would under-report the first by 128 floats per token per
        // layer and the second by ~3.9 M parameters across 57 layers.
        return &arch::gqa::attention_backend();

    case AttentionFamily::Mla:
        return &arch::mla::attention_backend();

    case AttentionFamily::MlaDsa:
        // DSA's attention IS MLA plus a learned sparse key indexer, so MLA's
        // SIZING is the right answer — including the indexer, which
        // weight_bytes_per_layer amortises over the stack from
        // AttentionSpec::dsa (roadmap D22: 0.90x -> 1.00x of the real tensors).
        //
        // This instance still carries no EXECUTION members, and that is the whole
        // of what it means. Running a DSA model is answered by
        // resolve_f32_backend — which now says yes — and by the fp32 path, which
        // selects keys before the softmax. Nothing can accidentally execute dense
        // attention through this pointer, because there is nothing here to call.
        return &arch::mla::attention_backend();

    case AttentionFamily::CompressedSparse:
        return &arch::compressed_sparse::attention_backend();

    case AttentionFamily::MlaKda:
        // Its OWN backend, not MLA's — which is the whole reason a family that
        // cannot yet execute still needs one here.
        //
        // MLA's sizing would charge 93 layers of latent cache for a stack that
        // has 24, and charge nothing for the 69 recurrent states that exist
        // whether or not anything can run them. At 1M context that is a ~4x
        // over-count on the quantity the verdict turns on, so the planner would
        // refuse a model that fits — a wrong answer arrived at confidently,
        // rather than the honest "not yet" that `resolve_f32_backend` returns.
        return &arch::kda::attention_backend();

    case AttentionFamily::GqaGdn:
        // Its OWN backend, and — as with MlaKda — that is precisely why a family
        // that cannot yet execute still needs one here.
        //
        // GQA's sizing would charge 92 layers of full K/V for a stack that has
        // 23, and charge nothing for the 69 recurrent states that exist whether
        // or not anything can run them. At the reference config's 262144 context
        // that is 184.0 GiB against 46.55 GiB — a 3.95x over-count on the quantity
        // the verdict turns on, so the planner would refuse a model that fits. A
        // wrong answer arrived at confidently, rather than the honest "not yet"
        // that `resolve_f32_backend` returns.
        return &arch::gdn::attention_backend();

    case AttentionFamily::Unknown:
        return nullptr;
    }
    return nullptr;
}

} // namespace soma
