// Soma — the block residual.
//
// Every `block_size`-th layer snapshots the residual stream onto a per-token
// stack; each layer then mixes over that stack with learned softmax scores. Two
// separable halves, tested separately:
//
//   THE MIX      is pinned by identities that hold whatever the reference does —
//                a single candidate must pass through untouched, identical
//                candidates must average to themselves, and a zero score vector
//                must give a uniform mean.
//   THE SEQUENCE is pinned by two traces computed BY HAND. With a zero score
//                vector the mixing collapses to an arithmetic mean, which makes
//                the whole layer loop hand-evaluable — so the expected values
//                below are derived, not transcribed from the implementation.
//
// The sharpest of them is the boundary reset. At a block boundary the reference
// sets `prefix_sum = None`: the layer does NOT carry its incoming residual
// forward, it restarts from its own attention output. Carrying it forward IS a
// different network, and case 2 separates the two.
//
// Zeroing the prefix rather than invalidating it is not — the next write is
// `prefix = branch` either way. That was checked by mutation rather than
// assumed, and no test here tries to distinguish them, because there is nothing
// to distinguish.

#include "soma/arch/kda.hpp"
#include "soma/arch_ir.hpp"
#include "soma/f32_model.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool check(bool value, const char* expression, int line) {
    if (value) return true;
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    return false;
}

#define CHECK(expr)                                                                                \
    do {                                                                                           \
        if (!check((expr), #expr, __LINE__)) return 1;                                             \
    } while (false)

bool close(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(a) + std::fabs(b));
}

constexpr std::uint32_t D = 4;
constexpr std::uint32_t T = 2;

namespace kda = soma::arch::kda;

soma::ArchIr make_arch(std::uint32_t block_size, std::uint32_t n_layers) {
    soma::ArchIr a;
    a.attention.family = soma::AttentionFamily::MlaKda;
    a.topology.n_layers = n_layers;
    a.topology.d_model = D;
    a.topology.vocab_size = 8;
    a.topology.layer_kinds.assign(n_layers, soma::LayerKind::Moe);
    a.rms_norm_eps = 1e-5f;
    a.block_residual.block_size = block_size;
    auto& k = a.attention.kda;
    k.n_heads = 1;
    k.head_dim = D;
    k.conv_kernel = 2;
    k.layer_kinds.assign(n_layers, soma::AttnLayerKind::Linear);
    return a;
}

/// Drive the hooks in exactly the order `forward_f32` calls them.
struct Harness {
    soma::ArchIr arch;
    kda::F32HybridWeights w;
    kda::F32HybridModel model;
    soma::F32LayerWeights lw;
    soma::ArchLayerPayload model_payload;
    soma::F32Workspace ws;
    std::vector<float> hidden;

    Harness(std::uint32_t block_size, std::uint32_t n_layers)
        : arch(make_arch(block_size, n_layers)) {
        // A zero score vector makes every candidate score 0, so softmax is
        // uniform and each mix is a plain arithmetic mean. That is what makes
        // the traces below hand-computable.
        w.attn_res_score.assign(D, 0.0f);
        w.mlp_res_score.assign(D, 0.0f);
        model.out_res_score.assign(D, 0.0f);
        lw.attn.adopt(&w, [](void*) {});
        model_payload.adopt(&model, [](void*) {});
    }

    /// `attn_is_identity` makes the attention branch equal to whatever
    /// `pre_attention` left in `hidden`.
    ///
    /// Without it the harness feeds a CONSTANT branch, and then nothing
    /// downstream depends on what `pre_attention` computed — so the pre-attention
    /// mix could be skipped, reordered, or fed the wrong candidate list and every
    /// trace would still match. A constant branch tests the bookkeeping and not
    /// the value.
    bool run(const std::vector<float>& x,
             const std::vector<float>& attn_branch,
             const std::vector<float>& ffn_branch,
             bool attn_is_identity = false) {
        hidden = x;
        if (kda::f32_begin_forward(arch, model_payload, nullptr, T, ws, hidden.data()) !=
            soma::StatusCode::Ok)
            return false;
        for (std::uint32_t l = 0; l < arch.topology.n_layers; ++l) {
            ws.current_layer = l;
            if (kda::f32_pre_attention(arch, lw, T, ws, hidden.data()) != soma::StatusCode::Ok)
                return false;
            const std::vector<float> branch = attn_is_identity ? hidden : attn_branch;
            if (kda::f32_merge_attention(arch, lw, branch.data(), T, ws, hidden.data()) !=
                soma::StatusCode::Ok)
                return false;
            if (kda::f32_pre_ffn(arch, lw, T, ws, hidden.data()) != soma::StatusCode::Ok)
                return false;
            if (kda::f32_merge_ffn(arch, lw, ffn_branch.data(), T, ws, hidden.data()) !=
                soma::StatusCode::Ok)
                return false;
        }
        return kda::f32_end_forward(arch, model_payload, T, ws, hidden.data()) ==
               soma::StatusCode::Ok;
    }
};

} // namespace

int main() {
    // ── 1. the mix ───────────────────────────────────────────────────────────
    {
        kda::BlockResidualState st;
        st.n_tokens = 1;
        st.width = D;
        st.n_blocks = 0;
        st.stack.assign(static_cast<std::size_t>(2) * D, 0.0f); // capacity for 2
        std::vector<float> score(D, 0.7f), prefix{1.0f, -2.0f, 3.0f, 0.5f}, out(D);

        // ONE candidate: softmax over a single element is 1, so the prefix must
        // come through UNCHANGED. Combining the RMS-normalized candidates —
        // rather than scoring with them and combining the raw ones — would
        // return the normalized prefix here, which is a different vector.
        kda::mix_block_residual(st, score, 1e-5f, prefix.data(), out.data());
        for (std::uint32_t i = 0; i < D; ++i) CHECK(close(out[i], prefix[i]));

        // Two IDENTICAL candidates average to themselves, whatever the weights.
        st.n_blocks = 1;
        for (std::uint32_t i = 0; i < D; ++i) st.stack[i] = prefix[i];
        kda::mix_block_residual(st, score, 1e-5f, prefix.data(), out.data());
        for (std::uint32_t i = 0; i < D; ++i) CHECK(close(out[i], prefix[i]));

        // A zero score vector scores every candidate 0, so the mix is a plain
        // mean — the property every trace below relies on.
        std::vector<float> zero(D, 0.0f), other{-1.0f, 4.0f, 0.0f, 2.5f};
        for (std::uint32_t i = 0; i < D; ++i) st.stack[i] = other[i];
        kda::mix_block_residual(st, zero, 1e-5f, prefix.data(), out.data());
        for (std::uint32_t i = 0; i < D; ++i) CHECK(close(out[i], 0.5f * (prefix[i] + other[i])));

        // Candidates are SCORED NORMALIZED. `2*v` and `v` normalize to the same
        // direction, so they must score equally and the mix must be their plain
        // mean, 1.5*v. Scoring the raw vectors would give `2*v` twice the dot
        // product and a different, larger weight.
        std::vector<float> v{0.3f, -0.9f, 1.4f, 0.2f}, twice(D);
        for (std::uint32_t i = 0; i < D; ++i) twice[i] = 2.0f * v[i];
        for (std::uint32_t i = 0; i < D; ++i) st.stack[i] = twice[i];
        kda::mix_block_residual(st, score, 1e-5f, v.data(), out.data());
        for (std::uint32_t i = 0; i < D; ++i) CHECK(close(out[i], 1.5f * v[i], 1e-4f));

        // The result is always a convex combination: never outside the range the
        // candidates span.
        for (std::uint32_t i = 0; i < D; ++i) st.stack[i] = other[i];
        kda::mix_block_residual(st, score, 1e-5f, prefix.data(), out.data());
        for (std::uint32_t i = 0; i < D; ++i) {
            const float lo = std::min(prefix[i], other[i]), hi = std::max(prefix[i], other[i]);
            CHECK(out[i] >= lo - 1e-5f && out[i] <= hi + 1e-5f);
        }
    }

    std::vector<float> x(static_cast<std::size_t>(T) * D);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = 0.5f + 0.25f * static_cast<float>(i);
    const std::vector<float> zeros(static_cast<std::size_t>(T) * D, 0.0f);

    // ── 2. four layers, block_size 2, zero branches ──────────────────────────
    //
    // Hand trace (uniform mixing, so every mix is a mean):
    //   L0  prefix=x; stack empty so no mix; 0%2==0 -> push x, prefix DROPPED
    //       merge_attn(0): prefix=0            (restart, NOT x+0)
    //       pre_ffn:       mix(0,[x]) = x/2
    //       merge_ffn(0):  prefix=0
    //   L1  prefix=0; mix(0,[x])=x/2; no push
    //       merge_attn(0): prefix=0;  pre_ffn: x/2;  merge_ffn(0): prefix=0
    //   L2  prefix=0; mix(0,[x])=x/2; 2%2==0 -> push 0, prefix DROPPED
    //       merge_attn(0): prefix=0;  pre_ffn: mix(0,[x,0])=x/3;  merge_ffn: 0
    //   L3  prefix=0; mix(0,[x,0])=x/3; no push; ... merge_ffn: prefix=0
    //   end mix(0,[x,0]) = x/3
    //
    // The boundary reset is what makes L0 end at 0 rather than at x: had the
    // prefix been carried forward, every value below would be larger.
    {
        Harness h(/*block_size=*/2, /*n_layers=*/4);
        CHECK(h.run(x, zeros, zeros));
        for (std::size_t i = 0; i < x.size(); ++i) CHECK(close(h.hidden[i], x[i] / 3.0f, 1e-5f));

        const auto* st = h.ws.arch_state.as<kda::BlockResidualState>();
        CHECK(st != nullptr && st->n_blocks == 2);
        // Snapshot 0 is the residual entering layer 0; snapshot 1 is what entered
        // layer 2, which the trace says is zero.
        for (std::uint32_t t = 0; t < T; ++t) {
            for (std::uint32_t i = 0; i < D; ++i) {
                CHECK(close(st->stack[(static_cast<std::size_t>(t) * 2 + 0) * D + i],
                            x[static_cast<std::size_t>(t) * D + i]));
                CHECK(close(st->stack[(static_cast<std::size_t>(t) * 2 + 1) * D + i], 0.0f));
            }
        }
    }

    // ── 3. two layers, block_size 4, non-zero branches ───────────────────────
    //
    // Only layer 0 is a boundary, so this exercises ACCUMULATION as well:
    //   L0  push x, prefix dropped;  merge_attn(a): prefix=a
    //       pre_ffn: (x+a)/2;        merge_ffn(f):  prefix=a+f
    //   L1  mix(a+f,[x]) = (x+a+f)/2; no push
    //       merge_attn(a): prefix=2a+f;  pre_ffn: (x+2a+f)/2;  merge_ffn(f): 2a+2f
    //   end mix(2a+2f,[x]) = (x+2a+2f)/2
    {
        std::vector<float> a(static_cast<std::size_t>(T) * D), f(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = 0.1f * static_cast<float>(i) - 0.3f;
            f[i] = 0.6f - 0.2f * static_cast<float>(i);
        }
        Harness h(/*block_size=*/4, /*n_layers=*/2);
        CHECK(h.run(x, a, f));
        for (std::size_t i = 0; i < x.size(); ++i)
            CHECK(close(h.hidden[i], 0.5f * (x[i] + 2.0f * a[i] + 2.0f * f[i]), 1e-5f));

        const auto* st = h.ws.arch_state.as<kda::BlockResidualState>();
        CHECK(st != nullptr && st->n_blocks == 1);
    }

    // ── 3b. identity attention, so the PRE-ATTENTION mix is observable ───────
    //
    // Cases 2 and 3 feed a constant attention branch, which means the value
    // `pre_attention` computes never reaches anything — only its bookkeeping
    // does. With the branch equal to that value, the mix's position relative to
    // the snapshot push becomes visible: at layer 2 the correct order mixes over
    // {x} before pushing, while pushing first mixes over {x, prefix} and shifts
    // the result toward the prefix.
    //
    // Hand trace, block_size 2, four layers, ffn branch zero, uniform mixing:
    //   L0  prefix=x; stack empty -> hidden=x; push x; prefix dropped
    //       merge_attn(x): prefix=x;   pre_ffn: mix(x,[x])=x;      merge_ffn: x
    //   L1  prefix=x; mix(x,[x])=x;    merge_attn(x): prefix=2x
    //       pre_ffn: mix(2x,[x])=1.5x; merge_ffn: prefix=2x
    //   L2  prefix=2x; mix(2x,[x])=1.5x; push 2x -> [x,2x]; prefix dropped
    //       merge_attn(1.5x): prefix=1.5x
    //       pre_ffn: mix(1.5x,[x,2x])=1.5x;  merge_ffn: prefix=1.5x
    //   L3  prefix=1.5x; mix(1.5x,[x,2x])=1.5x; merge_attn(1.5x): prefix=3x
    //       pre_ffn: mix(3x,[x,2x])=2x;       merge_ffn: prefix=3x
    //   end mix(3x,[x,2x]) = 2x
    {
        Harness h(/*block_size=*/2, /*n_layers=*/4);
        CHECK(h.run(x, zeros, zeros, /*attn_is_identity=*/true));
        for (std::size_t i = 0; i < x.size(); ++i) CHECK(close(h.hidden[i], 2.0f * x[i], 1e-5f));
    }

    // ── 4. block_size 0 is the ordinary residual stream ──────────────────────
    //
    // The hooks are registered unconditionally, so every family that does not
    // use them still has to get the plain `hidden += branch` back.
    {
        std::vector<float> a(static_cast<std::size_t>(T) * D, 0.25f),
            f(static_cast<std::size_t>(T) * D, -0.5f);
        Harness h(/*block_size=*/0, /*n_layers=*/3);
        CHECK(h.run(x, a, f));
        for (std::size_t i = 0; i < x.size(); ++i)
            CHECK(close(h.hidden[i], x[i] + 3.0f * (a[i] + f[i]), 1e-5f));
        // …and no state was allocated for a model that has no block residual.
        CHECK(h.ws.arch_state.as<kda::BlockResidualState>() == nullptr);
    }

    // ── 5. a second forward does not inherit the first one's stack ───────────
    {
        Harness h(/*block_size=*/2, /*n_layers=*/4);
        CHECK(h.run(x, zeros, zeros));
        std::vector<float> first = h.hidden;
        CHECK(h.run(x, zeros, zeros));
        for (std::size_t i = 0; i < first.size(); ++i) CHECK(close(h.hidden[i], first[i]));
        const auto* st = h.ws.arch_state.as<kda::BlockResidualState>();
        CHECK(st != nullptr && st->n_blocks == 2); // not 4
    }

    std::cout << "block_residual: OK\n";
    return 0;
}
