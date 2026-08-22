// Soma — the latent MoE.
//
// Routed experts running in a space narrower than the residual stream. Added for
// Kimi-K3, but it is a property of the FFN and not of any attention family, so it
// is exercised here through GQA — the simplest backend that reaches the expert
// loop — and through the real `forward_f32`, not a reimplementation of it.
//
// Four claims, each with a way to be wrong that still produces finite logits:
//
//   * the wrapping is transparent at identity — a latent MoE whose projections
//     are the identity and whose norm is off must equal no latent MoE at all;
//   * the SHARED expert lives outside it, reading the full-width input and
//     adding after the up-projection;
//   * a genuinely narrower latent changes the answer, i.e. the projections are
//     actually applied rather than skipped;
//   * the norm applies to the COMBINED top-k output, before the up-projection.

#include "soma/arch/gqa.hpp"
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

constexpr std::uint32_t D = 8, HEADS = 2, HD = 4, VOCAB = 6;
constexpr std::uint32_t NEXP = 4, TOPK = 2, FI = 6, SI = 6;

float syn(std::size_t i, float phase) {
    return 0.55f * std::sin(0.73f * static_cast<float>(i) + phase) +
           0.21f * std::cos(0.29f * static_cast<float>(i));
}

void fill(std::vector<float>& v, std::size_t n, float phase) {
    v.resize(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = syn(i, phase);
}

/// Everything the hand-built model points into. One instance per model, kept
/// alive for as long as the model is used — every WeightRef is a view.
struct Store {
    std::vector<float> embed, out_norm, in_norm, post_norm;
    std::vector<float> q, k, v, o, router;
    std::vector<float> eg[NEXP], eu[NEXP], ed[NEXP];
    std::vector<float> sg, su, sd;
    std::vector<float> down, up, lnorm;
    soma::arch::gqa::F32AttnWeights attn;
};

soma::ArchIr make_arch(std::uint32_t latent_width, bool latent_norm) {
    soma::ArchIr a;
    a.attention.family = soma::AttentionFamily::Gqa;
    // TWO layers, not one, and that is load-bearing for the test rather than
    // for the model. The latent staging buffers live in the workspace and are
    // reused across layers, so a single-layer model never sees a DIRTY one —
    // `ensure_latent` zeroes on first allocation and the leftover from layer 0
    // is the only thing that can expose a missing clear.
    a.topology.n_layers = 2;
    a.topology.d_model = D;
    a.topology.vocab_size = VOCAB;
    a.topology.layer_kinds = {soma::LayerKind::Moe, soma::LayerKind::Moe};
    a.rms_norm_eps = 1e-5f;
    a.attention.n_heads = HEADS;
    a.attention.n_kv_heads = HEADS;
    a.attention.head_dim = HD;
    a.attention.rope.theta = 10000.0f;
    a.router.n_experts = NEXP;
    a.router.top_k = TOPK;
    a.router.score_fn = soma::ScoreFn::Softmax;
    a.router.normalize_topk = true;
    a.router.n_shared_experts = 1;
    a.ffn.activation = soma::Activation::SwiGlu;
    a.ffn.expert_intermediate = FI;
    a.ffn.dense_intermediate = FI;
    a.ffn.shared_intermediate = SI;
    a.ffn.routed_expert_hidden = latent_width;
    a.ffn.routed_expert_norm = latent_norm;
    return a;
}

/// `identity` builds down/up as identity matrices, so the latent wrapping is a
/// no-op by construction; otherwise they are ordinary dense projections.
void build(soma::F32Model& m,
           Store& s,
           std::uint32_t latent_width,
           bool identity,
           bool zero_routed,
           bool zero_norm_weight) {
    const auto ew = latent_width != 0 ? latent_width : D;

    fill(s.embed, static_cast<std::size_t>(VOCAB) * D, 0.4f);
    s.out_norm.assign(D, 1.0f);
    s.in_norm.assign(D, 1.0f);
    s.post_norm.assign(D, 1.0f);
    fill(s.q, static_cast<std::size_t>(HEADS) * HD * D, 1.1f);
    fill(s.k, static_cast<std::size_t>(HEADS) * HD * D, 1.6f);
    fill(s.v, static_cast<std::size_t>(HEADS) * HD * D, 2.2f);
    fill(s.o, static_cast<std::size_t>(D) * HEADS * HD, 2.8f);
    fill(s.router, static_cast<std::size_t>(NEXP) * D, 3.3f);
    fill(s.sg, static_cast<std::size_t>(SI) * D, 4.1f);
    fill(s.su, static_cast<std::size_t>(SI) * D, 4.6f);
    fill(s.sd, static_cast<std::size_t>(D) * SI, 5.2f);

    for (std::uint32_t e = 0; e < NEXP; ++e) {
        fill(s.eg[e], static_cast<std::size_t>(FI) * ew, 6.0f + 0.3f * static_cast<float>(e));
        fill(s.eu[e], static_cast<std::size_t>(FI) * ew, 7.0f + 0.3f * static_cast<float>(e));
        fill(s.ed[e], static_cast<std::size_t>(ew) * FI, 8.0f + 0.3f * static_cast<float>(e));
        if (zero_routed) {
            std::fill(s.eg[e].begin(), s.eg[e].end(), 0.0f);
            std::fill(s.eu[e].begin(), s.eu[e].end(), 0.0f);
            std::fill(s.ed[e].begin(), s.ed[e].end(), 0.0f);
        }
    }

    if (latent_width != 0) {
        s.down.assign(static_cast<std::size_t>(ew) * D, 0.0f);
        s.up.assign(static_cast<std::size_t>(D) * ew, 0.0f);
        if (identity) {
            for (std::uint32_t i = 0; i < std::min(ew, D); ++i) {
                s.down[static_cast<std::size_t>(i) * D + i] = 1.0f;
                s.up[static_cast<std::size_t>(i) * ew + i] = 1.0f;
            }
        } else {
            fill(s.down, static_cast<std::size_t>(ew) * D, 9.1f);
            fill(s.up, static_cast<std::size_t>(D) * ew, 9.7f);
        }
        s.lnorm.assign(ew, zero_norm_weight ? 0.0f : 1.0f);
    }

    m.embed = soma::WeightRef::from_f32(s.embed, VOCAB, D);
    m.out_head = m.embed;
    m.out_norm = s.out_norm;

    s.attn.q_proj = soma::WeightRef::from_f32(s.q, HEADS * HD, D);
    s.attn.k_proj = soma::WeightRef::from_f32(s.k, HEADS * HD, D);
    s.attn.v_proj = soma::WeightRef::from_f32(s.v, HEADS * HD, D);
    s.attn.o_proj = soma::WeightRef::from_f32(s.o, D, HEADS * HD);

    m.layers.resize(m.arch.topology.n_layers);
    for (auto& lw : m.layers) {
        lw.kind = soma::LayerKind::Moe;
        lw.input_norm = s.in_norm;
        lw.post_attn_norm = s.post_norm;
        // Every layer borrows the same payload; the deleter is a no-op because
        // `Store` owns it.
        lw.attn.adopt(&s.attn, [](void*) {});
        lw.router = s.router;
        lw.expert_gate.resize(NEXP);
        lw.expert_up.resize(NEXP);
        lw.expert_down.resize(NEXP);
        for (std::uint32_t e = 0; e < NEXP; ++e) {
            lw.expert_gate[e] = soma::WeightRef::from_f32(s.eg[e], FI, ew);
            lw.expert_up[e] = soma::WeightRef::from_f32(s.eu[e], FI, ew);
            lw.expert_down[e] = soma::WeightRef::from_f32(s.ed[e], ew, FI);
        }
        lw.shared_gate = soma::WeightRef::from_f32(s.sg, SI, D);
        lw.shared_up = soma::WeightRef::from_f32(s.su, SI, D);
        lw.shared_down = soma::WeightRef::from_f32(s.sd, D, SI);
        if (latent_width != 0) {
            lw.latent_down = soma::WeightRef::from_f32(s.down, ew, D);
            lw.latent_up = soma::WeightRef::from_f32(s.up, D, ew);
            if (!s.lnorm.empty() && m.arch.ffn.routed_expert_norm) lw.latent_norm = s.lnorm;
        }
    }
}

int run(std::uint32_t latent_width,
        bool latent_norm,
        bool identity,
        bool zero_routed,
        bool zero_norm_weight,
        std::vector<float>& logits) {
    soma::F32Model m;
    m.arch = make_arch(latent_width, latent_norm);
    Store s;
    build(m, s, latent_width, identity, zero_routed, zero_norm_weight);
    const std::vector<soma::TokenId> toks{1, 3, 0, 4};
    soma::F32Workspace ws;
    return soma::forward_f32(m, toks, ws, logits).ok() ? 0 : 1;
}

} // namespace

int main() {
    // ── 1. no latent MoE: the baseline every other case is measured against ──
    std::vector<float> base;
    CHECK(run(0, false, false, false, false, base) == 0);
    {
        float mag = 0.0f;
        for (const auto f : base) mag += std::fabs(f);
        CHECK(mag > 1e-3f); // the model computed something
    }

    // ── 2. identity projections are transparent ──────────────────────────────
    //
    // A latent MoE at the SAME width, with identity down/up and no norm, is the
    // ordinary FFN wrapped in two no-ops. Any difference is the wrapping itself
    // being wrong — a routed output landing in the wrong buffer, the shared
    // expert double-counted, the up-projection accumulating instead of
    // overwriting.
    {
        std::vector<float> got;
        CHECK(run(D, false, /*identity=*/true, false, false, got) == 0);
        CHECK(got.size() == base.size());
        for (std::size_t i = 0; i < base.size(); ++i) CHECK(close(got[i], base[i], 1e-4f));
    }

    // ── 3. the shared expert is OUTSIDE the latent space ─────────────────────
    //
    // With every routed expert zeroed, the layer's FFN output is the shared
    // expert alone — and the shared expert reads the full-width input and adds
    // after the up-projection, so it cannot depend on the latent width or on the
    // projections. Running it through the latent space instead would change this.
    {
        std::vector<float> no_latent, narrow, wide;
        CHECK(run(0, false, false, /*zero_routed=*/true, false, no_latent) == 0);
        CHECK(run(4, false, /*identity=*/false, /*zero_routed=*/true, false, narrow) == 0);
        CHECK(run(D, false, /*identity=*/false, /*zero_routed=*/true, false, wide) == 0);
        for (std::size_t i = 0; i < no_latent.size(); ++i) {
            CHECK(close(narrow[i], no_latent[i], 1e-4f));
            CHECK(close(wide[i], no_latent[i], 1e-4f));
        }
        // …and the routed experts were genuinely carrying the difference, so
        // check 3 is not passing because everything is zero.
        bool routed_mattered = false;
        for (std::size_t i = 0; i < base.size(); ++i)
            if (!close(base[i], no_latent[i], 1e-3f)) routed_mattered = true;
        CHECK(routed_mattered);
    }

    // ── 4. a narrower latent actually narrows ────────────────────────────────
    {
        std::vector<float> narrow;
        CHECK(run(4, false, /*identity=*/false, false, false, narrow) == 0);
        bool differs = false;
        for (std::size_t i = 0; i < base.size(); ++i) {
            CHECK(std::isfinite(narrow[i]));
            if (!close(narrow[i], base[i], 1e-3f)) differs = true;
        }
        CHECK(differs);
    }

    // ── 5. the norm sits on the COMBINED routed output ───────────────────────
    //
    // A zero norm weight annihilates whatever it is applied to. Applied to the
    // combined top-k result before the up-projection — where it belongs — the
    // routed half vanishes entirely and the layer reduces to its shared expert,
    // which is exactly the run from check 3.
    //
    // If the norm were applied anywhere else (per expert, or after the
    // up-projection, or to the shared sum as well) this would not hold: after
    // the up-projection it would zero the shared expert too, and per-expert it
    // would still be zero but for a reason this comparison would not distinguish
    // — hence the shared-only reference rather than a bare "is it zero".
    {
        std::vector<float> shared_only, zeroed_norm;
        CHECK(run(0, false, false, /*zero_routed=*/true, false, shared_only) == 0);
        CHECK(run(4, /*latent_norm=*/true, false, false, /*zero_norm_weight=*/true,
                  zeroed_norm) == 0);
        for (std::size_t i = 0; i < shared_only.size(); ++i)
            CHECK(close(zeroed_norm[i], shared_only[i], 1e-4f));

        // With a unit norm weight the routed half comes back, so check 5 is not
        // passing because the norm is ignored.
        std::vector<float> unit_norm;
        CHECK(run(4, /*latent_norm=*/true, false, false, /*zero_norm_weight=*/false,
                  unit_norm) == 0);
        bool norm_mattered = false;
        for (std::size_t i = 0; i < unit_norm.size(); ++i)
            if (!close(unit_norm[i], zeroed_norm[i], 1e-3f)) norm_mattered = true;
        CHECK(norm_mattered);
    }

    std::cout << "latent_moe: OK\n";
    return 0;
}
