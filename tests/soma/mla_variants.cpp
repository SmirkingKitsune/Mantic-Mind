// Soma — MLA's two variants: NoPE, and the sigmoid output gate.
//
// Both were added for Kimi-K3's full-attention layers, and both are the kind of
// change that the existing MLA tests cannot see: those fixtures configure
// neither flag, so they prove only that the DEFAULT path still behaves. A
// variant guarded by an IR flag nothing sets is a variant nobody has run.
//
// Each is pinned by a consequence rather than by re-deriving the arithmetic:
//
//   NoPE        the output must not depend on rope.theta, because nothing is
//               rotated. Under a missed guard it does — and only away from
//               position 0, where every rotation is the identity for any theta.
//   output gate the gate multiplies the concatenated heads BEFORE o_proj, and
//               o_proj is linear, so a uniform gate of exactly one half must
//               halve the output. Gating after o_proj, or not at all, does not.

#include "soma/arch/mla.hpp"
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

constexpr std::uint32_t Dm = 16;
constexpr std::uint32_t H = 2;
// ROPE is 4, not 2, and that is a property of the TEST rather than of the model.
//
// The inverse frequencies for a rotated span of width R are theta^(-2i/R) for
// i < R/2. At R = 2 there is a single pair at i = 0, whose frequency is
// theta^0 = 1 for every theta — so the theta control below passes trivially and
// proves nothing. Two pairs are the minimum at which theta is observable.
constexpr std::uint32_t NOPE = 4, ROPE = 4, VD = 4, LORA = 8;
constexpr std::uint32_t QK = NOPE + ROPE;
constexpr std::uint32_t T = 5;

// The two thetas the control compares, and they are far apart on purpose.
//
// For a rope span of width R the frequencies are theta^(-2i/R). The i = 0 pair
// is theta^0 = 1 whatever theta is, so only the remaining pairs carry any
// theta dependence at all — and between 10000 and 500000 the i = 1 pair moves
// from 0.010 to 0.0014, which over five positions perturbs the output by less
// than the tolerance. The control then fails while the code is correct.
//
// 2.0 against 10000.0 moves that pair from 0.707 to 0.010, which is visible.
// The lower bound is 1.0: `yarn_inv_freq` treats anything at or below it as
// unset and substitutes 10000.
constexpr float kThetaLow = 2.0f;
constexpr float kThetaHigh = 10000.0f;

soma::ArchIr make_arch(bool nope, bool gate, float theta) {
    soma::ArchIr a;
    a.attention.family = soma::AttentionFamily::Mla;
    a.topology.n_layers = 1;
    a.topology.d_model = Dm;
    a.topology.vocab_size = 32;
    a.topology.layer_kinds.assign(1, soma::LayerKind::Dense);
    a.rms_norm_eps = 1e-5f;
    a.attention.n_heads = H;
    a.attention.n_kv_heads = H;
    a.attention.head_dim = QK;
    a.attention.rope.theta = theta;
    a.attention.rope.partial_dim = ROPE;
    auto& m = a.attention.mla;
    m.kv_lora_rank = LORA;
    m.q_lora_rank = 0; // project Q directly — the V2-Lite shape, and the simpler one
    m.qk_nope_head_dim = NOPE;
    m.qk_rope_head_dim = ROPE;
    m.v_head_dim = VD;
    m.nope = nope;
    m.output_gate = gate;
    return a;
}

float syn(std::size_t i, float phase) {
    return 0.5f * std::sin(0.61f * static_cast<float>(i) + phase) +
           0.25f * std::cos(0.23f * static_cast<float>(i));
}

struct Store {
    std::vector<float> q, kva, kvn, kvb, o, g;
};

void fill(std::vector<float>& d, std::size_t n, float phase) {
    d.resize(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = syn(i, phase);
}

/// `gate_w` is filled by the caller so the gate's VALUE can be controlled.
soma::arch::mla::F32AttnWeights build(Store& s, bool gate) {
    fill(s.q, static_cast<std::size_t>(H) * QK * Dm, 0.3f);
    fill(s.kva, static_cast<std::size_t>(LORA + ROPE) * Dm, 1.2f);
    fill(s.kvb, static_cast<std::size_t>(H) * (NOPE + VD) * LORA, 2.4f);
    fill(s.o, static_cast<std::size_t>(Dm) * H * VD, 3.6f);
    s.kvn.assign(LORA, 1.0f);

    soma::arch::mla::F32AttnWeights w;
    w.q_proj = soma::WeightRef::from_f32(s.q, H * QK, Dm);
    w.kv_a_proj = soma::WeightRef::from_f32(s.kva, LORA + ROPE, Dm);
    w.kv_a_norm = s.kvn;
    w.kv_b_proj = soma::WeightRef::from_f32(s.kvb, H * (NOPE + VD), LORA);
    w.o_proj = soma::WeightRef::from_f32(s.o, Dm, H * VD);
    if (gate) w.out_gate = soma::WeightRef::from_f32(s.g, H * VD, Dm);
    return w;
}

int run(const soma::ArchIr& arch,
        const soma::arch::mla::F32AttnWeights& w,
        const std::vector<float>& x,
        std::vector<float>& out) {
    soma::F32LayerWeights lw;
    lw.attn.adopt(const_cast<soma::arch::mla::F32AttnWeights*>(&w), [](void*) {});
    soma::F32Workspace ws;
    out.assign(static_cast<std::size_t>(T) * Dm, 0.0f);
    return soma::arch::mla::f32_attention(arch, lw, x.data(), T, ws, out.data()) ==
                   soma::StatusCode::Ok
               ? 0
               : 1;
}

} // namespace

int main() {
    std::vector<float> x(static_cast<std::size_t>(T) * Dm);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = syn(i, 7.7f);

    // ── 1. NoPE ignores theta; rotated MLA does not ──────────────────────────
    {
        Store s;
        const auto w = build(s, /*gate=*/false);
        std::vector<float> a, b;
        CHECK(run(make_arch(/*nope=*/true, false, kThetaLow), w, x, a) == 0);
        CHECK(run(make_arch(/*nope=*/true, false, kThetaHigh), w, x, b) == 0);
        for (std::size_t i = 0; i < a.size(); ++i) CHECK(close(a[i], b[i]));

        // The control: with the rotation restored, theta matters. Without this
        // the check above would also pass on a model that simply ignores theta
        // everywhere, which would prove nothing about the guard.
        std::vector<float> c, d;
        CHECK(run(make_arch(/*nope=*/false, false, kThetaLow), w, x, c) == 0);
        CHECK(run(make_arch(/*nope=*/false, false, kThetaHigh), w, x, d) == 0);
        bool differs = false;
        for (std::size_t i = 0; i < c.size(); ++i)
            if (!close(c[i], d[i], 1e-3f)) differs = true;
        CHECK(differs);

        // And NoPE is not merely "some theta": it is a different function from
        // the rotated form at the same theta. Position 0 is exempt — every
        // rotation is the identity there — so the difference is looked for over
        // the later positions, which is exactly where a missed guard hides.
        bool differs_from_rotated = false;
        for (std::size_t i = Dm; i < a.size(); ++i)
            if (!close(a[i], c[i], 1e-3f)) differs_from_rotated = true;
        CHECK(differs_from_rotated);
    }

    // ── 2. the output gate scales the heads before o_proj ────────────────────
    {
        Store s0;
        const auto ungated = build(s0, /*gate=*/false);
        std::vector<float> plain;
        CHECK(run(make_arch(true, /*gate=*/false, kThetaHigh), ungated, x, plain) == 0);

        // g_proj == 0 => sigmoid(0) == 0.5 for every element. o_proj is linear,
        // so halving its input must halve its output EXACTLY.
        Store s1;
        s1.g.assign(static_cast<std::size_t>(H) * VD * Dm, 0.0f);
        const auto gated = build(s1, /*gate=*/true);
        std::vector<float> half;
        CHECK(run(make_arch(true, /*gate=*/true, kThetaHigh), gated, x, half) == 0);
        for (std::size_t i = 0; i < plain.size(); ++i) CHECK(close(half[i], 0.5f * plain[i], 1e-4f));

        // A saturating gate is the identity, which is the other end of the same
        // statement and rules out "the gate is applied but ignores its input".
        Store s2;
        s2.g.assign(static_cast<std::size_t>(H) * VD * Dm, 0.0f);
        // One large weight per output row, against an input whose first element
        // is positive, drives every gate deep into saturation.
        std::vector<float> xpos(static_cast<std::size_t>(T) * Dm, 0.0f);
        for (std::uint32_t t = 0; t < T; ++t) xpos[static_cast<std::size_t>(t) * Dm] = 1.0f;
        for (std::uint32_t r = 0; r < H * VD; ++r) s2.g[static_cast<std::size_t>(r) * Dm] = 40.0f;
        const auto saturated = build(s2, /*gate=*/true);
        std::vector<float> on, off;
        CHECK(run(make_arch(true, /*gate=*/true, kThetaHigh), saturated, xpos, on) == 0);
        Store s3;
        const auto no_gate = build(s3, /*gate=*/false);
        CHECK(run(make_arch(true, /*gate=*/false, kThetaHigh), no_gate, xpos, off) == 0);
        for (std::size_t i = 0; i < on.size(); ++i) CHECK(close(on[i], off[i], 1e-4f));
    }

    std::cout << "mla_variants: OK\n";
    return 0;
}
