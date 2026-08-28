// Soma — the Gated DeltaNet kernels, isolated.
//
// The oracle in `tests/fixtures/tiny/Qwen3.5-MoE-Tiny` already grades the whole
// model, and it is the check that matters. What it CANNOT do is say which of the
// four operators is wrong when it fails: every one of them feeds the next, and a
// whole-model diff localises a defect to "the linear layer".
//
// So each operator is checked here against a closed form or an exact algebraic
// property, chosen so that the plausible misreading fails:
//
//   gate          the sign and the composition order. `-exp(A_log) *
//                 softplus(a + dt_bias)` — dropping the negation makes `exp(g)`
//                 a GROWTH factor and the state diverges rather than decays.
//   short_conv    which tap multiplies the current token, and that the window
//                 carries the INPUT rather than the activated output.
//   step          the four orderings — decay, predict, correct, read — and the
//                 key/value head broadcast.
//   gated_rmsnorm norm before gate, SiLU not sigmoid, weight shared across heads.
//
// Plus the property that no unit test of the parts can establish and no oracle
// run exercises directly: prefill and stepwise decode through
// `f32_linear_layer` must produce the same thing, because they are the same
// recurrence with the state coming from a different place.

#include "soma/arch/gdn.hpp"
#include "soma/quant_format.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

bool close(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

void check(bool ok, const char* what, int line) {
    if (ok) return;
    std::cerr << "FAIL line " << line << ": " << what << '\n';
    ++g_failures;
}

#define CHECK(expr) check((expr), #expr, __LINE__)

float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float siluf(float x) { return x * sigmoidf(x); }
float softplusf(float x) { return std::log1p(std::exp(x)); }

/// A tiny but NON-DEGENERATE spec: the key and value head counts differ, so the
/// broadcast is live. Equal counts would let an implementation that ignores the
/// broadcast pass every assertion below.
soma::ArchIr make_arch(std::uint32_t n_k_heads = 2,
                       std::uint32_t n_v_heads = 6,
                       std::uint32_t head_dim = 4,
                       std::uint32_t conv_kernel = 4) {
    soma::ArchIr a;
    a.schema_version = soma::kArchIrSchemaVersion;
    a.rms_norm_eps = 1e-6f;
    a.topology.n_layers = 1;
    a.topology.d_model = 8;
    a.topology.vocab_size = 32;
    a.topology.layer_kinds = {soma::LayerKind::Moe};
    a.attention.family = soma::AttentionFamily::GqaGdn;
    a.attention.n_heads = 4;
    a.attention.n_kv_heads = 2;
    a.attention.head_dim = 4;
    a.attention.rope.partial_dim = 2;
    auto& g = a.attention.gdn;
    g.n_k_heads = n_k_heads;
    g.n_v_heads = n_v_heads;
    g.head_k_dim = head_dim;
    g.head_v_dim = head_dim;
    g.conv_kernel = conv_kernel;
    g.layer_kinds = {soma::AttnLayerKind::Linear};
    return a;
}

// ── gate ─────────────────────────────────────────────────────────────────────

void test_gate() {
    const auto arch = make_arch();
    const auto nv = arch.attention.gdn.n_v_heads;
    std::vector<float> a_log(nv), dt(nv), raw(nv), out(nv);
    for (std::uint32_t h = 0; h < nv; ++h) {
        a_log[h] = -0.3f + 0.2f * static_cast<float>(h);
        dt[h] = 0.5f - 0.1f * static_cast<float>(h);
        raw[h] = 0.7f * static_cast<float>(h) - 1.0f;
    }
    soma::arch::gdn::gate(arch, a_log.data(), dt.data(), raw.data(), out.data());

    for (std::uint32_t h = 0; h < nv; ++h) {
        const float want = -std::exp(a_log[h]) * softplusf(raw[h] + dt[h]);
        CHECK(close(out[h], want, 1e-6f));
        // STRICTLY NEGATIVE, always: softplus is positive and exp is positive.
        // This is what makes `exp(g)` a contraction. Drop the sign and the state
        // grows without bound — which on a short test sequence still produces
        // finite numbers.
        CHECK(out[h] < 0.0f);
    }

    // dt_bias is INSIDE the softplus, not added after it. The two differ
    // wherever softplus is not the identity, which is everywhere near zero.
    std::vector<float> zero_dt(nv, 0.0f), out2(nv);
    soma::arch::gdn::gate(arch, a_log.data(), zero_dt.data(), raw.data(), out2.data());
    CHECK(!close(out[0], out2[0], 1e-4f));
}

// ── short_conv ───────────────────────────────────────────────────────────────

void test_short_conv() {
    constexpr std::uint32_t width = 3, kernel = 4, carried = kernel - 1;
    std::vector<float> w(width * kernel);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = 0.1f * static_cast<float>(i + 1);

    std::vector<float> state(width * carried, 0.0f);
    std::vector<float> out(width);

    // A zero state means only the LAST tap fires. That is what identifies which
    // end of the window the current token sits at: with the taps reversed this
    // would multiply by w[0] instead.
    const std::vector<float> x0{1.0f, 2.0f, 3.0f};
    soma::arch::gdn::short_conv(width, kernel, w.data(), x0.data(), state.data(), out.data());
    for (std::uint32_t c = 0; c < width; ++c) {
        CHECK(close(out[c], siluf(x0[c] * w[c * kernel + carried]), 1e-6f));
        // The window carries the INPUT, not the activated output. Feeding the
        // output back would make this an IIR filter with a completely different
        // impulse response.
        CHECK(close(state[c * carried + carried - 1], x0[c], 1e-6f));
        CHECK(close(state[c * carried + 0], 0.0f, 1e-6f));
    }

    // Second token: taps `carried-1` and `carried` both fire, oldest first.
    const std::vector<float> x1{-0.5f, 0.25f, 4.0f};
    soma::arch::gdn::short_conv(width, kernel, w.data(), x1.data(), state.data(), out.data());
    for (std::uint32_t c = 0; c < width; ++c) {
        const float* wc = w.data() + c * kernel;
        const float want = siluf(x0[c] * wc[carried - 1] + x1[c] * wc[carried]);
        CHECK(close(out[c], want, 1e-6f));
    }

    // The window is exactly `kernel - 1` long, so the first token stops
    // influencing the output at position `kernel + 1` — NOT at `kernel`.
    //
    // Worth spelling out because the off-by-one runs the other way from the
    // obvious count: token 1 is still in the window when token `kernel` is
    // convolved (it occupies the oldest slot), and only leaves once token
    // `kernel + 1` pushes it out. Four tail tokens, therefore, for a kernel of
    // four.
    std::vector<float> sa(width * carried, 0.0f), sb(width * carried, 0.0f), o(width);
    const std::vector<std::vector<float>> tail{
        {1.f, 1.f, 1.f}, {2.f, 2.f, 2.f}, {3.f, 3.f, 3.f}, {4.f, 4.f, 4.f}};
    const std::vector<float> first_a{9.0f, 9.0f, 9.0f}, first_b{-7.0f, -7.0f, -7.0f};
    soma::arch::gdn::short_conv(width, kernel, w.data(), first_a.data(), sa.data(), o.data());
    soma::arch::gdn::short_conv(width, kernel, w.data(), first_b.data(), sb.data(), o.data());
    std::vector<float> oa(width), ob(width);
    for (const auto& t : tail) {
        soma::arch::gdn::short_conv(width, kernel, w.data(), t.data(), sa.data(), oa.data());
        soma::arch::gdn::short_conv(width, kernel, w.data(), t.data(), sb.data(), ob.data());
    }
    for (std::uint32_t c = 0; c < width; ++c) CHECK(close(oa[c], ob[c], 1e-6f));
}

// ── step ─────────────────────────────────────────────────────────────────────

void l2_scaled(const float* src, std::uint32_t n, float extra, std::vector<float>& out) {
    float sum = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i) sum += src[i] * src[i];
    const float inv = 1.0f / std::sqrt(sum + 1e-6f);
    out.assign(n, 0.0f);
    for (std::uint32_t i = 0; i < n; ++i) out[i] = src[i] * inv * extra;
}

void test_step() {
    const auto arch = make_arch();
    const auto& g = arch.attention.gdn;
    const auto dk = g.head_k_dim, dv = g.head_v_dim, nv = g.n_v_heads, nk = g.n_k_heads;
    const auto repeat = nv / nk;

    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> q(nk * dk), k(nk * dk), v(nv * dv), gv(nv), beta(nv);
    for (auto& x : q) x = u(rng);
    for (auto& x : k) x = u(rng);
    for (auto& x : v) x = u(rng);
    for (auto& x : gv) x = -0.4f;
    for (auto& x : beta) x = 0.6f; // pre-sigmoid

    std::vector<float> state(static_cast<std::size_t>(nv) * dk * dv, 0.0f);
    std::vector<float> scratch(dv), out(nv * dv);
    soma::arch::gdn::step(
        arch, q.data(), k.data(), v.data(), gv.data(), beta.data(), state.data(),
        scratch.data(), out.data());

    // FROM A ZERO STATE the whole recurrence collapses to a closed form:
    //   S = beta * k_hat (x) v,  o = S^T q_hat_scaled = beta * (q_hat . k_hat) * v
    // which pins the decay (irrelevant here), the delta (v - 0), the outer
    // product, and the read — including the head_k_dim ** -0.5 on q ONLY.
    const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
    for (std::uint32_t h = 0; h < nv; ++h) {
        std::vector<float> qh, kh;
        l2_scaled(q.data() + (h / repeat) * dk, dk, scale, qh);
        l2_scaled(k.data() + (h / repeat) * dk, dk, 1.0f, kh);
        float dot = 0.0f;
        for (std::uint32_t i = 0; i < dk; ++i) dot += qh[i] * kh[i];
        const float b = sigmoidf(beta[h]);
        for (std::uint32_t j = 0; j < dv; ++j) {
            CHECK(close(out[h * dv + j], b * dot * v[h * dv + j], 1e-5f));
        }
    }

    // THE BROADCAST IS INTERLEAVED. Value heads sharing a key head are
    // consecutive: h and h+1 read key head h/repeat for repeat > 1. Perturbing
    // ONE key head must move exactly `repeat` consecutive value heads.
    {
        auto k2 = k;
        k2[0] += 0.5f; // key head 0
        std::vector<float> st2(state.size(), 0.0f), out2(nv * dv);
        soma::arch::gdn::step(arch, q.data(), k2.data(), v.data(), gv.data(), beta.data(),
                              st2.data(), scratch.data(), out2.data());
        for (std::uint32_t h = 0; h < nv; ++h) {
            bool moved = false;
            for (std::uint32_t j = 0; j < dv; ++j)
                if (!close(out[h * dv + j], out2[h * dv + j], 1e-6f)) moved = true;
            CHECK(moved == (h / repeat == 0));
        }
    }

    // THE DELTA RULE IS EXACT AT beta = 1. With the correction fully applied,
    // the updated state predicts v perfectly: S^T k_hat == v. That single
    // property pins "predict from the DECAYED state, then correct toward v" —
    // predicting from the pre-decay state, or from the updated one, breaks it.
    {
        std::vector<float> big(nv, 40.0f); // sigmoid(40) == 1 to float precision
        std::vector<float> st(static_cast<std::size_t>(nv) * dk * dv, 0.0f), o2(nv * dv);
        // Start from a NON-zero state so the decay and the prediction both matter.
        std::mt19937 r2(7);
        for (auto& x : st) x = u(r2) * 0.3f;
        soma::arch::gdn::step(arch, q.data(), k.data(), v.data(), gv.data(), big.data(),
                              st.data(), scratch.data(), o2.data());
        for (std::uint32_t h = 0; h < nv; ++h) {
            std::vector<float> kh;
            l2_scaled(k.data() + (h / repeat) * dk, dk, 1.0f, kh);
            const float* sh = st.data() + static_cast<std::size_t>(h) * dk * dv;
            for (std::uint32_t j = 0; j < dv; ++j) {
                float pred = 0.0f;
                for (std::uint32_t i = 0; i < dk; ++i) pred += sh[i * dv + j] * kh[i];
                CHECK(close(pred, v[h * dv + j], 1e-4f));
            }
        }
    }

    // THE DECAY IS A SCALAR over the whole matrix, not a vector along the key
    // axis. From a non-zero state, halving `g` for one head must scale that
    // head's ENTIRE stored matrix identically — a per-channel decay would not.
    {
        std::vector<float> st(static_cast<std::size_t>(nv) * dk * dv, 0.25f);
        std::vector<float> zero_beta(nv, -40.0f); // sigmoid(-40) == 0: no update
        std::vector<float> o3(nv * dv);
        soma::arch::gdn::step(arch, q.data(), k.data(), v.data(), gv.data(), zero_beta.data(),
                              st.data(), scratch.data(), o3.data());
        const float want = 0.25f * std::exp(gv[0]);
        for (std::size_t i = 0; i < st.size(); ++i) CHECK(close(st[i], want, 1e-6f));
    }
}

// ── gated_rmsnorm ────────────────────────────────────────────────────────────

void test_gated_rmsnorm() {
    const auto arch = make_arch();
    const auto& g = arch.attention.gdn;
    const auto dv = g.head_v_dim, nv = g.n_v_heads;

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.5f, 1.5f);
    std::vector<float> x(nv * dv), z(nv * dv), w(dv), out(nv * dv);
    for (auto& t : x) t = u(rng);
    for (auto& t : z) t = u(rng);
    for (std::uint32_t i = 0; i < dv; ++i) w[i] = 0.5f + 0.25f * static_cast<float>(i);

    soma::arch::gdn::gated_rmsnorm(arch, x.data(), z.data(), w.data(), arch.rms_norm_eps,
                                   out.data());

    for (std::uint32_t h = 0; h < nv; ++h) {
        const auto off = static_cast<std::size_t>(h) * dv;
        float sum = 0.0f;
        for (std::uint32_t i = 0; i < dv; ++i) sum += x[off + i] * x[off + i];
        const float inv = 1.0f / std::sqrt(sum / static_cast<float>(dv) + arch.rms_norm_eps);
        for (std::uint32_t i = 0; i < dv; ++i) {
            // NORM, then weight, then SiLU(z). Gating before the norm is a
            // different operator and both are plausible readings of the name.
            CHECK(close(out[off + i], x[off + i] * inv * w[i] * siluf(z[off + i]), 1e-5f));
        }
    }

    // SiLU, not sigmoid — this norm is gated differently from the FULL-attention
    // block's output gate in the same model. At z = 0 they disagree completely:
    // silu(0) == 0 kills the output, sigmoid(0) == 0.5 halves it.
    {
        std::vector<float> zero(nv * dv, 0.0f), o(nv * dv);
        soma::arch::gdn::gated_rmsnorm(arch, x.data(), zero.data(), w.data(),
                                       arch.rms_norm_eps, o.data());
        for (const auto t : o) CHECK(close(t, 0.0f, 1e-7f));
    }

    // The weight is ONE [head_v_dim] vector shared by every head, not one per
    // head. Two heads fed identical values must come out identical.
    {
        std::vector<float> xx(nv * dv), zz(nv * dv), o(nv * dv);
        for (std::uint32_t h = 0; h < nv; ++h) {
            for (std::uint32_t i = 0; i < dv; ++i) {
                xx[h * dv + i] = static_cast<float>(i) - 1.0f;
                zz[h * dv + i] = 0.3f * static_cast<float>(i);
            }
        }
        soma::arch::gdn::gated_rmsnorm(arch, xx.data(), zz.data(), w.data(),
                                       arch.rms_norm_eps, o.data());
        for (std::uint32_t h = 1; h < nv; ++h)
            for (std::uint32_t i = 0; i < dv; ++i)
                CHECK(close(o[h * dv + i], o[i], 1e-7f));
    }
}

// ── prefill == stepwise ──────────────────────────────────────────────────────

void test_prefill_matches_stepwise() {
    const auto arch = make_arch();
    const auto& g = arch.attention.gdn;
    const auto d = arch.topology.d_model;
    const auto cw = g.conv_width(), vd = g.value_dim(), nv = g.n_v_heads;

    std::mt19937 rng(20260823);
    std::uniform_real_distribution<float> u(-0.3f, 0.3f);
    auto fill = [&](std::size_t n) {
        std::vector<float> v(n);
        for (auto& t : v) t = u(rng);
        return v;
    };

    const auto qkv_w = fill(static_cast<std::size_t>(cw) * d);
    const auto z_w = fill(static_cast<std::size_t>(vd) * d);
    const auto b_w = fill(static_cast<std::size_t>(nv) * d);
    const auto a_w = fill(static_cast<std::size_t>(nv) * d);
    const auto o_w = fill(static_cast<std::size_t>(d) * vd);
    const auto conv_w = fill(static_cast<std::size_t>(cw) * g.conv_kernel);
    auto a_log = fill(nv);
    const auto dt = fill(nv);
    const auto o_norm = fill(g.head_v_dim);

    soma::arch::gdn::F32HybridWeights w;
    w.linear = true;
    w.in_proj_qkv = soma::WeightRef::from_f32(qkv_w, cw, d);
    w.in_proj_z = soma::WeightRef::from_f32(z_w, vd, d);
    w.in_proj_b = soma::WeightRef::from_f32(b_w, nv, d);
    w.in_proj_a = soma::WeightRef::from_f32(a_w, nv, d);
    w.out_proj = soma::WeightRef::from_f32(o_w, d, vd);
    w.conv_w = conv_w;
    w.a_log = a_log;
    w.dt_bias = dt;
    w.o_norm = o_norm;

    constexpr std::uint32_t T = 12;
    const auto x = fill(static_cast<std::size_t>(T) * d);

    const auto rec_n = static_cast<std::size_t>(g.recurrent_elems());
    const auto conv_n = static_cast<std::size_t>(cw) * (g.conv_kernel - 1);

    std::vector<float> rec_a(rec_n, 0.0f), cnv_a(conv_n, 0.0f), out_a(T * d, 0.0f);
    CHECK(soma::arch::gdn::f32_linear_layer(arch, w, x.data(), T, rec_a.data(), cnv_a.data(),
                                            out_a.data()) == soma::StatusCode::Ok);

    // The same tokens, one at a time, carrying the state forward — which is
    // exactly what cached decode does. Any disagreement means prefill and decode
    // are two different models, and conformance would never see it: greedy
    // generation re-runs the prefill every step.
    std::vector<float> rec_b(rec_n, 0.0f), cnv_b(conv_n, 0.0f), out_b(T * d, 0.0f);
    for (std::uint32_t t = 0; t < T; ++t) {
        CHECK(soma::arch::gdn::f32_linear_layer(arch, w, x.data() + static_cast<std::size_t>(t) * d,
                                                1, rec_b.data(), cnv_b.data(),
                                                out_b.data() + static_cast<std::size_t>(t) * d) ==
              soma::StatusCode::Ok);
    }

    float worst = 0.0f;
    for (std::size_t i = 0; i < out_a.size(); ++i)
        worst = std::max(worst, std::fabs(out_a[i] - out_b[i]));
    CHECK(worst < 1e-5f);

    // The carried state must agree too, or the NEXT token would diverge even
    // though every token so far matched.
    for (std::size_t i = 0; i < rec_n; ++i) CHECK(close(rec_a[i], rec_b[i], 1e-5f));
    for (std::size_t i = 0; i < conv_n; ++i) CHECK(close(cnv_a[i], cnv_b[i], 1e-6f));

    std::cout << "  prefill vs stepwise: max|diff| " << worst << "\n";
}

} // namespace

int main() {
    test_gate();
    test_short_conv();
    test_step();
    test_gated_rmsnorm();
    test_prefill_matches_stepwise();

    if (g_failures != 0) {
        std::cerr << "gdn_kernel: " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "gdn_kernel: OK\n";
    return 0;
}
