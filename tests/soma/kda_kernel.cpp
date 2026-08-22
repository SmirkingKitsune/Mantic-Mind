// Soma — the KDA kernel.
//
// Kimi Delta Attention is a delta-rule linear attention with a PER-CHANNEL
// forget gate. Almost every way of getting it wrong produces finite, plausible,
// converging numbers:
//
//   * decay along the value axis instead of the key axis
//   * predict from the state BEFORE the decay rather than after
//   * read the output BEFORE the delta update rather than after
//   * gate before the RMS norm rather than after
//   * reverse the convolution taps
//   * broadcast A_log per channel, or dt_bias per head
//
// So this file leans on INVARIANTS rather than on a transcription of the
// reference. Each of the checks below is derived from what the delta rule is
// supposed to do, and each fails under one of the misreadings above. The
// transcription comparison comes last, as a consistency check on top — it is the
// weakest test here, because it shares its reading of the reference with the
// implementation.

#include "soma/arch/kda.hpp"
#include "soma/arch_ir.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
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

constexpr std::uint32_t H = 2;  // heads
constexpr std::uint32_t D = 4;  // head_dim
constexpr std::uint32_t P = H * D;
constexpr std::uint32_t KS = 3; // conv kernel

/// Small enough to check by hand, with a heterogeneous layer split so the cache
/// layout is exercised on both kinds and on the transition between them.
soma::ArchIr make_arch(bool gate_bound = true) {
    soma::ArchIr a;
    a.attention.family = soma::AttentionFamily::MlaKda;
    a.topology.n_layers = 3;
    a.topology.d_model = 16;
    a.topology.vocab_size = 32;
    a.topology.layer_kinds.assign(3, soma::LayerKind::Moe);
    auto& k = a.attention.kda;
    k.n_heads = H;
    k.head_dim = D;
    k.conv_kernel = KS;
    k.full_rank_gate = true;
    k.has_gate_bound = gate_bound;
    k.gate_lower_bound = -5.0f;
    k.layer_kinds = {soma::AttnLayerKind::Linear,
                     soma::AttnLayerKind::Full,
                     soma::AttnLayerKind::Linear};
    auto& m = a.attention.mla;
    m.kv_lora_rank = 8;
    m.qk_rope_head_dim = 2;
    m.qk_nope_head_dim = 4;
    m.v_head_dim = 4;
    m.nope = true;
    return a;
}

float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// Beta arrives PRE-sigmoid, so these are the saturating logits that mean
/// "replace" and "ignore". Chosen rather than +-inf so the test exercises the
/// same sigmoid the kernel does.
constexpr float kBetaOne = 40.0f;
constexpr float kBetaZero = -40.0f;

} // namespace

int main() {
    const auto arch = make_arch();
    std::vector<float> state(static_cast<std::size_t>(H) * D * D, 0.0f);
    std::vector<float> scratch(D), out(P), q(P), k(P), v(P), g(P), beta(H);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // ── 1. the delta rule FITS THE PAIR IT JUST WROTE ────────────────────────
    //
    // With beta = 1 and no decay, writing (k, v) and immediately querying with
    // q = k must return exactly v * head_dim^-0.5, for ANY prior state:
    //
    //   S' = S + k(v - S^T k)     =>  S'^T k = S^T k + (k.k)(v - S^T k) = v
    //
    // because k is L2-normalized, so k.k = 1. This one identity pins the L2
    // normalization, the beta sigmoid, the delta correction, and — critically —
    // that the output is read AFTER the update. Reading before it yields
    // S^T k, the model's PRIOR guess, which is a plausible number and not v.
    for (std::uint32_t i = 0; i < P; ++i) {
        q[i] = k[i] = static_cast<float>((i % 3) + 1); // q == k, unnormalized
        v[i] = 0.25f * static_cast<float>(i) - 1.0f;
        g[i] = 0.0f; // no decay
    }
    beta[0] = beta[1] = kBetaOne;
    soma::arch::kda::step(arch, q.data(), k.data(), v.data(), g.data(), beta.data(),
                          state.data(), scratch.data(), out.data());
    for (std::uint32_t i = 0; i < P; ++i) CHECK(close(out[i], v[i] * scale));

    // It must keep holding against a NON-empty state — that is the part the
    // "S starts at zero" version of this test would not catch.
    std::vector<float> v2(P);
    for (std::uint32_t i = 0; i < P; ++i) v2[i] = -0.5f * static_cast<float>(i) + 2.0f;
    soma::arch::kda::step(arch, q.data(), k.data(), v2.data(), g.data(), beta.data(),
                          state.data(), scratch.data(), out.data());
    for (std::uint32_t i = 0; i < P; ++i) CHECK(close(out[i], v2[i] * scale));

    // And it must hold with the DECAY ENGAGED, which is what pins the
    // prediction to the state AFTER the decay rather than before it.
    //
    // The identity S'^T k = v survives any prior state, so a zero gate cannot
    // tell the two orderings apart — both give v, and the earlier cases pass
    // under either. With a real gate they separate: predicting from the
    // undecayed state leaves S'^T k = decay(S)^T k + v - S^T k, which equals v
    // only when the decay is the identity. This is the one check standing
    // between a correct kernel and a plausible one whose memory is subtly
    // wrong, and until it existed only the transcription below caught it.
    std::vector<float> v3(P);
    {
        for (std::uint32_t i = 0; i < P; ++i) {
            v3[i] = 0.3f * static_cast<float>(i) - 0.9f;
            g[i] = -0.2f * static_cast<float>(i % D) - 0.15f; // per channel, non-trivial
        }
        beta[0] = beta[1] = kBetaOne;
        soma::arch::kda::step(arch, q.data(), k.data(), v3.data(), g.data(), beta.data(),
                              state.data(), scratch.data(), out.data());
        for (std::uint32_t i = 0; i < P; ++i) CHECK(close(out[i], v3[i] * scale));
        for (std::uint32_t i = 0; i < P; ++i) g[i] = 0.0f; // restore for later cases
    }

    // ── 2. beta = 0 writes NOTHING ───────────────────────────────────────────
    //
    // The state must be untouched, so the same query returns the same answer —
    // `v3`, the most recent write, and not the wildly different value offered
    // alongside beta = 0.
    {
        auto before = state;
        std::vector<float> o2(P);
        beta[0] = beta[1] = kBetaZero;
        std::vector<float> vjunk(P, 999.0f);
        soma::arch::kda::step(arch, q.data(), k.data(), vjunk.data(), g.data(), beta.data(),
                              state.data(), scratch.data(), o2.data());
        for (std::size_t i = 0; i < state.size(); ++i) CHECK(close(state[i], before[i]));
        for (std::uint32_t i = 0; i < P; ++i) CHECK(close(o2[i], v3[i] * scale));
    }

    // ── 3. the decay runs along the KEY axis, per channel ────────────────────
    //
    // The axis is invisible to a uniform gate: for a rank-1 state k(x)v, decaying
    // either axis by one scalar scales the readout identically. A PER-CHANNEL
    // gate separates them, which is exactly why KDA's gate is per channel.
    //
    //   key axis:    o = scale * (sum_i khat_i^2 exp(g_i)) * v
    //   value axis:  o = scale * exp(g_j) * v_j            <- different vector
    {
        std::fill(state.begin(), state.end(), 0.0f);
        beta[0] = beta[1] = kBetaOne;
        for (std::uint32_t i = 0; i < P; ++i) g[i] = 0.0f;
        soma::arch::kda::step(arch, q.data(), k.data(), v.data(), g.data(), beta.data(),
                              state.data(), scratch.data(), out.data());

        // Now a decay-only step (beta = 0) with a gate that differs per channel.
        for (std::uint32_t i = 0; i < P; ++i) g[i] = -0.1f * static_cast<float>(i % D) - 0.05f;
        beta[0] = beta[1] = kBetaZero;
        soma::arch::kda::step(arch, q.data(), k.data(), v.data(), g.data(), beta.data(),
                              state.data(), scratch.data(), out.data());

        for (std::uint32_t h = 0; h < H; ++h) {
            // khat for this head
            float n = 0.0f;
            for (std::uint32_t i = 0; i < D; ++i) n += k[h * D + i] * k[h * D + i];
            n = 1.0f / std::sqrt(n + 1e-6f);
            float factor = 0.0f;
            for (std::uint32_t i = 0; i < D; ++i) {
                const float kh = k[h * D + i] * n;
                factor += kh * kh * std::exp(g[h * D + i]);
            }
            for (std::uint32_t j = 0; j < D; ++j) {
                const float want = scale * factor * v[h * D + j];
                CHECK(close(out[h * D + j], want));
                // And it is NOT the value-axis reading, which would be this:
                const float value_axis = scale * std::exp(g[h * D + j]) * v[h * D + j];
                CHECK(!close(out[h * D + j], value_axis, 1e-3f) ||
                      close(want, value_axis, 1e-3f));
            }
        }
    }

    // ── 4. the gate ──────────────────────────────────────────────────────────
    {
        std::vector<float> a_log{0.3f, -0.7f}, dt(P), graw(P), gout(P);
        for (std::uint32_t i = 0; i < P; ++i) {
            dt[i] = 0.1f * static_cast<float>(i) - 0.2f;
            graw[i] = 0.5f - 0.3f * static_cast<float>(i);
        }
        soma::arch::kda::gate(arch, a_log.data(), dt.data(), graw.data(), gout.data());
        for (std::uint32_t h = 0; h < H; ++h) {
            for (std::uint32_t c = 0; c < D; ++c) {
                const auto i = h * D + c;
                // safe gate: lower_bound * sigmoid(exp(A_log) * (g + dt))
                const float want =
                    -5.0f * sigmoidf(std::exp(a_log[h]) * (graw[i] + dt[i]));
                CHECK(close(gout[i], want));
                // Bounded strictly between the floor and zero, which is the
                // point of the safe form: the state can never be annihilated.
                CHECK(gout[i] > -5.0f && gout[i] < 0.0f);
            }
        }
        // A_log is per HEAD and dt_bias per CHANNEL. Perturbing A_log[0] must
        // move every channel of head 0 and none of head 1; perturbing dt[0]
        // must move exactly one channel.
        auto a2 = a_log;
        a2[0] += 1.0f;
        std::vector<float> g2(P);
        soma::arch::kda::gate(arch, a2.data(), dt.data(), graw.data(), g2.data());
        for (std::uint32_t c = 0; c < D; ++c) CHECK(!close(g2[c], gout[c]));
        for (std::uint32_t c = 0; c < D; ++c) CHECK(close(g2[D + c], gout[D + c]));

        auto dt2 = dt;
        dt2[0] += 1.0f;
        soma::arch::kda::gate(arch, a_log.data(), dt2.data(), graw.data(), g2.data());
        CHECK(!close(g2[0], gout[0]));
        for (std::uint32_t i = 1; i < P; ++i) CHECK(close(g2[i], gout[i]));

        // The unbounded form is a genuinely different function, not a limit of
        // the safe one: -exp(A_log) * softplus(g + dt), unbounded below.
        const auto arch_free = make_arch(/*gate_bound=*/false);
        soma::arch::kda::gate(arch_free, a_log.data(), dt.data(), graw.data(), g2.data());
        for (std::uint32_t h = 0; h < H; ++h) {
            for (std::uint32_t c = 0; c < D; ++c) {
                const auto i = h * D + c;
                const float x = graw[i] + dt[i];
                const float want = -std::exp(a_log[h]) * std::log1p(std::exp(x));
                CHECK(close(g2[i], want));
            }
        }
    }

    // ── 5. the convolution is causal and its taps are not reversed ───────────
    {
        const std::uint32_t W = 3, T = 5;
        std::vector<float> xs(T * W);
        for (std::uint32_t t = 0; t < T; ++t)
            for (std::uint32_t c = 0; c < W; ++c) xs[t * W + c] = static_cast<float>(t + 1) + c;

        // weight[c][KS-1] multiplies the CURRENT token.
        std::vector<float> ident_now(W * KS, 0.0f), ident_old(W * KS, 0.0f);
        for (std::uint32_t c = 0; c < W; ++c) {
            ident_now[c * KS + (KS - 1)] = 1.0f;
            ident_old[c * KS + 0] = 1.0f;
        }
        auto run = [&](const std::vector<float>& w, std::vector<float>& ys) {
            std::vector<float> st(static_cast<std::size_t>(W) * (KS - 1), 0.0f);
            ys.assign(T * W, 0.0f);
            for (std::uint32_t t = 0; t < T; ++t)
                soma::arch::kda::short_conv(W, KS, w.data(), nullptr, xs.data() + t * W,
                                            st.data(), ys.data() + t * W);
        };
        std::vector<float> ys;
        run(ident_now, ys);
        for (std::uint32_t t = 0; t < T; ++t)
            for (std::uint32_t c = 0; c < W; ++c) {
                const float x = xs[t * W + c];
                CHECK(close(ys[t * W + c], x * sigmoidf(x))); // SiLU of the CURRENT token
            }
        run(ident_old, ys);
        for (std::uint32_t t = 0; t < T; ++t)
            for (std::uint32_t c = 0; c < W; ++c) {
                // Reaches back exactly KS-1 positions; zero-padded before the start.
                const float x = t >= KS - 1 ? xs[(t - (KS - 1)) * W + c] : 0.0f;
                CHECK(close(ys[t * W + c], x * sigmoidf(x)));
            }

        // Against a dense convolution over an explicitly zero-padded sequence —
        // a different formulation from the streaming window the kernel keeps.
        std::vector<float> w(W * KS);
        for (std::uint32_t i = 0; i < w.size(); ++i) w[i] = 0.5f - 0.2f * static_cast<float>(i);
        std::vector<float> bias{0.1f, -0.3f, 0.7f};
        std::vector<float> st(static_cast<std::size_t>(W) * (KS - 1), 0.0f);
        for (std::uint32_t t = 0; t < T; ++t) {
            std::vector<float> got(W);
            soma::arch::kda::short_conv(W, KS, w.data(), bias.data(), xs.data() + t * W,
                                        st.data(), got.data());
            for (std::uint32_t c = 0; c < W; ++c) {
                float acc = bias[c];
                for (std::uint32_t j = 0; j < KS; ++j) {
                    const std::int64_t src = static_cast<std::int64_t>(t) - (KS - 1) + j;
                    if (src >= 0) acc += w[c * KS + j] * xs[static_cast<std::size_t>(src) * W + c];
                }
                CHECK(close(got[c], acc * sigmoidf(acc)));
            }
        }
    }

    // ── 6. the gate lands AFTER the norm ─────────────────────────────────────
    //
    // RMS norm is scale-invariant, so gating BEFORE it — Mamba's RMSNormGated
    // order, and an equally natural reading of the name `FusedRMSNormGated` —
    // would cancel a uniform gate entirely. With a constant gate c the two
    // readings differ by exactly sigmoid(c), which is what this measures.
    {
        std::vector<float> x(P), gr(P, 1.3f), w(D, 1.0f), o(P);
        for (std::uint32_t i = 0; i < P; ++i) x[i] = 0.7f * static_cast<float>(i) - 1.1f;
        soma::arch::kda::gated_rmsnorm(arch, x.data(), gr.data(), w.data(), 1e-5f, o.data());
        for (std::uint32_t h = 0; h < H; ++h) {
            float ss = 0.0f;
            for (std::uint32_t i = 0; i < D; ++i) ss += x[h * D + i] * x[h * D + i];
            const float inv = 1.0f / std::sqrt(ss / static_cast<float>(D) + 1e-5f);
            for (std::uint32_t i = 0; i < D; ++i) {
                const float after = x[h * D + i] * inv * sigmoidf(1.3f);
                const float before = x[h * D + i] * inv; // gate cancelled by the norm
                CHECK(close(o[h * D + i], after));
                CHECK(!close(o[h * D + i], before, 1e-3f));
            }
        }
        // The weight is per head_dim and applied per head, not across the
        // flattened projection.
        std::vector<float> w2{1.0f, 2.0f, 3.0f, 4.0f}, o2(P);
        soma::arch::kda::gated_rmsnorm(arch, x.data(), gr.data(), w2.data(), 1e-5f, o2.data());
        for (std::uint32_t h = 0; h < H; ++h)
            for (std::uint32_t i = 0; i < D; ++i)
                CHECK(close(o2[h * D + i], o[h * D + i] * w2[i]));
    }

    // ── 7. against a transcription of the reference recurrence ───────────────
    //
    // Weakest check here — it shares its reading of `naive_recurrent_kda` with
    // the implementation — but it is organized differently (a materialized
    // decayed copy and separate matvecs, rather than one fused pass), so it
    // still catches slips in the fused version.
    {
        std::mt19937 rng(20260821);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        const std::uint32_t T = 7;
        std::vector<float> S(static_cast<std::size_t>(H) * D * D, 0.0f), Sref = S;
        std::fill(state.begin(), state.end(), 0.0f);

        for (std::uint32_t t = 0; t < T; ++t) {
            for (std::uint32_t i = 0; i < P; ++i) {
                q[i] = u(rng);
                k[i] = u(rng);
                v[i] = u(rng);
                g[i] = -std::fabs(u(rng)); // log-space decay is negative
            }
            for (std::uint32_t h = 0; h < H; ++h) beta[h] = u(rng);

            soma::arch::kda::step(arch, q.data(), k.data(), v.data(), g.data(), beta.data(),
                                  state.data(), scratch.data(), out.data());

            for (std::uint32_t h = 0; h < H; ++h) {
                float* Sh = Sref.data() + static_cast<std::size_t>(h) * D * D;
                float qn = 0.0f, kn = 0.0f;
                for (std::uint32_t i = 0; i < D; ++i) {
                    qn += q[h * D + i] * q[h * D + i];
                    kn += k[h * D + i] * k[h * D + i];
                }
                qn = 1.0f / std::sqrt(qn + 1e-6f);
                kn = 1.0f / std::sqrt(kn + 1e-6f);
                const float b = sigmoidf(beta[h]);

                std::vector<float> dec(static_cast<std::size_t>(D) * D);
                for (std::uint32_t i = 0; i < D; ++i)
                    for (std::uint32_t j = 0; j < D; ++j)
                        dec[i * D + j] = Sh[i * D + j] * std::exp(g[h * D + i]);
                std::vector<float> pred(D, 0.0f);
                for (std::uint32_t j = 0; j < D; ++j)
                    for (std::uint32_t i = 0; i < D; ++i)
                        pred[j] += k[h * D + i] * kn * dec[i * D + j];
                for (std::uint32_t i = 0; i < D; ++i)
                    for (std::uint32_t j = 0; j < D; ++j)
                        dec[i * D + j] += b * k[h * D + i] * kn * (v[h * D + j] - pred[j]);
                std::vector<float> o(D, 0.0f);
                for (std::uint32_t j = 0; j < D; ++j)
                    for (std::uint32_t i = 0; i < D; ++i)
                        o[j] += q[h * D + i] * qn * scale * dec[i * D + j];
                for (std::uint32_t i = 0; i < D * D; ++i) Sh[i] = dec[i];
                for (std::uint32_t j = 0; j < D; ++j) CHECK(close(out[h * D + j], o[j], 1e-4f));
            }
        }
        for (std::size_t i = 0; i < state.size(); ++i) CHECK(close(state[i], Sref[i], 1e-4f));
    }

    // ── 8. the cache layout is the allocation ────────────────────────────────
    {
        const std::uint32_t ctx = 64;
        const auto total = soma::arch::kda::kv_bytes_for_context(arch, ctx);
        const auto r0 = soma::arch::kda::layer_region(arch, 0, ctx); // linear
        const auto r1 = soma::arch::kda::layer_region(arch, 1, ctx); // full
        const auto r2 = soma::arch::kda::layer_region(arch, 2, ctx); // linear
        CHECK(r2.end == total);
        // Monotone and non-overlapping across a heterogeneous stack.
        CHECK(r0.end <= r1.latent && r1.end <= r2.recurrent);
        CHECK(r0.recurrent % 64 == 0 && r1.latent % 64 == 0 && r2.recurrent % 64 == 0);
        // A full layer holds no recurrent state and a linear layer caches no
        // tokens: both are zero-length regions, not aliases onto a neighbour.
        CHECK(r1.recurrent == r1.end && r1.conv == r1.end);
        CHECK(r0.latent == r0.end);

        const auto rec = static_cast<std::size_t>(H) * D * D * sizeof(float);
        const auto cnv = static_cast<std::size_t>(3) * P * (KS - 1) * sizeof(float);
        CHECK(r0.conv - r0.recurrent == rec); // both already 64-aligned here
        CHECK(r0.end - r0.conv == cnv);
        CHECK(r1.end - r1.latent ==
              static_cast<std::size_t>(ctx) * (8 + 2) * sizeof(float));

        // The fixed term is what a zero-context layout costs, and the growth
        // rate is what one more token adds — both read off the layout rather
        // than recomputed, so padding cannot make them disagree.
        CHECK(soma::arch::kda::recurrent_state_bytes(arch) ==
              soma::arch::kda::kv_bytes_for_context(arch, 0));
        CHECK(soma::arch::kda::kv_bytes_for_context(arch, 0) == 2 * (rec + cnv));
        CHECK(soma::arch::kda::kv_bytes_per_token(arch) == 1 * (8 + 2) * sizeof(float));
    }

    std::cout << "kda_kernel: OK\n";
    return 0;
}
