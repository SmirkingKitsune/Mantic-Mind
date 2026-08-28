// Soma — the KDA layer forward, and the opaque cache it runs against.
//
// `soma_kda_kernel` pins the arithmetic. This pins the PLUMBING around it: that
// the projections, the three convolutions, the gate, the recurrence, the gated
// norm and the output projection are wired in the right order, and that the
// per-sequence state genuinely lives in the layer's own slice of the opaque
// cache.
//
// The load-bearing property is that PREFILL, STREAMING AND CACHED DECODE ALL
// AGREE. A recurrent layer is the one place where "process T tokens at once" and
// "process them one at a time" can diverge silently: a mis-advanced convolution
// window, a state read from the wrong offset, or a cacheless path that leaks the
// previous sequence's state all produce fluent output and a different model.
// None of the three is a reference for the others — they are three routes
// through the same recurrence, and disagreement means one of them is wrong.

#include "soma/arch/kda.hpp"
#include "soma/arch_ir.hpp"
#include "soma/f32_model.hpp"
#include "soma/kv_cache.hpp"

#include <cmath>
#include <cstddef>
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

constexpr std::uint32_t Dm = 16; // d_model
constexpr std::uint32_t H = 2;
constexpr std::uint32_t Dh = 4;
constexpr std::uint32_t P = H * Dh;
constexpr std::uint32_t KS = 3;
constexpr std::uint32_t CTX = 32;

soma::ArchIr make_arch() {
    soma::ArchIr a;
    a.attention.family = soma::AttentionFamily::MlaKda;
    a.topology.n_layers = 3;
    a.topology.d_model = Dm;
    a.topology.vocab_size = 32;
    a.topology.layer_kinds.assign(3, soma::LayerKind::Moe);
    a.rms_norm_eps = 1e-5f;
    auto& k = a.attention.kda;
    k.n_heads = H;
    k.head_dim = Dh;
    k.conv_kernel = KS;
    k.full_rank_gate = true;
    k.has_gate_bound = true;
    k.gate_lower_bound = -5.0f;
    // Layer 1 is a full-attention layer sitting between two linear ones, so the
    // linear layers are NOT adjacent in the cache — which is what makes the
    // offset check below meaningful.
    k.layer_kinds = {soma::AttnLayerKind::Linear,
                     soma::AttnLayerKind::Full,
                     soma::AttnLayerKind::Linear};
    auto& m = a.attention.mla;
    m.kv_lora_rank = 8;
    m.qk_rope_head_dim = 2;
    m.qk_nope_head_dim = 4;
    m.v_head_dim = 4;
    m.nope = true;
    m.output_gate = true;
    return a;
}

/// Deterministic, spread across sign and magnitude so nothing cancels by luck.
float syn(std::size_t i, float phase) {
    return 0.6f * std::sin(0.7f * static_cast<float>(i) + phase) +
           0.2f * std::cos(0.31f * static_cast<float>(i));
}

struct Weights {
    std::vector<float> q, k, v, fa, fb, b, g, o;      // projection matrices
    std::vector<float> qc, kc, vc;                    // conv kernels
    std::vector<float> qcb, kcb, vcb;                 // conv biases
    std::vector<float> a_log, dt_bias, o_norm;
};

void fill(std::vector<float>& dst, std::size_t n, float phase) {
    dst.resize(n);
    for (std::size_t i = 0; i < n; ++i) dst[i] = syn(i, phase);
}

soma::arch::kda::F32HybridWeights build(Weights& s) {
    fill(s.q, static_cast<std::size_t>(P) * Dm, 0.1f);
    fill(s.k, static_cast<std::size_t>(P) * Dm, 0.9f);
    fill(s.v, static_cast<std::size_t>(P) * Dm, 1.7f);
    fill(s.fa, static_cast<std::size_t>(Dh) * Dm, 2.3f);
    fill(s.fb, static_cast<std::size_t>(P) * Dh, 3.1f);
    fill(s.b, static_cast<std::size_t>(H) * Dm, 0.5f);
    fill(s.g, static_cast<std::size_t>(P) * Dm, 4.2f);
    fill(s.o, static_cast<std::size_t>(Dm) * P, 5.5f);
    fill(s.qc, static_cast<std::size_t>(P) * KS, 6.1f);
    fill(s.kc, static_cast<std::size_t>(P) * KS, 6.7f);
    fill(s.vc, static_cast<std::size_t>(P) * KS, 7.3f);
    fill(s.qcb, P, 8.0f);
    fill(s.kcb, P, 8.4f);
    fill(s.vcb, P, 8.8f);
    fill(s.a_log, H, 0.2f);
    fill(s.dt_bias, P, 1.1f);
    fill(s.o_norm, Dh, 2.9f);

    soma::arch::kda::F32HybridWeights w;
    w.linear = true;
    w.q_proj = soma::WeightRef::from_f32(s.q, P, Dm);
    w.k_proj = soma::WeightRef::from_f32(s.k, P, Dm);
    w.v_proj = soma::WeightRef::from_f32(s.v, P, Dm);
    w.f_a_proj = soma::WeightRef::from_f32(s.fa, Dh, Dm);
    w.f_b_proj = soma::WeightRef::from_f32(s.fb, P, Dh);
    w.b_proj = soma::WeightRef::from_f32(s.b, H, Dm);
    w.g_proj = soma::WeightRef::from_f32(s.g, P, Dm);
    w.o_proj = soma::WeightRef::from_f32(s.o, Dm, P);
    w.q_conv_w = s.qc;
    w.k_conv_w = s.kc;
    w.v_conv_w = s.vc;
    w.q_conv_b = s.qcb;
    w.k_conv_b = s.kcb;
    w.v_conv_b = s.vcb;
    w.a_log = s.a_log;
    w.dt_bias = s.dt_bias;
    w.o_norm = s.o_norm;
    return w;
}

} // namespace

int main() {
    const auto arch = make_arch();
    Weights store;
    const auto w = build(store);

    constexpr std::uint32_t T = 6;
    std::vector<float> x(static_cast<std::size_t>(T) * Dm);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = syn(i, 9.4f);

    const auto rec_floats = static_cast<std::size_t>(H) * Dh * Dh;
    const auto cnv_floats = static_cast<std::size_t>(3) * P * (KS - 1);

    // ── A. prefill: the whole span in one call, from a zero state ────────────
    std::vector<float> want(static_cast<std::size_t>(T) * Dm);
    std::vector<float> rec_a(rec_floats, 0.0f), cnv_a(cnv_floats, 0.0f);
    CHECK(soma::arch::kda::f32_linear_layer(arch, w, x.data(), T, rec_a.data(), cnv_a.data(),
                                            want.data()) == soma::StatusCode::Ok);
    // Something actually happened — an all-zero output would satisfy every
    // agreement check below without exercising anything.
    {
        float mag = 0.0f;
        for (const auto f : want) mag += std::fabs(f);
        CHECK(mag > 1e-3f);
    }

    // ── B. streaming: one token at a time, carrying the state ────────────────
    {
        std::vector<float> got(static_cast<std::size_t>(T) * Dm);
        std::vector<float> rec(rec_floats, 0.0f), cnv(cnv_floats, 0.0f);
        for (std::uint32_t t = 0; t < T; ++t) {
            CHECK(soma::arch::kda::f32_linear_layer(
                      arch, w, x.data() + static_cast<std::size_t>(t) * Dm, 1, rec.data(),
                      cnv.data(), got.data() + static_cast<std::size_t>(t) * Dm) ==
                  soma::StatusCode::Ok);
        }
        for (std::size_t i = 0; i < want.size(); ++i) CHECK(close(got[i], want[i], 1e-4f));
        // The state must land in the same place too, not merely the outputs: a
        // convolution window advanced one step out of phase can still produce
        // matching tokens for a short span.
        for (std::size_t i = 0; i < rec_floats; ++i) CHECK(close(rec[i], rec_a[i], 1e-4f));
        for (std::size_t i = 0; i < cnv_floats; ++i) CHECK(close(cnv[i], cnv_a[i], 1e-4f));
    }

    // ── C. the cacheless entry point carries NOTHING between calls ───────────
    //
    // `attention` has no cache by definition, so a recurrent layer must start
    // from zero every time. If it did not, the second call would differ from the
    // first and the uncached path would depend on whatever ran before it.
    soma::F32LayerWeights lw;
    lw.attn.adopt(const_cast<soma::arch::kda::F32HybridWeights*>(&w), [](void*) {});
    {
        soma::F32Workspace ws;
        std::vector<float> first(want.size()), second(want.size());
        CHECK(soma::arch::kda::f32_attention(arch, lw, x.data(), T, ws, first.data()) ==
              soma::StatusCode::Ok);
        CHECK(soma::arch::kda::f32_attention(arch, lw, x.data(), T, ws, second.data()) ==
              soma::StatusCode::Ok);
        for (std::size_t i = 0; i < want.size(); ++i) {
            CHECK(close(first[i], want[i], 1e-4f));
            CHECK(close(second[i], first[i]));
        }
    }

    // ── D. cached decode, through the opaque buffer ──────────────────────────
    //
    // The end-to-end one: state read from and written back to this layer's own
    // slice of the cache, one token per call, must reproduce the prefill span.
    const auto total = static_cast<std::size_t>(soma::arch::kda::kv_bytes_for_context(arch, CTX));
    std::vector<std::byte> cache(total, std::byte{0});
    {
        soma::F32Workspace ws;
        std::vector<float> got(want.size());
        for (std::uint32_t t = 0; t < T; ++t) {
            soma::KvRow row{};
            row.opaque_base = cache.data();
            row.opaque_bytes = cache.size();
            row.max_ctx = CTX;
            row.pos = t;
            row.len = t + 1;
            CHECK(soma::arch::kda::f32_attention_kv(
                      arch, lw, x.data() + static_cast<std::size_t>(t) * Dm, 1, /*layer=*/0, &row,
                      ws, got.data() + static_cast<std::size_t>(t) * Dm) == soma::StatusCode::Ok);
        }
        for (std::size_t i = 0; i < want.size(); ++i) CHECK(close(got[i], want[i], 1e-4f));

        // The state ended up in layer 0's region, byte for byte.
        const auto r0 = soma::arch::kda::layer_region(arch, 0, CTX);
        const auto* rec = reinterpret_cast<const float*>(cache.data() + r0.recurrent);
        for (std::size_t i = 0; i < rec_floats; ++i) CHECK(close(rec[i], rec_a[i], 1e-4f));

        // ── which window belongs to which projection ─────────────────────────
        //
        // The three convolution windows share one region, and NOTHING above
        // would notice if two of them aliased: prefill and streaming run the
        // same code, so they would agree on the same wrong answer. Aliasing q's
        // window onto k's is a real and quiet way to break this layer.
        //
        // So the windows are read directly. Each holds RAW projection outputs —
        // the convolution's inputs, not its outputs — oldest first, so after a
        // fresh sequence the newest slot of window `c` is exactly that
        // projection's value at the last token processed.
        const auto carried = static_cast<std::size_t>(KS - 1);
        const auto stride = static_cast<std::size_t>(P) * carried;
        const auto* cw = reinterpret_cast<const float*>(cache.data() + r0.conv);
        const std::vector<float>* mats[3] = {&store.q, &store.k, &store.v};
        for (std::uint32_t which = 0; which < 3; ++which) {
            const float* win = cw + which * stride;
            const auto& mat = *mats[which];
            for (std::uint32_t c = 0; c < P; ++c) {
                float proj_val = 0.0f;
                for (std::uint32_t j = 0; j < Dm; ++j)
                    proj_val += mat[static_cast<std::size_t>(c) * Dm + j] *
                                x[static_cast<std::size_t>(T - 1) * Dm + j];
                CHECK(close(win[static_cast<std::size_t>(c) * carried + (carried - 1)], proj_val,
                            1e-4f));
            }
        }
    }

    // ── E. a layer writes ONLY its own region ────────────────────────────────
    //
    // Layer 2 is also linear, and sits after a full-attention layer whose latent
    // plane is `CTX` tokens wide. If the region walk were wrong — a uniform
    // stride, say, or a latent charged to a linear layer — layer 2 would write
    // into layer 1's plane and the corruption would look like an attention bug
    // several layers downstream.
    {
        soma::F32Workspace ws;
        std::vector<std::byte> c2(total, std::byte{0});
        std::vector<float> got(Dm);
        soma::KvRow row{};
        row.opaque_base = c2.data();
        row.opaque_bytes = c2.size();
        row.max_ctx = CTX;
        row.pos = 0;
        row.len = 1;
        CHECK(soma::arch::kda::f32_attention_kv(arch, lw, x.data(), 1, /*layer=*/2, &row, ws,
                                                got.data()) == soma::StatusCode::Ok);

        const auto r2 = soma::arch::kda::layer_region(arch, 2, CTX);
        for (std::size_t i = 0; i < total; ++i) {
            const bool inside = i >= r2.recurrent && i < r2.end;
            if (!inside) CHECK(c2[i] == std::byte{0});
        }
        // …and it did write something inside it.
        bool touched = false;
        for (std::size_t i = r2.recurrent; i < r2.end; ++i)
            if (c2[i] != std::byte{0}) touched = true;
        CHECK(touched);
    }

    // ── F. a row with no cache is refused, not silently run stateless ────────
    {
        soma::F32Workspace ws;
        std::vector<float> got(Dm);
        soma::KvRow row{};
        row.max_ctx = CTX; // opaque_base deliberately null
        CHECK(soma::arch::kda::f32_attention_kv(arch, lw, x.data(), 1, 0, &row, ws, got.data()) !=
              soma::StatusCode::Ok);
        // And one whose buffer is too small for the region it would address.
        std::vector<std::byte> tiny(8, std::byte{0});
        row.opaque_base = tiny.data();
        row.opaque_bytes = tiny.size();
        CHECK(soma::arch::kda::f32_attention_kv(arch, lw, x.data(), 1, 0, &row, ws, got.data()) !=
              soma::StatusCode::Ok);
    }

    // ── G. the backend advertises the hybrid, and borrows the router ─────────
    {
        const auto& b = soma::arch::kda::f32_backend();
        CHECK(b.bind_layer != nullptr && b.attention != nullptr && b.attention_kv != nullptr);
        CHECK(b.route != nullptr);
        // No plane geometry: the cache is opaque, and a family that answered
        // this question would be describing planes it does not have.
        CHECK(b.kv_geometry == nullptr);
    }

    std::cout << "kda_layer: OK\n";
    return 0;
}
