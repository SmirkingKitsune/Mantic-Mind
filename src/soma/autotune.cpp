// Soma — shape extraction and measurement.
//
// Runs at admission, never while serving. The output is a table the loader
// resolves against once per weight; the hot path never asks a timing question.

#include "soma/kernel_registry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

namespace soma {

namespace {

using Clock = std::chrono::steady_clock;

void push_unique(std::vector<KernelShape>& v, const KernelShape& s) {
    for (const auto& e : v) {
        if (e.op == s.op && e.m == s.m && e.n == s.n && e.k == s.k && e.dtype == s.dtype) return;
    }
    v.push_back(s);
}

} // namespace

std::vector<KernelShape> model_shapes(const ArchIr& arch,
                                      const QuantMap& quant,
                                      std::span<const std::uint32_t> batch_sizes) {
    std::vector<KernelShape> out;

    const auto d = arch.topology.d_model;
    const auto hq = arch.attention.n_heads * arch.attention.head_dim;
    const auto hkv = arch.attention.n_kv_heads * arch.attention.head_dim;
    const auto fi = arch.ffn.expert_intermediate;
    const auto di = arch.ffn.dense_intermediate;

    const auto attn = quant.attn_proj.dtype;
    const auto eg = quant.expert_gate.dtype;
    const auto eu = quant.expert_up.dtype;
    const auto ed = quant.expert_down.dtype;
    const auto emb = quant.embed.dtype;

    for (const auto m : batch_sizes) {
        const auto op = (m == 1) ? KernelOp::Gemv : KernelOp::Gemm;

        push_unique(out, {op, m, hq, d, attn});  // q_proj
        push_unique(out, {op, m, hkv, d, attn}); // k_proj / v_proj
        push_unique(out, {op, m, d, hq, attn});  // o_proj

        // The router is F32 by schema constraint, so it is tuned as F32 no
        // matter what the rest of the map says.
        if (arch.router.n_experts > 0) {
            push_unique(out, {op, m, arch.router.n_experts, d, DType::F32});
        }
        push_unique(out, {op, m, arch.topology.vocab_size, d, emb}); // lm_head
        if (di > 0) {
            push_unique(out, {op, m, di, d, eg});
            push_unique(out, {op, m, d, di, ed});
        }
    }

    // Experts are ALWAYS applied one row at a time, even during prefill: a token
    // only visits the experts it routed to, so there is no batch to gather. That
    // makes m == 1 the shape that matters most for the streaming path, and it is
    // exactly where dequantize cost has nothing to amortize against.
    if (fi > 0) {
        push_unique(out, {KernelOp::Gemv, 1, fi, d, eg});
        push_unique(out, {KernelOp::Gemv, 1, fi, d, eu});
        push_unique(out, {KernelOp::Gemv, 1, d, fi, ed});
    }
    return out;
}

Status autotune_shapes(std::span<const KernelShape> shapes,
                       const AutotuneOptions& opts,
                       std::vector<TuneResult>& out) {
    out.clear();
    out.reserve(shapes.size());

    std::mt19937 rng(20260729);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);

    std::vector<float> wf32, x, y, scratch, ref;
    QTensor qw;

    for (const auto& shape : shapes) {
        const auto cands = candidates(shape.op, shape.dtype);
        if (cands.empty()) continue;

        const std::size_t elems = static_cast<std::size_t>(shape.n) * shape.k;
        wf32.resize(elems);
        for (auto& v : wf32)
            v = dist(rng);
        x.assign(shape.k, 0.0f);
        for (auto& v : x)
            v = dist(rng);
        y.assign(shape.n, 0.0f);
        ref.assign(shape.n, 0.0f);
        scratch.assign(shape.k, 0.0f);

        const bool quantized = is_quantized(shape.dtype);
        if (quantized) {
            if (auto s = quantize_tensor(wf32, shape.n, shape.k, shape.dtype, kDefaultGroup, qw);
                !s.ok()) {
                continue; // shape not expressible in this format; skip, do not fail
            }
        }

        TuneResult best;
        best.shape = shape;
        best.gflops = -1.0f;

        for (std::size_t ci = 0; ci < cands.size(); ++ci) {
            const auto& impl = cands[ci];
            auto run = [&]() {
                if (quantized) {
                    impl.q_fn(qw, x.data(), y.data(), scratch.data());
                } else {
                    impl.f32_fn(wf32.data(), x.data(), shape.n, shape.k, y.data());
                }
            };

            for (std::uint32_t i = 0; i < opts.warmup_iters; ++i)
                run();

            // Correctness before speed. Candidates for one (op, dtype) must
            // agree, or the autotuner is picking on time while varying output —
            // which would be a silent accuracy regression selected FOR.
            if (ci == 0) {
                ref.assign(y.begin(), y.end());
            } else {
                float worst = 0.0f, mag = 0.0f;
                for (std::uint32_t i = 0; i < shape.n; ++i) {
                    worst = std::max(worst, std::fabs(y[i] - ref[i]));
                    mag = std::max(mag, std::fabs(ref[i]));
                }
                const float tol = std::max(1e-4f, mag * 1e-4f);
                if (worst > tol) {
                    return {StatusCode::Internal,
                            std::string("kernel '") + impl.name + "' disagrees with '" +
                                cands[0].name + "' by " + std::to_string(worst) + " at n=" +
                                std::to_string(shape.n) + " k=" + std::to_string(shape.k) +
                                "; candidates must be numerically equivalent"};
                }
            }

            // Enough repeats that the measurement clears timer resolution.
            std::uint32_t reps = 1;
            for (;;) {
                const auto t0 = Clock::now();
                for (std::uint32_t i = 0; i < reps; ++i)
                    run();
                const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
                if (secs >= opts.min_seconds_per_measure || reps > (1u << 20)) break;
                reps *= 4;
            }

            double best_secs = 1e30;
            for (std::uint32_t trial = 0; trial < opts.measure_iters; ++trial) {
                const auto t0 = Clock::now();
                for (std::uint32_t i = 0; i < reps; ++i)
                    run();
                const double secs = std::chrono::duration<double>(Clock::now() - t0).count() / reps;
                best_secs = opts.take_minimum ? std::min(best_secs, secs)
                                              : (best_secs > 1e29 ? secs : (best_secs + secs) / 2);
            }

            const double flops = 2.0 * static_cast<double>(shape.n) * shape.k;
            const auto gflops = static_cast<float>(flops / best_secs / 1e9);
            best.all.emplace_back(impl.name, gflops);
            if (gflops > best.gflops) {
                best.gflops = gflops;
                best.impl = impl.name;
            }
        }
        out.push_back(std::move(best));
    }
    return {};
}

} // namespace soma
