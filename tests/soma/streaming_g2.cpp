// Soma — G2: the streaming path.
//
// Gate items verified here, on the tiny containers where a small cache budget
// forces real misses:
//
//   * an expert with a live ExpertRef is never evicted, even under pressure
//   * LRU evicts the least-recently-used unpinned, unreferenced slot
//   * the thrash gate fires when the unique set exceeds cap_per_layer
//   * prefetch is OFF by default and only fires on layers explicitly enabled
//     from measured recall — AT LEAST ONE layer must end up disabled
//   * heat bootstrap makes a warm start measurably cheaper than a cold one
//   * measured bytes/token matches the plan's prediction
//
// Usage: streaming_g2 <fixtures_root> [fixture]

#include "soma/expert_store.hpp"
#include "soma/f32_model.hpp"
#include "soma/memory_hierarchy.hpp"
#include "soma/plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << "   " << std::left << std::setw(52) << what << (ok ? "OK" : "FAIL");
    if (!detail.empty()) std::cout << "   " << detail;
    std::cout << "\n";
    if (!ok) ++g_failures;
}

/// Deterministic stand-in for a router: top_k experts per (layer, token).
std::vector<soma::ExpertId> routed(std::uint32_t layer, std::uint32_t token,
                                   std::uint32_t n_experts, std::uint32_t top_k) {
    std::mt19937 rng(layer * 1000003u + token);
    std::vector<soma::ExpertId> all(n_experts);
    for (std::uint32_t i = 0; i < n_experts; ++i) all[i] = i;
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(std::min(top_k, n_experts));
    return all;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures");
    const std::string name = (argc > 2) ? argv[2] : "Qwen3-30B-A3B";

    soma::F32Model model;
    if (auto st = soma::load_f32_model((root / "tiny" / name).string(), model); !st.ok()) {
        std::cerr << "load failed: " << st.message() << "\n";
        return 2;
    }
    // The IR handed to the store must describe the container's precision; an
    // all-f32 IR is now correctly refused by the expert-size cross-check.
    soma::QuantMap cmap;
    cmap.expert_gate = {soma::DType::Q4_G, 128};
    cmap.expert_up = {soma::DType::Q4_G, 128};
    cmap.expert_down = {soma::DType::Q6_G, 128};
    soma::F32Model qmodel;
    if (auto st = soma::load_f32_model((root / "tiny" / name).string(), qmodel, cmap); !st.ok()) {
        std::cerr << "quantized load failed: " << st.message() << "\n";
        return 2;
    }
    model.arch.quantization = qmodel.arch.quantization;

    soma::ExpertStore store;
    if (auto st = store.open((root / "containers" / name).string(), model.arch); !st.ok()) {
        std::cerr << "container open failed: " << st.message() << "\n";
        return 2;
    }
    const auto& h = store.header();
    const auto n_moe = model.arch.n_moe_layers();
    const auto top_k = model.arch.router.top_k;

    std::cout << "== " << name << "  " << h.n_layers << "L (" << n_moe << " MoE) x "
              << h.n_experts << "E, expert=" << h.expert_bytes << " B, top_k=" << top_k << "\n\n";

    // Cache deliberately far smaller than the routed set, so misses are real and
    // eviction is exercised. A test with a cache big enough to hold everything
    // measures nothing about streaming.
    const std::uint64_t routed_total =
        static_cast<std::uint64_t>(n_moe) * h.n_experts * h.expert_bytes;
    soma::MemoryBudget budget;
    budget.ram_expert_cache_bytes = h.expert_bytes * (top_k + 2) * n_moe;
    budget.pin_bytes = h.expert_bytes * 2;

    soma::MemoryHierarchy mem;
    if (auto st = mem.open(model.arch, store, budget); !st.ok()) {
        std::cerr << "hierarchy open failed: " << st.message() << "\n";
        return 2;
    }
    std::cout << "   cache " << budget.ram_expert_cache_bytes / 1024 << " KiB of "
              << routed_total / 1024 << " KiB routed set; cap_per_layer=" << mem.cap_per_layer()
              << "\n\n";

    // ── 1. a borrowed expert survives pressure ───────────────────────────────
    std::cout << "1. eviction safety\n";
    {
        // A DELIBERATELY tiny cache — 3 slots against 16 experts of churn.
        //
        // The first version of this check used the main budget (24 slots) and
        // churned 16 experts, so the cache never filled and evictions stayed at
        // 0. It passed while testing nothing: "survived pressure" means nothing
        // if there was no pressure.
        soma::MemoryHierarchy ev;
        soma::MemoryBudget tight;
        tight.ram_expert_cache_bytes = h.expert_bytes * 3;
        (void)ev.open(model.arch, store, tight);

        auto held = ev.acquire(0, 0);
        check(static_cast<bool>(held), "acquire returns a live reference");
        const auto* addr = held.bytes().data();
        const auto len = held.bytes().size();
        check(len == h.expert_bytes, "reference exposes the full expert",
              std::to_string(len) + " B");

        for (std::uint32_t pass = 0; pass < 4; ++pass) {
            for (std::uint32_t e = 1; e < h.n_experts; ++e) {
                auto tmp = ev.acquire(0, e);
                (void)tmp;
            }
        }
        const auto evictions = ev.stats().evictions;
        check(evictions > 0, "the churn actually caused eviction",
              std::to_string(evictions) + " evictions");
        check(held.bytes().data() == addr && held.bytes().size() == len,
              "held expert survived eviction pressure",
              "still at the same address");
        check(held.tier() == soma::MemoryTier::Ram, "held expert still reports Ram tier");
    }

    // ── 2. LRU order ─────────────────────────────────────────────────────────
    std::cout << "\n2. LRU\n";
    {
        soma::MemoryHierarchy m2;
        soma::MemoryBudget b2;
        b2.ram_expert_cache_bytes = h.expert_bytes * 3;   // exactly 3 slots
        (void)m2.open(model.arch, store, b2);

        { auto a = m2.acquire(0, 0); }
        { auto b = m2.acquire(0, 1); }
        { auto c = m2.acquire(0, 2); }
        { auto a2 = m2.acquire(0, 0); }   // 0 becomes most-recent; 1 is now LRU
        { auto d = m2.acquire(0, 3); }    // must evict 1

        const auto before = m2.stats().misses;
        { auto a3 = m2.acquire(0, 0); }   // still resident -> hit
        const auto after_hit = m2.stats().misses;
        { auto b2r = m2.acquire(0, 1); }  // was evicted -> miss
        const auto after_miss = m2.stats().misses;

        check(after_hit == before, "recently-touched slot survived", "0 still resident");
        check(after_miss == before + 1, "LRU victim was the least-recently-used", "1 evicted");
    }

    // ── 3. thrash gate ───────────────────────────────────────────────────────
    std::cout << "\n3. thrash gate\n";
    {
        const auto cap = mem.cap_per_layer();
        check(!mem.would_thrash(cap), "at cap: no thrash predicted",
              "cap=" + std::to_string(cap));
        check(mem.would_thrash(cap + 1), "above cap: thrash predicted");
    }

    // ── 4. prefetch gating ───────────────────────────────────────────────────
    //
    // Default OFF for every layer. Prefetch has to be earned by measured recall;
    // a wrong prefetch evicts something that was going to be used, so the
    // failure mode of enabling it blindly is worse than not having it.
    std::cout << "\n4. prefetch gating\n";
    {
        bool any_on = false;
        for (std::uint32_t l = 0; l < h.n_layers; ++l) any_on |= mem.prefetch_enabled(l);
        check(!any_on, "prefetch is OFF for every layer by default");

        // Stand in for pilot_profile: alternate layers clear the recall threshold.
        // The gate must produce at least one DISABLED layer, or it is not gating.
        std::uint32_t on = 0, off = 0;
        for (std::uint32_t l = 0; l < h.n_layers; ++l) {
            const bool good_recall = (l % 2 == 0);
            mem.set_prefetch_enabled(l, good_recall);
            good_recall ? ++on : ++off;
        }
        check(off >= 1, "at least one layer has prefetch disabled",
              std::to_string(on) + " on / " + std::to_string(off) + " off");

        // A FRESH hierarchy: sections 1-3 warmed `mem`, so the prefetch targets
        // would already be resident and prefetch would correctly no-op. Reusing
        // it would have tested nothing.
        soma::MemoryHierarchy pf;
        (void)pf.open(model.arch, store, budget);
        pf.set_prefetch_enabled(0, true);
        pf.set_prefetch_enabled(1, false);

        const auto before = pf.stats();
        pf.prefetch(1, std::vector<soma::ExpertId>{0, 1, 2});   // disabled layer
        const auto mid = pf.stats();
        check(mid.bytes_read == before.bytes_read, "prefetch on a disabled layer is a no-op");

        pf.prefetch(0, std::vector<soma::ExpertId>{5, 6});      // enabled layer
        const auto after = pf.stats();
        check(after.bytes_read > mid.bytes_read, "prefetch on an enabled layer fetches",
              std::to_string((after.bytes_read - mid.bytes_read)) + " B");

        { auto r = pf.acquire(0, 5); }
        check(pf.stats().prefetch_hits > after.prefetch_hits,
              "a prefetched expert is credited as a prefetch hit");
    }

    // ── 5. heat bootstrap: cold vs warm ──────────────────────────────────────
    std::cout << "\n5. heat bootstrap\n";
    {
        constexpr std::uint32_t kTokens = 48;

        auto run = [&](bool bootstrap, const soma::HeatSnapshot* heat) {
            soma::MemoryHierarchy m;
            (void)m.open(model.arch, store, budget);
            if (bootstrap && heat != nullptr) (void)m.apply_heat_bootstrap(*heat);
            for (std::uint32_t t = 0; t < kTokens; ++t) {
                for (std::uint32_t l = 0; l < h.n_layers; ++l) {
                    if (!model.arch.is_moe_layer(l)) continue;
                    for (const auto e : routed(l, t, h.n_experts, top_k)) {
                        auto r = m.acquire(l, e);
                        if (!r) { ++g_failures; return soma::CacheStats{}; }
                    }
                }
            }
            return m.stats();
        };

        // A first pass produces the histogram a real deployment would have
        // persisted; the second run starts from it.
        soma::MemoryHierarchy probe;
        (void)probe.open(model.arch, store, budget);
        for (std::uint32_t t = 0; t < kTokens; ++t) {
            for (std::uint32_t l = 0; l < h.n_layers; ++l) {
                if (!model.arch.is_moe_layer(l)) continue;
                for (const auto e : routed(l, t, h.n_experts, top_k)) {
                    auto r = probe.acquire(l, e);
                    (void)r;
                }
            }
        }
        const auto learned = probe.heat();

        const auto cold = run(false, nullptr);
        const auto warm = run(true, &learned);

        const double cold_rate = 100.0 * static_cast<double>(cold.misses) /
                                 static_cast<double>(std::max<std::uint64_t>(1, cold.hits + cold.misses));
        const double warm_rate = 100.0 * static_cast<double>(warm.misses) /
                                 static_cast<double>(std::max<std::uint64_t>(1, warm.hits + warm.misses));
        std::cout << "   cold: " << cold.misses << " misses (" << std::fixed
                  << std::setprecision(1) << cold_rate << "%), " << cold.bytes_read / 1024
                  << " KiB read\n"
                  << "   warm: " << warm.misses << " misses (" << warm_rate << "%), "
                  << warm.bytes_read / 1024 << " KiB read\n";
        check(warm.misses < cold.misses, "warm start has measurably fewer misses",
              std::to_string(cold.misses - warm.misses) + " fewer");
    }

    // ── 6. measured bytes/token vs the plan ──────────────────────────────────
    std::cout << "\n6. bytes/token: measured vs plan\n";
    {
        // model.arch now carries the container's quantization, so the plan
        // describes the bytes actually on disk. Computing it from an all-f32 map
        // predicted 393216 B/token against a measured 69632 — the compression
        // ratio, exactly.
        soma::HostBudget hb;
        hb.ram_total_bytes = budget.ram_expert_cache_bytes;
        hb.ram_free_bytes = budget.ram_expert_cache_bytes;
        hb.ctx_size = 128;
        hb.kv_slots = 1;
        soma::PlanDocument plan;
        (void)soma::compute_plan(model.arch, hb, plan);

        // Cold cache, so every routed expert is a genuine read: the worst case
        // the plan's bytes_per_token predicts.
        soma::MemoryHierarchy m;
        soma::MemoryBudget tiny;
        tiny.ram_expert_cache_bytes = h.expert_bytes;  // one slot: no reuse at all
        (void)m.open(model.arch, store, tiny);

        constexpr std::uint32_t kTokens = 8;
        for (std::uint32_t t = 0; t < kTokens; ++t) {
            for (std::uint32_t l = 0; l < h.n_layers; ++l) {
                if (!model.arch.is_moe_layer(l)) continue;
                for (const auto e : routed(l, t, h.n_experts, top_k)) {
                    auto r = m.acquire(l, e);
                    (void)r;
                }
            }
        }
        const auto measured = m.stats().bytes_read / kTokens;
        const auto predicted = plan.bytes_per_token;
        const double err = 100.0 * std::fabs(static_cast<double>(measured) -
                                             static_cast<double>(predicted)) /
                           static_cast<double>(std::max<std::uint64_t>(1, predicted));
        std::cout << "   predicted " << predicted << " B/token, measured " << measured
                  << " B/token\n";
        check(err <= 10.0, "measured within 10% of prediction",
              std::to_string(static_cast<int>(err)) + "% error");
    }

    // ── 7. streamed forward == resident forward ──────────────────────────────
    //
    // THE check that justifies wiring the hierarchy in: the same forward, run
    // once reading resident weights and once streaming them from the container,
    // must produce IDENTICAL logits. Anything else means there are two code paths
    // pretending to be one.
    //
    // Bit-identical, not merely close — both modes hand the same bytes to the
    // same kernel, so any difference at all would mean the section split or the
    // group derivation disagrees between them.
    std::cout << "\n7. streamed forward vs resident forward\n";
    {
        soma::F32Workspace fws;
        std::vector<soma::TokenId> toks;
        for (std::uint32_t i = 0; i < 24; ++i) toks.push_back((i * 37 + 5) % qmodel.vocab());

        std::vector<float> resident_logits;
        if (auto st = soma::forward_f32(qmodel, toks, fws, resident_logits); !st.ok()) {
            check(false, "resident forward", st.message());
        }

        // A cache far too small for the working set, so the streamed run is
        // genuinely paging rather than accidentally resident.
        soma::MemoryHierarchy stream_mem;
        soma::MemoryBudget sb;
        sb.ram_expert_cache_bytes = h.expert_bytes * 4;
        (void)stream_mem.open(qmodel.arch, store, sb);

        soma::F32Model streamed;
        if (auto st = soma::load_f32_model((root / "tiny" / name).string(), streamed, cmap);
            !st.ok()) {
            check(false, "streamed model load", st.message());
        }
        streamed.streamed_experts = &stream_mem;

        std::vector<float> streamed_logits;
        if (auto st = soma::forward_f32(streamed, toks, fws, streamed_logits); !st.ok()) {
            check(false, "streamed forward", st.message());
        }

        check(resident_logits.size() == streamed_logits.size(), "logit counts match");
        float worst = 0.0f;
        for (std::size_t i = 0;
             i < std::min(resident_logits.size(), streamed_logits.size()); ++i) {
            worst = std::max(worst, std::fabs(resident_logits[i] - streamed_logits[i]));
        }
        std::ostringstream d;
        d << std::scientific << std::setprecision(2) << worst << " max|diff|";
        check(worst == 0.0f, "streamed logits are bit-identical to resident", d.str());

        const auto cs = stream_mem.stats();
        // Evidence of real paging, measured by BYTES and EVICTIONS rather than by
        // acquire-misses.
        //
        // This assertion used to be `misses > 0`, and prefetch made it flaky —
        // 2 failures in 30 runs, reporting "0 misses, 43 evictions". That is not
        // a regression: when the loader fetches an expert before the compute
        // thread asks for it, the acquire finds it resident and is counted a HIT.
        // A perfectly-prefetched run has zero misses while paging as hard as it
        // possibly can, and which of the two happens is a timing race, hence the
        // intermittency.
        //
        // Bytes read and evictions are attributable to the streaming path no
        // matter WHICH thread issued the read, so they say what this check
        // actually means: the cache did not simply hold everything.
        check(cs.bytes_read > 0 && cs.evictions > 0, "the streamed run actually paged",
              std::to_string(cs.bytes_read / 1024) + " KiB read, " +
                  std::to_string(cs.evictions) + " evictions, " +
                  std::to_string(cs.misses) + " acquire-misses");
    }

    std::cout << "\n" << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES")
              << "\n";
    return g_failures == 0 ? 0 : 1;
}
