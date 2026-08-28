// Soma — G3: the step-major scheduler.
//
// The claim under test is the one the whole design rests on:
//
//   BATCHING SEQUENCES TOGETHER MUST NOT CHANGE WHAT ANY OF THEM SAYS.
//
// A sequence generated alongside three others must produce exactly the tokens it
// would have produced alone. If that fails, every throughput number the engine
// reports is measuring a different model than the conformance ladder validated,
// and no amount of speed is worth it.
//
// The second claim is the payoff: rows batched from INDEPENDENT sequences are
// independent draws on the expert set, which is the case the batch union was
// designed for. Adjacent tokens of one prompt are correlated and flatter it.
//
// Usage: scheduler_g3 <fixtures_root> [fixture]

#include "soma/expert_store.hpp"
#include "soma/memory_hierarchy.hpp"
#include "soma/scheduler.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
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

/// Run `prompts` through one scheduler and collect each sequence's tokens.
std::map<soma::SeqId, std::vector<soma::TokenId>> run(
    const soma::F32Model& model, soma::MemoryHierarchy* mem,
    const std::vector<std::vector<soma::TokenId>>& prompts, std::uint32_t max_tokens,
    std::uint32_t kv_slots, std::uint32_t max_batch, soma::SchedulerStats* out_stats,
    bool* out_mixed = nullptr) {
    bool mixed = false;
    soma::SchedulerConfig cfg;
    cfg.kv_slots = kv_slots;
    cfg.ctx_size = 128;
    cfg.max_batch = max_batch;

    soma::Scheduler sched;
    if (auto st = sched.open_f32(model, mem, cfg); !st.ok()) {
        std::cerr << "open failed: " << st.message() << "\n";
        return {};
    }

    std::map<soma::SeqId, std::vector<soma::TokenId>> got;
    sched.set_token_callback(
        [&](soma::SeqId id, soma::TokenId t, bool) { got[id].push_back(t); });

    std::vector<soma::SeqId> ids;
    for (const auto& p : prompts) {
        soma::SeqRequest req;
        req.prompt = p;
        req.max_tokens = max_tokens;
        soma::SeqId id = 0;
        soma::AdmitRejection why{};
        if (auto st = sched.admit(std::move(req), id, why); !st.ok()) {
            std::cerr << "admit refused: " << soma::to_string(why) << "\n";
            return {};
        }
        ids.push_back(id);
    }

    // Bounded: a scheduler bug that never retires a sequence should fail the
    // test, not hang the suite.
    const std::uint32_t max_steps = 4000;
    std::uint32_t steps = 0;
    soma::SchedulerStats peak{};
    while (!sched.idle() && steps < max_steps) {
        if (auto st = sched.step(); !st.ok()) {
            std::cerr << "step failed: " << st.message() << "\n";
            break;
        }
        // Sample at the WIDEST step, not the last one.
        //
        // SchedulerStats reports `*_last_step` because that is what a live
        // telemetry feed wants. For this test that is the wrong moment: sequences
        // retire at different times, so the final step of a four-sequence run has
        // one row in it, and reading it made a working union look like a ratio of
        // exactly 1.00 — which is the signature of the union being broken.
        const auto s = sched.stats();
        if (s.current_batch > peak.current_batch) peak = s;
        // Mixing is a property of the RUN, not of any one step, so it is
        // accumulated rather than sampled. The widest step here is the first —
        // every sequence is still prefilling — and reading mixing off it reports
        // "0 decode rows" for a scheduler that interleaves correctly two steps
        // later.
        if (s.prefill_rows_last_step > 0 && s.decode_rows_last_step > 0) mixed = true;
        ++steps;
    }
    peak.steps = steps;
    // Cumulative counters come from the END; per-step ones from the peak.
    peak.tokens_out = sched.stats().tokens_out;
    if (out_stats) *out_stats = peak;
    if (out_mixed) *out_mixed = mixed;
    return got;
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

    std::cout << "== " << name << "  " << model.arch.n_moe_layers() << " MoE layers x "
              << model.arch.router.n_experts << " experts, top_k=" << model.arch.router.top_k
              << "\n";

    // Distinct prompts of DIFFERENT LENGTHS, so the batch is genuinely ragged:
    // sequences finish prefill at different steps and sit at different positions.
    // Equal-length prompts would march in lockstep and hide every off-by-one in
    // per-row position handling.
    const std::vector<std::vector<soma::TokenId>> prompts = {
        {3, 11, 29},
        {7, 5},
        {13, 2, 41, 17},
        {23},
    };
    constexpr std::uint32_t kMaxTokens = 12;

    // ── 1. one sequence at a time ────────────────────────────────────────────
    std::cout << "\n1. baseline: each sequence alone (batch of 1)\n";
    std::map<soma::SeqId, std::vector<soma::TokenId>> alone;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        auto r = run(model, nullptr, {prompts[i]}, kMaxTokens, 1, 1, nullptr);
        if (r.size() != 1) {
            check(false, "solo run produced one sequence");
            return 1;
        }
        alone[static_cast<soma::SeqId>(i)] = r.begin()->second;
        std::cout << "   seq " << i << "  " << r.begin()->second.size() << " tokens:";
        for (const auto t : r.begin()->second) std::cout << " " << t;
        std::cout << "\n";
    }

    // ── 2. all four together ─────────────────────────────────────────────────
    std::cout << "\n2. batched: all four in one scheduler\n";
    soma::SchedulerStats st{};
    bool mixed = false;
    const auto together = run(model, nullptr, prompts, kMaxTokens, 8, 8, &st, &mixed);
    check(together.size() == prompts.size(), "every sequence retired",
          std::to_string(together.size()) + "/" + std::to_string(prompts.size()));

    std::size_t idx = 0;
    bool all_match = true;
    for (const auto& [id, toks] : together) {
        const auto& want = alone[static_cast<soma::SeqId>(idx)];
        const bool same = (toks == want);
        all_match &= same;
        if (!same) {
            std::cout << "   seq " << idx << " DIFFERS\n     alone:";
            for (const auto t : want) std::cout << " " << t;
            std::cout << "\n     batched:";
            for (const auto t : toks) std::cout << " " << t;
            std::cout << "\n";
        }
        ++idx;
    }
    check(all_match, "batched output is identical to solo output",
          "batching must not change what a sequence says");

    check(st.steps > 0 && st.tokens_out > 0, "the step loop actually ran",
          std::to_string(st.steps) + " steps, " + std::to_string(st.tokens_out) + " tokens");

    // Ragged by construction: prompts of length 1..4 mean prefill and decode rows
    // coexist in the same step. If the scheduler had serialised prefill ahead of
    // decode, one of these counters would stay at zero at the widest step.
    std::cout << "   widest step: " << st.prefill_rows_last_step << " prefill + "
              << st.decode_rows_last_step << " decode rows, batch " << st.current_batch
              << ", gate " << st.effective_max_batch << "\n";
    check(st.current_batch == prompts.size(), "all four sequences shared one step",
          "batch " + std::to_string(st.current_batch));
    // Mixing is reported, not asserted, HERE. With a 512-token chunk budget and
    // prompts of 1-4 tokens, every sequence finishes prefill in step 1, so there
    // is no step left in which prefill and decode could coexist. That is chunked
    // prefill working, not ragged batching failing — the old version of this
    // check asserted mixing and started failing the moment prefill got fast.
    // Section 5 tests it where it is actually load-bearing.
    std::cout << "   prefill/decode mixing seen: " << (mixed ? "yes" : "no (prefill fits in one step)")
              << "\n";

    // ── 3. the union, across sequences ───────────────────────────────────────
    std::cout << "\n3. union across concurrent sequences\n";
    {
        soma::SchedulerStats s1{}, s4{};
        (void)run(model, nullptr, {prompts[0]}, kMaxTokens, 1, 1, &s1);
        (void)run(model, nullptr, prompts, kMaxTokens, 8, 8, &s4);

        const double r1 = static_cast<double>(s1.naive_expert_reads_last_step) /
                          std::max(1u, s1.unique_experts_last_step);
        const double r4 = static_cast<double>(s4.naive_expert_reads_last_step) /
                          std::max(1u, s4.unique_experts_last_step);

        std::cout << "   batch 1   naive " << s1.naive_expert_reads_last_step << ", unique "
                  << s1.unique_experts_last_step << "  ratio " << std::fixed
                  << std::setprecision(2) << r1 << "\n"
                  << "   batch " << s4.current_batch << "   naive "
                  << s4.naive_expert_reads_last_step << ", unique "
                  << s4.unique_experts_last_step << "  ratio " << r4 << "\n";

        check(r4 > r1, "union ratio is higher with concurrent sequences",
              "rows from independent sequences share experts");
        check(s4.naive_expert_reads_last_step > s1.naive_expert_reads_last_step,
              "a batched step really does more expert work");
    }

    // ── 4. the cache-aware gate ──────────────────────────────────────────────
    //
    // A gate that never bites is decoration. With a cache too small for the
    // expected unique set, effective_max_batch must DROP — and the run must still
    // complete, because throttling is the correct response to pressure and
    // thrashing is not.
    std::cout << "\n4. cache-aware admission gate\n";
    {
        // The container is q4_g, and ExpertStore::open cross-checks its expert
        // size against the IR's quantization map. Loading with the default (all
        // f32) map makes that check fail — correctly. The model must be loaded
        // the way the container was written.
        soma::QuantMap cmap;
        cmap.expert_gate = {soma::DType::Q4_G, 128};
        cmap.expert_up = {soma::DType::Q4_G, 128};
        cmap.expert_down = {soma::DType::Q6_G, 128};

        soma::F32Model m;
        if (auto s = soma::load_f32_model((root / "tiny" / name).string(), m, cmap); !s.ok()) {
            std::cout << "   (quantized load failed; gate check skipped)\n";
        } else if (soma::ExpertStore store; !store.open((root / "containers" / name).string(),
                                                        m.arch).ok()) {
            std::cout << "   (no container; gate check skipped)\n";
        } else {

            soma::MemoryHierarchy tight, roomy;
            soma::MemoryBudget b_tight, b_roomy;
            b_tight.ram_expert_cache_bytes = store.header().expert_bytes * 2;
            b_roomy.ram_expert_cache_bytes = store.header().expert_bytes * 4096;
            (void)tight.open(m.arch, store, b_tight);
            (void)roomy.open(m.arch, store, b_roomy);

            soma::SchedulerConfig cfg;
            cfg.kv_slots = 8;
            cfg.ctx_size = 128;
            soma::Scheduler a, b;
            (void)a.open_f32(m, &tight, cfg);
            (void)b.open_f32(m, &roomy, cfg);

            std::cout << "   tight cache (2 experts)   gate = " << a.effective_max_batch()
                      << "\n   roomy cache (4096)        gate = " << b.effective_max_batch()
                      << "\n";
            check(a.effective_max_batch() < b.effective_max_batch(),
                  "a small cache lowers effective_max_batch",
                  "the gate bites instead of thrashing");
            check(a.effective_max_batch() >= 1, "the gate never reaches zero",
                  "a stalled engine is not a throttled one");
        }
    }

    // ── 5. chunked prefill and the fairness cap ──────────────────────────────
    //
    // The case the caps exist for: one long prompt sharing a batch with short
    // interactive turns. Chunking must (a) not change any sequence's output,
    // (b) let the short sequences decode WHILE the long one is still prefilling,
    // and (c) shrink the step count as the chunk grows.
    std::cout << "\n5. chunked prefill with a long prompt\n";
    {
        std::vector<soma::TokenId> long_prompt(40);
        for (std::size_t i = 0; i < long_prompt.size(); ++i) {
            long_prompt[i] = static_cast<soma::TokenId>((i * 13 + 7) % 500);
        }
        const std::vector<std::vector<soma::TokenId>> mix = {
            long_prompt, {7, 5}, {23}, {13, 2, 41},
        };

        // Chunk sizes chosen to straddle the prompt length: 4 forces ten steps of
        // prefill for the long sequence, 64 swallows it whole.
        std::map<std::uint32_t, std::vector<soma::TokenId>> first_seq;
        std::map<std::uint32_t, std::uint64_t> step_count;
        bool any_mixed = false;

        for (const std::uint32_t chunk : {4u, 8u, 64u}) {
            soma::SchedulerConfig c;
            c.kv_slots = 8;
            c.ctx_size = 128;
            c.max_batch = 8;
            c.prefill_chunk_tokens = chunk;
            c.max_prefill_rows_per_seq = chunk;

            soma::Scheduler sched;
            (void)sched.open_f32(model, nullptr, c);
            std::map<soma::SeqId, std::vector<soma::TokenId>> got;
            sched.set_token_callback(
                [&](soma::SeqId id, soma::TokenId t, bool) { got[id].push_back(t); });

            std::vector<soma::SeqId> ids;
            for (const auto& p : mix) {
                soma::SeqRequest req;
                req.prompt = p;
                req.max_tokens = 8;
                soma::SeqId id = 0;
                soma::AdmitRejection why{};
                (void)sched.admit(std::move(req), id, why);
                ids.push_back(id);
            }

            std::uint32_t steps = 0;
            std::uint32_t widest_prefill = 0;
            while (!sched.idle() && steps < 2000) {
                if (auto s = sched.step(); !s.ok()) break;
                const auto st2 = sched.stats();
                if (st2.prefill_rows_last_step > 0 && st2.decode_rows_last_step > 0) {
                    any_mixed = true;
                }
                widest_prefill = std::max(widest_prefill, st2.prefill_rows_last_step);
                ++steps;
            }
            step_count[chunk] = steps;
            first_seq[chunk] = got[ids.front()];

            std::cout << "   chunk " << std::setw(3) << chunk << ": " << std::setw(3) << steps
                      << " steps, widest prefill " << widest_prefill << " rows\n";

            // The per-sequence cap, checked against the widest step actually
            // taken. Four sequences prefilling at once could otherwise stack
            // into one enormous forward.
            check(widest_prefill <= chunk * mix.size(),
                  "chunk " + std::to_string(chunk) + ": prefill rows stay bounded",
                  std::to_string(widest_prefill) + " <= " +
                      std::to_string(chunk * mix.size()));
        }

        // The claim that matters: chunk size is a SCHEDULING decision and must not
        // be visible in the output. If it is, prefill rows are contributing to
        // each other's attention differently depending on how they were grouped.
        const bool stable = (first_seq[4u] == first_seq[8u]) && (first_seq[8u] == first_seq[64u]);
        check(stable, "output is identical across chunk sizes 4 / 8 / 64",
              stable ? "chunking is invisible to the model"
                     : "DIVERGED — grouping changed attention");

        check(any_mixed, "a long prompt prefills while others decode",
              "the fairness cap lets interactive turns through");
        check(step_count[64u] < step_count[4u], "a larger chunk finishes in fewer steps",
              std::to_string(step_count[4u]) + " -> " + std::to_string(step_count[64u]));
    }

    std::cout << "\n" << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES")
              << "\n";
    return g_failures == 0 ? 0 : 1;
}
