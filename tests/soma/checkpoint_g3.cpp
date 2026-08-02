// Soma — G3: KV checkpoints, preemption, and the gating that makes them safe.
//
// The gate: PREEMPT MID-GENERATION AND RESUME MUST PRODUCE THE SAME
// CONTINUATION AS NEVER HAVING BEEN INTERRUPTED. Not similar — the same tokens.
// A checkpoint that restores an approximation of the KV state is worse than no
// checkpoint, because the divergence appears mid-sentence and looks like a model
// quality problem rather than a persistence bug.
//
// The other half is the refusals. Every load is gated on version, arch_hash and
// format_id, and each has to FAIL rather than read: a checkpoint replayed into a
// different attention family has the wrong cache shape, all the bytes parse, and
// the output is quietly degraded.
//
// Usage: checkpoint_g3 <fixtures_root> [fixture]

#include "soma/kv_checkpoint.hpp"
#include "soma/scheduler.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
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

std::string tok_str(const std::vector<soma::TokenId>& v) {
    std::string s;
    for (const auto t : v) s += " " + std::to_string(t);
    return s;
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

    const auto dir = fs::temp_directory_path() / "soma_ckpt_g3";
    std::error_code ec;
    fs::remove_all(dir, ec);

    soma::KvCheckpointStore store;
    if (auto st = store.open(dir.string(), model.arch); !st.ok()) {
        std::cerr << "store open failed: " << st.message() << "\n";
        return 2;
    }

    const std::vector<soma::TokenId> prompt{3, 11, 29};
    constexpr std::uint32_t kMaxTokens = 14;

    soma::SchedulerConfig cfg;
    cfg.kv_slots = 4;
    cfg.ctx_size = 128;
    cfg.max_batch = 4;

    const auto run_to_completion = [&](std::uint32_t preempt_after) {
        soma::Scheduler sched;
        (void)sched.open_f32(model, nullptr, cfg, &store);

        std::vector<soma::TokenId> out;
        sched.set_token_callback(
            [&](soma::SeqId, soma::TokenId t, bool) { out.push_back(t); });

        soma::SeqRequest req;
        req.prompt = prompt;
        req.max_tokens = kMaxTokens;
        soma::SeqId id = 0;
        soma::AdmitRejection why{};
        if (auto st = sched.admit(std::move(req), id, why); !st.ok()) return out;

        std::uint32_t steps = 0;
        bool done_preempt = false;
        while (steps < 500) {
            if (preempt_after > 0 && steps == preempt_after && !done_preempt) {
                if (auto st = sched.preempt(id); !st.ok()) {
                    std::cerr << "preempt failed: " << st.message() << "\n";
                    return out;
                }
                if (auto st = sched.resume(id); !st.ok()) {
                    std::cerr << "resume failed: " << st.message() << "\n";
                    return out;
                }
                done_preempt = true;
            }
            if (sched.idle()) break;
            if (auto st = sched.step(); !st.ok()) break;
            ++steps;
        }
        return out;
    };

    // ── 1. uninterrupted baseline ────────────────────────────────────────────
    std::cout << "== " << name << "\n\n1. baseline, no preemption\n";
    const auto baseline = run_to_completion(0);
    std::cout << "  " << baseline.size() << " tokens:" << tok_str(baseline) << "\n";
    check(baseline.size() == kMaxTokens, "baseline generated the full continuation",
          std::to_string(baseline.size()) + "/" + std::to_string(kMaxTokens));

    // ── 2. preempt mid-generation, resume, continue ──────────────────────────
    //
    // Preempted at several different points, because a checkpoint taken during
    // prefill and one taken mid-decode exercise different lengths — and an
    // off-by-one in how many positions get written would survive a single
    // well-chosen preemption point.
    std::cout << "\n2. preempt -> resume at several points\n";
    for (const std::uint32_t at : {1u, 3u, 5u, 9u}) {
        const auto got = run_to_completion(at);
        const bool same = (got == baseline);
        std::cout << "   preempt after step " << std::setw(2) << at << ": " << got.size()
                  << " tokens" << (same ? "" : tok_str(got)) << "\n";
        check(same, "continuation identical after preempt at step " + std::to_string(at),
              same ? "byte-for-byte" : "DIVERGED");
    }

    // ── 3. the refusals ──────────────────────────────────────────────────────
    //
    // Each of these must FAIL. A checkpoint store that reads a mismatched file
    // produces plausible output from the wrong cache shape, which is the single
    // most confusing failure this subsystem can emit.
    std::cout << "\n3. gating: mismatched checkpoints are refused\n";
    {
        soma::KvCache kv;
        (void)kv.open(model.arch, 64);
        (void)kv.set_length(8);
        check(store.save("gate-probe", kv).ok(), "a valid checkpoint saves");
        check(store.exists("gate-probe"), "and is visible to exists()");

        soma::KvCache dst;
        (void)dst.open(model.arch, 64);
        check(store.load("gate-probe", dst).ok(), "and loads back into a matching engine");

        // The gate is only meaningful if the value it compares is real.
        //
        // This check exists because the first version of this test PASSED while
        // the checkpoint's arch_hash was the empty string: nothing on the fp32
        // load path computed it, so every comparison was "" against "" and
        // accepted anything. The refusal below would still have gone green.
        check(model.arch.arch_hash.size() >= 32, "arch_hash is actually populated",
              model.arch.arch_hash.empty() ? "EMPTY — the gate would be vacuous"
                                           : model.arch.arch_hash.substr(0, 16) + "...");

        // arch_hash mismatch: same file, a store opened against a different model.
        soma::ArchIr other = model.arch;
        other.arch_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        soma::KvCheckpointStore other_store;
        if (other_store.open(dir.string(), other).ok()) {
            soma::KvCache d2;
            (void)d2.open(other, 64);
            const auto st = other_store.load("gate-probe", d2);
            check(st.code() == soma::StatusCode::ArchMismatch,
                  "arch_hash mismatch is refused with ArchMismatch",
                  st.ok() ? "LOADED ANYWAY" : st.message());
        }

        // Version mismatch, forged by hand: the version field is at a known
        // offset right after the 8-byte magic.
        const auto p = dir / "gate-probe.somakv";
        {
            std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
            f.seekp(8);
            const std::uint32_t bogus = 999;
            f.write(reinterpret_cast<const char*>(&bogus), 4);
        }
        soma::KvCache d3;
        (void)d3.open(model.arch, 64);
        const auto vst = store.load("gate-probe", d3);
        check(vst.code() == soma::StatusCode::VersionMismatch,
              "version mismatch is refused with VersionMismatch",
              vst.ok() ? "LOADED ANYWAY" : vst.message());

        // And sweep must reclaim it, since it can never be loaded again.
        std::uint32_t removed = 0;
        (void)store.sweep(0, removed);
        check(removed >= 1, "sweep removes checkpoints that can never load",
              std::to_string(removed) + " removed");
        check(!store.exists("gate-probe"), "the unloadable file is gone");
    }

    // ── 4. the point of preemption: the memory is actually released ──────────
    std::cout << "\n4. preemption releases the KV buffer\n";
    {
        soma::Scheduler sched;
        (void)sched.open_f32(model, nullptr, cfg, &store);
        soma::SeqRequest req;
        req.prompt = prompt;
        req.max_tokens = kMaxTokens;
        soma::SeqId id = 0;
        soma::AdmitRejection why{};
        (void)sched.admit(std::move(req), id, why);
        for (int i = 0; i < 5; ++i) (void)sched.step();

        const auto before = sched.stats().preemptions;
        check(sched.preempt(id).ok(), "preempt succeeds mid-generation");
        check(sched.stats().preemptions == before + 1, "preemption is counted");
        // A sequence whose KV is on disk must not be scheduled: it has no cache
        // to attend over. Stepping here would read a released buffer.
        check(sched.idle(), "a preempted sequence is not schedulable");
        check(sched.resume(id).ok(), "resume restores it");
        check(!sched.idle(), "and it is schedulable again");
    }

    fs::remove_all(dir, ec);
    std::cout << "\n" << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES")
              << "\n";
    return g_failures == 0 ? 0 : 1;
}
