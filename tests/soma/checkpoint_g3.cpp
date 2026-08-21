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
    for (const auto t : v)
        s += " " + std::to_string(t);
    return s;
}

} // namespace

int main(int argc, char** argv) {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::path("tests/fixtures");
    const std::string name = (argc > 2) ? argv[2] : "Qwen3-30B-A3B";

    soma::F32Model model;
    if (auto st = soma::load_f32_model((root / "tiny" / name).string(), model); !st.ok()) {
        std::cerr << "load failed: " << st.message() << "\n";
        return 2;
    }

    // The CTest registrations deliberately cover all cache shapes: GQA's
    // equal K/V planes, plain MLA's absent V plane, and MLA+DSA's narrower V
    // plane, plus V4's backend-owned opaque live-state format. Without running
    // checkpointing on each, the save/load loops could keep assuming an older
    // shape while decode stayed green.
    {
        soma::KvCache probe;
        if (auto st = probe.open(model.arch, 8); !st.ok()) {
            std::cerr << "cache open failed: " << st.message() << "\n";
            return 2;
        }
        if (probe.is_opaque()) {
            check(probe.opaque_size() > 0 && probe.k_hkv() == 0 && probe.v_hkv() == 0,
                  "backend-owned cache uses only its opaque arena",
                  std::to_string(probe.opaque_size()) + " bytes");
        } else if (model.arch.attention.family == soma::AttentionFamily::Mla) {
            check(probe.v_hkv() == 0 && probe.v_at(0, 0) == nullptr,
                  "plain MLA allocates no V plane",
                  std::to_string(probe.v_hkv()) + " floats/position");
        } else if (model.arch.attention.family == soma::AttentionFamily::MlaDsa) {
            check(probe.v_hkv() == model.arch.attention.dsa.index_head_dim &&
                      probe.v_hkv() < probe.k_hkv(),
                  "MLA+DSA gives the index key its own narrower V plane",
                  std::to_string(probe.k_hkv()) + "/" + std::to_string(probe.v_hkv()) +
                      " K/V floats");
        } else {
            check(probe.k_hkv() == probe.v_hkv(),
                  "GQA keeps equal K and V planes",
                  std::to_string(probe.k_hkv()) + " floats each");
        }
    }

    const auto dir = fs::temp_directory_path() / ("soma_ckpt_g3_" + name);
    std::error_code ec;
    fs::remove_all(dir, ec);

    soma::KvCheckpointStore store;
    if (auto st = store.open(dir.string(), model.arch); !st.ok()) {
        std::cerr << "store open failed: " << st.message() << "\n";
        return 2;
    }

    // A speculative backend owns additional per-sequence state.  Its presence
    // selects v4 without changing the byte layout of ordinary v3 checkpoints.
    {
        soma::KvCache written, restored;
        check(written.open(model.arch, 8).ok() && restored.open(model.arch, 8).ok(),
              "auxiliary checkpoint cache opens");
        soma::SeqPersistState state;
        state.auxiliary = {std::byte{0x11}, std::byte{0x00}, std::byte{0x7f}, std::byte{0xa5}};
        check(store.save("auxiliary-v4", written, state).ok(),
              "backend state selects checkpoint v4");
        soma::KvCheckpointHeader header;
        check(store.stat("auxiliary-v4", header).ok() &&
                  header.version == soma::kKvCheckpointVersionSpeculative &&
                  header.auxiliary_bytes == state.auxiliary.size(),
              "v4 header records exact auxiliary extent");
        soma::SeqPersistState loaded;
        check(store.load("auxiliary-v4", restored, loaded).ok() &&
                  loaded.auxiliary == state.auxiliary,
              "v4 auxiliary state round-trips byte-exactly");
        (void)store.remove("auxiliary-v4");
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
        sched.set_token_callback([&](soma::SeqId, soma::TokenId t, bool) { out.push_back(t); });

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
    check(baseline.size() == kMaxTokens,
          "baseline generated the full continuation",
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
        check(same,
              "continuation identical after preempt at step " + std::to_string(at),
              same ? "byte-for-byte" : "DIVERGED");
    }

    // ── 2b. CROSS-PROCESS resume ─────────────────────────────────────────────
    //
    // The section above cannot see the bug this guards. preempt() and resume()
    // run against the same Scheduler, so the sampler object survives in memory
    // and the continuation matches whether or not the RNG was ever persisted.
    //
    // A fresh Scheduler is the honest test: it is what a restarted engine has.
    // Everything the resumed sequence knows must have come off disk.
    std::cout << "\n2b. resume into a FRESH scheduler (what a restart has)\n";
    {
        constexpr std::uint32_t kTail = 4;

        // One uninterrupted run, checkpointed partway, kept to completion. Its
        // tail is what every resume below is measured against.
        soma::Scheduler first;
        (void)first.open_f32(model, nullptr, cfg, &store);
        std::vector<soma::TokenId> full;
        first.set_token_callback([&](soma::SeqId, soma::TokenId t, bool) { full.push_back(t); });

        soma::SeqRequest req;
        req.prompt = prompt;
        req.max_tokens = kMaxTokens;
        soma::SeqId id = 0;
        soma::AdmitRejection why{};
        (void)first.admit(std::move(req), id, why);
        for (int i = 0; i < 6 && !first.idle(); ++i)
            (void)first.step();

        // Frozen HERE: the callback keeps appending as the run finishes.
        const std::size_t k = full.size();
        check(first.checkpoint(id, "cross-process").ok(),
              "a mid-generation checkpoint saves",
              std::to_string(k) + " tokens emitted so far");
        while (!first.idle()) {
            if (!first.step().ok()) break;
        }
        check(full.size() >= k + kTail,
              "and the original run continues past it",
              std::to_string(full.size()) + " tokens total");

        // What the original said NEXT. The checkpointed cache holds prompt plus
        // full[0..k-2] — the k'th token was sampled but never fed back — so a
        // resume's first output is full[k].
        std::vector<soma::TokenId> expected(full.begin() + static_cast<std::ptrdiff_t>(k),
                                            full.begin() + static_cast<std::ptrdiff_t>(k + kTail));

        // Two spellings of the same resume, which must agree:
        //
        //   "cached"    prompt == the cached tokens exactly. Nothing to prefill,
        //               so the next input can only come from `emitted`.
        //   "extended"  prompt carries the last emitted token too, so it arrives
        //               as an ordinary prefill row.
        //
        // The first is the one that exposes an unrestored next_token; the second
        // is what a client that appends its own output sends. A resume that only
        // handles the second looks correct until someone checkpoints at a turn
        // boundary.
        const auto resume_with = [&](std::size_t take) {
            soma::Scheduler s;
            (void)s.open_f32(model, nullptr, cfg, &store);
            std::vector<soma::TokenId> out;
            s.set_token_callback([&](soma::SeqId, soma::TokenId t, bool) { out.push_back(t); });

            soma::SeqRequest r;
            r.prompt = prompt;
            r.prompt.insert(
                r.prompt.end(), full.begin(), full.begin() + static_cast<std::ptrdiff_t>(take));
            r.max_tokens = kTail;
            r.resume_key = "cross-process";
            soma::SeqId rid = 0;
            soma::AdmitRejection rwhy{};
            if (!s.admit(std::move(r), rid, rwhy).ok()) return out;
            while (!s.idle()) {
                if (!s.step().ok()) break;
            }
            return out;
        };

        for (const auto& form : {std::make_pair(k - 1, "cached prefix exactly"),
                                 std::make_pair(k, "prompt extends past the cache")}) {
            const auto got = resume_with(form.first);
            const bool same = (got == expected);
            check(same,
                  std::string("fresh scheduler continues the run — ") + form.second,
                  same ? "byte-for-byte"
                       : ("got" + tok_str(got) + "  expected" + tok_str(expected)));
        }

        // The control. Same prompt, no checkpoint — so the cache is rebuilt by
        // prefill and is IDENTICAL, but the sampler starts at seed position 0.
        // If this matched, the checks above would prove nothing: they would pass
        // for any sampler whose stream did not depend on history, and the whole
        // point of persisting rng_state is that this one's does.
        soma::Scheduler cold;
        (void)cold.open_f32(model, nullptr, cfg, nullptr);
        std::vector<soma::TokenId> cold_out;
        cold.set_token_callback([&](soma::SeqId, soma::TokenId t, bool) { cold_out.push_back(t); });
        soma::SeqRequest fresh_req;
        fresh_req.prompt = prompt;
        fresh_req.prompt.insert(
            fresh_req.prompt.end(), full.begin(), full.begin() + static_cast<std::ptrdiff_t>(k));
        fresh_req.max_tokens = kTail;
        soma::SeqId cid = 0;
        (void)cold.admit(std::move(fresh_req), cid, why);
        while (!cold.idle()) {
            if (!cold.step().ok()) break;
        }
        std::cout << "   uninterrupted tail:" << tok_str(expected) << "\n";
        std::cout << "   cold, no rng_state:" << tok_str(cold_out) << "\n";
        check(cold_out != expected,
              "a cold start on the same prompt does NOT match",
              cold_out == expected ? "IDENTICAL — this test cannot detect a lost RNG"
                                   : "diverges, as it must");
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
        // v2 carries the ids occupying the cached positions, so a restore can be
        // checked against the prompt it is about to be attached to.
        soma::SeqPersistState probe;
        probe.tokens = {3, 1, 4, 1, 5, 9, 2, 6};
        // v3 also carries the sampler's stream position and the penalty history.
        // Without them a resumed sequence draws from a fresh stream with no
        // memory of what it has said: correct tokens, wrong continuation.
        probe.emitted = {9, 2, 6, 5};
        probe.rng_state = 0x0123456789ABCDEFull;
        check(store.save("gate-probe", kv, probe).ok(), "a valid checkpoint saves");
        check(store.exists("gate-probe"), "and is visible to exists()");

        soma::KvCache dst;
        soma::SeqPersistState read_back;
        (void)dst.open(model.arch, 64);
        check(store.load("gate-probe", dst, read_back).ok(),
              "and loads back into a matching engine");
        check(read_back.tokens == probe.tokens,
              "the token ids round-trip with the cache",
              std::to_string(read_back.tokens.size()) + " ids");
        check(read_back.emitted == probe.emitted,
              "the penalty history round-trips",
              std::to_string(read_back.emitted.size()) + " emitted");
        check(read_back.rng_state == probe.rng_state, "and so does the RNG stream position");

        // An EMPTY emitted list is distinct from an absent one: a sequence that
        // has produced nothing yet must not come back with someone else's history.
        soma::SeqPersistState fresh;
        fresh.tokens = probe.tokens;
        fresh.rng_state = 7;
        check(store.save("fresh", kv, fresh).ok(), "a sequence with no output saves");
        soma::KvCache d0;
        soma::SeqPersistState fresh_back;
        (void)d0.open(model.arch, 64);
        check(store.load("fresh", d0, fresh_back).ok() && fresh_back.emitted.empty() &&
                  fresh_back.rng_state == 7,
              "and comes back with an empty history, not the previous one's");

        // A token list that does not line up with the cached positions is refused
        // rather than padded: a prefix check against a misaligned list reads as a
        // guarantee while proving nothing.
        soma::KvCache short_kv;
        (void)short_kv.open(model.arch, 64);
        (void)short_kv.set_length(8);
        soma::SeqPersistState misaligned;
        misaligned.tokens = {1, 2, 3};
        check(!store.save("bad-count", short_kv, misaligned).ok(),
              "a token count that disagrees with the cache is refused");

        // The gate is only meaningful if the value it compares is real.
        //
        // This check exists because the first version of this test PASSED while
        // the checkpoint's arch_hash was the empty string: nothing on the fp32
        // load path computed it, so every comparison was "" against "" and
        // accepted anything. The refusal below would still have gone green.
        check(model.arch.arch_hash.size() >= 32,
              "arch_hash is actually populated",
              model.arch.arch_hash.empty() ? "EMPTY — the gate would be vacuous"
                                           : model.arch.arch_hash.substr(0, 16) + "...");

        // arch_hash mismatch: same file, a store opened against a different model.
        soma::ArchIr other = model.arch;
        other.arch_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        soma::KvCheckpointStore other_store;
        if (other_store.open(dir.string(), other).ok()) {
            soma::KvCache d2;
            soma::SeqPersistState ignored;
            (void)d2.open(other, 64);
            const auto st = other_store.load("gate-probe", d2, ignored);
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
        soma::SeqPersistState ignored3;
        (void)d3.open(model.arch, 64);
        const auto vst = store.load("gate-probe", d3, ignored3);
        check(vst.code() == soma::StatusCode::VersionMismatch,
              "version mismatch is refused with VersionMismatch",
              vst.ok() ? "LOADED ANYWAY" : vst.message());

        // And sweep must reclaim it, since it can never be loaded again.
        std::uint32_t removed = 0;
        (void)store.sweep(0, removed);
        check(removed >= 1,
              "sweep removes checkpoints that can never load",
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
        for (int i = 0; i < 5; ++i)
            (void)sched.step();

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
    std::cout << "\n"
              << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES") << "\n";
    return g_failures == 0 ? 0 : 1;
}
