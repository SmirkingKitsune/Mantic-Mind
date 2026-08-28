// Soma — G5: verdict routing.
//
// The gate, from the roadmap: "Mixtral-8x7B admits as resident-only and routes
// to the fallback, UNPROMPTED." Unprompted is the operative word — no override,
// no special case, no entry in a list of known-bad models. The verdict function
// looks at Mixtral's economics and says no, and the router honours it.
//
// That is what makes the verdict earn its keep. A verdict that only ever says
// yes is a decoration; the negative fixture is the test.
//
// Usage: routing_g5 [fixtures_root]

#include "soma/plan.hpp"
#include "soma/routing.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    std::cout << "   " << std::left << std::setw(56) << what << (ok ? "OK" : "FAIL");
    if (!detail.empty()) std::cout << "   " << detail;
    std::cout << "\n";
    if (!ok) ++g_failures;
}

soma::AdmissionRecord rec(soma::Verdict v, const std::string& hash = "H") {
    soma::AdmissionRecord r;
    r.present = true;
    r.verdict = v;
    r.arch_hash = hash;
    return r;
}

soma::AgentBackendConfig cfg(soma::BackendOverride o = soma::BackendOverride::Auto,
                             const std::string& hash = "H") {
    soma::AgentBackendConfig c;
    c.override = o;
    c.arch_hash = hash;
    return c;
}

} // namespace

int main() {
    using soma::BackendChoice;
    using soma::BackendOverride;
    using soma::BackendReason;
    using soma::Verdict;

    // ── 1. the policy table, exhaustively ────────────────────────────────────
    //
    // Four verdicts x three overrides x present/absent is small enough to state
    // completely, and a routing policy stated completely is one nobody has to
    // reconstruct from prose later.
    std::cout << "1. auto: the verdict decides\n";
    {
        struct Case {
            Verdict v;
            BackendChoice want;
        };

        const Case cases[] = {
            {Verdict::Stream, BackendChoice::Soma},
            {Verdict::Hybrid, BackendChoice::Soma},
            {Verdict::ResidentOnly, BackendChoice::Fallback},
            {Verdict::Reject, BackendChoice::Fallback},
        };
        for (const auto& c : cases) {
            const auto d = soma::select_backend(cfg(), rec(c.v));
            check(d.choice == c.want && d.reason == BackendReason::Verdict,
                  std::string("verdict ") + soma::to_string(c.v) + " -> " + soma::to_string(c.want),
                  d.explain());
        }

        // explain() is USER-VISIBLE — it is the `reason` on GET /v1/agents/{id}
        // and it is quoted verbatim inside the 422 that refuses an image to a
        // text-only engine. Until now nothing asserted its text, which is how it
        // shipped reading `soma (verdict, verdict=hybrid)`: the reason name and
        // the field name are the same word (D15).
        //
        // Pinned as a whole string rather than by substring. A contains-check
        // would have passed the stutter too.
        const auto v = soma::select_backend(cfg(), rec(Verdict::Hybrid));
        check(v.explain() == "soma (verdict=hybrid)",
              "a verdict decision names the verdict once",
              v.explain());

        // The other reasons are a DIFFERENT fact from the verdict, so they keep
        // both halves — dropping either would lose what was asked for or why it
        // was refused.
        const auto refused = soma::select_backend(cfg(BackendOverride::Soma), rec(Verdict::Reject));
        check(refused.explain() == "fallback (override_refused_conformance, verdict=reject)",
              "a refused override still says what it refused and why",
              refused.explain());

        // "fallback", not "llama-cpp": explain() names the ROLE, and
        // AgentScheduler::resolve_backend maps that role to an engine id one
        // layer up. Asserting the engine id here would bake a mapping this
        // function does not own.
        soma::AdmissionRecord absent;
        check(soma::select_backend(cfg(), absent).explain() == "fallback (no_admission_record)",
              "and with no record there is no verdict to name",
              soma::select_backend(cfg(), absent).explain());
    }

    std::cout << "\n2. absence and staleness are not admissibility\n";
    {
        soma::AdmissionRecord none;
        const auto d = soma::select_backend(cfg(), none);
        check(d.choice == BackendChoice::Fallback && d.reason == BackendReason::NoRecord,
              "no record -> fallback",
              d.explain());

        // A verdict computed for OTHER weights is not a verdict for these:
        // requantizing changes the economics, which is why the verdict is a
        // property of (model, quantization, host) rather than of the model.
        const auto s =
            soma::select_backend(cfg(BackendOverride::Auto, "H2"), rec(Verdict::Stream, "H1"));
        check(s.choice == BackendChoice::Fallback && s.reason == BackendReason::StaleRecord,
              "record for different weights -> fallback",
              s.explain());
    }

    std::cout << "\n3. operator override\n";
    {
        const auto f = soma::select_backend(cfg(BackendOverride::Fallback), rec(Verdict::Stream));
        check(f.choice == BackendChoice::Fallback && f.reason == BackendReason::OperatorOverride,
              "fallback override beats a stream verdict",
              f.explain());

        // The G4 case: force Soma onto a model that merely FITS. Legitimate —
        // it is a performance call, not a correctness one.
        const auto s = soma::select_backend(cfg(BackendOverride::Soma), rec(Verdict::ResidentOnly));
        check(s.choice == BackendChoice::Soma && s.reason == BackendReason::OperatorOverride,
              "soma override beats resident-only",
              s.explain());

        // The refusal. `reject` means conformance FAILED — Soma's output does
        // not match the reference — so honouring the override would serve
        // knowingly-wrong tokens because of a config flag.
        const auto r = soma::select_backend(cfg(BackendOverride::Soma), rec(Verdict::Reject));
        check(r.choice == BackendChoice::Fallback && r.reason == BackendReason::OverrideRefused,
              "soma override is REFUSED against a reject verdict",
              r.explain());

        soma::AdmissionRecord none;
        const auto n = soma::select_backend(cfg(BackendOverride::Soma), none);
        check(n.choice == BackendChoice::Fallback && n.reason == BackendReason::OverrideRefused,
              "soma override is refused with no record at all",
              n.explain());
    }

    // ── 4. THE GATE ──────────────────────────────────────────────────────────
    //
    // Driven from the real planner, not a hand-written verdict. Asserting
    // `select_backend(ResidentOnly) == Fallback` proves the router; it proves
    // nothing about whether Mixtral actually EARNS that verdict, which is the
    // half the gate is about.
    std::cout << "\n4. THE GATE: Mixtral routes to the fallback, unprompted\n";
    {
        // Mixtral-8x7B: 8 experts, top-2, intermediate 14336, d_model 4096,
        // 32 layers. At q4 an expert is ~88 MB and the routed set is ~22 GB —
        // but only 8 experts per layer, so 25% of them fire on every token.
        // Streaming that buys nothing a resident copy would not.
        soma::ArchIr mixtral{};
        // Both fixtures declare their attention family, which they did not need
        // to before compute_plan began asking whether a backend exists. An
        // ArchIr that does not say what attention it uses cannot be served by
        // anything, so it now plans as reject — correctly. These are GQA models
        // in reality and the fixture was simply incomplete; leaving the field at
        // Unknown would have been testing the verdict function against a
        // description of no real model.
        mixtral.attention.family = soma::AttentionFamily::Gqa;
        mixtral.topology.n_layers = 32;
        mixtral.topology.d_model = 4096;
        mixtral.topology.vocab_size = 32000;
        mixtral.topology.layer_kinds.assign(32, soma::LayerKind::Moe);
        mixtral.router.n_experts = 8;
        mixtral.router.top_k = 2;
        mixtral.ffn.expert_intermediate = 14336;
        mixtral.quantization.expert_gate = {soma::DType::Q4_G, 128};
        mixtral.quantization.expert_up = {soma::DType::Q4_G, 128};
        mixtral.quantization.expert_down = {soma::DType::Q4_G, 128};

        soma::ArchIr qwen{};
        qwen.attention.family = soma::AttentionFamily::Gqa;
        qwen.topology.n_layers = 48;
        qwen.topology.d_model = 2048;
        qwen.topology.vocab_size = 151936;
        qwen.topology.layer_kinds.assign(48, soma::LayerKind::Moe);
        qwen.router.n_experts = 128;
        qwen.router.top_k = 8;
        qwen.ffn.expert_intermediate = 768;
        qwen.quantization.expert_gate = {soma::DType::Q4_G, 128};
        qwen.quantization.expert_up = {soma::DType::Q4_G, 128};
        qwen.quantization.expert_down = {soma::DType::Q6_G, 128};

        // TWO hosts, because one cannot express the gate.
        //
        // The first draft used a single roomy host and asserted Mixtral
        // resident-only / Qwen3 streaming. That is unsatisfiable: Qwen3's routed
        // set (17 GiB) is SMALLER than Mixtral's (23 GiB), so every host where
        // Mixtral fits is one where Qwen3 fits too. The contradiction was in the
        // test, not the planner — and it is worth stating, because "big model
        // streams, small model is resident" is the intuition the verdict exists
        // to correct.
        //
        // What actually separates them is the ACTIVE FRACTION. Mixtral fires 2
        // of 8 experts per token (25%); Qwen3 fires 8 of 128 (6.25%). Streaming
        // reads what fires, so Mixtral moves ~5.5x the bytes per token despite
        // being only 1.35x larger.
        soma::HostBudget roomy{};
        roomy.ram_total_bytes = 64ull << 30;
        roomy.ram_free_bytes = 48ull << 30;
        roomy.disk_bandwidth = 1230ull * 1000 * 1000;

        soma::HostBudget tight{};
        tight.ram_total_bytes = 16ull << 30;
        tight.ram_free_bytes = 8ull << 30;
        tight.disk_bandwidth = 1230ull * 1000 * 1000;

        soma::PlanDocument mp{}, qp{}, mt{}, qt{};
        const bool ok = soma::compute_plan(mixtral, roomy, mp).ok() &&
                        soma::compute_plan(qwen, roomy, qp).ok() &&
                        soma::compute_plan(mixtral, tight, mt).ok() &&
                        soma::compute_plan(qwen, tight, qt).ok();
        check(ok, "all four plans computed");

        std::cout << "\n   " << std::left << std::setw(10) << "model" << std::setw(10) << "routed"
                  << std::setw(13) << "MiB/token" << std::setw(18) << "verdict @48GiB"
                  << "verdict @8GiB\n";
        std::cout << "   " << std::setw(10) << "mixtral" << std::setw(10)
                  << std::to_string(mp.total_routed_bytes >> 30) + " GiB" << std::setw(13)
                  << (mp.bytes_per_token >> 20) << std::setw(18) << soma::to_string(mp.verdict)
                  << soma::to_string(mt.verdict) << "\n";
        std::cout << "   " << std::setw(10) << "qwen3" << std::setw(10)
                  << std::to_string(qp.total_routed_bytes >> 30) + " GiB" << std::setw(13)
                  << (qp.bytes_per_token >> 20) << std::setw(18) << soma::to_string(qp.verdict)
                  << soma::to_string(qt.verdict) << "\n\n";

        // THE GATE. On a host with room, Mixtral fits — so streaming buys
        // nothing and the verdict says so without being told.
        check(mp.verdict == Verdict::ResidentOnly,
              "Mixtral admits as resident-only on a roomy host",
              soma::to_string(mp.verdict));
        const auto md = soma::select_backend(cfg(), rec(mp.verdict));
        check(md.choice == BackendChoice::Fallback,
              "-> and routes to the FALLBACK, unprompted",
              md.explain());

        // The discrimination, on the host where NEITHER can be resident. Both
        // must stream or neither can run, and only one of them is worth
        // streaming: same host, same policy, different answer.
        check(mt.verdict != qt.verdict,
              "on a tight host the two models get DIFFERENT verdicts",
              std::string("mixtral=") + soma::to_string(mt.verdict) +
                  " qwen3=" + soma::to_string(qt.verdict));

        const auto mtd = soma::select_backend(cfg(), rec(mt.verdict));
        const auto qtd = soma::select_backend(cfg(), rec(qt.verdict));
        check(mtd.choice != qtd.choice,
              "-> and therefore different backends",
              "mixtral=" + std::string(soma::to_string(mtd.choice)) +
                  " qwen3=" + soma::to_string(qtd.choice));
        check(qtd.choice == BackendChoice::Soma,
              "-> Qwen3 is the one that goes to Soma",
              qtd.explain());

        // The quantity the whole verdict rests on, asserted rather than assumed:
        // active fraction, not model size, is what makes streaming viable.
        check(mp.bytes_per_token > qp.bytes_per_token * 3,
              "Mixtral moves far more bytes/token despite being similar in size",
              std::to_string(mp.bytes_per_token / (qp.bytes_per_token ? qp.bytes_per_token : 1)) +
                  "x");
    }

    std::cout << "\n"
              << (g_failures == 0 ? "OK" : std::to_string(g_failures) + " FAILURES") << "\n";
    return g_failures == 0 ? 0 : 1;
}
