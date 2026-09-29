#pragma once

// Soma — which engine serves a served model.
//
// PURE, given (served-model config, admission record). No node, no placement,
// no I/O. That is deliberate: the decision can be ANSWERED without causing a
// placement, and it is the one piece of G5 that can be tested exhaustively
// rather than observed.
//
// `GET /v1/served-models/{id}` answers with it, and purity is what makes that
// safe: a read route must not cause a placement. So do `GET /v1/placements`
// and `PUT /v1/served-models/{id}/backend`.
//
// The per-model route did NOT carry it for as long as this comment once claimed
// it did — the handler (then per-agent) returned config, placement, status and
// node compatibility and no backend field at all. The sentence was true of the
// design and false of the code, in two files at once, because the deleted
// `placement_engine.hpp` made the same promise. Fixed rather than re-worded
// (roadmap D61).
//
// The policy, from docs/architecture.md §9:
//
//   stream / hybrid   -> Soma
//   resident-only     -> fallback   (it fits; streaming buys nothing)
//   reject            -> fallback   (failed conformance stage 1 or 2)
//   no record         -> fallback   (absence is not evidence of admissibility)
//
// Fallback is the safe default in every ambiguous case, and Soma is
// opt-in-by-verdict rather than opt-out.

#include "soma/types.hpp"

#include <string>

namespace soma {

enum class BackendChoice : std::uint8_t {
    Fallback = 0, ///< default-deny: the value you get when nothing says otherwise
    Soma,
};

/// Persistent, on the SERVED MODEL — not per placement.
///
/// Placement is recomputed on every acquisition, so a per-placement override
/// would evaporate on the next eviction cycle and read as a flapping bug rather
/// than as a setting that stopped applying.
enum class BackendOverride : std::uint8_t {
    Auto = 0,
    Soma,
    Fallback,
};

/// Why. Surfaced on `GET /v1/placements` so an operator can see the CAUSE of a
/// routing decision rather than inferring it from throughput.
enum class BackendReason : std::uint8_t {
    Verdict = 0,      ///< the admission verdict selected this engine
    NoRecord,         ///< nothing admitted this model
    StaleRecord,      ///< a record exists but for different weights
    OperatorOverride, ///< an operator forced it
    OverrideRefused,  ///< an operator asked for Soma and was refused; see below
};

struct AdmissionRecord {
    bool present = false;
    Verdict verdict = Verdict::Reject;
    std::string arch_hash;
};

/// What the caller wants served: the operator's override and the identity of
/// the weights. Named for the request it is rather than for whoever owns the
/// setting — that owner has been an agent and is now a served model, and the
/// engine policy should not have to change when it changes again.
struct BackendRequest {
    BackendOverride override = BackendOverride::Auto;
    /// The hash of the weights that will actually load. Compared against the
    /// record's, because a verdict computed for different weights is not a
    /// verdict for these.
    std::string arch_hash;
};

struct BackendDecision {
    BackendChoice choice = BackendChoice::Fallback;
    BackendReason reason = BackendReason::NoRecord;
    Verdict considered = Verdict::Reject;
    bool record_used = false;

    /// One line, suitable for an API field and a log. Never empty.
    std::string explain() const;
};

/// The decision.
///
/// ── One case where the operator override does NOT win ────────────────────────
///
/// `BackendOverride::Soma` against a `reject` verdict is REFUSED.
///
/// The other verdicts are economics: `resident-only` means streaming buys
/// nothing on this host, and an operator who wants Soma anyway is making a
/// legitimate performance call — G4 relies on exactly that to run DeepSeek
/// through the streaming path. But `reject` means the model FAILED CONFORMANCE
/// stage 1 or 2: Soma's output does not match the reference. Honouring the
/// override there would serve knowingly-wrong tokens because someone set a
/// config flag, and the failure would present as model quality rather than as
/// configuration.
///
/// The remedy for `reject` is to re-admit with a different quantization map,
/// which is what the failure is telling you. That is a different action from
/// flipping a routing switch, and the API should not let one masquerade as the
/// other.
BackendDecision select_backend(const BackendRequest& cfg,
                               const AdmissionRecord& record) noexcept;

const char* to_string(BackendChoice choice) noexcept;
const char* to_string(BackendOverride override) noexcept;
const char* to_string(BackendReason reason) noexcept;

} // namespace soma
