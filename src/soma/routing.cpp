// Soma — engine selection. See routing.hpp for the policy and its one exception.

#include "soma/routing.hpp"

namespace soma {

const char* to_string(BackendChoice choice) noexcept {
    switch (choice) {
    case BackendChoice::Fallback:
        return "fallback";
    case BackendChoice::Soma:
        return "soma";
    }
    return "unknown";
}

const char* to_string(BackendOverride override) noexcept {
    switch (override) {
    case BackendOverride::Auto:
        return "auto";
    case BackendOverride::Soma:
        return "soma";
    case BackendOverride::Fallback:
        return "fallback";
    }
    return "unknown";
}

const char* to_string(BackendReason reason) noexcept {
    switch (reason) {
    case BackendReason::Verdict:
        return "verdict";
    case BackendReason::NoRecord:
        return "no_admission_record";
    case BackendReason::StaleRecord:
        return "stale_admission_record";
    case BackendReason::OperatorOverride:
        return "operator_override";
    case BackendReason::OverrideRefused:
        return "override_refused_conformance";
    }
    return "unknown";
}

std::string BackendDecision::explain() const {
    std::string s = std::string(to_string(choice)) + " (" + to_string(reason);
    if (record_used) {
        s += ", verdict=";
        s += to_string(considered);
    }
    return s + ")";
}

BackendDecision select_backend(const AgentBackendConfig& cfg,
                               const AdmissionRecord& record) noexcept {
    BackendDecision d;

    // A record for DIFFERENT WEIGHTS is not a record for these. Requantizing
    // changes the verdict — that is the whole reason the verdict is a property
    // of (model, quantization, host) rather than of the model — so a stale hash
    // is treated as absence rather than as a usable answer.
    const bool usable = record.present && (cfg.arch_hash.empty() || record.arch_hash.empty() ||
                                           cfg.arch_hash == record.arch_hash);
    const bool stale = record.present && !usable;

    d.considered = usable ? record.verdict : Verdict::Reject;
    d.record_used = usable;
    d.reason = usable ? BackendReason::Verdict
                      : (stale ? BackendReason::StaleRecord : BackendReason::NoRecord);

    // Explicit fallback always wins: it can only ever be more conservative.
    if (cfg.override == BackendOverride::Fallback) {
        d.choice = BackendChoice::Fallback;
        d.reason = BackendReason::OperatorOverride;
        return d;
    }

    if (cfg.override == BackendOverride::Soma) {
        // The one refusal. See routing.hpp: `reject` is a CONFORMANCE failure,
        // not an economics one, and no config flag should be able to turn
        // "produces wrong tokens" into "serve it anyway".
        //
        // Note this fires for a stale or missing record too, since `considered`
        // is Reject in both cases: forcing Soma onto weights nothing has
        // admitted is the same bet with less evidence behind it.
        if (d.considered == Verdict::Reject) {
            d.choice = BackendChoice::Fallback;
            d.reason = BackendReason::OverrideRefused;
            return d;
        }
        d.choice = BackendChoice::Soma;
        d.reason = BackendReason::OperatorOverride;
        return d;
    }

    // Auto: the verdict decides, and only stream/hybrid select Soma.
    d.choice = (usable && verdict_selects_soma(record.verdict)) ? BackendChoice::Soma
                                                                : BackendChoice::Fallback;
    return d;
}

} // namespace soma
