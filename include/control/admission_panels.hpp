#pragma once

// Mantic-Mind — the Admissions tab.
//
// Admission is the longest and most expensive operation in the system: hours,
// hundreds of gigabytes, a staged pipeline with cancel and rejoin. It had a full
// progress protocol and NO operator surface — `grep -rn admission` over every
// TUI file returned zero, and the CLI's `models` group was `list` alone. The one
// operation that most needs watching was the only one you could not watch
// (roadmap D47).
//
// Same rule as soma_dashboard.hpp and engine_panels.hpp, for the same reason:
// every value here arrives over `/v1/*`. This panel does not touch
// ControlModelRegistry even though it renders its operations — a TUI that
// reached into it would be a client with privileges no other client has, and
// P1's claim that the API is the single control plane would stop being testable.
// tools/ci/check_ui_api.py guards this header and its .cpp.
//
// ── Rejoin is free, and that is the point ─────────────────────────────────────
//
// The tab polls `GET /v1/models/admissions`, which lists operations whether or
// not anyone is streaming them. So closing the TUI mid-conversion and reopening
// it shows the admission still running, with no reattach protocol on this side.
// The SSE stream is used for exactly one thing — learning the operation id when
// starting one — because `POST /v1/models/admit` answers only as a stream.
// Disconnecting does not cancel: control logs "client disconnected; conversion
// continues" and the worker is detached precisely so it survives.

#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mm {

/// One admission, as `GET /v1/models/admissions` reports it.
///
/// A view of the wire, named as the wire names it. Anything the panel wants that
/// is not here is a missing route rather than a reason to reach around the API.
struct AdmissionView {
    std::string operation_id;
    std::string source;
    std::string stage;
    std::string detail;
    int step = 0;
    int total_steps = 0;
    double fraction = 0.0;
    std::int64_t bytes_done = 0;
    std::int64_t bytes_total = 0;
    bool cancelable = true;
    bool done = false;
    bool canceled = false;
    std::string error;
    std::int64_t model_id = 0;
    std::int64_t started_at_ms = 0;
    std::int64_t finished_at_ms = 0;

    /// Running means "not finished". Distinct from `cancelable`, which the
    /// pipeline lowers for stages that cannot be interrupted safely.
    bool running() const noexcept { return !done; }
};

/// One admitted model, from `GET /v1/models`. The OUTCOME of an admission —
/// shown beside the operations because "did it work" is the question an operator
/// asks the moment one finishes.
struct AdmittedView {
    std::int64_t id = 0;
    std::string name;
    std::string verdict;
    std::string attention_family;
    std::string arch_hash;
    std::int64_t bytes_per_token = 0;
};

struct AdmissionSnapshot {
    bool reachable = false;
    std::string error;
    std::vector<AdmissionView> operations; ///< newest last, as the API orders them
    std::vector<AdmittedView> models;
};

/// Polls `/v1/models/admissions` and `/v1/models` on its own thread.
class AdmissionDashboard {
public:
    AdmissionDashboard(std::string base_url, std::string api_token);
    ~AdmissionDashboard();

    AdmissionDashboard(const AdmissionDashboard&) = delete;
    AdmissionDashboard& operator=(const AdmissionDashboard&) = delete;

    void start(int interval_ms);
    void stop();
    void refresh_now();

    AdmissionSnapshot snapshot() const;

    /// Start an admission. Returns the operation id, or "" with `out_error`.
    ///
    /// Empty quant fields mean "use the deployment default" — the request omits
    /// them rather than sending a blank, because the server reads a present-but
    /// -empty field as an explicit choice of nothing.
    ///
    /// BLOCKING only until the first progress frame arrives, not for the hours
    /// the admission runs: it reads the operation id off the stream and hangs
    /// up. Callers run it off the UI thread anyway — see `admission_tab`.
    std::string admit(const std::string& source,
                      const std::string& expert_gate,
                      const std::string& expert_down,
                      int group,
                      std::string& out_error);

    bool cancel(const std::string& operation_id, std::string& out_error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ── Pure renderers ────────────────────────────────────────────────────────────

/// The operation list: source, stage, and how far along.
///
/// `selected` indexes `snap.operations`; out of range renders without a
/// highlight rather than clamping, so a stale selection after an operation ages
/// out is visible instead of silently pointing at its neighbour.
ftxui::Element render_admission_list(const AdmissionSnapshot& snap, int selected);

/// The staged detail for the selected operation: every stage the pipeline will
/// run, which one it is on, and the gauge.
///
/// Renders the WHOLE ladder rather than only the current stage, because "convert
/// 3 of 7" answers a different question from "convert" — the first tells an
/// operator whether to wait.
ftxui::Element render_admission_detail(const AdmissionSnapshot& snap, int selected);

/// What admission has produced: the registry, with verdicts.
ftxui::Element render_admitted_models(const AdmissionSnapshot& snap);

/// The whole tab. `selected` indexes the operation list.
ftxui::Component admission_tab(AdmissionDashboard& dashboard, int& selected);

} // namespace mm
