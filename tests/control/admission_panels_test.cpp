// The Admissions tab's renderers, asserted as TEXT.
//
// The states worth testing are the ones nobody opens the app in: control
// unreachable, nothing admitted yet, a conversion mid-flight, a failure, and a
// cancel. Each of those is a different answer to "what is happening", and they
// are exactly the ones a human clicking through a healthy system never sees.
//
// Rendering to an off-screen buffer is what makes them assertable at all — a
// panel whose only test is "someone looked at it" gets the degraded cases wrong,
// because those are the cases nobody looks at.

#include "control/admission_panels.hpp"

#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>

#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  ok    " : "  FAIL  ") << what << "\n";
    if (!ok) ++failures;
}

/// Render one element to text at a fixed size.
std::string draw(const ftxui::Element& e, int w = 120, int h = 30) {
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(w), ftxui::Dimension::Fixed(h));
    auto page = ftxui::vbox({e});
    ftxui::Render(screen, page);
    return screen.ToString();
}

bool has(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

mm::AdmissionView running_op() {
    mm::AdmissionView op;
    op.operation_id = "op-running";
    op.source = "Qwen/Qwen3-30B-A3B";
    op.stage = "convert";
    op.detail = "shard 3/12";
    op.step = 2;
    op.total_steps = 7;
    op.fraction = 0.28;
    op.started_at_ms = 1000;
    return op;
}

} // namespace

int main() {
    // ── Unreachable ───────────────────────────────────────────────────────────
    // The distinction that matters most: an empty list because control is down
    // must not render like an empty list because nothing is admitting.
    {
        mm::AdmissionSnapshot snap;
        snap.reachable = false;
        snap.error = "GET /v1/models/admissions → HTTP 503";
        const auto out = draw(mm::render_admission_list(snap, 0));
        check(has(out, "not answering"), "unreachable says control is not answering");
        check(has(out, "503"), "unreachable carries the status");
        check(!has(out, "no admissions yet"), "unreachable is NOT reported as 'nothing yet'");
    }

    // ── Reachable and idle ────────────────────────────────────────────────────
    {
        mm::AdmissionSnapshot snap;
        snap.reachable = true;
        const auto out = draw(mm::render_admission_list(snap, 0));
        check(has(out, "no admissions yet"), "idle says so explicitly");
        // The empty state must tell the operator how to leave it. This tab
        // exists because admission had no discoverable entry point at all.
        check(has(out, "a to admit"), "idle names the key that starts one");
    }

    // ── Mid-conversion ────────────────────────────────────────────────────────
    {
        mm::AdmissionSnapshot snap;
        snap.reachable = true;
        snap.operations.push_back(running_op());
        const auto list = draw(mm::render_admission_list(snap, 0));
        check(has(list, "Qwen3-30B-A3B"), "list shows the source");
        check(has(list, "running"), "a live operation reads as running");
        check(has(list, "2/7"), "list shows step of total, not just the stage");

        const auto detail = draw(mm::render_admission_detail(snap, 0));
        check(has(detail, "convert"), "detail names the current stage");
        // The whole ladder, because "convert" alone does not tell an operator
        // whether to wait; "convert, 2 of 7" does. 7 steps is the local-source
        // shape — no fetch — and getting that wrong is what a flat position-
        // indexed list did: every label shifted by one and `finalize` was never
        // drawn.
        check(has(detail, "finalize"), "detail draws stages not yet reached");
        check(has(detail, "shard 3/12"), "detail carries the stage's own message");
    }

    // ── A ladder shape this build does not know ───────────────────────────────
    // Position comes off the wire and stays right; names are the part a client
    // can be wrong about. An unrecognized length draws numbers rather than
    // confidently mislabelling every rung.
    {
        mm::AdmissionSnapshot snap;
        snap.reachable = true;
        auto op = running_op();
        op.total_steps = 9; // no such ladder today
        op.step = 3;
        snap.operations.push_back(op);
        const auto detail = draw(mm::render_admission_detail(snap, 0));
        check(has(detail, "convert"), "the CURRENT stage is still named from the wire");
        check(!has(detail, "finalize"),
              "an unknown ladder does not invent names for unreached stages");
    }

    // ── The container ladder ──────────────────────────────────────────────────
    // What reprofile runs: three stages, starting at profile. A list that always
    // began at `fetch` would label these convert/tokenize/oracle.
    {
        mm::AdmissionSnapshot snap;
        snap.reachable = true;
        auto op = running_op();
        op.total_steps = 3;
        op.step = 1;
        op.stage = "profile";
        snap.operations.push_back(op);
        const auto detail = draw(mm::render_admission_detail(snap, 0));
        check(has(detail, "profile"), "container ladder starts at profile");
        check(has(detail, "conformance"), "container ladder names its second stage");
        check(!has(detail, "convert"), "container ladder does not claim a convert stage");
    }

    // ── Queued ────────────────────────────────────────────────────────────────
    // The state the protocol always had and never used until the concurrency cap
    // landed. "Queued" and "converting" are different answers to "why has
    // nothing happened", and rendering both as running hides the first.
    {
        mm::AdmissionSnapshot snap;
        snap.reachable = true;
        auto op = running_op();
        op.detail = "queued";
        op.step = 0;
        snap.operations.push_back(op);
        const auto out = draw(mm::render_admission_list(snap, 0));
        check(has(out, "queued"), "a queued operation says queued, not running");
    }

    // ── Failed ────────────────────────────────────────────────────────────────
    {
        mm::AdmissionSnapshot snap;
        snap.reachable = true;
        auto op = running_op();
        op.done = true;
        op.error = "conformance stage 2 failed: max|logit| 0.48";
        snap.operations.push_back(op);
        const auto out = draw(mm::render_admission_list(snap, 0));
        check(has(out, "failed"), "a failed operation reads as failed");
        // The reason, in the list — not only after selecting the row. A failure
        // an operator has to click to see is a failure they do not see.
        check(has(out, "conformance stage 2"), "the failure reason is on the list row");
    }

    // ── Canceled ──────────────────────────────────────────────────────────────
    // Distinct from failed: one is a fault, the other is an instruction that was
    // obeyed, and they call for opposite responses.
    {
        mm::AdmissionSnapshot snap;
        snap.reachable = true;
        auto op = running_op();
        op.done = true;
        op.canceled = true;
        snap.operations.push_back(op);
        const auto out = draw(mm::render_admission_list(snap, 0));
        check(has(out, "canceled"), "a canceled operation is not reported as failed");
    }

    // ── Selection out of range ────────────────────────────────────────────────
    // Renders without a highlight rather than clamping, so a stale selection
    // after an operation ages out is visible instead of silently pointing at its
    // neighbour.
    {
        mm::AdmissionSnapshot snap;
        snap.reachable = true;
        snap.operations.push_back(running_op());
        const auto out = draw(mm::render_admission_detail(snap, 5));
        check(has(out, "no operation selected"), "out-of-range selection says so");
    }

    // ── The outcome half ──────────────────────────────────────────────────────
    {
        mm::AdmissionSnapshot snap;
        snap.reachable = true;
        const auto empty = draw(mm::render_admitted_models(snap));
        check(has(empty, "nothing admitted yet"), "an empty registry says so");

        mm::AdmittedView m;
        m.id = 42;
        m.name = "Qwen3-30B-A3B";
        m.verdict = "stream";
        m.attention_family = "gqa";
        m.bytes_per_token = 3ll * 1024 * 1024 * 1024;
        snap.models.push_back(m);
        const auto out = draw(mm::render_admitted_models(snap));
        check(has(out, "Qwen3-30B-A3B"), "admitted model is listed");
        check(has(out, "stream"), "the verdict is shown — it decides whether Soma serves it");
        check(has(out, "GB"), "bytes/token is humanized rather than raw");
    }

    std::cout << (failures == 0
                      ? "\nadmission_panels: OK\n"
                      : "\nadmission_panels: " + std::to_string(failures) + " failure(s)\n");
    return failures == 0 ? 0 : 1;
}
