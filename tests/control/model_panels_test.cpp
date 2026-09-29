// The Models tab's renderers, asserted as TEXT.
//
// The states worth testing are the ones nobody opens the app in: control
// unreachable, an empty catalog, an admitted model on a cluster with no Soma, a
// model busy with requests. Rendering to an off-screen buffer is what makes them
// assertable at all.

#include "control/model_panels.hpp"

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

std::string draw(const ftxui::Element& e, int w = 140, int h = 30) {
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(w), ftxui::Dimension::Fixed(h));
    auto page = ftxui::vbox({e});
    ftxui::Render(screen, page);
    return screen.ToString();
}

bool has(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

mm::ServedModelView admitted() {
    mm::ServedModelView m;
    m.id = "qwen3-30b-a3b-q4_g-q6_g-g128";
    m.display_name = "Qwen3-30B-A3B (q4_g-q6_g-g128)";
    m.model_path = "hash-qwen";
    m.source = "admitted";
    m.inference_backend = "node";
    m.backend_override = "soma";
    m.backend = "soma";
    m.backend_reason = "soma (operator_override, verdict=stream)";
    m.status = "active";
    m.node_id = "node-alpha";
    m.slot_id = "slot-1";
    m.in_flight = 2;
    m.ctx_size = 4096;
    return m;
}

mm::ServedModelView remote() {
    mm::ServedModelView m;
    m.id = "remote-gpt";
    m.model_path = "gpt-x";
    m.source = "explicit";
    m.inference_backend = "api";
    m.backend_override = "auto";
    m.status = "api";
    m.tool_calling = true;
    return m;
}

} // namespace

int main() {
    // ── Unreachable is not empty ─────────────────────────────────────────────
    {
        mm::ServedModelSnapshot snap;
        snap.error = "GET /v1/served-models → HTTP 0";
        const auto out = draw(mm::render_served_model_list(snap, 0));
        check(has(out, "HTTP 0"), "an unreachable control says so, with the reason");
        check(!has(out, "No served models"), "and does not claim the catalog is empty");
    }

    // ── Empty catalog ────────────────────────────────────────────────────────
    {
        mm::ServedModelSnapshot snap;
        snap.reachable = true;
        const auto out = draw(mm::render_served_model_list(snap, 0));
        check(has(out, "No served models"), "an empty catalog says what to do next");
        check(has(draw(mm::render_served_model_detail(snap, 0)), "select a model"),
              "the detail pane with nothing selected says so");
    }

    // ── A busy admitted model and a remote one ───────────────────────────────
    {
        mm::ServedModelSnapshot snap;
        snap.reachable = true;
        snap.models = {admitted(), remote()};
        const auto list = draw(mm::render_served_model_list(snap, 0));
        check(has(list, "qwen3-30b-a3b-q4_g-q6_g"), "the admitted model is listed by id");
        check(has(list, "admitted") && has(list, "explicit"), "each row says where it comes from");
        check(has(list, "active") && has(list, "2"), "requests in flight are visible");
        check(has(list, "api"), "an API-backed model shows as api, not as an engine");

        const auto detail = draw(mm::render_served_model_detail(snap, 0));
        check(has(detail, "verdict=stream"), "the detail carries the routing reason");
        check(has(detail, "slot-1"), "and where it is placed");
        check(has(detail, "cannot be deleted"), "and that an admitted entry is disabled, not deleted");
    }

    // ── Admitted, but no Soma in the engine policy ───────────────────────────
    {
        mm::ServedModelSnapshot snap;
        snap.reachable = true;
        auto m = admitted();
        m.available = false;
        m.unavailable_reason = "the cluster engine policy does not include Soma";
        m.status = "unplaced";
        snap.models = {m};
        check(has(draw(mm::render_served_model_list(snap, 0)), "unavailable"),
              "an unservable entry is marked unavailable in the list");
        check(has(draw(mm::render_served_model_detail(snap, 0)), "does not include Soma"),
              "and the detail says why");
    }

    // ── Error bodies read like reasons ───────────────────────────────────────
    {
        const auto issue = mm::describe_error_body(
            400,
            R"({"error":"validation_failed","issues":[{"severity":"error","field":"model_path",)"
            R"("message":"Model path is required."},{"severity":"warning","field":"x","message":"w"}]})");
        check(has(issue, "HTTP 400") && has(issue, "model_path: Model path is required."),
              "validation errors are listed by field");
        check(!has(issue, "x: w"), "warnings are not reported as the reason a save failed");
        check(has(mm::describe_error_body(502, "not json"), "not json"),
              "a non-JSON body is still shown");
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " (" << failures << " failure(s))\n";
    return failures == 0 ? 0 : 1;
}
