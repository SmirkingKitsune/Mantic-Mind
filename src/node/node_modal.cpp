// Mantic-Mind — which modal owns the node TUI's screen.
//
// Its OWN translation unit, and that is the point: node_ui.cpp pulls in FTXUI,
// and a decision that can only be exercised by standing up a terminal does not
// get exercised. Nothing here includes FTXUI, so the reliability suite compiles
// this file directly and asserts the ladder as arithmetic.
//
// What it replaces: five independent `show_*_modal` booleans whose mutual
// exclusion was maintained by five scattered "if X then Y = false" statements
// in the render recompute — and whose precedence was written a SECOND time, in
// a different order, as an if-ladder in the event handler. The two agreed only
// because the recompute happened to leave at most one boolean true. Nothing
// enforced that, and nothing would have caught them diverging.

#include "node/node_ui.hpp"

namespace mm {

NodeModal resolve_node_modal(const NodeModalInputs& in, NodeModal current) noexcept {
    // Highest priority first. Work in flight outranks every prompt: a modal
    // that hid a running compile would leave it running with nothing on screen
    // saying so, which is why Escape on this one CANCELS rather than dismisses.
    if (in.progress_active) return NodeModal::Progress;

    // EngineSwitch is sticky-only. Nothing about the runtime asks for it, so it
    // shows only while already open — which means the operator pressed the
    // button. It outranks the auto-opening prompts because it is the one that
    // was asked for, and a prompt appearing over a menu you just opened is how
    // a keypress lands on the wrong thing.
    if (current == NodeModal::EngineSwitch && in.engine_switch_available &&
        in.engine_variants_listed)
        return NodeModal::EngineSwitch;

    // The rest auto-open on a change and stay shut once acknowledged — but stay
    // OPEN while current, or acknowledging from inside the modal would close it
    // out from under the operator mid-read.
    //
    // Each is still gated on its `can_`: a runtime that stops offering an
    // update must close the prompt rather than let stickiness pin it open
    // against its own precondition.
    if (in.can_troubleshoot &&
        (current == NodeModal::Troubleshoot || in.troubleshoot_unacknowledged))
        return NodeModal::Troubleshoot;

    if (in.can_install_target && (current == NodeModal::Target || in.target_unacknowledged))
        return NodeModal::Target;

    if (in.can_update && (current == NodeModal::Update || in.update_unacknowledged))
        return NodeModal::Update;

    return NodeModal::None;
}

const char* to_string(NodeModal modal) noexcept {
    switch (modal) {
    case NodeModal::None:
        return "none";
    case NodeModal::Progress:
        return "progress";
    case NodeModal::EngineSwitch:
        return "engine-switch";
    case NodeModal::Troubleshoot:
        return "troubleshoot";
    case NodeModal::Target:
        return "target";
    case NodeModal::Update:
        return "update";
    }
    return "none";
}

} // namespace mm
