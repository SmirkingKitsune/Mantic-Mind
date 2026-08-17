#pragma once

// Mantic-Mind — the Soma tab's FTXUI panels.
//
// Separate from soma_dashboard.hpp so that FTXUI is not forced on everything that
// wants the data types, and separate from control_ui.cpp because that file holds
// `NodeRegistry&`, `AgentManager&` and `AgentScheduler&`. A panel written there
// could read any of them without anyone noticing; a panel written here cannot,
// and tools/ci/check_ui_api.py guards both this header and its .cpp.
//
// Everything below takes a SNAPSHOT rather than the dashboard, so each renderer
// is a pure function of data it was handed. That is what lets the panels be
// rendered to an off-screen buffer and asserted on as text — a TUI whose only
// test is "a human looked at it" gets the graceful-degradation cases wrong,
// because those are exactly the states nobody opens the app in.

#include "control/soma_dashboard.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mm {

/// The engine list: id, backend, state, model, and who is on it.
///
/// `selected` is an index into `snap.engines`; out of range renders the list
/// without a highlight rather than clamping, so a stale selection after an
/// engine disappears is visible instead of silently pointing at its neighbour.
ftxui::Element render_engine_list(const SomaSnapshot& snap, int selected);

/// The brain grid for the selected engine.
///
/// Handles every degenerate case in one place, because they are the ones that
/// matter: no engine selected, a fallback engine that publishes no heat, a Soma
/// engine that has not been asked for anything yet, and a grid larger than the
/// viewport. Each says which it is — an empty grid and a cold grid look identical
/// otherwise, and they are completely different findings.
ftxui::Element
render_brain_grid(const SomaSnapshot& snap, std::uint32_t max_rows, std::uint32_t max_cols);

/// The memory-tier bar.
///
/// Shows the VRAM tier PRESENT AND EMPTY rather than hiding it. v1 is CPU-only by
/// design and `vram_hot_bytes` is always 0; a bar that omitted the tier would let
/// a reader assume the design has two tiers when it has three, one of which is
/// declared and stubbed. Visible-and-zero is the honest rendering of that.
ftxui::Element render_tier_bar(const SomaSnapshot& snap);

/// Live sequences on the selected engine: agent, phase, KV occupancy, and
/// determinism. An engine with no sequence telemetry says so explicitly.
ftxui::Element render_sequences(const SomaSnapshot& snap);

/// Staleness and the last error, as a single line.
ftxui::Element render_status_line(const SomaSnapshot& snap, std::int64_t now_ms);

/// The whole tab, wired to a live dashboard.
///
/// The only function here that touches the dashboard: it drives selection and
/// re-reads the snapshot each frame. Everything it draws goes through the
/// renderers above.
ftxui::Component soma_tab(SomaDashboard& dashboard, int& selected_index);

} // namespace mm
