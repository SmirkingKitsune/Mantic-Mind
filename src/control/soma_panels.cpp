// Mantic-Mind — the Soma tab's FTXUI panels.
//
// Includes FTXUI, the dashboard's own types, and the shared widget helpers.
// Nothing that carries in-process state; tools/ci/check_ui_api.py enforces it.

#include "control/soma_panels.hpp"

#include "common/tui_widgets.hpp"
#include "common/util.hpp"

#include <algorithm>
#include <cstdio>
#include <string>

namespace mm {

using namespace ftxui;

namespace {

/// The grid's glyph ramp. Five levels, because a terminal cell has to carry
/// "never fired" distinctly from "barely fired" — see layout_heat()'s intensity.
const char* kRamp[5] = {"·", "░", "▒", "▓", "█"};

/// Colour by HOW MUCH of this cell's traffic came from disk; brightness by how
/// much traffic there was.
///
/// Two channels for two independent facts, and collapsing them into one ramp is
/// the obvious simplification that destroys the panel's reason to exist — a hot
/// expert on disk and a hot expert in RAM would be the same cell, and the first
/// is the one costing throughput.
///
/// Banded rather than a smooth gradient because a terminal has eight usable
/// colours and an operator has to read a band off a legend, not interpolate one.
/// The bands are chosen so that ANY disk traffic is visibly not-green: a region
/// that reads from disk at all should never look clean.
Color cell_color(double cold_fraction, std::uint8_t intensity) {
    if (intensity == 0) return Color::GrayDark;
    if (cold_fraction <= 0.0) return Color::Green;  // entirely resident
    if (cold_fraction < 0.34) return Color::Cyan;   // mostly resident
    if (cold_fraction < 0.90) return Color::Yellow; // substantially streamed
    return Color::Magenta;                          // essentially all disk
}

std::string state_glyph(const std::string& state) {
    if (state == "Ready") return "●";
    if (state == "Loading" || state == "Suspending") return "◐";
    if (state == "Suspended") return "○";
    if (state == "Error") return "✖";
    return "·";
}

Color state_color(const std::string& state) {
    if (state == "Ready") return Color::Green;
    if (state == "Error") return Color::Red;
    if (state == "Suspended") return Color::GrayLight;
    return Color::Yellow;
}

/// Trailing component of a path, for a column that has to fit.
std::string short_model(const std::string& path) {
    if (path.empty()) return "-";
    const auto slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string age_label(std::int64_t at_ms, std::int64_t now_ms) {
    if (at_ms == 0) return "never";
    const auto secs = (now_ms - at_ms) / 1000;
    if (secs < 2) return "now";
    if (secs < 90) return std::to_string(secs) + "s ago";
    return std::to_string(secs / 60) + "m ago";
}

} // namespace

Element render_engine_list(const SomaSnapshot& snap, int selected) {
    if (snap.engines.empty()) {
        // Distinguished from "not polled yet", because a cluster with no engines
        // and a control that cannot be reached look identical on an empty list
        // and mean opposite things.
        const char* why = snap.engines_at_ms == 0 ? "no reading yet — is control reachable?"
                                                  : "no engines on any connected node";
        return tui::panel("ENGINES", text(why) | dim | center);
    }

    Elements rows;
    rows.push_back(hbox({
                       tui::col(text("ENGINE"), 14),
                       tui::col(text("BACKEND"), 11),
                       tui::col(text("STATE"), 12),
                       tui::col(text("MODEL"), 24),
                       tui::col(text("NODE"), 14),
                       tui::col_right(text("VRAM"), 9),
                       tui::col_right(text("CTX"), 8),
                       tui::col_right(text("AGENTS"), 8),
                   }) |
                   dim);

    for (int i = 0; i < static_cast<int>(snap.engines.size()); ++i) {
        const auto& e = snap.engines[static_cast<std::size_t>(i)];
        Element row = hbox({
            tui::col(text(e.id.substr(0, 12)), 14),
            // The backend is the field the whole tab turns on: a llama-cpp engine
            // is a first-class row with no grid, not a broken Soma one.
            tui::col(text(e.backend.empty() ? "-" : e.backend) |
                         color(e.soma() ? Color::Magenta : Color::GrayLight),
                     11),
            tui::col(hbox({text(state_glyph(e.state)) | color(state_color(e.state)),
                           text(" " + e.state)}),
                     12),
            tui::col(text(short_model(e.model_path)), 24),
            tui::col(text(e.node_id.substr(0, 12)) | dim, 14),
            tui::col_right(text(tui::mb_str(static_cast<std::int64_t>(e.vram_usage_mb))), 9),
            tui::col_right(
                text(e.effective_ctx_size > 0 ? std::to_string(e.effective_ctx_size) : "-"), 8),
            tui::col_right(text(std::to_string(e.agent_ids.size())), 8),
        });
        if (i == selected) row = std::move(row) | inverted;
        rows.push_back(std::move(row));
    }

    const auto caption = std::to_string(snap.engines.size()) + " engines";
    return tui::panel("ENGINES", caption, vbox(std::move(rows)));
}

Element
render_brain_grid(const SomaSnapshot& snap, std::uint32_t max_rows, std::uint32_t max_cols) {
    // Every one of these is a state an operator will actually be in, and each
    // says which. An empty grid and a cold grid are pixel-identical and mean
    // completely different things.
    if (snap.heat_engine_id.empty()) {
        return tui::panel("BRAIN", text("select an engine") | dim | center);
    }

    const auto it = std::find_if(snap.engines.begin(), snap.engines.end(), [&](const auto& e) {
        return e.id == snap.heat_engine_id;
    });
    if (it == snap.engines.end()) {
        return tui::panel("BRAIN",
                          text("engine " + snap.heat_engine_id + " is gone") | dim | center);
    }
    if (!it->soma()) {
        // The graceful-degradation case. A fallback engine publishes no expert
        // heat because it HAS no routed experts to report — that is a property
        // of the backend, not a fault, and the panel says so rather than showing
        // an empty grid that reads as an idle model.
        return tui::panel("BRAIN",
                          it->backend,
                          vbox({text("this engine is the fallback") | dim | center,
                                text("expert heat is a Soma capability") | dim | center}) |
                              vcenter);
    }
    if (snap.heat.empty()) {
        return tui::panel(
            "BRAIN",
            "soma",
            vbox({text("no heat frame yet") | dim | center,
                  text("the engine reports one once it routes a token") | dim | center}) |
                vcenter);
    }

    const auto g = layout_heat(snap.heat, max_rows, max_cols);
    if (g.cells.empty()) {
        return tui::panel("BRAIN", text("frame could not be laid out") | dim | center);
    }

    Elements grid_rows;
    for (std::uint32_t r = 0; r < g.rows; ++r) {
        Elements cells;
        for (std::uint32_t c = 0; c < g.cols; ++c) {
            const auto& cell = g.cells[static_cast<std::size_t>(r) * g.cols + c];
            if (!cell.present) {
                cells.push_back(text(" "));
                continue;
            }
            cells.push_back(text(kRamp[std::min<std::uint8_t>(cell.intensity, 4)]) |
                            color(cell_color(cell.cold_fraction(), cell.intensity)));
        }
        grid_rows.push_back(hbox(std::move(cells)));
    }

    // The caption carries the MODEL's shape and the two reductions separately.
    // They are different operations — the engine buckets to bound its payload,
    // the panel reduces to fit a terminal — and conflating them would tell the
    // operator the model is a shape it is not.
    char shape[160];
    std::snprintf(shape, sizeof(shape), "%ux%u experts", snap.heat.n_layers, snap.heat.n_experts);
    std::string caption = shape;
    if (snap.heat.bucketed()) {
        caption += "  · engine bucket " + std::to_string(snap.heat.layer_bucket) + "x" +
                   std::to_string(snap.heat.expert_bucket);
    }
    if (g.viewport_reduced) {
        caption += "  · view " + std::to_string(g.row_stride) + "x" + std::to_string(g.col_stride);
    }

    // Two lines rather than one. At the width a 64-column grid leaves, a single
    // legend line truncates mid-word — and the first thing it loses is "disk",
    // which is the tier the panel exists to make findable.
    Elements legend{
        hbox({
            text("from disk ") | dim,
            text("█ none") | color(Color::Green),
            text(" "),
            text("█ some") | color(Color::Cyan),
            text(" "),
            text("█ most") | color(Color::Yellow),
            text(" "),
            text("█ all") | color(Color::Magenta),
            filler(),
            text("peak " + std::to_string(g.hottest)) | dim,
        }),
        hbox({
            text("cold ") | dim,
            text("·░▒▓█") | dim,
            text(" hot") | dim,
            filler(),
            text(std::to_string(g.total) + " routes") | dim,
        }),
    };

    return tui::panel(
        "BRAIN", caption, vbox({vbox(std::move(grid_rows)), separator(), vbox(std::move(legend))}));
}

Element render_tier_bar(const SomaSnapshot& snap) {
    // Split by BACKEND, which `tier_summary` does not do. Summing every engine's
    // VRAM into one number puts a fallback's 22 GB under a tier that is supposed
    // to read as declared-and-empty — the operator then sees Soma using VRAM in a
    // release that is CPU-only by design, which is the exact misreading this
    // panel exists to prevent.
    std::int64_t soma_vram = 0, fallback_vram = 0;
    int soma_count = 0, fallback_count = 0;
    for (const auto& e : snap.engines) {
        if (e.soma()) {
            soma_vram += static_cast<std::int64_t>(e.vram_usage_mb);
            ++soma_count;
        } else {
            fallback_vram += static_cast<std::int64_t>(e.vram_usage_mb);
            ++fallback_count;
        }
    }
    const auto disk_mb = static_cast<std::int64_t>(snap.tiers.node_disk_free_mb);

    // Sized against the panel's 46 columns: 8 + 10 + 2 leaves 24 for the note,
    // and a note that truncates reads as a bug in the number beside it.
    const auto row = [](const char* label, const std::string& value, Element note) {
        return hbox({tui::col(text(label) | dim, 8),
                     tui::col_right(text(value), 10),
                     text("  "),
                     std::move(note)});
    };

    Elements rows;
    // VRAM is shown PRESENT AND EMPTY. v1 is CPU-only by design and the plan's
    // `vram_hot_bytes` is always 0; a bar that omitted the tier would let a
    // reader assume the design has two tiers when it has three, one declared and
    // stubbed. Hiding a stub is how a stub becomes a surprise.
    rows.push_back(row("VRAM",
                       tui::mb_str(soma_vram),
                       soma_vram == 0 ? text("declared; v1 is CPU-only") | dim
                                      : text("hot experts resident") | dim));
    rows.push_back(row("RAM", "-", text("per engine, in telemetry") | dim));
    rows.push_back(row("DISK", tui::mb_str(disk_mb), text("free on connected nodes") | dim));

    if (fallback_count > 0) {
        rows.push_back(separator());
        rows.push_back(row("fallback",
                           tui::mb_str(fallback_vram),
                           text(std::to_string(fallback_count) + " engine" +
                                (fallback_count == 1 ? "" : "s") + ", not Soma") |
                               dim));
    }

    const auto caption = std::to_string(soma_count) + " soma";
    return tui::panel("MEMORY TIERS", caption, vbox(std::move(rows)));
}

Element render_sequences(const SomaSnapshot& snap) {
    if (snap.sequences_engine_id.empty())
        return tui::panel("SEQUENCES", text("select an engine") | dim | center);

    if (snap.sequences.empty()) {
        const char* why = snap.sequences_at_ms == 0 ? "no reading yet" : "this engine reports none";
        return tui::panel("SEQUENCES", text(why) | dim | center);
    }

    Elements rows;
    rows.push_back(hbox({tui::col(text("SEQ"), 5),
                         tui::col(text("AGENT"), 14),
                         tui::col(text("PHASE"), 10),
                         tui::col_right(text("POS"), 7),
                         tui::col_right(text("KV"), 7)}) |
                   dim);
    for (const auto& s : snap.sequences) {
        const std::string phase = s.suspended ? "suspended" : s.prefilling ? "prefill" : "decode";
        rows.push_back(hbox({
            tui::col(text(std::to_string(s.index)), 5),
            tui::col(text(s.agent_id.empty() ? "-" : s.agent_id.substr(0, 12)), 14),
            tui::col(text(phase), 10),
            tui::col_right(text(std::to_string(s.position)), 7),
            tui::col_right(text(std::to_string(s.kv_tokens)), 7),
        }));
    }

    std::string caption = std::to_string(snap.sequences.size()) + " live";
    const auto& determinism = snap.sequences.front().determinism;
    if (!determinism.empty()) caption += " · " + determinism;
    return tui::panel("SEQUENCES", caption, vbox(std::move(rows)) | yframe);
}

Element render_status_line(const SomaSnapshot& snap, std::int64_t now_ms) {
    // Per-field staleness, not one timestamp for the snapshot: engines and heat
    // are separate requests and one can fail while the other succeeds. A single
    // "stale" flag would blank a panel that is fine.
    Elements parts{
        text(" engines ") | dim,
        text(age_label(snap.engines_at_ms, now_ms)),
        text("  · heat ") | dim,
        text(age_label(snap.heat_at_ms, now_ms)),
        text("  · sequences ") | dim,
        text(age_label(snap.sequences_at_ms, now_ms)),
    };
    if (!snap.last_error.empty()) {
        parts.push_back(text("  · ") | dim);
        parts.push_back(text(snap.last_error) | color(Color::Red));
    }
    parts.push_back(filler());
    return hbox(std::move(parts));
}

Component soma_tab(SomaDashboard& dashboard, int& selected_index) {
    // A menu over engine ids would need the list before the component exists.
    // Arrow keys drive the index instead, and the renderer resolves it against
    // whatever the latest snapshot holds — so an engine appearing or vanishing
    // between frames cannot leave the component pointing into a stale vector.
    auto events = CatchEvent(Renderer([] { return text(""); }), [&](Event e) {
        if (e == Event::ArrowDown || e == Event::Character('j')) {
            ++selected_index;
            return true;
        }
        if (e == Event::ArrowUp || e == Event::Character('k')) {
            if (selected_index > 0) --selected_index;
            return true;
        }
        return false;
    });

    return Renderer(events, [&] {
        const auto snap = dashboard.snapshot();
        const int count = static_cast<int>(snap.engines.size());
        if (count > 0) {
            selected_index = std::clamp(selected_index, 0, count - 1);
            const auto& want = snap.engines[static_cast<std::size_t>(selected_index)].id;
            // Told every frame rather than only on change: cheap, and it removes
            // the class of bug where the panel and the poller disagree about
            // which engine is on screen.
            if (dashboard.selected_engine() != want) dashboard.select_engine(want);
        } else {
            selected_index = 0;
        }

        // The grid takes the width; the tier bar is fixed. Letting both flex
        // gives the tier bar half the terminal to show four short rows, and
        // squeezes the grid until its legend truncates — losing "disk" first,
        // which is the tier the panel exists to make findable.
        return vbox({
            render_engine_list(snap, count > 0 ? selected_index : -1),
            hbox({
                render_brain_grid(snap, 24, 96) | flex,
                vbox({render_tier_bar(snap), render_sequences(snap) | flex}) |
                    size(WIDTH, EQUAL, 46),
            }),
            filler(),
            render_status_line(snap, util::now_ms()),
        });
    });
}

} // namespace mm
