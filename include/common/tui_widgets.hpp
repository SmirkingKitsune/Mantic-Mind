#pragma once

// Shared FTXUI widget vocabulary for the mantic-mind terminal UIs (node + control).
//
// Header-only + inline so both executables can use it without a common→FTXUI link
// dependency. Colors follow the "terminal default" convention: no forced
// background, named colors used only semantically (green/yellow/red for load,
// dim for secondary text). Everything here builds FTXUI `Element`s out of unicode
// so the redesign works on any UTF-8 terminal.

#include "common/models.hpp"
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/mouse.hpp>
#include "common/util.hpp"

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace mm::tui {

// ── Semantic load color ───────────────────────────────────────────────────────
// Green under 65%, yellow up to 85%, red above — matching the mockup thresholds.
inline ftxui::Color load_color(float pct) {
    if (pct > 85.0f) return ftxui::Color::Red;
    if (pct > 65.0f) return ftxui::Color::Yellow;
    return ftxui::Color::Green;
}

// ── Clock / spinner ───────────────────────────────────────────────────────────
inline std::string clock_hms() {
    std::time_t now = std::time(nullptr);
    std::tm tmv{};
#ifdef _WIN32
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d", tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    return buf;
}

// Wall-clock-driven braille spinner frame, so it advances steadily no matter
// how often the caller re-renders.
inline const char* spinner_frame(int64_t period_ms = 100) {
    static const char* kFrames[10] = {"⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"};
    return kFrames[static_cast<size_t>((mm::util::now_ms() / period_ms) % 10)];
}

// ── Byte formatting ───────────────────────────────────────────────────────────
inline std::string mb_str(int64_t mb) {
    if (mb >= 1024) {
        char b[32];
        std::snprintf(b, sizeof(b), "%.1fG", static_cast<double>(mb) / 1024.0);
        return b;
    }
    return std::to_string(mb) + "M";
}

// ── Fixed-width column helpers ────────────────────────────────────────────────
// Wrap an element to an exact column width (truncating / padding as FTXUI does).
inline ftxui::Element col(ftxui::Element e, int width) {
    return std::move(e) | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, width);
}
inline ftxui::Element col_right(ftxui::Element e, int width) {
    return ftxui::align_right(std::move(e)) | ftxui::size(ftxui::WIDTH, ftxui::EQUAL, width);
}

// ── Titled panel ──────────────────────────────────────────────────────────────
// A bordered box with its title embedded in the top edge (the mockup's panel
// look). The two-arg overload appends a dim right-hand caption to the title.
inline ftxui::Element panel(const std::string& title, ftxui::Element body) {
    using namespace ftxui;
    return window(text(" " + title + " "), std::move(body), LIGHT);
}
inline ftxui::Element panel(const std::string& title, const std::string& caption,
                            ftxui::Element body) {
    using namespace ftxui;
    if (caption.empty()) return panel(title, std::move(body));
    Element t = hbox({text(" " + title + "  "), text("· " + caption + " ") | dim});
    return window(std::move(t), std::move(body), LIGHT);
}

// ── Count tile ────────────────────────────────────────────────────────────────
// A small bordered stat: dim label over a bold value. Used by the 1a dashboards.
inline ftxui::Element count_tile(const std::string& label, const std::string& value,
                                 ftxui::Color value_color, int min_width = 12) {
    using namespace ftxui;
    return vbox({
               text(label) | dim,
               text(value) | bold | color(value_color),
           }) |
           border | size(WIDTH, GREATER_THAN, min_width);
}

// ── Horizontal gauge ──────────────────────────────────────────────────────────
// Unicode partial-block bar coloured by load, with a dim '·' remainder so the
// full track is always visible.
inline ftxui::Element gauge_bar(float pct, int width) {
    using namespace ftxui;
    if (width < 1) width = 1;
    float ratio = pct / 100.0f;
    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 1.0f) ratio = 1.0f;

    const float filled = ratio * static_cast<float>(width);
    int full = static_cast<int>(filled);
    if (full > width) full = width;
    const float rem = filled - static_cast<float>(full);

    static const char* kParts[7] = {"▏", "▎", "▍", "▌", "▋", "▊", "▉"};
    std::string bar;
    bar.reserve(static_cast<size_t>(width) * 3);
    for (int i = 0; i < full; ++i) bar += "█";
    int used = full;
    if (full < width && rem > 0.06f) {
        int idx = static_cast<int>(rem * 8.0f) - 1;
        idx = std::clamp(idx, 0, 6);
        bar += kParts[idx];
        ++used;
    }
    std::string rest;
    for (int i = used; i < width; ++i) rest += "·";
    return hbox({text(bar) | color(load_color(pct)), text(rest) | dim});
}

// A gauge followed by its percentage and an optional trailing detail. Handy for
// the CPU/RAM/GPU rows shared by the node and node-detail panels.
inline ftxui::Element gauge_line(const std::string& label, float pct,
                                 const std::string& detail = "", int label_w = 4,
                                 int bar_w = 16) {
    using namespace ftxui;
    char pct_buf[16];
    std::snprintf(pct_buf, sizeof(pct_buf), "%3d%%", static_cast<int>(pct + 0.5f));
    Elements row = {
        text(label) | dim | size(WIDTH, EQUAL, label_w),
        text(" "),
        gauge_bar(pct, bar_w),
        text(" "),
        text(pct_buf),
    };
    if (!detail.empty()) row.push_back(text("  " + detail) | dim);
    return hbox(std::move(row));
}

// ── Sparkline ─────────────────────────────────────────────────────────────────
// One-line block-height plot of a 0..100 series, coloured by its latest value.
template <class Seq>
inline ftxui::Element sparkline(const Seq& s, int width, bool colorize = true) {
    using namespace ftxui;
    static const char* kBlocks[8] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    if (width < 1) width = 1;
    const int n = static_cast<int>(s.size());
    if (n == 0) return text(std::string(static_cast<size_t>(width), ' ')) | dim;

    const int denom = (width > 1) ? (width - 1) : 1;
    std::string out;
    out.reserve(static_cast<size_t>(width) * 3);
    for (int i = 0; i < width; ++i) {
        int idx = (n <= 1)
                      ? 0
                      : static_cast<int>(static_cast<long long>(i) * (n - 1) / denom);
        idx = std::clamp(idx, 0, n - 1);
        float v = static_cast<float>(s[static_cast<size_t>(idx)]);
        v = std::clamp(v, 0.0f, 100.0f);
        int b = std::clamp(static_cast<int>(v / 100.0f * 8.0f), 0, 7);
        out += kBlocks[b];
    }
    float last = std::clamp(static_cast<float>(s[static_cast<size_t>(n - 1)]), 0.0f, 100.0f);
    Element e = text(out);
    return colorize ? (e | color(load_color(last))) : (e | dim);
}

// ── Braille graph ─────────────────────────────────────────────────────────────
// A w_cells×h_cells block of braille chars plotting a 0..100 series filled from
// the baseline up — 2×4 sub-pixels per cell, so the effective resolution is
// (w_cells*2) columns by (h_cells*4) rows.
template <class Seq>
inline ftxui::Element braille_graph(const Seq& series, int w_cells, int h_cells,
                                    ftxui::Color col) {
    using namespace ftxui;
    if (w_cells < 1) w_cells = 1;
    if (h_cells < 1) h_cells = 1;
    const int cols = w_cells * 2;
    const int rows = h_cells * 4;
    const int n = static_cast<int>(series.size());

    std::vector<std::vector<bool>> grid(static_cast<size_t>(rows),
                                        std::vector<bool>(static_cast<size_t>(cols), false));
    if (n > 0) {
        const int denom = (cols > 1) ? (cols - 1) : 1;
        for (int x = 0; x < cols; ++x) {
            int idx = (n <= 1)
                          ? 0
                          : static_cast<int>(static_cast<long long>(x) * (n - 1) / denom);
            idx = std::clamp(idx, 0, n - 1);
            float v = std::clamp(static_cast<float>(series[static_cast<size_t>(idx)]), 0.0f, 100.0f);
            int top = rows - 1 - static_cast<int>(v / 100.0f * static_cast<float>(rows - 1) + 0.5f);
            top = std::clamp(top, 0, rows);
            for (int y = top; y < rows; ++y)
                grid[static_cast<size_t>(y)][static_cast<size_t>(x)] = true;
        }
    }

    static const int bl[4] = {0x01, 0x02, 0x04, 0x40};
    static const int br[4] = {0x08, 0x10, 0x20, 0x80};
    Elements lines;
    for (int cy = 0; cy < h_cells; ++cy) {
        std::string line;
        line.reserve(static_cast<size_t>(w_cells) * 3);
        for (int cx = 0; cx < w_cells; ++cx) {
            int bits = 0;
            for (int dy = 0; dy < 4; ++dy) {
                const int gy = cy * 4 + dy;
                if (grid[static_cast<size_t>(gy)][static_cast<size_t>(cx * 2)]) bits |= bl[dy];
                if (grid[static_cast<size_t>(gy)][static_cast<size_t>(cx * 2 + 1)]) bits |= br[dy];
            }
            const unsigned int cp = 0x2800u + static_cast<unsigned int>(bits);
            // U+2800..U+28FF encodes as three UTF-8 bytes (lead 0xE2).
            line += static_cast<char>(0xE0u | (cp >> 12));
            line += static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu));
            line += static_cast<char>(0x80u | (cp & 0x3Fu));
        }
        lines.push_back(text(line) | color(col));
    }
    return vbox(std::move(lines));
}

// ── Slot state glyph ──────────────────────────────────────────────────────────
inline ftxui::Element slot_state_el(SlotState st) {
    using namespace ftxui;
    switch (st) {
        case SlotState::Ready:      return text("● ready")      | color(Color::Green);
        case SlotState::Loading:    return text("◐ loading")    | color(Color::Yellow);
        case SlotState::Suspending: return text("◑ suspending") | color(Color::Yellow);
        case SlotState::Suspended:  return text("◌ sleeping")   | color(Color::GrayDark);
        case SlotState::Error:      return text("✗ error")      | color(Color::Red);
        case SlotState::Empty:
        default:                    return text("· empty")      | dim;
    }
}

// Short model label: last path/repo segment, truncated to `max` display bytes.
inline std::string short_model(const std::string& path, size_t max = 26) {
    if (path.empty()) return "(none)";
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    if (name.empty()) name = path;
    if (name.size() > max) name = mm::util::utf8_truncate(name, max) + "…";
    return name;
}

// KV-cache usage cell: small gauge + percentage.
inline ftxui::Element kv_cell(double usage_0_1) {
    using namespace ftxui;
    float pct = static_cast<float>(usage_0_1) * 100.0f;
    pct = std::clamp(pct, 0.0f, 100.0f);
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%3d%%", static_cast<int>(pct + 0.5f));
    return hbox({gauge_bar(pct, 5), text(" "), text(buf) | dim});
}

// ── Slot table ────────────────────────────────────────────────────────────────
// SLOT | STATE | MODEL | [AGENTS] | KV | REQ | VRAM. `wide` adds the AGENTS
// column and roomier widths for the control node-detail; the narrow form is for
// the node console's SLOTS panel.
inline ftxui::Element slot_table(const std::vector<SlotInfo>& slots, bool wide) {
    using namespace ftxui;
    if (slots.empty()) return text("  (no tracked slots)") | dim;

    const int wSlot = 6;
    const int wState = wide ? 13 : 11;
    const int wModel = wide ? 26 : 15;
    const int wAgents = 14;
    const int wKv = 11;
    const int wReq = 7;
    const int wVram = 6;
    auto sep = [] { return text(" │ ") | dim; };

    auto make_row = [&](Element slot, Element state, Element model, Element agents,
                        Element kv, Element req, Element vram) -> Element {
        Elements cells = {col(std::move(slot), wSlot), sep(),
                          col(std::move(state), wState), sep(),
                          col(std::move(model), wModel), sep()};
        if (wide) {
            cells.push_back(col(std::move(agents), wAgents));
            cells.push_back(sep());
        }
        cells.push_back(col(std::move(kv), wKv));
        cells.push_back(sep());
        cells.push_back(col(std::move(req), wReq));
        cells.push_back(sep());
        cells.push_back(col_right(std::move(vram), wVram));
        return hbox(std::move(cells));
    };

    Elements rows;
    rows.push_back(make_row(text("SLOT") | dim, text("STATE") | dim, text("MODEL") | dim,
                            text("AGENTS") | dim, text("KV") | dim, text("REQ") | dim,
                            text("VRAM") | dim));
    rows.push_back(separator());
    for (const auto& s : slots) {
        std::string slot_id = s.id;
        if (slot_id.rfind("slot-", 0) == 0) slot_id = "s" + slot_id.substr(5);
        const std::string agents = s.agent_ids.empty() ? "—" : mm::util::join(s.agent_ids, ",");
        Element req = text("r" + std::to_string(s.num_requests_running) + " w" +
                           std::to_string(s.num_requests_waiting));
        if (s.num_requests_waiting > 0) req = std::move(req) | color(Color::Yellow);
        else                            req = std::move(req) | dim;
        Element vram = s.vram_usage_mb > 0 ? text(mb_str(s.vram_usage_mb)) : (text("—") | dim);
        Element model = s.state == SlotState::Empty ? (text(short_model(s.model_path)) | dim)
                                                    : text(short_model(s.model_path));
        rows.push_back(make_row(text(slot_id) | dim, slot_state_el(s.state), std::move(model),
                                text(agents) | color(Color::Cyan), kv_cell(s.kv_cache_usage),
                                std::move(req), std::move(vram)));
    }
    return vbox(std::move(rows));
}

inline std::string short_node_id(const NodeId& id, size_t width = 8) {
    return id.size() <= width ? id : id.substr(0, width);
}

// Human-first node identity. Hostnames are sufficient until two nodes share
// one; only then is a short stable identifier appended.
inline std::string node_display_name(const NodeInfo& node,
                                     const std::vector<NodeInfo>& all_nodes) {
    std::string base = node.hostname;
    if (base.empty()) {
        const auto parsed = mm::util::parse_url(node.url);
        base = parsed.first.empty() ? "node" : parsed.first;
    }
    const int duplicates = static_cast<int>(std::count_if(
        all_nodes.begin(), all_nodes.end(), [&](const NodeInfo& other) {
            std::string other_base = other.hostname;
            if (other_base.empty()) other_base = mm::util::parse_url(other.url).first;
            return other_base == base;
        }));
    return duplicates > 1 ? base + " [" + short_node_id(node.id) + "]" : base;
}

inline ftxui::Color connection_color(NodeConnectionStatus status) {
    switch (status) {
        case NodeConnectionStatus::Online:      return ftxui::Color::Green;
        case NodeConnectionStatus::Unreachable: return ftxui::Color::Yellow;
        case NodeConnectionStatus::Offline:     return ftxui::Color::Red;
        default:                                return ftxui::Color::GrayDark;
    }
}

inline ftxui::Element connection_status_el(const NodeInfo& node) {
    using namespace ftxui;
    return text("● " + to_string(node.connection_status)) |
           color(connection_color(node.connection_status));
}

inline std::string sample_age(int64_t sampled_at_ms, int64_t now = mm::util::now_ms()) {
    if (sampled_at_ms <= 0) return "no sample";
    const int64_t age_s = std::max<int64_t>(0, (now - sampled_at_ms) / 1000);
    if (age_s < 60) return std::to_string(age_s) + "s ago";
    if (age_s < 3600) return std::to_string(age_s / 60) + "m ago";
    return std::to_string(age_s / 3600) + "h ago";
}

inline ftxui::Element stale_caption(const NodeInfo& node) {
    using namespace ftxui;
    if (node.connection_status == NodeConnectionStatus::Online) return text("");
    return text("last metrics " + sample_age(node.metrics_sampled_at_ms)) | dim;
}

}  // namespace mm::tui

namespace mm::tui {
enum class SplitAxis { Columns, Rows };

class ResizableDivider : public ftxui::ComponentBase {
public:
    ResizableDivider(int* value, int minimum, int maximum, SplitAxis axis,
                     std::function<void(int)> changed = {})
        : value_(value), minimum_(minimum), maximum_(maximum), axis_(axis),
          changed_(std::move(changed)) {}

    ftxui::Element OnRender() override {
        using namespace ftxui;
        Element divider = axis_ == SplitAxis::Columns ? separator() : separator();
        if (dragging_) divider = std::move(divider) | color(Color::Cyan);
        return divider | reflect(box_);
    }

    bool OnEvent(ftxui::Event event) override {
        using namespace ftxui;
        if (!event.is_mouse()) return false;
        auto& mouse = event.mouse();
        if (mouse.button == Mouse::Left && mouse.motion == Mouse::Pressed &&
            box_.Contain(mouse.x, mouse.y)) {
            captured_ = CaptureMouse(event);
            if (!captured_) return false;
            dragging_ = true;
            start_position_ = axis_ == SplitAxis::Columns ? mouse.x : mouse.y;
            start_value_ = *value_;
            return true;
        }
        if (dragging_ && mouse.motion == Mouse::Moved) {
            const int position = axis_ == SplitAxis::Columns ? mouse.x : mouse.y;
            *value_ = std::clamp(start_value_ + position - start_position_, minimum_, maximum_);
            return true;
        }
        if (dragging_ && mouse.motion == Mouse::Released) {
            dragging_ = false;
            captured_.reset();
            if (changed_) changed_(*value_);
            return true;
        }
        return false;
    }

private:
    int* value_;
    int minimum_;
    int maximum_;
    SplitAxis axis_;
    std::function<void(int)> changed_;
    ftxui::Box box_{};
    bool dragging_ = false;
    int start_position_ = 0;
    int start_value_ = 0;
    ftxui::CapturedMouse captured_;
};

inline ftxui::Component resizable_divider(int* value, int minimum, int maximum,
                                          SplitAxis axis,
                                          std::function<void(int)> changed = {}) {
    return std::make_shared<ResizableDivider>(value, minimum, maximum, axis,
                                              std::move(changed));
}

class WrappedScrollView : public ftxui::ComponentBase {
public:
    explicit WrappedScrollView(std::function<std::string()> content)
        : content_(std::move(content)) {}

    ftxui::Element OnRender() override {
        using namespace ftxui;
        Element body = paragraph(content_()) |
                       focusPositionRelative(0.0f, scroll_position_) |
                       frame | vscroll_indicator | flex;
        if (Focused()) body = std::move(body) | color(Color::Cyan);
        return body | reflect(box_);
    }

    bool OnEvent(ftxui::Event event) override {
        using namespace ftxui;
        float delta = 0.0f;
        if (event == Event::ArrowUp) delta = -0.04f;
        else if (event == Event::ArrowDown) delta = 0.04f;
        else if (event == Event::PageUp) delta = -0.25f;
        else if (event == Event::PageDown) delta = 0.25f;
        else if (event == Event::Home) { scroll_position_ = 0.0f; return true; }
        else if (event == Event::End) { scroll_position_ = 1.0f; return true; }
        else if (event.is_mouse() && box_.Contain(event.mouse().x, event.mouse().y)) {
            if (event.mouse().motion == Mouse::Pressed) TakeFocus();
            if (event.mouse().button == Mouse::WheelUp) delta = -0.08f;
            else if (event.mouse().button == Mouse::WheelDown) delta = 0.08f;
            else return event.mouse().motion == Mouse::Pressed;
        } else {
            return false;
        }
        scroll_position_ = std::clamp(scroll_position_ + delta, 0.0f, 1.0f);
        return true;
    }

private:
    std::function<std::string()> content_;
    float scroll_position_ = 0.0f;
    ftxui::Box box_{};
};

inline ftxui::Component wrapped_scroll_view(std::function<std::string()> content) {
    return std::make_shared<WrappedScrollView>(std::move(content));
}

class LayoutStore {
public:
    explicit LayoutStore(std::filesystem::path path) : path_(std::move(path)) {
        std::ifstream input(path_);
        if (!input) return;
        try { input >> values_; } catch (...) { values_ = nlohmann::json::object(); }
    }

    int get(const std::string& key, int fallback, int minimum, int maximum) const {
        try { return std::clamp(values_.value(key, fallback), minimum, maximum); }
        catch (...) { return fallback; }
    }

    void set(const std::string& key, int value) { values_[key] = value; }

    void save() const {
        std::error_code ec;
        if (path_.has_parent_path()) std::filesystem::create_directories(path_.parent_path(), ec);
        std::ofstream output(path_, std::ios::trunc);
        if (output) output << values_.dump(2) << '\n';
    }

private:
    std::filesystem::path path_;
    nlohmann::json values_ = nlohmann::json::object();
};


}  // namespace mm::tui
