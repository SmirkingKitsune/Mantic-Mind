#include "node/node_ui.hpp"
#include "node/node_state.hpp"
#include "common/util.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>
#include <ftxui/screen/terminal.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mm {

namespace {

std::string make_temp_llama_log_path() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path base = fs::temp_directory_path(ec);
    if (ec || base.empty()) {
        base = fs::current_path(ec);
    }
    if (ec || base.empty()) {
        return {};
    }
    const auto stamp = std::to_string(mm::util::now_ms());
    fs::path out = base / ("mantic_mind_llama_output_" + stamp + ".log");
    return out.string();
}

std::string shorten_middle(const std::string& s, size_t max_len) {
    if (s.size() <= max_len) return s;
    if (max_len < 8) return s.substr(0, max_len);
    size_t keep = (max_len - 3) / 2;
    return s.substr(0, keep) + "..." + s.substr(s.size() - keep);
}

} // namespace

NodeUI::NodeUI(NodeState& state, uint16_t listen_port,
               UpdateRequestCallback update_request_cb)
    : state_(state)
    , listen_port_(listen_port)
    , update_request_cb_(std::move(update_request_cb)) {
    log_file_path_ = make_temp_llama_log_path();
    if (!log_file_path_.empty()) {
        log_file_.open(log_file_path_, std::ios::out | std::ios::app);
    }
}

NodeUI::~NodeUI() = default;

// ── append_log (thread-safe) ──────────────────────────────────────────────────
void NodeUI::append_log(const std::string& line) {
    {
        std::lock_guard<std::mutex> lk(log_mutex_);
        if (log_lines_.size() >= kMaxLogLines)
            log_lines_.erase(log_lines_.begin());
        log_lines_.push_back(line);
        if (log_file_.is_open()) {
            log_file_ << line << '\n';
            log_file_.flush();
        }
    }
    std::function<void()> fn;
    {
        std::lock_guard<std::mutex> lk(screen_mutex_);
        fn = refresh_fn_;
    }
    if (fn) fn();
}

// ── quit (thread-safe) ────────────────────────────────────────────────────────
void NodeUI::quit() {
    std::function<void()> fn;
    {
        std::lock_guard<std::mutex> lk(screen_mutex_);
        fn = quit_fn_;
    }
    if (fn) fn();
}

// ── run ───────────────────────────────────────────────────────────────────────
void NodeUI::run() {
    using namespace ftxui;

    auto screen = ScreenInteractive::Fullscreen();

    {
        std::lock_guard<std::mutex> lk(screen_mutex_);
        quit_fn_    = screen.ExitLoopClosure();
        refresh_fn_ = [&screen]() { screen.PostEvent(Event::Custom); };
    }

    static const std::array<const char*, 10> kSpinner{
        "⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"
    };
    int frame = 0;

    // ── Renderer ──────────────────────────────────────────────────────────────
    auto render = Renderer([&]() -> Element {
        frame = (frame + 1) % static_cast<int>(kSpinner.size());

        // Snapshot mutable state
        bool registered   = state_.is_registered();
        auto metrics      = state_.get_metrics();
        auto loaded_model = state_.get_loaded_model();
        auto active_agent = state_.get_active_agent();
        auto slots        = state_.get_slots();
        auto node_id      = state_.get_node_id();
        auto last_error   = state_.get_last_error();
        auto upd          = state_.get_llama_update_state();
        auto rt           = state_.get_llama_runtime_summary();
        auto api_keys     = state_.get_api_keys();
        auto streaming    = state_.get_streaming_text();

        std::vector<std::string> log_snap;
        int log_scroll_from_bottom = 0;
        std::string log_file_path;
        {
            std::lock_guard<std::mutex> lk(log_mutex_);
            log_snap = log_lines_;
            log_scroll_from_bottom = log_scroll_from_bottom_;
            log_file_path = log_file_path_;
        }

        // ── Helper: gauge row ─────────────────────────────────────────────────
        auto gauge_row = [&](const std::string& label, float pct,
                              const std::string& detail) -> Element {
            float ratio = pct / 100.0f;
            if (ratio < 0.0f) ratio = 0.0f;
            if (ratio > 1.0f) ratio = 1.0f;
            Color bar_color = Color::Green;
            if (pct > 80.0f)      bar_color = Color::Red;
            else if (pct > 60.0f) bar_color = Color::Yellow;

            // Build percent string
            char pct_buf[16];
            snprintf(pct_buf, sizeof(pct_buf), "%5.1f%%", static_cast<double>(pct));

            return hbox({
                text(label) | size(WIDTH, EQUAL, 4),
                text(" "),
                gauge(ratio) | color(bar_color) | flex,
                text(" "),
                text(pct_buf),
                detail.empty() ? text("") : text("  " + detail),
            });
        };

        // ── RAM detail string ─────────────────────────────────────────────────
        auto mb_str = [](int64_t mb) -> std::string {
            if (mb >= 1024) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.1fG", static_cast<double>(mb) / 1024.0);
                return buf;
            }
            return std::to_string(mb) + "M";
        };

        std::string ram_detail = mb_str(metrics.ram_used_mb) + " / " + mb_str(metrics.ram_total_mb);
        std::string gpu_detail;
        if (metrics.gpu_vram_total_mb > 0)
            gpu_detail = mb_str(metrics.gpu_vram_used_mb) + " / " + mb_str(metrics.gpu_vram_total_mb);

        auto short_commit = [](const std::string& s) -> std::string {
            if (s.empty()) return "?";
            return s.size() > 10 ? s.substr(0, 10) : s;
        };

        std::string runtime_summary = upd.status.empty() ? "idle" : upd.status;
        runtime_summary += rt.update_available ? " | update available" : " | up-to-date";

        if (!rt.installed_commit.empty() || !rt.remote_commit.empty()) {
            runtime_summary += " | ";
            runtime_summary += short_commit(rt.installed_commit);
            runtime_summary += " -> ";
            runtime_summary += short_commit(rt.remote_commit);
        }

        if (!upd.message.empty()) {
            runtime_summary += " | ";
            runtime_summary += upd.message;
        }

        if (!rt.remote_error.empty()) {
            runtime_summary += " | remote: ";
            runtime_summary += rt.remote_error;
        }

        if (!last_error.empty()) {
            runtime_summary += " | error: ";
            runtime_summary += last_error;
        }

        Color runtime_color = Color::GrayDark;
        if (upd.status == "failed" || !last_error.empty() || !rt.remote_error.empty()) {
            runtime_color = Color::RedLight;
        } else if (upd.status == "running" || rt.update_available) {
            runtime_color = Color::Yellow;
        } else if (upd.status == "succeeded") {
            runtime_color = Color::Green;
        }

        if (!registered) {
            // ── Waiting state ─────────────────────────────────────────────────
            std::string spinner_text = std::string(kSpinner[static_cast<size_t>(frame)])
                                     + "  Waiting for Mantic-Mind-Control...";

            // Check for a pending pairing request — show PIN prominently.
            auto pp = state_.get_pending_pair();

            // Compute time remaining for the PIN display
            int64_t expires_s = 0;
            if (pp) {
                int64_t ms_left = pp->expiry_ms - mm::util::now_ms();
                expires_s = ms_left > 0 ? ms_left / 1000 : 0;
            }

            Elements waiting_elems = {
                text(""),
                text(" Mantic-Mind Node ") | bold | hcenter,
                separator(),
                text(""),
                text(spinner_text) | hcenter,
                text(""),
                hbox({
                    text("  Listen port : "),
                    text(std::to_string(listen_port_)) | bold,
                }),
                hbox({
                    text("  Node ID     : "),
                    text(node_id.empty() ? "(pending)" : node_id) | bold,
                }),
            };

            if (pp && !pp->pin.empty()) {
                // Show PIN prominently
                waiting_elems.push_back(text(""));
                waiting_elems.push_back(
                    hbox({
                        text("  PIN: "),
                        text(pp->pin) | bold | color(Color::Yellow),
                    })
                );
                waiting_elems.push_back(
                    text("  Expires in " + std::to_string(expires_s) + "s")
                    | color(Color::GrayDark)
                );
            } else {
                // Show truncated API key as before
                std::string keys_text;
                if (!api_keys.empty())
                    keys_text = api_keys.front().size() > 20
                        ? api_keys.front().substr(0, 20) + "..."
                        : api_keys.front();
                else
                    keys_text = "(none)";
                waiting_elems.push_back(
                    hbox({
                        text("  API key     : "),
                        text(keys_text) | bold,
                    })
                );
            }

            waiting_elems.push_back(text(""));
            waiting_elems.push_back(
                text("  Press u to update llama.cpp, q to quit") | color(Color::GrayDark));

            return vbox(waiting_elems) | border;
        }

        // ── Connected state ───────────────────────────────────────────────────
        int slot_ready = 0;
        int slot_loading = 0;
        int slot_suspending = 0;
        int slot_suspended = 0;
        int slot_error = 0;
        for (const auto& s : slots) {
            switch (s.state) {
                case SlotState::Ready:      ++slot_ready; break;
                case SlotState::Loading:    ++slot_loading; break;
                case SlotState::Suspending: ++slot_suspending; break;
                case SlotState::Suspended:  ++slot_suspended; break;
                case SlotState::Error:      ++slot_error; break;
                case SlotState::Empty:
                default:
                    break;
            }
        }
        const int slot_in_use = slot_ready + slot_loading + slot_suspending + slot_error;

        auto compact_id = [](const std::string& s, size_t n = 8) -> std::string {
            if (s.empty()) return "-";
            return s.size() <= n ? s : s.substr(0, n);
        };
        auto compact_model = [](const std::string& p) -> std::string {
            if (p.empty()) return "(none)";
            std::string name = std::filesystem::path(p).filename().string();
            if (name.empty()) name = p;
            if (name.size() > 26) name = name.substr(0, 26) + "...";
            return name;
        };

        // Status panel
        Elements status_rows = {
            text("Status") | bold | underlined,
            text(""),
            hbox({text("Model    "), text(loaded_model.empty() ? "(none)" : loaded_model) | bold}),
            hbox({text("Agent    "), text(active_agent.empty() ? "(none)" : active_agent) | bold}),
            hbox({text("Slots    "),
                  text(std::to_string(slot_in_use) + " active  (ready " + std::to_string(slot_ready)
                       + ", loading " + std::to_string(slot_loading)
                       + ", suspended " + std::to_string(slot_suspended) + ")")
                      | bold}),
            hbox({
                text("Runtime  "),
                paragraph(runtime_summary) | color(runtime_color) | flex,
            }),
        };
        if (!slots.empty()) {
            status_rows.push_back(text(" "));
            status_rows.push_back(text("Slots Detail") | color(Color::GrayDark));
            constexpr size_t kMaxSlotLines = 4;
            size_t shown = 0;
            for (const auto& s : slots) {
                if (shown >= kMaxSlotLines) break;
                status_rows.push_back(
                    text("  [" + to_string(s.state) + "] slot:" + compact_id(s.id, 10) +
                         " agent:" + compact_id(s.assigned_agent, 12) +
                         " model:" + compact_model(s.model_path))
                    | color(Color::GrayDark));
                ++shown;
            }
            if (slots.size() > shown) {
                status_rows.push_back(
                    text("  ... +" + std::to_string(slots.size() - shown) + " more")
                    | color(Color::GrayDark));
            }
        }
        auto status_panel = vbox(std::move(status_rows)) | flex;

        // Health panel
        auto health_panel = vbox({
            text("Health") | bold | underlined,
            text(""),
            gauge_row("CPU", metrics.cpu_percent, ""),
            gauge_row("RAM", metrics.ram_percent, ram_detail),
            gauge_row("GPU", metrics.gpu_percent, gpu_detail),
        }) | flex;

        // ── Generated text panel ─────────────────────────────────────────────
        auto tail_text = [](const std::string& s, size_t max_chars) -> std::string {
            if (s.size() <= max_chars) return s;
            return "..." + s.substr(s.size() - max_chars + 3);
        };

        // Split text into wrapped lines that fit the terminal width
        auto wrap_lines = [](const std::string& s, size_t width) -> Elements {
            Elements result;
            std::istringstream iss(s);
            std::string line;
            while (std::getline(iss, line)) {
                while (line.size() > width) {
                    result.push_back(text("    " + line.substr(0, width)));
                    line = line.substr(width);
                }
                result.push_back(text("    " + line));
            }
            if (result.empty()) result.push_back(text("    " + s));
            return result;
        };

        const int term_width = std::max(40, ftxui::Terminal::Size().dimx - 6);
        Elements gen_elems;
        int gen_panel_lines = 0; // track how many lines the gen panel uses

        if (streaming.active) {
            char header_buf[128];
            int64_t elapsed_ms = mm::util::now_ms() - streaming.started_ms;
            snprintf(header_buf, sizeof(header_buf),
                     "Generated Text  (streaming slot:%s agent:%s  %.1fs)",
                     streaming.slot_id.empty() ? "?" : shorten_middle(streaming.slot_id, 8).c_str(),
                     streaming.agent_id.empty() ? "?" : shorten_middle(streaming.agent_id, 12).c_str(),
                     static_cast<double>(elapsed_ms) / 1000.0);
            gen_elems.push_back(text(header_buf) | bold | underlined | color(Color::Green));

            if (!streaming.thinking.empty()) {
                gen_elems.push_back(text("  [thinking]") | color(Color::GrayDark));
                auto thinking_tail = tail_text(streaming.thinking, 500);
                auto lines = wrap_lines(thinking_tail, static_cast<size_t>(term_width - 4));
                for (auto& l : lines) gen_elems.push_back(std::move(l) | color(Color::GrayDark));
            }
            if (!streaming.content.empty()) {
                auto content_tail = tail_text(streaming.content, 1000);
                auto lines = wrap_lines(content_tail, static_cast<size_t>(term_width - 4));
                for (auto& l : lines) gen_elems.push_back(std::move(l) | color(Color::GreenLight));
            }

            char count_buf[64];
            snprintf(count_buf, sizeof(count_buf),
                     "  content:%zuB  thinking:%zuB",
                     streaming.content.size(), streaming.thinking.size());
            gen_elems.push_back(text(count_buf) | color(Color::GrayDark));
        } else if (!streaming.finish_reason.empty()) {
            // Show last completed generation
            char header_buf[128];
            int64_t duration_ms = streaming.updated_ms - streaming.started_ms;
            snprintf(header_buf, sizeof(header_buf),
                     "Generated Text  (done: %s  tokens:%d  %.1fs  slot:%s)",
                     streaming.finish_reason.c_str(),
                     streaming.tokens_used,
                     static_cast<double>(duration_ms) / 1000.0,
                     streaming.slot_id.empty() ? "?" : shorten_middle(streaming.slot_id, 8).c_str());
            gen_elems.push_back(text(header_buf) | bold | underlined);

            if (!streaming.content.empty()) {
                auto content_tail = tail_text(streaming.content, 1000);
                auto lines = wrap_lines(content_tail, static_cast<size_t>(term_width - 4));
                for (auto& l : lines) gen_elems.push_back(std::move(l) | color(Color::GreenLight));
            }

            char count_buf[64];
            snprintf(count_buf, sizeof(count_buf),
                     "  content:%zuB  thinking:%zuB",
                     streaming.content.size(), streaming.thinking.size());
            gen_elems.push_back(text(count_buf) | color(Color::GrayDark));
        } else {
            gen_elems.push_back(text("Generated Text") | bold | underlined);
            gen_elems.push_back(text("  (idle)") | color(Color::GrayDark));
        }
        gen_panel_lines = static_cast<int>(gen_elems.size()) + 1; // +1 for separator
        auto gen_panel = vbox(gen_elems);

        // Log panel — viewport adapts to terminal height
        const int effective_viewport = std::max(4, ftxui::Terminal::Size().dimy - 17 - gen_panel_lines);
        Elements log_elems;
        size_t total_lines = log_snap.size();
        size_t max_scroll = total_lines > static_cast<size_t>(effective_viewport)
            ? total_lines - static_cast<size_t>(effective_viewport)
            : 0;
        size_t scroll = static_cast<size_t>(std::max(0, log_scroll_from_bottom));
        if (scroll > max_scroll) scroll = max_scroll;

        size_t start_idx = 0;
        size_t end_idx = total_lines;
        if (total_lines > 0) {
            if (total_lines > static_cast<size_t>(effective_viewport) + scroll) {
                start_idx = total_lines - static_cast<size_t>(effective_viewport) - scroll;
            }
            end_idx = total_lines - scroll;
            if (end_idx > total_lines) end_idx = total_lines;
            if (start_idx > end_idx) start_idx = end_idx;
        }

        std::string log_label = "node runtime output";
        if (!log_file_path.empty()) {
            log_label += "  (" + shorten_middle(log_file_path, 56) + ")";
        }
        log_elems.push_back(text(log_label) | bold | underlined);

        if (log_snap.empty()) {
            log_elems.push_back(text("  (no output yet)") | color(Color::GrayDark));
        } else {
            for (size_t i = start_idx; i < end_idx; ++i) {
                log_elems.push_back(text("  > " + log_snap[i]) | color(Color::Cyan));
            }
        }
        char stat_buf[128];
        std::snprintf(stat_buf, sizeof(stat_buf),
                      "  showing %zu-%zu of %zu  [k/up:older  j/down:newer  PgUp/PgDn  End:follow]",
                      total_lines == 0 ? 0 : (start_idx + 1),
                      end_idx,
                      total_lines);
        log_elems.push_back(text(stat_buf) | color(Color::GrayDark));
        auto log_panel = vbox(log_elems);

        return vbox({
            hbox({
                text(" Mantic-Mind Node") | bold,
                text("  [" + node_id + "]") | color(Color::GrayDark) | flex,
            }),
            separator(),
            hbox({
                status_panel,
                separator(),
                health_panel,
            }),
            separator(),
            gen_panel,
            separator(),
            log_panel | flex,
            separator(),
            text("  Press u:update llama.cpp  j/k or arrows:scroll logs  q:quit")
                | color(Color::GrayDark),
        }) | border;
    });

    auto component = CatchEvent(render, [&](Event ev) {
        const int viewport = std::max(4, ftxui::Terminal::Size().dimy - 17);
        auto scroll_logs = [&](int delta) {
            std::lock_guard<std::mutex> lk(log_mutex_);
            const int total = static_cast<int>(log_lines_.size());
            const int max_scroll = std::max(0, total - viewport);
            log_scroll_from_bottom_ = std::clamp(log_scroll_from_bottom_ + delta, 0, max_scroll);
        };
        auto scroll_to_end = [&]() {
            std::lock_guard<std::mutex> lk(log_mutex_);
            log_scroll_from_bottom_ = 0;
        };
        auto scroll_to_oldest = [&]() {
            std::lock_guard<std::mutex> lk(log_mutex_);
            const int total = static_cast<int>(log_lines_.size());
            const int max_scroll = std::max(0, total - viewport);
            log_scroll_from_bottom_ = max_scroll;
        };

        if (ev == Event::Character('k') || ev == Event::ArrowUp) {
            scroll_logs(+1);
            return true;
        }
        if (ev == Event::Character('j') || ev == Event::ArrowDown) {
            scroll_logs(-1);
            return true;
        }
        if (ev == Event::PageUp) {
            scroll_logs(+kLogScrollPage);
            return true;
        }
        if (ev == Event::PageDown) {
            scroll_logs(-kLogScrollPage);
            return true;
        }
        if (ev == Event::Home) {
            scroll_to_oldest();
            return true;
        }
        if (ev == Event::End || ev == Event::Character('G')) {
            scroll_to_end();
            return true;
        }
        if (ev == Event::Character('u') && update_request_cb_) {
            update_request_cb_();
            return true;
        }
        if (ev == Event::Escape || ev == Event::Character('q')) {
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    // Background thread to drive animation at 10 fps
    std::atomic<bool> ticker_running{true};
    std::thread ticker([&]() {
        while (ticker_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            screen.PostEvent(Event::Custom);
        }
    });

    screen.Loop(component);

    ticker_running = false;
    ticker.join();

    {
        std::lock_guard<std::mutex> lk(screen_mutex_);
        quit_fn_    = {};
        refresh_fn_ = {};
    }
}

} // namespace mm
