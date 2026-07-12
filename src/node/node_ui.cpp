#include "node/node_ui.hpp"
#include "node/node_state.hpp"
#include "common/util.hpp"
#include "common/logger.hpp"
#include "common/tui_widgets.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>
#include <ftxui/screen/terminal.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mm {

namespace {

std::string make_temp_runtime_log_path() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path base = fs::temp_directory_path(ec);
    if (ec || base.empty()) {
        base = fs::current_path(ec);
    }
    if (ec || base.empty()) {
        return {};
    }
    // Best-effort prune of stale logs from earlier runs so they don't accumulate in
    // the temp dir. Only remove our own "mantic_mind_runtime_output_*.log" files whose
    // last-write time is >24h old, so a concurrently-running node's active log (which
    // is appended to continuously) is preserved. The current run's file is created
    // below, after this sweep, so it is never a candidate.
    {
        std::error_code lec;
        const auto now = fs::file_time_type::clock::now();
        for (const auto& entry : fs::directory_iterator(base, lec)) {
            const auto& p = entry.path();
            const std::string name = p.filename().string();
            if (name.rfind("mantic_mind_runtime_output_", 0) != 0) continue;
            if (p.extension() != ".log") continue;
            std::error_code fec;
            const auto mtime = fs::last_write_time(p, fec);
            if (fec) continue;
            if (now - mtime > std::chrono::hours(24)) {
                std::error_code rec;
                fs::remove(p, rec);  // ignore failures (file in use, perms, etc.)
            }
        }
    }
    const auto stamp = std::to_string(mm::util::now_ms());
    fs::path out = base / ("mantic_mind_runtime_output_" + stamp + ".log");
    return out.string();
}

std::string shorten_middle(const std::string& s, size_t max_len) {
    if (s.size() <= max_len) return s;
    if (max_len < 8) return mm::util::utf8_truncate(s, max_len);
    size_t keep = (max_len - 3) / 2;
    std::string head = mm::util::utf8_truncate(s, keep);
    // Advance the tail start past any UTF-8 continuation bytes so the slice begins
    // on a codepoint boundary rather than mid-sequence.
    size_t start = s.size() - keep;
    while (start < s.size() && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) ++start;
    return head + "..." + s.substr(start);
}

// Byte-wise tail of a string with a leading ellipsis when truncated.
std::string tail_text(const std::string& s, size_t max_chars) {
    if (max_chars < 2) return s.substr(s.size() > max_chars ? s.size() - max_chars : 0);
    if (s.size() <= max_chars) return s;
    size_t start = s.size() - (max_chars - 1);
    while (start < s.size() && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) ++start;
    return "…" + s.substr(start);
}

std::string clock_hms() {
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

std::string bytes_label(int64_t bytes) {
    const double value = static_cast<double>(std::max<int64_t>(0, bytes));
    char buf[48];
    if (value >= 1024.0 * 1024.0 * 1024.0)
        std::snprintf(buf, sizeof(buf), "%.2f GB", value / (1024.0 * 1024.0 * 1024.0));
    else if (value >= 1024.0 * 1024.0)
        std::snprintf(buf, sizeof(buf), "%.1f MB", value / (1024.0 * 1024.0));
    else if (value >= 1024.0)
        std::snprintf(buf, sizeof(buf), "%.1f KB", value / 1024.0);
    else
        std::snprintf(buf, sizeof(buf), "%.0f B", value);
    return buf;
}

} // namespace

NodeUI::NodeUI(NodeState& state, uint16_t listen_port,
               ForgetPairingCallback forget_pairing_cb,
               RequestVllmUpdateCallback request_vllm_update_cb)
    : state_(state)
    , listen_port_(listen_port)
    , forget_pairing_cb_(std::move(forget_pairing_cb))
    , request_vllm_update_cb_(std::move(request_vllm_update_cb)) {
    started_ms_ = mm::util::now_ms();
    log_file_path_ = make_temp_runtime_log_path();
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
            log_lines_.pop_front();
        log_lines_.push_back(line);
        if (log_file_.is_open()) {
            log_file_ << line << '\n';
            log_file_.flush();
        }
    }
    // Invoke the refresh closure while holding screen_mutex_ so the snapshot-and-call
    // is atomic with run()'s clear-on-exit. This thread (the runtime log pump) is
    // never joined by run(), so a copy-then-call would leave a window to PostEvent on a
    // destroyed screen. PostEvent is thread-safe and the clear block never re-enters here.
    std::lock_guard<std::mutex> lk(screen_mutex_);
    if (refresh_fn_) refresh_fn_();
}

// ── quit (thread-safe) ────────────────────────────────────────────────────────
void NodeUI::quit() {
    std::lock_guard<std::mutex> lk(screen_mutex_);
    if (quit_fn_) quit_fn_();
}

// ── run ───────────────────────────────────────────────────────────────────────
void NodeUI::run() {
    using namespace ftxui;
    using mm::tui::panel;
    using mm::tui::gauge_line;
    using mm::tui::braille_graph;
    using mm::tui::slot_table;
    using mm::tui::mb_str;
    using mm::tui::spinner_frame;

    auto screen = ScreenInteractive::Fullscreen();

    {
        std::lock_guard<std::mutex> lk(screen_mutex_);
        quit_fn_    = screen.ExitLoopClosure();
        refresh_fn_ = [&screen]() { screen.PostEvent(Event::Custom); };
    }

    static const std::array<const char*, 10> kSpinner{
        "⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"
    };
    // Log viewport height as last computed by the renderer. The scroll handler reads
    // this so Home/PgUp can reach the oldest lines.
    int log_viewport = std::max(4, ftxui::Terminal::Size().dimy - 17);
    std::string model_action_status;
    auto forget_pairing = [&]() {
        std::string msg;
        bool ok = false;
        if (forget_pairing_cb_) ok = forget_pairing_cb_(&msg);
        else msg = "forget callback not configured";
        if (msg.empty()) msg = ok ? "forgot remembered pairing keys" : "forget pairing failed";
        model_action_status = msg;
        return true;
    };
    auto btn_forget_pairing = Button("[F] Forget Pairing", [&] { forget_pairing(); },
                                     ButtonOption::Simple());

    // ── vLLM update prompt state ────────────────────────────────────────────────
    // These are recomputed each frame by the renderer; the buttons and modal read
    // them by reference.
    bool        show_update_modal  = false;
    bool        show_update_button = false;
    bool        show_action_modal  = false;
    std::string cur_latest;          // latest_version this frame (for button lambdas)
    std::string modal_ack_version;   // latest_version the user has acted on/dismissed

    auto request_update = [&] {
        if (request_vllm_update_cb_) request_vllm_update_cb_();
    };
    auto btn_update = Button("[ Update vLLM ]", [&] {
        request_update();
        modal_ack_version = cur_latest;   // close the modal until a newer build appears
    }, ButtonOption::Simple());
    auto btn_update_maybe = Maybe(btn_update, &show_update_button);

    auto modal_update_now = Button("  Update now  ", [&] {
        request_update();
        modal_ack_version = cur_latest;
        show_update_modal = false;
    }, ButtonOption::Simple());
    auto modal_later = Button("    Later    ", [&] {
        modal_ack_version = cur_latest;
        show_update_modal = false;
    }, ButtonOption::Simple());
    auto modal_btns = Container::Horizontal({modal_update_now, modal_later});

    auto action_cancel = Button("  Cancel  ", [&] {
        state_.request_action_cancel();
    }, ButtonOption::Simple());
    auto action_btns = Container::Horizontal({action_cancel});

    auto main_container = Container::Horizontal({btn_forget_pairing, btn_update_maybe});

    // ── Renderer ──────────────────────────────────────────────────────────────
    auto render = Renderer(main_container, [&]() -> Element {

        // Snapshot mutable state
        bool registered   = state_.is_registered();
        auto metrics      = state_.get_metrics();
        auto loaded_model = state_.get_loaded_model();
        auto active_agent = state_.get_active_agent();
        auto slots        = state_.get_slots();
        auto node_id      = state_.get_node_id();
        auto last_error   = state_.get_last_error();
        auto vllm_rt      = state_.get_vllm_runtime();
        auto vllm_prog    = state_.get_vllm_install_progress();
        auto action_prog  = state_.get_action_progress();
        auto api_keys     = state_.get_api_keys();
        auto streaming    = state_.get_streaming_text();

        show_action_modal = action_prog.active;

        // Drive the update button/modal from current runtime state.
        cur_latest = vllm_rt.latest_version;
        const bool can_update = vllm_rt.update_available && vllm_rt.managed &&
                                static_cast<bool>(request_vllm_update_cb_) && !vllm_prog.active;
        show_update_button = can_update;
        show_update_modal  = !show_action_modal && can_update &&
                             (modal_ack_version != vllm_rt.latest_version);

        int log_scroll_from_bottom = 0;
        std::string log_file_path;
        {
            std::lock_guard<std::mutex> lk(log_mutex_);
            log_scroll_from_bottom = log_scroll_from_bottom_;
            log_file_path = log_file_path_;
        }

        // Wall-clock spinner + cursor so they advance steadily regardless of how
        // often the renderer runs (renders also fire on every log line / keypress).
        const size_t spin_frame =
            static_cast<size_t>(mm::util::now_ms() / 100) % kSpinner.size();
        const char* spin = kSpinner[spin_frame];
        const bool cursor_on = (mm::util::now_ms() / 500) % 2 == 0;

        // ── Waiting (unregistered) state ──────────────────────────────────────
        if (!registered) {
            std::string spinner_text = std::string(spin) + "  Waiting for Mantic-Mind-Control...";
            auto pp = state_.get_pending_pair();
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
                hbox({text("  Listen port : "), text(std::to_string(listen_port_)) | bold}),
                hbox({text("  Node ID     : "), text(node_id.empty() ? "(pending)" : node_id) | bold}),
            };

            if (pp && !pp->pin.empty()) {
                waiting_elems.push_back(text(""));
                waiting_elems.push_back(
                    hbox({text("  PIN: "), text(pp->pin) | bold | color(Color::Yellow)}));
                waiting_elems.push_back(
                    text("  Expires in " + std::to_string(expires_s) + "s") | color(Color::GrayDark));
            } else {
                std::string keys_text;
                if (!api_keys.empty())
                    keys_text = api_keys.front().size() > 20
                        ? api_keys.front().substr(0, 20) + "..."
                        : api_keys.front();
                else
                    keys_text = "(none)";
                waiting_elems.push_back(hbox({text("  API key     : "), text(keys_text) | bold}));
            }

            waiting_elems.push_back(text(""));
            if (!model_action_status.empty()) {
                waiting_elems.push_back(text("  " + model_action_status) | color(Color::Yellow));
            }
            waiting_elems.push_back(btn_forget_pairing->Render() | hcenter);
            waiting_elems.push_back(text(""));
            waiting_elems.push_back(
                text("  Press f:forget pairing  q:quit") | color(Color::GrayDark));

            return vbox(waiting_elems) | border;
        }

        // ── Connected state — 1c "Operator" console ───────────────────────────

        // Slot tallies
        int slot_ready = 0, slot_loading = 0, slot_suspending = 0, slot_suspended = 0, slot_error = 0;
        for (const auto& s : slots) {
            switch (s.state) {
                case SlotState::Ready:      ++slot_ready; break;
                case SlotState::Loading:    ++slot_loading; break;
                case SlotState::Suspending: ++slot_suspending; break;
                case SlotState::Suspended:  ++slot_suspended; break;
                case SlotState::Error:      ++slot_error; break;
                case SlotState::Empty:
                default: break;
            }
        }
        const int slot_in_use = slot_ready + slot_loading + slot_suspending + slot_error;

        // Sample metric history at ~1 Hz. This lambda runs only on the FTXUI loop
        // thread, so the deques need no separate lock.
        const int64_t nowms = mm::util::now_ms();
        if (nowms - last_hist_ms_ >= 1000) {
            last_hist_ms_ = nowms;
            auto push = [](std::deque<float>& d, float v) {
                d.push_back(v);
                while (d.size() > kHistLen) d.pop_front();
            };
            push(hist_cpu_, metrics.cpu_percent);
            push(hist_ram_, metrics.ram_percent);
            push(hist_gpu_, metrics.gpu_percent);
            float vram_pct = metrics.gpu_vram_total_mb > 0
                ? static_cast<float>(metrics.gpu_vram_used_mb) /
                      static_cast<float>(metrics.gpu_vram_total_mb) * 100.0f
                : 0.0f;
            push(hist_vram_, vram_pct);
        }

        const bool has_gpu = metrics.gpu_vram_total_mb > 0 || metrics.gpu_backend_available;
        std::string ram_detail = mb_str(metrics.ram_used_mb) + " / " + mb_str(metrics.ram_total_mb);
        std::string gpu_detail = metrics.gpu_vram_total_mb > 0
            ? mb_str(metrics.gpu_vram_used_mb) + " / " + mb_str(metrics.gpu_vram_total_mb)
            : "";

        std::string runtime_summary = "ready";
        Color runtime_color = Color::Green;
        if (!last_error.empty()) {
            runtime_summary = "error: " + last_error;
            runtime_color = Color::RedLight;
        } else if (!vllm_rt.accelerator.empty()) {
            runtime_summary = "ready (" + vllm_rt.accelerator + ")";
        }

        // ── STATUS panel ──────────────────────────────────────────────────────
        auto field = [](const std::string& label, Element val) -> Element {
            return hbox({text(label) | dim | size(WIDTH, EQUAL, 9), std::move(val)});
        };
        Element slots_summary = hbox({
            text(std::to_string(slot_in_use) + " active  "),
            text("(") | dim,
            text(std::to_string(slot_ready) + " ready") | color(Color::Green),
            text(", ") | dim,
            text(std::to_string(slot_loading) + " loading") | color(Color::Yellow),
            text(", ") | dim,
            text(std::to_string(slot_suspended) + " sleeping") | dim,
            text(")") | dim,
        });
        Elements status_rows = {
            field("model", text(loaded_model.empty() ? "(none)" : loaded_model) | bold),
            field("agent", text(active_agent.empty() ? "(none)" : active_agent) | bold),
            field("slots", std::move(slots_summary)),
            field("runtime", text(runtime_summary) | color(runtime_color)),
        };
        if (!model_action_status.empty())
            status_rows.push_back(field("action", text(model_action_status) | color(Color::Yellow)));
        if (vllm_prog.active) {
            // Loading bar for an in-progress install/upgrade.
            std::string head = vllm_prog.stage.empty() ? "installing" : vllm_prog.stage;
            if (vllm_prog.total_steps > 0)
                head += "  (" + std::to_string(vllm_prog.step) + "/"
                      + std::to_string(vllm_prog.total_steps) + ")";
            Elements pr = { text(head) | color(Color::Cyan) | bold };
            if (vllm_prog.fraction >= 0.0) {
                const int pct = static_cast<int>(vllm_prog.fraction * 100.0 + 0.5);
                pr.push_back(hbox({ gauge(static_cast<float>(vllm_prog.fraction)) | flex,
                                    text(" " + std::to_string(pct) + "%") }));
            } else {
                pr.push_back(text(std::string(spin) + " working…") | color(Color::Yellow));
            }
            if (!vllm_prog.last_line.empty())
                pr.push_back(text(shorten_middle(vllm_prog.last_line, 44)) | dim);
            status_rows.push_back(field("install", vbox(std::move(pr))));
        } else if (vllm_rt.update_available) {
            Element line = text("vLLM " +
                (vllm_rt.version.empty() ? std::string{"?"} : vllm_rt.version) +
                " → " + vllm_rt.latest_version) | color(Color::Cyan) | bold;
            if (show_update_button)
                status_rows.push_back(field("update", vbox({ line, btn_update->Render() })));
            else
                status_rows.push_back(field("update",
                    vbox({ line, text("(update via package manager)") | dim })));
        }
        Element status_panel = panel("STATUS", vbox(std::move(status_rows)));

        // ── HEALTH panel (gauges + braille history) ───────────────────────────
        Elements health_rows = {
            gauge_line("CPU", metrics.cpu_percent),
            gauge_line("RAM", metrics.ram_percent, ram_detail),
        };
        if (has_gpu)
            health_rows.push_back(gauge_line("GPU", metrics.gpu_percent, gpu_detail));
        else
            health_rows.push_back(text("GPU   — no discrete GPU") | dim);
        health_rows.push_back(text(""));
        health_rows.push_back(hbox({
            vbox({text("gpu% 60s") | dim, braille_graph(hist_gpu_, 20, 3, Color::Green)}),
            text("   "),
            vbox({text("vram% 60s") | dim, braille_graph(hist_vram_, 20, 3, Color::Cyan)}),
        }));
        Element health_panel = panel("HEALTH", vbox(std::move(health_rows)));

        Element left_col = vbox({
            status_panel,
            health_panel | flex,
        }) | size(WIDTH, EQUAL, 48);

        // ── GENERATED TEXT panel ──────────────────────────────────────────────
        std::string gen_caption;
        Elements gen_body;
        if (streaming.active) {
            int64_t elapsed_ms = mm::util::now_ms() - streaming.started_ms;
            char cap[96];
            std::snprintf(cap, sizeof(cap), "%s  %.1fs",
                          streaming.slot_id.empty() ? "slot ?" : streaming.slot_id.c_str(),
                          static_cast<double>(elapsed_ms) / 1000.0);
            gen_caption = std::string(spin) + " " + cap;
            if (!streaming.thinking.empty()) {
                gen_body.push_back(text("[thinking]") | dim);
                std::string th = tail_text(streaming.thinking, 600);
                if (streaming.content.empty() && cursor_on) th += "▌";
                gen_body.push_back(paragraph(th) | dim);
            }
            if (!streaming.content.empty()) {
                std::string bd = tail_text(streaming.content, 1400);
                if (cursor_on) bd += "▌";
                gen_body.push_back(paragraph(bd));
            }
            char cnt[64];
            std::snprintf(cnt, sizeof(cnt), "content:%zuB  thinking:%zuB",
                          streaming.content.size(), streaming.thinking.size());
            gen_body.push_back(text(cnt) | dim);
        } else if (!streaming.finish_reason.empty()) {
            int64_t dur = streaming.updated_ms - streaming.started_ms;
            char cap[96];
            std::snprintf(cap, sizeof(cap), "✓ %s · %d tok · %.1fs",
                          streaming.finish_reason.c_str(), streaming.tokens_used,
                          static_cast<double>(dur) / 1000.0);
            gen_caption = cap;
            if (!streaming.content.empty())
                gen_body.push_back(paragraph(tail_text(streaming.content, 1400)));
            else
                gen_body.push_back(text("  (no content)") | dim);
        } else {
            gen_caption = "idle";
            gen_body.push_back(text("  (idle — waiting for inference)") | dim);
        }
        Element gen_panel = panel("GENERATED TEXT", gen_caption, vbox(std::move(gen_body))) | flex;

        // ── SLOTS panel ───────────────────────────────────────────────────────
        Element slots_panel =
            panel("SLOTS", std::to_string(slots.size()) + " tracked", slot_table(slots, false));

        Element right_col = vbox({
            gen_panel,
            slots_panel,
        }) | flex;

        const int dimy = ftxui::Terminal::Size().dimy;
        const int top_h = std::clamp(dimy - 16, 14, 26);
        Element top_region = hbox({
            left_col,
            text(" "),
            right_col,
        }) | size(HEIGHT, EQUAL, top_h);

        // ── LOG panel (full width) ────────────────────────────────────────────
        const int effective_viewport = std::max(4, dimy - top_h - 7);
        log_viewport = effective_viewport;

        std::vector<std::string> visible;
        size_t total_lines = 0;
        size_t start_idx = 0;
        size_t end_idx = 0;
        {
            std::lock_guard<std::mutex> lk(log_mutex_);
            total_lines = log_lines_.size();
            size_t scroll = static_cast<size_t>(std::max(0, log_scroll_from_bottom));
            size_t max_scroll = total_lines > static_cast<size_t>(effective_viewport)
                ? total_lines - static_cast<size_t>(effective_viewport)
                : 0;
            if (scroll > max_scroll) scroll = max_scroll;
            end_idx = total_lines;
            if (total_lines > 0) {
                if (total_lines > static_cast<size_t>(effective_viewport) + scroll)
                    start_idx = total_lines - static_cast<size_t>(effective_viewport) - scroll;
                end_idx = total_lines - scroll;
                if (end_idx > total_lines) end_idx = total_lines;
                if (start_idx > end_idx) start_idx = end_idx;
            }
            for (size_t i = start_idx; i < end_idx; ++i) visible.push_back(log_lines_[i]);
        }

        Elements log_rows;
        if (total_lines == 0)
            log_rows.push_back(text("  (no output yet)") | dim);
        else
            for (const auto& l : visible) log_rows.push_back(text("› " + l) | color(Color::Cyan));

        char stat_buf[192];
        std::snprintf(stat_buf, sizeof(stat_buf), "  showing %zu-%zu of %zu   [j/k · PgUp/PgDn · End follow]",
                      total_lines == 0 ? 0 : (start_idx + 1), end_idx, total_lines);
        Element live = log_scroll_from_bottom > 0
            ? (text("  ▲ scrolled") | color(Color::Yellow))
            : (text("  ● live") | color(Color::Green));
        Element log_body = vbox({
            vbox(std::move(log_rows)) | flex,
            separator(),
            hbox({text(stat_buf) | dim, live}),
        });
        Element log_panel = panel("runtime output",
                                  log_file_path.empty() ? "" : shorten_middle(log_file_path, 56),
                                  std::move(log_body)) | flex;

        // ── Header + footer ───────────────────────────────────────────────────
        std::string serving = slot_ready > 0 ? "serving" : (slot_loading > 0 ? "loading" : "idle");
        int64_t up_s = (mm::util::now_ms() - started_ms_) / 1000;
        char upbuf[24];
        std::snprintf(upbuf, sizeof(upbuf), "%02d:%02d:%02d",
                      static_cast<int>(up_s / 3600),
                      static_cast<int>((up_s / 60) % 60),
                      static_cast<int>(up_s % 60));

        Element header = hbox({
            text(" mantic-mind ") | bold,
            text(node_id) | dim,
            filler(),
            text(spin) | color(Color::Green), text(" " + serving) | dim,
            text("  │  ") | dim, text("✓ registered") | color(Color::Green),
            text("  │  ") | dim, text("up " + std::string(upbuf)) | dim,
            text("  │  ") | dim, text(clock_hms()) | dim,
            text(" "),
        });

        Element footer = hbox({
            btn_forget_pairing->Render(),
            text("  f forget · j/k scroll · PgUp/PgDn · Home/End · q quit") | dim,
            filler(),
            text("mantic-mind · node ") | dim,
        });

        return vbox({
            header,
            top_region,
            log_panel,
            footer,
        }) | border;
    });

    // ── Update-available prompt (FTXUI modal with buttons) ──────────────────────
    auto update_modal_renderer = Renderer(modal_btns, [&]() -> Element {
        auto rt = state_.get_vllm_runtime();
        return vbox({
            text(" vLLM update available ") | bold | hcenter,
            separator(),
            hbox({text(" installed : "),
                  text(rt.version.empty() ? std::string{"(unknown)"} : rt.version)}),
            hbox({text(" latest    : "), text(rt.latest_version) | color(Color::Green)}),
            hbox({text(" build     : "),
                  text(rt.accelerator.empty() ? std::string{"?"} : rt.accelerator) | dim}),
            separator(),
            modal_btns->Render() | hcenter,
        }) | border | size(WIDTH, EQUAL, 46);
    });
    auto action_modal_renderer = Renderer(action_btns, [&]() -> Element {
        auto p = state_.get_action_progress();
        const double derived_fraction =
            p.bytes_total > 0
                ? static_cast<double>(p.bytes_done) / static_cast<double>(p.bytes_total)
                : p.fraction;
        const bool determinate = derived_fraction >= 0.0;
        const double clamped_fraction = std::clamp(derived_fraction, 0.0, 1.0);
        const int pct = static_cast<int>(clamped_fraction * 100.0 + 0.5);

        Elements rows;
        rows.push_back(text(" " + (p.action.empty() ? std::string{"Working"} : p.action) + " ") |
                       bold | hcenter);
        rows.push_back(separator());
        if (!p.target.empty())
            rows.push_back(hbox({text(" target : ") | dim, text(shorten_middle(p.target, 42)) | bold}));
        if (!p.stage.empty())
            rows.push_back(hbox({text(" stage  : ") | dim, text(shorten_middle(p.stage, 42))}));
        if (p.total_steps > 0)
            rows.push_back(hbox({text(" step   : ") | dim,
                                 text(std::to_string(p.step) + "/" +
                                      std::to_string(p.total_steps))}));
        if (determinate) {
            rows.push_back(hbox({gauge(static_cast<float>(clamped_fraction)) | flex,
                                 text(" " + std::to_string(pct) + "%")}));
        } else {
            rows.push_back(text(std::string(" ") + spinner_frame() + " working") |
                           color(Color::Yellow));
        }
        if (p.bytes_total > 0) {
            rows.push_back(hbox({text(" data   : ") | dim,
                                 text(bytes_label(p.bytes_done) + " / " +
                                      bytes_label(p.bytes_total))}));
        }
        if (!p.detail.empty())
            rows.push_back(paragraph(" " + shorten_middle(p.detail, 80)) | dim);
        if (!p.last_error.empty())
            rows.push_back(paragraph(" " + p.last_error) | color(Color::Red));
        if (p.cancel_requested) {
            rows.push_back(text(" cancel requested...") | color(Color::Yellow) | hcenter);
        } else if (p.cancelable) {
            rows.push_back(separator());
            rows.push_back(action_btns->Render() | hcenter);
        }

        return vbox(std::move(rows)) | border | size(WIDTH, EQUAL, 56);
    });
    auto with_update = Modal(render, update_modal_renderer, &show_update_modal);
    auto with_modal = Modal(with_update, action_modal_renderer, &show_action_modal);

    auto component = CatchEvent(with_modal, [&](Event ev) {
        if (show_action_modal) {
            if (ev == Event::Escape) {
                state_.request_action_cancel();
                return true;
            }
            return false;
        }
        if (show_update_modal) {
            // While the prompt is up, its buttons handle input; Esc dismisses it.
            if (ev == Event::Escape) {
                modal_ack_version = cur_latest;
                show_update_modal = false;
                return true;
            }
            return false;
        }
        const int viewport = log_viewport;
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
        if (ev == Event::Character('f')) {
            return forget_pairing();
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

    // Run the loop under try/catch so an exception escaping a render lambda or event
    // handler cannot unwind past the still-joinable ticker (whose destructor would
    // call std::terminate and skip terminal restoration). Catch, then always clean up.
    std::string loop_error;
    try {
        screen.Loop(component);
    } catch (const std::exception& e) {
        loop_error = e.what();
    } catch (...) {
        loop_error = "unknown exception";
    }

    // Clear the screen callbacks first so a late append_log()/quit() from another
    // thread is a no-op instead of posting to an exited screen.
    {
        std::lock_guard<std::mutex> lk(screen_mutex_);
        quit_fn_    = {};
        refresh_fn_ = {};
    }

    ticker_running = false;
    if (ticker.joinable()) ticker.join();

    if (!loop_error.empty()) {
        MM_ERROR("Node UI event loop terminated by exception: {}", loop_error);
    }
}

} // namespace mm
