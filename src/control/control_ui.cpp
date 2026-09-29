#include "control/control_ui.hpp"
#include "control/node_registry.hpp"
#include "control/audio_player.hpp"
// The Soma tab, built entirely from /v1/*. Its panels live in their own
// translation unit precisely so they CANNOT reach the three registries this
// file holds by reference; tools/ci/check_ui_api.py enforces that.
#include "control/soma_dashboard.hpp"
#include "control/admission_panels.hpp"
#include "control/model_panels.hpp"
#include "control/user_panels.hpp"
#include "control/engine_panels.hpp"
#include "control/soma_panels.hpp"
#include "common/http_client.hpp"
#include "common/models.hpp"
#include "common/logger.hpp"
#include "common/tui_widgets.hpp"
#include "common/util.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/component_options.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <deque>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mm {

namespace {

// Post-order event catcher. FTXUI's built-in CatchEvent runs its handler BEFORE
// the wrapped component, so global single-key shortcuts (1-5, f, q) would be
// consumed before a focused Input on the Chat/Curation tabs could ever see them —
// making it impossible to type those characters (and letting 'q' quit the app
// mid-typing). This variant gives the wrapped component (and its focused
// descendants) first chance at the event; the handler only runs for events that
// nothing else consumed, so the shortcuts still work when focus is on a
// menu/button but no longer steal keystrokes from text fields.
class CatchEventAfter : public ftxui::ComponentBase {
public:
    explicit CatchEventAfter(std::function<bool(ftxui::Event)> on_event)
        : on_event_(std::move(on_event)) {}

    bool OnEvent(ftxui::Event event) override {
        if (ftxui::ComponentBase::OnEvent(event)) {
            return true;
        }
        return on_event_(event);
    }

private:
    std::function<bool(ftxui::Event)> on_event_;
};

ftxui::Component MakeCatchEventAfter(ftxui::Component child,
                                     std::function<bool(ftxui::Event)> on_event) {
    auto component = std::make_shared<CatchEventAfter>(std::move(on_event));
    component->Add(std::move(child));
    return component;
}

} // namespace

ControlUI::ControlUI(NodeRegistry& registry,
                     std::string control_base_url,
                     std::string control_api_token)
    : registry_(registry)
    , control_api_token_(std::move(control_api_token))
    , control_base_url_(std::move(control_base_url))
{}

ControlUI::~ControlUI() = default;

//

void ControlUI::log(LogLevel level, const std::string& message) {
    {
        std::lock_guard<std::mutex> lk(log_mutex_);
        if (log_entries_.size() >= 500)
            log_entries_.pop_front();
        log_entries_.push_back({level, message, util::now_ms()});
    }
    refresh();
}

//

void ControlUI::refresh() {
    std::function<void()> fn;
    { std::lock_guard<std::mutex> lk(screen_mutex_); fn = refresh_fn_; }
    if (fn) fn();
}

void ControlUI::set_pairing_key(const std::string& k) {
    pairing_key_ = k;
}

void ControlUI::quit() {
    std::function<void()> fn;
    { std::lock_guard<std::mutex> lk(screen_mutex_); fn = quit_fn_; }
    if (fn) fn();
}

//

void ControlUI::run() {
    using namespace ftxui;
    namespace fs = std::filesystem;

    auto screen = ScreenInteractive::Fullscreen();
    {
        std::lock_guard<std::mutex> lk(screen_mutex_);
        quit_fn_    = screen.ExitLoopClosure();
        refresh_fn_ = [&screen]() { screen.PostEvent(Event::Custom); };
    }

    //

    int tab_index = 0;
    int soma_selected = 0;
    int engine_selected = 0;
    int admission_selected = 0;
    int model_selected = 0;
    // Owned here rather than inside the tab: the same fact drives the
    // banner on every other tab, and two owners is how those disagree.
    bool force_engine_setup = false;
    mm::tui::LayoutStore layout("data/control-tui-layout.json");
    int node_detail_height = layout.get("nodes.detail_height", 20, 12, 40);
    int node_name_width = layout.get("nodes.name_width", 18, 12, 36);

    //
    int  disc_sel      = 0;
    std::vector<std::string> disc_entries;

    //
    int  node_sel      = 0;
    std::vector<std::string> node_entries;
    int node_slot_sel = 0;
    std::vector<std::string> node_slot_entries;
    std::string node_operation_status;

    // 1a "Overview" dashboard state. node_rows is the per-frame node snapshot the
    // node-menu transform indexes into (set at the top of nodes_renderer, read
    // during node_menu->Render() in the same frame). The history maps back the
    // per-node CPU/GPU sparklines and the cluster GPU braille graph; they are
    // sampled at ~1 Hz from inside the renderer (FTXUI loop thread only, no lock).
    std::vector<NodeInfo>                  node_rows;
    std::map<NodeId, std::deque<float>>    node_hist_cpu;
    std::map<NodeId, std::deque<float>>    node_hist_gpu;
    std::deque<float>                      cluster_gpu_hist;
    int64_t                                nodes_last_sample_ms = 0;
    constexpr size_t                       kHistLen = 60;

    // Modals
    bool show_add_node  = false;
    bool show_pin_entry = false;
    bool remember_pair_node = true;
    std::string add_url;
    std::string pin_input, pair_url, pair_nonce;

    // Activity tab
    int log_filter = 0;
    std::vector<std::string> filter_labels = {"All", "Info", "Warn", "Error"};

    nlohmann::json performance_data = nlohmann::json::object();
    int64_t performance_refreshed_ms = 0;
    std::string performance_error;
    bool performance_force_refresh = true;
    // The Users, Chat and Voice tabs share one user focus; their state lives in
    // control/user_panels.cpp.
    AudioPlayer audio_player;
    // Set on shutdown so a chat stream in flight aborts promptly instead of
    // making quit wait for the answer.
    std::atomic<bool> ui_shutting_down{false};

    //

    auto mb_str = [](int64_t mb) -> std::string {
        if (mb >= 1024) {
            char b[32];
            snprintf(b, sizeof(b), "%.1fG", static_cast<double>(mb) / 1024.0);
            return b;
        }
        return std::to_string(mb) + "M";
    };

    auto parse_api_error = [](const HttpResponse& resp) -> std::string {
        std::string body = util::trim(resp.body);
        if (!body.empty()) {
            try {
                auto j = nlohmann::json::parse(body);
                std::string error = j.value("error", std::string{});
                std::string detail = j.value("detail", std::string{});
                std::string message = j.value("message", std::string{});
                if (!error.empty() && !detail.empty()) return error + ": " + detail;
                if (!detail.empty()) return detail;
                if (!message.empty()) return message;
                if (!error.empty()) return error;
            } catch (...) {
            }
            if (body.size() > 240) body = body.substr(0, 240) + "...";
            return body;
        }
        return "(empty response body)";
    };

    //

    // Discovered (unregistered) nodes section
    auto disc_menu    = Menu(&disc_entries, &disc_sel, MenuOption::Vertical());
    auto btn_pair     = Button("[P] Pair", [&] {
        auto dns = registry_.get_discovered_nodes();
        if (disc_sel < 0 || disc_sel >= static_cast<int>(dns.size())) return;
        pair_url = dns[disc_sel].url;
        remember_pair_node = true;
        if (!pairing_key_.empty()) {
            //
            auto key = registry_.pair_node(pair_url, pairing_key_, remember_pair_node);
            if (!key.empty())
                log(LogLevel::Info,
                    "Paired with " + pair_url + " (PSK)" +
                    (remember_pair_node ? " and remembered" : ""));
            else
                log(LogLevel::Error, "PSK pairing failed for " + pair_url);
        } else {
            //
            // then open the PIN entry modal for the user to read it from the node TUI.
            pair_nonce = registry_.start_pair(pair_url);
            if (pair_nonce.empty()) {
                log(LogLevel::Error, "Could not reach node for pairing: " + pair_url);
                return;
            }
            pin_input.clear();
            show_pin_entry = true;
        }
    }, ButtonOption::Simple());
    auto btn_add_manual = Button("[+] Add Manually", [&] {
        add_url.clear();
        remember_pair_node = true;
        show_add_node = true;
    }, ButtonOption::Simple());
    auto disc_btns    = Container::Horizontal({btn_pair, btn_add_manual});
    // Maybe wrapper: disc_menu is excluded from the component tree (events + render)
    // when disc_entries is empty, preventing FTXUI from indexing an empty vector.
    auto disc_menu_m  = Maybe(disc_menu, [&]() { return !disc_entries.empty(); });
    auto disc_comp    = Container::Vertical({disc_menu_m, disc_btns});

    // Connected (registered) nodes section. The menu keeps FTXUI's keyboard
    // navigation + click-to-select, but each entry is rendered as a rich 1a table
    // row (health, CPU/GPU sparklines, VRAM gauge, slot dots) by indexing the
    // per-frame node_rows snapshot the renderer fills in.
    MenuOption node_menu_opt = MenuOption::Vertical();
    node_menu_opt.entries_option.transform = [&](const EntryState& st) -> Element {
        if (st.index < 0 || st.index >= static_cast<int>(node_rows.size()))
            return text(st.label);
        const NodeInfo& n = node_rows[static_cast<size_t>(st.index)];
        auto sep = [] { return text(" │ ") | dim; };

        const std::string display_name = mm::tui::node_display_name(n, node_rows);
        Element reachability = mm::tui::connection_status_el(n);

        const bool gpu = n.metrics.gpu_vram_total_mb > 0 || n.metrics.gpu_backend_available;
        char cpu_pct[8], gpu_pct[8];
        std::snprintf(cpu_pct, sizeof(cpu_pct), "%3d%%", static_cast<int>(n.metrics.cpu_percent + 0.5f));
        std::snprintf(gpu_pct, sizeof(gpu_pct), "%3d%%", static_cast<int>(n.metrics.gpu_percent + 0.5f));

        Element cpu_cell = hbox({mm::tui::sparkline(node_hist_cpu[n.id], 7), text(" "),
                                 text(cpu_pct) | dim});
        Element gpu_cell = gpu ? hbox({mm::tui::sparkline(node_hist_gpu[n.id], 7), text(" "),
                                       text(gpu_pct) | dim})
                               : (text("— cpu") | dim);

        float vram_pct = n.metrics.gpu_vram_total_mb > 0
            ? static_cast<float>(n.metrics.gpu_vram_used_mb) /
                  static_cast<float>(n.metrics.gpu_vram_total_mb) * 100.0f
            : 0.0f;
        Element vram_cell = gpu ? hbox({mm::tui::gauge_bar(vram_pct, 8), text(" "),
                                        text(mm::tui::mb_str(n.metrics.gpu_vram_used_mb)) | dim})
                                : (text("—") | dim);

        int used = 0;
        for (const auto& s : n.slots)
            if (s.state != SlotState::Empty) ++used;
        Elements dots;
        dots.push_back(text(std::to_string(used) + "/" + std::to_string(n.max_slots) + " ") | dim);
        for (const auto& s : n.slots) {
            switch (s.state) {
                case SlotState::Ready:      dots.push_back(text("●") | color(Color::Green)); break;
                case SlotState::Loading:    dots.push_back(text("◐") | color(Color::Yellow)); break;
                case SlotState::Suspending: dots.push_back(text("◑") | color(Color::Yellow)); break;
                case SlotState::Suspended:  dots.push_back(text("◌") | dim); break;
                case SlotState::Error:      dots.push_back(text("✗") | color(Color::Red)); break;
                case SlotState::Empty:
                default:                    dots.push_back(text("○") | dim); break;
            }
        }
        std::string arch = n.capabilities.arch.empty() ? "—" : n.capabilities.arch;

        Element row = hbox({
            mm::tui::col(text(display_name) | bold, node_name_width), sep(),
            mm::tui::col(std::move(reachability), 15), sep(),
            mm::tui::col(std::move(cpu_cell), 12), sep(),
            mm::tui::col(std::move(gpu_cell), 12), sep(),
            mm::tui::col(std::move(vram_cell), 15), sep(),
            mm::tui::col(hbox(std::move(dots)), 12), sep(),
            mm::tui::col(text(arch) | dim, 8),
        });
        if (st.active) row = std::move(row) | inverted;
        if (n.connection_status != NodeConnectionStatus::Online) row = std::move(row) | dim;
        return row;
    };
    auto node_menu   = Menu(&node_entries, &node_sel, node_menu_opt);
    auto forget_selected_node = [&]() {
        auto ns = registry_.list_nodes();
        if (node_sel < 0 || node_sel >= static_cast<int>(ns.size())) return;
        const auto& n = ns[node_sel];
        const bool changed = registry_.forget_node(n.id);
        log(LogLevel::Info,
            changed ? ("Forgot saved node " + n.id)
                    : ("Node " + n.id + " was not remembered"));
    };
    auto btn_forget_n = Button("[F] Forget", forget_selected_node, ButtonOption::Simple());
    auto btn_rem_n   = Button("[-] Remove", [&] {
        auto ns = registry_.list_nodes();
        if (node_sel >= 0 && node_sel < static_cast<int>(ns.size())) {
            registry_.remove_node(ns[node_sel].id);
            if (node_sel >= static_cast<int>(ns.size()) - 1)
                node_sel = std::max(0, static_cast<int>(ns.size()) - 2);
        }
    }, ButtonOption::Simple());
    auto node_menu_m  = Maybe(node_menu, [&]() { return !node_entries.empty(); });
    auto node_btns    = Container::Horizontal({btn_forget_n, btn_rem_n});
    auto save_node_layout = [&]() {
        layout.set("nodes.detail_height", node_detail_height);
        layout.set("nodes.name_width", node_name_width);
        layout.save();
    };
    auto node_rows_divider = mm::tui::resizable_divider(
        &node_detail_height, 12, 40, mm::tui::SplitAxis::Rows,
        [&](int) { save_node_layout(); });
    auto node_name_divider = mm::tui::resizable_divider(
        &node_name_width, 12, 36, mm::tui::SplitAxis::Columns,
        [&](int) { save_node_layout(); });

    auto nodes_section = Container::Vertical({node_menu_m, node_btns});

    auto selected_control_node = [&]() -> std::optional<NodeInfo> {
        const auto nodes = registry_.list_nodes();
        if (node_sel < 0 || node_sel >= static_cast<int>(nodes.size())) return std::nullopt;
        return nodes[static_cast<std::size_t>(node_sel)];
    };
    auto post_selected_node = [&](const std::string& path, const nlohmann::json& body) {
        const auto node = selected_control_node();
        if (!node || !node->connected) { node_operation_status = "node is not online"; return false; }
        HttpClient client(node->url);
        client.set_bearer_token(node->api_key);
        const auto response = client.post(path, body);
        node_operation_status = response.ok() ? "ok: " + path
            : "failed " + std::to_string(response.status) + ": " + parse_api_error(response);
        return response.ok();
    };
    auto node_slot_menu = Menu(&node_slot_entries, &node_slot_sel, MenuOption::Vertical());
    auto btn_control_cancel = Button(" Cancel action ", [&] {
        post_selected_node("/api/node/actions/cancel", nlohmann::json::object());
    }, ButtonOption::Simple());
    auto control_action_buttons = Container::Horizontal({btn_control_cancel});
    auto node_operations = Container::Vertical({node_slot_menu, control_action_buttons});

    auto nodes_comp  = Container::Vertical({disc_comp, nodes_section, node_name_divider,
        node_rows_divider, node_operations});

    //

    InputOption url_iopt; url_iopt.multiline = false;
    url_iopt.placeholder = "http://hostname:7070";
    auto modal_url    = Input(&add_url, url_iopt);
    auto modal_remember_cb = Checkbox("Remember this node", &remember_pair_node);
    auto modal_ok     = Button("  Connect  ", [&] {
        if (!add_url.empty()) {
            if (!pairing_key_.empty()) {
                auto key = registry_.pair_node(add_url, pairing_key_, remember_pair_node);
                if (!key.empty()) {
                    log(LogLevel::Info,
                        "Paired with " + add_url + " (PSK)" +
                        (remember_pair_node ? " and remembered" : ""));
                } else {
                    log(LogLevel::Error, "PSK pairing failed for " + add_url);
                }
            } else {
                pair_url = add_url;
                pair_nonce = registry_.start_pair(pair_url);
                if (pair_nonce.empty()) {
                    log(LogLevel::Error, "Could not reach node for pairing: " + pair_url);
                } else {
                    pin_input.clear();
                    show_pin_entry = true;
                }
            }
        }
        show_add_node = false;
    }, ButtonOption::Simple());
    auto modal_cancel = Button("  Cancel  ", [&] { show_add_node = false; },
                               ButtonOption::Simple());
    auto modal_btns   = Container::Horizontal({modal_ok, modal_cancel});
    auto modal_inputs = Container::Vertical({modal_url, modal_remember_cb, modal_btns});

    auto modal_renderer = Renderer(modal_inputs, [&]() {
        return vbox({
            text(" Add Node Manually ") | bold | hcenter,
            separator(),
            hbox({text(" URL : "), modal_url->Render() | flex}),
            modal_remember_cb->Render(),
            text(" Uses pairing (PIN or configured PSK)") | color(Color::GrayDark),
            separator(),
            modal_btns->Render() | hcenter,
        }) | border | size(WIDTH, EQUAL, 54);
    });

    //

    InputOption pin_iopt; pin_iopt.multiline = false;
    pin_iopt.placeholder = "000000";

    auto pin_input_comp   = Input(&pin_input, pin_iopt);
    auto pin_remember_cb  = Checkbox("Remember this node", &remember_pair_node);
    auto pin_ok           = Button("  Pair  ", [&] {
        if (!pin_input.empty()) {
            auto key = registry_.complete_pair(pair_url, pair_nonce, pin_input, remember_pair_node);
            if (!key.empty())
                log(LogLevel::Info,
                    "Paired with " + pair_url +
                    (remember_pair_node ? " and remembered" : ""));
            else
                log(LogLevel::Error, "PIN pairing failed for " + pair_url);
        }
        show_pin_entry = false;
    }, ButtonOption::Simple());
    auto pin_cancel       = Button("  Cancel  ", [&] { show_pin_entry = false; },
                                   ButtonOption::Simple());
    auto pin_modal_btns   = Container::Horizontal({pin_ok, pin_cancel});
    auto pin_modal_inputs = Container::Vertical({pin_input_comp, pin_remember_cb, pin_modal_btns});

    auto pin_modal_renderer = Renderer(pin_modal_inputs, [&]() {
        return vbox({
            text(" Enter PIN ") | bold | hcenter,
            separator(),
            text(" PIN shown on node TUI") | color(Color::GrayDark),
            hbox({text(" PIN: "), pin_input_comp->Render() | flex}),
            pin_remember_cb->Render(),
            separator(),
            pin_modal_btns->Render() | hcenter,
        }) | border | size(WIDTH, EQUAL, 36);
    });

    //

    auto filter_toggle = Toggle(&filter_labels, &log_filter);
    auto activity_comp = Container::Vertical({filter_toggle});
    auto refresh_performance = [&]() {
        performance_force_refresh = true;
        refresh();
    };
    auto btn_perf_refresh = Button(" Refresh ", refresh_performance, ButtonOption::Simple());
    auto btn_perf_clear = Button(" Clear session ", [&]() {
        HttpClient cli(control_base_url_);
        if (!control_api_token_.empty()) cli.set_bearer_token(control_api_token_);
        auto response = cli.del("/v1/performance");
        if (!response.ok()) {
            performance_error = "clear failed: HTTP " + std::to_string(response.status);
        } else {
            performance_data = nlohmann::json::object();
            performance_error.clear();
        }
        performance_force_refresh = true;
        refresh();
    }, ButtonOption::Simple());
    auto performance_buttons = Container::Horizontal({btn_perf_refresh, btn_perf_clear});
    auto performance_comp = Container::Vertical({performance_buttons});
    //

    // Tab 1 — 1a "Overview" dashboard
    auto nodes_renderer = Renderer(nodes_comp, [&]() {
        using mm::tui::panel;
        using mm::tui::gauge_line;
        using mm::tui::braille_graph;
        using mm::tui::slot_table;
        using mm::tui::col;

        // Discovered (unregistered) nodes
        auto dns = registry_.get_discovered_nodes();
        disc_entries.resize(dns.size());
        for (size_t i = 0; i < dns.size(); ++i)
            disc_entries[i] = dns[i].url + "  [" + dns[i].node_id.substr(0, 8) + "…]";
        if (!dns.empty() && disc_sel >= static_cast<int>(dns.size()))
            disc_sel = static_cast<int>(dns.size()) - 1;
        else if (dns.empty())
            disc_sel = 0;

        // Connected nodes — snapshot the rich menu transform indexes into.
        node_rows = registry_.list_nodes();
        node_entries.resize(node_rows.size());
        for (size_t i = 0; i < node_rows.size(); ++i) node_entries[i] = mm::tui::node_display_name(node_rows[i], node_rows);
        if (!node_rows.empty() && node_sel >= static_cast<int>(node_rows.size()))
            node_sel = static_cast<int>(node_rows.size()) - 1;
        else if (node_rows.empty())
            node_sel = 0;

        // Sample metric history at ~1 Hz for the sparklines + cluster graph.
        const int64_t nowms = util::now_ms();
        if (nowms - nodes_last_sample_ms >= 1000) {
            nodes_last_sample_ms = nowms;
            auto push = [](std::deque<float>& d, float v) {
                d.push_back(v);
                while (d.size() > kHistLen) d.pop_front();
            };
            float gsum = 0.0f; int gn = 0;
            for (const auto& n : node_rows) {
                const bool gpu = n.metrics.gpu_vram_total_mb > 0 || n.metrics.gpu_backend_available;
                push(node_hist_cpu[n.id], n.metrics.cpu_percent);
                push(node_hist_gpu[n.id], gpu ? n.metrics.gpu_percent : 0.0f);
                if (gpu && n.connected) { gsum += n.metrics.gpu_percent; ++gn; }
            }
            cluster_gpu_hist.push_back(gn ? gsum / static_cast<float>(gn) : 0.0f);
            while (cluster_gpu_hist.size() > kHistLen) cluster_gpu_hist.pop_front();
        }

        // ── Cluster summary tiles ────────────────────────────────────────
        float sum_cpu = 0.0f, sum_gpu = 0.0f;
        int up = 0, gnodes = 0, engines = 0;
        int64_t vram_used = 0, vram_total = 0;
        int64_t ram_used = 0, ram_total = 0;
        size_t placed = 0;
        for (const auto& n : node_rows) {
            if (!n.connected) continue;
            ++up;
            sum_cpu += n.metrics.cpu_percent;
            ram_used  += n.metrics.ram_used_mb;
            ram_total += n.metrics.ram_total_mb;
            const bool gpu = n.metrics.gpu_vram_total_mb > 0 || n.metrics.gpu_backend_available;
            if (gpu) {
                ++gnodes;
                sum_gpu += n.metrics.gpu_percent;
                vram_used += n.metrics.gpu_vram_used_mb;
                vram_total += n.metrics.gpu_vram_total_mb;
            }
            for (const auto& s : n.slots) {
                if (s.state == SlotState::Ready || s.state == SlotState::Loading) ++engines;
                placed += s.holder_ids.size();
            }
        }
        float avg_cpu = up ? sum_cpu / static_cast<float>(up) : 0.0f;
        float avg_gpu = gnodes ? sum_gpu / static_cast<float>(gnodes) : 0.0f;
        float vram_pct = vram_total
            ? static_cast<float>(vram_used) / static_cast<float>(vram_total) * 100.0f : 0.0f;
        float ram_pct = ram_total
            ? static_cast<float>(ram_used) / static_cast<float>(ram_total) * 100.0f : 0.0f;

        Element cluster_panel = panel("CLUSTER", vbox({
            gauge_line("CPU", avg_cpu),
            gauge_line("RAM", ram_pct, mb_str(ram_used) + " / " + mb_str(ram_total)),
            gauge_line("GPU", avg_gpu),
            gauge_line("VRM", vram_pct, mb_str(vram_used) + " / " + mb_str(vram_total)),
            text("  " + std::to_string(engines) + " engines · " + std::to_string(placed) +
                 " agents placed") | dim,
        }));

        char gcap[24];
        std::snprintf(gcap, sizeof(gcap), "60s · %d%%", static_cast<int>(avg_gpu + 0.5f));
        Element gpu_panel = panel("GPU", gcap, braille_graph(cluster_gpu_hist, 34, 4, Color::Green));

        Element disc_panel = panel("DISCOVERED", std::to_string(dns.size()), vbox({
            dns.empty() ? (text("  listening…") | dim) : (disc_menu->Render() | yframe | flex),
            disc_btns->Render(),
        }));

        Element top_strip = hbox({
            cluster_panel | flex,
            gpu_panel | size(WIDTH, EQUAL, 44),
            disc_panel | size(WIDTH, EQUAL, 40),
        }) | size(HEIGHT, EQUAL, 8);

        // ── Node table ───────────────────────────────────────────────────
        auto sep = [] { return text(" │ ") | dim; };
        Element head = hbox({
            col(text("NODE"), node_name_width), node_name_divider->Render(),
            col(text("STATUS"), 15), sep(),
            col(text("CPU%"), 12), sep(),
            col(text("GPU%"), 12), sep(),
            col(text("VRAM"), 15), sep(),
            col(text("SLOTS"), 12), sep(),
            col(text("ARCH"), 8),
        }) | dim | bold;
        Element table_body = node_rows.empty()
            ? (text("  No connected nodes.") | dim | flex)
            : (node_menu->Render() | yframe | flex);
        Element table_panel = panel("NODES", std::to_string(node_rows.size()) + " total", vbox({
            head,
            separator(),
            table_body,
            node_btns->Render(),
        })) | flex;

        // ── Detail (selected node) ───────────────────────────────────────
        Element detail_body;
        std::string detail_title = "DETAIL";
        std::string detail_cap;
        if (!node_rows.empty() && node_sel >= 0 && node_sel < static_cast<int>(node_rows.size())) {
            const NodeInfo& n = node_rows[static_cast<size_t>(node_sel)];
            node_slot_entries.clear();
            for (const auto& slot : n.slots)
                node_slot_entries.push_back(slot.id + "  " + to_string(slot.state) + "  " +
                    mm::tui::short_model(slot.model_path) +
                    (slot.vision_enabled ? "  [vision " +
                        fs::path(slot.mmproj_path).filename().string() + "]" : ""));
            if (node_slot_sel >= static_cast<int>(n.slots.size()))
                node_slot_sel = std::max(0, static_cast<int>(n.slots.size()) - 1);

            detail_title = "DETAIL · " + mm::tui::node_display_name(n, node_rows);
            detail_cap = n.url;
            // Per-node CPU/GPU/VRAM live in the NODES table row (sparklines/gauge)
            // and the CLUSTER tile; matching 1a, the detail shows the slot table +
            // capabilities only — no duplicate gauges here.
            Elements rows = {
                hbox({mm::tui::connection_status_el(n), text("   resource health ") | dim,
                      text(to_string(n.health)), text("   "), mm::tui::stale_caption(n)}),

                hbox({text("saved ") | dim, text(n.remembered ? "yes" : "no"),
                      text("   platform ") | dim, text(n.platform.empty() ? "—" : n.platform),
                      text("   model ") | dim,
                      text(n.loaded_model.empty() ? "(engine-managed)" : n.loaded_model)}),
                text(""),
                slot_table(n.slots, true),
                node_slot_menu->Render() | yframe | size(HEIGHT, LESS_THAN, 4),
                text("Placements are suspended, restored and released per model, on the Models tab.") | dim,
                control_action_buttons->Render(),
                node_operation_status.empty() ? text("") : paragraph(node_operation_status),

                text(""),
            };
            std::vector<std::string> caps = {
                "arch " + (n.capabilities.arch.empty() ? std::string("—") : n.capabilities.arch),
                std::to_string(n.capabilities.gpu_count) + " GPU",
                "llama.cpp " + (n.capabilities.llama_cpp_version.empty()
                    ? std::string("—") : n.capabilities.llama_cpp_version),
                std::string("RPC ") + (n.capabilities.supports_llama_rpc ? "yes" : "no"),
            };
            rows.push_back(text("  " + util::join(caps, "  ·  ")) | dim);
            detail_body = vbox(std::move(rows));
        } else {
            detail_body = text("  No connected nodes.") | dim;
        }
        Element detail_panel = panel(detail_title, detail_cap, std::move(detail_body)) |
                               size(HEIGHT, EQUAL, node_detail_height);

        return vbox({
            top_strip,
            table_panel,
            detail_panel,
            node_rows_divider->Render(),
        });
    });

    // Tab 3 — 1a "Overview" activity dashboard
    auto activity_renderer = Renderer(activity_comp, [&]() {
        using mm::tui::panel;
        using mm::tui::count_tile;
        using mm::tui::braille_graph;
        using mm::tui::col;

        auto sep = [] { return text(" │ ") | dim; };
        int c_info = 0, c_warn = 0, c_err = 0;
        // Event-rate histogram over the last 30 minutes for the "events/min" graph.
        constexpr int kRateBuckets = 40;
        constexpr int64_t kRateSpanMs = 30LL * 60 * 1000;
        std::vector<float> rate(kRateBuckets, 0.0f);
        const int64_t act_now = util::now_ms();
        Elements rows;
        {
            std::lock_guard<std::mutex> lk(log_mutex_);
            for (const auto& e : log_entries_) {
                if (e.level == LogLevel::Warn)       ++c_warn;
                else if (e.level == LogLevel::Error) ++c_err;
                else                                 ++c_info;
                if (e.timestamp_ms > 0) {
                    const int64_t age = act_now - e.timestamp_ms;
                    if (age >= 0 && age < kRateSpanMs) {
                        const int b = kRateBuckets - 1 -
                                      static_cast<int>(age / (kRateSpanMs / kRateBuckets));
                        if (b >= 0 && b < kRateBuckets) rate[static_cast<size_t>(b)] += 1.0f;
                    }
                }
            }
            // Newest-first, up to 300 matching the level filter.
            for (int i = static_cast<int>(log_entries_.size()) - 1;
                 i >= 0 && rows.size() < 300; --i) {
                const auto& e = log_entries_[static_cast<size_t>(i)];
                if (log_filter == 1 && e.level != LogLevel::Info)  continue;
                if (log_filter == 2 && e.level != LogLevel::Warn)  continue;
                if (log_filter == 3 && e.level != LogLevel::Error) continue;

                std::string lvl = e.level == LogLevel::Warn  ? "WARN"
                                : e.level == LogLevel::Error ? "ERROR" : "INFO";
                std::string ts;
                if (e.timestamp_ms > 0) {
                    time_t secs = static_cast<time_t>(e.timestamp_ms / 1000);
                    struct tm tm_buf{};
#ifdef _WIN32
                    localtime_s(&tm_buf, &secs);
#else
                    localtime_r(&secs, &tm_buf);
#endif
                    char tbuf[16];
                    snprintf(tbuf, sizeof(tbuf), "%02d:%02d:%02d",
                             tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
                    ts = tbuf;
                }
                Element lvl_el = text(lvl);
                Element msg_el = paragraph(e.message);
                if (e.level == LogLevel::Warn) {
                    lvl_el = std::move(lvl_el) | color(Color::Yellow) | bold;
                    msg_el = std::move(msg_el) | color(Color::Yellow);
                } else if (e.level == LogLevel::Error) {
                    lvl_el = std::move(lvl_el) | color(Color::Red) | bold;
                    msg_el = std::move(msg_el) | color(Color::Red);
                } else {
                    lvl_el = std::move(lvl_el) | dim;
                }
                rows.push_back(hbox({
                    col(text(ts) | dim, 9), sep(),
                    col(std::move(lvl_el), 6), sep(),
                    std::move(msg_el) | flex,
                }));
            }
        }
        if (rows.empty())
            rows.push_back(text("  (no events yet)") | dim);

        // Scale the rate histogram to 0..100 (relative heights) for the graph.
        float rmax = 1.0f;
        for (float v : rate) rmax = std::max(rmax, v);
        for (float& v : rate) v = v / rmax * 100.0f;

        const int total = c_info + c_warn + c_err;
        Element tiles = hbox({
            count_tile("EVENTS", std::to_string(total), Color::Default),
            count_tile("INFO", std::to_string(c_info), Color::Green),
            count_tile("WARN", std::to_string(c_warn), Color::Yellow),
            count_tile("ERROR", std::to_string(c_err), Color::Red),
            panel("EVENTS/MIN", "30m", braille_graph(rate, 16, 2, Color::Cyan)),
            panel("FILTER", hbox({filter_toggle->Render(), filler()})) | flex,
        }) | size(HEIGHT, EQUAL, 4);

        Element head = hbox({
            col(text("TIME"), 9), sep(),
            col(text("LVL"), 6), sep(),
            text("EVENT") | flex,
        }) | dim | bold;

        Element table = panel("ACTIVITY", "GET /v1/activity", vbox({
            head,
            separator(),
            vbox(std::move(rows)) | yframe | flex,
        })) | flex;

        return vbox({tiles, table});
    });

    auto performance_renderer = Renderer(performance_comp, [&]() -> Element {
        const int64_t now = util::now_ms();
        if (performance_force_refresh || now - performance_refreshed_ms >= 1000) {
            performance_force_refresh = false;
            performance_refreshed_ms = now;
            HttpClient cli(control_base_url_);
            if (!control_api_token_.empty()) cli.set_bearer_token(control_api_token_);
            auto response = cli.get("/v1/performance?tail=300");
            if (response.ok()) {
                try {
                    performance_data = nlohmann::json::parse(response.body);
                    performance_error.clear();
                } catch (const std::exception& e) {
                    performance_error = e.what();
                }
            } else {
                performance_error = "HTTP " + std::to_string(response.status);
            }
        }

        const auto aggregate = performance_data.value("aggregate", nlohmann::json::object());
        const auto total_stats = aggregate.value("total_ms", nlohmann::json::object());
        const auto ttft_stats = aggregate.value("time_to_first_token_ms", nlohmann::json::object());
        const auto rate_stats = aggregate.value("output_tokens_per_second", nlohmann::json::object());
        auto number = [](double value, const char* suffix) {
            char buffer[48];
            std::snprintf(buffer, sizeof(buffer), "%.1f%s", value, suffix);
            return std::string(buffer);
        };
        Element summary = hbox({
            mm::tui::count_tile("REQUESTS", std::to_string(aggregate.value("requests", 0)), Color::Cyan),
            mm::tui::count_tile("SUCCESS", std::to_string(aggregate.value("successful", 0)), Color::Green),
            mm::tui::count_tile("P50 TOTAL", number(total_stats.value("p50", 0.0), " ms"), Color::Green),
            mm::tui::count_tile("P95 TTFT", number(ttft_stats.value("p95", 0.0), " ms"), Color::Yellow),
            mm::tui::count_tile("AVG TOK/S", number(rate_stats.value("average", 0.0), ""), Color::Cyan),
            mm::tui::count_tile("TOKENS", std::to_string(aggregate.value("output_tokens", 0)), Color::Magenta),
        });

        Elements rows;
        rows.push_back(hbox({
            mm::tui::col(text("MODEL") | bold, 18),
            mm::tui::col(text("USER") | bold, 12),
            mm::tui::col(text("BACKEND") | bold, 11),
            mm::tui::col(text("QUEUE") | bold, 10),
            mm::tui::col(text("TTFT") | bold, 10),
            mm::tui::col(text("TOTAL") | bold, 10),
            mm::tui::col(text("TOK/S") | bold, 10),
            text("RESULT") | bold,
        }) | dim);
        rows.push_back(separator());
        const auto samples = performance_data.value("samples", nlohmann::json::array());
        for (auto it = samples.rbegin(); it != samples.rend(); ++it) {
            const auto& sample = *it;
            const bool success = sample.value("success", false);
            const int64_t ttft = sample.value("time_to_first_token_ms", -1LL);
            rows.push_back(hbox({
                mm::tui::col(text(sample.value("served_model_id", std::string{})), 18),
                mm::tui::col(text(sample.value("user_id", std::string{})), 12),
                mm::tui::col(text(sample.value("backend", std::string{})), 11),
                mm::tui::col(text(std::to_string(sample.value("queue_ms", 0LL)) + " ms"), 10),
                mm::tui::col(text(ttft >= 0 ? std::to_string(ttft) + " ms" : "--"), 10),
                mm::tui::col(text(std::to_string(sample.value("total_ms", 0LL)) + " ms"), 10),
                mm::tui::col(text(number(sample.value("output_tokens_per_second", 0.0), "")), 10),
                text(success ? "ok" : "failed") | color(success ? Color::Green : Color::Red),
            }));
            const std::string error = sample.value("error", std::string{});
            if (!error.empty()) rows.push_back(paragraph("  " + error) | color(Color::Red));
        }
        if (samples.empty()) rows.push_back(text("No completed inference samples this session.") | dim);

        Elements page = {summary, separator()};
        if (!performance_error.empty())
            page.push_back(paragraph("Performance endpoint: " + performance_error) | color(Color::Red));
        page.push_back(mm::tui::panel("RECENT REQUESTS", vbox(std::move(rows)) | yframe | flex));
        page.push_back(performance_buttons->Render());
        page.push_back(text("Session rolling history; input-token counts are estimated when the backend omits usage.") | dim);
        return vbox(std::move(page));
    });

    // Polls on its own thread at the tab's cadence rather than inside a
    // renderer: an HTTP round trip on the render path stalls every other panel,
    // and the existing performance tab already shows what that feels like.
    SomaDashboard soma_dashboard(control_base_url_, control_api_token_);
    soma_dashboard.start(500);
    auto soma_renderer = soma_tab(soma_dashboard, soma_selected);

    // Same discipline as the Soma tab: its own translation unit, fed entirely
    // by /v1/*, unable to reach the three references this class holds.
    EngineDashboard engine_dashboard(control_base_url_, control_api_token_);
    engine_dashboard.start(1000);
    auto engines_renderer = engine_tab(engine_dashboard, engine_selected, force_engine_setup);

    // Admission is the longest operation in the system — hours, hundreds of GB —
    // and until now it had a full progress/cancel/rejoin protocol and no surface
    // that could show it (roadmap D47). Polled, not streamed, so reopening the
    // TUI mid-conversion rejoins with no protocol on this side.
    AdmissionDashboard admission_dashboard(control_base_url_, control_api_token_);
    admission_dashboard.start(1000);
    auto admissions_renderer = admission_tab(admission_dashboard, admission_selected);

    // The served-model catalog: what users pick a model from. HTTP-only like the
    // three tabs above.
    ModelDashboard model_dashboard(control_base_url_, control_api_token_);
    model_dashboard.start(1000);
    auto models_renderer = model_tab(model_dashboard, model_selected);

    // Users, Chat and Voice: one user at a time, the same one on all three.
    // HTTP-only like every tab above (control/user_panels.hpp).
    UserDashboard user_dashboard(control_base_url_, control_api_token_);
    user_dashboard.start(1000);
    auto users_renderer = users_tab(user_dashboard);
    auto chat_renderer = chat_tab(user_dashboard, ui_shutting_down);
    auto voice_renderer = voice_tab(user_dashboard, audio_player);

    auto main_tabs = Container::Tab({nodes_renderer, models_renderer, activity_renderer,
        chat_renderer, users_renderer, performance_renderer, voice_renderer,
        soma_renderer, engines_renderer, admissions_renderer}, &tab_index);

    // Mockup-style header: the tab strip is real Button components (clickable +
    // focusable) rendered as `n Label` chips, the active one inverted.
    static const std::array<const char*, 10> kTabLabels = {
        "1 Nodes", "2 Models", "3 Activity", "4 Chat", "5 Users", "6 Performance", "7 Voice",
        "8 Soma", "9 Engines", "0 Admissions"};
    Components tab_buttons;
    for (int i = 0; i < static_cast<int>(kTabLabels.size()); ++i) {
        ButtonOption opt = ButtonOption::Simple();
        opt.transform = [&, i](const EntryState& s) -> Element {
            Element e = text(" " + std::string(kTabLabels[static_cast<size_t>(i)]) + " ");
            if (tab_index == i)  e = std::move(e) | bold | inverted;
            else if (s.focused)  e = std::move(e) | underlined;
            else                 e = std::move(e) | dim;
            return e;
        };
        tab_buttons.push_back(Button(kTabLabels[static_cast<size_t>(i)],
                                     [&, i] { tab_index = i; }, opt));
    }
    auto tab_btn_row   = Container::Horizontal(tab_buttons);
    auto top_container = Container::Vertical({tab_btn_row, main_tabs});

    auto top_renderer = Renderer(top_container, [&]() -> Element {
        auto ns = registry_.list_nodes();
        const int tot = static_cast<int>(ns.size());
        const int conn = static_cast<int>(std::count_if(
            ns.begin(), ns.end(), [](const NodeInfo& n) { return n.connected; }));
        const Color dot = (tot > 0 && conn == tot) ? Color::Green
                        : (conn > 0)               ? Color::Yellow
                                                   : Color::Red;
        auto header = hbox({
            text(" mantic-mind-control ") | bold,
            text("│ ") | dim,
            tab_btn_row->Render(),
            filler(),
            text(mm::tui::spinner_frame()) | color(Color::Green),
            text(" polling") | dim,
            text("  │  ") | dim,
            text("●") | color(dot),
            text(" " + std::to_string(conn) + "/" + std::to_string(tot) + " nodes"),
            text("  │  ") | dim,
            text(mm::tui::clock_hms()) | dim,
            text(" "),
        });
        auto footer = hbox({
            text(" 1-9,0 tabs · ↑/↓ select · click rows · q quit") | dim,
            filler(),
            text("mantic-mind · control ") | dim,
        });

        // An unconfigured cluster cannot place anything, and the operator needs
        // to know that from wherever they are rather than only from the tab
        // they have not opened. The banner drives them to the one place that
        // fixes it, and disappears the moment it is fixed.
        const auto engine_snap = engine_dashboard.snapshot();
        force_engine_setup = engine_snap.reachable && !engine_snap.configured;
        Elements shell{header, separator()};
        if (force_engine_setup && tab_index != 8) {
            shell.push_back(render_unconfigured_banner(engine_snap));
            shell.push_back(separator());
        }
        shell.push_back(main_tabs->Render() | flex);
        shell.push_back(separator());
        shell.push_back(footer);
        return vbox(shell) | border;
    });

    //

    auto root = MakeCatchEventAfter(top_renderer, [&](Event ev) {
        if (show_add_node || show_pin_entry) return false;
        if (ev == Event::Character('1')) { tab_index = 0; return true; }
        if (ev == Event::Character('2')) { tab_index = 1; return true; }
        if (ev == Event::Character('3')) { tab_index = 2; return true; }
        if (ev == Event::Character('4')) { tab_index = 3; return true; }
        if (ev == Event::Character('5')) { tab_index = 4; return true; }
        if (ev == Event::Character('6')) { tab_index = 5; return true; }
        if (ev == Event::Character('7')) { tab_index = 6; return true; }
        if (ev == Event::Character('8')) { tab_index = 7; return true; }
        if (ev == Event::Character('9')) { tab_index = 8; return true; }
        if (ev == Event::Character('0')) { tab_index = 9; return true; }
        if (ev == Event::Character('f') && tab_index == 0) {
            forget_selected_node();
            return true;
        }
        if (ev == Event::Escape || ev == Event::Character('q')) {
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    //

    auto with_add    = Modal(root,        modal_renderer,     &show_add_node);
    auto final_comp  = Modal(with_add,    pin_modal_renderer, &show_pin_entry);

    //

    // 250ms keeps the header spinner/clock moving without redrawing excessively;
    // renders are also triggered by refresh()/log events as before.
    std::atomic<bool> ticker_running{true};
    std::thread ticker([&] {
        while (ticker_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            screen.PostEvent(Event::Custom);
        }
    });

    // Run the loop under try/catch: an exception escaping a render lambda or event
    // handler (e.g. a SQLite throw) must not unwind past the still-joinable worker
    // threads below — a joinable std::thread destructor calls std::terminate, which
    // would skip terminal restoration. Catch, then always clean up.
    std::string loop_error;
    try {
        screen.Loop(final_comp);
    } catch (const std::exception& e) {
        loop_error = e.what();
    } catch (...) {
        loop_error = "unknown exception";
    }
    layout.set("nodes.detail_height", node_detail_height);
    layout.set("nodes.name_width", node_name_width);
    layout.save();

    // Clear the screen callbacks first so any late refresh()/quit() from a worker
    // thread during the joins is a no-op instead of posting to an exited screen.
    {
        std::lock_guard<std::mutex> lk(screen_mutex_);
        quit_fn_    = {};
        refresh_fn_ = {};
    }

    // Signal a chat stream in flight to abort, then join the ticker. The panels
    // join their own workers as they are destroyed, after this.
    ui_shutting_down = true;
    ticker_running = false;
    if (ticker.joinable())      ticker.join();

    if (!loop_error.empty()) {
        MM_ERROR("Control UI event loop terminated by exception: {}", loop_error);
    }
}

} // namespace mm
