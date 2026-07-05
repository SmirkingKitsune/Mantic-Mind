#include "control/control_ui.hpp"
#include "control/node_registry.hpp"
#include "control/agent_manager.hpp"
#include "control/agent_config_validator.hpp"
#include "common/agent.hpp"
#include "common/agent_db.hpp"
#include "common/http_client.hpp"
#include "common/models.hpp"
#include "common/logger.hpp"
#include "common/tool_executor.hpp"
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

// A vertical divider the user can drag left/right to resize an adjacent pane.
// It writes the new pane width (in columns) through `split_`, clamped to
// [lo_, hi_]. On mouse-press it grabs the mouse capture so drag continues even
// when the cursor leaves the 1-column bar; once dragging, it consumes Moved
// events regardless of position. The ◄/► buttons remain a keyboard fallback.
class DragDivider : public ftxui::ComponentBase {
public:
    DragDivider(int* split, int lo, int hi) : split_(split), lo_(lo), hi_(hi) {}

    ftxui::Element OnRender() override {
        using namespace ftxui;
        Element bar = dragging_ ? (separatorHeavy() | color(Color::Cyan))
                                : (separator() | color(Color::GrayDark));
        return bar | reflect(box_);
    }

    bool OnEvent(ftxui::Event event) override {
        using namespace ftxui;
        if (!event.is_mouse()) return false;
        auto& m = event.mouse();
        if (m.button == Mouse::Left && m.motion == Mouse::Pressed &&
            box_.Contain(m.x, m.y)) {
            captured_ = CaptureMouse(event);
            if (!captured_) return false;
            dragging_ = true;
            start_x_ = m.x;
            start_split_ = *split_;
            return true;
        }
        if (dragging_ && m.motion == Mouse::Moved) {
            *split_ = std::clamp(start_split_ + (m.x - start_x_), lo_, hi_);
            return true;
        }
        if (dragging_ && m.motion == Mouse::Released) {
            dragging_ = false;
            captured_.reset();
            return true;
        }
        return false;
    }

private:
    int* split_;
    int  lo_;
    int  hi_;
    ftxui::Box  box_{};
    bool dragging_ = false;
    int  start_x_ = 0;
    int  start_split_ = 0;
    ftxui::CapturedMouse captured_;
};

ftxui::Component MakeDragDivider(int* split, int lo, int hi) {
    return std::make_shared<DragDivider>(split, lo, hi);
}

} // namespace

ControlUI::ControlUI(NodeRegistry& registry,
                     AgentManager& agents,
                     std::string models_dir,
                     std::string control_base_url,
                     LocalChatFallback local_chat_fallback)
    : registry_(registry)
    , agents_(agents)
    , models_dir_(std::move(models_dir))
    , control_base_url_(std::move(control_base_url))
    , local_chat_fallback_(std::move(local_chat_fallback))
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

    //
    int  disc_sel      = 0;
    std::vector<std::string> disc_entries;

    //
    int  node_sel      = 0;
    std::vector<std::string> node_entries;

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
    bool show_agent_validation_modal = false;
    bool remember_pair_node = true;
    std::string add_url;
    std::string pin_input, pair_url, pair_nonce;

    // Agents tab — the editor is re-fielded to vLLM engine settings (1b).
    int  agent_sel    = 0;
    bool show_editor  = false;
    std::vector<AgentConfig> agent_rows;   // per-frame snapshot the menu transform indexes
    std::string ed_id, ed_id_orig, ed_name, ed_model, ed_sysprompt, ed_pref_node;
    // Sampling (generation settings still flow through RuntimeSettings in the request contract)
    std::string ed_temp_s{"0.70"}, ed_topp_s{"0.90"}, ed_max_s{"1024"};
    // Engine · vLLM
    std::string ed_mml_s{"4096"};      // max_model_len
    std::string ed_seqs_s{"16"};       // max_num_seqs
    std::string ed_batched_s{"-1"};    // max_num_batched_tokens
    std::string ed_tp_s{"1"};          // tensor_parallel_size
    std::string ed_pp_s{"1"};          // pipeline_parallel_size
    std::string ed_gpumem_s{"0.90"};   // gpu_memory_utilization
    std::string ed_dtype{"auto"};      // dtype
    std::string ed_quant;              // quantization
    std::string ed_toolparser;         // tool_call_parser
    std::string ed_extra_args_text;    // vllm_settings.extra_args (one per line)
    bool ed_reasoning{false}, ed_memories{true}, ed_tools{false};
    bool ed_sleep{true}, ed_prefix{true}, ed_trust{false}, ed_autotool{false};
    ModelCapabilityInfo ed_model_info;
    std::vector<ValidationIssue> ed_validation_issues;
    std::string ed_validation_signature;
    std::string agent_validation_title;
    std::vector<std::string> agent_validation_lines;
    std::vector<std::string> agent_entries;

    // Activity tab
    int log_filter = 0;
    std::vector<std::string> filter_labels = {"All", "Info", "Warn", "Error"};

    // Chat tab
    int chat_agent_sel = 0;
    std::vector<std::string> chat_agent_entries;
    std::vector<AgentConfig> chat_agent_rows;   // per-frame snapshot for the rich menu rows
    std::string chat_input;

    std::mutex              chat_mutex;
    std::deque<std::string> chat_transcript;
    std::string              chat_partial_assistant;
    std::string              chat_status{"idle"};
    std::string              chat_last_error;
    std::string              chat_last_conv_id;

    std::atomic<bool> chat_inflight{false};
    std::thread       chat_thread;

    // Set on shutdown so the chat stream callback can abort a long, never-timed-out
    // inference promptly (returning false aborts the cpp-httplib stream) instead of
    // making quit block on chat_thread.join().
    std::atomic<bool> ui_shutting_down{false};

    // Curation tab
    int cur_agent_sel = 0;
    int cur_conv_sel  = 0;
    int cur_mem_sel   = 0;
    std::vector<std::string> cur_agent_entries;
    std::vector<std::string> cur_conv_entries;
    std::vector<std::string> cur_mem_entries;
    std::string cur_new_title;
    std::string cur_start_s{"0"};
    std::string cur_end_s{"0"};
    std::string cur_context_before_s{"2"};
    bool cur_new_set_active{true};
    bool cur_new_use_parent{true};
    bool show_cur_delete_confirm{false};
    bool cur_delete_is_memory{false};
    std::string cur_delete_agent_id;
    std::string cur_delete_target_id;
    std::string cur_delete_target_label;
    std::mutex cur_status_mutex;
    std::string cur_status{"idle"};
    std::string cur_last_error;
    std::atomic<bool> cur_inflight{false};
    std::thread cur_thread;

    // Curation DB read cache. The curation render reloads conversations, memories,
    // and the full selected conversation from SQLite on every frame/keypress. Cache
    // them and refetch only on selection change, a short throttle, or an explicit
    // mutation (cur_cache_dirty) — so typing into the curation inputs no longer
    // triggers a full conversation reload per keystroke. The throttle also surfaces
    // external writes (API server / chat stream) within the window.
    std::string                            cur_cache_agent_id;
    std::string                            cur_detail_loaded_id;
    std::chrono::steady_clock::time_point  cur_cache_at{};
    std::vector<Conversation>              cur_convs_cache;
    std::vector<Memory>                    cur_mems_cache;
    std::optional<Conversation>            cur_conv_detail_cache;
    std::atomic<bool>                      cur_cache_dirty{false};

    // File browser
    bool show_file_browser = false;
    fs::path                  fb_current_path;
    std::vector<std::string>  fb_display_names;
    std::vector<fs::path>     fb_entry_paths;
    std::vector<bool>         fb_entry_is_dir;
    int fb_sel = 0;

    //

    auto mb_str = [](int64_t mb) -> std::string {
        if (mb >= 1024) {
            char b[32];
            snprintf(b, sizeof(b), "%.1fG", static_cast<double>(mb) / 1024.0);
            return b;
        }
        return std::to_string(mb) + "M";
    };

    auto trim_chat_transcript = [&]() {
        constexpr size_t kMaxLines = 400;
        while (chat_transcript.size() > kMaxLines) {
            chat_transcript.pop_front();
        }
    };

    auto parse_int_or = [](const std::string& s, int fallback) {
        try { return std::stoi(s); } catch (...) { return fallback; }
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

    auto set_curation_status = [&](const std::string& status, const std::string& error) {
        std::lock_guard<std::mutex> lk(cur_status_mutex);
        cur_status = status;
        cur_last_error = error;
    };

    auto run_curation_async = [&](const std::string& op_name,
                                  std::function<std::string()> fn) {
        if (cur_inflight.load()) return;
        if (cur_thread.joinable()) {
            cur_thread.join();
        }
        cur_inflight = true;
        set_curation_status("running: " + op_name, "");
        refresh();

        cur_thread = std::thread([&, op_name, fn = std::move(fn)]() mutable {
            std::string err = fn();
            if (err.empty()) set_curation_status("done: " + op_name, "");
            else             set_curation_status("failed", err);
            cur_cache_dirty = true;  // op may have changed the DB; force a refetch
            cur_inflight = false;
            refresh();
        });
    };

    //

    auto refresh_fb = [&]() {
        fb_display_names.clear();
        fb_entry_paths.clear();
        fb_entry_is_dir.clear();
        fb_sel = 0;

        if (fb_current_path.empty()) {
            //
            // Probe every possible drive letter; works on Windows without
            // needing <windows.h>, and safely produces nothing on Linux/macOS.
            for (char c = 'A'; c <= 'Z'; ++c) {
                std::error_code ec;
                fs::path drive(std::string(1, c) + ":\\");
                if (fs::exists(drive, ec) && !ec) {
                    fb_display_names.push_back(std::string(1, c) + ":\\");
                    fb_entry_paths.push_back(drive);
                    fb_entry_is_dir.push_back(true);
                }
            }
            // Fallback for non-Windows: always offer the filesystem root.
            if (fb_display_names.empty()) {
                fb_display_names.push_back("/");
                fb_entry_paths.push_back(fs::path("/"));
                fb_entry_is_dir.push_back(true);
            }
            return;
        }

        //
        bool at_root = (fb_current_path == fb_current_path.root_path());
        if (at_root) {
            //
            fb_display_names.push_back("[~] Drives");
            fb_entry_paths.push_back(fs::path{});   // empty = drives sentinel
            fb_entry_is_dir.push_back(true);
        } else {
            fb_display_names.push_back("[..] (parent)");
            fb_entry_paths.push_back(fb_current_path.parent_path());
            fb_entry_is_dir.push_back(true);
        }
        try {
            std::vector<std::pair<fs::path, bool>> raw;
            for (auto& e : fs::directory_iterator(
                     fb_current_path,
                     fs::directory_options::skip_permission_denied)) {
                bool d = e.is_directory();
                bool g = e.is_regular_file() && e.path().extension() == ".gguf";
                if (d || g) raw.push_back({e.path(), d});
            }
            std::sort(raw.begin(), raw.end(), [](auto& a, auto& b) {
                if (a.second != b.second) return a.second > b.second; // dirs first
                return a.first.filename().string() < b.first.filename().string();
            });
            for (auto& [p, d] : raw) {
                fb_display_names.push_back(d ? "[D] " + p.filename().string()
                                             :         p.filename().string());
                fb_entry_paths.push_back(p);
                fb_entry_is_dir.push_back(d);
            }
        } catch (...) {}
    };

    auto build_editor_cfg = [&]() {
        AgentConfig cfg;
        cfg.id = ed_id;
        cfg.name = ed_name;
        cfg.model_path = ed_model;
        cfg.system_prompt = ed_sysprompt;
        cfg.preferred_node_id = ed_pref_node;
        // Generation (shared request contract → RuntimeSettings)
        try { cfg.runtime_settings.temperature = std::stof(ed_temp_s); } catch (...) {}
        try { cfg.runtime_settings.top_p = std::stof(ed_topp_s); } catch (...) {}
        try { cfg.runtime_settings.max_tokens = std::stoi(ed_max_s); } catch (...) {}
        // Engine · vLLM
        try { cfg.vllm_settings.max_model_len = std::stoi(ed_mml_s); } catch (...) {}
        try { cfg.vllm_settings.max_num_seqs = std::stoi(ed_seqs_s); } catch (...) {}
        try { cfg.vllm_settings.max_num_batched_tokens = std::stoi(ed_batched_s); } catch (...) {}
        try { cfg.vllm_settings.tensor_parallel_size = std::stoi(ed_tp_s); } catch (...) {}
        try { cfg.vllm_settings.pipeline_parallel_size = std::stoi(ed_pp_s); } catch (...) {}
        try { cfg.vllm_settings.gpu_memory_utilization = std::stod(ed_gpumem_s); } catch (...) {}
        cfg.vllm_settings.dtype = ed_dtype.empty() ? "auto" : ed_dtype;
        cfg.vllm_settings.quantization = ed_quant;
        cfg.vllm_settings.tool_call_parser = ed_toolparser;
        cfg.vllm_settings.enable_sleep_mode = ed_sleep;
        cfg.vllm_settings.enable_prefix_caching = ed_prefix;
        cfg.vllm_settings.trust_remote_code = ed_trust;
        cfg.vllm_settings.enable_auto_tool_choice = ed_autotool;
        // Keep ctx_size aligned with max_model_len so the shared contract has a context.
        cfg.runtime_settings.ctx_size = cfg.vllm_settings.max_model_len;
        cfg.reasoning_enabled = ed_reasoning;
        cfg.memories_enabled = ed_memories;
        cfg.tools_enabled = ed_tools;

        cfg.vllm_settings.extra_args.clear();
        size_t start = 0;
        while (start <= ed_extra_args_text.size()) {
            size_t end = ed_extra_args_text.find('\n', start);
            std::string line = ed_extra_args_text.substr(
                start, end == std::string::npos ? std::string::npos : (end - start));
            line = util::trim(line);
            if (!line.empty()) cfg.vllm_settings.extra_args.push_back(line);
            if (end == std::string::npos) break;
            start = end + 1;
        }

        return cfg;
    };

    auto capability_status = [](bool supported,
                                bool metadata_found,
                                bool heuristic_only) -> std::string {
        if (supported) return heuristic_only && !metadata_found ? "yes (heuristic)" : "yes";
        return metadata_found ? "no" : "unknown";
    };

    auto refresh_editor_validation = [&]() {
        const std::string signature =
            ed_id + '\n' + ed_name + '\n' + ed_model + '\n' + ed_sysprompt + '\n' + ed_pref_node +
            '\n' + ed_temp_s + '\n' + ed_topp_s + '\n' + ed_max_s + '\n' +
            ed_mml_s + '\n' + ed_seqs_s + '\n' + ed_batched_s + '\n' + ed_tp_s + '\n' + ed_pp_s +
            '\n' + ed_gpumem_s + '\n' + ed_dtype + '\n' + ed_quant + '\n' + ed_toolparser + '\n' +
            ed_extra_args_text + '\n' +
            (ed_reasoning ? "1" : "0") + (ed_memories ? "1" : "0") + (ed_tools ? "1" : "0") +
            (ed_sleep ? "1" : "0") + (ed_prefix ? "1" : "0") + (ed_trust ? "1" : "0") +
            (ed_autotool ? "1" : "0");
        if (signature == ed_validation_signature) return;
        ed_validation_signature = signature;
        auto validation = validate_agent_config(build_editor_cfg(), &registry_, models_dir_, &ed_model_info);
        ed_validation_issues = std::move(validation.issues);
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

        std::string short_id = n.id;
        if (short_id.rfind("node-", 0) == 0) short_id = short_id.substr(5);

        Color hc = n.health == NodeHealthStatus::Healthy    ? Color::Green
                 : n.health == NodeHealthStatus::Degraded   ? Color::Yellow
                 : n.health == NodeHealthStatus::Unhealthy  ? Color::Red
                                                            : Color::GrayDark;
        Element health = text("● " + to_string(n.health)) | color(hc);

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
            mm::tui::col(text(short_id) | bold, 12), sep(),
            mm::tui::col(std::move(health), 11), sep(),
            mm::tui::col(std::move(cpu_cell), 12), sep(),
            mm::tui::col(std::move(gpu_cell), 12), sep(),
            mm::tui::col(std::move(vram_cell), 15), sep(),
            mm::tui::col(hbox(std::move(dots)), 12), sep(),
            mm::tui::col(text(arch) | dim, 8),
        });
        if (st.active) row = std::move(row) | inverted;
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
    auto nodes_section = Container::Vertical({node_menu_m, node_btns});

    auto nodes_comp  = Container::Vertical({disc_comp, nodes_section});

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

    auto agent_validation_ok = Button(" OK ", [&] {
        show_agent_validation_modal = false;
    }, ButtonOption::Simple());
    auto agent_validation_modal_comp = Container::Horizontal({agent_validation_ok});
    auto agent_validation_modal_renderer = Renderer(agent_validation_modal_comp, [&]() {
        Elements lines;
        lines.push_back(text(" " + agent_validation_title) | bold | hcenter);
        lines.push_back(separator());
        if (agent_validation_lines.empty()) {
            lines.push_back(paragraph(" Validation failed.") | color(Color::Red));
        } else {
            for (const auto& line : agent_validation_lines) {
                lines.push_back(paragraph(" - " + line) | color(Color::Red));
            }
        }
        lines.push_back(separator());
        lines.push_back(agent_validation_modal_comp->Render() | hcenter);
        return vbox(std::move(lines)) | border | size(WIDTH, EQUAL, 78);
    });

    //

    // Agent list — rich 1b rows (NAME | MODEL | R M T | NODE) via the menu transform.
    MenuOption agent_menu_opt = MenuOption::Vertical();
    agent_menu_opt.entries_option.transform = [&](const EntryState& st) -> Element {
        if (st.index < 0 || st.index >= static_cast<int>(agent_rows.size()))
            return text(st.label);
        const AgentConfig& a = agent_rows[static_cast<size_t>(st.index)];
        auto sep = [] { return text(" │ ") | dim; };
        auto flag = [](bool on, const char* ch) -> Element {
            return on ? (text(ch) | bold) : (text(ch) | dim);
        };
        Element flags = hbox({flag(a.reasoning_enabled, "R"), text(" "),
                              flag(a.memories_enabled, "M"), text(" "),
                              flag(a.tools_enabled, "T")});
        std::string node = a.preferred_node_id.empty() ? "auto" : a.preferred_node_id;
        if (node.rfind("node-", 0) == 0) node = node.substr(5);
        Element row = hbox({
            mm::tui::col(text(a.name) | bold, 14), sep(),
            mm::tui::col(text(mm::tui::short_model(a.model_path, 26)), 26), sep(),
            mm::tui::col(std::move(flags), 7), sep(),
            mm::tui::col(text(node) | (a.preferred_node_id.empty() ? dim : color(Color::Cyan)), 10),
        });
        if (st.active) row = std::move(row) | inverted;
        return row;
    };
    auto agent_menu   = Menu(&agent_entries, &agent_sel, agent_menu_opt);
    auto btn_new_a    = Button("[+] New", [&] {
        ed_id.clear(); ed_id_orig.clear(); ed_name = "New Agent"; ed_model.clear();
        ed_sysprompt.clear(); ed_pref_node.clear();
        ed_temp_s = "0.70"; ed_topp_s = "0.90"; ed_max_s = "1024";
        ed_mml_s = "4096"; ed_seqs_s = "16"; ed_batched_s = "-1";
        ed_tp_s = "1"; ed_pp_s = "1"; ed_gpumem_s = "0.90";
        ed_dtype = "auto"; ed_quant.clear(); ed_toolparser.clear();
        ed_extra_args_text.clear();
        ed_reasoning = false; ed_memories = true; ed_tools = false;
        ed_sleep = true; ed_prefix = true; ed_trust = false; ed_autotool = false;
        ed_model_info = {};
        ed_validation_issues.clear();
        ed_validation_signature.clear();
        show_editor = true;
    }, ButtonOption::Simple());
    auto btn_edit_a   = Button("[E] Edit", [&] {
        auto cs = agents_.list_agents();
        if (agent_sel >= 0 && agent_sel < static_cast<int>(cs.size())) {
            auto& c = cs[agent_sel];
            ed_id = c.id; ed_id_orig = c.id; ed_name = c.name; ed_model = c.model_path;
            ed_sysprompt = c.system_prompt; ed_pref_node = c.preferred_node_id;
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "%.2f", static_cast<double>(c.runtime_settings.temperature));
            ed_temp_s = tmp;
            snprintf(tmp, sizeof(tmp), "%.2f", static_cast<double>(c.runtime_settings.top_p));
            ed_topp_s = tmp;
            ed_max_s = std::to_string(c.runtime_settings.max_tokens);
            ed_mml_s = std::to_string(c.vllm_settings.max_model_len);
            ed_seqs_s = std::to_string(c.vllm_settings.max_num_seqs);
            ed_batched_s = std::to_string(c.vllm_settings.max_num_batched_tokens);
            ed_tp_s = std::to_string(c.vllm_settings.tensor_parallel_size);
            ed_pp_s = std::to_string(c.vllm_settings.pipeline_parallel_size);
            snprintf(tmp, sizeof(tmp), "%.2f", c.vllm_settings.gpu_memory_utilization);
            ed_gpumem_s = tmp;
            ed_dtype = c.vllm_settings.dtype.empty() ? "auto" : c.vllm_settings.dtype;
            ed_quant = c.vllm_settings.quantization;
            ed_toolparser = c.vllm_settings.tool_call_parser;
            ed_extra_args_text.clear();
            for (size_t i = 0; i < c.vllm_settings.extra_args.size(); ++i) {
                if (i > 0) ed_extra_args_text += '\n';
                ed_extra_args_text += c.vllm_settings.extra_args[i];
            }
            ed_sleep = c.vllm_settings.enable_sleep_mode;
            ed_prefix = c.vllm_settings.enable_prefix_caching;
            ed_trust = c.vllm_settings.trust_remote_code;
            ed_autotool = c.vllm_settings.enable_auto_tool_choice;
            ed_reasoning = c.reasoning_enabled;
            ed_memories  = c.memories_enabled;
            ed_tools     = c.tools_enabled;
            ed_model_info = {};
            ed_validation_issues.clear();
            ed_validation_signature.clear();
            show_editor  = true;
        }
    }, ButtonOption::Simple());
    auto btn_del_a    = Button("[-] Delete", [&] {
        auto cs = agents_.list_agents();
        if (agent_sel >= 0 && agent_sel < static_cast<int>(cs.size())) {
            agents_.delete_agent(cs[agent_sel].id);
            if (agent_sel >= static_cast<int>(cs.size()) - 1)
                agent_sel = std::max(0, static_cast<int>(cs.size()) - 2);
        }
    }, ButtonOption::Simple());
    auto agent_list_btns = Container::Horizontal({btn_new_a, btn_edit_a, btn_del_a});
    // Maybe wrapper: keep agent_menu out of the component tree when empty, matching
    // every other list menu, so FTXUI never indexes/focuses an empty entries vector.
    auto agent_menu_m = Maybe(agent_menu, [&]() { return !agent_entries.empty(); });
    auto agent_list_comp = Container::Vertical({agent_menu_m, agent_list_btns});

    // 1b editor: collapsible sections + resizable split.
    bool ed_open_sampling = false;   // section expanded?
    bool ed_open_engine   = true;    // vLLM engine is the focus of the re-field → open by default
    bool ed_open_caps     = false;
    int  ed_split         = 26;      // AGENTS context-list width (columns)

    // Editor inputs
    InputOption sl; sl.multiline = false;
    InputOption ml; ml.multiline = true;

    InputOption id_iopt; id_iopt.multiline = false;
    id_iopt.placeholder = "(leave blank to auto-generate)";
    auto ed_inp_id    = Input(&ed_id,        id_iopt);
    auto ed_inp_name  = Input(&ed_name,      sl);
    auto ed_inp_model = Input(&ed_model,     sl);
    auto ed_inp_sys   = Input(&ed_sysprompt, ml);
    auto ed_inp_pnode = Input(&ed_pref_node, sl);
    auto ed_inp_temp  = Input(&ed_temp_s,    sl);
    auto ed_inp_topp  = Input(&ed_topp_s,    sl);
    auto ed_inp_max   = Input(&ed_max_s,     sl);
    auto ed_inp_mml       = Input(&ed_mml_s,      sl);
    auto ed_inp_seqs      = Input(&ed_seqs_s,     sl);
    auto ed_inp_batched   = Input(&ed_batched_s,  sl);
    auto ed_inp_tp        = Input(&ed_tp_s,       sl);
    auto ed_inp_pp        = Input(&ed_pp_s,       sl);
    auto ed_inp_gpumem    = Input(&ed_gpumem_s,   sl);
    auto ed_inp_dtype     = Input(&ed_dtype,      sl);
    auto ed_inp_quant     = Input(&ed_quant,      sl);
    auto ed_inp_toolparser = Input(&ed_toolparser, sl);
    auto ed_inp_extra = Input(&ed_extra_args_text, ml);

    auto ed_cb_sleep     = Checkbox("sleep_mode",   &ed_sleep);
    auto ed_cb_prefix    = Checkbox("prefix_cache", &ed_prefix);
    auto ed_cb_trust     = Checkbox("trust_remote", &ed_trust);
    auto ed_cb_autotool  = Checkbox("auto_tool",    &ed_autotool);
    auto ed_cb_reasoning = Checkbox("Reasoning",    &ed_reasoning);
    auto ed_cb_memories  = Checkbox("Memories",     &ed_memories);
    auto ed_cb_tools     = Checkbox("Tools",        &ed_tools);

    auto btn_browse_model = Button("[Browse]", [&] {
        fs::path init;
        if (!ed_model.empty()) {
            fs::path mp(ed_model);
            auto par = mp.parent_path();
            init = (!par.empty() && fs::exists(par)) ? par : fs::current_path();
        } else {
            init = fs::current_path();
        }
        fb_current_path = init;
        refresh_fb();
        show_file_browser = true;
    }, ButtonOption::Simple());
    auto model_row = Container::Horizontal({ed_inp_model, btn_browse_model});

    auto btn_save_a   = Button(" Save ", [&] {
        refresh_editor_validation();
        std::vector<std::string> error_lines;
        for (const auto& issue : ed_validation_issues) {
            if (issue.severity != ValidationSeverity::Error) continue;
            error_lines.push_back(issue.field + ": " + issue.message);
        }
        if (!error_lines.empty()) {
            agent_validation_title = "Agent Configuration Error";
            agent_validation_lines = std::move(error_lines);
            show_agent_validation_modal = true;
            return;
        }

        AgentConfig cfg = build_editor_cfg();
        try {
            if (ed_id_orig.empty()) agents_.create_agent(cfg);
            else                    agents_.update_agent(ed_id_orig, cfg);
        } catch (const std::exception& e) {
            const std::string err = e.what();
            log(LogLevel::Error, std::string("Agent save failed: ") + err);
            agent_validation_title = "Agent Save Failed";
            agent_validation_lines = {err};
            const std::string err_low = util::to_lower(err);
            if (err_low.find("locked") != std::string::npos ||
                err_low.find("busy") != std::string::npos) {
                agent_validation_lines.push_back(
                    "Hint: another mantic-mind-control process may be using the same data directory. Stop duplicates and retry.");
            }
            show_agent_validation_modal = true;
            return;
        }
        show_editor = false;
    }, ButtonOption::Simple());
    auto btn_cancel_a = Button(" Cancel ", [&] { show_editor = false; },
                               ButtonOption::Simple());
    auto ed_form_btns = Container::Horizontal({btn_save_a, btn_cancel_a});

    // Resizable split controls.
    auto ed_split_dec = Button("◄", [&] { ed_split = std::max(18, ed_split - 3); }, ButtonOption::Simple());
    auto ed_split_inc = Button("►", [&] { ed_split = std::min(46, ed_split + 3); }, ButtonOption::Simple());
    auto ed_split_row = Container::Horizontal({ed_split_dec, ed_split_inc});
    auto ed_divider   = MakeDragDivider(&ed_split, 18, 46);

    // Collapsible section headers — clickable buttons whose transform draws the
    // ▾/▸ caret + title + (when collapsed) a live summary.
    auto section_header = [](bool* open, const char* title,
                             std::function<std::string()> summary) -> ButtonOption {
        ButtonOption o = ButtonOption::Simple();
        o.transform = [open, title, summary = std::move(summary)](const EntryState& s) -> Element {
            Element caret = text(*open ? "▾ " : "▸ ") | color(Color::Cyan);
            Element head = hbox({caret, text(title) | bold});
            if (!*open) head = hbox({std::move(head), text("   " + summary()) | dim});
            return s.focused ? (std::move(head) | inverted) : head;
        };
        return o;
    };
    auto ed_sec_sampling_btn = Button("Sampling", [&] { ed_open_sampling = !ed_open_sampling; },
        section_header(&ed_open_sampling, "Sampling",
            [&] { return "temp " + ed_temp_s + " · top_p " + ed_topp_s + " · " + ed_max_s + " tok"; }));
    auto ed_sec_engine_btn = Button("Engine", [&] { ed_open_engine = !ed_open_engine; },
        section_header(&ed_open_engine, "Engine · vLLM",
            [&] { return "mml " + ed_mml_s + " · seqs " + ed_seqs_s + " · gpu " + ed_gpumem_s +
                         " · tp" + ed_tp_s + "/pp" + ed_pp_s; }));
    auto ed_sec_caps_btn = Button("Caps", [&] { ed_open_caps = !ed_open_caps; },
        section_header(&ed_open_caps, "Capabilities & options",
            [&] { return std::string("tools ") + (ed_tools ? "y" : "n") + " · reason " +
                         (ed_reasoning ? "y" : "n") + " · mem " + (ed_memories ? "y" : "n"); }));

    // Section field groups, Maybe-gated so collapsed sections leave the focus tree.
    auto sampling_fields = Container::Vertical({ed_inp_temp, ed_inp_topp, ed_inp_max});
    auto sampling_m = Maybe(sampling_fields, [&] { return ed_open_sampling; });
    auto engine_fields = Container::Vertical({
        ed_inp_mml, ed_inp_seqs, ed_inp_batched, ed_inp_tp, ed_inp_pp,
        ed_inp_gpumem, ed_inp_dtype, ed_inp_quant, ed_inp_toolparser,
        ed_cb_sleep, ed_cb_prefix, ed_cb_trust, ed_cb_autotool});
    auto engine_m = Maybe(engine_fields, [&] { return ed_open_engine; });
    auto caps_fields = Container::Vertical({ed_cb_reasoning, ed_cb_memories, ed_cb_tools});
    auto caps_m = Maybe(caps_fields, [&] { return ed_open_caps; });

    auto editor_comp  = Container::Vertical({
        ed_divider, ed_split_row,
        ed_inp_id, ed_inp_name, model_row, ed_inp_pnode, ed_inp_sys,
        ed_sec_sampling_btn, sampling_m,
        ed_sec_engine_btn, engine_m,
        ed_sec_caps_btn, caps_m,
        ed_inp_extra,
        ed_form_btns
    });

    // Stacked: list (when !show_editor) or editor (when show_editor)
    auto agent_list_m = Maybe(agent_list_comp, [&]() { return !show_editor; });
    auto agent_edit_m = Maybe(editor_comp,     [&]() { return  show_editor; });
    auto agents_comp  = Container::Stacked({agent_list_m, agent_edit_m});

    //

    auto fb_menu   = Menu(&fb_display_names, &fb_sel, MenuOption::Vertical());
    auto fb_menu_m = Maybe(fb_menu, [&]() { return !fb_display_names.empty(); });

    auto fb_navigate = [&]() {
        if (fb_sel < 0 || fb_sel >= static_cast<int>(fb_entry_paths.size())) return;
        if (!fb_entry_is_dir[fb_sel]) return;
        fb_current_path = fb_entry_paths[fb_sel];
        refresh_fb();
    };

    auto fb_confirm = [&]() {
        if (fb_sel < 0 || fb_sel >= static_cast<int>(fb_entry_paths.size())) return;
        if (fb_entry_is_dir[fb_sel]) { fb_navigate(); return; }
        auto chosen = fb_entry_paths[fb_sel];
        ed_model = chosen.string();
        refresh_editor_validation();
        show_file_browser = false;
    };

    auto fb_btn_sel = Button(" Select ", [&]() { fb_confirm(); },            ButtonOption::Simple());
    auto fb_btn_can = Button(" Cancel ", [&]() { show_file_browser = false; }, ButtonOption::Simple());
    auto fb_btns    = Container::Horizontal({fb_btn_sel, fb_btn_can});

    auto fb_inner = CatchEvent(
        Container::Vertical({fb_menu_m, fb_btns}),
        [&](Event ev) {
            if (ev == Event::Escape) { show_file_browser = false; return true; }
            if (ev == Event::Return) {
                if (fb_sel >= 0 && fb_sel < static_cast<int>(fb_entry_is_dir.size())) {
                    if (fb_entry_is_dir[fb_sel]) { fb_navigate(); return true; }
                    fb_confirm();
                    return true;
                }
            }
            return false;
        });

    auto fb_renderer_comp = Renderer(fb_inner, [&]() {
        std::string path_label = fb_current_path.empty()
                                     ? "Drives"
                                     : fb_current_path.string();
        std::string empty_msg  = fb_current_path.empty()
                                     ? "  (no drives detected)"
                                     : "  (no .gguf files or subdirectories here)";
        return vbox({
            text(" Select Model File ") | bold | hcenter,
            separator(),
            text("  " + path_label) | color(Color::GrayDark),
            separator(),
            fb_display_names.empty()
                ? (text(empty_msg) | color(Color::GrayDark))
                : (fb_menu->Render() | size(HEIGHT, LESS_THAN, 20) | yframe),
            separator(),
            fb_btns->Render() | hcenter,
            text("  Enter: navigate/select  Esc: close") | color(Color::GrayDark),
        }) | border | size(WIDTH, EQUAL, 72);
    });

    //

    auto filter_toggle = Toggle(&filter_labels, &log_filter);
    auto activity_comp = Container::Vertical({filter_toggle});

    //

    InputOption chat_iopt;
    chat_iopt.multiline = false;
    chat_iopt.placeholder = "Type a prompt to test the selected agent";
    auto chat_input_comp = Input(&chat_input, chat_iopt);
    // Chat agent list — 1b NAME │ MODEL rows via the menu transform.
    MenuOption chat_menu_opt = MenuOption::Vertical();
    chat_menu_opt.entries_option.transform = [&](const EntryState& st) -> Element {
        if (st.index < 0 || st.index >= static_cast<int>(chat_agent_rows.size()))
            return text(st.label);
        const AgentConfig& a = chat_agent_rows[static_cast<size_t>(st.index)];
        Element row = hbox({
            mm::tui::col(text(a.name) | bold, 12),
            text(" │ ") | dim,
            mm::tui::col(text(mm::tui::short_model(a.model_path, 16)) | dim, 16),
        });
        if (st.active) row = std::move(row) | inverted;
        return row;
    };
    auto chat_agent_menu = Menu(&chat_agent_entries, &chat_agent_sel, chat_menu_opt);
    auto chat_agent_menu_m = Maybe(chat_agent_menu, [&]() { return !chat_agent_entries.empty(); });

    auto btn_chat_send = Button(" Send ", [&] {
        if (chat_inflight.load()) return;
        if (chat_input.empty()) return;

        auto chat_agents = agents_.list_agents();
        if (chat_agents.empty()) return;
        if (chat_agent_sel < 0 || chat_agent_sel >= static_cast<int>(chat_agents.size())) return;

        const std::string agent_id = chat_agents[chat_agent_sel].id;
        const std::string agent_name = chat_agents[chat_agent_sel].name;
        const std::string prompt = chat_input;
        chat_input.clear();

        if (chat_thread.joinable()) {
            chat_thread.join();
        }

        {
            std::lock_guard<std::mutex> lk(chat_mutex);
            chat_status = "sending";
            chat_last_error.clear();
            chat_last_conv_id.clear();
            chat_partial_assistant.clear();
            chat_transcript.push_back("User [" + agent_name + "]: " + prompt);
            trim_chat_transcript();
        }
        chat_inflight = true;
        refresh();

        chat_thread = std::thread([&, agent_id, prompt]() {
            bool done_seen = false;
            bool done_success = true;
            std::string done_error;
            std::string done_conv_id;
            std::string stream_partial;
            bool stream_payload_seen = false;
            int stream_status = 0;
            std::string stream_body;
            std::string used_base_url;
            std::vector<std::string> attempted_base_urls;

            auto path = "/v1/agents/" + agent_id + "/chat";
            auto body = nlohmann::json{{"message", prompt}};

            // Intentionally single-shot: one user prompt maps to one backend request.
            // Avoids duplicate turns when transient transport issues happen mid-stream.
            const std::string base_url = control_base_url_;

            auto stream_cb = [&](const std::string& data) -> bool {
                // Abort promptly on shutdown: returning false ends the cpp-httplib
                // stream so chat_thread.join() doesn't block on a long inference.
                if (ui_shutting_down.load()) return false;
                if (data == "[DONE]") return true;
                stream_payload_seen = true;

                nlohmann::json j;
                try {
                    j = nlohmann::json::parse(data);
                } catch (...) {
                    return true;
                }

                const std::string type = j.value("type", std::string{});
                if (type == "delta") {
                    stream_partial += j.value("content", std::string{});
                    std::lock_guard<std::mutex> lk(chat_mutex);
                    chat_status = "streaming";
                    chat_partial_assistant = stream_partial;
                } else if (type == "done") {
                    done_seen = true;
                    done_success = j.value("success", true);
                    done_conv_id = j.value("conv_id", std::string{});
                    done_error = j.value("error", std::string{});
                } else if (type == "error") {
                    done_error = j.value("message", std::string{});
                }

                refresh();
                return true;
            };

            bool stream_ok = false;
            if (!base_url.empty()) {
                attempted_base_urls.push_back(base_url);
                HttpClient cli(base_url);
                stream_ok = cli.stream_post(path, body, stream_cb, &stream_status, &stream_body);
                if (stream_ok) {
                    used_base_url = base_url;
                }
            }

            bool local_fallback_used = false;
            // Only use local fallback when there was no transport-level response
            // and no stream payload at all. This prevents duplicate turns when
            // a partial or completed stream already reached the UI.
            if (!stream_ok && stream_status == 0 && !stream_payload_seen && !done_seen && local_chat_fallback_) {
                local_fallback_used = true;
                std::string local_text;
                std::string local_conv_id;
                std::string local_error;

                {
                    std::lock_guard<std::mutex> lk(chat_mutex);
                    chat_status = "local fallback";
                }
                refresh();

                bool local_ok = local_chat_fallback_(
                    agent_id, prompt, &local_text, &local_conv_id, &local_error);
                stream_ok = true;
                done_seen = true;
                done_success = local_ok;
                done_conv_id = local_conv_id;
                done_error = local_error.empty() ? "local in-process chat failed" : local_error;
                if (!local_text.empty()) {
                    stream_partial = local_text;
                }
            }
            // If we saw a valid done event, accept completion even if stream_post
            // reported failure due a trailing transport hiccup.
            if (!stream_ok && done_seen && done_success) {
                stream_ok = true;
            }

            {
                std::lock_guard<std::mutex> lk(chat_mutex);

                if (!stream_partial.empty()) {
                    chat_transcript.push_back("Assistant: " + stream_partial);
                }
                chat_partial_assistant.clear();
                trim_chat_transcript();

                if (!stream_ok) {
                    chat_status = "failed";
                    if (stream_status == 0) {
                        chat_last_error = "stream request failed (could not connect to control API; attempted: ";
                        for (size_t i = 0; i < attempted_base_urls.size(); ++i) {
                            if (i > 0) chat_last_error += ", ";
                            chat_last_error += attempted_base_urls[i];
                        }
                        if (local_fallback_used) {
                            chat_last_error += ", in-process local fallback";
                        }
                        chat_last_error += ")";
                    } else {
                        std::string body_preview = util::trim(stream_body);
                        std::string detail_preview;
                        if (!body_preview.empty()) {
                            try {
                                auto bj = nlohmann::json::parse(body_preview);
                                std::string error = bj.value("error", std::string{});
                                std::string detail = bj.value("detail", std::string{});
                                std::string message = bj.value("message", std::string{});
                                if (!detail.empty() && !error.empty()) {
                                    detail_preview = error + ": " + detail;
                                } else if (!detail.empty()) {
                                    detail_preview = detail;
                                } else if (!message.empty()) {
                                    detail_preview = message;
                                } else if (!error.empty()) {
                                    detail_preview = error;
                                }
                            } catch (...) {
                            }
                        }
                        if (body_preview.size() > 240) {
                            body_preview = body_preview.substr(0, 240) + "...";
                        }
                        if (detail_preview.size() > 240) {
                            detail_preview = detail_preview.substr(0, 240) + "...";
                        }
                        chat_last_error = "HTTP " + std::to_string(stream_status);
                        if (!used_base_url.empty()) {
                            chat_last_error += " @ " + used_base_url;
                        }
                        if (!detail_preview.empty()) {
                            chat_last_error += ": " + detail_preview;
                        } else if (!body_preview.empty()) {
                            chat_last_error += ": " + body_preview;
                        } else {
                            chat_last_error += ": (empty response body; check control/node logs)";
                        }
                    }
                    chat_transcript.push_back("Error: " + chat_last_error);
                } else if (!done_seen) {
                    chat_status = "failed";
                    chat_last_error = "chat stream ended without done event";
                    chat_transcript.push_back("Error: " + chat_last_error);
                } else if (!done_success) {
                    chat_status = "failed";
                    chat_last_error = done_error.empty()
                        ? "control reported success=false"
                        : done_error;
                    chat_last_conv_id = done_conv_id;
                    chat_transcript.push_back("Error: " + chat_last_error);
                } else {
                    chat_status = "done";
                    chat_last_error.clear();
                    chat_last_conv_id = done_conv_id;
                    if (stream_partial.empty()) {
                        chat_transcript.push_back("Assistant: (no content)");
                    }
                }

                trim_chat_transcript();
            }

            chat_inflight = false;
            refresh();
        });
    }, ButtonOption::Simple());

    auto btn_chat_clear = Button(" Clear View ", [&] {
        std::lock_guard<std::mutex> lk(chat_mutex);
        chat_transcript.clear();
        chat_partial_assistant.clear();
        chat_last_error.clear();
        // Don't relabel an in-flight request; the worker thread owns chat_status
        // (e.g. "sending"/"streaming"/"local fallback") while a request is active.
        if (!chat_inflight.load()) chat_status = "idle";
    }, ButtonOption::Simple());

    auto chat_btn_row = Container::Horizontal({btn_chat_send, btn_chat_clear});
    auto chat_comp = Container::Vertical({chat_agent_menu_m, chat_input_comp, chat_btn_row});

    //

    auto cur_agent_menu = Menu(&cur_agent_entries, &cur_agent_sel, MenuOption::Vertical());

    // Conversations — active marker + title + token bar (1b memList style).
    MenuOption cur_conv_opt = MenuOption::Vertical();
    cur_conv_opt.entries_option.transform = [&](const EntryState& st) -> Element {
        if (st.index < 0 || st.index >= static_cast<int>(cur_convs_cache.size()))
            return text(st.label);
        const Conversation& c = cur_convs_cache[static_cast<size_t>(st.index)];
        int maxtok = 1;
        for (const auto& x : cur_convs_cache) maxtok = std::max(maxtok, x.total_tokens);
        float pct = static_cast<float>(c.total_tokens) / static_cast<float>(maxtok) * 100.0f;
        std::string title = c.title.empty() ? "(untitled)" : c.title;
        Element marker = c.is_active ? (text("*") | color(Color::Green) | bold) : text(" ");
        Element row = hbox({
            marker, text(" "),
            mm::tui::col(text(title), 18), text(" "),
            mm::tui::gauge_bar(pct, 10), text(" "),
            mm::tui::col_right(text(std::to_string(c.total_tokens) + " tok") | dim, 9),
        });
        if (st.active) row = std::move(row) | inverted;
        return row;
    };
    auto cur_conv_menu  = Menu(&cur_conv_entries, &cur_conv_sel, cur_conv_opt);

    // Memories — importance value + gauge + content (1b importance style).
    MenuOption cur_mem_opt = MenuOption::Vertical();
    cur_mem_opt.entries_option.transform = [&](const EntryState& st) -> Element {
        if (st.index < 0 || st.index >= static_cast<int>(cur_mems_cache.size()))
            return text(st.label);
        const Memory& m = cur_mems_cache[static_cast<size_t>(st.index)];
        float pct = std::clamp(m.importance, 0.0f, 1.0f) * 100.0f;
        Color ic = m.importance > 0.7f ? Color::Green
                 : m.importance > 0.5f ? Color::Yellow : Color::GrayDark;
        char imp[8];
        std::snprintf(imp, sizeof(imp), "%.2f", static_cast<double>(m.importance));
        Element row = hbox({
            text(imp) | color(ic) | bold, text(" "),
            mm::tui::gauge_bar(pct, 8), text("  "),
            paragraph(m.content) | flex,
        });
        if (st.active) row = std::move(row) | inverted;
        return row;
    };
    auto cur_mem_menu   = Menu(&cur_mem_entries, &cur_mem_sel, cur_mem_opt);
    auto cur_agent_menu_m = Maybe(cur_agent_menu, [&]() { return !cur_agent_entries.empty(); });
    auto cur_conv_menu_m  = Maybe(cur_conv_menu,  [&]() { return !cur_conv_entries.empty(); });
    auto cur_mem_menu_m   = Maybe(cur_mem_menu,   [&]() { return !cur_mem_entries.empty(); });

    InputOption cur_iopt;
    cur_iopt.multiline = false;
    cur_iopt.placeholder = "Conversation title";
    auto cur_new_title_input = Input(&cur_new_title, cur_iopt);

    InputOption cur_num_opt;
    cur_num_opt.multiline = false;
    auto cur_start_input   = Input(&cur_start_s, cur_num_opt);
    auto cur_end_input     = Input(&cur_end_s, cur_num_opt);
    auto cur_ctx_input     = Input(&cur_context_before_s, cur_num_opt);
    auto cur_new_active_cb = Checkbox("Set active", &cur_new_set_active);
    auto cur_new_parent_cb = Checkbox("Use selected as parent", &cur_new_use_parent);

    auto btn_cur_new_conv = Button(" New Conversation ", [&] {
        auto agents = agents_.list_agents();
        if (agents.empty() || cur_agent_sel < 0 || cur_agent_sel >= static_cast<int>(agents.size())) return;
        auto a = agents_.get_agent(agents[cur_agent_sel].id);
        if (!a) return;

        try {
            auto convs = a->db().list_conversations();
            ConvId parent_id;
            if (cur_new_use_parent && cur_conv_sel >= 0 && cur_conv_sel < static_cast<int>(convs.size())) {
                parent_id = convs[cur_conv_sel].id;
            }

            ConvId cid = a->db().create_conversation(cur_new_title, parent_id);
            if (cid.empty()) {
                set_curation_status("failed", "failed to create conversation");
                refresh();
                return;
            }
            if (cur_new_set_active) a->db().set_active_conversation(cid);
            cur_new_title.clear();
            cur_cache_dirty = true;
            set_curation_status("done: create conversation", "");
        } catch (const std::exception& e) {
            set_curation_status("failed", e.what());
        }
        refresh();
    }, ButtonOption::Simple());

    auto btn_cur_activate = Button(" Set Active ", [&] {
        auto agents = agents_.list_agents();
        if (agents.empty() || cur_agent_sel < 0 || cur_agent_sel >= static_cast<int>(agents.size())) return;
        auto a = agents_.get_agent(agents[cur_agent_sel].id);
        if (!a) return;
        try {
            auto convs = a->db().list_conversations();
            if (cur_conv_sel < 0 || cur_conv_sel >= static_cast<int>(convs.size())) return;
            a->db().set_active_conversation(convs[cur_conv_sel].id);
            cur_cache_dirty = true;
            set_curation_status("done: set active conversation", "");
        } catch (const std::exception& e) {
            set_curation_status("failed", e.what());
        }
        refresh();
    }, ButtonOption::Simple());

    auto btn_cur_compact = Button(" Force Compact ", [&] {
        auto agents = agents_.list_agents();
        if (agents.empty() || cur_agent_sel < 0 || cur_agent_sel >= static_cast<int>(agents.size())) return;
        const auto& agent_id = agents[cur_agent_sel].id;
        auto a = agents_.get_agent(agent_id);
        if (!a) return;
        std::string conv_id;
        try {
            auto convs = a->db().list_conversations();
            if (cur_conv_sel < 0 || cur_conv_sel >= static_cast<int>(convs.size())) return;
            conv_id = convs[cur_conv_sel].id;
        } catch (const std::exception& e) {
            set_curation_status("failed", e.what());
            refresh();
            return;
        }

        run_curation_async("force compact", [&, agent_id, conv_id]() -> std::string {
            HttpClient cli(control_base_url_);
            auto resp = cli.post("/v1/agents/" + agent_id + "/conversations/" + conv_id + "/compact",
                                 nlohmann::json::object());
            if (!resp.ok()) {
                return "HTTP " + std::to_string(resp.status) + ": " + parse_api_error(resp);
            }
            return {};
        });
    }, ButtonOption::Simple());

    auto btn_cur_delete_conv = Button(" Delete Conversation ", [&] {
        auto agents = agents_.list_agents();
        if (agents.empty() || cur_agent_sel < 0 || cur_agent_sel >= static_cast<int>(agents.size())) return;
        const auto& agent_id = agents[cur_agent_sel].id;
        auto a = agents_.get_agent(agent_id);
        if (!a) return;
        try {
            auto convs = a->db().list_conversations();
            if (cur_conv_sel < 0 || cur_conv_sel >= static_cast<int>(convs.size())) return;
            const auto& conv = convs[cur_conv_sel];
            if (conv.is_active) {
                set_curation_status("failed", "cannot delete active conversation; activate another first");
                refresh();
                return;
            }
            cur_delete_is_memory = false;
            cur_delete_agent_id = agent_id;
            cur_delete_target_id = conv.id;
            cur_delete_target_label = conv.title.empty() ? conv.id : conv.title;
            show_cur_delete_confirm = true;
        } catch (const std::exception& e) {
            set_curation_status("failed", e.what());
            refresh();
        }
    }, ButtonOption::Simple());

    auto btn_cur_extract = Button(" Generate Memories ", [&] {
        auto agents = agents_.list_agents();
        if (agents.empty() || cur_agent_sel < 0 || cur_agent_sel >= static_cast<int>(agents.size())) return;
        const auto& agent_id = agents[cur_agent_sel].id;
        auto a = agents_.get_agent(agent_id);
        if (!a) return;

        std::string conv_id;
        int msg_count = 0;
        try {
            auto convs = a->db().list_conversations();
            if (cur_conv_sel < 0 || cur_conv_sel >= static_cast<int>(convs.size())) return;
            auto conv = a->db().load_conversation(convs[cur_conv_sel].id);
            if (!conv) return;
            conv_id = conv->id;
            msg_count = static_cast<int>(conv->messages.size());
        } catch (const std::exception& e) {
            set_curation_status("failed", e.what());
            refresh();
            return;
        }

        int start_i = parse_int_or(cur_start_s, 0);
        int end_i = parse_int_or(cur_end_s, start_i);
        int ctx_i = std::clamp(parse_int_or(cur_context_before_s, 2), 0, 20);
        if (start_i < 0 || end_i < 0 || start_i > end_i || end_i >= msg_count) {
            set_curation_status("failed", "invalid message range");
            refresh();
            return;
        }

        run_curation_async("memory extraction", [&, agent_id, conv_id, start_i, end_i, ctx_i]() -> std::string {
            HttpClient cli(control_base_url_);
            auto resp = cli.post(
                "/v1/agents/" + agent_id + "/memories/extract",
                nlohmann::json{
                    {"conversation_id", conv_id},
                    {"start_index", start_i},
                    {"end_index", end_i},
                    {"context_before", ctx_i}
                });
            if (!resp.ok()) {
                return "HTTP " + std::to_string(resp.status) + ": " + parse_api_error(resp);
            }
            return {};
        });
    }, ButtonOption::Simple());

    auto btn_cur_delete_mem = Button(" Delete Memory ", [&] {
        auto agents = agents_.list_agents();
        if (agents.empty() || cur_agent_sel < 0 || cur_agent_sel >= static_cast<int>(agents.size())) return;
        const auto& agent_id = agents[cur_agent_sel].id;
        auto a = agents_.get_agent(agent_id);
        if (!a) return;
        try {
            auto mems = a->db().list_memories();
            if (cur_mem_sel < 0 || cur_mem_sel >= static_cast<int>(mems.size())) return;
            cur_delete_is_memory = true;
            cur_delete_agent_id = agent_id;
            cur_delete_target_id = mems[cur_mem_sel].id;
            cur_delete_target_label = mems[cur_mem_sel].content;
            if (cur_delete_target_label.size() > 80) {
                cur_delete_target_label = cur_delete_target_label.substr(0, 80) + "...";
            }
            show_cur_delete_confirm = true;
        } catch (const std::exception& e) {
            set_curation_status("failed", e.what());
            refresh();
        }
    }, ButtonOption::Simple());

    auto curation_comp = Container::Vertical({
        cur_agent_menu_m,
        cur_conv_menu_m,
        cur_mem_menu_m,
        cur_new_title_input,
        cur_start_input,
        cur_end_input,
        cur_ctx_input,
        cur_new_active_cb,
        cur_new_parent_cb,
        btn_cur_new_conv,
        btn_cur_activate,
        btn_cur_compact,
        btn_cur_delete_conv,
        btn_cur_extract,
        btn_cur_delete_mem
    });

    auto btn_cur_confirm_delete = Button(" Confirm Delete ", [&] {
        auto a = agents_.get_agent(cur_delete_agent_id);
        if (!a) {
            show_cur_delete_confirm = false;
            return;
        }

        try {
            if (cur_delete_is_memory) {
                a->db().delete_memory(cur_delete_target_id);
                cur_cache_dirty = true;
                set_curation_status("done: delete memory", "");
            } else {
                if (a->db().is_conversation_active(cur_delete_target_id)) {
                    set_curation_status("failed", "cannot delete active conversation; activate another first");
                } else {
                    a->db().delete_conversation(cur_delete_target_id);
                    cur_cache_dirty = true;
                    set_curation_status("done: delete conversation", "");
                }
            }
        } catch (const std::exception& e) {
            set_curation_status("failed", e.what());
        }
        show_cur_delete_confirm = false;
        refresh();
    }, ButtonOption::Simple());
    auto btn_cur_cancel_delete = Button(" Cancel ", [&] {
        show_cur_delete_confirm = false;
    }, ButtonOption::Simple());
    auto cur_delete_modal_comp = Container::Horizontal({btn_cur_confirm_delete, btn_cur_cancel_delete});
    auto cur_delete_modal_renderer = Renderer(cur_delete_modal_comp, [&]() {
        return vbox({
            text(" Confirm Deletion ") | bold | hcenter,
            separator(),
            text(cur_delete_is_memory ? " Delete memory?" : " Delete conversation?"),
            text(" " + cur_delete_target_label) | color(Color::Yellow),
            separator(),
            cur_delete_modal_comp->Render() | hcenter,
        }) | border | size(WIDTH, EQUAL, 60);
    });

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
        for (size_t i = 0; i < node_rows.size(); ++i) {
            std::string sid = node_rows[i].id;
            if (sid.rfind("node-", 0) == 0) sid = sid.substr(5);
            node_entries[i] = sid;  // fallback; the transform renders the real row
        }
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
                placed += s.agent_ids.size();
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
            col(text("NODE"), 12), sep(),
            col(text("HEALTH"), 11), sep(),
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
            detail_title = "DETAIL · " + n.id;
            detail_cap = n.url;
            // Per-node CPU/GPU/VRAM live in the NODES table row (sparklines/gauge)
            // and the CLUSTER tile; matching 1a, the detail shows the slot table +
            // capabilities only — no duplicate gauges here.
            Elements rows = {
                hbox({text("saved ") | dim, text(n.remembered ? "yes" : "no"),
                      text("   platform ") | dim, text(n.platform.empty() ? "—" : n.platform),
                      text("   model ") | dim,
                      text(n.loaded_model.empty() ? "(engine-managed)" : n.loaded_model)}),
                text(""),
                slot_table(n.slots, true),
                text(""),
            };
            char bud[48];
            std::snprintf(bud, sizeof(bud), "%.2f/%.2f", n.vllm_gpu_fraction_used, n.vllm_gpu_budget);
            std::vector<std::string> caps = {
                "arch " + (n.capabilities.arch.empty() ? std::string("—") : n.capabilities.arch),
                std::to_string(n.capabilities.gpu_count) + " GPU",
                "vLLM " + (n.capabilities.vllm_version.empty() ? std::string("—") : n.capabilities.vllm_version),
                std::string("ray ") + (n.capabilities.supports_ray ? "yes" : "no"),
                n.capabilities.comm_backends.empty() ? std::string("comm —")
                                                     : util::join(n.capabilities.comm_backends, ","),
                std::string("budget ") + bud,
            };
            rows.push_back(text("  " + util::join(caps, "  ·  ")) | dim);
            detail_body = vbox(std::move(rows));
        } else {
            detail_body = text("  No connected nodes.") | dim;
        }
        Element detail_panel = panel(detail_title, detail_cap, std::move(detail_body)) |
                               size(HEIGHT, EQUAL, 13);

        return vbox({
            top_strip,
            table_panel,
            detail_panel,
        });
    });

    // Tab 2 — 1b "Commander" agent editor
    auto agents_renderer = Renderer(agents_comp, [&]() {
        using mm::tui::panel;
        using mm::tui::col;

        auto cs = agents_.list_agents();
        agent_rows = cs;                    // snapshot for the rich menu transform
        agent_entries.resize(cs.size());
        for (size_t i = 0; i < cs.size(); ++i)
            agent_entries[i] = cs[i].name;  // fallback; the transform renders the real row
        if (agent_sel >= static_cast<int>(cs.size()) && !cs.empty())
            agent_sel = static_cast<int>(cs.size()) - 1;
        else if (cs.empty())
            agent_sel = 0;

        auto fmt2 = [](double v) { char b[16]; std::snprintf(b, sizeof(b), "%.2f", v); return std::string(b); };
        auto field = [](const std::string& label, Element input, int label_w = 15) -> Element {
            return hbox({text(label) | dim | size(WIDTH, EQUAL, label_w), std::move(input)});
        };

        if (show_editor) {
            refresh_editor_validation();

            Elements validation;
            bool has_issues = false;
            for (const auto& issue : ed_validation_issues) {
                const bool err = issue.severity == ValidationSeverity::Error;
                validation.push_back(
                    paragraph(std::string(err ? "✗ " : "⚠ ") + issue.field + ": " + issue.message) |
                    color(err ? Color::Red : Color::Yellow));
                has_issues = true;
            }
            if (!has_issues) validation.push_back(text("✓ configuration valid") | color(Color::Green));

            std::vector<ToolDefinition> local_tools;
            if (ed_tools && ed_memories) local_tools = ToolExecutor::local_tool_catalog();
            Elements tool_lines;
            if (!ed_tools) tool_lines.push_back(text("tools disabled") | dim);
            else if (!ed_memories)
                tool_lines.push_back(paragraph("no executable tools until Memories is enabled") | color(Color::Yellow));
            else
                for (const auto& t : local_tools) {
                    tool_lines.push_back(text("· " + t.name) | color(Color::Cyan));
                    tool_lines.push_back(paragraph("   " + t.description) | dim);
                }

            // Left context list — highlights the agent being edited.
            Elements list_rows;
            for (const auto& a : cs) {
                if (a.id == ed_id_orig && !ed_id_orig.empty())
                    list_rows.push_back((text("▌ " + a.name) | bold) | inverted);
                else
                    list_rows.push_back(text("  " + a.name) | dim);
            }
            if (list_rows.empty()) list_rows.push_back(text("  (new agent)") | dim);
            Element left = panel("AGENTS", vbox(std::move(list_rows)) | yframe | flex) |
                           size(WIDTH, EQUAL, ed_split);

            // Flat sections (1b sec() style): a header line + an indented body.
            auto sec_static = [](const std::string& title, Element body) -> Element {
                return vbox({
                    text("  " + title) | bold | color(Color::Cyan),
                    hbox({text("  "), std::move(body) | flex}),
                    text(""),
                });
            };
            auto sec_toggle = [](Element header, bool open, Element body) -> Element {
                Elements v = {std::move(header)};
                if (open) v.push_back(hbox({text("  "), std::move(body) | flex}));
                v.push_back(text(""));
                return vbox(std::move(v));
            };

            Element sections = vbox({
                sec_static("Identity", vbox({
                    field(" id", ed_inp_id->Render() | flex),
                    field(" name", ed_inp_name->Render() | flex),
                })),
                sec_static("Model & placement", vbox({
                    hbox({text(" model_path") | dim | size(WIDTH, EQUAL, 15),
                          ed_inp_model->Render() | flex, text(" "), btn_browse_model->Render()}),
                    field(" preferred_node", ed_inp_pnode->Render() | flex),
                })),
                sec_static("System prompt",
                           ed_inp_sys->Render() | size(HEIGHT, LESS_THAN, 5) | border),
                sec_toggle(ed_sec_sampling_btn->Render(), ed_open_sampling, hbox({
                    field(" temperature", ed_inp_temp->Render() | flex, 13),
                    field(" top_p", ed_inp_topp->Render() | flex, 8),
                    field(" max_tokens", ed_inp_max->Render() | flex, 12),
                })),
                sec_toggle(ed_sec_engine_btn->Render(), ed_open_engine, vbox({
                    hbox({field(" max_model_len", ed_inp_mml->Render() | flex),
                          field(" max_num_seqs", ed_inp_seqs->Render() | flex)}),
                    hbox({field(" gpu_mem_util", ed_inp_gpumem->Render() | flex),
                          field(" max_batched", ed_inp_batched->Render() | flex)}),
                    hbox({field(" tensor_par", ed_inp_tp->Render() | flex),
                          field(" pipeline_par", ed_inp_pp->Render() | flex)}),
                    hbox({field(" dtype", ed_inp_dtype->Render() | flex),
                          field(" quantization", ed_inp_quant->Render() | flex)}),
                    field(" tool_call_parser", ed_inp_toolparser->Render() | flex),
                    hbox({ed_cb_sleep->Render(), text("  "), ed_cb_prefix->Render(), text("  "),
                          ed_cb_trust->Render(), text("  "), ed_cb_autotool->Render()}),
                })),
                sec_toggle(ed_sec_caps_btn->Render(), ed_open_caps, vbox({
                    hbox({ed_cb_reasoning->Render(), text("   "), ed_cb_memories->Render(), text("   "),
                          ed_cb_tools->Render()}),
                    text("context limit: " +
                         (ed_model_info.n_ctx_train > 0
                              ? std::to_string(ed_model_info.n_ctx_train) + " tokens"
                              : "unknown")) | dim,
                    text("tool calling: " + capability_status(ed_model_info.supports_tool_calls,
                                                              ed_model_info.metadata_found,
                                                              ed_model_info.used_filename_heuristics)) | dim,
                    text("reasoning: " + capability_status(ed_model_info.supports_reasoning,
                                                           ed_model_info.metadata_found,
                                                           ed_model_info.used_filename_heuristics)) | dim,
                    vbox(std::move(tool_lines)) | size(HEIGHT, LESS_THAN, 6) | yframe,
                })),
                sec_static("Extra vLLM args (one per line)",
                           ed_inp_extra->Render() | size(HEIGHT, LESS_THAN, 4) | border),
                sec_static("Validation", vbox(std::move(validation))),
                hbox({ed_form_btns->Render(), text("   Esc cancel · click ▸ to expand") | dim}),
            });

            Element edit_body = vbox({
                hbox({text(" split ") | dim, ed_split_dec->Render(), text(" "), ed_split_inc->Render(),
                      filler(), text("◄ ► or drag the divider ") | dim}),
                separator(),
                sections | yframe | flex,
            });
            Element right = panel("EDIT · " + (ed_name.empty() ? "(new)" : ed_name), edit_body) | flex;
            return hbox({left, ed_divider->Render(), right});
        }

        // ── List view (1b) ──────────────────────────────────────────────
        auto sep = [] { return text(" │ ") | dim; };
        Element head = hbox({
            col(text("NAME"), 14), sep(),
            col(text("MODEL"), 26), sep(),
            col(text("R M T"), 7), sep(),
            col(text("NODE"), 10),
        }) | dim | bold;
        Element list_body = cs.empty()
            ? (text("  No agents. Press [+] New to create one.") | dim | flex)
            : (agent_menu->Render() | yframe | flex);
        Element list_panel = panel("AGENTS", std::to_string(cs.size()) + " total", vbox({
            head, separator(), list_body, agent_list_btns->Render(),
        })) | flex;

        Element detail_panel;
        if (!cs.empty() && agent_sel >= 0 && agent_sel < static_cast<int>(cs.size())) {
            const auto& a = cs[static_cast<size_t>(agent_sel)];
            auto feat = [](bool v, const std::string& label) {
                return hbox({text(label + " ") | dim,
                             text(v ? "yes" : "no") | color(v ? Color::Green : Color::GrayDark)});
            };
            detail_panel = panel("AGENT · " + a.name, "POST /v1/agents/" + a.id + "/chat", vbox({
                hbox({text("model    ") | dim,
                      text(a.model_path.empty() ? "(none)" : a.model_path) | bold}),
                hbox({text("system   ") | dim,
                      paragraph(a.system_prompt.empty() ? "(none)" : a.system_prompt) |
                          color(Color::Cyan) | flex}),
                hbox({text("sampling ") | dim,
                      text("temp " + fmt2(a.runtime_settings.temperature) + " · top_p " +
                           fmt2(a.runtime_settings.top_p) + " · " +
                           std::to_string(a.runtime_settings.max_tokens) + " tok")}),
                hbox({text("engine   ") | dim,
                      text("mml " + std::to_string(a.vllm_settings.max_model_len) + " · seqs " +
                           std::to_string(a.vllm_settings.max_num_seqs) + " · gpu " +
                           fmt2(a.vllm_settings.gpu_memory_utilization) + " · tp" +
                           std::to_string(a.vllm_settings.tensor_parallel_size) + "/pp" +
                           std::to_string(a.vllm_settings.pipeline_parallel_size) +
                           (a.vllm_settings.enable_sleep_mode ? " · sleep" : ""))}),
                hbox({text("caps     ") | dim, feat(a.reasoning_enabled, "reasoning"), text("  "),
                      feat(a.memories_enabled, "memories"), text("  "),
                      feat(a.tools_enabled, "tools")}),
            })) | size(HEIGHT, EQUAL, 9);
        } else {
            detail_panel = panel("AGENT", text("  (no agents)") | dim) | size(HEIGHT, EQUAL, 9);
        }

        return vbox({list_panel, detail_panel});
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

    // Tab 4
    auto chat_renderer = Renderer(chat_comp, [&]() {
        auto cs = agents_.list_agents();

        chat_agent_rows = cs;               // snapshot for the rich menu transform
        chat_agent_entries.resize(cs.size());
        for (size_t i = 0; i < cs.size(); ++i)
            chat_agent_entries[i] = cs[i].name;  // fallback; the transform renders the row
        if (chat_agent_sel >= static_cast<int>(cs.size()) && !cs.empty())
            chat_agent_sel = static_cast<int>(cs.size()) - 1;
        else if (cs.empty())
            chat_agent_sel = 0;

        std::string selected_agent = "(none)";
        if (!cs.empty() && chat_agent_sel >= 0 && chat_agent_sel < static_cast<int>(cs.size()))
            selected_agent = cs[chat_agent_sel].name + " [" + cs[chat_agent_sel].id + "]";

        auto nodes = registry_.list_nodes();
        const size_t connected_nodes = std::count_if(
            nodes.begin(), nodes.end(), [](const NodeInfo& n) { return n.connected; });

        // Build transcript Elements directly under the lock instead of deep-copying
        // the whole transcript into an intermediate vector first.
        Elements transcript_lines;
        std::string partial_snapshot;
        std::string status_snapshot;
        std::string error_snapshot;
        std::string conv_snapshot;
        {
            std::lock_guard<std::mutex> lk(chat_mutex);
            transcript_lines.reserve(chat_transcript.size() + 1);
            for (const auto& line : chat_transcript) {
                transcript_lines.push_back(text("  " + line));
            }
            partial_snapshot = chat_partial_assistant;
            status_snapshot = chat_status;
            error_snapshot = chat_last_error;
            conv_snapshot = chat_last_conv_id;
        }
        if (chat_inflight.load()) {
            if (partial_snapshot.empty()) {
                transcript_lines.push_back(text("  Assistant: (streaming...)") | color(Color::Cyan));
            } else {
                transcript_lines.push_back(
                    text("  Assistant (streaming): " + partial_snapshot) | color(Color::Cyan));
            }
        }
        if (transcript_lines.empty()) {
            transcript_lines.push_back(text("  (no chat yet)") | color(Color::GrayDark));
        }

        using mm::tui::panel;
        Color status_color = Color::GrayDark;
        if (status_snapshot == "done") status_color = Color::Green;
        if (status_snapshot == "streaming" || status_snapshot == "sending") status_color = Color::Yellow;
        if (status_snapshot == "failed") status_color = Color::Red;

        Element header = hbox({
            text(" nodes ") | dim, text(std::to_string(connected_nodes)),
            text("  ·  ") | dim, text("agent ") | dim, text(selected_agent) | bold,
            text("  ·  ") | dim, text("status ") | dim, text(status_snapshot) | color(status_color),
            filler(),
            error_snapshot.empty() ? text("") : (text("error: " + error_snapshot + " ") | color(Color::Red)),
        });

        Element chat_head = hbox({
            mm::tui::col(text("NAME"), 12), text(" │ ") | dim, text("MODEL"),
        }) | dim | bold;
        Element left = panel("AGENT", vbox({
            chat_head,
            separator(),
            cs.empty() ? (text("  No agents configured.") | dim)
                       : (chat_agent_menu->Render() | yframe | flex),
            separator(),
            text(" prompt") | dim,
            chat_input_comp->Render() | border,
            chat_btn_row->Render(),
            text(chat_inflight.load() ? "  send blocked while streaming" : "  streams SSE") | dim,
        })) | size(WIDTH, EQUAL, 42);

        Element right = panel("TRANSCRIPT", conv_snapshot,
                              vbox(std::move(transcript_lines)) | yframe | flex) | flex;

        return vbox({header, hbox({left, text(" "), right}) | flex});
    });

    // Tab 5
    auto curation_renderer = Renderer(curation_comp, [&]() {
        auto cs = agents_.list_agents();
        cur_agent_entries.resize(cs.size());
        for (size_t i = 0; i < cs.size(); ++i) {
            cur_agent_entries[i] = " " + cs[i].name;
        }
        if (cur_agent_sel >= static_cast<int>(cs.size()) && !cs.empty())
            cur_agent_sel = static_cast<int>(cs.size()) - 1;
        else if (cs.empty())
            cur_agent_sel = 0;

        std::string selected_agent = "(none)";
        std::string cur_db_error;  // set if a DB read throws this frame
        const std::string cur_agent_id =
            (!cs.empty() && cur_agent_sel >= 0 && cur_agent_sel < static_cast<int>(cs.size()))
                ? cs[cur_agent_sel].id
                : std::string{};
        if (!cur_agent_id.empty())
            selected_agent = cs[cur_agent_sel].name;

        // Fetch the selected agent at most once per render, and only when actually
        // needed — a cache-hit frame fetches nothing. Replaces the previous two
        // get_agent() lookups per render.
        std::shared_ptr<Agent> frame_agent;
        bool frame_agent_fetched = false;
        auto curation_agent = [&]() -> const std::shared_ptr<Agent>& {
            if (!frame_agent_fetched) {
                frame_agent_fetched = true;
                if (!cur_agent_id.empty()) frame_agent = agents_.get_agent(cur_agent_id);
            }
            return frame_agent;
        };

        // Refetch conversations + memories from SQLite only when the agent selection
        // changes, a mutation flagged the cache dirty, or the throttle elapses — this
        // keeps the per-keystroke/per-frame render off the database. Reads run on the
        // loop thread; a locked/busy DB throws, so catch and degrade rather than
        // unwind out of Screen::Loop(). Do NOT log() here — log() calls refresh(),
        // which would spin a redraw loop while the DB stays locked.
        const auto cur_now = std::chrono::steady_clock::now();
        const bool cur_force = cur_cache_dirty.exchange(false);
        if (cur_force || cur_agent_id != cur_cache_agent_id ||
            cur_now - cur_cache_at > std::chrono::milliseconds(1000)) {
            cur_cache_agent_id = cur_agent_id;
            cur_cache_at = cur_now;
            cur_convs_cache.clear();
            cur_mems_cache.clear();
            cur_conv_detail_cache.reset();
            cur_detail_loaded_id.clear();
            if (!cur_agent_id.empty()) {
                if (auto a = curation_agent()) {
                    try {
                        cur_convs_cache = a->db().list_conversations();
                        cur_mems_cache  = a->db().list_memories();
                    } catch (const std::exception& e) {
                        cur_convs_cache.clear();
                        cur_mems_cache.clear();
                        cur_db_error = e.what();
                    }
                }
            }
        }
        const std::vector<Conversation>& convs = cur_convs_cache;
        const std::vector<Memory>&       mems  = cur_mems_cache;

        cur_conv_entries.resize(convs.size());
        for (size_t i = 0; i < convs.size(); ++i) {
            const auto& c = convs[i];
            std::string title = c.title.empty() ? "(untitled)" : c.title;
            if (title.size() > 28) title = mm::util::utf8_truncate(title, 28) + "...";
            std::string prefix = c.is_active ? "* " : "  ";
            std::string parent = c.parent_conv_id.empty() ? "" : (" <- " + c.parent_conv_id.substr(0, 8));
            cur_conv_entries[i] = prefix + title + " [" + std::to_string(c.total_tokens) + " tok]" + parent;
        }
        if (cur_conv_sel >= static_cast<int>(convs.size()) && !convs.empty())
            cur_conv_sel = static_cast<int>(convs.size()) - 1;
        else if (convs.empty())
            cur_conv_sel = 0;

        // Load the selected conversation's full detail, refetching only when the
        // selected conversation id changes (the throttle above already reset the
        // cached detail + loaded id when it elapsed or the agent changed).
        if (!convs.empty() && cur_conv_sel >= 0 && cur_conv_sel < static_cast<int>(convs.size())) {
            const std::string want_id = convs[cur_conv_sel].id;
            if (want_id != cur_detail_loaded_id) {
                cur_conv_detail_cache.reset();
                if (auto a = curation_agent()) {
                    try {
                        cur_conv_detail_cache = a->db().load_conversation(want_id);
                        cur_detail_loaded_id = want_id;
                    } catch (const std::exception& e) {
                        cur_conv_detail_cache.reset();
                        cur_detail_loaded_id.clear();
                        if (cur_db_error.empty()) cur_db_error = e.what();
                    }
                }
            }
        } else {
            cur_conv_detail_cache.reset();
            cur_detail_loaded_id.clear();
        }
        const std::optional<Conversation>& conv_detail = cur_conv_detail_cache;

        cur_mem_entries.resize(mems.size());
        for (size_t i = 0; i < mems.size(); ++i) {
            cur_mem_entries[i] = mems[i].content;  // fallback; the transform renders the row
        }
        if (cur_mem_sel >= static_cast<int>(mems.size()) && !mems.empty())
            cur_mem_sel = static_cast<int>(mems.size()) - 1;
        else if (mems.empty())
            cur_mem_sel = 0;

        std::string status_snapshot;
        std::string error_snapshot;
        {
            std::lock_guard<std::mutex> lk(cur_status_mutex);
            status_snapshot = cur_status;
            error_snapshot = cur_last_error;
        }

        // ── 1b layout: three columns + a compact DETAIL/actions bar ─────
        using mm::tui::panel;
        using mm::tui::col;

        Element c_agents = panel("AGENTS", vbox({
            text(" NAME") | dim | bold,
            separator(),
            cs.empty() ? (text("  (none)") | dim)
                       : (cur_agent_menu->Render() | yframe | flex),
        })) | size(WIDTH, EQUAL, 20);

        Element conv_head = hbox({
            text("  "), col(text("TITLE"), 19), text("TOKENS"),
        }) | dim | bold;
        Element c_convs = panel("CONVERSATIONS", std::to_string(convs.size()), vbox({
            conv_head,
            separator(),
            convs.empty() ? (text("  No conversations.") | dim)
                          : (cur_conv_menu->Render() | yframe | flex),
        })) | size(WIDTH, EQUAL, 48);

        Element mem_head = hbox({
            col(text("IMP"), 15), text("CONTENT"),
        }) | dim | bold;
        Element c_mems = panel("MEMORIES · importance", std::to_string(mems.size()), vbox({
            mem_head,
            separator(),
            mems.empty() ? (text("  No memories.") | dim)
                         : (cur_mem_menu->Render() | yframe | flex),
        })) | flex;

        // DETAIL bar: conversation summary · extract range · actions · new conversation.
        Elements dl;
        if (!convs.empty() && cur_conv_sel >= 0 && cur_conv_sel < static_cast<int>(convs.size())) {
            const auto& c = convs[static_cast<size_t>(cur_conv_sel)];
            const std::string title = c.title.empty() ? "(untitled)" : c.title;
            const int msgs = conv_detail ? static_cast<int>(conv_detail->messages.size()) : 0;
            Elements sum = {
                text("conversation ") | dim, text(title) | bold,
                text(" · " + std::to_string(c.total_tokens) + " tokens · ") | dim,
                c.is_active ? (text("active") | color(Color::Green)) : (text("inactive") | dim),
                text(" · " + std::to_string(msgs) + " messages") | dim,
            };
            if (msgs > 0)
                sum.push_back(text("   ·   extract range 0–" + std::to_string(msgs - 1)) |
                              color(Color::Cyan));
            dl.push_back(hbox(std::move(sum)));
        } else {
            dl.push_back(text("(no conversation selected)") | dim);
        }
        dl.push_back(hbox({
            text("extract ") | dim,
            text("start ") | dim, cur_start_input->Render() | size(WIDTH, EQUAL, 6),
            text(" end ") | dim, cur_end_input->Render() | size(WIDTH, EQUAL, 6),
            text(" context ") | dim, cur_ctx_input->Render() | size(WIDTH, EQUAL, 6),
            text(" "), btn_cur_extract->Render(),
            filler(),
        }));
        dl.push_back(hbox({
            btn_cur_activate->Render(), text(" "),
            btn_cur_compact->Render(), text(" "),
            btn_cur_delete_conv->Render(), text(" "),
            btn_cur_delete_mem->Render(),
            filler(),
        }));
        dl.push_back(hbox({
            text("new ") | dim,
            cur_new_title_input->Render() | size(WIDTH, EQUAL, 28),
            text(" "), cur_new_active_cb->Render(),
            text(" "), cur_new_parent_cb->Render(),
            text(" "), btn_cur_new_conv->Render(),
            filler(),
        }));
        if (!error_snapshot.empty())
            dl.push_back(text("error: " + error_snapshot) | color(Color::Red));
        if (!cur_db_error.empty())
            dl.push_back(text("db: " + cur_db_error) | color(Color::Red));

        const std::string busy = cur_inflight.load() ? " · busy" : "";
        Element detail = panel("DETAIL · " + selected_agent, status_snapshot + busy,
                               vbox(std::move(dl)));

        return vbox({
            hbox({c_agents, text(" "), c_convs, text(" "), c_mems}) | flex,
            detail,
        });
    });

    //

    auto main_tabs = Container::Tab(
        {nodes_renderer, agents_renderer, activity_renderer, chat_renderer, curation_renderer}, &tab_index);

    // Mockup-style header: the tab strip is real Button components (clickable +
    // focusable) rendered as `n Label` chips, the active one inverted.
    static const std::array<const char*, 5> kTabLabels = {
        "1 Nodes", "2 Agents", "3 Activity", "4 Chat", "5 Curation"};
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
            text(" 1-5 tabs · ↑/↓ select · click rows · q quit") | dim,
            filler(),
            text("mantic-mind · control ") | dim,
        });

        return vbox({
            header,
            separator(),
            main_tabs->Render() | flex,
            separator(),
            footer,
        }) | border;
    });

    //

    auto root = MakeCatchEventAfter(top_renderer, [&](Event ev) {
        if (show_add_node || show_pin_entry || show_agent_validation_modal ||
            show_file_browser || show_cur_delete_confirm) return false;
        if (!show_editor) {
            if (ev == Event::Character('1')) { tab_index = 0; return true; }
            if (ev == Event::Character('2')) { tab_index = 1; return true; }
            if (ev == Event::Character('3')) { tab_index = 2; return true; }
            if (ev == Event::Character('4')) { tab_index = 3; return true; }
            if (ev == Event::Character('5')) { tab_index = 4; return true; }
            if (ev == Event::Character('f') && tab_index == 0) {
                forget_selected_node();
                return true;
            }
        }
        if (ev == Event::Escape) {
            if (show_editor) { show_editor = false; return true; }
            screen.ExitLoopClosure()();
            return true;
        }
        if (ev == Event::Character('q') && !show_editor) {
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    //

    auto with_add    = Modal(root,        modal_renderer,     &show_add_node);
    auto with_pin    = Modal(with_add,    pin_modal_renderer, &show_pin_entry);
    auto with_agent_validation = Modal(with_pin, agent_validation_modal_renderer, &show_agent_validation_modal);
    auto with_fb     = Modal(with_agent_validation, fb_renderer_comp,   &show_file_browser);
    auto final_comp  = Modal(with_fb,     cur_delete_modal_renderer, &show_cur_delete_confirm);

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

    // Clear the screen callbacks first so any late refresh()/quit() from a worker
    // thread during the joins is a no-op instead of posting to an exited screen.
    {
        std::lock_guard<std::mutex> lk(screen_mutex_);
        quit_fn_    = {};
        refresh_fn_ = {};
    }

    // Signal the chat stream to abort, then join every thread so none is destroyed
    // while joinable (which would terminate the process) — including on the error path.
    ui_shutting_down = true;
    ticker_running = false;
    if (ticker.joinable())      ticker.join();
    if (chat_thread.joinable()) chat_thread.join();
    if (cur_thread.joinable())  cur_thread.join();

    if (!loop_error.empty()) {
        MM_ERROR("Control UI event loop terminated by exception: {}", loop_error);
    }
}

} // namespace mm
