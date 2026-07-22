#include "node/node_ui.hpp"
#include "node/node_state.hpp"
#include "node/llama_cpp_provisioner.hpp"
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
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <deque>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

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

#ifndef _WIN32
bool executable_on_path(const std::string& name) {
    const char* raw_path = std::getenv("PATH");
    if (!raw_path || !*raw_path) return false;
    for (const auto& directory : util::split(raw_path, ':')) {
        if (directory.empty()) continue;
        std::error_code ec;
        const auto candidate = std::filesystem::path(directory) / name;
        if (std::filesystem::is_regular_file(candidate, ec) && !ec) return true;
    }
    return false;
}
#endif

bool copy_text_to_clipboard(const std::string& text, std::string* error) {
#ifdef _WIN32
    const int wide_size = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
        nullptr, 0);
    if (wide_size <= 0) {
        if (error) *error = "the report is not valid UTF-8";
        return false;
    }
    std::wstring wide(static_cast<size_t>(wide_size), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
            wide.data(), wide_size) != wide_size) {
        if (error) *error = "could not convert the report to Unicode";
        return false;
    }

    bool opened = false;
    for (int attempt = 0; attempt < 5 && !opened; ++attempt) {
        opened = OpenClipboard(nullptr) != FALSE;
        if (!opened) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!opened) {
        if (error) *error = "Windows clipboard is busy";
        return false;
    }
    if (!EmptyClipboard()) {
        CloseClipboard();
        if (error) *error = "could not clear the Windows clipboard";
        return false;
    }

    const size_t bytes = (wide.size() + 1) * sizeof(wchar_t);
    HGLOBAL memory = GlobalAlloc(GMEM_MOVEABLE, bytes);
    if (!memory) {
        CloseClipboard();
        if (error) *error = "could not allocate clipboard memory";
        return false;
    }
    void* destination = GlobalLock(memory);
    if (!destination) {
        GlobalFree(memory);
        CloseClipboard();
        if (error) *error = "could not lock clipboard memory";
        return false;
    }
    std::memcpy(destination, wide.c_str(), bytes);
    GlobalUnlock(memory);
    if (!SetClipboardData(CF_UNICODETEXT, memory)) {
        GlobalFree(memory);
        CloseClipboard();
        if (error) *error = "could not publish the report to the clipboard";
        return false;
    }
    CloseClipboard();
    return true;
#else
    const char* command = nullptr;
#ifdef __APPLE__
    if (executable_on_path("pbcopy")) command = "pbcopy";
#else
    // clip.exe is the most reliable bridge from a WSL node to the Windows host.
    if (executable_on_path("clip.exe")) command = "clip.exe";
    else if (executable_on_path("wl-copy")) command = "wl-copy";
    else if (executable_on_path("xclip")) command = "xclip -selection clipboard";
    else if (executable_on_path("xsel")) command = "xsel --clipboard --input";
#endif
    if (!command) {
        if (error) {
            *error = "no clipboard helper found (install wl-clipboard, xclip, or xsel)";
        }
        return false;
    }
    FILE* pipe = ::popen(command, "w");
    if (!pipe) {
        if (error) *error = "could not start the clipboard helper";
        return false;
    }
    const size_t written = std::fwrite(text.data(), 1, text.size(), pipe);
    const int rc = ::pclose(pipe);
    if (written != text.size() || rc != 0) {
        if (error) *error = "clipboard helper failed";
        return false;
    }
    return true;
#endif
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

// Let focused controls handle input before global shortcuts. FTXUI's standard
// CatchEvent does the reverse, which lets shortcuts steal navigation keys.
class CatchEventAfter : public ftxui::ComponentBase {
public:
    explicit CatchEventAfter(std::function<bool(ftxui::Event)> on_event)
        : on_event_(std::move(on_event)) {}

    bool OnEvent(ftxui::Event event) override {
        if (ftxui::ComponentBase::OnEvent(event)) return true;
        return on_event_(std::move(event));
    }

private:
    std::function<bool(ftxui::Event)> on_event_;
};

ftxui::Component make_catch_event_after(
    ftxui::Component child,
    std::function<bool(ftxui::Event)> on_event) {
    auto component = std::make_shared<CatchEventAfter>(std::move(on_event));
    component->Add(std::move(child));
    return component;
}

} // namespace

NodeUI::NodeUI(NodeState& state, uint16_t listen_port,
               ForgetPairingCallback forget_pairing_cb,
               RequestLlamaUpdateCallback request_llama_update_cb,
               RequestLlamaSwitchCallback request_llama_switch_cb,
               RequestLlamaRecoveryCallback request_llama_recovery_cb)
    : state_(state)
    , listen_port_(listen_port)
    , forget_pairing_cb_(std::move(forget_pairing_cb))
    , request_llama_update_cb_(std::move(request_llama_update_cb))
    , request_llama_switch_cb_(std::move(request_llama_switch_cb))
    , request_llama_recovery_cb_(std::move(request_llama_recovery_cb)) {
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
    mm::tui::LayoutStore layout("data/node-tui-layout.json");
    int node_tab = 0;
    std::vector<std::string> node_tab_labels = {"1 Overview", "2 Runtime", "3 Models & Slots", "4 Logs"};
    bool show_node_tabs = false;
    int left_width = layout.get("overview.left_width", 48, 32, 78);
    int top_height = layout.get("overview.top_height", 22, 12, 38);
    int slot_sel = 0;
    std::vector<std::string> slot_entries;
    auto save_layout = [&]() {
        layout.set("overview.left_width", left_width);
        layout.set("overview.top_height", top_height);
        layout.save();
    };
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

    // ── llama.cpp runtime prompt state ─────────────────────────────────────────
    // These are recomputed each frame by the renderer; the buttons and modal read
    // them by reference.
    bool        show_llama_update_modal  = false;
    bool        show_llama_update_button = false;
    bool        show_llama_engine_modal = false;
    bool        show_llama_engine_button = false;
    bool        show_llama_target_modal = false;
    bool        show_llama_target_button = false;
    bool        show_llama_troubleshoot_button = false;
    bool        show_llama_troubleshoot_modal = false;
    bool        show_llama_release_choice = false;
    bool        show_llama_compile_anyway = false;
    bool        show_llama_current_action = false;
    bool        show_llama_alt_cuda = false;
    bool        show_llama_alt_rocm = false;
    bool        show_llama_alt_vulkan = false;
    bool        show_llama_alt_openvino = false;
    bool        show_llama_alt_metal = false;
    bool        show_llama_alt_cpu = false;
    bool        show_action_modal  = false;
    std::string llama_cur_latest;
    std::string llama_modal_ack_version;
    std::string llama_target_ack_key;
    std::string llama_troubleshoot_ack_fingerprint;
    std::string llama_clipboard_status;
    std::string llama_clipboard_report_key;
    bool llama_clipboard_status_ok = false;
    std::vector<std::string> llama_release_entries;
    std::vector<std::string> llama_release_ids;
    std::vector<std::string> llama_release_assets;
    int llama_release_selection = 0;
    std::vector<std::string> llama_engine_entries;
    std::vector<std::string> llama_engine_ids;
    std::vector<std::string> llama_engine_details;
    int llama_engine_selection = 0;

    auto high_contrast_button_option = [] {
        ButtonOption option = ButtonOption::Simple();
        option.transform = [](const EntryState& state) -> Element {
            Element label = text(state.label) | color(Color::White);
            if (state.focused) label = std::move(label) | inverted | bold;
            return label;
        };
        return option;
    };
    auto llama_target_key = [](const LlamaRuntimeStatus& runtime) {
        return runtime.accelerator + "|" + runtime.variant + "|" +
            runtime.method + "|" + runtime.cuda_architecture + "|" +
            runtime.target_accelerator + "|" + runtime.target_variant + "|" +
            runtime.target_method + "|" + runtime.target_cuda_architecture + "|" +
            runtime.executable_path;
    };

    auto request_llama_update = [&](const std::string& accelerator) {
        if (request_llama_update_cb_) request_llama_update_cb_(accelerator);
    };
    auto btn_llama_update = Button("[ Update llama.cpp ]", [&] {
        // Always reopen the decision prompt: a click must not silently start a
        // potentially long local compilation.
        llama_modal_ack_version.clear();
        show_llama_update_modal = true;
    }, ButtonOption::Simple());
    auto btn_llama_update_maybe = Maybe(btn_llama_update, &show_llama_update_button);

    auto modal_llama_current = Button(" Proceed with current target ", [&] {
        request_llama_update({});
        llama_modal_ack_version = llama_cur_latest;
        show_llama_update_modal = false;
    }, ButtonOption::Simple());
    auto modal_llama_current_maybe = Maybe(modal_llama_current, &show_llama_current_action);
    auto llama_alternative_button = [&](const std::string& label,
                                        const std::string& accelerator) {
        return Button(label, [&, accelerator] {
            request_llama_update(accelerator);
            llama_modal_ack_version = llama_cur_latest;
            show_llama_update_modal = false;
        }, ButtonOption::Simple());
    };
    auto modal_llama_cuda = llama_alternative_button(" Use CUDA release ", "cuda");
    auto modal_llama_rocm = llama_alternative_button(" Use ROCm release ", "rocm");
    auto modal_llama_vulkan = llama_alternative_button(" Use Vulkan release ", "vulkan");
    auto modal_llama_openvino =
        llama_alternative_button(" Use OpenVINO release ", "openvino");
    auto modal_llama_metal = llama_alternative_button(" Use Metal release ", "metal");
    auto modal_llama_cpu = llama_alternative_button(" Use CPU release ", "cpu");
    auto modal_llama_cuda_maybe = Maybe(modal_llama_cuda, &show_llama_alt_cuda);
    auto modal_llama_rocm_maybe = Maybe(modal_llama_rocm, &show_llama_alt_rocm);
    auto modal_llama_vulkan_maybe = Maybe(modal_llama_vulkan, &show_llama_alt_vulkan);
    auto modal_llama_openvino_maybe =
        Maybe(modal_llama_openvino, &show_llama_alt_openvino);
    auto modal_llama_metal_maybe = Maybe(modal_llama_metal, &show_llama_alt_metal);
    auto modal_llama_cpu_maybe = Maybe(modal_llama_cpu, &show_llama_alt_cpu);
    auto modal_llama_later = Button(" Later ", [&] {
        llama_modal_ack_version = llama_cur_latest;
        show_llama_update_modal = false;
    }, ButtonOption::Simple());
    auto modal_llama_btns = Container::Vertical({
        modal_llama_current_maybe,
        modal_llama_cuda_maybe,
        modal_llama_rocm_maybe,
        modal_llama_vulkan_maybe,
        modal_llama_openvino_maybe,
        modal_llama_metal_maybe,
        modal_llama_cpu_maybe,
        modal_llama_later,
    });

    auto btn_llama_engine = Button("[ Change llama.cpp engine ]", [&] {
        show_llama_engine_modal = true;
    }, high_contrast_button_option());
    auto btn_llama_engine_maybe = Maybe(btn_llama_engine, &show_llama_engine_button);
    MenuOption llama_engine_menu_option = MenuOption::Vertical();
    llama_engine_menu_option.entries_option.transform =
        [](const EntryState& state) -> Element {
            Element row = text(state.label) | color(Color::White);
            if (state.active) row = std::move(row) | inverted | bold;
            else if (state.focused) row = std::move(row) | underlined;
            return row;
        };
    auto llama_engine_menu = Menu(&llama_engine_entries, &llama_engine_selection,
                                  llama_engine_menu_option);
    auto modal_llama_engine_apply = Button("[ Use selected engine ]", [&] {
        if (request_llama_switch_cb_ && llama_engine_selection >= 0 &&
            llama_engine_selection < static_cast<int>(llama_engine_ids.size())) {
            request_llama_switch_cb_(
                llama_engine_ids[static_cast<size_t>(llama_engine_selection)]);
        }
        show_llama_engine_modal = false;
    }, high_contrast_button_option());
    auto modal_llama_engine_cancel = Button("[ Cancel ]", [&] {
        show_llama_engine_modal = false;
    }, high_contrast_button_option());
    auto llama_engine_actions = Container::Horizontal({
        modal_llama_engine_apply, modal_llama_engine_cancel,
    });
    auto llama_engine_controls = Container::Vertical({
        llama_engine_menu, llama_engine_actions,
    });

    auto request_llama_recovery = [&](const std::string& action,
                                      const std::string& variant = std::string{}) {
        if (request_llama_recovery_cb_) request_llama_recovery_cb_(action, variant);
        const auto rt = state_.get_llama_runtime();
        llama_troubleshoot_ack_fingerprint = rt.troubleshooting.fingerprint;
        show_llama_troubleshoot_modal = false;
    };
    auto btn_llama_target = Button("[ Review target build ]", [&] {
        show_llama_target_modal = true;
    }, high_contrast_button_option());
    auto btn_llama_target_maybe = Maybe(btn_llama_target, &show_llama_target_button);
    auto modal_llama_target_install = Button("[ Install target build ]", [&] {
        const auto rt = state_.get_llama_runtime();
        llama_target_ack_key = llama_target_key(rt);
        if (request_llama_recovery_cb_)
            request_llama_recovery_cb_("target", {});
        show_llama_target_modal = false;
    }, high_contrast_button_option());
    auto modal_llama_target_later = Button("[ Keep current for now ]", [&] {
        llama_target_ack_key = llama_target_key(state_.get_llama_runtime());
        show_llama_target_modal = false;
    }, high_contrast_button_option());
    auto llama_target_actions = Container::Horizontal({
        modal_llama_target_install, modal_llama_target_later,
    });
    auto btn_llama_troubleshoot = Button("[ Troubleshoot llama.cpp ]", [&] {
        llama_troubleshoot_ack_fingerprint.clear();
        if (request_llama_recovery_cb_)
            request_llama_recovery_cb_("diagnose", {});
    }, high_contrast_button_option());
    auto btn_llama_troubleshoot_maybe =
        Maybe(btn_llama_troubleshoot, &show_llama_troubleshoot_button);
    auto modal_llama_diagnose = Button("[ Re-run diagnostics ]", [&] {
        request_llama_recovery("diagnose");
    }, high_contrast_button_option());
    auto modal_llama_copy_report = Button("[ Copy report ]", [&] {
        const auto rt = state_.get_llama_runtime();
        const std::string report = format_llama_troubleshooting_report(
            rt.troubleshooting, rt.build_log_path);
        std::string error;
        llama_clipboard_status_ok = copy_text_to_clipboard(report, &error);
        llama_clipboard_status = llama_clipboard_status_ok
            ? "Troubleshooting report copied to the clipboard."
            : "Copy failed: " + error;
        llama_clipboard_report_key = rt.troubleshooting.fingerprint + "|" +
            rt.build_log_path;
    }, high_contrast_button_option());
    auto modal_llama_retry = Button("[ Retry normal provisioning ]", [&] {
        request_llama_recovery("retry");
    }, high_contrast_button_option());
    auto modal_llama_compile_anyway = Button("[ Compile anyway - one attempt ]", [&] {
        request_llama_recovery("compile-anyway");
    }, high_contrast_button_option());
    auto modal_llama_compile_anyway_maybe =
        Maybe(modal_llama_compile_anyway, &show_llama_compile_anyway);
    MenuOption llama_release_menu_option = MenuOption::Vertical();
    llama_release_menu_option.entries_option.transform =
        [](const EntryState& state) -> Element {
            Element row = text(state.label) | color(Color::White);
            if (state.active) row = std::move(row) | inverted | bold;
            else if (state.focused) row = std::move(row) | underlined;
            return row;
        };
    auto llama_release_menu = Menu(&llama_release_entries, &llama_release_selection,
                                   llama_release_menu_option);
    auto llama_release_menu_maybe = Maybe(llama_release_menu, &show_llama_release_choice);
    auto modal_llama_use_release = Button("[ Use selected release ]", [&] {
        if (llama_release_selection >= 0 &&
            llama_release_selection < static_cast<int>(llama_release_ids.size())) {
            request_llama_recovery(
                "release", llama_release_ids[static_cast<size_t>(llama_release_selection)]);
        }
    }, high_contrast_button_option());
    auto modal_llama_use_release_maybe =
        Maybe(modal_llama_use_release, &show_llama_release_choice);
    auto modal_llama_troubleshoot_dismiss = Button("[ Dismiss ]", [&] {
        const auto rt = state_.get_llama_runtime();
        llama_troubleshoot_ack_fingerprint = rt.troubleshooting.fingerprint;
        show_llama_troubleshoot_modal = false;
    }, high_contrast_button_option());

    auto action_cancel = Button("  Cancel  ", [&] {
        state_.request_action_cancel();
    }, ButtonOption::Simple());
    auto action_btns = Container::Horizontal({action_cancel});

    auto left_divider = mm::tui::resizable_divider(
        &left_width, 32, 78, mm::tui::SplitAxis::Columns, [&](int) { save_layout(); });
    auto row_divider = mm::tui::resizable_divider(
        &top_height, 12, 38, mm::tui::SplitAxis::Rows, [&](int) { save_layout(); });
    auto generated_scroll = mm::tui::wrapped_scroll_view([&]() {
        const auto stream = state_.get_streaming_text();
        std::string out;
        if (!stream.thinking.empty()) out += "[thinking]\n" + stream.thinking + "\n\n";
        out += stream.content;
        return out.empty() ? std::string{"(idle -- waiting for inference)"} : out;
    });
    auto llama_troubleshoot_details = mm::tui::wrapped_scroll_view([&]() {
        const auto rt = state_.get_llama_runtime();
        return format_llama_troubleshooting_report(
            rt.troubleshooting, rt.build_log_path);
    });
    auto llama_troubleshoot_primary_actions = Container::Horizontal({
        modal_llama_use_release_maybe,
        modal_llama_retry,
    });
    auto llama_troubleshoot_secondary_actions = Container::Horizontal({
        modal_llama_diagnose,
        modal_llama_copy_report,
        modal_llama_troubleshoot_dismiss,
    });
    auto llama_troubleshoot_btns = Container::Vertical({
        llama_troubleshoot_details,
        llama_release_menu_maybe,
        llama_troubleshoot_primary_actions,
        llama_troubleshoot_secondary_actions,
        modal_llama_compile_anyway_maybe,
    });
    auto slot_menu = Menu(&slot_entries, &slot_sel, MenuOption::Vertical());

    // Keep the selected page and the focus tree in lockstep. Previously all
    // controls from all pages shared one container, so hidden controls could
    // retain focus. Explicit button tabs also provide consistent mouse and
    // keyboard behavior across Windows terminal hosts.
    Components node_tab_buttons;
    for (int i = 0; i < static_cast<int>(node_tab_labels.size()); ++i) {
        ButtonOption option = ButtonOption::Simple();
        option.transform = [&, i](const EntryState& state) -> Element {
            Element label = text(" " + node_tab_labels[static_cast<size_t>(i)] + " ");
            if (node_tab == i) label = std::move(label) | inverted | bold;
            else if (state.focused) label = std::move(label) | underlined;
            else label = std::move(label) | dim;
            return label;
        };
        node_tab_buttons.push_back(Button(
            node_tab_labels[static_cast<size_t>(i)],
            [&, i] { node_tab = i; }, option));
    }
    auto tab_button_row = Container::Horizontal(std::move(node_tab_buttons));
    auto overview_controls = Container::Vertical({
        left_divider, row_divider, generated_scroll,
    });
    auto runtime_controls = Container::Vertical({
        btn_llama_engine_maybe,
        btn_llama_target_maybe, btn_llama_troubleshoot_maybe,
        btn_llama_update_maybe,
    });
    auto model_controls = Container::Vertical({
        slot_menu,
    });
    auto log_controls = Container::Vertical({});
    auto page_controls = Container::Tab({
        overview_controls, runtime_controls, model_controls, log_controls,
    }, &node_tab);
    auto tab_button_row_maybe = Maybe(tab_button_row, &show_node_tabs);
    auto page_controls_maybe = Maybe(page_controls, &show_node_tabs);
    auto main_container = Container::Vertical({
        tab_button_row_maybe, page_controls_maybe, btn_forget_pairing,
    });

    // ── Renderer ──────────────────────────────────────────────────────────────
    auto render = Renderer(main_container, [&]() -> Element {

        // Snapshot mutable state
        bool registered   = state_.is_registered();
        show_node_tabs    = registered;
        auto metrics      = state_.get_metrics();
        auto loaded_model = state_.get_loaded_model();
        auto active_agent = state_.get_active_agent();
        auto slots        = state_.get_slots();
        auto llama_rt     = state_.get_llama_runtime();
        slot_entries.clear();
        for (const auto& slot : slots)
            slot_entries.push_back(slot.id + "  " + to_string(slot.state) + "  " + mm::tui::short_model(slot.model_path));
        if (slot_sel >= static_cast<int>(slots.size()))
            slot_sel = std::max(0, static_cast<int>(slots.size()) - 1);

        auto node_id      = state_.get_node_id();
        auto last_error   = state_.get_last_error();
        auto action_prog  = state_.get_action_progress();
        auto api_keys     = state_.get_api_keys();
        auto streaming    = state_.get_streaming_text();

        show_action_modal = action_prog.active;

        // Drive the llama.cpp update and recovery controls from current runtime state.
        llama_cur_latest = llama_rt.latest_version;
        const bool llama_busy = action_prog.active &&
                                action_prog.operation_id == "llama-runtime";
        const bool llama_ready =
            (llama_rt.status == "ready" || llama_rt.status == "resolved") &&
            !llama_rt.executable_path.empty();
        show_llama_engine_button = static_cast<bool>(request_llama_switch_cb_) &&
                                   llama_ready && !llama_busy;
        show_llama_target_button = llama_rt.target_mismatch && llama_ready &&
                                   static_cast<bool>(request_llama_recovery_cb_) &&
                                   !llama_busy;
        show_llama_troubleshoot_button =
            static_cast<bool>(request_llama_recovery_cb_) && !llama_busy;
        const bool can_update_llama = llama_rt.update_available && llama_rt.managed &&
                                      static_cast<bool>(request_llama_update_cb_) &&
                                      !llama_busy;
        show_llama_update_button = can_update_llama;
        show_llama_current_action = llama_rt.update_action != "unavailable";
        auto has_llama_alternative = [&](const std::string& accelerator) {
            return std::find(llama_rt.update_release_alternatives.begin(),
                             llama_rt.update_release_alternatives.end(), accelerator)
                != llama_rt.update_release_alternatives.end();
        };
        show_llama_alt_cuda = has_llama_alternative("cuda");
        show_llama_alt_rocm = has_llama_alternative("rocm");
        show_llama_alt_vulkan = has_llama_alternative("vulkan");
        show_llama_alt_openvino = has_llama_alternative("openvino");
        show_llama_alt_metal = has_llama_alternative("metal");
        show_llama_alt_cpu = has_llama_alternative("cpu");
        show_llama_update_modal = !show_action_modal && can_update_llama &&
            (show_llama_update_modal ||
             llama_modal_ack_version != llama_rt.latest_version);

        llama_release_entries.clear();
        llama_release_ids.clear();
        llama_release_assets.clear();
        for (const auto& variant : llama_rt.troubleshooting.variants) {
            if (!variant.release_available) continue;
            llama_release_ids.push_back(variant.id);
            llama_release_assets.push_back(variant.release_asset);
            llama_release_entries.push_back((variant.recommended ? "* " : "  ") +
                                            variant.label +
                                            (variant.recommended ? " (recommended)" : ""));
        }
        if (llama_release_selection >= static_cast<int>(llama_release_entries.size()))
            llama_release_selection = std::max(
                0, static_cast<int>(llama_release_entries.size()) - 1);
        show_llama_release_choice = !llama_release_entries.empty();
        show_llama_compile_anyway =
            llama_rt.troubleshooting.can_override_checks;
        const bool can_troubleshoot = llama_rt.troubleshooting.required &&
            static_cast<bool>(request_llama_recovery_cb_);
        show_llama_troubleshoot_modal = !show_action_modal && can_troubleshoot &&
            (show_llama_troubleshoot_modal ||
             llama_troubleshoot_ack_fingerprint !=
                 llama_rt.troubleshooting.fingerprint);
        if (show_llama_troubleshoot_modal) show_llama_update_modal = false;

        llama_engine_entries.clear();
        llama_engine_ids.clear();
        llama_engine_details.clear();
        for (const auto& variant : llama_rt.available_variants) {
            if (!variant.platform_supported ||
                (!variant.release_available && !variant.source_supported))
                continue;
            const bool current = (!llama_rt.variant.empty() &&
                                  llama_rt.variant == variant.id) ||
                ((llama_rt.variant.empty() ||
                  llama_rt.variant == llama_rt.accelerator) &&
                 llama_rt.accelerator == variant.backend);
            const std::string action = variant.release_available
                ? "official release" : "compile locally";
            llama_engine_ids.push_back(variant.id);
            llama_engine_entries.push_back(
                (current ? "* " : "  ") + variant.label + " - " + action);
            llama_engine_details.push_back(
                variant.reason + (variant.release_asset.empty()
                    ? std::string{} : " [" + variant.release_asset + "]"));
        }
        if (llama_engine_selection >= static_cast<int>(llama_engine_entries.size()))
            llama_engine_selection = std::max(
                0, static_cast<int>(llama_engine_entries.size()) - 1);
        show_llama_engine_modal = !show_action_modal && show_llama_engine_button &&
                                  show_llama_engine_modal &&
                                  !llama_engine_entries.empty();
        if (show_llama_engine_modal) {
            show_llama_update_modal = false;
            show_llama_troubleshoot_modal = false;
        }
        const bool can_install_target = llama_rt.target_mismatch && llama_ready &&
            static_cast<bool>(request_llama_recovery_cb_) && !llama_busy;
        show_llama_target_modal = !show_action_modal &&
            !show_llama_troubleshoot_modal && !show_llama_engine_modal &&
            can_install_target &&
            (show_llama_target_modal ||
             llama_target_ack_key != llama_target_key(llama_rt));
        if (show_llama_target_modal) show_llama_update_modal = false;

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

        std::string runtime_summary = llama_rt.status.empty() ? "unknown" : llama_rt.status;
        Color runtime_color =
            (llama_rt.status == "ready" || llama_rt.status == "resolved")
                ? Color::Green
                : (llama_rt.status == "failed" ? Color::RedLight : Color::Yellow);
        if (!last_error.empty()) {
            runtime_summary = "error: " + last_error;
            runtime_color = Color::RedLight;
        } else if ((llama_rt.status == "ready" || llama_rt.status == "resolved") &&
                   !llama_rt.accelerator.empty()) {
            runtime_summary += " (" +
                (llama_rt.variant.empty() ? llama_rt.accelerator : llama_rt.variant) + ")";
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
            text(std::to_string(slot_suspended) + " suspended") | dim,
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
        if (action_prog.active) {
            std::string head = action_prog.stage.empty() ? "working" : action_prog.stage;
            if (action_prog.total_steps > 0)
                head += "  (" + std::to_string(action_prog.step) + "/"
                      + std::to_string(action_prog.total_steps) + ")";
            Elements pr = { text(head) | color(Color::Cyan) | bold };
            if (action_prog.fraction >= 0.0) {
                const int pct = static_cast<int>(action_prog.fraction * 100.0 + 0.5);
                pr.push_back(hbox({ gauge(static_cast<float>(action_prog.fraction)) | flex,
                                    text(" " + std::to_string(pct) + "%") }));
            } else {
                pr.push_back(text(std::string(spin) + " working…") | color(Color::Yellow));
            }
            if (!action_prog.detail.empty())
                pr.push_back(text(shorten_middle(action_prog.detail, 44)) | dim);
            status_rows.push_back(field("action", vbox(std::move(pr))));
        } else if (llama_rt.update_available) {
            Element line = text("llama.cpp " +
                (llama_rt.version.empty() ? std::string{"?"} : llama_rt.version) +
                " → " + llama_rt.latest_version) | color(Color::Cyan) | bold;
            if (show_llama_update_button)
                status_rows.push_back(field("update", vbox({ line, btn_llama_update->Render() })));
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
        }) | size(WIDTH, EQUAL, left_width);

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
        Element gen_panel = panel("GENERATED TEXT", gen_caption, generated_scroll->Render()) | flex;

        // ── SLOTS panel ───────────────────────────────────────────────────────
        Element slots_panel =
            panel("SLOTS", std::to_string(slots.size()) + " tracked", slot_table(slots, false));

        Element right_col = vbox({
            gen_panel,
            slots_panel,
        }) | flex;

        const int dimy = ftxui::Terminal::Size().dimy;
        const int top_h = top_height;
        Element top_region = hbox({
            left_col,
            left_divider->Render(),
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

        auto runtime_status = [](const std::string& name, const auto& runtime,
                                 const std::string& variant) {
            const std::string install = runtime.managed
                ? "managed " + (runtime.method.empty() ? std::string{"runtime"}
                                                        : runtime.method)
                : (runtime.method == "path" ? std::string{"external PATH"}
                                             : runtime.method);
            Elements rows = {
                hbox({text("engine        ") | dim, text(name) | bold}),
                hbox({text("status        ") | dim, text(runtime.status) | bold}),
                hbox({text("version       ") | dim,
                      text(runtime.version.empty() ? "not reported" : runtime.version)}),
                hbox({text("backend       ") | dim,
                      text(runtime.accelerator.empty() ? "unknown" : runtime.accelerator)}),
                hbox({text("variant       ") | dim,
                      text(variant.empty()
                          ? (runtime.accelerator.empty() ? "unknown" : runtime.accelerator)
                          : variant)}),
                hbox({text("install       ") | dim,
                      text(install.empty() ? "unknown" : install)}),
                paragraph("path          " + (runtime.executable_path.empty() ? std::string{"not resolved"} : runtime.executable_path)) | dim,
            };
            if (!runtime.last_error.empty()) rows.push_back(paragraph(runtime.last_error) | color(Color::Red));
            return vbox(std::move(rows));
        };
        Element llama_target_notice = text("");
        if (llama_rt.target_mismatch) {
            const std::string actual =
                (llama_rt.variant.empty() ? llama_rt.accelerator : llama_rt.variant) +
                (llama_rt.cuda_architecture.empty()
                    ? std::string{} : " / " + llama_rt.cuda_architecture);
            const std::string target =
                (llama_rt.target_variant.empty() ? llama_rt.target_accelerator
                                                  : llama_rt.target_variant) +
                (llama_rt.target_cuda_architecture.empty()
                    ? std::string{} : " / " + llama_rt.target_cuda_architecture);
            llama_target_notice = panel("LLAMA.CPP TARGET", vbox({
                hbox({text("active        ") | dim,
                      text(actual.empty() ? std::string{"unknown"} : actual) | bold}),
                hbox({text("target        ") | dim,
                      text(target.empty() ? std::string{"unknown"} : target) |
                          color(Color::Yellow) | bold}),
                paragraph(llama_rt.target_mismatch_reason) | color(Color::Yellow),
            }));
        }
        Element runtimes_page = vbox({
            panel("llama.cpp", runtime_status("llama.cpp", llama_rt,
                                               llama_rt.variant)),
            std::move(llama_target_notice),
            vbox({
                hbox({btn_llama_engine_maybe->Render(), text(" "),
                      btn_llama_target_maybe->Render()}),
                hbox({btn_llama_troubleshoot_maybe->Render(), text(" "),
                      btn_llama_update_maybe->Render()}),
            }),
        });

        std::vector<std::string> local_models;
        for (const auto& slot : slots) {
            if (!slot.model_path.empty() &&
                std::find(local_models.begin(), local_models.end(), slot.model_path) ==
                    local_models.end()) {
                local_models.push_back(slot.model_path);
            }
        }
        Elements local_model_rows;
        if (local_models.empty()) {
            local_model_rows.push_back(
                text("No local models are currently loaded or suspended.") | dim);
        } else {
            for (const auto& model : local_models)
                local_model_rows.push_back(paragraph(model));
        }
        local_model_rows.push_back(text(
            "Models are transferred into the node's local ModelStore by Mantic-Mind-Control.") | dim);
        Element models_page = vbox({
            panel("LOCAL MODELS", vbox(std::move(local_model_rows))),
            panel("SLOTS", vbox({
                slot_menu->Render() | yframe | flex,
                text("Slot lifecycle is managed by Mantic-Mind-Control's agent scheduler.") | dim,
            })) | flex,
        });
        Element overview_page = vbox({top_region, row_divider->Render(), log_panel});
        Element page = node_tab == 1 ? runtimes_page
                     : node_tab == 2 ? models_page
                     : node_tab == 3 ? log_panel
                                     : overview_page;

        Element header = hbox({
            text(" mantic-mind ") | bold,
            text(mm::util::hostname() + " [" + mm::tui::short_node_id(node_id) + "]") | dim,
            filler(),
            text(spin) | color(Color::Green), text(" " + serving) | dim,
            text("  │  ") | dim, text("✓ registered") | color(Color::Green),
            text("  │  ") | dim, text("up " + std::string(upbuf)) | dim,
            text("  │  ") | dim, text(clock_hms()) | dim,
            text(" "),
        });

        Element footer = hbox({
            btn_forget_pairing->Render(),
            text(node_tab == 3
                ? "  1-4 tabs · arrows focus/scroll · Enter select · f forget · q quit"
                : "  1-4 tabs · arrows focus · Enter select · f forget · q quit") | dim,
            filler(),
            text("mantic-mind · node ") | dim,
        });

        return vbox({
            header,
            tab_button_row->Render(),
            separator(),
            std::move(page) | flex,
            footer,
        }) | border;
    });

    // ── llama.cpp update prompt (FTXUI modal with buttons) ─────────────────────
    auto llama_update_modal_renderer = Renderer(modal_llama_btns, [&]() -> Element {
        const auto rt = state_.get_llama_runtime();
        const std::string action = rt.update_action == "compile"
            ? "compile llama-server locally"
            : (rt.update_action == "release"
                ? "download official release"
                : (rt.update_action == "unavailable"
                    ? "no current-target release"
                    : "retry release, then source fallback"));
        Elements rows = {
            text(" llama.cpp update available ") | bold | hcenter,
            separator(),
            hbox({text(" installed : "),
                  text(rt.version.empty() ? std::string{"(unknown)"} : rt.version)}),
            hbox({text(" latest    : "), text(rt.latest_version) | color(Color::Green)}),
            hbox({text(" target    : "),
                  text(rt.accelerator.empty() ? std::string{"?"} : rt.accelerator) | bold}),
            hbox({text(" action    : "), text(action) | color(
                rt.update_action == "compile" ? Color::Yellow : Color::Cyan)}),
        };
        if (!rt.update_warning.empty()) {
            rows.push_back(separator());
            rows.push_back(paragraph(" " + rt.update_warning) | color(Color::Yellow));
        }
        if (!rt.update_release_alternatives.empty()) {
            rows.push_back(paragraph(
                " Choose Proceed for the current target, or switch this update to an official release alternative below.") | dim);
        }
        rows.push_back(separator());
        rows.push_back(modal_llama_btns->Render() | hcenter);
        return vbox(std::move(rows)) | border | size(WIDTH, EQUAL, 68);
    });
    auto llama_engine_modal_renderer =
        Renderer(llama_engine_controls, [&]() -> Element {
            const auto rt = state_.get_llama_runtime();
            Elements rows{
                text(" Change llama.cpp engine ") | bold | hcenter |
                    color(Color::Cyan),
                separator(),
                hbox({text(" engine  : ") | dim, text("llama.cpp") | bold}),
                hbox({text(" backend : ") | dim,
                      text(rt.accelerator.empty() ? std::string{"unknown"}
                                                  : rt.accelerator) | bold}),
                hbox({text(" variant : ") | dim,
                      text(rt.variant.empty()
                          ? (rt.accelerator.empty() ? std::string{"unknown"}
                                                    : rt.accelerator)
                          : rt.variant)}),
                hbox({text(" install : ") | dim,
                      text(rt.managed ? "managed " + rt.method
                                      : (rt.method == "path" ? "external PATH"
                                                             : rt.method))}),
                separator(),
                llama_engine_menu->Render() | size(HEIGHT, LESS_THAN, 9),
            };
            if (llama_engine_selection >= 0 &&
                llama_engine_selection < static_cast<int>(llama_engine_details.size())) {
                rows.push_back(paragraph(
                    " " + llama_engine_details[static_cast<size_t>(
                              llama_engine_selection)]) | dim);
            }
            rows.push_back(paragraph(
                " Switching changes the managed runtime used by new or restarted llama.cpp slots. Existing running slots keep their current process until unloaded.") |
                color(Color::Yellow));
            rows.push_back(separator());
            rows.push_back(llama_engine_actions->Render() | hcenter);
            return vbox(std::move(rows)) | border | size(WIDTH, EQUAL, 76);
        });
    auto llama_target_modal_renderer =
        Renderer(llama_target_actions, [&]() -> Element {
            const auto rt = state_.get_llama_runtime();
            const std::string actual_variant = rt.variant.empty()
                ? rt.accelerator : rt.variant;
            const std::string target_variant = rt.target_variant.empty()
                ? rt.target_accelerator : rt.target_variant;
            const std::string actual = actual_variant +
                (rt.cuda_architecture.empty()
                    ? std::string{} : " / " + rt.cuda_architecture);
            const std::string target = target_variant +
                (rt.target_cuda_architecture.empty()
                    ? std::string{} : " / " + rt.target_cuda_architecture);

            bool target_release_assessed = false;
            for (const auto& variant : rt.available_variants) {
                const bool matches = variant.id == target_variant ||
                    (target_variant == rt.target_accelerator &&
                     variant.backend == rt.target_accelerator);
                if (matches && variant.release_available) {
                    target_release_assessed = true;
                    break;
                }
            }

            std::string consequence;
            Color consequence_color = Color::Yellow;
            if (rt.target_method == "source") {
                consequence =
                    "Installing the target will compile llama-server locally.";
            } else if (rt.target_method == "release") {
                consequence = target_release_assessed
                    ? "Installing the target will download its assessed official release."
                    : "No complete official target release is currently assessed; "
                      "release-only installation may fail.";
                if (target_release_assessed) consequence_color = Color::Cyan;
            } else if (target_release_assessed) {
                consequence =
                    "Installing the target will prefer its official release and use "
                    "source compilation only if the release cannot be used.";
                consequence_color = Color::Cyan;
            } else {
                consequence =
                    "No complete official target release is currently assessed. "
                    "Installation will compile locally if the release lookup confirms none.";
            }

            Elements rows{
                text(" llama.cpp target build differs ") | bold | hcenter |
                    color(Color::Yellow),
                separator(),
                hbox({text(" active : ") | dim,
                      text(actual.empty() ? std::string{"unknown"} : actual) | bold}),
                hbox({text(" target : ") | dim,
                      text(target.empty() ? std::string{"unknown"} : target) |
                          color(Color::Yellow) | bold}),
                hbox({text(" policy : ") | dim,
                      text(rt.target_method.empty() ? std::string{"auto"}
                                                    : rt.target_method)}),
                separator(),
                paragraph(" " + rt.target_mismatch_reason) | color(Color::Yellow),
                paragraph(" " + consequence) | color(consequence_color),
                paragraph(
                    " Keeping the current build leaves the active engine unchanged. "
                    "You can reopen this prompt from the Runtime tab.") | dim,
                separator(),
                llama_target_actions->Render() | hcenter,
            };
            return vbox(std::move(rows)) | border | size(WIDTH, EQUAL, 74);
        });
    auto llama_troubleshoot_modal_renderer =
        Renderer(llama_troubleshoot_btns, [&]() -> Element {
            const auto rt = state_.get_llama_runtime();
            const auto& report = rt.troubleshooting;
            const std::string report_key = report.fingerprint + "|" + rt.build_log_path;
            if (!llama_clipboard_report_key.empty() &&
                llama_clipboard_report_key != report_key) {
                llama_clipboard_report_key.clear();
                llama_clipboard_status.clear();
            }
            Elements rows{
                text(" llama.cpp troubleshooting wizard ") | bold | hcenter |
                    color(Color::Yellow),
                separator(),
                hbox({text(" host   : ") | dim,
                      text(report.platform + "/" + report.architecture)}),
                hbox({text(" target : ") | dim,
                      text(report.target_backend.empty() ? std::string{"?"}
                                                         : report.target_backend) | bold}),
                hbox({text(" stage  : ") | dim,
                      text(shorten_middle(report.failure_stage, 58)) |
                          color(Color::Red)}),
            };
            if (!rt.build_log_path.empty()) {
                rows.push_back(hbox({
                    text(" log    : ") | dim,
                    text(shorten_middle(rt.build_log_path, 58)) |
                        color(Color::GrayLight),
                }));
            }
            rows.push_back(paragraph(" " + report.summary));
            rows.push_back(separator());
            rows.push_back(llama_troubleshoot_details->Render() | frame |
                           vscroll_indicator | size(HEIGHT, EQUAL, 12));
            if (show_llama_release_choice) {
                rows.push_back(separator());
                rows.push_back(text(" Complete official release fallbacks ") | bold);
                rows.push_back(llama_release_menu_maybe->Render() |
                               size(HEIGHT, LESS_THAN, 5));
                if (llama_release_selection >= 0 &&
                    llama_release_selection < static_cast<int>(llama_release_assets.size())) {
                    rows.push_back(hbox({
                        text(" asset : ") | dim,
                        text(shorten_middle(
                            llama_release_assets[static_cast<size_t>(llama_release_selection)],
                            62)) | color(Color::GrayLight),
                    }));
                }
            }
            rows.push_back(separator());
            rows.push_back(hbox({
                modal_llama_use_release_maybe->Render(),
                text("  "),
                modal_llama_retry->Render(),
            }) | hcenter);
            rows.push_back(hbox({
                modal_llama_diagnose->Render(),
                text("  "),
                modal_llama_copy_report->Render(),
                text("  "),
                modal_llama_troubleshoot_dismiss->Render(),
            }) | hcenter);
            if (!llama_clipboard_status.empty()) {
                rows.push_back(paragraph(" " + llama_clipboard_status) |
                    color(llama_clipboard_status_ok ? Color::GreenLight
                                                    : Color::RedLight));
            }
            if (show_llama_compile_anyway) {
                rows.push_back(paragraph(
                    " One-attempt override skips only the environment preflight; CMake and compiler errors still stop the build.") |
                    color(Color::Yellow));
                rows.push_back(modal_llama_compile_anyway_maybe->Render() | hcenter);
            }
            return vbox(std::move(rows)) | border | size(WIDTH, EQUAL, 78);
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
    auto with_llama_update =
        Modal(render, llama_update_modal_renderer, &show_llama_update_modal);
    auto with_llama_engine =
        Modal(with_llama_update, llama_engine_modal_renderer,
              &show_llama_engine_modal);
    auto with_llama_target =
        Modal(with_llama_engine, llama_target_modal_renderer,
              &show_llama_target_modal);
    auto with_llama_troubleshooting = Modal(
        with_llama_target, llama_troubleshoot_modal_renderer,
        &show_llama_troubleshoot_modal);
    auto with_modal =
        Modal(with_llama_troubleshooting, action_modal_renderer, &show_action_modal);

    auto component = make_catch_event_after(with_modal, [&](Event ev) {
        if (show_action_modal) {
            if (ev == Event::Escape) {
                state_.request_action_cancel();
                return true;
            }
            return false;
        }
        if (show_llama_troubleshoot_modal) {
            if (ev == Event::Escape) {
                const auto rt = state_.get_llama_runtime();
                llama_troubleshoot_ack_fingerprint =
                    rt.troubleshooting.fingerprint;
                show_llama_troubleshoot_modal = false;
                return true;
            }
            return false;
        }
        if (show_llama_target_modal) {
            if (ev == Event::Escape) {
                llama_target_ack_key =
                    llama_target_key(state_.get_llama_runtime());
                show_llama_target_modal = false;
                return true;
            }
            return false;
        }
        if (show_llama_engine_modal) {
            if (ev == Event::Escape) {
                show_llama_engine_modal = false;
                return true;
            }
            return false;
        }
        if (show_llama_update_modal) {
            if (ev == Event::Escape) {
                llama_modal_ack_version = llama_cur_latest;
                show_llama_update_modal = false;
                return true;
            }
            return false;
        }
        if (ev == Event::Character('1')) { node_tab = 0; return true; }
        if (ev == Event::Character('2')) { node_tab = 1; return true; }
        if (ev == Event::Character('3')) { node_tab = 2; return true; }
        if (ev == Event::Character('4')) { node_tab = 3; return true; }

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

        // Log navigation belongs to the Logs page. On other pages these keys
        // remain available to the focused controls and their containers.
        if (node_tab == 3) {
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
    save_layout();

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
