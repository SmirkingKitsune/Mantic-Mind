#include "control/control_ui.hpp"
#include "control/node_registry.hpp"
#include "control/audio_player.hpp"
#include "control/agent_manager.hpp"
#include "control/agent_scheduler.hpp"
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
                     AgentScheduler& scheduler,
                     std::string models_dir,
                     std::string control_base_url,
                     std::string control_api_token,
                     LocalChatFallback local_chat_fallback,
                     ShutdownCallback shutdown_callback)
    : registry_(registry)
    , control_api_token_(std::move(control_api_token))
    , agents_(agents)
    , scheduler_(scheduler)
    , models_dir_(std::move(models_dir))
    , control_base_url_(std::move(control_base_url))
    , local_chat_fallback_(std::move(local_chat_fallback))
    , shutdown_callback_(std::move(shutdown_callback))
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
    std::mutex node_operation_mutex;
    std::atomic<bool> node_operation_inflight{false};
    std::thread node_operation_thread;
    std::atomic<bool> pairing_inflight{false};
    std::thread pairing_thread;
    std::mutex active_node_operation_mutex;
    NodeOperationsPtr active_node_operation;
    bool show_node_network_confirm = false;
    NodeId pending_node_network_id;
    enum class NodeRuntimeNetworkAction {
        Provision,
        CheckUpdate,
        Update,
        Switch,
        Recover,
    };
    NodeRuntimeNetworkAction pending_node_network_action =
        NodeRuntimeNetworkAction::Provision;
    std::string node_runtime_accelerator_input;
    std::string node_runtime_variant_input;
    std::string node_runtime_variant_hint;
    int node_runtime_recovery_action = 0;
    std::vector<std::string> node_runtime_recovery_labels = {
        "retry", "target", "compile-anyway", "release"};

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

    // Agents tab.
    int  agent_sel    = 0;
    bool show_editor  = false;
    std::vector<AgentConfig> agent_rows;   // per-frame snapshot the menu transform indexes
    std::string ed_id, ed_id_orig, ed_name, ed_model, ed_sysprompt, ed_pref_node;
    // Sampling (generation settings still flow through RuntimeSettings in the request contract)
    std::string ed_temp_s{"0.70"}, ed_topp_s{"0.90"}, ed_topk_s{"-1"};
    std::string ed_minp_s{"-1.00"}, ed_presence_s{"0.00"}, ed_repeat_s{"-1.00"};
    std::string ed_max_s{"1024"};
    // Local llama.cpp and remote API runtime settings.
    int ed_backend = 0;
    std::vector<std::string> ed_backend_labels = {"llama.cpp", "Remote API"};
    std::string ed_unsupported_backend;
    std::string ed_ctx_s{"4096"}, ed_gpu_layers_s{"-1"}, ed_threads_s{"-1"};
    std::string ed_threads_http_s{"-1"}, ed_parallel_s{"1"};
    std::string ed_batch_s{"-1"}, ed_ubatch_s{"-1"};
    std::string ed_llama_extra_args_text;
    bool ed_flash{true};
    std::string ed_served_model;
    std::string ed_api_base{"https://api.openai.com"}, ed_api_path{"/v1/chat/completions"};
    std::string ed_api_key, ed_api_key_env{"OPENAI_API_KEY"};
    bool ed_vision{false};
    std::string ed_mmproj;
    bool ed_reasoning{false}, ed_memories{true}, ed_tools{false};
    ModelCapabilityInfo ed_model_info;
    std::vector<ValidationIssue> ed_validation_issues;
    std::string ed_validation_signature;
    std::string agent_validation_title;
    std::vector<std::string> agent_validation_lines;
    std::vector<std::string> agent_entries;

    // Activity tab
    int log_filter = 0;
    std::vector<std::string> filter_labels = {"All", "Info", "Warn", "Error"};

    nlohmann::json performance_data = nlohmann::json::object();
    int64_t performance_refreshed_ms = 0;
    std::string performance_error;
    bool performance_force_refresh = true;
    // Chat tab
    int voice_agent_sel = 0;
    int voice_proposal_sel = 0;
    std::vector<std::string> voice_agent_entries;
    std::vector<std::string> voice_proposal_entries;
    nlohmann::json voice_state = nlohmann::json::object();
    int64_t voice_refreshed_ms = 0;
    std::atomic<bool> voice_force_refresh{true};
    std::atomic<bool> voice_busy{false};
    std::mutex voice_status_mutex;
    std::thread voice_thread;
    std::string voice_description;
    std::string voice_sample_text;
    std::string voice_speech_text;
    std::string voice_status;
    AudioPlayer audio_player;
    std::string voice_loaded_agent;

    int chat_agent_sel = 0;
    std::vector<std::string> chat_agent_entries;
    std::vector<AgentConfig> chat_agent_rows;   // per-frame snapshot for the rich menu rows
    std::string chat_input;
    std::vector<fs::path> chat_staged_paths;

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
    enum class FileBrowserTarget { Model, Projector, ChatImage };
    FileBrowserTarget fb_target = FileBrowserTarget::Model;

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

    auto set_node_operation_status = [&](std::string status) {
        std::lock_guard<std::mutex> lock(node_operation_mutex);
        node_operation_status = std::move(status);
    };

    auto summarize_node_operation = [](const std::string& operation,
                                       const NodeOperationResult& response) {
        if (!response.ok()) {
            return operation + " failed (" + std::to_string(response.status) + "): " +
                   response.error_message();
        }

        try {
            const auto& body = response.body;
            if (body.contains("lines") && body["lines"].is_array()) {
                std::string summary = operation + ":";
                for (const auto& line : body["lines"]) {
                    if (!line.is_string()) continue;
                    std::string value = line.get<std::string>();
                    if (value.size() > 240) value.resize(240);
                    summary += "\n" + value;
                    if (summary.size() >= 2400) {
                        summary.resize(2400);
                        summary += "...";
                        break;
                    }
                }
                if (body["lines"].empty()) summary += "\n(no log lines)";
                return summary;
            }

            const auto runtime = body.value("llama_runtime", nlohmann::json::object());
            if (runtime.is_object() && !runtime.empty()) {
                std::string summary = operation + ": " +
                    runtime.value("status", std::string{"unknown"});
                const std::string version = runtime.value("version", std::string{});
                const std::string accelerator = runtime.value("accelerator", std::string{});
                const std::string variant = runtime.value("variant", std::string{});
                const std::string latest = runtime.value("latest_version", std::string{});
                const std::string error = runtime.value("last_error", std::string{});
                if (!version.empty()) summary += " · " + version;
                if (!accelerator.empty()) summary += " · " + accelerator;
                if (!variant.empty() && variant != accelerator) summary += " · " + variant;
                if (!latest.empty()) {
                    summary += runtime.value("update_available", false)
                        ? "\nUpdate available: " + latest
                        : "\nLatest upstream: " + latest + " (current)";
                    const std::string update_action =
                        runtime.value("update_action", std::string{});
                    if (!update_action.empty()) summary += " · " + update_action;
                }
                const std::string update_warning =
                    runtime.value("update_warning", std::string{});
                if (!update_warning.empty()) summary += "\n" + update_warning;
                const auto troubleshooting = runtime.value(
                    "troubleshooting", nlohmann::json::object());
                const std::string diagnostic = troubleshooting.value(
                    "summary", std::string{});
                if (!diagnostic.empty()) summary += "\n" + diagnostic;
                if (!error.empty()) summary += "\n" + error;
                return summary;
            }

            if (body.contains("slots")) {
                const auto slots = body.value("slots", nlohmann::json::array());
                std::string summary = operation + ": " +
                    body.value("hostname", std::string{"node"}) + " · " +
                    std::to_string(slots.is_array() ? slots.size() : 0U) + " slots";
                const std::string runtime_status = body.value("llama_runtime",
                    nlohmann::json::object()).value("status", std::string{});
                if (!runtime_status.empty()) summary += " · runtime " + runtime_status;
                return summary;
            }

            if (body.contains("cancel_requested")) return operation + ": requested";
            std::string summary = operation + ": " + body.dump();
            if (summary.size() > 800) summary = summary.substr(0, 800) + "...";
            return summary;
        } catch (const std::exception& e) {
            return operation + " completed; response could not be summarized: " + e.what();
        }
    };

    auto run_node_operation_async = [&](std::string operation,
                                        NodeOperationsPtr node,
                                        std::function<NodeOperationResult(NodeOperations&)> fn) {
        if (node_operation_inflight.exchange(true)) {
            set_node_operation_status("another node operation is already running");
            return;
        }
        if (node_operation_thread.joinable()) node_operation_thread.join();
        {
            std::lock_guard<std::mutex> lock(active_node_operation_mutex);
            active_node_operation = node;
        }
        set_node_operation_status("running: " + operation);
        refresh();
        node_operation_thread = std::thread(
            [&, operation = std::move(operation), node = std::move(node), fn = std::move(fn)]() {
                try {
                    set_node_operation_status(summarize_node_operation(operation, fn(*node)));
                } catch (const std::exception& e) {
                    set_node_operation_status(operation + " failed: " + e.what());
                } catch (...) {
                    set_node_operation_status(operation + " failed: unknown error");
                }
                {
                    std::lock_guard<std::mutex> lock(active_node_operation_mutex);
                    if (active_node_operation == node) {
                        active_node_operation.reset();
                    }
                }
                node_operation_inflight = false;
                refresh();
            });
    };

    auto run_pairing_async = [&screen, &pairing_inflight, &pairing_thread,
                              &ui_shutting_down, &set_node_operation_status,
                              this](
        std::string operation,
        std::function<std::string()> work,
        std::function<void(const std::string&)> complete) {
        if (pairing_inflight.exchange(true)) {
            set_node_operation_status("another pairing operation is already running");
            return;
        }
        if (pairing_thread.joinable()) pairing_thread.join();
        set_node_operation_status("running: " + operation);
        refresh();
        pairing_thread = std::thread(
            [&, operation = std::move(operation), work = std::move(work),
             complete = std::move(complete)]() mutable {
                std::string result;
                std::string error;
                try {
                    result = work();
                } catch (const std::exception& exception) {
                    error = exception.what();
                } catch (...) {
                    error = "unknown error";
                }
                if (ui_shutting_down.load()) {
                    pairing_inflight = false;
                    return;
                }
                screen.Post([&, operation = std::move(operation),
                             result = std::move(result),
                             error = std::move(error),
                             complete = std::move(complete)]() mutable {
                    if (!error.empty()) {
                        set_node_operation_status(operation + " failed: " + error);
                    } else {
                        complete(result);
                    }
                    pairing_inflight = false;
                    refresh();
                });
            });
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
                const std::string ext = util::to_lower(e.path().extension().string());
                bool g = e.is_regular_file() &&
                    (fb_target == FileBrowserTarget::ChatImage
                        ? (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
                        : ext == ".gguf");
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
        if (!ed_id_orig.empty()) {
            if (auto current = agents_.get_agent(ed_id_orig)) cfg = current->get_config();
        }

        cfg.id = ed_id;
        cfg.name = ed_name;
        cfg.model_path = ed_model;
        cfg.system_prompt = ed_sysprompt;
        cfg.preferred_node_id = ed_pref_node;
        cfg.inference_backend = ed_backend == 1
            ? "api"
            : (ed_backend >= 2 && !ed_unsupported_backend.empty()
                ? ed_unsupported_backend
                : "llama-cpp");
        // Generation (shared request contract → RuntimeSettings)
        try { cfg.runtime_settings.temperature = std::stof(ed_temp_s); } catch (...) {}
        try { cfg.runtime_settings.top_p = std::stof(ed_topp_s); } catch (...) {}
        try { cfg.runtime_settings.top_k = std::stoi(ed_topk_s); } catch (...) {}
        try { cfg.runtime_settings.min_p = std::stof(ed_minp_s); } catch (...) {}
        try { cfg.runtime_settings.presence_penalty = std::stof(ed_presence_s); } catch (...) {}
        try { cfg.runtime_settings.repeat_penalty = std::stof(ed_repeat_s); } catch (...) {}
        try { cfg.runtime_settings.max_tokens = std::stoi(ed_max_s); } catch (...) {}
        try { cfg.runtime_settings.ctx_size = std::stoi(ed_ctx_s); } catch (...) {}
        try { cfg.runtime_settings.n_gpu_layers = std::stoi(ed_gpu_layers_s); } catch (...) {}
        try { cfg.runtime_settings.n_threads = std::stoi(ed_threads_s); } catch (...) {}
        try { cfg.runtime_settings.n_threads_http = std::stoi(ed_threads_http_s); } catch (...) {}
        try { cfg.runtime_settings.parallel = std::stoi(ed_parallel_s); } catch (...) {}
        try { cfg.runtime_settings.batch_size = std::stoi(ed_batch_s); } catch (...) {}
        try { cfg.runtime_settings.ubatch_size = std::stoi(ed_ubatch_s); } catch (...) {}
        cfg.runtime_settings.flash_attn = ed_flash;
        cfg.served_model_name = ed_served_model;
        cfg.api_settings.base_url = ed_api_base;
        cfg.api_settings.chat_completions_path = ed_api_path;
        cfg.api_settings.api_key_env = ed_api_key_env;
        if (!ed_api_key.empty()) cfg.api_settings.api_key = ed_api_key;
        cfg.vision_settings.enabled = ed_vision;
        cfg.vision_settings.mmproj_path = ed_mmproj;

        cfg.reasoning_enabled = ed_reasoning;
        cfg.memories_enabled = ed_memories;
        cfg.tools_enabled = ed_tools;

        cfg.runtime_settings.extra_args.clear();
        size_t llama_start = 0;
        while (llama_start <= ed_llama_extra_args_text.size()) {
            const size_t end = ed_llama_extra_args_text.find('\n', llama_start);
            std::string line = util::trim(ed_llama_extra_args_text.substr(
                llama_start, end == std::string::npos ? std::string::npos : end - llama_start));
            if (!line.empty()) cfg.runtime_settings.extra_args.push_back(std::move(line));
            if (end == std::string::npos) break;
            llama_start = end + 1;
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
            '\n' + ed_temp_s + '\n' + ed_topp_s + '\n' + ed_topk_s + '\n' +
            ed_minp_s + '\n' + ed_presence_s + '\n' + ed_repeat_s + '\n' + ed_max_s + '\n' +
            std::to_string(ed_backend) + '\n' + ed_unsupported_backend + '\n' +
            ed_ctx_s + '\n' + ed_gpu_layers_s + '\n' +
            ed_threads_s + '\n' + ed_threads_http_s + '\n' + ed_parallel_s + '\n' + ed_batch_s + '\n' +
            ed_ubatch_s + '\n' + ed_llama_extra_args_text + '\n' + ed_served_model + '\n' +
            ed_api_base + '\n' + ed_api_path + '\n' + ed_api_key + '\n' + ed_api_key_env + '\n' +
            (ed_vision ? "1" : "0") + '\n' + ed_mmproj + '\n' +
            (ed_flash ? "1" : "0") +
            (ed_reasoning ? "1" : "0") + (ed_memories ? "1" : "0") + (ed_tools ? "1" : "0");
        if (signature == ed_validation_signature) return;
        ed_validation_signature = signature;
        auto validation = validate_agent_config(build_editor_cfg(), &registry_, models_dir_, &ed_model_info);
        ed_validation_issues = std::move(validation.issues);
    };

    //

    // Discovered (unregistered) nodes section. AIO keeps this entire surface
    // out of the component tree when clustering is disabled, so hidden pairing
    // buttons cannot receive keyboard focus or invoke remote-node mutations.
    const bool remote_nodes_enabled = registry_.remote_nodes_enabled();
    auto disc_menu    = Menu(&disc_entries, &disc_sel, MenuOption::Vertical());
    auto btn_pair     = Button("[P] Pair", [&] {
        if (!remote_nodes_enabled) {
            set_node_operation_status("clustering is disabled; enable [cluster].enabled and restart");
            return;
        }
        auto dns = registry_.get_discovered_nodes();
        if (disc_sel < 0 || disc_sel >= static_cast<int>(dns.size())) return;
        pair_url = dns[disc_sel].url;
        remember_pair_node = true;
        if (!pairing_key_.empty()) {
            const std::string url = pair_url;
            const std::string psk = pairing_key_;
            const bool remember = remember_pair_node;
            run_pairing_async(
                "pair " + url,
                [this, url, psk, remember] {
                    return registry_.pair_node(url, psk, remember);
                },
                [this, url, remember](const std::string& key) {
                    if (!key.empty()) {
                        log(LogLevel::Info,
                            "Paired with " + url + " (PSK)" +
                            (remember ? " and remembered" : ""));
                    } else {
                        log(LogLevel::Error, "PSK pairing failed for " + url);
                    }
                });
        } else {
            const std::string url = pair_url;
            run_pairing_async(
                "request pairing PIN from " + url,
                [this, url] { return registry_.start_pair(url); },
                [&, url](const std::string& nonce) {
                    if (nonce.empty()) {
                        log(LogLevel::Error,
                            "Could not reach node for pairing: " + url);
                        return;
                    }
                    pair_url = url;
                    pair_nonce = nonce;
                    pin_input.clear();
                    show_pin_entry = true;
                });
        }
    }, ButtonOption::Simple());
    auto btn_add_manual = Button("[+] Add Manually", [&] {
        if (!remote_nodes_enabled) {
            set_node_operation_status("clustering is disabled; enable [cluster].enabled and restart");
            return;
        }
        add_url.clear();
        remember_pair_node = true;
        show_add_node = true;
    }, ButtonOption::Simple());
    auto disc_btns    = Container::Horizontal({btn_pair, btn_add_manual});
    // Maybe wrapper: disc_menu is excluded from the component tree (events + render)
    // when disc_entries is empty, preventing FTXUI from indexing an empty vector.
    auto disc_menu_m  = Maybe(disc_menu, [&]() { return !disc_entries.empty(); });
    auto disc_comp    = Container::Vertical({disc_menu_m, disc_btns});
    auto disc_comp_m  = Maybe(disc_comp, [&]() { return remote_nodes_enabled; });

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

        std::string display_name = mm::tui::node_display_name(n, node_rows);
        if (n.kind == "embedded" || n.id == "local") display_name += " [embedded]";
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
    auto selected_control_node = [&]() -> std::optional<NodeInfo> {
        if (node_sel >= 0 && node_sel < static_cast<int>(node_rows.size()))
            return node_rows[static_cast<std::size_t>(node_sel)];
        const auto nodes = registry_.list_nodes();
        if (node_sel < 0 || node_sel >= static_cast<int>(nodes.size())) return std::nullopt;
        return nodes[static_cast<std::size_t>(node_sel)];
    };
    auto selected_node_is_embedded = [&]() {
        const auto node = selected_control_node();
        return node && (node->kind == "embedded" || node->id == "local");
    };
    auto forget_selected_node = [&]() {
        const auto node = selected_control_node();
        if (!node) return;
        if (node->kind == "embedded" || node->id == "local") {
            set_node_operation_status("the embedded node is process-owned and cannot be forgotten");
            return;
        }
        try {
            const bool changed = registry_.forget_node(node->id);
            log(LogLevel::Info,
                changed ? ("Forgot saved node " + node->id)
                        : ("Node " + node->id + " was not remembered"));
        } catch (const std::exception& e) {
            set_node_operation_status(std::string{"forget failed: "} + e.what());
        }
    };
    auto btn_forget_n = Button("[F] Forget", forget_selected_node, ButtonOption::Simple());
    auto btn_rem_n   = Button("[-] Remove", [&] {
        const auto node = selected_control_node();
        if (!node) return;
        if (node->kind == "embedded" || node->id == "local") {
            set_node_operation_status("the embedded node is process-owned and cannot be removed");
            return;
        }
        try {
            registry_.remove_node(node->id);
            const auto remaining = registry_.list_nodes().size();
            if (node_sel >= static_cast<int>(remaining))
                node_sel = std::max(0, static_cast<int>(remaining) - 1);
        } catch (const std::exception& e) {
            set_node_operation_status(std::string{"remove failed: "} + e.what());
        }
    }, ButtonOption::Simple());
    auto node_menu_m  = Maybe(node_menu, [&]() { return !node_entries.empty(); });
    auto node_btns    = Container::Horizontal({btn_forget_n, btn_rem_n});
    auto node_btns_m  = Maybe(node_btns, [&]() {
        const auto node = selected_control_node();
        return node && node->kind != "embedded" && node->id != "local";
    });
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

    auto nodes_section = Container::Vertical({node_menu_m, node_btns_m});

    auto selected_node_operations = [&]() -> NodeOperationsPtr {
        const auto node = selected_control_node();
        if (!node || !node->connected) {
            set_node_operation_status("node is not online");
            return {};
        }
        try {
            return registry_.operations(node->id);
        } catch (const std::exception& e) {
            set_node_operation_status(std::string{"node operation unavailable: "} + e.what());
            return {};
        }
    };
    auto cancel_selected_node_action = [&]() {
        const auto operations = selected_node_operations();
        if (!operations) return false;
        const auto response = operations->cancel_action();
        set_node_operation_status(summarize_node_operation("cancel action", response));
        return response.ok();
    };
    auto node_slot_menu = Menu(&node_slot_entries, &node_slot_sel, MenuOption::Vertical());
    auto btn_control_status = Button(" Status ", [&] {
        auto operations = selected_node_operations();
        if (operations) run_node_operation_async("node status", std::move(operations),
            [](NodeOperations& node) { return node.status(); });
    }, ButtonOption::Simple());
    auto btn_control_logs = Button(" Log tail ", [&] {
        auto operations = selected_node_operations();
        if (operations) run_node_operation_async("last 50 node log lines", std::move(operations),
            [](NodeOperations& node) { return node.logs(50); });
    }, ButtonOption::Simple());
    auto btn_control_runtime = Button(" Runtime ", [&] {
        auto operations = selected_node_operations();
        if (operations) run_node_operation_async("llama.cpp runtime", std::move(operations),
            [](NodeOperations& node) { return node.llama_runtime(); });
    }, ButtonOption::Simple());
    auto btn_control_diagnose = Button(" Diagnose ", [&] {
        auto operations = selected_node_operations();
        if (operations) run_node_operation_async("llama.cpp diagnosis", std::move(operations),
            [](NodeOperations& node) { return node.llama_diagnose(); });
    }, ButtonOption::Simple());
    auto btn_control_cancel = Button(" Cancel action ", [&] {
        cancel_selected_node_action();
    }, ButtonOption::Simple());

    auto open_node_network_confirmation = [&](NodeRuntimeNetworkAction action) {
        const auto node = selected_control_node();
        if (!node || !node->connected) {
            set_node_operation_status("node is not online");
            return;
        }
        pending_node_network_action = action;
        pending_node_network_id = node->id;
        node_runtime_accelerator_input.clear();
        node_runtime_variant_input.clear();
        node_runtime_variant_hint.clear();
        node_runtime_recovery_action = 0;

        auto append_variant = [&](const LlamaRuntimeVariant& variant,
                                  bool release_only) {
            if (variant.id.empty() || !variant.platform_supported) return;
            if (release_only && !variant.release_available) return;
            if (!release_only && !variant.release_available && !variant.source_supported) return;
            if (!node_runtime_variant_hint.empty()) node_runtime_variant_hint += ", ";
            node_runtime_variant_hint += variant.id;
            if (node_runtime_variant_hint.size() > 120) {
                node_runtime_variant_hint.resize(120);
                node_runtime_variant_hint += "...";
            }
        };

        if (action == NodeRuntimeNetworkAction::Switch) {
            for (const auto& variant : node->llama_runtime.available_variants) {
                append_variant(variant, false);
                if (node_runtime_variant_input.empty() && variant.recommended &&
                    (variant.release_available || variant.source_supported)) {
                    node_runtime_variant_input = variant.id;
                }
            }
            if (node_runtime_variant_input.empty()) {
                node_runtime_variant_input = !node->llama_runtime.target_variant.empty()
                    ? node->llama_runtime.target_variant
                    : node->llama_runtime.variant;
            }
        } else if (action == NodeRuntimeNetworkAction::Recover) {
            for (const auto& variant : node->llama_runtime.troubleshooting.variants)
                append_variant(variant, true);
        }
        show_node_network_confirm = true;
    };

    auto btn_control_provision = Button(" Provision... ", [&] {
        open_node_network_confirmation(NodeRuntimeNetworkAction::Provision);
    }, ButtonOption::Simple());
    auto btn_control_check_update = Button(" Check update... ", [&] {
        open_node_network_confirmation(NodeRuntimeNetworkAction::CheckUpdate);
    }, ButtonOption::Simple());
    auto btn_control_update = Button(" Update... ", [&] {
        open_node_network_confirmation(NodeRuntimeNetworkAction::Update);
    }, ButtonOption::Simple());
    auto btn_control_switch = Button(" Switch... ", [&] {
        open_node_network_confirmation(NodeRuntimeNetworkAction::Switch);
    }, ButtonOption::Simple());
    auto btn_control_recover = Button(" Recover... ", [&] {
        open_node_network_confirmation(NodeRuntimeNetworkAction::Recover);
    }, ButtonOption::Simple());
    auto control_read_buttons = Container::Horizontal({btn_control_status, btn_control_logs,
        btn_control_runtime, btn_control_diagnose, btn_control_cancel});
    auto control_runtime_buttons = Container::Horizontal({btn_control_provision,
        btn_control_check_update, btn_control_update, btn_control_switch,
        btn_control_recover});
    auto control_action_buttons = Container::Vertical(
        {control_read_buttons, control_runtime_buttons});
    auto node_operations = Container::Vertical({node_slot_menu, control_action_buttons});

    auto nodes_comp  = Container::Vertical({disc_comp_m, nodes_section, node_name_divider,
        node_rows_divider, node_operations});

    //

    InputOption url_iopt; url_iopt.multiline = false;
    url_iopt.placeholder = "http://hostname:7070";
    auto modal_url    = Input(&add_url, url_iopt);
    auto modal_remember_cb = Checkbox("Remember this node", &remember_pair_node);
    auto modal_ok     = Button("  Connect  ", [&] {
        if (!add_url.empty()) {
            const std::string url = add_url;
            const bool remember = remember_pair_node;
            if (!pairing_key_.empty()) {
                const std::string psk = pairing_key_;
                run_pairing_async(
                    "pair " + url,
                    [this, url, psk, remember] {
                        return registry_.pair_node(url, psk, remember);
                    },
                    [this, url, remember](const std::string& key) {
                        if (!key.empty()) {
                            log(LogLevel::Info,
                                "Paired with " + url + " (PSK)" +
                                (remember ? " and remembered" : ""));
                        } else {
                            log(LogLevel::Error,
                                "PSK pairing failed for " + url);
                        }
                    });
            } else {
                run_pairing_async(
                    "request pairing PIN from " + url,
                    [this, url] { return registry_.start_pair(url); },
                    [&, url](const std::string& nonce) {
                        if (nonce.empty()) {
                            log(LogLevel::Error,
                                "Could not reach node for pairing: " + url);
                            return;
                        }
                        pair_url = url;
                        pair_nonce = nonce;
                        pin_input.clear();
                        show_pin_entry = true;
                    });
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
            const std::string url = pair_url;
            const std::string nonce = pair_nonce;
            const std::string pin = pin_input;
            const bool remember = remember_pair_node;
            run_pairing_async(
                "complete pairing with " + url,
                [this, url, nonce, pin, remember] {
                    return registry_.complete_pair(
                        url, nonce, pin, remember);
                },
                [this, url, remember](const std::string& key) {
                    if (!key.empty()) {
                        log(LogLevel::Info,
                            "Paired with " + url +
                            (remember ? " and remembered" : ""));
                    } else {
                        log(LogLevel::Error,
                            "PIN pairing failed for " + url);
                    }
                });
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

    // Runtime discovery, installation, switching, and recovery can query a
    // release service, download artifacts, or compile llama.cpp. Every such
    // operation comes through this one-shot consent modal and receives an
    // explicit allow_network flag. Standalone remote nodes simply ignore the
    // additive field while retaining their existing REST behavior.
    auto node_network_operation_label = [&]() -> std::string {
        switch (pending_node_network_action) {
            case NodeRuntimeNetworkAction::Provision:   return "provision llama.cpp";
            case NodeRuntimeNetworkAction::CheckUpdate: return "check for llama.cpp updates";
            case NodeRuntimeNetworkAction::Update:      return "update llama.cpp";
            case NodeRuntimeNetworkAction::Switch:      return "switch llama.cpp engine";
            case NodeRuntimeNetworkAction::Recover:     return "recover llama.cpp runtime";
        }
        return "run llama.cpp operation";
    };

    InputOption node_runtime_accelerator_option;
    node_runtime_accelerator_option.multiline = false;
    node_runtime_accelerator_option.placeholder = "optional: cuda, vulkan, cpu, ...";
    auto node_runtime_accelerator_input_comp =
        Input(&node_runtime_accelerator_input, node_runtime_accelerator_option);
    auto node_runtime_accelerator_field = Renderer(
        node_runtime_accelerator_input_comp, [&] {
            return hbox({text(" Accelerator: "),
                         node_runtime_accelerator_input_comp->Render() | flex});
        });
    auto node_runtime_accelerator_field_maybe = Maybe(
        node_runtime_accelerator_field, [&] {
            return pending_node_network_action == NodeRuntimeNetworkAction::Update;
        });

    auto node_runtime_recovery_toggle = Toggle(
        &node_runtime_recovery_labels, &node_runtime_recovery_action);
    auto node_runtime_recovery_field = Renderer(node_runtime_recovery_toggle, [&] {
        return vbox({text(" Recovery action: ") | dim,
                     node_runtime_recovery_toggle->Render()});
    });
    auto node_runtime_recovery_field_maybe = Maybe(
        node_runtime_recovery_field, [&] {
            return pending_node_network_action == NodeRuntimeNetworkAction::Recover;
        });

    InputOption node_runtime_variant_option;
    node_runtime_variant_option.multiline = false;
    node_runtime_variant_option.placeholder = "variant id";
    auto node_runtime_variant_input_comp =
        Input(&node_runtime_variant_input, node_runtime_variant_option);
    auto node_runtime_variant_field = Renderer(node_runtime_variant_input_comp, [&] {
        return hbox({text(" Variant: "), node_runtime_variant_input_comp->Render() | flex});
    });
    auto node_runtime_variant_field_maybe = Maybe(
        node_runtime_variant_field, [&] {
            return pending_node_network_action == NodeRuntimeNetworkAction::Switch ||
                   (pending_node_network_action == NodeRuntimeNetworkAction::Recover &&
                    node_runtime_recovery_action == 3);
        });

    auto node_network_confirm = Button(" Allow network and run ", [&] {
        nlohmann::json request = {{"allow_network", true}};
        std::string operation = node_network_operation_label();
        std::function<NodeOperationResult(NodeOperations&)> fn;

        switch (pending_node_network_action) {
            case NodeRuntimeNetworkAction::Provision:
                fn = [request](NodeOperations& node) {
                    return node.llama_provision(request);
                };
                break;
            case NodeRuntimeNetworkAction::CheckUpdate:
                fn = [request](NodeOperations& node) {
                    return node.llama_check_update(request);
                };
                break;
            case NodeRuntimeNetworkAction::Update: {
                const std::string accelerator =
                    util::trim(node_runtime_accelerator_input);
                if (!accelerator.empty()) request["accelerator"] = accelerator;
                fn = [request](NodeOperations& node) {
                    return node.llama_update(request);
                };
                break;
            }
            case NodeRuntimeNetworkAction::Switch: {
                const std::string variant = util::trim(node_runtime_variant_input);
                if (variant.empty()) {
                    set_node_operation_status("switch requires a runtime variant id");
                    return;
                }
                request["variant"] = variant;
                fn = [request](NodeOperations& node) {
                    return node.llama_switch(request);
                };
                break;
            }
            case NodeRuntimeNetworkAction::Recover: {
                const int action_index = std::clamp(
                    node_runtime_recovery_action, 0,
                    static_cast<int>(node_runtime_recovery_labels.size()) - 1);
                const std::string action =
                    node_runtime_recovery_labels[static_cast<std::size_t>(action_index)];
                request["action"] = action;
                if (action == "release") {
                    const std::string variant = util::trim(node_runtime_variant_input);
                    if (variant.empty()) {
                        set_node_operation_status(
                            "release recovery requires a published variant id");
                        return;
                    }
                    request["variant"] = variant;
                }
                fn = [request](NodeOperations& node) {
                    return node.llama_recover(request);
                };
                break;
            }
        }

        try {
            auto operations = registry_.operations(pending_node_network_id);
            show_node_network_confirm = false;
            pending_node_network_id.clear();
            run_node_operation_async(std::move(operation), std::move(operations),
                                     std::move(fn));
        } catch (const std::exception& e) {
            set_node_operation_status(
                std::string{"runtime operation unavailable: "} + e.what());
        }
    }, ButtonOption::Simple());
    auto node_network_cancel = Button(" Cancel ", [&] {
        show_node_network_confirm = false;
        pending_node_network_id.clear();
    }, ButtonOption::Simple());
    auto node_network_buttons = Container::Horizontal({node_network_confirm,
                                                        node_network_cancel});
    auto node_network_controls = Container::Vertical({
        node_runtime_accelerator_field_maybe,
        node_runtime_recovery_field_maybe,
        node_runtime_variant_field_maybe,
        node_network_buttons,
    });
    auto node_network_modal_renderer = Renderer(node_network_controls, [&]() {
        Elements rows = {
            text(" Runtime Network Consent ") | bold | hcenter,
            separator(),
            text("Operation: " + node_network_operation_label()) | bold,
            text("Node: " + pending_node_network_id) | dim,
        };
        if (pending_node_network_action == NodeRuntimeNetworkAction::Update)
            rows.push_back(node_runtime_accelerator_field->Render());
        if (pending_node_network_action == NodeRuntimeNetworkAction::Recover)
            rows.push_back(node_runtime_recovery_field->Render());
        if (pending_node_network_action == NodeRuntimeNetworkAction::Switch ||
            (pending_node_network_action == NodeRuntimeNetworkAction::Recover &&
             node_runtime_recovery_action == 3)) {
            rows.push_back(node_runtime_variant_field->Render());
            if (!node_runtime_variant_hint.empty())
                rows.push_back(paragraph("Available: " + node_runtime_variant_hint) | dim);
        }
        rows.push_back(paragraph(
            pending_node_network_action == NodeRuntimeNetworkAction::CheckUpdate
                ? "This check contacts the upstream release service."
                : "This action may query releases, download artifacts or source code, and "
                  "run a local compilation."));
        rows.push_back(paragraph(
            "Only this operation receives allow_network: true; automatic checks remain "
            "disabled under the prompt policy.") | color(Color::Yellow));
        rows.push_back(separator());
        rows.push_back(node_network_buttons->Render() | hcenter);
        return vbox(std::move(rows)) | border | size(WIDTH, EQUAL, 72);
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
        Element flags = hbox({flag(a.vision_settings.enabled, "V"), text(" "),
                              flag(a.reasoning_enabled, "R"), text(" "),
                              flag(a.memories_enabled, "M"), text(" "),
                              flag(a.tools_enabled, "T")});
        std::string node = a.preferred_node_id.empty() ? "auto" : a.preferred_node_id;
        if (node.rfind("node-", 0) == 0) node = node.substr(5);
        Element row = hbox({
            mm::tui::col(text(a.name) | bold, 14), sep(),
            mm::tui::col(text(mm::tui::short_model(a.model_path, 26)), 26), sep(),
            mm::tui::col(std::move(flags), 9), sep(),
            mm::tui::col(text(node) | (a.preferred_node_id.empty() ? dim : color(Color::Cyan)), 10),
        });
        if (st.active) row = std::move(row) | inverted;
        return row;
    };
    auto agent_menu   = Menu(&agent_entries, &agent_sel, agent_menu_opt);
    auto btn_new_a    = Button("[+] New", [&] {
        ed_id.clear(); ed_id_orig.clear(); ed_name = "New Agent"; ed_model.clear();
        ed_sysprompt.clear(); ed_pref_node.clear();
        ed_temp_s = "0.70"; ed_topp_s = "0.90"; ed_topk_s = "-1";
        ed_minp_s = "-1.00"; ed_presence_s = "0.00"; ed_repeat_s = "-1.00";
        ed_max_s = "1024";
        ed_backend = 0;
        ed_backend_labels = {"llama.cpp", "Remote API"};
        ed_unsupported_backend.clear();
        ed_ctx_s = "4096"; ed_gpu_layers_s = "-1"; ed_threads_s = "-1";
        ed_threads_http_s = "-1"; ed_parallel_s = "1"; ed_batch_s = "-1"; ed_ubatch_s = "-1";
        ed_llama_extra_args_text.clear(); ed_flash = true;
        ed_served_model.clear();
        ed_api_base = "https://api.openai.com"; ed_api_path = "/v1/chat/completions";
        ed_api_key.clear(); ed_api_key_env = "OPENAI_API_KEY";
        ed_vision = false; ed_mmproj.clear();
        ed_reasoning = false; ed_memories = true; ed_tools = false;
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
            ed_topk_s = std::to_string(c.runtime_settings.top_k);
            snprintf(tmp, sizeof(tmp), "%.2f", static_cast<double>(c.runtime_settings.min_p));
            ed_minp_s = tmp;
            snprintf(tmp, sizeof(tmp), "%.2f", static_cast<double>(c.runtime_settings.presence_penalty));
            ed_presence_s = tmp;
            snprintf(tmp, sizeof(tmp), "%.2f", static_cast<double>(c.runtime_settings.repeat_penalty));
            ed_repeat_s = tmp;
            ed_max_s = std::to_string(c.runtime_settings.max_tokens);
            const std::string backend = util::to_lower(c.inference_backend);
            ed_backend_labels = {"llama.cpp", "Remote API"};
            ed_unsupported_backend.clear();
            if (backend.empty() || backend == "llama-cpp" || backend == "llama.cpp" ||
                backend == "llama") {
                ed_backend = 0;
            } else if (backend == "api") {
                ed_backend = 1;
            } else {
                ed_unsupported_backend = backend;
                ed_backend_labels.push_back("Unsupported: " + backend);
                ed_backend = 2;
            }
            ed_ctx_s = std::to_string(c.runtime_settings.ctx_size);
            ed_gpu_layers_s = std::to_string(c.runtime_settings.n_gpu_layers);
            ed_threads_s = std::to_string(c.runtime_settings.n_threads);
            ed_threads_http_s = std::to_string(c.runtime_settings.n_threads_http);
            ed_parallel_s = std::to_string(c.runtime_settings.parallel);
            ed_batch_s = std::to_string(c.runtime_settings.batch_size);
            ed_ubatch_s = std::to_string(c.runtime_settings.ubatch_size);
            ed_flash = c.runtime_settings.flash_attn;
            ed_llama_extra_args_text = util::join(c.runtime_settings.extra_args, "\n");
            ed_served_model = c.served_model_name;
            ed_api_base = c.api_settings.base_url;
            ed_api_path = c.api_settings.chat_completions_path;
            ed_api_key.clear();
            ed_api_key_env = c.api_settings.api_key_env;
            ed_vision = c.vision_settings.enabled;
            ed_mmproj = c.vision_settings.mmproj_path;

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
            scheduler_.release_agent(cs[agent_sel].id);
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
    bool ed_open_engine   = true;
    bool ed_open_caps     = false;
    int  ed_split         = layout.get("agents.editor_width", 26, 18, 46);

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
    auto ed_inp_topk  = Input(&ed_topk_s,    sl);
    auto ed_inp_minp  = Input(&ed_minp_s,    sl);
    auto ed_inp_presence = Input(&ed_presence_s, sl);
    auto ed_inp_repeat = Input(&ed_repeat_s, sl);
    auto ed_inp_max   = Input(&ed_max_s,     sl);
    auto ed_backend_toggle = Toggle(&ed_backend_labels, &ed_backend);
    auto ed_inp_ctx = Input(&ed_ctx_s, sl);
    auto ed_inp_gpu_layers = Input(&ed_gpu_layers_s, sl);
    auto ed_inp_threads = Input(&ed_threads_s, sl);
    auto ed_inp_threads_http = Input(&ed_threads_http_s, sl);
    auto ed_inp_parallel = Input(&ed_parallel_s, sl);
    auto ed_inp_batch = Input(&ed_batch_s, sl);
    auto ed_inp_ubatch = Input(&ed_ubatch_s, sl);
    auto ed_inp_llama_extra = Input(&ed_llama_extra_args_text, ml);
    auto ed_cb_flash = Checkbox("flash_attention", &ed_flash);

    auto ed_inp_served_model = Input(&ed_served_model, sl);
    auto ed_inp_api_base = Input(&ed_api_base, sl);
    auto ed_inp_api_path = Input(&ed_api_path, sl);
    InputOption secret; secret.multiline = false; secret.password = true;
    secret.placeholder = "leave blank to retain current key";
    auto ed_inp_api_key = Input(&ed_api_key, secret);
    auto ed_inp_api_key_env = Input(&ed_api_key_env, sl);
    auto ed_inp_mmproj = Input(&ed_mmproj, sl);
    auto ed_cb_vision = Checkbox("Vision", &ed_vision);

    auto ed_cb_reasoning = Checkbox("Reasoning",    &ed_reasoning);
    auto ed_cb_memories  = Checkbox("Memories",     &ed_memories);
    auto ed_cb_tools     = Checkbox("Tools",        &ed_tools);

    auto btn_browse_model = Button("[Browse]", [&] {
        fb_target = FileBrowserTarget::Model;
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
    auto btn_browse_mmproj = Button("[Browse]", [&] {
        fb_target = FileBrowserTarget::Projector;
        fs::path init;
        if (!ed_mmproj.empty()) {
            const fs::path projector(ed_mmproj);
            init = fs::exists(projector.parent_path())
                ? projector.parent_path() : fs::current_path();
        } else if (!ed_model.empty() && fs::exists(fs::path(ed_model).parent_path())) {
            init = fs::path(ed_model).parent_path();
        } else {
            init = fs::current_path();
        }
        fb_current_path = init;
        refresh_fb();
        show_file_browser = true;
    }, ButtonOption::Simple());
    auto mmproj_row = Container::Horizontal({ed_inp_mmproj, btn_browse_mmproj});

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
            if (ed_id_orig.empty()) {
                agents_.create_agent(cfg);
            } else {
                auto current = agents_.get_agent(ed_id_orig);
                if (!current) {
                    throw std::runtime_error("Agent no longer exists: " + ed_id_orig);
                }
                const AgentConfig old_cfg = current->get_config();
                current.reset();
                auto normalized_backend = [](const AgentConfig& candidate) {
                    std::string backend = util::to_lower(
                        util::trim(candidate.inference_backend));
                    if (backend.empty() || backend == "llama" ||
                        backend == "llama.cpp" || backend == "llama-cpp") {
                        return std::string{"llama-cpp"};
                    }
                    return backend;
                };
                const bool id_changed = !cfg.id.empty() && cfg.id != ed_id_orig;
                if (id_changed && agents_.get_agent(cfg.id)) {
                    throw std::invalid_argument(
                        "Agent ID '" + cfg.id + "' is already in use");
                }
                const bool local_to_api =
                    normalized_backend(old_cfg) == "llama-cpp" &&
                    normalized_backend(cfg) == "api";
                if (id_changed || local_to_api) {
                    scheduler_.release_agent(ed_id_orig);
                }
                if (agents_.update_agent(ed_id_orig, cfg).empty()) {
                    throw std::runtime_error("Agent no longer exists: " + ed_id_orig);
                }
            }
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
            [&] { return "temp " + ed_temp_s + " · top_p " + ed_topp_s +
                         " · top_k " + ed_topk_s + " · " + ed_max_s + " tok"; }));
    auto ed_sec_engine_btn = Button("Engine", [&] { ed_open_engine = !ed_open_engine; },
        section_header(&ed_open_engine, "Runtime",
            [&] {
                if (ed_backend == 1) return "remote API · " + ed_api_base;
                if (ed_backend >= 2) return "unsupported · " + ed_unsupported_backend;
                return "llama.cpp · ctx " + ed_ctx_s + " · parallel " + ed_parallel_s;
            }));
    auto ed_sec_caps_btn = Button("Caps", [&] { ed_open_caps = !ed_open_caps; },
        section_header(&ed_open_caps, "Capabilities & options",
            [&] { return std::string("tools ") + (ed_tools ? "y" : "n") + " · reason " +
                         (ed_reasoning ? "y" : "n") + " · mem " + (ed_memories ? "y" : "n"); }));

    // Section field groups, Maybe-gated so collapsed sections leave the focus tree.
    auto sampling_fields = Container::Vertical({
        ed_inp_temp, ed_inp_topp, ed_inp_topk, ed_inp_minp,
        ed_inp_presence, ed_inp_repeat, ed_inp_max});
    auto sampling_m = Maybe(sampling_fields, [&] { return ed_open_sampling; });
    auto engine_fields = Container::Vertical({
        ed_backend_toggle,
        ed_inp_ctx, ed_inp_gpu_layers, ed_inp_threads, ed_inp_threads_http,
        ed_inp_parallel, ed_inp_batch, ed_inp_ubatch, ed_cb_flash, ed_inp_llama_extra,
        ed_cb_vision, mmproj_row,
        ed_inp_served_model, ed_inp_api_base, ed_inp_api_path, ed_inp_api_key_env, ed_inp_api_key});
    auto engine_m = Maybe(engine_fields, [&] { return ed_open_engine; });
    auto caps_fields = Container::Vertical({ed_cb_reasoning, ed_cb_memories, ed_cb_tools});
    auto caps_m = Maybe(caps_fields, [&] { return ed_open_caps; });

    auto editor_comp  = Container::Vertical({
        ed_divider, ed_split_row,
        ed_inp_id, ed_inp_name, model_row, ed_inp_pnode, ed_inp_sys,
        ed_sec_sampling_btn, sampling_m,
        ed_sec_engine_btn, engine_m,
        ed_sec_caps_btn, caps_m,
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
        if (fb_target == FileBrowserTarget::Model) {
            ed_model = chosen.string();
            refresh_editor_validation();
        } else if (fb_target == FileBrowserTarget::Projector) {
            ed_mmproj = chosen.string();
            refresh_editor_validation();
        } else {
            std::error_code ec;
            const auto size = fs::file_size(chosen, ec);
            int64_t total = 0;
            for (const auto& path : chat_staged_paths) {
                std::error_code size_ec;
                total += static_cast<int64_t>(fs::file_size(path, size_ec));
            }
            if (ec || size > 50ULL * 1024 * 1024 ||
                chat_staged_paths.size() >= 8 ||
                total + static_cast<int64_t>(size) > 400LL * 1024 * 1024) {
                std::lock_guard<std::mutex> lk(chat_mutex);
                chat_last_error = "Cannot stage image: 8 images, 50 MiB each, 400 MiB total maximum";
            } else if (std::find(chat_staged_paths.begin(), chat_staged_paths.end(), chosen) ==
                       chat_staged_paths.end()) {
                chat_staged_paths.push_back(chosen);
            }
        }
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
                                     : (fb_target == FileBrowserTarget::ChatImage
                                            ? "  (no JPEG/PNG files or subdirectories here)"
                                            : "  (no .gguf files or subdirectories here)");
        return vbox({
            text(fb_target == FileBrowserTarget::ChatImage
                     ? " Select JPEG/PNG Image "
                     : fb_target == FileBrowserTarget::Projector
                         ? " Select GGUF Projector " : " Select Model File ") | bold | hcenter,
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
    auto selected_voice_agent = [&]() -> std::string {
        const auto configs = agents_.list_agents();
        if (voice_agent_sel < 0 || voice_agent_sel >= static_cast<int>(configs.size())) return {};
        return configs[static_cast<std::size_t>(voice_agent_sel)].id;
    };
    auto selected_voice_proposal = [&]() -> nlohmann::json {
        const auto proposals = voice_state.value("proposals", nlohmann::json::array());
        if (voice_proposal_sel < 0 || voice_proposal_sel >= static_cast<int>(proposals.size()))
            return nlohmann::json::object();
        return proposals[static_cast<std::size_t>(voice_proposal_sel)];
    };
    auto set_voice_status = [&](const std::string& value) {
        std::lock_guard<std::mutex> lock(voice_status_mutex);
        voice_status = value;
    };
    auto run_voice_async = [&](const std::string& operation, std::function<std::string()> fn) {
        if (voice_busy.exchange(true)) return;
        if (voice_thread.joinable()) voice_thread.join();
        set_voice_status("running: " + operation);
        voice_thread = std::thread([&, operation, fn = std::move(fn)]() mutable {
            const std::string error = fn();
            set_voice_status(error.empty() ? "done: " + operation : "failed: " + error);
            voice_force_refresh = true;
            voice_busy = false;
            refresh();
        });
    };
    auto voice_client = [&]() {
        auto client = std::make_unique<HttpClient>(control_base_url_);
        if (!control_api_token_.empty()) client->set_bearer_token(control_api_token_);
        return client;
    };

    InputOption voice_multi; voice_multi.multiline = true;
    voice_multi.placeholder = "Describe an original synthetic voice";
    auto voice_description_input = Input(&voice_description, voice_multi);
    InputOption sample_multi; sample_multi.multiline = true;
    sample_multi.placeholder = "Text for the voice preview";
    auto voice_sample_input = Input(&voice_sample_text, sample_multi);
    InputOption speech_multi; speech_multi.multiline = true;
    speech_multi.placeholder = "Text to synthesize with the approved voice";
    auto voice_speech_input = Input(&voice_speech_text, speech_multi);
    auto voice_agent_menu = Menu(&voice_agent_entries, &voice_agent_sel, MenuOption::Vertical());
    auto voice_proposal_menu = Menu(&voice_proposal_entries, &voice_proposal_sel, MenuOption::Vertical());

    auto btn_voice_refresh = Button(" Refresh ", [&] { voice_force_refresh = true; }, ButtonOption::Simple());
    auto btn_voice_propose = Button(" Propose ", [&] {
        const std::string agent_id = selected_voice_agent();
        const std::string description = voice_description;
        const std::string sample_text = voice_sample_text;
        if (agent_id.empty()) return;
        run_voice_async("voice proposal", [&, agent_id, description, sample_text]() {
            auto client = voice_client();
            nlohmann::json body = nlohmann::json::object();
            if (!util::trim(description).empty()) body["voice_description"] = description;
            if (!util::trim(sample_text).empty()) body["sample_text"] = sample_text;
            auto response = client->post("/v1/agents/" + agent_id + "/voice/proposals", body);
            return response.ok() ? std::string{} : "HTTP " + std::to_string(response.status) + ": " + parse_api_error(response);
        });
    }, ButtonOption::Simple());
    auto btn_voice_sample = Button(" Generate sample ", [&] {
        const std::string agent_id = selected_voice_agent();
        const auto proposal = selected_voice_proposal();
        const std::string proposal_id = proposal.value("id", std::string{});
        if (agent_id.empty() || proposal_id.empty()) return;
        run_voice_async("sample", [&, agent_id, proposal_id]() {
            auto client = voice_client();
            auto response = client->post("/v1/agents/" + agent_id + "/voice/proposals/" + proposal_id + "/sample",
                                         nlohmann::json::object());
            if (!response.ok()) return "HTTP " + std::to_string(response.status) + ": " + parse_api_error(response);
            try {
                const auto json = nlohmann::json::parse(response.body);
                const std::string path = json.at("proposal").value("preview_audio_path", std::string{});
                if (!path.empty()) { audio_player.load(path); audio_player.play(); }
            } catch (...) {}
            return std::string{};
        });
    }, ButtonOption::Simple());
    auto proposal_decision = [&](const std::string& decision) {
        const std::string agent_id = selected_voice_agent();
        const std::string proposal_id = selected_voice_proposal().value("id", std::string{});
        if (agent_id.empty() || proposal_id.empty()) return;
        run_voice_async(decision, [&, agent_id, proposal_id, decision]() {
            auto client = voice_client();
            auto response = client->post("/v1/agents/" + agent_id + "/voice/proposals/" + proposal_id + "/" + decision,
                                         nlohmann::json::object());
            return response.ok() ? std::string{} : "HTTP " + std::to_string(response.status) + ": " + parse_api_error(response);
        });
    };
    auto btn_voice_approve = Button(" Approve ", [&] { proposal_decision("approve"); }, ButtonOption::Simple());
    auto btn_voice_reject = Button(" Reject ", [&] { proposal_decision("reject"); }, ButtonOption::Simple());
    auto btn_voice_preview = Button(" Play preview ", [&] {
        const std::string path = selected_voice_proposal().value("preview_audio_path", std::string{});
        if (!path.empty()) { audio_player.load(path); audio_player.play(); }
    }, ButtonOption::Simple());
    auto btn_voice_speak = Button(" Synthesize speech ", [&] {
        const std::string agent_id = selected_voice_agent();
        const std::string text_value = voice_speech_text;
        if (agent_id.empty() || util::trim(text_value).empty()) return;
        run_voice_async("speech", [&, agent_id, text_value]() {
            auto client = voice_client();
            auto response = client->post("/v1/agents/" + agent_id + "/speech",
                                         nlohmann::json{{"text", text_value}, {"format", "wav"}});
            if (!response.ok()) return "HTTP " + std::to_string(response.status) + ": " + parse_api_error(response);
            try {
                const auto json = nlohmann::json::parse(response.body);
                const std::string path = json.at("result").value("audio_path", std::string{});
                if (!path.empty()) { audio_player.load(path); audio_player.play(); }
            } catch (...) {}
            return std::string{};
        });
    }, ButtonOption::Simple());
    auto btn_audio_play = Button(" Play ", [&] { audio_player.play(); }, ButtonOption::Simple());
    auto btn_audio_pause = Button(" Pause ", [&] { audio_player.pause(); }, ButtonOption::Simple());
    auto btn_audio_stop = Button(" Stop ", [&] { audio_player.stop(); }, ButtonOption::Simple());
    auto btn_audio_back = Button(" -5s ", [&] { audio_player.seek_relative(-5.0f); }, ButtonOption::Simple());
    auto btn_audio_forward = Button(" +5s ", [&] { audio_player.seek_relative(5.0f); }, ButtonOption::Simple());
    auto voice_buttons = Container::Horizontal({btn_voice_refresh, btn_voice_propose, btn_voice_sample,
        btn_voice_preview, btn_voice_approve, btn_voice_reject, btn_voice_speak,
        btn_audio_back, btn_audio_play, btn_audio_pause, btn_audio_stop, btn_audio_forward});
    auto voice_comp = Container::Vertical({voice_agent_menu, voice_proposal_menu,
        voice_description_input, voice_sample_input, voice_speech_input, voice_buttons});



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
            a.vision_settings.enabled ? (text(" V") | color(Color::Cyan)) : text(""),
        });
        if (st.active) row = std::move(row) | inverted;
        return row;
    };
    auto chat_agent_menu = Menu(&chat_agent_entries, &chat_agent_sel, chat_menu_opt);
    auto chat_agent_menu_m = Maybe(chat_agent_menu, [&]() { return !chat_agent_entries.empty(); });
    auto btn_chat_attach = Button(" Attach Images ", [&] {
        if (chat_inflight.load()) return;
        fb_target = FileBrowserTarget::ChatImage;
        fb_current_path = chat_staged_paths.empty()
            ? fs::current_path() : chat_staged_paths.back().parent_path();
        refresh_fb();
        show_file_browser = true;
    }, ButtonOption::Simple());
    auto btn_chat_clear_images = Button(" Clear Images ", [&] {
        if (chat_inflight.load()) return;
        std::lock_guard<std::mutex> lk(chat_mutex);
        chat_staged_paths.clear();
    }, ButtonOption::Simple());

    auto btn_chat_send = Button(" Send ", [&] {
        if (chat_inflight.load()) return;
        std::vector<fs::path> staged_images;
        {
            std::lock_guard<std::mutex> lk(chat_mutex);
            staged_images = chat_staged_paths;
        }
        if (chat_input.empty() && staged_images.empty()) return;

        auto chat_agents = agents_.list_agents();
        if (chat_agents.empty()) return;
        if (chat_agent_sel < 0 || chat_agent_sel >= static_cast<int>(chat_agents.size())) return;

        const std::string agent_id = chat_agents[chat_agent_sel].id;
        const std::string agent_name = chat_agents[chat_agent_sel].name;
        if (!staged_images.empty() && !chat_agents[chat_agent_sel].vision_settings.enabled) {
            std::lock_guard<std::mutex> lk(chat_mutex);
            chat_status = "failed";
            chat_last_error = "Selected agent profile does not accept images";
            return;
        }
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
            for (const auto& image : staged_images) {
                std::error_code ec;
                const auto bytes = fs::file_size(image, ec);
                chat_transcript.push_back("  image: " + image.filename().string() +
                    (ec ? std::string{} : " (" + std::to_string(bytes) + " bytes)"));
            }
            trim_chat_transcript();
        }
        chat_inflight = true;
        refresh();

        chat_thread = std::thread([&, agent_id, prompt, staged_images]() {
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

            if (!staged_images.empty()) {
                if (base_url.empty()) {
                    std::lock_guard<std::mutex> lk(chat_mutex);
                    chat_status = "failed";
                    chat_last_error = "Image upload requires the control HTTP API";
                    chat_transcript.push_back("Error: " + chat_last_error);
                    chat_inflight = false;
                    refresh();
                    return;
                }
                std::vector<std::string> attachment_ids;
                HttpClient upload_client(base_url);
                upload_client.set_bearer_token(control_api_token_);
                for (const auto& image : staged_images) {
                    const std::string ext = util::to_lower(image.extension().string());
                    const std::string mime = ext == ".png" ? "image/png" : "image/jpeg";
                    auto response = upload_client.post_file(
                        "/v1/agents/" + agent_id + "/attachments",
                        image.string(), {{"X-Filename", image.filename().string()}}, mime);
                    if (!response.ok()) {
                        std::lock_guard<std::mutex> lk(chat_mutex);
                        chat_status = "failed";
                        chat_last_error = "Image upload failed: " + parse_api_error(response);
                        chat_transcript.push_back("Error: " + chat_last_error);
                        chat_inflight = false;
                        refresh();
                        return;
                    }
                    try {
                        attachment_ids.push_back(
                            nlohmann::json::parse(response.body).at("id").get<std::string>());
                    } catch (...) {
                        std::lock_guard<std::mutex> lk(chat_mutex);
                        chat_status = "failed";
                        chat_last_error = "Image upload returned invalid attachment metadata";
                        chat_transcript.push_back("Error: " + chat_last_error);
                        chat_inflight = false;
                        refresh();
                        return;
                    }
                }
                body["attachment_ids"] = attachment_ids;
                {
                    std::lock_guard<std::mutex> lk(chat_mutex);
                    for (const auto& sent : staged_images) {
                        chat_staged_paths.erase(
                            std::remove(chat_staged_paths.begin(), chat_staged_paths.end(), sent),
                            chat_staged_paths.end());
                    }
                    chat_status = "sending";
                }
                refresh();
            }

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
                stream_ok = cli.stream_post(
                    path,
                    body,
                    stream_cb,
                    &stream_status,
                    &stream_body,
                    [&ui_shutting_down] {
                        return ui_shutting_down.load();
                    });
                if (stream_ok) {
                    used_base_url = base_url;
                }
            }

            bool local_fallback_used = false;
            // Only use local fallback when there was no transport-level response
            // and no stream payload at all. This prevents duplicate turns when
            // a partial or completed stream already reached the UI.
            if (!ui_shutting_down.load() && staged_images.empty() &&
                !stream_ok && stream_status == 0 &&
                !stream_payload_seen && !done_seen && local_chat_fallback_) {
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
    auto chat_image_btn_row = Container::Horizontal({btn_chat_attach, btn_chat_clear_images});
    auto chat_transcript_scroll = mm::tui::wrapped_scroll_view([&]() {
        std::lock_guard<std::mutex> lk(chat_mutex);
        std::string transcript;
        for (const auto& line : chat_transcript) {
            if (!transcript.empty()) transcript += "\n\n";
            transcript += line;
        }
        if (chat_inflight.load()) {
            if (!transcript.empty()) transcript += "\n\n";
            transcript += chat_partial_assistant.empty()
                ? "Assistant: (streaming...)"
                : "Assistant (streaming): " + chat_partial_assistant;
        }
        return transcript.empty() ? std::string{"(no chat yet)"} : transcript;
    });
    auto chat_comp = Container::Vertical(
        {chat_agent_menu_m, chat_input_comp, chat_image_btn_row,
         chat_btn_row, chat_transcript_scroll});

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
            if (!control_api_token_.empty()) cli.set_bearer_token(control_api_token_);
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
            if (!control_api_token_.empty()) cli.set_bearer_token(control_api_token_);
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
        auto dns = remote_nodes_enabled ? registry_.get_discovered_nodes()
                                        : std::vector<DiscoveredNode>{};
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

        Element disc_panel;
        if (remote_nodes_enabled) {
            disc_panel = panel("DISCOVERED", std::to_string(dns.size()), vbox({
                dns.empty() ? (text("  listening…") | dim)
                            : (disc_menu->Render() | yframe | flex),
                disc_btns->Render(),
            }));
        } else {
            disc_panel = panel("CLUSTERING OFF", vbox({
                paragraph("This AIO instance accepts only its private embedded node.") |
                    color(Color::Green),
                paragraph("Set [cluster].enabled = true in mantic-mind-aio.toml and restart "
                          "to discover, pair, or add remote nodes.") | dim,
            }));
        }

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
            selected_node_is_embedded()
                ? (text("  Embedded node · process-owned · remove/forget disabled") |
                   color(Color::Cyan))
                : node_btns_m->Render(),
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

            const bool embedded = n.kind == "embedded" || n.id == "local";
            detail_title = "DETAIL · " + mm::tui::node_display_name(n, node_rows) +
                           (embedded ? " · EMBEDDED" : " · REMOTE");
            detail_cap = embedded ? "private in-process transport · no node API listener" : n.url;
            // Per-node CPU/GPU/VRAM live in the NODES table row (sparklines/gauge)
            // and the CLUSTER tile; matching 1a, the detail shows the slot table +
            // capabilities only — no duplicate gauges here.
            Elements rows = {
                hbox({mm::tui::connection_status_el(n), text("   resource health ") | dim,
                      text(to_string(n.health)), text("   "), mm::tui::stale_caption(n)}),

                hbox({text(embedded ? "ownership " : "saved ") | dim,
                      text(embedded ? "AIO process" : (n.remembered ? "yes" : "no")),
                      text("   platform ") | dim, text(n.platform.empty() ? "—" : n.platform),
                      text("   model ") | dim,
                      text(n.loaded_model.empty() ? "(engine-managed)" : n.loaded_model)}),
                text(""),
                slot_table(n.slots, true),
                node_slot_menu->Render() | yframe | size(HEIGHT, LESS_THAN, 4),
                text("Slot suspend, restore, and unload are managed by the agent scheduler.") | dim,
                hbox({text("runtime ") | dim,
                      text(n.llama_runtime.status.empty() ? "unknown" : n.llama_runtime.status) |
                          color(n.llama_runtime.status == "ready" ||
                                n.llama_runtime.status == "resolved"
                                    ? Color::Green : Color::Yellow),
                      text("   version ") | dim,
                      text(n.llama_runtime.version.empty() ? "—" : n.llama_runtime.version),
                      text("   accelerator ") | dim,
                      text(n.llama_runtime.accelerator.empty() ? "—" :
                           n.llama_runtime.accelerator)}),
                n.action_progress.active
                    ? hbox({text("action ") | dim,
                            text(n.action_progress.action) | color(Color::Yellow),
                            text(" · " + n.action_progress.stage + " · ") | dim,
                            text(n.action_progress.cancel_requested
                                ? "cancel requested" : "running")})
                    : (text("action idle") | dim),
                control_action_buttons->Render(),

                text(""),
            };
            std::string operation_status_snapshot;
            {
                std::lock_guard<std::mutex> lock(node_operation_mutex);
                operation_status_snapshot = node_operation_status;
            }
            if (node_operation_inflight.load())
                operation_status_snapshot += "  " + std::string(mm::tui::spinner_frame());
            if (!operation_status_snapshot.empty())
                rows.push_back(paragraph(operation_status_snapshot) |
                    color(node_operation_inflight.load() ? Color::Yellow : Color::GrayLight));
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

            Element engine_specific;
            if (ed_backend == 0) {
                engine_specific = vbox({
                    hbox({field(" context", ed_inp_ctx->Render() | flex),
                          field(" gpu_layers", ed_inp_gpu_layers->Render() | flex)}),
                    hbox({field(" threads", ed_inp_threads->Render() | flex),
                          field(" http_threads", ed_inp_threads_http->Render() | flex)}),
                    hbox({field(" parallel", ed_inp_parallel->Render() | flex),
                          field(" batch", ed_inp_batch->Render() | flex),
                          field(" ubatch", ed_inp_ubatch->Render() | flex)}),
                    ed_cb_flash->Render(),
                    ed_cb_vision->Render(),
                    hbox({text(" projector") | dim | size(WIDTH, EQUAL, 15),
                          ed_inp_mmproj->Render() | flex, text(" "), btn_browse_mmproj->Render()}),
                    [&]() -> Element {
                        const auto suggestions = suggest_mmproj_files(ed_model);
                        if (suggestions.empty()) return text("  no adjacent mmproj-*.gguf suggestions") | dim;
                        std::string line = "  suggestions: ";
                        for (std::size_t i = 0; i < suggestions.size() && i < 3; ++i) {
                            if (i > 0) line += " | ";
                            line += fs::path(suggestions[i]).filename().string();
                        }
                        return text(line) | color(Color::Cyan);
                    }(),
                    field(" extra args", ed_inp_llama_extra->Render() | border | size(HEIGHT, LESS_THAN, 4)),
                });
            } else if (ed_backend == 1) {
                engine_specific = vbox({
                    field(" base_url", ed_inp_api_base->Render() | flex),
                    field(" chat path", ed_inp_api_path->Render() | flex),
                    field(" key env", ed_inp_api_key_env->Render() | flex),
                    field(" API key", ed_inp_api_key->Render() | flex),
                    ed_cb_vision->Render(),
                    text("Vision declares that the remote model accepts images; projector must be empty") | dim,
                    hbox({text(" projector") | dim | size(WIDTH, EQUAL, 15),
                          ed_inp_mmproj->Render() | flex}),
                    text("API keys are process-local and never persisted") | dim,
                });
            } else {
                engine_specific = vbox({
                    paragraph("This profile uses the unsupported legacy backend '" +
                              ed_unsupported_backend +
                              "'. Select llama.cpp or Remote API before saving.") |
                        color(Color::Red),
                });
            }
            Element engine_body = vbox({
                field(" backend", ed_backend_toggle->Render() | flex),
                separator(),
                std::move(engine_specific),
            });
            Element sections = vbox({
                sec_static("Identity", vbox({
                    field(" id", ed_inp_id->Render() | flex),
                    field(" name", ed_inp_name->Render() | flex),
                })),
                sec_static("Model & placement", vbox({
                    hbox({text(" model_path") | dim | size(WIDTH, EQUAL, 15),
                          ed_inp_model->Render() | flex, text(" "), btn_browse_model->Render()}),
                    field(" served alias", ed_inp_served_model->Render() | flex),
                    field(" preferred_node", ed_inp_pnode->Render() | flex),
                })),
                sec_static("System prompt",
                           ed_inp_sys->Render() | size(HEIGHT, LESS_THAN, 5) | border),
                sec_toggle(ed_sec_sampling_btn->Render(), ed_open_sampling, vbox({
                    hbox({
                        field(" temperature", ed_inp_temp->Render() | flex, 13),
                        field(" top_p", ed_inp_topp->Render() | flex, 8),
                        field(" top_k", ed_inp_topk->Render() | flex, 8),
                        field(" min_p", ed_inp_minp->Render() | flex, 8),
                    }),
                    hbox({
                        field(" presence", ed_inp_presence->Render() | flex, 11),
                        field(" repeat", ed_inp_repeat->Render() | flex, 9),
                        field(" max_tokens", ed_inp_max->Render() | flex, 12),
                    }),
                })),
                sec_toggle(ed_sec_engine_btn->Render(), ed_open_engine, std::move(engine_body)),
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
            col(text("V R M T"), 9), sep(),
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
                           fmt2(a.runtime_settings.top_p) + " · top_k " +
                           std::to_string(a.runtime_settings.top_k) + " · min_p " +
                           fmt2(a.runtime_settings.min_p) + " · " +
                           std::to_string(a.runtime_settings.max_tokens) + " tok")}),
                hbox({text("engine   ") | dim,
                      text([&]() {
                          const std::string backend = util::to_lower(util::trim(a.inference_backend));
                          if (backend == "api") {
                              return "remote API · " + a.api_settings.base_url;
                          }
                          if (!backend.empty() && backend != "llama-cpp" &&
                              backend != "llama.cpp" && backend != "llama") {
                              return "unsupported · " + backend;
                          }
                          return "llama.cpp · ctx " +
                              std::to_string(a.runtime_settings.ctx_size) + " · parallel " +
                              std::to_string(a.runtime_settings.parallel);
                      }())}),
                hbox({text("caps     ") | dim, feat(a.reasoning_enabled, "reasoning"), text("  "),
                      feat(a.memories_enabled, "memories"), text("  "),
                      feat(a.tools_enabled, "tools"), text("  "),
                      feat(a.vision_settings.enabled, "vision")}),
                a.vision_settings.mmproj_path.empty()
                    ? text("")
                    : hbox({text("projector") | dim,
                            text(" " + fs::path(a.vision_settings.mmproj_path).filename().string()) |
                                color(Color::Cyan)}),
            })) | size(HEIGHT, EQUAL, 10);
        } else {
            detail_panel = panel("AGENT", text("  (no agents)") | dim) | size(HEIGHT, EQUAL, 10);
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

        std::string status_snapshot;
        std::string error_snapshot;
        std::string conv_snapshot;
        std::vector<fs::path> staged_snapshot;
        {
            std::lock_guard<std::mutex> lk(chat_mutex);
            status_snapshot = chat_status;
            error_snapshot = chat_last_error;
            conv_snapshot = chat_last_conv_id;
            staged_snapshot = chat_staged_paths;
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
        Elements staged_lines;
        if (staged_snapshot.empty()) {
            staged_lines.push_back(text("  (no staged images)") | dim);
        } else {
            for (const auto& image : staged_snapshot) {
                std::error_code ec;
                const auto bytes = fs::file_size(image, ec);
                staged_lines.push_back(text("  " + image.filename().string() +
                    (ec ? std::string{} : " - " + std::to_string(bytes) + " B")) |
                    color(Color::Cyan));
            }
        }
        Element left = panel("AGENT", vbox({
            chat_head,
            separator(),
            cs.empty() ? (text("  No agents configured.") | dim)
                       : (chat_agent_menu->Render() | yframe | flex),
            separator(),
            text(" prompt") | dim,
            chat_input_comp->Render() | border,
            hbox({text(" images") | dim, filler(),
                  text(std::to_string(staged_snapshot.size()) + "/8") | dim}),
            vbox(std::move(staged_lines)) | size(HEIGHT, LESS_THAN, 4) | yframe,
            chat_image_btn_row->Render(),
            chat_btn_row->Render(),
            text(chat_inflight.load() ? "  send blocked while streaming" : "  streams SSE") | dim,
        })) | size(WIDTH, EQUAL, 42);

        Element right = panel("TRANSCRIPT", conv_snapshot,
                              chat_transcript_scroll->Render()) | flex;

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
            mm::tui::col(text("AGENT") | bold, 15),
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
                mm::tui::col(text(sample.value("agent_id", std::string{})), 15),
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

    auto voice_renderer = Renderer(voice_comp, [&]() -> Element {
        const auto configs = agents_.list_agents();
        voice_agent_entries.clear();
        for (const auto& config : configs) voice_agent_entries.push_back(config.name + " [" + config.id + "]");
        if (voice_agent_sel >= static_cast<int>(configs.size()))
            voice_agent_sel = std::max(0, static_cast<int>(configs.size()) - 1);
        const std::string agent_id = selected_voice_agent();
        const int64_t now = util::now_ms();
        if (!agent_id.empty() && (voice_force_refresh.exchange(false) ||
            voice_loaded_agent != agent_id || now - voice_refreshed_ms >= 2000)) {
            voice_loaded_agent = agent_id;
            voice_refreshed_ms = now;
            auto client = voice_client();
            auto response = client->get("/v1/agents/" + agent_id + "/voice");
            if (response.ok()) {
                try { voice_state = nlohmann::json::parse(response.body); }
                catch (const std::exception& e) { set_voice_status(e.what()); }
            } else {
                set_voice_status("voice state HTTP " + std::to_string(response.status));
            }
        }

        const auto proposals = voice_state.value("proposals", nlohmann::json::array());
        voice_proposal_entries.clear();
        for (const auto& proposal : proposals) {
            voice_proposal_entries.push_back(
                proposal.value("display_name", std::string{"proposal"}) + "  [" +
                proposal.value("status", std::string{"pending"}) + "]");
        }
        if (voice_proposal_sel >= static_cast<int>(proposals.size()))
            voice_proposal_sel = std::max(0, static_cast<int>(proposals.size()) - 1);

        const auto proposal = selected_voice_proposal();
        Elements detail_rows;
        if (proposal.empty()) {
            detail_rows.push_back(text("No proposal selected. Use Propose with blank fields for an agent-generated design.") | dim);
        } else {
            detail_rows.push_back(text(proposal.value("display_name", std::string{})) | bold);
            detail_rows.push_back(text("status: " + proposal.value("status", std::string{})) |
                color(proposal.value("status", std::string{}) == "approved" ? Color::Green : Color::Yellow));
            detail_rows.push_back(paragraph("description: " + proposal.value("voice_description", std::string{})));
            detail_rows.push_back(paragraph("sample: " + proposal.value("sample_text", std::string{})) | dim);
            const std::string error = proposal.value("error", std::string{});
            if (!error.empty()) detail_rows.push_back(paragraph(error) | color(Color::Red));
        }

        const auto player = audio_player.status();
        const float ratio = player.duration_seconds > 0.0f
            ? std::clamp(player.position_seconds / player.duration_seconds, 0.0f, 1.0f) : 0.0f;
        char playback[96];
        std::snprintf(playback, sizeof(playback), "%s %.1f / %.1f sec  volume %.0f%%",
                      player.playing ? "playing" : "paused", player.position_seconds,
                      player.duration_seconds, player.volume * 100.0f);
        std::string status_snapshot;
        {
            std::lock_guard<std::mutex> lock(voice_status_mutex);
            status_snapshot = voice_status;
        }

        Element left = mm::tui::panel("AGENTS", voice_agent_menu->Render() | yframe | flex) |
                       size(WIDTH, EQUAL, 30);
        Element proposals_panel = mm::tui::panel("PROPOSALS", voice_proposal_menu->Render() | yframe | flex) |
                                  size(WIDTH, EQUAL, 34);
        Element editor = mm::tui::panel("VOICE DESIGN", vbox({
            text("Description (blank = agent proposes)") | dim,
            voice_description_input->Render() | border | size(HEIGHT, LESS_THAN, 5),
            text("Sample text") | dim,
            voice_sample_input->Render() | border | size(HEIGHT, LESS_THAN, 4),
            text("Speech text") | dim,
            voice_speech_input->Render() | border | size(HEIGHT, LESS_THAN, 4),
        })) | flex;
        Element player_panel = mm::tui::panel("AUDIO PLAYER", vbox({
            text(playback), gauge(ratio),
            text(player.path.empty() ? "No audio loaded" : player.path) | dim,
            player.error.empty() ? text("") : (paragraph(player.error) | color(Color::Yellow)),
        }));

        Elements page = {
            hbox({left, proposals_panel, editor}) | flex,
            mm::tui::panel("SELECTED PROPOSAL", vbox(std::move(detail_rows))),
            player_panel,
            voice_buttons->Render(),
            text(status_snapshot + (voice_busy ? "  " + std::string(mm::tui::spinner_frame()) : "")) |
                color(voice_busy ? Color::Yellow : Color::GrayDark),
            text(voice_state.value("tts_enabled", false) ? "TTS enabled" : "TTS disabled") |
                color(voice_state.value("tts_enabled", false) ? Color::Green : Color::Red),
        };
        return vbox(std::move(page));
    });

    auto main_tabs = Container::Tab({nodes_renderer, agents_renderer, activity_renderer,
        chat_renderer, curation_renderer, performance_renderer, voice_renderer}, &tab_index);

    // Mockup-style header: the tab strip is real Button components (clickable +
    // focusable) rendered as `n Label` chips, the active one inverted.
    static const std::array<const char*, 7> kTabLabels = {
        "1 Nodes", "2 Agents", "3 Activity", "4 Chat", "5 Curation", "6 Performance", "7 Voice"};
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
        const bool aio_mode = std::any_of(ns.begin(), ns.end(), [](const NodeInfo& n) {
            return n.kind == "embedded" || n.id == "local";
        });
        const Color dot = (tot > 0 && conn == tot) ? Color::Green
                        : (conn > 0)               ? Color::Yellow
                                                   : Color::Red;
        auto header = hbox({
            text(aio_mode ? " mantic-mind-aio " : " mantic-mind-control ") | bold,
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
            text(" 1-7 tabs · ↑/↓ select · click rows · q quit") | dim,
            filler(),
            text(aio_mode ? "mantic-mind · AIO " : "mantic-mind · control ") | dim,
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
        if (show_add_node || show_pin_entry || show_node_network_confirm ||
            show_agent_validation_modal ||
            show_file_browser || show_cur_delete_confirm) return false;
        if (!show_editor) {
            if (ev == Event::Character('1')) { tab_index = 0; return true; }
            if (ev == Event::Character('2')) { tab_index = 1; return true; }
            if (ev == Event::Character('3')) { tab_index = 2; return true; }
            if (ev == Event::Character('4')) { tab_index = 3; return true; }
            if (ev == Event::Character('6')) { tab_index = 5; return true; }
            if (ev == Event::Character('7')) { tab_index = 6; return true; }
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
    auto with_network_consent = Modal(with_pin, node_network_modal_renderer,
                                      &show_node_network_confirm);
    auto with_agent_validation = Modal(with_network_consent, agent_validation_modal_renderer,
                                       &show_agent_validation_modal);
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
    layout.set("agents.editor_width", ed_split);
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

    // Signal the chat stream to abort, then join every thread so none is destroyed
    // while joinable (which would terminate the process) — including on the error path.
    ui_shutting_down = true;
    if (shutdown_callback_) {
        try {
            // Stop ingress and signal the shared queue/node work before joining
            // UI-owned workers. In AIO this also reaches an embedded inference
            // that has not emitted its first byte yet.
            shutdown_callback_();
        } catch (const std::exception& exception) {
            MM_WARN("Control UI shutdown callback failed: {}", exception.what());
        } catch (...) {
            MM_WARN("Control UI shutdown callback failed");
        }
    }
    if (node_operation_inflight.load()) {
        try {
            NodeOperationsPtr operation;
            {
                std::lock_guard<std::mutex> lock(active_node_operation_mutex);
                operation = active_node_operation;
            }
            if (operation) {
                // Embedded cancellation is a direct, bounded state mutation.
                // Remote shutdown instead aborts the active HTTP transport;
                // sending a second request to a silent remote could itself
                // delay UI teardown.
                if (operation->embedded()) {
                    (void)operation->cancel_action();
                }
                operation->request_shutdown();
            }
        } catch (...) {
            // Best-effort cancellation; the worker is still joined below.
        }
    }
    ticker_running = false;
    if (ticker.joinable())      ticker.join();
    if (pairing_thread.joinable()) pairing_thread.join();
    if (node_operation_thread.joinable()) node_operation_thread.join();
    if (chat_thread.joinable()) chat_thread.join();
    if (cur_thread.joinable())  cur_thread.join();
    if (voice_thread.joinable()) voice_thread.join();

    if (!loop_error.empty()) {
        MM_ERROR("Control UI event loop terminated by exception: {}", loop_error);
    }
}

} // namespace mm
