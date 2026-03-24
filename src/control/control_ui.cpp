#include "control/control_ui.hpp"
#include "control/node_registry.hpp"
#include "control/agent_manager.hpp"
#include "control/agent_config_validator.hpp"
#include "common/agent.hpp"
#include "common/agent_db.hpp"
#include "common/gguf_metadata.hpp"
#include "common/http_client.hpp"
#include "common/model_catalog.hpp"
#include "common/models.hpp"
#include "common/logger.hpp"
#include "common/tool_executor.hpp"
#include "common/util.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/component_options.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mm {

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
            log_entries_.erase(log_entries_.begin());
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
    int  catalog_sel   = 0;
    std::vector<std::string> catalog_entries;
    int  node_model_sel = 0;
    std::vector<std::string> node_model_entries;
    std::vector<std::string> control_catalog_filenames;
    std::string node_models_cached_node_id;
    std::vector<StoredModel> node_models_cached;
    int64_t node_models_cached_disk_free_mb = 0;
    int64_t node_models_cached_at_ms = 0;
    int64_t control_catalog_cached_at_ms = 0;
    std::string node_model_action_status;

    // Modals
    bool show_add_node  = false;
    bool show_pin_entry = false;
    bool show_agent_validation_modal = false;
    std::string add_url;
    std::string pin_input, pair_url, pair_nonce;

    // Agents tab
    int  agent_sel    = 0;
    bool show_editor  = false;
    std::string ed_id, ed_id_orig, ed_name, ed_model, ed_sysprompt, ed_pref_node;
    std::string ed_ctx_s{"4096"}, ed_gpu_s{"-1"}, ed_thr_s{"-1"};
    std::string ed_temp_s{"0.70"}, ed_topp_s{"0.90"}, ed_max_s{"1024"};
    std::string ed_extra_args_text;
    bool ed_flash{true}, ed_reasoning{false}, ed_memories{true}, ed_tools{false};
    std::string ed_model_inspected_path;
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
    std::string chat_input;

    std::mutex              chat_mutex;
    std::vector<std::string> chat_transcript;
    std::string              chat_partial_assistant;
    std::string              chat_status{"idle"};
    std::string              chat_last_error;
    std::string              chat_last_conv_id;

    std::atomic<bool> chat_inflight{false};
    std::thread       chat_thread;

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

    // File browser
    bool show_file_browser = false;
    fs::path                  fb_current_path;
    std::vector<std::string>  fb_display_names;
    std::vector<fs::path>     fb_entry_paths;
    std::vector<bool>         fb_entry_is_dir;
    int fb_sel = 0;

    //

    auto gauge_row = [](const std::string& label, float pct,
                        const std::string& detail) -> Element {
        float r = std::max(0.0f, std::min(1.0f, pct / 100.0f));
        Color c = pct > 80.0f ? Color::Red
                : pct > 60.0f ? Color::Yellow
                              : Color::Green;
        char buf[16];
        snprintf(buf, sizeof(buf), "%5.1f%%", static_cast<double>(pct));
        return hbox({
            text(label) | size(WIDTH, EQUAL, 5),
            gauge(r) | color(c) | size(WIDTH, EQUAL, 16),
            text(" "), text(buf),
            detail.empty() ? text("") : (text("  " + detail) | color(Color::GrayDark)),
        });
    };

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
        if (chat_transcript.size() > kMaxLines) {
            const size_t erase_count = chat_transcript.size() - kMaxLines;
            chat_transcript.erase(chat_transcript.begin(),
                                  chat_transcript.begin() + erase_count);
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

    auto refresh_control_catalog = [&](bool force) {
        const int64_t now_ms = util::now_ms();
        if (!force && (now_ms - control_catalog_cached_at_ms) < 5000) return;
        control_catalog_cached_at_ms = now_ms;

        control_catalog_filenames.clear();
        std::set<std::string> unique_names;
        std::error_code ec;
        if (!models_dir_.empty() && fs::exists(models_dir_, ec)) {
            for (const auto& entry : fs::recursive_directory_iterator(models_dir_, ec)) {
                if (!entry.is_regular_file()) continue;
                std::string ext = util::to_lower(entry.path().extension().string());
                if (ext != ".gguf") continue;
                const std::string name = entry.path().filename().string();
                if (is_safe_model_filename(name)) unique_names.insert(name);
            }
        }

        control_catalog_filenames.assign(unique_names.begin(), unique_names.end());
        if (!control_catalog_filenames.empty() &&
            catalog_sel >= static_cast<int>(control_catalog_filenames.size())) {
            catalog_sel = static_cast<int>(control_catalog_filenames.size()) - 1;
        } else if (control_catalog_filenames.empty()) {
            catalog_sel = 0;
        }
    };

    auto refresh_node_storage_cache = [&](const NodeId& node_id, bool force) {
        const int64_t now_ms = util::now_ms();
        if (!force &&
            node_id == node_models_cached_node_id &&
            (now_ms - node_models_cached_at_ms) < 3000) {
            return;
        }
        node_models_cached_node_id = node_id;
        node_models_cached_at_ms = now_ms;

        node_models_cached.clear();
        node_models_cached_disk_free_mb = 0;
        if (node_id.empty()) return;

        HttpClient cli(control_base_url_);
        auto resp = cli.get("/v1/nodes/" + node_id + "/models");
        if (!resp.ok()) {
            node_model_action_status = "refresh failed for node storage (HTTP " +
                                       std::to_string(resp.status) + ")";
            return;
        }

        try {
            auto j = nlohmann::json::parse(resp.body);
            if (j.contains("stored_models"))
                node_models_cached = j["stored_models"].get<std::vector<StoredModel>>();
            if (j.contains("disk_free_mb"))
                node_models_cached_disk_free_mb = j["disk_free_mb"].get<int64_t>();
        } catch (const std::exception& e) {
            node_model_action_status = std::string("refresh parse error: ") + e.what();
            return;
        }

        if (!node_models_cached.empty() &&
            node_model_sel >= static_cast<int>(node_models_cached.size())) {
            node_model_sel = static_cast<int>(node_models_cached.size()) - 1;
        } else if (node_models_cached.empty()) {
            node_model_sel = 0;
        }
    };

    auto selected_catalog_filename = [&]() -> std::string {
        if (catalog_sel < 0 ||
            catalog_sel >= static_cast<int>(control_catalog_filenames.size())) {
            return {};
        }
        return control_catalog_filenames[static_cast<size_t>(catalog_sel)];
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
        try { cfg.llama_settings.ctx_size = std::stoi(ed_ctx_s); } catch (...) {}
        try { cfg.llama_settings.n_gpu_layers = std::stoi(ed_gpu_s); } catch (...) {}
        try { cfg.llama_settings.n_threads = std::stoi(ed_thr_s); } catch (...) {}
        try { cfg.llama_settings.temperature = std::stof(ed_temp_s); } catch (...) {}
        try { cfg.llama_settings.top_p = std::stof(ed_topp_s); } catch (...) {}
        try { cfg.llama_settings.max_tokens = std::stoi(ed_max_s); } catch (...) {}
        cfg.llama_settings.flash_attn = ed_flash;
        cfg.reasoning_enabled = ed_reasoning;
        cfg.memories_enabled = ed_memories;
        cfg.tools_enabled = ed_tools;

        cfg.llama_settings.extra_args.clear();
        size_t start = 0;
        while (start <= ed_extra_args_text.size()) {
            size_t end = ed_extra_args_text.find('\n', start);
            std::string line = ed_extra_args_text.substr(
                start, end == std::string::npos ? std::string::npos : (end - start));
            line = util::trim(line);
            if (!line.empty()) cfg.llama_settings.extra_args.push_back(line);
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

    auto refresh_editor_model_info = [&]() {
        const std::string resolved = resolve_model_path_for_metadata(ed_model, models_dir_);
        if (ed_model_inspected_path == ed_model && ed_model_info.source_path == resolved) return;
        ed_model_inspected_path = ed_model;
        ed_model_info = inspect_model_capabilities(ed_model, models_dir_);
        ed_validation_signature.clear();
    };

    auto refresh_editor_validation = [&]() {
        refresh_editor_model_info();
        const std::string signature =
            ed_id + '\n' + ed_name + '\n' + ed_model + '\n' + ed_sysprompt + '\n' + ed_pref_node +
            '\n' + ed_ctx_s + '\n' + ed_gpu_s + '\n' + ed_thr_s + '\n' + ed_temp_s + '\n' +
            ed_topp_s + '\n' + ed_max_s + '\n' + ed_extra_args_text + '\n' +
            (ed_flash ? "1" : "0") + (ed_reasoning ? "1" : "0") +
            (ed_memories ? "1" : "0") + (ed_tools ? "1" : "0");
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
        if (!pairing_key_.empty()) {
            //
            auto key = registry_.pair_node(pair_url, pairing_key_);
            if (!key.empty())
                log(LogLevel::Info, "Paired with " + pair_url + " (PSK)");
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
        show_add_node = true;
    }, ButtonOption::Simple());
    auto disc_btns    = Container::Horizontal({btn_pair, btn_add_manual});
    // Maybe wrapper: disc_menu is excluded from the component tree (events + render)
    // when disc_entries is empty, preventing FTXUI from indexing an empty vector.
    auto disc_menu_m  = Maybe(disc_menu, [&]() { return !disc_entries.empty(); });
    auto disc_comp    = Container::Vertical({disc_menu_m, disc_btns});

    // Connected (registered) nodes section
    auto node_menu   = Menu(&node_entries, &node_sel, MenuOption::Vertical());
    auto trigger_node_update = [&](bool build) {
        auto ns = registry_.list_nodes();
        if (node_sel < 0 || node_sel >= static_cast<int>(ns.size())) return;
        const auto& n = ns[node_sel];
        std::string msg;
        bool ok = registry_.request_llama_update(n.id, build, &msg);
        if (ok) {
            log(LogLevel::Info, "Update started on node " + n.id + ": " + msg);
        } else {
            log(LogLevel::Error, "Update failed for node " + n.id + ": " + msg);
        }
    };
    auto btn_upd_n   = Button("[U] Update llama.cpp", [&] {
        trigger_node_update(true);
    }, ButtonOption::Simple());
    auto btn_chk_n   = Button("[C] Check update", [&] {
        auto ns = registry_.list_nodes();
        if (node_sel < 0 || node_sel >= static_cast<int>(ns.size())) return;
        std::string msg;
        bool ok = registry_.request_llama_check_update(ns[node_sel].id, &msg);
        if (ok) log(LogLevel::Info, "Check update on node " + ns[node_sel].id + ": " + msg);
        else    log(LogLevel::Error, "Check update failed for node " + ns[node_sel].id + ": " + msg);
    }, ButtonOption::Simple());
    auto btn_rem_n   = Button("[-] Remove", [&] {
        auto ns = registry_.list_nodes();
        if (node_sel >= 0 && node_sel < static_cast<int>(ns.size())) {
            registry_.remove_node(ns[node_sel].id);
            if (node_sel >= static_cast<int>(ns.size()) - 1)
                node_sel = std::max(0, static_cast<int>(ns.size()) - 2);
        }
    }, ButtonOption::Simple());
    auto node_menu_m  = Maybe(node_menu, [&]() { return !node_entries.empty(); });
    auto node_btns    = Container::Horizontal({btn_upd_n, btn_chk_n, btn_rem_n});
    auto nodes_section = Container::Vertical({node_menu_m, node_btns});

    auto catalog_menu = Menu(&catalog_entries, &catalog_sel, MenuOption::Vertical());
    auto catalog_menu_m = Maybe(catalog_menu, [&]() { return !catalog_entries.empty(); });
    auto node_models_menu = Menu(&node_model_entries, &node_model_sel, MenuOption::Vertical());
    auto node_models_menu_m = Maybe(node_models_menu, [&]() { return !node_model_entries.empty(); });

    auto pull_catalog_to_node = [&]() {
        auto ns = registry_.list_nodes();
        if (node_sel < 0 || node_sel >= static_cast<int>(ns.size())) {
            node_model_action_status = "no node selected";
            return;
        }
        const std::string filename = selected_catalog_filename();
        if (!is_safe_model_filename(filename)) {
            node_model_action_status = "select a valid catalog model first";
            return;
        }

        HttpClient cli(control_base_url_);
        auto resp = cli.post("/v1/nodes/" + ns[node_sel].id + "/models/pull",
                             nlohmann::json{{"model_filename", filename}});
        if (!resp.ok()) {
            node_model_action_status = "pull failed: " + parse_api_error(resp);
            log(LogLevel::Error,
                "Node model pull failed on " + ns[node_sel].id + ": " + node_model_action_status);
            return;
        }

        node_model_action_status = "pull requested: " + filename;
        log(LogLevel::Info, "Requested node pull: " + filename + " -> " + ns[node_sel].id);
        refresh_node_storage_cache(ns[node_sel].id, true);
    };

    auto delete_model_on_node = [&]() {
        auto ns = registry_.list_nodes();
        if (node_sel < 0 || node_sel >= static_cast<int>(ns.size())) {
            node_model_action_status = "no node selected";
            return;
        }
        if (node_model_sel < 0 || node_model_sel >= static_cast<int>(node_models_cached.size())) {
            node_model_action_status = "no node model selected";
            return;
        }

        const std::string filename = canonical_model_filename(
            node_models_cached[static_cast<size_t>(node_model_sel)].model_path);
        if (!is_safe_model_filename(filename)) {
            node_model_action_status = "selected model has invalid filename";
            return;
        }

        HttpClient cli(control_base_url_);
        auto resp = cli.del("/v1/nodes/" + ns[node_sel].id + "/models/" + filename);
        if (!resp.ok()) {
            node_model_action_status = "delete failed: " + parse_api_error(resp);
            log(LogLevel::Error,
                "Node model delete failed on " + ns[node_sel].id + ": " + node_model_action_status);
            return;
        }

        node_model_action_status = "deleted from node: " + filename;
        log(LogLevel::Info, "Deleted node model copy: " + filename + " from " + ns[node_sel].id);
        refresh_node_storage_cache(ns[node_sel].id, true);
    };

    auto btn_pull_model_n = Button("[P] Pull Model", [&] { pull_catalog_to_node(); }, ButtonOption::Simple());
    auto btn_del_model_n  = Button("[D] Delete Model", [&] { delete_model_on_node(); }, ButtonOption::Simple());
    auto btn_refresh_models_n = Button("[R] Refresh Models", [&] {
        refresh_control_catalog(true);
        auto ns = registry_.list_nodes();
        if (node_sel >= 0 && node_sel < static_cast<int>(ns.size())) {
            refresh_node_storage_cache(ns[node_sel].id, true);
        }
        node_model_action_status = "refreshed model views";
    }, ButtonOption::Simple());
    auto node_model_btns = Container::Horizontal(
        {btn_pull_model_n, btn_del_model_n, btn_refresh_models_n});
    auto node_model_section = Container::Vertical(
        {catalog_menu_m, node_models_menu_m, node_model_btns});

    auto nodes_comp  = Container::Vertical({disc_comp, nodes_section, node_model_section});

    //

    InputOption url_iopt; url_iopt.multiline = false;
    url_iopt.placeholder = "http://hostname:7070";
    auto modal_url    = Input(&add_url, url_iopt);
    auto modal_ok     = Button("  Connect  ", [&] {
        if (!add_url.empty()) {
            if (!pairing_key_.empty()) {
                auto key = registry_.pair_node(add_url, pairing_key_);
                if (!key.empty()) {
                    log(LogLevel::Info, "Paired with " + add_url + " (PSK)");
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
    auto modal_inputs = Container::Vertical({modal_url, modal_btns});

    auto modal_renderer = Renderer(modal_inputs, [&]() {
        return vbox({
            text(" Add Node Manually ") | bold | hcenter,
            separator(),
            hbox({text(" URL : "), modal_url->Render() | flex}),
            text(" Uses pairing (PIN or configured PSK)") | color(Color::GrayDark),
            separator(),
            modal_btns->Render() | hcenter,
        }) | border | size(WIDTH, EQUAL, 54);
    });

    //

    InputOption pin_iopt; pin_iopt.multiline = false;
    pin_iopt.placeholder = "000000";

    auto pin_input_comp   = Input(&pin_input, pin_iopt);
    auto pin_ok           = Button("  Pair  ", [&] {
        if (!pin_input.empty()) {
            auto key = registry_.complete_pair(pair_url, pair_nonce, pin_input);
            if (!key.empty())
                log(LogLevel::Info, "Paired with " + pair_url);
            else
                log(LogLevel::Error, "PIN pairing failed for " + pair_url);
        }
        show_pin_entry = false;
    }, ButtonOption::Simple());
    auto pin_cancel       = Button("  Cancel  ", [&] { show_pin_entry = false; },
                                   ButtonOption::Simple());
    auto pin_modal_btns   = Container::Horizontal({pin_ok, pin_cancel});
    auto pin_modal_inputs = Container::Vertical({pin_input_comp, pin_modal_btns});

    auto pin_modal_renderer = Renderer(pin_modal_inputs, [&]() {
        return vbox({
            text(" Enter PIN ") | bold | hcenter,
            separator(),
            text(" PIN shown on node TUI") | color(Color::GrayDark),
            hbox({text(" PIN: "), pin_input_comp->Render() | flex}),
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

    auto agent_menu   = Menu(&agent_entries, &agent_sel, MenuOption::Vertical());
    auto btn_new_a    = Button("[+] New", [&] {
        ed_id.clear(); ed_id_orig.clear(); ed_name = "New Agent"; ed_model.clear();
        ed_sysprompt.clear(); ed_pref_node.clear();
        ed_ctx_s = "4096"; ed_gpu_s = "-1"; ed_thr_s = "-1";
        ed_temp_s = "0.70"; ed_topp_s = "0.90"; ed_max_s = "1024";
        ed_extra_args_text.clear();
        ed_flash = true; ed_reasoning = false; ed_memories = true; ed_tools = false;
        ed_model_inspected_path.clear();
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
            ed_ctx_s = std::to_string(c.llama_settings.ctx_size);
            ed_gpu_s = std::to_string(c.llama_settings.n_gpu_layers);
            ed_thr_s = std::to_string(c.llama_settings.n_threads);
            char tmp[32];
            snprintf(tmp, sizeof(tmp), "%.2f", static_cast<double>(c.llama_settings.temperature));
            ed_temp_s = tmp;
            snprintf(tmp, sizeof(tmp), "%.2f", static_cast<double>(c.llama_settings.top_p));
            ed_topp_s = tmp;
            ed_max_s = std::to_string(c.llama_settings.max_tokens);
            ed_extra_args_text.clear();
            for (size_t i = 0; i < c.llama_settings.extra_args.size(); ++i) {
                if (i > 0) ed_extra_args_text += '\n';
                ed_extra_args_text += c.llama_settings.extra_args[i];
            }
            ed_flash = c.llama_settings.flash_attn;
            ed_reasoning = c.reasoning_enabled;
            ed_memories  = c.memories_enabled;
            ed_tools     = c.tools_enabled;
            ed_model_inspected_path.clear();
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
    auto agent_list_comp = Container::Vertical({agent_menu, agent_list_btns});

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
    auto ed_inp_ctx   = Input(&ed_ctx_s,     sl);
    auto ed_inp_gpu   = Input(&ed_gpu_s,     sl);
    auto ed_inp_thr   = Input(&ed_thr_s,     sl);
    auto ed_inp_temp  = Input(&ed_temp_s,    sl);
    auto ed_inp_topp  = Input(&ed_topp_s,    sl);
    auto ed_inp_max   = Input(&ed_max_s,     sl);
    auto ed_inp_extra = Input(&ed_extra_args_text, ml);

    auto ed_cb_flash     = Checkbox("Flash Attention", &ed_flash);
    auto ed_cb_reasoning = Checkbox("Reasoning",       &ed_reasoning);
    auto ed_cb_memories  = Checkbox("Memories",        &ed_memories);
    auto ed_cb_tools     = Checkbox("Tools",            &ed_tools);

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

    auto editor_comp  = Container::Vertical({
        ed_inp_id, ed_inp_name, model_row, ed_inp_sys, ed_inp_pnode,
        ed_inp_ctx, ed_inp_gpu, ed_inp_thr,
        ed_inp_temp, ed_inp_topp, ed_inp_max,
        ed_inp_extra,
        ed_cb_flash, ed_cb_reasoning, ed_cb_memories, ed_cb_tools,
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
        ed_model_inspected_path.clear();
        refresh_editor_model_info();
        if (ed_model_info.n_ctx_train > 0) {
            ed_ctx_s = std::to_string(ed_model_info.n_ctx_train);
        }
        ed_reasoning = ed_model_info.supports_reasoning;
        if (ed_model_info.supports_tool_calls) {
            ed_tools = true;
        }
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
    auto chat_agent_menu = Menu(&chat_agent_entries, &chat_agent_sel, MenuOption::Vertical());
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
        chat_status = chat_inflight.load() ? "streaming" : "idle";
    }, ButtonOption::Simple());

    auto chat_btn_row = Container::Horizontal({btn_chat_send, btn_chat_clear});
    auto chat_comp = Container::Vertical({chat_agent_menu_m, chat_input_comp, chat_btn_row});

    //

    auto cur_agent_menu = Menu(&cur_agent_entries, &cur_agent_sel, MenuOption::Vertical());
    auto cur_conv_menu  = Menu(&cur_conv_entries, &cur_conv_sel, MenuOption::Vertical());
    auto cur_mem_menu   = Menu(&cur_mem_entries, &cur_mem_sel, MenuOption::Vertical());
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
        Agent* a = agents_.get_agent(agents[cur_agent_sel].id);
        if (!a) return;

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
        set_curation_status("done: create conversation", "");
        refresh();
    }, ButtonOption::Simple());

    auto btn_cur_activate = Button(" Set Active ", [&] {
        auto agents = agents_.list_agents();
        if (agents.empty() || cur_agent_sel < 0 || cur_agent_sel >= static_cast<int>(agents.size())) return;
        Agent* a = agents_.get_agent(agents[cur_agent_sel].id);
        if (!a) return;
        auto convs = a->db().list_conversations();
        if (cur_conv_sel < 0 || cur_conv_sel >= static_cast<int>(convs.size())) return;
        a->db().set_active_conversation(convs[cur_conv_sel].id);
        set_curation_status("done: set active conversation", "");
        refresh();
    }, ButtonOption::Simple());

    auto btn_cur_compact = Button(" Force Compact ", [&] {
        auto agents = agents_.list_agents();
        if (agents.empty() || cur_agent_sel < 0 || cur_agent_sel >= static_cast<int>(agents.size())) return;
        const auto& agent_id = agents[cur_agent_sel].id;
        Agent* a = agents_.get_agent(agent_id);
        if (!a) return;
        auto convs = a->db().list_conversations();
        if (cur_conv_sel < 0 || cur_conv_sel >= static_cast<int>(convs.size())) return;
        const std::string conv_id = convs[cur_conv_sel].id;

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
        Agent* a = agents_.get_agent(agent_id);
        if (!a) return;
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
    }, ButtonOption::Simple());

    auto btn_cur_extract = Button(" Generate Memories ", [&] {
        auto agents = agents_.list_agents();
        if (agents.empty() || cur_agent_sel < 0 || cur_agent_sel >= static_cast<int>(agents.size())) return;
        const auto& agent_id = agents[cur_agent_sel].id;
        Agent* a = agents_.get_agent(agent_id);
        if (!a) return;
        auto convs = a->db().list_conversations();
        if (cur_conv_sel < 0 || cur_conv_sel >= static_cast<int>(convs.size())) return;
        auto conv = a->db().load_conversation(convs[cur_conv_sel].id);
        if (!conv) return;

        int start_i = parse_int_or(cur_start_s, 0);
        int end_i = parse_int_or(cur_end_s, start_i);
        int ctx_i = std::clamp(parse_int_or(cur_context_before_s, 2), 0, 20);
        if (start_i < 0 || end_i < 0 || start_i > end_i ||
            end_i >= static_cast<int>(conv->messages.size())) {
            set_curation_status("failed", "invalid message range");
            refresh();
            return;
        }

        const std::string conv_id = conv->id;
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
        Agent* a = agents_.get_agent(agent_id);
        if (!a) return;
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
        Agent* a = agents_.get_agent(cur_delete_agent_id);
        if (!a) {
            show_cur_delete_confirm = false;
            return;
        }

        if (cur_delete_is_memory) {
            a->db().delete_memory(cur_delete_target_id);
            set_curation_status("done: delete memory", "");
        } else {
            if (a->db().is_conversation_active(cur_delete_target_id)) {
                set_curation_status("failed", "cannot delete active conversation; activate another first");
            } else {
                a->db().delete_conversation(cur_delete_target_id);
                set_curation_status("done: delete conversation", "");
            }
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

    // Tab 1
    auto nodes_renderer = Renderer(nodes_comp, [&]() {
        //
        auto dns = registry_.get_discovered_nodes();
        disc_entries.resize(dns.size());
        for (size_t i = 0; i < dns.size(); ++i)
            disc_entries[i] = dns[i].url + "  [" + dns[i].node_id.substr(0, 8) + "...]";
        // Clamp: when empty, leave sel at 0 (Maybe wrapper prevents menu access)
        if (!dns.empty() && disc_sel >= static_cast<int>(dns.size()))
            disc_sel = static_cast<int>(dns.size()) - 1;
        else if (dns.empty())
            disc_sel = 0;

        //
        auto ns = registry_.list_nodes();
        node_entries.resize(ns.size());
        for (size_t i = 0; i < ns.size(); ++i) {
            auto& n = ns[i];
            char buf[128];
            snprintf(buf, sizeof(buf), "%-35s  %-18s  [%s]",
                     n.url.c_str(),
                     n.loaded_model.empty() ? "(no model)"
                         : n.loaded_model.substr(0, 18).c_str(),
                     to_string(n.health).c_str());
            node_entries[i] = buf;
        }
        if (!ns.empty() && node_sel >= static_cast<int>(ns.size()))
            node_sel = static_cast<int>(ns.size()) - 1;
        else if (ns.empty())
            node_sel = 0;

        refresh_control_catalog(false);
        catalog_entries = control_catalog_filenames;
        if (!catalog_entries.empty() && catalog_sel >= static_cast<int>(catalog_entries.size()))
            catalog_sel = static_cast<int>(catalog_entries.size()) - 1;
        else if (catalog_entries.empty())
            catalog_sel = 0;

        if (!ns.empty() && node_sel >= 0 && node_sel < static_cast<int>(ns.size())) {
            refresh_node_storage_cache(ns[node_sel].id, false);
        } else {
            node_models_cached_node_id.clear();
            node_models_cached.clear();
            node_models_cached_disk_free_mb = 0;
            node_model_sel = 0;
        }

        node_model_entries.resize(node_models_cached.size());
        for (size_t i = 0; i < node_models_cached.size(); ++i) {
            const auto& m = node_models_cached[i];
            node_model_entries[i] = m.model_path + "  (" + mb_str(m.size_bytes / (1024 * 1024)) + ")";
        }
        if (!node_model_entries.empty() && node_model_sel >= static_cast<int>(node_model_entries.size()))
            node_model_sel = static_cast<int>(node_model_entries.size()) - 1;
        else if (node_model_entries.empty())
            node_model_sel = 0;

        Element detail;
        if (!ns.empty() && node_sel < static_cast<int>(ns.size())) {
            auto& n  = ns[node_sel];
            auto ram = mb_str(n.metrics.ram_used_mb) + "/" + mb_str(n.metrics.ram_total_mb);
            auto gpu = n.metrics.gpu_vram_total_mb > 0
                           ? mb_str(n.metrics.gpu_vram_used_mb) + "/" +
                             mb_str(n.metrics.gpu_vram_total_mb)
                           : std::string{};
            int slot_ready = 0;
            int slot_loading = 0;
            int slot_suspending = 0;
            int slot_suspended = 0;
            int slot_error = 0;
            for (const auto& s : n.slots) {
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
            const int slot_available = std::max(0, n.max_slots - slot_in_use);

            auto compact_id = [](const std::string& s, size_t max_len = 8) -> std::string {
                if (s.empty()) return "-";
                return s.size() <= max_len ? s : s.substr(0, max_len);
            };
            auto compact_model = [](const std::string& p) -> std::string {
                if (p.empty()) return "(none)";
                std::string name = std::filesystem::path(p).filename().string();
                if (name.empty()) name = p;
                if (name.size() > 28) name = name.substr(0, 28) + "...";
                return name;
            };

            Elements detail_rows = {
                hbox({text("  URL     : "), text(n.url) | bold}),
                hbox({text("  Platform: "), text(n.platform.empty() ? "-" : n.platform)}),
                hbox({text("  Model   : "),
                      text(n.loaded_model.empty() ? "(none)" : n.loaded_model) | bold}),
                hbox({text("  Slots   : "),
                      text(std::to_string(slot_in_use) + "/" + std::to_string(n.max_slots)
                           + " in use  (ready " + std::to_string(slot_ready)
                           + ", loading " + std::to_string(slot_loading)
                           + ", suspended " + std::to_string(slot_suspended)
                           + ", available " + std::to_string(slot_available) + ")")
                          | bold}),
                hbox({text("  Llama   : "),
                      text(n.llama_server_path.empty() ? "(unknown)" : n.llama_server_path)
                          | color(Color::GrayDark)}),
                hbox({text("  Updater : "),
                      text(n.llama_update_status.empty() ? "idle" : n.llama_update_status)
                          | (n.llama_update_status == "failed"
                                 ? color(Color::RedLight)
                                 : (n.llama_update_status == "succeeded"
                                        ? color(Color::Green)
                                        : color(Color::Yellow)))}),
                hbox({text("  Versions: "),
                      text((n.llama_installed_commit.empty() ? "?" : n.llama_installed_commit)
                          + " -> "
                          + (n.llama_remote_commit.empty() ? "?" : n.llama_remote_commit))
                          | color(Color::GrayDark)}),
                hbox({text("  Needs update: "),
                      text(n.llama_update_available ? "yes" : "no")
                          | (n.llama_update_available ? color(Color::Yellow) : color(Color::Green))}),
                hbox({text("  Update msg: "),
                      text(n.llama_update_message.empty() ? "(none)" : n.llama_update_message)
                          | color(Color::GrayDark)}),
                hbox({text("  Storage : "),
                      text(std::to_string(node_models_cached.size()) + " model(s), free " +
                           mb_str(node_models_cached_disk_free_mb))
                          | color(Color::GrayDark)}),
                separator(),
                gauge_row("  CPU", n.metrics.cpu_percent, ""),
                gauge_row("  RAM", n.metrics.ram_percent, ram),
                gauge_row("  GPU", n.metrics.gpu_percent, gpu),
                separator(),
                text("  Slot occupancy:") | bold,
            };

            if (n.slots.empty()) {
                detail_rows.push_back(text("    (no tracked slots)") | color(Color::GrayDark));
            } else {
                constexpr size_t kMaxSlotLines = 6;
                size_t shown = 0;
                for (const auto& s : n.slots) {
                    if (shown >= kMaxSlotLines) break;
                    const std::string state_txt = to_string(s.state);
                    const std::string slot_txt = compact_id(s.id, 10);
                    const std::string agent_txt = compact_id(s.assigned_agent, 12);
                    const std::string model_txt = compact_model(s.model_path);
                    detail_rows.push_back(
                        text("    [" + state_txt + "] slot:" + slot_txt +
                             " agent:" + agent_txt + " model:" + model_txt));
                    ++shown;
                }
                if (n.slots.size() > shown) {
                    detail_rows.push_back(
                        text("    ... +" + std::to_string(n.slots.size() - shown) + " more slot(s)")
                        | color(Color::GrayDark));
                }
            }

            detail_rows.push_back(separator());
            detail_rows.push_back(text("  Stored models on selected node:") | bold);
            if (node_models_cached.empty()) {
                detail_rows.push_back(text("    (none)") | color(Color::GrayDark));
            } else {
                constexpr size_t kMaxModelLines = 8;
                size_t shown = 0;
                for (const auto& m : node_models_cached) {
                    if (shown >= kMaxModelLines) break;
                    detail_rows.push_back(
                        text("    - " + compact_model(m.model_path) +
                             " (" + mb_str(m.size_bytes / (1024 * 1024)) + ")")
                        | color(Color::GrayDark));
                    ++shown;
                }
                if (node_models_cached.size() > shown) {
                    detail_rows.push_back(
                        text("    ... +" + std::to_string(node_models_cached.size() - shown) + " more")
                        | color(Color::GrayDark));
                }
            }

            detail = vbox(std::move(detail_rows));
        } else {
            detail = text("  No connected nodes.") | color(Color::GrayDark);
        }

        return vbox({
            vbox({
                text(" Discovered Nodes") | bold,
                separator(),
                dns.empty()
                    ? text("  Listening for nodes...") | color(Color::GrayDark)
                    : (disc_menu->Render() | yframe | flex),
                disc_btns->Render(),
            }) | size(HEIGHT, LESS_THAN, 10) | border,
            vbox({
                text(" Connected Nodes") | bold,
                separator(),
                ns.empty()
                    ? text("  No connected nodes.") | color(Color::GrayDark)
                    : (node_menu->Render() | yframe | flex),
                node_btns->Render(),
            }) | size(HEIGHT, LESS_THAN, 10) | border,
            vbox({
                text(" Model Management") | bold,
                separator(),
                hbox({
                    vbox({
                        text(" Control Catalog") | bold,
                        separator(),
                        catalog_entries.empty()
                            ? (text("  (no .gguf files in control models_dir)") | color(Color::GrayDark))
                            : (catalog_menu->Render() | yframe | flex),
                    }) | size(WIDTH, EQUAL, 58) | border,
                    separator(),
                    vbox({
                        text(" Selected Node Inventory") | bold,
                        separator(),
                        node_model_entries.empty()
                            ? (text("  (no models on selected node)") | color(Color::GrayDark))
                            : (node_models_menu->Render() | yframe | flex),
                    }) | border | flex,
                }) | size(HEIGHT, LESS_THAN, 14) | flex,
                separator(),
                node_model_btns->Render(),
                node_model_action_status.empty()
                    ? text("  p:pull selected catalog model  d:delete selected node model")
                        | color(Color::GrayDark)
                    : text("  " + node_model_action_status)
                        | color(Color::Yellow),
            }) | border,
            separator(),
            detail,
        });
    });

    // Tab 2
    auto agents_renderer = Renderer(agents_comp, [&]() {
        auto cs = agents_.list_agents();

        agent_entries.resize(cs.size());
        for (size_t i = 0; i < cs.size(); ++i) {
            auto& a  = cs[i];
            auto model_disp = a.model_path.empty() ? "no model"
                : (a.model_path.size() > 22 ? a.model_path.substr(0, 22) + "..."
                                             : a.model_path);
            agent_entries[i] = a.name + "  (" + model_disp + ")";
        }
        if (agent_sel >= static_cast<int>(cs.size()) && !cs.empty())
            agent_sel = static_cast<int>(cs.size()) - 1;

        if (show_editor) {
            refresh_editor_validation();

            Elements warning_lines;
            for (const auto& issue : ed_validation_issues) {
                if (issue.severity != ValidationSeverity::Warning) continue;
                warning_lines.push_back(paragraph(" - " + issue.field + ": " + issue.message) | color(Color::Yellow));
            }
            if (warning_lines.empty()) {
                warning_lines.push_back(text(" No current warnings.") | color(Color::GrayDark));
            }

            std::vector<ToolDefinition> local_tools;
            if (ed_tools && ed_memories) local_tools = ToolExecutor::local_tool_catalog();

            Elements tool_lines;
            if (!ed_tools) {
                tool_lines.push_back(text(" Tools disabled.") | color(Color::GrayDark));
            } else if (!ed_memories) {
                tool_lines.push_back(paragraph(" No executable tools until Memories is enabled.") | color(Color::Yellow));
            } else {
                for (const auto& tool : local_tools) {
                    tool_lines.push_back(text(" - " + tool.name) | color(Color::Cyan));
                    tool_lines.push_back(paragraph("   " + tool.description) | color(Color::GrayDark));
                }
            }

            return vbox({
                text(ed_id.empty() ? " New Agent" : " Edit: " + ed_name) | bold,
                separator(),
                hbox({
                    // Left column: fields
                    vbox({
                        hbox({text(" ID            : "), ed_inp_id->Render()    | flex}),
                        hbox({text(" Name          : "), ed_inp_name->Render()  | flex}),
                        hbox({text(" Model path    : "), ed_inp_model->Render() | flex,
                              text(" "), btn_browse_model->Render()}),
                        hbox({text(" Preferred node: "), ed_inp_pnode->Render() | flex}),
                        separator(),
                        text(" System Prompt:"),
                        ed_inp_sys->Render() | size(HEIGHT, LESS_THAN, 5) | border,
                        separator(),
                        text(" LLM Settings:"),
                        hbox({text("  ctx_size    : "), ed_inp_ctx->Render()  | flex,
                              text("  max_tokens  : "), ed_inp_max->Render()  | flex}),
                        hbox({text("  gpu_layers  : "), ed_inp_gpu->Render()  | flex,
                              text("  threads     : "), ed_inp_thr->Render()  | flex}),
                        hbox({text("  temperature : "), ed_inp_temp->Render() | flex,
                              text("  top_p       : "), ed_inp_topp->Render() | flex}),
                        separator(),
                        text(" Extra llama args (one per line):"),
                        ed_inp_extra->Render() | size(HEIGHT, LESS_THAN, 5) | border,
                    }) | flex,
                    separator(),
                    // Right column: toggles + status
                    vbox({
                        text(" Options:"),
                        ed_cb_flash->Render(),
                        ed_cb_reasoning->Render(),
                        ed_cb_memories->Render(),
                        ed_cb_tools->Render(),
                        separator(),
                        text(" Model Capabilities:") | bold,
                        text(" Context limit: " +
                             (ed_model_info.n_ctx_train > 0
                                  ? std::to_string(ed_model_info.n_ctx_train) + " tokens"
                                  : "unknown")),
                        text(" Tool calling: " + capability_status(
                                 ed_model_info.supports_tool_calls,
                                 ed_model_info.metadata_found,
                                 ed_model_info.used_filename_heuristics)),
                        text(" Reasoning: " + capability_status(
                                 ed_model_info.supports_reasoning,
                                 ed_model_info.metadata_found,
                                 ed_model_info.used_filename_heuristics)),
                        paragraph(" Source: " +
                                  (ed_model_info.source_path.empty() ? "(unresolved)" : ed_model_info.source_path))
                            | color(Color::GrayDark),
                        separator(),
                        text(" Available Tools:") | bold,
                        vbox(std::move(tool_lines)) | size(HEIGHT, LESS_THAN, 8) | yframe,
                        separator(),
                        text(" Warnings:") | bold,
                        vbox(std::move(warning_lines)) | size(HEIGHT, LESS_THAN, 8) | yframe,
                        separator(),
                        text(""),
                        ed_form_btns->Render(),
                        text(""),
                        text(" Esc to cancel") | color(Color::GrayDark),
                    }) | size(WIDTH, EQUAL, 44),
                }),
            });
        }

        // List view with detail panel
        Element detail;
        if (cs.empty()) {
            detail = text("  No agents. Press [+] New to create one.")
                     | color(Color::GrayDark);
        } else if (agent_sel < static_cast<int>(cs.size())) {
            auto& a = cs[agent_sel];
            auto feat = [](bool v, const std::string& label) {
                return hbox({text(label + ": "),
                             text(v ? "on" : "off")
                                 | color(v ? Color::Green : Color::GrayDark)});
            };
            detail = vbox({
                hbox({text("  Model  : "),
                      text(a.model_path.empty() ? "(none)" : a.model_path) | bold}),
                hbox({text("  System : "),
                      text(a.system_prompt.empty() ? "(none)"
                               : a.system_prompt.substr(0, 60) + "...")
                          | color(Color::Cyan)}),
                hbox({text("  ctx    : "),
                      text(std::to_string(a.llama_settings.ctx_size) + " tokens")}),
                hbox({text("  "),
                      feat(a.reasoning_enabled, "reasoning"),
                      text("   "),
                      feat(a.memories_enabled,  "memories"),
                      text("   "),
                      feat(a.tools_enabled,     "tools")}),
                text(""),
                text("  POST /v1/agents/" + a.id + "/chat") | color(Color::GrayDark),
            });
        } else {
            detail = text("");
        }

        return vbox({
            vbox({
                agent_menu->Render() | yframe | flex,
                agent_list_btns->Render(),
            }) | size(HEIGHT, LESS_THAN, 14) | border,
            separator(),
            detail,
        });
    });

    // Tab 3
    auto activity_renderer = Renderer(activity_comp, [&]() {
        std::vector<LogEntry> snap;
        {
            std::lock_guard<std::mutex> lk(log_mutex_);
            snap = log_entries_;
        }

        Elements lines;
        for (int i = static_cast<int>(snap.size()) - 1;
             i >= 0 && lines.size() < 300; --i) {
            auto& e = snap[i];
            if (log_filter == 1 && e.level != LogLevel::Info)  continue;
            if (log_filter == 2 && e.level != LogLevel::Warn)  continue;
            if (log_filter == 3 && e.level != LogLevel::Error) continue;
            Color c = Color::White;
            if (e.level == LogLevel::Warn)  c = Color::Yellow;
            if (e.level == LogLevel::Error) c = Color::Red;
            std::string prefix = e.level == LogLevel::Warn  ? "[WARN]  "
                                : e.level == LogLevel::Error ? "[ERROR] "
                                                              : "[INFO]  ";
            std::string ts;
            if (e.timestamp_ms > 0) {
                time_t sec = static_cast<time_t>(e.timestamp_ms / 1000);
                struct tm tm_buf{};
#ifdef _WIN32
                localtime_s(&tm_buf, &sec);
#else
                localtime_r(&sec, &tm_buf);
#endif
                char tbuf[16];
                snprintf(tbuf, sizeof(tbuf), "%02d:%02d:%02d ",
                         tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
                ts = tbuf;
            }
            lines.push_back(text(ts + prefix + e.message) | color(c));
        }
        if (lines.empty())
            lines.push_back(text("  (no events yet)") | color(Color::GrayDark));

        return vbox({
            hbox({text(" Filter: "), filter_toggle->Render()}),
            separator(),
            vbox(std::move(lines)) | yframe | flex,
        });
    });

    // Tab 4
    auto chat_renderer = Renderer(chat_comp, [&]() {
        auto cs = agents_.list_agents();

        chat_agent_entries.resize(cs.size());
        for (size_t i = 0; i < cs.size(); ++i) {
            auto& a = cs[i];
            chat_agent_entries[i] = a.name + " [" + a.id + "]";
        }
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

        std::vector<std::string> transcript_snapshot;
        std::string partial_snapshot;
        std::string status_snapshot;
        std::string error_snapshot;
        std::string conv_snapshot;
        {
            std::lock_guard<std::mutex> lk(chat_mutex);
            transcript_snapshot = chat_transcript;
            partial_snapshot = chat_partial_assistant;
            status_snapshot = chat_status;
            error_snapshot = chat_last_error;
            conv_snapshot = chat_last_conv_id;
        }

        Elements transcript_lines;
        for (const auto& line : transcript_snapshot) {
            transcript_lines.push_back(text("  " + line));
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

        Color status_color = Color::GrayDark;
        if (status_snapshot == "done") status_color = Color::Green;
        if (status_snapshot == "streaming" || status_snapshot == "sending") status_color = Color::Yellow;
        if (status_snapshot == "failed") status_color = Color::Red;

        return vbox({
            hbox({
                text(" Connected nodes: " + std::to_string(connected_nodes)),
                text("  |  "),
                text(" Selected agent: " + selected_agent),
                text("  |  "),
                text(" Status: " + status_snapshot) | color(status_color),
            }),
            error_snapshot.empty()
                ? text("")
                : (text(" Last error: " + error_snapshot) | color(Color::Red)),
            conv_snapshot.empty()
                ? text("")
                : (text(" Last conversation: " + conv_snapshot) | color(Color::GrayDark)),
            separator(),
            hbox({
                vbox({
                    text(" Agent") | bold,
                    separator(),
                    cs.empty()
                        ? (text("  No agents configured.") | color(Color::GrayDark))
                        : (chat_agent_menu->Render() | yframe | flex),
                    separator(),
                    text(" Prompt") | bold,
                    chat_input_comp->Render(),
                    chat_btn_row->Render(),
                    text(chat_inflight.load()
                             ? "  Send is blocked while a stream is active."
                             : "  Send a prompt to test this agent.")
                        | color(Color::GrayDark),
                }) | size(WIDTH, LESS_THAN, 52) | border,
                separator(),
                vbox({
                    text(" Transcript") | bold,
                    separator(),
                    vbox(std::move(transcript_lines)) | yframe | flex,
                }) | border | flex,
            }) | flex,
        });
    });

    // Tab 5
    auto curation_renderer = Renderer(curation_comp, [&]() {
        auto cs = agents_.list_agents();
        cur_agent_entries.resize(cs.size());
        for (size_t i = 0; i < cs.size(); ++i) {
            cur_agent_entries[i] = cs[i].name + " [" + cs[i].id + "]";
        }
        if (cur_agent_sel >= static_cast<int>(cs.size()) && !cs.empty())
            cur_agent_sel = static_cast<int>(cs.size()) - 1;
        else if (cs.empty())
            cur_agent_sel = 0;

        std::vector<Conversation> convs;
        std::vector<Memory> mems;
        std::optional<Conversation> conv_detail;
        std::optional<Memory> selected_mem;
        std::string selected_agent = "(none)";

        if (!cs.empty() && cur_agent_sel >= 0 && cur_agent_sel < static_cast<int>(cs.size())) {
            selected_agent = cs[cur_agent_sel].name + " [" + cs[cur_agent_sel].id + "]";
            Agent* a = agents_.get_agent(cs[cur_agent_sel].id);
            if (a) {
                convs = a->db().list_conversations();
                mems  = a->db().list_memories();
            }
        }

        cur_conv_entries.resize(convs.size());
        for (size_t i = 0; i < convs.size(); ++i) {
            const auto& c = convs[i];
            std::string title = c.title.empty() ? "(untitled)" : c.title;
            if (title.size() > 28) title = title.substr(0, 28) + "...";
            std::string prefix = c.is_active ? "* " : "  ";
            std::string parent = c.parent_conv_id.empty() ? "" : (" <- " + c.parent_conv_id.substr(0, 8));
            cur_conv_entries[i] = prefix + title + " [" + std::to_string(c.total_tokens) + " tok]" + parent;
        }
        if (cur_conv_sel >= static_cast<int>(convs.size()) && !convs.empty())
            cur_conv_sel = static_cast<int>(convs.size()) - 1;
        else if (convs.empty())
            cur_conv_sel = 0;

        if (!convs.empty() && cur_conv_sel >= 0 && cur_conv_sel < static_cast<int>(convs.size())) {
            Agent* a = agents_.get_agent(cs[cur_agent_sel].id);
            if (a) conv_detail = a->db().load_conversation(convs[cur_conv_sel].id);
        }

        cur_mem_entries.resize(mems.size());
        for (size_t i = 0; i < mems.size(); ++i) {
            const auto& m = mems[i];
            char imp[12];
            snprintf(imp, sizeof(imp), "%.2f", static_cast<double>(m.importance));
            std::string content = m.content;
            if (content.size() > 44) content = content.substr(0, 44) + "...";
            cur_mem_entries[i] = std::string("[") + imp + "] " + content;
        }
        if (cur_mem_sel >= static_cast<int>(mems.size()) && !mems.empty())
            cur_mem_sel = static_cast<int>(mems.size()) - 1;
        else if (mems.empty())
            cur_mem_sel = 0;

        if (!mems.empty() && cur_mem_sel >= 0 && cur_mem_sel < static_cast<int>(mems.size())) {
            selected_mem = mems[cur_mem_sel];
        }

        Elements conv_detail_lines;
        if (conv_detail && !conv_detail->messages.empty()) {
            const auto& msgs = conv_detail->messages;
            const std::string conv_title = conv_detail->title.empty() ? "(untitled)" : conv_detail->title;
            conv_detail_lines.push_back(text(" Title: " + conv_title) | bold);
            conv_detail_lines.push_back(text(" ID: " + conv_detail->id) | dim);
            conv_detail_lines.push_back(
                text(" Messages: " + std::to_string(msgs.size()) +
                     "  |  Tokens: " + std::to_string(conv_detail->total_tokens) +
                     "  |  Active: " + std::string(conv_detail->is_active ? "yes" : "no"))
            );
            if (!conv_detail->parent_conv_id.empty()) {
                conv_detail_lines.push_back(text(" Parent: " + conv_detail->parent_conv_id) | dim);
            }
            conv_detail_lines.push_back(separator());

            for (size_t i = 0; i < msgs.size(); ++i) {
                const auto& msg = msgs[i];
                const std::string role = to_string(msg.role);
                conv_detail_lines.push_back(
                    text(" [" + std::to_string(i) + "] " + role +
                         "  |  tokens: " + std::to_string(msg.token_count))
                    | bold
                );

                const auto content_lines = util::split(msg.content.empty() ? "(empty)" : msg.content, '\n');
                for (const auto& line : content_lines) {
                    conv_detail_lines.push_back(paragraph("   " + (line.empty() ? " " : line)));
                }

                if (!msg.thinking_text.empty()) {
                    conv_detail_lines.push_back(text("   Thinking:") | dim);
                    const auto thinking_lines = util::split(msg.thinking_text, '\n');
                    for (const auto& line : thinking_lines) {
                        conv_detail_lines.push_back(paragraph("   " + (line.empty() ? " " : line)) | dim);
                    }
                }

                if (!msg.tool_calls.empty()) {
                    conv_detail_lines.push_back(
                        text("   Tool calls: " + std::to_string(msg.tool_calls.size())) | color(Color::Cyan)
                    );
                }
                if (!msg.tool_call_id.empty()) {
                    conv_detail_lines.push_back(text("   Tool call id: " + msg.tool_call_id) | dim);
                }
                if (i + 1 < msgs.size()) {
                    conv_detail_lines.push_back(separatorLight());
                }
            }

            int start_i = parse_int_or(cur_start_s, 0);
            int end_i = parse_int_or(cur_end_s, 0);
            const int last_idx = static_cast<int>(msgs.size()) - 1;
            if (start_i < 0 || start_i > last_idx) cur_start_s = "0";
            if (end_i < 0 || end_i > last_idx) cur_end_s = std::to_string(last_idx);
        } else {
            conv_detail_lines.push_back(text("  (select a conversation)") | color(Color::GrayDark));
            cur_start_s = "0";
            cur_end_s = "0";
        }

        Elements mem_detail_lines;
        if (selected_mem) {
            char imp[12];
            snprintf(imp, sizeof(imp), "%.2f", static_cast<double>(selected_mem->importance));
            mem_detail_lines.push_back(text(" Importance: " + std::string(imp)) | bold);
            mem_detail_lines.push_back(text(" ID: " + selected_mem->id) | dim);
            if (!selected_mem->source_conv_id.empty()) {
                mem_detail_lines.push_back(text(" Source conversation: " + selected_mem->source_conv_id) | dim);
            }
            mem_detail_lines.push_back(separator());
            const auto mem_lines = util::split(
                selected_mem->content.empty() ? "(empty)" : selected_mem->content,
                '\n');
            for (const auto& line : mem_lines) {
                mem_detail_lines.push_back(paragraph(" " + (line.empty() ? " " : line)));
            }
        } else {
            mem_detail_lines.push_back(text("  (select a memory)") | color(Color::GrayDark));
        }

        std::string status_snapshot;
        std::string error_snapshot;
        {
            std::lock_guard<std::mutex> lk(cur_status_mutex);
            status_snapshot = cur_status;
            error_snapshot = cur_last_error;
        }
        Color status_color = Color::GrayDark;
        if (status_snapshot.rfind("done:", 0) == 0) status_color = Color::Green;
        else if (status_snapshot.rfind("running:", 0) == 0) status_color = Color::Yellow;
        else if (status_snapshot == "failed") status_color = Color::Red;

        return vbox({
            hbox({
                text(" Selected agent: " + selected_agent),
                text("  |  "),
                text(" Status: " + status_snapshot) | color(status_color),
                text(cur_inflight.load() ? "  (busy)" : ""),
            }),
            error_snapshot.empty() ? text("")
                                   : (text(" Last error: " + error_snapshot) | color(Color::Red)),
            separator(),
            hbox({
                vbox({
                    text(" Agent") | bold,
                    separator(),
                    cs.empty() ? (text("  No agents configured.") | color(Color::GrayDark))
                               : (cur_agent_menu->Render() | yframe | flex),
                    separator(),
                    text(" New Conversation") | bold,
                    cur_new_title_input->Render(),
                    cur_new_active_cb->Render(),
                    cur_new_parent_cb->Render(),
                    btn_cur_new_conv->Render(),
                }) | size(WIDTH, EQUAL, 38) | border,
                separator(),
                vbox({
                    text(" Conversations") | bold,
                    separator(),
                    convs.empty() ? (text("  No conversations.") | color(Color::GrayDark))
                                  : (cur_conv_menu->Render() | yframe | flex),
                    separator(),
                    text(" Actions") | dim,
                    hbox({btn_cur_activate->Render(), text(" "), btn_cur_compact->Render()}),
                    btn_cur_delete_conv->Render(),
                }) | size(WIDTH, EQUAL, 52) | border,
                separator(),
                vbox({
                    text(" Memories") | bold,
                    separator(),
                    mems.empty() ? (text("  No memories.") | color(Color::GrayDark))
                                 : (cur_mem_menu->Render() | yframe | flex),
                    separator(),
                    btn_cur_delete_mem->Render(),
                }) | border | flex,
            }) | flex,
            separator(),
            vbox({
                text(" Memory Extraction") | bold,
                hbox({
                    text(" start:"), cur_start_input->Render() | size(WIDTH, EQUAL, 8),
                    text("  end:"), cur_end_input->Render() | size(WIDTH, EQUAL, 8),
                    text("  context_before:"), cur_ctx_input->Render() | size(WIDTH, EQUAL, 8),
                    text("  "), btn_cur_extract->Render(),
                }),
            }) | border,
            hbox({
                vbox({
                    text(" Conversation Detail") | bold,
                    separator(),
                    vbox(std::move(conv_detail_lines)) | yframe | flex,
                }) | border | flex,
                separator(),
                vbox({
                    text(" Selected Memory") | bold,
                    separator(),
                    vbox(std::move(mem_detail_lines)) | yframe | flex,
                }) | border | size(WIDTH, EQUAL, 46),
            }) | flex,
        });
    });

    //

    auto main_tabs = Container::Tab(
        {nodes_renderer, agents_renderer, activity_renderer, chat_renderer, curation_renderer}, &tab_index);

    auto top_renderer = Renderer(main_tabs, [&]() -> Element {
        auto tab_item = [&](int idx, const std::string& label) {
            return text(tab_index == idx ? " [" + label + "] " : "  " + label + "  ")
                   | (tab_index == idx ? bold : nothing);
        };
        auto header = hbox({
            text(" mantic-mind-control ") | bold,
            separator(),
            tab_item(0, "1: Nodes"),
            tab_item(1, "2: Agents"),
            tab_item(2, "3: Activity"),
            tab_item(3, "4: Chat"),
            tab_item(4, "5: Curation"),
            filler(),
            text(" q:quit  Esc:back ") | color(Color::GrayDark),
        });

        return vbox({
            header,
            separator(),
            main_tabs->Render() | flex,
        }) | border;
    });

    //

    auto root = CatchEvent(top_renderer, [&](Event ev) {
        if (show_add_node || show_pin_entry || show_agent_validation_modal ||
            show_file_browser || show_cur_delete_confirm) return false;
        if (!show_editor) {
            if (ev == Event::Character('1')) { tab_index = 0; return true; }
            if (ev == Event::Character('2')) { tab_index = 1; return true; }
            if (ev == Event::Character('3')) { tab_index = 2; return true; }
            if (ev == Event::Character('4')) { tab_index = 3; return true; }
            if (ev == Event::Character('5')) { tab_index = 4; return true; }
            if (ev == Event::Character('u') && tab_index == 0) {
                trigger_node_update(true);
                return true;
            }
            if (ev == Event::Character('p') && tab_index == 0) {
                pull_catalog_to_node();
                return true;
            }
            if (ev == Event::Character('d') && tab_index == 0) {
                delete_model_on_node();
                return true;
            }
            if (ev == Event::Character('r') && tab_index == 0) {
                refresh_control_catalog(true);
                auto ns = registry_.list_nodes();
                if (node_sel >= 0 && node_sel < static_cast<int>(ns.size())) {
                    refresh_node_storage_cache(ns[node_sel].id, true);
                }
                node_model_action_status = "refreshed model views";
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

    std::atomic<bool> ticker_running{true};
    std::thread ticker([&] {
        while (ticker_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            screen.PostEvent(Event::Custom);
        }
    });

    screen.Loop(final_comp);

    ticker_running = false;
    ticker.join();

    if (chat_thread.joinable()) {
        chat_thread.join();
    }
    if (cur_thread.joinable()) {
        cur_thread.join();
    }

    {
        std::lock_guard<std::mutex> lk(screen_mutex_);
        quit_fn_    = {};
        refresh_fn_ = {};
    }
}

} // namespace mm
