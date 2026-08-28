// Mantic-Mind — the Engines tab. See include/control/engine_panels.hpp for the
// rule this file exists under: everything here arrives over /v1/*.

#include "control/engine_panels.hpp"

#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/tui_widgets.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <condition_variable>

namespace mm {

using namespace ftxui;

bool vllm_controls_visible(const std::string& primary,
                           const std::string& backup) noexcept {
    return primary == "vllm" || backup == "vllm";
}

namespace {

/// Colour per conformance state. `converging` is deliberately NOT red: a build
/// in progress on a fresh cluster is the healthy path, and painting it as a
/// fault trains an operator to ignore the colour that matters.
Color state_color(const std::string& state) {
    if (state == "conforming") return Color::Green;
    if (state == "converging") return Color::Cyan;
    if (state == "drifted") return Color::Yellow;
    if (state == "failed") return Color::Red;
    return Color::GrayDark; // unconfigured
}

Color runtime_color(const EnginePanelRuntime& r) {
    if (r.ready) return Color::Green;
    if (r.status == "building") return Color::Cyan;
    if (r.status == "error") return Color::Red;
    return Color::GrayDark;
}

std::string truncate(const std::string& s, std::size_t n) {
    if (s.size() <= n) return s;
    if (n <= 1) return s.substr(0, n);
    return s.substr(0, n - 1) + "…";
}

} // namespace

// ── EngineDashboard ───────────────────────────────────────────────────────────

struct EngineDashboard::Impl {
    std::string base_url;
    std::string api_token;

    mutable std::mutex mutex;
    EngineSnapshot snap;

    std::atomic<bool> running{false};
    std::thread worker;
    std::mutex wake_mutex;
    std::condition_variable wake;
    std::atomic<bool> wake_now{false};

    HttpClient client() const {
        HttpClient c(base_url);
        if (!api_token.empty()) c.set_bearer_token(api_token);
        return c;
    }

    void poll() {
        EngineSnapshot next;
        auto c = client();

        const auto cfg_res = c.get("/v1/cluster/engines/config");
        if (!cfg_res.ok()) {
            // Reported, not swallowed. A tab that renders an empty policy when
            // it simply could not reach control is telling the operator the
            // cluster is unconfigured, which is a different and much more
            // alarming fact.
            next.reachable = false;
            next.error = "GET /v1/cluster/engines/config → HTTP " + std::to_string(cfg_res.status);
            std::lock_guard<std::mutex> g(mutex);
            snap = std::move(next);
            return;
        }
        next.reachable = true;
        try {
            const auto j = nlohmann::json::parse(cfg_res.body);
            next.configured = j.value("configured", false);
            if (j.contains("config")) {
                const auto& cf = j.at("config");
                next.config_version = cf.value("version", 0u);
                next.primary_engine = cf.value("primary_engine", std::string{});
                next.backup_engine = cf.value("backup_engine", std::string{});
                next.share_builds = cf.value("share_builds", true);
                try {
                    const auto parsed = cf.get<ClusterEngineConfig>();
                    if (const auto* spec = parsed.find("vllm")) {
                        next.vllm = effective_vllm_config(*spec);
                        next.vllm_install_method = spec->install_method;
                        next.vllm_version = spec->version;
                    }
                } catch (...) {
                }
            }
        } catch (const std::exception& e) {
            next.error = std::string("config parse: ") + e.what();
        }

        const auto ray_res = c.get("/v1/cluster/engines/ray");
        if (ray_res.ok()) {
            try {
                const auto ray = nlohmann::json::parse(ray_res.body);
                next.ray_state = ray.value("state", std::string{});
                next.ray_detail = ray.value("detail", std::string{});
                next.ray_active_groups = ray.value("active_groups", 0);
            } catch (...) {
            }
        }

        const auto conf_res = c.get("/v1/cluster/engines/conformance");
        if (conf_res.ok()) {
            try {
                const auto j = nlohmann::json::parse(conf_res.body);
                next.conforming = j.value("conforming", 0);
                for (const auto& n : j.value("nodes", nlohmann::json::array())) {
                    EnginePanelNode node;
                    node.node_id = n.value("node_id", std::string{});
                    node.hostname = n.value("hostname", std::string{});
                    node.url = n.value("url", std::string{});
                    node.connected = n.value("connected", false);
                    node.config_version = n.value("engine_config_version", 0u);
                    node.placement_eligible = n.value("placement_eligible", false);
                    if (n.contains("conformance")) {
                        const auto& cn = n.at("conformance");
                        node.state = cn.value("state", std::string{"unconfigured"});
                        node.detail = cn.value("detail", std::string{});
                        node.needs_artifact = cn.value("needs_artifact", std::string{});
                    }
                    for (const auto& e : n.value("engines", nlohmann::json::array())) {
                        EnginePanelRuntime r;
                        r.engine_id = e.value("engine_id", std::string{});
                        r.status = e.value("status", std::string{});
                        r.version = e.value("version", std::string{});
                        r.variant = e.value("variant", std::string{});
                        r.last_error = e.value("last_error", std::string{});
                        r.ready = e.value("ready", false);
                        if (!r.engine_id.empty() &&
                            std::find(next.known_engines.begin(),
                                      next.known_engines.end(),
                                      r.engine_id) == next.known_engines.end())
                            next.known_engines.push_back(r.engine_id);
                        node.engines.push_back(std::move(r));
                    }
                    if (n.contains("action_progress")) {
                        const auto& p = n.at("action_progress");
                        node.progress_active = p.value("active", false);
                        node.progress_action = p.value("action", std::string{});
                        node.progress_stage = p.value("stage", std::string{});
                        node.progress_fraction = p.value("fraction", -1.0);
                    }
                    next.nodes.push_back(std::move(node));
                }
            } catch (const std::exception& e) {
                next.error = std::string("conformance parse: ") + e.what();
            }
        }

        // On a cluster with no nodes yet there is nothing to learn engine ids
        // from, and the setup panel still has to offer a choice. The built-in
        // pair is a fallback for that case only — a reported id always wins.
        for (const std::string builtin : {"soma", "llama-cpp", "vllm"}) {
            if (std::find(next.known_engines.begin(), next.known_engines.end(), builtin) ==
                next.known_engines.end())
                next.known_engines.push_back(builtin);
        }
        std::sort(next.known_engines.begin(), next.known_engines.end());

        std::lock_guard<std::mutex> g(mutex);
        snap = std::move(next);
    }
};

EngineDashboard::EngineDashboard(std::string base_url, std::string api_token)
    : impl_(std::make_unique<Impl>()) {
    impl_->base_url = std::move(base_url);
    impl_->api_token = std::move(api_token);
}

EngineDashboard::~EngineDashboard() {
    stop();
}

void EngineDashboard::start(int interval_ms) {
    if (impl_->running.exchange(true)) return;
    impl_->worker = std::thread([this, interval_ms]() {
        while (impl_->running.load()) {
            impl_->poll();
            std::unique_lock<std::mutex> lk(impl_->wake_mutex);
            impl_->wake.wait_for(lk, std::chrono::milliseconds(interval_ms), [this] {
                return !impl_->running.load() || impl_->wake_now.exchange(false);
            });
        }
    });
}

void EngineDashboard::stop() {
    if (!impl_->running.exchange(false)) return;
    impl_->wake.notify_all();
    if (impl_->worker.joinable()) impl_->worker.join();
}

void EngineDashboard::refresh_now() {
    impl_->wake_now.store(true);
    impl_->wake.notify_all();
}

EngineSnapshot EngineDashboard::snapshot() const {
    std::lock_guard<std::mutex> g(impl_->mutex);
    return impl_->snap;
}

namespace {

/// Pull the server's `error` field out of a response body, falling back to the
/// status line. The server's message is the useful half — "'accelerator' may
/// not appear in the cluster engine config" versus "HTTP 400".
std::string server_error(const HttpResponse& r) {
    try {
        const auto j = nlohmann::json::parse(r.body);
        if (j.contains("error") && j.at("error").is_string())
            return j.at("error").get<std::string>();
    } catch (...) {
    }
    return "HTTP " + std::to_string(r.status);
}

} // namespace

bool EngineDashboard::save(const std::string& primary,
                           const std::string& backup,
                           bool share_builds,
                           const std::optional<VllmEngineConfig>& vllm,
                           const std::string& vllm_install_method,
                           const std::string& vllm_version,
                           std::string& out_error) {
    out_error.clear();
    std::optional<ClusterEngineConfig> existing;
    const auto current = impl_->client().get("/v1/cluster/engines/config");
    if (current.ok()) {
        try {
            const auto root = nlohmann::json::parse(current.body);
            if (root.value("configured", false))
                existing = root.at("config").get<ClusterEngineConfig>();
        } catch (...) {
        }
    }
    nlohmann::json engines = nlohmann::json::array();
    auto spec = [&](const std::string& id) {
        nlohmann::json out{{"engine_id", id}};
        if (existing) {
            if (const auto* previous = existing->find(id)) out = *previous;
        }
        if (id == "vllm") {
            out["install_method"] = vllm_install_method;
            out["version"] = vllm_version;
            out["vllm"] = vllm.value_or(VllmEngineConfig{});
        }
        return out;
    };
    engines.push_back(spec(primary));
    if (!backup.empty()) engines.push_back(spec(backup));

    const nlohmann::json body{{"primary_engine", primary},
                              {"backup_engine", backup},
                              {"engines", engines},
                              {"share_builds", share_builds}};
    const auto r = impl_->client().put("/v1/cluster/engines/config", body);
    if (!r.ok()) {
        out_error = server_error(r);
        return false;
    }
    refresh_now();
    return true;
}

bool EngineDashboard::resync(std::string& out_error) {
    out_error.clear();
    const auto r = impl_->client().post("/v1/cluster/engines/resync", nlohmann::json::object());
    if (!r.ok()) {
        out_error = server_error(r);
        return false;
    }
    refresh_now();
    return true;
}

bool EngineDashboard::share(const std::string& fingerprint,
                            const std::string& target_node_id,
                            const std::string& source_node_id,
                            std::string& out_error) {
    out_error.clear();
    nlohmann::json body{{"fingerprint", fingerprint}, {"target_node_id", target_node_id}};
    if (!source_node_id.empty()) body["source_node_id"] = source_node_id;
    const auto r = impl_->client().post("/v1/cluster/engines/share", body);
    if (!r.ok()) {
        out_error = server_error(r);
        return false;
    }
    refresh_now();
    return true;
}

// ── Renderers ─────────────────────────────────────────────────────────────────

Element render_engine_config(const EngineSnapshot& snap) {
    if (!snap.reachable) {
        return vbox({text(" Engine configuration ") | bold,
                     separator(),
                     text(" control is not answering") | color(Color::Red),
                     text(" " + snap.error) | dim}) |
               border;
    }
    if (!snap.configured) {
        return vbox({
                   text(" Engine configuration ") | bold,
                   separator(),
                   text(" NOT CONFIGURED") | bold | color(Color::Yellow),
                   text(" Nothing can be placed until a primary engine is set.") | dim,
                   text(" Every node is waiting to be told what to run.") | dim,
                   text(""),
                   text(" Press e to configure.") | color(Color::Cyan),
               }) |
               border;
    }

    Elements rows;
    rows.push_back(hbox(
        {text(" primary      ") | dim, text(snap.primary_engine) | bold | color(Color::Green)}));
    // "(none)" spelled out. An empty backup is a real configuration — a
    // Soma-only cluster that never compiles llama.cpp — and a blank cell would
    // read as a value nobody set.
    rows.push_back(
        hbox({text(" backup       ") | dim,
              snap.backup_engine.empty() ? text("(none)") | dim
                                         : text(snap.backup_engine) | color(Color::Cyan)}));
    rows.push_back(hbox({text(" share builds ") | dim, text(snap.share_builds ? "yes" : "no")}));
    rows.push_back(
        hbox({text(" version      ") | dim, text("v" + std::to_string(snap.config_version))}));
    if (!snap.error.empty()) rows.push_back(text(" " + snap.error) | color(Color::Yellow) | dim);

    return vbox({text(" Engine configuration ") | bold, separator(), vbox(rows)}) | border;
}

Element render_conformance_table(const EngineSnapshot& snap, int selected) {
    if (!snap.reachable)
        return vbox({text(" Cluster conformance ") | bold,
                     separator(),
                     text(" unavailable") | dim}) |
               border;
    if (snap.nodes.empty())
        return vbox({text(" Cluster conformance ") | bold,
                     separator(),
                     text(" no nodes registered") | dim}) |
               border;

    Elements rows;
    rows.push_back(hbox({text("  NODE            ") | dim | bold,
                         text("STATE       ") | dim | bold,
                         text("CFG  ") | dim | bold,
                         text("PLACE ") | dim | bold,
                         text("ENGINES") | dim | bold}));

    for (std::size_t i = 0; i < snap.nodes.size(); ++i) {
        const auto& n = snap.nodes[i];
        const std::string label =
            n.hostname.empty() ? truncate(n.node_id, 15) : truncate(n.hostname, 15);

        Elements engines;
        for (const auto& e : n.engines) {
            std::string cell = e.engine_id;
            if (!e.variant.empty()) cell += "/" + e.variant;
            engines.push_back(text(cell + " ") | color(runtime_color(e)));
        }
        if (engines.empty()) engines.push_back(text("—") | dim);

        Element row = hbox({
            text("  " + label + std::string(16 - std::min<std::size_t>(15, label.size()), ' ')),
            text(n.state + std::string(12 - std::min<std::size_t>(11, n.state.size()), ' ')) |
                color(state_color(n.state)),
            text("v" + std::to_string(n.config_version) + "   "),
            // The single most useful cell: "why is nothing scheduling here".
            text(n.placement_eligible ? "yes   " : "no    ") |
                color(n.placement_eligible ? Color::Green : Color::Red),
            hbox(engines),
        });
        if (static_cast<int>(i) == selected) row = std::move(row) | inverted;
        rows.push_back(std::move(row));

        // The reason, indented under its node. Only for states that stop
        // placement — a conforming node's detail is noise.
        if (!n.detail.empty() && n.state != "conforming")
            rows.push_back(text("      " + truncate(n.detail, 90)) | dim);
        if (!n.needs_artifact.empty())
            rows.push_back(text("      needs: " + n.needs_artifact) | color(Color::Yellow));
    }

    rows.push_back(text(""));
    rows.push_back(hbox({text("  " + std::to_string(snap.conforming) + "/" +
                              std::to_string(snap.nodes.size()) + " conforming") |
                             dim,
                         text("   j/k select") | dim}));

    return vbox({text(" Cluster conformance ") | bold, separator(), vbox(rows) | flex}) | border;
}

Element render_engine_activity(const EngineSnapshot& snap) {
    Elements rows;
    for (const auto& n : snap.nodes) {
        if (!n.progress_active) continue;
        const std::string label = n.hostname.empty() ? n.node_id : n.hostname;
        Elements line{text("  " + truncate(label, 14) + "  "),
                      text(n.progress_action + " ") | color(Color::Cyan)};
        if (!n.progress_stage.empty()) line.push_back(text("· " + n.progress_stage) | dim);
        rows.push_back(hbox(line));
        if (n.progress_fraction >= 0.0)
            rows.push_back(
                hbox({text("    "),
                      gauge(static_cast<float>(n.progress_fraction)) | flex | color(Color::Cyan)}));
    }
    // Explicit rather than empty: "nothing is building" and "the poll is dead"
    // must not render identically.
    if (rows.empty())
        rows.push_back(
            text(snap.reachable ? "  no builds or transfers in progress" : "  unavailable") | dim);

    return vbox({text(" Build activity ") | bold, separator(), vbox(rows)}) | border;
}

Element render_unconfigured_banner(const EngineSnapshot& snap) {
    if (!snap.reachable || snap.configured) return emptyElement();
    return hbox({
               text(" ⚠ no cluster engine configuration — nothing can be placed. ") | bold,
               text("Press 9, then e, to configure.") | dim,
           }) |
           bgcolor(Color::DarkRed) | color(Color::White);
}

// ── The tab ───────────────────────────────────────────────────────────────────

Component engine_tab(EngineDashboard& dashboard, int& selected_node, bool& force_setup) {
    // Form state lives with the tab rather than in the snapshot: a poll landing
    // mid-edit must not overwrite what the operator is typing.
    struct FormState {
        bool open = false;
        bool seeded = false;
        int primary_index = 0;
        int backup_index = 0; // 0 == "(none)"
        bool share_builds = true;
        std::string status;
        bool status_ok = false;
        std::vector<std::string> engine_choices;
        std::vector<std::string> backup_choices;
        bool show_vllm = false;
        std::string max_model_len = "4096";
        std::string max_num_seqs = "16";
        std::string max_num_batched_tokens = "-1";
        std::string tensor_parallel_size = "1";
        std::string pipeline_parallel_size = "1";
        std::string gpu_memory_utilization = "0.90";
        std::string dtype = "auto";
        std::string quantization;
        std::string tool_call_parser;
        std::string extra_args;
        bool trust_remote_code = false;
        bool enable_prefix_caching = true;
        bool enable_auto_tool_choice = false;
        bool enable_sleep_mode = true;
        bool allow_experimental_gloo = false;
        int vllm_install_index = 0;
        std::vector<std::string> vllm_install_choices{"auto", "wheel", "source", "path"};
        std::string vllm_version = "latest";
    };

    auto form = std::make_shared<FormState>();

    auto sync_choices = [form](const EngineSnapshot& snap) {
        form->engine_choices = snap.known_engines;
        form->backup_choices.assign(1, "(none)");
        for (const auto& e : snap.known_engines)
            form->backup_choices.push_back(e);
        if (form->seeded) {
            const auto primary = form->engine_choices.empty()
                ? std::string{} : form->engine_choices[static_cast<std::size_t>(std::clamp(
                    form->primary_index, 0,
                    static_cast<int>(form->engine_choices.size()) - 1))];
            const auto backup = form->backup_index > 0 &&
                form->backup_index < static_cast<int>(form->backup_choices.size())
                ? form->backup_choices[static_cast<std::size_t>(form->backup_index)]
                : std::string{};
            form->show_vllm = vllm_controls_visible(primary, backup);
            return;
        }
        // Seed from the stored config once, so reopening the form shows what is
        // configured rather than the first alphabetical engine.
        for (std::size_t i = 0; i < form->engine_choices.size(); ++i)
            if (form->engine_choices[i] == snap.primary_engine)
                form->primary_index = static_cast<int>(i);
        for (std::size_t i = 0; i < form->backup_choices.size(); ++i)
            if (form->backup_choices[i] == snap.backup_engine)
                form->backup_index = static_cast<int>(i);
        if (snap.backup_engine.empty()) form->backup_index = 0;
        form->share_builds = snap.share_builds;
        if (snap.vllm.has_value()) {
            const auto& v = *snap.vllm;
            form->max_model_len = std::to_string(v.max_model_len);
            form->max_num_seqs = std::to_string(v.max_num_seqs);
            form->max_num_batched_tokens = std::to_string(v.max_num_batched_tokens);
            form->tensor_parallel_size = std::to_string(v.tensor_parallel_size);
            form->pipeline_parallel_size = std::to_string(v.pipeline_parallel_size);
            form->gpu_memory_utilization = std::to_string(v.gpu_memory_utilization);
            form->dtype = v.dtype;
            form->quantization = v.quantization;
            form->tool_call_parser = v.tool_call_parser;
            form->extra_args = util::join(v.extra_args, ",");
            form->trust_remote_code = v.trust_remote_code;
            form->enable_prefix_caching = v.enable_prefix_caching;
            form->enable_auto_tool_choice = v.enable_auto_tool_choice;
            form->enable_sleep_mode = v.enable_sleep_mode;
            form->allow_experimental_gloo = v.allow_experimental_gloo;
        }
        for (std::size_t i = 0; i < form->vllm_install_choices.size(); ++i) {
            if (form->vllm_install_choices[i] == snap.vllm_install_method)
                form->vllm_install_index = static_cast<int>(i);
        }
        form->vllm_version = snap.vllm_version;
        if (snap.configured) form->seeded = true;
        form->show_vllm = vllm_controls_visible(
            snap.primary_engine, snap.backup_engine);
    };

    auto primary_toggle = Toggle(&form->engine_choices, &form->primary_index);
    auto backup_toggle = Toggle(&form->backup_choices, &form->backup_index);
    auto share_checkbox = Checkbox("share builds between nodes", &form->share_builds);
    auto max_model_len = Input(&form->max_model_len, "4096");
    auto max_num_seqs = Input(&form->max_num_seqs, "16");
    auto max_num_batched_tokens = Input(&form->max_num_batched_tokens, "-1");
    auto tp = Input(&form->tensor_parallel_size, "1");
    auto pp = Input(&form->pipeline_parallel_size, "1");
    auto gpu_util = Input(&form->gpu_memory_utilization, "0.90");
    auto dtype = Input(&form->dtype, "auto");
    auto quantization = Input(&form->quantization, "optional");
    auto tool_parser = Input(&form->tool_call_parser, "optional");
    auto extra_args = Input(&form->extra_args, "comma-separated unmodeled flags");
    auto trust_remote_code = Checkbox("trust remote model code", &form->trust_remote_code);
    auto prefix_caching = Checkbox("prefix caching", &form->enable_prefix_caching);
    auto auto_tool_choice = Checkbox("automatic tool choice", &form->enable_auto_tool_choice);
    auto sleep_mode = Checkbox("sleep mode", &form->enable_sleep_mode);
    auto experimental_gloo = Checkbox(
        "EXPERIMENTAL Gloo fallback (not guaranteed by upstream vLLM)",
        &form->allow_experimental_gloo);
    auto vllm_install = Toggle(
        &form->vllm_install_choices, &form->vllm_install_index);
    auto vllm_version = Input(&form->vllm_version, "latest");

    auto save_button = Button("Save", [form, &dashboard, &force_setup] {
        if (form->engine_choices.empty()) return;
        const std::string primary = form->engine_choices[static_cast<std::size_t>(
            std::clamp(form->primary_index, 0, static_cast<int>(form->engine_choices.size()) - 1))];
        std::string backup;
        if (form->backup_index > 0 &&
            form->backup_index < static_cast<int>(form->backup_choices.size()))
            backup = form->backup_choices[static_cast<std::size_t>(form->backup_index)];
        // Silently dropping a backup equal to the primary would be a surprise;
        // the server refuses it, and the message says so.
        std::optional<VllmEngineConfig> profile;
        if (primary == "vllm" || backup == "vllm") {
            try {
                VllmEngineConfig v;
                v.max_model_len = std::stoi(form->max_model_len);
                v.max_num_seqs = std::stoi(form->max_num_seqs);
                v.max_num_batched_tokens = std::stoi(form->max_num_batched_tokens);
                v.tensor_parallel_size = std::stoi(form->tensor_parallel_size);
                v.pipeline_parallel_size = std::stoi(form->pipeline_parallel_size);
                v.gpu_memory_utilization = std::stod(form->gpu_memory_utilization);
                v.dtype = util::trim(form->dtype);
                v.quantization = util::trim(form->quantization);
                v.tool_call_parser = util::trim(form->tool_call_parser);
                v.trust_remote_code = form->trust_remote_code;
                v.enable_prefix_caching = form->enable_prefix_caching;
                v.enable_auto_tool_choice = form->enable_auto_tool_choice;
                v.enable_sleep_mode = form->enable_sleep_mode;
                v.allow_experimental_gloo = form->allow_experimental_gloo;
                for (const auto& raw : util::split(form->extra_args, ',')) {
                    const auto arg = util::trim(raw);
                    if (!arg.empty()) v.extra_args.push_back(arg);
                }
                profile = std::move(v);
            } catch (const std::exception&) {
                form->status = "vLLM numeric fields contain an invalid value";
                form->status_ok = false;
                return;
            }
        }
        std::string err;
        const std::string install_method = form->vllm_install_choices[
            static_cast<std::size_t>(std::clamp(
                form->vllm_install_index, 0,
                static_cast<int>(form->vllm_install_choices.size()) - 1))];
        if (dashboard.save(primary, backup, form->share_builds, profile,
                           install_method, util::trim(form->vllm_version), err)) {
            form->status = "saved";
            form->status_ok = true;
            form->open = false;
            force_setup = false;
        } else {
            form->status = err;
            form->status_ok = false;
        }
    });

    auto resync_button = Button("Resync nodes", [form, &dashboard] {
        std::string err;
        form->status_ok = dashboard.resync(err);
        form->status = form->status_ok ? "re-pushed to every node behind the current version" : err;
    });

    auto close_button = Button("Close", [form] { form->open = false; });

    // ── Share ─────────────────────────────────────────────────────────────────
    //
    // Targets the SELECTED node and takes its fingerprint from that node's own
    // `needs_artifact`. The alternative — a text field for a fingerprint — would
    // make an operator hand-type `llama-cpp|b4321|linux|x86_64|cuda-12`, where a
    // single wrong field is a silent 404 rather than a wrong build. The cluster
    // already knows which node is waiting and for what; asking the operator to
    // restate it is asking them to be a worse copy of the data.
    //
    // `source_node_id` is left empty so control picks a holder. Control is the
    // only party that sees every node's artifacts, so choosing here would mean
    // choosing from a strictly smaller view.
    auto share_button = Button("Share to selected", [form, &dashboard, &selected_node] {
        const auto snap = dashboard.snapshot();
        if (selected_node < 0 || selected_node >= static_cast<int>(snap.nodes.size())) {
            form->status = "no node selected";
            form->status_ok = false;
            return;
        }
        const auto& node = snap.nodes[static_cast<std::size_t>(selected_node)];
        if (node.needs_artifact.empty()) {
            // Refused with the reason rather than sent and failed. A node that
            // is merely unhealthy is not a node waiting on a build, and the two
            // want completely different responses from an operator.
            form->status = (node.hostname.empty() ? node.node_id : node.hostname) +
                           " is not waiting on an engine build";
            form->status_ok = false;
            return;
        }
        std::string err;
        if (dashboard.share(node.needs_artifact, node.node_id, {}, err)) {
            form->status = "shared " + node.needs_artifact + " to " +
                           (node.hostname.empty() ? node.node_id : node.hostname);
            form->status_ok = true;
        } else {
            form->status = err;
            form->status_ok = false;
        }
    });

    auto vllm_container = Container::Vertical({
        vllm_install, vllm_version,
        max_model_len, max_num_seqs, max_num_batched_tokens, tp, pp, gpu_util,
        dtype, quantization, tool_parser, extra_args,
        trust_remote_code, prefix_caching, auto_tool_choice, sleep_mode,
        experimental_gloo,
    });
    auto form_container = Container::Vertical({
        primary_toggle,
        backup_toggle,
        share_checkbox,
        Maybe(vllm_container, &form->show_vllm),
        Container::Horizontal({save_button, close_button}),
    });

    auto edit_button = Button("Configure (e)", [form] { form->open = true; });
    auto action_row = Container::Horizontal({edit_button, resync_button, share_button});

    auto container = Container::Vertical({action_row, Maybe(form_container, &form->open)});

    auto renderer = Renderer(
        container,
        [&dashboard,
         &selected_node,
         &force_setup,
         form,
         sync_choices,
         action_row,
         primary_toggle,
         backup_toggle,
         share_checkbox,
         max_model_len,
         max_num_seqs,
         max_num_batched_tokens,
         tp,
         pp,
         gpu_util,
         dtype,
         quantization,
         tool_parser,
         extra_args,
         trust_remote_code,
         prefix_caching,
         auto_tool_choice,
         sleep_mode,
         experimental_gloo,
         vllm_install,
         vllm_version,
         save_button,
         close_button]() -> Element {
            const auto snap = dashboard.snapshot();
            sync_choices(snap);
            // The caller owns "must configure"; the tab obeys it. Two owners of one
            // modal is how the banner and the form end up disagreeing about whether
            // the cluster is configured.
            if (force_setup && !snap.configured) form->open = true;

            Elements body{
                hbox({render_engine_config(snap) | size(WIDTH, EQUAL, 42),
                      render_engine_activity(snap) | flex}),
                render_conformance_table(snap, selected_node) | flex,
                hbox({text(" "), action_row->Render()}),
            };
            if (!form->status.empty())
                body.push_back(text(" " + form->status) |
                               color(form->status_ok ? Color::Green : Color::Red));

            if (form->open) {
                Elements fields{
                    text(" Cluster engine configuration") | bold,
                    separator(),
                    hbox({text(" primary  ") | dim, primary_toggle->Render()}),
                    hbox({text(" backup   ") | dim, backup_toggle->Render()}),
                    hbox({text(" "), share_checkbox->Render()}),
                    text(""),
                    // Stated in the form, because it is the decision operators get
                    // wrong: "(none)" is a supported configuration, not a blank.
                    text(" \"(none)\" means no backup engine — nodes will not build one.") | dim,
                    text(" Accelerator and build paths are resolved per node and are not") | dim,
                    text(" settable here; a cluster cannot state hardware it cannot see.") | dim,
                };
                if (form->show_vllm) {
                    fields.push_back(separator());
                    fields.push_back(text(" vLLM launch defaults") | bold);
                    fields.push_back(hbox({text(" install method          ") | dim,
                                           vllm_install->Render()}));
                    fields.push_back(hbox({text(" version                 ") | dim,
                                           vllm_version->Render()}));
                    fields.push_back(hbox({text(" max model length       ") | dim,
                                           max_model_len->Render()}));
                    fields.push_back(hbox({text(" max sequences          ") | dim,
                                           max_num_seqs->Render()}));
                    fields.push_back(hbox({text(" max batched tokens     ") | dim,
                                           max_num_batched_tokens->Render()}));
                    fields.push_back(hbox({text(" tensor parallel / GPUs ") | dim,
                                           tp->Render()}));
                    fields.push_back(hbox({text(" pipeline / Ray nodes   ") | dim,
                                           pp->Render()}));
                    fields.push_back(hbox({text(" GPU memory utilization ") | dim,
                                           gpu_util->Render()}));
                    fields.push_back(hbox({text(" dtype                  ") | dim,
                                           dtype->Render()}));
                    fields.push_back(hbox({text(" quantization           ") | dim,
                                           quantization->Render()}));
                    fields.push_back(hbox({text(" tool-call parser       ") | dim,
                                           tool_parser->Render()}));
                    fields.push_back(hbox({text(" extra args             ") | dim,
                                           extra_args->Render()}));
                    fields.push_back(hbox({text(" "), trust_remote_code->Render()}));
                    fields.push_back(hbox({text(" "), prefix_caching->Render()}));
                    fields.push_back(hbox({text(" "), auto_tool_choice->Render()}));
                    fields.push_back(hbox({text(" "), sleep_mode->Render()}));
                    fields.push_back(separator());
                    fields.push_back(text(" Ray: automatic when pipeline parallelism > 1") |
                                     color(Color::Cyan));
                    const std::string ray = snap.ray_state.empty()
                        ? "status unavailable" : snap.ray_state + " — " + snap.ray_detail;
                    fields.push_back(text(" " + ray) | dim);
                    fields.push_back(hbox({text(" "), experimental_gloo->Render()}));
                }
                fields.push_back(separator());
                fields.push_back(hbox({save_button->Render(), text("  "), close_button->Render()}));
                body.push_back(vbox(fields) | border);
            }
            return vbox(body);
        });

    return CatchEvent(renderer, [form, &dashboard, &selected_node](const Event& ev) {
        if (form->open) return false;
        if (ev == Event::Character('e')) {
            form->open = true;
            return true;
        }
        // Move the conformance selection. Without this the highlight was
        // rendered and never movable, so "the selected node" was always the
        // first one — which a share button would have quietly inherited.
        //
        // j/k rather than the arrows: arrows belong to whichever button has
        // focus in the action row, and stealing them would make the row
        // unnavigable.
        const auto count = static_cast<int>(dashboard.snapshot().nodes.size());
        if (count > 0) {
            if (ev == Event::Character('j')) {
                selected_node = std::min(count - 1, selected_node + 1);
                return true;
            }
            if (ev == Event::Character('k')) {
                selected_node = std::max(0, selected_node - 1);
                return true;
            }
        }
        return false;
    });
}

} // namespace mm
