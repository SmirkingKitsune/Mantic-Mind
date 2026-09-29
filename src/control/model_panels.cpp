// Mantic-Mind — the Models tab. See include/control/model_panels.hpp for the rule
// this file exists under: everything here arrives over /v1/*.

#include "control/model_panels.hpp"

#include "common/http_client.hpp"
#include "common/util.hpp"

#include <ftxui/component/event.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

namespace mm {

using namespace ftxui;

namespace {

std::string truncate(const std::string& s, std::size_t n) {
    if (s.size() <= n) return s;
    if (n <= 1) return s.substr(0, n);
    return s.substr(0, n - 1) + "…";
}

Color status_color(const ServedModelView& m) {
    if (!m.available) return Color::Red;
    if (m.status == "active") return Color::Green;
    if (m.status == "idle") return Color::Cyan;
    if (m.status == "suspended") return Color::Yellow;
    return Color::GrayLight;
}

ServedModelView parse_view(const nlohmann::json& m) {
    ServedModelView v;
    v.raw = m;
    v.id = m.value("id", std::string{});
    v.display_name = m.value("display_name", std::string{});
    v.model_path = m.value("model_path", std::string{});
    v.source = m.value("source", std::string{});
    v.inference_backend = m.value("inference_backend", std::string{});
    v.backend_override = m.value("backend_override", std::string{});
    v.enabled = m.value("enabled", true);
    v.available = m.value("available", true);
    v.unavailable_reason = m.value("unavailable_reason", std::string{});
    v.backend = m.value("backend", std::string{});
    v.backend_reason = m.value("backend_reason", std::string{});
    v.status = m.value("status", std::string{});
    v.tool_calling = m.value("tool_calling", false);
    if (m.contains("placement") && m["placement"].is_object()) {
        const auto& p = m["placement"];
        v.node_id = p.value("node_id", std::string{});
        v.slot_id = p.value("slot_id", std::string{});
        v.in_flight = p.value("in_flight", 0);
    }
    if (m.contains("runtime_settings") && m["runtime_settings"].is_object())
        v.ctx_size = m["runtime_settings"].value("ctx_size", 0);
    if (m.contains("vision_settings") && m["vision_settings"].is_object())
        v.vision = m["vision_settings"].value("enabled", false);
    return v;
}

} // namespace

std::string describe_error_body(int status, const std::string& body) {
    std::string out = "HTTP " + std::to_string(status);
    try {
        const auto j = nlohmann::json::parse(body);
        if (j.contains("error") && j["error"].is_string()) out += ": " + j["error"].get<std::string>();
        if (j.contains("issues") && j["issues"].is_array()) {
            for (const auto& issue : j["issues"]) {
                if (issue.value("severity", std::string{}) != "error") continue;
                out += "; " + issue.value("field", std::string{}) + ": " +
                       issue.value("message", std::string{});
            }
        }
    } catch (const std::exception&) {
        if (!body.empty()) out += ": " + truncate(body, 160);
    }
    return out;
}

// ── ModelDashboard ────────────────────────────────────────────────────────────

struct ModelDashboard::Impl {
    std::string base_url;
    std::string api_token;

    mutable std::mutex mutex;
    ServedModelSnapshot snap;

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
        ServedModelSnapshot next;
        auto c = client();
        const auto r = c.get("/v1/served-models");
        if (!r.ok()) {
            // "Unreachable" and "no models" render identically as an empty
            // list, and they are different findings.
            next.error = "GET /v1/served-models → HTTP " + std::to_string(r.status);
        } else {
            next.reachable = true;
            try {
                const auto j = nlohmann::json::parse(r.body);
                for (const auto& m : j.value("data", nlohmann::json::array()))
                    next.models.push_back(parse_view(m));
            } catch (const std::exception& e) {
                next.error = std::string("served-models parse: ") + e.what();
            }
        }
        std::lock_guard<std::mutex> g(mutex);
        snap = std::move(next);
    }

    bool call(const HttpResponse& r, std::string& out_error) {
        if (r.ok()) {
            out_error.clear();
            wake_now.store(true);
            wake.notify_all();
            return true;
        }
        out_error = describe_error_body(r.status, r.body);
        return false;
    }
};

ModelDashboard::ModelDashboard(std::string base_url, std::string api_token)
    : impl_(std::make_unique<Impl>()) {
    impl_->base_url = std::move(base_url);
    impl_->api_token = std::move(api_token);
}

ModelDashboard::~ModelDashboard() {
    stop();
}

void ModelDashboard::start(int interval_ms) {
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

void ModelDashboard::stop() {
    if (!impl_->running.exchange(false)) return;
    impl_->wake.notify_all();
    if (impl_->worker.joinable()) impl_->worker.join();
}

void ModelDashboard::refresh_now() {
    impl_->wake_now.store(true);
    impl_->wake.notify_all();
}

ServedModelSnapshot ModelDashboard::snapshot() const {
    std::lock_guard<std::mutex> g(impl_->mutex);
    return impl_->snap;
}

bool ModelDashboard::create(const nlohmann::json& model, std::string& out_error) {
    return impl_->call(impl_->client().post("/v1/served-models", model), out_error);
}

bool ModelDashboard::update(const std::string& id, const nlohmann::json& model,
                            std::string& out_error) {
    return impl_->call(impl_->client().put("/v1/served-models/" + id, model), out_error);
}

bool ModelDashboard::remove(const std::string& id, std::string& out_error) {
    return impl_->call(impl_->client().del("/v1/served-models/" + id), out_error);
}

bool ModelDashboard::set_backend(const std::string& id, const std::string& override_value,
                                 std::string& out_error) {
    return impl_->call(impl_->client().put("/v1/served-models/" + id + "/backend",
                                           nlohmann::json{{"backend_override", override_value}}),
                       out_error);
}

bool ModelDashboard::lifecycle(const std::string& id, const std::string& verb, bool force,
                               std::string& out_error) {
    if (verb != "suspend" && verb != "restore" && verb != "release") {
        out_error = "unknown action " + verb;
        return false;
    }
    auto c = impl_->client();
    // Restore can transfer a model to a node: minutes, not milliseconds.
    c.set_timeouts(10, 3600, 60);
    const std::string suffix = verb == "release" && force ? "?force=true" : "";
    return impl_->call(c.post("/v1/served-models/" + id + "/" + verb + suffix,
                              nlohmann::json::object()),
                       out_error);
}

// ── Renderers ─────────────────────────────────────────────────────────────────

Element render_served_model_list(const ServedModelSnapshot& snap, int selected) {
    if (!snap.reachable) {
        return vbox({text(" Served models ") | bold, separator(),
                     text(" " + (snap.error.empty() ? std::string("not reachable") : snap.error)) |
                         color(Color::Red)}) |
               border;
    }
    Elements rows;
    rows.push_back(hbox({text(" ID") | size(WIDTH, EQUAL, 34),
                         text("SOURCE") | size(WIDTH, EQUAL, 10),
                         text("ENGINE") | size(WIDTH, EQUAL, 11),
                         text("STATUS") | size(WIDTH, EQUAL, 11),
                         text("REQ") | size(WIDTH, EQUAL, 5)}) |
                   dim);
    if (snap.models.empty()) {
        rows.push_back(text(" No served models. Admit one (tab 0), or add one with n.") | dim);
    }
    for (int i = 0; i < static_cast<int>(snap.models.size()); ++i) {
        const auto& m = snap.models[static_cast<std::size_t>(i)];
        const std::string engine = m.inference_backend == "api" ? "api"
                                   : m.backend.empty()          ? "-"
                                                                : m.backend;
        const std::string status = m.available ? (m.status.empty() ? "-" : m.status)
                                               : std::string("unavailable");
        Element row = hbox({
            text(" " + truncate(m.id, 32)) | size(WIDTH, EQUAL, 34),
            text(m.source) | color(m.source == "admitted" ? Color::Magenta : Color::GrayLight) |
                size(WIDTH, EQUAL, 10),
            text(engine) | size(WIDTH, EQUAL, 11),
            text(status) | color(status_color(m)) | size(WIDTH, EQUAL, 11),
            text(m.in_flight > 0 ? std::to_string(m.in_flight) : "-") | size(WIDTH, EQUAL, 5),
        });
        if (i == selected) row = std::move(row) | inverted;
        rows.push_back(std::move(row));
    }
    return vbox({text(" Served models ") | bold, separator(), vbox(std::move(rows)) | yframe}) |
           border;
}

Element render_served_model_detail(const ServedModelSnapshot& snap, int selected) {
    if (selected < 0 || selected >= static_cast<int>(snap.models.size())) {
        return vbox({text(" Detail ") | bold, separator(), text(" select a model") | dim}) |
               border;
    }
    const auto& m = snap.models[static_cast<std::size_t>(selected)];
    const auto field = [](const std::string& label, Element value) {
        return hbox({text(" " + label) | dim | size(WIDTH, EQUAL, 16), std::move(value)});
    };
    Elements rows{
        field("name", text(m.display_name.empty() ? m.id : m.display_name) | bold),
        field("model", text(truncate(m.model_path, 60))),
        field("source", text(m.source)),
        field("backend", text(m.inference_backend + " / override " + m.backend_override)),
        field("routes to", text(m.backend.empty() ? "-" : m.backend)),
        field("because", paragraph(m.backend_reason.empty() ? "-" : m.backend_reason)),
        field("context", text(m.ctx_size > 0 ? std::to_string(m.ctx_size) + " tokens" : "-")),
        field("tools", text(m.tool_calling ? "yes" : "no")),
        field("vision", text(m.vision ? "yes" : "no")),
        field("enabled", text(m.enabled ? "yes" : "no") |
                             color(m.enabled ? Color::Green : Color::Yellow)),
    };
    if (!m.available) {
        rows.push_back(field("unavailable", paragraph(m.unavailable_reason) | color(Color::Red)));
    }
    if (!m.slot_id.empty()) {
        rows.push_back(field("placed on", text(truncate(m.node_id, 12) + " / " + m.slot_id)));
        rows.push_back(field("in flight", text(std::to_string(m.in_flight))));
    }
    if (m.source == "admitted") {
        rows.push_back(text(""));
        rows.push_back(paragraph(" Listed because it is admitted. Editing it saves an override "
                                 "under the same id; it cannot be deleted, only disabled.") |
                       dim);
    }
    return vbox({text(" Detail ") | bold, separator(), vbox(std::move(rows))}) | border;
}

// ── The tab ───────────────────────────────────────────────────────────────────

namespace {

/// One editable field of a served model, addressed by its JSON path.
struct FormField {
    enum class Kind { Text, Int, Float, Bool, List, Secret };
    std::string path;  ///< "runtime_settings.ctx_size"
    std::string label;
    Kind kind = Kind::Text;
    std::string value;
};

std::vector<FormField> make_fields() {
    using K = FormField::Kind;
    return {
        {"id", "id", K::Text, {}},
        {"display_name", "display name", K::Text, {}},
        {"model_path", "model path", K::Text, {}},
        {"inference_backend", "backend (node|api)", K::Text, "node"},
        {"backend_override", "engine (auto|soma|fallback)", K::Text, "auto"},
        {"engine_model_name", "engine model name", K::Text, {}},
        {"preferred_node_id", "preferred node", K::Text, {}},
        {"enabled", "enabled", K::Bool, "true"},
        {"tool_calling", "tool calling", K::Bool, "true"},
        {"runtime_settings.ctx_size", "ctx_size", K::Int, "4096"},
        {"runtime_settings.n_gpu_layers", "n_gpu_layers", K::Int, "-1"},
        {"runtime_settings.n_threads", "n_threads", K::Int, "-1"},
        {"runtime_settings.parallel", "parallel", K::Int, "1"},
        {"runtime_settings.batch_size", "batch_size", K::Int, "-1"},
        {"runtime_settings.ubatch_size", "ubatch_size", K::Int, "-1"},
        {"runtime_settings.flash_attn", "flash_attn", K::Bool, "true"},
        {"runtime_settings.extra_args", "extra args", K::List, {}},
        {"runtime_settings.temperature", "temperature", K::Float, "0.7"},
        {"runtime_settings.top_p", "top_p", K::Float, "0.9"},
        {"runtime_settings.top_k", "top_k", K::Int, "-1"},
        {"runtime_settings.min_p", "min_p", K::Float, "-1"},
        {"runtime_settings.presence_penalty", "presence_penalty", K::Float, "0"},
        {"runtime_settings.repeat_penalty", "repeat_penalty", K::Float, "-1"},
        {"runtime_settings.max_tokens", "max_tokens", K::Int, "1024"},
        {"vision_settings.enabled", "vision", K::Bool, "false"},
        {"vision_settings.mmproj_path", "mmproj path", K::Text, {}},
        {"api_settings.base_url", "api base url", K::Text, "https://api.openai.com"},
        {"api_settings.chat_completions_path", "api path", K::Text, "/v1/chat/completions"},
        {"api_settings.api_key_env", "api key env", K::Text, "OPENAI_API_KEY"},
        {"api_settings.api_key", "api key (blank=keep)", K::Secret, {}},
    };
}

const nlohmann::json* at_path(const nlohmann::json& j, const std::string& path) {
    const nlohmann::json* cur = &j;
    std::stringstream ss(path);
    std::string part;
    while (std::getline(ss, part, '.')) {
        if (!cur->is_object() || !cur->contains(part)) return nullptr;
        cur = &(*cur)[part];
    }
    return cur;
}

nlohmann::json& at_path_mut(nlohmann::json& j, const std::string& path) {
    nlohmann::json* cur = &j;
    std::stringstream ss(path);
    std::string part;
    while (std::getline(ss, part, '.')) {
        if (!cur->is_object()) *cur = nlohmann::json::object();
        cur = &(*cur)[part];
    }
    return *cur;
}

std::string field_text(const nlohmann::json& v, FormField::Kind kind) {
    using K = FormField::Kind;
    if (kind == K::List) {
        std::string out;
        if (v.is_array())
            for (const auto& item : v)
                if (item.is_string()) out += (out.empty() ? "" : " ") + item.get<std::string>();
        return out;
    }
    if (v.is_string()) return v.get<std::string>();
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_number_integer()) return std::to_string(v.get<long long>());
    if (v.is_number()) {
        std::ostringstream os;
        os << v.get<double>();
        return os.str();
    }
    return {};
}

/// Fields -> the JSON document, on top of `base` so fields the form does not
/// name are carried rather than reset. False with `error` naming the field.
bool build_document(const std::vector<FormField>& fields, nlohmann::json base,
                    nlohmann::json& out, std::string& error) {
    using K = FormField::Kind;
    // Server-computed keys are not part of the resource.
    for (const char* k : {"source", "available", "unavailable_reason", "admission_id", "arch_hash",
                          "backend", "backend_reason", "placement", "status", "issues",
                          "created_at_ms", "updated_at_ms", "node_compatibility",
                          "placement_history"})
        base.erase(k);
    if (base.contains("api_settings") && base["api_settings"].is_object())
        base["api_settings"].erase("api_key_configured");
    for (const auto& f : fields) {
        const std::string v = util::trim(f.value);
        auto& slot = at_path_mut(base, f.path);
        try {
            switch (f.kind) {
            case K::Text:
                slot = v;
                break;
            case K::Secret:
                if (v.empty()) {
                    // Blank means "keep": the catalog keeps a key an update
                    // does not carry, and a displayed key would be a leak.
                    nlohmann::json& parent = at_path_mut(base, "api_settings");
                    parent.erase("api_key");
                } else {
                    slot = v;
                }
                break;
            case K::Int:
                slot = std::stoi(v);
                break;
            case K::Float:
                slot = std::stof(v);
                break;
            case K::Bool: {
                const std::string b = util::to_lower(v);
                if (b == "true" || b == "yes" || b == "1" || b == "on") slot = true;
                else if (b == "false" || b == "no" || b == "0" || b == "off") slot = false;
                else {
                    error = f.label + ": expected true or false";
                    return false;
                }
                break;
            }
            case K::List: {
                nlohmann::json list = nlohmann::json::array();
                std::stringstream ss(v);
                std::string item;
                while (ss >> item) list.push_back(item);
                slot = list;
                break;
            }
            }
        } catch (const std::exception&) {
            error = f.label + ": not a number";
            return false;
        }
    }
    out = std::move(base);
    return true;
}

} // namespace

Component model_tab(ModelDashboard& dashboard, int& selected) {
    struct State {
        std::vector<FormField> fields = make_fields();
        bool form_open = false;
        bool editing = false; ///< PUT to edit_id rather than POST
        std::string edit_id;
        nlohmann::json edit_base = nlohmann::json::object();
        std::string pending_delete;

        std::mutex status_mutex;
        std::string status;
        bool status_ok = true;

        std::atomic<bool> busy{false};
        std::thread worker;

        void set_status(std::string s, bool ok) {
            std::lock_guard<std::mutex> lk(status_mutex);
            status = std::move(s);
            status_ok = ok;
        }
        std::pair<std::string, bool> get_status() {
            std::lock_guard<std::mutex> lk(status_mutex);
            return {status, status_ok};
        }
        /// Actions can place a model — a transfer of gigabytes — so they run
        /// off the UI thread. One at a time: a second press while one runs is
        /// ignored rather than queued behind it.
        void run(std::function<void()> fn) {
            if (busy.exchange(true)) return;
            if (worker.joinable()) worker.join();
            worker = std::thread([this, fn = std::move(fn)] {
                fn();
                busy.store(false);
            });
        }
        ~State() {
            if (worker.joinable()) worker.join();
        }
    };
    auto st = std::make_shared<State>();

    const auto current = [&dashboard, &selected]() -> std::optional<ServedModelView> {
        const auto snap = dashboard.snapshot();
        if (selected < 0 || selected >= static_cast<int>(snap.models.size())) return std::nullopt;
        return snap.models[static_cast<std::size_t>(selected)];
    };

    const auto open_new = [st] {
        st->fields = make_fields();
        st->editing = false;
        st->edit_id.clear();
        st->edit_base = nlohmann::json::object();
        st->form_open = true;
    };
    const auto open_edit = [st, current] {
        const auto m = current();
        if (!m) {
            st->set_status("no model selected", false);
            return;
        }
        for (auto& f : st->fields) {
            if (const auto* v = at_path(m->raw, f.path)) f.value = field_text(*v, f.kind);
            if (f.kind == FormField::Kind::Secret) f.value.clear();
        }
        st->editing = true;
        st->edit_id = m->id;
        st->edit_base = m->raw;
        st->form_open = true;
    };

    // Inputs bind to the field strings, which never move: the vector is sized
    // once and only its VALUES change.
    Components inputs;
    for (auto& f : st->fields) {
        InputOption opt;
        opt.multiline = false;
        if (f.kind == FormField::Kind::Secret) opt.password = true;
        opt.placeholder = f.label;
        inputs.push_back(Input(&f.value, opt));
    }
    auto save_button = Button("Save", [st, &dashboard] {
        nlohmann::json doc;
        std::string error;
        if (!build_document(st->fields, st->edit_base, doc, error)) {
            st->set_status(error, false);
            return;
        }
        st->run([st, &dashboard, doc] {
            std::string err;
            const bool ok = st->editing ? dashboard.update(st->edit_id, doc, err)
                                        : dashboard.create(doc, err);
            if (ok) {
                st->set_status(st->editing ? "saved " + st->edit_id : "created", true);
                st->form_open = false;
            } else {
                st->set_status(err, false);
            }
        });
    });
    auto close_button = Button("Close", [st] { st->form_open = false; });
    Components form_rows = inputs;
    form_rows.push_back(Container::Horizontal({save_button, close_button}));
    auto form_container = Container::Vertical(form_rows);

    const auto act = [st, &dashboard, current](const std::string& verb, bool force) {
        const auto m = current();
        if (!m) {
            st->set_status("no model selected", false);
            return;
        }
        const std::string id = m->id;
        st->set_status(verb + " " + id + "…", true);
        st->run([st, &dashboard, id, verb, force] {
            std::string err;
            if (dashboard.lifecycle(id, verb, force, err)) st->set_status(verb + " " + id + ": done", true);
            else st->set_status(err, false);
        });
    };
    const auto cycle_backend = [st, &dashboard, current] {
        const auto m = current();
        if (!m) {
            st->set_status("no model selected", false);
            return;
        }
        const std::string next = m->backend_override == "auto"   ? "soma"
                                 : m->backend_override == "soma" ? "fallback"
                                                                 : "auto";
        std::string err;
        if (dashboard.set_backend(m->id, next, err))
            st->set_status(m->id + ": engine override " + next, true);
        else
            st->set_status(err, false);
    };
    const auto delete_selected = [st, &dashboard, current] {
        const auto m = current();
        if (!m) {
            st->set_status("no model selected", false);
            return;
        }
        // Two presses: deleting a model changes what every user can pick.
        if (st->pending_delete != m->id) {
            st->pending_delete = m->id;
            st->set_status("press x again to delete " + m->id, false);
            return;
        }
        st->pending_delete.clear();
        std::string err;
        if (dashboard.remove(m->id, err)) st->set_status("deleted " + m->id, true);
        else st->set_status(err, false);
    };

    auto new_button = Button("New (n)", open_new);
    auto edit_button = Button("Edit (e)", open_edit);
    auto delete_button = Button("Delete (x)", delete_selected);
    auto backend_button = Button("Engine (b)", cycle_backend);
    auto suspend_button = Button("Suspend (s)", [act] { act("suspend", false); });
    auto restore_button = Button("Restore (r)", [act] { act("restore", false); });
    auto release_button = Button("Release (l)", [act] { act("release", false); });
    auto action_row = Container::Horizontal({new_button, edit_button, delete_button,
                                             backend_button, suspend_button, restore_button,
                                             release_button});
    auto container = Container::Vertical({action_row, Maybe(form_container, &st->form_open)});

    auto renderer = Renderer(container, [&dashboard, &selected, st, action_row, inputs,
                                         save_button, close_button]() -> Element {
        const auto snap = dashboard.snapshot();
        Elements body{
            hbox({render_served_model_list(snap, selected) | flex,
                  render_served_model_detail(snap, selected) | size(WIDTH, GREATER_THAN, 50)}) |
                flex,
            hbox({text(" "), action_row->Render()}),
        };
        const auto [status, ok] = st->get_status();
        if (!status.empty() || st->busy.load()) {
            body.push_back(text(" " + status + (st->busy.load() ? "  (working…)" : "")) |
                           color(ok ? Color::Green : Color::Red));
        }
        if (st->form_open) {
            Elements rows{text(st->editing ? " Edit " + st->edit_id : " New served model") | bold,
                          separator()};
            for (std::size_t i = 0; i < st->fields.size(); ++i) {
                rows.push_back(hbox({text(" " + st->fields[i].label) | dim |
                                         size(WIDTH, EQUAL, 30),
                                     inputs[i]->Render() | flex}));
            }
            rows.push_back(separator());
            rows.push_back(text(" Admitted models need no entry: they are listed already. Add GGUF "
                                "(node) and remote (api) models here.") |
                           dim);
            rows.push_back(hbox({save_button->Render(), text("  "), close_button->Render()}));
            body.push_back(vbox(std::move(rows)) | yframe | border | size(HEIGHT, LESS_THAN, 40));
        }
        return vbox(std::move(body));
    });

    return CatchEvent(renderer, [st, &dashboard, &selected, open_new, open_edit, act,
                                 cycle_backend, delete_selected](const Event& ev) {
        if (st->form_open) return false;
        if (ev == Event::Character('n')) { open_new(); return true; }
        if (ev == Event::Character('e')) { open_edit(); return true; }
        if (ev == Event::Character('x')) { delete_selected(); return true; }
        if (ev == Event::Character('b')) { cycle_backend(); return true; }
        if (ev == Event::Character('s')) { act("suspend", false); return true; }
        if (ev == Event::Character('r')) { act("restore", false); return true; }
        if (ev == Event::Character('l')) { act("release", false); return true; }
        const auto count = static_cast<int>(dashboard.snapshot().models.size());
        if (count > 0) {
            if (ev == Event::Character('j') || ev == Event::ArrowDown) {
                selected = std::min(count - 1, selected + 1);
                st->pending_delete.clear();
                return true;
            }
            if (ev == Event::Character('k') || ev == Event::ArrowUp) {
                selected = std::max(0, selected - 1);
                st->pending_delete.clear();
                return true;
            }
        }
        return false;
    });
}

} // namespace mm
