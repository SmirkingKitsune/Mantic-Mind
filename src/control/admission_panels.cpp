// Mantic-Mind — the Admissions tab. See include/control/admission_panels.hpp for
// the rule this file exists under: everything here arrives over /v1/*.

#include "control/admission_panels.hpp"

#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <ftxui/component/event.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>

namespace mm {

using namespace ftxui;

namespace {

/// The stage ladder for a run of `total_steps` stages, or empty when unknown.
///
/// The wire carries `stage`, `step` and `total_steps` — never the ladder — so a
/// client that wants to draw the stages not yet reached has to know the shapes.
/// There are exactly three, and the LENGTH identifies them:
///
///   3  an already-converted container (what reprofile runs)
///   7  a local source: no fetch
///   8  a repo id: fetch first
///
/// Keyed on the count rather than assumed, because a single flat list indexed by
/// position is wrong the moment `fetch` is absent — every label shifts by one
/// and the last stage is never drawn. That was this function's first version,
/// and `mm_admission_panels` caught it.
///
/// An unrecognized count returns EMPTY, and the renderer then draws positions
/// without names. A server that grows a stage should make this client say "step
/// 3 of 9" rather than confidently mislabel nine stages: position comes from the
/// wire and stays right, names are the part this can be wrong about, and
/// inventing them is worse than omitting them.
const std::vector<std::string>& ladder_for(int total_steps) {
    static const std::vector<std::string> container = {"profile", "conformance", "finalize"};
    static const std::vector<std::string> local = {
        "convert", "tokenize", "oracle", "reference", "profile", "conformance", "finalize"};
    static const std::vector<std::string> fetched = {"fetch",
                                                     "convert",
                                                     "tokenize",
                                                     "oracle",
                                                     "reference",
                                                     "profile",
                                                     "conformance",
                                                     "finalize"};
    static const std::vector<std::string> unknown;
    if (total_steps == static_cast<int>(container.size())) return container;
    if (total_steps == static_cast<int>(local.size())) return local;
    if (total_steps == static_cast<int>(fetched.size())) return fetched;
    return unknown;
}

Color stage_color(const AdmissionView& op) {
    if (!op.error.empty()) return Color::Red;
    if (op.canceled) return Color::Yellow;
    if (op.done) return Color::Green;
    return Color::Cyan;
}

std::string state_word(const AdmissionView& op) {
    if (!op.error.empty()) return "failed";
    if (op.canceled) return "canceled";
    if (op.done) return "done";
    // The queued state the protocol always had and never used until the
    // concurrency cap landed (roadmap D48). Worth showing distinctly: "queued"
    // and "converting" are different answers to "why has nothing happened".
    if (op.detail == "queued") return "queued";
    return "running";
}

std::string truncate(const std::string& s, std::size_t n) {
    if (s.size() <= n) return s;
    if (n <= 1) return s.substr(0, n);
    return s.substr(0, n - 1) + "…";
}

std::string human_bytes(std::int64_t n) {
    if (n <= 0) return "-";
    const char* unit[] = {"B", "KB", "MB", "GB", "TB"};
    double v = static_cast<double>(n);
    int i = 0;
    while (v >= 1024.0 && i < 4) {
        v /= 1024.0;
        ++i;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f %s", v, unit[i]);
    return buf;
}

std::string elapsed(const AdmissionView& op) {
    const auto end = op.finished_at_ms > 0 ? op.finished_at_ms : util::now_ms();
    if (op.started_at_ms <= 0 || end < op.started_at_ms) return "-";
    const auto s = (end - op.started_at_ms) / 1000;
    char buf[32];
    std::snprintf(buf,
                  sizeof(buf),
                  "%02d:%02d:%02d",
                  static_cast<int>(s / 3600),
                  static_cast<int>((s / 60) % 60),
                  static_cast<int>(s % 60));
    return buf;
}

} // namespace

// ── AdmissionDashboard ────────────────────────────────────────────────────────

struct AdmissionDashboard::Impl {
    std::string base_url;
    std::string api_token;

    mutable std::mutex mutex;
    AdmissionSnapshot snap;

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
        AdmissionSnapshot next;
        auto c = client();

        const auto ops = c.get("/v1/models/admissions");
        if (!ops.ok()) {
            // Reported rather than swallowed: an empty list because control is
            // unreachable and an empty list because nothing is admitting are
            // completely different findings, and they render identically.
            next.reachable = false;
            next.error = "GET /v1/models/admissions → HTTP " + std::to_string(ops.status);
            std::lock_guard<std::mutex> g(mutex);
            snap = std::move(next);
            return;
        }
        next.reachable = true;
        try {
            const auto j = nlohmann::json::parse(ops.body);
            for (const auto& o : j.value("data", nlohmann::json::array())) {
                AdmissionView v;
                v.operation_id = o.value("operation_id", std::string{});
                v.source = o.value("source", std::string{});
                v.stage = o.value("stage", std::string{});
                v.detail = o.value("detail", std::string{});
                v.step = o.value("step", 0);
                v.total_steps = o.value("total_steps", 0);
                v.fraction = o.value("fraction", 0.0);
                v.bytes_done = o.value("bytes_done", std::int64_t{0});
                v.bytes_total = o.value("bytes_total", std::int64_t{0});
                v.cancelable = o.value("cancelable", true);
                v.done = o.value("done", false);
                v.canceled = o.value("canceled", false);
                v.error = o.value("error", std::string{});
                v.model_id = o.value("model_id", std::int64_t{0});
                v.started_at_ms = o.value("started_at_ms", std::int64_t{0});
                v.finished_at_ms = o.value("finished_at_ms", std::int64_t{0});
                next.operations.push_back(std::move(v));
            }
        } catch (const std::exception& e) {
            next.error = std::string("admissions parse: ") + e.what();
        }

        // The registry is the OUTCOME half. A failure to read it is not a
        // failure of the tab — the operations list is the subject here — so it
        // degrades to an empty model list rather than blanking the page.
        const auto models = c.get("/v1/models");
        if (models.ok()) {
            try {
                const auto j = nlohmann::json::parse(models.body);
                for (const auto& m : j.value("data", nlohmann::json::array())) {
                    AdmittedView v;
                    v.id = m.value("id", std::int64_t{0});
                    v.name = m.value("name", std::string{});
                    v.verdict = m.value("verdict", std::string{});
                    v.attention_family = m.value("attention_family", std::string{});
                    v.arch_hash = m.value("arch_hash", std::string{});
                    v.bytes_per_token = m.value("bytes_per_token", std::int64_t{0});
                    next.models.push_back(std::move(v));
                }
            } catch (const std::exception&) {
            }
        }

        std::lock_guard<std::mutex> g(mutex);
        snap = std::move(next);
    }
};

AdmissionDashboard::AdmissionDashboard(std::string base_url, std::string api_token)
    : impl_(std::make_unique<Impl>()) {
    impl_->base_url = std::move(base_url);
    impl_->api_token = std::move(api_token);
}

AdmissionDashboard::~AdmissionDashboard() {
    stop();
}

void AdmissionDashboard::start(int interval_ms) {
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

void AdmissionDashboard::stop() {
    if (!impl_->running.exchange(false)) return;
    impl_->wake.notify_all();
    if (impl_->worker.joinable()) impl_->worker.join();
}

void AdmissionDashboard::refresh_now() {
    impl_->wake_now.store(true);
    impl_->wake.notify_all();
}

AdmissionSnapshot AdmissionDashboard::snapshot() const {
    std::lock_guard<std::mutex> g(impl_->mutex);
    return impl_->snap;
}

std::string AdmissionDashboard::admit(const std::string& source,
                                      const std::string& expert_gate,
                                      const std::string& expert_down,
                                      int group,
                                      std::string& out_error) {
    out_error.clear();
    if (util::trim(source).empty()) {
        out_error = "a source is required (a HuggingFace repo id or a local path)";
        return {};
    }

    nlohmann::json body{{"source", util::trim(source)}};
    // Omitted rather than sent blank: the server reads a present-but-empty field
    // as an explicit choice of nothing, where the operator meant "whatever the
    // deployment default is".
    nlohmann::json quant = nlohmann::json::object();
    if (!expert_gate.empty()) quant["expert_gate"] = expert_gate;
    if (!expert_down.empty()) quant["expert_down"] = expert_down;
    if (group > 0) quant["group"] = group;
    if (!quant.empty()) body["quantization"] = quant;

    // `POST /v1/models/admit` answers ONLY as a stream, so the id is learned
    // from the first frame and the stream is then dropped. Safe by design:
    // control logs "client disconnected; conversion continues" and the worker is
    // detached precisely so it outlives the request. Everything after this is
    // the poller's job.
    auto c = impl_->client();
    c.set_timeouts(10, 120, 30);

    std::string operation_id;
    int status = 0;
    std::string error_body;
    const bool connected = c.stream_post(
        "/v1/models/admit",
        body,
        [&operation_id](const std::string& line) {
            const auto pos = line.find("data:");
            if (pos == std::string::npos) return true;
            try {
                const auto j = nlohmann::json::parse(util::trim(line.substr(pos + 5)));
                const auto id = j.value("operation_id", std::string{});
                if (!id.empty()) {
                    operation_id = id;
                    return false; // got what we came for
                }
            } catch (...) {
                // A heartbeat or a partial frame; keep reading.
            }
            return true;
        },
        &status,
        &error_body);

    if (!operation_id.empty()) {
        refresh_now();
        return operation_id;
    }

    // A bad source is a 400 on the REQUEST — nothing started, so there is
    // nothing to stream about — and that body carries the reason.
    if (!error_body.empty()) {
        try {
            const auto j = nlohmann::json::parse(error_body);
            if (j.contains("error") && j.at("error").is_string()) {
                out_error = j.at("error").get<std::string>();
                return {};
            }
        } catch (...) {
        }
    }
    out_error = connected ? "admission started but reported no operation id"
                          : "cannot reach control (HTTP " + std::to_string(status) + ")";
    return {};
}

std::string AdmissionDashboard::reprofile(std::int64_t model_id, std::string& out_error) {
    out_error.clear();
    auto c = impl_->client();
    c.set_timeouts(10, 120, 30);

    // Same shape as admit: the route answers only as a stream, so the id is
    // read off the first frame and the stream dropped. The operation is
    // detached server-side and shows up in the operations list.
    std::string operation_id;
    int status = 0;
    std::string error_body;
    const bool connected = c.stream_post(
        "/v1/models/" + std::to_string(model_id) + "/reprofile",
        nlohmann::json::object(),
        [&operation_id](const std::string& line) {
            const auto pos = line.find("data:");
            if (pos == std::string::npos) return true;
            try {
                const auto j = nlohmann::json::parse(util::trim(line.substr(pos + 5)));
                const auto id = j.value("operation_id", std::string{});
                if (!id.empty()) {
                    operation_id = id;
                    return false;
                }
            } catch (...) {
            }
            return true;
        },
        &status,
        &error_body);

    if (!operation_id.empty()) {
        refresh_now();
        return operation_id;
    }
    if (!error_body.empty()) {
        try {
            const auto j = nlohmann::json::parse(error_body);
            if (j.contains("error") && j.at("error").is_string()) {
                out_error = j.at("error").get<std::string>();
                return {};
            }
        } catch (...) {
        }
    }
    out_error = connected ? "reprofile started but reported no operation id"
                          : "cannot reach control (HTTP " + std::to_string(status) + ")";
    return {};
}

bool AdmissionDashboard::cancel(const std::string& operation_id, std::string& out_error) {
    out_error.clear();
    if (operation_id.empty()) {
        out_error = "no operation selected";
        return false;
    }
    const auto r = impl_->client().post("/v1/models/admissions/" + operation_id + "/cancel",
                                        nlohmann::json::object());
    if (!r.ok()) {
        try {
            const auto j = nlohmann::json::parse(r.body);
            if (j.contains("error") && j.at("error").is_string()) {
                out_error = j.at("error").get<std::string>();
                return false;
            }
        } catch (...) {
        }
        out_error = "HTTP " + std::to_string(r.status);
        return false;
    }
    refresh_now();
    return true;
}

// ── Renderers ─────────────────────────────────────────────────────────────────

Element render_admission_list(const AdmissionSnapshot& snap, int selected) {
    if (!snap.reachable)
        return vbox({text(" Admissions ") | bold,
                     separator(),
                     text(" control is not answering") | color(Color::Red),
                     text(" " + snap.error) | dim}) |
               border;
    if (snap.operations.empty())
        return vbox({text(" Admissions ") | bold,
                     separator(),
                     text(" no admissions yet — press a to admit a model") | dim}) |
               border;

    Elements rows;
    rows.push_back(hbox({text("  SOURCE                        ") | dim | bold,
                         text("STATE     ") | dim | bold,
                         text("STAGE        ") | dim | bold,
                         text("STEP   ") | dim | bold,
                         text("ELAPSED") | dim | bold}));

    for (std::size_t i = 0; i < snap.operations.size(); ++i) {
        const auto& op = snap.operations[i];
        const std::string src = truncate(op.source, 29);
        const std::string state = state_word(op);
        const std::string step =
            op.total_steps > 0 ? std::to_string(op.step) + "/" + std::to_string(op.total_steps)
                               : std::string{"-"};
        Element row = hbox({
            text("  " + src + std::string(30 - std::min<std::size_t>(29, src.size()), ' ')),
            text(state + std::string(10 - std::min<std::size_t>(9, state.size()), ' ')) |
                color(stage_color(op)),
            text(truncate(op.stage, 12) +
                 std::string(13 - std::min<std::size_t>(12, op.stage.size()), ' ')),
            text(step + std::string(7 - std::min<std::size_t>(6, step.size()), ' ')),
            text(elapsed(op)) | dim,
        });
        if (static_cast<int>(i) == selected) row = std::move(row) | inverted;
        rows.push_back(std::move(row));
        if (!op.error.empty())
            rows.push_back(text("      " + truncate(op.error, 90)) | color(Color::Red));
    }
    rows.push_back(text(""));
    rows.push_back(text("  j/k select · a admit · c cancel") | dim);
    return vbox({text(" Admissions ") | bold, separator(), vbox(rows)}) | border;
}

Element render_admission_detail(const AdmissionSnapshot& snap, int selected) {
    if (selected < 0 || selected >= static_cast<int>(snap.operations.size()))
        return vbox({text(" Detail ") | bold, separator(), text(" no operation selected") | dim}) |
               border;

    const auto& op = snap.operations[static_cast<std::size_t>(selected)];
    Elements rows;
    rows.push_back(hbox({text(" source   ") | dim, text(op.source) | bold}));
    rows.push_back(
        hbox({text(" state    ") | dim, text(state_word(op)) | bold | color(stage_color(op))}));
    rows.push_back(hbox({text(" detail   ") | dim, text(truncate(op.detail, 70))}));

    // The whole ladder, not just the current rung. "convert" alone does not tell
    // an operator whether to wait; "convert, 3 of 7" does.
    if (op.total_steps > 0) {
        Elements ladder;
        const auto& names = ladder_for(op.total_steps);
        for (int i = 1; i <= op.total_steps; ++i) {
            // The live `stage` field always names the current rung — that comes
            // off the wire and is right even for a ladder shape this build does
            // not know. The others are labelled only when the shape is
            // recognized; otherwise they are positions, which is the honest
            // thing to draw for a stage whose name this client cannot know.
            std::string label;
            if (i == op.step)
                label = op.stage;
            else if (static_cast<std::size_t>(i - 1) < names.size())
                label = names[static_cast<std::size_t>(i - 1)];
            else
                label = std::to_string(i);

            Element cell = text(" " + label + " ");
            if (i < op.step)
                cell = std::move(cell) | color(Color::Green);
            else if (i == op.step)
                cell = std::move(cell) | bold | inverted;
            else
                cell = std::move(cell) | dim;
            ladder.push_back(std::move(cell));
        }
        rows.push_back(hbox({text(" stages   ") | dim, hbox(ladder)}));
    }

    if (op.fraction >= 0.0 && !op.done)
        rows.push_back(hbox({text(" progress ") | dim,
                             gauge(static_cast<float>(op.fraction)) | flex | color(Color::Cyan)}));
    // Populated by fetch alone — the one stage whose remaining time is
    // estimable. Shown only when reported, so zero does not read as "nothing to
    // transfer".
    if (op.bytes_total > 0)
        rows.push_back(
            hbox({text(" bytes    ") | dim,
                  text(human_bytes(op.bytes_done) + " / " + human_bytes(op.bytes_total))}));
    rows.push_back(hbox({text(" elapsed  ") | dim, text(elapsed(op))}));
    if (op.model_id > 0)
        rows.push_back(hbox({text(" model    ") | dim,
                             text("#" + std::to_string(op.model_id)) | color(Color::Green)}));
    if (!op.error.empty()) rows.push_back(paragraph(" " + op.error) | color(Color::Red));
    if (op.running() && !op.cancelable)
        rows.push_back(text(" this stage cannot be interrupted") | dim);

    return vbox({text(" Detail ") | bold, separator(), vbox(rows)}) | border;
}

Element render_admitted_models(const AdmissionSnapshot& snap) {
    Elements rows;
    if (snap.models.empty()) {
        rows.push_back(text("  nothing admitted yet") | dim);
    } else {
        rows.push_back(hbox({text("  MODEL                     ") | dim | bold,
                             text("VERDICT        ") | dim | bold,
                             text("FAMILY   ") | dim | bold,
                             text("BYTES/TOKEN") | dim | bold}));
        for (const auto& m : snap.models) {
            const std::string name = truncate(m.name, 25);
            // The verdict is what decides whether Soma serves it at all, so it
            // gets the colour rather than the name.
            const Color vc = m.verdict == "stream"          ? Color::Green
                             : m.verdict == "hybrid"        ? Color::Cyan
                             : m.verdict == "resident_only" ? Color::Yellow
                                                            : Color::Red;
            rows.push_back(hbox({
                text("  " + name + std::string(26 - std::min<std::size_t>(25, name.size()), ' ')),
                text(m.verdict +
                     std::string(15 - std::min<std::size_t>(14, m.verdict.size()), ' ')) |
                    color(vc),
                text(truncate(m.attention_family, 8) +
                     std::string(9 - std::min<std::size_t>(8, m.attention_family.size()), ' ')),
                text(human_bytes(m.bytes_per_token)) | dim,
            }));
        }
    }
    return vbox({text(" Admitted models ") | bold, separator(), vbox(rows)}) | border;
}

// ── The tab ───────────────────────────────────────────────────────────────────

Component admission_tab(AdmissionDashboard& dashboard, int& selected) {
    struct FormState {
        bool open = false;
        std::string source;
        std::string expert_gate;
        std::string expert_down;
        std::string group;
        std::string status;
        bool status_ok = false;
        /// An admission start is an HTTP round trip that waits for the first
        /// progress frame. Held so the UI thread never blocks on it — a TUI that
        /// freezes while starting a job is how an operator concludes it hung and
        /// starts a second.
        std::atomic<bool> starting{false};
        std::thread worker;

        ~FormState() {
            if (worker.joinable()) worker.join();
        }
    };

    auto form = std::make_shared<FormState>();

    auto source_input = Input(&form->source, "HF repo id or local path");
    auto gate_input = Input(&form->expert_gate, "expert_gate (blank = default)");
    auto down_input = Input(&form->expert_down, "expert_down (blank = default)");
    auto group_input = Input(&form->group, "group (blank = default)");

    auto start_button = Button("Start", [form, &dashboard] {
        if (form->starting.load()) return;
        form->starting.store(true);
        form->status = "starting…";
        form->status_ok = true;
        if (form->worker.joinable()) form->worker.join();
        form->worker = std::thread([form, &dashboard] {
            int group = 0;
            try {
                if (!util::trim(form->group).empty()) group = std::stoi(form->group);
            } catch (...) {
                // Unparseable means "unset" rather than an error: the field is
                // optional and a typo should fall back to the default, not
                // refuse the whole admission.
            }
            std::string err;
            const auto id = dashboard.admit(form->source,
                                            util::trim(form->expert_gate),
                                            util::trim(form->expert_down),
                                            group,
                                            err);
            if (id.empty()) {
                form->status = err;
                form->status_ok = false;
            } else {
                form->status = "started " + id;
                form->status_ok = true;
                form->open = false;
            }
            form->starting.store(false);
        });
    });

    auto close_button = Button("Close", [form] { form->open = false; });

    auto form_container = Container::Vertical({
        source_input,
        gate_input,
        down_input,
        group_input,
        Container::Horizontal({start_button, close_button}),
    });

    auto admit_button = Button("Admit a model (a)", [form] { form->open = true; });
    auto cancel_button = Button("Cancel selected (c)", [form, &dashboard, &selected] {
        const auto snap = dashboard.snapshot();
        if (selected < 0 || selected >= static_cast<int>(snap.operations.size())) {
            form->status = "no operation selected";
            form->status_ok = false;
            return;
        }
        const auto& op = snap.operations[static_cast<std::size_t>(selected)];
        if (op.done) {
            // Refused with the reason rather than sent and 404'd: "already
            // finished" is not a failure the operator caused.
            form->status = "that admission has already finished";
            form->status_ok = false;
            return;
        }
        std::string err;
        form->status_ok = dashboard.cancel(op.operation_id, err);
        form->status = form->status_ok ? "cancel requested" : err;
    });
    // Re-derive the verdict for the FIRST admitted model, which is the one the
    // registry panel lists at the top. A per-row selector is the better UI and a
    // larger change; this makes the capability reachable rather than leaving it
    // API-only, which is the property under test.
    auto reprofile_button = Button("Reprofile first model", [form, &dashboard] {
        const auto snap = dashboard.snapshot();
        if (snap.models.empty()) {
            form->status = "no admitted models to reprofile";
            form->status_ok = false;
            return;
        }
        std::string err;
        const auto id = dashboard.reprofile(snap.models.front().id, err);
        if (id.empty()) {
            form->status = err;
            form->status_ok = false;
        } else {
            form->status = "reprofiling " + snap.models.front().name + " as " + id;
            form->status_ok = true;
        }
    });

    auto action_row = Container::Horizontal({admit_button, cancel_button, reprofile_button});
    auto container = Container::Vertical({action_row, Maybe(form_container, &form->open)});

    auto renderer = Renderer(
        container,
        [&dashboard,
         &selected,
         form,
         action_row,
         source_input,
         gate_input,
         down_input,
         group_input,
         start_button,
         close_button]() -> Element {
            const auto snap = dashboard.snapshot();
            Elements body{
                render_admission_list(snap, selected) | flex,
                hbox({render_admission_detail(snap, selected) | flex,
                      render_admitted_models(snap) | flex}),
                hbox({text(" "), action_row->Render()}),
            };
            if (!form->status.empty())
                body.push_back(text(" " + form->status) |
                               color(form->status_ok ? Color::Green : Color::Red));

            if (form->open) {
                body.push_back(
                    vbox({
                        text(" Admit a model") | bold,
                        separator(),
                        hbox({text(" source      ") | dim, source_input->Render()}),
                        hbox({text(" expert_gate ") | dim, gate_input->Render()}),
                        hbox({text(" expert_down ") | dim, down_input->Render()}),
                        hbox({text(" group       ") | dim, group_input->Render()}),
                        text(""),
                        text(" Blank quantization fields use the deployment default.") | dim,
                        text(" The same weights at two quantizations are two admissions") | dim,
                        text(" with two verdicts — that is the premise, not a quirk.") | dim,
                        text(" Conversion runs for HOURS and survives closing this tab.") | dim,
                        separator(),
                        hbox({start_button->Render(), text("  "), close_button->Render()}),
                    }) |
                    border);
            }
            return vbox(body);
        });

    return CatchEvent(renderer, [form, &dashboard, &selected](const Event& ev) {
        if (form->open) return false;
        if (ev == Event::Character('a')) {
            form->open = true;
            return true;
        }
        const auto count = static_cast<int>(dashboard.snapshot().operations.size());
        if (count > 0) {
            if (ev == Event::Character('j')) {
                selected = std::min(count - 1, selected + 1);
                return true;
            }
            if (ev == Event::Character('k')) {
                selected = std::max(0, selected - 1);
                return true;
            }
        }
        return false;
    });
}

} // namespace mm
