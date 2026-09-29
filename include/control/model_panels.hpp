#pragma once

// Mantic-Mind — the Models tab: the served-model catalog users pick from.
//
// Every model an operator can offer — admitted models Soma can serve, which are
// listed without being created, and the llama.cpp GGUF and remote-API models an
// operator adds — with the engine each routes to, where it is placed, and how
// many requests are using it.
//
// Same rule as the Soma, Engines and Admissions tabs: every value here arrives
// over `/v1/*`, and every action is a `/v1/*` call. This panel does not touch
// ServedModelCatalog, the scheduler or the registry even though it renders
// their state — a TUI that reached into them would be a client with privileges
// no other client has. tools/ci/check_ui_api.py guards this header and its .cpp.

#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mm {

/// One catalog entry, as `GET /v1/served-models` reports it. The full wire
/// document is kept alongside, so the editor round-trips fields this view does
/// not name rather than silently resetting them.
struct ServedModelView {
    std::string id;
    std::string display_name;
    std::string model_path;
    std::string source;            ///< explicit | admitted
    std::string inference_backend; ///< node | api
    std::string backend_override;  ///< auto | soma | fallback
    bool enabled = true;
    bool available = true;
    std::string unavailable_reason;
    std::string backend;           ///< resolved engine, "" for API-backed
    std::string backend_reason;
    std::string status;            ///< unplaced | idle | active | suspended | api
    std::string node_id;
    std::string slot_id;
    int in_flight = 0;
    int ctx_size = 0;
    bool tool_calling = false;
    bool vision = false;
    nlohmann::json raw = nlohmann::json::object();
};

struct ServedModelSnapshot {
    bool reachable = false;
    std::string error;
    std::vector<ServedModelView> models;
};

/// Polls `/v1/served-models` on its own thread and performs the tab's actions.
class ModelDashboard {
public:
    ModelDashboard(std::string base_url, std::string api_token);
    ~ModelDashboard();

    ModelDashboard(const ModelDashboard&) = delete;
    ModelDashboard& operator=(const ModelDashboard&) = delete;

    void start(int interval_ms);
    void stop();
    void refresh_now();

    ServedModelSnapshot snapshot() const;

    /// POST /v1/served-models. `out_error` carries the server's reason,
    /// including every validation issue.
    bool create(const nlohmann::json& model, std::string& out_error);
    /// PUT /v1/served-models/{id}
    bool update(const std::string& id, const nlohmann::json& model, std::string& out_error);
    /// DELETE /v1/served-models/{id}
    bool remove(const std::string& id, std::string& out_error);
    /// PUT /v1/served-models/{id}/backend
    bool set_backend(const std::string& id, const std::string& override_value,
                     std::string& out_error);
    /// POST /v1/served-models/{id}/{suspend|restore|release}
    bool lifecycle(const std::string& id, const std::string& verb, bool force,
                   std::string& out_error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// A readable reason from an error response: the server's `error`, plus each
/// validation issue as `field: message`. Pure, so it can be asserted.
std::string describe_error_body(int status, const std::string& body);

// ── Pure renderers ────────────────────────────────────────────────────────────

/// The catalog: id, where it comes from, engine, status, requests in flight.
/// `selected` out of range renders without a highlight rather than clamping.
ftxui::Element render_served_model_list(const ServedModelSnapshot& snap, int selected);

/// The selected entry: routing and its reason, placement, and the settings an
/// operator most often needs to see.
ftxui::Element render_served_model_detail(const ServedModelSnapshot& snap, int selected);

/// The whole tab. `selected` indexes the model list.
ftxui::Component model_tab(ModelDashboard& dashboard, int& selected);

} // namespace mm
