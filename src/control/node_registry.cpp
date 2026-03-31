#include "control/node_registry.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/pairing.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

namespace mm {

NodeRegistry::NodeRegistry()  = default;
NodeRegistry::~NodeRegistry() { stop_health_poll(); }

// ── Node management ────────────────────────────────────────────────────────────
NodeId NodeRegistry::add_node(const std::string& url,
                               const std::string& api_key,
                               const std::string& platform) {
    NodeInfo info;
    info.id       = mm::util::generate_uuid();
    info.url      = url;
    info.api_key  = api_key;
    info.platform = platform;
    info.connected = false;
    info.health    = NodeHealthStatus::Unknown;
    std::lock_guard<std::mutex> g(mutex_);
    nodes_[info.id] = info;
    MM_INFO("NodeRegistry: added node {} @ {}", info.id, url);
    return info.id;
}

void NodeRegistry::remove_node(const NodeId& id) {
    std::lock_guard<std::mutex> g(mutex_);
    nodes_.erase(id);
    MM_INFO("NodeRegistry: removed node {}", id);
}

NodeInfo NodeRegistry::get_node(const NodeId& id) const {
    std::lock_guard<std::mutex> g(mutex_);
    return nodes_.at(id);
}

std::vector<NodeInfo> NodeRegistry::list_nodes() const {
    std::lock_guard<std::mutex> g(mutex_);
    std::vector<NodeInfo> out;
    out.reserve(nodes_.size());
    for (auto& [_, n] : nodes_) out.push_back(n);
    return out;
}

std::optional<NodeInfo> NodeRegistry::find_node_by_api_key(const std::string& api_key) const {
    std::lock_guard<std::mutex> g(mutex_);
    for (const auto& [_, n] : nodes_) {
        if (n.api_key == api_key) return n;
    }
    return std::nullopt;
}

void NodeRegistry::set_node_loaded_model(const NodeId& id, const std::string& model) {
    std::lock_guard<std::mutex> g(mutex_);
    auto it = nodes_.find(id);
    if (it != nodes_.end()) it->second.loaded_model = model;
}

std::vector<NodeInfo> NodeRegistry::nodes_with_model(const std::string& model) const {
    std::lock_guard<std::mutex> g(mutex_);
    std::vector<NodeInfo> out;
    for (auto& [_, n] : nodes_)
        if (n.loaded_model == model && n.connected) out.push_back(n);
    return out;
}

std::vector<NodeInfo> NodeRegistry::available_nodes() const {
    std::lock_guard<std::mutex> g(mutex_);
    std::vector<NodeInfo> out;
    for (auto& [_, n] : nodes_)
        if (n.connected) out.push_back(n);
    return out;
}

// ── Multi-slot queries ────────────────────────────────────────────────────────

void NodeRegistry::update_node_slots(const NodeId& id,
                                      const std::vector<SlotInfo>& slots) {
    std::lock_guard<std::mutex> g(mutex_);
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return;

    it->second.slots = slots;
    it->second.slot_ready = 0;
    it->second.slot_loading = 0;
    it->second.slot_suspending = 0;
    it->second.slot_suspended = 0;
    it->second.slot_error = 0;
    for (const auto& s : slots) {
        switch (s.state) {
            case SlotState::Ready:      ++it->second.slot_ready; break;
            case SlotState::Loading:    ++it->second.slot_loading; break;
            case SlotState::Suspending: ++it->second.slot_suspending; break;
            case SlotState::Suspended:  ++it->second.slot_suspended; break;
            case SlotState::Error:      ++it->second.slot_error; break;
            case SlotState::Empty:
            default:
                break;
        }
    }
    it->second.slot_in_use = it->second.slot_ready + it->second.slot_loading
                           + it->second.slot_suspending + it->second.slot_error;
    it->second.slot_available = std::max(0, it->second.max_slots - it->second.slot_in_use);
}

void NodeRegistry::update_node_storage(const NodeId& id,
                                        const std::vector<StoredModel>& models,
                                        int64_t disk_free_mb) {
    std::lock_guard<std::mutex> g(mutex_);
    auto it = nodes_.find(id);
    if (it != nodes_.end()) {
        it->second.stored_models = models;
        it->second.disk_free_mb  = disk_free_mb;
    }
}

std::vector<NodeInfo> NodeRegistry::nodes_with_model_loaded(
    const std::string& model_path) const
{
    std::lock_guard<std::mutex> g(mutex_);
    std::vector<NodeInfo> out;
    for (auto& [_, n] : nodes_) {
        if (!n.connected) continue;
        for (const auto& s : n.slots) {
            if (s.model_path == model_path && s.state == SlotState::Ready) {
                out.push_back(n);
                break;
            }
        }
    }
    return out;
}

std::vector<NodeInfo> NodeRegistry::nodes_with_model_stored(
    const std::string& model_path) const
{
    std::lock_guard<std::mutex> g(mutex_);
    std::vector<NodeInfo> out;
    for (auto& [_, n] : nodes_) {
        if (!n.connected) continue;
        for (const auto& m : n.stored_models) {
            if (m.model_path == model_path) {
                out.push_back(n);
                break;
            }
        }
    }
    return out;
}

std::vector<NodeInfo> NodeRegistry::nodes_with_available_vram(
    int64_t min_vram_mb) const
{
    std::lock_guard<std::mutex> g(mutex_);
    struct Candidate {
        NodeInfo info;
        bool     native_vram_fit = false;
        int64_t  score_mb = 0;
    };
    std::vector<Candidate> cands;

    constexpr int64_t kVramHeadroomMb = 1024;       // keep 1 GiB VRAM safety margin
    constexpr int64_t kRamHeadroomMb  = 2048;       // keep 2 GiB system RAM safety margin
    constexpr double  kRamOffloadWeight = 0.60;     // CPU-offloaded weights are slower/less reliable
    constexpr int64_t kMinGpuForOffloadMb = 8192;   // require at least 8 GiB GPU for hybrid loads

    for (auto& [_, n] : nodes_) {
        if (!n.connected) continue;

        int64_t vram_free_raw = n.metrics.gpu_vram_total_mb - n.metrics.gpu_vram_used_mb;
        int64_t ram_free_raw  = n.metrics.ram_total_mb - n.metrics.ram_used_mb;
        int64_t vram_free = std::max<int64_t>(0, vram_free_raw - kVramHeadroomMb);
        int64_t ram_free  = std::max<int64_t>(0, ram_free_raw  - kRamHeadroomMb);

        bool native_fit = vram_free >= min_vram_mb;
        bool allow_offload = (n.metrics.gpu_vram_total_mb >= kMinGpuForOffloadMb) ||
                             (n.metrics.gpu_vram_total_mb <= 0);
        int64_t effective_mb = vram_free + static_cast<int64_t>(ram_free * kRamOffloadWeight);
        bool offload_fit = allow_offload && effective_mb >= min_vram_mb;

        if (!native_fit && !offload_fit) continue;

        Candidate c;
        c.info = n;
        c.native_vram_fit = native_fit;
        c.score_mb = native_fit ? vram_free : effective_mb;
        cands.push_back(std::move(c));
    }

    std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b) {
        if (a.native_vram_fit != b.native_vram_fit) return a.native_vram_fit > b.native_vram_fit;
        if (a.score_mb != b.score_mb) return a.score_mb > b.score_mb;
        return a.info.id < b.info.id;
    });

    std::vector<NodeInfo> out;
    out.reserve(cands.size());
    for (const auto& c : cands) out.push_back(c.info);
    return out;
}

void NodeRegistry::set_update_callback(UpdateCallback cb) {
    std::lock_guard<std::mutex> g(mutex_); update_cb_ = std::move(cb);
}

// ── Discovery ──────────────────────────────────────────────────────────────────
void NodeRegistry::start_discovery_listen(uint16_t port) {
    discovery_listener_.set_callback([this](const DiscoveredNode&) {
        // Fire the update callback to nudge the UI.
        UpdateCallback cb;
        { std::lock_guard<std::mutex> g(mutex_); cb = update_cb_; }
        // We pass a sentinel NodeInfo to trigger a refresh; UI calls
        // get_discovered_nodes() independently.
        if (cb) {
            NodeInfo sentinel;
            cb(sentinel);
        }
    });
    discovery_listener_.start(port);
    MM_INFO("NodeRegistry: discovery listener started on port {}", port);
}

void NodeRegistry::stop_discovery_listen() {
    discovery_listener_.stop();
}

std::vector<DiscoveredNode> NodeRegistry::get_discovered_nodes() const {
    auto all = discovery_listener_.get_nodes();

    // Build set of already-registered URLs.
    std::unordered_map<std::string, bool> registered_urls;
    {
        std::lock_guard<std::mutex> g(mutex_);
        for (auto& [_, n] : nodes_)
            registered_urls[n.url] = true;
    }

    std::vector<DiscoveredNode> out;
    for (auto& dn : all)
        if (!registered_urls.count(dn.url))
            out.push_back(dn);
    return out;
}

// ── Pairing ────────────────────────────────────────────────────────────────────
std::string NodeRegistry::start_pair(const std::string& url) {
    HttpClient cli(url);
    std::string nonce = mm::pairing::generate_nonce();
    auto req_res = cli.post("/api/node/pair-request",
                            nlohmann::json{{"challenge", nonce}});
    if (!req_res.ok()) {
        MM_WARN("NodeRegistry: pair-request to {} failed (HTTP {})", url, req_res.status);
        return {};
    }
    try {
        auto j = nlohmann::json::parse(req_res.body);
        if (!j.value("accepted", false)) {
            MM_WARN("NodeRegistry: pair-request rejected by {}", url);
            return {};
        }
        MM_INFO("NodeRegistry: pair-request accepted by {} (mode={})",
                url, j.value("mode", std::string{}));
        return nonce;
    } catch (const std::exception& e) {
        MM_WARN("NodeRegistry: pair-request parse error from {}: {}", url, e.what());
        return {};
    }
}

std::string NodeRegistry::complete_pair(const std::string& url,
                                         const std::string& nonce,
                                         const std::string& pin_or_psk) {
    HttpClient cli(url);
    std::string response = mm::pairing::hmac_sha256_hex(pin_or_psk, nonce);
    auto cpl_res = cli.post("/api/node/pair-complete",
                            nlohmann::json{{"challenge", nonce},
                                           {"response",  response}});
    if (!cpl_res.ok()) {
        MM_WARN("NodeRegistry: pair-complete to {} failed (HTTP {})", url, cpl_res.status);
        return {};
    }
    try {
        auto j = nlohmann::json::parse(cpl_res.body);
        if (!j.value("accepted", false)) {
            MM_WARN("NodeRegistry: pair-complete rejected by {}: {}",
                    url, j.value("error", std::string{}));
            return {};
        }
        std::string api_key = j.value("api_key", std::string{});
        if (api_key.empty()) return {};
        add_node(url, api_key);
        MM_INFO("NodeRegistry: paired with {} — node added", url);
        return api_key;
    } catch (const std::exception& e) {
        MM_WARN("NodeRegistry: pair-complete parse error from {}: {}", url, e.what());
        return {};
    }
}

std::string NodeRegistry::pair_node(const std::string& url,
                                     const std::string& pin_or_psk) {
    std::string nonce = start_pair(url);
    if (nonce.empty()) return {};
    return complete_pair(url, nonce, pin_or_psk);
}

bool NodeRegistry::request_llama_update(const NodeId& id,
                                        bool build,
                                        bool force,
                                        std::string* out_message) {
    NodeInfo info;
    {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = nodes_.find(id);
        if (it == nodes_.end()) {
            if (out_message) *out_message = "node not found";
            return false;
        }
        info = it->second;
    }

    HttpClient cli(info.url);
    cli.set_bearer_token(info.api_key);

    auto res = cli.post("/api/node/llama/update",
                        nlohmann::json{{"build", build}, {"force", force}});
    if (!res.ok()) {
        if (out_message) {
            std::string msg = "HTTP " + std::to_string(res.status);
            if (!res.body.empty()) msg += ": " + res.body;
            *out_message = std::move(msg);
        }
        return false;
    }

    try {
        auto j = nlohmann::json::parse(res.body);
        bool accepted = j.value("accepted", false);
        std::string message = j.value("message", std::string{});
        if (out_message) *out_message = message.empty() ? "accepted" : message;
        return accepted;
    } catch (const std::exception& e) {
        if (out_message) *out_message = e.what();
        return false;
    }
}

bool NodeRegistry::request_llama_check_update(const NodeId& id,
                                              std::string* out_message) {
    NodeInfo info;
    {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = nodes_.find(id);
        if (it == nodes_.end()) {
            if (out_message) *out_message = "node not found";
            return false;
        }
        info = it->second;
    }

    HttpClient cli(info.url);
    cli.set_bearer_token(info.api_key);
    auto res = cli.post("/api/node/llama/check-update", nlohmann::json::object());
    if (!res.ok()) {
        if (out_message) {
            std::string msg = "HTTP " + std::to_string(res.status);
            if (!res.body.empty()) msg += ": " + res.body;
            *out_message = std::move(msg);
        }
        return false;
    }

    try {
        auto j = nlohmann::json::parse(res.body);
        bool ok = j.value("ok", true);
        std::string msg = j.value("message", std::string{});
        if (out_message) *out_message = msg.empty() ? (ok ? "ok" : "failed") : msg;
        return ok;
    } catch (const std::exception& e) {
        if (out_message) *out_message = e.what();
        return false;
    }
}

// ── Health polling ─────────────────────────────────────────────────────────────
void NodeRegistry::start_health_poll(int interval_s) {
    if (polling_.exchange(true)) return;
    poll_thread_ = std::thread([this, interval_s]() {
        while (polling_) {
            poll_all_nodes();
            std::this_thread::sleep_for(std::chrono::seconds(interval_s));
        }
    });
}

void NodeRegistry::stop_health_poll() {
    polling_ = false;
    if (poll_thread_.joinable()) poll_thread_.join();
}

// ── ping_node ─────────────────────────────────────────────────────────────────
// Makes HTTP calls to the node; updates info fields in-place.
// Returns true if node is responding.
bool NodeRegistry::ping_node(NodeInfo& info) {
    HttpClient cli(info.url);
    cli.set_bearer_token(info.api_key);

    // GET /api/node/health → metrics
    auto health_res = cli.get("/api/node/health");
    if (!health_res.ok()) {
        info.connected = false;
        info.health    = NodeHealthStatus::Unhealthy;
        return false;
    }

    try {
        auto j = nlohmann::json::parse(health_res.body);
        info.metrics   = j.get<NodeHealthMetrics>();
        info.connected = true;

        // Determine health status from resource usage
        float max_pct = std::max(info.metrics.cpu_percent, info.metrics.ram_percent);
        if (max_pct < 80.0f) info.health = NodeHealthStatus::Healthy;
        else if (max_pct < 95.0f) info.health = NodeHealthStatus::Degraded;
        else                      info.health = NodeHealthStatus::Unhealthy;
    } catch (const std::exception& e) {
        MM_WARN("NodeRegistry: health parse error for {}: {}", info.id, e.what());
        info.connected = false;
        info.health    = NodeHealthStatus::Unhealthy;
        return false;
    }

    // GET /api/node/status → slots, stored_models, loaded_model
    auto status_res = cli.get("/api/node/status");
    if (status_res.ok()) {
        try {
            auto sj = nlohmann::json::parse(status_res.body);
            info.loaded_model = sj.value("loaded_model", std::string{});

            // Parse multi-slot fields if present
            if (sj.contains("slots"))
                info.slots = sj["slots"].get<std::vector<SlotInfo>>();
            if (sj.contains("stored_models"))
                info.stored_models = sj["stored_models"].get<std::vector<StoredModel>>();
            if (sj.contains("disk_free_mb"))
                info.disk_free_mb = sj["disk_free_mb"].get<int64_t>();
            if (sj.contains("max_slots"))
                info.max_slots = sj["max_slots"].get<int>();
            if (sj.contains("slot_in_use"))
                info.slot_in_use = sj["slot_in_use"].get<int>();
            if (sj.contains("slot_available"))
                info.slot_available = sj["slot_available"].get<int>();
            if (sj.contains("slot_ready"))
                info.slot_ready = sj["slot_ready"].get<int>();
            if (sj.contains("slot_loading"))
                info.slot_loading = sj["slot_loading"].get<int>();
            if (sj.contains("slot_suspending"))
                info.slot_suspending = sj["slot_suspending"].get<int>();
            if (sj.contains("slot_suspended"))
                info.slot_suspended = sj["slot_suspended"].get<int>();
            if (sj.contains("slot_error"))
                info.slot_error = sj["slot_error"].get<int>();
            if (sj.contains("llama_server_path"))
                info.llama_server_path = sj["llama_server_path"].get<std::string>();
            if (sj.contains("llama_update_running"))
                info.llama_update_running = sj["llama_update_running"].get<bool>();
            if (sj.contains("llama_update_status"))
                info.llama_update_status = sj["llama_update_status"].get<std::string>();
            if (sj.contains("llama_update_message"))
                info.llama_update_message = sj["llama_update_message"].get<std::string>();
            if (sj.contains("llama_update_started_ms"))
                info.llama_update_started_ms = sj["llama_update_started_ms"].get<int64_t>();
            if (sj.contains("llama_update_finished_ms"))
                info.llama_update_finished_ms = sj["llama_update_finished_ms"].get<int64_t>();
            if (sj.contains("llama_install_root"))
                info.llama_install_root = sj["llama_install_root"].get<std::string>();
            if (sj.contains("llama_repo_dir"))
                info.llama_repo_dir = sj["llama_repo_dir"].get<std::string>();
            if (sj.contains("llama_build_dir"))
                info.llama_build_dir = sj["llama_build_dir"].get<std::string>();
            if (sj.contains("llama_binary_path"))
                info.llama_binary_path = sj["llama_binary_path"].get<std::string>();
            if (sj.contains("llama_installed_commit"))
                info.llama_installed_commit = sj["llama_installed_commit"].get<std::string>();
            if (sj.contains("llama_remote_commit"))
                info.llama_remote_commit = sj["llama_remote_commit"].get<std::string>();
            if (sj.contains("llama_remote_error"))
                info.llama_remote_error = sj["llama_remote_error"].get<std::string>();
            if (sj.contains("llama_remote_checked_ms"))
                info.llama_remote_checked_ms = sj["llama_remote_checked_ms"].get<int64_t>();
            if (sj.contains("llama_update_available"))
                info.llama_update_available = sj["llama_update_available"].get<bool>();
            if (sj.contains("llama_update_reason"))
                info.llama_update_reason = sj["llama_update_reason"].get<std::string>();
            if (sj.contains("llama_update_log_path"))
                info.llama_update_log_path = sj["llama_update_log_path"].get<std::string>();

            // Backfill occupancy counts when a node returns only raw slots.
            if (!sj.contains("slot_in_use")) {
                info.slot_ready = 0;
                info.slot_loading = 0;
                info.slot_suspending = 0;
                info.slot_suspended = 0;
                info.slot_error = 0;
                for (const auto& s : info.slots) {
                    switch (s.state) {
                        case SlotState::Ready:      ++info.slot_ready; break;
                        case SlotState::Loading:    ++info.slot_loading; break;
                        case SlotState::Suspending: ++info.slot_suspending; break;
                        case SlotState::Suspended:  ++info.slot_suspended; break;
                        case SlotState::Error:      ++info.slot_error; break;
                        case SlotState::Empty:
                        default:
                            break;
                    }
                }
                info.slot_in_use = info.slot_ready + info.slot_loading
                                 + info.slot_suspending + info.slot_error;
                info.slot_available = std::max(0, info.max_slots - info.slot_in_use);
            }
        } catch (const std::exception& e) {
            MM_WARN("NodeRegistry: status parse error for {}: {}", info.id, e.what());
        }
    }

    auto llama_status_res = cli.get("/api/node/llama/status");
    if (llama_status_res.ok()) {
        try {
            auto lj = nlohmann::json::parse(llama_status_res.body);
            if (lj.contains("binary_path"))
                info.llama_binary_path = lj["binary_path"].get<std::string>();
            if (lj.contains("installed_commit"))
                info.llama_installed_commit = lj["installed_commit"].get<std::string>();
            if (lj.contains("remote_commit"))
                info.llama_remote_commit = lj["remote_commit"].get<std::string>();
            if (lj.contains("remote_error"))
                info.llama_remote_error = lj["remote_error"].get<std::string>();
            if (lj.contains("remote_checked_ms"))
                info.llama_remote_checked_ms = lj["remote_checked_ms"].get<int64_t>();
            if (lj.contains("update_available"))
                info.llama_update_available = lj["update_available"].get<bool>();
            if (lj.contains("update_reason"))
                info.llama_update_reason = lj["update_reason"].get<std::string>();
            if (lj.contains("last_log_path"))
                info.llama_update_log_path = lj["last_log_path"].get<std::string>();
        } catch (const std::exception& e) {
            MM_WARN("NodeRegistry: llama status parse error for {}: {}", info.id, e.what());
        }
    }

    return true;
}

// ── poll_all_nodes ─────────────────────────────────────────────────────────────
void NodeRegistry::poll_all_nodes() {
    // Snapshot IDs to avoid holding the mutex during HTTP calls
    std::vector<NodeId> ids;
    {
        std::lock_guard<std::mutex> g(mutex_);
        for (auto& [id, _] : nodes_) ids.push_back(id);
    }

    for (auto& id : ids) {
        // Get a copy of the current info
        NodeInfo info;
        {
            std::lock_guard<std::mutex> g(mutex_);
            auto it = nodes_.find(id);
            if (it == nodes_.end()) continue;
            info = it->second;
        }

        bool was_connected = info.connected;
        ping_node(info); // modifies info in place

        // Write updated info back; grab callback outside the lock
        UpdateCallback cb;
        {
            std::lock_guard<std::mutex> g(mutex_);
            auto it = nodes_.find(id);
            if (it != nodes_.end()) {
                it->second.connected     = info.connected;
                it->second.health        = info.health;
                it->second.metrics       = info.metrics;
                it->second.loaded_model  = info.loaded_model;
                it->second.slots         = info.slots;
                it->second.stored_models = info.stored_models;
                it->second.disk_free_mb  = info.disk_free_mb;
                it->second.max_slots     = info.max_slots;
                it->second.slot_in_use   = info.slot_in_use;
                it->second.slot_available = info.slot_available;
                it->second.slot_ready    = info.slot_ready;
                it->second.slot_loading  = info.slot_loading;
                it->second.slot_suspending = info.slot_suspending;
                it->second.slot_suspended = info.slot_suspended;
                it->second.slot_error    = info.slot_error;
                it->second.llama_server_path = info.llama_server_path;
                it->second.llama_update_running     = info.llama_update_running;
                it->second.llama_update_status      = info.llama_update_status;
                it->second.llama_update_message     = info.llama_update_message;
                it->second.llama_update_started_ms  = info.llama_update_started_ms;
                it->second.llama_update_finished_ms = info.llama_update_finished_ms;
                it->second.llama_install_root = info.llama_install_root;
                it->second.llama_repo_dir = info.llama_repo_dir;
                it->second.llama_build_dir = info.llama_build_dir;
                it->second.llama_binary_path = info.llama_binary_path;
                it->second.llama_installed_commit = info.llama_installed_commit;
                it->second.llama_remote_commit = info.llama_remote_commit;
                it->second.llama_remote_error = info.llama_remote_error;
                it->second.llama_remote_checked_ms = info.llama_remote_checked_ms;
                it->second.llama_update_available = info.llama_update_available;
                it->second.llama_update_reason = info.llama_update_reason;
                it->second.llama_update_log_path = info.llama_update_log_path;
            }
            cb = update_cb_;
        }

        if (info.connected != was_connected)
            MM_INFO("Node {} is now {}", id, info.connected ? "connected" : "disconnected");

        if (cb) cb(info);
    }
}

} // namespace mm
