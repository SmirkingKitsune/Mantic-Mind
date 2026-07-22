#include "control/node_registry.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/pairing.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

namespace mm {

NodeConnectionStatus classify_node_reachability(int64_t unreachable_since_ms,
                                                int64_t now_ms,
                                                int64_t offline_after_ms) {
    if (unreachable_since_ms <= 0) return NodeConnectionStatus::Unreachable;
    if (offline_after_ms <= 0 || now_ms - unreachable_since_ms >= offline_after_ms)
        return NodeConnectionStatus::Offline;
    return NodeConnectionStatus::Unreachable;
}

NodeRegistry::NodeRegistry() = default;

NodeRegistry::NodeRegistry(std::string data_dir) {
    if (data_dir.empty()) return;

    namespace fs = std::filesystem;
    remembered_nodes_path_ = (fs::path(data_dir) / "nodes.json").string();
    load_remembered_nodes();
}
void NodeRegistry::set_offline_after_seconds(int seconds) {
    const int clamped = std::max(1, seconds);
    offline_after_ms_.store(static_cast<int64_t>(clamped) * 1000);
}


NodeRegistry::~NodeRegistry() { stop_health_poll(); }

// ── Node management ────────────────────────────────────────────────────────────
NodeId NodeRegistry::add_node(const std::string& url,
                               const std::string& api_key,
                               const std::string& platform,
                               bool remember,
                               const std::string& hostname) {
    std::lock_guard<std::mutex> g(mutex_);

    for (auto& [id, n] : nodes_) {
        if (n.url != url && (api_key.empty() || n.api_key != api_key)) continue;

        n.url = url;
        // Never wipe a stored credential with an empty one (e.g. a URL-only
        // re-registration must not break a remembered node).
        if (!api_key.empty()) n.api_key = api_key;
        if (!platform.empty()) n.platform = platform;
        if (remember) remembered_nodes_.insert(id);
        if (!hostname.empty()) n.hostname = hostname;
        n.remembered = remembered_nodes_.count(id) > 0;
        if (n.remembered) save_remembered_nodes_unlocked();
        MM_INFO("NodeRegistry: updated node {} @ {}", id, url);
        return id;
    }

    NodeInfo info;
    info.id       = mm::util::generate_uuid();
    info.url      = url;
    info.api_key  = api_key;
    info.platform = platform;
    info.connected = false;
    info.hostname = hostname;
    info.health    = NodeHealthStatus::Unknown;
    info.remembered = remember;
    info.connection_status = NodeConnectionStatus::Unknown;
    nodes_[info.id] = info;
    if (remember) {
        remembered_nodes_.insert(info.id);
        save_remembered_nodes_unlocked();
    }
    MM_INFO("NodeRegistry: added node {} @ {}", info.id, url);
    return info.id;
}

void NodeRegistry::remove_node(const NodeId& id) {
    std::lock_guard<std::mutex> g(mutex_);
    nodes_.erase(id);
    const bool was_remembered = remembered_nodes_.erase(id) > 0;
    if (was_remembered) save_remembered_nodes_unlocked();
    MM_INFO("NodeRegistry: removed node {}", id);
}

bool NodeRegistry::forget_node(const NodeId& id) {
    std::lock_guard<std::mutex> g(mutex_);
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return false;

    const bool was_remembered = remembered_nodes_.erase(id) > 0;
    it->second.remembered = false;
    if (was_remembered) save_remembered_nodes_unlocked();
    MM_INFO("NodeRegistry: forgot node {}", id);
    return was_remembered;
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
                                         const std::string& pin_or_psk,
                                         bool remember) {
    HttpClient cli(url);
    std::string response = mm::pairing::hmac_sha256_hex(pin_or_psk, nonce);
    auto cpl_res = cli.post("/api/node/pair-complete",
                            nlohmann::json{{"challenge", nonce},
                                           {"response",  response},
                                           {"remember",  remember}});
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
        const std::string hostname = j.value("hostname", std::string{});
        if (api_key.empty()) return {};
        add_node(url, api_key, {}, remember, hostname);
        MM_INFO("NodeRegistry: paired with {} — node added{}", url, remember ? " and remembered" : "");
        return api_key;
    } catch (const std::exception& e) {
        MM_WARN("NodeRegistry: pair-complete parse error from {}: {}", url, e.what());
        return {};
    }
}

std::string NodeRegistry::pair_node(const std::string& url,
                                     const std::string& pin_or_psk,
                                     bool remember) {
    std::string nonce = start_pair(url);
    if (nonce.empty()) return {};
    return complete_pair(url, nonce, pin_or_psk, remember);
}

void NodeRegistry::load_remembered_nodes() {
    if (remembered_nodes_path_.empty()) return;

    std::ifstream in(remembered_nodes_path_);
    if (!in.is_open()) return;

    try {
        auto root = nlohmann::json::parse(in);
        const auto& nodes_json = root.is_array() ? root : root.at("nodes");
        std::lock_guard<std::mutex> g(mutex_);
        for (const auto& item : nodes_json) {
            const std::string url = item.value("url", std::string{});
            const std::string api_key = item.value("api_key", std::string{});
            if (url.empty() || api_key.empty()) continue;

            NodeInfo info;
            info.id = item.value("id", mm::util::generate_uuid());
            if (info.id.empty()) info.id = mm::util::generate_uuid();
            info.url = url;
            info.api_key = api_key;
            info.platform = item.value("platform", std::string{});
            info.connected = false;
            info.hostname = item.value("hostname", std::string{});
            info.health = NodeHealthStatus::Unknown;
            info.remembered = true;
            info.connection_status = NodeConnectionStatus::Unknown;

            nodes_[info.id] = info;
            remembered_nodes_.insert(info.id);
        }
        MM_INFO("NodeRegistry: loaded {} remembered node(s)", remembered_nodes_.size());
    } catch (const std::exception& e) {
        MM_WARN("NodeRegistry: failed to load remembered nodes from {}: {}",
                remembered_nodes_path_, e.what());
    }
}

void NodeRegistry::save_remembered_nodes_unlocked() const {
    if (remembered_nodes_path_.empty()) return;

    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path path(remembered_nodes_path_);
    if (path.has_parent_path()) fs::create_directories(path.parent_path(), ec);

    nlohmann::json nodes_json = nlohmann::json::array();
    for (const auto& [id, n] : nodes_) {
        if (remembered_nodes_.count(id) == 0) continue;
        nodes_json.push_back(nlohmann::json{
            {"id", n.id},
            {"url", n.url},
            {"hostname", n.hostname},
            {"api_key", n.api_key},
            {"platform", n.platform}
        });
    }

    const nlohmann::json root{
        {"version", 2},
        {"nodes", nodes_json}
    };

    std::ofstream out(remembered_nodes_path_, std::ios::trunc);
    if (!out.is_open()) {
        MM_WARN("NodeRegistry: could not write remembered nodes to {}", remembered_nodes_path_);
        return;
    }
    out << root.dump(2) << '\n';
}

// ── Health polling ─────────────────────────────────────────────────────────────
void NodeRegistry::start_health_poll(int interval_s) {
    if (polling_.exchange(true)) return;
    poll_thread_ = std::thread([this, interval_s]() {
        while (polling_) {
            poll_all_nodes();
            // Interruptible wait: stop_health_poll() wakes us immediately
            // instead of letting shutdown lag by up to interval_s seconds.
            std::unique_lock<std::mutex> lk(poll_mutex_);
            poll_cv_.wait_for(lk, std::chrono::seconds(interval_s),
                              [this] { return !polling_; });
        }
    });
}

void NodeRegistry::stop_health_poll() {
    {
        std::lock_guard<std::mutex> lk(poll_mutex_);
        polling_ = false;
    }
    poll_cv_.notify_all();
    if (poll_thread_.joinable()) poll_thread_.join();
}

// ── ping_node ─────────────────────────────────────────────────────────────────
// Makes HTTP calls to the node; updates info fields in-place.
// Returns true if node is responding.
bool NodeRegistry::ping_node(NodeInfo& info) {
    HttpClient cli(info.url);
    cli.set_bearer_token(info.api_key);

    // GET /api/node/health → metrics

    auto mark_unreachable = [&]() {
        const int64_t now = mm::util::now_ms();
        if (info.unreachable_since_ms <= 0) info.unreachable_since_ms = now;
        ++info.consecutive_failures;
        info.connected = false;
        info.connection_status = classify_node_reachability(
            info.unreachable_since_ms, now, offline_after_ms_.load());
        // Retain the last metrics sample for display, but make it explicit that
        // current health cannot be known while the node is unreachable.
        info.health = NodeHealthStatus::Unknown;
    };
    auto health_res = cli.get("/api/node/health");
    if (!health_res.ok()) {
        mark_unreachable();
        return false;
    }

    try {
        auto j = nlohmann::json::parse(health_res.body);
        info.metrics   = j.get<NodeHealthMetrics>();
        info.connected = true;

        // Determine health status from resource usage
        float max_pct = std::max(info.metrics.cpu_percent, info.metrics.ram_percent);
        if (max_pct < 80.0f)      info.health = NodeHealthStatus::Healthy;
        else if (max_pct < 95.0f) info.health = NodeHealthStatus::Degraded;
        else                      info.health = NodeHealthStatus::Unhealthy;
        const int64_t now = mm::util::now_ms();
        info.connection_status = NodeConnectionStatus::Online;
        info.last_seen_ms = now;
        info.metrics_sampled_at_ms = now;
        info.unreachable_since_ms = 0;
        info.consecutive_failures = 0;
    } catch (const std::exception& e) {
        MM_WARN("NodeRegistry: health parse error for {}: {}", info.id, e.what());
        mark_unreachable();
        return false;
    }

    // GET /api/node/status → slots, loaded_model
    auto status_res = cli.get("/api/node/status");
    if (status_res.ok()) {
        try {
            auto sj = nlohmann::json::parse(status_res.body);
            info.loaded_model = sj.value("loaded_model", std::string{});

            // Parse multi-slot fields if present
            if (sj.contains("slots")) {
                info.slots = sj["slots"].get<std::vector<SlotInfo>>();
                info.slot_snapshot_at_ms = mm::util::now_ms();
            }
            if (sj.contains("hostname"))
                info.hostname = sj["hostname"].get<std::string>();
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
            if (sj.contains("llama_runtime"))
                info.llama_runtime = sj["llama_runtime"].get<LlamaRuntimeStatus>();
            if (sj.contains("action_progress"))
                info.action_progress = sj["action_progress"].get<NodeActionProgress>();
            if (sj.contains("capabilities"))
                info.capabilities = sj["capabilities"].get<NodeCapabilities>();

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

        const auto previous_connection = info.connection_status;
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
                it->second.disk_free_mb  = info.disk_free_mb;
                it->second.max_slots     = info.max_slots;
                it->second.slot_in_use   = info.slot_in_use;
                it->second.connection_status = info.connection_status;
                it->second.last_seen_ms = info.last_seen_ms;
                it->second.slot_snapshot_at_ms = info.slot_snapshot_at_ms;
                it->second.unreachable_since_ms = info.unreachable_since_ms;
                it->second.metrics_sampled_at_ms = info.metrics_sampled_at_ms;
                it->second.consecutive_failures = info.consecutive_failures;
                it->second.hostname = info.hostname;
                it->second.slot_available = info.slot_available;
                it->second.slot_ready    = info.slot_ready;
                it->second.slot_loading  = info.slot_loading;
                it->second.slot_suspending = info.slot_suspending;
                it->second.slot_suspended = info.slot_suspended;
                it->second.slot_error    = info.slot_error;
                it->second.llama_server_path = info.llama_server_path;
                it->second.llama_runtime = info.llama_runtime;
                it->second.action_progress = info.action_progress;
                it->second.capabilities = info.capabilities;
            }
            cb = update_cb_;
        }

        if (info.connection_status != previous_connection) {
            MM_INFO("Node {} is now {}", id, to_string(info.connection_status));
        }

        if (cb) cb(info);
    }
}

} // namespace mm
