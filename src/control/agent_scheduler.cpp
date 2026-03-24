#include "control/agent_scheduler.hpp"
#include "control/node_registry.hpp"
#include "control/model_distributor.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/model_catalog.hpp"
#include "common/gguf_metadata.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <regex>
#include <unordered_set>

namespace fs = std::filesystem;

namespace mm {

AgentScheduler::AgentScheduler(NodeRegistry& registry,
                               ModelDistributor& distributor,
                               std::string models_dir)
    : registry_(registry)
    , distributor_(distributor)
    , models_dir_(std::move(models_dir))
{}

// ── Main scheduling entry point ──────────────────────────────────────────────

std::optional<ScheduleResult> AgentScheduler::ensure_agent_running(
    const AgentConfig& cfg)
{
    std::lock_guard lock(mutex_);
    last_error_.clear();

    // 1. Existing active placement?
    auto pit = placements_.find(cfg.id);
    if (pit != placements_.end() && !pit->second.suspended) {
        pit->second.last_active_ms = util::now_ms();
        return ScheduleResult{pit->second.node_id, pit->second.slot_id};
    }

    int64_t vram_needed = estimate_vram_mb(cfg.model_path);
    const std::string model_filename = canonical_model_filename(cfg.model_path);
    if (model_filename != cfg.model_path) {
        MM_WARN("AgentScheduler: agent {} model_path '{}' treated as catalog filename '{}'",
                cfg.id, cfg.model_path, model_filename);
    }

    // 2. Suspended placement? Try to restore.
    if (pit != placements_.end() && pit->second.suspended) {
        auto& placement = pit->second;

        // Try restore on same node first.
        auto slot_id = restore_agent_on_node(placement, cfg, placement.node_id);
        if (slot_id) {
            placement.slot_id       = *slot_id;
            placement.suspended     = false;
            placement.last_active_ms = util::now_ms();
            MM_INFO("AgentScheduler: restored agent {} on original node {}",
                    cfg.id, placement.node_id);
            return ScheduleResult{placement.node_id, *slot_id};
        }

        // Same node didn't work — try suspending an idle agent there first.
        auto lru = find_lru_idle_agent(placement.node_id);
        if (lru && *lru != cfg.id) {
            suspend_agent(*lru);
            slot_id = restore_agent_on_node(placement, cfg, placement.node_id);
            if (slot_id) {
                placement.slot_id       = *slot_id;
                placement.suspended     = false;
                placement.last_active_ms = util::now_ms();
                return ScheduleResult{placement.node_id, *slot_id};
            }
        }

        // Try other nodes.
        for (const auto& node : registry_.available_nodes()) {
            if (node.id == placement.node_id) continue;
            slot_id = restore_agent_on_node(placement, cfg, node.id);
            if (slot_id) {
                placement.node_id       = node.id;
                placement.slot_id       = *slot_id;
                placement.suspended     = false;
                placement.last_active_ms = util::now_ms();
                return ScheduleResult{node.id, *slot_id};
            }
        }
    }

    // 3. No placement — find a node.

    // 3a. Preferred node has model stored + VRAM?
    if (!cfg.preferred_node_id.empty()) {
        auto slot_id = load_agent_on_node(cfg, cfg.preferred_node_id);
        if (slot_id) {
            AgentPlacement p;
            p.agent_id       = cfg.id;
            p.node_id        = cfg.preferred_node_id;
            p.slot_id        = *slot_id;
            p.placed_at_ms   = util::now_ms();
            p.last_active_ms = p.placed_at_ms;
            placements_[cfg.id] = p;
            return ScheduleResult{p.node_id, *slot_id};
        }
    }

    // 3b. Any node has model stored + VRAM?
    auto stored_nodes = registry_.nodes_with_model_stored(model_filename);
    for (const auto& node : stored_nodes) {
        if (node.id == cfg.preferred_node_id) continue; // already tried
        auto slot_id = load_agent_on_node(cfg, node.id);
        if (slot_id) {
            AgentPlacement p;
            p.agent_id       = cfg.id;
            p.node_id        = node.id;
            p.slot_id        = *slot_id;
            p.placed_at_ms   = util::now_ms();
            p.last_active_ms = p.placed_at_ms;
            placements_[cfg.id] = p;
            return ScheduleResult{node.id, *slot_id};
        }
    }

    // 3c. Any node has VRAM -> try load (node can auto-pull), then push fallback.
    auto vram_nodes = registry_.nodes_with_available_vram(vram_needed);
    for (const auto& node : vram_nodes) {
        // Check if model is on this node already.
        bool has_model = false;
        for (const auto& m : node.stored_models) {
            if (m.model_path == model_filename) { has_model = true; break; }
        }

        auto slot_id = load_agent_on_node(cfg, node.id);
        if (slot_id) {
            AgentPlacement p;
            p.agent_id       = cfg.id;
            p.node_id        = node.id;
            p.slot_id        = *slot_id;
            p.placed_at_ms   = util::now_ms();
            p.last_active_ms = p.placed_at_ms;
            placements_[cfg.id] = p;
            return ScheduleResult{node.id, *slot_id};
        }

        // Fallback path: if remote and model wasn't present, try control push then retry load.
        if (!has_model && !is_local_node(node.url)) {
            MM_INFO("AgentScheduler: load failed on node {}; trying distributor push fallback for {}",
                    node.id, model_filename);
            if (distributor_.upload_model(node.id, model_filename)) {
                slot_id = load_agent_on_node(cfg, node.id);
                if (slot_id) {
                    AgentPlacement p;
                    p.agent_id       = cfg.id;
                    p.node_id        = node.id;
                    p.slot_id        = *slot_id;
                    p.placed_at_ms   = util::now_ms();
                    p.last_active_ms = p.placed_at_ms;
                    placements_[cfg.id] = p;
                    return ScheduleResult{node.id, *slot_id};
                }
            } else {
                MM_WARN("AgentScheduler: model distributor fallback failed for node {}",
                        node.id);
            }
        }
    }

    // 3d. No VRAM anywhere → suspend LRU idle agent, retry.
    auto lru = find_lru_idle_agent();
    if (lru && *lru != cfg.id) {
        auto lru_placement = placements_.at(*lru);
        suspend_agent(*lru);

        // Retry on the freed node.
        auto slot_id = load_agent_on_node(cfg, lru_placement.node_id);
        if (slot_id) {
            AgentPlacement p;
            p.agent_id       = cfg.id;
            p.node_id        = lru_placement.node_id;
            p.slot_id        = *slot_id;
            p.placed_at_ms   = util::now_ms();
            p.last_active_ms = p.placed_at_ms;
            placements_[cfg.id] = p;
            return ScheduleResult{p.node_id, *slot_id};
        }
    }

    // 3e. All nodes full, no idle agents — return nullopt.
    MM_WARN("AgentScheduler: no capacity for agent {} (model={})",
            cfg.id, cfg.model_path);
    if (last_error_.empty()) {
        last_error_ = "no capacity: no connected node could load this model";
    }
    return std::nullopt;
}

void AgentScheduler::release_agent(const AgentId& agent_id) {
    std::lock_guard lock(mutex_);

    auto it = placements_.find(agent_id);
    if (it == placements_.end()) return;

    auto& p = it->second;
    if (!p.suspended) {
        // Unload the slot on the node.
        try {
            auto node = registry_.get_node(p.node_id);
            HttpClient cli(node.url);
            cli.set_bearer_token(node.api_key);
            cli.post("/api/node/unload-model",
                     nlohmann::json{{"slot_id", p.slot_id}});
        } catch (const std::exception& e) {
            MM_WARN("AgentScheduler: failed to unload slot for agent {}: {}",
                    agent_id, e.what());
        }
    }

    placements_.erase(it);
    MM_INFO("AgentScheduler: released agent {}", agent_id);
}

void AgentScheduler::mark_agent_idle(const AgentId& agent_id) {
    std::lock_guard lock(mutex_);
    auto it = placements_.find(agent_id);
    if (it != placements_.end()) {
        it->second.is_active      = false;
        it->second.last_active_ms = util::now_ms();
    }
}

void AgentScheduler::mark_agent_active(const AgentId& agent_id) {
    std::lock_guard lock(mutex_);
    auto it = placements_.find(agent_id);
    if (it != placements_.end()) {
        it->second.is_active      = true;
        it->second.last_active_ms = util::now_ms();
    }
}

std::optional<AgentPlacement> AgentScheduler::get_placement(
    const AgentId& agent_id) const
{
    std::lock_guard lock(mutex_);
    auto it = placements_.find(agent_id);
    if (it == placements_.end()) return std::nullopt;
    return it->second;
}

std::vector<AgentPlacement> AgentScheduler::list_placements() const {
    std::lock_guard lock(mutex_);
    std::vector<AgentPlacement> out;
    out.reserve(placements_.size());
    for (const auto& [_, p] : placements_)
        out.push_back(p);
    return out;
}

std::string AgentScheduler::last_error() const {
    std::lock_guard lock(mutex_);
    return last_error_;
}

void AgentScheduler::housekeeping(const std::vector<AgentConfig>& active_agents) {
    std::lock_guard lock(mutex_);

    // Build set of active agent IDs.
    std::unordered_map<std::string, bool> active_ids;
    for (const auto& a : active_agents)
        active_ids[a.id] = true;

    // Remove placements for deleted agents.
    std::vector<AgentId> to_remove;
    for (const auto& [id, p] : placements_) {
        if (!active_ids.count(id)) {
            to_remove.push_back(id);
        }
    }

    for (const auto& id : to_remove) {
        auto& p = placements_[id];
        if (!p.suspended) {
            try {
                auto node = registry_.get_node(p.node_id);
                HttpClient cli(node.url);
                cli.set_bearer_token(node.api_key);
                cli.post("/api/node/unload-model",
                         nlohmann::json{{"slot_id", p.slot_id}});
            } catch (...) {}
        }
        placements_.erase(id);
        MM_INFO("AgentScheduler: housekeeping removed placement for deleted agent {}",
                id);
    }
}

// ── Private helpers ─────────────────────────────────────────────────────────

bool AgentScheduler::is_local_node(const std::string& node_url) {
    static const std::regex local_re(
        R"(https?://(127\.0\.0\.1|localhost|::1)(:\d+)?(/.*)?)",
        std::regex::icase);
    return std::regex_match(node_url, local_re);
}

int64_t AgentScheduler::estimate_vram_mb(const std::string& model_path) const {
    std::error_code ec;
    const std::string resolved = resolve_model_path_for_metadata(model_path, models_dir_);
    auto sz = fs::file_size(resolved.empty() ? model_path : resolved, ec);
    if (ec) return 2048; // Conservative default: 2 GB
    return static_cast<int64_t>(static_cast<double>(sz) * 1.2 / (1024.0 * 1024.0));
}

std::optional<NodeInfo> AgentScheduler::find_node_with_vram(
    int64_t vram_mb, const NodeId& preferred) const
{
    auto nodes = registry_.nodes_with_available_vram(vram_mb);
    if (nodes.empty()) return std::nullopt;

    // Prefer the requested node.
    if (!preferred.empty()) {
        for (const auto& n : nodes) {
            if (n.id == preferred) return n;
        }
    }
    return nodes.front();
}

std::optional<AgentId> AgentScheduler::find_lru_idle_agent(
    const NodeId& on_node) const
{
    int64_t oldest_ms = INT64_MAX;
    std::optional<AgentId> oldest_id;

    for (const auto& [id, p] : placements_) {
        if (p.suspended) continue;
        if (p.is_active) continue;  // skip agents mid-inference
        if (!on_node.empty() && p.node_id != on_node) continue;
        if (p.last_active_ms < oldest_ms) {
            oldest_ms = p.last_active_ms;
            oldest_id = id;
        }
    }
    return oldest_id;
}

bool AgentScheduler::suspend_agent(const AgentId& agent_id) {
    auto it = placements_.find(agent_id);
    if (it == placements_.end() || it->second.suspended)
        return false;

    auto& p = it->second;

    try {
        auto node = registry_.get_node(p.node_id);
        HttpClient cli(node.url);
        cli.set_bearer_token(node.api_key);
        auto resp = cli.post("/api/node/suspend-slot",
                             nlohmann::json{{"slot_id", p.slot_id}});
        if (resp.ok()) {
            auto j = nlohmann::json::parse(resp.body);
            p.kv_cache_node_path = j.value("kv_cache_path", std::string{});
        }
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: suspend failed for agent {}: {}", agent_id, e.what());
    }

    p.suspended = true;
    MM_INFO("AgentScheduler: suspended agent {} on node {} (cache={})",
            agent_id, p.node_id, p.kv_cache_node_path);
    return true;
}

std::optional<SlotId> AgentScheduler::restore_agent_on_node(
    const AgentPlacement& placement,
    const AgentConfig& cfg,
    const NodeId& node_id)
{
    try {
        auto node = registry_.get_node(node_id);
        HttpClient cli(node.url);
        cli.set_bearer_token(node.api_key);
        const std::string model_filename = canonical_model_filename(cfg.model_path);
        std::string model_ref = model_filename;
        if (is_local_node(node.url)) {
            const std::string local_path =
                resolve_model_path_for_metadata(cfg.model_path, models_dir_);
            if (!local_path.empty()) model_ref = local_path;
        }
        for (int attempt = 0; attempt < 3; ++attempt) {
            nlohmann::json body = {
                {"model_path",    model_ref},
                {"settings",      cfg.llama_settings},
                {"kv_cache_path", placement.kv_cache_node_path},
                {"agent_id",      cfg.id}
            };

            auto resp = cli.post("/api/node/restore-slot", body);
            if (resp.ok()) {
                auto j = nlohmann::json::parse(resp.body);
                return j.value("slot_id", std::string{});
            }

            if (response_indicates_max_slots(resp.body)) {
                if (attempt == 0 && evict_slots_on_node(node_id, cfg.id, 1)) {
                    MM_INFO("AgentScheduler: restore-slot hit capacity pressure on node {}; evicted one slot and retrying",
                            node_id);
                    continue;
                }
                if (attempt == 1 && evict_slots_on_node(node_id, cfg.id, -1)) {
                    MM_INFO("AgentScheduler: restore-slot still constrained on node {}; evicted remaining slots and retrying",
                            node_id);
                    continue;
                }
            }

            std::string preview = resp.body;
            if (preview.size() > 300) preview = preview.substr(0, 300) + "...";
            last_error_ = "restore-slot failed on node " + node_id +
                        " (HTTP " + std::to_string(resp.status) + "): " + preview;
            return std::nullopt;
        }
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: restore failed on node {}: {}", node_id, e.what());
        last_error_ = "restore-slot exception on node " + node_id + ": " + e.what();
    }
    return std::nullopt;
}

std::optional<SlotId> AgentScheduler::load_agent_on_node(
    const AgentConfig& cfg,
    const NodeId& node_id)
{
    try {
        auto node = registry_.get_node(node_id);
        HttpClient cli(node.url);
        cli.set_bearer_token(node.api_key);

        if (cfg.llama_settings.n_gpu_layers != 0 && !node.metrics.gpu_backend_available) {
            MM_WARN("AgentScheduler: agent {} requesting GPU offload but node {} "
                    "lacks CUDA backend — will fall back to CPU",
                    cfg.id, node_id);
        }

        const std::string model_filename = canonical_model_filename(cfg.model_path);
        // For local nodes, pass full local path when available; for remote, pass filename.
        std::string model_ref = model_filename;
        if (is_local_node(node.url)) {
            const std::string local_path =
                resolve_model_path_for_metadata(cfg.model_path, models_dir_);
            if (!local_path.empty()) model_ref = local_path;
        }

        // Warn about large ctx_size before sending the request.
        if (cfg.llama_settings.ctx_size > 131072) {
            MM_WARN("AgentScheduler: agent {} has extremely large ctx_size={}; "
                    "consider reducing", cfg.id, cfg.llama_settings.ctx_size);
        } else if (cfg.llama_settings.ctx_size > 65536) {
            MM_WARN("AgentScheduler: agent {} has very large ctx_size={}; "
                    "load may fail or be slow", cfg.id, cfg.llama_settings.ctx_size);
        }

        bool attempted_pull = false;
        for (int attempt = 0; attempt < 3; ++attempt) {
            nlohmann::json body = {
                {"model_path", model_ref},
                {"settings",   cfg.llama_settings},
                {"agent_id",   cfg.id}
            };

            auto resp = cli.post("/api/node/load-model", body);
            if (resp.ok()) {
                auto j = nlohmann::json::parse(resp.body);
                auto slot_id = j.value("slot_id", std::string{});
                if (!slot_id.empty()) {
                    // Check if the node had to reduce ctx_size.
                    int effective_ctx = j.value("effective_ctx_size", 0);
                    if (effective_ctx > 0 && effective_ctx < cfg.llama_settings.ctx_size) {
                        MM_WARN("AgentScheduler: agent {}: ctx_size was auto-reduced "
                                "from {} to {} on node {}",
                                cfg.id, cfg.llama_settings.ctx_size, effective_ctx, node_id);
                    }
                    MM_INFO("AgentScheduler: loaded agent {} on node {} (slot={})",
                            cfg.id, node_id, slot_id);
                    return slot_id;
                }
                last_error_ = "load-model returned success but empty slot_id on node " + node_id;
                return std::nullopt;
            }

            if (response_indicates_max_slots(resp.body)) {
                if (attempt == 0 && evict_slots_on_node(node_id, cfg.id, 1)) {
                    MM_INFO("AgentScheduler: load-model hit capacity pressure on node {}; evicted one slot and retrying",
                            node_id);
                    continue;
                }
                if (attempt == 1 && evict_slots_on_node(node_id, cfg.id, -1)) {
                    MM_INFO("AgentScheduler: load-model still constrained on node {}; evicted remaining slots and retrying",
                            node_id);
                    continue;
                }
            }

            // If the node reports a missing model, ask node to pull from control and retry.
            if (!attempted_pull) {
                const std::string lower = util::to_lower(resp.body);
                if (resp.status == 404 ||
                    lower.find("model file not found") != std::string::npos ||
                    lower.find("model missing locally") != std::string::npos ||
                    lower.find("no such file") != std::string::npos) {
                    auto pull = cli.post("/api/node/models/pull",
                                         nlohmann::json{{"model_filename", model_filename},
                                                        {"force", false}});
                    attempted_pull = true;
                    if (pull.ok()) {
                        MM_INFO("AgentScheduler: node {} pulled model {}; retrying load",
                                node_id, model_filename);
                        continue;
                    }
                    MM_WARN("AgentScheduler: node {} failed to pull model {} (HTTP {})",
                            node_id, model_filename, pull.status);
                }
            }

            std::string preview = resp.body;
            if (preview.size() > 300) preview = preview.substr(0, 300) + "...";
            last_error_ = "load-model failed on node " + node_id +
                        " (HTTP " + std::to_string(resp.status) + "): " + preview;
            return std::nullopt;
        }
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: load failed on node {}: {}", node_id, e.what());
        last_error_ = "load-model exception on node " + node_id + ": " + e.what();
    }
    return std::nullopt;
}

bool AgentScheduler::response_indicates_max_slots(const std::string& body) {
    std::string lower = util::to_lower(body);
    return lower.find("max slots reached") != std::string::npos
        || lower.find("max active slots reached") != std::string::npos
        || lower.find("no available ports") != std::string::npos
        || lower.find("out of memory") != std::string::npos
        || lower.find("insufficient memory") != std::string::npos
        || lower.find("cuda out of memory") != std::string::npos;
}

bool AgentScheduler::evict_one_slot_on_node(const NodeId& node_id,
                                            const AgentId& preserve_agent) {
    return evict_slots_on_node(node_id, preserve_agent, 1);
}

bool AgentScheduler::evict_slots_on_node(const NodeId& node_id,
                                         const AgentId& preserve_agent,
                                         int max_to_evict) {
    int evicted = 0;
    auto can_evict_more = [&]() -> bool {
        return max_to_evict < 0 || evicted < max_to_evict;
    };

    // First choice: repeatedly suspend known idle placements on this node.
    while (can_evict_more()) {
        auto lru = find_lru_idle_agent(node_id);
        if (!lru || *lru == preserve_agent) break;

        auto pit = placements_.find(*lru);
        if (pit == placements_.end()) break;
        if (pit->second.is_active) break;

        if (!suspend_agent(*lru)) break;
        ++evicted;
    }
    if (!can_evict_more()) {
        return evicted > 0;
    }

    // Fallback: unload node-reported slots directly. This handles
    // orphaned/stale slots that are no longer tracked in placements_.
    try {
        auto node = registry_.get_node(node_id);
        HttpClient cli(node.url);
        cli.set_bearer_token(node.api_key);

        std::vector<SlotInfo> slots = node.slots;
        auto status = cli.get("/api/node/status");
        if (status.ok()) {
            try {
                auto j = nlohmann::json::parse(status.body);
                if (j.contains("slots")) {
                    slots = j["slots"].get<std::vector<SlotInfo>>();
                }
            } catch (const std::exception& e) {
                MM_WARN("AgentScheduler: failed to parse /api/node/status while evicting on {}: {}",
                        node_id, e.what());
            }
        }

        std::unordered_set<SlotId> protected_slots;
        for (const auto& [agent_id, p] : placements_) {
            if (p.node_id != node_id) continue;
            if (p.is_active || (!preserve_agent.empty() && agent_id == preserve_agent)) {
                if (!p.slot_id.empty()) protected_slots.insert(p.slot_id);
            }
        }

        std::vector<SlotInfo> candidates;
        candidates.reserve(slots.size());
        for (const auto& s : slots) {
            if (s.id.empty()) continue;
            if (s.state == SlotState::Suspended || s.state == SlotState::Empty) continue;
            if (!preserve_agent.empty() && s.assigned_agent == preserve_agent) continue;
            if (protected_slots.find(s.id) != protected_slots.end()) continue;
            candidates.push_back(s);
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const SlotInfo& a, const SlotInfo& b) {
                      const int64_t a_ts = a.last_active_ms > 0 ? a.last_active_ms : 0;
                      const int64_t b_ts = b.last_active_ms > 0 ? b.last_active_ms : 0;
                      return a_ts < b_ts;
                  });

        for (const auto& candidate : candidates) {
            if (!can_evict_more()) break;

            auto unload = cli.post("/api/node/unload-model",
                                   nlohmann::json{{"slot_id", candidate.id}});
            if (!unload.ok()) {
                MM_WARN("AgentScheduler: failed to unload slot {} on node {} during eviction (HTTP {})",
                        candidate.id, node_id, unload.status);
                continue;
            }

            for (auto it = placements_.begin(); it != placements_.end();) {
                const auto& p = it->second;
                if (p.node_id == node_id && p.slot_id == candidate.id) {
                    MM_INFO("AgentScheduler: removed stale placement {} -> {}/{} after direct unload",
                            it->first, node_id, candidate.id);
                    it = placements_.erase(it);
                } else {
                    ++it;
                }
            }

            MM_INFO("AgentScheduler: directly unloaded slot {} on node {} to free capacity",
                    candidate.id, node_id);
            ++evicted;
        }

        if (evicted > 0) {
            MM_INFO("AgentScheduler: evicted {} slot(s) on node {} (preserve_agent='{}')",
                    evicted, node_id, preserve_agent);
        }
        return evicted > 0;
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: eviction fallback failed on node {}: {}",
                node_id, e.what());
        return evicted > 0;
    }
}

} // namespace mm
