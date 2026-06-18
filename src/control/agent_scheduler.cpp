#include "control/agent_scheduler.hpp"
#include "control/node_registry.hpp"
#include "control/engine_group_planner.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <regex>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace mm {

AgentScheduler::AgentScheduler(NodeRegistry& registry,
                               std::string models_dir)
    : registry_(registry)
    , models_dir_(std::move(models_dir))
{}

// ── Main scheduling entry point ──────────────────────────────────────────────

std::optional<ScheduleResult> AgentScheduler::ensure_agent_running(
    const AgentConfig& cfg)
{
    // Scheduling decisions stay serialized end-to-end (same semantics as the
    // old single mutex), but placements_ is only touched through the
    // state_mutex_ helpers so reads and idle/active marks never block behind
    // a model load or upload happening here.
    std::lock_guard lock(schedule_mutex_);
    set_last_error({});

    // 1. Existing active placement?
    auto existing = find_placement_copy(cfg.id);
    if (existing && !existing->suspended) {
        mutate_placement(cfg.id, [](AgentPlacement& p) {
            p.last_active_ms = util::now_ms();
        });
        return ScheduleResult{existing->node_id, existing->slot_id};
    }

    // Multi-node engine group: pipeline_parallel_size > 1 means this model is
    // meant to span nodes. Plan the group from advertised node capabilities.
    // The live Ray launch is not yet implemented, so fail clearly here rather
    // than handing a multi-node config to a single node and getting a cryptic
    // engine error.
    if (cfg.vllm_settings.pipeline_parallel_size > 1) {
        EngineGroupRequest greq;
        greq.model_path = cfg.model_path;
        greq.world_size = std::max(1, cfg.vllm_settings.tensor_parallel_size) *
                          std::max(1, cfg.vllm_settings.pipeline_parallel_size);
        auto group = best_engine_group(greq, registry_.available_nodes());
        if (group) {
            MM_INFO("AgentScheduler: agent {} needs a {}-node engine group for {} "
                    "(tp={}, pp={}, backend={}); nodes=[{}]",
                    cfg.id, group->pipeline_parallel_size, cfg.model_path,
                    group->tensor_parallel_size, group->pipeline_parallel_size,
                    group->comm_backend, util::join(group->nodes, ","));
            set_last_error("engine group planned (" +
                           std::to_string(group->pipeline_parallel_size) +
                           " nodes, " + group->comm_backend +
                           ") but multi-node Ray engine launch is not yet "
                           "implemented");
        } else {
            set_last_error("no feasible multi-node engine group: need world_size " +
                           std::to_string(greq.world_size) +
                           " across Ray-capable nodes sharing a vLLM build");
        }
        return std::nullopt;
    }

    // vLLM nodes enforce their own GPU-memory budget at load time and reject
    // with an out-of-memory style error that response_indicates_max_slots()
    // detects — that is the real capacity authority. We only use the registry
    // VRAM filter as a coarse pre-screen, so a conservative zero estimate keeps
    // every connected node a candidate and lets the node make the final call.
    const int64_t vram_needed = 0;

    // 2. Suspended placement? Try to restore.
    if (existing && existing->suspended) {
        AgentPlacement placement = *existing;

        auto restored = [&](const NodeId& node_id, const SlotId& slot_id) {
            placement.node_id        = node_id;
            placement.slot_id        = slot_id;
            placement.suspended      = false;
            placement.last_active_ms = util::now_ms();
            store_placement(placement);
            return ScheduleResult{node_id, slot_id};
        };

        // Try restore on same node first.
        auto slot_id = restore_agent_on_node(placement, cfg, placement.node_id);
        if (slot_id) {
            MM_INFO("AgentScheduler: restored agent {} on original node {}",
                    cfg.id, placement.node_id);
            return restored(placement.node_id, *slot_id);
        }

        // Same node didn't work — try suspending an idle agent there first.
        auto lru = find_lru_idle_agent(placement.node_id);
        if (lru && *lru != cfg.id) {
            suspend_agent(*lru);
            slot_id = restore_agent_on_node(placement, cfg, placement.node_id);
            if (slot_id) {
                return restored(placement.node_id, *slot_id);
            }
        }

        // Try other nodes.
        for (const auto& node : registry_.available_nodes()) {
            if (node.id == placement.node_id) continue;
            slot_id = restore_agent_on_node(placement, cfg, node.id);
            if (slot_id) {
                return restored(node.id, *slot_id);
            }
        }
    }

    // 3. No placement — find a node.
    auto place = [&](const NodeId& node_id, const SlotId& slot_id) {
        AgentPlacement p;
        p.agent_id       = cfg.id;
        p.node_id        = node_id;
        p.slot_id        = slot_id;
        p.placed_at_ms   = util::now_ms();
        p.last_active_ms = p.placed_at_ms;
        store_placement(p);
        return ScheduleResult{node_id, slot_id};
    };

    // 3a. Preferred node has model stored + VRAM?
    if (!cfg.preferred_node_id.empty()) {
        auto slot_id = load_agent_on_node(cfg, cfg.preferred_node_id);
        if (slot_id) {
            return place(cfg.preferred_node_id, *slot_id);
        }
    }

    // 3a'. Shared engine: another agent already runs this model on some node —
    // join its engine instead of loading a fresh copy, preferring the least
    // loaded engine (vLLM /metrics: waiting, then running, then KV pressure).
    // The node only attaches when launch settings are actually compatible, so
    // a mismatch simply falls through to a fresh load there.
    {
        struct EngineCandidate {
            NodeId node_id;
            int    waiting  = 0;
            int    running  = 0;
            double kv_usage = 0.0;

            bool operator<(const EngineCandidate& o) const {
                return std::tie(waiting, running, kv_usage)
                     < std::tie(o.waiting, o.running, o.kv_usage);
            }
        };
        std::vector<EngineCandidate> engine_nodes;
        for (const auto& node : registry_.available_nodes()) {
            if (node.id == cfg.preferred_node_id) continue; // already tried
            std::optional<EngineCandidate> best;
            for (const auto& s : node.slots) {
                if (s.state != SlotState::Ready) continue;
                if (s.model_path != cfg.model_path) continue;
                EngineCandidate c{node.id, s.num_requests_waiting,
                                  s.num_requests_running, s.kv_cache_usage};
                if (!best || c < *best) best = c;
            }
            if (best) engine_nodes.push_back(*best);
        }
        std::sort(engine_nodes.begin(), engine_nodes.end());
        for (const auto& cand : engine_nodes) {
            auto slot_id = load_agent_on_node(cfg, cand.node_id);
            if (slot_id) {
                MM_INFO("AgentScheduler: agent {} joining existing vLLM engine "
                        "for {} on node {} (waiting={}, running={})",
                        cfg.id, cfg.model_path, cand.node_id,
                        cand.waiting, cand.running);
                return place(cand.node_id, *slot_id);
            }
        }
    }

    // 3b. Prefer a node that already has this model in its HF cache — avoids a
    // fresh multi-GB download. Only meaningful for HF repo ids; local-dir refs
    // are assumed present wherever they resolve.
    if (util::is_hf_repo_id(cfg.model_path)) {
        for (const auto& node : registry_.nodes_with_model_cached(cfg.model_path)) {
            if (node.id == cfg.preferred_node_id) continue; // already tried
            auto slot_id = load_agent_on_node(cfg, node.id);
            if (slot_id) {
                MM_INFO("AgentScheduler: placed agent {} on node {} which already "
                        "caches {}", cfg.id, node.id, cfg.model_path);
                return place(node.id, *slot_id);
            }
        }
    }

    // 3c. Any node has VRAM -> try load (the node makes the final budget call).
    auto vram_nodes = registry_.nodes_with_available_vram(vram_needed);
    for (const auto& node : vram_nodes) {
        auto slot_id = load_agent_on_node(cfg, node.id);
        if (slot_id) {
            return place(node.id, *slot_id);
        }
    }

    // 3d. No VRAM anywhere → suspend LRU idle agent, retry.
    auto lru = find_lru_idle_agent();
    if (lru && *lru != cfg.id) {
        auto lru_placement = find_placement_copy(*lru);
        if (lru_placement) {
            suspend_agent(*lru);

            // Retry on the freed node.
            auto slot_id = load_agent_on_node(cfg, lru_placement->node_id);
            if (slot_id) {
                return place(lru_placement->node_id, *slot_id);
            }
        }
    }

    // 3e. All nodes full, no idle agents — return nullopt.
    MM_WARN("AgentScheduler: no capacity for agent {} (model={})",
            cfg.id, cfg.model_path);
    if (last_error().empty()) {
        set_last_error("no capacity: no connected node could load this model");
    }
    return std::nullopt;
}

void AgentScheduler::release_agent(const AgentId& agent_id) {
    std::optional<AgentPlacement> placement;

    {
        // Serialize against scheduling so a release cannot interleave with an
        // in-flight placement decision for the same agent.
        std::lock_guard lock(schedule_mutex_);
        placement = find_placement_copy(agent_id);
        if (!placement) return;
        erase_placement_entry(agent_id);
    }

    if (placement && !placement->suspended) {
        // Detach from the slot on the node; the node unloads the engine only
        // when this was the last attached agent.
        try {
            auto node = registry_.get_node(placement->node_id);
            HttpClient cli(node.url);
            cli.set_bearer_token(node.api_key);
            cli.post("/api/node/detach-agent",
                     nlohmann::json{{"slot_id", placement->slot_id},
                                    {"agent_id", agent_id}});
        } catch (const std::exception& e) {
            MM_WARN("AgentScheduler: failed to detach slot for agent {}: {}",
                    agent_id, e.what());
        }
    }

    MM_INFO("AgentScheduler: released agent {}", agent_id);
}

void AgentScheduler::mark_agent_idle(const AgentId& agent_id) {
    mutate_placement(agent_id, [](AgentPlacement& p) {
        p.is_active      = false;
        p.last_active_ms = util::now_ms();
    });
}

void AgentScheduler::mark_agent_active(const AgentId& agent_id) {
    mutate_placement(agent_id, [](AgentPlacement& p) {
        p.is_active      = true;
        p.last_active_ms = util::now_ms();
    });
}

std::optional<AgentPlacement> AgentScheduler::get_placement(
    const AgentId& agent_id) const
{
    return find_placement_copy(agent_id);
}

std::vector<AgentPlacement> AgentScheduler::list_placements() const {
    std::lock_guard lock(state_mutex_);
    std::vector<AgentPlacement> out;
    out.reserve(placements_.size());
    for (const auto& [_, p] : placements_)
        out.push_back(p);
    return out;
}

std::string AgentScheduler::last_error() const {
    std::lock_guard lock(state_mutex_);
    return last_error_;
}

std::optional<AgentPlacement> AgentScheduler::find_placement_copy(
    const AgentId& id) const
{
    std::lock_guard lock(state_mutex_);
    auto it = placements_.find(id);
    if (it == placements_.end()) return std::nullopt;
    return it->second;
}

void AgentScheduler::store_placement(const AgentPlacement& p) {
    std::lock_guard lock(state_mutex_);
    placements_[p.agent_id] = p;
}

bool AgentScheduler::erase_placement_entry(const AgentId& id) {
    std::lock_guard lock(state_mutex_);
    return placements_.erase(id) > 0;
}

void AgentScheduler::set_last_error(const std::string& err) {
    std::lock_guard lock(state_mutex_);
    last_error_ = err;
}

void AgentScheduler::housekeeping(const std::vector<AgentConfig>& active_agents) {
    struct PendingUnload {
        AgentId agent_id;
        NodeId node_id;
        SlotId slot_id;
    };
    std::vector<PendingUnload> unloads;

    {
        // Serialize against scheduling; the map work itself is brief.
        std::lock_guard sched(schedule_mutex_);
        std::lock_guard state(state_mutex_);

        // Build set of active agent IDs.
        std::unordered_map<std::string, bool> active_ids;
        for (const auto& a : active_agents)
            active_ids[a.id] = true;

        // Remove placements for deleted agents.
        for (auto it = placements_.begin(); it != placements_.end();) {
            if (active_ids.count(it->first)) {
                ++it;
                continue;
            }

            const auto& id = it->first;
            const auto& p = it->second;
            if (!p.suspended) {
                unloads.push_back(PendingUnload{id, p.node_id, p.slot_id});
            }
            MM_INFO("AgentScheduler: housekeeping removed placement for deleted agent {}",
                    id);
            it = placements_.erase(it);
        }
    }

    for (const auto& unload : unloads) {
        try {
            auto node = registry_.get_node(unload.node_id);
            HttpClient cli(node.url);
            cli.set_bearer_token(node.api_key);
            cli.post("/api/node/detach-agent",
                     nlohmann::json{{"slot_id", unload.slot_id},
                                    {"agent_id", unload.agent_id}});
        } catch (const std::exception& e) {
            MM_WARN("AgentScheduler: housekeeping failed to detach slot for deleted agent {}: {}",
                    unload.agent_id, e.what());
        }
    }
}

// ── Private helpers ─────────────────────────────────────────────────────────

bool AgentScheduler::is_local_node(const std::string& node_url) {
    static const std::regex local_re(
        R"(https?://(127\.0\.0\.1|localhost|::1)(:\d+)?(/.*)?)",
        std::regex::icase);
    return std::regex_match(node_url, local_re);
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
    std::lock_guard lock(state_mutex_);
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
    // Called with schedule_mutex_ held; placement access goes through
    // state_mutex_ so the node HTTP call below runs without it.
    auto placement = find_placement_copy(agent_id);
    if (!placement || placement->suspended)
        return false;

    // Shared engine: suspending the slot suspends every agent placed on it,
    // and never while any of them is mid-inference.
    std::vector<AgentId> cohort{agent_id};
    {
        std::lock_guard lock(state_mutex_);
        for (const auto& [id, p] : placements_) {
            if (id == agent_id || p.suspended) continue;
            if (p.node_id != placement->node_id || p.slot_id != placement->slot_id)
                continue;
            if (p.is_active) {
                MM_INFO("AgentScheduler: not suspending agent {} — slot {} is "
                        "shared with active agent {}",
                        agent_id, placement->slot_id, id);
                return false;
            }
            cohort.push_back(id);
        }
    }

    std::string kv_cache_path;
    try {
        auto node = registry_.get_node(placement->node_id);
        HttpClient cli(node.url);
        cli.set_bearer_token(node.api_key);
        auto resp = cli.post("/api/node/suspend-slot",
                             nlohmann::json{{"slot_id", placement->slot_id}});
        if (resp.ok()) {
            auto j = nlohmann::json::parse(resp.body);
            kv_cache_path = j.value("kv_cache_path", std::string{});
        } else {
            std::string preview = resp.body;
            if (preview.size() > 300) preview = preview.substr(0, 300) + "...";
            MM_WARN("AgentScheduler: suspend failed for agent {} on node {} (HTTP {}): {}",
                    agent_id, placement->node_id, resp.status, preview);
            return false;
        }
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: suspend failed for agent {}: {}", agent_id, e.what());
        return false;
    }

    bool updated = false;
    for (const auto& id : cohort) {
        const bool ok = mutate_placement(id, [&](AgentPlacement& p) {
            p.suspended = true;
            p.kv_cache_node_path = kv_cache_path;
        });
        if (id == agent_id) updated = ok;
    }
    if (!updated) return false; // released concurrently

    if (cohort.size() > 1) {
        MM_INFO("AgentScheduler: suspended {} agents sharing slot {} on node {}",
                cohort.size(), placement->slot_id, placement->node_id);
    }
    MM_INFO("AgentScheduler: suspended agent {} on node {} (cache={})",
            agent_id, placement->node_id, kv_cache_path);
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
        for (int attempt = 0; attempt < 3; ++attempt) {
            nlohmann::json body = {
                {"model_path",    cfg.model_path},
                {"vllm_settings", cfg.vllm_settings},
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
            set_last_error("restore-slot failed on node " + node_id +
                           " (HTTP " + std::to_string(resp.status) + "): " + preview);
            return std::nullopt;
        }
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: restore failed on node {}: {}", node_id, e.what());
        set_last_error("restore-slot exception on node " + node_id + ": " + e.what());
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

        for (int attempt = 0; attempt < 3; ++attempt) {
            nlohmann::json body = {
                {"model_path", cfg.model_path},
                {"vllm_settings", cfg.vllm_settings},
                {"agent_id",   cfg.id}
            };

            auto resp = cli.post("/api/node/load-model", body);
            if (resp.ok()) {
                auto j = nlohmann::json::parse(resp.body);
                auto slot_id = j.value("slot_id", std::string{});
                if (!slot_id.empty()) {
                    MM_INFO("AgentScheduler: loaded agent {} on node {} (slot={})",
                            cfg.id, node_id, slot_id);
                    return slot_id;
                }
                set_last_error("load-model returned success but empty slot_id on node " + node_id);
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

            std::string preview = resp.body;
            if (preview.size() > 300) preview = preview.substr(0, 300) + "...";
            set_last_error("load-model failed on node " + node_id +
                           " (HTTP " + std::to_string(resp.status) + "): " + preview);
            return std::nullopt;
        }
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: load failed on node {}: {}", node_id, e.what());
        set_last_error("load-model exception on node " + node_id + ": " + e.what());
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
        || lower.find("insufficient gpu memory budget") != std::string::npos
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

        auto lru_placement = find_placement_copy(*lru);
        if (!lru_placement) break;
        if (lru_placement->is_active) break;

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
        {
            std::lock_guard state(state_mutex_);
            for (const auto& [agent_id, p] : placements_) {
                if (p.node_id != node_id) continue;
                if (p.is_active || (!preserve_agent.empty() && agent_id == preserve_agent)) {
                    if (!p.slot_id.empty()) protected_slots.insert(p.slot_id);
                }
            }
        }

        std::vector<SlotInfo> candidates;
        candidates.reserve(slots.size());
        for (const auto& s : slots) {
            if (s.id.empty()) continue;
            if (s.state == SlotState::Suspended || s.state == SlotState::Empty) continue;
            if (!preserve_agent.empty()) {
                if (s.assigned_agent == preserve_agent) continue;
                if (std::find(s.agent_ids.begin(), s.agent_ids.end(),
                              preserve_agent) != s.agent_ids.end()) continue;
            }
            if (protected_slots.find(s.id) != protected_slots.end()) continue;
            // The engine itself reports in-flight work — never yank it out
            // from under requests control may not know about.
            if (s.engine_metrics_valid &&
                (s.num_requests_running > 0 || s.num_requests_waiting > 0)) continue;
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

            {
                std::lock_guard state(state_mutex_);
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
