#include "control/agent_scheduler.hpp"

#include "common/inference_sizing.hpp"
#include "common/logger.hpp"
#include "common/pairing.hpp"
#include "common/util.hpp"
#include "control/node_registry.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mm {

namespace {

class ScopeExit {
public:
    explicit ScopeExit(std::function<void()> fn) : fn_(std::move(fn)) {}
    ~ScopeExit() { if (fn_) fn_(); }
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
private:
    std::function<void()> fn_;
};

std::optional<std::string> transfer_model_to_node(NodeOperations& node,
                                                  const std::string& model_ref,
                                                  bool pin,
                                                  bool force,
                                                  std::string* error,
                                                  const std::string& cache_id = {}) {
    const std::string model_id = cache_id.empty()
        ? util::model_id_from_ref(model_ref)
        : cache_id;
    const auto prepared = node.prepare_model(
        model_ref, model_id, pin, force, error);
    if (!prepared) return std::nullopt;
    return prepared->load_path;
}

std::string file_manifest_identity(const std::string& ref) {
    namespace fs = std::filesystem;
    const fs::path requested(ref);
    std::error_code ec;
    fs::path resolved = fs::weakly_canonical(requested, ec);
    if (ec) {
        ec.clear();
        resolved = fs::absolute(requested, ec);
    }
    if (ec) resolved = requested;
    resolved = resolved.lexically_normal();

    ec.clear();
    if (fs::is_regular_file(resolved, ec)) {
        std::error_code size_ec;
        std::error_code time_ec;
        const auto size = fs::file_size(resolved, size_ec);
        const auto modified = fs::last_write_time(resolved, time_ec);
        return "file:" + resolved.generic_string()
            + "\nsize:" + (size_ec ? std::string{"?"} : std::to_string(size))
            + "\nmtime:" + (time_ec ? std::string{"?"}
                                             : std::to_string(modified.time_since_epoch().count()));
    }

    ec.clear();
    if (fs::is_directory(resolved, ec)) {
        std::vector<std::string> entries;
        for (fs::recursive_directory_iterator it(resolved, ec), end;
             !ec && it != end; it.increment(ec)) {
            std::error_code file_ec;
            if (!it->is_regular_file(file_ec)) continue;

            std::error_code relative_ec;
            std::error_code size_ec;
            std::error_code time_ec;
            const auto relative = fs::relative(it->path(), resolved, relative_ec);
            const auto size = it->file_size(size_ec);
            const auto modified = it->last_write_time(time_ec);
            entries.push_back(
                (relative_ec ? it->path().filename().generic_string()
                             : relative.generic_string())
                + "\t" + (size_ec ? std::string{"?"} : std::to_string(size))
                + "\t" + (time_ec ? std::string{"?"}
                                             : std::to_string(modified.time_since_epoch().count())));
        }
        std::sort(entries.begin(), entries.end());
        std::string identity = "directory:" + resolved.generic_string();
        for (const auto& entry : entries) identity += "\n" + entry;
        return identity;
    }

    // A missing reference may intentionally be a path that exists only on the
    // target node. Keep its spelling stable without pretending it is local.
    return "reference:" + requested.lexically_normal().generic_string();
}

std::string manifest_cache_id(const std::string& path) {
    const std::string digest = pairing::hmac_sha256_hex(
        "mantic-model-cache-v2", file_manifest_identity(path));
    return util::model_id_from_ref(path) + "-" + digest.substr(0, 24);
}

std::string resolved_manifest_identity(const std::string& ref,
                                       const std::string& models_dir) {
    if (const auto local = util::resolve_existing_local_model_path(ref, models_dir)) {
        return file_manifest_identity(*local);
    }
    return file_manifest_identity(ref);
}

std::string engine_fingerprint(const AgentConfig& cfg,
                               const std::string& models_dir) {
    const auto& runtime = cfg.runtime_settings;
    const nlohmann::json identity = {
        {"backend", "llama-cpp"},
        {"model", resolved_manifest_identity(cfg.model_path, models_dir)},
        {"vision_enabled", cfg.vision_settings.enabled},
        {"projector", cfg.vision_settings.enabled
            ? resolved_manifest_identity(cfg.vision_settings.mmproj_path, models_dir)
            : std::string{}},
        {"launch", {
            {"ctx_size", runtime.ctx_size},
            {"n_gpu_layers", runtime.n_gpu_layers},
            {"n_threads", runtime.n_threads},
            {"n_threads_http", runtime.n_threads_http},
            {"parallel", runtime.parallel},
            {"batch_size", runtime.batch_size},
            {"ubatch_size", runtime.ubatch_size},
            {"flash_attn", runtime.flash_attn},
            {"extra_args", runtime.extra_args},
        }},
    };
    return pairing::hmac_sha256_hex(
        "mantic-engine-placement-v1", identity.dump());
}

int64_t projector_file_mb(const AgentConfig& cfg,
                          const std::string& models_dir) {
    if (!cfg.vision_settings.enabled || cfg.vision_settings.mmproj_path.empty()) {
        return 0;
    }
    const auto local = util::resolve_existing_local_model_path(
        cfg.vision_settings.mmproj_path, models_dir);
    if (!local) return 0;
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(*local, ec);
    if (ec) return 0;
    constexpr uint64_t kMib = 1024ULL * 1024ULL;
    return static_cast<int64_t>((bytes + kMib - 1) / kMib);
}

struct PreparedModel {
    std::string model_path;
    std::string mmproj_path;
    std::string model_id;
    std::string mmproj_model_id;
};

std::optional<PreparedModel> prepare_model_for_node(NodeOperations& node,
                                                    const AgentConfig& cfg,
                                                    const std::string& models_dir,
                                                    bool pin,
                                                    bool force,
                                                    std::string* error) {
    PreparedModel prepared;
    prepared.model_path = cfg.model_path;
    prepared.mmproj_path = cfg.vision_settings.enabled
        ? cfg.vision_settings.mmproj_path
        : std::string{};

    if (const auto model_path =
            util::resolve_existing_local_model_path(cfg.model_path, models_dir)) {
        const std::string cache_id = manifest_cache_id(*model_path);
        auto local = transfer_model_to_node(
            node, *model_path, pin, force, error, cache_id);
        if (!local) return std::nullopt;
        prepared.model_path = *local;
        prepared.model_id = cache_id;
    }

    if (cfg.vision_settings.enabled &&
        !cfg.vision_settings.mmproj_path.empty()) {
        if (const auto projector_path = util::resolve_existing_local_model_path(
                cfg.vision_settings.mmproj_path, models_dir)) {
            const std::string cache_id = manifest_cache_id(*projector_path);
            auto local = transfer_model_to_node(
                node, *projector_path, pin, force, error, cache_id);
            if (!local) return std::nullopt;
            prepared.mmproj_path = *local;
            prepared.mmproj_model_id = cache_id;
        }
    }

    return prepared;
}

bool same_model_reference(const std::string& lhs, const std::string& rhs) {
    namespace fs = std::filesystem;
    const auto left = util::to_lower(fs::path(lhs).lexically_normal().generic_string());
    const auto right = util::to_lower(fs::path(rhs).lexically_normal().generic_string());
    if (left == right) return true;
    return util::to_lower(fs::path(lhs).filename().string())
        == util::to_lower(fs::path(rhs).filename().string());
}

} // namespace

AgentScheduler::AgentScheduler(NodeRegistry& registry, std::string models_dir)
    : registry_(registry)
    , models_dir_(std::move(models_dir)) {}

std::optional<ScheduleResult> AgentScheduler::ensure_agent_running(
    const AgentConfig& cfg) {
    if (!is_llama_backend(cfg.inference_backend)) {
        // API-backed (and unsupported legacy) agents do not own node slots.
        // Release a prior local placement before reporting the routing result.
        release_agent(cfg.id);
        set_last_error("unsupported local inference backend '" + cfg.inference_backend
                       + "'; this branch supports llama-cpp only");
        return std::nullopt;
    }

    const std::string desired_fingerprint = engine_fingerprint(cfg, models_dir_);
    begin_agent_schedule(cfg.id);
    ScopeExit schedule_done([this, id = cfg.id] { end_agent_schedule(id); });
    set_last_error({});

    auto existing = find_placement_copy(cfg.id);
    if (existing && existing->engine_fingerprint != desired_fingerprint) {
        MM_INFO("AgentScheduler: engine identity changed for agent {}; "
                "releasing stale placement", cfg.id);
        erase_placement_entry(cfg.id);
        detach_placement_best_effort(*existing, cfg.id, "engine identity changed");
        existing.reset();
    }

    if (existing && !existing->suspended) {
        const auto nodes = registry_.list_nodes();
        const auto node_it = std::find_if(
            nodes.begin(), nodes.end(), [&](const NodeInfo& node) {
                return node.id == existing->node_id;
            });

        // A status snapshot taken before this placement cannot disprove it.
        // Once a newer snapshot exists, require the node to report the same
        // attached slot so direct lifecycle calls cannot leave stale routing.
        if (node_it != nodes.end() && node_it->connected
            && node_it->slot_snapshot_at_ms <= existing->placed_at_ms) {
            mutate_placement(cfg.id, [](AgentPlacement& placement) {
                placement.last_active_ms = util::now_ms();
            });
            return ScheduleResult{existing->node_id, existing->slot_id};
        }

        const SlotInfo* reported_slot = nullptr;
        if (node_it != nodes.end() && node_it->connected) {
            const auto slot_it = std::find_if(
                node_it->slots.begin(), node_it->slots.end(),
                [&](const SlotInfo& slot) { return slot.id == existing->slot_id; });
            if (slot_it != node_it->slots.end()) reported_slot = &*slot_it;
        }
        const bool attached = reported_slot
            && (reported_slot->assigned_agent == cfg.id
                || std::find(reported_slot->agent_ids.begin(),
                             reported_slot->agent_ids.end(), cfg.id)
                       != reported_slot->agent_ids.end());

        if (attached && reported_slot->state == SlotState::Ready) {
            mutate_placement(cfg.id, [](AgentPlacement& placement) {
                placement.last_active_ms = util::now_ms();
            });
            return ScheduleResult{existing->node_id, existing->slot_id};
        }
        if (attached && reported_slot->state == SlotState::Suspended) {
            existing->suspended = true;
            existing->kv_cache_node_path = reported_slot->kv_cache_path;
            store_placement(*existing);
        } else {
            MM_WARN("AgentScheduler: discarding stale placement for agent {} "
                    "on node {} slot {}", cfg.id, existing->node_id,
                    existing->slot_id);
            erase_placement_entry(cfg.id);
            detach_placement_best_effort(*existing, cfg.id,
                                         "node no longer reports attached ready slot");
            existing.reset();
        }
    }

    const int64_t vram_needed = estimate_inference_vram_mb(
        cfg.model_path, cfg.runtime_settings, models_dir_)
        + projector_file_mb(cfg, models_dir_);

    if (existing && existing->suspended) {
        AgentPlacement placement = *existing;
        auto publish_restored = [&](const NodeId& node_id, const SlotId& slot_id) {
            if (node_id != placement.node_id) {
                try {
                    auto old_client = registry_.operations(placement.node_id);
                    static_cast<void>(old_client->detach_agent(
                        nlohmann::json{{"slot_id", placement.slot_id},
                                       {"agent_id", cfg.id}}));
                } catch (const std::exception& e) {
                    MM_WARN("AgentScheduler: restored agent {} on node {} but could "
                            "not remove its suspended record from node {}: {}",
                            cfg.id, node_id, placement.node_id, e.what());
                }
            }
            placement.node_id = node_id;
            placement.slot_id = slot_id;
            placement.suspended = false;
            placement.placed_at_ms = util::now_ms();
            placement.last_active_ms = placement.placed_at_ms;
            placement.engine_fingerprint = desired_fingerprint;
            placement.kv_cache_node_path.clear();
            store_placement(placement);
            return ScheduleResult{node_id, slot_id};
        };

        auto slot_id = restore_agent_on_node(placement, cfg, placement.node_id);
        if (slot_id) {
            MM_INFO("AgentScheduler: restored agent {} on original node {}",
                    cfg.id, placement.node_id);
            return publish_restored(placement.node_id, *slot_id);
        }

        for (const auto& candidate_id : lru_idle_agents(placement.node_id)) {
            if (candidate_id == cfg.id || !suspend_agent(candidate_id)) continue;
            slot_id = restore_agent_on_node(placement, cfg, placement.node_id);
            if (slot_id) {
                return publish_restored(placement.node_id, *slot_id);
            }
        }

        for (const auto& node : registry_.nodes_with_available_vram(vram_needed)) {
            if (node.id == placement.node_id) continue;
            slot_id = restore_agent_on_node(placement, cfg, node.id);
            if (slot_id) return publish_restored(node.id, *slot_id);
        }
    }

    auto publish_new = [&](const NodeId& node_id, const SlotId& slot_id) {
        AgentPlacement placement;
        placement.agent_id = cfg.id;
        placement.node_id = node_id;
        placement.slot_id = slot_id;
        placement.placed_at_ms = util::now_ms();
        placement.last_active_ms = placement.placed_at_ms;
        placement.engine_fingerprint = desired_fingerprint;
        store_placement(placement);
        return ScheduleResult{node_id, slot_id};
    };

    std::unordered_set<NodeId> attempted_nodes;
    auto try_load = [&](const NodeId& node_id) -> std::optional<SlotId> {
        if (node_id.empty() || !attempted_nodes.insert(node_id).second) {
            return std::nullopt;
        }
        auto slot_id = load_agent_on_node(cfg, node_id);
        if (!slot_id) {
            MM_WARN("AgentScheduler: node {} could not run agent {}; "
                    "trying another candidate", node_id, cfg.id);
        }
        return slot_id;
    };

    if (!cfg.preferred_node_id.empty()) {
        if (const auto slot_id = try_load(cfg.preferred_node_id)) {
            return publish_new(cfg.preferred_node_id, *slot_id);
        }
    }

    struct SharedCandidate {
        NodeId node_id;
        size_t attached_agents = 0;
        int64_t last_active_ms = 0;
    };
    std::vector<SharedCandidate> shared_candidates;
    for (const auto& node : registry_.available_nodes()) {
        if (attempted_nodes.count(node.id)) continue;
        std::optional<SharedCandidate> best;
        for (const auto& slot : node.slots) {
            if (slot.state != SlotState::Ready || slot.backend != "llama-cpp") continue;
            if (!same_model_reference(slot.model_path, cfg.model_path)) continue;
            if (slot.vision_enabled != cfg.vision_settings.enabled) continue;
            if (cfg.vision_settings.enabled) {
                const std::string requested = util::to_lower(
                    std::filesystem::path(cfg.vision_settings.mmproj_path)
                        .filename().string());
                const std::string loaded = util::to_lower(
                    std::filesystem::path(slot.mmproj_path).filename().string());
                if (requested.empty() || requested != loaded) continue;
            }
            SharedCandidate candidate{
                node.id, slot.agent_ids.size(), slot.last_active_ms};
            if (!best ||
                std::tie(candidate.attached_agents, candidate.last_active_ms)
                    < std::tie(best->attached_agents, best->last_active_ms)) {
                best = candidate;
            }
        }
        if (best) shared_candidates.push_back(*best);
    }
    std::sort(shared_candidates.begin(), shared_candidates.end(),
              [](const SharedCandidate& lhs, const SharedCandidate& rhs) {
                  return std::tie(lhs.attached_agents, lhs.last_active_ms)
                       < std::tie(rhs.attached_agents, rhs.last_active_ms);
              });
    for (const auto& candidate : shared_candidates) {
        if (const auto slot_id = try_load(candidate.node_id)) {
            MM_INFO("AgentScheduler: agent {} joined a compatible llama.cpp "
                    "engine for {} on node {}", cfg.id, cfg.model_path,
                    candidate.node_id);
            return publish_new(candidate.node_id, *slot_id);
        }
    }

    for (const auto& node : registry_.nodes_with_available_vram(vram_needed)) {
        if (const auto slot_id = try_load(node.id)) {
            return publish_new(node.id, *slot_id);
        }
    }

    for (const auto& candidate_id : lru_idle_agents()) {
        if (candidate_id == cfg.id) continue;
        const auto candidate = find_placement_copy(candidate_id);
        if (!candidate || candidate->suspended || candidate->is_active
            || !suspend_agent(candidate_id)) {
            continue;
        }
        if (const auto slot_id = load_agent_on_node(cfg, candidate->node_id)) {
            return publish_new(candidate->node_id, *slot_id);
        }
    }

    MM_WARN("AgentScheduler: no capacity for agent {} (model={})",
            cfg.id, cfg.model_path);
    if (last_error().empty()) {
        set_last_error("no capacity: no connected node could load this model");
    }
    return std::nullopt;
}

void AgentScheduler::release_agent(const AgentId& agent_id) {
    begin_agent_schedule(agent_id);
    ScopeExit schedule_done([this, id = agent_id] { end_agent_schedule(id); });
    std::optional<AgentPlacement> placement;
    {
        placement = find_placement_copy(agent_id);
        if (!placement) return;
        erase_placement_entry(agent_id);
    }

    detach_placement_best_effort(*placement, agent_id, "placement released");
    MM_INFO("AgentScheduler: released agent {}", agent_id);
}

void AgentScheduler::mark_agent_idle(const AgentId& agent_id) {
    mutate_placement(agent_id, [](AgentPlacement& placement) {
        placement.is_active = false;
        placement.last_active_ms = util::now_ms();
    });
}

void AgentScheduler::mark_agent_active(const AgentId& agent_id) {
    mutate_placement(agent_id, [](AgentPlacement& placement) {
        placement.is_active = true;
        placement.last_active_ms = util::now_ms();
    });
}

std::optional<AgentPlacement> AgentScheduler::get_placement(
    const AgentId& agent_id) const {
    return find_placement_copy(agent_id);
}

std::vector<AgentPlacement> AgentScheduler::list_placements() const {
    std::lock_guard state_lock(state_mutex_);
    std::vector<AgentPlacement> result;
    result.reserve(placements_.size());
    for (const auto& [_, placement] : placements_) result.push_back(placement);
    return result;
}

std::string AgentScheduler::last_error() const {
    std::lock_guard state_lock(state_mutex_);
    return last_error_;
}

std::optional<AgentPlacement> AgentScheduler::find_placement_copy(
    const AgentId& id) const {
    std::lock_guard state_lock(state_mutex_);
    const auto it = placements_.find(id);
    if (it == placements_.end()) return std::nullopt;
    return it->second;
}

void AgentScheduler::store_placement(const AgentPlacement& placement) {
    std::lock_guard state_lock(state_mutex_);
    placements_[placement.agent_id] = placement;
}

bool AgentScheduler::erase_placement_entry(const AgentId& id) {
    std::lock_guard state_lock(state_mutex_);
    return placements_.erase(id) > 0;
}

void AgentScheduler::set_last_error(const std::string& error) {
    std::lock_guard state_lock(state_mutex_);
    last_error_ = error;
}

void AgentScheduler::begin_agent_schedule(const AgentId& id) {
    std::unique_lock lock(coordination_mutex_);
    coordination_cv_.wait(lock, [&] { return scheduling_agents_.count(id) == 0; });
    scheduling_agents_.insert(id);
}

void AgentScheduler::end_agent_schedule(const AgentId& id) {
    {
        std::lock_guard lock(coordination_mutex_);
        scheduling_agents_.erase(id);
    }
    coordination_cv_.notify_all();
}

void AgentScheduler::detach_placement_best_effort(
    const AgentPlacement& placement,
    const AgentId& agent_id,
    const std::string& reason) {
    try {
        auto client = registry_.operations(placement.node_id);
        const auto response = client->detach_agent(
            nlohmann::json{{"slot_id", placement.slot_id},
                           {"agent_id", agent_id}});
        if (!response.ok()) {
            MM_WARN("AgentScheduler: node {} rejected detach for agent {} "
                    "({}; HTTP {})", placement.node_id, agent_id, reason,
                    response.status);
        }
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: failed to detach slot for agent {} ({}): {}",
                agent_id, reason, e.what());
    }
}

void AgentScheduler::housekeeping(const std::vector<AgentConfig>& active_agents) {
    struct PendingDetach {
        AgentId agent_id;
        NodeId node_id;
        SlotId slot_id;
    };
    std::vector<PendingDetach> detaches;

    {
        std::lock_guard state_lock(state_mutex_);

        std::unordered_set<AgentId> active_ids;
        active_ids.reserve(active_agents.size());
        for (const auto& agent : active_agents) active_ids.insert(agent.id);

        for (auto it = placements_.begin(); it != placements_.end();) {
            if (active_ids.count(it->first)) {
                ++it;
                continue;
            }
            detaches.push_back({it->first, it->second.node_id, it->second.slot_id});
            MM_INFO("AgentScheduler: housekeeping removed placement for deleted agent {}",
                    it->first);
            it = placements_.erase(it);
        }
    }

    for (const auto& detach : detaches) {
        try {
            auto client = registry_.operations(detach.node_id);
            const auto response = client->detach_agent(
                nlohmann::json{{"slot_id", detach.slot_id},
                               {"agent_id", detach.agent_id}});
            if (!response.ok()) {
                MM_WARN("AgentScheduler: housekeeping detach failed for agent {} "
                        "on node {} (HTTP {})", detach.agent_id, detach.node_id,
                        response.status);
            }
        } catch (const std::exception& e) {
            MM_WARN("AgentScheduler: housekeeping failed to detach deleted agent {}: {}",
                    detach.agent_id, e.what());
        }
    }
}

std::vector<AgentId> AgentScheduler::lru_idle_agents(const NodeId& on_node) const {
    std::lock_guard state_lock(state_mutex_);
    std::vector<std::pair<AgentId, int64_t>> candidates;
    for (const auto& [id, placement] : placements_) {
        if (placement.suspended || placement.is_active) continue;
        if (!on_node.empty() && placement.node_id != on_node) continue;
        candidates.emplace_back(id, placement.last_active_ms);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.second, lhs.first) < std::tie(rhs.second, rhs.first);
    });

    std::vector<AgentId> result;
    result.reserve(candidates.size());
    for (auto& [id, _] : candidates) result.push_back(std::move(id));
    return result;
}

bool AgentScheduler::suspend_agent(const AgentId& agent_id) {
    const auto placement = find_placement_copy(agent_id);
    if (!placement || placement->suspended) return false;

    std::vector<AgentId> cohort{agent_id};
    {
        std::lock_guard state_lock(state_mutex_);
        for (const auto& [id, candidate] : placements_) {
            if (id == agent_id || candidate.suspended) continue;
            if (candidate.node_id != placement->node_id
                || candidate.slot_id != placement->slot_id) {
                continue;
            }
            if (candidate.is_active) {
                MM_INFO("AgentScheduler: not suspending agent {}; slot {} is "
                        "shared with active agent {}", agent_id,
                        placement->slot_id, id);
                return false;
            }
            cohort.push_back(id);
        }
    }

    std::string kv_cache_path;
    try {
        auto client = registry_.operations(placement->node_id);
        const auto response = client->suspend_slot(
            nlohmann::json{{"slot_id", placement->slot_id}});
        if (!response.ok()) {
            std::string preview = response.raw_body;
            if (preview.size() > 300) preview = preview.substr(0, 300) + "...";
            MM_WARN("AgentScheduler: suspend failed for agent {} on node {} "
                    "(HTTP {}): {}", agent_id, placement->node_id,
                    response.status, preview);
            return false;
        }
        kv_cache_path = response.body.value("kv_cache_path", std::string{});
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: suspend failed for agent {}: {}", agent_id, e.what());
        return false;
    }

    bool updated = false;
    for (const auto& id : cohort) {
        const bool changed = mutate_placement(id, [&](AgentPlacement& candidate) {
            candidate.suspended = true;
            candidate.kv_cache_node_path = kv_cache_path;
        });
        if (id == agent_id) updated = changed;
    }
    if (!updated) return false;

    MM_INFO("AgentScheduler: suspended agent {} on node {} (cache={})",
            agent_id, placement->node_id, kv_cache_path);
    return true;
}

std::optional<SlotId> AgentScheduler::restore_agent_on_node(
    const AgentPlacement& placement,
    const AgentConfig& cfg,
    const NodeId& node_id) {
    try {
        auto client = registry_.operations(node_id);
        const bool pin = !cfg.preferred_node_id.empty()
            && cfg.preferred_node_id == node_id;
        std::string prepare_error;
        const auto prepared = prepare_model_for_node(
            *client, cfg, models_dir_, pin, false, &prepare_error);
        if (!prepared) {
            set_last_error("failed to prepare model for restore on node " + node_id
                           + ": " + prepare_error);
            return std::nullopt;
        }

        for (int attempt = 0; attempt < 3; ++attempt) {
            nlohmann::json body = {
                {"model_path", prepared->model_path},
                {"mmproj_path", prepared->mmproj_path},
                {"vision_enabled", cfg.vision_settings.enabled},
                {"runtime_settings", cfg.runtime_settings},
                {"kv_cache_path", node_id == placement.node_id
                    ? placement.kv_cache_node_path : std::string{}},
                {"backend", "llama-cpp"},
                {"agent_id", cfg.id},
            };
            if (!prepared->model_id.empty()) {
                body["model_id"] = prepared->model_id;
            }
            if (!prepared->mmproj_model_id.empty())
                body["mmproj_model_id"] = prepared->mmproj_model_id;
            if (!prepared->model_id.empty() || !prepared->mmproj_model_id.empty())
                body["pin"] = pin;

            const auto response = client->restore_slot(body);
            if (response.ok()) {
                const auto slot_id = response.body.value("slot_id", std::string{});
                if (!slot_id.empty()) return slot_id;
                set_last_error("restore-slot returned an empty slot_id on node " + node_id);
                return std::nullopt;
            }

            if (response_indicates_capacity_pressure(response.raw_body)) {
                if (attempt == 0 && evict_slots_on_node(node_id, cfg.id, 1)) continue;
                if (attempt == 1 && evict_slots_on_node(node_id, cfg.id, -1)) continue;
            }

            std::string preview = response.raw_body;
            if (preview.size() > 300) preview = preview.substr(0, 300) + "...";
            set_last_error("restore-slot failed on node " + node_id + " (HTTP "
                           + std::to_string(response.status) + "): " + preview);
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
    const NodeId& node_id) {
    try {
        auto client = registry_.operations(node_id);
        const bool pin = !cfg.preferred_node_id.empty()
            && cfg.preferred_node_id == node_id;

        std::string prepare_error;
        auto prepared = prepare_model_for_node(
            *client, cfg, models_dir_, pin, false, &prepare_error);
        if (!prepared) {
            set_last_error("failed to prepare model for node " + node_id
                           + ": " + prepare_error);
            return std::nullopt;
        }

        bool retried_transfer = false;
        for (int attempt = 0; attempt < 3; ++attempt) {
            nlohmann::json body = {
                {"model_path", prepared->model_path},
                {"mmproj_path", prepared->mmproj_path},
                {"vision_enabled", cfg.vision_settings.enabled},
                {"runtime_settings", cfg.runtime_settings},
                {"backend", "llama-cpp"},
                {"agent_id", cfg.id},
            };
            if (!prepared->model_id.empty()) {
                body["model_id"] = prepared->model_id;
            }
            if (!prepared->mmproj_model_id.empty())
                body["mmproj_model_id"] = prepared->mmproj_model_id;
            if (!prepared->model_id.empty() || !prepared->mmproj_model_id.empty())
                body["pin"] = pin;

            const auto response = client->load_model(body);
            if (response.ok()) {
                const auto slot_id = response.body.value("slot_id", std::string{});
                if (!slot_id.empty()) {
                    MM_INFO("AgentScheduler: loaded agent {} on node {} (slot={})",
                            cfg.id, node_id, slot_id);
                    return slot_id;
                }
                set_last_error("load-model returned an empty slot_id on node " + node_id);
                return std::nullopt;
            }

            if (response_indicates_capacity_pressure(response.raw_body)) {
                if (attempt == 0 && evict_slots_on_node(node_id, cfg.id, 1)) continue;
                if (attempt == 1 && evict_slots_on_node(node_id, cfg.id, -1)) continue;
            }

            const std::string lower_body = util::to_lower(response.raw_body);
            const bool model_missing = lower_body.find("model not found on node")
                    != std::string::npos
                || lower_body.find("projector not found on node")
                    != std::string::npos;
            if ((!prepared->model_id.empty() || !prepared->mmproj_model_id.empty())
                && !retried_transfer && model_missing) {
                prepare_error.clear();
                auto refreshed = prepare_model_for_node(
                    *client, cfg, models_dir_, pin, true, &prepare_error);
                if (refreshed) {
                    prepared = std::move(refreshed);
                    retried_transfer = true;
                    continue;
                }
            }

            std::string preview = response.raw_body;
            if (preview.size() > 300) preview = preview.substr(0, 300) + "...";
            set_last_error("load-model failed on node " + node_id + " (HTTP "
                           + std::to_string(response.status) + "): " + preview);
            return std::nullopt;
        }
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: load failed on node {}: {}", node_id, e.what());
        set_last_error("load-model exception on node " + node_id + ": " + e.what());
    }
    return std::nullopt;
}

bool AgentScheduler::response_indicates_capacity_pressure(const std::string& body) {
    const std::string lower = util::to_lower(body);
    return lower.find("max slots reached") != std::string::npos
        || lower.find("max active slots reached") != std::string::npos
        || lower.find("no available ports") != std::string::npos
        || lower.find("out of memory") != std::string::npos
        || lower.find("insufficient memory") != std::string::npos
        || lower.find("insufficient vram") != std::string::npos;
}

bool AgentScheduler::evict_slots_on_node(const NodeId& node_id,
                                         const AgentId& preserve_agent,
                                         int max_to_evict) {
    int evicted = 0;
    const auto can_evict_more = [&] {
        return max_to_evict < 0 || evicted < max_to_evict;
    };

    for (const auto& candidate_id : lru_idle_agents(node_id)) {
        if (!can_evict_more()) break;
        if (candidate_id == preserve_agent) continue;
        const auto placement = find_placement_copy(candidate_id);
        if (!placement || placement->suspended || placement->is_active) continue;
        if (!suspend_agent(candidate_id)) continue;
        ++evicted;
    }
    if (!can_evict_more()) return evicted > 0;

    try {
        const auto node = registry_.get_node(node_id);
        auto client = registry_.operations(node_id);

        std::vector<SlotInfo> slots = node.slots;
        const auto status = client->status();
        if (status.ok()) {
            try {
                if (status.body.contains("slots")) {
                    slots = status.body["slots"].get<std::vector<SlotInfo>>();
                }
            } catch (const std::exception& e) {
                MM_WARN("AgentScheduler: failed to parse node status while evicting "
                        "on {}: {}", node_id, e.what());
            }
        }

        std::unordered_set<SlotId> protected_slots;
        {
            std::lock_guard state_lock(state_mutex_);
            for (const auto& [agent_id, placement] : placements_) {
                if (placement.node_id != node_id) continue;
                if (placement.is_active
                    || (!preserve_agent.empty() && agent_id == preserve_agent)) {
                    if (!placement.slot_id.empty()) {
                        protected_slots.insert(placement.slot_id);
                    }
                }
            }
        }

        std::vector<SlotInfo> candidates;
        for (const auto& slot : slots) {
            if (slot.id.empty() || slot.state == SlotState::Suspended
                || slot.state == SlotState::Empty) {
                continue;
            }
            if (!preserve_agent.empty()
                && (slot.assigned_agent == preserve_agent
                    || std::find(slot.agent_ids.begin(), slot.agent_ids.end(),
                                 preserve_agent) != slot.agent_ids.end())) {
                continue;
            }
            if (protected_slots.count(slot.id)) continue;
            candidates.push_back(slot);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const SlotInfo& lhs, const SlotInfo& rhs) {
                      return lhs.last_active_ms < rhs.last_active_ms;
                  });

        for (const auto& candidate : candidates) {
            if (!can_evict_more()) break;
            const auto response = client->unload_model(
                nlohmann::json{{"slot_id", candidate.id}});
            if (!response.ok()) {
                MM_WARN("AgentScheduler: failed to unload slot {} on node {} "
                        "during eviction (HTTP {})", candidate.id, node_id,
                        response.status);
                continue;
            }

            {
                std::lock_guard state_lock(state_mutex_);
                for (auto it = placements_.begin(); it != placements_.end();) {
                    if (it->second.node_id == node_id
                        && it->second.slot_id == candidate.id) {
                        it = placements_.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            ++evicted;
            MM_INFO("AgentScheduler: directly unloaded slot {} on node {} "
                    "to free capacity", candidate.id, node_id);
        }

        return evicted > 0;
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: eviction fallback failed on node {}: {}",
                node_id, e.what());
        return evicted > 0;
    }
}

} // namespace mm
