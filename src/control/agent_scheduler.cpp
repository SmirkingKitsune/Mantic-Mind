#include "control/agent_scheduler.hpp"

#include "soma/routing.hpp"

#include "common/http_client.hpp"
#include "common/inference_sizing.hpp"
#include "common/logger.hpp"
#include "common/pairing.hpp"
#include "common/util.hpp"
#include "control/model_registry.hpp"
#include "control/node_registry.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mm {

namespace {

std::optional<std::string> transfer_model_to_node(const NodeInfo& node,
                                                  const std::string& model_ref,
                                                  bool pin,
                                                  bool force,
                                                  std::string* error,
                                                  const std::string& cache_id = {}) {
    namespace fs = std::filesystem;
    const std::string model_id = cache_id.empty()
        ? util::model_id_from_ref(model_ref)
        : cache_id;

    struct FileEntry {
        std::string absolute_path;
        std::string relative_path;
        int64_t size = 0;
    };
    std::vector<FileEntry> files;
    int64_t total_size = 0;
    std::error_code ec;

    const fs::path root(model_ref);
    if (fs::is_regular_file(root, ec)) {
        const auto size = static_cast<int64_t>(fs::file_size(root, ec));
        files.push_back({root.string(), root.filename().string(), size});
        total_size += size;
    } else if (fs::is_directory(root, ec)) {
        for (auto it = fs::recursive_directory_iterator(root, ec);
             !ec && it != fs::recursive_directory_iterator();
             it.increment(ec)) {
            std::error_code file_ec;
            if (!it->is_regular_file(file_ec)) continue;
            const fs::path relative = fs::relative(it->path(), root, file_ec);
            const auto size = static_cast<int64_t>(it->file_size(file_ec));
            files.push_back({it->path().string(), relative.generic_string(), size});
            total_size += size;
        }
    } else {
        if (error) {
            *error = "model path is neither a file nor a directory: " + model_ref;
        }
        return std::nullopt;
    }

    if (files.empty()) {
        if (error) *error = "no files found for model: " + model_ref;
        return std::nullopt;
    }

    HttpClient client(node.url);
    client.set_bearer_token(node.api_key);
    client.set_timeouts(10, 3600, 3600);

    if (!force) {
        const auto query = client.get("/api/node/models/local?id=" + model_id);
        if (query.ok()) {
            try {
                const auto body = nlohmann::json::parse(query.body);
                if (body.value("present", false) &&
                    body.value("size_bytes", static_cast<int64_t>(-1)) == total_size) {
                    const std::string load_path =
                        body.value("load_path", std::string{});
                    if (!load_path.empty()) return load_path;
                }
            } catch (...) {
            }
        }
    }

    std::string load_path;
    for (const auto& file : files) {
        const std::vector<std::pair<std::string, std::string>> headers = {
            {"X-MM-Model-Id", model_id},
            {"X-MM-Rel-Path", file.relative_path},
            {"X-MM-Size", std::to_string(file.size)},
            {"X-MM-Pin", pin ? "true" : "false"},
        };
        const auto response = client.post_file(
            "/api/node/models/receive", file.absolute_path, headers);
        if (!response.ok()) {
            if (error) {
                std::string preview = response.body;
                if (preview.size() > 200) preview.resize(200);
                *error = "receive failed for " + file.relative_path + " (HTTP "
                       + std::to_string(response.status) + "): " + preview;
            }
            return std::nullopt;
        }
        try {
            const auto body = nlohmann::json::parse(response.body);
            load_path = body.value("load_path", load_path);
            if (files.size() == 1) {
                load_path = body.value("stored_path", load_path);
            }
        } catch (...) {
        }
    }

    if (load_path.empty()) {
        if (error) *error = "node did not report a load path after transfer";
        return std::nullopt;
    }
    return load_path;
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
                               const std::string& models_dir,
                               const std::string& engine_id) {
    const auto& runtime = cfg.runtime_settings;
    const nlohmann::json identity = {
        // The RESOLVED engine, passed IN. The fingerprint decides whether an
        // existing placement still describes this agent, so an agent whose
        // routing changed from fallback to Soma must not keep a slot running the
        // other engine — with a literal here, it would. Threaded through rather
        // than looked up again, so it cannot disagree with the decision the
        // caller already acted on.
        {"backend", engine_id},
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

/// `model_ref` is WHERE the weights are; `cfg.model_path` is what the agent calls
/// them. They are not the same string and conflating them is defect D7.
///
/// An admitted model lives at the registry's `model_dir` — since the container
/// directory carries its quantization, `containers/<name>-q4_g-q6_g-g128`, and
/// the agent asks for `<name>`. The node resolves whatever it is handed against
/// its OWN models_dir, so handing it the agent's name means handing it a
/// directory that does not exist. It used to work because containers were
/// written to `containers/<name>` and the two strings coincided; the coincidence
/// was the only thing making it work, and it is gone.
///
/// A model with no registry record passes its own path through unchanged, which
/// is the fallback's GGUF path and must keep working.
std::optional<PreparedModel> prepare_model_for_node(const NodeInfo& node,
                                                    const AgentConfig& cfg,
                                                    const std::string& model_ref,
                                                    const std::string& models_dir,
                                                    bool pin,
                                                    bool force,
                                                    std::string* error) {
    PreparedModel prepared;
    prepared.model_path = model_ref;
    prepared.mmproj_path = cfg.vision_settings.enabled
        ? cfg.vision_settings.mmproj_path
        : std::string{};

    if (const auto model_path =
            util::resolve_existing_local_model_path(model_ref, models_dir)) {
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

AgentScheduler::BackendRouting AgentScheduler::resolve_backend(
    const AgentConfig& cfg, const soma::AdmissionRecord& record) {
    if (!is_llama_backend(cfg.inference_backend)) {
        // API-backed (and unsupported legacy) agents own no node slot at all.
        // This is a different question from which local engine to use.
        return {{}, "inference_backend '" + cfg.inference_backend + "' is not node-local"};
    }

    soma::AgentBackendConfig rc;
    if (cfg.backend_override == "soma") rc.override = soma::BackendOverride::Soma;
    else if (cfg.backend_override == "fallback") rc.override = soma::BackendOverride::Fallback;
    rc.arch_hash = record.arch_hash;

    const auto decision = soma::select_backend(rc, record);
    return {decision.choice == soma::BackendChoice::Soma ? "soma" : "llama-cpp",
            decision.explain()};
}

void AgentScheduler::set_placement_audit(PlacementAudit audit) {
    audit_ = std::move(audit);
}

void AgentScheduler::flush_audit(std::vector<PendingAudit>& pending) const {
    for (const auto& e : pending) {
        if (e.placed) {
            if (audit_.placed)
                audit_.placed(
                    e.agent_id, e.node_id, e.slot_id, e.backend, e.backend_reason, e.footprint);
        } else if (audit_.released) {
            audit_.released(e.agent_id);
        }
    }
    pending.clear();
}

void AgentScheduler::set_model_registry(const ControlModelRegistry* registry) {
    models_ = registry;
}

void AgentScheduler::set_engine_config_gate(EngineConfigReadyFn ready) {
    engine_config_ready_ = std::move(ready);
    engine_config_required_ = static_cast<bool>(engine_config_ready_);
}

std::string AgentScheduler::model_cache_id(const AgentConfig& cfg) const {
    // The SAME derivation prepare_model_for_node() uses to name the transfer, so
    // "does this node hold it" is asked in the node's own vocabulary. Two
    // derivations would drift, and the failure would be silent: every node would
    // look like it holds nothing and every placement would be charged for a
    // transfer that never happens.
    const auto local = util::resolve_existing_local_model_path(model_location(cfg), models_dir_);
    if (!local) return {};
    return manifest_cache_id(*local);
}

ResourceFootprint AgentScheduler::footprint_for_node(const AgentConfig& cfg,
                                                     const ResourceFootprint& base,
                                                     const NodeInfo& node) const {
    ResourceFootprint out = base;

    // Disk is charged only where a transfer would actually happen.
    //
    // This is why `nodes_with_capacity_for()` exists: a container that the
    // target already holds costs it nothing, and one it lacks costs the whole
    // size, so the demand genuinely differs per node and a single footprint
    // cannot say it. Before this, `disk_mb` was always 0 — the axis was enforced
    // but only ever against the 8 GiB headroom, never against a model's real
    // demand (roadmap D65).
    const std::string cache_id = model_cache_id(cfg);
    if (cache_id.empty()) {
        // Control holds no local bytes, so nothing will be transferred and no
        // disk will be consumed by this placement. Same reasoning as
        // prepare_model_for_node(), which passes such a ref through untouched.
        return out;
    }

    const bool resident =
        std::find(node.local_model_ids.begin(), node.local_model_ids.end(), cache_id) !=
        node.local_model_ids.end();
    if (resident) return out;

    const std::int64_t bytes = measure_model_bytes(model_location(cfg), models_dir_);
    constexpr std::int64_t kMib = 1024 * 1024;
    out.disk_mb += bytes / kMib;
    return out;
}

ResourceFootprint AgentScheduler::soma_footprint(const AgentConfig& cfg) const {
    ResourceFootprint out;

    // vram_mb stays 0, and that is the fix rather than an omission. Soma v1 is
    // CPU-only; evaluate_fit() short-circuits to Native on a zero VRAM ask, so a
    // GPU-less node is now a valid host instead of an automatic rejection.

    const std::int64_t model_bytes = measure_model_bytes(model_location(cfg), models_dir_);
    std::int64_t routed_bytes = 0;
    if (models_ != nullptr) {
        if (const auto admitted = models_->resolve(cfg.model_path)) {
            routed_bytes = admitted->total_routed_bytes;
        }
    }

    // The RESIDENT half: everything that is not a streamed expert has to be in
    // RAM for the whole session, whatever the host decides its expert cache
    // should be. That makes it the part control can state without knowing the
    // target's budget — which is exactly what `plan_for_host()` refuses to
    // guess, since the verdict is a property of (model, quantization, host).
    //
    // A LOWER BOUND, deliberately. It excludes the KV cache (context-dependent)
    // and the expert cache (sized from the node's free RAM at load). Under-
    // charging is the safe direction here: the alternative is inventing the
    // node's cache policy on control and rejecting hosts that would have
    // worked, and the node re-derives the real plan before it loads anything.
    const std::int64_t resident_bytes = std::max<std::int64_t>(0, model_bytes - routed_bytes);
    constexpr std::int64_t kMib = 1024 * 1024;
    out.ram_mb = resident_bytes / kMib;

    // disk_mb stays 0, for the reason it always has: the container's residency
    // is not ADDITIONAL to what the target already holds, and NodeInfo still
    // cannot say whether it holds it. Charging it would reject every node that
    // already has the model — the ones that are cheapest to place on. Closing
    // that needs the node to report local model residency first (roadmap D65).
    return out;
}

std::string AgentScheduler::model_location(const AgentConfig& cfg) const {
    // The registry is the AUTHORITY on where an admitted model's bytes are, and
    // until this existed nothing asked it. resolve_backend_for() has always
    // looked the record up — it took `arch_hash` and `verdict` off it and threw
    // `model_dir` away, which is the whole of defect D7.
    //
    // Absolute, and deliberately so: the node resolves what it is handed against
    // its own models_dir, and an absolute path that exists short-circuits that
    // lookup. When control and the node are different machines the path will not
    // exist locally on control either, and prepare_model_for_node falls through
    // to passing it along — the same behaviour an unadmitted model gets.
    if (models_ != nullptr) {
        if (const auto admitted = models_->resolve(cfg.model_path)) {
            if (!admitted->model_dir.empty()) return admitted->model_dir;
        }
    }
    return cfg.model_path;
}

AgentScheduler::BackendRouting
AgentScheduler::resolve_backend_for(const AgentConfig& cfg) const {
    soma::AdmissionRecord record;   // absent by default, which routes to fallback
    if (models_ != nullptr) {
        if (const auto admitted = models_->resolve(cfg.model_path)) {
            record.present = true;
            record.arch_hash = admitted->arch_hash;
            // Two verdict enums, one meaning. ModelVerdict is the registry's
            // storage type and soma::Verdict is the engine's; they are mapped
            // explicitly rather than static_cast so adding a value to one cannot
            // silently reinterpret rows written under the other.
            switch (admitted->verdict) {
                case ModelVerdict::Stream:       record.verdict = soma::Verdict::Stream; break;
                case ModelVerdict::Hybrid:       record.verdict = soma::Verdict::Hybrid; break;
                case ModelVerdict::ResidentOnly: record.verdict = soma::Verdict::ResidentOnly; break;
                case ModelVerdict::Reject:       record.verdict = soma::Verdict::Reject; break;
            }
        }
    }
    return resolve_backend(cfg, record);
}

std::optional<ScheduleResult> AgentScheduler::ensure_agent_running(
    const AgentConfig& cfg) {
    // The cluster has no engine policy yet. Refusing here rather than placing
    // on whatever a node happened to have is the whole point of the master
    // configurator: an unconfigured cluster serving from an accidental engine
    // set is exactly the state nobody could see before.
    //
    // The message names the fix because this is the first thing a fresh install
    // hits, and "no available nodes" would send the operator to look at nodes
    // that are all healthy.
    if (engine_config_required_ && !engine_config_ready_()) {
        release_agent(cfg.id);
        set_failure(PlacementFailure::EngineConfigMissing,
                    "cluster engine configuration required: no primary engine has been set. "
                    "Configure it in the control TUI's Engines tab, with `engines setup` in "
                    "CLI mode, or via PUT /v1/cluster/engines/config");
        return std::nullopt;
    }

    const auto routing = resolve_backend_for(cfg);
    if (routing.engine_id.empty()) {
        // Release a prior local placement before reporting the routing result.
        release_agent(cfg.id);
        set_failure(PlacementFailure::NoLocalBackend,
                    "agent has no node-local backend: " + routing.reason);
        return std::nullopt;
    }

    const std::string desired_fingerprint =
        engine_fingerprint(cfg, models_dir_, routing.engine_id);

    // Audit events are QUEUED here and delivered by `audit_guard`'s destructor,
    // which runs after `schedule_lock` is released because destruction order is
    // the reverse of construction. Two reasons it is worth the eight lines:
    // `schedule_mutex_` already serializes multi-GB model transfers and a
    // synchronous SQLite insert has no business lengthening it, and a callback
    // invoked while holding a lock is the exact shape that killed the node in
    // D56. This function has a dozen return points, so a guard is also the only
    // way to get the flush right on all of them.
    std::vector<PendingAudit> pending_audit;

    struct AuditGuard {
        const AgentScheduler& self;
        std::vector<PendingAudit>& pending;

        ~AuditGuard() {
            // An audit row must never break the placement it describes — the
            // same contract record_placement() states for itself. A throwing
            // destructor would be std::terminate.
            try {
                self.flush_audit(pending);
            } catch (...) {
            }
        }
    } audit_guard{*this, pending_audit};

    std::lock_guard schedule_lock(schedule_mutex_);
    set_last_error({});

    auto existing = find_placement_copy(cfg.id);
    if (existing && existing->engine_fingerprint != desired_fingerprint) {
        MM_INFO("AgentScheduler: engine identity changed for agent {}; "
                "releasing stale placement", cfg.id);
        erase_placement_entry(cfg.id);
        pending_audit.push_back({false, cfg.id, {}, {}, {}, {}, {}});
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
            pending_audit.push_back({false, cfg.id, {}, {}, {}, {}, {}});
            detach_placement_best_effort(*existing, cfg.id,
                                         "node no longer reports attached ready slot");
            existing.reset();
        }
    }

    // Three axes, and now shaped by the ENGINE that will serve it.
    //
    // llama.cpp's estimate folds weights, KV and overhead into a single number
    // that behaves like VRAM, so that is where it goes and `ram_mb` stays zero —
    // the offload path inside evaluate_fit() is what trades RAM against it, as
    // before.
    //
    // Soma is the opposite shape: no VRAM at all in v1, and a resident half that
    // is genuinely RAM. Both used to go through the llama estimator, which meant
    // a Soma agent was charged VRAM it would never touch — and since
    // evaluate_fit() will not offload against a host below
    // `min_gpu_for_offload_mb`, a GPU-less node was rejected outright for a
    // CPU-only engine (D62).
    //
    // `disk_mb` stays zero for BOTH, and that part is unchanged: a container's
    // residency is not additional to what the target already holds, and NodeInfo
    // still cannot say whether it holds it (D65).
    ResourceFootprint needed;
    if (routing.engine_id == "soma") {
        needed = soma_footprint(cfg);
    } else {
        needed.vram_mb =
            estimate_inference_vram_mb(cfg.model_path, cfg.runtime_settings, models_dir_) +
            projector_file_mb(cfg, models_dir_);
    }

    if (existing && existing->suspended) {
        AgentPlacement placement = *existing;
        auto publish_restored = [&](const NodeId& node_id, const SlotId& slot_id) {
            if (node_id != placement.node_id) {
                try {
                    const auto old_node = registry_.get_node(placement.node_id);
                    HttpClient old_client(old_node.url);
                    old_client.set_bearer_token(old_node.api_key);
                    static_cast<void>(old_client.post(
                        "/api/node/detach-agent",
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
            pending_audit.push_back(
                {true, cfg.id, node_id, slot_id, routing.engine_id, routing.reason, needed});
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

        for (const auto& node : registry_.nodes_with_capacity(needed)) {
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
        pending_audit.push_back(
            {true, cfg.id, node_id, slot_id, routing.engine_id, routing.reason, needed});
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
            // Match the agent's RESOLVED engine, not "llama-cpp". Sharing a slot
            // running a different engine would attach the agent to a process
            // that cannot serve it.
            if (slot.state != SlotState::Ready || slot.backend != routing.engine_id) continue;
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

    // Per-node demand: `needed` is what the model costs anywhere, and
    // footprint_for_node() adds the container's disk only to nodes that would
    // have to fetch it.
    const auto demand = [&](const NodeInfo& node) { return footprint_for_node(cfg, needed, node); };
    for (const auto& node : registry_.nodes_with_capacity_for(demand)) {
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

    // `last_failure()`, not `last_error().empty()`. The old check asked "did a
    // load attempt already record something" by testing a STRING for emptiness,
    // which is the same defect one layer down from the one this fixes: the
    // control flow depended on prose. A code answers it directly, and cannot be
    // broken by rewording a message.
    if (last_failure() == PlacementFailure::None) {
        // THE distinction D64 exists for. Both of these used to read "no
        // capacity: no connected node could load this model", and they call for
        // opposite actions: one means fix the cluster, the other means wait or
        // add hardware. `available_nodes()` applies the connected + conforming
        // filter, so an empty list means nothing was ELIGIBLE — every node is
        // offline, unconfigured, or not conforming to the engine policy.
        if (registry_.available_nodes().empty()) {
            set_failure(PlacementFailure::NoEligibleNode,
                        "no eligible node: none is connected and conforming to the "
                        "cluster engine configuration");
        } else {
            set_failure(PlacementFailure::NoCapacity,
                        "no capacity: eligible nodes are connected, but none could "
                        "load this model");
        }
    }
    return std::nullopt;
}

void AgentScheduler::release_agent(const AgentId& agent_id) {
    std::optional<AgentPlacement> placement;
    {
        std::lock_guard schedule_lock(schedule_mutex_);
        placement = find_placement_copy(agent_id);
        if (!placement) return;
        erase_placement_entry(agent_id);
    }

    // Outside the lock, like every other audit call — see PlacementAudit.
    if (audit_.released) audit_.released(agent_id);
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

const char* to_string(PlacementFailure failure) noexcept {
    switch (failure) {
    case PlacementFailure::None:
        return "none";
    case PlacementFailure::EngineConfigMissing:
        return "engine_config_missing";
    case PlacementFailure::NoLocalBackend:
        return "no_local_backend";
    case PlacementFailure::NoEligibleNode:
        return "no_eligible_node";
    case PlacementFailure::NoCapacity:
        return "no_capacity";
    case PlacementFailure::ModelTransferFailed:
        return "model_transfer_failed";
    case PlacementFailure::NodeRejected:
        return "node_rejected";
    case PlacementFailure::NodeUnreachable:
        return "node_unreachable";
    case PlacementFailure::NodeProtocolError:
        return "node_protocol_error";
    }
    return "unknown";
}

bool placement_failure_retryable(PlacementFailure failure) noexcept {
    switch (failure) {
    case PlacementFailure::EngineConfigMissing: // an operator must set a policy
    case PlacementFailure::NoLocalBackend:      // the agent owns no slot by design
        return false;
    default:
        return true;
    }
}

void AgentScheduler::set_failure(PlacementFailure failure, const std::string& error) {
    std::lock_guard state_lock(state_mutex_);
    last_failure_ = failure;
    last_error_ = error;
}

void AgentScheduler::set_last_error(const std::string& error) {
    std::lock_guard state_lock(state_mutex_);
    last_error_ = error;
    // Cleared together. A stale code beside a fresh empty message would say a
    // placement failed when the caller had just recorded that it did not.
    if (error.empty()) last_failure_ = PlacementFailure::None;
}

PlacementFailure AgentScheduler::last_failure() const {
    std::lock_guard state_lock(state_mutex_);
    return last_failure_;
}

void AgentScheduler::detach_placement_best_effort(
    const AgentPlacement& placement,
    const AgentId& agent_id,
    const std::string& reason) {
    try {
        const auto node = registry_.get_node(placement.node_id);
        HttpClient client(node.url);
        client.set_bearer_token(node.api_key);
        const auto response = client.post(
            "/api/node/detach-agent",
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
        std::lock_guard schedule_lock(schedule_mutex_);
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
            const auto node = registry_.get_node(detach.node_id);
            HttpClient client(node.url);
            client.set_bearer_token(node.api_key);
            const auto response = client.post(
                "/api/node/detach-agent",
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
        const auto node = registry_.get_node(placement->node_id);
        HttpClient client(node.url);
        client.set_bearer_token(node.api_key);
        const auto response = client.post(
            "/api/node/suspend-slot",
            nlohmann::json{{"slot_id", placement->slot_id}});
        if (!response.ok()) {
            std::string preview = response.body;
            if (preview.size() > 300) preview = preview.substr(0, 300) + "...";
            MM_WARN("AgentScheduler: suspend failed for agent {} on node {} "
                    "(HTTP {}): {}", agent_id, placement->node_id,
                    response.status, preview);
            return false;
        }
        kv_cache_path = nlohmann::json::parse(response.body)
                            .value("kv_cache_path", std::string{});
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
        const auto node = registry_.get_node(node_id);
        HttpClient client(node.url);
        client.set_bearer_token(node.api_key);
        const bool pin = !cfg.preferred_node_id.empty()
            && cfg.preferred_node_id == node_id;
        std::string prepare_error;
        const auto prepared = prepare_model_for_node(
            node, cfg, model_location(cfg), models_dir_, pin, false, &prepare_error);
        if (!prepared) {
            set_failure(PlacementFailure::ModelTransferFailed,
                        "failed to prepare model for restore on node " + node_id + ": " +
                            prepare_error);
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
                // The RESOLVED engine id. Hardcoding "llama-cpp" here is what
                // made a Soma agent unplaceable regardless of its verdict.
                {"backend", resolve_backend_for(cfg).engine_id},
                {"agent_id", cfg.id},
            };
            if (!prepared->model_id.empty()) {
                body["model_id"] = prepared->model_id;
            }
            if (!prepared->mmproj_model_id.empty())
                body["mmproj_model_id"] = prepared->mmproj_model_id;
            if (!prepared->model_id.empty() || !prepared->mmproj_model_id.empty())
                body["pin"] = pin;

            const auto response = client.post("/api/node/restore-slot", body);
            if (response.ok()) {
                const auto slot_id = nlohmann::json::parse(response.body)
                                         .value("slot_id", std::string{});
                if (!slot_id.empty()) return slot_id;
                set_failure(PlacementFailure::NodeProtocolError,
                            "restore-slot returned an empty slot_id on node " + node_id);
                return std::nullopt;
            }

            if (response_indicates_capacity_pressure(response.body)) {
                if (attempt == 0 && evict_slots_on_node(node_id, cfg.id, 1)) continue;
                if (attempt == 1 && evict_slots_on_node(node_id, cfg.id, -1)) continue;
            }

            std::string preview = response.body;
            if (preview.size() > 300) preview = preview.substr(0, 300) + "...";
            set_failure(PlacementFailure::NodeRejected,
                        "restore-slot failed on node " + node_id + " (HTTP " +
                            std::to_string(response.status) + "): " + preview);
            return std::nullopt;
        }
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: restore failed on node {}: {}", node_id, e.what());
        set_failure(PlacementFailure::NodeUnreachable,
                    "restore-slot exception on node " + node_id + ": " + e.what());
    }
    return std::nullopt;
}

std::optional<SlotId> AgentScheduler::load_agent_on_node(
    const AgentConfig& cfg,
    const NodeId& node_id) {
    try {
        const auto node = registry_.get_node(node_id);
        HttpClient client(node.url);
        client.set_bearer_token(node.api_key);
        const bool pin = !cfg.preferred_node_id.empty()
            && cfg.preferred_node_id == node_id;

        std::string prepare_error;
        auto prepared = prepare_model_for_node(
            node, cfg, model_location(cfg), models_dir_, pin, false, &prepare_error);
        if (!prepared) {
            set_failure(PlacementFailure::ModelTransferFailed,
                        "failed to prepare model for node " + node_id + ": " + prepare_error);
            return std::nullopt;
        }

        bool retried_transfer = false;
        for (int attempt = 0; attempt < 3; ++attempt) {
            nlohmann::json body = {
                {"model_path", prepared->model_path},
                {"mmproj_path", prepared->mmproj_path},
                {"vision_enabled", cfg.vision_settings.enabled},
                {"runtime_settings", cfg.runtime_settings},
                // The RESOLVED engine id. Hardcoding "llama-cpp" here is what
                // made a Soma agent unplaceable regardless of its verdict.
                {"backend", resolve_backend_for(cfg).engine_id},
                {"agent_id", cfg.id},
            };
            if (!prepared->model_id.empty()) {
                body["model_id"] = prepared->model_id;
            }
            if (!prepared->mmproj_model_id.empty())
                body["mmproj_model_id"] = prepared->mmproj_model_id;
            if (!prepared->model_id.empty() || !prepared->mmproj_model_id.empty())
                body["pin"] = pin;

            const auto response = client.post("/api/node/load-model", body);
            if (response.ok()) {
                const auto slot_id = nlohmann::json::parse(response.body)
                                         .value("slot_id", std::string{});
                if (!slot_id.empty()) {
                    MM_INFO("AgentScheduler: loaded agent {} on node {} (slot={})",
                            cfg.id, node_id, slot_id);
                    return slot_id;
                }
                set_failure(PlacementFailure::NodeProtocolError,
                            "load-model returned an empty slot_id on node " + node_id);
                return std::nullopt;
            }

            if (response_indicates_capacity_pressure(response.body)) {
                if (attempt == 0 && evict_slots_on_node(node_id, cfg.id, 1)) continue;
                if (attempt == 1 && evict_slots_on_node(node_id, cfg.id, -1)) continue;
            }

            const std::string lower_body = util::to_lower(response.body);
            const bool model_missing = lower_body.find("model not found on node")
                    != std::string::npos
                || lower_body.find("projector not found on node")
                    != std::string::npos;
            if ((!prepared->model_id.empty() || !prepared->mmproj_model_id.empty())
                && !retried_transfer && model_missing) {
                prepare_error.clear();
                auto refreshed = prepare_model_for_node(
                    node, cfg, model_location(cfg), models_dir_, pin, true, &prepare_error);
                if (refreshed) {
                    prepared = std::move(refreshed);
                    retried_transfer = true;
                    continue;
                }
            }

            std::string preview = response.body;
            if (preview.size() > 300) preview = preview.substr(0, 300) + "...";
            set_failure(PlacementFailure::NodeRejected,
                        "load-model failed on node " + node_id + " (HTTP " +
                            std::to_string(response.status) + "): " + preview);
            return std::nullopt;
        }
    } catch (const std::exception& e) {
        MM_WARN("AgentScheduler: load failed on node {}: {}", node_id, e.what());
        set_failure(PlacementFailure::NodeUnreachable,
                    "load-model exception on node " + node_id + ": " + e.what());
    }
    return std::nullopt;
}

bool AgentScheduler::response_indicates_capacity_pressure(const std::string& body) {
    // A STRUCTURED code, with the six-phrase substring match retained only as a
    // fallback for engines that have not been updated.
    //
    // What it replaces: matching "max slots reached", "out of memory" and four
    // other English strings against the node's error body. Every engine had to
    // reproduce those literals verbatim to earn an evict-and-retry, so a new one
    // silently got a hard failure instead — and translating or rewording any of
    // those messages would have broken eviction with nothing to catch it.
    //
    // Both engines emit {"error":{"code":"capacity_pressure"}} now. The fallback
    // covers a stale NODE, not a stale llama-server — llama.cpp's prose never
    // reaches here unlabelled, because the node translates it to a code at the
    // boundary (`engine_error_code_for`). The only way a body arrives without a
    // code is a node old enough to predate that, on the far side of a rolling
    // upgrade. Deleting the fallback is therefore safe exactly when no such node
    // can still be in the cluster, which is a deployment fact, not a code one.
    // Two shapes, because two producers. `soma serve` and the node's own
    // proxy emit {"error":{"code":...}}; the node's slot handlers keep `error` as
    // a human string and carry the code alongside it, so existing clients are not
    // broken by the addition.
    try {
        const auto j = nlohmann::json::parse(body);
        std::string code;
        if (j.contains("error") && j["error"].is_object()) {
            const auto& e = j["error"];
            if (e.contains("code") && e["code"].is_string()) code = e["code"];
        } else if (j.contains("code") && j["code"].is_string()) {
            code = j["code"];
        }
        if (!code.empty()) {
            // A structured code is AUTHORITATIVE: one that says something else
            // means the engine has decided this is not capacity, and reading its
            // prose for a contradicting hint would undo the point of asking.
            return code == "capacity_pressure";
        }
    } catch (const std::exception&) {
        // Not JSON — fall through to the legacy match below.
    }

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
        HttpClient client(node.url);
        client.set_bearer_token(node.api_key);

        std::vector<SlotInfo> slots = node.slots;
        const auto status = client.get("/api/node/status");
        if (status.ok()) {
            try {
                const auto body = nlohmann::json::parse(status.body);
                if (body.contains("slots")) {
                    slots = body["slots"].get<std::vector<SlotInfo>>();
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
            const auto response = client.post(
                "/api/node/unload-model",
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
