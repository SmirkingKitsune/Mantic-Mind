#include "control/placement_scheduler.hpp"

#include "soma/routing.hpp"

#include "common/http_client.hpp"
#include "common/inference_sizing.hpp"
#include "common/logger.hpp"
#include "common/pairing.hpp"
#include "common/util.hpp"
#include "control/model_registry.hpp"
#include "control/node_registry.hpp"
#include "control/engine_group_planner.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mm {

namespace {

/// Is this file an ADMISSION artifact rather than part of the model?
///
/// A container directory holds two different kinds of thing, and until now
/// nothing distinguished them. The model is what a node needs in order to serve:
/// the index, the shards, the dense half, the tokenizer, config.json. The
/// admission artifacts are what CONTROL needs in order to judge the model, and
/// control writes them into the same directory after conversion —
/// `<container>/conformance/` holds the tiny-random oracle, and
/// `<container>/conformance/reference.bin` a teacher-forced bf16 pass over the
/// real checkpoint (`model_registry.cpp`), with `*-build/` as their scratch.
///
/// `transfer_model_to_node()` walks the directory recursively and ships every
/// regular file, so every node was receiving both. They are `positions × vocab`
/// float arrays — for a 151936-token vocabulary over the gate's 500-position
/// minimum that is ~300 MB each, sent to every node, for a judgement that already
/// happened on control and that no node ever re-runs. `soma conform` is the only
/// reader and it only ever runs during admission.
///
/// The same rule governs the cache identity below, and that half matters just as
/// much: identity is a hash of per-file (path, size, mtime), so an admission
/// write into the container changed the id of a model every node already held and
/// forced the whole thing to transfer again.
bool is_admission_artifact(const std::filesystem::path& relative) {
    const auto first = relative.begin();
    if (first == relative.end()) return false;
    const auto head = first->generic_string();
    // `conformance/` is the artifact; `*-build/` is where it is assembled before
    // the rename. Matching on the leading component rather than anywhere in the
    // path keeps a model file called `conformance.bin` shippable.
    return head == "conformance" || (head.size() > 6 && head.ends_with("-build"));
}

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
            if (!file_ec && is_admission_artifact(relative)) continue;
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
            // The identity covers what is TRANSFERRED, so that writing an
            // admission artifact into a container does not invalidate a copy
            // every node already holds. Without this, the conformance stage
            // re-transferred the whole model to the whole fleet.
            if (!relative_ec && is_admission_artifact(relative)) continue;
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

std::string engine_fingerprint(const ServedModel& cfg,
                               const std::string& models_dir,
                               const std::string& engine_id,
                               const std::optional<VllmEngineConfig>& vllm = std::nullopt) {
    const auto& runtime = cfg.runtime_settings;
    nlohmann::json identity = {
        // The RESOLVED engine, passed IN. The fingerprint decides whether an
        // existing placement still describes this model, so a model whose
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
    if (vllm) identity["vllm"] = *vllm;
    return pairing::hmac_sha256_hex(
        "mantic-engine-placement-v1", identity.dump());
}

int64_t projector_file_mb(const ServedModel& cfg,
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

/// `model_ref` is WHERE the weights are; `cfg.model_path` is what the model is
/// called. They are not the same string and conflating them is defect D7.
///
/// An admitted model lives at the registry's `model_dir` — since the container
/// directory carries its quantization, `containers/<name>-q4_g-q6_g-g128`, and
/// the served model asks for `<name>`. The node resolves whatever it is handed
/// against its OWN models_dir, so handing it the model's name means handing it a
/// directory that does not exist. It used to work because containers were
/// written to `containers/<name>` and the two strings coincided; the coincidence
/// was the only thing making it work, and it is gone.
///
/// A model with no registry record passes its own path through unchanged, which
/// is the fallback's GGUF path and must keep working.
std::optional<PreparedModel> prepare_model_for_node(const NodeInfo& node,
                                                    const ServedModel& cfg,
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

    // A model is a FILE for llama.cpp and a DIRECTORY for Soma, and this is the
    // one place that has to accept both: transfer_model_to_node() already walks
    // a directory and streams every file, but resolving through the file-only
    // form made that branch unreachable, so an admitted container was never
    // sent anywhere (roadmap D66).
    if (const auto model_path = util::resolve_existing_local_model_ref(model_ref, models_dir)) {
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

std::string host_from_node_url(std::string url) {
    const auto scheme = url.find("://");
    if (scheme != std::string::npos) url.erase(0, scheme + 3);
    const auto slash = url.find('/');
    if (slash != std::string::npos) url.resize(slash);
    if (!url.empty() && url.front() == '[') {
        const auto close = url.find(']');
        return close == std::string::npos ? std::string{} : url.substr(1, close - 1);
    }
    const auto colon = url.rfind(':');
    return colon == std::string::npos ? url : url.substr(0, colon);
}

std::vector<int> select_vllm_gpu_devices(const NodeInfo& node, int wanted) {
    std::unordered_set<int> reserved;
    for (const auto& slot : node.slots) {
        if (slot.backend != "vllm" || slot.state == SlotState::Empty ||
            slot.state == SlotState::Error) continue;
        reserved.insert(slot.gpu_devices.begin(), slot.gpu_devices.end());
    }
    std::vector<int> selected;
    for (int device = 0;
         device < node.capabilities.gpu_count &&
         static_cast<int>(selected.size()) < wanted;
         ++device) {
        if (reserved.count(device) == 0) selected.push_back(device);
    }
    return selected;
}

} // namespace

PlacementScheduler::PlacementScheduler(NodeRegistry& registry, std::string models_dir)
    : registry_(registry)
    , models_dir_(std::move(models_dir)) {}

PlacementScheduler::BackendRouting PlacementScheduler::resolve_backend(
    const ServedModel& cfg, const soma::AdmissionRecord& record) {
    if (!cfg.node_local()) {
        // API-backed models own no node slot at all.
        // This is a different question from which local engine to use.
        return {{}, "inference_backend '" + cfg.inference_backend + "' is not node-local"};
    }

    soma::BackendRequest rc;
    if (cfg.backend_override == "soma") rc.override = soma::BackendOverride::Soma;
    else if (cfg.backend_override == "fallback") rc.override = soma::BackendOverride::Fallback;
    rc.arch_hash = record.arch_hash;

    const auto decision = soma::select_backend(rc, record);
    return {decision.choice == soma::BackendChoice::Soma ? "soma" : "llama-cpp",
            decision.explain()};
}

void PlacementScheduler::set_placement_audit(PlacementAudit audit) {
    audit_ = std::move(audit);
}

void PlacementScheduler::flush_audit(std::vector<PendingAudit>& pending) const {
    for (const auto& e : pending) {
        if (e.placed) {
            if (audit_.placed)
                audit_.placed(
                    e.holder_id, e.node_id, e.slot_id, e.backend, e.backend_reason, e.footprint);
        } else if (audit_.released) {
            audit_.released(e.holder_id);
        }
    }
    pending.clear();
}

void PlacementScheduler::set_model_registry(const ControlModelRegistry* registry) {
    models_ = registry;
}

void PlacementScheduler::set_engine_config_gate(EngineConfigReadyFn ready) {
    engine_config_ready_ = std::move(ready);
    engine_config_required_ = static_cast<bool>(engine_config_ready_);
}

void PlacementScheduler::set_engine_config_provider(EngineConfigProvider provider) {
    engine_config_provider_ = std::move(provider);
}

std::vector<PlacementScheduler::BackendRouting>
PlacementScheduler::routing_candidates(const ServedModel& cfg) const {
    if (!cfg.node_local()) {
        return {{{}, "inference_backend '" + cfg.inference_backend +
                         "' is not node-local"}};
    }

    const auto admission_routing = [&] {
        soma::AdmissionRecord record;
        if (models_ != nullptr) {
            if (const auto admitted = models_->resolve(cfg.model_path)) {
                record.present = true;
                record.arch_hash = admitted->arch_hash;
                switch (admitted->verdict) {
                    case ModelVerdict::Stream:
                        record.verdict = soma::Verdict::Stream; break;
                    case ModelVerdict::Hybrid:
                        record.verdict = soma::Verdict::Hybrid; break;
                    case ModelVerdict::ResidentOnly:
                        record.verdict = soma::Verdict::ResidentOnly; break;
                    case ModelVerdict::Reject:
                        record.verdict = soma::Verdict::Reject; break;
                }
            }
        }
        return resolve_backend(cfg, record);
    };

    const auto configured = engine_config_provider_
        ? engine_config_provider_() : std::optional<ClusterEngineConfig>{};
    if (!configured) return {admission_routing()};

    const auto required = configured->required_engines();
    const auto contains = [&](const std::string& engine_id) {
        return std::find(required.begin(), required.end(), engine_id) !=
               required.end();
    };
    if (cfg.backend_override == "soma") {
        if (!contains("soma")) return {};
        const auto admitted = admission_routing();
        if (admitted.engine_id != "soma") return {};
        return {{"soma", "explicit Soma engine pin; " + admitted.reason}};
    }
    if (cfg.backend_override == "fallback") {
        if (!contains("llama-cpp")) return {};
        return {{"llama-cpp", "explicit llama.cpp engine pin"}};
    }

    std::vector<BackendRouting> out;
    for (const auto& engine_id : required) {
        if (engine_id == "soma") {
            const auto admitted = admission_routing();
            if (admitted.engine_id != "soma") continue;
            out.push_back({"soma", "cluster order; " + admitted.reason});
        } else {
            out.push_back({engine_id, "cluster engine order"});
        }
    }
    return out;
}

std::string PlacementScheduler::model_cache_id(const ServedModel& cfg) const {
    // The SAME derivation prepare_model_for_node() uses to name the transfer, so
    // "does this node hold it" is asked in the node's own vocabulary. Two
    // derivations would drift, and the failure would be silent: every node would
    // look like it holds nothing and every placement would be charged for a
    // transfer that never happens.
    const auto local = util::resolve_existing_local_model_ref(model_location(cfg), models_dir_);
    if (!local) return {};
    return manifest_cache_id(*local);
}

ResourceFootprint PlacementScheduler::footprint_for_node(const ServedModel& cfg,
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

ResourceFootprint PlacementScheduler::soma_footprint(const ServedModel& cfg) const {
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

std::string PlacementScheduler::model_location(const ServedModel& cfg) const {
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

std::string PlacementScheduler::request_model_name(const ServedModel& model,
                                                   const std::string& engine_id) {
    // vLLM serves under the name it was LAUNCHED with, which load_on_node sets
    // to engine_model_name when there is one. Every other engine is sent the
    // model's identity, which is what it was loaded from.
    if (engine_id == "vllm" && !model.engine_model_name.empty()) return model.engine_model_name;
    return model.model_path;
}

PlacementScheduler::BackendRouting
PlacementScheduler::resolve_backend_for(const ServedModel& cfg) const {
    if (engine_config_provider_) {
        const auto candidates = routing_candidates(cfg);
        if (!candidates.empty()) return candidates.front();
        return {{}, "explicit engine pin is not configured"};
    }
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

std::optional<PlacementScheduler::Claimed> PlacementScheduler::place(const ServedModel& cfg,
                                                                   bool claim,
                                                                   Attempt& attempt) {
    // The cluster has no engine policy yet. Refusing here rather than placing
    // on whatever a node happened to have is the whole point of the master
    // configurator: an unconfigured cluster serving from an accidental engine
    // set is exactly the state nobody could see before.
    //
    // The message names the fix because this is the first thing a fresh install
    // hits, and "no available nodes" would send the operator to look at nodes
    // that are all healthy.
    if (engine_config_required_ && !engine_config_ready_()) {
        release(cfg.id);
        attempt.fail(PlacementFailure::EngineConfigMissing,
                    "cluster engine configuration required: no primary engine has been set. "
                    "Configure it in the control TUI's Engines tab, with `engines setup` in "
                    "CLI mode, or via PUT /v1/cluster/engines/config");
        return std::nullopt;
    }

    const auto routings = routing_candidates(cfg);
    if (routings.size() == 1 && routings.front().engine_id.empty()) {
        // Release a prior local placement before reporting the routing result.
        release(cfg.id);
        attempt.fail(PlacementFailure::NoLocalBackend,
                     "model has no node-local backend: " + routings.front().reason);
        return std::nullopt;
    }
    if (routings.empty()) {
        release(cfg.id);
        attempt.fail(PlacementFailure::NoEligibleNode,
                     "the model's explicit engine pin is not configured as the "
                     "cluster primary or backup, or Soma admission refused it");
        return std::nullopt;
    }

    struct Candidate {
        BackendRouting routing;
        std::optional<VllmEngineConfig> vllm;
        std::uint32_t version = 0;
        std::string fingerprint;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(routings.size());
    for (const auto& routing : routings) {
        Candidate c;
        c.routing = routing;
        if (engine_config_provider_) {
            if (const auto cluster = engine_config_provider_()) {
                c.version = cluster->version;
                if (routing.engine_id == "vllm") {
                    if (const auto* spec = cluster->find("vllm"))
                        c.vllm = effective_vllm_config(*spec);
                }
            }
        }
        c.fingerprint = engine_fingerprint(cfg, models_dir_, routing.engine_id, c.vllm);
        candidates.push_back(std::move(c));
    }

    for (std::size_t candidate_index = 0; candidate_index < candidates.size();
         ++candidate_index) {
    const BackendRouting& routing = candidates[candidate_index].routing;
    const std::optional<VllmEngineConfig>& vllm = candidates[candidate_index].vllm;
    const std::uint32_t engine_config_version = candidates[candidate_index].version;
    const std::string& desired_fingerprint = candidates[candidate_index].fingerprint;

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
        const PlacementScheduler& self;
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
    attempt = {};

    // Reusing a live placement: refresh it and, for acquire(), CLAIM it while
    // the schedule lock is still held — there is no window between "placed" and
    // "in use" for an eviction to land in.
    const auto reuse_existing = [&](const Placement& live) -> std::optional<Claimed> {
        mutate_placement(cfg.id, [&](Placement& placement) {
            placement.last_active_ms = util::now_ms();
            if (claim) ++placement.in_flight;
        });
        return Claimed{live.node_id, live.slot_id, live.engine_id, live.epoch};
    };

    auto existing = find_placement_copy(cfg.id);
    if (existing && existing->engine_fingerprint != desired_fingerprint) {
        if (existing->in_flight > 0) {
            // Requests are still using the placement this call would replace.
            // If a LATER candidate describes it — it runs on the backup engine
            // while the primary is retried — that iteration reuses it; busy is
            // no time to move a model back to its preferred engine. Otherwise
            // its settings really did change, and it is replaced once it
            // drains rather than detached out from under those requests.
            const bool later_matches = std::any_of(
                candidates.begin() + static_cast<std::ptrdiff_t>(candidate_index) + 1,
                candidates.end(), [&](const Candidate& c) {
                    return c.fingerprint == existing->engine_fingerprint;
                });
            if (later_matches) continue;
            attempt.fail(PlacementFailure::Draining,
                         "the model's launch settings changed; its current placement is "
                         "replaced once the " + std::to_string(existing->in_flight) +
                         " request(s) using it finish");
            return std::nullopt;
        }
        MM_INFO("PlacementScheduler: engine identity changed for {}; "
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
            return reuse_existing(*existing);
        }

        const SlotInfo* reported_slot = nullptr;
        if (node_it != nodes.end() && node_it->connected) {
            const auto slot_it = std::find_if(
                node_it->slots.begin(), node_it->slots.end(),
                [&](const SlotInfo& slot) { return slot.id == existing->slot_id; });
            if (slot_it != node_it->slots.end()) reported_slot = &*slot_it;
        }
        const bool attached = reported_slot
            && std::find(reported_slot->holder_ids.begin(),
                         reported_slot->holder_ids.end(), cfg.id)
                   != reported_slot->holder_ids.end();

        if (attached && reported_slot->state == SlotState::Ready) {
            return reuse_existing(*existing);
        }
        if (attached && reported_slot->state == SlotState::Suspended) {
            existing->suspended = true;
            existing->kv_cache_node_path = reported_slot->kv_cache_path;
            store_placement(*existing);
        } else {
            MM_WARN("PlacementScheduler: discarding stale placement for {} "
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
    // a Soma model was charged VRAM it would never touch — and since
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
        Placement placement = *existing;
        auto publish_restored = [&](const NodeId& node_id, const SlotId& slot_id) {
            if (node_id != placement.node_id) {
                try {
                    const auto old_node = registry_.get_node(placement.node_id);
                    HttpClient old_client(old_node.url);
                    old_client.set_bearer_token(old_node.api_key);
                    static_cast<void>(old_client.post(
                        "/api/node/detach-holder",
                        nlohmann::json{{"slot_id", placement.slot_id},
                                       {"holder_id", cfg.id}}));
                } catch (const std::exception& e) {
                    MM_WARN("PlacementScheduler: restored {} on node {} but could "
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
            placement.engine_id = routing.engine_id;
            placement.reservation_state = routing.engine_id == "vllm"
                ? "exclusive" : std::string{};
            placement.kv_cache_node_path.clear();
            placement.in_flight = claim ? 1 : 0;
            const auto epoch = store_new_placement(placement);
            pending_audit.push_back(
                {true, cfg.id, node_id, slot_id, routing.engine_id, routing.reason, needed});
            return Claimed{node_id, slot_id, routing.engine_id, epoch};
        };

        auto slot_id = restore_on_node(
            placement, cfg, placement.node_id, routing, vllm, attempt);
        if (slot_id) {
            MM_INFO("PlacementScheduler: restored {} on original node {}",
                    cfg.id, placement.node_id);
            return publish_restored(placement.node_id, *slot_id);
        }

        const bool distributed_vllm = routing.engine_id == "vllm" && vllm &&
            vllm->pipeline_parallel_size > 1;
        if (distributed_vllm) {
            // A distributed sleeping server belongs to its existing Ray group.
            // If wake fails it cannot migrate like a checkpoint-backed local
            // engine: tear down the whole transaction, then let the normal Ray
            // planner form a fresh compatible group below.
            erase_placement_entry(cfg.id);
            pending_audit.push_back({false, cfg.id, {}, {}, {}, {}, {}});
            detach_placement_best_effort(
                placement, cfg.id, "distributed vLLM wake failed");
            existing.reset();
        } else {
            for (const auto& candidate_id : lru_idle_holders(placement.node_id)) {
                if (candidate_id == cfg.id ||
                    !evict_holder_for_capacity(candidate_id)) continue;
                slot_id = restore_on_node(
                    placement, cfg, placement.node_id, routing, vllm, attempt);
                if (slot_id) {
                    return publish_restored(placement.node_id, *slot_id);
                }
            }

            const auto restore_candidates = engine_config_provider_
                ? registry_.nodes_with_capacity_for_engine(
                      [&](const NodeInfo&) { return needed; }, routing.engine_id,
                      engine_config_version)
                : registry_.nodes_with_capacity(needed);
            for (const auto& node : restore_candidates) {
                if (node.id == placement.node_id) continue;
                slot_id = restore_on_node(
                    placement, cfg, node.id, routing, vllm, attempt);
                if (slot_id) return publish_restored(node.id, *slot_id);
            }
        }
    }

    auto publish_new = [&](const NodeId& node_id, const SlotId& slot_id) {
        Placement placement;
        placement.holder_id = cfg.id;
        placement.in_flight = claim ? 1 : 0;
        placement.node_id = node_id;
        placement.slot_id = slot_id;
        placement.placed_at_ms = util::now_ms();
        placement.last_active_ms = placement.placed_at_ms;
        placement.engine_fingerprint = desired_fingerprint;
        placement.engine_id = routing.engine_id;
        placement.reservation_state = routing.engine_id == "vllm"
            ? "exclusive" : std::string{};
        const auto epoch = store_new_placement(placement);
        pending_audit.push_back(
            {true, cfg.id, node_id, slot_id, routing.engine_id, routing.reason, needed});
        return Claimed{node_id, slot_id, routing.engine_id, epoch};
    };

    if (routing.engine_id == "vllm" && vllm &&
        vllm->pipeline_parallel_size > 1) {
        EngineGroupRequest request;
        request.model_ref = model_location(cfg);
        request.tensor_parallel_size = vllm->tensor_parallel_size;
        request.pipeline_parallel_size = vllm->pipeline_parallel_size;
        request.config_version = engine_config_version;
        request.allow_experimental_gloo = vllm->allow_experimental_gloo;
        const auto group_candidates = plan_engine_groups(request, registry_.list_nodes());
        if (group_candidates.empty()) {
            std::string detail;
            if (!distributed_vllm_model_ref_supported(request.model_ref, detail)) {
                attempt.fail(PlacementFailure::NoEligibleNode, detail);
            } else {
                attempt.fail(PlacementFailure::NoEligibleNode,
                            "no exclusive Ray group satisfies the configured vLLM "
                            "TP/PP topology and runtime compatibility");
            }
            continue;
        }

        for (const auto& group_candidate : group_candidates) {
        const auto* group = &group_candidate;

        const std::string group_id = util::generate_uuid();
        std::vector<NodeInfo> members;
        members.reserve(group->nodes.size());
        for (const auto& node_id : group->nodes)
            members.push_back(registry_.get_node(node_id));

        bool preflight_ok = true;
        for (const auto& member : members) {
            try {
                HttpClient client(member.url);
                client.set_bearer_token(member.api_key);
                const auto response = client.post(
                    "/api/node/vllm/preflight",
                    nlohmann::json{{"model_path", request.model_ref}});
                if (!response.ok()) {
                    preflight_ok = false;
                    attempt.fail(PlacementFailure::NodeRejected,
                                "vLLM model preflight failed on Ray member " +
                                    member.id + " (HTTP " +
                                    std::to_string(response.status) + "): " +
                                    response.body);
                    break;
                }
            } catch (const std::exception& e) {
                preflight_ok = false;
                attempt.fail(PlacementFailure::NodeUnreachable,
                            "vLLM model preflight could not reach Ray member " +
                                member.id + ": " + e.what());
                break;
            }
        }
        if (!preflight_ok) continue;

        std::vector<NodeInfo> started;
        std::string head_address;
        bool group_started = true;
        for (std::size_t index = 0; index < members.size(); ++index) {
            const auto& member = members[index];
            const std::string role = index == 0 ? "head" : "worker";
            const std::string node_ip = host_from_node_url(member.url);
            try {
                HttpClient client(member.url);
                client.set_bearer_token(member.api_key);
                const auto response = client.post(
                    "/api/node/ray/start",
                    nlohmann::json{{"group_id", group_id},
                                   {"holder_id", cfg.id},
                                   {"role", role},
                                   {"head_address", head_address},
                                   {"node_ip", node_ip},
                                   {"transport", group->transport},
                                   {"num_gpus", vllm->tensor_parallel_size}});
                if (!response.ok()) {
                    attempt.fail(PlacementFailure::NodeRejected,
                                "Ray " + role + " start failed on node " + member.id +
                                    " (HTTP " + std::to_string(response.status) +
                                    "): " + response.body);
                    group_started = false;
                    break;
                }
                const auto status = nlohmann::json::parse(response.body)
                                        .get<RayRuntimeStatus>();
                if (!status.active() || status.group_id != group_id ||
                    status.reserved_gpus != vllm->tensor_parallel_size) {
                    attempt.fail(PlacementFailure::NodeProtocolError,
                                "Ray member " + member.id +
                                    " did not confirm the requested group/GPU reservation");
                    group_started = false;
                    break;
                }
                if (index == 0) head_address = status.head_address;
                started.push_back(member);
            } catch (const std::exception& e) {
                attempt.fail(PlacementFailure::NodeUnreachable,
                            "Ray " + role + " start failed on node " + member.id +
                                ": " + e.what());
                group_started = false;
                break;
            }
        }

        auto rollback = [&] {
            for (auto it = started.rbegin(); it != started.rend(); ++it) {
                try {
                    HttpClient client(it->url);
                    client.set_bearer_token(it->api_key);
                    static_cast<void>(client.post(
                        "/api/node/ray/stop", nlohmann::json{{"group_id", group_id}}));
                } catch (...) {
                }
            }
        };
        if (!group_started) {
            rollback();
            continue;
        }

        std::vector<int> gpu_devices;
        for (int index = 0; index < vllm->tensor_parallel_size; ++index)
            gpu_devices.push_back(index);
        const NodeInfo& head = members.front();
        const auto slot_id = load_on_node(
            cfg, head.id, routing, vllm, attempt, head_address,
            host_from_node_url(head.url), gpu_devices);
        if (!slot_id) {
            rollback();
            continue;
        }

        Placement placement;
        placement.holder_id = cfg.id;
        placement.node_id = head.id;
        placement.slot_id = *slot_id;
        placement.engine_fingerprint = desired_fingerprint;
        placement.engine_id = routing.engine_id;
        placement.ray_group_id = group_id;
        placement.member_node_ids = group->nodes;
        placement.ray_role = "head";
        placement.ray_transport = group->transport;
        placement.reservation_state = "exclusive";
        placement.placed_at_ms = util::now_ms();
        placement.last_active_ms = placement.placed_at_ms;
        placement.in_flight = claim ? 1 : 0;
        const auto epoch = store_new_placement(placement);
        pending_audit.push_back(
            {true, cfg.id, head.id, *slot_id, routing.engine_id,
             routing.reason, needed});
        return Claimed{head.id, *slot_id, routing.engine_id, epoch};
        }
        continue;
    }

    std::unordered_set<NodeId> attempted_nodes;
    auto available_for_engine = engine_config_provider_
        ? registry_.available_nodes_for_engine(routing.engine_id,
                                               engine_config_version)
        : registry_.available_nodes();
    if (routing.engine_id == "vllm" && vllm) {
        available_for_engine.erase(
            std::remove_if(available_for_engine.begin(), available_for_engine.end(),
                           [&](const NodeInfo& node) {
                               return static_cast<int>(select_vllm_gpu_devices(
                                          node, vllm->tensor_parallel_size).size()) <
                                      vllm->tensor_parallel_size;
                           }),
            available_for_engine.end());
    }
    const auto is_available = [&](const NodeId& node_id) {
        return std::any_of(
            available_for_engine.begin(), available_for_engine.end(),
            [&](const NodeInfo& node) { return node.id == node_id; });
    };
    auto try_load = [&](const NodeId& node_id) -> std::optional<SlotId> {
        if (node_id.empty() || !is_available(node_id) ||
            !attempted_nodes.insert(node_id).second) {
            return std::nullopt;
        }
        std::vector<int> selected_gpu_devices;
        if (routing.engine_id == "vllm" && vllm) {
            const auto node = std::find_if(
                available_for_engine.begin(), available_for_engine.end(),
                [&](const NodeInfo& candidate) { return candidate.id == node_id; });
            if (node == available_for_engine.end()) return std::nullopt;
            selected_gpu_devices = select_vllm_gpu_devices(
                *node, vllm->tensor_parallel_size);
        }
        auto slot_id = load_on_node(
            cfg, node_id, routing, vllm, attempt, {}, {}, selected_gpu_devices);
        if (!slot_id) {
            MM_WARN("PlacementScheduler: node {} could not run {}; "
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
        size_t attached_holders = 0;
        int64_t last_active_ms = 0;
    };
    std::vector<SharedCandidate> shared_candidates;
    for (const auto& node : available_for_engine) {
        if (attempted_nodes.count(node.id)) continue;
        std::optional<SharedCandidate> best;
        for (const auto& slot : node.slots) {
            // Match the model's RESOLVED engine, not "llama-cpp". Sharing a slot
            // running a different engine would attach the model to a process
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
                node.id, slot.holder_ids.size(), slot.last_active_ms};
            if (!best ||
                std::tie(candidate.attached_holders, candidate.last_active_ms)
                    < std::tie(best->attached_holders, best->last_active_ms)) {
                best = candidate;
            }
        }
        if (best) shared_candidates.push_back(*best);
    }
    std::sort(shared_candidates.begin(), shared_candidates.end(),
              [](const SharedCandidate& lhs, const SharedCandidate& rhs) {
                  return std::tie(lhs.attached_holders, lhs.last_active_ms)
                       < std::tie(rhs.attached_holders, rhs.last_active_ms);
              });
    for (const auto& candidate : shared_candidates) {
        if (const auto slot_id = try_load(candidate.node_id)) {
            MM_INFO("PlacementScheduler: {} joined a compatible "
                    "engine for {} on node {}", cfg.id, cfg.model_path,
                    candidate.node_id);
            return publish_new(candidate.node_id, *slot_id);
        }
    }

    // Per-node demand: `needed` is what the model costs anywhere, and
    // footprint_for_node() adds the container's disk only to nodes that would
    // have to fetch it.
    const auto demand = [&](const NodeInfo& node) { return footprint_for_node(cfg, needed, node); };
    const auto capacity_candidates = engine_config_provider_
        ? registry_.nodes_with_capacity_for_engine(
              demand, routing.engine_id, engine_config_version)
        : registry_.nodes_with_capacity_for(demand);
    for (const auto& node : capacity_candidates) {
        if (const auto slot_id = try_load(node.id)) {
            return publish_new(node.id, *slot_id);
        }
    }

    for (const auto& candidate_id : lru_idle_holders()) {
        if (candidate_id == cfg.id) continue;
        const auto candidate = find_placement_copy(candidate_id);
        if (!candidate ||
            (candidate->suspended && candidate->engine_id != "vllm") ||
            candidate->in_flight > 0
            || !evict_holder_for_capacity(candidate_id)) {
            continue;
        }
        std::vector<int> selected_gpu_devices;
        if (routing.engine_id == "vllm" && vllm) {
            const auto node = std::find_if(
                available_for_engine.begin(), available_for_engine.end(),
                [&](const NodeInfo& available) {
                    return available.id == candidate->node_id;
                });
            if (node != available_for_engine.end())
                selected_gpu_devices = select_vllm_gpu_devices(
                    *node, vllm->tensor_parallel_size);
        }
        if (const auto slot_id = load_on_node(
                cfg, candidate->node_id, routing, vllm, attempt, {}, {},
                selected_gpu_devices)) {
            return publish_new(candidate->node_id, *slot_id);
        }
    }

    MM_WARN("PlacementScheduler: no capacity for {} (model={})",
            cfg.id, cfg.model_path);

    // The failure CODE, not the message. An older check asked "did a load
    // attempt already record something" by testing a STRING for emptiness,
    // which is the same defect one layer down from the one codes fix: control
    // flow depending on prose.
    if (attempt.failure == PlacementFailure::None) {
        // THE distinction D64 exists for. Both of these used to read "no
        // capacity: no connected node could load this model", and they call for
        // opposite actions: one means fix the cluster, the other means wait or
        // add hardware. `available_nodes()` applies the connected + conforming
        // filter, so an empty list means nothing was ELIGIBLE — every node is
        // offline, unconfigured, or not conforming to the engine policy.
        if (available_for_engine.empty()) {
            attempt.fail(PlacementFailure::NoEligibleNode,
                        "no eligible node: none is connected and conforming to the "
                        "cluster engine configuration");
        } else {
            attempt.fail(PlacementFailure::NoCapacity,
                        "no capacity: eligible nodes are connected, but none could "
                        "load this model");
        }
    }
    // A placement/startup failure is the only point at which routing moves to
    // the next engine. Inference has not begun yet, so this can never replay a
    // request on a different backend.
    MM_WARN("PlacementScheduler: exhausted engine {} for {}; trying backup",
            routing.engine_id, cfg.id);
    }
    return std::nullopt;
}

PlacementScheduler::Outcome PlacementScheduler::acquire(const ServedModel& model,
                                                         std::chrono::milliseconds wait) {
    const auto deadline = std::chrono::steady_clock::now() + wait;
    for (;;) {
        Attempt attempt;
        if (auto claimed = place(model, /*claim=*/true, attempt)) {
            Outcome out;
            out.node_id = claimed->node_id;
            out.slot_id = claimed->slot_id;
            out.engine_id = claimed->engine_id;
            Lease lease;
            lease.owner_ = this;
            lease.holder_id_ = model.id;
            lease.epoch_ = claimed->epoch;
            lease.node_id_ = claimed->node_id;
            lease.slot_id_ = claimed->slot_id;
            lease.engine_id_ = claimed->engine_id;
            lease.request_model_ = request_model_name(model, claimed->engine_id);
            out.lease = std::move(lease);
            return out;
        }
        if (attempt.failure == PlacementFailure::None) {
            // Every return path above records why. A path that does not is a
            // bug, but the caller still deserves a code rather than silence.
            attempt.fail(PlacementFailure::NoCapacity, "placement failed without a recorded reason");
        }
        if (!placement_failure_transient(attempt.failure) ||
            std::chrono::steady_clock::now() >= deadline) {
            Outcome out;
            out.failure = attempt.failure;
            out.error = attempt.error;
            return out;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

PlacementScheduler::Outcome PlacementScheduler::ensure_running(const ServedModel& model) {
    Attempt attempt;
    Outcome out;
    if (auto claimed = place(model, /*claim=*/false, attempt)) {
        out.node_id = claimed->node_id;
        out.slot_id = claimed->slot_id;
        out.engine_id = claimed->engine_id;
        return out;
    }
    out.failure = attempt.failure == PlacementFailure::None ? PlacementFailure::NoCapacity
                                                            : attempt.failure;
    out.error = attempt.error.empty() ? "placement failed without a recorded reason"
                                      : attempt.error;
    return out;
}

// ── leases ────────────────────────────────────────────────────────────────────

PlacementScheduler::Lease::Lease(Lease&& other) noexcept
    : owner_(other.owner_),
      holder_id_(std::move(other.holder_id_)),
      epoch_(other.epoch_),
      node_id_(std::move(other.node_id_)),
      slot_id_(std::move(other.slot_id_)),
      engine_id_(std::move(other.engine_id_)),
      request_model_(std::move(other.request_model_)) {
    other.owner_ = nullptr;
}

PlacementScheduler::Lease& PlacementScheduler::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        release();
        owner_ = other.owner_;
        holder_id_ = std::move(other.holder_id_);
        epoch_ = other.epoch_;
        node_id_ = std::move(other.node_id_);
        slot_id_ = std::move(other.slot_id_);
        engine_id_ = std::move(other.engine_id_);
        request_model_ = std::move(other.request_model_);
        other.owner_ = nullptr;
    }
    return *this;
}

PlacementScheduler::Lease::~Lease() {
    release();
}

void PlacementScheduler::Lease::release() noexcept {
    if (owner_ == nullptr) return;
    owner_->release_claim(holder_id_, epoch_);
    owner_ = nullptr;
}

void PlacementScheduler::release_claim(const HolderId& holder_id, std::uint64_t epoch) noexcept {
    try {
        std::lock_guard state_lock(state_mutex_);
        const auto it = placements_.find(holder_id);
        // A different epoch means the placement this claim was taken on has
        // since been replaced; decrementing the replacement would let it be
        // evicted while its own requests were still running.
        if (it == placements_.end() || it->second.epoch != epoch) return;
        if (it->second.in_flight > 0) --it->second.in_flight;
        it->second.last_active_ms = util::now_ms();
    } catch (...) {
        // Locking a std::mutex does not throw in practice; a destructor must not.
    }
}

// ── release ───────────────────────────────────────────────────────────────────

PlacementScheduler::ReleaseResult PlacementScheduler::release(const HolderId& holder_id,
                                                              bool force) {
    std::optional<Placement> placement;
    {
        std::lock_guard schedule_lock(schedule_mutex_);
        placement = find_placement_copy(holder_id);
        if (!placement) return ReleaseResult::NotPlaced;
        if (placement->in_flight > 0 && !force) return ReleaseResult::Busy;
        erase_placement_entry(holder_id);
    }

    // Outside the lock, like every other audit call — see PlacementAudit.
    if (audit_.released) audit_.released(holder_id);
    detach_placement_best_effort(*placement, holder_id,
                                 placement->in_flight > 0
                                     ? "placement released while requests were in flight"
                                     : "placement released");
    MM_INFO("PlacementScheduler: released {}", holder_id);
    return ReleaseResult::Released;
}

std::optional<Placement> PlacementScheduler::get_placement(const HolderId& holder_id) const {
    return find_placement_copy(holder_id);
}

std::vector<Placement> PlacementScheduler::list_placements() const {
    std::lock_guard state_lock(state_mutex_);
    std::vector<Placement> result;
    result.reserve(placements_.size());
    for (const auto& [_, placement] : placements_) result.push_back(placement);
    return result;
}

std::optional<Placement> PlacementScheduler::find_placement_copy(const HolderId& id) const {
    std::lock_guard state_lock(state_mutex_);
    const auto it = placements_.find(id);
    if (it == placements_.end()) return std::nullopt;
    return it->second;
}

std::uint64_t PlacementScheduler::store_new_placement(Placement placement) {
    std::lock_guard state_lock(state_mutex_);
    placement.epoch = next_epoch_++;
    const auto epoch = placement.epoch;
    placements_[placement.holder_id] = std::move(placement);
    return epoch;
}

void PlacementScheduler::store_placement(const Placement& placement) {
    std::lock_guard state_lock(state_mutex_);
    placements_[placement.holder_id] = placement;
}

bool PlacementScheduler::erase_placement_entry(const HolderId& id) {
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
    case PlacementFailure::Draining:
        return "draining";
    }
    return "unknown";
}

bool placement_failure_retryable(PlacementFailure failure) noexcept {
    switch (failure) {
    case PlacementFailure::EngineConfigMissing: // an operator must set a policy
    case PlacementFailure::NoLocalBackend:      // the model owns no slot by design
        return false;
    default:
        return true;
    }
}

bool placement_failure_transient(PlacementFailure failure) noexcept {
    switch (failure) {
    case PlacementFailure::NoCapacity:     // something else finishes and is evicted
    case PlacementFailure::NoEligibleNode: // a node finishes converging or rejoins
    case PlacementFailure::Draining:       // the old placement's requests finish
        return true;
    default:
        return false;
    }
}

void PlacementScheduler::detach_placement_best_effort(
    const Placement& placement,
    const HolderId& holder_id,
    const std::string& reason) {
    try {
        const auto node = registry_.get_node(placement.node_id);
        HttpClient client(node.url);
        client.set_bearer_token(node.api_key);
        const auto response = client.post(
            "/api/node/detach-holder",
            nlohmann::json{{"slot_id", placement.slot_id},
                           {"holder_id", holder_id}});
        if (!response.ok()) {
            MM_WARN("PlacementScheduler: node {} rejected detach for {} "
                    "({}; HTTP {})", placement.node_id, holder_id, reason,
                    response.status);
        }
    } catch (const std::exception& e) {
        MM_WARN("PlacementScheduler: failed to detach slot for {} ({}): {}",
                holder_id, reason, e.what());
    }
    stop_ray_group_best_effort(placement, reason);
}

void PlacementScheduler::stop_ray_group_best_effort(
    const Placement& placement, const std::string& reason) const {
    if (placement.ray_group_id.empty()) return;
    for (auto it = placement.member_node_ids.rbegin();
         it != placement.member_node_ids.rend(); ++it) {
        try {
            const auto node = registry_.get_node(*it);
            HttpClient client(node.url);
            client.set_bearer_token(node.api_key);
            const auto response = client.post(
                "/api/node/ray/stop",
                nlohmann::json{{"group_id", placement.ray_group_id}});
            if (!response.ok()) {
                MM_WARN("PlacementScheduler: Ray cleanup for group {} on node {} "
                        "failed ({}; HTTP {})", placement.ray_group_id, *it,
                        reason, response.status);
            }
        } catch (const std::exception& e) {
            MM_WARN("PlacementScheduler: Ray cleanup for group {} on node {} "
                    "failed ({}): {}", placement.ray_group_id, *it, reason,
                    e.what());
        }
    }
}

void PlacementScheduler::housekeeping(const std::vector<ServedModel>& live) {
    std::vector<Placement> detaches;

    {
        std::lock_guard schedule_lock(schedule_mutex_);
        std::lock_guard state_lock(state_mutex_);

        std::unordered_set<HolderId> live_ids;
        live_ids.reserve(live.size());
        for (const auto& model : live) live_ids.insert(model.id);

        for (auto it = placements_.begin(); it != placements_.end();) {
            // A holder deleted while a request is still finishing on it is left
            // for the next pass: detaching now would kill the response.
            if (live_ids.count(it->first) || it->second.in_flight > 0) {
                ++it;
                continue;
            }
            detaches.push_back(it->second);
            MM_INFO("PlacementScheduler: housekeeping removed placement for deleted holder {}",
                    it->first);
            it = placements_.erase(it);
        }
    }

    for (const auto& detach : detaches) {
        if (audit_.released) {
            try { audit_.released(detach.holder_id); } catch (...) {}
        }
        detach_placement_best_effort(
            detach, detach.holder_id, "holder deleted during housekeeping");
    }

    // Control placements are intentionally in-memory, while node-owned Ray
    // daemons survive a control restart. Reconstruct only a complete group
    // whose head still advertises the holder's ready vLLM slot; anything else is
    // incomplete, conflicting, or orphaned and must release its reservations.
    std::unordered_set<HolderId> active_ids;
    for (const auto& model : live) active_ids.insert(model.id);
    std::unordered_map<std::string, std::vector<NodeInfo>> ray_groups;
    for (const auto& node : registry_.list_nodes()) {
        if (!node.ray.group_id.empty())
            ray_groups[node.ray.group_id].push_back(node);
    }
    std::optional<VllmEngineConfig> expected_profile;
    std::uint32_t expected_version = 0;
    if (engine_config_provider_) {
        if (const auto cluster = engine_config_provider_()) {
            expected_version = cluster->version;
            if ((cluster->primary_engine == "vllm" || cluster->backup_engine == "vllm")) {
                if (const auto* spec = cluster->find("vllm"))
                    expected_profile = effective_vllm_config(*spec);
            }
        }
    }
    for (const auto& [group_id, members] : ray_groups) {
        Placement recovered;
        recovered.ray_group_id = group_id;
        bool valid = expected_profile.has_value() &&
                     static_cast<int>(members.size()) ==
                         expected_profile->pipeline_parallel_size;
        int heads = 0;
        std::string runtime_fingerprint;
        for (const auto& member : members) {
            recovered.member_node_ids.push_back(member.id);
            if (recovered.holder_id.empty()) recovered.holder_id = member.ray.holder_id;
            if (recovered.ray_transport.empty())
                recovered.ray_transport = member.ray.transport;
            const auto runtime = std::find_if(
                member.engines.begin(), member.engines.end(),
                [](const RuntimeStatus& status) {
                    return status.engine_id == "vllm" && status.ready;
                });
            const std::string member_fingerprint = runtime == member.engines.end()
                ? std::string{}
                : member.capabilities.arch + "|" + runtime->version + "|" +
                      runtime->variant;
            if (runtime_fingerprint.empty()) runtime_fingerprint = member_fingerprint;
            const bool transport_available = member.ray.transport == "nccl"
                ? std::find(member.capabilities.comm_backends.begin(),
                            member.capabilities.comm_backends.end(), "nccl") !=
                      member.capabilities.comm_backends.end()
                : expected_profile->allow_experimental_gloo &&
                      member.ray.transport == "gloo" &&
                      std::find(member.capabilities.comm_backends.begin(),
                                member.capabilities.comm_backends.end(), "gloo") !=
                          member.capabilities.comm_backends.end();
            valid = valid && member.connected &&
                    util::to_lower(member.platform) == "linux" &&
                    member.capabilities.supports_ray && runtime != member.engines.end() &&
                    !member_fingerprint.empty() &&
                    member_fingerprint == runtime_fingerprint && transport_available &&
                    member.capabilities.gpu_count >=
                        expected_profile->tensor_parallel_size && member.ray.active() &&
                    member.ray.holder_id == recovered.holder_id &&
                    member.ray.transport == recovered.ray_transport &&
                    member.ray.reserved_gpus == expected_profile->tensor_parallel_size &&
                    member.engine_config_version == expected_version;
            if (member.ray.role == "head") {
                ++heads;
                recovered.node_id = member.id;
                for (const auto& slot : member.slots) {
                    const bool attached =
                        std::find(slot.holder_ids.begin(), slot.holder_ids.end(),
                                  recovered.holder_id) != slot.holder_ids.end();
                    if (slot.backend == "vllm" &&
                        (slot.state == SlotState::Ready ||
                         slot.state == SlotState::Suspended) && attached) {
                        recovered.slot_id = slot.id;
                        recovered.suspended = slot.state == SlotState::Suspended;
                    }
                }
            }
        }
        valid = valid && heads == 1 && !recovered.holder_id.empty() &&
                active_ids.count(recovered.holder_id) > 0 && !recovered.slot_id.empty();
        const auto recovered_cfg = std::find_if(
            live.begin(), live.end(), [&](const ServedModel& cfg) {
                return cfg.id == recovered.holder_id;
            });
        valid = valid && recovered_cfg != live.end();
        const auto existing = recovered.holder_id.empty()
            ? std::optional<Placement>{}
            : find_placement_copy(recovered.holder_id);
        if (existing && existing->ray_group_id != group_id) valid = false;
        if (valid) {
            if (!existing) {
                recovered.engine_id = "vllm";
                recovered.ray_role = "head";
                recovered.reservation_state = recovered.suspended
                    ? "sleeping" : "exclusive";
                recovered.engine_fingerprint = engine_fingerprint(
                    *recovered_cfg, models_dir_, "vllm", expected_profile);
                recovered.placed_at_ms = util::now_ms();
                recovered.last_active_ms = recovered.placed_at_ms;
                store_new_placement(recovered);
                MM_INFO("PlacementScheduler: reconciled surviving Ray group {} for {}",
                        group_id, recovered.holder_id);
            }
            continue;
        }
        // detach_placement_best_effort enforces the required ordering: the
        // head server is stopped first, followed by owned Ray members.
        detach_placement_best_effort(
            recovered, recovered.holder_id,
            "incomplete, conflicting, or orphaned Ray group during reconciliation");
    }
}

std::vector<HolderId> PlacementScheduler::lru_idle_holders(const NodeId& on_node) const {
    std::lock_guard state_lock(state_mutex_);
    std::vector<std::pair<HolderId, int64_t>> candidates;
    for (const auto& [id, placement] : placements_) {
        if ((placement.suspended && placement.engine_id != "vllm") ||
            placement.in_flight > 0) continue;
        if (!on_node.empty() && placement.node_id != on_node) continue;
        candidates.emplace_back(id, placement.last_active_ms);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.second, lhs.first) < std::tie(rhs.second, rhs.first);
    });

    std::vector<HolderId> result;
    result.reserve(candidates.size());
    for (auto& [id, _] : candidates) result.push_back(std::move(id));
    return result;
}

PlacementScheduler::SuspendResult PlacementScheduler::suspend(const HolderId& holder_id) {
    // Under the schedule lock, like every other change to a placement: without
    // it a concurrent acquire() could claim the placement between the in-flight
    // check below and the node actually suspending it.
    std::lock_guard schedule_lock(schedule_mutex_);
    return suspend_locked(holder_id);
}

PlacementScheduler::SuspendResult PlacementScheduler::suspend_locked(const HolderId& holder_id) {
    const auto placement = find_placement_copy(holder_id);
    if (!placement || placement->suspended) return SuspendResult::NotPlaced;

    std::vector<HolderId> cohort{holder_id};
    {
        std::lock_guard state_lock(state_mutex_);
        // Its OWN requests first. The old check looked only at co-holders on
        // the slot, so a placement could be suspended in the middle of a
        // response it was itself producing.
        const auto self = placements_.find(holder_id);
        if (self != placements_.end() && self->second.in_flight > 0) {
            MM_INFO("PlacementScheduler: not suspending {}; {} request(s) in flight",
                    holder_id, self->second.in_flight);
            return SuspendResult::Busy;
        }
        for (const auto& [id, candidate] : placements_) {
            if (id == holder_id || candidate.suspended) continue;
            if (candidate.node_id != placement->node_id
                || candidate.slot_id != placement->slot_id) {
                continue;
            }
            if (candidate.in_flight > 0) {
                MM_INFO("PlacementScheduler: not suspending {}; slot {} is "
                        "shared with busy holder {}", holder_id,
                        placement->slot_id, id);
                return SuspendResult::Busy;
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
            MM_WARN("PlacementScheduler: suspend failed for {} on node {} "
                    "(HTTP {}): {}", holder_id, placement->node_id,
                    response.status, preview);
            return SuspendResult::Failed;
        }
        kv_cache_path = nlohmann::json::parse(response.body)
                            .value("kv_cache_path", std::string{});
    } catch (const std::exception& e) {
        MM_WARN("PlacementScheduler: suspend failed for {}: {}", holder_id, e.what());
        return SuspendResult::Failed;
    }

    bool updated = false;
    for (const auto& id : cohort) {
        const bool changed = mutate_placement(id, [&](Placement& candidate) {
            candidate.suspended = true;
            candidate.kv_cache_node_path = kv_cache_path;
            if (candidate.engine_id == "vllm")
                candidate.reservation_state = "sleeping";
        });
        if (id == holder_id) updated = changed;
    }
    if (!updated) return SuspendResult::Failed;

    MM_INFO("PlacementScheduler: suspended {} on node {} (cache={})",
            holder_id, placement->node_id, kv_cache_path);
    return SuspendResult::Suspended;
}

std::optional<SlotId> PlacementScheduler::restore_on_node(
    const Placement& placement,
    const ServedModel& cfg,
    const NodeId& node_id,
    const BackendRouting& routing,
    const std::optional<VllmEngineConfig>& vllm,
    Attempt& attempt) {
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
            attempt.fail(PlacementFailure::ModelTransferFailed,
                        "failed to prepare model for restore on node " + node_id + ": " +
                            prepare_error);
            return std::nullopt;
        }

        for (int try_index = 0; try_index < 3; ++try_index) {
            nlohmann::json body = {
                {"model_path", prepared->model_path},
                {"mmproj_path", prepared->mmproj_path},
                {"vision_enabled", cfg.vision_settings.enabled},
                {"runtime_settings", cfg.runtime_settings},
                {"kv_cache_path", node_id == placement.node_id
                    ? placement.kv_cache_node_path : std::string{}},
                // The RESOLVED engine id. Hardcoding "llama-cpp" here is what
                // made a Soma model unplaceable regardless of its verdict.
                {"backend", routing.engine_id},
                {"served_model_name", cfg.engine_model_name.empty()
                    ? cfg.model_path : cfg.engine_model_name},
                {"holder_id", cfg.id},
            };
            if (vllm) body["vllm"] = *vllm;
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
                attempt.fail(PlacementFailure::NodeProtocolError,
                            "restore-slot returned an empty slot_id on node " + node_id);
                return std::nullopt;
            }

            if (response_indicates_capacity_pressure(response.body)) {
                if (try_index == 0 && evict_slots_on_node(node_id, cfg.id, 1)) continue;
                if (try_index == 1 && evict_slots_on_node(node_id, cfg.id, -1)) continue;
            }

            std::string preview = response.body;
            if (preview.size() > 300) preview = preview.substr(0, 300) + "...";
            attempt.fail(PlacementFailure::NodeRejected,
                        "restore-slot failed on node " + node_id + " (HTTP " +
                            std::to_string(response.status) + "): " + preview);
            return std::nullopt;
        }
    } catch (const std::exception& e) {
        MM_WARN("PlacementScheduler: restore failed on node {}: {}", node_id, e.what());
        attempt.fail(PlacementFailure::NodeUnreachable,
                    "restore-slot exception on node " + node_id + ": " + e.what());
    }
    return std::nullopt;
}

std::optional<SlotId> PlacementScheduler::load_on_node(
    const ServedModel& cfg,
    const NodeId& node_id,
    const BackendRouting& routing,
    const std::optional<VllmEngineConfig>& vllm,
    Attempt& attempt,
    const std::string& ray_address,
    const std::string& host_ip,
    const std::vector<int>& gpu_devices) {
    try {
        const auto node = registry_.get_node(node_id);
        HttpClient client(node.url);
        client.set_bearer_token(node.api_key);
        const bool pin = !cfg.preferred_node_id.empty()
            && cfg.preferred_node_id == node_id;

        std::string prepare_error;
        std::optional<PreparedModel> prepared;
        if (!ray_address.empty()) {
            // Every member preflighted this exact shared reference. Copying it
            // into the head's managed cache would destroy the identical-path
            // invariant that distributed vLLM relies on.
            PreparedModel shared;
            shared.model_path = model_location(cfg);
            shared.mmproj_path = cfg.vision_settings.mmproj_path;
            prepared = std::move(shared);
        } else {
            prepared = prepare_model_for_node(
                node, cfg, model_location(cfg), models_dir_, pin, false, &prepare_error);
        }
        if (!prepared) {
            attempt.fail(PlacementFailure::ModelTransferFailed,
                        "failed to prepare model for node " + node_id + ": " + prepare_error);
            return std::nullopt;
        }

        bool retried_transfer = false;
        for (int try_index = 0; try_index < 3; ++try_index) {
            nlohmann::json body = {
                {"model_path", prepared->model_path},
                {"mmproj_path", prepared->mmproj_path},
                {"vision_enabled", cfg.vision_settings.enabled},
                {"runtime_settings", cfg.runtime_settings},
                // The RESOLVED engine id. Hardcoding "llama-cpp" here is what
                // made a Soma model unplaceable regardless of its verdict.
                {"backend", routing.engine_id},
                {"served_model_name", cfg.engine_model_name.empty()
                    ? cfg.model_path : cfg.engine_model_name},
                {"holder_id", cfg.id},
            };
            if (vllm) body["vllm"] = *vllm;
            if (!ray_address.empty()) body["ray_address"] = ray_address;
            if (!host_ip.empty()) body["host_ip"] = host_ip;
            if (!gpu_devices.empty()) body["gpu_devices"] = gpu_devices;
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
                    MM_INFO("PlacementScheduler: loaded {} on node {} (slot={})",
                            cfg.id, node_id, slot_id);
                    return slot_id;
                }
                attempt.fail(PlacementFailure::NodeProtocolError,
                            "load-model returned an empty slot_id on node " + node_id);
                return std::nullopt;
            }

            if (response_indicates_capacity_pressure(response.body)) {
                if (try_index == 0 && evict_slots_on_node(node_id, cfg.id, 1)) continue;
                if (try_index == 1 && evict_slots_on_node(node_id, cfg.id, -1)) continue;
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
            attempt.fail(PlacementFailure::NodeRejected,
                        "load-model failed on node " + node_id + " (HTTP " +
                            std::to_string(response.status) + "): " + preview);
            return std::nullopt;
        }
    } catch (const std::exception& e) {
        MM_WARN("PlacementScheduler: load failed on node {}: {}", node_id, e.what());
        attempt.fail(PlacementFailure::NodeUnreachable,
                    "load-model exception on node " + node_id + ": " + e.what());
    }
    return std::nullopt;
}

bool PlacementScheduler::response_indicates_capacity_pressure(const std::string& body) {
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

bool PlacementScheduler::evict_slots_on_node(const NodeId& node_id,
                                             const HolderId& preserve_holder,
                                             int max_to_evict) {
    int evicted = 0;
    const auto can_evict_more = [&] {
        return max_to_evict < 0 || evicted < max_to_evict;
    };

    for (const auto& candidate_id : lru_idle_holders(node_id)) {
        if (!can_evict_more()) break;
        if (candidate_id == preserve_holder) continue;
        const auto placement = find_placement_copy(candidate_id);
        if (!placement || placement->suspended || placement->in_flight > 0) continue;
        if (!evict_holder_for_capacity(candidate_id)) continue;
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
                MM_WARN("PlacementScheduler: failed to parse node status while evicting "
                        "on {}: {}", node_id, e.what());
            }
        }

        std::unordered_set<SlotId> protected_slots;
        {
            std::lock_guard state_lock(state_mutex_);
            for (const auto& [holder_id, placement] : placements_) {
                if (placement.node_id != node_id) continue;
                if (placement.in_flight > 0
                    || (!preserve_holder.empty() && holder_id == preserve_holder)) {
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
            if (!preserve_holder.empty()
                && std::find(slot.holder_ids.begin(), slot.holder_ids.end(),
                             preserve_holder) != slot.holder_ids.end()) {
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
                MM_WARN("PlacementScheduler: failed to unload slot {} on node {} "
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
            MM_INFO("PlacementScheduler: directly unloaded slot {} on node {} "
                    "to free capacity", candidate.id, node_id);
        }

        return evicted > 0;
    } catch (const std::exception& e) {
        MM_WARN("PlacementScheduler: eviction fallback failed on node {}: {}",
                node_id, e.what());
        return evicted > 0;
    }
}

bool PlacementScheduler::evict_holder_for_capacity(const HolderId& holder_id) {
    // Called only from inside place(), which holds the schedule lock.
    const auto placement = find_placement_copy(holder_id);
    if (!placement) return false;
    if (placement->engine_id != "vllm")
        return suspend_locked(holder_id) == SuspendResult::Suspended;

    // vLLM sleep deliberately keeps its GPU/Ray reservation. Under capacity
    // pressure that cannot help the pending load, so unload the placement and
    // release the owned group instead of pretending its GPUs became reusable.
    if (!erase_placement_entry(holder_id)) return false;
    detach_placement_best_effort(
        *placement, holder_id, "vLLM placement evicted under capacity pressure");
    if (audit_.released) {
        try { audit_.released(holder_id); } catch (...) {}
    }
    return true;
}

} // namespace mm
