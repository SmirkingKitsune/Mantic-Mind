// Mantic-Mind — the engine registry: llama.cpp and Soma as two DESCRIPTORS.
//
// The point of this file is what is NOT in it. Today the node has
// `backend != "llama-cpp" -> 400` duplicated at node_api_server.cpp:449 and
// :911, `SlotManager::Slot` hardcodes `unique_ptr<RuntimeProcess>` plus llama
// paths, and adding an engine means touching all of them. Here an engine is a
// row of callbacks, and the node's only question is "is this id registered?".
//
// A consequence worth stating: an unknown backend now returns a message listing
// the registry's ACTUAL contents. The old 400 named "llama-cpp" from a string
// literal, so it stayed accurate only as long as nothing else was ever added.

#include "node/engine_descriptor.hpp"

#include "common/engine_client.hpp"
#include "node/engine_process.hpp"
#include "node/kv_checkpoint_backend.hpp"

#include <httplib.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

namespace fs = std::filesystem;

namespace mm {

namespace {

/// Recursive directory sizing.
///
/// `estimate_inference_vram_mb()` calls fs::file_size() on the path, which sets
/// an error_code on a directory and falls through to a flat 2048 MB. Every
/// multi-shard HF checkpoint and every Soma container therefore sizes
/// identically today, and it is the single scalar the scheduler feeds to
/// nodes_with_available_vram(). This is a live bug on the FALLBACK path too, not
/// only for Soma.
std::uint64_t path_size_bytes(const std::string& path) {
    std::error_code ec;
    if (!fs::exists(path, ec)) return 0;
    if (!fs::is_directory(path, ec)) {
        const auto sz = fs::file_size(path, ec);
        return ec ? 0 : sz;
    }
    std::uint64_t total = 0;
    for (const auto& e : fs::recursive_directory_iterator(path, ec)) {
        if (ec) break;
        std::error_code fe;
        if (e.is_regular_file(fe)) {
            const auto sz = e.file_size(fe);
            if (!fe) total += sz;
        }
    }
    return total;
}

/// One instance each, owned here. EngineDescriptor holds a raw pointer because a
/// descriptor is copied into the registry and the backend is stateless — every
/// call carries its own base_url and path.
LlamaKvBackend& llama_kv() {
    static LlamaKvBackend backend;
    return backend;
}

SomaKvBackend& soma_kv() {
    static SomaKvBackend backend;
    return backend;
}

} // namespace

/// Descriptors are held indirectly so that a pointer returned by find() stays
/// valid across later registrations. With a vector of values, registering a
/// second engine reallocates and dangles every pointer already handed out — and
/// the node caches those pointers in its slot records.
struct EngineRegistry::Impl {
    mutable std::mutex mu;
    std::vector<std::unique_ptr<EngineDescriptor>> engines;
};

EngineRegistry& EngineRegistry::instance() {
    static EngineRegistry reg;
    return reg;
}

EngineRegistry::EngineRegistry() : impl_(std::make_unique<Impl>()) {}

EngineRegistry::~EngineRegistry() = default;

void EngineRegistry::register_engine(EngineDescriptor descriptor) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (auto& e : impl_->engines) {
        if (e->id == descriptor.id) {
            // Update in place. The provisioner re-registers once it resolves a
            // real executable path, and appending instead would leave the
            // placeholder ahead of it in the search order.
            *e = std::move(descriptor);
            return;
        }
    }
    impl_->engines.push_back(std::make_unique<EngineDescriptor>(std::move(descriptor)));
}

const EngineDescriptor* EngineRegistry::find(const std::string& id) const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (const auto& e : impl_->engines) {
        if (e->id == id) return e.get();
    }
    return nullptr;
}

std::vector<std::string> EngineRegistry::ids() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<std::string> out;
    out.reserve(impl_->engines.size());
    for (const auto& e : impl_->engines)
        out.push_back(e->id);
    return out;
}

std::vector<const EngineDescriptor*> EngineRegistry::all() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<const EngineDescriptor*> out;
    out.reserve(impl_->engines.size());
    for (const auto& e : impl_->engines)
        out.push_back(e.get());
    return out;
}

// ── descriptors ──────────────────────────────────────────────────────────────

EngineDescriptor make_soma_descriptor(const std::string& executable) {
    EngineDescriptor d;
    d.id = "soma";
    d.display_name = "Soma (MoE streaming)";
    d.supports_vision = false;
    d.supports_suspend = true;
    // The one capability that distinguishes it from the fallback at this layer:
    // real per-sequence state, so several agents share one engine rather than
    // one process each.
    d.supports_multi_seq = true;

    d.readiness.kind = ReadinessProbe::Kind::HttpHealth;
    d.readiness.http_path = "/health";

    d.build_launch = [executable](const EngineLoadRequest& req) {
        EngineLaunchSpec spec;
        spec.runtime_name = "soma";
        spec.executable = executable;
        spec.port = req.port;
        spec.readiness.kind = ReadinessProbe::Kind::HttpHealth;
        spec.readiness.http_path = "/health";

        spec.args = {"serve",
                     "--model-dir",
                     req.model_path,
                     "--port",
                     std::to_string(req.port),
                     "--host",
                     "127.0.0.1"};
        if (!req.kv_checkpoint_dir.empty()) {
            spec.args.push_back("--kv-dir");
            spec.args.push_back(req.kv_checkpoint_dir);
        }
        return spec;
    };

    d.make_client = [](const std::string& base_url) -> std::unique_ptr<EngineClient> {
        return std::make_unique<SomaEngineClient>(base_url);
    };

    d.kv = &soma_kv();

    d.telemetry_path = "/internal/telemetry";
    d.heat_path = "/internal/heat";

    d.fetch_sequences = [](const std::string& base_url, std::string& out_json) {
        httplib::Client cli(base_url);
        cli.set_connection_timeout(2);
        cli.set_read_timeout(5);
        auto res = cli.Get("/internal/sessions");
        if (!res || res->status != 200) return false;
        out_json = res->body;
        return true;
    };

    d.estimate_footprint =
        [](const EngineLoadRequest& req, ResourceFootprint& out, std::string& error) {
            // The honest version reads `soma plan --json`, which is headers-only and
            // safe on a node that could not host the model. Until that call is wired
            // in, size the directory rather than return a constant — a wrong number
            // that varies with the model beats a constant that does not.
            const auto bytes = path_size_bytes(req.model_path);
            if (bytes == 0) {
                error = "cannot size " + req.model_path;
                return false;
            }
            out.disk_mb = static_cast<std::int64_t>(bytes / (1024 * 1024));
            out.ram_mb = out.disk_mb; // resident upper bound; the plan refines it
            out.vram_mb = 0;          // CPU-only in v1
            return true;
        };

    // Two agents may share one Soma process: per-sequence KV is the entire point
    // of the scheduler. The fallback cannot, which is why this is a descriptor
    // field rather than a global policy.
    //
    // Note what is ABSENT versus llama_launch_compatible(): ctx_size. In
    // llama-server ctx_size is carved per slot at launch, so two agents wanting
    // different context lengths need different processes. In Soma the KV slot is
    // per-sequence and sized on admission, so ctx_size rides the request. That
    // single omission is what lets agents co-locate — and it is also why this is
    // a per-descriptor predicate rather than one shared function.
    //
    // The model itself is not compared here: it is not in RuntimeSettings, and
    // the caller has already matched EngineLoadRequest::model_path before asking.
    d.launch_compatible = [](const RuntimeSettings& a, const RuntimeSettings& b) {
        return a.n_threads == b.n_threads // process-wide pool
               && a.n_threads_http == b.n_threads_http &&
               a.batch_size == b.batch_size // scheduler, not per-seq
               && a.ubatch_size == b.ubatch_size &&
               a.extra_args == b.extra_args; // arbitrary launch flags
    };

    return d;
}

EngineDescriptor make_llama_descriptor(const std::string& executable) {
    EngineDescriptor d;
    d.id = "llama-cpp";
    d.display_name = "llama.cpp";
    d.supports_vision = true;
    d.supports_suspend = true;
    // llama-server's slot state is per-process in practice; the existing
    // checkpoint path hardcodes sequence 0, so a --parallel > 1 slot only ever
    // saves its first sequence. Reporting false here is what stops the scheduler
    // from co-locating agents on that assumption.
    d.supports_multi_seq = false;

    d.readiness.kind = ReadinessProbe::Kind::HttpHealth;
    d.readiness.http_path = "/health";

    d.build_launch = [executable](const EngineLoadRequest& req) {
        EngineLaunchSpec spec;
        spec.runtime_name = "llama-cpp";
        spec.executable = executable;
        spec.port = req.port;
        spec.readiness.kind = ReadinessProbe::Kind::HttpHealth;
        spec.readiness.http_path = "/health";
        spec.args = {
            "-m", req.model_path, "--port", std::to_string(req.port), "--host", "127.0.0.1"};
        if (!req.mmproj_path.empty()) {
            spec.args.push_back("--mmproj");
            spec.args.push_back(req.mmproj_path);
        }
        // Without this llama-server resolves the basename in POST
        // /slots/0?action=save against its own default, so the node writes a
        // checkpoint it will never find again. LlamaKvBackend::save verifies the
        // file appeared for exactly this reason.
        if (!req.kv_checkpoint_dir.empty()) {
            spec.args.push_back("--slot-save-path");
            spec.args.push_back(req.kv_checkpoint_dir);
        }
        return spec;
    };

    d.make_client = [](const std::string& base_url) -> std::unique_ptr<EngineClient> {
        return std::make_unique<LlamaEngineClient>(base_url);
    };

    d.kv = &llama_kv();

    d.estimate_footprint =
        [](const EngineLoadRequest& req, ResourceFootprint& out, std::string& error) {
            const auto bytes = path_size_bytes(req.model_path);
            if (bytes == 0) {
                error = "cannot size " + req.model_path;
                return false;
            }
            out.disk_mb = static_cast<std::int64_t>(bytes / (1024 * 1024));
            out.vram_mb = out.disk_mb;
            out.ram_mb = 0;
            return true;
        };

    // The existing, unit-tested predicate from common/models.hpp — not `false`.
    // llama.cpp agents already share processes today, and answering "never" here
    // would silently spawn one llama-server per agent.
    d.launch_compatible = [](const RuntimeSettings& a, const RuntimeSettings& b) {
        return llama_launch_compatible(a, b);
    };
    return d;
}

} // namespace mm
