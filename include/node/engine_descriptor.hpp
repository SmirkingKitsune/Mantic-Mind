#pragma once

// Mantic-Mind — the engine descriptor registry.
//
// THE central abstraction of the supervision rebuild. llama.cpp and Soma become
// two descriptors, not two code paths.
//
// What this replaces, concretely:
//
//   * `backend != "llama-cpp" -> 400`, hardcoded twice (node_api_server.cpp:449
//     and :911). Duplicated string literals are how the second one gets missed.
//     Now one registry lookup, and the 400 lists what is actually registered.
//   * `LlamaRuntimeStatus` as a single scalar on NodeState, 7 Llama*Callback
//     types, and 6 /api/node/runtime/llama/* routes — a surface that silently
//     assumes exactly one runtime exists.
//   * build_llama_server_args() called directly from RuntimeProcess.
//   * `SlotInfo::backend` hardcoded to "llama-cpp" in make_slot_info().

#include "common/engine_config.hpp"
#include "common/footprint.hpp"
#include "common/models.hpp"
#include "node/engine_process.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mm {

class EngineClient;
class KvCheckpointBackend;

/// What the node was asked to load.
struct EngineLoadRequest {
    std::string model_path; ///< file (fallback) or converted dir (Soma)
    std::string mmproj_path;
    RuntimeSettings settings{};
    std::optional<VllmEngineConfig> vllm;
    std::string served_model_name;
    std::string ray_address; ///< non-empty for a control-owned Ray group
    std::string host_ip;     ///< private address advertised to vLLM/Ray
    std::vector<int> gpu_devices;
    std::string kv_checkpoint_dir;
    std::uint16_t port = 0;
};

/// One engine kind. A struct of callbacks resolved once at node startup.
/// ResourceFootprint lives in common/footprint.hpp — control needs it too.
struct EngineDescriptor {
    std::string id; ///< "llama-cpp" | "soma" — the wire value
    std::string display_name;

    /// Turn a load request into argv + env. For llama.cpp this wraps the
    /// existing, unit-tested build_llama_server_args(); for Soma it emits the
    /// --model-dir / --ram-budget / --kv-slots family. Kept as a pure function
    /// for the same reason the llama one is: it is the part worth testing
    /// without spawning anything.
    std::function<EngineLaunchSpec(const EngineLoadRequest&)> build_launch;

    ReadinessProbe readiness{};

    std::function<std::unique_ptr<EngineClient>(const std::string& base_url)> make_client;

    /// Owns the wire format for save/restore. llama.cpp's is
    /// POST /slots/0?action=save; Soma's is per-sequence and versioned.
    KvCheckpointBackend* kv = nullptr;

    /// Post-start probe. llama.cpp checks GET /props for modalities.vision when
    /// --mmproj was passed. Optional; absent means "nothing to verify".
    std::function<bool(std::uint16_t port, std::string& detail)> verify_capabilities;

    /// Footprint source.
    ///
    /// Soma reads `soma plan --json` — headers only, allocates nothing, safe on
    /// a node that could not host the model. The fallback estimates from file
    /// size, but RECURSIVELY over directories: the current
    /// estimate_inference_vram_mb() calls fs::file_size() on the path, which
    /// errors on a directory and silently returns a flat 2048 MB. Every
    /// multi-shard HF model sizes identically today.
    std::function<bool(const EngineLoadRequest&, ResourceFootprint&, std::string& error)>
        estimate_footprint;

    /// Whether two agents may share one live engine process.
    std::function<bool(const EngineLoadRequest& a, const EngineLoadRequest& b)> launch_compatible;

    /// Is this model reference loadable by THIS engine, on THIS node?
    ///
    /// llama.cpp wants a node-local GGUF **file**; Soma wants a converted
    /// container **directory** with a `container_meta.json` in it. Both are real
    /// preconditions and neither is the other's.
    ///
    /// It lives here because it used to live in the handler, ungated: the
    /// load-model route validated `backend` against this registry at the top and
    /// dispatched on it at the bottom, and in between ran llama.cpp's
    /// `is_regular_file` check against every request. A Soma container is a
    /// directory, so it could never load — the dispatch that would have sent it
    /// to the right engine sat twenty lines below the check that refused it.
    /// That was defect D8, and it is the same shape as every other thing this
    /// descriptor exists to hold: knowledge about one engine, written down where
    /// only that engine reads it.
    ///
    /// Absent means "nothing to verify", which is a legitimate answer for an
    /// engine that resolves its own references.
    std::function<bool(const std::string& model_ref, std::string& detail)> validate_model_ref;

    /// The engine's own telemetry paths, relative to its base URL.
    ///
    /// DATA, not a code path: the node proxies whatever is here and never learns
    /// that Soma calls it /internal/telemetry. Empty means the engine publishes
    /// none, which is the honest answer for llama.cpp rather than a route that
    /// 404s halfway down the chain.
    std::string telemetry_path; ///< SSE
    std::string heat_path;      ///< snapshot

    /// Live per-sequence state as JSON, for GET /v1/engines/{id}/slots.
    ///
    /// Optional; absent means the engine cannot report it, and the supervisor
    /// returns an empty list rather than synthesising one row per attached agent.
    /// A fabricated row looks like per-sequence data while carrying none, which
    /// is the confusion a bare request counter already causes.
    std::function<bool(const std::string& base_url, std::string& out_json)> fetch_sequences;

    bool supports_vision = false;
    bool supports_suspend = false;
    bool supports_multi_seq = false; ///< true for Soma: real per-sequence state
};

/// Every engine this node can run. Populated once at startup.
/// The two engines this node knows how to run, as DATA.
///
/// `executable` is passed in rather than discovered, because where the binary
/// lives is deployment configuration and neither descriptor should be reading
/// the environment behind the caller's back.
EngineDescriptor make_soma_descriptor(const std::string& executable);
EngineDescriptor make_llama_descriptor(const std::string& executable);
EngineDescriptor make_vllm_descriptor(const std::string& executable,
                                      const std::string& hf_cache_dir = {});

class EngineRegistry {
public:
    static EngineRegistry& instance();
    ~EngineRegistry();

    EngineRegistry(const EngineRegistry&) = delete;
    EngineRegistry& operator=(const EngineRegistry&) = delete;

    /// Replaces any descriptor already registered under the same id, so a
    /// re-registration (a provisioner resolving a new executable path) updates
    /// rather than shadowing.
    void register_engine(EngineDescriptor descriptor);

    /// nullptr when unknown. Callers turn that into a 400 whose body lists
    /// ids() — so the error is accurate by construction rather than by a
    /// hand-maintained literal.
    ///
    /// The returned pointer stays valid across later registrations: descriptors
    /// are held indirectly for exactly that reason. A vector of values would
    /// dangle every previously-handed-out pointer on the next push_back.
    const EngineDescriptor* find(const std::string& id) const;

    std::vector<std::string> ids() const;
    std::vector<const EngineDescriptor*> all() const;

private:
    EngineRegistry();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// RuntimeStatus moved to common/engine_config.hpp — control holds a vector of
// them on NodeInfo now, and a per-engine status that only the node could name
// was the reason control could see llama.cpp's health and no other engine's.
// Reachable from here unchanged: this header includes it transitively.

} // namespace mm
