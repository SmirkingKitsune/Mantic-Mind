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
    std::function<bool(const RuntimeSettings& a, const RuntimeSettings& b)> launch_compatible;

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

/// Per-runtime provisioning/health status, keyed by engine id.
///
/// Generalizes LlamaRuntimeStatus, which was a single scalar field on NodeState.
struct RuntimeStatus {
    std::string engine_id;
    std::string status; ///< resolved | ready | building | error | absent
    std::string executable_path;
    std::string version;
    std::string variant;
    std::string last_error;
    bool ready = false;
};

} // namespace mm
