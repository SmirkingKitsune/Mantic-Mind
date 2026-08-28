#pragma once

#include "common/engine_config.hpp"
#include "common/models.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <cstdint>
#include <vector>

namespace mm {

class NodeState;
class EngineSupervisor;
class HttpServer;
class ModelStore;
class NodeEngineManager;
class RayController;

// Hosts the node REST API.
// Most endpoints require "Authorization: Bearer <node-api-key>".
// The /pair-request and /pair-complete endpoints are unauthenticated.
class NodeApiServer {
public:
    NodeApiServer(NodeState& state,
                  EngineSupervisor& engines,
                  std::string control_url = {},
                  std::string pairing_key = {});
    ~NodeApiServer();

    // Registers all routes and starts listening.  Blocks until stop() called.
    bool listen(uint16_t port);

    /// Stop accepting work: the listener comes down and /api/node/infer starts
    /// refusing. Does NOT wait for work already running — see drain_workers().
    void stop();

    /// Join the inference workers.
    ///
    /// Split from stop() because the two have to happen on either side of the
    /// engines being killed. /api/node/infer answers on a chunked SSE provider
    /// while the actual generation runs on its own thread, and that thread holds
    /// a lease into EngineSupervisor and captures this server. Detached, as it
    /// used to be, nothing waited for it: shutdown stopped the listener, unloaded
    /// every engine, and returned from main while the worker was still writing
    /// through a freed EngineClient into a destroyed NodeState.
    ///
    /// Joining alone is not enough either — an untouched stream ends when the
    /// model stops generating, which can be minutes. The caller therefore kills
    /// the engine children first (EngineSupervisor::stop_processes()), which
    /// fails every outstanding request in about as long as a socket takes to
    /// close, and only then drains.
    ///
    /// Safe to call more than once; safe to call with nothing in flight.
    void drain_workers();

    using RuntimeLogsProvider = std::function<std::vector<std::string>(int tail)>;
    using RememberApiKeyCallback = std::function<void(const std::string& key)>;
    using LlamaProvisionCallback = std::function<LlamaRuntimeStatus()>;
    // accelerator is empty for the current target, or an explicit release
    // alternative selected from LlamaRuntimeStatus.
    using LlamaUpdateCallback = std::function<LlamaRuntimeStatus(const std::string& accelerator)>;
    using LlamaSwitchCallback = std::function<LlamaRuntimeStatus(const std::string& variant)>;
    using LlamaCheckUpdateCallback = std::function<LlamaRuntimeStatus()>;
    using LlamaDiagnoseCallback = std::function<LlamaRuntimeStatus()>;
    using LlamaRecoveryCallback =
        std::function<LlamaRuntimeStatus(const std::string& action,
                                         const std::string& variant)>;
    void set_runtime_logs_provider(RuntimeLogsProvider provider);
    void set_remember_api_key_callback(RememberApiKeyCallback callback);
    void set_llama_provision_callback(LlamaProvisionCallback callback);
    void set_llama_update_callback(LlamaUpdateCallback callback);
    void set_llama_switch_callback(LlamaSwitchCallback callback);
    void set_llama_check_update_callback(LlamaCheckUpdateCallback callback);
    void set_llama_diagnose_callback(LlamaDiagnoseCallback callback);
    void set_llama_recovery_callback(LlamaRecoveryCallback callback);
    // Local model cache: control-transferred models + LRU eviction. Optional;
    // when unset the model transfer/receive endpoints report unavailable.
    void set_model_store(ModelStore* store);

    // ── Cluster engine configuration ──────────────────────────────────────────
    // The manager answers "what do I run and am I conforming"; the callback is
    // how a pushed config is APPLIED, because applying it also updates NodeState
    // and starts a background provision that the server must not own.
    //
    // Both optional: unset, the engine routes report unavailable rather than
    // half-working. A node that cannot apply a config should say so to the
    // master that pushed it, not accept it and quietly do nothing.
    void set_engine_manager(NodeEngineManager* manager);
    using EngineConfigCallback = std::function<void(const ClusterEngineConfig&)>;
    void set_engine_config_callback(EngineConfigCallback callback);
    void set_ray_controller(RayController* controller);

private:
    /// One in-flight /api/node/infer generation. Its `finished` flag lets a
    /// later request harvest it: a node that served a million requests must not
    /// be holding a million joinable threads.
    struct InferWorker;

    std::mutex workers_mu_;
    std::vector<std::shared_ptr<InferWorker>> workers_;
    std::atomic<bool> draining_{false};

    NodeState&        state_;
    EngineSupervisor& engines_;
    ModelStore*    model_store_ = nullptr;
    NodeEngineManager* engine_manager_ = nullptr;
    EngineConfigCallback engine_config_cb_;
    RayController* ray_controller_ = nullptr;
    std::string    control_url_;
    std::string    pairing_key_;
    std::unique_ptr<HttpServer> server_;
    RuntimeLogsProvider runtime_logs_provider_;
    RememberApiKeyCallback remember_api_key_cb_;
    LlamaProvisionCallback llama_provision_cb_;
    LlamaUpdateCallback llama_update_cb_;
    LlamaSwitchCallback llama_switch_cb_;
    LlamaCheckUpdateCallback llama_check_update_cb_;
    LlamaDiagnoseCallback llama_diagnose_cb_;
    LlamaRecoveryCallback llama_recovery_cb_;

    void register_routes();
    bool check_auth(const std::string& auth_header);
};

} // namespace mm
