// Mantic-Mind — EngineSupervisor: the node's pool of engine subprocesses.
//
// REPLACES SlotManager.
//
// Three properties are load-bearing here, and two of them are new.
//
// 1. LOCK DISCIPLINE. A process is never spawned, stopped, or probed while
//    mutex_ is held. SlotManager already did this for load (via pending_loads_)
//    and it matters more now: the crash watchdog fires from its own thread into
//    on_engine_crash(), which takes mutex_. If unload() held mutex_ while calling
//    EngineProcess::stop() — which joins that watchdog — the two would deadlock.
//    Every mutating path therefore extracts the engine under the lock and does
//    the slow, blocking work after releasing it.
//
// 2. THE WATCHDOG. Nothing polled the child after it reached Ready, so a crashed
//    engine advertised SlotState::Ready until an inference request happened to
//    fail. Now the crash promotes the engine to Error and detaches its agents,
//    so control's next placement decision sees the truth.
//
// 3. ENGINES ARE DESCRIPTORS. There is no `if (backend == "llama-cpp")` in this
//    file. An unknown id fails with a message listing EngineRegistry::ids(),
//    which is accurate by construction rather than by a maintained literal.

#include "node/engine_supervisor.hpp"

#include "common/engine_client.hpp"
#include "common/inference_sizing.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "node/kv_checkpoint_backend.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <utility>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace mm {

namespace {

/// Bind-probe a loopback port. Lifted verbatim in behaviour from
/// SlotManager::test_port_available — it was right, and a port allocator that
/// only consults its own bookkeeping collides with anything else on the host.
bool test_port_available(std::uint16_t port) {
#ifdef _WIN32
    SOCKET sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const int rc = ::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    ::closesocket(sock);
    return rc == 0;
#else
    const int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;
    int option = 1;
    ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const int rc = ::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    ::close(sock);
    return rc == 0;
#endif
}

std::string join_ids(const std::vector<std::string>& ids) {
    std::string out;
    for (const auto& id : ids) {
        if (!out.empty()) out += ", ";
        out += id;
    }
    return out.empty() ? "(none registered)" : out;
}

} // namespace

// ── Lease ─────────────────────────────────────────────────────────────────────

EngineSupervisor::Lease::Lease(EngineSupervisor* owner, SlotId slot_id, EngineClient* client)
    : owner_(owner), slot_id_(std::move(slot_id)), client_(client) {}

EngineSupervisor::Lease::Lease(Lease&& other) noexcept
    : owner_(other.owner_), slot_id_(std::move(other.slot_id_)), client_(other.client_) {
    other.owner_ = nullptr;
    other.client_ = nullptr;
}

EngineSupervisor::Lease& EngineSupervisor::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        reset();
        owner_ = other.owner_;
        slot_id_ = std::move(other.slot_id_);
        client_ = other.client_;
        other.owner_ = nullptr;
        other.client_ = nullptr;
    }
    return *this;
}

EngineSupervisor::Lease::~Lease() {
    reset();
}

void EngineSupervisor::Lease::reset() {
    if (owner_ != nullptr && client_ != nullptr) owner_->release_request(slot_id_);
    owner_ = nullptr;
    client_ = nullptr;
}

// ── construction ──────────────────────────────────────────────────────────────

EngineSupervisor::EngineSupervisor(std::uint16_t port_range_start,
                                   std::uint16_t port_range_end,
                                   int max_slots)
    : port_range_start_(port_range_start), port_range_end_(port_range_end),
      max_slots_(max_slots > 0 ? max_slots : 1) {}

EngineSupervisor::~EngineSupervisor() {
    unload_all(true);
}

void EngineSupervisor::set_log_callback(LogCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    log_cb_ = std::move(cb);
}

void EngineSupervisor::set_kv_checkpoint_dir(const std::string& dir) {
    std::lock_guard<std::mutex> lk(mutex_);
    kv_checkpoint_dir_ = dir;
}

void EngineSupervisor::set_models_dir(const std::string& dir) {
    std::lock_guard<std::mutex> lk(mutex_);
    models_dir_ = dir;
}

// ── load ──────────────────────────────────────────────────────────────────────

std::optional<SlotId> EngineSupervisor::try_attach(const std::string& engine_id,
                                                   const EngineLoadRequest& request,
                                                   const AgentId& agent_id) {
    // Caller holds mutex_.
    for (auto& e : engines_) {
        if (e->state != SlotState::Ready) continue;
        if (e->descriptor == nullptr || e->descriptor->id != engine_id) continue;
        if (e->model_path != request.model_path) continue;
        if (e->mmproj_path != request.mmproj_path) continue;
        // The descriptor decides, not this function. Soma omits ctx_size from
        // its predicate because its KV slot is per-sequence; llama.cpp cannot,
        // because ctx_size is carved per slot at launch.
        if (!e->descriptor->launch_compatible) continue;
        if (!e->descriptor->launch_compatible(e->launch_settings, request.settings)) continue;

        if (!agent_id.empty() &&
            std::find(e->agents.begin(), e->agents.end(), agent_id) == e->agents.end()) {
            e->agents.push_back(agent_id);
        }
        e->last_active_ms = util::now_ms();
        return e->id;
    }
    return std::nullopt;
}

SlotId EngineSupervisor::load(const std::string& engine_id,
                              const EngineLoadRequest& request,
                              const AgentId& agent_id) {
    const EngineDescriptor* descriptor = EngineRegistry::instance().find(engine_id);
    if (descriptor == nullptr) {
        std::lock_guard<std::mutex> lk(mutex_);
        last_error_ = "unknown engine '" + engine_id +
                      "'; registered engines: " + join_ids(EngineRegistry::instance().ids());
        MM_ERROR("EngineSupervisor: {}", last_error_);
        return {};
    }

    std::uint16_t port = 0;
    std::string kv_dir;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (auto existing = try_attach(engine_id, request, agent_id)) {
            MM_INFO(
                "EngineSupervisor: attached agent to existing {} engine {}", engine_id, *existing);
            return *existing;
        }

        const int live = static_cast<int>(
            std::count_if(engines_.begin(), engines_.end(), [](const std::unique_ptr<Engine>& e) {
                // Suspended engines hold no process and no port, so they do not
                // count against capacity. Preserved from SlotManager.
                return e->state != SlotState::Suspended;
            }));
        if (live + pending_loads_ >= max_slots_) {
            last_error_ = "max slots reached";
            return {};
        }

        const auto allocated = allocate_port();
        if (!allocated) {
            last_error_ = "no available ports in range " + std::to_string(port_range_start_) + "-" +
                          std::to_string(port_range_end_);
            MM_ERROR("EngineSupervisor: {}", last_error_);
            return {};
        }
        port = *allocated;
        ++pending_loads_;
        kv_dir = kv_checkpoint_dir_;
    }

    auto engine = std::make_unique<Engine>();
    engine->id = util::generate_uuid();
    engine->descriptor = descriptor;
    engine->port = port;
    engine->model_path = request.model_path;
    engine->mmproj_path = request.mmproj_path;
    engine->launch_settings = request.settings;
    engine->state = SlotState::Loading;
    engine->last_active_ms = util::now_ms();
    engine->effective_ctx_size = request.settings.ctx_size;
    if (!agent_id.empty()) engine->agents.push_back(agent_id);

    EngineLoadRequest spawn_request = request;
    spawn_request.port = port;
    if (spawn_request.kv_checkpoint_dir.empty()) spawn_request.kv_checkpoint_dir = kv_dir;

    // Footprint BEFORE launching. `soma plan --json` reads headers only and
    // allocates nothing, which is the point: a node that could not host the model
    // still gets a real number instead of the flat 2048 MB fallback.
    if (descriptor->estimate_footprint) {
        std::string err;
        if (!descriptor->estimate_footprint(spawn_request, engine->footprint, err)) {
            MM_WARN("EngineSupervisor: footprint estimate failed for {}: {}",
                    spawn_request.model_path,
                    err);
        }
    }

    auto spec = descriptor->build_launch(spawn_request);
    if (spec.readiness.kind == ReadinessProbe::Kind::HttpHealth &&
        spec.readiness.http_path.empty()) {
        spec.readiness = descriptor->readiness;
    }

    const SlotId slot_id = engine->id;
    engine->process = std::make_unique<EngineProcess>();
    engine->process->set_crash_callback([this, slot_id](int code, const std::string& detail) {
        on_engine_crash(slot_id, code, detail);
    });
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (log_cb_) engine->process->set_log_callback(log_cb_);
    }

    // Outside the lock: this blocks for as long as the model takes to load.
    const bool started = engine->process->start(spec);

    if (!started) {
        const std::string why = engine->process->last_error();
        std::lock_guard<std::mutex> lk(mutex_);
        --pending_loads_;
        release_port(port);
        last_error_ = "engine failed to start: " + why;
        MM_ERROR("EngineSupervisor: {}", last_error_);
        return {};
    }

    if (descriptor->verify_capabilities) {
        std::string detail;
        if (!descriptor->verify_capabilities(port, detail)) {
            engine->process->stop();
            std::lock_guard<std::mutex> lk(mutex_);
            --pending_loads_;
            release_port(port);
            last_error_ = "engine capability check failed: " + detail;
            MM_ERROR("EngineSupervisor: {}", last_error_);
            return {};
        }
    }

    if (descriptor->make_client) engine->client = descriptor->make_client(engine->process->url());
    engine->state = SlotState::Ready;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        --pending_loads_;
        engines_.push_back(std::move(engine));
    }
    MM_INFO("EngineSupervisor: {} engine {} ready on port {}", engine_id, slot_id, port);
    return slot_id;
}

// ── crash ─────────────────────────────────────────────────────────────────────

void EngineSupervisor::on_engine_crash(const SlotId& slot_id,
                                       int exit_code,
                                       const std::string& detail) {
    std::vector<AgentId> orphaned;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        for (auto& e : engines_) {
            if (e->id != slot_id) continue;
            // Error, not removal: control has to be able to SEE that this
            // placement died. Deleting the record here would present a crash as
            // an engine that was never there.
            e->state = SlotState::Error;
            orphaned = e->agents;
            e->agents.clear();
            e->client.reset();
            last_error_ =
                "engine " + slot_id + " exited (code " + std::to_string(exit_code) + "): " + detail;
            break;
        }
    }
    MM_ERROR("EngineSupervisor: engine {} crashed with code {} ({}); {} agent(s) detached",
             slot_id,
             exit_code,
             detail,
             orphaned.size());
}

// ── unload / detach ───────────────────────────────────────────────────────────

EngineOpResult EngineSupervisor::unload(const SlotId& slot_id) {
    std::unique_ptr<Engine> victim;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = std::find_if(engines_.begin(),
                               engines_.end(),
                               [&](const std::unique_ptr<Engine>& e) { return e->id == slot_id; });
        if (it == engines_.end()) return {EngineOpStatus::NotFound, "no such engine", "", {}};
        if ((*it)->active_requests > 0) {
            return {EngineOpStatus::Busy,
                    "engine has " + std::to_string((*it)->active_requests) +
                        " in-flight request(s)",
                    "capacity_pressure",
                    {}};
        }
        victim = std::move(*it);
        engines_.erase(it);
        release_port(victim->port);
    }

    // Outside the lock. stop() joins the watchdog thread, and that thread's
    // callback takes mutex_.
    if (victim->process) victim->process->stop();
    return {EngineOpStatus::Ok, "unloaded", "", victim->kv_checkpoint_path};
}

DetachResult EngineSupervisor::detach_agent(const SlotId& slot_id, const AgentId& agent_id) {
    bool unload_now = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = std::find_if(engines_.begin(),
                               engines_.end(),
                               [&](const std::unique_ptr<Engine>& e) { return e->id == slot_id; });
        if (it == engines_.end()) return {EngineOpStatus::NotFound, "no such engine", 0, false};

        auto& agents = (*it)->agents;
        agents.erase(std::remove(agents.begin(), agents.end(), agent_id), agents.end());
        if (!agents.empty()) {
            return {EngineOpStatus::Ok, "detached", static_cast<int>(agents.size()), false};
        }
        unload_now = (*it)->active_requests == 0;
        if (!unload_now) {
            // Last agent gone but a request is still streaming. Leaving the
            // engine up is correct: the response belongs to a request that was
            // accepted, and killing it mid-stream to reclaim a slot is a worse
            // failure than holding the slot a few seconds longer.
            return {EngineOpStatus::Busy, "last agent detached but requests in flight", 0, false};
        }
    }

    const auto result = unload(slot_id);
    return {result.status, result.message, 0, result.ok()};
}

EngineOpResult EngineSupervisor::unload_all(bool force) {
    std::vector<std::unique_ptr<Engine>> victims;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!force) {
            const bool busy =
                std::any_of(engines_.begin(), engines_.end(), [](const std::unique_ptr<Engine>& e) {
                    return e->active_requests > 0;
                });
            if (busy) return {EngineOpStatus::Busy, "requests in flight", "capacity_pressure", {}};
        }
        victims = std::move(engines_);
        engines_.clear();
        used_ports_.clear();
    }
    for (auto& v : victims) {
        if (v->process) v->process->stop();
    }
    return {
        EngineOpStatus::Ok, "unloaded " + std::to_string(victims.size()) + " engine(s)", "", {}};
}

// ── suspend / restore ─────────────────────────────────────────────────────────

EngineOpResult EngineSupervisor::suspend(const SlotId& slot_id) {
    KvCheckpointBackend* kv = nullptr;
    std::string checkpoint_path;
    std::string base_url;
    std::size_t sequence_count = 0;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = std::find_if(engines_.begin(),
                               engines_.end(),
                               [&](const std::unique_ptr<Engine>& e) { return e->id == slot_id; });
        if (it == engines_.end()) return {EngineOpStatus::NotFound, "no such engine", "", {}};
        if ((*it)->state != SlotState::Ready) {
            return {EngineOpStatus::Failed, "engine is not ready", "", {}};
        }
        if ((*it)->active_requests > 0) {
            return {EngineOpStatus::Busy, "requests in flight", "capacity_pressure", {}};
        }

        kv = (*it)->descriptor != nullptr ? (*it)->descriptor->kv : nullptr;
        if (kv == nullptr) {
            return {EngineOpStatus::Unsupported,
                    "engine '" + ((*it)->descriptor ? (*it)->descriptor->id : std::string("?")) +
                        "' has no KV checkpoint backend",
                    "unsupported_content",
                    {}};
        }
        sequence_count = (*it)->agents.size();
        if (sequence_count > 1 && !kv->supports_multi_sequence()) {
            // The bug this refuses to repeat: the current path saves sequence 0
            // and silently discards the rest, so a resumed slot comes back with
            // one agent's context and several agents' expectations.
            return {EngineOpStatus::Unsupported,
                    "engine holds " + std::to_string(sequence_count) +
                        " sequences but its KV backend can only checkpoint sequence 0",
                    "unsupported_content",
                    {}};
        }

        // The extension is the BACKEND's, not the supervisor's: llama.cpp writes
        // llama-server session blobs and Soma writes its own versioned format,
        // and this function must not know which.
        checkpoint_path =
            (fs::path(kv_checkpoint_dir_) / (slot_id + kv->file_extension())).string();
        base_url = (*it)->process ? (*it)->process->url() : std::string{};

        // Suspending, not removed. acquire() requires Ready, so this closes the
        // window without taking the record away from anyone reading slots().
        (*it)->state = SlotState::Suspending;
    }

    // Save BEFORE stopping anything. A failed suspend must be a no-op, not a
    // kill: the earlier ordering stopped the process first and then discovered
    // the checkpoint had not been written, which loses the context it was trying
    // to preserve and the engine along with it.
    std::string err;
    std::error_code ec;
    fs::create_directories(fs::path(checkpoint_path).parent_path(), ec);
    const bool saved = kv->save(base_url, {}, 0, checkpoint_path, err);

    if (!saved) {
        std::lock_guard<std::mutex> lk(mutex_);
        for (auto& e : engines_) {
            if (e->id == slot_id) e->state = SlotState::Ready;
        }
        last_error_ = "KV checkpoint failed: " + err;
        MM_ERROR("EngineSupervisor: suspend of {} failed, engine left running: {}", slot_id, err);
        return {EngineOpStatus::Failed, "KV checkpoint failed: " + err, "internal", {}};
    }

    std::unique_ptr<Engine> engine;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = std::find_if(engines_.begin(),
                               engines_.end(),
                               [&](const std::unique_ptr<Engine>& e) { return e->id == slot_id; });
        if (it == engines_.end()) return {EngineOpStatus::NotFound, "no such engine", "", {}};
        engine = std::move(*it);
        engines_.erase(it);
        release_port(engine->port);
    }

    if (engine->process) engine->process->stop(); // joins the watchdog; not under the lock

    {
        std::lock_guard<std::mutex> lk(mutex_);
        engine->kv_checkpoint_path = checkpoint_path;
        engine->state = SlotState::Suspended;
        engine->process.reset();
        engine->client.reset();
        engine->port = 0;
        engines_.push_back(std::move(engine));
    }
    MM_INFO("EngineSupervisor: engine {} suspended to {}", slot_id, checkpoint_path);
    return {EngineOpStatus::Ok, "suspended", "", checkpoint_path};
}

SlotId EngineSupervisor::restore(const std::string& engine_id,
                                 const EngineLoadRequest& request,
                                 const std::string& kv_checkpoint_path,
                                 const AgentId& agent_id) {
    const EngineDescriptor* descriptor = EngineRegistry::instance().find(engine_id);
    if (descriptor == nullptr) {
        std::lock_guard<std::mutex> lk(mutex_);
        last_error_ = "unknown engine '" + engine_id +
                      "'; registered engines: " + join_ids(EngineRegistry::instance().ids());
        return {};
    }

    // Reject a cross-architecture resume BEFORE spawning anything. The header is
    // read without the payload precisely so this costs nothing, and the
    // difference between failing here and failing after a 60-second model load
    // is the difference between a clear error and a confusing one.
    if (descriptor->kv != nullptr && !kv_checkpoint_path.empty()) {
        KvCheckpointInfo info;
        std::string err;
        if (!descriptor->kv->stat(kv_checkpoint_path, info, err)) {
            std::lock_guard<std::mutex> lk(mutex_);
            last_error_ = "unusable KV checkpoint " + kv_checkpoint_path + ": " + err;
            MM_ERROR("EngineSupervisor: {}", last_error_);
            return {};
        }
    }

    // Drop any suspended record for this checkpoint; load() re-creates it.
    {
        std::lock_guard<std::mutex> lk(mutex_);
        engines_.erase(std::remove_if(engines_.begin(),
                                      engines_.end(),
                                      [&](const std::unique_ptr<Engine>& e) {
                                          return e->state == SlotState::Suspended &&
                                                 e->kv_checkpoint_path == kv_checkpoint_path;
                                      }),
                       engines_.end());
    }

    const SlotId slot_id = load(engine_id, request, agent_id);
    if (slot_id.empty()) return {};

    if (descriptor->kv == nullptr || kv_checkpoint_path.empty()) return slot_id;

    std::string base_url;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        for (auto& e : engines_) {
            if (e->id == slot_id && e->process) base_url = e->process->url();
        }
    }

    std::string err;
    if (!descriptor->kv->restore(base_url, {}, 0, kv_checkpoint_path, err)) {
        // The engine is up and serving; only the warm context was lost. Reporting
        // this as a failed placement would trade a cold start for no engine.
        MM_WARN("EngineSupervisor: KV restore failed for {} ({}); continuing cold", slot_id, err);
        std::lock_guard<std::mutex> lk(mutex_);
        last_error_ = "KV restore failed: " + err;
        return slot_id;
    }

    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& e : engines_) {
        if (e->id == slot_id) e->kv_checkpoint_path = kv_checkpoint_path;
    }
    return slot_id;
}

// ── leases ────────────────────────────────────────────────────────────────────

EngineSupervisor::Lease EngineSupervisor::acquire(const SlotId& slot_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& e : engines_) {
        if (e->id != slot_id) continue;
        if (e->state != SlotState::Ready || !e->client) return {};
        ++e->active_requests;
        e->last_active_ms = util::now_ms();
        return Lease(this, slot_id, e->client.get());
    }
    return {};
}

void EngineSupervisor::release_request(const SlotId& slot_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& e : engines_) {
        if (e->id != slot_id) continue;
        if (e->active_requests > 0) --e->active_requests;
        e->last_active_ms = util::now_ms();
        return;
    }
}

bool EngineSupervisor::touch(const SlotId& slot_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& e : engines_) {
        if (e->id != slot_id) continue;
        e->last_active_ms = util::now_ms();
        return true;
    }
    return false;
}

std::optional<SlotId> EngineSupervisor::find_by_agent(const AgentId& agent_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    for (const auto& e : engines_) {
        if (std::find(e->agents.begin(), e->agents.end(), agent_id) != e->agents.end()) {
            return e->id;
        }
    }
    return std::nullopt;
}

// ── introspection ─────────────────────────────────────────────────────────────

SlotInfo EngineSupervisor::make_slot_info(const Engine& engine) const {
    SlotInfo info;
    info.id = engine.id;
    info.port = engine.port;
    info.model_path = engine.model_path;
    info.mmproj_path = engine.mmproj_path;
    info.vision_enabled = !engine.mmproj_path.empty();
    // From the descriptor, never the literal "llama-cpp". make_slot_info() used
    // to hardcode it, so every slot reported the same backend regardless.
    info.backend =
        engine.descriptor != nullptr ? engine.descriptor->id : engine.fallback_backend_id;
    info.assigned_agent = engine.agents.empty() ? AgentId{} : engine.agents.front();
    info.agent_ids = engine.agents;
    info.state = engine.state;
    // The single-scalar view, for the wire field that still expects one. Reported
    // as the dominant axis rather than as VRAM, because Soma's cost is RAM.
    info.vram_usage_mb =
        engine.footprint.vram_mb > 0 ? engine.footprint.vram_mb : engine.footprint.ram_mb;
    info.last_active_ms = engine.last_active_ms;
    info.kv_cache_path = engine.kv_checkpoint_path;
    info.effective_ctx_size = engine.effective_ctx_size;
    return info;
}

std::vector<SlotInfo> EngineSupervisor::slots() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<SlotInfo> out;
    out.reserve(engines_.size());
    for (const auto& e : engines_)
        out.push_back(make_slot_info(*e));
    return out;
}

std::optional<SlotInfo> EngineSupervisor::find(const SlotId& slot_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    for (const auto& e : engines_) {
        if (e->id == slot_id) return make_slot_info(*e);
    }
    return std::nullopt;
}

std::vector<SequenceInfo> EngineSupervisor::sequences(const SlotId& slot_id) const {
    // Per-sequence state is the engine's to report, and the route that exposes it
    // lands with the telemetry gate. Returning an empty list is the honest answer
    // until then: synthesising one row per attached agent would look like real
    // per-sequence data while carrying none of it — precisely the confusion a
    // request counter already causes. So it is asked of the ENGINE; an engine
    // with no such route (llama.cpp) reports nothing, and nothing is the truth
    // there.
    std::function<bool(const std::string&, std::string&)> fetch;
    std::string base_url;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        for (const auto& e : engines_) {
            if (e->id != slot_id || e->descriptor == nullptr || !e->process) continue;
            fetch = e->descriptor->fetch_sequences;
            base_url = e->process->url();
        }
    }
    if (!fetch || base_url.empty()) return {};

    std::string body;
    if (!fetch(base_url, body)) return {};

    std::vector<SequenceInfo> out;
    try {
        const auto j = nlohmann::json::parse(body);
        std::uint32_t index = 0;
        for (const auto& s : j.value("sessions", nlohmann::json::array())) {
            SequenceInfo info;
            info.index = index++;
            info.agent_id = s.value("conversation", std::string{});
            info.kv_tokens = s.value("kv_tokens", 0u);
            info.position = info.kv_tokens;
            info.determinism = "batched";
            out.push_back(std::move(info));
        }
    } catch (const std::exception& e) {
        MM_WARN("EngineSupervisor: unparseable sequence report from {}: {}", base_url, e.what());
        return {};
    }
    return out;
}

EngineSupervisor::EngineEndpoint EngineSupervisor::endpoint(const SlotId& slot_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    for (const auto& e : engines_) {
        if (e->id != slot_id || !e->process) continue;
        EngineEndpoint out;
        out.base_url = e->process->url();
        if (e->descriptor != nullptr) {
            out.telemetry_path = e->descriptor->telemetry_path;
            out.heat_path = e->descriptor->heat_path;
        }
        return out;
    }
    return {};
}

int EngineSupervisor::available_slot_count() const {
    std::lock_guard<std::mutex> lk(mutex_);
    const int live = static_cast<int>(
        std::count_if(engines_.begin(), engines_.end(), [](const std::unique_ptr<Engine>& e) {
            return e->state != SlotState::Suspended;
        }));
    return std::max(0, max_slots_ - live - pending_loads_);
}

int EngineSupervisor::max_slots() const {
    return max_slots_;
}

ResourceFootprint EngineSupervisor::total_footprint() const {
    std::lock_guard<std::mutex> lk(mutex_);
    ResourceFootprint total;
    for (const auto& e : engines_) {
        if (e->state == SlotState::Suspended) continue; // holds no process
        total.vram_mb += e->footprint.vram_mb;
        total.ram_mb += e->footprint.ram_mb;
        total.disk_mb += e->footprint.disk_mb;
    }
    return total;
}

std::string EngineSupervisor::last_error() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_error_;
}

// ── per-engine runtime status ─────────────────────────────────────────────────

void EngineSupervisor::set_runtime_status(const RuntimeStatus& status) {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& s : runtime_statuses_) {
        if (s.engine_id == status.engine_id) {
            s = status;
            return;
        }
    }
    runtime_statuses_.push_back(status);
}

std::vector<RuntimeStatus> EngineSupervisor::runtime_statuses() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return runtime_statuses_;
}

bool EngineSupervisor::runtime_ready(const std::string& engine_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    for (const auto& s : runtime_statuses_) {
        if (s.engine_id == engine_id) return s.ready;
    }
    return false;
}

#ifdef MM_TESTING
SlotId EngineSupervisor::add_ready_test_engine(const std::string& engine_id,
                                               std::string model_path,
                                               AgentId agent_id,
                                               RuntimeSettings settings,
                                               std::string mmproj_path) {
    // The descriptor may be absent in a test that never registered one; the
    // record still needs to exist, so `backend` falls back to the requested id
    // rather than the record being refused.
    const EngineDescriptor* descriptor = EngineRegistry::instance().find(engine_id);

    auto engine = std::make_unique<Engine>();
    engine->id = util::generate_uuid();
    engine->descriptor = descriptor;
    engine->model_path = std::move(model_path);
    engine->mmproj_path = std::move(mmproj_path);
    if (!agent_id.empty()) engine->agents.push_back(std::move(agent_id));
    engine->launch_settings = std::move(settings);
    engine->client = std::make_unique<LlamaEngineClient>("http://127.0.0.1:0");
    engine->state = SlotState::Ready;
    engine->last_active_ms = util::now_ms();
    engine->effective_ctx_size = effective_llama_server_ctx_tokens(engine->launch_settings);
    engine->fallback_backend_id = engine_id;

    const SlotId id = engine->id;
    std::lock_guard<std::mutex> lk(mutex_);
    engines_.push_back(std::move(engine));
    return id;
}

SlotId EngineSupervisor::add_suspended_test_engine(const std::string& engine_id,
                                                   std::string model_path,
                                                   AgentId agent_id,
                                                   RuntimeSettings settings) {
    const SlotId id = add_ready_test_engine(
        engine_id, std::move(model_path), std::move(agent_id), std::move(settings));
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& e : engines_) {
        if (e->id != id) continue;
        e->state = SlotState::Suspended;
        e->client.reset();
        e->port = 0;
    }
    return id;
}
#endif

// ── ports ─────────────────────────────────────────────────────────────────────

std::optional<std::uint16_t> EngineSupervisor::allocate_port() {
    // Caller holds mutex_.
    for (std::uint32_t candidate = port_range_start_; candidate <= port_range_end_; ++candidate) {
        const auto port = static_cast<std::uint16_t>(candidate);
        if (used_ports_.count(port) != 0) continue;
        if (!test_port_available(port)) continue;
        used_ports_.insert(port);
        return port;
    }
    return std::nullopt;
}

void EngineSupervisor::release_port(std::uint16_t port) {
    used_ports_.erase(port);
}

} // namespace mm
