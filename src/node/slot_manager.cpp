
#include "node/slot_manager.hpp"
#include "node/vllm_runtime.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <utility>

#ifdef _WIN32
#  include <winsock2.h>
#  pragma comment(lib, "ws2_32.lib")
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <unistd.h>
#endif

namespace mm {

SlotManager::SlotLease::SlotLease(SlotManager* manager, SlotId slot_id, RuntimeClient* client)
    : manager_(manager)
    , slot_id_(std::move(slot_id))
    , client_(client)
{}

SlotManager::SlotLease::SlotLease(SlotLease&& other) noexcept
    : manager_(other.manager_)
    , slot_id_(std::move(other.slot_id_))
    , client_(other.client_)
{
    other.manager_ = nullptr;
    other.client_ = nullptr;
}
SlotManager::SlotLease& SlotManager::SlotLease::operator=(SlotLease&& other) noexcept {
    if (this != &other) {
        reset();
        manager_ = other.manager_;
        slot_id_ = std::move(other.slot_id_);
        client_ = other.client_;
        other.manager_ = nullptr;
        other.client_ = nullptr;
    }
    return *this;
}

SlotManager::SlotLease::~SlotLease() {
    reset();
}

void SlotManager::SlotLease::reset() {
    if (manager_ && !slot_id_.empty()) {
        manager_->release_slot_request(slot_id_);
    }
    manager_ = nullptr;
    client_ = nullptr;
    slot_id_.clear();
}

SlotManager::SlotManager(uint16_t port_range_start,
                         uint16_t port_range_end,
                         int max_slots,
                         std::string vllm_server_path,
                         double vllm_gpu_budget)
    : vllm_server_path_(std::move(vllm_server_path))
    , port_range_start_(port_range_start)
    , port_range_end_(port_range_end)
    , max_slots_(max_slots)
    , vllm_gpu_budget_((vllm_gpu_budget > 0.0 && vllm_gpu_budget <= 1.0)
                           ? vllm_gpu_budget : 0.90)
{}

SlotManager::~SlotManager() {
    static_cast<void>(unload_all(true));
}

void SlotManager::set_log_callback(LogCallback cb) {
    std::lock_guard lock(mutex_);
    log_cb_ = std::move(cb);
}

SlotId SlotManager::load_model(const std::string& model_path,
                               VllmSettings vllm_settings,
                               const AgentId& agent_id,
                               bool vision_enabled) {
    // A compatible engine already running (or sleeping) serves this agent too —
    // attach or wake instead of spawning a second process.
    if (auto shared = try_attach_or_wake(model_path, vllm_settings, agent_id,
                                         vision_enabled)) {
        return *shared;
    }

    // Phase 1 (locked): capacity + budget + port reservation. The slow part —
    // spawning the engine and waiting for /health — runs without the lock so
    // status queries and inference on other slots never stall behind a load.
    uint16_t port = 0;
    LogCallback log_cb;
    double vllm_fraction = 0.0;
    {
        std::lock_guard lock(mutex_);
        last_error_.clear();

        // Count active (non-suspended) slots plus loads already in flight.
        int active_count = pending_loads_;
        for (const auto& s : slots_) {
            if (s->state != SlotState::Suspended) ++active_count;
        }
        if (active_count >= max_slots_) {
            MM_WARN("SlotManager: max active slots ({}) reached", max_slots_);
            last_error_ = "max slots reached";
            return {};
        }

        auto granted = reserve_vllm_fraction_locked(vllm_settings.gpu_memory_utilization);
        if (!granted) {
            MM_WARN("SlotManager: {}", last_error_);
            return {};
        }
        vllm_fraction = *granted;
        vllm_settings.gpu_memory_utilization = vllm_fraction;

        auto port_opt = allocate_port();
        if (!port_opt) {
            MM_ERROR("SlotManager: no available ports in range {}-{}",
                     port_range_start_, port_range_end_);
            last_error_ = "no available ports";
            pending_vllm_fraction_ -= vllm_fraction;
            return {};
        }
        port = *port_opt;
        ++pending_loads_;
        log_cb = log_cb_;
    }

    auto slot = std::make_unique<Slot>();
    slot->id             = util::generate_uuid();
    slot->port           = port;
    slot->model_path     = model_path;
    slot->vision_enabled = vision_enabled;
    if (!agent_id.empty()) slot->agents.push_back(agent_id);
    slot->launch_settings = vllm_settings;
    slot->state          = SlotState::Loading;
    slot->last_active_ms = util::now_ms();

    slot->process = std::make_unique<RuntimeProcess>(vllm_server_path_);
    if (log_cb) slot->process->set_log_callback(log_cb);

    MM_INFO("SlotManager: loading vllm model {} on port {} (slot {})",
            model_path, port, slot->id);

    // Phase 2 (unlocked): spawn the engine and wait for health. The slot is
    // private to this thread until it is pushed into slots_.
    bool started = slot->process->start_vllm(model_path, vllm_settings, port);

    // Phase 3 (locked): publish the result.
    std::lock_guard lock(mutex_);
    --pending_loads_;
    pending_vllm_fraction_ -= vllm_fraction;

    if (!started) {
        MM_ERROR("SlotManager: failed to start vLLM for slot {}", slot->id);
        auto proc_err = slot->process->last_error();
        if (!proc_err.empty()) {
            last_error_ = "failed to start vLLM (path=" + vllm_server_path_
                        + "): " + proc_err;
        } else {
            last_error_ = "failed to start vLLM (path=" + vllm_server_path_
                        + "; check model_path/settings)";
        }
        release_port(port);
        return {};
    }

    slot->effective_ctx_size = vllm_settings.max_model_len;
    slot->gpu_mem_fraction = vllm_fraction;
    if (gpu_vram_total_mb_ > 0) {
        slot->vram_usage_mb = static_cast<int64_t>(
            vllm_fraction * static_cast<double>(gpu_vram_total_mb_));
    }

    slot->client = std::make_unique<RuntimeClient>(
        "http://127.0.0.1:" + std::to_string(port));
    slot->state = SlotState::Ready;

    SlotId id = slot->id;
    slots_.push_back(std::move(slot));

    MM_INFO("SlotManager: slot {} ready (model={}, agent={}, port={}, ctx={})",
            id, model_path, agent_id, port, slot->effective_ctx_size);
    return id;
}

SlotOperationResult SlotManager::unload_slot(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);

    auto it = std::find_if(slots_.begin(), slots_.end(),
                           [&](const auto& s) { return s->id == slot_id; });
    if (it == slots_.end()) {
        return {SlotOperationStatus::NotFound, "slot not found"};
    }

    auto& slot = *it;
    if (slot->active_requests > 0) {
        return {SlotOperationStatus::Busy, "slot has active inference requests"};
    }

    MM_INFO("SlotManager: unloading slot {} (port {})", slot->id, slot->port);

    if (slot->process) slot->process->stop();
    release_port(slot->port);

    slots_.erase(it);
    return {SlotOperationStatus::Ok, "unloaded"};
}

DetachResult SlotManager::detach_agent(const SlotId& slot_id,
                                       const AgentId& agent_id) {
    std::lock_guard lock(mutex_);

    auto it = std::find_if(slots_.begin(), slots_.end(),
                           [&](const auto& s) { return s->id == slot_id; });
    if (it == slots_.end()) {
        return {SlotOperationStatus::NotFound, "slot not found", 0, false};
    }

    auto& slot = *it;
    auto& agents = slot->agents;
    agents.erase(std::remove(agents.begin(), agents.end(), agent_id),
                 agents.end());
    const int remaining = static_cast<int>(agents.size());

    if (remaining > 0) {
        MM_INFO("SlotManager: detached agent {} from slot {} ({} agent(s) remain)",
                agent_id, slot_id, remaining);
        return {SlotOperationStatus::Ok, "detached", remaining, false};
    }

    if (slot->active_requests > 0) {
        // Last agent left but inference is still draining; keep the engine
        // up — the scheduler can unload it once requests finish.
        MM_INFO("SlotManager: detached last agent {} from slot {}; slot busy, "
                "deferring unload", agent_id, slot_id);
        return {SlotOperationStatus::Ok, "detached; slot busy", 0, false};
    }

    MM_INFO("SlotManager: detached last agent {} from slot {}; unloading",
            agent_id, slot_id);
    if (slot->process) slot->process->stop();
    if (slot->port != 0) release_port(slot->port);
    slots_.erase(it);
    return {SlotOperationStatus::Ok, "detached; slot unloaded", 0, true};
}

SlotOperationResult SlotManager::suspend_slot(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);

    auto it = std::find_if(slots_.begin(), slots_.end(),
                           [&](const auto& s) { return s->id == slot_id; });
    if (it == slots_.end()) {
        return {SlotOperationStatus::NotFound, "slot not found"};
    }

    auto& slot = *it;
    if (slot->active_requests > 0) {
        return {SlotOperationStatus::Busy, "slot has active inference requests"};
    }
    if (slot->state != SlotState::Ready) {
        return {SlotOperationStatus::Failed, "slot is not ready"};
    }

    slot->state = SlotState::Suspending;
    MM_INFO("SlotManager: suspending slot {} ({} agent(s), first={})",
            slot->id, slot->agents.size(),
            slot->agents.empty() ? std::string{"-"} : slot->agents.front());

    // Sleep mode: offload weights to host RAM, free the GPU, and keep the
    // process alive for a fast wake instead of a full engine restart.
    if (slot->launch_settings.enable_sleep_mode && slot->client &&
        slot->port != 0) {
        bool asleep = false;
        try {
            HttpClient cli("http://127.0.0.1:" + std::to_string(slot->port));
            cli.set_timeouts(5, 120, 30);
            asleep = cli.post("/sleep?level=1", nlohmann::json::object()).ok();
        } catch (const std::exception& e) {
            MM_WARN("SlotManager: sleep request failed for slot {}: {}",
                    slot->id, e.what());
        }
        if (asleep) {
            slot->state = SlotState::Suspended;
            slot->sleeping = true;
            MM_INFO("SlotManager: vLLM slot {} sleeping (weights offloaded "
                    "to host RAM)", slot->id);
            return {SlotOperationStatus::Ok, "sleeping"};
        }
        MM_WARN("SlotManager: sleep failed for slot {}; stopping the engine "
                "instead", slot->id);
    }

    if (slot->process) slot->process->stop();
    release_port(slot->port);

    slot->state = SlotState::Suspended;
    slot->sleeping = false;
    slot->port = 0;
    slot->process.reset();
    slot->client.reset();

    MM_INFO("SlotManager: vLLM slot {} suspended (engine stopped)", slot->id);
    return {SlotOperationStatus::Ok, "suspended"};
}

SlotId SlotManager::restore_slot(const std::string& model_path,
                                 const VllmSettings& vllm_settings,
                                 const AgentId& agent_id,
                                 bool vision_enabled) {
    // Restoring starts a fresh engine anyway (no KV cache to recover), so a
    // compatible running engine serves the restored agent directly — and a
    // compatible sleeping engine wakes far faster than a cold start.
    if (auto shared = try_attach_or_wake(model_path, vllm_settings, agent_id,
                                         vision_enabled)) {
        return *shared;
    }

    uint16_t port = 0;
    LogCallback log_cb;
    VllmSettings vllm_cfg = vllm_settings;
    double vllm_fraction = 0.0;
    {
        std::lock_guard lock(mutex_);
        last_error_.clear();

        int active_count = pending_loads_;
        for (const auto& s : slots_) {
            if (s->state != SlotState::Suspended) ++active_count;
        }
        if (active_count >= max_slots_) {
            MM_WARN("SlotManager: max active slots ({}) reached for restore", max_slots_);
            last_error_ = "max active slots reached";
            return {};
        }

        auto granted = reserve_vllm_fraction_locked(vllm_cfg.gpu_memory_utilization);
        if (!granted) {
            MM_WARN("SlotManager: {}", last_error_);
            return {};
        }
        vllm_fraction = *granted;
        vllm_cfg.gpu_memory_utilization = vllm_fraction;

        auto port_opt = allocate_port();
        if (!port_opt) {
            MM_ERROR("SlotManager: no available ports for restore");
            last_error_ = "no available ports for restore";
            pending_vllm_fraction_ -= vllm_fraction;
            return {};
        }
        port = *port_opt;
        ++pending_loads_;
        log_cb = log_cb_;
    }

    auto slot = std::make_unique<Slot>();
    slot->id             = util::generate_uuid();
    slot->port           = port;
    slot->model_path     = model_path;
    slot->vision_enabled = vision_enabled;
    if (!agent_id.empty()) slot->agents.push_back(agent_id);
    slot->launch_settings = vllm_cfg;
    slot->state          = SlotState::Loading;
    slot->last_active_ms = util::now_ms();
    slot->process = std::make_unique<RuntimeProcess>(vllm_server_path_);
    if (log_cb) slot->process->set_log_callback(log_cb);

    MM_INFO("SlotManager: restoring vllm model {} on port {} (slot {})",
            model_path, port, slot->id);

    bool started = slot->process->start_vllm(model_path, vllm_cfg, port);
    if (!started) {
        MM_ERROR("SlotManager: failed to start vLLM for restore");
        auto proc_err = slot->process->last_error();
        std::lock_guard lock(mutex_);
        --pending_loads_;
        pending_vllm_fraction_ -= vllm_fraction;
        last_error_ = "failed to start vLLM for restore (path=" + vllm_server_path_ + ")"
                    + (proc_err.empty() ? "" : ": " + proc_err);
        release_port(port);
        return {};
    }

    slot->client = std::make_unique<RuntimeClient>(
        "http://127.0.0.1:" + std::to_string(port));
    slot->state = SlotState::Ready;
    slot->effective_ctx_size = vllm_cfg.max_model_len;
    SlotId id = slot->id;

    // Phase 3 (locked): drop the old suspended record and publish the slot.
    std::lock_guard lock(mutex_);
    --pending_loads_;
    pending_vllm_fraction_ -= vllm_fraction;
    slot->gpu_mem_fraction = vllm_fraction;
    if (gpu_vram_total_mb_ > 0) {
        slot->vram_usage_mb = static_cast<int64_t>(
            vllm_fraction * static_cast<double>(gpu_vram_total_mb_));
    }
    if (!agent_id.empty())
        remove_agent_from_suspended_locked(agent_id);
    slots_.push_back(std::move(slot));

    MM_INFO("SlotManager: slot {} restored (agent={})", id, agent_id);
    return id;
}

SlotOperationResult SlotManager::unload_all(bool force) {
    std::lock_guard lock(mutex_);
    if (!force) {
        for (const auto& slot : slots_) {
            if (slot->active_requests > 0) {
                return {SlotOperationStatus::Busy,
                        "one or more slots have active inference requests"};
            }
        }
    }
    for (auto& slot : slots_) {
        if (slot->process)
            slot->process->stop();
        if (slot->port != 0)
            release_port(slot->port);
    }
    slots_.clear();
    used_ports_.clear();
    MM_INFO("SlotManager: all slots unloaded");
    return {SlotOperationStatus::Ok, "unloaded"};
}

std::optional<SlotId> SlotManager::find_slot_by_agent(const AgentId& agent_id) const {
    std::lock_guard lock(mutex_);
    for (const auto& s : slots_) {
        if (s->state != SlotState::Ready) continue;
        if (std::find(s->agents.begin(), s->agents.end(), agent_id)
                != s->agents.end())
            return s->id;
    }
    return std::nullopt;
}

SlotManager::SlotLease SlotManager::acquire_slot(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);
    for (auto& s : slots_) {
        if (s->id == slot_id && s->state == SlotState::Ready && s->client) {
            ++s->active_requests;
            s->last_active_ms = util::now_ms();
            return SlotLease(this, slot_id, s->client.get());
        }
    }
    return {};
}

bool SlotManager::touch_slot(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);
    for (auto& s : slots_) {
        if (s->id == slot_id) {
            s->last_active_ms = util::now_ms();
            return true;
        }
    }
    return false;
}

std::vector<SlotInfo> SlotManager::get_slot_info() const {
    std::lock_guard lock(mutex_);
    std::vector<SlotInfo> result;
    result.reserve(slots_.size());
    for (const auto& s : slots_)
        result.push_back(make_slot_info(*s));
    return result;
}

std::optional<SlotInfo> SlotManager::find_slot(const SlotId& slot_id) const {
    std::lock_guard lock(mutex_);
    for (const auto& s : slots_) {
        if (s->id == slot_id) return make_slot_info(*s);
    }
    return std::nullopt;
}

int SlotManager::available_slot_count() const {
    std::lock_guard lock(mutex_);
    int active = pending_loads_;
    for (const auto& s : slots_) {
        if (s->state != SlotState::Suspended) ++active;
    }
    return max_slots_ - active;
}

int64_t SlotManager::total_vram_usage() const {
    std::lock_guard lock(mutex_);
    int64_t total = 0;
    for (const auto& s : slots_) {
        if (s->state == SlotState::Ready || s->state == SlotState::Loading)
            total += s->vram_usage_mb;
    }
    return total;
}

double SlotManager::vllm_gpu_budget() const {
    std::lock_guard lock(mutex_);
    return vllm_gpu_budget_;
}

double SlotManager::vllm_gpu_fraction_used() const {
    std::lock_guard lock(mutex_);
    return vllm_fraction_allocated_locked();
}

void SlotManager::set_gpu_vram_total_mb(int64_t mb) {
    std::lock_guard lock(mutex_);
    gpu_vram_total_mb_ = mb;
}

void SlotManager::refresh_vllm_metrics() {
    struct Target {
        SlotId   id;
        uint16_t port = 0;
    };
    std::vector<Target> targets;
    {
        std::lock_guard lock(mutex_);
        for (const auto& s : slots_) {
            if (s->state != SlotState::Ready || s->port == 0) continue;
            // Prometheus /metrics scraping here parses vLLM's exposition format;
            targets.push_back({s->id, s->port});
        }
    }

    for (const auto& t : targets) {
        VllmEngineMetrics m;
        try {
            HttpClient cli("http://127.0.0.1:" + std::to_string(t.port));
            cli.set_timeouts(2, 2, 2);
            auto resp = cli.get("/metrics");
            if (resp.ok()) m = parse_vllm_metrics_text(resp.body);
        } catch (const std::exception&) {
            // Engine unreachable — leave metrics invalid.
        }

        std::lock_guard lock(mutex_);
        for (auto& s : slots_) {
            if (s->id != t.id) continue;
            s->engine_metrics_valid = m.valid;
            s->num_requests_running = m.num_requests_running;
            s->num_requests_waiting = m.num_requests_waiting;
            s->kv_cache_usage       = m.kv_cache_usage;
            break;
        }
    }
}

std::string SlotManager::last_error() const {
    std::lock_guard lock(mutex_);
    return last_error_;
}

#ifdef MM_TESTING
SlotId SlotManager::add_ready_test_slot(std::string model_path,
                                        AgentId agent_id,
                                        double gpu_mem_fraction) {
    std::lock_guard lock(mutex_);

    auto slot = std::make_unique<Slot>();
    slot->id = util::generate_uuid();
    slot->model_path = std::move(model_path);
    if (!agent_id.empty()) slot->agents.push_back(std::move(agent_id));
    slot->gpu_mem_fraction = gpu_mem_fraction;
    slot->client = std::make_unique<RuntimeClient>("http://127.0.0.1:0");
    slot->state = SlotState::Ready;
    slot->last_active_ms = util::now_ms();

    SlotId id = slot->id;
    slots_.push_back(std::move(slot));
    return id;
}

bool SlotManager::mark_test_slot_sleeping(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);
    for (auto& s : slots_) {
        if (s->id == slot_id) {
            s->state    = SlotState::Suspended;
            s->sleeping = true;
            return true;
        }
    }
    return false;
}
#endif

void SlotManager::set_vllm_server_path(const std::string& path) {
    std::lock_guard lock(mutex_);
    vllm_server_path_ = path;
}

std::string SlotManager::vllm_server_path() const {
    std::lock_guard lock(mutex_);
    return vllm_server_path_;
}

void SlotManager::release_slot_request(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);
    for (auto& s : slots_) {
        if (s->id == slot_id) {
            if (s->active_requests > 0) --s->active_requests;
            return;
        }
    }
}

// ── Private helpers ─────────────────────────────────────────────────────────

double SlotManager::vllm_fraction_allocated_locked() const {
    double allocated = pending_vllm_fraction_;
    for (const auto& s : slots_) {
        if (s->state != SlotState::Suspended) {
            allocated += s->gpu_mem_fraction;
        }
    }
    return allocated;
}

std::optional<double> SlotManager::reserve_vllm_fraction_locked(double requested) {
    // vLLM pre-allocates --gpu-memory-utilization of total GPU memory at
    // startup, so co-resident engines must split an explicit budget instead
    // of each claiming the default 0.90.
    constexpr double kDefaultVllmGpuFraction = 0.90;
    constexpr double kMinVllmGpuFraction     = 0.05;

    if (requested <= 0.0 || requested > 1.0) requested = kDefaultVllmGpuFraction;

    const double allocated = vllm_fraction_allocated_locked();
    const double remaining = vllm_gpu_budget_ - allocated;
    if (remaining < kMinVllmGpuFraction) {
        last_error_ = "insufficient GPU memory budget for vLLM (allocated="
                    + std::to_string(allocated)
                    + ", budget=" + std::to_string(vllm_gpu_budget_) + ")";
        return std::nullopt;
    }

    const double granted = std::min(requested, remaining);
    if (granted < requested) {
        MM_WARN("SlotManager: clamping vLLM gpu-memory-utilization from {} to {} "
                "(budget={}, already allocated={})",
                requested, granted, vllm_gpu_budget_, allocated);
    }
    pending_vllm_fraction_ += granted;
    return granted;
}

SlotManager::Slot* SlotManager::find_compatible_vllm_slot_locked(
    const std::string& model_path,
    const VllmSettings& settings)
{
    for (auto& s : slots_) {
        if (s->state != SlotState::Ready) continue;
        if (s->model_path != model_path) continue;
        if (!vllm_launch_compatible(s->launch_settings, settings)) continue;
        return s.get();
    }
    return nullptr;
}

std::optional<SlotId> SlotManager::try_attach_or_wake(const std::string& model_path,
                                                      const VllmSettings& settings,
                                                      const AgentId& agent_id,
                                                      bool vision_enabled) {
    SlotId   wake_id;
    uint16_t wake_port = 0;
    {
        std::lock_guard lock(mutex_);

        if (Slot* shared = find_compatible_vllm_slot_locked(model_path, settings)) {
            attach_agent_locked(*shared, agent_id);
            shared->vision_enabled = shared->vision_enabled || vision_enabled;
            remove_agent_from_suspended_locked(agent_id);
            MM_INFO("SlotManager: attached agent {} to shared vLLM slot {} "
                    "(model={}, {} agent(s) attached)",
                    agent_id, shared->id, model_path, shared->agents.size());
            return shared->id;
        }

        for (auto& s : slots_) {
            if (s->state != SlotState::Suspended || !s->sleeping) continue;
            if (s->model_path != model_path) continue;
            if (!vllm_launch_compatible(s->launch_settings, settings)) continue;
            // Waking re-claims the engine's original GPU slice.
            if (vllm_fraction_allocated_locked() + s->gpu_mem_fraction
                    > vllm_gpu_budget_ + 1e-9) {
                MM_INFO("SlotManager: sleeping engine {} matches but waking "
                        "would exceed the GPU budget; skipping", s->id);
                continue;
            }
            // Loading counts against the budget while the wake is in flight.
            s->state = SlotState::Loading;
            wake_id   = s->id;
            wake_port = s->port;
            break;
        }
    }
    if (wake_id.empty()) return std::nullopt;

    MM_INFO("SlotManager: waking sleeping vLLM slot {} (model={})",
            wake_id, model_path);
    bool woke = false;
    try {
        HttpClient cli("http://127.0.0.1:" + std::to_string(wake_port));
        cli.set_timeouts(5, 300, 30);  // weight upload back to GPU can be slow
        woke = cli.post("/wake_up", nlohmann::json::object()).ok();
    } catch (const std::exception& e) {
        MM_WARN("SlotManager: wake_up request failed for slot {}: {}",
                wake_id, e.what());
    }

    std::lock_guard lock(mutex_);
    auto it = std::find_if(slots_.begin(), slots_.end(),
                           [&](const auto& s) { return s->id == wake_id; });
    if (it == slots_.end()) return std::nullopt;
    auto& slot = *it;

    if (!woke) {
        MM_WARN("SlotManager: failed to wake slot {}; discarding the dead engine",
                wake_id);
        if (slot->process) slot->process->stop();
        if (slot->port != 0) release_port(slot->port);
        slots_.erase(it);
        return std::nullopt;  // caller starts a fresh engine
    }

    slot->state    = SlotState::Ready;
    slot->sleeping = false;
    attach_agent_locked(*slot, agent_id);
    slot->vision_enabled = slot->vision_enabled || vision_enabled;
    remove_agent_from_suspended_locked(agent_id);
    MM_INFO("SlotManager: woke vLLM slot {} (model={}, {} agent(s) attached)",
            wake_id, model_path, slot->agents.size());
    return slot->id;
}

void SlotManager::attach_agent_locked(Slot& slot, const AgentId& agent_id) {
    if (agent_id.empty()) return;
    if (std::find(slot.agents.begin(), slot.agents.end(), agent_id)
            != slot.agents.end())
        return;
    slot.agents.push_back(agent_id);
    slot.last_active_ms = util::now_ms();
}

void SlotManager::remove_agent_from_suspended_locked(const AgentId& agent_id) {
    if (agent_id.empty()) return;
    for (auto it = slots_.begin(); it != slots_.end();) {
        auto& slot = *it;
        if (slot->state != SlotState::Suspended) { ++it; continue; }
        auto& agents = slot->agents;
        agents.erase(std::remove(agents.begin(), agents.end(), agent_id),
                     agents.end());
        if (agents.empty()) {
            // A sleeping record still owns a live process and port; a suspended
            if (slot->process) slot->process->stop();
            if (slot->port != 0) release_port(slot->port);
            it = slots_.erase(it);
        } else {
            ++it;
        }
    }
}

std::optional<uint16_t> SlotManager::allocate_port() {
    for (uint16_t p = port_range_start_; p <= port_range_end_; ++p) {
        if (used_ports_.count(p)) continue;
        if (!test_port_available(p)) continue;
        used_ports_.insert(p);
        return p;
    }
    return std::nullopt;
}

void SlotManager::release_port(uint16_t port) {
    used_ports_.erase(port);
}

bool SlotManager::test_port_available(uint16_t port) {
#ifdef _WIN32
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) return false;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    int result = bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    closesocket(sock);
    return result == 0;
#else
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    int result = ::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    close(sock);
    return result == 0;
#endif
}

SlotInfo SlotManager::make_slot_info(const Slot& s) const {
    SlotInfo info;
    info.id              = s.id;
    info.port            = s.port;
    info.model_path      = s.model_path;
    info.vision_enabled  = s.vision_enabled;
    info.assigned_agent  = s.agents.empty() ? AgentId{} : s.agents.front();
    info.agent_ids       = s.agents;
    info.state           = s.state;
    info.sleeping        = s.sleeping;
    info.engine_metrics_valid = s.engine_metrics_valid;
    info.num_requests_running = s.num_requests_running;
    info.num_requests_waiting = s.num_requests_waiting;
    info.kv_cache_usage  = s.kv_cache_usage;
    info.vram_usage_mb   = s.vram_usage_mb;
    info.gpu_mem_fraction = s.gpu_mem_fraction;
    info.last_active_ms  = s.last_active_ms;
    info.effective_ctx_size = s.effective_ctx_size;
    return info;
}

} // namespace mm
