#include "node/slot_manager.hpp"

#include "common/http_client.hpp"
#include "common/inference_sizing.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "node/llama_runtime.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <utility>

#ifdef _WIN32
#  include <winsock2.h>
#  pragma comment(lib, "ws2_32.lib")
#else
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

namespace mm {

namespace {

int64_t projector_file_mb(const std::string& path) {
    if (path.empty()) return 0;
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(path, ec);
    if (ec) return 0;
    return static_cast<int64_t>((bytes + (1024 * 1024 - 1)) / (1024 * 1024));
}

std::string projector_identity(const std::string& path) {
    if (path.empty()) return {};
    namespace fs = std::filesystem;
    std::error_code size_ec;
    std::error_code time_ec;
    const auto size = fs::file_size(path, size_ec);
    const auto modified = fs::last_write_time(path, time_ec);
    std::string identity = util::to_lower(fs::path(path).filename().string());
    if (!size_ec) identity += ":" + std::to_string(size);
    if (!time_ec) identity += ":" + std::to_string(modified.time_since_epoch().count());
    return identity;
}

bool llama_reports_vision(uint16_t port, std::string& detail) {
    try {
        HttpClient client("http://127.0.0.1:" + std::to_string(port));
        client.set_timeouts(5, 15, 15);
        const auto response = client.get("/props");
        if (!response.ok()) {
            detail = "GET /props returned HTTP " + std::to_string(response.status);
            return false;
        }
        const auto props = nlohmann::json::parse(response.body);
        if (!props.contains("modalities") || !props["modalities"].is_object() ||
            !props["modalities"].value("vision", false)) {
            detail = "GET /props did not report modalities.vision=true";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        detail = e.what();
        return false;
    }
}

} // namespace

SlotManager::SlotLease::SlotLease(SlotManager* manager,
                                  SlotId slot_id,
                                  RuntimeClient* client)
    : manager_(manager)
    , slot_id_(std::move(slot_id))
    , client_(client) {}

SlotManager::SlotLease::SlotLease(SlotLease&& other) noexcept
    : manager_(other.manager_)
    , slot_id_(std::move(other.slot_id_))
    , client_(other.client_) {
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
                         int max_slots)
    : port_range_start_(port_range_start)
    , port_range_end_(port_range_end)
    , max_slots_(max_slots) {}

SlotManager::~SlotManager() {
    static_cast<void>(unload_all(true));
}

void SlotManager::set_log_callback(LogCallback cb) {
    std::lock_guard lock(mutex_);
    log_cb_ = std::move(cb);
}

SlotId SlotManager::load_model(const std::string& model_path,
                               const std::string& mmproj_path,
                               RuntimeSettings settings,
                               const AgentId& agent_id) {
    if (auto shared = try_attach(model_path, mmproj_path, settings, agent_id)) {
        return *shared;
    }

    uint16_t port = 0;
    LogCallback log_cb;
    std::string llama_path;
    std::string kv_dir;
    std::string models_dir;
    {
        std::lock_guard lock(mutex_);
        last_error_.clear();

        int active_count = pending_loads_;
        for (const auto& slot : slots_) {
            if (slot->state != SlotState::Suspended) ++active_count;
        }
        if (active_count >= max_slots_) {
            MM_WARN("SlotManager: max active slots ({}) reached", max_slots_);
            last_error_ = "max slots reached";
            return {};
        }

        const auto available_port = allocate_port();
        if (!available_port) {
            MM_ERROR("SlotManager: no available ports in range {}-{}",
                     port_range_start_, port_range_end_);
            last_error_ = "no available ports";
            return {};
        }
        port = *available_port;
        ++pending_loads_;
        log_cb = log_cb_;
        llama_path = llama_server_path_;
        kv_dir = kv_cache_dir_;
        models_dir = models_dir_;
    }

    auto slot = std::make_unique<Slot>();
    slot->id = util::generate_uuid();
    slot->port = port;
    slot->model_path = model_path;
    slot->mmproj_path = mmproj_path;
    if (!agent_id.empty()) slot->agents.push_back(agent_id);
    slot->launch_settings = settings;
    slot->state = SlotState::Loading;
    slot->last_active_ms = util::now_ms();
    slot->vram_usage_mb = estimate_inference_vram_mb(model_path, settings, models_dir)
                        + projector_file_mb(mmproj_path);

    slot->process = std::make_unique<RuntimeProcess>(llama_path);
    if (log_cb) slot->process->set_log_callback(log_cb);

    MM_INFO("SlotManager: loading llama.cpp model {} on port {} (slot {})",
            model_path, port, slot->id);

    bool started = slot->process->start_llama_server(
        model_path, mmproj_path, settings, port, kv_dir);

    if (started && !mmproj_path.empty()) {
        std::string capability_error;
        if (!llama_reports_vision(port, capability_error)) {
            slot->process->stop();
            started = false;
            std::lock_guard lock(mutex_);
            last_error_ =
                "llama-server started with --mmproj but cannot verify multimodal support: "
                + capability_error
                + ". Upgrade the configured llama-server runtime.";
        }
    }

    std::lock_guard lock(mutex_);
    --pending_loads_;

    if (!started) {
        MM_ERROR("SlotManager: failed to start llama-server for slot {}", slot->id);
        const auto process_error = slot->process->last_error();
        if (last_error_.empty()) {
            last_error_ = "failed to start llama-server (path=" + llama_path + ")"
                        + (process_error.empty() ? "" : ": " + process_error);
        }
        release_port(port);
        return {};
    }

    slot->effective_ctx_size = effective_llama_server_ctx_tokens(settings);
    slot->client = std::make_unique<RuntimeClient>(
        "http://127.0.0.1:" + std::to_string(port));
    slot->state = SlotState::Ready;

    const SlotId id = slot->id;
    const int effective_ctx_size = slot->effective_ctx_size;
    slots_.push_back(std::move(slot));

    MM_INFO("SlotManager: slot {} ready (llama.cpp, model={}, agent={}, port={}, ctx={})",
            id, model_path, agent_id, port, effective_ctx_size);
    return id;
}

SlotOperationResult SlotManager::unload_slot(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);

    const auto it = std::find_if(slots_.begin(), slots_.end(),
                                 [&](const auto& slot) { return slot->id == slot_id; });
    if (it == slots_.end()) {
        return {SlotOperationStatus::NotFound, "slot not found", {}};
    }

    auto& slot = *it;
    if (slot->active_requests > 0) {
        return {SlotOperationStatus::Busy, "slot has active inference requests", {}};
    }

    MM_INFO("SlotManager: unloading slot {} (port {})", slot->id, slot->port);
    if (slot->process) slot->process->stop();
    if (slot->port != 0) release_port(slot->port);
    remove_kv_cache_file_locked(*slot);
    slots_.erase(it);
    return {SlotOperationStatus::Ok, "unloaded", {}};
}

DetachResult SlotManager::detach_agent(const SlotId& slot_id,
                                       const AgentId& agent_id) {
    std::lock_guard lock(mutex_);

    const auto it = std::find_if(slots_.begin(), slots_.end(),
                                 [&](const auto& slot) { return slot->id == slot_id; });
    if (it == slots_.end()) {
        return {SlotOperationStatus::NotFound, "slot not found", 0, false};
    }

    auto& slot = *it;
    auto& agents = slot->agents;
    agents.erase(std::remove(agents.begin(), agents.end(), agent_id), agents.end());
    const int remaining = static_cast<int>(agents.size());

    if (remaining > 0) {
        MM_INFO("SlotManager: detached agent {} from slot {} ({} agent(s) remain)",
                agent_id, slot_id, remaining);
        return {SlotOperationStatus::Ok, "detached", remaining, false};
    }

    if (slot->active_requests > 0) {
        MM_INFO("SlotManager: detached last agent {} from slot {}; slot busy, "
                "deferring unload", agent_id, slot_id);
        return {SlotOperationStatus::Ok, "detached; slot busy", 0, false};
    }

    MM_INFO("SlotManager: detached last agent {} from slot {}; unloading",
            agent_id, slot_id);
    if (slot->process) slot->process->stop();
    if (slot->port != 0) release_port(slot->port);
    remove_kv_cache_file_locked(*slot);
    slots_.erase(it);
    return {SlotOperationStatus::Ok, "detached; slot unloaded", 0, true};
}

SlotOperationResult SlotManager::suspend_slot(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);

    const auto it = std::find_if(slots_.begin(), slots_.end(),
                                 [&](const auto& slot) { return slot->id == slot_id; });
    if (it == slots_.end()) {
        return {SlotOperationStatus::NotFound, "slot not found", {}};
    }

    auto& slot = *it;
    if (slot->active_requests > 0) {
        return {SlotOperationStatus::Busy, "slot has active inference requests", {}};
    }
    if (slot->state != SlotState::Ready) {
        return {SlotOperationStatus::Failed, "slot is not ready", {}};
    }

    slot->state = SlotState::Suspending;
    MM_INFO("SlotManager: suspending slot {} ({} agent(s), first={})",
            slot->id, slot->agents.size(),
            slot->agents.empty() ? std::string{"-"} : slot->agents.front());

    namespace fs = std::filesystem;
    const std::string cache_file = slot->id + ".kvbin";
    std::string cache_path = (fs::path(kv_cache_dir_) / cache_file).string();

    if (slot->client && slot->port != 0) {
        bool saved = false;
        try {
            HttpClient client("http://127.0.0.1:" + std::to_string(slot->port));
            client.set_timeouts(5, 120, 30);
            saved = client.post("/slots/0?action=save",
                                nlohmann::json{{"filename", cache_file}}).ok();
        } catch (const std::exception& e) {
            MM_WARN("SlotManager: KV cache save failed for slot {}: {}",
                    slot->id, e.what());
        }
        if (!saved) {
            MM_WARN("SlotManager: KV cache save failed for slot {}; "
                    "suspending without cached context", slot->id);
            cache_path.clear();
        }
    } else {
        cache_path.clear();
    }

    if (slot->process) slot->process->stop();
    if (slot->port != 0) release_port(slot->port);

    slot->kv_cache_path = cache_path;
    slot->state = SlotState::Suspended;
    slot->port = 0;
    slot->process.reset();
    slot->client.reset();

    MM_INFO("SlotManager: llama.cpp slot {} suspended (cache={})",
            slot->id, cache_path.empty() ? std::string{"none"} : cache_path);
    return {SlotOperationStatus::Ok, "suspended", cache_path};
}

SlotId SlotManager::restore_slot(const std::string& model_path,
                                 const std::string& mmproj_path,
                                 const RuntimeSettings& settings,
                                 const std::string& kv_cache_path,
                                 const AgentId& agent_id) {
    if (auto shared = try_attach(model_path, mmproj_path, settings, agent_id)) {
        return *shared;
    }

    uint16_t port = 0;
    LogCallback log_cb;
    std::string llama_path;
    std::string kv_dir;
    std::string models_dir;
    {
        std::lock_guard lock(mutex_);
        last_error_.clear();

        int active_count = pending_loads_;
        for (const auto& slot : slots_) {
            if (slot->state != SlotState::Suspended) ++active_count;
        }
        if (active_count >= max_slots_) {
            MM_WARN("SlotManager: max active slots ({}) reached for restore", max_slots_);
            last_error_ = "max active slots reached";
            return {};
        }

        const auto available_port = allocate_port();
        if (!available_port) {
            MM_ERROR("SlotManager: no available ports for restore");
            last_error_ = "no available ports for restore";
            return {};
        }
        port = *available_port;
        ++pending_loads_;
        log_cb = log_cb_;
        llama_path = llama_server_path_;
        kv_dir = kv_cache_dir_;
        models_dir = models_dir_;
    }

    auto slot = std::make_unique<Slot>();
    slot->id = util::generate_uuid();
    slot->port = port;
    slot->model_path = model_path;
    slot->mmproj_path = mmproj_path;
    if (!agent_id.empty()) slot->agents.push_back(agent_id);
    slot->launch_settings = settings;
    slot->state = SlotState::Loading;
    slot->last_active_ms = util::now_ms();
    slot->kv_cache_path = kv_cache_path;
    slot->vram_usage_mb = estimate_inference_vram_mb(model_path, settings, models_dir)
                        + projector_file_mb(mmproj_path);
    slot->process = std::make_unique<RuntimeProcess>(llama_path);
    if (log_cb) slot->process->set_log_callback(log_cb);

    MM_INFO("SlotManager: restoring llama.cpp model {} on port {} (slot {}, cache={})",
            model_path, port, slot->id, kv_cache_path);

    bool started = slot->process->start_llama_server(
        model_path, mmproj_path, settings, port, kv_dir);
    if (started && !mmproj_path.empty()) {
        std::string capability_error;
        if (!llama_reports_vision(port, capability_error)) {
            slot->process->stop();
            started = false;
            std::lock_guard lock(mutex_);
            last_error_ = "llama-server restore cannot verify multimodal support: "
                        + capability_error
                        + ". Upgrade the configured llama-server runtime.";
        }
    }
    if (!started) {
        MM_ERROR("SlotManager: failed to start llama-server for restore");
        const auto process_error = slot->process->last_error();
        std::lock_guard lock(mutex_);
        --pending_loads_;
        if (last_error_.empty()) {
            last_error_ = "failed to start llama-server for restore (path="
                        + llama_path + ")"
                        + (process_error.empty() ? "" : ": " + process_error);
        }
        release_port(port);
        return {};
    }

    if (!kv_cache_path.empty()) {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (fs::exists(kv_cache_path, ec)) {
            const std::string cache_file = fs::path(kv_cache_path).filename().string();
            try {
                HttpClient client("http://127.0.0.1:" + std::to_string(port));
                client.set_timeouts(5, 120, 30);
                const auto response = client.post(
                    "/slots/0?action=restore",
                    nlohmann::json{{"filename", cache_file}});
                if (response.ok()) {
                    MM_INFO("SlotManager: KV cache restored for slot {}", slot->id);
                } else {
                    MM_WARN("SlotManager: KV cache restore failed (HTTP {}), "
                            "proceeding without cached context", response.status);
                }
            } catch (const std::exception& e) {
                MM_WARN("SlotManager: KV cache restore request failed: {}", e.what());
            }
        } else {
            MM_WARN("SlotManager: KV cache file not found: {}", kv_cache_path);
        }
    }

    slot->client = std::make_unique<RuntimeClient>(
        "http://127.0.0.1:" + std::to_string(port));
    slot->state = SlotState::Ready;
    // The checkpoint belongs to the suspended record and is removed when that
    // record is retired below. Do not advertise a stale path on the live slot.
    slot->kv_cache_path.clear();
    slot->effective_ctx_size = effective_llama_server_ctx_tokens(settings);
    const SlotId id = slot->id;

    std::lock_guard lock(mutex_);
    --pending_loads_;
    if (!agent_id.empty()) remove_agent_from_suspended_locked(agent_id);
    slots_.push_back(std::move(slot));

    MM_INFO("SlotManager: llama.cpp slot {} restored (agent={})", id, agent_id);
    return id;
}

SlotOperationResult SlotManager::unload_all(bool force) {
    std::lock_guard lock(mutex_);
    if (!force) {
        for (const auto& slot : slots_) {
            if (slot->active_requests > 0) {
                return {SlotOperationStatus::Busy,
                        "one or more slots have active inference requests", {}};
            }
        }
    }
    for (auto& slot : slots_) {
        if (slot->process) slot->process->stop();
        if (slot->port != 0) release_port(slot->port);
        remove_kv_cache_file_locked(*slot);
    }
    slots_.clear();
    used_ports_.clear();
    MM_INFO("SlotManager: all slots unloaded");
    return {SlotOperationStatus::Ok, "unloaded", {}};
}

std::optional<SlotId> SlotManager::find_slot_by_agent(const AgentId& agent_id) const {
    std::lock_guard lock(mutex_);
    for (const auto& slot : slots_) {
        if (slot->state != SlotState::Ready) continue;
        if (std::find(slot->agents.begin(), slot->agents.end(), agent_id)
            != slot->agents.end()) {
            return slot->id;
        }
    }
    return std::nullopt;
}

SlotManager::SlotLease SlotManager::acquire_slot(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);
    for (auto& slot : slots_) {
        if (slot->id == slot_id && slot->state == SlotState::Ready && slot->client) {
            ++slot->active_requests;
            slot->last_active_ms = util::now_ms();
            return SlotLease(this, slot_id, slot->client.get());
        }
    }
    return {};
}

bool SlotManager::touch_slot(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);
    for (auto& slot : slots_) {
        if (slot->id == slot_id) {
            slot->last_active_ms = util::now_ms();
            return true;
        }
    }
    return false;
}

std::vector<SlotInfo> SlotManager::get_slot_info() const {
    std::lock_guard lock(mutex_);
    std::vector<SlotInfo> result;
    result.reserve(slots_.size());
    for (const auto& slot : slots_) result.push_back(make_slot_info(*slot));
    return result;
}

std::optional<SlotInfo> SlotManager::find_slot(const SlotId& slot_id) const {
    std::lock_guard lock(mutex_);
    for (const auto& slot : slots_) {
        if (slot->id == slot_id) return make_slot_info(*slot);
    }
    return std::nullopt;
}

int SlotManager::available_slot_count() const {
    std::lock_guard lock(mutex_);
    int active = pending_loads_;
    for (const auto& slot : slots_) {
        if (slot->state != SlotState::Suspended) ++active;
    }
    return max_slots_ - active;
}

int64_t SlotManager::total_vram_usage() const {
    std::lock_guard lock(mutex_);
    int64_t total = 0;
    for (const auto& slot : slots_) {
        if (slot->state == SlotState::Ready || slot->state == SlotState::Loading) {
            total += slot->vram_usage_mb;
        }
    }
    return total;
}

std::string SlotManager::last_error() const {
    std::lock_guard lock(mutex_);
    return last_error_;
}

#ifdef MM_TESTING
SlotId SlotManager::add_ready_test_slot(std::string model_path,
                                        AgentId agent_id,
                                        RuntimeSettings settings,
                                        std::string mmproj_path) {
    std::lock_guard lock(mutex_);

    auto slot = std::make_unique<Slot>();
    slot->id = util::generate_uuid();
    slot->model_path = std::move(model_path);
    slot->mmproj_path = std::move(mmproj_path);
    if (!agent_id.empty()) slot->agents.push_back(std::move(agent_id));
    slot->launch_settings = std::move(settings);
    slot->client = std::make_unique<RuntimeClient>("http://127.0.0.1:0");
    slot->state = SlotState::Ready;
    slot->last_active_ms = util::now_ms();
    slot->effective_ctx_size = effective_llama_server_ctx_tokens(slot->launch_settings);

    const SlotId id = slot->id;
    slots_.push_back(std::move(slot));
    return id;
}
#endif

void SlotManager::set_llama_server_path(const std::string& path) {
    std::lock_guard lock(mutex_);
    if (!path.empty()) llama_server_path_ = path;
}

std::string SlotManager::llama_server_path() const {
    std::lock_guard lock(mutex_);
    return llama_server_path_;
}

void SlotManager::set_kv_cache_dir(const std::string& dir) {
    std::lock_guard lock(mutex_);
    if (!dir.empty()) kv_cache_dir_ = dir;
}

void SlotManager::set_models_dir(const std::string& dir) {
    std::lock_guard lock(mutex_);
    models_dir_ = dir;
}

void SlotManager::release_slot_request(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);
    for (auto& slot : slots_) {
        if (slot->id == slot_id) {
            if (slot->active_requests > 0) --slot->active_requests;
            return;
        }
    }
}

std::optional<SlotId> SlotManager::try_attach(const std::string& model_path,
                                              const std::string& mmproj_path,
                                              const RuntimeSettings& settings,
                                              const AgentId& agent_id) {
    std::lock_guard lock(mutex_);
    for (auto& slot : slots_) {
        if (slot->state != SlotState::Ready) continue;
        if (normalize_llama_model_path(slot->model_path)
            != normalize_llama_model_path(model_path)) {
            continue;
        }
        if (normalize_llama_model_path(slot->mmproj_path)
            != normalize_llama_model_path(mmproj_path)) {
            continue;
        }
        if (!llama_launch_compatible(slot->launch_settings, settings)) continue;

        attach_agent_locked(*slot, agent_id);
        const SlotId id = slot->id;
        const auto agent_count = slot->agents.size();
        remove_agent_from_suspended_locked(agent_id);
        MM_INFO("SlotManager: attached agent {} to shared llama.cpp slot {} "
                "(model={}, {} agent(s) attached)",
                agent_id, id, model_path, agent_count);
        return id;
    }
    return std::nullopt;
}

void SlotManager::remove_kv_cache_file_locked(const Slot& slot) const {
    if (slot.kv_cache_path.empty()) return;
    std::error_code ec;
    std::filesystem::remove(slot.kv_cache_path, ec);
}

void SlotManager::attach_agent_locked(Slot& slot, const AgentId& agent_id) {
    if (agent_id.empty()) return;
    if (std::find(slot.agents.begin(), slot.agents.end(), agent_id)
        != slot.agents.end()) {
        return;
    }
    slot.agents.push_back(agent_id);
    slot.last_active_ms = util::now_ms();
}

void SlotManager::remove_agent_from_suspended_locked(const AgentId& agent_id) {
    if (agent_id.empty()) return;
    for (auto it = slots_.begin(); it != slots_.end();) {
        auto& slot = *it;
        if (slot->state != SlotState::Suspended) {
            ++it;
            continue;
        }
        auto& agents = slot->agents;
        agents.erase(std::remove(agents.begin(), agents.end(), agent_id), agents.end());
        if (agents.empty()) {
            if (slot->process) slot->process->stop();
            if (slot->port != 0) release_port(slot->port);
            remove_kv_cache_file_locked(*slot);
            it = slots_.erase(it);
        } else {
            ++it;
        }
    }
}

std::optional<uint16_t> SlotManager::allocate_port() {
    for (uint32_t candidate = port_range_start_; candidate <= port_range_end_; ++candidate) {
        const auto port = static_cast<uint16_t>(candidate);
        if (used_ports_.count(port)) continue;
        if (!test_port_available(port)) continue;
        used_ports_.insert(port);
        return port;
    }
    return std::nullopt;
}

void SlotManager::release_port(uint16_t port) {
    used_ports_.erase(port);
}

bool SlotManager::test_port_available(uint16_t port) {
#ifdef _WIN32
    SOCKET socket_handle = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket_handle == INVALID_SOCKET) return false;

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    const int result = bind(socket_handle,
                            reinterpret_cast<sockaddr*>(&address),
                            sizeof(address));
    closesocket(socket_handle);
    return result == 0;
#else
    const int socket_handle = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_handle < 0) return false;

    int option = 1;
    setsockopt(socket_handle, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    const int result = ::bind(socket_handle,
                              reinterpret_cast<sockaddr*>(&address),
                              sizeof(address));
    close(socket_handle);
    return result == 0;
#endif
}

SlotInfo SlotManager::make_slot_info(const Slot& slot) const {
    SlotInfo info;
    info.id = slot.id;
    info.port = slot.port;
    info.model_path = slot.model_path;
    info.mmproj_path = slot.mmproj_path;
    info.vision_enabled = !slot.mmproj_path.empty();
    info.projector_identity = projector_identity(slot.mmproj_path);
    info.backend = "llama-cpp";
    info.assigned_agent = slot.agents.empty() ? AgentId{} : slot.agents.front();
    info.agent_ids = slot.agents;
    info.state = slot.state;
    info.vram_usage_mb = slot.vram_usage_mb;
    info.last_active_ms = slot.last_active_ms;
    info.kv_cache_path = slot.kv_cache_path;
    info.effective_ctx_size = slot.effective_ctx_size;
    return info;
}

} // namespace mm
