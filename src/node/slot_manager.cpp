#include "node/slot_manager.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>

#ifdef _WIN32
#  include <winsock2.h>
#  pragma comment(lib, "ws2_32.lib")
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace mm {

SlotManager::SlotManager(std::string llama_server_path,
                         uint16_t port_range_start,
                         uint16_t port_range_end,
                         int max_slots,
                         std::string kv_cache_dir)
    : llama_server_path_(std::move(llama_server_path))
    , port_range_start_(port_range_start)
    , port_range_end_(port_range_end)
    , max_slots_(max_slots)
    , kv_cache_dir_(std::move(kv_cache_dir))
{
    std::error_code ec;
    fs::create_directories(kv_cache_dir_, ec);
}

SlotManager::~SlotManager() {
    unload_all();
}

void SlotManager::set_log_callback(LogCallback cb) {
    std::lock_guard lock(mutex_);
    log_cb_ = std::move(cb);
}

SlotId SlotManager::load_model(const std::string& model_path,
                               LlamaSettings settings,
                               const AgentId& agent_id) {
    std::lock_guard lock(mutex_);
    last_error_.clear();

    // Count only active (non-suspended) slots against the limit,
    // matching restore_slot() and available_slot_count().
    int active_count = 0;
    for (const auto& s : slots_) {
        if (s->state != SlotState::Suspended) ++active_count;
    }
    if (active_count >= max_slots_) {
        MM_WARN("SlotManager: max active slots ({}) reached", max_slots_);
        last_error_ = "max slots reached";
        return {};
    }

    auto port_opt = allocate_port();
    if (!port_opt) {
        MM_ERROR("SlotManager: no available ports in range {}-{}",
                 port_range_start_, port_range_end_);
        last_error_ = "no available ports";
        return {};
    }
    uint16_t port = *port_opt;

    auto slot = std::make_unique<Slot>();
    slot->id             = util::generate_uuid();
    slot->port           = port;
    slot->model_path     = model_path;
    slot->assigned_agent = agent_id;
    slot->state          = SlotState::Loading;
    slot->vram_usage_mb  = estimate_vram_mb(model_path);
    slot->last_active_ms = util::now_ms();

    slot->process = std::make_unique<LlamaServerProcess>(llama_server_path_);
    if (log_cb_) slot->process->set_log_callback(log_cb_);

    MM_INFO("SlotManager: loading model {} on port {} (slot {})",
            model_path, port, slot->id);

    // Retry loop: on start failure, halve ctx_size (min 2048, max 3 retries).
    constexpr int kMinCtxSize    = 2048;
    constexpr int kMaxRetries    = 3;
    const int original_ctx_size  = settings.ctx_size;
    bool started = false;

    for (int attempt = 0; attempt <= kMaxRetries; ++attempt) {
        if (attempt > 0) {
            int reduced = settings.ctx_size / 2;
            if (reduced < kMinCtxSize) reduced = kMinCtxSize;
            if (reduced == settings.ctx_size) break; // can't reduce further
            MM_WARN("SlotManager: reducing ctx_size from {} to {} and retrying (attempt {}/{})",
                    settings.ctx_size, reduced, attempt, kMaxRetries);
            settings.ctx_size = reduced;

            // Re-create process for retry
            slot->process = std::make_unique<LlamaServerProcess>(llama_server_path_);
            if (log_cb_) slot->process->set_log_callback(log_cb_);
        }

        if (slot->process->start(model_path, settings, port)) {
            started = true;
            break;
        }
    }

    if (!started) {
        if (settings.ctx_size != original_ctx_size) {
            MM_ERROR("SlotManager: all ctx_size reduction attempts exhausted for slot {} "
                     "(tried {} down to {})", slot->id, original_ctx_size, settings.ctx_size);
        } else {
            MM_ERROR("SlotManager: failed to start llama-server for slot {}", slot->id);
        }
        auto proc_err = slot->process->last_error();
        if (!proc_err.empty()) {
            last_error_ = "failed to start llama-server (path=" + llama_server_path_
                        + "): " + proc_err;
        } else {
            last_error_ = "failed to start llama-server (path=" + llama_server_path_
                        + "; check model_path/settings)";
        }
        release_port(port);
        return {};
    }

    slot->effective_ctx_size = settings.ctx_size;
    if (settings.ctx_size != original_ctx_size) {
        MM_WARN("SlotManager: slot {} started with reduced ctx_size {} (originally {})",
                slot->id, settings.ctx_size, original_ctx_size);
    }

    slot->client = std::make_unique<LlamaCppClient>(
        "http://127.0.0.1:" + std::to_string(port));
    slot->state = SlotState::Ready;

    SlotId id = slot->id;
    slots_.push_back(std::move(slot));

    MM_INFO("SlotManager: slot {} ready (model={}, agent={}, port={}, ctx_size={})",
            id, model_path, agent_id, port, settings.ctx_size);
    return id;
}

bool SlotManager::unload_slot(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);

    auto it = std::find_if(slots_.begin(), slots_.end(),
                           [&](const auto& s) { return s->id == slot_id; });
    if (it == slots_.end()) return false;

    auto& slot = *it;
    MM_INFO("SlotManager: unloading slot {} (port {})", slot->id, slot->port);

    slot->process->stop();
    release_port(slot->port);

    // Clean up any KV cache file for this slot.
    if (!slot->kv_cache_path.empty()) {
        std::error_code ec;
        fs::remove(slot->kv_cache_path, ec);
    }

    slots_.erase(it);
    return true;
}

std::string SlotManager::suspend_slot(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);

    auto it = std::find_if(slots_.begin(), slots_.end(),
                           [&](const auto& s) { return s->id == slot_id; });
    if (it == slots_.end()) return {};

    auto& slot = *it;
    if (slot->state != SlotState::Ready) return {};

    slot->state = SlotState::Suspending;
    MM_INFO("SlotManager: suspending slot {} (agent={})",
            slot->id, slot->assigned_agent);

    // Build a unique KV cache filename.
    std::string cache_filename = slot->id + ".kvbin";
    std::string cache_path = (fs::path(kv_cache_dir_) / cache_filename).string();

    // Request KV cache save from llama-server via its slot API.
    // POST /slots/0?action=save with {"filename": "<path>"}
    if (slot->client) {
        mm::HttpClient raw_client("http://127.0.0.1:" + std::to_string(slot->port));
        nlohmann::json save_body = {{"filename", cache_path}};
        auto resp = raw_client.post("/slots/0?action=save", save_body);
        if (!resp.ok()) {
            MM_WARN("SlotManager: KV cache save failed for slot {} (HTTP {})",
                    slot->id, resp.status);
            // Continue with suspend even if cache save fails.
            cache_path.clear();
        }
    }

    slot->process->stop();
    release_port(slot->port);

    slot->kv_cache_path = cache_path;
    slot->state = SlotState::Suspended;
    slot->port = 0;
    slot->process.reset();
    slot->client.reset();

    MM_INFO("SlotManager: slot {} suspended (cache={})", slot->id, cache_path);
    return cache_path;
}

SlotId SlotManager::restore_slot(const std::string& model_path,
                                 const LlamaSettings& settings,
                                 const std::string& kv_cache_path,
                                 const AgentId& agent_id) {
    std::lock_guard lock(mutex_);
    last_error_.clear();

    // Remove any existing suspended slot for this agent.
    auto susp_it = std::find_if(slots_.begin(), slots_.end(),
        [&](const auto& s) {
            return s->state == SlotState::Suspended &&
                   s->assigned_agent == agent_id;
        });
    if (susp_it != slots_.end())
        slots_.erase(susp_it);

    // Count only active (non-suspended) slots against max.
    int active_count = 0;
    for (const auto& s : slots_) {
        if (s->state != SlotState::Suspended) ++active_count;
    }
    if (active_count >= max_slots_) {
        MM_WARN("SlotManager: max active slots ({}) reached for restore", max_slots_);
        last_error_ = "max active slots reached";
        return {};
    }

    auto port_opt = allocate_port();
    if (!port_opt) {
        MM_ERROR("SlotManager: no available ports for restore");
        last_error_ = "no available ports for restore";
        return {};
    }
    uint16_t port = *port_opt;

    auto slot = std::make_unique<Slot>();
    slot->id             = util::generate_uuid();
    slot->port           = port;
    slot->model_path     = model_path;
    slot->assigned_agent = agent_id;
    slot->state          = SlotState::Loading;
    slot->vram_usage_mb  = estimate_vram_mb(model_path);
    slot->last_active_ms = util::now_ms();
    slot->kv_cache_path  = kv_cache_path;

    slot->process = std::make_unique<LlamaServerProcess>(llama_server_path_);
    if (log_cb_) slot->process->set_log_callback(log_cb_);

    MM_INFO("SlotManager: restoring model {} on port {} (slot {}, cache={})",
            model_path, port, slot->id, kv_cache_path);

    if (!slot->process->start(model_path, settings, port)) {
        MM_ERROR("SlotManager: failed to start llama-server for restore");
        auto proc_err = slot->process->last_error();
        if (!proc_err.empty()) {
            last_error_ = "failed to start llama-server for restore (path="
                        + llama_server_path_ + "): " + proc_err;
        } else {
            last_error_ = "failed to start llama-server for restore (path="
                        + llama_server_path_ + ")";
        }
        release_port(port);
        return {};
    }

    slot->client = std::make_unique<LlamaCppClient>(
        "http://127.0.0.1:" + std::to_string(port));

    // Attempt to restore KV cache.
    if (!kv_cache_path.empty()) {
        std::error_code ec;
        if (fs::exists(kv_cache_path, ec)) {
            HttpClient raw_client("http://127.0.0.1:" + std::to_string(port));
            nlohmann::json restore_body = {{"filename", kv_cache_path}};
            auto resp = raw_client.post("/slots/0?action=restore", restore_body);
            if (!resp.ok()) {
                MM_WARN("SlotManager: KV cache restore failed (HTTP {}), "
                        "proceeding without cached context", resp.status);
            } else {
                MM_INFO("SlotManager: KV cache restored for slot {}", slot->id);
            }
        } else {
            MM_WARN("SlotManager: KV cache file not found: {}", kv_cache_path);
        }
    }

    slot->state = SlotState::Ready;
    SlotId id = slot->id;
    slots_.push_back(std::move(slot));

    MM_INFO("SlotManager: slot {} restored (agent={})", id, agent_id);
    return id;
}

void SlotManager::unload_all() {
    std::lock_guard lock(mutex_);
    for (auto& slot : slots_) {
        if (slot->process)
            slot->process->stop();
        if (slot->port != 0)
            release_port(slot->port);
    }
    slots_.clear();
    used_ports_.clear();
    MM_INFO("SlotManager: all slots unloaded");
}

std::optional<SlotId> SlotManager::find_slot_by_agent(const AgentId& agent_id) const {
    std::lock_guard lock(mutex_);
    for (const auto& s : slots_) {
        if (s->assigned_agent == agent_id && s->state == SlotState::Ready)
            return s->id;
    }
    return std::nullopt;
}

LlamaCppClient* SlotManager::get_client(const SlotId& slot_id) {
    std::lock_guard lock(mutex_);
    for (const auto& s : slots_) {
        if (s->id == slot_id) return s->client.get();
    }
    return nullptr;
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
    int active = 0;
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

std::string SlotManager::last_error() const {
    std::lock_guard lock(mutex_);
    return last_error_;
}

void SlotManager::set_llama_server_path(const std::string& path) {
    std::lock_guard lock(mutex_);
    llama_server_path_ = path;
}

std::string SlotManager::llama_server_path() const {
    std::lock_guard lock(mutex_);
    return llama_server_path_;
}

// ── Private helpers ─────────────────────────────────────────────────────────

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

int64_t SlotManager::estimate_vram_mb(const std::string& model_path) {
    std::error_code ec;
    auto sz = fs::file_size(model_path, ec);
    if (ec) return 0;
    // Rough heuristic: file_size * 1.2 for runtime overhead.
    return static_cast<int64_t>(static_cast<double>(sz) * 1.2 / (1024.0 * 1024.0));
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
    info.assigned_agent  = s.assigned_agent;
    info.state           = s.state;
    info.vram_usage_mb   = s.vram_usage_mb;
    info.last_active_ms  = s.last_active_ms;
    info.kv_cache_path   = s.kv_cache_path;
    info.effective_ctx_size = s.effective_ctx_size;
    return info;
}

} // namespace mm
