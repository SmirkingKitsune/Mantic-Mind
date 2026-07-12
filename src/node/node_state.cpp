#include "node/node_state.hpp"
#include "common/util.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <thread>

#ifdef _WIN32
#  include <Windows.h>
#else
#  include <fstream>
#  include <sstream>
#endif

namespace mm {

NodeState::NodeState()  = default;
NodeState::~NodeState() { stop_metrics_poll(); }

// ── Registration ───────────────────────────────────────────────────────────────
bool   NodeState::is_registered() const { std::lock_guard<std::mutex> g(mutex_); return registered_; }
NodeId NodeState::get_node_id()   const { std::lock_guard<std::mutex> g(mutex_); return node_id_; }

void NodeState::set_registered(bool v, const NodeId& id) {
    std::lock_guard<std::mutex> g(mutex_);
    registered_ = v;
    if (!id.empty()) node_id_ = id;
    if (v) last_control_contact_ms_ = mm::util::now_ms();
}

void NodeState::mark_control_contact() {
    std::lock_guard<std::mutex> g(mutex_);
    last_control_contact_ms_ = mm::util::now_ms();
}

int64_t NodeState::get_last_control_contact_ms() const {
    std::lock_guard<std::mutex> g(mutex_);
    return last_control_contact_ms_;
}

bool NodeState::has_recent_control_contact(int64_t max_age_ms) const {
    std::lock_guard<std::mutex> g(mutex_);
    if (last_control_contact_ms_ <= 0) return false;
    return (mm::util::now_ms() - last_control_contact_ms_) <= max_age_ms;
}

// ── Model / agent ──────────────────────────────────────────────────────────────
std::string NodeState::get_loaded_model() const { std::lock_guard<std::mutex> g(mutex_); return loaded_model_; }
std::string NodeState::get_active_agent() const { std::lock_guard<std::mutex> g(mutex_); return active_agent_; }
void NodeState::set_loaded_model(const std::string& m) { std::lock_guard<std::mutex> g(mutex_); loaded_model_ = m; }
void NodeState::set_active_agent(const std::string& a) { std::lock_guard<std::mutex> g(mutex_); active_agent_ = a; }

// ── Multi-slot tracking ──────────────────────────────────────────────────────
std::vector<SlotInfo> NodeState::get_slots() const { std::lock_guard<std::mutex> g(mutex_); return slots_; }
void NodeState::set_slots(const std::vector<SlotInfo>& slots) {
    std::lock_guard<std::mutex> g(mutex_);
    slots_ = slots;

    // Keep legacy single-model/agent fields aligned with multi-slot state.
    loaded_model_.clear();
    active_agent_.clear();
    for (const auto& s : slots_) {
        if (s.state != SlotState::Ready) continue;
        if (loaded_model_.empty()) loaded_model_ = s.model_path;
        if (active_agent_.empty()) active_agent_ = s.assigned_agent;
        if (!loaded_model_.empty() && !active_agent_.empty()) break;
    }
}

// ── Metrics ────────────────────────────────────────────────────────────────────
NodeHealthMetrics NodeState::get_metrics() const { std::lock_guard<std::mutex> g(mutex_); return metrics_; }
void NodeState::update_metrics(const NodeHealthMetrics& m) {
    std::lock_guard<std::mutex> g(mutex_);
    metrics_ = m;
    if (metrics_cb_) metrics_cb_(m);
}

std::string NodeState::get_last_error() const {
    std::lock_guard<std::mutex> g(mutex_);
    return last_error_;
}

void NodeState::set_last_error(const std::string& err) {
    std::lock_guard<std::mutex> g(mutex_);
    last_error_ = err;
}

// ── Streaming text ─────────────────────────────────────────────────────────────
StreamingTextState NodeState::get_streaming_text() const {
    std::lock_guard<std::mutex> g(mutex_);
    return streaming_text_;
}

void NodeState::start_streaming_text(const SlotId& slot_id, const AgentId& agent_id) {
    std::lock_guard<std::mutex> g(mutex_);
    streaming_text_ = {};
    streaming_text_.slot_id    = slot_id;
    streaming_text_.agent_id   = agent_id;
    streaming_text_.active     = true;
    streaming_text_.started_ms = mm::util::now_ms();
    streaming_text_.updated_ms = streaming_text_.started_ms;
}

void NodeState::append_streaming_text(const std::string& delta_content,
                                      const std::string& thinking_delta,
                                      int tokens_used) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!delta_content.empty()) streaming_text_.content += delta_content;
    if (!thinking_delta.empty()) streaming_text_.thinking += thinking_delta;
    if (tokens_used > 0) streaming_text_.tokens_used = tokens_used;
    streaming_text_.updated_ms = mm::util::now_ms();
}

void NodeState::finish_streaming_text(const std::string& finish_reason, int tokens_used) {
    std::lock_guard<std::mutex> g(mutex_);
    streaming_text_.active        = false;
    streaming_text_.finish_reason = finish_reason;
    streaming_text_.tokens_used   = tokens_used;
    streaming_text_.updated_ms    = mm::util::now_ms();
}

void NodeState::clear_streaming_text() {
    std::lock_guard<std::mutex> g(mutex_);
    streaming_text_ = {};
}

// ── API keys ───────────────────────────────────────────────────────────────────
void NodeState::add_api_key(const std::string& key)    { std::lock_guard<std::mutex> g(mutex_); api_keys_.insert(key); }
void NodeState::remove_api_key(const std::string& key) { std::lock_guard<std::mutex> g(mutex_); api_keys_.erase(key); }
bool NodeState::validate_api_key(const std::string& key) const {
    std::lock_guard<std::mutex> g(mutex_); return api_keys_.count(key) > 0;
}
std::vector<std::string> NodeState::get_api_keys() const {
    std::lock_guard<std::mutex> g(mutex_);
    return { api_keys_.begin(), api_keys_.end() };
}

// ── Pairing ────────────────────────────────────────────────────────────────────
void NodeState::set_pending_pair(const PendingPair& p) {
    std::lock_guard<std::mutex> g(mutex_);
    pending_pair_ = p;
}

std::optional<PendingPair> NodeState::get_pending_pair() const {
    std::lock_guard<std::mutex> g(mutex_);
    if (!pending_pair_) return std::nullopt;
    if (mm::util::now_ms() >= pending_pair_->expiry_ms) return std::nullopt;
    return pending_pair_;
}

void NodeState::clear_pending_pair() {
    std::lock_guard<std::mutex> g(mutex_);
    pending_pair_.reset();
}

void NodeState::set_metrics_callback(MetricsCallback cb) {
    std::lock_guard<std::mutex> g(mutex_); metrics_cb_ = std::move(cb);
}

// ── Background polling ─────────────────────────────────────────────────────────
void NodeState::start_metrics_poll(int interval_ms) {
    if (polling_.exchange(true)) return;
    poll_thread_ = std::thread([this, interval_ms]() {
        while (polling_) {
            update_metrics(sample_metrics());
            // Interruptible wait: stop_metrics_poll() wakes us immediately so
            // node shutdown is not delayed by up to interval_ms.
            std::unique_lock<std::mutex> lk(poll_mutex_);
            poll_cv_.wait_for(lk, std::chrono::milliseconds(interval_ms),
                              [this] { return !polling_; });
        }
    });
}

void NodeState::stop_metrics_poll() {
    {
        std::lock_guard<std::mutex> lk(poll_mutex_);
        polling_ = false;
    }
    poll_cv_.notify_all();
    if (poll_thread_.joinable()) poll_thread_.join();
}

NodeCapabilities NodeState::get_capabilities() const {
    std::lock_guard<std::mutex> g(mutex_);
    return capabilities_;
}

void NodeState::set_capabilities(const NodeCapabilities& caps) {
    std::lock_guard<std::mutex> g(mutex_);
    capabilities_ = caps;
}

VllmRuntimeStatus NodeState::get_vllm_runtime() const {
    std::lock_guard<std::mutex> g(mutex_);
    return vllm_runtime_;
}

void NodeState::set_vllm_runtime(const VllmRuntimeStatus& runtime) {
    std::lock_guard<std::mutex> g(mutex_);
    vllm_runtime_ = runtime;
}

LlamaRuntimeStatus NodeState::get_llama_runtime() const {
    std::lock_guard<std::mutex> g(mutex_);
    return llama_runtime_;
}

void NodeState::set_llama_runtime(const LlamaRuntimeStatus& runtime) {
    std::lock_guard<std::mutex> g(mutex_);
    llama_runtime_ = runtime;
}

VllmInstallProgress NodeState::get_vllm_install_progress() const {
    std::lock_guard<std::mutex> g(mutex_);
    return vllm_install_progress_;
}

void NodeState::set_vllm_install_progress(const VllmInstallProgress& p) {
    std::lock_guard<std::mutex> g(mutex_);
    vllm_install_progress_ = p;
    if (p.active) {
        NodeActionProgress action;
        action.active = true;
        action.operation_id = "vllm-runtime";
        action.kind = "runtime";
        action.action = "Downloading runtime";
        action.target = "vLLM";
        action.stage = p.stage;
        action.detail = p.last_line;
        action.step = p.step;
        action.total_steps = p.total_steps;
        action.fraction = p.fraction;
        action.cancelable = true;
        action.cancel_requested =
            action_progress_.active && action_progress_.operation_id == action.operation_id
                ? action_progress_.cancel_requested
                : false;
        action_progress_ = std::move(action);
    } else if (action_progress_.kind == "runtime") {
        action_progress_ = {};
    }
}

NodeActionProgress NodeState::get_action_progress() const {
    std::lock_guard<std::mutex> g(mutex_);
    return action_progress_;
}

void NodeState::set_action_progress(const NodeActionProgress& p) {
    std::lock_guard<std::mutex> g(mutex_);
    const bool same_operation =
        action_progress_.active && !p.operation_id.empty() &&
        action_progress_.operation_id == p.operation_id;
    const bool keep_cancel = same_operation && action_progress_.cancel_requested;
    action_progress_ = p;
    if (keep_cancel) action_progress_.cancel_requested = true;
}

void NodeState::clear_action_progress(const std::string& operation_id) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!operation_id.empty() && action_progress_.operation_id != operation_id) return;
    action_progress_ = {};
}

bool NodeState::request_action_cancel() {
    std::lock_guard<std::mutex> g(mutex_);
    if (!action_progress_.active || !action_progress_.cancelable) return false;
    action_progress_.cancel_requested = true;
    return true;
}

bool NodeState::action_cancel_requested(const std::string& operation_id) const {
    std::lock_guard<std::mutex> g(mutex_);
    if (!action_progress_.active || !action_progress_.cancel_requested) return false;
    return operation_id.empty() || action_progress_.operation_id == operation_id;
}

// ── Platform metrics sampling ──────────────────────────────────────────────────
NodeHealthMetrics NodeState::sample_metrics() {
    NodeHealthMetrics m;

#ifdef _WIN32
    // ── CPU (Windows) ──────────────────────────────────────────────────────────
    {
        static FILETIME prev_idle{}, prev_kernel{}, prev_user{};
        FILETIME idle, kernel, user;
        if (GetSystemTimes(&idle, &kernel, &user)) {
            auto ft2u64 = [](const FILETIME& ft) -> uint64_t {
                return (static_cast<uint64_t>(ft.dwHighDateTime) << 32)
                     | static_cast<uint64_t>(ft.dwLowDateTime);
            };
            uint64_t d_idle   = ft2u64(idle)   - ft2u64(prev_idle);
            uint64_t d_kernel = ft2u64(kernel) - ft2u64(prev_kernel);
            uint64_t d_user   = ft2u64(user)   - ft2u64(prev_user);
            uint64_t d_total  = d_kernel + d_user;
            if (d_total > 0)
                m.cpu_percent = static_cast<float>(
                    100.0 * static_cast<double>(d_total - d_idle) / static_cast<double>(d_total));
            prev_idle   = idle;
            prev_kernel = kernel;
            prev_user   = user;
        }
    }

    // ── RAM (Windows) ──────────────────────────────────────────────────────────
    {
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            m.ram_percent  = static_cast<float>(ms.dwMemoryLoad);
            m.ram_total_mb = static_cast<int64_t>(ms.ullTotalPhys / (1024ULL * 1024ULL));
            m.ram_used_mb  = static_cast<int64_t>(
                (ms.ullTotalPhys - ms.ullAvailPhys) / (1024ULL * 1024ULL));
        }
    }

#else
    // ── CPU (Linux /proc/stat) ─────────────────────────────────────────────────
    {
        static uint64_t prev_total = 0, prev_idle_v = 0;
        std::ifstream f("/proc/stat");
        if (f) {
            std::string cpu;
            uint64_t user = 0, nice = 0, system = 0, idle = 0,
                     iowait = 0, irq = 0, softirq = 0;
            f >> cpu >> user >> nice >> system >> idle;
            f >> iowait >> irq >> softirq;
            uint64_t total_idle = idle + iowait;
            uint64_t total = user + nice + system + idle + iowait + irq + softirq;
            if (prev_total > 0) {
                uint64_t d_total = total - prev_total;
                uint64_t d_idle  = total_idle - prev_idle_v;
                if (d_total > 0)
                    m.cpu_percent = static_cast<float>(
                        100.0 * static_cast<double>(d_total - d_idle)
                              / static_cast<double>(d_total));
            }
            prev_total  = total;
            prev_idle_v = total_idle;
        }
    }

    // ── RAM (Linux /proc/meminfo) ──────────────────────────────────────────────
    {
        std::ifstream f("/proc/meminfo");
        if (f) {
            uint64_t mem_total = 0, mem_available = 0;
            std::string key, unit;
            uint64_t val = 0;
            for (int i = 0; i < 30 && (mem_total == 0 || mem_available == 0); ++i) {
                f >> key >> val >> unit;
                if (key == "MemTotal:")     mem_total     = val;
                else if (key == "MemAvailable:") mem_available = val;
            }
            m.ram_total_mb = static_cast<int64_t>(mem_total / 1024);
            m.ram_used_mb  = static_cast<int64_t>((mem_total - mem_available) / 1024);
            if (mem_total > 0)
                m.ram_percent = static_cast<float>(
                    100.0 * static_cast<double>(mem_total - mem_available)
                          / static_cast<double>(mem_total));
        }
    }
#endif

    // ── Disk free space ───────────────────────────────────────────────────────
    {
        std::error_code ec;
        auto si = std::filesystem::space(".", ec);
        if (!ec)
            m.disk_free_mb = static_cast<int64_t>(si.available / (1024 * 1024));
    }

    // ── GPU via nvidia-smi (cross-platform) ───────────────────────────────────
    {
        static const char* cmd =
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total"
            " --format=csv,noheader,nounits 2>"
#ifdef _WIN32
            "nul"
#else
            "/dev/null"
#endif
            ;

#ifdef _WIN32
        FILE* f = _popen(cmd, "r");
#else
        FILE* f = ::popen(cmd, "r");
#endif
        bool gpu_present = false;
        if (f) {
            char buf[256] = {};
            if (fgets(buf, static_cast<int>(sizeof(buf)), f) &&
                !mm::util::trim(buf).empty()) {
                // A CSV row means nvidia-smi sees a GPU — even when fields read
                // "[N/A]", as unified-memory parts (GB10/GH200, Jetson) do for
                // memory.total.
                gpu_present = true;
                float gpu_util = 0, vram_used = 0, vram_total = 0;
                // Utilization is reported even when memory.total is "[N/A]".
                if (sscanf(buf, "%f", &gpu_util) == 1)
                    m.gpu_percent = gpu_util;
                // nvidia-smi may use " , " or ", " separators.
                if (sscanf(buf, "%*f , %f , %f", &vram_used, &vram_total) == 2 ||
                    sscanf(buf, "%*f, %f, %f",   &vram_used, &vram_total) == 2 ||
                    sscanf(buf, "%*f,%f,%f",      &vram_used, &vram_total) == 2) {
                    m.gpu_vram_used_mb  = static_cast<int64_t>(vram_used);
                    m.gpu_vram_total_mb = static_cast<int64_t>(vram_total);
                }
            }
#ifdef _WIN32
            _pclose(f);
#else
            ::pclose(f);
#endif
        }

        // A CUDA-capable GPU backend is available whenever nvidia-smi reports a
        // GPU, regardless of whether it exposes a discrete VRAM total. Unified-
        // memory parts (GB10/GH200, Jetson) return "[N/A]" for memory.total, so
        // gating on VRAM size would wrongly hide the GPU — and NCCL — there.
        // This drives NCCL advertisement in capability detection.
        m.gpu_backend_available = gpu_present;
    }

    return m;
}

} // namespace mm
