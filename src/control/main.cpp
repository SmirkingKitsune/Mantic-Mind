#include "common/config_file.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "control/control_config.hpp"
#include "control/agent_manager.hpp"
#include "control/node_registry.hpp"
#include "control/model_distributor.hpp"
#include "control/agent_scheduler.hpp"
#include "control/model_router.hpp"
#include "control/agent_queue.hpp"
#include "control/control_api_server.hpp"
#include "control/control_ui.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
#include <functional>
#include <memory>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/file.h>
#  include <unistd.h>
#endif
// ── Config loading ─────────────────────────────────────────────────────────────
// Priority: config file < environment variables.

static mm::ControlConfig load_config(
    std::string* loaded_cfg_path = nullptr,
    const std::string& cfg_name = "mantic-mind-control.toml") {
    mm::ControlConfig cfg;
    mm::ConfigFile file;

    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;
    std::vector<std::string> seen;
    auto add_candidate = [&](const fs::path& p) {
        if (p.empty()) return;
        auto s = p.lexically_normal().string();
        if (std::find(seen.begin(), seen.end(), s) != seen.end()) return;
        seen.push_back(s);
        candidates.push_back(p);
    };

    const char* control_cfg_env = std::getenv("MM_CONTROL_CONFIG_FILE");
    if (control_cfg_env && *control_cfg_env) add_candidate(fs::path(control_cfg_env));
    const char* cfg_env = std::getenv("MM_CONFIG_FILE");
    if (cfg_env && *cfg_env) add_candidate(fs::path(cfg_env));

    std::error_code ec;
    fs::path dir = fs::current_path(ec);
    if (!ec) {
        for (int i = 0; i < 10; ++i) {
            add_candidate(dir / cfg_name);
            if (!dir.has_parent_path()) break;
            auto parent = dir.parent_path();
            if (parent == dir) break;
            dir = parent;
        }
    }

    std::string loaded_from;
    for (const auto& c : candidates) {
        if (file.load(c.string())) {
            loaded_from = c.string();
            break;
        }
    }
    if (loaded_cfg_path) *loaded_cfg_path = loaded_from;

    if (!loaded_from.empty()) {
        cfg.listen_port = static_cast<uint16_t>(
            file.get_int("listen_port", static_cast<int>(cfg.listen_port)));
        cfg.data_dir    = file.get("data_dir",  cfg.data_dir);
        cfg.log_file    = file.get("log_file",   cfg.log_file);
        cfg.node_health_poll_interval_s = static_cast<uint32_t>(
            file.get_int("node_health_poll_interval_s",
                         static_cast<int>(cfg.node_health_poll_interval_s)));
        cfg.models_dir     = file.get("models_dir",     cfg.models_dir);
        cfg.pairing_key    = file.get("pairing_key",    "");
        cfg.discovery_port = static_cast<uint16_t>(
            file.get_int("discovery_port", static_cast<int>(cfg.discovery_port)));
    }

    // Environment variables override file values.
    auto env = [](const char* name, const std::string& cur) -> std::string {
        const char* v = std::getenv(name);
        if (!v) return cur;
        std::string s(v);
        return s.empty() ? cur : s;
    };
    auto env_int = [](const char* name, int cur) -> int {
        const char* v = std::getenv(name);
        if (!v) return cur;
        try { return std::stoi(v); } catch (...) { return cur; }
    };

    cfg.listen_port = static_cast<uint16_t>(
        env_int("MM_CONTROL_PORT", static_cast<int>(cfg.listen_port)));
    cfg.data_dir    = env("MM_DATA_DIR",    cfg.data_dir);
    cfg.log_file    = env("MM_LOG_FILE",    cfg.log_file);
    cfg.models_dir  = env("MM_MODELS_DIR",  cfg.models_dir);
    cfg.pairing_key = env("MM_PAIRING_KEY", cfg.pairing_key);
    cfg.node_health_poll_interval_s = static_cast<uint32_t>(
        env_int("MM_POLL_INTERVAL_S",
                static_cast<int>(cfg.node_health_poll_interval_s)));
    cfg.discovery_port = static_cast<uint16_t>(
        env_int("MM_DISCOVERY_PORT", static_cast<int>(cfg.discovery_port)));

    return cfg;
}

// ── main ──────────────────────────────────────────────────────────────────────

namespace {

class ProcessSingletonLock {
public:
    ~ProcessSingletonLock() {
#ifdef _WIN32
        if (handle_) {
            ReleaseMutex(handle_);
            CloseHandle(handle_);
        }
#else
        if (fd_ >= 0) {
            flock(fd_, LOCK_UN);
            close(fd_);
        }
#endif
    }

    ProcessSingletonLock(const ProcessSingletonLock&) = delete;
    ProcessSingletonLock& operator=(const ProcessSingletonLock&) = delete;

    static std::unique_ptr<ProcessSingletonLock> try_acquire(const std::string& data_dir,
                                                             uint16_t port) {
        const std::string key = data_dir + "|" + std::to_string(port);
        const std::string suffix = std::to_string(std::hash<std::string>{}(key));

#ifdef _WIN32
        const std::string mutex_name = "Global\\mantic-mind-control-" + suffix;
        HANDLE handle = CreateMutexA(nullptr, TRUE, mutex_name.c_str());
        if (!handle || GetLastError() == ERROR_ALREADY_EXISTS) {
            if (handle) CloseHandle(handle);
            return nullptr;
        }

        auto lock = std::unique_ptr<ProcessSingletonLock>(new ProcessSingletonLock());
        lock->handle_ = handle;
        return lock;
#else
        const std::string lock_path = "/tmp/mantic-mind-control-" + suffix + ".lock";
        int fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd < 0) return nullptr;
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            close(fd);
            return nullptr;
        }

        auto lock = std::unique_ptr<ProcessSingletonLock>(new ProcessSingletonLock());
        lock->fd_ = fd;
        return lock;
#endif
    }

private:
    ProcessSingletonLock() = default;

#ifdef _WIN32
    HANDLE handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

} // namespace
int main() {
    std::string cfg_path;
    auto cfg = load_config(&cfg_path);

    std::error_code data_ec;
    std::filesystem::path data_dir_abs =
        std::filesystem::absolute(cfg.data_dir, data_ec);
    const std::string lock_data_dir =
        data_ec ? cfg.data_dir : data_dir_abs.lexically_normal().string();
    auto instance_lock =
        ProcessSingletonLock::try_acquire(lock_data_dir, cfg.listen_port);
    if (!instance_lock) {
        std::fprintf(stderr,
                     "Another mantic-mind-control instance appears to be running for data_dir='%s' and port=%u.\n",
                     cfg.data_dir.c_str(),
                     static_cast<unsigned>(cfg.listen_port));
        return 1;
    }

    // Ensure models directory exists.
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(cfg.models_dir, ec);
    }

    // Disable console logging — the TUI owns the terminal.
    mm::init_logger(cfg.log_file, "mm-control",
                    spdlog::level::off,
                    spdlog::level::trace);

    MM_INFO("mantic-mind-control starting on port {}", cfg.listen_port);
    MM_INFO("Control config source: {}",
            cfg_path.empty() ? "(defaults/env only; no config file found)" : cfg_path);

    // ── Core services ─────────────────────────────────────────────────────────

    mm::AgentManager agents(cfg.data_dir);
    agents.load_all();

    mm::NodeRegistry      registry;
    mm::ModelDistributor  distributor(registry, cfg.models_dir);
    mm::AgentScheduler    scheduler(registry, distributor, cfg.models_dir);
    mm::ModelRouter       router(scheduler);
    mm::AgentQueue        queue;
    mm::ControlApiServer  api_server(agents, queue, registry, router, scheduler, cfg.models_dir);
    mm::ControlUI         ui(
        registry,
        agents,
        cfg.models_dir,
        "http://127.0.0.1:" + std::to_string(cfg.listen_port),
        [&api_server](const std::string& agent_id,
                      const std::string& message,
                      std::string* out_text,
                      std::string* out_conv_id,
                      std::string* out_error) -> bool {
            auto res = api_server.chat_local(agent_id, message);
            if (out_text) {
                std::string text;
                for (const auto& c : res.chunks) {
                    if (!c.delta_content.empty()) text += c.delta_content;
                }
                *out_text = std::move(text);
            }
            if (out_conv_id) *out_conv_id = res.conv_id;
            if (out_error) *out_error = res.error;
            return res.success;
        });

    api_server.set_log_callback([&ui](int level, const std::string& message) {
        auto ll = level == 2 ? mm::ControlUI::LogLevel::Error
                : level == 1 ? mm::ControlUI::LogLevel::Warn
                :              mm::ControlUI::LogLevel::Info;
        ui.log(ll, message);
    });

    registry.set_update_callback([&ui](const mm::NodeInfo& n) {
        if (n.id.empty()) {
            // Sentinel from discovery listener — just trigger a refresh.
            ui.refresh();
            return;
        }
        MM_INFO("Node {} → {} ({})",
                n.id, mm::to_string(n.health), n.connected ? "up" : "down");
        ui.log(mm::ControlUI::LogLevel::Info,
               "Node " + n.id.substr(0, 8) + "… → " +
               mm::to_string(n.health) +
               (n.connected ? " [up]" : " [down]"));
    });
    registry.start_health_poll(
        static_cast<int>(cfg.node_health_poll_interval_s));
    registry.start_discovery_listen(cfg.discovery_port);

    ui.set_pairing_key(cfg.pairing_key);

    // ── Housekeeping thread (every 5 min) ─────────────────────────────────────

    std::atomic<bool> stop_housekeeping{false};
    std::thread housekeeping_thread([&]() {
        while (!stop_housekeeping) {
            // Sleep 5 minutes in small increments for responsive shutdown.
            for (int i = 0; i < 300 && !stop_housekeeping; ++i)
                std::this_thread::sleep_for(std::chrono::seconds(1));

            if (!stop_housekeeping) {
                MM_INFO("Running scheduler housekeeping");
                scheduler.housekeeping(agents.list_agents());
            }
        }
    });

    // ── API server on background thread ───────────────────────────────────────

    std::thread server_thread([&] {
        MM_INFO("API server listening on 0.0.0.0:{}", cfg.listen_port);
        ui.log(mm::ControlUI::LogLevel::Info,
               "API server listening on port " +
               std::to_string(cfg.listen_port));
        if (!api_server.listen(cfg.listen_port)) {
            MM_ERROR("Server failed on port {}", cfg.listen_port);
            ui.log(mm::ControlUI::LogLevel::Error,
                   "Server failed to start on port " +
                   std::to_string(cfg.listen_port));
            ui.quit();
        }
    });

    // ── TUI on main thread (blocks until user quits) ──────────────────────────

    ui.run();

    // ── Graceful shutdown ─────────────────────────────────────────────────────

    MM_INFO("UI exited — shutting down");

    stop_housekeeping = true;
    if (housekeeping_thread.joinable()) housekeeping_thread.join();

    api_server.stop();
    registry.stop_discovery_listen();
    registry.stop_health_poll();
    queue.shutdown();
    if (server_thread.joinable()) server_thread.join();
    MM_INFO("mantic-mind-control stopped");

    return 0;
}
