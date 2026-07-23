#include "control/control_host.hpp"

#include "common/logger.hpp"
#include "control/agent_manager.hpp"
#include "control/agent_queue.hpp"
#include "control/agent_scheduler.hpp"
#include "control/control_api_server.hpp"
#include "control/node_registry.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

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

namespace mm {
namespace {

class ProcessSingletonLock final {
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

    static std::unique_ptr<ProcessSingletonLock> try_acquire(
        const std::string& data_dir, uint16_t port) {
        std::error_code ec;
        const auto absolute = std::filesystem::absolute(data_dir, ec);
        const std::string normalized = ec
            ? data_dir
            : absolute.lexically_normal().string();
        const std::string key = normalized + "|" + std::to_string(port);
        const std::string suffix = std::to_string(std::hash<std::string>{}(key));

#ifdef _WIN32
        const std::string mutex_name = "Global\\mantic-mind-control-" + suffix;
        HANDLE handle = CreateMutexA(nullptr, TRUE, mutex_name.c_str());
        if (!handle || GetLastError() == ERROR_ALREADY_EXISTS) {
            if (handle) CloseHandle(handle);
            return nullptr;
        }
        auto lock = std::unique_ptr<ProcessSingletonLock>(
            new ProcessSingletonLock());
        lock->handle_ = handle;
        return lock;
#else
        const std::string path = "/tmp/mantic-mind-control-" + suffix + ".lock";
        const int fd = open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd < 0) return nullptr;
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            close(fd);
            return nullptr;
        }
        auto lock = std::unique_ptr<ProcessSingletonLock>(
            new ProcessSingletonLock());
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

struct ControlHost::Impl {
    explicit Impl(Options value) : options(std::move(value)) {}

    Options options;
    std::unique_ptr<ProcessSingletonLock> singleton_lock;
    std::unique_ptr<AgentManager> agents;
    std::unique_ptr<NodeRegistry> registry;
    std::unique_ptr<AgentScheduler> scheduler;
    std::unique_ptr<AgentQueue> queue;
    std::unique_ptr<ControlApiServer> api;

    std::atomic<bool> initialized{false};
    std::atomic<bool> started{false};
    std::atomic<bool> stopping{false};
    std::atomic<bool> shutdown_requested{false};
    std::atomic<bool> listener_failed{false};
    std::atomic<bool> stop_housekeeping{false};
    std::thread control_listener;
    std::thread openai_listener;
    std::thread housekeeping;
    std::mutex housekeeping_mutex;
    std::condition_variable housekeeping_cv;
    std::mutex callback_mutex;
    std::function<void(const std::string&)> failure_callback;

    void report_failure(std::string message) noexcept {
        listener_failed = true;
        MM_ERROR("{}", message);
        std::function<void(const std::string&)> callback;
        {
            std::lock_guard<std::mutex> lock(callback_mutex);
            callback = failure_callback;
        }
        if (!callback) return;
        try {
            callback(message);
        } catch (...) {
            // A callback supplied by presentation code cannot be allowed to
            // escape a host-owned worker thread.
        }
    }

    bool wait_for_listeners(std::string* error) {
        for (int attempt = 0; attempt < 100; ++attempt) {
            const bool control_ready = api && api->is_running();
            const bool openai_ready = options.config.openai_compat_port == 0 ||
                                      (api && api->is_openai_compat_running());
            if (control_ready && openai_ready) return true;
            if (listener_failed) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (error) {
            *error = "failed to bind control listener(s) on " +
                     options.bind_host;
        }
        return false;
    }

    void reset_graph() {
        api.reset();
        queue.reset();
        scheduler.reset();
        registry.reset();
        agents.reset();
    }
};

ControlHost::ControlHost(Options options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

ControlHost::~ControlHost() { stop(); }

bool ControlHost::acquire_singleton_lock(std::string* error) {
    if (error) error->clear();
    if (impl_->singleton_lock) return true;
    impl_->singleton_lock = ProcessSingletonLock::try_acquire(
        impl_->options.config.data_dir, impl_->options.config.listen_port);
    if (impl_->singleton_lock) return true;
    if (error) {
        *error = "another control instance owns this data directory and port";
    }
    return false;
}

bool ControlHost::initialize(std::string* error) {
    if (error) error->clear();
    if (impl_->initialized) return true;
    if (!acquire_singleton_lock(error)) return false;

    const auto& config = impl_->options.config;
    if (config.openai_compat_port != 0 &&
        config.openai_compat_port == config.listen_port) {
        if (error) {
            *error = "openai_compat_port must differ from listen_port, or be 0";
        }
        impl_->singleton_lock.reset();
        return false;
    }

    try {
        std::error_code ec;
        std::filesystem::create_directories(config.data_dir, ec);
        if (ec) {
            throw std::runtime_error("failed to create control data directory: " +
                                     ec.message());
        }
        ec.clear();
        std::filesystem::create_directories(config.models_dir, ec);
        if (ec) {
            throw std::runtime_error("failed to create models directory: " +
                                     ec.message());
        }

        impl_->agents = std::make_unique<AgentManager>(config.data_dir);
        impl_->agents->load_all();
        impl_->registry = std::make_unique<NodeRegistry>(
            config.data_dir, impl_->options.enable_remote_nodes);
        impl_->registry->set_offline_after_seconds(
            static_cast<int>(config.node_offline_after_s));
        impl_->scheduler = std::make_unique<AgentScheduler>(
            *impl_->registry, config.models_dir);
        impl_->queue = std::make_unique<AgentQueue>();
        impl_->api = std::make_unique<ControlApiServer>(
            *impl_->agents, *impl_->queue, *impl_->registry, *impl_->scheduler,
            config.data_dir, config.models_dir, config.external_api_token,
            config.tts, impl_->options.allow_legacy_environment);
        impl_->api->cleanup_expired_tts_cache();
        impl_->initialized = true;
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
    } catch (...) {
        if (error) *error = "unknown control initialization failure";
    }
    impl_->reset_graph();
    impl_->singleton_lock.reset();
    return false;
}

bool ControlHost::start(std::string* error) {
    if (error) error->clear();
    if (!impl_->initialized && !initialize(error)) return false;
    if (impl_->started) {
        if (error) *error = "control host is already running";
        return false;
    }

    impl_->stopping = false;
    impl_->shutdown_requested = false;
    impl_->listener_failed = false;
    impl_->stop_housekeeping = false;
    impl_->started = true;

    impl_->control_listener = std::thread([owner = impl_.get()] {
        try {
            if (!owner->api->listen(owner->options.config.listen_port,
                                    owner->options.bind_host) &&
                !owner->stopping && !owner->shutdown_requested) {
                owner->report_failure("control listener stopped unexpectedly");
            }
        } catch (const std::exception& exception) {
            if (!owner->stopping) {
                owner->report_failure(
                    std::string("control listener failed: ") + exception.what());
            }
        } catch (...) {
            if (!owner->stopping) {
                owner->report_failure("control listener failed: unknown exception");
            }
        }
    });

    if (impl_->options.config.openai_compat_port != 0) {
        impl_->openai_listener = std::thread([owner = impl_.get()] {
            try {
                if (!owner->api->listen_openai_compat(
                        owner->options.config.openai_compat_port,
                        owner->options.bind_host) &&
                    !owner->stopping && !owner->shutdown_requested) {
                    owner->report_failure(
                        "OpenAI-compatible listener stopped unexpectedly");
                }
            } catch (const std::exception& exception) {
                if (!owner->stopping) {
                    owner->report_failure(
                        std::string("OpenAI-compatible listener failed: ") +
                        exception.what());
                }
            } catch (...) {
                if (!owner->stopping) {
                    owner->report_failure(
                        "OpenAI-compatible listener failed: unknown exception");
                }
            }
        });
    }

    if (!impl_->wait_for_listeners(error)) {
        // A failed bind is a failed lifecycle, not a paused one: AgentQueue
        // has already stopped accepting, so retain neither the graph nor its
        // singleton lock. This makes a later initialize/start a clean retry.
        stop();
        return false;
    }

    impl_->registry->start_health_poll(
        static_cast<int>(impl_->options.config.node_health_poll_interval_s));
    if (impl_->options.enable_remote_nodes && impl_->options.enable_discovery) {
        if (!impl_->registry->start_discovery_listen(
                impl_->options.config.discovery_port)) {
            if (error) {
                *error = "failed to bind discovery listener on UDP port " +
                         std::to_string(impl_->options.config.discovery_port);
            }
            stop();
            return false;
        }
    }

    impl_->housekeeping = std::thread([owner = impl_.get()] {
        try {
            std::unique_lock<std::mutex> lock(owner->housekeeping_mutex);
            while (!owner->stop_housekeeping) {
                if (owner->housekeeping_cv.wait_for(
                        lock, std::chrono::minutes(5), [owner] {
                            return owner->stop_housekeeping.load();
                        })) {
                    break;
                }
                lock.unlock();
                MM_INFO("Running scheduler housekeeping");
                owner->scheduler->housekeeping(owner->agents->list_agents());
                owner->api->cleanup_expired_tts_cache();
                lock.lock();
            }
        } catch (const std::exception& exception) {
            if (!owner->stopping) {
                owner->report_failure(
                    std::string("control housekeeping failed: ") +
                    exception.what());
            }
        } catch (...) {
            if (!owner->stopping) {
                owner->report_failure(
                    "control housekeeping failed: unknown exception");
            }
        }
    });
    return true;
}

void ControlHost::request_shutdown() {
    if (!impl_->initialized) return;
    if (impl_->shutdown_requested.exchange(true)) return;
    impl_->stop_housekeeping = true;
    impl_->housekeeping_cv.notify_all();
    if (impl_->queue) {
        impl_->queue->stop_accepting();
        impl_->queue->request_cancel_active();
    }
    if (impl_->api) {
        impl_->api->stop();
        impl_->api->stop_openai_compat();
    }
    if (impl_->registry) {
        impl_->registry->request_operations_shutdown();
    }
}

void ControlHost::stop() {
    if (!impl_->initialized && !impl_->singleton_lock) return;
    if (impl_->stopping.exchange(true)) return;

    request_shutdown();
    if (impl_->housekeeping.joinable()) impl_->housekeeping.join();

    // Queue workers may still need the registry and embedded node while they
    // observe cancellation and finish their callbacks.
    if (impl_->queue) impl_->queue->shutdown();
    if (impl_->registry) {
        impl_->registry->stop_discovery_listen();
        impl_->registry->stop_health_poll();
    }
    if (impl_->control_listener.joinable()) impl_->control_listener.join();
    if (impl_->openai_listener.joinable()) impl_->openai_listener.join();

    impl_->started = false;
    impl_->initialized = false;
    impl_->reset_graph();
    impl_->singleton_lock.reset();
    impl_->stopping = false;
}

bool ControlHost::initialized() const { return impl_->initialized; }
bool ControlHost::running() const { return impl_->started; }
bool ControlHost::listener_failed() const { return impl_->listener_failed; }

void ControlHost::set_failure_callback(
    std::function<void(const std::string&)> callback) {
    std::lock_guard<std::mutex> lock(impl_->callback_mutex);
    impl_->failure_callback = std::move(callback);
}

AgentManager& ControlHost::agents() { return *impl_->agents; }
NodeRegistry& ControlHost::registry() { return *impl_->registry; }
AgentScheduler& ControlHost::scheduler() { return *impl_->scheduler; }
AgentQueue& ControlHost::queue() { return *impl_->queue; }
ControlApiServer& ControlHost::api() { return *impl_->api; }
const ControlConfig& ControlHost::config() const { return impl_->options.config; }

} // namespace mm
