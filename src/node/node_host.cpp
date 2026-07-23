#include "node/node_host.hpp"

#include "common/logger.hpp"
#include "node/model_store.hpp"
#include "node/node_service.hpp"
#include "node/node_state.hpp"
#include "node/singleton_lock.hpp"
#include "node/slot_manager.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>

namespace mm {

struct NodeHost::Impl {
    explicit Impl(Options value) : options(std::move(value)) {}

    Options options;
    std::unique_ptr<SingletonLock> singleton_lock;
    std::unique_ptr<NodeState> state;
    std::unique_ptr<SlotManager> slots;
    std::unique_ptr<ModelStore> model_store;
    std::unique_ptr<NodeService> service;

    std::atomic<bool> initialized{false};
    std::atomic<bool> stopping{false};
    std::atomic<bool> shutdown_requested{false};
    std::atomic<bool> metrics_ready{false};
    std::atomic<bool> stop_model_housekeeping{false};
    std::thread model_housekeeping;
    std::mutex housekeeping_mutex;
    std::condition_variable housekeeping_cv;

    void reset_graph() {
        service.reset();
        model_store.reset();
        slots.reset();
        state.reset();
    }
};

NodeHost::NodeHost(Options options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

NodeHost::~NodeHost() { stop(); }

bool NodeHost::acquire_singleton_lock(std::string* error) {
    if (error) error->clear();
    if (impl_->singleton_lock) return true;
    impl_->singleton_lock = SingletonLock::try_acquire(
        impl_->options.singleton_lock_name.empty()
            ? std::string{"mantic-mind-node"}
            : impl_->options.singleton_lock_name);
    if (impl_->singleton_lock) return true;
    if (error) *error = "another mantic-mind node instance is running";
    return false;
}

bool NodeHost::initialize(std::string* error) {
    if (error) error->clear();
    if (impl_->initialized) return true;
    if (!acquire_singleton_lock(error)) return false;

    impl_->stopping = false;
    impl_->shutdown_requested = false;
    impl_->stop_model_housekeeping = false;
    impl_->metrics_ready = false;
    impl_->initialized = true;

    try {
        const auto& config = impl_->options.config;
        auto ensure_directory = [](const std::string& path) {
            if (path.empty()) return;
            std::error_code ec;
            std::filesystem::create_directories(path, ec);
            if (ec) {
                throw std::runtime_error(
                    "failed to create node directory '" + path + "': " +
                    ec.message());
            }
        };
        ensure_directory(config.kv_cache_dir);
        if (impl_->options.manage_model_cache) {
            ensure_directory(config.models_dir);
        }

        impl_->state = std::make_unique<NodeState>();
        impl_->state->set_registered(
            impl_->options.registered, impl_->options.node_id);
        if (impl_->options.mark_control_contact) {
            impl_->state->mark_control_contact();
        }
        impl_->state->start_metrics_poll(
            std::max(1, impl_->options.metrics_interval_ms));
        impl_->metrics_ready = impl_->state->wait_for_initial_metrics(
            std::max(0, impl_->options.initial_metrics_timeout_ms));

        impl_->slots = std::make_unique<SlotManager>(
            config.runtime_port_range_start,
            config.runtime_port_range_end,
            config.max_slots);
        impl_->slots->reset_shutdown();
        impl_->slots->set_llama_server_path(config.llama_server_path);
        impl_->slots->set_kv_cache_dir(config.kv_cache_dir);
        impl_->slots->set_models_dir(config.models_dir);

        if (impl_->options.manage_model_cache) {
            impl_->model_store = std::make_unique<ModelStore>(
                config.models_dir, config.model_cache_min_free_mb);
        }
        impl_->service = std::make_unique<NodeService>(
            *impl_->state, *impl_->slots, impl_->model_store.get());

        if (impl_->model_store) {
            impl_->model_housekeeping = std::thread([owner = impl_.get()] {
                std::unique_lock<std::mutex> wait_lock(owner->housekeeping_mutex);
                while (!owner->stop_model_housekeeping) {
                    if (owner->housekeeping_cv.wait_for(
                            wait_lock, std::chrono::milliseconds(2500), [owner] {
                                return owner->stop_model_housekeeping.load();
                            })) {
                        break;
                    }
                    wait_lock.unlock();
                    std::set<std::string> in_use;
                    for (const auto& slot : owner->slots->get_slot_info()) {
                        if (!slot.model_path.empty()) in_use.insert(slot.model_path);
                        if (!slot.mmproj_path.empty()) in_use.insert(slot.mmproj_path);
                    }
                    for (const auto& id :
                         owner->model_store->enforce_min_free(in_use)) {
                        MM_INFO("Model cache: disk pressure evicted {}", id);
                    }
                    wait_lock.lock();
                }
            });
        }
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
    } catch (...) {
        if (error) *error = "unknown node initialization failure";
    }
    stop();
    return false;
}

void NodeHost::request_shutdown() {
    if (!impl_->initialized) return;
    if (impl_->shutdown_requested.exchange(true)) return;
    impl_->stop_model_housekeeping = true;
    impl_->housekeeping_cv.notify_all();
    if (impl_->slots) impl_->slots->request_shutdown();
    if (impl_->state) impl_->state->request_action_cancel();
    if (impl_->service) impl_->service->cancel_action();
}

void NodeHost::stop() {
    if (!impl_->initialized && !impl_->singleton_lock) return;
    if (impl_->stopping.exchange(true)) return;

    request_shutdown();
    if (impl_->model_housekeeping.joinable()) {
        impl_->model_housekeeping.join();
    }

    // Callers stop NodeApiServer/LocalNodeOperations first. Force is still
    // required here so a partially failed startup or busy slot cannot leak a
    // llama-server child past host teardown.
    if (impl_->slots) impl_->slots->unload_all(true);
    if (impl_->state) impl_->state->stop_metrics_poll();

    if (impl_->model_store &&
        impl_->options.config.model_cache_clear_on_shutdown) {
        const auto cleared = impl_->model_store->clear_unpinned({});
        if (!cleared.empty()) {
            MM_INFO("Model cache: cleared {} unpinned model(s) on shutdown",
                    cleared.size());
        }
    }

    impl_->reset_graph();
    impl_->singleton_lock.reset();
    impl_->metrics_ready = false;
    impl_->initialized = false;
    impl_->stopping = false;
}

bool NodeHost::initialized() const { return impl_->initialized; }
bool NodeHost::initial_metrics_ready() const { return impl_->metrics_ready; }
NodeState& NodeHost::state() { return *impl_->state; }
SlotManager& NodeHost::slots() { return *impl_->slots; }
NodeService& NodeHost::service() { return *impl_->service; }
ModelStore* NodeHost::model_store() { return impl_->model_store.get(); }
const NodeConfig& NodeHost::config() const { return impl_->options.config; }

} // namespace mm
