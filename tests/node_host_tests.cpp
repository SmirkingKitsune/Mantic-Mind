#include "common/util.hpp"
#include "node/node_api_server.hpp"
#include "node/node_host.hpp"
#include "node/model_store.hpp"
#include "node/node_service.hpp"
#include "node/node_state.hpp"
#include "node/slot_manager.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace {

#define CHECK(expr) do {                                                       \
    if (!(expr)) {                                                             \
        std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__       \
                  << ": " #expr << '\n';                                     \
        return false;                                                          \
    }                                                                          \
} while (false)

class TempDirectory {
public:
    TempDirectory() {
        path_ = std::filesystem::temp_directory_path() /
                ("mm-node-host-" + mm::util::generate_uuid());
        std::filesystem::create_directories(path_);
    }
    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

mm::NodeHost::Options make_options(const TempDirectory& temp,
                                   std::string lock_name,
                                   std::string node_id,
                                   bool cache) {
    mm::NodeHost::Options options;
    options.config.data_dir = (temp.path() / "data").string();
    options.config.models_dir = (temp.path() / "models").string();
    options.config.kv_cache_dir = (temp.path() / "data" / "kv").string();
    options.config.runtime_port_range_start = 48100;
    options.config.runtime_port_range_end = 48102;
    options.config.max_slots = 2;
    options.config.model_cache_min_free_mb = 0;
    options.config.model_cache_clear_on_shutdown = false;
    options.node_id = std::move(node_id);
    options.registered = true;
    options.mark_control_contact = true;
    options.manage_model_cache = cache;
    options.metrics_interval_ms = 10;
    options.initial_metrics_timeout_ms = 2000;
    options.singleton_lock_name = std::move(lock_name);
    return options;
}

int run_never_healthy_runtime(int argc, char** argv) {
    std::string model_path;
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == "--model") {
            model_path = argv[index + 1];
            break;
        }
    }
    if (model_path.empty()) return -1;

    // The parent waits for this marker before requesting shutdown, proving the
    // child was spawned and RuntimeProcess is inside its health wait.
    {
        std::ofstream marker(model_path + ".runtime-started", std::ios::trunc);
        marker << "started\n";
    }
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

bool wait_for_file(const std::filesystem::path& path,
                   std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        std::error_code error;
        if (std::filesystem::is_regular_file(path, error)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

bool run_startup_cancel_case(mm::NodeHost& host,
                             const std::filesystem::path& model_path,
                             bool restore) {
    {
        std::ofstream model(model_path, std::ios::binary | std::ios::trunc);
        model << "test model";
    }
    const auto marker_path = std::filesystem::path(
        model_path.string() + ".runtime-started");

    mm::LlamaRuntimeStatus runtime;
    runtime.status = "ready";
    runtime.executable_path = host.config().llama_server_path;
    host.state().set_llama_runtime(runtime);

    mm::NodeLoadModelResult result;
    std::thread loader([&] {
        if (restore) {
            mm::NodeRestoreModelRequest request;
            request.model_path = model_path.string();
            request.agent_id = "restore-agent";
            result = host.service().restore_slot(request);
        } else {
            mm::NodeLoadModelRequest request;
            request.model_path = model_path.string();
            request.agent_id = "load-agent";
            result = host.service().load_model(request);
        }
    });

    const bool child_started = wait_for_file(marker_path, std::chrono::seconds(10));
    const auto shutdown_started = std::chrono::steady_clock::now();
    host.request_shutdown();
    if (loader.joinable()) loader.join();
    const auto shutdown_elapsed =
        std::chrono::steady_clock::now() - shutdown_started;

    if (!child_started) return false;
    if (shutdown_elapsed >= std::chrono::seconds(5)) return false;
    if (result.status != mm::NodeServiceStatus::Canceled) return false;
    if (result.error.find("canceled") == std::string::npos) return false;
    return host.slots().get_slot_info().empty();
}

bool test_lock_graph_adapter_and_restart() {
    TempDirectory temp;
    const std::string lock_name = "mantic-mind-node-test-" +
                                  mm::util::generate_uuid();
    mm::NodeHost first(make_options(temp, lock_name, "local", false));
    mm::NodeHost second(make_options(temp, lock_name, "second", false));

    std::string error = "stale";
    CHECK(first.acquire_singleton_lock(&error));
    CHECK(error.empty());
    CHECK(!second.acquire_singleton_lock(&error));
    CHECK(error.find("node instance") != std::string::npos);

    CHECK(first.initialize(&error));
    CHECK(error.empty());
    CHECK(first.initialized());
    CHECK(first.state().get_node_id() == "local");
    CHECK(first.state().is_registered());
    CHECK(first.state().get_last_control_contact_ms() > 0);
    CHECK(first.slots().max_slots() == 2);
    CHECK(first.model_store() == nullptr);
    CHECK(first.service().snapshot().node_id == "local");

    // The wire adapter must share, rather than duplicate, the host service.
    {
        mm::NodeApiServer adapter(
            first.service(), first.state(), first.slots());
        CHECK(first.service().snapshot().max_slots == 2);
    }

    first.request_shutdown();
    first.request_shutdown();
    first.stop();
    first.stop();
    CHECK(!first.initialized());

    // The failed contender can acquire the released role and initialize a
    // fresh graph without reconstructing its NodeHost object.
    CHECK(second.initialize(&error));
    CHECK(error.empty());
    CHECK(second.state().get_node_id() == "second");
    second.stop();

    // A stopped host itself is reusable and resets cancellation/metrics state.
    CHECK(first.initialize(&error));
    CHECK(first.state().get_node_id() == "local");
    first.stop();
    return true;
}

bool test_optional_remote_cache_is_host_owned() {
    TempDirectory temp;
    const std::string lock_name = "mantic-mind-node-test-" +
                                  mm::util::generate_uuid();
    auto options = make_options(temp, lock_name, "remote-cache", true);
    mm::NodeHost host(std::move(options));
    std::string error;
    CHECK(host.initialize(&error));
    CHECK(host.model_store() != nullptr);
    CHECK(std::filesystem::is_directory(temp.path() / "models"));
    CHECK(host.service().snapshot().model_cache_free_bytes >= 0);
    host.stop();
    host.stop();
    return true;
}

bool test_initialization_failure_releases_lock() {
    TempDirectory temp;
    const auto blocked_path = temp.path() / "not-a-directory";
    {
        std::ofstream output(blocked_path);
        output << "occupied";
    }
    const std::string lock_name = "mantic-mind-node-test-" +
                                  mm::util::generate_uuid();
    auto bad_options = make_options(temp, lock_name, "bad", false);
    bad_options.config.kv_cache_dir = blocked_path.string();
    mm::NodeHost failed(std::move(bad_options));
    std::string error;
    CHECK(!failed.initialize(&error));
    CHECK(!error.empty());
    CHECK(!failed.initialized());

    mm::NodeHost recovered(make_options(temp, lock_name, "recovered", false));
    CHECK(recovered.initialize(&error));
    CHECK(error.empty());
    recovered.stop();
    return true;
}

bool test_cache_cleanup_preserves_pinned_models() {
    TempDirectory temp;
    const std::string lock_name = "mantic-mind-node-test-" +
                                  mm::util::generate_uuid();
    auto options = make_options(temp, lock_name, "cache-cleanup", true);
    options.config.model_cache_clear_on_shutdown = true;
    mm::NodeHost host(std::move(options));
    std::string error;
    CHECK(host.initialize(&error));
    auto* store = host.model_store();
    CHECK(store != nullptr);

    const auto unpinned_file = temp.path() / "models" / "unpinned" /
                               "model.gguf";
    const auto pinned_file = temp.path() / "models" / "pinned" /
                             "model.gguf";
    std::filesystem::create_directories(unpinned_file.parent_path());
    std::filesystem::create_directories(pinned_file.parent_path());
    {
        std::ofstream output(unpinned_file, std::ios::binary);
        output << "unpinned";
    }
    {
        std::ofstream output(pinned_file, std::ios::binary);
        output << "pinned";
    }
    store->register_model("unpinned", false);
    store->register_model("pinned", true);

    host.stop();
    CHECK(!std::filesystem::exists(unpinned_file));
    CHECK(std::filesystem::exists(pinned_file));
    return true;
}

bool test_shutdown_cancels_pending_load_and_restore(
    const std::filesystem::path& executable) {
    TempDirectory temp;
    const std::string lock_name = "mantic-mind-node-test-" +
                                  mm::util::generate_uuid();
    auto options = make_options(temp, lock_name, "startup-cancel", false);
    options.config.llama_server_path = executable.string();
    mm::NodeHost host(std::move(options));
    std::string error;

    CHECK(host.initialize(&error));
    CHECK(run_startup_cancel_case(
        host, temp.path() / "never-healthy-load.gguf", false));
    host.stop();

    // Reinitialization creates a fresh SlotManager and clears the host's
    // shutdown state before exercising the restore path.
    CHECK(host.initialize(&error));
    CHECK(run_startup_cancel_case(
        host, temp.path() / "never-healthy-restore.gguf", true));
    host.stop();
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const int runtime_result = run_never_healthy_runtime(argc, argv);
    if (runtime_result >= 0) return runtime_result;

    const auto executable = std::filesystem::absolute(argv[0]);
    if (!test_lock_graph_adapter_and_restart()) return 1;
    std::cout << "[PASS] lock_graph_adapter_and_restart\n";
    if (!test_optional_remote_cache_is_host_owned()) return 1;
    std::cout << "[PASS] optional_remote_cache_is_host_owned\n";
    if (!test_initialization_failure_releases_lock()) return 1;
    std::cout << "[PASS] initialization_failure_releases_lock\n";
    if (!test_cache_cleanup_preserves_pinned_models()) return 1;
    std::cout << "[PASS] cache_cleanup_preserves_pinned_models\n";
    if (!test_shutdown_cancels_pending_load_and_restore(executable)) return 1;
    std::cout << "[PASS] shutdown_cancels_pending_load_and_restore\n";
    return 0;
}
