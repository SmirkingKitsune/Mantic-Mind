#include "aio/aio_host.hpp"
#include "common/http_client.hpp"
#include "common/node_discovery.hpp"
#include "node/singleton_lock.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __func__ << ':' << __LINE__                            \
                      << ": check failed: " #condition "\n";                  \
            return false;                                                       \
        }                                                                       \
    } while (false)

class TempDirectory {
public:
    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now()
                               .time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("mantic-aio-host-integration-" + std::to_string(stamp));
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

void write_file(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("failed to create " + path.string());
    output << content;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed to read " + path.string());
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

class CountingRememberedNode {
public:
    CountingRememberedNode() {
        server_.set_pre_routing_handler(
            [this](const httplib::Request&, httplib::Response& response) {
                ++requests_;
                response.status = 503;
                response.set_content(R"({"error":"unexpected request"})",
                                     "application/json");
                return httplib::Server::HandlerResponse::Handled;
            });
        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) throw std::runtime_error("failed to bind fake node");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        for (int attempt = 0; attempt < 500 && !server_.is_running(); ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!server_.is_running()) {
            server_.stop();
            if (thread_.joinable()) thread_.join();
            throw std::runtime_error("fake remembered node did not start");
        }
    }

    ~CountingRememberedNode() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    std::string url() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }

    int requests() const { return requests_.load(); }

private:
    httplib::Server server_;
    std::thread thread_;
    int port_ = 0;
    std::atomic<int> requests_{0};
};

class HealthyRememberedNode {
public:
    HealthyRememberedNode() {
        server_.set_pre_routing_handler(
            [this](const httplib::Request&, httplib::Response&) {
                ++requests_;
                return httplib::Server::HandlerResponse::Unhandled;
            });
        server_.Get("/api/node/health", [](const httplib::Request&,
                                            httplib::Response& response) {
            response.set_content(nlohmann::json{
                {"cpu_percent", 12.0},
                {"ram_percent", 34.0},
                {"gpu_percent", 56.0},
                {"gpu_vram_used_mb", 1024},
                {"gpu_vram_total_mb", 8192},
                {"ram_used_mb", 4096},
                {"ram_total_mb", 16384},
                {"disk_free_mb", 50000},
                {"gpu_backend_available", true},
            }.dump(), "application/json");
        });
        server_.Get("/api/node/status", [](const httplib::Request&,
                                            httplib::Response& response) {
            response.set_content(nlohmann::json{
                {"node_id", "remembered-remote"},
                {"hostname", "healthy-remote"},
                {"slots", nlohmann::json::array()},
                {"max_slots", 2},
                {"slot_in_use", 0},
                {"slot_available", 2},
                {"slot_ready", 0},
                {"slot_loading", 0},
                {"slot_suspending", 0},
                {"slot_suspended", 0},
                {"slot_error", 0},
                {"disk_free_mb", 50000},
            }.dump(), "application/json");
        });
        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) throw std::runtime_error("failed to bind healthy fake node");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        for (int attempt = 0; attempt < 500 && !server_.is_running(); ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!server_.is_running()) {
            server_.stop();
            if (thread_.joinable()) thread_.join();
            throw std::runtime_error("healthy fake node did not start");
        }
    }

    ~HealthyRememberedNode() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    std::string url() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }

    int requests() const { return requests_.load(); }

private:
    httplib::Server server_;
    std::thread thread_;
    int port_ = 0;
    std::atomic<int> requests_{0};
};

uint16_t find_free_loopback_port() {
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    if (port <= 0) throw std::runtime_error("failed to reserve control port");
    std::thread listener([&probe] { probe.listen_after_bind(); });
    for (int attempt = 0; attempt < 500 && !probe.is_running(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    probe.stop();
    if (listener.joinable()) listener.join();
    if (port > 65535) throw std::runtime_error("invalid ephemeral port");
    return static_cast<uint16_t>(port);
}

mm::AioConfig make_config(const TempDirectory& temp, uint16_t control_port) {
    mm::AioConfig config;
    config.shared.data_dir = (temp.path() / "data").string();
    config.shared.models_dir = (temp.path() / "models").string();
    config.shared.log_file = (temp.path() / "logs" / "aio.log").string();

    config.control.bind_host = "127.0.0.1";
    config.control.listen_port = control_port;
    config.control.openai_compat_port = 0;
    config.control.node_health_poll_interval_s = 1;
    config.control.node_offline_after_s = 2;
    config.control.external_api_token.clear();
    config.control.tts.enabled = false;

    config.node.llama_server_path =
        (temp.path() / "missing-llama-server").string();
    config.node.llama_provision_dir =
        (temp.path() / "data" / "runtimes" / "llama.cpp").string();
    config.node.llama_auto_provision = false;
    config.node.llama_update_check = false;
    config.node.llama_update_policy = "manual";
    config.node.runtime_network_policy = mm::AioRuntimeNetworkPolicy::Prompt;
    config.node.kv_cache_dir = (temp.path() / "data" / "kv_cache").string();
    config.node.max_slots = 1;

    config.cluster.enabled = false;
    config.cluster.discovery_enabled = false;
    config.cluster.pairing_key.clear();
    return config;
}

nlohmann::json parse_response(const mm::HttpResponse& response) {
    return nlohmann::json::parse(response.body);
}

bool check_local_node_snapshot(mm::HttpClient& client,
                               std::string* local_id = nullptr) {
    const auto response = client.get("/v1/nodes");
    CHECK(response.status == 200);
    const auto nodes = parse_response(response);
    CHECK(nodes.is_array());
    CHECK(nodes.size() == 1);
    const auto& local = nodes.at(0);
    CHECK(local.value("id", std::string{}) == "local");
    CHECK(local.value("kind", std::string{}) == "embedded");
    CHECK(local.value("url", std::string{"unexpected"}).empty());
    CHECK(!local.contains("api_key"));
    CHECK(local.value("connected", false));
    CHECK(local.value("remembered", true) == false);
    if (local_id) *local_id = local.value("id", std::string{});
    return true;
}

bool test_aio_host_local_only_lifecycle() {
    TempDirectory temp;
    CountingRememberedNode remembered_node;
    const uint16_t control_port = find_free_loopback_port();
    const auto config = make_config(temp, control_port);
    const auto nodes_path = temp.path() / "data" / "nodes.json";
    const std::string remembered_contents = nlohmann::json{
        {"version", 2},
        {"nodes", nlohmann::json::array({
            {
                {"id", "remembered-remote"},
                {"url", remembered_node.url()},
                {"hostname", "must-not-be-contacted"},
                {"api_key", "remembered-secret"},
                {"platform", "test"},
            },
        })},
    }.dump(2) + "\n";
    write_file(nodes_path, remembered_contents);

    const std::string base_url =
        "http://127.0.0.1:" + std::to_string(control_port);
    std::string first_local_id;

    mm::AioHost host(config, false);
    {
        std::string error;
        CHECK(host.start(true, &error));
        CHECK(error.empty());

        mm::HttpClient client(base_url);
        client.set_timeouts(2, 5, 5);
        CHECK(check_local_node_snapshot(client, &first_local_id));

        const auto status = client.get("/v1/nodes/local/status");
        CHECK(status.status == 200);
        const auto status_body = parse_response(status);
        CHECK(status_body.value("node_id", std::string{}) == "local");
        CHECK(status_body.contains("slots"));
        CHECK(status_body.at("slots").is_array());

        const auto logs = client.get("/v1/nodes/local/logs?tail=25");
        CHECK(logs.status == 200);
        const auto logs_body = parse_response(logs);
        CHECK(logs_body.contains("lines"));
        CHECK(logs_body.at("lines").is_array());

        const auto remove_local = client.del("/v1/nodes/local");
        CHECK(remove_local.status == 409);

        const auto pair_local = client.post(
            "/v1/nodes/pair/start", nlohmann::json{{"node_id", "local"}});
        CHECK(pair_local.status == 409);

        const auto add_remote = client.post("/v1/nodes", nlohmann::json{
            {"url", remembered_node.url()},
            {"api_key", "new-secret"},
            {"remember", true},
        });
        CHECK(add_remote.status == 403);

        const auto forget_remote = client.post(
            "/v1/nodes/remembered-remote/forget", nlohmann::json::object());
        CHECK(forget_remote.status == 403);

        const auto cancel_remote = client.post(
            "/v1/nodes/remembered-remote/actions/cancel",
            nlohmann::json::object());
        CHECK(cancel_remote.status == 403);

        const auto provision_remote = client.post(
            "/v1/nodes/remembered-remote/runtime/llama/provision",
            nlohmann::json{{"allow_network", true}});
        CHECK(provision_remote.status == 403);

        const auto provision = client.post(
            "/v1/nodes/local/runtime/llama/provision",
            nlohmann::json::object());
        CHECK(provision.status == 403);
        const auto provision_body = parse_response(provision);
        CHECK(provision_body.value("requires_network_consent", false));
        CHECK(provision_body.value("runtime_network_policy", std::string{}) ==
              "prompt");

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(remembered_node.requests() == 0);
        CHECK(read_file(nodes_path) == remembered_contents);

        host.stop();
        host.stop();
    }

    // Starting the same configuration while the stopped first host object is
    // still alive proves stop(), rather than destruction, released both
    // singleton locks and the listener socket.
    {
        mm::AioHost restarted(config, false);
        std::string error;
        CHECK(restarted.start(true, &error));
        CHECK(error.empty());

        mm::HttpClient client(base_url);
        client.set_timeouts(2, 5, 5);
        std::string restarted_local_id;
        CHECK(check_local_node_snapshot(client, &restarted_local_id));
        CHECK(restarted_local_id == first_local_id);
        CHECK(restarted_local_id == "local");
        CHECK(remembered_node.requests() == 0);
        CHECK(read_file(nodes_path) == remembered_contents);
        restarted.stop();
    }

    CHECK(remembered_node.requests() == 0);
    CHECK(read_file(nodes_path) == remembered_contents);
    return true;
}

bool test_aio_startup_rollback_and_singleton_conflicts() {
    TempDirectory temp;
    const uint16_t control_port = find_free_loopback_port();
    const auto config = make_config(temp, control_port);

    // Occupy the configured control port. start() must unwind its metrics,
    // workers, both singleton locks, and every partially constructed service.
    httplib::Server blocker;
    CHECK(blocker.bind_to_port("127.0.0.1", control_port));
    std::thread blocker_thread([&blocker] { blocker.listen_after_bind(); });
    for (int attempt = 0; attempt < 500 && !blocker.is_running(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(blocker.is_running());

    mm::AioHost rollback_host(config, false);
    std::string error;
    CHECK(!rollback_host.start(true, &error));
    CHECK(!error.empty());
    rollback_host.stop();
    blocker.stop();
    blocker_thread.join();

    error = "stale error";
    CHECK(rollback_host.start(true, &error));
    CHECK(error.empty());
    rollback_host.request_quit();
    rollback_host.stop();

    // The same object is reusable after request_quit/stop; start() resets all
    // run flags instead of carrying a stale quit/cancel state forward.
    CHECK(rollback_host.start(true, &error));
    CHECK(error.empty());
    rollback_host.stop();

    // A standalone node lock blocks AIO before it opens sockets, and releasing
    // it makes the exact same host configuration startable.
    auto standalone_node_lock = mm::SingletonLock::try_acquire();
    CHECK(standalone_node_lock != nullptr);
    mm::AioHost node_conflict(config, false);
    CHECK(!node_conflict.start(true, &error));
    CHECK(error.find("node instance") != std::string::npos);
    standalone_node_lock.reset();
    CHECK(node_conflict.start(true, &error));
    node_conflict.stop();

    // Two controls targeting the same data root/port conflict on the control
    // role lock. Once the owner stops, the second host can acquire both roles.
    mm::AioHost first(config, false);
    mm::AioHost second(config, false);
    CHECK(first.start(true, &error));
    CHECK(!second.start(true, &error));
    CHECK(error.find("control instance") != std::string::npos);
    first.stop();
    CHECK(second.start(true, &error));
    second.stop();
    return true;
}

bool test_cluster_enabled_loads_and_polls_remotes_without_persisting_local() {
    TempDirectory temp;
    HealthyRememberedNode remembered_node;
    const uint16_t control_port = find_free_loopback_port();
    auto config = make_config(temp, control_port);
    config.cluster.enabled = true;

    const auto nodes_path = temp.path() / "data" / "nodes.json";
    const nlohmann::json remembered_file{
        {"version", 2},
        {"nodes", nlohmann::json::array({
            {
                {"id", "remembered-remote"},
                {"url", remembered_node.url()},
                {"hostname", "stored-hostname"},
                {"api_key", "remembered-secret"},
                {"platform", "test"},
            },
        })},
    };
    write_file(nodes_path, remembered_file.dump(2) + "\n");

    mm::AioHost host(config, false);
    std::string error;
    CHECK(host.start(true, &error));
    CHECK(error.empty());

    mm::HttpClient client(
        "http://127.0.0.1:" + std::to_string(control_port));
    client.set_timeouts(2, 5, 5);

    nlohmann::json nodes;
    bool remote_online = false;
    for (int attempt = 0; attempt < 200 && !remote_online; ++attempt) {
        const auto response = client.get("/v1/nodes");
        CHECK(response.status == 200);
        nodes = parse_response(response);
        for (const auto& node : nodes) {
            if (node.value("id", std::string{}) == "remembered-remote") {
                remote_online = node.value("connected", false);
                break;
            }
        }
        if (!remote_online) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    CHECK(remote_online);
    CHECK(nodes.size() == 2);
    bool saw_local = false;
    bool saw_remote = false;
    for (const auto& node : nodes) {
        const std::string id = node.value("id", std::string{});
        if (id == "local") {
            saw_local = true;
            CHECK(node.value("kind", std::string{}) == "embedded");
            CHECK(node.value("url", std::string{"unexpected"}).empty());
        } else if (id == "remembered-remote") {
            saw_remote = true;
            CHECK(node.value("kind", std::string{}) == "remote");
            CHECK(node.value("url", std::string{}) == remembered_node.url());
            CHECK(!node.contains("api_key"));
        }
    }
    CHECK(saw_local);
    CHECK(saw_remote);
    CHECK(remembered_node.requests() >= 2);

    const auto remote_status =
        client.get("/v1/nodes/remembered-remote/status");
    CHECK(remote_status.status == 200);
    CHECK(parse_response(remote_status).value("hostname", std::string{}) ==
          "healthy-remote");

    host.stop();

    const auto persisted = nlohmann::json::parse(read_file(nodes_path));
    CHECK(persisted.value("version", 0) == 2);
    CHECK(persisted.at("nodes").size() == 1);
    CHECK(persisted.at("nodes").at(0).value("id", std::string{}) ==
          "remembered-remote");
    return true;
}

bool test_discovery_bind_failure_rolls_back_and_restarts() {
    TempDirectory temp;
    const uint16_t control_port = find_free_loopback_port();
    auto config = make_config(temp, control_port);
    config.cluster.enabled = true;
    config.cluster.discovery_enabled = true;

    mm::NodeDiscoveryListener blocker;
    CHECK(blocker.start(0));
    CHECK(blocker.bound_port() != 0);
    config.cluster.discovery_port = blocker.bound_port();

    mm::AioHost host(config, false);
    std::string error;
    CHECK(!host.start(true, &error));
    CHECK(error.find("discovery") != std::string::npos);
    host.stop();

    blocker.stop();
    error = "stale error";
    CHECK(host.start(true, &error));
    CHECK(error.empty());
    host.stop();
    return true;
}

} // namespace

int main() {
    if (!test_aio_host_local_only_lifecycle()) return 1;
    std::cout << "[PASS] aio_host_local_only_lifecycle\n";
    if (!test_aio_startup_rollback_and_singleton_conflicts()) return 1;
    std::cout << "[PASS] aio_startup_rollback_and_singleton_conflicts\n";
    if (!test_cluster_enabled_loads_and_polls_remotes_without_persisting_local())
        return 1;
    std::cout << "[PASS] cluster_enabled_loads_and_polls_remotes_without_persisting_local\n";
    if (!test_discovery_bind_failure_rolls_back_and_restarts()) return 1;
    std::cout << "[PASS] discovery_bind_failure_rolls_back_and_restarts\n";
    return 0;
}
