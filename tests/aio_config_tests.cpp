#include "aio/aio_config.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __func__ << ':' << __LINE__                           \
                      << ": check failed: " #condition "\n";                  \
            return false;                                                       \
        }                                                                       \
    } while (false)

void set_environment(const std::string& name,
                     const std::optional<std::string>& value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value ? value->c_str() : "");
#else
    if (value) setenv(name.c_str(), value->c_str(), 1);
    else unsetenv(name.c_str());
#endif
}

class ScopedEnvironment {
public:
    explicit ScopedEnvironment(std::string name)
        : name_(std::move(name)) {
        if (const char* current = std::getenv(name_.c_str())) {
            previous_ = std::string(current);
        }
    }

    ~ScopedEnvironment() { set_environment(name_, previous_); }

    void set(const std::string& value) { set_environment(name_, value); }
    void unset() { set_environment(name_, std::nullopt); }

private:
    std::string name_;
    std::optional<std::string> previous_;
};

class TempDirectory {
public:
    explicit TempDirectory(const std::string& label) {
        const auto stamp = std::chrono::steady_clock::now()
                               .time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("mantic-aio-config-" + label + "-" +
                 std::to_string(stamp));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

void write_file(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << text;
}

bool has_issue(const std::vector<mm::AioConfigIssue>& issues,
               const std::string& key,
               const std::string& fragment = {}) {
    for (const auto& issue : issues) {
        if (issue.key != key) continue;
        if (fragment.empty() || issue.message.find(fragment) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string minimal_file(uint16_t control_port) {
    return
        "[shared]\n"
        "data_dir = \"state\"\n"
        "models_dir = \"weights\"\n"
        "log_file = \"logs/aio.log\"\n"
        "\n"
        "[control]\n"
        "bind_host = \"127.0.0.1\"\n"
        "listen_port = " + std::to_string(control_port) + "\n"
        "openai_compat_port = 0\n"
        "\n"
        "[node]\n"
        "runtime_port_range_start = 18080\n"
        "runtime_port_range_end = 18083\n"
        "max_slots = 4\n"
        "runtime_network_policy = \"prompt\"\n"
        "\n"
        "[cluster]\n"
        "enabled = false\n"
        "discovery_enabled = false\n";
}

bool test_defaults_and_materialization() {
    TempDirectory temp("defaults");
    mm::AioConfigLoadOptions options;
    options.search_start = temp.path();
    options.upward_search_limit = 1;
    options.use_environment = false;

    const auto result = mm::load_aio_config(options);
    CHECK(result.ok());
    CHECK(result.source_path.empty());
    CHECK(result.config.node.runtime_network_policy ==
          mm::AioRuntimeNetworkPolicy::Prompt);
    CHECK(mm::aio_host_is_loopback(result.config.control.bind_host));

    auto configured = result.config;
    configured.shared.data_dir = "state-root";
    configured.shared.models_dir = "model-root";
    configured.shared.log_file = "aio.log";
    configured.cluster.discovery_port = 7332;
    configured.cluster.pairing_key = "pair-secret";

    const auto control = mm::make_control_config(configured);
    const auto node = mm::make_node_config(configured);
    CHECK(control.data_dir == "state-root");
    CHECK(control.models_dir == "model-root");
    CHECK(control.log_file == "aio.log");
    CHECK(control.discovery_port == 7332);
    CHECK(control.pairing_key == "pair-secret");
    CHECK(node.listen_port == 0);
    CHECK(node.control_url.empty());
    CHECK(node.discovery_port == 0);
    CHECK(node.data_dir == "state-root");
    CHECK(node.models_dir == "model-root");
    CHECK(std::filesystem::path(node.kv_cache_dir) ==
          std::filesystem::path("state-root") / "kv_cache");
    CHECK(std::filesystem::path(node.llama_provision_dir) ==
          std::filesystem::path("state-root") / "runtimes" / "llama.cpp");
    return true;
}

bool test_namespaced_file_and_list_parsing() {
    TempDirectory temp("sections");
    const auto config_path = temp.path() / "aio.toml";
    write_file(
        config_path,
        "[shared]\n"
        "data_dir = \"state#one\" # comment\n"
        "models_dir = \"weights\"\n"
        "log_file = \"aio.log\"\n"
        "[control]\n"
        "bind_host = \"localhost\"\n"
        "listen_port = 19090\n"
        "openai_compat_port = 19091\n"
        "node_health_poll_interval_s = 7\n"
        "[node]\n"
        "runtime_port_range_start = 18100\n"
        "runtime_port_range_end = 18105\n"
        "max_slots = 6\n"
        "runtime_network_policy = \"deny\"\n"
        "llama_cmake_args = [\"-DGGML_CUDA=ON\", \"-DTEST=A#B\"]\n"
        "[cluster]\n"
        "enabled = false\n"
        "discovery_enabled = false\n");

    mm::AioConfigLoadOptions options;
    options.explicit_path = config_path;
    options.use_environment = false;
    const auto result = mm::load_aio_config(options);
    CHECK(result.ok());
    CHECK(result.config.shared.data_dir == "state#one");
    CHECK(result.config.control.listen_port == 19090);
    CHECK(result.config.control.openai_compat_port == 19091);
    CHECK(result.config.control.node_health_poll_interval_s == 7);
    CHECK(result.config.node.runtime_network_policy ==
          mm::AioRuntimeNetworkPolicy::Deny);
    CHECK(result.config.node.llama_cmake_args.size() == 2);
    CHECK(result.config.node.llama_cmake_args[0] == "-DGGML_CUDA=ON");
    CHECK(result.config.node.llama_cmake_args[1] == "-DTEST=A#B");
    return true;
}

bool test_file_and_environment_precedence() {
    TempDirectory temp("precedence");
    const auto search_dir = temp.path() / "nested" / "child";
    std::filesystem::create_directories(search_dir);
    const auto upward_path = temp.path() / "mantic-mind-aio.toml";
    const auto environment_path = temp.path() / "environment.toml";
    const auto explicit_path = temp.path() / "explicit.toml";
    write_file(upward_path, minimal_file(19100));
    write_file(environment_path, minimal_file(19200));
    write_file(explicit_path, minimal_file(19300));

    ScopedEnvironment config_env("MM_AIO_CONFIG_FILE");
    ScopedEnvironment port_env("MM_AIO_CONTROL_LISTEN_PORT");
    ScopedEnvironment policy_env("MM_AIO_NODE_RUNTIME_NETWORK_POLICY");
    config_env.set(environment_path.string());
    port_env.unset();
    policy_env.unset();

    mm::AioConfigLoadOptions options;
    options.explicit_path = explicit_path;
    options.search_start = search_dir;
    auto result = mm::load_aio_config(options);
    CHECK(result.ok());
    CHECK(result.config.control.listen_port == 19300);
    CHECK(result.source_path.filename() == explicit_path.filename());

    options.explicit_path.reset();
    result = mm::load_aio_config(options);
    CHECK(result.ok());
    CHECK(result.config.control.listen_port == 19200);
    CHECK(result.source_path.filename() == environment_path.filename());

    config_env.unset();
    result = mm::load_aio_config(options);
    CHECK(result.ok());
    CHECK(result.config.control.listen_port == 19100);
    CHECK(result.source_path.filename() == upward_path.filename());

    options.explicit_path = explicit_path;
    port_env.set("19400");
    policy_env.set("allow");
    result = mm::load_aio_config(options);
    CHECK(result.ok());
    CHECK(result.config.control.listen_port == 19400);
    CHECK(result.config.node.runtime_network_policy ==
          mm::AioRuntimeNetworkPolicy::Allow);

    // A supplied but missing --config path is strict and never falls through to
    // MM_AIO_CONFIG_FILE or upward search.
    config_env.set(environment_path.string());
    port_env.unset();
    policy_env.unset();
    options.explicit_path = temp.path() / "missing.toml";
    result = mm::load_aio_config(options);
    CHECK(!result.ok());
    CHECK(result.source_path.filename() == "missing.toml");
    CHECK(result.config.control.listen_port == 9090);
    return true;
}

bool test_network_and_credential_validation() {
    mm::AioConfig config;
    config.control.bind_host = "0.0.0.0";
    auto issues = mm::validate_aio_config(config);
    CHECK(has_issue(issues, "control.bind_host", "cluster.enabled"));
    CHECK(has_issue(issues, "control.external_api_token", "required"));
    CHECK(has_issue(issues, "cluster.pairing_key", "required"));

    config.cluster.enabled = true;
    config.control.external_api_token = "external-secret";
    config.cluster.pairing_key = "pair-secret";
    issues = mm::validate_aio_config(config);
    CHECK(!has_issue(issues, "control.bind_host"));
    CHECK(!has_issue(issues, "control.external_api_token"));
    CHECK(!has_issue(issues, "cluster.pairing_key"));

    config.control.bind_host = "127.0.0.1";
    config.control.external_api_token.clear();
    config.cluster.pairing_key.clear();
    config.cluster.discovery_enabled = true;
    issues = mm::validate_aio_config(config);
    CHECK(!has_issue(issues, "cluster.pairing_key"));

    config.cluster.enabled = false;
    issues = mm::validate_aio_config(config);
    CHECK(has_issue(issues, "cluster.discovery_enabled", "must be false"));

    CHECK(mm::aio_host_is_loopback("127.10.20.30"));
    CHECK(mm::aio_host_is_loopback("http://[::1]:9090/path"));
    CHECK(!mm::aio_host_is_loopback("0.0.0.0"));
    CHECK(!mm::aio_host_is_loopback("192.168.1.10"));
    return true;
}

bool test_port_collision_and_range_validation() {
    mm::AioConfig config;
    config.control.listen_port = 8080;
    auto issues = mm::validate_aio_config(config);
    CHECK(has_issue(issues, "control.listen_port", "runtime port range"));

    config.control.listen_port = 9090;
    config.control.openai_compat_port = 9090;
    issues = mm::validate_aio_config(config);
    CHECK(has_issue(issues, "control.openai_compat_port", "differ"));

    config.control.openai_compat_port = 0;
    config.node.runtime_port_range_start = 8100;
    config.node.runtime_port_range_end = 8101;
    config.node.max_slots = 3;
    issues = mm::validate_aio_config(config);
    CHECK(has_issue(issues, "node.max_slots", "fit"));

    config.node.max_slots = 2;
    config.cluster.enabled = true;
    config.cluster.discovery_enabled = true;
    config.cluster.discovery_port = 8100;
    config.cluster.pairing_key = "pair-secret";
    issues = mm::validate_aio_config(config);
    CHECK(has_issue(issues, "cluster.discovery_port", "runtime port range"));
    return true;
}

bool test_poll_interval_validation() {
    mm::AioConfig config;

    config.control.node_health_poll_interval_s = 0;
    auto issues = mm::validate_aio_config(config);
    CHECK(has_issue(issues, "control.node_health_poll_interval_s", "1.."));

    config.control.node_health_poll_interval_s =
        static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1U;
    issues = mm::validate_aio_config(config);
    CHECK(has_issue(issues, "control.node_health_poll_interval_s", "1.."));

    config.control.node_health_poll_interval_s = 1;
    config.control.node_offline_after_s = 0;
    issues = mm::validate_aio_config(config);
    CHECK(has_issue(issues, "control.node_offline_after_s", "1.."));

    config.control.node_offline_after_s =
        static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1U;
    issues = mm::validate_aio_config(config);
    CHECK(has_issue(issues, "control.node_offline_after_s", "1.."));

    config.control.node_offline_after_s = 1;
    issues = mm::validate_aio_config(config);
    CHECK(!has_issue(issues, "control.node_health_poll_interval_s"));
    CHECK(!has_issue(issues, "control.node_offline_after_s"));
    return true;
}

} // namespace

int main() {
    struct TestCase {
        const char* name;
        bool (*run)();
    };
    const TestCase tests[] = {
        {"defaults_and_materialization", test_defaults_and_materialization},
        {"namespaced_file_and_list_parsing", test_namespaced_file_and_list_parsing},
        {"file_and_environment_precedence", test_file_and_environment_precedence},
        {"network_and_credential_validation", test_network_and_credential_validation},
        {"port_collision_and_range_validation", test_port_collision_and_range_validation},
        {"poll_interval_validation", test_poll_interval_validation},
    };

    for (const auto& test : tests) {
        if (!test.run()) return 1;
        std::cout << "[PASS] " << test.name << '\n';
    }
    return 0;
}
