#include "aio/aio_config.hpp"
#include "aio/aio_host.hpp"
#include "common/util.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <thread>

namespace {

enum class RunMode { Tui, Cli };

volatile std::sig_atomic_t shutdown_signal = 0;

extern "C" void handle_shutdown_signal(int signal_number) {
    shutdown_signal = signal_number;
}

struct Args {
    RunMode mode = RunMode::Tui;
    bool json_output = false;
    bool allow_network = false;
    bool help = false;
    std::optional<std::filesystem::path> config_path;
    std::string error;
};

Args parse_args(int argc, char** argv) {
    Args result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i] ? argv[i] : "";
        auto require_value = [&](const char* option) -> std::string {
            if (i + 1 >= argc) {
                result.error = std::string(option) + " requires a value";
                return {};
            }
            return argv[++i] ? argv[i] : "";
        };

        if (arg == "--help" || arg == "-h") {
            result.help = true;
        } else if (arg == "--allow-network") {
            result.allow_network = true;
        } else if (arg == "--config") {
            const auto value = require_value("--config");
            if (!result.error.empty()) return result;
            result.config_path = std::filesystem::path(value);
        } else if (arg.rfind("--config=", 0) == 0) {
            result.config_path = std::filesystem::path(arg.substr(9));
        } else if (arg == "--mode") {
            const auto value = mm::util::to_lower(require_value("--mode"));
            if (!result.error.empty()) return result;
            if (value == "tui") result.mode = RunMode::Tui;
            else if (value == "cli") result.mode = RunMode::Cli;
            else result.error = "--mode must be tui or cli";
        } else if (arg.rfind("--mode=", 0) == 0) {
            const auto value = mm::util::to_lower(arg.substr(7));
            if (value == "tui") result.mode = RunMode::Tui;
            else if (value == "cli") result.mode = RunMode::Cli;
            else result.error = "--mode must be tui or cli";
        } else if (arg == "--output") {
            const auto value = mm::util::to_lower(require_value("--output"));
            if (!result.error.empty()) return result;
            if (value == "json") result.json_output = true;
            else if (value == "text") result.json_output = false;
            else result.error = "--output must be text or json";
        } else if (arg.rfind("--output=", 0) == 0) {
            const auto value = mm::util::to_lower(arg.substr(9));
            if (value == "json") result.json_output = true;
            else if (value == "text") result.json_output = false;
            else result.error = "--output must be text or json";
        } else {
            result.error = "unknown argument: " + arg;
        }
        if (!result.error.empty()) return result;
    }
    return result;
}

void print_usage() {
    std::cout
        << "mantic-mind-aio - control and embedded node in one process\n\n"
        << "Usage: mantic-mind-aio [--config PATH] [--mode tui|cli] "
           "[--output text|json] [--allow-network]\n\n"
        << "AIO configuration resolution: --config, MM_AIO_CONFIG_FILE, then "
           "an upward search for mantic-mind-aio.toml.\n"
        << "--allow-network grants consent to CLI runtime download/update "
           "commands for this process.\n";
}

} // namespace

int main(int argc, char** argv) {
    const auto args = parse_args(argc, argv);
    if (!args.error.empty()) {
        std::cerr << "ERROR: " << args.error << "\n\n";
        print_usage();
        return 1;
    }
    if (args.help) {
        print_usage();
        return 0;
    }

    mm::AioConfigLoadOptions load_options;
    load_options.explicit_path = args.config_path;
    const auto loaded = mm::load_aio_config(load_options);
    if (!loaded.ok()) {
        std::cerr << "AIO configuration is invalid:\n";
        for (const auto& issue : loaded.issues) {
            std::cerr << "  " << issue.source;
            if (!issue.key.empty()) std::cerr << " [" << issue.key << "]";
            std::cerr << ": " << issue.message << '\n';
        }
        return 1;
    }

    mm::AioHost host(loaded.config, args.allow_network);
    shutdown_signal = 0;
    const auto previous_sigint = std::signal(SIGINT, handle_shutdown_signal);
    const auto previous_sigterm = std::signal(SIGTERM, handle_shutdown_signal);
    auto restore_signal_handlers = [&] {
        if (previous_sigint != SIG_ERR) std::signal(SIGINT, previous_sigint);
        if (previous_sigterm != SIG_ERR) std::signal(SIGTERM, previous_sigterm);
    };

    std::string start_error;
    if (!host.start(args.mode == RunMode::Tui, &start_error)) {
        restore_signal_handlers();
        std::cerr << "ERROR: " << start_error << '\n';
        return 1;
    }

    std::atomic_bool stop_signal_watcher{false};
    std::thread signal_watcher([&host, &stop_signal_watcher] {
        while (!stop_signal_watcher.load(std::memory_order_acquire)) {
            if (shutdown_signal != 0) {
                host.request_quit();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    });

    const int result = args.mode == RunMode::Tui
        ? host.run_tui()
        : host.run_cli(args.json_output);
    stop_signal_watcher.store(true, std::memory_order_release);
    if (signal_watcher.joinable()) signal_watcher.join();
    host.stop();
    restore_signal_handlers();
    return result;
}
