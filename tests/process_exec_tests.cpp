#include "common/process_exec.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <Windows.h>
#else
#  include <csignal>
#  include <sys/types.h>
#  include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

[[noreturn]] void wait_forever() {
    for (;;) std::this_thread::sleep_for(std::chrono::hours(1));
}

int spawn_descendant(const std::string& exe, const fs::path& ready_marker = {}) {
#if defined(_WIN32)
    std::string command = "\"" + exe + "\" --grandchild";
    std::vector<char> mutable_command(command.begin(), command.end());
    mutable_command.push_back('\0');

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(exe.c_str(), mutable_command.data(), nullptr, nullptr, TRUE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        std::cerr << "helper CreateProcess failed: " << GetLastError() << "\n";
        return 2;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
#else
    (void)exe;
    const pid_t child = ::fork();
    if (child < 0) {
        std::cerr << "helper fork failed\n";
        return 2;
    }
    if (child == 0) wait_forever();
#endif

    if (!ready_marker.empty()) {
        std::ofstream marker(ready_marker, std::ios::trunc);
        if (!marker) return 3;
        marker << "ready\n";
    }
    std::cout << "descendant-ready" << std::endl;
    wait_forever();
}

fs::path model_dir_argument(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--model-dir") return fs::path(argv[i + 1]);
    }
    return {};
}

} // namespace

int main(int argc, char** argv) {
    std::error_code ec;
    const std::string exe = fs::absolute(argv[0], ec).string();

    if (argc > 1 && std::string(argv[1]) == "--grandchild") wait_forever();
    if (argc > 1 && std::string(argv[1]) == "--spawn-descendant") {
        return spawn_descendant(exe);
    }
    // ControlModelRegistry invokes its architecture probe as `soma plan ...`.
    // This mode makes that probe long-running and leaves a descendant holding
    // the inherited output pipes, which exercises registry shutdown and process
    // tree cancellation together from mm_reliability_tests.
    if (argc > 1 && std::string(argv[1]) == "plan") {
        const auto model_dir = model_dir_argument(argc, argv);
        if (model_dir.empty()) return 4;
        return spawn_descendant(exe, model_dir / "process-helper-ready");
    }

    std::atomic<bool> cancel{false};
    std::atomic<bool> saw_descendant{false};
    std::string error;
    const auto started = std::chrono::steady_clock::now();
    const int rc = mm::run_streamed_command(
        {exe, "--spawn-descendant"}, {},
        [&](const std::string& line, bool) {
            if (line == "descendant-ready") {
                saw_descendant.store(true);
                cancel.store(true);
            }
        },
        [&] { return cancel.load(); }, &error);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    if (!saw_descendant.load()) {
        std::cerr << "FAIL: descendant never became ready\n";
        return 1;
    }
    if (rc != 130 || error != "command canceled") {
        std::cerr << "FAIL: cancel returned " << rc << " with '" << error << "'\n";
        return 1;
    }
    if (elapsed >= 5.0) {
        std::cerr << "FAIL: cancel waited " << elapsed
                  << "s; a descendant likely retained the output pipes\n";
        return 1;
    }

    std::cout << "process_exec: OK (tree canceled in " << elapsed << "s)\n";
    return 0;
}
