// Mantic-Mind — EngineProcess: launch one engine, decide it is serving, and
// notice when it dies.
//
// Engine-neutral by construction. It receives an EngineLaunchSpec and never
// learns which engine produced it — that is what lets llama.cpp and Soma be two
// descriptors rather than two code paths. `RuntimeProcess::start_with_args` was
// already this shape and was private, with `start_llama_server` as a ~15-line
// adapter over it; this promotes it.
//
// The crash watchdog is NEW. Today a dead engine stays Ready until a request
// happens to fail, so the node advertises capacity it does not have and the
// scheduler keeps placing work on it. Streaming makes crashes likelier — I/O
// pressure, OOM under a mis-sized cache cap — so it is fixed as part of the
// rebuild rather than carried forward.

#include "node/engine_process.hpp"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#else
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace mm {

namespace {

/// Poll the child's health endpoint until it answers, the child dies, or the
/// budget runs out.
///
/// Readiness is an HTTP poll rather than a stdout sentinel on purpose: a
/// sentinel depends on the child's line buffering, and on Windows a child that
/// buffers its stdout looks identical to a child that never started.
bool poll_http_ready(const std::string& host,
                     std::uint16_t port,
                     const std::string& path,
                     int timeout_s,
                     const std::function<bool()>& still_alive) {
    httplib::Client cli(host, port);
    cli.set_connection_timeout(1, 0);
    cli.set_read_timeout(2, 0);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        // Checked EVERY iteration, not once at the end. A child that exits two
        // seconds in should fail in two seconds, not after the full 600-second
        // budget — the difference between a clear error and an apparent hang.
        if (still_alive && !still_alive()) return false;

        if (auto res = cli.Get(path.c_str()); res && res->status == 200) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    return false;
}

} // namespace

struct EngineProcess::Impl {
    std::mutex mu;
    std::atomic<ProcessState> state{ProcessState::Stopped};
    std::uint16_t port = 0;
    std::string last_error;
    std::string exe;

    LogCallback on_log;
    CrashCallback on_crash;

    /// Set before a deliberate stop, so the watchdog can tell a crash from a
    /// shutdown. Without it every clean stop would fire the crash callback and
    /// the signal would be worthless.
    std::atomic<bool> stopping{false};
    std::thread watchdog;

#if defined(_WIN32)
    HANDLE proc = nullptr;
    DWORD pid = 0;
#else
    pid_t pid = 0;
#endif

    bool child_alive() {
#if defined(_WIN32)
        if (proc == nullptr) return false;
        DWORD code = 0;
        return GetExitCodeProcess(proc, &code) && code == STILL_ACTIVE;
#else
        if (pid <= 0) return false;
        return ::kill(pid, 0) == 0;
#endif
    }

    /// During startup there is no watchdog waiting on the child yet, so POSIX
    /// must reap a terminated child here. `kill(pid, 0)` alone reports a zombie
    /// as alive and would make a startup error consume the entire readiness
    /// budget (plus the shutdown grace period).
    bool startup_child_alive() {
#if defined(_WIN32)
        return child_alive();
#else
        if (pid <= 0) return false;
        int status = 0;
        const pid_t result = ::waitpid(pid, &status, WNOHANG);
        if (result == 0) return true;
        if (result == pid) {
            pid = 0;
            return false;
        }
        // A transient interruption does not prove that the child exited. For
        // any other error, retain the existing liveness probe as a fallback.
        return ::kill(pid, 0) == 0;
#endif
    }

    void reap_and_notify() {
        int code = -1;
#if defined(_WIN32)
        if (proc != nullptr) {
            WaitForSingleObject(proc, INFINITE);
            DWORD c = 0;
            if (GetExitCodeProcess(proc, &c)) code = static_cast<int>(c);
        }
#else
        if (pid > 0) {
            int status = 0;
            if (::waitpid(pid, &status, 0) == pid) {
                code = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
            }
        }
#endif
        if (stopping.load()) return; // a clean shutdown, not a crash

        state.store(ProcessState::Crashed);
        {
            std::lock_guard<std::mutex> lk(mu);
            last_error = "engine exited unexpectedly with code " + std::to_string(code);
        }
        CrashCallback cb;
        {
            std::lock_guard<std::mutex> lk(mu);
            cb = on_crash;
        }
        if (cb) cb(code, "engine exited unexpectedly");
    }

    void kill_child() {
#if defined(_WIN32)
        if (proc != nullptr) {
            TerminateProcess(proc, 1);
            WaitForSingleObject(proc, 5000);
            CloseHandle(proc);
            proc = nullptr;
        }
#else
        if (pid > 0) {
            // Graceful first. Soma persists KV on shutdown, so a hard kill costs
            // the warm-reopen state that made preemption nearly free.
            ::kill(pid, SIGTERM);
            for (int i = 0; i < 50 && child_alive(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (child_alive()) ::kill(pid, SIGKILL);
            int status = 0;
            ::waitpid(pid, &status, 0);
            pid = 0;
        }
#endif
    }

    bool spawn(const EngineLaunchSpec& spec) {
#if defined(_WIN32)
        std::string cmd = "\"" + spec.executable + "\"";
        for (const auto& a : spec.args)
            cmd += " \"" + a + "\"";

        std::string env_block;
        bool have_env = !spec.env.empty();
        if (have_env) {
            for (LPWCH p = GetEnvironmentStringsW(); p != nullptr;) {
                (void)p;
                break; // inherit implicitly; explicit vars appended below
            }
            for (const auto& [k, v] : spec.env)
                env_block += k + "=" + v + '\0';
            env_block += '\0';
        }

        STARTUPINFOA si{};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        std::vector<char> mutable_cmd(cmd.begin(), cmd.end());
        mutable_cmd.push_back('\0');

        // Env vars are set on THIS process before spawning rather than passed as
        // a block: a partial block replaces the child's whole environment, which
        // strips PATH and makes the child fail to find its own DLLs — a failure
        // that presents as "the engine will not start" with no further detail.
        for (const auto& [k, v] : spec.env)
            SetEnvironmentVariableA(k.c_str(), v.c_str());

        const BOOL ok = CreateProcessA(nullptr,
                                       mutable_cmd.data(),
                                       nullptr,
                                       nullptr,
                                       FALSE,
                                       CREATE_NO_WINDOW,
                                       nullptr,
                                       nullptr,
                                       &si,
                                       &pi);
        if (!ok) {
            last_error = "CreateProcess failed for " + spec.executable + " (error " +
                         std::to_string(GetLastError()) + ")";
            return false;
        }
        CloseHandle(pi.hThread);
        proc = pi.hProcess;
        pid = pi.dwProcessId;
        return true;
#else
        pid = ::fork();
        if (pid < 0) {
            last_error = "fork failed";
            return false;
        }
        if (pid == 0) {
            for (const auto& [k, v] : spec.env)
                ::setenv(k.c_str(), v.c_str(), 1);
            std::vector<char*> argv;
            argv.push_back(const_cast<char*>(spec.executable.c_str()));
            for (const auto& a : spec.args)
                argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);
            ::execv(spec.executable.c_str(), argv.data());
            ::_exit(127);
        }
        return true;
#endif
    }
};

EngineProcess::EngineProcess() : impl_(std::make_unique<Impl>()) {}

EngineProcess::~EngineProcess() {
    stop();
}

void EngineProcess::set_log_callback(LogCallback cb) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->on_log = std::move(cb);
}

void EngineProcess::set_crash_callback(CrashCallback cb) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->on_crash = std::move(cb);
}

bool EngineProcess::start(const EngineLaunchSpec& spec) {
    if (impl_->state.load() != ProcessState::Stopped) stop();

    auto& im = *impl_;
    im.stopping.store(false);
    im.port = spec.port;
    im.exe = spec.executable;
    runtime_name_ = spec.runtime_name;
    im.state.store(ProcessState::Starting);
    {
        std::lock_guard<std::mutex> lk(im.mu);
        im.last_error.clear();
    }

    if (!im.spawn(spec)) {
        im.state.store(ProcessState::Error);
        return false;
    }

    const int budget = spec.readiness.timeout_seconds > 0 ? spec.readiness.timeout_seconds : 600;
    bool ready = false;
    if (spec.readiness.kind == ReadinessProbe::Kind::HttpHealth) {
        ready = poll_http_ready("127.0.0.1", spec.port, spec.readiness.http_path, budget, [&] {
            return im.startup_child_alive();
        });
    } else {
        // The stdout-sentinel path is declared in the header for engines that
        // have no health endpoint. Soma and llama-server both do, so it has no
        // caller yet, and shipping an untested implementation of it would be
        // worse than saying so.
        std::lock_guard<std::mutex> lk(im.mu);
        im.last_error = "StdoutJsonLine readiness is not implemented; both current engines "
                        "expose an HTTP health endpoint";
        im.state.store(ProcessState::Error);
        im.kill_child();
        return false;
    }

    if (!ready) {
        const bool died = !im.child_alive();
        {
            std::lock_guard<std::mutex> lk(im.mu);
            im.last_error =
                died ? "engine exited before becoming ready"
                     : "engine did not become ready within " + std::to_string(budget) + "s";
        }
        im.stopping.store(true);
        im.kill_child();
        im.state.store(ProcessState::Error);
        return false;
    }

    im.state.store(ProcessState::Ready);
    // Watchdog starts only AFTER readiness, so a start-up failure is reported by
    // start() returning false rather than as a crash callback racing it.
    im.watchdog = std::thread([p = impl_.get()] { p->reap_and_notify(); });
    return true;
}

void EngineProcess::stop() {
    auto& im = *impl_;
    if (im.state.load() == ProcessState::Stopped) return;

    im.stopping.store(true);
    im.kill_child();
    if (im.watchdog.joinable()) im.watchdog.join();
    im.state.store(ProcessState::Stopped);
}

ProcessState EngineProcess::state() const {
    return impl_->state.load();
}

std::uint16_t EngineProcess::port() const {
    return impl_->port;
}

std::string EngineProcess::url() const {
    return "http://127.0.0.1:" + std::to_string(impl_->port);
}

std::string EngineProcess::last_error() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->last_error;
}

bool EngineProcess::alive() const {
    return impl_->child_alive();
}

std::uint32_t EngineProcess::pid() const {
    return impl_->child_alive() ? static_cast<std::uint32_t>(impl_->pid) : 0u;
}

} // namespace mm
