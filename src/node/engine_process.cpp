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

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

#if defined(_WIN32)
/// Build a complete child environment block: this process's environment with
/// `overrides` applied on top.
///
/// The child needs the WHOLE environment — PATH above all, or it cannot find
/// its own DLLs — which is why the previous code called SetEnvironmentVariableA
/// on the NODE instead of passing a block. That worked for one child and was
/// wrong for every child after it: the values were never restored, so the next
/// engine inherited the last one's CUDA_VISIBLE_DEVICES, RAY_ADDRESS and
/// HF_HOME, and two concurrent starts could interleave their writes and launch
/// both children pointed at the same GPU.
///
/// A block is `KEY=VALUE\0KEY=VALUE\0\0`, sorted case-insensitively — the
/// order is documented for the Unicode form and costs nothing to honour here.
/// Names are matched case-insensitively because Windows environment names are.
std::vector<char> build_child_environment(
    const std::vector<std::pair<std::string, std::string>>& overrides) {
    const auto ci_less = [](const std::string& a, const std::string& b) {
        return std::lexicographical_compare(
            a.begin(), a.end(), b.begin(), b.end(), [](unsigned char x, unsigned char y) {
                return std::tolower(x) < std::tolower(y);
            });
    };
    const auto ci_equal = [](const std::string& a, const std::string& b) {
        return a.size() == b.size() &&
               std::equal(a.begin(), a.end(), b.begin(), [](unsigned char x, unsigned char y) {
                   return std::tolower(x) == std::tolower(y);
               });
    };

    std::vector<std::pair<std::string, std::string>> entries;

    // GetEnvironmentStringsA hands back a block this function owns; it is freed
    // before returning. The old code called GetEnvironmentStringsW, broke out of
    // the loop immediately and never freed it — a leak on every engine start,
    // for a value it then did not use.
    if (LPCH block = GetEnvironmentStringsA(); block != nullptr) {
        for (const char* p = block; *p != '\0';) {
            const std::string entry(p);
            p += entry.size() + 1;
            // Windows keeps per-drive working directories as "=C:=C:\dir".
            // The name is "=C:", so the split must skip the leading '='.
            const auto eq = entry.find('=', entry.empty() ? 0 : 1);
            if (eq == std::string::npos) continue;
            entries.emplace_back(entry.substr(0, eq), entry.substr(eq + 1));
        }
        FreeEnvironmentStringsA(block);
    }

    for (const auto& [k, v] : overrides) {
        if (k.empty()) continue;
        const auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& e) {
            return ci_equal(e.first, k);
        });
        if (it != entries.end()) {
            it->second = v;
        } else {
            entries.emplace_back(k, v);
        }
    }

    std::sort(entries.begin(), entries.end(), [&](const auto& a, const auto& b) {
        return ci_less(a.first, b.first);
    });

    std::vector<char> out;
    for (const auto& [k, v] : entries) {
        out.insert(out.end(), k.begin(), k.end());
        out.push_back('=');
        out.insert(out.end(), v.begin(), v.end());
        out.push_back('\0');
    }
    // A block is terminated by an extra NUL, and an EMPTY block is still two:
    // handing CreateProcess a single NUL is a malformed block, not an empty one.
    if (out.empty()) out.push_back('\0');
    out.push_back('\0');
    return out;
}
#endif

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

    /// Guards the child's IDENTITY — `proc`/`pid` — and nothing else.
    ///
    /// Separate from `mu` because `mu` protects the callbacks, and the crash
    /// callback runs out to EngineSupervisor::on_engine_crash(), which takes the
    /// supervisor's own mutex. Identity must never be held across that call.
    ///
    /// The invariant this buys: the identity is cleared in the same critical
    /// section that reaps the child. Before, reap_and_notify() waited on the pid
    /// and left it set, so a later unload() or destructor called kill_child() and
    /// signalled a numeric pid the OS had since handed to something else — the
    /// node terminating an unrelated process on the box. Worse, stop() and the
    /// watchdog could both sit inside waitpid() on the same child at once,
    /// racing on a plain pid_t.
    std::mutex child_mu;
    std::condition_variable child_gone;

#if defined(_WIN32)
    HANDLE proc = nullptr;
    DWORD pid = 0;
#else
    pid_t pid = 0;
#endif

    /// Whoever reaps the child clears its identity here and wakes the waiters.
    /// Caller holds child_mu.
    void forget_child_locked() {
#if defined(_WIN32)
        if (proc != nullptr) {
            CloseHandle(proc);
            proc = nullptr;
        }
#endif
        pid = 0;
        child_gone.notify_all();
    }

    bool child_alive() {
        std::lock_guard<std::mutex> lk(child_mu);
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
        std::unique_lock<std::mutex> lk(child_mu);
        if (pid <= 0) return false;
        int status = 0;
        const pid_t result = ::waitpid(pid, &status, WNOHANG);
        if (result == 0) return true;
        if (result == pid) {
            forget_child_locked();
            return false;
        }
        // A transient interruption does not prove that the child exited. For
        // any other error, retain the existing liveness probe as a fallback.
        return ::kill(pid, 0) == 0;
#endif
    }

    /// The watchdog body, and the SOLE reaper once it is running.
    void reap_and_notify() {
        int code = -1;

        // The blocking wait is deliberately outside child_mu: holding it here
        // would make every alive() probe and every stop() block for the whole
        // life of the engine.
#if defined(_WIN32)
        HANDLE handle = nullptr;
        {
            std::lock_guard<std::mutex> lk(child_mu);
            handle = proc;
        }
        if (handle != nullptr) {
            WaitForSingleObject(handle, INFINITE);
            DWORD c = 0;
            if (GetExitCodeProcess(handle, &c)) code = static_cast<int>(c);
        }
#else
        pid_t target = 0;
        {
            std::lock_guard<std::mutex> lk(child_mu);
            target = pid;
        }
        if (target > 0) {
            int status = 0;
            if (::waitpid(target, &status, 0) == target) {
                code = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
            }
        }
#endif

        // BEFORE the `stopping` check, not after: a clean stop is waiting on
        // exactly this, and an early return that skipped it would hang
        // terminate_child() for its full grace period on every shutdown.
        {
            std::lock_guard<std::mutex> lk(child_mu);
            forget_child_locked();
        }

        if (stopping.load()) return; // a clean shutdown, not a crash

        state.store(ProcessState::Crashed);
        CrashCallback cb;
        {
            std::lock_guard<std::mutex> lk(mu);
            last_error = "engine exited unexpectedly with code " + std::to_string(code);
            cb = on_crash;
        }
        if (cb) cb(code, "engine exited unexpectedly");
    }

    /// Ask the child to die and wait for the WATCHDOG to reap it.
    ///
    /// Used only when a watchdog exists. It owns waitpid() for this child, so a
    /// second waitpid() here would be two threads reaping one process: one gets
    /// the status, the other gets ECHILD, and which is which is a coin flip.
    void terminate_child() {
        using namespace std::chrono_literals;
        std::unique_lock<std::mutex> lk(child_mu);
#if defined(_WIN32)
        if (proc == nullptr) return;
        TerminateProcess(proc, 1);
        child_gone.wait_for(lk, 5s, [this] { return proc == nullptr; });
#else
        if (pid <= 0) return;
        // Graceful first. Soma persists KV on shutdown, so a hard kill costs the
        // warm-reopen state that made preemption nearly free.
        ::kill(pid, SIGTERM);
        if (child_gone.wait_for(lk, 5s, [this] { return pid == 0; })) return;
        if (pid > 0) ::kill(pid, SIGKILL);
        child_gone.wait_for(lk, 5s, [this] { return pid == 0; });
#endif
        // Either way stop() joins the watchdog next, and that join is the real
        // barrier; these waits only bound how long the escalation is deferred.
    }

    /// Kill AND reap inline. Valid only while no watchdog is running — the
    /// start() failure paths, which run before one is created.
    void kill_child() {
        using namespace std::chrono_literals;
#if defined(_WIN32)
        HANDLE handle = nullptr;
        {
            std::lock_guard<std::mutex> lk(child_mu);
            handle = proc;
            if (handle == nullptr) return;
            TerminateProcess(handle, 1);
        }
        WaitForSingleObject(handle, 5000);
        std::lock_guard<std::mutex> lk(child_mu);
        forget_child_locked();
#else
        pid_t target = 0;
        {
            std::lock_guard<std::mutex> lk(child_mu);
            target = pid;
        }
        if (target <= 0) return;

        // Graceful first. Soma persists KV on shutdown, so a hard kill costs the
        // warm-reopen state that made preemption nearly free.
        ::kill(target, SIGTERM);
        // waitpid(WNOHANG) rather than kill(pid, 0): with nobody else reaping,
        // an exited child is a zombie and kill(pid, 0) still reports it alive,
        // so the old loop burned the whole five-second grace on every stop.
        int status = 0;
        bool reaped = false;
        for (int i = 0; i < 50; ++i) {
            const pid_t r = ::waitpid(target, &status, WNOHANG);
            if (r == target || (r < 0 && errno != EINTR)) {
                reaped = true;
                break;
            }
            std::this_thread::sleep_for(100ms);
        }
        if (!reaped) {
            ::kill(target, SIGKILL);
            ::waitpid(target, &status, 0);
        }
        std::lock_guard<std::mutex> lk(child_mu);
        forget_child_locked();
#endif
    }

    bool spawn(const EngineLaunchSpec& spec) {
#if defined(_WIN32)
        std::string cmd = "\"" + spec.executable + "\"";
        for (const auto& a : spec.args)
            cmd += " \"" + a + "\"";

        STARTUPINFOA si{};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        std::vector<char> mutable_cmd(cmd.begin(), cmd.end());
        mutable_cmd.push_back('\0');

        // A COMPLETE child-only block: the node's own environment with this
        // spec's overrides applied on top. The objection that killed the earlier
        // attempt — "a partial block replaces the child's whole environment,
        // strips PATH, and the child cannot find its own DLLs" — was right about
        // a PARTIAL block and wrong to conclude that the only alternative was
        // writing into the node. This block is not partial.
        //
        // Nothing is written to the node's environment, so there is nothing to
        // restore, nothing leaks into the next engine's CUDA_VISIBLE_DEVICES or
        // RAY_ADDRESS or HF_HOME, and two starts racing each other can no longer
        // cross their GPU selections before either reaches CreateProcess.
        std::vector<char> env_block = build_child_environment(spec.env);

        const BOOL ok = CreateProcessA(nullptr,
                                       mutable_cmd.data(),
                                       nullptr,
                                       nullptr,
                                       FALSE,
                                       CREATE_NO_WINDOW,
                                       env_block.data(),
                                       nullptr,
                                       &si,
                                       &pi);
        if (!ok) {
            last_error = "CreateProcess failed for " + spec.executable + " (error " +
                         std::to_string(GetLastError()) + ")";
            return false;
        }
        CloseHandle(pi.hThread);
        // Published under child_mu like every other write to the identity. No
        // reader can reach this EngineProcess yet, but a guard that holds
        // everywhere except at birth is the kind that stops holding later.
        std::lock_guard<std::mutex> lk(child_mu);
        proc = pi.hProcess;
        pid = pi.dwProcessId;
        return true;
#else
        const pid_t child = ::fork();
        if (child < 0) {
            last_error = "fork failed";
            return false;
        }
        if (child == 0) {
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
        // Published under child_mu like every other write to the identity. A
        // local until here so the child branch tests fork()'s return rather
        // than a member the reaper is entitled to clear.
        std::lock_guard<std::mutex> lk(child_mu);
        pid = child;
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
    if (im.watchdog.joinable()) {
        // The watchdog owns waitpid() for this child. stop() therefore SIGNALS
        // and waits; it does not reap. Calling kill_child() here put two threads
        // inside waitpid() on one pid and had them race to clear it.
        im.terminate_child();
        im.watchdog.join();
    } else {
        // start() failed before the watchdog existed, so there is no other
        // reaper and this call is the one.
        im.kill_child();
    }
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
    // One critical section, not a probe followed by an unguarded read: the
    // watchdog can clear the identity between the two, and a caller that just
    // saw "alive" would be handed the zero, or worse a pid the reaper had
    // already released.
    std::lock_guard<std::mutex> lk(impl_->child_mu);
#if defined(_WIN32)
    if (impl_->proc == nullptr) return 0u;
    DWORD code = 0;
    if (!GetExitCodeProcess(impl_->proc, &code) || code != STILL_ACTIVE) return 0u;
#else
    if (impl_->pid <= 0 || ::kill(impl_->pid, 0) != 0) return 0u;
#endif
    return static_cast<std::uint32_t>(impl_->pid);
}

} // namespace mm
