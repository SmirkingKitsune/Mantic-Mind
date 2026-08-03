#pragma once

// Mantic-Mind — generic engine subprocess supervision.
//
// REPLACES RuntimeProcess.
//
// The old class was already 90% of the way here: start_with_args(runtime_name,
// exe, argv, port, timeout) is engine-neutral, and start_llama_server() is a
// ~15-line adapter over it. The rebuild PROMOTES that function rather than
// wrapping it a third time, and adds the two things it lacked:
//
//   * an env block (CreateProcessA and execvp both inherited the parent's
//     verbatim, so a subprocess could only be configured through argv)
//   * crash supervision (nothing polled the child after it reached Ready, so a
//     dead engine stayed SlotState::Ready until an inference request failed at
//     the HTTP layer)
//
// Readiness needs no invention. RuntimeProcess::poll_health already polls
// GET /health with early abort on child exit; there is no log sentinel anywhere
// in this codebase, so the Windows sentinel fragility is not a risk being
// carried forward. StdoutJsonLine is declared because it costs nothing, not
// because HttpHealth is suspect.

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mm {

enum class ProcessState { Stopped, Starting, Ready, Error, Crashed };

/// How to decide the child is serving.
struct ReadinessProbe {
    enum class Kind : std::uint8_t {
        HttpHealth,     ///< GET <path>; 200 with empty/ok body means ready
        StdoutJsonLine, ///< a JSON line on stdout carrying <key>
    };

    Kind kind = Kind::HttpHealth;
    std::string http_path = "/health";
    std::string stdout_key;
    int timeout_seconds = 600; ///< MM_LOAD_TIMEOUT_S overrides
};

/// Everything needed to launch one engine. Built by an EngineDescriptor; this
/// class never constructs one itself and therefore never learns which engine it
/// is running.
struct EngineLaunchSpec {
    std::string runtime_name;
    std::string executable;
    std::vector<std::string> args;
    std::vector<std::pair<std::string, std::string>> env;
    std::uint16_t port = 0;
    ReadinessProbe readiness{};
};

/// Fired when the child exits without stop() having been called.
using CrashCallback = std::function<void(int exit_code, const std::string& detail)>;

class EngineProcess {
public:
    using LogCallback = std::function<void(const std::string& line, bool is_stderr)>;

    EngineProcess();
    EngineProcess(const EngineProcess&) = delete;
    EngineProcess& operator=(const EngineProcess&) = delete;
    ~EngineProcess();

    void set_log_callback(LogCallback cb);

    /// Fires from the watchdog thread on unexpected child exit.
    ///
    /// New. Without it a crashed engine is only discovered when a request fails,
    /// and the slot advertises Ready the whole time. Streaming makes crashes
    /// likelier (I/O pressure, OOM under a mis-sized cache cap), so it is fixed
    /// as part of the rebuild rather than left as a pre-existing gap.
    void set_crash_callback(CrashCallback cb);

    /// Spawns, then blocks until the readiness probe succeeds, the child exits,
    /// or the timeout expires. Calls stop() first if not Stopped.
    bool start(const EngineLaunchSpec& spec);

    /// SIGTERM then SIGKILL on POSIX; TerminateProcess on Windows.
    ///
    /// The old header claimed "Graceful SIGTERM → SIGKILL fallback" while the
    /// Windows path was an unconditional hard kill. Soma persists KV on
    /// shutdown, so the Windows path gains a real graceful phase: a request to
    /// the engine's own shutdown endpoint, then the hard kill as fallback.
    void stop();

    ProcessState state() const;
    std::uint16_t port() const;
    std::string url() const; ///< "http://127.0.0.1:{port}"
    std::string last_error() const;
    bool alive() const;

    /// OS process id, or 0 when not running.
    ///
    /// Exposed because "which process" is otherwise unanswerable from outside:
    /// the supervisor logs it, and the conformance harness needs it to kill
    /// exactly this child. Matching on the image name instead would take down
    /// every engine on the machine, including ones a developer is using.
    std::uint32_t pid() const;

    /// The descriptor id this process was launched from, for labelling.
    const std::string& runtime_name() const { return runtime_name_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // State lives in Impl, not here. It is read from the watchdog thread while
    // start()/stop() write it, so it needs the atomics and the mutex that Impl
    // already owns — and a second copy out here would be a second answer to
    // "is this engine alive", which is the exact question the watchdog exists
    // to settle.
    std::string runtime_name_;
};

} // namespace mm
