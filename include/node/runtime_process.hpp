#pragma once

#include "common/models.hpp"
// ProcessState now comes from engine_process.hpp, which owns it.
//
// Both headers declared `enum class ProcessState` in namespace mm — the old
// four-state one here and the five-state one there (it adds Crashed, which is
// the whole point of the watchdog). Including both was an ODR violation waiting
// for the first translation unit that needed the engine registry and a slot
// manager at once. EngineProcess REPLACES this class, so the definition follows
// the replacement rather than being duplicated during the transition.
#include "node/engine_process.hpp"

#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <memory>
#include <vector>

namespace mm {

// Manages a llama-server inference child process.
// Platform specifics are hidden behind a PIMPL.
// Pipe output is forwarded via LogCallback on background threads.
class RuntimeProcess {
public:
    explicit RuntimeProcess(std::string server_path);
    ~RuntimeProcess();

    using LogCallback = std::function<void(const std::string& line, bool is_stderr)>;
    void set_log_callback(LogCallback cb);

    // Launches `llama-server` on the given port; blocks until /health is ready
    // or the startup timeout expires. When slot_save_path is non-empty the
    // engine's KV-cache slot save/restore endpoints are enabled (for suspend).
    // Returns true on success.
    bool start_llama_server(const std::string& model_path,
                            const std::string& mmproj_path,
                            const RuntimeSettings& settings,
                            uint16_t port = 8080,
                            const std::string& slot_save_path = {});

    // Graceful SIGTERM → SIGKILL fallback.
    void stop();

    ProcessState get_state()  const;
    uint16_t     get_port()   const;
    std::string  get_url()    const;  // "http://127.0.0.1:{port}"
    std::string  last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::string           runtime_path_;
    uint16_t              port_  = 8080;
    std::atomic<ProcessState> state_{ProcessState::Stopped};
    std::string           last_error_;
    LogCallback           log_cb_;

    bool start_with_args(const std::string& runtime_name,
                         const std::string& executable_path,
                         std::vector<std::string> args,
                         uint16_t port,
                         int health_timeout_seconds);
    bool poll_health(int timeout_seconds = 60);
};

} // namespace mm
