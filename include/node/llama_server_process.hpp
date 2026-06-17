#pragma once

#include "common/models.hpp"
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <memory>
#include <vector>

namespace mm {

enum class ProcessState { Stopped, Starting, Ready, Error };

// Manages an inference engine child process (`vllm serve`).
// Platform specifics are hidden behind a PIMPL.
// Pipe output is forwarded via LogCallback on background threads.
class LlamaServerProcess {
public:
    explicit LlamaServerProcess(std::string server_path);
    ~LlamaServerProcess();

    using LogCallback = std::function<void(const std::string& line, bool is_stderr)>;
    void set_log_callback(LogCallback cb);

    // Launches `vllm serve` on the given port; blocks until /health is ready
    // or the startup timeout expires. Returns true on success.
    bool start_vllm(const std::string& model_ref,
                    const VllmSettings& settings,
                    uint16_t port = 8080);

    // Graceful SIGTERM → SIGKILL fallback.
    void stop();

    ProcessState get_state()  const;
    uint16_t     get_port()   const;
    std::string  get_url()    const;  // "http://127.0.0.1:{port}"
    std::string  last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::string           llama_server_path_;
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
