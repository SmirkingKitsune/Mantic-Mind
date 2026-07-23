#pragma once

#include "aio/aio_config.hpp"

#include <memory>
#include <string>

namespace mm {

// Lifecycle owner for the combined control + embedded-node process. The
// implementation keeps the embedded node private and exposes only the control
// and optional OpenAI-compatible listeners.
class AioHost final {
public:
    AioHost(AioConfig config, bool allow_network = false);
    ~AioHost();

    AioHost(const AioHost&) = delete;
    AioHost& operator=(const AioHost&) = delete;

    bool start(bool tui_mode, std::string* error = nullptr);
    int run_tui();
    int run_cli(bool json_output = false);
    void request_quit();
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mm
