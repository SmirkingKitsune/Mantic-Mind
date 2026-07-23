#pragma once

#include <httplib.h>

#include <functional>
#include <string>

namespace mm {

// Plain-HTTP cpp-httplib client whose socket waits periodically return to the
// request thread to observe cancellation. This avoids cross-thread socket
// closure (and its descriptor-reuse race), while still allowing long inference
// read timeouts when no cancellation is requested.
class CancellableHttpClient final : public httplib::ClientImpl {
public:
    using CancelCheck = std::function<bool()>;

    CancellableHttpClient(const std::string& base_url,
                          CancelCheck cancel_requested);

    bool cancellation_requested() const noexcept;

protected:
    bool process_socket(
        const Socket& socket,
        std::chrono::time_point<std::chrono::steady_clock> start_time,
        std::function<bool(httplib::Stream& stream)> callback) override;

private:
    CancelCheck cancel_requested_;
};

// CancellableHttpClient intentionally handles only unencrypted HTTP. TLS is
// still handled by cpp-httplib's SSLClient and is outside the AIO transport
// change in this release.
bool is_plain_http_url(const std::string& base_url);

} // namespace mm
