#include "common/cancellable_http.hpp"

#include "common/util.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

namespace mm {
namespace {

using Clock = std::chrono::steady_clock;

class PollingSocketStream final : public httplib::Stream {
public:
    PollingSocketStream(socket_t socket,
                        time_t read_timeout_seconds,
                        time_t read_timeout_microseconds,
                        time_t write_timeout_seconds,
                        time_t write_timeout_microseconds,
                        CancellableHttpClient::CancelCheck cancel_requested,
                        Clock::time_point start_time)
        : socket_(socket)
        , read_timeout_seconds_(read_timeout_seconds)
        , read_timeout_microseconds_(read_timeout_microseconds)
        , write_timeout_seconds_(write_timeout_seconds)
        , write_timeout_microseconds_(write_timeout_microseconds)
        , cancel_requested_(std::move(cancel_requested))
        , start_time_(start_time) {}

    bool is_readable() const override {
        if (canceled()) return false;
        return httplib::detail::select_read(socket_, 0, 0) > 0;
    }

    bool wait_readable() const override {
        return wait_for_socket(true, read_timeout_seconds_,
                               read_timeout_microseconds_);
    }

    bool wait_writable() const override {
        return wait_for_socket(false, write_timeout_seconds_,
                               write_timeout_microseconds_);
    }

    ssize_t read(char* data, size_t size) override {
        if (!wait_readable()) {
            error_ = canceled() ? httplib::Error::Canceled
                                : httplib::Error::Timeout;
            return -1;
        }
#ifdef _WIN32
        size = (std::min)(
            size,
            static_cast<size_t>((std::numeric_limits<int>::max)()));
#else
        size = (std::min)(
            size,
            static_cast<size_t>((std::numeric_limits<ssize_t>::max)()));
#endif
        const auto result = httplib::detail::read_socket(
            socket_, data, size, CPPHTTPLIB_RECV_FLAGS);
        if (result == 0) {
            error_ = httplib::Error::ConnectionClosed;
        } else if (result < 0) {
            error_ = canceled() ? httplib::Error::Canceled
                                : httplib::Error::Read;
        }
        return result;
    }

    ssize_t write(const char* data, size_t size) override {
        if (!wait_writable()) {
            error_ = canceled() ? httplib::Error::Canceled
                                : httplib::Error::Timeout;
            return -1;
        }
#if defined(_WIN32) && !defined(_WIN64)
        size = (std::min)(
            size,
            static_cast<size_t>((std::numeric_limits<int>::max)()));
#endif
        const auto result = httplib::detail::send_socket(
            socket_, data, size, CPPHTTPLIB_SEND_FLAGS);
        if (result < 0) {
            error_ = canceled() ? httplib::Error::Canceled
                                : httplib::Error::Write;
        }
        return result;
    }

    void get_remote_ip_and_port(std::string& ip, int& port) const override {
        httplib::detail::get_remote_ip_and_port(socket_, ip, port);
    }

    void get_local_ip_and_port(std::string& ip, int& port) const override {
        httplib::detail::get_local_ip_and_port(socket_, ip, port);
    }

    socket_t socket() const override { return socket_; }

    time_t duration() const override {
        return static_cast<time_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - start_time_)
                .count());
    }

private:
    bool canceled() const noexcept {
        if (!cancel_requested_) return false;
        try {
            return cancel_requested_();
        } catch (...) {
            return true;
        }
    }

    bool wait_for_socket(bool read,
                         time_t timeout_seconds,
                         time_t timeout_microseconds) const {
        using namespace std::chrono;
        constexpr auto kCancelPoll = milliseconds(20);
        const auto timeout = seconds(std::max<time_t>(0, timeout_seconds)) +
                             microseconds(
                                 std::max<time_t>(0, timeout_microseconds));
        const auto deadline = Clock::now() + timeout;

        for (;;) {
            if (canceled()) return false;
            const auto now = Clock::now();
            if (now >= deadline) return false;
            const auto remaining = deadline - now;
            const auto slice = std::min<Clock::duration>(remaining, kCancelPoll);
            const auto slice_us = duration_cast<microseconds>(slice).count();
            const time_t seconds_part =
                static_cast<time_t>(slice_us / 1'000'000);
            const time_t microseconds_part =
                static_cast<time_t>(slice_us % 1'000'000);
            const auto result = read
                ? httplib::detail::select_read(
                      socket_, seconds_part, microseconds_part)
                : httplib::detail::select_write(
                      socket_, seconds_part, microseconds_part);
            if (result != 0) return result > 0;
        }
    }

    socket_t socket_ = INVALID_SOCKET;
    time_t read_timeout_seconds_ = 0;
    time_t read_timeout_microseconds_ = 0;
    time_t write_timeout_seconds_ = 0;
    time_t write_timeout_microseconds_ = 0;
    CancellableHttpClient::CancelCheck cancel_requested_;
    Clock::time_point start_time_;
};

std::string client_host(const std::string& base_url) {
    return util::parse_url(base_url).first;
}

int client_port(const std::string& base_url) {
    return util::parse_url(base_url).second;
}

} // namespace

CancellableHttpClient::CancellableHttpClient(
    const std::string& base_url,
    CancelCheck cancel_requested)
    : httplib::ClientImpl(client_host(base_url), client_port(base_url))
    , cancel_requested_(std::move(cancel_requested)) {}

bool CancellableHttpClient::cancellation_requested() const noexcept {
    if (!cancel_requested_) return false;
    try {
        return cancel_requested_();
    } catch (...) {
        return true;
    }
}

bool CancellableHttpClient::process_socket(
    const Socket& socket,
    std::chrono::time_point<std::chrono::steady_clock> start_time,
    std::function<bool(httplib::Stream& stream)> callback) {
    PollingSocketStream stream(
        socket.sock, read_timeout_sec_, read_timeout_usec_,
        write_timeout_sec_, write_timeout_usec_, cancel_requested_, start_time);
    return callback(stream);
}

bool is_plain_http_url(const std::string& base_url) {
    const std::string normalized = util::to_lower(util::trim(base_url));
    return normalized.rfind("https://", 0) != 0;
}

} // namespace mm
