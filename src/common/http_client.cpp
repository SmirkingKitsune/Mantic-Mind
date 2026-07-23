#include "common/http_client.hpp"
#include "common/cancellable_http.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <httplib.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

namespace mm {

// ── URL helpers ───────────────────────────────────────────────────────────────
namespace {

// Read-timeout floor for SSE streams, which usually carry LLM inference.
// Slow hardware can spend a long time in prompt processing before the first
// byte arrives, so the floor is large and env-overridable.
int stream_read_timeout_floor_s() {
    constexpr int kDefaultSeconds = 3600;
    if (const char* env = std::getenv("MM_INFER_READ_TIMEOUT_S")) {
        try {
            int parsed = std::stoi(env);
            if (parsed > 0) return parsed;
        } catch (...) {
        }
    }
    return kDefaultSeconds;
}

std::string client_base_url(const std::string& base_url) {
    std::string url = util::trim(base_url);
    if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
        return url;
    }
    return "http://" + url;
}

httplib::Client make_cli(const std::string& base_url,
                         int connect_timeout_s,
                         int read_timeout_s,
                         int write_timeout_s) {
    httplib::Client cli(client_base_url(base_url));
    cli.set_connection_timeout(connect_timeout_s);
    cli.set_read_timeout(read_timeout_s);
    cli.set_write_timeout(write_timeout_s);
    return cli;
}

std::unique_ptr<CancellableHttpClient> make_cancellable_cli(
    const std::string& base_url,
    int connect_timeout_s,
    int read_timeout_s,
    int write_timeout_s,
    HttpClient::CancelCheck cancel_requested) {
    auto client = std::make_unique<CancellableHttpClient>(
        client_base_url(base_url), std::move(cancel_requested));
    client->set_connection_timeout(connect_timeout_s);
    client->set_read_timeout(read_timeout_s);
    client->set_write_timeout(write_timeout_s);
    return client;
}

httplib::Headers make_headers(const std::string& bearer) {
    if (bearer.empty()) return {};
    return { {"Authorization", "Bearer " + bearer} };
}

HttpResponse to_resp(const httplib::Result& res) {
    if (!res) return { 0, "" };
    return { res->status, res->body };
}

class ClientCancelWatcher {
public:
    ClientCancelWatcher(httplib::Client& client,
                        std::function<bool()> cancel_requested) {
        if (!cancel_requested) return;
        client.set_socket_options([this](socket_t socket) {
            httplib::default_socket_options(socket);
            socket_ = socket;
        });
#ifdef _WIN32
        // cpp-httplib's Client::stop() shuts down the socket, but a Windows
        // synchronous recv waiting for response headers is not guaranteed to
        // wake from shutdown alone. Keep a real handle to the request thread
        // so cancellation can also interrupt that pending synchronous I/O.
        (void)DuplicateHandle(
            GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
            &request_thread_, 0, FALSE, DUPLICATE_SAME_ACCESS);
#endif
        thread_ = std::thread([this, &client,
                               check = std::move(cancel_requested)] {
            while (!finished_) {
                bool canceled = false;
                try { canceled = check(); } catch (...) { canceled = true; }
                if (canceled) {
                    const auto socket = socket_.load();
                    if (socket != static_cast<socket_t>(-1)) {
#ifdef _WIN32
                        (void)::shutdown(socket, SD_BOTH);
#else
                        (void)::shutdown(socket, SHUT_RDWR);
#endif
                    }
#ifdef _WIN32
                    if (request_thread_) {
                        (void)CancelSynchronousIo(request_thread_);
                    }
#endif
                    client.stop();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }

    ~ClientCancelWatcher() { finish(); }

    void finish() {
        finished_ = true;
        if (thread_.joinable()) thread_.join();
#ifdef _WIN32
        if (request_thread_) {
            CloseHandle(request_thread_);
            request_thread_ = nullptr;
        }
#endif
    }

    ClientCancelWatcher(const ClientCancelWatcher&) = delete;
    ClientCancelWatcher& operator=(const ClientCancelWatcher&) = delete;

private:
    std::atomic<bool> finished_{false};
    std::atomic<socket_t> socket_{static_cast<socket_t>(-1)};
    std::thread thread_;
#ifdef _WIN32
    HANDLE request_thread_ = nullptr;
#endif
};

} // namespace

// ── HttpClient ────────────────────────────────────────────────────────────────
HttpClient::HttpClient(std::string base_url) : base_url_(std::move(base_url)) {}

void HttpClient::set_bearer_token(const std::string& token) {
    bearer_token_ = token;
}

void HttpClient::set_timeouts(int connect_s, int read_s, int write_s) {
    connect_timeout_s_ = connect_s > 0 ? connect_s : 10;
    read_timeout_s_ = read_s > 0 ? read_s : 30;
    write_timeout_s_ = write_s > 0 ? write_s : 10;
}

HttpResponse HttpClient::get(const std::string& path,
                             CancelCheck cancel_requested) {
    if (cancel_requested && cancel_requested()) return {0, {}};
    if (cancel_requested && is_plain_http_url(base_url_)) {
        auto cli = make_cancellable_cli(
            base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_,
            std::move(cancel_requested));
        return to_resp(cli->Get(path, make_headers(bearer_token_)));
    }
    auto cli = make_cli(base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_);
    ClientCancelWatcher cancel_watcher(cli, cancel_requested);
    const auto response = cli.Get(path, make_headers(bearer_token_));
    cancel_watcher.finish();
    return to_resp(response);
}

HttpResponse HttpClient::post(const std::string& path,
                              const nlohmann::json& body,
                              CancelCheck cancel_requested) {
    if (cancel_requested && cancel_requested()) return {0, {}};
    if (cancel_requested && is_plain_http_url(base_url_)) {
        auto cli = make_cancellable_cli(
            base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_,
            std::move(cancel_requested));
        return to_resp(cli->Post(path, make_headers(bearer_token_), body.dump(),
                                 "application/json"));
    }
    auto cli = make_cli(base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_);
    ClientCancelWatcher cancel_watcher(cli, cancel_requested);
    const auto response = cli.Post(path, make_headers(bearer_token_),
                                   body.dump(), "application/json");
    cancel_watcher.finish();
    return to_resp(response);
}

HttpResponse HttpClient::put(const std::string& path,
                             const nlohmann::json& body,
                             CancelCheck cancel_requested) {
    if (cancel_requested && cancel_requested()) return {0, {}};
    if (cancel_requested && is_plain_http_url(base_url_)) {
        auto cli = make_cancellable_cli(
            base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_,
            std::move(cancel_requested));
        return to_resp(cli->Put(path, make_headers(bearer_token_), body.dump(),
                                "application/json"));
    }
    auto cli = make_cli(base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_);
    ClientCancelWatcher cancel_watcher(cli, cancel_requested);
    const auto response = cli.Put(path, make_headers(bearer_token_),
                                  body.dump(), "application/json");
    cancel_watcher.finish();
    return to_resp(response);
}

HttpResponse HttpClient::del(const std::string& path,
                             CancelCheck cancel_requested) {
    if (cancel_requested && cancel_requested()) return {0, {}};
    if (cancel_requested && is_plain_http_url(base_url_)) {
        auto cli = make_cancellable_cli(
            base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_,
            std::move(cancel_requested));
        return to_resp(cli->Delete(path, make_headers(bearer_token_)));
    }
    auto cli = make_cli(base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_);
    ClientCancelWatcher cancel_watcher(cli, cancel_requested);
    const auto response = cli.Delete(path, make_headers(bearer_token_));
    cancel_watcher.finish();
    return to_resp(response);
}

HttpResponse HttpClient::post_file(
    const std::string& path,
    const std::string& file_path,
    const std::vector<std::pair<std::string, std::string>>& extra_headers,
    const std::string& content_type,
    CancelCheck cancel_requested) {
    if (cancel_requested && cancel_requested()) return {0, {}};
    std::error_code ec;
    const auto file_bytes = std::filesystem::file_size(file_path, ec);
    if (ec) {
        MM_WARN("HttpClient::post_file cannot stat {}: {}", file_path, ec.message());
        return { 0, "" };
    }
    auto fp = std::make_shared<std::ifstream>(file_path, std::ios::binary);
    if (!fp->is_open()) {
        MM_WARN("HttpClient::post_file cannot open {}", file_path);
        return { 0, "" };
    }

    httplib::Headers headers = make_headers(bearer_token_);
    for (const auto& [k, v] : extra_headers) headers.emplace(k, v);

    // Large transfers can take a while; lift the write timeout to the same
    // generous floor used for streaming reads.
    auto cli = make_cli(base_url_, connect_timeout_s_, read_timeout_s_,
                        std::max(write_timeout_s_, stream_read_timeout_floor_s()));

    // Known-length content provider: httplib drives it with the absolute body
    // offset and the remaining length; we seek + hand back one bounded chunk
    // per call so the file streams from disk to socket without buffering.
    auto provider = [fp, cancel_requested](
                        size_t offset, size_t length,
                        httplib::DataSink& sink) -> bool {
        if (cancel_requested) {
            try {
                if (cancel_requested()) return false;
            } catch (...) {
                return false;
            }
        }
        constexpr size_t kChunk = 1u << 20; // 1 MiB
        fp->clear();
        fp->seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        const size_t want = std::min(length, kChunk);
        std::vector<char> buf(want);
        fp->read(buf.data(), static_cast<std::streamsize>(want));
        const std::streamsize got = fp->gcount();
        if (got <= 0) return false;
        return sink.write(buf.data(), static_cast<size_t>(got));
    };

    if (cancel_requested && is_plain_http_url(base_url_)) {
        auto cancellable_cli = make_cancellable_cli(
            base_url_, connect_timeout_s_, read_timeout_s_,
            std::max(write_timeout_s_, stream_read_timeout_floor_s()),
            std::move(cancel_requested));
        return to_resp(cancellable_cli->Post(
            path, headers, static_cast<size_t>(file_bytes), provider,
            content_type));
    }

    ClientCancelWatcher cancel_watcher(cli, cancel_requested);
    const auto response = cli.Post(path, headers, static_cast<size_t>(file_bytes),
                                   provider, content_type);
    cancel_watcher.finish();
    return to_resp(response);
}

bool HttpClient::stream_get(const std::string& path, SseLineCallback line_cb) {
    auto cli = make_cli(base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_);
    cli.set_read_timeout(std::max(read_timeout_s_, stream_read_timeout_floor_s()));

    std::string buf;
    auto res = cli.Get(path, make_headers(bearer_token_),
        [&](const char* data, size_t len) -> bool {
            buf.append(data, len);
            for (auto& payload : util::drain_sse_lines(buf))
                if (!line_cb(payload)) return false;
            return true;
        });

    if (!res) {
        MM_WARN("HttpClient::stream_get failed: {}", base_url_ + path);
        return false;
    }
    return true;
}

bool HttpClient::stream_post(const std::string& path,
                              const nlohmann::json& body,
                              SseLineCallback line_cb,
                              int* out_status,
                              std::string* out_body,
                              std::function<bool()> cancel_requested) {
    const auto canceled = [&cancel_requested] {
        if (!cancel_requested) return false;
        try { return cancel_requested(); } catch (...) { return true; }
    };
    if (canceled()) {
        if (out_status) *out_status = 0;
        if (out_body) out_body->clear();
        return false;
    }

    std::string body_str = body.dump();
    std::string sse_buf;
    std::string raw_body;
    constexpr size_t kRawBodyCaptureLimit = 32 * 1024;

    auto receiver = [&](const char* data, size_t len) -> bool {
            if (canceled()) return false;
            if (raw_body.size() < kRawBodyCaptureLimit) {
                size_t keep = std::min(len, kRawBodyCaptureLimit - raw_body.size());
                raw_body.append(data, keep);
            }
            sse_buf.append(data, len);
            for (auto& payload : util::drain_sse_lines(sse_buf))
                if (!line_cb(payload)) return false;
            return true;
        };

    httplib::Result res;
    if (cancel_requested && is_plain_http_url(base_url_)) {
        auto cli = make_cancellable_cli(
            base_url_, connect_timeout_s_,
            std::max(read_timeout_s_, stream_read_timeout_floor_s()),
            std::max(write_timeout_s_, 30), cancel_requested);
        res = cli->Post(path, make_headers(bearer_token_), body_str,
                        "application/json", receiver, nullptr);
    } else {
        httplib::Client cli(client_base_url(base_url_));
        cli.set_connection_timeout(connect_timeout_s_);
        cli.set_read_timeout(
            std::max(read_timeout_s_, stream_read_timeout_floor_s()));
        cli.set_write_timeout(std::max(write_timeout_s_, 30));
        ClientCancelWatcher cancel_watcher(cli, cancel_requested);
        res = cli.Post(path, make_headers(bearer_token_), body_str,
                       "application/json", receiver, nullptr);
        cancel_watcher.finish();
    }

    if (!res) {
        if (out_status) *out_status = 0;
        if (out_body) out_body->clear();
        MM_WARN("HttpClient::stream_post failed: {}", base_url_ + path);
        return false;
    }
    if (out_status) *out_status = res->status;
    if (out_body) {
        if (!res->body.empty()) {
            *out_body = res->body;
        } else {
            *out_body = raw_body;
        }
    }
    if (res->status < 200 || res->status >= 300) {
        std::string body_preview = !res->body.empty() ? res->body : raw_body;
        if (body_preview.size() > 300) body_preview = body_preview.substr(0, 300);
        MM_WARN("HttpClient::stream_post non-2xx {} for {}{}: {}",
                res->status, base_url_, path, body_preview);
        return false;
    }
    return true;
}

} // namespace mm
