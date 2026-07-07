#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <httplib.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

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

httplib::Headers make_headers(const std::string& bearer) {
    if (bearer.empty()) return {};
    return { {"Authorization", "Bearer " + bearer} };
}

HttpResponse to_resp(const httplib::Result& res) {
    if (!res) return { 0, "" };
    return { res->status, res->body };
}

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

HttpResponse HttpClient::get(const std::string& path) {
    auto cli = make_cli(base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_);
    return to_resp(cli.Get(path, make_headers(bearer_token_)));
}

HttpResponse HttpClient::post(const std::string& path, const nlohmann::json& body) {
    auto cli = make_cli(base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_);
    return to_resp(cli.Post(path, make_headers(bearer_token_),
                            body.dump(), "application/json"));
}

HttpResponse HttpClient::put(const std::string& path, const nlohmann::json& body) {
    auto cli = make_cli(base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_);
    return to_resp(cli.Put(path, make_headers(bearer_token_),
                           body.dump(), "application/json"));
}

HttpResponse HttpClient::del(const std::string& path) {
    auto cli = make_cli(base_url_, connect_timeout_s_, read_timeout_s_, write_timeout_s_);
    return to_resp(cli.Delete(path, make_headers(bearer_token_)));
}

HttpResponse HttpClient::post_file(
    const std::string& path,
    const std::string& file_path,
    const std::vector<std::pair<std::string, std::string>>& extra_headers,
    const std::string& content_type) {
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
    auto provider = [fp](size_t offset, size_t length, httplib::DataSink& sink) -> bool {
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

    return to_resp(cli.Post(path, headers, static_cast<size_t>(file_bytes),
                            provider, content_type));
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
                              std::string* out_body) {
    httplib::Client cli(client_base_url(base_url_));
    cli.set_connection_timeout(connect_timeout_s_);
    cli.set_read_timeout(std::max(read_timeout_s_, stream_read_timeout_floor_s()));
    cli.set_write_timeout(std::max(write_timeout_s_, 30));

    std::string body_str = body.dump();
    std::string sse_buf;
    std::string raw_body;
    constexpr size_t kRawBodyCaptureLimit = 32 * 1024;

    auto res = cli.Post(
        path,
        make_headers(bearer_token_),
        body_str,
        "application/json",
        [&](const char* data, size_t len) -> bool {
            if (raw_body.size() < kRawBodyCaptureLimit) {
                size_t keep = std::min(len, kRawBodyCaptureLimit - raw_body.size());
                raw_body.append(data, keep);
            }
            sse_buf.append(data, len);
            for (auto& payload : util::drain_sse_lines(sse_buf))
                if (!line_cb(payload)) return false;
            return true;
        },
        nullptr // no upload-progress callback
    );

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
