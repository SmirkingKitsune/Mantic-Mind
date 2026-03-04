#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <httplib.h>
#include <algorithm>

namespace mm {

// ── URL helpers ───────────────────────────────────────────────────────────────
namespace {

httplib::Client make_cli(const std::string& base_url) {
    auto [h, p] = util::parse_url(base_url);
    httplib::Client cli(h, p);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(30);
    cli.set_write_timeout(10);
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

HttpResponse HttpClient::get(const std::string& path) {
    auto cli = make_cli(base_url_);
    return to_resp(cli.Get(path, make_headers(bearer_token_)));
}

HttpResponse HttpClient::post(const std::string& path, const nlohmann::json& body) {
    auto cli = make_cli(base_url_);
    return to_resp(cli.Post(path, make_headers(bearer_token_),
                            body.dump(), "application/json"));
}

HttpResponse HttpClient::put(const std::string& path, const nlohmann::json& body) {
    auto cli = make_cli(base_url_);
    return to_resp(cli.Put(path, make_headers(bearer_token_),
                           body.dump(), "application/json"));
}

HttpResponse HttpClient::del(const std::string& path) {
    auto cli = make_cli(base_url_);
    return to_resp(cli.Delete(path, make_headers(bearer_token_)));
}

bool HttpClient::stream_get(const std::string& path, SseLineCallback line_cb) {
    auto cli = make_cli(base_url_);
    cli.set_read_timeout(300);

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
    auto [h, p] = util::parse_url(base_url_);
    httplib::Client cli(h, p);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(300);
    cli.set_write_timeout(30);

    std::string body_str = body.dump();
    std::string sse_buf;
    std::string raw_body;
    constexpr size_t kRawBodyCaptureLimit = 32 * 1024;

    // cpp-httplib: ContentProviderWithoutLength + ContentReceiver overload
    auto res = cli.Post(
        path,
        make_headers(bearer_token_),
        [&](size_t /*offset*/, httplib::DataSink& sink) -> bool {
            sink.write(body_str.data(), body_str.size());
            sink.done();
            return true;
        },
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
