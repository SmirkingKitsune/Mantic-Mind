#pragma once

#include <string>
#include <functional>
#include <optional>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace mm {

struct HttpResponse {
    int         status  = 0;
    std::string body;
    bool        ok() const { return status >= 200 && status < 300; }
};

// Simple synchronous HTTP client wrapping cpp-httplib.
// For SSE streaming, use stream_get().
class HttpClient {
public:
    // base_url example: "http://localhost:9090"
    explicit HttpClient(std::string base_url);

    // Optional bearer token applied to all requests.
    void set_bearer_token(const std::string& token);
    void set_timeouts(int connect_s, int read_s, int write_s);

    using CancelCheck = std::function<bool()>;
    HttpResponse get (const std::string& path,
                      CancelCheck cancel_requested = {});
    HttpResponse post(const std::string& path,
                      const nlohmann::json& body,
                      CancelCheck cancel_requested = {});
    HttpResponse put (const std::string& path,
                      const nlohmann::json& body,
                      CancelCheck cancel_requested = {});
    HttpResponse del (const std::string& path,
                      CancelCheck cancel_requested = {});

    // Stream a file as the raw request body without buffering it in memory —
    // for large model transfers. extra_headers carry out-of-band metadata
    // (destination path, model id, etc.). Blocks until the upload completes.
    HttpResponse post_file(
        const std::string& path,
        const std::string& file_path,
        const std::vector<std::pair<std::string, std::string>>& extra_headers = {},
        const std::string& content_type = "application/octet-stream",
        CancelCheck cancel_requested = {});

    // SSE streaming GET.  line_cb is called for each raw "data: ..." line.
    // Returns false if the connection could not be established.
    using SseLineCallback = std::function<bool(const std::string& line)>;
    bool stream_get (const std::string& path,
                     SseLineCallback line_cb);
    // SSE streaming POST (e.g. for /api/node/infer).
    bool stream_post(const std::string& path,
                     const nlohmann::json& body,
                     SseLineCallback line_cb,
                     int* out_status = nullptr,
                     std::string* out_body = nullptr,
                     CancelCheck cancel_requested = {});

private:
    std::string base_url_;
    std::string bearer_token_;
    int connect_timeout_s_ = 10;
    int read_timeout_s_ = 30;
    int write_timeout_s_ = 10;
};

} // namespace mm
