#pragma once

#include <string>
#include <functional>
#include <optional>
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

    HttpResponse get (const std::string& path);
    HttpResponse post(const std::string& path, const nlohmann::json& body);
    HttpResponse put (const std::string& path, const nlohmann::json& body);
    HttpResponse del (const std::string& path);

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
                     std::string* out_body = nullptr);

private:
    std::string base_url_;
    std::string bearer_token_;
};

} // namespace mm
