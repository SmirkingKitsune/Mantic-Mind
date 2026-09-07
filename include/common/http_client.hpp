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

    HttpResponse get (const std::string& path);
    HttpResponse post(const std::string& path, const nlohmann::json& body);
    HttpResponse put (const std::string& path, const nlohmann::json& body);
    HttpResponse del (const std::string& path);

    // Stream a file as the raw request body without buffering it in memory —
    // for large model transfers. extra_headers carry out-of-band metadata
    // (destination path, model id, etc.). Blocks until the upload completes.
    HttpResponse post_file(
        const std::string& path,
        const std::string& file_path,
        const std::vector<std::pair<std::string, std::string>>& extra_headers = {},
        const std::string& content_type = "application/octet-stream");

    // SSE streaming GET/POST.
    //
    // THE CALLBACK RECEIVES THE PAYLOAD, NOT THE WIRE LINE. `util::drain_sse_lines`
    // has already stripped the `data: ` prefix and dropped everything that is not
    // a data frame — comments, keepalives, blank separators — so the argument is
    // the JSON itself and is ready to parse.
    //
    // This comment used to say "each raw `data: ...` line", and that sentence
    // cost four call sites: the admission and reprofile starters in both the TUI
    // and the CLI all searched for a `data:` prefix that is never there, found
    // none, and returned "started but reported no operation id" for an admission
    // that had in fact started correctly (roadmap D68). A false claim in a header
    // is copied by everyone who believes it, which is what a header is for.
    //
    // Return false to STOP reading. httplib treats that as a cancelled transfer,
    // so `stream_post` then returns false with status 0 — check what you captured
    // BEFORE you check the return value.
    using SseLineCallback = std::function<bool(const std::string& payload)>;
    bool stream_get (const std::string& path,
                     SseLineCallback line_cb);
    // SSE streaming POST (e.g. for /api/node/infer).
    bool stream_post(const std::string& path,
                     const nlohmann::json& body,
                     SseLineCallback line_cb,
                     int* out_status = nullptr,
                     std::string* out_body = nullptr);

    /// A line callback that captures the first non-empty string `field` from an
    /// SSE payload and then stops the stream.
    ///
    /// One implementation because there were four, all identical, all wrong the
    /// same way. `out` is captured BY REFERENCE and must outlive the stream
    /// call — every caller passes a local, which is the only shape this is for.
    static SseLineCallback capture_first_field(std::string field, std::string& out);

private:
    std::string base_url_;
    std::string bearer_token_;
    int connect_timeout_s_ = 10;
    int read_timeout_s_ = 30;
    int write_timeout_s_ = 10;
};

} // namespace mm
