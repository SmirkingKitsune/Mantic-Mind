#pragma once

#include <string>
#include <functional>
#include <memory>
#include <cstddef>
#include <cstdint>

// Thin wrapper around cpp-httplib Server.
namespace httplib { class Server; struct Request; struct Response; }

namespace mm {

class HttpServer {
public:
    HttpServer();
    ~HttpServer();

    using Handler = std::function<void(const httplib::Request&, httplib::Response&)>;
    // Return true to continue normal route handling, false when the hook has
    // fully handled the request.
    using PreRoutingHandler = std::function<bool(const httplib::Request&, httplib::Response&)>;

    // Streaming upload route. The request body is delivered incrementally so a
    // multi-GB upload never buffers in memory: the handler calls
    // `pump(sink)`, and `sink(data, len)` is invoked for each chunk as it
    // arrives (return false from the sink to abort). `pump` returns false if
    // the client disconnected mid-body.
    using BodySink = std::function<bool(const char* data, std::size_t len)>;
    using UploadPump = std::function<bool(const BodySink&)>;
    using UploadHandler = std::function<void(const httplib::Request& req,
                                             httplib::Response& res,
                                             const UploadPump& pump)>;

    void SetPreRoutingHandler(PreRoutingHandler h);
    void Get(const std::string& pattern, Handler h);
    void Post(const std::string& pattern, Handler h);
    void Put(const std::string& pattern, Handler h);
    void Delete(const std::string& pattern, Handler h);
    void PostUpload(const std::string& pattern, UploadHandler h);

    // Raise the maximum accepted request-body size (cpp-httplib defaults to
    // 100 MB, which would reject streamed model uploads). Applies to all routes.
    void set_payload_max_length(std::size_t length);

    // Blocks until stop() is called from another thread.
    bool listen(const std::string& host, uint16_t port);
    void stop();

    bool is_running() const;

private:
    // pimpl to avoid exposing httplib in this header
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mm
