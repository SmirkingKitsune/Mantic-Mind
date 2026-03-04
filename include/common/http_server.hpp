#pragma once

#include <string>
#include <functional>
#include <memory>
#include <cstdint>

// Thin wrapper around cpp-httplib Server.
namespace httplib { class Server; struct Request; struct Response; }

namespace mm {

class HttpServer {
public:
    HttpServer();
    ~HttpServer();

    using Handler = std::function<void(const httplib::Request&, httplib::Response&)>;

    void Get(const std::string& pattern, Handler h);
    void Post(const std::string& pattern, Handler h);
    void Put(const std::string& pattern, Handler h);
    void Delete(const std::string& pattern, Handler h);

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
