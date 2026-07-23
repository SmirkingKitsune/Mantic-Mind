#include "common/http_server.hpp"
#include <httplib.h>
#include <memory>

namespace mm {

namespace {

// cpp-httplib enables SO_REUSEADDR on Windows (and SO_REUSEPORT where it is
// available) by default.  That is convenient for quick restarts, but it can
// also allow two application listeners to claim the same endpoint.  Mantic-
// Mind treats a port collision as a startup error, so use exclusive ownership
// on Windows and the restart-safe, non-load-sharing SO_REUSEADDR behavior on
// POSIX.
void configure_listener_socket(httplib::Server& server) {
    server.set_socket_options([](socket_t socket) {
#ifdef _WIN32
        const BOOL exclusive = TRUE;
        (void)::setsockopt(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                           reinterpret_cast<const char*>(&exclusive),
                           static_cast<int>(sizeof(exclusive)));
#else
        const int reuse_address = 1;
        (void)::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR,
                           &reuse_address, sizeof(reuse_address));
#endif
    });
}

} // namespace

struct HttpServer::Impl {
    httplib::Server srv;
};

HttpServer::HttpServer() : impl_(std::make_unique<Impl>()) {
    configure_listener_socket(impl_->srv);
}
HttpServer::~HttpServer() { stop(); }

void HttpServer::SetPreRoutingHandler(PreRoutingHandler h) {
    impl_->srv.set_pre_routing_handler(
        [handler = std::move(h)](const httplib::Request& req, httplib::Response& res) {
            return handler(req, res)
                ? httplib::Server::HandlerResponse::Unhandled
                : httplib::Server::HandlerResponse::Handled;
        });
}

void HttpServer::Get(const std::string& pattern, Handler h) {
    impl_->srv.Get(pattern, std::move(h));
}
void HttpServer::Post(const std::string& pattern, Handler h) {
    impl_->srv.Post(pattern, std::move(h));
}
void HttpServer::Put(const std::string& pattern, Handler h) {
    impl_->srv.Put(pattern, std::move(h));
}
void HttpServer::Delete(const std::string& pattern, Handler h) {
    impl_->srv.Delete(pattern, std::move(h));
}

void HttpServer::PostUpload(const std::string& pattern, UploadHandler h) {
    impl_->srv.Post(
        pattern,
        [handler = std::move(h)](const httplib::Request& req, httplib::Response& res,
                                 const httplib::ContentReader& content_reader) {
            UploadPump pump = [&content_reader](const BodySink& sink) -> bool {
                return content_reader([&sink](const char* data, size_t len) {
                    return sink(data, len);
                });
            };
            handler(req, res, pump);
        });
}

void HttpServer::set_payload_max_length(std::size_t length) {
    impl_->srv.set_payload_max_length(length);
}

bool HttpServer::listen(const std::string& host, uint16_t port) {
    return impl_->srv.listen(host, port);
}
void HttpServer::stop() { impl_->srv.stop(); }
bool HttpServer::is_running() const { return impl_->srv.is_running(); }

} // namespace mm
