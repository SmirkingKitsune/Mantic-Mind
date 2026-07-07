#include "common/http_server.hpp"
#include <httplib.h>
#include <memory>

namespace mm {

struct HttpServer::Impl {
    httplib::Server srv;
};

HttpServer::HttpServer() : impl_(std::make_unique<Impl>()) {}
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
