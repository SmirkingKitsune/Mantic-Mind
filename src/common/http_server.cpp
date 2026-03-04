#include "common/http_server.hpp"
#include <httplib.h>
#include <memory>

namespace mm {

struct HttpServer::Impl {
    httplib::Server srv;
};

HttpServer::HttpServer() : impl_(std::make_unique<Impl>()) {}
HttpServer::~HttpServer() { stop(); }

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

bool HttpServer::listen(const std::string& host, uint16_t port) {
    return impl_->srv.listen(host, port);
}
void HttpServer::stop() { impl_->srv.stop(); }
bool HttpServer::is_running() const { return impl_->srv.is_running(); }

} // namespace mm
