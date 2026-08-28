#include "common/http_server.hpp"
#include <httplib.h>
#include <memory>

namespace mm {

struct HttpServer::Impl {
    httplib::Server srv;
    /// Recorded as they are registered. httplib keeps its own handler lists but
    /// does not expose them, and the alternative — a hand-maintained second list
    /// — is exactly the thing the coverage check exists to make unnecessary.
    std::vector<std::string> routes;
};

HttpServer::HttpServer() : impl_(std::make_unique<Impl>()) {}
HttpServer::~HttpServer() { stop(); }

void HttpServer::SetPreRoutingHandler(PreRoutingHandler h) {
    // A large streamed upload arrives with `Expect: 100-continue`. cpp-httplib
    // consults its expectation handler before normal pre-routing; leaving that
    // handler at the default sends 100 immediately, so an unauthorized client
    // transmits the entire model before the auth middleware can return 401.
    // Share one handler object between both phases. Allowed requests receive
    // 100 and are checked again during ordinary routing; rejected requests keep
    // the status/body populated by the middleware and never send their body.
    auto handler = std::make_shared<PreRoutingHandler>(std::move(h));
    impl_->srv.set_expect_100_continue_handler(
        [handler](const httplib::Request& req, httplib::Response& res) {
            if ((*handler)(req, res)) return static_cast<int>(httplib::StatusCode::Continue_100);
            return res.status > 0
                       ? res.status
                       : static_cast<int>(httplib::StatusCode::ExpectationFailed_417);
        });
    impl_->srv.set_pre_routing_handler(
        [handler](const httplib::Request& req, httplib::Response& res) {
            return (*handler)(req, res)
                ? httplib::Server::HandlerResponse::Unhandled
                : httplib::Server::HandlerResponse::Handled;
        });
}

void HttpServer::Get(const std::string& pattern, Handler h) {
    impl_->routes.push_back("GET " + pattern);
    impl_->srv.Get(pattern, std::move(h));
}
void HttpServer::Post(const std::string& pattern, Handler h) {
    impl_->routes.push_back("POST " + pattern);
    impl_->srv.Post(pattern, std::move(h));
}
void HttpServer::Put(const std::string& pattern, Handler h) {
    impl_->routes.push_back("PUT " + pattern);
    impl_->srv.Put(pattern, std::move(h));
}
void HttpServer::Delete(const std::string& pattern, Handler h) {
    impl_->routes.push_back("DELETE " + pattern);
    impl_->srv.Delete(pattern, std::move(h));
}

std::vector<std::string> HttpServer::registered_routes() const { return impl_->routes; }

void HttpServer::PostUpload(const std::string& pattern, UploadHandler h) {
    impl_->routes.push_back("POST " + pattern);
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
