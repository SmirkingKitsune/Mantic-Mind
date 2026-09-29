#include "common/http_server.hpp"
#include <httplib.h>
#include <cstdlib>
#include <memory>

namespace mm {

namespace {

/// The largest request body a refusal waits to read; see SetPreRoutingHandler.
constexpr std::size_t kDeferredRefusalBodyLimit = 1024 * 1024;

/// A refusal decided before routing, to be delivered after the body is read.
/// Per thread: httplib runs a request's pre-routing, body read and pre-request
/// steps on one worker, and the owning server is recorded so two servers in
/// one process cannot deliver each other's refusals.
thread_local const void* t_refusal_deferred_by = nullptr;

/// A body the client has already sent — no `Expect: 100-continue` to hold it
/// back — of a known length small enough to read.
bool body_in_flight(const httplib::Request& req) {
    if (req.has_header("Expect")) return false;
    if (!req.has_header("Content-Length")) return false;
    const auto length = std::strtoull(req.get_header_value("Content-Length").c_str(), nullptr, 10);
    return length > 0 && length <= kDeferredRefusalBodyLimit;
}

} // namespace

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
    // A refusal must not be sent while the request's body is still unread.
    // httplib's client sends `Connection: close`, and closing a socket with
    // unread bytes makes the OS reset the connection — a reset that can reach
    // the client before the refusal does, so a wrong or under-scoped token saw
    // a dropped connection instead of its 401/403. A small body already on its
    // way is therefore read first: the refusal is kept in `res`, routing goes on
    // to read the body, and the pre-request step delivers the refusal before
    // any handler runs. (An unmatched route gets the refusal too, since `res`
    // already carries it.) A large body, or one of unknown length, is refused
    // unread as before, rather than read in full for a client that may not be
    // allowed to send it.
    const void* owner = impl_.get();
    impl_->srv.set_pre_routing_handler(
        [handler, owner](const httplib::Request& req, httplib::Response& res) {
            t_refusal_deferred_by = nullptr;
            if ((*handler)(req, res)) return httplib::Server::HandlerResponse::Unhandled;
            if (!body_in_flight(req)) return httplib::Server::HandlerResponse::Handled;
            t_refusal_deferred_by = owner;
            return httplib::Server::HandlerResponse::Unhandled;
        });
    impl_->srv.set_pre_request_handler(
        [owner](const httplib::Request&, httplib::Response&) {
            if (t_refusal_deferred_by != owner) return httplib::Server::HandlerResponse::Unhandled;
            t_refusal_deferred_by = nullptr;
            return httplib::Server::HandlerResponse::Handled; // `res` holds the refusal
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
