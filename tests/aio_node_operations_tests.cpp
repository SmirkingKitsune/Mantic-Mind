#include "aio/local_node_operations.hpp"
#include "control/node_operations.hpp"
#include "control/node_registry.hpp"
#include "common/runtime_client.hpp"
#include "node/node_service.hpp"
#include "node/node_api_server.hpp"
#include "node/node_state.hpp"
#include "node/slot_manager.hpp"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __func__ << ':' << __LINE__                            \
                      << ": check failed: " #condition "\n";                  \
            return false;                                                       \
        }                                                                       \
    } while (false)

class TempDirectory {
public:
    explicit TempDirectory(const std::string& label) {
        const auto stamp = std::chrono::steady_clock::now()
                               .time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("mantic-aio-node-ops-" + label + "-" +
                 std::to_string(stamp));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

void write_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << text;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

// A deterministic embedded transport used by the contract tests.  The same
// observation helpers can be applied to LocalNodeOperations once its concrete
// NodeService fixture is available.  It follows the local adapter's contract:
// one typed terminal JSON event followed by one [DONE] transport sentinel;
// returning false from the callback suppresses every later callback.
class ScriptedEmbeddedOperations final : public mm::NodeOperations {
public:
    bool embedded() const override { return true; }
    std::string endpoint() const override { return {}; }
    void request_shutdown() override { ++shutdown_calls; }

    mm::NodeOperationResult health() override {
        ++health_calls;
        return mm::NodeOperationResult::success(nlohmann::json{
            {"cpu_percent", 5.0},
            {"ram_percent", 10.0},
            {"gpu_percent", 0.0},
            {"gpu_vram_used_mb", 0},
            {"gpu_vram_total_mb", 0},
            {"ram_used_mb", 1024},
            {"ram_total_mb", 16384},
            {"disk_free_mb", 2048},
            {"gpu_backend_available", false},
        });
    }

    mm::NodeOperationResult status() override {
        ++status_calls;
        return mm::NodeOperationResult::success(nlohmann::json{
            {"hostname", "aio-test"},
            {"slots", nlohmann::json::array()},
            {"max_slots", 2},
            {"slot_in_use", 0},
            {"slot_available", 2},
        });
    }

    mm::NodeOperationResult logs(int tail) override {
        return mm::NodeOperationResult::success(
            nlohmann::json{{"lines", std::vector<std::string>{
                                          "tail=" + std::to_string(tail)}}});
    }

    mm::NodeOperationResult cancel_action() override {
        ++cancel_calls;
        if (!streaming_) {
            return mm::NodeOperationResult::failure(
                409, "no cancelable action is active");
        }
        cancel_requested_ = true;
        return mm::NodeOperationResult::success(
            nlohmann::json{{"cancel_requested", true}});
    }

    std::optional<mm::PreparedNodeModel> prepare_model(
        const std::string& source_path,
        const std::string& model_id,
        bool /*pin*/,
        bool /*force*/,
        std::string* error) override {
        if (source_path.empty()) {
            if (error) *error = "model path is empty";
            return std::nullopt;
        }
        return mm::PreparedNodeModel{source_path, model_id};
    }

    mm::NodeOperationResult load_model(const nlohmann::json& request) override {
        if (request.value("model_path", std::string{}).empty()) {
            auto result = mm::NodeOperationResult::failure(404, "model missing");
            result.body["code"] = "model_not_found";
            result.raw_body = result.body.dump();
            return result;
        }
        if (request.value("simulate_capacity_error", false)) {
            auto result = mm::NodeOperationResult::failure(409, "capacity exhausted");
            result.body["code"] = "capacity_exhausted";
            result.raw_body = result.body.dump();
            return result;
        }
        return mm::NodeOperationResult::success(
            nlohmann::json{{"slot_id", "slot-0"}});
    }

    mm::NodeOperationResult unload_model(const nlohmann::json&) override {
        return mm::NodeOperationResult::success();
    }
    mm::NodeOperationResult detach_agent(const nlohmann::json&) override {
        return mm::NodeOperationResult::success();
    }
    mm::NodeOperationResult suspend_slot(const nlohmann::json&) override {
        return mm::NodeOperationResult::success();
    }
    mm::NodeOperationResult restore_slot(const nlohmann::json&) override {
        return mm::NodeOperationResult::success();
    }
    mm::NodeOperationResult llama_runtime() override {
        return mm::NodeOperationResult::success(
            nlohmann::json{{"available", false}});
    }
    mm::NodeOperationResult llama_provision(const nlohmann::json&) override {
        return mm::NodeOperationResult::failure(403, "network consent required");
    }
    mm::NodeOperationResult llama_update(const nlohmann::json&) override {
        return mm::NodeOperationResult::failure(403, "network consent required");
    }
    mm::NodeOperationResult llama_check_update(
        const nlohmann::json&) override {
        return mm::NodeOperationResult::failure(403, "network consent required");
    }
    mm::NodeOperationResult llama_switch(const nlohmann::json&) override {
        return mm::NodeOperationResult::failure(403, "network consent required");
    }
    mm::NodeOperationResult llama_diagnose() override {
        return mm::NodeOperationResult::success();
    }
    mm::NodeOperationResult llama_recover(const nlohmann::json&) override {
        return mm::NodeOperationResult::failure(403, "network consent required");
    }

    mm::NodeInferResult infer(
        const mm::NodeInferRequest& request,
        InferenceChunkCallback chunk_cb = {}) override {
        ++typed_infer_calls;
        mm::NodeInferResult result;
        result.slot_id = request.slot_id;
        result.message.role = mm::MessageRole::Assistant;

        for (const char* token : {"alpha", "beta"}) {
            if (request.cancel_requested && request.cancel_requested()) {
                result.status = mm::NodeServiceStatus::Canceled;
                result.error = "inference canceled";
                return result;
            }
            mm::InferenceChunk chunk;
            chunk.delta_content = token;
            result.message.content += token;
            if (chunk_cb && !chunk_cb(chunk)) {
                result.status = mm::NodeServiceStatus::Canceled;
                result.error = "inference canceled";
                return result;
            }
        }

        result.status = mm::NodeServiceStatus::Ok;
        result.tokens_used = 2;
        result.finish_reason = "stop";
        result.message.token_count = result.tokens_used;
        return result;
    }

    bool stream_infer(const nlohmann::json&,
                      SseLineCallback line_cb,
                      int* out_status,
                      std::string* out_body) override {
        ++stream_calls;
        streaming_ = true;
        cancel_requested_ = false;

        auto finish = [&](int status, std::string body, bool result) {
            streaming_ = false;
            if (out_status) *out_status = status;
            if (out_body) *out_body = std::move(body);
            return result;
        };

        auto emit = [&](const nlohmann::json& event) {
            return line_cb(event.dump());
        };

        for (const char* token : {"alpha", "beta"}) {
            if (cancel_requested_) break;
            if (!emit(nlohmann::json{{"type", "token"},
                                     {"content", token}})) {
                return finish(499, "stream consumer canceled", false);
            }
        }

        if (cancel_requested_) {
            if (!emit(nlohmann::json{{"type", "error"},
                                     {"code", "canceled"},
                                     {"error", "inference canceled"}})) {
                return finish(499, "inference canceled", false);
            }
            if (!line_cb("[DONE]")) {
                return finish(499, "inference canceled", false);
            }
            return finish(499, "inference canceled", false);
        }

        if (!emit(nlohmann::json{{"type", "done"},
                                 {"finish_reason", "stop"}})) {
            return finish(200, {}, false);
        }
        if (!line_cb("[DONE]")) return finish(200, {}, false);
        return finish(200, {}, true);
    }

    mm::NodeOperationResult pair_request(const nlohmann::json&) override {
        return mm::NodeOperationResult::failure(
            409, "embedded node cannot be paired");
    }
    mm::NodeOperationResult pair_complete(const nlohmann::json&) override {
        return mm::NodeOperationResult::failure(
            409, "embedded node cannot be paired");
    }

    int health_calls = 0;
    int status_calls = 0;
    int cancel_calls = 0;
    int stream_calls = 0;
    int typed_infer_calls = 0;
    int shutdown_calls = 0;

private:
    bool streaming_ = false;
    bool cancel_requested_ = false;
};

struct StreamObservation {
    std::vector<std::string> lines;
    int token_events = 0;
    int terminal_events = 0;
    int done_sentinels = 0;
    bool event_after_terminal = false;
};

void observe_line(StreamObservation& observation, const std::string& line) {
    if (line == "[DONE]") {
        ++observation.done_sentinels;
        return;
    }

    const auto event = nlohmann::json::parse(line);
    const std::string type = event.value("type", std::string{});
    const bool terminal = type == "done" || type == "error";
    if (observation.terminal_events > 0 && !terminal) {
        observation.event_after_terminal = true;
    }
    if (type == "token" || type == "delta") ++observation.token_events;
    if (terminal) ++observation.terminal_events;
}

class HttpAdapterFixture {
public:
    HttpAdapterFixture() {
        server_.new_task_queue = [] { return new httplib::ThreadPool(4); };
        register_routes();

        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) throw std::runtime_error("failed to bind HTTP contract server");

        server_thread_ = std::thread([this] { server_.listen_after_bind(); });
        for (int attempt = 0; attempt < 500 && !server_.is_running(); ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!server_.is_running()) {
            server_.stop();
            if (server_thread_.joinable()) server_thread_.join();
            throw std::runtime_error("HTTP contract server did not start");
        }

        operations_ = std::make_unique<mm::HttpNodeOperations>(
            "http://127.0.0.1:" + std::to_string(port_), api_key_);
    }

    ~HttpAdapterFixture() {
        server_.stop();
        if (server_thread_.joinable()) server_thread_.join();
    }

    HttpAdapterFixture(const HttpAdapterFixture&) = delete;
    HttpAdapterFixture& operator=(const HttpAdapterFixture&) = delete;

    mm::HttpNodeOperations& operations() { return *operations_; }
    int cancel_calls() const { return cancel_calls_.load(); }

private:
    struct StreamScript {
        std::string mode;
        int step = 0;
    };

    bool authorize(const httplib::Request& request,
                   httplib::Response& response) const {
        if (request.get_header_value("Authorization") ==
            "Bearer " + api_key_) {
            return true;
        }
        set_json(response, 401, {{"error", "unauthorized"}});
        return false;
    }

    static void set_json(httplib::Response& response,
                         int status,
                         const nlohmann::json& body) {
        response.status = status;
        response.set_content(body.dump(), "application/json");
    }

    static bool write_sse(httplib::DataSink& sink,
                          const nlohmann::json& event) {
        const std::string payload = "data: " + event.dump() + "\n\n";
        return sink.write(payload.data(), payload.size());
    }

    static bool write_done_sentinel(httplib::DataSink& sink) {
        static constexpr char payload[] = "data: [DONE]\n\n";
        return sink.write(payload, sizeof(payload) - 1);
    }

    void register_routes() {
        server_.Get("/api/node/health",
            [this](const httplib::Request& request, httplib::Response& response) {
                if (!authorize(request, response)) return;
                set_json(response, 200, {
                    {"status", "ok"},
                    {"cpu_percent", 5.0},
                    {"ram_percent", 10.0},
                });
            });

        server_.Get("/api/node/status",
            [this](const httplib::Request& request, httplib::Response& response) {
                if (!authorize(request, response)) return;
                set_json(response, 200, {
                    {"hostname", "http-adapter-test"},
                    {"slots", nlohmann::json::array()},
                    {"max_slots", 2},
                    {"slot_in_use", 0},
                    {"slot_available", 2},
                });
            });

        server_.Post("/api/node/load-model",
            [this](const httplib::Request& request, httplib::Response& response) {
                if (!authorize(request, response)) return;
                nlohmann::json body;
                try {
                    body = nlohmann::json::parse(request.body);
                } catch (...) {
                    set_json(response, 400, {{"error", "invalid JSON"}});
                    return;
                }
                if (body.value("model_path", std::string{}).empty()) {
                    set_json(response, 404, {
                        {"error", "model missing"},
                        {"code", "model_not_found"},
                    });
                } else if (body.value("simulate_capacity_error", false)) {
                    set_json(response, 409, {
                        {"error", "capacity exhausted"},
                        {"code", "capacity_exhausted"},
                    });
                } else {
                    set_json(response, 200, {{"slot_id", "slot-http-0"}});
                }
            });

        server_.Post("/api/node/actions/cancel",
            [this](const httplib::Request& request, httplib::Response& response) {
                if (!authorize(request, response)) return;
                ++cancel_calls_;
                if (!streaming_) {
                    set_json(response, 409,
                             {{"error", "no cancelable action is active"}});
                    return;
                }
                cancel_requested_ = true;
                set_json(response, 200, {{"cancel_requested", true}});
            });

        server_.Post("/api/node/infer",
            [this](const httplib::Request& request, httplib::Response& response) {
                if (!authorize(request, response)) return;

                std::string mode = "normal";
                try {
                    const auto body = nlohmann::json::parse(request.body);
                    mode = body.value(
                        "test_mode",
                        body.value("model", std::string{"normal"}));
                } catch (...) {
                    set_json(response, 400, {{"error", "invalid JSON"}});
                    return;
                }

                streaming_ = true;
                cancel_requested_ = false;
                auto script = std::make_shared<StreamScript>();
                script->mode = std::move(mode);

                response.set_chunked_content_provider(
                    "text/event-stream",
                    [this, script](std::size_t, httplib::DataSink& sink) {
                        const int step = script->step++;

                        if (step == 0 && script->mode == "stalled_cancel") {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(500));
                            const bool written = write_sse(sink, {
                                {"type", "delta"}, {"content", "late"},
                            });
                            streaming_ = false;
                            sink.done();
                            return written;
                        }

                        if (step == 0 && script->mode == "consumer_cancel") {
                            const std::string payload =
                                "data: {\"type\":\"delta\",\"content\":\"alpha\"}\n\n"
                                "data: {\"type\":\"delta\",\"content\":\"beta\"}\n\n"
                                "data: {\"type\":\"done\",\"finish_reason\":\"stop\"}\n\n"
                                "data: [DONE]\n\n";
                            const bool written =
                                sink.write(payload.data(), payload.size());
                            streaming_ = false;
                            sink.done();
                            return written;
                        }

                        if (step == 0) {
                            return write_sse(sink, {
                                {"type", "delta"}, {"content", "alpha"},
                            });
                        }

                        if (script->mode == "action_cancel") {
                            if (step == 1) {
                                // Give the client callback time to issue its
                                // independent cancellation POST. The server
                                // has a multi-thread task queue, so this wait
                                // does not block that request.
                                for (int attempt = 0;
                                     attempt < 500 && !cancel_requested_;
                                     ++attempt) {
                                    std::this_thread::sleep_for(
                                        std::chrono::milliseconds(1));
                                }
                                if (!cancel_requested_) {
                                    return write_sse(sink, {
                                        {"type", "delta"}, {"content", "beta"},
                                    });
                                }
                                return write_sse(sink, {
                                    {"type", "error"},
                                    {"code", "canceled"},
                                    {"message", "inference canceled"},
                                });
                            }
                            if (step == 2) {
                                const bool written = write_done_sentinel(sink);
                                streaming_ = false;
                                sink.done();
                                return written;
                            }
                        } else if (script->mode == "legacy_duplicate") {
                            if (step == 1) {
                                return write_sse(sink, {
                                    {"type", "error"},
                                    {"message", "legacy runtime error"},
                                });
                            }
                            if (step == 2) {
                                return write_sse(sink, {
                                    {"type", "delta"}, {"content", "late"},
                                });
                            }
                            if (step == 3) {
                                return write_sse(sink, {
                                    {"type", "done"},
                                    {"finish_reason", "stop"},
                                });
                            }
                            if (step == 4) {
                                const bool written = write_done_sentinel(sink);
                                streaming_ = false;
                                sink.done();
                                return written;
                            }
                        } else {
                            if (step == 1) {
                                return write_sse(sink, {
                                    {"type", "delta"}, {"content", "beta"},
                                });
                            }
                            if (step == 2) {
                                return write_sse(sink, {
                                    {"type", "done"},
                                    {"finish_reason", "stop"},
                                });
                            }
                            if (step == 3) {
                                const bool written = write_done_sentinel(sink);
                                streaming_ = false;
                                sink.done();
                                return written;
                            }
                        }

                        streaming_ = false;
                        sink.done();
                        return true;
                    },
                    [this](bool success) {
                        streaming_ = false;
                        (void)success;
                    });
            });
    }

    const std::string api_key_ = "http-contract-key";
    httplib::Server server_;
    std::thread server_thread_;
    int port_ = 0;
    std::unique_ptr<mm::HttpNodeOperations> operations_;
    std::atomic<bool> streaming_{false};
    std::atomic<bool> cancel_requested_{false};
    std::atomic<int> cancel_calls_{0};
};

bool test_result_and_typed_error_contract() {
    ScriptedEmbeddedOperations operations;

    const auto missing = operations.load_model(nlohmann::json::object());
    CHECK(!missing.ok());
    CHECK(missing.status == 404);
    CHECK(missing.error_message() == "model missing");
    CHECK(missing.body.value("code", std::string{}) == "model_not_found");

    const auto capacity = operations.load_model(nlohmann::json{
        {"model_path", "canonical-model.gguf"},
        {"simulate_capacity_error", true},
    });
    CHECK(!capacity.ok());
    CHECK(capacity.status == 409);
    CHECK(capacity.error_message() == "capacity exhausted");
    CHECK(capacity.body.value("code", std::string{}) == "capacity_exhausted");

    const auto pair = operations.pair_request(nlohmann::json::object());
    CHECK(pair.status == 409);
    CHECK(pair.error_message().find("cannot be paired") != std::string::npos);
    return true;
}

bool test_exactly_once_stream_terminal() {
    ScriptedEmbeddedOperations operations;
    StreamObservation observation;
    int status = 0;
    std::string body;

    const bool completed = operations.stream_infer(
        nlohmann::json::object(),
        [&](const std::string& line) {
            observation.lines.push_back(line);
            observe_line(observation, line);
            return true;
        },
        &status,
        &body);

    CHECK(completed);
    CHECK(status == 200);
    CHECK(body.empty());
    CHECK(observation.token_events == 2);
    CHECK(observation.terminal_events == 1);
    CHECK(observation.done_sentinels == 1);
    CHECK(!observation.event_after_terminal);
    CHECK(observation.lines.size() == 4);
    CHECK(observation.lines.back() == "[DONE]");
    return true;
}

bool test_action_cancellation_has_one_error_terminal() {
    ScriptedEmbeddedOperations operations;
    StreamObservation observation;
    int status = 0;
    std::string body;
    bool cancellation_accepted = false;

    const bool completed = operations.stream_infer(
        nlohmann::json::object(),
        [&](const std::string& line) {
            observation.lines.push_back(line);
            observe_line(observation, line);
            if (observation.token_events == 1 &&
                observation.terminal_events == 0 &&
                !cancellation_accepted) {
                cancellation_accepted = operations.cancel_action().ok();
            }
            return true;
        },
        &status,
        &body);

    CHECK(!completed);
    CHECK(cancellation_accepted);
    CHECK(status == 499);
    CHECK(body == "inference canceled");
    CHECK(observation.token_events == 1);
    CHECK(observation.terminal_events == 1);
    CHECK(observation.done_sentinels == 1);
    CHECK(!observation.event_after_terminal);
    CHECK(operations.cancel_calls == 1);

    const auto idle_cancel = operations.cancel_action();
    CHECK(idle_cancel.status == 409);
    CHECK(operations.cancel_calls == 2);
    return true;
}

bool test_consumer_cancellation_suppresses_later_callbacks() {
    ScriptedEmbeddedOperations operations;
    StreamObservation observation;
    int status = 0;
    std::string body;

    const bool completed = operations.stream_infer(
        nlohmann::json::object(),
        [&](const std::string& line) {
            observation.lines.push_back(line);
            observe_line(observation, line);
            return false;
        },
        &status,
        &body);

    CHECK(!completed);
    CHECK(status == 499);
    CHECK(body == "stream consumer canceled");
    CHECK(observation.lines.size() == 1);
    CHECK(observation.token_events == 1);
    CHECK(observation.terminal_events == 0);
    CHECK(observation.done_sentinels == 0);
    return true;
}

bool test_http_adapter_status_and_typed_error_contract() {
    HttpAdapterFixture fixture;
    auto& operations = fixture.operations();

    CHECK(!operations.embedded());
    CHECK(operations.endpoint().rfind("http://127.0.0.1:", 0) == 0);

    const auto health = operations.health();
    CHECK(health.ok());
    CHECK(health.body.value("status", std::string{}) == "ok");

    const auto status = operations.status();
    CHECK(status.ok());
    CHECK(status.body.value("hostname", std::string{}) == "http-adapter-test");
    CHECK(status.body.value("slot_available", -1) == 2);

    const auto missing = operations.load_model(nlohmann::json::object());
    CHECK(!missing.ok());
    CHECK(missing.status == 404);
    CHECK(missing.error_message() == "model missing");
    CHECK(missing.body.value("code", std::string{}) == "model_not_found");

    const auto capacity = operations.load_model(nlohmann::json{
        {"model_path", "canonical-model.gguf"},
        {"simulate_capacity_error", true},
    });
    CHECK(!capacity.ok());
    CHECK(capacity.status == 409);
    CHECK(capacity.error_message() == "capacity exhausted");
    CHECK(capacity.body.value("code", std::string{}) == "capacity_exhausted");
    return true;
}

bool test_http_adapter_exactly_once_stream_terminal() {
    HttpAdapterFixture fixture;
    StreamObservation observation;
    int status = 0;
    std::string body;

    const bool completed = fixture.operations().stream_infer(
        nlohmann::json{{"test_mode", "normal"}},
        [&](const std::string& line) {
            observation.lines.push_back(line);
            observe_line(observation, line);
            return true;
        },
        &status,
        &body);

    CHECK(completed);
    CHECK(status == 200);
    CHECK(observation.token_events == 2);
    CHECK(observation.terminal_events == 1);
    CHECK(observation.done_sentinels == 1);
    CHECK(!observation.event_after_terminal);
    CHECK(observation.lines.size() == 4);
    CHECK(observation.lines.back() == "[DONE]");
    return true;
}

bool test_http_adapter_action_cancellation_has_one_error_terminal() {
    HttpAdapterFixture fixture;
    auto& operations = fixture.operations();
    StreamObservation observation;
    int status = 0;
    std::string body;
    bool cancellation_accepted = false;

    const bool completed = operations.stream_infer(
        nlohmann::json{{"test_mode", "action_cancel"}},
        [&](const std::string& line) {
            observation.lines.push_back(line);
            observe_line(observation, line);
            if (observation.token_events == 1 &&
                observation.terminal_events == 0 &&
                !cancellation_accepted) {
                cancellation_accepted = operations.cancel_action().ok();
            }
            return true;
        },
        &status,
        &body);

    CHECK(completed);
    CHECK(cancellation_accepted);
    CHECK(status == 200);
    CHECK(observation.token_events == 1);
    CHECK(observation.terminal_events == 1);
    CHECK(observation.done_sentinels == 1);
    CHECK(!observation.event_after_terminal);
    CHECK(fixture.cancel_calls() == 1);

    const auto idle_cancel = operations.cancel_action();
    CHECK(idle_cancel.status == 409);
    CHECK(fixture.cancel_calls() == 2);
    return true;
}

bool test_http_adapter_consumer_cancellation_suppresses_later_callbacks() {
    HttpAdapterFixture fixture;
    StreamObservation observation;
    int status = -1;
    std::string body;

    const bool completed = fixture.operations().stream_infer(
        nlohmann::json{{"test_mode", "consumer_cancel"}},
        [&](const std::string& line) {
            observation.lines.push_back(line);
            observe_line(observation, line);
            return false;
        },
        &status,
        &body);

    CHECK(!completed);
    CHECK(status == 0);
    CHECK(observation.lines.size() == 1);
    CHECK(observation.token_events == 1);
    CHECK(observation.terminal_events == 0);
    CHECK(observation.done_sentinels == 0);
    return true;
}

bool test_http_adapter_filters_legacy_duplicate_terminals() {
    HttpAdapterFixture fixture;
    StreamObservation observation;
    int status = 0;
    std::string body;

    const bool completed = fixture.operations().stream_infer(
        nlohmann::json{{"test_mode", "legacy_duplicate"}},
        [&](const std::string& line) {
            observation.lines.push_back(line);
            observe_line(observation, line);
            return true;
        },
        &status,
        &body);

    CHECK(completed);
    CHECK(status == 200);
    CHECK(observation.token_events == 1);
    CHECK(observation.terminal_events == 1);
    CHECK(observation.done_sentinels == 1);
    CHECK(!observation.event_after_terminal);
    CHECK(observation.lines.size() == 3);
    CHECK(observation.lines.back() == "[DONE]");
    return true;
}

bool test_typed_embedded_inference_contract() {
    ScriptedEmbeddedOperations operations;
    mm::NodeInferRequest request;
    request.request.model = "normal";
    request.slot_id = "slot-local-0";

    std::string content;
    int callback_count = 0;
    int callback_terminals = 0;
    const auto result = operations.infer(
        request,
        [&](const mm::InferenceChunk& chunk) {
            ++callback_count;
            if (chunk.is_done) ++callback_terminals;
            content += chunk.delta_content;
            return true;
        });

    CHECK(result.ok());
    CHECK(result.status == mm::NodeServiceStatus::Ok);
    CHECK(result.slot_id == "slot-local-0");
    CHECK(result.message.content == "alphabeta");
    CHECK(content == result.message.content);
    CHECK(result.tokens_used == 2);
    CHECK(result.finish_reason == "stop");
    CHECK(callback_count == 2);
    CHECK(callback_terminals == 0);
    CHECK(operations.typed_infer_calls == 1);

    callback_count = 0;
    const auto canceled = operations.infer(
        request,
        [&](const mm::InferenceChunk&) {
            ++callback_count;
            return false;
        });
    CHECK(canceled.status == mm::NodeServiceStatus::Canceled);
    CHECK(callback_count == 1);
    CHECK(operations.typed_infer_calls == 2);
    return true;
}

bool test_http_adapter_typed_inference_contract() {
    HttpAdapterFixture fixture;
    mm::NodeInferRequest request;
    request.request.model = "normal";
    request.slot_id = "slot-http-0";

    std::string content;
    int callback_count = 0;
    int callback_terminals = 0;
    const auto result = fixture.operations().infer(
        request,
        [&](const mm::InferenceChunk& chunk) {
            ++callback_count;
            if (chunk.is_done) ++callback_terminals;
            content += chunk.delta_content;
            return true;
        });

    CHECK(result.ok());
    CHECK(result.status == mm::NodeServiceStatus::Ok);
    CHECK(result.slot_id == "slot-http-0");
    CHECK(result.message.content == "alphabeta");
    CHECK(content == result.message.content);
    CHECK(result.finish_reason == "stop");
    CHECK(callback_count == 2);
    CHECK(callback_terminals == 0);

    request.request.model = "legacy_duplicate";
    content.clear();
    callback_count = 0;
    const auto failed = fixture.operations().infer(
        request,
        [&](const mm::InferenceChunk& chunk) {
            ++callback_count;
            content += chunk.delta_content;
            return true;
        });
    CHECK(failed.status == mm::NodeServiceStatus::Failed);
    CHECK(failed.error == "legacy runtime error");
    CHECK(content == "alpha");
    CHECK(callback_count == 1);
    return true;
}

bool test_http_adapter_typed_cancellation_contract() {
    HttpAdapterFixture fixture;
    mm::NodeInferRequest request;
    request.request.model = "consumer_cancel";
    request.slot_id = "slot-http-0";

    int callback_count = 0;
    const auto consumer_canceled = fixture.operations().infer(
        request,
        [&](const mm::InferenceChunk&) {
            ++callback_count;
            return false;
        });
    CHECK(consumer_canceled.status == mm::NodeServiceStatus::Canceled);
    CHECK(callback_count == 1);

    std::atomic<bool> cancel_requested{false};
    request.request.model = "stalled_cancel";
    request.cancel_requested = [&] { return cancel_requested.load(); };
    callback_count = 0;
    std::thread cancel_thread([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cancel_requested = true;
    });
    const auto started = std::chrono::steady_clock::now();
    const auto stalled_canceled = fixture.operations().infer(
        request,
        [&](const mm::InferenceChunk&) {
            ++callback_count;
            return true;
        });
    const auto elapsed = std::chrono::steady_clock::now() - started;
    cancel_thread.join();

    CHECK(stalled_canceled.status == mm::NodeServiceStatus::Canceled);
    CHECK(callback_count == 0);
    CHECK(elapsed < std::chrono::seconds(2));
    return true;
}

// A real llama-server transport boundary for the adapter contract below.  It
// speaks the same OpenAI SSE protocol RuntimeClient consumes, but remains
// deterministic and model-free so the test can exercise NodeService and the
// node REST adapter in CI.
class FakeLlamaRuntime {
public:
    FakeLlamaRuntime() {
        server_.new_task_queue = [] { return new httplib::ThreadPool(4); };
        server_.Post("/v1/chat/completions",
            [this](const httplib::Request& request, httplib::Response& response) {
                ++request_count_;
                const auto body = nlohmann::json::parse(request.body);
                const std::string model = body.value("model", std::string{});

                if (model == "runtime-error") {
                    response.status = 503;
                    response.set_content(
                        R"({"error":{"message":"fake runtime unavailable"}})",
                        "application/json");
                    return;
                }

                if (model == "slow-cancel") {
                    auto steps = std::make_shared<std::atomic<int>>(0);
                    response.set_chunked_content_provider(
                        "text/event-stream",
                        [steps](size_t, httplib::DataSink& sink) {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(10));
                            const int step = ++*steps;
                            if (step < 200) {
                                // Keep producing real deltas so cancellation
                                // is exercised while both local and remote
                                // transports are actively delivering data.
                                static constexpr char delta[] =
                                    "data: {\"choices\":[{\"delta\":{"
                                    "\"content\":\"slow-token-\"}}]}\n\n";
                                return sink.write(delta, sizeof(delta) - 1);
                            }

                            static constexpr char terminal[] =
                                "data: {\"choices\":[{\"delta\":{},"
                                "\"finish_reason\":\"stop\"}],"
                                "\"usage\":{\"completion_tokens\":0}}\n\n"
                                "data: [DONE]\n\n";
                            if (!sink.write(terminal, sizeof(terminal) - 1)) {
                                return false;
                            }
                            sink.done();
                            return true;
                        },
                        [](bool) {});
                    return;
                }

                // The second event intentionally combines content, finish
                // reason, and usage.  This catches adapters which treat final
                // data and terminal accounting as mutually exclusive.
                const std::string stream =
                    "data: {\"choices\":[{\"delta\":{\"content\":"
                    "\"alpha-segment-\"}}]}\n\n"
                    "data: {\"choices\":[{\"delta\":{\"content\":"
                    "\"beta-final\"},\"finish_reason\":\"stop\"}],"
                    "\"usage\":{\"completion_tokens\":3}}\n\n"
                    "data: [DONE]\n\n";
                response.set_content(stream, "text/event-stream");
            });

        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) {
            throw std::runtime_error("failed to bind fake llama runtime");
        }
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        for (int attempt = 0; attempt < 500 && !server_.is_running(); ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!server_.is_running()) {
            server_.stop();
            if (thread_.joinable()) thread_.join();
            throw std::runtime_error("fake llama runtime did not start");
        }
    }

    ~FakeLlamaRuntime() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    std::string endpoint() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }
    int request_count() const { return request_count_.load(); }

private:
    httplib::Server server_;
    std::thread thread_;
    int port_ = 0;
    std::atomic<int> request_count_{0};
};

uint16_t find_released_loopback_port() {
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    if (port <= 0 || port > 65535) {
        throw std::runtime_error("failed to reserve node API test port");
    }
    // cpp-httplib's stop() only closes an active listener.  Briefly start the
    // bound probe before stopping it so the selected socket is truly released
    // for NodeApiServer rather than remaining held by a merely-bound Server.
    std::thread listener([&] { probe.listen_after_bind(); });
    for (int attempt = 0; attempt < 500 && !probe.is_running(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!probe.is_running()) {
        probe.stop();
        if (listener.joinable()) listener.join();
        throw std::runtime_error("node API port probe did not start");
    }
    probe.stop();
    if (listener.joinable()) listener.join();
    return static_cast<uint16_t>(port);
}

class RealAdapterContractFixture {
public:
    RealAdapterContractFixture()
        : temp_("real-adapter-contract")
        , models_dir_(temp_.path() / "models")
        , local_slots_(29710, 29711, 1)
        , local_service_(local_state_, local_slots_)
        , local_operations_(local_service_, models_dir_.string(), "offline")
        , remote_slots_(29720, 29721, 1) {
        std::filesystem::create_directories(models_dir_);
        capacity_model_path_ = models_dir_ / "capacity-model.gguf";
        write_file(capacity_model_path_, "capacity contract model");

        local_state_.set_registered(true, "local");
        remote_state_.set_registered(true, "real-http-node");
        remote_state_.add_api_key(api_key_);

        mm::LlamaRuntimeStatus runtime;
        runtime.status = "ready";
        runtime.executable_path = "fake-llama-server";
        local_state_.set_llama_runtime(runtime);
        remote_state_.set_llama_runtime(runtime);

        local_slot_id_ = local_slots_.add_ready_test_slot(
            "contract-model.gguf", "local-agent", {}, {}, llama_.endpoint());
        remote_slot_id_ = remote_slots_.add_ready_test_slot(
            "contract-model.gguf", "remote-agent", {}, {}, llama_.endpoint());
        if (local_slot_id_.empty() || remote_slot_id_.empty()) {
            throw std::runtime_error("failed to inject ready contract slots");
        }

        api_port_ = find_released_loopback_port();
        remote_api_ = std::make_unique<mm::NodeApiServer>(
            remote_state_, remote_slots_);
        api_thread_ = std::thread([this] {
            api_listen_ok_ = remote_api_->listen(api_port_);
        });

        remote_operations_ = std::make_unique<mm::HttpNodeOperations>(
            "http://127.0.0.1:" + std::to_string(api_port_), api_key_);
        bool ready = false;
        for (int attempt = 0; attempt < 200; ++attempt) {
            if (remote_operations_->health().status == 200) {
                ready = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (!ready) {
            remote_api_->stop();
            if (api_thread_.joinable()) api_thread_.join();
            throw std::runtime_error("real NodeApiServer did not start");
        }
    }

    ~RealAdapterContractFixture() {
        if (remote_api_) remote_api_->stop();
        if (api_thread_.joinable()) api_thread_.join();
    }

    mm::LocalNodeOperations& local() { return local_operations_; }
    mm::HttpNodeOperations& remote() { return *remote_operations_; }
    const std::string& local_slot_id() const { return local_slot_id_; }
    const std::string& remote_slot_id() const { return remote_slot_id_; }
    const std::filesystem::path& capacity_model_path() const {
        return capacity_model_path_;
    }
    std::filesystem::path missing_model_path() const {
        return models_dir_ / "missing-model.gguf";
    }
    int runtime_request_count() const { return llama_.request_count(); }

private:
    FakeLlamaRuntime llama_;
    TempDirectory temp_;
    std::filesystem::path models_dir_;
    mm::NodeState local_state_;
    mm::SlotManager local_slots_;
    mm::NodeService local_service_;
    mm::LocalNodeOperations local_operations_;
    mm::NodeState remote_state_;
    mm::SlotManager remote_slots_;
    const std::string api_key_ = "real-node-api-contract-key";
    uint16_t api_port_ = 0;
    std::unique_ptr<mm::NodeApiServer> remote_api_;
    std::thread api_thread_;
    std::atomic<bool> api_listen_ok_{false};
    std::unique_ptr<mm::HttpNodeOperations> remote_operations_;
    std::string local_slot_id_;
    std::string remote_slot_id_;
    std::filesystem::path capacity_model_path_;
};

mm::NodeInferRequest real_infer_request(const std::string& model,
                                        const std::string& slot_id) {
    mm::NodeInferRequest request;
    request.request.model = model;
    request.request.stream = true;
    mm::Message user;
    user.role = mm::MessageRole::User;
    user.content = "contract request";
    request.request.messages.push_back(std::move(user));
    request.slot_id = slot_id;
    return request;
}

struct TypedInferenceObservation {
    mm::NodeInferResult result;
    std::string content;
    int callbacks = 0;
    int callback_terminals = 0;
};

TypedInferenceObservation observe_typed_inference(
    mm::NodeOperations& operations,
    mm::NodeInferRequest request) {
    TypedInferenceObservation observation;
    observation.result = operations.infer(
        request,
        [&](const mm::InferenceChunk& chunk) {
            ++observation.callbacks;
            if (chunk.is_done) ++observation.callback_terminals;
            observation.content += chunk.delta_content;
            return true;
        });
    return observation;
}

bool test_real_local_and_http_adapter_success_contract() {
    RealAdapterContractFixture fixture;
    const auto local = observe_typed_inference(
        fixture.local(), real_infer_request("contract-success",
                                            fixture.local_slot_id()));
    const auto remote = observe_typed_inference(
        fixture.remote(), real_infer_request("contract-success",
                                              fixture.remote_slot_id()));

    CHECK(local.result.ok());
    CHECK(remote.result.ok());
    CHECK(local.result.status == remote.result.status);
    CHECK(local.result.message.content == "alpha-segment-beta-final");
    CHECK(remote.result.message.content == local.result.message.content);
    CHECK(local.content == local.result.message.content);
    CHECK(remote.content == remote.result.message.content);
    CHECK(local.result.tokens_used == 3);
    CHECK(remote.result.tokens_used == local.result.tokens_used);
    CHECK(local.result.finish_reason == "stop");
    CHECK(remote.result.finish_reason == local.result.finish_reason);
    CHECK(local.callbacks > 0);
    CHECK(remote.callbacks == local.callbacks);
    CHECK(local.callback_terminals == 0);
    CHECK(remote.callback_terminals == 0);

    // Exercise both legacy transport facades over the same real typed core.
    StreamObservation local_wire;
    StreamObservation remote_wire;
    auto run_wire = [](mm::NodeOperations& operations,
                       const std::string& slot_id,
                       StreamObservation& observation) {
        auto request = real_infer_request("contract-success", slot_id);
        nlohmann::json body = request.request;
        body["slot_id"] = slot_id;
        int status = 0;
        std::string response_body;
        const bool completed = operations.stream_infer(
            body,
            [&](const std::string& line) {
                observation.lines.push_back(line);
                observe_line(observation, line);
                return true;
            },
            &status,
            &response_body);
        // HttpClient retains a bounded copy of successful SSE bytes in its
        // diagnostic body, while the in-process adapter has no wire body.
        // Completion/status and the observed events are the shared contract.
        return completed && status == 200;
    };

    CHECK(run_wire(fixture.local(), fixture.local_slot_id(), local_wire));
    CHECK(run_wire(fixture.remote(), fixture.remote_slot_id(), remote_wire));
    CHECK(local_wire.token_events > 0);
    CHECK(remote_wire.token_events == local_wire.token_events);
    CHECK(local_wire.terminal_events == 1);
    CHECK(remote_wire.terminal_events == 1);
    CHECK(local_wire.done_sentinels == 1);
    CHECK(remote_wire.done_sentinels == 1);
    CHECK(!local_wire.event_after_terminal);
    CHECK(!remote_wire.event_after_terminal);
    CHECK(fixture.runtime_request_count() == 4);
    return true;
}

bool test_real_local_and_http_adapter_failure_contract() {
    RealAdapterContractFixture fixture;
    const auto local = observe_typed_inference(
        fixture.local(), real_infer_request("runtime-error",
                                            fixture.local_slot_id()));
    const auto remote = observe_typed_inference(
        fixture.remote(), real_infer_request("runtime-error",
                                              fixture.remote_slot_id()));

    CHECK(local.result.status == mm::NodeServiceStatus::Failed);
    CHECK(remote.result.status == local.result.status);
    CHECK(local.result.error.find("HTTP 503") != std::string::npos);
    CHECK(remote.result.error.find("HTTP 503") != std::string::npos);
    CHECK(local.callbacks == 0);
    CHECK(remote.callbacks == 0);
    CHECK(local.callback_terminals == 0);
    CHECK(remote.callback_terminals == 0);
    CHECK(fixture.runtime_request_count() == 2);

    // Model and capacity failures travel through the concrete NodeService in
    // both cases.  HTTP status codes remain backward-compatible, while the
    // additive typed code is identical at the transport-neutral boundary.
    const nlohmann::json missing_request{
        {"model_path", fixture.missing_model_path().string()},
        {"backend", "llama-cpp"},
    };
    const auto local_missing = fixture.local().load_model(missing_request);
    const auto remote_missing = fixture.remote().load_model(missing_request);
    CHECK(!local_missing.ok());
    CHECK(!remote_missing.ok());
    CHECK(local_missing.body.value("code", std::string{}) == "model_not_found");
    CHECK(remote_missing.body.value("code", std::string{}) ==
          local_missing.body.value("code", std::string{}));

    const nlohmann::json capacity_request{
        {"model_path", fixture.capacity_model_path().string()},
        {"backend", "llama-cpp"},
    };
    const auto local_capacity = fixture.local().load_model(capacity_request);
    const auto remote_capacity = fixture.remote().load_model(capacity_request);
    CHECK(!local_capacity.ok());
    CHECK(!remote_capacity.ok());
    CHECK(local_capacity.body.value("code", std::string{}) ==
          "capacity_exhausted");
    CHECK(remote_capacity.body.value("code", std::string{}) ==
          local_capacity.body.value("code", std::string{}));
    return true;
}

bool test_real_local_and_http_adapter_cancellation_contract() {
    RealAdapterContractFixture fixture;

    auto run_canceled = [](mm::NodeOperations& operations,
                           const std::string& slot_id) {
        std::atomic<bool> cancel_requested{false};
        auto request = real_infer_request("slow-cancel", slot_id);
        request.cancel_requested = [&] { return cancel_requested.load(); };
        int callbacks = 0;
        std::thread cancel_thread([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(75));
            cancel_requested = true;
        });
        const auto started = std::chrono::steady_clock::now();
        const auto result = operations.infer(
            request,
            [&](const mm::InferenceChunk&) {
                ++callbacks;
                return true;
            });
        const auto elapsed = std::chrono::steady_clock::now() - started;
        cancel_thread.join();
        return std::tuple{result, callbacks, elapsed};
    };

    const auto [local, local_callbacks, local_elapsed] =
        run_canceled(fixture.local(), fixture.local_slot_id());
    const auto [remote, remote_callbacks, remote_elapsed] =
        run_canceled(fixture.remote(), fixture.remote_slot_id());

    CHECK(local.status == mm::NodeServiceStatus::Canceled);
    CHECK(remote.status == local.status);
    CHECK(local_callbacks > 0);
    CHECK(remote_callbacks > 0);
    CHECK(local_elapsed < std::chrono::seconds(1));
    CHECK(remote_elapsed < std::chrono::seconds(1));
    CHECK(fixture.runtime_request_count() == 2);
    return true;
}

class LocalAdapterFixture {
public:
    explicit LocalAdapterFixture(const std::string& policy = "prompt")
        : temp_("local-adapter")
        , models_dir_(temp_.path() / "models")
        , slots_(29600, 29601, 2)
        , service_(state_, slots_)
        , operations_(service_, models_dir_.string(), policy) {
        std::filesystem::create_directories(models_dir_);
        state_.set_registered(true, "local");
        slots_.set_models_dir(models_dir_.string());
    }

    const std::filesystem::path& models_dir() const { return models_dir_; }
    mm::LocalNodeOperations& operations() { return operations_; }
    mm::NodeService& service() { return service_; }

private:
    TempDirectory temp_;
    std::filesystem::path models_dir_;
    mm::NodeState state_;
    mm::SlotManager slots_;
    mm::NodeService service_;
    mm::LocalNodeOperations operations_;
};

bool test_local_adapter_direct_model_and_consent_contract() {
    LocalAdapterFixture fixture;
    auto& operations = fixture.operations();
    CHECK(operations.embedded());
    CHECK(operations.endpoint().empty());

    const auto model_path = fixture.models_dir() / "catalog-model.gguf";
    write_file(model_path, "test model bytes");
    std::string error;
    const auto prepared = operations.prepare_model(
        model_path.string(), "catalog-id", false, false, &error);
    CHECK(prepared.has_value());
    CHECK(error.empty());
    CHECK(prepared->model_id == "catalog-id");
    CHECK(std::filesystem::equivalent(prepared->load_path, model_path));

    // Direct preparation must not create an embedded-node cache copy.
    int model_file_count = 0;
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(fixture.models_dir())) {
        if (entry.is_regular_file()) ++model_file_count;
    }
    CHECK(model_file_count == 1);
    CHECK(std::filesystem::exists(model_path));

    const auto invalid_load = operations.load_model(nlohmann::json::object());
    CHECK(invalid_load.status == 400);
    CHECK(invalid_load.body.value("code", std::string{}) ==
          "invalid_argument");

    const auto provision_without_consent =
        operations.llama_provision(nlohmann::json::object());
    CHECK(provision_without_consent.status == 403);
    CHECK(provision_without_consent.body.value(
        "requires_network_consent", false));
    CHECK(operations.llama_update(nlohmann::json::object()).status == 403);
    CHECK(operations.llama_check_update().status == 403);
    CHECK(operations.pair_request(nlohmann::json::object()).status == 409);

    operations.request_shutdown();
    CHECK(std::filesystem::exists(model_path));

    return true;
}

bool test_runtime_network_policy_matrix() {
    // The fixtures deliberately leave runtime callbacks unconfigured.  A 501
    // therefore proves the request passed the policy gate without performing
    // network I/O, while a 403 proves it was rejected at that gate.
    LocalAdapterFixture prompt("prompt");
    const auto prompt_allowed = prompt.operations().llama_provision(
        nlohmann::json{{"allow_network", true}});
    CHECK(prompt_allowed.status == 501);
    CHECK(prompt.operations().llama_check_update(
        nlohmann::json{{"allow_network", true}}).status == 501);

    LocalAdapterFixture automatic("auto");
    CHECK(automatic.operations().llama_provision(
        nlohmann::json::object()).status == 501);
    CHECK(automatic.operations().llama_check_update().status == 501);

    LocalAdapterFixture offline("offline");
    const auto offline_denied = offline.operations().llama_provision(
        nlohmann::json{{"allow_network", true}});
    CHECK(offline_denied.status == 403);
    CHECK(offline_denied.body.value("runtime_network_policy", std::string{}) ==
          "offline");
    CHECK(!offline_denied.body.value("requires_network_consent", true));
    CHECK(offline.operations().llama_check_update(
        nlohmann::json{{"allow_network", true}}).status == 403);
    return true;
}

bool test_local_runtime_queue_rechecks_shutdown() {
    LocalAdapterFixture fixture("auto");
    auto& operations = fixture.operations();
    std::atomic<bool> update_entered{false};
    std::atomic<bool> release_update{false};
    std::atomic<int> check_callbacks{0};

    fixture.service().set_llama_update_callback([&](const std::string&) {
        update_entered = true;
        while (!release_update.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        mm::LlamaRuntimeStatus status;
        status.status = "ready";
        status.executable_path = "test-llama-server";
        return status;
    });
    fixture.service().set_llama_check_update_callback([&] {
        ++check_callbacks;
        mm::LlamaRuntimeStatus status;
        status.status = "ready";
        return status;
    });

    mm::NodeOperationResult active_result;
    mm::NodeOperationResult queued_result;
    std::thread active([&] {
        active_result = operations.llama_update(nlohmann::json::object());
    });
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    while (!update_entered.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    std::thread queued([&] {
        queued_result = operations.llama_check_update();
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    operations.request_shutdown();
    release_update = true;
    active.join();
    queued.join();

    CHECK(update_entered.load());
    CHECK(active_result.status == 409);
    CHECK(queued_result.status == 409);
    CHECK(check_callbacks.load() == 0);
    CHECK(operations.llama_provision(nlohmann::json::object()).status == 409);
    CHECK(operations.llama_update(nlohmann::json::object()).status == 409);
    CHECK(operations.llama_check_update().status == 409);
    CHECK(operations.llama_switch(
        nlohmann::json{{"variant", "cpu"}}).status == 409);
    CHECK(operations.llama_diagnose().status == 409);
    CHECK(operations.llama_recover(
        nlohmann::json{{"action", "retry"}}).status == 409);
    return true;
}

bool test_local_adapter_exactly_once_failure_terminal() {
    LocalAdapterFixture fixture;
    auto& operations = fixture.operations();
    StreamObservation observation;
    int status = 0;
    std::string body;

    mm::NodeInferRequest typed_request;
    typed_request.request.stream = true;
    int typed_callbacks = 0;
    const auto typed_result = operations.infer(
        typed_request,
        [&](const mm::InferenceChunk&) {
            ++typed_callbacks;
            return true;
        });
    CHECK(typed_result.status == mm::NodeServiceStatus::Unavailable);
    CHECK(typed_result.error == "no ready slot available");
    CHECK(typed_callbacks == 0);

    // No slot is loaded.  The typed unavailable failure must still produce
    // exactly one terminal error and exactly one transport sentinel.
    const bool completed = operations.stream_infer(
        nlohmann::json::object(),
        [&](const std::string& line) {
            observation.lines.push_back(line);
            observe_line(observation, line);
            return true;
        },
        &status,
        &body);
    CHECK(!completed);
    CHECK(status == 503);
    CHECK(observation.terminal_events == 1);
    CHECK(observation.done_sentinels == 1);
    CHECK(!observation.event_after_terminal);
    CHECK(observation.lines.size() == 2);
    CHECK(nlohmann::json::parse(body).value("code", std::string{}) ==
          "unavailable");

    // Consumer cancellation on that first terminal callback suppresses the
    // [DONE] callback and every possible later callback.
    observation = {};
    status = -1;
    body.clear();
    const bool canceled = operations.stream_infer(
        nlohmann::json::object(),
        [&](const std::string& line) {
            observation.lines.push_back(line);
            observe_line(observation, line);
            return false;
        },
        &status,
        &body);
    CHECK(!canceled);
    CHECK(status == 0);
    CHECK(observation.lines.size() == 1);
    CHECK(observation.terminal_events == 1);
    CHECK(observation.done_sentinels == 0);
    CHECK(body.find("stream canceled by consumer") != std::string::npos);
    return true;
}

bool test_embedded_registry_invariants_and_remote_gate() {
    TempDirectory temp("registry");
    const auto nodes_path = temp.path() / "nodes.json";
    const std::string remembered_contents =
        R"({"nodes":[{"id":"remembered","url":"http://192.0.2.1:7070","api_key":"secret"}]})";
    write_file(nodes_path, remembered_contents);

    mm::NodeRegistry registry(temp.path().string(), false);
    CHECK(!registry.remote_nodes_enabled());
    CHECK(registry.list_nodes().empty());

    auto first = std::make_shared<ScriptedEmbeddedOperations>();
    CHECK(registry.add_embedded_node(first, "test-platform", "test-host") ==
          "local");
    CHECK(first->health_calls == 1);
    CHECK(first->status_calls == 1);

    const auto local = registry.get_node("local");
    CHECK(local.id == "local");
    CHECK(local.kind == "embedded");
    CHECK(local.url.empty());
    CHECK(local.api_key.empty());
    CHECK(local.connected);
    CHECK(local.connection_status == mm::NodeConnectionStatus::Online);
    CHECK(!local.remembered);
    CHECK(registry.is_embedded_node("local"));
    CHECK(registry.operations("local") == first);

    const nlohmann::json public_json = local;
    CHECK(public_json.value("kind", std::string{}) == "embedded");
    CHECK(public_json.value("url", std::string{"not-empty"}).empty());
    CHECK(!public_json.contains("api_key"));

    // Re-registration keeps the reserved identity and replaces its in-process
    // transport without creating a second node.
    auto replacement = std::make_shared<ScriptedEmbeddedOperations>();
    CHECK(registry.add_embedded_node(replacement) == "local");
    CHECK(first->shutdown_calls == 1);
    CHECK(registry.list_nodes().size() == 1);
    CHECK(registry.operations("local") == replacement);

    registry.remove_node("local");
    CHECK(registry.list_nodes().size() == 1);
    CHECK(registry.is_embedded_node("local"));
    CHECK(!registry.forget_node("local"));
    CHECK(registry.get_node("local").kind == "embedded");

    bool add_rejected = false;
    try {
        (void)registry.add_node("http://127.0.0.1:7070", "remote-key");
    } catch (const std::runtime_error& error) {
        add_rejected = std::string(error.what()).find("disabled") !=
                       std::string::npos;
    }
    CHECK(add_rejected);
    CHECK(registry.start_pair("http://127.0.0.1:7070").empty());
    CHECK(registry.complete_pair("http://127.0.0.1:7070", "nonce", "pin")
              .empty());
    CHECK(registry.pair_node("http://127.0.0.1:7070", "pin").empty());
    registry.start_discovery_listen(0);
    CHECK(registry.get_discovered_nodes().empty());
    registry.stop_discovery_listen();

    // Local-only runs neither load nor rewrite remembered remote nodes.
    CHECK(read_file(nodes_path) == remembered_contents);
    return true;
}

bool test_missing_kind_defaults_to_remote() {
    const auto legacy = nlohmann::json{
        {"id", "legacy-node"},
        {"url", "http://127.0.0.1:7070"},
        {"connected", true},
    }.get<mm::NodeInfo>();
    CHECK(legacy.kind == "remote");
    return true;
}

bool test_persisted_reserved_local_id_is_normalized() {
    TempDirectory temp("reserved-local-id");
    const auto nodes_path = temp.path() / "nodes.json";
    write_file(nodes_path,
               R"({"version":2,"nodes":[{"id":"local","url":"http://127.0.0.1:65534","api_key":"remote-secret","hostname":"legacy-remote"}]})");

    mm::NodeRegistry registry(temp.path().string(), true);
    const auto loaded = registry.list_nodes();
    CHECK(loaded.size() == 1);
    CHECK(loaded.front().id != "local");
    CHECK(loaded.front().kind == "remote");
    CHECK(loaded.front().remembered);

    const auto persisted = nlohmann::json::parse(read_file(nodes_path));
    CHECK(persisted.at("nodes").size() == 1);
    CHECK(persisted.at("nodes").at(0).value("id", std::string{}) != "local");

    auto embedded = std::make_shared<ScriptedEmbeddedOperations>();
    CHECK(registry.add_embedded_node(embedded) == "local");
    CHECK(registry.list_nodes().size() == 2);
    const auto after_embed = nlohmann::json::parse(read_file(nodes_path));
    for (const auto& item : after_embed.at("nodes")) {
        CHECK(item.value("id", std::string{}) != "local");
    }
    return true;
}

bool test_http_stream_cancel_before_first_byte() {
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(2); };
    std::atomic<bool> handler_entered{false};
    std::atomic<bool> release_handler{false};
    server.Post("/stall-before-headers",
                [&](const httplib::Request&, httplib::Response& response) {
                    handler_entered = true;
                    const auto deadline = std::chrono::steady_clock::now() +
                                          std::chrono::seconds(2);
                    while (!release_handler.load() &&
                           std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(5));
                    }
                    response.set_content(
                        "data: {\"type\":\"done\"}\n\n"
                        "data: [DONE]\n\n",
                        "text/event-stream");
                });

    const int port = server.bind_to_any_port("127.0.0.1");
    CHECK(port > 0);
    std::thread server_thread([&] { server.listen_after_bind(); });
    for (int attempt = 0; attempt < 500 && !server.is_running(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!server.is_running()) {
        server.stop();
        if (server_thread.joinable()) server_thread.join();
        CHECK(false);
    }

    mm::HttpClient client("http://127.0.0.1:" + std::to_string(port));
    client.set_timeouts(2, 3600, 30);
    std::atomic<bool> cancel_requested{false};
    std::thread cancel_thread([&] {
        for (int attempt = 0; attempt < 200 && !handler_entered.load();
             ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(75));
        cancel_requested = true;
    });

    int status = -1;
    std::string response_body;
    const auto started = std::chrono::steady_clock::now();
    const bool completed = client.stream_post(
        "/stall-before-headers",
        nlohmann::json::object(),
        [](const std::string&) { return true; },
        &status,
        &response_body,
        [&] { return cancel_requested.load(); });
    const auto elapsed = std::chrono::steady_clock::now() - started;

    release_handler = true;
    if (cancel_thread.joinable()) cancel_thread.join();
    server.stop();
    if (server_thread.joinable()) server_thread.join();

    CHECK(handler_entered.load());
    CHECK(!completed);
    CHECK(status == 0);
    CHECK(response_body.empty());
    CHECK(elapsed < std::chrono::seconds(1));
    return true;
}

bool test_http_node_runtime_shutdown_cancels_silent_remote() {
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(2); };
    std::atomic<bool> handler_entered{false};
    std::atomic<bool> release_handler{false};
    server.Post("/api/node/runtime/llama/provision",
                [&](const httplib::Request&, httplib::Response& response) {
                    handler_entered = true;
                    const auto deadline = std::chrono::steady_clock::now() +
                                          std::chrono::seconds(2);
                    while (!release_handler.load() &&
                           std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(5));
                    }
                    response.set_content(
                        R"({"llama_runtime":{"status":"ready"}})",
                        "application/json");
                });

    const int port = server.bind_to_any_port("127.0.0.1");
    CHECK(port > 0);
    std::thread server_thread([&] { server.listen_after_bind(); });
    for (int attempt = 0; attempt < 500 && !server.is_running(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!server.is_running()) {
        server.stop();
        if (server_thread.joinable()) server_thread.join();
        CHECK(false);
    }

    mm::HttpNodeOperations operations(
        "http://127.0.0.1:" + std::to_string(port), "node-key");
    mm::NodeOperationResult result;
    std::thread operation_thread([&] {
        result = operations.llama_provision(nlohmann::json::object());
    });
    for (int attempt = 0; attempt < 500 && !handler_entered.load(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const auto shutdown_started = std::chrono::steady_clock::now();
    operations.request_shutdown();
    if (operation_thread.joinable()) operation_thread.join();
    const auto shutdown_elapsed =
        std::chrono::steady_clock::now() - shutdown_started;

    release_handler = true;
    server.stop();
    if (server_thread.joinable()) server_thread.join();

    CHECK(handler_entered.load());
    CHECK(result.status == 0);
    CHECK(shutdown_elapsed < std::chrono::seconds(1));
    return true;
}

bool test_registry_shutdown_cancels_transient_pairing() {
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(2); };
    std::atomic<bool> handler_entered{false};
    std::atomic<bool> release_handler{false};
    server.Post("/api/node/pair-request",
                [&](const httplib::Request&, httplib::Response& response) {
                    handler_entered = true;
                    const auto deadline = std::chrono::steady_clock::now() +
                                          std::chrono::seconds(2);
                    while (!release_handler.load() &&
                           std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(5));
                    }
                    response.set_content(
                        R"({"accepted":true,"mode":"pin"})",
                        "application/json");
                });

    const int port = server.bind_to_any_port("127.0.0.1");
    CHECK(port > 0);
    std::thread server_thread([&] { server.listen_after_bind(); });
    for (int attempt = 0; attempt < 500 && !server.is_running(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    mm::NodeRegistry registry({}, true);
    std::string nonce;
    std::thread pair_thread([&] {
        nonce = registry.start_pair(
            "http://127.0.0.1:" + std::to_string(port));
    });
    for (int attempt = 0; attempt < 500 && !handler_entered.load(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const auto shutdown_started = std::chrono::steady_clock::now();
    registry.request_operations_shutdown();
    pair_thread.join();
    const auto shutdown_elapsed =
        std::chrono::steady_clock::now() - shutdown_started;

    release_handler = true;
    server.stop();
    server_thread.join();

    CHECK(handler_entered.load());
    CHECK(nonce.empty());
    CHECK(shutdown_elapsed < std::chrono::seconds(1));
    return true;
}

bool test_runtime_stream_cancel_before_first_byte() {
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(2); };
    std::atomic<bool> handler_entered{false};
    std::atomic<bool> release_handler{false};
    server.Post("/v1/chat/completions",
                [&](const httplib::Request&, httplib::Response& response) {
                    handler_entered = true;
                    const auto deadline = std::chrono::steady_clock::now() +
                                          std::chrono::seconds(2);
                    while (!release_handler.load() &&
                           std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(5));
                    }
                    response.set_content(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n"
                        "data: [DONE]\n\n",
                        "text/event-stream");
                });

    const int port = server.bind_to_any_port("127.0.0.1");
    CHECK(port > 0);
    std::thread server_thread([&] { server.listen_after_bind(); });
    for (int attempt = 0; attempt < 500 && !server.is_running(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    mm::RuntimeClient runtime(
        "http://127.0.0.1:" + std::to_string(port));
    mm::InferenceRequest request;
    request.messages.push_back(
        mm::Message{.role = mm::MessageRole::User, .content = "cancel"});
    std::atomic<bool> cancel_requested{false};
    std::thread cancel_thread([&] {
        while (!handler_entered.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(75));
        cancel_requested = true;
    });

    int chunks = 0;
    int errors = 0;
    const auto started = std::chrono::steady_clock::now();
    runtime.stream_complete(
        request,
        [&](const mm::InferenceChunk&) { ++chunks; },
        [&](const std::string&) { ++errors; },
        [&] { return cancel_requested.load(); });
    const auto elapsed = std::chrono::steady_clock::now() - started;

    release_handler = true;
    cancel_thread.join();
    server.stop();
    server_thread.join();

    CHECK(handler_entered.load());
    CHECK(chunks == 0);
    CHECK(errors == 0);
    CHECK(elapsed < std::chrono::seconds(1));
    return true;
}

bool test_runtime_complete_cancel_before_first_byte() {
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(2); };
    std::atomic<bool> handler_entered{false};
    std::atomic<bool> release_handler{false};
    server.Post("/v1/chat/completions",
                [&](const httplib::Request&, httplib::Response& response) {
                    handler_entered = true;
                    const auto deadline = std::chrono::steady_clock::now() +
                                          std::chrono::seconds(2);
                    while (!release_handler.load() &&
                           std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(5));
                    }
                    response.set_content(
                        R"({"choices":[{"message":{"role":"assistant","content":"late"}}]})",
                        "application/json");
                });

    const int port = server.bind_to_any_port("127.0.0.1");
    CHECK(port > 0);
    std::thread server_thread([&] { server.listen_after_bind(); });
    for (int attempt = 0; attempt < 500 && !server.is_running(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    mm::RuntimeClient runtime(
        "http://127.0.0.1:" + std::to_string(port));
    mm::InferenceRequest request;
    request.messages.push_back(
        mm::Message{.role = mm::MessageRole::User, .content = "cancel"});
    std::atomic<bool> cancel_requested{false};
    std::thread cancel_thread([&] {
        while (!handler_entered.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(75));
        cancel_requested = true;
    });

    const auto started = std::chrono::steady_clock::now();
    const auto response = runtime.complete(
        request, [&] { return cancel_requested.load(); });
    const auto elapsed = std::chrono::steady_clock::now() - started;

    release_handler = true;
    cancel_thread.join();
    server.stop();
    server_thread.join();

    CHECK(handler_entered.load());
    CHECK(response.content.empty());
    CHECK(elapsed < std::chrono::seconds(1));
    return true;
}

bool test_forced_slot_cleanup_waits_for_active_lease() {
    mm::SlotManager slots(29810, 29811, 1);
    const auto slot_id = slots.add_ready_test_slot();
    CHECK(!slot_id.empty());
    auto lease = slots.acquire_slot(slot_id);
    CHECK(static_cast<bool>(lease));

    std::atomic<bool> unload_finished{false};
    mm::SlotOperationResult unload_result;
    std::thread unload_thread([&] {
        unload_result = slots.unload_all(true);
        unload_finished = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(75));
    const bool returned_while_leased = unload_finished.load();

    lease = {};
    if (unload_thread.joinable()) unload_thread.join();

    CHECK(!returned_while_leased);
    CHECK(unload_finished.load());
    CHECK(unload_result.ok());
    CHECK(!slots.find_slot(slot_id).has_value());
    return true;
}

} // namespace

int main() {
    struct TestCase {
        const char* name;
        bool (*run)();
    };
    const TestCase tests[] = {
        {"result_and_typed_error_contract", test_result_and_typed_error_contract},
        {"exactly_once_stream_terminal", test_exactly_once_stream_terminal},
        {"action_cancellation_has_one_error_terminal",
         test_action_cancellation_has_one_error_terminal},
        {"consumer_cancellation_suppresses_later_callbacks",
         test_consumer_cancellation_suppresses_later_callbacks},
        {"http_adapter_status_and_typed_error_contract",
         test_http_adapter_status_and_typed_error_contract},
        {"http_adapter_exactly_once_stream_terminal",
         test_http_adapter_exactly_once_stream_terminal},
        {"http_adapter_action_cancellation_has_one_error_terminal",
         test_http_adapter_action_cancellation_has_one_error_terminal},
        {"http_adapter_consumer_cancellation_suppresses_later_callbacks",
         test_http_adapter_consumer_cancellation_suppresses_later_callbacks},
        {"http_adapter_filters_legacy_duplicate_terminals",
         test_http_adapter_filters_legacy_duplicate_terminals},
        {"typed_embedded_inference_contract",
         test_typed_embedded_inference_contract},
        {"http_adapter_typed_inference_contract",
         test_http_adapter_typed_inference_contract},
        {"http_adapter_typed_cancellation_contract",
         test_http_adapter_typed_cancellation_contract},
        {"real_local_and_http_adapter_success_contract",
         test_real_local_and_http_adapter_success_contract},
        {"real_local_and_http_adapter_failure_contract",
         test_real_local_and_http_adapter_failure_contract},
        {"real_local_and_http_adapter_cancellation_contract",
         test_real_local_and_http_adapter_cancellation_contract},
        {"local_adapter_direct_model_and_consent_contract",
         test_local_adapter_direct_model_and_consent_contract},
        {"runtime_network_policy_matrix", test_runtime_network_policy_matrix},
        {"local_runtime_queue_rechecks_shutdown",
         test_local_runtime_queue_rechecks_shutdown},
        {"local_adapter_exactly_once_failure_terminal",
         test_local_adapter_exactly_once_failure_terminal},
        {"embedded_registry_invariants_and_remote_gate",
         test_embedded_registry_invariants_and_remote_gate},
        {"missing_kind_defaults_to_remote", test_missing_kind_defaults_to_remote},
        {"persisted_reserved_local_id_is_normalized",
         test_persisted_reserved_local_id_is_normalized},
        {"http_stream_cancel_before_first_byte",
         test_http_stream_cancel_before_first_byte},
        {"http_node_runtime_shutdown_cancels_silent_remote",
         test_http_node_runtime_shutdown_cancels_silent_remote},
        {"registry_shutdown_cancels_transient_pairing",
         test_registry_shutdown_cancels_transient_pairing},
        {"runtime_stream_cancel_before_first_byte",
         test_runtime_stream_cancel_before_first_byte},
        {"runtime_complete_cancel_before_first_byte",
         test_runtime_complete_cancel_before_first_byte},
        {"forced_slot_cleanup_waits_for_active_lease",
         test_forced_slot_cleanup_waits_for_active_lease},
    };

    for (const auto& test : tests) {
        if (!test.run()) return 1;
        std::cout << "[PASS] " << test.name << '\n';
    }
    return 0;
}
