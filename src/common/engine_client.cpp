// Mantic-Mind — EngineClient: the HTTP surface of an engine, behind a virtual
// streaming call.
//
// The OpenAI wire format is genuinely identical for both engines, so the SSE
// machinery is NOT reimplemented here: both impls delegate to RuntimeClient,
// which already owns request building, <think> extraction, tool-call delta
// accumulation, and — the property this header promises — exactly one is_done
// chunk on every path including error.
//
// What this layer adds is the two things RuntimeClient cannot express:
//
//   * a VIRTUAL stream_complete. Non-virtual today, which is why control gives
//     up on the abstraction and calls node_cli.stream_post("/api/node/infer")
//     inline at control_api_server.cpp:1949 with its own retry loop.
//   * a STRUCTURED error. AgentScheduler currently substring-matches six English
//     phrases against the node's error body to decide whether to evict and
//     retry; a new engine would have to emit those literals verbatim to earn the
//     same treatment.

#include "common/engine_client.hpp"

#include "common/logger.hpp"
#include "common/runtime_client.hpp"
#include "common/util.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <exception>
#include <string>

namespace mm {

namespace {

std::string normalize_base_url(const std::string& url) {
    const std::string trimmed = util::trim(url);
    if (trimmed.rfind("http://", 0) == 0 || trimmed.rfind("https://", 0) == 0) return trimmed;
    return "http://" + trimmed;
}

httplib::Client make_client(const std::string& base_url, const std::string& api_key) {
    httplib::Client cli(normalize_base_url(base_url));
    cli.set_connection_timeout(5);
    cli.set_read_timeout(30);
    if (!api_key.empty()) cli.set_bearer_token_auth(api_key);
    return cli;
}

EngineError parse_engine_error(const std::string& raw) {
    EngineError err;
    err.message = raw;

    std::string body = raw;
    if (raw.rfind("HTTP ", 0) == 0) {
        const auto colon = raw.find(':');
        const std::string status_text =
            raw.substr(5, (colon == std::string::npos ? raw.size() : colon) - 5);
        try {
            err.http_status = std::stoi(util::trim(status_text));
        } catch (...) {
            err.http_status = 0;
        }
        body = (colon == std::string::npos) ? std::string{} : util::trim(raw.substr(colon + 1));
    }

    if (!body.empty()) {
        try {
            const auto j = nlohmann::json::parse(body);
            if (j.contains("error") && j["error"].is_object()) {
                const auto& e = j["error"];
                if (e.contains("code") && e["code"].is_string()) err.code = e["code"];
                if (e.contains("message") && e["message"].is_string()) err.message = e["message"];
            }
        } catch (const std::exception&) {
            // Not JSON. Leave the code empty rather than guessing: an unrecognised
            // body is exactly the case where the old substring matcher produced
            // false positives.
        }
    }

    if (err.code.empty()) {
        // Status-derived fallback, and only for codes a status determines
        // unambiguously. 503 from an engine means it cannot take the work now.
        if (err.http_status == 503)
            err.code = "capacity_pressure";
        else if (err.http_status == 404)
            err.code = "model_not_found";
        else if (err.http_status == 422)
            err.code = "unsupported_content";
        else if (err.http_status >= 500)
            err.code = "internal";
        else if (err.http_status == 0)
            err.code = "internal"; // connection failure
    }
    return err;
}

} // namespace

// ── EngineError ───────────────────────────────────────────────────────────────

EngineError EngineError::parse(const std::string& raw) {
    return parse_engine_error(raw);
}

bool EngineError::is_capacity_pressure() const {
    return code == "capacity_pressure";
}

// ── EngineClient ──────────────────────────────────────────────────────────────

EngineClient::EngineClient(std::string base_url, std::string api_key)
    : base_url_(std::move(base_url)), api_key_(std::move(api_key)) {}

const std::string& EngineClient::base_url() const {
    return base_url_;
}

bool EngineClient::count_tokens(const std::string&, int&) {
    return false;
}

bool EngineClient::query_capabilities(std::string&) {
    return false;
}

// ── LlamaEngineClient ─────────────────────────────────────────────────────────

LlamaEngineClient::LlamaEngineClient(std::string base_url, std::string api_key)
    : EngineClient(std::move(base_url), std::move(api_key)) {}

Message LlamaEngineClient::complete(const InferenceRequest& req) {
    RuntimeClient rc(base_url_, api_key_);
    return rc.complete(req);
}

void LlamaEngineClient::stream_complete(const InferenceRequest& req,
                                        ChunkCallback chunk_cb,
                                        ErrorCallback error_cb) {
    RuntimeClient rc(base_url_, api_key_);
    rc.stream_complete(req, std::move(chunk_cb), [error_cb](const std::string& raw) {
        if (error_cb) error_cb(parse_engine_error(raw));
    });
}

bool LlamaEngineClient::health_check() {
    RuntimeClient rc(base_url_, api_key_);
    return rc.health_check();
}

bool LlamaEngineClient::count_tokens(const std::string& text, int& out_tokens) {
    if (text.empty()) {
        out_tokens = 0;
        return true;
    }
    RuntimeClient rc(base_url_, api_key_);
    const int n = rc.count_tokens(text);
    // RuntimeClient returns 0 for both "no tokens" and "the call failed", which
    // is why this signature reports success separately. Non-empty text is never
    // legitimately zero tokens, so the empty case is handled above and 0 here
    // means the request failed.
    if (n <= 0) return false;
    out_tokens = n;
    return true;
}

bool LlamaEngineClient::query_capabilities(std::string& out_json) {
    // GET /props, for modalities.vision after a --mmproj launch. Called from
    // SlotManager directly today, which is why the interface had to grow a
    // llama-shaped hole; it is at least a NAMED hole now, defaulting to false.
    auto cli = make_client(base_url_, api_key_);
    auto res = cli.Get("/props");
    if (!res || res->status != 200) return false;
    out_json = res->body;
    return true;
}

// ── VllmEngineClient ─────────────────────────────────────────────────────────

VllmEngineClient::VllmEngineClient(std::string base_url, std::string api_key)
    : EngineClient(std::move(base_url), std::move(api_key)) {}

Message VllmEngineClient::complete(const InferenceRequest& req) {
    RuntimeClient rc(base_url_, api_key_);
    return rc.complete(req);
}

void VllmEngineClient::stream_complete(const InferenceRequest& req,
                                       ChunkCallback chunk_cb,
                                       ErrorCallback error_cb) {
    RuntimeClient rc(base_url_, api_key_);
    rc.stream_complete(req, std::move(chunk_cb), [error_cb](const std::string& raw) {
        if (error_cb) error_cb(parse_engine_error(raw));
    });
}

bool VllmEngineClient::health_check() {
    auto cli = make_client(base_url_, api_key_);
    const auto res = cli.Get("/health");
    return res && res->status == 200;
}

bool VllmEngineClient::sleep(int level, std::string& error) {
    error.clear();
    auto cli = make_client(base_url_, api_key_);
    const auto res = cli.Post(("/sleep?level=" + std::to_string(level)).c_str());
    if (res && res->status >= 200 && res->status < 300) return true;
    error = res ? "HTTP " + std::to_string(res->status) + ": " + res->body
                : "vLLM sleep endpoint is unreachable";
    return false;
}

bool VllmEngineClient::wake(std::string& error) {
    error.clear();
    auto cli = make_client(base_url_, api_key_);
    const auto res = cli.Post("/wake_up");
    if (res && res->status >= 200 && res->status < 300) return true;
    error = res ? "HTTP " + std::to_string(res->status) + ": " + res->body
                : "vLLM wake endpoint is unreachable";
    return false;
}

// ── SomaEngineClient ──────────────────────────────────────────────────────────

SomaEngineClient::SomaEngineClient(std::string base_url, std::string api_key)
    : EngineClient(std::move(base_url), std::move(api_key)) {}

Message SomaEngineClient::complete(const InferenceRequest& req) {
    RuntimeClient rc(base_url_, api_key_);
    return rc.complete(req);
}

void SomaEngineClient::stream_complete(const InferenceRequest& req,
                                       ChunkCallback chunk_cb,
                                       ErrorCallback error_cb) {
    RuntimeClient rc(base_url_, api_key_);
    rc.stream_complete(req, std::move(chunk_cb), [error_cb](const std::string& raw) {
        if (error_cb) error_cb(parse_engine_error(raw));
    });
}

bool SomaEngineClient::health_check() {
    // Not RuntimeClient::health_check(): `soma serve` answers /health with 503
    // and status:"loading" while the model is still being read, and a bare
    // status check cannot distinguish that from a dead process. The supervisor
    // needs to.
    auto cli = make_client(base_url_, api_key_);
    auto res = cli.Get("/health");
    if (!res || res->status != 200) return false;
    try {
        const auto j = nlohmann::json::parse(res->body);
        return j.value("status", "") == "ok";
    } catch (const std::exception&) {
        return false;
    }
}

bool SomaEngineClient::fetch_plan(std::string& out_json) {
    auto cli = make_client(base_url_, api_key_);
    auto res = cli.Get("/internal/plan");
    if (!res || res->status != 200) return false;
    out_json = res->body;
    return true;
}

/// The engine route this reads (`GET /internal/telemetry`) lands with the
/// telemetry gate; `soma serve` publishes /health, /v1/models, /internal/plan and
/// the chat surface today. Stated rather than left to be discovered: the call
/// simply returns without frames until then.
void SomaEngineClient::stream_telemetry(TelemetryCallback cb, std::atomic<bool>& stop_flag) {
    if (!cb) return;
    httplib::Client cli(normalize_base_url(base_url_));
    cli.set_connection_timeout(5);
    // The feed is long-lived by design and stop_flag is the termination signal,
    // so the timeout is an hour rather than a tick interval.
    //
    // It was `set_read_timeout(0, 0)`, which cpp-httplib reads as ZERO seconds
    // rather than as "no limit" — so the comment above it, "a timeout here would
    // look like the engine going quiet", described exactly what the line did.
    // Every read timed out immediately and the stream ended before its first
    // frame. Defect D10, in all three places it was written.
    cli.set_read_timeout(std::chrono::seconds{3600});
    if (!api_key_.empty()) cli.set_bearer_token_auth(api_key_);

    std::string buf;
    cli.Get("/internal/telemetry", [&](const char* data, std::size_t len) -> bool {
        if (stop_flag.load()) return false;
        buf.append(data, len);
        for (const auto& payload : util::drain_sse_lines(buf)) {
            if (payload == "[DONE]") return false;
            try {
                cb(payload);
            } catch (const std::exception& e) {
                MM_ERROR("SomaEngineClient telemetry callback threw: {}", e.what());
                return false;
            }
        }
        return !stop_flag.load();
    });
}

} // namespace mm
