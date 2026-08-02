#pragma once

// Soma — the serve surface. OpenAI-compatible HTTP, so the node supervises Soma
// and llama.cpp behind the SAME boundary rather than growing a parallel
// universe.
//
// Endpoints:
//   GET  /health              readiness; what the node's health poll hits
//   GET  /v1/models
//   POST /v1/chat/completions JSON and SSE
//   GET  /internal/telemetry  SSE, node-only, terse frames
//   GET  /internal/plan       the plan document for the loaded model
//
// Readiness needs no invention: RuntimeProcess::poll_health() already polls
// GET /health with early abort on child exit. There is no log sentinel anywhere
// in this codebase, so the Windows sentinel fragility the design brief warns
// about cannot recur.

#include "soma/plan.hpp"
#include "soma/scheduler.hpp"
#include "soma/telemetry.hpp"
#include "soma/types.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace soma {

class ModelState;
class MemoryHierarchy;

/// Every field is settable by BOTH a CLI flag and an env var, because the node
/// spawns this as a subprocess and argv quoting across Windows and POSIX is a
/// worse place to discover a mistake than an environment block.
struct ServeConfig {
    std::string host = "127.0.0.1"; // --host        SOMA_HOST
    std::uint16_t port = 8080;      // --port        SOMA_PORT
    std::string model_dir;          // --model-dir   SOMA_MODEL_DIR
    std::string checkpoint_dir;     // --kv-dir      SOMA_KV_DIR
    std::string served_model_name;  // --served-name SOMA_SERVED_NAME

    std::uint64_t ram_budget_bytes = 0;  // --ram-budget    SOMA_RAM_BUDGET
    std::uint64_t vram_expert_bytes = 0; // --vram-expert   SOMA_VRAM_EXPERT  (v1: 0)
    std::uint64_t pin_bytes = 0;         // --pin           SOMA_PIN

    std::uint32_t ctx_size = 4096; // --ctx-size    SOMA_CTX_SIZE
    std::uint32_t kv_slots = 4;    // --kv-slots    SOMA_KV_SLOTS
    std::uint32_t max_batch = 0;   // --max-batch   SOMA_MAX_BATCH (0 = gate decides)

    float top_p_expert_prune = 0.0f;                // --expert-prune SOMA_EXPERT_PRUNE
    Determinism determinism = Determinism::Batched; // --determinism SOMA_DETERMINISM

    std::uint32_t telemetry_hz = kDefaultTelemetryHz; // --telemetry-hz
};

/// Reasons the server refuses a request, mapped to HTTP by the implementation.
enum class ServeError : std::uint8_t {
    None = 0,
    BadRequest,         ///< 400
    NotFound,           ///< 404
    UnsupportedContent, ///< 422 — image parts; text-only v1
    CapacityPressure,   ///< 503, structured code
    Internal,           ///< 500
};

/// Structured error body.
///
/// `{"error":{"code":"capacity_pressure", ...}}`. The existing scheduler detects
/// pressure by SUBSTRING-MATCHING six English phrases against the node's error
/// body; a new engine would otherwise have to reproduce those literals verbatim
/// to earn an evict-and-retry. Both engines emit codes instead.
struct ErrorBody {
    ServeError kind = ServeError::None;
    const char* code = nullptr;
    std::string message;
};

class ServeServer {
public:
    ServeServer();
    ServeServer(const ServeServer&) = delete;
    ServeServer& operator=(const ServeServer&) = delete;
    ~ServeServer();

    Status open(const ServeConfig& config);

    /// Blocks. Health reports ready only once the model is loaded, the expert
    /// cache is warmed from the heat bootstrap, and the scheduler is accepting.
    Status listen();
    void stop();

    bool ready() const noexcept;
    const PlanDocument& plan() const noexcept;
    const ServeConfig& config() const noexcept;
    TelemetryChannel& telemetry() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char* to_string(ServeError error) noexcept;
int http_status_for(ServeError error) noexcept;

/// Parse argv and the environment into a config. CLI wins over env; env wins
/// over defaults.
Status parse_serve_config(int argc, const char* const* argv, ServeConfig& out);

} // namespace soma
