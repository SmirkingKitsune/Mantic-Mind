#pragma once

// Soma — the serve surface. OpenAI-compatible HTTP, so the node supervises Soma
// and llama.cpp behind the SAME boundary rather than growing a parallel
// universe.
//
// Endpoints:
//   GET  /health                  readiness; what the node's health poll hits
//   GET  /v1/models
//   POST /v1/chat/completions     JSON and SSE
//
// Node-only, all under /internal:
//   GET  /internal/plan           the plan document for the loaded model
//   GET  /internal/sessions
//   POST /internal/kv/save
//   POST /internal/kv/restore
//   GET  /internal/telemetry      SSE, terse frames
//   GET  /internal/telemetry/dump
//   GET  /internal/heat           expert-access heat map
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

class MemoryHierarchy;

enum class SpeculativeMode : std::uint8_t { Off = 0, Auto, Required };

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

    /// Precision for the RESIDENT half at load: embeddings, attention
    /// projections, shared experts. Empty means whatever the container says,
    /// which is F32 — the converter stores the dense half at full precision
    /// deliberately, so this can be chosen per host without reconverting.
    ///
    /// The routed experts are NOT settable here and must not be: they are
    /// quantized on disk by the converter, and the map that describes them is read
    /// from container_meta.json. Serve used to hardcode `q4_g/q4_g/q6_g @128`
    /// instead, which meant it could only open containers matching that guess and
    /// had no way to express the quantized dense half `plan --quant-dense` was
    /// already reporting (roadmap D41).
    std::string quant_dense; // --quant-dense SOMA_QUANT_DENSE

    std::uint32_t ctx_size = 4096; // --ctx-size    SOMA_CTX_SIZE
    std::uint32_t kv_slots = 4;    // --kv-slots    SOMA_KV_SLOTS
    std::uint32_t max_batch = 0;   // --max-batch   SOMA_MAX_BATCH (0 = gate decides)
    std::uint32_t generation_timeout_seconds = 600; // --generation-timeout SOMA_GENERATION_TIMEOUT

    float top_p_expert_prune = 0.0f;                // --expert-prune SOMA_EXPERT_PRUNE
    Determinism determinism = Determinism::Batched; // --determinism SOMA_DETERMINISM

    SpeculativeMode speculative = SpeculativeMode::Auto; // --speculative SOMA_SPECULATIVE
    std::uint32_t speculative_tokens = 7;          // --speculative-tokens SOMA_SPECULATIVE_TOKENS
    float speculative_confidence_threshold = 0.0f; // --dspark-confidence-threshold

    std::uint32_t telemetry_hz = kDefaultTelemetryHz; // --telemetry-hz

    /// Serve an unstamped or legacy-v1 container.            // --allow-unstamped
    /// An escape hatch, not a default. Without the stamp, the checks that remain
    /// compare the container against the quantization map recorded in its own
    /// container_meta.json — self-consistent by construction, and blind to the IR
    /// having moved since. Run `soma stamp <dir>` instead; this exists for a
    /// container that cannot be reconverted or stamped right now.
    bool allow_unstamped = false;

    /// Serve a container with no compiled tokenizer. // --allow-byte-tokenizer
    ///
    /// Off by default, because the fallback is one token per byte folded into the
    /// vocabulary: real tokens, real weights, real logits, and text that means
    /// nothing. That is the one failure shape this engine refuses everywhere
    /// else — a container whose quantization map is wrong is refused, a shard
    /// whose bytes moved is refused — and it was the last place a wrong answer
    /// could still be returned as if it were a right one.
    ///
    /// It stays available because it is genuinely useful: it is how the engine,
    /// the scheduler and the KV path are exercised on a family whose pretokenizer
    /// is not compiled yet, which is most of them. What it must not be is the
    /// default, and it must not be quiet — `/v1/models` reports it and so does
    /// the line `soma serve` prints once it is listening.
    bool allow_byte_tokenizer = false;

    /// A measured expert-heat snapshot to warm the cache from. // --heat
    ///
    /// The return half of a loop that only ran outward. Serving MEASURES heat and
    /// publishes it — telemetry to the node, the node to control, control into
    /// `expert_heat` — and nothing ever brought it back, so every restart began
    /// cold and `--pin` reserved a budget it never filled. This is the file that
    /// closes it, in exactly the shape `ControlModelRegistry::heat()` already
    /// emits: `{"experts": [{"layer", "expert", "count", "decayed"}, ...]}`.
    ///
    /// Advisory. A snapshot for a different model, or one naming experts this
    /// container does not have, warms what it can and says what it skipped —
    /// refusing to serve because a cache hint is stale would be worse than
    /// starting cold, which is the alternative it is competing with.
    std::string heat_path; // --heat
};

/// Reasons the server refuses a request, mapped to HTTP by the implementation.
enum class ServeError : std::uint8_t {
    None = 0,
    BadRequest,         ///< 400
    NotFound,           ///< 404
    UnsupportedContent, ///< 422 — image parts; text-only v1
    CapacityPressure,   ///< 503, structured code
    ProtocolError,      ///< 502 — malformed model-specific completion protocol
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

    /// Why this server is encoding bytes rather than tokens, or empty when it is
    /// not. Non-empty only under `--allow-byte-tokenizer`.
    const std::string& byte_tokenizer_reason() const noexcept;

    /// How many experts `--heat` pinned, and how many of those are actually
    /// resident. Both zero when no snapshot was given.
    void warm_state(std::uint32_t& pinned,
                    std::uint32_t& resident,
                    double& seconds,
                    std::string& reason) const noexcept;
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
