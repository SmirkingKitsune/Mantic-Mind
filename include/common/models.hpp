#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <nlohmann/json.hpp>

namespace mm {

// ── Type aliases ──────────────────────────────────────────────────────────────
using AgentId  = std::string;
using ConvId   = std::string;
using MemoryId = std::string;
using NodeId   = std::string;
using SlotId   = std::string;
using JobId    = std::string;
using VoiceProfileId = std::string;
using VoiceProposalId = std::string;
using TtsCacheId = std::string;

// ── Enumerations ──────────────────────────────────────────────────────────────
enum class MessageRole { System, User, Assistant, Tool };

enum class NodeHealthStatus { Unknown, Healthy, Degraded, Unhealthy };

// Reachability is intentionally separate from health. A node that cannot be
// contacted has unknown health; it is not necessarily resource-unhealthy.
enum class NodeConnectionStatus { Unknown, Online, Unreachable, Offline };

enum class SlotState { Empty, Loading, Ready, Suspending, Suspended, Error };

// Local inference on this branch is always llama.cpp. Keep backend parsing at
// API/config boundaries so legacy vLLM agents fail explicitly instead of being
// silently reinterpreted as llama.cpp agents.
inline bool is_llama_backend(const std::string& backend) {
    return backend.empty() || backend == "llama-cpp" || backend == "llama.cpp" ||
           backend == "llama";
}

// ── ToolCall ──────────────────────────────────────────────────────────────────
struct ToolCall {
    std::string id;
    std::string function_name;
    std::string arguments_json;  // raw JSON string of function arguments
};

struct TraceEvent {
    std::string    id;
    std::string    type;
    std::string    category;
    std::string    title;
    std::string    summary;
    std::string    detail;
    std::string    content;
    std::string    message;
    int64_t        timestamp_ms = 0;
    int            sequence = 0;
    int            message_index = -1;
    std::string    source_id;
    nlohmann::json metadata = nlohmann::json::object();
};

// Durable metadata for a managed image. relative_path is control-internal and
// is deliberately omitted from JSON so API responses never expose filesystem
// layout. Attachments are scoped by the per-agent database that owns them.
struct ImageAttachment {
    std::string id;
    std::string original_filename;
    std::string mime_type;
    std::string relative_path;
    int64_t     size_bytes = 0;
    int64_t     created_at_ms = 0;
    int64_t     expires_at_ms = 0; // pending-upload expiry; 0 once referenced
};

// Ordered message content. Persisted image parts reference a managed
// attachment_id; hydrated inference-only parts additionally carry image_url as
// a data URL. Plain legacy messages continue to use Message::content alone.
struct MessageContentPart {
    std::string type = "text"; // text | image_attachment | image_url
    std::string text;
    std::string attachment_id;
    std::string image_url;
    std::string mime_type;
};

// ── Message ───────────────────────────────────────────────────────────────────
struct Message {
    MessageRole          role         = MessageRole::User;
    std::string          content;
    std::vector<ToolCall> tool_calls;
    std::string          tool_call_id;   // populated for role=Tool response messages
    std::string          thinking_text;  // content stripped from <think>…</think>
    int                  token_count  = 0;
    int64_t              timestamp_ms = 0;
    std::vector<TraceEvent> trace_events;
    std::vector<MessageContentPart> content_parts;
    int64_t              storage_id = 0; // internal SQLite row id; not serialized
};

// ── RuntimeSettings ─────────────────────────────────────────────────────────────
struct RuntimeSettings {
    int   ctx_size     = 4096;
    int   n_gpu_layers = -1;   // -1 = all layers on GPU
    int   n_threads    = -1;   // -1 = auto-detect
    int   n_threads_http = -1; // -1 = runtime default
    int   parallel     = 1;    // request slots; ctx_size is per slot
    int   batch_size   = -1;   // -1 = runtime default
    int   ubatch_size  = -1;   // -1 = runtime default
    float temperature  = 0.7f;
    float top_p        = 0.9f;
    int   top_k        = -1;   // -1 = omit and use provider/runtime default
    float min_p        = -1.0f; // -1 = omit; 0 explicitly disables min-p
    float presence_penalty = 0.0f;
    float repeat_penalty   = -1.0f; // -1 = omit; llama.cpp request field
    int   max_tokens   = 1024;
    bool  flash_attn   = true;
    std::vector<std::string> extra_args;  // additional runtime CLI flags
};

// OpenAI-compatible remote API settings for agents whose inference_backend is
// "api". model_path remains the served model id sent in the request body.
struct ApiSettings {
    std::string base_url = "https://api.openai.com";
    std::string chat_completions_path = "/v1/chat/completions";
    std::string api_key; // accepted from JSON for process-local use; never serialized
    std::string api_key_env = "OPENAI_API_KEY";
};

// Image-input capability declared by an agent profile. llama.cpp agents need
// an explicit local GGUF projector; API agents use their model-native
// multimodal implementation and therefore leave mmproj_path empty.
struct VisionSettings {
    bool        enabled = false;
    std::string mmproj_path;
};

/// True when two llama.cpp configurations describe the same engine launch, i.e.
/// agents using either can share one llama-server process. Only the launch-time
/// (engine identity) fields gate sharing; per-request generation parameters
/// (sampling controls and max_tokens) ride each request and are excluded.
inline bool llama_launch_compatible(const RuntimeSettings& a, const RuntimeSettings& b) {
    return a.ctx_size        == b.ctx_size
        && a.n_gpu_layers    == b.n_gpu_layers
        && a.n_threads       == b.n_threads
        && a.n_threads_http  == b.n_threads_http
        && a.parallel        == b.parallel
        && a.batch_size      == b.batch_size
        && a.ubatch_size     == b.ubatch_size
        && a.flash_attn      == b.flash_attn
        && a.extra_args      == b.extra_args;
}

// ── AgentConfig ───────────────────────────────────────────────────────────────
struct AgentConfig {
    AgentId       id;
    std::string   name;
    std::string   model_path;
    std::string   system_prompt;
    std::string   inference_backend = "llama-cpp"; // llama-cpp (default) | api
    // Public alias advertised by the control server's OpenAI-compatible model
    // catalog. Older profiles stored this inside their backend-specific block.
    std::string   served_model_name;
    RuntimeSettings runtime_settings;
    ApiSettings   api_settings;
    VisionSettings vision_settings;
    bool          reasoning_enabled = false;
    bool          memories_enabled  = true;
    bool          tools_enabled     = false;
    NodeId        preferred_node_id;
};

struct AgentVoiceProfile {
    VoiceProfileId id;
    AgentId        agent_id;
    std::string    display_name;
    std::string    language = "Auto";
    std::string    voice_description;
    std::string    sample_text;
    std::string    rationale;
    std::string    provider = "qwen3-tts";
    std::string    voice_design_model_id;
    std::string    clone_model_id;
    std::string    reference_audio_path;
    std::string    voice_clone_prompt_path;
    VoiceProposalId approved_from_proposal_id;
    bool           active = false;
    int64_t        created_at_ms = 0;
    int64_t        updated_at_ms = 0;
};

struct VoiceDesignProposal {
    VoiceProposalId id;
    AgentId         agent_id;
    std::string     display_name;
    std::string     language = "Auto";
    std::string     voice_description;
    std::string     sample_text;
    std::string     rationale;
    std::string     status = "pending"; // pending|sampled|approved|rejected|error
    std::string     provider = "qwen3-tts";
    std::string     voice_design_model_id;
    std::string     clone_model_id;
    std::string     preview_audio_path;
    std::string     voice_clone_prompt_path;
    std::string     error;
    int64_t         created_at_ms = 0;
    int64_t         updated_at_ms = 0;
};

struct TtsSynthesisRequest {
    AgentId        agent_id;
    std::string    text;
    VoiceProfileId voice_profile_id;
    ConvId         conversation_id;
    int            message_index = -1;
    std::string    language = "Auto";
    std::string    format = "wav";
    bool           use_cache = true;
};

struct TtsSynthesisResult {
    TtsCacheId     cache_id;
    AgentId        agent_id;
    VoiceProfileId voice_profile_id;
    ConvId         conversation_id;
    int            message_index = -1;
    std::string    text_hash;
    std::string    audio_path;
    std::string    mime_type = "audio/wav";
    std::string    format = "wav";
    int            sample_rate = 0;
    int            duration_ms = 0;
    bool           cached = false;
    int64_t        created_at_ms = 0;
    int64_t        expires_at_ms = 0;
};

enum class ValidationSeverity {
    Warning,
    Error
};

inline std::string to_string(ValidationSeverity severity) {
    return severity == ValidationSeverity::Error ? "error" : "warning";
}

inline ValidationSeverity validation_severity_from_string(const std::string& s) {
    return s == "error" ? ValidationSeverity::Error : ValidationSeverity::Warning;
}

struct ModelCapabilityInfo {
    bool                     metadata_found = false;
    int64_t                  n_ctx_train = 0;
    bool                     supports_tool_calls = false;
    bool                     supports_reasoning = false;
    bool                     used_filename_heuristics = false;
    std::string              source_path;
    std::vector<std::string> warnings;
};

struct ValidationIssue {
    ValidationSeverity severity = ValidationSeverity::Error;
    std::string        field;
    std::string        message;
};

struct AgentValidationResult {
    std::vector<ValidationIssue> issues;
    std::optional<ModelCapabilityInfo> model_info;

    bool ok() const {
        for (const auto& issue : issues) {
            if (issue.severity == ValidationSeverity::Error) return false;
        }
        return true;
    }
};

// ── Conversation ──────────────────────────────────────────────────────────────
struct Conversation {
    ConvId               id;
    AgentId              agent_id;
    std::string          title;
    std::vector<Message> messages;
    int                  total_tokens       = 0;
    bool                 is_active          = true;
    std::string          compaction_summary;
    ConvId               parent_conv_id;
    int64_t              created_at_ms = 0;
    int64_t              updated_at_ms = 0;
};

// ── Memory (global) ──────────────────────────────────────────────────────────
struct Memory {
    MemoryId    id;
    AgentId     agent_id;
    std::string content;
    ConvId      source_conv_id;
    float       importance    = 0.5f;
    int64_t     created_at_ms = 0;
    int64_t     updated_at_ms = 0;
};

// ── LocalMemory (per-conversation) ───────────────────────────────────────────
struct LocalMemory {
    std::string id;
    ConvId      conversation_id;
    std::string content;
    int64_t     created_at_ms = 0;
    int64_t     updated_at_ms = 0;
};

// ── NodeHealthMetrics ─────────────────────────────────────────────────────────
struct NodeHealthMetrics {
    float   cpu_percent          = 0.0f;
    float   ram_percent          = 0.0f;
    float   gpu_percent          = 0.0f;
    int64_t gpu_vram_used_mb     = 0;
    int64_t gpu_vram_total_mb    = 0;
    int64_t ram_used_mb          = 0;
    int64_t ram_total_mb         = 0;
    int64_t disk_free_mb         = 0;
    bool    gpu_backend_available = false;
};

// ── SlotInfo ─────────────────────────────────────────────────────────────────
struct SlotInfo {
    SlotId      id;
    uint16_t    port            = 0;
    std::string model_path;
    std::string mmproj_path;             // node-local; only returned by authenticated node/control status
    bool        vision_enabled  = false;
    std::string projector_identity;      // stable basename/manifest identity, never a managed attachment path
    std::string backend         = "llama-cpp"; // retained on the wire for compatibility
    AgentId     assigned_agent;          // first attached agent (legacy display)
    std::vector<AgentId> agent_ids;      // all agents attached to this slot
    SlotState   state           = SlotState::Empty;
    int64_t     vram_usage_mb   = 0;
    int64_t     last_active_ms  = 0;
    std::string kv_cache_path;
    int         effective_ctx_size = 0;
};

// ── AgentPlacement ───────────────────────────────────────────────────────────
struct AgentPlacement {
    AgentId     agent_id;
    NodeId      node_id;
    SlotId      slot_id;
    bool        suspended       = false;
    bool        is_active       = false;  // true while inference/memory-extraction is in-flight
    std::string kv_cache_node_path;
    std::string engine_fingerprint;
    int64_t     placed_at_ms    = 0;
    int64_t     last_active_ms  = 0;
};

// ── NodeCapabilities ──────────────────────────────────────────────────────────
// Runtime capabilities a llama.cpp node advertises.
struct NodeCapabilities {
    std::string              arch;            // "x86_64", "aarch64", "" = unknown
    int                      gpu_count = 0;    // GPUs visible to this node
    // RPC members must share a compatible llama.cpp build fingerprint.
    std::string              llama_cpp_version; // llama.cpp build fingerprint, "" = unknown
    bool                     supports_llama_rpc = false; // can host/join a ggml RPC group

};

// ── NodeInfo ──────────────────────────────────────────────────────────────────
// One environment check performed after a llama.cpp provisioning failure.
// `status` is pass|warn|fail|info; blocking means the normal source-build
// preflight will reject the environment until the issue is resolved.
struct LlamaDiagnosticCheck {
    std::string id;
    std::string label;
    std::string status = "info";
    std::string detected;
    std::string required;
    std::string remediation;
    bool        blocking = false;
};

// A known llama.cpp backend/precision variant assessed against this OS,
// architecture, and the assets attached to the selected upstream release.
struct LlamaRuntimeVariant {
    std::string id;                 // cuda-12|cuda-13|vulkan|...|cpu
    std::string label;
    std::string backend;            // normalized build backend
    bool        platform_supported = false;
    bool        source_supported = false;
    bool        release_available = false;
    std::string release_asset;
    std::string reason;
    bool        recommended = false;
};

// Comprehensive, serializable failure report consumed by the forced node-TUI
// troubleshooting popup and by the REST API.
struct LlamaTroubleshootingReport {
    bool        required = false;
    std::string fingerprint;
    std::string failure_stage;
    std::string failure_detail;
    std::string platform;
    std::string architecture;
    std::string target_backend;
    std::string release_tag;
    std::string summary;
    bool        can_override_checks = false;
    std::vector<LlamaDiagnosticCheck> checks;
    std::vector<LlamaRuntimeVariant> variants;
};

// Managed llama.cpp runtime status.
struct LlamaRuntimeStatus {
    std::string status = "disabled"; // resolved|provisioning|ready|failed|disabled
    std::string platform;
    std::string method;              // release|source|path
    std::string source_repo;
    std::string version;             // llama-server build number / commit
    bool        managed = false;
    std::string executable_path;
    std::string last_error;
    // Dedicated stdout/stderr/command transcript for the most recent managed
    // install or update attempt. Separate from the rotating application log.
    std::string build_log_path;
    // Accelerator-correct build this node targets: cuda|rocm|vulkan|metal|cpu.
    std::string accelerator;
    // Concrete selected engine variant when known (for example cuda-12,
    // cuda-13, vulkan, or cpu). Older markers may report only accelerator.
    std::string variant;
    // The configured/detected build the node ultimately wants. These remain
    // stable when a user selects a temporary Vulkan/CPU fallback, allowing the
    // UI to distinguish the active runtime from the intended target.
    std::string target_method;       // auto|release|source
    std::string target_accelerator;  // cuda|rocm|vulkan|metal|cpu|...
    std::string target_variant;      // concrete variant, or target_accelerator
    // Effective CMAKE_CUDA_ARCHITECTURES value for source builds. The active
    // value is persisted in the managed-runtime marker; target is derived from
    // current environment/configuration (for example 120a-real on Blackwell).
    std::string cuda_architecture;
    std::string target_cuda_architecture;
    // Computed whenever provisioner status changes. A usable fallback can be
    // ready while this is true; callers should offer, not silently force, the
    // target installation.
    bool        target_mismatch = false;
    std::string target_mismatch_reason;
    // Platform-aware engine/backend choices. Populated immediately with source
    // support and enriched with official release availability after an online
    // update/variant assessment.
    std::vector<LlamaRuntimeVariant> available_variants;
    // Newest build available upstream; "" when unknown (offline / not checked).
    std::string latest_version;
    // True when the installed runtime is older than latest_version. Orthogonal
    // to `status`: a runtime stays "ready" while advertising an update.
    bool        update_available = false;
    // Planned action for the detected update: release|compile|unavailable.
    // Empty when no update is pending or release assets could not be assessed.
    std::string update_action;
    // True only when an official asset matches this node's current accelerator.
    bool        update_release_available = false;
    // Same-platform accelerators that do have official assets (excluding the
    // current accelerator), e.g. vulkan/cpu when Linux CUDA needs compilation.
    std::vector<std::string> update_release_alternatives;
    // Human-readable consequence shown before the user approves the update.
    std::string update_warning;
    // Populated after a managed provisioning/compilation failure or an explicit
    // diagnostic refresh. Empty/default while no troubleshooting is required.
    LlamaTroubleshootingReport troubleshooting;
};

// Live progress of an in-program runtime install/upgrade.
struct RuntimeInstallProgress {
    bool        active = false;   // true while an install/upgrade is running
    int         step = 0;         // 1-based index of the current step
    int         total_steps = 0;  // number of steps in the plan
    double      fraction = -1.0;  // 0..1 within the current step; <0 = indeterminate
    std::string stage;            // human label of the current step
    std::string last_line;        // most recent streamed output line
};

// Generic long-running node-side action shown by the node TUI modal. This is
// intentionally transport-neutral: runtime provisioning and model receives both
// update the same shape.
struct NodeActionProgress {
    bool        active = false;
    std::string operation_id;
    std::string kind;             // runtime | model_receive | ...
    std::string action;           // human title, e.g. "Downloading runtime"
    std::string target;           // model id, runtime name, etc.
    std::string stage;
    std::string detail;
    int         step = 0;
    int         total_steps = 0;
    int64_t     bytes_done = 0;
    int64_t     bytes_total = 0;
    double      fraction = -1.0;  // 0..1; <0 = indeterminate
    bool        cancelable = true;
    bool        cancel_requested = false;
    std::string last_error;
};

struct NodeInfo {
    NodeId           id;
    std::string      hostname;
    std::string      url;
    std::string      api_key;
    std::string      loaded_model;  // deprecated: kept for backwards compat
    NodeHealthStatus health    = NodeHealthStatus::Unknown;
    NodeConnectionStatus connection_status = NodeConnectionStatus::Unknown;
    bool             connected = false;
    bool             remembered = false;
    std::string      platform;
    NodeCapabilities capabilities;
    NodeHealthMetrics metrics;
    int64_t          last_seen_ms = 0;
    int64_t          slot_snapshot_at_ms = 0; // last successfully parsed /api/node/status slots
    int64_t          unreachable_since_ms = 0;
    int64_t          metrics_sampled_at_ms = 0;
    int              consecutive_failures = 0;

    // Multi-slot fields
    std::vector<SlotInfo>    slots;
    int64_t                  disk_free_mb = 0;
    int                      max_slots    = 1;
    int                      slot_in_use = 0;
    int                      slot_available = 0;
    int                      slot_ready = 0;
    int                      slot_loading = 0;
    int                      slot_suspending = 0;
    int                      slot_suspended = 0;
    int                      slot_error = 0;
    std::string              llama_server_path;
    LlamaRuntimeStatus       llama_runtime;
    NodeActionProgress       action_progress;
};

// ── ToolDefinition ────────────────────────────────────────────────────────────
struct ToolDefinition {
    std::string    name;
    std::string    description;
    nlohmann::json parameters_schema = nlohmann::json::object();
};

// ── InferenceRequest ──────────────────────────────────────────────────────────
struct InferenceRequest {
    std::string          model;
    std::vector<Message> messages;
    RuntimeSettings        settings;
    bool                 stream = true;
    std::vector<ToolDefinition> tools;
};

// ── InferenceChunk (SSE payload) ──────────────────────────────────────────────
struct InferenceChunk {
    std::string              delta_content;
    std::optional<ToolCall>  tool_call_delta;
    std::string              thinking_delta;
    std::string              tool_result_json;  // raw JSON for tool_result SSE events
    bool                     is_done      = false;
    int                      tokens_used  = 0;
    std::string              finish_reason;
    ConvId                   conv_id;     // only on is_done == true
};

// ── JSON helpers ──────────────────────────────────────────────────────────────
inline std::string to_string(MessageRole role) {
    switch (role) {
        case MessageRole::System:    return "system";
        case MessageRole::Assistant: return "assistant";
        case MessageRole::Tool:      return "tool";
        default:                     return "user";
    }
}

inline MessageRole message_role_from_string(const std::string& s) {
    if (s == "system")    return MessageRole::System;
    if (s == "assistant") return MessageRole::Assistant;
    if (s == "tool")      return MessageRole::Tool;
    return MessageRole::User;
}

inline std::string to_string(NodeHealthStatus h) {
    switch (h) {
        case NodeHealthStatus::Healthy:   return "healthy";
        case NodeHealthStatus::Degraded:  return "degraded";
        case NodeHealthStatus::Unhealthy: return "unhealthy";
        default:                          return "unknown";
    }
}

inline NodeHealthStatus node_health_from_string(const std::string& s) {
    if (s == "healthy")   return NodeHealthStatus::Healthy;
    if (s == "degraded")  return NodeHealthStatus::Degraded;
    if (s == "unhealthy") return NodeHealthStatus::Unhealthy;
    return NodeHealthStatus::Unknown;
}

inline std::string to_string(NodeConnectionStatus s) {
    switch (s) {
        case NodeConnectionStatus::Online:      return "online";
        case NodeConnectionStatus::Unreachable: return "unreachable";
        case NodeConnectionStatus::Offline:     return "offline";
        default:                                return "unknown";
    }
}

inline NodeConnectionStatus node_connection_from_string(const std::string& s) {
    if (s == "online")      return NodeConnectionStatus::Online;
    if (s == "unreachable") return NodeConnectionStatus::Unreachable;
    if (s == "offline")     return NodeConnectionStatus::Offline;
    return NodeConnectionStatus::Unknown;
}

inline std::string to_string(SlotState s) {
    switch (s) {
        case SlotState::Loading:    return "loading";
        case SlotState::Ready:      return "ready";
        case SlotState::Suspending: return "suspending";
        case SlotState::Suspended:  return "suspended";
        case SlotState::Error:      return "error";
        default:                    return "empty";
    }
}

inline SlotState slot_state_from_string(const std::string& s) {
    if (s == "loading")    return SlotState::Loading;
    if (s == "ready")      return SlotState::Ready;
    if (s == "suspending") return SlotState::Suspending;
    if (s == "suspended")  return SlotState::Suspended;
    if (s == "error")      return SlotState::Error;
    return SlotState::Empty;
}

// ─── ToolCall ────────────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const ToolCall& t) {
    j = { {"id", t.id}, {"function_name", t.function_name},
          {"arguments_json", t.arguments_json} };
}
inline void from_json(const nlohmann::json& j, ToolCall& t) {
    j.at("id").get_to(t.id);
    j.at("function_name").get_to(t.function_name);
    j.at("arguments_json").get_to(t.arguments_json);
}

inline void to_json(nlohmann::json& j, const TraceEvent& t) {
    j = {
        {"id", t.id},
        {"type", t.type},
        {"category", t.category},
        {"title", t.title},
        {"summary", t.summary},
        {"detail", t.detail},
        {"content", t.content},
        {"message", t.message},
        {"timestamp_ms", t.timestamp_ms},
        {"sequence", t.sequence},
        {"message_index", t.message_index},
        {"source_id", t.source_id},
        {"metadata", t.metadata}
    };
}
inline void from_json(const nlohmann::json& j, TraceEvent& t) {
    if (j.contains("id"))            j.at("id").get_to(t.id);
    if (j.contains("type"))          j.at("type").get_to(t.type);
    if (j.contains("category"))      j.at("category").get_to(t.category);
    if (j.contains("title"))         j.at("title").get_to(t.title);
    if (j.contains("summary"))       j.at("summary").get_to(t.summary);
    if (j.contains("detail"))        j.at("detail").get_to(t.detail);
    if (j.contains("content"))       j.at("content").get_to(t.content);
    if (j.contains("message"))       j.at("message").get_to(t.message);
    if (j.contains("timestamp_ms"))  j.at("timestamp_ms").get_to(t.timestamp_ms);
    if (j.contains("sequence"))      j.at("sequence").get_to(t.sequence);
    if (j.contains("message_index")) j.at("message_index").get_to(t.message_index);
    if (j.contains("source_id"))     j.at("source_id").get_to(t.source_id);
    if (j.contains("metadata"))      t.metadata = j.at("metadata");
}

inline void to_json(nlohmann::json& j, const ImageAttachment& a) {
    j = { {"id", a.id},
          {"original_filename", a.original_filename},
          {"mime_type", a.mime_type},
          {"size_bytes", a.size_bytes},
          {"created_at_ms", a.created_at_ms},
          {"expires_at_ms", a.expires_at_ms} };
}
inline void from_json(const nlohmann::json& j, ImageAttachment& a) {
    if (j.contains("id"))                j.at("id").get_to(a.id);
    if (j.contains("original_filename")) j.at("original_filename").get_to(a.original_filename);
    if (j.contains("mime_type"))         j.at("mime_type").get_to(a.mime_type);
    if (j.contains("relative_path"))     j.at("relative_path").get_to(a.relative_path);
    if (j.contains("size_bytes"))        j.at("size_bytes").get_to(a.size_bytes);
    if (j.contains("created_at_ms"))     j.at("created_at_ms").get_to(a.created_at_ms);
    if (j.contains("expires_at_ms"))     j.at("expires_at_ms").get_to(a.expires_at_ms);
}

inline void to_json(nlohmann::json& j, const MessageContentPart& p) {
    j = {{"type", p.type}};
    if (!p.text.empty())          j["text"] = p.text;
    if (!p.attachment_id.empty()) j["attachment_id"] = p.attachment_id;
    if (!p.image_url.empty())     j["image_url"] = {{"url", p.image_url}};
    if (!p.mime_type.empty())     j["mime_type"] = p.mime_type;
}
inline void from_json(const nlohmann::json& j, MessageContentPart& p) {
    if (j.contains("type"))          j.at("type").get_to(p.type);
    if (j.contains("text"))          j.at("text").get_to(p.text);
    if (j.contains("attachment_id")) j.at("attachment_id").get_to(p.attachment_id);
    if (j.contains("mime_type"))     j.at("mime_type").get_to(p.mime_type);
    if (j.contains("image_url")) {
        const auto& iu = j.at("image_url");
        if (iu.is_string()) p.image_url = iu.get<std::string>();
        else if (iu.is_object() && iu.contains("url")) iu.at("url").get_to(p.image_url);
    }
}

// ─── Message ─────────────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const Message& m) {
    j = { {"role",         to_string(m.role)},
          {"content",      m.content},
          {"tool_calls",   m.tool_calls},
          {"tool_call_id", m.tool_call_id},
          {"thinking_text",m.thinking_text},
          {"token_count",  m.token_count},
          {"timestamp_ms", m.timestamp_ms},
          {"trace_events", m.trace_events} };
    if (!m.content_parts.empty()) j["content_parts"] = m.content_parts;
}
inline void from_json(const nlohmann::json& j, Message& m) {
    m.role = message_role_from_string(j.at("role").get<std::string>());
    j.at("content").get_to(m.content);
    if (j.contains("tool_calls"))   j.at("tool_calls").get_to(m.tool_calls);
    if (j.contains("tool_call_id")) j.at("tool_call_id").get_to(m.tool_call_id);
    if (j.contains("thinking_text"))j.at("thinking_text").get_to(m.thinking_text);
    if (j.contains("token_count"))  j.at("token_count").get_to(m.token_count);
    if (j.contains("timestamp_ms")) j.at("timestamp_ms").get_to(m.timestamp_ms);
    if (j.contains("trace_events")) j.at("trace_events").get_to(m.trace_events);
    if (j.contains("content_parts"))j.at("content_parts").get_to(m.content_parts);
}

// ─── RuntimeSettings ───────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const RuntimeSettings& s) {
    j = { {"ctx_size",     s.ctx_size},
          {"n_gpu_layers", s.n_gpu_layers},
          {"n_threads",    s.n_threads},
          {"n_threads_http", s.n_threads_http},
          {"parallel",     s.parallel},
          {"batch_size",   s.batch_size},
          {"ubatch_size",  s.ubatch_size},
          {"temperature",  s.temperature},
          {"top_p",        s.top_p},
          {"top_k",        s.top_k},
          {"min_p",        s.min_p},
          {"presence_penalty", s.presence_penalty},
          {"repeat_penalty", s.repeat_penalty},
          {"max_tokens",   s.max_tokens},
          {"flash_attn",   s.flash_attn},
          {"extra_args",   s.extra_args} };
}
inline void from_json(const nlohmann::json& j, RuntimeSettings& s) {
    if (j.contains("ctx_size"))     j.at("ctx_size").get_to(s.ctx_size);
    if (j.contains("n_gpu_layers")) j.at("n_gpu_layers").get_to(s.n_gpu_layers);
    if (j.contains("n_threads"))    j.at("n_threads").get_to(s.n_threads);
    if (j.contains("n_threads_http")) j.at("n_threads_http").get_to(s.n_threads_http);
    if (j.contains("parallel"))     j.at("parallel").get_to(s.parallel);
    if (j.contains("batch_size"))   j.at("batch_size").get_to(s.batch_size);
    if (j.contains("ubatch_size"))  j.at("ubatch_size").get_to(s.ubatch_size);
    if (j.contains("temperature"))  j.at("temperature").get_to(s.temperature);
    if (j.contains("top_p"))        j.at("top_p").get_to(s.top_p);
    if (j.contains("top_k"))        j.at("top_k").get_to(s.top_k);
    if (j.contains("min_p"))        j.at("min_p").get_to(s.min_p);
    if (j.contains("presence_penalty")) j.at("presence_penalty").get_to(s.presence_penalty);
    if (j.contains("repeat_penalty")) j.at("repeat_penalty").get_to(s.repeat_penalty);
    if (j.contains("max_tokens"))   j.at("max_tokens").get_to(s.max_tokens);
    if (j.contains("flash_attn"))   j.at("flash_attn").get_to(s.flash_attn);
    if (j.contains("extra_args"))   j.at("extra_args").get_to(s.extra_args);
}

inline void to_json(nlohmann::json& j, const ApiSettings& s) {
    j = { {"base_url",              s.base_url},
          {"chat_completions_path", s.chat_completions_path},
          {"api_key_env",           s.api_key_env},
          {"api_key_configured",    !s.api_key.empty()} };
}
inline void from_json(const nlohmann::json& j, ApiSettings& s) {
    if (j.contains("base_url"))              j.at("base_url").get_to(s.base_url);
    if (j.contains("chat_completions_path")) j.at("chat_completions_path").get_to(s.chat_completions_path);
    if (j.contains("api_key"))               j.at("api_key").get_to(s.api_key);
    if (j.contains("api_key_env"))           j.at("api_key_env").get_to(s.api_key_env);
}

inline void to_json(nlohmann::json& j, const VisionSettings& s) {
    j = {{"enabled", s.enabled}, {"mmproj_path", s.mmproj_path}};
}
inline void from_json(const nlohmann::json& j, VisionSettings& s) {
    if (j.contains("enabled"))     j.at("enabled").get_to(s.enabled);
    if (j.contains("mmproj_path")) j.at("mmproj_path").get_to(s.mmproj_path);
}

// ─── AgentConfig ─────────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const AgentConfig& a) {
    j = { {"id",                a.id},
          {"name",              a.name},
          {"model_path",        a.model_path},
          {"system_prompt",     a.system_prompt},
          {"inference_backend", a.inference_backend},
          {"served_model_name", a.served_model_name},
          {"runtime_settings",  a.runtime_settings},
          {"api_settings",      a.api_settings},
          {"vision_settings",   a.vision_settings},
          {"reasoning_enabled", a.reasoning_enabled},
          {"memories_enabled",  a.memories_enabled},
          {"tools_enabled",     a.tools_enabled},
          {"preferred_node_id", a.preferred_node_id} };
}
inline void from_json(const nlohmann::json& j, AgentConfig& a) {
    if (j.contains("id"))   j.at("id").get_to(a.id);
    j.at("name").get_to(a.name);
    if (j.contains("model_path"))        j.at("model_path").get_to(a.model_path);
    if (j.contains("system_prompt"))     j.at("system_prompt").get_to(a.system_prompt);
    if (j.contains("inference_backend")) j.at("inference_backend").get_to(a.inference_backend);
    if (j.contains("served_model_name")) j.at("served_model_name").get_to(a.served_model_name);
    if (j.contains("runtime_settings"))  j.at("runtime_settings").get_to(a.runtime_settings);
    // Compatibility reader for agents serialized before the runtime split.
    if (a.served_model_name.empty() && j.contains("vllm_settings") &&
        j.at("vllm_settings").is_object()) {
        a.served_model_name = j.at("vllm_settings").value("served_model_name", "");
    }
    if (j.contains("api_settings"))      j.at("api_settings").get_to(a.api_settings);
    if (j.contains("vision_settings"))   j.at("vision_settings").get_to(a.vision_settings);
    if (j.contains("reasoning_enabled")) j.at("reasoning_enabled").get_to(a.reasoning_enabled);
    if (j.contains("memories_enabled"))  j.at("memories_enabled").get_to(a.memories_enabled);
    if (j.contains("tools_enabled"))     j.at("tools_enabled").get_to(a.tools_enabled);
    if (j.contains("preferred_node_id")) j.at("preferred_node_id").get_to(a.preferred_node_id);
}

inline void to_json(nlohmann::json& j, const AgentVoiceProfile& p) {
    j = { {"id", p.id},
          {"agent_id", p.agent_id},
          {"display_name", p.display_name},
          {"language", p.language},
          {"voice_description", p.voice_description},
          {"sample_text", p.sample_text},
          {"rationale", p.rationale},
          {"provider", p.provider},
          {"voice_design_model_id", p.voice_design_model_id},
          {"clone_model_id", p.clone_model_id},
          {"reference_audio_path", p.reference_audio_path},
          {"voice_clone_prompt_path", p.voice_clone_prompt_path},
          {"approved_from_proposal_id", p.approved_from_proposal_id},
          {"active", p.active},
          {"created_at_ms", p.created_at_ms},
          {"updated_at_ms", p.updated_at_ms} };
}

inline void from_json(const nlohmann::json& j, AgentVoiceProfile& p) {
    if (j.contains("id")) j.at("id").get_to(p.id);
    if (j.contains("agent_id")) j.at("agent_id").get_to(p.agent_id);
    if (j.contains("display_name")) j.at("display_name").get_to(p.display_name);
    if (j.contains("language")) j.at("language").get_to(p.language);
    if (j.contains("voice_description")) j.at("voice_description").get_to(p.voice_description);
    if (j.contains("sample_text")) j.at("sample_text").get_to(p.sample_text);
    if (j.contains("rationale")) j.at("rationale").get_to(p.rationale);
    if (j.contains("provider")) j.at("provider").get_to(p.provider);
    if (j.contains("voice_design_model_id")) j.at("voice_design_model_id").get_to(p.voice_design_model_id);
    if (j.contains("clone_model_id")) j.at("clone_model_id").get_to(p.clone_model_id);
    if (j.contains("reference_audio_path")) j.at("reference_audio_path").get_to(p.reference_audio_path);
    if (j.contains("voice_clone_prompt_path")) j.at("voice_clone_prompt_path").get_to(p.voice_clone_prompt_path);
    if (j.contains("approved_from_proposal_id")) j.at("approved_from_proposal_id").get_to(p.approved_from_proposal_id);
    if (j.contains("active")) j.at("active").get_to(p.active);
    if (j.contains("created_at_ms")) j.at("created_at_ms").get_to(p.created_at_ms);
    if (j.contains("updated_at_ms")) j.at("updated_at_ms").get_to(p.updated_at_ms);
}

inline void to_json(nlohmann::json& j, const VoiceDesignProposal& p) {
    j = { {"id", p.id},
          {"agent_id", p.agent_id},
          {"display_name", p.display_name},
          {"language", p.language},
          {"voice_description", p.voice_description},
          {"sample_text", p.sample_text},
          {"rationale", p.rationale},
          {"status", p.status},
          {"provider", p.provider},
          {"voice_design_model_id", p.voice_design_model_id},
          {"clone_model_id", p.clone_model_id},
          {"preview_audio_path", p.preview_audio_path},
          {"voice_clone_prompt_path", p.voice_clone_prompt_path},
          {"error", p.error},
          {"created_at_ms", p.created_at_ms},
          {"updated_at_ms", p.updated_at_ms} };
}

inline void from_json(const nlohmann::json& j, VoiceDesignProposal& p) {
    if (j.contains("id")) j.at("id").get_to(p.id);
    if (j.contains("agent_id")) j.at("agent_id").get_to(p.agent_id);
    if (j.contains("display_name")) j.at("display_name").get_to(p.display_name);
    if (j.contains("language")) j.at("language").get_to(p.language);
    if (j.contains("voice_description")) j.at("voice_description").get_to(p.voice_description);
    if (j.contains("sample_text")) j.at("sample_text").get_to(p.sample_text);
    if (j.contains("rationale")) j.at("rationale").get_to(p.rationale);
    if (j.contains("status")) j.at("status").get_to(p.status);
    if (j.contains("provider")) j.at("provider").get_to(p.provider);
    if (j.contains("voice_design_model_id")) j.at("voice_design_model_id").get_to(p.voice_design_model_id);
    if (j.contains("clone_model_id")) j.at("clone_model_id").get_to(p.clone_model_id);
    if (j.contains("preview_audio_path")) j.at("preview_audio_path").get_to(p.preview_audio_path);
    if (j.contains("voice_clone_prompt_path")) j.at("voice_clone_prompt_path").get_to(p.voice_clone_prompt_path);
    if (j.contains("error")) j.at("error").get_to(p.error);
    if (j.contains("created_at_ms")) j.at("created_at_ms").get_to(p.created_at_ms);
    if (j.contains("updated_at_ms")) j.at("updated_at_ms").get_to(p.updated_at_ms);
}

inline void to_json(nlohmann::json& j, const TtsSynthesisRequest& r) {
    j = { {"agent_id", r.agent_id},
          {"text", r.text},
          {"voice_profile_id", r.voice_profile_id},
          {"conversation_id", r.conversation_id},
          {"message_index", r.message_index},
          {"language", r.language},
          {"format", r.format},
          {"use_cache", r.use_cache} };
}

inline void from_json(const nlohmann::json& j, TtsSynthesisRequest& r) {
    if (j.contains("agent_id")) j.at("agent_id").get_to(r.agent_id);
    if (j.contains("text")) j.at("text").get_to(r.text);
    if (j.contains("input")) j.at("input").get_to(r.text);
    if (j.contains("voice_profile_id")) j.at("voice_profile_id").get_to(r.voice_profile_id);
    if (j.contains("conversation_id")) j.at("conversation_id").get_to(r.conversation_id);
    if (j.contains("message_index")) j.at("message_index").get_to(r.message_index);
    if (j.contains("language")) j.at("language").get_to(r.language);
    if (j.contains("format")) j.at("format").get_to(r.format);
    if (j.contains("response_format")) j.at("response_format").get_to(r.format);
    if (j.contains("use_cache")) j.at("use_cache").get_to(r.use_cache);
}

inline void to_json(nlohmann::json& j, const TtsSynthesisResult& r) {
    j = { {"cache_id", r.cache_id},
          {"agent_id", r.agent_id},
          {"voice_profile_id", r.voice_profile_id},
          {"conversation_id", r.conversation_id},
          {"message_index", r.message_index},
          {"text_hash", r.text_hash},
          {"audio_path", r.audio_path},
          {"mime_type", r.mime_type},
          {"format", r.format},
          {"sample_rate", r.sample_rate},
          {"duration_ms", r.duration_ms},
          {"cached", r.cached},
          {"created_at_ms", r.created_at_ms},
          {"expires_at_ms", r.expires_at_ms} };
}

inline void from_json(const nlohmann::json& j, TtsSynthesisResult& r) {
    if (j.contains("cache_id")) j.at("cache_id").get_to(r.cache_id);
    if (j.contains("agent_id")) j.at("agent_id").get_to(r.agent_id);
    if (j.contains("voice_profile_id")) j.at("voice_profile_id").get_to(r.voice_profile_id);
    if (j.contains("conversation_id")) j.at("conversation_id").get_to(r.conversation_id);
    if (j.contains("message_index")) j.at("message_index").get_to(r.message_index);
    if (j.contains("text_hash")) j.at("text_hash").get_to(r.text_hash);
    if (j.contains("audio_path")) j.at("audio_path").get_to(r.audio_path);
    if (j.contains("mime_type")) j.at("mime_type").get_to(r.mime_type);
    if (j.contains("format")) j.at("format").get_to(r.format);
    if (j.contains("sample_rate")) j.at("sample_rate").get_to(r.sample_rate);
    if (j.contains("duration_ms")) j.at("duration_ms").get_to(r.duration_ms);
    if (j.contains("cached")) j.at("cached").get_to(r.cached);
    if (j.contains("created_at_ms")) j.at("created_at_ms").get_to(r.created_at_ms);
    if (j.contains("expires_at_ms")) j.at("expires_at_ms").get_to(r.expires_at_ms);
}

// ─── ValidationSeverity ─────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const ValidationSeverity& severity) {
    j = to_string(severity);
}
inline void from_json(const nlohmann::json& j, ValidationSeverity& severity) {
    severity = validation_severity_from_string(j.get<std::string>());
}

// ─── ModelCapabilityInfo ────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const ModelCapabilityInfo& info) {
    j = {
        {"metadata_found", info.metadata_found},
        {"n_ctx_train", info.n_ctx_train},
        {"supports_tool_calls", info.supports_tool_calls},
        {"supports_reasoning", info.supports_reasoning},
        {"used_filename_heuristics", info.used_filename_heuristics},
        {"source_path", info.source_path},
        {"warnings", info.warnings}
    };
}
inline void from_json(const nlohmann::json& j, ModelCapabilityInfo& info) {
    if (j.contains("metadata_found")) j.at("metadata_found").get_to(info.metadata_found);
    if (j.contains("n_ctx_train")) j.at("n_ctx_train").get_to(info.n_ctx_train);
    if (j.contains("supports_tool_calls")) j.at("supports_tool_calls").get_to(info.supports_tool_calls);
    if (j.contains("supports_reasoning")) j.at("supports_reasoning").get_to(info.supports_reasoning);
    if (j.contains("used_filename_heuristics")) j.at("used_filename_heuristics").get_to(info.used_filename_heuristics);
    if (j.contains("source_path")) j.at("source_path").get_to(info.source_path);
    if (j.contains("warnings")) j.at("warnings").get_to(info.warnings);
}

// ─── ValidationIssue ────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const ValidationIssue& issue) {
    j = {
        {"severity", issue.severity},
        {"field", issue.field},
        {"message", issue.message}
    };
}
inline void from_json(const nlohmann::json& j, ValidationIssue& issue) {
    if (j.contains("severity")) j.at("severity").get_to(issue.severity);
    if (j.contains("field")) j.at("field").get_to(issue.field);
    if (j.contains("message")) j.at("message").get_to(issue.message);
}

// ─── AgentValidationResult ──────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const AgentValidationResult& result) {
    j = {{"issues", result.issues}};
    if (result.model_info) j["model_info"] = *result.model_info;
}
inline void from_json(const nlohmann::json& j, AgentValidationResult& result) {
    if (j.contains("issues")) j.at("issues").get_to(result.issues);
    if (j.contains("model_info")) result.model_info = j.at("model_info").get<ModelCapabilityInfo>();
}

// ─── Conversation ─────────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const Conversation& c) {
    j = { {"id",                 c.id},
          {"agent_id",           c.agent_id},
          {"title",              c.title},
          {"messages",           c.messages},
          {"total_tokens",       c.total_tokens},
          {"is_active",          c.is_active},
          {"compaction_summary", c.compaction_summary},
          {"parent_conv_id",     c.parent_conv_id},
          {"created_at_ms",      c.created_at_ms},
          {"updated_at_ms",      c.updated_at_ms} };
}
inline void from_json(const nlohmann::json& j, Conversation& c) {
    j.at("id").get_to(c.id);
    if (j.contains("agent_id"))           j.at("agent_id").get_to(c.agent_id);
    if (j.contains("title"))              j.at("title").get_to(c.title);
    if (j.contains("messages"))           j.at("messages").get_to(c.messages);
    if (j.contains("total_tokens"))       j.at("total_tokens").get_to(c.total_tokens);
    if (j.contains("is_active"))          j.at("is_active").get_to(c.is_active);
    if (j.contains("compaction_summary")) j.at("compaction_summary").get_to(c.compaction_summary);
    if (j.contains("parent_conv_id"))     j.at("parent_conv_id").get_to(c.parent_conv_id);
    if (j.contains("created_at_ms"))      j.at("created_at_ms").get_to(c.created_at_ms);
    if (j.contains("updated_at_ms"))      j.at("updated_at_ms").get_to(c.updated_at_ms);
}

// ─── Memory (global) ──────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const Memory& m) {
    j = { {"id",             m.id},
          {"agent_id",       m.agent_id},
          {"content",        m.content},
          {"source_conv_id", m.source_conv_id},
          {"importance",     m.importance},
          {"created_at_ms",  m.created_at_ms},
          {"updated_at_ms",  m.updated_at_ms} };
}
inline void from_json(const nlohmann::json& j, Memory& m) {
    j.at("id").get_to(m.id);
    if (j.contains("agent_id"))       j.at("agent_id").get_to(m.agent_id);
    j.at("content").get_to(m.content);
    if (j.contains("source_conv_id")) j.at("source_conv_id").get_to(m.source_conv_id);
    if (j.contains("importance"))     j.at("importance").get_to(m.importance);
    if (j.contains("created_at_ms"))  j.at("created_at_ms").get_to(m.created_at_ms);
    if (j.contains("updated_at_ms"))  j.at("updated_at_ms").get_to(m.updated_at_ms);
}

// ─── LocalMemory ──────────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const LocalMemory& m) {
    j = { {"id",              m.id},
          {"conversation_id", m.conversation_id},
          {"content",         m.content},
          {"created_at_ms",   m.created_at_ms},
          {"updated_at_ms",   m.updated_at_ms} };
}
inline void from_json(const nlohmann::json& j, LocalMemory& m) {
    j.at("id").get_to(m.id);
    if (j.contains("conversation_id")) j.at("conversation_id").get_to(m.conversation_id);
    j.at("content").get_to(m.content);
    if (j.contains("created_at_ms"))   j.at("created_at_ms").get_to(m.created_at_ms);
    if (j.contains("updated_at_ms"))   j.at("updated_at_ms").get_to(m.updated_at_ms);
}

// ─── NodeHealthMetrics ────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const NodeHealthMetrics& h) {
    j = { {"cpu_percent",           h.cpu_percent},
          {"ram_percent",           h.ram_percent},
          {"gpu_percent",           h.gpu_percent},
          {"gpu_vram_used_mb",      h.gpu_vram_used_mb},
          {"gpu_vram_total_mb",     h.gpu_vram_total_mb},
          {"ram_used_mb",           h.ram_used_mb},
          {"ram_total_mb",          h.ram_total_mb},
          {"disk_free_mb",          h.disk_free_mb},
          {"gpu_backend_available", h.gpu_backend_available} };
}
inline void from_json(const nlohmann::json& j, NodeHealthMetrics& h) {
    if (j.contains("cpu_percent"))       j.at("cpu_percent").get_to(h.cpu_percent);
    if (j.contains("ram_percent"))       j.at("ram_percent").get_to(h.ram_percent);
    if (j.contains("gpu_percent"))       j.at("gpu_percent").get_to(h.gpu_percent);
    if (j.contains("gpu_vram_used_mb"))  j.at("gpu_vram_used_mb").get_to(h.gpu_vram_used_mb);
    if (j.contains("gpu_vram_total_mb")) j.at("gpu_vram_total_mb").get_to(h.gpu_vram_total_mb);
    if (j.contains("ram_used_mb"))       j.at("ram_used_mb").get_to(h.ram_used_mb);
    if (j.contains("ram_total_mb"))      j.at("ram_total_mb").get_to(h.ram_total_mb);
    if (j.contains("disk_free_mb"))           j.at("disk_free_mb").get_to(h.disk_free_mb);
    if (j.contains("gpu_backend_available")) j.at("gpu_backend_available").get_to(h.gpu_backend_available);
}

// ─── SlotInfo ────────────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const SlotInfo& s) {
    j = { {"id",              s.id},
          {"port",            s.port},
          {"model_path",      s.model_path},
          {"mmproj_path",     s.mmproj_path},
          {"vision_enabled",  s.vision_enabled},
          {"projector_identity", s.projector_identity},
          {"backend",         s.backend},
          {"assigned_agent",  s.assigned_agent},
          {"agent_ids",       s.agent_ids},
          {"state",           to_string(s.state)},
          {"vram_usage_mb",   s.vram_usage_mb},
          {"last_active_ms",  s.last_active_ms},
          {"kv_cache_path",   s.kv_cache_path},
          {"effective_ctx_size", s.effective_ctx_size} };
}
inline void from_json(const nlohmann::json& j, SlotInfo& s) {
    j.at("id").get_to(s.id);
    if (j.contains("port"))           j.at("port").get_to(s.port);
    if (j.contains("model_path"))     j.at("model_path").get_to(s.model_path);
    if (j.contains("mmproj_path"))    j.at("mmproj_path").get_to(s.mmproj_path);
    if (j.contains("vision_enabled")) j.at("vision_enabled").get_to(s.vision_enabled);
    if (j.contains("projector_identity")) j.at("projector_identity").get_to(s.projector_identity);
    if (j.contains("backend"))        j.at("backend").get_to(s.backend);
    if (j.contains("assigned_agent")) j.at("assigned_agent").get_to(s.assigned_agent);
    if (j.contains("agent_ids"))      j.at("agent_ids").get_to(s.agent_ids);
    // Older nodes only report assigned_agent.
    if (s.agent_ids.empty() && !s.assigned_agent.empty())
        s.agent_ids.push_back(s.assigned_agent);
    if (j.contains("state"))          s.state = slot_state_from_string(j.at("state").get<std::string>());
    if (j.contains("vram_usage_mb"))  j.at("vram_usage_mb").get_to(s.vram_usage_mb);
    if (j.contains("last_active_ms")) j.at("last_active_ms").get_to(s.last_active_ms);
    if (j.contains("kv_cache_path"))  j.at("kv_cache_path").get_to(s.kv_cache_path);
    if (j.contains("effective_ctx_size")) j.at("effective_ctx_size").get_to(s.effective_ctx_size);
}

// ─── AgentPlacement ──────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const AgentPlacement& p) {
    j = { {"agent_id",           p.agent_id},
          {"node_id",            p.node_id},
          {"slot_id",            p.slot_id},
          {"suspended",          p.suspended},
          {"is_active",          p.is_active},
          {"kv_cache_node_path", p.kv_cache_node_path},
          {"engine_fingerprint", p.engine_fingerprint},
          {"placed_at_ms",       p.placed_at_ms},
          {"last_active_ms",     p.last_active_ms} };
}
inline void from_json(const nlohmann::json& j, AgentPlacement& p) {
    j.at("agent_id").get_to(p.agent_id);
    if (j.contains("node_id"))            j.at("node_id").get_to(p.node_id);
    if (j.contains("slot_id"))            j.at("slot_id").get_to(p.slot_id);
    if (j.contains("suspended"))          j.at("suspended").get_to(p.suspended);
    if (j.contains("is_active"))          j.at("is_active").get_to(p.is_active);
    if (j.contains("kv_cache_node_path")) j.at("kv_cache_node_path").get_to(p.kv_cache_node_path);
    if (j.contains("engine_fingerprint")) j.at("engine_fingerprint").get_to(p.engine_fingerprint);
    if (j.contains("placed_at_ms"))       j.at("placed_at_ms").get_to(p.placed_at_ms);
    if (j.contains("last_active_ms"))     j.at("last_active_ms").get_to(p.last_active_ms);
}

// ─── NodeCapabilities ────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const NodeCapabilities& c) {
    j = { {"arch",          c.arch},
          {"gpu_count",     c.gpu_count},
          {"llama_cpp_version",  c.llama_cpp_version},
          {"supports_llama_rpc", c.supports_llama_rpc} };
}
inline void from_json(const nlohmann::json& j, NodeCapabilities& c) {
    if (j.contains("arch"))          j.at("arch").get_to(c.arch);
    if (j.contains("gpu_count"))     j.at("gpu_count").get_to(c.gpu_count);
    if (j.contains("llama_cpp_version"))  j.at("llama_cpp_version").get_to(c.llama_cpp_version);
    if (j.contains("supports_llama_rpc")) j.at("supports_llama_rpc").get_to(c.supports_llama_rpc);
}

// ─── NodeInfo ─────────────────────────────────────────────────────────────────
// Note: api_key is intentionally NOT serialized — NodeInfo JSON is returned to
// external clients (/v1/nodes) and must never leak node credentials. The
// persistence path (NodeRegistry::save_remembered_nodes) writes api_key
// explicitly.
inline void to_json(nlohmann::json& j, const LlamaDiagnosticCheck& c) {
    j = {{"id", c.id}, {"label", c.label}, {"status", c.status},
         {"detected", c.detected}, {"required", c.required},
         {"remediation", c.remediation}, {"blocking", c.blocking}};
}
inline void from_json(const nlohmann::json& j, LlamaDiagnosticCheck& c) {
    if (j.contains("id")) j.at("id").get_to(c.id);
    if (j.contains("label")) j.at("label").get_to(c.label);
    if (j.contains("status")) j.at("status").get_to(c.status);
    if (j.contains("detected")) j.at("detected").get_to(c.detected);
    if (j.contains("required")) j.at("required").get_to(c.required);
    if (j.contains("remediation")) j.at("remediation").get_to(c.remediation);
    if (j.contains("blocking")) j.at("blocking").get_to(c.blocking);
}

inline void to_json(nlohmann::json& j, const LlamaRuntimeVariant& v) {
    j = {{"id", v.id}, {"label", v.label}, {"backend", v.backend},
         {"platform_supported", v.platform_supported},
         {"source_supported", v.source_supported},
         {"release_available", v.release_available},
         {"release_asset", v.release_asset}, {"reason", v.reason},
         {"recommended", v.recommended}};
}
inline void from_json(const nlohmann::json& j, LlamaRuntimeVariant& v) {
    if (j.contains("id")) j.at("id").get_to(v.id);
    if (j.contains("label")) j.at("label").get_to(v.label);
    if (j.contains("backend")) j.at("backend").get_to(v.backend);
    if (j.contains("platform_supported"))
        j.at("platform_supported").get_to(v.platform_supported);
    if (j.contains("source_supported"))
        j.at("source_supported").get_to(v.source_supported);
    if (j.contains("release_available"))
        j.at("release_available").get_to(v.release_available);
    if (j.contains("release_asset")) j.at("release_asset").get_to(v.release_asset);
    if (j.contains("reason")) j.at("reason").get_to(v.reason);
    if (j.contains("recommended")) j.at("recommended").get_to(v.recommended);
}

inline void to_json(nlohmann::json& j, const LlamaTroubleshootingReport& r) {
    j = {{"required", r.required}, {"fingerprint", r.fingerprint},
         {"failure_stage", r.failure_stage}, {"failure_detail", r.failure_detail},
         {"platform", r.platform}, {"architecture", r.architecture},
         {"target_backend", r.target_backend}, {"release_tag", r.release_tag},
         {"summary", r.summary}, {"can_override_checks", r.can_override_checks},
         {"checks", r.checks}, {"variants", r.variants}};
}
inline void from_json(const nlohmann::json& j, LlamaTroubleshootingReport& r) {
    if (j.contains("required")) j.at("required").get_to(r.required);
    if (j.contains("fingerprint")) j.at("fingerprint").get_to(r.fingerprint);
    if (j.contains("failure_stage")) j.at("failure_stage").get_to(r.failure_stage);
    if (j.contains("failure_detail")) j.at("failure_detail").get_to(r.failure_detail);
    if (j.contains("platform")) j.at("platform").get_to(r.platform);
    if (j.contains("architecture")) j.at("architecture").get_to(r.architecture);
    if (j.contains("target_backend")) j.at("target_backend").get_to(r.target_backend);
    if (j.contains("release_tag")) j.at("release_tag").get_to(r.release_tag);
    if (j.contains("summary")) j.at("summary").get_to(r.summary);
    if (j.contains("can_override_checks"))
        j.at("can_override_checks").get_to(r.can_override_checks);
    if (j.contains("checks")) j.at("checks").get_to(r.checks);
    if (j.contains("variants")) j.at("variants").get_to(r.variants);
}

inline void to_json(nlohmann::json& j, const LlamaRuntimeStatus& r) {
    j = { {"status",           r.status},
          {"platform",         r.platform},
          {"method",           r.method},
          {"source_repo",      r.source_repo},
          {"version",          r.version},
          {"managed",          r.managed},
          {"executable_path",  r.executable_path},
          {"last_error",       r.last_error},
          {"build_log_path",   r.build_log_path},
          {"accelerator",      r.accelerator},
          {"variant",          r.variant},
          {"target_method",    r.target_method},
          {"target_accelerator", r.target_accelerator},
          {"target_variant",   r.target_variant},
          {"cuda_architecture", r.cuda_architecture},
          {"target_cuda_architecture", r.target_cuda_architecture},
          {"target_mismatch", r.target_mismatch},
          {"target_mismatch_reason", r.target_mismatch_reason},
          {"available_variants", r.available_variants},
          {"latest_version",   r.latest_version},
          {"update_available", r.update_available},
          {"update_action",    r.update_action},
          {"update_release_available", r.update_release_available},
          {"update_release_alternatives", r.update_release_alternatives},
          {"update_warning",   r.update_warning},
          {"troubleshooting",  r.troubleshooting} };
}
inline void from_json(const nlohmann::json& j, LlamaRuntimeStatus& r) {
    if (j.contains("status"))           j.at("status").get_to(r.status);
    if (j.contains("platform"))         j.at("platform").get_to(r.platform);
    if (j.contains("method"))           j.at("method").get_to(r.method);
    if (j.contains("source_repo"))      j.at("source_repo").get_to(r.source_repo);
    if (j.contains("version"))          j.at("version").get_to(r.version);
    if (j.contains("managed"))          j.at("managed").get_to(r.managed);
    if (j.contains("executable_path"))  j.at("executable_path").get_to(r.executable_path);
    if (j.contains("last_error"))       j.at("last_error").get_to(r.last_error);
    if (j.contains("build_log_path"))   j.at("build_log_path").get_to(r.build_log_path);
    if (j.contains("accelerator"))      j.at("accelerator").get_to(r.accelerator);
    if (j.contains("variant"))          j.at("variant").get_to(r.variant);
    if (j.contains("target_method"))
        j.at("target_method").get_to(r.target_method);
    if (j.contains("target_accelerator"))
        j.at("target_accelerator").get_to(r.target_accelerator);
    if (j.contains("target_variant"))
        j.at("target_variant").get_to(r.target_variant);
    if (j.contains("cuda_architecture"))
        j.at("cuda_architecture").get_to(r.cuda_architecture);
    if (j.contains("target_cuda_architecture"))
        j.at("target_cuda_architecture").get_to(r.target_cuda_architecture);
    if (j.contains("target_mismatch"))
        j.at("target_mismatch").get_to(r.target_mismatch);
    if (j.contains("target_mismatch_reason"))
        j.at("target_mismatch_reason").get_to(r.target_mismatch_reason);
    if (j.contains("available_variants"))
        j.at("available_variants").get_to(r.available_variants);
    if (j.contains("latest_version"))   j.at("latest_version").get_to(r.latest_version);
    if (j.contains("update_available")) j.at("update_available").get_to(r.update_available);
    if (j.contains("update_action")) j.at("update_action").get_to(r.update_action);
    if (j.contains("update_release_available"))
        j.at("update_release_available").get_to(r.update_release_available);
    if (j.contains("update_release_alternatives"))
        j.at("update_release_alternatives").get_to(r.update_release_alternatives);
    if (j.contains("update_warning")) j.at("update_warning").get_to(r.update_warning);
    if (j.contains("troubleshooting"))
        j.at("troubleshooting").get_to(r.troubleshooting);
}

inline void to_json(nlohmann::json& j, const RuntimeInstallProgress& p) {
    j = { {"active",      p.active},
          {"step",        p.step},
          {"total_steps", p.total_steps},
          {"fraction",    p.fraction},
          {"stage",       p.stage},
          {"last_line",   p.last_line} };
}
inline void from_json(const nlohmann::json& j, RuntimeInstallProgress& p) {
    if (j.contains("active"))      j.at("active").get_to(p.active);
    if (j.contains("step"))        j.at("step").get_to(p.step);
    if (j.contains("total_steps")) j.at("total_steps").get_to(p.total_steps);
    if (j.contains("fraction"))    j.at("fraction").get_to(p.fraction);
    if (j.contains("stage"))       j.at("stage").get_to(p.stage);
    if (j.contains("last_line"))   j.at("last_line").get_to(p.last_line);
}

inline void to_json(nlohmann::json& j, const NodeActionProgress& p) {
    j = { {"active",           p.active},
          {"operation_id",     p.operation_id},
          {"kind",             p.kind},
          {"action",           p.action},
          {"target",           p.target},
          {"stage",            p.stage},
          {"detail",           p.detail},
          {"step",             p.step},
          {"total_steps",      p.total_steps},
          {"bytes_done",       p.bytes_done},
          {"bytes_total",      p.bytes_total},
          {"fraction",         p.fraction},
          {"cancelable",       p.cancelable},
          {"cancel_requested", p.cancel_requested},
          {"last_error",       p.last_error} };
}
inline void from_json(const nlohmann::json& j, NodeActionProgress& p) {
    if (j.contains("active"))           j.at("active").get_to(p.active);
    if (j.contains("operation_id"))     j.at("operation_id").get_to(p.operation_id);
    if (j.contains("kind"))             j.at("kind").get_to(p.kind);
    if (j.contains("action"))           j.at("action").get_to(p.action);
    if (j.contains("target"))           j.at("target").get_to(p.target);
    if (j.contains("stage"))            j.at("stage").get_to(p.stage);
    if (j.contains("detail"))           j.at("detail").get_to(p.detail);
    if (j.contains("step"))             j.at("step").get_to(p.step);
    if (j.contains("total_steps"))      j.at("total_steps").get_to(p.total_steps);
    if (j.contains("bytes_done"))       j.at("bytes_done").get_to(p.bytes_done);
    if (j.contains("bytes_total"))      j.at("bytes_total").get_to(p.bytes_total);
    if (j.contains("fraction"))         j.at("fraction").get_to(p.fraction);
    if (j.contains("cancelable"))       j.at("cancelable").get_to(p.cancelable);
    if (j.contains("cancel_requested")) j.at("cancel_requested").get_to(p.cancel_requested);
    if (j.contains("last_error"))       j.at("last_error").get_to(p.last_error);
}

inline void to_json(nlohmann::json& j, const NodeInfo& n) {
    j = { {"id",            n.id},
          {"hostname",      n.hostname},
          {"url",           n.url},
          {"loaded_model",  n.loaded_model},
          {"health",        to_string(n.health)},
          {"connection_status", to_string(n.connection_status)},
          {"connected",     n.connected},
          {"remembered",    n.remembered},
          {"platform",      n.platform},
          {"capabilities",  n.capabilities},
          {"metrics",       n.metrics},
          {"last_seen_ms",  n.last_seen_ms},
          {"slot_snapshot_at_ms", n.slot_snapshot_at_ms},
          {"unreachable_since_ms", n.unreachable_since_ms},
          {"metrics_sampled_at_ms", n.metrics_sampled_at_ms},
          {"consecutive_failures", n.consecutive_failures},
          {"slots",         n.slots},
          {"disk_free_mb",  n.disk_free_mb},
          {"max_slots",     n.max_slots},
          {"slot_in_use",   n.slot_in_use},
          {"slot_available", n.slot_available},
          {"slot_ready",    n.slot_ready},
          {"slot_loading",  n.slot_loading},
          {"slot_suspending", n.slot_suspending},
          {"slot_suspended", n.slot_suspended},
          {"slot_error",    n.slot_error},
          {"llama_server_path",        n.llama_server_path},
          {"llama_runtime",            n.llama_runtime},
          {"action_progress",          n.action_progress} };
}
inline void from_json(const nlohmann::json& j, NodeInfo& n) {
    j.at("id").get_to(n.id);
    j.at("url").get_to(n.url);
    if (j.contains("hostname"))      j.at("hostname").get_to(n.hostname);
    if (j.contains("api_key"))       j.at("api_key").get_to(n.api_key);
    if (j.contains("loaded_model"))  j.at("loaded_model").get_to(n.loaded_model);
    if (j.contains("health"))        n.health = node_health_from_string(j.at("health").get<std::string>());
    if (j.contains("connected"))     j.at("connected").get_to(n.connected);
    if (j.contains("connection_status"))
        n.connection_status = node_connection_from_string(j.at("connection_status").get<std::string>());
    else if (n.connected)
        n.connection_status = NodeConnectionStatus::Online;
    if (j.contains("remembered"))    j.at("remembered").get_to(n.remembered);
    if (j.contains("platform"))      j.at("platform").get_to(n.platform);
    if (j.contains("capabilities"))  j.at("capabilities").get_to(n.capabilities);
    if (j.contains("metrics"))       j.at("metrics").get_to(n.metrics);
    if (j.contains("last_seen_ms"))  j.at("last_seen_ms").get_to(n.last_seen_ms);
    if (j.contains("slot_snapshot_at_ms"))
        j.at("slot_snapshot_at_ms").get_to(n.slot_snapshot_at_ms);
    if (j.contains("unreachable_since_ms")) j.at("unreachable_since_ms").get_to(n.unreachable_since_ms);
    if (j.contains("metrics_sampled_at_ms")) j.at("metrics_sampled_at_ms").get_to(n.metrics_sampled_at_ms);
    if (j.contains("consecutive_failures")) j.at("consecutive_failures").get_to(n.consecutive_failures);
    if (j.contains("slots"))         j.at("slots").get_to(n.slots);
    if (j.contains("disk_free_mb"))  j.at("disk_free_mb").get_to(n.disk_free_mb);
    if (j.contains("max_slots"))     j.at("max_slots").get_to(n.max_slots);
    if (j.contains("slot_in_use"))   j.at("slot_in_use").get_to(n.slot_in_use);
    if (j.contains("slot_available")) j.at("slot_available").get_to(n.slot_available);
    if (j.contains("slot_ready"))    j.at("slot_ready").get_to(n.slot_ready);
    if (j.contains("slot_loading"))  j.at("slot_loading").get_to(n.slot_loading);
    if (j.contains("slot_suspending")) j.at("slot_suspending").get_to(n.slot_suspending);
    if (j.contains("slot_suspended")) j.at("slot_suspended").get_to(n.slot_suspended);
    if (j.contains("slot_error"))    j.at("slot_error").get_to(n.slot_error);
    if (j.contains("llama_server_path")) j.at("llama_server_path").get_to(n.llama_server_path);
    if (j.contains("llama_runtime")) j.at("llama_runtime").get_to(n.llama_runtime);
    if (j.contains("action_progress")) j.at("action_progress").get_to(n.action_progress);
}

// ─── ToolDefinition ───────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const ToolDefinition& t) {
    j = { {"name",              t.name},
          {"description",       t.description},
          {"parameters_schema", t.parameters_schema} };
}
inline void from_json(const nlohmann::json& j, ToolDefinition& t) {
    j.at("name").get_to(t.name);
    if (j.contains("description"))       j.at("description").get_to(t.description);
    if (j.contains("parameters_schema")) j.at("parameters_schema").get_to(t.parameters_schema);
}

// ─── InferenceRequest ─────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const InferenceRequest& r) {
    j = { {"model",    r.model},
          {"messages", r.messages},
          {"settings", r.settings},
          {"stream",   r.stream},
          {"tools",    r.tools} };
}
inline void from_json(const nlohmann::json& j, InferenceRequest& r) {
    if (j.contains("model"))    j.at("model").get_to(r.model);
    if (j.contains("messages")) j.at("messages").get_to(r.messages);
    if (j.contains("settings")) j.at("settings").get_to(r.settings);
    if (j.contains("stream"))   j.at("stream").get_to(r.stream);
    if (j.contains("tools"))    j.at("tools").get_to(r.tools);
}

// ─── InferenceChunk ───────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const InferenceChunk& c) {
    j = { {"delta_content",    c.delta_content},
          {"thinking_delta",   c.thinking_delta},
          {"tool_result_json", c.tool_result_json},
          {"is_done",          c.is_done},
          {"tokens_used",      c.tokens_used},
          {"finish_reason",    c.finish_reason},
          {"conv_id",          c.conv_id} };
    if (c.tool_call_delta)
        j["tool_call_delta"] = *c.tool_call_delta;
}
inline void from_json(const nlohmann::json& j, InferenceChunk& c) {
    if (j.contains("delta_content"))    j.at("delta_content").get_to(c.delta_content);
    if (j.contains("thinking_delta"))   j.at("thinking_delta").get_to(c.thinking_delta);
    if (j.contains("tool_result_json")) j.at("tool_result_json").get_to(c.tool_result_json);
    if (j.contains("is_done"))          j.at("is_done").get_to(c.is_done);
    if (j.contains("tokens_used"))      j.at("tokens_used").get_to(c.tokens_used);
    if (j.contains("finish_reason"))    j.at("finish_reason").get_to(c.finish_reason);
    if (j.contains("conv_id"))          j.at("conv_id").get_to(c.conv_id);
    if (j.contains("tool_call_delta"))
        c.tool_call_delta = j.at("tool_call_delta").get<ToolCall>();
}

} // namespace mm
