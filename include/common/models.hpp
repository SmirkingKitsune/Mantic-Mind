#pragma once

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

enum class SlotState { Empty, Loading, Ready, Suspending, Suspended, Error };

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
};

// ── LlamaSettings ─────────────────────────────────────────────────────────────
struct LlamaSettings {
    int   ctx_size     = 4096;
    int   n_gpu_layers = -1;   // -1 = all layers on GPU
    int   n_threads    = -1;   // -1 = auto-detect
    int   n_threads_http = -1; // -1 = llama-server default
    int   parallel     = 1;    // llama-server request slots; ctx_size is per slot
    int   batch_size   = -1;   // -1 = llama-server default
    int   ubatch_size  = -1;   // -1 = llama-server default
    float temperature  = 0.7f;
    float top_p        = 0.9f;
    int   max_tokens   = 1024;
    bool  flash_attn   = true;
    std::vector<std::string> extra_args;  // additional llama-server CLI flags
};

// ── AgentConfig ───────────────────────────────────────────────────────────────
struct AgentConfig {
    AgentId       id;
    std::string   name;
    std::string   model_path;
    std::string   system_prompt;
    LlamaSettings llama_settings;
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
    AgentId     assigned_agent;
    SlotState   state           = SlotState::Empty;
    int64_t     vram_usage_mb   = 0;
    int64_t     last_active_ms  = 0;
    std::string kv_cache_path;
    int         effective_ctx_size = 0;
};

// ── StoredModel ──────────────────────────────────────────────────────────────
struct StoredModel {
    std::string model_path;
    std::string sha256;
    int64_t     size_bytes   = 0;
    int         shard_count  = 1;
};

// ── ModelTransferRequest ─────────────────────────────────────────────────────
struct ModelTransferRequest {
    std::string model_filename;
    std::string sha256;
    int64_t     total_bytes  = 0;
    int         shard_index  = 0;
    int         shard_count  = 1;
};

// ── AgentPlacement ───────────────────────────────────────────────────────────
struct AgentPlacement {
    AgentId     agent_id;
    NodeId      node_id;
    SlotId      slot_id;
    bool        suspended       = false;
    bool        is_active       = false;  // true while inference/memory-extraction is in-flight
    std::string kv_cache_node_path;
    int64_t     placed_at_ms    = 0;
    int64_t     last_active_ms  = 0;
};

// ── NodeInfo ──────────────────────────────────────────────────────────────────
struct NodeInfo {
    NodeId           id;
    std::string      url;
    std::string      api_key;
    std::string      loaded_model;  // deprecated: kept for backwards compat
    NodeHealthStatus health    = NodeHealthStatus::Unknown;
    bool             connected = false;
    bool             remembered = false;
    std::string      platform;
    NodeHealthMetrics metrics;

    // Multi-slot fields
    std::vector<SlotInfo>    slots;
    std::vector<StoredModel> stored_models;
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

    // Node-managed llama.cpp updater status
    bool                     llama_update_running = false;
    std::string              llama_update_status  = "idle"; // idle|running|succeeded|failed
    std::string              llama_update_message;
    int64_t                  llama_update_started_ms  = 0;
    int64_t                  llama_update_finished_ms = 0;

    // Runtime/version summary
    std::string              llama_install_root;
    std::string              llama_repo_dir;
    std::string              llama_build_dir;
    std::string              llama_binary_path;
    std::string              llama_installed_commit;
    std::string              llama_remote_commit;
    std::string              llama_remote_error;
    int64_t                  llama_remote_checked_ms = 0;
    bool                     llama_update_available = false;
    std::string              llama_update_reason = "unknown";
    std::string              llama_update_log_path;
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
    LlamaSettings        settings;
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
}

// ─── LlamaSettings ───────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const LlamaSettings& s) {
    j = { {"ctx_size",     s.ctx_size},
          {"n_gpu_layers", s.n_gpu_layers},
          {"n_threads",    s.n_threads},
          {"n_threads_http", s.n_threads_http},
          {"parallel",     s.parallel},
          {"batch_size",   s.batch_size},
          {"ubatch_size",  s.ubatch_size},
          {"temperature",  s.temperature},
          {"top_p",        s.top_p},
          {"max_tokens",   s.max_tokens},
          {"flash_attn",   s.flash_attn},
          {"extra_args",   s.extra_args} };
}
inline void from_json(const nlohmann::json& j, LlamaSettings& s) {
    if (j.contains("ctx_size"))     j.at("ctx_size").get_to(s.ctx_size);
    if (j.contains("n_gpu_layers")) j.at("n_gpu_layers").get_to(s.n_gpu_layers);
    if (j.contains("n_threads"))    j.at("n_threads").get_to(s.n_threads);
    if (j.contains("n_threads_http")) j.at("n_threads_http").get_to(s.n_threads_http);
    if (j.contains("parallel"))     j.at("parallel").get_to(s.parallel);
    if (j.contains("batch_size"))   j.at("batch_size").get_to(s.batch_size);
    if (j.contains("ubatch_size"))  j.at("ubatch_size").get_to(s.ubatch_size);
    if (j.contains("temperature"))  j.at("temperature").get_to(s.temperature);
    if (j.contains("top_p"))        j.at("top_p").get_to(s.top_p);
    if (j.contains("max_tokens"))   j.at("max_tokens").get_to(s.max_tokens);
    if (j.contains("flash_attn"))   j.at("flash_attn").get_to(s.flash_attn);
    if (j.contains("extra_args"))   j.at("extra_args").get_to(s.extra_args);
}

// ─── AgentConfig ─────────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const AgentConfig& a) {
    j = { {"id",                a.id},
          {"name",              a.name},
          {"model_path",        a.model_path},
          {"system_prompt",     a.system_prompt},
          {"llama_settings",    a.llama_settings},
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
    if (j.contains("llama_settings"))    j.at("llama_settings").get_to(a.llama_settings);
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
          {"assigned_agent",  s.assigned_agent},
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
    if (j.contains("assigned_agent")) j.at("assigned_agent").get_to(s.assigned_agent);
    if (j.contains("state"))          s.state = slot_state_from_string(j.at("state").get<std::string>());
    if (j.contains("vram_usage_mb"))  j.at("vram_usage_mb").get_to(s.vram_usage_mb);
    if (j.contains("last_active_ms")) j.at("last_active_ms").get_to(s.last_active_ms);
    if (j.contains("kv_cache_path"))  j.at("kv_cache_path").get_to(s.kv_cache_path);
    if (j.contains("effective_ctx_size")) j.at("effective_ctx_size").get_to(s.effective_ctx_size);
}

// ─── StoredModel ─────────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const StoredModel& m) {
    j = { {"model_path",   m.model_path},
          {"sha256",       m.sha256},
          {"size_bytes",   m.size_bytes},
          {"shard_count",  m.shard_count} };
}
inline void from_json(const nlohmann::json& j, StoredModel& m) {
    j.at("model_path").get_to(m.model_path);
    if (j.contains("sha256"))      j.at("sha256").get_to(m.sha256);
    if (j.contains("size_bytes"))  j.at("size_bytes").get_to(m.size_bytes);
    if (j.contains("shard_count")) j.at("shard_count").get_to(m.shard_count);
}

// ─── ModelTransferRequest ────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const ModelTransferRequest& r) {
    j = { {"model_filename", r.model_filename},
          {"sha256",         r.sha256},
          {"total_bytes",    r.total_bytes},
          {"shard_index",    r.shard_index},
          {"shard_count",    r.shard_count} };
}
inline void from_json(const nlohmann::json& j, ModelTransferRequest& r) {
    j.at("model_filename").get_to(r.model_filename);
    if (j.contains("sha256"))      j.at("sha256").get_to(r.sha256);
    if (j.contains("total_bytes")) j.at("total_bytes").get_to(r.total_bytes);
    if (j.contains("shard_index")) j.at("shard_index").get_to(r.shard_index);
    if (j.contains("shard_count")) j.at("shard_count").get_to(r.shard_count);
}

// ─── AgentPlacement ──────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const AgentPlacement& p) {
    j = { {"agent_id",           p.agent_id},
          {"node_id",            p.node_id},
          {"slot_id",            p.slot_id},
          {"suspended",          p.suspended},
          {"is_active",          p.is_active},
          {"kv_cache_node_path", p.kv_cache_node_path},
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
    if (j.contains("placed_at_ms"))       j.at("placed_at_ms").get_to(p.placed_at_ms);
    if (j.contains("last_active_ms"))     j.at("last_active_ms").get_to(p.last_active_ms);
}

// ─── NodeInfo ─────────────────────────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const NodeInfo& n) {
    j = { {"id",            n.id},
          {"url",           n.url},
          {"api_key",       n.api_key},
          {"loaded_model",  n.loaded_model},
          {"health",        to_string(n.health)},
          {"connected",     n.connected},
          {"remembered",    n.remembered},
          {"platform",      n.platform},
          {"metrics",       n.metrics},
          {"slots",         n.slots},
          {"stored_models", n.stored_models},
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
          {"llama_update_running",     n.llama_update_running},
          {"llama_update_status",      n.llama_update_status},
          {"llama_update_message",     n.llama_update_message},
          {"llama_update_started_ms",  n.llama_update_started_ms},
          {"llama_update_finished_ms", n.llama_update_finished_ms},
          {"llama_install_root",       n.llama_install_root},
          {"llama_repo_dir",           n.llama_repo_dir},
          {"llama_build_dir",          n.llama_build_dir},
          {"llama_binary_path",        n.llama_binary_path},
          {"llama_installed_commit",   n.llama_installed_commit},
          {"llama_remote_commit",      n.llama_remote_commit},
          {"llama_remote_error",       n.llama_remote_error},
          {"llama_remote_checked_ms",  n.llama_remote_checked_ms},
          {"llama_update_available",   n.llama_update_available},
          {"llama_update_reason",      n.llama_update_reason},
          {"llama_update_log_path",    n.llama_update_log_path} };
}
inline void from_json(const nlohmann::json& j, NodeInfo& n) {
    j.at("id").get_to(n.id);
    j.at("url").get_to(n.url);
    if (j.contains("api_key"))       j.at("api_key").get_to(n.api_key);
    if (j.contains("loaded_model"))  j.at("loaded_model").get_to(n.loaded_model);
    if (j.contains("health"))        n.health = node_health_from_string(j.at("health").get<std::string>());
    if (j.contains("connected"))     j.at("connected").get_to(n.connected);
    if (j.contains("remembered"))    j.at("remembered").get_to(n.remembered);
    if (j.contains("platform"))      j.at("platform").get_to(n.platform);
    if (j.contains("metrics"))       j.at("metrics").get_to(n.metrics);
    if (j.contains("slots"))         j.at("slots").get_to(n.slots);
    if (j.contains("stored_models")) j.at("stored_models").get_to(n.stored_models);
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
    if (j.contains("llama_update_running"))     j.at("llama_update_running").get_to(n.llama_update_running);
    if (j.contains("llama_update_status"))      j.at("llama_update_status").get_to(n.llama_update_status);
    if (j.contains("llama_update_message"))     j.at("llama_update_message").get_to(n.llama_update_message);
    if (j.contains("llama_update_started_ms"))  j.at("llama_update_started_ms").get_to(n.llama_update_started_ms);
    if (j.contains("llama_update_finished_ms")) j.at("llama_update_finished_ms").get_to(n.llama_update_finished_ms);
    if (j.contains("llama_install_root"))      j.at("llama_install_root").get_to(n.llama_install_root);
    if (j.contains("llama_repo_dir"))          j.at("llama_repo_dir").get_to(n.llama_repo_dir);
    if (j.contains("llama_build_dir"))         j.at("llama_build_dir").get_to(n.llama_build_dir);
    if (j.contains("llama_binary_path"))       j.at("llama_binary_path").get_to(n.llama_binary_path);
    if (j.contains("llama_installed_commit"))  j.at("llama_installed_commit").get_to(n.llama_installed_commit);
    if (j.contains("llama_remote_commit"))     j.at("llama_remote_commit").get_to(n.llama_remote_commit);
    if (j.contains("llama_remote_error"))      j.at("llama_remote_error").get_to(n.llama_remote_error);
    if (j.contains("llama_remote_checked_ms")) j.at("llama_remote_checked_ms").get_to(n.llama_remote_checked_ms);
    if (j.contains("llama_update_available"))  j.at("llama_update_available").get_to(n.llama_update_available);
    if (j.contains("llama_update_reason"))     j.at("llama_update_reason").get_to(n.llama_update_reason);
    if (j.contains("llama_update_log_path"))   j.at("llama_update_log_path").get_to(n.llama_update_log_path);
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
