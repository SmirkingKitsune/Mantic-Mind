#include "control/control_api_server.hpp"
#include "control/agent_manager.hpp"
#include "control/agent_queue.hpp"
#include "control/node_registry.hpp"
#include "control/agent_scheduler.hpp"
#include "control/agent_config_validator.hpp"
#include "control/tts_service_client.hpp"
#include "common/agent.hpp"
#include "common/agent_db.hpp"
#include "common/http_server.hpp"
#include "common/http_client.hpp"
#include "common/memory_manager.hpp"
#include "common/conversation_manager.hpp"
#include "common/tool_executor.hpp"
#include "common/trace_provenance.hpp"
#include "common/runtime_client.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "common/sse_infer_ctx.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <string>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mm {

namespace {

std::string preview_text(const std::string& text, std::size_t max_len) {
    std::string out = util::trim(text);
    if (out.size() <= max_len) return out;
    return out.substr(0, max_len);
}

std::string first_user_preview(const Conversation& conv, std::size_t max_len) {
    for (const auto& msg : conv.messages) {
        if (msg.role == MessageRole::User && !util::trim(msg.content).empty()) {
            return preview_text(msg.content, max_len);
        }
    }
    for (const auto& msg : conv.messages) {
        if (!util::trim(msg.content).empty()) return preview_text(msg.content, max_len);
    }
    return {};
}

nlohmann::json curation_proposal(const std::string& action,
                                 const std::string& target_type,
                                 const std::string& target_id,
                                 const ConvId& conversation_id,
                                 nlohmann::json current,
                                 nlohmann::json proposed,
                                 const std::string& rationale) {
    const std::string seed = action + ":" + target_type + ":" + target_id + ":" + conversation_id;
    return nlohmann::json{
        {"id", util::generate_uuid()},
        {"action", action},
        {"target_type", target_type},
        {"target_id", target_id},
        {"conversation_id", conversation_id},
        {"current", std::move(current)},
        {"proposed", std::move(proposed)},
        {"rationale", rationale},
        {"dedupe_key", seed}
    };
}

struct CurationApplyPlan {
    std::string proposal_id;
    std::string action;
    std::string target_type;
    std::string target_id;
    ConvId      conversation_id;
    ConvId      source_conv_id;
    std::string title;
    std::string content;
    float       importance = 0.5f;
};

std::string json_string_field(const nlohmann::json& j, const char* key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return {};
    if (!it->is_string()) {
        throw std::invalid_argument(std::string(key) + " must be a string");
    }
    return it->get<std::string>();
}

std::string required_json_string_field(const nlohmann::json& j, const char* key) {
    const std::string value = json_string_field(j, key);
    if (value.empty()) {
        throw std::invalid_argument(std::string(key) + " is required");
    }
    return value;
}

float json_float_field(const nlohmann::json& j, const char* key, float fallback) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return fallback;
    if (!it->is_number()) {
        throw std::invalid_argument(std::string(key) + " must be a number");
    }
    return it->get<float>();
}

const nlohmann::json& proposal_payload(const nlohmann::json& proposal,
                                       const char* key) {
    auto it = proposal.find(key);
    if (it == proposal.end() || !it->is_object()) {
        throw std::invalid_argument(std::string(key) + " must be an object");
    }
    return *it;
}

void require_target_type(const CurationApplyPlan& plan,
                         const std::string& expected) {
    if (plan.target_type != expected) {
        throw std::invalid_argument(
            plan.action + " requires target_type '" + expected + "'");
    }
}

CurationApplyPlan validate_curation_apply_proposal(AgentDB& db,
                                                   const nlohmann::json& proposal) {
    if (!proposal.is_object()) {
        throw std::invalid_argument("proposal must be an object");
    }

    CurationApplyPlan plan;
    plan.proposal_id     = required_json_string_field(proposal, "id");
    plan.action          = required_json_string_field(proposal, "action");
    plan.target_type     = required_json_string_field(proposal, "target_type");
    plan.target_id       = json_string_field(proposal, "target_id");
    plan.conversation_id = json_string_field(proposal, "conversation_id");

    const nlohmann::json& proposed = proposal_payload(proposal, "proposed");

    if (plan.action == "rename_conversation") {
        require_target_type(plan, "conversation");
        ConvId conv_id = plan.conversation_id.empty() ? plan.target_id : plan.conversation_id;
        if (conv_id.empty()) {
            throw std::invalid_argument("conversation target is required");
        }
        if (!plan.target_id.empty() && plan.target_id != conv_id) {
            throw std::invalid_argument(
                "target_id must match conversation_id for rename_conversation");
        }
        if (!db.conversation_exists(conv_id)) {
            throw std::invalid_argument("conversation target not found");
        }
        plan.target_id = conv_id;
        plan.conversation_id = conv_id;
        plan.title = util::trim(required_json_string_field(proposed, "title"));
        if (plan.title.empty()) {
            throw std::invalid_argument("proposed.title is required");
        }
        return plan;
    }

    if (plan.action == "delete_conversation") {
        require_target_type(plan, "conversation");
        if (plan.target_id.empty()) {
            throw std::invalid_argument("target_id is required");
        }
        if (!plan.conversation_id.empty() && plan.conversation_id != plan.target_id) {
            throw std::invalid_argument(
                "conversation_id must match target_id for delete_conversation");
        }
        if (!db.conversation_exists(plan.target_id)) {
            throw std::invalid_argument("conversation target not found");
        }
        if (db.is_conversation_active(plan.target_id)) {
            throw std::invalid_argument(
                "cannot delete active conversation; activate another conversation first");
        }
        plan.conversation_id = plan.target_id;
        return plan;
    }

    if (plan.action == "create_local_memory") {
        require_target_type(plan, "local_memory");
        if (!plan.target_id.empty()) {
            throw std::invalid_argument("create_local_memory target_id must be empty");
        }
        if (plan.conversation_id.empty()) {
            throw std::invalid_argument("conversation_id is required");
        }
        if (!db.conversation_exists(plan.conversation_id)) {
            throw std::invalid_argument("conversation target not found");
        }
        plan.content = util::trim(required_json_string_field(proposed, "content"));
        if (plan.content.empty()) {
            throw std::invalid_argument("proposed.content is required");
        }
        return plan;
    }

    if (plan.action == "update_local_memory" || plan.action == "delete_local_memory") {
        require_target_type(plan, "local_memory");
        if (plan.target_id.empty()) {
            throw std::invalid_argument("target_id is required");
        }
        if (plan.conversation_id.empty()) {
            throw std::invalid_argument("conversation_id is required");
        }
        if (!db.conversation_exists(plan.conversation_id)) {
            throw std::invalid_argument("conversation target not found");
        }
        auto existing = db.get_local_memory(plan.target_id);
        if (!existing) {
            throw std::invalid_argument("local memory target not found");
        }
        if (existing->conversation_id != plan.conversation_id) {
            throw std::invalid_argument(
                "local memory target does not belong to proposal conversation");
        }
        if (plan.action == "update_local_memory") {
            plan.content = util::trim(required_json_string_field(proposed, "content"));
            if (plan.content.empty()) {
                throw std::invalid_argument("proposed.content is required");
            }
        }
        return plan;
    }

    if (plan.action == "create_global_memory") {
        require_target_type(plan, "global_memory");
        if (!plan.target_id.empty()) {
            throw std::invalid_argument("create_global_memory target_id must be empty");
        }
        plan.source_conv_id = json_string_field(proposed, "source_conv_id");
        if (plan.source_conv_id.empty()) plan.source_conv_id = plan.conversation_id;
        if (!plan.source_conv_id.empty() && !db.conversation_exists(plan.source_conv_id)) {
            throw std::invalid_argument("source conversation target not found");
        }
        plan.content = util::trim(required_json_string_field(proposed, "content"));
        if (plan.content.empty()) {
            throw std::invalid_argument("proposed.content is required");
        }
        plan.importance = json_float_field(proposed, "importance", 0.5f);
        return plan;
    }

    if (plan.action == "update_global_memory" || plan.action == "delete_global_memory") {
        require_target_type(plan, "global_memory");
        if (plan.target_id.empty()) {
            throw std::invalid_argument("target_id is required");
        }
        auto existing = db.get_memory(plan.target_id);
        if (!existing) {
            throw std::invalid_argument("global memory target not found");
        }
        if (!plan.conversation_id.empty() && existing->source_conv_id != plan.conversation_id) {
            throw std::invalid_argument(
                "global memory target source conversation does not match proposal");
        }
        plan.source_conv_id = existing->source_conv_id;
        if (plan.action == "update_global_memory") {
            plan.content = proposed.contains("content")
                ? util::trim(json_string_field(proposed, "content"))
                : existing->content;
            if (plan.content.empty()) {
                throw std::invalid_argument("proposed.content is required");
            }
            plan.importance = json_float_field(proposed, "importance", existing->importance);
        }
        return plan;
    }

    throw std::invalid_argument("unsupported curation proposal action: " + plan.action);
}

nlohmann::json apply_curation_plan(AgentDB& db, const CurationApplyPlan& plan) {
    nlohmann::json result = {
        {"proposal_id", plan.proposal_id},
        {"action", plan.action},
        {"target_type", plan.target_type},
        {"target_id", plan.target_id},
        {"status", "applied"}
    };

    if (plan.action == "rename_conversation") {
        db.rename_conversation(plan.conversation_id, plan.title);
        auto saved = db.load_conversation(plan.conversation_id);
        result["target_id"] = plan.conversation_id;
        result["conversation"] = saved ? nlohmann::json(*saved) : nlohmann::json::object();
        return result;
    }

    if (plan.action == "delete_conversation") {
        db.delete_conversation(plan.target_id);
        result["deleted"] = true;
        return result;
    }

    if (plan.action == "create_local_memory") {
        LocalMemory mem;
        mem.id = util::generate_uuid();
        mem.conversation_id = plan.conversation_id;
        mem.content = plan.content;
        db.add_local_memory(mem);
        auto saved = db.get_local_memory(mem.id);
        result["target_id"] = mem.id;
        result["local_memory"] = saved ? nlohmann::json(*saved) : nlohmann::json(mem);
        return result;
    }

    if (plan.action == "update_local_memory") {
        auto existing = db.get_local_memory(plan.target_id);
        LocalMemory mem = existing ? *existing : LocalMemory{};
        mem.id = plan.target_id;
        mem.conversation_id = plan.conversation_id;
        mem.content = plan.content;
        db.update_local_memory(mem);
        auto saved = db.get_local_memory(plan.target_id);
        result["local_memory"] = saved ? nlohmann::json(*saved) : nlohmann::json(mem);
        return result;
    }

    if (plan.action == "delete_local_memory") {
        db.delete_local_memory(plan.target_id);
        result["deleted"] = true;
        return result;
    }

    if (plan.action == "create_global_memory") {
        Memory mem;
        mem.id = util::generate_uuid();
        mem.content = plan.content;
        mem.source_conv_id = plan.source_conv_id;
        mem.importance = plan.importance;
        db.add_memory(mem);
        auto saved = db.get_memory(mem.id);
        result["target_id"] = mem.id;
        result["global_memory"] = saved ? nlohmann::json(*saved) : nlohmann::json(mem);
        return result;
    }

    if (plan.action == "update_global_memory") {
        auto existing = db.get_memory(plan.target_id);
        Memory mem = existing ? *existing : Memory{};
        mem.id = plan.target_id;
        mem.content = plan.content;
        mem.source_conv_id = plan.source_conv_id;
        mem.importance = plan.importance;
        db.update_memory(mem);
        auto saved = db.get_memory(plan.target_id);
        result["global_memory"] = saved ? nlohmann::json(*saved) : nlohmann::json(mem);
        return result;
    }

    if (plan.action == "delete_global_memory") {
        db.delete_memory(plan.target_id);
        result["deleted"] = true;
        return result;
    }

    throw std::logic_error("validated curation proposal action is not implemented");
}

void set_json_error(httplib::Response& res,
                    int status,
                    const std::string& error,
                    const std::string& detail = {}) {
    res.status = status;
    nlohmann::json body{{"error", error}};
    if (!detail.empty()) body["detail"] = detail;
    res.set_content(body.dump(), "application/json");
}

void set_openai_error(httplib::Response& res,
                      int status,
                      const std::string& message,
                      const std::string& type = "invalid_request_error") {
    res.status = status;
    res.set_content(
        nlohmann::json{
            {"error", {
                {"message", message},
                {"type", type},
                {"param", nullptr},
                {"code", nullptr}
            }}
        }.dump(),
        "application/json");
}

nlohmann::json parse_request_json(const httplib::Request& req) {
    if (util::trim(req.body).empty()) return nlohmann::json::object();
    auto j = nlohmann::json::parse(req.body);
    if (!j.is_object()) {
        throw std::invalid_argument("request body must be a JSON object");
    }
    return j;
}

std::string extract_json_object_text(const std::string& text) {
    std::string trimmed = util::trim(text);
    if (trimmed.rfind("```", 0) == 0) {
        const auto first_newline = trimmed.find('\n');
        const auto last_fence = trimmed.rfind("```");
        if (first_newline != std::string::npos &&
            last_fence != std::string::npos &&
            last_fence > first_newline) {
            trimmed = util::trim(trimmed.substr(first_newline + 1,
                                               last_fence - first_newline - 1));
        }
    }

    const auto first = trimmed.find('{');
    const auto last = trimmed.rfind('}');
    if (first == std::string::npos || last == std::string::npos || last <= first) {
        throw std::invalid_argument("model did not return a JSON object");
    }
    return trimmed.substr(first, last - first + 1);
}

std::string openai_agent_model_id(const AgentConfig& cfg) {
    return "agent:" + cfg.id;
}

std::string agent_backend(const AgentConfig& cfg) {
    std::string backend = util::to_lower(util::trim(cfg.inference_backend));
    if (backend.empty()) return "llama-cpp";  // unspecified => default runtime
    if (backend == "llama.cpp" || backend == "llama" || backend == "llama-cpp")
        return "llama-cpp";
    return backend;
}

bool agent_uses_api_backend(const AgentConfig& cfg) {
    return agent_backend(cfg) == "api";
}

std::string resolve_api_key(const ApiSettings& settings) {
    if (!settings.api_key.empty()) return settings.api_key;
    const std::string env_name = util::trim(settings.api_key_env);
    if (env_name.empty()) return {};
    const char* value = std::getenv(env_name.c_str());
    return value ? std::string(value) : std::string{};
}

std::unique_ptr<RuntimeClient> make_api_runtime_client(const AgentConfig& cfg) {
    return std::make_unique<RuntimeClient>(
        cfg.api_settings.base_url,
        resolve_api_key(cfg.api_settings),
        cfg.api_settings.chat_completions_path);
}

nlohmann::json openai_model_entry(const AgentConfig& cfg) {
    return nlohmann::json{
        {"id", openai_agent_model_id(cfg)},
        {"object", "model"},
        {"created", 0},
        {"owned_by", "mantic-mind"},
        {"metadata", {
            {"agent_id", cfg.id},
            {"agent_name", cfg.name},
            {"inference_backend", agent_backend(cfg)},
            {"model_path", cfg.model_path},
            {"vision_enabled", cfg.vision_settings.enabled},
            {"served_model_name", cfg.served_model_name},
            {"api_base_url", agent_uses_api_backend(cfg) ? cfg.api_settings.base_url : ""}
        }}
    };
}

nlohmann::json mantic_model_entry(const AgentConfig& cfg) {
    return nlohmann::json{
        {"id", openai_agent_model_id(cfg)},
        {"agent_id", cfg.id},
        {"agent_name", cfg.name},
        {"inference_backend", agent_backend(cfg)},
        {"model_path", cfg.model_path},
        {"vision_enabled", cfg.vision_settings.enabled},
        {"served_model_name", cfg.served_model_name},
        {"api_base_url", agent_uses_api_backend(cfg) ? cfg.api_settings.base_url : ""}
    };
}

nlohmann::json openai_model_list(const std::vector<AgentConfig>& configs) {
    nlohmann::json data = nlohmann::json::array();
    for (const auto& cfg : configs) data.push_back(openai_model_entry(cfg));
    return nlohmann::json{{"object", "list"}, {"data", data}};
}

nlohmann::json mantic_model_list(const std::vector<AgentConfig>& configs) {
    nlohmann::json data = nlohmann::json::array();
    for (const auto& cfg : configs) data.push_back(mantic_model_entry(cfg));
    return nlohmann::json{
        {"models", data},
        {"openai_compat_note", "Use agent:{agent_id} as the OpenAI model id."}
    };
}

std::optional<AgentConfig> resolve_openai_agent_model(AgentManager& agents,
                                                      const std::string& raw_model,
                                                      std::string* error) {
    const std::string requested = util::trim(raw_model);
    if (requested.empty()) {
        if (error) *error = "model is required";
        return std::nullopt;
    }

    const auto configs = agents.list_agents();
    const std::string prefix = "agent:";
    const bool explicit_agent_id = requested.rfind(prefix, 0) == 0;
    const std::string requested_id =
        explicit_agent_id ? requested.substr(prefix.size()) : requested;

    for (const auto& cfg : configs) {
        if (cfg.id == requested_id) return cfg;
    }
    if (explicit_agent_id) {
        if (error) *error = "model not found: " + requested;
        return std::nullopt;
    }

    std::vector<AgentConfig> matches;
    auto add_match = [&matches](const AgentConfig& cfg) {
        for (const auto& existing : matches) {
            if (existing.id == cfg.id) return;
        }
        matches.push_back(cfg);
    };

    for (const auto& cfg : configs) {
        if (cfg.name == requested) add_match(cfg);
        if (cfg.model_path == requested) add_match(cfg);
        if (!cfg.served_model_name.empty() &&
            cfg.served_model_name == requested) {
            add_match(cfg);
        }
    }

    if (matches.size() == 1) return matches.front();
    if (matches.empty()) {
        if (error) *error = "model not found: " + requested + "; use agent:{agent_id}";
    } else if (error) {
        *error = "model matches multiple agents; use agent:{agent_id}";
    }
    return std::nullopt;
}

nlohmann::json openai_stream_chunk(const std::string& id,
                                   int64_t created,
                                   const std::string& model,
                                   nlohmann::json delta,
                                   nlohmann::json finish_reason) {
    return nlohmann::json{
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", nlohmann::json::array({
            {
                {"index", 0},
                {"delta", std::move(delta)},
                {"finish_reason", std::move(finish_reason)}
            }
        })}
    };
}

nlohmann::json openai_chat_completion_response(const std::string& id,
                                               int64_t created,
                                               const std::string& model,
                                               const ConvId& conv_id,
                                               const std::vector<InferenceChunk>& chunks) {
    std::string content;
    int completion_tokens = 0;
    for (const auto& chunk : chunks) {
        content += chunk.delta_content;
        completion_tokens = std::max(completion_tokens, chunk.tokens_used);
    }

    return nlohmann::json{
        {"id", id},
        {"object", "chat.completion"},
        {"created", created},
        {"model", model},
        {"choices", nlohmann::json::array({
            {
                {"index", 0},
                {"message", {
                    {"role", "assistant"},
                    {"content", content}
                }},
                {"finish_reason", "stop"}
            }
        })},
        {"usage", {
            {"prompt_tokens", 0},
            {"completion_tokens", completion_tokens},
            {"total_tokens", completion_tokens}
        }},
        {"mantic_conversation_id", conv_id}
    };
}

std::string sha256_hex(const std::string& text) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) throw std::runtime_error("failed to create SHA-256 context");
    const bool ok =
        EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1 &&
        EVP_DigestUpdate(ctx, text.data(), text.size()) == 1 &&
        EVP_DigestFinal_ex(ctx, digest, &digest_len) == 1;
    EVP_MD_CTX_free(ctx);
    if (!ok) throw std::runtime_error("failed to compute SHA-256");

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_len; ++i) {
        out << std::setw(2) << static_cast<int>(digest[i]);
    }
    return out.str();
}

bool is_supported_tts_format(const std::string& format) {
    const std::string f = util::to_lower(util::trim(format));
    return f.empty() || f == "wav";
}

std::string tts_mime_type_for_format(const std::string& format) {
    const std::string f = util::to_lower(util::trim(format));
    if (f == "wav" || f.empty()) return "audio/wav";
    return "application/octet-stream";
}

std::filesystem::path agent_tts_dir(const std::string& data_dir,
                                    const AgentId& agent_id,
                                    const std::string& category) {
    return std::filesystem::path(data_dir) / "agents" / agent_id / "tts" / category;
}

std::filesystem::path agent_tts_cache_dir(const TtsServiceConfig& config,
                                          const std::string& data_dir,
                                          const AgentId& agent_id) {
    if (util::trim(config.cache_dir).empty()) {
        return agent_tts_dir(data_dir, agent_id, "cache");
    }
    return std::filesystem::path(config.cache_dir) / agent_id;
}

bool path_file_exists(const std::string& path) {
    std::error_code ec;
    return !path.empty() && std::filesystem::is_regular_file(path, ec);
}

void remove_file_quietly(const std::string& path) {
    if (path.empty()) return;
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

bool set_file_response(httplib::Response& res,
                       const std::string& path,
                       const std::string& mime_type) {
    auto file = std::make_shared<std::ifstream>(path, std::ios::binary);
    if (!file->is_open()) return false;
    const std::string content_type =
        mime_type.empty() ? "application/octet-stream" : mime_type;
    res.set_chunked_content_provider(
        content_type,
        [file](size_t /*offset*/, httplib::DataSink& sink) -> bool {
            char buf[64 * 1024];
            file->read(buf, static_cast<std::streamsize>(sizeof(buf)));
            const std::streamsize n = file->gcount();
            if (n > 0) return sink.write(buf, static_cast<size_t>(n));
            sink.done();
            return true;
        });
    return true;
}

constexpr int64_t kMaxImageBytes = 50LL * 1024 * 1024;
constexpr int     kMaxImagesPerContext = 8;
constexpr int64_t kMaxDecodedImageBytes = 400LL * 1024 * 1024;
constexpr int64_t kPendingAttachmentTtlMs = 24LL * 60 * 60 * 1000;
constexpr std::size_t kOpenAiJsonCeiling =
    static_cast<std::size_t>((kMaxDecodedImageBytes * 4 + 2) / 3) +
    (16ULL * 1024 * 1024);

std::string normalized_image_mime(std::string mime) {
    const auto semicolon = mime.find(';');
    if (semicolon != std::string::npos) mime.resize(semicolon);
    mime = util::to_lower(util::trim(mime));
    if (mime == "image/jpg") mime = "image/jpeg";
    return mime;
}

std::string detected_image_mime(const std::vector<unsigned char>& bytes) {
    static constexpr unsigned char png[] =
        {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
    if (bytes.size() >= sizeof(png) &&
        std::equal(std::begin(png), std::end(png), bytes.begin())) {
        return "image/png";
    }
    if (bytes.size() >= 3 && bytes[0] == 0xff && bytes[1] == 0xd8 &&
        bytes[2] == 0xff) {
        return "image/jpeg";
    }
    return {};
}

std::string safe_attachment_filename(std::string name,
                                     const std::string& mime_type) {
    name = std::filesystem::path(name).filename().string();
    name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char ch) {
        return ch < 0x20 || ch == 0x7f;
    }), name.end());
    if (name.empty()) name = mime_type == "image/png" ? "image.png" : "image.jpg";
    if (name.size() > 255) name.resize(255);
    return name;
}

std::string base64_encode(const std::vector<unsigned char>& bytes) {
    if (bytes.empty()) return {};
    std::string encoded(4 * ((bytes.size() + 2) / 3), '\0');
    const int length = EVP_EncodeBlock(
        reinterpret_cast<unsigned char*>(encoded.data()),
        bytes.data(), static_cast<int>(bytes.size()));
    if (length < 0) throw std::runtime_error("base64 encoding failed");
    encoded.resize(static_cast<std::size_t>(length));
    return encoded;
}

std::vector<unsigned char> base64_decode_image(const std::string& encoded) {
    if (encoded.empty() || encoded.size() % 4 != 0) {
        throw std::invalid_argument("image data URL contains invalid base64");
    }
    const std::size_t upper_bound = (encoded.size() / 4) * 3;
    if (upper_bound > static_cast<std::size_t>(kMaxImageBytes + 2)) {
        throw std::invalid_argument("image exceeds the 50 MiB decoded limit");
    }
    std::vector<unsigned char> decoded(upper_bound);
    const int length = EVP_DecodeBlock(decoded.data(),
        reinterpret_cast<const unsigned char*>(encoded.data()),
        static_cast<int>(encoded.size()));
    if (length < 0) throw std::invalid_argument("image data URL contains invalid base64");
    std::size_t actual = static_cast<std::size_t>(length);
    if (!encoded.empty() && encoded.back() == '=') --actual;
    if (encoded.size() >= 2 && encoded[encoded.size() - 2] == '=') --actual;
    decoded.resize(actual);
    if (decoded.size() > static_cast<std::size_t>(kMaxImageBytes)) {
        throw std::invalid_argument("image exceeds the 50 MiB decoded limit");
    }
    return decoded;
}

std::vector<unsigned char> read_attachment_bytes(const std::string& path,
                                                 int64_t expected_size) {
    if (expected_size < 0 || expected_size > kMaxImageBytes) {
        throw std::runtime_error("stored attachment exceeds the image size limit");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("stored attachment file is unavailable");
    std::vector<unsigned char> bytes;
    bytes.resize(static_cast<std::size_t>(expected_size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
            throw std::runtime_error("stored attachment size does not match metadata");
        }
    }
    char extra = 0;
    if (input.read(&extra, 1)) {
        throw std::runtime_error("stored attachment size does not match metadata");
    }
    return bytes;
}

ImageAttachment store_image_bytes(AgentDB& db,
                                  const std::vector<unsigned char>& bytes,
                                  const std::string& declared_mime,
                                  const std::string& original_filename) {
    const std::string detected = detected_image_mime(bytes);
    const std::string mime = normalized_image_mime(declared_mime);
    if (mime != "image/jpeg" && mime != "image/png") {
        throw std::invalid_argument("only JPEG and PNG images are supported");
    }
    if (detected.empty() || detected != mime) {
        throw std::invalid_argument("image signature does not match the declared MIME type");
    }
    if (bytes.empty() || bytes.size() > static_cast<std::size_t>(kMaxImageBytes)) {
        throw std::invalid_argument("image must be between 1 byte and 50 MiB");
    }

    ImageAttachment attachment;
    attachment.id = util::generate_uuid();
    attachment.original_filename = safe_attachment_filename(original_filename, mime);
    attachment.mime_type = mime;
    attachment.relative_path = "attachments/" + attachment.id +
        (mime == "image/png" ? ".png" : ".jpg");
    attachment.size_bytes = static_cast<int64_t>(bytes.size());
    attachment.created_at_ms = util::now_ms();
    attachment.expires_at_ms = attachment.created_at_ms + kPendingAttachmentTtlMs;

    const std::filesystem::path destination(db.attachment_file_path(attachment));
    std::filesystem::create_directories(destination.parent_path());
    const std::filesystem::path temporary = destination.string() + ".part";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot create managed attachment file");
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            std::error_code remove_ec;
            std::filesystem::remove(temporary, remove_ec);
            throw std::runtime_error("failed to write managed attachment file");
        }
    }
    std::error_code rename_ec;
    std::filesystem::rename(temporary, destination, rename_ec);
    if (rename_ec) {
        std::error_code remove_ec;
        std::filesystem::remove(temporary, remove_ec);
        throw std::runtime_error("failed to commit managed attachment file");
    }
    try {
        db.save_attachment(attachment);
    } catch (...) {
        std::error_code remove_ec;
        std::filesystem::remove(destination, remove_ec);
        throw;
    }
    return attachment;
}

struct PreparedChatTurn {
    std::string message;
    std::vector<MessageContentPart> parts;
    int image_count = 0;
    int64_t decoded_bytes = 0;
};

PreparedChatTurn prepare_attachment_turn(AgentDB& db,
                                         const std::string& message,
                                         const std::vector<std::string>& attachment_ids) {
    PreparedChatTurn turn;
    turn.message = message;
    if (!message.empty()) {
        MessageContentPart text_part;
        text_part.type = "text";
        text_part.text = message;
        turn.parts.push_back(std::move(text_part));
    }
    if (attachment_ids.size() > static_cast<std::size_t>(kMaxImagesPerContext)) {
        throw std::invalid_argument("a turn may contain at most 8 images");
    }
    for (const auto& id : attachment_ids) {
        auto attachment = db.get_attachment(id);
        if (!attachment) throw std::invalid_argument("attachment not found for this agent: " + id);
        turn.decoded_bytes += attachment->size_bytes;
        if (turn.decoded_bytes > kMaxDecodedImageBytes) {
            throw std::invalid_argument("decoded images exceed the 400 MiB turn limit");
        }
        MessageContentPart part;
        part.type = "image_attachment";
        part.attachment_id = id;
        part.mime_type = attachment->mime_type;
        turn.parts.push_back(std::move(part));
        ++turn.image_count;
    }
    return turn;
}

void hydrate_attachment_parts(AgentDB& db,
                              std::vector<Message>& messages,
                              int* image_count,
                              int64_t* decoded_bytes) {
    int count = 0;
    int64_t total = 0;
    for (auto& message : messages) {
        for (auto& part : message.content_parts) {
            if (part.type != "image_attachment") continue;
            auto attachment = db.get_attachment(part.attachment_id);
            if (!attachment) {
                throw std::runtime_error("conversation references a missing attachment");
            }
            ++count;
            total += attachment->size_bytes;
            if (count > kMaxImagesPerContext) {
                throw std::runtime_error("hydrated context exceeds the 8-image limit");
            }
            if (total > kMaxDecodedImageBytes) {
                throw std::runtime_error("hydrated context exceeds the 400 MiB decoded-image limit");
            }
            auto bytes = read_attachment_bytes(db.attachment_file_path(*attachment),
                                               attachment->size_bytes);
            if (detected_image_mime(bytes) != attachment->mime_type) {
                throw std::runtime_error("stored attachment failed signature validation");
            }
            part.type = "image_url";
            part.image_url = "data:" + attachment->mime_type + ";base64," +
                             base64_encode(bytes);
            part.attachment_id.clear();
        }
    }
    if (image_count) *image_count = count;
    if (decoded_bytes) *decoded_bytes = total;
}

void append_ordered_text(PreparedChatTurn& turn, const std::string& text) {
    if (text.empty()) return;
    turn.message += text;
    if (!turn.parts.empty() && turn.parts.back().type == "text") {
        turn.parts.back().text += text;
    } else {
        MessageContentPart part;
        part.type = "text";
        part.text = text;
        turn.parts.push_back(std::move(part));
    }
}

PreparedChatTurn ingest_openai_messages(AgentDB& db,
                                        const nlohmann::json& messages) {
    if (!messages.is_array()) throw std::invalid_argument("messages must be an array");
    if (messages.empty()) throw std::invalid_argument("messages must not be empty");
    PreparedChatTurn turn;
    const bool simple_user = messages.size() == 1 && messages.front().is_object() &&
        util::to_lower(messages.front().value("role", "user")) == "user";

    for (const auto& message : messages) {
        if (!message.is_object()) {
            throw std::invalid_argument("each message must be an object");
        }
        std::string role = util::to_lower(util::trim(message.value("role", "user")));
        if (role.empty()) role = "user";
        if (!simple_user) append_ordered_text(turn, role + ": ");

        const auto content_it = message.find("content");
        if (content_it != message.end() && content_it->is_string()) {
            append_ordered_text(turn, content_it->get<std::string>());
        } else if (content_it != message.end() && content_it->is_array()) {
            for (const auto& item : *content_it) {
                if (!item.is_object()) {
                    throw std::invalid_argument("message content parts must be objects");
                }
                const std::string type = util::to_lower(item.value("type", std::string{}));
                if (type == "text") {
                    if (!item.contains("text") || !item.at("text").is_string()) {
                        throw std::invalid_argument("text content parts require string text");
                    }
                    append_ordered_text(turn, item.at("text").get<std::string>());
                    continue;
                }
                if (type != "image_url") {
                    throw std::invalid_argument("unsupported message content part type: " + type);
                }
                if (role != "user") {
                    throw std::invalid_argument("image_url content parts are allowed only on user messages");
                }
                if (++turn.image_count > kMaxImagesPerContext) {
                    throw std::invalid_argument("a request may contain at most 8 images");
                }
                if (!item.contains("image_url")) {
                    throw std::invalid_argument("image_url content part is missing image_url");
                }
                std::string url;
                const auto& image_url = item.at("image_url");
                if (image_url.is_string()) url = image_url.get<std::string>();
                else if (image_url.is_object() && image_url.contains("url") &&
                         image_url.at("url").is_string()) {
                    url = image_url.at("url").get<std::string>();
                } else {
                    throw std::invalid_argument("image_url must be a string or an object with a string url");
                }
                const std::string lower = util::to_lower(url);
                if (lower.rfind("http://", 0) == 0 || lower.rfind("https://", 0) == 0 ||
                    lower.rfind("file:", 0) == 0) {
                    throw std::invalid_argument("remote and file image URLs are not supported");
                }
                if (lower.rfind("data:", 0) != 0) {
                    throw std::invalid_argument("image_url must use a JPEG or PNG base64 data URL");
                }
                const auto comma = url.find(',');
                if (comma == std::string::npos) {
                    throw std::invalid_argument("malformed image data URL");
                }
                const std::string metadata = util::to_lower(url.substr(5, comma - 5));
                const auto semicolon = metadata.find(';');
                if (semicolon == std::string::npos ||
                    metadata.substr(semicolon + 1) != "base64") {
                    throw std::invalid_argument("image data URL must use base64 encoding");
                }
                const std::string mime = normalized_image_mime(metadata.substr(0, semicolon));
                auto bytes = base64_decode_image(url.substr(comma + 1));
                turn.decoded_bytes += static_cast<int64_t>(bytes.size());
                if (turn.decoded_bytes > kMaxDecodedImageBytes) {
                    throw std::invalid_argument("decoded images exceed the 400 MiB request limit");
                }
                const std::string filename = "openai-image-" +
                    std::to_string(turn.image_count) +
                    (mime == "image/png" ? ".png" : ".jpg");
                const auto attachment = store_image_bytes(db, bytes, mime, filename);
                MessageContentPart part;
                part.type = "image_attachment";
                part.attachment_id = attachment.id;
                part.mime_type = attachment.mime_type;
                turn.parts.push_back(std::move(part));
            }
        } else if (content_it != message.end() && !content_it->is_null()) {
            throw std::invalid_argument("message content must be a string or array");
        }
        if (message.contains("tool_calls") && !message.at("tool_calls").is_null()) {
            append_ordered_text(turn, "\ntool_calls: " + message.at("tool_calls").dump());
        }
        if (!simple_user) append_ordered_text(turn, "\n\n");
    }
    turn.message = util::trim(turn.message);
    if (turn.parts.empty()) {
        throw std::invalid_argument("messages must include text or image content");
    }
    return turn;
}

} // namespace

// ── Constructor / destructor ───────────────────────────────────────────────────

ControlApiServer::ControlApiServer(AgentManager& agents,
                                   AgentQueue& queue,
                                   NodeRegistry& registry,
                                   AgentScheduler& scheduler,
                                   std::string data_dir,
                                   std::string models_dir,
                                   std::string external_api_token,
                                   TtsServiceConfig tts_config,
                                   bool allow_legacy_environment)
    : agents_(agents)
    , queue_(queue)
    , registry_(registry)
    , scheduler_(scheduler)
    , data_dir_(std::move(data_dir))
    , models_dir_(std::move(models_dir))
    , external_api_token_(std::move(external_api_token))
    , tts_(std::move(tts_config))
    , allow_legacy_environment_(allow_legacy_environment)
    , server_(std::make_unique<HttpServer>())
    , openai_server_(std::make_unique<HttpServer>())
{}

ControlApiServer::~ControlApiServer() { stop(); }

bool ControlApiServer::listen(uint16_t port, const std::string& bind_host) {
    server_->set_payload_max_length(static_cast<std::size_t>(kMaxImageBytes) +
                                    (2ULL * 1024 * 1024));
    server_->SetPreRoutingHandler([this](const httplib::Request& req,
                                         httplib::Response& res) {
        return authorize_external_request(req, res);
    });
    register_routes();
    MM_INFO("ControlApiServer listening on {}:{}", bind_host, port);
    return server_->listen(bind_host, port);
}

bool ControlApiServer::listen_openai_compat(uint16_t port,
                                             const std::string& bind_host) {
    openai_server_->set_payload_max_length(kOpenAiJsonCeiling);
    openai_server_->SetPreRoutingHandler([this](const httplib::Request& req,
                                                httplib::Response& res) {
        return authorize_openai_compat_request(req, res);
    });
    register_openai_compat_routes();
    MM_INFO("OpenAI-compatible API listening on {}:{}", bind_host, port);
    return openai_server_->listen(bind_host, port);
}

void ControlApiServer::stop() {
    server_->stop();
    stop_openai_compat();
}

void ControlApiServer::stop_openai_compat() { openai_server_->stop(); }

bool ControlApiServer::is_running() const { return server_->is_running(); }

bool ControlApiServer::is_openai_compat_running() const {
    return openai_server_->is_running();
}

void ControlApiServer::cleanup_expired_tts_cache() {
    const int64_t now = util::now_ms();
    int removed = 0;
    for (const auto& cfg : agents_.list_agents()) {
        auto agent = agents_.get_agent(cfg.id);
        if (!agent) continue;
        auto expired = agent->db().delete_expired_tts_cache_entries(now);
        for (const auto& entry : expired) {
            remove_file_quietly(entry.audio_path);
            ++removed;
        }
        removed += static_cast<int>(
            agent->db().delete_expired_unreferenced_attachments(now).size());
    }
    if (removed > 0) {
        activity_log(0, "Expired managed cache entries cleaned: " + std::to_string(removed));
    }
}

void ControlApiServer::set_log_callback(LogCallback cb) { log_cb_ = std::move(cb); }

void ControlApiServer::publish_activity(int level, const std::string& message) {
    nlohmann::json entry = {
        {"timestamp_ms", util::now_ms()},
        {"level", level},
        {"message", message}
    };
    {
        std::lock_guard<std::mutex> lk(activity_mutex_);
        if (activity_entries_.size() >= kMaxActivityEntries) activity_entries_.pop_front();
        activity_entries_.push_back(entry);
    }
    if (log_cb_) log_cb_(level, message);
}

void ControlApiServer::activity_log(int level, const std::string& message) {
    publish_activity(level, message);
}

bool ControlApiServer::authorize_external_request(const httplib::Request& req,
                                                  httplib::Response& res) const {
    if (external_api_token_.empty()) return true;
    // external_api_token_ is only for public control clients. Internal node
    // routes under /api/control/* keep their separate registered-node auth.
    if (req.path.rfind("/v1/", 0) != 0 && req.path != "/v1") return true;

    static const std::string kBearer = "Bearer ";
    const std::string auth = req.get_header_value("Authorization");
    if (auth.rfind(kBearer, 0) != 0) {
        res.status = 401;
        res.set_content(R"({"error":"missing bearer token"})", "application/json");
        return false;
    }
    if (auth.substr(kBearer.size()) != external_api_token_) {
        res.status = 403;
        res.set_content(R"({"error":"invalid bearer token"})", "application/json");
        return false;
    }
    return true;
}

bool ControlApiServer::authorize_openai_compat_request(const httplib::Request& req,
                                                       httplib::Response& res) const {
    if (external_api_token_.empty()) return true;

    static const std::string kBearer = "Bearer ";
    const std::string auth = req.get_header_value("Authorization");
    if (auth.rfind(kBearer, 0) != 0) {
        set_openai_error(res, 401, "missing bearer token", "authentication_error");
        return false;
    }
    if (auth.substr(kBearer.size()) != external_api_token_) {
        set_openai_error(res, 403, "invalid bearer token", "authentication_error");
        return false;
    }
    return true;
}

void ControlApiServer::register_openai_compat_routes() {
    using namespace httplib;

    openai_server_->Get("/v1/models", [this](const Request& /*req*/, Response& res) {
        res.set_content(openai_model_list(agents_.list_agents()).dump(), "application/json");
    });

    openai_server_->Get("/v1/models/:model", [this](const Request& req, Response& res) {
        std::string error;
        auto cfg = resolve_openai_agent_model(agents_, req.path_params.at("model"), &error);
        if (!cfg) {
            set_openai_error(res, 404, error.empty() ? "model not found" : error);
            return;
        }
        res.set_content(openai_model_entry(*cfg).dump(), "application/json");
    });

    openai_server_->Post("/v1/chat/completions", [this](const Request& req, Response& res) {
        nlohmann::json body;
        std::string requested_model;
        ConvId conv_id_hint;
        int max_tokens_override = 0;
        bool stream = false;

        try {
            body = parse_request_json(req);
            requested_model = util::trim(body.value("model", std::string{}));
            if (!body.contains("messages")) {
                throw std::invalid_argument("messages is required");
            }
            conv_id_hint = util::trim(body.value("conversation_id", std::string{}));
            if (conv_id_hint.empty()) {
                conv_id_hint = util::trim(body.value("mantic_conversation_id", std::string{}));
            }
            if (body.contains("stream") && !body.at("stream").is_null()) {
                if (!body.at("stream").is_boolean()) {
                    throw std::invalid_argument("stream must be a boolean");
                }
                stream = body.at("stream").get<bool>();
            }
            for (const char* key : {"max_tokens", "max_completion_tokens"}) {
                if (!body.contains(key) || body.at(key).is_null()) continue;
                if (!body.at(key).is_number_integer()) {
                    throw std::invalid_argument(std::string(key) + " must be an integer");
                }
                max_tokens_override = std::max(0, body.at(key).get<int>());
            }
        } catch (const std::exception& e) {
            set_openai_error(res, 400, e.what());
            return;
        }

        std::string resolve_error;
        auto cfg = resolve_openai_agent_model(agents_, requested_model, &resolve_error);
        if (!cfg) {
            set_openai_error(res, 404,
                             resolve_error.empty() ? "model not found" : resolve_error);
            return;
        }

        auto agent = agents_.get_agent(cfg->id);
        if (!agent) {
            set_openai_error(res, 404, "agent not found");
            return;
        }
        PreparedChatTurn prepared;
        try {
            prepared = ingest_openai_messages(agent->db(), body.at("messages"));
        } catch (const std::exception& e) {
            set_openai_error(res, 400, e.what());
            return;
        }
        if (prepared.image_count > 0 && !cfg->vision_settings.enabled) {
            set_openai_error(res, 422, "this agent profile does not accept images",
                             "invalid_request_error");
            return;
        }

        const std::string response_id = "chatcmpl-" + util::generate_uuid();
        const int64_t created = util::now_ms() / 1000;

        if (!stream) {
            LocalChatResult result;
            auto chunk_cb = [&result](const InferenceChunk& chunk) {
                result.chunks.push_back(chunk);
            };
            auto done_cb = [&result](const ConvId& conv_id, bool success,
                                     const std::string& error) {
                result.conv_id = conv_id;
                result.success = success;
                result.error = error;
            };
            handle_chat(cfg->id, prepared.message, conv_id_hint,
                        std::move(chunk_cb), std::move(done_cb),
                        max_tokens_override, prepared.parts);
            if (!result.success) {
                set_openai_error(res,
                                 500,
                                 result.error.empty() ? "chat completion failed" : result.error,
                                 "server_error");
                return;
            }
            res.set_content(
                openai_chat_completion_response(response_id,
                                                created,
                                                requested_model,
                                                result.conv_id,
                                                result.chunks).dump(),
                "application/json");
            return;
        }

        auto ctx = std::make_shared<SseInferCtx>();
        auto push_event = [ctx](const nlohmann::json& payload) {
            if (ctx->canceled) return;
            std::lock_guard<std::mutex> lk(ctx->mx);
            ctx->lines.push_back("data: " + payload.dump() + "\n\n");
            ctx->cv.notify_one();
        };

        push_event(openai_stream_chunk(response_id,
                                       created,
                                       requested_model,
                                       nlohmann::json{{"role", "assistant"}},
                                       nullptr));

        auto chunk_fn = [push_event,
                         response_id,
                         created,
                         requested_model](const InferenceChunk& chunk) {
            if (chunk.delta_content.empty()) return;
            push_event(openai_stream_chunk(response_id,
                                           created,
                                           requested_model,
                                           nlohmann::json{{"content", chunk.delta_content}},
                                           nullptr));
        };

        auto done_sent = std::make_shared<std::atomic_bool>(false);
        auto done_fn = [ctx,
                        push_event,
                        done_sent,
                        response_id,
                        created,
                        requested_model](const ConvId& /*conv_id*/,
                                         bool success,
                                         const std::string& error) {
            if (done_sent->exchange(true)) return;
            if (success) {
                push_event(openai_stream_chunk(response_id,
                                               created,
                                               requested_model,
                                               nlohmann::json::object(),
                                               "stop"));
            } else {
                push_event(nlohmann::json{
                    {"error", {
                        {"message", error.empty() ? "chat completion failed" : error},
                        {"type", "server_error"},
                        {"param", nullptr},
                        {"code", nullptr}
                    }}
                });
            }
            std::lock_guard<std::mutex> lk(ctx->mx);
            ctx->done = true;
            ctx->cv.notify_one();
        };

        InferenceJob job;
        job.job_id = util::generate_uuid();
        job.agent_id = cfg->id;
        job.conversation_id = conv_id_hint;
        job.done_cb = [done_fn](const ConvId& conv_id, bool success) mutable {
            done_fn(conv_id, success, success ? std::string{} : "queued chat job failed");
        };
        job.process_fn = [this,
                          agent_id = cfg->id,
                          prepared,
                          conv_id_hint,
                          max_tokens_override,
                          chunk_fn,
                          ctx,
                          done_for_handle = done_fn,
                          done_for_catch = done_fn]() mutable {
            try {
                handle_chat(agent_id,
                            prepared.message,
                            conv_id_hint,
                            std::move(chunk_fn),
                            std::move(done_for_handle),
                            max_tokens_override,
                            std::move(prepared.parts),
                            [ctx] { return ctx->canceled.load(); });
            } catch (const std::exception& e) {
                MM_ERROR("OpenAI-compatible chat job for agent '{}' failed: {}",
                         agent_id,
                         e.what());
                done_for_catch(conv_id_hint, false, e.what());
            } catch (...) {
                MM_ERROR("OpenAI-compatible chat job for agent '{}' failed with unknown exception",
                         agent_id);
                done_for_catch(conv_id_hint, false, "queued chat job failed");
            }
        };
        queue_.enqueue(std::move(job));

        res.set_chunked_content_provider("text/event-stream",
            [ctx](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::unique_lock<std::mutex> lk(ctx->mx);
                ctx->cv.wait(lk, [&] {
                    return !ctx->lines.empty() || ctx->done || ctx->canceled;
                });

                if (ctx->canceled) return false;

                while (!ctx->lines.empty()) {
                    std::string payload = std::move(ctx->lines.front());
                    ctx->lines.pop_front();
                    lk.unlock();
                    if (!sink.write(payload.data(), payload.size())) {
                        ctx->canceled = true;
                        ctx->cv.notify_all();
                        return false;
                    }
                    lk.lock();
                }

                if (ctx->done && ctx->lines.empty()) {
                    lk.unlock();
                    const std::string fin = "data: [DONE]\n\n";
                    if (!sink.write(fin.data(), fin.size())) {
                        ctx->canceled = true;
                        ctx->cv.notify_all();
                        return false;
                    }
                    sink.done();
                    return true;
                }
                return true;
            },
            [ctx](bool success) {
                if (!success) {
                    ctx->canceled = true;
                    ctx->cv.notify_all();
                }
            });
    });
}

ControlApiServer::LocalChatResult ControlApiServer::chat_local(
    const AgentId& agent_id,
    const std::string& message,
    const ConvId& conv_id_hint,
    int max_tokens_override,
    const std::vector<std::string>& attachment_ids) {
    LocalChatResult result;

    std::vector<MessageContentPart> content_parts;
    if (!attachment_ids.empty()) {
        auto agent = agents_.get_agent(agent_id);
        if (!agent) {
            result.error = "agent not found";
            return result;
        }
        if (!agent->get_config().vision_settings.enabled) {
            result.error = "this agent profile does not accept images";
            return result;
        }
        try {
            content_parts = prepare_attachment_turn(agent->db(), message, attachment_ids).parts;
        } catch (const std::exception& e) {
            result.error = e.what();
            return result;
        }
    }

    auto chunk_cb = [&result](const InferenceChunk& chunk) {
        result.chunks.push_back(chunk);
    };
    auto done_cb = [&result](const ConvId& conv_id, bool success, const std::string& error) {
        result.conv_id = conv_id;
        result.success = success;
        result.error = error;
    };

    handle_chat(agent_id, message, conv_id_hint,
                std::move(chunk_cb), std::move(done_cb), max_tokens_override,
                std::move(content_parts));
    return result;
}

// ── SSE context shared between worker and provider ────────────────────────────

namespace {

// Routes LLM calls through the transport-neutral node contract. Embedded AIO
// nodes call NodeService directly; remote nodes translate through /api/node/infer.
class NodeProxyRuntimeClient : public RuntimeClient {
public:
    NodeProxyRuntimeClient(NodeOperationsPtr operations,
                           std::string slot_id = {},
                           CancelCheck cancel_requested = {})
        : RuntimeClient("http://127.0.0.1")
        , operations_(std::move(operations))
        , slot_id_(std::move(slot_id))
        , cancel_requested_(std::move(cancel_requested))
    {}

    Message complete(const InferenceRequest& req,
                     CancelCheck call_cancel_requested = {}) override {
        const auto canceled = [this, &call_cancel_requested] {
            const auto requested = [](const CancelCheck& check) {
                if (!check) return false;
                try { return check(); } catch (...) { return true; }
            };
            return requested(cancel_requested_) ||
                   requested(call_cancel_requested);
        };
        Message canceled_result;
        canceled_result.role = MessageRole::Assistant;
        canceled_result.timestamp_ms = util::now_ms();
        if (canceled()) return canceled_result;
        if (!operations_) throw std::runtime_error("node operations unavailable");

        const NodeInferRequest request{req, slot_id_, canceled};
        NodeInferResult inference = operations_->infer(request);
        if (!inference.ok()) {
            if (inference.status == NodeServiceStatus::Canceled &&
                canceled()) {
                return canceled_result;
            }
            std::string detail = util::trim(inference.error);
            if (detail.empty()) {
                detail = std::string("node inference failed: ") +
                         to_string(inference.status);
            }
            MM_WARN("NodeProxyRuntimeClient::complete: {}", detail);
            throw std::runtime_error(detail);
        }
        return inference.message;
    }

private:
    NodeOperationsPtr operations_;
    std::string slot_id_;
    CancelCheck cancel_requested_;
};

class SchedulerIdleGuard {
public:
    void arm(AgentScheduler& scheduler, AgentId agent_id) {
        scheduler_ = &scheduler;
        agent_id_ = std::move(agent_id);
    }
    ~SchedulerIdleGuard() {
        if (!scheduler_) return;
        try { scheduler_->mark_agent_idle(agent_id_); }
        catch (...) {}
    }
    SchedulerIdleGuard() = default;
    SchedulerIdleGuard(const SchedulerIdleGuard&) = delete;
    SchedulerIdleGuard& operator=(const SchedulerIdleGuard&) = delete;

private:
    AgentScheduler* scheduler_ = nullptr;
    AgentId agent_id_;
};

bool contains_prefill_thinking_incompatibility(const std::string& text) {
    const std::string lowered = util::to_lower(text);
    return lowered.find("assistant response prefill is incompatible with enable_thinking")
        != std::string::npos;
}

void enforce_prefill_safe_tail(std::vector<Message>& messages) {
    if (messages.empty()) return;
    if (messages.back().role != MessageRole::Assistant) return;

    Message m;
    m.role = MessageRole::User;
    m.content = "Continue with the next response based on the latest context.";
    m.timestamp_ms = util::now_ms();
    messages.push_back(std::move(m));
}

} // namespace

// ── handle_chat ───────────────────────────────────────────────────────────────
// Runs on the AgentQueue worker thread for the given agent.

void ControlApiServer::handle_chat(const AgentId& agent_id,
                                    const std::string& message,
                                    const ConvId& conv_id_hint,
                                    ChunkCb chunk_cb,
                                    DoneCb done_cb,
                                    int max_tokens_override,
                                    std::vector<MessageContentPart> content_parts,
                                    CancelCheck cancel_requested) {
    const auto canceled = [&] {
        return queue_.cancellation_requested() ||
               (cancel_requested && cancel_requested());
    };
    if (canceled()) {
        done_cb({}, false, "inference canceled");
        return;
    }
    // 1. Look up agent
    const int64_t request_started_ms = util::now_ms();
    auto agent = agents_.get_agent(agent_id);
    if (!agent) {
        MM_WARN("handle_chat: agent '{}' not found", agent_id);
        done_cb({}, false, "agent not found");
        return;
    }

    AgentConfig cfg = agent->get_config();
    const bool has_images = std::any_of(
        content_parts.begin(), content_parts.end(), [](const MessageContentPart& part) {
            return part.type == "image_attachment" || part.type == "image_url";
        });
    if (has_images && !cfg.vision_settings.enabled) {
        done_cb({}, false, "this agent profile does not accept images");
        return;
    }
    PerformanceSample perf;
    perf.request_id = util::generate_uuid();
    perf.agent_id = agent_id;
    perf.backend = agent_backend(cfg);
    perf.model = cfg.model_path;
    perf.vision_routing = has_images;
    if (!cfg.vision_settings.mmproj_path.empty()) {
        perf.projector_basename =
            std::filesystem::path(cfg.vision_settings.mmproj_path).filename().string();
    }
    perf.started_at_ms = request_started_ms;

    AgentDB& db = agent->db();

    const bool api_backend = agent_uses_api_backend(cfg);
    activity_log(0, "Chat started: agent='" + agent_id + "' backend='" +
                    agent_backend(cfg) + "' model='" + cfg.model_path + "'");

    // 2. Get or create active conversation
    ConvId conv_id = conv_id_hint;
    if (conv_id.empty()) {
        auto active = db.get_active_conversation_id();
        if (active) {
            conv_id = *active;
        } else {
            conv_id = db.create_conversation();
            db.set_active_conversation(conv_id);
        }
    } else if (!db.conversation_exists(conv_id)) {
        MM_WARN("handle_chat: conversation '{}' not found for agent '{}'", conv_id, agent_id);
        done_cb(conv_id, false, "conversation not found");
        return;
    }

    // 3. Append user message
    auto prior_msgs = db.load_messages(conv_id);
    int seq = static_cast<int>(prior_msgs.size());

    Message user_msg;
    user_msg.role         = MessageRole::User;
    user_msg.content      = message;
    if (content_parts.empty() && !message.empty()) {
        MessageContentPart text_part;
        text_part.type = "text";
        text_part.text = message;
        content_parts.push_back(std::move(text_part));
    }
    user_msg.content_parts = std::move(content_parts);
    user_msg.token_count = static_cast<int>((message.size() + 3) / 4);
    if (has_images) {
        user_msg.token_count += 2048 * static_cast<int>(std::count_if(
            user_msg.content_parts.begin(), user_msg.content_parts.end(),
            [](const MessageContentPart& part) {
                return part.type == "image_attachment" || part.type == "image_url";
            }));
    }
    user_msg.timestamp_ms = util::now_ms();
    db.append_message(conv_id, user_msg, seq);

    // 4. Route to either a control-local API client or a node slot.
    std::unique_ptr<RuntimeClient> api_llm_storage;
    std::unique_ptr<NodeProxyRuntimeClient> node_llm_storage;
    RuntimeClient* selected_runtime = nullptr;
    std::optional<ScheduleResult> schedule_result;
    NodeOperationsPtr selected_node_operations;
    SlotId active_slot_id;
    SchedulerIdleGuard node_idle_guard;

    if (api_backend) {
        api_llm_storage = make_api_runtime_client(cfg);
        selected_runtime = api_llm_storage.get();
        activity_log(0, "Chat routed: agent='" + agent_id + "' -> api=" +
                        cfg.api_settings.base_url);
    } else {
    // Route to a node via AgentScheduler. If capacity is temporarily unavailable,
    // wait and retry so queued prompts can run when the next node/slot opens up.
    int64_t wait_budget_ms = 180000; // default: 3 minutes
    if (allow_legacy_environment_) {
        if (const char* env_wait = std::getenv("MM_CHAT_QUEUE_WAIT_MS")) {
            try {
                int64_t parsed = std::stoll(env_wait);
                if (parsed > 0) wait_budget_ms = parsed;
            } catch (...) {}
        }
    }
    constexpr int64_t kRetrySleepMs = 1000;

    std::string sched_err;
    const int64_t wait_started_ms = util::now_ms();
    int wait_attempt = 0;
    while (true) {
        if (canceled()) {
            sched_err = "inference canceled";
            break;
        }
        schedule_result = scheduler_.ensure_agent_running(cfg);
        if (schedule_result) break;

        sched_err = scheduler_.last_error();
        std::string err_l = util::to_lower(sched_err);
        bool queueable =
            sched_err.empty() ||
            err_l.find("no capacity") != std::string::npos ||
            err_l.find("no connected node") != std::string::npos ||
            err_l.find("no node available") != std::string::npos;

        if (!queueable) break;

        int64_t waited_ms = util::now_ms() - wait_started_ms;
        if (waited_ms >= wait_budget_ms) {
            if (!sched_err.empty()) {
                sched_err += " (timed out waiting for available node)";
            } else {
                sched_err = "timed out waiting for available node";
            }
            break;
        }

        if (wait_attempt == 0 || (wait_attempt % 15) == 0) {
            MM_INFO("handle_chat: waiting for available node for agent '{}' (waited {}s / {}s)",
                    agent_id,
                    static_cast<int>(waited_ms / 1000),
                    static_cast<int>(wait_budget_ms / 1000));
            if (wait_attempt == 0)
                activity_log(1, "Chat queued: agent='" + agent_id + "' waiting for node");
        }
        ++wait_attempt;
        // Keep shutdown/client-disconnect latency bounded while a request is
        // queued for capacity instead of sleeping for the whole retry period.
        for (int slept = 0; slept < kRetrySleepMs && !canceled(); slept += 50) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    if (!schedule_result) {
        MM_WARN("handle_chat: no node available for agent '{}': {}",
                agent_id, sched_err);
        if (sched_err.empty()) {
            sched_err = "no node available or model failed to load";
        }
        activity_log(2, "Chat failed: agent='" + agent_id + "' reason='" + sched_err + "'");
        done_cb(conv_id, false, sched_err);
        return;
    }

    scheduler_.mark_agent_active(agent_id);
    node_idle_guard.arm(scheduler_, agent_id);

    try { selected_node_operations = registry_.operations(schedule_result->node_id); }
    catch (const std::exception& e) {
        MM_WARN("handle_chat: node '{}' disappeared after routing: {}",
                schedule_result->node_id, e.what());
        done_cb(conv_id, false, std::string("node lookup failed: ") + e.what());
        return;
    }

    active_slot_id = schedule_result->slot_id;

    activity_log(0, "Chat routed: agent='" + agent_id + "' -> node=" +
                 schedule_result->node_id.substr(0, 8) + " slot=" +
                 active_slot_id.substr(0, 8));

    // 5. Real LLM client routed through the typed node boundary. Remote nodes
    //    use /api/node/infer internally; the embedded node stays in-process.
    node_llm_storage = std::make_unique<NodeProxyRuntimeClient>(
        selected_node_operations, active_slot_id, canceled);
    selected_runtime = node_llm_storage.get();
    }

    RuntimeClient& real_runtime = *selected_runtime;

    // 6. Compact the conversation if it is approaching context limits.
    //    may_compact() creates a new conversation with a summary if >80% full.
    ConversationManager conv_mgr(db, real_runtime);
    const ConvId conv_id_before_compact = conv_id;
    conv_id = conv_mgr.maybe_compact(conv_id, cfg, canceled);
    if (conv_id != conv_id_before_compact) {
        // Compaction started a fresh conversation with only the kept-recent
        // messages re-appended. `seq` tracks the last message's sequence number
        // (the next append uses seq + 1), so point it at the last re-appended
        // message in the new conversation rather than the pre-compaction count.
        const int new_len = static_cast<int>(db.load_messages(conv_id).size());
        seq = new_len > 0 ? new_len - 1 : 0;
    }

    // 7. Retrieve memories and build the full context to send to the LLM.
    std::vector<Memory> memories;
    if (cfg.memories_enabled) {
        MemoryManager mem_mgr(db, real_runtime);
        memories = mem_mgr.get_relevant_memories(
            conv_id, cfg, 40, 8, canceled);
    }

    // 8. Set up tool executor for memory tools
    ToolExecutor tool_exec(db);

    // 9. Tool call loop — infer, execute tools, re-infer until done or max rounds
    static constexpr int kMaxToolRounds = 10;
    bool ok = true;
    std::string failure_reason;
    std::string final_assistant_content;
    std::string final_thinking_content;
    int         final_total_tokens = 0;
    const int64_t infer_start_ms = util::now_ms();
    bool        force_prefill_safe = false;
    perf.queue_ms = infer_start_ms - request_started_ms;
    int64_t first_token_at_ms = 0;


    for (int tool_round = 0; tool_round < kMaxToolRounds; ++tool_round) {
        // Build context fresh each round (local memories may have changed)
        std::vector<TraceEvent> context_trace_events =
            build_context_trace_events(db, conv_id, memories);
        std::vector<Message> context_msgs = conv_mgr.build_context(conv_id, cfg, memories);

        try {
            int context_images = 0;
            int64_t context_image_bytes = 0;
            hydrate_attachment_parts(db, context_msgs,
                                     &context_images, &context_image_bytes);
            if (tool_round == 0) {
                perf.image_count = context_images;
                perf.decoded_image_bytes = context_image_bytes;
                perf.vision_routing = context_images > 0;
            }
        } catch (const std::exception& e) {
            ok = false;
            failure_reason = e.what();
            break;
        }

        if (tool_round == 0) {
            for (const auto& context_message : context_msgs) {
                const auto chars = context_message.content.size() + context_message.thinking_text.size();
                perf.input_tokens += static_cast<int>((chars + 3) / 4);
            }
        }

        // Build inference request
        InferenceRequest infer_req;
        infer_req.model    = cfg.model_path;
        infer_req.messages = context_msgs;
        infer_req.settings = cfg.runtime_settings;
        infer_req.stream   = true;

        // Tools are only advertised when explicitly enabled.
        if (cfg.tools_enabled && cfg.memories_enabled) {
            infer_req.tools = ToolExecutor::local_tool_catalog();
        }

        // Per-request max_tokens override
        if (max_tokens_override != 0) {
            infer_req.settings.max_tokens = max_tokens_override;
        }

        // Stream typed chunks from either the direct embedded service or the
        // remote HTTP adapter. Wire-format parsing stays inside the adapter.
        std::string assistant_content;
        std::string thinking_content;
        int         total_tokens  = 0;
        std::string finish_reason;
        bool        infer_done_seen = false;
        bool        stream_had_error = false;
        std::string stream_error_message;
        NodeServiceStatus infer_node_status = NodeServiceStatus::Failed;
        std::string infer_error_detail;
        std::vector<ToolCall> accumulated_tool_calls;

        auto node_operations = selected_node_operations;

        auto reset_infer_state = [&]() {
            assistant_content.clear();
            thinking_content.clear();
            total_tokens = 0;
            finish_reason.clear();
            infer_done_seen = false;
            stream_had_error = false;
            stream_error_message.clear();
            infer_node_status = NodeServiceStatus::Failed;
            infer_error_detail.clear();
            accumulated_tool_calls.clear();
        };

        auto run_infer_once = [&](bool stream_mode) -> bool {
            InferenceRequest req_to_send = infer_req;
            req_to_send.stream = stream_mode;
            if (force_prefill_safe) {
                enforce_prefill_safe_tail(req_to_send.messages);
            }
            if (api_backend) {
                if (stream_mode) {
                    bool api_ok = true;
                    real_runtime.stream_complete(
                        req_to_send,
                        [&](const InferenceChunk& chunk) {
                            if (!chunk.is_done && first_token_at_ms == 0 &&
                                (!chunk.delta_content.empty() || !chunk.thinking_delta.empty() || chunk.tool_call_delta)) {
                                first_token_at_ms = util::now_ms();
                            }
                            if (chunk.is_done) {
                                infer_done_seen = true;
                                total_tokens = chunk.tokens_used;
                                finish_reason = chunk.finish_reason;
                                return;
                            }
                            if (!chunk.thinking_delta.empty()) {
                                thinking_content += chunk.thinking_delta;
                            }
                            if (!chunk.delta_content.empty()) {
                                assistant_content += chunk.delta_content;
                            }
                            if (chunk.tool_call_delta) {
                                accumulated_tool_calls.push_back(*chunk.tool_call_delta);
                            }
                            chunk_cb(chunk);
                        },
                        [&](const std::string& error) {
                            api_ok = false;
                            stream_had_error = true;
                            stream_error_message = error;
                            infer_error_detail = error;
                        },
                        canceled);
                    return api_ok;
                }

                Message response = real_runtime.complete(req_to_send, canceled);
                if (response.role != MessageRole::Assistant) {
                    stream_had_error = true;
                    stream_error_message = "API chat completion failed";
                    infer_error_detail = stream_error_message;
                    return false;
                }
                if (first_token_at_ms == 0 &&
                    (!response.content.empty() || !response.thinking_text.empty() || !response.tool_calls.empty()))
                    first_token_at_ms = util::now_ms();
                thinking_content = response.thinking_text;
                assistant_content = response.content;
                total_tokens = response.token_count;
                accumulated_tool_calls = response.tool_calls;
                infer_done_seen = true;
                if (!thinking_content.empty()) {
                    InferenceChunk c;
                    c.thinking_delta = thinking_content;
                    chunk_cb(c);
                }
                if (!assistant_content.empty()) {
                    InferenceChunk c;
                    c.delta_content = assistant_content;
                    chunk_cb(c);
                }
                for (const auto& tc : accumulated_tool_calls) {
                    InferenceChunk c;
                    c.tool_call_delta = tc;
                    chunk_cb(c);
                }
                return true;
            }

            const NodeInferRequest node_request{
                req_to_send,
                active_slot_id,
                canceled,
            };
            NodeInferResult node_result = node_operations->infer(
                node_request,
                [&](const InferenceChunk& chunk) -> bool {
                    if (canceled()) return false;
                    if (first_token_at_ms == 0 &&
                        (!chunk.delta_content.empty() ||
                         !chunk.thinking_delta.empty() ||
                         chunk.tool_call_delta)) {
                        first_token_at_ms = util::now_ms();
                    }
                    thinking_content += chunk.thinking_delta;
                    assistant_content += chunk.delta_content;
                    if (chunk.tool_call_delta) {
                        accumulated_tool_calls.push_back(*chunk.tool_call_delta);
                    }
                    chunk_cb(chunk);
                    return !canceled();
                });
            infer_node_status = node_result.status;
            infer_error_detail = node_result.error;
            if (!node_result.ok()) {
                stream_had_error = true;
                stream_error_message = node_result.error;
                return false;
            }
            infer_done_seen = true;
            total_tokens = node_result.tokens_used;
            finish_reason = node_result.finish_reason;
            return true;
        };

        auto finalize_attempt = [&](bool transport_ok) -> bool {
            const std::string infer_label = api_backend ? "api infer" : "node infer";
            bool ok_now = transport_ok;
            if (!ok_now) {
                if (infer_done_seen && !stream_had_error) {
                    MM_WARN("handle_chat: {} transport ended non-successfully after done event;"
                            " accepting completion (agent='{}', conv='{}')",
                            infer_label, agent_id, conv_id);
                    ok_now = true;
                } else if (!api_backend) {
                    std::string detail = util::trim(infer_error_detail);
                    if (detail.size() > 400) detail = detail.substr(0, 400) + "...";
                    failure_reason = infer_label + " failed (" +
                                     to_string(infer_node_status) + ")";
                    if (!detail.empty()) failure_reason += ": " + detail;
                } else {
                    std::string detail = util::trim(infer_error_detail);
                    failure_reason = infer_label + " stream failed";
                    if (!detail.empty()) failure_reason += ": " + detail;
                }
            }
            if (stream_had_error) {
                ok_now = false;
                MM_WARN("handle_chat: {} stream error for agent '{}': {}",
                        infer_label, agent_id, stream_error_message);
                failure_reason = stream_error_message.empty()
                    ? infer_label + " reported stream error" : stream_error_message;
            }
            return ok_now;
        };

        static constexpr int kMaxStreamAttempts = 3;
        bool stream_ok = false;
        for (int attempt = 1; attempt <= kMaxStreamAttempts; ++attempt) {
            reset_infer_state();
            const bool transport_ok = run_infer_once(true);
            stream_ok = finalize_attempt(transport_ok);
            if (stream_ok) break;
            if (canceled()) {
                failure_reason = "inference canceled";
                break;
            }

            if (contains_prefill_thinking_incompatibility(failure_reason)
                || contains_prefill_thinking_incompatibility(infer_error_detail)
                || contains_prefill_thinking_incompatibility(stream_error_message)) {
                force_prefill_safe = true;
            }

            if (attempt < kMaxStreamAttempts) {
                const int backoff_ms = 250 * attempt;
                MM_WARN("handle_chat: retrying stream infer for agent '{}' in {} ms (attempt {}/{})",
                        agent_id, backoff_ms, attempt + 1, kMaxStreamAttempts);
                std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            }
        }

        if (!stream_ok && !canceled()) {
            reset_infer_state();
            const bool non_stream_transport_ok = run_infer_once(false);
            const bool non_stream_ok = finalize_attempt(non_stream_transport_ok);
            if (non_stream_ok) {
                MM_WARN("handle_chat: recovered with non-stream infer fallback (agent='{}')", agent_id);
                stream_ok = true;
            }
        }

        if (!stream_ok) {
            if (canceled()) failure_reason = "inference canceled";
            ok = false;
            break;
        }

        final_total_tokens += total_tokens;

        // Persist assistant message (with any tool calls)
        Message asst_msg;
        asst_msg.role          = MessageRole::Assistant;
        asst_msg.content       = assistant_content;
        asst_msg.thinking_text = thinking_content;
        asst_msg.tool_calls    = accumulated_tool_calls;
        asst_msg.token_count   = total_tokens;
        asst_msg.timestamp_ms  = util::now_ms();
        asst_msg.trace_events  = std::move(context_trace_events);
        db.append_message(conv_id, asst_msg, seq + 1);
        ++seq;

        // Check for memory tool calls that need execution
        bool has_memory_tools = false;
        for (const auto& tc : accumulated_tool_calls) {
            if (tool_exec.is_memory_tool(tc.function_name)) {
                has_memory_tools = true;
                break;
            }
        }

        if (!has_memory_tools || accumulated_tool_calls.empty()) {
            // No tool calls or no memory tools — final response
            final_assistant_content = assistant_content;
            final_thinking_content  = thinking_content;
            break;
        }

        // Execute each tool call and persist results
        for (const auto& tc : accumulated_tool_calls) {
            Message tool_result;
            if (tool_exec.is_memory_tool(tc.function_name)) {
                tool_result = tool_exec.execute_tool(tc, conv_id);
            } else {
                // Non-memory tool call — return error (not handled server-side)
                tool_result.role         = MessageRole::Tool;
                tool_result.tool_call_id = tc.id;
                tool_result.content      = nlohmann::json{{"error", "tool not available"}}.dump();
                tool_result.timestamp_ms = util::now_ms();
            }
            if (auto trace = build_tool_access_trace(tc, tool_result, conv_id)) {
                tool_result.trace_events.push_back(*trace);
            }

            db.append_message(conv_id, tool_result, seq + 1);
            ++seq;

            // Emit tool_result SSE event to client
            InferenceChunk tool_chunk;
            tool_chunk.tool_result_json = nlohmann::json{
                {"type", "tool_result"},
                {"tool_call_id", tc.id},
                {"function_name", tc.function_name},
                {"result", tool_result.content}
            }.dump();
            chunk_cb(tool_chunk);
        }

        // Loop: re-infer with updated context (tool results now in messages)
        MM_DEBUG("handle_chat: tool round {} complete, re-inferring (agent='{}')",
                 tool_round + 1, agent_id);
    }

    // Log completion or failure
    {
        int64_t duration_ms = util::now_ms() - infer_start_ms;
        if (ok) {
            char buf[384];
            snprintf(buf, sizeof(buf),
                     "Chat complete: agent='%s' tokens=%d content=%zuB duration=%lldms "
                     "images=%d image_bytes=%lld vision=%s projector='%s'",
                     agent_id.c_str(), final_total_tokens,
                     final_assistant_content.size(), static_cast<long long>(duration_ms),
                     perf.image_count,
                     static_cast<long long>(perf.decoded_image_bytes),
                     perf.vision_routing ? "true" : "false",
                     perf.projector_basename.c_str());
            activity_log(0, buf);
        } else {
            char buf[384];
            if (api_backend) {
                snprintf(buf, sizeof(buf),
                         "Chat error: agent='%s' backend=api reason='%s' duration=%lldms "
                         "images=%d image_bytes=%lld vision=%s projector='%s'",
                         agent_id.c_str(),
                         failure_reason.substr(0, 80).c_str(),
                         static_cast<long long>(duration_ms),
                         perf.image_count,
                         static_cast<long long>(perf.decoded_image_bytes),
                         perf.vision_routing ? "true" : "false",
                         perf.projector_basename.c_str());
            } else {
                snprintf(buf, sizeof(buf),
                         "Chat error: agent='%s' node=%s slot=%s reason='%s' duration=%lldms "
                         "images=%d image_bytes=%lld vision=%s projector='%s'",
                         agent_id.c_str(),
                         schedule_result ? schedule_result->node_id.substr(0, 8).c_str() : "",
                         active_slot_id.substr(0, 8).c_str(),
                         failure_reason.substr(0, 80).c_str(),
                         static_cast<long long>(duration_ms),
                         perf.image_count,
                         static_cast<long long>(perf.decoded_image_bytes),
                         perf.vision_routing ? "true" : "false",
                         perf.projector_basename.c_str());
            }
            activity_log(2, buf);
        }
    }

    if (!ok) {
        MM_WARN("handle_chat: skipping assistant persistence for failed stream (agent='{}', conv='{}')",
                agent_id, conv_id);
    }

    // 10. Signal completion to the SSE client. The scheduler idle guard runs
    // on every exit path, including compaction and memory lookup exceptions.
    done_cb(conv_id, ok, failure_reason);
    const int64_t completed_at_ms = util::now_ms();
    perf.node_id = schedule_result ? schedule_result->node_id : std::string{};
    perf.time_to_first_token_ms = first_token_at_ms > 0
        ? first_token_at_ms - infer_start_ms : -1;
    perf.total_ms = completed_at_ms - infer_start_ms;
    perf.output_tokens = final_total_tokens;
    perf.success = ok;
    perf.error = failure_reason;
    performance_.record(std::move(perf));

}

// ── queue_global_recall ───────────────────────────────────────────────────────
// Queues an internal inference job where the agent reviews the old conversation
// and creates global memories from local notes + conversation content.

void ControlApiServer::queue_global_recall(const AgentId& agent_id,
                                           const ConvId& conv_id) {
    auto agent = agents_.get_agent(agent_id);
    if (!agent) return;

    AgentConfig cfg = agent->get_config();
    if (!cfg.memories_enabled) return;

    activity_log(0, "Global recall queued: agent='" + agent_id +
                 "' conv='" + conv_id.substr(0, 8) + "'");

    InferenceJob recall_job;
    recall_job.job_id   = util::generate_uuid();
    recall_job.agent_id = agent_id;
    recall_job.process_fn = [this, agent_id, conv_id]() {
        const auto canceled = [this] { return queue_.cancellation_requested(); };
        if (canceled()) return;
        auto a = agents_.get_agent(agent_id);
        if (!a) return;
        AgentConfig acfg = a->get_config();
        if (!acfg.memories_enabled) return;

        activity_log(0, "Global recall starting: agent='" + agent_id + "'");

        const bool recall_api_backend = agent_uses_api_backend(acfg);
        std::unique_ptr<RuntimeClient> api_recall_storage;
        std::unique_ptr<NodeProxyRuntimeClient> node_recall_storage;
        RuntimeClient* recall_client = nullptr;
        std::optional<ScheduleResult> sched_result;
        SchedulerIdleGuard idle_guard;

        if (recall_api_backend) {
            api_recall_storage = make_api_runtime_client(acfg);
            recall_client = api_recall_storage.get();
        } else {
            if (canceled()) return;
            sched_result = scheduler_.ensure_agent_running(acfg);
            if (!sched_result) {
                MM_WARN("Global recall: no node for agent '{}': {}",
                        agent_id, scheduler_.last_error());
                activity_log(1, "Global recall failed: agent='" + agent_id + "' no node");
                return;
            }

            scheduler_.mark_agent_active(agent_id);
            idle_guard.arm(scheduler_, agent_id);

            NodeOperationsPtr recall_operations;
            try { recall_operations = registry_.operations(sched_result->node_id); }
            catch (const std::exception& e) {
                MM_WARN("Global recall: node lookup failed: {}", e.what());
                return;
            }

            node_recall_storage = std::make_unique<NodeProxyRuntimeClient>(
                std::move(recall_operations), sched_result->slot_id, canceled);
            recall_client = node_recall_storage.get();
        }

        if (canceled()) return;

        AgentDB& db = a->db();
        ToolExecutor tool_exec(db);

        // Build recall context: system prompt + conversation local memories
        auto local_mems = db.list_local_memories(conv_id);
        auto global_mems = db.list_memories();
        auto conv = db.load_conversation(conv_id);

        std::string system_text =
            "You are reviewing a completed conversation chain to maintain global summaries "
            "for future conversations.\n\n"
            "Review the conversation below and use the available tools to create global memories "
            "as concise conversation-chain summaries. Each summary should explain what this "
            "source conversation contains and why it may matter later, so future agents can "
            "decide whether to inspect the origin conversation. Be selective; only save "
            "durable context that is genuinely useful and not already captured in existing "
            "global summaries.\n\n"
            "You can also update or delete existing global memories if this conversation "
            "revealed new information that changes them.";

        if (!global_mems.empty()) {
            system_text += "\n\n## Existing global summaries\n";
            for (const auto& gm : global_mems) {
                system_text += "- [" + gm.id + "]";
                if (!gm.source_conv_id.empty()) {
                    system_text += " source_conversation=" + gm.source_conv_id;
                }
                system_text += " (importance=" + std::to_string(gm.importance) +
                               ") " + gm.content + "\n";
            }
        }

        if (!local_mems.empty()) {
            system_text += "\n\n## Local notes from this conversation\n";
            for (const auto& lm : local_mems) {
                system_text += "- [" + lm.id + "] " + lm.content + "\n";
            }
        }

        std::vector<Message> recall_ctx;
        Message sys;
        sys.role    = MessageRole::System;
        sys.content = system_text;
        recall_ctx.push_back(sys);

        // Add conversation messages as context
        if (conv) {
            if (!conv->compaction_summary.empty()) {
                Message summary_msg;
                summary_msg.role    = MessageRole::Assistant;
                summary_msg.content = "[Previous conversation summary]\n" + conv->compaction_summary;
                recall_ctx.push_back(summary_msg);
            }
            for (const auto& m : conv->messages) {
                recall_ctx.push_back(m);
            }
        }

        // Add a user prompt to trigger the recall
        Message recall_prompt;
        recall_prompt.role    = MessageRole::User;
        recall_prompt.content = "Please review this conversation and save any important "
                                "conversation-chain summaries as global memories.";
        recall_ctx.push_back(recall_prompt);

        // Run tool call loop (internal, not streamed to client)
        RuntimeClient& recall_runtime = *recall_client;
        static constexpr int kMaxRecallRounds = 5;

        for (int round = 0; round < kMaxRecallRounds; ++round) {
            if (canceled()) break;
            InferenceRequest req;
            req.model    = acfg.model_path;
            req.messages = recall_ctx;
            req.settings = acfg.runtime_settings;
            req.tools    = tool_exec.get_all_tool_definitions();
            req.stream   = false;

            Message response = recall_runtime.complete(req, canceled);
            if (canceled()) break;

            // Add assistant response to context
            recall_ctx.push_back(response);

            // Check for tool calls
            bool has_tools = false;
            for (const auto& tc : response.tool_calls) {
                if (tool_exec.is_memory_tool(tc.function_name)) {
                    has_tools = true;
                    break;
                }
            }

            if (!has_tools || response.tool_calls.empty()) break;

            // Execute tool calls
            for (const auto& tc : response.tool_calls) {
                if (canceled()) break;
                Message tool_result;
                if (tool_exec.is_memory_tool(tc.function_name)) {
                    tool_result = tool_exec.execute_tool(tc, conv_id);
                } else {
                    tool_result.role         = MessageRole::Tool;
                    tool_result.tool_call_id = tc.id;
                    tool_result.content      = nlohmann::json{{"error", "tool not available"}}.dump();
                    tool_result.timestamp_ms = util::now_ms();
                }
                recall_ctx.push_back(tool_result);
            }
        }

        if (!canceled()) {
            activity_log(0, "Global recall complete: agent='" + agent_id + "'");
        }
    };
    queue_.enqueue(std::move(recall_job));
}

// ── register_routes ───────────────────────────────────────────────────────────

void ControlApiServer::register_routes() {
    using namespace httplib;

    // Helper: compute node_compatibility info for an agent config.
    auto compute_node_compat = [this](const AgentConfig& cfg) -> nlohmann::json {
        if (agent_uses_api_backend(cfg)) {
            return nlohmann::json{
                {"backend", "api"},
                {"requires_node", false},
                {"compatible_nodes", 0},
                {"total_nodes", 0}
            };
        }

        const int64_t estimated_vram = 2048; // conservative default

        auto compatible = registry_.nodes_with_available_vram(estimated_vram);
        auto all_nodes  = registry_.list_nodes();
        int total_connected = 0;
        for (const auto& n : all_nodes) {
            if (n.connected) ++total_connected;
        }

        return nlohmann::json{
            {"estimated_vram_mb",  estimated_vram},
            {"compatible_nodes",   static_cast<int>(compatible.size())},
            {"total_nodes",        total_connected}
        };
    };

    auto extract_bearer = [](const std::string& auth_header) -> std::string {
        static const std::string kBearer = "Bearer ";
        if (auth_header.rfind(kBearer, 0) != 0) return {};
        return auth_header.substr(kBearer.size());
    };

    auto write_node_result = [](Response& res, const NodeOperationResult& result) {
        res.status = result.status > 0 ? result.status : 500;
        const std::string body = result.raw_body.empty()
            ? result.body.dump()
            : result.raw_body;
        res.set_content(body.empty() ? R"({"error":"empty node response"})" : body,
                        "application/json");
    };

    // ── POST /api/control/register-node ───────────────────────────────────────
    server_->Post("/api/control/register-node", [this, extract_bearer](const Request& req, Response& res) {
        if (!registry_.remote_nodes_enabled()) {
            res.status = 403;
            res.set_content(R"({"error":"cluster mode is disabled"})", "application/json");
            return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            std::string node_url = j.value("node_url", "");
            std::string api_key  = j.value("api_key", "");
            std::string platform = j.value("platform", "");
            std::string hostname = j.value("hostname", "");

            if (node_url.empty() || api_key.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"node_url and api_key required"})",
                                "application/json");
                return;
            }

            // A node may only self-register if it is already known (re-announce
            // after restart / address change, matched by its api_key) or if the
            // request carries the control's external bearer token. New nodes
            // must be added through pairing or POST /v1/nodes.
            const bool known_node = registry_.find_node_by_api_key(api_key).has_value();
            const bool bearer_ok = !external_api_token_.empty() &&
                extract_bearer(req.get_header_value("Authorization")) == external_api_token_;
            if (!known_node && !bearer_ok) {
                MM_WARN("register-node rejected for {}: unknown api_key and no valid bearer token",
                        node_url);
                res.status = 401;
                res.set_content(
                    R"({"error":"unauthorized: pair this node first or send the control bearer token"})",
                    "application/json");
                return;
            }

            NodeId node_id = registry_.add_node(node_url, api_key, platform, false, hostname);
            MM_INFO("Node registered: {} @ {}", node_id, node_url);

            res.set_content(
                nlohmann::json{{"node_id", node_id}, {"accepted", true}}.dump(),
                "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── GET /v1/nodes ─────────────────────────────────────────────────────────
    server_->Get("/v1/nodes/discovered", [this](const Request& /*req*/, Response& res) {
        auto discovered = registry_.get_discovered_nodes();
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& d : discovered) {
            arr.push_back(nlohmann::json{
                {"url", d.url},
                {"node_id", d.node_id},
                {"hostname", d.hostname},
                {"last_seen_ms", d.last_seen_ms}
            });
        }
        res.set_content(arr.dump(), "application/json");
    });

    server_->Post("/v1/nodes", [this](const Request& req, Response& res) {
        if (!registry_.remote_nodes_enabled()) {
            res.status = 403;
            res.set_content(R"({"error":"cluster mode is disabled"})", "application/json");
            return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string url = util::trim(j.value("url", std::string{}));
            const std::string api_key = util::trim(j.value("api_key", std::string{}));
            const std::string platform = util::trim(j.value("platform", std::string{}));
            const bool remember = j.value("remember", false);
            const std::string hostname = util::trim(j.value("hostname", std::string{}));
            if (url.empty() || api_key.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"url and api_key required"})", "application/json");
                return;
            }
            const NodeId node_id = registry_.add_node(url, api_key, platform, remember, hostname);
            publish_activity(0, "Node added via API: " + node_id + " @ " + url +
                                (remember ? " (remembered)" : ""));
            res.status = 201;
            res.set_content(nlohmann::json{{"node_id", node_id}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    server_->Delete("/v1/nodes/:id", [this](const Request& req, Response& res) {
        const std::string node_id = req.path_params.at("id");
        if (registry_.is_embedded_node(node_id)) {
            res.status = 409;
            res.set_content(R"({"error":"embedded node cannot be removed"})", "application/json");
            return;
        }
        if (!registry_.remote_nodes_enabled()) {
            res.status = 403;
            res.set_content(R"({"error":"cluster mode is disabled"})", "application/json");
            return;
        }
        try {
            registry_.get_node(node_id);
        } catch (...) {
            res.status = 404;
            res.set_content(R"({"error":"node not found"})", "application/json");
            return;
        }
        registry_.remove_node(node_id);
        publish_activity(0, "Node removed via API: " + node_id);
        res.set_content(R"({"status":"removed"})", "application/json");
    });

    server_->Post("/v1/nodes/:id/forget", [this](const Request& req, Response& res) {
        const std::string node_id = req.path_params.at("id");
        if (registry_.is_embedded_node(node_id)) {
            res.status = 409;
            res.set_content(R"({"error":"embedded node cannot be forgotten"})", "application/json");
            return;
        }
        if (!registry_.remote_nodes_enabled()) {
            res.status = 403;
            res.set_content(R"({"error":"cluster mode is disabled"})", "application/json");
            return;
        }
        try {
            registry_.get_node(node_id);
        } catch (...) {
            res.status = 404;
            res.set_content(R"({"error":"node not found"})", "application/json");
            return;
        }
        const bool changed = registry_.forget_node(node_id);
        publish_activity(0, "Node forget requested via API: " + node_id);
        res.set_content(nlohmann::json{{"status", "forgotten"}, {"changed", changed}}.dump(),
                        "application/json");
    });

    auto pair_targets_embedded = [](const Request& req) {
        try {
            const auto body = nlohmann::json::parse(req.body);
            const auto is_local = [](const std::string& value) {
                return util::to_lower(util::trim(value)) == "local";
            };
            return is_local(body.value("node_id", std::string{})) ||
                   is_local(body.value("url", std::string{}));
        } catch (...) {
            return false;
        }
    };

    server_->Post("/v1/nodes/pair/start", [this, pair_targets_embedded](
        const Request& req, Response& res) {
        if (pair_targets_embedded(req)) {
            res.status = 409;
            res.set_content(R"({"error":"embedded node cannot be paired"})",
                            "application/json");
            return;
        }
        if (!registry_.remote_nodes_enabled()) {
            res.status = 403;
            res.set_content(R"({"error":"cluster mode is disabled"})", "application/json");
            return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string url = util::trim(j.value("url", std::string{}));
            if (url.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"url required"})", "application/json");
                return;
            }
            std::string nonce = registry_.start_pair(url);
            if (nonce.empty()) {
                res.status = 502;
                res.set_content(R"({"error":"pair start failed"})", "application/json");
                return;
            }
            publish_activity(0, "Pair start requested for " + url);
            res.set_content(nlohmann::json{{"nonce", nonce}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    server_->Post("/v1/nodes/pair/complete", [this, pair_targets_embedded](
        const Request& req, Response& res) {
        if (pair_targets_embedded(req)) {
            res.status = 409;
            res.set_content(R"({"error":"embedded node cannot be paired"})",
                            "application/json");
            return;
        }
        if (!registry_.remote_nodes_enabled()) {
            res.status = 403;
            res.set_content(R"({"error":"cluster mode is disabled"})", "application/json");
            return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string url = util::trim(j.value("url", std::string{}));
            const std::string nonce = util::trim(j.value("nonce", std::string{}));
            const std::string pin_or_psk = util::trim(j.value("pin_or_psk", std::string{}));
            const bool remember = j.value("remember", false);
            if (url.empty() || nonce.empty() || pin_or_psk.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"url, nonce, and pin_or_psk required"})", "application/json");
                return;
            }
            std::string key = registry_.complete_pair(url, nonce, pin_or_psk, remember);
            if (key.empty()) {
                res.status = 502;
                res.set_content(R"({"error":"pair complete failed"})", "application/json");
                return;
            }
            publish_activity(0, "Pair complete accepted for " + url +
                                (remember ? " (remembered)" : ""));
            res.set_content(nlohmann::json{{"api_key", key}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    server_->Post("/v1/nodes/pair/psk", [this, pair_targets_embedded](
        const Request& req, Response& res) {
        if (pair_targets_embedded(req)) {
            res.status = 409;
            res.set_content(R"({"error":"embedded node cannot be paired"})",
                            "application/json");
            return;
        }
        if (!registry_.remote_nodes_enabled()) {
            res.status = 403;
            res.set_content(R"({"error":"cluster mode is disabled"})", "application/json");
            return;
        }
        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string url = util::trim(j.value("url", std::string{}));
            std::string psk = util::trim(j.value("psk", std::string{}));
            const bool remember = j.value("remember", false);
            if (url.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"url required"})", "application/json");
                return;
            }
            if (psk.empty() && allow_legacy_environment_) {
                const char* env_psk = std::getenv("MM_PAIRING_KEY");
                if (env_psk && *env_psk) psk = env_psk;
            }
            if (psk.empty()) {
                res.status = 400;
                const std::string detail = allow_legacy_environment_
                    ? "psk required (or set MM_PAIRING_KEY)"
                    : "psk required";
                res.set_content(nlohmann::json{{"error", detail}}.dump(),
                                "application/json");
                return;
            }
            std::string key = registry_.pair_node(url, psk, remember);
            if (key.empty()) {
                res.status = 502;
                res.set_content(R"({"error":"psk pair failed"})", "application/json");
                return;
            }
            publish_activity(0, "PSK pair accepted for " + url +
                                (remember ? " (remembered)" : ""));
            res.set_content(nlohmann::json{{"api_key", key}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    server_->Get("/v1/nodes", [this](const Request& /*req*/, Response& res) {
        auto nodes = registry_.list_nodes();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& n : nodes) arr.push_back(nlohmann::json(n));
        res.set_content(arr.dump(), "application/json");
    });

    // Transport-neutral node management routes used by AIO's unified control
    // UI/CLI and also useful for remote nodes.  Embedded calls never leave the
    // process; remote calls retain the existing node REST contract.
    server_->Get("/v1/nodes/:id/status", [this, write_node_result](
        const Request& req, Response& res) {
        try {
            write_node_result(res, registry_.operations(req.path_params.at("id"))->status());
        } catch (const std::exception& e) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });
    server_->Get("/v1/nodes/:id/logs", [this, write_node_result](
        const Request& req, Response& res) {
        int tail = 100;
        if (req.has_param("tail")) {
            try { tail = std::stoi(req.get_param_value("tail")); }
            catch (...) { tail = 100; }
        }
        tail = std::clamp(tail, 1, 5000);
        try {
            write_node_result(res, registry_.operations(req.path_params.at("id"))->logs(tail));
        } catch (const std::exception& e) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });
    server_->Get("/v1/nodes/:id/runtime/llama", [this, write_node_result](
        const Request& req, Response& res) {
        try {
            write_node_result(
                res, registry_.operations(req.path_params.at("id"))->llama_runtime());
        } catch (const std::exception& e) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });
    auto allow_node_mutation = [this](const Request& req, Response& res) {
        const std::string& node_id = req.path_params.at("id");
        if (registry_.is_embedded_node(node_id) || registry_.remote_nodes_enabled()) {
            return true;
        }
        res.status = 403;
        res.set_content(R"({"error":"cluster mode is disabled"})",
                        "application/json");
        return false;
    };
    server_->Post("/v1/nodes/:id/actions/cancel",
        [this, write_node_result, allow_node_mutation](
            const Request& req, Response& res) {
            if (!allow_node_mutation(req, res)) return;
            try {
                write_node_result(
                    res, registry_.operations(req.path_params.at("id"))->cancel_action());
            } catch (const std::exception& e) {
                res.status = 404;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                                "application/json");
            }
        });

    auto parse_node_action_body = [](const Request& req) {
        if (req.body.empty()) return nlohmann::json::object();
        return nlohmann::json::parse(req.body);
    };
    server_->Post("/v1/nodes/:id/runtime/llama/provision",
        [this, write_node_result, parse_node_action_body, allow_node_mutation](
            const Request& req, Response& res) {
            if (!allow_node_mutation(req, res)) return;
            try {
                write_node_result(res, registry_.operations(req.path_params.at("id"))
                    ->llama_provision(parse_node_action_body(req)));
            } catch (const nlohmann::json::exception& e) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 404;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            }
        });
    server_->Post("/v1/nodes/:id/runtime/llama/update",
        [this, write_node_result, parse_node_action_body, allow_node_mutation](
            const Request& req, Response& res) {
            if (!allow_node_mutation(req, res)) return;
            try {
                write_node_result(res, registry_.operations(req.path_params.at("id"))
                    ->llama_update(parse_node_action_body(req)));
            } catch (const nlohmann::json::exception& e) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 404;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            }
        });
    server_->Post("/v1/nodes/:id/runtime/llama/check-update",
        [this, write_node_result, parse_node_action_body, allow_node_mutation](
            const Request& req, Response& res) {
            if (!allow_node_mutation(req, res)) return;
            try {
                write_node_result(res, registry_.operations(req.path_params.at("id"))
                    ->llama_check_update(parse_node_action_body(req)));
            } catch (const nlohmann::json::exception& e) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 404;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            }
        });
    server_->Post("/v1/nodes/:id/runtime/llama/switch",
        [this, write_node_result, parse_node_action_body, allow_node_mutation](
            const Request& req, Response& res) {
            if (!allow_node_mutation(req, res)) return;
            try {
                write_node_result(res, registry_.operations(req.path_params.at("id"))
                    ->llama_switch(parse_node_action_body(req)));
            } catch (const nlohmann::json::exception& e) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 404;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            }
        });
    server_->Post("/v1/nodes/:id/runtime/llama/diagnose",
        [this, write_node_result, allow_node_mutation](
            const Request& req, Response& res) {
            if (!allow_node_mutation(req, res)) return;
            try {
                write_node_result(res, registry_.operations(req.path_params.at("id"))
                    ->llama_diagnose());
            } catch (const std::exception& e) {
                res.status = 404;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            }
        });
    server_->Post("/v1/nodes/:id/runtime/llama/recover",
        [this, write_node_result, parse_node_action_body, allow_node_mutation](
            const Request& req, Response& res) {
            if (!allow_node_mutation(req, res)) return;
            try {
                write_node_result(res, registry_.operations(req.path_params.at("id"))
                    ->llama_recover(parse_node_action_body(req)));
            } catch (const nlohmann::json::exception& e) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 404;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            }
        });

    server_->Get("/v1/activity", [this](const Request& req, Response& res) {
        int tail = 20;
        if (req.has_param("tail")) {
            try {
                tail = std::stoi(req.get_param_value("tail"));
            } catch (...) {
                res.status = 400;
                res.set_content(R"({"error":"tail must be an integer"})", "application/json");
                return;
            }
        }
        if (tail < 1) tail = 1;
        if (tail > 5000) tail = 5000;

        int level_filter = -1;
        if (req.has_param("level")) {
            std::string raw = util::to_lower(util::trim(req.get_param_value("level")));
            if (raw == "info" || raw == "0") level_filter = 0;
            else if (raw == "warn" || raw == "warning" || raw == "1") level_filter = 1;
            else if (raw == "error" || raw == "2") level_filter = 2;
            else {
                res.status = 400;
                res.set_content(R"({"error":"level must be info|warn|error|0|1|2"})", "application/json");
                return;
            }
        }

        std::vector<nlohmann::json> filtered;
        {
            std::lock_guard<std::mutex> lk(activity_mutex_);
            filtered.reserve(activity_entries_.size());
            for (const auto& entry : activity_entries_) {
                if (level_filter >= 0) {
                    const int lv = entry.value("level", 0);
                    if (lv != level_filter) continue;
                }
                filtered.push_back(entry);
            }
        }
        const int total = static_cast<int>(filtered.size());
        const int start = std::max(0, total - tail);
        nlohmann::json out = nlohmann::json::array();
        for (int i = start; i < total; ++i) out.push_back(filtered[static_cast<std::size_t>(i)]);
        res.set_content(nlohmann::json{{"entries", out}}.dump(), "application/json");
    });

    server_->Get("/v1/performance", [this](const Request& req, Response& res) {
        int tail = 200;
        if (req.has_param("tail")) {
            try { tail = std::stoi(req.get_param_value("tail")); }
            catch (...) {
                res.status = 400;
                res.set_content(R"({"error":"tail must be an integer"})", "application/json");
                return;
            }
        }
        tail = std::clamp(tail, 1, 2000);
        res.set_content(performance_.snapshot(static_cast<std::size_t>(tail)).dump(),
                        "application/json");
    });

    server_->Delete("/v1/performance", [this](const Request&, Response& res) {
        performance_.clear();
        res.set_content(R"({"cleared":true})", "application/json");
    });

    // ── GET /v1/agents ────────────────────────────────────────────────────────
    server_->Get("/v1/models", [this](const Request& /*req*/, Response& res) {
        res.set_content(mantic_model_list(agents_.list_agents()).dump(), "application/json");
    });

    server_->Get("/v1/agents", [this](const Request& /*req*/, Response& res) {
        auto configs = agents_.list_agents();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& c : configs) {
            nlohmann::json j = c;
            auto placement = scheduler_.get_placement(c.id);
            if (placement) {
                j["placement"] = *placement;
                j["status"] = placement->suspended
                    ? "suspended"
                    : (placement->is_active ? "active" : "idle");
            } else if (agent_uses_api_backend(c)) {
                j["status"] = "api";
            } else {
                j["status"] = "unplaced";
            }
            arr.push_back(std::move(j));
        }
        res.set_content(arr.dump(), "application/json");
    });

    // ── POST /v1/agents ───────────────────────────────────────────────────────
    server_->Post("/v1/agents", [this, compute_node_compat](const Request& req, Response& res) {
        try {
            AgentConfig cfg = nlohmann::json::parse(req.body).get<AgentConfig>();
            auto validation = validate_agent_config(cfg, &registry_, models_dir_);
            if (!validation.ok()) {
                nlohmann::json body = {
                    {"error", "validation_failed"},
                    {"issues", validation.issues}
                };
                if (validation.model_info) body["model_info"] = *validation.model_info;
                res.status = 400;
                res.set_content(body.dump(), "application/json");
                return;
            }
            AgentId id = agents_.create_agent(cfg);
            auto a = agents_.get_agent(id);
            nlohmann::json resp = a ? nlohmann::json(a->get_config())
                                    : nlohmann::json{{"id", id}};
            if (a) {
                resp["node_compatibility"] = compute_node_compat(a->get_config());
            }
            res.status = 201;
            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── GET /v1/agents/:id ────────────────────────────────────────────────────
    server_->Get("/v1/agents/:id", [this, compute_node_compat](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        auto cfg = a->get_config();
        nlohmann::json j = cfg;
        // Include placement info if available.
        auto placement = scheduler_.get_placement(id);
        if (placement) {
            j["placement"] = *placement;
            j["status"] = placement->suspended
                ? "suspended"
                : (placement->is_active ? "active" : "idle");
        } else if (agent_uses_api_backend(cfg)) {
            j["status"] = "api";
        } else {
            j["status"] = "unplaced";
        }
        j["node_compatibility"] = compute_node_compat(cfg);
        res.set_content(j.dump(), "application/json");
    });

    // ── PUT /v1/agents/:id ────────────────────────────────────────────────────
    server_->Put("/v1/agents/:id", [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        try {
            AgentConfig cfg = nlohmann::json::parse(req.body).get<AgentConfig>();
            // If body omits "id", keep the current ID (no rename).
            if (cfg.id.empty()) cfg.id = id;
            auto validation = validate_agent_config(cfg, &registry_, models_dir_);
            if (!validation.ok()) {
                nlohmann::json body = {
                    {"error", "validation_failed"},
                    {"issues", validation.issues}
                };
                if (validation.model_info) body["model_info"] = *validation.model_info;
                res.status = 400;
                res.set_content(body.dump(), "application/json");
                return;
            }
            auto current = agents_.get_agent(id);
            if (!current) {
                res.status = 404;
                return;
            }
            const AgentConfig old_cfg = current->get_config();
            current.reset();
            const bool id_changed = cfg.id != id;
            if (id_changed && agents_.get_agent(cfg.id)) {
                res.status = 409;
                res.set_content(
                    nlohmann::json{{"error", "Agent ID '" + cfg.id +
                        "' is already in use"}}.dump(),
                    "application/json");
                return;
            }
            const bool local_to_api =
                agent_backend(old_cfg) == "llama-cpp" && agent_uses_api_backend(cfg);
            if (id_changed || local_to_api) {
                scheduler_.release_agent(id);
            }
            AgentId final_id = agents_.update_agent(id, cfg);
            if (final_id.empty()) { res.status = 404; return; }
            auto a = agents_.get_agent(final_id);
            if (!a) { res.status = 404; return; }
            res.set_content(nlohmann::json(a->get_config()).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── DELETE /v1/agents/:id ─────────────────────────────────────────────────
    server_->Delete("/v1/agents/:id", [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        scheduler_.release_agent(id);
        if (!agents_.delete_agent(id)) { res.status = 404; return; }
        res.set_content(R"({"status":"deleted"})", "application/json");
    });

    auto voice_request_mentions_real_person = [](const VoiceDesignProposal& p) {
        const std::string text = util::to_lower(p.display_name + "\n" +
                                                p.voice_description + "\n" +
                                                p.rationale);
        return text.find("celebrity") != std::string::npos ||
               text.find("impersonate") != std::string::npos ||
               text.find("real person") != std::string::npos ||
               text.find("sound like ") != std::string::npos ||
               text.find("sounds like ") != std::string::npos;
    };

    auto infer_voice_design_proposal =
        [this](const AgentConfig& acfg,
               std::string* error) -> std::optional<VoiceDesignProposal> {
        const auto canceled = [this] {
            return queue_.cancellation_requested();
        };
        if (canceled()) {
            if (error) *error = "voice proposal inference canceled";
            return std::nullopt;
        }
        const bool voice_api_backend = agent_uses_api_backend(acfg);
        std::unique_ptr<RuntimeClient> api_voice_client;
        std::unique_ptr<NodeProxyRuntimeClient> node_voice_client;
        RuntimeClient* client = nullptr;
        if (voice_api_backend) {
            api_voice_client = make_api_runtime_client(acfg);
            client = api_voice_client.get();
        } else {
            auto sched_result = scheduler_.ensure_agent_running(acfg);
            if (!sched_result) {
                if (error) *error = scheduler_.last_error().empty()
                    ? "no node available for voice design proposal"
                    : scheduler_.last_error();
                return std::nullopt;
            }

            NodeOperationsPtr voice_operations;
            try {
                voice_operations = registry_.operations(sched_result->node_id);
            } catch (const std::exception& e) {
                if (error) *error = e.what();
                return std::nullopt;
            }

            scheduler_.mark_agent_active(acfg.id);
            node_voice_client = std::make_unique<NodeProxyRuntimeClient>(
                std::move(voice_operations), sched_result->slot_id,
                [this] { return queue_.cancellation_requested(); });
            client = node_voice_client.get();
        }
        try {
            std::vector<Message> messages;
            Message sys;
            sys.role = MessageRole::System;
            sys.content =
                "You are designing a synthetic TTS voice for yourself as an AI agent. "
                "Return only a strict JSON object with these string fields: "
                "display_name, language, voice_description, sample_text, rationale. "
                "Do not imitate, reference, or request the voice of any real person, "
                "celebrity, private person, or copyrighted character. Design an original "
                "voice that fits your role and temperament. Keep sample_text under 280 characters.";
            messages.push_back(sys);

            Message user;
            user.role = MessageRole::User;
            user.content =
                "Agent name: " + acfg.name + "\n\n"
                "Agent system prompt:\n" + acfg.system_prompt + "\n\n"
                "Design your own original voice now.";
            messages.push_back(user);

            InferenceRequest req;
            req.model = acfg.model_path;
            req.messages = std::move(messages);
            req.settings = acfg.runtime_settings;
            req.settings.max_tokens = std::min(std::max(req.settings.max_tokens, 256), 768);
            req.stream = false;

            Message response = client->complete(req, canceled);
            if (canceled()) {
                if (error) *error = "voice proposal inference canceled";
                return std::nullopt;
            }
            if (!voice_api_backend) {
                scheduler_.mark_agent_idle(acfg.id);
            }

            const auto parsed = nlohmann::json::parse(extract_json_object_text(response.content));
            VoiceDesignProposal proposal;
            proposal.display_name = util::trim(parsed.value("display_name", std::string{}));
            proposal.language = util::trim(parsed.value("language", std::string{"Auto"}));
            proposal.voice_description =
                util::trim(parsed.value("voice_description", std::string{}));
            proposal.sample_text = util::trim(parsed.value("sample_text", std::string{}));
            proposal.rationale = util::trim(parsed.value("rationale", std::string{}));
            return proposal;
        } catch (const std::exception& e) {
            if (!voice_api_backend) {
                scheduler_.mark_agent_idle(acfg.id);
            }
            if (error) *error = e.what();
            return std::nullopt;
        }
    };

    auto build_voice_state = [this](Agent& agent) {
        AgentDB& db = agent.db();
        nlohmann::json body;
        auto active = db.get_active_voice_profile();
        body["active_profile"] = active ? nlohmann::json(*active) : nlohmann::json(nullptr);
        body["profiles"] = db.list_voice_profiles();
        body["proposals"] = db.list_voice_proposals();
        body["tts_enabled"] = tts_.enabled();
        body["provider"] = tts_.provider_name();
        body["backend"] = tts_.backend();
        return body;
    };

    auto synthesize_agent_speech =
        [this](const AgentId& agent_id,
               TtsSynthesisRequest request,
               TtsSynthesisResult* out,
               std::string* error,
               int* status) -> bool {
        auto agent = agents_.get_agent(agent_id);
        if (!agent) {
            if (status) *status = 404;
            if (error) *error = "agent not found";
            return false;
        }
        request.agent_id = agent_id;
        request.text = util::trim(request.text);
        request.format = util::to_lower(util::trim(request.format.empty() ? "wav" : request.format));
        if (request.text.empty()) {
            if (status) *status = 400;
            if (error) *error = "text is required";
            return false;
        }
        if (!is_supported_tts_format(request.format)) {
            if (status) *status = 400;
            if (error) *error = "only wav speech output is supported in this runtime";
            return false;
        }
        if (!tts_.enabled()) {
            if (status) *status = 503;
            if (error) *error = "TTS service is disabled";
            return false;
        }

        AgentDB& db = agent->db();
        std::optional<AgentVoiceProfile> profile =
            request.voice_profile_id.empty()
                ? db.get_active_voice_profile()
                : db.get_voice_profile(request.voice_profile_id);
        if (!profile) {
            if (status) *status = request.voice_profile_id.empty() ? 409 : 404;
            if (error) *error = request.voice_profile_id.empty()
                ? "agent has no active voice profile"
                : "voice profile not found";
            return false;
        }
        if (profile->voice_clone_prompt_path.empty() ||
            !path_file_exists(profile->voice_clone_prompt_path)) {
            if (status) *status = 409;
            if (error) *error = "active voice profile is missing its clone-prompt artifact";
            return false;
        }

        const std::string language = request.language.empty() ? profile->language : request.language;
        const std::string text_hash = sha256_hex(profile->id + "\n" +
                                                 request.format + "\n" +
                                                 language + "\n" +
                                                 request.text);
        if (request.use_cache) {
            auto cached = db.find_tts_cache_entry(profile->id,
                                                  text_hash,
                                                  request.conversation_id,
                                                  request.message_index);
            if (cached && path_file_exists(cached->audio_path)) {
                if (out) *out = *cached;
                return true;
            }
        }

        const TtsCacheId cache_id = util::generate_uuid();
        const auto cache_dir = agent_tts_cache_dir(tts_.config(), data_dir_, agent_id);
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);
        if (ec) {
            if (status) *status = 500;
            if (error) *error = "failed to create TTS cache directory: " + ec.message();
            return false;
        }
        const auto audio_path = (cache_dir / (cache_id + ".wav")).string();
        auto svc = tts_.synthesize(request, *profile, audio_path);
        if (!svc.ok) {
            remove_file_quietly(audio_path);
            if (status) *status = svc.status == 0 ? 503 : svc.status;
            if (error) *error = svc.error.empty() ? "TTS synthesis failed" : svc.error;
            return false;
        }

        TtsSynthesisResult result;
        result.cache_id = cache_id;
        result.agent_id = agent_id;
        result.voice_profile_id = profile->id;
        result.conversation_id = request.conversation_id;
        result.message_index = request.message_index;
        result.text_hash = text_hash;
        result.audio_path = svc.audio_path.empty() ? audio_path : svc.audio_path;
        result.mime_type = tts_mime_type_for_format(request.format);
        result.format = request.format.empty() ? "wav" : request.format;
        result.sample_rate = svc.sample_rate;
        result.duration_ms = svc.duration_ms;
        result.cached = false;
        result.created_at_ms = util::now_ms();
        result.expires_at_ms = result.created_at_ms + std::max<int64_t>(1, tts_.config().cache_ttl_ms);
        db.save_tts_cache_entry(result);
        if (out) *out = result;
        return true;
    };

    server_->Get("/v1/agents/:id/voice", [this, build_voice_state](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        auto agent = agents_.get_agent(id);
        if (!agent) {
            set_json_error(res, 404, "agent not found");
            return;
        }
        res.set_content(build_voice_state(*agent).dump(), "application/json");
    });

    server_->Get("/v1/agents/:id/voice/proposals", [this](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        auto agent = agents_.get_agent(id);
        if (!agent) {
            set_json_error(res, 404, "agent not found");
            return;
        }
        res.set_content(nlohmann::json{{"proposals", agent->db().list_voice_proposals()}}.dump(),
                        "application/json");
    });

    server_->Post("/v1/agents/:id/voice/proposals",
                  [this, infer_voice_design_proposal, voice_request_mentions_real_person]
                  (const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        auto agent = agents_.get_agent(id);
        if (!agent) {
            set_json_error(res, 404, "agent not found");
            return;
        }

        try {
            const auto j = parse_request_json(req);
            VoiceDesignProposal proposal;
            const bool has_explicit_design =
                !util::trim(j.value("voice_description", std::string{})).empty() ||
                !util::trim(j.value("sample_text", std::string{})).empty();
            if (!has_explicit_design) {
                std::string infer_error;
                auto inferred = infer_voice_design_proposal(agent->get_config(), &infer_error);
                if (!inferred) {
                    set_json_error(res, 503, "voice proposal inference failed", infer_error);
                    return;
                }
                proposal = *inferred;
            }

            const AgentConfig cfg = agent->get_config();
            proposal.id = util::generate_uuid();
            proposal.agent_id = id;
            proposal.display_name = util::trim(
                j.value("display_name", proposal.display_name.empty()
                    ? cfg.name + " Voice"
                    : proposal.display_name));
            proposal.language = util::trim(j.value("language", proposal.language));
            if (proposal.language.empty()) proposal.language = "Auto";
            proposal.voice_description = util::trim(
                j.value("voice_description", proposal.voice_description));
            proposal.sample_text = util::trim(j.value("sample_text", proposal.sample_text));
            proposal.rationale = util::trim(j.value("rationale", proposal.rationale));
            proposal.status = "pending";
            proposal.provider = tts_.provider_name();
            proposal.voice_design_model_id =
                j.value("voice_design_model_id", tts_.config().voice_design_model_id);
            proposal.clone_model_id =
                j.value("clone_model_id", tts_.default_synthesis_model_id());
            proposal.created_at_ms = util::now_ms();
            proposal.updated_at_ms = proposal.created_at_ms;

            if (proposal.voice_description.empty() || proposal.sample_text.empty()) {
                set_json_error(res, 400, "voice_description and sample_text are required");
                return;
            }
            if (voice_request_mentions_real_person(proposal)) {
                set_json_error(res, 400, "voice proposals must describe an original synthetic voice");
                return;
            }

            agent->db().save_voice_proposal(proposal);
            res.status = 201;
            res.set_content(nlohmann::json{{"proposal", proposal}}.dump(), "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid voice proposal request", e.what());
        }
    });

    server_->Post("/v1/agents/:id/voice/proposals/:proposal_id/sample",
                  [this](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        const std::string proposal_id = req.path_params.at("proposal_id");
        auto agent = agents_.get_agent(id);
        if (!agent) {
            set_json_error(res, 404, "agent not found");
            return;
        }
        auto proposal = agent->db().get_voice_proposal(proposal_id);
        if (!proposal) {
            set_json_error(res, 404, "voice proposal not found");
            return;
        }
        if (!tts_.enabled()) {
            set_json_error(res, 503, "TTS service is disabled");
            return;
        }

        const auto proposal_dir = agent_tts_dir(data_dir_, id, "proposals");
        std::error_code ec;
        std::filesystem::create_directories(proposal_dir, ec);
        if (ec) {
            set_json_error(res, 500, "failed to create TTS proposal directory", ec.message());
            return;
        }

        const auto preview_path = (proposal_dir / (proposal_id + ".wav")).string();
        const auto prompt_path = (proposal_dir / (proposal_id + ".prompt.pkl")).string();
        auto svc = tts_.generate_voice_sample(*proposal, preview_path, prompt_path);
        if (!svc.ok) {
            agent->db().update_voice_proposal_status(
                proposal_id,
                "error",
                svc.error.empty() ? "voice sample generation failed" : svc.error);
            set_json_error(res,
                           svc.status == 0 ? 503 : svc.status,
                           "voice sample generation failed",
                           svc.error);
            return;
        }

        proposal->preview_audio_path = svc.audio_path.empty() ? preview_path : svc.audio_path;
        proposal->voice_clone_prompt_path =
            svc.voice_clone_prompt_path.empty() ? prompt_path : svc.voice_clone_prompt_path;
        proposal->status = "sampled";
        proposal->error.clear();
        proposal->updated_at_ms = util::now_ms();
        agent->db().save_voice_proposal(*proposal);
        res.set_content(nlohmann::json{
            {"proposal", *proposal},
            {"preview_url", "/v1/agents/" + id + "/voice/proposals/" + proposal_id + "/sample"}
        }.dump(), "application/json");
    });

    server_->Get("/v1/agents/:id/voice/proposals/:proposal_id/sample",
                 [this](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        const std::string proposal_id = req.path_params.at("proposal_id");
        auto agent = agents_.get_agent(id);
        if (!agent) {
            set_json_error(res, 404, "agent not found");
            return;
        }
        auto proposal = agent->db().get_voice_proposal(proposal_id);
        if (!proposal || proposal->preview_audio_path.empty() ||
            !path_file_exists(proposal->preview_audio_path)) {
            set_json_error(res, 404, "voice proposal sample not found");
            return;
        }
        if (!set_file_response(res, proposal->preview_audio_path, "audio/wav")) {
            set_json_error(res, 404, "voice proposal sample not found");
        }
    });

    server_->Post("/v1/agents/:id/voice/proposals/:proposal_id/approve",
                  [this](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        const std::string proposal_id = req.path_params.at("proposal_id");
        auto agent = agents_.get_agent(id);
        if (!agent) {
            set_json_error(res, 404, "agent not found");
            return;
        }
        auto proposal = agent->db().get_voice_proposal(proposal_id);
        if (!proposal) {
            set_json_error(res, 404, "voice proposal not found");
            return;
        }
        if (proposal->voice_clone_prompt_path.empty() ||
            !path_file_exists(proposal->voice_clone_prompt_path)) {
            set_json_error(res, 400, "proposal must be sampled before approval");
            return;
        }

        AgentVoiceProfile profile;
        profile.id = util::generate_uuid();
        profile.agent_id = id;
        profile.display_name = proposal->display_name;
        profile.language = proposal->language;
        profile.voice_description = proposal->voice_description;
        profile.sample_text = proposal->sample_text;
        profile.rationale = proposal->rationale;
        profile.provider = proposal->provider;
        profile.voice_design_model_id = proposal->voice_design_model_id;
        profile.clone_model_id = proposal->clone_model_id;
        profile.reference_audio_path = proposal->preview_audio_path;
        profile.voice_clone_prompt_path = proposal->voice_clone_prompt_path;
        profile.approved_from_proposal_id = proposal->id;
        profile.active = true;
        profile.created_at_ms = util::now_ms();
        profile.updated_at_ms = profile.created_at_ms;
        agent->db().save_voice_profile(profile);
        agent->db().update_voice_proposal_status(proposal_id, "approved");
        proposal->status = "approved";
        proposal->error.clear();
        proposal->updated_at_ms = util::now_ms();

        res.set_content(nlohmann::json{
            {"profile", profile},
            {"proposal", *proposal}
        }.dump(), "application/json");
    });

    server_->Post("/v1/agents/:id/voice/proposals/:proposal_id/reject",
                  [this](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        const std::string proposal_id = req.path_params.at("proposal_id");
        auto agent = agents_.get_agent(id);
        if (!agent) {
            set_json_error(res, 404, "agent not found");
            return;
        }
        auto proposal = agent->db().get_voice_proposal(proposal_id);
        if (!proposal) {
            set_json_error(res, 404, "voice proposal not found");
            return;
        }
        agent->db().update_voice_proposal_status(proposal_id, "rejected");
        proposal->status = "rejected";
        proposal->updated_at_ms = util::now_ms();
        res.set_content(nlohmann::json{{"proposal", *proposal}}.dump(), "application/json");
    });

    server_->Post("/v1/agents/:id/speech",
                  [synthesize_agent_speech](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        try {
            TtsSynthesisRequest request = parse_request_json(req).get<TtsSynthesisRequest>();
            TtsSynthesisResult result;
            std::string error;
            int status = 500;
            if (!synthesize_agent_speech(id, request, &result, &error, &status)) {
                set_json_error(res, status, error.empty() ? "speech synthesis failed" : error);
                return;
            }
            const std::string audio_url =
                "/v1/agents/" + id + "/speech/cache/" + result.cache_id;
            res.set_content(nlohmann::json{
                {"result", result},
                {"audio_url", audio_url}
            }.dump(), "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid speech synthesis request", e.what());
        }
    });

    server_->Get("/v1/agents/:id/speech/cache/:cache_id",
                 [this](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        const std::string cache_id = req.path_params.at("cache_id");
        auto agent = agents_.get_agent(id);
        if (!agent) {
            set_json_error(res, 404, "agent not found");
            return;
        }
        auto entry = agent->db().get_tts_cache_entry(cache_id);
        if (!entry || !path_file_exists(entry->audio_path)) {
            set_json_error(res, 404, "speech cache entry not found");
            return;
        }
        if (!set_file_response(res, entry->audio_path, entry->mime_type)) {
            set_json_error(res, 404, "speech audio file not found");
        }
    });

    server_->Post("/v1/audio/speech",
                  [synthesize_agent_speech](const Request& req, Response& res) {
        try {
            const auto j = parse_request_json(req);
            const std::string voice = util::trim(j.value("voice", std::string{}));
            static const std::string prefix = "agent:";
            if (voice.rfind(prefix, 0) != 0 || voice.size() <= prefix.size()) {
                set_json_error(res, 400, "voice must be of the form agent:{agent_id}");
                return;
            }
            const std::string agent_id = voice.substr(prefix.size());

            TtsSynthesisRequest request = j.get<TtsSynthesisRequest>();
            request.agent_id = agent_id;
            if (request.format.empty()) {
                request.format = j.value("response_format", std::string{"wav"});
            }

            TtsSynthesisResult result;
            std::string error;
            int status = 500;
            if (!synthesize_agent_speech(agent_id, request, &result, &error, &status)) {
                set_json_error(res, status, error.empty() ? "speech synthesis failed" : error);
                return;
            }
            if (!set_file_response(res, result.audio_path, result.mime_type)) {
                set_json_error(res, 404, "speech audio file not found");
            }
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid audio speech request", e.what());
        }
    });

    // ── POST /v1/agents/:id/chat  (SSE streaming) ─────────────────────────────
    // Managed image attachments. Uploads stream directly to the agent-owned
    // directory; SQLite stores metadata and relative paths only.
    server_->PostUpload("/v1/agents/:id/attachments",
        [this](const httplib::Request& req, httplib::Response& res,
               const HttpServer::UploadPump& pump) {
        const std::string agent_id = req.path_params.at("id");
        auto agent = agents_.get_agent(agent_id);
        if (!agent) { set_json_error(res, 404, "agent not found"); return; }

        const std::string mime = normalized_image_mime(
            req.get_header_value("Content-Type"));
        if (mime != "image/jpeg" && mime != "image/png") {
            set_json_error(res, 415, "only image/jpeg and image/png uploads are supported");
            return;
        }
        if (req.has_header("Content-Length")) {
            try {
                if (std::stoll(req.get_header_value("Content-Length")) > kMaxImageBytes) {
                    set_json_error(res, 413, "image exceeds the 50 MiB limit");
                    return;
                }
            } catch (...) {
                set_json_error(res, 400, "invalid Content-Length");
                return;
            }
        }

        AgentDB& db = agent->db();
        ImageAttachment attachment;
        attachment.id = util::generate_uuid();
        attachment.original_filename = safe_attachment_filename(
            req.get_header_value("X-Filename"), mime);
        attachment.mime_type = mime;
        attachment.relative_path = "attachments/" + attachment.id +
            (mime == "image/png" ? ".png" : ".jpg");
        attachment.created_at_ms = util::now_ms();
        attachment.expires_at_ms = attachment.created_at_ms + kPendingAttachmentTtlMs;

        const std::filesystem::path destination(db.attachment_file_path(attachment));
        const std::filesystem::path temporary = destination.string() + ".part";
        std::filesystem::create_directories(destination.parent_path());
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            set_json_error(res, 500, "cannot create managed attachment");
            return;
        }
        std::vector<unsigned char> signature;
        signature.reserve(8);
        int64_t received = 0;
        bool too_large = false;
        bool write_failed = false;
        const bool pumped = pump([&](const char* data, std::size_t length) {
            if (received + static_cast<int64_t>(length) > kMaxImageBytes) {
                too_large = true;
                return false;
            }
            for (std::size_t i = 0; i < length && signature.size() < 8; ++i)
                signature.push_back(static_cast<unsigned char>(data[i]));
            output.write(data, static_cast<std::streamsize>(length));
            if (!output) { write_failed = true; return false; }
            received += static_cast<int64_t>(length);
            return true;
        });
        output.close();

        auto remove_temp = [&]() {
            std::error_code ec;
            std::filesystem::remove(temporary, ec);
        };
        if (too_large) {
            remove_temp();
            set_json_error(res, 413, "image exceeds the 50 MiB limit");
            return;
        }
        if (!pumped || write_failed || !output || received == 0) {
            remove_temp();
            set_json_error(res, 400, "image upload was interrupted or empty");
            return;
        }
        if (detected_image_mime(signature) != mime) {
            remove_temp();
            set_json_error(res, 415,
                           "image signature does not match the declared MIME type");
            return;
        }
        attachment.size_bytes = received;
        std::error_code rename_ec;
        std::filesystem::rename(temporary, destination, rename_ec);
        if (rename_ec) {
            remove_temp();
            set_json_error(res, 500, "failed to commit managed attachment");
            return;
        }
        try {
            db.save_attachment(attachment);
        } catch (const std::exception& e) {
            std::error_code remove_ec;
            std::filesystem::remove(destination, remove_ec);
            set_json_error(res, 500, "failed to persist attachment metadata", e.what());
            return;
        }
        res.status = 201;
        res.set_content(nlohmann::json(attachment).dump(), "application/json");
    });

    server_->Get("/v1/agents/:id/attachments/:attachment_id",
        [this](const Request& req, Response& res) {
        auto agent = agents_.get_agent(req.path_params.at("id"));
        if (!agent) { set_json_error(res, 404, "agent not found"); return; }
        auto attachment = agent->db().get_attachment(req.path_params.at("attachment_id"));
        if (!attachment) { set_json_error(res, 404, "attachment not found"); return; }
        if (!set_file_response(res, agent->db().attachment_file_path(*attachment),
                               attachment->mime_type)) {
            set_json_error(res, 404, "attachment file not found");
        }
    });

    server_->Delete("/v1/agents/:id/attachments/:attachment_id",
        [this](const Request& req, Response& res) {
        auto agent = agents_.get_agent(req.path_params.at("id"));
        if (!agent) { set_json_error(res, 404, "agent not found"); return; }
        bool referenced = false;
        if (!agent->db().delete_attachment(req.path_params.at("attachment_id"), &referenced)) {
            if (referenced) set_json_error(res, 409, "attachment is referenced by a message");
            else set_json_error(res, 404, "attachment not found");
            return;
        }
        res.status = 204;
    });

    server_->Post("/v1/agents/:id/chat", [this](const Request& req, Response& res) {
        std::string agent_id = req.path_params.at("id");
        auto agent = agents_.get_agent(agent_id);
        if (!agent) { res.status = 404; return; }

        std::string message;
        std::vector<std::string> attachment_ids;
        PreparedChatTurn prepared;
        ConvId      conv_id_hint;
        int         max_tokens_override = 0;
        try {
            auto j      = nlohmann::json::parse(req.body);
            message     = j.value("message", "");
            if (j.contains("attachment_ids")) {
                if (!j.at("attachment_ids").is_array()) {
                    throw std::invalid_argument("attachment_ids must be an array");
                }
                attachment_ids = j.at("attachment_ids").get<std::vector<std::string>>();
            }
            conv_id_hint = j.value("conversation_id", "");
            max_tokens_override = j.value("max_tokens", 0);
            prepared = prepare_attachment_turn(agent->db(), message, attachment_ids);
        } catch (const std::exception& e) {
            MM_WARN("POST /v1/agents/:id/chat: invalid request: {}", e.what());
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
            return;
        }
        if (message.empty() && attachment_ids.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"message required unless attachment_ids are supplied"})", "application/json");
            return;
        }
        if (!attachment_ids.empty() && !agent->get_config().vision_settings.enabled) {
            res.status = 422;
            res.set_content(R"({"error":"this agent profile does not accept images"})",
                            "application/json");
            return;
        }

        // Shared context between worker thread and SSE content provider.
        auto ctx = std::make_shared<SseInferCtx>();

        // Chunk callback: forward InferenceChunks as SSE events to client.
        auto chunk_fn = [ctx](const InferenceChunk& chunk) {
            if (ctx->canceled) return;
            std::string payload;
            if (!chunk.tool_result_json.empty()) {
                payload = "data: " + chunk.tool_result_json + "\n\n";
            } else if (!chunk.thinking_delta.empty()) {
                payload = "data: " +
                    nlohmann::json{{"type", "thinking"},
                                   {"content", chunk.thinking_delta}}.dump() + "\n\n";
            } else if (chunk.tool_call_delta) {
                auto& tc = *chunk.tool_call_delta;
                payload = "data: " +
                    nlohmann::json{{"type",      "tool_call"},
                                   {"id",        tc.id},
                                   {"name",      tc.function_name},
                                   {"arguments", tc.arguments_json}}.dump() + "\n\n";
            } else if (!chunk.delta_content.empty()) {
                payload = "data: " +
                    nlohmann::json{{"type",    "delta"},
                                   {"content", chunk.delta_content}}.dump() + "\n\n";
            }
            if (!payload.empty()) {
                std::lock_guard<std::mutex> lk(ctx->mx);
                ctx->lines.push_back(std::move(payload));
                ctx->cv.notify_one();
            }
        };

        // Done callback: emit done event and signal provider to stop.
        auto done_sent = std::make_shared<std::atomic_bool>(false);
        auto done_fn = [ctx, done_sent](const ConvId& conv_id,
                                        bool success,
                                        const std::string& error) {
            if (done_sent->exchange(true)) return;
            nlohmann::json done_payload = {
                {"type", "done"},
                {"conv_id", conv_id},
                {"success", success},
            };
            if (!success && !error.empty()) {
                done_payload["error"] = error;
            }
            std::string payload = "data: " + done_payload.dump() + "\n\n";
            std::lock_guard<std::mutex> lk(ctx->mx);
            ctx->lines.push_back(std::move(payload));
            ctx->done = true;
            ctx->cv.notify_one();
        };

        // Enqueue job — worker thread calls handle_chat via process_fn.
        InferenceJob job;
        job.job_id   = util::generate_uuid();
        job.agent_id = agent_id;
        job.conversation_id = conv_id_hint;
        job.done_cb = [done_fn](const ConvId& conv_id, bool success) mutable {
            done_fn(conv_id, success, success ? std::string{} : "queued chat job failed");
        };
        job.process_fn = [this,
                          agent_id,
                          message,
                          content_parts = prepared.parts,
                          conv_id_hint,
                          max_tokens_override,
                          chunk_fn,
                          ctx,
                          done_for_handle = done_fn,
                          done_for_catch = done_fn]() mutable {
            try {
                handle_chat(agent_id, message, conv_id_hint,
                            std::move(chunk_fn), std::move(done_for_handle),
                            max_tokens_override, std::move(content_parts),
                            [ctx] { return ctx->canceled.load(); });
            } catch (const std::exception& e) {
                MM_ERROR("queued chat job for agent '{}' failed: {}", agent_id, e.what());
                done_for_catch(conv_id_hint, false, e.what());
            } catch (...) {
                MM_ERROR("queued chat job for agent '{}' failed with unknown exception", agent_id);
                done_for_catch(conv_id_hint, false, "queued chat job failed");
            }
        };
        queue_.enqueue(std::move(job));

        // SSE content provider: drain lines from SseInferCtx.
        res.set_chunked_content_provider("text/event-stream",
            [ctx](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::unique_lock<std::mutex> lk(ctx->mx);
                ctx->cv.wait(lk, [&] {
                    return !ctx->lines.empty() || ctx->done || ctx->canceled;
                });

                if (ctx->canceled) return false;

                while (!ctx->lines.empty()) {
                    std::string payload = std::move(ctx->lines.front());
                    ctx->lines.pop_front();
                    lk.unlock();
                    if (!sink.write(payload.data(), payload.size())) {
                        ctx->canceled = true;
                        ctx->cv.notify_all();
                        return false;
                    }
                    lk.lock();
                }

                if (ctx->done && ctx->lines.empty()) {
                    lk.unlock();
                    const std::string fin = "data: [DONE]\n\n";
                    if (!sink.write(fin.data(), fin.size())) {
                        ctx->canceled = true;
                        ctx->cv.notify_all();
                        return false;
                    }
                    sink.done();
                    return true;
                }
                return true;
            },
            [ctx](bool success) {
                if (!success) {
                    ctx->canceled = true;
                    ctx->cv.notify_all();
                }
            });
    });

    // ── GET /v1/placements ──────────────────────────────────────────────────
    server_->Get("/v1/placements", [this](const Request& /*req*/, Response& res) {
        auto placements = scheduler_.list_placements();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& p : placements) arr.push_back(nlohmann::json(p));
        res.set_content(arr.dump(), "application/json");
    });

    // ── POST /v1/agents/:id/curation/proposals ───────────────────────────────
    server_->Post("/v1/agents/:id/curation/proposals",
                  [this](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        try {
            nlohmann::json body = nlohmann::json::object();
            if (!util::trim(req.body).empty()) body = nlohmann::json::parse(req.body);
            const std::string requested_conv_id = body.value("conversation_id", std::string{});
            int max_items = body.value("max_items", 12);
            max_items = std::clamp(max_items, 1, 50);

            auto& db = a->db();
            nlohmann::json proposals = nlohmann::json::array();
            auto add = [&](nlohmann::json proposal) {
                if (static_cast<int>(proposals.size()) < max_items) {
                    proposals.push_back(std::move(proposal));
                }
            };

            std::vector<Conversation> convs = db.list_conversations();
            std::optional<Conversation> focus;
            if (!requested_conv_id.empty()) {
                focus = db.load_conversation(requested_conv_id);
                if (!focus) { res.status = 404; return; }
            } else {
                for (const auto& item : convs) {
                    if (item.is_active) {
                        focus = db.load_conversation(item.id);
                        break;
                    }
                }
                if (!focus && !convs.empty()) focus = db.load_conversation(convs.front().id);
            }

            if (focus) {
                const std::string title = util::trim(focus->title);
                const std::string prompt_preview = first_user_preview(*focus, 56);
                if (!prompt_preview.empty() &&
                    (title.empty() || title == "New Conversation" || title == "Untitled conversation")) {
                    add(curation_proposal(
                        "rename_conversation",
                        "conversation",
                        focus->id,
                        focus->id,
                        nlohmann::json{{"title", focus->title}},
                        nlohmann::json{{"title", prompt_preview}},
                        "The conversation has a generic title; the first user prompt gives it a clearer label."));
                }

                auto local_mems = db.list_local_memories(focus->id);
                if (local_mems.empty() && !prompt_preview.empty()) {
                    add(curation_proposal(
                        "create_local_memory",
                        "local_memory",
                        "",
                        focus->id,
                        nlohmann::json::object(),
                        nlohmann::json{{"content", prompt_preview}},
                        "No local memory exists for the selected conversation; this captures the main prompt."));
                }
                for (const auto& mem : local_mems) {
                    const std::string trimmed = util::trim(mem.content);
                    if (trimmed.empty()) {
                        add(curation_proposal(
                            "delete_local_memory",
                            "local_memory",
                            mem.id,
                            focus->id,
                            nlohmann::json(mem),
                            nlohmann::json::object(),
                            "This local memory is empty."));
                    } else if (trimmed != mem.content) {
                        add(curation_proposal(
                            "update_local_memory",
                            "local_memory",
                            mem.id,
                            focus->id,
                            nlohmann::json(mem),
                            nlohmann::json{{"content", trimmed}},
                            "This local memory has leading or trailing whitespace."));
                    }
                }
            }

            for (const auto& conv_meta : convs) {
                if (static_cast<int>(proposals.size()) >= max_items) break;
                if (conv_meta.is_active) continue;
                auto conv = db.load_conversation(conv_meta.id);
                if (conv && conv->messages.empty()) {
                    add(curation_proposal(
                        "delete_conversation",
                        "conversation",
                        conv->id,
                        conv->id,
                        nlohmann::json(*conv),
                        nlohmann::json::object(),
                        "This inactive conversation has no messages."));
                }
            }

            auto memories = db.list_memories();
            if (memories.empty() && focus) {
                const std::string prompt_preview = first_user_preview(*focus, 96);
                if (!prompt_preview.empty()) {
                    add(curation_proposal(
                        "create_global_memory",
                        "global_memory",
                        "",
                        focus->id,
                        nlohmann::json::object(),
                        nlohmann::json{
                            {"content", "Conversation '" + focus->title + "' covers: " + prompt_preview},
                            {"importance", 0.5}
                        },
                        "No global summary exists yet; this creates a traceable summary for future conversations."));
                }
            }
            for (const auto& mem : memories) {
                if (static_cast<int>(proposals.size()) >= max_items) break;
                const std::string trimmed = util::trim(mem.content);
                if (trimmed.empty()) {
                    add(curation_proposal(
                        "delete_global_memory",
                        "global_memory",
                        mem.id,
                        mem.source_conv_id,
                        nlohmann::json(mem),
                        nlohmann::json::object(),
                        "This global memory is empty."));
                } else if (trimmed != mem.content) {
                    add(curation_proposal(
                        "update_global_memory",
                        "global_memory",
                        mem.id,
                        mem.source_conv_id,
                        nlohmann::json(mem),
                        nlohmann::json{{"content", trimmed}, {"importance", mem.importance}},
                        "This global memory has leading or trailing whitespace."));
                }
            }

            res.set_content(nlohmann::json{{"proposals", proposals}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /v1/agents/:id/curation/apply
    auto apply_curation_proposals =
        [this](const Request& req, Response& res) {
        const std::string id = req.path_params.at("id");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        try {
            nlohmann::json body = nlohmann::json::object();
            if (!util::trim(req.body).empty()) body = nlohmann::json::parse(req.body);

            const nlohmann::json* proposals = nullptr;
            if (body.is_array()) {
                proposals = &body;
            } else if (body.contains("proposals")) {
                proposals = &body.at("proposals");
            } else if (body.contains("approved_proposals")) {
                proposals = &body.at("approved_proposals");
            }

            if (!proposals || !proposals->is_array()) {
                res.status = 400;
                res.set_content(
                    nlohmann::json{{"error", "proposals must be an array"}}.dump(),
                    "application/json");
                return;
            }
            if (proposals->empty()) {
                res.status = 400;
                res.set_content(
                    nlohmann::json{{"error", "at least one proposal is required"}}.dump(),
                    "application/json");
                return;
            }

            auto& db = a->db();
            std::vector<CurationApplyPlan> plans;
            plans.reserve(proposals->size());
            for (std::size_t i = 0; i < proposals->size(); ++i) {
                const auto& proposal = proposals->at(i);
                try {
                    plans.push_back(validate_curation_apply_proposal(db, proposal));
                } catch (const std::exception& e) {
                    nlohmann::json rejection = {
                        {"error", "invalid curation proposal"},
                        {"index", i},
                        {"reason", e.what()}
                    };
                    if (proposal.is_object() && proposal.contains("id") &&
                        proposal.at("id").is_string()) {
                        rejection["proposal_id"] = proposal.at("id").get<std::string>();
                    }
                    res.status = 400;
                    res.set_content(rejection.dump(), "application/json");
                    return;
                }
            }

            nlohmann::json results = nlohmann::json::array();
            for (const auto& plan : plans) {
                results.push_back(apply_curation_plan(db, plan));
            }

            res.set_content(
                nlohmann::json{
                    {"status", "applied"},
                    {"applied_count", results.size()},
                    {"results", results}
                }.dump(),
                "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    };
    server_->Post("/v1/agents/:id/curation/apply", apply_curation_proposals);
    server_->Post("/v1/agents/:id/curation/proposals/apply", apply_curation_proposals);

    // GET /v1/agents/:id/conversations
    server_->Get("/v1/agents/:id/conversations",
                 [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        auto convs = a->db().list_conversations();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& c : convs) arr.push_back(nlohmann::json(c));
        res.set_content(arr.dump(), "application/json");
    });

    // ── POST /v1/agents/:id/conversations ─────────────────────────────────────
    server_->Post("/v1/agents/:id/conversations",
                  [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        try {
            nlohmann::json j = nlohmann::json::object();
            if (!util::trim(req.body).empty()) {
                j = nlohmann::json::parse(req.body);
            }

            const std::string title = j.value("title", std::string{});
            const std::string parent_conv_id = j.value("parent_conv_id", std::string{});
            const bool set_active = j.value("set_active", j.value("activate", false));

            if (!parent_conv_id.empty() && !a->db().conversation_exists(parent_conv_id)) {
                res.status = 400;
                res.set_content(
                    nlohmann::json{{"error", "parent conversation not found"}}.dump(),
                    "application/json");
                return;
            }

            ConvId cid = a->db().create_conversation(title, parent_conv_id);
            if (cid.empty()) {
                res.status = 400;
                res.set_content(
                    nlohmann::json{{"error", "failed to create conversation"}}.dump(),
                    "application/json");
                return;
            }

            if (set_active) {
                // Trigger global recall on the previously active conversation
                auto prev_active = a->db().get_active_conversation_id();
                if (prev_active && *prev_active != cid) {
                    queue_global_recall(id, *prev_active);
                }
                a->db().set_active_conversation(cid);
            }

            auto conv_opt = a->db().load_conversation(cid);
            if (!conv_opt) {
                res.status = 500;
                res.set_content(
                    nlohmann::json{{"error", "conversation was created but could not be loaded"}}.dump(),
                    "application/json");
                return;
            }

            res.status = 201;
            res.set_content(nlohmann::json{{"conversation", *conv_opt}}.dump(),
                            "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── GET /v1/agents/:id/conversations/:cid ─────────────────────────────────
    server_->Get("/v1/agents/:id/conversations/:cid",
                 [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        auto conv_opt = a->db().load_conversation(cid);
        if (!conv_opt) { res.status = 404; return; }
        res.set_content(nlohmann::json(*conv_opt).dump(), "application/json");
    });

    // ── PUT /v1/agents/:id/conversations/:cid ────────────────────────────────
    server_->Put("/v1/agents/:id/conversations/:cid",
                 [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }

        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string title = util::trim(j.value("title", std::string{}));
            if (title.empty()) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "title is required"}}.dump(),
                                "application/json");
                return;
            }
            a->db().rename_conversation(cid, title);
            auto conv_opt = a->db().load_conversation(cid);
            if (!conv_opt) { res.status = 404; return; }
            res.set_content(nlohmann::json(*conv_opt).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── POST /v1/agents/:id/conversations/:cid/activate ──────────────────────
    server_->Post("/v1/agents/:id/conversations/:cid/activate",
                  [this](const Request& req, Response& res) {
        (void)req.body;
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }

        // Trigger global recall on the previously active conversation
        auto prev_active = a->db().get_active_conversation_id();
        if (prev_active && *prev_active != cid) {
            queue_global_recall(id, *prev_active);
        }

        a->db().set_active_conversation(cid);
        res.set_content(
            nlohmann::json{{"status", "ok"}, {"active_conversation_id", cid}}.dump(),
            "application/json");
    });

    // ── POST /v1/agents/:id/conversations/:cid/compact ───────────────────────
    server_->Post("/v1/agents/:id/conversations/:cid/compact",
                  [this](const Request& req, Response& res) {
        (void)req.body;
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");

        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }

        auto cfg = a->get_config();
        if (agent_uses_api_backend(cfg)) {
            try {
                auto api_runtime = make_api_runtime_client(cfg);
                ConversationManager conv_mgr(a->db(), *api_runtime);
                ConvId new_id = conv_mgr.force_compact(
                    cid, cfg,
                    [this] { return queue_.cancellation_requested(); });

                res.set_content(
                    nlohmann::json{
                        {"status", "ok"},
                        {"old_conversation_id", cid},
                        {"new_conversation_id", new_id}
                    }.dump(),
                    "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                                "application/json");
            }
            return;
        }

        auto sched_result = scheduler_.ensure_agent_running(cfg);
        if (!sched_result) {
            std::string err = scheduler_.last_error();
            if (err.empty()) err = "no node available or model failed to load";
            res.status = 503;
            res.set_content(nlohmann::json{{"error", err}}.dump(), "application/json");
            return;
        }

        scheduler_.mark_agent_active(id);
        auto mark_idle = [&]() { scheduler_.mark_agent_idle(id); };

        try {
            NodeProxyRuntimeClient real_runtime(
                registry_.operations(sched_result->node_id), sched_result->slot_id,
                [this] { return queue_.cancellation_requested(); });
            ConversationManager conv_mgr(a->db(), real_runtime);
            ConvId new_id = conv_mgr.force_compact(
                cid, cfg,
                [this] { return queue_.cancellation_requested(); });
            mark_idle();

            res.set_content(
                nlohmann::json{
                    {"status", "ok"},
                    {"old_conversation_id", cid},
                    {"new_conversation_id", new_id}
                }.dump(),
                "application/json");
        } catch (const std::exception& e) {
            mark_idle();
            res.status = 500;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── DELETE /v1/agents/:id/conversations/:cid ──────────────────────────────
    server_->Delete("/v1/agents/:id/conversations/:cid",
                    [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }
        if (a->db().is_conversation_active(cid)) {
            res.status = 409;
            res.set_content(
                nlohmann::json{
                    {"error", "cannot delete active conversation; activate another conversation first"}
                }.dump(),
                "application/json");
            return;
        }
        a->db().delete_conversation(cid);
        res.set_content(R"({"status":"deleted"})", "application/json");
    });

    // ── GET /v1/agents/:id/conversations/:cid/local-memories ─────────────────
    server_->Get("/v1/agents/:id/conversations/:cid/local-memories",
                 [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }

        auto mems = a->db().list_local_memories(cid);
        nlohmann::json arr = nlohmann::json::array();
        for (auto& m : mems) arr.push_back(nlohmann::json(m));
        res.set_content(arr.dump(), "application/json");
    });

    // ── POST /v1/agents/:id/conversations/:cid/local-memories ────────────────
    server_->Post("/v1/agents/:id/conversations/:cid/local-memories",
                  [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }

        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string content = util::trim(j.value("content", std::string{}));
            if (content.empty()) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "content is required"}}.dump(),
                                "application/json");
                return;
            }
            LocalMemory mem;
            mem.id = util::generate_uuid();
            mem.conversation_id = cid;
            mem.content = content;
            a->db().add_local_memory(mem);
            auto saved = a->db().get_local_memory(mem.id);
            res.status = 201;
            res.set_content(nlohmann::json(saved ? *saved : mem).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── PUT /v1/agents/:id/conversations/:cid/local-memories/:mid ────────────
    server_->Put("/v1/agents/:id/conversations/:cid/local-memories/:mid",
                 [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        std::string mid = req.path_params.at("mid");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }

        auto existing = a->db().get_local_memory(mid);
        if (!existing || existing->conversation_id != cid) { res.status = 404; return; }

        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string content = util::trim(j.value("content", std::string{}));
            if (content.empty()) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "content is required"}}.dump(),
                                "application/json");
                return;
            }
            LocalMemory mem = *existing;
            mem.content = content;
            a->db().update_local_memory(mem);
            auto saved = a->db().get_local_memory(mid);
            res.set_content(nlohmann::json(saved ? *saved : mem).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── DELETE /v1/agents/:id/conversations/:cid/local-memories/:mid ─────────
    server_->Delete("/v1/agents/:id/conversations/:cid/local-memories/:mid",
                    [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string cid = req.path_params.at("cid");
        std::string mid = req.path_params.at("mid");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        if (!a->db().conversation_exists(cid)) { res.status = 404; return; }
        auto existing = a->db().get_local_memory(mid);
        if (!existing || existing->conversation_id != cid) { res.status = 404; return; }
        a->db().delete_local_memory(mid);
        res.set_content(R"({"status":"deleted"})", "application/json");
    });

    // ── POST /v1/agents/:id/memories/extract ──────────────────────────────────
    server_->Post("/v1/agents/:id/memories/extract",
                  [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string conv_id = j.value("conversation_id", std::string{});
            const int start_index = j.value("start_index", -1);
            const int end_index = j.value("end_index", -1);
            int context_before = j.value("context_before", 2);
            context_before = std::clamp(context_before, 0, 20);

            if (conv_id.empty()) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "conversation_id is required"}}.dump(),
                                "application/json");
                return;
            }
            if (start_index < 0 || end_index < 0 || start_index > end_index) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "invalid message range"}}.dump(),
                                "application/json");
                return;
            }

            auto conv_opt = a->db().load_conversation(conv_id);
            if (!conv_opt) { res.status = 404; return; }

            const auto& msgs = conv_opt->messages;
            if (msgs.empty()) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "conversation has no messages"}}.dump(),
                                "application/json");
                return;
            }
            if (end_index >= static_cast<int>(msgs.size())) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "message range out of bounds"}}.dump(),
                                "application/json");
                return;
            }

            const int begin_index = std::max(0, start_index - context_before);
            std::vector<Message> selected(
                msgs.begin() + begin_index,
                msgs.begin() + end_index + 1);

            InferenceJob job;
            job.job_id   = util::generate_uuid();
            job.agent_id = id;
            job.process_fn = [this, id, conv_id, selected]() {
                const auto canceled = [this] {
                    return queue_.cancellation_requested();
                };
                if (canceled()) return;
                auto a_local = agents_.get_agent(id);
                if (!a_local) return;

                AgentConfig cfg = a_local->get_config();
                if (agent_uses_api_backend(cfg)) {
                    if (canceled()) return;
                    try {
                        auto api_runtime = make_api_runtime_client(cfg);
                        MemoryManager mem_mgr(a_local->db(), *api_runtime);
                        mem_mgr.extract_and_store_memories_from_messages(
                            conv_id, selected, cfg, canceled);
                    } catch (const std::exception& e) {
                        MM_WARN("Manual memory extraction failed for '{}': {}", id, e.what());
                    }
                    return;
                }

                if (canceled()) return;
                auto sched_result = scheduler_.ensure_agent_running(cfg);
                if (!sched_result) {
                    MM_WARN("Manual memory extraction: no node for agent '{}': {}",
                            id, scheduler_.last_error());
                    return;
                }

                scheduler_.mark_agent_active(id);
                SchedulerIdleGuard idle_guard;
                idle_guard.arm(scheduler_, id);

                try {
                    NodeProxyRuntimeClient ext_runtime(
                        registry_.operations(sched_result->node_id),
                        sched_result->slot_id, canceled);
                    MemoryManager mem_mgr(a_local->db(), ext_runtime);
                    if (!canceled()) {
                        mem_mgr.extract_and_store_memories_from_messages(
                            conv_id, selected, cfg, canceled);
                    }
                } catch (const std::exception& e) {
                    MM_WARN("Manual memory extraction failed for '{}': {}", id, e.what());
                }
            };
            queue_.enqueue(std::move(job));

            res.set_content(
                nlohmann::json{
                    {"status", "queued"},
                    {"conversation_id", conv_id},
                    // Count the messages actually sent (the requested range plus
                    // the context_before lead-in), matching `selected`.
                    {"selected_count", static_cast<int>(selected.size())}
                }.dump(),
                "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── GET /v1/agents/:id/memories ───────────────────────────────────────────
    server_->Get("/v1/agents/:id/memories",
                 [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        auto mems = a->db().list_memories();
        nlohmann::json arr = nlohmann::json::array();
        for (auto& m : mems) arr.push_back(nlohmann::json(m));
        res.set_content(arr.dump(), "application/json");
    });

    // ── POST /v1/agents/:id/memories ──────────────────────────────────────────
    server_->Post("/v1/agents/:id/memories",
                  [this](const Request& req, Response& res) {
        std::string id = req.path_params.at("id");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }

        try {
            auto j = nlohmann::json::parse(req.body);
            const std::string content = util::trim(j.value("content", std::string{}));
            if (content.empty()) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "content is required"}}.dump(),
                                "application/json");
                return;
            }
            const std::string source_conv_id = j.value("source_conv_id", std::string{});
            if (!source_conv_id.empty() && !a->db().conversation_exists(source_conv_id)) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "source conversation not found"}}.dump(),
                                "application/json");
                return;
            }

            Memory mem;
            mem.id = util::generate_uuid();
            mem.agent_id = id;
            mem.content = content;
            mem.source_conv_id = source_conv_id;
            mem.importance = j.value("importance", 0.5f);
            a->db().add_memory(mem);
            auto saved = a->db().get_memory(mem.id);
            res.status = 201;
            res.set_content(nlohmann::json(saved ? *saved : mem).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── PUT /v1/agents/:id/memories/:mid ──────────────────────────────────────
    server_->Put("/v1/agents/:id/memories/:mid",
                 [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string mid = req.path_params.at("mid");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        try {
            auto j = nlohmann::json::parse(req.body);
            auto existing = a->db().get_memory(mid);
            if (!existing) { res.status = 404; return; }
            Memory mem = *existing;
            mem.content    = util::trim(j.value("content", mem.content));
            mem.importance = j.value("importance", mem.importance);
            if (mem.content.empty()) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error", "content is required"}}.dump(),
                                "application/json");
                return;
            }
            a->db().update_memory(mem);
            auto saved = a->db().get_memory(mid);
            res.set_content(nlohmann::json(saved ? *saved : mem).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(),
                            "application/json");
        }
    });

    // ── DELETE /v1/agents/:id/memories/:mid ───────────────────────────────────
    server_->Delete("/v1/agents/:id/memories/:mid",
                    [this](const Request& req, Response& res) {
        std::string id  = req.path_params.at("id");
        std::string mid = req.path_params.at("mid");
        auto a = agents_.get_agent(id);
        if (!a) { res.status = 404; return; }
        a->db().delete_memory(mid);
        res.set_content(R"({"status":"deleted"})", "application/json");
    });
}

} // namespace mm
