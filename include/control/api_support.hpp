#pragma once

// Mantic-Mind — helpers shared by the control API's translation units: the
// server itself (control_api_server.cpp), the user routes (user_routes.cpp)
// and the chat pipeline (chat_pipeline.cpp).
//
// One definition each. They were file-local to control_api_server.cpp while it
// was the only file with routes; splitting the user surface out of a 5000-line
// file would otherwise have meant copying them, and two copies of the image
// limits or of the error body is how a client comes to see two shapes.

#include "common/models.hpp"
#include "common/runtime_client.hpp"
#include "control/tts_service_client.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace httplib {
struct Request;
struct Response;
} // namespace httplib

namespace mm {

class UserDB;

namespace api {

/// One uploaded image.
inline constexpr std::int64_t kMaxImageBytes = 50LL * 1024 * 1024;
/// An upload no message references is removed after this long.
inline constexpr std::int64_t kPendingAttachmentTtlMs = 24LL * 60 * 60 * 1000;

// ── Requests and errors ───────────────────────────────────────────────────────

/// `{"error": ..., "detail": ...}` — the error body of the whole :9090 surface.
void set_json_error(httplib::Response& res,
                    int status,
                    const std::string& error,
                    const std::string& detail = {});

/// The same body with a machine-readable `code` a client can branch on
/// (`model_required`, `model_not_found`, `model_unavailable`, …).
void set_json_error_code(httplib::Response& res,
                         int status,
                         const std::string& error,
                         const std::string& code);

/// An empty body is `{}`; anything but an object throws std::invalid_argument.
nlohmann::json parse_request_json(const httplib::Request& req);

/// The JSON object inside a model's answer, fences and prose around it removed.
/// Throws std::invalid_argument when there is none.
std::string extract_json_object_text(const std::string& text);

std::string sha256_hex(const std::string& text);

// ── Files ─────────────────────────────────────────────────────────────────────

bool path_file_exists(const std::string& path);
void remove_file_quietly(const std::string& path);
/// Streams `path` as the response body. False when it cannot be opened.
bool set_file_response(httplib::Response& res,
                       const std::string& path,
                       const std::string& mime_type);

// ── Images ────────────────────────────────────────────────────────────────────

std::string normalized_image_mime(std::string mime);
/// From the file's own signature: "image/png", "image/jpeg", or "".
std::string detected_image_mime(const std::vector<unsigned char>& bytes);
std::string safe_attachment_filename(std::string name, const std::string& mime_type);

/// A user turn's content parts: the text, then each attachment.
struct PreparedChatTurn {
    std::string message;
    std::vector<MessageContentPart> parts;
    int image_count = 0;
    std::int64_t decoded_bytes = 0;
};

/// Throws std::invalid_argument for an unknown attachment or a turn over the
/// per-request image limits.
PreparedChatTurn prepare_attachment_turn(UserDB& db,
                                         const std::string& message,
                                         const std::vector<std::string>& attachment_ids);

/// Replaces each stored image part with an inline data URL, re-checking the
/// file's signature and the per-request limits. Throws std::runtime_error.
void hydrate_attachment_parts(UserDB& db,
                              std::vector<Message>& messages,
                              int* image_count,
                              std::int64_t* decoded_bytes);

// ── Speech ────────────────────────────────────────────────────────────────────

bool is_supported_tts_format(const std::string& format);
std::string tts_mime_type_for_format(const std::string& format);
/// {data_dir}/users/{id}/tts/{category}: removed with the user.
std::filesystem::path user_tts_dir(const std::string& data_dir,
                                   const std::string& user_id,
                                   const std::string& category);
/// Where synthesized speech is cached: under the user, unless the TTS config
/// names a cache directory, in which case {cache_dir}/{user}.
std::filesystem::path user_tts_cache_dir(const TtsServiceConfig& config,
                                         const std::string& data_dir,
                                         const std::string& user_id);

// ── Model clients ─────────────────────────────────────────────────────────────

/// A client for an API-backed served model: its base URL and path, and its key
/// from process memory or the environment.
std::unique_ptr<RuntimeClient> make_api_runtime_client(const ApiSettings& settings);

/// Routes completions through a node's /api/node/infer: the engine is bound to
/// localhost on its node and control cannot reach it directly.
class NodeProxyRuntimeClient : public RuntimeClient {
public:
    NodeProxyRuntimeClient(std::string node_url, std::string api_key, std::string slot_id = {});
    Message complete(const InferenceRequest& req) override;

private:
    std::string node_url_;
    std::string api_key_;
    std::string slot_id_;
};

/// Some templates refuse a trailing assistant message while thinking is on;
/// the retry path appends a user turn when the engine says so.
bool contains_prefill_thinking_incompatibility(const std::string& text);
void enforce_prefill_safe_tail(std::vector<Message>& messages);

} // namespace api
} // namespace mm
