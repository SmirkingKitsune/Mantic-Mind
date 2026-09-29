#include "control/api_support.hpp"

#include "common/conversation_context.hpp"
#include "common/http_client.hpp"
#include "common/logger.hpp"
#include "common/served_model.hpp"
#include "common/user_db.hpp"
#include "common/util.hpp"

#include <httplib.h>
#include <openssl/evp.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mm::api {

namespace {

std::string base64_encode(const std::vector<unsigned char>& bytes) {
    if (bytes.empty()) return {};
    std::string encoded(4 * ((bytes.size() + 2) / 3), '\0');
    const int length = EVP_EncodeBlock(reinterpret_cast<unsigned char*>(encoded.data()),
                                       bytes.data(), static_cast<int>(bytes.size()));
    if (length < 0) throw std::runtime_error("base64 encoding failed");
    encoded.resize(static_cast<std::size_t>(length));
    return encoded;
}

std::vector<unsigned char> read_attachment_bytes(const std::string& path,
                                                 std::int64_t expected_size) {
    if (expected_size < 0 || expected_size > kMaxImageBytes)
        throw std::runtime_error("stored attachment exceeds the image size limit");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("stored attachment file is unavailable");
    std::vector<unsigned char> bytes;
    bytes.resize(static_cast<std::size_t>(expected_size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (input.gcount() != static_cast<std::streamsize>(bytes.size()))
            throw std::runtime_error("stored attachment size does not match metadata");
    }
    char extra = 0;
    if (input.read(&extra, 1)) throw std::runtime_error("stored attachment size does not match metadata");
    return bytes;
}

} // namespace

// ── Requests and errors ───────────────────────────────────────────────────────

void set_json_error(httplib::Response& res,
                    int status,
                    const std::string& error,
                    const std::string& detail) {
    res.status = status;
    nlohmann::json body{{"error", error}};
    if (!detail.empty()) body["detail"] = detail;
    res.set_content(body.dump(), "application/json");
}

void set_json_error_code(httplib::Response& res,
                         int status,
                         const std::string& error,
                         const std::string& code) {
    res.status = status;
    nlohmann::json body{{"error", error}};
    if (!code.empty()) body["code"] = code;
    res.set_content(body.dump(), "application/json");
}

nlohmann::json parse_request_json(const httplib::Request& req) {
    if (util::trim(req.body).empty()) return nlohmann::json::object();
    auto j = nlohmann::json::parse(req.body);
    if (!j.is_object()) throw std::invalid_argument("request body must be a JSON object");
    return j;
}

std::string extract_json_object_text(const std::string& text) {
    std::string trimmed = util::trim(text);
    if (trimmed.rfind("```", 0) == 0) {
        const auto first_newline = trimmed.find('\n');
        const auto last_fence = trimmed.rfind("```");
        if (first_newline != std::string::npos && last_fence != std::string::npos &&
            last_fence > first_newline) {
            trimmed = util::trim(trimmed.substr(first_newline + 1, last_fence - first_newline - 1));
        }
    }
    const auto first = trimmed.find('{');
    const auto last = trimmed.rfind('}');
    if (first == std::string::npos || last == std::string::npos || last <= first)
        throw std::invalid_argument("model did not return a JSON object");
    return trimmed.substr(first, last - first + 1);
}

std::string sha256_hex(const std::string& text) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) throw std::runtime_error("failed to create SHA-256 context");
    const bool ok = EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(ctx, text.data(), text.size()) == 1 &&
                    EVP_DigestFinal_ex(ctx, digest, &digest_len) == 1;
    EVP_MD_CTX_free(ctx);
    if (!ok) throw std::runtime_error("failed to compute SHA-256");
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < digest_len; ++i) out << std::setw(2) << static_cast<int>(digest[i]);
    return out.str();
}

// ── Files ─────────────────────────────────────────────────────────────────────

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
    const std::string content_type = mime_type.empty() ? "application/octet-stream" : mime_type;
    res.set_chunked_content_provider(content_type,
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

// ── Images ────────────────────────────────────────────────────────────────────

std::string normalized_image_mime(std::string mime) {
    const auto semicolon = mime.find(';');
    if (semicolon != std::string::npos) mime.resize(semicolon);
    mime = util::to_lower(util::trim(mime));
    if (mime == "image/jpg") mime = "image/jpeg";
    return mime;
}

std::string detected_image_mime(const std::vector<unsigned char>& bytes) {
    static constexpr unsigned char png[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
    if (bytes.size() >= sizeof(png) && std::equal(std::begin(png), std::end(png), bytes.begin()))
        return "image/png";
    if (bytes.size() >= 3 && bytes[0] == 0xff && bytes[1] == 0xd8 && bytes[2] == 0xff)
        return "image/jpeg";
    return {};
}

std::string safe_attachment_filename(std::string name, const std::string& mime_type) {
    name = std::filesystem::path(name).filename().string();
    name.erase(std::remove_if(name.begin(), name.end(),
                              [](unsigned char ch) { return ch < 0x20 || ch == 0x7f; }),
               name.end());
    if (name.empty()) name = mime_type == "image/png" ? "image.png" : "image.jpg";
    if (name.size() > 255) name.resize(255);
    return name;
}

PreparedChatTurn prepare_attachment_turn(UserDB& db,
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
    if (attachment_ids.size() > static_cast<std::size_t>(kMaxContextImages))
        throw std::invalid_argument("a turn may contain at most " +
                                    std::to_string(kMaxContextImages) + " images");
    for (const auto& id : attachment_ids) {
        auto attachment = db.get_attachment(id);
        if (!attachment) throw std::invalid_argument("attachment not found for this user: " + id);
        turn.decoded_bytes += attachment->size_bytes;
        if (turn.decoded_bytes > kMaxContextImageBytes)
            throw std::invalid_argument("decoded images exceed the 400 MiB turn limit");
        MessageContentPart part;
        part.type = "image_attachment";
        part.attachment_id = id;
        part.mime_type = attachment->mime_type;
        turn.parts.push_back(std::move(part));
        ++turn.image_count;
    }
    return turn;
}

void hydrate_attachment_parts(UserDB& db,
                              std::vector<Message>& messages,
                              int* image_count,
                              std::int64_t* decoded_bytes) {
    int count = 0;
    std::int64_t total = 0;
    for (auto& message : messages) {
        for (auto& part : message.content_parts) {
            if (part.type != "image_attachment") continue;
            auto attachment = db.get_attachment(part.attachment_id);
            if (!attachment) throw std::runtime_error("conversation references a missing attachment");
            ++count;
            total += attachment->size_bytes;
            if (count > kMaxContextImages)
                throw std::runtime_error("hydrated context exceeds the per-request image limit");
            if (total > kMaxContextImageBytes)
                throw std::runtime_error("hydrated context exceeds the 400 MiB decoded-image limit");
            auto bytes = read_attachment_bytes(db.attachment_file_path(*attachment),
                                               attachment->size_bytes);
            if (detected_image_mime(bytes) != attachment->mime_type)
                throw std::runtime_error("stored attachment failed signature validation");
            part.type = "image_url";
            part.image_url = "data:" + attachment->mime_type + ";base64," + base64_encode(bytes);
            part.attachment_id.clear();
        }
    }
    if (image_count) *image_count = count;
    if (decoded_bytes) *decoded_bytes = total;
}

// ── Speech ────────────────────────────────────────────────────────────────────

bool is_supported_tts_format(const std::string& format) {
    const std::string f = util::to_lower(util::trim(format));
    return f.empty() || f == "wav";
}

std::string tts_mime_type_for_format(const std::string& format) {
    const std::string f = util::to_lower(util::trim(format));
    if (f == "wav" || f.empty()) return "audio/wav";
    return "application/octet-stream";
}

std::filesystem::path user_tts_dir(const std::string& data_dir,
                                   const std::string& user_id,
                                   const std::string& category) {
    return std::filesystem::path(data_dir) / "users" / user_id / "tts" / category;
}

std::filesystem::path user_tts_cache_dir(const TtsServiceConfig& config,
                                         const std::string& data_dir,
                                         const std::string& user_id) {
    if (util::trim(config.cache_dir).empty()) return user_tts_dir(data_dir, user_id, "cache");
    return std::filesystem::path(config.cache_dir) / user_id;
}

// ── Model clients ─────────────────────────────────────────────────────────────

std::unique_ptr<RuntimeClient> make_api_runtime_client(const ApiSettings& settings) {
    return std::make_unique<RuntimeClient>(settings.base_url, resolve_api_key(settings),
                                           settings.chat_completions_path);
}

NodeProxyRuntimeClient::NodeProxyRuntimeClient(std::string node_url,
                                               std::string api_key,
                                               std::string slot_id)
    : RuntimeClient(node_url),
      node_url_(std::move(node_url)),
      api_key_(std::move(api_key)),
      slot_id_(std::move(slot_id)) {}

Message NodeProxyRuntimeClient::complete(const InferenceRequest& req) {
    HttpClient cli(node_url_);
    cli.set_bearer_token(api_key_);

    Message result;
    result.role = MessageRole::Assistant;
    result.timestamp_ms = util::now_ms();

    nlohmann::json body = nlohmann::json(req);
    if (!slot_id_.empty()) body["slot_id"] = slot_id_;

    bool failed = false;
    const bool ok = cli.stream_post("/api/node/infer", body, [&](const std::string& data) -> bool {
        if (data == "[DONE]") return true;
        try {
            auto j = nlohmann::json::parse(data);
            const std::string type = j.value("type", "");
            if (type == "delta") {
                result.content += j.value("content", "");
            } else if (type == "thinking") {
                result.thinking_text += j.value("content", "");
            } else if (type == "tool_call") {
                ToolCall tc;
                tc.id = j.value("id", std::string{});
                tc.function_name = j.value("name", "");
                tc.arguments_json = j.value("arguments", "");
                result.tool_calls.push_back(tc);
            } else if (type == "error") {
                failed = true;
            } else if (type == "done") {
                result.token_count = j.value("tokens_used", 0);
            }
        } catch (const std::exception& e) {
            MM_WARN("NodeProxyRuntimeClient::complete: parse error: {}", e.what());
        }
        return true;
    });

    // A failure is an empty, non-assistant message — exactly what
    // RuntimeClient::complete returns for one — so a caller cannot mistake an
    // engine error for an empty answer.
    if (!ok || failed) {
        MM_WARN("NodeProxyRuntimeClient::complete: completion failed on {}", node_url_);
        return Message{};
    }
    return result;
}

bool contains_prefill_thinking_incompatibility(const std::string& text) {
    return util::to_lower(text).find(
               "assistant response prefill is incompatible with enable_thinking") !=
           std::string::npos;
}

void enforce_prefill_safe_tail(std::vector<Message>& messages) {
    if (messages.empty() || messages.back().role != MessageRole::Assistant) return;
    Message m;
    m.role = MessageRole::User;
    m.content = "Continue with the next response based on the latest context.";
    m.timestamp_ms = util::now_ms();
    messages.push_back(std::move(m));
}

} // namespace mm::api
