#include "common/memory_manager.hpp"
#include "common/conversation_context.hpp"
#include "common/logger.hpp"
#include "common/runtime_client.hpp"
#include "common/user_db.hpp"
#include "common/util.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace mm {

static constexpr double kMemoryDecayMs = 30.0 * 24 * 3600 * 1000; // 30 days in ms
static constexpr double kBaseWeight = 0.7;
static constexpr double kRecencyWeight = 0.3;
static constexpr int kContextPreviewMessages = 8;
static constexpr int kFallbackSelectedMemories = 3;
static constexpr int kExtractionMaxTokens = 512;

namespace {

std::string role_label(MessageRole role) {
    switch (role) {
        case MessageRole::System: return "System";
        case MessageRole::User: return "User";
        case MessageRole::Assistant: return "Assistant";
        case MessageRole::Tool: return "Tool";
    }
    return "Unknown";
}

/// The most recent messages that fit `max_tokens`, oldest first. A transcript
/// that no longer fits the model is cut from the front: the latest exchange is
/// what an extraction or a relevance check is about.
std::string build_transcript(const std::vector<Message>& messages, int max_tokens) {
    std::vector<std::string> lines;
    int used = 0;
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (it->role == MessageRole::System) continue;
        std::string line = role_label(it->role) + ": " + it->content + "\n";
        const int cost = estimate_tokens(line);
        if (!lines.empty() && used + cost > max_tokens) break;
        used += cost;
        lines.push_back(std::move(line));
    }
    std::string transcript;
    for (auto it = lines.rbegin(); it != lines.rend(); ++it) transcript += *it;
    return transcript;
}

double rank_score(const Memory& memory, int64_t now_ms) {
    const double age_ms = static_cast<double>(std::max<int64_t>(0, now_ms - memory.created_at_ms));
    const double decay = std::exp(-age_ms / kMemoryDecayMs);
    return static_cast<double>(memory.importance) * (kBaseWeight + kRecencyWeight * decay);
}

std::vector<Memory> rank_candidate_memories(std::vector<Memory> memories, int max_candidates) {
    const int64_t now_ms = util::now_ms();
    std::stable_sort(memories.begin(), memories.end(), [now_ms](const Memory& lhs, const Memory& rhs) {
        return rank_score(lhs, now_ms) > rank_score(rhs, now_ms);
    });
    if (max_candidates > 0 && static_cast<int>(memories.size()) > max_candidates)
        memories.resize(static_cast<size_t>(max_candidates));
    return memories;
}

std::string build_relevance_prompt(UserDB& db,
                                   const ConvId& conv_id,
                                   const std::string& instructions,
                                   const std::vector<Memory>& candidates,
                                   int max_selected) {
    std::ostringstream prompt;
    prompt << "Review the current conversation and decide which stored global memories "
              "from other conversations are relevant to the next reply.\n\n"
              "Select only memories that materially help with the next response. "
              "Ignore memories that are merely related or generally interesting.\n\n"
              "Return ONLY a valid JSON object with this shape:\n"
              "{\"memory_ids\": [\"id-1\", \"id-2\"]}\n"
              "Use an empty array if none are relevant. Do not return more than "
           << max_selected << " ids.\n\n"
           << "## Instructions the reply is written under\n"
           << (util::trim(instructions).empty() ? std::string("(none)") : instructions) << "\n\n";

    const auto conv = db.get_conversation(conv_id);
    prompt << "## Active conversation context\n";
    if (!conv) {
        prompt << "(conversation not found)\n\n";
    } else {
        if (!util::trim(conv->summary).empty())
            prompt << "[Summary of earlier messages]\n" << conv->summary << "\n\n";
        auto recent = db.load_messages(conv_id, conv->summary_through_seq);
        recent.erase(std::remove_if(recent.begin(), recent.end(),
                                    [](const Message& m) { return m.role == MessageRole::System; }),
                     recent.end());
        if (static_cast<int>(recent.size()) > kContextPreviewMessages)
            recent.erase(recent.begin(), recent.end() - kContextPreviewMessages);
        if (recent.empty()) {
            prompt << "(no messages yet)\n\n";
        } else {
            for (const auto& message : recent)
                prompt << role_label(message.role) << ": " << message.content << "\n";
            prompt << "\n";
        }
    }

    prompt << "## Candidate global memories\n";
    for (const auto& memory : candidates) {
        prompt << "- [" << memory.id << "] importance=" << memory.importance
               << " content=" << memory.content << "\n";
    }
    return prompt.str();
}

std::optional<std::vector<std::string>> parse_selected_memory_ids(const std::string& response_text) {
    const auto object_pos = response_text.find('{');
    if (object_pos == std::string::npos) return std::nullopt;
    std::string json_text = response_text.substr(object_pos);
    const auto object_end = json_text.rfind('}');
    if (object_end == std::string::npos) return std::nullopt;
    json_text.resize(object_end + 1);

    auto parsed = nlohmann::json::parse(json_text, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return std::nullopt;
    const auto ids_it = parsed.find("memory_ids");
    if (ids_it == parsed.end() || !ids_it->is_array()) return std::nullopt;

    std::vector<std::string> ids;
    for (const auto& item : *ids_it)
        if (item.is_string()) ids.push_back(item.get<std::string>());
    return ids;
}

std::vector<Memory> fallback_selected_memories(const std::vector<Memory>& candidates, int max_selected) {
    const int fallback_count =
        std::min({static_cast<int>(candidates.size()), max_selected, kFallbackSelectedMemories});
    return std::vector<Memory>(candidates.begin(), candidates.begin() + fallback_count);
}

} // namespace

MemoryManager::MemoryManager(UserDB& db,
                             RuntimeClient& runtime,
                             std::string request_model,
                             RuntimeSettings settings)
    : db_(db), runtime_(runtime), request_model_(std::move(request_model)), settings_(std::move(settings)) {}

int MemoryManager::extract_and_store_memories_from_messages(const ConvId& conv_id,
                                                            const std::vector<Message>& messages) {
    if (messages.empty()) return 0;

    static const std::string kInstructions =
        "Analyze the conversation below and create concise global conversation-chain "
        "summaries for future conversations. Each memory should summarize what this "
        "conversation chain contains and why it may matter later. Do not copy short-lived "
        "scratch notes; preserve durable context that helps another conversation decide "
        "whether to inspect this source conversation.\n\n"
        "Return ONLY a valid JSON array. Each element must be an object with:\n"
        "  \"content\"    (string)  a concise source-grounded summary\n"
        "  \"importance\" (number)  0.0 (trivial) to 1.0 (critical)\n\n"
        "If there is nothing worth remembering, return an empty array [].\n\n"
        "Conversation:\n";

    RuntimeSettings settings = settings_;
    settings.max_tokens = kExtractionMaxTokens;
    settings.temperature = 0.2f;
    // Sized for the model doing the extracting: a transcript that fits a large
    // model is cut, not sent whole, when a small one is asked.
    const int room = prompt_budget(settings) - estimate_tokens(kInstructions) - 64;
    const std::string transcript = build_transcript(messages, std::max(room, 64));
    if (transcript.empty()) return 0;

    InferenceRequest req;
    req.model = request_model_;
    req.settings = settings;
    req.messages = {{.role = MessageRole::User, .content = kInstructions + transcript + "\n\nJSON:"}};

    const Message resp = runtime_.complete(req);
    if (resp.role != MessageRole::Assistant || resp.content.empty()) {
        MM_WARN("Memory extraction failed for conv {}", conv_id);
        return -1;
    }

    const auto pos = resp.content.find('[');
    if (pos == std::string::npos) {
        MM_WARN("Memory extraction: no JSON array found in response");
        return -1;
    }
    std::string json_str = resp.content.substr(pos);
    const auto end = json_str.rfind(']');
    if (end != std::string::npos) json_str = json_str.substr(0, end + 1);

    try {
        const auto arr = nlohmann::json::parse(json_str);
        if (!arr.is_array()) return -1;
        int stored = 0;
        for (const auto& item : arr) {
            if (!item.is_object() || !item.contains("content") || !item["content"].is_string()) continue;
            const std::string content = util::trim(item["content"].get<std::string>());
            if (content.empty()) continue;
            float importance = item.contains("importance") && item["importance"].is_number()
                                   ? item["importance"].get<float>()
                                   : 0.5f;
            importance = std::clamp(importance, 0.0f, 1.0f);

            Memory mem;
            mem.id = util::generate_uuid();
            mem.content = content;
            mem.source_conv_id = conv_id;
            mem.importance = importance;
            mem.created_at_ms = util::now_ms();
            db_.add_memory(mem);
            ++stored;
        }
        MM_INFO("Extracted {} memories from conversation {}", stored, conv_id);
        return stored;
    } catch (const std::exception& e) {
        MM_WARN("Memory extraction JSON parse error: {}", e.what());
        return -1;
    }
}

std::vector<Memory> MemoryManager::get_relevant_memories(const ConvId& conv_id,
                                                         const std::string& instructions,
                                                         int max_candidates,
                                                         int max_selected) const {
    if (max_candidates <= 0 || max_selected <= 0) return {};

    const auto candidates = rank_candidate_memories(db_.list_memories(), max_candidates);
    if (candidates.empty()) return {};

    InferenceRequest req;
    req.model = request_model_;
    req.settings = settings_;
    req.settings.max_tokens = 256;
    req.settings.temperature = 0.1f;
    req.messages = {{.role = MessageRole::User,
                     .content = build_relevance_prompt(db_, conv_id, instructions, candidates,
                                                       max_selected)}};

    const Message response = runtime_.complete(req);
    if (response.role != MessageRole::Assistant || response.content.empty()) {
        MM_WARN("Relevant memory selection failed for conv {}", conv_id);
        return fallback_selected_memories(candidates, max_selected);
    }

    const auto selected_ids = parse_selected_memory_ids(response.content);
    if (!selected_ids) {
        MM_WARN("Relevant memory selection returned invalid JSON for conv {}", conv_id);
        return fallback_selected_memories(candidates, max_selected);
    }

    const std::unordered_set<std::string> selected_id_set(selected_ids->begin(), selected_ids->end());
    std::vector<Memory> selected;
    selected.reserve(static_cast<size_t>(max_selected));
    for (const auto& memory : candidates) {
        if (!selected_id_set.contains(memory.id)) continue;
        selected.push_back(memory);
        if (static_cast<int>(selected.size()) >= max_selected) break;
    }
    return selected;
}

} // namespace mm
