#include "common/memory_manager.hpp"
#include "common/agent_db.hpp"
#include "common/llama_cpp_client.hpp"
#include "common/logger.hpp"
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
std::string build_transcript(const std::vector<Message>& messages) {
    std::ostringstream transcript;
    for (const auto& m : messages) {
        if (m.role == MessageRole::System) continue;
        transcript << role_label(m.role) << ": " << m.content << "\n";
    }
    return transcript.str();
}

double rank_score(const Memory& memory, int64_t now_ms) {
    const double age_ms = static_cast<double>(std::max<int64_t>(0, now_ms - memory.created_at_ms));
    const double decay = std::exp(-age_ms / kMemoryDecayMs);
    return static_cast<double>(memory.importance) * (kBaseWeight + kRecencyWeight * decay);
}

std::vector<Memory> rank_candidate_memories(std::vector<Memory> memories, int max_candidates) {
    const int64_t now_ms = util::now_ms();
    std::stable_sort(memories.begin(), memories.end(),
        [now_ms](const Memory& lhs, const Memory& rhs) {
            return rank_score(lhs, now_ms) > rank_score(rhs, now_ms);
        });

    if (max_candidates > 0 && static_cast<int>(memories.size()) > max_candidates) {
        memories.resize(static_cast<size_t>(max_candidates));
    }
    return memories;
}

std::string build_relevance_prompt(AgentDB& db,
                                   const ConvId& conv_id,
                                   const AgentConfig& cfg,
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
              "## System prompt\n"
           << (cfg.system_prompt.empty() ? "(empty)" : cfg.system_prompt) << "\n\n";

    auto conv = db.load_conversation(conv_id);
    if (!conv) {
        prompt << "## Active conversation context\n(conversation not found)\n\n";
    } else {
        prompt << "## Active conversation context\n";
        if (!conv->compaction_summary.empty()) {
            prompt << "[Previous conversation summary]\n"
                   << conv->compaction_summary << "\n\n";
        }

        std::vector<Message> recent_messages;
        for (const auto& message : conv->messages) {
            if (message.role == MessageRole::System) continue;
            recent_messages.push_back(message);
        }
        if (static_cast<int>(recent_messages.size()) > kContextPreviewMessages) {
            recent_messages.erase(recent_messages.begin(),
                                  recent_messages.end() - kContextPreviewMessages);
        }

        if (recent_messages.empty()) {
            prompt << "(no messages yet)\n\n";
        } else {
            for (const auto& message : recent_messages) {
                prompt << role_label(message.role) << ": " << message.content << "\n";
            }
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
    for (const auto& item : *ids_it) {
        if (!item.is_string()) continue;
        ids.push_back(item.get<std::string>());
    }
    return ids;
}

std::vector<Memory> fallback_selected_memories(const std::vector<Memory>& candidates,
                                               int max_selected) {
    const int fallback_count = std::min({static_cast<int>(candidates.size()),
                                         max_selected,
                                         kFallbackSelectedMemories});
    return std::vector<Memory>(candidates.begin(), candidates.begin() + fallback_count);
}

} // namespace

MemoryManager::MemoryManager(AgentDB& db, LlamaCppClient& llama)
    : db_(db), llama_(llama) {}

void MemoryManager::extract_and_store_memories(const ConvId& conv_id,
                                               const AgentConfig& cfg) {
    if (!cfg.memories_enabled) return;

    auto messages = db_.load_messages(conv_id);
    extract_and_store_memories_from_messages(conv_id, messages, cfg);
}

void MemoryManager::extract_and_store_memories_from_messages(
    const ConvId& conv_id,
    const std::vector<Message>& messages,
    const AgentConfig& cfg) {
    if (messages.empty()) return;

    const std::string transcript = build_transcript(messages);
    if (transcript.empty()) return;

    const std::string extraction_prompt =
        "Analyze the conversation below and extract important reusable information "
        "worth remembering for future conversations. The memories may be about a "
        "person, agent, project, world state, task, preference, decision, or other "
        "durable context.\n\n"
        "Return ONLY a valid JSON array. Each element must be an object with:\n"
        "  \"content\"    (string)  the memory, written as a concise durable fact\n"
        "  \"importance\" (number)  0.0 (trivial) to 1.0 (critical)\n\n"
        "If there is nothing worth remembering, return an empty array [].\n\n"
        "Conversation:\n" + transcript + "\n\nJSON:";

    InferenceRequest req;
    req.model = cfg.model_path;
    req.settings = cfg.llama_settings;
    req.settings.max_tokens = 512;
    req.settings.temperature = 0.2f;
    req.messages = {{.role = MessageRole::User, .content = extraction_prompt}};

    Message resp = llama_.complete(req);
    if (resp.content.empty()) {
        MM_WARN("Memory extraction returned empty response for conv {}", conv_id);
        return;
    }

    auto pos = resp.content.find('[');
    if (pos == std::string::npos) {
        MM_WARN("Memory extraction: no JSON array found in response");
        return;
    }
    std::string json_str = resp.content.substr(pos);
    auto end = json_str.rfind(']');
    if (end != std::string::npos) json_str = json_str.substr(0, end + 1);

    try {
        auto arr = nlohmann::json::parse(json_str);
        if (!arr.is_array()) return;

        int stored = 0;
        for (const auto& item : arr) {
            if (!item.contains("content")) continue;
            const std::string content = item["content"].get<std::string>();
            if (content.empty()) continue;
            float importance = item.value("importance", 0.5f);
            importance = std::clamp(importance, 0.0f, 1.0f);

            Memory mem;
            mem.id = util::generate_uuid();
            mem.agent_id = cfg.id;
            mem.content = content;
            mem.source_conv_id = conv_id;
            mem.importance = importance;
            mem.created_at_ms = util::now_ms();
            db_.add_memory(mem);
            ++stored;
        }
        MM_INFO("Extracted {} memories from selected messages in conv {}", stored, conv_id);
    } catch (const std::exception& e) {
        MM_WARN("Memory extraction JSON parse error: {}", e.what());
    }
}

std::vector<Memory> MemoryManager::get_relevant_memories(const ConvId& conv_id,
                                                         const AgentConfig& cfg,
                                                         int max_candidates,
                                                         int max_selected) const {
    if (!cfg.memories_enabled || max_candidates <= 0 || max_selected <= 0) return {};

    auto candidates = rank_candidate_memories(db_.list_memories(), max_candidates);
    if (candidates.empty()) return {};

    InferenceRequest req;
    req.model = cfg.model_path;
    req.settings = cfg.llama_settings;
    req.settings.max_tokens = 256;
    req.settings.temperature = 0.1f;
    req.messages = {{.role = MessageRole::User,
                     .content = build_relevance_prompt(db_, conv_id, cfg, candidates, max_selected)}};

    Message response = llama_.complete(req);
    if (response.content.empty()) {
        MM_WARN("Relevant memory selection returned empty response for conv {}", conv_id);
        return fallback_selected_memories(candidates, max_selected);
    }

    auto selected_ids = parse_selected_memory_ids(response.content);
    if (!selected_ids) {
        MM_WARN("Relevant memory selection returned invalid JSON for conv {}", conv_id);
        return fallback_selected_memories(candidates, max_selected);
    }

    std::unordered_set<std::string> selected_id_set(selected_ids->begin(), selected_ids->end());
    std::vector<Memory> selected;
    selected.reserve(static_cast<size_t>(max_selected));
    for (const auto& memory : candidates) {
        if (!selected_id_set.contains(memory.id)) continue;
        selected.push_back(memory);
        if (static_cast<int>(selected.size()) >= max_selected) break;
    }

    return selected;
}

std::string MemoryManager::format_memories_for_context(const std::vector<Memory>& memories) {
    if (memories.empty()) return {};

    std::ostringstream oss;
    oss << "## Remembered from other conversations\n";
    for (const auto& memory : memories) {
        oss << "- " << memory.content << "\n";
    }
    return oss.str();
}

} // namespace mm
