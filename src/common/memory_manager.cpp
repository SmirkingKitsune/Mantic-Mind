#include "common/memory_manager.hpp"
#include "common/agent_db.hpp"
#include "common/llama_cpp_client.hpp"
#include "common/util.hpp"
#include "common/logger.hpp"

#include <nlohmann/json.hpp>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace mm {

static constexpr double kMemoryDecayMs  = 30.0 * 24 * 3600 * 1000; // 30 days in ms
static constexpr double kBaseWeight     = 0.7;
static constexpr double kRecencyWeight  = 0.3;

namespace {

std::string build_transcript(const std::vector<Message>& messages) {
    std::ostringstream transcript;
    for (auto& m : messages) {
        if (m.role == MessageRole::System) continue;
        transcript << (m.role == MessageRole::User ? "User" : "Assistant")
                   << ": " << m.content << "\n";
    }
    return transcript.str();
}

} // namespace

MemoryManager::MemoryManager(AgentDB& db, LlamaCppClient& llama)
    : db_(db), llama_(llama) {}

// ── Memory extraction ─────────────────────────────────────────────────────────
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

    std::string transcript = build_transcript(messages);
    if (transcript.empty()) return;

    std::string extraction_prompt =
        "Analyse the conversation below and extract a list of important, "
        "reusable facts about the user — preferences, goals, background, "
        "decisions, or other details worth remembering for future conversations.\n\n"
        "Return ONLY a valid JSON array. Each element must be an object with:\n"
        "  \"content\"    (string)  — the fact, written in third person\n"
        "  \"importance\" (number)  — 0.0 (trivial) to 1.0 (critical)\n\n"
        "If there is nothing worth remembering, return an empty array [].\n\n"
        "Conversation:\n" + transcript + "\n\nJSON:";

    InferenceRequest req;
    req.settings           = cfg.llama_settings;
    req.settings.max_tokens = 512;
    req.settings.temperature = 0.2f;  // low temperature for structured output
    req.messages = {{ .role = MessageRole::User, .content = extraction_prompt }};

    Message resp = llama_.complete(req);
    if (resp.content.empty()) {
        MM_WARN("Memory extraction returned empty response for conv {}", conv_id);
        return;
    }

    // Parse JSON — find the first '[' in the response (model may prepend text).
    auto pos = resp.content.find('[');
    if (pos == std::string::npos) {
        MM_WARN("Memory extraction: no JSON array found in response");
        return;
    }
    std::string json_str = resp.content.substr(pos);
    // Trim after the last ']'
    auto end = json_str.rfind(']');
    if (end != std::string::npos) json_str = json_str.substr(0, end + 1);

    try {
        auto arr = nlohmann::json::parse(json_str);
        if (!arr.is_array()) return;

        int stored = 0;
        for (auto& item : arr) {
            if (!item.contains("content")) continue;
            std::string content = item["content"].get<std::string>();
            if (content.empty()) continue;
            float importance = item.value("importance", 0.5f);
            importance = std::clamp(importance, 0.0f, 1.0f);

            Memory mem;
            mem.id             = util::generate_uuid();
            mem.agent_id       = cfg.id;
            mem.content        = content;
            mem.source_conv_id = conv_id;
            mem.importance     = importance;
            mem.created_at_ms  = util::now_ms();
            db_.add_memory(mem);
            ++stored;
        }
        MM_INFO("Extracted {} memories from selected messages in conv {}", stored, conv_id);
    } catch (const std::exception& e) {
        MM_WARN("Memory extraction JSON parse error: {}", e.what());
    }
}

// ── Memory retrieval ──────────────────────────────────────────────────────────
std::vector<Memory> MemoryManager::get_relevant_memories(const AgentConfig& cfg,
                                                          int max) const {
    if (!cfg.memories_enabled) return {};

    // Retrieve ranked by importance DESC, recency DESC (DB handles ordering).
    auto all = db_.list_memories();

    // Score = importance * recency_factor where recency decays over 30 days.
    int64_t now = util::now_ms();
    const double decay_ms = kMemoryDecayMs;

    for (auto& m : all) {
        double age   = static_cast<double>(now - m.created_at_ms);
        double decay = std::exp(-age / decay_ms);  // 0→1, 1 = brand new
        // Store combined score back into importance for sorting purposes
        // (importance is not persisted here — this is ephemeral)
        m.importance = static_cast<float>(m.importance * (kBaseWeight + kRecencyWeight * decay));
    }

    std::stable_sort(all.begin(), all.end(),
        [](const Memory& a, const Memory& b){ return a.importance > b.importance; });

    if (static_cast<int>(all.size()) > max)
        all.resize(static_cast<size_t>(max));
    return all;
}

// ── Formatting ────────────────────────────────────────────────────────────────
std::string MemoryManager::format_memories_for_context(
    const std::vector<Memory>& memories)
{
    if (memories.empty()) return {};
    std::ostringstream oss;
    oss << "## Remembered facts about the user\n";
    for (auto& m : memories)
        oss << "- " << m.content << "\n";
    return oss.str();
}

} // namespace mm
