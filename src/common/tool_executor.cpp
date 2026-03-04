#include "common/tool_executor.hpp"
#include "common/agent_db.hpp"
#include "common/util.hpp"
#include "common/logger.hpp"

#include <nlohmann/json.hpp>

namespace mm {

ToolExecutor::ToolExecutor(AgentDB& db) : db_(db) {}

// ── Tool definitions ──────────────────────────────────────────────────────────

std::vector<ToolDefinition> ToolExecutor::local_tool_catalog() {
    return {
        {
            "save_local_memory",
            "Save a note for this conversation. Use this to remember important information, "
            "decisions, or context that you want to refer back to later in this conversation.",
            {{"type", "object"},
             {"properties", {
                 {"content", {{"type", "string"}, {"description", "The content to remember"}}}
             }},
             {"required", {"content"}}}
        },
        {
            "update_local_memory",
            "Update an existing local memory by ID.",
            {{"type", "object"},
             {"properties", {
                 {"id", {{"type", "string"}, {"description", "The memory ID to update"}}},
                 {"content", {{"type", "string"}, {"description", "The new content"}}}
             }},
             {"required", {"id", "content"}}}
        },
        {
            "delete_local_memory",
            "Delete a local memory by ID.",
            {{"type", "object"},
             {"properties", {
                 {"id", {{"type", "string"}, {"description", "The memory ID to delete"}}}
             }},
             {"required", {"id"}}}
        },
        {
            "list_local_memories",
            "List all local memories for this conversation.",
            {{"type", "object"}, {"properties", nlohmann::json::object()}}
        }
    };
}

std::vector<ToolDefinition> ToolExecutor::all_tool_catalog() {
    auto tools = local_tool_catalog();

    // Global memory tools
    tools.push_back({
        "save_global_memory",
        "Save a cross-conversation memory. Use this to persist important information "
        "that should be available in all future conversations.",
        {{"type", "object"},
         {"properties", {
             {"content", {{"type", "string"}, {"description", "The content to remember globally"}}},
             {"importance", {{"type", "number"}, {"description", "Importance score 0.0-1.0 (default 0.5)"},
                            {"minimum", 0.0}, {"maximum", 1.0}}}
         }},
         {"required", {"content"}}}
    });

    tools.push_back({
        "update_global_memory",
        "Update an existing global memory by ID.",
        {{"type", "object"},
         {"properties", {
             {"id", {{"type", "string"}, {"description", "The global memory ID to update"}}},
             {"content", {{"type", "string"}, {"description", "The new content"}}},
             {"importance", {{"type", "number"}, {"description", "New importance score 0.0-1.0"},
                            {"minimum", 0.0}, {"maximum", 1.0}}}
         }},
         {"required", {"id", "content"}}}
    });

    tools.push_back({
        "delete_global_memory",
        "Delete a global memory by ID.",
        {{"type", "object"},
         {"properties", {
             {"id", {{"type", "string"}, {"description", "The global memory ID to delete"}}}
         }},
         {"required", {"id"}}}
    });

    tools.push_back({
        "list_conversations",
        "List past conversations for review.",
        {{"type", "object"},
         {"properties", {
             {"limit", {{"type", "integer"}, {"description", "Max conversations to return (default 10)"},
                       {"minimum", 1}, {"maximum", 50}}}
         }}}
    });

    tools.push_back({
        "get_conversation_summary",
        "Get details of a conversation including its local memories.",
        {{"type", "object"},
         {"properties", {
             {"conversation_id", {{"type", "string"}, {"description", "The conversation ID to review"}}}
         }},
         {"required", {"conversation_id"}}}
    });

    return tools;
}

std::vector<ToolDefinition> ToolExecutor::get_local_tool_definitions() const {
    return local_tool_catalog();
}

std::vector<ToolDefinition> ToolExecutor::get_all_tool_definitions() const {
    return all_tool_catalog();
}

// ── Tool dispatch ─────────────────────────────────────────────────────────────

bool ToolExecutor::is_memory_tool(const std::string& function_name) const {
    return function_name == "save_local_memory"
        || function_name == "update_local_memory"
        || function_name == "delete_local_memory"
        || function_name == "list_local_memories"
        || function_name == "save_global_memory"
        || function_name == "update_global_memory"
        || function_name == "delete_global_memory"
        || function_name == "list_conversations"
        || function_name == "get_conversation_summary";
}

Message ToolExecutor::execute_tool(const ToolCall& call, const ConvId& conv_id) {
    Message result;
    result.role         = MessageRole::Tool;
    result.tool_call_id = call.id;
    result.timestamp_ms = util::now_ms();

    try {
        nlohmann::json args;
        if (!call.arguments_json.empty()) {
            args = nlohmann::json::parse(call.arguments_json);
        }

        std::string output;
        if (call.function_name == "save_local_memory") {
            output = handle_save_local_memory(args, conv_id);
        } else if (call.function_name == "update_local_memory") {
            output = handle_update_local_memory(args);
        } else if (call.function_name == "delete_local_memory") {
            output = handle_delete_local_memory(args);
        } else if (call.function_name == "list_local_memories") {
            output = handle_list_local_memories(conv_id);
        } else if (call.function_name == "save_global_memory") {
            output = handle_save_global_memory(args, conv_id);
        } else if (call.function_name == "update_global_memory") {
            output = handle_update_global_memory(args);
        } else if (call.function_name == "delete_global_memory") {
            output = handle_delete_global_memory(args);
        } else if (call.function_name == "list_conversations") {
            output = handle_list_conversations(args);
        } else if (call.function_name == "get_conversation_summary") {
            output = handle_get_conversation_summary(args);
        } else {
            output = nlohmann::json{{"error", "unknown tool: " + call.function_name}}.dump();
        }

        result.content = output;
    } catch (const std::exception& e) {
        MM_WARN("ToolExecutor::execute_tool: error in '{}': {}", call.function_name, e.what());
        result.content = nlohmann::json{{"error", e.what()}}.dump();
    }

    return result;
}

// ── Local memory handlers ─────────────────────────────────────────────────────

std::string ToolExecutor::handle_save_local_memory(const nlohmann::json& args,
                                                    const ConvId& conv_id) {
    std::string content = args.at("content").get<std::string>();
    LocalMemory mem;
    mem.id              = util::generate_uuid();
    mem.conversation_id = conv_id;
    mem.content         = content;
    db_.add_local_memory(mem);
    return nlohmann::json{{"status", "saved"}, {"id", mem.id}}.dump();
}

std::string ToolExecutor::handle_update_local_memory(const nlohmann::json& args) {
    std::string id      = args.at("id").get<std::string>();
    std::string content = args.at("content").get<std::string>();

    auto existing = db_.get_local_memory(id);
    if (!existing) {
        return nlohmann::json{{"error", "local memory not found: " + id}}.dump();
    }

    LocalMemory mem = *existing;
    mem.content = content;
    db_.update_local_memory(mem);
    return nlohmann::json{{"status", "updated"}, {"id", id}}.dump();
}

std::string ToolExecutor::handle_delete_local_memory(const nlohmann::json& args) {
    std::string id = args.at("id").get<std::string>();
    db_.delete_local_memory(id);
    return nlohmann::json{{"status", "deleted"}, {"id", id}}.dump();
}

std::string ToolExecutor::handle_list_local_memories(const ConvId& conv_id) {
    auto memories = db_.list_local_memories(conv_id);
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& m : memories) {
        arr.push_back({{"id", m.id}, {"content", m.content},
                       {"created_at_ms", m.created_at_ms}});
    }
    return nlohmann::json{{"memories", arr}, {"count", arr.size()}}.dump();
}

// ── Global memory handlers ────────────────────────────────────────────────────

std::string ToolExecutor::handle_save_global_memory(const nlohmann::json& args,
                                                     const ConvId& conv_id) {
    std::string content = args.at("content").get<std::string>();
    float importance = args.value("importance", 0.5f);

    Memory mem;
    mem.id             = util::generate_uuid();
    mem.content        = content;
    mem.source_conv_id = conv_id;
    mem.importance     = importance;
    db_.add_memory(mem);
    return nlohmann::json{{"status", "saved"}, {"id", mem.id}}.dump();
}

std::string ToolExecutor::handle_update_global_memory(const nlohmann::json& args) {
    std::string id      = args.at("id").get<std::string>();
    std::string content = args.at("content").get<std::string>();

    auto existing = db_.get_memory(id);
    if (!existing) {
        return nlohmann::json{{"error", "global memory not found: " + id}}.dump();
    }

    Memory mem = *existing;
    mem.content = content;
    if (args.contains("importance")) {
        mem.importance = args.at("importance").get<float>();
    }
    db_.update_memory(mem);
    return nlohmann::json{{"status", "updated"}, {"id", id}}.dump();
}

std::string ToolExecutor::handle_delete_global_memory(const nlohmann::json& args) {
    std::string id = args.at("id").get<std::string>();
    db_.delete_memory(id);
    return nlohmann::json{{"status", "deleted"}, {"id", id}}.dump();
}

std::string ToolExecutor::handle_list_conversations(const nlohmann::json& args) {
    int limit = args.value("limit", 10);
    auto convs = db_.list_conversations();

    nlohmann::json arr = nlohmann::json::array();
    int count = 0;
    for (const auto& c : convs) {
        if (count >= limit) break;
        arr.push_back({
            {"id", c.id},
            {"title", c.title},
            {"is_active", c.is_active},
            {"total_tokens", c.total_tokens},
            {"created_at_ms", c.created_at_ms},
            {"has_compaction_summary", !c.compaction_summary.empty()}
        });
        ++count;
    }
    return nlohmann::json{{"conversations", arr}, {"count", arr.size()}}.dump();
}

std::string ToolExecutor::handle_get_conversation_summary(const nlohmann::json& args) {
    std::string conv_id = args.at("conversation_id").get<std::string>();

    auto conv = db_.load_conversation(conv_id);
    if (!conv) {
        return nlohmann::json{{"error", "conversation not found: " + conv_id}}.dump();
    }

    // Include local memories for this conversation
    auto local_mems = db_.list_local_memories(conv_id);
    nlohmann::json mems_arr = nlohmann::json::array();
    for (const auto& m : local_mems) {
        mems_arr.push_back({{"id", m.id}, {"content", m.content}});
    }

    // Build a brief summary of messages
    nlohmann::json msg_summary = nlohmann::json::array();
    for (const auto& m : conv->messages) {
        nlohmann::json entry = {
            {"role", to_string(m.role)},
            {"content_preview", m.content.substr(0, 200)}
        };
        if (m.content.size() > 200) entry["truncated"] = true;
        msg_summary.push_back(entry);
    }

    return nlohmann::json{
        {"conversation_id", conv_id},
        {"title", conv->title},
        {"total_tokens", conv->total_tokens},
        {"message_count", conv->messages.size()},
        {"messages", msg_summary},
        {"compaction_summary", conv->compaction_summary},
        {"local_memories", mems_arr}
    }.dump();
}

} // namespace mm
