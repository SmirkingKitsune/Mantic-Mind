#pragma once

#include "common/models.hpp"
#include <string>
#include <vector>

namespace mm {

class AgentDB;

// Server-side executor for agent memory tool calls.
// Provides local tools (per-conversation) and global tools (cross-conversation).
class ToolExecutor {
public:
    explicit ToolExecutor(AgentDB& db);

    // Tool descriptions available to the model/UI when local tools are enabled.
    static std::vector<ToolDefinition> local_tool_catalog();

    // Tool descriptions including global tools used during global recall flows.
    static std::vector<ToolDefinition> all_tool_catalog();

    // Local-only tools (used during normal chat)
    std::vector<ToolDefinition> get_local_tool_definitions() const;

    // All tools including global (used during global recall phase)
    std::vector<ToolDefinition> get_all_tool_definitions() const;

    // Execute a tool call and return the result as a Tool message.
    Message execute_tool(const ToolCall& call, const ConvId& conv_id);

    // Returns true if the function name is a memory tool handled by this executor.
    bool is_memory_tool(const std::string& function_name) const;

private:
    AgentDB& db_;

    // Individual tool handlers — return JSON result string
    std::string handle_save_local_memory(const nlohmann::json& args, const ConvId& conv_id);
    std::string handle_update_local_memory(const nlohmann::json& args);
    std::string handle_delete_local_memory(const nlohmann::json& args);
    std::string handle_list_local_memories(const ConvId& conv_id);

    std::string handle_save_global_memory(const nlohmann::json& args, const ConvId& conv_id);
    std::string handle_update_global_memory(const nlohmann::json& args);
    std::string handle_delete_global_memory(const nlohmann::json& args);
    std::string handle_list_conversations(const nlohmann::json& args);
    std::string handle_get_conversation_summary(const nlohmann::json& args);
};

} // namespace mm
