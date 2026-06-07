#include "common/trace_provenance.hpp"

#include "common/agent_db.hpp"
#include "common/util.hpp"

namespace mm {

TraceEvent make_trace_event(const std::string& category,
                            const std::string& title,
                            const std::string& detail,
                            const std::string& source_id,
                            nlohmann::json metadata) {
    TraceEvent event;
    event.id = util::generate_uuid();
    event.type = category;
    event.category = category;
    event.title = title;
    event.detail = detail;
    event.timestamp_ms = util::now_ms();
    event.source_id = source_id;
    event.metadata = std::move(metadata);
    return event;
}

void assign_trace_sequences(std::vector<TraceEvent>& events) {
    for (int i = 0; i < static_cast<int>(events.size()); ++i) {
        events[i].sequence = i;
        if (!events[i].timestamp_ms) events[i].timestamp_ms = util::now_ms();
    }
}

std::vector<TraceEvent> build_context_trace_events(AgentDB& db,
                                                   const ConvId& conv_id,
                                                   const std::vector<Memory>& memories) {
    std::vector<TraceEvent> events;

    auto conv = db.load_conversation(conv_id);
    if (conv) {
        if (!conv->parent_conv_id.empty()) {
            events.push_back(make_trace_event(
                "conversation",
                "Parent conversation accessed",
                conv->parent_conv_id,
                conv->parent_conv_id,
                {{"conversation_id", conv->id}}));
        }
        if (!conv->compaction_summary.empty()) {
            const std::string source_conv_id =
                conv->parent_conv_id.empty() ? conv->id : conv->parent_conv_id;
            events.push_back(make_trace_event(
                "conversation",
                "Compaction summary reviewed",
                conv->compaction_summary,
                source_conv_id,
                {
                    {"conversation_id", conv->id},
                    {"parent_conv_id", conv->parent_conv_id}
                }));
        }
    }

    for (const auto& local : db.list_local_memories(conv_id)) {
        events.push_back(make_trace_event(
            "local-memory",
            "Conversation-local memory reviewed",
            local.content,
            local.id,
            {{"conversation_id", local.conversation_id}}));
    }

    for (const auto& memory : memories) {
        events.push_back(make_trace_event(
            "global-memory",
            "Global memory reviewed",
            memory.content,
            memory.id,
            {
                {"source_conv_id", memory.source_conv_id},
                {"importance", memory.importance}
            }));
    }

    assign_trace_sequences(events);
    return events;
}

std::optional<TraceEvent> build_tool_access_trace(const ToolCall& tc,
                                                  const Message& tool_result,
                                                  const ConvId& conv_id) {
    nlohmann::json args = nlohmann::json::object();
    try {
        if (!tc.arguments_json.empty()) args = nlohmann::json::parse(tc.arguments_json);
    } catch (...) {
        args = nlohmann::json::object();
    }

    nlohmann::json metadata = {
        {"tool_call_id", tc.id},
        {"function_name", tc.function_name},
        {"arguments", args}
    };

    if (tc.function_name == "list_local_memories") {
        return make_trace_event(
            "local-memory",
            "Conversation-local memories listed",
            tool_result.content,
            conv_id,
            metadata);
    }
    if (tc.function_name == "get_global_memory_origin") {
        const std::string memory_id = args.value("memory_id", std::string{});
        return make_trace_event(
            "global-memory",
            "Global memory origin accessed",
            tool_result.content,
            memory_id,
            metadata);
    }
    if (tc.function_name == "list_conversations") {
        return make_trace_event(
            "conversation",
            "Conversation list accessed",
            tool_result.content,
            {},
            metadata);
    }
    if (tc.function_name == "get_conversation_summary") {
        const std::string source_conv_id = args.value("conversation_id", std::string{});
        return make_trace_event(
            "conversation",
            "Conversation summary accessed",
            tool_result.content,
            source_conv_id,
            metadata);
    }

    return std::nullopt;
}

} // namespace mm
