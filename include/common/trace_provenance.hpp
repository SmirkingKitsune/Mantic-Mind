#pragma once

#include "common/models.hpp"

#include <optional>
#include <vector>

namespace mm {

class AgentDB;

TraceEvent make_trace_event(const std::string& category,
                            const std::string& title,
                            const std::string& detail,
                            const std::string& source_id = {},
                            nlohmann::json metadata = nlohmann::json::object());

void assign_trace_sequences(std::vector<TraceEvent>& events);

std::vector<TraceEvent> build_context_trace_events(AgentDB& db,
                                                   const ConvId& conv_id,
                                                   const std::vector<Memory>& memories);

std::optional<TraceEvent> build_tool_access_trace(const ToolCall& tc,
                                                  const Message& tool_result,
                                                  const ConvId& conv_id);

} // namespace mm
