#pragma once

// Mantic-Mind — making a conversation's history readable by the model that
// answers the next turn.
//
// A conversation outlives the model it started with. History written while a
// vision model answered holds images; history written while a tool-calling
// model answered holds tool calls and their results. Sent unchanged to a model
// that has neither, it fails EVERY later turn, not just one: Soma refuses tool
// history with a 422, and llama.cpp without a projector refuses the image. So
// history is rewritten — per request, never in storage — into what this model
// can read:
//
//   no vision  each image becomes a text placeholder naming its file
//   no tools   each tool round (the calls, their results, and the reply that
//              followed) becomes ONE assistant message, the calls and results
//              written out as text
//
// Nothing disappears without a trace: the placeholder says an image was there,
// and the folded text says which tool was called with what, and what came back.

#include "common/engine_capabilities.hpp"
#include "common/models.hpp"

#include <functional>
#include <string>
#include <vector>

namespace mm {

/// The name to show for an attachment in its placeholder; "" when unknown.
using AttachmentNamer = std::function<std::string(const std::string& attachment_id)>;

/// `messages` rewritten for a model with `caps`. Messages the model can read
/// as they are come back unchanged.
std::vector<Message> sanitize_history(std::vector<Message> messages,
                                      const ModelCapabilities& caps,
                                      const AttachmentNamer& name_of = {});

/// "[image omitted: cat.png]", or "[image omitted]" without a name.
std::string image_placeholder(const std::string& name);

} // namespace mm
