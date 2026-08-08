#include "common/engine_capabilities.hpp"

namespace mm {

bool engine_supports_vision(const std::string& engine_id) noexcept {
    // Deliberately a closed list rather than a registry lookup. Control does not
    // link the node's EngineRegistry, and threading per-engine capabilities
    // through node health polling would make a static property of the software
    // depend on a node being reachable — so an agent's image would be accepted
    // or refused based on whether a health poll had landed yet.
    if (engine_id == "llama-cpp") return true;
    if (engine_id == "soma") return false;
    return false; // unknown: refuse, see the header
}

std::string image_refusal_for(bool profile_vision_enabled,
                              const std::string& engine_id,
                              const std::string& routing_reason) {
    // Profile first: it is the operator's own setting, so when it is off that is
    // the whole answer and naming an engine would only confuse it.
    if (!profile_vision_enabled) {
        return "this agent profile does not accept images";
    }
    if (engine_id.empty()) {
        return {}; // API-backed; not ours to refuse
    }
    if (!engine_supports_vision(engine_id)) {
        // The reason already NAMES the engine — it renders as
        // `soma (verdict=hybrid)` — so a message that also quotes the id says
        // "soma" twice inside nested parentheses. Colon form, and let the reason
        // carry the name; the id is still there for anyone scanning.
        return routing_reason.empty()
                   ? "the '" + engine_id + "' engine serving this agent does not accept images"
                   : "the engine serving this agent does not accept images: " + routing_reason;
    }
    return {};
}

} // namespace mm
