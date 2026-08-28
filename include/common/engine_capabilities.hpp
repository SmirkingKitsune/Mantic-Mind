#pragma once

// What an engine can be asked to do — stated ONCE, for two readers.
//
// `EngineDescriptor::supports_vision` is the node's copy of this, and it was the
// only copy: `false` in make_soma_descriptor, `true` in make_llama_descriptor,
// and never read outside the file that set it. Control, which is where a request
// carrying an image actually arrives, had no way to ask. So all three 422 gates
// tested the AGENT PROFILE's vision_settings.enabled and nothing tested whether
// the engine on the other end could accept an image at all — an agent with
// vision on, serving a model that earned a streamable verdict, sent image parts
// to a text-only engine (roadmap D12).
//
// The obvious fix — a literal "soma" comparison in control — is the thing the
// descriptor pattern exists to prevent, and it is how the backend attribution
// bug got in (see EngineSupervisor::make_slot_info). So the fact lives here,
// where the node descriptors and the control API can both read it, and neither
// owns it.
//
// Host-level facts do NOT belong here. Whether a node has a GPU, or how much
// VRAM is free, is NodeCapabilities and varies per machine. This is about what
// the engine SOFTWARE can do, which is the same everywhere the binary runs.

#include <string>

namespace mm {

/// Can this engine accept image content parts?
///
/// Soma is text-only by construction — it is an MoE streaming engine with no
/// vision tower and no projector, and adding one is not a configuration change.
/// llama.cpp accepts images when an mmproj is supplied, so it answers true and
/// the mmproj requirement is enforced separately by the agent profile.
///
/// An UNKNOWN engine id answers false. A new engine is assumed unable to take
/// images until it says otherwise: the failure that direction is a clear refusal
/// the operator can act on, while the other direction silently drops the image
/// and answers as though it had been read, which is the defect this closes.
bool engine_supports_vision(const std::string& engine_id) noexcept;

/// Why an image may not be sent to an agent, or empty if it may. PURE.
///
/// Split out from ControlApiServer::image_refusal so the RULE can be asserted
/// without standing up a server, a model registry and an admitted record just to
/// discover what a two-condition predicate does. The server method is then only
/// the lookup — which engine serves this agent — and that half is covered by the
/// existing HTTP negatives.
///
/// `engine_id` empty means the agent is API-backed and owns no node-local
/// engine; the remote provider's capabilities govern and this returns empty.
/// `routing_reason` is appended so the message says WHY that engine was chosen,
/// because "does not accept images" alone reads as a profile problem and sends
/// the operator to switch on a setting that is already on.
std::string image_refusal_for(bool profile_vision_enabled,
                              const std::string& engine_id,
                              const std::string& routing_reason);

} // namespace mm
