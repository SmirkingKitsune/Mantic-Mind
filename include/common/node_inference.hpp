#pragma once

#include "common/models.hpp"

#include <functional>
#include <string>

namespace mm {

// Typed status shared by NodeService and every NodeOperations transport.  The
// HTTP adapter maps the existing REST/SSE protocol onto these values; local
// callers never need to manufacture or inspect HTTP status codes.
enum class NodeServiceStatus {
    Ok,
    InvalidArgument,
    UnsupportedBackend,
    RuntimeUnavailable,
    ModelNotFound,
    CapacityExhausted,
    SlotNotFound,
    Busy,
    Conflict,
    Canceled,
    Unavailable,
    Failed,
};

inline const char* to_string(NodeServiceStatus status) {
    switch (status) {
        case NodeServiceStatus::Ok:                 return "ok";
        case NodeServiceStatus::InvalidArgument:    return "invalid_argument";
        case NodeServiceStatus::UnsupportedBackend: return "unsupported_backend";
        case NodeServiceStatus::RuntimeUnavailable: return "runtime_unavailable";
        case NodeServiceStatus::ModelNotFound:       return "model_not_found";
        case NodeServiceStatus::CapacityExhausted:   return "capacity_exhausted";
        case NodeServiceStatus::SlotNotFound:        return "slot_not_found";
        case NodeServiceStatus::Busy:                return "busy";
        case NodeServiceStatus::Conflict:            return "conflict";
        case NodeServiceStatus::Canceled:            return "canceled";
        case NodeServiceStatus::Unavailable:         return "unavailable";
        case NodeServiceStatus::Failed:              return "failed";
    }
    return "failed";
}

struct NodeInferRequest {
    InferenceRequest request;
    // Empty retains the legacy behavior of selecting the first ready slot.
    SlotId slot_id;
    // Optional caller-owned cancellation check.  Local inference forwards it
    // to the loopback llama runtime; the HTTP transport aborts its request.
    std::function<bool()> cancel_requested;
};

// Exactly one NodeInferResult is returned for every typed inference call.
// InferenceChunk callbacks carry only incremental data; wire-level done/error
// events are translated into this terminal result by the transport adapter.
struct NodeInferResult {
    NodeServiceStatus status = NodeServiceStatus::Failed;
    std::string       error;
    SlotId            slot_id;
    Message           message;
    int               tokens_used = 0;
    std::string       finish_reason;

    bool ok() const { return status == NodeServiceStatus::Ok; }
};

} // namespace mm
