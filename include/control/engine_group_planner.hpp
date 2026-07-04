#pragma once

#include "common/models.hpp"

#include <string>
#include <vector>
#include <optional>

namespace mm {

// Plans multi-node vLLM engine groups: how to spread one model across several
// nodes when it does not fit on one. Tensor parallelism stays *within* a node
// (needs fast intra-node links); pipeline parallelism spans nodes (tolerates
// Ethernet). Members must share a build fingerprint (arch + vLLM version) and
// a common collective-comms backend.
//
// This module is pure: it ranks candidate groups from a node snapshot. Actual
// Ray orchestration / engine launch is a separate concern.

struct EngineGroupRequest {
    std::string model_path;
    // Total GPUs the model needs across the whole group (tensor_parallel_size *
    // pipeline_parallel_size from the agent config).
    int         world_size = 1;
};

struct EngineGroupCandidate {
    std::vector<NodeId> nodes;             // members, in launch order (head first)
    std::string         comm_backend;      // "nccl" | "gloo"
    int                 tensor_parallel_size   = 1;  // GPUs per node
    int                 pipeline_parallel_size = 1;  // node count
    int                 world_size             = 1;  // tp * pp
    double              score = 0.0;        // lower is better
    bool                valid = false;
    std::string         reason;             // why invalid, or notes

    bool spans_nodes() const { return pipeline_parallel_size > 1; }
};

// Rank feasible engine groups best-first. A single-node candidate (pp=1) is
// included when one node alone satisfies world_size — callers can use that to
// fall back to the normal placement path. NCCL groups always rank above Gloo
// groups; Gloo only appears when a needed member lacks NCCL.
std::vector<EngineGroupCandidate> plan_engine_groups(
    const EngineGroupRequest& req,
    const std::vector<NodeInfo>& nodes);

// Convenience: the single best candidate, or nullopt if none is feasible.
std::optional<EngineGroupCandidate> best_engine_group(
    const EngineGroupRequest& req,
    const std::vector<NodeInfo>& nodes);

// Derive the "ip:port" Ray workers join from the head node's API URL and the
// head's GCS port, e.g. ("http://192.168.68.82:7070", 6379) →
// "192.168.68.82:6379". Empty string when the URL has no parsable host.
std::string derive_ray_head_address(const std::string& node_url, int gcs_port);

// Apply a planned group split to an agent's engine settings: the candidate's
// tensor/pipeline sizes replace whatever the agent asked for (the plan is the
// authority on how the world size maps onto the actual fleet).
VllmSettings apply_group_plan(const EngineGroupCandidate& group,
                              VllmSettings settings);

} // namespace mm
