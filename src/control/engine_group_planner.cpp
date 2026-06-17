#include "control/engine_group_planner.hpp"

#include <algorithm>
#include <map>
#include <tuple>

namespace mm {

namespace {

// Gloo over GPU tensors stages through host memory — fine for the occasional
// pipeline-stage handoff across nodes, ruinous for per-layer tensor-parallel
// all-reduces. We never put TP across nodes, so Gloo is only ever a cross-node
// PP link here; this penalty keeps it strictly a last resort behind NCCL.
constexpr double kGlooPenalty       = 1000.0;
constexpr double kPerExtraNodeCost  = 10.0;   // prefer fewer nodes (less fragmentation)
constexpr double kPerWastedGpuCost  = 1.0;    // prefer a tight fit to world_size

bool node_usable(const NodeInfo& n) {
    const auto& c = n.capabilities;
    // A member must declare a fingerprint (arch + vLLM build) and at least one
    // GPU and comm backend, or it cannot co-host an engine.
    return !c.arch.empty()
        && !c.vllm_version.empty()
        && c.gpu_count > 0
        && !c.comm_backends.empty();
}

} // namespace

std::vector<EngineGroupCandidate> plan_engine_groups(
    const EngineGroupRequest& req,
    const std::vector<NodeInfo>& nodes)
{
    const int world_needed = req.world_size > 0 ? req.world_size : 1;

    // Pool nodes by (fingerprint, gpu_count): tensor parallelism needs a
    // uniform per-node GPU count, and every member of one engine must share
    // the same arch + vLLM build.
    std::map<std::tuple<std::string, int>, std::vector<NodeInfo>> pools;
    for (const auto& n : nodes) {
        if (!node_usable(n)) continue;
        pools[{n.capabilities.fingerprint(), n.capabilities.gpu_count}]
            .push_back(n);
    }

    std::vector<EngineGroupCandidate> out;

    for (auto& [key, pool] : pools) {
        const int gpus_per_node = std::get<1>(key);
        if (gpus_per_node <= 0) continue;

        // Order members so an all-NCCL subset is preferred, then by faster
        // interconnect, then by id for determinism.
        std::sort(pool.begin(), pool.end(), [](const NodeInfo& a, const NodeInfo& b) {
            const bool an = a.capabilities.has_comm_backend("nccl");
            const bool bn = b.capabilities.has_comm_backend("nccl");
            if (an != bn) return an;  // NCCL-capable first
            if (a.capabilities.interconnect_gbps != b.capabilities.interconnect_gbps)
                return a.capabilities.interconnect_gbps > b.capabilities.interconnect_gbps;
            return a.id < b.id;
        });

        // Smallest member count whose GPUs satisfy world_needed wins for this
        // pool; larger groups only add fragmentation.
        for (int k = 1; k <= static_cast<int>(pool.size()); ++k) {
            const int world = k * gpus_per_node;
            if (world < world_needed) continue;

            EngineGroupCandidate c;
            c.tensor_parallel_size   = gpus_per_node;
            c.pipeline_parallel_size = k;
            c.world_size             = world;

            bool all_nccl = true;
            bool all_have_gloo = true;
            bool all_ray = true;
            for (int i = 0; i < k; ++i) {
                const auto& cap = pool[i].capabilities;
                c.nodes.push_back(pool[i].id);
                if (!cap.has_comm_backend("nccl")) all_nccl = false;
                if (!cap.has_comm_backend("gloo")) all_have_gloo = false;
                if (!cap.supports_ray) all_ray = false;
            }

            // Multi-node groups need Ray on every member.
            if (k > 1 && !all_ray) {
                c.valid  = false;
                c.reason = "not all members support Ray";
                out.push_back(std::move(c));
                break;
            }

            if (all_nccl) {
                c.comm_backend = "nccl";
                c.valid = true;
            } else if (all_have_gloo) {
                c.comm_backend = "gloo";
                c.valid = true;
                c.reason = "a member lacks NCCL; Gloo cross-node link (slower)";
            } else {
                c.valid  = false;
                c.reason = "no common comm backend across members";
            }

            // Score: comm penalty dominates, then fragmentation, then waste,
            // then a mild interconnect tiebreak (faster link scores lower).
            double score = 0.0;
            if (c.comm_backend == "gloo") score += kGlooPenalty;
            score += (k - 1) * kPerExtraNodeCost;
            score += (world - world_needed) * kPerWastedGpuCost;
            double avg_link = 0.0;
            for (int i = 0; i < k; ++i) avg_link += pool[i].capabilities.interconnect_gbps;
            if (k > 0) avg_link /= k;
            score -= avg_link * 0.001;
            c.score = score;

            out.push_back(std::move(c));
            break;  // one (smallest) candidate per pool
        }
    }

    std::sort(out.begin(), out.end(),
              [](const EngineGroupCandidate& a, const EngineGroupCandidate& b) {
                  if (a.valid != b.valid) return a.valid;  // valid first
                  return a.score < b.score;
              });
    return out;
}

std::optional<EngineGroupCandidate> best_engine_group(
    const EngineGroupRequest& req,
    const std::vector<NodeInfo>& nodes)
{
    auto ranked = plan_engine_groups(req, nodes);
    if (!ranked.empty() && ranked.front().valid) return ranked.front();
    return std::nullopt;
}

} // namespace mm
