#include "control/engine_group_planner.hpp"

#include "common/util.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <functional>
#include <map>

namespace mm {

namespace {

const RuntimeStatus* vllm_runtime(const NodeInfo& node) {
    const auto it = std::find_if(node.engines.begin(), node.engines.end(),
                                 [](const RuntimeStatus& s) {
                                     return s.engine_id == "vllm";
                                 });
    return it == node.engines.end() ? nullptr : &*it;
}

bool has_backend(const NodeInfo& node, const std::string& backend) {
    return std::find(node.capabilities.comm_backends.begin(),
                     node.capabilities.comm_backends.end(), backend) !=
           node.capabilities.comm_backends.end();
}

bool reachable_cluster_url(const std::string& url) {
    const auto lower = util::to_lower(url);
    return lower.find("127.0.0.1") == std::string::npos &&
           lower.find("localhost") == std::string::npos &&
           lower.find("[::1]") == std::string::npos;
}

bool idle_for_exclusive_group(const NodeInfo& node) {
    if (!node.slots.empty()) return false; // suspended records count too
    return node.ray.group_id.empty();
}

} // namespace

bool distributed_vllm_model_ref_supported(const std::string& model_ref,
                                          std::string& reason) {
    reason.clear();
    if (util::is_hf_repo_id(model_ref)) return true;
    const std::filesystem::path path(model_ref);
    const bool posix_absolute = !model_ref.empty() && model_ref.front() == '/';
    const bool windows_absolute = model_ref.size() >= 3 &&
        std::isalpha(static_cast<unsigned char>(model_ref[0])) &&
        model_ref[1] == ':' && (model_ref[2] == '\\' || model_ref[2] == '/');
    const bool unc_absolute = model_ref.rfind("\\\\", 0) == 0;
    if (path.is_absolute() || posix_absolute || windows_absolute || unc_absolute)
        return true;
    reason = "multi-node vLLM requires a Hugging Face repository id or an "
             "absolute path mounted identically on every Ray member";
    return false;
}

std::vector<EngineGroupCandidate> plan_engine_groups(
    const EngineGroupRequest& req,
    const std::vector<NodeInfo>& nodes) {
    std::vector<EngineGroupCandidate> out;
    if (req.tensor_parallel_size < 1 || req.pipeline_parallel_size < 2) return out;
    std::string model_error;
    if (!distributed_vllm_model_ref_supported(req.model_ref, model_error)) return out;

    std::map<std::string, std::vector<const NodeInfo*>> pools;
    for (const auto& node : nodes) {
        if (!node.connected || util::to_lower(node.platform) != "linux") continue;
        if (!node.capabilities.supports_ray ||
            node.capabilities.gpu_count < req.tensor_parallel_size) continue;
        if (!reachable_cluster_url(node.url) || !idle_for_exclusive_group(node)) continue;
        if (node.engine_config_version != req.config_version) continue;
        const RuntimeStatus* runtime = vllm_runtime(node);
        if (runtime == nullptr || !runtime->ready || runtime->version.empty()) continue;
        const std::string fingerprint = node.capabilities.arch + "|" +
                                        runtime->version + "|" + runtime->variant;
        if (node.capabilities.arch.empty()) continue;
        pools[fingerprint].push_back(&node);
    }

    for (auto& [fingerprint, pool] : pools) {
        if (pool.size() < static_cast<std::size_t>(req.pipeline_parallel_size)) continue;
        std::sort(pool.begin(), pool.end(), [](const NodeInfo* a, const NodeInfo* b) {
            const bool an = has_backend(*a, "nccl");
            const bool bn = has_backend(*b, "nccl");
            if (an != bn) return an;
            if (a->capabilities.interconnect_gbps != b->capabilities.interconnect_gbps)
                return a->capabilities.interconnect_gbps >
                       b->capabilities.interconnect_gbps;
            return a->id < b->id;
        });

        std::vector<std::size_t> selection;
        const auto emit = [&] {
            EngineGroupCandidate candidate;
            candidate.runtime_fingerprint = fingerprint;
            bool all_nccl = true;
            bool all_gloo = true;
            double bandwidth = 0.0;
            for (const auto selected : selection) {
                const NodeInfo& node = *pool[selected];
                candidate.nodes.push_back(node.id);
                all_nccl = all_nccl && has_backend(node, "nccl");
                all_gloo = all_gloo && has_backend(node, "gloo");
                bandwidth += node.capabilities.interconnect_gbps;
            }
            if (all_nccl) {
                candidate.transport = "nccl";
            } else if (req.allow_experimental_gloo && all_gloo) {
                candidate.transport = "gloo";
                candidate.experimental = true;
                candidate.reason =
                    "experimental Gloo fallback; upstream vLLM does not guarantee it";
            } else {
                return;
            }
            candidate.score = bandwidth / req.pipeline_parallel_size;
            if (candidate.experimental) candidate.score -= 1000000.0;
            out.push_back(std::move(candidate));
        };
        std::function<void(std::size_t)> choose = [&](std::size_t start) {
            if (selection.size() ==
                static_cast<std::size_t>(req.pipeline_parallel_size)) {
                emit();
                return;
            }
            const auto remaining = static_cast<std::size_t>(
                req.pipeline_parallel_size) - selection.size();
            for (std::size_t i = start; i + remaining <= pool.size(); ++i) {
                selection.push_back(i);
                choose(i + 1);
                selection.pop_back();
            }
        };
        choose(0);
    }
    std::sort(out.begin(), out.end(), [](const EngineGroupCandidate& a,
                                         const EngineGroupCandidate& b) {
        if (a.experimental != b.experimental) return !a.experimental;
        if (a.score != b.score) return a.score > b.score;
        return a.nodes < b.nodes;
    });
    return out;
}

std::optional<EngineGroupCandidate> best_engine_group(
    const EngineGroupRequest& req,
    const std::vector<NodeInfo>& nodes) {
    auto planned = plan_engine_groups(req, nodes);
    if (planned.empty()) return std::nullopt;
    return planned.front();
}

} // namespace mm
