#include "node/vllm_runtime.hpp"

#include <sstream>

namespace mm {

namespace {

std::string number_to_string(double value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

void value(std::vector<std::string>& args,
           const char* flag,
           const std::string& setting) {
    args.emplace_back(flag);
    args.push_back(setting);
}

} // namespace

std::vector<std::string> build_vllm_server_args(
    const std::string& model_ref,
    const std::string& served_model_name,
    const VllmEngineConfig& s,
    std::uint16_t port) {
    std::vector<std::string> args{"serve", model_ref};
    value(args, "--host", "127.0.0.1");
    value(args, "--port", std::to_string(port));
    value(args, "--max-model-len", std::to_string(s.max_model_len));
    value(args, "--max-num-seqs", std::to_string(s.max_num_seqs));
    if (s.max_num_batched_tokens > 0)
        value(args, "--max-num-batched-tokens", std::to_string(s.max_num_batched_tokens));
    value(args, "--tensor-parallel-size", std::to_string(s.tensor_parallel_size));
    value(args, "--pipeline-parallel-size", std::to_string(s.pipeline_parallel_size));
    if (s.pipeline_parallel_size > 1)
        value(args, "--distributed-executor-backend", "ray");
    value(args, "--gpu-memory-utilization", number_to_string(s.gpu_memory_utilization));
    value(args, "--dtype", s.dtype);
    if (!s.quantization.empty()) value(args, "--quantization", s.quantization);
    if (!served_model_name.empty()) value(args, "--served-model-name", served_model_name);
    if (s.trust_remote_code) args.emplace_back("--trust-remote-code");
    if (s.enable_prefix_caching) args.emplace_back("--enable-prefix-caching");
    if (s.enable_auto_tool_choice) args.emplace_back("--enable-auto-tool-choice");
    if (s.enable_sleep_mode) args.emplace_back("--enable-sleep-mode");
    if (!s.tool_call_parser.empty()) value(args, "--tool-call-parser", s.tool_call_parser);
    args.insert(args.end(), s.extra_args.begin(), s.extra_args.end());
    return args;
}

} // namespace mm
