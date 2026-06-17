#include "node/vllm_runtime.hpp"

#include "common/util.hpp"

#include <initializer_list>
#include <sstream>

namespace mm {

namespace {

bool arg_matches_flag(const std::string& raw, const std::string& flag) {
    const std::string arg = mm::util::trim(raw);
    return arg == flag
        || arg.rfind(flag + "=", 0) == 0
        || arg.rfind(flag + " ", 0) == 0;
}

bool has_any_flag(const std::vector<std::string>& extra,
                  std::initializer_list<const char*> flags) {
    for (const auto& arg : extra) {
        for (const char* flag : flags) {
            if (arg_matches_flag(arg, flag)) return true;
        }
    }
    return false;
}

std::string number_to_string(double value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

void append_value_unless_extra(std::vector<std::string>& args,
                               const std::vector<std::string>& extra,
                               std::initializer_list<const char*> flags,
                               const std::string& flag,
                               const std::string& value) {
    if (has_any_flag(extra, flags)) return;
    args.push_back(flag);
    args.push_back(value);
}

void append_flag_unless_extra(std::vector<std::string>& args,
                              const std::vector<std::string>& extra,
                              std::initializer_list<const char*> flags,
                              const std::string& flag) {
    if (has_any_flag(extra, flags)) return;
    args.push_back(flag);
}

} // namespace

std::string default_vllm_repo_url_for_platform() {
#ifdef _WIN32
    return kWindowsVllmRepoUrl;
#else
    return kOfficialVllmRepoUrl;
#endif
}

std::string default_vllm_branch_for_platform() {
#ifdef _WIN32
    return kWindowsVllmBranch;
#else
    return "main";
#endif
}

std::vector<std::string> build_vllm_server_args(const std::string& model_ref,
                                                const VllmSettings& settings,
                                                uint16_t port) {
    const auto& extra = settings.extra_args;
    std::vector<std::string> args;
    args.push_back("serve");
    args.push_back(model_ref);

    append_value_unless_extra(args, extra, {"--host"}, "--host", "127.0.0.1");
    append_value_unless_extra(args, extra, {"--port"}, "--port", std::to_string(port));

    if (settings.max_model_len > 0) {
        append_value_unless_extra(args, extra, {"--max-model-len"},
                                  "--max-model-len", std::to_string(settings.max_model_len));
    }
    if (settings.max_num_seqs > 0) {
        append_value_unless_extra(args, extra, {"--max-num-seqs"},
                                  "--max-num-seqs", std::to_string(settings.max_num_seqs));
    }
    if (settings.max_num_batched_tokens > 0) {
        append_value_unless_extra(args, extra, {"--max-num-batched-tokens"},
                                  "--max-num-batched-tokens",
                                  std::to_string(settings.max_num_batched_tokens));
    }
    if (settings.tensor_parallel_size > 1) {
        append_value_unless_extra(args, extra, {"--tensor-parallel-size"},
                                  "--tensor-parallel-size",
                                  std::to_string(settings.tensor_parallel_size));
    }
    if (settings.pipeline_parallel_size > 1) {
        append_value_unless_extra(args, extra, {"--pipeline-parallel-size"},
                                  "--pipeline-parallel-size",
                                  std::to_string(settings.pipeline_parallel_size));
    }
    if (settings.gpu_memory_utilization > 0.0) {
        append_value_unless_extra(args, extra,
                                  {"--gpu-memory-utilization", "--gpu_memory_utilization"},
                                  "--gpu-memory-utilization",
                                  number_to_string(settings.gpu_memory_utilization));
    }
    if (!settings.dtype.empty()) {
        append_value_unless_extra(args, extra, {"--dtype"}, "--dtype", settings.dtype);
    }
    if (!settings.quantization.empty()) {
        append_value_unless_extra(args, extra, {"--quantization"},
                                  "--quantization", settings.quantization);
    }
    if (!settings.served_model_name.empty()) {
        append_value_unless_extra(args, extra, {"--served-model-name"},
                                  "--served-model-name", settings.served_model_name);
    }
    if (settings.trust_remote_code) {
        append_flag_unless_extra(args, extra, {"--trust-remote-code"},
                                 "--trust-remote-code");
    }
    if (settings.enable_prefix_caching) {
        append_flag_unless_extra(args, extra,
                                 {"--enable-prefix-caching", "--no-enable-prefix-caching"},
                                 "--enable-prefix-caching");
    }
    if (settings.enable_auto_tool_choice) {
        append_flag_unless_extra(args, extra,
                                 {"--enable-auto-tool-choice",
                                  "--no-enable-auto-tool-choice"},
                                 "--enable-auto-tool-choice");
    }
    if (settings.enable_sleep_mode) {
        append_flag_unless_extra(args, extra,
                                 {"--enable-sleep-mode", "--no-enable-sleep-mode"},
                                 "--enable-sleep-mode");
    }
    if (!settings.tool_call_parser.empty()) {
        append_value_unless_extra(args, extra, {"--tool-call-parser"},
                                  "--tool-call-parser", settings.tool_call_parser);
    }

    args.insert(args.end(), extra.begin(), extra.end());
    return args;
}

VllmEngineMetrics parse_vllm_metrics_text(const std::string& text) {
    VllmEngineMetrics m;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        const std::string trimmed = mm::util::trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        const size_t name_end = trimmed.find_first_of("{ \t");
        if (name_end == std::string::npos) continue;
        const std::string name = trimmed.substr(0, name_end);

        const size_t value_start = trimmed.find_last_of(" \t");
        if (value_start == std::string::npos) continue;
        double value = 0.0;
        try {
            value = std::stod(trimmed.substr(value_start + 1));
        } catch (...) {
            continue;
        }

        auto name_ends_with = [&](const char* suffix) {
            const std::string s(suffix);
            return name.size() >= s.size() &&
                   name.compare(name.size() - s.size(), s.size(), s) == 0;
        };

        if (name_ends_with("num_requests_running")) {
            m.num_requests_running += static_cast<int>(value);
            m.valid = true;
        } else if (name_ends_with("num_requests_waiting")) {
            m.num_requests_waiting += static_cast<int>(value);
            m.valid = true;
        } else if (name_ends_with("gpu_cache_usage_perc") ||
                   name_ends_with("kv_cache_usage_perc")) {
            if (value > m.kv_cache_usage) m.kv_cache_usage = value;
            m.valid = true;
        }
    }
    return m;
}

} // namespace mm
