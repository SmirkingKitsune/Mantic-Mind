#include "node/vllm_runtime.hpp"

#include "common/util.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <sstream>
#include <system_error>
#include <vector>

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

std::string current_vllm_platform() {
#ifdef _WIN32
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#else
    return "linux";
#endif
}

std::string current_vllm_arch() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
    return "x86_64";
#else
    return {};
#endif
}

bool is_apple_silicon_environment(const std::string& platform,
                                  const std::string& arch) {
    return platform == "macos" && (arch == "aarch64" || arch == "arm64");
}

std::string detect_vllm_accelerator(const std::string& platform,
                                    const std::string& arch,
                                    bool has_cuda,
                                    bool has_rocm) {
    if (is_apple_silicon_environment(platform, arch)) return "metal";
    if (platform == "windows") return "windows";
    if (has_cuda) return "cuda";
    if (has_rocm) return "rocm";
    return "cpu";
}

bool detect_rocm_present() {
#ifdef _WIN32
    return false;
#else
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists("/opt/rocm", ec)) return true;
    if (const char* path_env = std::getenv("PATH")) {
        for (const auto& dir : mm::util::split(path_env, ':')) {
            if (dir.empty()) continue;
            if (fs::exists(fs::path(dir) / "rocminfo", ec)) return true;
        }
    }
    return false;
#endif
}

namespace {

// Extract the leading dotted-numeric components of a version string. Accepts a
// bare "0.6.3", a "v0.6.3" tag, or a full "vllm 0.6.3.dev1+g..." line, stopping
// at the first non-numeric component. Returns {} when nothing parses.
std::vector<int> parse_vllm_version_components(const std::string& raw) {
    std::string s = mm::util::trim(raw);
    if (s.find(' ') != std::string::npos) {
        for (const auto& tok : mm::util::split(s, ' ')) {
            const std::string t = mm::util::trim(tok);
            if (t.empty()) continue;
            const bool starts_digit = std::isdigit(static_cast<unsigned char>(t[0])) != 0;
            const bool starts_v_digit = (t[0] == 'v' || t[0] == 'V') && t.size() > 1 &&
                std::isdigit(static_cast<unsigned char>(t[1])) != 0;
            if (starts_digit || starts_v_digit) { s = t; break; }
        }
    }
    if (!s.empty() && (s[0] == 'v' || s[0] == 'V')) s = s.substr(1);

    std::vector<int> comps;
    for (const auto& part : mm::util::split(s, '.')) {
        std::string digits;
        for (char c : part) {
            if (std::isdigit(static_cast<unsigned char>(c))) digits.push_back(c);
            else break;
        }
        if (digits.empty()) break;
        try {
            comps.push_back(std::stoi(digits));
        } catch (...) {
            break;
        }
    }
    return comps;
}

} // namespace

int compare_vllm_versions(const std::string& a, const std::string& b) {
    const std::vector<int> va = parse_vllm_version_components(a);
    const std::vector<int> vb = parse_vllm_version_components(b);
    if (va.empty() || vb.empty()) return 0;
    const size_t n = std::max(va.size(), vb.size());
    for (size_t i = 0; i < n; ++i) {
        const int x = i < va.size() ? va[i] : 0;
        const int y = i < vb.size() ? vb[i] : 0;
        if (x != y) return x < y ? -1 : 1;
    }
    return 0;
}

std::string default_vllm_repo_url_for_environment(const std::string& platform,
                                                  const std::string& arch) {
    if (platform == "windows") return kWindowsVllmRepoUrl;
    if (is_apple_silicon_environment(platform, arch)) return kMetalVllmRepoUrl;
    return kOfficialVllmRepoUrl;
}

std::string default_vllm_branch_for_environment(const std::string& platform,
                                                const std::string& arch) {
    (void)arch;
    if (platform == "windows") return kWindowsVllmBranch;
    return "main";
}

std::string default_vllm_repo_url_for_platform() {
    return default_vllm_repo_url_for_environment(current_vllm_platform(),
                                                 current_vllm_arch());
}

std::string default_vllm_branch_for_platform() {
    return default_vllm_branch_for_environment(current_vllm_platform(),
                                               current_vllm_arch());
}

bool model_ref_is_gguf(const std::string& model_ref) {
    const std::string s = mm::util::to_lower(mm::util::trim(model_ref));
    static const std::string ext = ".gguf";
    return s.size() > ext.size() &&
           s.compare(s.size() - ext.size(), ext.size(), ext) == 0;
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
