#include "node/llama_runtime.hpp"

#include "common/util.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <optional>

namespace mm {

std::string current_runtime_platform() {
#ifdef _WIN32
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#else
    return "linux";
#endif
}

std::string current_runtime_arch() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
    return "x86_64";
#else
    return {};
#endif
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

std::string strip_wrapping_quotes(std::string s) {
    s = mm::util::trim(s);
    if (s.size() >= 2) {
        char a = s.front();
        char b = s.back();
        if ((a == '"' && b == '"') || (a == '\'' && b == '\'')) {
            s = mm::util::trim(s.substr(1, s.size() - 2));
        }
    }
    return s;
}

} // namespace

std::string normalize_llama_model_path(const std::string& raw) {
    std::string p = strip_wrapping_quotes(raw);
    if (p.empty()) return p;

#ifdef _WIN32
    // Accept WSL-style mount paths when the node runtime is Windows:
    // /mnt/y/foo/bar.gguf -> Y:\foo\bar.gguf
    if (p.size() >= 6 &&
        p[0] == '/' && p[1] == 'm' && p[2] == 'n' && p[3] == 't' && p[4] == '/' &&
        std::isalpha(static_cast<unsigned char>(p[5])) &&
        (p.size() == 6 || p[6] == '/')) {
        char drive = static_cast<char>(std::toupper(static_cast<unsigned char>(p[5])));
        std::string out;
        out.reserve(p.size() + 2);
        out.push_back(drive);
        out.push_back(':');
        if (p.size() == 6) {
            out.push_back('\\');
            return out;
        }
        for (size_t i = 6; i < p.size(); ++i) {
            char ch = p[i];
            out.push_back(ch == '/' ? '\\' : ch);
        }
        return out;
    }
    return p;
#else
    // Best-effort support for Windows-style paths when the node runtime is Linux:
    // Y:\foo\bar.gguf -> /mnt/y/foo/bar.gguf
    if (p.size() >= 3 &&
        std::isalpha(static_cast<unsigned char>(p[0])) &&
        p[1] == ':' &&
        (p[2] == '\\' || p[2] == '/')) {
        std::string out;
        out.reserve(p.size() + 8);
        out += "/mnt/";
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(p[0]))));
        for (size_t i = 2; i < p.size(); ++i) {
            char ch = p[i];
            out.push_back(ch == '\\' ? '/' : ch);
        }
        std::error_code ec;
        if (std::filesystem::exists(out, ec)) return out;
    }
    return p;
#endif
}

std::vector<std::string> build_llama_server_args(const std::string& model_path,
                                                 const std::string& mmproj_path,
                                                 const RuntimeSettings& s,
                                                 uint16_t port,
                                                 const std::string& slot_save_path) {
    std::vector<std::string> args;
    const auto& extra = s.extra_args;

    auto arg_matches_flag = [](const std::string& raw, const std::string& flag) {
        const std::string arg = mm::util::trim(raw);
        return arg == flag
            || arg.rfind(flag + "=", 0) == 0
            || arg.rfind(flag + " ", 0) == 0;
    };

    auto has_any_flag = [&](std::initializer_list<const char*> flags) {
        for (const auto& arg : extra) {
            for (const char* flag : flags) {
                if (arg_matches_flag(arg, flag)) return true;
            }
        }
        return false;
    };

    auto find_flag_value = [&](std::initializer_list<const char*> flags) -> std::optional<std::string> {
        for (size_t i = 0; i < extra.size(); ++i) {
            const std::string arg = mm::util::trim(extra[i]);
            for (const char* flag_raw : flags) {
                const std::string flag(flag_raw);
                if (arg == flag) {
                    if (i + 1 < extra.size()) return mm::util::trim(extra[i + 1]);
                    return std::nullopt;
                }
                if (arg.rfind(flag + "=", 0) == 0) {
                    return mm::util::trim(arg.substr(flag.size() + 1));
                }
                if (arg.rfind(flag + " ", 0) == 0) {
                    return mm::util::trim(arg.substr(flag.size() + 1));
                }
            }
        }
        return std::nullopt;
    };

    auto append_value_unless_extra = [&](std::initializer_list<const char*> flags,
                                         const std::string& flag,
                                         const std::string& value) {
        if (has_any_flag(flags)) return;
        args.push_back(flag);
        args.push_back(value);
    };

    // llama-server allocates one context of ctx_size*parallel tokens shared
    // across request slots, so the per-slot ctx_size is multiplied up here.
    int parallel_for_ctx = s.parallel > 0 ? s.parallel : 1;
    if (auto override = find_flag_value({"--parallel"})) {
        try {
            int parsed = std::stoi(*override);
            if (parsed > 0) parallel_for_ctx = parsed;
        } catch (...) {
        }
    }
    const long long server_ctx_size =
        static_cast<long long>(s.ctx_size) * static_cast<long long>(parallel_for_ctx);

    args.push_back("--model");       args.push_back(strip_wrapping_quotes(model_path));
    if (!mmproj_path.empty()) {
        args.push_back("--mmproj");
        args.push_back(strip_wrapping_quotes(mmproj_path));
    }
    args.push_back("--port");        args.push_back(std::to_string(port));
    args.push_back("--host");        args.push_back("127.0.0.1");
    append_value_unless_extra({"--ctx-size", "-c"},
                              "--ctx-size", std::to_string(server_ctx_size));
    append_value_unless_extra({"--gpu-layers", "-ngl"},
                              "--gpu-layers", std::to_string(s.n_gpu_layers));
    if (s.n_threads > 0) {
        append_value_unless_extra({"--threads", "-t"},
                                  "--threads", std::to_string(s.n_threads));
    }
    if (s.n_threads_http > 0) {
        append_value_unless_extra({"--threads-http"},
                                  "--threads-http", std::to_string(s.n_threads_http));
    }
    if (s.parallel > 1) {
        append_value_unless_extra({"--parallel"},
                                  "--parallel", std::to_string(s.parallel));
    }
    if (s.batch_size > 0) {
        append_value_unless_extra({"--batch-size", "-b"},
                                  "--batch-size", std::to_string(s.batch_size));
    }
    if (s.ubatch_size > 0) {
        append_value_unless_extra({"--ubatch-size", "-ub"},
                                  "--ubatch-size", std::to_string(s.ubatch_size));
    }
    // Modern llama-server takes a value for flash attention ([on|off|auto]);
    // the bare flag is rejected by recent builds, so emit the valued form.
    if (s.flash_attn && !has_any_flag({"--flash-attn", "-fa"})) {
        args.push_back("--flash-attn");
        args.push_back("on");
    }
    // Enable KV-cache slot persistence when a save directory is provided, so
    // suspend/restore can round-trip context via /slots/{id}?action=save|restore.
    if (!slot_save_path.empty() && !has_any_flag({"--slot-save-path"})) {
        args.push_back("--slot-save-path");
        args.push_back(slot_save_path);
    }
    for (auto& a : extra) args.push_back(a);
    return args;
}

std::string detect_llama_accelerator(const std::string& platform,
                                     const std::string& arch,
                                     bool has_cuda,
                                     bool has_rocm) {
    // current_runtime_platform() reports macOS as "macos"; accept "darwin" too in
    // case a caller passes the raw uname value.
    const bool is_macos = (platform == "macos" || platform == "darwin");
    if (is_macos) return "metal"; // Apple GPUs (and Intel macs) use Metal
    (void)arch;
    if (has_cuda) return "cuda";
    if (has_rocm) return "rocm";
    return "cpu";
}

} // namespace mm
