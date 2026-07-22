#include "node/llama_cpp_provisioner.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"
#include "node/llama_runtime.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <system_error>
#include <utility>

#ifdef _WIN32
#  include <Windows.h>
#endif

namespace fs = std::filesystem;

namespace mm {
namespace {

// Best-effort progress extraction from installer output. Prefers an explicit
// percentage and falls back to an A/B ratio; negative means indeterminate.
double parse_install_fraction(const std::string& line) {
    auto is_num = [](char c) { return (c >= '0' && c <= '9') || c == '.'; };
    auto clamp01 = [](double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); };

    if (const size_t p = line.rfind('%'); p != std::string::npos && p > 0) {
        size_t b = p;
        while (b > 0 && is_num(line[b - 1])) --b;
        if (b < p) {
            try {
                const double pct = std::stod(line.substr(b, p - b));
                if (pct >= 0.0 && pct <= 100.0) return clamp01(pct / 100.0);
            } catch (...) {}
        }
    }

    for (size_t slash = line.find('/'); slash != std::string::npos;
         slash = line.find('/', slash + 1)) {
        size_t b = slash;
        while (b > 0 && is_num(line[b - 1])) --b;
        size_t e = slash + 1;
        while (e < line.size() && is_num(line[e])) ++e;
        if (b < slash && slash + 1 < e) {
            try {
                const double num = std::stod(line.substr(b, slash - b));
                const double den = std::stod(line.substr(slash + 1, e - slash - 1));
                if (den > 0.0 && num >= 0.0 && num <= den) return clamp01(num / den);
            } catch (...) {}
        }
    }
    return -1.0;
}

std::string strip_wrapping_quotes(std::string s) {
    s = util::trim(s);
    if (s.size() >= 2) {
        const char a = s.front();
        const char b = s.back();
        if ((a == '"' && b == '"') || (a == '\'' && b == '\'')) {
            s = util::trim(s.substr(1, s.size() - 2));
        }
    }
    return s;
}

bool is_path_like(const std::string& s) {
    return s.find('/') != std::string::npos
        || s.find('\\') != std::string::npos
        || s.find(':') != std::string::npos;
}

bool is_generic_llama_server_request(const std::string& raw) {
    const std::string request = util::to_lower(strip_wrapping_quotes(raw));
    return request == "llama-server" || request == "llama-server.exe";
}

// A bare `llama-server` on PATH has no accelerator identity. In particular,
// the Windows Winget package is a generic CPU build and must not shadow a
// managed CUDA/Vulkan/ROCm runtime. Explicit paths/wrapper names remain trusted,
// and disabling auto-provision preserves the traditional PATH-only behavior.
bool use_unmanaged_path_runtime(const LlamaProvisionConfig& cfg) {
    if (!is_generic_llama_server_request(cfg.requested_executable)) return true;
    if (!cfg.auto_provision) return true;
    if (cfg.accelerator == "cpu") return true;
    // llama.cpp's normal macOS build enables Metal, so the conventional PATH
    // binary is accelerator-correct there rather than analogous to Winget CPU.
    return cfg.platform == "macos" && cfg.accelerator == "metal";
}

bool file_exists(const fs::path& p) {
    std::error_code ec;
    return fs::exists(p, ec) && fs::is_regular_file(p, ec);
}

std::string exe_suffix(const std::string& platform) {
    return platform == "windows" ? ".exe" : std::string{};
}

std::string release_arch(const std::string& arch) {
    const std::string normalized = util::to_lower(util::trim(arch));
    if (normalized == "x86_64" || normalized == "amd64" || normalized == "x64")
        return "x64";
    if (normalized == "x86" || normalized == "i386" || normalized == "i686")
        return "x86";
    if (normalized == "aarch64" || normalized == "arm64" ||
        normalized == "apple-silicon" || normalized == "apple silicon")
        return "arm64";
    return normalized;
}

std::string safe_path_component(std::string value) {
    value = util::to_lower(util::trim(value));
    for (char& ch : value) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_')
            ch = '-';
    }
    return value.empty() ? std::string{"unknown"} : value;
}

// CMAKE_CUDA_ARCHITECTURES accepts a semicolon-separated list and optional
// "-real"/"-virtual" suffixes. Reduce those values to the architecture names
// reported by `nvcc --list-gpu-arch` so the source-build preflight can verify
// that the compiler selected by CUDACXX/PATH actually supports the detected GPU.
std::vector<std::string> cuda_arch_targets(const std::string& value) {
    std::vector<std::string> targets;
    std::string token;
    auto flush = [&] {
        token = util::to_lower(util::trim(token));
        if (token.rfind("compute_", 0) == 0) token.erase(0, 8);
        if (token.rfind("sm_", 0) == 0) token.erase(0, 3);
        for (const std::string& suffix : {std::string{"-real"},
                                          std::string{"-virtual"}}) {
            if (token.size() > suffix.size() &&
                token.compare(token.size() - suffix.size(), suffix.size(), suffix) == 0) {
                token.erase(token.size() - suffix.size());
                break;
            }
        }
        const bool valid = !token.empty() &&
            std::isdigit(static_cast<unsigned char>(token.front())) != 0 &&
            std::all_of(token.begin(), token.end(), [](unsigned char ch) {
                return std::isdigit(ch) != 0 || (ch >= 'a' && ch <= 'z');
            });
        if (valid && std::find(targets.begin(), targets.end(), token) == targets.end())
            targets.push_back(token);
        token.clear();
    };
    for (const char ch : value) {
        if (ch == ';' || ch == ',' || std::isspace(static_cast<unsigned char>(ch)))
            flush();
        else
            token.push_back(ch);
    }
    flush();
    return targets;
}

bool cuda_arch_needs_architecture_specific_target(const std::string& target) {
    return target.size() == 3 && target[0] == '1' && target[1] == '2' &&
        std::isdigit(static_cast<unsigned char>(target[2])) != 0;
}

std::string cuda_feature_target(const std::string& target) {
    return cuda_arch_needs_architecture_specific_target(target)
        ? target + "a"
        : target;
}

// llama.cpp's Blackwell MXFP4 kernels use architecture-specific tensor-core
// instructions. Plain 12X is a baseline/forward-compatible target and cannot
// assemble those instructions, so detected 12X devices must be compiled as
// 12Xa. Use a real-only target for detected hardware to avoid an unnecessary,
// non-forward-compatible PTX payload and to work with CMake 3.28 in WSL.
std::string cuda_cmake_architectures(const std::string& value) {
    std::vector<std::string> mapped;
    for (const auto& target : cuda_arch_targets(value)) {
        mapped.push_back(cuda_arch_needs_architecture_specific_target(target)
            ? target + "a-real"
            : target);
    }
    return util::join(mapped, ";");
}

std::string cuda_arch_requirement(const std::vector<std::string>& targets) {
    std::vector<std::string> labels;
    labels.reserve(targets.size());
    for (const auto& target : targets)
        labels.push_back("compute_" + cuda_feature_target(target));
    std::string requirement = "NVCC support for " + util::join(labels, ", ");
    if (std::find(targets.begin(), targets.end(), "120") != targets.end())
        requirement +=
            " (CUDA Toolkit 12.8 or newer; llama.cpp FP4 kernels require sm_120a)";
    return requirement;
}

bool cuda_removed_default_arch_failure(const std::string& failure_detail) {
    const std::string lower = util::to_lower(failure_detail);
    const bool removed_arch = lower.find("sm_52") != std::string::npos ||
        lower.find("compute_52") != std::string::npos;
    const bool rejected = lower.find("not defined for option 'gpu-name'") !=
            std::string::npos ||
        lower.find("unsupported gpu architecture") != std::string::npos;
    const bool compiler_id = lower.find("cmakecudacompilerid") !=
            std::string::npos ||
        lower.find("cmakedeterminecudacompiler") != std::string::npos;
    return removed_arch && rejected && compiler_id;
}

bool cuda_blackwell_baseline_target_failure(const std::string& failure_detail) {
    const std::string lower = util::to_lower(failure_detail);
    const bool baseline_target =
        lower.find("target 'sm_120'") != std::string::npos ||
        lower.find("target 'sm_121'") != std::string::npos ||
        lower.find("-arch=sm_120 ") != std::string::npos ||
        lower.find("-arch=sm_121 ") != std::string::npos;
    const bool architecture_specific_feature =
        lower.find(".kind::mxf4") != std::string::npos ||
        lower.find(".block_scale") != std::string::npos ||
        lower.find("mma with block scale") != std::string::npos;
    return baseline_target && architecture_specific_feature;
}

bool nvcc_supports_arch(const std::string& output, const std::string& target) {
    const std::regex token("(^|[\\s,])compute_" + target + "([\\s,]|$)",
                           std::regex::icase);
    return std::regex_search(output, token);
}

std::string validated_llama_version(std::string output) {
    output = util::trim(output);
    if (output.empty()) return {};
    const std::string lower = util::to_lower(output);
    for (const std::string& error : {
             std::string{"not recognized"}, std::string{"unrecognized option"},
             std::string{"unknown option"}, std::string{"unknown argument"},
             std::string{"invalid option"}, std::string{"invalid argument"}}) {
        if (lower.find(error) != std::string::npos) return {};
    }
    if (std::regex_match(lower, std::regex("b[0-9]+"))) return output;
    // Known llama.cpp forms include "version: 6389 (commit)",
    // "llama.cpp version: 6389", and release tags such as "version b6389".
    // Requiring a version marker plus a numeric build avoids presenting stderr
    // or shell diagnostics as the installed version.
    if (lower.find("version") == std::string::npos) return {};
    if (!std::regex_search(lower, std::regex("(^|[^0-9])b?[0-9]+([^0-9]|$)")))
        return {};
    return output;
}

std::string backend_for_variant(std::string variant) {
    variant = util::to_lower(util::trim(variant));
    if (variant == "cuda-12" || variant == "cuda-13") return "cuda";
    if (variant == "hip") return "rocm";
    return variant;
}

fs::path create_llama_build_log_path(const LlamaProvisionConfig& cfg) {
    const fs::path log_dir = fs::path(cfg.provision_dir) / "logs";
    std::error_code ec;
    fs::create_directories(log_dir, ec);
    if (ec) return {};

    struct ExistingLog {
        fs::path path;
        fs::file_time_type modified;
    };
    std::vector<ExistingLog> existing;
    for (const auto& entry : fs::directory_iterator(log_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("llama-build-", 0) != 0 || entry.path().extension() != ".log")
            continue;
        const auto modified = entry.last_write_time(ec);
        if (ec) {
            ec.clear();
            continue;
        }
        existing.push_back({entry.path(), modified});
    }
    std::sort(existing.begin(), existing.end(), [](const auto& a, const auto& b) {
        return a.modified < b.modified;
    });
    constexpr size_t kRetainedBuildLogs = 20;
    while (existing.size() >= kRetainedBuildLogs) {
        fs::remove(existing.front().path, ec);
        ec.clear();
        existing.erase(existing.begin());
    }

    std::string id = util::generate_uuid();
    if (id.size() > 8) id.resize(8);
    return log_dir / ("llama-build-" + std::to_string(util::now_ms()) + "-" +
                      id + ".log");
}

fs::path source_build_dir(const LlamaProvisionConfig& cfg) {
    const std::string variant = safe_path_component(cfg.platform) + "-"
        + safe_path_component(release_arch(cfg.arch)) + "-"
        + safe_path_component(cfg.accelerator);
    return fs::path(cfg.provision_dir) / "llama.cpp-src" / "build" / variant;
}

std::vector<fs::path> source_executable_candidates(const LlamaProvisionConfig& cfg) {
    const std::string exe = "llama-server" + exe_suffix(cfg.platform);
    const fs::path variant = source_build_dir(cfg) / "bin";
    const fs::path legacy = fs::path(cfg.provision_dir) / "llama.cpp-src" / "build" / "bin";
    return {
        variant / "Release" / exe, // MSVC multi-config
        variant / exe,             // Ninja / Unix Makefiles
        legacy / "Release" / exe, // pre-variant managed build
        legacy / exe,
    };
}

LlamaProvisionConfig normalized_config(LlamaProvisionConfig cfg) {
    cfg.requested_executable = strip_wrapping_quotes(cfg.requested_executable);
    if (cfg.requested_executable.empty()) cfg.requested_executable = "llama-server";
    cfg.install_method = normalize_llama_install_method(cfg.install_method);
    cfg.version = util::trim(cfg.version);
    if (cfg.version.empty()) cfg.version = "latest";
    if (cfg.platform.empty()) cfg.platform = current_runtime_platform();
    if (cfg.arch.empty()) cfg.arch = current_runtime_arch();
    cfg.accelerator = util::to_lower(util::trim(cfg.accelerator));
    if (cfg.accelerator.empty()) {
        // Without a live GPU probe, infer only what platform/arch make certain;
        // callers pass cuda/rocm after detection. Default to cpu otherwise.
        cfg.accelerator = detect_llama_accelerator(cfg.platform, cfg.arch,
                                                   /*has_cuda=*/false,
                                                   /*has_rocm=*/false);
    }
    cfg.cuda_arch = util::trim(cfg.cuda_arch);
    cfg.release_variant = util::to_lower(util::trim(cfg.release_variant));
    if (cfg.build_jobs < 0) cfg.build_jobs = 0;
    if (cfg.provision_dir.empty()) {
        cfg.provision_dir = (fs::path("data") / "runtimes" / "llama.cpp").string();
    }
    {
        std::error_code ec;
        const fs::path abs = fs::absolute(cfg.provision_dir, ec);
        if (!ec) cfg.provision_dir = abs.lexically_normal().string();
    }
    return cfg;
}

std::vector<std::string> accelerator_cmake_flags(const LlamaProvisionConfig& cfg) {
    std::vector<std::string> flags;
    if (cfg.accelerator == "cuda") {
        flags.push_back("-DGGML_CUDA=ON");
        if (!cfg.cuda_arch.empty()) {
            // This cache variable is visible before enable_language(CUDA), so
            // it fixes CUDA 13's removed sm_52 compiler-id default without a
            // global -arch flag. A global CMAKE_CUDA_FLAGS=-arch=sm_120 would
            // also leak into every target and break llama.cpp's sm_120a MXFP4
            // compilation even after upstream selects the correct target.
            flags.push_back("-DCMAKE_CUDA_ARCHITECTURES=" +
                            cuda_cmake_architectures(cfg.cuda_arch));
        }
    } else if (cfg.accelerator == "rocm" || cfg.accelerator == "hip") {
        flags.push_back("-DGGML_HIP=ON");
    } else if (cfg.accelerator == "vulkan") {
        flags.push_back("-DGGML_VULKAN=ON");
    } else if (cfg.accelerator == "metal") {
        flags.push_back("-DGGML_METAL=ON");
    } else if (cfg.accelerator == "openvino") {
        flags.push_back("-DGGML_OPENVINO=ON");
    } else if (cfg.accelerator == "sycl-fp32" || cfg.accelerator == "sycl-fp16") {
        // llama.cpp currently exposes one SYCL build toggle. Precision remains
        // a runtime/device capability; the wizard keeps FP32/FP16 as distinct
        // compatibility choices without inventing a nonexistent CMake flag.
        flags.push_back("-DGGML_SYCL=ON");
    }
    // cpu: no accelerator flag (default CPU backend).
    return flags;
}

// Candidate on-disk locations for the built/downloaded llama-server, most
// specific first. CMake's output dir differs between single-config generators
// (build/bin/llama-server) and multi-config MSVC (build/bin/Release/…).
std::vector<fs::path> managed_executable_candidates(const LlamaProvisionConfig& cfg) {
    const fs::path root(cfg.provision_dir);
    const std::string exe = "llama-server" + exe_suffix(cfg.platform);
    const std::vector<fs::path> release = {
        root / "release" / "bin" / exe,
        root / "release" / exe,  // legacy release layout
    };
    const std::vector<fs::path> source = source_executable_candidates(cfg);
    if (cfg.install_method == "release") {
        return release;
    }
    if (cfg.install_method == "source") return source;

    std::vector<fs::path> candidates = release;
    candidates.insert(candidates.end(), source.begin(), source.end());
    return candidates;
}

std::optional<LlamaRuntimeStatus> read_active_runtime(const LlamaProvisionConfig& cfg) {
    const fs::path marker = fs::path(cfg.provision_dir) / "active-runtime.json";
    std::ifstream in(marker);
    if (!in) return std::nullopt;
    try {
        const auto json = nlohmann::json::parse(in);
        const std::string method = normalize_llama_install_method(
            json.value("method", std::string{}));
        const std::string accelerator = util::to_lower(util::trim(
            json.value("accelerator", std::string{})));
        const fs::path executable(json.value("executable_path", std::string{}));
        if ((method != "release" && method != "source") || accelerator.empty()
            || executable.empty() || !file_exists(executable))
            return std::nullopt;
        LlamaProvisionConfig active_cfg = cfg;
        active_cfg.install_method = method;
        active_cfg.accelerator = accelerator;
        const fs::path normalized = fs::absolute(executable).lexically_normal();
        bool known_managed_path = false;
        for (const auto& candidate : managed_executable_candidates(active_cfg)) {
            if (fs::absolute(candidate).lexically_normal() == normalized) {
                known_managed_path = true;
                break;
            }
        }
        if (!known_managed_path) return std::nullopt;

        LlamaRuntimeStatus status;
        status.status = "ready";
        status.platform = cfg.platform;
        status.method = method;
        status.source_repo = kLlamaCppRepoUrl;
        status.version = validated_llama_version(
            json.value("version", std::string{}));
        status.managed = true;
        status.executable_path = normalized.string();
        status.accelerator = accelerator;
        status.variant = json.value("variant", accelerator);
        status.cuda_architecture =
            json.value("cuda_architecture", std::string{});
        status.build_log_path = json.value("build_log_path", std::string{});
        return status;
    } catch (...) {
        return std::nullopt;
    }
}

void write_active_runtime(const LlamaProvisionConfig& cfg,
                          const LlamaRuntimeStatus& status) {
    const fs::path marker = fs::path(cfg.provision_dir) / "active-runtime.json";
    const fs::path temp = fs::path(cfg.provision_dir) / "active-runtime.json.next";
    try {
        std::ofstream out(temp, std::ios::out | std::ios::trunc);
        if (!out) return;
        out << nlohmann::json{
            {"method", status.method},
            {"accelerator", status.accelerator},
            {"version", status.version},
            {"executable_path", status.executable_path},
            {"variant", status.variant.empty() ? status.accelerator : status.variant},
            {"cuda_architecture", status.cuda_architecture},
            {"build_log_path", status.build_log_path},
        }.dump(2);
        out.close();
        std::error_code ec;
        fs::remove(marker, ec);
        ec.clear();
        fs::rename(temp, marker, ec);
        if (ec) MM_WARN("Could not persist active llama.cpp runtime: {}", ec.message());
    } catch (const std::exception& e) {
        MM_WARN("Could not persist active llama.cpp runtime: {}", e.what());
    }
}

std::string join_shell_command(const std::vector<std::string>& args);

std::string default_capture_output(const std::vector<std::string>& args,
                                   const fs::path& cwd) {
    if (args.empty()) return {};
    std::string cmd = join_shell_command(args);
    if (!cwd.empty()) {
#ifdef _WIN32
        cmd = "cd /d \"" + cwd.string() + "\" && " + cmd;
#else
        cmd = "cd '" + cwd.string() + "' && " + cmd;
#endif
    }
    // llama-server historically prints `--version` to stderr, so merge it into
    // the captured stream instead of discarding it.
#ifdef _WIN32
    FILE* f = _popen((cmd + " 2>&1").c_str(), "r");
#else
    FILE* f = ::popen((cmd + " 2>&1").c_str(), "r");
#endif
    if (!f) return {};
    std::string output;
    char buf[512];
    while (fgets(buf, static_cast<int>(sizeof(buf)), f)) output += buf;
#ifdef _WIN32
    _pclose(f);
#else
    ::pclose(f);
#endif
    return output;
}

std::string default_capture_first_line(const std::vector<std::string>& args,
                                       const fs::path& cwd) {
    const std::string output = default_capture_output(args, cwd);
    for (const auto& line : util::split(output, '\n')) {
        const std::string trimmed = util::trim(line);
        if (!trimmed.empty()) return trimmed;
    }
    return {};
}

std::string shell_quote(const std::string& value) {
#ifdef _WIN32
    std::string out = "\"";
    for (char ch : value) { if (ch == '"') out += "\\\""; else out.push_back(ch); }
    out += "\"";
    return out;
#else
    std::string out = "'";
    for (char ch : value) { if (ch == '\'') out += "'\\''"; else out.push_back(ch); }
    out += "'";
    return out;
#endif
}

std::string join_shell_command(const std::vector<std::string>& args) {
    std::string out;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) out += " ";
        out += shell_quote(args[i]);
    }
    return out;
}

// GitHub release-tag probe for ggml-org/llama.cpp. Windows uses PowerShell
// (system cert store); POSIX uses python3 + urllib. "" on any failure.
std::string default_fetch_latest(const LlamaProvisionConfig& cfg,
                                 const LlamaCommandRunner::CaptureFn& capture) {
    if (!capture) return {};
    const std::string url =
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";
    if (!cfg.bypass_environment_checks && cfg.platform == "windows") {
        const std::string ps =
            "try { (Invoke-RestMethod -Uri '" + url +
            "' -Headers @{ 'User-Agent' = 'mantic-mind' }).tag_name } catch { '' }";
        return util::trim(capture({"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                                   "-Command", ps}, {}));
    }
    const std::string program =
        "import json, urllib.request\n"
        "req = urllib.request.Request(\"" + url + "\", headers={\"User-Agent\": \"mantic-mind\"})\n"
        "d = json.load(urllib.request.urlopen(req, timeout=10))\n"
        "print(d.get(\"tag_name\", \"\"))\n";
    return util::trim(capture({"python3", "-c", program}, {}));
}

std::vector<std::string> default_fetch_release_assets(const LlamaProvisionConfig& cfg,
                                                      const std::string& tag) {
    if (tag.empty()) return {};
    const std::string url =
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/" + tag;
    std::string output;
    if (cfg.platform == "windows") {
        const std::string ps =
            "try { (Invoke-RestMethod -Uri '" + url +
            "' -Headers @{ 'User-Agent' = 'mantic-mind' }).assets | "
            "ForEach-Object { $_.name } } catch { '' }";
        output = default_capture_output(
            {"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps}, {});
    } else {
        const std::string program =
            "import json, urllib.request\n"
            "req = urllib.request.Request(\"" + url +
            "\", headers={\"User-Agent\": \"mantic-mind\"})\n"
            "d = json.load(urllib.request.urlopen(req, timeout=10))\n"
            "print('\\n'.join(a.get('name', '') for a in d.get('assets', [])))\n";
        output = default_capture_output({"python3", "-c", program}, {});
    }

    std::vector<std::string> assets;
    for (const auto& line : util::split(output, '\n')) {
        const std::string name = util::trim(line);
        if (!name.empty()) assets.push_back(name);
    }
    return assets;
}

bool any_asset_matches(const std::vector<std::string>& assets,
                       const std::string& pattern) {
    const std::regex rx(pattern, std::regex::icase);
    return std::any_of(assets.begin(), assets.end(), [&](const std::string& name) {
        return std::regex_match(name, rx);
    });
}

bool installed_version_matches_tag(const std::string& installed,
                                   const std::string& tag) {
    if (installed.find(tag) != std::string::npos) return true;
    // Official tags are b<build-number>, while llama-server --version often
    // prints only "version: <build-number> (<commit>)".
    if (tag.size() > 1 && (tag.front() == 'b' || tag.front() == 'B') &&
        std::all_of(tag.begin() + 1, tag.end(), [](unsigned char ch) {
            return std::isdigit(ch) != 0;
        })) {
        const std::string build = tag.substr(1);
        return std::regex_search(installed,
                                 std::regex("(^|[^0-9])" + build + "([^0-9]|$)"));
    }
    return false;
}

bool is_llama_release_tag(const std::string& tag) {
    return tag.size() > 1 && (tag.front() == 'b' || tag.front() == 'B') &&
        std::all_of(tag.begin() + 1, tag.end(), [](unsigned char ch) {
            return std::isdigit(ch) != 0;
        });
}

// Build a plan that downloads the official assets matching this node. Current
// Windows releases are modular: the CPU archive supplies llama-server and an
// accelerator archive supplies the dynamic backend (plus cudart for CUDA).
// Linux/macOS accelerator archives are self-contained. The complete runnable
// directory is normalized to <root>/release/bin.
std::vector<LlamaInstallStep> build_release_plan(const LlamaProvisionConfig& cfg) {
    const fs::path root(cfg.provision_dir);
    const fs::path rel = root / "release";
    const fs::path stage = rel / "staging";
    const fs::path next_bin = rel / "bin.next";
    const fs::path bin = rel / "bin";
    const std::string asset_arch = release_arch(cfg.arch);
    const std::string tag_filter =
        cfg.version == "latest" ? std::string("releases/latest")
                                : "releases/tags/" + cfg.version;
    const std::string api =
        "https://api.github.com/repos/ggml-org/llama.cpp/" + tag_filter;
    std::vector<LlamaInstallStep> plan;
    if (cfg.platform == "windows") {
        std::ostringstream ps;
        ps << "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; ";
        ps << "$api='" << api << "'; $arch='" << asset_arch
           << "'; $backend='" << cfg.accelerator << "'; $variant='"
           << cfg.release_variant << "'; ";
        ps << "$rel='" << rel.string() << "'; $stage='" << stage.string()
           << "'; $next='" << next_bin.string() << "'; $bin='" << bin.string() << "'; ";
        ps << "New-Item -ItemType Directory -Force -Path $rel | Out-Null; ";
        ps << "$r=Invoke-RestMethod -Uri $api -Headers @{'User-Agent'='mantic-mind'}; $assets=@($r.assets); ";
        ps << "$archRx=[regex]::Escape($arch); ";
        ps << "$base=@($assets | Where-Object {$_.name -match ('^llama-.*-bin-win-cpu-'+$archRx+'\\.zip$')})[0]; ";
        ps << "if ($backend -eq 'openvino') {$full=$assets | Where-Object {$_.name -match ('^llama-.*-bin-win-openvino-.*-'+$archRx+'\\.zip$')} | Select-Object -First 1; if($null -eq $full){throw 'No matching llama.cpp Windows OpenVINO release'}; $selected=@($full)} else { ";
        ps << "if ($null -eq $base) { throw ('No llama.cpp Windows base release for '+$arch) }; $selected=@($base); ";
        ps << "if ($backend -eq 'cuda') { ";
        ps << "$gpu=@(); foreach($a in $assets) { if($a.name -match ('^llama-.*-bin-win-cuda-([0-9.]+)-'+$archRx+'\\.zip$')) {$gpu += [pscustomobject]@{Asset=$a; Version=[version]$Matches[1]}} }; ";
        ps << "if($variant -eq 'cuda-12'){$gpu=@($gpu | Where-Object {$_.Version.Major -eq 12})} elseif($variant -eq 'cuda-13'){$gpu=@($gpu | Where-Object {$_.Version.Major -eq 13})}; ";
        ps << "$supported=$null; try {$txt=(& nvidia-smi 2>$null) -join \"`n\"; $m=[regex]::Match($txt,'CUDA Version:\\s*([0-9.]+)'); if($m.Success){$supported=[version]$m.Groups[1].Value}} catch {}; ";
        ps << "if($null -ne $supported){$choice=$gpu | Where-Object {$_.Version -le $supported} | Sort-Object Version -Descending | Select-Object -First 1} else {$choice=$gpu | Sort-Object Version | Select-Object -First 1}; ";
        ps << "if($null -eq $choice){throw ('No compatible llama.cpp Windows CUDA release for '+$arch)}; $selected += $choice.Asset; ";
        ps << "$cudartName=('cudart-llama-bin-win-cuda-'+$choice.Version+'-'+$arch+'.zip'); $cudart=$assets | Where-Object {$_.name -eq $cudartName} | Select-Object -First 1; if($null -ne $cudart){$selected += $cudart}; ";
        ps << "} elseif ($backend -eq 'vulkan') {$addon=$assets | Where-Object {$_.name -match ('^llama-.*-bin-win-vulkan-'+$archRx+'\\.zip$')} | Select-Object -First 1; if($null -eq $addon){throw 'No matching llama.cpp Windows Vulkan release'}; $selected += $addon; ";
        ps << "} elseif ($backend -eq 'rocm') {$addon=$assets | Where-Object {$_.name -match ('^llama-.*-bin-win-(hip|rocm).*-'+$archRx+'\\.zip$')} | Select-Object -First 1; if($null -eq $addon){throw 'No matching llama.cpp Windows ROCm/HIP release'}; $selected += $addon; ";
        ps << "} elseif ($backend -ne 'cpu') {throw ('No matching llama.cpp Windows release backend: '+$backend)} }; ";
        ps << "Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue; Remove-Item -LiteralPath $next -Recurse -Force -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force -Path $stage,$next | Out-Null; ";
        ps << "$n=0; foreach($a in $selected){$n++; $archive=Join-Path $rel ('asset-'+$n+'.zip'); Write-Host ('Downloading '+$a.name); Invoke-WebRequest -Uri $a.browser_download_url -OutFile $archive; Expand-Archive -LiteralPath $archive -DestinationPath $stage -Force}; ";
        ps << "$exe=Get-ChildItem -LiteralPath $stage -Recurse -File -Filter 'llama-server.exe' | Select-Object -First 1; if($null -eq $exe){throw 'llama-server.exe not found in selected release assets'}; ";
        ps << "Copy-Item -Path (Join-Path $exe.Directory.FullName '*') -Destination $next -Recurse -Force; Get-ChildItem -LiteralPath $stage -Recurse -File -Filter '*.dll' | Copy-Item -Destination $next -Force; ";
        ps << "if(-not (Test-Path -LiteralPath (Join-Path $next 'llama-server.exe'))){Copy-Item -LiteralPath $exe.FullName -Destination $next -Force}; ";
        ps << "Remove-Item -LiteralPath $bin -Recurse -Force -ErrorAction SilentlyContinue; Move-Item -LiteralPath $next -Destination $bin -Force";
        plan.push_back({"Downloading matched llama.cpp release assets",
                        {"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                         "-Command", ps.str()}, root, false});
    } else {
        const std::string py = R"PY(import json
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

api, platform, arch, backend, variant, rel_arg = sys.argv[1:]
rel = pathlib.Path(rel_arg)
stage = rel / "staging"
next_bin = rel / "bin.next"
bin_dir = rel / "bin"
req = urllib.request.Request(api, headers={"User-Agent": "mantic-mind"})
with urllib.request.urlopen(req, timeout=20) as response:
    assets = json.load(response).get("assets", [])

def matches(pattern):
    return [a for a in assets if re.match(pattern, a.get("name", ""))]

def cuda_version():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL, text=True)
        match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
        return tuple(int(x) for x in match.group(1).split(".")) if match else None
    except Exception:
        return None

def choose_cuda(candidates):
    versioned = []
    for asset in candidates:
        match = re.search(r"-cuda-([0-9.]+)-", asset["name"])
        if match:
            versioned.append((tuple(int(x) for x in match.group(1).split(".")), asset))
    supported = cuda_version()
    eligible = [item for item in versioned if supported is None or item[0] <= supported]
    if not eligible:
        return None
    eligible.sort(key=lambda item: item[0], reverse=supported is not None)
    return eligible[0][1]

selected = []
arch_rx = re.escape(arch)
if platform == "linux":
    if backend == "cpu":
        candidates = matches(r"^llama-.*-bin-ubuntu-" + arch_rx + r"\.tar\.gz$")
    elif backend == "vulkan":
        candidates = matches(r"^llama-.*-bin-ubuntu-vulkan-" + arch_rx + r"\.tar\.gz$")
    elif backend == "rocm":
        candidates = matches(r"^llama-.*-bin-ubuntu-rocm-.*-" + arch_rx + r"\.tar\.gz$")
    elif backend == "cuda":
        candidates = matches(r"^llama-.*-bin-ubuntu-cuda-[0-9.]+-" + arch_rx + r"\.tar\.gz$")
        if variant in ("cuda-12", "cuda-13"):
            major = variant.split("-")[1]
            candidates = [a for a in candidates if re.search(r"-cuda-" + major + r"\.", a["name"])]
        choice = choose_cuda(candidates)
        candidates = [choice] if choice else []
    elif backend == "openvino":
        candidates = matches(r"^llama-.*-bin-ubuntu-openvino-.*-" + arch_rx + r"\.tar\.gz$")
    else:
        candidates = []
elif platform == "macos" and backend in ("metal", "cpu"):
    candidates = matches(r"^llama-.*-bin-macos-" + arch_rx + r"\.tar\.gz$")
else:
    candidates = []

if not candidates:
    raise RuntimeError(f"No official llama.cpp release matches {platform}/{arch}/{backend}")
selected.append(candidates[0])

rel.mkdir(parents=True, exist_ok=True)
shutil.rmtree(stage, ignore_errors=True)
shutil.rmtree(next_bin, ignore_errors=True)
stage.mkdir(parents=True)
next_bin.mkdir(parents=True)
for index, asset in enumerate(selected, 1):
    name = asset["name"]
    archive = rel / f"asset-{index}-" / pathlib.Path(name).name
    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name}", flush=True)
    request = urllib.request.Request(asset["browser_download_url"], headers={"User-Agent": "mantic-mind"})
    with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(stage)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(stage)
    else:
        raise RuntimeError(f"Unsupported release archive: {name}")

servers = list(stage.rglob("llama-server"))
if not servers:
    raise RuntimeError("llama-server not found in selected release assets")
server = servers[0]
shutil.copytree(server.parent, next_bin, dirs_exist_ok=True, symlinks=True)
server_target = next_bin / "llama-server"
if not server_target.exists():
    shutil.copy2(server, server_target)
server_target.chmod(server_target.stat().st_mode | 0o111)
shutil.rmtree(bin_dir, ignore_errors=True)
next_bin.replace(bin_dir)
)PY";
        plan.push_back({"Downloading matched llama.cpp release asset",
                        {"python3", "-c", py, api, cfg.platform, asset_arch,
                         cfg.accelerator, cfg.release_variant, rel.string()}, root, false});
    }
    return plan;
}

std::vector<LlamaInstallStep> build_source_plan(const LlamaProvisionConfig& cfg, bool upgrade) {
    const fs::path root(cfg.provision_dir);
    const fs::path src = root / "llama.cpp-src";
    // Keep each OS/architecture/accelerator in its own CMake cache. Reusing a
    // CPU/Vulkan cache for CUDA (or vice versa) produces deceptively successful
    // reconfigures followed by broken or incorrectly linked binaries.
    const fs::path build = source_build_dir(cfg);
    const std::string ref = cfg.version == "latest" ? std::string("master") : cfg.version;
    const int build_jobs = cfg.build_jobs > 0
        ? cfg.build_jobs
        : ((cfg.accelerator == "cuda" || cfg.accelerator == "rocm") ? 2 : 4);
    const auto cuda_targets = cuda_arch_targets(cfg.cuda_arch);
    std::vector<std::string> cuda_feature_targets;
    cuda_feature_targets.reserve(cuda_targets.size());
    for (const auto& target : cuda_targets)
        cuda_feature_targets.push_back(cuda_feature_target(target));
    const std::string cuda_target_arg = util::join(cuda_feature_targets, ",");
    const std::string cuda_probe_arch = cuda_feature_targets.empty()
        ? std::string{}
        : cuda_feature_targets.front();

    std::vector<LlamaInstallStep> plan;
    if (cfg.platform == "windows" && !cfg.bypass_environment_checks) {
        std::ostringstream ps;
        ps << "$ErrorActionPreference='Stop'; $required=@('git','cmake'); ";
        ps << "$missing=@($required | Where-Object {-not (Get-Command $_ -ErrorAction SilentlyContinue)}); ";
        // CMake can discover Visual Studio through the registry even when cl.exe
        // is not on this PowerShell process's PATH, so let configure validate it.
        if (cfg.accelerator == "cuda") {
            ps << "$configuredNvcc=if($env:CUDACXX){Get-Command $env:CUDACXX -ErrorAction SilentlyContinue}"
                  "else{Get-Command nvcc -ErrorAction SilentlyContinue}; "
                  "if(-not $configuredNvcc){ "
                  "$missing += 'nvcc (CUDA Toolkit; the display driver alone is insufficient)' }; ";
        } else if (cfg.accelerator == "rocm" || cfg.accelerator == "hip") {
            ps << "if(-not (Get-Command hipcc -ErrorAction SilentlyContinue)){ "
                  "$missing += 'hipcc (ROCm/HIP SDK)' }; ";
        } else if (cfg.accelerator == "vulkan") {
            ps << "if(-not (Get-Command glslc -ErrorAction SilentlyContinue)){ "
                  "$missing += 'glslc (Vulkan SDK)' }; ";
        } else if (cfg.accelerator == "sycl-fp32" ||
                   cfg.accelerator == "sycl-fp16") {
            ps << "if(-not (Get-Command icx -ErrorAction SilentlyContinue)){ "
                  "$missing += 'icx (Intel oneAPI DPC++ compiler)' }; ";
        } else if (cfg.accelerator == "openvino") {
            ps << "if(-not $env:OpenVINO_DIR -and -not $env:INTEL_OPENVINO_DIR){ "
                  "$missing += 'OpenVINO toolkit environment' }; ";
        }
        ps << "if($missing.Count){throw ('Missing llama.cpp source-build prerequisites: '+"
              "($missing -join ', '))}; cmake --version | Select-Object -First 1; ";
        if (cfg.accelerator == "cuda") {
            ps << "$nvcc=if($env:CUDACXX){(Get-Command $env:CUDACXX -ErrorAction Stop).Source}"
                  "else{(Get-Command nvcc -ErrorAction Stop).Source}; ";
            ps << "& $nvcc --version | Select-Object -Last 1; ";
            if (!cuda_targets.empty()) {
                ps << "$supported=((& $nvcc --list-gpu-arch 2>$null) -join ' '); "
                      "$requested='" << cuda_target_arg << "' -split ','; "
                      "foreach($arch in $requested){$target='compute_'+$arch; "
                      "if($supported -notmatch ('(^|\\s)'+[regex]::Escape($target)+'(\\s|$)')){"
                      "$hint=if($arch -eq '120a'){' CUDA Toolkit 12.8 or newer is required for sm_120a.'}else{''}; "
                      "throw ('Selected NVCC '+$nvcc+' does not support compute_'+$arch+'.'+$hint+"
                      "' The CUDA version shown by nvidia-smi is driver compatibility, not the installed compiler version.')}}; ";
                ps << "$probe=Join-Path $env:TEMP ('mantic-mind-cuda-'+[guid]::NewGuid().ToString('N')+'.cu'); "
                      "$object=$probe+'.obj'; try { "
                      "Set-Content -LiteralPath $probe -Encoding Ascii -Value 'extern \"C\" __global__ void mm_cuda_probe() {}'; "
                      "$probeArch='" << cuda_probe_arch << "'; "
                      "& $nvcc ('-arch=sm_'+$probeArch) -c $probe -o $object; "
                      "if($LASTEXITCODE -ne 0){throw ('CUDA compiler/assembler smoke test failed for sm_'+$probeArch+'. Ensure nvcc and ptxas come from the same CUDA Toolkit.')} "
                      "} finally {Remove-Item -LiteralPath $probe,$object -Force -ErrorAction SilentlyContinue}; ";
            }
        }
        plan.push_back({"Checking source-build prerequisites",
                        {"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                         "-Command", ps.str()}, root, false});
    } else if (!cfg.bypass_environment_checks) {
        const std::string preflight = R"SH(set -eu
backend="$1"
cuda_arches="$2"
cuda_probe_arch="$3"
missing=""
for tool in git cmake; do
  command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
cxx="${CXX:-c++}"
command -v "$cxx" >/dev/null 2>&1 || missing="$missing C++-compiler"
nvcc_path=""
if [ "$backend" = "cuda" ]; then
  if [ -n "${CUDACXX:-}" ]; then
    nvcc_path=$(command -v "$CUDACXX" 2>/dev/null || true)
    if [ -z "$nvcc_path" ] && [ -x "$CUDACXX" ]; then nvcc_path="$CUDACXX"; fi
  else
    nvcc_path=$(command -v nvcc 2>/dev/null || true)
  fi
  [ -n "$nvcc_path" ] || missing="$missing nvcc-CUDA-Toolkit"
elif [ "$backend" = "rocm" ] || [ "$backend" = "hip" ]; then
  command -v hipcc >/dev/null 2>&1 || missing="$missing hipcc-ROCm-SDK"
elif [ "$backend" = "vulkan" ]; then
  command -v glslc >/dev/null 2>&1 || missing="$missing glslc-Vulkan-SDK"
elif [ "$backend" = "openvino" ]; then
  [ -n "${OpenVINO_DIR:-}${INTEL_OPENVINO_DIR:-}" ] || missing="$missing OpenVINO-toolkit-environment"
elif [ "$backend" = "sycl-fp32" ] || [ "$backend" = "sycl-fp16" ]; then
  command -v icpx >/dev/null 2>&1 || missing="$missing icpx-oneAPI-DPC++-compiler"
fi
if [ -n "$missing" ]; then
  echo "Missing llama.cpp source-build prerequisites:$missing" >&2
  if [ "$backend" = "cuda" ]; then
    echo "CUDA GPU visibility (nvidia-smi) is not enough; install the CUDA Toolkit inside this Linux/WSL distribution so nvcc is on PATH." >&2
  fi
  exit 2
fi
cmake --version | head -n 1
"$cxx" --version | head -n 1
if [ "$backend" = "cuda" ]; then
  "$nvcc_path" --version | tail -n 1
  if [ -n "$cuda_arches" ]; then
    supported=$("$nvcc_path" --list-gpu-arch 2>/dev/null || true)
    for arch in $(printf '%s' "$cuda_arches" | tr ',' ' '); do
      if ! printf '%s\n' "$supported" | tr '[:space:]' '\n' | grep -Fqx "compute_$arch"; then
        echo "Selected NVCC $nvcc_path does not support compute_$arch." >&2
        if [ "$arch" = "120a" ]; then
          echo "CUDA Toolkit 12.8 or newer is required to compile for sm_120a." >&2
        fi
        echo "The CUDA version shown by nvidia-smi is driver compatibility, not the installed compiler version." >&2
        exit 3
      fi
    done
    probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/mantic-mind-cuda-probe.XXXXXX")
    trap 'rm -rf "$probe_dir"' EXIT HUP INT TERM
    printf '%s\n' 'extern "C" __global__ void mm_cuda_probe() {}' > "$probe_dir/probe.cu"
    if ! "$nvcc_path" "-arch=sm_$cuda_probe_arch" -c "$probe_dir/probe.cu" -o "$probe_dir/probe.o"; then
      echo "CUDA compiler/assembler smoke test failed for sm_$cuda_probe_arch." >&2
      echo "Ensure nvcc and ptxas come from the same CUDA Toolkit installation." >&2
      exit 4
    fi
    rm -rf "$probe_dir"
    trap - EXIT HUP INT TERM
  fi
  if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    echo "WSL detected: using the Linux CUDA toolkit and a conservative parallel build."
    command -v nvidia-smi >/dev/null 2>&1 || echo "Warning: nvidia-smi is not visible inside WSL; the build can finish, but CUDA runtime validation may fail." >&2
  fi
fi
available_kb=$(df -Pk . 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "${available_kb:-}" ] && [ "$available_kb" -lt 4194304 ]; then
  echo "Warning: less than 4 GiB is free for the llama.cpp source build." >&2
fi
)SH";
        plan.push_back({"Checking source-build prerequisites",
                        {"sh", "-c", preflight, "mantic-mind-llama-preflight",
                         cfg.accelerator, cuda_target_arg, cuda_probe_arch}, root, false});
    }
    if (!fs::exists(src / ".git"))
        plan.push_back({"Cloning llama.cpp source",
                        {"git", "clone", kLlamaCppRepoUrl, src.string()}, root, false});
    plan.push_back({"Fetching tags", {"git", "fetch", "--tags", "--all", "--prune"}, src, false});
    plan.push_back({"Checking out " + ref, {"git", "checkout", ref}, src, false});
    if (cfg.version == "latest")
        plan.push_back({"Updating source", {"git", "pull", "--ff-only"}, src, true});

    if (cfg.accelerator == "cuda") {
        // CMake persists CMAKE_CUDA_COMPILER across attempts. This is especially
        // hazardous in WSL after upgrading from distro /usr/bin/nvcc to a newer
        // toolkit selected through CUDACXX/PATH: the old compiler keeps winning
        // even though all live preflight probes see the new one. Clear only the
        // generated configure state; the source checkout remains intact and
        // the next configure/build selects the active toolkit afresh.
        plan.push_back({"Clearing cached CUDA compiler selection",
                        {"cmake", "-E", "remove_directory",
                         (build / "CMakeFiles").string()}, src, true});
        plan.push_back({"Clearing cached CUDA configure state",
                        {"cmake", "-E", "remove",
                         (build / "CMakeCache.txt").string()}, src, true});
    }

    std::vector<std::string> configure = {
        "cmake", "-S", src.string(), "-B", build.string(),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
    };
    for (auto& f : accelerator_cmake_flags(cfg)) configure.push_back(f);
    for (auto& f : cfg.cmake_args) configure.push_back(f);
    plan.push_back({"Configuring llama.cpp (CMake)", configure, src, false});

    plan.push_back({"Building llama-server",
                    {"cmake", "--build", build.string(), "--config", "Release",
                     "--target", "llama-server", "--parallel", std::to_string(build_jobs)},
                    src, false});
    if (cfg.platform == "windows") {
        const fs::path multi = build / "bin" / "Release" / "llama-server.exe";
        const fs::path single = build / "bin" / "llama-server.exe";
        const std::string ps =
            "$ErrorActionPreference='Stop'; $exe='" + multi.string() +
            "'; if(-not (Test-Path -LiteralPath $exe)){ $exe='" + single.string() +
            "' }; if(-not (Test-Path -LiteralPath $exe)){ throw 'llama-server.exe was not produced' }; & $exe --version";
        plan.push_back({"Validating built llama-server",
                        {"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                         "-Command", ps}, root, false});
    } else {
        const fs::path executable = build / "bin" / "llama-server";
        plan.push_back({"Validating built llama-server",
                        {executable.string(), "--version"}, executable.parent_path(), false});
    }
    (void)upgrade;  // git pull above already moves "latest" forward; a pinned
                    // ref rebuilds identically, so upgrade needs no extra steps.
    return plan;
}

} // namespace

std::string normalize_llama_install_method(const std::string& method) {
    const std::string m = util::to_lower(util::trim(method));
    if (m == "release" || m == "source") return m;
    return "auto";
}

std::vector<std::string> llama_release_accelerators(
    const std::vector<std::string>& asset_names,
    const LlamaProvisionConfig& raw_cfg) {
    const LlamaProvisionConfig cfg = normalized_config(raw_cfg);
    const std::string arch = release_arch(cfg.arch);
    // Architecture values accepted by the config are restricted to simple
    // identifiers, but escape regex punctuation defensively for test hooks.
    std::string arch_rx;
    for (const char ch : arch) {
        if (std::string{".^$|()[]{}*+?\\"}.find(ch) != std::string::npos)
            arch_rx.push_back('\\');
        arch_rx.push_back(ch);
    }

    std::set<std::string> available;
    if (cfg.platform == "windows") {
        const bool base = any_asset_matches(
            asset_names, "^llama-.*-bin-win-cpu-" + arch_rx + "\\.zip$");
        if (base) available.insert("cpu");
        if (base && any_asset_matches(
                asset_names, "^llama-.*-bin-win-cuda-[0-9.]+-" + arch_rx + "\\.zip$") &&
            any_asset_matches(
                asset_names, "^cudart-llama-bin-win-cuda-[0-9.]+-" + arch_rx + "\\.zip$"))
            available.insert("cuda");
        if (base && any_asset_matches(
                asset_names, "^llama-.*-bin-win-vulkan-" + arch_rx + "\\.zip$"))
            available.insert("vulkan");
        if (base && any_asset_matches(
                asset_names, "^llama-.*-bin-win-(hip|rocm).*-" + arch_rx + "\\.zip$"))
            available.insert("rocm");
        if (any_asset_matches(
                asset_names, "^llama-.*-bin-win-openvino-.*-" + arch_rx + "\\.zip$"))
            available.insert("openvino");
    } else if (cfg.platform == "linux") {
        const std::string suffix = "-" + arch_rx + "(\\.tar\\.gz|\\.zip)$";
        if (any_asset_matches(asset_names, "^llama-.*-bin-ubuntu" + suffix))
            available.insert("cpu");
        if (any_asset_matches(asset_names, "^llama-.*-bin-ubuntu-vulkan" + suffix))
            available.insert("vulkan");
        if (any_asset_matches(asset_names, "^llama-.*-bin-ubuntu-rocm-.*" + suffix))
            available.insert("rocm");
        if (any_asset_matches(asset_names, "^llama-.*-bin-ubuntu-cuda-[0-9.]+" + suffix))
            available.insert("cuda");
        if (any_asset_matches(asset_names, "^llama-.*-bin-ubuntu-openvino-.*" + suffix))
            available.insert("openvino");
    } else if (cfg.platform == "macos") {
        if (any_asset_matches(asset_names,
                              "^llama-.*-bin-macos-" + arch_rx +
                                  "(\\.tar\\.gz|\\.zip)$")) {
            available.insert("metal");
            available.insert("cpu");
        }
    }

    const std::vector<std::string> preference = cfg.platform == "linux"
        ? std::vector<std::string>{"vulkan", "openvino", "cpu", "rocm", "cuda"}
        : (cfg.platform == "windows"
            ? std::vector<std::string>{"cuda", "vulkan", "openvino", "cpu", "rocm"}
            : std::vector<std::string>{"metal", "cpu"});
    std::vector<std::string> out;
    for (const auto& accelerator : preference)
        if (available.count(accelerator)) out.push_back(accelerator);
    return out;
}

std::vector<LlamaRuntimeVariant> llama_runtime_variants(
    const std::vector<std::string>& asset_names,
    const LlamaProvisionConfig& raw_cfg) {
    const LlamaProvisionConfig cfg = normalized_config(raw_cfg);
    const std::string arch = release_arch(cfg.arch);
    std::string arch_rx;
    for (const char ch : arch) {
        if (std::string{".^$|()[]{}*+?\\"}.find(ch) != std::string::npos)
            arch_rx.push_back('\\');
        arch_rx.push_back(ch);
    }
    auto asset = [&](const std::string& pattern) -> std::string {
        const std::regex rx(pattern, std::regex::icase);
        for (const auto& name : asset_names)
            if (std::regex_match(name, rx)) return name;
        return {};
    };

    const bool win = cfg.platform == "windows";
    const bool linux = cfg.platform == "linux";
    const bool mac = cfg.platform == "macos";
    const bool x64 = arch == "x64";
    const bool arm64 = arch == "arm64";
    const bool cpu_arch = arch == "x86" || x64 || arm64 || arch == "s390x";
    const std::string win_base = win
        ? asset("^llama-.*-bin-win-cpu-" + arch_rx + "\\.zip$") : std::string{};

    std::vector<LlamaRuntimeVariant> out;
    auto add = [&](std::string id, std::string label, std::string backend,
                   bool supported, bool source, std::string release_asset) {
        LlamaRuntimeVariant v;
        v.id = std::move(id);
        v.label = std::move(label);
        v.backend = std::move(backend);
        v.platform_supported = supported;
        v.source_supported = supported && source;
        v.release_asset = std::move(release_asset);
        v.release_available = supported && !v.release_asset.empty();
        if (!supported) {
            v.reason = "not supported for " + cfg.platform + "/" + arch;
        } else if (v.release_available) {
            v.reason = "complete compatible release is available";
        } else if (v.source_supported) {
            v.reason = "no complete server release; source compilation is available";
        } else {
            v.reason = "no complete server release or supported source build";
        }
        out.push_back(std::move(v));
    };

    auto windows_modular = [&](const std::string& addon_pattern) {
        const std::string addon = asset(addon_pattern);
        return !win_base.empty() && !addon.empty() ? addon : std::string{};
    };
    auto windows_cuda = [&](const std::string& major) {
        const std::string addon = windows_modular(
            "^llama-.*-bin-win-cuda-" + major + "[.][0-9.]*-" + arch_rx + "\\.zip$");
        const std::string runtime = asset(
            "^cudart-llama-bin-win-cuda-" + major + "[.][0-9.]*-" + arch_rx + "\\.zip$");
        return !addon.empty() && !runtime.empty() ? addon : std::string{};
    };
    const std::string archive_suffix = "-" + arch_rx + "(\\.tar\\.gz|\\.zip)$";

    add("cuda-12", "CUDA 12", "cuda", (win || linux) && x64, true,
        win ? windows_cuda("12")
            : asset("^llama-.*-bin-ubuntu-cuda-12[.][0-9.]*" + archive_suffix));
    add("cuda-13", "CUDA 13", "cuda", (win || linux) && x64, true,
        win ? windows_cuda("13")
            : asset("^llama-.*-bin-ubuntu-cuda-13[.][0-9.]*" + archive_suffix));
    add("vulkan", "Vulkan", "vulkan", (win || linux) && (x64 || arm64), true,
        win ? windows_modular("^llama-.*-bin-win-vulkan-" + arch_rx + "\\.zip$")
            : asset("^llama-.*-bin-ubuntu-vulkan" + archive_suffix));
    add("rocm", "ROCm", "rocm", linux && x64, true,
        linux ? asset("^llama-.*-bin-ubuntu-rocm-.*" + archive_suffix) : std::string{});
    add("openvino", "OpenVINO", "openvino", (win || linux) && x64, true,
        win ? asset("^llama-.*-bin-win-openvino-.*-" + arch_rx + "\\.zip$")
            : asset("^llama-.*-bin-ubuntu-openvino-.*" + archive_suffix));
    add("sycl-fp32", "SYCL FP32", "sycl-fp32", (win || linux) && x64, true, {});
    add("sycl-fp16", "SYCL FP16", "sycl-fp16", (win || linux) && x64, true, {});
    add("metal", "Metal", "metal", mac && arm64, true,
        mac && arm64 ? asset("^llama-.*-bin-macos-" + arch_rx + "(\\.tar\\.gz|\\.zip)$")
                     : std::string{});
    add("hip", "HIP", "rocm", win && x64, true,
        win ? windows_modular("^llama-.*-bin-win-(hip|rocm).*-" + arch_rx + "\\.zip$")
            : std::string{});
    std::string cpu_release;
    if (win) cpu_release = win_base;
    else if (linux) cpu_release = asset("^llama-.*-bin-ubuntu" + archive_suffix);
    else if (mac) cpu_release = asset("^llama-.*-bin-macos-" + arch_rx + "(\\.tar\\.gz|\\.zip)$");
    add("cpu", "CPU", "cpu", (win || linux || mac) && cpu_arch, true, cpu_release);

    auto preferred = std::find_if(out.begin(), out.end(), [&](const auto& v) {
        return v.release_available &&
            (v.backend == cfg.accelerator || v.id == cfg.release_variant);
    });
    if (preferred == out.end()) {
        for (const std::string& id : {std::string{"vulkan"}, std::string{"openvino"},
                                      std::string{"cpu"}}) {
            preferred = std::find_if(out.begin(), out.end(), [&](const auto& v) {
                return v.id == id && v.release_available;
            });
            if (preferred != out.end()) break;
        }
    }
    if (preferred == out.end())
        preferred = std::find_if(out.begin(), out.end(), [](const auto& v) {
            return v.release_available;
        });
    if (preferred != out.end()) preferred->recommended = true;
    return out;
}

fs::path managed_llama_executable_path(const LlamaProvisionConfig& raw_cfg) {
    const LlamaProvisionConfig cfg = normalized_config(raw_cfg);
    return managed_executable_candidates(cfg).front();
}

std::vector<LlamaInstallStep> build_llama_install_plan(const LlamaProvisionConfig& raw_cfg,
                                                       bool upgrade) {
    const LlamaProvisionConfig cfg = normalized_config(raw_cfg);
    if (cfg.install_method == "source") return build_source_plan(cfg, upgrade);
    // Both release and auto begin with the environment-matched release plan.
    // The provisioner adds a source attempt after this plan only for auto.
    return build_release_plan(cfg);
}

std::string resolve_llama_executable(const std::string& raw_executable) {
    const std::string executable = strip_wrapping_quotes(raw_executable);
    if (executable.empty()) return {};
    if (is_path_like(executable)) {
        const fs::path p(executable);
        return file_exists(p) ? p.string() : std::string{};
    }
#ifdef _WIN32
    std::array<char, 32768> full{};
    DWORD n = SearchPathA(nullptr, executable.c_str(), nullptr,
                          static_cast<DWORD>(full.size()), full.data(), nullptr);
    if (n > 0 && n < full.size() && file_exists(full.data())) return std::string(full.data());
    n = SearchPathA(nullptr, executable.c_str(), ".exe",
                    static_cast<DWORD>(full.size()), full.data(), nullptr);
    if (n > 0 && n < full.size() && file_exists(full.data())) return std::string(full.data());
    return {};
#else
    const char* path_env = std::getenv("PATH");
    if (!path_env) return {};
    for (const auto& dir : util::split(path_env, ':')) {
        if (dir.empty()) continue;
        const fs::path candidate = fs::path(dir) / executable;
        if (file_exists(candidate)) return candidate.string();
    }
    return {};
#endif
}

bool llama_runtime_usable(const LlamaRuntimeStatus& status) {
    return (status.status == "resolved" || status.status == "ready")
        && !status.executable_path.empty();
}

std::string llama_runtime_target_mismatch_reason(
    const LlamaRuntimeStatus& status) {
    if (!llama_runtime_usable(status)) return {};

    const std::string actual_backend = util::to_lower(util::trim(status.accelerator));
    const std::string target_backend = util::to_lower(util::trim(
        status.target_accelerator.empty() ? status.accelerator
                                          : status.target_accelerator));
    if (!target_backend.empty() && actual_backend != target_backend) {
        return "Active llama.cpp backend is " +
            (actual_backend.empty() ? std::string{"unknown"} : actual_backend) +
            "; this node targets " + target_backend + ".";
    }

    const std::string target_variant = util::to_lower(util::trim(
        status.target_variant.empty() ? target_backend : status.target_variant));
    const bool concrete_target_variant = !target_variant.empty() &&
        target_variant != target_backend &&
        backend_for_variant(target_variant) == target_backend;
    const std::string actual_variant = util::to_lower(util::trim(
        status.variant.empty() ? actual_backend : status.variant));
    if (concrete_target_variant && actual_variant != target_variant) {
        return "Active llama.cpp variant is " +
            (actual_variant.empty() ? std::string{"unknown"} : actual_variant) +
            "; this node targets " + target_variant + ".";
    }

    const std::string target_method = normalize_llama_install_method(
        status.target_method.empty() ? "auto" : status.target_method);
    const std::string actual_method = normalize_llama_install_method(status.method);
    if (status.managed && target_method != "auto" &&
        actual_method != target_method) {
        return "Active llama.cpp install is " +
            (actual_method.empty() ? std::string{"unknown"} : actual_method) +
            "; this node targets a " + target_method + " build.";
    }

    if (target_backend == "cuda" && actual_method == "source" &&
        !status.target_cuda_architecture.empty() &&
        !status.cuda_architecture.empty() &&
        status.cuda_architecture != status.target_cuda_architecture) {
        return "Active llama.cpp CUDA architecture is " +
            status.cuda_architecture + "; this node targets " +
            status.target_cuda_architecture + ".";
    }
    return {};
}

std::string format_llama_troubleshooting_report(
    const LlamaTroubleshootingReport& report,
    const std::string& build_log_path) {
    std::ostringstream out;
    out << "llama.cpp troubleshooting report\n"
        << "Host: " << report.platform << "/" << report.architecture << "\n"
        << "Target: " << (report.target_backend.empty() ? "?" : report.target_backend)
        << "\n"
        << "Release: " << (report.release_tag.empty() ? "?" : report.release_tag)
        << "\n"
        << "Failure stage: " << report.failure_stage << "\n";
    if (!build_log_path.empty()) out << "Full build log: " << build_log_path << "\n";
    out << "\n" << report.failure_detail << "\n\n"
        << report.summary << "\n\nEnvironment checks:\n";
    for (const auto& check : report.checks) {
        const char* marker = check.status == "pass" ? "PASS"
            : (check.status == "fail" ? "FAIL"
                : (check.status == "warn" ? "WARN" : "INFO"));
        out << "[" << marker << "] " << check.label;
        if (!check.detected.empty()) out << ": " << check.detected;
        out << "\n";
        if (!check.required.empty()) out << "  Required: " << check.required << "\n";
        if (!check.remediation.empty()) out << "  Fix: " << check.remediation << "\n";
    }
    out << "\nRuntime variants (" << report.platform << "/"
        << report.architecture << "):\n";
    for (const auto& variant : report.variants) {
        const char* availability = variant.release_available ? "RELEASE"
            : (variant.source_supported ? "COMPILE" : "UNSUPPORTED");
        out << "[" << availability << "] " << variant.label;
        if (variant.recommended) out << " (recommended)";
        out << ": " << variant.reason;
        if (!variant.release_asset.empty()) out << " [" << variant.release_asset << "]";
        out << "\n";
    }
    return out.str();
}

LlamaCppProvisioner::LlamaCppProvisioner(LlamaProvisionConfig cfg,
                                         LlamaCommandRunner runner)
    : cfg_(normalized_config(std::move(cfg)))
    , runner_(std::move(runner))
{
    if (!runner_.run) {
        runner_.run = [](const std::vector<std::string>& argv,
                         const fs::path& cwd,
                         const StreamLineCallback& on_line,
                         const CancelCheckCallback& cancel_requested,
                         std::string* error) {
            return run_streamed_command(argv, cwd, on_line, cancel_requested, error);
        };
    }
    if (!runner_.capture_output) runner_.capture_output = default_capture_output;
    if (!runner_.capture_first_line) runner_.capture_first_line = default_capture_first_line;
    if (!runner_.resolve_executable) runner_.resolve_executable = resolve_llama_executable;
    if (!runner_.fetch_latest) {
        runner_.fetch_latest =
            [capture = runner_.capture_first_line](const LlamaProvisionConfig& c) {
                return default_fetch_latest(c, capture);
            };
    }
    if (!runner_.fetch_release_assets)
        runner_.fetch_release_assets = default_fetch_release_assets;
    status_ = make_base_status();
}

void LlamaCppProvisioner::set_log_sink(LogSink sink) { log_sink_ = std::move(sink); }
void LlamaCppProvisioner::set_progress_sink(ProgressSink sink) { progress_sink_ = std::move(sink); }
void LlamaCppProvisioner::set_cancel_check(CancelCheck check) { cancel_check_ = std::move(check); }

void LlamaCppProvisioner::emit_progress(const RuntimeInstallProgress& p) {
    if (progress_sink_) progress_sink_(p);
}

LlamaRuntimeStatus LlamaCppProvisioner::make_base_status() const {
    LlamaRuntimeStatus status;
    status.status = "disabled";
    status.platform = cfg_.platform;
    status.method = cfg_.install_method;
    status.source_repo = kLlamaCppRepoUrl;
    status.version = cfg_.version;
    status.accelerator = cfg_.accelerator;
    status.variant = cfg_.release_variant.empty() ? cfg_.accelerator
                                                  : cfg_.release_variant;
    status.target_method = cfg_.install_method;
    status.target_accelerator = cfg_.accelerator;
    status.target_variant = cfg_.release_variant.empty() ? cfg_.accelerator
                                                         : cfg_.release_variant;
    if (cfg_.accelerator == "cuda" && !cfg_.cuda_arch.empty())
        status.target_cuda_architecture = cuda_cmake_architectures(cfg_.cuda_arch);
    status.available_variants = llama_runtime_variants({}, cfg_);
    return status;
}

void LlamaCppProvisioner::set_status(LlamaRuntimeStatus& status) {
    if (status.target_method.empty()) status.target_method = cfg_.install_method;
    if (status.target_accelerator.empty())
        status.target_accelerator = cfg_.accelerator;
    if (status.target_variant.empty()) {
        status.target_variant = cfg_.release_variant.empty()
            ? cfg_.accelerator : cfg_.release_variant;
    }
    if (status.target_cuda_architecture.empty() &&
        status.target_accelerator == "cuda" && !cfg_.cuda_arch.empty()) {
        status.target_cuda_architecture = cuda_cmake_architectures(cfg_.cuda_arch);
    }
    status.target_mismatch_reason =
        llama_runtime_target_mismatch_reason(status);
    status.target_mismatch = !status.target_mismatch_reason.empty();
    std::lock_guard<std::mutex> lock(status_mutex_);
    status_ = status;
}

LlamaRuntimeStatus LlamaCppProvisioner::status() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_;
}

LlamaProvisionConfig LlamaCppProvisioner::config() const { return cfg_; }

std::string LlamaCppProvisioner::capture_version(const std::string& executable) const {
    return validated_llama_version(
        runner_.capture_first_line({executable, "--version"}, {}));
}

LlamaRuntimeStatus LlamaCppProvisioner::ensure_runtime() {
    LlamaRuntimeStatus status = make_base_status();

    const std::string resolved = runner_.resolve_executable(cfg_.requested_executable);
    if (!resolved.empty() && use_unmanaged_path_runtime(cfg_)) {
        status.status = "resolved";
        status.method = "path";
        status.managed = false;
        status.executable_path = resolved;
        const std::string version = capture_version(resolved);
        // cfg_.version describes a managed target (often "latest"), not an
        // arbitrary PATH executable. Unknown is more honest than echoing a
        // rejected CLI error or the desired managed version.
        status.version = version;
        set_status(status);
        return status;
    }
    if (!resolved.empty()) {
        MM_INFO("Ignoring generic PATH llama-server '{}' for requested {} backend; "
                "selecting an environment-matched managed runtime instead",
                resolved, cfg_.accelerator);
    }

    // Honor the last successful managed selection before scanning generic
    // candidates. This keeps a user-selected Vulkan/CPU release active across
    // restarts instead of mislabeling it as the auto-detected CUDA target.
    const auto active = read_active_runtime(cfg_);
    if (active) {
        const std::string version = capture_version(active->executable_path);
        auto restored = *active;
        restored.available_variants = status.available_variants;
        if (!version.empty()) restored.version = version;
        set_status(restored);
        return restored;
    }
    const bool ignored_active_marker =
        fs::exists(fs::path(cfg_.provision_dir) / "active-runtime.json");

    // An already-built managed runtime is usable regardless of auto-provision.
    for (const auto& cand : managed_executable_candidates(cfg_)) {
        // A malformed/stale marker must not cause a generic release artifact
        // to be rediscovered with the configured target's identity.
        if (ignored_active_marker) {
            const fs::path release_root =
                (fs::path(cfg_.provision_dir) / "release").lexically_normal();
            const std::string candidate_string = cand.lexically_normal().string();
            const std::string release_string = release_root.string();
            if (candidate_string.rfind(release_string, 0) == 0) continue;
        }
        if (!file_exists(cand)) continue;
        const std::string version = capture_version(cand.string());
        status.status = "ready";
        status.managed = true;
        status.executable_path = cand.string();
        status.version = version.empty() ? validated_llama_version(cfg_.version)
                                         : version;
        set_status(status);
        return status;
    }

    if (!cfg_.auto_provision) {
        status.status = "disabled";
        status.last_error =
            "llama-server not found and auto-provisioning is disabled: "
            + cfg_.requested_executable;
        set_status(status);
        return status;
    }

    return run_managed_install(/*upgrade=*/false);
}

LlamaRuntimeStatus LlamaCppProvisioner::run_managed_install(
    bool upgrade,
    const std::string& accelerator_override,
    bool force_source,
    bool bypass_environment_checks,
    bool release_only_override) {
    LlamaProvisionConfig install_cfg = cfg_;
    const LlamaRuntimeStatus prior = this->status();
    if (upgrade && !prior.latest_version.empty())
        install_cfg.version = prior.latest_version; // build/download the assessed release, not moving master
    if (upgrade && accelerator_override.empty() && !prior.accelerator.empty())
        install_cfg.accelerator = prior.accelerator;
    if (!accelerator_override.empty()) {
        install_cfg.release_variant = util::to_lower(util::trim(accelerator_override));
        install_cfg.accelerator = backend_for_variant(install_cfg.release_variant);
        if (release_only_override)
            install_cfg.install_method = "release"; // update alternatives represent published assets
    }
    if (force_source) install_cfg.install_method = "source";
    install_cfg.bypass_environment_checks = bypass_environment_checks;
    if (install_cfg.install_method == "auto" && install_cfg.version == "latest" &&
        runner_.fetch_latest) {
        const std::string release_tag = util::trim(runner_.fetch_latest(install_cfg));
        if (is_llama_release_tag(release_tag)) install_cfg.version = release_tag;
    }
    LlamaRuntimeStatus status = make_base_status();
    status.accelerator = install_cfg.accelerator;
    status.variant = install_cfg.release_variant.empty()
        ? install_cfg.accelerator : install_cfg.release_variant;
    status.version = install_cfg.version;
    if (!prior.available_variants.empty())
        status.available_variants = prior.available_variants;

    const fs::path root(install_cfg.provision_dir);
    std::error_code ec;
    fs::create_directories(root, ec);
    if (ec) {
        status.status = "failed";
        status.last_error = "failed to create llama.cpp provision directory: " + ec.message();
        status.troubleshooting = build_troubleshooting_report(status.last_error, install_cfg);
        set_status(status);
        return status;
    }

    std::ofstream build_log;
    const fs::path build_log_path = create_llama_build_log_path(install_cfg);
    if (!build_log_path.empty()) {
        build_log.open(build_log_path, std::ios::out | std::ios::trunc);
        if (build_log) {
            status.build_log_path = build_log_path.string();
            MM_INFO("llama.cpp build attempt log: {}", status.build_log_path);
        } else {
            MM_WARN("Could not open llama.cpp build attempt log: {}",
                    build_log_path.string());
        }
    }
    auto write_build_log = [&](const std::string& line) {
        if (!build_log) return;
        build_log << line << '\n';
        build_log.flush();
    };
    write_build_log("# Mantic-Mind llama.cpp managed runtime attempt");
    write_build_log("started_ms: " + std::to_string(util::now_ms()));
    write_build_log(std::string{"operation: "} + (upgrade ? "update" : "install"));
    write_build_log("platform: " + install_cfg.platform + "/" +
                    release_arch(install_cfg.arch));
    write_build_log("backend: " + install_cfg.accelerator);
    write_build_log("variant: " + (install_cfg.release_variant.empty()
        ? install_cfg.accelerator : install_cfg.release_variant));
    write_build_log("method: " + install_cfg.install_method);
    write_build_log("version: " + install_cfg.version);
    if (!install_cfg.cuda_arch.empty()) {
        write_build_log("cuda_architectures: " + install_cfg.cuda_arch);
        write_build_log("cuda_cmake_architectures: " +
                        cuda_cmake_architectures(install_cfg.cuda_arch));
    }
    write_build_log("provision_dir: " + install_cfg.provision_dir);
    write_build_log("");

    status.status = "provisioning";
    status.managed = true;
    set_status(status);

    auto finish_progress = [&]() {
        RuntimeInstallProgress done;   // active=false
        emit_progress(done);
    };
    auto canceled_status = [&]() {
        status.status = "failed";
        status.managed = true;
        status.last_error =
            std::string("llama.cpp ") + (upgrade ? "update" : "install") + " canceled";
        write_build_log("result: canceled");
        set_status(status);
        finish_progress();
        return status;
    };

    struct InstallAttempt {
        LlamaProvisionConfig config;
        std::string label;
    };
    std::vector<InstallAttempt> attempts;
    if (install_cfg.install_method == "auto") {
        // check_for_update() already inspected the release's asset matrix. Do
        // not repeat a known-failing release download before a warned/approved
        // source compilation.
        const bool known_compile = upgrade && accelerator_override.empty()
            && prior.update_action == "compile";
        if (!known_compile) {
            auto release = install_cfg;
            release.install_method = "release";
            attempts.push_back({std::move(release), "matched release"});
        }
        auto source = install_cfg;
        source.install_method = "source";
        attempts.push_back({std::move(source),
                            known_compile ? "approved source build" : "source fallback"});
    } else {
        attempts.push_back({install_cfg, install_cfg.install_method});
    }

    std::vector<std::string> attempt_errors;
    for (size_t attempt_index = 0; attempt_index < attempts.size(); ++attempt_index) {
        const auto& attempt = attempts[attempt_index];
        status.method = attempt.config.install_method;
        status.last_error.clear();
        set_status(status);

        const auto plan = build_llama_install_plan(attempt.config, upgrade);
        const int total = static_cast<int>(plan.size());
        MM_INFO("llama.cpp {} {} plan: {} step(s)",
                upgrade ? "update" : "install", attempt.label, total);
        write_build_log("## attempt " + std::to_string(attempt_index + 1) + "/" +
                        std::to_string(attempts.size()) + ": " + attempt.label);
        write_build_log("planned_steps: " + std::to_string(total));

        std::string attempt_error;
        if (plan.empty()) {
            attempt_error = "install plan is empty";
            write_build_log("error: " + attempt_error);
        }
        for (int i = 0; i < total && attempt_error.empty(); ++i) {
            const LlamaInstallStep& step = plan[static_cast<size_t>(i)];
            if (cancel_check_ && cancel_check_()) return canceled_status();

            RuntimeInstallProgress prog;
            prog.active = true;
            prog.step = i + 1;
            prog.total_steps = total;
            prog.fraction = -1.0;
            prog.stage = step.label;
            emit_progress(prog);
            if (log_sink_) log_sink_("[llama-install] " + step.label, false);
            const fs::path cwd = step.cwd.empty() ? root : step.cwd;
            write_build_log("");
            write_build_log("### step " + std::to_string(i + 1) + "/" +
                            std::to_string(total) + ": " + step.label);
            write_build_log("cwd: " + cwd.string());
            write_build_log("command: " + join_shell_command(step.argv));

            // Preserve enough context for CMake compiler-identification and
            // nested call stacks. Twelve lines routinely discarded the actual
            // CUDA/ptxas error before the troubleshooting wizard opened.
            std::deque<std::string> output_tail;
            auto on_line = [&](const std::string& line, bool is_stderr) {
                if (log_sink_) log_sink_(line, is_stderr);
                write_build_log(std::string{is_stderr ? "[stderr] " : "[stdout] "} +
                                line);
                const double f = parse_install_fraction(line);
                if (f >= 0.0) prog.fraction = f;
                prog.last_line = line;
                const std::string trimmed = util::trim(line);
                if (!trimmed.empty()) {
                    if (output_tail.size() >= 80) output_tail.pop_front();
                    output_tail.push_back(trimmed);
                }
                emit_progress(prog);
            };

            std::string step_err;
            const int rc = runner_.run(step.argv, cwd, on_line, cancel_check_, &step_err);
            write_build_log("exit_code: " + std::to_string(rc));
            if (!step_err.empty()) write_build_log("runner_error: " + step_err);
            if (cancel_check_ && cancel_check_()) return canceled_status();
            if (rc != 0 && !step.allow_failure) {
                std::string detail = step_err;
                if (detail.empty() && !output_tail.empty()) {
                    std::vector<std::string> lines(output_tail.begin(), output_tail.end());
                    detail = util::join(lines, "\n");
                }
                attempt_error = "step '" + step.label + "' failed"
                    + (detail.empty() ? (" (exit " + std::to_string(rc) + ")")
                                      : (": " + detail));
                write_build_log("step_failure: " + attempt_error);
            }
        }

        std::string found;
        if (attempt_error.empty()) {
            for (const auto& cand : managed_executable_candidates(attempt.config)) {
                if (file_exists(cand)) { found = cand.string(); break; }
            }
            if (found.empty()) {
                attempt_error = "install completed but no llama-server executable was found under "
                    + root.string();
                write_build_log("attempt_failure: " + attempt_error);
            }
        }

        if (!found.empty()) {
            const std::string version = capture_version(found);
            status.status = "ready";
            status.managed = true;
            status.method = attempt.config.install_method;
            status.cuda_architecture =
                attempt.config.accelerator == "cuda" && status.method == "source"
                ? cuda_cmake_architectures(attempt.config.cuda_arch)
                : std::string{};
            status.executable_path = found;
            status.version = version.empty()
                ? validated_llama_version(attempt.config.version) : version;
            status.latest_version.clear();
            status.update_available = false;
            status.update_action.clear();
            status.update_release_available = false;
            status.update_release_alternatives.clear();
            status.update_warning.clear();
            status.troubleshooting = {};
            status.last_error.clear();
            write_build_log("");
            write_build_log("result: success");
            write_build_log("method: " + status.method);
            write_build_log("executable: " + status.executable_path);
            write_build_log("reported_version: " + status.version);
            write_active_runtime(attempt.config, status);
            set_status(status);
            finish_progress();
            MM_INFO("llama.cpp runtime {} from {}: {}",
                    upgrade ? "updated" : "ready", attempt.label,
                    status.executable_path);
            return status;
        }

        attempt_errors.push_back(attempt.label + ": " + attempt_error);
        write_build_log("attempt_result: failed");
        write_build_log("attempt_error: " + attempt_error);
        if (attempt_index + 1 < attempts.size()) {
            const std::string message =
                "[llama-install] " + attempt.label + " unavailable (" + attempt_error
                + "); falling back to source build";
            MM_WARN("{}", message);
            if (log_sink_) log_sink_(message, true);
        }
    }

    status.status = "failed";
    status.last_error = "llama.cpp install failed";
    for (const auto& error : attempt_errors) status.last_error += "; " + error;
    status.troubleshooting = build_troubleshooting_report(status.last_error, install_cfg);
    write_build_log("");
    write_build_log("result: failed");
    write_build_log(format_llama_troubleshooting_report(
        status.troubleshooting, status.build_log_path));
    set_status(status);
    finish_progress();
    return status;
}

LlamaTroubleshootingReport LlamaCppProvisioner::build_troubleshooting_report(
    const std::string& failure_detail,
    const LlamaProvisionConfig& raw_install_cfg) const {
    const LlamaProvisionConfig install_cfg = normalized_config(raw_install_cfg);
    LlamaTroubleshootingReport report;
    report.required = true;
    report.failure_detail = failure_detail;
    report.platform = install_cfg.platform;
    report.architecture = release_arch(install_cfg.arch);
    report.target_backend = install_cfg.accelerator;
    report.release_tag = install_cfg.version;

    std::smatch stage_match;
    if (std::regex_search(failure_detail, stage_match,
                          std::regex("step '([^']+)' failed")))
        report.failure_stage = stage_match[1].str();
    if (report.failure_stage.empty()) report.failure_stage = "Managed provisioning";

    if (report.release_tag == "latest" && runner_.fetch_latest) {
        const std::string latest = util::trim(runner_.fetch_latest(install_cfg));
        if (!latest.empty()) report.release_tag = latest;
    }
    const auto assets = runner_.fetch_release_assets && !report.release_tag.empty()
        ? runner_.fetch_release_assets(install_cfg, report.release_tag)
        : std::vector<std::string>{};
    report.variants = llama_runtime_variants(assets, install_cfg);

    auto add = [&](std::string id, std::string label, std::string status,
                   std::string detected, std::string required,
                   std::string remediation, bool blocking) {
        report.checks.push_back({std::move(id), std::move(label), std::move(status),
                                 std::move(detected), std::move(required),
                                 std::move(remediation), blocking});
    };
    auto capture = [&](const std::vector<std::string>& argv) {
        return runner_.capture_output
            ? util::trim(runner_.capture_output(argv, install_cfg.provision_dir))
            : std::string{};
    };
    auto locate = [&](const std::string& tool) {
        if (install_cfg.platform == "windows") {
            const std::string script =
                "$c=Get-Command '" + tool +
                "' -ErrorAction SilentlyContinue; if($c){$c.Source}";
            return capture({"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                            "-Command", script});
        }
        return capture({"sh", "-c", "command -v \"$1\" 2>/dev/null || true",
                        "mantic-mind-diagnostic", tool});
    };
    auto tool_check = [&](const std::string& id, const std::string& label,
                          const std::string& tool, const std::string& required,
                          const std::string& remediation, bool blocking) {
        const std::string found = locate(tool);
        add(id, label, found.empty() ? (blocking ? "fail" : "warn") : "pass",
            found.empty() ? "not found on PATH" : found, required, remediation,
            blocking && found.empty());
        return !found.empty();
    };

    add("host", "Host platform", "pass",
        install_cfg.platform + "/" + report.architecture,
        "Windows, Linux, or macOS on a supported architecture", {}, false);
    if (install_cfg.platform == "linux") {
        const std::string wsl = capture(
            {"sh", "-c", "grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null && echo WSL || true"});
        add("wsl", "Linux environment", "info", wsl.empty() ? "native Linux" : "WSL",
            "GPU toolkits must be installed inside this Linux environment",
            wsl.empty() ? std::string{}
                        : "Install Linux build packages and GPU toolkits inside the WSL distribution, not only on Windows.",
            false);
    }

    std::error_code space_ec;
    const auto space = fs::space(install_cfg.provision_dir, space_ec);
    if (space_ec) {
        add("disk", "Build disk space", "warn", "could not measure: " + space_ec.message(),
            "at least 4 GiB free", "Free space in the runtime provision directory.", false);
    } else {
        const uint64_t gib = space.available / (1024ULL * 1024ULL * 1024ULL);
        add("disk", "Build disk space", gib >= 4 ? "pass" : "warn",
            std::to_string(gib) + " GiB free", "at least 4 GiB free",
            gib >= 4 ? std::string{} : "Free at least 4 GiB before compiling llama.cpp.",
            false);
    }
    const std::string memory = install_cfg.platform == "windows"
        ? capture({"powershell", "-NoProfile", "-Command",
                   "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2).ToString()+' GiB free'"})
        : (install_cfg.platform == "macos"
            ? capture({"sh", "-c", "vm_stat 2>/dev/null | head -n 2 | tail -n 1 || true"})
            : capture({"sh", "-c", "awk '/MemAvailable/{printf \"%.2f GiB free\",$2/1048576}' /proc/meminfo 2>/dev/null || true"}));
    add("memory", "Available memory", memory.empty() ? "warn" : "info",
        memory.empty() ? "could not measure" : memory,
        "enough RAM for the selected parallel build",
        "Reduce llama_build_jobs if the compiler is killed by the OOM manager.", false);

    const std::string package_hint = install_cfg.platform == "windows"
        ? "Install the tool and reopen the node so its executable is on PATH."
        : "Install the package with the distribution package manager and reopen the node.";
    tool_check("git", "Git", "git", "git client", package_hint, true);
    const bool cmake_found = tool_check(
        "cmake", "CMake", "cmake", "CMake 3.14 or newer", package_hint, true);
    std::string cmake_version;
    if (cmake_found) {
        const std::string cmake_output = capture({"cmake", "--version"});
        for (const auto& line : util::split(cmake_output, '\n')) {
            const std::string trimmed = util::trim(line);
            if (!trimmed.empty()) {
                cmake_version = trimmed;
                break;
            }
        }
        add("cmake-version", "CMake version",
            cmake_version.empty() ? "warn" : "info",
            cmake_version.empty() ? "version could not be queried" : cmake_version,
            "a CMake release compatible with the selected CUDA Toolkit",
            cmake_version.empty()
                ? "Run cmake --version in the node environment. CUDA 13 support "
                  "requires either Mantic-Mind's architecture pin or a current CMake."
                : std::string{},
            false);
    }
    if (install_cfg.platform == "windows") {
        tool_check("compiler", "C/C++ compiler", "cl", "Visual Studio C++ build tools",
                   "Install Visual Studio Build Tools with Desktop development with C++; CMake may also discover an installed compiler outside PATH.",
                   false);
    } else {
        tool_check("compiler", "C++ compiler", "c++", "a C++17 compiler (GCC or Clang)",
                   "Install build-essential/g++ or clang with the distribution package manager.", true);
    }

    const std::string backend = install_cfg.accelerator;
    if (backend == "cuda") {
        const bool driver = tool_check(
            "nvidia-driver", "NVIDIA driver visibility", "nvidia-smi",
            "NVIDIA driver with GPU visibility",
            install_cfg.platform == "linux"
                ? "Enable NVIDIA GPU passthrough for this Linux/WSL environment."
                : "Install or repair the NVIDIA display driver.", false);
        const std::string nvcc_path = install_cfg.platform == "windows"
            ? capture({"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                       "$c=if($env:CUDACXX){Get-Command $env:CUDACXX -ErrorAction SilentlyContinue}else{Get-Command nvcc -ErrorAction SilentlyContinue}; if($c){$c.Source}"})
            : capture({"sh", "-c",
                       "if [ -n \"${CUDACXX:-}\" ]; then command -v \"$CUDACXX\" 2>/dev/null || { [ -x \"$CUDACXX\" ] && printf '%s' \"$CUDACXX\"; }; else command -v nvcc 2>/dev/null || true; fi"});
        const bool toolkit = !nvcc_path.empty();
        std::string nvcc_version;
        if (toolkit) {
            const std::string version_output = capture({nvcc_path, "--version"});
            for (const auto& line : util::split(version_output, '\n')) {
                const std::string trimmed = util::trim(line);
                if (!trimmed.empty()) nvcc_version = trimmed;
            }
        }
        add("cuda-toolkit", "CUDA compiler toolkit", toolkit ? "pass" : "fail",
            toolkit ? nvcc_path + (nvcc_version.empty() ? std::string{} : " | " + nvcc_version)
                    : "not found via CUDACXX or PATH",
            install_cfg.release_variant == "cuda-13" ? "CUDA 13 toolkit" : "CUDA toolkit",
            install_cfg.platform == "linux"
                ? "Install the CUDA Toolkit inside this Linux/WSL distribution and ensure CUDACXX or PATH selects its nvcc."
                : "Install the NVIDIA CUDA Toolkit and ensure CUDACXX or PATH selects its nvcc.",
            !toolkit);
        const auto requested_arches = cuda_arch_targets(install_cfg.cuda_arch);
        if (cuda_removed_default_arch_failure(failure_detail)) {
            const std::string target = requested_arches.empty()
                ? std::string{"the detected GPU architecture"}
                : "sm_" + cuda_feature_target(requested_arches.front());
            add("cuda-cmake-compiler-id", "CUDA CMake compiler identification",
                "fail",
                "CMake invoked ptxas for removed legacy target sm_52 instead of " +
                    target,
                "compiler identification targeting " + target,
                "Retry with Mantic-Mind's architecture-pinned configure command. "
                "If it still fails, ensure nvcc and ptxas resolve from the same "
                "CUDA Toolkit and upgrade CMake to a release with CUDA 13 support.",
                true);
        }
        if (cuda_blackwell_baseline_target_failure(failure_detail)) {
            const std::string target = requested_arches.empty()
                ? std::string{"sm_12Xa"}
                : "sm_" + cuda_feature_target(requested_arches.front());
            add("cuda-blackwell-feature-target",
                "Blackwell architecture-specific CUDA target", "fail",
                "llama.cpp's MXFP4/block-scaled MMA kernel was assembled for a "
                "baseline sm_12X target",
                target + " for Blackwell FP4 tensor-core instructions",
                "Retry with the architecture-specific CMake target. Remove any "
                "CMAKE_CUDA_FLAGS=-arch=sm_120 override; use "
                "CMAKE_CUDA_ARCHITECTURES=120a-real for an RTX 50-series GPU.",
                true);
        }
        if (toolkit) {
            const std::string ptxas_path = locate("ptxas");
            if (!ptxas_path.empty()) {
                const std::string ptxas_output = capture({ptxas_path, "--version"});
                std::string ptxas_version;
                for (const auto& line : util::split(ptxas_output, '\n')) {
                    const std::string trimmed = util::trim(line);
                    if (!trimmed.empty()) ptxas_version = trimmed;
                }
                const bool same_bin = fs::path(ptxas_path).parent_path() ==
                    fs::path(nvcc_path).parent_path();
                add("cuda-assembler", "CUDA assembler", same_bin ? "pass" : "warn",
                    ptxas_path + (ptxas_version.empty() ? std::string{}
                                                       : " | " + ptxas_version),
                    "ptxas from the same CUDA Toolkit bin directory as nvcc",
                    same_bin ? std::string{}
                        : "Resolve PATH/CUDACXX so nvcc and ptxas come from one "
                          "CUDA Toolkit; mixed toolkit components can advertise "
                          "sm_120 but fail during assembly.",
                    false);
            }
        }
        if (toolkit && !requested_arches.empty()) {
            const std::string supported = capture({nvcc_path, "--list-gpu-arch"});
            std::vector<std::string> unsupported;
            for (const auto& arch : requested_arches) {
                if (!nvcc_supports_arch(supported, cuda_feature_target(arch)))
                    unsupported.push_back(arch);
            }
            std::vector<std::string> unsupported_labels;
            for (const auto& arch : unsupported)
                unsupported_labels.push_back("compute_" + cuda_feature_target(arch));
            const bool compatible = unsupported.empty();
            add("cuda-architecture", "CUDA compiler architecture support",
                compatible ? "pass" : "fail",
                compatible
                    ? nvcc_path + " meets " + cuda_arch_requirement(requested_arches)
                    : nvcc_path + " does not list " + util::join(unsupported_labels, ", ")
                      + (nvcc_version.empty() ? std::string{} : " | " + nvcc_version),
                cuda_arch_requirement(requested_arches),
                compatible ? std::string{}
                    : "Install a compatible CUDA Toolkit inside the active environment and ensure CUDACXX or PATH selects it. For sm_120a use CUDA Toolkit 12.8 or newer; nvidia-smi's CUDA Version reports driver compatibility, not the selected NVCC.",
                !compatible);
        }
        if (driver && !toolkit) {
            add("cuda-driver-toolkit-mismatch", "CUDA driver/toolkit mismatch", "fail",
                "GPU driver is visible but nvcc is missing",
                "both nvidia-smi and nvcc for source compilation",
                "The display driver alone cannot compile llama.cpp; install the CUDA Toolkit in the active OS environment.",
                true);
        }
    } else if (backend == "rocm" || backend == "hip") {
        tool_check("hipcc", "HIP compiler", "hipcc", "ROCm/HIP SDK",
                   "Install the ROCm/HIP development SDK and ensure hipcc is on PATH.", true);
        tool_check("rocminfo", "ROCm device visibility", "rocminfo", "ROCm device runtime",
                   "Install the ROCm runtime and grant this user access to the GPU device.", false);
        if (install_cfg.platform == "linux") {
            const std::string kfd = capture({"sh", "-c", "test -e /dev/kfd && echo /dev/kfd || true"});
            add("kfd", "ROCm kernel device", kfd.empty() ? "warn" : "pass",
                kfd.empty() ? "/dev/kfd is missing" : kfd, "/dev/kfd device access",
                "Load the amdgpu driver and grant the user video/render group access.", false);
        }
    } else if (backend == "vulkan") {
        tool_check("glslc", "Vulkan shader compiler", "glslc", "Vulkan SDK shader tools",
                   "Install the Vulkan SDK (including glslc) and make it available on PATH.", true);
        tool_check("vulkaninfo", "Vulkan device visibility", "vulkaninfo", "Vulkan loader and driver",
                   "Install a Vulkan-capable GPU driver and Vulkan loader tools.", false);
    } else if (backend == "openvino") {
        const std::string ov = install_cfg.platform == "windows"
            ? capture({"powershell", "-NoProfile", "-Command",
                       "if($env:OpenVINO_DIR){$env:OpenVINO_DIR}elseif($env:INTEL_OPENVINO_DIR){$env:INTEL_OPENVINO_DIR}"})
            : capture({"sh", "-c", "printf '%s' \"${OpenVINO_DIR:-${INTEL_OPENVINO_DIR:-}}\""});
        add("openvino", "OpenVINO toolkit", ov.empty() ? "fail" : "pass",
            ov.empty() ? "environment is not initialized" : ov,
            "OpenVINO development toolkit",
            "Install OpenVINO and run its setupvars script before starting the node.", ov.empty());
    } else if (backend == "sycl-fp32" || backend == "sycl-fp16") {
        const std::string compiler = install_cfg.platform == "windows" ? "icx" : "icpx";
        tool_check("sycl-compiler", "oneAPI DPC++ compiler", compiler,
                   "Intel oneAPI DPC++ compiler",
                   "Install Intel oneAPI Base Toolkit and initialize setvars before starting the node.", true);
        tool_check("sycl-ls", "SYCL device visibility", "sycl-ls", "a SYCL-capable device/runtime",
                   "Install the matching Level Zero/OpenCL device runtime.", false);
    } else if (backend == "metal") {
        tool_check("xcrun", "Apple developer toolchain", "xcrun",
                   "Xcode command-line tools and Metal framework",
                   "Run xcode-select --install and accept the Xcode license.", true);
    }

    const int release_count = static_cast<int>(std::count_if(
        report.variants.begin(), report.variants.end(),
        [](const auto& v) { return v.release_available; }));
    add("release-assets", "Compatible server releases",
        assets.empty() ? "warn" : (release_count > 0 ? "pass" : "warn"),
        assets.empty() ? "release assets could not be retrieved"
                       : std::to_string(release_count) + " compatible option(s)",
        "a complete llama-server bundle for a one-click fallback",
        assets.empty() ? "Check network access to api.github.com or retry diagnostics."
                       : (release_count > 0 ? std::string{}
                                            : "Use source compilation after fixing the failed checks."),
        false);

    report.can_override_checks = std::any_of(
        report.variants.begin(), report.variants.end(), [&](const auto& v) {
            return v.source_supported && v.backend == backend;
        });
    const int blocking = static_cast<int>(std::count_if(
        report.checks.begin(), report.checks.end(),
        [](const auto& c) { return c.blocking && c.status == "fail"; }));
    report.summary = blocking > 0
        ? std::to_string(blocking) + " blocking prerequisite issue(s) detected; "
          + std::to_string(release_count) + " compatible release fallback(s) available."
        : "No blocking prerequisite was detected by the static probes; inspect the failed build stage and retry with full logs.";

    const std::string fingerprint_input = report.platform + "|" + report.architecture + "|"
        + report.target_backend + "|" + report.failure_stage + "|" + failure_detail + "|"
        + std::to_string(util::now_ms());
    std::ostringstream fingerprint;
    fingerprint << std::hex << std::hash<std::string>{}(fingerprint_input);
    report.fingerprint = fingerprint.str();
    return report;
}

LlamaRuntimeStatus LlamaCppProvisioner::diagnose_environment() {
    LlamaRuntimeStatus current = status();
    LlamaProvisionConfig install_cfg = cfg_;
    if (!current.target_accelerator.empty())
        install_cfg.accelerator = current.target_accelerator;
    else if (!current.accelerator.empty())
        install_cfg.accelerator = current.accelerator;
    if (!current.latest_version.empty()) install_cfg.version = current.latest_version;
    const std::string failure = current.last_error.empty()
        ? std::string{"Manual llama.cpp environment diagnostic"}
        : current.last_error;
    current.troubleshooting = build_troubleshooting_report(failure, install_cfg);
    set_status(current);
    return current;
}

LlamaRuntimeStatus LlamaCppProvisioner::recover_runtime(
    const std::string& raw_action,
    const std::string& raw_variant) {
    const std::string action = util::to_lower(util::trim(raw_action));
    const std::string variant = util::to_lower(util::trim(raw_variant));
    if (action == "retry")
        return run_managed_install(/*upgrade=*/false);
    if (action == "target")
        return run_managed_install(/*upgrade=*/false);
    if (action == "compile-anyway")
        return run_managed_install(/*upgrade=*/false, {}, /*force_source=*/true,
                                   /*bypass_environment_checks=*/true);
    if (action == "release") {
        LlamaRuntimeStatus current = status();
        if (!current.troubleshooting.required) current = diagnose_environment();
        const auto found = std::find_if(
            current.troubleshooting.variants.begin(),
            current.troubleshooting.variants.end(),
            [&](const auto& item) { return item.id == variant; });
        if (found == current.troubleshooting.variants.end() ||
            !found->release_available) {
            current.last_error = "no assessed complete llama.cpp release for variant: "
                + (variant.empty() ? std::string{"(none)"} : variant);
            set_status(current);
            return current;
        }
        return run_managed_install(/*upgrade=*/false, variant);
    }
    LlamaRuntimeStatus current = status();
    current.last_error = "unsupported llama.cpp recovery action: " + raw_action;
    set_status(current);
    return current;
}

LlamaRuntimeStatus LlamaCppProvisioner::check_for_update() {
    LlamaRuntimeStatus status = this->status();
    const std::string latest =
        runner_.fetch_latest ? util::trim(runner_.fetch_latest(cfg_)) : std::string{};
    status.latest_version = latest;
    const std::string installed = util::trim(status.version);
    status.update_action.clear();
    status.update_release_available = false;
    status.update_release_alternatives.clear();
    status.update_warning.clear();
    const auto assets = runner_.fetch_release_assets && !latest.empty()
        ? runner_.fetch_release_assets(cfg_, latest)
        : std::vector<std::string>{};
    status.available_variants = llama_runtime_variants(assets, cfg_);
    // `llama-server --version` commonly returns a sentence containing the tag,
    // so treat a contained tag as current instead of comparing whole strings.
    status.update_available = !latest.empty() && !installed.empty()
        && !installed_version_matches_tag(installed, latest);
    if (status.update_available) {
        MM_INFO("llama.cpp update available: {} -> {}", installed, latest);
        const auto available = llama_release_accelerators(assets, cfg_);
        const std::string target = status.accelerator.empty()
            ? cfg_.accelerator
            : util::to_lower(util::trim(status.accelerator));
        status.update_release_available =
            std::find(available.begin(), available.end(), target) != available.end();
        for (const auto& accelerator : available)
            if (accelerator != target)
                status.update_release_alternatives.push_back(accelerator);

        if (assets.empty()) {
            if (cfg_.install_method == "source") status.update_action = "compile";
            status.update_warning =
                "Release assets could not be assessed. Approval will retry the release lookup"
                " and may compile llama-server from source.";
        } else if (cfg_.install_method == "source") {
            status.update_action = "compile";
        } else if (status.update_release_available) {
            status.update_action = "release";
        } else if (cfg_.install_method == "release") {
            status.update_action = "unavailable";
        } else {
            status.update_action = "compile";
        }

        if (status.update_action == "compile") {
            status.update_warning =
                "No official llama.cpp " + status.platform + "/" + release_arch(cfg_.arch)
                + "/" + target + " asset is available for " + latest
                + ". Approving this update will compile llama-server from source on this node.";
        } else if (status.update_action == "unavailable") {
            status.update_warning =
                "No official llama.cpp " + status.platform + "/" + release_arch(cfg_.arch)
                + "/" + target + " asset is available for " + latest
                + ", and llama_install_method=release forbids a source build.";
        }
        if (!status.update_release_alternatives.empty() &&
            status.update_action != "release") {
            status.update_warning += (status.update_warning.empty() ? std::string{} : std::string{" "})
                + "Official release alternatives: "
                + util::join(status.update_release_alternatives, ", ") + ".";
        }
        if (!status.update_warning.empty()) MM_WARN("{}", status.update_warning);
    }
    set_status(status);
    return status;
}

LlamaRuntimeStatus LlamaCppProvisioner::update_runtime(
    const std::string& accelerator_override) {
    const std::string resolved = runner_.resolve_executable(cfg_.requested_executable);
    if (!resolved.empty() && use_unmanaged_path_runtime(cfg_)) {
        LlamaRuntimeStatus status = this->status();
        status.last_error =
            "llama-server is resolved from PATH (" + resolved +
            "); update it yourself. Point llama_server_path at a managed build to "
            "let the node update it.";
        set_status(status);
        return status;
    }
    LlamaRuntimeStatus current = this->status();
    if (accelerator_override.empty() && current.update_action == "unavailable") {
        current.last_error = current.update_warning;
        set_status(current);
        return current;
    }
    const std::string requested = util::to_lower(util::trim(accelerator_override));
    if (!requested.empty()) {
        const std::set<std::string> allowed{
            "cuda", "rocm", "vulkan", "openvino", "metal", "cpu"};
        if (!allowed.count(requested)) {
            current.last_error = "unsupported llama.cpp update accelerator: " + requested;
            set_status(current);
            return current;
        }
        if (current.update_available &&
            std::find(current.update_release_alternatives.begin(),
                      current.update_release_alternatives.end(), requested)
                == current.update_release_alternatives.end() &&
            !(requested == current.accelerator && current.update_release_available)) {
            current.last_error = "no assessed official llama.cpp release for accelerator: "
                + requested;
            set_status(current);
            return current;
        }
    }
    return run_managed_install(/*upgrade=*/true, requested);
}

LlamaRuntimeStatus LlamaCppProvisioner::switch_runtime(
    const std::string& raw_variant) {
    const std::string variant = util::to_lower(util::trim(raw_variant));
    LlamaRuntimeStatus current = status();
    auto variants = current.available_variants;
    if (variants.empty()) variants = llama_runtime_variants({}, cfg_);
    const auto selected = std::find_if(
        variants.begin(), variants.end(),
        [&](const auto& candidate) { return candidate.id == variant; });
    if (selected == variants.end() || !selected->platform_supported ||
        (!selected->release_available && !selected->source_supported)) {
        current.last_error = "unsupported llama.cpp engine variant for "
            + cfg_.platform + "/" + release_arch(cfg_.arch) + ": "
            + (variant.empty() ? std::string{"(none)"} : variant);
        set_status(current);
        return current;
    }

    // A backend switch is not contingent on an update. Respect the configured
    // install method: auto prefers a matching release and falls back to source,
    // release remains release-only, and source compiles directly.
    return run_managed_install(/*upgrade=*/true, variant,
                               /*force_source=*/false,
                               /*bypass_environment_checks=*/false,
                               /*release_only_override=*/false);
}

} // namespace mm
