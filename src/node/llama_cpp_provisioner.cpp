#include "node/llama_cpp_provisioner.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"
#include "node/llama_runtime.hpp"
#include "node/vllm_provisioner.hpp"  // parse_vllm_install_fraction (shared line parser)
#include "node/vllm_runtime.hpp"      // current_vllm_platform / current_vllm_arch

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>

#ifdef _WIN32
#  include <Windows.h>
#endif

namespace fs = std::filesystem;

namespace mm {
namespace {

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

bool file_exists(const fs::path& p) {
    std::error_code ec;
    return fs::exists(p, ec) && fs::is_regular_file(p, ec);
}

std::string exe_suffix(const std::string& platform) {
    return platform == "windows" ? ".exe" : std::string{};
}

LlamaProvisionConfig normalized_config(LlamaProvisionConfig cfg) {
    cfg.requested_executable = strip_wrapping_quotes(cfg.requested_executable);
    if (cfg.requested_executable.empty()) cfg.requested_executable = "llama-server";
    cfg.install_method = normalize_llama_install_method(cfg.install_method);
    cfg.version = util::trim(cfg.version);
    if (cfg.version.empty()) cfg.version = "latest";
    if (cfg.platform.empty()) cfg.platform = current_vllm_platform();
    if (cfg.arch.empty()) cfg.arch = current_vllm_arch();
    cfg.accelerator = util::to_lower(util::trim(cfg.accelerator));
    if (cfg.accelerator.empty()) {
        // Without a live GPU probe, infer only what platform/arch make certain;
        // callers pass cuda/rocm after detection. Default to cpu otherwise.
        cfg.accelerator = detect_llama_accelerator(cfg.platform, cfg.arch,
                                                   /*has_cuda=*/false,
                                                   /*has_rocm=*/false);
    }
    cfg.cuda_arch = util::trim(cfg.cuda_arch);
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
        if (!cfg.cuda_arch.empty())
            flags.push_back("-DCMAKE_CUDA_ARCHITECTURES=" + cfg.cuda_arch);
    } else if (cfg.accelerator == "rocm") {
        flags.push_back("-DGGML_HIP=ON");
    } else if (cfg.accelerator == "vulkan") {
        flags.push_back("-DGGML_VULKAN=ON");
    } else if (cfg.accelerator == "metal") {
        flags.push_back("-DGGML_METAL=ON");
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
    if (cfg.install_method == "release") {
        return { root / "release" / exe };
    }
    const fs::path build = root / "llama.cpp-src" / "build" / "bin";
    return {
        build / "Release" / exe,   // MSVC multi-config
        build / exe,               // Ninja / Unix Makefiles
    };
}

std::string join_shell_command(const std::vector<std::string>& args);

std::string default_capture_first_line(const std::vector<std::string>& args,
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
    // the captured stream instead of discarding it (unlike the vLLM capture).
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
    if (cfg.platform == "windows") {
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

// Build a plan that downloads a prebuilt release archive and normalizes the
// extracted binary to <root>/release/llama-server[.exe]. Best-effort: asset
// naming varies, so the source path stays the reliable default.
std::vector<LlamaInstallStep> build_release_plan(const LlamaProvisionConfig& cfg) {
    const fs::path root(cfg.provision_dir);
    const fs::path rel = root / "release";
    const std::string tag_filter =
        cfg.version == "latest" ? std::string("releases/latest")
                                : "releases/tags/" + cfg.version;
    const std::string api =
        "https://api.github.com/repos/ggml-org/llama.cpp/" + tag_filter;

    std::vector<LlamaInstallStep> plan;
#ifdef _WIN32
    // Pick an asset whose name matches the accelerator keyword; fall back to any
    // Windows zip. Extract and hoist llama-server.exe to a deterministic path.
    std::ostringstream ps;
    ps << "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; ";
    ps << "New-Item -ItemType Directory -Force -Path '" << rel.string() << "' | Out-Null; ";
    ps << "$r = Invoke-RestMethod -Uri '" << api << "' -Headers @{ 'User-Agent'='mantic-mind' }; ";
    ps << "$kw = '" << cfg.accelerator << "'; ";
    ps << "$a = $r.assets | Where-Object { $_.name -like '*win*' -and $_.name -like ('*' + $kw + '*') } | Select-Object -First 1; ";
    ps << "if ($null -eq $a) { $a = $r.assets | Where-Object { $_.name -like '*win*' -and $_.name -like '*.zip' } | Select-Object -First 1 }; ";
    ps << "if ($null -eq $a) { throw 'No matching llama.cpp Windows release asset found.' }; ";
    ps << "$zip = Join-Path '" << rel.string() << "' $a.name; ";
    ps << "Write-Host ('Downloading ' + $a.name); ";
    ps << "Invoke-WebRequest -Uri $a.browser_download_url -OutFile $zip; ";
    ps << "Expand-Archive -Path $zip -DestinationPath '" << rel.string() << "' -Force; ";
    ps << "$exe = Get-ChildItem -Path '" << rel.string() << "' -Recurse -Filter 'llama-server.exe' | Select-Object -First 1; ";
    ps << "if ($null -eq $exe) { throw 'llama-server.exe not found in the extracted archive.' }; ";
    ps << "Copy-Item -Path $exe.FullName -Destination '" << (rel / "llama-server.exe").string() << "' -Force";
    plan.push_back({"Downloading llama.cpp release",
                    {"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps.str()},
                    root, false});
#else
    // POSIX: resolve the asset URL via python3, download with curl, unzip, and
    // hoist the binary. Requires curl + unzip on PATH.
    const std::string py =
        "import json, urllib.request\n"
        "req = urllib.request.Request(\"" + api + "\", headers={\"User-Agent\": \"mantic-mind\"})\n"
        "d = json.load(urllib.request.urlopen(req, timeout=20))\n"
        "kw = \"" + cfg.accelerator + "\"\n"
        "assets = d.get(\"assets\", [])\n"
        "cand = [a for a in assets if kw in a[\"name\"] and a[\"name\"].endswith(\".zip\")]\n"
        "cand = cand or [a for a in assets if a[\"name\"].endswith(\".zip\")]\n"
        "print(cand[0][\"browser_download_url\"] if cand else \"\")\n";
    std::ostringstream sh;
    sh << "set -e; mkdir -p " << shell_quote(rel.string()) << "; ";
    sh << "url=$(python3 -c " << shell_quote(py) << "); ";
    sh << "if [ -z \"$url\" ]; then echo 'No matching llama.cpp release asset found.' >&2; exit 1; fi; ";
    sh << "echo \"Downloading $url\"; ";
    sh << "curl -fsSL \"$url\" -o " << shell_quote((rel / "llama.zip").string()) << "; ";
    sh << "unzip -o " << shell_quote((rel / "llama.zip").string()) << " -d " << shell_quote(rel.string()) << "; ";
    sh << "bin=$(find " << shell_quote(rel.string()) << " -type f -name llama-server | head -n1); ";
    sh << "if [ -z \"$bin\" ]; then echo 'llama-server not found in archive.' >&2; exit 1; fi; ";
    sh << "cp \"$bin\" " << shell_quote((rel / "llama-server").string()) << "; ";
    sh << "chmod +x " << shell_quote((rel / "llama-server").string());
    plan.push_back({"Downloading llama.cpp release",
                    {"bash", "-c", sh.str()}, root, false});
#endif
    return plan;
}

std::vector<LlamaInstallStep> build_source_plan(const LlamaProvisionConfig& cfg, bool upgrade) {
    const fs::path root(cfg.provision_dir);
    const fs::path src = root / "llama.cpp-src";
    const fs::path build = src / "build";
    const std::string ref = cfg.version == "latest" ? std::string("master") : cfg.version;

    std::vector<LlamaInstallStep> plan;
    if (!fs::exists(src / ".git"))
        plan.push_back({"Cloning llama.cpp source",
                        {"git", "clone", kLlamaCppRepoUrl, src.string()}, root, false});
    plan.push_back({"Fetching tags", {"git", "fetch", "--tags", "--all"}, src, false});
    plan.push_back({"Checking out " + ref, {"git", "checkout", ref}, src, false});
    if (cfg.version == "latest")
        plan.push_back({"Updating source", {"git", "pull", "--ff-only"}, src, true});

    std::vector<std::string> configure = {
        "cmake", "-B", build.string(),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
    };
    for (auto& f : accelerator_cmake_flags(cfg)) configure.push_back(f);
    for (auto& f : cfg.cmake_args) configure.push_back(f);
    plan.push_back({"Configuring llama.cpp (CMake)", configure, src, false});

    plan.push_back({"Building llama-server",
                    {"cmake", "--build", build.string(), "--config", "Release",
                     "--target", "llama-server", "--parallel"},
                    src, false});
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

fs::path managed_llama_executable_path(const LlamaProvisionConfig& raw_cfg) {
    const LlamaProvisionConfig cfg = normalized_config(raw_cfg);
    return managed_executable_candidates(cfg).front();
}

std::vector<LlamaInstallStep> build_llama_install_plan(const LlamaProvisionConfig& raw_cfg,
                                                       bool upgrade) {
    const LlamaProvisionConfig cfg = normalized_config(raw_cfg);
    if (cfg.install_method == "release") return build_release_plan(cfg);
    // auto and source both build from source: it is the only path that works
    // everywhere (notably the DGX Spark's aarch64 + sm_121, which has no wheel).
    return build_source_plan(cfg, upgrade);
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
    if (!runner_.capture_first_line) runner_.capture_first_line = default_capture_first_line;
    if (!runner_.fetch_latest) {
        runner_.fetch_latest =
            [capture = runner_.capture_first_line](const LlamaProvisionConfig& c) {
                return default_fetch_latest(c, capture);
            };
    }
    status_ = make_base_status();
}

void LlamaCppProvisioner::set_log_sink(LogSink sink) { log_sink_ = std::move(sink); }
void LlamaCppProvisioner::set_progress_sink(ProgressSink sink) { progress_sink_ = std::move(sink); }
void LlamaCppProvisioner::set_cancel_check(CancelCheck check) { cancel_check_ = std::move(check); }

void LlamaCppProvisioner::emit_progress(const VllmInstallProgress& p) {
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
    return status;
}

void LlamaCppProvisioner::set_status(const LlamaRuntimeStatus& status) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    status_ = status;
}

LlamaRuntimeStatus LlamaCppProvisioner::status() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_;
}

LlamaProvisionConfig LlamaCppProvisioner::config() const { return cfg_; }

std::string LlamaCppProvisioner::capture_version(const std::string& executable) const {
    return runner_.capture_first_line({executable, "--version"}, {});
}

LlamaRuntimeStatus LlamaCppProvisioner::ensure_runtime() {
    LlamaRuntimeStatus status = make_base_status();

    if (const std::string resolved = resolve_llama_executable(cfg_.requested_executable);
        !resolved.empty()) {
        status.status = "resolved";
        status.method = "path";
        status.managed = false;
        status.executable_path = resolved;
        const std::string version = capture_version(resolved);
        if (!version.empty()) status.version = version;
        set_status(status);
        return status;
    }

    // An already-built managed runtime is usable regardless of auto-provision.
    for (const auto& cand : managed_executable_candidates(cfg_)) {
        if (!file_exists(cand)) continue;
        const std::string version = capture_version(cand.string());
        status.status = "ready";
        status.managed = true;
        status.executable_path = cand.string();
        if (!version.empty()) status.version = version;
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

LlamaRuntimeStatus LlamaCppProvisioner::run_managed_install(bool upgrade) {
    LlamaRuntimeStatus status = make_base_status();

    const fs::path root(cfg_.provision_dir);
    std::error_code ec;
    fs::create_directories(root, ec);
    if (ec) {
        status.status = "failed";
        status.last_error = "failed to create llama.cpp provision directory: " + ec.message();
        set_status(status);
        return status;
    }

    status.status = "provisioning";
    status.managed = true;
    set_status(status);

    auto finish_progress = [&]() {
        VllmInstallProgress done;   // active=false
        emit_progress(done);
    };
    auto canceled_status = [&]() {
        status.status = "failed";
        status.managed = true;
        status.last_error =
            std::string("llama.cpp ") + (upgrade ? "update" : "install") + " canceled";
        set_status(status);
        finish_progress();
        return status;
    };

    const std::vector<LlamaInstallStep> plan = build_llama_install_plan(cfg_, upgrade);
    const int total = static_cast<int>(plan.size());
    MM_INFO("llama.cpp {} plan: {} step(s)", upgrade ? "upgrade" : "install", total);

    for (int i = 0; i < total; ++i) {
        const LlamaInstallStep& step = plan[static_cast<size_t>(i)];
        if (cancel_check_ && cancel_check_()) return canceled_status();

        VllmInstallProgress prog;
        prog.active = true;
        prog.step = i + 1;
        prog.total_steps = total;
        prog.fraction = -1.0;
        prog.stage = step.label;
        emit_progress(prog);
        if (log_sink_) log_sink_("[llama-install] " + step.label, false);

        auto on_line = [&](const std::string& line, bool is_stderr) {
            if (log_sink_) log_sink_(line, is_stderr);
            const double f = parse_vllm_install_fraction(line);
            if (f >= 0.0) prog.fraction = f;
            prog.last_line = line;
            emit_progress(prog);
        };

        std::string step_err;
        const fs::path cwd = step.cwd.empty() ? root : step.cwd;
        const int rc = runner_.run(step.argv, cwd, on_line, cancel_check_, &step_err);
        if (cancel_check_ && cancel_check_()) return canceled_status();
        if (rc != 0 && !step.allow_failure) {
            status.status = "failed";
            status.last_error = "llama.cpp install step '" + step.label + "' failed"
                + (step_err.empty() ? (" (exit " + std::to_string(rc) + ")")
                                    : (": " + step_err));
            set_status(status);
            finish_progress();
            return status;
        }
    }

    std::string found;
    for (const auto& cand : managed_executable_candidates(cfg_)) {
        if (file_exists(cand)) { found = cand.string(); break; }
    }
    if (found.empty()) {
        status.status = "failed";
        status.last_error =
            "llama.cpp build finished but no llama-server executable was found under "
            + root.string();
        set_status(status);
        finish_progress();
        return status;
    }

    const std::string version = capture_version(found);
    status.status = "ready";
    status.managed = true;
    status.executable_path = found;
    if (!version.empty()) status.version = version;
    status.latest_version.clear();
    status.update_available = false;
    status.last_error.clear();
    set_status(status);
    finish_progress();
    MM_INFO("llama.cpp runtime {}: {}", upgrade ? "updated" : "ready", status.executable_path);
    return status;
}

LlamaRuntimeStatus LlamaCppProvisioner::check_for_update() {
    LlamaRuntimeStatus status = this->status();
    const std::string latest =
        runner_.fetch_latest ? util::trim(runner_.fetch_latest(cfg_)) : std::string{};
    status.latest_version = latest;
    const std::string installed = util::trim(status.version);
    // llama.cpp release tags are ordered build numbers (e.g. b4321); a plain
    // string inequality against a non-empty newer tag is a good-enough signal.
    status.update_available =
        !latest.empty() && !installed.empty() && installed != latest;
    if (status.update_available)
        MM_INFO("llama.cpp update available: {} -> {}", installed, latest);
    set_status(status);
    return status;
}

LlamaRuntimeStatus LlamaCppProvisioner::update_runtime() {
    if (const std::string resolved = resolve_llama_executable(cfg_.requested_executable);
        !resolved.empty()) {
        LlamaRuntimeStatus status = this->status();
        status.last_error =
            "llama-server is resolved from PATH (" + resolved +
            "); update it yourself. Point llama_server_path at a managed build to "
            "let the node update it.";
        set_status(status);
        return status;
    }
    return run_managed_install(/*upgrade=*/true);
}

} // namespace mm
