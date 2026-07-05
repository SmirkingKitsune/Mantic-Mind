#include "node/vllm_provisioner.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"
#include "node/vllm_runtime.hpp"

#include <algorithm>
#include <array>
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

VllmProvisionConfig normalized_config(VllmProvisionConfig cfg) {
    cfg.requested_executable = strip_wrapping_quotes(cfg.requested_executable);
    if (cfg.requested_executable.empty()) cfg.requested_executable = "vllm";
    cfg.install_method = normalize_vllm_install_method(cfg.install_method);
    cfg.version = util::trim(cfg.version);
    if (cfg.version.empty()) cfg.version = "latest";
    cfg.python_path = strip_wrapping_quotes(cfg.python_path);
    if (cfg.platform.empty()) cfg.platform = current_vllm_platform();
    if (cfg.arch.empty()) cfg.arch = current_vllm_arch();
    if (cfg.provision_dir.empty()) {
        cfg.provision_dir = (fs::path("data") / "runtimes" / "vllm").string();
    }
    return cfg;
}

std::string shell_quote_posix(const std::string& value) {
    std::string out = "'";
    for (char ch : value) {
        if (ch == '\'') out += "'\\''";
        else out.push_back(ch);
    }
    out += "'";
    return out;
}

std::string shell_quote_windows(const std::string& value) {
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') out += "\\\"";
        else out.push_back(ch);
    }
    out += "\"";
    return out;
}

std::string shell_quote(const std::string& value) {
#ifdef _WIN32
    return shell_quote_windows(value);
#else
    return shell_quote_posix(value);
#endif
}

std::string ps_quote(const std::string& value) {
    return "'" + util::replace_all(value, "'", "''") + "'";
}

std::string bash_quote(const std::string& value) {
    return shell_quote_posix(value);
}

std::string join_shell_command(const std::vector<std::string>& args) {
    std::string out;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i > 0) out += " ";
        out += shell_quote(args[i]);
    }
    return out;
}

int default_run_command(const std::vector<std::string>& args,
                        const fs::path& cwd,
                        std::string* error) {
    if (args.empty()) {
        if (error) *error = "empty command";
        return 1;
    }
    std::string cmd = join_shell_command(args);
    if (!cwd.empty()) {
#ifdef _WIN32
        cmd = "cd /d " + shell_quote(cwd.string()) + " && " + cmd;
#else
        cmd = "cd " + shell_quote(cwd.string()) + " && " + cmd;
#endif
    }
    const int rc = std::system(cmd.c_str());
    if (rc != 0 && error) {
        *error = "command exited with status " + std::to_string(rc) + ": " + args.front();
    }
    return rc;
}

std::string default_capture_first_line(const std::vector<std::string>& args,
                                       const fs::path& cwd) {
    if (args.empty()) return {};
    std::string cmd = join_shell_command(args);
    if (!cwd.empty()) {
#ifdef _WIN32
        cmd = "cd /d " + shell_quote(cwd.string()) + " && " + cmd;
#else
        cmd = "cd " + shell_quote(cwd.string()) + " && " + cmd;
#endif
    }
#ifdef _WIN32
    FILE* f = _popen((cmd + " 2>nul").c_str(), "r");
#else
    FILE* f = ::popen((cmd + " 2>/dev/null").c_str(), "r");
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

std::string version_package_suffix(const std::string& version) {
    if (version.empty() || version == "latest") return {};
    return "==" + version;
}

std::string windows_source_checkout_ref(const VllmProvisionConfig& cfg) {
    return cfg.version == "latest" ? kWindowsVllmBranch : cfg.version;
}

class ProvisionLock {
public:
    explicit ProvisionLock(fs::path path) : path_(std::move(path)) {
        std::error_code ec;
        acquired_ = fs::create_directory(path_, ec);
    }
    ~ProvisionLock() {
        if (!acquired_) return;
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    bool acquired() const { return acquired_; }

private:
    fs::path path_;
    bool acquired_ = false;
};

std::string build_windows_script(const VllmProvisionConfig& cfg) {
    const fs::path root(cfg.provision_dir);
    const fs::path venv = root / "venv";
    const fs::path src = root / "vllm-windows-src";
    const std::string python = cfg.python_path.empty() ? "python" : cfg.python_path;
    const std::string version = cfg.version;
    const std::string method = cfg.install_method;

    std::ostringstream s;
    s << "$ErrorActionPreference = 'Stop'\n";
    s << "$Venv = " << ps_quote(venv.string()) << "\n";
    s << "$Src = " << ps_quote(src.string()) << "\n";
    s << "$Python = " << ps_quote(python) << "\n";
    s << "$Method = " << ps_quote(method) << "\n";
    s << "$Version = " << ps_quote(version) << "\n";
    s << "New-Item -ItemType Directory -Force -Path " << ps_quote(root.string()) << " | Out-Null\n";
    s << "if (Get-Command uv -ErrorAction SilentlyContinue) { uv venv $Venv --python $Python } else { & $Python -m venv $Venv }\n";
    s << "$VenvPython = Join-Path $Venv 'Scripts\\python.exe'\n";
    s << "& $VenvPython -m pip install --upgrade pip\n";
    s << "function Install-Source {\n";
    s << "  if (!(Test-Path (Join-Path $Src '.git'))) { git clone " << ps_quote(kWindowsVllmRepoUrl) << " $Src }\n";
    s << "  Push-Location $Src\n";
    s << "  git fetch --tags --all\n";
    s << "  git checkout " << ps_quote(windows_source_checkout_ref(cfg)) << "\n";
    s << "  & $VenvPython -m pip install -e .\n";
    s << "  Pop-Location\n";
    s << "}\n";
    s << "if ($Method -eq 'source') { Install-Source; exit 0 }\n";
    s << "$ReleaseUri = if ($Version -eq 'latest') { 'https://api.github.com/repos/SystemPanic/vllm-windows/releases/latest' } else { \"https://api.github.com/repos/SystemPanic/vllm-windows/releases/tags/$Version\" }\n";
    s << "$Release = Invoke-RestMethod -Uri $ReleaseUri\n";
    s << "$Asset = $Release.assets | Where-Object { $_.name -like '*.whl' } | Select-Object -First 1\n";
    s << "if ($null -eq $Asset) {\n";
    s << "  if ($Method -eq 'auto') { Install-Source; exit 0 }\n";
    s << "  throw 'No vLLM Windows wheel found in the requested release.'\n";
    s << "}\n";
    s << "$Wheel = Join-Path $env:TEMP $Asset.name\n";
    s << "Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $Wheel\n";
    s << "& $VenvPython -m pip install $Wheel\n";
    return s.str();
}

std::string build_linux_script(const VllmProvisionConfig& cfg) {
    const fs::path root(cfg.provision_dir);
    const fs::path venv = root / "venv";
    const fs::path src = root / "vllm-src";
    const std::string python = cfg.python_path.empty() ? "python3" : cfg.python_path;
    const std::string pkg = "vllm" + version_package_suffix(cfg.version);

    std::ostringstream s;
    s << "#!/usr/bin/env bash\n";
    s << "set -euo pipefail\n";
    s << "ROOT=" << bash_quote(root.string()) << "\n";
    s << "VENV=" << bash_quote(venv.string()) << "\n";
    s << "SRC=" << bash_quote(src.string()) << "\n";
    s << "PY=" << bash_quote(python) << "\n";
    s << "METHOD=" << bash_quote(cfg.install_method) << "\n";
    s << "VERSION=" << bash_quote(cfg.version) << "\n";
    s << "mkdir -p \"$ROOT\"\n";
    s << "install_source() {\n";
    s << "  if [[ ! -d \"$SRC/.git\" ]]; then git clone " << bash_quote(kOfficialVllmRepoUrl) << " \"$SRC\"; fi\n";
    s << "  cd \"$SRC\"\n";
    s << "  git fetch --tags --all\n";
    s << "  if [[ \"$VERSION\" == \"latest\" ]]; then git checkout main; git pull --ff-only || true; else git checkout \"$VERSION\"; fi\n";
    s << "  \"$PY\" -m venv \"$VENV\"\n";
    s << "  \"$VENV/bin/python\" -m pip install --upgrade pip\n";
    s << "  \"$VENV/bin/python\" -m pip install -e .\n";
    s << "}\n";
    s << "if [[ \"$METHOD\" == \"source\" ]]; then install_source; exit 0; fi\n";
    s << "if command -v uv >/dev/null 2>&1; then\n";
    s << "  uv venv \"$VENV\" --python \"$PY\"\n";
    s << "  uv pip install --python \"$VENV/bin/python\" " << bash_quote(pkg) << " --torch-backend=auto\n";
    s << "else\n";
    s << "  \"$PY\" -m venv \"$VENV\"\n";
    s << "  \"$VENV/bin/python\" -m pip install --upgrade pip\n";
    s << "  \"$VENV/bin/python\" -m pip install " << bash_quote(pkg) << "\n";
    s << "fi\n";
    return s.str();
}

std::string build_macos_metal_script(const VllmProvisionConfig& cfg) {
    const fs::path root(cfg.provision_dir);
    const fs::path src = root / "vllm-metal-src";
    const std::string python = cfg.python_path.empty() ? "python3" : cfg.python_path;

    std::ostringstream s;
    s << "#!/usr/bin/env bash\n";
    s << "set -euo pipefail\n";
    s << "ROOT=" << bash_quote(root.string()) << "\n";
    s << "SRC=" << bash_quote(src.string()) << "\n";
    s << "PY=" << bash_quote(python) << "\n";
    s << "VERSION=" << bash_quote(cfg.version) << "\n";
    s << "if [[ \"$(uname -s)\" != \"Darwin\" || \"$(uname -m)\" != \"arm64\" ]]; then echo 'vllm-metal requires macOS on Apple Silicon.' >&2; exit 2; fi\n";
    s << "\"$PY\" - <<'PY'\n";
    s << "import platform, sys\n";
    s << "if platform.machine() != 'arm64' or sys.version_info[:2] != (3, 12):\n";
    s << "    raise SystemExit('vllm-metal requires native arm64 Python 3.12')\n";
    s << "PY\n";
    s << "xcode-select -p >/dev/null\n";
    s << "mkdir -p \"$ROOT\"\n";
    s << "if [[ ! -d \"$SRC/.git\" ]]; then git clone " << bash_quote(kMetalVllmRepoUrl) << " \"$SRC\"; fi\n";
    s << "cd \"$SRC\"\n";
    s << "git fetch --tags --all\n";
    s << "if [[ \"$VERSION\" == \"latest\" ]]; then git checkout main; git pull --ff-only || true; else git checkout \"$VERSION\"; fi\n";
    s << "export PATH=\"$(dirname \"$PY\"):$PATH\"\n";
    s << "bash ./install.sh\n";
    return s.str();
}

} // namespace

std::string normalize_vllm_install_method(const std::string& method) {
    const std::string m = util::to_lower(util::trim(method));
    if (m == "wheel" || m == "source") return m;
    return "auto";
}

fs::path managed_vllm_executable_path(const VllmProvisionConfig& raw_cfg) {
    const VllmProvisionConfig cfg = normalized_config(raw_cfg);
    const fs::path root(cfg.provision_dir);
    if (is_apple_silicon_environment(cfg.platform, cfg.arch)) {
        return root / "vllm-metal-src" / ".venv-vllm-metal" / "bin" / "vllm";
    }
    const bool windows_target = cfg.platform == "windows";
    if (windows_target) {
        return root / "venv" / "Scripts" / "vllm.exe";
    }
    return root / "venv" / "bin" / "vllm";
}

std::string build_vllm_provision_script(const VllmProvisionConfig& raw_cfg) {
    const VllmProvisionConfig cfg = normalized_config(raw_cfg);
    if (cfg.platform == "windows") return build_windows_script(cfg);
    if (cfg.platform == "macos") return build_macos_metal_script(cfg);
    return build_linux_script(cfg);
}

std::string resolve_vllm_executable(const std::string& raw_executable) {
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

bool vllm_runtime_usable(const VllmRuntimeStatus& status) {
    return (status.status == "resolved" || status.status == "ready")
        && !status.executable_path.empty();
}

VllmProvisioner::VllmProvisioner(VllmProvisionConfig cfg,
                                 VllmCommandRunner runner)
    : cfg_(normalized_config(std::move(cfg)))
    , runner_(std::move(runner))
{
    if (!runner_.run) runner_.run = default_run_command;
    if (!runner_.capture_first_line) runner_.capture_first_line = default_capture_first_line;
    status_ = make_base_status();
}

VllmRuntimeStatus VllmProvisioner::make_base_status() const {
    VllmRuntimeStatus status;
    status.status = "disabled";
    status.platform = cfg_.platform;
    status.method = is_apple_silicon_environment(cfg_.platform, cfg_.arch)
        ? "metal"
        : cfg_.install_method;
    status.source_repo = default_vllm_repo_url_for_environment(cfg_.platform, cfg_.arch);
    status.version = cfg_.version;
    return status;
}

void VllmProvisioner::set_status(const VllmRuntimeStatus& status) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    status_ = status;
}

VllmRuntimeStatus VllmProvisioner::status() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_;
}

VllmProvisionConfig VllmProvisioner::config() const {
    return cfg_;
}

std::string VllmProvisioner::capture_version(const std::string& executable) const {
    return runner_.capture_first_line({executable, "--version"}, {});
}

int VllmProvisioner::run_script(const fs::path& script_path, std::string* error) {
    if (cfg_.platform == "windows") {
        return runner_.run({"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                            "-File", script_path.string()},
                           fs::path(cfg_.provision_dir), error);
    }
    return runner_.run({"bash", script_path.string()}, fs::path(cfg_.provision_dir), error);
}

VllmRuntimeStatus VllmProvisioner::ensure_runtime() {
    VllmRuntimeStatus status = make_base_status();

    if (const std::string resolved = resolve_vllm_executable(cfg_.requested_executable);
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

    if (cfg_.platform == "macos" &&
        !is_apple_silicon_environment(cfg_.platform, cfg_.arch)) {
        status.status = "failed";
        status.last_error = "vllm-metal requires macOS on Apple Silicon; detected arch="
                          + (cfg_.arch.empty() ? std::string{"unknown"} : cfg_.arch);
        set_status(status);
        return status;
    }

    if (!cfg_.auto_provision) {
        status.status = "disabled";
        status.last_error = "vLLM executable not found and auto-provisioning is disabled: "
                          + cfg_.requested_executable;
        set_status(status);
        return status;
    }

    const fs::path root(cfg_.provision_dir);
    std::error_code ec;
    fs::create_directories(root, ec);
    if (ec) {
        status.status = "failed";
        status.last_error = "failed to create vLLM provision directory: " + ec.message();
        set_status(status);
        return status;
    }

    if (const fs::path managed = managed_vllm_executable_path(cfg_);
        file_exists(managed)) {
        const std::string version = capture_version(managed.string());
        if (!version.empty()) {
            status.status = "ready";
            status.managed = true;
            status.executable_path = managed.string();
            status.version = version;
            set_status(status);
            return status;
        }
    }

    ProvisionLock lock(root / ".provision.lock");
    if (!lock.acquired()) {
        status.status = "failed";
        status.last_error = "another vLLM provisioning run is already active in "
                          + root.string();
        set_status(status);
        return status;
    }

    status.status = "provisioning";
    status.managed = true;
    set_status(status);

    const fs::path script_path = root / (cfg_.platform == "windows"
        ? "install-vllm.ps1"
        : "install-vllm.sh");
    {
        std::ofstream script(script_path, std::ios::out | std::ios::trunc);
        if (!script) {
            status.status = "failed";
            status.last_error = "failed to write vLLM provisioning script: "
                              + script_path.string();
            set_status(status);
            return status;
        }
        script << build_vllm_provision_script(cfg_);
    }

#ifndef _WIN32
    fs::permissions(script_path,
                    fs::perms::owner_exec | fs::perms::owner_read | fs::perms::owner_write,
                    fs::perm_options::add, ec);
#endif

    std::string run_error;
    const int rc = run_script(script_path, &run_error);
    if (rc != 0) {
        status.status = "failed";
        status.last_error = run_error.empty()
            ? "vLLM provisioning command failed"
            : run_error;
        set_status(status);
        return status;
    }

    const fs::path managed = managed_vllm_executable_path(cfg_);
    if (!file_exists(managed)) {
        status.status = "failed";
        status.last_error = "vLLM provisioning finished but no managed executable was found at "
                          + managed.string();
        set_status(status);
        return status;
    }

    const std::string version = capture_version(managed.string());
    if (version.empty()) {
        status.status = "failed";
        status.last_error = "managed vLLM executable failed validation: "
                          + managed.string() + " --version produced no output";
        set_status(status);
        return status;
    }

    status.status = "ready";
    status.managed = true;
    status.executable_path = managed.string();
    status.version = version;
    status.last_error.clear();
    set_status(status);
    MM_INFO("vLLM runtime ready: {}", status.executable_path);
    return status;
}

} // namespace mm
