#include "node/vllm_provisioner.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"
#include "node/vllm_runtime.hpp"

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
#else
#  include <csignal>
#  include <cerrno>
#  include <unistd.h>
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
    cfg.package_manager = util::to_lower(util::trim(cfg.package_manager));
    cfg.accelerator = util::to_lower(util::trim(cfg.accelerator));
    if (cfg.accelerator.empty()) {
        // The variants we can infer without a runtime GPU probe. cuda/rocm/cpu
        // are supplied by the caller after detection; otherwise fall through to
        // torch-backend auto-detection at install time.
        if (is_apple_silicon_environment(cfg.platform, cfg.arch)) cfg.accelerator = "metal";
        else if (cfg.platform == "windows") cfg.accelerator = "windows";
    }
    if (cfg.provision_dir.empty()) {
        cfg.provision_dir = (fs::path("data") / "runtimes" / "vllm").string();
    }
    // Absolutize the provision dir. Install steps run with cwd=provision_dir and
    // pass the venv path as an argument; a relative provision_dir would resolve
    // that argument against the step cwd and nest a second copy
    // (data/runtimes/vllm/data/runtimes/vllm/venv), which then never matches the
    // executable path we validate. An absolute root makes every derived path
    // cwd-independent and self-consistent.
    {
        std::error_code ec;
        const fs::path abs = fs::absolute(cfg.provision_dir, ec);
        if (!ec) cfg.provision_dir = abs.lexically_normal().string();
    }
    return cfg;
}

// Map a detected accelerator to an explicit uv/pip --torch-backend value for the
// Linux install. "auto" lets uv resolve the exact CUDA/ROCm build; only CPU-only
// nodes pin an explicit backend.
std::string torch_backend_for_accelerator(const std::string& accelerator) {
    if (accelerator == "cpu") return "cpu";
    return "auto";
}

#ifndef _WIN32
std::string shell_quote_posix(const std::string& value) {
    std::string out = "'";
    for (char ch : value) {
        if (ch == '\'') out += "'\\''";
        else out.push_back(ch);
    }
    out += "'";
    return out;
}
#endif

#ifdef _WIN32
std::string shell_quote_windows(const std::string& value) {
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '"') out += "\\\"";
        else out.push_back(ch);
    }
    out += "\"";
    return out;
}
#endif

std::string shell_quote(const std::string& value) {
#ifdef _WIN32
    return shell_quote_windows(value);
#else
    return shell_quote_posix(value);
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

long current_process_id() {
#ifdef _WIN32
    return static_cast<long>(::GetCurrentProcessId());
#else
    return static_cast<long>(::getpid());
#endif
}

// True when a process with this id is currently running. Best-effort and
// cross-platform: a false positive keeps a lock we would otherwise steal, a
// false negative never happens for a live owner.
bool process_is_alive(long pid) {
    if (pid <= 0) return false;
#ifdef _WIN32
    HANDLE h = ::OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(pid));
    if (h == nullptr) return false;
    const DWORD w = ::WaitForSingleObject(h, 0);
    ::CloseHandle(h);
    return w == WAIT_TIMEOUT;   // still running (not yet signaled)
#else
    if (::kill(static_cast<pid_t>(pid), 0) == 0) return true;
    return errno == EPERM;      // exists but owned by another user
#endif
}

// A directory-based provisioning lock that survives a crashed owner. The lock
// dir records the owning PID; a contender steals it when that PID is gone (or
// when a PID-less dir is older than a short grace window), so a node killed
// mid-install does not brick provisioning forever.
class ProvisionLock {
public:
    explicit ProvisionLock(fs::path path) : path_(std::move(path)) {
        acquire();
    }
    ~ProvisionLock() {
        if (!acquired_) return;
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    bool acquired() const { return acquired_; }

private:
    void acquire() {
        std::error_code ec;
        if (fs::create_directory(path_, ec)) {
            write_owner();
            acquired_ = true;
            return;
        }
        if (owner_is_dead()) {
            fs::remove_all(path_, ec);
            if (fs::create_directory(path_, ec)) {
                write_owner();
                acquired_ = true;
                return;
            }
        }
        acquired_ = false;
    }

    void write_owner() {
        std::ofstream f(path_ / "owner.pid", std::ios::trunc);
        if (f) f << current_process_id() << "\n";
    }

    bool owner_is_dead() const {
        std::ifstream f(path_ / "owner.pid");
        long pid = 0;
        if (f && (f >> pid) && pid > 0) return !process_is_alive(pid);
        // No or unreadable PID file: only steal once the dir is past a short
        // grace window, so we never race a creator that hasn't written it yet.
        std::error_code ec;
        const auto mtime = fs::last_write_time(path_, ec);
        if (ec) return false;
        return (fs::file_time_type::clock::now() - mtime) > std::chrono::seconds(30);
    }

    fs::path path_;
    bool acquired_ = false;
};

// The venv's Python interpreter for this platform.
std::string venv_python(const fs::path& venv, const std::string& platform) {
    if (platform == "windows") return (venv / "Scripts" / "python.exe").string();
    return (venv / "bin" / "python").string();
}

bool tool_on_path(const std::string& name) {
    return !resolve_vllm_executable(name).empty();
}

std::vector<VllmInstallStep> build_linux_plan(const VllmProvisionConfig& cfg, bool upgrade) {
    const fs::path root(cfg.provision_dir);
    const fs::path venv = root / "venv";
    const fs::path src = root / "vllm-src";
    const std::string python = cfg.python_path.empty() ? "python3" : cfg.python_path;
    const std::string vpy = venv_python(venv, "linux");
    const std::string pkg = "vllm" + version_package_suffix(cfg.version);
    const std::string torch_backend = torch_backend_for_accelerator(cfg.accelerator);

    std::vector<VllmInstallStep> plan;
    if (cfg.install_method == "source") {
        if (!fs::exists(src / ".git"))
            plan.push_back({"Cloning vLLM source", {"git", "clone", kOfficialVllmRepoUrl, src.string()}, root, false});
        plan.push_back({"Fetching tags", {"git", "fetch", "--tags", "--all"}, src, false});
        const std::string ref = cfg.version == "latest" ? std::string("main") : cfg.version;
        plan.push_back({"Checking out " + ref, {"git", "checkout", ref}, src, false});
        if (cfg.version == "latest")
            plan.push_back({"Updating source", {"git", "pull", "--ff-only"}, src, true});
        plan.push_back({"Creating virtualenv", {python, "-m", "venv", venv.string()}, root, false});
        plan.push_back({"Upgrading pip", {vpy, "-m", "pip", "install", "--upgrade", "pip"}, root, false});
        std::vector<std::string> ins = {vpy, "-m", "pip", "install"};
        if (upgrade) ins.push_back("--upgrade");
        ins.push_back("-e");
        ins.push_back(".");
        plan.push_back({"Installing vLLM (building from source)", ins, src, false});
        return plan;
    }

    if (cfg.package_manager == "uv") {
        plan.push_back({"Creating virtualenv (uv)", {"uv", "venv", venv.string(), "--python", python}, root, false});
        std::vector<std::string> ins = {"uv", "pip", "install", "--python", vpy};
        if (upgrade) ins.push_back("--upgrade");
        ins.push_back(pkg);
        ins.push_back("--torch-backend=" + torch_backend);
        plan.push_back({"Downloading & installing vLLM", ins, root, false});
    } else {
        plan.push_back({"Creating virtualenv", {python, "-m", "venv", venv.string()}, root, false});
        plan.push_back({"Upgrading pip", {vpy, "-m", "pip", "install", "--upgrade", "pip"}, root, false});
        std::vector<std::string> ins = {vpy, "-m", "pip", "install"};
        if (upgrade) ins.push_back("--upgrade");
        ins.push_back(pkg);
        plan.push_back({"Downloading & installing vLLM", ins, root, false});
    }
    return plan;
}

std::vector<VllmInstallStep> build_windows_plan(const VllmProvisionConfig& cfg, bool upgrade) {
    const fs::path root(cfg.provision_dir);
    const fs::path venv = root / "venv";
    const fs::path src = root / "vllm-windows-src";
    const std::string python = cfg.python_path.empty() ? "python" : cfg.python_path;
    const std::string vpy = venv_python(venv, "windows");

    std::vector<VllmInstallStep> plan;
    if (cfg.package_manager == "uv")
        plan.push_back({"Creating virtualenv (uv)", {"uv", "venv", venv.string(), "--python", python}, root, false});
    else
        plan.push_back({"Creating virtualenv", {python, "-m", "venv", venv.string()}, root, false});
    plan.push_back({"Upgrading pip", {vpy, "-m", "pip", "install", "--upgrade", "pip"}, root, false});

    if (cfg.install_method == "source") {
        const std::string ref = windows_source_checkout_ref(cfg);
        if (!fs::exists(src / ".git"))
            plan.push_back({"Cloning vLLM (Windows) source", {"git", "clone", kWindowsVllmRepoUrl, src.string()}, root, false});
        plan.push_back({"Fetching tags", {"git", "fetch", "--tags", "--all"}, src, false});
        plan.push_back({"Checking out " + ref, {"git", "checkout", ref}, src, false});
        if (cfg.version == "latest")
            plan.push_back({"Updating source", {"git", "pull", "--ff-only"}, src, true});
        std::vector<std::string> ins = {vpy, "-m", "pip", "install"};
        if (upgrade) ins.push_back("--upgrade");
        ins.push_back("-e");
        ins.push_back(".");
        plan.push_back({"Installing vLLM (building from source)", ins, src, false});
        return plan;
    }

    // wheel/auto: resolve + download the release wheel (the single unavoidable
    // inline HTTP call — no in-process HTTPS), then pip install it (which pulls
    // torch + deps with visible progress).
    const fs::path wheel = root / "vllm-windows.whl";
    const std::string rel_uri = cfg.version == "latest"
        ? std::string("https://api.github.com/repos/SystemPanic/vllm-windows/releases/latest")
        : "https://api.github.com/repos/SystemPanic/vllm-windows/releases/tags/" + cfg.version;
    std::ostringstream ps;
    ps << "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; ";
    ps << "$r = Invoke-RestMethod -Uri '" << rel_uri << "' -Headers @{ 'User-Agent'='mantic-mind' }; ";
    ps << "$a = $r.assets | Where-Object { $_.name -like '*.whl' } | Select-Object -First 1; ";
    ps << "if ($null -eq $a) { throw 'No vLLM Windows wheel found in the requested release.' }; ";
    ps << "Write-Host ('Downloading ' + $a.name); ";
    ps << "Invoke-WebRequest -Uri $a.browser_download_url -OutFile '" << wheel.string() << "'";
    plan.push_back({"Resolving & downloading vLLM wheel",
                    {"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps.str()},
                    root, false});

    std::vector<std::string> ins = {vpy, "-m", "pip", "install"};
    if (upgrade) { ins.push_back("--upgrade"); ins.push_back("--force-reinstall"); }
    ins.push_back(wheel.string());
    plan.push_back({"Downloading & installing vLLM", ins, root, false});
    return plan;
}

std::vector<VllmInstallStep> build_macos_metal_plan(const VllmProvisionConfig& cfg, bool upgrade) {
    // Metal is a source build: the fetch/checkout/pull below moves to the newest
    // ref for "latest" (or the exact tag otherwise) and install.sh rebuilds — so
    // fresh install and upgrade run the same steps.
    (void)upgrade;
    const fs::path root(cfg.provision_dir);
    const fs::path src = root / "vllm-metal-src";
    const std::string python = cfg.python_path.empty() ? "python3" : cfg.python_path;
    const std::string ref = cfg.version == "latest" ? std::string("main") : cfg.version;

    std::vector<VllmInstallStep> plan;
    if (!fs::exists(src / ".git"))
        plan.push_back({"Cloning vLLM Metal source", {"git", "clone", kMetalVllmRepoUrl, src.string()}, root, false});
    plan.push_back({"Fetching tags", {"git", "fetch", "--tags", "--all"}, src, false});
    plan.push_back({"Checking out " + ref, {"git", "checkout", ref}, src, false});
    if (cfg.version == "latest")
        plan.push_back({"Updating source", {"git", "pull", "--ff-only"}, src, true});
    // install.sh is the vendor build entry point; run it with the chosen Python
    // on PATH via an inline `bash -c` (a command, not a persisted script file).
    const fs::path py_path(python);
    std::string sh = "set -e; ";
    if (py_path.has_parent_path())
        sh += "export PATH=\"" + py_path.parent_path().string() + ":$PATH\"; ";
    sh += "bash ./install.sh";
    plan.push_back({"Building vLLM Metal (install.sh)", {"bash", "-c", sh}, src, false});
    return plan;
}

// Best-effort progress extraction from a pip/uv/git output line. Prefers an
// explicit "NN%"; falls back to an "A/B" byte ratio. Returns <0 for no signal,
// so the caller shows an indeterminate bar.
double parse_vllm_install_fraction_impl(const std::string& line) {
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
                const double den = std::stod(line.substr(slash + 1, e - (slash + 1)));
                if (den > 0.0 && num >= 0.0 && num <= den) return clamp01(num / den);
            } catch (...) {}
        }
    }
    return -1.0;
}

// Default upstream-version probe. Reuses the command-capture seam so it stays
// injectable in tests. Windows queries the SystemPanic release tag via
// PowerShell (system cert store); POSIX uses the node's Python + urllib (the
// same interpreter vLLM needs) against PyPI (official) or the vllm-metal
// releases (Apple Silicon). Returns "" on any failure — callers treat an empty
// result as "unknown", never as "up to date".
std::string default_fetch_latest(const VllmProvisionConfig& cfg,
                                 const VllmCommandRunner::CaptureFn& capture) {
    if (!capture) return {};

    if (cfg.platform == "windows") {
        const std::string url =
            "https://api.github.com/repos/SystemPanic/vllm-windows/releases/latest";
        const std::string ps =
            "try { (Invoke-RestMethod -Uri '" + url +
            "' -Headers @{ 'User-Agent' = 'mantic-mind' }).tag_name } catch { '' }";
        return util::trim(capture({"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                                   "-Command", ps}, {}));
    }

    const std::string python = cfg.python_path.empty() ? "python3" : cfg.python_path;
    std::string url;
    std::string expr;
    if (is_apple_silicon_environment(cfg.platform, cfg.arch)) {
        url = "https://api.github.com/repos/vllm-project/vllm-metal/releases/latest";
        expr = "d.get(\"tag_name\", \"\")";
    } else {
        url = "https://pypi.org/pypi/vllm/json";
        expr = "d.get(\"info\", {}).get(\"version\", \"\")";
    }
    // Double-quoted Python string literals keep the whole program free of single
    // quotes so POSIX shell-quoting wraps it in one clean '...' argument.
    const std::string program =
        "import json, urllib.request\n"
        "req = urllib.request.Request(\"" + url + "\", headers={\"User-Agent\": \"mantic-mind\"})\n"
        "d = json.load(urllib.request.urlopen(req, timeout=10))\n"
        "print(" + expr + ")\n";
    return util::trim(capture({python, "-c", program}, {}));
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

std::vector<VllmInstallStep> build_vllm_install_plan(const VllmProvisionConfig& raw_cfg,
                                                     bool upgrade) {
    VllmProvisionConfig cfg = normalized_config(raw_cfg);
    if (cfg.package_manager.empty())
        cfg.package_manager = tool_on_path("uv") ? "uv" : "pip";
    if (cfg.platform == "windows") return build_windows_plan(cfg, upgrade);
    if (cfg.platform == "macos") return build_macos_metal_plan(cfg, upgrade);
    return build_linux_plan(cfg, upgrade);
}

double parse_vllm_install_fraction(const std::string& line) {
    return parse_vllm_install_fraction_impl(line);
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
            [capture = runner_.capture_first_line](const VllmProvisionConfig& c) {
                return default_fetch_latest(c, capture);
            };
    }
    status_ = make_base_status();
}

void VllmProvisioner::set_log_sink(LogSink sink) {
    log_sink_ = std::move(sink);
}

void VllmProvisioner::set_progress_sink(ProgressSink sink) {
    progress_sink_ = std::move(sink);
}

void VllmProvisioner::set_cancel_check(CancelCheck check) {
    cancel_check_ = std::move(check);
}

void VllmProvisioner::emit_progress(const VllmInstallProgress& p) {
    if (progress_sink_) progress_sink_(p);
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
    status.accelerator = cfg_.accelerator;
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

    // An already-provisioned managed runtime is usable regardless of the
    // auto-provision setting — that flag only gates *new* installs.
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

    return run_managed_install(/*upgrade=*/false);
}

VllmRuntimeStatus VllmProvisioner::run_managed_install(bool upgrade) {
    VllmRuntimeStatus status = make_base_status();

    if (cfg_.platform == "macos" &&
        !is_apple_silicon_environment(cfg_.platform, cfg_.arch)) {
        status.status = "failed";
        status.last_error = "vllm-metal requires macOS on Apple Silicon; detected arch="
                          + (cfg_.arch.empty() ? std::string{"unknown"} : cfg_.arch);
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

    auto finish_progress = [&]() {
        VllmInstallProgress done;   // active=false
        emit_progress(done);
    };
    auto canceled_status = [&]() {
        status.status = "failed";
        status.managed = true;
        status.last_error = std::string("vLLM ")
                          + (upgrade ? "update" : "install")
                          + " canceled";
        set_status(status);
        finish_progress();
        return status;
    };

    const std::vector<VllmInstallStep> plan = build_vllm_install_plan(cfg_, upgrade);
    const int total = static_cast<int>(plan.size());
    MM_INFO("vLLM {} plan: {} step(s)", upgrade ? "upgrade" : "install", total);

    for (int i = 0; i < total; ++i) {
        const VllmInstallStep& step = plan[static_cast<size_t>(i)];
        if (cancel_check_ && cancel_check_()) {
            return canceled_status();
        }

        VllmInstallProgress prog;
        prog.active = true;
        prog.step = i + 1;
        prog.total_steps = total;
        prog.fraction = -1.0;   // indeterminate until a line carries progress
        prog.stage = step.label;
        emit_progress(prog);
        if (log_sink_) log_sink_("[vllm-install] " + step.label, false);

        // run_streamed_command serializes on_line across its reader threads, so
        // mutating `prog` here is race-free.
        auto on_line = [&](const std::string& line, bool is_stderr) {
            if (log_sink_) log_sink_(line, is_stderr);
            const double f = parse_vllm_install_fraction_impl(line);
            if (f >= 0.0) prog.fraction = f;
            prog.last_line = line;
            emit_progress(prog);
        };

        std::string step_err;
        const fs::path cwd = step.cwd.empty() ? root : step.cwd;
        const int rc = runner_.run(step.argv, cwd, on_line, cancel_check_, &step_err);
        if (cancel_check_ && cancel_check_()) {
            return canceled_status();
        }
        if (rc != 0 && !step.allow_failure) {
            status.status = "failed";
            status.last_error = "vLLM install step '" + step.label + "' failed"
                + (step_err.empty() ? (" (exit " + std::to_string(rc) + ")")
                                    : (": " + step_err));
            set_status(status);
            finish_progress();
            return status;
        }
    }

    const fs::path managed = managed_vllm_executable_path(cfg_);
    if (!file_exists(managed)) {
        status.status = "failed";
        status.last_error = "vLLM install finished but no managed executable was found at "
                          + managed.string();
        set_status(status);
        finish_progress();
        return status;
    }

    const std::string version = capture_version(managed.string());
    if (version.empty()) {
        status.status = "failed";
        status.last_error = "managed vLLM executable failed validation: "
                          + managed.string() + " --version produced no output";
        set_status(status);
        finish_progress();
        return status;
    }

    status.status = "ready";
    status.managed = true;
    status.executable_path = managed.string();
    status.version = version;
    status.latest_version.clear();
    status.update_available = false;
    status.last_error.clear();
    set_status(status);
    finish_progress();
    MM_INFO("vLLM runtime {}: {}", upgrade ? "updated" : "ready", status.executable_path);
    return status;
}

VllmRuntimeStatus VllmProvisioner::check_for_update() {
    VllmRuntimeStatus status = this->status();

    const std::string latest =
        runner_.fetch_latest ? util::trim(runner_.fetch_latest(cfg_)) : std::string{};
    status.latest_version = latest;

    const std::string installed = util::trim(status.version);
    status.update_available =
        !latest.empty() && !installed.empty() &&
        compare_vllm_versions(installed, latest) < 0;

    if (status.update_available) {
        MM_INFO("vLLM update available: {} -> {}", installed, latest);
    }
    set_status(status);
    return status;
}

VllmRuntimeStatus VllmProvisioner::update_runtime() {
    // A vLLM resolved from PATH lives in an environment the node does not own;
    // refuse to mutate it rather than shadow it with a divergent managed copy.
    if (const std::string resolved = resolve_vllm_executable(cfg_.requested_executable);
        !resolved.empty()) {
        VllmRuntimeStatus status = this->status();
        status.last_error =
            "vLLM is resolved from PATH (" + resolved +
            "); update it via your package manager (e.g. pip install -U vllm). "
            "Point vllm_server_path at a managed runtime to let the node update it.";
        set_status(status);
        return status;
    }
    return run_managed_install(/*upgrade=*/true);
}

} // namespace mm
