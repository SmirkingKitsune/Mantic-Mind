#include "node/llama_cpp_provisioner.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"
#include "node/llama_runtime.hpp"
#include "node/vllm_provisioner.hpp"  // parse_vllm_install_fraction (shared line parser)
#include "node/vllm_runtime.hpp"      // current_vllm_platform / current_vllm_arch

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

std::string release_arch(const std::string& arch) {
    const std::string normalized = util::to_lower(util::trim(arch));
    if (normalized == "x86_64" || normalized == "amd64" || normalized == "x64")
        return "x64";
    if (normalized == "aarch64" || normalized == "arm64") return "arm64";
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
        if (cfg.install_method != "auto" && cfg.install_method != method)
            return std::nullopt;
        if (cfg.accelerator_explicit && cfg.accelerator != accelerator)
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
        status.version = json.value("version", cfg.version);
        status.managed = true;
        status.executable_path = normalized.string();
        status.accelerator = accelerator;
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
    const std::string release_arch =
        (cfg.arch == "x86_64" || cfg.arch == "amd64" || cfg.arch == "x64")
            ? "x64"
            : (cfg.arch == "aarch64" ? "arm64" : cfg.arch);
    const std::string tag_filter =
        cfg.version == "latest" ? std::string("releases/latest")
                                : "releases/tags/" + cfg.version;
    const std::string api =
        "https://api.github.com/repos/ggml-org/llama.cpp/" + tag_filter;
    std::vector<LlamaInstallStep> plan;
    if (cfg.platform == "windows") {
        std::ostringstream ps;
        ps << "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; ";
        ps << "$api='" << api << "'; $arch='" << release_arch
           << "'; $backend='" << cfg.accelerator << "'; ";
        ps << "$rel='" << rel.string() << "'; $stage='" << stage.string()
           << "'; $next='" << next_bin.string() << "'; $bin='" << bin.string() << "'; ";
        ps << "New-Item -ItemType Directory -Force -Path $rel | Out-Null; ";
        ps << "$r=Invoke-RestMethod -Uri $api -Headers @{'User-Agent'='mantic-mind'}; $assets=@($r.assets); ";
        ps << "$archRx=[regex]::Escape($arch); ";
        ps << "$base=@($assets | Where-Object {$_.name -match ('^llama-.*-bin-win-cpu-'+$archRx+'\\.zip$')})[0]; ";
        ps << "if ($null -eq $base) { throw ('No llama.cpp Windows base release for '+$arch) }; $selected=@($base); ";
        ps << "if ($backend -eq 'cuda') { ";
        ps << "$gpu=@(); foreach($a in $assets) { if($a.name -match ('^llama-.*-bin-win-cuda-([0-9.]+)-'+$archRx+'\\.zip$')) {$gpu += [pscustomobject]@{Asset=$a; Version=[version]$Matches[1]}} }; ";
        ps << "$supported=$null; try {$txt=(& nvidia-smi 2>$null) -join \"`n\"; $m=[regex]::Match($txt,'CUDA Version:\\s*([0-9.]+)'); if($m.Success){$supported=[version]$m.Groups[1].Value}} catch {}; ";
        ps << "if($null -ne $supported){$choice=$gpu | Where-Object {$_.Version -le $supported} | Sort-Object Version -Descending | Select-Object -First 1} else {$choice=$gpu | Sort-Object Version | Select-Object -First 1}; ";
        ps << "if($null -eq $choice){throw ('No compatible llama.cpp Windows CUDA release for '+$arch)}; $selected += $choice.Asset; ";
        ps << "$cudartName=('cudart-llama-bin-win-cuda-'+$choice.Version+'-'+$arch+'.zip'); $cudart=$assets | Where-Object {$_.name -eq $cudartName} | Select-Object -First 1; if($null -ne $cudart){$selected += $cudart}; ";
        ps << "} elseif ($backend -eq 'vulkan') {$addon=$assets | Where-Object {$_.name -match ('^llama-.*-bin-win-vulkan-'+$archRx+'\\.zip$')} | Select-Object -First 1; if($null -eq $addon){throw 'No matching llama.cpp Windows Vulkan release'}; $selected += $addon; ";
        ps << "} elseif ($backend -eq 'rocm') {$addon=$assets | Where-Object {$_.name -match ('^llama-.*-bin-win-(hip|rocm).*-'+$archRx+'\\.zip$')} | Select-Object -First 1; if($null -eq $addon){throw 'No matching llama.cpp Windows ROCm/HIP release'}; $selected += $addon; ";
        ps << "} elseif ($backend -ne 'cpu') {throw ('No matching llama.cpp Windows release backend: '+$backend)}; ";
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

api, platform, arch, backend, rel_arg = sys.argv[1:]
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
        choice = choose_cuda(candidates)
        candidates = [choice] if choice else []
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
                        {"python3", "-c", py, api, cfg.platform, release_arch,
                         cfg.accelerator, rel.string()}, root, false});
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

    std::vector<LlamaInstallStep> plan;
    if (cfg.platform == "windows") {
        std::ostringstream ps;
        ps << "$ErrorActionPreference='Stop'; $required=@('git','cmake'); ";
        ps << "$missing=@($required | Where-Object {-not (Get-Command $_ -ErrorAction SilentlyContinue)}); ";
        // CMake can discover Visual Studio through the registry even when cl.exe
        // is not on this PowerShell process's PATH, so let configure validate it.
        if (cfg.accelerator == "cuda") {
            ps << "if(-not (Get-Command nvcc -ErrorAction SilentlyContinue)){ "
                  "$missing += 'nvcc (CUDA Toolkit; the display driver alone is insufficient)' }; ";
        } else if (cfg.accelerator == "rocm") {
            ps << "if(-not (Get-Command hipcc -ErrorAction SilentlyContinue)){ "
                  "$missing += 'hipcc (ROCm/HIP SDK)' }; ";
        }
        ps << "if($missing.Count){throw ('Missing llama.cpp source-build prerequisites: '+"
              "($missing -join ', '))}; cmake --version | Select-Object -First 1; ";
        if (cfg.accelerator == "cuda") ps << "nvcc --version | Select-Object -Last 1; ";
        plan.push_back({"Checking source-build prerequisites",
                        {"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                         "-Command", ps.str()}, root, false});
    } else {
        const std::string preflight = R"SH(set -eu
backend="$1"
missing=""
for tool in git cmake; do
  command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
cxx="${CXX:-c++}"
command -v "$cxx" >/dev/null 2>&1 || missing="$missing C++-compiler"
if [ "$backend" = "cuda" ]; then
  command -v nvcc >/dev/null 2>&1 || missing="$missing nvcc-CUDA-Toolkit"
elif [ "$backend" = "rocm" ]; then
  command -v hipcc >/dev/null 2>&1 || missing="$missing hipcc-ROCm-SDK"
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
  nvcc --version | tail -n 1
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
                         cfg.accelerator}, root, false});
    }
    if (!fs::exists(src / ".git"))
        plan.push_back({"Cloning llama.cpp source",
                        {"git", "clone", kLlamaCppRepoUrl, src.string()}, root, false});
    plan.push_back({"Fetching tags", {"git", "fetch", "--tags", "--all", "--prune"}, src, false});
    plan.push_back({"Checking out " + ref, {"git", "checkout", ref}, src, false});
    if (cfg.version == "latest")
        plan.push_back({"Updating source", {"git", "pull", "--ff-only"}, src, true});

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
                asset_names, "^llama-.*-bin-win-cuda-[0-9.]+-" + arch_rx + "\\.zip$"))
            available.insert("cuda");
        if (base && any_asset_matches(
                asset_names, "^llama-.*-bin-win-vulkan-" + arch_rx + "\\.zip$"))
            available.insert("vulkan");
        if (base && any_asset_matches(
                asset_names, "^llama-.*-bin-win-(hip|rocm).*-" + arch_rx + "\\.zip$"))
            available.insert("rocm");
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
    } else if (cfg.platform == "macos") {
        if (any_asset_matches(asset_names,
                              "^llama-.*-bin-macos-" + arch_rx +
                                  "(\\.tar\\.gz|\\.zip)$")) {
            available.insert("metal");
            available.insert("cpu");
        }
    }

    const std::vector<std::string> preference = cfg.platform == "linux"
        ? std::vector<std::string>{"vulkan", "cpu", "rocm", "cuda"}
        : (cfg.platform == "windows"
            ? std::vector<std::string>{"cuda", "vulkan", "cpu", "rocm"}
            : std::vector<std::string>{"metal", "cpu"});
    std::vector<std::string> out;
    for (const auto& accelerator : preference)
        if (available.count(accelerator)) out.push_back(accelerator);
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
    if (!runner_.fetch_release_assets)
        runner_.fetch_release_assets = default_fetch_release_assets;
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

    // Honor the last successful managed selection before scanning generic
    // candidates. This keeps a user-selected Vulkan/CPU release active across
    // restarts instead of mislabeling it as the auto-detected CUDA target.
    const auto active = read_active_runtime(cfg_);
    if (active) {
        const std::string version = capture_version(active->executable_path);
        auto restored = *active;
        if (!version.empty()) restored.version = version;
        set_status(restored);
        return restored;
    }
    const bool ignored_active_marker =
        fs::exists(fs::path(cfg_.provision_dir) / "active-runtime.json");

    // An already-built managed runtime is usable regardless of auto-provision.
    for (const auto& cand : managed_executable_candidates(cfg_)) {
        // A marker rejected because explicit config superseded its method or
        // accelerator must not be rediscovered as an unlabeled generic release.
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

LlamaRuntimeStatus LlamaCppProvisioner::run_managed_install(
    bool upgrade,
    const std::string& accelerator_override) {
    LlamaProvisionConfig install_cfg = cfg_;
    const LlamaRuntimeStatus prior = this->status();
    if (upgrade && !prior.latest_version.empty())
        install_cfg.version = prior.latest_version; // build/download the assessed release, not moving master
    if (upgrade && accelerator_override.empty() && !cfg_.accelerator_explicit &&
        !prior.accelerator.empty())
        install_cfg.accelerator = prior.accelerator;
    if (!accelerator_override.empty()) {
        install_cfg.accelerator = util::to_lower(util::trim(accelerator_override));
        install_cfg.install_method = "release"; // alternative buttons only represent published assets
    }
    if (install_cfg.install_method == "auto" && install_cfg.version == "latest" &&
        runner_.fetch_latest) {
        const std::string release_tag = util::trim(runner_.fetch_latest(install_cfg));
        if (is_llama_release_tag(release_tag)) install_cfg.version = release_tag;
    }
    LlamaRuntimeStatus status = make_base_status();
    status.accelerator = install_cfg.accelerator;
    status.version = install_cfg.version;

    const fs::path root(install_cfg.provision_dir);
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

        std::string attempt_error;
        if (plan.empty()) attempt_error = "install plan is empty";
        for (int i = 0; i < total && attempt_error.empty(); ++i) {
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

            std::deque<std::string> output_tail;
            auto on_line = [&](const std::string& line, bool is_stderr) {
                if (log_sink_) log_sink_(line, is_stderr);
                const double f = parse_vllm_install_fraction(line);
                if (f >= 0.0) prog.fraction = f;
                prog.last_line = line;
                const std::string trimmed = util::trim(line);
                if (!trimmed.empty()) {
                    if (output_tail.size() >= 12) output_tail.pop_front();
                    output_tail.push_back(trimmed);
                }
                emit_progress(prog);
            };

            std::string step_err;
            const fs::path cwd = step.cwd.empty() ? root : step.cwd;
            const int rc = runner_.run(step.argv, cwd, on_line, cancel_check_, &step_err);
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
            }
        }

        if (!found.empty()) {
            const std::string version = capture_version(found);
            status.status = "ready";
            status.managed = true;
            status.method = attempt.config.install_method;
            status.executable_path = found;
            if (!version.empty()) status.version = version;
            status.latest_version.clear();
            status.update_available = false;
            status.update_action.clear();
            status.update_release_available = false;
            status.update_release_alternatives.clear();
            status.update_warning.clear();
            status.last_error.clear();
            write_active_runtime(attempt.config, status);
            set_status(status);
            finish_progress();
            MM_INFO("llama.cpp runtime {} from {}: {}",
                    upgrade ? "updated" : "ready", attempt.label,
                    status.executable_path);
            return status;
        }

        attempt_errors.push_back(attempt.label + ": " + attempt_error);
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
    set_status(status);
    finish_progress();
    return status;
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
    // `llama-server --version` commonly returns a sentence containing the tag,
    // so treat a contained tag as current instead of comparing whole strings.
    status.update_available = !latest.empty() && !installed.empty()
        && !installed_version_matches_tag(installed, latest);
    if (status.update_available) {
        MM_INFO("llama.cpp update available: {} -> {}", installed, latest);
        const auto assets = runner_.fetch_release_assets
            ? runner_.fetch_release_assets(cfg_, latest)
            : std::vector<std::string>{};
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
    LlamaRuntimeStatus current = this->status();
    if (accelerator_override.empty() && current.update_action == "unavailable") {
        current.last_error = current.update_warning;
        set_status(current);
        return current;
    }
    const std::string requested = util::to_lower(util::trim(accelerator_override));
    if (!requested.empty()) {
        const std::set<std::string> allowed{"cuda", "rocm", "vulkan", "metal", "cpu"};
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

} // namespace mm
