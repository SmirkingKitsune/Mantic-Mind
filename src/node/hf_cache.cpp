#include "node/hf_cache.hpp"

#include "common/logger.hpp"
#include "common/util.hpp"

#include <algorithm>
#include <filesystem>

#ifndef _WIN32
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace mm {

namespace {
const std::string kCachePrefix = "models--";
} // namespace

std::string hf_repo_id_from_cache_dirname(const std::string& dirname) {
    if (dirname.rfind(kCachePrefix, 0) != 0) return {};
    std::string rest = dirname.substr(kCachePrefix.size());
    if (rest.empty()) return {};
    // HF encodes the repo id's '/' as '--'. Single '-' chars are literal.
    std::string repo;
    for (size_t i = 0; i < rest.size(); ++i) {
        if (rest[i] == '-' && i + 1 < rest.size() && rest[i + 1] == '-') {
            repo.push_back('/');
            ++i;
        } else {
            repo.push_back(rest[i]);
        }
    }
    return repo;
}

std::vector<std::string> scan_hf_cache_models(const std::string& hub_dir) {
    std::vector<std::string> models;
    std::error_code ec;
    if (hub_dir.empty() || !fs::is_directory(hub_dir, ec)) return models;

    for (const auto& entry : fs::directory_iterator(hub_dir, ec)) {
        if (ec) break;
        if (!entry.is_directory(ec)) continue;
        const std::string name = entry.path().filename().string();
        std::string repo = hf_repo_id_from_cache_dirname(name);
        if (!repo.empty()) models.push_back(std::move(repo));
    }
    std::sort(models.begin(), models.end());
    return models;
}

std::string resolve_hf_hub_cache_dir(const std::string& configured,
                                     const std::string& hf_hub_cache_env,
                                     const std::string& hf_home_env,
                                     const std::string& home_dir) {
    if (!configured.empty()) return configured;
    if (!hf_hub_cache_env.empty()) return hf_hub_cache_env;
    if (!hf_home_env.empty())
        return (fs::path(hf_home_env) / "hub").string();
    if (!home_dir.empty())
        return (fs::path(home_dir) / ".cache" / "huggingface" / "hub").string();
    return {};
}

std::vector<std::string> build_hf_download_args(const std::string& repo,
                                                const std::string& cache_dir) {
    std::vector<std::string> args;
    args.push_back("download");
    args.push_back(repo);
    if (!cache_dir.empty()) {
        args.push_back("--cache-dir");
        args.push_back(cache_dir);
    }
    return args;
}

bool hf_prefetch_supported() {
#ifdef _WIN32
    return false;
#else
    return true;
#endif
}

#ifndef _WIN32
namespace {
bool run_blocking(const std::string& exe,
                  const std::vector<std::string>& args,
                  std::string* error_out) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(exe.c_str()));
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    pid_t pid = ::fork();
    if (pid < 0) {
        if (error_out) *error_out = "fork failed";
        return false;
    }
    if (pid == 0) {
        ::execvp(exe.c_str(), argv.data());
        ::_exit(127);
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        if (error_out) *error_out = "waitpid failed";
        return false;
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return true;
    if (error_out) {
        *error_out = "'" + exe + " download' exited with status " +
                     std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    }
    return false;
}
} // namespace
#endif

bool hf_download(const std::string& hf_cli_path,
                 const std::string& repo,
                 const std::string& cache_dir,
                 std::string* error_out) {
    if (!hf_prefetch_supported()) {
        if (error_out)
            *error_out = "HF pre-fetch is only supported on Linux nodes; vLLM "
                         "will download the model at load time instead";
        return false;
    }
    if (!util::is_hf_repo_id(repo)) {
        if (error_out)
            *error_out = "'" + repo + "' is not an HF repo id; nothing to pre-fetch";
        return false;
    }
#ifdef _WIN32
    (void)hf_cli_path; (void)cache_dir;
    return false;
#else
    const auto args = build_hf_download_args(repo, cache_dir);
    MM_INFO("HfCache: pre-fetching {} into {}", repo,
            cache_dir.empty() ? "default cache" : cache_dir);
    const bool ok = run_blocking(hf_cli_path, args, error_out);
    if (!ok && error_out) MM_WARN("HfCache: pre-fetch failed: {}", *error_out);
    return ok;
#endif
}

} // namespace mm
