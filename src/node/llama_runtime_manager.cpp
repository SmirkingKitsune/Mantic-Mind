#include "node/llama_runtime_manager.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"
#include "node/node_state.hpp"

#include <deque>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>

#ifdef _WIN32
#  include <cstdio>
#else
#  include <cstdio>
#  include <sys/wait.h>
#endif

namespace mm {

namespace {

std::string quote_shell_arg(const std::string& s) {
    std::string out = "\"";
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

int run_command_capture_lines(const std::string& cmd,
                              const std::function<void(const std::string&)>& on_line) {
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = ::popen(cmd.c_str(), "r");
#endif
    if (!pipe) return -1;

    char buf[4096];
    while (std::fgets(buf, static_cast<int>(sizeof(buf)), pipe)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        on_line(line);
    }

#ifdef _WIN32
    int rc = _pclose(pipe);
#else
    int rc = ::pclose(pipe);
    if (rc != -1 && WIFEXITED(rc)) rc = WEXITSTATUS(rc);
#endif
    return rc;
}

std::string run_command_capture_stdout(const std::string& cmd, int* out_rc = nullptr) {
    std::string out;
    int rc = run_command_capture_lines(cmd, [&out](const std::string& line) {
        if (!out.empty()) out.push_back('\n');
        out += line;
    });
    if (out_rc) *out_rc = rc;
    return out;
}

std::string parse_head_commit_from_ls_remote(const std::string& s) {
    auto line = mm::util::trim(s);
    if (line.empty()) return {};
    auto tab = line.find('\t');
    if (tab == std::string::npos) {
        auto sp = line.find(' ');
        if (sp == std::string::npos) return line;
        return mm::util::trim(line.substr(0, sp));
    }
    return mm::util::trim(line.substr(0, tab));
}

std::filesystem::path resolve_binary_candidate(const std::filesystem::path& build_dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::vector<fs::path> candidates = {
        build_dir / "bin/llama-server.exe",
        build_dir / "bin/Release/llama-server.exe",
        build_dir / "bin/Debug/llama-server.exe",
        build_dir / "bin/llama-server",
        build_dir / "bin/Release/llama-server",
        build_dir / "bin/Debug/llama-server"
    };
    for (const auto& p : candidates) {
        if (fs::exists(p, ec) && fs::is_regular_file(p, ec)) return p;
    }
    return {};
}

std::filesystem::path resolve_prebuilt_binary_candidate(const std::filesystem::path& repo_dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path prebuilt_root = repo_dir / "prebuilt";
    if (!fs::exists(prebuilt_root, ec) || !fs::is_directory(prebuilt_root, ec)) return {};

    fs::path best;
    fs::file_time_type best_time{};
    bool found = false;
    for (fs::recursive_directory_iterator it(prebuilt_root, ec), end; it != end; it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        if (!it->is_regular_file(ec)) {
            ec.clear();
            continue;
        }
        const auto name = mm::util::to_lower(it->path().filename().string());
        if (name != "llama-server.exe" && name != "llama-server") continue;

        const auto wt = fs::last_write_time(it->path(), ec);
        if (ec) {
            ec.clear();
            continue;
        }
        if (!found || wt > best_time) {
            best = it->path();
            best_time = wt;
            found = true;
        }
    }
    return best;
}

std::filesystem::path parse_reported_binary_path(const std::string& line) {
    constexpr const char* kPrefix = "[llama.cpp] llama-server path:";
    if (!mm::util::starts_with(line, kPrefix)) return {};
    auto path_text = mm::util::trim(line.substr(std::char_traits<char>::length(kPrefix)));
    if (path_text.empty()) return {};
    if (path_text.front() == '(') return {};
    return std::filesystem::path(path_text);
}

std::vector<std::string> read_tail_lines(const std::filesystem::path& path, size_t max_lines) {
    std::vector<std::string> out;
    if (max_lines == 0) return out;

    std::ifstream in(path);
    if (!in) return out;

    std::deque<std::string> ring;
    std::string line;
    while (std::getline(in, line)) {
        ring.push_back(line);
        if (ring.size() > max_lines) ring.pop_front();
    }

    out.assign(ring.begin(), ring.end());
    return out;
}

std::string make_job_id() {
    return std::to_string(mm::util::now_ms()) + "-" + mm::util::generate_uuid().substr(0, 8);
}

} // namespace

LlamaRuntimeManager::LlamaRuntimeManager(NodeState& state, Options opts)
    : state_(state), opts_(std::move(opts)) {
    std::lock_guard<std::mutex> lk(mutex_);
    status_.repo_url = opts_.repo_url;
    status_.install_root = opts_.install_root.string();
    status_.repo_dir = (opts_.install_root / "repo").string();
    status_.build_dir = (opts_.install_root / "repo" / "build").string();
    status_.binary_path = resolve_binary_candidate(status_.build_dir).string();
    if (status_.binary_path.empty()) {
        status_.binary_path = resolve_prebuilt_binary_candidate(status_.repo_dir).string();
    }
    load_metadata_locked();
    refresh_versions_locked(/*force_remote=*/false, nullptr);
    sync_state_locked();
}

LlamaRuntimeManager::~LlamaRuntimeManager() = default;

void LlamaRuntimeManager::set_log_callback(LogCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    log_cb_ = std::move(cb);
}

void LlamaRuntimeManager::set_binary_ready_callback(BinaryReadyCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    binary_ready_cb_ = std::move(cb);
}

std::filesystem::path LlamaRuntimeManager::find_script_path() const {
    namespace fs = std::filesystem;
    std::error_code ec;

    if (!opts_.updater_script.empty() && fs::exists(opts_.updater_script, ec)) {
        return opts_.updater_script;
    }

    const char* env_script = std::getenv("MM_LLAMA_UPDATE_SCRIPT");
    if (env_script && *env_script) {
        fs::path p(env_script);
        if (fs::exists(p, ec)) return p;
    }

#ifdef _WIN32
    const char* name = "update-llama-cpp.ps1";
#else
    const char* name = "update-llama-cpp.sh";
#endif

    fs::path dir = fs::current_path(ec);
    if (ec) return {};
    for (int i = 0; i < 12; ++i) {
        fs::path p = dir / "tools" / name;
        if (fs::exists(p, ec) && fs::is_regular_file(p, ec)) return p;
        if (!dir.has_parent_path()) break;
        auto parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return {};
}

bool LlamaRuntimeManager::load_metadata_locked() {
    std::ifstream in(opts_.metadata_file);
    if (!in) return false;

    try {
        auto j = nlohmann::json::parse(in);
        status_.binary_path       = j.value("binary_path", status_.binary_path);
        status_.installed_commit  = j.value("installed_commit", status_.installed_commit);
        status_.remote_commit     = j.value("remote_commit", status_.remote_commit);
        status_.remote_error      = j.value("remote_error", status_.remote_error);
        status_.remote_checked_ms = j.value("remote_checked_ms", status_.remote_checked_ms);
        status_.update_available  = j.value("update_available", status_.update_available);
        status_.update_reason     = j.value("update_reason", status_.update_reason);
        status_.last_job_id       = j.value("last_job_id", status_.last_job_id);
        status_.last_log_path     = j.value("last_log_path", status_.last_log_path);
        return true;
    } catch (const std::exception& e) {
        MM_WARN("Failed to parse llama runtime metadata '{}': {}",
                opts_.metadata_file.string(), e.what());
        return false;
    }
}

bool LlamaRuntimeManager::persist_metadata_locked() const {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(opts_.metadata_file.parent_path(), ec);

    nlohmann::json j = {
        {"repo_url", status_.repo_url},
        {"install_root", status_.install_root},
        {"repo_dir", status_.repo_dir},
        {"build_dir", status_.build_dir},
        {"binary_path", status_.binary_path},
        {"installed_commit", status_.installed_commit},
        {"remote_commit", status_.remote_commit},
        {"remote_error", status_.remote_error},
        {"remote_checked_ms", status_.remote_checked_ms},
        {"update_available", status_.update_available},
        {"update_reason", status_.update_reason},
        {"last_job_id", status_.last_job_id},
        {"last_log_path", status_.last_log_path}
    };

    std::ofstream out(opts_.metadata_file, std::ios::out | std::ios::trunc);
    if (!out) return false;
    out << j.dump(2) << '\n';
    return true;
}

void LlamaRuntimeManager::sync_state_locked() const {
    LlamaRuntimeSummary s;
    s.install_root = status_.install_root;
    s.repo_dir = status_.repo_dir;
    s.build_dir = status_.build_dir;
    s.binary_path = status_.binary_path;
    s.installed_commit = status_.installed_commit;
    s.remote_commit = status_.remote_commit;
    s.remote_error = status_.remote_error;
    s.remote_checked_ms = status_.remote_checked_ms;
    s.update_available = status_.update_available;
    s.update_reason = status_.update_reason;
    s.last_log_path = status_.last_log_path;
    state_.set_llama_runtime_summary(s);
}

bool LlamaRuntimeManager::refresh_versions_locked(bool force_remote, std::string* out_message) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto repo_dir = fs::path(status_.repo_dir);
    const bool has_repo = fs::exists(repo_dir / ".git", ec);

    status_.installed_commit.clear();
    if (has_repo) {
        int rc = 0;
        std::string cmd = "git -C " + quote_shell_arg(repo_dir.string()) + " rev-parse --short HEAD";
        auto out = run_command_capture_stdout(cmd, &rc);
        if (rc == 0) status_.installed_commit = mm::util::trim(out);
    }

    const int64_t now = mm::util::now_ms();
    const bool stale_remote = (now - status_.remote_checked_ms) > (10 * 60 * 1000);
    if (force_remote || stale_remote || status_.remote_commit.empty()) {
        int rc = 0;
        const std::string cmd = "git ls-remote " + quote_shell_arg(status_.repo_url) + " HEAD";
        auto out = run_command_capture_stdout(cmd, &rc);
        if (rc == 0) {
            status_.remote_commit = parse_head_commit_from_ls_remote(out);
            status_.remote_error.clear();
            status_.remote_checked_ms = now;
        } else {
            status_.remote_error = "git ls-remote failed (exit code " + std::to_string(rc) + ")";
            status_.remote_checked_ms = now;
        }
    }

    if (status_.installed_commit.empty()) {
        status_.update_available = true;
        status_.update_reason = "not_installed";
    } else if (!status_.remote_error.empty() || status_.remote_commit.empty()) {
        status_.update_available = false;
        status_.update_reason = "check_failed";
    } else if (status_.installed_commit != status_.remote_commit) {
        status_.update_available = true;
        status_.update_reason = "remote_ahead";
    } else {
        status_.update_available = false;
        status_.update_reason = "up_to_date";
    }

    if (out_message) {
        if (status_.update_reason == "remote_ahead") {
            *out_message = "update available (" + status_.installed_commit + " -> "
                + status_.remote_commit + ")";
        } else if (status_.update_reason == "up_to_date") {
            *out_message = "llama.cpp is up to date (" + status_.installed_commit + ")";
        } else if (status_.update_reason == "not_installed") {
            *out_message = "llama.cpp is not installed";
        } else {
            *out_message = "update check failed: "
                + (status_.remote_error.empty() ? "unknown error" : status_.remote_error);
        }
    }

    persist_metadata_locked();
    sync_state_locked();
    return status_.update_reason != "check_failed";
}

bool LlamaRuntimeManager::check_update(std::string* out_message, bool force_remote) {
    std::lock_guard<std::mutex> lk(mutex_);
    return refresh_versions_locked(force_remote, out_message);
}

bool LlamaRuntimeManager::start_update(bool build,
                                       bool force,
                                       std::string* out_job_id,
                                       std::string* out_message) {
    std::string job_id;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (status_.running) {
            if (out_message) *out_message = "update already running";
            return false;
        }

        job_id = make_job_id();
        status_.running = true;
        status_.status = "running";
        status_.message = "queued";
        status_.started_ms = mm::util::now_ms();
        status_.finished_ms = 0;
        status_.last_job_id = job_id;
        status_.last_log_tail.clear();
        persist_metadata_locked();
        sync_state_locked();
    }

    state_.start_llama_update("Starting llama runtime update...");

    if (out_job_id) *out_job_id = job_id;
    if (out_message) *out_message = "accepted";

    std::thread(&LlamaRuntimeManager::run_update_worker, this, job_id, build, force).detach();
    return true;
}

void LlamaRuntimeManager::run_update_worker(std::string job_id, bool build, bool force) {
    namespace fs = std::filesystem;

    std::filesystem::path script_path;
    std::filesystem::path log_path;
    std::string repo_dir;
    std::string repo_url;
    std::filesystem::path updater_reported_binary;
    LogCallback log_cb;
    BinaryReadyCallback binary_ready_cb;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        script_path = find_script_path();
        repo_dir = status_.repo_dir;
        repo_url = status_.repo_url;
        log_path = opts_.log_dir / ("update-" + job_id + ".log");
        job_logs_[job_id] = log_path;
        status_.last_log_path = log_path.string();
        log_cb = log_cb_;
        binary_ready_cb = binary_ready_cb_;
        persist_metadata_locked();
        sync_state_locked();
    }

    std::error_code ec;
    fs::create_directories(opts_.log_dir, ec);
    std::ofstream log_out(log_path, std::ios::out | std::ios::trunc);

    auto emit_line = [&](const std::string& line) {
        if (log_out) {
            log_out << line << '\n';
            log_out.flush();
        }
        auto reported = parse_reported_binary_path(line);
        if (!reported.empty()) {
            updater_reported_binary = reported;
        }
        if (log_cb) log_cb(line);
    };

    if (script_path.empty()) {
        const std::string msg = "Updater script not found; set MM_LLAMA_UPDATE_SCRIPT";
        emit_line("[llama.runtime] " + msg);
        std::lock_guard<std::mutex> lk(mutex_);
        status_.running = false;
        status_.status = "failed";
        status_.message = msg;
        status_.finished_ms = mm::util::now_ms();
        status_.last_log_tail = read_tail_lines(log_path, 120);
        persist_metadata_locked();
        sync_state_locked();
        state_.finish_llama_update(false, msg + " (log: " + log_path.string() + ")");
        return;
    }

    std::string check_message;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        refresh_versions_locked(/*force_remote=*/false, &check_message);
    }

    if (!force) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!status_.update_available && status_.update_reason == "up_to_date") {
            const std::string msg = "llama.cpp already up to date";
            emit_line("[llama.runtime] " + msg);
            status_.running = false;
            status_.status = "succeeded";
            status_.message = msg;
            status_.finished_ms = mm::util::now_ms();
            status_.last_log_tail = read_tail_lines(log_path, 120);
            persist_metadata_locked();
            sync_state_locked();
            state_.finish_llama_update(true, msg + " (log: " + log_path.string() + ")");
            return;
        }
    }

    std::string cmd;
#ifdef _WIN32
    cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File "
        + quote_shell_arg(script_path.string())
        + " -RepoDir " + quote_shell_arg(repo_dir)
        + " -RepoUrl " + quote_shell_arg(repo_url);
    if (build) cmd += " -Build";
#else
    cmd = "/bin/bash " + quote_shell_arg(script_path.string())
        + " --repo-dir " + quote_shell_arg(repo_dir)
        + " --repo-url " + quote_shell_arg(repo_url);
    if (build) cmd += " --build";
#endif

    emit_line("[llama.runtime] command: " + cmd);
    state_.set_llama_update_message("Running updater (log: " + log_path.string() + ")");

    int rc = run_command_capture_lines(cmd, emit_line);

    std::string final_message;
    bool success = false;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (rc == 0) {
            std::error_code ec2;
            std::filesystem::path bin;
            if (!updater_reported_binary.empty()
                && std::filesystem::exists(updater_reported_binary, ec2)
                && std::filesystem::is_regular_file(updater_reported_binary, ec2)) {
                bin = updater_reported_binary;
            }
            if (bin.empty()) {
                bin = resolve_binary_candidate(status_.build_dir);
            }
            if (bin.empty()) {
                bin = resolve_prebuilt_binary_candidate(status_.repo_dir);
            }
            if (!bin.empty()) {
                status_.binary_path = bin.string();
                if (binary_ready_cb) {
                    binary_ready_cb(status_.binary_path);
                }
                success = true;
                final_message = "llama.cpp update completed; binary path: " + status_.binary_path;
            } else {
                final_message = "update completed but llama-server binary was not found";
            }
        } else {
            final_message = "llama.cpp update failed (exit code " + std::to_string(rc) + ")";
        }

        refresh_versions_locked(/*force_remote=*/false, nullptr);
        status_.running = false;
        status_.status = success ? "succeeded" : "failed";
        status_.message = final_message;
        status_.finished_ms = mm::util::now_ms();
        status_.last_log_tail = read_tail_lines(log_path, 120);
        persist_metadata_locked();
        sync_state_locked();
    }

    state_.finish_llama_update(success, final_message + " (log: " + log_path.string() + ")");
}

LlamaRuntimeStatus LlamaRuntimeManager::get_status(size_t log_tail_lines) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto out = status_;
    if (!out.last_log_path.empty()) {
        out.last_log_tail = read_tail_lines(out.last_log_path, log_tail_lines);
    } else {
        out.last_log_tail.clear();
    }
    return out;
}

LlamaRuntimeLogChunk LlamaRuntimeManager::read_job_log(const std::string& job_id,
                                                       size_t offset,
                                                       size_t limit) const {
    LlamaRuntimeLogChunk chunk;
    chunk.job_id = job_id;
    chunk.offset = offset;
    if (limit == 0) return chunk;

    std::filesystem::path path;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = job_logs_.find(job_id);
        if (it != job_logs_.end()) {
            path = it->second;
        } else if (job_id == status_.last_job_id && !status_.last_log_path.empty()) {
            path = status_.last_log_path;
        }
    }

    if (path.empty()) return chunk;

    std::ifstream in(path);
    if (!in) return chunk;

    chunk.found = true;
    chunk.log_path = path.string();

    std::string line;
    size_t idx = 0;
    while (std::getline(in, line)) {
        if (idx >= offset && chunk.lines.size() < limit) {
            chunk.lines.push_back(line);
        }
        ++idx;
    }

    chunk.next_offset = offset + chunk.lines.size();
    return chunk;
}

} // namespace mm
