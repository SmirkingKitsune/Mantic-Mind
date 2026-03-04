#include "node/llama_server_process.hpp"
#include "common/logger.hpp"
#include "common/util.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <array>
#include <cctype>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#  include <Windows.h>
#else
#  include <cerrno>
#  include <csignal>
#  include <cstring>
#  include <fcntl.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace mm {

// ── Impl ──────────────────────────────────────────────────────────────────────
struct LlamaServerProcess::Impl {
    std::atomic<bool> pipe_running{false};
    std::thread       stdout_reader;
    std::thread       stderr_reader;
    std::mutex        cb_mutex;
    LogCallback       log_cb;

#ifdef _WIN32
    HANDLE proc_handle  = INVALID_HANDLE_VALUE;
    HANDLE stdout_read  = INVALID_HANDLE_VALUE;
    HANDLE stderr_read  = INVALID_HANDLE_VALUE;
    HANDLE stdout_write = INVALID_HANDLE_VALUE; // child write-end (closed after spawn)
    HANDLE stderr_write = INVALID_HANDLE_VALUE;
#else
    pid_t pid      = -1;
    int   stdout_fd = -1;
    int   stderr_fd = -1;
#endif

    void call_log(const std::string& line, bool is_stderr) {
        std::lock_guard<std::mutex> lk(cb_mutex);
        if (log_cb) log_cb(line, is_stderr);
    }

    void join_readers() {
        if (stdout_reader.joinable()) stdout_reader.join();
        if (stderr_reader.joinable()) stderr_reader.join();
    }

    void close_read_pipes() {
#ifdef _WIN32
        if (stdout_read != INVALID_HANDLE_VALUE) {
            CloseHandle(stdout_read); stdout_read = INVALID_HANDLE_VALUE;
        }
        if (stderr_read != INVALID_HANDLE_VALUE) {
            CloseHandle(stderr_read); stderr_read = INVALID_HANDLE_VALUE;
        }
#else
        if (stdout_fd != -1) { ::close(stdout_fd); stdout_fd = -1; }
        if (stderr_fd != -1) { ::close(stderr_fd); stderr_fd = -1; }
#endif
    }
};

// ── Arg builder ───────────────────────────────────────────────────────────────
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

std::string normalize_model_path_for_runtime(std::string p) {
    p = strip_wrapping_quotes(std::move(p));
    if (p.empty()) return p;

#ifdef _WIN32
    // Accept WSL-style mount paths when node runtime is Windows:
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
    // Best-effort support for Windows-style paths when node runtime is Linux:
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

#ifdef _WIN32
// Quote according to Windows command-line parsing rules.
std::string quote_windows_arg(const std::string& arg) {
    if (arg.empty()) return "\"\"";
    if (arg.find_first_of(" \t\n\v\"") == std::string::npos) return arg;

    std::string out;
    out.reserve(arg.size() + 8);
    out.push_back('"');

    size_t backslashes = 0;
    for (char ch : arg) {
        if (ch == '\\') {
            ++backslashes;
            continue;
        }
        if (ch == '"') {
            out.append(backslashes * 2 + 1, '\\');
            out.push_back('"');
            backslashes = 0;
            continue;
        }
        if (backslashes > 0) {
            out.append(backslashes, '\\');
            backslashes = 0;
        }
        out.push_back(ch);
    }

    if (backslashes > 0) {
        out.append(backslashes * 2, '\\');
    }
    out.push_back('"');
    return out;
}

bool is_windows_path_like(const std::string& s) {
    return s.find('\\') != std::string::npos
        || s.find('/')  != std::string::npos
        || s.find(':')  != std::string::npos;
}

bool windows_file_exists(const std::string& p) {
    if (p.empty()) return false;
    DWORD attrs = GetFileAttributesA(p.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
}

std::string resolve_windows_executable(const std::string& exe_name) {
    if (exe_name.empty()) return exe_name;
    if (is_windows_path_like(exe_name)) return exe_name;

    std::array<char, 32768> full{};
    DWORD n = SearchPathA(nullptr, exe_name.c_str(), nullptr,
                          static_cast<DWORD>(full.size()), full.data(), nullptr);
    if (n > 0 && n < full.size()) return std::string(full.data());

    n = SearchPathA(nullptr, exe_name.c_str(), ".exe",
                    static_cast<DWORD>(full.size()), full.data(), nullptr);
    if (n > 0 && n < full.size()) return std::string(full.data());

    return exe_name;
}
#endif

std::vector<std::string> build_args(const std::string& model_path,
                                    const LlamaSettings& s, uint16_t port) {
    std::vector<std::string> args;
    args.push_back("--model");       args.push_back(strip_wrapping_quotes(model_path));
    args.push_back("--port");        args.push_back(std::to_string(port));
    args.push_back("--ctx-size");    args.push_back(std::to_string(s.ctx_size));
    args.push_back("--gpu-layers"); args.push_back(std::to_string(s.n_gpu_layers));
    if (s.n_threads > 0) {
        args.push_back("--threads"); args.push_back(std::to_string(s.n_threads));
    }
    if (s.flash_attn) args.push_back("--flash-attn");
    for (auto& a : s.extra_args) args.push_back(a);
    return args;
}

} // namespace

// ── Construction ──────────────────────────────────────────────────────────────
LlamaServerProcess::LlamaServerProcess(std::string path)
    : impl_(std::make_unique<Impl>())
    , llama_server_path_(std::move(path))
{}

LlamaServerProcess::~LlamaServerProcess() { stop(); }

void LlamaServerProcess::set_log_callback(LogCallback cb) {
    std::lock_guard<std::mutex> lk(impl_->cb_mutex);
    impl_->log_cb = std::move(cb);
}

// ── start ─────────────────────────────────────────────────────────────────────
bool LlamaServerProcess::start(const std::string& model_path,
                                const LlamaSettings& settings,
                                uint16_t port) {
    if (state_ != ProcessState::Stopped) stop();

    port_  = port;
    state_ = ProcessState::Starting;
    last_error_.clear();

    std::string exe_path = strip_wrapping_quotes(llama_server_path_);
    if (exe_path.empty()) {
        MM_ERROR("llama_server_path is empty");
        last_error_ = "llama_server_path is empty";
        impl_->call_log(last_error_, true);
        state_ = ProcessState::Error;
        return false;
    }

    const std::string runtime_model_path = normalize_model_path_for_runtime(model_path);
    if (runtime_model_path != strip_wrapping_quotes(model_path)) {
        MM_INFO("Normalized model path for runtime: '{}' -> '{}'",
                strip_wrapping_quotes(model_path), runtime_model_path);
    }

    auto args = build_args(runtime_model_path, settings, port);

#ifdef _WIN32
    std::string resolved_exe_path = resolve_windows_executable(exe_path);
    const bool exe_path_like = is_windows_path_like(exe_path);
    if (exe_path_like && !windows_file_exists(exe_path)) {
        MM_ERROR("llama-server executable not found at path: {}", exe_path);
        last_error_ = "llama-server executable not found at path: " + exe_path;
        impl_->call_log(last_error_, true);
        state_ = ProcessState::Error;
        return false;
    }
    if (!exe_path_like && !windows_file_exists(resolved_exe_path)) {
        MM_ERROR("llama-server executable '{}' not found in PATH", exe_path);
        last_error_ = "llama-server executable not found in PATH: " + exe_path
                    + " (set llama_server_path or MM_LLAMA_PATH)";
        impl_->call_log(last_error_, true);
        state_ = ProcessState::Error;
        return false;
    }

    // ── Windows: CreateProcess with anonymous pipes ───────────────────────────
    SECURITY_ATTRIBUTES sa{};
    sa.nLength              = sizeof(sa);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    // stdout pipe
    if (!CreatePipe(&impl_->stdout_read, &impl_->stdout_write, &sa, 0)) {
        MM_ERROR("CreatePipe(stdout) failed: {}", static_cast<unsigned>(GetLastError()));
        last_error_ = "CreatePipe(stdout) failed";
        impl_->call_log(last_error_, true);
        state_ = ProcessState::Error;
        return false;
    }
    SetHandleInformation(impl_->stdout_read, HANDLE_FLAG_INHERIT, 0);

    // stderr pipe
    if (!CreatePipe(&impl_->stderr_read, &impl_->stderr_write, &sa, 0)) {
        MM_ERROR("CreatePipe(stderr) failed: {}", static_cast<unsigned>(GetLastError()));
        last_error_ = "CreatePipe(stderr) failed";
        impl_->call_log(last_error_, true);
        CloseHandle(impl_->stdout_read);  impl_->stdout_read  = INVALID_HANDLE_VALUE;
        CloseHandle(impl_->stdout_write); impl_->stdout_write = INVALID_HANDLE_VALUE;
        state_ = ProcessState::Error;
        return false;
    }
    SetHandleInformation(impl_->stderr_read, HANDLE_FLAG_INHERIT, 0);

    // Build quoted command line
    std::ostringstream cmd;
    cmd << quote_windows_arg(resolved_exe_path);
    for (auto& a : args) {
        cmd << " " << quote_windows_arg(a);
    }
    std::string cmd_str = cmd.str();

    // STARTF_USESTDHANDLES requires valid inheritable handles for all std streams.
    // Use current stdin when available; otherwise fall back to NUL.
    HANDLE base_stdin = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE nul_stdin  = INVALID_HANDLE_VALUE;
    if (base_stdin == nullptr || base_stdin == INVALID_HANDLE_VALUE) {
        nul_stdin = CreateFileA(
            "NUL",
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            &sa, // inheritable
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        base_stdin = nul_stdin;
    }
    if (base_stdin == nullptr || base_stdin == INVALID_HANDLE_VALUE) {
        MM_ERROR("CreateProcess prep failed: could not get valid stdin handle");
        last_error_ = "invalid stdin handle for CreateProcess";
        impl_->call_log(last_error_, true);
        CloseHandle(impl_->stdout_write); impl_->stdout_write = INVALID_HANDLE_VALUE;
        CloseHandle(impl_->stderr_write); impl_->stderr_write = INVALID_HANDLE_VALUE;
        impl_->close_read_pipes();
        state_ = ProcessState::Error;
        return false;
    }

    HANDLE child_stdin = INVALID_HANDLE_VALUE;
    if (!DuplicateHandle(GetCurrentProcess(),
                         base_stdin,
                         GetCurrentProcess(),
                         &child_stdin,
                         0,
                         TRUE, // inheritable in child
                         DUPLICATE_SAME_ACCESS)) {
        auto err = static_cast<unsigned>(GetLastError());
        MM_ERROR("CreateProcess prep failed: DuplicateHandle(stdin) failed: {}", err);
        last_error_ = "DuplicateHandle(stdin) failed: " + std::to_string(err);
        impl_->call_log(last_error_, true);
        if (nul_stdin != INVALID_HANDLE_VALUE) CloseHandle(nul_stdin);
        CloseHandle(impl_->stdout_write); impl_->stdout_write = INVALID_HANDLE_VALUE;
        CloseHandle(impl_->stderr_write); impl_->stderr_write = INVALID_HANDLE_VALUE;
        impl_->close_read_pipes();
        state_ = ProcessState::Error;
        return false;
    }
    if (nul_stdin != INVALID_HANDLE_VALUE) {
        CloseHandle(nul_stdin);
    }
    auto make_cmd_buf = [&]() {
        std::vector<char> buf(cmd_str.begin(), cmd_str.end());
        buf.push_back('\0');
        return buf;
    };

    const bool use_application_name =
        resolved_exe_path.find('\\') != std::string::npos ||
        resolved_exe_path.find('/')  != std::string::npos ||
        resolved_exe_path.find(':')  != std::string::npos;
    const char* app_name = use_application_name ? resolved_exe_path.c_str() : nullptr;

    auto spawn = [&](bool use_stdio,
                     bool inherit_handles,
                     HANDLE stdin_handle,
                     PROCESS_INFORMATION& out_pi) -> std::pair<BOOL, unsigned> {
        STARTUPINFOA si{};
        si.cb = sizeof(si);
        if (use_stdio) {
            si.dwFlags    = STARTF_USESTDHANDLES;
            si.hStdOutput = impl_->stdout_write;
            si.hStdError  = impl_->stderr_write;
            si.hStdInput  = stdin_handle;
        }
        auto cmd_buf = make_cmd_buf();
        BOOL ok = CreateProcessA(
            app_name, cmd_buf.data(),
            nullptr, nullptr, inherit_handles ? TRUE : FALSE,
            CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
            nullptr, nullptr,
            &si, &out_pi);
        unsigned err = ok ? 0u : static_cast<unsigned>(GetLastError());
        return {ok, err};
    };

    PROCESS_INFORMATION pi{};
    auto [ok, err] = spawn(/*use_stdio=*/true,
                           /*inherit_handles=*/true,
                           child_stdin,
                           pi);

    if (child_stdin != INVALID_HANDLE_VALUE) {
        CloseHandle(child_stdin);
    }

    bool used_stdio = true;
    if (!ok && err == static_cast<unsigned>(ERROR_INVALID_PARAMETER)) {
        MM_WARN("CreateProcess with redirected stdio failed (87); retrying without stdio redirection");
        impl_->call_log("CreateProcess stdio redirection failed (87); retrying without live output capture", true);

        if (impl_->stdout_write != INVALID_HANDLE_VALUE) {
            CloseHandle(impl_->stdout_write);
            impl_->stdout_write = INVALID_HANDLE_VALUE;
        }
        if (impl_->stderr_write != INVALID_HANDLE_VALUE) {
            CloseHandle(impl_->stderr_write);
            impl_->stderr_write = INVALID_HANDLE_VALUE;
        }
        impl_->close_read_pipes();

        auto [ok2, err2] = spawn(/*use_stdio=*/false,
                                 /*inherit_handles=*/false,
                                 INVALID_HANDLE_VALUE,
                                 pi);
        if (ok2) {
            ok = ok2;
            used_stdio = false;
        } else {
            err = err2;
        }
    }

    // Close write ends in parent right after spawn (or retry failure).
    if (impl_->stdout_write != INVALID_HANDLE_VALUE) {
        CloseHandle(impl_->stdout_write);
        impl_->stdout_write = INVALID_HANDLE_VALUE;
    }
    if (impl_->stderr_write != INVALID_HANDLE_VALUE) {
        CloseHandle(impl_->stderr_write);
        impl_->stderr_write = INVALID_HANDLE_VALUE;
    }

    if (!ok) {
        MM_ERROR("CreateProcess failed: {}", err);
        if (used_stdio) {
            last_error_ = "CreateProcess failed: " + std::to_string(err)
                        + " (path=" + resolved_exe_path + ")";
        } else {
            last_error_ = "CreateProcess failed after stdio-retry: "
                        + std::to_string(err)
                        + " (path=" + resolved_exe_path + ")";
        }
        impl_->call_log(last_error_, true);
        impl_->close_read_pipes();
        state_ = ProcessState::Error;
        return false;
    }

    impl_->proc_handle = pi.hProcess;
    CloseHandle(pi.hThread); // not needed

    MM_INFO("llama-server started (PID {}): {}", pi.dwProcessId, cmd_str);
    impl_->call_log("spawned llama-server: " + cmd_str, false);

    // Start pipe reader threads (capture handle values)
    if (used_stdio
        && impl_->stdout_read != INVALID_HANDLE_VALUE
        && impl_->stderr_read != INVALID_HANDLE_VALUE) {
        HANDLE h_out = impl_->stdout_read;
        HANDLE h_err = impl_->stderr_read;
        impl_->pipe_running = true;

        impl_->stdout_reader = std::thread([this, h_out]() {
            char buf[4096];
            std::string line_buf;
            while (impl_->pipe_running.load()) {
                DWORD n = 0;
                BOOL ok2 = ReadFile(h_out, buf, sizeof(buf), &n, nullptr);
                if (!ok2 || n == 0) break;
                line_buf.append(buf, static_cast<size_t>(n));
                size_t pos;
                while ((pos = line_buf.find('\n')) != std::string::npos) {
                    std::string line = line_buf.substr(0, pos);
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    line_buf = line_buf.substr(pos + 1);
                    impl_->call_log(line, false);
                }
            }
            if (!line_buf.empty()) impl_->call_log(line_buf, false);
        });

        impl_->stderr_reader = std::thread([this, h_err]() {
            char buf[4096];
            std::string line_buf;
            while (impl_->pipe_running.load()) {
                DWORD n = 0;
                BOOL ok2 = ReadFile(h_err, buf, sizeof(buf), &n, nullptr);
                if (!ok2 || n == 0) break;
                line_buf.append(buf, static_cast<size_t>(n));
                size_t pos;
                while ((pos = line_buf.find('\n')) != std::string::npos) {
                    std::string line = line_buf.substr(0, pos);
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    line_buf = line_buf.substr(pos + 1);
                    impl_->call_log(line, true);
                }
            }
            if (!line_buf.empty()) impl_->call_log(line_buf, true);
        });
    } else {
        impl_->pipe_running = false;
        MM_WARN("llama-server started without stdio pipe capture (CreateProcess stdio fallback path)");
        impl_->call_log("llama-server started, but stdout/stderr capture is unavailable in this launch mode", true);
    }

#else
    // ── Linux: fork + exec ───────────────────────────────────────────────────
    int out_pipe[2], err_pipe[2];
    if (::pipe(out_pipe) != 0 || ::pipe(err_pipe) != 0) {
        MM_ERROR("pipe() failed: {}", ::strerror(errno));
        last_error_ = std::string("pipe() failed: ") + ::strerror(errno);
        state_ = ProcessState::Error;
        return false;
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        MM_ERROR("fork() failed: {}", ::strerror(errno));
        last_error_ = std::string("fork() failed: ") + ::strerror(errno);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        state_ = ProcessState::Error;
        return false;
    }

    if (pid == 0) {
        // Child process
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);

        std::vector<const char*> argv;
        argv.push_back(llama_server_path_.c_str());
        for (auto& a : args) argv.push_back(a.c_str());
        argv.push_back(nullptr);

        ::execvp(llama_server_path_.c_str(), const_cast<char* const*>(argv.data()));
        ::_exit(127);
    }

    // Parent: close write ends
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    impl_->pid       = pid;
    impl_->stdout_fd = out_pipe[0];
    impl_->stderr_fd = err_pipe[0];

    MM_INFO("llama-server started (PID {}): {}", pid, llama_server_path_);

    int fd_out = impl_->stdout_fd;
    int fd_err = impl_->stderr_fd;
    impl_->pipe_running = true;

    impl_->stdout_reader = std::thread([this, fd_out]() {
        char buf[4096];
        std::string line_buf;
        while (impl_->pipe_running.load()) {
            ssize_t n = ::read(fd_out, buf, sizeof(buf));
            if (n <= 0) break;
            line_buf.append(buf, static_cast<size_t>(n));
            size_t pos;
            while ((pos = line_buf.find('\n')) != std::string::npos) {
                std::string line = line_buf.substr(0, pos);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                line_buf = line_buf.substr(pos + 1);
                impl_->call_log(line, false);
            }
        }
        if (!line_buf.empty()) impl_->call_log(line_buf, false);
    });

    impl_->stderr_reader = std::thread([this, fd_err]() {
        char buf[4096];
        std::string line_buf;
        while (impl_->pipe_running.load()) {
            ssize_t n = ::read(fd_err, buf, sizeof(buf));
            if (n <= 0) break;
            line_buf.append(buf, static_cast<size_t>(n));
            size_t pos;
            while ((pos = line_buf.find('\n')) != std::string::npos) {
                std::string line = line_buf.substr(0, pos);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                line_buf = line_buf.substr(pos + 1);
                impl_->call_log(line, true);
            }
        }
        if (!line_buf.empty()) impl_->call_log(line_buf, true);
    });
#endif

    // Poll /health until ready or timeout
    bool ready = poll_health(60);
    if (!ready) {
        MM_ERROR("llama-server did not become healthy within 60s — aborting");
        last_error_ = "llama-server did not become healthy within 60s (startup failed)";
        impl_->call_log(last_error_, true);
        stop();
        return false;
    }

    state_ = ProcessState::Ready;
    MM_INFO("llama-server is ready on port {}", port_);
    return true;
}

// ── stop ──────────────────────────────────────────────────────────────────────
void LlamaServerProcess::stop() {
    if (state_ == ProcessState::Stopped) return;

    impl_->pipe_running = false;

#ifdef _WIN32
    if (impl_->proc_handle != INVALID_HANDLE_VALUE) {
        TerminateProcess(impl_->proc_handle, 0);
        WaitForSingleObject(impl_->proc_handle, 5000);
        CloseHandle(impl_->proc_handle);
        impl_->proc_handle = INVALID_HANDLE_VALUE;
    }
    // Process is dead: reader threads will get EOF from ReadFile, then we join.
    impl_->join_readers();
    impl_->close_read_pipes();
#else
    if (impl_->pid > 0) {
        ::kill(impl_->pid, SIGTERM);
        for (int i = 0; i < 30; ++i) {
            int status = 0;
            if (::waitpid(impl_->pid, &status, WNOHANG) == impl_->pid) {
                impl_->pid = -1;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (impl_->pid > 0) {
            ::kill(impl_->pid, SIGKILL);
            ::waitpid(impl_->pid, nullptr, 0);
            impl_->pid = -1;
        }
    }
    impl_->join_readers();
    impl_->close_read_pipes();
#endif

    state_ = ProcessState::Stopped;
    MM_INFO("llama-server stopped");
}

// ── Accessors ─────────────────────────────────────────────────────────────────
ProcessState LlamaServerProcess::get_state() const { return state_.load(); }
uint16_t     LlamaServerProcess::get_port()  const { return port_; }

std::string LlamaServerProcess::get_url() const {
    return "http://127.0.0.1:" + std::to_string(port_);
}

std::string LlamaServerProcess::last_error() const { return last_error_; }

// ── poll_health ───────────────────────────────────────────────────────────────
bool LlamaServerProcess::poll_health(int timeout_seconds) {
    httplib::Client cli("127.0.0.1", port_);
    cli.set_connection_timeout(2);
    cli.set_read_timeout(3);

    auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::seconds(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        auto res = cli.Get("/health");
        if (res && res->status == 200) {
            try {
                auto j = nlohmann::json::parse(res->body);
                if (j.value("status", std::string{}) == "ok")
                    return true;
            } catch (const std::exception& e) {
                MM_DEBUG("poll_health: parse error: {}", e.what());
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return false;
}

} // namespace mm
