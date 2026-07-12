#include "node/process_exec.hpp"

#include <chrono>
#include <mutex>
#include <thread>

#ifdef _WIN32
#  include <Windows.h>
#  include <sstream>
#  include <vector>
#else
#  include <cerrno>
#  include <csignal>
#  include <cstring>
#  include <fcntl.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace mm {
namespace {

// Split a growing byte buffer into complete lines, forwarding each (minus a
// trailing CR) to the sink. Leftover partial data stays in `buf` for next time.
void drain_lines(std::string& buf, bool is_stderr,
                 const std::function<void(const std::string&, bool)>& sink) {
    size_t pos;
    while ((pos = buf.find('\n')) != std::string::npos) {
        std::string line = buf.substr(0, pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        buf.erase(0, pos + 1);
        sink(line, is_stderr);
    }
}

#ifdef _WIN32
// Quote a single argument per the CommandLineToArgvW rules (mirrors
// runtime_process.cpp's quoting so both spawn paths behave identically).
std::string quote_windows_arg(const std::string& arg) {
    if (!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string::npos) return arg;

    std::string out;
    out.reserve(arg.size() + 8);
    out.push_back('"');
    size_t backslashes = 0;
    for (char ch : arg) {
        if (ch == '\\') { ++backslashes; continue; }
        if (ch == '"') {
            out.append(backslashes * 2 + 1, '\\');
            out.push_back('"');
            backslashes = 0;
            continue;
        }
        if (backslashes > 0) { out.append(backslashes, '\\'); backslashes = 0; }
        out.push_back(ch);
    }
    out.append(backslashes * 2, '\\');
    out.push_back('"');
    return out;
}
#endif

} // namespace

int run_streamed_command(const std::vector<std::string>& argv,
                         const std::filesystem::path& cwd,
                         const StreamLineCallback& line_cb,
                         std::string* error) {
    return run_streamed_command(argv, cwd, line_cb, CancelCheckCallback{}, error);
}

int run_streamed_command(const std::vector<std::string>& argv,
                         const std::filesystem::path& cwd,
                         const StreamLineCallback& line_cb,
                         const CancelCheckCallback& cancel_requested,
                         std::string* error) {
    if (argv.empty()) {
        if (error) *error = "empty command";
        return -1;
    }

    // Serialize callback invocations across the two reader threads.
    std::mutex cb_mutex;
    auto emit = [&](const std::string& line, bool is_stderr) {
        if (!line_cb) return;
        std::lock_guard<std::mutex> lk(cb_mutex);
        line_cb(line, is_stderr);
    };

#ifdef _WIN32
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE out_read = INVALID_HANDLE_VALUE, out_write = INVALID_HANDLE_VALUE;
    HANDLE err_read = INVALID_HANDLE_VALUE, err_write = INVALID_HANDLE_VALUE;
    if (!CreatePipe(&out_read, &out_write, &sa, 0)) {
        if (error) *error = "CreatePipe(stdout) failed: " + std::to_string(GetLastError());
        return -1;
    }
    SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0);
    if (!CreatePipe(&err_read, &err_write, &sa, 0)) {
        if (error) *error = "CreatePipe(stderr) failed: " + std::to_string(GetLastError());
        CloseHandle(out_read); CloseHandle(out_write);
        return -1;
    }
    SetHandleInformation(err_read, HANDLE_FLAG_INHERIT, 0);

    std::ostringstream cmd;
    cmd << quote_windows_arg(argv[0]);
    for (size_t i = 1; i < argv.size(); ++i) cmd << " " << quote_windows_arg(argv[i]);
    std::string cmd_str = cmd.str();
    std::vector<char> cmd_buf(cmd_str.begin(), cmd_str.end());
    cmd_buf.push_back('\0');

    const bool has_sep = argv[0].find_first_of("\\/:") != std::string::npos;
    const char* app_name = has_sep ? argv[0].c_str() : nullptr;
    const std::string cwd_str = cwd.empty() ? std::string{} : cwd.string();

    // Feed stdin from NUL: install tools never block on console prompts, and
    // STARTF_USESTDHANDLES requires a valid inheritable stdin handle.
    HANDLE nul_in = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                &sa, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = out_write;
    si.hStdError = err_write;
    si.hStdInput = nul_in;

    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessA(app_name, cmd_buf.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW, nullptr,
                             cwd_str.empty() ? nullptr : cwd_str.c_str(),
                             &si, &pi);
    // Parent never writes to the child's stdio; close write ends now so the
    // reader threads observe EOF when the child exits.
    CloseHandle(out_write);
    CloseHandle(err_write);
    if (nul_in != INVALID_HANDLE_VALUE) CloseHandle(nul_in);
    if (!ok) {
        if (error) *error = "CreateProcess failed: " + std::to_string(GetLastError())
                          + " (" + argv[0] + ")";
        CloseHandle(out_read); CloseHandle(err_read);
        return -1;
    }
    CloseHandle(pi.hThread);

    auto pump = [&](HANDLE h, bool is_stderr) {
        char buf[4096];
        std::string acc;
        for (;;) {
            DWORD n = 0;
            if (!ReadFile(h, buf, sizeof(buf), &n, nullptr) || n == 0) break;
            acc.append(buf, static_cast<size_t>(n));
            drain_lines(acc, is_stderr, emit);
        }
        if (!acc.empty()) emit(acc, is_stderr);
    };
    std::thread t_out(pump, out_read, false);
    std::thread t_err(pump, err_read, true);

    bool canceled = false;
    for (;;) {
        const DWORD wait = WaitForSingleObject(pi.hProcess, 100);
        if (wait == WAIT_OBJECT_0) break;
        if (wait == WAIT_FAILED) break;
        if (cancel_requested && cancel_requested()) {
            canceled = true;
            TerminateProcess(pi.hProcess, 130);
            WaitForSingleObject(pi.hProcess, INFINITE);
            break;
        }
    }
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);

    if (t_out.joinable()) t_out.join();
    if (t_err.joinable()) t_err.join();
    CloseHandle(out_read);
    CloseHandle(err_read);
    CloseHandle(pi.hProcess);
    if (canceled) {
        if (error) *error = "command canceled";
        return 130;
    }
    return static_cast<int>(code);
#else
    int out_pipe[2], err_pipe[2];
    if (::pipe(out_pipe) != 0 || ::pipe(err_pipe) != 0) {
        if (error) *error = std::string("pipe() failed: ") + ::strerror(errno);
        return -1;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        if (error) *error = std::string("fork() failed: ") + ::strerror(errno);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        return -1;
    }

    if (pid == 0) {
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::dup2(err_pipe[1], STDERR_FILENO);
        // stdin from /dev/null so install tools never block on a prompt.
        const int devnull = ::open("/dev/null", O_RDONLY);
        if (devnull >= 0) { ::dup2(devnull, STDIN_FILENO); ::close(devnull); }
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);

        // Restore default signal disposition/mask before exec (see the matching
        // rationale in runtime_process.cpp): a blocked/ignored SIGCHLD inherited
        // from a worker thread breaks child tools that manage their own
        // subprocesses. Only async-signal-safe calls between fork and exec.
        sigset_t empty_mask;
        ::sigemptyset(&empty_mask);
        ::sigprocmask(SIG_SETMASK, &empty_mask, nullptr);
        struct sigaction dfl{};
        dfl.sa_handler = SIG_DFL;
        ::sigemptyset(&dfl.sa_mask);
        ::sigaction(SIGCHLD, &dfl, nullptr);
        ::sigaction(SIGPIPE, &dfl, nullptr);

        if (!cwd.empty()) {
            if (::chdir(cwd.c_str()) != 0) ::_exit(126);
        }

        std::vector<const char*> cargv;
        cargv.reserve(argv.size() + 1);
        for (const auto& a : argv) cargv.push_back(a.c_str());
        cargv.push_back(nullptr);
        ::execvp(argv[0].c_str(), const_cast<char* const*>(cargv.data()));
        ::_exit(127);
    }

    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    auto pump = [&](int fd, bool is_stderr) {
        char buf[4096];
        std::string acc;
        for (;;) {
            ssize_t n = ::read(fd, buf, sizeof(buf));
            if (n <= 0) break;
            acc.append(buf, static_cast<size_t>(n));
            drain_lines(acc, is_stderr, emit);
        }
        if (!acc.empty()) emit(acc, is_stderr);
    };
    std::thread t_out(pump, out_pipe[0], false);
    std::thread t_err(pump, err_pipe[0], true);

    int status = 0;
    bool canceled = false;
    for (;;) {
        const pid_t got = ::waitpid(pid, &status, WNOHANG);
        if (got == pid) break;
        if (got < 0 && errno != EINTR) break;
        if (cancel_requested && cancel_requested()) {
            canceled = true;
            ::kill(pid, SIGTERM);
            for (int i = 0; i < 20; ++i) {
                if (::waitpid(pid, &status, WNOHANG) == pid) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (::waitpid(pid, &status, WNOHANG) == 0) {
                ::kill(pid, SIGKILL);
                while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            }
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (t_out.joinable()) t_out.join();
    if (t_err.joinable()) t_err.join();
    ::close(out_pipe[0]);
    ::close(err_pipe[0]);

    if (canceled) {
        if (error) *error = "command canceled";
        return 130;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
#endif
}

} // namespace mm
