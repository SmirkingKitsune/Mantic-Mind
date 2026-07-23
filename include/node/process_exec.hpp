#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace mm {

// Run a command to completion, streaming each stdout/stderr line to `line_cb`
// as it arrives (rather than buffering silently). This is the visible,
// in-program alternative to writing an install script and running it blind.
//
// - argv[0] is the executable: a bare name is resolved via PATH, a path is used
//   directly. Remaining elements are passed verbatim (no shell involved).
// - `cwd` sets the working directory; empty inherits the parent's.
// - `line_cb` is invoked (possibly from reader threads) once per output line,
//   with is_stderr distinguishing the two streams. Trailing CR/LF are stripped.
//
// Returns the process exit code, or -1 if the process could not be launched
// (with *error set when non-null). Blocks until the process exits. The
// cancellation overload returns 130 after terminating the complete spawned
// process tree (Windows Job Object / POSIX process group).
using StreamLineCallback = std::function<void(const std::string& line, bool is_stderr)>;
using CancelCheckCallback = std::function<bool()>;

int run_streamed_command(const std::vector<std::string>& argv,
                         const std::filesystem::path& cwd,
                         const StreamLineCallback& line_cb,
                         std::string* error = nullptr);
int run_streamed_command(const std::vector<std::string>& argv,
                         const std::filesystem::path& cwd,
                         const StreamLineCallback& line_cb,
                         const CancelCheckCallback& cancel_requested,
                         std::string* error = nullptr);

} // namespace mm
