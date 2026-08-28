#include "common/logger.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <chrono>
#include <filesystem>
#include <vector>

namespace mm {

void init_logger(
    const std::string& log_file,
    const std::string& logger_name,
    spdlog::level::level_enum console_level,
    spdlog::level::level_enum file_level)
{
    // No-op if already initialised.
    if (spdlog::get(logger_name))
        return;

    std::vector<spdlog::sink_ptr> sinks;

    // ── Console (colored) ─────────────────────────────────────────────────────
    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console->set_level(console_level);
    console->set_pattern("[%H:%M:%S.%e] [%^%-5l%$] %v");
    sinks.push_back(console);

    // ── Rotating file ─────────────────────────────────────────────────────────
    if (!log_file.empty()) {
        std::filesystem::path p(log_file);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());

        // 5 MB per file, keep 3 files
        auto file = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_file, 5 * 1024 * 1024, 3);
        file->set_level(file_level);
        file->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%-5l] [t=%t] %v");
        sinks.push_back(file);
    }

    auto logger = std::make_shared<spdlog::logger>(
        logger_name, sinks.begin(), sinks.end());
    logger->set_level(spdlog::level::trace);  // let individual sinks filter
    // Flush on WARN, not just ERR.
    //
    // A warning that is still in a buffer when the process is killed did not
    // happen, as far as anyone reading the log is concerned. That is not
    // hypothetical: the node warns "Control rejected registration" and, in CLI
    // mode where the console sink is off, the message existed only in a buffer —
    // so an operator whose node would not pair saw a 0-byte log file and no
    // explanation anywhere (roadmap D20).
    //
    // Warnings are rare by construction, so flushing them costs nothing that
    // matters and buys the property that the last thing a dying process
    // complained about is on disk.
    logger->flush_on(spdlog::level::warn);

    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);

    // …and a periodic flush behind it, because flush_on(warn) only saves the
    // tail of a process that WARNED on its way out.
    //
    // A process that aborts saves nothing. The node died to an uncaught
    // exception on the engine-provisioning thread and the log ended four lines
    // early, at an unrelated warning — every info line since was still in a
    // buffer, including the one naming the config it had just been handed. The
    // report was "it closes without notice and does not even show in the logs",
    // and the second half of that was this: the crash was invisible in the file
    // written to explain it (D57).
    //
    // Two seconds, not sub-second: the cost is a wakeup on one shared thread,
    // and the property being bought is "the last thing the process did is on
    // disk", which a 2 s window preserves for anything a human is diagnosing
    // after the fact. Idempotent across init_logger's early return — this is
    // registry-wide, not per-logger.
    spdlog::flush_every(std::chrono::seconds(2));
}

std::shared_ptr<spdlog::logger> get_logger(const std::string& name) {
    auto named = spdlog::get(name);
    if (named) return named;
    return spdlog::default_logger();
}

} // namespace mm
