#include "common/logger.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
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
    logger->flush_on(spdlog::level::err);

    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);
}

std::shared_ptr<spdlog::logger> get_logger(const std::string& name) {
    auto named = spdlog::get(name);
    if (named) return named;
    return spdlog::default_logger();
}

} // namespace mm
