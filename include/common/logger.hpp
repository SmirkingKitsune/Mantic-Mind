#pragma once

#include <string>
#include <memory>
#include <spdlog/spdlog.h>

namespace mm {

// Initialize global logger with rotating file + colored console sinks.
// Call once at startup before any logging. Safe to call multiple times
// (subsequent calls are no-ops if logger is already registered).
void init_logger(
    const std::string& log_file,
    const std::string& logger_name   = "mm",
    spdlog::level::level_enum console_level = spdlog::level::info,
    spdlog::level::level_enum file_level    = spdlog::level::trace
);

// Returns the named logger (nullptr if not yet initialized).
std::shared_ptr<spdlog::logger> get_logger(const std::string& name = "mm");

} // namespace mm

// ── Convenience macros ────────────────────────────────────────────────────────
// These are safe to call even before init_logger() — they check for nullptr.
#define MM_TRACE(...)    do { auto _l = mm::get_logger(); if (_l) _l->trace(__VA_ARGS__);    } while(0)
#define MM_DEBUG(...)    do { auto _l = mm::get_logger(); if (_l) _l->debug(__VA_ARGS__);    } while(0)
#define MM_INFO(...)     do { auto _l = mm::get_logger(); if (_l) _l->info(__VA_ARGS__);     } while(0)
#define MM_WARN(...)     do { auto _l = mm::get_logger(); if (_l) _l->warn(__VA_ARGS__);     } while(0)
#define MM_ERROR(...)    do { auto _l = mm::get_logger(); if (_l) _l->error(__VA_ARGS__);    } while(0)
#define MM_CRITICAL(...) do { auto _l = mm::get_logger(); if (_l) _l->critical(__VA_ARGS__); } while(0)
