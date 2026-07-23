#pragma once

#include "common/models.hpp"
#include "common/util.hpp"

#include <array>
#include <cctype>
#include <optional>
#include <string>
#include <string_view>

namespace mm {

// These llama-server arguments are owned by the node lifecycle. Allowing an
// agent profile to supply a second copy could expose the private runtime,
// redirect it to a different model/port, or move KV data outside node storage.
// Keep this check shared by control-side validation and the final node launch
// seam so direct node API callers receive the same protection.
inline std::optional<std::string> managed_llama_server_extra_arg(
    const RuntimeSettings& settings) {
    static constexpr std::array<std::string_view, 8> managed_flags = {
        "--host",
        "--port",
        "--model",
        "-m",
        "--slot-save-path",
        "--mmproj",
        "-mm",
        "--mmproj-url",
    };

    for (const auto& raw : settings.extra_args) {
        const std::string arg = util::to_lower(util::trim(raw));
        for (const std::string_view flag : managed_flags) {
            if (arg.size() < flag.size() ||
                arg.compare(0, flag.size(), flag) != 0) {
                continue;
            }
            if (arg.size() == flag.size()) return std::string(flag);

            const unsigned char boundary =
                static_cast<unsigned char>(arg[flag.size()]);
            if (boundary == '=' || std::isspace(boundary)) {
                return std::string(flag);
            }
        }
    }
    return std::nullopt;
}

} // namespace mm
