#pragma once

#include <string_view>

namespace mm {

struct ExternalApiAuthResult {
    int status;
    const char* error;
};

inline ExternalApiAuthResult authorize_external_api(std::string_view peer,
                                                    std::string_view authorization,
                                                    std::string_view token) {
    // The listeners supply the socket peer address.
    const bool loopback = peer.starts_with("127.") || peer == "::1" ||
                          peer.starts_with("::ffff:127.");
    if (token.empty() && loopback) return {200, nullptr};

    constexpr std::string_view bearer = "Bearer ";
    if (!authorization.starts_with(bearer) || authorization.size() == bearer.size()) {
        return {401, "missing bearer token"};
    }
    if (token.empty() || authorization.substr(bearer.size()) != token) {
        return {403, "invalid bearer token"};
    }
    return {200, nullptr};
}

} // namespace mm
