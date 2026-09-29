#include "control/token_binding.hpp"

#include "common/util.hpp"

#include <vector>

namespace mm {

namespace {

BindingDecision refuse(std::string error) {
    return {false, std::move(error)};
}

bool starts_with_segment(const std::string& path, const std::string& prefix) {
    // "/v1/tokens" and "/v1/tokens/7", never "/v1/tokensmith".
    return path.compare(0, prefix.size(), prefix) == 0 &&
           (path.size() == prefix.size() || path[prefix.size()] == '/');
}

} // namespace

BindingDecision binding_permits(const std::string& bound_user_id,
                                const std::string& method,
                                const std::string& path) {
    if (bound_user_id.empty()) return {};

    std::vector<std::string> segments;
    for (const auto& segment : util::split(path, '/'))
        if (!segment.empty()) segments.push_back(segment);

    if (segments.size() >= 2 && segments[0] == "v1" && segments[1] == "users") {
        if (segments.size() == 2) {
            if (method == "GET") return {}; // filtered to the owner by the handler
            return refuse("a token bound to user '" + bound_user_id + "' cannot create users");
        }
        if (segments[2] == bound_user_id) return {};
        return refuse("this token is bound to user '" + bound_user_id +
                      "' and cannot reach user '" + segments[2] + "'");
    }

    if (starts_with_segment(path, "/v1/tokens"))
        return refuse("a token bound to a user cannot list, mint or revoke tokens");
    if (starts_with_segment(path, "/v1/activity") || starts_with_segment(path, "/v1/performance"))
        return refuse("a token bound to a user cannot read other users' activity");

    if (method == "GET") return {};
    if (method == "POST" && (path == "/v1/chat/completions" || path == "/v1/audio/speech"))
        return {};
    return refuse("a token bound to user '" + bound_user_id +
                  "' may change only that user's data");
}

} // namespace mm
