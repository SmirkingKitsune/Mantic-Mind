#pragma once

// Mantic-Mind — scoped API authorization.
//
// Auth today is ONE flat bearer token, compared with a plain != , gating every
// /v1/* path identically, and entirely opt-in (empty external_api_token means
// the whole surface is open). There is no scope, role, or permission concept
// anywhere in the codebase.
//
// That is untenable once admission exists. `POST /v1/models/admit` starts hours
// of conversion and profiling and consumes tens of GB; it must not sit behind
// the same credential that lets a client send a chat message. And a telemetry
// dashboard — which only ever reads — should not hold a token that can delete an
// agent.
//
// Three scopes, and the split is by BLAST RADIUS rather than by resource:
//
//   read      every GET, every telemetry SSE stream        no side effects
//   chat      agent chat, conversations, memories, uploads  bounded, per-agent
//   operator  admit, reprofile, delete, verdict override,   unbounded or
//             backend override, suspend/restore             destructive

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace httplib {
struct Request;
struct Response;
} // namespace httplib

namespace mm {

enum class Scope : std::uint8_t {
    Read = 1 << 0,
    Chat = 1 << 1,
    Operator = 1 << 2,
};

using ScopeSet = std::uint8_t;

inline constexpr ScopeSet kScopeNone = 0;
inline constexpr ScopeSet kScopeAll = static_cast<ScopeSet>(Scope::Read) |
                                      static_cast<ScopeSet>(Scope::Chat) |
                                      static_cast<ScopeSet>(Scope::Operator);

/// One route's requirement. The table is exhaustive and checked at startup —
/// see require_complete_coverage().
struct RouteScope {
    const char* method = nullptr;  ///< "GET" | "POST" | "PUT" | "DELETE"
    const char* pattern = nullptr; ///< "/v1/agents/:id/chat"
    Scope required = Scope::Read;
};

struct AuthResult {
    bool allowed = false;
    ScopeSet granted = kScopeNone;
    std::string token_label;
    std::int64_t token_id = 0;
    int status = 200; ///< 401 missing, 403 insufficient
    std::string error;
};

class ControlModelRegistry;

/// Pre-routing authorization.
///
/// Installed via SetPreRoutingHandler, the same hook
/// ControlApiServer::authorize_external_request uses today — one middleware,
/// replaced rather than layered.
class ScopeAuthorizer {
public:
    ScopeAuthorizer();
    ~ScopeAuthorizer();

    /// `legacy_token` is ControlConfig::external_api_token, grandfathered as an
    /// all-scopes credential so existing deployments keep working unchanged.
    ///
    /// Matched BEFORE the database lookup and never inserted as a row, so
    /// rotating it in config cannot leave a stale grant behind.
    void configure(ControlModelRegistry* registry, std::string legacy_token);

    /// Empty legacy token AND no rows in api_token means auth is off, matching
    /// today's behaviour exactly. Logged loudly at startup — a system that is
    /// open because nobody configured it should say so.
    bool auth_enabled() const;

    AuthResult authorize(const httplib::Request& req) const;

    /// Convenience for SetPreRoutingHandler; writes the error body on refusal.
    bool authorize_or_reject(const httplib::Request& req, httplib::Response& res) const;

    Scope required_scope(const std::string& method, const std::string& path) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// The complete route -> scope table.
///
/// Every route in ControlApiServer appears here, including all 51 that predate
/// scoping. A route absent from the table is a startup failure, not a default —
/// defaulting an unlisted route to `read` would silently under-protect a new
/// mutation, and defaulting to `operator` would silently break a new GET. Both
/// are worse than refusing to start.
std::vector<RouteScope> route_scope_table();

/// Verifies at startup that every registered handler has a table entry.
bool require_complete_coverage(const std::vector<std::string>& registered_routes,
                               std::vector<std::string>& out_missing);

bool scope_permits(ScopeSet held, Scope required);
const char* to_string(Scope scope);
bool parse_scopes(const std::string& csv, ScopeSet& out);
std::string format_scopes(ScopeSet scopes);

} // namespace mm
