// Mantic-Mind — scoped API authorization.
//
// Auth before this was ONE flat bearer token compared with a plain !=, gating
// every /v1/* path identically, and entirely opt-in: an empty
// external_api_token left the whole surface open.
//
// That was untenable once admission existed. `POST /v1/models/admit` starts
// hours of conversion and consumes tens of GB; it must not sit behind the same
// credential that lets a client send a chat message. And a telemetry dashboard —
// which only ever reads — should not hold a token that can delete an agent.
//
// The split is by BLAST RADIUS, not by resource. That is why `GET
// /v1/agents/{id}/memories` is `read` while `POST .../memories` is `chat`: they
// touch the same rows, and only one of them can change anything.

#include "control/route_scope.hpp"

#include "common/logger.hpp"
#include "common/pairing.hpp"
#include "common/util.hpp"
#include "control/model_registry.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <mutex>

namespace mm {

const char* to_string(Scope scope) {
    switch (scope) {
    case Scope::Read:
        return "read";
    case Scope::Chat:
        return "chat";
    case Scope::Operator:
        return "operator";
    }
    return "unknown";
}

bool scope_permits(ScopeSet held, Scope required) {
    // `operator` implies the other two. A credential that can delete an agent
    // being unable to read it would be a distinction nobody wants to administer,
    // and `chat` implies `read` for the same reason: sending a message you
    // cannot then fetch is not a coherent permission.
    if ((held & static_cast<ScopeSet>(Scope::Operator)) != 0) return true;
    if (required == Scope::Read) {
        return (held & (static_cast<ScopeSet>(Scope::Read) | static_cast<ScopeSet>(Scope::Chat))) !=
               0;
    }
    return (held & static_cast<ScopeSet>(required)) != 0;
}

bool parse_scopes(const std::string& csv, ScopeSet& out) {
    out = kScopeNone;
    bool ok = true;
    for (const auto& raw : util::split(csv, ',')) {
        const auto s = util::to_lower(util::trim(raw));
        if (s.empty()) continue;
        if (s == "read")
            out |= static_cast<ScopeSet>(Scope::Read);
        else if (s == "chat")
            out |= static_cast<ScopeSet>(Scope::Chat);
        else if (s == "operator")
            out |= static_cast<ScopeSet>(Scope::Operator);
        else
            ok = false; // reported, not silently dropped
    }
    return ok;
}

std::string format_scopes(ScopeSet scopes) {
    std::string out;
    const auto add = [&](Scope s) {
        if ((scopes & static_cast<ScopeSet>(s)) == 0) return;
        if (!out.empty()) out += ",";
        out += to_string(s);
    };
    add(Scope::Read);
    add(Scope::Chat);
    add(Scope::Operator);
    return out;
}

// ── the table ─────────────────────────────────────────────────────────────────

std::vector<RouteScope> route_scope_table() {
    // Exhaustive on purpose. A route absent from this table is a startup
    // failure, not a default: defaulting an unlisted route to `read` would
    // silently under-protect a new mutation, and defaulting to `operator` would
    // silently break a new GET. Both are worse than refusing to start.
    return {
        // ── agents: read ─────────────────────────────────────────────────────
        {"GET", "/v1/agents", Scope::Read},
        {"GET", "/v1/agents/:id", Scope::Read},
        {"GET", "/v1/agents/:id/conversations", Scope::Read},
        {"GET", "/v1/agents/:id/conversations/:cid", Scope::Read},
        {"GET", "/v1/agents/:id/conversations/:cid/local-memories", Scope::Read},
        {"GET", "/v1/agents/:id/memories", Scope::Read},
        {"GET", "/v1/agents/:id/attachments/:attachment_id", Scope::Read},
        {"GET", "/v1/agents/:id/voice", Scope::Read},
        {"GET", "/v1/agents/:id/voice/proposals", Scope::Read},
        {"GET", "/v1/agents/:id/voice/proposals/:proposal_id/sample", Scope::Read},
        {"GET", "/v1/agents/:id/speech/cache/:cache_id", Scope::Read},

        // ── agents: chat ─────────────────────────────────────────────────────
        //
        // Bounded and per-agent. Everything here changes one agent's own
        // conversational state and nothing else's.
        {"POST", "/v1/agents/:id/chat", Scope::Chat},
        {"POST", "/v1/agents/:id/conversations", Scope::Chat},
        {"PUT", "/v1/agents/:id/conversations/:cid", Scope::Chat},
        {"POST", "/v1/agents/:id/conversations/:cid/activate", Scope::Chat},
        {"POST", "/v1/agents/:id/conversations/:cid/compact", Scope::Chat},
        {"POST", "/v1/agents/:id/conversations/:cid/local-memories", Scope::Chat},
        {"PUT", "/v1/agents/:id/conversations/:cid/local-memories/:mid", Scope::Chat},
        {"POST", "/v1/agents/:id/memories", Scope::Chat},
        {"PUT", "/v1/agents/:id/memories/:mid", Scope::Chat},
        {"POST", "/v1/agents/:id/memories/extract", Scope::Chat},
        {"POST", "/v1/agents/:id/curation/proposals", Scope::Chat},
        {"POST", "/v1/agents/:id/curation/proposals/apply", Scope::Chat},
        {"POST", "/v1/agents/:id/curation/apply", Scope::Chat},
        {"POST", "/v1/agents/:id/speech", Scope::Chat},
        // Registered through PostUpload rather than Post, which is exactly why
        // the coverage check reads the SERVER's registrations instead of a list
        // someone greps for: this one was missed on the first pass and the check
        // refused to start until it was added.
        {"POST", "/v1/agents/:id/attachments", Scope::Chat},
        {"POST", "/v1/audio/speech", Scope::Chat},
        {"POST", "/v1/chat/completions", Scope::Chat},
        {"POST", "/v1/agents/:id/voice/proposals", Scope::Chat},
        {"POST", "/v1/agents/:id/voice/proposals/:proposal_id/sample", Scope::Chat},

        // DELETEs are operator even when they are per-agent and small. Losing a
        // conversation is not recoverable from the API, and "bounded" is about
        // what a mistake costs, not how many rows it touches.
        {"DELETE", "/v1/agents/:id/conversations/:cid", Scope::Operator},
        {"DELETE", "/v1/agents/:id/conversations/:cid/local-memories/:mid", Scope::Operator},
        {"DELETE", "/v1/agents/:id/memories/:mid", Scope::Operator},
        {"DELETE", "/v1/agents/:id/attachments/:attachment_id", Scope::Operator},

        // ── agents: operator ─────────────────────────────────────────────────
        {"POST", "/v1/agents", Scope::Operator},
        {"PUT", "/v1/agents/:id", Scope::Operator},
        {"DELETE", "/v1/agents/:id", Scope::Operator},
        // Which ENGINE serves an agent. A performance and correctness decision,
        // and the one place an operator can overrule a verdict.
        {"PUT", "/v1/agents/:id/backend", Scope::Operator},
        {"POST", "/v1/agents/:id/voice/proposals/:proposal_id/approve", Scope::Operator},
        {"POST", "/v1/agents/:id/voice/proposals/:proposal_id/reject", Scope::Operator},

        // ── models ───────────────────────────────────────────────────────────
        {"GET", "/v1/models", Scope::Read},
        {"GET", "/v1/models/:id", Scope::Read},
        {"GET", "/v1/models/:id/plan", Scope::Read},
        {"GET", "/v1/models/:id/heat", Scope::Read},
        {"GET", "/v1/models/admissions", Scope::Read},
        {"GET", "/v1/models/admissions/:op", Scope::Read},
        // THE motivating case. Hours of CPU and tens of GB of disk.
        {"POST", "/v1/models/admit", Scope::Operator},
        {"POST", "/v1/models", Scope::Operator},
        {"POST", "/v1/models/:id/reprofile", Scope::Operator},
        {"POST", "/v1/models/admissions/:op/cancel", Scope::Operator},
        {"PUT", "/v1/models/:id/verdict", Scope::Operator},
        {"DELETE", "/v1/models/:id", Scope::Operator},

        // ── engines ──────────────────────────────────────────────────────────
        //
        // All read: telemetry is observation. The routes that CHANGE an engine —
        // suspend, restore, unload — go through the agent and model surfaces and
        // are operator there.
        {"GET", "/v1/engines", Scope::Read},
        {"GET", "/v1/engines/:id/telemetry", Scope::Read},
        {"GET", "/v1/engines/:id/heat", Scope::Read},
        {"GET", "/v1/engines/:id/slots", Scope::Read},

        // ── cluster ──────────────────────────────────────────────────────────
        {"GET", "/v1/nodes", Scope::Read},
        {"GET", "/v1/nodes/discovered", Scope::Read},
        {"GET", "/v1/placements", Scope::Read},
        {"GET", "/v1/performance", Scope::Read},
        {"GET", "/v1/activity", Scope::Read},
        {"POST", "/v1/nodes", Scope::Operator},
        {"DELETE", "/v1/nodes/:id", Scope::Operator},
        {"POST", "/v1/nodes/:id/forget", Scope::Operator},
        {"DELETE", "/v1/performance", Scope::Operator},

        // ── pairing ──────────────────────────────────────────────────────────
        //
        // Pairing has its OWN credential (the pre-shared key) and is how a node
        // obtains one in the first place. Requiring an API scope here would mean
        // a node could not be paired without already holding a token, which is
        // the wrong way round. Marked `read` so the table is complete and the
        // coverage check has an entry, and gated by the PSK in the handler.
        {"POST", "/v1/nodes/pair/start", Scope::Read},
        {"POST", "/v1/nodes/pair/complete", Scope::Read},
        {"POST", "/v1/nodes/pair/psk", Scope::Operator},

        // ── tokens ───────────────────────────────────────────────────────────
        //
        // Operator-only, necessarily: a credential that can mint credentials is
        // the most powerful one there is, and listing them tells an attacker
        // exactly which scopes exist to target.
        {"GET", "/v1/tokens", Scope::Operator},
        {"POST", "/v1/tokens", Scope::Operator},
        {"DELETE", "/v1/tokens/:id", Scope::Operator},
    };
}

bool require_complete_coverage(const std::vector<std::string>& registered_routes,
                               std::vector<std::string>& out_missing) {
    const auto table = route_scope_table();
    out_missing.clear();
    for (const auto& route : registered_routes) {
        const auto space = route.find(' ');
        if (space == std::string::npos) continue;
        const auto method = route.substr(0, space);
        const auto pattern = route.substr(space + 1);
        const bool found = std::any_of(table.begin(), table.end(), [&](const RouteScope& rs) {
            return method == rs.method && pattern == rs.pattern;
        });
        if (!found) out_missing.push_back(route);
    }
    return out_missing.empty();
}

// ── ScopeAuthorizer ───────────────────────────────────────────────────────────

namespace {

/// Does a concrete path match a route pattern with `:name` segments?
bool pattern_matches(const std::string& pattern, const std::string& path) {
    const auto pat = util::split(pattern, '/');
    const auto seg = util::split(path, '/');
    if (pat.size() != seg.size()) return false;
    for (std::size_t i = 0; i < pat.size(); ++i) {
        if (!pat[i].empty() && pat[i][0] == ':') continue; // a parameter matches anything
        if (pat[i] != seg[i]) return false;
    }
    return true;
}

} // namespace

struct ScopeAuthorizer::Impl {
    mutable std::mutex mu;
    ControlModelRegistry* registry = nullptr;
    std::string legacy_token;
    std::vector<RouteScope> table = route_scope_table();
};

ScopeAuthorizer::ScopeAuthorizer() : impl_(std::make_unique<Impl>()) {}

ScopeAuthorizer::~ScopeAuthorizer() = default;

void ScopeAuthorizer::configure(ControlModelRegistry* registry, std::string legacy_token) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->registry = registry;
    impl_->legacy_token = std::move(legacy_token);
}

bool ScopeAuthorizer::auth_enabled() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->legacy_token.empty()) return true;
    return impl_->registry != nullptr && impl_->registry->has_api_tokens();
}

Scope ScopeAuthorizer::required_scope(const std::string& method, const std::string& path) const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    // Longest literal prefix wins, so "/v1/models/admissions" is not shadowed by
    // "/v1/models/:id". Matching in table order would make the table's ORDER
    // load-bearing, which is exactly the kind of thing that breaks when someone
    // sorts it.
    const RouteScope* best = nullptr;
    std::size_t best_literals = 0;
    for (const auto& rs : impl_->table) {
        if (method != rs.method) continue;
        if (!pattern_matches(rs.pattern, path)) continue;
        std::size_t literals = 0;
        for (const auto& seg : util::split(rs.pattern, '/')) {
            if (!seg.empty() && seg[0] != ':') ++literals;
        }
        if (best == nullptr || literals > best_literals) {
            best = &rs;
            best_literals = literals;
        }
    }
    // An unmatched path is `operator`, not `read`. Startup coverage should make
    // this unreachable for registered routes; if something slips through, the
    // safe answer is the restrictive one.
    return best != nullptr ? best->required : Scope::Operator;
}

AuthResult ScopeAuthorizer::authorize(const httplib::Request& req) const {
    AuthResult result;

    std::string legacy;
    ControlModelRegistry* registry = nullptr;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        legacy = impl_->legacy_token;
        registry = impl_->registry;
    }

    const bool have_tokens = registry != nullptr && registry->has_api_tokens();
    if (legacy.empty() && !have_tokens) {
        // Auth off, matching the previous behaviour exactly. Logged loudly at
        // startup rather than here: a per-request warning on an open system is
        // noise that trains people to ignore it.
        result.allowed = true;
        result.granted = kScopeAll;
        result.token_label = "unauthenticated";
        return result;
    }

    const auto auth = req.get_header_value("Authorization");
    static const std::string kBearer = "Bearer ";
    if (auth.size() <= kBearer.size() || auth.compare(0, kBearer.size(), kBearer) != 0) {
        result.status = 401;
        result.error = "missing bearer token";
        return result;
    }
    const auto presented = auth.substr(kBearer.size());

    // The legacy token is matched FIRST and is never a row.
    //
    // Grandfathered as all-scopes so existing deployments keep working
    // unchanged. Checked before the table so that rotating it in config takes
    // effect immediately and cannot leave a stale grant behind — which is
    // exactly what would happen if startup inserted it as a row.
    if (!legacy.empty() && presented == legacy) {
        result.allowed = true;
        result.granted = kScopeAll;
        result.token_label = "legacy-config-token";
        return result;
    }

    if (registry != nullptr) {
        ApiToken row;
        // Looked up by HASH: the table stores token_sha256 and never the token,
        // so a leaked database backup does not hand over working credentials.
        if (registry->find_api_token(pairing::sha256_hex(presented), row)) {
            result.allowed = true;
            result.granted = row.scopes;
            result.token_label = row.label;
            result.token_id = row.id;
            return result;
        }
    }

    result.status = 403;
    result.error = "invalid bearer token";
    return result;
}

bool ScopeAuthorizer::authorize_or_reject(const httplib::Request& req,
                                          httplib::Response& res) const {
    auto auth = authorize(req);
    if (!auth.allowed) {
        res.status = auth.status;
        res.set_content(nlohmann::json{{"error", auth.error}}.dump(), "application/json");
        return false;
    }

    const auto required = required_scope(req.method, req.path);
    if (!scope_permits(auth.granted, required)) {
        res.status = 403;
        // Says WHICH scope was needed and which were held. "403 forbidden" with
        // no detail sends an operator to read source to find out what to grant.
        res.set_content(nlohmann::json{{"error", "insufficient scope"},
                                       {"required", to_string(required)},
                                       {"granted", format_scopes(auth.granted)}}
                            .dump(),
                        "application/json");
        return false;
    }
    return true;
}

} // namespace mm
