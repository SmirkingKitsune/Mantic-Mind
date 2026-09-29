#pragma once

// Mantic-Mind — what a token bound to one user may reach.
//
// Binding is optional. A token minted with a user id reaches that user's
// routes under the ordinary scope rules and nothing that would show it — or let
// it change — anyone else's. Unbound tokens, the legacy config token and an
// auth-off deployment behave exactly as they did before users existed.
//
// Checked AFTER the scope check: binding only ever narrows what a scope grants.
//
//   /v1/users/{own}[/...]           the token's scopes apply unchanged
//   /v1/users/{other}[/...]         403
//   GET /v1/users                   allowed; the handler lists only the owner
//   POST /v1/users                  403 — creating users is not one user's act
//   /v1/tokens*                     403 — minting and revoking reach everyone
//   /v1/activity, /v1/performance   403 — both record every user's requests
//   POST /v1/chat/completions       allowed (:9091 — stateless, stores nothing)
//   POST /v1/audio/speech           allowed; the handler requires voice user:{own}
//   any other GET                   allowed — the shared catalog, cluster state
//   any other request               403 — a user's token changes only its user
//
// A pure function of (binding, method, path), so the whole matrix is tested
// without a server.

#include <string>

namespace mm {

struct BindingDecision {
    bool allowed = true;
    std::string error; ///< why not, when not
};

/// `bound_user_id` empty: an unbound token, always allowed.
BindingDecision binding_permits(const std::string& bound_user_id,
                                const std::string& method,
                                const std::string& path);

} // namespace mm
