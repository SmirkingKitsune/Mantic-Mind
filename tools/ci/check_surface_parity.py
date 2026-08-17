#!/usr/bin/env python3
"""Every operator capability the TUI or CLI reaches is reachable through /v1/*.

P1 says the API is the single control plane: "The FTXUI TUI is *one client*. If
the TUI can do it, an API client can do it through the same route. There are no
TUI-only features and no internal-only capabilities."

Two checks already guard neighbouring properties and neither guards this one:

  check_api_docs.py    docs -> scope table   (a documented route exists)
  check_ui_api.py      soma/engine/admission panels include nothing in-process
  THIS                 CLI -> scope table    (a CLI verb hits a real route)
                       TUI mutations -> scope table (nothing mutates only in-process)

The gap it closes, concretely: `AgentScheduler::suspend_agent` was PRIVATE, the
node API had `/api/node/suspend-slot`, and no `/v1/*` route could reach either.
An operator holding the entire control API could not suspend an agent, while the
scheduler did it on its own under capacity pressure. `placement_engine.hpp` said
so in a comment and is compiled by nothing, so nothing failed.

WHAT THIS CANNOT DO, stated plainly: it cannot prove a TUI button and a route do
the same thing. It reads what the CLI *calls* and what control_ui *mutates*, and
asserts both land on a registered route. A curated allowlist carries the residue
— the in-process reads (list_agents, list_nodes) that are legitimately reads of
state the process already owns, and the composite handlers the TUI reimplements.
That residue is listed, not hidden, so it shrinks on purpose rather than by
accident.

Usage: check_surface_parity.py [repo_root]
"""
import re
import sys
from pathlib import Path

SCOPE_TABLE = "src/control/route_scope.cpp"
CLI = "src/control/main.cpp"
TUI = "src/control/control_ui.cpp"

# {"GET", "/v1/agents/:id", Scope::Read},
ROUTE_RE = re.compile(r'\{"(GET|POST|PUT|DELETE|PATCH)",\s*"([^"]+)"')

# self.get("/v1/..."), self.post("/v1/..." , ...), self.stream_post("/v1/...", ...
CLI_CALL_RE = re.compile(r'self\.(get|post|put|del|stream_post)\(\s*"([^"]+)"')
# Paths the CLI builds by concatenation: "/v1/agents/" + tokens[2] + "/suspend"
CLI_CONCAT_RE = re.compile(r'self\.(get|post|put|del|stream_post)\(\s*"(/v1/[^"]*)"\s*\+')

VERB = {"get": "GET", "post": "POST", "put": "PUT", "del": "DELETE",
        "stream_post": "POST"}

# control_ui.cpp reaches these three references directly. MUTATIONS through them
# bypass the API handler; reads do not carry the same risk, because the handler
# for a read has no side effect the TUI could skip.
# Listed exhaustively rather than only the ones in use: a mutator that appears
# in the TUI later is caught by the "no route mapped" branch instead of passing
# silently because nobody thought to add it here.
MUTATORS = [
    "agents_.create_agent", "agents_.update_agent", "agents_.delete_agent",
    "registry_.add_node", "registry_.remove_node", "registry_.forget_node",
    "registry_.start_pair", "registry_.complete_pair", "registry_.pair_node",
    "scheduler_.release_agent", "scheduler_.suspend_agent",
    "scheduler_.ensure_agent_running",
]

# Each TUI mutation must name the route that does the same job. This is the
# curated residue: the TUI still calls in-process, but the capability is proven
# reachable through /v1, which is what P1 actually requires. Shrinking this list
# by making the TUI an HTTP client is a separate, larger change.
TUI_MUTATION_ROUTES = {
    "agents_.create_agent":          ("POST", "/v1/agents"),
    "agents_.update_agent":          ("PUT", "/v1/agents/:id"),
    "agents_.delete_agent":          ("DELETE", "/v1/agents/:id"),
    "registry_.remove_node":         ("DELETE", "/v1/nodes/:id"),
    "registry_.forget_node":         ("POST", "/v1/nodes/:id/forget"),
    "registry_.start_pair":          ("POST", "/v1/nodes/pair/start"),
    "registry_.complete_pair":       ("POST", "/v1/nodes/pair/complete"),
    "registry_.pair_node":           ("POST", "/v1/nodes/pair/psk"),
    "scheduler_.release_agent":      ("POST", "/v1/agents/:id/release"),
}


def normalize(path: str) -> str:
    """A CLI path prefix -> the scope table's pattern.

    The CLI builds `/v1/agents/` + id + `/suspend`; the table holds
    `/v1/agents/:id/suspend`. Compare on the literal prefix before the first
    interpolation, which is what the CLI source actually contains.
    """
    return path.rstrip("/")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    failures = []

    table_src = (root / SCOPE_TABLE).read_text(encoding="utf-8", errors="replace")
    routes = set(ROUTE_RE.findall(table_src))
    if not routes:
        print(f"FAIL  no routes parsed from {SCOPE_TABLE}")
        return 1

    # ── 1. every CLI call lands on a registered route ─────────────────────────
    cli_src = (root / CLI).read_text(encoding="utf-8", errors="replace")
    cli_calls = set()
    for verb, path in CLI_CALL_RE.findall(cli_src):
        cli_calls.add((VERB[verb], normalize(path)))
    for verb, prefix in CLI_CONCAT_RE.findall(cli_src):
        cli_calls.add((VERB[verb], normalize(prefix)))

    def covered(verb, path):
        if (verb, path) in routes:
            return True
        # A concatenated prefix matches any table route that starts with it, so
        # `/v1/agents/` + id covers `/v1/agents/:id`. Deliberately loose: this
        # check is about a CLI verb reaching the API at all, not about argument
        # shapes, which the server validates anyway.
        return any(v == verb and p.startswith(path) for v, p in routes)

    for verb, path in sorted(cli_calls):
        if not path.startswith("/v1/"):
            continue
        if not covered(verb, path):
            failures.append(f"CLI calls {verb} {path}, which is in no scope-table route")

    # ── 2. every TUI in-process mutation has an equivalent route ──────────────
    tui_src = (root / TUI).read_text(encoding="utf-8", errors="replace")
    for mutator in MUTATORS:
        if mutator + "(" not in tui_src:
            continue
        want = TUI_MUTATION_ROUTES.get(mutator)
        if want is None:
            failures.append(
                f"{TUI} mutates via {mutator}() and this check has no route mapped for it — "
                f"either add the /v1 route or record why it needs none")
            continue
        if want not in routes:
            failures.append(
                f"{TUI} mutates via {mutator}() whose /v1 equivalent "
                f"{want[0]} {want[1]} is NOT in the scope table — that is a TUI-only "
                f"capability, which P1 forbids")

    # ── 3. the mapping table cannot rot ───────────────────────────────────────
    # A mapped mutator that no longer appears in the TUI means the entry is
    # stale. Reported, not fatal: removing a TUI button is progress, and the
    # entry should be dropped in the same change.
    stale = [m for m in TUI_MUTATION_ROUTES if m + "(" not in tui_src]

    for f in failures:
        print("FAIL  " + f)
    for m in stale:
        print(f"note  {m}() is mapped here but no longer used in {TUI}; drop the entry")

    if failures:
        print(f"\n{len(failures)} parity failure(s)")
        return 1

    print(f"OK    {len(cli_calls)} CLI route call(s), "
          f"{len(TUI_MUTATION_ROUTES) - len(stale)} TUI mutation(s) mapped, "
          f"{len(routes)} routes in the scope table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
