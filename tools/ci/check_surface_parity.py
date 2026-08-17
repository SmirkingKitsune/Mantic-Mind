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
# Every /v1 literal ANYWHERE in the CLI, verb unknown.
#
# The two patterns above only see a path written inline at the call. The CLI also
# builds paths into a variable first — `std::string path = "/v1/activity?tail=" +
# n; ... self.get(path)` — which both miss entirely. That blind spot made
# /v1/activity look uncovered in a hand-audit and, worse, would have let a CLI
# call to a NON-EXISTENT route pass unnoticed, because an unseen call cannot
# fail the check.
#
# Used only for the COVERAGE direction, where a verb-less "the CLI knows this
# path" is exactly the right signal.
CLI_ANY_PATH_RE = re.compile(r'"(/v1/[^"]*)"')

# Routes with no CLI form ON PURPOSE. Each needs a reason, because the point of
# the list is that it stays short and every entry is a decision someone made
# rather than something nobody got to.
CLI_COVERAGE_EXEMPT = {
    ("GET", "/v1/engines/:id/telemetry"):
        "an unbounded SSE stream; a REPL has no shape for it that is not invented",
    ("POST", "/v1/chat/completions"):
        "the OpenAI-compat request shape for the SAME capability `chat send` covers",
    ("POST", "/v1/audio/speech"):
        "returns binary audio; `curation`/`chat` cover the text path and a REPL "
        "cannot usefully render a WAV",
    ("POST", "/v1/agents/:id/attachments"):
        "multipart upload of an image; covered by `chat send` for text",
    ("GET", "/v1/agents/:id/attachments/:attachment_id"):
        "returns binary image bytes",
    ("DELETE", "/v1/agents/:id/attachments/:attachment_id"):
        "attachment lifecycle follows its conversation",
    ("GET", "/v1/agents/:id/speech/cache/:cache_id"):
        "returns binary audio",
    ("POST", "/v1/agents/:id/speech"):
        "returns binary audio",
}

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

    # ── 1b. every route has a CLI form, or a recorded reason ──────────────────
    #
    # The other direction, and the one that actually measures parity. Check 1
    # asks "does this CLI verb hit a real route"; this asks "can an operator do
    # from the CLI what the API can do". Sixteen routes had no CLI form when
    # this was first measured, including all three token routes — so a headless
    # deployment could not mint its first credential, and the scoped-auth system
    # was unreachable on exactly the deployments it was built for.
    cli_paths = {p.split("?")[0] for p in CLI_ANY_PATH_RE.findall(cli_src)}
    # Every string literal in the CLI, because a path suffix is frequently a
    # COMMAND WORD rather than part of the URL literal:
    #     self.post("/v1/agents/" + tokens[2] + "/" + sub, ...)   // sub=="suspend"
    # There is no "/suspend" literal anywhere; there is a `sub == "suspend"`.
    # Matching only URL literals reported routes as uncovered that the CLI had
    # reached since the previous commit.
    cli_literals = set(re.findall(r'"([^"\\\n]{1,64})"', cli_src))

    def cli_knows(path: str) -> bool:
        segments = [s for s in path.split("/") if s]
        head_parts, tail_parts, seen_param = [], [], False
        for s in segments:
            if s.startswith(":"):
                seen_param = True
                continue
            (tail_parts if seen_param else head_parts).append(s)
        head = "/" + "/".join(head_parts)

        # No parameters: the whole path must appear as a literal.
        if not seen_param:
            return head in cli_paths or head + "/" in cli_paths

        # Parameterised: the CLI must name this resource AND supply every
        # literal segment that follows a parameter — as part of a URL or as a
        # command word.
        names_resource = any(c == head or c.startswith(head + "/") for c in cli_paths)
        if not names_resource:
            return False
        # Each remaining segment must appear as a WHOLE token — either a bare
        # command word (`sub == "suspend"`) or a complete path segment in a
        # fragment the CLI concatenates (`"/conversations/"`).
        #
        # Whole-token, not substring, and the difference is the whole value of
        # this check. A plain `t in lit` test reported `PUT /v1/agents/:id/
        # backend` as covered after its command had been DELETED, because
        # "backend" is a substring of the unrelated literal "backend_override".
        # A checker that reports coverage which does not exist is worse than no
        # checker: it converts an unknown into a false assurance.
        def segment_present(t: str) -> bool:
            for lit in cli_literals:
                if lit == t:
                    return True
                if ("/" + t + "/") in lit or lit.endswith("/" + t):
                    return True
            return False

        return all(segment_present(t) for t in tail_parts)

    uncovered = []
    for verb, path in sorted(routes):
        if (verb, path) in CLI_COVERAGE_EXEMPT:
            continue
        if not cli_knows(path):
            uncovered.append((verb, path))

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

    for verb, path in uncovered:
        # FATAL, now that the backlog is zero.
        #
        # This was advisory while a backlog existed — failing the build for gaps
        # that predate the check would only have taught people to disable it.
        # With every route either reachable or exempt-with-a-reason, a new gap is
        # a NEW decision, and the right moment to make it is when the route is
        # added rather than at the next audit. Adding a route now costs one of:
        # a CLI verb, or one line in CLI_COVERAGE_EXEMPT saying why not.
        failures.append(
            f"{verb} {path} has no CLI form and no recorded exemption — add a CLI "
            f"verb, or an entry in CLI_COVERAGE_EXEMPT stating why it needs none")
    for f in failures:
        print("FAIL  " + f)
    for m in stale:
        print(f"note  {m}() is mapped here but no longer used in {TUI}; drop the entry")

    if failures:
        print(f"\n{len(failures)} parity failure(s)")
        return 1

    covered_n = len(routes) - len(uncovered) - len(CLI_COVERAGE_EXEMPT)
    print(f"OK    {len(cli_calls)} CLI route call(s), "
          f"{len(TUI_MUTATION_ROUTES) - len(stale)} TUI mutation(s) mapped, "
          f"{len(routes)} routes in the scope table")
    print(f"      CLI coverage: {covered_n} reachable, "
          f"{len(CLI_COVERAGE_EXEMPT)} exempt by reason, {len(uncovered)} gap(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
