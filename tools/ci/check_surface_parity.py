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
                       node TUI -> node API  (same rule, one tier down)

The node half was added after the control half missed it for a release: the
llama.cpp provisioning wizard existed only on the node's screen, its five API
routes had no caller anywhere, and control's Engines tab could configure a
cluster it could not repair. Watching control alone and calling that "every
in-process mutation" is the same shape of mistake as watching three references
and calling that every channel.

The gap it closes, concretely: `AgentScheduler::suspend_agent` was PRIVATE, the
node API had `/api/node/suspend-slot`, and no `/v1/*` route could reach either.
An operator holding the entire control API could not suspend an agent, while the
scheduler did it on its own under capacity pressure. A design header said so in a
comment and was compiled by nothing, so nothing failed. This check exists so the
claim lives somewhere a build can refuse it; that header is gone (roadmap D46).

WHAT THIS CANNOT DO, stated plainly and kept current as each limit is found:

  * It cannot prove a button and a route do the same thing. It proves a surface
    NAMES a route, not that it drives it correctly.
  * It matches path segments INDEPENDENTLY, so sibling routes cover for each
    other. Breaking `/curation/proposals` was not caught, because
    `/curation/proposals/apply` still supplied both segments. Measured, not
    theorised — a mutation test found it. Removing the only user of a segment
    IS caught: breaking `/local-memories` failed all four of its routes.
  * Reachability is not execution. Disabling a command with `if (false && ...)`
    leaves its literals in place and passes.

The residue that carries what the heuristics cannot: CLI_COVERAGE_EXEMPT,
TUI_COVERAGE_EXEMPT, TUI_INPROCESS_READS and TUI_MUTATION_ROUTES. All four are
listed rather than hidden, each entry with a reason, so they shrink on purpose
rather than by accident.

Usage: check_surface_parity.py [repo_root]
"""
import re
import sys
from pathlib import Path

SCOPE_TABLE = "src/control/route_scope.cpp"
CLI = "src/control/main.cpp"
TUI = "src/control/control_ui.cpp"

# The TUI is not one file. control_ui.cpp holds the tabs that reach NodeRegistry,
# AgentManager and AgentScheduler in-process; the guarded panels are HTTP clients
# of /v1 like any other. Coverage has to read both, or the Soma, Engines and
# Admissions tabs look like they do nothing.
TUI_FILES = [
    "src/control/control_ui.cpp",
    "src/control/soma_dashboard.cpp",
    "src/control/soma_panels.cpp",
    "src/control/engine_panels.cpp",
    "src/control/admission_panels.cpp",
]

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
    # A FOURTH channel, and the one this check originally missed entirely.
    #
    # The Curation tab does not go through AgentManager — it takes an agent
    # handle and reaches `a->db()`, the raw AgentDB. So `set_active_conversation`,
    # `create_conversation`, `delete_conversation` and `delete_memory` were
    # mutations the TUI performed that check 2 never watched, because it only
    # knew about agents_/registry_/scheduler_. Watching three references and
    # calling that "every in-process mutation" is the same shape of mistake as
    # a gate registered for one architecture.
    #
    # These bypass ConversationManager and MemoryManager as well as the API
    # handler, so the drift risk is two layers deep rather than one.
    "db().set_active_conversation", "db().create_conversation",
    "db().delete_conversation", "db().delete_memory",
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
    "scheduler_.suspend_agent":      ("POST", "/v1/agents/:id/suspend"),
    # restore IS ensure_agent_running — its first step is existing/suspended
    # placement, so the Restore button and POST .../restore make the same call.
    "scheduler_.ensure_agent_running": ("POST", "/v1/agents/:id/restore"),
    "db().set_active_conversation":  ("POST", "/v1/agents/:id/conversations/:cid/activate"),
    "db().create_conversation":      ("POST", "/v1/agents/:id/conversations"),
    "db().delete_conversation":      ("DELETE", "/v1/agents/:id/conversations/:cid"),
    "db().delete_memory":            ("DELETE", "/v1/agents/:id/memories/:mid"),
}


# Routes the TUI serves by reading state in-process rather than over HTTP. The
# capability IS on the screen; the path it took is control_ui.cpp holding the
# reference directly. Listed so the coverage count reflects what an operator can
# see, without pretending an HTTP call happened.
TUI_INPROCESS_READS = {
    "/v1/agents",            # Agents tab list          — agents_.list_agents()
    "/v1/nodes",             # Nodes tab                — registry_.list_nodes()
    "/v1/nodes/discovered",  # Nodes tab, discovery     — get_discovered_nodes()
    "/v1/activity",          # Activity tab             — the in-process log deque
    "/v1/placements",        # Nodes/Agents slot columns
}

# Routes with no TUI form ON PURPOSE, each with a reason. Kept short and
# specific for the same reason the CLI list is: every entry should be a decision
# someone made, not something nobody got to.
TUI_COVERAGE_EXEMPT = {
    ("POST", "/v1/chat/completions"):
        "OpenAI-compat request shape; the Chat tab covers the capability",
    ("GET", "/v1/tokens"): "credential administration is a CLI/API task, not a dashboard one",
    ("POST", "/v1/tokens"): "the plaintext is shown once and must be captured, not glanced at",
    ("DELETE", "/v1/tokens/:id"): "see tokens.create",
    ("POST", "/v1/models"): "registering a pre-converted container is a scripted deploy step",
    ("GET", "/v1/engines/:id/telemetry"):
        "consumed as a live SSE stream by the Soma tab's dashboard, not as a route it names",
    ("GET", "/v1/agents/:id/attachments/:attachment_id"): "binary image bytes",
    ("DELETE", "/v1/agents/:id/attachments/:attachment_id"):
        "attachment lifecycle follows its conversation",
    ("POST", "/v1/agents/:id/attachments"): "multipart upload; the Chat tab covers text",
    ("GET", "/v1/agents/:id/speech/cache/:cache_id"): "binary audio, fetched by the player",
    ("POST", "/v1/audio/speech"): "OpenAI-compat spelling of the Voice tab's synthesis",
    ("POST", "/v1/cluster/engines/nodes/:node_id/switch"):
        "picking a build variant needs that node's troubleshooting report — release "
        "assets, CUDA architectures, a compile assessment — and none of it crosses "
        "the conformance route control polls. A variant menu control cannot populate "
        "is a list whose every entry might fail; the Engines tab offers Retry instead "
        "and the choice stays on the node's own screen until a route carries the report",
}

# ── The node's TUI is a second control plane, and nothing watched it ──────────
#
# Everything above measures control. The node has its own FTXUI screen, its own
# API, and — until this list existed — its own private capabilities: the
# llama.cpp troubleshooting wizard was wired to in-process callbacks and reached
# five node routes that NO client ever called. control could not provision, could
# not diagnose, could not retry a failed build. An operator holding the entire
# cluster API had to open a session on the machine.
#
# That is the same defect the control half of this file was written for, one tier
# down, and it passed CI for the same reason: nothing looked. The rule is the
# rule P1 states for control, applied to the node — if the node's screen can do
# it, an API client can do it through the same route.
#
# Mapped to /api/node/* rather than /v1/*: the node's own API is its control
# plane. Control reaching those routes is a separate property, and the /v1
# coverage checks above already carry it.
NODE_TUI = "src/node/node_ui.cpp"
NODE_API = "src/node/node_api_server.cpp"

# `server_->Post("/api/node/engines/:id/provision", ...)`
NODE_ROUTE_RE = re.compile(r'server_->(Get|Post|Put|Delete|PostUpload)\(\s*"(/api/node/[^"]+)"')

# The node TUI mutates by calling these callbacks. Each names the /api/node
# route that performs the same action, so a button with no route fails here
# rather than becoming a node-only capability nobody can automate.
NODE_TUI_MUTATION_ROUTES = {
    # One callback, five wizard actions, all of them recovery verbs on one
    # engine — so they share a route that takes the verb in its body. The
    # exception is "target", which the node maps onto the same recover call.
    'request_llama_recovery_cb_': ("POST", "/api/node/engines/:id/recover"),
    'request_llama_update_cb_':   ("POST", "/api/node/engines/:id/provision"),
    'request_llama_switch_cb_':   ("POST", "/api/node/engines/:id/switch"),
    # Clearing remembered pairings is local credential hygiene on that machine.
    # Deliberately NOT an API route: a route that lets a caller drop the keys it
    # authenticated with is a way to lock an operator out remotely.
    'forget_pairing_cb_': None,
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

    # ── 1c. every route has a TUI form, or a recorded reason ──────────────────
    #
    # The third direction, and the one nothing measured. Check 2 below asks the
    # narrow question "does each TUI MUTATION have a route" — it proves the TUI
    # introduces no capability the API lacks. It says nothing about the reverse:
    # capabilities the API has that an operator cannot reach from the screen.
    #
    # Read from the same two signals the TUI actually uses: /v1 literals in the
    # guarded panels, and the in-process calls in control_ui.cpp.
    tui_src_all = "\n".join(
        (root / f).read_text(encoding="utf-8", errors="replace")
        for f in TUI_FILES if (root / f).exists())
    tui_paths = {p.split("?")[0] for p in CLI_ANY_PATH_RE.findall(tui_src_all)}
    tui_literals = set(re.findall(r'"([^"\\\n]{1,64})"', tui_src_all))
    # An in-process mutation reaches its mapped route just as surely as an HTTP
    # call does — the operator gets the capability either way, which is what
    # this direction measures.
    #
    # Counted DIRECTLY as (verb, path) rather than fed through the path matcher.
    # The map is already proof; re-deriving it from literals reported
    # POST /v1/nodes/:id/forget as a gap because the TUI's button says "Forget
    # Pairing" and the matcher wanted a lowercase "forget" segment. A check that
    # doubts its own evidence produces noise, and noise is what gets a check
    # switched off.
    tui_direct = {mapped for mapped in TUI_MUTATION_ROUTES.values()}
    # Reads the TUI performs in-process, mapped to the route that serves them.
    for path in TUI_INPROCESS_READS:
        tui_paths.add(path)

    def surface_knows(path: str, paths: set, literals: set) -> bool:
        segments = [s for s in path.split("/") if s]
        head_parts, tail_parts, seen_param = [], [], False
        for s in segments:
            if s.startswith(":"):
                seen_param = True
                continue
            (tail_parts if seen_param else head_parts).append(s)
        head = "/" + "/".join(head_parts)
        if not seen_param:
            return head in paths or head + "/" in paths
        if not any(c == head or c.startswith(head + "/") for c in paths):
            return False

        def segment_present(t: str) -> bool:
            for lit in literals:
                if lit == t or ("/" + t + "/") in lit or lit.endswith("/" + t):
                    return True
            return False

        return all(segment_present(t) for t in tail_parts)

    tui_uncovered = []
    for verb, path in sorted(routes):
        if (verb, path) in TUI_COVERAGE_EXEMPT or (verb, path) in tui_direct:
            continue
        if not surface_knows(path, tui_paths, tui_literals):
            tui_uncovered.append((verb, path))

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

    # ── 2b. every node-TUI mutation has an /api/node route ────────────────────
    node_tui_path = root / NODE_TUI
    node_api_path = root / NODE_API
    if node_tui_path.exists() and node_api_path.exists():
        node_tui_src = node_tui_path.read_text(encoding="utf-8", errors="replace")
        # Upper-cased on the way in: httplib spells the verb `Post`, every
        # table in this file spells it `POST`, and comparing the two silently
        # reported every route missing.
        node_routes = {(v.upper().replace("UPLOAD", ""), path)
                       for v, path in NODE_ROUTE_RE.findall(
                           node_api_path.read_text(encoding="utf-8", errors="replace"))}
        if not node_routes:
            failures.append(f"no /api/node routes parsed from {NODE_API}")
        for cb, want in NODE_TUI_MUTATION_ROUTES.items():
            if cb + "(" not in node_tui_src:
                continue
            if want is None:
                continue
            if want not in node_routes:
                failures.append(
                    f"{NODE_TUI} mutates via {cb}() whose node-API equivalent "
                    f"{want[0]} {want[1]} is NOT registered — that is a node-TUI-only "
                    f"capability, which is how provisioning became unreachable from control")
        # The other direction: a callback the node TUI grew that this list has
        # never heard of. Caught here rather than at the next audit, which is the
        # whole difference between a check and a comment.
        for m in sorted(set(re.findall(
                r'\b(request_[a-z_]+_cb_|forget_[a-z_]+_cb_)\(', node_tui_src))):
            if m not in NODE_TUI_MUTATION_ROUTES:
                failures.append(
                    f"{NODE_TUI} calls {m}() and this check has no node-API route mapped "
                    f"for it — either add the route or record why it needs none")

    # ── 2c. no SSE consumer re-strips the `data: ` prefix ─────────────────────
    #
    # Here because this file's subject is operator surfaces that actually work,
    # and this is the way one stopped: `HttpClient`'s line callback is handed the
    # PAYLOAD — `util::drain_sse_lines` removed `data: ` and dropped the
    # keepalives — but the header comment claimed it got "each raw `data: ...`
    # line". Four starters believed it, searched for a prefix that is never
    # there, captured nothing, and reported "admission started but reported no
    # operation id" about admissions that had started correctly (D68).
    #
    # A grep, and honest about being one: it cannot tell a correct consumer from
    # an incorrect one in general. What it CAN do is catch the exact literal that
    # produced four identical bugs, in the files where an operator surface talks
    # to a stream. `HttpClient::capture_first_field` is the shared helper that
    # should make writing this by hand unnecessary.
    SSE_CONSUMERS = TUI_FILES + [CLI]
    for rel in SSE_CONSUMERS:
        f = root / rel
        if not f.exists():
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace")
                                 .splitlines(), 1):
            if 'find("data:")' in line or 'rfind("data:"' in line:
                failures.append(
                    f"{rel}:{n} searches an SSE payload for a `data:` prefix that "
                    f"drain_sse_lines already removed — the payload IS the JSON. Use "
                    f"HttpClient::capture_first_field, or parse it directly")

    # ── 3. the mapping table cannot rot ───────────────────────────────────────
    # A mapped mutator that no longer appears in the TUI means the entry is
    # stale. Reported, not fatal: removing a TUI button is progress, and the
    # entry should be dropped in the same change.
    stale = [m for m in TUI_MUTATION_ROUTES if m + "(" not in tui_src]

    for verb, path in tui_uncovered:
        # FATAL, now that this direction is also at zero — the same ratchet the
        # CLI half went through. A check stays advisory exactly as long as a
        # pre-existing backlog makes failing it unfair, and no longer: with
        # every route reachable or exempt-with-a-reason, a new gap is a new
        # decision, and the cheapest moment to make it is when the route lands.
        failures.append(
            f"{verb} {path} has no TUI form and no recorded exemption — add a control, "
            f"or an entry in TUI_COVERAGE_EXEMPT stating why it needs none")
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
    tui_covered_n = len(routes) - len(tui_uncovered) - len(TUI_COVERAGE_EXEMPT)
    print(f"OK    {len(cli_calls)} CLI route call(s), "
          f"{len(TUI_MUTATION_ROUTES) - len(stale)} TUI mutation(s) mapped, "
          f"{len(routes)} routes in the scope table")
    print(f"      CLI coverage: {covered_n} reachable, "
          f"{len(CLI_COVERAGE_EXEMPT)} exempt by reason, {len(uncovered)} gap(s)")
    print(f"      TUI coverage: {tui_covered_n} reachable, "
          f"{len(TUI_COVERAGE_EXEMPT)} exempt by reason, {len(tui_uncovered)} gap(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
