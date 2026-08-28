#!/usr/bin/env python3
"""G7 — every Soma panel's data comes from `/v1/*`.

The gate says "no panel reaches into in-process engine state", and adds that this
is grep-verifiable. This is the grep, run as a test, because the alternative is a
comment asking nicely — and the temptation is live: the existing TUI already holds
`NodeRegistry&`, `AgentManager&` and `AgentScheduler&`, so a panel that needs one
number is four keystrokes from taking it directly.

Why it matters beyond tidiness: a TUI that reads private state is a second client
with privileges no other client has. P1 says the API is the single control plane,
and a dashboard that quietly bypasses it makes that claim untestable — every gap
in `/v1/*` stays invisible for exactly as long as the only serious consumer does
not need the API.

Rule: the Soma dashboard TU and its header may include the HTTP client, JSON, the
standard library, and their own header. Nothing that carries engine, node, agent
or registry state.

Usage: check_ui_api.py [repo_root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The files under the rule. Adding a panel means adding it here — deliberately,
# so the choice to put UI code outside the rule is visible in a diff.
GUARDED = [
    "include/control/soma_dashboard.hpp",
    "src/control/soma_dashboard.cpp",
    "include/control/soma_panels.hpp",
    "src/control/soma_panels.cpp",
    "include/control/engine_panels.hpp",
    "src/control/engine_panels.cpp",
    "include/control/admission_panels.hpp",
    "src/control/admission_panels.cpp",
]

# Includes that carry in-process state. Matched on the include PATH, so a header
# that merely mentions one of these words in a comment is not a violation.
FORBIDDEN = [
    ("node_registry.hpp", "the live node list and its health poll"),
    ("agent_manager.hpp", "agent records and their databases"),
    ("agent_scheduler.hpp", "placement state"),
    ("agent_queue.hpp", "in-flight request queues"),
    ("model_registry.hpp", "control.db"),
    ("engine_supervisor.hpp", "node-side engine state"),
    ("engine_process.hpp", "a live subprocess"),
    ("engine_descriptor.hpp", "the engine registry"),
    ("node_state.hpp", "node-side state"),
    ("slot_manager.hpp", "the removed slot manager"),
    ("performance_tracker.hpp", "in-process metrics"),
    ("control_api_server.hpp", "the server it is supposed to be a client of"),
    # soma/ is the ENGINE. Control links none of it, and a dashboard that did
    # would be reading the model rather than the API describing it.
    ("soma/", "the engine itself"),
]

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.MULTILINE)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    failures = 0
    checked = 0

    for rel in GUARDED:
        path = root / rel
        if not path.exists():
            print(f"FAIL  {rel}: guarded file does not exist")
            failures += 1
            continue

        text = path.read_text(encoding="utf-8")
        includes = INCLUDE_RE.findall(text)
        checked += 1
        bad = []
        for inc in includes:
            for needle, why in FORBIDDEN:
                if needle in inc:
                    bad.append((inc, why))
        if bad:
            failures += 1
            print(f"FAIL  {rel}")
            for inc, why in bad:
                print(f"        #include \"{inc}\"  - {why}")
                print("        A Soma panel's data comes from /v1/*. If the API cannot answer")
                print("        this, the missing route is the bug; reaching around it hides one.")
        else:
            print(f"OK    {rel}  ({len(includes)} includes, none carry in-process state)")

    # The rule is worth nothing if it guards a file nobody uses. A dashboard TU
    # that no target compiles would pass every check here while shipping nothing.
    cml = root / "src" / "control" / "CMakeLists.txt"
    cml_text = cml.read_text(encoding="utf-8") if cml.exists() else ""
    for rel in GUARDED:
        if not rel.endswith(".cpp"):
            continue
        name = Path(rel).name
        if name in cml_text:
            print(f"OK    {name} is actually built")
        else:
            print(f"FAIL  {name} is not in src/control/CMakeLists.txt - the rule would")
            print("        be guarding a file that never compiles")
            failures += 1

    # The engine form's conditional is part of the UI contract: a stale vLLM
    # profile may remain in JSON, but it must not reveal vLLM/Ray controls until
    # vLLM occupies either routing position. Keep this as a source gate because
    # FormState is intentionally private to the FTXUI component.
    engine_panel = (root / "src/control/engine_panels.cpp").read_text(encoding="utf-8")
    ui_needles = [
        'return primary == "vllm" || backup == "vllm";',
        'Maybe(vllm_container, &form->show_vllm)',
        'Ray: automatic when pipeline parallelism > 1',
    ]
    for needle in ui_needles:
        if needle in engine_panel:
            print(f"OK    engine form contract: {needle}")
        else:
            print(f"FAIL  engine form contract missing: {needle}")
            failures += 1

    cli = (root / "src/control/main.cpp").read_text(encoding="utf-8")
    for flag in ("--vllm-install-method", "--vllm-version", "--vllm-tp",
                 "--vllm-pp", "--vllm-experimental-gloo"):
        if flag in cli:
            print(f"OK    CLI exposes {flag}")
        else:
            print(f"FAIL  CLI does not expose {flag}")
            failures += 1

    print(f"\n{checked} files checked, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
