#!/usr/bin/env python3
"""Every route the API document describes is a route that exists.

`require_complete_coverage()` already checks the other direction — every
REGISTERED handler has a scope-table entry, asserted at startup. Nothing walked
this way, and the gap is exactly the shape of a defect that survived: `GET
/v1/models/{id}/conformance` was documented from the day the API surface was
written and never registered. It answered 404 for its entire life, and the only
reason anyone noticed was someone reading the docs and calling what they said.

A documented route that does not exist is worse than an undocumented one. It is
a promise, and a client written against it fails at runtime with a 404 that looks
like the resource is missing rather than the endpoint.

The chain this closes:

    external-api.md  ->  route_scope.cpp's table   (this script)
    route_scope.cpp  ->  registered handlers       (require_complete_coverage,
                                                    at startup)

Together those are bidirectional. Neither alone is.

Usage: check_api_docs.py [repo_root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOC = "docs/external-api.md"
TABLE = "src/control/route_scope.cpp"

# `### `GET /v1/models/{id}/plan?node_id=<id>` — `read``
DOC_ROUTE = re.compile(r"^###\s+`(GET|POST|PUT|DELETE|PATCH)\s+([^`\s?]+)")
# A heading may mark itself as designed-but-not-built. Those are reported and not
# failed — the point is that the distinction is EXPLICIT, so a reader can tell a
# shipped route from a proposed one without calling it to find out.
PLANNED = re.compile(r"\(planned\)\s*$")
# `{"GET", "/v1/models/:id/plan", Scope::Read},`
TABLE_ROUTE = re.compile(r'\{\s*"(GET|POST|PUT|DELETE|PATCH)"\s*,\s*"([^"]+)"')

# Documented surfaces this control plane deliberately does not serve. Each needs
# a reason, so that "not implemented yet" cannot hide among them.
EXEMPT = {
    # The OpenAI-compat listener is a SEPARATE server on :9091 with its own
    # (flat-token) auth, so its routes are not in this scope table by design.
    ("GET", "/v1/models:9091"): "openai-compat listener",
}


def normalise(path: str) -> str:
    """`{id}` and `:id` are the same parameter written two ways."""
    path = re.sub(r"\{([^}]+)\}", r":\1", path)
    # Parameter NAMES are free to differ between prose and code; only position
    # and count are contractual.
    return re.sub(r":[A-Za-z_][A-Za-z0-9_]*", ":p", path.rstrip("/"))


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    doc, table = root / DOC, root / TABLE
    for f in (doc, table):
        if not f.exists():
            print(f"FAIL  missing {f}")
            return 1

    documented, planned = [], []
    for line in doc.read_text(encoding="utf-8").splitlines():
        m = DOC_ROUTE.match(line)
        if not m:
            continue
        (planned if PLANNED.search(line.strip()) else documented).append(
            (m.group(1), m.group(2)))

    known = {(m.group(1), normalise(m.group(2)))
             for m in TABLE_ROUTE.finditer(table.read_text(encoding="utf-8"))}

    if not documented:
        print("FAIL  parsed no routes out of the document — the heading format changed")
        return 1
    if not known:
        print("FAIL  parsed no routes out of the scope table — its format changed")
        return 1

    failures = 0
    for verb, path in documented:
        key = (verb, normalise(path))
        if key in known or (verb, path) in EXEMPT:
            continue
        failures += 1
        print(f"FAIL  {verb} {path}")
        print(f"        documented in {DOC}, absent from {TABLE}")
        print( "        A documented route that does not exist is a promise the")
        print( "        server does not keep. Register it, or delete the section.")

    for verb, path in planned:
        print(f"planned  {verb} {path}  (documented, deliberately not registered)")

    print(f"\n{len(documented)} shipped routes documented, {len(planned)} planned, "
          f"{len(known)} in the scope table, {failures} unkept")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
