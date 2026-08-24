#!/usr/bin/env python3
"""Seam falsification check for the Soma engine.

The seam between the architecture-invariant core and the per-architecture
backends is a structural requirement, not a style preference. This script makes
it a mechanical one.

Two rules:

  R1  No core file includes "soma/arch/...". Only the arch backends themselves,
      the single arch_registry.cpp resolver TU, and the tests may.

  R2  Architecture-specific identifiers appear in core CODE nowhere except
      arch_ir.hpp.

R2 needs care, and the distinction is the whole point of the design:

  * arch_ir.hpp is a DESCRIPTION of an architecture. It has to be able to say
    "this model is MLA with kv_lora_rank 512" — that is data passing through the
    core to a backend, not the core knowing how to execute MLA. Allow-listed.

  * Everything else in include/soma/ and src/soma/ is core LOGIC. An architecture
    name appearing there means a branch on architecture has leaked into a place
    that should be family-agnostic.

Comments and string literals are stripped before matching, so an explanatory
comment ("MLA folds up-projections into Q; GQA has no analogue") does not trip
the check. Explaining why an interface is shaped a certain way is the opposite of
a leak.

Usage:  python tools/ci/check_seam.py [repo_root]
Exit:   0 clean, 1 violations found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Identifiers that name a specific ARCHITECTURE, attention family, or reference
# checkpoint.
#
# Deliberately NOT on this list: softmax, sigmoid, swiglu, geglu, relu2, situ,
# rope,
# rmsnorm. Those are math primitives, not architectures. Kernels ARE core — a
# rule that forbade `softmax` in core would mean the invariant core could not
# contain a kernel, which is absurd. The variant a family *selects* is IR data;
# the operation itself is shared.
#
# The distinction that matters: does the identifier name a thing that EXECUTES
# differently per family (mla vs gqa attention, weight absorption), or a thing
# every family shares and merely parameterizes (softmax, rope)?
ARCH_TOKENS = [
    # attention families
    "mla", "gqa", "mha", "dsa", "kda", "gdn",
    # family-specific mechanisms with no cross-family analogue
    "absorb", "lora", "yarn",
    # reference checkpoints / model families
    "deepseek", "qwen", "mixtral", "olmoe", "granite", "gptoss", "llama",
    "kimi",
]

_ARCH_TOKEN_SET = frozenset(ARCH_TOKENS)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Split an identifier into its parts: snake_case on underscores, camelCase and
# PascalCase on case transitions, and runs of caps (HTTPServer -> http, server).
_PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z]+|[0-9]+")


def identifier_parts(ident: str):
    """Yield the lowercase sub-tokens of an identifier.

    Word-boundary matching is NOT sufficient here, and getting that wrong makes
    the check useless in the exact case it exists for: `\\bmla` does not match
    inside `is_mla_family`, because `_` is a word character. A leak is far more
    likely to be named `is_mla_family` or `use_gqa_path` than bare `mla`.

    Substring matching would catch those but invites false positives. Splitting
    the identifier and comparing parts exactly does both jobs.
    """
    for part in ident.split("_"):
        for m in _PART_RE.finditer(part):
            yield m.group(0).lower()


def arch_hits(line: str):
    """Architecture identifiers appearing in one line of code."""
    for m in _IDENT_RE.finditer(line):
        ident = m.group(0)
        for part in identifier_parts(ident):
            if part in _ARCH_TOKEN_SET:
                yield ident, part
                break

# Files permitted to include soma/arch/ (R1).
ARCH_INCLUDE_ALLOWED = (
    "include/soma/arch/",
    "src/soma/arch/",
    "src/soma/arch_registry.cpp",   # the ONE resolver TU
    "tests/soma/",
    "tools/ci/",
)

# Files permitted to name an architecture in code (R2).
ARCH_TOKEN_ALLOWED = ARCH_INCLUDE_ALLOWED + (
    "include/soma/arch_ir.hpp",     # the IR is a description, not logic
    "src/soma/arch_ir.cpp",         # the per-family ADAPTER — see below
)

# arch_ir.cpp is allow-listed for the same reason as its header, and the reason
# is worth stating: the adapter's entire job is knowing that upstream spells the
# same concept three ways, that a family's qk-norm form is implied by model_type
# rather than declared, and that an omitted key takes a family-specific default.
# Forbidding family names there would forbid the adapter from doing its job.
#
# What it may NOT do — and what R1 still catches — is include soma/arch/ or
# dispatch execution. It maps names to IR data; arch_registry.cpp maps IR data to
# code.

SCAN_ROOTS = ("include/soma", "src/soma")
SCAN_SUFFIXES = (".hpp", ".h", ".cpp", ".cc", ".inl")


def strip_comments_and_strings(text: str) -> str:
    """Blank out // and /* */ comments and "..." / '...' literals.

    Replaces with spaces rather than deleting so line numbers survive.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        elif c == "/" and nxt == "*":
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i = min(i + 2, n)
        elif c in ('"', "'"):
            quote = c
            out.append(" ")
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    out.append("  ")
                    i += 2
                    continue
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            out.append(" ")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_allowed(rel: str, allow: tuple[str, ...]) -> bool:
    return any(rel.startswith(prefix) for prefix in allow)


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else Path(__file__).resolve().parents[2]

    files: list[Path] = []
    for scan_root in SCAN_ROOTS:
        base = root / scan_root
        if not base.is_dir():
            continue
        files.extend(p for p in base.rglob("*") if p.suffix in SCAN_SUFFIXES and p.is_file())

    if not files:
        print(f"check_seam: no Soma sources found under {root}; nothing to check.")
        return 0

    violations: list[str] = []

    for path in sorted(files):
        rel = rel_posix(path, root)
        raw = path.read_text(encoding="utf-8", errors="replace")
        code = strip_comments_and_strings(raw)

        # R1 — include discipline. Checked against the RAW text, since an
        # #include is never inside a comment we care about.
        if not is_allowed(rel, ARCH_INCLUDE_ALLOWED):
            for lineno, line in enumerate(raw.splitlines(), 1):
                if re.search(r'#\s*include\s*[<"]soma/arch/', line):
                    violations.append(
                        f"{rel}:{lineno}: R1 core file includes soma/arch/ "
                        f"-> {line.strip()}"
                    )

        # R2 — no architecture names in core code.
        if not is_allowed(rel, ARCH_TOKEN_ALLOWED):
            for lineno, line in enumerate(code.splitlines(), 1):
                for ident, part in arch_hits(line):
                    violations.append(
                        f"{rel}:{lineno}: R2 architecture identifier "
                        f"'{ident}' (matched '{part}') in core code"
                    )

    if violations:
        print("SEAM VIOLATIONS")
        print("=" * 70)
        for v in violations:
            print(v)
        print("=" * 70)
        print(f"{len(violations)} violation(s).")
        print()
        print("The invariant core must not learn about architectures. Move the")
        print("logic into include/soma/arch/ + src/soma/arch/, or express the")
        print("difference through the F32Backend / AttentionBackend function")
        print("pointers. See docs/architecture.md §2 and §11.")
        return 1

    print(f"check_seam: OK — {len(files)} file(s) scanned, seam intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
