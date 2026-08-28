#!/usr/bin/env python3
"""Every test the build DECLARES must be a test ctest actually runs.

Roadmap D44: the suite reported 29 tests while 31 were registered. `ctest` reads
the generated CTestTestfile.cmake in the build directory, not CMakeLists.txt, and
that copy had gone stale — so `soma_checkpoint_mla_g3` and `soma_checkpoint_dsa_g3`
were defined and never ran. They were the MLA and DSA KV-checkpoint round-trips,
which is to say the two most relevant tests for the change being validated at the
time, and "29/29 passed" read exactly like a full green run.

A suite that silently shrinks is the same failure this project keeps finding in
other clothes: a gate that is structurally present and does nothing reads as
covered. The count is the only outward signal, and nobody remembers last week's
count.

WHY THIS CAN CHECK ITSELF. Staleness manifests as new tests MISSING, never as old
ones disappearing — a stale file is simply an older complete file. So this check,
once registered, keeps running and keeps seeing whatever was added after it. The
day it stops appearing in `ctest -N` is the day something worse than staleness
happened.

Usage:
    check_test_registry.py <source_dir> <build_dir>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# add_test(NAME foo COMMAND ...) — the form every CMakeLists here uses.
DECLARED = re.compile(r"add_test\s*\(\s*NAME\s+([A-Za-z0-9_.\-]+)")

# Depending on the generator, CMake writes either add_test("foo" ...) or its
# bracket-quoted equivalent add_test([=[foo]=] ...). Ubuntu's single-config
# generators use the latter while Visual Studio uses the former.
REGISTERED = re.compile(
    r'^\s*add_test\s*\(\s*(?:"([^"]+)"|\[(=*)\[([A-Za-z0-9_.\-]+)\]\2\])'
)


def declared_tests(source: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for f in source.rglob("CMakeLists.txt"):
        # Skip anything vendored: vcpkg trees, build outputs and the admission
        # venvs carry their own tests that this project neither declares nor
        # runs. Match venvs by prefix so a new one is covered on arrival.
        parts = {p.lower() for p in f.parts}
        if parts & {"build", "cmbuild", "vcpkg_installed"} or any(
            p.startswith(".venv") for p in parts
        ):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name in DECLARED.findall(text):
            out[name] = f
    return out


def registered_tests(build: Path) -> set[str]:
    out: set[str] = set()
    for f in build.rglob("CTestTestfile.cmake"):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            m = REGISTERED.match(line)
            if m:
                out.add(m.group(1) or m.group(3))
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("  skipped  usage: check_test_registry.py <source_dir> <build_dir>")
        return 77

    source, build = Path(argv[1]), Path(argv[2])
    declared = declared_tests(source)
    registered = registered_tests(build)

    if not declared:
        print(f"  FAILED  no add_test(NAME ...) found under {source}; the pattern has changed "
              f"and this check is now blind")
        return 1
    if not registered:
        print(f"  FAILED  no CTestTestfile.cmake under {build}; nothing to compare against")
        return 1

    missing = sorted(set(declared) - registered)
    if missing:
        print(f"  FAILED  {len(missing)} test(s) declared but NOT registered with ctest:")
        for name in missing:
            print(f"             {name}   ({declared[name].relative_to(source)})")
        print("           The build directory's generated test list is stale. Re-run cmake:")
        print(f"             cmake {build}")
        print("           Until then the suite passes while running fewer tests than it defines,")
        print("           which is indistinguishable from a full green run (roadmap D44).")
        return 1

    # Extra registered tests are NOT an error: a build directory may legitimately
    # carry tests from a configuration this source tree no longer declares, and
    # refusing that would make the check fail for a reason nobody can act on.
    print(f"  OK       {len(declared)} declared test(s), all registered with ctest")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
