#!/usr/bin/env python3
"""Do the two halves of the conversion record's schema agree?

`container_meta.json` has one schema and two implementations of it: the writer
and validator in `tools/admission/record.py`, and the reader in
`validate_container_record()` in `src/soma/arch_ir.cpp`. They are in different
languages and cannot share a declaration, so the schema is transcribed twice —
which is exactly the kind of repetition that drifts, and exactly the kind the
record's own rule permits only when something checks it.

This is that something. It mutates a known-good record one way at a time and
requires BOTH sides to reach the same verdict about every mutation:

  drop a required field          -> both reject
  drop a discriminated field     -> both reject when its discriminator demands it
  add a field the discriminators say does not apply  -> both reject
  add a field neither knows      -> both reject
  change record_version          -> both reject
  flip a discriminator           -> both reject, because the fields no longer fit

A field added to one side and forgotten on the other shows up here as a
disagreement, which is the failure this file exists to make loud. The specific
history: `dspark_profiled_speedup` was read by C++ and written by nobody for
months, and nothing said so because a missing key and a key that was never meant
to be there looked identical to every reader.

Usage:
    check_container_record.py <repo-root> <soma-executable>
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "admission"))

import record  # noqa: E402  (needs the path above)

FAILURES = 0


def fail(case: str, why: str) -> None:
    global FAILURES
    print(f"  FAILED  {case}: {why}")
    FAILURES += 1


def cpp_accepts(soma: Path, container: Path, meta: dict) -> tuple[bool, str]:
    """Does the ENGINE accept this record?

    Through `soma plan`, which resolves the IR and therefore reads the record,
    and which touches no payload — so a mutation is answered in milliseconds
    rather than by re-reading every expert.
    """
    (container / "container_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # --ctx 512 because the tiny fixtures declare short maximums and the default
    # would fail for a reason that has nothing to do with the record.
    r = subprocess.run([str(soma), "plan", "--model-dir", str(container),
                        "--ctx", "512", "--json"],
                       capture_output=True, text=True)
    lines = (r.stderr or r.stdout).strip().splitlines()
    return r.returncode == 0, lines[-1] if lines else ""


def py_accepts(meta: dict) -> tuple[bool, str]:
    problems = record.validate(meta)
    return not problems, "; ".join(problems)


def mutations(good: dict):
    """One named mutation at a time. Never two, or a disagreement is ambiguous."""
    for name in sorted(good):
        m = dict(good)
        del m[name]
        yield f"drop {name}", m

    # A field whose discriminator says it does not apply. Picked from the schema
    # rather than hardcoded, so a new discriminated field is covered the day it
    # is declared.
    for field, _types, when, _why in record.FIELDS:
        if when == "always" or field in good:
            continue
        m = dict(good)
        m[field] = 0 if field.endswith(("_bytes", "_tensors", "_rank", "_stages")) else "x"
        yield f"add inapplicable {field}", m

    m = dict(good)
    m["a_field_nobody_declared"] = 1
    yield "add unknown field", m

    m = dict(good)
    m["record_version"] = record.RECORD_VERSION + 1
    yield "record_version from the future", m

    m = dict(good)
    m["dense_storage"] = "quantized" if m.get("dense_storage") != "quantized" else "source"
    yield "flip dense_storage", m

    m = dict(good)
    m["dspark"] = "present" if m.get("dspark") != "present" else "omitted"
    yield "flip dspark", m


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    root, soma = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    # BOTH shapes the discriminators produce. A base record exercises no
    # dspark_* field at all, so running only that one leaves half the table
    # untested — which is how a field transcribed into one side and not the
    # other survived the first version of this check.
    sources = [root / "tests" / "fixtures" / "containers" / "Qwen3.5-MoE-Tiny",
               root / "tests" / "fixtures" / "oracles" / "DeepSeek-V4-Pro-0813-DSpark"]

    work = Path(tempfile.mkdtemp(prefix="soma-container-record-"))
    try:
        for source in sources:
            container = work / source.name
            shutil.copytree(source, container)
            good = json.loads(
                (container / "container_meta.json").read_text(encoding="utf-8"))
            shape = "dspark" if good.get("dspark") == "present" else "base"
            label = f"{source.name} ({shape})"

            # The control. A disagreement here means the rest proves nothing.
            cpp_ok, cpp_why = cpp_accepts(soma, container, good)
            py_ok, py_why = py_accepts(good)
            if not cpp_ok:
                fail(label, f"the engine rejects an unmutated fixture record: {cpp_why}")
                return 1
            if not py_ok:
                fail(label, f"record.py rejects an unmutated fixture record: {py_why}")
                return 1
            print(f"  ok      {label}  -> both accept the committed record")

            for case, mutated in mutations(good):
                cpp_ok, cpp_why = cpp_accepts(soma, container, mutated)
                py_ok, py_why = py_accepts(mutated)
                if cpp_ok and py_ok:
                    fail(f"{shape}: {case}",
                         "BOTH accepted a record that should not validate")
                elif cpp_ok != py_ok:
                    who = ("the engine accepted it, record.py did not" if cpp_ok
                           else "record.py accepted it, the engine did not")
                    fail(f"{shape}: {case}",
                         f"the two halves disagree — {who} ({py_why or cpp_why})")
                else:
                    print(f"  ok      {shape}: {case}  -> both reject")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if FAILURES:
        print(f"  FAILED   {FAILURES} disagreement(s) between record.py and arch_ir.cpp")
        return 1
    print("  OK       the record's two schema implementations agree on every mutation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
