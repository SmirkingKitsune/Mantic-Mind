#!/usr/bin/env python3
"""Are the committed container fixtures still what the converter produces?

Nine container directories are checked into this repository and every
container-dependent gate reads one of them. They are build output, not source:
the only thing that says what they should contain is `convert.py` plus the tiny
checkpoint each was made from. Nothing re-derived them, so they drifted, and the
drift was invisible until a format change forced a regeneration:

  - Five carried no `tokenizer.unsupported`, because they predate the tokenizer
    stage. One of them, Qwen3-30B-A3B, is what `admission_fetch_stage` copies to
    build its "tokenizer that fails its oracle" case — and the marker's arrival
    silently turned that conformance stage from "failed" into "skipped".
  - `container_meta.json` was missing `dense_storage`, `dense_narrow_tensors` and
    `source_quantization` across the board.
  - DeepSeek-V4-Pro-0813 recorded a `config_sha256` matching neither its own
    `config.json` nor the tiny checkpoint's: a provenance field that had stopped
    describing the thing it names, which is worse than not having one.

None of that would fail a test. That is the point of this check: a regeneration
is a legitimate, sometimes necessary act, and the only thing distinguishing a
deliberate one from an accident is whether anything can still reproduce the
result. Run it after any converter or format change; when it fails on purpose,
regenerate with --write and commit the fixtures in the same change.

Line endings are normalized before comparing. Python writes text files with CRLF
on Windows and `.gitattributes` normalizes them back on commit, so a newline-only
difference is not a difference in anything that ships.

Usage:
    check_fixture_containers.py <repo-root> <soma-executable> [--write]
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# (tiny checkpoint name, container dir relative to the repo root, extra args)
#
# The DSpark oracle is f32 throughout: an oracle that quantized its own weights
# would compare the engine against a rounded copy of its own reference.
FIXTURES = [(f, f"tests/fixtures/containers/{f}", [])
            for f in ("DeepSeek-V2-Lite", "GLM-5.2", "Mixtral-8x7B-v0.1",
                      "Moonlight-16B-A3B", "OLMoE-1B-7B-0924", "Qwen3-30B-A3B",
                      "Qwen3.5-MoE-Tiny")]
FIXTURES += [
    ("DeepSeek-V4-Pro-0813", "tests/fixtures/containers/DeepSeek-V4-Pro-0813",
     ["--test-fixture", "--no-resume"]),
    ("DeepSeek-V4-Pro-0813-DSpark", "tests/fixtures/oracles/DeepSeek-V4-Pro-0813-DSpark",
     ["--test-fixture", "--no-resume", "--include-dspark",
      "--quant", "f32", "--expert-down", "f32"]),
]

FAILURES = 0


def fail(what: str, why: str) -> None:
    global FAILURES
    print(f"  FAILED  {what}: {why}")
    FAILURES += 1


def convert(root: Path, family: str, out: Path, extra: list) -> None:
    # The source path is RELATIVE and the converter runs from the repo root:
    # container_meta.json records it verbatim, and an absolute one would pin
    # every fixture to the machine that regenerated it.
    cmd = [sys.executable, str(root / "tools" / "admission" / "convert.py"),
           str(Path("tests") / "fixtures" / "tiny" / family),
           "--out", str(out), "--group", "128"]
    if "--quant" not in extra:
        cmd += ["--quant", "q4_g", "--expert-down", "q6_g"]
    cmd += extra
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{family}: convert exited {r.returncode}")


def stamp(soma: Path, container: Path) -> None:
    # Conversion leaves the container unstamped; the identity is written by the
    # engine because the canonical hash is defined by the C++ IR canonicalization.
    # Until `soma stamp` folds into conversion, reproducing a fixture means
    # reproducing both halves.
    r = subprocess.run([str(soma), "stamp", str(container)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"{container.name}: stamp exited {r.returncode}")


def same(a: bytes, b: bytes) -> bool:
    return a == b or a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")


def compare(name: str, fresh: Path, committed: Path) -> None:
    fresh_files = {p.name for p in fresh.iterdir()}
    old_files = {p.name for p in committed.iterdir()}
    for missing in sorted(old_files - fresh_files):
        fail(name, f"{missing} is committed but the converter no longer writes it")
    for extra in sorted(fresh_files - old_files):
        fail(name, f"the converter writes {extra} and it is not committed")
    for shared in sorted(fresh_files & old_files):
        a, b = (fresh / shared).read_bytes(), (committed / shared).read_bytes()
        if same(a, b):
            continue
        if shared.endswith(".json"):
            ja, jb = json.loads(a.decode("utf-8")), json.loads(b.decode("utf-8"))
            moved = [k for k in sorted(set(ja) | set(jb)) if ja.get(k) != jb.get(k)]
            detail = ", ".join(f"{k}: committed={jb.get(k)!r} fresh={ja.get(k)!r}"[:120]
                               for k in moved[:4]) or "nested values differ"
            fail(name, f"{shared} differs — {detail}")
        else:
            fail(name, f"{shared} differs — {len(b)} committed vs {len(a)} fresh bytes")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    root, soma = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    write = "--write" in sys.argv[3:]

    try:
        import numpy  # noqa: F401
        import safetensors  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        # 77 so ctest reports "Skipped" rather than green. See D28.
        print(f"  SKIP     admission dependencies unavailable: {e}")
        return 77

    work = Path(tempfile.mkdtemp(prefix="soma-fixture-containers-"))
    try:
        for family, rel, extra in FIXTURES:
            committed = root / rel
            fresh = work / family
            convert(root, family, fresh, extra)
            stamp(soma, fresh)
            if write:
                for p in sorted(fresh.iterdir()):
                    shutil.copy2(p, committed / p.name)
                print(f"  wrote   {rel}")
                continue
            before = FAILURES
            compare(rel, fresh, committed)
            if FAILURES == before:
                print(f"  ok      {rel}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if write:
        print("  WROTE    fixtures regenerated; review the diff before committing")
        return 0
    if FAILURES:
        print(f"  FAILED   {FAILURES} difference(s). If this was intentional, rerun "
              f"with --write and commit the fixtures in the same change")
        return 1
    print(f"  OK       all {len(FIXTURES)} container fixtures reproduce from their "
          f"tiny checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
