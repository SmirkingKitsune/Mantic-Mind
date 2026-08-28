#!/usr/bin/env python3
"""Does verify_payload.py actually catch a corrupted container?

A verifier nobody has broken on purpose is a verifier that reports "OK" for
reasons unknown. Roadmap D27 exists because admission passed 460 GB without
reading it; replacing that with a checker of unknown sensitivity would be the
same mistake wearing a hat.

Each case below damages a GOOD container in one specific way and requires the
right pass to notice. Asserting on WHICH pass fires matters: a verifier that
refused everything would satisfy a bare "exit non-zero" test.

  truncated shard        -> STRUCTURE, and named as truncation
  trailing bytes         -> STRUCTURE, unaccounted tail
  index offset shifted   -> STRUCTURE, gap/overlap in the shard's packing
  one flipped byte       -> EXACT, byte-for-byte against a re-quantized source
  two experts swapped    -> DECODE's decoy margin, i.e. right bytes, wrong slot

The swap case is the one that justifies the decode pass existing at all: it is
the failure that structure cannot see (every length and offset is still perfect)
and it is what a wrong gate/up split or a mis-indexed expert would look like.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "admission"))

VERIFY = ROOT / "tools" / "admission" / "verify_payload.py"
CONVERT = ROOT / "tools" / "admission" / "convert.py"
FIXTURE = ROOT / "tests" / "fixtures" / "tiny" / "GLM-5.2"


def run_verify(container: Path, source: Path, *extra: str):
    out = subprocess.run(
        [sys.executable, str(VERIFY), str(container), "--source", str(source),
         "--samples", "6", "--json", *extra],
        capture_output=True, text=True)
    try:
        return out.returncode, json.loads(out.stdout)
    except json.JSONDecodeError:
        return out.returncode, {"stdout": out.stdout, "stderr": out.stderr}


def index_header_len(container: Path) -> int:
    raw = (container / "soma.container").read_bytes()
    (hash_len,) = struct.unpack_from("<I", raw, 16)
    return 8 + 8 + 4 + hash_len + 16 + 4 + 16


def first_live_slot(container: Path) -> tuple[int, int, int, int]:
    """(slot, shard, off, length) of the first non-empty entry."""
    raw = (container / "soma.container").read_bytes()
    base = index_header_len(container)
    n = (len(raw) - base) // 16
    for i in range(n):
        shard, off, length = struct.unpack_from("<IQI", raw, base + i * 16)
        if length:
            return i, shard, off, length
    raise SystemExit("fixture has no live slots")


def fail(case: str, why: str) -> None:
    print(f"  FAILED  {case}: {why}")
    globals()["FAILURES"] += 1


FAILURES = 0


def expect_caught(case: str, container: Path, source: Path, pass_name: str, needle: str):
    code, rep = run_verify(container, source)
    if code == 0:
        return fail(case, "verify_payload reported OK on a container we damaged")
    blob = json.dumps(rep).lower()
    if rep.get(pass_name) != "failed":
        return fail(case, f"expected the {pass_name} pass to fail, got "
                          f"structure={rep.get('structure')} content={rep.get('content')}")
    if needle.lower() not in blob:
        return fail(case, f"caught it, but never said {needle!r}: "
                          f"{rep.get('reason') or rep.get('failures')}")
    print(f"  ok      {case}  -> {pass_name}")


def main() -> int:
    try:
        import numpy  # noqa: F401
        import safetensors  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        # 77 so ctest reports "Skipped" rather than green. See D28.
        print(f"  skipped  {e.name} is not installed")
        return 77

    work = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "verify_payload_tmp"
    good = work / "good"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    conv = subprocess.run(
        [sys.executable, str(CONVERT), str(FIXTURE), "--out", str(good),
         "--quant", "q4_g", "--expert-down", "q6_g", "--group", "32"],
        capture_output=True, text=True)
    if conv.returncode != 0:
        print(f"  FAILED  could not convert the fixture: {conv.stdout}{conv.stderr}")
        return 1

    # The control. If this does not pass, nothing below means anything.
    code, rep = run_verify(good, FIXTURE)
    if code != 0 or rep.get("structure") != "passed" or rep.get("content") != "passed":
        print(f"  FAILED  the UNDAMAGED container did not verify: {json.dumps(rep)[:400]}")
        return 1
    checked = rep.get("checked", [])
    if not checked:
        print("  FAILED  control run sampled zero experts, so the content pass is vacuous")
        return 1
    if not all(r.get("exact") for r in checked):
        print("  FAILED  control run reported a non-exact expert")
        return 1
    print(f"  ok      control    -> structure + {len(checked)} experts, "
          f"decoy margin {min(r.get('decoy_margin', 0) for r in checked):.1f}x")

    slot, shard, off, length = first_live_slot(good)
    shard_name = f"experts-{shard:05d}.bin"

    def fresh(tag: str) -> Path:
        d = work / tag
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(good, d)
        return d

    # 1. Truncated shard — a conversion that died mid-write.
    d = fresh("truncated")
    p = d / shard_name
    with open(p, "r+b") as f:
        f.truncate(p.stat().st_size - 4096)
    expect_caught("truncated shard", d, FIXTURE, "structure", "truncated")

    # 2. Trailing bytes — a shard longer than the index accounts for.
    d = fresh("trailing")
    with open(d / shard_name, "ab") as f:
        f.write(b"\0" * 4096)
    expect_caught("trailing bytes", d, FIXTURE, "structure", "trailing bytes")

    # 3. An index offset shifted one page — every length still perfect.
    d = fresh("shifted")
    raw = bytearray((d / "soma.container").read_bytes())
    base = index_header_len(d)
    struct.pack_into("<IQI", raw, base + slot * 16, shard, off + 4096, length)
    (d / "soma.container").write_bytes(raw)
    expect_caught("index offset shifted", d, FIXTURE, "structure", "gap")

    # 4. One flipped byte, mid-payload. Structure cannot see this.
    d = fresh("flipped")
    with open(d / shard_name, "r+b") as f:
        f.seek(off + length // 2)
        b = f.read(1)
        f.seek(off + length // 2)
        f.write(bytes([b[0] ^ 0xFF]))
    expect_caught("one flipped byte", d, FIXTURE, "content", "differs from a re-quantized")

    # 5. Two experts swapped. Structure is perfect; only the decoy margin sees it.
    d = fresh("swapped")
    raw_ix = (d / "soma.container").read_bytes()
    base = index_header_len(d)
    s0, o0, l0 = struct.unpack_from("<IQI", raw_ix, base + slot * 16)
    s1, o1, l1 = struct.unpack_from("<IQI", raw_ix, base + (slot + 1) * 16)
    if (s0, l0) != (s1, l1):
        print("  FAILED  swap case needs two same-size slots in one shard")
        return 1
    with open(d / shard_name, "r+b") as f:
        f.seek(o0)
        a = f.read(l0)
        f.seek(o1)
        b = f.read(l1)
        f.seek(o0)
        f.write(b)
        f.seek(o1)
        f.write(a)
    code, rep = run_verify(d, FIXTURE)
    if code == 0:
        fail("two experts swapped", "reported OK")
    elif rep.get("structure") != "passed":
        fail("two experts swapped",
             "the structure pass failed, so this did not exercise the decode pass")
    elif not any("wrong slot" in f for f in rep.get("failures", [])):
        fail("two experts swapped",
             f"content failed but not via the decoy margin: {rep.get('failures')}")
    else:
        print("  ok      two experts swapped -> content (decoy margin)")

    # 6. --structure-only must not silently claim the contents were checked.
    code, rep = run_verify(good, FIXTURE, "--structure-only")
    if code != 0 or rep.get("content") != "skipped":
        fail("--structure-only", f"content should read 'skipped', got {rep.get('content')}")
    else:
        print("  ok      --structure-only  -> content reported as skipped, not passed")

    if FAILURES:
        print(f"  FAILED  {FAILURES} case(s)")
        return 1
    print("  OK       verify_payload catches all 5 corruptions and stays honest about scope")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
