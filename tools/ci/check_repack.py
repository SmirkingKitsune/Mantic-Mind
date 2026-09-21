#!/usr/bin/env python3
"""Does the repack move experts without changing them?

`tools/admission/repack.py` rewrites a container with its experts in heat order.
Everything about it is a promise that only their PLACEMENT changed, and every one
of those promises is checkable:

  the bytes        every expert's payload is identical to the source's, compared
                   directly rather than through the digest table — because the
                   table is COPIED by the repack, so checking the output against
                   it would be checking the tool against its own assumption
  the digests      `soma verify` matches the copied table against bytes that have
                   MOVED, which is the guarantee that matters
  the identity     arch_hash covers the architecture and the quant map, neither of
                   which a repack touches, so the output is the same model and the
                   registry and its KV checkpoints stay valid
  the layout       the ranges still tile each shard, which `soma verify` also
                   checks, and the pinned set collapses to one run per shard
  write-once       repacking onto the input is refused, because an in-place
                   rewrite re-transfers every shard to every node holding it

The last one is not a style rule. Node cache identity is a hash over each file's
(relpath, size, mtime); a repack in place would be the most expensive possible way
to save a few seconds of startup.

Usage:
    check_repack.py <repo-root> <soma-executable>
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "admission"))

from verify_payload import read_index  # noqa: E402

FAILURES = 0


def fail(case: str, why: str) -> None:
    global FAILURES
    print(f"  FAILED  {case}: {why}")
    FAILURES += 1


def experts_of(container: Path) -> dict:
    """{slot: payload}, read through the index rather than off the front."""
    ix = read_index(container / "soma.container")
    out = {}
    handles = {}
    try:
        for slot, (shard, off, length) in enumerate(ix["entries"]):
            if length == 0:
                continue
            fh = handles.get(shard)
            if fh is None:
                fh = open(container / f"experts-{shard:05d}.bin", "rb")
                handles[shard] = fh
            fh.seek(off)
            out[slot] = fh.read(length)
    finally:
        for fh in handles.values():
            fh.close()
    return out


def runs(soma: Path, container: Path, heat: Path, pin: int) -> int:
    r = subprocess.run([str(soma), "heat-layout", str(container), "--heat", str(heat),
                        "--pin", str(pin), "--json"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"heat-layout failed on {container.name}: {r.stderr.strip()}")
    return int(json.loads(r.stdout)["runs"])


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    root, soma = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    source = root / "tests" / "fixtures" / "containers" / "Qwen3.5-MoE-Tiny"
    repack = root / "tools" / "admission" / "repack.py"

    work = Path(tempfile.mkdtemp(prefix="soma-repack-"))
    try:
        # A deterministic heat snapshot whose hot experts are scattered through
        # each layer — which is the case worth fixing, and the case a
        # conversion-order layout produces.
        rng = random.Random(7)
        ix = read_index(source / "soma.container")
        heat = work / "heat.json"
        heat.write_text(json.dumps({"experts": [
            {"layer": layer, "expert": expert, "count": 0,
             "decayed": round(rng.random(), 4)}
            for layer in range(ix["n_layers"]) for expert in range(ix["n_experts"])]}),
            encoding="utf-8")
        pin = max(ix["total"] // 4, 1)

        out = work / "repacked"
        r = subprocess.run([sys.executable, str(repack), str(source), "--out", str(out),
                            "--heat", str(heat), "--soma", str(soma)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fail("repack", f"exited {r.returncode}: {(r.stdout + r.stderr)[-500:]}")
            return 1
        print("  ok      repack  -> wrote a new container and it verifies")

        before, after = runs(soma, source, heat, pin), runs(soma, out, heat, pin)
        if after >= before:
            fail("scatter", f"the pinned set still reads as {after} run(s), was {before}")
        else:
            print(f"  ok      scatter  -> {before} run(s) becomes {after}")

        src_experts, out_experts = experts_of(source), experts_of(out)
        if set(src_experts) != set(out_experts):
            fail("slots", "the repack changed which slots hold an expert")
        else:
            differing = [s for s in src_experts if src_experts[s] != out_experts[s]]
            if differing:
                fail("bytes", f"{len(differing)} expert(s) changed, first at slot "
                              f"{differing[0]}")
            else:
                print(f"  ok      bytes  -> all {len(src_experts)} experts are "
                      f"byte-identical, compared directly")

        moved = sum(1 for s in src_experts
                    if read_index(source / "soma.container")["entries"][s][1] !=
                       read_index(out / "soma.container")["entries"][s][1])
        if moved == 0:
            fail("placement", "nothing moved, so this proved nothing about a repack")
        else:
            print(f"  ok      placement  -> {moved} expert(s) sit at a new offset")

        src_hash = read_index(source / "soma.container")["arch_hash"]
        out_hash = read_index(out / "soma.container")["arch_hash"]
        if src_hash != out_hash:
            fail("identity", f"arch_hash moved: {src_hash[:16]} -> {out_hash[:16]}")
        else:
            print("  ok      identity  -> arch_hash unchanged, so it is the same model")

        # In place is refused. The tool's whole premise is that a container is
        # written once.
        r = subprocess.run([sys.executable, str(repack), str(out), "--out", str(out),
                            "--heat", str(heat)], capture_output=True, text=True)
        if r.returncode == 0:
            fail("in-place", "repacking onto the input was allowed")
        else:
            print("  ok      in-place  -> refused")

        # A dry run writes nothing.
        empty = work / "never-written"
        r = subprocess.run([sys.executable, str(repack), str(source), "--out", str(empty),
                            "--heat", str(heat), "--dry-run"],
                           capture_output=True, text=True)
        if r.returncode != 0 or empty.exists():
            fail("dry-run", f"exited {r.returncode}, created={empty.exists()}")
        else:
            print("  ok      dry-run  -> reported the gain and wrote nothing")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if FAILURES:
        print(f"  FAILED   {FAILURES} repack promise(s) broken")
        return 1
    print("  OK       the repack moves experts without changing them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
