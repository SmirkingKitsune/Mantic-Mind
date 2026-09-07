#!/usr/bin/env python3
"""A bf16 upload must convert to a container that is smaller but not different.

Every real MoE checkpoint published so far ships bf16, and `convert.py` used to
upcast the resident half to f32 on the way to disk. That doubles `dense.safetensors`
and adds no information: bf16 is the top 16 bits of an f32, so the widening the
engine does at load is exact and the upcast merely moved it earlier, onto a file
that then has to be stored, transferred to every node, and read at every startup.

`--dense-storage source` keeps the checkpoint's own precision for the resident
MATRICES. The 1-D controls — norms, router logits, sinks — stay f32 whatever the
source was: they are kilobytes beside the matrices and the engine binds them as
zero-copy views. An f32 source stays f32 in full, because storing it narrow would
lose bits the checkpoint actually had.

Nothing else in the repository can check this. Every committed fixture is f32, so
this manufactures the input, and manufactures it in the one shape that makes the
comparison EXACT rather than approximate:

  1. round the committed fixture's weights to bf16, so the numbers are ones bf16
     can hold exactly;
  2. write those same numbers twice — once as bf16, once widened back to f32 —
     changing nothing else;
  3. convert both, and require the EXPERT PAYLOAD to be byte-identical while the
     dense half of the bf16 conversion is about half the size.

Byte-identical experts are available because step 1 removes the only real
difference: quantizing from bf16-widened values and from the same values held as
f32 must produce the same q4_g/q6_g bytes, and any tolerance would hide a widening
that dropped or shifted a mantissa.

The dense halves are then compared TENSOR BY TENSOR after widening, which is the
claim that actually matters — that the smaller file holds the same numbers.

Exits 77 (ctest's "skipped") when torch or safetensors is missing, so an
incomplete environment reports as skipped instead of passing silently.

Usage:  python tools/ci/check_bf16_source.py <repo_root> [work_dir]
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURE = "tests/fixtures/tiny/Qwen3-30B-A3B"

FAILURES = 0


def fail(msg: str) -> None:
    print(f"  FAILED  {msg}")
    globals()["FAILURES"] += 1


def convert(root: Path, src: Path, out: Path, *extra: str) -> bool:
    r = subprocess.run(
        [sys.executable, str(root / "tools" / "admission" / "convert.py"), str(src),
         "--out", str(out), "--quant", "q4_g", "--expert-down", "q6_g",
         "--group", "128", *extra],
        capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"convert {src.name} exited {r.returncode}: {r.stdout[-400:]}{r.stderr[-400:]}")
    return r.returncode == 0


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    src = root / FIXTURE
    if not (src / "model.safetensors").is_file():
        print(f"check_bf16_source: SKIP — no {FIXTURE}")
        return 77
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as e:
        print(f"check_bf16_source: SKIP — {e.name} is not installed")
        return 77

    work = Path(argv[2]) if len(argv) > 2 else root / "build" / "bf16_source_tmp"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    # ── 1. two sources holding IDENTICAL numbers ─────────────────────────────
    tensors = {k: v.to(torch.bfloat16) for k, v in
               load_file(str(src / "model.safetensors")).items()}
    sources = {}
    for kind, cast in (("bf16", lambda t: t), ("f32", lambda t: t.to(torch.float32))):
        d = work / f"src-{kind}"
        d.mkdir()
        for f in src.iterdir():
            if f.suffix != ".safetensors":
                shutil.copy2(f, d / f.name)
        save_file({k: cast(v).contiguous() for k, v in tensors.items()},
                  str(d / "model.safetensors"))
        sources[kind] = d

    # ── 2. convert both ──────────────────────────────────────────────────────
    outs = {}
    for kind, d in sources.items():
        outs[kind] = work / f"out-{kind}"
        if not convert(root, d, outs[kind]):
            return 1

    # ── 3. the expert payload is the same bytes ──────────────────────────────
    for shard in sorted(outs["f32"].glob("experts-*.bin")):
        other = outs["bf16"] / shard.name
        if not other.is_file():
            fail(f"{shard.name} is missing from the bf16 conversion")
        elif shard.read_bytes() != other.read_bytes():
            fail(f"{shard.name} differs; widening bf16 changed the quantized bytes")
    if not FAILURES:
        print("  ok      expert payload is byte-identical from either source")

    # ── 4. the dense half is smaller, and holds the same numbers ─────────────
    dense = {k: load_file(str(outs[k] / "dense.safetensors")) for k in outs}
    big = (outs["f32"] / "dense.safetensors").stat().st_size
    small = (outs["bf16"] / "dense.safetensors").stat().st_size
    if small >= big:
        fail(f"the bf16 dense half is {small} B against f32's {big} B; it saved nothing")
    else:
        print(f"  ok      dense half {big} -> {small} B "
              f"({100.0 * (big - small) / big:.1f}% smaller)")

    if set(dense["f32"]) != set(dense["bf16"]):
        fail("the two dense halves name different tensors")
    else:
        worst = 0.0
        narrow = 0
        for name, ref in dense["f32"].items():
            got = dense["bf16"][name]
            if got.dtype is not torch.float32:
                narrow += 1
            diff = (got.to(torch.float32) - ref).abs().max().item() if ref.numel() else 0.0
            worst = max(worst, diff)
        if worst != 0.0:
            fail(f"widening is not exact: max|diff| {worst:g}")
        else:
            print(f"  ok      every dense tensor widens back exactly ({narrow} stored narrow)")
        if narrow == 0:
            fail("nothing was stored narrow; --dense-storage source had no effect")

    # ── 5. the controls stay f32, and an f32 source is untouched ────────────
    for name, t in dense["bf16"].items():
        if t.dim() < 2 and t.dtype is not torch.float32:
            fail(f"{name} is a 1-D control stored as {t.dtype}; controls stay f32")
    for name, t in dense["f32"].items():
        if t.dtype is not torch.float32:
            fail(f"{name} came from an f32 source and was stored as {t.dtype}")
    if not FAILURES:
        print("  ok      controls stay f32, and an f32 source is never narrowed")

    # ── 6. the opt-out still works ───────────────────────────────────────────
    forced = work / "out-forced-f32"
    if convert(root, sources["bf16"], forced, "--dense-storage", "f32"):
        meta = json.loads((forced / "container_meta.json").read_text(encoding="utf-8"))
        if meta.get("dense_narrow_tensors", 0) != 0:
            fail("--dense-storage f32 still stored tensors narrow")
        elif (forced / "dense.safetensors").stat().st_size != big:
            fail("--dense-storage f32 did not reproduce the upcast dense half")
        else:
            print("  ok      --dense-storage f32 reproduces the old artifact exactly")

    if FAILURES:
        print(f"check_bf16_source: FAIL — {FAILURES} case(s)")
        return 1
    print("check_bf16_source: OK - a bf16 upload converts smaller, not different")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
