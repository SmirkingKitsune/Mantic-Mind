#!/usr/bin/env python3
"""The FUSED expert layout must convert identically to the per-expert one.

`convert.py` reads two layouts. Per-expert is one tensor per (expert, role).
Fused stacks every expert into `experts.gate_up_proj` of shape
`(experts, 2*intermediate, hidden)` plus `experts.down_proj` of shape
`(experts, hidden, intermediate)`, and recent `transformers` emits it — anything
`make_oracle.py` produces for such a family carries it.

The hazard is the gate/up split, which is why reading the layout went
unimplemented for so long (roadmap D4): this family concatenates gate rows then
up rows, while gpt_oss stores the same conceptual thing INTERLEAVED. Both
readings produce a container that loads and runs. Only one produces the right
answer, so guessing yields a model that is fluently, silently wrong.

This check removes the guess. It takes a committed PER-EXPERT fixture, rewrites
it into the fused layout changing nothing else, converts both, and requires the
expert payloads to be BYTE-IDENTICAL. That is a real comparison rather than a
restatement: the per-expert container is already verified against the engine's
own quantizer by container_g2's round-trip, so byte-equality here transitively
inherits that.

A test rather than a one-off, because the split is exactly the kind of thing a
future refactor reverses without noticing.

Runs offline. Needs numpy + safetensors, which the admission venv already has.

Usage: check_fused_experts.py <repo_root>
"""

from __future__ import annotations

import filecmp
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURE = "tests/fixtures/tiny/OLMoE-1B-7B-0924"


def refuse(msg: str) -> int:
    print(f"check_fused_experts: FAIL — {msg}")
    return 1


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    src = root / FIXTURE
    # 77, not 0: ctest is configured with SKIP_RETURN_CODE 77, so a skip prints
    # as "Skipped". Returning 0 is what let this test report Passed in 0.11 s
    # without importing numpy, for its whole existence (roadmap D28).
    if not (src / "model.safetensors").is_file():
        print(f"check_fused_experts: SKIP — no {FIXTURE}")
        return 77
    try:
        import numpy as np
        from safetensors.numpy import load_file, save_file
    except ImportError as e:
        print(f"check_fused_experts: SKIP — {e}")
        return 77

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fused_dir = tmp / "fused"
        fused_dir.mkdir()

        raw = load_file(str(src / "model.safetensors"))
        cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
        n_exp = int(cfg.get("num_experts") or cfg.get("n_routed_experts")
                    or cfg.get("num_local_experts") or 0)
        if n_exp <= 0:
            return refuse("fixture declares no expert count")

        out: dict = {}
        bases = set()
        for k in raw:
            if ".experts." in k:
                bases.add(k.split(".experts.")[0])
            else:
                out[k] = raw[k]
        if not bases:
            return refuse("fixture has no expert tensors")

        for base in sorted(bases):
            def stack(role: str):
                return np.stack([raw[f"{base}.experts.{e}.{role}_proj.weight"]
                                 for e in range(n_exp)])
            gate, up, down = stack("gate"), stack("up"), stack("down")
            # CONTIGUOUS: gate rows, then up rows. This is the convention the
            # reference implementation applies — `linear(x, gate_up_proj[e])`
            # followed by `.chunk(2, dim=-1)`, i.e. contiguous halves of the
            # output and therefore contiguous rows of the weight.
            out[f"{base}.experts.gate_up_proj"] = np.concatenate([gate, up], axis=1)
            out[f"{base}.experts.down_proj"] = down

        save_file(out, str(fused_dir / "model.safetensors"))
        for name in ("config.json", "meta.json"):
            if (src / name).is_file():
                shutil.copy(src / name, fused_dir / name)

        convert = root / "tools" / "admission" / "convert.py"
        made = {}
        for label, model_dir in (("perexpert", src), ("fused", fused_dir)):
            out_dir = tmp / f"c-{label}"
            proc = subprocess.run(
                [sys.executable, str(convert), str(model_dir), "--out", str(out_dir),
                 "--quant", "q4_g", "--expert-down", "q6_g", "--group", "128"],
                capture_output=True, text=True)
            if proc.returncode != 0:
                return refuse(f"{label} conversion failed: "
                              f"{(proc.stdout + proc.stderr).strip()[-300:]}")
            # The layout each run actually took, so a fixture that silently
            # stopped being fused cannot pass this by converting twice the same
            # way — which would compare a thing to itself and prove nothing.
            want = "fused" if label == "fused" else "per-expert"
            if f"expert layout: {want}" not in proc.stdout:
                return refuse(f"{label} run did not report the {want} layout; "
                              f"this comparison would be vacuous")
            payload = out_dir / "experts-00000.bin"
            if not payload.is_file():
                return refuse(f"{label} produced no expert payload")
            made[label] = payload

        if not filecmp.cmp(made["perexpert"], made["fused"], shallow=False):
            a, b = (p.stat().st_size for p in (made["perexpert"], made["fused"]))
            return refuse("expert payloads DIFFER between layouts "
                          f"(per-expert {a} B, fused {b} B). If the sizes match, "
                          "the gate/up split is probably reversed — contiguous vs "
                          "interleaved — which produces a model that runs and is wrong.")

    print("check_fused_experts: OK — fused and per-expert layouts convert byte-identically")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
