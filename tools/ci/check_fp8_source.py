#!/usr/bin/env python3
"""A blockwise-fp8 upload must convert to the same container as its bf16 twin.

`convert.py` refused every checkpoint carrying a `quantization_config`, and for
compressed-tensors/AWQ/GPTQ that refusal is still right — quantizing packed
sub-byte levels as if they were weights produces a container that loads, streams
and generates noise. Blockwise fp8 is different in kind: `F8_E4M3` weights beside
a `<tensor>_scale_inv` of one f32 multiplier per tile dequantize EXACTLY, with a
single multiply and no layout to infer.

GLM-5.3 is why that distinction had to be made. It is the same base model as
GLM-5.2 — the two `config.json` files differ only in `transformers_version` — so
nothing in the engine, the IR or the tokenizer moved. What moved is the upload:
`zai-org/GLM-5.3` is fp8 at 756 GB and `zai-org/GLM-5.3-BF16` is a separate
1.5 TB repo, so refusing the primary meant supporting GLM-5.3 by fetching it
twice and keeping the larger copy.

Nothing else in the repository can check the dequantization. Every committed
fixture is f32, and the checkpoints that would exercise it start at 756 GB. So
this manufactures the missing input, and manufactures it in the one shape that
makes the comparison EXACT rather than approximate:

  1. quantize the committed fixture's weights blockwise to e4m3, keeping both
     halves — the fp8 levels with their scales, and the f32 product of the two;
  2. write the product as an ordinary f32 checkpoint, and the two halves as an
     fp8 one, changing nothing else;
  3. convert both, and require the expert payload and the dense half to be
     BYTE-IDENTICAL.

Byte-identical is available because step 1 removes the only real difference: the
f32 fixture already holds exactly what correct dequantization must produce, so
any tolerance at all would be hiding something. A scale applied in the wrong
direction, a tile grid transposed, a remainder tile mis-cropped, or an fp8 tensor
read unscaled all land orders of magnitude out, not within a tolerance.

Three block shapes run, and the shapes are chosen rather than convenient. 16x16
divides every dimension of the fixture evenly. 48x16 and 16x48 each leave a
ragged remainder — the case a `reshape` would pass and a ceil-and-crop must
handle — and, being NON-SQUARE, they are the only ones that can tell the two tile
axes apart: with square tiles alone, expanding the scale grid by `by` rows and
`bx` columns is indistinguishable from doing it correctly. The fixture's experts
are FUSED — rank-3 `(experts, 2*inter, hidden)` stacks — so the leading-axis path
is covered too.

Two refusals are checked alongside, because an exception that quietly widened
would be worse than no exception: a compressed-tensors config must still be
refused, and an fp8 tensor whose scale is missing must refuse rather than convert
a matrix ~400x too small.

Exits 77 (ctest's "skipped") when torch or safetensors is missing, so an
incomplete environment reports as skipped instead of passing silently.

Usage:  python tools/ci/check_fp8_source.py <repo_root> [work_dir]
"""

from __future__ import annotations

import filecmp
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURE = "tests/fixtures/tiny/GLM-5.2"

# e4m3's largest finite magnitude. The publisher's choice of scale is not
# something this check has to agree with — any positive scale round-trips through
# the same identity — but dividing by the tile's amax over this keeps the levels
# spread across the format instead of clustered near zero, which is what makes a
# mis-applied scale obvious rather than merely different.
E4M3_MAX = 448.0

# Mirrors a real upload's `modules_to_not_convert`: an fp8 checkpoint publishes
# its embeddings, its output head and its routers unquantized. Kept here so the
# generated fixture is MIXED — if every tensor were fp8, a converter that
# dequantized unconditionally would pass, and that converter mangles every norm
# in a real checkpoint.
NEVER_FP8 = ("model.embed_tokens.weight", "lm_head.weight", "mlp.gate.weight")


def refuse(msg: str) -> int:
    print(f"check_fp8_source: FAIL — {msg}")
    return 1


def quantize_blockwise(w, block, torch):
    """`w` (f32, rank >= 2) -> (levels as e4m3, scales as f32, exact product).

    The scale grid is CEILED over the last two axes, one scale per tile, and the
    third return value is what correct dequantization must reproduce bit for bit.
    """
    bx, by = block
    rows, cols = w.shape[-2], w.shape[-1]
    gr, gc = -(-rows // bx), -(-cols // by)

    scale = torch.empty(w.shape[:-2] + (gr, gc), dtype=torch.float32)
    for i in range(gr):
        for j in range(gc):
            tile = w[..., i * bx:(i + 1) * bx, j * by:(j + 1) * by]
            amax = tile.abs().amax(dim=(-2, -1))
            # An all-zero tile has no scale that means anything; 1.0 keeps the
            # round trip exact instead of producing a NaN out of 0/0.
            scale[..., i, j] = torch.where(amax > 0, amax / E4M3_MAX,
                                           torch.ones_like(amax))

    expanded = scale.repeat_interleave(bx, dim=-2).repeat_interleave(by, dim=-1)
    expanded = expanded[..., :rows, :cols]
    levels = (w / expanded).to(torch.float8_e4m3fn)
    return levels, scale, levels.to(torch.float32) * expanded


def is_fp8_candidate(name: str, ndim: int) -> bool:
    return ndim >= 2 and not any(name.endswith(suffix) for suffix in NEVER_FP8)


def write_pair(fixture: Path, plain: Path, fp8: Path, block, torch) -> int:
    """Write the two checkpoints. Returns the number of tensors quantized."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    plain.mkdir(parents=True, exist_ok=True)
    fp8.mkdir(parents=True, exist_ok=True)

    plain_t: dict = {}
    fp8_t: dict = {}
    quantized = 0
    with safe_open(str(fixture / "model.safetensors"), framework="pt") as h:
        for name in h.keys():
            t = h.get_tensor(name).to(torch.float32)
            if not is_fp8_candidate(name, t.ndim):
                plain_t[name] = t
                fp8_t[name] = t
                continue
            levels, scale, exact = quantize_blockwise(t, block, torch)
            plain_t[name] = exact
            fp8_t[name] = levels
            fp8_t[name + "_scale_inv"] = scale
            quantized += 1

    save_file(plain_t, str(plain / "model.safetensors"))
    save_file(fp8_t, str(fp8 / "model.safetensors"))

    cfg = json.loads((fixture / "config.json").read_text(encoding="utf-8"))
    (plain / "config.json").write_text(json.dumps(cfg, indent=2) + "\n",
                                       encoding="utf-8")
    cfg["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": list(block),
    }
    (fp8 / "config.json").write_text(json.dumps(cfg, indent=2) + "\n",
                                     encoding="utf-8")
    for name in ("meta.json",):
        if (fixture / name).is_file():
            shutil.copy(fixture / name, plain / name)
            shutil.copy(fixture / name, fp8 / name)
    return quantized


def convert(root: Path, model_dir: Path, out_dir: Path):
    return subprocess.run(
        [sys.executable, str(root / "tools" / "admission" / "convert.py"),
         str(model_dir), "--out", str(out_dir),
         "--quant", "q4_g", "--expert-down", "q6_g", "--group", "128"],
        capture_output=True, text=True)


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    fixture = root / FIXTURE
    # 77, not 0: ctest is configured with SKIP_RETURN_CODE 77. Returning 0 is how
    # a check reports Passed without having imported anything (roadmap D28).
    if not (fixture / "model.safetensors").is_file():
        print(f"check_fp8_source: SKIP — no {FIXTURE}")
        return 77
    try:
        import torch
        import safetensors  # noqa: F401
    except ImportError as e:
        print(f"check_fp8_source: SKIP — {e}")
        return 77
    if not hasattr(torch, "float8_e4m3fn"):
        print("check_fp8_source: SKIP — torch has no float8_e4m3fn")
        return 77

    work = Path(argv[2]).resolve() if len(argv) > 2 else None
    if work is not None:
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
    holder = tempfile.TemporaryDirectory() if work is None else None
    tmp = Path(holder.name) if holder is not None else work

    try:
        for block in ((16, 16), (48, 16), (16, 48)):
            tag = f"{block[0]}x{block[1]}"
            plain_dir, fp8_dir = tmp / f"plain-{tag}", tmp / f"fp8-{tag}"
            quantized = write_pair(fixture, plain_dir, fp8_dir, block, torch)
            if quantized == 0:
                return refuse("the generated fp8 checkpoint quantized nothing; "
                              "the comparison would be vacuous")

            made: dict[str, Path] = {}
            for label, model_dir in (("plain", plain_dir), ("fp8", fp8_dir)):
                out_dir = tmp / f"c-{label}-{tag}"
                proc = convert(root, model_dir, out_dir)
                if proc.returncode != 0:
                    return refuse(f"{tag} {label} conversion failed: "
                                  f"{(proc.stdout + proc.stderr).strip()[-400:]}")
                # Which path each run actually took, stated by the run itself.
                # The byte comparison below is only meaningful if one side went
                # through the dequantizer and the other did not; a fixture that
                # silently stopped being fp8 would compare a thing to itself and
                # prove nothing while printing OK.
                saw_fp8 = "fp8 source:" in proc.stdout
                if saw_fp8 != (label == "fp8"):
                    return refuse(f"{tag} {label} run "
                                  f"{'reported' if saw_fp8 else 'did not report'} "
                                  f"an fp8 source; the comparison would be vacuous")
                meta = json.loads(
                    (out_dir / "container_meta.json").read_text(encoding="utf-8"))
                want = f"fp8-e4m3-block-{tag}" if label == "fp8" else "none"
                if meta.get("source_quantization") != want:
                    return refuse(f"{tag} {label} recorded source_quantization "
                                  f"{meta.get('source_quantization')!r}, expected "
                                  f"{want!r} — a container cannot say what codec "
                                  f"it was built FROM")
                made[label] = out_dir

            for artifact in ("experts-00000.bin", "dense.safetensors"):
                a, b = made["plain"] / artifact, made["fp8"] / artifact
                if not (a.is_file() and b.is_file()):
                    return refuse(f"{tag}: {artifact} missing from one container")
                if not filecmp.cmp(a, b, shallow=False):
                    return refuse(
                        f"{tag}: {artifact} DIFFERS between the f32 and fp8 "
                        f"sources ({a.stat().st_size} B vs {b.stat().st_size} B). "
                        f"The two checkpoints hold the same numbers by "
                        f"construction, so this is the dequantization: a scale "
                        f"divided instead of multiplied, a transposed tile grid, "
                        f"or a remainder tile cropped wrong.")

        # ── the exception must stay narrow ───────────────────────────────────
        packed = tmp / "packed"
        shutil.copytree(tmp / "fp8-16x16", packed)
        cfg = json.loads((packed / "config.json").read_text(encoding="utf-8"))
        cfg["quantization_config"] = {"quant_method": "compressed-tensors",
                                      "format": "pack-quantized"}
        (packed / "config.json").write_text(json.dumps(cfg, indent=2) + "\n",
                                            encoding="utf-8")
        proc = convert(root, packed, tmp / "c-packed")
        if proc.returncode == 0 or "already quantized" not in proc.stdout:
            return refuse("a compressed-tensors checkpoint was NOT refused; the "
                          "fp8 exception has widened into 're-quantize anything'")

        # ── an fp8 weight with no scale must refuse, not under-scale ─────────
        from safetensors.torch import load_file, save_file
        stripped = tmp / "stripped"
        shutil.copytree(tmp / "fp8-16x16", stripped)
        tensors = load_file(str(stripped / "model.safetensors"))
        victim = next(k for k in tensors if k.endswith("_scale_inv"))
        del tensors[victim]
        save_file(tensors, str(stripped / "model.safetensors"))
        proc = convert(root, stripped, tmp / "c-stripped")
        combined = proc.stdout + proc.stderr
        if proc.returncode == 0 or victim not in combined:
            return refuse(f"removing {victim} did not stop the conversion; an "
                          f"fp8 weight read without its scale is ~400x too small "
                          f"and converts, loads and answers nonsense")
    finally:
        if holder is not None:
            holder.cleanup()

    print("check_fp8_source: OK — blockwise fp8 and f32 sources convert to "
          "byte-identical containers")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
