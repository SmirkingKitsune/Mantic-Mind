#!/usr/bin/env python3
"""Conformance stage 3: an fp16/bf16 reference from a REAL checkpoint.

Stages 1 and 2 use tiny-random models and demand token-exactness. Stage 3 cannot:
a real checkpoint runs at a precision Soma does not reproduce bit-for-bit
(bf16 reference vs quantized engine), so the bar is DISTRIBUTIONAL — logit-KL
under threshold — rather than exact.

That difference is the whole point of separating the stages. Failing stage 3
while 1 and 2 pass is a QUANTIZATION finding, not a correctness bug, and the
remediation is different: requantize a role or raise a group-scale granularity,
not debug a kernel.

Output is the same SOMAORCL container the tiny oracles use, so the C++ side has
one reader.

Usage:
    make_reference.py <model_dir> --out <dir> [--positions 512] [--dtype bfloat16]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=20260730)
    args = ap.parse_args(argv[1:])

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    src = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(str(src))
    vocab = int(cfg.vocab_size)
    print(f"  {src.name}: vocab={vocab} layers={cfg.num_hidden_layers} dtype={args.dtype}")

    t0 = time.time()
    print("  loading (this is the slow part) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        dtype=getattr(torch, args.dtype),
        low_cpu_mem_usage=True,
        device_map="cpu",
    ).eval()
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    # Real text, not random ids.
    #
    # Random tokens put the model far off-distribution, where its logits are
    # flat and a KL comparison is dominated by noise rather than by whether the
    # engine reproduces the model. Stage 3 is supposed to measure the latter.
    tok = AutoTokenizer.from_pretrained(str(src))
    corpus = (
        "The mitochondrion is a double-membrane-bound organelle found in most "
        "eukaryotic organisms. Mitochondria generate most of the cell's supply of "
        "adenosine triphosphate, used as a source of chemical energy.\n\n"
        "def quicksort(items):\n"
        "    if len(items) <= 1:\n"
        "        return items\n"
        "    pivot = items[len(items) // 2]\n"
        "    left = [x for x in items if x < pivot]\n"
        "    middle = [x for x in items if x == pivot]\n"
        "    right = [x for x in items if x > pivot]\n"
        "    return quicksort(left) + middle + quicksort(right)\n\n"
        "In 1687 Newton published the Principia, stating the laws of motion and "
        "universal gravitation. The work established classical mechanics and "
        "remained the dominant framework until relativity and quantum theory.\n\n"
        "Le renard brun rapide saute par-dessus le chien paresseux. "
        "素早い茶色の狐が怠け者の犬を飛び越える。 "
        "The quick brown fox jumps over the lazy dog, repeatedly and without complaint."
    ) * 6

    ids = tok(corpus, return_tensors="pt").input_ids[0][: args.positions]
    if ids.shape[0] < args.positions:
        print(f"  WARNING: corpus yielded {ids.shape[0]} tokens, wanted {args.positions}")
    ids = ids.unsqueeze(0)
    n = ids.shape[1]

    print(f"  forward over {n} positions ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0].to(torch.float32)
    print(f"  forward in {time.time() - t0:.0f}s", flush=True)

    ids_np = ids[0].numpy().astype("<i4")
    lg_np = logits.numpy().astype("<f4")

    # Same SOMAORCL layout as the tiny oracles: one reader on the C++ side.
    # greedy_* are written empty — stage 3 is distributional, and a greedy
    # sequence from a quantized engine is not expected to match.
    with open(out_dir / "oracle.bin", "wb") as fh:
        fh.write(b"SOMAORCL")
        fh.write(np.array([1, n, vocab, 0, 0], dtype="<u4").tobytes())
        fh.write(ids_np.tobytes())
        fh.write(lg_np.tobytes())

    meta = {
        "kind": "stage3_reference",
        "source": str(src),
        "dtype": args.dtype,
        "positions": int(n),
        "vocab": vocab,
        "torch_version": torch.__version__,
        "logit_mean": float(lg_np.mean()),
        "logit_std": float(lg_np.std()),
    }
    (out_dir / "reference_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"  OK  {n} positions x {vocab} vocab, logit std={lg_np.std():.3f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
