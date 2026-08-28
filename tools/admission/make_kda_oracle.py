#!/usr/bin/env python3
"""Emit a KERNEL-level oracle for Kimi Delta Attention.

`soma_kda_kernel` checks the recurrence against identities derived from what a
delta rule is supposed to do. That is strong — it caught six distinct
misreadings — but every one of those identities was written by the same reader
who wrote the kernel. A consistently wrong reading passes them all.

This produces the missing thing: the SAME inputs run through `fla`'s own
`fused_recurrent_kda`, at the exact flags `modeling_kimi_linear.py` passes, so
the C++ kernel can be graded against the implementation Moonshot actually
depends on rather than against a second opinion from its author.

WHY THE RECURRENT KERNEL AND NOT THE CHUNKED ONE. `KimiDeltaAttention` calls
`chunk_kda` whenever it has more than one token, so the chunked path is what
production runs. It is also not a reference: measured on this box, fla's chunked
and recurrent kernels disagree with EACH OTHER by 5.8e-04 on random unit-scale
input (6.8e-04 on the final state), and that gap is unaffected by torch's TF32
switches because Triton chooses its own `tl.dot` precision. The chunked kernel
reformulates the decay in exp2 and reassociates the products; the difference is
the algorithm's, not a bug.

Soma implements the sequential recurrence, so `fused_recurrent_kda` is the
apples-to-apples reference and the tight tolerance is meaningful there. What
this file therefore CANNOT establish is that Soma matches production prefill to
better than ~1e-03 — nothing can, because fla does not match itself to better
than that.

REQUIRES CUDA. fla's kernels are Triton; there is no CPU backend to fall back
on. See tools/admission/requirements-kimi.txt for the interpreter.

Usage:
    .venv-kimi/Scripts/python tools/admission/make_kda_oracle.py \\
        --out tests/fixtures/kernels/kda
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

MAGIC = b"SOMAKDA1"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # NOT under tests/fixtures/tiny: the conformance harnesses walk that
    # directory and expect every entry to be a MODEL fixture with a config.json.
    # A kernel oracle placed there is reported as a broken model.
    ap.add_argument("--out", default="tests/fixtures/kernels/kda")
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--heads", type=int, default=2)
    # 16 is a FLOOR, not a preference: fla's Triton kernels reject a head
    # dimension below it with "Input shapes should have M >= 1, N >= 1 and
    # K >= 16" from inside `tl.dot`. A fixture at 8 does not merely run slowly,
    # it fails to compile.
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--gate-lower-bound", type=float, default=-5.0,
                    help="Kimi-K3 configures -5.0; pass 0 for the unbounded softplus gate")
    args = ap.parse_args(argv[1:])

    import torch
    from fla.ops.kda import fused_recurrent_kda

    if not torch.cuda.is_available():
        print("error: fla's KDA kernels are Triton and need CUDA", file=sys.stderr)
        return 2

    H, D, T = args.heads, args.head_dim, args.tokens
    dev = "cuda"
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.empty(*shape, dtype=torch.float32).uniform_(-1.0, 1.0, generator=gen)

    # Generated on the CPU from a seeded generator, then moved: CUDA's RNG is not
    # reproducible across driver versions, and a fixture whose inputs drift is
    # not a fixture.
    q, k, v = rnd(1, T, H, D), rnd(1, T, H, D), rnd(1, T, H, D)
    g_raw = rnd(1, T, H, D)
    beta_raw = rnd(1, T, H)
    # A_log is per HEAD; dt_bias is per CHANNEL. Ranges chosen so `exp(A_log)`
    # spans a decade — a fixture where every head decays at the same rate cannot
    # tell a per-head broadcast from a per-channel one.
    a_log = (rnd(H) * 1.5)
    dt_bias = rnd(H * D)

    has_bound = args.gate_lower_bound != 0.0
    kw = dict(
        A_log=a_log.to(dev),
        dt_bias=dt_bias.to(dev),
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    if has_bound:
        kw["lower_bound"] = args.gate_lower_bound

    o, state = fused_recurrent_kda(
        q=q.to(dev), k=k.to(dev), v=v.to(dev), g=g_raw.to(dev), beta=beta_raw.to(dev), **kw)

    o = o.float().cpu().contiguous()          # [1, T, H, D]
    state = state.float().cpu().contiguous()  # [1, H, D, D] — key axis outer

    if not torch.isfinite(o).all() or not torch.isfinite(state).all():
        print("error: reference produced non-finite values", file=sys.stderr)
        return 3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<IIII", H, D, T, 1 if has_bound else 0)
    blob += struct.pack("<f", args.gate_lower_bound if has_bound else 0.0)
    for t in (a_log, dt_bias, q, k, v, g_raw, beta_raw, o, state):
        blob += t.detach().cpu().contiguous().numpy().astype("<f4").tobytes()
    (out_dir / "kda_oracle.bin").write_bytes(bytes(blob))

    import json
    (out_dir / "meta.json").write_text(json.dumps({
        "producer": "fla.ops.kda.fused_recurrent_kda",
        "fla_version": __import__("fla").__version__ if hasattr(__import__("fla"), "__version__")
                       else "unknown",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "heads": H, "head_dim": D, "tokens": T,
        "gate_lower_bound": args.gate_lower_bound if has_bound else None,
        "flags": ["use_qk_l2norm_in_kernel", "use_gate_in_kernel",
                  "use_beta_sigmoid_in_kernel"] + (["safe_gate"] if has_bound else []),
        "state_layout": "[heads][key][value]",
        "bytes": len(blob),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"  wrote       : {out_dir / 'kda_oracle.bin'}  ({len(blob)} bytes)")
    print(f"  shape       : H={H} D={D} T={T}  gate_lower_bound="
          f"{args.gate_lower_bound if has_bound else 'none (softplus)'}")
    print(f"  |o| range   : {float(o.abs().min()):.3e} .. {float(o.abs().max()):.3e}")
    print(f"  |S| max     : {float(state.abs().max()):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
