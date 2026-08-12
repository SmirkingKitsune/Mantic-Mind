#!/usr/bin/env python3
"""Dump the transformers reference's per-layer activations, and diff them
against the engine's.

Why this exists
---------------
Whole-model logit comparison localises a defect to "the model is wrong" and
nothing more. G4 spent four rounds of hypothesise-code-build-measure narrowing an
MLA defect to "somewhere positional", and each round was a guess. A per-layer
diff replaces guessing with bisection: the first layer whose input matches and
whose output does not IS the layer with the bug, and the attention tap inside it
says whether it is attention or the FFN.

Taps match the engine's exactly (see F32Workspace::Sink):

    hidden_in    the residual stream entering the layer
    attn_out     attention's contribution, BEFORE the residual add
    hidden_out   the residual stream leaving the layer

Usage
-----
    dump_activations.py <fixture_dir> --out ref.somaact [--positions N]
    dump_activations.py --diff ref.somaact engine.somaact
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"SOMAACT1"


# ── container ────────────────────────────────────────────────────────────────
#
# Deliberately trivial: a length-prefixed list of (layer, point, float32[]). Both
# sides write it, one reader diffs it, and there is no version negotiation to get
# wrong for a debug artefact.

def write_records(path: Path, records: list[tuple[int, str, np.ndarray]]) -> None:
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(records)))
        for layer, point, data in records:
            name = point.encode()
            flat = np.ascontiguousarray(data, dtype=np.float32).ravel()
            f.write(struct.pack("<I", layer & 0xFFFFFFFF))
            f.write(struct.pack("<I", len(name)))
            f.write(name)
            f.write(struct.pack("<I", flat.size))
            f.write(flat.tobytes())


def read_records(path: Path) -> list[tuple[int, str, np.ndarray]]:
    raw = path.read_bytes()
    if raw[:8] != MAGIC:
        raise SystemExit(f"{path}: bad magic")
    at = 8
    (count,) = struct.unpack_from("<I", raw, at)
    at += 4
    out = []
    for _ in range(count):
        layer, nlen = struct.unpack_from("<II", raw, at)
        at += 8
        point = raw[at : at + nlen].decode()
        at += nlen
        (n,) = struct.unpack_from("<I", raw, at)
        at += 4
        data = np.frombuffer(raw, dtype=np.float32, count=n, offset=at).copy()
        at += n * 4
        out.append((layer, point, data))
    return out


# ── reference side ───────────────────────────────────────────────────────────

def dump_reference(fixture: Path, out: Path, positions: int) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(fixture)
    model = AutoModelForCausalLM.from_pretrained(
        fixture, config=cfg, torch_dtype=torch.float32
    )
    model.eval()

    # The oracle's own token ids. Inventing a sequence here would make the two
    # dumps incomparable while still producing a plausible-looking table.
    raw = (fixture / "oracle.bin").read_bytes()
    if raw[:8] != b"SOMAORCL":
        raise SystemExit("oracle.bin: bad magic")
    avail = struct.unpack_from("<I", raw, 12)[0]
    ids = np.frombuffer(raw, dtype=np.int32, count=avail, offset=28)
    n = min(positions, avail)
    tokens = torch.tensor(ids[:n].astype(np.int64)).unsqueeze(0)

    records: list[tuple[int, str, np.ndarray]] = []
    handles = []

    def layer_hook(idx):
        def fn(_mod, args, output):
            # `args[0]` is hidden_states entering the layer; the layer's output
            # is a tuple whose first element is the residual stream leaving it.
            hin = args[0] if args else None
            if hin is not None:
                records.append((idx, "hidden_in", hin.detach().float().numpy()))
            hout = output[0] if isinstance(output, tuple) else output
            records.append((idx, "hidden_out", hout.detach().float().numpy()))
        return fn

    def attn_hook(idx):
        def fn(_mod, _args, output):
            o = output[0] if isinstance(output, tuple) else output
            records.append((idx, "attn_out", o.detach().float().numpy()))
        return fn

    def module_out_hook(idx, name):
        def fn(_mod, _args, output):
            o = output[0] if isinstance(output, tuple) else output
            records.append((idx, name, o.detach().float().numpy()))
        return fn

    def module_in_hook(idx, name):
        def fn(_mod, args):
            if args:
                records.append((idx, name, args[0].detach().float().numpy()))
        return fn

    # Sub-layer taps, matched to the engine's by NAME. These land on module
    # boundaries specifically so both sides can produce them without reshaping:
    # a tap in the middle of a fused step is cheap in C++ and unhookable here.
    #
    # `o_proj` is taken on its INPUT rather than its output, because that is the
    # only observable between the attention math and the output projection — and
    # if every projection matches while this does not, the fault is in the
    # rope/score/softmax/value path, which no module boundary can subdivide.
    SUB = [
        ("q_proj", "out"),
        ("kv_a_proj_with_mqa", "out"),
        ("kv_a_layernorm", "out"),
        ("kv_b_proj", "out"),
        ("o_proj", "in"),
    ]
    # The engine's tap names, in the same order.
    ENGINE_NAME = {
        "q_proj": "q_proj",
        "kv_a_proj_with_mqa": "kv_a_proj",
        "kv_a_layernorm": "kv_a_layernorm",
        "kv_b_proj": "kv_b_proj",
        "o_proj": "o_proj_in",
    }

    layers = model.model.layers
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(layer_hook(i), with_kwargs=False))
        handles.append(layer.self_attn.register_forward_hook(attn_hook(i)))
        for attr, when in SUB:
            mod = getattr(layer.self_attn, attr, None)
            if mod is None:
                continue   # absent on this family (e.g. q_a/q_b vs q_proj)
            tap = ENGINE_NAME[attr]
            if when == "out":
                handles.append(mod.register_forward_hook(module_out_hook(i, tap)))
            else:
                handles.append(mod.register_forward_pre_hook(module_in_hook(i, tap)))

    # ── rope taps, by monkey-patch ───────────────────────────────────────────
    #
    # q_pe/k_pe after rotation are intermediate tensors: no module produces them,
    # so no hook can see them. They are also the last unobserved step before the
    # attention scores, which makes them the taps that matter most here.
    #
    # The patch is scoped to this run and restored in `finally`. A layer counter
    # rather than a passed-in index, because apply_rotary_pos_emb is a free
    # function with no idea which layer called it — it is called exactly once per
    # layer, in order, which is the only reason this is sound.
    import importlib

    def _reinterleave(t):
        """cat(rot(even), rot(odd)) -> interleaved (even, odd, even, odd, ...)."""
        half = t.shape[-1] // 2
        out = t.new_empty(t.shape)
        out[..., 0::2] = t[..., :half]
        out[..., 1::2] = t[..., half:]
        return out

    mod = importlib.import_module(type(model.model.layers[0]).__module__)
    patched: list[tuple[str, object]] = []
    call_count = {"n": 0}

    # Name varies by family and by transformers version. `apply_rotary_emb` is
    # what 4.57's native DeepseekV2 uses — a COMPLEX-valued form
    # (`view_as_complex` over adjacent pairs), which is the interleaved
    # convention. The absence of the others is not an error; the absence of ALL
    # of them is, and is reported rather than silently producing no taps.
    for fname in ("apply_rotary_emb", "apply_rotary_pos_emb",
                  "apply_rotary_pos_emb_interleave"):
        orig = getattr(mod, fname, None)
        if orig is None or not callable(orig):
            continue

        def make(orig_fn, label):
            def wrapper(q, k, *a, **kw):
                out = orig_fn(q, k, *a, **kw)
                qe, ke = out[0], out[1]

                # DSA calls rope TWICE per `full` layer — once for the main
                # attention (the interleave variant) and once for the indexer
                # (the half-split one). A single shared counter therefore stopped
                # meaning "layer" at GLM-5.2's very first layer, and every tap
                # after it was filed against the wrong layer: q_pe_rot alternated
                # between 4096 and 32768 elements, which is 4 attention heads and
                # 32 INDEXER heads.
                #
                # The indexer's rotation is not one of these taps, so it is
                # counted separately and not recorded. The counter that matters
                # advances once per layer again.
                if label == "apply_rotary_pos_emb" and "apply_rotary_pos_emb_interleave" in {
                        f for f, _ in patched}:
                    call_count["indexer"] = call_count.get("indexer", 0) + 1
                    return out

                idx = call_count["n"]
                call_count["n"] += 1

                # The interleave variant READS interleaved pairs and WRITES them
                # concatenated: out = cat(rot(even), rot(odd)). Same values, a
                # permuted layout — which is invisible to attention, because q and
                # k are permuted identically and their dot product is unchanged,
                # and highly visible to a tap. The engine keeps the interleaved
                # layout, so undo the permutation here rather than report a
                # divergence of 1.6 that no amount of reading the rotation code
                # explains.
                if label == "apply_rotary_pos_emb_interleave":
                    qe, ke = _reinterleave(qe), _reinterleave(ke)

                records.append((idx, "q_pe_rot", qe.detach().float().numpy()))
                records.append((idx, "k_pe_rot", ke.detach().float().numpy()))
                return out
            wrapper.__name__ = label
            return wrapper

        patched.append((fname, orig))
        setattr(mod, fname, make(orig, fname))

    if not patched:
        print(f"  warning: no apply_rotary_pos_emb* found in {mod.__name__};"
              " rope taps will be absent", file=sys.stderr)

    try:
        with torch.no_grad():
            logits = model(tokens).logits
    finally:
        for fname, orig in patched:
            setattr(mod, fname, orig)
        for h in handles:
            h.remove()

    # Records arrive in hook order (attn before the enclosing layer's output);
    # sorted by (layer, point) so both sides agree without depending on it.
    records.append((0xFFFFFFFF, "logits", logits.detach().float().numpy()))
    write_records(out, records)
    print(f"wrote {len(records)} records for {n} positions to {out}")


# ── diff ─────────────────────────────────────────────────────────────────────

def diff(ref_path: Path, eng_path: Path) -> int:
    ref = {(l, p): d for l, p, d in read_records(ref_path)}
    eng = {(l, p): d for l, p, d in read_records(eng_path)}

    keys = sorted(set(ref) & set(eng), key=lambda k: (k[0], k[1]))
    if not keys:
        print("no common taps — the two dumps do not describe the same run")
        return 2

    print(f"{'layer':>6}  {'tap':<12} {'n':>8}  {'max|diff|':>11}  {'mean|diff|':>11}")
    print("-" * 56)

    # Execution order, so "first divergence" means what it says.
    ORDER = {
        "hidden_in": 0,
        "q_proj": 1,
        "kv_a_proj": 2,
        "kv_a_layernorm": 3,
        "kv_b_proj": 4,
        "k_pe_rot": 5,
        "q_pe_rot": 6,
        "o_proj_in": 7,
        "attn_out": 8,
        "hidden_out": 9,
    }
    keys.sort(key=lambda k: (k[0], ORDER.get(k[1], 9)))

    first_bad = None
    for layer, point in keys:
        a, b = ref[(layer, point)], eng[(layer, point)]
        if a.size != b.size:
            print(f"{layer:>6}  {point:<12} {a.size:>8}  SIZE MISMATCH vs {b.size}")
            continue
        d = np.abs(a - b)
        mx, mean = float(d.max()), float(d.mean())
        tag = "" if mx < 1e-4 else "   <-- DIVERGES"
        if mx >= 1e-4 and first_bad is None:
            first_bad = (layer, point)
        name = "logits" if layer == 0xFFFFFFFF else str(layer)
        print(f"{name:>6}  {point:<12} {a.size:>8}  {mx:>11.3e}  {mean:>11.3e}{tag}")

    print()
    if first_bad is None:
        print("OK: every tap agrees to 1e-4.")
        return 0

    layer, point = first_bad
    print(f"FIRST DIVERGENCE: layer {layer}, tap '{point}'")
    WHY = {
        "hidden_in": "The layer was handed bad input — the fault is upstream of it.",
        "q_proj": "The query projection itself. Check the weight binding and the\n"
                  "  [n_heads * (nope+rope), d_model] shape assumption.",
        "kv_a_proj": "kv_a_proj_with_mqa. Its output is latent ++ shared-rope; a\n"
                     "  wrong split point would show here.",
        "kv_a_layernorm": "The latent norm. Everything feeding it matched, so this is\n"
                          "  the norm itself — weight, eps, or normalising the wrong slice.",
        "kv_b_proj": "The latent expansion. Its output layout is per-head\n"
                     "  (K-nope ++ V), h-major — the assumption most likely wrong here.",
        "k_pe_rot": "The shared rope segment after rotation. Projections all matched,\n"
                    "  so this is the rotation itself: pairing, frequencies, or position.",
        "q_pe_rot": "The query rope segments after rotation. If k_pe_rot matched and\n"
                    "  this did not, the fault is in the per-head q path, not the\n"
                    "  rotation formula.",
        "o_proj_in": "Every projection matched and the attention OUTPUT did not, so\n"
                     "  the fault is in the math no module boundary can subdivide:\n"
                     "  rope application, the score scale, the softmax, or the value\n"
                     "  accumulation.",
        "attn_out": "Input was good and attention's output is not. The defect is in\n"
                    "  this backend's attention, not in the FFN or the router.",
        "hidden_out": "Input and attention are both good, so the FFN/MoE block is where\n"
                      "  divergence starts: routing, expert application, or the shared expert.",
    }
    print("  " + WHY.get(point, "No guidance for this tap."))
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture", nargs="?", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--diff", nargs=2, type=Path, metavar=("REF", "ENGINE"))
    args = ap.parse_args()

    if args.diff:
        return diff(args.diff[0], args.diff[1])
    if not args.fixture or not args.out:
        ap.error("need <fixture> --out <file>, or --diff REF ENGINE")
    dump_reference(args.fixture, args.out, args.positions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
