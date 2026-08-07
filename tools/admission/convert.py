#!/usr/bin/env python3
"""Convert an HF checkpoint into a Soma container.

Format: schemas/container.md

Everything here exists to make an expert cache miss cheap. The three things that
cost real time on every miss are a seek per tensor, an unaligned read, and a JSON
header parse — so experts are concatenated, 4 KB-aligned, and indexed in a
sidecar.

Offline only. Never a runtime dependency.

Usage:
    convert.py <model_dir> --out <container_dir> [--quant q4_g] [--group 128]
               [--expert-down q6_g] [--shard-bytes 4294967296] [--layers N]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

MAGIC = b"SOMACTNR"
FORMAT_VERSION = 1
ALIGN = 4096

# Mirrors soma::DType. Kept as an explicit table rather than an enum import
# because the engine's header is the authority and this must fail loudly if the
# two drift, not silently renumber.
DTYPE_ID = {"f32": 0, "f16": 1, "bf16": 2, "q8_0": 3, "q6_g": 4, "q5_g": 5,
            "q4_g": 6, "q4_0": 7}

# Roles whose tensors stay resident. Everything not listed here and not a routed
# expert is an error rather than a silent omission — a dropped tensor produces a
# model that loads and is wrong.
DENSE_SUFFIXES = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "mlp.gate.weight", "block_sparse_moe.gate.weight",
)


def align_up(n: int) -> int:
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def fused_expert_diagnosis(get, layer: int, moe_block: str) -> str | None:
    """Is this checkpoint using the FUSED expert layout, rather than missing a tensor?

    "missing model.layers.0.mlp.experts.0.gate_proj.weight" sends a reader
    looking for a corrupt download. The truth is usually that the checkpoint
    stacks every expert into one 3-D tensor — a layout recent `transformers`
    emits and this converter does not read — and the converter can SEE that,
    because the fused tensor is sitting right there under a different name.

    Naming the real problem is the whole of this function. Reading the layout is
    deliberately NOT attempted; see the message it produces.
    """
    prefix = f"model.layers.{layer}.{moe_block}.experts."
    fused = [(n, tuple(t.shape))
             for n in (prefix + "gate_up_proj", prefix + "down_proj",
                       prefix + "gate_proj", prefix + "up_proj")
             if (t := get(n)) is not None]
    if not fused:
        return None
    shapes = ", ".join(f"{n.split('.')[-1]}{s}" for n, s in fused)
    return "\n           ".join([
        f"{prefix}<i>.* not found, but this checkpoint carries the FUSED expert "
        f"layout ({shapes}) - every expert stacked into one 3-D tensor. This "
        f"converter reads the per-expert layout only.",
        "NOT a corrupt download, and not something to work around by guessing: "
        "the fused convention is family-specific and the ones in the wild "
        "disagree. gpt_oss stores [experts, in, 2*inter] and splits gate/up "
        "INTERLEAVED; this checkpoint is [experts, 2*inter, in]. A wrong guess "
        "produces a container that loads, runs, and is silently wrong.",
        "Re-export in the per-expert layout, or add support here together with "
        "an oracle that can verify it.",
    ])


def layer_kinds(cfg: dict, n_layers: int) -> list[str]:
    """Which layers are MoE. Mirrors resolve_layer_kinds() in src/soma/arch_ir.cpp.

    Not every layer has experts. DeepSeek sets first_k_dense_replace=1, so layer 0
    is a plain FFN and asking for its experts fails. Qwen3 expresses the same idea
    with decoder_sparse_step + mlp_only_layers, Mixtral with nothing at all.

    Dense layers still get index slots, with length 0 — dropping them would shift
    every subsequent (layer, expert) lookup by one and produce a container that
    reads the wrong expert for every layer after the first dense one.
    """
    kinds = ["moe"] * n_layers
    first_dense = int(cfg.get("first_k_dense_replace", 0) or 0)
    for i in range(min(first_dense, n_layers)):
        kinds[i] = "dense"

    step = max(1, int(cfg.get("decoder_sparse_step", 1) or 1))
    mlp_only = set(cfg.get("mlp_only_layers", []) or [])
    freq = cfg.get("moe_layer_freq", 1)
    for i in range(n_layers):
        if kinds[i] == "dense":
            continue
        if i in mlp_only:
            kinds[i] = "dense"
        elif step > 1 and i % step != 0:
            kinds[i] = "dense"
        elif isinstance(freq, int) and freq > 1 and i % freq != 0:
            kinds[i] = "dense"
    return kinds


# ── quantization (mirrors src/soma/quant_format.cpp) ─────────────────────────
#
# Deliberately a second implementation of the same formats, in a different
# language. The container round-trip test compares this writer's bytes against
# the engine's reader, so a divergence in either direction shows up as a
# mismatch rather than as both sides agreeing on the same mistake.

def usable_group(cols: int, want: int) -> int:
    g = min(want, cols)
    while g > 1 and cols % g != 0:
        g -= 1
    return max(g, 1)


def group_bytes(dtype: str, g: int) -> int:
    if dtype == "q8_0":
        return 4 + g
    if dtype == "q4_0":
        return 4 + (g + 1) // 2
    if dtype == "q4_g":
        return 8 + (g + 1) // 2
    if dtype == "q6_g":
        return 4 + (g * 3 + 3) // 4
    if dtype == "f32":
        return 4 * g
    raise ValueError(f"unsupported dtype {dtype}")


def _round_half_away(x):
    """Round half away from zero, matching C++ std::lround.

    numpy's rint() rounds half to EVEN. std::lround rounds half AWAY FROM ZERO.
    They disagree on every .5 boundary — rint(0.5)=0 vs lround(0.5)=1 — and one
    weight landing exactly there in one block of one expert is enough to break
    byte-identity between this writer and the engine's reader.
    That is not a theoretical concern: it is how this was found. Mixtral's
    up-projection, layer 0, expert 3, row 18 hit it while OLMoE and Qwen3 (whose
    group size and value distribution differ) happened not to.
    The engine is the authority here — it is what serves — so Python matches C++,
    not the other way around.
    """
    import numpy as np
    return np.trunc(x + np.copysign(0.5, x))


def quantize_rows(mat, dtype: str, want_group: int) -> tuple[bytes, int]:
    """mat: numpy [rows, cols] float32 -> (packed bytes, group)."""
    import numpy as np

    rows, cols = mat.shape
    g = usable_group(cols, want_group)

    if dtype == "f32":
        return mat.astype("<f4").tobytes(), g

    blocks = mat.reshape(rows, cols // g, g).astype(np.float32)

    if dtype in ("q8_0", "q4_0", "q6_g"):
        maxlev = {"q8_0": 127, "q4_0": 7, "q6_g": 31}[dtype]
        amax = np.abs(blocks).max(axis=2)
        scale = np.where(amax > 0, amax / maxlev, 0.0).astype(np.float32)
        inv = np.where(scale > 0, 1.0 / scale, 0.0)
        lev = np.clip(_round_half_away(blocks * inv[:, :, None]),
                      -maxlev, maxlev).astype(np.int32)
    else:  # q4_g, asymmetric
        lo = blocks.min(axis=2)
        hi = blocks.max(axis=2)
        scale = np.where(hi > lo, (hi - lo) / 15.0, 0.0).astype(np.float32)
        inv = np.where(scale > 0, 1.0 / scale, 0.0)
        lev = np.clip(_round_half_away((blocks - lo[:, :, None]) * inv[:, :, None]),
                      0, 15).astype(np.int32)

    # Vectorized packing.
    #
    # The obvious per-(row, block) Python loop is unusable at production scale:
    # Qwen3-30B is ~226 million iterations, which is hours. Everything below
    # builds the full byte image with numpy and writes it once.
    #
    # Layout per group is [scale][(min)][payload], so the output is a
    # (rows, n_blocks, group_bytes) uint8 image whose columns are assigned in
    # slices.
    nb = cols // g
    gb = group_bytes(dtype, g)
    img = np.zeros((rows, nb, gb), dtype=np.uint8)

    # f32 scale occupies the first 4 bytes of every group.
    img[:, :, 0:4] = scale.astype("<f4").view(np.uint8).reshape(rows, nb, 4)
    payload = 4

    if dtype == "q4_g":
        img[:, :, 4:8] = lo.astype("<f4").view(np.uint8).reshape(rows, nb, 4)
        payload = 8

    if dtype == "q8_0":
        img[:, :, payload:payload + g] = lev.astype(np.int8).view(np.uint8)

    elif dtype in ("q4_0", "q4_g"):
        nib = ((lev + 8) & 0x0F) if dtype == "q4_0" else (lev & 0x0F)
        nib = nib.astype(np.uint8)
        img[:, :, payload:payload + g // 2] = nib[:, :, 0::2] | (nib[:, :, 1::2] << 4)

    elif dtype == "q6_g":
        u = ((lev + 32) & 0x3F).astype(np.uint32)
        packed = (u[:, :, 0::4] | (u[:, :, 1::4] << 6) |
                  (u[:, :, 2::4] << 12) | (u[:, :, 3::4] << 18))
        trip = np.empty((rows, nb, packed.shape[2], 3), dtype=np.uint8)
        trip[:, :, :, 0] = (packed & 0xFF).astype(np.uint8)
        trip[:, :, :, 1] = ((packed >> 8) & 0xFF).astype(np.uint8)
        trip[:, :, :, 2] = ((packed >> 16) & 0xFF).astype(np.uint8)
        img[:, :, payload:payload + packed.shape[2] * 3] = trip.reshape(rows, nb, -1)

    return img.tobytes(), g



def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="q4_g", help="gate/up dtype")
    ap.add_argument("--expert-down", default=None, help="down dtype (defaults to --quant)")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--shard-bytes", type=int, default=4 * 1024 ** 3)
    ap.add_argument("--layers", type=int, default=0, help="limit layers (debug)")
    args = ap.parse_args(argv[1:])

    import numpy as np
    import torch
    from safetensors import safe_open

    src = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    model_type = cfg.get("model_type")

    n_layers = int(cfg.get("num_hidden_layers", 0))
    if args.layers:
        n_layers = min(n_layers, args.layers)

    n_experts = 0
    for key in ("num_experts", "num_local_experts", "n_routed_experts"):
        if cfg.get(key):
            n_experts = int(cfg[key])
            break
    if n_experts == 0:
        print(f"  REFUSED  {src.name}: no routed experts; nothing to stream")
        return 3

    moe_block = "block_sparse_moe" if model_type == "mixtral" else "mlp"
    names = ({"gate": "w1.weight", "up": "w3.weight", "down": "w2.weight"}
             if model_type == "mixtral"
             else {"gate": "gate_proj.weight", "up": "up_proj.weight",
                   "down": "down_proj.weight"})

    dt_gate = args.quant
    dt_up = args.quant
    dt_down = args.expert_down or args.quant
    for d in (dt_gate, dt_up, dt_down):
        if d not in DTYPE_ID:
            print(f"  REFUSED  unknown dtype {d!r}")
            return 3

    # Open every shard once; safetensors keeps them mmapped.
    shard_files = sorted(src.glob("*.safetensors"))
    if not shard_files:
        print(f"  REFUSED  {src.name}: no safetensors")
        return 3
    # framework="pt", not "np".
    #
    # Real checkpoints are bf16 and numpy has no bfloat16 dtype, so the numpy
    # backend simply cannot represent them. torch can, and .float() is an exact
    # widening — bf16 is the top 16 bits of an f32, so nothing is lost.
    handles = [safe_open(str(f), framework="pt") for f in shard_files]
    owner: dict[str, int] = {}
    for i, h in enumerate(handles):
        for k in h.keys():
            owner[k] = i

    def get(name: str):
        i = owner.get(name)
        if i is None:
            return None
        # .copy() is NOT optional, and its absence is invisible on f32 inputs.
        #
        # get_tensor() returns an mmap-backed torch tensor. On an f32 checkpoint
        # .to(float32) is a no-op that returns that same tensor, whose storage
        # lives as long as the safe_open handle — so .numpy() is safe.
        #
        # On a BF16 checkpoint .to(float32) ALLOCATES, and .numpy() is a view into
        # that new tensor's storage. The tensor is freed the moment this function
        # returns, leaving a dangling view. Large arrays often survive by luck
        # (the allocator has not reused the pages yet); small ones are clobbered
        # immediately.
        #
        # Observed exactly that way: a real Qwen3 conversion where embed_tokens
        # and lm_head (1.2 GB each) were correct while every per-layer tensor was
        # all zeros. The engine then emitted a uniform distribution — KL 11.93
        # nats against a reference, which is ln(151936) to five figures.
        return handles[i].get_tensor(name).to(torch.float32).numpy().copy()

    # ── experts ──────────────────────────────────────────────────────────────
    index: list[tuple[int, int, int]] = []
    shard_idx = 0
    shard_off = 0
    total = 0
    fh = open(out_dir / f"experts-{shard_idx:05d}.bin", "wb")
    uniform_len = -1
    groups: dict[str, int] = {}

    kinds = layer_kinds(cfg, n_layers)
    n_moe = sum(1 for k in kinds if k == "moe")
    print(f"  {src.name}: {n_layers} layers ({n_moe} MoE) x {n_experts} experts, "
          f"gate/up={dt_gate} down={dt_down} group={args.group}")

    for layer in range(n_layers):
        if kinds[layer] == "dense":
            # Zero-length slots keep the (layer, expert) index arithmetic intact.
            index.extend((shard_idx, shard_off, 0) for _ in range(n_experts))
            continue
        for e in range(n_experts):
            base = f"model.layers.{layer}.{moe_block}.experts.{e}."
            blob = bytearray()
            for role, dt in (("gate", dt_gate), ("up", dt_up), ("down", dt_down)):
                t = get(base + names[role])
                if t is None:
                    fh.close()
                    fused = fused_expert_diagnosis(get, layer, moe_block)
                    if fused:
                        print(f"  REFUSED  {fused}")
                    else:
                        print(f"  REFUSED  missing {base + names[role]}")
                    return 3
                packed, g = quantize_rows(t, dt, args.group)
                groups[dt] = g
                blob += packed

            if shard_off + len(blob) > args.shard_bytes and shard_off > 0:
                fh.close()
                shard_idx += 1
                shard_off = 0
                fh = open(out_dir / f"experts-{shard_idx:05d}.bin", "wb")

            index.append((shard_idx, shard_off, len(blob)))
            fh.write(blob)
            total += len(blob)

            # Pad to the next 4 KB boundary so the NEXT expert starts aligned.
            pad = align_up(shard_off + len(blob)) - (shard_off + len(blob))
            if pad:
                fh.write(b"\0" * pad)
            shard_off = align_up(shard_off + len(blob))

            if uniform_len == -1:
                uniform_len = len(blob)
            elif uniform_len != len(blob):
                uniform_len = 0
        if (layer + 1) % 8 == 0 or layer + 1 == n_layers:
            print(f"    layer {layer + 1}/{n_layers}  {total / 1e9:.2f} GB", flush=True)
    fh.close()

    # ── dense half ───────────────────────────────────────────────────────────
    from safetensors.numpy import save_file

    dense: dict[str, "np.ndarray"] = {}
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        t = get(name)
        if t is not None:
            dense[name] = t
    for layer in range(n_layers):
        for suf in DENSE_SUFFIXES:
            name = f"model.layers.{layer}.{suf}"
            t = get(name)
            if t is not None:
                dense[name] = t
        for shared in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{layer}.{moe_block}.shared_experts.{shared}.weight"
            t = get(name)
            if t is not None:
                dense[name] = t
    save_file(dense, str(out_dir / "dense.safetensors"))

    # ── index ────────────────────────────────────────────────────────────────
    # arch_hash is left empty here: it is computed by the engine from the
    # canonical IR, and inventing a second hash function in Python would give two
    # answers that agree until they do not.
    arch_hash = b""
    with open(out_dir / "soma.container", "wb") as ix:
        ix.write(MAGIC)
        ix.write(struct.pack("<II", FORMAT_VERSION, 0))
        ix.write(struct.pack("<I", len(arch_hash)))
        ix.write(arch_hash)
        ix.write(struct.pack("<IIII", n_layers, n_experts, shard_idx + 1, DTYPE_ID[dt_gate]))
        ix.write(struct.pack("<I", groups.get(dt_gate, args.group)))
        ix.write(struct.pack("<QQ", max(uniform_len, 0), total))
        for shard, off, length in index:
            ix.write(struct.pack("<IQI", shard, off, length))

    # The container must be self-describing: load_f32_model() adapts the IR from
    # config.json in the directory it is pointed at, so a container without one
    # cannot be opened at all.
    import shutil
    shutil.copy2(src / "config.json", out_dir / "config.json")

    meta = {
        "container_version": FORMAT_VERSION,
        "source": str(src),
        "model_type": model_type,
        "n_layers": n_layers,
        "n_moe_layers": n_moe,
        "layer_kinds": kinds,
        "n_experts": n_experts,
        "n_shards": shard_idx + 1,
        "expert_bytes": uniform_len,
        "total_expert_bytes": total,
        "dtype_gate_up": dt_gate,
        "dtype_down": dt_down,
        "group": args.group,
        "effective_groups": groups,
        "dense_tensors": len(dense),
        "align": ALIGN,
    }
    (out_dir / "container_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    padded = sum(align_up(l) for _, _, l in index)
    print(f"  OK       {len(index)} experts, {total / 1e9:.3f} GB payload, "
          f"{(padded - total) / 1e6:.1f} MB padding ({100.0 * (padded - total) / max(total, 1):.3f}%), "
          f"{shard_idx + 1} shard(s), dense {len(dense)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
