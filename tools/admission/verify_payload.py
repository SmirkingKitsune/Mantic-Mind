#!/usr/bin/env python3
"""Verify a Soma container's EXPERT payload against the source checkpoint.

Format: schemas/container.md

Why this exists (roadmap D27): admission passed a 460 GB GLM-5.2 container
without reading one byte of `experts-*.bin`. `quant_codec` round-trips the DENSE
f32 tensors through each declared codec — it validates the codec against the
model's weight distributions and says nothing about the payload on disk. A wrong
gate/up split, an off-by-one shard offset, or a truncated final shard passes
admission today.

Three passes, and they prove different things. Keeping them apart matters,
because the cheap one is the only one that covers everything:

  STRUCTURE  Whole index, no tensor reads, no source needed. Every slot's offset,
             length, alignment, shard membership and packing is checked against
             the shard files that actually exist on disk. This is the pass that
             covers 100% of a 460 GB payload in under a second, and it is what
             catches truncation — the failure most likely to survive a crash.

  EXACT      Sampled. Re-quantizes the source expert and requires the container's
             bytes to match. Catches wrong offset, wrong order, wrong expert,
             wrong gate/up split, short write. Deliberately uses convert.py's own
             `quantize_rows`, so it proves PLACEMENT, not the codec: a codec bug
             would be reproduced identically on both sides and cancel out.

  DECODE     Sampled. Unpacks the stored bytes with the decoder below — written
             against the layout rather than by calling the packer — and compares
             to the source f32. Two numbers, and the second is the one that
             means something:
               - rel_rms against the CORRECT source tensor, which must sit under
                 the codec's ceiling;
               - rel_rms against DECOYS (a neighbouring expert, the sibling
                 projection), which must be far worse.
             The decoy margin is what proves the right tensor is in the right
             slot. It is robust to a bug in this decoder: a broken decoder makes
             correct and decoy equally bad and the margin collapses, so the test
             fails loudly instead of passing vacuously.

Offline only. Never a runtime dependency.

Usage:
    verify_payload.py <container_dir> [--source DIR] [--samples N] [--seed S]
                      [--structure-only] [--json]
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert import (  # noqa: E402
    DTYPE_ID,
    MAGIC,
    align_up,
    expert_reader,
    group_bytes,
    layer_kinds,
    quantize_rows,
    usable_group,
)

ENTRY = struct.Struct("<IQI")

# Mirrors the ceilings stage_quant_codec enforces in src/soma/main.cpp. Same
# codecs, same distributions, so a payload that decodes worse than the codec
# itself can achieve is a payload problem rather than a quantization one.
REL_RMS_CEILING = {"q8_0": 0.03, "q6_g": 0.08, "q5_g": 0.15,
                   "q4_g": 0.30, "q4_0": 0.40, "f32": 1e-6}

ID_TO_DTYPE = {v: k for k, v in DTYPE_ID.items()}


class Failure(Exception):
    pass


# ── the index ────────────────────────────────────────────────────────────────


def read_index(path: Path) -> dict:
    raw = path.read_bytes()
    if raw[:8] != MAGIC:
        raise Failure(f"{path.name}: bad magic {raw[:8]!r}, expected {MAGIC!r}")
    off = 8
    version, reserved = struct.unpack_from("<II", raw, off)
    off += 8
    (hash_len,) = struct.unpack_from("<I", raw, off)
    off += 4
    arch_hash = raw[off:off + hash_len]
    off += hash_len
    n_layers, n_experts, n_shards, dtype_id = struct.unpack_from("<IIII", raw, off)
    off += 16
    (group,) = struct.unpack_from("<I", raw, off)
    off += 4
    uniform_len, total = struct.unpack_from("<QQ", raw, off)
    off += 16

    want = n_layers * n_experts
    have = (len(raw) - off) // ENTRY.size
    if have != want:
        raise Failure(f"index holds {have} slots, header declares "
                      f"{n_layers} x {n_experts} = {want}")
    if (len(raw) - off) % ENTRY.size:
        raise Failure(f"index has {(len(raw) - off) % ENTRY.size} trailing bytes")

    entries = [ENTRY.unpack_from(raw, off + i * ENTRY.size) for i in range(want)]
    return {"version": version, "reserved": reserved, "arch_hash": arch_hash,
            "n_layers": n_layers, "n_experts": n_experts, "n_shards": n_shards,
            "dtype_id": dtype_id, "group": group, "uniform_len": uniform_len,
            "total": total, "entries": entries}


def check_structure(container: Path, ix: dict, meta: dict, kinds: list[str]) -> list[str]:
    """Every slot, against the shard files that exist. No source, no tensor reads."""
    notes: list[str] = []
    n_layers, n_experts = ix["n_layers"], ix["n_experts"]

    if ix["version"] != 1:
        raise Failure(f"container_version {ix['version']}, expected 1")
    for key, got in (("n_layers", n_layers), ("n_experts", n_experts),
                     ("n_shards", ix["n_shards"])):
        if key in meta and int(meta[key]) != got:
            raise Failure(f"index says {key}={got}, container_meta.json says {meta[key]}")

    shard_paths = sorted(container.glob("experts-*.bin"))
    if len(shard_paths) != ix["n_shards"]:
        raise Failure(f"{len(shard_paths)} shard file(s) on disk, index declares "
                      f"{ix['n_shards']}")
    sizes = [p.stat().st_size for p in shard_paths]

    expert_bytes = int(meta.get("expert_bytes", ix["uniform_len"]))
    # Slots grouped by shard so packing can be checked in write order.
    per_shard: dict[int, list[tuple[int, int, int]]] = {}
    live = 0
    summed = 0

    for slot, (shard, off, length) in enumerate(ix["entries"]):
        layer, expert = divmod(slot, n_experts)
        where = f"layer {layer} expert {expert}"

        if kinds[layer] == "dense":
            if length != 0:
                raise Failure(f"{where}: dense layer must hold a zero-length slot, "
                              f"got {length}")
            continue

        live += 1
        summed += length
        if length != expert_bytes:
            raise Failure(f"{where}: length {length}, expected a uniform "
                          f"{expert_bytes}")
        if not 0 <= shard < len(shard_paths):
            raise Failure(f"{where}: shard {shard} out of range")
        if off % 4096:
            raise Failure(f"{where}: offset {off} is not 4 KB aligned")
        end = off + length
        if end > sizes[shard]:
            raise Failure(f"{where}: runs to {end} in {shard_paths[shard].name}, "
                          f"which is only {sizes[shard]} bytes — TRUNCATED")
        per_shard.setdefault(shard, []).append((off, length, slot))

    for shard, slots in per_shard.items():
        slots.sort()
        cursor = 0
        for off, length, slot in slots:
            layer, expert = divmod(slot, n_experts)
            if off != cursor:
                raise Failure(f"layer {layer} expert {expert}: starts at {off}, but the "
                              f"previous slot in shard {shard} ends aligned at {cursor} "
                              f"— {'overlap' if off < cursor else 'gap'}")
            cursor = align_up(off + length)
        # The converter pads after every expert, including the last in a shard, so
        # the file ends exactly on that boundary. A short final file is the
        # signature of a conversion that died mid-write.
        if sizes[shard] != cursor:
            raise Failure(f"{shard_paths[shard].name}: {sizes[shard]} bytes on disk, "
                          f"{cursor} accounted for by the index "
                          f"({'trailing bytes' if sizes[shard] > cursor else 'TRUNCATED'})")

    if "total_expert_bytes" in meta and summed != int(meta["total_expert_bytes"]):
        raise Failure(f"slot lengths sum to {summed}, container_meta.json says "
                      f"{meta['total_expert_bytes']}")
    if ix["total"] and summed != ix["total"]:
        raise Failure(f"slot lengths sum to {summed}, index header says {ix['total']}")

    n_moe = sum(1 for k in kinds if k == "moe")
    notes.append(f"{live} live slots across {len(shard_paths)} shard(s), "
                 f"{summed / 1e9:.3f} GB, {n_moe} MoE layer(s)")
    notes.append(f"{n_layers * n_experts - live} zero-length slots for "
                 f"{n_layers - n_moe} dense layer(s)")
    return notes


# ── the decoder, written against the layout rather than by calling the packer ─


def dequantize_rows(blob: memoryview, rows: int, cols: int, dtype: str, g: int):
    import numpy as np

    if dtype == "f32":
        return np.frombuffer(blob, dtype="<f4", count=rows * cols).reshape(rows, cols)

    nb = cols // g
    gb = group_bytes(dtype, g)
    img = np.frombuffer(blob, dtype=np.uint8, count=rows * nb * gb).reshape(rows, nb, gb)
    scale = img[:, :, 0:4].copy().view("<f4").reshape(rows, nb)

    if dtype == "q4_g":
        lo = img[:, :, 4:8].copy().view("<f4").reshape(rows, nb)
        packed = img[:, :, 8:8 + g // 2]
        lev = np.empty((rows, nb, g), dtype=np.float32)
        lev[:, :, 0::2] = packed & 0x0F
        lev[:, :, 1::2] = packed >> 4
        out = lev * scale[:, :, None] + lo[:, :, None]

    elif dtype == "q4_0":
        packed = img[:, :, 4:4 + g // 2]
        lev = np.empty((rows, nb, g), dtype=np.float32)
        lev[:, :, 0::2] = (packed & 0x0F).astype(np.int8) - 8
        lev[:, :, 1::2] = (packed >> 4).astype(np.int8) - 8
        out = lev * scale[:, :, None]

    elif dtype == "q8_0":
        lev = img[:, :, 4:4 + g].view(np.int8).astype(np.float32)
        out = lev * scale[:, :, None]

    elif dtype == "q6_g":
        n24 = (g + 3) // 4
        trip = img[:, :, 4:4 + n24 * 3].reshape(rows, nb, n24, 3).astype(np.uint32)
        word = trip[..., 0] | (trip[..., 1] << 8) | (trip[..., 2] << 16)
        lev = np.empty((rows, nb, g), dtype=np.float32)
        for k in range(4):
            lev[:, :, k::4] = ((word >> (6 * k)) & 0x3F).astype(np.float32) - 32.0
        out = lev * scale[:, :, None]

    else:
        raise Failure(f"no decoder for dtype {dtype}")

    return out.reshape(rows, cols)


def rel_rms(a, b) -> float:
    import numpy as np

    d = a.astype(np.float64) - b.astype(np.float64)
    ref = float(np.sqrt(np.mean(np.square(b.astype(np.float64)))))
    if ref == 0.0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(d)))) / ref


# ── source access ────────────────────────────────────────────────────────────


def source_reader(src: Path):
    """Lazy per-shard handles, as convert.py does. 282 shards cannot all be open."""
    import torch
    from safetensors import safe_open

    owner: dict[str, str] = {}
    index_json = src / "model.safetensors.index.json"
    if index_json.is_file():
        wm = json.loads(index_json.read_text(encoding="utf-8")).get("weight_map", {})
        owner = {name: str(src / shard) for name, shard in wm.items()}
    if not owner:
        for f in sorted(src.glob("*.safetensors")):
            with safe_open(str(f), framework="pt") as h:
                for k in h.keys():
                    owner[k] = str(f)

    cache: dict[str, object] = {}
    order: list[str] = []

    def handle_for(path: str):
        if path not in cache:
            if len(order) >= 4:
                cache.pop(order.pop(0), None)
            cache[path] = safe_open(path, framework="pt")
            order.append(path)
        return cache[path]

    def get(name: str):
        path = owner.get(name)
        if path is None:
            return None
        return handle_for(path).get_tensor(name).to(torch.float32).numpy().copy()

    return get


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("container")
    ap.add_argument("--source", default=None, help="checkpoint dir (default: from meta)")
    ap.add_argument("--samples", type=int, default=8, help="RANDOM experts to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    container = Path(args.container)
    report: dict = {"container": str(container), "structure": "failed",
                    "content": "skipped", "checked": []}

    try:
        meta = json.loads((container / "container_meta.json").read_text(encoding="utf-8"))
        cfg = json.loads((container / "config.json").read_text(encoding="utf-8"))
        ix = read_index(container / "soma.container")
        kinds = layer_kinds(cfg, ix["n_layers"])
        notes = check_structure(container, ix, meta, kinds)
        report["structure"] = "passed"
        report["structure_notes"] = notes
    except Failure as e:
        report["reason"] = str(e)
        print(json.dumps(report, indent=2) if args.json else f"  FAILED  structure: {e}")
        return 1
    except (OSError, KeyError, ValueError) as e:
        report["reason"] = f"cannot read container: {e}"
        print(json.dumps(report, indent=2) if args.json else f"  FAILED  {report['reason']}")
        return 1

    if not args.json:
        print(f"  {container.name}")
        for n in notes:
            print(f"    structure  {n}")

    if args.structure_only:
        if not args.json:
            print("  OK       structure only; payload contents NOT checked")
        else:
            print(json.dumps(report, indent=2))
        return 0

    src = Path(args.source or meta.get("source", ""))
    if not src.is_dir():
        report["content"] = "skipped"
        report["content_reason"] = f"source checkpoint not readable: {src}"
        if not args.json:
            print(f"    content    skipped — {report['content_reason']}")
            print("  OK       structure passed; contents unverified")
        else:
            print(json.dumps(report, indent=2))
        return 0

    import numpy as np  # noqa: F401  (imported for the decoder's sake)

    n_experts = ix["n_experts"]
    moe_layers = [i for i, k in enumerate(kinds) if k == "moe"]
    dt_gate = meta.get("dtype_gate_up", ID_TO_DTYPE.get(ix["dtype_id"], "q4_g"))
    dt_down = meta.get("dtype_down", dt_gate)
    want_group = int(meta.get("group", ix["group"] or 128))
    moe_block = "block_sparse_moe" if cfg.get("model_type") == "mixtral" else "mlp"
    names = ({"gate": "w1.weight", "up": "w3.weight", "down": "w2.weight"}
             if cfg.get("model_type") == "mixtral"
             else {"gate": "gate_proj.weight", "up": "up_proj.weight",
                   "down": "down_proj.weight"})

    # Sample where offset arithmetic actually breaks: the first expert of every
    # shard, plus the two ends of the payload. Random draws are added on top;
    # they are the least likely of the three to find anything.
    picks: list[tuple[int, int]] = []
    seen_shard: set[int] = set()
    for slot, (shard, _off, length) in enumerate(ix["entries"]):
        if length and shard not in seen_shard:
            seen_shard.add(shard)
            picks.append(divmod(slot, n_experts))
    picks.append((moe_layers[0], 0))
    picks.append((moe_layers[-1], n_experts - 1))
    rng = random.Random(args.seed)
    for _ in range(args.samples):
        picks.append((rng.choice(moe_layers), rng.randrange(n_experts)))
    picks = sorted(set(picks))

    get = source_reader(src)
    shard_files = sorted(container.glob("experts-*.bin"))
    handles = {}

    def slot_bytes(layer: int, expert: int) -> bytes:
        shard, off, length = ix["entries"][layer * n_experts + expert]
        if shard not in handles:
            handles[shard] = open(shard_files[shard], "rb")
        handles[shard].seek(off)
        return handles[shard].read(length)

    failures: list[str] = []
    for layer, expert in picks:
        read, layout = expert_reader(get, layer, moe_block, names)
        if read is None:
            failures.append(f"layer {layer}: no readable experts in the source")
            break

        stored = slot_bytes(layer, expert)
        row = {"layer": layer, "expert": expert, "layout": layout}

        # EXACT — placement, ordering, split, short writes.
        rebuilt = bytearray()
        pieces = []
        for role, dt in (("gate", dt_gate), ("up", dt_gate), ("down", dt_down)):
            t = read(expert, role)
            if t is None:
                failures.append(f"layer {layer} expert {expert}: source is missing {role}")
                break
            packed, g = quantize_rows(t, dt, want_group)
            pieces.append((role, dt, t, g, len(packed)))
            rebuilt += packed
        if len(pieces) != 3:
            break

        row["exact"] = bytes(rebuilt) == stored
        if not row["exact"]:
            where = next((i for i in range(min(len(rebuilt), len(stored)))
                          if rebuilt[i] != stored[i]), min(len(rebuilt), len(stored)))
            failures.append(f"layer {layer} expert {expert}: payload differs from a "
                            f"re-quantized source at byte {where} of {len(stored)} "
                            f"(rebuilt {len(rebuilt)} bytes)")

        # DECODE — is the RIGHT tensor here? The decoy margin is the real assertion.
        cursor = 0
        margins = []
        for (role, dt, t, g, nbytes) in pieces:
            rows, cols = t.shape
            got = dequantize_rows(memoryview(stored)[cursor:cursor + nbytes],
                                  rows, cols, dt, g)
            cursor += nbytes
            err = rel_rms(got, t)
            ceiling = REL_RMS_CEILING.get(dt, 0.5)

            decoys = []
            other = read((expert + 1) % n_experts, role)
            if other is not None and other.shape == t.shape:
                decoys.append(rel_rms(got, other))
            if role != "down":  # down has a different shape; there is no sibling to confuse it with
                sibling = read(expert, "up" if role == "gate" else "gate")
                if sibling is not None and sibling.shape == t.shape:
                    decoys.append(rel_rms(got, sibling))

            row[f"rel_rms_{role}"] = round(err, 6)
            if decoys:
                row[f"decoy_{role}"] = round(min(decoys), 6)
                margins.append(min(decoys) / err if err > 0 else float("inf"))
            if err > ceiling:
                failures.append(f"layer {layer} expert {expert} {role}: decoded "
                                f"rel_rms {err:.4f} exceeds the {dt} ceiling {ceiling}")
            if decoys and min(decoys) < 2.0 * err:
                failures.append(f"layer {layer} expert {expert} {role}: decoded payload is "
                                f"no closer to its own source ({err:.4f}) than to a "
                                f"different tensor ({min(decoys):.4f}) — wrong slot, or "
                                f"this decoder is broken")
        if margins:
            row["decoy_margin"] = round(min(margins), 2)
        report["checked"].append(row)

    for h in handles.values():
        h.close()

    report["content"] = "failed" if failures else "passed"
    if failures:
        report["failures"] = failures[:20]

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for row in report["checked"][:6]:
            bits = " ".join(f"{k}={v}" for k, v in row.items()
                            if k.startswith(("rel_rms", "decoy", "exact")))
            print(f"    layer {row['layer']:>3} expert {row['expert']:>3}  {bits}")
        if len(report["checked"]) > 6:
            print(f"    ... {len(report['checked']) - 6} more sampled")
        if failures:
            print(f"  FAILED  {len(failures)} problem(s):")
            for f in failures[:10]:
                print(f"             {f}")
        else:
            print(f"  OK       structure + {len(report['checked'])} sampled experts "
                  f"({len(seen_shard)} shard-first, 2 edge, {args.samples} random)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
