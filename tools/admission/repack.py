#!/usr/bin/env python3
"""Rewrite a container with its experts in heat order.

A streaming engine's startup cost is dominated by reading the PINNED set — the
experts hot enough to keep resident — and under a conversion-order layout those
sit wherever their index position put them. `soma heat-layout` measures what that
costs: on real geometry it found the pinned set arriving as **32-45x** more
separate read runs than the layout could deliver, because the hot experts of a
layer are scattered through it rather than adjacent.

This writes the same experts in a different order, so the hot set is a contiguous
prefix of each shard and the warm pass is one sequential run per file.

WHAT IT DOES NOT CHANGE, and why each matters:

  * **The expert bytes.** Every payload is copied verbatim. A repack that
    requantized would be a reconversion wearing a disguise.
  * **The digests.** A digest binds `(layer, expert, length)` and the payload —
    NOT the offset — so moving an expert leaves its digest exactly as the
    converter recorded it. The table is copied and re-indexed, and `soma verify`
    on the output checks it against bytes that have moved, which is precisely the
    guarantee worth having here.
  * **The identity.** `arch_hash` covers the architecture and the quantization
    map, neither of which a repack touches, so the output IS the same model: the
    registry keys it the same way and KV checkpoints written against the original
    still gate correctly. The hash is not copied on faith — it is recomputed from
    the output with `soma arch-hash` and required to match.

WHY A NEW DIRECTORY, ALWAYS. Node cache identity is a hash over each file's
`(relpath, size, mtime)`, so rewriting a container in place re-transfers every
shard to every node holding it. A repack in place would be the most expensive
possible way to save a few seconds of startup. A container is written once; this
writes a different one.

Because the two carry the same `arch_hash`, they are the same model to the
registry — so the output REPLACES the input rather than coexisting with it.

Usage:
    repack.py <container> --out DIR --heat FILE [--soma PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert import (ALIGN, align_up, container_identity, find_soma,  # noqa: E402
                     write_index)
from verify_payload import ID_TO_DTYPE, ID_TO_ROLE, Failure, read_index  # noqa: E402


def read_heat(path: Path) -> dict:
    """{(layer, expert): decayed}, as `soma serve` writes its snapshot.

    Same file `soma heat-layout --heat` reads, and deliberately so: the thing
    that measures the problem and the thing that fixes it must be looking at the
    same evidence, or a repack can report an improvement the measurement never
    sees.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    cells = doc.get("experts")
    if not isinstance(cells, list):
        raise Failure(f"{path.name}: no `experts` array")
    heat = {}
    for c in cells:
        if not isinstance(c, dict) or "layer" not in c or "expert" not in c:
            continue
        heat[(int(c["layer"]), int(c["expert"]))] = float(c.get("decayed", 0.0))
    if not heat:
        raise Failure(f"{path.name}: no usable heat cells")
    return heat


def plan_layout(ix: dict, heat: dict) -> tuple[list, dict]:
    """New (shard, offset, length) per slot, hottest first WITHIN each shard.

    Within each shard rather than globally, because a shard is a file and a run
    of reads cannot span two of them: one contiguous hot prefix per file is the
    best any layout can do for the warm pass, and it is what `heat-layout`
    reports as `runs_ideal_global`.

    Slots keep their shard. Moving an expert between shards would change every
    shard's size and buy nothing — the scatter this fixes is WITHIN a file.
    """
    n_experts = ix["n_experts"]
    by_shard: dict[int, list[int]] = {}
    for slot, (shard, _off, length) in enumerate(ix["entries"]):
        if length == 0:
            continue  # a dense layer's empty slot keeps offset 0, length 0
        by_shard.setdefault(shard, []).append(slot)

    entries = [(shard, 0, 0) for shard, _o, _l in ix["entries"]]
    for slot, (shard, _off, length) in enumerate(ix["entries"]):
        entries[slot] = (shard, 0, length)

    moves = {}
    for shard, slots in sorted(by_shard.items()):
        # Hottest first; ties and unmentioned experts fall back to slot order, so
        # a partial heat snapshot produces a deterministic layout rather than an
        # arbitrary one.
        slots.sort(key=lambda s: (-heat.get(divmod(s, n_experts), 0.0), s))
        cursor = 0
        for slot in slots:
            length = ix["entries"][slot][2]
            entries[slot] = (shard, cursor, length)
            moves[slot] = (ix["entries"][slot][1], cursor)
            cursor = align_up(cursor + length)
    return entries, moves


def predicted_runs(entries: list, heat: dict, n_experts: int, pin_bytes: int) -> tuple[int, int]:
    """(runs, shards touched) for the pinned set under a layout.

    The same metric `soma heat-layout` prints, computed here so the tool can
    state what it achieved instead of asserting that it helped.
    """
    ranked = sorted(((h, le) for le, h in heat.items()), reverse=True)
    pinned, used = [], 0
    for _h, (layer, expert) in ranked:
        slot = layer * n_experts + expert
        if slot >= len(entries):
            continue
        shard, off, length = entries[slot]
        if length == 0 or used + length > pin_bytes:
            continue
        used += length
        pinned.append((shard, off, length))
    if not pinned:
        return 0, 0
    pinned.sort()
    runs, shards = 1, {pinned[0][0]}
    for i in range(1, len(pinned)):
        pshard, poff, plen = pinned[i - 1]
        shard, off, _l = pinned[i]
        shards.add(shard)
        if shard != pshard or align_up(poff + plen) != off:
            runs += 1
    return runs, len(shards)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("container")
    ap.add_argument("--out", required=True, help="a NEW directory; never the input")
    ap.add_argument("--heat", required=True, help="heat snapshot, as `soma serve` writes it")
    ap.add_argument("--pin", type=int, default=0,
                    help="pin budget in bytes, for the run counts reported "
                         "(default: a quarter of the routed payload)")
    ap.add_argument("--soma", default=None,
                    help="path to the soma engine, which recomputes the identity "
                         "(default: $MM_SOMA_PATH, then PATH)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what the layout would achieve and write nothing")
    args = ap.parse_args(argv[1:])

    src, out = Path(args.container), Path(args.out)
    if out.resolve() == src.resolve():
        print("  REFUSED  --out is the input. A repack writes a NEW container: "
              "rewriting one in\n           place re-transfers every shard to every "
              "node holding it.")
        return 2

    ix = read_index(src / "soma.container")
    meta = json.loads((src / "container_meta.json").read_text(encoding="utf-8"))
    heat = read_heat(Path(args.heat))
    n_experts = ix["n_experts"]

    pin_bytes = args.pin or max(ix["total"] // 4, 1)
    before = predicted_runs(ix["entries"], heat, n_experts, pin_bytes)
    entries, moves = plan_layout(ix, heat)
    after = predicted_runs(entries, heat, n_experts, pin_bytes)

    print(f"  {src.name}")
    print(f"    pin budget  {pin_bytes / 1e6:.1f} MB")
    print(f"    runs        {before[0]} -> {after[0]}  "
          f"(ideal {after[1]}, one per shard the pinned set touches)")
    moved = sum(1 for s, (a, b) in moves.items() if a != b)
    print(f"    moving      {moved} of {len(moves)} live experts")
    if args.dry_run:
        print("  OK       dry run; nothing written")
        return 0

    if out.exists() and any(out.iterdir()):
        print(f"  REFUSED  {out} exists and is not empty")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    # Everything that is not a routed shard or the index, verbatim. The resident
    # half, the config, the record and the tokenizer describe a model this does
    # not change.
    for item in sorted(src.iterdir()):
        if not item.is_file():
            continue
        if item.name == "soma.container" or item.name.startswith("experts-"):
            continue
        shutil.copy2(item, out / item.name)

    # The shards, expert by expert, in the new order. Read in SOURCE offset order
    # per shard so the input is a sequential sweep; the output is written in the
    # new order, which is sequential by construction.
    for shard in sorted({e[0] for e in entries if e[2] != 0}):
        name = f"experts-{shard:05d}.bin"
        placed = sorted(((entries[s][1], s) for s in range(len(entries))
                         if entries[s][0] == shard and entries[s][2] != 0))
        with open(src / name, "rb") as fin, open(out / name, "wb") as fout:
            for new_off, slot in placed:
                old_shard, old_off, length = ix["entries"][slot]
                fin.seek(old_off)
                blob = fin.read(length)
                if len(blob) != length:
                    raise Failure(f"{name}: short read of {length} B at {old_off}")
                assert fout.tell() == new_off, "layout and write disagree"
                fout.write(blob)
                pad = align_up(new_off + length) - (new_off + length)
                if pad:
                    fout.write(b"\0" * pad)
        print(f"    wrote       {name}")

    roles = ix["roles"]
    role_groups = {ID_TO_ROLE[r]: g for r, (_d, g) in roles.items()}
    dt = {ID_TO_ROLE[r]: ID_TO_DTYPE[d] for r, (d, _g) in roles.items()}
    # The digests are COPIED, not recomputed. They bind (layer, expert, length)
    # and the payload, none of which moved; recomputing them here would make this
    # tool an authority on what the bytes are, which it is not. `soma verify`
    # below checks them against the relocated bytes, which is the real test.
    write_index(out / "soma.container", arch_hash=ix["arch_hash"],
                n_layers=ix["n_layers"], n_experts=n_experts,
                n_shards=ix["n_shards"], dt_gate=dt["gate"], dt_up=dt["up"],
                dt_down=dt["down"], role_groups=role_groups,
                requested_group=int(meta["group"]), uniform_len=ix["uniform_len"],
                total_bytes=ix["total"], entries=entries, digests=ix["digests"])

    # The identity, recomputed rather than assumed. A repack changes no field the
    # hash covers, so a difference here means this tool changed something it had
    # no business changing.
    exe = find_soma(args.soma)
    if exe is None:
        print("  WARNING  no soma engine found, so the identity was copied without "
              "being\n           rechecked. Run `soma verify` on the output.")
    else:
        recomputed = container_identity(out, args.soma)
        if recomputed != ix["arch_hash"]:
            print(f"  FAILED   the repacked container hashes to "
                  f"{recomputed.decode()[:16]}... and the source to "
                  f"{ix['arch_hash'].decode()[:16]}...; a repack must not move the "
                  f"identity")
            return 1
        r = subprocess.run([exe, "verify", str(out)], capture_output=True, text=True)
        print("".join(f"    {line}\n" for line in r.stdout.strip().splitlines()))
        if r.returncode != 0:
            print(f"  FAILED   the repacked container does not verify: {r.stderr.strip()}")
            return 1

    print(f"  OK       {out} holds the same experts in {after[0]} run(s) instead of "
          f"{before[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
