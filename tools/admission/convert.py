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
import re
import struct
import sys
from pathlib import Path
from typing import Any

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
    # ── GQA / MHA attention ──
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    # ── MLA attention ──
    #
    # Absent until now, so every MLA container was written without the tensors
    # that actually do its attention and died at load with "binding attention
    # weights failed at layer 0". G4 passed because conformance runs against the
    # fp32 SOURCE, which has them; nothing ever served an MLA container.
    #
    # q_a/q_b appear only when q_lora_rank > 0 — V2-Lite projects Q directly and
    # uses q_proj above, the full V2 and GLM-5.2 compress it. Both are listed
    # because a suffix that is absent from a checkpoint costs nothing.
    "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight",
    "self_attn.kv_a_layernorm.weight",
    # ── routers ──
    "mlp.gate.weight", "block_sparse_moe.gate.weight",
    # The per-expert selection bias of `noaux_tc` routing. Dropping it does not
    # fail to load — it routes to the wrong experts, quietly, which is strictly
    # worse. DeepSeek-V3, Moonlight and GLM-5.2 all use it.
    "mlp.gate.e_score_correction_bias",
    # ── DSA indexer ──
    #
    # Only `full` layers carry these; `shared` layers borrow the index computed
    # by the preceding full one, so their absence is correct rather than missing.
    # `k_norm.bias` is rank 1 and the only bias in this attention block — the
    # loader binds it through the rank-1 path that norms already use, not
    # bind_weight, which requires rank 2.
    "self_attn.indexer.wk.weight", "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.k_norm.weight", "self_attn.indexer.k_norm.bias",
    # ── dense-layer FFN ──
    #
    # `first_k_dense_replace` layers carry a plain MLP instead of experts. Also
    # absent until now, so the leading layers of every DeepSeek-family model were
    # dropped along with the attention.
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)

# Named exclusions, so "unclaimed" below means genuinely unaccounted for.
#
# Anything matching these is deliberately not copied; everything else that is
# neither dense nor a routed expert stops the conversion. That is what the
# comment above DENSE_SUFFIXES has always claimed and never enforced — the list
# only ever covered the tensors the first three models happened to have, and a
# fourth architecture's weights vanished without a word.
# Non-layer tensors, named once so the completeness check and the copy loop
# below cannot disagree about what counts as claimed.
TOP_LEVEL_TENSORS = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")

IGNORED_PATTERNS = (
    ".experts.",              # routed experts: written to the expert payload
    "shared_experts.",        # copied separately, below
    "rotary_emb.inv_freq",    # derived at load, not a weight
    # NOTE: multi-token-prediction heads are excluded by LAYER INDEX in
    # is_ignored() below, not here. They are ordinary `model.layers.<N>.*` names
    # with N >= num_hidden_layers, so no substring identifies them — a ".mtp"
    # entry sat here matching nothing until GLM-5.2's layer 78 proved it.
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


def expert_reader(get, layer: int, moe_block: str, names: dict):
    """Return `(read(expert, role) -> tensor | None, layout_name)` for one layer.

    Two layouts exist in the wild and this reads both.

    PER-EXPERT: `experts.<i>.{gate,up,down}_proj.weight`, one tensor each.

    FUSED: `experts.gate_up_proj` of shape `(experts, 2*intermediate, hidden)` and
    `experts.down_proj` of shape `(experts, hidden, intermediate)` — every expert
    stacked into one 3-D tensor. Recent `transformers` emits this, and anything
    `make_oracle.py` produces for such a family carries it.

    **The gate/up split is CONTIGUOUS, and that was measured rather than assumed.**
    `modeling_glm_moe_dsa.py` declares the parameter as
    `(num_experts, 2 * intermediate_dim, hidden_dim)` and applies it as
    `linear(x, gate_up_proj[e]).chunk(2, dim=-1)` — contiguous halves of the
    OUTPUT, hence contiguous ROWS of the weight, gate first. Verified against the
    real fixture: reconstructing gate/up from rows `[0:inter]` and `[inter:]`
    reproduces that `chunk` exactly (max diff 0.00e+00), while the interleaved
    reading is off by 1.35. This matters because gpt_oss stores the same
    conceptual thing INTERLEAVED — guessing would have produced a container that
    loads, runs, and is quietly wrong, which is why this was left unimplemented
    until an oracle existed to settle it (roadmap D4).

    Resolved once per layer, not per expert: the fused tensor covers all of them,
    and `get()` copies plus dtype-converts on every call.
    """
    prefix = f"model.layers.{layer}.{moe_block}.experts."

    if get(prefix + "0." + names["gate"]) is not None:
        def read_per_expert(e: int, role: str):
            return get(f"{prefix}{e}.{names[role]}")
        return read_per_expert, "per-expert"

    gate_up = get(prefix + "gate_up_proj")
    down = get(prefix + "down_proj")
    if gate_up is None or down is None:
        return None, None
    if gate_up.ndim != 3 or down.ndim != 3:
        return None, None

    inter = gate_up.shape[1] // 2

    def read_fused(e: int, role: str):
        if role == "down":
            return down[e]
        return gate_up[e][:inter] if role == "gate" else gate_up[e][inter:]

    return read_fused, "fused"


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
    # Shard handles are opened ON DEMAND, with a small cache. They used to be
    # opened all at once:
    #
    #     handles = [safe_open(str(f), framework="pt") for f in shard_files]
    #
    # which is fine for a 5-shard model and SEGFAULTS on a 282-shard one. GLM-5.2
    # has 282 shards; that line mapped all 1.4 TB, indexed 59,585 tensors, read
    # six expert tensors, and died on the seventh. Reading the same tensors with
    # one handle open at a time works indefinitely, which is what identifies the
    # handle count rather than any particular tensor as the cause.
    #
    # Worth stating how it presented, because it nearly passed for success: the
    # crash left a 46 MB partial container behind, and a shell `&&` chain reported
    # exit 0 from the command that followed it. A conversion that dies 0.01% in
    # and looks like it worked is the worst available outcome.
    #
    # The tensor->shard map comes from `model.safetensors.index.json` when the
    # checkpoint ships one, so the common case never opens a shard just to
    # enumerate its keys.
    owner: dict[str, str] = {}
    index_json = src / "model.safetensors.index.json"
    if index_json.is_file():
        try:
            wm = json.loads(index_json.read_text(encoding="utf-8")).get("weight_map", {})
            owner = {name: str(src / shard) for name, shard in wm.items()}
        except Exception:
            owner = {}
    if not owner:
        # No index (single-shard, or a hand-assembled directory): enumerate each
        # shard's keys and close it again immediately.
        for f in shard_files:
            with safe_open(str(f), framework="pt") as h:
                for k in h.keys():
                    owner[k] = str(f)

    # A handful of handles, because consecutive reads are overwhelmingly from the
    # same shard — experts are stored in order. Small enough that the crash above
    # cannot recur, large enough that a tensor family straddling a boundary does
    # not reopen per read.
    open_handles: dict[str, Any] = {}
    HANDLE_CACHE = 4

    def handle_for(path: str):
        h = open_handles.get(path)
        if h is not None:
            return h
        if len(open_handles) >= HANDLE_CACHE:
            victim, vh = next(iter(open_handles.items()))
            del open_handles[victim]
            try:
                vh.__exit__(None, None, None)
            except Exception:
                pass
        h = safe_open(path, framework="pt")
        h.__enter__()
        open_handles[path] = h
        return h

    def get(name: str):
        path = owner.get(name)
        if path is None:
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
        return handle_for(path).get_tensor(name).to(torch.float32).numpy().copy()

    # ── experts ──────────────────────────────────────────────────────────────
    index: list[tuple[int, int, int]] = []
    shard_idx = 0
    shard_off = 0
    total = 0
    uniform_len = -1
    groups: dict[str, int] = {}

    kinds = layer_kinds(cfg, n_layers)
    n_moe = sum(1 for k in kinds if k == "moe")
    print(f"  {src.name}: {n_layers} layers ({n_moe} MoE) x {n_experts} experts, "
          f"gate/up={dt_gate} down={dt_down} group={args.group}")

    # ── every source tensor must be accounted for — CHECKED FIRST ────────────
    #
    # The rule DENSE_SUFFIXES has always claimed. Without it the allow-list
    # silently covered only what the first three models happened to carry: MLA's
    # attention, the dense-layer MLPs and the noaux_tc router bias were all
    # dropped without a word, producing a container that either failed to load
    # or — worse — loaded and routed wrongly.
    #
    # This used to run AFTER the expert payload was written, and GLM-5.2 showed
    # what that costs: 4.5 hours and 439 GiB to discover an unhandled tensor
    # family that is pure NAME arithmetic, knowable before a single byte is read.
    # Nothing here opens a tensor — it compares the shard index's key list against
    # the names this converter will claim — so it belongs before the work, not
    # after it.
    #
    # A REFUSAL, not a warning: a warning inside hours of output is not read.
    #
    # Built once, as an ordered list, and used for BOTH this check and the dense
    # copy loop below. Two transcriptions of one allow-list is how the check ends
    # up asserting something the copy loop does not do. Ordered rather than a set
    # so dense.safetensors is byte-reproducible — mm_fused_experts compares two
    # conversions byte for byte and set iteration order would break it.
    claimed: list[str] = list(TOP_LEVEL_TENSORS)
    for layer in range(n_layers):
        claimed += [f"model.layers.{layer}.{suf}" for suf in DENSE_SUFFIXES]
        claimed += [f"model.layers.{layer}.{moe_block}.shared_experts.{s}.weight"
                    for s in ("gate_proj", "up_proj", "down_proj")]
    claimed_set = set(claimed)

    def is_ignored(name: str) -> bool:
        if any(p in name for p in IGNORED_PATTERNS):
            return True
        # Layers at or beyond num_hidden_layers are MULTI-TOKEN-PREDICTION heads,
        # not part of the served stack. GLM-5.2 declares
        # num_nextn_predict_layers=1 and ships layer 78 with a complete attention
        # block, its own experts, and the MTP-specific eh_proj/enorm/hnorm — 791
        # tensors in all. It is identified by its layer INDEX, which is why a
        # substring rule like ".mtp" matched none of it.
        m = re.match(r"model\.layers\.(\d+)\.", name)
        return m is not None and int(m.group(1)) >= n_layers

    unclaimed = sorted(n for n in owner if n not in claimed_set and not is_ignored(n))
    if unclaimed:
        kind_names = sorted({re.sub(r"^model\.layers\.\d+\.", "", n) for n in unclaimed})
        print(f"  REFUSED  {src.name}: {len(unclaimed)} source tensor(s) match no known role.")
        print("           A dropped tensor produces a model that loads and is wrong, so this")
        print("           stops BEFORE the expert payload rather than after it.")
        print("           Unhandled kinds:")
        for k in kind_names[:12]:
            print(f"             {k}")
        if len(kind_names) > 12:
            print(f"             ... and {len(kind_names) - 12} more")
        print("           Add them to DENSE_SUFFIXES, or to IGNORED_PATTERNS with a reason.")
        return 3

    # Opened only now: a refusal above must leave no output behind, or a failed
    # run looks from the outside like a partial success.
    fh = open(out_dir / f"experts-{shard_idx:05d}.bin", "wb")

    first_moe_layer = next((i for i, k in enumerate(kinds) if k == "moe"), -1)
    for layer in range(n_layers):
        if kinds[layer] == "dense":
            # Zero-length slots keep the (layer, expert) index arithmetic intact.
            index.extend((shard_idx, shard_off, 0) for _ in range(n_experts))
            continue
        read, layout = expert_reader(get, layer, moe_block, names)
        if read is None:
            fh.close()
            fused = fused_expert_diagnosis(get, layer, moe_block)
            print(f"  REFUSED  {fused}" if fused else
                  f"  REFUSED  no expert tensors at model.layers.{layer}.{moe_block}.experts.*")
            return 3
        if layer == first_moe_layer:
            print(f"    expert layout: {layout}")

        for e in range(n_experts):
            base = f"model.layers.{layer}.{moe_block}.experts.{e}."
            blob = bytearray()
            for role, dt in (("gate", dt_gate), ("up", dt_up), ("down", dt_down)):
                t = read(e, role)
                if t is None:
                    fh.close()
                    print(f"  REFUSED  missing {base + names[role]} ({layout} layout)")
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

    # Iterates the SAME list the completeness check was built from, so "claimed"
    # and "copied" cannot drift into disagreement.
    dense: dict[str, "np.ndarray"] = {}
    for name in claimed:
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
