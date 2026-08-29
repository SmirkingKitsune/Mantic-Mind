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

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    # ── block-sparse indexer ──
    #
    # Only `minimax_m3_sparse` layers carry these; the leading dense-attention
    # layers ship none, so their absence is correct rather than missing.
    #
    # SOMA spellings, which for this family are not the checkpoint's -- see
    # SOURCE_DIALECTS. `indexer.k_proj` is ONE head wide against
    # `indexer.q_proj`'s four, which is not a typo: every indexer head scores its
    # own query against the same shared key.
    "self_attn.indexer.q_proj.weight", "self_attn.indexer.k_proj.weight",
    "self_attn.indexer.q_norm.weight", "self_attn.indexer.k_norm.weight",
    # ── dense-layer FFN ──
    #
    # `first_k_dense_replace` layers carry a plain MLP instead of experts. Also
    # absent until now, so the leading layers of every DeepSeek-family model were
    # dropped along with the attention.
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    # ...and the FUSED spelling of the same thing, which recent `transformers`
    # emits for any family whose MLP is built as `Linear(hidden, 2 * inter)`.
    # `[2 * inter, hidden]`, gate rows first -- the rank-2 twin of the rank-3
    # expert layout below, and copied VERBATIM rather than split, because
    # `bind_fused_glu` in the loader reads it directly. Splitting here would mean
    # two writers of one layout that the container round-trip cannot compare.
    "mlp.gate_up_proj.weight",
    # ── Gated DeltaNet linear attention ──
    #
    # A hybrid stack's LINEAR layers, under `linear_attn` rather than
    # `self_attn` — which is why none of the attention suffixes above covers
    # them and why, before this, three quarters of a Qwen3.5 checkpoint's token
    # mixers were unaccounted for.
    #
    # `A_log` and `dt_bias` are bare nn.Parameters and carry no `.weight`
    # suffix; `conv1d.weight` has no bias sibling, because this family's
    # convolution is built `bias=False`. All of them are small, resident, and
    # kept at F32 — the recurrence reads every one of them on every token.
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
    "linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias",
    "linear_attn.norm.weight", "linear_attn.out_proj.weight",
    # ── gated shared expert ──
    #
    # A SIBLING of the shared expert, not part of it: `mlp.shared_expert_gate`
    # against `mlp.shared_expert.*`. One row, and it decides how much of the
    # whole shared branch reaches the residual stream.
    "mlp.shared_expert_gate.weight",
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
    # NOTE: multi-token-prediction heads are excluded in is_ignored() below, not
    # here, and they need TWO rules because the checkpoints spell them two ways.
    # GLM-5.2's is an ordinary `model.layers.<N>.*` name with
    # N >= num_hidden_layers, so no substring identifies it — a ".mtp" entry sat
    # here matching nothing until its layer 78 proved it. Qwen3.5's is a
    # top-level `mtp.` PREFIX, which no layer-index rule can see.
)


# Checkpoints whose tensor names are not the ones `transformers` produces, and
# the exact map that makes them so.
#
# THE ENGINE BINDS ONE DIALECT and this is where a checkpoint joins it. Soma's
# loader knows the names `transformers` emits, because that is what every tiny
# fixture carries and therefore what the whole conformance ladder is graded
# against. MiniMax-M3's 59 production shards carry something else: the text stack
# under a `language_model.` prefix, `block_sparse_moe` where the modeling code
# says `mlp`, a selection bias that hangs off the BLOCK rather than the gate, and
# four indexer tensors spelled `index_q_proj` rather than `indexer.q_proj`.
#
# This DIVERGES from how DeepSeek-V4 was handled, deliberately. There, Soma binds
# the production names and the fixture is normalized to match
# (`native_to_soma.json`), on the grounds that the production shards are what has
# to work. The same reasoning inverts here: V4's production names are the ones
# its own pinned reference implementation uses, so binding them kept ONE
# contract, whereas MiniMax-M3's production names are used by no reference
# implementation at all. Binding them would mean the fixture -- the only thing
# that can prove the engine right -- was reached through a hand-written map that
# nothing validates. So the rewrite happens here, where the completeness check
# below turns a wrong entry into a refusal instead of a dropped tensor.
#
# A SUFFIX MAP, not a substring rewrite, and that distinction is the whole reason
# this table looks verbose. `block_sparse_moe -> mlp` is unambiguous going
# forward and is NOT invertible: in the production checkpoint the MoE layers say
# `block_sparse_moe` while the three dense layers say `mlp`, so a blanket inverse
# turns `mlp.gate_proj.weight` -- a real dense-layer tensor, present under that
# exact name in both dialects -- into a `block_sparse_moe.gate_proj.weight` that
# exists nowhere. The completeness check would then refuse a healthy checkpoint,
# and a slightly luckier version of the same mistake would drop three layers'
# FFN in silence.
SOURCE_DIALECTS = {
    "minimax_m3_vl": {
        # The text stack, lifted out of the multimodal wrapper.
        "prefix": "language_model.",
        # soma suffix -> source suffix, for names below `model.layers.<N>.`.
        # Anything absent is identical in both dialects.
        "suffixes": {
            "mlp.gate.weight": "block_sparse_moe.gate.weight",
            "mlp.gate.e_score_correction_bias":
                "block_sparse_moe.e_score_correction_bias",
            "self_attn.indexer.q_proj.weight": "self_attn.index_q_proj.weight",
            "self_attn.indexer.k_proj.weight": "self_attn.index_k_proj.weight",
            "self_attn.indexer.q_norm.weight": "self_attn.index_q_norm.weight",
            "self_attn.indexer.k_norm.weight": "self_attn.index_k_norm.weight",
        },
        # The half Soma does not serve. Named by module prefix rather than left
        # to a wildcard, so a tower that grows a new module kind lands in
        # `unclaimed` and refuses instead of disappearing.
        "drop": ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp."),
        # SOURCE spellings for the blocks the expert reader walks by hand.
        "moe_block": "block_sparse_moe",
        "shared_block": "block_sparse_moe.shared_experts",
        # Mixtral's expert spelling, on a model that is not Mixtral.
        "experts": {"gate": "w1.weight", "up": "w3.weight", "down": "w2.weight"},
    },
}

# ── blockwise fp8 source checkpoints ─────────────────────────────────────────
#
# DeepSeek-V3 introduced the layout and it is now what a frontier MoE ships
# FIRST: every large Linear weight stored as `F8_E4M3` beside a
# `<tensor>_scale_inv` holding one f32 multiplier per `weight_block_size` tile,
# with the small tensors — norms, routers, embeddings, lm_head — published
# unquantized and named in `modules_to_not_convert`.
#
# GLM-5.3 is the case that forced this. It is the SAME base model as GLM-5.2 —
# byte-identical `config.json` but for `transformers_version` — so the engine,
# the IR and the tokenizer needed nothing at all. What it changed is the upload:
# GLM-5.2's primary is bf16, GLM-5.3's primary is fp8 at 756 GB with the bf16
# twin in a separate 1.5 TB repo. Refusing the primary meant "Soma supports
# GLM-5.3" quietly meant "fetch it twice and keep the larger copy".
#
# This is NOT the general "re-quantize whatever is already packed", which the
# refusal in main() still stops. Blockwise fp8 dequantizes EXACTLY — one
# multiply per tile, no unpacking, no codebook, no shape inference — so what
# reaches `quantize_rows` is the same fp32 matrix the bf16 upload would have
# produced, less the fp8 rounding the publisher had already applied. AWQ, GPTQ
# and compressed-tensors pack sub-byte levels in layouts of their own, and
# quantizing their packed bytes yields a container that loads, streams and
# generates noise.
FP8_SCALE_SUFFIX = "_scale_inv"

# `torch.float8_e4m3fn` is the dtype safetensors reports as `F8_E4M3`; the `fnuz`
# variant is the same width with a different exponent bias and is what ROCm-side
# uploads carry. Named here rather than tested inline so that adding a third
# spelling is one edit, not a grep.
FP8_DTYPE_NAMES = ("float8_e4m3fn", "float8_e4m3fnuz")


def source_fp8_block(cfg: dict) -> tuple[tuple[int, int] | None, str | None]:
    """The blockwise-fp8 tile shape of this checkpoint, or why it is refused.

    Returns `(None, None)` for an ordinary bf16/f32 upload, `((bx, by), None)`
    for one this converter dequantizes, and `(None, reason)` for a packed format
    it must refuse. Never raises: the caller owns the refusal text.
    """
    quant = cfg.get("quantization_config")
    if not isinstance(quant, dict):
        return None, None
    method = str(quant.get("quant_method", "?"))
    # `fmt` is the spelling DeepSeek and GLM upload; `format` is
    # compressed-tensors'. Reading only `format` printed "format ?" about a
    # checkpoint that states `e4m3` plainly, which is the least useful half of a
    # refusal message.
    fmt = str(quant.get("fmt") or quant.get("format") or "?")
    if method != "fp8" or fmt != "e4m3":
        return None, f"{method}, format {fmt}"

    block = quant.get("weight_block_size")
    if not (isinstance(block, list) and len(block) == 2
            and all(isinstance(b, int) and b > 0 for b in block)):
        return None, (f"fp8/e4m3 with weight_block_size {block!r} — per-tensor "
                      f"and per-channel fp8 scales are a different layout, and "
                      f"guessing between them mis-scales every weight")
    # ue8m0 scales are DeepSeek-V4's, and V4 has its own converter. The generic
    # path has never been handed one, so it says so rather than assuming the
    # exponent bias and being wrong by a power of two per tile.
    scale_fmt = quant.get("scale_fmt")
    if scale_fmt not in (None, "f32", "float32", "e4m3"):
        return None, f"fp8/e4m3 with scale_fmt {scale_fmt!r}"
    return (int(block[0]), int(block[1])), None


def dequantize_fp8_block(w, s, block: tuple[int, int], name: str):
    """Blockwise fp8 -> f32: one scale per `block` tile of the last two axes.

    `weight_scale_inv` is the MULTIPLIER — the inverse of the divisor that pushed
    the weight into e4m3 range — so this multiplies. Getting that backwards
    produces a container whose every weight is off by a per-tile factor near 400,
    and it loads, streams and answers nonsense.

    The tile grid is CEILED rather than exact: a row or column count that is not
    a multiple of the tile is covered by one more scale, so the expanded grid is
    CROPPED to the weight rather than reshaped to it. Reshaping would silently
    succeed on the common case where everything divides evenly and raise on the
    one checkpoint that does not.

    Leading axes are carried through untouched, which is what makes a FUSED
    expert stack — `(experts, 2*intermediate, hidden)` — work without a second
    code path.
    """
    import numpy as np

    if w.ndim < 2:
        raise SystemExit(
            f"  REFUSED  {name} is fp8 with a blockwise scale, but it is "
            f"{w.ndim}-D; the layout is defined on the last two axes")
    bx, by = block
    rows, cols = w.shape[-2], w.shape[-1]
    want = w.shape[:-2] + (-(-rows // bx), -(-cols // by))
    s = np.asarray(s, dtype=np.float32)
    if s.shape != want:
        raise SystemExit(
            f"  REFUSED  fp8 scale for {name} is {s.shape}, expected {want} for "
            f"a {'x'.join(str(d) for d in w.shape)} weight in {bx}x{by} tiles")
    s = np.repeat(np.repeat(s, bx, axis=-2), by, axis=-1)
    return w * s[..., :rows, :cols]


_LAYER_RE = re.compile(r"^(model\.layers\.\d+\.)(.*)$")


def dialect_for(model_type: str | None) -> dict:
    """The source dialect for this checkpoint, or the identity."""
    return SOURCE_DIALECTS.get(model_type or "", {})


def to_soma_name(name: str, dialect: dict) -> str:
    """SOURCE tensor name -> the name Soma's loader binds."""
    if not dialect:
        return name
    prefix = dialect.get("prefix", "")
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    m = _LAYER_RE.match(name)
    if m is None:
        return name
    head, suffix = m.groups()
    for soma_suffix, source_suffix in dialect.get("suffixes", {}).items():
        if suffix == source_suffix:
            return head + soma_suffix
    # Prefix rules, for the blocks that carry a variable tail: experts and the
    # shared expert. Matched on the block name alone so `experts.<i>.w1.weight`
    # and `shared_experts.down_proj.weight` both land without enumerating them.
    src_moe = dialect.get("moe_block")
    if src_moe and suffix.startswith(src_moe + "."):
        return head + "mlp." + suffix[len(src_moe) + 1:]
    return head + suffix


def to_source_name(name: str, dialect: dict) -> str:
    """The inverse, for building the list of names to ask the shards for."""
    if not dialect:
        return name
    m = _LAYER_RE.match(name)
    if m is None:
        return dialect.get("prefix", "") + name
    head, suffix = m.groups()
    mapped = dialect.get("suffixes", {}).get(suffix)
    if mapped is None:
        mapped = suffix
    return dialect.get("prefix", "") + head + mapped


def align_up(n: int) -> int:
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def fused_expert_diagnosis(get, layer: int, moe_block: str, src_prefix: str = "") -> str | None:
    """Is this checkpoint using the FUSED expert layout, rather than missing a tensor?

    "missing model.layers.0.mlp.experts.0.gate_proj.weight" sends a reader
    looking for a corrupt download. The truth is usually that the checkpoint
    stacks every expert into one 3-D tensor — a layout recent `transformers`
    emits and this converter does not read — and the converter can SEE that,
    because the fused tensor is sitting right there under a different name.

    Naming the real problem is the whole of this function. Reading the layout is
    deliberately NOT attempted; see the message it produces.
    """
    # SOURCE names. A dialect that nests the text stack (MiniMax-M3 under
    # `language_model.`) makes every one of these miss without it, and the
    # symptom is "no expert tensors" on a checkpoint that has 7296 of them.
    prefix = f"{src_prefix}model.layers.{layer}.{moe_block}.experts."
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


def expert_reader(get, layer: int, moe_block: str, names: dict, src_prefix: str = ""):
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
    # SOURCE names. A dialect that nests the text stack (MiniMax-M3 under
    # `language_model.`) makes every one of these miss without it, and the
    # symptom is "no expert tensors" on a checkpoint that has 7296 of them.
    prefix = f"{src_prefix}model.layers.{layer}.{moe_block}.experts."

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

    # A STATED per-layer list wins over any derivation -- the same rule, in the
    # same order, as resolve_layer_kinds() in the engine. GLM-5.2 ships this and
    # the derivation below happens to agree with it; MiniMax-M3 ships a pattern
    # ("the first three are dense") that no stride expresses at all.
    explicit = cfg.get("mlp_layer_types")
    if isinstance(explicit, list) and len(explicit) >= n_layers:
        return ["moe" if str(t) in ("sparse", "moe", "hash_moe") else "dense"
                for t in explicit[:n_layers]]

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
        # `moe_layer_freq` is TWO things under one key: a scalar STRIDE, or a
        # per-layer 0/1 MASK. Reading only the scalar leaves every layer marked
        # MoE, and the converter then asks a dense layer for experts it does not
        # have -- which is exactly the "no expert tensors at
        # model.layers.0.mlp.experts.*" this branch exists to prevent.
        elif isinstance(freq, list) and i < len(freq) and not freq[i]:
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
    ap.add_argument("--source-revision", default=None,
                    help="expected immutable Hub revision (V4 is pinned)")
    ap.add_argument("--validate-only", action="store_true",
                    help="check config/tensor coverage without reading payloads")
    ap.add_argument("--no-resume", action="store_true",
                    help="refuse reuse of a compatible conversion manifest")
    ap.add_argument("--include-dspark", action="store_true",
                    help="augment DeepSeek-V4 with its three-stage DSpark draft model")
    ap.add_argument("--test-fixture", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv[1:])

    import numpy as np
    import torch
    from safetensors import safe_open

    src = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    model_type = cfg.get("model_type")
    if model_type == "deepseek_v4":
        import convert_deepseek_v4
        return convert_deepseek_v4.run(args)

    # ── refuse for the REAL reason, not a downstream symptom ─────────────────
    #
    # A multimodal checkpoint nests the language model under `text_config`, so
    # every key read below is absent at the top level. Left alone, this function
    # finds n_layers 0 and n_experts 0 and prints "no routed experts; nothing to
    # stream" — about a model with 896 of them. The operator then goes looking
    # for a routing problem that does not exist.
    #
    # Refusing here is also cheap in the way that matters: these checkpoints run
    # to a terabyte and more, and the alternative is discovering the problem
    # after hours of reading.
    dialect = dialect_for(model_type)
    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict) and dialect:
        # A wrapper this converter KNOWS how to take the text stack out of.
        #
        # The refusal below is still right for every other multimodal
        # checkpoint, and the difference is not that this one is special -- it is
        # that SOURCE_DIALECTS states exactly which tensors belong to the tower,
        # so dropping them is a decision on the record rather than an accident.
        #
        # The vision half is still DECLARED: `config.json` is copied into the
        # container verbatim below, so the IR reports `vision+text` and the plan
        # says which half it is serving. A container that quietly claimed to be a
        # text model would be a model answering about images it never received.
        print(f"  text-only: {src.name} is a multimodal wrapper (model_type "
              f"{model_type!r}); converting the language model under "
              f"'text_config' and dropping the vision tower.")
        cfg.update(text_cfg)
    elif isinstance(text_cfg, dict):
        inner = text_cfg.get("model_type", "?")
        print(f"  REFUSED  {src.name}: multimodal wrapper (model_type "
              f"{model_type!r}, language model nested under 'text_config' as "
              f"{inner!r}).")
        print("           Soma converts a text stack only, and a vision tower "
              "dropped silently would")
        print("           serve a model answering about images it never "
              "received. Run `soma plan` for")
        print("           the architecture verdict before converting.")
        return 3

    # compressed-tensors, AWQ and GPTQ ship weights already packed at 4 bits with
    # their own scale layout. Every reader below assumes f32-widenable source
    # tensors and would quantize the PACKED BYTES as if they were weights —
    # producing a container that loads, streams, and generates noise.
    #
    # Blockwise fp8 is the one exception, and it is an exception on the merits
    # rather than as a convenience — see FP8_SCALE_SUFFIX above.
    fp8_block, quant_refusal = source_fp8_block(cfg)
    if quant_refusal is not None:
        print(f"  REFUSED  {src.name}: checkpoint is already quantized "
              f"({quant_refusal}).")
        print("           convert.py quantizes FROM f32, bf16, or blockwise fp8; "
              "re-quantizing")
        print("           packed bytes yields a container that loads and "
              "generates noise. Convert")
        print("           from an unquantized upload.")
        return 3
    if fp8_block is not None:
        print(f"  fp8 source: {src.name} is blockwise fp8, "
              f"{fp8_block[0]}x{fp8_block[1]} tiles; every fp8 tensor "
              f"dequantizes to f32 on read.")

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

    # SOURCE spellings, which is what `get()` is asked for. `to_soma_name`
    # rewrites them on the way into the container.
    src_prefix = dialect.get("prefix", "")
    moe_block = dialect.get(
        "moe_block", "block_sparse_moe" if model_type == "mixtral" else "mlp")
    # SINGULAR for the Qwen families, plural everywhere else. Mirrors
    # `TensorNaming::shared_block` in src/soma/arch_ir.cpp — one character, and
    # getting it wrong drops the shared expert from the container without a word
    # because the copy loop simply finds no tensor of that name.
    shared_block = dialect.get(
        "shared_block",
        "mlp.shared_expert"
        if model_type in ("qwen3_5_moe_text", "qwen3_5_moe", "qwen2_moe")
        else f"{moe_block}.shared_experts")
    names = dialect.get(
        "experts",
        {"gate": "w1.weight", "up": "w3.weight", "down": "w2.weight"}
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
        raw = handle_for(path).get_tensor(name)
        is_fp8 = str(raw.dtype).rsplit(".", 1)[-1] in FP8_DTYPE_NAMES
        w = raw.to(torch.float32).numpy().copy()
        if not is_fp8:
            # Includes every tensor an fp8 upload publishes unquantized — the
            # norms, the router, embed_tokens, lm_head. Driven by the tensor's
            # own DTYPE rather than by matching `modules_to_not_convert` by name:
            # the file already states which tensors are fp8, and a second copy of
            # that fact is a second thing that can be stale.
            return w
        scale_path = owner.get(name + FP8_SCALE_SUFFIX)
        if scale_path is None:
            # Loud, because the quiet version of this is the failure this
            # codebase keeps paying for: an fp8 weight read without its scale is
            # a real matrix roughly 400x too small, and it converts, loads,
            # streams and answers nonsense.
            raise SystemExit(
                f"  REFUSED  {name} is stored as fp8 and no "
                f"{name + FP8_SCALE_SUFFIX} accompanies it. Reading it unscaled "
                f"would produce a container that loads and is wrong.")
        # .copy() for the same reason as above — an already-f32 scale makes
        # .to(float32) a no-op and .numpy() a view into mmapped storage that the
        # handle cache is free to close.
        s = handle_for(scale_path).get_tensor(
            name + FP8_SCALE_SUFFIX).to(torch.float32).numpy().copy()
        return dequantize_fp8_block(w, s, fp8_block, name)

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
    # SOURCE names throughout -- these are what `get()` is asked for and what the
    # completeness check compares against the shard index. Container names come
    # from `to_soma_name` at copy time.
    #
    # DENSE_SUFFIXES holds SOMA spellings, so the dialect is inverted here rather
    # than the list being duplicated per dialect. `shared_block` is already a
    # source spelling and is joined AFTER the inversion for that reason.
    claimed: list[str] = [to_source_name(n, dialect) for n in TOP_LEVEL_TENSORS]
    for layer in range(n_layers):
        claimed += [to_source_name(f"model.layers.{layer}.{suf}", dialect)
                    for suf in DENSE_SUFFIXES]
        claimed += [src_prefix +
                    f"model.layers.{layer}.{shared_block}.{s}.weight"
                    for s in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")]
    claimed_set = set(claimed)

    def is_ignored(name: str) -> bool:
        if any(p in name for p in IGNORED_PATTERNS):
            return True
        # A blockwise-fp8 scale is not a tensor this converter DROPS — it is
        # consumed, by dequantizing the weight it belongs to. So it is claimed
        # exactly when that weight is, which is stricter than an IGNORED_PATTERNS
        # entry would be: a scale whose weight is unaccounted for still refuses,
        # rather than vanishing alongside the tensor it was supposed to scale.
        if fp8_block is not None and name.endswith(FP8_SCALE_SUFFIX):
            weight = name[: -len(FP8_SCALE_SUFFIX)]
            return weight in claimed_set or is_ignored(weight)
        # The vision half of a multimodal wrapper, by module prefix.
        if any(name.startswith(d) for d in dialect.get("drop", ())):
            return True
        # The rules below are written against SOMA names, so a dialect that
        # prefixes its stack has that prefix removed first. Without this the
        # layer-index regex matches nothing and a wrapper's MTP layers land in
        # `unclaimed` rather than being ignored.
        name = to_soma_name(name, dialect)
        # Layers at or beyond num_hidden_layers are MULTI-TOKEN-PREDICTION heads,
        # not part of the served stack. GLM-5.2 declares
        # num_nextn_predict_layers=1 and ships layer 78 with a complete attention
        # block, its own experts, and the MTP-specific eh_proj/enorm/hnorm — 791
        # tensors in all. It is identified by its layer INDEX, which is why a
        # substring rule like ".mtp" matched none of it.
        # …and the OTHER spelling. Qwen3.5 hangs its MTP head off a TOP-LEVEL
        # `mtp.` prefix — `mtp.fc`, `mtp.layers.0.*`, `mtp.norm`,
        # `mtp.pre_fc_norm_*` — which no `model.layers.<N>` rule can see. The
        # comment above says a substring matched none of GLM's; here a prefix
        # matches all of Qwen3.5's, and the two conventions need one rule each.
        #
        # Excluded rather than converted because nothing can serve it: the IR
        # records the head as `source_declared` with `present` false, so the plan
        # names it and the engine never claims to run it.
        if name.startswith("mtp."):
            return True
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

    # ── the tokenizer, compiled INTO the container ────────────────────────────
    #
    # A container without one is not servable as text. `soma serve` falls back to
    # one-token-per-byte, which produces real tokens from real weights and
    # meaningless output; `conform`'s tokenizer_roundtrip stage reports skipped.
    # GLM-5.2 was converted, verified, planned and served before anyone noticed,
    # because every one of those steps works on token IDS.
    #
    # Compiled HERE rather than left to the operator so a container is
    # self-sufficient by construction — the alternative is remembering three files
    # by hand, which is exactly what was done once and got two of the three.
    #
    # NON-FATAL, deliberately. Some families' pretokenizers are not compiled yet
    # (Mixtral's SentencePiece and granite's legacy vocab.json), and aborting a
    # multi-hour conversion over a tokenizer would be a
    # disproportionate response to a gap the container can be used without. The
    # outcome is recorded in container_meta.json and repeated in the final summary
    # line, because an early message is invisible after four hours of layer output.
    #
    # Run BEFORE the expert loop for the same reason the completeness check is:
    # it costs seconds and it is better known now than at the end.
    tokenizer_status = "skipped"
    tokenizer_outputs = tuple(out_dir / name for name in (
        "tokenizer.soma", "tokenizer_oracle.bin", "tokenizer_meta.json"))
    tokenizer_unsupported = out_dir / "tokenizer.unsupported"

    # Conversion directories are variant-stable and may already exist after a
    # refused or interrupted attempt. Never let that attempt's tokenizer make a
    # new conversion look text-capable. `compile_tokenizer` also writes
    # tokenizer.soma before it asks HF tokenizers for the oracle, so an exception
    # after that point would otherwise leave a plausible, incomplete artifact.
    for path in (*tokenizer_outputs, tokenizer_unsupported):
        path.unlink(missing_ok=True)
    try:
        import compile_tokenizer

        rc = compile_tokenizer.main(["compile_tokenizer", str(src), "--out", str(out_dir)])
        if rc == 0:
            missing = [path.name for path in tokenizer_outputs if not path.is_file()]
            if missing:
                raise RuntimeError(
                    "tokenizer compiler returned success without " + ", ".join(missing))
            tokenizer_unsupported.unlink(missing_ok=True)
            tokenizer_status = "compiled"
        else:
            tokenizer_status = "unsupported"
            for path in tokenizer_outputs:
                path.unlink(missing_ok=True)
            if not tokenizer_unsupported.exists():
                tokenizer_unsupported.write_text(
                    f"tokenizer compiler returned {rc}\n", encoding="utf-8")
    except Exception as e:  # a missing dep must not cost the conversion
        tokenizer_status = "unsupported"
        for path in tokenizer_outputs:
            path.unlink(missing_ok=True)
        if not tokenizer_unsupported.exists():
            tokenizer_unsupported.write_text(
                f"{type(e).__name__}: {e}\n", encoding="utf-8")
        print(f"  tokenizer: not compiled ({type(e).__name__}: {e})")

    # Opened only now: a refusal above must leave no output behind, or a failed
    # run looks from the outside like a partial success.
    fh = open(out_dir / f"experts-{shard_idx:05d}.bin", "wb")

    first_moe_layer = next((i for i, k in enumerate(kinds) if k == "moe"), -1)
    for layer in range(n_layers):
        if kinds[layer] == "dense":
            # Zero-length slots keep the (layer, expert) index arithmetic intact.
            index.extend((shard_idx, shard_off, 0) for _ in range(n_experts))
            continue
        read, layout = expert_reader(get, layer, moe_block, names, src_prefix)
        if read is None:
            fh.close()
            fused = fused_expert_diagnosis(get, layer, moe_block, src_prefix)
            print(f"  REFUSED  {fused}" if fused else
                  f"  REFUSED  no expert tensors at "
                  f"{src_prefix}model.layers.{layer}.{moe_block}.experts.*")
            return 3
        if layer == first_moe_layer:
            print(f"    expert layout: {layout}")

        for e in range(n_experts):
            base = f"{src_prefix}model.layers.{layer}.{moe_block}.experts.{e}."
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
            dense[to_soma_name(name, dialect)] = t

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
    # VERBATIM, including `vision_config` for a wrapper. The IR then reports
    # `vision+text` and the plan states which half is being served -- which is
    # the whole job of ModalitySpec, and the opposite of what writing a
    # text-only config here would achieve.
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
        # What the SOURCE was, not what this container is. A q4_g container built
        # from a bf16 upload and one built from that upload's fp8 twin are not
        # the same artifact — only the first can be compared against bf16 weights
        # at all — and nothing downstream could tell them apart, because the
        # container records the codec it WROTE and never the one it read.
        "source_quantization": (
            f"fp8-e4m3-block-{fp8_block[0]}x{fp8_block[1]}"
            if fp8_block is not None else "none"),
        # Whether this container can be served as TEXT. Recorded rather than left
        # to be inferred from which files happen to exist, so a consumer can tell
        # "no tokenizer was possible for this family" from "someone deleted one".
        "tokenizer": tokenizer_status,
    }
    (out_dir / "container_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    padded = sum(align_up(l) for _, _, l in index)
    print(f"  OK       {len(index)} experts, {total / 1e9:.3f} GB payload, "
          f"{(padded - total) / 1e6:.1f} MB padding ({100.0 * (padded - total) / max(total, 1):.3f}%), "
          f"{shard_idx + 1} shard(s), dense {len(dense)} tensors, "
          f"tokenizer {tokenizer_status}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
