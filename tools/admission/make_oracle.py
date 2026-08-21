#!/usr/bin/env python3
"""Build a tiny-random conformance fixture and its `transformers` oracle.

The G0/G1 gates are token-exact comparisons against this oracle. That only works
if the tiny model has the **real** architecture — a generic small transformer
would pass while the actual model failed.

So the rule here is exact and it is the whole design:

    SHRINK DIMENSIONS. PRESERVE SEMANTICS.

Dimensional (safe to shrink): layer count, hidden size, intermediate sizes,
expert count, vocab size, head count, latent ranks.

Semantic (preserved verbatim): attention family, GQA head ratio, whether head_dim
is independent of hidden_size, qk-norm, RoPE theta/scaling/partial dims, router
scoring function, bias correction, group-limited routing, top-k normalization,
routed scaling factor, shared-expert presence, which layers are dense vs MoE,
activation, norm placement and epsilon, and every architecture-specific
multiplier.

A tiny-random model is either exactly right or obviously wrong. A real checkpoint
can be *approximately* right in ways that hide a bug for weeks — which is why
this exists and why it runs per-commit.

Usage:
    make_oracle.py <model_dir_or_repo> [--out DIR] [--seed N]
                   [--layers N] [--experts N] [--positions N] [--generate N]

Example:
    make_oracle.py Z:/.../models/allenai/OLMoE-1B-7B-0924 \\
        --out tests/fixtures/tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

# ── Dimension targets ─────────────────────────────────────────────────────────
# Small enough to commit and to run per-commit; large enough that a broken
# reduction, a bad mask, or a mis-indexed expert actually shows up.

TARGET_LAYERS = 4  # >= 2 so a dense/MoE split is representable
TARGET_HEAD_DIM = 16
TARGET_KV_HEADS = 2  # for GQA; MHA keeps heads == kv_heads
TARGET_EXPERTS = 16
TARGET_TOPK = 4
TARGET_MOE_INTERMEDIATE = 32
TARGET_DENSE_INTERMEDIATE = 128
TARGET_SHARED_INTERMEDIATE = 64
TARGET_VOCAB = 512
TARGET_MAX_POS = 2048

# Config keys naming the same concept across families. Upstream is inconsistent;
# resolving that here means nothing downstream has to care.
EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
TOPK_KEYS = ("num_experts_per_tok",)
MOE_INTERMEDIATE_KEYS = ("moe_intermediate_size",)
SHARED_INTERMEDIATE_KEYS = ("shared_expert_intermediate_size",)

# MLA latent dims. Shrunk coherently or the projections do not compose.
MLA_KEYS = {
    "kv_lora_rank": 32,
    "q_lora_rank": 48,  # only when not None — None means "no Q down-projection"
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
}


def _first_present(d: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in d:
            return k
    return None


def _shrink(cfg: dict[str, Any], key: str, target: int) -> int | None:
    """Set cfg[key] = min(current, target). NEVER grows a dimension.

    Growing is not a harmless overshoot. Mixtral has 8 experts; raising it to 16
    would have turned a 25%-active coarse-grained model into a 12.5%-active one
    — silently converting the deliberate `resident-only` negative fixture into
    something that looks streamable, which is the exact judgement the verdict
    function is supposed to get right.
    """
    cur = cfg.get(key)
    if not isinstance(cur, int) or cur <= 0:
        return None
    cfg[key] = min(cur, target)
    return cfg[key]


def shrink_attention(cfg: dict[str, Any]) -> None:
    """Shrink attention while preserving the GQA ratio and head_dim independence.

    Two properties must survive, because both are load-bearing for the backend:

    1. The heads:kv_heads RATIO. A tiny model that collapsed GQA to MHA would
       never exercise repeat_kv, and the GQA backend would pass G0 with that path
       dead.
    2. Whether head_dim is INDEPENDENT of hidden_size. Qwen3-MoE sets head_dim
       128 with hidden_size 2048 and 32 heads (2048/32 = 64 != 128). A fixture
       that derived head_dim from hidden_size would silently drop that.
    """
    heads = int(cfg.get("num_attention_heads", 8))
    kv = int(cfg.get("num_key_value_heads", heads))
    ratio = max(1, heads // max(1, kv))

    if ratio == 1:
        # MHA. Keep it MHA — it is the degenerate case of the GQA backend and
        # deserves its own fixture rather than being quietly upgraded.
        new_kv = 4
        new_heads = 4
    else:
        new_kv = TARGET_KV_HEADS
        new_heads = new_kv * ratio

    cfg["num_attention_heads"] = new_heads
    cfg["num_key_value_heads"] = new_kv

    if "head_dim" in cfg and cfg["head_dim"] is not None:
        # Independent of hidden_size upstream; keep it independent here.
        cfg["head_dim"] = TARGET_HEAD_DIM
        cfg["hidden_size"] = new_heads * TARGET_HEAD_DIM // max(1, ratio) or 64
        # hidden_size need not equal heads*head_dim when head_dim is explicit;
        # o_proj maps heads*head_dim -> hidden_size. Pick a small, clean value.
        cfg["hidden_size"] = 64
    else:
        # head_dim is implied by hidden_size / heads, so hidden_size must stay
        # divisible by the new head count.
        cfg["hidden_size"] = new_heads * TARGET_HEAD_DIM


def shrink_deepseek_v4(cfg: dict[str, Any], layers: int) -> None:
    """Shrink V4 without erasing any of its distinguishing mechanisms.

    V4 is MQA with an intentionally enormous production head ratio, so the
    generic GQA shrinker would preserve 128:1 by creating 256 tiny query heads.
    That is dimensional, not semantic, and makes the fixture larger than the
    models it is meant to replace. Keep shared-KV MQA while shrinking the query
    fanout, then preserve HCA+CSA, the sparse indexer, grouped Q/O low rank,
    four HC streams, and the three-layer hash bootstrap explicitly.
    """
    cfg["num_attention_heads"] = 4
    cfg["num_key_value_heads"] = 1
    cfg["head_dim"] = 16
    cfg["hidden_size"] = 64
    cfg["q_lora_rank"] = 32
    cfg["qk_rope_head_dim"] = 8
    cfg["partial_rotary_factor"] = 0.5
    cfg["o_groups"] = 2
    cfg["o_lora_rank"] = 16
    cfg["index_n_heads"] = 16
    cfg["index_head_dim"] = 16
    cfg["index_topk"] = 16
    cfg["sliding_window"] = 64
    cfg["compress_ratios"] = [128, 128, 4, 128][:layers]
    if layers > 4:
        cfg["compress_ratios"] += [4 if i % 2 == 0 else 128 for i in range(4, layers)]
    cfg["num_hash_layers"] = min(3, layers)
    cfg["hc_mult"] = 4
    cfg["max_position_embeddings"] = TARGET_MAX_POS
    if isinstance(cfg.get("rope_scaling"), dict):
        factor = float(cfg["rope_scaling"].get("factor", 1.0) or 1.0)
        cfg["rope_scaling"]["original_max_position_embeddings"] = max(
            1, int(TARGET_MAX_POS / factor))
    # DSpark is deliberately outside this base-model milestone. Keeping its
    # production noise id after shrinking the vocabulary creates a misleading
    # native-config warning even though no MTP module is instantiated.
    cfg["dspark_noise_token_id"] = None
    cfg["dspark_target_layer_ids"] = []
    # Native Transformers executes an fp32 model made with `from_config` as an
    # ordinary dense model: the quantization_config is only consumed by the
    # `from_pretrained` quantizer.  Record that fact explicitly for Soma's V4
    # semantic low-precision switches.  Production configs do not carry these
    # fixture overrides and therefore keep both source FP8/FP4 operations on.
    cfg["semantic_fp8_quant_dequant"] = False
    cfg["semantic_fp4_quant_dequant"] = False


def shrink_mla(cfg: dict[str, Any]) -> None:
    """Shrink MLA latent dims coherently. No-op for non-MLA families."""
    if "kv_lora_rank" not in cfg:
        return
    for key, target in MLA_KEYS.items():
        if key not in cfg:
            continue
        if cfg[key] is None:
            continue  # q_lora_rank None means no Q down-projection: preserve
        cfg[key] = target

    # MLA derives head_dim from qk_nope + qk_rope, so an explicit head_dim would
    # conflict. num_attention_heads was already reduced by shrink_attention;
    # hidden_size must stay compatible with the v_head_dim output projection.
    heads = int(cfg.get("num_attention_heads", 4))
    cfg["hidden_size"] = max(64, heads * MLA_KEYS["v_head_dim"] // 2)
    cfg["hidden_size"] = 64
    cfg.pop("head_dim", None)

    # `qk_head_dim` is DERIVED — it is nope ++ rope — and some families state it
    # explicitly rather than leaving it implied. GLM-5.2 does. Left at its
    # original value it disagrees with the shrunken halves and the forward dies
    # inside torch.split: "expects split_sizes to sum exactly to 256 ... but got
    # [16, 8]". Recomputed rather than popped, because a family that states it
    # presumably reads it.
    if "qk_head_dim" in cfg:
        cfg["qk_head_dim"] = MLA_KEYS["qk_nope_head_dim"] + MLA_KEYS["qk_rope_head_dim"]


def shrink_dsa(cfg: dict[str, Any]) -> None:
    """Shrink DeepSeek Sparse Attention. No-op for families without an indexer.

    `index_topk` is the one that matters, and for the same reason
    `sliding_window` is clamped below: with fewer tokens than `index_topk`, the
    top-k selects EVERYTHING and the sparse path becomes bit-identical to dense
    attention. A fixture built without shrinking it would pass whether or not the
    indexer works at all.

    That is measured rather than assumed. On a shrunk GLM-5.2 at `index_topk=8`
    against a 40-token prompt, positions 0-7 match a dense run to 0.0 while
    positions 20-39 differ by 0.48 max|logit|. 64 against the default 512
    evaluated positions leaves seven eighths of the sequence exercising real
    selection.

    IndexShare is the other half: `indexer_types` names which layers own an
    indexer (`full`) and which borrow the previous one's (`shared`). It is
    TRUNCATED, never regenerated — the pattern is the semantics, and a shrink
    that preserved only the list's length would silently change which layers
    carry indexer weights, which is the single property this fixture exists to
    exercise.
    """
    if "index_topk" not in cfg and "indexer_types" not in cfg:
        return

    # `index_n_heads` is deliberately NOT shrunk, and that is a correctness
    # requirement rather than a size preference.
    #
    # An index score is `sum_h w[h] * relu(q[h].k)`, so it is EXACTLY 0.0 only
    # when ReLU zeroes every head at once. The probability of that is ~2^-H, and
    # ties at exactly the top-k cut are then resolved by whatever `torch.topk`
    # does internally — which is neither ascending nor descending index and is not
    # a property of the architecture at all.
    #
    # Measured on this fixture, scores that are exactly 0.0:
    #
    #     1 head  50.69%      3 heads 13.52%
    #     2 heads 27.12%      4 heads  6.99%
    #
    # a clean halving per head, extrapolating to ~2e-8% at GLM-5.2's real 32.
    # Shrinking to 4 therefore MANUFACTURED a tie in 47 of 768 selective queries.
    # Soma reproduced the reference's selection on 721 of 721 queries where the
    # cut was untied and on 7 of 47 where it was tied, which is exactly the
    # signature of a correct implementation being graded on a coin flip: the
    # fixture made token-exactness untestable at the only positions that exercise
    # sparse selection.
    #
    # Keeping 32 heads costs a [1024, 48] wq_b — about 49k parameters, or 6% of
    # the fixture. `index_head_dim` still shrinks, because head WIDTH does not
    # affect the sign statistics that produce ties.
    _shrink(cfg, "index_head_dim", 32)
    _shrink(cfg, "index_topk", 64)

    n = int(cfg["num_hidden_layers"])
    for key in ("indexer_types", "mlp_layer_types"):
        val = cfg.get(key)
        if isinstance(val, list) and len(val) > n:
            cfg[key] = val[:n]

    # A shrunk `indexer_types` must still contain at least one `shared` layer, or
    # IndexShare is not represented and the fixture cannot fail on it.
    types = cfg.get("indexer_types")
    if isinstance(types, list) and types and "shared" not in types:
        raise SystemExit(
            f"  REFUSED: indexer_types[:{n}] = {types} contains no 'shared' layer, so this "
            f"fixture cannot exercise IndexShare at all. Raise --layers past the first "
            f"'shared' entry."
        )


def shrink_moe(cfg: dict[str, Any]) -> None:
    """Shrink expert counts and widths, preserving every routing semantic."""
    ekey = _first_present(cfg, EXPERT_COUNT_KEYS)
    n_experts = _shrink(cfg, ekey, TARGET_EXPERTS) if ekey else None

    tkey = _first_present(cfg, TOPK_KEYS)
    if tkey is not None and isinstance(cfg.get(tkey), int):
        # Preserve 1 < top_k < n_experts. Both bounds matter: top_k == 1 skips
        # weight normalization entirely, top_k == n_experts skips selection.
        # Clamped against the SHRUNKEN expert count, not the original.
        upper = (n_experts - 1) if n_experts else TARGET_TOPK
        cfg[tkey] = max(2, min(TARGET_TOPK, int(cfg[tkey]), upper))

    mkey = _first_present(cfg, MOE_INTERMEDIATE_KEYS)
    if mkey is not None:
        _shrink(cfg, mkey, TARGET_MOE_INTERMEDIATE)

    skey = _first_present(cfg, SHARED_INTERMEDIATE_KEYS)
    if skey is not None:
        _shrink(cfg, skey, TARGET_SHARED_INTERMEDIATE)

    if "intermediate_size" in cfg:
        # For Mixtral and OLMoE this IS the expert width (no moe_intermediate_size).
        _shrink(cfg, "intermediate_size",
                TARGET_MOE_INTERMEDIATE if mkey is None else TARGET_DENSE_INTERMEDIATE)

    # Group-limited routing: n_group must divide n_experts and topk_group <= n_group.
    if "n_group" in cfg and cfg["n_group"]:
        n_group = int(cfg["n_group"])
        if TARGET_EXPERTS % n_group != 0:
            n_group = 1
        cfg["n_group"] = n_group
        cfg["topk_group"] = min(int(cfg.get("topk_group", 1) or 1), n_group)


def shrink_config(raw: dict[str, Any], layers: int, experts: int) -> dict[str, Any]:
    """Produce the tiny config. Everything not touched here is preserved."""
    cfg = json.loads(json.dumps(raw))  # deep copy

    global TARGET_EXPERTS
    TARGET_EXPERTS = experts

    cfg["num_hidden_layers"] = layers

    # Fixtures are initialized, executed and saved as fp32 below. Transformers
    # 5.x treats `dtype` as authoritative when reloading, so leaving a source
    # checkpoint's bf16 declaration here casts the saved fp32 tensors back to
    # bf16 and makes a clean reload disagree with the oracle that produced them.
    # Keep both spellings aligned because older families still serialize the
    # deprecated `torch_dtype` key while newer ones use `dtype`.
    cfg["dtype"] = "float32"
    cfg["torch_dtype"] = "float32"

    if cfg.get("model_type") == "deepseek_v4":
        shrink_deepseek_v4(cfg, layers)
    else:
        shrink_attention(cfg)
        shrink_mla(cfg)
        shrink_dsa(cfg)  # after shrink_mla: DSA is MLA plus an indexer
    shrink_moe(cfg)

    _shrink(cfg, "vocab_size", TARGET_VOCAB)
    _shrink(cfg, "max_position_embeddings", TARGET_MAX_POS)

    # `auto_map` points at the family's modeling_*.py by Python class path. Those
    # files live next to the ORIGINAL config and are not vendored into the
    # fixture — the model object is built from the source directory instead
    # (see main()). Carrying a dangling auto_map would make the fixture look
    # loadable and then fail on a missing module.
    #
    # It is also meaningless to the IR, which records an `adapter` name rather
    # than a Python import path.
    cfg.pop("auto_map", None)

    # Special-token ids must land inside the shrunken vocab. Remapped rather than
    # dropped: a None eos changes generation semantics, which is exactly the kind
    # of silent difference this fixture exists to catch.
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        val = cfg.get(key)
        if isinstance(val, int):
            cfg[key] = val % TARGET_VOCAB
        elif isinstance(val, list):
            cfg[key] = [v % TARGET_VOCAB for v in val]

    # Sliding window, if present, must be shorter than the eval sequence or the
    # windowed path never fires and the fixture silently tests full attention.
    if cfg.get("sliding_window"):
        cfg["sliding_window"] = 64
        cfg["max_window_layers"] = min(int(cfg.get("max_window_layers", layers)), layers)

    # first_k_dense_replace must leave at least one MoE layer.
    if "first_k_dense_replace" in cfg:
        cfg["first_k_dense_replace"] = min(int(cfg["first_k_dense_replace"]), layers - 1)

    cfg["torch_dtype"] = "float32"  # G0 is the fp32 path
    cfg["use_cache"] = True
    return cfg


def layer_kinds(cfg: dict[str, Any]) -> list[str]:
    """Materialize which layers are dense vs MoE — the IR's `layer_kinds`.

    Upstream expresses this three different ways (first_k_dense_replace,
    decoder_sparse_step + mlp_only_layers, moe_layer_freq). Resolving it here and
    writing it into the fixture means the C++ side can be checked against a
    concrete answer instead of re-deriving the same tangle.
    """
    n = int(cfg["num_hidden_layers"])
    kinds = ["moe"] * n

    # `mlp_layer_types` states it OUTRIGHT, so believe it rather than re-deriving.
    #
    # GLM-5.2 ships this list, and the derivation below happens to agree with it
    # — first_k_dense_replace=3 with moe_layer_freq=1 gives the same answer. That
    # coincidence is exactly why an explicit list should win: a family with an
    # irregular pattern would be silently mis-derived, and there would be nothing
    # to notice it.
    explicit = cfg.get("mlp_layer_types")
    if isinstance(explicit, list) and len(explicit) >= n:
        return ["moe" if str(t) in {"sparse", "moe", "hash_moe"} else "dense"
                for t in explicit[:n]]

    first_dense = int(cfg.get("first_k_dense_replace", 0) or 0)
    for i in range(min(first_dense, n)):
        kinds[i] = "dense"

    step = int(cfg.get("decoder_sparse_step", 1) or 1)
    mlp_only = set(cfg.get("mlp_only_layers", []) or [])
    freq = cfg.get("moe_layer_freq", 1)

    for i in range(n):
        if kinds[i] == "dense":
            continue
        if i in mlp_only:
            kinds[i] = "dense"
        elif step > 1 and (i % step) != 0:
            kinds[i] = "dense"
        elif isinstance(freq, int) and freq > 1 and (i % freq) != 0:
            kinds[i] = "dense"
        elif isinstance(freq, list) and i < len(freq) and not freq[i]:
            kinds[i] = "dense"

    return kinds


def canonicalize_deepseek_v4_state(model: Any,
                                   name_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Translate native Transformers V4 names to the release/runtime dialect.

    DeepSeek's reference checkpoint uses the short inference names (``wq_a``,
    ``hc_attn_fn``, ``ffn``), while Transformers exposes descriptive module names
    (``q_a_proj``, ``attn_hc.fn``, ``mlp``).  Soma intentionally binds the former
    because that is what the 66 production shards contain.  Tiny fixtures are
    normalized once at generation time so production execution has one naming
    contract instead of a V4-specific alias tree in its hot loader.
    """
    state = model.state_dict()
    out: dict[str, Any] = {}

    exact = {
        "model.hc_head.hc_fn": "model.hc_head_fn",
        "model.hc_head.hc_base": "model.hc_head_base",
        "model.hc_head.hc_scale": "model.hc_head_scale",
    }
    suffixes = {
        "attn_hc.fn": "hc_attn_fn",
        "attn_hc.base": "hc_attn_base",
        "attn_hc.scale": "hc_attn_scale",
        "ffn_hc.fn": "hc_ffn_fn",
        "ffn_hc.base": "hc_ffn_base",
        "ffn_hc.scale": "hc_ffn_scale",
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "self_attn.sinks": "self_attn.attn_sink",
        "self_attn.q_a_proj.weight": "self_attn.wq_a.weight",
        "self_attn.q_a_norm.weight": "self_attn.q_norm.weight",
        "self_attn.q_b_proj.weight": "self_attn.wq_b.weight",
        "self_attn.kv_proj.weight": "self_attn.wkv.weight",
        "self_attn.kv_norm.weight": "self_attn.kv_norm.weight",
        "self_attn.o_a_proj.weight": "self_attn.wo_a.weight",
        "self_attn.o_b_proj.weight": "self_attn.wo_b.weight",
        "self_attn.compressor.position_bias": "self_attn.compressor.ape",
        "self_attn.compressor.kv_proj.weight": "self_attn.compressor.wkv.weight",
        "self_attn.compressor.gate_proj.weight": "self_attn.compressor.wgate.weight",
        "self_attn.compressor.kv_norm.weight": "self_attn.compressor.norm.weight",
        "self_attn.compressor.indexer.q_b_proj.weight": "self_attn.indexer.wq_b.weight",
        "self_attn.compressor.indexer.scorer.weights_proj.weight":
            "self_attn.indexer.weights_proj.weight",
        "self_attn.compressor.indexer.position_bias": "self_attn.indexer.compressor.ape",
        "self_attn.compressor.indexer.kv_proj.weight":
            "self_attn.indexer.compressor.wkv.weight",
        "self_attn.compressor.indexer.gate_proj.weight":
            "self_attn.indexer.compressor.wgate.weight",
        "self_attn.compressor.indexer.kv_norm.weight":
            "self_attn.indexer.compressor.norm.weight",
        "mlp.gate.weight": "ffn.gate.weight",
        "mlp.gate.tid2eid": "ffn.gate.tid2eid",
        "mlp.gate.e_score_correction_bias": "ffn.gate.bias",
        "mlp.experts.gate_up_proj": "ffn.experts.gate_up_proj",
        "mlp.experts.down_proj": "ffn.experts.down_proj",
        "mlp.shared_experts.gate_proj.weight": "ffn.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight": "ffn.shared_experts.up_proj.weight",
        "mlp.shared_experts.down_proj.weight": "ffn.shared_experts.down_proj.weight",
    }

    for name, tensor in state.items():
        mapped = exact.get(name)
        if mapped is None:
            match = re.match(r"^(model\.layers\.\d+)\.(.+)$", name)
            if match is not None:
                tail = suffixes.get(match.group(2))
                if tail is not None:
                    mapped = f"{match.group(1)}.{tail}"
        if mapped is None:
            # Non-persistent rotary buffers are absent. Every persistent V4
            # tensor must be understood; silently preserving a native-only name
            # would produce a fixture whose unconsumed mechanisms go unnoticed.
            if name in {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}:
                mapped = name
            else:
                raise RuntimeError(f"unmapped native DeepSeek V4 tensor: {name}")
        # Hash ids are integer buffers in Transformers and exact small integers.
        # Soma's release checkpoint carries routing metadata as lossless floats,
        # allowing the backend-owned payload binder to stay dtype-uniform.
        if mapped.endswith("ffn.gate.tid2eid"):
            tensor = tensor.to(dtype=next(model.parameters()).dtype)
        out[mapped] = tensor.detach().contiguous()
        if name_map is not None:
            name_map[name] = mapped
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="local model dir or HF repo id")
    ap.add_argument("--out", default="tests/fixtures/tiny", help="fixture root")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--layers", type=int, default=TARGET_LAYERS)
    ap.add_argument("--experts", type=int, default=TARGET_EXPERTS)
    ap.add_argument("--positions", type=int, default=512,
                    help="teacher-forced positions (gate requires >= 512)")
    ap.add_argument("--generate", type=int, default=256,
                    help="greedy tokens (gate requires >= 256)")
    ap.add_argument("--name", default=None, help="fixture slug override")
    args = ap.parse_args(argv[1:])

    # Imported late so --help works without torch installed.
    import numpy as np
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    src = Path(args.source)
    is_local = src.exists()
    raw_cfg_path = src / "config.json" if is_local else None
    if is_local and not raw_cfg_path.exists():
        print(f"error: {raw_cfg_path} not found", file=sys.stderr)
        return 2

    raw = json.loads(raw_cfg_path.read_text(encoding="utf-8")) if is_local else None

    slug = args.name or (src.name if is_local else args.source.split("/")[-1])
    out_dir = Path(args.out) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    if raw is None:
        cfg_obj = AutoConfig.from_pretrained(args.source, trust_remote_code=True)
        raw = cfg_obj.to_dict()

    tiny_cfg = shrink_config(raw, args.layers, args.experts)
    kinds = layer_kinds(tiny_cfg)

    print(f"  source      : {args.source}")
    print(f"  model_type  : {raw.get('model_type')}")
    print(f"  layers      : {raw.get('num_hidden_layers')} -> {tiny_cfg['num_hidden_layers']}  {kinds}")
    print(f"  hidden      : {raw.get('hidden_size')} -> {tiny_cfg['hidden_size']}")
    print(f"  heads/kv    : {raw.get('num_attention_heads')}/{raw.get('num_key_value_heads')}"
          f" -> {tiny_cfg['num_attention_heads']}/{tiny_cfg['num_key_value_heads']}")
    ek = _first_present(tiny_cfg, EXPERT_COUNT_KEYS)
    tk = _first_present(tiny_cfg, TOPK_KEYS)
    if ek:
        print(f"  experts     : {raw.get(ek)} -> {tiny_cfg[ek]}  top_k {raw.get(tk)} -> {tiny_cfg[tk]}")
    print(f"  vocab       : {raw.get('vocab_size')} -> {tiny_cfg['vocab_size']}")

    (out_dir / "config.json").write_text(
        json.dumps(tiny_cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Build the model from the tiny config. trust_remote_code pulls the family's
    # modeling_*.py from the source dir for custom_code repos (DeepSeek, Moonlight).
    torch.manual_seed(args.seed)
    # Build the config object from the SOURCE directory, then apply the tiny
    # overrides on top.
    #
    # custom_code families (DeepSeek-V2-Lite, Moonlight) resolve
    # configuration_deepseek.py / modeling_deepseek.py relative to whatever path
    # the config was loaded from. Loading from the fixture directory makes
    # transformers look for those .py files there, and vendoring them would mean
    # copying third-party modeling code into this repo. Loading from the source
    # and mutating fields avoids both.
    # Prefer transformers' NATIVE implementation over the checkpoint's vendored
    # modeling_*.py, and fall back to remote code only for genuinely unknown
    # architectures.
    #
    # This is not just tidiness. A checkpoint's remote code is a snapshot frozen
    # at whatever transformers version it was published against —
    # DeepSeek-V2-Lite's targets 4.33 and calls DynamicCache.get_usable_length(),
    # removed since. Meanwhile Qwen3-MoE needs >= 4.51. No single version
    # satisfies both through remote code.
    #
    # Native support makes the fixture reproducible from `pip install` alone,
    # with no dependency on files inside the model repo.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    model_type = raw.get("model_type")
    native = model_type in CONFIG_MAPPING_NAMES

    if native:
        # Load from the fixture's own clean config (auto_map already stripped),
        # so transformers applies its normal normalization to the tiny values.
        cfg_obj = AutoConfig.from_pretrained(str(out_dir))
    else:
        load_from = str(src) if is_local else args.source
        cfg_obj = AutoConfig.from_pretrained(load_from, trust_remote_code=True)
        # Apply ONLY the keys shrink_config changed. Re-applying every raw key
        # would clobber fields transformers normalizes during from_pretrained —
        # e.g. the legacy `rope_scaling` migrating into `rope_parameters`, where
        # writing back `rope_scaling: None` leaves construction to fail on
        # `config.rope_parameters["rope_type"]`.
        for key, value in {k: v for k, v in tiny_cfg.items() if raw.get(k) != v}.items():
            setattr(cfg_obj, key, value)

    print(f"  impl        : {'native transformers' if native else 'remote code'}")

    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_config(cfg_obj, trust_remote_code=not native)
    model = model.to(torch.float32).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params      : {n_params:,}  ({n_params * 4 / 1e6:.1f} MB fp32)")

    # Deterministic weights. Default init is already seeded, but re-initializing
    # explicitly makes the fixture independent of transformers' init changing
    # between versions — which would otherwise silently invalidate every golden
    # file on a library bump.
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if p.dim() >= 2:
                p.copy_(torch.empty_like(p).uniform_(-0.08, 0.08, generator=gen))
            else:
                # Norms initialize to ~1, biases to ~0. Perturbed so a dropped
                # norm weight cannot pass by multiplying by exactly 1.
                base = 1.0 if "norm" in name.lower() or "layernorm" in name.lower() else 0.0
                p.copy_(torch.full_like(p, base)
                        + torch.empty_like(p).uniform_(-0.02, 0.02, generator=gen))

        # The native initializers intentionally leave routing metadata at zero
        # for a real checkpoint to overwrite. A random fixture has no checkpoint,
        # so populate both routing modes deterministically and non-degenerately.
        for name, b in sorted(model.named_buffers()):
            if name.endswith("tid2eid"):
                layer_match = re.search(r"layers\.(\d+)\.", name)
                layer = int(layer_match.group(1)) if layer_match else 0
                token = torch.arange(b.shape[0], device=b.device).unsqueeze(1)
                slot = torch.arange(b.shape[1], device=b.device).unsqueeze(0)
                b.copy_((token * 13 + slot * 3 + layer * 5) % tiny_cfg[ek])
            elif name.endswith("e_score_correction_bias"):
                b.copy_(torch.linspace(-0.02, 0.02, b.numel(), dtype=b.dtype, device=b.device))

    vocab = tiny_cfg["vocab_size"]
    ids = torch.randint(0, vocab, (1, args.positions), generator=gen, dtype=torch.long)

    with torch.no_grad():
        # use_cache=False: G0 is a single-sequence, no-KV-reuse comparison, and
        # the greedy loop below re-runs the full prefix each step anyway. Running
        # the reference without a cache removes an entire class of divergence
        # between the oracle and the engine from the ground truth.
        tf_logits = model(ids, use_cache=False).logits.to(torch.float32)

    # Greedy from a short prefix. Hand-rolled rather than model.generate() so the
    # loop matches what the engine does: argmax, append, re-run. generate()
    # applies sampling config and stopping criteria that would make an exact
    # comparison depend on transformers' generation defaults.
    prefix = ids[:, :16]
    cur = prefix.clone()
    greedy: list[int] = []
    with torch.no_grad():
        for _ in range(args.generate):
            out = model(cur, use_cache=False).logits[:, -1, :]
            nxt = int(torch.argmax(out, dim=-1).item())
            greedy.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long)], dim=1)

    ids_np = ids.numpy().astype(np.int32)
    tf_np = tf_logits.numpy().astype(np.float32)
    prefix_np = prefix.numpy().astype(np.int32)
    greedy_np = np.array(greedy, dtype=np.int32)

    np.savez_compressed(
        out_dir / "oracle.npz",
        input_ids=ids_np, tf_logits=tf_np,
        greedy_prefix=prefix_np, greedy_tokens=greedy_np,
    )

    # Flat little-endian sidecar for the C++ conformance test.
    #
    # .npz is a zip archive of .npy members, so reading it from C++ would mean
    # pulling in zlib and an npy parser to move numbers that were never
    # compressible in the first place (fp32 logits). The Python consumers keep
    # the .npz; the engine reads this.
    with open(out_dir / "oracle.bin", "wb") as fh:
        fh.write(b"SOMAORCL")
        fh.write(np.array([1, args.positions, vocab, prefix_np.shape[1], len(greedy)],
                          dtype="<u4").tobytes())
        fh.write(ids_np.astype("<i4").tobytes())
        fh.write(tf_np.astype("<f4").tobytes())
        fh.write(prefix_np.astype("<i4").tobytes())
        fh.write(greedy_np.astype("<i4").tobytes())

    # Weights last, so a crash above never leaves a fixture that looks complete.
    #
    # save_model rather than save_file: with tie_word_embeddings (granite),
    # lm_head.weight and embed_tokens.weight are the SAME storage, and save_file
    # refuses to write shared tensors. save_model drops the duplicate and records
    # the tie in metadata. The C++ loader reconstructs it from
    # config.tie_word_embeddings, which is already in the fixture config.
    if model_type == "deepseek_v4":
        from safetensors.torch import save_file
        native_to_soma: dict[str, str] = {}
        save_file(canonicalize_deepseek_v4_state(model, native_to_soma),
                  str(out_dir / "model.safetensors"))
        (out_dir / "native_to_soma.json").write_text(
            json.dumps(native_to_soma, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        from safetensors.torch import save_model
        save_model(model, str(out_dir / "model.safetensors"))

    weights_sha = hashlib.sha256((out_dir / "model.safetensors").read_bytes()).hexdigest()
    logits_sha = hashlib.sha256(tf_logits.numpy().tobytes()).hexdigest()

    meta = {
        "fixture_version": 1,
        "slug": slug,
        "source": str(args.source),
        "source_model_type": raw.get("model_type"),
        "implementation": "native" if native else "remote_code",
        "source_architectures": raw.get("architectures"),
        "seed": args.seed,
        "layer_kinds": kinds,
        "n_params": n_params,
        "positions": args.positions,
        "generate": args.generate,
        "weights_sha256": weights_sha,
        "tf_logits_sha256": logits_sha,
        "greedy_first_16": greedy[:16],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    print(f"  greedy[:8]  : {greedy[:8]}")
    print(f"  logits sha  : {logits_sha[:16]}")
    print(f"  -> {out_dir}  ({total / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
