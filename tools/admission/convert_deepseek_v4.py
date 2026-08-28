"""Pinned DeepSeek-V4-Pro-0813 -> Soma conversion.

The published checkpoint uses E4M3/E8M0 block-scaled dense tensors and packed
E2M1/E8M0 experts.  Soma deliberately has no runtime dependency on either
format: this converter dequantizes one tensor at a time, then writes the existing
Soma resident qweight sidecars and routed q4/q6 container layouts. Norms,
routing data, sinks, and hyper-connection controls remain lossless F32.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
from pathlib import Path
from typing import Any

from convert import (ALIGN, DTYPE_ID, FORMAT_VERSION, MAGIC, align_up,
                     quantize_rows)

PINNED_REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"
PINNED_CONFIG_SHA256 = "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0"
PINNED_INDEX_SHA256 = "2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a"
FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
DSPARK_STAGES = 3
DSPARK_KV_BYTES_PER_SEQUENCE = 3 * 128 * 512 * 2


def canonical_name(name: str) -> str:
    """Map the release checkpoint spelling to Soma's stable loader spelling."""
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name == "head.weight":
        return "lm_head.weight"
    if name.startswith("hc_head_"):
        return "model." + name
    if not name.startswith("layers."):
        return name
    name = "model." + name
    name = name.replace(".attn.", ".self_attn.")
    name = name.replace(".attn_norm.weight", ".input_layernorm.weight")
    name = name.replace(".ffn_norm.weight", ".post_attention_layernorm.weight")
    name = name.replace(".ffn.shared_experts.w1.",
                        ".ffn.shared_experts.gate_proj.")
    name = name.replace(".ffn.shared_experts.w3.",
                        ".ffn.shared_experts.up_proj.")
    name = name.replace(".ffn.shared_experts.w2.",
                        ".ffn.shared_experts.down_proj.")
    return name


def dspark_canonical_name(name: str) -> str:
    """Map one mtp.* release tensor into a stable DSpark namespace."""
    match = re.match(r"^mtp\.([012])\.(.+)$", name)
    if not match:
        raise ValueError(f"not a DSpark tensor: {name}")
    stage, tail = int(match.group(1)), match.group(2)
    special = {
        (0, "main_proj.weight"): "model.dspark.main_proj.weight",
        (0, "main_proj.scale"): "model.dspark.main_proj.scale",
        (0, "main_norm.weight"): "model.dspark.main_norm.weight",
        (2, "norm.weight"): "model.dspark.norm.weight",
        (2, "markov_head.markov_w1.weight"): "model.dspark.markov_w1.weight",
        (2, "markov_head.markov_w2.weight"): "model.dspark.markov_w2.weight",
        (2, "confidence_head.proj.weight"): "model.dspark.confidence_proj.weight",
        (2, "hc_head_fn"): "model.dspark.hc_head_fn",
        (2, "hc_head_base"): "model.dspark.hc_head_base",
        (2, "hc_head_scale"): "model.dspark.hc_head_scale",
    }
    if (stage, tail) in special:
        return special[(stage, tail)]
    prefix = f"model.dspark.layers.{stage}."
    tail = tail.replace("attn.", "self_attn.", 1)
    tail = tail.replace("attn_norm.weight", "input_layernorm.weight")
    tail = tail.replace("ffn_norm.weight", "post_attention_layernorm.weight")
    tail = tail.replace("ffn.shared_experts.w1.",
                        "ffn.shared_experts.gate_proj.")
    tail = tail.replace("ffn.shared_experts.w3.",
                        "ffn.shared_experts.up_proj.")
    tail = tail.replace("ffn.shared_experts.w2.",
                        "ffn.shared_experts.down_proj.")
    return prefix + tail


def dense_names(cfg: dict, n_layers: int) -> list[list[str]]:
    top = [
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        "model.hc_head_fn", "model.hc_head_base", "model.hc_head_scale",
    ]
    groups = [top]
    ratios = cfg["compress_ratios"]
    for layer in range(n_layers):
        p = f"model.layers.{layer}."
        a = p + "self_attn."
        names = [
            p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
            p + "hc_attn_fn", p + "hc_attn_base", p + "hc_attn_scale",
            p + "hc_ffn_fn", p + "hc_ffn_base", p + "hc_ffn_scale",
            a + "attn_sink", a + "wq_a.weight", a + "wq_b.weight",
            a + "q_norm.weight", a + "wkv.weight", a + "kv_norm.weight",
            a + "wo_a.weight", a + "wo_b.weight",
            a + "compressor.ape", a + "compressor.wkv.weight",
            a + "compressor.wgate.weight", a + "compressor.norm.weight",
            p + "ffn.gate.weight",
            p + ("ffn.gate.tid2eid" if layer < int(cfg["num_hash_layers"])
                 else "ffn.gate.bias"),
            p + "ffn.shared_experts.gate_proj.weight",
            p + "ffn.shared_experts.up_proj.weight",
            p + "ffn.shared_experts.down_proj.weight",
        ]
        if int(ratios[layer]) == 4:
            names += [
                a + "indexer.wq_b.weight", a + "indexer.weights_proj.weight",
                a + "indexer.compressor.ape", a + "indexer.compressor.wkv.weight",
                a + "indexer.compressor.wgate.weight",
                a + "indexer.compressor.norm.weight",
            ]
        groups.append(names)
    return groups


def dspark_dense_names() -> list[list[str]]:
    """All non-routed DSpark tensors, grouped into atomic resident shards."""
    groups: list[list[str]] = []
    for stage in range(DSPARK_STAGES):
        p = f"model.dspark.layers.{stage}."
        a = p + "self_attn."
        names = [
            p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
            p + "hc_attn_fn", p + "hc_attn_base", p + "hc_attn_scale",
            p + "hc_ffn_fn", p + "hc_ffn_base", p + "hc_ffn_scale",
            a + "attn_sink", a + "wq_a.weight", a + "wq_b.weight",
            a + "q_norm.weight", a + "wkv.weight", a + "kv_norm.weight",
            a + "wo_a.weight", a + "wo_b.weight",
            p + "ffn.gate.weight", p + "ffn.gate.bias",
            p + "ffn.shared_experts.gate_proj.weight",
            p + "ffn.shared_experts.up_proj.weight",
            p + "ffn.shared_experts.down_proj.weight",
        ]
        if stage == 0:
            names += ["model.dspark.main_proj.weight", "model.dspark.main_norm.weight"]
        if stage == 2:
            names += [
                "model.dspark.norm.weight", "model.dspark.markov_w1.weight",
                "model.dspark.markov_w2.weight", "model.dspark.confidence_proj.weight",
                "model.dspark.hc_head_fn", "model.dspark.hc_head_base",
                "model.dspark.hc_head_scale",
            ]
        groups.append(names)
    return groups


def is_quantized_resident(name: str) -> bool:
    """Resident tensors stored directly in a Soma runtime quant layout."""
    if name in {"model.embed_tokens.weight", "lm_head.weight"}:
        return True
    if ".ffn.shared_experts." in name and name.endswith("_proj.weight"):
        return True
    if ".self_attn." not in name or not name.endswith(".weight"):
        return False
    return name.endswith((
        ".wq_a.weight", ".wq_b.weight", ".wkv.weight", ".wo_a.weight",
        ".wo_b.weight", ".wgate.weight", ".weights_proj.weight",
    ))


def is_quantized_dspark_resident(name: str) -> bool:
    """DSpark matrices stored in the existing indexed QTensor sidecar."""
    if name.endswith("ffn.gate.weight"):
        return False  # routing remains lossless by architecture contract
    if ".ffn.shared_experts." in name and name.endswith("_proj.weight"):
        return True
    if name in {
        "model.dspark.main_proj.weight", "model.dspark.markov_w1.weight",
        "model.dspark.markov_w2.weight", "model.dspark.confidence_proj.weight",
    }:
        return True
    return ".self_attn." in name and name.endswith((
        ".wq_a.weight", ".wq_b.weight", ".wkv.weight",
        ".wo_a.weight", ".wo_b.weight",
    ))


def source_revision(src: Path) -> str:
    meta = src / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if not meta.is_file():
        return ""
    return meta.read_text(encoding="utf-8").splitlines()[0].strip()


def run(args) -> int:
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.numpy import save_file

    src, out_dir = Path(args.model_dir), Path(args.out)
    cfg_raw = (src / "config.json").read_bytes()
    config_sha = hashlib.sha256(cfg_raw).hexdigest()
    cfg = json.loads(cfg_raw)
    if cfg.get("model_type") != "deepseek_v4":
        raise ValueError("DeepSeek-V4 converter called for a different model_type")

    fixture_mode = bool(getattr(args, "test_fixture", False))
    fixture_has_dspark_oracle = cfg.get("dspark_oracle") == "pinned-model-py-cpu-v1"
    if fixture_mode and not (
        int(cfg.get("num_hidden_layers", 0)) <= 8
        and int(cfg.get("n_routed_experts", 0)) <= 32
        and ((cfg.get("semantic_fp8_quant_dequant") is False
              and cfg.get("semantic_fp4_quant_dequant") is False)
             or fixture_has_dspark_oracle)
        and (src / "model.safetensors").is_file()
    ):
        print("  REFUSED  --test-fixture is restricted to the tiny native V4 fixture")
        return 3

    revision = source_revision(src)
    requested = "tiny-fixture" if fixture_mode else (args.source_revision or PINNED_REVISION)
    if not fixture_mode and not revision:
        print("  REFUSED  production V4 conversion requires Hugging Face revision metadata")
        return 3
    if revision and revision != requested:
        print(f"  REFUSED  source revision {revision} != pinned {requested}")
        return 3
    if not fixture_mode and requested != PINNED_REVISION:
        print(f"  REFUSED  DeepSeek-V4-Pro-0813 support is pinned to {PINNED_REVISION}")
        return 3

    n_layers = int(cfg["num_hidden_layers"])
    if args.layers:
        n_layers = min(n_layers, args.layers)
    n_experts = int(cfg["n_routed_experts"])
    dt_gate = args.quant
    dt_down = args.expert_down or "q6_g"
    if dt_gate not in DTYPE_ID or dt_down not in DTYPE_ID:
        print("  REFUSED  unsupported target quantization")
        return 3

    index_path = src / "model.safetensors.index.json"
    if index_path.is_file():
        index_raw = index_path.read_bytes()
        weight_map = json.loads(index_raw).get("weight_map", {})
    elif fixture_mode:
        single = src / "model.safetensors"
        with safe_open(str(single), framework="pt", device="cpu") as fixture:
            weight_map = {name: single.name for name in fixture.keys()}
        index_raw = json.dumps(
            {"weight_map": weight_map}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    else:
        print("  REFUSED  pinned V4 release requires model.safetensors.index.json")
        return 3
    index_sha = hashlib.sha256(index_raw).hexdigest()
    if not fixture_mode and (config_sha != PINNED_CONFIG_SHA256 or
                             index_sha != PINNED_INDEX_SHA256):
        print("  REFUSED  config/index hashes do not match the pinned V4 release")
        print(f"             config {config_sha}")
        print(f"             index  {index_sha}")
        return 3
    mtp_namespaces = sorted({".".join(name.split(".")[:2])
                             for name in weight_map if name.startswith("mtp.")})
    unexpected_mtp = [name for name in weight_map
                      if name.startswith("mtp.") and not re.match(r"^mtp\.[012]\.", name)]
    if (not fixture_mode and
            (unexpected_mtp or mtp_namespaces != ["mtp.0", "mtp.1", "mtp.2"])):
        print("  REFUSED  only the three pinned DSpark namespaces mtp.0/1/2 may be omitted")
        return 3
    include_dspark = bool(getattr(args, "include_dspark", False))
    if fixture_mode and mtp_namespaces and not fixture_has_dspark_oracle:
        print("  REFUSED  only a pinned-model.py DSpark oracle fixture may contain mtp tensors")
        return 3
    if fixture_mode and include_dspark and mtp_namespaces != ["mtp.0", "mtp.1", "mtp.2"]:
        print("  REFUSED  the DSpark oracle fixture requires exactly mtp.0/1/2")
        return 3
    if fixture_mode and include_dspark and not fixture_has_dspark_oracle:
        print("  REFUSED  the base-only tiny fixture has no DSpark tensors")
        return 3
    canonical: dict[str, str] = {}
    dspark_canonical: dict[str, str] = {}
    for raw in weight_map:
        if raw.startswith("mtp."):
            name = dspark_canonical_name(raw)
            if name in dspark_canonical:
                print(f"  REFUSED  DSpark tensor collision: {raw} and {dspark_canonical[name]}")
                return 3
            dspark_canonical[name] = raw
            continue
        name = canonical_name(raw)
        if name in canonical:
            print(f"  REFUSED  canonical tensor collision: {raw} and {canonical[name]}")
            return 3
        canonical[name] = raw

    groups = dense_names(cfg, n_layers)
    dense_claimed = {name for group in groups for name in group}
    expert_re = re.compile(r"^model\.layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight$")
    fixture_experts = {
        f"model.layers.{layer}.ffn.experts.{suffix}"
        for layer in range(n_layers)
        for suffix in ("gate_up_proj", "down_proj")
    } if fixture_mode else set()
    unclaimed = []
    omitted_mtp = [name for name in weight_map if name.startswith("mtp.")]
    for name, raw in canonical.items():
        if name in dense_claimed or name in fixture_experts or expert_re.match(name):
            continue
        if name.endswith(".scale") and name[:-6] + ".weight" in canonical:
            continue
        # Debug --layers deliberately omits the tail of the base stack.
        m = re.match(r"model\.layers\.(\d+)\.", name)
        if m and int(m.group(1)) >= n_layers:
            continue
        unclaimed.append((name, raw))
    missing = sorted(name for name in dense_claimed | fixture_experts
                     if name not in canonical)
    if unclaimed or missing:
        print(f"  REFUSED  V4 tensor coverage: {len(missing)} missing, {len(unclaimed)} unclaimed")
        for name in missing[:12]:
            print(f"             missing {name}")
        for name, raw in unclaimed[:12]:
            print(f"             unclaimed {raw} -> {name}")
        return 3

    dspark_groups = dspark_dense_names() if include_dspark else []
    dspark_dense_claimed = {name for group in dspark_groups for name in group}
    dspark_expert_re = re.compile(
        r"^model\.dspark\.layers\.([012])\.ffn\.experts\.(\d+)\.w([123])\.weight$")
    if include_dspark:
        dspark_unclaimed = []
        for name, raw in dspark_canonical.items():
            if name in dspark_dense_claimed or dspark_expert_re.match(name):
                continue
            if name.endswith(".scale") and name[:-6] + ".weight" in dspark_canonical:
                continue
            dspark_unclaimed.append((name, raw))
        dspark_missing = sorted(name for name in dspark_dense_claimed
                                if name not in dspark_canonical)
        expected_experts = DSPARK_STAGES * n_experts * 3
        actual_experts = sum(bool(dspark_expert_re.match(name))
                             for name in dspark_canonical)
        if dspark_unclaimed or dspark_missing or actual_experts != expected_experts:
            print("  REFUSED  DSpark tensor coverage: "
                  f"{len(dspark_missing)} missing, {len(dspark_unclaimed)} unclaimed, "
                  f"{actual_experts}/{expected_experts} expert projections")
            for name in dspark_missing[:12]:
                print(f"             missing {name}")
            for name, raw in dspark_unclaimed[:12]:
                print(f"             unclaimed {raw} -> {name}")
            return 3

    identity = requested if fixture_mode else f"pinned {requested[:12]}"
    dspark_status = (f"included ({len(omitted_mtp)} tensors)" if include_dspark
                     else f"omitted ({len(omitted_mtp)} tensors)")
    print(f"  {src.name}: {identity}, {n_layers} layers x {n_experts} experts; "
          f"gate/up={dt_gate}, down={dt_down}, DSpark {dspark_status}")
    if args.validate_only:
        what = "base and all DSpark" if include_dspark else "all base"
        print(f"  OK       {what} tensors accounted for; no weight payload read")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "conversion-manifest.json"
    manifest = {
        "format": 1, "source_revision": requested, "config_sha256": config_sha,
        "index_sha256": index_sha,
        "quant_gate_up": dt_gate, "quant_down": dt_down, "group": args.group,
        "completed_expert_layers": {}, "completed_dense_shards": {},
        "completed_qweight_shards": {},
        "completed_dspark_expert_layers": {}, "completed_dspark_dense_shards": {},
        "completed_dspark_qweight_shards": {},
        "omitted_namespaces": mtp_namespaces,
    }
    if not args.no_resume and manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = ("source_revision", "config_sha256", "index_sha256",
                    "quant_gate_up", "quant_down", "group")
        if any(old.get(k) != manifest.get(k) for k in identity):
            print("  REFUSED  existing conversion manifest describes different source/quantization")
            return 3
        manifest = old
        manifest.setdefault("completed_qweight_shards", {})
        manifest.setdefault("completed_dspark_expert_layers", {})
        manifest.setdefault("completed_dspark_dense_shards", {})
        manifest.setdefault("completed_dspark_qweight_shards", {})
        manifest.setdefault("omitted_namespaces", mtp_namespaces)

    owner = {name: str(src / shard) for name, shard in weight_map.items()}
    open_handles: dict[str, Any] = {}

    def handle_for(path: str):
        if path in open_handles:
            return open_handles[path]
        if len(open_handles) >= 2:
            key, h = next(iter(open_handles.items()))
            del open_handles[key]
            h.__exit__(None, None, None)
        h = safe_open(path, framework="pt", device="cpu")
        h.__enter__()
        open_handles[path] = h
        return h

    def raw_tensor(raw: str):
        return handle_for(owner[raw]).get_tensor(raw)

    fp4_table = torch.tensor(FP4_LEVELS, dtype=torch.float32)

    def get(name: str):
        raw = canonical[name]
        t = raw_tensor(raw)
        scale_name = name[:-7] + ".scale" if name.endswith(".weight") else ""
        raw_scale = canonical.get(scale_name)
        if t.dtype == torch.int8 and raw_scale:
            scale = raw_tensor(raw_scale).float()
            u = t.view(torch.uint8)
            vals = torch.stack((fp4_table[(u & 15).long()],
                                fp4_table[((u >> 4) & 15).long()]), dim=-1).flatten(-2)
            vals = vals * scale.repeat_interleave(32, dim=-1)
            return vals.numpy().copy()
        if str(t.dtype).startswith("torch.float8") and raw_scale:
            scale = raw_tensor(raw_scale).float()
            out = t.float()
            for ro in range(scale.shape[0]):
                r0, r1 = ro * 128, min((ro + 1) * 128, out.shape[0])
                for co in range(scale.shape[1]):
                    c0, c1 = co * 128, min((co + 1) * 128, out.shape[1])
                    out[r0:r1, c0:c1] *= scale[ro, co]
            return out.numpy().copy()
        return t.float().numpy().copy()

    def get_dspark(name: str):
        raw = dspark_canonical[name]
        t = raw_tensor(raw)
        scale_name = name[:-7] + ".scale" if name.endswith(".weight") else ""
        raw_scale = dspark_canonical.get(scale_name)
        if t.dtype == torch.int8 and raw_scale:
            scale = raw_tensor(raw_scale).float()
            u = t.view(torch.uint8)
            vals = torch.stack((fp4_table[(u & 15).long()],
                                fp4_table[((u >> 4) & 15).long()]), dim=-1).flatten(-2)
            vals = vals * scale.repeat_interleave(32, dim=-1)
            return vals.numpy().copy()
        if str(t.dtype).startswith("torch.float8") and raw_scale:
            scale = raw_tensor(raw_scale).float()
            out = t.float()
            for ro in range(scale.shape[0]):
                r0, r1 = ro * 128, min((ro + 1) * 128, out.shape[0])
                for co in range(scale.shape[1]):
                    c0, c1 = co * 128, min((co + 1) * 128, out.shape[1])
                    out[r0:r1, c0:c1] *= scale[ro, co]
            return out.numpy().copy()
        return t.float().numpy().copy()

    def save_manifest():
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, manifest_path)

    # Routed experts: one atomic file per layer makes resume validation cheap and
    # ensures no partial shard can masquerade as complete.
    all_index: list[tuple[int, int, int]] = []
    total_payload = 0
    uniform_len = -1
    effective_groups: dict[str, int] = {}
    for layer in range(n_layers):
        final = out_dir / f"experts-{layer:05d}.bin"
        done = manifest["completed_expert_layers"].get(str(layer))
        entries = done.get("entries", []) if done else []
        if not (done and final.is_file() and final.stat().st_size == done.get("file_bytes")
                and len(entries) == n_experts):
            tmp = final.with_suffix(".bin.tmp")
            offsets = []
            off = 0
            fused_gate_up = None
            fused_down = None
            if fixture_mode:
                fused_gate_up = get(f"model.layers.{layer}.ffn.experts.gate_up_proj")
                fused_down = get(f"model.layers.{layer}.ffn.experts.down_proj")
                inter = int(cfg["moe_intermediate_size"])
                if (fused_gate_up.shape[0] != n_experts or
                        fused_gate_up.shape[1] != 2 * inter or
                        fused_down.shape[0] != n_experts):
                    raise ValueError(f"tiny fixture layer {layer} has invalid fused expert shapes")
            with open(tmp, "wb") as fh:
                for expert in range(n_experts):
                    blob = bytearray()
                    for proj, dtype in (("w1", dt_gate), ("w3", dt_gate), ("w2", dt_down)):
                        if fixture_mode:
                            if proj == "w1": arr = fused_gate_up[expert, :inter]
                            elif proj == "w3": arr = fused_gate_up[expert, inter:]
                            else: arr = fused_down[expert]
                        else:
                            name = f"model.layers.{layer}.ffn.experts.{expert}.{proj}.weight"
                            arr = get(name)
                        packed, group = quantize_rows(arr, dtype, args.group)
                        effective_groups[dtype] = group
                        blob += packed
                    offsets.append([off, len(blob)])
                    fh.write(blob)
                    pad = align_up(off + len(blob)) - (off + len(blob))
                    if pad:
                        fh.write(b"\0" * pad)
                    off = align_up(off + len(blob))
                    if uniform_len < 0:
                        uniform_len = len(blob)
                    elif uniform_len != len(blob):
                        uniform_len = 0
            os.replace(tmp, final)
            done = {"file_bytes": final.stat().st_size, "entries": offsets}
            manifest["completed_expert_layers"][str(layer)] = done
            save_manifest()
            entries = offsets
        for off, length in entries:
            all_index.append((layer, int(off), int(length)))
            total_payload += int(length)
            if uniform_len < 0:
                uniform_len = int(length)
            elif uniform_len != int(length):
                uniform_len = 0
        print(f"    experts layer {layer + 1}/{n_layers}  {total_payload / 1e9:.2f} GB", flush=True)

    # Resident tensors: top-level then one shard per layer. Lossless controls,
    # norms, sinks, and routing metadata stay SafeTensors. Projections,
    # embeddings, shared experts, and the output head are quantized one tensor at
    # a time into a parallel binary sidecar already laid out exactly as QTensor.
    dense_weight_map: dict[str, str] = {}
    qweight_map: dict[str, dict[str, Any]] = {}
    dense_total = 0
    qweight_total = 0
    for group_id, names in enumerate(groups):
        lossless = [name for name in names if not is_quantized_resident(name)]
        quantized = [name for name in names if is_quantized_resident(name)]
        shard = f"dense-{group_id:05d}.safetensors"
        final = out_dir / shard
        done = manifest["completed_dense_shards"].get(str(group_id))
        if not (done and final.is_file() and final.stat().st_size == done.get("file_bytes")):
            # These groups are intentionally small: the large matrices are in
            # `quantized` below. `get()` still decodes each source tensor once.
            payload = {name: get(name).astype("<f4", copy=False) for name in lossless}
            tmp = final.with_suffix(".safetensors.tmp")
            save_file(payload, str(tmp))
            os.replace(tmp, final)
            done = {"file_bytes": final.stat().st_size, "names": lossless}
            manifest["completed_dense_shards"][str(group_id)] = done
            save_manifest()
        for name in done["names"]:
            dense_weight_map[name] = shard
        dense_total += int(done["file_bytes"])

        qshard = f"dense-q-{group_id:05d}.bin"
        qfinal = out_dir / qshard
        qdone = manifest["completed_qweight_shards"].get(str(group_id))
        qentries = qdone.get("entries", {}) if qdone else {}
        if not (qdone and qfinal.is_file() and qfinal.stat().st_size == qdone.get("file_bytes")
                and set(qentries) == set(quantized)):
            qtmp = qfinal.with_suffix(".bin.tmp")
            qentries = {}
            off = 0
            with open(qtmp, "wb") as fh:
                for name in quantized:
                    arr = get(name)
                    if arr.ndim != 2:
                        raise ValueError(f"resident projection {name} is rank {arr.ndim}, expected 2")
                    packed, group = quantize_rows(arr, args.quant, args.group)
                    qentries[name] = {
                        "file": qshard, "offset": off, "length": len(packed),
                        "dtype": args.quant, "group": group,
                        "shape": [int(arr.shape[0]), int(arr.shape[1])],
                    }
                    fh.write(packed)
                    padded = align_up(off + len(packed))
                    if padded > off + len(packed): fh.write(b"\0" * (padded - off - len(packed)))
                    off = padded
                    del arr, packed
            os.replace(qtmp, qfinal)
            qdone = {"file_bytes": qfinal.stat().st_size, "entries": qentries}
            manifest["completed_qweight_shards"][str(group_id)] = qdone
            save_manifest()
        qweight_map.update(qdone["entries"])
        qweight_total += sum(int(entry["length"]) for entry in qdone["entries"].values())
        print(f"    resident shard {group_id + 1}/{len(groups)}  "
              f"{(dense_total + qweight_total) / 1e9:.2f} GB", flush=True)

    # DSpark is an augmentation, not a second base conversion. Its files use a
    # distinct namespace and the old container metadata is not changed until
    # every expert and resident shard below is complete. A crash therefore
    # leaves an autoregressive-only container which remains safe to serve; a
    # retry validates each atomic shard and resumes at the first missing one.
    dspark_all_index: list[tuple[int, int, int]] = []
    dspark_total_payload = 0
    dspark_uniform_len = -1
    dspark_dense_weight_map: dict[str, str] = {}
    dspark_qweight_map: dict[str, dict[str, Any]] = {}
    dspark_dense_total = 0
    dspark_qweight_total = 0
    if include_dspark:
        for stage in range(DSPARK_STAGES):
            final = out_dir / f"dspark-experts-{stage:05d}.bin"
            done = manifest["completed_dspark_expert_layers"].get(str(stage))
            entries = done.get("entries", []) if done else []
            if not (done and final.is_file() and final.stat().st_size == done.get("file_bytes")
                    and len(entries) == n_experts):
                tmp = final.with_suffix(".bin.tmp")
                offsets = []
                off = 0
                with open(tmp, "wb") as fh:
                    for expert in range(n_experts):
                        blob = bytearray()
                        for proj, dtype in (("w1", dt_gate), ("w3", dt_gate),
                                            ("w2", dt_down)):
                            name = (f"model.dspark.layers.{stage}.ffn.experts."
                                    f"{expert}.{proj}.weight")
                            packed, group = quantize_rows(get_dspark(name), dtype, args.group)
                            effective_groups[dtype] = group
                            blob += packed
                        offsets.append([off, len(blob)])
                        fh.write(blob)
                        padded = align_up(off + len(blob))
                        if padded > off + len(blob):
                            fh.write(b"\0" * (padded - off - len(blob)))
                        off = padded
                        if dspark_uniform_len < 0:
                            dspark_uniform_len = len(blob)
                        elif dspark_uniform_len != len(blob):
                            dspark_uniform_len = 0
                os.replace(tmp, final)
                done = {"file_bytes": final.stat().st_size, "entries": offsets}
                manifest["completed_dspark_expert_layers"][str(stage)] = done
                save_manifest()
                entries = offsets
            for off, length in entries:
                dspark_all_index.append((stage, int(off), int(length)))
                dspark_total_payload += int(length)
                if dspark_uniform_len < 0:
                    dspark_uniform_len = int(length)
                elif dspark_uniform_len != int(length):
                    dspark_uniform_len = 0
            print(f"    DSpark experts stage {stage + 1}/{DSPARK_STAGES}  "
                  f"{dspark_total_payload / 1e9:.2f} GB", flush=True)

        for group_id, names in enumerate(dspark_groups):
            lossless = [name for name in names if not is_quantized_dspark_resident(name)]
            quantized = [name for name in names if is_quantized_dspark_resident(name)]
            shard = f"dspark-dense-{group_id:05d}.safetensors"
            final = out_dir / shard
            done = manifest["completed_dspark_dense_shards"].get(str(group_id))
            if not (done and final.is_file() and final.stat().st_size == done.get("file_bytes")):
                payload = {name: get_dspark(name).astype("<f4", copy=False)
                           for name in lossless}
                tmp = final.with_suffix(".safetensors.tmp")
                save_file(payload, str(tmp))
                os.replace(tmp, final)
                done = {"file_bytes": final.stat().st_size, "names": lossless}
                manifest["completed_dspark_dense_shards"][str(group_id)] = done
                save_manifest()
            for name in done["names"]:
                dspark_dense_weight_map[name] = shard
            dspark_dense_total += int(done["file_bytes"])

            qshard = f"dspark-dense-q-{group_id:05d}.bin"
            qfinal = out_dir / qshard
            qdone = manifest["completed_dspark_qweight_shards"].get(str(group_id))
            qentries = qdone.get("entries", {}) if qdone else {}
            if not (qdone and qfinal.is_file() and
                    qfinal.stat().st_size == qdone.get("file_bytes") and
                    set(qentries) == set(quantized)):
                qtmp = qfinal.with_suffix(".bin.tmp")
                qentries = {}
                off = 0
                with open(qtmp, "wb") as fh:
                    for name in quantized:
                        arr = get_dspark(name)
                        if arr.ndim != 2:
                            raise ValueError(
                                f"DSpark projection {name} is rank {arr.ndim}, expected 2")
                        packed, group = quantize_rows(arr, args.quant, args.group)
                        qentries[name] = {
                            "file": qshard, "offset": off, "length": len(packed),
                            "dtype": args.quant, "group": group,
                            "shape": [int(arr.shape[0]), int(arr.shape[1])],
                        }
                        fh.write(packed)
                        padded = align_up(off + len(packed))
                        if padded > off + len(packed):
                            fh.write(b"\0" * (padded - off - len(packed)))
                        off = padded
                        del arr, packed
                os.replace(qtmp, qfinal)
                qdone = {"file_bytes": qfinal.stat().st_size, "entries": qentries}
                manifest["completed_dspark_qweight_shards"][str(group_id)] = qdone
                save_manifest()
            dspark_qweight_map.update(qdone["entries"])
            dspark_qweight_total += sum(int(entry["length"])
                                       for entry in qdone["entries"].values())
            print(f"    DSpark resident shard {group_id + 1}/{len(dspark_groups)}  "
                  f"{(dspark_dense_total + dspark_qweight_total) / 1e9:.2f} GB", flush=True)

        ds_dense_index = {
            "metadata": {"total_size": dspark_dense_total},
            "weight_map": dspark_dense_weight_map,
        }
        tmp = out_dir / "dspark.safetensors.index.json.tmp"
        tmp.write_text(json.dumps(ds_dense_index, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, out_dir / "dspark.safetensors.index.json")
        ds_qindex = {
            "format": 1, "metadata": {"total_size": dspark_qweight_total},
            "weight_map": dspark_qweight_map,
        }
        tmp = out_dir / "dspark.qweights.index.json.tmp"
        tmp.write_text(json.dumps(ds_qindex, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, out_dir / "dspark.qweights.index.json")

        with open(out_dir / "soma.dspark.tmp", "wb") as ix:
            ix.write(MAGIC)
            ix.write(struct.pack("<II", FORMAT_VERSION, 0))
            ix.write(struct.pack("<I", 0))
            ix.write(struct.pack("<IIII", DSPARK_STAGES, n_experts,
                                 DSPARK_STAGES, DTYPE_ID[dt_gate]))
            ix.write(struct.pack("<I", args.group))
            ix.write(struct.pack("<QQ", max(dspark_uniform_len, 0),
                                 dspark_total_payload))
            for shard_id, off, length in dspark_all_index:
                ix.write(struct.pack("<IQI", shard_id, off, length))
        os.replace(out_dir / "soma.dspark.tmp", out_dir / "soma.dspark")

    # This field describes tensors left out of the converted container, not the
    # names they had upstream.  A DSpark augmentation may resume a manifest
    # created by an earlier base-only pass, so overwrite the old omission record
    # only after all auxiliary shards and indexes have committed atomically.
    manifest["omitted_namespaces"] = [] if include_dspark else mtp_namespaces
    manifest["dspark_included"] = include_dspark
    save_manifest()

    dense_index = {"metadata": {"total_size": dense_total}, "weight_map": dense_weight_map}
    dense_tmp = out_dir / "dense.safetensors.index.json.tmp"
    dense_tmp.write_text(json.dumps(dense_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(dense_tmp, out_dir / "dense.safetensors.index.json")
    qindex = {"format": 1, "metadata": {"total_size": qweight_total},
              "weight_map": qweight_map}
    qtmp = out_dir / "dense.qweights.index.json.tmp"
    qtmp.write_text(json.dumps(qindex, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(qtmp, out_dir / "dense.qweights.index.json")

    with open(out_dir / "soma.container.tmp", "wb") as ix:
        ix.write(MAGIC)
        ix.write(struct.pack("<II", FORMAT_VERSION, 0))
        ix.write(struct.pack("<I", 0))
        ix.write(struct.pack("<IIII", n_layers, n_experts, n_layers, DTYPE_ID[dt_gate]))
        ix.write(struct.pack("<I", args.group))
        ix.write(struct.pack("<QQ", max(uniform_len, 0), total_payload))
        for shard, off, length in all_index:
            ix.write(struct.pack("<IQI", shard, off, length))
    os.replace(out_dir / "soma.container.tmp", out_dir / "soma.container")
    shutil.copy2(src / "config.json", out_dir / "config.json")

    # DSpark is an additive, resumable conversion. Do not invalidate a tokenizer
    # that the completed base conversion already compiled merely because the
    # minimal augmentation environment omitted the optional `tokenizers`
    # package. A first conversion (or a genuinely missing artifact) still runs
    # the exact compiler and its round-trip gate.
    tokenizer_artifacts = (
        out_dir / "tokenizer.soma",
        out_dir / "tokenizer_oracle.bin",
        out_dir / "tokenizer_meta.json",
    )
    preserve_tokenizer = include_dspark and all(path.is_file() for path in tokenizer_artifacts)
    tokenizer_status = "compiled" if preserve_tokenizer else "unsupported"
    if not preserve_tokenizer:
        for name in ("tokenizer.soma", "tokenizer_oracle.bin", "tokenizer_meta.json",
                     "tokenizer.unsupported"):
            (out_dir / name).unlink(missing_ok=True)
        try:
            import compile_tokenizer
            tokenizer_status = "compiled" if compile_tokenizer.main(
                ["compile_tokenizer", str(src), "--out", str(out_dir)]) == 0 else "unsupported"
        except Exception as exc:
            (out_dir / "tokenizer.unsupported").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    meta = {
        "container_version": FORMAT_VERSION,
        "source": str(src),
        "source_repo": ("fixture/DeepSeek-V4-Pro-0813" if fixture_mode
                        else "deepseek-ai/DeepSeek-V4-Pro-0813"),
        "source_revision": requested, "config_sha256": config_sha,
        "index_sha256": index_sha,
        "model_type": "deepseek_v4", "n_layers": n_layers,
        "n_moe_layers": n_layers, "layer_kinds": ["moe"] * n_layers,
        "n_experts": n_experts, "n_shards": n_layers,
        "expert_bytes": uniform_len, "total_expert_bytes": total_payload,
        "dtype_gate_up": dt_gate, "dtype_down": dt_down, "dtype_dense": args.quant,
        "group": args.group, "effective_groups": effective_groups,
        "dense_tensors": len(dense_weight_map), "quantized_resident_tensors": len(qweight_map),
        "lossless_resident_bytes": dense_total, "quantized_resident_bytes": qweight_total,
        "dense_sharded": True,
        "align": ALIGN, "tokenizer": tokenizer_status,
        "dspark": ("present" if include_dspark else
                    ("not-present" if fixture_mode else "omitted")),
        "omitted_mtp_tensors": (0 if include_dspark else len(omitted_mtp)),
        "omitted_mtp_namespaces": ([] if include_dspark else mtp_namespaces),
    }
    if include_dspark:
        meta.update({
            "dspark_format": 1,
            "dspark_tensors": len(omitted_mtp),
            "dspark_stages": DSPARK_STAGES,
            "dspark_target_layer_ids": cfg["dspark_target_layer_ids"],
            "dspark_trained_block_size": int(cfg["dspark_block_size"]),
            "dspark_noise_token_id": int(cfg["dspark_noise_token_id"]),
            "dspark_markov_rank": int(cfg["dspark_markov_rank"]),
            "dspark_confidence_head": True,
            "dspark_expert_bytes": dspark_uniform_len,
            "dspark_total_expert_bytes": dspark_total_payload,
            "dspark_lossless_resident_bytes": dspark_dense_total,
            "dspark_quantized_resident_bytes": dspark_qweight_total,
            "dspark_resident_bytes": dspark_dense_total + dspark_qweight_total,
            "dspark_kv_bytes_per_sequence": (
                DSPARK_STAGES * int(cfg["sliding_window"]) * int(cfg["head_dim"]) * 2
                if fixture_mode else DSPARK_KV_BYTES_PER_SEQUENCE
            ),
            "dtype_dspark": args.quant,
        })
    (out_dir / "container_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  OK       {len(all_index)} experts, {total_payload / 1e9:.3f} GB routed, "
          f"{(dense_total + qweight_total) / 1e9:.3f} GB resident, tokenizer {tokenizer_status}" +
          (f", DSpark {dspark_total_payload / 1e9:.3f} GB routed + "
           f"{(dspark_dense_total + dspark_qweight_total) / 1e9:.3f} GB resident"
           if include_dspark else ""))
    return 0
