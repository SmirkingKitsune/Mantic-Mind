#!/usr/bin/env python3
"""The MiniMax-M3 source dialect must be a pure RENAMING, and nothing else.

`convert.py`'s SOURCE_DIALECTS rewrites a production checkpoint's tensor names
into the ones Soma's loader binds: the text stack out from under
`language_model.`, `block_sparse_moe` to `mlp`, a selection bias off the block
and onto the gate, and four `index_*` tensors into an `indexer.*` block.

Nothing else in the repository can check that map. The tiny fixture carries the
names `transformers` emits, so the whole conformance ladder runs without ever
touching the rewrite; the 59 production shards that would exercise it are most of
a terabyte. A wrong entry is therefore invisible until admission drops a tensor
on a real checkpoint -- and a DROPPED tensor is the failure this codebase keeps
paying for, because the model still loads.

So this manufactures the missing input. It rewrites the committed fixture INTO
the production dialect -- renaming, splitting the fused projections that
checkpoint ships per-projection, prefixing the stack, and bolting on a token
vision tower -- converts both, and asserts the two containers describe the same
weights.

What that proves, precisely:

  * every production name the rewrite claims maps onto a name the loader binds,
    because the completeness check inside convert.py refuses anything unaccounted
    for and this feeds it a checkpoint made only of production names;
  * the vision tower is dropped by a STATED rule rather than by a wildcard, since
    a tower module the `drop` list does not name would land in `unclaimed`;
  * the expert payload is byte-identical across the two dialects, so `w1/w3/w2`
    really is `gate/up/down` in that order and not some permutation of it;
  * every dense tensor survives with its bytes intact, under the fused-or-split
    correspondence the two dialects genuinely differ by.

Exits 77 (ctest's "skipped") when numpy or safetensors is missing, so an
incomplete environment reports as skipped instead of passing silently.

Usage:  python tools/ci/check_minimax_dialect.py <repo_root> [work_dir]
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURE = "MiniMax-M3-Tiny"

# soma name -> production name, for the suffixes that differ. Deliberately a
# SECOND transcription of convert.py's table: a checker that imported the map it
# is checking would agree with any mistake in it.
SUFFIX = {
    "mlp.gate.weight": "block_sparse_moe.gate.weight",
    "mlp.gate.e_score_correction_bias": "block_sparse_moe.e_score_correction_bias",
    "self_attn.indexer.q_proj.weight": "self_attn.index_q_proj.weight",
    "self_attn.indexer.k_proj.weight": "self_attn.index_k_proj.weight",
    "self_attn.indexer.q_norm.weight": "self_attn.index_q_norm.weight",
    "self_attn.indexer.k_norm.weight": "self_attn.index_k_norm.weight",
}


def emit_production_checkpoint(fixture: Path, dest: Path) -> None:
    """Rewrite the fixture into the dialect the 59 production shards carry."""
    import numpy as np
    from safetensors.numpy import load_file, save_file

    src = load_file(str(fixture / "model.safetensors"))
    cfg = json.loads((fixture / "config.json").read_text(encoding="utf-8"))
    inter = int(cfg["intermediate_size"])
    d_model = int(cfg["hidden_size"])
    out: dict[str, "np.ndarray"] = {}

    for name, tensor in src.items():
        # Everything below the layer prefix, so the suffix rules can be exact.
        head, _, suffix = name.partition("layers.")
        if not suffix:
            out["language_model." + name] = tensor
            continue
        layer, _, tail = suffix.partition(".")
        base = f"language_model.{head}layers.{layer}."

        if tail in SUFFIX:
            out[base + SUFFIX[tail]] = tensor
            continue

        # The routed experts: one stacked rank-3 tensor upstream, one tensor per
        # expert per projection in production, spelled Mixtral's way.
        if tail == "mlp.experts.gate_up_proj":
            for e in range(tensor.shape[0]):
                out[f"{base}block_sparse_moe.experts.{e}.w1.weight"] = tensor[e][:inter]
                out[f"{base}block_sparse_moe.experts.{e}.w3.weight"] = tensor[e][inter:]
            continue
        if tail == "mlp.experts.down_proj":
            for e in range(tensor.shape[0]):
                out[f"{base}block_sparse_moe.experts.{e}.w2.weight"] = tensor[e]
            continue

        # The shared expert and the dense-layer MLP: one fused `[2*inter, hidden]`
        # projection upstream, two separate ones in production.
        if tail == "mlp.shared_experts.gate_up_proj.weight":
            half = tensor.shape[0] // 2
            out[f"{base}block_sparse_moe.shared_experts.gate_proj.weight"] = tensor[:half]
            out[f"{base}block_sparse_moe.shared_experts.up_proj.weight"] = tensor[half:]
            continue
        if tail == "mlp.shared_experts.down_proj.weight":
            out[f"{base}block_sparse_moe.shared_experts.down_proj.weight"] = tensor
            continue
        if tail == "mlp.gate_up_proj.weight":
            half = tensor.shape[0] // 2
            out[f"{base}mlp.gate_proj.weight"] = tensor[:half]
            out[f"{base}mlp.up_proj.weight"] = tensor[half:]
            continue

        out[base + tail] = tensor

    # A token vision tower, so the `drop` rule is exercised rather than merely
    # declared. Shapes are irrelevant -- what is under test is that these names
    # are dropped by a stated rule instead of landing in `unclaimed`.
    for name in ("vision_tower.vision_model.embeddings.patch_embedding.weight",
                 "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
                 "multi_modal_projector.linear_1.weight",
                 "patch_merge_mlp.linear_1.weight"):
        out[name] = np.zeros((4, d_model), dtype=np.float32)

    dest.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dest / "model.safetensors"))

    # The WRAPPER config: the language model nested, a vision tower declared.
    wrapper = {
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "model_type": "minimax_m3_vl",
        "text_config": cfg,
        "vision_config": {
            "hidden_size": 1280, "num_hidden_layers": 32,
            "num_attention_heads": 16, "patch_size": 14, "image_size": 2016,
            "model_type": "clip_vision_model",
        },
        "image_token_index": 200025,
        "dtype": "float32",
        "torch_dtype": "float32",
    }
    (dest / "config.json").write_text(json.dumps(wrapper, indent=2) + "\n", encoding="utf-8")
    for extra in ("tokenizer.json", "tokenizer_config.json"):
        if (fixture / extra).exists():
            shutil.copy2(fixture / extra, dest / extra)


def convert(repo: Path, model_dir: Path, out: Path) -> None:
    # cwd is tools/admission because convert.py imports compile_tokenizer as a
    # sibling module; the script path is absolute so the two cannot fight.
    cmd = [sys.executable, str((repo / "tools/admission/convert.py").resolve()),
           str(model_dir.resolve()),
           "--out", str(out.resolve()), "--quant", "q8_0", "--group", "32"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str((repo / "tools/admission").resolve()))
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"convert.py failed on {model_dir.name} (exit {r.returncode})")


def main(argv: list[str]) -> int:
    repo = Path(argv[1]) if len(argv) > 1 else Path(".")
    fixture = repo / "tests/fixtures/tiny" / FIXTURE
    if not fixture.is_dir():
        print(f"SKIP: {fixture} not found")
        return 77
    try:
        import numpy as np  # noqa: F401
        from safetensors.numpy import load_file
    except ImportError as e:
        print(f"SKIP: {e}")
        return 77

    work = Path(argv[2]) if len(argv) > 2 else Path(tempfile.mkdtemp(prefix="mm_minimax_"))
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    production = work / "production"
    emit_production_checkpoint(fixture, production)

    native_out, prod_out = work / "c_native", work / "c_production"
    convert(repo, fixture, native_out)
    convert(repo, production, prod_out)

    failures: list[str] = []

    # ── the expert payload, byte for byte ────────────────────────────────────
    #
    # The strongest single assertion here. Both conversions read the same weights
    # through different names and pack them into the same layout, so any
    # disagreement about WHICH tensor is gate, up or down shows up as different
    # bytes. `w1/w3/w2 -> gate/up/down` is a permutation that is easy to get
    # wrong and impossible to notice downstream.
    for blob in sorted(p.name for p in native_out.glob("experts-*.bin")):
        a, b = (native_out / blob).read_bytes(), (prod_out / blob).read_bytes()
        if a != b:
            failures.append(f"expert payload {blob} differs: {len(a)} vs {len(b)} bytes")
    if not list(native_out.glob("experts-*.bin")):
        failures.append("no expert payload was written; the conversion streamed nothing")

    # ── the dense half ───────────────────────────────────────────────────────
    #
    # Not name-for-name identical, and it should not be: the two dialects
    # genuinely differ in whether gate and up arrive fused. Compared under
    # exactly that correspondence, so a container that merely LOST a tensor
    # cannot pass by having fewer of them.
    native = load_file(str(native_out / "dense.safetensors"))
    prod = load_file(str(prod_out / "dense.safetensors"))

    import numpy as np
    for name, tensor in sorted(native.items()):
        if name.endswith("gate_up_proj.weight"):
            half = tensor.shape[0] // 2
            for part, want in (("gate_proj", tensor[:half]), ("up_proj", tensor[half:])):
                other = name.replace("gate_up_proj", part)
                if other not in prod:
                    failures.append(f"production container is missing {other}")
                elif not np.array_equal(prod[other], want):
                    failures.append(f"{other} differs from the fused half it came from")
            continue
        if name not in prod:
            failures.append(f"production container is missing {name}")
        elif not np.array_equal(prod[name], tensor):
            failures.append(f"{name} differs between the two dialects")

    # And the other direction, so a rewrite that INVENTED a tensor is caught too.
    expected = set()
    for name in native:
        if name.endswith("gate_up_proj.weight"):
            expected.add(name.replace("gate_up_proj", "gate_proj"))
            expected.add(name.replace("gate_up_proj", "up_proj"))
        else:
            expected.add(name)
    for extra in sorted(set(prod) - expected):
        failures.append(f"production container carries an unexpected tensor: {extra}")

    # ── the indexer actually survived ────────────────────────────────────────
    #
    # Named explicitly rather than left to the set comparison above: these four
    # are the whole reason the dialect exists, and "both containers are missing
    # them" would satisfy every assertion so far.
    indexed = [n for n in prod if ".self_attn.indexer." in n]
    if not indexed:
        failures.append("no indexer tensor reached the container; "
                        "the block-sparse layers would bind nothing")

    # ── the vision tower is declared, not served ─────────────────────────────
    cfg = json.loads((prod_out / "config.json").read_text(encoding="utf-8"))
    if "vision_config" not in cfg:
        failures.append("container config.json dropped vision_config; the plan would "
                        "report a text model that is not one")
    if any(n.startswith(("vision_tower.", "multi_modal_projector.", "patch_merge_mlp."))
           for n in prod):
        failures.append("a vision tensor reached the container")

    print(f"  fixture       : {FIXTURE}")
    print(f"  dense tensors : {len(native)} native / {len(prod)} production")
    print(f"  indexer       : {len(indexed)} tensor(s)")
    print(f"  expert payload: {len(list(native_out.glob('experts-*.bin')))} shard(s), identical")

    if failures:
        print(f"\nFAIL: {len(failures)} problem(s) with the MiniMax-M3 source dialect")
        for f in failures:
            print(f"  {f}")
        return 1
    print("check_minimax_dialect: OK - the production dialect is a pure renaming.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
