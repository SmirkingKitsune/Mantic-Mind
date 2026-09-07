#!/usr/bin/env python3
"""The Kimi-K3 source dialect must be a pure RENAMING, and nothing else.

The sibling of `check_minimax_dialect.py`, for the second wrapper `convert.py`
knows how to take a text stack out of. Same hazard, same method, one instructive
difference.

WHAT MAKES KIMI DIFFERENT, and why it still needs a check. MiniMax-M3's
production shards rename a great deal — `block_sparse_moe` for `mlp`, a bias off
the block onto the gate, four `index_*` tensors into an `indexer.*` block.
Kimi-K3 renames NOTHING. `KimiK3ForConditionalGeneration.__init__` builds exactly
three submodules — `vision_tower`, `mm_projector`, and a `language_model` that is
a `KimiLinearForCausalLM`, which is `tests/fixtures/tiny/Kimi-Linear-Tiny`
verbatim. So the whole dialect is a prefix and a drop list, and its `suffixes`
map is empty on purpose.

An empty map is exactly the kind of entry that looks like it cannot be wrong, so
it is worth stating what this still catches:

  * `moe_block` DELIBERATELY ABSENT. The key does not describe the checkpoint —
    in `to_soma_name` it MEANS "rewrite this to `mlp`", which is right for
    MiniMax and catastrophic for Kimi, whose loader binds `block_sparse_moe`
    itself. The first draft of the entry declared it and renamed every expert to
    a name nothing binds. This test fails on that.
  * the vision tower dropped by a STATED rule, not a wildcard — a tower module
    the `drop` list does not name lands in `unclaimed` and refuses.
  * the twenty tensor roles the Kimi family needed and `convert.py` did not have.
    That gap was not a wrapper problem at all: it refused the UNWRAPPED fixture
    too, so no Kimi container could be produced by any route, while
    `arch_ir.cpp`, `kda.cpp` and `f32_model.cpp` all bound the tensors happily.
    Both halves of this test would pass trivially if that regressed to a refusal,
    so the native conversion is asserted to succeed first.

Exits 77 (ctest's "skipped") when numpy or safetensors is missing, so an
incomplete environment reports as skipped instead of passing silently.

Usage:  python tools/ci/check_kimi_dialect.py <repo_root> [work_dir]
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURE = "Kimi-Linear-Tiny"

# The three submodules the wrapper builds beside the language model, from
# `modeling_kimi_k3.py`. A SECOND transcription of convert.py's `drop` list, for
# the reason the MiniMax checker gives: a checker that imported the map it is
# checking would agree with any mistake in it.
DROP_PREFIXES = ("vision_tower.", "mm_projector.")


def emit_wrapper_checkpoint(fixture: Path, dest: Path) -> None:
    """Rewrite the fixture into the dialect the 96 production shards carry."""
    import numpy as np
    from safetensors.numpy import load_file, save_file

    src = load_file(str(fixture / "model.safetensors"))
    cfg = json.loads((fixture / "config.json").read_text(encoding="utf-8"))
    d_model = int(cfg["hidden_size"])

    # The entire rewrite. If this loop ever needs a second rule, the dialect's
    # empty `suffixes` map has stopped being true.
    out = {"language_model." + name: tensor for name, tensor in src.items()}

    # A token vision tower, so the `drop` rule is exercised rather than merely
    # declared. Shapes are irrelevant — what is under test is that these names
    # are dropped by a stated rule instead of landing in `unclaimed`.
    for name in ("vision_tower.patch_embed.proj.weight",
                 "vision_tower.encoder.blocks.0.attn.qkv.weight",
                 "mm_projector.linear_1.weight"):
        out[name] = np.zeros((4, d_model), dtype=np.float32)

    dest.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dest / "model.safetensors"))

    # The WRAPPER config: the language model nested, a vision tower declared.
    wrapper = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "text_config": cfg,
        "vision_config": {
            "model_type": "moonvit3d",
            "vt_hidden_size": 1152, "vt_num_hidden_layers": 27,
            "vt_num_attention_heads": 16, "patch_size": 14,
            "mm_projector_type": "mlp", "mm_hidden_size": 1152,
        },
        "media_placeholder_token_id": 163605,
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

    work = Path(argv[2]) if len(argv) > 2 else Path(tempfile.mkdtemp(prefix="mm_kimi_"))
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    wrapped = work / "wrapper"
    emit_wrapper_checkpoint(fixture, wrapped)

    native_out, wrap_out = work / "c_native", work / "c_wrapper"
    # Native FIRST, and its failure is this script's failure. Everything below
    # compares two containers, and two identical refusals would compare equal.
    convert(repo, fixture, native_out)
    convert(repo, wrapped, wrap_out)

    failures: list[str] = []

    # ── the expert payload, byte for byte ────────────────────────────────────
    #
    # The strongest single assertion here. Both conversions read the same weights
    # through different names and pack them into the same layout, so any
    # disagreement about WHICH tensor is gate, up or down shows up as different
    # bytes — and `w1/w3/w2 -> gate/up/down` is a permutation that is easy to get
    # wrong and impossible to notice downstream.
    payload = sorted(p.name for p in native_out.glob("experts-*.bin"))
    if not payload:
        failures.append("no expert payload was written; the conversion streamed nothing")
    for blob in payload:
        a, b = (native_out / blob).read_bytes(), (wrap_out / blob).read_bytes()
        if a != b:
            failures.append(f"expert payload {blob} differs: {len(a)} vs {len(b)} bytes")

    # ── the dense half, name for name ────────────────────────────────────────
    #
    # Identical here, unlike MiniMax: this dialect renames nothing below the
    # prefix, so anything but an exact match is the bug.
    native = load_file(str(native_out / "dense.safetensors"))
    wrap = load_file(str(wrap_out / "dense.safetensors"))

    import numpy as np
    for name, tensor in sorted(native.items()):
        if name not in wrap:
            failures.append(f"wrapper container is missing {name}")
        elif not np.array_equal(wrap[name], tensor):
            failures.append(f"{name} differs between the two dialects")
    for extra in sorted(set(wrap) - set(native)):
        failures.append(f"wrapper container carries an unexpected tensor: {extra}")

    # ── the block names survived the prefix strip ────────────────────────────
    #
    # Named explicitly rather than left to the set comparison, because it is the
    # exact mistake the first draft made: declaring `moe_block` in the dialect
    # renames `block_sparse_moe.*` to `mlp.*`, and BOTH containers would then be
    # wrong in the same way if the native path were ever taught the same rename.
    if not any(".block_sparse_moe." in n for n in wrap):
        failures.append("no block_sparse_moe tensor reached the container; the dialect "
                        "renamed the MoE block Soma binds")

    # ── the hybrid's two halves both survived ────────────────────────────────
    #
    # A Kimi stack is MLA and KDA in one model. Asserted by name because a
    # dialect that dropped either half would still produce a container that
    # loads — the failure this whole completeness check exists for.
    if not any(".self_attn.A_log" in n for n in wrap):
        failures.append("no KDA tensor reached the container; the linear layers "
                        "would bind nothing")
    if not any(".self_attn.kv_a_proj_with_mqa." in n for n in wrap):
        failures.append("no MLA tensor reached the container; the full-attention "
                        "layers would bind nothing")
    if not any(".routed_expert_up_proj." in n for n in wrap):
        failures.append("no latent-MoE projection reached the container; f32_model "
                        "refuses a layer that declares one and has none")

    # ── the vision tower is declared, not served ─────────────────────────────
    cfg = json.loads((wrap_out / "config.json").read_text(encoding="utf-8"))
    if "vision_config" not in cfg:
        failures.append("container config.json dropped vision_config; the plan would "
                        "report a text model that is not one")
    if any(n.startswith(DROP_PREFIXES) for n in wrap):
        failures.append("a vision tensor reached the container")

    print(f"  fixture       : {FIXTURE}")
    print(f"  dense tensors : {len(native)} native / {len(wrap)} wrapped")
    print(f"  expert payload: {len(payload)} shard(s), identical")

    if failures:
        print(f"\nFAIL: {len(failures)} problem(s) with the Kimi-K3 source dialect")
        for f in failures:
            print(f"  {f}")
        return 1
    print("check_kimi_dialect: OK - the wrapper dialect is a pure prefix strip.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
