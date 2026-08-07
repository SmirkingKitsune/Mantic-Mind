#!/usr/bin/env python3
"""Admission fetch — which files a repo transfer actually pulls.

The selection rule is the whole economic argument for this stage: a published
checkpoint routinely ships the same weights three times, so a fetch that takes
everything triples the transfer and the disk for bytes nothing will read. It is
also the safety boundary — a repo with no safetensors can only be converted by
unpickling it, which executes code from the repo.

Runs with no network and no huggingface_hub: select() is pure, which is exactly
why it is a separate function.

Usage: check_fetch_selection.py [repo_root]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

failures = 0


def check(ok: bool, what: str, detail: str = "") -> None:
    global failures
    print(f"   {what:<62}{'OK' if ok else 'FAIL'}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures += 1


def load_fetch(root: Path):
    path = root / "tools" / "admission" / "fetch.py"
    spec = importlib.util.spec_from_file_location("soma_fetch", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    fetch = load_fetch(root)

    # A realistic sharded repo: safetensors AND the pytorch duplicates AND a
    # TF/Flax copy, which is what the big published MoE checkpoints look like.
    repo = [
        ("config.json", 1_200),
        ("generation_config.json", 200),
        ("tokenizer.json", 11_000_000),
        ("tokenizer_config.json", 7_000),
        ("model.safetensors.index.json", 90_000),
        ("model-00001-of-00002.safetensors", 5_000_000_000),
        ("model-00002-of-00002.safetensors", 4_000_000_000),
        ("pytorch_model-00001-of-00002.bin", 5_000_000_000),
        ("pytorch_model-00002-of-00002.bin", 4_000_000_000),
        ("pytorch_model.bin.index.json", 90_000),
        ("tf_model.h5", 9_000_000_000),
        ("flax_model.msgpack", 9_000_000_000),
        ("model.gguf", 5_000_000_000),
        ("README.md", 4_000),
        (".gitattributes", 300),
    ]

    print("\n1. a repo that ships the same weights three times")
    keep, notes = fetch.select(repo, allow_pickle=False)
    kept = set(keep)

    check(all(f"model-0000{i}-of-00002.safetensors" in kept for i in (1, 2)),
          "both safetensors shards are taken")
    check(not any(k.endswith(".bin") for k in kept),
          "and no .bin duplicate", f"{sum(1 for f, _ in repo if f.endswith('.bin'))} available")
    check(not any(k.endswith((".h5", ".msgpack", ".gguf")) for k in kept),
          "no TF, Flax or GGUF copy")
    check("model.safetensors.index.json" in kept,
          "the safetensors index IS taken - the shards cannot be found without it")
    check("pytorch_model.bin.index.json" not in kept,
          "but not the index for the shards we skipped",
          "a map of files that are not there is worse than no map")
    check("config.json" in kept and "tokenizer.json" in kept,
          "config and tokenizer come along")
    check("README.md" not in kept and ".gitattributes" not in kept,
          "repo furniture does not")

    total = sum(s for f, s in repo if f in kept)
    everything = sum(s for _, s in repo)
    check(total < everything / 2, "the transfer is less than half the repo",
          f"{total / 1024**3:.1f} GiB of {everything / 1024**3:.1f} GiB")
    check(len(notes) == 2, "and both skips are reported, not silent", "; ".join(notes))

    # ── 2. the pickle boundary ───────────────────────────────────────────────
    #
    # A repo with no safetensors cannot be converted without unpickling, which
    # runs code from the repo. Refusing is the default; the flag is the operator
    # saying they meant it.
    print("\n2. a repo with no safetensors")
    pickle_only = [
        ("config.json", 1_200),
        ("pytorch_model.bin", 8_000_000_000),
    ]
    refused = False
    try:
        fetch.select(pickle_only, allow_pickle=False)
    except SystemExit as e:
        refused = "unpickle" in str(e)
    check(refused, "is refused by default, and the message says why")

    keep2, _ = fetch.select(pickle_only, allow_pickle=True)
    check("pytorch_model.bin" in keep2, "and taken when --allow-pickle is given")

    # ── 3. repo id validation ────────────────────────────────────────────────
    #
    # The id becomes a directory name. Mirrors valid_repo_id() in
    # src/control/model_registry.cpp — two implementations of one rule, so both
    # are tested rather than trusting that they agree.
    print("\n3. a repo id becomes a directory name")
    for bad in ["../../etc/passwd", "org/../../x", "a/b/c", "/abs/path", "org/model@../evil"]:
        rejected = False
        try:
            repo_id, rev = fetch.split_ref(bad)
            fetch.validate(repo_id, rev)
        except SystemExit:
            rejected = True
        check(rejected, f"rejected: {bad}")
    for good in ["gpt2", "Qwen/Qwen3-30B-A3B", "org/model@refs/pr/1", "a.b/c-d_e"]:
        accepted = True
        try:
            repo_id, rev = fetch.split_ref(good)
            fetch.validate(repo_id, rev)
        except SystemExit as e:
            accepted = False
            print(f"      {e}")
        check(accepted, f"accepted: {good}")

    print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
