#!/usr/bin/env python3
"""Fetch a HuggingFace repo into a local directory, for admission stage 1.

The first stage of the pipeline, and the only one that touches the network.
Everything downstream — convert, tokenize, plan — already works on a local
directory, so this stage exists purely to produce one.

WHAT IT DELIBERATELY DOES NOT DOWNLOAD.  A published checkpoint routinely ships
the same weights three times: safetensors, PyTorch .bin, and TF/Flax.  Pulling
all of them triples the transfer and the disk for a model where two thirds will
never be read.  Safetensors wins when present; .bin is the fallback, and the
fallback is announced rather than silent, because a .bin repo means conversion
will be unpickling arbitrary code from that repo.

Progress is reported by WATCHING THE OUTPUT DIRECTORY rather than by hooking the
downloader's progress bars.  A 20 GB shard is one file, so per-file granularity
would report nothing for twenty minutes; and any implementation that puts bytes
on disk is observable this way, which a tqdm hook is not.

Output is line-oriented and parsed by src/control/model_registry.cpp:

    manifest <n_files> <total_bytes>
    progress <bytes_done> <bytes_total>
    resolved <absolute_path>

Anything else is human-readable detail and is forwarded to the operator as-is.

Auth is whatever huggingface_hub already resolves — HF_TOKEN, or a cached login.
This script never reads, prints, or stores a credential.

Usage:
    fetch.py <repo_id>[@revision] --out <dir> [--allow-pickle]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from pathlib import Path

# Repo ids become a directory name, so they are validated rather than trusted:
# `../../etc` is a legal-looking string and an illegal directory.
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")
_REV_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")

# Weights.
_SAFETENSORS = (".safetensors",)
_PICKLE = (".bin", ".pt", ".pth", ".ckpt")
# Formats we never want: other frameworks' copies of the same tensors, and
# already-quantized artifacts for a different engine.
_NEVER = (".h5", ".msgpack", ".tflite", ".onnx", ".gguf", ".ggml")
# Everything conversion and the tokenizer compiler read.
_METADATA = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "tokenizer.model",
    "chat_template.jinja",
}


def emit(line: str) -> None:
    print(line, flush=True)


def split_ref(ref: str) -> tuple[str, str | None]:
    """`org/model@revision` -> (`org/model`, `revision`)."""
    if "@" in ref:
        repo, rev = ref.rsplit("@", 1)
        return repo.strip(), rev.strip() or None
    return ref.strip(), None


def validate(repo_id: str, revision: str | None) -> None:
    if not _REPO_RE.match(repo_id):
        raise SystemExit(f"not a valid repo id: {repo_id!r}")
    if ".." in repo_id:
        raise SystemExit(f"not a valid repo id: {repo_id!r}")
    if revision is not None and (not _REV_RE.match(revision) or ".." in revision):
        raise SystemExit(f"not a valid revision: {revision!r}")


def select(files: list[tuple[str, int]], allow_pickle: bool) -> tuple[list[str], list[str]]:
    """Pick the files worth transferring. Returns (keep, notes)."""
    notes: list[str] = []
    have_safetensors = any(f.endswith(_SAFETENSORS) for f, _ in files)

    keep: list[str] = []
    skipped_frameworks = 0
    skipped_pickle = 0
    for path, _size in files:
        name = Path(path).name
        low = path.lower()

        if low.endswith(_NEVER):
            skipped_frameworks += 1
            continue
        if low.endswith(_SAFETENSORS):
            keep.append(path)
            continue
        if low.endswith(_PICKLE):
            if have_safetensors:
                skipped_pickle += 1
                continue
            if not allow_pickle:
                raise SystemExit(
                    f"{path}: this repo publishes no safetensors, so conversion would have to "
                    "unpickle it — which executes code from the repo. Re-run with "
                    "--allow-pickle if that is intended."
                )
            keep.append(path)
            continue
        # Metadata: named files plus index maps, which are how sharded weights
        # are found at all.
        #
        # An index is only kept if the weights it indexes are. A repo ships
        # `model.safetensors.index.json` AND `pytorch_model.bin.index.json`, and
        # keeping the second after dropping every .bin leaves conversion looking
        # at a map of files that are not there.
        if name.endswith(".index.json"):
            if any(f".{ext.lstrip('.')}." in name for ext in _PICKLE) and have_safetensors:
                skipped_pickle += 1
                continue
            keep.append(path)
            continue
        if name in _METADATA:
            keep.append(path)

    if skipped_frameworks:
        notes.append(f"skipped {skipped_frameworks} non-PyTorch weight files")
    if skipped_pickle:
        notes.append(f"skipped {skipped_pickle} .bin duplicates (safetensors present)")
    return keep, notes


def directory_bytes(root: Path) -> int:
    """Every byte under `root`, partial downloads included.

    Partial files are the point — a transfer in progress lives in a temp file
    under the output directory, so counting only finished files would report
    zero for the entire download of a single-shard model.
    """
    total = 0
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, n))
            except OSError:
                # Raced with a rename; it will be counted on the next tick.
                pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="HuggingFace repo id, optionally @revision")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-pickle", action="store_true",
                    help="permit .bin weights when a repo publishes no safetensors")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    repo_id, revision = split_ref(args.repo)
    validate(repo_id, revision)

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        emit("huggingface_hub is not installed; see tools/admission/requirements.txt")
        return 2

    api = HfApi()
    try:
        info = api.repo_info(repo_id, revision=revision, files_metadata=True)
    except Exception as e:  # noqa: BLE001 — surfaced to the operator verbatim
        emit(f"cannot read {repo_id}: {e}")
        return 3

    files = [(s.rfilename, s.size or 0) for s in (info.siblings or [])]
    if not files:
        emit(f"{repo_id} lists no files")
        return 3

    keep, notes = select(files, args.allow_pickle)
    if not keep:
        emit(f"{repo_id} has no weights this pipeline can read")
        return 3
    for n in notes:
        emit(n)

    sizes = dict(files)
    total = sum(sizes.get(p, 0) for p in keep)
    emit(f"manifest {len(keep)} {total}")
    emit(f"{repo_id}{'@' + revision if revision else ''}: {len(keep)} files, {total / 1e9:.2f} GB")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # The watcher runs while snapshot_download blocks. Daemon so a fatal error in
    # the main thread cannot leave the process alive on its account.
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(args.interval):
            emit(f"progress {directory_bytes(out)} {total}")

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    try:
        snapshot_download(
            repo_id,
            revision=revision,
            local_dir=str(out),
            allow_patterns=keep,
            # Already-present files are skipped by hash, so a re-run after a
            # network failure resumes rather than restarts.
            max_workers=4,
        )
    except Exception as e:  # noqa: BLE001
        stop.set()
        emit(f"download failed: {e}")
        return 4
    finally:
        stop.set()

    emit(f"progress {directory_bytes(out)} {total}")
    emit(f"resolved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
