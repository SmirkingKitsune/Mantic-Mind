#!/usr/bin/env python3
"""Run the pinned DeepSeek-V4-Pro-0813 full-checkpoint acceptance smoke.

This is intentionally a post-conversion validator, not another converter.  It
checks the immutable source/container identity, records the headers-only 1M
plan, then performs the same deterministic request against two independently
started Soma processes.  The artifact is written atomically even on failure so
an interrupted or failed acceptance run remains diagnosable.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
import re
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODEL_REPO = "deepseek-ai/DeepSeek-V4-Pro-0813"
PINNED_REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"
CONFIG_SHA256 = "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0"
INDEX_SHA256 = "2de2ac1e43134f8b03bf6156067715b7c3c73b1a507329e606023c601a56d30a"
N_LAYERS = 61
N_EXPERTS = 384
TOP_K = 6
MAX_CONTEXT = 1_048_576
# The repository's 66 SafeTensors files include 16,741,976 bytes of headers.
# Pin both quantities: file bytes prove the complete download, while payload
# bytes prove the tensor data covered by the published index.
SOURCE_SHARD_FILE_BYTES = 892_744_322_880
SOURCE_TENSOR_PAYLOAD_BYTES = 892_727_580_904
OMITTED_MTP_NAMESPACES = ["mtp.0", "mtp.1", "mtp.2"]
DSPARK_TENSORS = 7009
DSPARK_STAGES = 3
DSPARK_TARGET_LAYERS = [58, 59, 60]
DSPARK_TRAINED_BLOCK = 5
DSPARK_NOISE_TOKEN = 128799
DSPARK_MARKOV_RANK = 512
DSPARK_EXPERT_BYTES = 41_975_808
DSPARK_ROUTED_BYTES = 48_356_130_816
DSPARK_RESIDENT_BYTES = 828_988_820
DSPARK_KV_BYTES = 393_216


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def safetensors_payload_bytes(path: Path) -> int:
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated SafeTensors header: {path}")
        header_bytes = struct.unpack("<Q", raw)[0]
        header = json.loads(fh.read(header_bytes))
    return sum(
        int(entry["data_offsets"][1]) - int(entry["data_offsets"][0])
        for name, entry in header.items()
        if name != "__metadata__"
    )


def source_revision(source: Path) -> str:
    metadata = source / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if not metadata.is_file():
        raise ValueError(f"missing Hugging Face revision metadata: {metadata}")
    lines = metadata.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty Hugging Face revision metadata: {metadata}")
    return lines[0].strip()


def available_ram_bytes() -> int:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        state = MemoryStatusEx()
        state.dwLength = ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(state.ullAvailPhys)

    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return int(pages * page_size)


def choose_port(requested: int) -> int:
    if requested:
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_json(command: list[str], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command!r}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command did not return JSON: {completed.stdout[-4000:]}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("command returned JSON that is not an object")
    return value, elapsed


def http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError(f"{url} returned non-object JSON")
    return value


def http_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def has_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(has_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(has_nonfinite(item) for item in value)
    return False


def wait_for_health(process: subprocess.Popen[Any], url: str, timeout: float) -> float:
    started = time.perf_counter()
    deadline = started + timeout
    last_error = "server did not answer"
    while time.perf_counter() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"Soma exited before health readiness with code {code}")
        try:
            health = http_json(url, None, timeout=2.0)
            if health.get("status") in ("ok", "ready"):
                return time.perf_counter() - started
            last_error = f"unexpected health response: {health}"
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"Soma health timeout after {timeout:.0f}s: {last_error}")


def smoke_once(
    soma: Path,
    container: Path,
    port: int,
    ram_budget: int,
    startup_timeout: float,
    generation_timeout: float,
    log_dir: Path,
    ordinal: int,
    require_dspark: bool,
) -> dict[str, Any]:
    command = [
        str(soma), "serve", "--model-dir", str(container),
        "--served-name", MODEL_REPO, "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", "4096", "--kv-slots", "1", "--max-batch", "1",
        "--ram-budget", str(ram_budget),
        "--generation-timeout", str(max(1, math.ceil(generation_timeout))),
    ]
    if require_dspark:
        command.extend(["--speculative", "dspark", "--speculative-tokens", "7"])
    log_path = log_dir / f"cold-launch-{ordinal}.log"
    launched_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    total_started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        crashed = False
        try:
            startup_s = wait_for_health(
                process, f"http://127.0.0.1:{port}/health", startup_timeout
            )
            payload = {
                "model": MODEL_REPO,
                "messages": [{"role": "user", "content": "Reply with exactly OK"}],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 32,
                "stream": False,
                "soma_return_token_ids": True,
            }
            generation_started = time.perf_counter()
            response = http_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                payload,
                generation_timeout,
            )
            generation_s = time.perf_counter() - generation_started
            crashed = process.poll() is not None
            if crashed:
                raise RuntimeError(f"Soma exited during generation with code {process.returncode}")
            if has_nonfinite(response):
                raise RuntimeError("completion response contains NaN or infinity")
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise RuntimeError(f"invalid completion choices: {choices!r}")
            message = choices[0].get("message", {})
            content = message.get("content")
            token_ids = response.get("soma_token_ids")
            if not isinstance(content, str) or content != "OK":
                raise RuntimeError(f"model did not reply exactly OK: {content!r}")
            if not isinstance(token_ids, list) or not token_ids or not all(
                isinstance(token, int) and token >= 0 for token in token_ids
            ):
                raise RuntimeError(f"missing or invalid soma_token_ids: {token_ids!r}")
            telemetry_dump = http_text(
                f"http://127.0.0.1:{port}/internal/telemetry/dump", timeout=10.0
            )
            speculative_match = re.search(r"\bspec=(\d+)/(\d+)\b", telemetry_dump)
            if require_dspark and (
                    speculative_match is None or int(speculative_match.group(2)) == 0):
                raise RuntimeError(
                    "DSpark was selected but telemetry recorded no speculative draft tokens"
                )
            return {
                "ordinal": ordinal,
                "launched_utc": launched_utc,
                "command": command,
                "startup_seconds": startup_s,
                "generation_seconds": generation_s,
                "total_seconds": time.perf_counter() - total_started,
                "token_ids": token_ids,
                "content": content,
                "finish_reason": choices[0].get("finish_reason"),
                "response": response,
                "telemetry_dump": telemetry_dump,
                "speculative_accepted_tokens": (
                    int(speculative_match.group(1)) if speculative_match else 0
                ),
                "speculative_draft_tokens": (
                    int(speculative_match.group(2)) if speculative_match else 0
                ),
                "alive_after_response": True,
            }
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
            log.flush()
            # The process is deliberately terminated after a successful request;
            # its post-termination return code is therefore not a crash signal.
            _ = crashed


def log_excerpt(path: Path, limit: int = 64 * 1024) -> str:
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def validate_static(source: Path, container: Path, require_dspark: bool) -> dict[str, Any]:
    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    revision = source_revision(source)
    config_hash = sha256(config_path)
    index_hash = sha256(index_path)
    if revision != PINNED_REVISION:
        raise ValueError(f"source revision {revision} != pinned {PINNED_REVISION}")
    if config_hash != CONFIG_SHA256 or index_hash != INDEX_SHA256:
        raise ValueError(
            f"source identity mismatch: config={config_hash}, index={index_hash}"
        )

    shards = []
    hf_download = source / ".cache" / "huggingface" / "download"
    for number in range(1, 67):
        path = source / f"model-{number:05d}-of-00066.safetensors"
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty source shard: {path.name}")
        shard_metadata = hf_download / f"{path.name}.metadata"
        lines = shard_metadata.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or lines[0].strip() != PINNED_REVISION:
            raise ValueError(f"missing or wrong pinned metadata for {path.name}")
        shards.append({
            "name": path.name,
            "bytes": path.stat().st_size,
            "huggingface_blob_hash": lines[1].strip(),
        })
    source_shard_file_bytes = sum(item["bytes"] for item in shards)
    if source_shard_file_bytes != SOURCE_SHARD_FILE_BYTES:
        raise ValueError(
            f"source shard file bytes {source_shard_file_bytes} != pinned "
            f"{SOURCE_SHARD_FILE_BYTES}"
        )
    source_tensor_payload_bytes = sum(
        safetensors_payload_bytes(source / item["name"]) for item in shards
    )
    if source_tensor_payload_bytes != SOURCE_TENSOR_PAYLOAD_BYTES:
        raise ValueError(
            f"source tensor payload bytes {source_tensor_payload_bytes} != pinned "
            f"{SOURCE_TENSOR_PAYLOAD_BYTES}"
        )

    config = read_json(config_path)
    expected_config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": N_LAYERS,
        "n_routed_experts": N_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "max_position_embeddings": MAX_CONTEXT,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"config {key}={config.get(key)!r}, expected {expected!r}")

    meta_path = container / "container_meta.json"
    manifest_path = container / "conversion-manifest.json"
    meta = read_json(meta_path)
    manifest = read_json(manifest_path)
    required_meta = {
        "source_repo": MODEL_REPO,
        "source_revision": PINNED_REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "model_type": "deepseek_v4",
        "n_layers": N_LAYERS,
        "n_experts": N_EXPERTS,
        "dtype_gate_up": "q4_g",
        "dtype_down": "q6_g",
        "dtype_dense": "q4_g",
        "group": 128,
    }
    if require_dspark:
        required_meta.update({
            "dspark": "present",
            "dspark_format": 1,
            "dspark_tensors": DSPARK_TENSORS,
            "dspark_stages": DSPARK_STAGES,
            "dspark_target_layer_ids": DSPARK_TARGET_LAYERS,
            "dspark_trained_block_size": DSPARK_TRAINED_BLOCK,
            "dspark_noise_token_id": DSPARK_NOISE_TOKEN,
            "dspark_markov_rank": DSPARK_MARKOV_RANK,
            "dspark_confidence_head": True,
            "dspark_expert_bytes": DSPARK_EXPERT_BYTES,
            "dspark_total_expert_bytes": DSPARK_ROUTED_BYTES,
            "dspark_resident_bytes": DSPARK_RESIDENT_BYTES,
            "dspark_kv_bytes_per_sequence": DSPARK_KV_BYTES,
            "dtype_dspark": "q4_g",
            "omitted_mtp_namespaces": [],
        })
    else:
        required_meta.update({
            "dspark": "omitted",
            "omitted_mtp_namespaces": OMITTED_MTP_NAMESPACES,
        })
    for key, expected in required_meta.items():
        if meta.get(key) != expected:
            raise ValueError(f"container {key}={meta.get(key)!r}, expected {expected!r}")
    expected_omitted = 0 if require_dspark else DSPARK_TENSORS
    if meta.get("omitted_mtp_tensors") != expected_omitted:
        raise ValueError(
            f"container omitted_mtp_tensors={meta.get('omitted_mtp_tensors')!r}, "
            f"expected {expected_omitted}"
        )

    completed_layers = manifest.get("completed_expert_layers", {})
    completed_dense = manifest.get("completed_dense_shards", {})
    completed_qweights = manifest.get("completed_qweight_shards", {})
    required_manifest = {
        "source_revision": PINNED_REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "quant_gate_up": "q4_g",
        "quant_down": "q6_g",
        "group": 128,
        "omitted_namespaces": [] if require_dspark else OMITTED_MTP_NAMESPACES,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"conversion manifest {key}={manifest.get(key)!r}, expected {expected!r}"
            )
    if len(completed_layers) != N_LAYERS:
        raise ValueError(f"conversion manifest has {len(completed_layers)}/{N_LAYERS} expert layers")
    if len(completed_dense) != N_LAYERS + 1 or len(completed_qweights) != N_LAYERS + 1:
        raise ValueError("conversion manifest does not contain every resident shard")
    for layer in range(N_LAYERS):
        completed = completed_layers.get(str(layer), {})
        path = container / f"experts-{layer:05d}.bin"
        if (not path.is_file() or path.stat().st_size != completed.get("file_bytes") or
                len(completed.get("entries", [])) != N_EXPERTS):
            raise ValueError(f"incomplete converted expert layer {layer}")
    for group in range(N_LAYERS + 1):
        dense_path = container / f"dense-{group:05d}.safetensors"
        qweight_path = container / f"dense-q-{group:05d}.bin"
        if (not dense_path.is_file() or
                dense_path.stat().st_size != completed_dense.get(str(group), {}).get("file_bytes")):
            raise ValueError(f"incomplete lossless resident shard {group}")
        if (not qweight_path.is_file() or
                qweight_path.stat().st_size != completed_qweights.get(str(group), {}).get("file_bytes")):
            raise ValueError(f"incomplete quantized resident shard {group}")
    for required in ("soma.container", "tokenizer.soma", "tokenizer_oracle.bin"):
        if not (container / required).is_file():
            raise ValueError(f"converted container is missing {required}")

    dspark_identity: dict[str, Any] | None = None
    if require_dspark:
        completed_dspark_experts = manifest.get("completed_dspark_expert_layers", {})
        completed_dspark_dense = manifest.get("completed_dspark_dense_shards", {})
        completed_dspark_qweights = manifest.get("completed_dspark_qweight_shards", {})
        if manifest.get("dspark_included") is not True:
            raise ValueError("conversion manifest does not commit DSpark inclusion")
        if not all(len(group) == DSPARK_STAGES for group in (
                completed_dspark_experts,
                completed_dspark_dense,
                completed_dspark_qweights,
        )):
            raise ValueError("conversion manifest does not contain all three DSpark stages")

        expert_shards = []
        for stage in range(DSPARK_STAGES):
            complete = completed_dspark_experts.get(str(stage), {})
            path = container / f"dspark-experts-{stage:05d}.bin"
            entries = complete.get("entries", [])
            if (not path.is_file() or path.stat().st_size != complete.get("file_bytes") or
                    len(entries) != N_EXPERTS or
                    any(int(length) != DSPARK_EXPERT_BYTES for _, length in entries)):
                raise ValueError(f"incomplete converted DSpark expert stage {stage}")
            expert_shards.append({"name": path.name, "bytes": path.stat().st_size})

        dense_index_path = container / "dspark.safetensors.index.json"
        qweight_index_path = container / "dspark.qweights.index.json"
        draft_index_path = container / "soma.dspark"
        for path in (dense_index_path, qweight_index_path, draft_index_path):
            if not path.is_file():
                raise ValueError(f"converted container is missing {path.name}")
        expected_draft_index_bytes = 56 + DSPARK_STAGES * N_EXPERTS * 16
        if draft_index_path.stat().st_size != expected_draft_index_bytes:
            raise ValueError(
                f"soma.dspark bytes={draft_index_path.stat().st_size}, "
                f"expected {expected_draft_index_bytes}"
            )

        ds_dense_index = read_json(dense_index_path)
        ds_qweight_index = read_json(qweight_index_path)
        dspark_names = list(ds_dense_index.get("weight_map", {})) + list(
            ds_qweight_index.get("weight_map", {})
        )
        bad_names = [name for name in dspark_names
                     if not name.startswith("model.dspark.") or name.startswith("mtp.")]
        if bad_names:
            raise ValueError(f"DSpark sidecars contain non-canonical names: {bad_names[:3]}")
        if int(ds_dense_index.get("metadata", {}).get("total_size", 0)) != meta.get(
                "dspark_lossless_resident_bytes"):
            raise ValueError("DSpark lossless resident byte count disagrees with its index")
        if int(ds_qweight_index.get("metadata", {}).get("total_size", 0)) != meta.get(
                "dspark_quantized_resident_bytes"):
            raise ValueError("DSpark quantized resident byte count disagrees with its index")
        if list(container.glob("*.tmp")):
            raise ValueError("converted container contains uncommitted temporary files")
        dspark_identity = {
            "source_tensors_consumed": DSPARK_TENSORS,
            "canonical_resident_tensor_names": len(dspark_names),
            "expert_shards": expert_shards,
            "routed_bytes": DSPARK_ROUTED_BYTES,
            "resident_bytes": DSPARK_RESIDENT_BYTES,
            "kv_bytes_per_sequence": DSPARK_KV_BYTES,
            "soma_dspark_sha256": sha256(draft_index_path),
            "dense_index_sha256": sha256(dense_index_path),
            "qweight_index_sha256": sha256(qweight_index_path),
        }

    loaded_names: list[str] = []
    quantized_payload_bytes = 0
    for name in ("dense.safetensors.index.json", "dense.qweights.index.json"):
        sidecar = read_json(container / name)
        loaded_names.extend(sidecar.get("weight_map", {}).keys())
        if name == "dense.qweights.index.json":
            quantized_payload_bytes = int(sidecar.get("metadata", {}).get("total_size", 0))
    loaded_mtp = sorted(name for name in loaded_names if name.startswith("mtp."))
    if loaded_mtp:
        raise ValueError(f"converted container loads MTP tensors: {loaded_mtp[:3]}")
    lossless_payload_bytes = sum(
        safetensors_payload_bytes(container / f"dense-{group:05d}.safetensors")
        for group in range(N_LAYERS + 1)
    )

    result = {
        "source_repo": MODEL_REPO,
        "source_revision": revision,
        "config_sha256": config_hash,
        "index_sha256": index_hash,
        "source_shards": shards,
        "source_shard_file_bytes": source_shard_file_bytes,
        "source_tensor_payload_bytes": source_tensor_payload_bytes,
        "conversion_manifest_sha256": sha256(manifest_path),
        "conversion_manifest": manifest,
        "container_meta": meta,
        "loaded_mtp_tensors": loaded_mtp,
        "resident_tensor_payload_bytes": lossless_payload_bytes + quantized_payload_bytes,
    }
    if dspark_identity is not None:
        result["dspark"] = dspark_identity
    return result


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    container = args.container.resolve()
    soma = args.soma.resolve()
    artifact_path = args.artifact.resolve()
    artifact: dict[str, Any] = {
        "format": 2 if args.require_dspark else 1,
        "model": MODEL_REPO,
        "dspark_required": args.require_dspark,
        "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    try:
        artifact["identity"] = validate_static(source, container, args.require_dspark)

        plan_command = [
            str(soma), "plan", "--model-dir", str(container), "--json",
            "--ram", "3TiB", "--ram-free", "2TiB", "--disk-bw", "10GiB",
            "--ctx", str(MAX_CONTEXT), "--kv-slots", "1", "--min-tok-s", "0.01",
        ]
        if args.require_dspark:
            plan_command.extend(["--speculative", "dspark"])
        plan, plan_seconds = run_json(plan_command, args.plan_timeout)
        expected_plan = {
            "model_name": MODEL_REPO,
            "schema_version": 2,
            "attention_family": "compressed+sparse",
            "n_layers": N_LAYERS,
            "n_experts": N_EXPERTS,
            "top_k": TOP_K,
            "ctx_size": MAX_CONTEXT,
            "max_context": MAX_CONTEXT,
            "kv_slots": 1,
            "arch_supported": True,
        }
        if args.require_dspark:
            expected_plan.update({
                "speculative_available": True,
                "speculative_selected": True,
                "speculative_method": "dspark",
                "speculative_stages": DSPARK_STAGES,
                "speculative_trained_block_size": DSPARK_TRAINED_BLOCK,
                "speculative_routed_bytes": DSPARK_ROUTED_BYTES,
                "speculative_resident_bytes": DSPARK_RESIDENT_BYTES,
                "speculative_kv_bytes_per_slot": DSPARK_KV_BYTES,
                "speculative_kv_bytes_at_ctx": DSPARK_KV_BYTES,
            })
        for key, expected in expected_plan.items():
            if plan.get(key) != expected:
                raise ValueError(f"1M plan {key}={plan.get(key)!r}, expected {expected!r}")
        resident_payload = artifact["identity"]["resident_tensor_payload_bytes"]
        if args.require_dspark:
            resident_payload += DSPARK_RESIDENT_BYTES
        if plan.get("dense_resident_bytes") != resident_payload:
            raise ValueError(
                f"resident plan bytes {plan.get('dense_resident_bytes')} != "
                f"converted tensor payload {resident_payload}"
            )
        artifact["one_million_context_plan"] = {
            "command": plan_command,
            "seconds": plan_seconds,
            "output": plan,
        }

        detected_available = available_ram_bytes()
        reserve = 8 * 1024**3
        ram_budget = args.ram_budget or max(0, detected_available - reserve)
        if ram_budget <= 0:
            raise ValueError("no RAM remains after the validation safety reserve")
        smoke_plan_command = [
            str(soma), "plan", "--model-dir", str(container), "--json",
            "--ram", str(ram_budget), "--ram-free", str(ram_budget),
            "--disk-bw", "10GiB", "--ctx", "4096", "--kv-slots", "1",
            "--min-tok-s", "0.001",
        ]
        if args.require_dspark:
            smoke_plan_command.extend(["--speculative", "dspark"])
        smoke_plan, smoke_plan_seconds = run_json(smoke_plan_command, args.plan_timeout)
        if smoke_plan.get("dense_resident_bytes", 0) + smoke_plan.get("kv_bytes_at_ctx", 0) > ram_budget:
            raise ValueError("4K resident weights and KV exceed the selected smoke RAM budget")
        if smoke_plan.get("expert_cache_bytes", 0) < smoke_plan.get("expert_bytes", 0):
            raise ValueError("4K smoke RAM budget cannot hold one routed expert")
        artifact["smoke_host"] = {
            "detected_available_ram_bytes": detected_available,
            "reserved_ram_bytes": reserve,
            "selected_ram_budget_bytes": ram_budget,
            "plan_seconds": smoke_plan_seconds,
            "plan": smoke_plan,
        }

        port = choose_port(args.port)
        launches = []
        with tempfile.TemporaryDirectory(prefix="soma-v4-full-smoke-") as directory:
            log_dir = Path(directory)
            try:
                for ordinal in (1, 2):
                    result = smoke_once(
                        soma, container, port, ram_budget,
                        args.startup_timeout, args.generation_timeout, log_dir, ordinal,
                        args.require_dspark,
                    )
                    log_path = log_dir / f"cold-launch-{ordinal}.log"
                    excerpt = log_excerpt(log_path)
                    if re.search(r"(?i)(?:\bnan\b|not a number)", excerpt):
                        raise RuntimeError(f"cold launch {ordinal} log reports NaN")
                    result["log_tail"] = excerpt
                    launches.append(result)
            except Exception:
                artifact["partial_cold_launches"] = launches
                artifact["cold_launch_log_tails"] = {
                    path.name: log_excerpt(path)
                    for path in sorted(log_dir.glob("cold-launch-*.log"))
                }
                raise

        if launches[0]["token_ids"] != launches[1]["token_ids"]:
            raise RuntimeError("cold launches produced different token sequences")
        artifact["cold_launches"] = launches
        artifact["deterministic_token_ids"] = launches[0]["token_ids"]
        artifact["status"] = "passed"
        artifact["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write(artifact_path, artifact)
        print(f"PASS: {artifact_path}")
        return 0
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        artifact["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write(artifact_path, artifact)
        print(f"FAIL: {artifact['error']}")
        print(f"artifact: {artifact_path}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--soma", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ram-budget", type=int, default=0,
                        help="serving budget in bytes (default: available physical RAM minus 8 GiB)")
    parser.add_argument(
        "--require-dspark", action="store_true",
        help="require the translated DSpark payload and smoke with speculative decoding enabled",
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--plan-timeout", type=float, default=300.0)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--generation-timeout", type=float, default=7200.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
