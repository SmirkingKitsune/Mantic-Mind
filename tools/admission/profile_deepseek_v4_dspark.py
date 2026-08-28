#!/usr/bin/env python3
"""Profile full-weight DeepSeek V4 autoregressive serving against DSpark.

The benchmark brackets one DSpark launch with two ordinary autoregressive
launches.  Every launch serves the same greedy workload twice: pass zero
captures process-cold behavior (the OS page cache is explicitly *not* flushed),
and the final pass captures a warmed Soma expert cache.  The gate is exact
output-token equivalence plus a measured warm speedup of at least 1.05x.

The artifact is updated atomically after every request so a long full-weight
run remains useful if it is interrupted.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import math
import os
import platform
import re
import socket
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any

from validate_deepseek_v4_full import (
    CONFIG_SHA256,
    INDEX_SHA256,
    MODEL_REPO,
    PINNED_REVISION,
    available_ram_bytes,
    has_nonfinite,
    http_json,
    http_text,
    read_json,
    sha256,
    source_revision,
    wait_for_health,
)


MODEL_PY_SHA256 = "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f"
KERNEL_PY_SHA256 = "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2"

WORKLOADS = [
    {
        "name": "knowledge",
        "prompt": "Explain in plain language why the daytime sky appears blue.",
    },
    {
        "name": "code",
        "prompt": (
            "Write a compact Python function that returns the first n Fibonacci "
            "numbers. Return code and one short example."
        ),
    },
    {
        "name": "reasoning",
        "prompt": (
            "A box has 3 red, 4 blue, and 5 green balls. Two balls are drawn "
            "without replacement. Briefly compute the probability both are blue."
        ),
    },
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def choose_port(requested: int) -> int:
    if requested:
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def total_ram_bytes() -> int:
    if os.name != "nt":
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(os.sysconf("SC_PHYS_PAGES") * page_size)

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
    return int(state.ullTotalPhys)


def process_memory(process: subprocess.Popen[Any]) -> dict[str, int]:
    if os.name != "nt" or process.poll() is not None:
        return {}

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(int(process._handle)),  # noqa: SLF001 - Windows subprocess handle
        ctypes.byref(counters),
        ctypes.sizeof(counters),
    )
    if not ok:
        return {}
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "private_bytes": int(counters.PrivateUsage),
        "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
        "page_faults": int(counters.PageFaultCount),
    }


def parse_telemetry_dump(text: str) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    tier = re.search(r"\bram=(\d+)/(\d+) MiB\b", text)
    cache = re.search(
        r"\bhits=(\d+) misses=(\d+) evictions=(\d+) read=(\d+) MiB\b", text
    )
    sched = re.search(r"\bsteps=(\d+) tokens=(\d+)(?: spec=(\d+)/(\d+))?\b", text)
    if tier:
        result.update(
            ram_resident_mib=int(tier.group(1)),
            ram_capacity_mib=int(tier.group(2)),
        )
    if cache:
        result.update(
            cache_hits=int(cache.group(1)),
            cache_misses=int(cache.group(2)),
            cache_evictions=int(cache.group(3)),
            cache_read_mib=int(cache.group(4)),
        )
    if sched:
        result.update(
            scheduler_steps=int(sched.group(1)),
            scheduler_tokens=int(sched.group(2)),
            speculative_accepted_tokens=int(sched.group(3) or 0),
            speculative_draft_tokens=int(sched.group(4) or 0),
        )
    drafts = int(result.get("speculative_draft_tokens", 0))
    accepted = int(result.get("speculative_accepted_tokens", 0))
    result["speculative_acceptance_rate"] = accepted / drafts if drafts else 0.0
    return result


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "cache_hits",
        "cache_misses",
        "cache_evictions",
        "cache_read_mib",
        "scheduler_steps",
        "scheduler_tokens",
        "speculative_accepted_tokens",
        "speculative_draft_tokens",
    }
    out = {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}
    drafts = out["speculative_draft_tokens"]
    out["speculative_acceptance_rate"] = (
        out["speculative_accepted_tokens"] / drafts if drafts else 0.0
    )
    return out


def log_excerpt(path: Path, limit: int = 128 * 1024) -> str:
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def validate_identity(source: Path, container: Path, soma: Path) -> dict[str, Any]:
    revision = source_revision(source)
    config_hash = sha256(source / "config.json")
    index_hash = sha256(source / "model.safetensors.index.json")
    model_hash = sha256(source / "inference" / "model.py")
    kernel_hash = sha256(source / "inference" / "kernel.py")
    if revision != PINNED_REVISION:
        raise ValueError(f"source revision {revision!r} is not pinned {PINNED_REVISION}")
    if config_hash != CONFIG_SHA256 or index_hash != INDEX_SHA256:
        raise ValueError("source config/index hash does not match the pinned checkpoint")
    if model_hash != MODEL_PY_SHA256 or kernel_hash != KERNEL_PY_SHA256:
        raise ValueError("pinned inference/model.py or inference/kernel.py hash mismatch")

    meta = read_json(container / "container_meta.json")
    required = {
        "source_repo": MODEL_REPO,
        "source_revision": PINNED_REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "model_type": "deepseek_v4",
        "n_layers": 61,
        "n_experts": 384,
        "dspark": "present",
        "dspark_stages": 3,
        "dspark_tensors": 7009,
        "omitted_mtp_tensors": 0,
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise ValueError(f"container {key}={meta.get(key)!r}, expected {expected!r}")

    manifest = container / "conversion-manifest.json"
    return {
        "source_revision": revision,
        "config_sha256": config_hash,
        "index_sha256": index_hash,
        "model_py_sha256": model_hash,
        "kernel_py_sha256": kernel_hash,
        "container_meta_sha256": sha256(container / "container_meta.json"),
        "conversion_manifest_sha256": sha256(manifest),
        "arch": {
            "layers": meta["n_layers"],
            "experts": meta["n_experts"],
            "top_k": 6,
            "dspark_stages": meta["dspark_stages"],
            "dspark_tensors": meta["dspark_tensors"],
            "dspark_routed_bytes": meta["dspark_total_expert_bytes"],
            "dspark_resident_bytes": meta["dspark_resident_bytes"],
        },
        "soma_executable": str(soma),
        "soma_sha256": sha256(soma),
    }


def summarize_requests(requests: list[dict[str, Any]], pass_index: int | None = None) -> dict[str, Any]:
    selected = [r for r in requests if pass_index is None or r["pass"] == pass_index]
    seconds = sum(float(r["seconds"]) for r in selected)
    visible = sum(len(r["token_ids"]) for r in selected)
    scheduler = sum(int(r["telemetry_delta"]["scheduler_tokens"]) for r in selected)
    drafted = sum(int(r["telemetry_delta"]["speculative_draft_tokens"]) for r in selected)
    accepted = sum(int(r["telemetry_delta"]["speculative_accepted_tokens"]) for r in selected)
    return {
        "requests": len(selected),
        "seconds": seconds,
        "visible_output_tokens": visible,
        "scheduler_tokens": scheduler,
        "visible_tokens_per_second": visible / seconds if seconds else 0.0,
        "scheduler_tokens_per_second": scheduler / seconds if seconds else 0.0,
        "speculative_draft_tokens": drafted,
        "speculative_accepted_tokens": accepted,
        "speculative_acceptance_rate": accepted / drafted if drafted else 0.0,
        "cache_read_mib": sum(int(r["telemetry_delta"]["cache_read_mib"]) for r in selected),
        "cache_hits": sum(int(r["telemetry_delta"]["cache_hits"]) for r in selected),
        "cache_misses": sum(int(r["telemetry_delta"]["cache_misses"]) for r in selected),
    }


def finalize_launch(launch: dict[str, Any], passes: int) -> None:
    requests = launch["requests"]
    launch["summary"] = summarize_requests(requests)
    launch["pass_summaries"] = [summarize_requests(requests, i) for i in range(passes)]


def run_launch(
    args: argparse.Namespace,
    artifact: dict[str, Any],
    launch_index: int,
    mode: str,
    port: int,
    ram_budget: int,
    log_dir: Path,
) -> None:
    speculative = mode == "dspark"
    command = [
        str(args.soma),
        "serve",
        "--model-dir",
        str(args.container),
        "--served-name",
        MODEL_REPO,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(args.ctx_size),
        "--kv-slots",
        "1",
        "--max-batch",
        "1",
        "--ram-budget",
        str(ram_budget),
        "--generation-timeout",
        str(max(1, math.ceil(args.generation_timeout))),
        "--speculative",
        "dspark" if speculative else "off",
    ]
    if speculative:
        command.extend(
            [
                "--speculative-tokens",
                str(args.speculative_tokens),
                "--dspark-confidence-threshold",
                str(args.confidence_threshold),
            ]
        )

    log_path = log_dir / f"launch-{launch_index:02d}-{mode}.log"
    launch: dict[str, Any] = {
        "index": launch_index,
        "mode": mode,
        "command": command,
        "started_utc": utc_now(),
        "requests": [],
        "status": "starting",
    }
    artifact["launches"].append(launch)
    atomic_write(args.artifact, artifact)

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            launch["startup_seconds"] = wait_for_health(
                process, f"http://127.0.0.1:{port}/health", args.startup_timeout
            )
            launch["status"] = "running"
            atomic_write(args.artifact, artifact)

            for pass_index in range(args.passes):
                for workload in WORKLOADS:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Soma exited during profile with code {process.returncode}"
                        )
                    before_dump = http_text(
                        f"http://127.0.0.1:{port}/internal/telemetry/dump", timeout=15.0
                    )
                    before = parse_telemetry_dump(before_dump)
                    available_before = available_ram_bytes()
                    payload = {
                        "model": MODEL_REPO,
                        "messages": [{"role": "user", "content": workload["prompt"]}],
                        "temperature": 0,
                        "top_p": 1,
                        "max_tokens": args.max_tokens,
                        "stream": False,
                        "soma_return_token_ids": True,
                    }
                    started = time.perf_counter()
                    response = http_json(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        payload,
                        args.generation_timeout,
                    )
                    elapsed = time.perf_counter() - started
                    after_dump = http_text(
                        f"http://127.0.0.1:{port}/internal/telemetry/dump", timeout=15.0
                    )
                    after = parse_telemetry_dump(after_dump)
                    if has_nonfinite(response):
                        raise RuntimeError(f"{mode}/{workload['name']} returned NaN or infinity")
                    choices = response.get("choices")
                    token_ids = response.get("soma_token_ids")
                    if not isinstance(choices, list) or len(choices) != 1:
                        raise RuntimeError(f"invalid choices for {mode}/{workload['name']}")
                    if not isinstance(token_ids, list) or not all(
                        isinstance(token, int) and token >= 0 for token in token_ids
                    ):
                        raise RuntimeError(f"invalid token IDs for {mode}/{workload['name']}")
                    message = choices[0].get("message", {})
                    request_result = {
                        "pass": pass_index,
                        "workload": workload["name"],
                        "prompt": workload["prompt"],
                        "seconds": elapsed,
                        "finish_reason": choices[0].get("finish_reason"),
                        "token_ids": token_ids,
                        "content": message.get("content", ""),
                        "reasoning_content": message.get("reasoning_content", ""),
                        "telemetry_before": before,
                        "telemetry_after": after,
                        "telemetry_delta": counter_delta(before, after),
                        "available_ram_before_bytes": available_before,
                        "available_ram_after_bytes": available_ram_bytes(),
                        "process_memory": process_memory(process),
                    }
                    launch["requests"].append(request_result)
                    finalize_launch(launch, args.passes)
                    atomic_write(args.artifact, artifact)
                    delta = request_result["telemetry_delta"]
                    print(
                        f"[{launch_index}:{mode}] pass={pass_index} {workload['name']} "
                        f"tokens={len(token_ids)} seconds={elapsed:.3f} "
                        f"spec={delta['speculative_accepted_tokens']}/"
                        f"{delta['speculative_draft_tokens']}",
                        flush=True,
                    )

            launch["process_memory"] = process_memory(process)
            finalize_launch(launch, args.passes)
            launch["status"] = "passed"
            launch["completed_utc"] = utc_now()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
            log.flush()
            launch["log_tail"] = log_excerpt(log_path)
            if re.search(r"(?i)(?:\bnan\b|not a number)", launch["log_tail"]):
                raise RuntimeError(f"{mode} server log reports NaN")
            atomic_write(args.artifact, artifact)


def build_comparison(launches: list[dict[str, Any]], passes: int) -> dict[str, Any]:
    baselines = [launch for launch in launches if launch["mode"] == "off"]
    dsparks = [launch for launch in launches if launch["mode"] == "dspark"]
    if len(baselines) < 2 or len(dsparks) != 1:
        raise ValueError("representative order requires two off launches bracketing one DSpark launch")

    reference: dict[tuple[int, str], list[int]] = {}
    for request in baselines[0]["requests"]:
        reference[(request["pass"], request["workload"])] = request["token_ids"]
    mismatches = []
    for launch in launches[1:]:
        for request in launch["requests"]:
            key = (request["pass"], request["workload"])
            if request["token_ids"] != reference.get(key):
                mismatches.append(
                    {
                        "launch": launch["index"],
                        "mode": launch["mode"],
                        "pass": key[0],
                        "workload": key[1],
                        "expected": reference.get(key),
                        "actual": request["token_ids"],
                    }
                )

    dspark = dsparks[0]
    pass_comparisons = []
    for pass_index in range(passes):
        baseline_seconds = mean(
            launch["pass_summaries"][pass_index]["seconds"] for launch in baselines
        )
        dspark_seconds = dspark["pass_summaries"][pass_index]["seconds"]
        pass_comparisons.append(
            {
                "pass": pass_index,
                "baseline_mean_seconds": baseline_seconds,
                "baseline_range_seconds": [
                    min(launch["pass_summaries"][pass_index]["seconds"] for launch in baselines),
                    max(launch["pass_summaries"][pass_index]["seconds"] for launch in baselines),
                ],
                "dspark_seconds": dspark_seconds,
                "dspark_speedup": baseline_seconds / dspark_seconds if dspark_seconds else 0.0,
                "dspark_acceptance_rate": dspark["pass_summaries"][pass_index][
                    "speculative_acceptance_rate"
                ],
                "dspark_accepted_tokens": dspark["pass_summaries"][pass_index][
                    "speculative_accepted_tokens"
                ],
                "dspark_draft_tokens": dspark["pass_summaries"][pass_index][
                    "speculative_draft_tokens"
                ],
            }
        )

    warm = pass_comparisons[-1]
    return {
        "exact_token_equivalence": not mismatches,
        "token_mismatches": mismatches,
        "pass_comparisons": pass_comparisons,
        "primary_pass": passes - 1,
        "profiled_speedup": warm["dspark_speedup"],
        "profiled_acceptance_rate": warm["dspark_acceptance_rate"],
        "auto_threshold": 1.05,
        "auto_eligible": not mismatches and warm["dspark_speedup"] >= 1.05,
        "recommendation": (
            "enable-auto"
            if not mismatches and warm["dspark_speedup"] >= 1.05
            else "keep-auto-disabled"
        ),
    }


def run(args: argparse.Namespace) -> int:
    args.source = args.source.resolve()
    args.container = args.container.resolve()
    args.soma = args.soma.resolve()
    args.artifact = args.artifact.resolve()
    order = [item.strip() for item in args.order.split(",") if item.strip()]
    if order != ["off", "dspark", "off"]:
        raise ValueError("representative profile order must be off,dspark,off")
    if args.passes < 2:
        raise ValueError("representative profile requires at least two passes per launch")
    if args.max_tokens < 8:
        raise ValueError("representative profile requires at least eight tokens per workload")

    available = available_ram_bytes()
    reserve = args.ram_reserve_gib * 1024**3
    ram_budget = args.ram_budget or max(0, available - reserve)
    if ram_budget <= 0:
        raise ValueError("no RAM remains after the requested safety reserve")

    artifact: dict[str, Any] = {
        "format": 1,
        "kind": "deepseek-v4-dspark-full-weight-performance-profile",
        "status": "running",
        "started_utc": utc_now(),
        "model": MODEL_REPO,
        "identity": validate_identity(args.source, args.container, args.soma),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
            "logical_processors": os.cpu_count(),
            "total_ram_bytes": total_ram_bytes(),
            "available_ram_at_start_bytes": available,
            "ram_budget_bytes": ram_budget,
            "ram_reserve_bytes": reserve,
        },
        "methodology": {
            "order": order,
            "ctx_size": args.ctx_size,
            "max_batch": 1,
            "kv_slots": 1,
            "passes_per_launch": args.passes,
            "max_tokens_per_workload": args.max_tokens,
            "workloads": WORKLOADS,
            "temperature": 0,
            "top_p": 1,
            "speculative_tokens": args.speculative_tokens,
            "confidence_threshold": args.confidence_threshold,
            "cold_definition": (
                "new Soma process and empty Soma expert cache; OS page/standby cache not flushed"
            ),
            "primary_metric": "final-pass matched wall-clock speedup",
        },
        "launches": [],
    }
    atomic_write(args.artifact, artifact)

    port = choose_port(args.port)
    log_dir = args.artifact.parent / f"{args.artifact.stem}-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        for launch_index, mode in enumerate(order):
            run_launch(args, artifact, launch_index, mode, port, ram_budget, log_dir)
        artifact["comparison"] = build_comparison(artifact["launches"], args.passes)
        if not artifact["comparison"]["exact_token_equivalence"]:
            raise RuntimeError("DSpark output diverged from autoregressive greedy tokens")
        artifact["status"] = "passed"
        artifact["completed_utc"] = utc_now()
        atomic_write(args.artifact, artifact)
        comparison = artifact["comparison"]
        print(
            f"PASS: speedup={comparison['profiled_speedup']:.4f}x "
            f"acceptance={comparison['profiled_acceptance_rate']:.2%} "
            f"recommendation={comparison['recommendation']}\nartifact: {args.artifact}",
            flush=True,
        )
        return 0
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        artifact["completed_utc"] = utc_now()
        atomic_write(args.artifact, artifact)
        print(f"FAIL: {artifact['error']}\nartifact: {args.artifact}", flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--soma", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--order", default="off,dspark,off")
    parser.add_argument("--speculative-tokens", type=int, default=7)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--ram-budget", type=int, default=0)
    parser.add_argument("--ram-reserve-gib", type=int, default=12)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--generation-timeout", type=float, default=7200.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
