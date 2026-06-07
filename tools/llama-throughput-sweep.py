#!/usr/bin/env python3
"""Mantic-Mind llama-server throughput sweep.

This drives the node API instead of bare llama-server so results reflect
Mantic-Mind's real slot and SSE inference path.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_int_list(value: str | None, default: str) -> list[int]:
    raw = value or default
    return [int(part) for part in raw.replace(",", " ").split() if part]


def parse_optional_int_list(value: str | None, default: str) -> list[int | None]:
    raw = value or default
    out: list[int | None] = []
    for part in raw.replace(",", " ").split():
        if part.lower() == "default":
            out.append(None)
        else:
            out.append(int(part))
    return out or [None]


def json_request(
    base_url: str,
    path: str,
    api_key: str,
    payload: dict,
    timeout_s: float,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {text}") from exc


def stream_infer(
    base_url: str,
    api_key: str,
    slot_id: str,
    model: str,
    settings: dict,
    prompt: str,
    timeout_s: float,
) -> int:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "settings": settings,
        "stream": True,
        "slot_id": slot_id,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/node/infer",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        },
        method="POST",
    )

    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                typ = event.get("type")
                if typ == "done":
                    tokens = int(event.get("tokens_used") or 0)
                elif typ == "error":
                    raise RuntimeError(event.get("message") or "infer error")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} /api/node/infer: {text}") from exc
    return tokens


def make_settings(args: argparse.Namespace, parallel: int, batch: int | None, ubatch: int | None, max_tokens: int) -> dict:
    return {
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "n_threads": args.n_threads,
        "n_threads_http": args.n_threads_http,
        "parallel": parallel,
        "batch_size": -1 if batch is None else batch,
        "ubatch_size": -1 if ubatch is None else ubatch,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_tokens,
        "flash_attn": not args.no_flash_attn,
        "extra_args": args.server_arg or [],
    }


def load_slot(args: argparse.Namespace, settings: dict) -> str:
    if not args.model_path:
        if not args.slot_id:
            raise RuntimeError("Either --model-path or --slot-id is required.")
        return args.slot_id

    payload = {
        "model_path": args.model_path,
        "settings": settings,
        "agent_id": args.agent_id,
    }
    data = json_request(args.node_url, "/api/node/load-model", args.api_key, payload, args.load_timeout_s)
    slot_id = data.get("slot_id")
    if not slot_id:
        raise RuntimeError(f"load-model did not return slot_id: {data}")
    return slot_id


def unload_slot(args: argparse.Namespace, slot_id: str) -> None:
    if not args.model_path or args.no_unload:
        return
    try:
        json_request(
            args.node_url,
            "/api/node/unload-model",
            args.api_key,
            {"slot_id": slot_id},
            args.request_timeout_s,
        )
    except Exception as exc:
        print(f"warn unload slot {slot_id}: {exc}", file=sys.stderr)


def run_batch(args: argparse.Namespace, slot_id: str, settings: dict, concurrency: int, total_requests: int) -> dict:
    started = time.time()
    tokens = 0
    errors = 0
    last_error = ""

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                stream_infer,
                args.node_url,
                args.api_key,
                slot_id,
                args.model_path or args.model,
                settings,
                args.prompt,
                args.request_timeout_s,
            )
            for _ in range(total_requests)
        ]
        for future in as_completed(futures):
            try:
                tokens += future.result()
            except Exception as exc:
                errors += 1
                last_error = str(exc)

    elapsed = time.time() - started
    return {
        "tokens": tokens,
        "elapsed": elapsed,
        "throughput": tokens / elapsed if elapsed > 0 else 0.0,
        "errors": errors,
        "last_error": last_error,
    }


def default_output_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / "mantic_llama_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"mantic_llama_sweep_{stamp}.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-url", required=True, help="Node API base URL, e.g. http://127.0.0.1:8081")
    parser.add_argument("--api-key", required=True, help="Node API bearer token")
    parser.add_argument("--model-path", help="Model path/ref to load for each server-setting cell")
    parser.add_argument("--model", default="benchmark", help="Model field for existing --slot-id mode")
    parser.add_argument("--slot-id", help="Existing ready slot to benchmark without reloading")
    parser.add_argument("--agent-id", default="benchmark", help="Agent id recorded on loaded benchmark slots")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Per-session context size")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-threads", type=int, default=-1)
    parser.add_argument("--n-threads-http", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--server-arg", action="append", help="Extra llama-server arg; repeat once per arg")
    parser.add_argument("--parallel-list", default="1,2,4,8")
    parser.add_argument("--batch-size-list", default="default")
    parser.add_argument("--ubatch-size-list", default="default")
    parser.add_argument("--max-tokens-list", default="64,128,256")
    parser.add_argument("--concurrency-list", default="1,2,4,8")
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--requests-multiplier", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--prompt", default="Share three practical optimization tips for model serving.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--load-timeout-s", type=float, default=300)
    parser.add_argument("--request-timeout-s", type=float, default=180)
    parser.add_argument("--no-unload", action="store_true", help="Keep slots loaded after model-path cells")
    args = parser.parse_args()

    if args.requests_multiplier < 1:
        args.requests_multiplier = 1
    if not args.model_path and not args.slot_id:
        parser.error("Either --model-path or --slot-id is required.")
    if args.slot_id and not args.model_path:
        print("slot-id mode benchmarks an existing slot; server-setting lists do not reload llama-server.", file=sys.stderr)
        parallel_values = [1]
        batch_values: list[int | None] = [None]
        ubatch_values: list[int | None] = [None]
    else:
        parallel_values = parse_int_list(args.parallel_list, "1")
        batch_values = parse_optional_int_list(args.batch_size_list, "default")
        ubatch_values = parse_optional_int_list(args.ubatch_size_list, "default")

    max_tokens_values = parse_int_list(args.max_tokens_list, "128")
    concurrency_values = parse_int_list(args.concurrency_list, "1")
    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "parallel",
        "batch_size",
        "ubatch_size",
        "max_tokens",
        "concurrency",
        "total_requests",
        "throughput_tps",
        "total_tokens",
        "elapsed_s",
        "errors",
        "slot_id",
        "last_error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for parallel, batch, ubatch in itertools.product(parallel_values, batch_values, ubatch_values):
            first_tokens = max_tokens_values[0]
            load_settings = make_settings(args, parallel, batch, ubatch, first_tokens)
            slot_id = load_slot(args, load_settings)
            try:
                for _ in range(max(0, args.warmup_requests)):
                    warm = dict(load_settings)
                    warm["max_tokens"] = min(8, first_tokens)
                    stream_infer(
                        args.node_url,
                        args.api_key,
                        slot_id,
                        args.model_path or args.model,
                        warm,
                        "warmup",
                        args.request_timeout_s,
                    )

                for max_tokens, concurrency in itertools.product(max_tokens_values, concurrency_values):
                    settings = make_settings(args, parallel, batch, ubatch, max_tokens)
                    total_requests = args.num_requests or max(1, concurrency * args.requests_multiplier)
                    result = run_batch(args, slot_id, settings, concurrency, total_requests)
                    row = {
                        "parallel": parallel,
                        "batch_size": "default" if batch is None else batch,
                        "ubatch_size": "default" if ubatch is None else ubatch,
                        "max_tokens": max_tokens,
                        "concurrency": concurrency,
                        "total_requests": total_requests,
                        "throughput_tps": f"{result['throughput']:.2f}",
                        "total_tokens": result["tokens"],
                        "elapsed_s": f"{result['elapsed']:.2f}",
                        "errors": result["errors"],
                        "slot_id": slot_id,
                        "last_error": result["last_error"],
                    }
                    writer.writerow(row)
                    handle.flush()
                    print(",".join(str(row[name]) for name in fields))
            finally:
                unload_slot(args, slot_id)

    print(f"results_file={output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
