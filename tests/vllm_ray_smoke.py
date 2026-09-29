#!/usr/bin/env python3
"""Opt-in Linux multi-node vLLM/Ray hardware smoke.

Required environment:
  MM_TEST_VLLM_RAY_CONTROL_URL  control API base URL (the Mantic port, :9090)
  MM_TEST_VLLM_RAY_OPENAI_URL   the OpenAI-compat listener (:9091)
  MM_TEST_VLLM_RAY_MODEL_ID     a node-served model backed by a small HF model
Optional:
  MM_TEST_VLLM_RAY_TOKEN        bearer token (the same one works on both ports)
  MM_TEST_VLLM_RAY_EXPECT_GLOO  1 selects the experimental fallback smoke

The request goes through the stateless :9091 listener, which is the shortest
path from a served model to its placement: no user, no conversation, nothing
stored — only the Ray group the placement forms is under test.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import urllib.error
import urllib.request
import uuid


SKIP = 77


def request(base: str, path: str, token: str, body: dict | None = None) -> bytes:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=1200) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc


def ray_status(base: str, token: str) -> dict:
    return json.loads(request(base, "/v1/cluster/engines/ray", token))


def main() -> int:
    base = os.environ.get("MM_TEST_VLLM_RAY_CONTROL_URL", "").strip()
    openai = os.environ.get("MM_TEST_VLLM_RAY_OPENAI_URL", "").strip()
    model = os.environ.get("MM_TEST_VLLM_RAY_MODEL_ID", "").strip()
    token = os.environ.get("MM_TEST_VLLM_RAY_TOKEN", "").strip()
    expect_gloo = os.environ.get("MM_TEST_VLLM_RAY_EXPECT_GLOO", "").lower() in {
        "1", "true", "yes", "on"
    }
    if platform.system() != "Linux" or not base or not openai or not model:
        print("SKIP: set MM_TEST_VLLM_RAY_CONTROL_URL, MM_TEST_VLLM_RAY_OPENAI_URL and "
              "MM_TEST_VLLM_RAY_MODEL_ID on a Linux multi-node cluster")
        return SKIP

    before = ray_status(base, token)
    pp = int(before.get("pipeline_parallel_size", 0))
    if not before.get("configured") or not before.get("required") or pp < 2:
        raise RuntimeError("cluster must select vLLM with pipeline_parallel_size > 1")
    if expect_gloo and not before.get("allow_experimental_gloo"):
        raise RuntimeError("Gloo smoke requested but allow_experimental_gloo is false")

    completion = json.loads(request(
        openai,
        "/v1/chat/completions",
        token,
        {"model": model,
         "messages": [{"role": "user",
                       "content": f"Reply with the single word READY. smoke={uuid.uuid4()}"}]},
    ))
    choices = completion.get("choices") or []
    if completion.get("model") != model or not choices:
        raise RuntimeError(f"completion did not come back for {model!r}: {completion}")

    after = ray_status(base, token)
    # Ray groups are owned by the placement's holder: the served-model id.
    groups = [g for g in after.get("groups", []) if g.get("served_model_id") == model]
    if not groups:
        if expect_gloo:
            print("PASS: experimental Gloo launch failed cleanly and backup served the request; "
                  "this is not a release support guarantee")
            return 0
        raise RuntimeError("request succeeded but no owned Ray group is active")

    group = groups[0]
    members = group.get("members", [])
    if group.get("state") != "active" or len(members) != pp:
        raise RuntimeError(f"Ray group is incomplete: expected {pp} active members, got {members}")
    if sum(1 for member in members if member.get("role") == "head") != 1:
        raise RuntimeError("Ray group must contain exactly one head")
    expected_tp = int(after.get("tensor_parallel_size", 0))
    if any(int(member.get("reserved_gpus", -1)) != expected_tp for member in members):
        raise RuntimeError("Ray member GPU reservations do not match configured TP")

    transport = group.get("transport")
    if expect_gloo:
        if transport != "gloo":
            raise RuntimeError(f"experimental smoke expected Gloo, got {transport!r}")
        print("PASS: experimental Gloo group served a request; not a release support guarantee")
    elif transport != "nccl":
        raise RuntimeError(f"release smoke requires NCCL, got {transport!r}")
    else:
        print(f"PASS: NCCL vLLM/Ray group served through {pp} Linux nodes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # concise CTest failure output
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
