#!/usr/bin/env python3
"""Opt-in end-to-end smoke test for a real llama.cpp vision agent.

The control server and at least one compatible node must already be running.
This script creates a temporary vision profile, uploads one PNG through the
managed attachment API, asks an image question, then asks a text-only follow-up
in the same persisted conversation. The temporary agent is removed afterward.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
import uuid


def request(
    base_url: str,
    method: str,
    path: str,
    token: str,
    body: bytes | None = None,
    content_type: str = "application/json",
    extra_headers: dict[str, str] | None = None,
    timeout: int = 900,
) -> tuple[int, bytes]:
    headers = {"Content-Type": content_type}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed with HTTP {error.code}: {detail}") from error


def json_request(
    base_url: str,
    method: str,
    path: str,
    token: str,
    payload: dict[str, object] | None = None,
    timeout: int = 900,
) -> dict[str, object]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    _, response = request(base_url, method, path, token, body, timeout=timeout)
    return json.loads(response.decode("utf-8")) if response else {}


def parse_chat_events(body: bytes) -> tuple[str, str]:
    text = body.decode("utf-8", errors="replace")
    answer: list[str] = []
    conversation_id = ""
    success = False
    failure = "chat ended without a done event"
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        event = json.loads(payload)
        if event.get("type") == "delta":
            answer.append(str(event.get("content", "")))
        elif event.get("type") == "done":
            conversation_id = str(event.get("conv_id", ""))
            success = bool(event.get("success", False))
            failure = str(event.get("error", "chat failed"))
    if not success:
        raise RuntimeError(failure)
    rendered = "".join(answer).strip()
    if not rendered:
        raise RuntimeError("vision agent returned no assistant text")
    if not conversation_id:
        raise RuntimeError("chat response omitted the persisted conversation id")
    return conversation_id, rendered


def chat(
    base_url: str,
    token: str,
    agent_id: str,
    message: str,
    conversation_id: str = "",
    attachment_ids: list[str] | None = None,
) -> tuple[str, str]:
    payload: dict[str, object] = {"message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if attachment_ids:
        payload["attachment_ids"] = attachment_ids
    _, response = request(
        base_url,
        "POST",
        f"/v1/agents/{agent_id}/chat",
        token,
        json.dumps(payload).encode("utf-8"),
        timeout=1800,
    )
    return parse_chat_events(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9090")
    parser.add_argument("--token", default="", help="control external API bearer token")
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--mmproj", required=True, type=pathlib.Path)
    parser.add_argument("--image", required=True, type=pathlib.Path)
    args = parser.parse_args()

    model = args.model.expanduser().resolve()
    projector = args.mmproj.expanduser().resolve()
    image = args.image.expanduser().resolve()
    for label, path in (("model", model), ("projector", projector), ("image", image)):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    projector_name = projector.name.lower()
    if not (projector_name.startswith("mmproj-") and projector_name.endswith(".gguf")):
        parser.error("--mmproj must name a matching mmproj-*.gguf file")
    if image.stat().st_size > 50 * 1024 * 1024:
        parser.error("--image exceeds the 50 MiB limit")
    png = image.read_bytes()
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        parser.error("--image must be a PNG for this smoke test")

    agent_id = ""
    try:
        created = json_request(
            args.base_url,
            "POST",
            "/v1/agents",
            args.token,
            {
                "name": f"vision-smoke-{int(time.time())}-{uuid.uuid4().hex[:8]}",
                "model_path": str(model),
                "inference_backend": "llama-cpp",
                "vision_settings": {
                    "enabled": True,
                    "mmproj_path": str(projector),
                },
                "memories_enabled": False,
                "tools_enabled": False,
            },
        )
        agent_id = str(created["id"])
        status, uploaded_body = request(
            args.base_url,
            "POST",
            f"/v1/agents/{agent_id}/attachments",
            args.token,
            png,
            content_type="image/png",
            extra_headers={"X-Filename": image.name},
        )
        if status != 201:
            raise RuntimeError(f"attachment upload returned HTTP {status}")
        attachment = json.loads(uploaded_body.decode("utf-8"))
        attachment_id = str(attachment["id"])

        conversation_id, first = chat(
            args.base_url,
            args.token,
            agent_id,
            "Describe the main subject and visible details in this image.",
            attachment_ids=[attachment_id],
        )
        followup_conversation_id, second = chat(
            args.base_url,
            args.token,
            agent_id,
            "What was the main visual subject in the image I just sent?",
            conversation_id=conversation_id,
        )
        if followup_conversation_id != conversation_id:
            raise RuntimeError("follow-up did not continue the persisted conversation")

        print("Vision answer:", first)
        print("Persisted-context follow-up:", second)
        print("PASS: real GGUF projector vision smoke test")
        return 0
    finally:
        if agent_id:
            try:
                request(args.base_url, "DELETE", f"/v1/agents/{agent_id}", args.token)
            except Exception as error:  # cleanup should not hide the smoke result
                print(f"warning: could not delete temporary agent: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
