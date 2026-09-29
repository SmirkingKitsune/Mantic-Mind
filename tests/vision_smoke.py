#!/usr/bin/env python3
"""Opt-in end-to-end smoke test for a real vision model served to a user.

The control server and at least one compatible node must already be running.
This script adds a temporary vision entry to the served-model catalog and a
temporary user, uploads one PNG through the managed attachment API, asks an
image question, then asks a text-only follow-up in the same persisted
conversation. Both are removed afterward.

With --text-model it also runs a third turn on a model that CANNOT see, in the
same conversation. That is the switch the user system exists for: the stored
image stays where it is, the text-only model is sent `[image omitted: <file>]`
in its place, and the turn succeeds instead of failing on history it cannot
represent.
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


class Turn:
    """What one chat turn produced: the answer, and who wrote it."""

    def __init__(self, conversation_id: str, text: str, model_id: str) -> None:
        self.conversation_id = conversation_id
        self.text = text
        self.model_id = model_id


def parse_chat_events(body: bytes) -> Turn:
    text = body.decode("utf-8", errors="replace")
    answer: list[str] = []
    conversation_id = ""
    model_id = ""
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
            model_id = str(event.get("model_id", ""))
            success = bool(event.get("success", False))
            # The code says WHICH failure; without it a caller has only prose.
            code = str(event.get("code", ""))
            failure = str(event.get("error", "chat failed"))
            if code:
                failure += f"  [{code}]"
    if not success:
        raise RuntimeError(failure)
    rendered = "".join(answer).strip()
    if not rendered:
        raise RuntimeError("the model returned no assistant text")
    if not conversation_id:
        raise RuntimeError("chat response omitted the persisted conversation id")
    return Turn(conversation_id, rendered, model_id)


def chat(
    base_url: str,
    token: str,
    user_id: str,
    message: str,
    conversation_id: str = "",
    attachment_ids: list[str] | None = None,
    model: str = "",
) -> Turn:
    payload: dict[str, object] = {"message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if attachment_ids:
        payload["attachment_ids"] = attachment_ids
    if model:
        payload["model"] = model
    _, response = request(
        base_url,
        "POST",
        f"/v1/users/{user_id}/chat",
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
    parser.add_argument(
        "--text-model",
        default="",
        help="an existing served-model id WITHOUT vision; when given, the "
        "conversation switches to it for a third turn",
    )
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

    stamp = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    served_model_id = f"vision-smoke-{stamp}"
    user_id = f"vision-smoke-{stamp}"
    created_model = False
    created_user = False
    try:
        if args.text_model:
            # Fail before anything is created if the switch target is not there.
            json_request(args.base_url, "GET", f"/v1/served-models/{args.text_model}", args.token)

        json_request(
            args.base_url,
            "POST",
            "/v1/served-models",
            args.token,
            {
                "id": served_model_id,
                "display_name": "Vision smoke",
                "model_path": str(model),
                "inference_backend": "node",
                "vision_settings": {"enabled": True, "mmproj_path": str(projector)},
            },
        )
        created_model = True
        json_request(
            args.base_url,
            "POST",
            "/v1/users",
            args.token,
            {
                "id": user_id,
                "display_name": "Vision smoke",
                "default_model": served_model_id,
                "memories_enabled": False,
                "tools_enabled": False,
            },
        )
        created_user = True

        status, uploaded_body = request(
            args.base_url,
            "POST",
            f"/v1/users/{user_id}/attachments",
            args.token,
            png,
            content_type="image/png",
            extra_headers={"X-Filename": image.name},
        )
        if status != 201:
            raise RuntimeError(f"attachment upload returned HTTP {status}")
        attachment = json.loads(uploaded_body.decode("utf-8"))
        attachment_id = str(attachment["id"])

        first = chat(
            args.base_url,
            args.token,
            user_id,
            "Describe the main subject and visible details in this image.",
            attachment_ids=[attachment_id],
        )
        if first.model_id != served_model_id:
            raise RuntimeError(f"image turn was answered by {first.model_id!r}")
        second = chat(
            args.base_url,
            args.token,
            user_id,
            "What was the main visual subject in the image I just sent?",
            conversation_id=first.conversation_id,
        )
        if second.conversation_id != first.conversation_id:
            raise RuntimeError("follow-up did not continue the persisted conversation")

        print("Vision answer:", first.text)
        print("Persisted-context follow-up:", second.text)

        if args.text_model:
            switched = chat(
                args.base_url,
                args.token,
                user_id,
                "Ignore the picture. Reply with one short sentence of plain text.",
                conversation_id=first.conversation_id,
                model=args.text_model,
            )
            if switched.model_id != args.text_model:
                raise RuntimeError(
                    f"switch turn was answered by {switched.model_id!r}, not {args.text_model!r}"
                )
            conversation = json_request(
                args.base_url,
                "GET",
                f"/v1/users/{user_id}/conversations/{first.conversation_id}",
                args.token,
            )
            messages = conversation.get("messages", [])
            if not isinstance(messages, list):
                raise RuntimeError("conversation response carried no messages")
            # The image is STORED; only what is sent to a blind model is rewritten.
            kept = any(
                part.get("type") == "image_attachment"
                for message in messages
                if isinstance(message, dict)
                for part in message.get("content_parts", [])
                if isinstance(part, dict)
            )
            if not kept:
                raise RuntimeError("the stored image turn lost its attachment on the switch")
            answered_by = [
                str(m.get("model_id", ""))
                for m in messages
                if isinstance(m, dict) and m.get("role") == "assistant"
            ]
            if answered_by[-1] != args.text_model or answered_by[0] != served_model_id:
                raise RuntimeError(f"messages do not record who answered: {answered_by}")
            print("Text-only model after the switch:", switched.text)
            print("Answered by, in order:", ", ".join(answered_by))

        print("PASS: real GGUF projector vision smoke test")
        return 0
    finally:
        # Cleanup must not hide the smoke result: report and carry on.
        if created_user:
            try:
                request(args.base_url, "DELETE", f"/v1/users/{user_id}", args.token)
            except Exception as error:
                print(f"warning: could not delete temporary user: {error}", file=sys.stderr)
        if created_model:
            try:
                request(args.base_url, "DELETE", f"/v1/served-models/{served_model_id}", args.token)
            except Exception as error:
                print(f"warning: could not delete temporary served model: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
