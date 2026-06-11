#!/usr/bin/env python3
"""Local HTTP sidecar for Mantic-Mind Qwen3-TTS.

The C++ control server treats this as an opaque HTTP service. Real Qwen3-TTS
calls are lazy-loaded; use --fake or MM_TTS_FAKE=1 for tests and UI wiring.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_len = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_len)
    except ValueError as exc:
        raise ValueError("invalid Content-Length") from exc
    data = handler.rfile.read(length) if length > 0 else b"{}"
    try:
        body = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def _send_json(handler: BaseHTTPRequestHandler, status: int, body: dict[str, Any]) -> None:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _required_str(body: dict[str, Any], key: str) -> str:
    value = body.get(key, "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _ensure_parent(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_fake_wav(path: str, *, seconds: float = 0.35, sample_rate: int = 24000) -> tuple[int, int]:
    """Write a tiny sine-wave WAV and return (sample_rate, duration_ms)."""
    p = _ensure_parent(path)
    frames = max(1, int(sample_rate * seconds))
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(frames):
            sample = int(1800 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
            wf.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True))
    return sample_rate, int(seconds * 1000)


class QwenRuntime:
    def __init__(self, *, fake: bool = False) -> None:
        self.fake = fake
        self._models: dict[str, Any] = {}

    def _load_model(self, model_id: str) -> Any:
        if model_id in self._models:
            return self._models[model_id]
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except Exception as exc:  # pragma: no cover - exercised in real runtime only
            raise RuntimeError(
                "qwen-tts runtime is not installed; install qwen-tts or run with --fake"
            ) from exc

        dtype_name = os.environ.get("QWEN_TTS_DTYPE", "bfloat16")
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        kwargs: dict[str, Any] = {
            "device_map": os.environ.get("QWEN_TTS_DEVICE_MAP", "cuda:0"),
            "dtype": dtype,
        }
        attn_impl = os.environ.get("QWEN_TTS_ATTN_IMPLEMENTATION", "flash_attention_2")
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
        self._models[model_id] = model
        return model

    def voice_design(self, body: dict[str, Any]) -> dict[str, Any]:
        text = _required_str(body, "text")
        language = str(body.get("language") or "Auto")
        instruct = _required_str(body, "instruct")
        output_audio_path = _required_str(body, "output_audio_path")
        output_prompt_path = _required_str(body, "output_prompt_path")
        voice_design_model_id = str(body.get("voice_design_model_id") or DEFAULT_VOICE_DESIGN_MODEL)
        clone_model_id = str(body.get("clone_model_id") or DEFAULT_CLONE_MODEL)

        if self.fake:
            sr, duration_ms = write_fake_wav(output_audio_path)
            _ensure_parent(output_prompt_path).write_text(
                json.dumps(
                    {
                        "fake": True,
                        "ref_audio": output_audio_path,
                        "ref_text": text,
                        "instruct": instruct,
                        "language": language,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return {
                "ok": True,
                "audio_path": output_audio_path,
                "voice_clone_prompt_path": output_prompt_path,
                "sample_rate": sr,
                "duration_ms": duration_ms,
            }

        try:  # pragma: no cover - depends on local GPU/model install
            import soundfile as sf

            design_model = self._load_model(voice_design_model_id)
            wavs, sr = design_model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )
            _ensure_parent(output_audio_path)
            sf.write(output_audio_path, wavs[0], sr)

            clone_model = self._load_model(clone_model_id)
            prompt = clone_model.create_voice_clone_prompt(
                ref_audio=(wavs[0], sr),
                ref_text=text,
            )
            with _ensure_parent(output_prompt_path).open("wb") as fh:
                pickle.dump(prompt, fh)
            duration_ms = int(len(wavs[0]) * 1000 / sr) if sr else 0
            return {
                "ok": True,
                "audio_path": output_audio_path,
                "voice_clone_prompt_path": output_prompt_path,
                "sample_rate": int(sr),
                "duration_ms": duration_ms,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def synthesize(self, body: dict[str, Any]) -> dict[str, Any]:
        text = _required_str(body, "text")
        language = str(body.get("language") or "Auto")
        voice_clone_prompt_path = _required_str(body, "voice_clone_prompt_path")
        output_audio_path = _required_str(body, "output_audio_path")
        clone_model_id = str(body.get("clone_model_id") or DEFAULT_CLONE_MODEL)

        if self.fake:
            sr, duration_ms = write_fake_wav(output_audio_path)
            return {
                "ok": True,
                "audio_path": output_audio_path,
                "sample_rate": sr,
                "duration_ms": duration_ms,
            }

        try:  # pragma: no cover - depends on local GPU/model install
            import soundfile as sf

            with Path(voice_clone_prompt_path).open("rb") as fh:
                prompt = pickle.load(fh)
            model = self._load_model(clone_model_id)
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
            )
            _ensure_parent(output_audio_path)
            sf.write(output_audio_path, wavs[0], sr)
            duration_ms = int(len(wavs[0]) * 1000 / sr) if sr else 0
            return {
                "ok": True,
                "audio_path": output_audio_path,
                "sample_rate": int(sr),
                "duration_ms": duration_ms,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


class Handler(BaseHTTPRequestHandler):
    runtime: QwenRuntime

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("qwen_tts_service: " + (fmt % args) + "\n")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _send_json(
                self,
                200,
                {
                    "ok": True,
                    "provider": "qwen3-tts",
                    "fake": self.runtime.fake,
                },
            )
            return
        _send_json(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = _read_json(self)
            if self.path == "/voice-design":
                result = self.runtime.voice_design(body)
            elif self.path == "/synthesize":
                result = self.runtime.synthesize(body)
            else:
                _send_json(self, 404, {"ok": False, "error": "not found"})
                return
            _send_json(self, 200 if result.get("ok") else 500, result)
        except ValueError as exc:
            _send_json(self, 400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            _send_json(self, 500, {"ok": False, "error": str(exc)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("MM_TTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MM_TTS_PORT", "9188")))
    parser.add_argument("--fake", action="store_true", default=_truthy(os.environ.get("MM_TTS_FAKE")))
    args = parser.parse_args(argv)

    Handler.runtime = QwenRuntime(fake=args.fake)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"qwen_tts_service listening on http://{args.host}:{args.port} "
        f"(fake={Handler.runtime.fake})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
