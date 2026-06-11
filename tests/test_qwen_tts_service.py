import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_service_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "qwen_tts_service.py"
    spec = importlib.util.spec_from_file_location("qwen_tts_service", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QwenTtsServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = load_service_module()

    def test_fake_voice_design_creates_preview_and_prompt(self):
        runtime = self.service.QwenRuntime(fake=True)
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "preview.wav"
            prompt = Path(tmp) / "voice.prompt.pkl"
            result = runtime.voice_design(
                {
                    "text": "Hello from the local sidecar.",
                    "language": "English",
                    "instruct": "A clear original synthetic voice.",
                    "output_audio_path": str(audio),
                    "output_prompt_path": str(prompt),
                }
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["audio_path"], str(audio))
            self.assertEqual(result["voice_clone_prompt_path"], str(prompt))
            self.assertGreater(result["sample_rate"], 0)
            self.assertGreater(result["duration_ms"], 0)
            self.assertTrue(audio.exists())
            self.assertTrue(prompt.exists())

    def test_fake_synthesis_creates_wav(self):
        runtime = self.service.QwenRuntime(fake=True)
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "voice.prompt.pkl"
            prompt.write_text("{}", encoding="utf-8")
            audio = Path(tmp) / "speech.wav"
            result = runtime.synthesize(
                {
                    "text": "Speak this.",
                    "language": "English",
                    "voice_clone_prompt_path": str(prompt),
                    "output_audio_path": str(audio),
                }
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["audio_path"], str(audio))
            self.assertTrue(audio.exists())

    def test_validation_rejects_missing_required_fields(self):
        runtime = self.service.QwenRuntime(fake=True)
        with self.assertRaises(ValueError):
            runtime.synthesize({"text": "missing prompt"})


if __name__ == "__main__":
    unittest.main()
