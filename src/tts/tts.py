"""
Text-to-Speech via Kokoro (ONNX). No CUDA needed — runs on CPU or DirectML.

Produces a WAV file (and optionally raw float samples) from text. The audio
path + sample rate are consumed by the face-render layer for lip sync.

Install:  pip install kokoro-onnx soundfile
Models are auto-downloaded on first use into models/kokoro/.
"""
from __future__ import annotations

import os
import numpy as np
import soundfile as sf
from kokoro import Kokoro

from src.config import Config


class TTS:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.models_dir, exist_ok=True)
        kokoro_dir = os.path.join(cfg.models_dir, "kokoro")
        os.makedirs(kokoro_dir, exist_ok=True)
        model_path = os.path.join(kokoro_dir, "kokoro-v1.0.onnx")
        voices_path = os.path.join(kokoro_dir, "voices.json")
        # Kokoro auto-downloads to these paths if missing.
        self.kokoro = Kokoro(model_path, voices_path)
        # Pick execution provider: DirectML if available (AMD GPU), else CPU.
        providers = ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            if cfg.face_backend in ("directml", "auto") and "DmlExecutionProvider" in avail:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
                print("[tts] Using DirectML (AMD GPU) for Kokoro.")
            else:
                print("[tts] Using CPU for Kokoro.")
        except Exception:  # noqa: BLE001
            pass
        self.kokoro.set_execution_providers(providers)

    def speak(self, text: str, out_wav: str | None = None) -> tuple[str, int]:
        """Return (wav_path, sample_rate). Writes the synthesized speech to out_wav."""
        if out_wav is None:
            out_wav = os.path.join(self.cfg.models_dir, "last_spoken.wav")
        audio, sr = self.kokoro.create(
            text, voice=self.cfg.tts_voice, speed=self.cfg.tts_speed,
            lang="en-us",
        )
        audio = np.asarray(audio, dtype=np.float32)
        sf.write(out_wav, audio, sr)
        return out_wav, int(sr)


if __name__ == "__main__":
    from src.config import load_config
    tts = TTS(load_config())
    p, sr = tts.speak("Hello team, I'm your AI avatar. Nice to meet you.")
    print(f"[tts] wrote {p} ({sr} Hz)")
