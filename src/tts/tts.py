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
from kokoro_onnx import Kokoro

from src.config import Config


class TTS:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.models_dir, exist_ok=True)
        kokoro_dir = os.path.join(cfg.models_dir, "kokoro")
        os.makedirs(kokoro_dir, exist_ok=True)
        model_path = os.path.join(kokoro_dir, "kokoro-v1.0.onnx")
        voices_path = os.path.join(kokoro_dir, "voices-v1.0.bin")
        # Kokoro loads voices via np.load(); the release ships voices-v1.0.bin.
        if not os.path.exists(voices_path):
            voices_path = os.path.join(kokoro_dir, "voices.json")
        # Kokoro auto-downloads to these paths if missing.
        # It selects the best available execution provider automatically
        # (DmlExecutionProvider if onnxruntime-directml is installed -> AMD GPU).
        self.kokoro = Kokoro(model_path, voices_path)

    def speak(self, text: str, out_wav: str | None = None) -> tuple[str, int]:
        """Return (wav_path, sample_rate). Writes the synthesized speech to out_wav."""
        if out_wav is None:
            out_wav = os.path.join(self.cfg.models_dir, "last_spoken.wav")
        audio, sr = self.kokoro.create(
            text, voice=self.cfg.tts_voice, speed=self.cfg.tts_speed,
        )
        audio = np.asarray(audio, dtype=np.float32)
        sf.write(out_wav, audio, sr)
        return out_wav, int(sr)


if __name__ == "__main__":
    from src.config import load_config
    tts = TTS(load_config())
    p, sr = tts.speak("Hello team, I'm your AI avatar. Nice to meet you.")
    print(f"[tts] wrote {p} ({sr} Hz)")
