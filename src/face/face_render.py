"""
Face-render layer: drives the avatar image with audio -> talking-head frames.

This is the GPU-dependent piece. Strategy:
  - Try LivePortrait (image + audio driven) via its ONNX/DirectML path.
  - Fall back to a lightweight "mouth animation" renderer (CPU) if the heavy
    model can't load on this machine. Either way, downstream code gets frames.

Backends are swappable: set AVATAR_FACE_MODEL=liveportrait|musetalk and
AVATAR_FACE_BACKEND=auto|directml|cpu.
"""
from __future__ import annotations

import os
import numpy as np
from abc import ABC, abstractmethod

from src.config import Config


class FaceRenderer(ABC):
    @abstractmethod
    def render_audio(self, audio_wav: str, sample_rate: int):
        """Yield (frame_bgr_numpy, is_speaking) frames synchronized to audio."""
        raise NotImplementedError

    @staticmethod
    def pick(cfg: Config) -> "FaceRenderer":
        backend = cfg.face_backend
        if backend == "auto":
            try:
                import onnxruntime as ort
                if "DmlExecutionProvider" in ort.get_available_providers():
                    backend = "directml"
                else:
                    backend = "cpu"
            except Exception:  # noqa: BLE001
                backend = "cpu"
        print(f"[face] backend = {backend}")
        # LivePortrait is the quality choice; we wrap it to fall back gracefully.
        try:
            from src.face.liveportrait import LivePortraitRenderer
            return LivePortraitRenderer(cfg, backend)
        except Exception as e:  # noqa: BLE001
            print(f"[face] LivePortrait unavailable ({e}); using lightweight renderer.")
            from src.face.lightweight import LightweightRenderer
            return LightweightRenderer(cfg, backend)


class LightweightRenderer(FaceRenderer):
    """CPU-only fallback: animates a simple mouth opening/closing to audio RMS.

    Not photorealistic, but always works and proves the pipeline end-to-end.
    """

    def __init__(self, cfg: Config, backend: str):
        self.cfg = cfg
        self.backend = backend
        self.img = _load_avatar(cfg.avatar_image, cfg.camera_width, cfg.camera_height)

    def render_audio(self, audio_wav, sample_rate):
        import soundfile as sf
        import librosa  # for envelope
        y, sr = sf.read(audio_wav)
        if y.ndim > 1:
            y = y.mean(axis=1)
        # frame-level energy envelope
        hop = int(sr / cfg_fps(self.cfg))
        env = np.abs(y[::max(1, hop // 4)])
        env = env / (env.max() + 1e-6)
        i = 0
        frame_idx = 0
        while i < len(y):
            e = float(env[min(len(env) - 1, frame_idx)])
            speaking = e > 0.15
            frame = self._draw_mouth(self.img.copy(), speaking, e)
            yield frame, speaking
            i += hop
            frame_idx += 1

    def _draw_mouth(self, frame, speaking, amount):
        # simple overlay: darken a mouth ellipse when speaking
        h, w = frame.shape[:2]
        cx, cy = w // 2, int(h * 0.62)
        import cv2
        r = int(h * 0.04 * (1.0 + amount * 1.2))
        color = (20, 20, 20) if speaking else (60, 60, 60)
        cv2.ellipse(frame, (cx, cy), (int(w * 0.06), r), 0, 0, 360, color, -1)
        return frame


def _load_avatar(path: str, w: int, h: int):
    import cv2
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (w, h))
    return img


def cfg_fps(cfg: Config) -> int:
    return cfg.camera_fps
