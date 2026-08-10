"""Lightweight CPU mouth-animation fallback (see face_render.py)."""
from __future__ import annotations

import os
import numpy as np
import cv2
from src.face.face_render import FaceRenderer, _load_avatar, cfg_fps


class LightweightRenderer(FaceRenderer):
    def __init__(self, cfg, backend: str):
        self.cfg = cfg
        self.backend = backend
        self.img = _load_avatar(cfg.avatar_image, cfg.camera_width, cfg.camera_height)

    def render_audio(self, audio_wav, sample_rate):
        import soundfile as sf
        y, sr = sf.read(audio_wav)
        if y.ndim > 1:
            y = y.mean(axis=1)
        hop = max(1, int(sr / cfg_fps(self.cfg)))
        env = np.abs(y[:: max(1, hop // 4)])
        if env.max() > 0:
            env = env / env.max()
        frame_idx = 0
        i = 0
        while i < len(y):
            e = float(env[min(len(env) - 1, frame_idx)])
            speaking = e > 0.15
            frame = self._draw_mouth(self.img.copy(), speaking, e)
            yield frame, speaking
            i += hop
            frame_idx += 1

    def _draw_mouth(self, frame, speaking, amount):
        h, w = frame.shape[:2]
        cx, cy = w // 2, int(h * 0.62)
        r = int(h * 0.04 * (1.0 + amount * 1.2))
        color = (20, 20, 20) if speaking else (60, 60, 60)
        cv2.ellipse(frame, (cx, cy), (int(w * 0.06), r), 0, 0, 360, color, -1)
        return frame
