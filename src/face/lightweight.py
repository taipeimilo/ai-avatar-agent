"""Lightweight mouth-animation renderer (clean, real-time, always works).

Draws an animated mouth ellipse synced to the audio envelope. Uses SCRFD to
locate the face so the mouth lands in the right place on arbitrary photos
(incl. tall portraits), with a center-mouth fallback.
"""
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
        self.face_box = self._detect_box()

    def _detect_box(self):
        det_path = os.path.join(
            self.cfg.models_dir, "wav2lip", "insightface_func", "models",
            "antelope", "scrfd_2.5g_bnkps.onnx")
        try:
            from src.face.scrfd import SCRFDetector
            det = SCRFDetector(det_path, providers=["CPUExecutionProvider"])
            res = det.detect(self.img)
            if res:
                return res["box"]
        except Exception as e:  # noqa: BLE001
            print(f"[lightweight] face detect failed ({e}); center-mouth fallback.")
        h, w = self.img.shape[:2]
        return (int(w * 0.3), int(h * 0.35), int(w * 0.7), int(h * 0.7))

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
        x1, y1, x2, y2 = self.face_box
        cx = (x1 + x2) // 2
        # mouth sits in the lower third of the detected face box
        cy = int(y1 + (y2 - y1) * 0.72)
        fw = x2 - x1
        mw = int(fw * 0.16)
        # clear, visible opening: 4px (closed) -> 26px (wide open) driven by envelope
        mh = int(4 + amount * 22)
        color = (20, 20, 20) if speaking else (55, 55, 55)
        cv2.ellipse(frame, (cx, cy), (mw, max(2, mh)), 0, 0, 360, color, -1)
        # subtle lip line so it reads as a mouth even when nearly closed
        cv2.ellipse(frame, (cx, cy), (mw, max(2, mh)), 0, 0, 360, (10, 10, 10), 2)
        return frame
