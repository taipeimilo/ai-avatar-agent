"""Lightweight mouth-animation renderer (clean, real-time, always works).

Draws an animated mouth synced to the audio envelope. Uses SCRFD to locate the
face (and its mouth landmarks) so the mouth lands in the right place on
arbitrary photos. The photo's own (often smiling/static) mouth is masked out
with surrounding skin tone and a fresh, clearly animating mouth is drawn on top
so the result reads as "talking" rather than a static smile.
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
        self.face_box, self.landmarks = self._detect()
        self._skin = self._sample_skin()

    def _detect(self):
        det_path = os.path.join(
            self.cfg.models_dir, "wav2lip", "insightface_func", "models",
            "antelope", "scrfd_2.5g_bnkps.onnx")
        try:
            from src.face.scrfd import SCRFDetector
            det = SCRFDetector(det_path, providers=["CPUExecutionProvider"])
            res = det.detect(self.img)
            if res:
                return res["box"], res.get("landmarks")
        except Exception as e:  # noqa: BLE001
            print(f"[lightweight] face detect failed ({e}); center-mouth fallback.")
        h, w = self.img.shape[:2]
        return (int(w * 0.3), int(h * 0.35), int(w * 0.7), int(h * 0.7)), None

    def _sample_skin(self):
        """Average skin tone from a band just above the mouth (upper cheeks)."""
        x1, y1, x2, y2 = self.face_box
        cx = (x1 + x2) // 2
        band = self.img[int(y1 + (y2 - y1) * 0.55):int(y1 + (y2 - y1) * 0.62),
                         cx - (x2 - x1) // 4:cx + (x2 - x1) // 4]
        if band.size == 0:
            return tuple(int(c) for c in self.img.mean(axis=(0, 1)))
        return tuple(int(c) for c in band.reshape(-1, 3).mean(axis=0))

    def _mouth_anchor(self):
        x1, y1, x2, y2 = self.face_box
        if self.landmarks is not None and self.landmarks.shape[0] >= 5:
            # landmarks order: left-eye, right-eye, nose, left-mouth, right-mouth
            lm = self.landmarks
            mouth_l, mouth_r = lm[3], lm[4]
            cx = int((mouth_l[0] + mouth_r[0]) / 2)
            # place slightly below the eye/nose midpoint; use mouth landmark line
            cy = int((mouth_l[1] + mouth_r[1]) / 2)
            mw = int(max(8, abs(mouth_r[0] - mouth_l[0]) * 1.05))
            return cx, cy, mw
        # fallback: centered in lower face
        cx = (x1 + x2) // 2
        cy = int(y1 + (y2 - y1) * 0.72)
        mw = int((x2 - x1) * 0.22)
        return cx, cy, mw

    def render_audio(self, audio_wav, sample_rate):
        import soundfile as sf
        y, sr = sf.read(audio_wav)
        if y.ndim > 1:
            y = y.mean(axis=1)
        hop = max(1, int(sr / cfg_fps(self.cfg)))
        n_frames = max(1, int(len(y) / hop))
        env = np.abs(y[: n_frames * hop].reshape(n_frames, -1).mean(axis=1))
        if env.max() > 0:
            env = env / env.max()
        for frame_idx in range(n_frames):
            e = float(env[frame_idx])
            speaking = e > 0.15
            frame = self._draw_mouth(self.img.copy(), speaking, e)
            yield frame, speaking

    def _draw_mouth(self, frame, speaking, amount):
        cx, cy, mw = self._mouth_anchor()
        fw = self.face_box[2] - self.face_box[0]
        # vertical opening driven by envelope (clearly open when speaking)
        mh = int(3 + amount * 34)
        skin = self._skin

        # 1) Mask out the photo's original (static) mouth with a soft skin patch
        mask_h = int(fw * 0.34)
        mask_w = int(fw * 0.46)
        y0, y1 = cy - mask_h // 2, cy + mask_h // 2
        x0, x1 = cx - mask_w // 2, cx + mask_w // 2
        y0, y1 = max(0, y0), min(frame.shape[0], y1)
        x0, x1 = max(0, x0), min(frame.shape[1], x1)
        if y1 > y0 and x1 > x0:
            patch = np.array(skin, dtype=np.uint8) * np.ones(
                (y1 - y0, x1 - x0, 3), dtype=np.uint8)
            sub = frame[y0:y1, x0:x1].astype(np.float32)
            # blend so it doesn't look like a hard rectangle
            w = np.hanning(y1 - y0)[:, None] * np.hanning(x1 - x0)[None, :]
            w = w[:, :, None]
            frame[y0:y1, x0:x1] = (sub * (1 - w) + patch * w).astype(np.uint8)

        # 2) Draw the animating mouth on top
        # Lip line follows the opening; interior shade shifts with openness.
        fill = (30, 24, 26) if speaking else (70, 62, 62)
        cv2.ellipse(frame, (cx, cy), (mw, max(3, mh)), 0, 0, 360, fill, -1)
        # lip line
        cv2.ellipse(frame, (cx, cy), (mw, max(3, mh)), 0, 0, 360, (15, 12, 12), 3)
        # upper lip hint so a closed mouth still reads as a mouth
        cv2.ellipse(frame, (cx, cy - max(3, mh) - 2), (mw - 4, 3), 0, 0, 360,
                    (45, 38, 38), -1)
        # teeth glint when clearly open
        if speaking and amount > 0.35:
            cv2.ellipse(frame, (cx, cy - mh // 3), (mw - 6, max(2, mh // 3)),
                        0, 0, 360, (225, 220, 220), -1)
        return frame
