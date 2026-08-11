"""Lightweight mouth-animation renderer (clean, real-time, always works).

Draws an animated mouth synced to the audio envelope. It letterboxes the avatar
photo (no stretching) and draws a clean, realistically-sized mouth on top,
driven by the audio RMS, so the result reads as "talking" without distorting the
original headshot.
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

    def _detect(self):
        # Find the HEAD (not the whole body) from warm skin pixels in the upper
        # half of the image, so the mouth anchor lands on the real face even
        # when the subject is off-center or the photo includes shoulders.
        h, w = self.img.shape[:2]
        b, g, r = (self.img[:, :, c].astype(int) for c in range(3))
        warm = (r > g) & (g > b) & (r > 90) & (r - b > 25)
        # keep only the upper 55% of the image where the head sits
        upper = warm.copy()
        upper[h // 2:, :] = False
        coords = np.where(upper)
        if coords[0].size < 300:
            # fall back to whole-image warm bbox
            coords = np.where(warm)
        if coords[0].size < 300:
            return (int(w * 0.3), int(h * 0.35), int(w * 0.7), int(h * 0.7)), None
        x0, x1 = int(coords[1].min()), int(coords[1].max())
        y0, y1 = int(coords[0].min()), int(coords[0].max())
        # The head bbox; the mouth sits in the lower-middle of the head.
        return (x0, y0, x1, y1), None

    def _mouth_anchor(self):
        x1, y1, x2, y2 = self.face_box
        # mouth sits ~85% down the detected head bbox (top of bbox = hairline,
        # so the face occupies the lower portion). Verified against this headshot
        # where the real mouth center is ~canvas y=290.
        cx = (x1 + x2) // 2
        cy = int(y1 + (y2 - y1) * 0.85)
        mw = int((x2 - x1) * 0.16)      # realistic mouth width (~16% of head)
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

    def render_idle(self, blink: bool = False):
        """A neutral idle frame. When `blink` is True, briefly draws closed
        eyelids so the avatar reads as a live person (not a frozen image)."""
        frame = self.img.copy()
        x1, y1, x2, y2 = self.face_box
        cx = (x1 + x2) // 2
        fw = x2 - x1
        fh = y2 - y1
        ey = int(y1 + fh * 0.40)          # eye line height
        dx = int(fw * 0.22)               # half-distance between eyes
        ew = int(fw * 0.13)               # eye width
        if blink:
            for ex in (cx - dx, cx + dx):
                cv2.line(frame, (ex - ew, ey), (ex + ew, ey), (35, 28, 30), 3,
                         cv2.LINE_AA)
        return frame

    def _draw_mouth(self, frame, speaking, amount):
        cx, cy, mw = self._mouth_anchor()
        mh = int(2 + amount * 10)       # natural opening, driven by audio
        # Clean, small animated mouth drawn directly on the photo's lips. A soft
        # lip/rose tone (not near-black) + thin lip line so it reads as a mouth
        # rather than a dark bar/line.
        fill = (92, 48, 60) if speaking else (112, 62, 72)
        cv2.ellipse(frame, (cx, cy), (mw, max(2, mh)), 0, 0, 360, fill, -1)
        cv2.ellipse(frame, (cx, cy), (mw, max(2, mh)), 0, 0, 360, (40, 22, 28), 1)
        # subtle teeth glint when clearly open
        if speaking and amount > 0.5:
            cv2.ellipse(frame, (cx, cy - mh // 3), (mw - 8, max(2, mh // 3)),
                        0, 0, 360, (225, 220, 220), -1)
        return frame
