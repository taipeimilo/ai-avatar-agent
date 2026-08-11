"""
Virtual camera output. Makes the rendered avatar appear as a real webcam that
Zoom / Teams / Discord can select.

PRIMARY approach (reliable, proven VTuber method):
  Run the avatar in a visible window named "AI Avatar". OBS Studio captures that
  window via a "Window Capture" source and exposes it through its Virtual Camera.
  This avoids pyvirtualcam's flaky OBS-backend frame delivery and also gives the
  user a live local preview.

FALLBACK: if OBS is not in use, we can still push frames via pyvirtualcam, but
the OBS-compositor path is what we recommend (and what the .bat wires up).

Either way, the "AI Avatar" window is always shown so the user sees the avatar
and (when OBS is running) OBS can grab it.
"""
from __future__ import annotations

import os
import time
import numpy as np
import cv2

from src.config import Config

WINDOW_NAME = "AI Avatar"


class VirtualCamera:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.width, self.height, self.fps = cfg.camera_width, cfg.camera_height, cfg.camera_fps
        self.vc = None
        # Try pyvirtualcam as a secondary direct device (optional). If it fails,
        # we rely solely on the window + OBS compositor.
        self.use_pyvirtualcam = False
        if cfg.camera_backend == "pyvirtualcam":
            try:
                import pyvirtualcam
                self.vc = pyvirtualcam.Camera(
                    width=self.width, height=self.height, fps=self.fps, backend="obs")
                self.use_pyvirtualcam = True
                print(f"[cam] OBS virtual camera device ready: {self.vc.device}")
            except Exception as e:  # noqa: BLE001
                print(f"[cam] pyvirtualcam unavailable ({e}); using OBS Window Capture path.")
                self.vc = None

    def send(self, frame_bgr: np.ndarray, speaking: bool = False):
        # Always show the window (OBS captures it + user sees a preview).
        disp = frame_bgr.copy()
        label = "AI Avatar  -  speaking" if speaking else "AI Avatar  -  idle"
        cv2.rectangle(disp, (0, 0), (self.width, 40), (0, 0, 0), -1)
        cv2.putText(disp, label, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 200, 255), 2)
        cv2.imshow(WINDOW_NAME, disp)
        cv2.waitKey(1)
        # Optionally also push to the pyvirtualcam device (secondary).
        if self.vc is not None:
            try:
                self.vc.send(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            except Exception:  # noqa: BLE001
                pass

    def close(self):
        if self.vc is not None:
            try:
                self.vc.close()
            except Exception:  # noqa: BLE001
                pass
        cv2.destroyAllWindows()
