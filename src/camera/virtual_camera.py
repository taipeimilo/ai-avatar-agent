"""
Virtual camera output. Makes the rendered avatar appear as a real webcam that
Zoom / Teams / Discord / OBS can select.

Two backends:
  - "obs":  control OBS Studio's Virtual Camera (start it; OBS does the heavy
            lifting of exposing the camera). Best on Windows.
  - "pyvirtualcam": a pure-python virtual webcam (needs a virtual-cam driver such
            as OBS's or a v4l2loopback-equivalent; on Windows we use OBS).

We also expose a simple window preview (OpenCV) so you can see the avatar even
without a virtual camera configured.
"""
from __future__ import annotations

import os
import time
import subprocess
import numpy as np
import cv2

from src.config import Config


class VirtualCamera:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.width, self.height, self.fps = cfg.camera_width, cfg.camera_height, cfg.camera_fps
        self.obs = None
        self.vc = None
        self.use_window = False
        if cfg.camera_backend == "obs":
            self._init_obs()
        elif cfg.camera_backend == "pyvirtualcam":
            self._init_pyvirtualcam()
        else:
            self.use_window = True

    def _init_obs(self):
        try:
            import obswebsocket
            from obswebsocket import obsws, requests as obsreq
        except Exception as e:  # noqa: BLE001
            print(f"[cam] obs-websocket not installed ({e}); falling back to window preview.")
            self.use_window = True
            return
        # Try to launch OBS if not running, then connect.
        self.obs_ws = obsws("localhost", 4455, os.getenv("OBS_PASSWORD", ""))
        try:
            self.obs_ws.connect()
            self.obs_ws.call(obsreq.StartVirtualCamera())
            print("[cam] OBS virtual camera started.")
        except Exception as e:  # noqa: BLE001
            print(f"[cam] OBS not reachable ({e}); falling back to window preview.")
            self.use_window = True

    def _init_pyvirtualcam(self):
        try:
            import pyvirtualcam
            # pyvirtualcam auto-detects the OBS Virtual Camera backend on Windows.
            self.vc = pyvirtualcam.Camera(
                width=self.width, height=self.height, fps=self.fps,
                backend="obs",
            )
            print(f"[cam] OBS virtual camera ready: {self.vc.device}")
        except Exception as e:  # noqa: BLE001
            print(f"[cam] OBS virtual camera unavailable ({e}); window preview only.")
            self.use_window = True

    def send(self, frame_bgr: np.ndarray, speaking: bool = False):
        if self.vc is not None:
            # pyvirtualcam wants RGB
            self.vc.send(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self.use_window or self.vc is None:
            disp = frame_bgr.copy()
            label = "● speaking" if speaking else "○ idle"
            cv2.putText(disp, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
            cv2.imshow("AI Avatar", disp)
            cv2.waitKey(1)

    def close(self):
        if self.vc is not None:
            self.vc.close()
        if hasattr(self, "obs_ws"):
            try:
                self.obs_ws.disconnect()
            except Exception:  # noqa: BLE001
                pass
        cv2.destroyAllWindows()
