"""
Audio OUTPUT for the avatar.

Routes synthesized speech to a chosen playback device so it can be captured by
a meeting app. For "avatar is heard in Teams" we target the VB-Audio Virtual
Cable's INPUT (which appears to Teams as a microphone named "CABLE Input").

Falls back to the system default output if the cable isn't installed yet, so the
avatar is never silent-broken.
"""
from __future__ import annotations

import os
import numpy as np
import sounddevice as sd
import soundfile as sf

CABLE_NAME = "CABLE Input"  # VB-Audio Virtual Cable input (Teams "microphone")


def find_device(name_substr: str) -> int | None:
    """Return the output device index whose name contains `name_substr`, else None."""
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if name_substr.lower() in d["name"].lower() and d["max_output_channels"] > 0:
            return i
    return None


def play_wav(wav_path: str, device: str | int | None = None, block: bool = True):
    """Play a WAV file to `device` (name substring or index). None = cable-or-default.

    Returns the device index actually used.
    """
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim == 1:
        data = data[:, None]  # (N,1) -> ensure 2D for sounddevice

    if device is None:
        device = os.getenv("AVATAR_AUDIO_OUT", CABLE_NAME)
    idx = device if isinstance(device, int) else find_device(str(device))
    used_default = False
    if idx is None:
        idx = sd.default.device[1]  # system default output
        used_default = True
    try:
        sd.play(data, sr, device=idx, blocking=block)
    except Exception as e:  # noqa: BLE001
        if not used_default:
            print(f"[audio] device {device} failed ({e}); using default output.")
            sd.play(data, sr, device=sd.default.device[1], blocking=block)
        else:
            raise
    return idx


def cable_available() -> bool:
    return find_device(CABLE_NAME) is not None


if __name__ == "__main__":
    print("CABLE Input available:", cable_available())
    d = find_device(CABLE_NAME)
    print("cable device index:", d, "(None = install VB-Audio Virtual Cable)")
