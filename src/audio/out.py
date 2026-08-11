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


def find_device(name_substr: str, prefer_channels: int = 2) -> int | None:
    """Return the output device index whose name contains `name_substr`, else None.

    Prefers the standard 2-channel device (the one Teams actually exposes as a
    "microphone") over 16-channel variants from Voicemeeter/Hi-Fi Cable, so we
    don't send audio to a cable your meeting app isn't listening to.
    """
    devs = sd.query_devices()
    candidates = [
        (i, d) for i, d in enumerate(devs)
        if name_substr.lower() in d["name"].lower() and d["max_output_channels"] > 0
    ]
    if not candidates:
        return None
    for i, d in candidates:
        if d["max_output_channels"] == prefer_channels:
            return i
    return candidates[0][0]


def _safe_play(data, sr, idx: int, block: bool):
    try:
        sd.play(data, sr, device=idx, blocking=block)
    except Exception as e:  # noqa: BLE001
        default = sd.default.device[1]
        print(f"[audio] device {idx} failed ({e}); falling back to default output.")
        sd.play(data, sr, device=default, blocking=block)


def play_wav(wav_path: str, device: str | int | None = None, block: bool = True,
             monitor: bool | None = None):
    """Play a WAV file to `device` (name substring or index). None = cable-or-default.

    The avatar's voice is sent to the virtual cable so a meeting app (Teams/Zoom)
    hears it when its microphone is set to that cable. By default we ALSO play it
    on the system default speakers (monitoring) so you can hear the avatar on your
    own PC too. Set AVATAR_AUDIO_MONITOR=0 to disable local monitoring.

    Returns the primary device index actually used.
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
    _safe_play(data, sr, idx, block)

    # Local monitoring so the user can hear the avatar on their own speakers.
    mon = AVATAR_AUDIO_MONITOR if monitor is None else monitor
    if mon and not used_default:
        default_out = sd.default.device[1]
        if isinstance(default_out, int) and default_out != idx:
            try:
                sd.play(data, sr, device=default_out, blocking=block)
            except Exception as e:  # noqa: BLE001
                print(f"[audio] monitor (speakers) failed: {e}")
    return idx


# Local-speaker monitoring toggle (set AVATAR_AUDIO_MONITOR=0 to disable).
AVATAR_AUDIO_MONITOR = os.getenv("AVATAR_AUDIO_MONITOR", "1") not in ("0", "false", "no")


def cable_available() -> bool:
    return find_device(CABLE_NAME) is not None


if __name__ == "__main__":
    print("CABLE Input available:", cable_available())
    d = find_device(CABLE_NAME)
    print("cable device index:", d, "(None = install VB-Audio Virtual Cable)")
