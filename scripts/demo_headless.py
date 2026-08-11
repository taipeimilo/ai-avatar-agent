"""Headless demo: run the real ai-avatar pipeline and render an animated GIF.

This exercises the actual production code paths:
  text -> Brain(ollama qwen2.5:3b) -> TTS(Kokoro) -> LightweightRenderer -> GIF

No window / no OBS needed, so it runs from a shell. Output is a looping GIF you
can open or drop into a chat to show the avatar "talking".

Run:
  python scripts/demo_headless.py
Optional args:
  --prompt "your line for the avatar"
  --out    demo.gif
  --voice  af_sarah
"""
from __future__ import annotations
import os
import sys
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import cv2

from src.config import load_config
from src.agent.brain import Brain
from src.tts.tts import TTS
from src.face.lightweight import LightweightRenderer


def _frames_to_gif(frames, path, fps, loop=True):
    """frames: list[(bgr_h_w_c uint8)]. Write looping GIF (RGB, dithered)."""
    if not frames:
        raise RuntimeError("no frames produced")
    # Resize to a web-friendly width, keep aspect.
    h, w = frames[0].shape[:2]
    target_w = 480
    scale = target_w / w
    tw, th = target_w, max(1, int(h * scale))
    pil_frames = []
    for f in frames:
        rgb = cv2.cvtColor(cv2.resize(f, (tw, th)), cv2.COLOR_BGR2RGB)
        pil_frames.append(rgb)
    try:
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"PIL required for GIF: {e}")
    imgs = [Image.fromarray(a) for a in pil_frames]
    dur_ms = max(20, int(round(1000 / fps)))
    imgs[0].save(
        path, save_all=True, append_images=imgs[1:],
        duration=dur_ms, loop=0 if loop else 1, optimize=True,
    )
    return path, (tw, th), len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None,
                    help="Line for the avatar to say (default: auto from brain).")
    ap.add_argument("--out", default=os.path.join(ROOT, "demo_avatar.gif"))
    ap.add_argument("--voice", default=None, help="Kokoro voice id.")
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    cfg = load_config()
    if args.voice:
        cfg.tts_voice = args.voice

    print("=== AI Avatar — headless demo ===")
    print(f"[config] model={cfg.ollama_model} voice={cfg.tts_voice} "
          f"face={cfg.face_model} backend={cfg.face_backend}")

    # 1) Brain
    t0 = time.time()
    brain = Brain(cfg)
    user_prompt = args.prompt or "Greet the team and say what you are in one sentence."
    reply = brain.reply(user_prompt)
    print(f"[brain] ({time.time()-t0:.2f}s) avatar says: {reply!r}")

    # 2) TTS
    t0 = time.time()
    tts = TTS(cfg)
    wav, sr = tts.speak(reply)
    print(f"[tts]   ({time.time()-t0:.2f}s) wrote {wav} ({sr} Hz)")

    # 3) Face render (lightweight mouth animation -> frames)
    t0 = time.time()
    face = LightweightRenderer.pick(cfg) if hasattr(LightweightRenderer, "pick") \
        else LightweightRenderer(cfg, cfg.face_backend)
    frames = []
    speaking_frames = 0
    for frame, speaking in face.render_audio(wav, sr):
        frames.append(frame)
        speaking_frames += int(speaking)
    print(f"[face]  ({time.time()-t0:.2f}s) {len(frames)} frames "
          f"({speaking_frames} speaking)")

    # 4) GIF
    t0 = time.time()
    path, size, n = _frames_to_gif(frames, args.out, args.fps)
    print(f"[gif]   ({time.time()-t0:.2f}s) wrote {path} "
          f"({size[0]}x{size[1]}, {n} frames)")
    print("\nDone. Open the GIF to see the avatar talking.")
    return path


if __name__ == "__main__":
    out = main()
    print(f"GIF_PATH={out}")
