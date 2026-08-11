"""
Main entry point for the real-time AI avatar.

Two modes:
  console   : you type, avatar replies (text -> brain -> tts -> face -> preview)
  live      : mic -> Whisper -> brain -> tts -> face -> OBS Virtual Camera
              (the avatar appears as a selectable webcam in Zoom/Teams/Discord)

Run:
  python src/main.py                 # console mode
  python src/main.py --live          # live mic + virtual camera
"""
from __future__ import annotations

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.agent.brain import Brain
from src.agent.listen import Listener
from src.tts.tts import TTS
from src.face.face_render import FaceRenderer
from src.camera.virtual_camera import VirtualCamera
from src.audio.out import play_wav, cable_available


def run_console(cfg):
    brain, tts, face, cam = _build(cfg)
    print("\nType to the avatar (or 'quit'):")
    try:
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user or user.lower() in ("quit", "exit", "q"):
                break
            _turn(brain, tts, face, cam, user)
    finally:
        cam.close()


def run_live(cfg):
    brain, tts, face, cam = _build(cfg)
    listener = Listener(cfg, model_size="base", device="cpu")
    listener.start()
    print("\n[LIVE] Listening... speak, then pause ~1s. Ctrl-C to stop.")
    try:
        while True:
            try:
                user = listener.listen_once(timeout=6.0)
            except KeyboardInterrupt:
                break
            if user:
                print(f"you> {user}")
                _turn(brain, tts, face, cam, user)
    finally:
        listener.stop()
        cam.close()


def _build(cfg):
    brain = Brain(cfg)
    tts = TTS(cfg)
    face = FaceRenderer.pick(cfg)
    cam = VirtualCamera(cfg)
    return brain, tts, face, cam


def _turn(brain, tts, face, cam, user):
    reply = brain.reply(user)
    print(f"avatar> {reply}")
    wav, sr = tts.speak(reply)
    # Play the voice to the virtual cable (Teams mic) so the avatar is HEARD.
    # Non-blocking: the face animation runs in parallel.
    play_wav(wav, block=False)
    t0 = time.time()
    for frame, speaking in face.render_audio(wav, sr):
        cam.send(frame, speaking)
    remaining = _wav_duration(wav) - (time.time() - t0)
    if remaining > 0:
        time.sleep(remaining)


def _wav_duration(wav_path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return float(info.frames / info.samplerate)
    except Exception:  # noqa: BLE001
        return 2.0


def main():
    ap = argparse.ArgumentParser(description="Real-time AI avatar agent")
    ap.add_argument("--live", action="store_true", help="mic + OBS virtual camera")
    args = ap.parse_args()
    cfg = load_config()
    print("=== AI Avatar Agent ===")
    print(f"model={cfg.ollama_model} face={cfg.face_model}/{cfg.face_backend} cam={cfg.camera_backend}")
    try:
        from src.audio.out import cable_available
        if cable_available():
            print("[audio] VB-Audio Virtual Cable detected -> avatar voice will be heard in Teams.")
        else:
            print("[audio] VB-Audio Virtual Cable NOT found -> voice plays on speakers only.")
            print("        Install it (https://vb-audio.com/Cable/) so Teams can capture the avatar's voice.")
    except Exception:  # noqa: BLE001
        pass
    if args.live:
        run_live(cfg)
    else:
        run_console(cfg)


if __name__ == "__main__":
    main()
