"""
Main orchestration loop for the real-time AI avatar.

Flow per turn:
  1. Get input text (typed in console for now; mic/Whisper is a later layer).
  2. Brain -> reply text.
  3. TTS -> audio wav.
  4. Face renderer -> frames synced to audio.
  5. Virtual camera -> shows the talking avatar.
"""
from __future__ import annotations

import os
import sys
import time

# make `src` importable when run as `python src/main.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.agent.brain import Brain
from src.tts.tts import TTS
from src.face.face_render import FaceRenderer
from src.camera.virtual_camera import VirtualCamera


def main():
    cfg = load_config()
    print("=== AI Avatar Agent ===")
    print(f"model={cfg.ollama_model} face={cfg.face_model}/{cfg.face_backend} cam={cfg.camera_backend}")

    brain = Brain(cfg)
    tts = TTS(cfg)
    face = FaceRenderer.pick(cfg)
    cam = VirtualCamera(cfg)

    print("\nType to the avatar (or 'quit'):")
    try:
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                continue
            if user.lower() in ("quit", "exit", "q"):
                break

            reply = brain.reply(user)
            print(f"avatar> {reply}")

            wav, sr = tts.speak(reply)
            audio_dur = _wav_duration(wav)
            t0 = time.time()
            # stream frames in time with the audio
            for frame, speaking in face.render_audio(wav, sr):
                cam.send(frame, speaking)
            # pad the remaining audio duration so the avatar "finishes speaking"
            remaining = audio_dur - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        cam.close()
        print("bye.")


def _wav_duration(wav_path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return float(info.frames / info.samplerate)
    except Exception:  # noqa: BLE001
        return 2.0


if __name__ == "__main__":
    main()
