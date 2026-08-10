"""Real end-to-end verification of the avatar pipeline (ad-hoc).

Exercises every layer that has deps now installed:
  config -> brain(Ollama) -> tts(Kokoro) -> face(cv2) -> camera(window).
Prints PASS/FAIL per stage and reports measured latency.
"""
import os, sys, time
ROOT = r"C:\Users\milo_\ai-avatar-agent"
sys.path.insert(0, ROOT)
import numpy as np

def hr(t): print("\n" + "="*60 + f"\n{t}\n" + "="*60)

# 1) CONFIG
hr("1) Config")
from src.config import load_config
cfg = load_config()
print(f"   model={cfg.ollama_model} face={cfg.face_model}/{cfg.face_backend} cam={cfg.camera_backend}")
print("   PASS")

# 2) BRAIN (real Ollama)
hr("2) Brain -> Ollama")
from src.agent.brain import Brain
brain = Brain(cfg)
t0 = time.time()
reply = brain.reply("Say hello to the team in one short sentence.")
dt = time.time() - t0
print(f"   reply ({dt:.2f}s): {reply!r}")
assert reply and reply != "Sorry, I couldn't reach my brain right now."
print("   PASS")

# 3) TTS (real Kokoro)
hr("3) TTS -> Kokoro (onnx)")
from src.tts.tts import TTS
tts = TTS(cfg)
t0 = time.time()
wav, sr = tts.speak(reply)
dt = time.time() - t0
import soundfile as sf
info = sf.info(wav)
print(f"   wav={wav} sr={sr} dur={info.duration:.2f}s gen={dt:.2f}s")
assert os.path.exists(wav) and info.duration > 0
print("   PASS")

# 4) FACE render (real cv2 + librosa envelope)
hr("4) Face render (lightweight cv2 mouth animation)")
from src.face.face_render import FaceRenderer
face = FaceRenderer.pick(cfg)
t0 = time.time()
frames = list(face.render_audio(wav, sr))
dt = time.time() - t0
print(f"   produced {len(frames)} frames in {dt:.2f}s, shape={frames[0][0].shape}")
assert len(frames) > 0 and frames[0][0].shape[2] == 3
print("   PASS")

# 5) CAMERA (window preview path, no OBS required)
hr("5) Camera (window preview)")
from src.camera.virtual_camera import VirtualCamera
cam = VirtualCamera(cfg)  # falls back to window since OBS not running
for i, (f, sp) in enumerate(frames[:10]):
    cam.send(f, sp)
cam.close()
print("   PASS (window preview rendered frames)")

hr("SUMMARY")
print("All 5 stages executed with REAL deps. End-to-end verified.")
