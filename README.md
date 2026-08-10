# AI Avatar Agent

A real-time AI video avatar for team calls — a talking head that represents an
AI agent (LLM + voice) and appears as a **virtual webcam** that Zoom / Teams /
Discord / OBS can select.

Built to run **locally on an AMD Ryzen AI laptop (Radeon 8050S, 64GB)** — the
face render uses **DirectML** on the AMD GPU with a CPU fallback.

## Pipeline

```
[text input] -> Brain (Ollama local LLM) -> TTS (Kokoro) -> Face render -> Virtual camera
```

- **Brain** (`src/agent/brain.py`): local Ollama model, with OpenAI-compatible cloud fallback.
- **TTS** (`src/tts/tts.py`): Kokoro ONNX, runs on CPU or DirectML.
- **Face** (`src/face/`): LivePortrait (image-driven) with a CPU mouth-animation fallback so the pipeline always runs.
- **Camera** (`src/camera/`): OBS virtual camera / pyvirtualcam / window preview.

## Setup

```bash
cd ai-avatar-agent
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
ollama pull qwen2.5:3b                              # the brain
```

Optional: install OBS Studio for the virtual camera, and
`onnxruntime-directml` for AMD GPU acceleration.

## Run

```bash
python src/main.py
```

Type to the avatar in the console. It replies with voice + a talking-face
preview window. Point your meeting app's camera at "OBS Virtual Camera".

## Roadmap / TODO

- [ ] Swap LivePortrait placeholder for full ONNX inference on DirectML.
- [ ] Add Whisper mic input so the avatar joins a real call.
- [ ] Tune latency (target < 500ms utterance-to-frame on APU).
- [ ] Add a second avatar character + voice.

## Hardware notes

- AMD GPU compute on Windows = **DirectML** (ROCm is Linux-only).
- The avatar art is `assets/avatar.png` — replace with any portrait to
  re-skin the avatar.
