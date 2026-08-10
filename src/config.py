"""
Configuration for the AI Avatar Agent.
All tunable knobs live here. Override via environment variables if you like.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- Agent / brain ----
    # Local Ollama model (install with `ollama pull <model>`).
    # "qwen2.5:3b" is a good small/fast default for an APU; use "llama3.1:8b" for smarter.
    ollama_model: str = os.getenv("AVATAR_MODEL", "qwen2.5:3b")
    ollama_base_url: str = os.getenv("AVATAR_OLLAMA_URL", "http://localhost:11434")
    ollama_enabled: bool = os.getenv("AVATAR_USE_OLLAMA", "1") == "1"
    # Optional cloud LLM fallback (OpenAI-compatible). Leave blank to use local only.
    api_base_url: str = os.getenv("AVATAR_API_URL", "")
    api_key: str = os.getenv("AVATAR_API_KEY", "")
    api_model: str = os.getenv("AVATAR_API_MODEL", "")
    system_prompt: str = os.getenv(
        "AVATAR_SYSTEM_PROMPT",
        "You are MiloBot, a friendly, concise AI teammate who joins calls as a "
        "talking avatar. Keep spoken replies short (1-3 sentences) and natural.",
    )

    # ---- Voice (TTS) ----
    tts_voice: str = os.getenv("AVATAR_VOICE", "af_sarah")  # Kokoro voice id
    tts_speed: float = float(os.getenv("AVATAR_TTS_SPEED", "1.0"))

    # ---- Face render ----
    # "auto" tries DirectML (AMD GPU) then falls back to CPU.
    face_backend: str = os.getenv("AVATAR_FACE_BACKEND", "auto")  # auto|directml|cpu
    face_model: str = os.getenv(
        "AVATAR_FACE_MODEL", "wav2lip"
    )  # wav2lip|liveportrait|musetalk
    avatar_image: str = os.getenv(
        "AVATAR_IMAGE", os.path.join("assets", "avatar.png")
    )

    # ---- Camera / output ----
    camera_backend: str = os.getenv("AVATAR_CAMERA", "obs")  # obs|pyvirtualcam
    camera_width: int = int(os.getenv("AVATAR_W", "1280"))
    camera_height: int = int(os.getenv("AVATAR_H", "720"))
    camera_fps: int = int(os.getenv("AVATAR_FPS", "24"))

    # ---- Paths ----
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir: str = field(default="")
    assets_dir: str = field(default="")

    def __post_init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models")
        self.assets_dir = os.path.join(self.project_root, "assets")
        if not os.path.isabs(self.avatar_image):
            self.avatar_image = os.path.join(self.assets_dir, os.path.basename(self.avatar_image))


def load_config() -> Config:
    return Config()
