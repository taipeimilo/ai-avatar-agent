"""
Agent "brain": turns a user text input into the avatar's spoken reply.

Tries a local Ollama model first (privacy, no network), then falls back to an
OpenAI-compatible cloud API if configured and Ollama is unavailable.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional

from src.config import Config


class Brain:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history: list[dict] = []

    def _ollama_chat(self, prompt: str) -> Optional[str]:
        url = f"{self.cfg.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.cfg.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.cfg.system_prompt},
                *self.history,
                {"role": "user", "content": prompt},
            ],
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            return data.get("message", {}).get("content", "").strip()
        except Exception as e:  # noqa: BLE001
            print(f"[brain] Ollama unavailable: {e}")
            return None

    def _api_chat(self, prompt: str) -> Optional[str]:
        if not (self.cfg.api_base_url and self.cfg.api_key and self.cfg.api_model):
            return None
        url = f"{self.cfg.api_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.cfg.api_model,
            "messages": [
                {"role": "system", "content": self.cfg.system_prompt},
                *self.history,
                {"role": "user", "content": prompt},
            ],
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.cfg.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            print(f"[brain] API chat failed: {e}")
            return None

    def reply(self, user_text: str) -> str:
        out = None
        if self.cfg.ollama_enabled:
            out = self._ollama_chat(user_text)
        if out is None:
            out = self._api_chat(user_text)
        if out is None:
            out = "Sorry, I couldn't reach my brain right now."
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": out})
        # keep history bounded
        if len(self.history) > 20:
            self.history = self.history[-20:]
        return out


if __name__ == "__main__":
    from src.config import load_config
    b = Brain(load_config())
    # quick self-test
    print("reply ->", b.reply("Say hello in one sentence."))
