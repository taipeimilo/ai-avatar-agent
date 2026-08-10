"""
Speech-to-text input via faster-whisper (local, CPU/DirectML friendly).

Captures microphone audio in short blocks, transcribes, and yields text
segments. Used to let the avatar join a live call hands-free (step C).
"""
from __future__ import annotations

import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

from src.config import Config


class Listener:
    def __init__(self, cfg: Config, model_size: str = "base", device: str = "cpu"):
        self.cfg = cfg
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.blocksize = 16000  # 1s of 16kHz audio
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        self.q.put(indata.copy())

    def start(self):
        self.stream = sd.InputStream(
            callback=self._callback, channels=1, samplerate=16000,
            blocksize=self.blocksize, dtype="float32",
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def listen_once(self, timeout: float = 8.0) -> str:
        """Collect audio for `timeout` seconds (or until queue drains) and transcribe."""
        import io, soundfile as sf
        if self.stream is None:
            self.start()
        chunks = []
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                chunks.append(self.q.get(timeout=1.0))
            except queue.Empty:
                if chunks:
                    break
        if not chunks:
            return ""
        audio = np.concatenate(chunks, axis=0).flatten()
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        buf.seek(0)
        segments, _ = self.model.transcribe(buf.read(), language="en")
        return " ".join(s.text for s in segments).strip()


if __name__ == "__main__":
    from src.config import load_config
    l = Listener(load_config())
    print("Listening 6s... speak now.")
    txt = l.listen_once(timeout=6.0)
    print("You said:", txt)
    l.stop()
