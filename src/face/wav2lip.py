"""
Photorealistic talking-head renderer using Wav2Lip-ONNX.

Drives a portrait IMAGE with the user's VOICE -> lip-synced frames.
Runs on the AMD GPU via onnxruntime-directml (DmlExecutionProvider) with CPU
fallback. Weights are in models/wav2lip/ (wav2lip.onnx, scrfd detector).

Exact ONNX contract (verified):
  inputs : mel_spectrogram [B,1,80,16], video_frames [B,6,96,96]
  output : predicted_frames [B,3,96,96]  (RGB, 0..1)
So each output frame consumes 16 mel columns AND a 6-frame window of face crops.
"""
from __future__ import annotations

import os
import numpy as np
import cv2
from src.face.face_render import FaceRenderer, _load_avatar, cfg_fps


class Wav2LipRenderer(FaceRenderer):
    def __init__(self, cfg, backend: str):
        self.cfg = cfg
        self.backend = backend
        self.img = _load_avatar(cfg.avatar_image, cfg.camera_width, cfg.camera_height)
        self.providers = (
            ["DmlExecutionProvider", "CPUExecutionProvider"]
            if backend == "directml" else ["CPUExecutionProvider"]
        )
        self._load()
        self.face, self.box = self._detect_face(self.img)

    def _load(self):
        import onnxruntime as ort
        d = os.path.join(self.cfg.models_dir, "wav2lip")
        self.w2l = ort.InferenceSession(
            os.path.join(d, "wav2lip.onnx"), providers=self.providers)

    def _detect_face(self, img):
        """Return (cropped 96x96 face float 0..1, (x1,y1,x2,y2) in img space).

        cv2 5.x removed the legacy Haar cascade, so for a centered avatar we use
        a stable center crop (60% of the smaller side), padded for chin/forehead.
        Swap in insightface's scrfd ONNX here for arbitrary photos if needed.
        """
        h, w = img.shape[:2]
        s = int(min(w, h) * 0.6)
        x1 = (w - s) // 2
        y1 = (h - s) // 2
        x2, y2 = x1 + s, y1 + s
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (96, 96))
        return crop.astype(np.float32) / 255.0, (x1, y1, x2, y2)

    def _mel_from_wav(self, wav_path):
        import librosa
        y, _ = librosa.load(wav_path, sr=16000)
        mel = librosa.feature.melspectrogram(
            y=y, sr=16000, n_mels=80, n_fft=800, hop_length=200, win_length=800)
        mel = librosa.power_to_db(mel, ref=np.max)
        # standardize roughly to Wav2Lip's expected range
        mel = np.clip((mel + 100) / 100.0, -1.0, 1.0)
        return mel  # (80, T)

    def render_audio(self, audio_wav, sample_rate):
        h, w = self.img.shape[:2]
        x1, y1, x2, y2 = self.box
        mel = self._mel_from_wav(audio_wav)
        T = mel.shape[1]
        # 16 mel cols per output frame @ ~25fps; slide a 6-frame window.
        out_per_chunk = 16
        n_frames = max(1, T // out_per_chunk)
        face_win = []  # rolling 6-frame window of crops
        for i in range(n_frames):
            m = mel[:, i * out_per_chunk:(i + 1) * out_per_chunk]
            if m.shape[1] < 16:
                m = np.pad(m, ((0, 0), (0, 16 - m.shape[1])), mode="edge")
            m = m[None, None].astype(np.float32)  # [1,1,80,16]
            face_win.append(self.face)
            if len(face_win) > 6:
                face_win.pop(0)
            while len(face_win) < 6:
                face_win.insert(0, self.face)
            # Wav2Lip expects 6 consecutive GRAYSCALE crops as channels:
            # [B, 6, 96, 96]
            gray_win = [cv2.cvtColor((c * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                        for c in face_win]
            vf = np.stack(gray_win, axis=0)[None].astype(np.float32)  # [1,6,96,96]
            out = self.w2l.run(None, {
                "mel_spectrogram": m, "video_frames": vf})[0]
            synced = np.transpose(out[0], (1, 2, 0))  # (96,96,3)
            synced = np.clip(synced, 0, 1)
            frame = self.img.copy()
            frame[y1:y2, x1:x2] = cv2.resize(
                (synced * 255).astype(np.uint8), (x2 - x1, y2 - y1))
            yield frame, True
