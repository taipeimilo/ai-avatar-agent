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
        # NOTE: wav2lip.onnx segfaults under onnxruntime-directml on AMD, so we
        # pin CPUExecutionProvider here. It's a small single-crop model and runs
        # fast enough; the heavy TTS/Kokoro still uses DirectML elsewhere.
        self.w2l = ort.InferenceSession(
            os.path.join(d, "wav2lip.onnx"), providers=["CPUExecutionProvider"])

    def _detect_face(self, img):
        """Return (cropped 96x96 face float 0..1, (x1,y1,x2,y2) in img space).

        Uses the bundled SCRFD ONNX detector so arbitrary photos (incl. tall
        portraits) get a real face crop, not a center-crop guess. Falls back to a
        center crop if detection fails.
        """
        import os
        det_path = os.path.join(
            self.cfg.models_dir, "wav2lip", "insightface_func", "models",
            "antelope", "scrfd_2.5g_bnkps.onnx")
        try:
            from src.face.scrfd import SCRFDetector
            det = SCRFDetector(det_path, providers=self.providers)
            res = det.detect(img)
            if res:
                x1, y1, x2, y2 = res["box"]
                # pad for chin/forehead so the mouth region is intact
                h, w = img.shape[:2]
                pad = int((y2 - y1) * 0.2)
                x1 = max(0, x1 - pad // 2)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad // 2)
                y2 = min(h, y2 + pad)
                crop = img[y1:y2, x1:x2]
                crop = cv2.resize(crop, (96, 96))
                return crop.astype(np.float32) / 255.0, (x1, y1, x2, y2)
        except Exception as e:  # noqa: BLE001
            print(f"[wav2lip] face detection failed ({e}); center crop fallback.")
        h, w = img.shape[:2]
        s = int(min(w, h) * 0.6)
        x1, y1 = (w - s) // 2, (h - s) // 2
        x2, y2 = x1 + s, y1 + s
        crop = cv2.resize(img[y1:y2, x1:x2], (96, 96))
        return crop.astype(np.float32) / 255.0, (x1, y1, x2, y2)

    def _mel_from_wav(self, wav_path):
        # Match instant-high/wav2lip-onnx-HQ hparams exactly. The model was
        # trained on mels normalized to [-4, 4] (symmetric_mels, max_abs=4,
        # min_level_db=-100, ref_level_db=20). Feeding the wrong scale makes the
        # generator emit a constant (red block), so this MUST match.
        import librosa
        import scipy.signal
        y, _ = librosa.load(wav_path, sr=16000)
        y = scipy.signal.lfilter([1, -0.97], [1], y)  # preemphasis
        D = librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)
        S = np.abs(D)
        mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
        mel = np.dot(mel_basis, S)
        mel = 20 * np.log10(np.maximum(1e-5, mel)) - 20  # _amp_to_db - ref_level_db
        # _normalize: symmetric, allow_clipping
        mel = np.clip(2 * 4.0 * ((mel - (-100)) / 100.0) - 4.0, -4.0, 4.0)
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
