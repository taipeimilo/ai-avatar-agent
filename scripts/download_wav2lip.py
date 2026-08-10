"""
Download Wav2Lip-ONNX (HQ) model weights into models/wav2lip/.

Source: instant-high/wav2lip-onnx-HQ (Google Drive). We fetch the two ONNX
files needed: wav2lip.onnx and face_detection.onnx.
Run once:  python scripts/download_wav2lip.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models", "wav2lip")
os.makedirs(MODELS, exist_ok=True)


def main():
    try:
        import gdown
    except Exception as e:  # noqa: BLE001
        print("gdown not installed:", e)
        sys.exit(1)
    # Google Drive folder from instant-high/wav2lip-onnx-HQ
    folder_id = "1BGl9bmMtlGEMx_wwKufJrZChFyqjnlsQ"
    print(f"Downloading Wav2Lip-ONNX weights into {MODELS} ...")
    gdown.download_folder(
        f"https://drive.google.com/drive/folders/{folder_id}",
        output=MODELS, quiet=False,
    )
    for f in ("wav2lip.onnx", "face_detection.onnx"):
        p = os.path.join(MODELS, f)
        print(("FOUND " if os.path.exists(p) else "MISSING ") + f)


if __name__ == "__main__":
    main()
