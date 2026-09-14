"""
Config loader untuk backend.

Semua nilai di sini HARUS sama persis dengan yang dipakai saat training di notebook (Cell 2) --
jangan diubah manual kecuali kamu juga mengubahnya di notebook dan re-export model_package.
Nilai default di bawah diisi otomatis dari `model_package/config.json` yang di-export dari
Cell 21 (EXPORT MODEL PACKAGE) di notebook -- lihat README.md untuk cara export-nya.
"""
import json
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_PACKAGE_DIR = Path(os.environ.get("MODEL_PACKAGE_DIR", BACKEND_DIR / "model_package"))
MODEL_PATH = MODEL_PACKAGE_DIR / "model.keras"
CONFIG_PATH = MODEL_PACKAGE_DIR / "config.json"

MEDIAPIPE_MODEL_DIR = Path(os.environ.get("MEDIAPIPE_MODEL_DIR", BACKEND_DIR / ".mediapipe_models"))
FACE_LANDMARKER_MODEL = MEDIAPIPE_MODEL_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{CONFIG_PATH} tidak ditemukan. Jalankan cell 'EXPORT MODEL PACKAGE' di notebook "
            f"dulu, lalu copy folder model_package/ ke sini (lihat README.md)."
        )
    with open(CONFIG_PATH) as f:
        return json.load(f)


CFG = load_config() if CONFIG_PATH.exists() else None

# Tuning alert -- boleh dioverride lewat environment variable tanpa perlu re-export model,
# supaya bisa dikalibrasi ulang di lapangan tanpa retraining.
ALERT_THRESHOLD = float(os.environ.get("ALERT_THRESHOLD", (CFG or {}).get("alert_threshold", 0.5)))
ALERT_CONSECUTIVE_HITS = int(os.environ.get(
    "ALERT_CONSECUTIVE_HITS", (CFG or {}).get("alert_consecutive_hits", 3)
))
INFERENCE_INTERVAL_SEC = float(os.environ.get(
    "INFERENCE_INTERVAL_SEC", (CFG or {}).get("inference_interval_sec", 1.0)
))
SESSION_TIMEOUT_SEC = float(os.environ.get("SESSION_TIMEOUT_SEC", 120.0))
