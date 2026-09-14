"""
Wrapper tipis di atas model .keras yang di-export dari notebook. Loading pakai compile=False
(sama seperti Cell 15 notebook) -- tidak perlu custom_objects untuk focal loss karena kita cuma
butuh forward pass (predict), bukan training/compile ulang.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .config import CFG, MODEL_PATH
from .features import compute_summary_features

CLASSES = CFG["classes"]
CLASS_TO_ID = CFG["class_to_id"]
NUM_CLASSES = len(CLASSES)
SEQ_LEN = CFG["seq_len"]
IMG_SIZE = CFG["img_size"]
USE_SUMMARY_BRANCH = CFG.get("use_summary_branch", True)
USE_AUDIO_BRANCH = CFG.get("use_audio_branch", False)
NUM_AUDIO_FEATURES = CFG.get("num_audio_features", 0)
NGANTUK_CLASS_ID = CLASS_TO_ID.get("Ngantuk")


class DrowsinessModel:
    """Satu instance, dipakai bersama (thread-safe untuk inference karena Keras model.predict
    tidak mutate state internal antar panggilan -- aman dipanggil dari beberapa request FastAPI
    selama tidak dipanggil BENERAN paralel di thread berbeda tanpa lock; lihat catatan di
    server.py soal ini)."""

    def __init__(self, model_path=MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def predict_window(self, frames: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """frames: (SEQ_LEN, IMG_SIZE, IMG_SIZE, 3) uint8 RGB, sudah di-letterbox.
        landmarks: (SEQ_LEN, 4) float32 [MAR, EAR, mouth_width_norm, MAR_delta].
        Return: probabilitas per kelas, shape (NUM_CLASSES,).
        """
        video_arr = preprocess_input(frames.astype(np.float32))
        landmark_arr = landmarks.astype(np.float32)

        batch_x = {
            "video_input": video_arr[np.newaxis, ...],
            "landmark_input": landmark_arr[np.newaxis, ...],
        }
        if USE_SUMMARY_BRANCH:
            batch_x["summary_input"] = compute_summary_features(landmark_arr)[np.newaxis, ...]
        if USE_AUDIO_BRANCH:
            # tidak ada audio realtime dari webcam/kamera -> netral (nol), bukan ditebak
            batch_x["audio_input"] = np.zeros((1, SEQ_LEN, NUM_AUDIO_FEATURES), dtype=np.float32)

        probs = self.model.predict(batch_x, verbose=0)[0]
        return probs

    def ngantuk_probability(self, probs: np.ndarray) -> float:
        if NGANTUK_CLASS_ID is None:
            raise RuntimeError(
                "Model package ini bukan model biner (tidak ada kelas 'Ngantuk' di config.json)."
            )
        return float(probs[NGANTUK_CLASS_ID])


_model_singleton = None


def get_model() -> DrowsinessModel:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = DrowsinessModel()
    return _model_singleton
