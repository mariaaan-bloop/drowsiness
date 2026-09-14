"""Prediksi dari file video utuh (upload), bukan stream realtime -- dipakai endpoint
POST /predict/video. Port dari cara Cell 16 (predict_with_tta) & Cell 20 (predict_video)
notebook bekerja, tapi tanpa layer caching disk (satu-kali proses per request)."""
import cv2
import numpy as np

from .config import CFG
from .face_landmarks import FaceLandmarkExtractor, letterbox

SEQ_LEN = CFG["seq_len"]
TTA_WINDOWS = CFG.get("tta_windows", 3)


def select_subsequence(cached_len, seq_len=SEQ_LEN, offset_frac=0.0):
    if cached_len <= seq_len:
        idx = np.arange(cached_len)
        if cached_len < seq_len:
            idx = np.concatenate([idx, np.full(seq_len - cached_len, cached_len - 1, dtype=np.int64)])
        return idx.astype(np.int32)

    base = np.linspace(0, cached_len - 1, seq_len)
    gap = (cached_len - 1) / (seq_len - 1) if seq_len > 1 else 0.0
    shift = (offset_frac - 0.5) * gap
    idx = np.clip(base + shift, 0, cached_len - 1)
    return idx.astype(np.int32)


def extract_full_video(video_path: str):
    """Loop lewat SEMUA frame video sekali, jalankan face+landmark extraction tiap frame.
    Return: (frames_letterboxed: (N, IMG_SIZE, IMG_SIZE, 3) uint8, landmarks: (N, 4) float32,
    fallback_rate: float)."""
    extractor = FaceLandmarkExtractor()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Tidak bisa membuka video: {video_path}")

    frames, landmarks = [], []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            crop, features, _ = extractor.process(frame_rgb)
            frames.append(letterbox(crop))
            landmarks.append(features)
        fallback_rate = extractor.fallback_rate
    finally:
        cap.release()
        extractor.close()

    if not frames:
        raise ValueError("Video tidak punya frame yang bisa dibaca.")

    return np.stack(frames, axis=0), np.stack(landmarks, axis=0).astype(np.float32), fallback_rate


def predict_video_file(video_path: str, model, n_windows: int = TTA_WINDOWS) -> dict:
    frames, landmarks, fallback_rate = extract_full_video(video_path)
    cache_len = frames.shape[0]

    window_probs = []
    for w in range(n_windows):
        offset = 0.0 if n_windows == 1 else w / (n_windows - 1)
        sel = select_subsequence(cache_len, SEQ_LEN, offset_frac=offset)
        window_probs.append(model.predict_window(frames[sel], landmarks[sel]))

    probs = np.mean(window_probs, axis=0)
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_class": CFG["classes"][pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {CFG["classes"][i]: float(probs[i]) for i in range(len(CFG["classes"]))},
        "face_crop_fallback_rate": float(fallback_rate),
    }
