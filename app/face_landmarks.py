"""
Face + landmark extraction -- port LANGSUNG dari Cell 4 notebook (FaceLandmarkExtractor,
letterbox). Logic-nya SENGAJA tidak diubah sama sekali dari versi training, supaya prediksi di
backend ini konsisten dengan model yang sudah divalidasi di notebook. Kalau kamu ubah training
pipeline-nya, ubah juga di sini (dan re-export model_package).
"""
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .config import CFG, FACE_LANDMARKER_MODEL, FACE_LANDMARKER_URL, MEDIAPIPE_MODEL_DIR

MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_UP_INNER, MOUTH_DOWN_INNER = 13, 14
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FACE_OUTLINE_IDX = [10, 152, 234, 454]

IMG_SIZE = CFG["img_size"] if CFG else 128
FACE_MARGIN = CFG["face_margin"] if CFG else 0.35
MIN_DETECTION_CONFIDENCE = CFG.get("min_detection_confidence", 0.5) if CFG else 0.5
NUM_LANDMARK_FEATURES = CFG["num_landmark_features"] if CFG else 4


def ensure_face_landmarker_model():
    MEDIAPIPE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not FACE_LANDMARKER_MODEL.exists():
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
    if not FACE_LANDMARKER_MODEL.exists() or FACE_LANDMARKER_MODEL.stat().st_size < 1_000_000:
        raise RuntimeError("File face_landmarker.task gagal didownload dengan benar.")
    return FACE_LANDMARKER_MODEL


def _dist(p1, p2):
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def _eye_aspect_ratio(landmarks_px, eye_idx):
    p = [landmarks_px[i] for i in eye_idx]
    vertical = _dist(p[1], p[5]) + _dist(p[2], p[4])
    horizontal = 2.0 * _dist(p[0], p[3])
    return vertical / (horizontal + 1e-6)


class FaceLandmarkExtractor:
    """MediaPipe Tasks Face Landmarker + simple bbox tracking fallback.

    Return dari .process(): (crop RGB, features [MAR, EAR, mouth_width_norm, MAR_delta], used_fallback)
    """

    def __init__(self, min_detection_confidence=MIN_DETECTION_CONFIDENCE):
        model_path = ensure_face_landmarker_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)
        self.last_bbox = None
        self.last_features = None
        self.last_landmarks_px = None
        self.n_frames = 0
        self.n_fallback = 0

    def close(self):
        self.detector.close()

    def reset_tracking(self):
        self.last_bbox = None
        self.last_features = None
        self.last_landmarks_px = None
        self.n_frames = 0
        self.n_fallback = 0

    @property
    def fallback_rate(self):
        return self.n_fallback / self.n_frames if self.n_frames > 0 else 0.0

    def detect(self, frame_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        h, w = frame_rgb.shape[:2]
        lm = result.face_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    def process(self, frame_rgb):
        self.n_frames += 1
        h, w = frame_rgb.shape[:2]
        pts_px = self.detect(frame_rgb)

        if pts_px is not None:
            xs, ys = pts_px[:, 0], pts_px[:, 1]
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            bw, bh = x2 - x1, y2 - y1
            x1 -= bw * FACE_MARGIN
            x2 += bw * FACE_MARGIN
            y1 -= bh * FACE_MARGIN
            y2 += bh * FACE_MARGIN
            bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

            mouth_width = _dist(pts_px[MOUTH_LEFT], pts_px[MOUTH_RIGHT])
            mouth_open = _dist(pts_px[MOUTH_UP_INNER], pts_px[MOUTH_DOWN_INNER])
            mar = mouth_open / (mouth_width + 1e-6)
            ear = (_eye_aspect_ratio(pts_px, LEFT_EYE) + _eye_aspect_ratio(pts_px, RIGHT_EYE)) / 2.0
            face_width = _dist(pts_px[FACE_OUTLINE_IDX[2]], pts_px[FACE_OUTLINE_IDX[3]])
            mouth_width_norm = mouth_width / (face_width + 1e-6)

            prev_mar = self.last_features[0] if self.last_features is not None else mar
            mar_delta = mar - prev_mar

            features = np.array([mar, ear, mouth_width_norm, mar_delta], dtype=np.float32)

            self.last_bbox = bbox
            self.last_features = features
            self.last_landmarks_px = pts_px
            used_fallback = False
        else:
            self.n_fallback += 1
            used_fallback = True
            if self.last_bbox is not None:
                bbox = self.last_bbox
                features = self.last_features.copy()
                features[3] = 0.0
            else:
                bbox = (0, 0, w, h)
                features = np.zeros(NUM_LANDMARK_FEATURES, dtype=np.float32)

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        crop = frame_rgb[y1:y2, x1:x2]

        return crop, features, used_fallback


def letterbox(frame, size=IMG_SIZE):
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas
