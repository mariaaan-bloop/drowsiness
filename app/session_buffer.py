"""
Buffer per-sesi + state machine alert -- porting dari logic Cell 22 (realtime) notebook, tapi
DIPISAH dari UI apa pun (bukan Gradio-specific lagi). Bisa dipakai dari REST API polling, dari
WebSocket, dari script lokal, dsb -- selama frontend-nya kirim frame satu-satu ke `.push_frame()`.

Satu SessionBuffer = satu "sesi" pemantauan (misal: satu kamera / satu pengemudi). Untuk banyak
kamera sekaligus, backend menyimpan banyak SessionBuffer sekaligus dikunci oleh session_id
(lihat SessionManager di bawah, dipakai server.py).
"""
import time
import threading
from collections import deque

import numpy as np

from .config import ALERT_THRESHOLD, ALERT_CONSECUTIVE_HITS, INFERENCE_INTERVAL_SEC, \
    SESSION_TIMEOUT_SEC, CFG
from .face_landmarks import FaceLandmarkExtractor, letterbox

SEQ_LEN = CFG["seq_len"]


class SessionBuffer:
    def __init__(self, model, alert_threshold=ALERT_THRESHOLD,
                 alert_consecutive_hits=ALERT_CONSECUTIVE_HITS,
                 inference_interval_sec=INFERENCE_INTERVAL_SEC):
        self.model = model
        self.alert_threshold = alert_threshold
        self.alert_consecutive_hits = alert_consecutive_hits
        self.inference_interval_sec = inference_interval_sec

        self.extractor = FaceLandmarkExtractor()
        self.frame_buffer = deque(maxlen=SEQ_LEN)
        self.feature_buffer = deque(maxlen=SEQ_LEN)

        self.last_infer_time = 0.0
        self.consecutive_hits = 0
        self.is_alert = False
        self.ngantuk_prob = 0.0
        self.last_seen = time.time()
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.extractor.reset_tracking()
            self.frame_buffer.clear()
            self.feature_buffer.clear()
            self.last_infer_time = 0.0
            self.consecutive_hits = 0
            self.is_alert = False
            self.ngantuk_prob = 0.0

    def push_frame(self, frame_rgb: np.ndarray) -> dict:
        """Proses 1 frame. Return dict status -- 'alert_rising_edge' True HANYA di frame yang
        MEMICU alert baru (dipakai frontend untuk tahu kapan harus bunyikan beep sekali, bukan
        setiap panggilan)."""
        with self._lock:
            self.last_seen = time.time()

            crop, features, used_fallback = self.extractor.process(frame_rgb)
            face_img = letterbox(crop)
            self.frame_buffer.append(face_img)
            self.feature_buffer.append(features)

            buffer_ready = len(self.frame_buffer) == SEQ_LEN
            now = time.time()
            should_infer = buffer_ready and (now - self.last_infer_time >= self.inference_interval_sec)

            alert_rising_edge = False
            if should_infer:
                self.last_infer_time = now
                frames = np.stack(self.frame_buffer, axis=0)
                landmarks = np.stack(self.feature_buffer, axis=0)
                probs = self.model.predict_window(frames, landmarks)
                self.ngantuk_prob = self.model.ngantuk_probability(probs)

                is_hit = self.ngantuk_prob >= self.alert_threshold
                self.consecutive_hits = self.consecutive_hits + 1 if is_hit else 0

                was_alert = self.is_alert
                self.is_alert = self.consecutive_hits >= self.alert_consecutive_hits
                alert_rising_edge = self.is_alert and not was_alert

            return {
                "buffer_filled": buffer_ready,
                "buffer_size": len(self.frame_buffer),
                "buffer_capacity": SEQ_LEN,
                "ran_inference": should_infer,
                "ngantuk_probability": self.ngantuk_prob,
                "consecutive_hits": self.consecutive_hits,
                "alert_consecutive_hits_needed": self.alert_consecutive_hits,
                "is_alert": self.is_alert,
                "alert_rising_edge": alert_rising_edge,
                "face_bbox": self.extractor.last_bbox,
                "used_fallback": used_fallback,
                "fallback_rate": self.extractor.fallback_rate,
            }

    def close(self):
        self.extractor.close()


class SessionManager:
    """Menyimpan banyak SessionBuffer sekaligus (1 per session_id/kamera), dan membersihkan
    sesi yang sudah tidak aktif supaya tidak bocor memori kalau banyak client konek-putus."""

    def __init__(self, model):
        self.model = model
        self._sessions: dict[str, SessionBuffer] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionBuffer:
        with self._lock:
            self._cleanup_stale()
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionBuffer(self.model)
            return self._sessions[session_id]

    def reset(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].reset()

    def remove(self, session_id: str):
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.close()

    def _cleanup_stale(self):
        now = time.time()
        stale = [sid for sid, s in self._sessions.items() if now - s.last_seen > SESSION_TIMEOUT_SEC]
        for sid in stale:
            self._sessions.pop(sid).close()
