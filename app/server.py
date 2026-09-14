"""
Backend inference server -- terpisah total dari notebook training. Jalankan dengan:

    uvicorn app.server:app --host 0.0.0.0 --port 8000

Endpoint:
    GET  /health                         status server + info model
    POST /predict/video                  upload 1 file video, dapat prediksi (dengan TTA)
    POST /session/{session_id}/frame     kirim 1 frame (JPEG) utk mode realtime, dapat status
    POST /session/{session_id}/reset     reset buffer sesi (mis. mulai pemantauan baru)
    DELETE /session/{session_id}         hapus sesi (bebaskan resource)
    GET  /alert.wav                      file beep (dipakai frontend saat alert_rising_edge)

Kenapa dipisah dari notebook: server ini TIDAK bergantung Colab/Gradio share-link sama sekali --
bisa dijalankan di laptop/server/edge device apa pun yang punya model_package/ hasil export dari
notebook (lihat README.md). Ini yang membuatnya pantas untuk "lapangan" (bukan cuma demo), selama
model & threshold-nya sudah divalidasi cukup (lihat catatan Cell 0/22 notebook soal ini).
"""
import io
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .config import CFG
from .model import get_model
from .session_buffer import SessionManager
from .video_utils import predict_video_file
from .alerts import make_beep_wav_bytes

app = FastAPI(title="Drowsiness Alert Backend")

_model = None
_sessions = None
_beep_bytes = make_beep_wav_bytes()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(STATIC_DIR), html=True), name="demo")


def _get_model():
    global _model
    if _model is None:
        _model = get_model()
    return _model


def _get_sessions():
    global _sessions
    if _sessions is None:
        _sessions = SessionManager(_get_model())
    return _sessions


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": CFG["classes"],
        "seq_len": CFG["seq_len"],
        "img_size": CFG["img_size"],
        "model_exported_at": CFG.get("exported_at"),
    }


@app.post("/predict/video")
async def predict_video_endpoint(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict_video_file(tmp_path, _get_model())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@app.post("/session/{session_id}/frame")
async def push_frame(session_id: str, file: UploadFile = File(...)):
    """file = 1 frame JPEG/PNG (dikirim dari frontend tiap ~100-300ms). Server yang mengatur
    sendiri seberapa sering model benar-benar dijalankan (lihat INFERENCE_INTERVAL_SEC di
    config.py) -- frontend cukup kirim sesering mungkin, tidak perlu tahu throttling-nya."""
    import cv2

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Gagal decode gambar frame.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    session = _get_sessions().get_or_create(session_id)
    status = session.push_frame(frame_rgb)
    return status


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str):
    _get_sessions().reset(session_id)
    return {"reset": True, "session_id": session_id}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    _get_sessions().remove(session_id)
    return {"deleted": True, "session_id": session_id}


@app.get("/alert.wav")
def alert_wav():
    return Response(content=_beep_bytes, media_type="audio/wav")
