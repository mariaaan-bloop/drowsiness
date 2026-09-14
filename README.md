# Drowsiness Alert Backend

Backend inference yang **terpisah total dari notebook training**. Notebook (Colab) tetap dipakai
untuk training + export model; backend ini yang menjalankan model untuk prediksi/alert -- bisa
jalan di laptop, server, atau edge device, tanpa perlu Colab/internet tetap terhubung ke Google.

## Kenapa dipisah dari notebook

Notebook + Gradio share-link cocok untuk **demo/presentasi**, tapi tidak cocok untuk "lapangan"
(dipakai beneran, terus-menerus) karena bergantung koneksi internet stabil + runtime Colab tetap
hidup. Backend ini bisa jalan lokal (bahkan tanpa internet, setelah model_package/ dan model
MediaPipe di-download sekali) -- jauh lebih cocok untuk deployment nyata.

## 1. Export model dari notebook

Di notebook (`drowsiness_model_v10_realtime.ipynb`), jalankan cell **"21. EXPORT MODEL PACKAGE"**
(cell baru di v11) -- ini membuat folder `model_package/` berisi:
- `model.keras` -- checkpoint terbaik (macro-F1, lihat Cell 15)
- `config.json` -- semua konstanta yang dibutuhkan backend (IMG_SIZE, SEQ_LEN, CLASSES, dst.)

Download folder `model_package/` itu (klik kanan di panel file Colab, atau lewat Google Drive
kalau `OUTPUT_DIR` di Drive-mu), lalu taruh persis di dalam folder backend ini:

```
drowsiness_backend/
  model_package/
    model.keras
    config.json
  app/
  static/
  ...
```

## 2. Install & jalankan

```bash
python -m venv venv && source venv/bin/activate   # opsional tapi disarankan
pip install -r requirements.txt

uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Download model MediaPipe Face Landmarker (`face_landmarker.task`, ~cuma sekali, otomatis) akan
terjadi otomatis saat backend pertama kali menerima request yang butuh deteksi wajah.

## 3. Coba

- `GET http://localhost:8000/health` -- cek server & model ke-load dengan benar.
- `POST http://localhost:8000/predict/video` (form-data, field `file` = video) -- prediksi 1 video
  utuh (sama persis dengan `predict_video()`/Cell 20 di notebook).
- Buka `http://localhost:8000/demo/` di browser -- demo realtime standalone (webcam + beep),
  **tidak butuh Colab/Gradio sama sekali**. Cocok dicoba di laptop yang sama dengan server, atau
  device lain di jaringan yang sama (ganti "API base URL" di halaman itu ke IP server).

## 4. Integrasi realtime dari frontend lain (mobile app, dsb.)

Kirim frame JPEG satu-satu ke:

```
POST /session/{session_id}/frame     (form-data, field "file" = 1 frame JPEG)
```

`session_id` bebas (misal ID device/kamera). Server balas JSON:

```json
{
  "buffer_filled": true,
  "ngantuk_probability": 0.83,
  "consecutive_hits": 2,
  "alert_consecutive_hits_needed": 3,
  "is_alert": false,
  "alert_rising_edge": false,
  "face_bbox": [120, 80, 340, 300],
  "fallback_rate": 0.02
}
```

Bunyikan beep di sisi frontend saat `alert_rising_edge == true` (server sengaja cuma memberi tahu
SEKALI di awal alert, bukan berulang tiap frame, supaya frontend tidak perlu logic anti-spam
sendiri). File beep tersedia di `GET /alert.wav`.

Panggil `POST /session/{session_id}/reset` untuk mulai ulang buffer (mis. sesi pemantauan baru),
dan `DELETE /session/{session_id}` untuk membebaskan resource kalau sesi sudah tidak dipakai
(server otomatis membersihkan sesi yang tidak aktif >2 menit juga).

## 5. Kalibrasi ulang tanpa retraining

Threshold alert bisa diubah lewat environment variable, tidak perlu re-export model:

```bash
ALERT_THRESHOLD=0.6 ALERT_CONSECUTIVE_HITS=4 uvicorn app.server:app --port 8000
```

## Catatan jujur soal kesiapan "lapangan"

Backend ini menyelesaikan masalah ARSITEKTUR (tidak lagi bergantung Colab). Itu **tidak sama**
dengan model-nya sudah tervalidasi cukup untuk dipakai beneran mencegah kecelakaan -- lihat Cell 0
& Cell 22 di notebook untuk gap yang masih ada (dataset validasi kecil, belum K-Fold, belum diuji
kondisi lapangan sungguhan). Backend ini membuat kamu SIAP menguji model di kondisi lebih dekat ke
lapangan (device lokal, tanpa dependency Colab) -- bukan pengganti validasi itu sendiri.
