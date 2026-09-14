"""Sintesis beep -- port dari Cell 22 notebook. Dikembalikan sebagai bytes WAV supaya bisa
dikirim langsung lewat HTTP response (audio/wav) tanpa perlu file sementara."""
import io
import wave

import numpy as np


def make_beep_wav_bytes(freq=880, duration=0.25, sr=22050, volume=0.5) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = volume * np.sin(2 * np.pi * freq * t)

    fade_len = max(1, int(sr * 0.01))
    fade = np.ones_like(t)
    fade[:fade_len] = np.linspace(0, 1, fade_len)
    fade[-fade_len:] = np.linspace(1, 0, fade_len)

    pcm = (tone * fade * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
