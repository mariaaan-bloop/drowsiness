"""Port langsung dari compute_summary_features() di notebook Cell 4 (V6)."""
import numpy as np


def compute_summary_features(landmarks_seq: np.ndarray) -> np.ndarray:
    """landmarks_seq: (SEQ_LEN, 4) = [MAR, EAR, mouth_width_norm, MAR_delta].
    Return: [mar_mean, mar_std, mar_max, ear_mean, ear_std, mar_oscillation_rate]
    """
    mar = landmarks_seq[:, 0]
    ear = landmarks_seq[:, 1]
    mar_delta = landmarks_seq[:, 3]

    sign = np.sign(mar_delta)
    sign[sign == 0] = 1.0
    n_crossings = int(np.sum(sign[1:] != sign[:-1]))
    oscillation_rate = n_crossings / max(1, len(mar_delta) - 1)

    return np.array([
        mar.mean(), mar.std(), mar.max(),
        ear.mean(), ear.std(),
        oscillation_rate,
    ], dtype=np.float32)
