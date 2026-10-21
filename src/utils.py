"""
Hilfsfunktionen für mathematische Operationen und Signalverarbeitung.
"""
from typing import Union
import numpy as np


def convert_to_db(magnitude: Union[np.ndarray, float], epsilon: float = 1e-10) -> np.ndarray:
    """Konvertiert Amplitudenwerte in DeziBel (dB)."""
    magnitude_array = np.asarray(magnitude)
    return 20 * np.log10(np.abs(magnitude_array) + epsilon)


def apply_hann_window(samples: np.ndarray) -> np.ndarray:
    """Wendet ein Hann-Fenster zur Glättung von Signalgrenzen an."""
    if len(samples) == 0:
        return np.array([])
    return samples * np.hanning(len(samples))


def calculate_rms(samples: np.ndarray) -> float:
    """Berechnet den Effektivwert (RMS) eines Audiosignals."""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Begrenzt einen Wert strikt auf das Intervall [min_val, max_val]."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value