"""
AudioAnalyzer-Modul zur Analyse von Audio-Dateien und Live-Frames.
"""
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from utils import convert_to_db, apply_hann_window, calculate_rms
from models import AudioMetadata


class AudioAnalyzer:
    DEFAULT_CHUNK_SIZE = 4096

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        if chunk_size < 16:
            raise ValueError("chunk_size muss mindestens 16 sein")
        self._chunk_size = chunk_size
        self._audio_data: Optional[np.ndarray] = None
        self._metadata: Optional[AudioMetadata] = None

    def load_file(self, filepath: str) -> bool:
        """Lädt eine Audiodatei über librosa und erfasst deren Metadaten."""
        if not HAS_LIBROSA:
            raise RuntimeError("Bibliothek 'librosa' fehlt. Bitte installieren.")

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {filepath}")
        if not path.is_file():
            raise ValueError(f"Pfad ist keine Datei: {filepath}")

        try:
            audio, sr = librosa.load(str(path), sr=None, mono=True)
        except (RuntimeError, ValueError, OSError, AttributeError) as err:
            raise RuntimeError(f"Fehler beim Decodieren der Audiodatei: {err}")

        if audio is None or len(audio) == 0:
            raise ValueError("Audio-Datei enthält keine Samples")

        self._audio_data = audio
        self._metadata = AudioMetadata(
            filepath=str(path.resolve()),
            filename=path.name,
            sample_rate=int(sr),
            duration_sec=float(len(audio) / sr) if sr > 0 else 0.0,
            channels=1 if audio.ndim == 1 else audio.shape[1],
            total_samples=len(audio),
            loaded_at=datetime.now().isoformat()
        )
        return True

    def analyze_frame(self, start_sample: int, end_sample: int) -> Dict[str, float]:
        """Berechnet Pegelwerte (LUFS/Peak) LIVE für das aktuell abgespielte Zeitfenster."""
        if self._audio_data is None or len(self._audio_data) == 0:
            return {'lufs_integrated': -70.0, 'true_peak': -70.0}

        total_len = len(self._audio_data)
        start = max(0, min(total_len, start_sample))
        end = max(start + 1, min(total_len, end_sample))

        frame = self._audio_data[start:end]
        if len(frame) == 0:
            return {'lufs_integrated': -70.0, 'true_peak': -70.0}

        rms = calculate_rms(frame)
        lufs = 20 * np.log10(rms + 1e-10) - 3.0
        peak_amp = float(np.max(np.abs(frame))) if len(frame) > 0 else 0.0
        peak = 20 * np.log10(peak_amp + 1e-10)

        return {
            'lufs_integrated': float(lufs),
            'true_peak': float(peak)
        }

    def analyze_loudness(self) -> Dict[str, float]:
        """Gesamtanalyse für die komplette Datei (Statisch)."""
        if self._audio_data is None:
            raise ValueError("Keine Audio-Datei geladen")
        return self.analyze_frame(0, len(self._audio_data))

    def analyze_spectrum(self) -> Dict[str, np.ndarray]:
        """Berechnet das Frequenzspektrum des aktuellen Signals."""
        if self._audio_data is None or self._metadata is None:
            raise ValueError("Keine Audio-Datei geladen")

        audio_length = len(self._audio_data)
        sr = self._metadata.sample_rate
        fft_size = max(self._chunk_size, 1024)

        if audio_length < fft_size:
            padded = np.zeros(fft_size)
            padded[:audio_length] = self._audio_data
            windowed = apply_hann_window(padded)
        else:
            windowed = apply_hann_window(self._audio_data[:fft_size])

        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        magnitude_db = convert_to_db(magnitude)
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / sr)

        return {'frequency': freqs, 'magnitude_db': magnitude_db}

    def get_metadata(self) -> Dict:
        """Gibt die geladenen Metadaten als Dictionary zurück."""
        if self._metadata is None:
            raise ValueError("Keine Datei geladen")
        return self._metadata.to_dict()

    def get_audio_data(self) -> np.ndarray:
        """Gibt die rohen Audiosamples zurück."""
        if self._audio_data is None:
            raise ValueError("Keine Datei geladen")
        return self._audio_data.copy()