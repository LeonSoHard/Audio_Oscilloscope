"""
Modul für Datenmodelle und Datenstrukturen.
Kapselt Metadaten als strukturierte Klasse.
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AudioMetadata:
    """Modell zur Repräsentation von Metadaten einer Audiodatei."""
    filepath: str
    filename: str
    sample_rate: int
    duration_sec: float
    channels: int
    total_samples: int
    loaded_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert die Metadaten in ein Dictionary."""
        return {
            'filepath': self.filepath,
            'filename': self.filename,
            'sample_rate': self.sample_rate,
            'duration_sec': self.duration_sec,
            'channels': self.channels,
            'total_samples': self.total_samples,
            'loaded_at': self.loaded_at
        }