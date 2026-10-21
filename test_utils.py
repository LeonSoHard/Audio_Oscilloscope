"""
Unittests für das utils-Modul (Mathematik und Signalverarbeitung).
"""
import unittest
import numpy as np

from src.utils import convert_to_db, apply_hann_window, calculate_rms, clamp


class TestUtils(unittest.TestCase):

    # ─── Tests für clamp ────────────────────────────────────────────────────────

    def test_clamp_within_range(self):
        """Prüft, ob Werte innerhalb der Grenzen unverändert bleiben."""
        self.assertEqual(clamp(5.0, 0.0, 10.0), 5.0)

    def test_clamp_below_min(self):
        """Prüft, ob Werte unterhalb des Minimums auf min_val begrenzt werden."""
        self.assertEqual(clamp(-5.0, 0.0, 10.0), 0.0)

    def test_clamp_above_max(self):
        """Prüft, ob Werte oberhalb des Maximums auf max_val begrenzt werden."""
        self.assertEqual(clamp(15.0, 0.0, 10.0), 10.0)

    def test_clamp_boundaries(self):
        """Prüft exakte Grenzwerte."""
        self.assertEqual(clamp(0.0, 0.0, 10.0), 0.0)
        self.assertEqual(clamp(10.0, 0.0, 10.0), 10.0)

    # ─── Tests für calculate_rms ────────────────────────────────────────────────

    def test_calculate_rms_empty(self):
        """Prüft, ob bei einem leeren Array 0.0 zurückgegeben wird."""
        empty_array = np.array([], dtype=np.float32)
        self.assertEqual(calculate_rms(empty_array), 0.0)

    def test_calculate_rms_dc_signal(self):
        """Prüft RMS-Berechnung bei einem konstanten Gleichspannungssignal (DC)."""
        samples = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(calculate_rms(samples), 1.0, places=5)

    def test_calculate_rms_alternating_signal(self):
        """Prüft RMS-Berechnung bei einem Rechtecksignal [1, -1, 1, -1]."""
        samples = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        self.assertAlmostEqual(calculate_rms(samples), 1.0, places=5)

    def test_calculate_rms_sine_wave(self):
        """Prüft RMS einer Sinuswelle mit Amplitude 1 (Sollwert = 1 / sqrt(2) ≈ 0.7071)."""
        t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        sine_wave = np.sin(t)
        expected_rms = 1.0 / np.sqrt(2.0)
        self.assertAlmostEqual(calculate_rms(sine_wave), expected_rms, places=3)

    # ─── Tests für convert_to_db ────────────────────────────────────────────────

    def test_convert_to_db_full_scale(self):
        """Prüft, ob eine Amplitude von 1.0 zu 0 dBFS führt."""
        self.assertAlmostEqual(float(convert_to_db(1.0)), 0.0, places=4)

    def test_convert_to_db_half_scale(self):
        """Prüft, ob eine Amplitude von 0.5 zu ca. -6.02 dBFS führt."""
        self.assertAlmostEqual(float(convert_to_db(0.5)), -6.02059, places=3)

    def test_convert_to_db_zero_amplitude(self):
        """Prüft, ob Null-Amplitude durch epsilon stabil berechnet wird (ca. -200 dB)."""
        db_val = float(convert_to_db(0.0, epsilon=1e-10))
        self.assertAlmostEqual(db_val, -200.0, places=1)

    def test_convert_to_db_array_input(self):
        """Prüft die Konvertierung eines NumPy-Arrays."""
        amplitudes = np.array([1.0, 0.5, 0.1])
        db_values = convert_to_db(amplitudes)
        self.assertTrue(isinstance(db_values, np.ndarray))
        self.assertAlmostEqual(float(db_values[0]), 0.0, places=3)
        self.assertAlmostEqual(float(db_values[1]), -6.02, places=2)
        self.assertAlmostEqual(float(db_values[2]), -20.0, places=2)

    # ─── Tests für apply_hann_window ───────────────────────────────────────────

    def test_apply_hann_window_empty(self):
        """Prüft leere Eingabe."""
        empty_array = np.array([], dtype=np.float32)
        result = apply_hann_window(empty_array)
        self.assertEqual(len(result), 0)

    def test_apply_hann_window_edges_are_zero(self):
        """Prüft, ob das Hann-Fenster die Ränder des Signals auf 0 dämpft."""
        samples = np.ones(100, dtype=np.float32)
        windowed = apply_hann_window(samples)
        self.assertAlmostEqual(float(windowed[0]), 0.0, places=5)
        self.assertAlmostEqual(float(windowed[-1]), 0.0, places=5)

    def test_apply_hann_window_center_preserved(self):
        """Prüft, ob die Mitte des Signals bei ungerader Länge nahezu 1.0 bleibt."""
        samples = np.ones(101, dtype=np.float32)
        windowed = apply_hann_window(samples)
        self.assertAlmostEqual(float(windowed[50]), 1.0, places=3)


if __name__ == '__main__':
    unittest.main()