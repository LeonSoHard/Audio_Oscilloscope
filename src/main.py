"""
Dual-Sample Oszilloskop mit synchronisiertem 4/4-Beat-Sequencer.
Sample 1 (Kick/Clap) sitzt im 4/4-Takt auf den Viertel-Beats.
Sample 2 (Bassline) wird automatisch zeitlich auf das BPM-Raster eingepasst (Time-Stretch/Resampling).
Timeline per Mausklick/Drag scrubbar.
"""
import sys
import math
import numpy as np
import sounddevice as sd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QFileDialog, QMessageBox, QGroupBox, QToolBar,
    QLabel, QSpinBox, QHBoxLayout, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QAction, QFont, QKeyEvent, QLinearGradient

from analyzer import AudioAnalyzer
from utils import clamp


# ─── Audio Engine ──────────────────────────────────────────────────────────────

class AudioEngine:
    """Spielt zwei Samples sample-genau synchron in einer Endlosschleife ab."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.buffer = None
        self.stream = None
        self.is_playing = False
        self.read_pos = 0
        self._volume = 1.0
        self.display_1 = np.array([], dtype=np.float32)
        self.display_2 = np.array([], dtype=np.float32)
        self.loop_length = 0

    def build_loop(self, audio_1, audio_2, sr_1, sr_2, bar_samples: int):
        """Mischt beide Samples perfekt synchron im Takt.

        - Sample 1 (Kick/Clap) wird auf allen 4 Viertel-Beats pro Takt getriggert.
        - Sample 2 (Bassline) wird zeitlich exakt auf das Nächstbeste Takt-Raster gerastert.
        """
        a1 = self._prepare(audio_1, sr_1)
        a2 = self._prepare(audio_2, sr_2)

        beat_samples = max(1, bar_samples // 4)

        # 1. Bestimme, wie viele Takte Sample 2 lang ist (aufgerundet auf 1, 2, 4, 8... Takte)
        if len(a2) > 0:
            raw_bars = len(a2) / bar_samples
            # Auf die nächste Zweierpotenz an Takten aufrunden für saubere Musical Loops
            target_bars = max(1, int(2 ** math.ceil(math.log2(max(1.0, raw_bars)))))
            loop_len = target_bars * bar_samples

            # Sample 2 exakt auf die Taktlänge resamplen/stretchen, damit es nicht aus dem Takt läuft
            a2 = self._stretch_to_length(a2, loop_len)
        else:
            loop_len = 4 * bar_samples
            a2 = np.zeros(loop_len, dtype=np.float32)

        # 2. Sample 1 im 4/4-Takt triggern
        out1 = np.zeros(loop_len, dtype=np.float32)
        if len(a1) > 0:
            for start in range(0, loop_len, beat_samples):
                end = min(start + len(a1), loop_len)
                out1[start:end] += a1[:end - start]

        mixed = np.tanh((out1 + a2) * 0.7).astype(np.float32)
        self.buffer = np.column_stack([mixed, mixed])
        self.display_1 = out1
        self.display_2 = a2
        self.loop_length = loop_len

    def _prepare(self, audio, sr: int) -> np.ndarray:
        """Mono machen und auf Ziel-Samplerate bringen."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            return np.array([], dtype=np.float32)
        if sr != self.sample_rate and sr > 0:
            idx = (np.arange(0, len(audio), self.sample_rate / sr)).astype(int)
            audio = audio[idx[idx < len(audio)]]
        return audio.astype(np.float32)

    def _stretch_to_length(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Passt die Länge von Audio per linearer Interpolation genau an das Taktraster an."""
        if len(audio) == 0 or target_length <= 0:
            return np.zeros(target_length, dtype=np.float32)
        old_indices = np.linspace(0, len(audio) - 1, num=len(audio))
        new_indices = np.linspace(0, len(audio) - 1, num=target_length)
        return np.interp(new_indices, old_indices, audio).astype(np.float32)

    def play(self):
        if self.buffer is None or self.loop_length == 0:
            return
        self.stop()
        self.read_pos = 0
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=2, dtype='float32',
            callback=self._callback, blocksize=1024
        )
        self.stream.start()
        self.is_playing = True

    def _callback(self, outdata, frames, time_info, status):
        if self.buffer is None or self.loop_length == 0:
            outdata.fill(0)
            return

        end = self.read_pos + frames
        if end <= self.loop_length:
            outdata[:] = self.buffer[self.read_pos:end] * self._volume
        else:
            first = self.loop_length - self.read_pos
            outdata[:first] = self.buffer[self.read_pos:] * self._volume
            outdata[first:] = self.buffer[:frames - first] * self._volume
        self.read_pos = end % self.loop_length

    def seek(self, seconds: float):
        if self.loop_length > 0:
            pos = int(seconds * self.sample_rate) % self.loop_length
            self.read_pos = max(0, pos)

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_playing = False
        self.read_pos = 0

    def current_position(self) -> int:
        return self.read_pos % self.loop_length if self.loop_length else 0

    def set_volume(self, factor: float):
        self._volume = factor


# ─── GUI Widgets ───────────────────────────────────────────────────────────────

class KnobControl(QWidget):
    """Potentiometer mit einheitlicher visueller Geometrie und präziser dB-Skalierung."""

    def __init__(self, label_text: str, min_val: int = 0, max_val: int = 100,
                 default_val: int = 50, is_db: bool = False, callback=None):
        super().__init__()
        self.label_text = label_text
        self.min_val = min_val
        self.max_val = max_val
        self.current_value = default_val
        self.is_db = is_db
        self.callback = callback
        self.setMinimumSize(140, 160)
        self.setMaximumSize(160, 180)

    def value(self) -> int:
        return self.current_value

    def setValue(self, val: int):
        self.current_value = clamp(val, self.min_val, self.max_val)
        self.update()
        if self.callback:
            self.callback(self.current_value)

    def mousePressEvent(self, event):
        self._update_from_mouse(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._update_from_mouse(event.position())

    def _update_from_mouse(self, pos: QPointF):
        cx, cy = self.width() / 2.0, self.height() / 2.0 - 12
        angle_deg = math.degrees(math.atan2(-(pos.y() - cy), pos.x() - cx))
        gauge = (225 - angle_deg) % 360
        if gauge > 270:
            gauge = 270 if gauge < 315 else 0
        self.setValue(int(self.min_val + (gauge / 270.0) * (self.max_val - self.min_val)))

    def _ratio_to_db(self, ratio: float) -> float:
        if ratio <= 0.02:
            return -70.0
        return -60.0 * ((1.0 - ratio) ** 2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0 - 12
        radius = min(cx, cy) - 28

        if self.is_db:
            ticks = [
                (0.0, "-∞"),
                (0.25, "-34"),
                (0.50, "-15"),
                (0.75, "-4"),
                (1.0, "0")
            ]
        else:
            ticks = [(i / 4.0, f"{i * 25}") for i in range(5)]

        painter.setFont(QFont("Arial", 7))
        for ratio, lbl in ticks:
            rad = math.radians(225 - ratio * 270)
            painter.setPen(QPen(QColor(120, 120, 120), 1.5))
            painter.drawLine(
                QPointF(cx + (radius + 4) * math.cos(rad), cy - (radius + 4) * math.sin(rad)),
                QPointF(cx + (radius + 9) * math.cos(rad), cy - (radius + 9) * math.sin(rad))
            )
            tx = cx + (radius + 18) * math.cos(rad)
            ty = cy - (radius + 18) * math.sin(rad)
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(int(tx - 15), int(ty - 8), 30, 16,
                             Qt.AlignmentFlag.AlignCenter, lbl)

        grad = QLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius)
        grad.setColorAt(0.0, QColor(80, 80, 80))
        grad.setColorAt(1.0, QColor(25, 25, 25))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(QPointF(cx, cy), radius + 2, radius + 2)

        grad2 = QLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius)
        grad2.setColorAt(0.0, QColor(45, 45, 45))
        grad2.setColorAt(1.0, QColor(15, 15, 15))
        painter.setBrush(QBrush(grad2))
        painter.drawEllipse(QPointF(cx, cy), radius - 2, radius - 2)

        val_ratio = (self.current_value - self.min_val) / float(self.max_val - self.min_val or 1)
        rad = math.radians(225 - val_ratio * 270)
        painter.setPen(QPen(QColor(100, 200, 255), 3.0,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(
            QPointF(cx + (radius - 12) * math.cos(rad), cy - (radius - 12) * math.sin(rad)),
            QPointF(cx + (radius - 2) * math.cos(rad), cy - (radius - 2) * math.sin(rad))
        )

        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(0, h - 26, w, 14, Qt.AlignmentFlag.AlignCenter, self.label_text)

        painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        painter.setPen(QColor(100, 200, 255))
        if self.is_db:
            db = self._ratio_to_db(val_ratio)
            txt = f"{db:+.1f} dB" if db > -60.0 else "-∞ dB"
        else:
            txt = f"{self.current_value}%"
        painter.drawText(0, h - 12, w, 12, Qt.AlignmentFlag.AlignCenter, txt)


class AnalogMixerMeter(QWidget):
    """Vertikale Pegelanzeige."""

    def __init__(self, label_text: str):
        super().__init__()
        self.label_text = label_text
        self.value_db = -70.0
        self.setMinimumSize(85, 180)
        self.setMaximumWidth(100)

    def set_value(self, value_db: float):
        self.value_db = float(value_db)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(0, 4, w, 16, Qt.AlignmentFlag.AlignCenter, self.label_text)

        bx, by, bw, bh = 15, 25, 20, h - 55
        painter.fillRect(bx, by, bw, bh, QColor(10, 10, 10))

        norm = clamp((self.value_db + 60.0) / 60.0, 0.0, 1.0)
        fill_h = int(norm * bh)
        if fill_h > 0:
            grad = QLinearGradient(bx, by + bh, bx, by)
            grad.setColorAt(0.0, QColor(0, 220, 100))
            grad.setColorAt(0.7, QColor(255, 200, 0))
            grad.setColorAt(1.0, QColor(255, 40, 40))
            painter.fillRect(bx, by + bh - fill_h, bw, fill_h, QBrush(grad))

        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(bx, by, bw, bh)

        painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        painter.setPen(QColor(100, 200, 255))
        txt = f"{self.value_db:+.1f}" if self.value_db > -60.0 else "-∞"
        painter.drawText(0, h - 18, w, 14, Qt.AlignmentFlag.AlignCenter, f"{txt} dB")


class WaveformViewport(QWidget):
    """Dual-Channel Oszilloskop mit Transienten-Overlay und Scrubbing."""

    def __init__(self, scrub_callback=None):
        super().__init__()
        self.setMinimumSize(800, 320)
        self.data_1 = np.array([], dtype=np.float32)
        self.data_2 = np.array([], dtype=np.float32)
        self.sample_rate = 44100
        self.view_duration = 0.5
        self.play_offset_sec = 0.0
        self.is_playing = False
        self.scrub_callback = scrub_callback
        self.loop_duration = 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_audio(self, data_1, data_2, sr: int, loop_duration: float = 0.0):
        self.data_1 = np.asarray(data_1, dtype=np.float32)
        self.data_2 = np.asarray(data_2, dtype=np.float32)
        self.sample_rate = sr
        self.loop_duration = loop_duration
        self.update()

    def set_position(self, seconds: float):
        self.play_offset_sec = seconds
        self.update()

    def _canvas_metrics(self):
        pl, pr = 40, 65
        cw = max(1, self.width() - pl - pr)
        return pl, cw

    def _scrub(self, pos: QPointF):
        if not self.scrub_callback or self.loop_duration <= 0:
            return
        pl, cw = self._canvas_metrics()
        ratio = clamp((pos.x() - pl) / cw, 0.0, 1.0)
        self.scrub_callback(ratio * self.loop_duration)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._scrub(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._scrub(event.position())

    def _draw_wave(self, painter, data, color, pl, yc, cw, ch):
        if data.size == 0:
            return
        view_samples = int(self.view_duration * self.sample_rate)
        cur = int(self.play_offset_sec * self.sample_rate)
        painter.setPen(QPen(color, 2))
        prev = None
        for px in range(cw):
            off = int(((cw - 1 - px) / cw) * view_samples)
            idx = (cur - off) % len(data)
            y = yc - int(data[idx] * ch / 2)
            if prev is not None:
                painter.drawLine(pl + px - 1, prev, pl + px, y)
            prev = y

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(25, 25, 25))

        if self.data_1.size == 0 and self.data_2.size == 0:
            painter.setPen(QPen(QColor(220, 220, 220), 14))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Datei öffnen zum Starten")
            return

        w, h = self.width(), self.height()
        pl, pr, pt, pb = 40, 65, 25, 40
        cw, ch = w - pl - pr, h - pt - pb
        yc = pt + ch // 2

        total_ms = self.view_duration * 1000
        for i in range(6):
            ratio = i / 5
            gx = pl + int(cw * ratio)
            painter.setPen(QPen(QColor(50, 50, 50), 1, Qt.PenStyle.DashLine))
            painter.drawLine(gx, pt, gx, pt + ch)
            painter.setPen(QColor(180, 180, 180))
            painter.setFont(QFont("Arial", 8))
            lbl = "0 ms" if i == 5 else str(int(-total_ms * (1 - ratio)))
            painter.drawText(gx - 25, 0, 50, pt - 3,
                             Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, lbl)

        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawLine(pl, yc, pl + cw, yc)
        for norm, lbl in [(1.0, "0"), (0.71, "3"), (0.5, "6"), (0.25, "12")]:
            yp, yn = yc - int(norm * ch / 2), yc + int(norm * ch / 2)
            painter.setPen(QPen(QColor(50, 50, 50), 1, Qt.PenStyle.DashLine))
            painter.drawLine(pl, yp, pl + cw, yp)
            painter.drawLine(pl, yn, pl + cw, yn)
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(pl + cw + 5, yp - 6, 55, 12,
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"+{lbl}")
            painter.drawText(pl + cw + 5, yn - 6, 55, 12,
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"-{lbl}")

        self._draw_wave(painter, self.data_1, QColor(255, 180, 180), pl, yc, cw, ch)
        self._draw_wave(painter, self.data_2, QColor(0, 229, 255), pl, yc, cw, ch)

        if self.data_1.size > 0 and self.data_2.size > 0:
            view_s = int(self.view_duration * self.sample_rate)
            cur = int(self.play_offset_sec * self.sample_rate)
            for px in range(0, cw, 2):
                off = int(((cw - 1 - px) / cw) * view_s)
                i1 = (cur - off) % len(self.data_1)
                i2 = (cur - off) % len(self.data_2)
                if abs(self.data_1[i1]) > 0.4 and abs(self.data_2[i2]) > 0.4:
                    painter.fillRect(pl + px, pt, 2, ch, QColor(255, 230, 0, 40))

        painter.setPen(QPen(QColor(255, 40, 40), 3))
        painter.drawLine(pl + cw, pt, pl + cw, pt + ch)


# ─── Hauptfenster ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Dual-Sample Oszilloskop mit 4/4-Beat-Sequencer und Scrubbing."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Oscilloscope v2.0")
        self.setGeometry(100, 100, 1200, 750)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.analyzer_1 = AudioAnalyzer()
        self.analyzer_2 = AudioAnalyzer()
        self.engine = AudioEngine()
        self.file_1 = None
        self.file_2 = None

        self._setup_ui()
        self._setup_menu()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        tb = QToolBar()
        self.addToolBar(tb)

        btn1 = QPushButton("Sample 1 (Rosa)")
        btn1.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn1.clicked.connect(self._open_1)
        tb.addWidget(btn1)
        tb.addSeparator()

        btn2 = QPushButton("Sample 2 (Cyan)")
        btn2.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn2.clicked.connect(self._open_2)
        tb.addWidget(btn2)
        tb.addSeparator()

        tb.addWidget(QLabel(" BPM: "))
        self.bpm_box = QSpinBox()
        self.bpm_box.setRange(40, 240)
        self.bpm_box.setValue(120)
        self.bpm_box.setFixedWidth(70)
        self.bpm_box.valueChanged.connect(self._rebuild_loop)
        tb.addWidget(self.bpm_box)

        self.waveform = WaveformViewport(scrub_callback=self._on_scrub)
        layout.addWidget(self.waveform)

        group = QGroupBox()
        group.setStyleSheet("QGroupBox { border: none; }")
        hbox = QHBoxLayout(group)
        hbox.addStretch()

        inner = QHBoxLayout()
        inner.setSpacing(30)
        self.meter_lufs = AnalogMixerMeter("LUFS")
        self.meter_peak = AnalogMixerMeter("PEAK")
        self.meter_rms = AnalogMixerMeter("RMS")
        for m in (self.meter_lufs, self.meter_peak, self.meter_rms):
            inner.addWidget(m)
        inner.addSpacing(30)

        self.knob_speed = KnobControl("Scope Speed", 0, 100, 30, False, self._speed_changed)
        self.knob_vol = KnobControl("Output Vol", 0, 100, 70, True, self._vol_changed)
        inner.addWidget(self.knob_speed)
        inner.addWidget(self.knob_vol)

        hbox.addLayout(inner)
        hbox.addStretch()
        layout.addWidget(group)

        self._timer = QTimer()
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._tick)

    def _setup_menu(self):
        m = self.menuBar().addMenu("Datei")
        a1 = QAction("Sample 1 öffnen...", self)
        a1.setShortcut("Ctrl+O")
        a1.triggered.connect(self._open_1)
        m.addAction(a1)
        a2 = QAction("Sample 2 öffnen...", self)
        a2.setShortcut("Ctrl+Shift+O")
        a2.triggered.connect(self._open_2)
        m.addAction(a2)

    def _bar_duration_sec(self) -> float:
        return (60.0 / self.bpm_box.value()) * 4

    def _rebuild_loop(self):
        if not self.file_1 and not self.file_2:
            return
        a1 = self.analyzer_1.get_audio_data() if self.file_1 else np.zeros(0, dtype=np.float32)
        a2 = self.analyzer_2.get_audio_data() if self.file_2 else np.zeros(0, dtype=np.float32)
        sr1 = self.analyzer_1.get_metadata().get('sample_rate', 44100) if self.file_1 else 44100
        sr2 = self.analyzer_2.get_metadata().get('sample_rate', 44100) if self.file_2 else 44100

        bar_samples = int(self._bar_duration_sec() * self.engine.sample_rate)
        self.engine.build_loop(a1, a2, sr1, sr2, bar_samples)

        loop_dur = self.engine.loop_length / self.engine.sample_rate
        self.waveform.set_audio(self.engine.display_1, self.engine.display_2,
                                self.engine.sample_rate, loop_dur)

        if self.engine.is_playing:
            self.engine.stop()
            self.engine.play()

    def _on_scrub(self, seconds: float):
        self.engine.seek(seconds)
        self.waveform.set_position(seconds)

    def _play(self):
        if self.engine.buffer is None:
            self._rebuild_loop()
        if self.engine.buffer is None:
            return
        self.engine.play()
        self.waveform.is_playing = True
        self._timer.start()
        self.statusBar().showMessage("▶ Abspielen")

    def _stop(self):
        self.engine.stop()
        self.waveform.is_playing = False
        self._timer.stop()
        self.statusBar().showMessage("⏹ Stopp")

    def _toggle(self):
        if self.engine.is_playing:
            self._stop()
        else:
            self._play()

    def _tick(self):
        if not self.engine.is_playing:
            return
        pos = self.engine.current_position()
        self.waveform.set_position(pos / self.engine.sample_rate)

        if self.engine.buffer is not None and self.engine.loop_length > 0:
            start = max(0, pos - 4410)
            end = min(pos, self.engine.loop_length)
            if end > start:
                frame = self.engine.buffer[start:end, 0]
                peak = 20 * math.log10(np.max(np.abs(frame)) + 1e-9)
                rms = 20 * math.log10(np.sqrt(np.mean(frame ** 2)) + 1e-9)
                self.meter_peak.set_value(peak)
                self.meter_rms.set_value(rms)
                self.meter_lufs.set_value(rms - 3.0)

    def _speed_changed(self, val: int):
        self.waveform.view_duration = 0.1 + (val / 100.0) * 1.9

    def _vol_changed(self, val: int):
        self.engine.set_volume((val / 100.0) ** 2)

    def _open_1(self):
        self._load_sample(1)

    def _open_2(self):
        self._load_sample(2)

    def _load_sample(self, index: int):
        analyzer = self.analyzer_1 if index == 1 else self.analyzer_2
        path, _ = QFileDialog.getOpenFileName(self, f"Sample {index}", "",
                                              "Audio (*.wav *.flac *.mp3)")
        if not path:
            return
        try:
            if analyzer.load_file(path):
                if index == 1:
                    self.file_1 = path
                else:
                    self.file_2 = path
                self._rebuild_loop()
        except (FileNotFoundError, PermissionError, RuntimeError) as e:
            QMessageBox.critical(self, "Fehler", str(e))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space:
            self._toggle()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._stop()
        super().closeEvent(event)


# ─── Start ─────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()