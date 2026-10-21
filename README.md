# Audio Oscilloscope & Live Analyzer

Eine grafische Python-Anwendung zur Echtzeit-Analyse und visuellen Oszilloskop-Darstellung von Audio-Samples beliebiger Länge.

## Ziel der Anwendung
Das Programm ermöglicht die präzise Analyse von Audio-Files (WAV, FLAC, MP3). Es stellt das Audiosignal in einer echten Oszilloskop-Darstellung dar, bei der das Signal rechts am Playhead (0 ms) eintritt und flüssig nach links durch das Koordinatensystem wandert.

Zusätzlich bietet die Anwendung:
- Einen **4/4-Sequencer**, der geladene Samples rhythmisch zur eingestellten BPM im Takt abspielt.
- **Live-Meter-Anzeigen** für integrierte Lautstärke (LUFS), Peak- und RMS-Pegel in Echtzeit.
- Einen synchronisierten **Audio Output (Rückgabe)** mit eigener Lautstärkeregelung.
- Ein **Timeline-Scrubbing** zum manuellen Spulen durch das Audio-Material per Mausklick und Drag.

## Technische Voraussetzungen
- **Python-Version:** 3.10 oder neuer
- **Betriebssystem:** macOS, Windows, Linux

## Benötigte Bibliotheken
Die benötigten Abhängigkeiten sind in `requirements.txt` definiert:
- `PyQt6` (GUI-Framework)
- `numpy` (Numerische Signalverarbeitung & Array-Operationen)
- `sounddevice` (Audio-Ausgabe & Stream-Handling)
- `soundfile` (Echtzeit-Dateizugriff)
- `librosa` (Audio-Laden & Decodierung)
- `scipy` (Erweiterte Signalverarbeitung)
- `pyloudnorm` (Lautstärkemessung nach ITU-R BS.1770-4)

## Installation
1. Repository klonen oder Projektordner in PyCharm öffnen.
2. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Auf macOS/Linux
   .venv\Scripts\activate     # Auf Windows
3. Benötigte Bibliotheken aus der requirements.txt installieren:
   ```bash
   pip install -r requirements.txt
4. Programm starten 
   ```bash
   python main.py
   

## Wichtigste Funktionen
- **Sample-Loading: Laden von zwei unabhängigen Audio-Files (z. B. Kick/Clap und Bassline) über die Buttons in der Toolbar**
- **4/4-Beat-Sequencer: Automatische Anpassung und zeitliche Ausrichtung der Samples im Taktraster basierend auf der wählbaren BPM**
- **Dual-Channel Oszilloskop: Synchronisierte Live-Visualisierung beider Wellenformen mit einstellbarer Anzeigegeschwindigkeit via "Scope Speed"-Regler**
- **Timeline-Scrubbing: Klick- und Drag-Funktion direkt im Wellenform-Display, um die Wiedergabe an eine beliebige Stelle der Schleife zu setzen**
- **Live-Pegelanzeigen: Analoge VU-Meter für LUFS-, Peak- und RMS-Werte zur Überwachung der Ausgangsdynamik.**
- **Tastatursteuerung: Starten und Stoppen der Wiedergabe bequem per Leertaste.**

## Projektstruktur
- **main.py: Zentraler Startpunkt des Programms; enthält das PyQt6-Hauptfenster, Custom-Anzeigewidgets (WaveformViewport, KnobControl, AnalogMixerMeter) und die AudioEngine.**
- **analyzer.py: Modul/Service zur Audioanalyse, Spektrumsberechnung, Pegelauswertung und Metadaten-Extraktion.**
- **models.py: Datenmodell (AudioMetadata) zur Kapselung und Strukturierung aller relevanten Datei- und Audioinformationen.**
- **utils.py: Reine Hilfsfunktionen für mathematische Berechnungen und Signalverarbeitung (dB-Konvertierung, Fensterung, RMS- und Clamp-Funktionen).**