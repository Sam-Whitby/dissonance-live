# Dissonance Live Visualiser

Real-time visualisation of harmonic dissonance as you play.
Based on William Sethares' psychoacoustic roughness model and the Python
implementations at
- <https://gist.github.com/endolith/3066664>
- <https://github.com/aatishb/dissonance>

---

## What it shows

| Panel | Description |
|---|---|
| **Main plot (left)** | Sethares dissonance curve (2-note mode) or surface (3-note mode). Your live chord is marked with a dot; its path through the space is drawn as a coloured trajectory line. The shape of the curve/surface changes in real time as the richness of your sound changes. |
| **Spectrum (right)** | Log-frequency FFT of the microphone. Detected fundamentals are shown as coloured vertical lines; their harmonics as dashed lines. Inspired by the [OSP Sound Analyzer](https://www.compadre.org/osp/pwa/soundanalyzer/). |
| **Time series (bottom)** | Dissonance of the live sound plotted against time, so you can see how the roughness evolves. |

---

## Theory

### Sethares roughness

Two pure tones at frequencies *f₁*, *f₂* with amplitudes *a₁*, *a₂* produce
roughness

```
r = a₁ a₂ [ exp(−b₁ s Δf) − exp(−b₂ s Δf) ]
```

where `Δf = |f₂ − f₁|`, `s = d* / (s₁ min(f) + s₂)`, and the constants
`b₁ = 3.5`, `b₂ = 5.75`, `s₁ = 0.0207`, `s₂ = 18.96`, `d* = 0.24` are
fitted to Plomp–Levelt consonance data.

For complex tones the roughness is summed over every pair of partials.

### Dissonance curve (2-note mode)

Note 1 is held at a reference pitch (detected fundamental).
The horizontal axis is the ratio `r₂ = f₂/f₁`.
The curve shows the total roughness of the two-note chord as a function of
that ratio.

### Dissonance surface (3-note mode)

Three notes with ratios `r₂ = f₂/f₁` and `r₃ = f₃/f₁`.
The 2-D grid of `(r₂, r₃)` values maps to a surface of total roughness.
Your live chord is a point on (or above) this surface.

### Harmonic amplitudes

The FFT reveals the overtone spectrum of each string.
At each detected fundamental `f₀` the code reads the FFT amplitude at
`k f₀` for `k = 1 … n_harmonics`.
These measured amplitudes replace the synthetic `1/k` decay in the
roughness formula, so the curve/surface shape reflects the **actual** timbral
richness of the instruments in the room.
The shape changes gradually (exponential smoothing) as the playing changes.

### Note detection: iterative HPS

Multiple simultaneous notes are found by **iterative Harmonic Product
Spectrum**:

1. Compute the Blackman–Harris-windowed FFT magnitude.
2. Apply the HPS (multiply spectrum with its own downsampled copies ×2, ×3, …
   ×5). The peak in the result is the dominant fundamental `f₀₁`.
3. Zero out the spectral content near every harmonic of `f₀₁`.
4. Repeat on the residual spectrum to find `f₀₂`, then `f₀₃`.

If fewer notes than expected are detected the loudest found note is
duplicated (so the plot stays in the same dimensional space).
If more are detected, only the three loudest are kept.

---

## Installation

### macOS / Linux

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# On macOS you may also need:
#   brew install portaudio   (before pip install sounddevice)
# On Ubuntu/Debian:
#   sudo apt-get install libportaudio2

# 3. Run
python main.py
```

### Windows

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Quick one-liner (no venv)

```bash
pip install numpy scipy matplotlib sounddevice
python main.py
```

---

## Controls

| Control | Function |
|---|---|
| **2 notes / 3 notes** radio | Switch between 2-D curve and 3-D surface mode |
| **FFT window** slider | Width of the analysis window in milliseconds. Wider → better pitch resolution; narrower → faster response |
| **Harmonics** slider | How many overtones to include in the dissonance calculation |
| **Clear path** button | Erase the trajectory drawn on the main plot |
| **Mouse drag** (3-D surface) | Rotate the surface for a better viewing angle |

---

## Running as a desktop app

### macOS – create an app bundle with PyInstaller

```bash
pip install pyinstaller
pyinstaller --onefile --windowed main.py
# Produces dist/main.app (or dist/main on Linux)
```

### Shortcut scripts

**run.sh** (macOS / Linux):
```bash
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
python main.py
```

**run.bat** (Windows):
```bat
@echo off
cd /d "%~dp0"
call venv\Scripts\activate 2>nul
python main.py
```

---

## Tips for a string trio

- **Start in 2-note mode** and have two players sustain a dyad.
  Watch where the dot sits on the curve and how it moves as intonation shifts.
- **Switch to 3-note mode** for full-chord visualisation.
  Try a pure 4:5:6 major chord and observe how the dot sits in a valley.
- **Wider FFT window** (≥ 500 ms) gives cleaner fundamental detection for
  sustained tones. For fast passages use ~150–200 ms.
- The trajectory line records the last ~30 seconds of motion so you can see
  the harmonic journey of an improvisation.
- The surface shape itself will slowly "breathe" as the mix of overtones in
  the room changes — watch the peaks rise when players add bow pressure.

---

## File structure

```
dissonance_live/
├── main.py          # Full application (single file)
├── requirements.txt
└── README.md
```

---

## Licence

MIT. Dissonance model parameters from Sethares (2004), roughness function
from endolith's gist (public domain).
