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

The CQT reveals the overtone spectrum of each string.
At each detected fundamental `f₀` the code reads the CQT magnitude at the
bin corresponding to `k f₀` for `k = 1 … N_HARMONICS` (default 16).
These measured amplitudes drive the roughness formula, so the
curve/surface shape reflects the **actual** timbral richness of the
instruments.  The shape changes gradually (exponential smoothing) as the
playing changes.

### Note detection: CQT harmonic salience

Simultaneous notes are found with a **Constant-Q Transform** and a
vectorised harmonic-salience function:

1. Compute a Blackman–Harris-windowed FFT; interpolate to log-spaced CQT bins.
2. For each CQT bin *k* (candidate fundamental), sum magnitudes at harmonics
   `k + bpo·log₂(h)` for `h = 1 … N_HARMONICS` — a fixed offset in log-frequency.
3. The peak salience bin is the dominant fundamental.
4. Mask its harmonics (±1.5 bins) in the CQT and repeat for subsequent notes.

CQT salience outperforms linear-FFT HPS for music because the harmonic
series maps to **constant bin offsets** at all pitches simultaneously,
and masking is precise in log-frequency space.

### Bow position and overtone richness

The synthesiser's spectral envelope is derived from **Helmholtz bow-motion
theory**.  For a string bowed at fractional contact point **β** from the bridge
(0 = bridge, 1 = nut), the amplitude of harmonic *h* in the ideal Helmholtz
triangular velocity wave is:

```
A_h  ∝  |sin(h π β)| / h
```

This comes directly from the Fourier series of a sawtooth wave with its
kink at position β.

| β range | Playing style | Character |
|---|---|---|
| 0.03–0.06 | Sul ponticello (near bridge) | Bright, glassy — all harmonics roughly equal because sin(hπβ) ≈ hπβ |
| 0.07–0.12 | Normal bowing | Warm — harmonics peak around *h* ≈ 1/(2β) then fall off |
| 0.13–0.25 | Sul tasto (near fingerboard) | Dark, flute-like — harmonics above *h* ≈ 1/β are suppressed; nodes appear wherever *hβ* is an integer |

The amplitudes are sum-normalised so overall volume stays constant across
bow positions.  The **Harmonics** slider sets how many overtones are
synthesised (4–20); more harmonics = richer timbre and more salience peaks
for the CQT detector to work with.

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
| **Microphone / Synthesiser** radio | Toggle audio source |
| **2 notes – curve / 3 notes – contour / 3 notes – 3D** radio | View mode |
| **FFT window** slider | Analysis window width (ms). Wider → better pitch resolution; narrower → faster response |
| **Harmonics** slider | Number of overtones synthesised (4–20). More = richer timbre |
| **Bow pos** slider | Spectral envelope. Left (sul ponticello) = bright, many harmonics. Right (sul tasto) = dark, few harmonics |
| **SPACE bar** | Advance to next random chord (synthesiser mode only) |
| **Mouse drag** (3-D surface) | Rotate the 3-D dissonance surface |

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
