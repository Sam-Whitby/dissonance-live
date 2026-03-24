#!/usr/bin/env python3
"""
Dissonance Live Visualizer
==========================
Real-time visualisation of harmonic dissonance based on William Sethares' theory.

Source modes  (bottom-left radio):
  Microphone  – live audio from the default input device
  Synthesiser – synthesised string-instrument chords, played through speakers

View modes (bottom-centre radio):
  2 notes – curve     : 2D dissonance vs frequency-ratio
  3 notes – contour   : 2D colour-contour dissonance map (default for 3-note)
  3 notes – 3D        : interactive 3D surface

Pitch detection uses a Constant-Q Transform (CQT) with a harmonic-salience
function for robust multi-pitch detection in polyphonic signals.

Usage:  python main.py
"""

import time
import threading
import warnings
from collections import deque

import numpy as np
from scipy.signal import windows as sig_windows

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("sounddevice not found – synthesiser output disabled.")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 44100
DEFAULT_WIN_MS    = 500
MIN_WIN_MS        = 100
MAX_WIN_MS        = 2000

N_HARMONICS       = 16          # max harmonics tracked in analysis / dissonance model
SYNTH_CHUNK       = 2048        # audio chunk size for synthesiser callback
SYNTH_GAIN        = 0.18        # overall output gain (keeps signal from clipping)
SYNTH_N_HARM_DEF  = 12          # default number of synthesised harmonics
SYNTH_BETA_DEF    = 0.07        # default bow position (0.03=sul pont, 0.25=sul tasto)

# Constant-Q Transform parameters
CQT_BPO           = 24          # bins per octave (2 per semitone)
CQT_FMIN          = 55.0        # lowest CQT bin  (A1 ≈ 55 Hz)
CQT_FMAX          = 4400.0      # highest CQT bin
SALIENCE_THRESH   = 0.10        # minimum salience relative to peak to accept a note

RATIO_MIN         = 1.0
RATIO_MAX         = 2.0
CURVE_POINTS      = 300
SURF_POINTS       = 40
HIST_SECONDS      = 20.0
ANIM_INTERVAL_MS  = 120

HW_SMOOTH         = 0.20
HW_CHANGE_THRESH  = 0.005
SURF_SMOOTH       = 0.12
MIN_RMS           = 0.001
HARM_MASK_FRAC    = 0.09

CONTOUR_LEVELS    = 12
CONTOUR_REDRAW_N  = 3

NOTE_COLS  = ['#FF5555', '#55DD55', '#5599FF', '#FFAA00']
LABEL_NOTES = {
    'C2': 65.4,  'A2': 110.0, 'E3': 164.8, 'A3': 220.0,
    'C4': 261.6, 'E4': 329.6, 'G4': 392.0, 'A4': 440.0,
    'C5': 523.3, 'A5': 880.0, 'C6': 1046.5,
}
SIMPLE_RATIOS = {
    '1:1': 1.0, '16:15': 16/15, '9:8': 9/8, '6:5': 6/5, '5:4': 5/4,
    '4:3': 4/3, '√2': 2**0.5, '3:2': 3/2, '8:5': 8/5, '5:3': 5/3,
    '16:9': 16/9, '15:8': 15/8, '2:1': 2.0,
}

# ─────────────────────────────────────────────────────────────────────────────
#  NOTE / STAFF UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
_SEM_STEPS  = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_DISP_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
_CHROM2DIAT = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6]   # chromatic → diatonic index

def note_to_freq(name: str) -> float:
    """'C4', 'Eb4', 'F#3', 'Ab3' → Hz  (A4 = 440 Hz, equal temperament)."""
    letter = name[0]
    acc    = 1 if '#' in name else (-1 if 'b' in name else 0)
    octave = int(name[-1])
    semi   = _SEM_STEPS[letter] + acc
    midi   = 12 * (octave + 1) + semi
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def midi_to_note(midi: int) -> str:
    """Convert MIDI note number to name string, e.g. 60 → 'C4', 61 → 'Db4'."""
    return f"{_DISP_NAMES[midi % 12]}{midi // 12 - 1}"


def freq_to_staff(freq: float):
    """
    Map a frequency to a grand-staff y-coordinate.

    Convention
    ----------
    y = 0   : middle C (C4, ledger line between treble & bass)
    y = 0.5 : each diatonic scale step

    Treble staff lines : y = 1, 2, 3, 4, 5   (E4, G4, B4, D5, F5)
    Bass   staff lines : y = -1, -2, -3, -4, -5  (A3, F3, D3, B2, G2)

    Returns
    -------
    (display_name, staff_y, is_accidental)
    """
    midi   = int(round(69 + 12 * np.log2(max(freq, 20.0) / 440.0)))
    chroma = midi % 12
    # C4 = MIDI 60;  60 // 12 = 5  → oct_from_c4 = 0
    oct_rel = midi // 12 - 5
    diat    = _CHROM2DIAT[chroma]
    steps   = oct_rel * 7 + diat
    return _DISP_NAMES[chroma], steps * 0.5, ('#' in _DISP_NAMES[chroma] or
                                               'b' in _DISP_NAMES[chroma])


# ─────────────────────────────────────────────────────────────────────────────
#  SETHARES DISSONANCE MODEL
# ─────────────────────────────────────────────────────────────────────────────
_B1, _B2    = 3.5, 5.75
_S1, _S2    = 0.0207, 18.96
_DSTAR      = 0.24


def _rough(fi, fj, ai, aj):
    f_lo = np.minimum(fi, fj)
    diff = np.maximum(fi, fj) - f_lo
    s    = _DSTAR / (_S1 * f_lo + _S2)
    return np.maximum(ai * aj * (np.exp(-_B1*s*diff) - np.exp(-_B2*s*diff)), 0.0)


def dissonance_curve(f0, hw, ratios):
    hw   = np.asarray(hw, float)
    if hw.max() > 0: hw = hw / hw.max()
    kv   = np.arange(1, len(hw) + 1, dtype=float)
    f1v  = f0 * kv
    f2m  = np.outer(ratios, f0 * kv)
    n_r  = len(ratios)
    d_n1 = sum(_rough(f1v[i], f1v[i+1:], hw[i], hw[i+1:]).sum()
                for i in range(len(hw) - 1))
    total = np.full(n_r, d_n1)
    for i in range(len(hw) - 1):
        total += _rough(f2m[:, i][:, None], f2m[:, i+1:], hw[i], hw[i+1:]).sum(axis=1)
    for i in range(len(hw)):
        for j in range(len(hw)):
            total += _rough(f1v[i], f2m[:, j], hw[i], hw[j])
    return total


def dissonance_surface(f0, hw, r2v, r3v):
    hw  = np.asarray(hw, float)
    if hw.max() > 0: hw = hw / hw.max()
    kv  = np.arange(1, len(hw) + 1, dtype=float)
    f1v = f0 * kv
    f2m = np.outer(r2v, f0 * kv)
    f3m = np.outer(r3v, f0 * kv)
    nh  = len(hw)
    Z   = np.zeros((len(r3v), len(r2v)))
    Z  += sum(_rough(f1v[i], f1v[i+1:], hw[i], hw[i+1:]).sum() for i in range(nh-1))
    d2  = np.zeros(len(r2v))
    for i in range(nh-1):
        d2 += _rough(f2m[:,i][:,None], f2m[:,i+1:], hw[i], hw[i+1:]).sum(axis=1)
    Z  += d2[np.newaxis, :]
    d3  = np.zeros(len(r3v))
    for i in range(nh-1):
        d3 += _rough(f3m[:,i][:,None], f3m[:,i+1:], hw[i], hw[i+1:]).sum(axis=1)
    Z  += d3[:, np.newaxis]
    c12 = np.zeros(len(r2v))
    for i in range(nh):
        for j in range(nh): c12 += _rough(f1v[i], f2m[:,j], hw[i], hw[j])
    Z  += c12[np.newaxis, :]
    c13 = np.zeros(len(r3v))
    for i in range(nh):
        for j in range(nh): c13 += _rough(f1v[i], f3m[:,j], hw[i], hw[j])
    Z  += c13[:, np.newaxis]
    for i in range(nh):
        for j in range(nh):
            Z += _rough(f2m[:,i][np.newaxis,:], f3m[:,j][:,np.newaxis], hw[i], hw[j])
    return Z


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANT-Q TRANSFORM  (CQT)
#
#  Implementation: FFT → CQT by log-frequency interpolation, then a
#  vectorised harmonic-salience function for multi-pitch detection.
#
#  Advantage over linear-FFT HPS for music:
#   • Better frequency resolution at low pitches (proportional to f, not fixed).
#   • Harmonic series maps to constant bin-offsets in the log-frequency domain,
#     making the salience sum exact at all pitches simultaneously.
#   • Iterative masking removes harmonics precisely in log-frequency space,
#     leaving fewer artefacts for the next note's detection.
# ─────────────────────────────────────────────────────────────────────────────
class CQTAnalyzer:
    def __init__(self, sr=SAMPLE_RATE, fmin=CQT_FMIN, fmax=CQT_FMAX,
                 bpo=CQT_BPO, n_harmonics=N_HARMONICS):
        self.sr  = sr
        self.bpo = bpo
        n_octaves   = np.log2(fmax / fmin)
        self.n_bins = int(np.ceil(bpo * n_octaves))
        self.freqs  = fmin * 2 ** (np.arange(self.n_bins) / bpo)  # CQT centre freqs
        self.Q      = 1.0 / (2 ** (1.0 / bpo) - 1)                # Q factor

        # Harmonic bin offsets (fractional): for harmonic h, offset = bpo*log2(h)
        self._harm_offsets = np.array([bpo * np.log2(h)
                                       for h in range(1, n_harmonics + 1)])
        # Weights: 1/h decay emphasises the fundamental
        self._harm_weights = 1.0 / np.arange(1, n_harmonics + 1, dtype=float)

    # ── CQT magnitude from FFT ────────────────────────────────────────────────

    def from_fft(self, fft_freqs: np.ndarray, fft_mag: np.ndarray) -> np.ndarray:
        """
        Map a standard FFT magnitude spectrum to CQT bins by linear interpolation
        in log-frequency.  This is O(n_bins) and runs in a few microseconds.
        """
        return np.interp(self.freqs, fft_freqs, fft_mag, left=0.0, right=0.0)

    # ── Harmonic salience ─────────────────────────────────────────────────────

    def harmonic_salience(self, cqt_mag: np.ndarray) -> np.ndarray:
        """
        Compute the harmonic salience for each CQT bin.

        For bin k (centre frequency f_k), the salience is the weighted sum of
        CQT magnitudes at the harmonic frequencies f_k, 2f_k, 3f_k, …

        In log-frequency space these harmonics lie at fixed bin offsets
        (bpo * log2(h)), so the sum is a simple strided gather — the key
        advantage of CQT over linear FFT for pitch detection.
        """
        n       = self.n_bins
        sal     = np.zeros(n)
        src_idx = np.arange(n, dtype=float)   # bin indices of the 'fundamental'

        for offset, w in zip(self._harm_offsets, self._harm_weights):
            harm_bins = src_idx + offset          # target bin indices (float)
            lo        = np.floor(harm_bins).astype(int)
            frac      = harm_bins - lo
            valid     = (lo >= 0) & (lo < n - 1)
            sal[valid] += w * (
                (1 - frac[valid]) * cqt_mag[lo[valid]] +
                     frac[valid]  * cqt_mag[lo[valid] + 1]
            )
        return sal

    # ── Multi-pitch detection ─────────────────────────────────────────────────

    def find_notes(self, cqt_mag: np.ndarray, n_notes: int,
                   fmin: float = CQT_FMIN) -> list:
        """
        Iterative CQT salience peak-picking for multi-pitch detection.

        1. Compute harmonic salience.
        2. Find the peak → first fundamental f₀.
        3. Mask its harmonic bins in BOTH salience and CQT magnitude.
        4. Repeat for subsequent notes.

        Returns list of (f0_hz, harm_amp_array) sorted by ascending frequency.
        """
        n         = self.n_bins
        valid     = self.freqs >= fmin

        sal_w     = self.harmonic_salience(cqt_mag)
        cqt_w     = cqt_mag.copy()

        # Adaptive threshold: fraction of the peak salience in the valid range
        peak_sal  = sal_w[valid].max() if valid.any() else 1e-15
        threshold = SALIENCE_THRESH * peak_sal

        notes = []
        for _ in range(n_notes):
            masked = np.where(valid, sal_w, 0.0)
            if masked.max() < threshold:
                break
            k0 = int(np.argmax(masked))

            # Parabolic interpolation in log-frequency for sub-bin f0 accuracy
            if 0 < k0 < n - 1:
                y0, y1, y2 = sal_w[k0-1], sal_w[k0], sal_w[k0+1]
                den = 2.0 * (2*y1 - y0 - y2)
                delta = (y2 - y0) / den if den > 0 else 0.0
                f0 = self.freqs[k0] * 2 ** (delta / self.bpo)
            else:
                f0 = self.freqs[k0]

            # Extract harmonic amplitudes from the (unmasked) CQT
            harm_amps = np.zeros(len(self._harm_offsets))
            for h, offset in enumerate(self._harm_offsets):
                hb = k0 + offset
                lo = int(hb); frac = hb - lo
                if 0 <= lo < n - 1:
                    harm_amps[h] = (1-frac)*cqt_mag[lo] + frac*cqt_mag[lo+1]

            if harm_amps[0] < threshold * 0.05:
                break
            notes.append((f0, harm_amps))

            # Mask harmonics  ±1.5 bins wide in both sal_w and cqt_w
            for offset in self._harm_offsets:
                hb   = k0 + offset
                i_lo = max(0, int(hb - 1.5))
                i_hi = min(n, int(hb + 2.5))
                sal_w[i_lo:i_hi] = 0.0
                cqt_w[i_lo:i_hi] = 0.0

        notes.sort(key=lambda x: x[0])
        return notes


# ─────────────────────────────────────────────────────────────────────────────
#  SYNTHESISER
#
#  Generates string-like chords (10 harmonics, A_h = 1/h^1.5 decay) as a
#  continuous background thread.  Audio is simultaneously:
#    • Injected into the AudioProcessor buffer for analysis.
#    • Written to a sounddevice OutputStream for playback through speakers.
#
#  A gentle vibrato (5 Hz, ±0.4 %) makes the timbre more natural and aids
#  CQT pitch detection by spreading each partial slightly.
# ─────────────────────────────────────────────────────────────────────────────
class Synthesiser:
    """
    Generates string-like chords on demand with a physically-motivated
    spectral envelope based on Helmholtz bow-motion theory.

    See README §Bow position for the derivation.  Audio is produced
    directly inside the sounddevice OutputStream callback — no inter-thread
    buffer, no rate-mismatch artefacts.  A single phase accumulator
    (_phase, audio-thread private) gives sample-accurate continuity.
    """

    def __init__(self, sr=SAMPLE_RATE):
        self.sr            = sr
        self._lock         = threading.Lock()
        self._active       = False
        self._n_notes      = 2
        self._advance_flag = False
        self._audio_proc   = None
        self._out_stream   = None
        self._running      = False
        self._phase        = 0.0          # audio-thread only
        self._rng          = np.random.default_rng()

        # Bow-position spectral parameters (user-adjustable)
        self._beta        = SYNTH_BETA_DEF    # contact point from bridge (0=bridge)
        self._n_harmonics = SYNTH_N_HARM_DEF
        self._harm_amps   = self._bow_amps(self._beta, self._n_harmonics)

        # Public readable state (protected by _lock)
        self.chord_name : str  = ''
        self.note_names : list = []
        self.note_freqs : list = []
        with self._lock:
            self._new_random_chord()   # populate with first chord

    # ── public API ────────────────────────────────────────────────────────────

    def start(self, audio_proc):
        self._audio_proc = audio_proc
        if AUDIO_AVAILABLE:
            try:
                self._out_stream = sd.OutputStream(
                    samplerate=self.sr, channels=1, dtype='float32',
                    blocksize=SYNTH_CHUNK, callback=self._out_cb)
                self._out_stream.start()
                return
            except Exception as e:
                print(f"Audio output unavailable: {e}")
        self._running = True
        threading.Thread(target=self._inject_loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._out_stream:
            try: self._out_stream.stop(); self._out_stream.close()
            except Exception: pass

    def set_active(self, active: bool, n_notes: int = 2):
        with self._lock:
            changed = (n_notes != self._n_notes)
            self._active  = active
            self._n_notes = n_notes
            if changed:
                self._new_random_chord()

    def next_chord(self):
        """Queue a new random chord — consumed by the audio-producing thread."""
        with self._lock:
            self._advance_flag = True

    def set_bow_pos(self, beta: float):
        """β = bow contact point / string length  (0.03 = bridge, 0.25 = fingerboard)."""
        with self._lock:
            self._beta      = float(beta)
            self._harm_amps = self._bow_amps(self._beta, self._n_harmonics)

    def set_n_harmonics(self, n: int):
        with self._lock:
            self._n_harmonics = int(n)
            self._harm_amps   = self._bow_amps(self._beta, self._n_harmonics)

    def get_state(self):
        with self._lock:
            return self.chord_name, list(self.note_names), list(self.note_freqs)

    # ── spectral envelope ─────────────────────────────────────────────────────

    @staticmethod
    def _bow_amps(beta: float, n: int) -> np.ndarray:
        """
        Helmholtz bow-motion spectral envelope.

        For a string bowed at fractional position β from the bridge, the
        amplitude of harmonic h in ideal Helmholtz motion is:

            A_h  ∝  |sin(h π β)| / h

        Derivation: the string velocity has a triangular Helmholtz wave
        whose Fourier coefficients are sin(hπβ)/h (standard sawtooth
        with the kink at β).  Near the bridge (β → 0), sin(hπβ) ≈ hπβ,
        so A_h ≈ πβ — all harmonics are roughly equal (bright, glassy).
        As β increases toward 0.25 (sul tasto), higher harmonics are
        increasingly suppressed and nodes appear where hβ is an integer.
        """
        h    = np.arange(1, n + 1, dtype=float)
        amps = np.abs(np.sin(h * np.pi * beta)) / h
        s    = amps.sum()
        if s > 0:
            amps /= s           # unit-sum → peak stays within SYNTH_GAIN
        return amps

    # ── random chord ──────────────────────────────────────────────────────────

    def _new_random_chord(self):
        """
        Pick a new random chord with all notes within one octave.
        Caller must hold self._lock.

        Root: MIDI 45–72  (A2–C5)
        2-note: root + 1–12 semitones
        3-note: root + i1 + i2  where 2 ≤ i1 < i2 ≤ 12
        """
        root = int(self._rng.integers(45, 73))
        if self._n_notes == 2:
            midis = [root, root + int(self._rng.integers(1, 13))]
        else:
            i1 = int(self._rng.integers(2, 11))
            i2 = int(self._rng.integers(i1 + 1, 13))
            midis = [root, root + i1, root + i2]
        freqs = [440.0 * 2.0 ** ((m - 69) / 12.0) for m in midis]
        self.chord_name = '  '.join(midi_to_note(m) for m in midis)
        self.note_names = [freq_to_staff(f) for f in freqs]
        self.note_freqs = freqs

    # ── audio generation ──────────────────────────────────────────────────────

    def _gen(self, frames: int, freqs: list, amps: np.ndarray) -> np.ndarray:
        """
        Synthesise `frames` samples.  `amps` is a snapshot taken under the
        lock so we never hold it during the (slow) synthesis loop.
        """
        t   = np.arange(frames, dtype=float) / self.sr + self._phase
        sig = np.zeros(frames)
        n   = max(len(freqs), 1)
        for f in freqs:
            for h_i, amp in enumerate(amps, start=1):
                sig += (amp / n) * np.sin(2.0 * np.pi * f * h_i * t)
        self._phase += frames / self.sr
        return (sig * SYNTH_GAIN).astype(np.float32)

    # ── sounddevice callback ───────────────────────────────────────────────────

    def _out_cb(self, outdata, frames, time_info, status):
        with self._lock:
            if self._advance_flag:
                self._advance_flag = False
                self._new_random_chord()
            active = self._active
            freqs  = list(self.note_freqs)
            amps   = self._harm_amps.copy()   # snapshot — don't hold lock during synthesis

        if active and freqs:
            audio = self._gen(frames, freqs, amps)
            outdata[:, 0] = audio
            if self._audio_proc:
                self._audio_proc.inject(audio)
        else:
            outdata[:] = 0.0

    # ── fallback inject loop ───────────────────────────────────────────────────

    def _inject_loop(self):
        dt = SYNTH_CHUNK / self.sr
        while self._running:
            with self._lock:
                if self._advance_flag:
                    self._advance_flag = False
                    self._new_random_chord()
                active = self._active
                freqs  = list(self.note_freqs)
                amps   = self._harm_amps.copy()
            if active and freqs and self._audio_proc:
                self._audio_proc.inject(self._gen(SYNTH_CHUNK, freqs, amps))
            time.sleep(dt)


# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class AudioProcessor:
    def __init__(self, sr=SAMPLE_RATE):
        self.sr          = sr
        self.win_samples = int(DEFAULT_WIN_MS * sr / 1000)
        buf_len          = int(MAX_WIN_MS * sr / 1000) * 3
        self._buf        = np.zeros(buf_len)
        self._buf_lock   = threading.Lock()
        self._wpos       = 0
        self._stream     = None
        self.running     = False
        self.accept_mic  = True   # set False in synth mode

        self.result_lock   = threading.Lock()
        self.fundamentals  : list = []
        self.harm_amps     : list = []
        self.avg_harmonics        = np.array([1.0/k for k in range(1, N_HARMONICS+1)])
        self.avg_harmonics       /= self.avg_harmonics.max()
        self.rms          = 0.0
        self.fft_freqs    = np.array([1.0])
        self.fft_mag      = np.array([0.0])
        self.cqt_mag      = np.array([0.0])

    def _callback(self, indata, frames, time_info, status):
        if not self.accept_mic:
            return
        data = indata[:, 0]
        with self._buf_lock:
            n = len(data); end = self._wpos + n; L = len(self._buf)
            if end <= L:
                self._buf[self._wpos:end] = data
            else:
                f = L - self._wpos
                self._buf[self._wpos:] = data[:f]; self._buf[:end-L] = data[f:]
            self._wpos = end % L

    def _get_window(self):
        n = self.win_samples
        with self._buf_lock:
            pos = self._wpos; buf = self._buf.copy()
        L = len(buf); s = (pos - n) % L
        return buf[s:s+n] if s+n <= L else np.concatenate([buf[s:], buf[:n-(L-s)]])

    def inject(self, chunk: np.ndarray):
        with self._buf_lock:
            n = len(chunk); end = self._wpos + n; L = len(self._buf)
            if end <= L:
                self._buf[self._wpos:end] = chunk
            else:
                f = L - self._wpos
                self._buf[self._wpos:] = chunk[:f]; self._buf[:end-L] = chunk[f:]
            self._wpos = end % L

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sr, channels=1,
            callback=self._callback, blocksize=1024)
        self._stream.start(); self.running = True

    def stop(self):
        if self._stream: self._stream.stop(); self._stream.close()
        self.running = False

    def _fft(self, win):
        w   = sig_windows.blackmanharris(len(win))
        mag = np.abs(np.fft.rfft(win * w)) / w.sum()
        frq = np.fft.rfftfreq(len(win), 1.0 / self.sr)
        return frq, mag

    def analyse(self, n_expected: int, cqt: CQTAnalyzer):
        """
        Analyse the current window using CQT harmonic salience.
        Fills self.fundamentals, harm_amps, avg_harmonics, rms, fft_*, cqt_mag.
        """
        win  = self._get_window()
        rms  = float(np.sqrt(np.mean(win ** 2)))
        frqs, mag = self._fft(win)

        cqt_mag = cqt.from_fft(frqs, mag)

        notes  = cqt.find_notes(cqt_mag, n_expected)

        # Pad / trim to exactly n_expected
        notes.sort(key=lambda x: x[0])
        if notes:
            loudest = max(notes, key=lambda x: x[1][0])
            while len(notes) < n_expected:
                notes.append(loudest)
            if len(notes) > n_expected:
                notes.sort(key=lambda x: -x[1][0])
                notes = notes[:n_expected]
                notes.sort(key=lambda x: x[0])

        if notes:
            all_ha = np.array([n[1] for n in notes])
            avg    = all_ha.mean(axis=0)
            if avg.max() > 0: avg /= avg.max()
        else:
            avg = None

        with self.result_lock:
            self.fundamentals  = [n[0] for n in notes]
            self.harm_amps     = [n[1] for n in notes]
            self.avg_harmonics = avg
            self.rms           = rms
            self.fft_freqs     = frqs
            self.fft_mag       = mag
            self.cqt_mag       = cqt_mag

    def set_win_ms(self, ms: int):
        self.win_samples = int(ms * self.sr / 1000)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class App:
    """
    view_mode : 'curve' | 'contour' | 'surface'
    source    : 'mic'   | 'synth'
    """
    def __init__(self):
        self.audio     = AudioProcessor()
        self.cqt       = CQTAnalyzer(SAMPLE_RATE)
        self.synth     = Synthesiser(SAMPLE_RATE)

        self.view_mode = 'curve'
        self.n_notes   = 2
        self.source    = 'mic'

        self.ratios = np.linspace(RATIO_MIN, RATIO_MAX, CURVE_POINTS)
        self.sr2    = np.linspace(RATIO_MIN, RATIO_MAX, SURF_POINTS)
        self.sr3    = np.linspace(RATIO_MIN, RATIO_MAX, SURF_POINTS)

        self.curve_z   = np.zeros(CURVE_POINTS)
        self.surf_z    = np.zeros((SURF_POINTS, SURF_POINTS))
        self.surf_z_sm = np.zeros_like(self.surf_z)

        self.f0_ref   = 220.0
        self.r2       = 1.5
        self.r3       = 1.33
        self.diss_now = 0.0

        self.hw = np.array([1.0/k for k in range(1, N_HARMONICS+1)])
        self.hw /= self.hw.max()

        n_hist = int(HIST_SECONDS * 1000 / ANIM_INTERVAL_MS) + 10
        self.diss_hist = deque(maxlen=n_hist)
        self.time_hist = deque(maxlen=n_hist)
        self.t0        = time.time()

        self._frame    = 0
        self._az       = -60.0
        self._el       =  30.0

        # Stave artists (populated by _init_stave_ax)
        self._stave_notes    = []   # Circle patches
        self._stave_texts    = []   # note-name Text objects
        self._stave_ledgers  = []   # ledger-line Line2D objects
        self._stave_shadow   = []   # shadow/glow circles

        # Contour artists
        self._pcm         = None
        self._cbar        = None
        self._contour_set = None

        self._build_figure()
        self._build_controls()
        self._recompute_curve()
        self._init_2d_ax()

    # ── figure layout ─────────────────────────────────────────────────────────

    def _build_figure(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(17, 10), facecolor='#10101e')
        try:
            self.fig.canvas.manager.set_window_title('Dissonance Live Visualizer')
        except Exception:
            pass

        outer = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[3, 1], hspace=0.40,
            left=0.05, right=0.97, top=0.94, bottom=0.24,
        )
        top = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0],
            width_ratios=[3, 2], wspace=0.30,
        )
        right = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=top[1],
            height_ratios=[2, 1], hspace=0.40,
        )
        bottom = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[1],
            width_ratios=[3, 2], wspace=0.30,
        )

        self._gs_main    = top[0]
        self.ax_main     = self.fig.add_subplot(self._gs_main)
        self.ax_spectrum = self.fig.add_subplot(right[0])
        self.ax_stave    = self.fig.add_subplot(right[1])
        self.ax_time     = self.fig.add_subplot(bottom[0])
        # bottom[1] is empty — reserved for layout balance

        for ax in (self.ax_main, self.ax_spectrum, self.ax_stave, self.ax_time):
            _style_ax(ax)

        self._init_spectrum_ax()
        self._init_stave_ax()
        self._init_time_ax()

    # ── 2D curve axes ─────────────────────────────────────────────────────────

    def _init_2d_ax(self):
        ax = self.ax_main
        ax.set_facecolor('#0a0a18')
        ax.set_xlabel('Frequency ratio  f₂ / f₁', color='#9999bb', fontsize=9)
        ax.set_ylabel('Dissonance', color='#9999bb', fontsize=9)
        ax.grid(True, alpha=0.18, color='#222244')
        self.line2d, = ax.plot(self.ratios, self.curve_z,
                               color='#6699ff', lw=1.8, alpha=0.9, zorder=4)
        self.fill2d  = ax.fill_between(self.ratios, 0, self.curve_z,
                                       alpha=0.18, color='#4466cc', zorder=3)
        self.pt2d,   = ax.plot([], [], 'o', color='#ffcc00', ms=12, zorder=8,
                               markeredgecolor='#ffffff', markeredgewidth=0.8)
        ax.set_xlim(RATIO_MIN, RATIO_MAX)
        ax.set_ylim(0, 1.15)
        for label, r in SIMPLE_RATIOS.items():
            if RATIO_MIN <= r <= RATIO_MAX:
                ax.axvline(r, color='#334466', lw=0.8, alpha=0.6)
                ax.text(r, 0.01, label, color='#445577', fontsize=6,
                        ha='center', va='bottom')
        self.status_txt = ax.text(
            0.98, 0.97, '', transform=ax.transAxes,
            color='#aabbff', fontsize=8, ha='right', va='top',
            fontfamily='monospace')

    # ── 2D contour axes ───────────────────────────────────────────────────────

    def _init_contour_ax(self):
        ax = self.ax_main
        ax.set_facecolor('#0a0a18')
        ax.set_xlabel('r₂ = f₂ / f₁', color='#9999bb', fontsize=9)
        ax.set_ylabel('r₃ = f₃ / f₁', color='#9999bb', fontsize=9)
        ax.grid(False)
        R2, R3 = np.meshgrid(self.sr2, self.sr3)
        self._pcm = ax.pcolormesh(
            R2, R3, self.surf_z_sm, cmap='plasma',
            vmin=0, vmax=1, shading='gouraud', zorder=1)
        self._contour_set = ax.contour(
            R2, R3, self.surf_z_sm, levels=CONTOUR_LEVELS,
            colors='white', alpha=0.25, linewidths=0.6, zorder=2)
        self._cbar = self.fig.colorbar(
            self._pcm, ax=ax, fraction=0.03, pad=0.02, label='Dissonance')
        self._cbar.ax.yaxis.label.set_color('#9999bb')
        self._cbar.ax.tick_params(colors='#9999bb', labelsize=7)
        self.pt3_dot,    = ax.plot([], [], 'o', color='white', ms=11,
                                   zorder=6, markeredgecolor='#ffcc00',
                                   markeredgewidth=1.5)
        ax.set_xlim(RATIO_MIN, RATIO_MAX)
        ax.set_ylim(RATIO_MIN, RATIO_MAX)
        ax.set_aspect('equal')
        for label, r in SIMPLE_RATIOS.items():
            if RATIO_MIN <= r <= RATIO_MAX:
                ax.axvline(r, color='white', lw=0.5, alpha=0.20, zorder=3)
                ax.axhline(r, color='white', lw=0.5, alpha=0.20, zorder=3)
                ax.text(r, RATIO_MIN+0.01, label, color='#667799',
                        fontsize=5, ha='center', va='bottom', rotation=90)
                ax.text(RATIO_MIN+0.01, r, label, color='#667799',
                        fontsize=5, ha='left', va='center')
        self.status_txt = ax.text(
            0.98, 0.97, '', transform=ax.transAxes,
            color='#aabbff', fontsize=8, ha='right', va='top',
            fontfamily='monospace', zorder=7)

    # ── 3D surface axes ───────────────────────────────────────────────────────

    def _init_3d_ax(self):
        ax = self.ax_main
        ax.set_title('Dissonance Surface — 3 notes', color='#ccccff', fontsize=11, pad=14)
        ax.set_xlabel('r₂', color='#9999bb', fontsize=8, labelpad=4)
        ax.set_ylabel('r₃', color='#9999bb', fontsize=8, labelpad=4)
        ax.set_zlabel('Dissonance', color='#9999bb', fontsize=8, labelpad=4)
        ax.tick_params(colors='#9999bb', labelsize=7)
        ax.set_facecolor('#0a0a18')
        ax.grid(True, alpha=0.12)
        R2, R3 = np.meshgrid(self.sr2, self.sr3)
        self._surf3d = ax.plot_surface(
            R2, R3, self.surf_z_sm, cmap='plasma',
            alpha=0.82, linewidth=0, antialiased=True)
        self.pt3d,    = ax.plot([], [], [], 'o',
                                color='#ff4400', ms=8, zorder=11)
        ax.set_xlim(RATIO_MIN, RATIO_MAX)
        ax.set_ylim(RATIO_MIN, RATIO_MAX)
        ax.view_init(elev=self._el, azim=self._az)

    # ── spectrum axes ─────────────────────────────────────────────────────────

    def _init_spectrum_ax(self):
        ax = self.ax_spectrum
        ax.set_title('Frequency Spectrum (CQT)', color='#ccccff', fontsize=10, pad=6)
        ax.set_xlabel('Frequency (Hz)', color='#9999bb', fontsize=8)
        ax.set_ylabel('Amplitude', color='#9999bb', fontsize=8)
        ax.set_xscale('log')
        ax.set_xlim(CQT_FMIN * 0.85, CQT_FMAX)
        ax.set_ylim(0, 1.08)
        ax.grid(True, which='both', alpha=0.12, color='#222244')
        # CQT spectrum line (log-spaced, naturally matches the CQT bins)
        self.spec_line, = ax.plot([], [], color='#44ccff', lw=1.0, alpha=0.9, zorder=5)
        self.spec_glow, = ax.plot([], [], color='#44ccff', lw=4.0, alpha=0.12, zorder=4)
        self.spec_fill  = None
        self.f_lines = []
        for i in range(3):
            vl = ax.axvline(CQT_FMIN, color=NOTE_COLS[i], lw=2.2, alpha=0.0, zorder=8)
            self.f_lines.append(vl)
        self._harm_artists = []
        for name, freq in LABEL_NOTES.items():
            if CQT_FMIN * 0.85 <= freq <= CQT_FMAX:
                ax.axvline(freq, color='#1e1e3a', lw=0.7)
                ax.text(freq, -0.07, name, color='#445577', fontsize=5,
                        ha='center', va='top', transform=ax.get_xaxis_transform())

    # ── stave axes ────────────────────────────────────────────────────────────

    def _init_stave_ax(self):
        ax = self.ax_stave
        ax.set_facecolor('#05050f')
        ax.set_xlim(0, 10)
        ax.set_ylim(-7, 7)
        ax.axis('off')

        lw = 0.9
        col = '#ccccdd'

        # Treble staff lines (E4 G4 B4 D5 F5 → y = 1 2 3 4 5)
        for y in [1, 2, 3, 4, 5]:
            ax.plot([1.2, 9.0], [y, y], color=col, lw=lw, zorder=1)

        # Bass staff lines (A3 F3 D3 B2 G2 → y = -1 -2 -3 -4 -5)
        for y in [-1, -2, -3, -4, -5]:
            ax.plot([1.2, 9.0], [y, y], color=col, lw=lw, zorder=1)

        # Thin connector bar on the left
        ax.plot([1.2, 1.2], [-5, 5], color=col, lw=lw*1.5, zorder=1)

        # Clef labels (attempt Unicode music symbols; fall back to letters)
        for sym, fallback, ypos in [('𝄞', 'G', 3.0), ('𝄢', 'F', -3.0)]:
            try:
                ax.text(0.5, ypos, sym, color=col, fontsize=22,
                        ha='center', va='center', zorder=2)
            except Exception:
                ax.text(0.5, ypos, fallback, color=col, fontsize=14,
                        ha='center', va='center', zorder=2, fontstyle='italic')

        # Middle-C dashed ledger guide
        ax.plot([3.5, 6.5], [0, 0], color=col, lw=0.4, ls='--', alpha=0.4, zorder=1)
        ax.text(0.2, 0, 'C4', color='#445566', fontsize=5,
                ha='left', va='center', zorder=2)

        # Title
        ax.set_title('Sounding Notes', color='#ccccff', fontsize=9, pad=4)

        # Chord name text (updated each frame)
        self._chord_txt = ax.text(
            5, 6.3, '', color='#ffdd88', fontsize=9,
            ha='center', va='top', fontweight='bold', zorder=5)
        self._source_txt = ax.text(
            5, -6.3, '', color='#7799bb', fontsize=7,
            ha='center', va='bottom', zorder=5)

    # ── time axes ─────────────────────────────────────────────────────────────

    def _init_time_ax(self):
        ax = self.ax_time
        ax.set_title('Dissonance vs Time', color='#ccccff', fontsize=10, pad=4)
        ax.set_xlabel('Time (s)', color='#9999bb', fontsize=8)
        ax.set_ylabel('Dissonance', color='#9999bb', fontsize=8)
        ax.grid(True, alpha=0.18, color='#222244')
        self.time_line, = ax.plot([], [], color='#ffaa44', lw=1.6, alpha=0.9)
        self.time_fill  = None
        ax.set_xlim(0, HIST_SECONDS)
        ax.set_ylim(0, 1.0)

    # ── controls ──────────────────────────────────────────────────────────────

    def _build_controls(self):
        sl_kw = dict(color='#334466', track_color='#1a1a33')

        # Source radio
        ax_src = plt.axes([0.05, 0.05, 0.14, 0.14])
        ax_src.set_facecolor('#10101e')
        self.radio_src = RadioButtons(
            ax_src, ('Microphone', 'Synthesiser'), active=0, activecolor='#55dd88')
        for lbl in self.radio_src.labels:
            lbl.set_color('#9999bb'); lbl.set_fontsize(8)

        # View mode radio
        ax_view = plt.axes([0.49, 0.04, 0.26, 0.17])
        ax_view.set_facecolor('#10101e')
        self.radio_view = RadioButtons(
            ax_view,
            ('2 notes  –  curve', '3 notes  –  contour', '3 notes  –  3D'),
            active=0, activecolor='#6699ff')
        for lbl in self.radio_view.labels:
            lbl.set_color('#9999bb'); lbl.set_fontsize(8)

        # ── Three stacked sliders (centre column) ─────────────────────────────
        ax_win = plt.axes([0.24, 0.17, 0.20, 0.022])
        self.sl_win = Slider(ax_win, 'FFT window (ms)',
                             MIN_WIN_MS, MAX_WIN_MS,
                             valinit=DEFAULT_WIN_MS, valstep=50, **sl_kw)
        self.sl_win.label.set_color('#9999bb')
        self.sl_win.valtext.set_color('#ccddff')

        ax_nharm = plt.axes([0.24, 0.11, 0.20, 0.022])
        self.sl_nharm = Slider(ax_nharm, 'Harmonics',
                               4, 20, valinit=SYNTH_N_HARM_DEF, valstep=1, **sl_kw)
        self.sl_nharm.label.set_color('#9999bb')
        self.sl_nharm.valtext.set_color('#ccddff')

        ax_bow = plt.axes([0.24, 0.05, 0.20, 0.022])
        self.sl_bow = Slider(ax_bow, 'Bow pos',
                             0.03, 0.25, valinit=SYNTH_BETA_DEF, **sl_kw)
        self.sl_bow.label.set_color('#9999bb')
        self.sl_bow.valtext.set_color('#ccddff')
        # Add endpoint labels so the slider is self-explanatory
        ax_bow.text(0.0, -0.9, 'sul\nponticello', transform=ax_bow.transAxes,
                    color='#667799', fontsize=6, ha='left', va='top')
        ax_bow.text(1.0, -0.9, 'sul\ntasto', transform=ax_bow.transAxes,
                    color='#667799', fontsize=6, ha='right', va='top')

        self.sl_win.on_changed(self._on_win)
        self.sl_nharm.on_changed(self._on_nharm)
        self.sl_bow.on_changed(self._on_bow)
        self.radio_src.on_clicked(self._on_source)
        self.radio_view.on_clicked(self._on_view)

    def _on_win(self, v):
        self.audio.set_win_ms(int(v))

    def _on_nharm(self, v):
        self.synth.set_n_harmonics(int(v))

    def _on_bow(self, v):
        self.synth.set_bow_pos(float(v))

    def _on_source(self, label):
        synth = ('Synth' in label)
        self.source = 'synth' if synth else 'mic'
        self.audio.accept_mic = not synth
        self.synth.set_active(synth, self.n_notes)

    def _on_view(self, label):
        if 'curve' in label:
            mode, n = 'curve', 2
        elif 'contour' in label:
            mode, n = 'contour', 3
        else:
            mode, n = 'surface', 3
        if mode == self.view_mode:
            return
        self.view_mode = mode; self.n_notes = n
        self.synth.set_active(self.source == 'synth', n)
        self._switch_main_ax()

    def _on_key(self, event):
        if event.key == ' ' and self.source == 'synth':
            self.synth.next_chord()

    # ── mode switch ───────────────────────────────────────────────────────────

    def _switch_main_ax(self):
        if self._cbar is not None:
            try:    self._cbar.remove()
            except: pass
            self._cbar = None
        self.ax_main.remove()
        if self.view_mode == 'curve':
            self.ax_main = self.fig.add_subplot(self._gs_main)
            _style_ax(self.ax_main)
            self._recompute_curve()
            self._init_2d_ax()
        elif self.view_mode == 'contour':
            self.ax_main = self.fig.add_subplot(self._gs_main)
            _style_ax(self.ax_main)
            self._recompute_surface()
            self.surf_z_sm = self.surf_z.copy()
            self._init_contour_ax()
        else:
            self.ax_main = self.fig.add_subplot(self._gs_main, projection='3d')
            _style_ax(self.ax_main)
            self._recompute_surface()
            self.surf_z_sm = self.surf_z.copy()
            self._init_3d_ax()

    # ── dissonance recompute ──────────────────────────────────────────────────

    def _recompute_curve(self):
        c = dissonance_curve(self.f0_ref, self.hw, self.ratios)
        if c.max() > 0: c /= c.max()
        self.curve_z = c

    def _recompute_surface(self):
        Z = dissonance_surface(self.f0_ref, self.hw, self.sr2, self.sr3)
        if Z.max() > 0: Z /= Z.max()
        self.surf_z = Z

    # ── animation ─────────────────────────────────────────────────────────────

    def animate(self, frame):
        self._frame += 1

        self.audio.analyse(self.n_notes, self.cqt)

        with self.audio.result_lock:
            funds     = list(self.audio.fundamentals)
            avg_hw    = self.audio.avg_harmonics
            rms       = self.audio.rms
            fft_freqs = self.audio.fft_freqs.copy()
            fft_mag   = self.audio.fft_mag.copy()
            cqt_mag   = self.audio.cqt_mag.copy()

        notes_detected = len(funds) > 0

        # Update hw only when real notes are detected
        hw_changed = False
        if notes_detected and avg_hw is not None:
            n   = min(len(avg_hw), len(self.hw))
            new = HW_SMOOTH * avg_hw[:n] + (1 - HW_SMOOTH) * self.hw[:n]
            if np.max(np.abs(new - self.hw[:n])) > HW_CHANGE_THRESH:
                self.hw[:n] = new
                hw_changed  = True

        if notes_detected:
            self.f0_ref = 0.85 * self.f0_ref + 0.15 * funds[0]

        if hw_changed:
            self._recompute_curve()
            if self.view_mode in ('contour', 'surface'):
                self._recompute_surface()

        if self.view_mode in ('contour', 'surface'):
            self.surf_z_sm = (SURF_SMOOTH * self.surf_z +
                              (1 - SURF_SMOOTH) * self.surf_z_sm)

        # Ratios from detected notes
        if len(funds) >= 2:
            self.r2 = float(np.clip(funds[1] / funds[0], RATIO_MIN, RATIO_MAX))
        if len(funds) >= 3:
            self.r3 = float(np.clip(funds[2] / funds[0], RATIO_MIN, RATIO_MAX))

        # Current dissonance value
        if self.view_mode == 'curve':
            self.diss_now = float(np.interp(self.r2, self.ratios, self.curve_z))
        else:
            self.diss_now = _bilinear(self.sr2, self.sr3, self.surf_z_sm,
                                      self.r2, self.r3)

        # History
        t_now = time.time() - self.t0
        self.time_hist.append(t_now)
        self.diss_hist.append(self.diss_now if notes_detected else float('nan'))

        # ── draw ──────────────────────────────────────────────────────────────
        if self.view_mode == 'curve':
            self._upd_2d(funds)
        elif self.view_mode == 'contour':
            self._upd_contour(funds)
        else:
            self._upd_3d()

        self._upd_spectrum(cqt_mag, funds)
        self._upd_time()
        self._upd_stave(funds)
        return []

    # ── 2D curve update ───────────────────────────────────────────────────────

    def _upd_2d(self, funds):
        ax = self.ax_main
        self.line2d.set_ydata(self.curve_z)
        if self.fill2d is not None: self.fill2d.remove()
        self.fill2d = ax.fill_between(self.ratios, 0, self.curve_z,
                                      alpha=0.18, color='#4466cc', zorder=3)
        ax.set_ylim(0, 1.15)
        self.pt2d.set_data([self.r2], [self.diss_now])
        if funds:
            parts = [f'f₁ = {funds[0]:.1f} Hz']
            if len(funds) >= 2:
                parts.append(f'f₂ = {funds[1]:.1f} Hz  ×{funds[1]/funds[0]:.3f}')
            self.status_txt.set_text('\n'.join(parts))
        else:
            self.status_txt.set_text('(no notes detected)')
        n_act = int((self.hw > 0.05).sum())
        ax.set_title(f'Dissonance Curve — 2 notes   [{n_act} harmonics]',
                     color='#ccccff', fontsize=11, pad=8)

    # ── contour update ────────────────────────────────────────────────────────

    def _upd_contour(self, funds):
        ax = self.ax_main
        self._pcm.set_array(self.surf_z_sm.ravel())
        self._pcm.set_clim(0, max(self.surf_z_sm.max(), 1e-6))
        if self._frame % CONTOUR_REDRAW_N == 0:
            if self._contour_set is not None:
                for coll in self._contour_set.collections:
                    try: coll.remove()
                    except: pass
            R2, R3 = np.meshgrid(self.sr2, self.sr3)
            self._contour_set = ax.contour(
                R2, R3, self.surf_z_sm, levels=CONTOUR_LEVELS,
                colors='white', alpha=0.25, linewidths=0.6, zorder=2)
        self.pt3_dot.set_data([self.r2], [self.r3])
        if funds:
            parts = [f'f₁={funds[0]:.1f}']
            if len(funds) >= 2: parts.append(f'f₂={funds[1]:.1f} r₂={self.r2:.3f}')
            if len(funds) >= 3: parts.append(f'f₃={funds[2]:.1f} r₃={self.r3:.3f}')
            self.status_txt.set_text('\n'.join(parts))
        else:
            self.status_txt.set_text('(no notes detected)')
        n_act = int((self.hw > 0.05).sum())
        ax.set_title(f'Dissonance Map — 3 notes   [{n_act} harmonics]',
                     color='#ccccff', fontsize=11, pad=8)

    # ── 3D surface update ─────────────────────────────────────────────────────

    def _upd_3d(self):
        ax = self.ax_main
        if self._frame % 4 == 0:
            try:   self._az = ax.azim; self._el = ax.elev
            except: pass
            try:   self._surf3d.remove()
            except: ax.collections.clear()
            R2, R3 = np.meshgrid(self.sr2, self.sr3)
            self._surf3d = ax.plot_surface(
                R2, R3, self.surf_z_sm, cmap='plasma',
                alpha=0.8, linewidth=0, antialiased=True)
            ax.set_xlim(RATIO_MIN, RATIO_MAX); ax.set_ylim(RATIO_MIN, RATIO_MAX)
            ax.set_zlim(0, max(self.surf_z_sm.max()*1.1, 0.05))
            ax.view_init(elev=self._el, azim=self._az)
        self.pt3d.set_data([self.r2], [self.r3])
        self.pt3d.set_3d_properties([self.diss_now])

    # ── spectrum update (CQT log-spaced) ─────────────────────────────────────

    def _upd_spectrum(self, cqt_mag, funds):
        ax   = self.ax_spectrum
        freqs = self.cqt.freqs
        mag   = cqt_mag
        if mag.max() > 0: mag = mag / mag.max()

        self.spec_line.set_data(freqs, mag)
        self.spec_glow.set_data(freqs, mag)
        if self.spec_fill is not None: self.spec_fill.remove()
        self.spec_fill = ax.fill_between(freqs, 0, mag,
                                         alpha=0.22, color='#226688', zorder=3)
        for i, vl in enumerate(self.f_lines):
            if i < len(funds): vl.set_xdata([funds[i]]); vl.set_alpha(0.85)
            else:               vl.set_alpha(0.0)

        for a in self._harm_artists:
            try: a.remove()
            except: pass
        self._harm_artists.clear()
        for i, f0 in enumerate(funds):
            col = NOTE_COLS[i % len(NOTE_COLS)]
            for h in range(2, N_HARMONICS + 1):
                hf = f0 * h
                if hf <= CQT_FMAX:
                    ln = ax.axvline(hf, color=col, lw=0.8, alpha=0.35,
                                    linestyle='--', zorder=6)
                    self._harm_artists.append(ln)
        ax.set_ylim(0, 1.08)

    # ── stave update ──────────────────────────────────────────────────────────

    def _upd_stave(self, detected_funds):
        ax = self.ax_stave

        # ── Remove previous note artists ──────────────────────────────────────
        for p in self._stave_notes:
            try: p.remove()
            except: pass
        for t in self._stave_texts:
            try: t.remove()
            except: pass
        for l in self._stave_ledgers:
            try: l.remove()
            except: pass
        for s in self._stave_shadow:
            try: s.remove()
            except: pass
        self._stave_notes.clear()
        self._stave_texts.clear()
        self._stave_ledgers.clear()
        self._stave_shadow.clear()

        # ── Decide what to show ───────────────────────────────────────────────
        if self.source == 'synth':
            chord_name, note_info, note_freqs = self.synth.get_state()
        else:
            # Mic mode: show detected notes (best-effort)
            chord_name = ''
            note_freqs = detected_funds
            note_info  = [freq_to_staff(f) for f in note_freqs]

        self._chord_txt.set_text(chord_name)
        self._source_txt.set_text(
            '▶ Synthesiser  [SPACE = next chord]' if self.source == 'synth'
            else '🎤 Microphone')

        if not note_freqs:
            return

        # ── x-positions for note heads (spread if > 1 at same staff y) ───────
        xs      = np.linspace(3.5, 6.5, max(len(note_freqs), 1))
        used_ys = {}          # staff_y → list of x already placed

        for i, (nname, sy, is_acc) in enumerate(note_info):
            col  = NOTE_COLS[i % len(NOTE_COLS)]
            x    = float(xs[i])

            # If two notes share a staff position, nudge the second rightward
            key  = round(sy * 2)   # integer half-steps
            if key in used_ys:
                x += 0.55
            used_ys.setdefault(key, []).append(x)

            # Ledger lines:  drawn only for notes outside both staves
            treble_range = (1.0, 5.0)
            bass_range   = (-5.0, -1.0)
            in_treble    = treble_range[0] <= sy <= treble_range[1]
            in_bass      = bass_range[0]   <= sy <= bass_range[1]

            # Middle C ledger line
            if abs(sy) < 0.3:
                l = ax.plot([x-0.5, x+0.5], [0, 0],
                            color='#ccccdd', lw=0.9, zorder=3)[0]
                self._stave_ledgers.append(l)

            # Above treble staff
            if sy > 5.0:
                for ly in np.arange(6.0, sy + 0.5, 1.0):
                    if abs(ly - round(ly)) < 0.1:
                        l = ax.plot([x-0.5, x+0.5], [ly, ly],
                                    color='#ccccdd', lw=0.9, zorder=3)[0]
                        self._stave_ledgers.append(l)

            # Below bass staff
            if sy < -5.0:
                for ly in np.arange(-6.0, sy - 0.5, -1.0):
                    if abs(ly - round(ly)) < 0.1:
                        l = ax.plot([x-0.5, x+0.5], [ly, ly],
                                    color='#ccccdd', lw=0.9, zorder=3)[0]
                        self._stave_ledgers.append(l)

            # Between staves (but not middle C): just the note, no line needed
            if -1.0 < sy < 1.0 and abs(sy) > 0.3:
                pass   # gap between staves, no ledger needed

            # Glow / shadow circle for visual pop
            shadow = ax.add_patch(
                plt.Circle((x, sy), 0.44, color=col, alpha=0.25, zorder=4))
            self._stave_shadow.append(shadow)

            # Note head (filled ellipse approximated by Circle)
            note_patch = ax.add_patch(
                plt.Circle((x, sy), 0.35, color=col, zorder=5))
            self._stave_notes.append(note_patch)

            # Accidental symbol
            acc_txt = ''
            if '#' in nname:  acc_txt = '♯'
            elif 'b' in nname: acc_txt = '♭'
            if acc_txt:
                t = ax.text(x - 0.62, sy, acc_txt, color=col, fontsize=9,
                            ha='center', va='center', zorder=6)
                self._stave_texts.append(t)

            # Note name label
            display = nname[0] + ('' if not acc_txt else '')
            t = ax.text(x + 0.55, sy, nname, color=col, fontsize=7,
                        ha='left', va='center', zorder=6)
            self._stave_texts.append(t)

    # ── time-series update ────────────────────────────────────────────────────

    def _upd_time(self):
        ax = self.ax_time
        if len(self.time_hist) < 2: return
        ts = np.array(self.time_hist); ds = np.array(self.diss_hist, float)
        self.time_line.set_data(ts, ds)
        t_now = ts[-1]
        ax.set_xlim(max(0, t_now - HIST_SECONDS), max(HIST_SECONDS, t_now))
        finite = ds[np.isfinite(ds)]
        ax.set_ylim(0, max(finite.max() * 1.15, 0.05) if len(finite) else 1.0)
        if self.time_fill is not None: self.time_fill.remove()
        ds_f = np.where(np.isfinite(ds), ds, 0.0)
        self.time_fill = ax.fill_between(ts, 0, ds_f,
                                         alpha=0.20, color='#884400', zorder=3)

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        mic_ok = False
        if AUDIO_AVAILABLE:
            try:
                self.audio.start()
                print("Microphone active.")
                mic_ok = True
            except Exception as e:
                print(f"Microphone unavailable ({e}).")

        # Always start the synthesiser thread (it produces audio when active)
        self.synth.start(self.audio)

        if not mic_ok and not AUDIO_AVAILABLE:
            print("No audio I/O – forcing synthesiser mode.")
            self.source = 'synth'
            self.audio.accept_mic = False
            self.synth.set_active(True, self.n_notes)

        print("Controls: select 'Synthesiser' to hear and analyse generated chords.")

        self._key_cid = self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self.anim = FuncAnimation(
            self.fig, self.animate,
            interval=ANIM_INTERVAL_MS,
            blit=False, cache_frame_data=False,
        )
        plt.show()
        self.synth.stop()
        if self.audio.running:
            self.audio.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor('#0a0a18')
    ax.tick_params(colors='#9999bb', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#222244')


def _bilinear(xv, yv, Z, x, y):
    xi = int(np.clip(np.searchsorted(xv, x) - 1, 0, len(xv) - 2))
    yi = int(np.clip(np.searchsorted(yv, y) - 1, 0, len(yv) - 2))
    tx = (x - xv[xi]) / (xv[xi+1] - xv[xi])
    ty = (y - yv[yi]) / (yv[yi+1] - yv[yi])
    return float(Z[yi,   xi]   * (1-tx)*(1-ty) +
                 Z[yi,   xi+1] *    tx *(1-ty) +
                 Z[yi+1, xi]   * (1-tx)*   ty  +
                 Z[yi+1, xi+1] *    tx *   ty)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for backend in ('TkAgg', 'Qt5Agg', 'Qt6Agg', 'WXAgg', 'macosx'):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
    App().run()


if __name__ == '__main__':
    main()
