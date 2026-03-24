#!/usr/bin/env python3
"""
Dissonance Live Visualizer
==========================
Real-time visualisation of harmonic dissonance based on William Sethares' theory.
Captures microphone audio, detects fundamental pitches via iterative HPS,
measures harmonic amplitudes, and plots the live position on a Sethares
dissonance curve (2-note mode) or surface (3-note mode).

Usage:  python main.py
"""

import sys
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
    print("sounddevice not found – running in demo mode.")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 44100
DEFAULT_WIN_MS    = 500        # FFT window width (ms)
MIN_WIN_MS        = 100
MAX_WIN_MS        = 2000

N_HARMONICS       = 8          # harmonics per note used in dissonance calc
HPS_DEPTH         = 5          # HPS product depth for fundamental detection
MIN_FREQ          = 60.0       # lowest detectable fundamental (Hz)
MAX_FREQ          = 4200.0     # highest frequency displayed in spectrum

RATIO_MIN         = 1.0        # frequency ratio range for dissonance axis
RATIO_MAX         = 2.5
CURVE_POINTS      = 300        # resolution of 2D dissonance curve
SURF_POINTS       = 30         # grid points per axis for 3D surface

TRAJ_LENGTH       = 300        # trajectory history depth (frames)
HIST_SECONDS      = 20.0       # dissonance-vs-time history shown
ANIM_INTERVAL_MS  = 120        # animation frame period

SURF_ALPHA        = 0.10       # surface smoothing (0=frozen, 1=instant)
MIN_RMS           = 0.003      # minimum signal level before ignoring audio
HARM_MASK_FRAC    = 0.09       # fractional half-width of harmonic mask in HPS

# Colours for detected notes
NOTE_COLS = ['#FF5555', '#55DD55', '#5599FF', '#FFAA00']

# Reference musical note frequencies (Hz) for spectrum x-axis labels
LABEL_NOTES = {
    'C2': 65.4, 'A2': 110.0, 'E3': 164.8, 'A3': 220.0,
    'C4': 261.6, 'E4': 329.6, 'G4': 392.0, 'A4': 440.0,
    'C5': 523.3, 'A5': 880.0, 'C6': 1046.5,
}

# ─────────────────────────────────────────────────────────────────────────────
#  SETHARES DISSONANCE MODEL
#  Based on: W. A. Sethares, "Tuning, Timbre, Spectrum, Scale", 2nd ed.
#  Python reference: https://gist.github.com/endolith/3066664
# ─────────────────────────────────────────────────────────────────────────────
_B1    = 3.5
_B2    = 5.75
_S1    = 0.0207
_S2    = 18.96
_DSTAR = 0.24


def _rough(fi, fj, ai, aj):
    """Vectorised roughness between two (arrays of) pure-tone pairs."""
    f_lo = np.minimum(fi, fj)
    f_hi = np.maximum(fi, fj)
    diff = f_hi - f_lo
    s    = _DSTAR / (_S1 * f_lo + _S2)
    r    = ai * aj * (np.exp(-_B1 * s * diff) - np.exp(-_B2 * s * diff))
    return np.maximum(r, 0.0)


def _total_rough(freqs, amps):
    """Total Sethares dissonance for a list of partials."""
    freqs = np.asarray(freqs, float)
    amps  = np.asarray(amps,  float)
    if len(freqs) < 2:
        return 0.0
    if amps.max() > 0:
        amps = amps / amps.max()
    total = 0.0
    n = len(freqs)
    for i in range(n - 1):
        total += _rough(freqs[i], freqs[i+1:], amps[i], amps[i+1:]).sum()
    return float(total)


def dissonance_curve(f0, hw, ratios):
    """
    2D dissonance curve: two-note chord where note 2 is at f0*ratio.
    hw: harmonic weights array (length = n_harmonics).
    Returns array of dissonance values, same length as ratios.
    """
    n_h = len(hw)
    hw  = hw / hw.max() if hw.max() > 0 else hw
    kv  = np.arange(1, n_h + 1, dtype=float)

    # Note 1 partials (fixed)
    f1v = f0 * kv                  # (n_h,)

    # Within-note-1 contribution (constant)
    d_n1 = 0.0
    for i in range(n_h - 1):
        d_n1 += _rough(f1v[i], f1v[i+1:], hw[i], hw[i+1:]).sum()

    n_r = len(ratios)
    total = np.full(n_r, d_n1)

    # Note 2 partials: shape (n_r, n_h)
    f2m = np.outer(ratios, f0 * kv)

    # Within-note-2 contribution (varies with ratio)
    for i in range(n_h - 1):
        fi = f2m[:, i]               # (n_r,)
        fj = f2m[:, i+1:]            # (n_r, n_h-i-1)
        total += _rough(fi[:, None], fj, hw[i], hw[i+1:]).sum(axis=1)

    # Cross-note contribution
    for i in range(n_h):
        for j in range(n_h):
            total += _rough(f1v[i], f2m[:, j], hw[i], hw[j])

    return total


def dissonance_surface(f0, hw, r2v, r3v):
    """
    3D dissonance surface: three-note chord.
    r2v, r3v: 1-D ratio arrays.
    Returns Z matrix of shape (len(r3v), len(r2v)).
    """
    n_h = len(hw)
    hw  = hw / hw.max() if hw.max() > 0 else hw
    kv  = np.arange(1, n_h + 1, dtype=float)

    f1v = f0 * kv                          # (n_h,)
    f2m = np.outer(r2v, f0 * kv)          # (n_r2, n_h)
    f3m = np.outer(r3v, f0 * kv)          # (n_r3, n_h)

    n_r2, n_r3 = len(r2v), len(r3v)
    Z = np.zeros((n_r3, n_r2))

    # Within note 1 (constant)
    d1 = 0.0
    for i in range(n_h - 1):
        d1 += _rough(f1v[i], f1v[i+1:], hw[i], hw[i+1:]).sum()
    Z += d1

    # Within note 2 (varies with r2 only)
    d2 = np.zeros(n_r2)
    for i in range(n_h - 1):
        fi = f2m[:, i]
        fj = f2m[:, i+1:]
        d2 += _rough(fi[:, None], fj, hw[i], hw[i+1:]).sum(axis=1)
    Z += d2[np.newaxis, :]

    # Within note 3 (varies with r3 only)
    d3 = np.zeros(n_r3)
    for i in range(n_h - 1):
        fi = f3m[:, i]
        fj = f3m[:, i+1:]
        d3 += _rough(fi[:, None], fj, hw[i], hw[i+1:]).sum(axis=1)
    Z += d3[:, np.newaxis]

    # Cross note1 vs note2 (varies with r2)
    cross12 = np.zeros(n_r2)
    for i in range(n_h):
        for j in range(n_h):
            cross12 += _rough(f1v[i], f2m[:, j], hw[i], hw[j])
    Z += cross12[np.newaxis, :]

    # Cross note1 vs note3 (varies with r3)
    cross13 = np.zeros(n_r3)
    for i in range(n_h):
        for j in range(n_h):
            cross13 += _rough(f1v[i], f3m[:, j], hw[i], hw[j])
    Z += cross13[:, np.newaxis]

    # Cross note2 vs note3 (varies with both r2 and r3) — vectorised outer
    for i in range(n_h):
        for j in range(n_h):
            fi = f2m[:, i][np.newaxis, :]    # (1,  n_r2)
            fj = f3m[:, j][:, np.newaxis]    # (n_r3, 1)
            Z += _rough(fi, fj, hw[i], hw[j])

    return Z


# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class AudioProcessor:
    """Captures microphone audio and extracts note fundamentals + harmonics."""

    def __init__(self, sr=SAMPLE_RATE):
        self.sr           = sr
        self.win_samples  = int(DEFAULT_WIN_MS * sr / 1000)
        buf_len           = int(MAX_WIN_MS * sr / 1000) * 3
        self._buf         = np.zeros(buf_len)
        self._buf_lock    = threading.Lock()
        self._wpos        = 0
        self._stream      = None
        self.running      = False

        # Published results (protected by result_lock)
        self.result_lock  = threading.Lock()
        self.fundamentals : list  = []   # Hz, sorted ascending
        self.harm_amps    : list  = []   # per-note harmonic amp arrays
        self.avg_harmonics = np.ones(N_HARMONICS) / N_HARMONICS
        self.rms          = 0.0
        self.fft_freqs    = np.array([1.0])
        self.fft_mag      = np.array([0.0])

    # ── audio capture ────────────────────────────────────────────────────────

    def _callback(self, indata, frames, time_info, status):
        data = indata[:, 0]
        with self._buf_lock:
            n   = len(data)
            end = self._wpos + n
            L   = len(self._buf)
            if end <= L:
                self._buf[self._wpos:end] = data
            else:
                f = L - self._wpos
                self._buf[self._wpos:] = data[:f]
                self._buf[:end - L]    = data[f:]
            self._wpos = end % L

    def _get_window(self):
        n = self.win_samples
        with self._buf_lock:
            pos = self._wpos
            buf = self._buf.copy()
        L = len(buf)
        s = (pos - n) % L
        if s + n <= L:
            return buf[s:s+n]
        return np.concatenate([buf[s:], buf[:n-(L-s)]])

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sr, channels=1,
            callback=self._callback, blocksize=1024)
        self._stream.start()
        self.running = True

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self.running = False

    def inject(self, audio_chunk):
        """Inject synthetic audio (demo mode)."""
        with self._buf_lock:
            n   = len(audio_chunk)
            end = self._wpos + n
            L   = len(self._buf)
            if end <= L:
                self._buf[self._wpos:end] = audio_chunk
            else:
                f = L - self._wpos
                self._buf[self._wpos:] = audio_chunk[:f]
                self._buf[:end - L]    = audio_chunk[f:]
            self._wpos = end % L

    # ── DSP ──────────────────────────────────────────────────────────────────

    def _fft(self, win):
        w   = sig_windows.blackmanharris(len(win))
        mag = np.abs(np.fft.rfft(win * w)) / w.sum()
        frq = np.fft.rfftfreq(len(win), 1.0 / self.sr)
        return frq, mag

    def _hps(self, mag, depth=HPS_DEPTH):
        """Harmonic Product Spectrum."""
        h = mag.copy()
        for k in range(2, depth + 1):
            ds = mag[::k]
            ml = min(len(h), len(ds))
            h[:ml] *= ds[:ml]
            h[ml:] = 0
        return h

    def _find_f0(self, freqs, mag):
        """Find strongest fundamental in [MIN_FREQ, MAX_FREQ/2] via HPS."""
        mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ / 2)
        if not mask.any():
            return None, 0.0
        h = self._hps(mag)
        h_masked = np.where(mask, h, 0.0)
        idx = int(np.argmax(h_masked))
        strength = h_masked[idx]
        if strength < 1e-12:
            return None, 0.0
        # Parabolic interpolation for sub-bin accuracy
        if 0 < idx < len(h_masked) - 1:
            y0, y1, y2 = h_masked[idx-1], h_masked[idx], h_masked[idx+1]
            den = 2.0 * (2*y1 - y0 - y2)
            if den != 0:
                delta = (y2 - y0) / den
                idx_r = idx + delta
                f0 = np.interp(idx_r, np.arange(len(freqs)), freqs)
            else:
                f0 = freqs[idx]
        else:
            f0 = freqs[idx]
        return f0, strength

    def _harm_amps(self, freqs, mag, f0, n=N_HARMONICS):
        """Extract amplitude at each harmonic of f0 from the spectrum."""
        amps = np.zeros(n)
        for k in range(1, n + 1):
            target = f0 * k
            if target > freqs[-1]:
                break
            lo, hi = target * (1 - HARM_MASK_FRAC), target * (1 + HARM_MASK_FRAC)
            m = (freqs >= lo) & (freqs <= hi)
            if m.any():
                amps[k-1] = mag[m].max()
        return amps

    def _mask_harmonics(self, freqs, mag, f0, n=N_HARMONICS):
        """Zero out harmonics of f0 in the spectrum (for iterative detection)."""
        out = mag.copy()
        for k in range(1, n + 1):
            target = f0 * k
            if target > freqs[-1]:
                break
            lo, hi = target * (1 - HARM_MASK_FRAC), target * (1 + HARM_MASK_FRAC)
            out[(freqs >= lo) & (freqs <= hi)] = 0.0
        return out

    # ── main analysis (called each animation frame) ──────────────────────────

    def analyse(self, n_expected):
        """
        Analyse the current audio window.
        n_expected: how many note fundamentals to detect (2 or 3).
        """
        win = self._get_window()
        rms = float(np.sqrt(np.mean(win**2)))
        freqs, mag = self._fft(win)

        notes = []          # list of (f0, harm_amp_array)
        mag_w = mag.copy()
        for _ in range(n_expected):
            f0, strength = self._find_f0(freqs, mag_w)
            if f0 is None:
                break
            ha = self._harm_amps(freqs, mag, f0)
            if ha[0] < 1e-6:
                break
            notes.append((f0, ha))
            mag_w = self._mask_harmonics(freqs, mag_w, f0)

        # Sort by frequency
        notes.sort(key=lambda x: x[0])

        # Pad / trim to n_expected
        if notes:
            loudest = max(notes, key=lambda x: x[1][0])
            while len(notes) < n_expected:
                notes.append(loudest)
            if len(notes) > n_expected:
                notes.sort(key=lambda x: -x[1][0])
                notes = notes[:n_expected]
                notes.sort(key=lambda x: x[0])

        # Average harmonic envelope
        if notes:
            all_ha = np.array([n[1] for n in notes])
            avg    = all_ha.mean(axis=0)
            if avg.max() > 0:
                avg /= avg.max()
        else:
            avg = np.ones(N_HARMONICS) / N_HARMONICS

        with self.result_lock:
            self.fundamentals  = [n[0] for n in notes]
            self.harm_amps     = [n[1] for n in notes]
            self.avg_harmonics = avg
            self.rms           = rms
            self.fft_freqs     = freqs
            self.fft_mag       = mag

    def set_win_ms(self, ms):
        self.win_samples = int(ms * self.sr / 1000)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class App:
    """Main matplotlib application."""

    def __init__(self):
        self.audio   = AudioProcessor()
        self.n_notes = 2          # 2 or 3

        self.ratios  = np.linspace(RATIO_MIN, RATIO_MAX, CURVE_POINTS)
        self.sr2     = np.linspace(RATIO_MIN, RATIO_MAX, SURF_POINTS)
        self.sr3     = np.linspace(RATIO_MIN, RATIO_MAX, SURF_POINTS)

        # Dissonance data
        self.curve_z     = np.zeros(CURVE_POINTS)
        self.surf_z      = np.zeros((SURF_POINTS, SURF_POINTS))
        self.surf_z_sm   = np.zeros_like(self.surf_z)   # smoothed surface

        # Detected state
        self.r2          = 1.5      # current ratio of note2/note1
        self.r3          = 1.33     # current ratio of note3/note1
        self.diss_now    = 0.0

        # Default harmonic weights (1/k – typical string instrument decay)
        n = N_HARMONICS
        self.hw = np.array([1.0 / k for k in range(1, n + 1)])
        self.hw /= self.hw.max()
        self._n_harm_used = N_HARMONICS

        # Trajectory
        self.traj2 = deque(maxlen=TRAJ_LENGTH)   # (ratio2, diss)
        self.traj3 = deque(maxlen=TRAJ_LENGTH)   # (r2, r3, diss)

        # Time-series dissonance
        n_hist = int(HIST_SECONDS * 1000 / ANIM_INTERVAL_MS) + 10
        self.diss_hist = deque(maxlen=n_hist)
        self.time_hist = deque(maxlen=n_hist)
        self.t0        = time.time()

        # Surface dirty flag
        self._dirty       = True
        self._frame       = 0
        self._surf_redraw = 0    # frame counter for 3D full redraws

        # 3D view angles (preserved across redraws)
        self._az = -60.0
        self._el =  30.0

        self._build_figure()
        self._build_controls()
        self._initial_compute()

    # ── figure construction ──────────────────────────────────────────────────

    def _build_figure(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10), facecolor='#10101e')
        try:
            self.fig.canvas.manager.set_window_title('Dissonance Live Visualizer')
        except Exception:
            pass

        # Outer: top row + bottom strip
        outer = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[3, 1], hspace=0.4,
            left=0.06, right=0.97, top=0.94, bottom=0.18,
        )
        # Top row: main plot + spectrum
        top = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0],
            width_ratios=[3, 2], wspace=0.32,
        )
        self._gs_main = top[0]     # Keep spec so we can re-add axes

        self.ax_main     = self.fig.add_subplot(self._gs_main)
        self.ax_spectrum = self.fig.add_subplot(top[1])
        self.ax_time     = self.fig.add_subplot(outer[1])

        for ax in (self.ax_main, self.ax_spectrum, self.ax_time):
            ax.set_facecolor('#0a0a18')
            ax.tick_params(colors='#9999bb', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#222244')

        self._init_spectrum_ax()
        self._init_time_ax()

    def _style_ax(self, ax):
        ax.set_facecolor('#0a0a18')
        ax.tick_params(colors='#9999bb', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#222244')

    # ── 2D main plot ─────────────────────────────────────────────────────────

    def _init_2d_ax(self):
        ax = self.ax_main
        ax.set_title('Dissonance Curve — 2 notes', color='#ccccff', fontsize=11, pad=8)
        ax.set_xlabel('Frequency ratio  f₂ / f₁', color='#9999bb', fontsize=9)
        ax.set_ylabel('Dissonance', color='#9999bb', fontsize=9)
        ax.grid(True, alpha=0.18, color='#222244')

        self.line2d,  = ax.plot(self.ratios, self.curve_z,
                                color='#6699ff', lw=1.8, alpha=0.9, zorder=4)
        self.fill2d   = ax.fill_between(self.ratios, 0, self.curve_z,
                                        alpha=0.18, color='#4466cc', zorder=3)
        self.pt2d,    = ax.plot([], [], 'o', color='#ffcc00',
                                ms=11, zorder=8, label='Now')
        self.traj2d,  = ax.plot([], [], '-', color='#ff6633',
                                lw=1.5, alpha=0.75, zorder=7)
        ax.set_xlim(RATIO_MIN, RATIO_MAX)

        # Mark simple ratios
        simple = {
            'unison\n1:1': 1.0, 'oct\n2:1': 2.0, '5th\n3:2': 1.5,
            '4th\n4:3': 4/3, 'M3\n5:4': 1.25, 'm3\n6:5': 1.2,
            'M6\n5:3': 5/3,
        }
        for label, r in simple.items():
            if RATIO_MIN <= r <= RATIO_MAX:
                ax.axvline(r, color='#334466', lw=0.8, alpha=0.6)
                ax.text(r, 0, label, color='#445566', fontsize=6,
                        ha='center', va='bottom', rotation=0)

    # ── 3D main plot ─────────────────────────────────────────────────────────

    def _init_3d_ax(self):
        ax = self.ax_main
        ax.set_title('Dissonance Surface — 3 notes', color='#ccccff', fontsize=11, pad=14)
        ax.set_xlabel('r₂ = f₂/f₁', color='#9999bb', fontsize=8, labelpad=4)
        ax.set_ylabel('r₃ = f₃/f₁', color='#9999bb', fontsize=8, labelpad=4)
        ax.set_zlabel('Dissonance', color='#9999bb', fontsize=8, labelpad=4)
        ax.tick_params(colors='#9999bb', labelsize=7)
        ax.set_facecolor('#0a0a18')
        ax.grid(True, alpha=0.12)

        R2, R3 = np.meshgrid(self.sr2, self.sr3)
        self._surf3d = ax.plot_surface(
            R2, R3, self.surf_z_sm,
            cmap='plasma', alpha=0.82, linewidth=0, antialiased=True,
        )
        self.traj3d,  = ax.plot([], [], [], '-',
                                color='#ffcc00', lw=2.2, alpha=0.95, zorder=10)
        self.pt3d,    = ax.plot([], [], [], 'o',
                                color='#ff4400', ms=8, zorder=11)
        ax.set_xlim(RATIO_MIN, RATIO_MAX)
        ax.set_ylim(RATIO_MIN, RATIO_MAX)
        ax.view_init(elev=self._el, azim=self._az)

    # ── spectrum axis ─────────────────────────────────────────────────────────

    def _init_spectrum_ax(self):
        ax = self.ax_spectrum
        ax.set_title('Frequency Spectrum', color='#ccccff', fontsize=11, pad=8)
        ax.set_xlabel('Frequency (Hz)', color='#9999bb', fontsize=9)
        ax.set_ylabel('Amplitude (norm.)', color='#9999bb', fontsize=9)
        ax.set_xscale('log')
        ax.set_xlim(MIN_FREQ * 0.85, MAX_FREQ)
        ax.set_ylim(0, 1.08)
        ax.grid(True, which='both', alpha=0.12, color='#222244')

        # Spectrum line & fill (to be populated in update)
        self.spec_line, = ax.plot([], [], color='#44ccff', lw=1.0, alpha=0.9, zorder=5)
        self.spec_fill  = None
        # Glow overlay
        self.spec_glow, = ax.plot([], [], color='#44ccff', lw=4.0, alpha=0.18, zorder=4)

        # Fundamental markers (one per note)
        self.f_lines = []
        for i in range(3):
            vl = ax.axvline(MIN_FREQ, color=NOTE_COLS[i],
                            lw=2.2, alpha=0.0, zorder=8)
            self.f_lines.append(vl)

        # Harmonic tick marks (redrawn each frame)
        self._harm_artists = []

        # Musical note labels on x-axis
        for name, freq in LABEL_NOTES.items():
            if MIN_FREQ * 0.85 <= freq <= MAX_FREQ:
                ax.axvline(freq, color='#1e1e3a', lw=0.7)
                ax.text(freq, -0.07, name, color='#445577', fontsize=6,
                        ha='center', va='top',
                        transform=ax.get_xaxis_transform())

    # ── time axis ────────────────────────────────────────────────────────────

    def _init_time_ax(self):
        ax = self.ax_time
        ax.set_title('Dissonance vs Time', color='#ccccff', fontsize=11, pad=4)
        ax.set_xlabel('Time (s)', color='#9999bb', fontsize=9)
        ax.set_ylabel('Dissonance', color='#9999bb', fontsize=9)
        ax.grid(True, alpha=0.18, color='#222244')
        self.time_line,  = ax.plot([], [], color='#ffaa44', lw=1.6, alpha=0.9)
        self.time_fill   = None
        ax.set_xlim(0, HIST_SECONDS)
        ax.set_ylim(0, 1.0)

    # ── controls ─────────────────────────────────────────────────────────────

    def _build_controls(self):
        # Window size slider
        ax_win = plt.axes([0.07, 0.08, 0.22, 0.025])
        self.sl_win = Slider(ax_win, 'FFT window (ms)',
                             MIN_WIN_MS, MAX_WIN_MS,
                             valinit=DEFAULT_WIN_MS, valstep=50,
                             color='#334466', track_color='#1a1a33')
        _style_slider(self.sl_win)

        # Max harmonics slider
        ax_nh = plt.axes([0.37, 0.08, 0.18, 0.025])
        self.sl_harm = Slider(ax_nh, 'Harmonics', 2, 16,
                              valinit=N_HARMONICS, valstep=1,
                              color='#334466', track_color='#1a1a33')
        _style_slider(self.sl_harm)

        # Mode radio
        ax_mode = plt.axes([0.62, 0.05, 0.14, 0.08])
        ax_mode.set_facecolor('#10101e')
        self.radio = RadioButtons(
            ax_mode, ('2 notes  (2D)', '3 notes  (3D)'),
            active=0, activecolor='#6699ff',
        )
        for lbl in self.radio.labels:
            lbl.set_color('#9999bb')
            lbl.set_fontsize(8)

        # Clear trajectory button
        ax_clr = plt.axes([0.79, 0.07, 0.09, 0.04])
        self.btn_clr = Button(ax_clr, 'Clear path',
                              color='#1a1a33', hovercolor='#2a2a55')
        self.btn_clr.label.set_color('#9999bb')
        self.btn_clr.label.set_fontsize(8)

        # Wire up callbacks
        self.sl_win.on_changed(self._on_win)
        self.sl_harm.on_changed(self._on_harm)
        self.radio.on_clicked(self._on_mode)
        self.btn_clr.on_clicked(lambda _: (self.traj2.clear(), self.traj3.clear()))

    def _on_win(self, v):
        self.audio.set_win_ms(int(v))
        self._dirty = True

    def _on_harm(self, v):
        self._n_harm_used = int(v)
        self._dirty = True

    def _on_mode(self, label):
        new_n = 2 if '2' in label else 3
        if new_n == self.n_notes:
            return
        self.n_notes = new_n
        self._switch_main_ax()
        self._dirty = True

    # ── switching 2D ↔ 3D ───────────────────────────────────────────────────

    def _switch_main_ax(self):
        self.ax_main.remove()
        if self.n_notes == 3:
            self.ax_main = self.fig.add_subplot(self._gs_main, projection='3d')
            self._style_ax(self.ax_main)
            self._init_3d_ax()
        else:
            self.ax_main = self.fig.add_subplot(self._gs_main)
            self._style_ax(self.ax_main)
            self._init_2d_ax()
        self.traj2.clear()
        self.traj3.clear()

    # ── initial compute ───────────────────────────────────────────────────────

    def _initial_compute(self):
        hw = self.hw[:self._n_harm_used]
        self.curve_z   = dissonance_curve(220.0, hw, self.ratios)
        if self.curve_z.max() > 0:
            self.curve_z /= self.curve_z.max()
        self.surf_z    = dissonance_surface(220.0, hw, self.sr2, self.sr3)
        if self.surf_z.max() > 0:
            self.surf_z /= self.surf_z.max()
        self.surf_z_sm = self.surf_z.copy()
        self._dirty    = False
        self._init_2d_ax()    # default to 2D

    def _recompute(self, hw):
        """Recompute curve and surface with given harmonic weights."""
        hw = hw / hw.max() if hw.max() > 0 else hw
        self.curve_z = dissonance_curve(220.0, hw, self.ratios)
        if self.curve_z.max() > 0:
            self.curve_z /= self.curve_z.max()
        new_surf = dissonance_surface(220.0, hw, self.sr2, self.sr3)
        if new_surf.max() > 0:
            new_surf /= new_surf.max()
        self.surf_z = new_surf

    # ── animation ─────────────────────────────────────────────────────────────

    def animate(self, frame):
        self._frame += 1

        # ── 1. audio analysis
        self.audio.analyse(self.n_notes)

        with self.audio.result_lock:
            funds     = list(self.audio.fundamentals)
            avg_hw    = self.audio.avg_harmonics.copy()
            rms       = self.audio.rms
            fft_freqs = self.audio.fft_freqs.copy()
            fft_mag   = self.audio.fft_mag.copy()

        # ── 2. update harmonic weights (smooth)
        if rms > MIN_RMS and len(avg_hw) > 0:
            n = min(len(avg_hw), self._n_harm_used, len(self.hw))
            alpha = 0.25
            self.hw[:n] = alpha * avg_hw[:n] + (1 - alpha) * self.hw[:n]
            self._dirty = True

        # ── 3. recompute dissonance (when dirty)
        if self._dirty:
            hw = self.hw[:self._n_harm_used]
            self._recompute(hw)
            self._dirty = False

        # ── 4. smooth surface towards new surface
        self.surf_z_sm = SURF_ALPHA * self.surf_z + (1 - SURF_ALPHA) * self.surf_z_sm

        # ── 5. determine current ratios from detected notes
        if len(funds) >= 2:
            self.r2 = np.clip(funds[1] / funds[0], RATIO_MIN, RATIO_MAX)
        if len(funds) >= 3:
            self.r3 = np.clip(funds[2] / funds[0], RATIO_MIN, RATIO_MAX)

        # ── 6. compute current dissonance by interpolation
        if self.n_notes == 2:
            self.diss_now = float(np.interp(self.r2, self.ratios, self.curve_z))
        else:
            self.diss_now = _bilinear(self.sr2, self.sr3, self.surf_z_sm,
                                      self.r2, self.r3)

        # ── 7. record history
        t_now = time.time() - self.t0
        self.time_hist.append(t_now)
        self.diss_hist.append(self.diss_now)

        if rms > MIN_RMS:
            self.traj2.append((self.r2, self.diss_now))
            self.traj3.append((self.r2, self.r3, self.diss_now))

        # ── 8. update plots
        if self.n_notes == 2:
            self._upd_2d()
        else:
            self._upd_3d()
        self._upd_spectrum(fft_freqs, fft_mag, funds)
        self._upd_time()

        return []

    # ── 2D plot update ────────────────────────────────────────────────────────

    def _upd_2d(self):
        ax = self.ax_main
        # Curve
        self.line2d.set_ydata(self.curve_z)
        if self.fill2d is not None:
            self.fill2d.remove()
        self.fill2d = ax.fill_between(self.ratios, 0, self.curve_z,
                                      alpha=0.15, color='#4466cc', zorder=3)
        z_top = max(self.curve_z.max() * 1.12, 0.05)
        ax.set_ylim(0, z_top)
        # Marker
        self.pt2d.set_data([self.r2], [self.diss_now])
        # Trajectory
        if len(self.traj2) > 1:
            t = np.array(self.traj2)
            self.traj2d.set_data(t[:, 0], t[:, 1])
        else:
            self.traj2d.set_data([], [])

    # ── 3D plot update ────────────────────────────────────────────────────────

    def _upd_3d(self):
        ax = self.ax_main
        # Only redraw surface every few frames (slow operation)
        if self._frame % 4 == 0:
            try:
                self._az = ax.azim
                self._el = ax.elev
            except Exception:
                pass
            try:
                self._surf3d.remove()
            except Exception:
                ax.collections.clear()
            R2, R3 = np.meshgrid(self.sr2, self.sr3)
            self._surf3d = ax.plot_surface(
                R2, R3, self.surf_z_sm,
                cmap='plasma', alpha=0.8, linewidth=0, antialiased=True,
            )
            ax.set_xlim(RATIO_MIN, RATIO_MAX)
            ax.set_ylim(RATIO_MIN, RATIO_MAX)
            z_top = max(self.surf_z_sm.max() * 1.1, 0.05)
            ax.set_zlim(0, z_top)
            ax.view_init(elev=self._el, azim=self._az)

        # Trajectory
        if len(self.traj3) > 1:
            t = np.array(self.traj3)
            self.traj3d.set_data(t[:, 0], t[:, 1])
            self.traj3d.set_3d_properties(t[:, 2])
            self.pt3d.set_data([self.r2], [self.r3])
            self.pt3d.set_3d_properties([self.diss_now])
        else:
            self.traj3d.set_data([], [])
            self.traj3d.set_3d_properties([])

    # ── spectrum update ───────────────────────────────────────────────────────

    def _upd_spectrum(self, freqs, mag, funds):
        ax = self.ax_spectrum
        if len(freqs) < 2 or len(mag) < 2:
            return

        mask = (freqs >= MIN_FREQ * 0.85) & (freqs <= MAX_FREQ)
        df   = freqs[mask]
        dm   = mag[mask]
        dm_n = dm / dm.max() if dm.max() > 0 else dm

        self.spec_line.set_data(df, dm_n)
        self.spec_glow.set_data(df, dm_n)

        # Refill
        if self.spec_fill is not None:
            self.spec_fill.remove()
        self.spec_fill = ax.fill_between(df, 0, dm_n,
                                         alpha=0.22, color='#226688', zorder=3)

        # Update fundamental marker lines
        for i, vl in enumerate(self.f_lines):
            if i < len(funds):
                vl.set_xdata([funds[i]])
                vl.set_alpha(0.85)
            else:
                vl.set_alpha(0.0)

        # Redraw harmonic ticks
        for a in self._harm_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._harm_artists.clear()
        for i, f0 in enumerate(funds):
            col = NOTE_COLS[i % len(NOTE_COLS)]
            for k in range(2, self._n_harm_used + 1):
                hf = f0 * k
                if hf <= MAX_FREQ:
                    ln = ax.axvline(hf, color=col, lw=0.8, alpha=0.3,
                                    linestyle='--', zorder=6)
                    self._harm_artists.append(ln)

        ax.set_ylim(0, 1.08)

    # ── time-series update ────────────────────────────────────────────────────

    def _upd_time(self):
        ax = self.ax_time
        if len(self.time_hist) < 2:
            return
        ts = np.array(self.time_hist)
        ds = np.array(self.diss_hist)
        self.time_line.set_data(ts, ds)
        t_now = ts[-1]
        ax.set_xlim(max(0, t_now - HIST_SECONDS), max(HIST_SECONDS, t_now))
        ax.set_ylim(0, max(ds.max() * 1.1, 0.05))
        if self.time_fill is not None:
            self.time_fill.remove()
        self.time_fill = ax.fill_between(ts, 0, ds,
                                         alpha=0.2, color='#884400', zorder=3)

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        if AUDIO_AVAILABLE:
            try:
                self.audio.start()
                print("Microphone active. Play your instrument!")
            except Exception as e:
                print(f"Microphone unavailable ({e}). Starting demo mode.")
                self._demo_thread_start()
        else:
            print("sounddevice not installed. Starting demo mode.")
            self._demo_thread_start()

        self.anim = FuncAnimation(
            self.fig, self.animate,
            interval=ANIM_INTERVAL_MS,
            blit=False, cache_frame_data=False,
        )
        plt.show()
        if self.audio.running:
            self.audio.stop()

    def _demo_thread_start(self):
        """Synthesise a slowly shifting chord for testing without a mic."""
        def _run():
            sr = self.audio.sr
            chunk = 1024
            phase = np.zeros(6)
            t = 0
            while True:
                dt   = chunk / sr
                slow = time.time() * 0.07
                f1   = 220.0
                f2   = f1 * (1.5 + 0.3 * np.sin(slow))
                f3   = f1 * (1.25 + 0.2 * np.cos(slow * 0.7))
                base = np.zeros(chunk)
                for ki, (ff, amp) in enumerate(
                    [(f1, 0.6), (f2, 0.5), (f3, 0.4)]
                ):
                    for h in range(1, 6):
                        p0 = phase[ki * 2]
                        ph = np.linspace(p0, p0 + 2*np.pi*ff*h*dt, chunk, endpoint=False)
                        base += (amp / h) * np.sin(ph)
                    phase[ki * 2] = (phase[ki * 2] + 2*np.pi*ff*dt) % (2*np.pi)
                base *= 0.18
                self.audio.inject(base)
                time.sleep(dt * 0.5)
        threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bilinear(xv, yv, Z, x, y):
    """Bilinear interpolation of Z on grid xv × yv at point (x, y)."""
    xi = np.searchsorted(xv, x) - 1
    yi = np.searchsorted(yv, y) - 1
    xi = int(np.clip(xi, 0, len(xv) - 2))
    yi = int(np.clip(yi, 0, len(yv) - 2))
    tx = (x - xv[xi]) / (xv[xi+1] - xv[xi])
    ty = (y - yv[yi]) / (yv[yi+1] - yv[yi])
    return float(
        Z[yi,   xi]   * (1-tx)*(1-ty) +
        Z[yi,   xi+1] *    tx *(1-ty) +
        Z[yi+1, xi]   * (1-tx)*   ty  +
        Z[yi+1, xi+1] *    tx *   ty
    )


def _style_slider(sl):
    sl.label.set_color('#9999bb')
    sl.valtext.set_color('#ccddff')


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Choose a backend that supports interactive windows
    for backend in ('TkAgg', 'Qt5Agg', 'Qt6Agg', 'WXAgg', 'macosx'):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue

    app = App()
    app.run()


if __name__ == '__main__':
    main()
