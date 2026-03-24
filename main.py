#!/usr/bin/env python3
"""
Dissonance Live Visualizer
==========================
Real-time visualisation of harmonic dissonance based on William Sethares' theory.

Modes (selectable via radio buttons):
  2 notes          – 2D dissonance curve vs ratio
  3 notes contour  – 2D colour-contour map of the 3-note surface (default)
  3 notes 3D       – interactive 3D surface (slower)

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
    print("sounddevice not found – running in demo mode.")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 44100
DEFAULT_WIN_MS    = 500
MIN_WIN_MS        = 100
MAX_WIN_MS        = 2000

N_HARMONICS       = 8
HPS_DEPTH         = 5
MIN_FREQ          = 60.0
MAX_FREQ          = 4200.0
PEAK_REL_THRESH   = 0.02       # relative threshold for note detection

RATIO_MIN         = 1.0
RATIO_MAX         = 2.5
CURVE_POINTS      = 300        # resolution of 2-note curve
SURF_POINTS       = 40         # grid resolution per axis for 3-note surface
                               # (higher than before – contour rendering is cheap)

TRAJ_LENGTH       = 200
HIST_SECONDS      = 20.0
ANIM_INTERVAL_MS  = 120

HW_SMOOTH         = 0.20       # harmonic-weight IIR smoothing factor
HW_CHANGE_THRESH  = 0.005      # minimum Δ before recomputing dissonance
SURF_ALPHA        = 0.12       # surface animation speed (0=frozen, 1=instant)
MIN_RMS           = 0.002
HARM_MASK_FRAC    = 0.09

CONTOUR_LEVELS    = 12         # contour lines drawn on the map
CONTOUR_REDRAW_N  = 3          # redraw contour lines every N animation frames

NOTE_COLS = ['#FF5555', '#55DD55', '#5599FF', '#FFAA00']

LABEL_NOTES = {
    'C2': 65.4,  'A2': 110.0, 'E3': 164.8, 'A3': 220.0,
    'C4': 261.6, 'E4': 329.6, 'G4': 392.0, 'A4': 440.0,
    'C5': 523.3, 'A5': 880.0, 'C6': 1046.5,
}

# Simple musical ratios marked on both axes of the contour / 3D plots
SIMPLE_RATIOS = {
    '1:1': 1.0, '6:5': 6/5, '5:4': 5/4,
    '4:3': 4/3, '3:2': 3/2, '5:3': 5/3, '2:1': 2.0,
}

# ─────────────────────────────────────────────────────────────────────────────
#  SETHARES DISSONANCE MODEL
# ─────────────────────────────────────────────────────────────────────────────
_B1    = 3.5
_B2    = 5.75
_S1    = 0.0207
_S2    = 18.96
_DSTAR = 0.24


def _rough(fi, fj, ai, aj):
    f_lo = np.minimum(fi, fj)
    diff = np.maximum(fi, fj) - f_lo
    s    = _DSTAR / (_S1 * f_lo + _S2)
    return np.maximum(ai * aj * (np.exp(-_B1 * s * diff) - np.exp(-_B2 * s * diff)), 0.0)


def dissonance_curve(f0, hw, ratios):
    """2-note dissonance curve over an array of frequency ratios."""
    hw  = np.asarray(hw, float)
    if hw.max() > 0: hw = hw / hw.max()
    kv  = np.arange(1, len(hw) + 1, dtype=float)
    f1v = f0 * kv
    f2m = np.outer(ratios, f0 * kv)
    n_r = len(ratios)

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
    """3-note dissonance surface. Returns Z of shape (len(r3v), len(r2v))."""
    hw  = np.asarray(hw, float)
    if hw.max() > 0: hw = hw / hw.max()
    kv  = np.arange(1, len(hw) + 1, dtype=float)
    f1v = f0 * kv
    f2m = np.outer(r2v, f0 * kv)
    f3m = np.outer(r3v, f0 * kv)
    n_r2, n_r3 = len(r2v), len(r3v)
    Z = np.zeros((n_r3, n_r2))
    nh = len(hw)

    Z += sum(_rough(f1v[i], f1v[i+1:], hw[i], hw[i+1:]).sum() for i in range(nh-1))

    d2 = np.zeros(n_r2)
    for i in range(nh - 1):
        d2 += _rough(f2m[:, i][:, None], f2m[:, i+1:], hw[i], hw[i+1:]).sum(axis=1)
    Z += d2[np.newaxis, :]

    d3 = np.zeros(n_r3)
    for i in range(nh - 1):
        d3 += _rough(f3m[:, i][:, None], f3m[:, i+1:], hw[i], hw[i+1:]).sum(axis=1)
    Z += d3[:, np.newaxis]

    c12 = np.zeros(n_r2)
    for i in range(nh):
        for j in range(nh):
            c12 += _rough(f1v[i], f2m[:, j], hw[i], hw[j])
    Z += c12[np.newaxis, :]

    c13 = np.zeros(n_r3)
    for i in range(nh):
        for j in range(nh):
            c13 += _rough(f1v[i], f3m[:, j], hw[i], hw[j])
    Z += c13[:, np.newaxis]

    for i in range(nh):
        for j in range(nh):
            Z += _rough(f2m[:, i][np.newaxis, :], f3m[:, j][:, np.newaxis], hw[i], hw[j])

    return Z


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

        self.result_lock   = threading.Lock()
        self.fundamentals  : list = []
        self.harm_amps     : list = []
        self.avg_harmonics        = np.array([1.0/k for k in range(1, N_HARMONICS+1)])
        self.avg_harmonics       /= self.avg_harmonics.max()
        self.rms          = 0.0
        self.fft_freqs    = np.array([1.0])
        self.fft_mag      = np.array([0.0])

    def _callback(self, indata, frames, time_info, status):
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

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sr, channels=1,
            callback=self._callback, blocksize=1024)
        self._stream.start(); self.running = True

    def stop(self):
        if self._stream: self._stream.stop(); self._stream.close()
        self.running = False

    def inject(self, chunk):
        with self._buf_lock:
            n = len(chunk); end = self._wpos + n; L = len(self._buf)
            if end <= L:
                self._buf[self._wpos:end] = chunk
            else:
                f = L - self._wpos
                self._buf[self._wpos:] = chunk[:f]; self._buf[:end-L] = chunk[f:]
            self._wpos = end % L

    def _fft(self, win):
        w   = sig_windows.blackmanharris(len(win))
        mag = np.abs(np.fft.rfft(win * w)) / w.sum()
        frq = np.fft.rfftfreq(len(win), 1.0 / self.sr)
        return frq, mag

    def _hps(self, mag):
        h = mag.copy()
        for k in range(2, HPS_DEPTH + 1):
            ds = mag[::k]; ml = min(len(h), len(ds))
            h[:ml] *= ds[:ml]; h[ml:] = 0
        return h

    def _find_f0(self, freqs, mag):
        mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ / 2)
        if not mask.any(): return None, 0.0
        hm = np.where(mask, self._hps(mag), 0.0)
        peak = hm.max()
        if peak < 1e-15: return None, 0.0
        idx = int(np.argmax(hm))
        if hm[idx] < PEAK_REL_THRESH * peak: return None, 0.0
        if 0 < idx < len(hm) - 1:
            y0, y1, y2 = hm[idx-1], hm[idx], hm[idx+1]
            den = 2.0 * (2*y1 - y0 - y2)
            f0  = np.interp(idx + (y2 - y0) / den if den else idx,
                            np.arange(len(freqs)), freqs)
        else:
            f0 = freqs[idx]
        return f0, hm[idx]

    def _harm_amps(self, freqs, mag, f0):
        amps = np.zeros(N_HARMONICS)
        for k in range(1, N_HARMONICS + 1):
            target = f0 * k
            if target > freqs[-1]: break
            m = (freqs >= target*(1-HARM_MASK_FRAC)) & (freqs <= target*(1+HARM_MASK_FRAC))
            if m.any(): amps[k-1] = mag[m].max()
        return amps

    def _mask_harmonics(self, freqs, mag, f0):
        out = mag.copy()
        for k in range(1, N_HARMONICS + 1):
            target = f0 * k
            if target > freqs[-1]: break
            out[(freqs >= target*(1-HARM_MASK_FRAC)) & (freqs <= target*(1+HARM_MASK_FRAC))] = 0.0
        return out

    def analyse(self, n_expected):
        win  = self._get_window()
        rms  = float(np.sqrt(np.mean(win**2)))
        frqs, mag = self._fft(win)

        orig_peak = mag[(frqs >= MIN_FREQ) & (frqs <= MAX_FREQ)].max() \
                    if (frqs >= MIN_FREQ).any() else 1.0
        min_amp   = PEAK_REL_THRESH * orig_peak

        notes = []; mag_w = mag.copy()
        for _ in range(n_expected):
            f0, _ = self._find_f0(frqs, mag_w)
            if f0 is None: break
            ha = self._harm_amps(frqs, mag, f0)
            if ha[0] < min_amp: break
            notes.append((f0, ha))
            mag_w = self._mask_harmonics(frqs, mag_w, f0)

        notes.sort(key=lambda x: x[0])

        if notes:
            loudest = max(notes, key=lambda x: x[1][0])
            while len(notes) < n_expected: notes.append(loudest)
            if len(notes) > n_expected:
                notes.sort(key=lambda x: -x[1][0])
                notes = notes[:n_expected]; notes.sort(key=lambda x: x[0])

        if notes:
            avg = np.array([n[1] for n in notes]).mean(axis=0)
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

    def set_win_ms(self, ms):
        self.win_samples = int(ms * self.sr / 1000)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class App:
    """
    view_mode values:
        'curve'    – 2-note 2D dissonance curve
        'contour'  – 3-note 2D colour-contour map  (default for 3-note mode)
        'surface'  – 3-note 3D surface
    """

    def __init__(self):
        self.audio     = AudioProcessor()
        self.view_mode = 'curve'     # updated by radio; drives everything else
        self.n_notes   = 2

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

        self.traj2 = deque(maxlen=TRAJ_LENGTH)
        self.traj3 = deque(maxlen=TRAJ_LENGTH)

        n_hist = int(HIST_SECONDS * 1000 / ANIM_INTERVAL_MS) + 10
        self.diss_hist = deque(maxlen=n_hist)
        self.time_hist = deque(maxlen=n_hist)
        self.t0        = time.time()

        self._frame    = 0
        self._az       = -60.0
        self._el       =  30.0
        self._hw_prev  = self.hw.copy()

        # Contour plot artists (populated by _init_contour_ax)
        self._pcm          = None
        self._cbar         = None
        self._contour_set  = None

        self._build_figure()
        self._build_controls()
        self._recompute_curve()
        self._init_2d_ax()   # start in 2-note mode

    # ── figure ────────────────────────────────────────────────────────────────

    def _build_figure(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10), facecolor='#10101e')
        try:
            self.fig.canvas.manager.set_window_title('Dissonance Live Visualizer')
        except Exception:
            pass

        outer = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[3, 1], hspace=0.42,
            left=0.06, right=0.97, top=0.94, bottom=0.16,
        )
        top = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0],
            width_ratios=[3, 2], wspace=0.32,
        )
        self._gs_main    = top[0]
        self.ax_main     = self.fig.add_subplot(self._gs_main)
        self.ax_spectrum = self.fig.add_subplot(top[1])
        self.ax_time     = self.fig.add_subplot(outer[1])

        for ax in (self.ax_main, self.ax_spectrum, self.ax_time):
            _style_ax(ax)

        self._init_spectrum_ax()
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
        self.traj2d, = ax.plot([], [], '-', color='#ff6633', lw=1.8, alpha=0.80, zorder=7)

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
            fontfamily='monospace',
        )

    # ── 2D contour axes (3-note default) ─────────────────────────────────────

    def _init_contour_ax(self):
        ax = self.ax_main
        ax.set_facecolor('#0a0a18')
        ax.set_xlabel('r₂ = f₂ / f₁', color='#9999bb', fontsize=9)
        ax.set_ylabel('r₃ = f₃ / f₁', color='#9999bb', fontsize=9)
        ax.grid(False)   # grid confuses the contour; we use ratio lines instead

        R2, R3 = np.meshgrid(self.sr2, self.sr3)

        # Filled colour map — updated each frame via set_array (very fast)
        self._pcm = ax.pcolormesh(
            R2, R3, self.surf_z_sm,
            cmap='plasma', vmin=0, vmax=1,
            shading='gouraud',   # smooth colour interpolation between grid nodes
            zorder=1,
        )

        # Contour lines — redrawn every CONTOUR_REDRAW_N frames
        self._contour_set = ax.contour(
            R2, R3, self.surf_z_sm,
            levels=CONTOUR_LEVELS,
            colors='white', alpha=0.25, linewidths=0.6, zorder=2,
        )

        # Compact colorbar inside the axes area
        self._cbar = self.fig.colorbar(
            self._pcm, ax=ax,
            fraction=0.03, pad=0.02,
            label='Dissonance',
        )
        self._cbar.ax.yaxis.label.set_color('#9999bb')
        self._cbar.ax.tick_params(colors='#9999bb', labelsize=7)

        # Trajectory tail and position marker (plain 2D — easy and fast)
        self.traj3_line, = ax.plot([], [], '-', color='#00ffcc',
                                   lw=2.0, alpha=0.90, zorder=5)
        self.pt3_dot,    = ax.plot([], [], 'o', color='white', ms=11,
                                   zorder=6, markeredgecolor='#ffcc00',
                                   markeredgewidth=1.5)

        ax.set_xlim(RATIO_MIN, RATIO_MAX)
        ax.set_ylim(RATIO_MIN, RATIO_MAX)
        ax.set_aspect('equal')

        # Mark musically simple ratios on both axes
        for label, r in SIMPLE_RATIOS.items():
            if RATIO_MIN <= r <= RATIO_MAX:
                ax.axvline(r, color='white', lw=0.5, alpha=0.20, zorder=3)
                ax.axhline(r, color='white', lw=0.5, alpha=0.20, zorder=3)
                ax.text(r, RATIO_MIN + 0.01, label, color='#667799',
                        fontsize=5, ha='center', va='bottom', rotation=90, zorder=4)
                ax.text(RATIO_MIN + 0.01, r, label, color='#667799',
                        fontsize=5, ha='left', va='center', zorder=4)

        self.status_txt = ax.text(
            0.98, 0.97, '', transform=ax.transAxes,
            color='#aabbff', fontsize=8, ha='right', va='top',
            fontfamily='monospace', zorder=7,
        )

    # ── 3D surface axes ───────────────────────────────────────────────────────

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

    # ── spectrum axes ─────────────────────────────────────────────────────────

    def _init_spectrum_ax(self):
        ax = self.ax_spectrum
        ax.set_title('Frequency Spectrum', color='#ccccff', fontsize=11, pad=8)
        ax.set_xlabel('Frequency (Hz)', color='#9999bb', fontsize=9)
        ax.set_ylabel('Amplitude (norm.)', color='#9999bb', fontsize=9)
        ax.set_xscale('log')
        ax.set_xlim(MIN_FREQ * 0.85, MAX_FREQ)
        ax.set_ylim(0, 1.08)
        ax.grid(True, which='both', alpha=0.12, color='#222244')

        self.spec_line, = ax.plot([], [], color='#44ccff', lw=1.0, alpha=0.9, zorder=5)
        self.spec_glow, = ax.plot([], [], color='#44ccff', lw=4.0, alpha=0.15, zorder=4)
        self.spec_fill  = None

        self.f_lines = []
        for i in range(3):
            vl = ax.axvline(MIN_FREQ, color=NOTE_COLS[i], lw=2.2, alpha=0.0, zorder=8)
            self.f_lines.append(vl)
        self._harm_artists = []

        for name, freq in LABEL_NOTES.items():
            if MIN_FREQ * 0.85 <= freq <= MAX_FREQ:
                ax.axvline(freq, color='#1e1e3a', lw=0.7)
                ax.text(freq, -0.07, name, color='#445577', fontsize=6,
                        ha='center', va='top',
                        transform=ax.get_xaxis_transform())

    # ── time axes ─────────────────────────────────────────────────────────────

    def _init_time_ax(self):
        ax = self.ax_time
        ax.set_title('Dissonance vs Time', color='#ccccff', fontsize=11, pad=4)
        ax.set_xlabel('Time (s)', color='#9999bb', fontsize=9)
        ax.set_ylabel('Dissonance', color='#9999bb', fontsize=9)
        ax.grid(True, alpha=0.18, color='#222244')
        self.time_line, = ax.plot([], [], color='#ffaa44', lw=1.6, alpha=0.9)
        self.time_fill  = None
        ax.set_xlim(0, HIST_SECONDS)
        ax.set_ylim(0, 1.0)

    # ── controls ──────────────────────────────────────────────────────────────

    def _build_controls(self):
        # FFT window slider
        ax_win = plt.axes([0.07, 0.07, 0.25, 0.025])
        self.sl_win = Slider(ax_win, 'FFT window (ms)',
                             MIN_WIN_MS, MAX_WIN_MS,
                             valinit=DEFAULT_WIN_MS, valstep=50,
                             color='#334466', track_color='#1a1a33')
        self.sl_win.label.set_color('#9999bb')
        self.sl_win.valtext.set_color('#ccddff')

        # Three-way mode radio: 2-note curve | 3-note contour | 3-note 3D
        ax_mode = plt.axes([0.38, 0.03, 0.22, 0.10])
        ax_mode.set_facecolor('#10101e')
        self.radio = RadioButtons(
            ax_mode,
            ('2 notes  –  curve', '3 notes  –  contour', '3 notes  –  3D'),
            active=0, activecolor='#6699ff',
        )
        for lbl in self.radio.labels:
            lbl.set_color('#9999bb'); lbl.set_fontsize(8)

        # Clear path button
        ax_clr = plt.axes([0.68, 0.065, 0.10, 0.04])
        self.btn_clr = Button(ax_clr, 'Clear path',
                              color='#1a1a33', hovercolor='#2a2a55')
        self.btn_clr.label.set_color('#9999bb')
        self.btn_clr.label.set_fontsize(8)

        self.sl_win.on_changed(self._on_win)
        self.radio.on_clicked(self._on_radio)
        self.btn_clr.on_clicked(self._on_clear)

    def _on_win(self, v):
        self.audio.set_win_ms(int(v))

    def _on_radio(self, label):
        if 'curve' in label:
            new_mode, new_n = 'curve', 2
        elif 'contour' in label:
            new_mode, new_n = 'contour', 3
        else:
            new_mode, new_n = 'surface', 3

        if new_mode == self.view_mode:
            return
        self.view_mode = new_mode
        self.n_notes   = new_n
        self._switch_main_ax()

    def _on_clear(self, event):
        self.traj2.clear(); self.traj3.clear()
        for attr in ('traj2d', 'traj3_line', 'traj3d'):
            if hasattr(self, attr):
                a = getattr(self, attr)
                try:    a.set_data([], [])
                except: pass
                try:    a.set_3d_properties([])
                except: pass
        self.fig.canvas.draw_idle()

    # ── mode switch ───────────────────────────────────────────────────────────

    def _switch_main_ax(self):
        # Remove old colorbar before removing axes (avoids orphaned axes)
        if self._cbar is not None:
            try:    self._cbar.remove()
            except: pass
            self._cbar = None

        self.ax_main.remove()
        self.traj2.clear(); self.traj3.clear()

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

        else:  # 'surface'
            self.ax_main = self.fig.add_subplot(self._gs_main, projection='3d')
            _style_ax(self.ax_main)
            self._recompute_surface()
            self.surf_z_sm = self.surf_z.copy()
            self._init_3d_ax()

    # ── dissonance computation ────────────────────────────────────────────────

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

        # 1. Analyse audio
        self.audio.analyse(self.n_notes)
        with self.audio.result_lock:
            funds     = list(self.audio.fundamentals)
            avg_hw    = self.audio.avg_harmonics   # None if no notes
            rms       = self.audio.rms
            fft_freqs = self.audio.fft_freqs.copy()
            fft_mag   = self.audio.fft_mag.copy()

        notes_detected = len(funds) > 0

        # 2. Update harmonic weights (only when notes detected)
        hw_changed = False
        if notes_detected and avg_hw is not None:
            n   = min(len(avg_hw), len(self.hw))
            new = HW_SMOOTH * avg_hw[:n] + (1 - HW_SMOOTH) * self.hw[:n]
            if np.max(np.abs(new - self.hw[:n])) > HW_CHANGE_THRESH:
                self.hw[:n] = new
                hw_changed  = True

        # 3. Update reference pitch (smooth)
        if notes_detected:
            self.f0_ref = 0.85 * self.f0_ref + 0.15 * funds[0]

        # 4. Recompute dissonance data when timbre has changed
        if hw_changed:
            if self.view_mode == 'curve':
                self._recompute_curve()
            else:
                self._recompute_curve()       # still needed for diss_now lookup
                self._recompute_surface()

        # 5. Smooth surface towards target (drives contour animation)
        if self.view_mode in ('contour', 'surface'):
            self.surf_z_sm = (SURF_ALPHA * self.surf_z +
                              (1 - SURF_ALPHA) * self.surf_z_sm)

        # 6. Update detected ratios
        if len(funds) >= 2:
            self.r2 = float(np.clip(funds[1] / funds[0], RATIO_MIN, RATIO_MAX))
        if len(funds) >= 3:
            self.r3 = float(np.clip(funds[2] / funds[0], RATIO_MIN, RATIO_MAX))

        # 7. Current dissonance value
        if self.view_mode == 'curve':
            self.diss_now = float(np.interp(self.r2, self.ratios, self.curve_z))
        else:
            self.diss_now = _bilinear(self.sr2, self.sr3, self.surf_z_sm,
                                      self.r2, self.r3)

        # 8. Record history
        t_now = time.time() - self.t0
        self.time_hist.append(t_now)
        self.diss_hist.append(self.diss_now if notes_detected else float('nan'))

        if notes_detected and rms > MIN_RMS:
            self.traj2.append((self.r2, self.diss_now))
            if self.n_notes == 3:
                self.traj3.append((self.r2, self.r3, self.diss_now))

        # 9. Draw
        if self.view_mode == 'curve':
            self._upd_2d(funds)
        elif self.view_mode == 'contour':
            self._upd_contour(funds)
        else:
            self._upd_3d()

        self._upd_spectrum(fft_freqs, fft_mag, funds)
        self._upd_time()
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
        if len(self.traj2) > 1:
            t = np.array(self.traj2)
            self.traj2d.set_data(t[:, 0], t[:, 1])
        else:
            self.traj2d.set_data([], [])

        if funds:
            parts = [f'f₁ = {funds[0]:.1f} Hz']
            if len(funds) >= 2:
                parts.append(f'f₂ = {funds[1]:.1f} Hz  ×{funds[1]/funds[0]:.3f}')
            self.status_txt.set_text('\n'.join(parts))
        else:
            self.status_txt.set_text('(no notes detected)')

        n_active = int((self.hw > 0.05).sum())
        ax.set_title(f'Dissonance Curve — 2 notes   [{n_active} harmonics active]',
                     color='#ccccff', fontsize=11, pad=8)

    # ── contour update ────────────────────────────────────────────────────────

    def _upd_contour(self, funds):
        ax = self.ax_main

        # Fast path: update pcolormesh colours without redrawing geometry
        self._pcm.set_array(self.surf_z_sm.ravel())
        self._pcm.set_clim(0, max(self.surf_z_sm.max(), 1e-6))

        # Contour lines: redraw every CONTOUR_REDRAW_N frames
        if self._frame % CONTOUR_REDRAW_N == 0:
            if self._contour_set is not None:
                # Remove old contour collections
                for coll in self._contour_set.collections:
                    try: coll.remove()
                    except Exception: pass
            R2, R3 = np.meshgrid(self.sr2, self.sr3)
            self._contour_set = ax.contour(
                R2, R3, self.surf_z_sm,
                levels=CONTOUR_LEVELS,
                colors='white', alpha=0.25, linewidths=0.6, zorder=2,
            )

        # Trajectory tail
        if len(self.traj3) > 1:
            t = np.array(self.traj3)
            self.traj3_line.set_data(t[:, 0], t[:, 1])
        else:
            self.traj3_line.set_data([], [])

        # Position dot
        self.pt3_dot.set_data([self.r2], [self.r3])

        # Status text
        if funds:
            parts = [f'f₁ = {funds[0]:.1f} Hz']
            if len(funds) >= 2:
                parts.append(f'f₂ = {funds[1]:.1f} Hz  r₂={self.r2:.3f}')
            if len(funds) >= 3:
                parts.append(f'f₃ = {funds[2]:.1f} Hz  r₃={self.r3:.3f}')
            self.status_txt.set_text('\n'.join(parts))
        else:
            self.status_txt.set_text('(no notes detected)')

        n_active = int((self.hw > 0.05).sum())
        ax.set_title(f'Dissonance Map — 3 notes   [{n_active} harmonics active]',
                     color='#ccccff', fontsize=11, pad=8)

    # ── 3D surface update ─────────────────────────────────────────────────────

    def _upd_3d(self):
        ax = self.ax_main
        if self._frame % 4 == 0:
            try:
                self._az = ax.azim; self._el = ax.elev
            except Exception:
                pass
            try:    self._surf3d.remove()
            except: ax.collections.clear()
            R2, R3 = np.meshgrid(self.sr2, self.sr3)
            self._surf3d = ax.plot_surface(
                R2, R3, self.surf_z_sm,
                cmap='plasma', alpha=0.8, linewidth=0, antialiased=True,
            )
            ax.set_xlim(RATIO_MIN, RATIO_MAX)
            ax.set_ylim(RATIO_MIN, RATIO_MAX)
            ax.set_zlim(0, max(self.surf_z_sm.max() * 1.1, 0.05))
            ax.view_init(elev=self._el, azim=self._az)

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
        if len(freqs) < 2: return

        mask = (freqs >= MIN_FREQ * 0.85) & (freqs <= MAX_FREQ)
        df = freqs[mask]; dm = mag[mask]
        if dm.max() > 0: dm = dm / dm.max()

        self.spec_line.set_data(df, dm)
        self.spec_glow.set_data(df, dm)
        if self.spec_fill is not None: self.spec_fill.remove()
        self.spec_fill = ax.fill_between(df, 0, dm, alpha=0.22, color='#226688', zorder=3)

        for i, vl in enumerate(self.f_lines):
            if i < len(funds): vl.set_xdata([funds[i]]); vl.set_alpha(0.85)
            else:               vl.set_alpha(0.0)

        for a in self._harm_artists:
            try: a.remove()
            except: pass
        self._harm_artists.clear()
        for i, f0 in enumerate(funds):
            col = NOTE_COLS[i % len(NOTE_COLS)]
            for k in range(2, N_HARMONICS + 1):
                hf = f0 * k
                if hf <= MAX_FREQ:
                    ln = ax.axvline(hf, color=col, lw=0.8, alpha=0.35,
                                    linestyle='--', zorder=6)
                    self._harm_artists.append(ln)
        ax.set_ylim(0, 1.08)

    # ── time-series update ────────────────────────────────────────────────────

    def _upd_time(self):
        ax = self.ax_time
        if len(self.time_hist) < 2: return
        ts = np.array(self.time_hist)
        ds = np.array(self.diss_hist, dtype=float)
        self.time_line.set_data(ts, ds)
        t_now = ts[-1]
        ax.set_xlim(max(0, t_now - HIST_SECONDS), max(HIST_SECONDS, t_now))
        finite = ds[np.isfinite(ds)]
        ax.set_ylim(0, max(finite.max() * 1.15, 0.05) if len(finite) else 1.0)
        if self.time_fill is not None: self.time_fill.remove()
        ds_fill = np.where(np.isfinite(ds), ds, 0.0)
        self.time_fill = ax.fill_between(ts, 0, ds_fill,
                                         alpha=0.2, color='#884400', zorder=3)

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        if AUDIO_AVAILABLE:
            try:
                self.audio.start()
                print("Microphone active — play your instrument.")
            except Exception as e:
                print(f"Microphone unavailable ({e}). Starting demo mode.")
                self._start_demo()
        else:
            print("sounddevice not installed. Starting demo mode.")
            self._start_demo()

        self.anim = FuncAnimation(
            self.fig, self.animate,
            interval=ANIM_INTERVAL_MS,
            blit=False, cache_frame_data=False,
        )
        plt.show()
        if self.audio.running:
            self.audio.stop()

    def _start_demo(self):
        def _run():
            sr = self.audio.sr; chunk = 2048; t = 0.0
            while True:
                dt   = chunk / sr
                slow = time.time() * 0.05
                f1   = 220.0
                r2   = 1.0 + 0.75 * (0.5 + 0.5 * np.sin(slow))
                r3   = 1.0 + 0.75 * (0.5 + 0.5 * np.cos(slow * 0.7))
                sig  = np.zeros(chunk)
                tt   = np.arange(chunk) / sr + t
                for h in range(1, 7):
                    sig += (0.6/h) * np.sin(2*np.pi*f1*h*tt)
                    sig += (0.45/h) * np.sin(2*np.pi*f1*r2*h*tt)
                    sig += (0.35/h) * np.sin(2*np.pi*f1*r3*h*tt)
                sig *= 0.10
                self.audio.inject(sig); t += dt; time.sleep(dt * 0.5)
        threading.Thread(target=_run, daemon=True).start()


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
