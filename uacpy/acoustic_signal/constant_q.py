"""Constant-Q transform family (Brown 1991, *J. Acoust. Soc. Am.* 89, 425-434).

Unlike the STFT (constant absolute bandwidth) the constant-Q transform uses
geometrically spaced centre frequencies and a *constant* quality factor, so the
frequency resolution scales with frequency — fine at low frequency, coarse at
high frequency, mirroring octave/critical-band hearing.

For ``B`` bins per octave:

* centre frequencies ``f_k = fmin * 2**(k / B)``, ``k = 0..K-1``;
* constant quality factor ``Q = f_k / df_k = 1 / (2**(1/B) - 1)`` (df_k is the
  spacing to the next bin);
* per-bin window length ``N_k = ceil(Q * fs / f_k)`` samples — long at low
  frequency, short at high frequency;
* coefficient ``X_cq[k] = (1/sum w) * sum_n w_Nk[n] x[n] exp(-2j*pi*Q*n/N_k)``,
  where ``Q/N_k = f_k/fs`` so each bin is a windowed correlation at ``f_k``.

``constant_q_transform`` returns the complex spectrum of one (centred) frame;
``constant_q_spectrogram`` slides that over time; ``constant_q_psd`` averages it;
``probabilistic_constant_q`` histograms the per-bin dB levels over time, the
constant-Q analogue of :func:`uacpy.acoustic_signal.ppsd` (McNamara & Buland
2004, *BSSA* 94, 1517).

**Power scaling** (``scaling=``): ``'spectrum'`` (default) returns the one-sided
band power per constant-Q bin — a tone of amplitude ``A`` peaks at its
mean-square power ``A**2/2`` (matching :func:`scipy.signal.welch`
``scaling='spectrum'``). ``'density'`` further divides by the window's
noise-equivalent bandwidth to give a one-sided power *spectral density* (per
Hz), calibrated to match :func:`scipy.signal.welch` density and hence comparable
to Welch PSDs / Wenz curves. (The complex coefficients from
:func:`constant_q_transform` are the analytic band amplitude ``A/2``; the power
estimators double ``|X|**2`` to this one-sided convention.)

**Edge frames:** the averaging estimators (``constant_q_psd`` /
``probabilistic_constant_q``) reduce, *per bin*, only over frames whose window
lay fully inside the signal — the zero-padded edge frames (which bias low
frequencies down) are excluded. ``constant_q_spectrogram`` keeps every frame so
the full time axis (with visible edge effects) is shown.
"""
import warnings
from collections import namedtuple

import numpy as np
from scipy.signal import get_window

from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.acoustics import power_to_db
from uacpy.acoustic_signal._signal_validate import require_finite_signal

CQTResult = namedtuple("CQTResult", "frequencies coefficients")
CQPSDResult = namedtuple("CQPSDResult", "frequencies power")
CQSpectrogramResult = namedtuple("CQSpectrogramResult", "frequencies times power")
CQPPSDResult = namedtuple(
    "CQPPSDResult", "frequencies level_edges pdf mean_db std_db binwidth_db")

_SCALINGS = ("spectrum", "density")


# ── shared kernel construction ──────────────────────────────────────────────
def _cq_frequencies(fmin, fmax, bins_per_octave):
    if fmin <= 0 or fmax <= fmin:
        raise ConfigurationError(
            "constant-Q: require 0 < fmin < fmax; got "
            f"fmin={fmin}, fmax={fmax}")
    n = int(np.floor(bins_per_octave * np.log2(fmax / fmin))) + 1
    return fmin * 2.0 ** (np.arange(n) / float(bins_per_octave))


def _cq_quality(bins_per_octave):
    """Constant quality factor Q = 1 / (2**(1/B) - 1)."""
    return 1.0 / (2.0 ** (1.0 / float(bins_per_octave)) - 1.0)


def _cq_kernels(frequencies, Q, fs, window):
    """List of ``(N_k, kernel, density_factor)`` per bin.

    ``kernel = w·exp(-2j·pi·Q·n/N_k) / Σw`` (so ``|X|**2`` is band power, the
    'spectrum' scaling). ``density_factor = (Σw)**2 / (fs·Σw**2)`` converts that
    band power to a one-sided PSD (per Hz), matching ``scipy.signal.welch``
    density scaling; the one-sided factor 2 is applied in
    :func:`_cq_power_frames`.
    """
    kernels = []
    for fk in frequencies:
        Nk = max(1, int(np.ceil(Q * fs / fk)))
        w = get_window(window, Nk, fftbins=True)
        sw = float(np.sum(w))
        sw2 = float(np.sum(w * w))
        n = np.arange(Nk)
        ker = (w * np.exp(-2j * np.pi * Q * n / Nk)) / sw
        # |X|**2 captures only the positive-frequency sideband of a real signal
        # (a tone A·cos gives |X|=A/2); _cq_power_frames doubles it to one-sided
        # band power A**2/2. density_factor then divides by the noise-equivalent
        # bandwidth fs·Σw²/(Σw)² to give a one-sided PSD (welch-density).
        density_factor = sw * sw / (fs * sw2)
        kernels.append((Nk, ker.astype(np.complex128), density_factor))
    return kernels


def _cq_frame(x, center, kernels):
    """``(coeffs, valid)``: complex constant-Q band amplitudes with each bin's
    window centred on sample ``center`` (zero-padded at the signal edges), and a
    per-bin bool — ``True`` where the window lay fully inside the signal."""
    nx = x.size
    coeffs = np.empty(len(kernels), dtype=np.complex128)
    valid = np.empty(len(kernels), dtype=bool)
    for i, (Nk, ker, _) in enumerate(kernels):
        start = center - Nk // 2
        stop = start + Nk
        inside = start >= 0 and stop <= nx
        valid[i] = inside
        if inside:
            coeffs[i] = np.sum(x[start:stop] * ker)
        else:
            a, b = max(start, 0), min(stop, nx)
            seg = np.zeros(Nk, dtype=float)
            seg[a - start: b - start] = x[a:b]
            coeffs[i] = np.sum(seg * ker)
    return coeffs, valid


def _cq_power_frames(x, fs, kernels, hop, scaling):
    """Per-frame constant-Q power ``(n_bins, n_frames)`` and a matching per-bin
    validity mask (``True`` where the window lay fully inside the signal)."""
    centers = np.arange(0, x.size, hop)
    dfac = np.array([k[2] for k in kernels])
    power = np.empty((len(kernels), centers.size), dtype=float)
    valid = np.empty((len(kernels), centers.size), dtype=bool)
    for j, c in enumerate(centers):
        coeffs, ok = _cq_frame(x, int(c), kernels)
        p = 2.0 * np.abs(coeffs) ** 2          # one-sided band power (tone -> A²/2)
        power[:, j] = p * dfac if scaling == "density" else p
        valid[:, j] = ok
    return centers / fs, power, valid


def _check_scaling(scaling, caller):
    if scaling not in _SCALINGS:
        raise ConfigurationError(
            f"{caller}: scaling must be one of {_SCALINGS}; got {scaling!r}")


def _resolve_hop(hop, kernels, n_samples, caller):
    n_lowest = kernels[0][0]
    if hop is None:
        # 1/8 of the longest window (8x overlap of the lowest bin), but capped
        # so a short signal still yields enough frame centres to average.
        hop = max(1, min(n_lowest // 8, max(1, n_samples // 8)))
    hop = int(hop)
    if hop < 1:
        raise ConfigurationError(f"{caller}: hop must be >= 1")
    return hop


def _cq_setup(data, sample_rate, fmin, fmax, bins_per_octave, window, caller):
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise ConfigurationError(
            f"{caller}: data must be a real signal; got complex input "
            "(a complex array would be silently real-cast).")
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ConfigurationError(f"{caller}: data must be 1-D")
    require_finite_signal(x, caller)
    fs = float(sample_rate)
    if fmax is None:
        fmax = fs / 2.0
    if fmax > fs / 2.0:
        raise ConfigurationError(
            f"{caller}: fmax ({fmax}) exceeds Nyquist ({fs / 2.0})")
    freqs = _cq_frequencies(fmin, fmax, bins_per_octave)
    Q = _cq_quality(bins_per_octave)
    kernels = _cq_kernels(freqs, Q, fs, window)
    n_lowest = kernels[0][0]
    if n_lowest > x.size:
        warnings.warn(
            f"{caller}: lowest bin needs {n_lowest} samples (Q·fs/fmin) but the "
            f"signal has {x.size}; low-frequency bins never fit a full window "
            "and are dropped from the average. Raise fmin or lengthen the "
            "signal.",
            UserWarning, stacklevel=2)
    return x, fs, freqs, kernels


# ── public API ──────────────────────────────────────────────────────────────
def constant_q_transform(data, sample_rate, *, fmin=20.0, fmax=None,
                         bins_per_octave=24, window="hann"):
    """Complex constant-Q spectrum of one frame centred on the signal.

    Returns a :class:`CQTResult` ``(frequencies, coefficients)`` — the centre
    frequencies (Hz) and complex band amplitudes (``abs`` is the magnitude
    spectrum). Best for short transients; for longer signals use
    :func:`constant_q_spectrogram` / :func:`constant_q_psd`.
    """
    x, fs, freqs, kernels = _cq_setup(
        data, sample_rate, fmin, fmax, bins_per_octave, window,
        "constant_q_transform")
    coeffs, _ = _cq_frame(x, x.size // 2, kernels)
    return CQTResult(freqs, coeffs)


def constant_q_spectrogram(data, sample_rate, *, fmin=20.0, fmax=None,
                           bins_per_octave=24, hop=None, window="hann",
                           scaling="spectrum"):
    """Constant-Q spectrogram: constant-Q power over time.

    Slides the constant-Q analysis at ``hop`` samples (default: 1/8 of the
    longest window, i.e. of the lowest-frequency bin). ``scaling='spectrum'``
    (default) returns band power ``|X_cq|**2``; ``scaling='density'`` returns a
    one-sided PSD per Hz. Returns a :class:`CQSpectrogramResult`
    ``(frequencies, times, power)`` with ``power`` shaped ``(n_freqs,
    n_frames)`` — every frame is kept, so edge frames carry visible edge
    effects.
    """
    _check_scaling(scaling, "constant_q_spectrogram")
    x, fs, freqs, kernels = _cq_setup(
        data, sample_rate, fmin, fmax, bins_per_octave, window,
        "constant_q_spectrogram")
    hop = _resolve_hop(hop, kernels, x.size, "constant_q_spectrogram")
    times, power, _valid = _cq_power_frames(x, fs, kernels, hop, scaling)
    return CQSpectrogramResult(freqs, times, power)


def constant_q_psd(data, sample_rate, *, fmin=20.0, fmax=None,
                   bins_per_octave=24, hop=None, window="hann",
                   scaling="spectrum"):
    """Time-averaged constant-Q power per geometric frequency bin.

    Returns a :class:`CQPSDResult` ``(frequencies, power)`` — the constant-Q
    analogue of :func:`uacpy.acoustic_signal.psd`. The average is taken, *per
    bin*, only over frames whose window lay fully inside the signal (zero-padded
    edge frames are excluded); a bin with no such frame is ``NaN``.
    ``scaling='spectrum'`` (default) gives band power per bin; ``'density'``
    gives a one-sided PSD per Hz.
    """
    _check_scaling(scaling, "constant_q_psd")
    x, fs, freqs, kernels = _cq_setup(
        data, sample_rate, fmin, fmax, bins_per_octave, window,
        "constant_q_psd")
    hop = _resolve_hop(hop, kernels, x.size, "constant_q_psd")
    _, power, valid = _cq_power_frames(x, fs, kernels, hop, scaling)
    avg = np.full(freqs.size, np.nan)
    counts = valid.sum(axis=1)
    good = counts > 0
    avg[good] = np.array([power[i, valid[i]].mean() for i in np.flatnonzero(good)])
    return CQPSDResult(freqs, avg)


def probabilistic_constant_q(data, sample_rate, *, fmin=20.0, fmax=None,
                             bins_per_octave=24, hop=None, window="hann",
                             scaling="spectrum", ddB=1.0, lvlmin=0,
                             lvlmax=150, ref=REFERENCE_PRESSURE_WATER):
    """Probability density of constant-Q power levels over time.

    Histograms the per-bin dB levels of the constant-Q frames, the constant-Q
    analogue of :func:`uacpy.acoustic_signal.ppsd`. Only frames whose window lay
    fully inside the signal contribute (per bin). Returns a :class:`CQPPSDResult`
    ``(frequencies, level_edges, pdf, mean_db, std_db, binwidth_db)``; ``pdf`` is
    shaped ``(n_levels, n_freqs)`` and density-normalised per frequency column
    (empty bins are ``NaN``). With ``scaling='density'`` the levels are PSD
    levels (dB re ref²/Hz) rather than band-power levels (dB re ref²).
    """
    _check_scaling(scaling, "probabilistic_constant_q")
    x, fs, freqs, kernels = _cq_setup(
        data, sample_rate, fmin, fmax, bins_per_octave, window,
        "probabilistic_constant_q")
    hop = _resolve_hop(hop, kernels, x.size, "probabilistic_constant_q")
    _, power, valid = _cq_power_frames(x, fs, kernels, hop, scaling)
    level_edges = np.arange(lvlmin, lvlmax + ddB, ddB)
    pdf = np.zeros((level_edges.size - 1, freqs.size))
    mean_db = np.full(freqs.size, np.nan)
    std_db = np.full(freqs.size, np.nan)
    with np.errstate(divide="ignore"):
        levels_db = power_to_db(power, ref)
    for i in range(freqs.size):
        vals = levels_db[i, valid[i]]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            hist, _ = np.histogram(vals, bins=level_edges, density=True)
        pdf[:, i] = hist
        mean_db[i] = vals.mean()
        std_db[i] = vals.std()
    pdf[pdf == 0] = np.nan
    return CQPPSDResult(freqs, level_edges, pdf, mean_db, std_db, ddB)
