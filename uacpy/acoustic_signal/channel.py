"""Time-domain channel simulation from a modelled multipath structure.

Turns a propagation model's arrival structure (amplitudes + travel times, e.g.
from a Bellhop ARRIVALS run) or a transfer function ``H(f)`` into a channel
impulse response, then convolves a transmit waveform with it to synthesise the
received signal — the basis of replay benchmarks for underwater communications
and active-sonar echo simulation.
"""

from __future__ import annotations

import warnings

import numpy as np

from uacpy.core.exceptions import ConfigurationError


# Windowed-sinc fractional-delay kernel: taps each side of the arrival and
# the Kaiser window shape. L=8 is flat to ~0.01 dB below f/fs = 0.35.
_FRAC_DELAY_HALF_LEN = 8
_FRAC_DELAY_KAISER_BETA = 8.0


def impulse_response(amplitudes, delays_s, sample_rate: float, *,
                     n_samples: int = None, fractional: bool = True):
    """Channel impulse response from discrete arrivals.

    Places each arrival ``amplitudes[i]`` at delay ``delays_s[i]``. With
    ``fractional=True`` the arrival is placed with a windowed-sinc
    fractional-delay kernel; otherwise it is quantised to the nearest sample.

    The kernel matters because a two-tap linear split is **not** a fractional
    delay: its response ``|(1-frac) + frac*e^{-jw}|`` is a lowpass whose
    attenuation depends on ``frac``, with a full null at Nyquist for
    ``frac = 0.5`` (-3.0 dB at ``f/fs = 0.25``, -10.2 dB at 0.40). Two
    arrivals a propagation model reports as equal then came back differing by
    up to 10 dB, decided by the sub-sample part of their travel times — at the
    right time, at the wrong level. The windowed sinc is flat to ~0.01 dB
    over the same band. Group delay was correct either way.

    Parameters
    ----------
    amplitudes : 1-D array
        Complex (or real) arrival amplitudes.
    delays_s : 1-D array
        Arrival travel times (s), >= 0.
    sample_rate : float
        Sample rate (Hz).
    n_samples : int, optional
        Length of the IR. Default: just past the latest arrival.
    fractional : bool
        Windowed-sinc fractional-delay placement. ``False`` quantises the
        delay to the nearest sample (+/- 0.5 sample of timing error).

    Returns
    -------
    t : ndarray
        Time axis (s).
    h : ndarray
        Impulse response (complex if amplitudes are complex).
    """
    a = np.asarray(amplitudes)
    d = np.asarray(delays_s, dtype=float)
    if a.shape != d.shape or a.ndim != 1:
        raise ConfigurationError("impulse_response: amplitudes and delays_s must be 1-D, equal length")
    if np.any(d < 0):
        raise ConfigurationError("impulse_response: delays_s must be >= 0")
    fs = float(sample_rate)
    pos = d * fs
    L = _FRAC_DELAY_HALF_LEN
    if n_samples is None:
        if pos.size:
            # Non-fractional rounds to the nearest sample, so the last
            # occupied index is round(pos.max()), not floor(pos.max()).
            n_samples = int(np.floor(pos.max()) + L + 1 if fractional
                            else np.round(pos.max()) + 1)
        else:
            n_samples = 1
    n_samples = int(n_samples)
    dtype = complex if np.iscomplexobj(a) else float
    h = np.zeros(n_samples, dtype=dtype)
    n_clipped = 0
    for amp, p in zip(a, pos):
        if not fractional:
            i0 = int(np.round(p))
            if 0 <= i0 < n_samples:
                h[i0] += amp
            continue
        i0 = int(np.floor(p))
        k = np.arange(i0 - L + 1, i0 + L + 1)
        u = (k - p) / L
        win = (np.i0(_FRAC_DELAY_KAISER_BETA
                     * np.sqrt(np.maximum(0.0, 1.0 - u * u)))
               / np.i0(_FRAC_DELAY_KAISER_BETA))
        g = np.sinc(k - p) * win
        # Normalise to unit DC gain: windowing truncates the sinc, so the
        # raw taps sum to 0.99999579 and a constant signal would lose
        # 3.7e-5 dB. Standard for an interpolation kernel.
        gsum = g.sum()
        if gsum:
            g = g / gsum
        ok = (k >= 0) & (k < n_samples)
        # A truncated kernel loses amplitude; say so rather than silently
        # dumping the whole arrival on one sample, which is what the old
        # else-branch did (an arrival at 10.9 samples landed entirely at 10).
        if not ok.all() and p != i0:
            n_clipped += 1
        h[k[ok]] += amp * g[ok]
    if n_clipped:
        warnings.warn(
            f"impulse_response: {n_clipped} fractional arrival(s) sit within "
            f"{L} samples of the ends of an {n_samples}-sample response, so "
            f"their interpolation kernel is truncated and they lose amplitude. "
            f"Lengthen n_samples, or use fractional=False to quantise instead.",
            UserWarning, stacklevel=2)
    t = np.arange(n_samples) / fs
    return t, h


def simulate_reception(transmit, amplitudes, delays_s, sample_rate: float):
    """Received signal = ``transmit`` convolved with the channel IR.

    Returns ``(t, received)`` with ``t`` the output time axis (s).
    """
    x = np.asarray(transmit)
    _, h = impulse_response(amplitudes, delays_s, sample_rate)
    y = np.convolve(x, h)
    t = np.arange(y.size) / float(sample_rate)
    return t, y


def impulse_response_from_transfer_function(H, frequencies, sample_rate: float,
                                            n_samples: int = None):
    """Real impulse response from a one-sided transfer function ``H(f)``.

    Resamples ``H`` onto a uniform DFT grid ``[0, fs/2]`` and inverse-transforms.
    ``frequencies`` must be non-negative and increasing. Grid bins outside
    ``[frequencies[0], frequencies[-1]]`` are **zero** — a band-limited model
    ``H(f)`` carries no energy out of band (constant extrapolation would
    fabricate a DC/high-frequency plateau in the impulse response).

    The DFT grid spacing ``df`` sets the **unambiguous delay window**
    ``1/df``. An arrival later than that wraps to ``tau mod 1/df`` and is
    then indistinguishable from a genuine early one — measured, ``df = 62.5``
    Hz (a 16 ms window) puts a 20 ms delay at 4.000 ms, and nothing in the
    returned ``h`` reveals it. No check here can catch it, so the caller
    sizes the grid: ``df < 1 / tau_max`` for the longest delay the channel
    can produce (``range_max / c_min`` for a propagation model).
    ``df = sample_rate / n_samples`` when ``n_samples`` is given, else the
    spacing of ``frequencies``.

    Returns ``(t, h)``.
    """
    f = np.asarray(frequencies, dtype=float)
    Hc = np.asarray(H, dtype=complex)
    if f.ndim != 1 or f.shape != Hc.shape:
        raise ConfigurationError("impulse_response_from_transfer_function: H and frequencies shapes differ")
    if np.any(np.diff(f) <= 0) or f[0] < 0:
        raise ConfigurationError("frequencies must be non-negative and strictly increasing")
    fs = float(sample_rate)
    if n_samples is None:
        # Inverse of the rfft bin count: an even-length real signal of
        # 2*(K - 1) samples has exactly K one-sided bins, so the default grid
        # is as fine as the supplied H(f) and no finer.
        n_samples = 2 * (f.size - 1) if f.size > 1 else 2
    grid = np.fft.rfftfreq(int(n_samples), d=1.0 / fs)
    Hr = (np.interp(grid, f, Hc.real, left=0.0, right=0.0)
          + 1j * np.interp(grid, f, Hc.imag, left=0.0, right=0.0))
    h = np.fft.irfft(Hr, n=n_samples)
    t = np.arange(n_samples) / fs
    return t, h
