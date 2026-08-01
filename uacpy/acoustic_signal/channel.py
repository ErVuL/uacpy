"""Time-domain channel simulation from a modelled multipath structure.

Turns a propagation model's arrival structure (amplitudes + travel times, e.g.
from a Bellhop ARRIVALS run) or a transfer function ``H(f)`` into a channel
impulse response, then convolves a transmit waveform with it to synthesise the
received signal — the basis of replay benchmarks for underwater communications
and active-sonar echo simulation.
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def impulse_response(amplitudes, delays_s, sample_rate: float, *,
                     n_samples: int = None, fractional: bool = True):
    """Channel impulse response from discrete arrivals.

    Places each arrival ``amplitudes[i]`` at delay ``delays_s[i]``. With
    ``fractional=True`` the amplitude is split linearly between the two
    nearest samples (band-unlimited fractional delay); otherwise it is placed
    at the nearest sample.

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
        Linear two-tap fractional-delay placement.

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
    if n_samples is None:
        if pos.size:
            n_samples = int(np.floor(pos.max()) + 2 if fractional
                            else np.round(pos.max()) + 1)
        else:
            n_samples = 1
    dtype = complex if np.iscomplexobj(a) else float
    h = np.zeros(int(n_samples), dtype=dtype)
    for amp, p in zip(a, pos):
        i0 = int(np.floor(p)) if fractional else int(np.round(p))
        if i0 < 0 or i0 >= n_samples:
            continue
        if fractional and i0 + 1 < n_samples:
            frac = p - i0
            h[i0] += amp * (1.0 - frac)
            h[i0 + 1] += amp * frac
        else:
            h[i0] += amp
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
        n_samples = 2 * (f.size - 1) if f.size > 1 else 2
    grid = np.fft.rfftfreq(int(n_samples), d=1.0 / fs)
    Hr = (np.interp(grid, f, Hc.real, left=0.0, right=0.0)
          + 1j * np.interp(grid, f, Hc.imag, left=0.0, right=0.0))
    h = np.fft.irfft(Hr, n=n_samples)
    t = np.arange(n_samples) / fs
    return t, h
