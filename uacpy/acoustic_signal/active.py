"""Active-sonar waveform processing: matched filtering and ambiguity.

Processes the echoes of the waveforms produced by
:mod:`uacpy.acoustic_signal.waveforms` / :mod:`~uacpy.acoustic_signal.sequences`
(LFM/HFM chirps, m-sequences, ...): replica correlation (pulse compression),
matched-filter processing gain, and the narrowband range-Doppler ambiguity
function for waveform-resolution analysis.

References
----------
Burdic, W.S. *Underwater Acoustic System Analysis*, Ch. 9.
Richards, M.A. *Fundamentals of Radar Signal Processing* (matched filter,
    pulse compression, ambiguity function).
"""

from __future__ import annotations

import numpy as np
import scipy.signal as _sig
from scipy.signal import fftconvolve

from uacpy.core.exceptions import ConfigurationError


def matched_filter(received, replica, *, mode: str = "full", normalize: bool = True):
    """Matched-filter (replica-correlation) output = pulse compression.

    Cross-correlates ``received`` with ``replica`` by convolving against the
    conjugated, time-reversed replica. For an LFM/HFM/m-sequence transmit, the
    output peaks at each echo delay with main-lobe width ~ ``1/B``. Complex
    inputs are supported (pass analytic signals for bandpass data).

    Parameters
    ----------
    received, replica : 1-D array
        Received record and the transmitted replica.
    mode : {'full', 'same', 'valid'}
        Output length convention (see ``scipy.signal.fftconvolve``).
    normalize : bool
        Divide by the replica energy so a perfectly matched, unit-amplitude
        echo compresses to unit peak.

    Returns
    -------
    ndarray
        Matched-filter output (complex if either input is complex).
    """
    r = np.asarray(received)
    h = np.asarray(replica)
    if r.ndim != 1 or h.ndim != 1:
        raise ConfigurationError("matched_filter: received and replica must be 1-D")
    if h.size == 0:
        raise ConfigurationError("matched_filter: replica must be non-empty")
    y = fftconvolve(r, np.conj(h[::-1]), mode=mode)
    if normalize:
        y = y / np.sum(np.abs(h) ** 2)
    return y


def pulse_compression(received, replica, sample_rate: float, *, normalize: bool = True):
    """Matched filter returning a lag axis in seconds.

    Returns ``(lags_s, compressed)`` where ``lags_s`` is the echo-delay axis
    (lag 0 = replica aligned with the start of ``received``).
    """
    r = np.asarray(received)
    h = np.asarray(replica)
    comp = matched_filter(r, h, mode="full", normalize=normalize)
    lags = (np.arange(comp.size) - (h.size - 1)) / float(sample_rate)
    return lags, comp


def processing_gain(bandwidth_hz: float, duration_s: float) -> float:
    """Matched-filter processing gain (dB) = ``10*log10(B*T)``."""
    bt = float(bandwidth_hz) * float(duration_s)
    if bt <= 0.0:
        raise ConfigurationError("processing_gain: B*T must be > 0")
    return float(10.0 * np.log10(bt))


def ambiguity_function(waveform, sample_rate: float, *, doppler_hz=None,
                       n_doppler: int = 101):
    """Narrowband range-Doppler ambiguity surface ``|chi(tau, nu)|``.

    ``chi(tau, nu) = sum_t s(t) conj(s(t - tau)) exp(j*2*pi*nu*t)``, normalised
    so the matched peak at ``(0, 0)`` is 1.

    Parameters
    ----------
    waveform : 1-D array
        Complex (or real) baseband transmit waveform.
    sample_rate : float
        Sample rate (Hz).
    doppler_hz : array, optional
        Doppler shifts (Hz) to evaluate. Default: ``n_doppler`` points spanning
        ``+/- sample_rate/20``.
    n_doppler : int
        Number of Doppler bins when ``doppler_hz`` is None.

    Returns
    -------
    lags_s : ndarray
        Delay axis (s), length ``2*N-1``.
    doppler_hz : ndarray
        Doppler axis (Hz).
    amplitude : ndarray
        ``|chi|`` with shape ``(n_doppler, 2*N-1)``.
    """
    s = np.asarray(waveform, dtype=complex)
    if s.ndim != 1 or s.size == 0:
        raise ConfigurationError("ambiguity_function: waveform must be 1-D non-empty")
    n = s.size
    fs = float(sample_rate)
    t = np.arange(n) / fs
    if doppler_hz is None:
        fmax = fs / 20.0
        doppler_hz = np.linspace(-fmax, fmax, int(n_doppler))
    doppler_hz = np.atleast_1d(np.asarray(doppler_hz, dtype=float))
    energy = np.sum(np.abs(s) ** 2)
    rev = np.conj(s[::-1])
    amp = np.empty((doppler_hz.size, 2 * n - 1))
    for i, nu in enumerate(doppler_hz):
        sd = s * np.exp(2j * np.pi * nu * t)
        amp[i] = np.abs(fftconvolve(sd, rev, mode="full"))
    amp /= energy
    lags = (np.arange(2 * n - 1) - (n - 1)) / fs
    return lags, doppler_hz, amp


def shift_to_max_correlation(x, y):
    """
    Shift two signals based on the maximum cross-correlation.

    Computes the cross-correlation between the signals `x` and `y`,
    extracts the lag that maximizes the correlation, and shifts one of
    the signals accordingly.

    Parameters
    ----------
    x : ndarray
        First signal to be shifted.
    y : ndarray
        Second signal to be shifted.

    Returns
    -------
    x : ndarray
        Shifted first signal.
    y : ndarray
        Shifted second signal.
    """

    correlation = _sig.correlate(x, y, mode="full")
    lags = _sig.correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(correlation)]

    if lag < 0:
        y = y[-lag:]
        x = x[:lag]
    elif lag > 0:
        x = x[lag:]
        y = y[:-lag]
    # lag == 0: signals already aligned, no trimming (y[:-0] would empty y).

    return x, y


