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

from collections import namedtuple

import numpy as np
from scipy.signal import fftconvolve

from uacpy.core.exceptions import ConfigurationError
from uacpy.acoustic_signal._signal_validate import require_positive_finite_scalar


AmbiguityResult = namedtuple("AmbiguityResult", "delays_s doppler_hz amplitude")


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
        raise ConfigurationError(
            "matched_filter: received and replica must be 1-D; got received "
            f"shape {r.shape} and replica shape {h.shape}")
    if h.size == 0:
        raise ConfigurationError("matched_filter: replica must be non-empty")
    energy = float(np.sum(np.abs(h) ** 2))
    if normalize and energy == 0.0:
        raise ConfigurationError(
            "matched_filter: replica carries no energy, so the unit-peak "
            "normalisation divides by zero and every output sample is nan. "
            "Pass the transmitted waveform as the replica, or normalize=False "
            "for the un-normalised correlation.")
    y = fftconvolve(r, np.conj(h[::-1]), mode=mode)
    if normalize:
        y = y / energy
    return y


def pulse_compression(received, replica, sample_rate: float, *, normalize: bool = True):
    """Matched filter returning a lag axis in seconds.

    Returns ``(lags_s, compressed)`` where ``lags_s`` is the echo-delay axis
    (lag 0 = replica aligned with the start of ``received``).
    """
    r = np.asarray(received)
    h = np.asarray(replica)
    fs = require_positive_finite_scalar(
        sample_rate, "pulse_compression", "sample_rate", " Hz")
    comp = matched_filter(r, h, mode="full", normalize=normalize)
    lags = (np.arange(comp.size) - (h.size - 1)) / fs
    return lags, comp


def processing_gain(bandwidth_hz: float, duration_s: float) -> float:
    """Matched-filter processing gain (dB) = ``10*log10(B*T)``."""
    bt = float(bandwidth_hz) * float(duration_s)
    if bt <= 0.0:
        raise ConfigurationError(
            f"processing_gain: B*T must be > 0; got bandwidth_hz="
            f"{bandwidth_hz!r} x duration_s={duration_s!r} = {bt:g}")
    return float(10.0 * np.log10(bt))


# Largest ambiguity surface ``ambiguity_function`` will allocate:
# n_doppler x (2N-1) float64 cells. 2**27 cells is 1 GiB, matching the 1 GiB
# ceiling ``channel._MAX_DEFAULT_TAPS`` sets for the same reason. A 10 s pulse
# at 96 kHz on the default 101-Doppler grid asks for 1.44 GiB, and the Python
# loop of full-length fftconvolves over it runs for minutes, so past this the
# caller decimates or narrows the grid rather than having one long waveform
# allocate that much silently.
_MAX_AMBIGUITY_CELLS = 1 << 27


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
    AmbiguityResult
        Namedtuple ``(delays_s, doppler_hz, amplitude)``: the delay axis (s,
        length ``2*N-1``), the Doppler axis (Hz), and ``|chi|`` with shape
        ``(n_doppler, 2*N-1)``.
    """
    s = np.asarray(waveform, dtype=complex)
    if s.ndim != 1 or s.size == 0:
        raise ConfigurationError(
            "ambiguity_function: waveform must be 1-D non-empty; "
            f"got shape {s.shape}")
    n = s.size
    fs = require_positive_finite_scalar(
        sample_rate, "ambiguity_function", "sample_rate", " Hz")
    t = np.arange(n) / fs
    if doppler_hz is None:
        fmax = fs / 20.0
        doppler_hz = np.linspace(-fmax, fmax, int(n_doppler))
    doppler_hz = np.atleast_1d(np.asarray(doppler_hz, dtype=float))
    cells = int(doppler_hz.size) * (2 * n - 1)
    if cells > _MAX_AMBIGUITY_CELLS:
        raise ConfigurationError(
            f"ambiguity_function: the surface would be "
            f"{doppler_hz.size} x {2 * n - 1} = {cells} float64 cells "
            f"({cells * 8 / 2 ** 30:.2f} GiB), past the "
            f"{_MAX_AMBIGUITY_CELLS} cell cap "
            f"({_MAX_AMBIGUITY_CELLS * 8 / 2 ** 30:.2f} GiB). It also runs "
            f"one full-length FFT convolution per Doppler row. Shorten the "
            f"waveform, decimate it, or pass fewer n_doppler / doppler_hz "
            f"candidates.")
    energy = float(np.sum(np.abs(s) ** 2))
    if energy == 0.0:
        raise ConfigurationError(
            "ambiguity_function: waveform carries no energy, so normalising "
            "the (0, 0) peak to 1 divides by zero and the whole surface is "
            "nan. Pass the transmit waveform.")
    rev = np.conj(s[::-1])
    amp = np.empty((doppler_hz.size, 2 * n - 1))
    for i, nu in enumerate(doppler_hz):
        sd = s * np.exp(2j * np.pi * nu * t)
        amp[i] = np.abs(fftconvolve(sd, rev, mode="full"))
    amp /= energy
    lags = (np.arange(2 * n - 1) - (n - 1)) / fs
    return AmbiguityResult(lags, doppler_hz, amp)
