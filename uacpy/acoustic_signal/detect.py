"""Finding a known signal in a record: matched filtering and ambiguity.

One question — *is my transmission in there, and where* — answered by
:func:`matched_filter` and :func:`pulse_compression`, with
:func:`processing_gain_dB` for what the compression buys and
:func:`ambiguity_function` for how delay and Doppler trade against each other.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional
import warnings
import numpy as np
from scipy.signal import fftconvolve
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._validate import require_positive_finite_scalar
from uacpy.acoustic_signal._results import PlottedResult
from uacpy.acoustic_signal.spectral import _time_axis_last


# ──────────────────────────────────────────────────────────────────────
# Matched filtering and ambiguity
#
# Finding a known transmission in a
# record, and what the compression buys.
# ──────────────────────────────────────────────────────────────────────


class AmbiguityResult(PlottedResult,
                      namedtuple("AmbiguityResult", "delays doppler_shifts amplitude")):
    """Ambiguity surface ``|chi|`` over ``delays`` and ``doppler_shifts``.

    The tuple is the measurement, so ``delays, doppler_shifts, amplitude = ...``
    keeps working; :meth:`plot` is the one obvious way to draw it.
    """

    __slots__ = ()

    _plotter = "plot_ambiguity"
    _plot_fields = ("delays", "doppler_shifts", "amplitude")

    def _field_units(self):
        return {"delays": "s", "doppler_shifts": "Hz", "amplitude": ""}


def matched_filter(received, replica, *, mode: str = "full", normalize: bool = True,
                   axis=None):
    """Matched-filter (replica-correlation) output = pulse compression.

    The sonar operation: correlate against a known transmit waveform. The
    comms receive filter that completes a Nyquist pulse is
    :func:`uacpy.comms.rrc_matched_filter`, a different thing under a name
    that is standard in its own field.

    Cross-correlates ``received`` with ``replica`` by convolving against the
    conjugated, time-reversed replica. For an LFM/HFM/m-sequence transmit, the
    output peaks at each echo delay with main-lobe width ~ ``1/B``. Complex
    inputs are supported (pass analytic signals for bandpass data).

    Parameters
    ----------
    received : array_like
        Received record, time along ``axis``; every other axis is a channel
        (an array recording), each filtered with the one replica.
    replica : 1-D array
        The transmitted replica.
    mode : {'full', 'same', 'valid'}
        Output length convention (see ``scipy.signal.fftconvolve``).
    normalize : bool
        Divide by the replica energy so a perfectly matched, unit-amplitude
        echo compresses to unit peak.
    axis : int, optional
        The time axis of a multichannel ``received``, as on
        :func:`~uacpy.acoustic_signal.welch`: unset, the last axis, and a 2-D
        record shaped like ``read_wav``'s ``(n_samples, n_channels)`` is
        refused rather than filtered across its channels — pass ``axis=0``.

    Returns
    -------
    ndarray
        Matched-filter output (complex if either input is complex): the
        channel axes first, the lag axis last.
    """
    h = np.asarray(replica)
    if h.ndim != 1:
        raise ConfigurationError(
            f"matched_filter: replica must be 1-D; got replica shape "
            f"{h.shape}.")
    r = np.asarray(_time_axis_last(received, axis, "matched_filter"))
    if r.ndim == 0:
        raise ConfigurationError(
            "matched_filter: received has no time axis; got a scalar.")
    if h.size == 0:
        raise ConfigurationError("matched_filter: replica must be non-empty.")
    energy = float(np.sum(np.abs(h) ** 2))
    if normalize and energy == 0.0:
        raise ConfigurationError(
            "matched_filter: replica carries no energy, so the unit-peak "
            "normalisation divides by zero and every output sample is nan. "
            "Pass the transmitted waveform as the replica, or normalize=False "
            "for the un-normalised correlation.")
    kernel = np.conj(h[::-1]).reshape((1,) * (r.ndim - 1) + (h.size,))
    y = fftconvolve(r, kernel, mode=mode, axes=-1)
    if normalize:
        y = y / energy
    return y


def pulse_compression(received, replica, sample_rate: float, *, normalize: bool = True,
                      axis=None):
    """Matched filter returning a lag axis in seconds.

    Returns ``(lags, compressed)`` where ``lags`` is the echo-delay axis in
    seconds (lag 0 = replica aligned with the start of ``received``).

    Parameters
    ----------
    received : array_like
        The received record.
    replica : array_like
        The transmitted waveform.
    sample_rate : float
        Sample rate (Hz).
    normalize : bool, optional
        Divide by the replica energy, so a matched unit-amplitude echo
        compresses to unit peak. Default True.
    axis : int, optional
        The time axis of a multichannel ``received``, as on
        :func:`matched_filter`; ``compressed`` then carries the channel axes
        first and the lag axis last.
    """
    h = np.asarray(replica)
    fs = require_positive_finite_scalar(
        sample_rate, "pulse_compression", "sample_rate", " Hz")
    comp = matched_filter(received, h, mode="full", normalize=normalize,
                          axis=axis)
    lags = (np.arange(comp.shape[-1]) - (h.size - 1)) / fs
    return lags, comp


def processing_gain_dB(bandwidth: float, duration: float) -> float:
    """Matched-filter processing gain (dB) = ``10*log10(B*T)``.

    For a spreading code the same gain is ``10*log10(N)`` with ``N`` the chip
    count (``N = B*T`` for chips of duration ``1/B``); that form, taking the
    code itself, is :func:`uacpy.comms.spreading_gain_dB`.

    Parameters
    ----------
    bandwidth : float
        Signal bandwidth (Hz).
    duration : float
        Signal duration (s).
    """
    # Each factor on its own: two negatives make a positive product, and a
    # NaN fails every comparison.
    bandwidth, duration = float(bandwidth), float(duration)
    if not (np.isfinite(bandwidth) and bandwidth > 0.0
            and np.isfinite(duration) and duration > 0.0):
        raise ConfigurationError(
            f"processing_gain_dB: B and T must each be > 0 and finite; got "
            f"bandwidth={bandwidth!r}, duration={duration!r}.")
    return float(10.0 * np.log10(bandwidth * duration))


# Largest ambiguity surface ``ambiguity_function`` will allocate:
# n_doppler_shifts x (2N-1) float64 cells. 2**27 cells is 1 GiB, matching the 1 GiB
# ceiling ``channel._MAX_DEFAULT_TAPS`` sets for the same reason. A 10 s pulse
# at 96 kHz on the default 101-Doppler grid asks for 1.44 GiB, and the Python
# loop of full-length fftconvolves over it runs for minutes, so past this the
# caller decimates or narrows the grid rather than having one long waveform
# allocate that much silently.
_MAX_AMBIGUITY_CELLS = 1 << 27


#: Doppler bins :func:`ambiguity_function` evaluates when ``n_doppler_shifts`` is
#: unset, spanning ``+/- sample_rate / _AMBIGUITY_DOPPLER_SPAN_DIVISOR``
#: (uacpy guidance).
_AMBIGUITY_N_DOPPLER = 101
_AMBIGUITY_DOPPLER_SPAN_DIVISOR = 20.0


def ambiguity_function(waveform, sample_rate: float, *, doppler_shifts=None,
                       n_doppler_shifts: Optional[int] = None):
    """Narrowband range-Doppler ambiguity surface ``|chi(tau, nu)|``.

    ``chi(tau, nu) = sum_t s(t) conj(s(t - tau)) exp(j*2*pi*nu*t)``, normalised
    so the matched peak at ``(0, 0)`` is 1: the narrowband ambiguity function
    (Abraham, *Underwater Acoustic Signal Processing*, §8.3.2.1).

    Parameters
    ----------
    waveform : 1-D array
        Complex (or real) baseband transmit waveform.
    sample_rate : float
        Sample rate (Hz).
    doppler_shifts : array, optional
        Doppler shifts (Hz) to evaluate. Default: ``n_doppler_shifts`` points spanning
        ``+/- sample_rate/20`` (uacpy guidance).
    n_doppler_shifts : int, optional
        Number of Doppler bins when ``doppler_shifts`` is None; unset, 101
        (uacpy guidance). Given with ``doppler_shifts`` it is not used and
        gets a ``FallbackWarning``.

    Returns
    -------
    AmbiguityResult
        Namedtuple ``(delays, doppler_shifts, amplitude)``: the delay axis (s,
        length ``2*N-1``), the Doppler axis (Hz), and ``|chi|`` with shape
        ``(n_doppler_shifts, 2*N-1)``.
    """
    s = np.asarray(waveform, dtype=complex)
    if s.ndim != 1 or s.size == 0:
        raise ConfigurationError(
            "ambiguity_function: waveform must be 1-D non-empty; "
            f"got shape {s.shape}.")
    n = s.size
    fs = require_positive_finite_scalar(
        sample_rate, "ambiguity_function", "sample_rate", " Hz")
    t = np.arange(n) / fs
    if doppler_shifts is None:
        fmax = fs / _AMBIGUITY_DOPPLER_SPAN_DIVISOR
        doppler_shifts = np.linspace(
            -fmax, fmax,
            _AMBIGUITY_N_DOPPLER if n_doppler_shifts is None else int(n_doppler_shifts))
    elif n_doppler_shifts is not None:
        warnings.warn(
            f"ambiguity_function: doppler_shifts= is the Doppler axis itself, "
            f"so n_doppler_shifts={n_doppler_shifts!r}, which builds one, is not used.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    doppler_shifts = np.atleast_1d(np.asarray(doppler_shifts, dtype=float))
    cells = int(doppler_shifts.size) * (2 * n - 1)
    if cells > _MAX_AMBIGUITY_CELLS:
        raise ConfigurationError(
            f"ambiguity_function: the surface would be "
            f"{doppler_shifts.size} x {2 * n - 1} = {cells} float64 cells "
            f"({cells * 8 / 2 ** 30:.2f} GiB), past the "
            f"{_MAX_AMBIGUITY_CELLS} cell cap "
            f"({_MAX_AMBIGUITY_CELLS * 8 / 2 ** 30:.2f} GiB). It also runs "
            f"one full-length FFT convolution per Doppler row. Shorten the "
            f"waveform, decimate it, or pass fewer n_doppler_shifts / doppler_shifts "
            f"candidates.")
    energy = float(np.sum(np.abs(s) ** 2))
    if energy == 0.0:
        raise ConfigurationError(
            "ambiguity_function: waveform carries no energy, so normalising "
            "the (0, 0) peak to 1 divides by zero and the whole surface is "
            "nan. Pass the transmit waveform.")
    rev = np.conj(s[::-1])
    amp = np.empty((doppler_shifts.size, 2 * n - 1))
    for i, nu in enumerate(doppler_shifts):
        sd = s * np.exp(2j * np.pi * nu * t)
        amp[i] = np.abs(fftconvolve(sd, rev, mode="full"))
    amp /= energy
    lags = (np.arange(2 * n - 1) - (n - 1)) / fs
    return AmbiguityResult(lags, doppler_shifts, amp)
