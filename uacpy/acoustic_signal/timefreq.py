"""Time-resolved views of a signal: when its content occurs.

:func:`spectrogram`, :func:`stft`, :func:`cwt` / :func:`inverse_cwt`,
:func:`wigner_ville`, the analytic signal and its :func:`envelope` and
:func:`instantaneous_frequency`, and the cepstra (:func:`cepstrum`,
:func:`complex_cepstrum` / :func:`inverse_complex_cepstrum`).
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections import namedtuple
from typing import Optional
import numpy as np
import scipy.signal as _sig
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning,
)
from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.core.acoustics import power_to_dB
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._validate import (
    normalize_axis, require_at_most_nyquist, require_finite_signal,
    require_positive_finite_scalar, require_real_signal,
)
from uacpy.acoustic_signal._results import POWER_UNITS, PlottedResult
from functools import lru_cache
from scipy.integrate import cumulative_trapezoid
from scipy.signal import hilbert
from scipy.special import gamma
from uacpy.acoustic_signal.spectral import (
    _require_bin_scaling, _time_axis_last, _warn_two_sided,
)
from uacpy.acoustic_signal.windows import _default_noverlap, _welch_nperseg


# ──────────────────────────────────────────────────────────────────────
# Time-resolved views
#
# What the record contains as a function of
# time as well as frequency: Hilbert, spectrogram, wavelet, Wigner-Ville,
# cepstrum.
# ──────────────────────────────────────────────────────────────────────


class WignerVilleResult(PlottedResult,
                        namedtuple("WignerVilleResult", "frequencies times distribution")):
    """Wigner-Ville distribution over ``frequencies`` and ``times``.

    The tuple is the measurement, so ``frequencies, times, distribution = ...``
    keeps working; :meth:`plot` is the one obvious way to draw it.
    """

    __slots__ = ()

    _plotter = "plot_wigner_ville"
    _plot_fields = ("frequencies", "times", "distribution")

    def _field_units(self):
        return {"frequencies": "Hz", "times": "s", "distribution": None}


class CWTResult(PlottedResult,
                namedtuple("CWTResult", "frequencies times coefficients")):
    """Continuous wavelet transform: the analysis ``frequencies``, the
    ``times`` of the samples, and the ``coefficients`` over both.

    The tuple is the measurement, so
    ``frequencies, times, coefficients = cwt(x, fs)`` unpacks it, in the
    order :class:`SpectrogramResult` uses; :meth:`plot` is the one obvious
    way to draw it. A multichannel record's ``coefficients`` carry its
    channel axes first, then ``(frequencies, times)``.

    The forward call's settings ride on attributes, which
    :func:`inverse_cwt` reads back so the result goes back whole:
    ``sample_rate`` (Hz), ``wavelet``, and the shape setting that wavelet
    reads — ``w0`` for ``'morlet'``, ``order`` for ``'paul'`` / ``'dog'``,
    the other ``None``. Filtered coefficients keep them through
    ``result._replace(coefficients=filtered)``.
    """

    _attrs = ("sample_rate", "wavelet", "w0", "order")
    _plotter = "plot_cwt"

    def __new__(cls, frequencies, times, coefficients, *, sample_rate,
                wavelet, w0=None, order=None):
        self = super().__new__(cls, frequencies, times, coefficients)
        self.sample_rate = float(sample_rate)
        self.wavelet = str(wavelet)
        self.w0 = None if w0 is None else float(w0)
        self.order = None if order is None else int(order)
        return self

    def _field_units(self):
        return {"frequencies": "Hz", "times": "s", "coefficients": None}

    def _declared_dims(self):
        # Channel axes lead, so a channel axis as long as the frequency or
        # time axis is not read as either.
        ndim = np.ndim(self.coefficients)
        return {"coefficients":
                tuple(f"coefficients_dim{i}" for i in range(ndim - 2))
                + ("frequencies", "times")}


#: Unit of the STFT coefficients, per scaling: they carry the square root of
#: the scaling's factor.
_STFT_UNITS = {"density": "Pa/√Hz", "spectrum": "Pa"}


class SpectrogramResult(PlottedResult,
                        namedtuple("SpectrogramResult",
                                   "frequencies times power")):
    """Spectrogram: ``frequencies``, ``times`` and the ``power`` panel.

    The tuple is the measurement, so ``frequencies, times, power = ...``
    keeps working; :meth:`plot` is the one obvious way to draw it. What the
    panel holds rides on the ``scaling`` attribute, as on
    :class:`SpectralEstimate`: ``'density'`` (Pa²/Hz) or ``'spectrum'``
    (Pa²). :meth:`level_dB` gives the panel as levels in dB re a reference
    pressure.
    """

    _attrs = ("scaling",)
    _plotter = "plot_spectrogram"
    _plot_fields = ("frequencies", "times", "power")
    _plot_defaults = ("scaling",)

    def __new__(cls, frequencies, times, power, scaling="density"):
        self = super().__new__(cls, frequencies, times, power)
        self.scaling = str(scaling)
        return self

    def level_dB(self, ref=REFERENCE_PRESSURE_WATER):
        """The panel as levels in dB re the reference pressure ``ref`` (Pa,
        1 µPa by default): :func:`~uacpy.core.acoustics.power_to_dB` of
        ``power``, in dB re 1 µPa²/Hz under ``scaling='density'`` and
        dB re 1 µPa² under ``'spectrum'`` — the levels :meth:`plot` draws.

        Parameters
        ----------
        ref : float, optional
            Reference pressure (Pa).

        Returns
        -------
        numpy.ndarray
            The levels, shaped as ``power``.
        """
        return power_to_dB(self.power, ref)

    def _field_units(self):
        return {"frequencies": "Hz", "times": "s",
                "power": POWER_UNITS[self.scaling]}


class STFTResult(PlottedResult,
                 namedtuple("STFTResult", "frequencies times coefficients")):
    """Short-time Fourier transform: ``frequencies``, ``times`` and the
    complex ``coefficients`` panel, one column per segment.

    The tuple is the measurement, so ``frequencies, times, coefficients =
    stft(x, fs)`` unpacks it. ``scaling`` rides as an attribute and names
    the coefficients' unit: the windowed transform scaled by the square root
    of the scaling's factor, Pa/√Hz under ``'density'`` and Pa under
    ``'spectrum'`` (a tone of amplitude A reads A/2), with no one-sided
    doubling. ``np.abs`` gives the magnitude and ``np.angle`` the phase in
    radians. :meth:`plot` draws ``|coefficients|²``, the two-sided density
    (or band power), with :func:`~uacpy.plot.plot_spectrogram`, which reads
    the panel kind from :attr:`mode`.
    """

    _attrs = ("scaling",)
    _plotter = "plot_spectrogram"
    _plot_fields = ("frequencies", "times", "coefficients")
    _plot_defaults = ("scaling", "mode")

    #: The panel kind :func:`~uacpy.plot.plot_spectrogram` draws: complex
    #: coefficients, drawn as their square.
    mode = "complex"

    def __new__(cls, frequencies, times, coefficients, scaling="density"):
        self = super().__new__(cls, frequencies, times, coefficients)
        self.scaling = str(scaling)
        return self

    def _field_units(self):
        return {"frequencies": "Hz", "times": "s",
                "coefficients": _STFT_UNITS[self.scaling]}


def _real_record(data, axis, who, why):
    """``(x, axis)``: ``data`` as a real ``float`` array with a time axis,
    and that axis as a non-negative index. Every other axis is a channel."""
    arr = np.asarray(data)
    if arr.ndim == 0:
        raise ConfigurationError(
            f"{who}: data must have a time axis; got a scalar.")
    x = require_real_signal(arr, who, ndim=arr.ndim, why=why)
    return x, normalize_axis(x, axis, who)


def analytic_signal(data, *, axis=-1):
    """Analytic signal ``data + j*Hilbert(data)`` of a real signal, along
    ``axis``; the output has the shape of ``data``.

    Parameters
    ----------
    data : array_like
        A real signal, time along ``axis``; every other axis is a channel.
    axis : int, optional
        The time axis. Default ``-1``.
    """
    return _analytic_signal(data, "analytic_signal", axis)


def _analytic_signal(data, who, axis=-1):
    """:func:`analytic_signal`, its refusals naming ``who`` — the public
    function the caller used."""
    xr, axis = _real_record(
        data, axis, who,
        why="; the analytic/Hilbert representation is only defined for a "
            "real signal.")
    return hilbert(xr, axis=axis)


def envelope(data, *, axis=-1):
    """Instantaneous amplitude envelope ``|analytic_signal(data)|``, along
    ``axis``; the output has the shape of ``data``.

    Parameters
    ----------
    data : array_like
        A real signal, time along ``axis``; every other axis is a channel.
    axis : int, optional
        The time axis. Default ``-1``.
    """
    return np.abs(_analytic_signal(data, "envelope", axis))


def instantaneous_frequency(data, sample_rate: float, *, axis=-1):
    """Instantaneous frequency (Hz) from the analytic-signal phase derivative.

    Returns an array of the shape of ``data`` (centred differences of the
    unwrapped phase via :func:`numpy.gradient` along ``axis``, time-aligned
    with ``data``).

    Parameters
    ----------
    data : array_like
        A real signal, time along ``axis``; every other axis is a channel.
    sample_rate : float
        Sample rate (Hz).
    axis : int, optional
        The time axis. Default ``-1``.
    """
    fs = require_positive_finite_scalar(
        sample_rate, "instantaneous_frequency", "sample_rate", " Hz")
    z = _analytic_signal(data, "instantaneous_frequency", axis)
    axis = normalize_axis(z, axis, "instantaneous_frequency")
    phase = np.unwrap(np.angle(z), axis=axis)
    return np.gradient(phase, axis=axis) / (2.0 * np.pi) * fs


def _smoothing_window(spec, name):
    """Centered, odd-length smoothing window from a ``None`` / int / array spec.

    ``None`` -> no smoothing. An int ``L`` -> a symmetric Hann window of
    length ``L``. A 1-D array is used verbatim. An even length, int or
    array, is refused: an even window has no centre sample to put the peak
    on. Returns ``(w, half)`` with ``w`` of length ``2*half+1`` centered at
    index ``half`` (so ``w[half+k]`` weights offset ``k``); ``(None, 0)`` for
    no smoothing.
    """
    if spec is None:
        return None, 0
    if np.isscalar(spec):
        L = int(spec)
        if L < 1:
            raise ConfigurationError(
                f"wigner_ville: {name} length must be >= 1; got {L}.")
        size = L
    else:
        w = np.asarray(spec, dtype=float)
        if w.ndim != 1 or w.size < 1:
            raise ConfigurationError(
                f"wigner_ville: {name} must be a 1-D window; "
                f"got shape {w.shape}.")
        size = w.size
    if size % 2 == 0:
        # The transform's .real relies on acc(-tau) = conj(acc(tau)), which
        # a window holds only with its peak on a centre sample: trimming an
        # even symmetric window (hann(6)[:-1]) puts the peak off-centre
        # (measured up to 59 % error at L = 4), and any other odd window made
        # from it would be a window the caller did not pass.
        raise ConfigurationError(
            f"wigner_ville: {name} is {size} samples long, an even length; "
            f"the smoothing window must have an odd length, centred on its "
            f"middle sample.",
            remediation=f"Pass a window of {size - 1} or {size + 1} samples"
                        + ("." if np.isscalar(spec)
                           else ", e.g. np.delete(w, w.size // 2) to drop "
                                "one of its two middle samples."))
    if np.isscalar(spec):
        w = _sig.get_window("hann", size, fftbins=False).astype(float)
    return w, w.size // 2


# Cap on a time-frequency surface, in bytes, checked before it is allocated:
# 1 GiB, the ceiling ``detect._MAX_AMBIGUITY_CELLS`` sets for its surface.
_MAX_SURFACE_BYTES = 1 << 30

# The (NF, n) float64 Wigner-Ville distribution under that cap: 2**27 cells.
_MAX_WIGNER_CELLS = _MAX_SURFACE_BYTES // 8


# Largest fraction of a complex signal's energy ``wigner_ville`` accepts at
# negative frequencies (-30 dB); more would draw visible ridges at fs/2 - f.
_WIGNER_NEGATIVE_ENERGY_MAX = 1e-3


def wigner_ville(data, sample_rate: float, *, analytic: Optional[bool] = None,
                 lag_smoothing=None, time_smoothing=None, nfft=None):
    """Discrete (smoothed-pseudo-) Wigner-Ville distribution of a 1-D signal.

    Returns a :class:`WignerVilleResult` ``(frequencies, times, distribution)``
    with the distribution real, shape ``(NF, n)``; ``f`` spans
    ``[0, fs/2)``. The kernel ``z(t+tau)z*(t-tau)`` doubles the apparent
    frequency, so the physical frequency axis is ``k*fs/(2*NF)``, and the
    discrete distribution is periodic in frequency with period ``fs/2``:
    content at ``-f`` lands at ``fs/2 - f``. The axis is therefore alias-free
    only for a signal whose spectrum sits in ``[0, fs/2)`` — the analytic
    signal of a real record, or a complex signal with no negative-frequency
    content.

    The Wigner-Ville distribution is Mallat, *A Wavelet Tour of Signal
    Processing*, §4.5.1, Eq. 4.108. Mallat's discrete computation (§4.5.4)
    interpolates the signal at half-integer samples; this integer-lag kernel
    keeps the samples and halves the alias-free band instead, as above.

    A quadratic energy distribution — there is no routine inverse (like a
    spectrogram, it maps a signal to a 2-D density, not reversibly).

    Parameters
    ----------
    data : 1-D array
        Real or complex signal. A complex signal carrying more than
        ``_WIGNER_NEGATIVE_ENERGY_MAX`` (0.1 %) of its energy at negative
        frequencies is refused: its ridges would be drawn at ``fs/2 - f``.
        Shift a complex baseband signal whose band sits in
        ``(-fs/4, fs/4)`` up by ``fs/4`` (``z * exp(2j*pi*(fs/4)*t)``) and
        subtract ``fs/4`` from the returned frequencies.
    sample_rate : float
        Sample rate (Hz).
    analytic : bool, optional
        Use the analytic signal for real input (unset: ``True``),
        suppressing cross-terms with the negative spectrum. ``False`` runs
        the raw signal, whose ``-f`` image draws a second ridge at
        ``fs/2 - f`` and whose ``±f`` cross-term draws a third at 0 Hz (a
        100 Hz cosine at ``fs = 1`` kHz shows ridges at 0, 100 and 400 Hz).
        Not read when ``data`` is already complex: a given one then gets a
        ``FallbackWarning``.
    lag_smoothing : None, int, or 1-D array
        Lag-domain smoothing window ``h(tau)`` — the *pseudo*-WVD. Smooths
        along frequency and limits the lag extent (shorter window -> more
        cross-term suppression, coarser frequency resolution). ``None`` is the
        full-lag WVD. An int ``L`` gives a symmetric Hann window of length
        ``L``. The window is centred on its middle sample, so an even
        length (int or array) is refused.
    time_smoothing : None, int, or 1-D array
        Time-domain smoothing window ``g`` — the *smoothed*-pseudo-WVD. Averages
        the instantaneous autocorrelation over neighbouring times (more
        cross-term suppression, coarser time resolution). ``None`` disables it.
        Odd lengths only, as ``lag_smoothing``.
    nfft : int, optional
        Zero-pad the lag FFT to ``nfft >= n`` bins (finer frequency spacing).
        ``None`` uses ``n``.

    Returns
    -------
    WignerVilleResult
        ``(frequencies, times, distribution)``: frequency axis (Hz), time axis
        (s), and the distribution ``(NF, n)`` — the same axis order as
        :class:`SpectrogramResult`.
    """
    xc = np.asarray(data)
    if xc.ndim != 1:
        raise ConfigurationError(
            f"wigner_ville: data must be 1-D; got shape {xc.shape}.")
    require_finite_signal(xc, "wigner_ville")
    if np.iscomplexobj(xc):
        # Negative-frequency energy fraction, measured twice and the smaller
        # kept: a real -f component shows in both. The rectangular DFT reads
        # an analytic signal's exact zero there but leaks an off-bin +f tone
        # into it (0.4 % for 100 Hz in 256 samples at 1 kHz); the Hann-
        # windowed DFT suppresses that leakage.
        bins = np.fft.fftfreq(xc.size)
        # The Nyquist bin (-0.5 for an even length) is both +fs/2 and -fs/2.
        neg = (bins < 0.0) & (bins > -0.5)
        fractions = []
        for taper in (np.ones(xc.size), np.hanning(xc.size)):
            spectrum = np.abs(np.fft.fft(xc * taper)) ** 2
            fractions.append(float(spectrum[neg].sum())
                             / max(float(spectrum.sum()), np.finfo(float).tiny))
        fraction = min(fractions)
        if fraction > _WIGNER_NEGATIVE_ENERGY_MAX:
            raise ConfigurationError(
                f"wigner_ville: {100.0 * fraction:.3g} % of the "
                f"complex signal's energy is at negative frequencies; the "
                f"discrete Wigner-Ville distribution is periodic in "
                f"frequency with period fs/2, so content at -f would be "
                f"drawn at fs/2 - f on the returned [0, fs/2) axis.",
                remediation="Shift a complex baseband signal whose band "
                            "sits in (-fs/4, fs/4) up by fs/4 (z * "
                            "exp(2j*pi*(fs/4)*t)) and subtract fs/4 from "
                            "the returned frequencies, or pass the real "
                            "signal and let analytic=True form the "
                            "analytic signal.")
        if analytic is not None:
            warnings.warn(
                f"wigner_ville: analytic={analytic!r} chooses how a REAL "
                f"record is formed, and data is already complex, so analytic "
                f"is not used.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        z = xc
    elif analytic is None or analytic:
        z = analytic_signal(xc)
    else:
        z = xc.astype(complex)
    n = z.size
    fs = require_positive_finite_scalar(sample_rate, "wigner_ville",
                                        "sample_rate", " Hz")
    NF = n if nfft is None else int(nfft)
    if NF < n:
        raise ConfigurationError(
            f"wigner_ville: nfft ({NF}) must be >= n ({n}) (zero-pad only)")
    hv, Lh = _smoothing_window(lag_smoothing, "lag_smoothing")
    gv, Lg = _smoothing_window(time_smoothing, "time_smoothing")
    lag_cap = n - 1 if hv is None else Lh
    cells = NF * n
    if cells > _MAX_WIGNER_CELLS:
        raise ConfigurationError(
            f"wigner_ville: the distribution would be {NF} x {n} = {cells} "
            f"float64 cells ({cells * 8 / 2 ** 30:.2f} GiB), past the "
            f"{_MAX_WIGNER_CELLS} cell cap "
            f"({_MAX_WIGNER_CELLS * 8 / 2 ** 30:.2f} GiB). Shorten or "
            f"decimate the record, or pass a smaller nfft; lag_smoothing "
            f"bounds the lags, not the surface.")
    W = np.zeros((NF, n))
    for ti in range(n):
        taumax = min(ti, n - 1 - ti, lag_cap)
        taus = np.arange(-taumax, taumax + 1)
        if gv is None:
            acc = z[ti + taus] * np.conj(z[ti - taus])
        else:
            # All lags at once over the full time-smoothing support
            # m in [-Lg, Lg]. Per lag tau the valid support is |m| <= mmax
            # with mmax = min(Lg, ti - |tau|, n - 1 - ti - |tau|) (both
            # z[ti+tau+m] and z[ti-tau+m] in range); out-of-support terms
            # are masked to zero, indices clipped so the masked positions
            # never index out of bounds.
            ms = np.arange(-Lg, Lg + 1)
            mmax = np.minimum(Lg, np.minimum(ti - np.abs(taus),
                                             n - 1 - ti - np.abs(taus)))
            valid = np.abs(ms)[None, :] <= mmax[:, None]
            ip = np.clip(ti + taus[:, None] + ms[None, :], 0, n - 1)
            im = np.clip(ti - taus[:, None] + ms[None, :], 0, n - 1)
            gw = gv[Lg + ms]
            prod = np.where(valid, gw[None, :] * z[ip] * np.conj(z[im]), 0.0)
            wsum = np.where(valid, gw[None, :], 0.0).sum(axis=1)
            acc = prod.sum(axis=1) / wsum
        if hv is not None:
            acc = acc * hv[Lh + taus]
        kernel = np.zeros(NF, dtype=complex)
        # Lag axis in FFT order: lag 0 at index 0, negative lags wrapped to the
        # top of the buffer, zeros in the unfilled middle (the zero-padding).
        kernel[(taus + NF) % NF] = acc
        # acc(-tau) = conj(acc(tau)) — a symmetric smoothing window preserves
        # that — so the transform is real and `.real` drops only rounding
        # error, not signal. An asymmetric `lag_smoothing` array breaks the
        # symmetry and the discarded imaginary part is then meaningful.
        W[:, ti] = np.real(np.fft.fft(kernel))
    f = np.arange(NF) * fs / (2.0 * NF)
    t = np.arange(n) / fs
    return WignerVilleResult(f, t, W)


#: The Morlet central (non-dimensional) frequency :func:`cwt` and
#: :func:`inverse_cwt` take when ``w0`` is unset: Torrence & Compo (1998),
#: Eq. 1, "here taken to be 6 to satisfy the admissibility condition".
_MORLET_W0 = 6.0


def _wavelet_settings(wavelet, w0, order, who):
    """``(w0, order)`` as the transform reads them: unset, ``w0`` is
    :data:`_MORLET_W0` and ``order`` 4 for ``'paul'``, 2 for ``'dog'``. The
    Morlet wavelet reads ``w0`` alone and Paul / DOG read ``order`` alone,
    so a given setting the chosen wavelet does not read gets a
    FallbackWarning naming it, and the transform runs as if it were unset."""
    if wavelet == "morlet" and order is not None:
        warnings.warn(
            f"{who}: wavelet='morlet' does not read order= (got {order!r}); "
            f"the Morlet wavelet is set by w0 alone.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        order = None
    if wavelet in ("paul", "dog") and w0 is not None:
        warnings.warn(
            f"{who}: wavelet={wavelet!r} does not read w0= (got {w0!r}); its "
            f"shape is set by order alone.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        w0 = None
    return (_MORLET_W0 if w0 is None else w0,
            (4 if wavelet == "paul" else 2) if order is None else order)


def _wavelet_fourier(wavelet, s, omega, w0, order):
    """Fourier-domain daughter wavelet ``psi_hat(s*omega)`` and the Fourier
    frequency<->scale factor ``f = factor*fs/s`` (Torrence & Compo 1998, Table 1).
    ``omega`` is in rad/sample."""
    so = s * omega
    pos = omega > 0.0
    if wavelet == "morlet":
        psi = (np.pi ** -0.25) * np.exp(-0.5 * (so - w0) ** 2) * pos
        factor = (w0 + np.sqrt(2.0 + w0 ** 2)) / (4.0 * np.pi)
    elif wavelet == "paul":
        m = order
        norm = 2.0 ** m / np.sqrt(m * math.factorial(2 * m - 1))
        psi = np.zeros(omega.shape, dtype=complex)
        psi[pos] = norm * (so[pos] ** m) * np.exp(-so[pos])
        factor = (2.0 * m + 1.0) / (4.0 * np.pi)
    elif wavelet == "dog":
        m = order
        norm = -(1j ** m) / np.sqrt(gamma(m + 0.5))
        psi = norm * (so ** m) * np.exp(-0.5 * so ** 2)  # real wavelet: all omega
        factor = np.sqrt(m + 0.5) / (2.0 * np.pi)
    else:
        raise ConfigurationError(
            f"cwt: unknown wavelet {wavelet!r}; choose 'morlet', 'paul', or 'dog'"
        )
    return psi, factor


@lru_cache(maxsize=None)
def _reconstruction_constants(wavelet, w0, order):
    """``(C_delta, psi0(0))`` for the Torrence & Compo (1998) eq.-11 inverse.

    ``psi0(0)``, the wavelet at zero lag, is ``(2*pi)**-0.5 * int psihat0(w) dw``.
    ``C_delta`` is the delta-function calibration of T&C eq. 13-14 taken in the
    continuous-scale limit: for a unit impulse, ``Re{W(s)}/sqrt(s)`` collapses to
    ``psi0(0)*R(pi*s)/s``, where ``R(a)`` is the fraction of the wavelet's
    spectral mass inside the discrete band ``|w| <= pi`` rad/sample. The
    calibration is therefore a one-dimensional quadrature over scale, and it is
    a property of the wavelet alone — not of the scale set the caller analysed
    with, so a band-limited scale set still reconstructs only its own band.

    Against T&C's Table 2 (whose caption calls those factors "empirically
    derived"): ``psi0(0)`` agrees to every tabulated digit for Morlet ``w0=6``,
    Paul ``m=4`` and DOG ``m=2``/``m=6``; ``C_delta`` agrees to 0.3 % for
    Morlet and Paul but differs by ~2 % for the DOG pair (0.776/1.132/3.541/
    1.966 tabulated against 0.778/1.133/3.616/1.929 here). Extends the table to
    any admissible ``w0`` / ``order``.
    """
    u_max = 60.0 + w0 + 6.0 * order
    u = np.linspace(-u_max, u_max, 100001)
    psi_hat, factor = _wavelet_fourier(wavelet, 1.0, u, w0, order)
    mass = cumulative_trapezoid(np.real(psi_hat), u, initial=0.0)
    total = mass[-1] - mass[0]
    psi0_zero = total / np.sqrt(2.0 * np.pi)
    if abs(psi0_zero) < 1e-9:
        raise ConfigurationError(
            f"inverse_cwt: the {wavelet!r} wavelet of order {order} is an odd "
            "function, so psi0(0) = 0 and the Torrence & Compo eq.-11 "
            "reconstruction is undefined for it. Use an even order.")
    dj = 0.02
    s_min = 2.0 * factor / 256.0            # well below the Nyquist-cut scale
    n_scale = int(np.log2(2.0e6 / s_min) / dj) + 1
    scales = s_min * 2.0 ** (np.arange(n_scale) * dj)
    a = np.minimum(np.pi * scales, u_max)
    in_band = (np.interp(a, u, mass) - np.interp(-a, u, mass)) / total
    return float(np.sum(in_band / scales) * dj), float(psi0_zero)


def cwt(data, sample_rate, frequencies=None, wavelet="morlet", *, w0=None,
        order=None, n_frequencies=64, axis=-1):
    """Continuous wavelet transform with a selectable wavelet (FFT-based).

    At each scale the signal is filtered by the chosen analysing wavelet
    (Mallat, *A Wavelet Tour of Signal Processing*, Ch. 4; scale<->frequency
    factors from Torrence & Compo 1998). Linear in time, logarithmic in
    frequency — well suited to dispersive / transient ocean-acoustic arrivals.

    Parameters
    ----------
    data : array_like
        Real signal, time along ``axis``; every other axis is a channel.
    sample_rate : float
        Sample rate (Hz).
    frequencies : array, optional
        Frequencies (Hz) to analyse. Default: ``n_frequencies`` (64)
        log-spaced points from ``4*fs/N`` (four cycles per record) to
        ``fs/2`` (uacpy guidance).
    wavelet : {'morlet', 'paul', 'dog'}
        Analysing wavelet. ``'morlet'`` (complex, best frequency resolution),
        ``'paul'`` (complex, best time resolution), ``'dog'`` (real Derivative
        Of Gaussian; ``order=2`` is the Mexican-hat / Ricker wavelet).
    w0 : float, optional
        Morlet central (non-dimensional) frequency; ``>= 5`` keeps it
        admissible. Unset, 6 (Torrence & Compo 1998, Eq. 1). ``'paul'`` and
        ``'dog'`` do not read it: a given one gets a ``FallbackWarning``.
    order : int, optional
        Wavelet order ``m``. Unset, 4 for ``'paul'``, 2 for ``'dog'``.
        ``'morlet'`` does not read it: a given one gets a
        ``FallbackWarning``.
    n_frequencies : int
        Number of log-spaced frequencies when ``frequencies`` is None.
    axis : int, optional
        The time axis of ``data``. Default ``-1``.

    Notes
    -----
    **Periodic borders.** The transform is one FFT of the record at its own
    length, so the record is treated as periodic — the convention Mallat
    adopts for finite signals ("we treat f[n] and the wavelets as periodic
    signals of period N", *A Wavelet Tour of Signal Processing*, sect. 4.3).
    Near each end the wavelet reaches round to the other end: a unit
    impulse five samples before the end of a 2048-sample record at 8 kHz
    reads 0.999, 0.992 and 0.812 of its own peak at ``t = 0`` at 50, 200 and
    1000 Hz. Read the edges of a scalogram over about one wavelet width —
    wider at low frequency — as border, and zero-pad the record yourself
    when a transient sits near an end.

    Returns
    -------
    CWTResult
        ``(frequencies, times, coefficients)``: the analysis frequencies
        (Hz), the time (s) of each sample, ``arange(len(data)) /
        sample_rate``, and the complex CWT coefficients, shape
        ``(n_frequencies, n_samples)`` for one record — the channel axes of
        a multichannel one first; ``abs(coefficients)`` is the scalogram.
    """
    xr, axis = _real_record(data, axis, "cwt",
                            why="; the transform analyses a real signal.")
    xr = np.moveaxis(xr, axis, -1)
    fs = require_positive_finite_scalar(sample_rate, "cwt", "sample_rate",
                                        " Hz")
    w0, order = _wavelet_settings(wavelet, w0, order, "cwt")
    n = xr.shape[-1]
    channels = xr.shape[:-1]
    if frequencies is None:
        f_lo = 4.0 * fs / n  # lowest default frequency = 4 cycles per record
        if f_lo >= fs / 2.0:
            raise ConfigurationError(
                f"cwt: signal too short (n={n}) for the default frequency "
                f"range — the lowest analysis frequency {f_lo:.1f} Hz already "
                f"exceeds Nyquist {fs / 2.0:.1f} Hz. Pass an explicit "
                "`frequencies=` below Nyquist, or use a longer signal.")
        frequencies = np.geomspace(f_lo, fs / 2.0, int(n_frequencies))
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if np.any(frequencies <= 0):
        bad = frequencies <= 0
        raise ConfigurationError(
            f"cwt: frequencies must be > 0; got {int(bad.sum())} value(s) "
            f"<= 0, first at index {int(np.argmax(bad))} "
            f"({frequencies[bad][0]:g} Hz)")
    require_at_most_nyquist(frequencies, fs, "cwt", "analysis frequencies",
                            "the sampled wavelet aliases and the coefficients "
                            "are numerical residue, not band content")
    # The scalogram is one complex128 row per frequency, as long as the
    # record, per channel.
    n_channels = int(np.prod(channels))
    nbytes = 16 * frequencies.size * n * n_channels
    if nbytes > _MAX_SURFACE_BYTES:
        per = "" if n_channels == 1 else f" x {n_channels} channels"
        raise ConfigurationError(
            f"cwt: the scalogram would be {frequencies.size} x {n} complex "
            f"cells{per} ({nbytes / 2 ** 30:.2f} GiB), past the "
            f"{_MAX_SURFACE_BYTES / 2 ** 30:.2f} GiB cap. Decimate or shorten "
            f"the record, or pass fewer n_frequencies / frequencies.")
    omega = 2.0 * np.pi * np.fft.fftfreq(n)  # rad/sample
    Xf = np.fft.fft(xr, axis=-1)
    # One scale gives the factor; scales follow from f = factor*fs/s.
    _, factor = _wavelet_fourier(wavelet, 1.0, omega, w0, order)
    scales = factor * fs / frequencies  # scale in samples
    W = np.empty(channels + (frequencies.size, n), dtype=complex)
    for i, s in enumerate(scales):
        psi_hat, _ = _wavelet_fourier(wavelet, s, omega, w0, order)
        # Torrence & Compo eq. 6 normalisation sqrt(2*pi*s/dt), with dt = 1
        # because `omega` is rad/sample: equal energy at every scale.
        psi_hat = np.sqrt(2.0 * np.pi * s) * psi_hat
        W[..., i, :] = np.fft.ifft(Xf * np.conj(psi_hat), axis=-1)
    morlet = wavelet == "morlet"
    return CWTResult(frequencies, np.arange(n) / fs, W, sample_rate=fs,
                     wavelet=wavelet, w0=w0 if morlet else None,
                     order=None if morlet else order)


def inverse_cwt(W):
    """Inverse CWT (Torrence & Compo 1998, eq. 11).

    ``x_n = dj/(C_delta*psi0(0)) * sum_j Re(W_n(s_j))/sqrt(s_j)`` with ``dj``
    the log2 scale spacing. The reconstruction constants ``C_delta`` and
    ``psi0(0)`` are derived for the wavelet order actually in use
    (:func:`_reconstruction_constants`), so non-default ``w0`` / ``order``
    reconstruct at the right amplitude. Amplitude is recovered to the
    accuracy of the scale coverage (a band-limited scale set reconstructs
    only the band it spans). The quadrature assumes the scale grid is
    **uniform in log2** — true of :func:`cwt`'s default log-spaced
    frequencies; a non-uniform grid (e.g. linearly spaced frequencies, or a
    single frequency) biases the amplitude and raises a ``NumericsWarning``.

    Parameters
    ----------
    W : CWTResult
        What :func:`cwt` returned, whole: its ``coefficients`` (any leading
        channel axes, then ``(n_frequencies, n_time)``) over its
        ``frequencies``, with the ``sample_rate``, ``wavelet``, ``w0`` and
        ``order`` the forward call used. Filtered coefficients go back as
        ``W._replace(coefficients=filtered)``.

    Returns
    -------
    ndarray
        The reconstructed signal, one per channel: the leading channel axes
        of ``coefficients``, then time.
    """
    if not isinstance(W, CWTResult):
        raise ConfigurationError(
            f"inverse_cwt: takes the CWTResult cwt returned, which carries "
            f"the sample rate and wavelet the inverse needs; got "
            f"{type(W).__name__}.",
            remediation="Pass the cwt result whole; for filtered "
                        "coefficients, result._replace(coefficients="
                        "filtered).")
    Wc = np.asarray(W.coefficients)
    frequencies = np.atleast_1d(np.asarray(W.frequencies, dtype=float))
    if Wc.ndim < 2 or Wc.shape[-2] != frequencies.size:
        raise ConfigurationError(
            "inverse_cwt: coefficients must be (..., n_frequencies, n_time) "
            f"matching frequencies; got coefficients shape {Wc.shape} and "
            f"{frequencies.size} frequencies.")
    wavelet = W.wavelet
    w0, order = _wavelet_settings(wavelet, W.w0, W.order, "inverse_cwt")
    fs = require_positive_finite_scalar(W.sample_rate, "inverse_cwt",
                                        "sample_rate", " Hz")
    _, factor = _wavelet_fourier(wavelet, 1.0, np.array([1.0]), w0, order)
    scales = factor * fs / frequencies
    if scales.size > 1:
        dlog = np.abs(np.diff(np.log2(scales)))
        dj = float(np.mean(dlog))
        # Eq. 11 is a quadrature over log2-uniform scales with spacing dj; on
        # a non-uniform grid a single mean dj misweights every scale and the
        # amplitude comes out wrong.
        if float(np.std(dlog)) > 0.01 * dj:
            warnings.warn(
                "inverse_cwt: the scale grid implied by `frequencies` is not "
                "uniform in log2 (std(diff(log2(scales))) = "
                f"{float(np.std(dlog)):.3g} vs mean {dj:.3g}), but the "
                "Torrence & Compo eq.-11 sum assumes log2-uniform scale "
                "spacing dj — the reconstruction amplitude will be biased. "
                "Use log-spaced frequencies (cwt's default grid).",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    else:
        dj = 1.0
        warnings.warn(
            "inverse_cwt: a single scale cannot calibrate the log2 scale "
            "spacing dj the Torrence & Compo eq.-11 sum assumes (dj is taken "
            "as 1.0), so the reconstruction amplitude is arbitrary. Analyse "
            "with several log-spaced frequencies.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    c_delta, psi0_zero = _reconstruction_constants(wavelet, float(w0), int(order))
    return (np.sum(np.real(Wc) / np.sqrt(scales)[:, None], axis=-2)
            * dj / (c_delta * psi0_zero))


def _apply_lifter(c, lifter):
    """Quefrency-domain liftering of a real cepstrum ``c`` (length ``NF``).

    ``lifter`` is an int cutoff or a 1-D weight array. A positive int ``L``
    short-passes (keeps ``|quefrency| <= L`` -> spectral envelope); a negative
    int long-passes (zeros ``|quefrency| <= |L|`` -> excitation / echo). An
    array multiplies the cepstrum element-wise (symmetric weighting is on you).
    """
    nf = c.shape[-1]
    if np.isscalar(lifter):
        L = int(lifter)
        w = np.zeros(nf)
        keep = abs(L)
        w[:keep + 1] = 1.0
        if keep:
            # Negative quefrencies live at the top of the buffer. max(1, ...)
            # is what makes an over-long cutoff (keep >= nf) keep everything:
            # a bare nf - keep would go negative and w[-2:] would set two
            # elements instead of the whole tail.
            w[max(1, nf - keep):] = 1.0
        if L < 0:
            w = 1.0 - w
        return c * w
    w = np.asarray(lifter, dtype=float)
    if w.shape != (nf,):
        raise ConfigurationError(
            f"cepstrum: lifter array {w.shape} must be one weight per "
            f"quefrency, ({nf},).")
    return c * w


class Cepstrum(PlottedResult,
               namedtuple("Cepstrum", "quefrencies cepstrum")):
    """A real cepstrum and its quefrency axis.

    The tuple is the measurement, so ``quefrencies, c = cepstrum(x)``
    unpacks it. ``sample_rate`` rides as an attribute: given, the
    quefrencies are in seconds (``arange(n) / sample_rate``); ``None``, in
    samples (``arange(n)``). It survives pickling and copying and takes no
    part in equality. A multichannel record's ``cepstrum`` carries its
    channel axes first and the quefrencies last.
    """

    _attrs = ("sample_rate",)
    _plotter = "plot_cepstrum"

    def __new__(cls, quefrencies, cepstrum, *, sample_rate=None):
        self = super().__new__(cls, quefrencies, cepstrum)
        self.sample_rate = sample_rate
        return self

    def _field_units(self):
        return {"quefrencies": "samples" if self.sample_rate is None else "s",
                "cepstrum": ""}

    def _declared_dims(self):
        ndim = np.ndim(self.cepstrum)
        return {"cepstrum": tuple(f"cepstrum_dim{i}" for i in range(ndim - 1))
                + ("quefrencies",)}


def cepstrum(data, sample_rate=None, *, window="boxcar", nfft=None,
             lifter=None, axis=-1):
    """Real cepstrum ``irfft(log|rfft(data)|)``, as a :class:`Cepstrum`.

    Not invertible: discards phase. Use :func:`complex_cepstrum` /
    :func:`inverse_complex_cepstrum` for a reversible homomorphic transform.

    Parameters
    ----------
    data : array_like
        Real signal, time along ``axis``; every other axis is a channel.
    sample_rate : float, optional
        Sample rate (Hz) of ``data``. Given, the quefrencies are in seconds;
        ``None``, in samples.
    window : str or tuple, optional
        :func:`scipy.signal.get_window` spec applied before the FFT to curb
        spectral leakage. Default ``'boxcar'``, no taper, which ``None``
        also selects.
    nfft : int, optional
        Zero-pad the FFT to ``nfft >= n_samples`` bins (finer quefrency
        spacing). ``None`` uses the record length.
    lifter : None, int, or 1-D array
        Quefrency liftering — see :func:`_apply_lifter`. ``None`` returns the
        raw cepstrum; a positive int keeps low quefrencies (spectral envelope),
        a negative int keeps high quefrencies (pitch / echo structure).
    axis : int, optional
        The time axis of ``data``. Default ``-1``.

    Returns
    -------
    Cepstrum
        ``(quefrencies, cepstrum)``, both of length ``nfft`` along the
        quefrency axis; a multichannel ``cepstrum`` carries the channel axes
        first and the quefrencies last.
    """
    xr, axis = _real_record(
        data, axis, "cepstrum",
        why="; the real cepstrum irfft(log|rfft(x)|) is defined for a real "
            "signal. For a complex spectrum use complex_cepstrum.")
    xr = np.moveaxis(xr, axis, -1)
    n = xr.shape[-1]
    NF = n if nfft is None else int(nfft)
    if NF < n:
        raise ConfigurationError(
            f"cepstrum: nfft ({NF}) must be >= the record length ({n}) "
            f"(zero-pad only)")
    if window is None:
        window = "boxcar"
    xr = xr * _sig.get_window(window, n, fftbins=True).astype(float)
    spectrum = np.abs(np.fft.rfft(xr, n=NF, axis=-1))
    spectrum = np.maximum(spectrum, np.finfo(float).tiny)
    c = np.fft.irfft(np.log(spectrum), n=NF, axis=-1)
    if lifter is not None:
        c = _apply_lifter(c, lifter)
    if sample_rate is None:
        return Cepstrum(np.arange(NF, dtype=float), c)
    rate = require_positive_finite_scalar(sample_rate, "cepstrum",
                                          "sample_rate", " Hz")
    return Cepstrum(np.arange(NF) / rate, c, sample_rate=rate)


class ComplexCepstrum(PlottedResult,
                      namedtuple("ComplexCepstrum", "cepstrum delay")):
    """Complex cepstrum and the ``delay`` its phase unwrapping removed.

    The tuple is the measurement, so ``cepstrum, delay = ...``
    keeps working; :meth:`plot` is the one obvious way to draw it.

    ``sample_rate`` rides as an attribute and :attr:`quefrencies` is the
    axis of ``cepstrum``: ``arange(n) / sample_rate`` seconds when the rate
    was given, ``arange(n)`` samples otherwise, as :func:`cepstrum`'s are.
    """

    _attrs = ("sample_rate",)
    _plotter = "plot_cepstrum"
    _plot_fields = ("cepstrum",)
    _plot_defaults = ("sample_rate",)

    def __new__(cls, cepstrum, delay, *, sample_rate=None):
        self = super().__new__(cls, cepstrum, delay)
        self.sample_rate = sample_rate
        return self

    @property
    def quefrencies(self):
        """The quefrency of each entry of ``cepstrum``: seconds when
        ``sample_rate`` is set, samples otherwise."""
        n = np.size(self.cepstrum)
        return (np.arange(n, dtype=float) if self.sample_rate is None
                else np.arange(n) / float(self.sample_rate))

    def _field_units(self):
        return {"cepstrum": "", "delay": "samples"}

    def _declared_dims(self):
        return {"cepstrum": ("quefrencies",)}

    def _axis_coords(self):
        return {"quefrencies": (
            self.quefrencies,
            "samples" if self.sample_rate is None else "s")}


def complex_cepstrum(data, sample_rate=None):
    """Complex cepstrum with the linear-phase (rotation) term removed.

    Returns ``ComplexCepstrum(cepstrum, delay)``. ``delay`` is the integer
    number of samples of linear phase taken out: the unwrapped phase at the
    half-length bin over pi, rounded. It moves with the signal (shifting
    ``data`` by d samples adds d) but it is not the signal's onset, since the
    phase of every zero of the transform outside the unit circle is in it
    too; a band-limited burst starting at sample 500 can report 601.
    :func:`inverse_complex_cepstrum` needs it to reconstruct ``data``. Where
    the spectrum has fallen to the numerical floor before that bin, the
    unwrapped phase there is rounding noise and ``delay`` is arbitrary (the
    caveat MATLAB's ``cceps`` carries).

    Without the removal the unwrapped phase carries a ramp whose inverse
    transform is a ``1/q`` tail that swamps the echo structure the cepstrum
    exists to show, and makes the result depend on the signal's absolute
    arrival time rather than on its echo delays.

    The cepstrum stays **complex**: unwrapping breaks the Hermitian symmetry
    of ``log(fft(x))``, and the imaginary part is what makes the homomorphic
    transform reversible. Its quefrency axis is the result's
    :attr:`~ComplexCepstrum.quefrencies`: ``np.arange(len(data)) /
    sample_rate`` seconds when ``sample_rate`` is given, samples otherwise.

    Parameters
    ----------
    data : array_like
        A real 1-D signal.
    sample_rate : float, optional
        Sample rate (Hz) of ``data``, for the quefrency axis in seconds.
    """
    xr = require_real_signal(
        data, "complex_cepstrum",
        why="; the homomorphic cepstrum is defined for a real signal.")
    spectrum = np.fft.fft(xr)
    n = xr.size
    mag = np.maximum(np.abs(spectrum), np.finfo(float).tiny)
    phase = np.unwrap(np.angle(spectrum))
    # Remove the linear-phase term: round the end-to-end ramp to a whole
    # number of samples and subtract it, so the residual phase carries only
    # the echo structure.
    # A delay of d samples is the phase ramp -2*pi*d*k/n, so the ramp's
    # coefficient is -d: negate it, and ``delay`` is the delay in samples
    # (positive for a late signal), as the docstring says.
    delay = -int(np.round(phase[n // 2] * n / (2.0 * np.pi * (n // 2)))) if n > 1 else 0
    phase = phase + 2.0 * np.pi * delay * np.arange(n) / n
    log_spectrum = np.log(mag) + 1j * phase
    if sample_rate is not None:
        sample_rate = require_positive_finite_scalar(
            sample_rate, "complex_cepstrum", "sample_rate", " Hz")
    return ComplexCepstrum(np.fft.ifft(log_spectrum), delay,
                           sample_rate=sample_rate)


def inverse_complex_cepstrum(c):
    """Invert :func:`complex_cepstrum`: ``x = real(ifft(exp(fft(c))))``.

    Takes the :class:`ComplexCepstrum` namedtuple :func:`complex_cepstrum`
    returns (the imaginary part of its ``cepstrum`` field is significant —
    see there) and reconstructs the real signal ``x``. Both fields are
    required: the ``delay`` restores the linear-phase term the forward
    transform removed, without which the signal comes back at the wrong
    arrival time.

    Parameters
    ----------
    c : ComplexCepstrum
        What :func:`complex_cepstrum` returned, both fields.
    """
    if not isinstance(c, ComplexCepstrum):
        raise ConfigurationError(
            "inverse_complex_cepstrum: c must be the ComplexCepstrum "
            "namedtuple returned by complex_cepstrum — its delay field "
            "restores the linear-phase term the forward transform removed, "
            "and a bare cepstrum array carries no delay. Pass the "
            "complex_cepstrum result unchanged, or "
            "ComplexCepstrum(cepstrum=your_array, delay=your_delay). "
            f"Got {type(c).__name__}.")
    c, delay = c.cepstrum, int(c.delay)
    cr = np.asarray(c, dtype=complex)
    if cr.ndim != 1:
        raise ConfigurationError(
            f"inverse_complex_cepstrum: c must be 1-D; got shape {cr.shape}.")
    n = cr.size
    log_spectrum = np.fft.fft(cr)
    # Restore the linear-phase term complex_cepstrum took out.
    log_spectrum = log_spectrum - 1j * 2.0 * np.pi * delay * np.arange(n) / n
    return np.real(np.fft.ifft(np.exp(log_spectrum)))


#: The segment taper :func:`spectrogram` and :func:`stft` take when
#: ``window`` is ``None``: hann under both scalings (see :func:`spectrogram`).
_SHORT_TIME_WINDOW = "hann"


def _short_time_panel(data, sample_rate, *, window, nperseg, noverlap, nfft,
                      detrend, scaling, mode, axis, who):
    """``(f, t, panel)`` of :func:`scipy.signal.spectrogram` in ``mode``
    (``'psd'`` or ``'complex'``), after the checks :func:`spectrogram` and
    :func:`stft` share; refusals name ``who``."""
    data = _time_axis_last(data, axis, who)
    data = require_finite_signal(data, who)
    _warn_two_sided(who, data)
    sample_rate = require_positive_finite_scalar(
        sample_rate, who, "sample_rate", " Hz")
    if window is None:
        window = _SHORT_TIME_WINDOW
    _require_bin_scaling(scaling, who)
    # scipy rejects these untyped.
    if nperseg is not None and (
            isinstance(nperseg, bool)
            or not isinstance(nperseg, numbers.Integral) or nperseg < 1):
        raise ConfigurationError(
            f"{who}: nperseg must be an integer number of samples of "
            f"at least 1, or None for the default; got {nperseg!r}.")
    nperseg = _welch_nperseg(nperseg, window, np.shape(data)[-1], who)
    if noverlap is None:
        noverlap = _default_noverlap(window, nperseg)
    return _sig.spectrogram(data, sample_rate, window=window,
                            nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                            detrend=detrend, scaling=scaling, mode=mode)


def spectrogram(data, sample_rate, *, window=_SHORT_TIME_WINDOW,
                nperseg=None, noverlap=None, nfft=None,
                detrend="constant", scaling="density", axis=None):
    """Short-time power spectral density. Returns a
    :class:`SpectrogramResult` ``(frequencies, times, power)``, in Pa²/Hz
    under the default ``scaling='density'`` and Pa² per bin under
    ``'spectrum'``.

    ``window='hann'`` under both scalings, where :func:`welch` takes flat-top
    for ``scaling='spectrum'``: a spectrogram is read for where the energy
    sits in time and frequency, which hann's main lobe resolves (its
    noise-equivalent bandwidth is 1.50 bins against flat-top's 3.77); a tone
    half a bin off centre reads 1.42 dB low under it, so read tone levels
    from ``welch(scaling='spectrum')``.

    ``nperseg=None`` (default) takes 8192 samples (uacpy guidance), or the
    whole record when it is shorter, or an array ``window``'s own length, as
    on :func:`welch`; a given ``nperseg`` is used as given, and refused when
    longer than the record. ``noverlap=None`` (default) is the window's
    overlap on that segment, as on :func:`welch` (half a segment for hann);
    pass an int to override. ``nfft`` (zero-pad length) and ``detrend`` mirror
    :func:`uacpy.acoustic_signal.welch`: ``detrend`` is ``'constant'``
    (subtract each segment's mean, scipy's default), ``'linear'``, a
    callable, or ``False``. A segment holding part of a pulse has a mean the
    pulse does not have, and subtracting it adds a step across the segment
    that puts energy into the lowest bins; pass ``detrend=False`` for pulses
    and model time series. The complex short-time transform itself (its
    magnitude, its phase) is :func:`stft`'s. For logarithmic / constant-Q
    frequency resolution, see
    :func:`uacpy.acoustic_signal.constant_q_spectrogram`.

    **Complex input** is accepted, unlike :func:`cwt` or
    :func:`analytic_signal`, but returns a two-sided spectrum on an unsorted
    frequency axis (``0 .. fs/2`` then ``-fs/2 .. 0``) rather than the
    one-sided density above; a ``ValidityWarning`` says so.

    ``axis`` is the time axis of a multichannel ``data``, as on
    :func:`welch`: ``power`` then carries ``(..., frequency, time)``. The
    result records ``.scaling`` so its plot labels the unit it holds.

    Parameters
    ----------
    data : array_like
        Pressure record (Pa) with time along ``axis``.
    sample_rate : float
        Sample rate (Hz).
    window : str, tuple or array_like, optional
        Segment taper: a :func:`scipy.signal.get_window` name or
        ``(name, *params)`` tuple, or an array. Default ``'hann'``, which
        ``None`` also selects; ``'boxcar'`` is no taper.
    nperseg : int, optional
        Segment length, an integer number of samples. Unset: 8192 (uacpy
        guidance), or the whole record when shorter (see above).
    noverlap, nfft, detrend : optional
        Overlap, zero-pad length and per-segment detrending (see above).
    scaling : {'density', 'spectrum'}, optional
        Pa²/Hz or Pa² per bin. Default ``'density'``.
    axis : int, optional
        The time axis of a multichannel ``data``, as on :func:`welch`.
    """
    f, t, Sxx = _short_time_panel(
        data, sample_rate, window=window, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, detrend=detrend, scaling=scaling, mode="psd", axis=axis,
        who="spectrogram")
    return SpectrogramResult(f, t, Sxx, scaling=scaling)


def stft(data, sample_rate, *, window=_SHORT_TIME_WINDOW,
         nperseg=None, noverlap=None, nfft=None,
         detrend="constant", scaling="density", axis=None):
    """Short-time Fourier transform on :func:`spectrogram`'s segments.
    Returns an :class:`STFTResult` ``(frequencies, times, coefficients)``.

    The coefficients are the windowed transform of each segment
    (:func:`scipy.signal.spectrogram` with ``mode='complex'``), scaled by
    the square root of the scaling's factor: Pa/√Hz under the default
    ``scaling='density'``, Pa under ``'spectrum'``, where a tone of
    amplitude A reads A/2. They carry no one-sided doubling, so
    ``|coefficients|²`` is the two-sided density — half
    :func:`spectrogram`'s ``power`` off DC and Nyquist. ``np.abs`` gives the
    magnitude and ``np.angle`` the phase (radians).

    Every parameter is :func:`spectrogram`'s, with the same defaults and the
    same meaning: ``window`` (``'hann'``, which ``None`` also selects;
    ``'boxcar'`` is no taper), ``nperseg`` (unset: 8192 samples, uacpy
    guidance, or the whole record when shorter; a given one longer than the
    record is refused), ``noverlap`` (the window's overlap), ``nfft``, ``detrend``
    (``'constant'``; pass ``False`` for pulses), ``scaling`` and ``axis``
    (the time axis of a multichannel ``data``: ``coefficients`` then carries
    ``(..., frequency, time)``).

    Parameters
    ----------
    data : array_like
        Pressure record (Pa) with time along ``axis``.
    sample_rate : float
        Sample rate (Hz).
    window, nperseg, noverlap, nfft, detrend, scaling, axis : optional
        As on :func:`spectrogram`.
    """
    f, t, Z = _short_time_panel(
        data, sample_rate, window=window, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, detrend=detrend, scaling=scaling, mode="complex",
        axis=axis, who="stft")
    return STFTResult(f, t, Z, scaling=scaling)
