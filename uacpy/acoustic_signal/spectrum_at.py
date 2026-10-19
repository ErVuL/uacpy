"""A spectrum evaluated at the frequencies asked for.

:func:`tone_phasor` returns the amplitude and phase of ONE tone in a
record, and :func:`waveform_spectrum_at` a whole spectrum at arbitrary
frequencies. Both evaluate the transform **at** the frequency rather than
sampling the nearest DFT bin, which is what makes them exact off the
record's own grid.
"""

from __future__ import annotations

import warnings
from typing import Optional
import numpy as np
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core._validate import (
    normalize_axis, require_finite_signal, require_positive_finite_scalar,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.acoustic_signal.windows import _taper
from uacpy.core._fourier import _BLOCK_ELEMENTS, dtft_at


# ──────────────────────────────────────────────────────────────────────
# One tone, and a spectrum where you ask for it
#
# The phasor of a single tone in a record, and a whole spectrum
# evaluated AT the frequencies asked for rather than at DFT bins.
# ──────────────────────────────────────────────────────────────────────


#: The window :func:`tone_phasor` takes when ``window`` is ``None``.
_TONE_WINDOW = 'hann'

#: How far ``|ρ|`` may sit below 1 and still count as 1: the image coincides
#: with the tone (0 Hz, or Nyquist on a uniform grid) and only the in-phase
#: part is observable. A rounding margin (uacpy guidance): ``|ρ| = 1`` holds
#: exactly there in exact arithmetic, and a floating-point sum of ``N`` unit
#: phasors lands within a few ULP of it.
_COINCIDENT_IMAGE_TOL = 1e-9


def tone_phasor(data, times, frequency, *, window: str = _TONE_WINDOW,
                axis: int = -1):
    """Complex amplitude and phase of one tone in a record: the weighted
    least-squares fit of ``Re(A·e^{2πift})`` at the known ``frequency``.

    For a complex record (baseband or analytic) the fit is the windowed
    transform evaluated **at** ``frequency``,

    .. math::
        A = \\frac{1}{\\sum w}\\sum_n x_n w_n e^{-2\\pi i f t_n},

    since ``A·e^{2πift}`` puts all of ``A`` at ``+f``; an analytic signal
    ``data + i·H{data}`` returns the same phasor as its real part, and
    ``frequency`` may be negative.

    For a real record the tone is ``(A·e^{2πift} + A*·e^{-2πift})/2``, so
    twice that transform, ``E = 2·X/Σw``, is ``A + A*·ρ`` with
    ``ρ = Σ w e^{-4πift} / Σ w`` the weight of the negative-frequency image.
    Solving that pair of equations, ``A = (E − E*·ρ)/(1 − |ρ|²)``, is the
    least-squares fit of ``α₁cos + α₂sin`` at the known frequency, weighted
    by the window: Kay, *Fundamentals of Statistical Signal Processing*
    Vol. III, Algorithm 9.2 (``(HᵀH)⁻¹Hᵀx``), whose approximation
    ``(2/N)·Σ x e^{-j2πf0n}`` "is recognized as the Fourier transform of the
    data", and of which Algorithm 9.3 says the matrix form is needed "to
    avoid the 'interference' for the complex components of the real
    sinusoid at frequencies near 0 or 1/2". Away from 0 Hz and Nyquist
    ``|ρ|`` is negligible and ``A`` is ``E``; near them it is not — under
    Hann on a 1 s record (1 Hz bins), ``|ρ|`` is 0.85 at 0.25 Hz, 0.50 at
    0.5 Hz, 0.17 at 0.75 Hz, 0.024 at 1.25 Hz and 0.008 at 1.75 Hz
    (measured), and the same distance below Nyquist gives the same. At
    0 Hz, and at Nyquist on a uniform grid, ``|ρ| = 1``: the image coincides
    with the tone and only its in-phase part is observable, so ``A = E/2``
    (the in-phase part, quadrature zero) with a ``NumericsWarning`` saying
    so.

    The transform is evaluated **at** ``frequency``, not sampled at the
    nearest DFT bin. Off a bin, ``X[k]`` is a leakage sample of the window
    transform — neither the phasor at ``frequency`` nor the one at
    ``freqs[k]``. A model-produced trace picks its own ``nt`` and ``fs``,
    so the frequency of interest is essentially never on a bin, and the
    nearest-bin answer is wrong by a growing amount across the bin:
    measured against this sum, ``-0.056 dB`` and ``18°`` at a tenth of a
    bin, ``-1.418 dB`` and ``89.8°`` at half of one. **The phase reaches
    90° before the level has moved 1.5 dB**, which is why a level check
    alone does not find it.

    Parameters
    ----------
    data : array_like
        The record. ``axis`` is time; every other axis is carried through.
    times : array_like
        Time of each sample (s), one per sample along ``axis``. Passed
        rather than derived, so a record that does not start at zero
        carries its own offset into the phase.
    frequency : float
        The tone (Hz).
    window : {'hann', 'hamming', 'blackman', 'tukey', 'boxcar'}, optional
        Window across the record, the weights of the fit. Default
        ``'hann'``, which ``None`` also selects; ``'boxcar'`` is the
        unweighted fit.
    axis : int, default -1
        Time axis.

    Returns
    -------
    ndarray or complex
        The phasor, with ``axis`` removed.
    """
    return _tone_phasor(data, times, frequency, window=window, axis=axis)


def _tone_phasor(data, times, frequency, *, window: str = _TONE_WINDOW,
                axis: int = -1, who: str = "tone_phasor",
                steady_tone: bool = True):
    """:func:`tone_phasor` reporting its refusals and notices as ``who``.

    ``steady_tone=False`` returns the windowed transform ``g·X/Σw``
    (``g = 2`` for a real record) without solving for a negative-frequency
    image, for a caller whose record is not a steady tone — a transient
    whose DTFT is taken for a convolution-theorem ratio has no image to
    solve for."""
    data = np.asarray(data)
    t = np.asarray(times, dtype=float).ravel()
    if data.ndim == 0:
        raise ConfigurationError(f"{who}: data must have a time axis.")
    axis = normalize_axis(data, axis, who)
    if t.size != data.shape[axis]:
        raise ConfigurationError(
            f"{who}: times has {t.size} samples but axis {axis} of data has "
            f"{data.shape[axis]}.")
    frequency = float(frequency)
    if not np.isfinite(frequency):
        raise ConfigurationError(
            f"{who}: frequency must be finite (Hz); got {frequency!r}.")
    win = _taper(_TONE_WINDOW if window is None else window, t.size, who=who)
    weight = np.sum(win)
    # The one DTFT kernel (core/_fourier), at the frequency itself.
    transform = dtft_at(data, t, frequency, window=win, axis=axis) / weight
    if np.iscomplexobj(data):
        return transform
    estimate = 2.0 * transform
    if not steady_tone:
        return estimate
    rho = complex(dtft_at(win, t, 2.0 * frequency) / weight)
    if abs(rho) > 1.0 - _COINCIDENT_IMAGE_TOL:
        warnings.warn(
            f"{who}: at {frequency:g} Hz a real tone's negative-frequency "
            f"image coincides with it (0 Hz, or Nyquist on a uniform grid), "
            f"so only its in-phase part is observable; the phasor returned "
            f"is that part, with no quadrature part.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return estimate / 2.0
    return (estimate - np.conj(estimate) * rho) / (1.0 - abs(rho) ** 2)


def _chirp_step(freqs: np.ndarray, n: int, sample_rate: float) -> Optional[float]:
    """The step of a uniform ascending ``freqs``, or ``None`` if it has none.

    "Uniform enough" is not a matter of taste here. The chirp-z transform walks
    the contour ``f[0] + k*df``, so what has to hold is that the walk LANDS on
    the frequencies asked for: a landing error of ``eps`` Hz costs at most
    ``2*pi*eps*n/sample_rate`` radians of phase in the DTFT below, which the bound here
    holds under 1e-9 rad — two decades inside the transform's own agreement
    with the dense sum. Fewer than 2 in-band frequencies has no step to find
    (and the dense sum is one row there anyway).
    """
    m = freqs.size
    if m < 2:
        return None
    df = float(freqs[-1] - freqs[0]) / (m - 1)
    if not (df > 0.0):
        return None
    drift = float(np.max(np.abs(freqs[0] + df * np.arange(m) - freqs)))
    return df if drift * max(n, 1) <= 1e-10 * sample_rate else None




def waveform_spectrum_at(
    waveform: np.ndarray, sample_rate: float, frequencies: np.ndarray,
) -> np.ndarray:
    """Spectrum ``S(f)`` of a sampled waveform, at arbitrary frequencies.

    The vector counterpart of :func:`tone_phasor`: that one gives the
    phasor of a single tone in a record, this one gives the waveform's own
    spectrum wherever you ask for it. Both evaluate the transform AT the
    frequency rather than sampling a DFT bin.

    Evaluates the DTFT of the sampled waveform directly::

        S(f) = (1/fs) * sum_n w[n] exp(-2 pi i f n / fs)

    which reproduces ``rfft(w)/fs`` exactly on the waveform's own DFT grid and
    stays exact off it. Interpolating the rfft samples instead is only correct
    when the two grids coincide: linear interpolation is a convolution with a
    triangular kernel in frequency, i.e. a ``sinc^2(pi df_src t)`` taper
    anchored at ``t = 0`` plus periodisation at ``1/df_src`` in time. On a
    half-bin-offset grid that is a >100% median error in ``S(f)``.

    A real waveform is evaluated on ``[0, fs/2]``; frequencies outside it
    return 0 — a band-limited source carries no out-of-band energy, the DTFT
    would alias there, and a real waveform's negative-frequency spectrum is
    the conjugate of the positive one. A complex (baseband or analytic)
    waveform has independent content at ``-f`` and ``+f``, so it is evaluated
    on ``[-fs/2, fs/2]`` and returns 0 outside that.

    Parameters
    ----------
    waveform : array_like
        The sampled waveform, real or complex, 1-D, non-empty and finite.
    sample_rate : float
        Sample rate (Hz), finite and > 0.
    frequencies : array_like
        Frequencies (Hz) to evaluate at, finite.

    Returns
    -------
    ndarray
        Complex ``S(f)``, one entry per ``frequencies``.
    """
    who = "waveform_spectrum_at"
    fs = require_positive_finite_scalar(sample_rate, who, "sample_rate",
                                        " Hz")
    wf = require_finite_signal(waveform, who, "waveform")
    if wf.ndim > 1:
        raise ConfigurationError(
            f"{who}: waveform must be 1-D; got shape {wf.shape}. Flattening "
            f"it would evaluate the spectrum of {wf.size} samples end to end "
            f"as one waveform. Pick the channel or trace you meant, e.g. "
            f"waveform[0].")
    wf = np.atleast_1d(wf)
    wf = wf.astype(np.complex128 if np.iscomplexobj(wf) else np.float64)
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if not np.all(np.isfinite(frequencies)):
        raise ConfigurationError(
            f"{who}: frequencies must be finite (Hz); got "
            f"{int(np.count_nonzero(~np.isfinite(frequencies)))} non-finite "
            f"value(s).")
    n = wf.size

    out = np.zeros(frequencies.size, dtype=np.complex128)
    freq_min = -0.5 * fs if np.iscomplexobj(wf) else 0.0
    in_band = (frequencies >= freq_min) & (frequencies <= 0.5 * fs)
    if not np.any(in_band):
        return out

    sel = np.flatnonzero(in_band)
    df = _chirp_step(frequencies[sel], n, fs)
    if df is not None:
        # A uniform ascending run of frequencies is a chirp-z contour: with
        # z_k = a*w**-k, a = exp(2i*pi*f[0]/fs) and w = exp(-2i*pi*df/fs),
        # czt's sum_n x[n]*z_k**-n IS the sum below, evaluated by FFT
        # convolution in O((n+m) log(n+m)) rather than the O(n*m) of the
        # outer product — 2281 ms to 49 ms for 120000 frequencies against a
        # 512-sample waveform. Imported here, like the taper windows: scipy
        # is not needed to hold a Field, only to synthesise from one.
        from scipy.signal import czt
        out[sel] = czt(wf, m=sel.size,
                       w=np.exp(-2j * np.pi * df / fs),
                       a=np.exp(2j * np.pi * frequencies[sel[0]] / fs))
        return out / fs

    # No such contour — and the contract above is ARBITRARY frequencies, which a
    # caller does use (a bare in-band/out-of-band pair, say). Evaluate the
    # sum directly, chunked over frequency so the phase matrix stays bounded
    # regardless of waveform length x grid size (_BLOCK_ELEMENTS per
    # block).
    idx = np.arange(n, dtype=np.float64)
    step = max(1, int(_BLOCK_ELEMENTS // max(n, 1)))
    for a in range(0, sel.size, step):
        blk = sel[a:a + step]
        phase = np.exp(-2j * np.pi * np.outer(frequencies[blk], idx) / fs)
        out[blk] = phase @ wf
    return out / fs
