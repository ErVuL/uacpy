"""A spectrum evaluated at the frequencies asked for.

:func:`tone_phasor` returns the amplitude and phase of ONE tone in a
record, and :func:`waveform_spectrum_at` a whole spectrum at arbitrary
frequencies. Both evaluate the transform **at** the frequency rather than
sampling the nearest DFT bin, which is what makes them exact off the
record's own grid.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._validate import (
    normalize_axis, require_finite_signal, require_positive_finite_scalar,
)
from uacpy.acoustic_signal._synthesis import _SCRATCH_BLOCK_ELEMS
from uacpy.acoustic_signal.windows import _taper


# ──────────────────────────────────────────────────────────────────────
# One tone, and a spectrum where you ask for it
#
# The phasor of a single tone in a record, and a whole spectrum
# evaluated AT the frequencies asked for rather than at DFT bins.
# ──────────────────────────────────────────────────────────────────────


def tone_phasor(data, times, frequency, *, window: str = 'hann',
                axis: int = -1):
    """Complex amplitude and phase of one tone in a record.

    .. math::
        A = \\frac{g}{\\sum w}\\sum_n x_n w_n e^{-2\\pi i f t_n}

    with ``g = 2`` for a real record and ``g = 1`` for a complex one.

    The transform is evaluated **at** ``frequency``, not sampled at the
    nearest DFT bin. Off a bin, ``X[k]`` is a leakage sample of the window
    transform — neither the phasor at ``frequency`` nor the one at
    ``freqs[k]``. A model-produced trace picks its own ``nt`` and ``fs``,
    so the frequency of interest is essentially never on a bin, and the
    nearest-bin answer is wrong by a growing amount across the bin:
    measured against this sum, ``-0.056 dB`` and ``18°`` at a tenth of a
    bin, ``-1.418 dB`` and ``89.8°`` at half of one. **The phase reaches
    90° before the level has moved 1.5 dB**, which is why a level check
    alone does not find it. On a bin the two agree to ~1e-15, differing
    only in summation order.

    For a real record the ``2·X/Σw`` estimator assumes a non-DC,
    non-Nyquist tone: the 2 restores the half of the energy sitting in the
    negative-frequency image, and ``Σw`` undoes both the transform's
    ``1/N`` and the taper's amplitude loss. A complex record (baseband or
    analytic) has no such image — ``A·e^{2πift}`` puts all of ``A`` at
    ``+f`` — so it takes ``X/Σw``; an analytic signal ``data + i·H{data}``
    returns the same phasor as its real part ``data``. ``frequency`` may be
    negative for a complex record.

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
    window : {'hann', 'hamming', 'blackman', 'tukey', 'boxcar', None}
        Window across the record, normalised by its sum. ``None`` is
        the rectangular sum.
    axis : int, default -1
        Time axis.

    Returns
    -------
    ndarray or complex
        The phasor, with ``axis`` removed.
    """
    return _tone_phasor(data, times, frequency, window=window, axis=axis)


def _tone_phasor(data, times, frequency, *, window: str = 'hann',
                axis: int = -1, who: str = "tone_phasor"):
    """:func:`tone_phasor` reporting its refusals as ``who``."""
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
    win = _taper(window, t.size, who=who)
    shape = [1] * data.ndim
    shape[axis] = t.size
    kernel = (win * np.exp(-2j * np.pi * frequency * t)).reshape(shape)
    gain = 1.0 if np.iscomplexobj(data) else 2.0
    return gain * np.sum(data * kernel, axis=axis) / np.sum(win)


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
        The sampled waveform, real or complex, non-empty and finite;
        flattened to 1-D.
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
    wf = require_finite_signal(waveform, who, "waveform").ravel()
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
    # regardless of waveform length x grid size (_SCRATCH_BLOCK_ELEMS per
    # block).
    idx = np.arange(n, dtype=np.float64)
    step = max(1, int(_SCRATCH_BLOCK_ELEMS // max(n, 1)))
    for a in range(0, sel.size, step):
        blk = sel[a:a + step]
        phase = np.exp(-2j * np.pi * np.outer(frequencies[blk], idx) / fs)
        out[blk] = phase @ wf
    return out / fs
