"""Playing a signal through a channel, and operating on its transfer
function, on plain arrays.

:func:`impulse_response` and :func:`simulate_reception` play a signal
through a known channel; :func:`simulate_arrival_reception` places each
arrival of an arrival list as a phase-rotated, absorbed copy of the
transmitted waveform; the transfer-function operations
(:func:`arrival_transfer_function`, :func:`broadband_propagation_loss`,
:func:`gate_transfer_function`, :func:`synthesize_time_series`, …) take a
channel that did not come from a uacpy model.
"""

from __future__ import annotations

import warnings
import numpy as np
import scipy.fft as _sp_fft
import scipy.signal as _sig
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from typing import Optional, Tuple
from uacpy.core._validate import (
    normalize_axis, require_finite_signal, require_increasing_axis,
    require_positive_finite_scalar, steps_are_uniform,
)
from uacpy.acoustic_signal._synthesis import (
    require_source_waveform, synthesize_cells, warn_unsolved_bins,
    waveform_synthesis_setup,
)
from uacpy.core.absorption import arrival_absorption_exponent


# ──────────────────────────────────────────────────────────────────────
# Channel simulation
#
# Playing a signal through a known impulse
# response, and building that response from a transfer function.
# ──────────────────────────────────────────────────────────────────────

# Windowed-sinc fractional-delay kernel: taps each side of the arrival and
# the Kaiser window shape. L=8 is flat to ~0.01 dB below f/fs = 0.35.
_FRAC_DELAY_HALF_LEN = 8


_FRAC_DELAY_KAISER_BETA = 8.0


# Largest impulse response ``impulse_response`` will size for itself off the
# latest arrival (delays.max() * sample_rate taps): 2**26 taps occupy 1 GiB
# as complex128, beyond any physical channel spread (2**26 taps is 699 s of
# channel at 96 kHz). Past it the caller states n_samples rather than having
# one mis-scaled delay allocate that much silently.
_MAX_DEFAULT_TAPS = 1 << 26


# Largest default DFT length ``impulse_response_from_transfer_function`` will
# size for itself (~4 M samples, ~100 MB across the working arrays). Beyond it
# the caller states n_samples rather than having a finely spaced frequency
# vector allocate that much silently.
_MAX_DEFAULT_IR_SAMPLES = 1 << 22


def fractional_delay_taps(frac: float, half_len: int = _FRAC_DELAY_HALF_LEN,
                          beta: float = _FRAC_DELAY_KAISER_BETA):
    """Kaiser-windowed sinc taps that delay a signal by ``frac`` samples.

    Returns ``2*half_len`` taps for offsets ``-half_len+1 .. half_len``
    relative to the sample the delay floors to, normalised to unit DC gain
    (windowing truncates the sinc, so the raw taps sum to a little under
    one — 0.9999784 at worst over ``frac``, 0.99999579 at half a sample —
    and an unnormalised constant would lose up to 1.9e-4 dB).

    Shared with :func:`delayandsum`, which places whole
    waveforms rather than single taps but needs the identical kernel: a
    two-tap linear split is NOT a fractional delay — its response
    ``|(1-frac) + frac*e^{-jw}|`` is a lowpass whose attenuation depends on
    ``frac``, with a full null at Nyquist for ``frac = 0.5``. The windowed
    sinc is flat to ~0.01 dB below ``f/fs = 0.35``.

    Parameters
    ----------
    frac : float
        The fractional delay, in samples.
    half_len : int, optional
        Half the tap count. Default 8.
    beta : float, optional
        Kaiser window shape. Default 8.
    """
    offsets = np.arange(-half_len + 1, half_len + 1) - float(frac)
    u = offsets / half_len
    win = (np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - u * u))) / np.i0(beta))
    taps = np.sinc(offsets) * win
    total = taps.sum()
    return taps / total if total else taps


def impulse_response(amplitudes, delays_s, *, sample_rate: float,
                     n_samples: Optional[int] = None, fractional: bool = True):
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
        Length of the IR. Default: just past the latest arrival; defaults
        above ``2**26`` taps (1 GiB as complex128) raise rather than
        allocate. An explicit ``n_samples`` can end before an arrival: any
        arrival falling entirely outside the window is dropped from ``h``,
        with a ``NumericsWarning`` counting the drops (on both placement
        paths).
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
        raise ConfigurationError(
            "impulse_response: amplitudes and delays_s must be 1-D, equal "
            f"length; got amplitudes shape {a.shape} and delays_s shape "
            f"{d.shape}.")
    if np.any(d < 0):
        raise ConfigurationError(
            f"impulse_response: delays_s must be >= 0; got "
            f"{int(np.count_nonzero(d < 0))} negative value(s), first at "
            f"index {int(np.argmax(d < 0))} ({d[d < 0][0]:g} s)")
    fs = require_positive_finite_scalar(
        sample_rate, "impulse_response", "sample_rate", " Hz")
    pos = d * fs
    L = _FRAC_DELAY_HALF_LEN
    if n_samples is None:
        if pos.size:
            # Non-fractional rounds to the nearest sample, so the last
            # occupied index is round(pos.max()), not floor(pos.max()).
            # Compared as float before the int conversion so an inf/NaN
            # delay hits the typed bound, not a raw OverflowError.
            n_est = float(np.floor(pos.max()) + L + 1 if fractional
                          else np.round(pos.max()) + 1)
            if not n_est <= _MAX_DEFAULT_TAPS:
                raise ConfigurationError(
                    f"impulse_response: the latest delay {d.max():g} s at "
                    f"sample_rate {fs:g} Hz implies a {n_est:.0f}-tap "
                    f"response ({n_est * 16 / 2 ** 30:.3g} GiB as "
                    f"complex128), above the {_MAX_DEFAULT_TAPS}-tap "
                    f"default limit. Check the delay units (delays_s is "
                    f"seconds), or pass n_samples explicitly.")
            n_samples = n_est
        else:
            n_samples = 1
    n_samples = int(n_samples)
    dtype = complex if np.iscomplexobj(a) else float
    h = np.zeros(n_samples, dtype=dtype)
    n_clipped = 0
    n_dropped = 0
    for amp, p in zip(a, pos):
        if not fractional:
            i0 = int(np.round(p))
            if 0 <= i0 < n_samples:
                h[i0] += amp
            else:
                n_dropped += 1
            continue
        i0 = int(np.floor(p))
        k = np.arange(i0 - L + 1, i0 + L + 1)
        g = fractional_delay_taps(p - i0, half_len=L)       # taps on k
        ok = (k >= 0) & (k < n_samples)
        # An integer-delay arrival lives entirely on sample i0 (every other
        # tap is sinc(integer) = 0), so clipping its zero taps loses nothing
        # and it is dropped only when i0 itself falls outside the window. A
        # non-integer arrival with no tap in the window is likewise dropped
        # whole. One with only some taps clipped keeps a truncated kernel —
        # its amplitude changes rather than the whole arrival landing on one
        # sample (an arrival at 10.9 samples would land entirely at 10) —
        # and the change is not one-directional: the sinc tail can sum
        # either way, and the worst case is a GAIN — measured DC gain 1.1274
        # (+1.04 dB) for an arrival 0.5 samples from the start against
        # 0.9862 at 3.5 samples, at fs = 20 kHz over 128 samples.
        if p == i0:
            if not 0 <= i0 < n_samples:
                n_dropped += 1
        elif not ok.any():
            n_dropped += 1
        elif not ok.all():
            n_clipped += 1
        h[k[ok]] += amp * g[ok]
    if n_dropped:
        warnings.warn(
            f"impulse_response: {n_dropped} arrival(s) lie entirely outside "
            f"the {n_samples}-sample response window ({n_samples / fs:g} s "
            f"at {fs:g} Hz) and are dropped from h. Lengthen n_samples, or "
            f"leave it None to size the response just past the latest "
            f"arrival.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if n_clipped:
        warnings.warn(
            f"impulse_response: {n_clipped} fractional arrival(s) sit within "
            f"{L} samples of the ends of an {n_samples}-sample response, so "
            f"their interpolation kernel is truncated, so their amplitude "
            f"is wrong in either direction (measured +1.04 dB for an arrival "
            f"half a sample from the end). Lengthen n_samples, or use "
            f"fractional=False to put each such arrival wholly on its "
            f"nearest sample (a ±0.5-sample timing error; a nearest sample "
            f"outside the window is a dropped arrival, which warns).",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    t = np.arange(n_samples) / fs
    return t, h


def simulate_reception(source_waveform, amplitudes, delays_s, sample_rate: float):
    """Received signal = ``source_waveform`` convolved with the channel IR.

    Returns ``(t, received)`` with ``t`` the output time axis (s) and
    ``received`` of length ``len(source_waveform) + len(h) - 1``.

    The convolution multiplies ``source_waveform`` by each amplitude as given, so a
    complex amplitude ``A·e^{iφ}`` rotates the phase of a complex (baseband
    or analytic) ``source_waveform``, and the result is complex. A real passband
    ``source_waveform`` with complex amplitudes is refused: multiplying a real
    waveform by ``e^{iφ}`` is not a phase rotation, and the real part of
    that product is not the reception. Pass ``analytic_signal(source_waveform)``
    and take the real part of the result, or use
    :func:`simulate_arrival_reception`, which applies ``Re{A·e^{iφ}·
    hilbert(s)}`` itself and adds the volume absorption carried in the
    imaginary travel times and an output window.

    Parameters
    ----------
    source_waveform : array_like
        The transmitted waveform.
    amplitudes : array_like
        Arrival amplitudes, complex to carry a phase (see above).
    delays_s : array_like
        Arrival delays (s).
    sample_rate : float
        Sample rate (Hz).
    """
    x = np.asarray(source_waveform)
    if np.iscomplexobj(amplitudes) and not np.iscomplexobj(x):
        raise ConfigurationError(
            "simulate_reception: amplitudes are complex but source_waveform is "
            "real; multiplying a real waveform by e^{i phi} is not a phase "
            "rotation, so the real part of the result is not the reception.",
            remediation="Pass analytic_signal(source_waveform) and take np.real of "
                        "the result, or call simulate_arrival_reception("
                        "source_waveform, abs(A), delays_s, sample_rate, fc, "
                        "phases_rad=angle(A)).")
    _, h = impulse_response(amplitudes, delays_s, sample_rate=sample_rate)
    # scipy picks the direct sum for short inputs (np.convolve itself, so
    # those are unchanged) and the FFT once the direct O(N·M) cost is larger:
    # a 1 s source_waveform at 48 kHz through a 2 s channel is 4.6e9 multiply-adds
    # direct.
    y = _sig.convolve(x, h)
    t = np.arange(y.size) / float(sample_rate)
    return t, y


def channel_response(h, sample_rate: float, *, nfft: Optional[int] = None):
    """Frequency response ``H(f)`` of a baseband impulse response ``h``.

    Returns ``(frequencies, H)`` with ``H`` **complex** and both axes
    centred on 0 Hz: a baseband channel response is two-sided and not
    conjugate symmetric, so the negative half carries information the
    positive half does not, and ``rfft`` rejects complex input outright.

    This is the direction :func:`impulse_response_from_transfer_function`
    does not go, but the two are not exact inverses and composing them is
    not a no-op. That one models a **real** channel from a one-sided
    ``H(f)`` sampled anywhere, so it resamples onto its own DFT grid and
    zeroes every bin outside the band it was handed — including Nyquist,
    which this function's grid reaches and its does not. Measured: a
    40-tap real ``h`` taken here at ``nfft=1024``, half-band sliced and
    passed back, returns with a peak error of 1.8e-4 and the same 1.8e-4
    smeared past the original support.

    ``nfft`` defaults to ``max(1024, 2 * h.size)``. Zero-padding
    **interpolates** between DFT bins — it draws the shape between the
    nulls, it does not resolve anything the ``h.size / sample_rate`` record
    cannot. The floor of 1024 is there so a short tap set still plots as a
    curve rather than a polygon; the factor of two puts a sample between
    every pair of natural bins. Pass ``nfft`` to pin it.

    Magnitude in dB is the caller's: ``20 * log10(abs(H))`` is ``-inf`` at a
    perfect null, so whoever renders it picks the floor that sets how deep a
    null is drawn, rather than inheriting one chosen here.

    Returns ``(frequencies, H)``, each of length ``nfft``.

    Parameters
    ----------
    h : array_like
        Baseband impulse response (taps).
    sample_rate : float
        Sample rate (Hz).
    nfft : int, optional
        Transform length; ``None`` is ``max(1024, 2 * len(h))``.
    """
    ir = np.asarray(h, dtype=complex)
    if ir.ndim != 1:
        raise ConfigurationError(
            f"channel_response: h must be a 1-D impulse response; got shape "
            f"{ir.shape}. Transform one channel at a time.")
    if ir.size == 0:
        raise ConfigurationError(
            "channel_response: h is empty, so there is no channel to "
            "transform.")
    fs = require_positive_finite_scalar(
        sample_rate, "channel_response", "sample_rate", " Hz")
    n = max(1024, 2 * ir.size) if nfft is None else int(nfft)
    if n < ir.size:
        raise ConfigurationError(
            f"channel_response: nfft={n} is shorter than the {ir.size}-tap "
            f"impulse response, which would truncate it — the tail beyond "
            f"the {n}th tap would be dropped, not folded. Pass nfft >= "
            f"{ir.size}.")
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    H = np.fft.fftshift(np.fft.fft(ir, n))
    return freqs, H


def transfer_function_from_impulse_response(h, sample_rate: float, *,
                                            t0: float = 0.0, band=None,
                                            axis: int = -1):
    """One-sided ``H(f)`` from a real impulse response — the inverse of
    :func:`impulse_response_from_transfer_function`.

    For an ``h`` whose first sample sits at ``t0``::

        H(f) = rfft(h) * exp(-2j pi f t0)

    Returns ``(frequencies, H)``, both 1-D, with ``frequencies`` running
    from 0 to the Nyquist frequency unless ``band`` narrows it.

    **The rotation is what makes it an inverse.** A record starting at
    ``t0`` carries that offset in every sample, so a bare ``rfft`` returns
    ``H`` multiplied by ``exp(+2j pi f t0)`` — right in magnitude, wrong in
    phase. Checked against a two-path ``H``: with the rotation the round
    trip reproduces it to ``max|err| = 0.0000``; without it, 3.16. Magnitude
    alone looks perfect either way, which is why this is easy to get wrong
    and hard to notice — the error only shows once two such spectra
    interfere.

    **Unscaled, because its counterpart is.**
    :func:`impulse_response_from_transfer_function` is a plain
    ``irfft`` — no ``df``, no ``fs`` — so this is a plain ``rfft`` and the
    pair round-trips to floating point. Do not add a ``1/sample_rate``
    here to make it a spectral density: the first draft of this function
    did, and the round trip came back a factor of ``fs`` short with the
    PHASE still perfect, which is the hardest kind of error to notice.
    ``Field.to_time_trace`` and :meth:`Field.to_transfer_function` use the
    density convention instead, and that pair round-trips too — the two
    conventions each close, and must not be mixed.

    **What it cannot recover.** Only what the record holds: an ``h`` that
    was itself band-limited carries no information outside that band, and
    the bins there come back as whatever the synthesis left — near zero for
    a clean record, and the edge artefacts of the original transform
    otherwise. Pass ``band`` to keep the span the data actually supports.
    Nor can it undo a fold: an arrival later than the record wrapped before
    it ever reached these samples.

    Parameters
    ----------
    h : array_like
        Real impulse response, 1-D. A complex array's real part is used —
        a pressure history is real, and transforming an analytic signal as
        though it were one would double the positive-frequency content.
    sample_rate : float
        Rate (Hz) ``h`` is sampled at, positive and finite.
    t0 : float, default 0.0
        Time (s) of ``h[0]``. Leave at 0 for a response that starts at the
        origin; pass the record's start for one that does not, e.g.
        ``Field.to_time_trace``'s ``times[0]``.
    band : (float, float), optional
        ``(low, high)`` in Hz to keep.
    axis : int, default -1
        Time axis of ``h``. Every other axis is carried through untouched,
        so a ``(depth, range, time)`` block of responses transforms in one
        call — which is how :meth:`~uacpy.Field.to_transfer_function` uses
        this.

    Returns
    -------
    (ndarray, ndarray)
        ``(frequencies, H)``. ``H`` keeps ``h``'s shape with ``axis``
        replaced by the kept bins.

    Raises
    ------
    ConfigurationError
        A zero-dimensional ``h`` or one with fewer than two samples along
        ``axis``, an ``axis`` the array does not have, a non-positive or
        non-finite ``sample_rate``, a non-finite ``t0``, or a ``band``
        keeping no bin.
    """
    return _transfer_function_from_impulse_response(
        h,
        sample_rate,
        t0=t0,
        band=band,
        axis=axis)


def _transfer_function_from_impulse_response(h, sample_rate: float, *,
                                            t0: float = 0.0, band=None,
                                            axis: int = -1, who=None):
    """:func:`transfer_function_from_impulse_response` reporting its refusals as ``who``."""
    who = who or "transfer_function_from_impulse_response"
    samples = np.asarray(h)
    if samples.ndim == 0:
        raise ConfigurationError(
            f"{who}: h must have a time axis; got a scalar.")
    axis = normalize_axis(samples, axis, who)
    if samples.shape[axis] < 2:
        raise ConfigurationError(
            f"{who}: h needs at least two samples along axis {axis}; got "
            f"{samples.shape[axis]} (shape {samples.shape}).")
    if np.iscomplexobj(samples):
        samples = samples.real
    samples = samples.astype(float)
    fs = require_positive_finite_scalar(sample_rate, who, "sample_rate", " Hz")
    if not np.isfinite(float(t0)):
        raise ConfigurationError(
            f"{who}: t0 must be finite; got {t0!r}.")
    n_time = samples.shape[axis]
    frequencies = np.fft.rfftfreq(n_time, 1.0 / fs)
    H = np.fft.rfft(samples, axis=axis)
    # The rotation multiplies along the frequency axis, which is `axis` in
    # the result; broadcasting it needs that shape, not a bare 1-D vector.
    shape = [1] * H.ndim
    shape[axis] = frequencies.size
    H = H * np.exp(-2j * np.pi * frequencies * float(t0)).reshape(shape)
    if band is not None:
        low, high = float(band[0]), float(band[1])
        # An edge bin sits ON `low` or `high` by construction whenever the
        # band came from the grid that produced h, and a bare `>=`/`<=`
        # then keeps or drops it on floating-point dust: re-inverting a
        # sample rate moved the 995 Hz edge by 1e-13 Hz and cost a bin.
        # A billionth of a bin is far below the df that separates
        # neighbours, so this admits the edge and nothing else.
        tol = 1e-9 * (fs / n_time)
        keep = (frequencies >= low - tol) & (frequencies <= high + tol)
        if not keep.any():
            raise ConfigurationError(
                f"{who}: band=({low:g}, {high:g}) Hz keeps no bin of a "
                f"spectrum spanning {frequencies[0]:g} to "
                f"{frequencies[-1]:g} Hz at {fs / n_time:g} Hz "
                f"spacing.")
        frequencies = frequencies[keep]
        H = np.compress(keep, H, axis=axis)
    return frequencies, H


def impulse_response_from_transfer_function(H, *, frequencies, sample_rate: float,
                                            n_samples: Optional[int] = None):
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
    spacing of ``frequencies`` (its smallest spacing if it is non-uniform), so
    the default grid resolves every delay the supplied ``H(f)`` resolves. A
    band-limited ``H`` therefore costs a full-band grid: 100-200 Hz on a 1 Hz
    spacing at ``fs = 10`` kHz returns a 10 000-sample response whose 5 001
    grid bins are zero outside the supplied 101. Pass ``n_samples`` to trade
    that window down. Defaults above ``2**22`` samples raise rather than
    allocate.

    **Samples of H are used where they are, never interpolated between**
    when they can be placed exactly. A uniform ``frequencies`` whose first
    sample is not a multiple of its spacing (``np.arange(100.5, 200.5)``, most
    ``linspace`` grids) sits between the DFT bins; it is placed on them
    shifted by the common offset, and the offset is taken back out of the
    time series, which is exact. Interpolating ``H`` between its samples
    instead averages two phasors of every arrival: at delay ``tau`` it
    attenuates by ``|cos(pi tau df)|``, 10.3 dB at ``tau df = 0.4``, inside
    the window above. Where no exact placement exists — a non-uniform
    ``frequencies``, or an ``n_samples`` whose grid spacing is not the
    spacing of ``frequencies`` — ``H`` is interpolated and a ``NumericsWarning``
    says so.

    **Taps, not a received signal.** ``h`` is the discrete channel (an
    unscaled ``irfft``) that :func:`simulate_reception` convolves. For the
    pressure a transmitted waveform produces through a model's ``H(f)`` —
    ``H`` read as a spectral density — use :func:`synthesize_time_series`.

    Returns ``(t, h)``.

    Parameters
    ----------
    H : array_like
        One-sided complex transfer function.
    frequencies : array_like
        The frequency (Hz) of each sample of ``H``, non-negative and
        increasing.
    sample_rate : float
        Sample rate (Hz).
    n_samples : int, optional
        Length of the response; ``None`` takes the grid spacing of
        ``frequencies`` (see above).
    """
    f = np.asarray(frequencies, dtype=float)
    Hc = np.asarray(H, dtype=complex)
    if f.ndim != 1 or f.shape != Hc.shape:
        raise ConfigurationError(
            f"impulse_response_from_transfer_function: H and frequencies "
            f"shapes differ — frequencies is {f.shape}, H is {Hc.shape}. "
            f"Both must be the same 1-D shape: one transfer-function sample "
            f"per frequency.")
    if f.size and (np.any(np.diff(f) <= 0) or f[0] < 0):
        raise ConfigurationError(
            f"impulse_response_from_transfer_function: frequencies must be "
            f"non-negative and strictly increasing; got first={f[0]:g} Hz and "
            f"smallest step "
            f"{np.min(np.diff(f)) if f.size > 1 else float('nan'):g} Hz. "
            f"Sort the axis and drop duplicates before calling.")
    require_increasing_axis(
        f, "impulse_response_from_transfer_function: frequencies")
    # The shared guard rather than a bare float(): every check below divides by
    # fs or by fs/2, so an unvalidated rate escaped as an untyped error about
    # the wrong thing — sample_rate=0 with frequencies[0]=0 raised a bare
    # ZeroDivisionError, NaN a "cannot convert float NaN to integer", Inf an
    # OverflowError, and a negative rate a ConfigurationError announcing "the
    # Nyquist frequency -500 Hz", a true statement about the wrong argument.
    fs = require_positive_finite_scalar(
        sample_rate, "impulse_response_from_transfer_function",
        "sample_rate", " Hz")
    # The DFT grid stops at fs/2, and out-of-grid bins are zero (see above), so
    # a band sitting above Nyquist contributes nothing: it came back as an
    # all-zero h with no diagnostic. Partial overlap is legitimate but lossy —
    # a band straddling fs/2 silently loses the half above it — so it warns.
    nyquist = fs / 2.0
    if f.size and f[0] > nyquist:
        raise ConfigurationError(
            f"impulse_response_from_transfer_function: frequencies span "
            f"{f[0]:g}-{f[-1]:g} Hz, entirely above the Nyquist frequency "
            f"sample_rate/2 = {nyquist:g} Hz, so every DFT bin would be zero "
            f"and h all-zero. Raise sample_rate above {2 * f[-1]:g} Hz, or "
            f"pass a band inside [0, sample_rate/2].")
    if f.size and f[-1] > nyquist:
        warnings.warn(
            f"impulse_response_from_transfer_function: frequencies reach "
            f"{f[-1]:g} Hz, above the Nyquist frequency sample_rate/2 = "
            f"{nyquist:g} Hz; the part of H above it is dropped from h. Raise "
            f"sample_rate above {2 * f[-1]:g} Hz to keep the whole band.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if n_samples is None:
        # The grid spacing, not the bin count, is what the caller sizes the
        # delay window with, so the default follows the spacing of
        # `frequencies`: n = fs/df. Sizing it as 2*(f.size - 1) instead — the
        # rfft bin count — agrees with that only when `frequencies` spans the
        # whole of [0, fs/2]; on a band-limited vector it silently gave a much
        # coarser grid than the caller's own spacing (100-200 Hz at 1 Hz
        # spacing, fs = 10 kHz: df = 50 Hz, a 20 ms window, wrapping a 30 ms
        # arrival onto 10 ms). The smallest spacing is used when `frequencies`
        # is non-uniform, so no part of it is under-resolved.
        if f.size < 2:
            n_samples = 2
        else:
            df = float(np.min(np.diff(f)))
            n_samples = max(2, int(round(fs / df)))
            if n_samples > _MAX_DEFAULT_IR_SAMPLES:
                raise ConfigurationError(
                    f"impulse_response_from_transfer_function: the spacing of "
                    f"frequencies ({df:g} Hz) implies a default grid of "
                    f"{n_samples} samples at fs = {fs:g} Hz, above the "
                    f"{_MAX_DEFAULT_IR_SAMPLES}-sample default limit. Pass "
                    f"n_samples explicitly (its 1/df delay window must still "
                    f"exceed the longest arrival), or coarsen frequencies."
                )
    n_samples = int(n_samples)
    t = np.arange(n_samples) / fs
    df_grid = fs / n_samples
    if f.size >= 2:
        steps = np.diff(f)
        uniform = bool(np.all(np.abs(steps - steps.mean())
                              <= 1e-9 * steps.mean()))
        on_spacing = uniform and abs(steps.mean() - df_grid) <= 1e-9 * df_grid
    else:
        uniform = on_spacing = True
    offset = float(f[0] - np.round(f[0] / df_grid) * df_grid) if f.size else 0.0
    if on_spacing and f.size and abs(offset) > 1e-9 * df_grid:
        # Heterodyne: every sample lands on bin round((f - offset)/df), and
        # the common offset comes back as exp(2 pi i offset t) on the
        # analytic sum — the placement Field synthesis uses.
        bins = np.rint((f - offset) / df_grid).astype(int)
        keep = (bins >= 0) & (bins < (n_samples + 1) // 2) & (f <= nyquist)
        spec = np.zeros(n_samples, dtype=complex)
        spec[bins[keep]] = Hc[keep]
        analytic = np.fft.ifft(spec) * np.exp(2j * np.pi * offset * t)
        return t, 2.0 * np.real(analytic)
    if not on_spacing and f.size >= 2:
        warnings.warn(
            f"impulse_response_from_transfer_function: frequencies "
            f"{'are not uniformly spaced' if not uniform else 'are spaced ' + format(steps.mean(), 'g') + ' Hz'}"
            f" while the DFT grid is spaced {df_grid:g} Hz, so H is "
            f"interpolated between its samples, which attenuates an arrival "
            f"at delay tau by up to |cos(pi tau df)|. Pass a uniform "
            f"frequencies and leave n_samples unset (or n_samples = "
            f"sample_rate / spacing) for an exact transform.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    grid = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    Hr = (np.interp(grid, f, Hc.real, left=0.0, right=0.0)
          + 1j * np.interp(grid, f, Hc.imag, left=0.0, right=0.0))
    h = np.fft.irfft(Hr, n=n_samples)
    return t, h


# ──────────────────────────────────────────────────────────────────────
# Transfer-function operations
#
# Building an H(f) from a path list, averaging one over a band, and
# cutting one down to the paths a pulse of a given length can overlap.
# ──────────────────────────────────────────────────────────────────────


def arrival_transfer_function(frequencies, amplitudes, delays_s, *,
                              delays_imag_s=None, phases_rad=None,
                              phase_offset: float = 0.0, trace_frequency=None,
                              absorption=None):
    """``H(f)`` of a discrete multipath arrival list.

    .. math::
        H(f) = \\sum_i A_i\\,e^{i\\varphi_i}\\,
               e^{\\omega\\,\\mathrm{Im}\\tau_i}\\,
               e^{-i\\omega\\,\\mathrm{Re}\\tau_i}

    The absorption term is separate because ray codes put volume absorption
    in the **imaginary travel time**, not in the amplitude: Bellhop writes
    ``delay_imag`` as its own field (``ArrMod.f90:118-125``), so the
    received amplitude is ``A·exp(ω·Im τ)`` at the frequency the arrivals
    were traced at. Scoring on ``A`` alone treats a late, heavily absorbed
    path as though the water were lossless. Across a band, ``Im τ`` is
    scaled by :func:`~uacpy.core.absorption.arrival_absorption_exponent`:
    with ``trace_frequency`` and a Thorp or Francois-Garrison
    ``absorption`` the exponent follows the law, ``ω_t Im τ · α(f)/α(f_t)``
    (exact); without them it is ``ω · Im τ``, linear in frequency (exact for
    a :class:`~uacpy.core.absorption.ConstantAbsorption`).

    Parameters
    ----------
    frequencies : array_like
        Frequencies (Hz) to evaluate at.
    amplitudes : array_like
        Arrival amplitudes, as **non-negative magnitudes**. A sign belongs
        in ``phases_rad`` as ``pi``, not here — see below.
    delays_s : array_like
        Real travel times (s), same length as ``amplitudes``.
    delays_imag_s : array_like, optional
        Imaginary travel times (s). ``None`` means a lossless list, i.e.
        zeros — which is what a hand-built arrival set without the field
        carries.
    phases_rad : array_like, optional
        Per-arrival phase (radians). ``None`` means zeros.
    phase_offset : float, default 0
        Added to every ``phases_rad``.
    trace_frequency : float, optional
        Frequency (Hz) the arrivals were traced at, where ``Im τ`` is exact.
    absorption : Absorption, optional
        The water-column law ``Im τ`` carries (the traced environment's
        ``absorption``). With ``trace_frequency`` it sets how the absorption
        scales across ``frequencies``.

    Returns
    -------
    ndarray
        Complex ``H(f)``, one entry per frequency.

    Raises
    ------
    ConfigurationError
        Mismatched lengths, a complex or negative amplitude, or a
        non-finite delay.

    Notes
    -----
    **Why a negative amplitude is refused rather than accepted.** This
    package had two implementations of this sum, and they disagreed on
    exactly that input: one took ``abs`` of the amplitude and the other did
    not, so a negative entry silently produced two different spectra — up
    to 10.7 dB apart per bin — with no error on either side. An amplitude
    here is a magnitude and the phase column carries the sign, so a
    negative value means the caller has mixed the two conventions. Saying
    so is the only answer that cannot be silently wrong.
    """
    return _arrival_transfer_function(
        frequencies,
        amplitudes,
        delays_s,
        delays_imag_s=delays_imag_s,
        phases_rad=phases_rad,
        phase_offset=phase_offset,
        trace_frequency=trace_frequency,
        absorption=absorption)


def _arrival_transfer_function(frequencies, amplitudes, delays_s, *,
                              delays_imag_s=None, phases_rad=None,
                              phase_offset: float = 0.0,
                              trace_frequency=None, absorption=None,
                              who: str = "arrival_transfer_function"):
    """:func:`arrival_transfer_function` reporting its refusals as ``who``."""
    freqs = np.asarray(frequencies, dtype=float).ravel()
    amp = np.asarray(amplitudes).ravel()
    delays = np.asarray(delays_s, dtype=float).ravel()
    if np.iscomplexobj(amp):
        raise ConfigurationError(
            f"{who}: amplitudes must be real magnitudes; got a complex "
            f"array. Pass |A| here and its argument in phases_rad.")
    amp = amp.astype(float)
    if amp.size != delays.size:
        raise ConfigurationError(
            f"{who}: amplitudes and delays_s must have the same length; "
            f"got {amp.size} and {delays.size}.")
    if np.any(amp < 0.0):
        n = int(np.count_nonzero(amp < 0.0))
        raise ConfigurationError(
            f"{who}: {n} of {amp.size} amplitudes are negative. An arrival "
            f"amplitude is a magnitude and its sign lives in the phase, so "
            f"pass abs(A) with pi added to phases_rad for each one. "
            f"Accepting it silently is what made two paths in this package "
            f"disagree by up to 10.7 dB per bin.")
    if not np.all(np.isfinite(delays)):
        raise ConfigurationError(
            f"{who}: an arrival has a non-finite delay, so its phase term "
            f"is undefined at every frequency.")
    imag = (np.zeros_like(delays) if delays_imag_s is None
            else np.asarray(delays_imag_s, dtype=float).ravel())
    phase = (np.zeros_like(delays) if phases_rad is None
             else np.asarray(phases_rad, dtype=float).ravel())
    for name, arr in (('delays_imag_s', imag), ('phases_rad', phase)):
        if arr.size != delays.size:
            raise ConfigurationError(
                f"{who}: {name} must have one entry per arrival "
                f"({delays.size}); got {arr.size}.")
    omega = 2.0 * np.pi * freqs
    gains = amp * np.exp(1j * (phase + float(phase_offset)))
    exponent = arrival_absorption_exponent(
        imag, freqs, trace_frequency=trace_frequency, absorption=absorption)
    with np.errstate(over='ignore'):
        contrib = gains[:, None] * np.exp(exponent
                                          - 1j * np.outer(delays, omega))
    return contrib.sum(axis=0)

def arrival_grid_transfer_function(frequencies, cells, *,
                                   phase_offset: float = 0.0,
                                   trace_frequency=None,
                                   absorption=None) -> np.ndarray:
    """``H(f)`` of every receiver of a grid, each from its own arrival list
    (:func:`arrival_transfer_function` per cell).

    Parameters
    ----------
    frequencies : array_like
        Frequencies (Hz) to evaluate at.
    cells : sequence of sequences of mappings
        ``cells[i][j]``, the arrival list of receiver ``(i, j)``, as
        :func:`simulate_arrival_grid` takes it: ``n_arrivals``,
        ``amplitudes``, ``delays`` (s), ``delays_imag`` (s), ``phases``
        (rad); without ``n_arrivals`` the count is the length of
        ``delays``.
    phase_offset : float, default 0
        Added to every arrival's phase.
    trace_frequency, absorption : optional
        The frequency the arrivals were traced at and the water-column law
        their ``Im tau`` carries, as :func:`arrival_transfer_function` takes
        them.

    Returns
    -------
    ndarray
        Complex, shape ``(n_i, n_j, n_frequencies)``. A cell no arrival
        reached is NaN — no data, not a real and perfectly quiet channel
        (the same cell the TL modes report as NaN).
    """
    return _arrival_grid_transfer_function(
        frequencies,
        cells,
        phase_offset=phase_offset,
        trace_frequency=trace_frequency,
        absorption=absorption)


def _arrival_grid_transfer_function(frequencies, cells, *,
                                   phase_offset: float = 0.0,
                                   trace_frequency=None, absorption=None,
                                   who: str = "arrival_grid_transfer_function"
                                   ) -> np.ndarray:
    """:func:`arrival_grid_transfer_function` reporting its refusals as ``who``."""
    freqs = np.atleast_1d(np.asarray(frequencies))
    n_i, n_j = len(cells), len(cells[0])
    H = np.zeros((n_i, n_j, freqs.size), dtype=complex)
    for i in range(n_i):
        for j in range(n_j):
            cell = cells[i][j]
            if _arrival_count(cell) == 0:
                H[i, j, :] = np.nan
                continue
            H[i, j, :] = _arrival_transfer_function(
                freqs, cell['amplitudes'], cell['delays'],
                delays_imag_s=cell['delays_imag'],
                phases_rad=cell['phases'], phase_offset=phase_offset,
                trace_frequency=trace_frequency, absorption=absorption,
                who=who)
    return H



def received_amplitudes(amplitudes, delays_imag_s, phases, frequency: float,
                        *, trace_frequency: Optional[float] = None,
                        absorption=None) -> np.ndarray:
    """Complex amplitude each ray arrival delivers at ``frequency`` (Hz):
    ``|A| · exp(e) · exp(i·phase)``.

    ``amplitudes`` is a ray code's GEOMETRIC amplitude, which carries no
    volume absorption: Bellhop keeps that in the imaginary travel time
    ``delays_imag_s``, and ``e`` is its exponent at ``frequency``
    (:func:`~uacpy.core.absorption.arrival_absorption_exponent`, scaled from
    the ``trace_frequency`` the arrivals were traced at by the
    ``absorption`` law). ``phases`` are in radians. Use this, not the
    geometric column, whenever arrivals are compared or summed: on a 1 km
    near-bottom link at 40 kHz a 3161 m surface-bounce path reads 20 dB
    stronger than a 1000 m bottom bounce on the geometric column and 7.5 dB
    weaker once its 41 dB of absorption is applied.

    Parameters
    ----------
    amplitudes : array_like
        Geometric arrival amplitudes.
    delays_imag_s : array_like
        Imaginary travel times (s).
    phases : array_like
        Arrival phases (rad).
    frequency : float
        Frequency (Hz) to evaluate at.
    trace_frequency : float, optional
        Frequency (Hz) the arrivals were traced at; ``None`` is
        ``frequency``.
    absorption : Absorption, optional
        The traced water-column law, which scales the absorption between the
        two frequencies.

    Returns
    -------
    ndarray, complex, shape (n_arrivals,)
    """
    amplitude = np.abs(np.asarray(amplitudes, dtype=float).ravel())
    delays_imag = np.asarray(delays_imag_s, dtype=float).ravel()
    phase = np.asarray(phases, dtype=float).ravel()
    exponent = arrival_absorption_exponent(
        delays_imag, [float(frequency)], trace_frequency=trace_frequency,
        absorption=absorption)[:, 0]
    with np.errstate(over='ignore'):
        received = amplitude * np.exp(exponent)
    return received * np.exp(1j * phase)


def remove_delay(H, frequencies, delay, *, axis: int = -1) -> np.ndarray:
    """Advance a transfer function by ``delay`` seconds:
    ``H(f) · exp(+2πi·f·τ)``.

    The frequency-domain counterpart of shifting a time axis by ``-τ``. The
    phase of a delay wraps at ``1/τ`` in frequency, so on a grid of spacing
    ``Δf`` it is unambiguous only for ``τ < 1/(2·Δf)``; removing the bulk
    delay leaves the multipath residual, which the grid does resolve. The
    magnitude is untouched.

    Parameters
    ----------
    H : array_like, complex
        The transfer function, with its frequency axis on ``axis``.
    frequencies : float or array_like
        Hz: one per sample of ``axis``, or one value for an ``H`` collapsed
        onto a single frequency (``axis`` is then not read).
    delay : float or array_like
        Seconds, signed (a negative one adds delay); an array broadcasts
        against ``H`` (one delay per range, say, for reduced time).
    axis : int, keyword-only
        The frequency axis of ``H``.

    Returns
    -------
    ndarray, complex
    """
    H = np.asarray(H)
    if not np.iscomplexobj(H):
        raise ConfigurationError(
            "remove_delay: the transfer function is real, so it carries no "
            "phase to move.",
            remediation="Pass the complex H(f), before a dB view throws the "
                        "phase away.")
    hertz = np.asarray(frequencies, dtype=float)
    if hertz.ndim:
        axis = normalize_axis(H, axis, 'remove_delay')
        if hertz.shape != (H.shape[axis],):
            raise ConfigurationError(
                f"remove_delay: {hertz.size} frequencies for the "
                f"{H.shape[axis]} samples of axis {axis}.")
        shape = [1] * H.ndim
        shape[axis] = hertz.size
        hertz = hertz.reshape(shape)
    else:
        hertz = float(hertz)
    if not np.all(np.isfinite(np.asarray(delay, dtype=float))):
        raise ConfigurationError(
            f"remove_delay: delay={delay!r} is not finite.",
            remediation="Pass the delay to remove in seconds, usually the "
                        "geometric travel time r/c.")
    return H * np.exp(2j * np.pi * hertz * delay)


def broadband_propagation_loss(H, weights=None, *, axis: int = -1):
    """Propagation loss of a signal with bandwidth — Ainslie Eq. 11.46.

    .. math::
        \\mathrm{PL} = 10\\log_{10}
        \\frac{\\sum_f w(f)}{\\sum_f w(f)\\,|H(f)|^2}

    with ``w(f) = |S(f)|²`` the source's power spectrum. This is **the**
    transmission loss of a transient: the frequency average of the
    *coherent* ``|H|²``, weighted by the spectrum the signal actually puts
    on each bin (Ainslie, *Sonar Performance Modeling*, sect. 11.3.3,
    Eq. 11.46 and its footnote 13 for the coloured-spectrum form). The
    same quantity appears as Abraham's pulse loss
    ``∫|U|²df / ∫|H|²|U|²df`` (sect. 3.2.4.2) and as Ainslie's total path
    loss (sect. 3.3.2.1).

    Do not confuse it with **incoherent** TL (Eq. 11.47), the average of
    ``|H|²`` over paths rather than over frequency: that one is a cheap
    stand-in the KRAKEN manual endorses, and Ainslie marks it invalid
    within a few wavelengths of a boundary.

    Parameters
    ----------
    H : array_like
        Complex transfer function. Any shape; ``axis`` is the frequency
        axis, and the others are carried through, so a
        ``(depth, range, frequency)`` grid returns a map.
    weights : array_like, optional
        ``w(f) = |S(f)|²`` on the same grid, one per frequency. ``None``
        weights every bin equally — the flat-spectrum case, which is the
        band average of ``|H|²``.
    axis : int, default -1
        Frequency axis of ``H``.

    Returns
    -------
    ndarray
        Loss in dB, ``H``'s shape without ``axis``. A cell with a NaN at
        any weighted frequency stays NaN: a no-data bin must not average
        away into a finite level.

    Raises
    ------
    ConfigurationError
        A ``weights`` of the wrong length, non-finite, or carrying no
        energy at all.

    Notes
    -----
    Accumulated one frequency at a time rather than as
    ``sum(w * |H|**2, axis)``: that spelling materialises a temporary the
    size of the whole broadband grid, on top of the grid itself. Measured
    on 120 x 400 cells over 401 frequencies (a 0.29 GiB grid), peak extra
    allocation is 0.001 GiB against 0.287 GiB for the one temporary, and
    the two agree to 1e-14 dB.
    """
    return _broadband_propagation_loss(H, weights, axis=axis)


def _broadband_propagation_loss(H, weights=None, *, axis: int = -1,
                               who: str = "broadband_propagation_loss"):
    """:func:`broadband_propagation_loss` reporting its refusals as ``who``."""
    data = np.asarray(H)
    if data.ndim == 0:
        raise ConfigurationError(
            f"{who}: H must have a frequency axis; got a scalar.")
    axis = normalize_axis(data, axis, who)
    n_f = data.shape[axis]
    if weights is None:
        w = np.ones(n_f, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ConfigurationError(
                f"{who}: weights must be one value per frequency; got "
                f"shape {w.shape}. It is the source power spectrum ON "
                f"this frequency axis.")
        if w.size != n_f:
            raise ConfigurationError(
                f"{who}: weights has {w.size} samples but the frequency "
                f"axis has {n_f}.")
        if not np.all(np.isfinite(w)):
            raise ConfigurationError(f"{who}: weights must be finite.")
        if w.sum() <= 0.0:
            raise ConfigurationError(
                f"{who}: weights carry no energy, so the weighted average "
                f"is undefined.")
    moved = np.moveaxis(data, axis, 0)
    power = np.zeros(moved.shape[1:], dtype=float)
    for i, weight in enumerate(w):
        if weight:
            power += weight * np.abs(moved[i]) ** 2
        else:
            # A zero-weight bin contributes nothing but must still carry a
            # no-data cell forward: skipping it silently would let a NaN
            # column average to a finite level.
            power += np.where(np.isnan(moved[i]), np.nan, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(w.sum() / power)


def synthesize_time_series(H, *, frequencies, source_waveform, sample_rate: float,
                           t_start: float = 0.0, window: Optional[str] = None,
                           nfft: Optional[int] = None, axis: int = -1):
    """The received pressure signal: ``source_waveform`` through ``H(f)``.

    .. math::
        p(t) = 2\\,\\mathrm{Re} \\sum_k H(f_k)\\,S(f_k)\\,
               e^{2\\pi i f_k t}\\,\\Delta f

    with ``S(f)`` the waveform's exact DTFT on ``frequencies``
    (:func:`~uacpy.acoustic_signal.waveform_spectrum_at`). A Riemann sum of
    the continuous inverse transform, so the amplitude does not depend on
    ``nfft`` or on the bin grid: with ``H ≡ 1`` over the waveform's band it
    returns the waveform itself. ``H`` is a transfer function in the
    package's travelling-wave convention — ``exp(-2πifτ)`` for a delay τ —
    so an arrival at τ lands at ``t = τ``.

    The plain-array form of :meth:`uacpy.Field.synthesize_time_series`
    (same grid, same windowing, same warnings); that method adds a start
    time estimated from the model's range and sound speed, per-cell
    warnings named by depth and range, and the Field re-wrap.

    **Received signal, not taps.** This is the DENSITY convention: ``H`` is
    a spectral density (pressure per hertz per unit source spectrum), so
    the result is a pressure history in the waveform's units.
    :func:`impulse_response_from_transfer_function` answers the other
    question — the discrete channel taps, an unscaled ``irfft`` that
    :func:`simulate_reception` convolves — and the two are not
    interchangeable.

    Parameters
    ----------
    H : array_like
        Complex transfer function; ``axis`` is frequency, every other axis
        is carried through (one trace per cell).
    frequencies : array_like
        The uniform, ascending frequency axis of ``H`` (Hz). Its spacing
        ``Δf`` sets the record length ``1/Δf``; a first sample that is not
        a multiple of ``Δf`` is placed exactly by a common bin offset.
    source_waveform : array_like
        The real 1-D transmitted waveform.
    sample_rate : float
        Rate (Hz) ``source_waveform`` is sampled at; also the least output
        rate (the realised one is ``nfft·Δf``, a power of two above it).
    t_start : float, default 0.0
        Time (s) of the first output sample; 0 is the emission. The record
        is periodic with period ``1/Δf``, so an arrival outside
        ``[t_start, t_start + 1/Δf)`` wraps.
    window : {None, 'hann', 'hamming', 'blackman', 'tukey', 'boxcar'}
        Spectral window across the band; ``None`` leaves the received
        signal ``S·H`` unfiltered.
    nfft : int, optional
        IFFT length; sized automatically when omitted.
    axis : int, default -1
        Frequency axis of ``H``.

    Returns
    -------
    (ndarray, ndarray)
        ``(t, p)``: the time axis (s) and the pressure, ``H``'s shape with
        ``axis`` replaced by time.
    """
    who = "synthesize_time_series"
    data = np.asarray(H)
    if data.ndim == 0:
        raise ConfigurationError(
            f"{who}: H must have a frequency axis; got a scalar.")
    axis = normalize_axis(data, axis, who)
    freqs = np.asarray(frequencies, dtype=float).ravel()
    if data.shape[axis] != freqs.size:
        raise ConfigurationError(
            f"{who}: H has {data.shape[axis]} samples along axis {axis} but "
            f"frequencies has {freqs.size}.")
    wf = require_source_waveform(source_waveform, who)
    t_start = float(t_start)
    if not np.isfinite(t_start):
        raise ConfigurationError(
            f"{who}: t_start must be finite (s); got {t_start!r}.")
    # The grid, the source spectrum and the cell synthesis are the ones
    # Field.synthesize_time_series runs.
    source_spectrum, plan = waveform_synthesis_setup(
        freqs, wf, sample_rate, window=window, nfft=nfft,
        n_samples_floor=0, who=who)
    moved = np.moveaxis(data, axis, -1)
    spectra = moved.reshape(-1, freqs.size)
    warn_unsolved_bins(spectra, who=who)
    traces, time, _, _ = synthesize_cells(spectra, source_spectrum, plan,
                                          t_start)
    out = traces.reshape(moved.shape[:-1] + (traces.shape[-1],))
    return time, np.moveaxis(out, -1, axis)


def uniform_frequency_step(frequencies) -> float:
    """The step of a uniformly-spaced ascending frequency axis, or a refusal.

    Bin placement presumes such a grid: off one, frequencies land at the
    wrong bins and can collide (the later value overwrites). Every routine
    that turns an ``H(f)`` into a time record asks this first. ``who``
    (keyword-only) is the name the refusal carries, for a routine that
    delegates here.

    Parameters
    ----------
    frequencies : array_like
        The frequency axis (Hz).
    """
    return _uniform_frequency_step(frequencies)


def _uniform_frequency_step(frequencies, *,
                           who: str = "uniform_frequency_step") -> float:
    """:func:`uniform_frequency_step` reporting its refusals as ``who``."""
    freqs = np.asarray(frequencies, dtype=float)
    if freqs.ndim != 1 or freqs.size < 2:
        raise ConfigurationError(
            f"{who}: a frequency step needs a 1-D axis of at least two "
            f"frequencies; got shape {freqs.shape}.",
            remediation="Run the model on an equispaced frequencies= array "
                        "of two or more values.")
    df = float(freqs[1] - freqs[0])
    spacings = np.diff(freqs)
    if df <= 0 or not steps_are_uniform(spacings, df):
        raise ConfigurationError(
            f"{who}: the frequency axis must be uniformly spaced and "
            f"ascending; spacing runs from {spacings.min():.6g} Hz to "
            f"{spacings.max():.6g} Hz against a leading Δf of {df:.6g} Hz. "
            f"Resample H(f) onto an equispaced grid before synthesising.",
            remediation="Run the model on an equispaced frequencies= array.",
        )
    return df


def _warn_if_response_wraps(h, offset, record, who: str) -> None:
    """Warn when the impulse response is still live where the window is not,
    at the far side of the circular record.

    The two cases a cut cannot tell apart are an arrival that is genuinely
    late and one that was later than ``1/df`` and folded back. This measures
    the only thing visible from inside: how much of the response sits in the
    half of the record furthest from the window. A decayed response leaves
    that part empty; a folding one does not, and then the energy the window
    removed is not the energy the pulse would resolve.

    It is deliberately NOT the whole complement of the window. Energy
    immediately outside the window is what a cut exists to remove — a
    genuine late path — so counting it warns on channels that cannot fold:
    two equal paths 100 ms apart in a 500 ms record read 50 %.

    **It is a heuristic with a blind spot, not a test.** What it detects is
    a response that has not decayed where a well-sized record would be
    empty. A fold that lands NEAR the origin — a path just past the
    record's end, which is the likeliest kind — arrives in the near half
    and is never counted at all. Size the grid from the arrivals; this only
    catches the loud, obvious case.
    """
    power = np.abs(h) ** 2
    total = power.sum(axis=-1)
    # The FAR HALF of the record from the window, not the whole complement.
    # Energy just outside the window is what a cut is FOR — a genuine late
    # path being removed — and counting it made this warn on channels that
    # cannot fold at all: two equal paths 100 ms apart in a 500 ms record
    # read 50% and were told to refine the grid.
    #
    # ``offset`` is the caller's own circular distance from the window
    # centre, passed in rather than re-derived. Recovering it here as
    # ``argmax(taper)`` gives a rectangular window's left edge, not its
    # centre — the mask would come out rotated by the window's half-width, so
    # the same channel and cut would warn under 'none' and not under 'hann'.
    far = offset > record / 4.0
    live = np.where(total > 0.0,
                    power.sum(axis=-1, where=far)
                    / np.where(total > 0.0, total, 1.0), 0.0)
    worst = float(np.max(live)) if live.size else 0.0
    # 0.02, from a 48-case sweep against this far-half measure: it misses
    # 11.1 % of folds and false-alarms on none of 21 clean channels. The
    # residual 11 % is the measure's blind spot, not a tuning failure. A
    # threshold of 0.33, calibrated on the whole complement of the window,
    # misses 77.8 % of folds against the far half.
    if worst > 0.02:
        warnings.warn(
            f"{who}: {100.0 * worst:.0f}% of the response sits in the "
            f"half of the record furthest from the window. A decayed "
            f"response leaves that empty, so either a path is arriving "
            f"that late, or one later than the 1/df record has folded "
            f"onto it — from H(f) alone these are the same measurement. "
            f"If the arrivals are known to fit the record, ignore this; "
            f"otherwise size the grid from them "
            f"(Arrivals.synthesis_band).",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


#: Points per band sample the peak search evaluates the response on. A path
#: between two samples reads low on the band's own grid by up to 2/pi
#: (-3.9 dB), enough to hand the peak to a weaker path; on a grid an eighth
#: of a sample fine the shortfall is under 0.06 dB.
_PEAK_OVERSAMPLE = 8


#: Samples one block of the oversampled search holds (64 MiB complex128), so
#: a field of many cells is searched in bounded memory.
_PEAK_BLOCK_SAMPLES = 1 << 22


def _response_peak_times(flat, record):
    """Time of each row's ``max|h|`` (s), found on the band-limited response
    interpolated to :data:`_PEAK_OVERSAMPLE` points per sample (a zero-padded
    inverse transform) rather than on the band's own sample grid."""
    n_up = _PEAK_OVERSAMPLE * flat.shape[-1]
    times = np.empty(flat.shape[0])
    block = max(1, _PEAK_BLOCK_SAMPLES // n_up)
    for a in range(0, flat.shape[0], block):
        up = np.fft.ifft(flat[a:a + block], n=n_up, axis=-1)
        times[a:a + block] = np.argmax(np.abs(up), axis=-1) * (record / n_up)
    return times


def gate_transfer_function(H, *, frequencies, duration: float, origin='peak',
                           window: Optional[str] = None, axis: int = -1):
    """Keep only the part of a channel's impulse response within ``duration``
    of its centre, and return the ``H(f)`` of what is left.

    Transform the band to its own baseband response, gate it, transform
    back. It answers "what would this channel look like if only the paths
    arriving within ±``duration`` existed?" — the separability question a
    pulse of that length asks, since a pulse only interferes with the paths
    it can overlap.

    The record the gate lives in is ``1/df`` long and wraps, so the window
    is applied on **circular** distance: one centred near an end reaches
    round to the other, which is where a path near the record edge actually
    sits.

    Parameters
    ----------
    H : array_like
        Complex transfer function on a uniform ascending frequency grid.
    frequencies : array_like
        That grid (Hz). Its spacing sets the record length ``1/df``.
    duration : float
        Half-width of the gate (s). It reaches ``duration`` either side of
        the centre, so ``2*duration`` must fit inside the record.
    origin : {'peak'} or float
        ``'peak'`` centres the gate on each cell's own ``argmax|h|``, found
        on the response interpolated to an eighth of a sample, so a strong
        path between samples is not outranked by a weaker one on the grid;
        a number centres every cell on that fixed time (s) from the start
        of the record, which is what comparing cells requires.
    window : {None, 'hann'}
        ``None`` (a rectangular gate, the package's one spelling for no
        taper) is the separability criterion stated plainly;
        ``'hann'`` is the same cut, tapered.
    axis : int, default -1
        Frequency axis of ``H``.

    Returns
    -------
    ndarray
        Gated ``H(f)``, same shape as ``H``.

    Warns
    -----
    NumericsWarning
        When the response is still live in the half of the record furthest
        from the window — either a genuinely late path or a fold, which
        ``H(f)`` alone cannot distinguish.
    """
    return _gate_transfer_function(
        H,
        frequencies=frequencies,
        duration=duration,
        origin=origin,
        window=window,
        axis=axis)


def _gate_transfer_function(H, *, frequencies, duration: float,
                           origin='peak', window: Optional[str] = None,
                           axis: int = -1,
                           who: str = "gate_transfer_function"):
    """:func:`gate_transfer_function` reporting its refusals as ``who``."""
    data = np.asarray(H)
    freqs = np.asarray(frequencies, dtype=float).ravel()
    if freqs.size < 2:
        raise ConfigurationError(
            f"{who}: needs at least 2 frequencies to define a record "
            f"length; got {freqs.size}.")
    df = _uniform_frequency_step(freqs, who=who)
    record = 1.0 / df
    if window not in (None, 'hann'):
        raise ConfigurationError(
            f"{who}: window must be None (the separability criterion "
            f"stated plainly) or 'hann' (the same cut, tapered); got "
            f"{window!r}.")
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ConfigurationError(
            f"{who}: duration must be a positive number of seconds; got "
            f"{duration!r}.")
    if 2.0 * duration >= record:
        raise ConfigurationError(
            f"{who}: the window reaches {duration:g} s either side of the "
            f"origin, which is not inside the {record:g} s record the grid "
            f"defines (1/df, df = {df:g} Hz), so it keeps everything and "
            f"the call would be a no-op. Refine the grid or shorten the "
            f"pulse.")
    axis = normalize_axis(data, axis, who)
    n_f = freqs.size
    if data.shape[axis] != n_f:
        raise ConfigurationError(
            f"{who}: H has {data.shape[axis]} samples along axis {axis} "
            f"but frequencies has {n_f}.")
    moved = np.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, n_f)
    # The band's own baseband response: one period is 1/df and the step is
    # 1/(n_f*df), which is the resolution the band itself has.
    h = np.fft.ifft(flat, axis=-1)
    times = np.arange(n_f) * (record / n_f)
    if isinstance(origin, str) and origin == 'peak':
        centres = _response_peak_times(flat, record)
    else:
        try:
            fixed = float(origin)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"{who}: origin must be 'peak' or a time in seconds from "
                f"the start of the record; got {origin!r}.") from exc
        if not np.isfinite(fixed) or not 0.0 <= fixed < record:
            raise ConfigurationError(
                f"{who}: origin {fixed:g} s is outside the "
                f"[0, {record:g}) s record this grid defines.")
        centres = np.full(flat.shape[0], fixed)
    # Circular distance: the record wraps, so a window centred near one end
    # reaches round to the other, which is where the response of a path near
    # the record edge actually sits.
    offset = np.abs(times[None, :] - centres[:, None])
    offset = np.minimum(offset, record - offset)
    if window is None:
        taper = (offset <= duration).astype(float)
    else:
        taper = np.where(offset <= duration,
                         0.5 * (1.0 + np.cos(np.pi * offset / duration)),
                         0.0)
    _warn_if_response_wraps(h, offset, record, who)
    out = np.fft.fft(h * taper, axis=-1).reshape(moved.shape)
    return np.moveaxis(out, -1, axis)


# ──────────────────────────────────────────────────────────────────────
# Reception synthesis
#
# Placing each arrival as a phase-rotated, absorbed copy of the
# transmitted waveform, and reporting what the output window left out.
# ──────────────────────────────────────────────────────────────────────


#: Length of the silent trace an empty arrival record returns, when no
#: time_window says otherwise.
_EMPTY_TRACE_SECONDS = 0.1


def _echo_window_counts(starts, ends, powers, n_samples: int) -> dict:
    """Count the placed echoes a ``[0, n_samples)`` record omits or clips.

    ``starts``/``ends`` are each echo's first and one-past-last sample on
    the record, ``powers`` the received power each carries. An echo with no
    sample inside the record is omitted outright — a delay-and-sum drops an
    echo, it does not fold it the way an inverse FFT does — and one that
    straddles an edge is clipped there: at the start it loses its leading
    edge, the first samples of the waveform, so a chirp arrives as a
    different signal; at the end it loses its tail.
    """
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    powers = np.asarray(powers, dtype=float)
    omitted = (ends <= 0) | (starts >= n_samples)
    return {
        'omitted': int(omitted.sum()),
        'clipped_start': int((~omitted & (starts < 0)).sum()),
        'clipped_end': int((~omitted & (ends > n_samples)).sum()),
        'omitted_power': float(powers[omitted].sum()),
        'total_power': float(powers.sum()),
        # First sample of the earliest echo, relative to the record start;
        # negative when it lands before the window. Merged by ``min``.
        'earliest_start': int(starts.min()),
    }


def _echo_window_notice(counts: dict, t_start: float, time_window: float,
                        who: str, run_hint: bool = True) -> Optional[str]:
    """The text for a window that does not hold every echo, or ``None``.

    ``run_hint`` adds the model ``run()`` names of the two window arguments,
    for a notice raised by a model run; a direct function call leaves it off.
    """
    if not (counts.get('omitted') or counts.get('clipped_start')
            or counts.get('clipped_end')):
        return None
    parts = []
    if counts['omitted']:
        total = counts['total_power']
        share = counts['omitted_power'] / total if total > 0.0 else 0.0
        if share >= 1.0 - 1e-12:
            level = "all of the received energy"
        elif share > 0.0:
            level = f"{10.0 * np.log10(share):.0f} dB of the received energy"
        else:
            level = "no measurable energy"
        parts.append(f"{counts['omitted']} echo(es) fall entirely outside it "
                     f"and are omitted — a delay-and-sum drops an echo rather "
                     f"than folding it — carrying {level}")
    if counts['clipped_start']:
        parts.append(f"{counts['clipped_start']} echo(es) begin before it and "
                     f"lose their leading edge, the first samples of the "
                     f"waveform, so a chirp arrives as a different signal")
    if counts['clipped_end']:
        parts.append(f"{counts['clipped_end']} echo(es) run past its end and "
                     f"lose their tail")
    if 'earliest_s' in counts:
        parts.append(
            f"the earliest echo arrives at {counts['earliest_s']:g} s")
    hint = (" (on run(): output_duration= and t_start=)" if run_hint
            else "")
    return (f"{who}: the [{t_start:g}, {t_start + time_window:g}] s window "
            f"does not hold every echo: " + "; ".join(parts) + ". Widen "
            f"time_window= or move t_start={hint}, or leave both unset to "
            f"size the window from the arrivals.")


def _arrival_absorption_filter(sts_analytic, sample_rate: float, fc: float,
                               absorption):
    """``(spectrum, per_unit_imag)`` for filtering an analytic waveform by an
    arrival's volume absorption: the zero-padded FFT of the waveform, and
    the absorption exponent per second of ``Im tau`` on its bins
    (:func:`~uacpy.core.absorption.arrival_absorption_exponent`; the
    exponent is linear in ``Im tau``). The law is evaluated on the bins that
    carry the waveform (within 120 dB of its peak), so a fitted law is never
    asked outside the band the pulse occupies; the empty bins take the
    linear scaling, which moves nothing measurable there."""
    n = sts_analytic.size
    nfft = int(_sp_fft.next_fast_len(2 * n))
    spectrum = np.fft.fft(sts_analytic, nfft)
    bins = np.abs(np.fft.fftfreq(nfft, 1.0 / float(sample_rate)))
    per_unit_imag = 2.0 * np.pi * bins
    magnitude = np.abs(spectrum)
    occupied = (bins > 0.0) & (magnitude >= 1e-6 * magnitude.max())
    if np.any(occupied):
        per_unit_imag[occupied] = arrival_absorption_exponent(
            [1.0], bins[occupied], trace_frequency=fc,
            absorption=absorption)[0]
    return spectrum, per_unit_imag


def simulate_arrival_reception(source_waveform: np.ndarray, amplitudes,
                               delays_s, sample_rate: float, fc: float, *,
                               delays_imag_s=None, phases_rad=None,
                               output_duration: Optional[float] = None,
                               t_start: Optional[float] = None,
                               phase_offset: float = 0.0,
                               fractional: bool = True,
                               report: Optional[dict] = None,
                               absorption=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Received waveform from a sparse arrival list, with carrier and absorption.

    Places a phase-shifted, amplitude-scaled copy of the source waveform at
    each arrival time.  Uses the Hilbert transform (analytic signal) to apply
    arbitrary phase rotations from caustics and boundary reflections, as
    described in the Bellhop User Guide (Sec. 9.3).

    Parameters
    ----------
    amplitudes, delays_s : array_like
        Arrival magnitudes and real travel times (s).
    delays_imag_s : array_like, optional
        Imaginary travel times (s), carrying the volume absorption as
        ``exp(omega * Im tau)`` at ``fc``. ``None`` is a lossless list. Each
        arrival's copy of the waveform is filtered by that absorption at
        every frequency of the waveform, scaled from ``fc`` by
        :func:`~uacpy.core.absorption.arrival_absorption_exponent` — the
        same law :func:`arrival_transfer_function` applies, so a pulse and a
        transfer function built from one arrival list agree.
    phases_rad : array_like, optional
        Per-arrival phase (radians) from caustics and boundary reflections.
        ``None`` is zeros.
    source_waveform : ndarray
        Source waveform (1-D), used as-is.
    sample_rate : float
        Sample rate in Hz.
    fc : float
        Frequency (Hz) the arrivals were traced at, where ``Im tau`` is
        exact (Bellhop's carrier).
    absorption : Absorption, optional
        The water-column law ``Im tau`` carries. A Thorp or
        Francois-Garrison law scales the absorption as ``α(f)/α(fc)``
        (exact); otherwise it scales linearly in ``f``.
    output_duration : float, optional
        Output time window in seconds.  If *None*, estimated from the
        latest arrival plus the source waveform duration plus margin.
    t_start : float, optional
        Start time for the output.  If *None*, set to just before the
        earliest arrival.
    phase_offset : float, optional
        Constant phase (radians) added to every arrival, applied on the
        analytic signal. ``0.0`` (default): a :class:`Bellhop` ARRIVALS
        result already carries the line-source ``exp(-i*pi/4)`` in its
        ``phases``.
    fractional : bool, optional
        Place each echo at its exact delay with a windowed-sinc kernel
        (default). ``False`` rounds every delay to the nearest sample: an
        error up to half a sample, which is tens
        of degrees of carrier phase at any frequency a modem uses, summed
        coherently into the interference pattern. Continuous placement is
        what the formulation asks for — COA eq. 8.30 shifts the waveform by
        ``t - tau(s)``, not by a whole number of samples.
    report : dict, optional
        Collect, instead of warning, what the window omits or clips: the
        counts from :func:`_echo_window_counts` are ADDED into it, so one
        dict passed across every receiver cell of a run totals the grid and
        the run says it once. Left ``None``, this warns itself when an echo
        falls outside the window or straddles one of its edges — a
        delay-and-sum drops such an echo silently otherwise, and the record
        reads as complete.

    Returns
    -------
    time_vector : ndarray
        Time of each sample (s).
    rts : ndarray
        Received time series, shape ``(n_samples,)``.

    The axis comes FIRST, as it does from ``impulse_response``,
    ``simulate_reception`` and the two transform pairs. Both are real
    arrays of the same length, so the wrong order is silent — which is
    why the package has one order rather than two.
    (:func:`delayandsum` keeps its own ``(signal, time)`` and swaps on
    the way out.)

    References
    ----------
    Bellhop User Guide, Section 9.3
    Original MATLAB code: delayandsum.m by M. B. Porter, 8/96
    """
    return _simulate_arrival_reception(
        source_waveform,
        amplitudes,
        delays_s,
        sample_rate,
        fc,
        delays_imag_s=delays_imag_s,
        phases_rad=phases_rad,
        output_duration=output_duration,
        t_start=t_start,
        phase_offset=phase_offset,
        fractional=fractional,
        report=report,
        absorption=absorption)


def _simulate_arrival_reception(
    source_waveform: np.ndarray,
    amplitudes,
    delays_s,
    sample_rate: float,
    fc: float,
    *,
    delays_imag_s=None,
    phases_rad=None,
    output_duration: Optional[float] = None,
    t_start: Optional[float] = None,
    phase_offset: float = 0.0,
    fractional: bool = True,
    report: Optional[dict] = None,
    absorption=None,
    who: str = "simulate_arrival_reception",
) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`simulate_arrival_reception` reporting its refusals as ``who``."""
    sample_rate = require_positive_finite_scalar(sample_rate, who,
                                                 "sample_rate", " Hz")
    fc = require_positive_finite_scalar(fc, who, "fc", " Hz")
    sts = require_finite_signal(source_waveform, who, "source_waveform")
    if sts.ndim != 1 or np.iscomplexobj(sts):
        raise ConfigurationError(
            f"{who}: source_waveform must be a real 1-D waveform; got "
            f"{'a complex' if np.iscomplexobj(sts) else 'a'} array of shape "
            f"{sts.shape}. Each arrival's phase is applied to its analytic "
            f"signal here, so pass the real waveform.")
    sts = sts.astype(float)
    amps = np.atleast_1d(np.asarray(amplitudes)).ravel()
    if np.iscomplexobj(amps):
        raise ConfigurationError(
            f"{who}: amplitudes must be real magnitudes; got a complex "
            f"array. Pass |A| here and its argument in phases_rad.")
    amps = amps.astype(float)
    delays = np.atleast_1d(np.asarray(delays_s, dtype=float)).ravel()
    if amps.size != delays.size:
        raise ConfigurationError(
            f"{who}: amplitudes and delays_s must have "
            f"the same length; got {amps.size} and {delays.size}.")
    if not (np.all(np.isfinite(amps)) and np.all(np.isfinite(delays))):
        raise ConfigurationError(
            f"{who}: amplitudes and delays_s must be finite; an arrival "
            f"with a non-finite amplitude or delay has no place on the "
            f"record.")
    delays_imag = (np.zeros_like(delays) if delays_imag_s is None
                   else np.atleast_1d(
                       np.asarray(delays_imag_s, dtype=float)).ravel())
    phases = (np.zeros_like(delays) if phases_rad is None
              else np.atleast_1d(np.asarray(phases_rad, dtype=float)).ravel())
    for name, arr in (('delays_imag_s', delays_imag), ('phases_rad', phases)):
        if arr.size != delays.size:
            raise ConfigurationError(
                f"{who}: {name} must have one entry per "
                f"arrival ({delays.size}); got {arr.size}.")
    n_arr = delays.size
    if n_arr == 0:
        if output_duration is not None:
            nrts = int(np.ceil(output_duration * sample_rate))
        else:
            nrts = int(_EMPTY_TRACE_SECONDS * sample_rate)
        t0 = 0.0 if t_start is None else float(t_start)
        return t0 + np.arange(nrts) / sample_rate, np.zeros(nrts)

    nsts = len(sts)

    # Compute analytic signal via Hilbert transform
    sts_analytic = _sig.hilbert(sts)
    absorption_filter = (_arrival_absorption_filter(sts_analytic, sample_rate,
                                                    fc, absorption)
                         if np.any(delays_imag != 0.0) else None)

    deltat = 1.0 / sample_rate
    src_duration = nsts * deltat

    # Determine time window
    min_delay = float(np.min(delays))
    max_delay = float(np.max(delays))

    # Every arrival places a whole copy of the source waveform starting at its
    # own delay, so the window must reach ``max_delay + src_duration`` for the
    # last one to fit; ``2 *`` leaves a further source duration of tail. The
    # lead-in keeps the earliest arrival's leading edge inside the window, and
    # the max(0, ...) stops the clock from starting before source emission.
    if t_start is None:
        t_start = max(0.0, min_delay - 0.1 * src_duration)

    if output_duration is None:
        output_duration = (max_delay - t_start) + 2.0 * src_duration

    nrts = int(np.ceil(output_duration * sample_rate))
    rts = np.zeros(nrts)

    omega_c = 2.0 * np.pi * fc
    # Where each echo's waveform copy lands on the record — first sample and
    # one past its last — and the power it carries, for the window report.
    starts, ends, powers = [], [], []
    for ia in range(n_arr):
        phase_rad = phases[ia] + phase_offset
        phase_factor = np.exp(1j * phase_rad)

        # ``delays_imag`` is Im(tau) in seconds; the volume-attenuation
        # factor at the trace frequency is exp(omega * Im(tau))
        # (delayandsum.m:134), and the window report weighs each echo by it.
        atten = np.exp(omega_c * delays_imag[ia])

        scaled_amp = amps[ia] * atten

        delay_samples = (delays[ia] - t_start) / deltat

        # Add this arrival's shifted, absorbed copy of the source signal as a
        # single clipped slice-add (vectorised over the source samples). A
        # lossy arrival's copy is filtered at every frequency of the
        # waveform rather than scaled by the one factor at fc.
        if absorption_filter is None:
            contrib = scaled_amp * np.real(sts_analytic * phase_factor)
        else:
            spectrum, per_unit_imag = absorption_filter
            absorbed = np.fft.ifft(
                spectrum * np.exp(delays_imag[ia] * per_unit_imag))[:nsts]
            contrib = amps[ia] * np.real(absorbed * phase_factor)
        if fractional:
            # Resolve the sub-sample part of the delay with the same
            # windowed-sinc kernel impulse_response (this module) uses. Convolving
            # by taps centred on offset 0 delays by (half_len - 1) + frac
            # samples, so the placement index backs off by that integer part
            # and the echo lands at delay_samples exactly.
            i_start = int(np.floor(delay_samples))
            nominal_start = i_start
            taps = fractional_delay_taps(delay_samples - i_start)
            placed = np.convolve(contrib, taps)
            i_start -= taps.size // 2 - 1
        else:
            i_start = int(np.round(delay_samples))
            nominal_start = i_start
            placed = contrib
        lo = max(0, i_start)
        hi = min(nrts, i_start + placed.size)
        if lo < hi:
            rts[lo:hi] += placed[lo - i_start:hi - i_start]
        # The report reads the WAVEFORM's extent, not the kernel's: the
        # sinc taps ring a few samples ahead of the echo, and losing that
        # pre-ring at the window's start is not a cut leading edge.
        starts.append(nominal_start)
        ends.append(nominal_start + nsts)
        powers.append(float(scaled_amp) ** 2)

    counts = _echo_window_counts(starts, ends, powers, nrts)
    earliest = counts.pop('earliest_start')
    counts['earliest_s'] = t_start + earliest * deltat
    if report is not None:
        for key, value in counts.items():
            if key == 'earliest_s':
                report[key] = min(report.get(key, np.inf), value)
            else:
                report[key] = report.get(key, 0) + value
    else:
        notice = _echo_window_notice(counts, t_start, output_duration,
                                     who=who, run_hint=False)
        if notice is not None:
            warnings.warn(notice, NumericsWarning,
                          skip_file_prefixes=USER_FRAME_SKIP)

    time_vector = t_start + np.arange(nrts) * deltat
    return time_vector, rts


def delayandsum(
    rcv_arrivals: dict,
    source_timeseries: np.ndarray,
    sample_rate: float,
    fc: float,
    time_window: Optional[float] = None,
    t_start: Optional[float] = None,
    phase_offset: float = 0.0,
    fractional: bool = True,
    report: Optional[dict] = None,
    absorption=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convolve a source waveform with the channel a Bellhop arrival record
    describes. ``fc`` is the frequency the arrivals were traced at and
    ``absorption`` the traced environment's law, which together scale the
    volume absorption across the waveform's band.

    The reader's per-receiver dict unpacked into
    :func:`~uacpy.acoustic_signal.simulate_arrival_reception`, which is the
    same synthesis over plain arrays — use that one for an arrival list from
    anywhere else.

    ``rcv_arrivals`` carries ``n_arrivals``, ``amplitudes``, ``delays``,
    ``delays_imag`` and ``phases``; an empty record returns a silent trace
    of the requested window.

    Parameters
    ----------
    rcv_arrivals : dict
        One receiver's arrival record (see above).
    source_timeseries : ndarray
        The source waveform.
    sample_rate : float
        Sample rate (Hz).
    fc : float
        Frequency (Hz) the arrivals were traced at.
    time_window : float, optional
        Output duration (s); ``None`` covers the latest arrival.
    t_start, phase_offset, fractional, report, absorption : optional
        As on :func:`~uacpy.acoustic_signal.simulate_arrival_reception`.
    """
    if rcv_arrivals['n_arrivals'] == 0:
        amps = delays = imag = phases = np.zeros(0)
    else:
        amps = rcv_arrivals['amplitudes']
        delays = rcv_arrivals['delays']
        imag = rcv_arrivals['delays_imag']
        phases = rcv_arrivals['phases']
    # This function's own contract is (signal, time); the shared one puts
    # the axis first, as the rest of the package does.
    times, rts = _simulate_arrival_reception(
        source_timeseries, amps, delays, sample_rate, fc,
        delays_imag_s=imag, phases_rad=phases, output_duration=time_window,
        t_start=t_start, phase_offset=phase_offset, fractional=fractional,
        report=report, absorption=absorption, who='delayandsum')
    return rts, times



def _arrival_count(cell) -> int:
    """The arrivals of one receiver cell: its ``n_arrivals``, else the
    length of its ``delays`` (a record built by hand may carry no count)."""
    if 'n_arrivals' in cell:
        return int(cell['n_arrivals'])
    return int(np.size(cell.get('delays', ())))


def simulate_arrival_grid(source_timeseries: np.ndarray, cells,
                          sample_rate: float, fc: float, *,
                          time_window: Optional[float] = None,
                          t_start: Optional[float] = None,
                          absorption=None) -> Tuple[np.ndarray, np.ndarray]:
    """Received waveforms over a grid of receivers, each from its own arrival
    list, on ONE clock.

    :func:`simulate_arrival_reception` per cell, with the window every cell
    shares: a window taken from a single cell covers only that cell's
    delays, and every farther receiver — whose energy lands later — would
    convolve to exactly zero. Left ``None``, ``t_start`` and ``time_window``
    are the reception's own auto-window rule over the delay span of the
    whole grid (the earliest arrival less a tenth of the pulse; the latest
    arrival plus two pulse lengths).

    Parameters
    ----------
    source_timeseries : ndarray
        Source waveform (1-D, real).
    cells : sequence of sequences of mappings
        ``cells[i][j]``, the arrival list of receiver ``(i, j)``: a mapping
        with ``n_arrivals`` (else the length of ``delays``),
        ``amplitudes``, ``delays`` (s), ``delays_imag``
        (s) and ``phases`` (rad), the record :class:`~uacpy.Arrivals` keeps
        per receiver.
    sample_rate : float
        Sample rate in Hz.
    fc : float
        Frequency (Hz) the arrivals were traced at.
    time_window, t_start : float, optional
        The shared window (s); see above.
    absorption : Absorption, optional
        The water-column law ``Im tau`` carries, as
        :func:`simulate_arrival_reception` takes it.

    Returns
    -------
    time_vector : ndarray
        Time of each sample (s).
    traces : ndarray
        Shape ``(n_i, n_j, n_samples)``. A cell no arrival reached is NaN —
        no data, not a silent record that reads as a quiet receiver.
    """
    return _simulate_arrival_grid(
        source_timeseries,
        cells,
        sample_rate,
        fc,
        time_window=time_window,
        t_start=t_start,
        absorption=absorption)


def _simulate_arrival_grid(
    source_timeseries: np.ndarray,
    cells,
    sample_rate: float,
    fc: float,
    *,
    time_window: Optional[float] = None,
    t_start: Optional[float] = None,
    absorption=None,
    who: str = "simulate_arrival_grid",
) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`simulate_arrival_grid` reporting its refusals as ``who``."""
    def unpacked(cell):
        if _arrival_count(cell) == 0:
            empty = np.zeros(0)
            return empty, empty, empty, empty
        return (cell['amplitudes'], cell['delays'], cell['delays_imag'],
                cell['phases'])

    n_i, n_j = len(cells), len(cells[0])
    lock_arrivals = cells[0][0]
    for cell in (cells[i][j] for i in range(n_i) for j in range(n_j)):
        if _arrival_count(cell) > 0:
            lock_arrivals = cell
            break
    if time_window is None or t_start is None:
        spans = [np.asarray(cells[i][j]['delays'], dtype=float)
                 for i in range(n_i) for j in range(n_j)
                 if _arrival_count(cells[i][j]) > 0]
        if spans:
            src_duration = len(source_timeseries) / float(sample_rate)
            grid_min = float(min(s.min() for s in spans))
            grid_max = float(max(s.max() for s in spans))
            if t_start is None:
                t_start = max(0.0, grid_min - 0.1 * src_duration)
            if time_window is None:
                time_window = (grid_max - t_start) + 2.0 * src_duration
    amps, delays, imag, phases = unpacked(lock_arrivals)
    t_vec, _ = _simulate_arrival_reception(
        source_timeseries, amps, delays, sample_rate, fc,
        delays_imag_s=imag, phases_rad=phases, output_duration=time_window,
        t_start=t_start, report={}, absorption=absorption, who=who)
    t_start_locked = float(t_vec[0])
    time_window_locked = float(t_vec[-1] - t_vec[0]) + 1.0 / sample_rate
    n_t = len(t_vec)

    data = np.zeros((n_i, n_j, n_t), dtype=float)
    # One report for the grid: the window is shared, so the cells' omitted
    # and clipped echoes are totalled and said once below.
    clip_report: dict = {}
    for i in range(n_i):
        for j in range(n_j):
            cell = cells[i][j]
            if _arrival_count(cell) == 0:
                data[i, j, :] = np.nan
                continue
            amps, delays, imag, phases = unpacked(cell)
            _, rts = _simulate_arrival_reception(
                source_timeseries, amps, delays, sample_rate, fc,
                delays_imag_s=imag, phases_rad=phases,
                output_duration=time_window_locked, t_start=t_start_locked,
                report=clip_report, absorption=absorption, who=who)
            # A cell's record may differ from the locked one by a sample.
            m = min(len(rts), n_t)
            data[i, j, :m] = np.asarray(rts[:m], dtype=float)

    notice = _echo_window_notice(clip_report, t_start_locked,
                                 time_window_locked, who=who)
    if notice is not None:
        warnings.warn(notice, NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return t_vec, data
