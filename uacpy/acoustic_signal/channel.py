"""Playing a signal through a channel, and operating on its transfer
function, on plain arrays.

:func:`impulse_response` and :func:`impulse_response_from_transfer_function`
build a channel's taps, :func:`apply_impulse_response` plays a signal of any
length through them (and :func:`apply_impulse_response_blocks` one that
arrives in blocks), :func:`simulate_reception` does both for an arrival
list; :func:`simulate_arrival_reception` places each
arrival of an arrival list as a phase-rotated, absorbed copy of the
transmitted waveform; the transfer-function operations
(:func:`arrival_transfer_function`, :func:`broadband_loss`,
:func:`truncate_response`, :func:`synthesize_time_series`, …) take a
channel that did not come from a uacpy model.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
from collections.abc import Mapping
import numpy as np
import scipy.fft as _sp_fft
import scipy.linalg as _sp_linalg
import scipy.signal as _sig
from uacpy.acoustic_signal._results import PlottedResult
from uacpy.core.constants import PRESSURE_FLOOR
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from typing import Optional, Tuple
from uacpy.core._validate import (
    normalize_axis, real_array_or_none, real_scalar_or_none,
    require_finite_signal, require_positive_finite_scalar,
    require_strictly_increasing, steps_are_uniform,
)
from uacpy.acoustic_signal._synthesis import (
    ARRIVAL_IR_MARGIN_SAMPLES, arrival_ir_margin, band_sum, record_samples,
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

    The kernel :func:`impulse_response` places each arrival with: a
    two-tap linear split is NOT a fractional delay — its response
    ``|(1-frac) + frac*e^{-jw}|`` is a lowpass whose attenuation depends on
    ``frac``, with a full null at Nyquist for ``frac = 0.5``. The windowed
    sinc is flat to ~0.01 dB below ``f/fs = 0.35`` and rolls off above it;
    :func:`arrival_impulse_response`, the band-limited impulse of each path,
    holds the band up to the Nyquist frequency.

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


def impulse_response(amplitudes, delays, *, sample_rate: float,
                     n_samples: Optional[int] = None, fractional: bool = True):
    """Channel impulse response from discrete arrivals.

    Places each arrival ``amplitudes[i]`` at delay ``delays[i]``. With
    ``fractional=True`` the arrival is placed with a windowed-sinc
    fractional-delay kernel; otherwise it is quantised to the nearest sample.

    The kernel matters because a two-tap linear split is **not** a fractional
    delay: its response ``|(1-frac) + frac*e^{-jw}|`` is a lowpass whose
    attenuation depends on ``frac``, with a full null at Nyquist for
    ``frac = 0.5`` (-3.0 dB at ``f/fs = 0.25``, -10.2 dB at 0.40), so two
    arrivals a propagation model reports as equal would differ by up to
    10 dB, decided by the sub-sample part of their travel times — at the
    right time, at the wrong level. The windowed sinc
    (:func:`fractional_delay_taps`) is flat to ~0.01 dB below
    ``f/fs = 0.35`` and rolls off above it (measured, worst fractional
    delay: -0.58 dB at 0.40, -4.1 dB at 0.45, -11 dB at 0.48); keep the
    band below ``0.35·fs``, or build the taps with
    :func:`arrival_impulse_response`, which holds the band up to the
    Nyquist frequency. Group delay is correct either way.

    Parameters
    ----------
    amplitudes : 1-D array
        Complex (or real) arrival amplitudes.
    delays : 1-D array
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
    d = np.asarray(delays, dtype=float)
    if a.shape != d.shape or a.ndim != 1:
        raise ConfigurationError(
            "impulse_response: amplitudes and delays must be 1-D, equal "
            f"length; got amplitudes shape {a.shape} and delays shape "
            f"{d.shape}.")
    if np.any(d < 0):
        raise ConfigurationError(
            f"impulse_response: delays must be >= 0; got "
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
                    f"default limit. Check the delay units (delays is "
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


def simulate_reception(amplitudes, delays, *, source_waveform,
                       sample_rate: float):
    """Received signal = ``source_waveform`` convolved with the channel IR
    of an arrival list.

    Returns ``(t, received)`` with ``t`` the output time axis (s, from the
    emission) and ``received`` of length ``len(source_waveform) + len(h) -
    1``: the taps of :func:`impulse_response` applied by
    :func:`apply_impulse_response`.

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
    amplitudes : array_like
        Arrival amplitudes, complex to carry a phase (see above).
    delays : array_like
        Arrival delays (s).
    source_waveform : array_like
        The transmitted waveform.
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
                        "abs(A), delays, source_waveform=..., sample_rate=..., "
                        "trace_frequency=..., phases=angle(A)).")
    _, h = impulse_response(amplitudes, delays, sample_rate=sample_rate)
    return _apply_impulse_response(h, source_waveform=x,
                                   sample_rate=sample_rate, t_start=0.0,
                                   axis=-1, who="simulate_reception")


#: Taps below which a convolution is summed directly rather than by FFT:
#: Abraham, *Underwater Acoustic Signal Processing*, sect. 4.9.4 — "when M
#: is small enough (less than 19 ...), direct computation is more
#: efficient".
_DIRECT_CONVOLUTION_MAX_TAPS = 19

#: Least transform length of the streaming overlap-add, and the multiple of
#: the response length it grows with. Abraham sect. 4.9.4 gives the
#: principle — the optimal FFT size grows with the filter's length (512 for
#: M = 75, 1024 for M = 100, 11 operations per sample against the direct
#: form's 100) — but neither number: 8 per tap and the 65 536 floor are
#: uacpy guidance (no recorded measurement), and the floor is above
#: Abraham's optimum for a short filter.
_BLOCK_FFT_MIN = 65536
_BLOCK_FFT_PER_TAP = 8


def apply_impulse_response(h, *, source_waveform, sample_rate: float,
                           t_start: float = 0.0, axis: int = -1):
    """A waveform played through sampled channel taps: the linear
    convolution ``y[n] = sum_m h[m] x[n − m]``.

    By FFT overlap-add (``scipy.signal.oaconvolve``; the direct sum for
    fewer than 19 taps, Abraham sect. 4.9.4), so a long signal through a
    long channel costs ``O((N + M) log M)``, not the ``N·M`` of the direct
    form. The output is the whole convolution, ``N + M − 1`` samples at the
    rate of both inputs, starting at ``t_start``, the time of ``h[0]``
    (the waveform's first sample is the emission). Linear, not circular:
    nothing folds, however long the waveform.

    ``h`` is the discrete channel (the unscaled taps of
    :func:`impulse_response` and
    :func:`impulse_response_from_transfer_function`); a response stored as
    a density in 1/s — a :class:`~uacpy.Field` of ``kind=
    'impulse_response'`` — is ``data / sample_rate`` in taps, which
    :meth:`~uacpy.Field.apply` does for you. A complex ``h`` or waveform
    gives a complex output. For a signal too long to hold in memory, pass
    it in blocks to :func:`apply_impulse_response_blocks`.

    Parameters
    ----------
    h : array_like
        Channel taps along ``axis``; every other axis is carried through
        (one response per cell, all applied to the one waveform).
    source_waveform : array_like
        The 1-D signal, sampled at ``sample_rate``.
    sample_rate : float
        Rate (Hz) of both ``h`` and ``source_waveform``.
    t_start : float, default 0.0
        Time (s) of ``h[0]``, and so of the output's first sample.
    axis : int, default -1
        Time axis of ``h``.

    Returns
    -------
    (ndarray, ndarray)
        ``(t, y)``: ``t = t_start + n/sample_rate`` and ``y`` of ``h``'s
        shape with ``axis`` replaced by ``N + M − 1`` samples.
    """
    return _apply_impulse_response(h, source_waveform=source_waveform,
                                   sample_rate=sample_rate, t_start=t_start,
                                   axis=axis, who="apply_impulse_response")


def _require_start_time(t_start, who: str, *, remediation: str) -> float:
    """``t_start`` as one finite time in seconds, or a ``ConfigurationError``
    naming ``who``: the estimated placements (``None``, ``'reduced'``) and a
    start per cell belong to other calls, which ``remediation`` names."""
    value = real_scalar_or_none(t_start)
    if value is None or not np.isfinite(value):
        raise ConfigurationError(
            f"{who}: t_start must be one finite time in seconds; got "
            f"{t_start!r}.", remediation=remediation)
    return value


def _per_cell_start_times(t_start, who: str, *,
                          remediation: str) -> np.ndarray:
    """``t_start`` (one number, or one per cell) as finite floats, or a
    ``ConfigurationError`` naming ``who`` with ``remediation``: real
    numbers only, element by element (a bool among them is not one), as
    :func:`_require_start_time` takes one."""
    starts = real_array_or_none(t_start)
    if starts is None or not np.all(np.isfinite(starts)):
        raise ConfigurationError(
            f"{who}: t_start must be finite times in seconds, one number or "
            f"one per cell; got {t_start!r}.", remediation=remediation)
    return starts


def _apply_impulse_response(h, *, source_waveform, sample_rate, t_start,
                            axis, who):
    """:func:`apply_impulse_response` reporting its refusals as ``who``."""
    taps = np.asarray(h)
    if taps.ndim == 0:
        raise ConfigurationError(
            f"{who}: h must have a time axis; got a scalar.")
    axis = normalize_axis(taps, axis, who)
    if taps.shape[axis] == 0:
        raise ConfigurationError(
            f"{who}: h has no taps along axis {axis}; pass at least one — "
            f"h=[1.0] is the identity channel.")
    x = np.asarray(source_waveform)
    if x.ndim != 1 or x.size == 0:
        raise ConfigurationError(
            f"{who}: source_waveform must be a non-empty 1-D signal; got "
            f"shape {x.shape}.")
    if x.dtype.kind not in 'biufc' or not np.all(np.isfinite(x)):
        raise ConfigurationError(
            f"{who}: source_waveform must hold finite numbers.")
    fs = require_positive_finite_scalar(sample_rate, who, "sample_rate",
                                        " Hz")
    t_start = _require_start_time(
        t_start, who, remediation="Pass the time (s) of h[0]; Field.apply "
                                  "places a Field's response on its own "
                                  "time axis.")
    n_taps = taps.shape[axis]
    if taps.ndim == 1 and n_taps < _DIRECT_CONVOLUTION_MAX_TAPS:
        y = np.convolve(x, taps)
    else:
        shape = [1] * taps.ndim
        shape[axis] = x.size
        y = _sig.oaconvolve(x.reshape(shape), taps, mode='full', axes=axis)
    return t_start + np.arange(y.shape[axis]) / fs, y


def apply_impulse_response_blocks(h, blocks, *, block_size: Optional[int] = None,
                                  axis: int = -1):
    """A signal arriving in blocks played through channel taps ``h``,
    yielded in blocks: overlap-add in ``O(block + M)`` memory.

    The streaming form of :func:`apply_impulse_response`: the yielded
    blocks, concatenated along ``axis``, are its output — the whole linear
    convolution, ``N + M − 1`` samples, the last ``M − 1`` (the channel's
    ring after the signal ends) yielded once ``blocks`` is exhausted. Only
    one transform block and the ``M − 1``-sample overlap are held, so a
    recording of any length (read with ``uacpy.io.read_wav(start=,
    frames=)``, say) goes through a long channel in a few megabytes.

    The input is cut into segments of ``L`` samples and each is convolved
    by an FFT of length ``n = L + M − 1``, its last ``M − 1`` samples added
    to the start of the next (overlap-and-add, Abraham, *Underwater
    Acoustic Signal Processing*, sect. 4.9.4). The optimal ``n`` grows with
    the response length ``M`` (512 for ``M = 75``, 1024 for ``M = 100``
    there); the default, ``n = next_fast_len(max(8·M, 65536))``, follows
    that growth with a factor and a floor that are uacpy guidance, not
    Abraham's numbers.

    Parameters
    ----------
    h : array_like
        Channel taps along ``axis``; every other axis is carried through.
    blocks : iterable of array_like
        The signal, as 1-D blocks of any lengths, in order.
    block_size : int, optional
        Input samples per transform, ``L``; default from the rule above.
    axis : int, default -1
        Time axis of ``h``, and of each yielded block.

    Yields
    ------
    ndarray
        ``h``'s shape with ``axis`` replaced by ``block_size`` samples (the
        last block shorter), real when ``h`` and the signal are.

    Raises
    ------
    ConfigurationError
        An empty ``h``, a ``block_size`` below 1, a block that is not a
        finite 1-D signal, or a stream with no samples at all.
    """
    who = "apply_impulse_response_blocks"
    taps = np.asarray(h)
    if taps.ndim == 0:
        raise ConfigurationError(
            f"{who}: h must have a time axis; got a scalar.")
    axis = normalize_axis(taps, axis, who)
    n_taps = taps.shape[axis]
    if n_taps == 0:
        raise ConfigurationError(
            f"{who}: h has no taps along axis {axis}; pass at least one — "
            f"h=[1.0] is the identity channel.")
    taps = np.moveaxis(taps, axis, -1)
    if block_size is None:
        n_fft = _sp_fft.next_fast_len(
            max(_BLOCK_FFT_PER_TAP * n_taps, _BLOCK_FFT_MIN))
        segment = n_fft - n_taps + 1
    else:
        if int(block_size) != block_size or int(block_size) < 1:
            raise ConfigurationError(
                f"{who}: block_size must be a whole number of samples, at "
                f"least 1; got {block_size!r}.")
        segment = int(block_size)
        n_fft = _sp_fft.next_fast_len(segment + n_taps - 1)
    spectra = {}

    def response(complex_input):
        # One spectrum per input kind: rfft for real data through real taps.
        real = not (complex_input or np.iscomplexobj(taps))
        if real not in spectra:
            spectra[real] = (_sp_fft.rfft(taps, n_fft, axis=-1) if real
                             else _sp_fft.fft(taps, n_fft, axis=-1))
        return real, spectra[real]

    def convolve(segment_samples):
        real, spectrum = response(np.iscomplexobj(segment_samples))
        length = segment_samples.size + n_taps - 1
        if real:
            out = _sp_fft.irfft(_sp_fft.rfft(segment_samples, n_fft)
                                * spectrum, n_fft, axis=-1)
        else:
            out = _sp_fft.ifft(_sp_fft.fft(segment_samples, n_fft)
                               * spectrum, axis=-1)
        return out[..., :length]

    def emit(values):
        return np.moveaxis(values, -1, axis)

    overlap = None
    pending = []
    held = 0
    seen = 0
    for index, block in enumerate(blocks):
        samples = np.asarray(block)
        if (samples.ndim != 1 or samples.dtype.kind not in 'biufc'
                or not np.all(np.isfinite(samples))):
            raise ConfigurationError(
                f"{who}: block {index} must be a 1-D signal of finite "
                f"numbers; got shape {samples.shape}, dtype {samples.dtype}.")
        if samples.size == 0:
            continue
        pending.append(samples)
        held += samples.size
        seen += samples.size
        while held >= segment:
            joined = np.concatenate(pending)
            current, rest = joined[:segment], joined[segment:]
            pending = [rest] if rest.size else []
            held = rest.size
            out = convolve(current)
            if overlap is not None:
                if np.iscomplexobj(overlap) and not np.iscomplexobj(out):
                    out = out.astype(complex)
                out[..., :n_taps - 1] += overlap
            overlap = out[..., segment:].copy()
            yield emit(out[..., :segment])
    if seen == 0:
        raise ConfigurationError(
            f"{who}: the signal had no samples; there is nothing to "
            f"convolve.")
    if held:
        out = convolve(np.concatenate(pending))
        if overlap is not None:
            if np.iscomplexobj(overlap) and not np.iscomplexobj(out):
                out = out.astype(complex)
            out[..., :n_taps - 1] += overlap
        yield emit(out)
    elif overlap is not None and n_taps > 1:
        yield emit(overlap)


class ChannelResponse(PlottedResult,
                      namedtuple("ChannelResponse", "frequencies response")):
    """The two-sided frequency response of a baseband channel: complex
    ``response`` at each of ``frequencies`` (Hz, centred on 0 Hz), what
    :func:`channel_response` returns.

    The tuple is the response, so ``f, H = channel_response(h, fs)`` unpacks
    it. :meth:`plot` draws ``|H(f)|`` in dB with
    :func:`~uacpy.plot.plot_channel_response`.
    """

    _plotter = "plot_channel_response"

    def _field_units(self):
        # The taps carry no stated unit, so neither does their transform.
        return {"frequencies": "Hz", "response": None}


def channel_response(h, sample_rate: Optional[float] = None, *,
                     nfft: Optional[int] = None) -> ChannelResponse:
    """Frequency response ``H(f)`` of a baseband impulse response ``h``.

    Returns ``(frequencies, H)`` with ``H`` **complex** and both axes
    centred on 0 Hz: a baseband channel response is two-sided and not
    conjugate symmetric, so the negative half carries information the
    positive half does not, and ``rfft`` rejects complex input outright.
    ``h`` may be a :class:`~uacpy.comms.ChannelTaps`, whose taps sit at
    ``symbol_rate * sps`` samples per second; ``sample_rate`` is then
    that rate and is not passed.

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

    Parameters
    ----------
    h : array_like or ChannelTaps
        Baseband impulse response (taps).
    sample_rate : float
        Sample rate (Hz); required for array taps.
    nfft : int, optional
        Transform length; ``None`` is ``max(1024, 2 * len(h))``.

    Returns
    -------
    ChannelResponse
        ``(frequencies, response)``, each of length ``nfft``; ``.plot()``
        draws ``|H(f)|`` in dB.
    """
    if all(hasattr(h, name) for name in ('taps', 'symbol_rate', 'sps')):
        if sample_rate is not None:
            raise ConfigurationError(
                "channel_response: a ChannelTaps carries its sample rate "
                "(symbol_rate * sps); sample_rate= would be a second one.",
                remediation="Drop sample_rate=, or pass the taps array "
                            "(taps.taps) with its rate.")
        sample_rate = float(h.symbol_rate) * int(h.sps)
        h = h.taps
    elif sample_rate is None:
        raise ConfigurationError(
            "channel_response: array taps need their sample_rate (Hz).",
            remediation="channel_response(h, sample_rate), or pass a "
                        "ChannelTaps, which carries its rate.")
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
    return ChannelResponse(freqs, H)


def transfer_function_from_impulse_response(h, sample_rate: float, *,
                                            t0: float = 0.0, band=None,
                                            frequencies=None,
                                            axis: int = -1):
    """One-sided ``H(f)`` from a real impulse response — the inverse of
    :func:`impulse_response_from_transfer_function`.

    For an ``h`` whose first sample sits at ``t0``::

        H(f) = rfft(h) * exp(-2j pi f t0)

    Returns ``(frequencies, H)``, both 1-D, with ``frequencies`` running
    from 0 to the Nyquist frequency unless ``band`` narrows it.

    **Pass the synthesis grid as** ``frequencies`` **for an exact inverse
    on any grid.** The ``rfft`` bins are the multiples of
    ``sample_rate/len(h)``. A grid whose first frequency is not one — most
    ``linspace`` grids, and every default BROADBAND grid — is synthesised
    with a common bin offset, so its lines fall between those bins and an
    ``rfft`` returns them leaked, on a shifted axis — off by tens of per
    cent at the band edges of a two-path ``H``. ``frequencies=`` returns ``H`` at
    exactly those frequencies, the offset rotated out and the mirror image
    at the negative frequencies removed, to rounding. Within a bin of 0 Hz
    or of the Nyquist frequency a line and its image barely differ in one
    record: a line within about 0.0025 of a bin of either comes back with a
    warning stating its accuracy, and a band of more than 1024 bins reaching
    there comes back approximate, with a warning. A real record holds only
    the cosine of a line exactly on the Nyquist frequency: an even-length
    record returns its real part there (the quadrature
    :func:`impulse_response_from_transfer_function` warns it drops), and an
    odd-length record, whose Nyquist frequency falls half a bin off its
    lattice, refuses a grid ending on it.

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
    pair round-trips to floating point. A ``1/sample_rate`` here, to make
    it a spectral density, would put the round trip a factor of ``fs``
    short with the PHASE still perfect, which is the hardest kind of error
    to notice.
    ``Field.impulse_response`` and :meth:`Field.to_transfer_function` use the
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
        ``(low, high)`` in Hz to keep, of the ``rfft`` bins.
    frequencies : array_like, optional
        The grid ``h`` was synthesised from (Hz): uniform with spacing
        ``sample_rate/len(h)``, from 0 Hz up to the Nyquist frequency. ``H``
        is returned at exactly these frequencies (see above). Not with
        ``band``.
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
        non-finite ``sample_rate``, a non-finite ``t0``, a ``band``
        keeping no bin, ``band`` and ``frequencies`` together, or a
        ``frequencies`` outside ``[0, sample_rate/2]``, not spaced
        ``sample_rate/len(h)``, or ending on the Nyquist frequency of an
        odd-length ``h``.
    """
    return _transfer_function_from_impulse_response(
        h,
        sample_rate,
        t0=t0,
        band=band,
        frequencies=frequencies,
        axis=axis)


def _transfer_function_from_impulse_response(h, sample_rate: float, *,
                                            t0: float = 0.0, band=None,
                                            frequencies=None,
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
    if frequencies is not None:
        if band is not None:
            raise ConfigurationError(
                f"{who}: pass frequencies= or band=, not both; frequencies "
                f"already names every bin returned.")
        freqs = _record_band_frequencies(frequencies, fs, n_time, who)
        H = np.moveaxis(_record_band_spectrum(
            np.moveaxis(samples, axis, -1), fs, freqs, who), -1, axis)
        shape = [1] * H.ndim
        shape[axis] = freqs.size
        rotation = np.exp(-2j * np.pi * freqs * float(t0)).reshape(shape)
        return freqs, H * rotation
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


#: Most passes :func:`_record_band_spectrum` makes to remove the mirror
#: image; each shrinks the error by the image's leakage into the band
#: (~1e-3 for a band hundreds of bins above 0 Hz, so a handful of passes).
_MIRROR_PASSES = 50

#: Most band bins :func:`_record_band_spectrum` solves for directly (a
#: real system of twice that order) when the passes do not converge.
_MIRROR_DENSE_MAX_BINS = 1024

#: Error bound (relative to the peak) past which the direct solve of
#: :func:`_record_band_spectrum` warns: rounding times the system's
#: condition number.
_MIRROR_ACCURACY = 1e-9


def _record_band_frequencies(frequencies, fs: float, n_time: int,
                             who: str) -> np.ndarray:
    """``frequencies`` validated as the band a real ``n_time``-sample record
    at ``fs`` was synthesised on: 1-D, finite, ascending from 0 Hz or above,
    up to the Nyquist frequency (below it for an odd ``n_time``), and — with
    two or more — spaced exactly the record's resolution ``fs / n_time``."""
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    df = fs / n_time
    if freqs.ndim != 1 or not np.all(np.isfinite(freqs)):
        raise ConfigurationError(
            f"{who}: frequencies must be a finite 1-D axis (Hz); got shape "
            f"{freqs.shape}.")
    if freqs[0] < 0.0 or freqs[-1] > fs / 2.0:
        raise ConfigurationError(
            f"{who}: frequencies run {freqs[0]:g}-{freqs[-1]:g} Hz; a real "
            f"record at {fs:g} Hz holds 0 to {fs / 2.0:g} Hz.")
    # A line on the Nyquist frequency is its own mirror image, and the sine
    # there is zero at every sample. An even record has a bin on it and
    # returns the cosine; an odd record's equations for it are singular.
    # The tolerance is the one that takes a band as on the lattice.
    if n_time % 2 and fs / 2.0 - freqs[-1] <= 1e-9 * df:
        raise ConfigurationError(
            f"{who}: frequencies end on the Nyquist frequency {fs / 2.0:g} "
            f"Hz, half a bin off the lattice of this {n_time}-sample record; "
            f"a line there is its own mirror image, and a real record holds "
            f"only its cosine, so H there cannot be recovered.",
            remediation="Pass an even-length record, or a grid ending below "
                        "sample_rate/2.")
    if freqs.size > 1 and not steps_are_uniform(np.diff(freqs), df):
        raise ConfigurationError(
            f"{who}: frequencies must be spaced the record's resolution "
            f"sample_rate/len(h) = {df:g} Hz, as the grid it was synthesised "
            f"on is; got steps of {np.diff(freqs).min():g} to "
            f"{np.diff(freqs).max():g} Hz.",
            remediation="Pass the grid h was synthesised from, or band= "
                        "for the record's own bins.")
    return freqs


def _record_band_spectrum(samples: np.ndarray, fs: float, freqs: np.ndarray,
                          who: str) -> np.ndarray:
    """The ``G_k`` of a real record ``samples[..., n] = 2 Re s[n]``,
    ``s[n] = (1/N) sum_k G_k exp(2 pi i f_k n / fs)``, on the uniform
    ``freqs`` of spacing ``fs/N`` (:func:`_record_band_frequencies`).

    A band on the record's lattice (``f_0`` a multiple of ``fs/N``) is read
    straight off the ``rfft``. A band offset from it by a fraction of a bin
    — where :func:`impulse_response_from_transfer_function` and the Field
    synthesis place it with a common offset — has lines that are not
    whole cycles per record, so an ``rfft`` leaks each of them into every
    bin (Jacobsen & Juhl, *Fundamentals of General Linear Acoustics*,
    App. B.7.2, Fig. B.30: a line between two spectral samples gives
    "false" components). Rotating the record by the offset puts the lines
    back on bins; what remains is the mirror image ``conj(s)``, whose lines
    sit at ``-f_k`` and still leak. It is removed by fixed-point passes,
    ``G = a - image(G)``, each of which shrinks the error by the image's
    leakage into the band. Within a bin of 0 Hz or of the Nyquist frequency
    a line and its image are less than a bin apart, the leakage approaches
    one and the passes stall; the same equations are then solved directly,
    for up to :data:`_MIRROR_DENSE_MAX_BINS` bins, warning when the system's
    condition puts the error above :data:`_MIRROR_ACCURACY`, and a larger
    band warns that it is approximate.
    """
    n = samples.shape[-1]
    df = fs / n
    b0 = int(np.floor(freqs[0] / df + 0.5))
    bins = b0 + np.arange(freqs.size)
    delta = freqs[0] / df - b0
    if abs(delta) <= 1e-9:
        return np.fft.rfft(samples, axis=-1)[..., bins]
    rotation = np.exp(-2j * np.pi * delta * np.arange(n) / n)
    measured = np.fft.fft(samples * rotation, axis=-1)[..., bins]
    scale = float(np.max(np.abs(measured))) if measured.size else 0.0
    estimate = measured
    spectrum = np.zeros(samples.shape, dtype=complex)
    for _ in range(_MIRROR_PASSES):
        spectrum[..., bins] = estimate
        analytic = np.fft.ifft(spectrum, axis=-1) * np.conj(rotation)
        image = np.fft.fft(np.conj(analytic) * rotation, axis=-1)[..., bins]
        update = measured - image
        change = float(np.max(np.abs(update - estimate)))
        estimate = update
        if change <= 1e-14 * scale:
            return estimate
    if freqs.size <= _MIRROR_DENSE_MAX_BINS:
        # Solve measured = G + C conj(G) directly, C[m, j] the image of
        # line j leaked into bin m: (1/N) sum_n exp(-2 pi i x n / N) with
        # x = b_m + b_j + 2 delta, which sums to
        # sin(pi h) exp(-i pi h) exp(i pi x/N) / (N sin(pi x/N)), h the
        # fractional part of 2 delta, and to 1 where the image lands on the
        # bin. The sum depends on x only modulo N and on 2 delta only
        # modulo 1, so both are reduced first: next to the Nyquist
        # frequency x sits near N (and, on an odd record, 2 delta near 1),
        # where sin(pi x/N) and sin(2 pi delta) would be evaluated near pi
        # and keep only the digits the cancellation leaves, while the
        # reduced arguments are small and exact. One real system serves
        # every cell.
        turns = int(np.rint(2.0 * delta))
        half = 2.0 * delta - turns
        pair = bins[:, None] + bins[None, :] + turns
        x = pair - n * ((pair + n // 2) // n) + half
        sine = np.sin(np.pi * x / n)
        whole = np.abs(sine) < 1e-12
        leak = np.where(
            whole, 1.0,
            np.sin(np.pi * half) * np.exp(-1j * np.pi * half)
            * np.exp(1j * np.pi * x / n) / (n * np.where(whole, 1.0, sine)))
        eye = np.eye(freqs.size)
        system = np.block([[eye + leak.real, leak.imag],
                           [leak.imag, eye - leak.real]])
        rhs = measured.reshape(-1, freqs.size).T
        factors = _sp_linalg.lu_factor(system)
        solved = _sp_linalg.lu_solve(factors,
                                     np.concatenate([rhs.real, rhs.imag]))
        rcond = float(_sp_linalg.lapack.dgecon(
            factors[0], np.abs(system).sum(axis=0).max(), norm='1')[0])
        bound = np.finfo(float).eps / rcond if rcond > 0.0 else np.inf
        if bound > _MIRROR_ACCURACY:
            warnings.warn(
                f"{who}: the band starts {freqs[0]:g} Hz, a fraction "
                f"{abs(delta):.2g} of a bin, from a multiple of the record's "
                f"resolution {df:g} Hz next to 0 Hz or the Nyquist "
                f"frequency, where a real record barely separates a line "
                f"from its mirror image; H is accurate to about "
                f"{bound:.0e} of its peak there.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return (solved[:freqs.size] + 1j * solved[freqs.size:]).T.reshape(
            measured.shape)
    warnings.warn(
        f"{who}: the band {freqs[0]:g}-{freqs[-1]:g} Hz comes within a bin "
        f"of 0 Hz or of the Nyquist frequency {fs / 2.0:g} Hz, where a real "
        f"record barely separates a line from its mirror image, and its "
        f"{freqs.size} bins are too many to solve for directly; H is "
        f"approximate there (the last pass still moved it by "
        f"{change / scale if scale else 0.0:.1g} of its peak).",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return estimate


def impulse_response_from_transfer_function(H, *, frequencies, sample_rate: float,
                                            t_start=0.0,
                                            n_samples: Optional[int] = None,
                                            axis: int = -1):
    """Real impulse response (taps) from a one-sided transfer function
    ``H(f)``, at exactly ``sample_rate``.

    .. math::
        h[n] = \\frac{\\Delta f}{f_s}\\, 2\\,\\mathrm{Re} \\sum_k w_k\\,
               H(f_k)\\, e^{2\\pi i f_k (t_0 + n/f_s)}

    with ``t_0 = t_start``, ``w_k = 1/2`` for a line at 0 Hz or on the
    Nyquist frequency (a sampled record holds those once) and 1 otherwise —
    an unscaled ``irfft`` when ``frequencies`` are the multiples of
    ``sample_rate/n``, so ``H ≡ 1`` gives ``h = δ[n]``. Out of band the
    response is zero: a band-limited model ``H(f)`` carries no energy there
    (holding the edge value would fabricate a DC or high-frequency plateau).

    **The record is** ``[t_start, t_start + n_samples/fs)``. A uniform
    ``frequencies`` of spacing ``Δf`` makes the response periodic with
    period ``1/Δf``: an arrival outside the record lands at its delay
    modulo ``1/Δf`` and is then indistinguishable from a genuine one —
    measured, ``Δf = 62.5`` Hz (a 16 ms record) puts a 20 ms delay at 4.000
    ms, and nothing in the returned ``h`` reveals it. No check here can
    catch it, so the caller places the record: ``t_start`` a little before
    the first arrival (``range/c_max`` for a propagation model), and ``Δf <
    1/τ_spread`` for the spread of arrivals behind it. ``t_start`` moves the
    record so a model's bulk delay need not be held in leading zeros: a
    5 km channel on a 0.25 Hz grid is 83 % pure delay from ``t_start=0``.
    ``n_samples`` defaults to one record, ``fs/Δf`` samples when that is a
    whole number and the next whole number above it otherwise (the smallest
    spacing for a non-uniform ``frequencies``); defaults above ``2**22``
    samples raise rather than allocate.

    **Samples of H are used where they are, never interpolated between.**
    A uniform ``frequencies`` is summed at exactly ``t_start + n/fs``,
    whatever its spacing against ``fs`` and wherever its first sample sits
    against a multiple of it: an inverse FFT when ``fs/Δf`` is a whole
    number, a chirp-z evaluation of the same sum otherwise
    (:func:`~uacpy.acoustic_signal._synthesis.band_sum`). Interpolating ``H``
    between its samples instead averages two phasors of every arrival: at
    delay ``tau`` it attenuates by ``|cos(pi tau df)|``, 10.3 dB at ``tau df
    = 0.4``. Only a non-uniform ``frequencies`` has no exact placement; it
    is interpolated onto the ``n_samples`` grid, with a ``NumericsWarning``.
    A real response holds only the cosine of a line exactly on the Nyquist
    frequency (the sine there is zero at every sample); when ``H`` there,
    taken at the record's start, has an imaginary part, a
    ``NumericsWarning`` says it is dropped.

    **Taps, not a received signal.** ``h`` is the discrete channel (the
    unscaled convention) that :func:`apply_impulse_response` convolves a
    waveform with. For the pressure a transmitted waveform produces through
    a model's ``H(f)`` on the record ``1/Δf`` — ``H`` read as a spectral
    density — use :func:`synthesize_time_series`; the two differ by the
    factor ``sample_rate``.

    Parameters
    ----------
    H : array_like
        One-sided complex transfer function; ``axis`` is frequency and
        every other axis is carried through (one response per cell).
    frequencies : array_like
        The frequency (Hz) of each sample of ``H`` along ``axis``,
        non-negative and increasing, at least two: one line states no
        spacing, so it has no record and no impulse response.
    sample_rate : float
        Sample rate (Hz) of the response.
    t_start : float or array_like, default 0.0
        Time (s) of ``h[0]``: a number for every cell, or one per cell
        (an array broadcasting against ``H`` without ``axis``).
    n_samples : int, optional
        Length of the response; ``None`` is one record (see above).
    axis : int, default -1
        Frequency axis of ``H``.

    Returns
    -------
    (ndarray, ndarray)
        ``(t, h)``: ``h`` is ``H``'s shape with ``axis`` replaced by time,
        and ``t`` the time (s) of each sample, ``t_start + n/fs`` — 1-D for
        a number ``t_start``, and ``t_start``'s shape plus a trailing time
        axis for one start per cell.
    """
    who = "impulse_response_from_transfer_function"
    f = np.asarray(frequencies, dtype=float)
    Hc = np.asarray(H, dtype=complex)
    if f.ndim != 1:
        raise ConfigurationError(
            f"{who}: frequencies must be 1-D, one per sample of H along "
            f"axis; got shape {f.shape}.")
    if Hc.ndim == 0:
        raise ConfigurationError(
            f"{who}: H must have a frequency axis; got a scalar.")
    axis = normalize_axis(Hc, axis, who)
    if Hc.shape[axis] != f.size:
        raise ConfigurationError(
            f"{who}: H and frequencies shapes differ — frequencies is "
            f"{f.shape}, H has {Hc.shape[axis]} samples along axis {axis} "
            f"(shape {Hc.shape}). One transfer-function sample per "
            f"frequency.")
    if f.size and (np.any(np.diff(f) <= 0) or f[0] < 0):
        raise ConfigurationError(
            f"{who}: frequencies must be "
            f"non-negative and strictly increasing; got first={f[0]:g} Hz and "
            f"smallest step "
            f"{np.min(np.diff(f)) if f.size > 1 else float('nan'):g} Hz. "
            f"Sort the axis and drop duplicates before calling.")
    require_strictly_increasing(f, f"{who}: frequencies")
    if f.size < 2:
        raise ConfigurationError(
            f"{who}: need at least 2 frequencies; got {f.size}. A single "
            f"line has no impulse response: the record is 1/Δf long, and one "
            f"frequency states no Δf.",
            remediation="Pass H over a band of two or more frequencies.")
    # The shared guard rather than a bare float(): every check below divides by
    # fs or by fs/2, so an unvalidated rate would escape as an untyped error
    # about the wrong thing — sample_rate=0 with frequencies[0]=0 a bare
    # ZeroDivisionError, NaN a "cannot convert float NaN to integer", Inf an
    # OverflowError, and a negative rate a ConfigurationError announcing "the
    # Nyquist frequency -500 Hz", a true statement about the wrong argument.
    fs = require_positive_finite_scalar(sample_rate, who, "sample_rate",
                                        " Hz")
    rest = Hc.shape[:axis] + Hc.shape[axis + 1:]
    starts_in = _per_cell_start_times(
        t_start, who,
        remediation="Pass the time (s) of h[0]: one number, or one per cell "
                    "of H (its shape without the frequency axis).")
    try:
        starts = np.broadcast_to(starts_in, rest)
    except ValueError:
        raise ConfigurationError(
            f"{who}: t_start of shape {starts_in.shape} does not broadcast "
            f"against H without its frequency axis, shape {rest}; pass one "
            f"number, or one start per cell.") from None
    # The response stops at fs/2, and out-of-band samples are zero (see
    # above), so a band sitting above Nyquist contributes nothing: an
    # all-zero h, refused here. Partial overlap is legitimate but lossy — a
    # band straddling fs/2 loses the half above it — so it warns.
    nyquist = fs / 2.0
    if f[0] > nyquist:
        raise ConfigurationError(
            f"{who}: frequencies span "
            f"{f[0]:g}-{f[-1]:g} Hz, entirely above the Nyquist frequency "
            f"sample_rate/2 = {nyquist:g} Hz, so every DFT bin would be zero "
            f"and h all-zero. Raise sample_rate above {2 * f[-1]:g} Hz, or "
            f"pass a band inside [0, sample_rate/2].")
    if f[-1] > nyquist:
        warnings.warn(
            f"{who}: frequencies reach "
            f"{f[-1]:g} Hz, above the Nyquist frequency sample_rate/2 = "
            f"{nyquist:g} Hz; the part of H above it is dropped from h. Raise "
            f"sample_rate above {2 * f[-1]:g} Hz to keep the whole band.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    uniform = steps_are_uniform(np.diff(f), np.diff(f).mean())
    if n_samples is None:
        # The grid spacing, not the bin count, is what sizes the record: the
        # response a band-limited H describes is 1/df long however few bins
        # the band has. The smallest spacing is used when `frequencies` is
        # non-uniform, so no part of it is under-resolved.
        df = float(np.min(np.diff(f)))
        n_samples = (record_samples(fs, float(np.diff(f).mean()))
                     if uniform else max(2, int(round(fs / df))))
        if n_samples > _MAX_DEFAULT_IR_SAMPLES:
            raise ConfigurationError(
                f"{who}: the spacing of "
                f"frequencies ({df:g} Hz) implies a default record of "
                f"{n_samples} samples at fs = {fs:g} Hz, above the "
                f"{_MAX_DEFAULT_IR_SAMPLES}-sample default limit. Pass "
                f"n_samples explicitly (its record must still hold the "
                f"arrivals), or coarsen frequencies."
            )
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ConfigurationError(
            f"{who}: n_samples must be at least 1; got {n_samples}.")
    rows = np.moveaxis(Hc, axis, -1).reshape(-1, f.size)
    row_starts = starts.reshape(-1)
    elapsed = np.arange(n_samples) / fs
    if uniform:
        df = float(np.diff(f).mean())
        tol = 1e-9 * df
        kept = f <= nyquist + tol
        on_nyquist = kept & (np.abs(f - nyquist) <= tol)
        if on_nyquist.any():
            line = rows[:, on_nyquist][:, 0] * np.exp(
                2j * np.pi * np.remainder(nyquist * row_starts, 1.0))
            _warn_nyquist_quadrature_dropped(line, rows, nyquist)
        weights = np.where((np.abs(f) <= tol) | on_nyquist, 0.5, 1.0)
        G = rows[:, kept] * (weights[kept] * df / fs)
        h = 2.0 * band_sum(G, f[kept], sample_rate=fs, t_start=row_starts,
                           n_samples=n_samples).real
    else:
        warnings.warn(
            f"{who}: frequencies are not "
            f"uniformly spaced, so H is interpolated onto the "
            f"{fs / n_samples:g} Hz grid of the response, which "
            f"attenuates an arrival at delay tau by up to "
            f"|cos(pi tau df)|. Pass a uniform frequencies for an exact "
            f"transform.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
        grid = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        # The record's start is taken out of H at its own frequencies, where
        # it is exact, before H is interpolated: a bulk delay left in would
        # be a phase interpolated across.
        rotated = rows * np.exp(2j * np.pi * np.remainder(
            f[None, :] * row_starts[:, None], 1.0))
        on_grid = np.empty((rows.shape[0], grid.size), dtype=complex)
        for i, row in enumerate(rotated):
            on_grid[i] = (np.interp(grid, f, row.real, left=0.0, right=0.0)
                          + 1j * np.interp(grid, f, row.imag, left=0.0,
                                           right=0.0))
        if n_samples % 2 == 0:
            # irfft takes only the real part of the Nyquist bin.
            _warn_nyquist_quadrature_dropped(on_grid[:, -1], on_grid, nyquist)
        h = np.fft.irfft(on_grid, n=n_samples, axis=-1)
    h = np.moveaxis(h.reshape(rest + (n_samples,)), -1, axis)
    if starts_in.ndim == 0:
        return float(starts_in) + elapsed, h
    return starts[..., None] + elapsed, h


def _warn_nyquist_quadrature_dropped(value, H, nyquist: float) -> None:
    """Warn when ``value``, the ``H`` placed on the Nyquist frequency (one
    entry per cell, taken at the record's start), has an imaginary part
    above rounding: the sine at ``sample_rate/2`` is zero at every sample,
    so a real response keeps only the real part there."""
    finite = np.isfinite(H)
    peak = float(np.max(np.abs(H[finite]))) if finite.any() else 0.0
    dropped = np.abs(np.imag(np.asarray(value)))
    dropped = float(np.max(dropped[np.isfinite(dropped)])) if np.any(
        np.isfinite(dropped)) else 0.0
    if dropped > 1e-9 * peak:
        warnings.warn(
            f"impulse_response_from_transfer_function: H at the Nyquist "
            f"frequency {nyquist:g} Hz has an imaginary part "
            f"{dropped / peak:.2g} of the peak |H|, which a real response "
            f"cannot carry (the sine at sample_rate/2 is zero at every "
            f"sample); h keeps only the real part there. Raise sample_rate "
            f"so the band ends below sample_rate/2 to keep all of H.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


# ──────────────────────────────────────────────────────────────────────
# Transfer-function operations
#
# Building an H(f) from a path list, averaging one over a band, and
# cutting one down to the paths a pulse of a given length can overlap.
# ──────────────────────────────────────────────────────────────────────


def arrival_transfer_function(frequencies, amplitudes, delays, *,
                              delays_imag=None, phases=None,
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
        Frequencies (Hz) to evaluate at. With ``trace_frequency`` and a
        Thorp or Francois-Garrison ``absorption`` they must be non-negative
        (the law has no value at a negative frequency); a 0 Hz bin, the
        first of ``np.fft.rfftfreq``, carries no absorption.
    amplitudes : array_like
        Arrival amplitudes, as **non-negative magnitudes**. A sign belongs
        in ``phases`` as ``pi``, not here — see below.
    delays : array_like
        Real travel times (s), same length as ``amplitudes``.
    delays_imag : array_like, optional
        Imaginary travel times (s). ``None`` means a lossless list, i.e.
        zeros — which is what a hand-built arrival set without the field
        carries.
    phases : array_like, optional
        Per-arrival phase (radians). ``None`` means zeros.
    phase_offset : float, default 0
        Added to every ``phases``.
    trace_frequency : float, optional
        Frequency (Hz) the arrivals were traced at, where ``Im τ`` is exact.
    absorption : Absorption or AbsorptionCoefficient, optional
        The water-column law ``Im τ`` carries (the traced environment's
        ``absorption``: a law, or a measured table). With
        ``trace_frequency`` it sets how the absorption scales across
        ``frequencies``.

    Returns
    -------
    ndarray
        Complex ``H(f)``, one entry per frequency. An empty arrival list
        returns zeros, the empty sum: a receiver no path reaches is
        silent, here and in a cell of
        :func:`arrival_grid_transfer_function`.

    Raises
    ------
    ConfigurationError
        Mismatched lengths, a complex or negative amplitude, or a
        non-finite delay.

    Notes
    -----
    **A negative amplitude is refused.** An amplitude here is a magnitude
    and the phase column carries the sign, so a negative value means the
    caller has mixed the two conventions: reading it as ``abs`` drops a
    ``pi`` of phase and reading it signed keeps one, two spectra that differ
    by decibels per bin. :func:`received_amplitudes` and
    :func:`simulate_arrival_reception` refuse it the same way.
    """
    return _arrival_transfer_function(
        frequencies,
        amplitudes,
        delays,
        delays_imag=delays_imag,
        phases=phases,
        phase_offset=phase_offset,
        trace_frequency=trace_frequency,
        absorption=absorption)


def _arrival_column(value, name: str, who: str) -> np.ndarray:
    """One column of an arrival list as a 1-D float array, read by the
    package's one rule for numbers
    (:func:`~uacpy.core._validate.real_array_or_none`): a bool, a complex
    value or text in it is refused, naming ``who`` and the column."""
    column = real_array_or_none(value)
    if column is None:
        raise ConfigurationError(
            f"{who}: {name} must be real numbers, one per arrival; a bool, "
            f"a complex value or text is not one.",
            remediation="Pass one real number per arrival: the delays and "
                        "their imaginary parts in seconds, the amplitudes "
                        "as magnitudes, the phases in radians.")
    return np.atleast_1d(column).ravel()


def _arrival_magnitudes(amplitudes, who: str) -> np.ndarray:
    """An arrival list's amplitudes as a 1-D float array of non-negative
    real magnitudes, refusing a complex or a negative entry: an arrival's
    sign and phase live in its phase column, so a signed amplitude means the
    two conventions were mixed, and the arrival functions — the transfer
    function, the received amplitudes, the reception — would otherwise
    disagree on it."""
    if np.iscomplexobj(amplitudes):
        raise ConfigurationError(
            f"{who}: amplitudes must be real magnitudes; got a complex "
            f"array. Pass |A| here and its argument in phases.")
    amp = _arrival_column(amplitudes, 'amplitudes', who)
    if np.any(amp < 0.0):
        n = int(np.count_nonzero(amp < 0.0))
        raise ConfigurationError(
            f"{who}: {n} of {amp.size} amplitudes are negative. An arrival "
            f"amplitude is a magnitude and its sign lives in the phase, so "
            f"pass abs(A) with pi added to phases for each one.")
    return amp


def _arrival_transfer_function(frequencies, amplitudes, delays, *,
                              delays_imag=None, phases=None,
                              phase_offset: float = 0.0,
                              trace_frequency=None, absorption=None,
                              who: str = "arrival_transfer_function"):
    """:func:`arrival_transfer_function` reporting its refusals as ``who``."""
    freqs = np.asarray(frequencies, dtype=float).ravel()
    phase_offset = _require_phase_offset(phase_offset, who)
    amp = _arrival_magnitudes(amplitudes, who)
    delays = _arrival_column(delays, 'delays', who)
    if amp.size != delays.size:
        raise ConfigurationError(
            f"{who}: amplitudes and delays must have the same length; "
            f"got {amp.size} and {delays.size}.")
    if not np.all(np.isfinite(delays)):
        raise ConfigurationError(
            f"{who}: an arrival has a non-finite delay, so its phase term "
            f"is undefined at every frequency.")
    imag = (np.zeros_like(delays) if delays_imag is None
            else _arrival_column(delays_imag, 'delays_imag', who))
    phase = (np.zeros_like(delays) if phases is None
             else _arrival_column(phases, 'phases', who))
    for name, arr in (('delays_imag', imag), ('phases', phase)):
        if arr.size != delays.size:
            raise ConfigurationError(
                f"{who}: {name} must have one entry per arrival "
                f"({delays.size}); got {arr.size}.")
    gains = amp * np.exp(1j * (phase + phase_offset))
    return _arrival_phasor_sums(
        gains[None, :], freqs, delays, imag,
        trace_frequency=trace_frequency, absorption=absorption)[0]


#: Elements (arrivals x frequencies) of one block of an arrival list's
#: phasors (~32 MB complex): a list of any length over a grid of any size
#: is summed in bounded memory, not as its whole outer product.
_ARRIVAL_SUM_BLOCK_ELEMS = 2_000_000


def _arrival_phasor_sums(weights, freqs, delays, delays_imag, *,
                         trace_frequency, absorption) -> np.ndarray:
    """``weights @ E``, shape ``(n_weights, n_frequencies)``, for the
    arrivals' phasors ``E[a, k] = exp(x_k Im τ_a − i ω_k τ_a)``, ``x_k Im
    τ_a`` the absorption exponent of
    :func:`~uacpy.core.absorption.arrival_absorption_exponent` (``Im τ``
    times one rate per frequency).

    Summed over blocks of arrivals of at most
    :data:`_ARRIVAL_SUM_BLOCK_ELEMS` phasors, so the scratch is bounded
    whatever the list's length and the grid's size. ``weights`` is
    ``(n_weights, n_arrivals)``, real or complex; ``delays`` real (s)."""
    freqs = np.asarray(freqs, dtype=float).ravel()
    delays = np.asarray(delays, dtype=float).ravel()
    delays_imag = np.asarray(delays_imag, dtype=float).ravel()
    weights = np.atleast_2d(weights)
    omega = 2.0 * np.pi * freqs
    # The exponent is Im tau times a rate per frequency on every law, so the
    # law is consulted once for the band and each block scales it.
    rate = arrival_absorption_exponent(
        np.ones(1), freqs, trace_frequency=trace_frequency,
        absorption=absorption)[0]
    sums = np.zeros((weights.shape[0], freqs.size),
                    dtype=np.result_type(weights.dtype, complex))
    step = max(1, _ARRIVAL_SUM_BLOCK_ELEMS // max(freqs.size, 1))
    for a in range(0, delays.size, step):
        block = slice(a, a + step)
        phasors = np.empty((delays[block].size, freqs.size), dtype=complex)
        phasors.real = np.outer(delays_imag[block], rate)
        phasors.imag = np.outer(delays[block], -omega)
        with np.errstate(over='ignore', invalid='ignore'):
            np.exp(phasors, out=phasors)
            for row, weight in zip(sums, weights[:, block]):
                row += (weight[:, None] * phasors).sum(axis=0)
    return sums

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
        Complex, shape ``(n_i, n_j, n_frequencies)``. A cell whose arrival
        list is empty holds zeros, the empty sum, as
        :func:`arrival_transfer_function` returns for it; its dB view is
        the 600 dB no-energy marker, which plots and metrics leave out
        (:func:`~uacpy.core.acoustics.no_energy_mask`). A cell the engine
        never evaluated is the model wrapper's to mark NaN (Bellhop's
        ``r = 0`` column).
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
    phase_offset = _require_phase_offset(phase_offset, who)
    n_i, n_j = len(cells), len(cells[0])
    H = np.zeros((n_i, n_j, freqs.size), dtype=complex)
    for i in range(n_i):
        for j in range(n_j):
            cell = cells[i][j]
            if _arrival_count(cell, who, f" at cells[{i}][{j}]") == 0:
                continue
            H[i, j, :] = _arrival_transfer_function(
                freqs, cell['amplitudes'], cell['delays'],
                delays_imag=cell.get('delays_imag'),
                phases=cell.get('phases'), phase_offset=phase_offset,
                trace_frequency=trace_frequency, absorption=absorption,
                who=who)
    return H



def received_amplitudes(amplitudes, delays_imag, phases, frequency: float,
                        *, trace_frequency: Optional[float] = None,
                        absorption=None) -> np.ndarray:
    """Complex amplitude each ray arrival delivers at ``frequency`` (Hz):
    ``|A| · exp(e) · exp(i·phase)``.

    ``amplitudes`` is a ray code's GEOMETRIC amplitude, which carries no
    volume absorption: Bellhop keeps that in the imaginary travel time
    ``delays_imag``, and ``e`` is its exponent at ``frequency``
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
        Geometric arrival amplitudes, as non-negative magnitudes (a sign
        belongs in ``phases`` as ``pi``; a negative entry is refused, as
        :func:`arrival_transfer_function` refuses it).
    delays_imag : array_like
        Imaginary travel times (s).
    phases : array_like
        Arrival phases (rad).
    frequency : float
        Frequency (Hz) to evaluate at.
    trace_frequency : float, optional
        Frequency (Hz) the arrivals were traced at; ``None`` is
        ``frequency``.
    absorption : Absorption or AbsorptionCoefficient, optional
        The traced water-column law (or measured table), which scales the
        absorption between the two frequencies.

    Returns
    -------
    ndarray, complex, shape (n_arrivals,)
    """
    amplitude = _arrival_magnitudes(amplitudes, "received_amplitudes")
    delays_imag = np.asarray(delays_imag, dtype=float).ravel()
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


def broadband_loss(H, weights=None, *, axis: int = -1):
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
        away into a finite level. A cell with no energy in any weighted bin
        is the 600 dB no-energy marker
        (:data:`~uacpy.core.constants.NO_ENERGY_DB`), as
        :func:`~uacpy.core.acoustics.transmission_loss_dB` reports a zero
        pressure.

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
    return _broadband_loss(H, weights, axis=axis)


def _broadband_loss(H, weights=None, *, axis: int = -1,
                               who: str = "broadband_loss",
                               weights_name: str = "weights"):
    """:func:`broadband_loss` reporting its refusals as ``who``,
    naming ``weights`` as the caller's own argument ``weights_name`` (a
    :meth:`Field.broadband_loss` caller passed ``source_spectrum``, whose
    squared magnitude the weights are)."""
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
                f"{who}: {weights_name} must be 1-D, one value per frequency "
                f"of this axis; got shape {w.shape}. Flattening it would pair "
                f"values with frequencies in an order you did not choose.")
        if w.size != n_f:
            raise ConfigurationError(
                f"{who}: {weights_name} has {w.size} samples but the "
                f"frequency axis has {n_f}; it is the source spectrum ON "
                f"this frequency axis.")
        if not np.all(np.isfinite(w)):
            raise ConfigurationError(f"{who}: {weights_name} must be finite.")
        if w.sum() <= 0.0:
            raise ConfigurationError(
                f"{who}: {weights_name} holds no energy, so the weighted "
                f"average is undefined.")
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
    # A cell with no energy in any weighted bin (an empty arrival list, an
    # empty modal sum) reports the one no-energy level, the 600 dB that
    # PRESSURE_FLOOR gives transmission_loss_dB for a zero pressure.
    total = w.sum()
    return 10.0 * np.log10(total / np.maximum(power,
                                              total * PRESSURE_FLOOR ** 2))


def synthesize_time_series(H, *, frequencies, source_waveform, sample_rate: float,
                           t_start, window: Optional[str] = 'boxcar',
                           axis: int = -1):
    """The received pressure signal: ``source_waveform`` through ``H(f)``,
    at exactly ``sample_rate``.

    .. math::
        p(t) = 2\\,\\mathrm{Re} \\sum_k H(f_k)\\,S(f_k)\\,
               e^{2\\pi i f_k t}\\,\\Delta f

    with ``S(f)`` the waveform's exact DTFT on ``frequencies``
    (:func:`~uacpy.acoustic_signal.waveform_spectrum_at`). A Riemann sum of
    the continuous inverse transform, so the amplitude does not depend on
    the sampling: with ``H ≡ 1`` over the waveform's band it returns the
    waveform itself. ``H`` is a transfer function in the package's
    travelling-wave convention — ``exp(-2πifτ)`` for a delay τ — so an
    arrival at τ lands at ``t = τ`` when τ lies in the record
    ``[t_start, t_start + 1/Δf)``, and folded into it otherwise.

    The record is sampled at ``t_start + n/sample_rate``: ``sample_rate/Δf``
    samples when that is a whole number (an inverse FFT on that lattice),
    else the next whole number above it (the same sum by chirp-z), so the
    result is on the waveform's own time grid. A record shorter than the
    waveform is refused: sampled every ``Δf``, the waveform's spectrum is
    that of the waveform folded with period ``1/Δf``, and the result would
    be that fold. :func:`apply_impulse_response` convolves instead, which
    holds a signal of any length.

    The plain-array form of :meth:`uacpy.Field.synthesize_time_series`
    (same grid, same windowing, same warnings); that method adds a start
    time estimated from the model's range and sound speed, per-cell
    warnings named by depth and range, and the Field re-wrap.

    **Received signal, not taps.** This is the DENSITY convention: ``H`` is
    a spectral density (pressure per hertz per unit source spectrum), so
    the result is a pressure history in the waveform's units.
    :func:`impulse_response_from_transfer_function` answers the other
    question — the discrete channel taps, which
    :func:`apply_impulse_response` convolves — and the two are not
    interchangeable.

    Parameters
    ----------
    H : array_like
        Complex transfer function; ``axis`` is frequency, every other axis
        is carried through (one trace per cell).
    frequencies : array_like
        The uniform, ascending frequency axis of ``H`` (Hz). Its spacing
        ``Δf`` sets the record length ``1/Δf``.
    source_waveform : array_like
        The real 1-D transmitted waveform.
    sample_rate : float
        Rate (Hz) ``source_waveform`` is sampled at, and the rate of the
        result.
    t_start : float
        Time (s) of the first output sample; 0 is the emission. Required:
        the record is periodic with period ``1/Δf``, so an arrival outside
        ``[t_start, t_start + 1/Δf)`` wraps, and nothing in the output
        shows it — ``H`` sampled every ``Δf`` is the same for a delay τ and
        τ + 1/Δf, so a record folded onto ``t_start`` reads exactly like an
        arrival at ``t_start``. An ``H`` from a model at range ``r`` has
        its arrivals near ``r/c``, usually many records past 0: pass a
        start a little before ``r/c_max``, or use
        :meth:`uacpy.Field.synthesize_time_series`, which estimates one
        (``t_start=None``) or gives each range its own record
        (``'reduced'``); here either word is refused, as is an array.
    window : {'boxcar', 'hann', 'hamming', 'blackman', 'tukey'} or None
        Spectral window across the band; ``'boxcar'`` (default; ``None``
        takes it too) leaves the received signal ``S·H`` unfiltered.
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
    t_start = _require_start_time(
        t_start, who,
        remediation="Pass the time (s) of the first output sample, a little "
                    "before the first arrival (r/c_max for a model's H at "
                    "range r). On a Field, Field.synthesize_time_series "
                    "estimates it (t_start=None) or gives each range its "
                    "own record (t_start='reduced').")
    # The grid, the source spectrum and the cell synthesis are the ones
    # Field.synthesize_time_series runs.
    source_spectrum, plan = waveform_synthesis_setup(
        freqs, wf, sample_rate, window=window, who=who)
    moved = np.moveaxis(data, axis, -1)
    spectra = moved.reshape(-1, freqs.size)
    warn_unsolved_bins(spectra, who=who)
    traces, elapsed, _, _ = synthesize_cells(spectra, source_spectrum, plan,
                                             t_start, who=who)
    out = traces.reshape(moved.shape[:-1] + (traces.shape[-1],))
    return t_start + elapsed, np.moveaxis(out, -1, axis)


def uniform_frequency_step(frequencies) -> float:
    """The step of a uniformly-spaced ascending frequency axis, or a refusal.

    Bin placement presumes such a grid: off one, frequencies land at the
    wrong bins and can collide (the later value overwrites). Every routine
    that turns an ``H(f)`` into a time record asks this first.

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
    # path being removed — and counting it would warn on channels that
    # cannot fold at all: two equal paths 100 ms apart in a 500 ms record
    # read 50 %.
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
    # 0.02, uacpy guidance from a 48-case uacpy sweep against this far-half
    # measure (no source states a threshold): it misses 11.1 % of folds and
    # false-alarms on none of 21 clean channels. The residual 11 % is the
    # measure's blind spot, not a tuning failure. A threshold of 0.33,
    # calibrated on the whole complement of the window, misses 77.8 % of
    # folds against the far half.
    if worst > 0.02:
        warnings.warn(
            f"{who}: {100.0 * worst:.0f}% of the response sits in the "
            f"half of the record furthest from the window. A decayed "
            f"response leaves that empty, so either a path is arriving "
            f"that late, or one later than the 1/df record has folded "
            f"onto it — from H(f) alone these are the same measurement. "
            f"If the arrivals are known to fit the record, ignore this; "
            f"otherwise size the grid from them "
            f"(Arrivals.synthesis_band). Its silence does not show the "
            f"record holds the response: a path just past the record's end "
            f"folds near the origin, where this does not look.",
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


def truncate_response(H, *, frequencies, duration: float, origin='peak',
                      window: Optional[str] = 'boxcar', axis: int = -1):
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
    window : {'boxcar', 'hann'}, optional
        ``'boxcar'`` (the default, a rectangular gate, which ``None`` also
        selects) is the separability criterion stated plainly; ``'hann'`` is
        the same cut, tapered.
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
    return _truncate_response(
        H,
        frequencies=frequencies,
        duration=duration,
        origin=origin,
        window=window,
        axis=axis)


def _truncate_response(H, *, frequencies, duration: float,
                           origin='peak', window: Optional[str] = 'boxcar',
                           axis: int = -1,
                           who: str = "truncate_response"):
    """:func:`truncate_response` reporting its refusals as ``who``."""
    data = np.asarray(H)
    freqs = np.asarray(frequencies, dtype=float).ravel()
    if freqs.size < 2:
        raise ConfigurationError(
            f"{who}: needs at least 2 frequencies to define a record "
            f"length; got {freqs.size}.")
    df = _uniform_frequency_step(freqs, who=who)
    record = 1.0 / df
    if window is None:
        window = 'boxcar'               # the gate's documented default
    if window not in ('boxcar', 'hann'):
        raise ConfigurationError(
            f"{who}: window must be 'boxcar' (the separability criterion "
            f"stated plainly; None selects it) or 'hann' (the same cut, "
            f"tapered); got {window!r}.")
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
    if window == 'boxcar':
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
#: output_duration says otherwise: uacpy guidance, with no source.
_EMPTY_TRACE_SECONDS = 0.1


def _echo_window_counts(starts, ends, powers, n_samples: int) -> dict:
    """Count the placed echoes a ``[0, n_samples)`` record omits or clips.

    ``starts``/``ends`` are each echo's first and one-past-last sample on
    the record, ``powers`` the received power each carries. An echo with no
    sample inside the record is omitted outright — the record is cut, not
    folded the way an inverse FFT's record is — and one that
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


def _echo_window_notice(counts: dict, t_start: float, output_duration: float,
                        who: str) -> Optional[str]:
    """The text for a window that does not hold every echo, or ``None``.
    The two window arguments it names, ``output_duration=`` and
    ``t_start=``, are spelled so on every reception and on a model's
    ``run()``."""
    if not (counts.get('omitted') or counts.get('clipped_start')
            or counts.get('clipped_end')):
        return None
    parts = []
    if counts['omitted']:
        level = _energy_share_words(counts['omitted_power'],
                                    counts['total_power'])
        parts.append(f"{counts['omitted']} echo(es) fall entirely outside it "
                     f"and are omitted — the window cuts an echo off rather "
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
    return (f"{who}: the [{t_start:g}, {t_start + output_duration:g}] s "
            "window does not hold every echo: " + "; ".join(parts) + ". "
            "Widen output_duration= or move t_start=, or leave both unset "
            "to size the window from the arrivals.")


#: A reception takes the waveform's Hilbert transform on a record padded
#: either side by this many waveform lengths, at least
#: :data:`~uacpy.acoustic_signal._synthesis.ARRIVAL_IR_MARGIN_SAMPLES` and
#: at most :data:`_HILBERT_PAD_MAX_SAMPLES` samples
#: (:func:`_quadrature_source`).
_HILBERT_PAD_WAVEFORMS = 4


#: Most samples of padding either side of a reception's waveform (~22 s at
#: 48 kHz): past a few lengths of its slowest oscillation more padding
#: does not change the result (a 5 s LFM sweep from 100 Hz: 1.3e-7 of the
#: peak at 60 ms of padding), and a long recording is not tripled.
_HILBERT_PAD_MAX_SAMPLES = 1 << 20


#: Half-length (samples) of the record each arrival's band-limited kernel
#: is built on (:func:`_arrival_kernels`): 2K + 1 samples. It sets how close
#: to the Nyquist frequency the kernel holds the band (the window method's
#: transition, about fs/K wide): white noise up to the Nyquist frequency
#: through 3 / 50 fractional paths comes back within 1.4e-3 / 4.4e-3 of its
#: peak at 4096 (1.2e-2 / 2.1e-2 at 256), and a list of n arrivals costs
#: n·(2K + 1) samples of kernel whatever its span.
_ARRIVAL_KERNEL_HALF_SAMPLES = 4096


#: Kernel samples (arrivals x (2K + 1)) built at once (:func:`_arrival_kernels`),
#: about 4 MB per working array, so a list's scratch is bounded whatever its
#: length.
_ARRIVAL_KERNEL_BLOCK_ELEMS = 500_000


#: How an arrival-list record that leaves arrivals out is widened, on the
#: plain functions (:func:`arrival_impulse_response`,
#: :func:`arrival_grid_impulse_response`).
_ARRIVAL_RECORD_REMEDY = ("Lengthen n_samples= or move t_start=, or leave "
                          "both unset to size the record from the arrivals.")


def _require_phase_offset(phase_offset, who: str) -> float:
    """``phase_offset`` as one finite real angle (rad), or a
    ``ConfigurationError`` naming ``who``."""
    value = real_scalar_or_none(phase_offset)
    if value is None or not np.isfinite(value):
        raise ConfigurationError(
            f"{who}: phase_offset must be one finite angle in radians; got "
            f"{phase_offset!r}.",
            remediation="Pass the phase (rad) added to every arrival, or "
                        "leave it at 0.")
    return value


def _whole_number_or_none(value) -> Optional[int]:
    """``value`` as an ``int`` when it is one finite whole real number (an
    integer, or a float with no fraction), ``None`` otherwise — a bool, a
    string, a sequence, NaN, 2.5 (:func:`real_scalar_or_none`)."""
    number = real_scalar_or_none(value)
    if number is None or not np.isfinite(number) or number != int(number):
        return None
    return int(number)


def _arrival_band(band, fs: float, who: str) -> Tuple[float, float]:
    """``band`` of an arrival list's impulse response as ``(low, high)``
    Hz — two real numbers by the package's one rule (a bool in the pair is
    not one) with ``0 <= low < high <= fs/2`` — or a ``ConfigurationError``
    naming ``who``."""
    pair = real_array_or_none(band)
    if pair is not None and pair.shape == (2,):
        low, high = float(pair[0]), float(pair[1])
        if 0.0 <= low < high <= fs / 2.0:
            return low, high
    raise ConfigurationError(
        f"{who}: band={band!r} must be two numbers (low, high) in Hz "
        f"inside [0, sample_rate/2 = {fs / 2.0:g}] with low < high.",
        remediation="Pass band=(low, high) in Hz inside that range, or None "
                    "for 0 Hz to the Nyquist frequency.")


def _record_arrival_notice(omitted: int, cells: int, n_cells: int,
                           omitted_power: float, total_power: float,
                           record_s: float, latest_s: float, who: str,
                           remedy: str) -> str:
    """The text for arrival-list records that leave ``omitted`` arrivals
    out, in ``cells`` of the ``n_cells`` cells, carrying ``omitted_power``
    of the ``total_power`` the lists receive; ``latest_s`` is how far into
    its record the latest arrival lies."""
    where = (f"the record ({record_s:g} s) does" if n_cells == 1
             else f"the records ({record_s:g} s) of {cells} of {n_cells} "
                  f"cells do")
    return (f"{who}: {where} not hold every arrival: "
            f"{omitted} arrival(s) fall outside and are left out — on the "
            f"record each would leave only a cut fragment of its impulse — "
            f"carrying {_energy_share_words(omitted_power, total_power)}; "
            f"the latest arrival lies {latest_s:g} s into its record. "
            f"{remedy}")


def _energy_share_words(part: float, total: float) -> str:
    """``part`` of ``total`` received energy in words: all of it, a level
    in dB, or none measurable."""
    share = part / total if total > 0.0 else 0.0
    if share >= 1.0 - 1e-12:
        return "all of the received energy"
    if share > 0.0:
        return f"{10.0 * np.log10(share):.0f} dB of the received energy"
    return "no measurable energy"


def arrival_impulse_response(amplitudes, delays, *, sample_rate: float,
                             delays_imag=None, phases=None,
                             phase_offset: float = 0.0, trace_frequency=None,
                             absorption=None, t_start=None, n_samples=None,
                             band=None):
    """Real impulse response (taps) of a discrete multipath arrival list, at
    exactly ``sample_rate``: the channel :func:`apply_impulse_response`
    plays a waveform through.

    The taps of :func:`arrival_transfer_function` — each path's amplitude,
    phase and ``Im τ`` absorption, scaled to every frequency by
    ``trace_frequency`` and ``absorption`` — on the record: every path is a
    band-limited impulse at its delay, at its phase, filtered by its
    absorption. A phase other than 0 or pi makes that impulse
    ``cos φ·sinc − sin φ·(Hilbert sinc)``: the in-phase part of every path
    is built on a record of its own about it (:func:`_arrival_kernels`, at
    most 8193 samples, so a list costs its arrivals and not its span), and
    the quadrature part is rotated once on the whole record, by
    ``−i·sign(f)`` on its ``rfft`` grid, the Nyquist line's cosine kept on
    an even record. The Hilbert part falls only as ``1/(π n)`` over ``n``
    samples, so the record keeps a margin either side of the arrivals —
    their delay span, and at least 256 samples — and tapers it to zero with
    a half Hann window from the first and last arrival out to the record's
    ends. A record that ``t_start`` or ``n_samples`` states with less margin
    than that is built padded to the margin and cut back to its own
    samples, so its taps are the response's samples there; it then costs
    the padded record, up to the record sized from its arrivals — a window
    cut tight about 1000 arrivals over 20 s at 48 kHz takes about what the
    default record takes (195 against 241 MB, 2.7 s each). A path on a
    sample at phase 0 is one tap.

    **A finite record cannot rotate the phase of content below about
    ``1/margin``.** The Hilbert part of a path is the kernel ``1/(π t)``,
    and cut to the record it is a filter whose ``−i·sign(f)`` is reached
    only above a transition about ``1/margin`` wide at 0 Hz (the window
    method of FIR design, Abraham sect. 4.9.2) — 188 Hz at the 256-sample
    margin at 48 kHz. Played through these taps
    (:func:`apply_impulse_response`) and measured against the exact
    product ``X·H``: a 20 ms 10 kHz pulse through 20 paths over 50 ms at
    48 kHz within 5.1e-8 of the peak when each path loses up to about 1 dB
    at 10 kHz (``|Im τ|`` up to 2e-6 s), within 1.9e-7 when it loses up to
    about 11 dB (2e-5 s; six random lists each, Thorp's law or linear in
    frequency), white noise to the Nyquist frequency through three paths
    over 110 ms within 6e-4 to 2.5e-3 (five lists, on samples or between),
    but a 4-cycle 50 Hz pulse through a lone quarter-turn arrival 0.55 of
    the peak off. For a
    waveform with content that low, :func:`simulate_arrival_reception` (or
    :meth:`Arrivals.to_time_series <uacpy.core.results.Arrivals.
    to_time_series>`) gives the reception itself, the phase rotation taken
    on the waveform, exactly.

    Parameters
    ----------
    amplitudes, delays : array_like
        Arrival magnitudes (non-negative; a sign belongs in ``phases``
        as ``pi``) and real travel times (s).
    sample_rate : float
        Rate (Hz) of the taps.
    delays_imag, phases, phase_offset, trace_frequency, absorption
        As :func:`arrival_transfer_function` takes them.
    t_start : float, optional
        Time (s) of ``h[0]``; ``None`` is the first arrival less the margin.
    n_samples : int, optional
        Length of the record; ``None`` reaches the last arrival plus the
        margin. An arrival outside the record that ``t_start`` and
        ``n_samples`` state is left out — on the record it would leave only
        a cut fragment of its impulse, its phase rotation wrapped round the
        record — and a ``NumericsWarning`` names the share of the received
        energy left out.
    band : (float, float), optional
        ``(low, high)`` in Hz: ``H`` is kept there and zero outside it.
        ``None``: 0 Hz to the Nyquist frequency.

    Returns
    -------
    (ndarray, ndarray)
        ``(t, h)``: the time (s) of each tap and the taps. An empty list
        returns zeros from ``t_start`` (0 when not given): a receiver no
        path reaches is silent.

    Raises
    ------
    ConfigurationError
        The refusals of :func:`arrival_transfer_function`, a non-positive
        ``sample_rate`` or ``n_samples``, a non-finite ``t_start``, and a
        ``band`` outside ``[0, sample_rate/2]`` or keeping fewer than two
        bins of the record's grid.
    """
    who = "arrival_impulse_response"
    columns = _arrival_columns(amplitudes, delays, delays_imag,
                               phases, who)
    if t_start is not None:
        t_start = _require_start_time(
            t_start, who,
            remediation="Pass the time (s) of h[0], or None to start one "
                        "margin before the first arrival.")
    t, h = _arrival_grid_impulse_response(
        [[columns if columns[1].size else None]], sample_rate=sample_rate,
        phase_offset=phase_offset, trace_frequency=trace_frequency,
        absorption=absorption, t_start=t_start, n_samples=n_samples,
        band=band, who=who)
    return np.asarray(t)[0, 0], h[0, 0]


def arrival_grid_impulse_response(cells, *, sample_rate: float,
                                  phase_offset: float = 0.0,
                                  trace_frequency=None, absorption=None,
                                  t_start=None, n_samples=None, band=None):
    """The impulse response (taps) of every receiver of a grid, each from
    its own arrival list (:func:`arrival_impulse_response` per cell).

    Parameters
    ----------
    cells : sequence of sequences of mappings
        ``cells[i][j]``, the arrival list of receiver ``(i, j)``, as
        :func:`arrival_grid_transfer_function` takes it.
    sample_rate : float
        Rate (Hz) of the taps.
    phase_offset, trace_frequency, absorption
        As :func:`arrival_grid_transfer_function` takes them.
    t_start : float or array_like, optional
        Time (s) of ``h[0]``: one number for every cell, one per cell
        (``(n_i, n_j)``), or ``None`` for each cell's first arrival less
        its margin.
    n_samples : int, optional
        Length of every cell's record; ``None`` reaches the latest cell's
        last arrival plus its margin.
    band : (float, float), optional
        As on :func:`arrival_impulse_response`.

    Returns
    -------
    (ndarray, ndarray)
        ``(t, h)``: ``h`` of shape ``(n_i, n_j, n_samples)``, zeros in a
        cell whose arrival list is empty (the empty sum, as
        :func:`arrival_impulse_response` returns for it); ``t`` the time of
        each tap, 1-D for a number ``t_start`` and ``(n_i, n_j, n_samples)``
        otherwise.
    """
    t, h = _arrival_cells_impulse_response(
        cells, sample_rate=sample_rate, phase_offset=phase_offset,
        trace_frequency=trace_frequency, absorption=absorption,
        t_start=t_start, n_samples=n_samples, band=band,
        who="arrival_grid_impulse_response")
    if np.ndim(t_start) == 0 and t_start is not None:
        return np.asarray(t)[0, 0], h
    return t, h


def _arrival_cells_impulse_response(cells, *, sample_rate, phase_offset=0.0,
                                    trace_frequency=None, absorption=None,
                                    t_start=None, n_samples=None, band=None,
                                    who: str,
                                    remedy: str = _ARRIVAL_RECORD_REMEDY,
                                    length_knob: str = "n_samples="):
    """``(t, h)`` of :func:`arrival_grid_impulse_response`, ``t`` per cell,
    its refusals and notices reported as ``who`` with ``remedy`` for a
    record that leaves arrivals out and ``length_knob`` the argument that
    states the record's length."""
    columns = [[None if _arrival_count(cell, who,
                                       f" at cells[{i}][{j}]") == 0
                else _arrival_columns(cell['amplitudes'], cell['delays'],
                                      cell.get('delays_imag'),
                                      cell.get('phases'), who)
                for j, cell in enumerate(row)]
               for i, row in enumerate(cells)]
    return _arrival_grid_impulse_response(
        columns, sample_rate=sample_rate, phase_offset=phase_offset,
        trace_frequency=trace_frequency, absorption=absorption,
        t_start=t_start, n_samples=n_samples, band=band, who=who,
        remedy=remedy, length_knob=length_knob)


def _arrival_grid_impulse_response(columns, *, sample_rate, phase_offset,
                                   trace_frequency, absorption, t_start,
                                   n_samples, band, who,
                                   remedy: str = _ARRIVAL_RECORD_REMEDY,
                                   length_knob: str = "n_samples="):
    """``(t, h)`` of :func:`arrival_grid_impulse_response` from validated
    arrival columns (:func:`_arrival_columns`, ``None`` for a cell no
    arrival reached, left at zero); ``t`` always per cell.

    An arrival outside its cell's record is left out of that cell's taps
    and a ``NumericsWarning`` names the share of the received energy left
    out (:func:`_record_arrival_notice`, ending with ``remedy``): on the
    record such an arrival would leave only a cut fragment of its impulse,
    its phase rotation wrapped round the record. A record with less than
    the margin (the arrivals' delay span, at least 256 samples) either side
    of its arrivals is built padded to it and cut back. ``length_knob`` is
    the argument a refusal names for the record's length."""
    fs = require_positive_finite_scalar(sample_rate, who, "sample_rate",
                                        " Hz")
    phase_offset = _require_phase_offset(phase_offset, who)
    if band is not None:
        band = _arrival_band(band, fs, who)
    n_i, n_j = len(columns), len(columns[0])
    first = np.full((n_i, n_j), np.nan)
    last = np.full((n_i, n_j), np.nan)
    for i in range(n_i):
        for j in range(n_j):
            if columns[i][j] is not None:
                first[i, j] = columns[i][j][1].min()
                last[i, j] = columns[i][j][1].max()
    reached = np.isfinite(first)
    margins = np.where(reached, np.vectorize(arrival_ir_margin)(
        np.where(reached, last - first, 0.0), fs), 0.0)
    if t_start is None:
        starts = np.where(reached, first - margins, 0.0)
    else:
        starts_in = _per_cell_start_times(
            t_start, who,
            remediation="Pass the time (s) of h[0]: one number, or one per "
                        "cell of the grid; None starts each cell one margin "
                        "before its first arrival.")
        try:
            starts = np.array(np.broadcast_to(starts_in, (n_i, n_j)))
        except ValueError:
            raise ConfigurationError(
                f"{who}: t_start of shape {starts_in.shape} is neither one "
                f"number nor one per cell of the {n_i} x {n_j} grid.") from None
    if n_samples is None:
        if reached.any():
            reach = np.where(reached, last + margins - starts, -np.inf)
            # The 1e-9 keeps floating-point dust from adding a sample.
            n_samples = int(np.ceil(float(reach.max()) * fs - 1e-9)) + 1
            if n_samples < 1:
                raise ConfigurationError(
                    f"{who}: t_start is after the last arrival and its "
                    f"margin ({float(np.nanmax(last + margins)):g} s), so a "
                    f"record sized from the arrivals has no length.",
                    remediation=f"Start the record before the arrivals, "
                                f"or state its length ({length_knob}).")
        else:
            n_samples = 1
    count = _whole_number_or_none(n_samples)
    if count is None or count < 1:
        raise ConfigurationError(
            f"{who}: n_samples must be a whole number of samples, at least "
            f"1; got {n_samples!r}.",
            remediation="Pass the record's length in samples, or None to "
                        "size it from the arrivals.")
    n_samples = count
    elapsed = np.arange(n_samples) / fs
    times = starts[..., None] + elapsed
    taps = np.zeros((n_i, n_j, n_samples), dtype=float)
    filled = [(i, j) for i in range(n_i) for j in range(n_j)
              if columns[i][j] is not None]
    held = _arrivals_on_records(columns, filled, starts, n_samples, fs,
                                trace_frequency, who, remedy)
    if not filled:
        return times, taps
    for i, j in filled:
        delays = held[i, j][1]
        if not delays.size:
            taps[i, j] = 0.0
            continue
        lead = (delays.min() - starts[i, j]) * fs
        end = (delays.max() - starts[i, j]) * fs
        # A record that keeps less than the margin either side of its
        # arrivals — a start or a length the caller stated close to them —
        # is built on the record padded to the margin, in whole samples, and
        # cut back to it: its taps are then the impulse response's own
        # samples, the quadrature tails rotated and tapered off the record.
        room = arrival_ir_margin(delays.max() - delays.min(), fs) * fs
        before = max(0, int(np.round(room - lead)))
        after = max(0, int(np.round(room - (n_samples - 1 - end))))
        n_padded = n_samples + before + after
        # A path at phase phi is A cos(phi) g + Hilbert(-A sin(phi) g), g its
        # band-limited impulse: the in-phase part as kernels, the rotation
        # once on the whole record, whose 1/(pi n) tails it keeps.
        h_c, h_s = _arrival_kernels(
            held[i, j], starts[i, j], n_padded, sample_rate=fs,
            phase_offset=phase_offset, trace_frequency=trace_frequency,
            absorption=absorption, lead=before)
        # The Hilbert transform's -i sign(f) on the record's rfft grid: no
        # quadrature at 0 Hz or on the Nyquist line of an even record
        # (Abraham sect. 7.3.1, eq. 7.29), where a real response holds its
        # cosine.
        freqs = np.arange(n_padded // 2 + 1) * (fs / n_padded)
        rotate = np.full(freqs.size, -1j)
        rotate[0] = 0.0
        if n_padded % 2 == 0:
            rotate[-1] = 0.0
        spectrum = np.fft.rfft(h_c) + rotate * np.fft.rfft(h_s)
        if band is not None:
            # Read on the grid the response is built on, the Nyquist line
            # included on an even record.
            kept = (freqs >= band[0]) & (freqs <= band[1])
            if np.count_nonzero(kept) < 2:
                raise ConfigurationError(
                    f"{who}: band=({band[0]:g}, {band[1]:g}) Hz keeps "
                    f"{int(np.count_nonzero(kept))} bin(s) of the "
                    f"{fs / n_padded:g} Hz grid of the {n_padded}-sample "
                    f"record the response is built on; widen it.")
            spectrum[~kept] = 0.0
        # The taper runs from the samples bracketing the first and last
        # arrival, so an arrival a hair inside the record's first or last
        # sample keeps its weight; a position within rounding of a sample
        # (a few ulps of the times: 5e-9 samples at 1000 s and 48 kHz) is
        # on it.
        dust = (64.0 * np.finfo(float).eps * fs
                * max(abs(starts[i, j]), float(np.abs(delays).max())))
        record = np.fft.irfft(spectrum, n_padded) * _arrival_record_taper(
            np.floor(lead + before + dust), np.ceil(end + before - dust),
            n_padded)
        taps[i, j] = record[before:before + n_samples]
    return times, taps


def _arrivals_on_records(columns, filled, starts, n_samples: int, fs: float,
                         trace_frequency, who: str, remedy: str) -> dict:
    """``{(i, j): columns}``: each filled cell's arrivals that lie on its
    ``n_samples``-sample record from ``starts[i, j]``, warning once (a
    ``NumericsWarning``, :func:`_record_arrival_notice`) when any cell's
    record leaves arrivals out. Each arrival is weighed by its received
    power at ``trace_frequency`` (its magnitude alone without one)."""
    omega_t = (0.0 if trace_frequency is None
               else 2.0 * np.pi * float(trace_frequency))
    held = {}
    omitted = short = 0
    lost = received = latest = 0.0
    for i, j in filled:
        amps, delays, delays_imag, phases = columns[i][j]
        position = (delays - starts[i, j]) * fs
        # The 1e-9 keeps an arrival on the first or last sample inside.
        inside = (position > -1e-9) & (position < n_samples - 1 + 1e-9)
        with np.errstate(over='ignore'):
            power = (amps * np.exp(omega_t * delays_imag)) ** 2
        received += float(power.sum())
        latest = max(latest, float(position.max()) / fs)
        if not inside.all():
            omitted += int((~inside).sum())
            short += 1
            lost += float(power[~inside].sum())
        held[i, j] = (amps[inside], delays[inside], delays_imag[inside],
                      phases[inside])
    if omitted:
        warnings.warn(
            _record_arrival_notice(omitted, short, len(filled), lost,
                                   received, n_samples / fs, latest, who,
                                   remedy),
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return held


def _arrival_record_taper(first, last, n_samples: int) -> np.ndarray:
    """The window an arrival list's record is tapered by: 1 from the first
    arrival (sample ``first``) to the last (sample ``last``), a half Hann
    window from each out to the record's end, so the band-limited tails of
    the paths reach zero before the record does. ``first`` and ``last`` may
    be arrays (one window per pair, on a leading axis); one arrival's
    kernel is ``first = last`` at its position (:func:`_arrival_kernels`).
    Only the two half windows are computed, into a window of ones."""
    first = np.asarray(first, dtype=float)
    last = np.asarray(last, dtype=float)
    window = np.ones(first.shape + (n_samples,))
    rows = window.reshape(-1, n_samples)
    for row, lead, end in zip(rows, first.ravel(), last.ravel()):
        if lead > 0.0:
            k = np.arange(min(int(np.ceil(lead)), n_samples), dtype=float)
            row[:k.size] = 0.5 * (1.0 - np.cos(np.pi * k / lead))
        tail_len = (n_samples - 1) - end
        if tail_len > 0.0:
            from_sample = max(int(np.floor(end)) + 1, 0)
            k = np.arange(from_sample, n_samples, dtype=float)
            row[from_sample:] = 0.5 * (1.0 - np.cos(
                np.pi * ((n_samples - 1) - k) / tail_len))
    return window


def _quadrature_source(sts: np.ndarray) -> Tuple[int, np.ndarray]:
    """``(pad, x − i·x̂)``: the waveform ``x`` and its Hilbert transform
    ``x̂`` on a record padded ``pad`` samples either side, its sample 0 the
    waveform's sample ``−pad``.

    ``x̂`` is ``−i·sign(f)·X(f)`` in the frequency domain (Abraham, sect.
    7.3.1, eq. 7.29: ``1/(π t) ↔ −j sign(f)``), taken on the padded record,
    where it is exact at every bin but 0 Hz and the Nyquist frequency (no
    quadrature there); its only error is its own tail beyond the padding,
    folded back by the record. That tail falls the faster the less
    content the waveform has near 0 Hz, and its extent scales with the
    waveform: ``pad`` is :data:`_HILBERT_PAD_WAVEFORMS` waveform lengths,
    at least
    :data:`~uacpy.acoustic_signal._synthesis.ARRIVAL_IR_MARGIN_SAMPLES` and
    at most :data:`_HILBERT_PAD_MAX_SAMPLES` samples."""
    n = sts.size
    pad = int(min(max(ARRIVAL_IR_MARGIN_SAMPLES, _HILBERT_PAD_WAVEFORMS * n),
                  _HILBERT_PAD_MAX_SAMPLES))
    n_fft = _sp_fft.next_fast_len(n + 2 * pad)
    padded = np.zeros(n_fft)
    padded[pad:pad + n] = sts
    spectrum = _sp_fft.rfft(padded)
    spectrum *= -1j
    spectrum[0] = 0.0
    if n_fft % 2 == 0:
        spectrum[-1] = 0.0
    return pad, padded - 1j * _sp_fft.irfft(spectrum, n_fft)


def _arrival_kernels(columns, start: float, n_samples: int, *,
                     sample_rate: float, phase_offset: float,
                     trace_frequency, absorption, lead: int = 0):
    """``(h_c, h_s)`` on the ``n_samples``-sample record that opens
    ``lead`` whole samples before ``start``: every arrival's in-phase part
    ``A·cos φ`` (``h_c``) and quadrature part
    ``−A·sin φ`` (``h_s``) as a band-limited impulse at its delay, filtered
    by its absorption (JKPS eq. 8.30: a path is ``A·(cos φ·x − sin φ·x̂)``).

    Each arrival's impulse is built on a record of its own, ``2K + 1``
    samples centred on it (``K`` = :data:`_ARRIVAL_KERNEL_HALF_SAMPLES`, or
    less to fit the record, at least 1): the arrival's transfer function on
    that record's ``rfft`` grid — an odd record has no Nyquist line, so a
    path on a sample is one tap exactly — tapered to zero at the record's
    ends by a Hann window about the arrival (:func:`_arrival_record_taper`),
    the window method of FIR design (Abraham sect. 4.9.2), and added into
    the record, a block of :data:`_ARRIVAL_KERNEL_BLOCK_ELEMS` kernel
    samples at a time. A list costs
    ``n_arrivals × K`` whatever its span; the samples of an impulse beyond
    the record are left out. Neither part carries the ``sign(f)`` of a
    phase rotation, so neither has the Hilbert kernel's slow ``1/(π n)``
    tail at 0 Hz: a reception rotates the waveform instead
    (:func:`_quadrature_source`), an impulse response the whole record's
    ``h_s`` (:func:`_arrival_grid_impulse_response`)."""
    amps, delays, delays_imag, phases = columns
    fs = float(sample_rate)
    # At least one sample either side, so the window is 1 at the arrival.
    half = int(min(_ARRIVAL_KERNEL_HALF_SAMPLES,
                   max((n_samples - 1) // 2, 1)))
    n_kernel = 2 * half + 1
    bins = np.arange(half + 1)
    # The exponent is Im tau times a rate per frequency on every law
    # (_arrival_phasor_sums), consulted once for the kernels' grid.
    rate = arrival_absorption_exponent(
        np.ones(1), bins * (fs / n_kernel), trace_frequency=trace_frequency,
        absorption=absorption)[0]
    position = (np.asarray(delays, dtype=float) - start) * fs + lead
    kernel_start = np.round(position).astype(np.int64) - half
    # Each arrival on its own record, within half a sample of its centre.
    centre = position - kernel_start
    angle = np.asarray(phases, dtype=float) + float(phase_offset)
    weights = (amps * np.cos(angle), -amps * np.sin(angle))
    offsets = np.arange(n_kernel)
    h_c, h_s = np.zeros(n_samples), np.zeros(n_samples)
    step = max(1, _ARRIVAL_KERNEL_BLOCK_ELEMS // n_kernel)
    for a in range(0, position.size, step):
        block = slice(a, a + step)
        phasors = np.empty((centre[block].size, bins.size), dtype=complex)
        phasors.real = np.outer(delays_imag[block], rate)
        phasors.imag = np.outer(centre[block], (-2.0 * np.pi / n_kernel) * bins)
        with np.errstate(over='ignore', invalid='ignore'):
            np.exp(phasors, out=phasors)
        window = _arrival_record_taper(centre[block], centre[block],
                                       n_kernel)
        index = kernel_start[block, None] + offsets
        inside = (index >= 0) & (index < n_samples)
        for out, weight in zip((h_c, h_s), weights):
            kernel = np.fft.irfft(phasors * weight[block, None], n_kernel)
            out += np.bincount(index[inside], weights=(kernel * window)[inside],
                               minlength=n_samples)[:n_samples]
    return h_c, h_s


def simulate_arrival_reception(amplitudes, delays, *,
                               source_waveform: np.ndarray,
                               sample_rate: float,
                               trace_frequency: Optional[float] = None,
                               delays_imag=None, phases=None,
                               output_duration: Optional[float] = None,
                               t_start: Optional[float] = None,
                               phase_offset: float = 0.0,
                               fractional: bool = True,
                               absorption=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Received waveform from a sparse arrival list, with carrier and absorption.

    Every arrival delivers the source waveform at its delay, rotated by its
    phase from caustics and boundary reflections and filtered by its volume
    absorption — ``A_r S(t − τ) − A_i Ŝ(t − τ)`` with ``Ŝ`` the Hilbert
    transform (JKPS eq. 8.30; the Bellhop User Guide, Sec. 9). ``Ŝ`` is
    taken in the frequency domain, ``−i·sign(f)·S(f)``, on a record padded
    four waveform lengths either side (at least 256 samples, at most
    2**20), and ``S`` and ``Ŝ`` go through the arrival list's in-phase and
    quadrature taps — every path's ``A_r`` and ``−A_i`` as a band-limited
    impulse at its delay, filtered by its absorption — in one linear
    convolution (:func:`apply_impulse_response`), placed on the output
    window. Each path's impulse is built on a record of its own of at most
    8193 samples about it (:func:`_arrival_kernels`), so a list costs its
    arrivals and not its delay span: 1000 arrivals over 4 s at 48 kHz take
    about 1 s. The phase rotation is the waveform's, exact at every
    frequency it holds, not a finite response's (which cannot rotate
    content below about one over its length; see
    :func:`arrival_impulse_response`). Measured against the exact product
    ``X·H`` on a 2**21-sample record at 48 kHz: a 20 ms 10 kHz pulse
    through 20 paths over 50 ms within 1.2e-8 of the peak when each path
    loses up to about 1 dB at 10 kHz (``|Im τ|`` up to 2e-6 s), within
    9.6e-8 when it loses up to about 11 dB (2e-5 s; six random lists each,
    Thorp's law or linear in frequency); a 4-cycle
    50 Hz pulse through a lone quarter-turn arrival 1.7e-5, through two
    paths 5 ms apart 3.3e-5; a 2-cycle 20 Hz pulse 1.7e-4; white noise to
    the Nyquist frequency through a path on a sample at phase 0 to
    rounding, through 3 or 50 paths between samples 1.4e-3 or 4.4e-3 (the
    impulses' kernels near the Nyquist frequency). What is left at low
    frequency is ``Ŝ``'s own tail beyond the padding, which falls the
    faster the less content the waveform has near 0 Hz. An echo the window
    omits — no sample of its waveform on it, as the window notice counts —
    is left out whole, the precursor and tail of its ``Ŝ`` with it.

    Parameters
    ----------
    amplitudes, delays : array_like
        Arrival magnitudes (non-negative; a sign belongs in ``phases``
        as ``pi``) and real travel times (s).
    delays_imag : array_like, optional
        Imaginary travel times (s), carrying the volume absorption as
        ``exp(omega * Im tau)`` at ``trace_frequency``. ``None`` is a
        lossless list. Each arrival is filtered by that absorption at every
        frequency, scaled from ``trace_frequency`` by
        :func:`~uacpy.core.absorption.arrival_absorption_exponent` — the
        same law :func:`arrival_transfer_function` applies, so a pulse and a
        transfer function built from one arrival list agree.
    phases : array_like, optional
        Per-arrival phase (radians) from caustics and boundary reflections.
        ``None`` is zeros.
    source_waveform : ndarray
        Source waveform (1-D), used as-is.
    sample_rate : float
        Sample rate in Hz.
    trace_frequency : float, optional
        Frequency (Hz) the arrivals were traced at, where ``Im tau`` is
        exact (Bellhop's carrier). Required when ``delays_imag`` holds a
        non-zero value (refused without it); a lossless list needs none.
    absorption : Absorption or AbsorptionCoefficient, optional
        The water-column law ``Im tau`` carries (or a measured table). A
        Thorp or Francois-Garrison law scales the absorption as
        ``α(f)/α(trace_frequency)`` (exact); otherwise it scales linearly
        in ``f``.
    output_duration : float, optional
        Output time window in seconds.  If *None*, it runs to the latest
        arrival plus one source duration, where the last copy of the
        waveform ends, plus a quiet tail of one source duration capped at
        the arrivals' delay span.
    t_start : float, optional
        Start time for the output.  If *None*, set to just before the
        earliest arrival.
    phase_offset : float, optional
        Constant phase (radians) added to every arrival. ``0.0`` (default):
        a :class:`Bellhop` ARRIVALS result already carries the line-source
        ``exp(-i*pi/4)`` in its ``phases``.
    fractional : bool, optional
        Place each echo at its exact delay (default: the band-limited
        impulse at that delay). ``False`` rounds every delay to the nearest
        sample of the output: an error up to half a sample, which is tens
        of degrees of carrier phase at any frequency a modem uses, summed
        coherently into the interference pattern. Continuous placement is
        what the formulation asks for — COA eq. 8.30 shifts the waveform by
        ``t - tau(s)``, not by a whole number of samples.

    Warns
    -----
    NumericsWarning
        When an echo falls outside the window or straddles one of its edges
        — the window cuts such an echo off rather than folding it, and the
        record reads as complete.

    Returns
    -------
    time_vector : ndarray
        Time of each sample (s).
    rts : ndarray
        Received time series, shape ``(n_samples,)``. An empty arrival
        list returns zeros: a receiver no path reaches is silent, here and
        in a cell of :func:`simulate_arrival_grid`.

    The axis comes FIRST, as it does from ``impulse_response``,
    ``simulate_reception``, :func:`delayandsum` and the two transform
    pairs. Both are real arrays of the same length, so the wrong order is
    silent — which is why the package has one order.

    References
    ----------
    Bellhop User Guide, Section 9.3
    Original MATLAB code: delayandsum.m by M. B. Porter, 8/96
    """
    return _simulate_arrival_reception(
        amplitudes,
        delays,
        source_waveform=source_waveform,
        sample_rate=sample_rate,
        trace_frequency=trace_frequency,
        delays_imag=delays_imag,
        phases=phases,
        output_duration=output_duration,
        t_start=t_start,
        phase_offset=phase_offset,
        fractional=fractional,
        absorption=absorption)


def _simulate_arrival_reception(
    amplitudes,
    delays,
    *,
    source_waveform: np.ndarray,
    sample_rate: float,
    trace_frequency: Optional[float] = None,
    delays_imag=None,
    phases=None,
    output_duration: Optional[float] = None,
    t_start: Optional[float] = None,
    phase_offset: float = 0.0,
    fractional: bool = True,
    absorption=None,
    who: str = "simulate_arrival_reception",
) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`simulate_arrival_reception` reporting its refusals and its
    window notice as ``who``."""
    sample_rate, fc, sts = _reception_waveform(source_waveform, sample_rate,
                                               trace_frequency, who)
    phase_offset = _require_phase_offset(phase_offset, who)
    if t_start is not None:
        t_start = _require_start_time(
            t_start, who,
            remediation="Pass the time (s) of the first output sample, or "
                        "None to start just before the earliest arrival.")
    if output_duration is not None:
        output_duration = require_positive_finite_scalar(
            output_duration, who, "output_duration", " s",
            why=" Pass the window's length, or None to size it from the "
                "arrivals.")
    columns = _arrival_columns(amplitudes, delays, delays_imag,
                               phases, who)
    _refuse_absorption_without_trace_frequency([columns], fc, who)
    delays = columns[1]
    if delays.size == 0:
        if output_duration is not None:
            nrts = int(np.ceil(output_duration * sample_rate))
        else:
            nrts = int(_EMPTY_TRACE_SECONDS * sample_rate)
        t0 = 0.0 if t_start is None else float(t_start)
        return t0 + np.arange(nrts) / sample_rate, np.zeros(nrts)

    t_start, output_duration = _reception_window(
        float(np.min(delays)), float(np.max(delays)), sts.size, sample_rate,
        t_start, output_duration, who)
    nrts = int(np.ceil(output_duration * sample_rate))
    traces, counts = _receptions_on_window(
        [columns], sts, sample_rate=sample_rate, trace_frequency=fc,
        absorption=absorption, t_start=t_start, n_samples=nrts,
        phase_offset=phase_offset, fractional=fractional, who=who)
    notice = _echo_window_notice(counts, t_start, output_duration, who=who)
    if notice is not None:
        warnings.warn(notice, NumericsWarning,
                      skip_file_prefixes=USER_FRAME_SKIP)
    return t_start + np.arange(nrts) * (1.0 / sample_rate), traces[0]


def _receptions_on_window(columns, sts: np.ndarray, *, sample_rate: float,
                          trace_frequency: float, absorption, t_start: float,
                          n_samples: int, phase_offset: float,
                          fractional: bool, who: str):
    """``(traces, counts)``: the ``n_samples`` record from ``t_start`` each
    arrival list of ``columns`` (:func:`_arrival_columns`) delivers of the
    waveform ``sts``, one row per list, and the window counts of
    :func:`_echo_window_counts` over every list, with the earliest echo's
    first sample as ``earliest_s`` (s).

    Each list is its own reception: the waveform ``x`` and its Hilbert
    transform ``x̂`` (:func:`_quadrature_source`) through the list's
    in-phase and quadrature taps (:func:`_arrival_kernels`) on its
    own record, which starts a whole number of samples from ``t_start`` —
    its first arrival less its margin, rounded down — so the convolution
    (:func:`apply_impulse_response`) lands on the window's own samples,
    and a list's trace does not depend on the others it is computed
    beside. An echo the counts report omitted — no sample of its waveform
    on the window — is left out of the reception too, so the Hilbert
    transform's tails, which reach either side of every echo at a phase
    other than 0 or pi, do not carry an omitted echo into the window."""
    fs = sample_rate
    if not fractional:
        columns = [(c[0], t_start + np.round((c[1] - t_start) * fs) / fs,
                    c[2], c[3]) for c in columns]
    traces = np.zeros((len(columns), n_samples))
    counts: dict = {}
    omega_c = (0.0 if trace_frequency is None
               else 2.0 * np.pi * float(trace_frequency))
    pad, source = _quadrature_source(sts)
    for row, (amps, delays, delays_imag, phases) in enumerate(columns):
        # The window report reads each echo's WAVEFORM extent from its
        # delay, and weighs it by its received power at the trace frequency.
        nominal = np.round((delays - t_start) * fs).astype(int)
        powers = (amps * np.exp(omega_c * delays_imag)) ** 2
        part = _echo_window_counts(nominal, nominal + sts.size, powers,
                                   n_samples)
        part['earliest_s'] = t_start + part.pop('earliest_start') / fs
        _merge_echo_counts(counts, part)
        held = (nominal + sts.size > 0) & (nominal < n_samples)
        if not held.any():
            continue
        amps, delays, delays_imag, phases = (
            amps[held], delays[held], delays_imag[held], phases[held])
        first, last = float(delays.min()), float(delays.max())
        # Room for every arrival's kernel (_arrival_kernels) either side.
        margin = (_ARRIVAL_KERNEL_HALF_SAMPLES + 1) / fs
        offset = int(np.floor((first - margin - t_start) * fs))
        start = t_start + offset / fs
        n_taps = int(np.ceil((last + margin - start) * fs - 1e-9)) + 1
        h_c, h_s = _arrival_kernels(
            (amps, delays, delays_imag, phases), start, n_taps,
            sample_rate=fs, phase_offset=phase_offset,
            trace_frequency=trace_frequency, absorption=absorption)
        _, received = apply_impulse_response(h_c + 1j * h_s,
                                             source_waveform=source,
                                             sample_rate=fs)
        received = received.real
        # The source's first sample is the waveform's sample -pad.
        offset -= pad
        lo, hi = max(0, offset), min(n_samples, offset + received.size)
        if lo < hi:
            traces[row, lo:hi] = received[lo - offset:hi - offset]
    return traces, counts


def _reception_waveform(source_waveform, sample_rate, trace_frequency,
                        who: str) -> Tuple[float, float, np.ndarray]:

    """``(sample_rate, trace_frequency, waveform)`` of a reception,
    validated: positive finite frequencies (``trace_frequency`` may be
    ``None``, :func:`_refuse_absorption_without_trace_frequency`) and a
    real, finite 1-D waveform (float)."""
    sample_rate = require_positive_finite_scalar(sample_rate, who,
                                                 "sample_rate", " Hz")
    fc = (None if trace_frequency is None
          else require_positive_finite_scalar(trace_frequency, who,
                                              "trace_frequency", " Hz"))
    sts = require_finite_signal(source_waveform, who, "source_waveform")
    if sts.ndim != 1 or np.iscomplexobj(sts):
        raise ConfigurationError(
            f"{who}: source_waveform must be a real 1-D waveform; got "
            f"{'a complex' if np.iscomplexobj(sts) else 'a'} array of shape "
            f"{sts.shape}. Each arrival's phase is applied to its analytic "
            f"signal here, so pass the real waveform.")
    return sample_rate, fc, sts.astype(float)


def _refuse_absorption_without_trace_frequency(columns, trace_frequency,
                                               who: str) -> None:
    """Refuse a reception of arrival lists (:func:`_arrival_columns`) that
    carry volume absorption — a non-zero ``Im tau`` — with no
    ``trace_frequency``: ``Im tau`` states the loss at the frequency the
    arrivals were traced at, and nothing else places it on the waveform's
    band. A lossless list needs none."""
    if trace_frequency is not None:
        return
    if any(np.any(np.asarray(c[2]) != 0.0) for c in columns):
        raise ConfigurationError(
            f"{who}: the arrivals carry volume absorption (a non-zero "
            f"delays_imag), which Im tau states at the frequency they were "
            f"traced at; without trace_frequency= it cannot be placed on the "
            f"waveform's band.",
            remediation="Pass trace_frequency=, the frequency (Hz) the "
                        "arrivals were traced at (Bellhop's carrier, "
                        "Arrivals.f0).")


def _arrival_columns(amplitudes, delays, delays_imag, phases,
                     who: str) -> Tuple[np.ndarray, ...]:
    """``(amplitudes, delays, delays_imag, phases)`` of an arrival list as
    1-D float arrays, validated: non-negative real magnitudes
    (:func:`_arrival_magnitudes`), finite amplitudes and delays, and one
    entry per arrival in every column (``None`` columns are zeros)."""
    amps = _arrival_magnitudes(amplitudes, who)
    delays = _arrival_column(delays, 'delays', who)
    if amps.size != delays.size:
        raise ConfigurationError(
            f"{who}: amplitudes and delays must have "
            f"the same length; got {amps.size} and {delays.size}.")
    if not (np.all(np.isfinite(amps)) and np.all(np.isfinite(delays))):
        raise ConfigurationError(
            f"{who}: amplitudes and delays must be finite; an arrival "
            f"with a non-finite amplitude or delay has no place on the "
            f"record.")
    delays_imag = (np.zeros_like(delays) if delays_imag is None
                   else _arrival_column(delays_imag, 'delays_imag', who))
    phases = (np.zeros_like(delays) if phases is None
              else _arrival_column(phases, 'phases', who))
    for name, arr in (('delays_imag', delays_imag), ('phases', phases)):
        if arr.size != delays.size:
            raise ConfigurationError(
                f"{who}: {name} must have one entry per "
                f"arrival ({delays.size}); got {arr.size}.")
    return amps, delays, delays_imag, phases


def _reception_window(min_delay: float, max_delay: float, n_source: int,
                      sample_rate: float, t_start: Optional[float],
                      output_duration: Optional[float], who: str
                      ) -> Tuple[float, float]:
    """``(t_start, output_duration)`` of a reception, each one left ``None``
    sized from the arrivals' delays ``[min_delay, max_delay]`` (s); a
    stated ``t_start`` after the last echo has ended, with the window left
    to the arrivals, is refused naming ``who``.

    Every arrival places a whole copy of the ``n_source``-sample waveform
    starting at its own delay, so the window reaches ``max_delay`` plus one
    source duration for the last copy to fit. The quiet tail after it is one
    source duration for a pulse, but never longer than the arrivals' delay
    span — a long signal through a short channel is not padded with a
    second signal length of silence — nor shorter than eight samples
    (``_FRAC_DELAY_HALF_LEN``), so a lone arrival's window does not close
    on the last sample of its copy: a path between samples is a
    band-limited impulse whose ``1/(π n)`` ring is down to 4 % of its peak
    eight samples on. The lead-in keeps the earliest arrival's leading edge
    inside the window, and the ``max(0, ...)`` stops the clock from starting
    before the emission.
    """
    src_duration = n_source / sample_rate
    if t_start is None:
        t_start = max(0.0, min_delay - 0.1 * src_duration)
    if output_duration is None:
        tail = min(src_duration, max(max_delay - min_delay,
                                     _FRAC_DELAY_HALF_LEN / sample_rate))
        output_duration = (max_delay - t_start) + src_duration + tail
        if output_duration * sample_rate < 1.0:
            end = max_delay + src_duration + tail
            raise ConfigurationError(
                f"{who}: t_start={t_start:g} s is after the last echo has "
                f"ended ({max_delay:g} s plus the {src_duration:g} s "
                f"waveform and its tail, {end:g} s), so a window sized from "
                f"the arrivals has no length.",
                remediation=f"Pass a t_start before {end:g} s, or "
                            f"output_duration= for a window of your own.")
    return float(t_start), float(output_duration)


def _merge_echo_counts(report: dict, counts: dict) -> None:
    """Add one reception's window ``counts`` into ``report``: totals summed,
    the earliest echo the earliest of all."""
    for key, value in counts.items():
        if key == 'earliest_s':
            report[key] = min(report.get(key, np.inf), value)
        else:
            report[key] = report.get(key, 0) + value


def delayandsum(
    rcv_arrivals: dict,
    *,
    source_waveform: np.ndarray,
    sample_rate: float,
    trace_frequency: Optional[float] = None,
    output_duration: Optional[float] = None,
    t_start: Optional[float] = None,
    phase_offset: float = 0.0,
    fractional: bool = True,
    absorption=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convolve a source waveform with the channel a Bellhop arrival record
    describes: ``(time, signal)``, the axis first as from every sibling. ``trace_frequency`` is the frequency the arrivals were traced
    at and ``absorption`` the traced environment's law, which together scale
    the volume absorption across the waveform's band.

    The reader's per-receiver dict unpacked into
    :func:`~uacpy.acoustic_signal.simulate_arrival_reception`, which is the
    same synthesis over plain arrays — use that one for an arrival list from
    anywhere else.

    ``rcv_arrivals`` carries ``amplitudes`` and ``delays``, and optionally
    ``delays_imag`` and ``phases`` (zeros when absent) and ``n_arrivals``
    (else the length of ``delays``), as every arrival-cell function takes
    them; an empty record returns a silent trace
    of the requested window, as :func:`simulate_arrival_reception` and a
    cell of :func:`simulate_arrival_grid` do: a receiver no path reaches
    is silent.

    Parameters
    ----------
    rcv_arrivals : dict
        One receiver's arrival record (see above).
    source_waveform : ndarray
        The source waveform.
    sample_rate : float
        Sample rate (Hz).
    trace_frequency : float, optional
        Frequency (Hz) the arrivals were traced at; required when the
        record carries a non-zero ``delays_imag``.
    output_duration : float, optional
        Output duration (s); ``None`` covers the latest arrival.
    t_start, phase_offset, fractional, absorption : optional
        As on :func:`~uacpy.acoustic_signal.simulate_arrival_reception`.

    Returns
    -------
    time_vector : ndarray
        Time of each sample (s).
    rts : ndarray
        Received time series.
    """
    if _arrival_count(rcv_arrivals, 'delayandsum') == 0:
        amps = delays = imag = phases = np.zeros(0)
    else:
        amps = rcv_arrivals['amplitudes']
        delays = rcv_arrivals['delays']
        imag = rcv_arrivals.get('delays_imag')
        phases = rcv_arrivals.get('phases')
    return _simulate_arrival_reception(
        amps, delays, source_waveform=source_waveform,
        sample_rate=sample_rate, trace_frequency=trace_frequency,
        delays_imag=imag, phases=phases,
        output_duration=output_duration,
        t_start=t_start, phase_offset=phase_offset, fractional=fractional,
        absorption=absorption, who='delayandsum')



def _arrival_count(cell, who: str, where: str = "") -> int:
    """The arrivals of one receiver cell: the length of its ``delays``,
    which its ``n_arrivals`` states when it carries one (a record built by
    hand may carry no count). Refused naming ``who`` and the cell's place
    ``where`` (``" at cells[i][j]"`` on a grid): a cell that is not a
    mapping; a cell holding arrivals — a count above 0, amplitudes or
    delays — without the other column; amplitudes and delays of different
    lengths; a count that is not a whole number, or that the columns
    contradict. Read as empty, any of these would be a receiver nothing
    reaches, or a list missing arrivals."""
    if not isinstance(cell, Mapping):
        raise ConfigurationError(
            f"{who}: an arrival cell{where} is a {type(cell).__name__}, not "
            f"a mapping of columns.",
            remediation="Give each cell as a dict: 'amplitudes' and "
                        "'delays' (s), one of each per arrival, and "
                        "optionally 'delays_imag', 'phases' and "
                        "'n_arrivals'.")
    n_amplitudes = int(np.size(cell.get('amplitudes', ())))
    n_delays = int(np.size(cell.get('delays', ())))
    count = n_delays
    if 'n_arrivals' in cell:
        count = _whole_number_or_none(cell['n_arrivals'])
        if count is None or count < 0:
            raise ConfigurationError(
                f"{who}: an arrival cell{where} states n_arrivals="
                f"{cell['n_arrivals']!r}, which is not a count.",
                remediation="State n_arrivals as the number of arrivals the "
                            "cell holds (0 for a receiver nothing reaches), "
                            "or leave it out to count the delays.")
    held = count > 0 or n_amplitudes > 0 or n_delays > 0
    for column in ('delays', 'amplitudes'):
        if held and column not in cell:
            raise ConfigurationError(
                f"{who}: an arrival cell{where} holds arrivals but no "
                f"{column!r}; got the keys {sorted(cell)}.",
                remediation="Give each cell 'amplitudes' and 'delays' (s), "
                            "one of each per arrival — 'delays_imag' and "
                            "'phases' are zeros when left out — or "
                            "n_arrivals=0 for a receiver nothing reaches.")
    if n_amplitudes != n_delays:
        raise ConfigurationError(
            f"{who}: an arrival cell{where} holds {n_amplitudes} "
            f"amplitude(s) and {n_delays} delay(s).",
            remediation="Give each cell one amplitude and one delay (s) per "
                        "arrival.")
    if count != n_delays:
        raise ConfigurationError(
            f"{who}: an arrival cell{where} states n_arrivals={count} but "
            f"holds {n_delays} arrival(s).",
            remediation="State n_arrivals as the number of delays the cell "
                        "holds, or leave it out to count them.")
    return count


def simulate_arrival_grid(cells, *, source_waveform: np.ndarray,
                          sample_rate: float,
                          trace_frequency: Optional[float] = None,
                          output_duration: Optional[float] = None,
                          t_start: Optional[float] = None,
                          phase_offset: float = 0.0,
                          fractional: bool = True,
                          absorption=None) -> Tuple[np.ndarray, np.ndarray]:
    """Received waveforms over a grid of receivers, each from its own arrival
    list, on ONE clock.

    :func:`simulate_arrival_reception` per cell, with the window every cell
    shares: a window taken from a single cell covers only that cell's
    delays, and every farther receiver — whose energy lands later — would
    convolve to exactly zero. Left ``None``, ``t_start`` and
    ``output_duration`` are the reception's own auto-window rule over the delay span of the
    whole grid (the earliest arrival less a tenth of the pulse; the latest
    arrival plus one pulse length, plus a quiet tail of one pulse length
    capped at the grid's delay span). An echo the window omits is left out
    whole, the precursor and tail of its Hilbert transform with it, as in
    :func:`simulate_arrival_reception`.

    Parameters
    ----------
    cells : sequence of sequences of mappings
        ``cells[i][j]``, the arrival list of receiver ``(i, j)``: a mapping
        with ``amplitudes`` and ``delays`` (s), and optionally
        ``delays_imag`` (s) and ``phases`` (rad), zeros when absent, and
        ``n_arrivals`` (else the length of ``delays``) — the record
        :class:`~uacpy.Arrivals` keeps per receiver.
    source_waveform : ndarray
        Source waveform (1-D, real).
    sample_rate : float
        Sample rate in Hz.
    trace_frequency : float, optional
        Frequency (Hz) the arrivals were traced at; required when a cell
        carries a non-zero ``delays_imag``.
    output_duration, t_start : float, optional
        The shared window (s); see above.
    phase_offset, fractional : optional
        As on :func:`simulate_arrival_reception`, for every cell:
        ``phase_offset`` (radians) added to every arrival's phase (a
        Bellhop ARRIVALS result already carries the line-source phase),
        ``fractional=False`` rounding every delay to a sample.
    absorption : Absorption or AbsorptionCoefficient, optional
        The water-column law ``Im tau`` carries, as
        :func:`simulate_arrival_reception` takes it.

    Returns
    -------
    time_vector : ndarray
        Time of each sample (s).
    traces : ndarray
        Shape ``(n_i, n_j, n_samples)``. A cell whose arrival list is
        empty holds a silent trace, as :func:`simulate_arrival_reception`
        returns for it. A cell the engine never evaluated is the model
        wrapper's to mark NaN (Bellhop's ``r = 0`` column).
    """
    return _simulate_arrival_grid(
        cells,
        source_waveform=source_waveform,
        sample_rate=sample_rate,
        trace_frequency=trace_frequency,
        output_duration=output_duration,
        t_start=t_start,
        phase_offset=phase_offset,
        fractional=fractional,
        absorption=absorption)


def _simulate_arrival_grid(
    cells,
    *,
    source_waveform: np.ndarray,
    sample_rate: float,
    trace_frequency: Optional[float] = None,
    output_duration: Optional[float] = None,
    t_start: Optional[float] = None,
    phase_offset: float = 0.0,
    fractional: bool = True,
    absorption=None,
    who: str = "simulate_arrival_grid",
) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`simulate_arrival_grid` reporting its refusals as ``who``."""
    sample_rate, fc, sts = _reception_waveform(source_waveform, sample_rate,
                                               trace_frequency, who)
    phase_offset = _require_phase_offset(phase_offset, who)
    if t_start is not None:
        t_start = _require_start_time(
            t_start, who,
            remediation="Pass the time (s) of the first output sample, or "
                        "None to start just before the earliest arrival.")
    if output_duration is not None:
        output_duration = require_positive_finite_scalar(
            output_duration, who, "output_duration", " s",
            why=" Pass the window's length, or None to size it from the "
                "arrivals.")
    n_i, n_j = len(cells), len(cells[0])
    columns = [[None if _arrival_count(cells[i][j], who,
                                       f" at cells[{i}][{j}]") == 0
                else _arrival_columns(
                    cells[i][j]['amplitudes'], cells[i][j]['delays'],
                    cells[i][j].get('delays_imag'),
                    cells[i][j].get('phases'), who)
                for j in range(n_j)] for i in range(n_i)]
    filled = [c for row in columns for c in row if c is not None]
    _refuse_absorption_without_trace_frequency(filled, fc, who)
    if filled:
        t_start, output_duration = _reception_window(
            min(float(c[1].min()) for c in filled),
            max(float(c[1].max()) for c in filled),
            sts.size, sample_rate, t_start, output_duration, who)
        n_t = int(np.ceil(output_duration * sample_rate))
    else:
        t_start = 0.0 if t_start is None else float(t_start)
        n_t = (int(_EMPTY_TRACE_SECONDS * sample_rate)
               if output_duration is None
               else int(np.ceil(output_duration * sample_rate)))
    t_vec = t_start + np.arange(n_t) * (1.0 / sample_rate)

    # A cell no arrival reached stays silent. Every other cell is its own
    # reception, on the window every cell shares.
    data = np.zeros((n_i, n_j, n_t))
    # One report for the grid: the window is shared, so the cells' omitted
    # and clipped echoes are totalled and said once below.
    clip_report: dict = {}
    if filled:
        where = [(i, j) for i in range(n_i) for j in range(n_j)
                 if columns[i][j] is not None]
        traces, clip_report = _receptions_on_window(
            [columns[i][j] for i, j in where], sts, sample_rate=sample_rate,
            trace_frequency=fc, absorption=absorption, t_start=t_start,
            n_samples=n_t, phase_offset=phase_offset, fractional=fractional,
            who=who)
        for (i, j), trace in zip(where, traces):
            data[i, j] = trace

    notice = _echo_window_notice(clip_report, t_start, n_t / sample_rate,
                                 who=who)
    if notice is not None:
        warnings.warn(notice, NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return t_vec, data
