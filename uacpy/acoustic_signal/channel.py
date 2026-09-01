"""Time-domain channel simulation from a modelled multipath structure.

Turns a propagation model's arrival structure (amplitudes + travel times, e.g.
from a Bellhop ARRIVALS run) or a transfer function ``H(f)`` into a channel
impulse response, then convolves a transmit waveform with it to synthesise the
received signal — the basis of replay benchmarks for underwater communications
and active-sonar echo simulation.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.acoustic_signal._signal_validate import (
    require_increasing_axis, require_positive_finite_scalar)


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

    Shared with :func:`uacpy.models.bellhop.delayandsum`, which places whole
    waveforms rather than single taps but needs the identical kernel: a
    two-tap linear split is NOT a fractional delay — its response
    ``|(1-frac) + frac*e^{-jw}|`` is a lowpass whose attenuation depends on
    ``frac``, with a full null at Nyquist for ``frac = 0.5``. The windowed
    sinc is flat to ~0.01 dB below ``f/fs = 0.35``.
    """
    offsets = np.arange(-half_len + 1, half_len + 1) - float(frac)
    u = offsets / half_len
    win = (np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - u * u))) / np.i0(beta))
    taps = np.sinc(offsets) * win
    total = taps.sum()
    return taps / total if total else taps


def impulse_response(amplitudes, delays_s, sample_rate: float, *,
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
        with a ``UserWarning`` counting the drops (on both placement
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
            f"{d.shape}")
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
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
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
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
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

    Returns ``(t, h)``.
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
    fs = float(sample_rate)
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
            UserWarning, stacklevel=2)
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
    grid = np.fft.rfftfreq(int(n_samples), d=1.0 / fs)
    Hr = (np.interp(grid, f, Hc.real, left=0.0, right=0.0)
          + 1j * np.interp(grid, f, Hc.imag, left=0.0, right=0.0))
    h = np.fft.irfft(Hr, n=n_samples)
    t = np.arange(n_samples) / fs
    return t, h
