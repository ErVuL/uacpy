"""The channel a transmission crosses: additive noise at a given Eb/N0,
static and fading multipath, and the baseband tap set a propagation result
hands this package (:class:`ChannelTaps`, from
:meth:`uacpy.core.results.Arrivals.channel_taps`)."""

from __future__ import annotations

from collections import namedtuple
import warnings

import numpy as np

from uacpy.acoustic_signal._results import ResultTuple
from uacpy.acoustic_signal.channel import impulse_response
from uacpy.comms.phy import rc_pulse, rrc_pulse
from uacpy.core._validate import require_positive_finite_scalar
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._plotting import plotter
from uacpy.core.exceptions import (
    ConfigurationError, NumericsWarning, ValidityWarning,
)


class ChannelTaps(ResultTuple, namedtuple(
        'ChannelTaps',
        'taps delays_s symbol_rate fc sps first_arrival_s')):
    """Discrete-time baseband channel, as :meth:`Arrivals.channel_taps` and
    :func:`pulse_shaped_taps` return it.

    ``taps[k]`` multiplies the baseband sample ``k`` samples after the one
    the first arrival delivers; ``delays_s[k]`` is that tap's time relative
    to the first arrival's centre (negative for the leading skirt of a
    pulse), ``symbol_rate`` and ``fc`` are the rate and carrier frequency
    the taps were built for (``fc`` is ``None`` for taps built from
    amplitudes already at baseband, which no carrier rotated), ``sps`` the samples per symbol they are spaced at,
    and ``first_arrival_s`` the absolute travel time the delays were
    re-referenced from. ``comms.apply_channel`` and ``comms.simulate_link``
    take the whole tuple or its ``.taps``.

    On the export protocol: ``to_dict`` / ``from_dict``, and
    ``to_xarray`` / ``to_netcdf`` with the taps on the ``delays_s`` axis.
    """

    __slots__ = ()

    def __new__(cls, taps, delays_s, symbol_rate, fc, sps,
                first_arrival_s):
        # The scalars as Python numbers, so a record read back from a
        # file (0-d arrays) is the record that was written.
        return super().__new__(
            cls, np.asarray(taps), np.asarray(delays_s, dtype=float),
            float(symbol_rate), None if fc is None else float(fc), int(sps),
            float(first_arrival_s))

    def _field_units(self):
        return {'taps': None, 'delays_s': 's', 'symbol_rate': 'Hz',
                'fc': 'Hz', 'sps': '', 'first_arrival_s': 's'}

    def _axes(self):
        # Each tap sits at its delay, so the delays are the one axis.
        return {name: (('delays_s',) if value.ndim == 1 else ())
                for name, value in self._arrays().items()}

    def plot(self, **kwargs):
        """Draw these taps through :func:`uacpy.plot.plot_channel`.

        The plotter takes this whole carrier, so it reads both the things it
        needs that are not fields of it: the sample rate, which taps at
        ``sps`` per symbol at ``symbol_rate`` symbols per second sit at
        (``symbol_rate * sps`` Hz), and the delay axis, which is
        :attr:`delays_s` and not ``arange(n)/fs`` — the grid starts on the
        transmit pulse's leading skirt, ahead of the first arrival.
        ``kwargs`` reach the plotter.

        Returns ``(fig, ax)`` where ``ax`` is a **pair** — ``plot_channel``
        draws the delay and frequency panels side by side — unlike the
        single axis the rest of the family returns.
        """
        return plotter('plot_channel')(self, **kwargs)


# Retained DFT bins below which a band-limited tap process has too few degrees
# of freedom for its envelope to be Rayleigh.
_MIN_DOPPLER_BINS = 8


def awgn(signal, snr_dB, *, rng=None):
    """Add complex (or real) AWGN for a target SNR (dB) over the sampled band.

    A zero-power signal is returned unchanged with a ValidityWarning: the SNR
    target scales the noise power off the signal power, so zero signal
    power means zero noise power at any ``snr_dB``.

    Parameters
    ----------
    signal : array_like
        Clean signal, real or complex. A complex signal is given ``n0/2`` in
        each quadrature, so ``snr_dB`` means the same thing for a
        complex-baseband and a real-passband input.
    snr_dB : float
        Target signal-to-noise ratio in dB: the mean power of ``signal``
        over the total noise power, which is white over the whole sampled
        band — ``[-fs/2, fs/2]`` for a complex signal, ``[0, fs/2]`` for a
        real one. For an oversampled signal occupying a bandwidth ``B``,
        the SNR inside ``B`` is higher by ``10*log10(fs/B)`` (complex, ``B``
        two-sided) or ``10*log10(fs/(2*B))`` (real): a complex RRC baseband
        at ``sps=8``, ``rolloff=0.25`` (``B = 1.25`` x symbol rate) reads
        8.06 dB above ``snr_dB`` in-band (measured 8.07 dB). At one sample
        per symbol, as :func:`simulate_link` runs, the sampled band is the
        symbol-rate band and the two coincide.
    rng : numpy.random.Generator, optional
        Random generator for the noise realisation. Pass a seeded generator
        for a reproducible result.

    Returns
    -------
    ndarray
        ``signal`` plus the noise realisation, same shape and dtype class.
    """
    x = np.asarray(signal)
    rng = np.random.default_rng() if rng is None else rng
    p = np.mean(np.abs(x) ** 2)
    if p == 0:
        warnings.warn(
            "awgn: the signal has zero power, so the noise power that "
            "realises the requested SNR is zero and the signal is returned "
            "unchanged.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
    n0 = p / (10.0 ** (float(snr_dB) / 10.0))
    if np.iscomplexobj(x):
        # n0 is the *total* noise power, split n0/2 into each quadrature so
        # E|noise|^2 = n0 and the requested SNR means the same thing for a
        # complex-baseband and a real-passband signal.
        noise = np.sqrt(n0 / 2) * (rng.standard_normal(x.shape)
                                   + 1j * rng.standard_normal(x.shape))
    else:
        noise = np.sqrt(n0) * rng.standard_normal(x.shape)
    return x + noise


def _snr_per_ebn0(who, bits_per_symbol, symbol_rate, sample_rate,
                  code_rate, real):
    """``SNR / (Eb/N0)`` as a linear ratio, with every argument validated."""
    for name, value in (("bits_per_symbol", bits_per_symbol),
                        ("symbol_rate", symbol_rate),
                        ("sample_rate", sample_rate)):
        require_positive_finite_scalar(value, who, name)
    r = float(code_rate)
    if not 0.0 < r <= 1.0:
        raise ConfigurationError(
            f"{who}: code_rate is information bits per channel bit, in "
            f"(0, 1]; got {code_rate!r}.")
    # Noise power over the sampled band is N0 * B: B = sample_rate for a
    # complex signal ([-fs/2, fs/2]) and sample_rate/2 for a real one
    # ([0, fs/2]), the band awgn spreads its noise over. Signal power is
    # Eb * bits_per_symbol * code_rate * symbol_rate.
    band = float(sample_rate) / 2.0 if real else float(sample_rate)
    return float(bits_per_symbol) * r * float(symbol_rate) / band


def ebn0_to_snr_dB(ebn0_dB, *, bits_per_symbol, symbol_rate, sample_rate,
                   code_rate=1.0, real=False):
    """The :func:`awgn` ``snr_dB`` that realises an information-bit ``Eb/N0``.

    ``awgn`` spreads its noise over the whole sampled band, ``[-fs/2, fs/2]``
    for a complex signal and ``[0, fs/2]`` for a real one, so the SNR it
    takes depends on how many samples each symbol spans::

        SNR = Eb/N0 * bits_per_symbol * code_rate * symbol_rate / B

    with ``B = sample_rate`` (complex) or ``sample_rate / 2`` (real). At one
    complex sample per symbol this is ``Es/N0 = k R Eb/N0``, the conversion
    :func:`simulate_link` makes.

    Parameters
    ----------
    ebn0_dB : float or array_like
        Energy per information bit over the one-sided noise density, in dB.
    bits_per_symbol : float
        Channel bits per symbol (``Modulator.bits_per_symbol``; 1 for binary
        FSK).
    symbol_rate : float
        Symbols per second (Bd).
    sample_rate : float
        Sample rate of the signal ``awgn`` is applied to, in Hz.
    code_rate : float, optional
        Information bits per channel bit, in ``(0, 1]`` (``ConvCode.rate``).
        Defaults to ``1.0``, uncoded.
    real : bool, optional
        ``True`` for a real (passband) signal, ``False`` for complex
        baseband. Defaults to ``False``.

    Returns
    -------
    float or ndarray
        ``snr_dB`` to pass to :func:`awgn`.
    """
    g = _snr_per_ebn0("ebn0_to_snr_dB", bits_per_symbol, symbol_rate,
                      sample_rate, code_rate, real)
    out = np.asarray(ebn0_dB, dtype=float) + 10.0 * np.log10(g)
    return float(out) if out.ndim == 0 else out


def snr_to_ebn0_dB(snr_dB, *, bits_per_symbol, symbol_rate, sample_rate,
                   code_rate=1.0, real=False):
    """Inverse of :func:`ebn0_to_snr_dB`: the information-bit ``Eb/N0`` in dB
    that an :func:`awgn` ``snr_dB`` over the sampled band amounts to. The
    arguments are :func:`ebn0_to_snr_dB`'s.

    Parameters
    ----------
    snr_dB : float or array_like
        The :func:`awgn` SNR over the sampled band, in dB.
    bits_per_symbol : float
        Channel bits per symbol.
    symbol_rate : float
        Symbols per second (Bd).
    sample_rate : float
        Sample rate of the signal (Hz).
    code_rate : float, optional
        Information bits per channel bit, in ``(0, 1]``. Default 1.0.
    real : bool, optional
        ``True`` for a real (passband) signal. Default False.
    """
    g = _snr_per_ebn0("snr_to_ebn0_dB", bits_per_symbol, symbol_rate,
                      sample_rate, code_rate, real)
    out = np.asarray(snr_dB, dtype=float) - 10.0 * np.log10(g)
    return float(out) if out.ndim == 0 else out


def pulse_shaped_taps(amplitudes, delays_s, symbol_rate, *, pulse='rc',
                      rolloff=0.25, sps=1, span=8):
    """Symbol-rate tap vector from sparse arrivals, through a pulse shape.

    Where :func:`multipath_channel` lays each arrival on the nearest
    sample of a tap-delay line, this lays each one down as the pulse the
    modem actually transmits::

        h[k] = sum_i g_i * p(k*T - tau_i)

    with ``p`` a raised-cosine or root-raised-cosine. It is the right
    placement when the taps will be used as a **symbol-rate** channel for
    an equalizer, because what such an equalizer sees is the combined
    response of the transmit filter, the channel and the receive filter —
    not the channel alone. ``impulse_response``'s windowed-sinc
    fractional-delay kernel is a different, and for this purpose wrong,
    placement: it interpolates a bandlimited sample, it does not apply a
    modulation pulse.

    Parameters
    ----------
    amplitudes : array_like
        Complex path amplitudes, carrying per-path phase.
    delays_s : array_like
        Path delays (s), measured from whatever origin the caller wants
        the tap grid to start at. They are used as given: subtract the
        first arrival yourself if the taps should start there.
    symbol_rate : float
        Symbol rate (Bd). ``T = 1/symbol_rate``.
    pulse : {'rc', 'rrc'}
        Raised cosine (normalised to unit peak, so it samples at Nyquist
        instants) or root raised cosine (normalised to unit energy over
        its own span, matching what a matched filter pair expects).
    rolloff : float
        Excess bandwidth of the pulse, in ``[0, 1]``.
    sps : int
        Samples per symbol of the returned grid. ``1`` is symbol-rate.
    span : int
        Pulse length in symbols. The grid is offset by ``span/2`` symbols
        so the pulse's own centre lands on its arrival.

    Returns
    -------
    ChannelTaps
        The taps (``.taps``) on their delay axis (``.delays_s``, starting at
        ``-span/(2*symbol_rate)`` by construction), with the ``symbol_rate``
        and ``sps`` they are spaced at, ``fc=None`` (the amplitudes are
        already baseband) and ``first_arrival_s=0.0`` (the delays are used
        as given) — the type :meth:`Arrivals.channel_taps` returns, which
        :func:`simulate_link` and :func:`uacpy.plot.plot_channel` read.
    """
    return _pulse_shaped_taps(
        amplitudes,
        delays_s,
        symbol_rate,
        pulse=pulse,
        rolloff=rolloff,
        sps=sps,
        span=span)


def _pulse_shaped_taps(amplitudes, delays_s, symbol_rate, *, pulse='rc',
                      rolloff=0.25, sps=1, span=8,
                      who: str = "pulse_shaped_taps"):
    """:func:`pulse_shaped_taps` reporting its refusals as ``who``."""
    amplitudes = np.asarray(amplitudes)
    rel = np.asarray(delays_s, dtype=float).ravel()
    if amplitudes.size != rel.size:
        raise ConfigurationError(
            f"{who}: amplitudes and delays_s must have the same length; got "
            f"{amplitudes.size} and {rel.size}.")
    if pulse not in ('rc', 'rrc'):
        raise ConfigurationError(
            f"{who}: pulse must be 'rc' or 'rrc'; got {pulse!r}.")
    symbol_rate = require_positive_finite_scalar(
        symbol_rate, who, "symbol_rate", " Bd")
    sps = int(sps)
    span = int(span)
    if span < 1:
        raise ConfigurationError(
            f"{who}: span must be >= 1 symbol; got {span!r}.")
    if sps < 1:
        raise ConfigurationError(
            f"{who}: sps must be >= 1 sample per symbol; got {sps!r}.")
    fs = symbol_rate * sps
    half = span * sps / 2.0
    # The 1e-9 keeps an arrival a rounding error past a sample instant
    # from adding an empty tap.
    n_taps = int(np.ceil(rel.max() * fs - 1e-9)) + span * sps + 1
    times = (np.arange(n_taps) - half) / fs
    if pulse == 'rc':
        shape, norm = rc_pulse, 1.0     # unit peak: Nyquist samples
    else:
        grid = (np.arange(span * sps + 1) - half) / sps
        shape = rrc_pulse
        norm = float(np.sqrt(np.sum(rrc_pulse(grid, rolloff) ** 2)))
    taps = np.zeros(n_taps, dtype=complex)
    for gain, tau in zip(amplitudes, rel):
        arg = (times - tau) * symbol_rate
        g = shape(arg, rolloff)
        # The pulse ends at +-span/2 inclusive, as rrc_filter's grid does;
        # the 1e-9 keeps rounding from dropping an end tap.
        g[np.abs(arg) > span / 2.0 + 1e-9] = 0.0
        taps += gain * g / norm
    return ChannelTaps(taps=taps, delays_s=times, symbol_rate=symbol_rate,
                       fc=None, sps=sps, first_arrival_s=0.0)


def multipath_channel(amplitudes, delays_s, *, sample_rate, fractional=False):
    """Static FIR tap vector from sparse arrivals ``(gain, delay)``.

    Adapter over :func:`uacpy.acoustic_signal.impulse_response` — same
    arguments, returning only the complex tap vector that :func:`apply_channel`
    and the equalizers consume. ``amplitudes`` may be complex (carry per-path
    phase). A tap-delay line is an integer-tap FIR, hence ``fractional=False``
    by default.

    For a channel that comes from a propagation model rather than from
    hand-typed numbers, :meth:`uacpy.core.results.Arrivals.channel_taps`
    does the carrier rotation and the pulse for you: it returns the
    baseband taps of a Bellhop ``ARRIVALS`` result at a symbol rate, and
    with ``pulse=None, sps=1`` it reduces tap for tap to this function
    called on ``received_amplitudes`` and the delays re-referenced to the
    earliest one.

    Parameters
    ----------
    amplitudes : array_like
        Path amplitudes, complex to carry a phase.
    delays_s : array_like
        Path delays (s).
    sample_rate : float
        Sample rate of the tap vector (Hz).
    fractional : bool, optional
        Place a delay between samples with a fractional-delay kernel instead
        of the nearest tap. Default False.
    """
    _, h = impulse_response(amplitudes, delays_s, sample_rate=sample_rate, fractional=fractional)
    return h.astype(complex)


def apply_channel(signal, h):
    """Convolve a signal with a static channel ``h`` (returns full convolution).

    ``h`` is a tap vector or a
    :class:`ChannelTaps`, whose ``.taps`` are
    used: tap 0 is its ``delays_s[0]``, on the pulse's leading skirt, so the
    first arrival reaches the output ``-delays_s[0]*symbol_rate*sps``
    samples in.

    Parameters
    ----------
    signal : array_like
        The transmitted samples.
    h : array_like or ChannelTaps
        The channel taps, at least one.
    """
    taps = _channel_array(h)
    if taps.size == 0:
        raise ConfigurationError(
            "apply_channel: empty channel h (0 taps); pass at least one tap "
            "— h=[1.0] is the identity channel.")
    return np.convolve(np.asarray(signal), taps)


def fading_taps(n_taps, n_samples, doppler_hz, *, sample_rate, rician_k=0.0,
                rng=None):
    """Time-varying complex tap gains ``H`` of shape ``(n_taps, n_samples)``.

    Each tap is a band-limited complex-Gaussian process (max Doppler
    ``doppler_hz``) — Rayleigh fading; ``rician_k > 0`` adds a line-of-sight
    component with the given Rice K-factor (linear). Unit average power over
    the ensemble, not per realisation. ``doppler_hz = 0`` is the static limit:
    only the DC bin survives, so each tap is a single complex-Gaussian draw
    held constant over the block; a negative ``doppler_hz`` raises
    :class:`~uacpy.core.exceptions.ConfigurationError`.

    The Doppler band is resolved to ``sample_rate/n_samples``, so it spans
    ``2·doppler_hz·n_samples/sample_rate`` DFT bins. That bin count is the
    process's degrees of freedom: below a handful the block is too short to
    realise the requested spread and the envelope is not Rayleigh, which
    warns.

    Parameters
    ----------
    n_taps : int
        Number of channel taps, one row of ``H`` each.
    n_samples : int
        Block length in samples, one column of ``H`` each. It also sets the
        Doppler resolution, ``sample_rate/n_samples``.
    doppler_hz : float
        Maximum Doppler shift in Hz — the half-width of the flat Doppler
        band. ``0`` is the static limit; negative raises
        :class:`~uacpy.core.exceptions.ConfigurationError`.
    sample_rate : float
        Sample rate in Hz.
    rician_k : float, optional
        Rice K-factor, linear (not dB), as the ratio of line-of-sight power
        to diffuse power. Defaults to ``0.0``, which is pure Rayleigh.
    rng : numpy.random.Generator, optional
        Random generator for the tap draw. Pass a seeded generator for a
        reproducible result.

    Returns
    -------
    ndarray
        Complex tap gains of shape ``(n_taps, n_samples)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    fs = float(sample_rate)
    n = int(n_samples)
    # Written as the negation of the admissible condition so NaN is refused
    # too: `doppler_hz < 0` is False for NaN, which then made `|f| <= nan`
    # all-False and every tap NaN. `isfinite` is the other half — `|f| <= inf`
    # is all-True, which disables the Doppler low-pass entirely and returns
    # unfiltered white noise that looks like a plausible fading process.
    if not (np.isfinite(doppler_hz) and doppler_hz >= 0):
        raise ConfigurationError(
            f"fading_taps: doppler_hz must be >= 0 Hz and finite; got "
            f"{doppler_hz!r} (0 is the static limit — a constant complex gain "
            f"per tap).")
    # Negated so NaN fails too: `rician_k > 0` below is False for NaN and for
    # a negative K, which would hand back pure Rayleigh in silence.
    if not (np.isfinite(rician_k) and rician_k >= 0):
        raise ConfigurationError(
            f"fading_taps: rician_k must be >= 0 and finite (a linear power "
            f"ratio; 0 is Rayleigh); got {rician_k!r}.")
    # White complex Gaussian per tap, low-pass filtered to the Doppler
    # bandwidth. doppler_hz = 0 keeps the DC bin alone, so each tap is one
    # complex-Gaussian draw held constant over the block.
    H = (rng.standard_normal((n_taps, n)) + 1j * rng.standard_normal((n_taps, n)))
    f = np.fft.fftfreq(n, d=1.0 / fs)
    mask = (np.abs(f) <= float(doppler_hz)).astype(float)
    n_bins = int(mask.sum())
    H = np.fft.ifft(np.fft.fft(H, axis=1) * mask[None, :], axis=1)
    if doppler_hz > 0 and n_bins < _MIN_DOPPLER_BINS:
        warnings.warn(
            f"fading_taps: doppler_hz={float(doppler_hz):g} spans {n_bins} DFT "
            f"bin(s) at sample_rate/n_samples={fs / n:g} Hz resolution, so the "
            f"tap process is drawn from fewer than {_MIN_DOPPLER_BINS} degrees "
            f"of freedom and its envelope statistics are not Rayleigh. Lengthen "
            f"n_samples to at least {int(np.ceil(_MIN_DOPPLER_BINS * fs / (2.0 * float(doppler_hz))))} "
            f"samples, or raise doppler_hz.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    # Normalise by the filter's analytic gain, not per realisation: dividing each
    # tap by its own RMS pins |H| to a constant when few bins survive, which is a
    # unit-modulus phase, not a fading process.
    H /= np.sqrt(2.0 * n_bins / n)
    if rician_k > 0:
        k = float(rician_k)
        H = np.sqrt(k / (k + 1)) + np.sqrt(1 / (k + 1)) * H
    return H


def apply_fading_channel(signal, taps, delays_samples):
    """Apply a time-varying channel: ``y[n] = sum_i taps[i, n-d_i] * x[n-d_i]``.

    Each tap's gain is sampled at the **input** time of the sample it delays
    (``taps[i, m]`` multiplies ``x[m]``), which is why ``taps`` only needs
    ``len(signal)`` columns. The textbook tap-delay line samples the gain at
    the output time instead (``taps[i, n]·x[n-d_i]``); the two differ only
    by a per-tap gain shift of ``d_i`` samples and are statistically
    indistinguishable when Doppler × delay spread ≪ 1 — always true for a
    physical underwater channel.

    ``taps`` is ``(n_taps, >=len(signal))``; ``delays_samples`` the integer tap
    delays. Returns ``y`` of length ``len(signal) + max(delay)``.

    Parameters
    ----------
    signal : array_like
        The transmitted samples.
    taps : ndarray
        ``(n_taps, >= len(signal))`` time-varying tap gains
        (:func:`fading_taps`).
    delays_samples : array_like of int
        Each tap's delay in samples, >= 0.
    """
    x = np.asarray(signal, dtype=complex)
    n = x.size
    d = np.asarray(delays_samples, dtype=int)
    taps = np.asarray(taps, dtype=complex)
    if taps.shape[0] != d.size:
        raise ConfigurationError(
            f"apply_fading_channel: taps rows must match delays — taps has "
            f"{taps.shape[0]} rows for {d.size} delays. fading_taps("
            f"n_taps={d.size}, ...) sizes them to match.")
    if taps.shape[1] < n:
        raise ConfigurationError(
            f"apply_fading_channel: taps shorter than signal — taps carries "
            f"{taps.shape[1]} columns for a {n}-sample signal. Call "
            f"fading_taps(..., n_samples={n}) to cover it.")
    if d.size and d.min() < 0:
        # y[di:di+n] indexes from the END for a negative di, so a negative
        # delay either raises a bare broadcast ValueError or — when
        # di < -n <= -len(y) — quietly places the echo near the tail of y
        # (delays [10, -8] on a 6-sample input put it at samples 8-13).
        raise ConfigurationError(
            f"apply_fading_channel: delays_samples must be >= 0; got "
            f"{d.min()}. A tap cannot arrive before the signal.")
    y = np.zeros(n + int(d.max()), dtype=complex)
    for i, di in enumerate(d):
        y[di:di + n] += taps[i, :n] * x
    return y


def _channel_array(h):
    """The tap vector of a channel given as taps or as a
    :class:`ChannelTaps`.

    A ``ChannelTaps`` is its ``.taps``: tap 0 sits at ``delays_s[0]``, on the
    transmit pulse's leading skirt, so every channel consumer here treats the
    taps as the causal channel from that tap on, exactly as it treats the
    bare ``.taps``. Only :func:`uacpy.comms.simulate_link`, which reads the
    received symbols at fixed positions, reads ``delays_s`` too, to start at
    the zero-delay tap.
    """
    return np.asarray(h.taps if isinstance(h, ChannelTaps) else h)
