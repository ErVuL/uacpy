"""A link end to end: the channel it crosses, the PHY it speaks, and the
transceiver that drives both.

One question — *what does this link achieve* — from a channel model
(:func:`apply_channel`, :func:`awgn`) and a PHY definition through
:class:`OFDMTransmitter` / :class:`OFDMReceiver` to a swept
:func:`ber_sweep`.
"""

from __future__ import annotations
import warnings
import numpy as np
from uacpy.acoustic_signal.system import impulse_response
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from dataclasses import dataclass
from typing import Optional
from uacpy.comms.modulate import (
    ConvCode,
    Modulator,
    apply_cfo,
    equalize_subcarriers,
    estimate_channel,
    ofdm_demodulate,
    ofdm_symbol,
    schmidl_cox_preamble,
    schmidl_cox_sync)
from uacpy.comms.receive import (
    DFE,
    bit_error_rate,
    compensate_doppler,
    detect_preamble,
    estimate_doppler_scale,
    evm,
    slicer)


# ──────────────────────────────────────────────────────────────────────
# Channel models
#
# The medium a link crosses.
# ──────────────────────────────────────────────────────────────────────

# Retained DFT bins below which a band-limited tap process has too few degrees
# of freedom for its envelope to be Rayleigh.
_MIN_DOPPLER_BINS = 8


def awgn(signal, snr_dB, *, rng=None):
    """Add complex (or real) AWGN for a target in-band SNR (dB).

    A zero-power signal is returned unchanged with a UserWarning: the SNR
    target scales the noise power off the signal power, so zero signal
    power means zero noise power at any ``snr_dB``.

    Parameters
    ----------
    signal : array_like
        Clean signal, real or complex. A complex signal is given ``n0/2`` in
        each quadrature, so ``snr_dB`` means the same thing for a
        complex-baseband and a real-passband input.
    snr_dB : float
        Target in-band signal-to-noise ratio in dB, referred to the mean
        power of ``signal``.
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
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
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


def multipath_channel(gains, delays_s, sample_rate, *, fractional=False):
    """Static FIR tap vector from sparse arrivals ``(gain, delay)``.

    Adapter over :func:`uacpy.acoustic_signal.impulse_response` — same
    arguments, returning only the complex tap vector that :func:`apply_channel`
    and the equalizers consume. ``gains`` may be complex (carry per-path
    phase). A tap-delay line is an integer-tap FIR, hence ``fractional=False``
    by default.
    """
    _, h = impulse_response(gains, delays_s, sample_rate, fractional=fractional)
    return h.astype(complex)


def apply_channel(signal, h):
    """Convolve a signal with a static channel ``h`` (returns full convolution)."""
    taps = np.asarray(h)
    if taps.size == 0:
        raise ConfigurationError(
            "apply_channel: empty channel h (0 taps); pass at least one tap "
            "— h=[1.0] is the identity channel.")
    return np.convolve(np.asarray(signal), taps)


def fading_taps(n_taps, n_samples, doppler_hz, sample_rate, *, rician_k=0.0,
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
            UserWarning, stacklevel=2)
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


# ──────────────────────────────────────────────────────────────────────
# PHY definitions
#
# The parameter sets a link speaks.
# ──────────────────────────────────────────────────────────────────────

def _require_integer_sps(caller, sps):
    """Return ``sps`` as an ``int``, accepting exactly-integral floats.

    ``fs / baud`` naturally produces a float (``4.0``, ``np.float64(4.0)``)
    that names the same sample grid as the integer, so it coerces. A
    fractional value names no grid at all: the upsampler's ``up[::sps]``
    stride and the RRC time axis ``arange(span*sps + 1) / sps`` are defined
    only for a whole number of samples per symbol.
    """
    try:
        i = int(sps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(
            f"{caller}: sps must be a whole number of samples per symbol; "
            f"got {sps!r} ({exc}).") from exc
    if i != sps:
        raise ConfigurationError(
            f"{caller}: sps must be a whole number of samples per symbol; "
            f"got {sps!r}. If fs/baud is fractional, pick a sample rate the "
            f"baud rate divides evenly.")
    return i


def rrc_filter(sps, rolloff, span):
    """Root-raised-cosine taps: ``span`` symbols, ``sps`` samples/symbol, unit energy."""
    if not 0.0 <= rolloff <= 1.0:
        raise ConfigurationError(
            f"rrc_filter: rolloff must be in [0, 1]; got {rolloff!r}")
    sps = _require_integer_sps("rrc_filter", sps)
    # sps is the divisor of the symbol-period axis: 0 makes every tap NaN and
    # returns a length-1 filter, a negative value makes `span*sps` negative
    # and returns an empty one — both convolve without complaint.
    if sps < 1:
        raise ConfigurationError(
            f"rrc_filter: sps must be >= 1 (samples per symbol); got "
            f"{sps!r}. It sets the taps' time axis (arange(span*sps + 1) "
            f"scaled by 1/sps), so the filter is undefined below 1.")
    if int(span) < 1:
        raise ConfigurationError(
            f"rrc_filter: span must be >= 1 (symbols); got {span!r}.")
    n = span * sps
    t = (np.arange(n + 1) - n / 2) / sps      # time in symbol periods
    b = float(rolloff)
    h = np.empty_like(t)
    # The general expression below divides by ``pi*t*(1 - (4*b*t)**2)``, which
    # vanishes at t = 0 and at |t| = 1/(4b) (t in symbol periods). Both are
    # removable singularities of the RRC impulse response, so the two branches
    # substitute their analytic limits; the 1e-8 tolerance catches the sampled
    # grid landing on (or numerically next to) either point.
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1 - b + 4 * b / np.pi
        elif b > 0 and abs(abs(ti) - 1 / (4 * b)) < 1e-8:
            h[i] = (b / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * b))
                                       + (1 - 2 / np.pi) * np.cos(np.pi / (4 * b)))
        else:
            num = (np.sin(np.pi * ti * (1 - b))
                   + 4 * b * ti * np.cos(np.pi * ti * (1 + b)))
            den = np.pi * ti * (1 - (4 * b * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def pulse_shape(symbols, sps, rolloff=0.25, span=8):
    """Upsample symbols by ``sps`` and root-raised-cosine filter -> baseband samples."""
    sps = _require_integer_sps("pulse_shape", sps)
    if sps < 1:
        raise ConfigurationError(
            f"pulse_shape: need sps >= 1 (samples per symbol); got {sps!r}")
    s = np.asarray(symbols, dtype=complex).ravel()
    up = np.zeros(s.size * sps, dtype=complex)
    up[::sps] = s
    return np.convolve(up, rrc_filter(sps, rolloff, span))


def matched_filter(samples, sps, rolloff=0.25, span=8):
    """Matched root-raised-cosine filter (completes the Nyquist response)."""
    return np.convolve(np.asarray(samples, dtype=complex),
                       rrc_filter(sps, rolloff, span))


def upconvert(baseband, sample_rate, fc):
    """Mix complex baseband up to a real passband signal at carrier ``fc``."""
    x = np.asarray(baseband, dtype=complex)
    n = np.arange(x.size)
    return np.real(x * np.exp(2j * np.pi * fc * n / sample_rate))


def downconvert(passband, sample_rate, fc):
    """Mix a real passband signal down to complex baseband (image left for the LPF/MF).

    The factor 2 makes the ``upconvert``/``downconvert`` pair unity-gain:
    ``upconvert`` emits ``Re{b·e^{jwn}} = (b·e^{jwn} + b*·e^{-jwn})/2``, so
    ``2·x·e^{-jwn} = b + b*·e^{-2jwn}`` and the low-pass part is ``b`` itself.
    """
    x = np.asarray(passband, dtype=float)
    n = np.arange(x.size)
    return 2.0 * x * np.exp(-2j * np.pi * fc * n / sample_rate)


def _interp(x, idx):
    """Linear interpolation of ``x`` at fractional index ``idx``."""
    i = int(np.floor(idx))
    if i < 0:
        return x[0]
    if i >= x.size - 1:
        return x[-1]
    f = idx - i
    return x[i] * (1 - f) + x[i + 1] * f


def symbol_sync(samples, sps, loop_bw=0.005, damping=1.0, start=0):
    """Gardner timing recovery: resolve one symbol-rate sample per symbol.

    A non-data-aided Gardner timing-error detector drives a PI loop filter that
    steers a linear interpolator. The integer stride is pinned at ``sps`` (only
    the fractional delay is adjusted, with anti-windup clamps), which avoids the
    half-rate false lock that a free-running accumulator falls into. Returns the
    symbol-rate complex output.

    Parameters
    ----------
    samples : array_like
        Complex baseband at ``sps`` samples/symbol (matched-filtered).
    sps : int
        Samples per symbol (must be >= 2).
    loop_bw : float
        Normalized loop bandwidth (smaller = slower but steadier lock).
    damping : float
        Loop damping factor (~1.0 critically damped).
    start : int
        Initial sample index — pass the matched-filter group delay
        (``span*sps``) so the loop starts on the symbol grid.

    Notes
    -----
    Pull-in, measured noise-free on 16-QAM at ``sps=8``, ``rolloff=0.25``,
    ``span=8``, ``loop_bw=0.005``, taking lock as a timing residual under 5 %
    of a symbol: from a quarter-symbol offset the residual first crosses that
    threshold at 91-160 symbols and stays under it from 420-560; from a half
    symbol, 267-509 and 695-978. Half a symbol is the timing-error detector's
    unstable equilibrium, which is why it costs several times more than the
    larger-looking three-quarter case would suggest — pass ``start`` so the
    loop begins near the grid rather than relying on a preamble to absorb a
    half-symbol pull-in.

    Locked, the loop sits a little late: about 0.06-0.11 sample (~1 % of a
    symbol) with ~0.07 sample rms jitter and 0.3-0.4 sample peak-to-peak at
    ``sps=8``. This is a steady-state property of the Gardner detector driving
    the 2-point linear interpolator, not a pull-in residual — it is there even
    when the loop starts exactly on the symbol grid — and it scales as a
    fraction of a symbol (~0.013-0.016 symbol at ``sps`` 4 and 16).
    """
    x = np.asarray(samples, dtype=complex).ravel()
    sps = _require_integer_sps("symbol_sync", sps)
    if sps < 2:
        raise ConfigurationError(
            f"symbol_sync: need sps >= 2 for Gardner; got {sps!r}")
    # theta below is loop_bw / (damping + 1/(4*damping)): zero damping is a
    # bare ZeroDivisionError, and a non-positive loop bandwidth leaves the
    # loop gains at zero so the interpolator never steers.
    if not (np.isfinite(damping) and damping > 0):
        raise ConfigurationError(
            f"symbol_sync: damping must be > 0 and finite; got {damping!r} "
            f"(~1.0 is critically damped). The loop constant divides by "
            f"damping + 1/(4*damping).")
    if not (np.isfinite(loop_bw) and loop_bw > 0):
        raise ConfigurationError(
            f"symbol_sync: loop_bw must be > 0 and finite; got {loop_bw!r}. "
            f"At zero the proportional and integral gains are zero and the "
            f"timing estimate never moves off `start`.")
    # Standard second-order proportional-integral loop filter: map the
    # requested noise bandwidth and damping to a per-sample loop constant
    # ``theta``, then to the proportional (``kp``) and integral (``ki``) gains.
    # A second-order loop has noise-equivalent bandwidth
    # ``B_n = (w_n/2)*(damping + 1/(4*damping))`` (Proakis & Salehi eq. 5.2-22
    # with ``tau2*w_n = 2*damping``), so ``theta = w_n*T/2`` inverts to the line
    # below for a per-symbol-period normalized ``loop_bw = B_n*T``.
    theta = loop_bw / (damping + 0.25 / damping)
    denom = 1 + 2 * damping * theta + theta * theta
    kp = 4 * damping * theta / denom
    ki = 4 * theta * theta / denom
    # Normalize the TED gain to the signal level: the detector's numerator
    # scales as amplitude**2, and dividing by the record's own mean power puts
    # the error back at O(1) whatever the record is scaled to. Adding an
    # absolute floor here made the loop gain a function of that scale instead:
    # once mean|x|**2 fell to the floor (~1e-6 amplitude) the error shrank with
    # amplitude**2, the correction went to zero, and the loop stopped adapting
    # and decimated at a fixed stride — measured, the recovered symbols drifted
    # 7.2e-3 from the unit-amplitude answer at 1e-6 and 1.4e-2 by 1e-8. An
    # all-zero record carries no timing information and has a zero numerator
    # too, so the 1.0 below keeps its error at 0 rather than 0/0.
    power = float(np.mean(np.abs(x) ** 2)) if x.size else 1.0
    if power == 0.0:
        power = 1.0

    mu = 0.0          # fractional delay in [0, 1)
    idx = int(start)
    vi = 0.0          # loop-filter integrator
    prev_on = 0j
    out = []
    while idx < x.size - 2:
        y_on = _interp(x, idx + mu)
        y_mid = _interp(x, idx - sps // 2 + mu)
        # Gardner TED (negative feedback for the positive-slope zero crossing)
        diff = y_on - prev_on
        e = -(diff.real * y_mid.real + diff.imag * y_mid.imag) / power
        # Three heuristic bounds on the normalized error scale (``power``
        # divides the TED down to O(1)): the first caps one outlier sample's
        # contribution, the second is integrator anti-windup, and the third
        # holds the per-symbol timing correction to half a symbol so the
        # interpolator cannot step past the crossing it is tracking.
        e = float(np.clip(e, -2.0, 2.0))
        vi = float(np.clip(vi + ki * e, -0.25, 0.25))
        adj = float(np.clip(kp * e + vi, -0.5, 0.5))
        out.append(y_on)
        prev_on = y_on
        # ``adj`` is in SYMBOLS — the loop constants are normalised to the
        # symbol period — and ``mu`` is in samples, so the correction scales
        # by ``sps``. Applied unscaled it was ``sps`` times too small (the
        # TED slope, ~0.7 per symbol, made the effective bandwidth
        # loop_bw/11): pull-in from a quarter-symbol offset took ~380
        # symbols instead of ~50, well past a 64-symbol preamble.
        mu += adj * sps
        idx += sps + int(np.floor(mu))
        mu -= np.floor(mu)
    return np.array(out, dtype=complex)


# ──────────────────────────────────────────────────────────────────────
# Link simulation
#
# One pass end to end, and a sweep over SNR.
# ──────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class LinkResult:
    """Outcome of one :func:`simulate_link` run."""

    ber: float
    evm: float
    scheme: str
    ebn0_dB: float
    tx_symbols: np.ndarray
    rx_symbols: np.ndarray
    mse: np.ndarray | None = None

    def __eq__(self, other):
        """Field-wise equality, with the symbol arrays compared element-wise.

        Hand-written because ``tx_symbols`` / ``rx_symbols`` / ``mse`` are
        ndarrays: the generated ``__eq__`` compares the field tuples, which
        puts an array inside a ``bool()`` and raises "The truth value of an
        array with more than one element is ambiguous".
        """
        if not isinstance(other, LinkResult):
            return NotImplemented
        if (self.ber, self.evm, self.scheme, self.ebn0_dB) != (
                other.ber, other.evm, other.scheme, other.ebn0_dB):
            return False
        if (self.mse is None) != (other.mse is None):
            return False
        if self.mse is not None and not np.array_equal(np.asarray(self.mse),
                                                       np.asarray(other.mse)):
            return False
        return (np.array_equal(np.asarray(self.tx_symbols),
                               np.asarray(other.tx_symbols))
                and np.array_equal(np.asarray(self.rx_symbols),
                                   np.asarray(other.rx_symbols)))



def simulate_link(scheme, ebn0_dB, n_bits=20000, *, channel=None,
                  equalizer=None, code=None, n_train=400, rng=None):
    """Simulate one link and return a :class:`LinkResult`.

    Symbols are unit-average-energy, so the per-symbol SNR is ``k * Eb/N0``
    (``k`` bits/symbol). With ``channel=None`` and ``equalizer=None`` the BER
    matches the AWGN theory curve. ``channel`` is a static FIR ``h``;
    ``equalizer`` a :class:`~uacpy.comms.receive.DFE` (trained on the first
    ``n_train`` symbols); ``code`` a
    :class:`~uacpy.comms.modulate.ConvCode` applied around the modem (BER then
    measured on the information bits).

    Parameters
    ----------
    scheme : str
        Modulation name accepted by
        :class:`~uacpy.comms.modulate.Modulator` — ``'bpsk'``, ``'qpsk'``,
        ``'16qam'`` and the rest of its table.
    ebn0_dB : float
        Information-bit energy to noise density ratio in dB. A ``code``
        lowers the channel Ec/N0 by its rate, so ``ebn0_dB`` stays the
        information-bit figure that BER curves are plotted against.
    n_bits : int, optional
        Number of information bits to transmit. Defaults to ``20000``.
    channel : array_like, optional
        Static FIR channel ``h`` convolved with the transmitted symbols.
        ``None`` is the AWGN-only link.
    equalizer : uacpy.comms.receive.DFE, optional
        Equaliser trained on the first ``n_train`` symbols. ``None`` slices
        the received symbols straight out.
    code : uacpy.comms.modulate.ConvCode, optional
        Code applied around the modem; BER is then measured on the
        information bits rather than the coded ones.
    n_train : int, optional
        Training symbols given to ``equalizer``. Defaults to ``400``.
    rng : numpy.random.Generator, optional
        Random generator for the information bits and the noise. Pass a
        seeded generator for a reproducible result.

    Returns
    -------
    LinkResult
        BER, EVM, the transmitted and received symbols, and the equaliser's
        learning curve when one was used.
    """
    rng = np.random.default_rng() if rng is None else rng
    mod = Modulator(scheme)
    k = mod.bits_per_symbol
    info = rng.integers(0, 2, int(n_bits))
    bits = code.encode(info) if code is not None else info
    tx = mod.modulate(bits)

    delay = equalizer.n_ff // 2 if isinstance(equalizer, DFE) else 0
    if channel is not None:
        rx = apply_channel(tx, channel)
    else:
        rx = tx.copy()
    rx = rx[: tx.size + delay]
    if rx.size < tx.size + delay:
        rx = np.concatenate([rx, np.zeros(tx.size + delay - rx.size, dtype=complex)])

    ebn0 = 10.0 ** (float(ebn0_dB) / 10.0)
    # Es/N0 = (info Eb/N0) x bits-per-symbol x code rate: a coded frame
    # carries R information bits per transmitted bit, so omitting R labels
    # Ec/N0 as Eb/N0 and overstates coding gain by 10log10(1/R).
    rate = float(getattr(code, 'rate', 1.0) or 1.0) if code is not None else 1.0
    rx = awgn(rx, 10.0 * np.log10(k * rate * ebn0), rng=rng)

    mse = None
    if equalizer is not None:
        ref = np.concatenate([np.zeros(delay, dtype=complex), tx])
        eq, mse = equalizer.equalize(rx, mod.constellation, train=ref[: n_train + delay])
        rx_sym = eq[delay: delay + tx.size]
    else:
        rx_sym = rx[: tx.size]

    rx_bits = mod.demodulate(rx_sym)[: bits.size]
    if code is not None:
        rx_bits = code.decode(rx_bits)[: info.size]
        ber = bit_error_rate(info, rx_bits)
    else:
        ber = bit_error_rate(bits, rx_bits)
    return LinkResult(
        ber=ber,
        evm=evm(rx_sym, tx),
        scheme=scheme,
        ebn0_dB=float(ebn0_dB),
        tx_symbols=tx,
        rx_symbols=rx_sym,
        mse=mse,
    )


def ber_sweep(scheme, ebn0_dB_list, n_bits=50000, *, channel=None,
              equalizer=None, code=None, rng=None):
    """Measured BER over a list of Eb/N0 values. Returns a NumPy array.

    One :func:`simulate_link` per Eb/N0, all drawing from the same ``rng``,
    so the points of a sweep are independent realisations rather than the
    same bit stream re-noised.

    Parameters
    ----------
    scheme : str
        Modulation name, as :func:`simulate_link` takes it.
    ebn0_dB_list : array_like
        Information-bit Eb/N0 values in dB, one BER measured at each. A
        scalar is accepted and returns a length-1 array.
    n_bits : int, optional
        Information bits per point. Defaults to ``50000``; the BER floor a
        sweep can measure is roughly ``1/n_bits``.
    channel : array_like, optional
        Static FIR channel ``h``, applied at every point.
    equalizer : uacpy.comms.receive.DFE, optional
        Equaliser settings used at every point. ``DFE.equalize`` re-adapts
        its taps from scratch on each call, so the points do not inherit
        each other's convergence.
    code : uacpy.comms.modulate.ConvCode, optional
        Code applied around the modem at every point.
    rng : numpy.random.Generator, optional
        Random generator shared by every point of the sweep. Pass a seeded
        generator for a reproducible result.

    Returns
    -------
    ndarray
        Measured BER, one per entry of ``ebn0_dB_list``.
    """
    rng = np.random.default_rng() if rng is None else rng
    return np.array([
        simulate_link(scheme, e, n_bits, channel=channel, equalizer=equalizer,
                      code=code, rng=rng).ber
        for e in np.atleast_1d(ebn0_dB_list)
    ])


# ──────────────────────────────────────────────────────────────────────
# Transceivers
#
# The objects that drive a whole chain.
# ──────────────────────────────────────────────────────────────────────

def _require_passband_fits(sample_rate, fc, sps, rolloff, where):
    """Raise unless the RRC band sits clear of DC and of Nyquist at ``fc``.

    The occupied bandwidth is ``Rs·(1+rolloff)`` for symbol rate
    ``Rs = sample_rate/sps``. Synchronous demodulation needs the whole band
    above DC and its image at ``-2·fc`` clear of it, so ``fc`` has to keep
    half a bandwidth from both edges; ``fc < sample_rate/2`` alone lets a
    sideband fold back onto the signal.
    """
    fs = float(sample_rate)
    bw = fs * (1.0 + float(rolloff)) / float(sps)
    lo, hi = bw / 2.0, fs / 2.0 - bw / 2.0
    if lo >= hi:
        raise ConfigurationError(
            f"{where}: the RRC band (Rs*(1+rolloff) = {bw:g} Hz) does not fit "
            f"below Nyquist ({fs / 2:g} Hz) at any carrier.",
            remediation="Raise sample_rate, raise sps, or lower rolloff.")
    if not lo < float(fc) < hi:
        raise ConfigurationError(
            f"{where}: fc ({float(fc):g} Hz) must lie in ({lo:g}, {hi:g}) Hz so "
            f"the RRC band (Rs*(1+rolloff) = {bw:g} Hz) and its image at -2*fc "
            f"do not fold onto the signal.",
            remediation=f"Use a carrier inside ({lo:g}, {hi:g}) Hz, or change "
                        f"sample_rate/sps/rolloff to narrow the band.")


# The seeds are the contract between the two ends of a link: a receiver
# regenerates the preamble / pilot from the same seed and correlates against
# exactly these symbols.
_PREAMBLE_SEED = 0xC0FFEE
_PILOT_SEED = 0xACE0FDA


def _seeded_symbols(n_symbols, scheme, seed):
    """``n_symbols`` pseudo-random ``scheme`` symbols from a fixed seed: the
    default preamble (good autocorrelation) or the known pilot loaded on every
    OFDM subcarrier for channel estimation."""
    rng = np.random.default_rng(seed)
    mod = Modulator(scheme)
    return mod.modulate(rng.integers(0, 2, n_symbols * mod.bits_per_symbol))


class Transmitter:
    """Map an information payload to symbols, and optionally to real passband.

    Parameters
    ----------
    modulation : str
        Constellation name (see :class:`~uacpy.comms.modulate.Modulator`).
    code : ConvCode, optional
        FEC codec applied before modulation.
    preamble : array_like or int, optional
        Known leading symbols for the receiver's sync + training. An int means
        "generate this many" (matched by :class:`CommsReceiver` with the same count).
    """

    def __init__(self, modulation: str, code: Optional[ConvCode] = None,
                 preamble=None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.code = code
        if preamble is None or np.isscalar(preamble):
            n = 64 if preamble is None else int(np.asarray(preamble).item())
            self.preamble = _seeded_symbols(n, modulation, _PREAMBLE_SEED)
        else:
            self.preamble = np.asarray(preamble, dtype=complex)

    def transmit(self, bits):
        """Information bits -> complex symbols ``[preamble | payload]``."""
        b = self.code.encode(bits) if self.code is not None else np.asarray(bits, int)
        sym = self.modulator.modulate(b)
        return np.concatenate([self.preamble, sym])

    def to_passband(self, symbols, sample_rate, fc, sps=8, rolloff=0.25, span=8):
        """Pulse-shape and up-convert symbols to a real passband signal at ``fc``."""
        _require_passband_fits(sample_rate, fc, sps, rolloff, 'to_passband')
        return upconvert(pulse_shape(symbols, sps, rolloff, span), sample_rate, fc)

    def transmit_passband(self, bits, sample_rate, fc, sps=8, rolloff=0.25, span=8):
        """Information bits straight to real passband samples (one call)."""
        return self.to_passband(self.transmit(bits), sample_rate, fc, sps, rolloff, span)


class CommsReceiver:
    """Recover information bits from symbols or real passband samples.

    Parameters mirror :class:`Transmitter`; ``equalizer`` is an optional
    :class:`~uacpy.comms.receive.DFE` (trained on the preamble, with its PLL
    tracking residual carrier offset). ``preamble`` must match the transmitter's.
    """

    def __init__(self, modulation: str, code: Optional[ConvCode] = None,
                 equalizer: Optional[DFE] = None, preamble=None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.code = code
        self.equalizer = equalizer
        if preamble is None or np.isscalar(preamble):
            n = 64 if preamble is None else int(np.asarray(preamble).item())
            self.preamble = _seeded_symbols(n, modulation, _PREAMBLE_SEED)
        else:
            self.preamble = np.asarray(preamble, dtype=complex)
        if equalizer is not None and getattr(equalizer, 'forget', None) is None:
            # LMS converges in ~20 N symbols against RLS's ~2 N (Istepanian &
            # Stojanovic); a preamble shorter than that leaves the taps
            # half-trained when the payload starts — measured, 16-QAM at
            # step 0.01 with 64 training symbols decoded at BER 0.22 from a
            # 0.8 rad carrier offset, while 256 symbols or RLS gave 0.
            n_taps = int(getattr(equalizer, 'n_ff', 0)) + int(getattr(equalizer, 'n_fb', 0))
            needed = 20 * n_taps
            if self.preamble.size < needed:
                warnings.warn(
                    f"CommsReceiver: the equalizer adapts by LMS, which needs "
                    f"about 20 x {n_taps} = {needed} training symbols to "
                    f"converge, and the preamble has {self.preamble.size}. "
                    f"Higher-order constellations may decode wrongly after a "
                    f"carrier-phase offset. Pass preamble={needed} or more, or "
                    f"DFE(forget=0.99...) for RLS, which converges in ~2 N.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def from_passband(self, samples, sample_rate, fc, sps=8, rolloff=0.25, span=8,
                      loop_bw=0.005):
        """Down-convert, matched-filter, and timing-recover to symbol-rate samples."""
        _require_passband_fits(sample_rate, fc, sps, rolloff, 'from_passband')
        bb = downconvert(np.asarray(samples, dtype=float), sample_rate, fc)
        mf = matched_filter(bb, sps, rolloff, span)
        return symbol_sync(mf, sps, loop_bw=loop_bw, start=span * sps)

    def receive(self, symbols, threshold=0.4):
        """Symbols ``[preamble | payload]`` -> information bits.

        Detects the preamble (frame sync), trains the equalizer on it, then
        equalizes/demodulates/decodes the payload. Without an equalizer the
        known preamble still sets the carrier phase and gain — one complex
        least-squares scalar ``<pre, rx_pre> / <pre, pre>`` divides the
        payload — and the payload is assumed to start exactly
        ``len(preamble)`` symbols after the detected start: residual channel
        delay spread leaks preamble ISI into the first payload symbols, and
        nothing here tracks a phase that drifts through the frame (an
        equalizer with ``pll_bandwidth`` does).
        """
        sym = np.asarray(symbols, dtype=complex).ravel()
        start = 0
        k, metric = detect_preamble(sym, self.preamble, threshold=threshold)
        if k is not None:
            start = k
        else:
            warnings.warn(
                f"CommsReceiver.receive: preamble not detected (best metric "
                f"{float(np.max(metric)):.3f} < threshold {float(threshold):.3f}); "
                f"decoding from sample 0. The returned bits are not frame-aligned "
                f"and carry no indication of that.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
        sym = sym[start:]
        pre = self.preamble
        if self.equalizer is not None:
            delay = self.equalizer.n_ff // 2
            ref = np.concatenate([np.zeros(delay, dtype=complex), pre])
            eq, _ = self.equalizer.equalize(sym, self.modulator.constellation, train=ref)
            payload = eq[delay + pre.size:]
        else:
            # A passband record delayed by a fraction of a sample arrives
            # with tens of degrees of carrier phase, and any gain moves the
            # QAM decision rings: measured, QPSK decoded at BER 0.50 and
            # 16-QAM at 0.44 with the preamble detected and no warning. The
            # preamble is known, so its least-squares complex gain is one
            # vdot away.
            rx_pre = sym[:pre.size]
            denom = np.vdot(pre, pre)
            gain = np.vdot(pre, rx_pre) / denom if denom else 1.0
            payload = sym[pre.size:]
            if gain and np.isfinite(gain):
                payload = payload / gain
        bits = self.modulator.demodulate(payload)
        if self.code is not None:
            bits = self.code.decode(bits)
        return bits

    def receive_passband(self, samples, sample_rate, fc, sps=8, rolloff=0.25, span=8,
                         loop_bw=0.005, threshold=0.4):
        """Real passband samples straight to information bits (one call)."""
        syms = self.from_passband(samples, sample_rate, fc, sps, rolloff, span, loop_bw)
        return self.receive(syms, threshold=threshold)


class OFDMTransmitter:
    """OFDM passband transmitter (FEC + QAM + Schmidl-Cox preamble + pilot + CP).

    Frame layout: ``[SC preamble | pilot symbol | data symbols...]``, each block a
    cyclic-prefixed OFDM symbol.

    Parameters
    ----------
    modulation : str
        Subcarrier constellation (see :class:`~uacpy.comms.modulate.Modulator`).
    n_subcarriers, cp_len : int
        FFT size and cyclic-prefix length.
    code : ConvCode, optional
        FEC codec applied before mapping.
    """

    def __init__(self, modulation: str, n_subcarriers: int = 256,
                 cp_len: int = 32, code: Optional[ConvCode] = None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.n_subcarriers = int(n_subcarriers)
        self.cp_len = int(cp_len)
        self.code = code
        self.preamble = schmidl_cox_preamble(self.n_subcarriers, self.cp_len)
        self.pilot_freq = _seeded_symbols(self.n_subcarriers, modulation,
                                          _PILOT_SEED)

    def transmit(self, bits):
        """Information bits -> baseband OFDM frame (complex time samples)."""
        b = self.code.encode(bits) if self.code is not None else np.asarray(bits, int)
        sym = self.modulator.modulate(b)
        nsc = self.n_subcarriers
        if sym.size % nsc:
            sym = np.concatenate([sym, np.zeros(nsc - sym.size % nsc, dtype=complex)])
        data = [ofdm_symbol(sym[i:i + nsc], nsc, self.cp_len)
                for i in range(0, sym.size, nsc)]
        pilot = ofdm_symbol(self.pilot_freq, nsc, self.cp_len)
        guard = np.zeros(nsc + self.cp_len, dtype=complex)   # protects the last block
        return np.concatenate([self.preamble, pilot] + data + [guard])

    def to_passband(self, baseband, sample_rate, fc, oversample=4):
        """Up-convert a baseband OFDM frame to real passband at carrier ``fc``.

        The baseband is interpolated by ``oversample`` so the OFDM band occupies
        ``sample_rate/oversample`` Hz around ``fc`` (leaving room in the passband and an
        image gap the receiver's decimation filter rejects).
        """
        from scipy.signal import resample_poly
        os = int(oversample)
        if fc - sample_rate / (2 * os) <= 0 or fc + sample_rate / (2 * os) >= sample_rate / 2:
            raise ConfigurationError(
                "to_passband: OFDM band fc +/- sample_rate/(2*oversample) "
                "must lie in (0, sample_rate/2); got fc="
                f"{float(fc):g} Hz, sample_rate={float(sample_rate):g} Hz, "
                f"oversample={os} — band "
                f"{fc - sample_rate / (2 * os):g}-"
                f"{fc + sample_rate / (2 * os):g} Hz against Nyquist "
                f"{sample_rate / 2:g} Hz")
        up = resample_poly(baseband, os, 1)
        return upconvert(up, sample_rate, fc)

    def transmit_passband(self, bits, sample_rate, fc, oversample=4):
        """Information bits straight to real passband samples (one call)."""
        return self.to_passband(self.transmit(bits), sample_rate, fc, oversample)


class OFDMReceiver:
    """OFDM passband receiver: resample (Doppler) + Schmidl-Cox + residual CFO.

    Implements the practical underwater multicarrier receiver — estimate and
    remove the common Doppler scale by resampling, then correct the residual
    carrier frequency offset, FFT each block, estimate the channel from the pilot
    symbol, and equalize each subcarrier (ZF, or MMSE with ``snr_linear``).

    Parameters mirror :class:`OFDMTransmitter`.
    """

    def __init__(self, modulation: str, n_subcarriers: int = 256,
                 cp_len: int = 32, code: Optional[ConvCode] = None,
                 snr_linear=None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.n_subcarriers = int(n_subcarriers)
        self.cp_len = int(cp_len)
        self.code = code
        self.snr_linear = snr_linear
        self.preamble = schmidl_cox_preamble(self.n_subcarriers, self.cp_len)
        self.pilot_freq = _seeded_symbols(self.n_subcarriers, modulation,
                                          _PILOT_SEED)

    def receive(self, baseband):
        """Baseband OFDM frame -> information bits (sync, channel est, equalize).

        Every whole block after the pilot is decoded as data — the
        transmitter's trailing zero guard block and any extra captured
        samples included — so the returned stream runs past the payload (the
        guard alone contributes ``n_subcarriers * bits_per_symbol`` coded
        bits of noise). Slice the result to the known payload length.
        """
        nsc, cp = self.n_subcarriers, self.cp_len
        blk = nsc + cp
        x = np.asarray(baseband, dtype=complex).ravel()
        start, cfo = schmidl_cox_sync(x, nsc)
        if start is None:
            warnings.warn(
                "OFDMReceiver.receive: Schmidl-Cox timing metric never reached "
                "the 0.5 plateau threshold, so no preamble was found; decoding "
                "from sample 0 with cfo=0. The returned bits are not frame-"
                "aligned and carry no indication of that.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
            start = 0
        x = apply_cfo(x[start:], cfo)
        nblocks = x.size // blk
        if nblocks < 3:
            raise ConfigurationError(
                "OFDMReceiver: frame too short (need preamble+pilot+data); "
                f"got {nblocks} block(s) of {blk} samples from {x.size} "
                f"samples, need >= 3")

        h = estimate_channel(x[blk:2 * blk], self.pilot_freq, nsc, cp)
        c = self.modulator.constellation
        data = []
        # Every whole block after the pilot, CP stripped and transformed;
        # equalised here (channel=None) by the pilot estimate h rather than
        # by a channel the caller describes.
        spectra = ofdm_demodulate(x[2 * blk:], nsc, cp, channel=None)
        for d in spectra.reshape(-1, nsc):
            d = equalize_subcarriers(d, h, self.snr_linear)
            # decision-directed common-phase-error correction (residual CFO drift)
            dec = slicer(d, c)
            d *= np.exp(-1j * np.angle(np.vdot(dec, d)))
            data.append(d)
        syms = np.concatenate(data) if data else np.array([], dtype=complex)
        bits = self.modulator.demodulate(syms)
        if self.code is not None:
            bits = self.code.decode(bits)
        return bits

    def from_passband(self, samples, sample_rate, fc, oversample=4, doppler_scale=None,
                      scales=None):
        """Resample for Doppler, down-convert, and decimate to a baseband frame.

        Estimates and removes the common Doppler scale by resampling the passband
        (``doppler_scale=None`` estimates it from the known passband preamble),
        then down-converts and decimates by ``oversample`` (the polyphase filter
        rejects the image), leaving residual CFO for :meth:`receive`.
        """
        from scipy.signal import resample_poly
        os = int(oversample)
        pb = np.asarray(samples, dtype=float)
        if doppler_scale is None:
            probe = upconvert(resample_poly(self.preamble, os, 1), sample_rate, fc)
            doppler_scale, _, _ = estimate_doppler_scale(pb, probe, scales)
        if abs(doppler_scale) > 1e-9:
            # doppler_scale is a = v/c; compensate_doppler(pb, a) removes it.
            pb = np.real(compensate_doppler(pb, doppler_scale))
        bb = downconvert(pb, sample_rate, fc)
        return resample_poly(bb, 1, os)          # LPF + decimate removes 2*fc image

    def receive_passband(self, samples, sample_rate, fc, oversample=4, doppler_scale=None,
                         scales=None):
        """Real passband samples straight to information bits (one call)."""
        return self.receive(self.from_passband(samples, sample_rate, fc, oversample,
                                               doppler_scale, scales))
