"""Passband physical layer: pulse shaping, up/down-conversion, timing recovery.

Bridges the complex-baseband symbol domain (where the rest of ``uacpy.comms``
lives) and real passband samples at a hardware sample rate — what a transducer
emits and a hydrophone records. The chain is:

    symbols --RRC pulse shape--> baseband --upconvert(fc)--> real samples
    real samples --downconvert(fc)--> baseband --matched RRC + timing recovery--> symbols

Transmit shaping uses a root-raised-cosine filter (≥2 samples/symbol); the
receiver's matched RRC completes the raised-cosine (Nyquist, zero-ISI) response.
A Gardner timing-recovery loop locks the symbol clock from the data itself.

References
----------
Istepanian & Stojanovic. *Underwater Acoustic DSP & Comms* — "signals are shaped
    at the transmitter… raised-cosine spectrum shaping… roll-off factor α";
    baseband front end translates to baseband, low-pass filters, frame-syncs.
Gardner (1986), *A BPSK/QPSK timing-error detector for sampled receivers*, IEEE
    Trans. Comms — the non-data-aided timing-error detector used here.
Proakis & Salehi. *Digital Communications*, eqs. (9.2-28)/(9.2-29) — with a
    matched receiver filter ``G_T(f) = sqrt(|X_rc(f)|)`` and ``G_R(f) =
    G_T*(f)``, so "the overall raised cosine spectral characteristic is split
    evenly between the transmitting filter and the receiving filter"; eq.
    (5.2-22) for the loop noise-equivalent bandwidth used in `symbol_sync`.
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError


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
