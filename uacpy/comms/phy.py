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
Proakis & Salehi. *Digital Communications* (matched filtering, synchronization).
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def rrc_filter(sps, rolloff, span):
    """Root-raised-cosine taps: ``span`` symbols, ``sps`` samples/symbol, unit energy."""
    if not 0.0 <= rolloff <= 1.0:
        raise ConfigurationError("rrc_filter: rolloff must be in [0, 1]")
    n = span * sps
    t = (np.arange(n + 1) - n / 2) / sps      # time in symbol periods
    b = float(rolloff)
    h = np.empty_like(t)
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
    """Mix a real passband signal down to complex baseband (image left for the LPF/MF)."""
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
        (``span*sps``) for clean lock from the first symbol.
    """
    x = np.asarray(samples, dtype=complex).ravel()
    if sps < 2:
        raise ConfigurationError("symbol_sync: need sps >= 2 for Gardner")
    theta = loop_bw / (damping + 0.25 / damping)
    denom = 1 + 2 * damping * theta + theta * theta
    kp = 4 * damping * theta / denom
    ki = 4 * theta * theta / denom
    power = np.mean(np.abs(x) ** 2) + 1e-12   # normalize TED gain to signal level

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
        e = float(np.clip(e, -2.0, 2.0))
        vi = float(np.clip(vi + ki * e, -0.25, 0.25))
        adj = float(np.clip(kp * e + vi, -0.5, 0.5))
        out.append(y_on)
        prev_on = y_on
        mu += adj
        idx += sps + int(np.floor(mu))
        mu -= np.floor(mu)
    return np.array(out, dtype=complex)
