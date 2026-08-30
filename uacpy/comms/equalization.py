"""Adaptive channel equalization for intersymbol-interference (ISI) removal.

ISI from time-varying multipath is *the* dominant impairment of underwater
acoustic links, and adaptive equalization is the classical fix (Istepanian &
Stojanovic). This module provides linear LMS/RLS equalizers, an MMSE block
equalizer (known channel), and the adaptive **decision-feedback equalizer
(DFE)** with an optional carrier-phase PLL — the Stojanovic-Proakis phase-
coherent receiver, since *"carrier phase is the most rapidly changing parameter
in the underwater acoustic channel."*

References
----------
Proakis & Salehi. *Digital Communications* (LMS/RLS, DFE, MMSE equalizers).
Stojanovic, Catipovic & Proakis (1994), *Phase-coherent digital communications
    for underwater acoustic channels*, IEEE JOE — joint DFE + PLL.
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.comms._equalizer_core import regularizer


def slicer(x, constellation):
    """Nearest constellation point(s) to ``x``."""
    c = np.asarray(constellation, dtype=complex)
    x = np.atleast_1d(np.asarray(x, dtype=complex))
    return c[np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)]


def mmse_equalizer(rx, h, snr_linear):
    """Block MMSE (Wiener) equalization of ``rx`` for a known channel ``h``.

    Frequency-domain ``W(f) = H*(f) / (|H(f)|^2 + mean(|H|^2)/snr)``.
    ``snr_linear`` is the operating SNR **at the equalizer input** — received
    signal power over noise power — so it means the same thing here as in
    :func:`uacpy.comms.ofdm.ofdm_demodulate`, and the same number can be
    calibrated once and passed to either. The Wiener regularizer is a
    noise-to-signal power ratio and so must be expressed in the units of
    ``|H|^2``: writing it as a bare ``1/snr`` assumed a channel of unit mean
    power, and equalizing the same link with ``h`` scaled by a propagation
    gain then changed the damping instead of leaving it alone.
    ``snr_linear -> inf`` gives the zero-forcing inverse.

    The FFT makes the equalization **circular**: ``rx`` must carry a cyclic
    prefix of at least ``len(h)-1`` samples, or the first ``len(h)-1`` outputs
    (which wrap the linear-convolution tail) must be discarded. For the
    CP-based path see :func:`uacpy.comms.ofdm.ofdm_demodulate`.

    Returns the equalized signal only (a single ndarray). Unlike the *adaptive*
    :func:`lms_equalizer` / :func:`rls_equalizer`, which return
    ``(equalized, mse)`` because they converge over symbols, this is a one-shot
    block (Wiener) solution with no per-symbol learning curve.
    """
    r = np.asarray(rx, dtype=complex)
    snr = float(snr_linear)
    if not snr > 0.0:
        raise ConfigurationError(
            f"mmse_equalizer: snr_linear must be > 0 (linear power ratio, not "
            f"dB); got {snr_linear!r}. A non-positive value makes the Wiener "
            "regularizer 1/snr zero or negative, which un-damps the inverse.")
    hc = np.asarray(h, dtype=complex).ravel()
    if hc.size > r.size:
        # np.fft.fft(h, r.size) would truncate the channel to the transform
        # length and equalize a channel the caller never described.
        raise ConfigurationError(
            f"mmse_equalizer: channel h has {hc.size} taps but rx is only "
            f"{r.size} samples, so the length-{r.size} transform would drop "
            f"the tail of h. The circular equalization needs rx at least as "
            f"long as h (and a cyclic prefix of >= {hc.size - 1} samples).")
    H = np.fft.fft(hc, r.size)
    h2 = np.abs(H) ** 2
    eps = regularizer(h2, snr)
    if eps <= 0.0:
        # A channel with no power anywhere: nothing is recoverable, which is
        # the all-zero output conj(H)/(0 + 1/snr) already gave.
        return np.zeros_like(r)
    W = np.conj(H) / (h2 + eps)
    return np.fft.ifft(np.fft.fft(r) * W)


def lms_equalizer(rx, constellation, n_taps=11, step=0.01, train=None):
    """Symbol-spaced linear LMS equalizer. Returns ``(eq_symbols, mse)``.

    Trains on ``train`` symbols while available, then switches to
    decision-directed mode. ``constellation`` is the (Gray-mapped) symbol set.
    """
    return _dfe_core(rx, constellation, n_taps, 0, step, None, 0.0, train)


def rls_equalizer(rx, constellation, n_taps=11, forget=0.99, train=None):
    """Symbol-spaced linear RLS equalizer (faster convergence than LMS).

    Istepanian & Stojanovic put RLS convergence at ~``2N`` symbol intervals
    against LMS's ~``20N``, for ``N`` the total adaptive coefficient count, at
    higher per-symbol cost. Returns ``(eq_symbols, mse)``.
    """
    return _dfe_core(rx, constellation, n_taps, 0, 0.0, forget, 0.0, train)


class DFE:
    """Adaptive decision-feedback equalizer with an optional carrier-phase PLL.

    ``n_ff`` feedforward taps act on the received samples; ``n_fb`` feedback taps
    cancel ISI from past *decisions*. Adapt with LMS (``step``) or RLS
    (``forget``). Set ``pll_bandwidth > 0`` to track residual carrier phase
    jointly with equalization (the key UW-channel enhancement).

    Parameters
    ----------
    n_ff, n_fb : int
        Feedforward / feedback tap counts.
    step : float
        LMS step size (used when ``forget`` is None).
    forget : float, optional
        RLS forgetting factor in (0, 1]; enables RLS adaptation.
    pll_bandwidth : float
        Proportional PLL gain (0 disables the PLL).
    """

    def __init__(self, n_ff: int = 12, n_fb: int = 6, *, step: float = 0.01,
                 forget=None, pll_bandwidth: float = 0.0):
        self.n_ff = int(n_ff)
        self.n_fb = int(n_fb)
        self.step = float(step)
        self.forget = forget
        self.pll_bandwidth = float(pll_bandwidth)

    def equalize(self, rx, constellation, train=None):
        """Equalize ``rx`` (symbol-spaced). Returns ``(eq_symbols, mse)``."""
        return _dfe_core(rx, constellation, self.n_ff, self.n_fb, self.step,
                         self.forget, self.pll_bandwidth, train)


def _dfe_core(rx, constellation, n_ff, n_fb, step, forget, pll_bw, train):
    rx = np.asarray(rx, dtype=complex).ravel()
    c = np.asarray(constellation, dtype=complex)
    N = rx.size
    ntaps = n_ff + n_fb
    if n_ff < 1:
        raise ConfigurationError(
            f"equalizer: n_ff must be >= 1; got {n_ff}")
    if forget is not None and not 0.0 < float(forget) <= 1.0:
        raise ConfigurationError(
            f"equalizer: the RLS forgetting factor must be in (0, 1]; got "
            f"{forget!r}. The inverse-correlation update divides by it, so 0 "
            f"or a negative value makes the taps non-finite."
        )
    # Bring the record to unit mean power before adapting. Everything below
    # carries an absolute scale — LMS's stability bound is step < 2/(ntaps*P),
    # RLS's P(0) = I/1e-2 is an inverse-power, the center-spike init is 1.0,
    # and the slicer compares against a unit-energy constellation — so the
    # answer depended on the units the caller held the signal in. Measured on
    # a 16-QAM passband link through CommsReceiver, BER was 0.0000 at unit
    # amplitude but 0.2375 at 0.3x, 0.3600 at 3x and ~0.48 at 1e-9 or 1e9:
    # silent below the window, and above it only a raw numpy overflow from
    # abs(e)**2. Normalising here rather than tap-by-tap is what actually
    # works: the register is mixed-scale by construction (feedforward samples
    # at the record's amplitude, feedback decisions at constellation
    # amplitude), so a single normalised-LMS divisor over-corrects one half
    # and starves the other. At unit input power this is a no-op, so an
    # already-calibrated record is bit-identical.
    p_in = float(np.mean(np.abs(rx) ** 2)) if rx.size else 0.0
    if np.isfinite(p_in) and p_in > 0.0:
        rx = rx / np.sqrt(p_in)
    w = np.zeros(ntaps, dtype=complex)
    w[n_ff // 2] = 1.0                       # center-spike feedforward init
    uff = np.zeros(n_ff, dtype=complex)      # feedforward register (newest first)
    ufb = np.zeros(n_fb, dtype=complex)      # feedback register (past decisions)
    theta = 0.0
    phase_acc = 0.0
    kp = float(pll_bw)
    ki = kp * kp / 4.0                       # critically damped: kp = 2*z*wn, ki = wn^2 at z = 1
    use_rls = forget is not None
    if use_rls:
        lam = float(forget)
        P = np.eye(ntaps, dtype=complex) / 1e-2   # RLS init P(0) = I/delta, delta = 1e-2
    ntrain = 0 if train is None else len(np.asarray(train))
    train = None if train is None else np.asarray(train, dtype=complex)
    out = np.empty(N, dtype=complex)
    mse = np.empty(N)
    for k in range(N):
        uff = np.roll(uff, 1); uff[0] = rx[k]
        # Carrier de-rotation applies to the received (feedforward)
        # section only — the feedback register holds decisions already in
        # the de-rotated constellation domain (Stojanovic-Proakis 1994).
        ur = np.concatenate([uff * np.exp(-1j * theta), ufb])
        d_hat = np.vdot(w, ur)               # w^H u
        d = train[k] if k < ntrain else slicer(d_hat, c)[0]
        e = d - d_hat
        mse[k] = abs(e) ** 2
        if use_rls:
            Pu = P @ ur
            g = Pu / (lam + np.vdot(ur, Pu))
            w = w + g * np.conj(e)
            P = (P - np.outer(g, np.conj(ur) @ P)) / lam
        else:
            w = w + step * ur * np.conj(e)
        if kp > 0:
            phi = np.angle(d_hat * np.conj(d))
            # NCO-based 2nd-order PLL (Stojanovic-Proakis 1994): theta is the
            # NCO phase accumulator, advanced each step by the proportional
            # correction kp*phi plus the integral (frequency) state phase_acc.
            # theta therefore tracks a constant CFO's ramping phase — it is the
            # loop integrator, not a double integration of phi.
            phase_acc += ki * phi
            theta += kp * phi + phase_acc
        ufb = np.roll(ufb, 1)
        if n_fb:
            ufb[0] = d
        out[k] = d_hat
    return out, mse
