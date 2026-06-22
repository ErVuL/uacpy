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


def _slicer(x, constellation):
    """Nearest constellation point(s) to ``x``."""
    c = np.asarray(constellation, dtype=complex)
    x = np.atleast_1d(np.asarray(x, dtype=complex))
    return c[np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)]


def mmse_equalizer(rx, h, snr_linear):
    """Block MMSE (Wiener) equalization of ``rx`` for a known channel ``h``.

    Frequency-domain ``W(f) = H*(f) / (|H(f)|^2 + 1/snr)``. ``snr_linear`` is the
    operating SNR; ``snr_linear -> inf`` gives the zero-forcing inverse.
    """
    r = np.asarray(rx, dtype=complex)
    H = np.fft.fft(np.asarray(h, dtype=complex), r.size)
    W = np.conj(H) / (np.abs(H) ** 2 + 1.0 / float(snr_linear))
    return np.fft.ifft(np.fft.fft(r) * W)


def lms_equalizer(rx, constellation, n_taps=11, step=0.01, train=None):
    """Symbol-spaced linear LMS equalizer. Returns ``(eq_symbols, mse)``.

    Trains on ``train`` symbols while available, then switches to
    decision-directed mode. ``constellation`` is the (Gray-mapped) symbol set.
    """
    return _dfe_core(rx, constellation, n_taps, 0, step, None, 0.0, train)


def rls_equalizer(rx, constellation, n_taps=11, forget=0.99, train=None):
    """Symbol-spaced linear RLS equalizer (faster convergence than LMS).

    Istepanian notes RLS converges in ~``2N`` vs LMS's ~``20N`` taps, at higher
    per-symbol cost. Returns ``(eq_symbols, mse)``.
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
        raise ConfigurationError("equalizer: n_ff must be >= 1")
    w = np.zeros(ntaps, dtype=complex)
    w[n_ff // 2] = 1.0                       # center-spike feedforward init
    uff = np.zeros(n_ff, dtype=complex)      # feedforward register (newest first)
    ufb = np.zeros(n_fb, dtype=complex)      # feedback register (past decisions)
    theta = 0.0
    phase_acc = 0.0
    kp = float(pll_bw)
    ki = kp * kp / 4.0
    use_rls = forget is not None
    if use_rls:
        lam = float(forget)
        P = np.eye(ntaps, dtype=complex) / 1e-2
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
        d = train[k] if k < ntrain else _slicer(d_hat, c)[0]
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
