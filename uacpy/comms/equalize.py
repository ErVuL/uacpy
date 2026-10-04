"""Undoing the delay spread: least-squares and sparse (OMP) channel
estimates, the MMSE, LMS and RLS equalisers, and the decision-feedback
equaliser (:class:`DFE`) with its optional carrier PLL."""

from __future__ import annotations

import numpy as np

from uacpy.comms.channel import _channel_array
from uacpy.comms.constellations import slicer
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._repr import SettingsRepr


#: The LMS step size and the RLS forgetting factor the adaptive
#: equalizers default to.
LMS_DEFAULT_STEP = 0.01
RLS_DEFAULT_FORGET = 0.99


#: The RLS initialisation ``P(0) = I / RLS_INIT_DELTA``.
RLS_INIT_DELTA = 1e-2


#: Column-norm floor for OMP atom normalisation, RELATIVE to the largest
#: column norm. Guards the division against an all-zero column without
#: putting an absolute scale into a dimensional quantity.
_COLUMN_NORM_REL_FLOOR = 1e-12


def _conv_matrix(tx, n_taps, n_rows):
    """Tall convolution matrix ``A`` with ``A[k, l] = tx[k - l]`` (causal)."""
    if n_taps > n_rows:
        raise ConfigurationError(
            f"channel estimate: n_taps ({n_taps}) exceeds the number of "
            f"available pilot/received samples ({n_rows}); reduce n_taps or "
            "provide more pilots."
        )
    p = np.asarray(tx, dtype=complex)
    A = np.zeros((n_rows, n_taps), dtype=complex)
    for l in range(n_taps):
        A[l:, l] = p[: n_rows - l]
    return A


def ls_estimate(received, tx_pilots, n_taps):
    """Least-squares estimate of an ``n_taps`` channel from known pilots.

    Solves ``received ≈ A h`` where ``A`` is the pilot convolution matrix. Returns the
    complex tap vector ``h``.

    Parameters
    ----------
    received : array_like
        The received samples over the pilots.
    tx_pilots : array_like
        The transmitted pilot samples.
    n_taps : int
        Channel taps to estimate.
    """
    r = np.asarray(received, dtype=complex).ravel()
    n = min(r.size, np.asarray(tx_pilots).size)
    A = _conv_matrix(tx_pilots, n_taps, n)
    h, *_ = np.linalg.lstsq(A, r[:n], rcond=None)
    return h


def omp_estimate(received, tx_pilots, n_taps, sparsity):
    """Sparse channel estimate via orthogonal matching pursuit.

    Recovers at most ``sparsity`` non-zero taps out of ``n_taps`` candidate
    delays — the right model for the sparse UW multipath channel. Returns the
    full ``n_taps`` complex vector (zeros off the support).

    Parameters
    ----------
    received : array_like
        The received samples over the pilots.
    tx_pilots : array_like
        The transmitted pilot samples.
    n_taps : int
        Candidate delays (taps).
    sparsity : int
        Non-zero taps to recover, in ``1..n_taps``.
    """
    if sparsity < 1 or sparsity > n_taps:
        raise ConfigurationError(
            f"omp_estimate: need 1 <= sparsity <= n_taps; got "
            f"sparsity={sparsity}, n_taps={n_taps}.")
    r = np.asarray(received, dtype=complex).ravel()
    n = min(r.size, np.asarray(tx_pilots).size)
    A = _conv_matrix(tx_pilots, n_taps, n)
    y = r[:n]
    residual = y.copy()
    support: list[int] = []
    coeffs = np.zeros(0, dtype=complex)
    # The selection rule |A^H r| / norms is scale-invariant on its own:
    # scaling the pilots and the record together scales every projection
    # identically, so the argmax does not move. A bare ``+ 1e-12`` breaks
    # that, because the offset is absolute while the norms are dimensional.
    # It bites whenever adjacent columns are near-degenerate, which is the
    # NORMAL condition here — neighbouring delay lags are highly correlated.
    # Measured on a 40-lag grid whose column norms span 49x, with the top
    # three normalised projections tied to six decimals: the epsilon is
    # 2.3e-11 of the smallest norm at unit scale (harmless, support [3, 30])
    # and 2.3e-5 at 1e-6 scale, where it reorders the tie and returns
    # [0, 30]. The same channel in Pa rather than µPa estimated a different
    # delay. A relative floor keeps the guard against an all-zero column —
    # the only thing the offset was needed for — without setting a scale.
    # Same idiom as ``_ZF_REL_FLOOR`` / ``_PILOT_REL_FLOOR`` in comms.ofdm.
    raw_norms = np.linalg.norm(A, axis=0)
    peak_norm = float(raw_norms.max()) if raw_norms.size else 0.0
    norms = np.maximum(raw_norms, _COLUMN_NORM_REL_FLOOR * peak_norm) \
        if peak_norm > 0.0 else np.ones_like(raw_norms)
    for _ in range(int(sparsity)):
        proj = np.abs(A.conj().T @ residual) / norms
        proj[support] = -1.0
        support.append(int(np.argmax(proj)))
        As = A[:, support]
        coeffs, *_ = np.linalg.lstsq(As, y, rcond=None)
        residual = y - As @ coeffs
    h = np.zeros(n_taps, dtype=complex)
    h[support] = coeffs
    return h


# Zero-forcing floor, as a fraction of the channel's own peak power |H|^2.
# Relative because |H|^2 carries whatever amplitude scale the caller's channel
# is in: an absolute floor silently makes the result a function of those units
# — a channel holding 1e-5 of propagation gain equalised to EVM 0.014, and
# 1e-6 to 0.54. `acoustic_signal.frf._etfe_divide` states the same rule for
# a transfer function, and `comms.ofdm._PILOT_REL_FLOOR` for a pilot magnitude.
_ZF_REL_FLOOR = 1e-12


def regularizer(h2, snr_linear):
    """Denominator offset for ``conj(H)/(|H|^2 + eps)``, in the units of ``h2``.

    Zero-forcing (``snr_linear is None``) uses a floor at ``_ZF_REL_FLOOR`` of
    the peak subcarrier power, which only keeps a spectral null finite. MMSE
    uses the physical noise-to-signal ratio, expressed in those same units:
    ``snr_linear`` is the SNR at the equalizer input, so the noise power that
    goes with it is ``mean(|H|^2) / snr``. Both return 0.0 for a channel with
    no power at all, which leaves the caller's ``0/0`` to be masked there.
    """
    peak = float(h2.max()) if h2.size else 0.0
    if peak <= 0.0:
        return 0.0
    if snr_linear is None:
        return _ZF_REL_FLOOR * peak
    return float(h2.mean()) / float(snr_linear)


def complex_gain(reference, received):
    """The least-squares complex gain ``g`` of ``received ≈ g · reference``:
    ``<reference, received> / <reference, reference>``, with ``<a, b>`` the
    conjugating inner product (:func:`numpy.vdot`). One complex number that
    restores a known sequence's amplitude and carrier phase — the one-tap
    equaliser a receiver applies with a training sequence in hand.

    NaN when ``reference`` carries no energy, since then no gain is
    defined.

    Parameters
    ----------
    reference : array_like
        The known sequence.
    received : array_like
        The received copy of it.
    """
    reference = np.asarray(reference, dtype=complex).ravel()
    received = np.asarray(received, dtype=complex).ravel()
    denom = np.vdot(reference, reference)
    return np.vdot(reference, received) / denom if denom else complex(np.nan)


def mmse_equalizer(received, h, snr_linear):
    """Block MMSE (Wiener) equalization of ``received`` for a known channel ``h``.

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

    The FFT makes the equalization **circular**: ``received`` must carry a cyclic
    prefix of at least ``len(h)-1`` samples, or the first ``len(h)-1`` outputs
    (which wrap the linear-convolution tail) must be discarded. For the
    CP-based path see :func:`uacpy.comms.ofdm.ofdm_demodulate`.

    Returns the equalized signal only (a single ndarray). Unlike the *adaptive*
    :func:`lms_equalizer` / :func:`rls_equalizer`, which return
    ``(equalized, mse)`` because they converge over symbols, this is a one-shot
    block (Wiener) solution with no per-symbol learning curve.

    Parameters
    ----------
    received : array_like
        The received samples, at least as long as ``h``.
    h : array_like or ChannelTaps
        The channel impulse response.
    snr_linear : float
        SNR at the equalizer input, a linear power ratio > 0.
    """
    r = np.asarray(received, dtype=complex)
    snr = float(snr_linear)
    if not snr > 0.0:
        raise ConfigurationError(
            f"mmse_equalizer: snr_linear must be > 0 (linear power ratio, not "
            f"dB); got {snr_linear!r}. A non-positive value makes the Wiener "
            "regularizer 1/snr zero or negative, which un-damps the inverse.")
    hc = np.asarray(_channel_array(h), dtype=complex).ravel()
    if hc.size > r.size:
        # np.fft.fft(h, r.size) would truncate the channel to the transform
        # length and equalize a channel the caller never described.
        raise ConfigurationError(
            f"mmse_equalizer: channel h has {hc.size} taps but received is only "
            f"{r.size} samples, so the length-{r.size} transform would drop "
            f"the tail of h. The circular equalization needs received at least as "
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


def lms_equalizer(received, constellation, n_taps=11, step=LMS_DEFAULT_STEP,
                  train=None):
    """Symbol-spaced linear LMS equalizer. Returns ``(eq_symbols, mse)``.

    Trains on ``train`` symbols while available, then switches to
    decision-directed mode. ``constellation`` is the (Gray-mapped) symbol set.

    Alignment: the taps start as a centre spike at index ``n_taps // 2``, so
    the output at step ``k`` estimates the symbol received ``n_taps // 2``
    samples earlier. ``train`` must therefore be the transmitted symbols
    delayed by ``n_taps // 2`` (zero-padded in front), and ``eq_symbols``
    lags ``received`` by the same amount: ``eq_symbols[n_taps // 2:]`` lines up
    with the transmitted sequence. :func:`~uacpy.comms.link.simulate_link`
    applies both. An undelayed ``train`` sets LMS chasing a target one
    spike-width away (measured with LMS on 16-QAM: 1e-2 BER on a noiseless
    identity channel; QPSK hides it because its decisions ignore amplitude).

    Parameters
    ----------
    received : array_like
        The received symbols, one sample per symbol.
    constellation : array_like
        The symbol set the decisions are sliced to (Gray-mapped).
    n_taps : int, optional
        Equalizer taps. Default 11.
    step : float, optional
        LMS step size. Default 0.01.
    train : array_like, optional
        Training symbols, delayed by ``n_taps // 2`` (see above); after them
        the equalizer runs decision-directed. ``None`` runs
        decision-directed from the start.
    """
    return _dfe_core(received, constellation, n_taps, 0, step, None, 0.0, train)


def rls_equalizer(received, constellation, n_taps=11,
                  forget=RLS_DEFAULT_FORGET, train=None):
    """Symbol-spaced linear RLS equalizer (faster convergence than LMS).

    Istepanian & Stojanovic put RLS convergence at ~``2N`` symbol intervals
    against LMS's ~``20N``, for ``N`` the total adaptive coefficient count, at
    higher per-symbol cost. Returns ``(eq_symbols, mse)``.

    Same alignment as :func:`lms_equalizer`: delay ``train`` by
    ``n_taps // 2`` and read ``eq_symbols[n_taps // 2:]``. RLS re-learns a
    causal filter within a few symbols on a minimum-phase channel, so an
    undelayed ``train`` is usually recovered, but the delayed form is the
    contract.

    Parameters
    ----------
    received : array_like
        The received symbols, one sample per symbol.
    constellation : array_like
        The symbol set the decisions are sliced to (Gray-mapped).
    n_taps : int, optional
        Equalizer taps. Default 11.
    forget : float, optional
        RLS forgetting factor, in ``(0, 1]``. Default 0.99.
    train : array_like, optional
        Training symbols, delayed by ``n_taps // 2`` (see above); after them
        the equalizer runs decision-directed. ``None`` runs
        decision-directed from the start.
    """
    return _dfe_core(received, constellation, n_taps, 0, 0.0, forget, 0.0, train)


class DFE(SettingsRepr):
    """Adaptive decision-feedback equalizer with an optional carrier-phase PLL.

    ``n_ff`` feedforward taps act on the received samples; ``n_fb`` feedback taps
    cancel ISI from past *decisions*. Adapt with LMS (``step``) or RLS
    (``forget``). Set ``pll_gain > 0`` to track residual carrier phase
    jointly with equalization (the key UW-channel enhancement).

    Parameters
    ----------
    n_ff, n_fb : int
        Feedforward / feedback tap counts.
    step : float
        LMS step size (used when ``forget`` is None).
    forget : float, optional
        RLS forgetting factor in (0, 1]; enables RLS adaptation.
    pll_gain : float
        Proportional gain ``kp`` of the second-order carrier PLL, in radians
        of phase correction per radian of phase error per symbol; the
        integral gain is ``kp**2/4`` (critical damping). ``0`` disables the
        PLL.
    """

    def __init__(self, n_ff: int = 12, n_fb: int = 6, *,
                 step: float = LMS_DEFAULT_STEP,
                 forget=None, pll_gain: float = 0.0):
        self.n_ff = int(n_ff)
        self.n_fb = int(n_fb)
        self.step = float(step)
        self.forget = forget
        self.pll_gain = float(pll_gain)

    @property
    def output_delay(self) -> int:
        """Symbols by which :meth:`equalize`'s output lags ``rx``:
        ``n_ff // 2``, the centre-spike tap index. Delay ``train`` by this
        many symbols and read ``eq_symbols[output_delay:]``."""
        return self.n_ff // 2

    def equalize(self, received, constellation, train=None):
        """Equalize ``received`` (symbol-spaced). Returns ``(eq_symbols, mse)``.

        ``train`` must be the transmitted symbols delayed by
        :attr:`output_delay` (zero-padded in front); the output lags ``received``
        by the same amount, so ``eq_symbols[output_delay:]`` lines up with
        the transmitted sequence (see :func:`lms_equalizer`)."""
        return _dfe_core(received, constellation, self.n_ff, self.n_fb, self.step,
                         self.forget, self.pll_gain, train)


def _dfe_core(received, constellation, n_ff, n_fb, step, forget, pll_gain, train):
    received = np.asarray(received, dtype=complex).ravel()
    c = np.asarray(constellation, dtype=complex)
    N = received.size
    ntaps = n_ff + n_fb
    if n_ff < 1:
        raise ConfigurationError(
            f"equalizer: n_ff must be >= 1; got {n_ff}.")
    if forget is not None and not 0.0 < float(forget) <= 1.0:
        raise ConfigurationError(
            f"equalizer: the RLS forgetting factor must be in (0, 1]; got "
            f"{forget!r}. The inverse-correlation update divides by it, so 0 "
            f"or a negative value makes the taps non-finite."
        )
    # Bring the record to unit mean power before adapting: the LMS step
    # bound, RLS's P(0), the centre-spike init and the unit-energy slicer all
    # carry an absolute scale, so unnormalised the answer depended on the
    # caller's units (test_comms_equalizers pins the amplitude ladder). Done here
    # rather than tap-by-tap because the register mixes record-scale samples
    # with constellation-scale decisions; a no-op at unit input power.
    p_in = float(np.mean(np.abs(received) ** 2)) if received.size else 0.0
    if np.isfinite(p_in) and p_in > 0.0:
        received = received / np.sqrt(p_in)
    w = np.zeros(ntaps, dtype=complex)
    w[n_ff // 2] = 1.0                       # center-spike feedforward init
    uff = np.zeros(n_ff, dtype=complex)      # feedforward register (newest first)
    ufb = np.zeros(n_fb, dtype=complex)      # feedback register (past decisions)
    theta = 0.0
    phase_acc = 0.0
    kp = float(pll_gain)
    ki = kp * kp / 4.0                       # critically damped: kp = 2*z*wn, ki = wn^2 at z = 1
    use_rls = forget is not None
    if use_rls:
        lam = float(forget)
        P = np.eye(ntaps, dtype=complex) / RLS_INIT_DELTA   # P(0) = I/delta
    ntrain = 0 if train is None else len(np.asarray(train))
    train = None if train is None else np.asarray(train, dtype=complex)
    out = np.empty(N, dtype=complex)
    mse = np.empty(N)
    for k in range(N):
        uff = np.roll(uff, 1); uff[0] = received[k]
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
