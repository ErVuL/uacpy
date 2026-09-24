"""Recovering a transmission: synchronisation, channel estimate,
equalisation, Doppler, and what the link achieved.

One question — *turn this recording back into bits* — in the order a receiver
does it: find the preamble, estimate the channel, undo the delay spread,
track the Doppler scale a moving platform imposes, and measure what came out
(:func:`bit_error_rate`, :func:`evm`).
"""

from __future__ import annotations
import numpy as np
from uacpy.core.exceptions import ConfigurationError
from scipy.signal import resample
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from scipy.special import erfc


# ──────────────────────────────────────────────────────────────────────
# Synchronisation
#
# Finding the preamble a frame starts with.
# ──────────────────────────────────────────────────────────────────────

def matched_filter_metric(rx, preamble):
    """Energy-normalized cross-correlation magnitude of ``rx`` with ``preamble``.

    Returns a real metric in ``[0, 1]`` (1 = perfect match) aligned so that
    ``metric[k]`` scores the window ``rx[k:k+len(preamble)]``.
    """
    r = np.asarray(rx, dtype=complex)
    p = np.asarray(preamble, dtype=complex)
    if p.size > r.size:
        raise ConfigurationError(
            f"matched_filter_metric: preamble longer than signal — preamble "
            f"is {p.size} samples, rx {r.size}.")
    corr = np.correlate(r, p, mode="valid")
    pe = np.sum(np.abs(p) ** 2)
    win = np.convolve(np.abs(r) ** 2, np.ones(p.size), mode="valid")
    # Windows with no energy score 0; everywhere else Cauchy-Schwarz bounds
    # the ratio by 1, so no epsilon is needed to keep it finite. An absolute
    # one is wrong here: `pe * win` scales as amplitude**4, so `+ 1e-12` made
    # the metric a function of the units the caller held the record in — a
    # Pa-scale record scored 3e-5 where the same signal in µPa scored 1, and
    # detection failed. Same reasoning as `system_id._etfe_divide`.
    denom = pe * win
    ok = denom > 0.0
    return np.where(ok, np.abs(corr) / np.sqrt(np.where(ok, denom, 1.0)), 0.0)


def detect_preamble(rx, preamble, threshold=0.5):
    """Index of the best preamble alignment if its metric exceeds ``threshold``.

    Returns ``(start_index, metric_array)``; ``start_index`` is ``None`` when no
    peak clears the threshold.
    """
    metric = matched_filter_metric(rx, preamble)
    k = int(np.argmax(metric))
    return (k if metric[k] >= threshold else None), metric


def detect_frames(rx, preamble, threshold=0.5, min_gap=None):
    """All frame-start indices whose metric clears ``threshold`` (peak-picked).

    ``min_gap`` (samples) suppresses detections closer than that to a stronger
    one; defaults to the preamble length. Returns ``(starts, metric_array)``.
    """
    metric = matched_filter_metric(rx, preamble)
    gap = int(min_gap) if min_gap is not None else np.asarray(preamble).size
    cand = np.flatnonzero(metric >= threshold)
    starts = []
    for k in cand[np.argsort(metric[cand])[::-1]]:
        if all(abs(k - s) >= gap for s in starts):
            starts.append(int(k))
    return sorted(starts), metric


# ──────────────────────────────────────────────────────────────────────
# Channel estimation
#
# Least-squares and sparse estimates of the channel taps.
# ──────────────────────────────────────────────────────────────────────

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


def ls_estimate(rx, tx_pilots, n_taps):
    """Least-squares estimate of an ``n_taps`` channel from known pilots.

    Solves ``rx ≈ A h`` where ``A`` is the pilot convolution matrix. Returns the
    complex tap vector ``h``.
    """
    r = np.asarray(rx, dtype=complex).ravel()
    n = min(r.size, np.asarray(tx_pilots).size)
    A = _conv_matrix(tx_pilots, n_taps, n)
    h, *_ = np.linalg.lstsq(A, r[:n], rcond=None)
    return h


def omp_estimate(rx, tx_pilots, n_taps, sparsity):
    """Sparse channel estimate via orthogonal matching pursuit.

    Recovers at most ``sparsity`` non-zero taps out of ``n_taps`` candidate
    delays — the right model for the sparse UW multipath channel. Returns the
    full ``n_taps`` complex vector (zeros off the support).
    """
    if sparsity < 1 or sparsity > n_taps:
        raise ConfigurationError(
            f"omp_estimate: need 1 <= sparsity <= n_taps; got "
            f"sparsity={sparsity}, n_taps={n_taps}")
    r = np.asarray(rx, dtype=complex).ravel()
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
    # Same idiom as ``_ZF_REL_FLOOR`` / ``_PILOT_REL_FLOOR`` in comms.modulate.
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


# ──────────────────────────────────────────────────────────────────────
# Shared regulariser
#
# The ridge term both equalisers use.
# ──────────────────────────────────────────────────────────────────────

# Zero-forcing floor, as a fraction of the channel's own peak power |H|^2.
# Relative because |H|^2 carries whatever amplitude scale the caller's channel
# is in: an absolute floor silently makes the result a function of those units
# — a channel holding 1e-5 of propagation gain equalised to EVM 0.014, and
# 1e-6 to 0.54. `system_id._etfe_divide` states the same rule for a transfer
# function, and `ofdm._PILOT_REL_FLOOR` for a pilot magnitude.
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


# ──────────────────────────────────────────────────────────────────────
# Equalisation
#
# Undoing the delay spread, linear and decision-feedback.
# ──────────────────────────────────────────────────────────────────────

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
    :func:`uacpy.comms.modulate.ofdm_demodulate`, and the same number can be
    calibrated once and passed to either. The Wiener regularizer is a
    noise-to-signal power ratio and so must be expressed in the units of
    ``|H|^2``: writing it as a bare ``1/snr`` assumed a channel of unit mean
    power, and equalizing the same link with ``h`` scaled by a propagation
    gain then changed the damping instead of leaving it alone.
    ``snr_linear -> inf`` gives the zero-forcing inverse.

    The FFT makes the equalization **circular**: ``rx`` must carry a cyclic
    prefix of at least ``len(h)-1`` samples, or the first ``len(h)-1`` outputs
    (which wrap the linear-convolution tail) must be discarded. For the
    CP-based path see :func:`uacpy.comms.modulate.ofdm_demodulate`.

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

    Alignment: the taps start as a centre spike at index ``n_taps // 2``, so
    the output at step ``k`` estimates the symbol received ``n_taps // 2``
    samples earlier. ``train`` must therefore be the transmitted symbols
    delayed by ``n_taps // 2`` (zero-padded in front), and ``eq_symbols``
    lags ``rx`` by the same amount: ``eq_symbols[n_taps // 2:]`` lines up
    with the transmitted sequence. :func:`~uacpy.comms.link.simulate_link`
    applies both. An undelayed ``train`` sets LMS chasing a target one
    spike-width away (measured with LMS on 16-QAM: 1e-2 BER on a noiseless
    identity channel; QPSK hides it because its decisions ignore amplitude).
    """
    return _dfe_core(rx, constellation, n_taps, 0, step, None, 0.0, train)


def rls_equalizer(rx, constellation, n_taps=11, forget=0.99, train=None):
    """Symbol-spaced linear RLS equalizer (faster convergence than LMS).

    Istepanian & Stojanovic put RLS convergence at ~``2N`` symbol intervals
    against LMS's ~``20N``, for ``N`` the total adaptive coefficient count, at
    higher per-symbol cost. Returns ``(eq_symbols, mse)``.

    Same alignment as :func:`lms_equalizer`: delay ``train`` by
    ``n_taps // 2`` and read ``eq_symbols[n_taps // 2:]``. RLS re-learns a
    causal filter within a few symbols on a minimum-phase channel, so an
    undelayed ``train`` is usually recovered, but the delayed form is the
    contract.
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

    @property
    def output_delay(self) -> int:
        """Symbols by which :meth:`equalize`'s output lags ``rx``:
        ``n_ff // 2``, the centre-spike tap index. Delay ``train`` by this
        many symbols and read ``eq_symbols[output_delay:]``."""
        return self.n_ff // 2

    def equalize(self, rx, constellation, train=None):
        """Equalize ``rx`` (symbol-spaced). Returns ``(eq_symbols, mse)``.

        ``train`` must be the transmitted symbols delayed by
        :attr:`output_delay` (zero-padded in front); the output lags ``rx``
        by the same amount, so ``eq_symbols[output_delay:]`` lines up with
        the transmitted sequence (see :func:`lms_equalizer`)."""
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
    # Bring the record to unit mean power before adapting: the LMS step
    # bound, RLS's P(0), the centre-spike init and the unit-energy slicer all
    # carry an absolute scale, so unnormalised the answer depended on the
    # caller's units (test_comms_equalizers pins the amplitude ladder). Done here
    # rather than tap-by-tap because the register mixes record-scale samples
    # with constellation-scale decisions; a no-op at unit input power.
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


# ──────────────────────────────────────────────────────────────────────
# Doppler
#
# Estimating and compensating the scale a moving platform imposes.
# ──────────────────────────────────────────────────────────────────────

# Largest |scale| ``compensate_doppler`` accepts. scale is the Doppler factor
# a = v/c: 0.1 corresponds to a 150 m/s platform at c = 1500 m/s, an order of
# magnitude beyond any underwater vehicle (a few m/s gives |a| ~ 1e-3), while
# the (1+a)*N output length keeps resampling memory within 1.1x the input.
_MAX_DOPPLER_SCALE = 0.1


def doppler_from_speed(speed_mps, sound_speed_mps=DEFAULT_SOUND_SPEED):
    """Doppler scale factor ``a = v/c`` (positive when range is closing)."""
    c = float(sound_speed_mps)
    if not (np.isfinite(c) and c > 0):
        raise ConfigurationError(
            f"doppler_from_speed: sound_speed_mps must be > 0 m/s and "
            f"finite (got {sound_speed_mps!r}); a = v/c divides by it.")
    return float(speed_mps) / c


def compensate_doppler(signal, scale):
    """Undo a Doppler dilation: resample ``signal`` to ``(1+scale)*N`` samples.

    ``scale = a = v/c``. A closing geometry (``a > 0``) compresses the received
    waveform; resampling to ``(1+a)*N`` samples stretches it back to the
    transmit time base. Returns the resampled signal of the input's own kind:
    a real record comes back real (float), a complex one complex.
    ``|scale| > 0.1`` raises: no underwater platform reaches a tenth of the
    sound speed, and the output length scales as ``(1+scale)*N``.
    """
    x = np.asarray(signal)
    a = float(scale)
    if not abs(a) <= _MAX_DOPPLER_SCALE:
        raise ConfigurationError(
            f"compensate_doppler: scale = {scale!r}, but scale is the "
            f"Doppler factor a = v/c: |a| > {_MAX_DOPPLER_SCALE:g} means a "
            f"platform faster than {_MAX_DOPPLER_SCALE:g}·c (150 m/s at "
            f"c = 1500), an order of magnitude beyond any underwater "
            f"vehicle (a few m/s gives |a| ~ 1e-3). The output holds "
            f"round((1+scale)·N) samples, so a mis-scaled value (a speed "
            f"in m/s, a frequency shift in Hz) multiplies memory instead "
            f"of compensating Doppler.")
    n_out = int(round(x.size * (1.0 + a)))
    if n_out < 2:
        raise ConfigurationError(
            f"compensate_doppler: resampling {x.size} sample(s) by "
            f"1 + scale = {1.0 + a:g} leaves {n_out} output sample(s), "
            f"fewer than the 2 the resampler needs; the signal is too "
            f"short for this scale.")
    return resample(x, n_out)


def estimate_doppler_scale(rx, template, scales=None):
    """Estimate the Doppler scale ``a = v/c`` that distorts ``rx``.

    For each candidate ``a`` the receive signal is compensated by that same ``a``
    (i.e. ``compensate_doppler(rx, a)``) and scored against ``template`` with the
    energy-normalized matched-filter metric (:func:`sync.matched_filter_metric`,
    a value in ``[0, 1]``); the best-scoring scale wins. Returns ``(best_scale,
    scales, peak_metric)`` — the last two for plotting the ambiguity curve.

    ``best_scale`` is the CENTRE of the winning plateau, not the candidate
    ``argmax`` picks out. The metric is a staircase (see below), so a whole run
    of candidates shares the top score and ``argmax`` would return that run's
    low edge — a one-sided bias of up to one plateau width. It is therefore a
    midpoint of two scanned candidates rather than one of them verbatim, and
    for a run of even length it falls between grid nodes. A run truncated by
    the end of the scan gives the midpoint of the scanned part, so the answer
    never leaves the range the caller asked for.

    The returned ``a`` follows the package convention (``doppler_from_speed`` /
    ``compensate_doppler``): ``a = v/c``, positive for a closing geometry. It is
    the value to feed straight back: ``compensate_doppler(rx, a)`` removes the
    Doppler. (A closing geometry compresses ``rx``; the best compensation
    stretches it back, so the estimate is ``+v/c``.)

    Compensating ``rx`` (rather than dilating the template) and using the doubly
    normalized metric keeps the score comparable across candidates — a raw,
    template-length-dependent inner product otherwise rails to a scan edge on a
    Doppler-free, periodic, or multipath-smeared probe.

    With ``scales=None`` the default +/-5e-3 grid of 601 candidates (step
    1.67e-5, i.e. 0.025 m/s at c = 1500) is searched in two stages — every
    ``stride``-th candidate first, then the ``2*stride - 1`` grid candidates
    within +/-(``stride`` - 1) steps of the coarse peak — so only the
    evaluated candidates come back in ``scales``/``peak``.

    ``stride`` is set from the record length. The metric is a plateau
    staircase in ``a``, not a smooth curve: ``compensate_doppler`` resamples
    ``rx`` to ``int(round(N*(1 + a)))`` samples, so the score takes one value
    per distinct output length and changes every ``1/N`` in ``a``. A coarse
    stride wider than that quantum puts several plateaus inside one coarse
    interval, the surface stops being single-peaked *at the coarse sampling*,
    and the fine window can close on the wrong plateau. Sampling at most one
    quantum apart — ``stride <= 1/(N * grid_step)``, capped at 15 — keeps the
    two-stage answer equal to a full scan's; the stride falls to 1, i.e. a
    full scan, for the longest records. An explicit ``scales`` array is
    scanned in full.
    """
    r = np.asarray(rx)
    t = np.asarray(template)
    if t.size == 0:
        raise ConfigurationError(
            "estimate_doppler_scale: empty template; the matched-filter "
            "metric needs the transmitted probe to score against.")
    if scales is not None:
        scales = np.asarray(scales, dtype=float)
        peak = _metric_scan(r, t, scales)
        _reject_all_zero_metric(peak, r, t, scales)
        return _plateau_centre(scales, peak), scales, peak

    grid = np.linspace(-5e-3, 5e-3, 601)
    stride = _coarse_stride(r.size, float(grid[1] - grid[0]))
    # The last grid index joins the coarse pass: ``arange`` stops short of it
    # whenever the stride does not divide the span, leaving the truncated
    # plateau at the top edge of the scan unsampled — and that plateau can
    # hold the global maximum.
    coarse_idx = np.unique(np.append(np.arange(0, grid.size, stride),
                                     grid.size - 1))
    coarse_peak = _metric_scan(r, t, grid[coarse_idx])
    if not np.any(coarse_peak > 0):
        # A metric confined to a sliver narrower than the coarse spacing is
        # the one case the coarse pass can miss; scan the full grid before
        # concluding the surface is all-zero.
        peak = _metric_scan(r, t, grid)
        _reject_all_zero_metric(peak, r, t, grid)
        return _plateau_centre(grid, peak), grid, peak
    centre = int(coarse_idx[int(np.argmax(coarse_peak))])
    fine_idx = np.arange(max(0, centre - (stride - 1)),
                         min(grid.size, centre + stride))
    fine_peak = _metric_scan(r, t, grid[fine_idx])
    # Merge the two stages onto one ascending candidate axis (the centre
    # candidate appears in both; keep one copy) so the winning plateau's run
    # is contiguous and its midpoint is the one a full ascending scan finds.
    idx = np.concatenate([coarse_idx, fine_idx])
    vals = np.concatenate([coarse_peak, fine_peak])
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], vals[order]
    keep = np.ones(idx.size, dtype=bool)
    keep[1:] = idx[1:] != idx[:-1]
    idx, vals = idx[keep], vals[keep]
    scales_out = grid[idx]
    return _plateau_centre(scales_out, vals), scales_out, vals


_MAX_COARSE_STRIDE = 15


def _coarse_stride(n_samples: int, grid_step: float) -> int:
    """Coarse-pass stride that samples every resample-length plateau.

    ``compensate_doppler`` resamples an ``n_samples`` record to
    ``int(round(n*(1 + a)))``, so the matched-filter metric is constant over
    each interval of width ``1/n`` in ``a`` and steps between them. The coarse
    pass must land at least once in every such interval — ``stride *
    grid_step <= 1/n`` — or a plateau higher than the coarse peak's neighbours
    can sit unsampled inside a coarse gap and the fine window closes on the
    wrong one.

    The cap is ``_MAX_COARSE_STRIDE``: the bound is slack for records shorter
    than ``1/(_MAX_COARSE_STRIDE * grid_step)`` (4000 samples on the default
    grid), where the widest stride is the cheapest correct one. Above that the
    stride shrinks with ``n`` and reaches 1 — the coarse pass is then the full
    grid — for records past ``1/grid_step`` (60000 samples).
    """
    return int(min(float(_MAX_COARSE_STRIDE),
                   max(1.0, np.floor(1.0 / (max(int(n_samples), 1)
                                            * grid_step)))))


def _plateau_centre(scales, peak):
    """Centre of the contiguous run of maximal candidates around the argmax.

    The metric is a plateau staircase, not a peaked curve: ``compensate_doppler``
    resamples to ``int(round(N*(1 + a)))`` samples, so every candidate inside a
    ``1/N``-wide interval of ``a`` produces the *same* resampled record and
    therefore a bit-identical score. ``np.argmax`` returns the first index of
    that run, i.e. the plateau's LOW EDGE, which is a one-sided bias of up to
    one plateau width — measured 7/7 on the package's chirp fixture, where a
    ``1/N`` of 1.72e-4 in ``a`` is 0.26 m/s at c = 1500. Returning the run's
    midpoint puts the estimate where the metric actually stops distinguishing.

    A run truncated by the end of the scan (the winning plateau straddling the
    top or bottom candidate) gives the midpoint of the part that was scanned:
    that is the best available, and unlike the analytic plateau centre
    ``n_out/N - 1`` it never returns a scale outside the range the caller
    asked for.
    """
    best_index = int(np.argmax(peak))
    top = peak[best_index]
    lo = best_index
    while lo - 1 >= 0 and peak[lo - 1] == top:
        lo -= 1
    hi = best_index
    while hi + 1 < peak.size and peak[hi + 1] == top:
        hi += 1
    return float(0.5 * (scales[lo] + scales[hi]))


def _metric_scan(r, t, scales):
    """Peak matched-filter metric of ``compensate_doppler(r, a)`` against ``t``
    for each candidate ``a``; candidates the record cannot support score 0."""
    from uacpy.comms.receive import matched_filter_metric

    peak = np.zeros(scales.size)
    for i, a in enumerate(scales):
        try:
            comp = compensate_doppler(r, float(a))
        except ConfigurationError:
            continue
        if t.size > comp.size:
            continue
        peak[i] = float(matched_filter_metric(comp, t).max())
    return peak


def _reject_all_zero_metric(peak, r, t, scales):
    """Raise when no candidate produced a metric — the argmax of an all-zero
    surface would return the scan edge as a confident estimate."""
    if not np.any(peak > 0):
        raise ConfigurationError(
            f"estimate_doppler_scale: no scale candidate produced a metric "
            f"(template of {t.size} samples vs record of {r.size}) — the "
            f"argmax of an all-zero surface would return the scan edge "
            f"{float(scales[0]):g} as a confident estimate.")


# ──────────────────────────────────────────────────────────────────────
# Link metrics
#
# What came out: bit error rate, EVM.
# ──────────────────────────────────────────────────────────────────────

def _q(x):
    """Gaussian Q-function ``Q(x) = 0.5*erfc(x/sqrt(2))``."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def bit_error_rate(tx_bits, rx_bits):
    """Fraction of differing bits over the overlap of the two bit streams."""
    a = np.asarray(tx_bits, dtype=int).ravel()
    b = np.asarray(rx_bits, dtype=int).ravel()
    n = min(a.size, b.size)
    if n == 0:
        raise ConfigurationError(
            f"bit_error_rate: empty input — tx_bits carries {a.size} bits, "
            f"rx_bits {b.size}. The BER is taken over the overlap of the two "
            f"streams, so both must be non-empty.")
    return float(np.mean(a[:n] != b[:n]))


def symbol_error_rate(tx_symbols_or_labels, rx_symbols_or_labels):
    """SER over the overlap; accepts integer labels or complex symbols (compared exactly)."""
    a = np.asarray(tx_symbols_or_labels).ravel()
    b = np.asarray(rx_symbols_or_labels).ravel()
    n = min(a.size, b.size)
    if n == 0:
        raise ConfigurationError(
            f"symbol_error_rate: empty input — tx carries {a.size} symbols, "
            f"rx {b.size}. The SER is taken over the overlap of the two "
            f"streams, so both must be non-empty.")
    return float(np.mean(a[:n] != b[:n]))


def evm(rx_symbols, ref_symbols):
    """RMS error-vector magnitude (fraction; multiply by 100 for percent).

    ``sqrt(mean|rx-ref|^2 / mean|ref|^2)`` over the overlap.
    """
    r = np.asarray(rx_symbols, dtype=complex).ravel()
    s = np.asarray(ref_symbols, dtype=complex).ravel()
    n = min(r.size, s.size)
    if n == 0:
        raise ConfigurationError(
            f"evm: empty input — rx_symbols carries {r.size} symbols, "
            f"ref_symbols {s.size}. The EVM is taken over the overlap of the "
            f"two streams, so both must be non-empty.")
    err = np.mean(np.abs(r[:n] - s[:n]) ** 2)
    ref = np.mean(np.abs(s[:n]) ** 2)
    if ref == 0.0:
        # EVM is a ratio to the reference power, so a zero-energy reference
        # leaves it undefined; returning inf behind a numpy divide warning
        # reads as "infinitely bad", not "not defined".
        raise ConfigurationError(
            "evm: reference symbols carry no energy, so the error vector has "
            "nothing to be relative to. Pass the transmitted constellation "
            "symbols as ref_symbols.")
    return float(np.sqrt(err / ref))


_BER_PSK_ORDERS = {"8psk": 8, "16psk": 16}
#: Constellation order per QAM scheme. Declared here, not in
#: ``modulate``, because ``modulate`` already imports from this
#: module and the reverse would be a cycle.
_QAM_ORDERS = {"16qam": 16, "64qam": 64, "256qam": 256}


def ber_theory(scheme, ebn0_dB):
    """Theoretical AWGN BER vs Eb/N0 (dB) for a Gray-mapped scheme.

    Exact for BPSK/QPSK; standard nearest-neighbour approximations for higher
    M-PSK and square M-QAM (Proakis & Salehi):

    * M-PSK symbol error ``P_M = 2 Q(sqrt(2 k Eb/N0) sin(pi/M))``, eq. (4.3-17).
    * square M-QAM ``P_M ~= 4 (1 - 1/sqrt(M)) Q(sqrt(3 k Eb/N0 / (M-1)))``,
      eqs. (4.3-29) into (4.3-27), dropping the second-order term.

    Both are **symbol** error rates; the ``1/k`` factor that converts them to a
    bit error rate is only valid under Gray mapping (eq. 4.3-20): adjacent
    constellation points then differ in a single bit, so the dominant
    nearest-neighbour symbol error costs exactly one of the ``k`` bits.
    :func:`uacpy.comms.constellation` is Gray-mapped throughout.
    """
    ebn0 = 10.0 ** (np.asarray(ebn0_dB, dtype=float) / 10.0)
    s = scheme.lower()
    if s in ("bpsk", "qpsk"):
        return _q(np.sqrt(2.0 * ebn0))
    if s in _BER_PSK_ORDERS:
        M = _BER_PSK_ORDERS[s]
        k = np.log2(M)
        return (2.0 / k) * _q(np.sqrt(2.0 * k * ebn0) * np.sin(np.pi / M))
    if s in _QAM_ORDERS:
        M = _QAM_ORDERS[s]
        k = np.log2(M)
        c = 4.0 / k * (1.0 - 1.0 / np.sqrt(M))
        return c * _q(np.sqrt(3.0 * k / (M - 1.0) * ebn0))
    valid = ("bpsk", "qpsk", *_BER_PSK_ORDERS, *_QAM_ORDERS)
    raise ConfigurationError(
        f"ber_theory: unsupported scheme {scheme!r}; valid: {', '.join(valid)}")
