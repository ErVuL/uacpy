"""Finding the transmission and its timing: the matched-filter preamble
metric and detectors, the Doppler scale a moving platform imposes and its
compensation, and symbol-timing recovery."""

from __future__ import annotations

from scipy.signal import resample
import numpy as np
import scipy.signal as _sig

from uacpy.comms.phy import _require_integer_sps
from uacpy.core._validate import require_positive_finite_scalar
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError


#: The default Doppler time-scale search: ``DOPPLER_SCALE_COUNT``
#: candidates over ``±DOPPLER_SCALE_MAX``.
DOPPLER_SCALE_MAX = 5e-3
DOPPLER_SCALE_COUNT = 601


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
            f"symbol_sync: need sps >= 2 for Gardner; got {sps!r}.")
    # theta below is loop_bw / (damping + 1/(4*damping)): zero damping is a
    # bare ZeroDivisionError, and a non-positive loop bandwidth leaves the
    # loop gains at zero so the interpolator never steers.
    require_positive_finite_scalar(
        damping, "symbol_sync", "damping",
        why=" ~1.0 is critically damped; the loop constant divides by "
            "damping + 1/(4*damping).")
    require_positive_finite_scalar(
        loop_bw, "symbol_sync", "loop_bw",
        why=" At zero the proportional and integral gains are zero and the "
            "timing estimate never moves off `start`.")
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


def matched_filter_metric(received, preamble):
    """Energy-normalized cross-correlation magnitude of ``received`` with ``preamble``.

    Returns a real metric in ``[0, 1]`` (1 = perfect match) aligned so that
    ``metric[k]`` scores the window ``received[k:k+len(preamble)]``.

    Parameters
    ----------
    received : array_like
        The record to search.
    preamble : array_like
        The known preamble, no longer than ``received``.
    """
    r = np.asarray(received, dtype=complex)
    p = np.asarray(preamble, dtype=complex)
    if p.size > r.size:
        raise ConfigurationError(
            f"matched_filter_metric: preamble longer than signal — preamble "
            f"is {p.size} samples, received {r.size}.")
    return _matched_filter_core(r, p, float(np.sum(np.abs(p) ** 2)))


#: Largest rounding error, relative to a window's own energy, that
#: ``matched_filter_metric`` accepts from the whole-record running sums;
#: a window where the bound is larger is recomputed from its own stretch.
_MF_REL_PRECISION = 1e-8


def _matched_filter_core(r, p, pe):
    """``matched_filter_metric`` of a validated complex record ``r``.

    The correlation is an FFT correlation, O(N log N) where np.correlate
    is O(N*M): 0.08 s against 0.008 s for a 2000-sample preamble in a
    100 000-sample record, and estimate_doppler_scale calls this once per
    candidate. It is FFT whatever the sizes: scipy's cost model picks the
    direct sum for a preamble nearly as long as the record, which took
    4.6 s per candidate for a 1 s probe at 12 kHz. The sliding window
    energy is a difference of running sums, O(N).

    Both carry a rounding error set by the record's WHOLE energy — about
    N*eps of it for the running sum — so on their own they resolve a window
    far quieter than the loudest part of the record only coarsely (from the
    whole-record sums, a preamble 100 dB below a burst reads 0.999999 and
    one 110 dB below reads 0). A window whose energy is not at least
    ``1/_MF_REL_PRECISION`` times that bound is therefore recomputed from
    its own stretch of samples by the same two sums, whose error is then set
    by that stretch's energy; a stretch holding no energy at all scores 0.
    Every floor is relative to the energy it is compared with, so the metric
    does not depend on the units the record is held in; an absolute epsilon
    would make a Pa-scale record score 3e-5 where the same signal in µPa
    scores 1. Above the floor Cauchy-Schwarz bounds the ratio by 1;
    rounding can put a perfect match a few ulps above it, which the minimum
    removes.
    """
    m = p.size
    corr = _sig.correlate(r, p, mode="valid", method="fft")
    csum = np.concatenate(([0.0], np.cumsum(np.abs(r) ** 2)))
    win = csum[m:] - csum[:-m]
    floor = (4.0 * np.finfo(float).eps * r.size * csum[-1]
             / _MF_REL_PRECISION)
    denom = pe * win
    ok = (win > floor) & (denom > 0.0)
    metric = np.where(ok, np.minimum(
        np.abs(corr) / np.sqrt(np.where(ok, denom, 1.0)), 1.0), 0.0)
    # Runs of unresolved windows [a, b): their samples r[a:b+m-1] are all
    # quieter than the floor, so recomputing on that stretch alone resolves
    # them to the stretch's own precision. A stretch as long as the record
    # (nothing louder to separate from) or with no energy stays at 0.
    low = ~ok
    if np.any(low):
        edges = np.flatnonzero(np.diff(np.concatenate(([0], low.astype(int),
                                                        [0]))))
        for a, b in zip(edges[::2], edges[1::2]):
            span = r[a:b + m - 1]
            if span.size < r.size and np.any(span):
                metric[a:b] = _matched_filter_core(span, p, pe)
    return metric


def detect_preamble(received, preamble, threshold=0.5):
    """Index of the best preamble alignment if its metric exceeds ``threshold``.

    Returns ``(start_index, metric_array)``; ``start_index`` is ``None`` when no
    peak clears the threshold.

    Parameters
    ----------
    received : array_like
        The record to search.
    preamble : array_like
        The known preamble.
    threshold : float, optional
        Lowest metric (in ``[0, 1]``) taken as a detection. Default 0.5.
    """
    metric = matched_filter_metric(received, preamble)
    k = int(np.argmax(metric))
    return (k if metric[k] >= threshold else None), metric


def detect_frames(received, preamble, threshold=0.5, min_gap=None):
    """All frame-start indices whose metric clears ``threshold`` (peak-picked).

    ``min_gap`` (samples) suppresses detections closer than that to a stronger
    one; defaults to the preamble length. Returns ``(starts, metric_array)``.

    Parameters
    ----------
    received : array_like
        The record to search.
    preamble : array_like
        The known preamble.
    threshold : float, optional
        Lowest metric (in ``[0, 1]``) taken as a detection. Default 0.5.
    min_gap : int, optional
        Samples within which a weaker detection is suppressed; ``None`` is
        the preamble length.
    """
    metric = matched_filter_metric(received, preamble)
    gap = int(min_gap) if min_gap is not None else np.asarray(preamble).size
    cand = np.flatnonzero(metric >= threshold)
    starts = []
    for k in cand[np.argsort(metric[cand])[::-1]]:
        if all(abs(k - s) >= gap for s in starts):
            starts.append(int(k))
    return sorted(starts), metric


# Largest |scale| ``compensate_doppler`` accepts. scale is the Doppler factor
# a = v/c: 0.1 corresponds to a 150 m/s platform at c = 1500 m/s, an order of
# magnitude beyond any underwater vehicle (a few m/s gives |a| ~ 1e-3), while
# the (1+a)*N output length keeps resampling memory within 1.1x the input.
_MAX_DOPPLER_SCALE = 0.1


def doppler_from_speed(speed_mps, sound_speed=DEFAULT_SOUND_SPEED):
    """Doppler scale factor ``a = v/c`` (positive when range is closing).

    ``speed_mps`` may be a scalar (returns a float) or an array (returns an
    array of the same shape); ``sound_speed`` is one scalar.

    This is the dimensionless scale a receiver compensates
    (:func:`compensate_doppler`, :func:`estimate_doppler_scale`). The
    Doppler-shifted *frequency* a tone of frequency ``f`` arrives at is
    :func:`uacpy.acoustics.doppler`.

    Parameters
    ----------
    speed_mps : float or array_like
        Radial speed (m/s), positive when the range is closing.
    sound_speed : float, optional
        Sound speed (m/s). Default
        :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
    """
    c = require_positive_finite_scalar(
        sound_speed, "doppler_from_speed", "sound_speed", " m/s",
        why=" a = v/c divides by it.")
    v = np.asarray(speed_mps, dtype=float)
    return float(v) / c if v.ndim == 0 else v / c


def compensate_doppler(signal, scale):
    """Undo a Doppler dilation: resample ``signal`` to ``(1+scale)*N`` samples.

    ``scale = a = v/c``. A closing geometry (``a > 0``) compresses the received
    waveform; resampling to ``(1+a)*N`` samples stretches it back to the
    transmit time base. Returns the resampled signal of the input's own kind:
    a real record comes back real (float), a complex one complex.

    Apply it to the passband (or analytic) record, where the dilation acts
    on the carrier too. After down-conversion the carrier has been removed at
    its transmit frequency, so resampling baseband restores the time base but
    leaves a carrier offset of ``scale * fc`` to remove separately.
    ``|scale| > 0.1`` raises: no underwater platform reaches a tenth of the
    sound speed, and the output length scales as ``(1+scale)*N``.

    Parameters
    ----------
    signal : array_like
        The received record, real or complex.
    scale : float
        The Doppler factor ``a = v/c``, ``|a| <= 0.1``.
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


def estimate_doppler_scale(received, template, scales=None):
    """Estimate the Doppler scale ``a = v/c`` that distorts ``received``.

    For each candidate ``a`` the receive signal is compensated by that same ``a``
    (i.e. ``compensate_doppler(received, a)``) and scored against ``template`` with the
    energy-normalized matched-filter metric (:func:`matched_filter_metric`,
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
    the value to feed straight back: ``compensate_doppler(received, a)`` removes the
    Doppler. (A closing geometry compresses ``received``; the best compensation
    stretches it back, so the estimate is ``+v/c``.)

    Compensating ``received`` (rather than dilating the template) and using the doubly
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
    ``received`` to ``int(round(N*(1 + a)))`` samples, so the score takes one value
    per distinct output length and changes every ``1/N`` in ``a``. A coarse
    stride wider than that quantum puts several plateaus inside one coarse
    interval, the surface stops being single-peaked *at the coarse sampling*,
    and the fine window can close on the wrong plateau. Sampling at most one
    quantum apart — ``stride <= 1/(N * grid_step)``, capped at 15 — keeps the
    two-stage answer equal to a full scan's; the stride falls to 1, i.e. a
    full scan, for the longest records. An explicit ``scales`` array is
    scanned in full.

    Parameters
    ----------
    received : array_like
        The received record.
    template : array_like
        The known transmitted waveform.
    scales : array_like, optional
        Candidate scales; ``None`` is the default grid described below.
    """
    r = np.asarray(received)
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

    grid = np.linspace(-DOPPLER_SCALE_MAX, DOPPLER_SCALE_MAX,
                       DOPPLER_SCALE_COUNT)
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
