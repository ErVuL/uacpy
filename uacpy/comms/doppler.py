"""Doppler estimation and compensation for moving underwater platforms.

Platform motion at speed ``v`` scales the received waveform in time by the
Doppler factor ``a = v/c`` (``c`` the sound speed) — a *dilation/compression* of
the whole passband, not just a carrier shift, because the UW fractional
bandwidth is large. Compensation resamples the signal back to the transmit time
base (Sharif/Stojanovic resampling receiver).

References
----------
Sharif, Neasham, Hinton & Adams (2000), *A computationally efficient Doppler
    compensation system for underwater acoustic communications*, IEEE JOE.
Stojanovic, in Istepanian & Stojanovic. *Underwater Acoustic DSP & Comms*.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError

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
    transmit time base. Returns the resampled (complex) signal.
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
        best = float(scales[int(np.argmax(peak))])
        return best, scales, peak

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
        best = float(grid[int(np.argmax(peak))])
        return best, grid, peak
    centre = int(coarse_idx[int(np.argmax(coarse_peak))])
    fine_idx = np.arange(max(0, centre - (stride - 1)),
                         min(grid.size, centre + stride))
    fine_peak = _metric_scan(r, t, grid[fine_idx])
    # Merge the two stages onto one ascending candidate axis (the centre
    # candidate appears in both; keep one copy) so the argmax resolves ties
    # toward the lowest scale, as a full ascending scan does.
    idx = np.concatenate([coarse_idx, fine_idx])
    vals = np.concatenate([coarse_peak, fine_peak])
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], vals[order]
    keep = np.ones(idx.size, dtype=bool)
    keep[1:] = idx[1:] != idx[:-1]
    idx, vals = idx[keep], vals[keep]
    scales_out = grid[idx]
    best = float(scales_out[int(np.argmax(vals))])
    return best, scales_out, vals


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


def _metric_scan(r, t, scales):
    """Peak matched-filter metric of ``compensate_doppler(r, a)`` against ``t``
    for each candidate ``a``; candidates the record cannot support score 0."""
    from uacpy.comms.sync import matched_filter_metric

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
