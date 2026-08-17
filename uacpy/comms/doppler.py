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


def doppler_from_speed(speed_mps, sound_speed_mps=DEFAULT_SOUND_SPEED):
    """Doppler scale factor ``a = v/c`` (positive when range is closing)."""
    return float(speed_mps) / float(sound_speed_mps)


def compensate_doppler(signal, scale):
    """Undo a Doppler dilation: resample ``signal`` to ``(1+scale)*N`` samples.

    ``scale = a = v/c``. A closing geometry (``a > 0``) compresses the received
    waveform; resampling to ``(1+a)*N`` samples stretches it back to the
    transmit time base. Returns the resampled (complex) signal.
    """
    x = np.asarray(signal)
    a = float(scale)
    n_out = int(round(x.size * (1.0 + a)))
    if n_out < 2:
        raise ConfigurationError("compensate_doppler: scale too large for signal length")
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
    15th candidate first, then the 29 grid candidates within +/-14 steps of
    the coarse peak — so only the evaluated candidates come back in
    ``scales``/``peak``. The metric's peak region spans ~120 grid steps, far
    wider than the 15-step coarse spacing, and for a single-peaked surface
    the global argmax lies strictly between the coarse neighbours of the best
    coarse sample, i.e. inside the +/-14 window — the two stages then return
    the same grid candidate a full scan does. An explicit ``scales`` array is
    scanned in full.
    """
    r = np.asarray(rx)
    t = np.asarray(template)
    if scales is not None:
        scales = np.asarray(scales, dtype=float)
        peak = _metric_scan(r, t, scales)
        _reject_all_zero_metric(peak, r, t, scales)
        best = float(scales[int(np.argmax(peak))])
        return best, scales, peak

    grid = np.linspace(-5e-3, 5e-3, 601)
    coarse_idx = np.arange(0, grid.size, 15)           # 41 candidates
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
    fine_idx = np.arange(max(0, centre - 14),
                         min(grid.size, centre + 15))  # 29 candidates
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
