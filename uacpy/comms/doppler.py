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

from uacpy.core.exceptions import ConfigurationError


def doppler_from_speed(speed_mps, sound_speed_mps=1500.0):
    """Doppler scale factor ``a = v/c`` (positive when range is closing)."""
    return float(speed_mps) / float(sound_speed_mps)


def compensate_doppler(signal, scale):
    """Undo a Doppler dilation: resample ``signal`` by ``1/(1+scale)``.

    ``scale = a = v/c``. A closing geometry (``a > 0``) compresses the received
    waveform; this stretches it back to the transmit time base. Returns the
    resampled (complex) signal.
    """
    x = np.asarray(signal)
    a = float(scale)
    n_out = int(round(x.size * (1.0 + a)))
    if n_out < 2:
        raise ConfigurationError("compensate_doppler: scale too large for signal length")
    return resample(x, n_out)


def estimate_doppler_scale(rx, template, scales=None):
    """Estimate the Doppler scale by resampling ``template`` over candidate scales.

    For each candidate ``a`` the template is dilated by ``1+a`` and matched
    against ``rx``; the peak-correlation scale wins. Returns ``(best_scale,
    scales, peak_metric)`` — the last two for plotting the ambiguity curve.
    """
    r = np.asarray(rx)
    t = np.asarray(template)
    if scales is None:
        scales = np.linspace(-5e-3, 5e-3, 51)
    scales = np.asarray(scales, dtype=float)
    peak = np.empty(scales.size)
    for i, a in enumerate(scales):
        n = max(2, int(round(t.size * (1.0 + a))))
        tt = resample(t, n)
        corr = np.abs(np.correlate(r, tt, mode="valid"))
        peak[i] = corr.max() if corr.size else 0.0
    best = float(scales[int(np.argmax(peak))])
    return best, scales, peak


def plot_doppler_ambiguity(scales, peak_metric, ax=None, title="", **kwargs):
    """Plot the Doppler-scale ambiguity curve (peak correlation vs scale).

    Returns ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    s = np.asarray(scales, dtype=float)
    p = np.asarray(peak_metric, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    ax.plot(s * 1e3, p / (p.max() + 1e-12), **kwargs)
    best = s[int(np.argmax(p))] * 1e3
    ax.axvline(best, color="r", ls="--", lw=1, label=f"a = {best:.2f} e-3")
    ax.set_xlabel("Doppler scale a [×10⁻³]"); ax.set_ylabel("Norm. peak correlation")
    ax.set_title(f"[doppler] ambiguity {title}", loc="left")
    ax.grid(alpha=0.3); ax.legend()
    return fig, ax
