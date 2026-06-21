"""Decidecade (base-10 one-third-octave) frequency bands and band levels.

Decidecade bands are the ISO 18405 / IEC 61260-1 standard for reporting
underwater sound (ship radiated noise, soundscapes): a band ratio of
``10^(1/10)``, indexed from the 1 kHz reference. A decidecade is one tenth of a
decade (0.3322 octave) — the *base-10* one-third-octave band; it is analogous
to, but not identical with, the base-2 third-octave (0.3333 octave) bands in
:mod:`uacpy.acoustic_signal.analysis`.

References
----------
IEC 61260-1:2014, *Electroacoustics — Octave-band and fractional-octave-band
    filters — Part 1: Specifications* (base-10 system, ``G = 10^(3/10)``).
ISO 18405:2017, *Underwater acoustics — Terminology* (decidecade = 1/10 decade =
    0.332 octave; band levels).
"""

from __future__ import annotations

import warnings

import numpy as np

from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.core.exceptions import ConfigurationError

_REF_FREQ = 1000.0      # reference frequency [Hz]


def decidecade_bands(f_low, f_high):
    """Decidecade band ``(lower, center, upper)`` edges spanning ``[f_low, f_high]``.

    Centre frequencies are ``1000 * 10^(n/10)`` (IEC 61260-1 base-10); band edges
    are ``center * 10^(±1/20)``. Returns three arrays of equal length covering
    every band that overlaps the requested range.
    """
    if f_low <= 0 or f_high <= f_low:
        raise ConfigurationError("decidecade_bands: need 0 < f_low < f_high")
    n_lo = int(np.floor(10.0 * np.log10(f_low / _REF_FREQ)))
    n_hi = int(np.ceil(10.0 * np.log10(f_high / _REF_FREQ)))
    centers = _REF_FREQ * 10.0 ** (np.arange(n_lo, n_hi + 1) / 10.0)
    lower = centers * 10.0 ** (-1.0 / 20.0)
    upper = centers * 10.0 ** (1.0 / 20.0)
    keep = (upper >= f_low) & (lower <= f_high)
    return lower[keep], centers[keep], upper[keep]


def decidecade_band_levels(psd, freqs, ref=REFERENCE_PRESSURE_WATER):
    """Integrate a one-sided PSD into decidecade band levels.

    Parameters
    ----------
    psd : array_like
        One-sided power spectral density [pressure²/Hz, e.g. Pa²/Hz].
    freqs : array_like
        Frequencies [Hz] matching ``psd`` (monotonic, > 0).
    ref : float
        Reference pressure (default ``1e-6`` Pa = 1 µPa, the water standard).

    Returns
    -------
    centers, levels : numpy.ndarray
        Band centre frequencies [Hz] and band levels [dB re ``ref²``]; bands with
        no spectral support are ``nan``.

    Notes
    -----
    A band straddling fewer than two PSD grid points is integrated by a
    rectangular ``psd · bandwidth`` estimate (``np.trapezoid`` over a single
    point is 0, which would silently ``NaN`` the band on a grid too coarse to
    resolve it) and a one-time :class:`UserWarning` names how many bands were
    under-resolved, so grid coarseness is visible rather than read as "no
    energy".
    """
    psd = np.asarray(psd, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    pos = freqs > 0
    lower, centers, upper = decidecade_bands(freqs[pos].min(), freqs[pos].max())
    levels = np.full(centers.size, np.nan)
    n_coarse = 0
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        m = (freqs >= lo) & (freqs < hi)
        n = int(np.count_nonzero(m))
        if n >= 2:
            power = np.trapezoid(psd[m], freqs[m])
        elif n == 1:
            # Coarse grid: trapezoid over one point is 0; fall back to a
            # rectangular psd·bandwidth estimate so the band isn't lost.
            n_coarse += 1
            power = float(psd[m][0]) * (hi - lo)
        else:
            continue
        if power > 0:
            levels[i] = 10.0 * np.log10(power / ref ** 2)
    if n_coarse:
        warnings.warn(
            f"decidecade_band_levels: {n_coarse} band(s) straddled only one PSD "
            "grid point and were estimated rectangularly (psd·bandwidth); the "
            "PSD grid is too coarse to resolve them. Use a finer-resolution PSD "
            "for an integrated level.",
            UserWarning, stacklevel=2,
        )
    return centers, levels


def plot_band_levels(centers, levels, ax=None, title="", width=0.8,
                     ref_label="1 µPa²", **kwargs):
    """Bar plot of decidecade band levels vs centre frequency. Returns ``(fig, ax)``.

    ``ref_label`` is the reference shown in the y-axis label; pass the reference
    matching the ``ref`` used in :func:`decidecade_band_levels` (e.g. ``"20 µPa²"``
    in air) so the units stay honest.
    """
    import matplotlib.pyplot as plt
    c = np.asarray(centers, dtype=float)
    lv = np.asarray(levels, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure
    x = np.log10(c)
    bw = width * np.median(np.diff(x)) if c.size > 1 else 0.04
    ax.bar(x, lv, width=bw, **kwargs)
    ticks = x[:: max(1, c.size // 12)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.0f}" for v in 10 ** ticks], rotation=45)
    ax.set_xlabel("Decidecade band centre [Hz]")
    ax.set_ylabel(f"Band level [dB re {ref_label}]")
    ax.set_title(f"[decidecade] band levels {title}", loc="left")
    ax.grid(alpha=0.3, axis="y")
    return fig, ax
