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


def decidecade_band_levels(psd, frequencies, ref=REFERENCE_PRESSURE_WATER):
    """Integrate a one-sided PSD into decidecade band levels.

    Parameters
    ----------
    psd : array_like
        One-sided power spectral density [pressure²/Hz, e.g. Pa²/Hz].
    frequencies : array_like
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
    Each band is integrated over its full support ``[lo, hi]``: the band edges
    are spliced into the in-band grid points and the PSD is interpolated onto
    them, so the edge intervals carry their true width. A band reaching past
    the ends of ``frequencies`` is integrated over the covered part only, never
    extrapolated, so its level reflects the data actually supplied.

    A band holding fewer than two interior grid points rests almost entirely on
    its interpolated edges; a one-time :class:`UserWarning` names how many such
    bands the grid produced.
    """
    psd = np.asarray(psd, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.shape != psd.shape:
        raise ConfigurationError(
            f"decidecade_band_levels: psd shape {psd.shape} and frequencies "
            f"shape {frequencies.shape} differ")
    if frequencies.size > 1 and np.any(np.diff(frequencies) <= 0):
        raise ConfigurationError(
            "decidecade_band_levels: frequencies must be strictly increasing. "
            "A two-sided np.fft.fftfreq grid is not — take the one-sided "
            "np.fft.rfftfreq half (and the matching half of the PSD).")
    pos = frequencies > 0
    f_min = frequencies[pos].min()
    f_max = frequencies[pos].max()
    lower, centers, upper = decidecade_bands(f_min, f_max)
    levels = np.full(centers.size, np.nan)
    n_coarse = 0
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        # Integrate over [lo, hi] itself: splice the edges into the in-band
        # grid points and interpolate the PSD onto them. Clip to the supplied
        # frequency support so a partly-covered edge band is not extrapolated.
        interior = frequencies[(frequencies > lo) & (frequencies < hi)]
        nodes = np.unique(np.concatenate(([lo], interior, [hi])))
        nodes = nodes[(nodes >= f_min) & (nodes <= f_max)]
        if nodes.size < 2:
            continue
        if interior.size < 2:
            n_coarse += 1
        power = np.trapezoid(np.interp(nodes, frequencies, psd), nodes)
        if power > 0:
            levels[i] = 10.0 * np.log10(power / ref ** 2)
    if n_coarse:
        warnings.warn(
            f"decidecade_band_levels: {n_coarse} band(s) hold fewer than two "
            "interior PSD grid points and rest almost entirely on interpolated "
            "band edges; the PSD grid is too coarse to resolve them. Use a "
            "finer-resolution PSD for a fully integrated level.",
            UserWarning, stacklevel=2,
        )
    return centers, levels
