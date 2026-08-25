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
from uacpy.acoustic_signal._signal_validate import (
    require_increasing_axis, require_positive_finite_scalar)

_REF_FREQ = 1000.0      # reference frequency [Hz]


def decidecade_bands(f_low, f_high):
    """Decidecade band ``(lower, center, upper)`` edges spanning ``[f_low, f_high]``.

    Centre frequencies are ``1000 * 10^(n/10)`` (IEC 61260-1 base-10); band edges
    are ``center * 10^(±1/20)``. Returns three arrays of equal length covering
    every band that overlaps the requested range.
    """
    if f_low <= 0 or f_high <= f_low:
        raise ConfigurationError(
            "decidecade_bands: need 0 < f_low < f_high; "
            f"got f_low={f_low!r}, f_high={f_high!r}")
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
    the ends of ``frequencies`` is returned as ``nan`` — a partial integral is
    not a band level.

    **The first and last band are normally ``nan``, and that is structural.**
    The band set comes from :func:`decidecade_bands`, which keeps every band
    *overlapping* ``[min(frequencies), max(frequencies)]``, so the band holding
    the first frequency starts below it and the band holding the last ends
    above it unless both land exactly on decidecade band edges — which no
    ``rfftfreq`` grid does. Those two ``nan`` levels are the diagnostic; they
    are not warned about, because a warning that fires on every well-formed
    call cannot distinguish "the grid is too short" from "the function was
    called". The returned arrays stay parallel to :func:`decidecade_bands` on
    the same span, so a caller masks with ``np.isfinite(levels)``.

    A band holding fewer than two interior grid points rests almost entirely on
    its interpolated edges; a :class:`UserWarning` names how many such bands
    the grid produced. That one *is* a warning: it qualifies levels that came
    back finite.
    """
    psd = np.asarray(psd, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    ref = require_positive_finite_scalar(
        ref, "decidecade_band_levels", "ref", " Pa")
    if np.any(psd < 0):
        raise ConfigurationError(
            "decidecade_band_levels: psd contains negative values; a power "
            "spectral density is non-negative, so this input is a dB level "
            "or a signed spectrum, whose band integral is not a band level. "
            f"Got {int(np.count_nonzero(psd < 0))} negative value(s), "
            f"minimum {psd.min():g}.")
    if frequencies.shape != psd.shape:
        raise ConfigurationError(
            f"decidecade_band_levels: psd shape {psd.shape} and frequencies "
            f"shape {frequencies.shape} differ")
    if frequencies.size > 1 and np.any(np.diff(frequencies) <= 0):
        raise ConfigurationError(
            "decidecade_band_levels: frequencies must be strictly increasing. "
            "A two-sided np.fft.fftfreq grid is not — take the one-sided "
            "np.fft.rfftfreq half (and the matching half of the PSD). Got "
            f"{int(np.count_nonzero(np.diff(frequencies) <= 0))} "
            f"non-increasing step(s), first at index "
            f"{int(np.argmax(np.diff(frequencies) <= 0))}.")
    require_increasing_axis(frequencies, "decidecade_band_levels: frequencies")
    if frequencies.size < 2:
        # A one-point axis has f_min == f_max, which reached decidecade_bands
        # as "need 0 < f_low < f_high" — an error naming two arguments this
        # caller never passed.
        raise ConfigurationError(
            f"decidecade_band_levels: frequencies needs at least 2 samples to "
            f"span a band; got {frequencies.size}. A single point has no "
            f"width, so no decidecade band covers it.")
    pos = frequencies > 0
    f_min = frequencies[pos].min()
    f_max = frequencies[pos].max()
    lower, centers, upper = decidecade_bands(f_min, f_max)
    levels = np.full(centers.size, np.nan)
    n_coarse = 0
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        # Integrate over [lo, hi] itself: splice the edges into the in-band
        # grid points and interpolate the PSD onto them.
        #
        # A band the supplied grid does not fully cover is left ``nan``.
        # Clipping the nodes to the support instead returned the integral over
        # the covered part, which is not that band's level — measured 3.8 dB
        # (first band) and 3.2 dB (last) off their own trend on a flat PSD,
        # and 5.5 dB low on the realistic psd -> band_levels path.
        if lo < f_min * (1.0 - 1e-12) or hi > f_max * (1.0 + 1e-12):
            continue
        interior = frequencies[(frequencies > lo) & (frequencies < hi)]
        # `nodes` always holds at least the two spliced edges: hi/lo is the
        # constant 10**0.1 for every band, so lo < hi strictly at every
        # positive centre and np.unique keeps both.
        nodes = np.unique(np.concatenate(([lo], interior, [hi])))
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
