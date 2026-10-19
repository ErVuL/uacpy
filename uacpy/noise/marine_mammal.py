"""Marine-mammal auditory weighting (Southall et al. 2019) on level spectra.

The weighting itself, ``W(f)`` [dB] for each hearing group, is
:func:`auditory_weighting`; its formula, parameter table and references live
with the other frequency weightings in :mod:`uacpy.core.acoustics.weighting`,
which also filters a pressure record (:func:`~uacpy.acoustics.weighted_pressure`)
and serves every ``weighting=`` argument. This module applies it to a level
spectrum (:func:`apply_weighting`) and integrates a weighted level density
(:func:`weighted_level`).
"""

from __future__ import annotations

import numpy as np

from uacpy.core.acoustics import band_level, no_energy_mask
from uacpy.core.acoustics.weighting import (  # noqa: F401  (re-exported by uacpy.noise)
    EXPOSURE_THRESHOLDS,
    HEARING_GROUPS,
    WEIGHTING_PARAMS,
    auditory_weighting,
    frequency_weighting,
)
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._validate import normalize_axis


def apply_weighting(level_dB, *, frequency, weighting, axis: int = -1):
    """Apply a frequency weighting to a per-frequency level spectrum:
    ``L + W(f)`` [dB].

    ``W(f)`` is :func:`~uacpy.acoustics.frequency_weighting`'s, so
    ``weighting`` takes the three forms every ``weighting=`` takes, and a
    frequency at or below 0 Hz (a spectrum's DC bin) carries no weight:
    ``-inf`` there.

    Parameters
    ----------
    level_dB : float or array_like
        Level per frequency (dB), one spectrum or one per cell of the other
        axes, frequency along ``axis``; a single level is weighted at every
        frequency.
    frequency : float or array_like
        Frequency (Hz): one value, or one per sample along ``axis``.
    weighting : str, callable or (array_like, array_like)
        A hearing group of :data:`HEARING_GROUPS` (Southall et al. 2019),
        your own function ``W(frequency) -> dB``, or a
        ``(frequencies, W_dB)`` table.
    axis : int, default -1
        The frequency axis of ``level_dB``, as on :func:`weighted_level`.
    """
    level = np.asarray(level_dB, dtype=float)
    f = np.asarray(frequency, dtype=float)
    if f.ndim > 1:
        raise ConfigurationError(
            f"apply_weighting: frequency is one value or one axis of "
            f"values; got shape {f.shape}.")
    weight = frequency_weighting(frequency, weighting)
    if level.ndim == 0 or f.ndim == 0:
        return level + weight
    axis = normalize_axis(level, axis, "apply_weighting")
    if level.shape[axis] != f.size:
        raise ConfigurationError(
            f"apply_weighting: level_dB has {level.shape[axis]} samples "
            f"along axis={axis} (shape {level.shape}) but frequency has "
            f"{f.size}; one level per frequency along axis=.")
    return np.moveaxis(np.moveaxis(level, axis, -1) + weight, -1, axis)


def weighted_level(psd_dB, *, frequency, weighting, axis: int = -1):
    """Broadband weighted level [dB] from a level-*density* spectrum.

    Integrates the weighted spectral density over frequency::

        10·log10( ∫ 10^((L(f) + W(f))/10) df )

    where ``psd_dB`` is a level density (dB re ref²/Hz) at ``frequency`` [Hz].
    Integrating — rather than summing the samples — makes the result
    **independent of the frequency-grid spacing** (a bare sum is not: it scales
    with the number of bins). Mirrors how :func:`uacpy.acoustic_signal.welch`
    and SEL integrate a PSD. ``frequency`` need not be pre-sorted, but it needs
    at least two entries to span a bandwidth; the level of a single frequency
    is :func:`apply_weighting`. A DC bin (``f <= 0``, as :func:`welch`
    returns first) carries no weight, so a welch spectrum is taken as it
    comes. The integral is :func:`uacpy.core.acoustics.band_level`'s, so a
    spectrum that carries no power reads ``-inf``, the level of an empty band
    throughout uacpy, and one whose every bin is the no-energy marker
    (-600 dB, a silent record) reads the marker: the weighting does not move
    a marker bin.

    Parameters
    ----------
    psd_dB : array_like
        Level density (dB re ref²/Hz): one spectrum, or one per cell of the
        other axes.
    frequency : array_like
        Frequencies (Hz), at least two, one per sample along ``axis``.
    weighting : str, callable or (array_like, array_like)
        As on :func:`apply_weighting`.
    axis : int, default -1
        The frequency axis of ``psd_dB``.

    Returns
    -------
    float or ndarray
        The weighted level: a float for one spectrum, ``psd_dB``'s shape
        without ``axis`` otherwise.
    """
    f = np.atleast_1d(np.asarray(frequency, dtype=float))
    # Fewer than two samples span zero bandwidth, so the trapezoid integral is
    # 0 and the returned level would be the 10·log10(float-tiny) floor
    # (-3076.5 dB), so it is refused. A scalar is normalised to shape (1,) first,
    # so it is refused with the same message rather than crashing on indexing.
    if f.size < 2:
        raise ConfigurationError(
            f"weighted_level: integrating the weighted density over frequency "
            f"needs at least two frequencies to span a bandwidth; got "
            f"{f.size}.",
            remediation="For the weighted level at a single frequency use "
                        "apply_weighting(psd_dB, frequency=f, "
                        "weighting=weighting).")
    psd = np.asarray(psd_dB, dtype=float)
    if psd.ndim == 0:
        raise ConfigurationError(
            f"weighted_level: psd_dB is one value; it holds one level per "
            f"frequency along axis=, {f.size} here.")
    axis = normalize_axis(psd, axis, "weighted_level")
    if psd.shape[axis] != f.size:
        raise ConfigurationError(
            f"weighted_level: psd_dB has {psd.shape[axis]} samples along "
            f"axis={axis} (shape {psd.shape}) but frequency has {f.size}; "
            f"one level per frequency along axis=.")
    order = np.argsort(f)
    density = np.moveaxis(psd, axis, -1)[..., order]
    # A no-energy bin (-600 dB) holds no level to weight: it stays the
    # marker, so a silent spectrum's weighted level is the marker too.
    weighted = np.where(no_energy_mask(density), density,
                        density + frequency_weighting(f[order], weighting))
    return band_level(weighted, f[order], axis=-1)
