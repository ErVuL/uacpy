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

from uacpy.core.acoustics import band_level
from uacpy.core.acoustics.weighting import (  # noqa: F401  (re-exported by uacpy.noise)
    EXPOSURE_THRESHOLDS,
    HEARING_GROUPS,
    WEIGHTING_PARAMS,
    auditory_weighting,
)
from uacpy.core.exceptions import ConfigurationError


def apply_weighting(level_dB, *, frequency, group):
    """Apply the group weighting to a per-frequency level spectrum: ``L + W(f)`` [dB].

    Parameters
    ----------
    level_dB : float or array_like
        Level per frequency (dB).
    frequency : float or array_like
        Frequency (Hz).
    group : str
        A hearing group of :data:`HEARING_GROUPS`.
    """
    return np.asarray(level_dB, dtype=float) + auditory_weighting(frequency, group)


def weighted_level(psd_dB, *, frequency, group):
    """Broadband group-weighted level [dB] from a level-*density* spectrum.

    Integrates the weighted spectral density over frequency::

        10·log10( ∫ 10^((L(f) + W(f))/10) df )

    where ``psd_dB`` is a level density (dB re ref²/Hz) at ``frequency`` [Hz].
    Integrating — rather than summing the samples — makes the result
    **independent of the frequency-grid spacing** (a bare sum is not: it scales
    with the number of bins). Mirrors how :func:`uacpy.acoustic_signal.welch`
    and SEL integrate a PSD. ``frequency`` need not be pre-sorted, but it needs
    at least two entries to span a bandwidth; the level of a single frequency
    is :func:`apply_weighting`. The integral is
    :func:`uacpy.core.acoustics.band_level`'s, so a spectrum that carries no
    power reads ``-inf``, the level of an empty band throughout uacpy.

    Parameters
    ----------
    psd_dB : array_like
        Level density (dB re ref²/Hz) at each frequency.
    frequency : array_like
        Frequencies (Hz), at least two.
    group : str
        A hearing group of :data:`HEARING_GROUPS`.
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
            f"{f.size}. For the weighted level at a single frequency use "
            f"apply_weighting(psd_dB, frequency, group).")
    w = np.atleast_1d(
        np.asarray(apply_weighting(psd_dB, frequency=frequency, group=group), dtype=float))
    order = np.argsort(f)
    return band_level(w[order], f[order])
