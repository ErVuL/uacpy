"""Marine-mammal auditory weighting functions (Southall et al. 2019).

Frequency weighting that accounts for the differential hearing sensitivity of
marine-mammal groups — applied to a received noise spectrum to assess auditory
impact (temporary/permanent threshold shift). The weighting ``W(f)`` [dB] for a
hearing group is (Southall et al. 2019, Eq. 2):

    W(f) = C + 10·log10[ (f/f1)^(2a) / ( (1+(f/f1)^2)^a · (1+(f/f2)^2)^b ) ]

with ``f, f1, f2`` in kHz; the low- and high-frequency roll-off slopes are
``+20a`` and ``-20b`` dB/decade, and ``C`` sets the function peak to 0 dB.

References
----------
Southall, B. L., Finneran, J. J., Reichmuth, C., et al. (2019). "Marine Mammal
    Noise Exposure Criteria: Updated Scientific Recommendations for Residual
    Hearing Effects." *Aquatic Mammals* 45(2), 125-232 — Table 5. The
    parameters agree with NMFS (2018) Technical Guidance (NMFS-OPR-59) where
    the two overlap, but Southall renamed the cetacean groups, so the labels
    move one step: NMFS LF / MF / HF / PW / OW are Southall LF / HF / VHF /
    PCW / OCW here. NMFS "HF" (harbour porpoise) is this module's ``"VHF"``;
    this module's ``"HF"`` is NMFS "MF" (most dolphins): Southall et al.
    (2019) draw "the distinction between HF and VHF cetacean groups (as
    opposed to mid- and high-frequency)", and their HF group "represents
    most of the same species identified as MF cetaceans by ... NMFS".
"""

from __future__ import annotations

import numpy as np

from uacpy.core.acoustics import band_level
from uacpy.core.exceptions import ConfigurationError

# Southall et al. (2019) Table 5: a, b, f1 [kHz], f2 [kHz], C [dB] (weighting)
# and K [dB]. Only a/b/f1/f2/C enter :func:`auditory_weighting`; K is carried
# as published reference for callers setting their own exposure criteria.
#
# K positions the **non-impulsive TTS** exposure function only — Table 5 is
# titled "auditory weighting function and TTS exposure function parameters",
# and K was fitted to minimise the error against measured TTS onset. It is not
# a PTS constant: Southall's PTS onset is uniformly TTS onset + 20 dB (their
# Table 6, exactly +20 dB for all eight groups), so reading K as PTS is 20 dB
# low. K is also non-impulsive; the impulsive exposure functions keep the same
# shape with their own K, ~11 dB lower for the groups without impulsive data.
# The published TTS threshold is the *minimum of the exposure function* rather
# than K itself, so the two differ by a dB or so (HF: K = 177, threshold 178).
WEIGHTING_PARAMS = {
    "LF":  {"a": 1.0, "b": 2, "f1": 0.20, "f2": 19.0,  "C": 0.13, "K": 179},
    "HF":  {"a": 1.6, "b": 2, "f1": 8.8,  "f2": 110.0, "C": 1.20, "K": 177},
    "VHF": {"a": 1.8, "b": 2, "f1": 12.0, "f2": 140.0, "C": 1.36, "K": 152},
    "SI":  {"a": 1.8, "b": 2, "f1": 4.3,  "f2": 25.0,  "C": 2.62, "K": 183},
    "PCW": {"a": 1.0, "b": 2, "f1": 1.9,  "f2": 30.0,  "C": 0.75, "K": 180},
    "OCW": {"a": 2.0, "b": 2, "f1": 0.94, "f2": 25.0,  "C": 0.64, "K": 198},
    "PCA": {"a": 2.0, "b": 2, "f1": 0.75, "f2": 8.3,   "C": 1.50, "K": 132},
    "OCA": {"a": 1.4, "b": 2, "f1": 2.0,  "f2": 20.0,  "C": 1.39, "K": 156},
}

#: Southall et al. (2019) hearing groups. The cetacean labels are NOT NMFS
#: (2018)'s: NMFS MF is ``"HF"`` here and NMFS HF is ``"VHF"`` (see the module
#: docstring); NMFS PW / OW are ``"PCW"`` / ``"OCW"``.
HEARING_GROUPS = {
    "LF": "Low-frequency cetaceans",
    "HF": "High-frequency cetaceans",
    "VHF": "Very-high-frequency cetaceans",
    "SI": "Sirenians",
    "PCW": "Phocid carnivores in water",
    "OCW": "Other marine carnivores in water",
    "PCA": "Phocid carnivores in air",
    "OCA": "Other marine carnivores in air",
}


#: NMFS (2018) group names with no Southall (2019) group of the same name,
#: and the Southall group each one is.
_NMFS_ONLY_GROUPS = {"MF": "HF", "PW": "PCW", "OW": "OCW"}


def auditory_weighting(frequency, group):
    """Auditory weighting ``W(f)`` [dB] at ``frequency`` [Hz] for a hearing group.

    ``group`` is one of :data:`HEARING_GROUPS` (e.g. ``"LF"``, ``"VHF"``,
    ``"PCW"``), in Southall et al. (2019)'s naming: a harbour porpoise, NMFS
    (2018) "HF", is ``"VHF"`` here. The peak of the function is 0 dB; all
    other values are negative.

    Parameters
    ----------
    frequency : float or array_like
        Frequency (Hz).
    group : str
        A hearing group of :data:`HEARING_GROUPS`.
    """
    g = str(group).upper()
    if g in _NMFS_ONLY_GROUPS:
        raise ConfigurationError(
            f"auditory_weighting: {group!r} is an NMFS (2018) group name; "
            f"this module uses Southall et al. (2019)'s, where NMFS "
            f"{g} is {_NMFS_ONLY_GROUPS[g]!r}. Note that NMFS 'HF' is "
            f"Southall 'VHF', and Southall 'HF' is NMFS 'MF'.")
    if g not in WEIGHTING_PARAMS:
        raise ConfigurationError(
            f"auditory_weighting: unknown group {group!r}; choose from "
            f"{sorted(WEIGHTING_PARAMS)}.")
    p = WEIGHTING_PARAMS[g]
    f = np.asarray(frequency, dtype=float) / 1000.0        # Hz -> kHz
    # Eq. (2) is even in f, so a negative frequency squares away and comes back
    # as a plausible finite weighting; f = 0 gives -inf behind a bare numpy
    # divide-by-zero warning. Both are input errors, rejected here the way
    # WenzNoise rejects a non-positive frequency grid.
    # ``~(f > 0)`` rather than ``f <= 0``: NaN passes every comparison and would
    # return a NaN weighting silently.
    if f.size == 0 or np.any(~(f > 0.0)):
        raise ConfigurationError(
            "auditory_weighting: frequency must be > 0 Hz and finite (the "
            "weighting is a log10 of (f/f1)^2a); drop the DC bin, e.g. "
            "f[f > 0].")
    r1 = (f / p["f1"]) ** 2
    r2 = (f / p["f2"]) ** 2
    return p["C"] + 10.0 * np.log10(
        r1 ** p["a"] / ((1 + r1) ** p["a"] * (1 + r2) ** p["b"]))


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
    # (-3076.5 dB) — refused the way wind_noise_level(band_integrate=True)
    # refuses a single frequency. A scalar is normalised to shape (1,) first,
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
