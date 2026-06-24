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
    Hearing Effects." *Aquatic Mammals* 45(2), 125-232 — Table 5. Consistent
    with NMFS (2018) Technical Guidance (NMFS-OPR-59).
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError

# Southall et al. (2019) Table 5: a, b, f1 [kHz], f2 [kHz], C [dB] (weighting)
# and K [dB] (TTS/PTS exposure-function position).
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


def auditory_weighting(frequency, group):
    """Auditory weighting ``W(f)`` [dB] at ``frequency`` [Hz] for a hearing group.

    ``group`` is one of :data:`HEARING_GROUPS` (e.g. ``"LF"``, ``"VHF"``,
    ``"PCW"``). The peak of the function is 0 dB; all other values are negative.
    """
    g = str(group).upper()
    if g not in WEIGHTING_PARAMS:
        raise ConfigurationError(
            f"auditory_weighting: unknown group {group!r}; choose from "
            f"{sorted(WEIGHTING_PARAMS)}")
    p = WEIGHTING_PARAMS[g]
    f = np.asarray(frequency, dtype=float) / 1000.0        # Hz -> kHz
    r1 = (f / p["f1"]) ** 2
    r2 = (f / p["f2"]) ** 2
    return p["C"] + 10.0 * np.log10(
        r1 ** p["a"] / ((1 + r1) ** p["a"] * (1 + r2) ** p["b"]))


def apply_weighting(level_db, frequency, group):
    """Apply the group weighting to a per-frequency level spectrum: ``L + W(f)`` [dB]."""
    return np.asarray(level_db, dtype=float) + auditory_weighting(frequency, group)


def weighted_level(psd_db, frequency, group):
    """Broadband group-weighted level [dB] from a level-*density* spectrum.

    Integrates the weighted spectral density over frequency::

        10·log10( ∫ 10^((L(f) + W(f))/10) df )

    where ``psd_db`` is a level density (dB re ref²/Hz) at ``frequency`` [Hz].
    Integrating — rather than summing the samples — makes the result
    **independent of the frequency-grid spacing** (a bare sum is not: it scales
    with the number of bins). Mirrors how :func:`uacpy.acoustic_signal.bands`
    and SEL integrate a PSD. ``frequency`` need not be pre-sorted.
    """
    w = np.asarray(apply_weighting(psd_db, frequency, group), dtype=float)
    f = np.asarray(frequency, dtype=float)
    order = np.argsort(f)
    integral = np.trapezoid(10.0 ** (w[order] / 10.0), f[order])
    return float(10.0 * np.log10(max(float(integral), np.finfo(float).tiny)))
