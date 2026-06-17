"""Ship underwater radiated noise — RNL and equivalent monopole source level.

Converts a far-field hydrophone measurement of a passing ship into its Radiated
Noise Level (RNL) and the equivalent Monopole Source Level (MSL), per
ISO 17208-1/-2. The MSL removes the sea-surface (Lloyd's mirror) interference
assuming a pressure-release surface, giving the omni-directional point-source
description that long-range propagation models consume. Levels are reported in
decidecade bands (see :mod:`uacpy.acoustic_signal.bands`).

Standards
---------
The Radiated Noise Level here is conformant with both **ISO 17208-1:2016** and
**ANSI/ASA S12.64-2009** — the two are harmonized and use the RNL metric
unmodified (S12.64 was the basis for ISO 17208-1). The Monopole Source Level
conversion (the Lloyd's-mirror ``ΔL``) is from **ISO 17208-2:2019** §4,
Formulas 1-3, which post-processes the ANSI/ISO-17208-1 RNL.

References
----------
ISO 17208-1:2016 / ISO 17208-2:2019, *Underwater acoustics — Quantities and
    procedures for description and measurement of underwater sound from ships*
    (Part 2 §4, Formulas 1-3: ``d_s = 0.7 D``, ``L_s = L_RN + ΔL``, and ΔL).
ANSI/ASA S12.64-2009, *Quantities and Procedures for Description and Measurement
    of Underwater Sound from Ships — Part 1* (harmonized RNL).
"""

from __future__ import annotations

import numpy as np

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError

# Combined RNL measurement uncertainty by band (ISO 17208-2:2019 §5), in dB.
RNL_UNCERTAINTY_DB = {
    "low": 5.0,     # 10 Hz - 100 Hz bands
    "mid": 3.0,     # 125 Hz - 16 kHz bands
    "high": 4.0,    # > 20 kHz bands
}


def radiated_noise_level(received_spl_db, distance_m):
    """Radiated Noise Level from a far-field SPL measurement (ISO 17208-1).

    ``L_RN = L_p + 20*log10(r)`` — the received decidecade-band SPL plus spherical
    (``20 log r``) spreading. ``distance_m`` is the slant range from the ship
    reference point to the hydrophone. Returns dB re 1 µPa·m.
    """
    r = np.asarray(distance_m, dtype=float)
    if np.any(r <= 0):
        raise ConfigurationError("radiated_noise_level: distance must be > 0")
    return np.asarray(received_spl_db, dtype=float) + 20.0 * np.log10(r)


def nominal_source_depth(draught_m):
    """Nominal monopole source depth ``d_s = 0.7 * draught`` (ISO 17208-2 Formula 1)."""
    if draught_m <= 0:
        raise ConfigurationError("nominal_source_depth: draught must be > 0")
    return 0.7 * float(draught_m)


def lloyd_mirror_correction(frequency, source_depth, sound_speed=DEFAULT_SOUND_SPEED):
    """ISO 17208-2 Formula 3 surface-image correction ``ΔL = L_s - L_RN`` [dB].

    Pressure-release sea surface (Lloyd's mirror), broadside aspect:

    ``ΔL = -10 log10[ (2(kd)^4 + 14(kd)^2) / (14 + 2(kd)^2 + (kd)^4) ]``,
    ``k = 2πf/c``, ``d = d_s``.

    ``frequency`` is the decidecade band centre [Hz]. ΔL → -3.01 dB at high
    frequency (incoherent source+image) and grows large and positive at low
    frequency (the surface dipole suppresses radiation).
    """
    k = 2.0 * np.pi * np.asarray(frequency, dtype=float) / float(sound_speed)
    kd = k * float(source_depth)
    num = 2.0 * kd ** 4 + 14.0 * kd ** 2
    den = 14.0 + 2.0 * kd ** 2 + kd ** 4
    return -10.0 * np.log10(num / den)


def monopole_source_level(rnl_db, frequency, source_depth, sound_speed=DEFAULT_SOUND_SPEED):
    """Equivalent Monopole Source Level ``L_s = L_RN + ΔL`` (ISO 17208-2 Formula 2).

    ``frequency`` is the decidecade band centre(s) [Hz]; ``source_depth`` the
    nominal source depth (``0.7 * draught``). Returns dB re 1 µPa·m.
    """
    return np.asarray(rnl_db, dtype=float) + lloyd_mirror_correction(
        frequency, source_depth, sound_speed)


def plot_source_level(frequency, level_db, ax=None, title="", label="", **kwargs):
    """Plot a ship source-level spectrum (dB re 1 µPa·m vs band centre). Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    f = np.asarray(frequency, dtype=float)
    lv = np.asarray(level_db, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    ax.semilogx(f, lv, marker="o", label=label, **kwargs)
    ax.set_xlabel("Decidecade band centre [Hz]")
    ax.set_ylabel("Source level [dB re 1 µPa·m]")
    ax.set_title(f"[ship] radiated noise {title}", loc="left")
    ax.grid(which="both", alpha=0.3)
    if label:
        ax.legend()
    return fig, ax
