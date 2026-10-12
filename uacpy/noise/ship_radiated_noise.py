"""Ship underwater radiated noise — RNL and equivalent monopole source level,
measured or modelled.

Converts a far-field hydrophone measurement of a passing ship into its Radiated
Noise Level (RNL) and the equivalent Monopole Source Level (MSL), per
ISO 17208-1/-2; without a measurement,
:func:`ship_source_level_wales_heitmeyer` is the mean monopole source spectrum
of a merchant ship (Wales and Heitmeyer 2002, as Ainslie 2010 Eq. 8.195 gives
it). The MSL removes the sea-surface (Lloyd's mirror) interference
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
from uacpy.core._validate import require_positive_finite_scalar

# Combined RNL measurement uncertainty by band (ISO 17208-2:2019 §5), in dB.
# Reference figures to report alongside a measured level; no function here
# applies them, since the uncertainty of a *modelled* level is the model's.
RNL_UNCERTAINTY_DB = {
    "low": 5.0,     # 10 Hz - 100 Hz bands
    "mid": 3.0,     # 125 Hz - 16 kHz bands
    "high": 4.0,    # > 20 kHz bands
}


def radiated_noise_level(received_spl_dB, distance_m):
    """Radiated Noise Level from a far-field SPL measurement (ISO 17208-1).

    ``L_RN = L_p + 20*log10(r)`` — the received decidecade-band SPL plus spherical
    (``20 log r``) spreading. ``distance_m`` is the slant range from the ship
    reference point to the hydrophone. Returns dB re 1 µPa²·m².

    Parameters
    ----------
    received_spl_dB : float or array_like
        Received decidecade-band SPL (dB re 1 µPa²).
    distance_m : float or array_like
        Slant range (m) from the ship reference point.
    """
    r = np.asarray(distance_m, dtype=float)
    # Written as the negation of the admissible condition so NaN is refused
    # too: every comparison against NaN is False, so `r <= 0` let a NaN range
    # through to a NaN level. `isfinite` is the other half — an infinite range
    # returned +inf, "no ship is loud enough to hear", the safe-looking
    # direction for a source level to be wrong in.
    if not np.all(np.isfinite(r) & (r > 0)):
        raise ConfigurationError(
            f"radiated_noise_level: distance must be > 0 m and finite; got "
            f"{int(np.count_nonzero(~(np.isfinite(r) & (r > 0))))} bad "
            f"value(s) of {r.size}, first {float(np.ravel(r)[np.argmax(~(np.isfinite(r) & (r > 0)).ravel())])!r}.")
    return np.asarray(received_spl_dB, dtype=float) + 20.0 * np.log10(r)


def nominal_source_depth(draught_m):
    """Nominal monopole source depth ``d_s = 0.7 * draught`` (ISO 17208-2 Formula 1).

    ``draught_m`` may be one draught (a float comes back) or an array of
    them (a fleet).

    Parameters
    ----------
    draught_m : float or array_like
        Ship draught (m).
    """
    d = np.asarray(draught_m, dtype=float)
    if not np.all(np.isfinite(d) & (d > 0)):
        raise ConfigurationError(
            f"nominal_source_depth: draught must be > 0 m and finite; got "
            f"{draught_m!r}.")
    return 0.7 * float(d) if d.ndim == 0 else 0.7 * d


def lloyd_mirror_correction(frequency, source_depth, sound_speed=DEFAULT_SOUND_SPEED):
    """ISO 17208-2 Formula 3 surface-image correction ``ΔL = L_s - L_RN`` [dB].

    Pressure-release sea surface (Lloyd's mirror), broadside aspect:

    ``ΔL = -10 log10[ (2(kd)^4 + 14(kd)^2) / (14 + 2(kd)^2 + (kd)^4) ]``,
    ``k = 2πf/c``, ``d = d_s``.

    ``frequency`` is the decidecade band centre [Hz]. ΔL → -3.01 dB at high
    frequency (incoherent source+image) and grows large and positive at low
    frequency (the surface dipole suppresses radiation). ``frequency`` and
    ``source_depth`` broadcast against each other.

    Parameters
    ----------
    frequency : float or array_like
        Decidecade band centre (Hz).
    source_depth : float or array_like
        Source depth (m).
    sound_speed : float, optional
        Sound speed (m/s). Default :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
    """
    # ``kd`` enters only as even powers, so a negative depth returns exactly
    # what its positive twin does: an upstream sign error would be invisible.
    # Zero is admitted — it is the physical surface-mounted limit the
    # ``errstate`` below is for — and NaN is refused by the negated form.
    d = np.asarray(source_depth, dtype=float)
    if not np.all(np.isfinite(d) & (d >= 0)):
        raise ConfigurationError(
            f"lloyd_mirror_correction: source_depth must be >= 0 m and "
            f"finite; got {source_depth!r}. The correction depends on "
            f"(k*d)**2 and (k*d)**4 only, so a negative depth would return "
            f"the same value as its positive twin.")
    c = require_positive_finite_scalar(
        sound_speed, "lloyd_mirror_correction", "sound_speed", " m/s")
    k = 2.0 * np.pi * np.asarray(frequency, dtype=float) / c
    kd = k * d
    num = 2.0 * kd ** 4 + 14.0 * kd ** 2
    den = 14.0 + 2.0 * kd ** 2 + kd ** 4
    # At kd→0 (surface-mounted source) num→0 and ΔL→+inf: the pressure-release
    # image exactly cancels the source. That limit is physical, so allow it
    # without polluting NumPy's warning state.
    with np.errstate(divide='ignore'):
        return -10.0 * np.log10(num / den)


def monopole_source_level(rnl_dB, frequency, source_depth, sound_speed=DEFAULT_SOUND_SPEED):
    """Equivalent Monopole Source Level ``L_s = L_RN + ΔL`` (ISO 17208-2 Formula 2).

    ``frequency`` is the decidecade band centre(s) [Hz]; ``source_depth`` the
    nominal source depth (``0.7 * draught``). Returns dB re 1 µPa²·m².

    Parameters
    ----------
    rnl_dB : float or array_like
        Radiated noise level (dB re 1 µPa²·m²).
    frequency : float or array_like
        Decidecade band centre(s) (Hz).
    source_depth : float or array_like
        Nominal source depth (m).
    sound_speed : float, optional
        Sound speed (m/s). Default :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
    """
    return np.asarray(rnl_dB, dtype=float) + lloyd_mirror_correction(
        frequency, source_depth, sound_speed)


#: The band (Hz) the Wales and Heitmeyer (2002) ensemble was measured over
#: (Ainslie 2010, sect. 8.3.1.3).
_WH02_MEASURED_BAND_HZ = (30.0, 1200.0)

#: The highest frequency (Hz) of Ainslie's extrapolation of the WH02 mean
#: (sect. 8.3.1.4, Fig. 8.16).
_WH02_EXTRAPOLATED_MAX_HZ = 15000.0


def _wh02_mean(f_hz):
    """Ainslie (2010) Eq. (8.195), dB re 1 µPa² m²/Hz at ``f_hz`` (Hz)."""
    return (230.0 - 35.94 * np.log10(f_hz)
            + 9.17 * np.log10(1.0 + (f_hz / 340.0) ** 2))


def ship_source_level_wales_heitmeyer(frequency):
    """Mean merchant-ship source spectral density level, dB re 1 µPa² m²/Hz.

    A level per hertz of the monopole source spectrum: the Wales and
    Heitmeyer (2002) ensemble mean. A band level is this density integrated
    over the band (below).

    Ainslie (*Principles of Sonar Performance Modelling*, 2010, sect.
    8.3.1.3, Eq. 8.195) gives it as::

        SL_mp,f = 230.0 - 35.94·log10 f + 9.17·log10[1 + (f/340)²]

    with ``f`` in Hz: the mean of the monopole source spectra Wales and
    Heitmeyer measured on 272 ships in the Mediterranean Sea and the eastern
    Atlantic, between 30 Hz and 1200 Hz. Above 1200 Hz it continues with
    the gradient Ainslie uses to extrapolate it (sect. 8.3.1.4), ``-23·log10
    f``, joined at 1200 Hz, up to 15 kHz; that gradient was chosen to match
    one cargo ship measured by Arveson and Vendittis (2000) at low speed,
    and the extrapolated curve agrees with that ship near 6.2 m/s (12 kn)
    (Fig. 8.16). A monopole level is what a propagation model takes as the
    source level of a point source at the ship's nominal source depth
    (:func:`nominal_source_depth`); a measured ship's is
    :func:`monopole_source_level`.

    It is an ensemble mean, not one ship's spectrum, and has no speed or
    length term: Ainslie states none for it. One ship's spectrum moves with
    its speed (Fig. 8.16 shows the Overseas Harriette at several speeds),
    and the Scrimger and Heitmeyer (1991) mean over 50 ships approaching
    Genoa at 7.2 m/s (14 kn) lies 6-12 dB above this one, a difference
    Ainslie attributes in part to the propagation loss each measurement
    subtracted.

    A band level is this density integrated over the band: about the level
    at the band centre plus ``10·log10(bandwidth)`` for a decidecade band
    (``uacpy.acoustics.band_level`` integrates it exactly).

    Parameters
    ----------
    frequency : float or array_like
        Frequency (Hz), between 30 Hz and 15 kHz.

    Returns
    -------
    float or ndarray
        dB re 1 µPa² m²/Hz, shaped like ``frequency``.

    Raises
    ------
    ConfigurationError
        A frequency outside 30 Hz-15 kHz: below 30 Hz nothing was measured,
        and above 15 kHz the extrapolation has no data to agree with.

    Examples
    --------
    >>> round(float(ship_source_level_wales_heitmeyer(100.0)), 2)
    158.45
    """
    f = np.asarray(frequency, dtype=float)
    lo, hi = _WH02_MEASURED_BAND_HZ[0], _WH02_EXTRAPOLATED_MAX_HZ
    if f.size == 0:
        raise ConfigurationError(
            "ship_source_level_wales_heitmeyer: frequency is empty.",
            remediation="Pass at least one frequency in 30-15000 Hz.")
    if not np.all(np.isfinite(f)) or np.any((f < lo) | (f > hi)):
        raise ConfigurationError(
            f"ship_source_level_wales_heitmeyer: frequency must lie in "
            f"{lo:g}-{hi:g} Hz, the band the Wales and Heitmeyer (2002) mean "
            f"was measured over ({_WH02_MEASURED_BAND_HZ[0]:g}-"
            f"{_WH02_MEASURED_BAND_HZ[1]:g} Hz) and extrapolated to; got "
            f"{np.atleast_1d(f).min():g}-{np.atleast_1d(f).max():g} Hz.",
            remediation="Evaluate it on the band it covers; a spectrum "
                        "below 30 Hz or above 15 kHz has to come from a "
                        "measurement.")
    edge = _WH02_MEASURED_BAND_HZ[1]
    level = np.where(f <= edge, _wh02_mean(f),
                     _wh02_mean(edge) - 23.0 * np.log10(f / edge))
    return float(level) if level.ndim == 0 else level

