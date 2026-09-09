"""Cell-scattering boundary and volume reverberation (Urick 1983, Ch. 8).

Monostatic reverberation level versus range under the short-pulse
cell-scattering approximation:

    boundary: ``RL(r) = SL - 2*TL(r) + S_b + 10*log10(Phi * r * c*tau/2)``
    volume:   ``RL(r) = SL - 2*TL(r) + S_v + 10*log10(Psi * r^2 * c*tau/2)``

The reverberating cell has range extent ``c*tau/2`` (half the pulse travel).
``TL`` defaults to spherical spreading ``20*log10(r)``; pass a model-derived
one-way TL array (e.g. from a uacpy Field) to use a real propagation field.

* ``Phi`` — equivalent two-way horizontal beamwidth (rad) of the array.
* ``Psi`` — equivalent two-way solid-angle beamwidth (sr) of the array.

References
----------
Urick, R.J. (1983). *Principles of Underwater Sound*, 3rd ed., Ch. 8.
Etter, P.C. *Underwater Acoustic Modeling and Simulation*, Sec. 10.3.
"""

from __future__ import annotations

import warnings

import numpy as np

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.acoustic_signal._signal_validate import require_positive_finite_scalar


def _check_ranges(caller: str, ranges_m) -> np.ndarray:
    """Slant ranges as a float array, refusing negatives.

    A negative slant range has no meaning, and the formulas do not report one:
    ``20*log10(r)`` and ``10*log10(cell)`` both go NaN, so the level came back
    as NaN carrying a bare ``invalid value encountered in log10`` from numpy
    rather than anything naming the argument. ``r == 0`` stays NaN — it is the
    package's no-data convention and the cell there genuinely has zero area —
    but it is produced deliberately below instead of falling out of inf - inf.
    """
    r = np.asarray(ranges_m, dtype=float)
    # ``~(r >= 0)`` rather than ``r < 0``: it keeps admitting r == 0 (the
    # zero-area cell handled below) while also refusing NaN and -inf, which
    # pass every ``<`` comparison and would produce a NaN level indistinguishable
    # from the deliberate zero-range one. ``isfinite`` carries the other half of
    # the message's "and finite", which the sign test alone does not: ``+inf >= 0``
    # is True, so an infinite range reached the formula and came back as exactly
    # that ambiguous NaN — the spreading term's ``-2*20*log10(inf)`` against the
    # cell term's ``+10*log10(inf)`` is the inf - inf this guard exists to avoid.
    invalid = ~np.isfinite(r) | ~(r >= 0.0)
    if np.any(invalid):
        bad = r[invalid]
        raise ConfigurationError(
            f"{caller}: ranges_m must be >= 0 and finite; got {bad.size} "
            f"invalid value(s), e.g. {float(bad.flat[0]):g} m.",
            remediation="Pass slant ranges as positive distances in metres.",
        )
    return r


def _warn_if_cell_is_not_short(caller: str, r: np.ndarray,
                               sound_speed: float,
                               pulse_length_s: float) -> None:
    """Warn where the cell-scattering approximation has left its domain.

    The published form takes the scattering area as ``Phi * r * (c*tau/2)``,
    i.e. an annulus of width ``c*tau/2`` treated as if it sat entirely at
    range ``r``. The exact annulus between ``r`` and ``r + c*tau/2`` has area
    ``Phi * (r2**2 - r1**2) / 2``, so the approximation is low by exactly

        10*log10(1 + c*tau/(4*r))

    — verified against the closed-form annulus area to every digit: 0.022 dB
    at ``c*tau/2 = r/100``, 0.212 dB at ``r/10``, 0.969 dB at ``r/2`` and
    1.761 dB once the cell is as long as the range to it. That last case is
    the natural place to stop: the annulus' inner edge has reached the source,
    so the geometry the formula describes no longer exists. Short ranges with
    a long pulse otherwise returned a confident number — 178.75 dB at r = 1 m
    with a 75 m cell — with nothing to say it was meaningless.
    """
    extent = float(sound_speed) * float(pulse_length_s) / 2.0
    near = r[(r > 0.0) & (r < extent)]
    if near.size:
        r_min = float(near.min())
        err = 10.0 * np.log10(1.0 + float(sound_speed) *
                              float(pulse_length_s) / (4.0 * r_min))
        warnings.warn(
            f"{caller}: the cell's range extent c*tau/2 = {extent:g} m is "
            f"longer than {near.size} of the requested range(s) (shortest "
            f"{r_min:g} m), so the short-pulse cell-scattering approximation "
            f"no longer holds there — the level is low by about {err:.2f} dB "
            f"at that range and worse closer in.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )


def _resolve_tl(ranges_m: np.ndarray, tl_dB) -> np.ndarray:
    """One-way TL (dB) at each range. ``None`` -> spherical ``20*log10(r)``."""
    r = np.asarray(ranges_m, dtype=float)
    if tl_dB is None:
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(r)
    if callable(tl_dB):
        return np.asarray(tl_dB(r), dtype=float)
    tl = np.asarray(tl_dB, dtype=float)
    if tl.shape != r.shape:
        raise ConfigurationError(
            f"reverberation: tl_dB shape {tl.shape} != ranges shape {r.shape}"
        )
    return tl


def _reverberation(caller, beam_name, beam_value, cell_of_range, ranges_m,
                   source_level, scattering_strength_dB, pulse_length_s,
                   sound_speed, tl_dB):
    """``SL - 2*TL + S + 10*log10(cell_of_range(r) * c*tau/2)``; the guards
    run pulse/beam, sound speed, ranges, short-cell warning, TL, so several
    bad arguments report the first (``beam_name`` names the caller's)."""
    # Negated admissible condition so NaN is refused: a NaN pulse length or
    # beamwidth otherwise returned an all-NaN reverberation level silently.
    # ``isfinite`` is the other half of the message's "and finite": the sign
    # test admits ``+inf``, and an infinite pulse length or beamwidth made the
    # cell infinite and returned an infinite level just as silently.
    if (not np.isfinite(pulse_length_s) or not (pulse_length_s > 0.0)
            or not np.isfinite(beam_value) or not (beam_value > 0.0)):
        raise ConfigurationError(
            f"{caller}: pulse_length_s and {beam_name}"
            f" must be > 0 and finite; got pulse_length_s={pulse_length_s!r}, "
            f"{beam_name}={beam_value!r}"
        )
    # The range extent is c*tau/2, so c == 0 collapses the cell to zero area
    # (-inf level), c < 0 takes log10 of a negative cell (NaN) and c = inf
    # returns +inf; the shared positive-scalar guard refuses all three, as
    # ``target_strength`` does for the same argument.
    sound_speed = require_positive_finite_scalar(sound_speed, caller,
                                                 "sound_speed", " m/s")
    r = _check_ranges(caller, ranges_m)
    _warn_if_cell_is_not_short(caller, r, sound_speed, pulse_length_s)
    tl = _resolve_tl(r, tl_dB)
    s = np.asarray(scattering_strength_dB, dtype=float)
    cell = cell_of_range(r) * (sound_speed * pulse_length_s / 2.0)
    zero_range = r == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = source_level - 2.0 * tl + s + 10.0 * np.log10(cell)
    return np.where(zero_range, np.nan, out)


def boundary_reverberation(
    ranges_m,
    source_level: float,
    scattering_strength_dB,
    *,
    pulse_length_s: float,
    horizontal_beamwidth_rad: float,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    tl_dB=None,
):
    """Boundary (surface or bottom) reverberation level vs range (dB).

    Parameters
    ----------
    ranges_m : array
        Slant ranges to the scattering cell (m).
    source_level : float
        Source level (dB re 1 uPa @ 1 m).
    scattering_strength_dB : float or array
        Boundary scattering strength ``S_b`` (dB); scalar or per-range (e.g.
        Lambert's law evaluated at the grazing angle of each range).
    pulse_length_s : float
        Transmit pulse length ``tau`` (s).
    horizontal_beamwidth_rad : float
        Equivalent two-way horizontal beamwidth ``Phi`` (rad).
    sound_speed : float
        Sound speed (m/s).
    tl_dB : None, callable, or array
        One-way transmission loss (dB). ``None`` -> spherical spreading.

    Returns
    -------
    ndarray
        Reverberation level (dB) at each range.
    """
    return _reverberation(
        "boundary_reverberation", "horizontal_beamwidth_rad", horizontal_beamwidth_rad,
        lambda r: horizontal_beamwidth_rad * r, ranges_m, source_level,
        scattering_strength_dB, pulse_length_s, sound_speed, tl_dB)


def volume_reverberation(
    ranges_m,
    source_level: float,
    scattering_strength_dB,
    *,
    pulse_length_s: float,
    solid_angle_beamwidth_sr: float,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    tl_dB=None,
):
    """Volume reverberation level vs range (dB).

    Same arguments as :func:`boundary_reverberation`, except
    ``scattering_strength_dB`` is the volume scattering strength ``S_v``
    (dB re 1/m) and ``solid_angle_beamwidth_sr`` is the equivalent two-way
    solid-angle beamwidth ``Psi`` (sr). The cell volume grows as ``r^2``.
    """
    return _reverberation(
        "volume_reverberation", "solid_angle_beamwidth_sr", solid_angle_beamwidth_sr,
        lambda r: solid_angle_beamwidth_sr * r ** 2, ranges_m, source_level,
        scattering_strength_dB, pulse_length_s, sound_speed, tl_dB)


def total_reverberation(*levels_dB):
    """Incoherent (power) sum of reverberation components (dB).

    ``RL = 10*log10(sum_i 10^(RL_i/10))`` over surface, bottom, volume, ...
    Each argument is a scalar or an array of matching shape.
    """
    if not levels_dB:
        raise ConfigurationError("total_reverberation: need at least one level")
    arrs = [np.asarray(x, dtype=float) for x in levels_dB]
    if any(a.size == 0 for a in arrs):
        raise ConfigurationError(
            "total_reverberation: received a zero-length level array; "
            "every component must carry at least one sample. Got sizes "
            f"{[a.size for a in arrs]}."
        )
    stack = np.stack(np.broadcast_arrays(*arrs), axis=0)
    return 10.0 * np.log10(np.sum(10.0 ** (stack / 10.0), axis=0))
