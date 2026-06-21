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

import numpy as np

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError


def _resolve_tl(ranges_m: np.ndarray, tl_db) -> np.ndarray:
    """One-way TL (dB) at each range. ``None`` -> spherical ``20*log10(r)``."""
    r = np.asarray(ranges_m, dtype=float)
    if tl_db is None:
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(r)
    if callable(tl_db):
        return np.asarray(tl_db(r), dtype=float)
    tl = np.asarray(tl_db, dtype=float)
    if tl.shape != r.shape:
        raise ConfigurationError(
            f"reverberation: tl_db shape {tl.shape} != ranges shape {r.shape}"
        )
    return tl


def boundary_reverberation(
    ranges_m,
    source_level_db: float,
    scattering_strength_db,
    *,
    pulse_length_s: float,
    horizontal_beamwidth_rad: float,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    tl_db=None,
):
    """Boundary (surface or bottom) reverberation level vs range (dB).

    Parameters
    ----------
    ranges_m : array
        Slant ranges to the scattering cell (m).
    source_level_db : float
        Source level (dB re 1 uPa @ 1 m).
    scattering_strength_db : float or array
        Boundary scattering strength ``S_b`` (dB); scalar or per-range (e.g.
        Lambert's law evaluated at the grazing angle of each range).
    pulse_length_s : float
        Transmit pulse length ``tau`` (s).
    horizontal_beamwidth_rad : float
        Equivalent two-way horizontal beamwidth ``Phi`` (rad).
    sound_speed : float
        Sound speed (m/s).
    tl_db : None, callable, or array
        One-way transmission loss (dB). ``None`` -> spherical spreading.

    Returns
    -------
    ndarray
        Reverberation level (dB) at each range.
    """
    if pulse_length_s <= 0.0 or horizontal_beamwidth_rad <= 0.0:
        raise ConfigurationError(
            "boundary_reverberation: pulse_length_s and horizontal_beamwidth_rad"
            " must be > 0"
        )
    r = np.asarray(ranges_m, dtype=float)
    tl = _resolve_tl(r, tl_db)
    s = np.asarray(scattering_strength_db, dtype=float)
    cell = horizontal_beamwidth_rad * r * (sound_speed * pulse_length_s / 2.0)
    with np.errstate(divide="ignore"):
        return source_level_db - 2.0 * tl + s + 10.0 * np.log10(cell)


def volume_reverberation(
    ranges_m,
    source_level_db: float,
    scattering_strength_db,
    *,
    pulse_length_s: float,
    solid_angle_beamwidth_sr: float,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    tl_db=None,
):
    """Volume reverberation level vs range (dB).

    Same arguments as :func:`boundary_reverberation`, except
    ``scattering_strength_db`` is the volume scattering strength ``S_v``
    (dB re 1/m) and ``solid_angle_beamwidth_sr`` is the equivalent two-way
    solid-angle beamwidth ``Psi`` (sr). The cell volume grows as ``r^2``.
    """
    if pulse_length_s <= 0.0 or solid_angle_beamwidth_sr <= 0.0:
        raise ConfigurationError(
            "volume_reverberation: pulse_length_s and solid_angle_beamwidth_sr"
            " must be > 0"
        )
    r = np.asarray(ranges_m, dtype=float)
    tl = _resolve_tl(r, tl_db)
    s = np.asarray(scattering_strength_db, dtype=float)
    cell = solid_angle_beamwidth_sr * r ** 2 * (sound_speed * pulse_length_s / 2.0)
    with np.errstate(divide="ignore"):
        return source_level_db - 2.0 * tl + s + 10.0 * np.log10(cell)


def total_reverberation(*levels_db):
    """Incoherent (power) sum of reverberation components (dB).

    ``RL = 10*log10(sum_i 10^(RL_i/10))`` over surface, bottom, volume, ...
    Each argument is a scalar or an array of matching shape.
    """
    if not levels_db:
        raise ConfigurationError("total_reverberation: need at least one level")
    arrs = [np.asarray(x, dtype=float) for x in levels_db]
    if any(a.size == 0 for a in arrs):
        raise ConfigurationError(
            "total_reverberation: received a zero-length level array; "
            "every component must carry at least one sample."
        )
    stack = np.stack(np.broadcast_arrays(*arrs), axis=0)
    return 10.0 * np.log10(np.sum(10.0 ** (stack / 10.0), axis=0))
