"""Seafloor sediment → geoacoustic ``BoundaryProperties``.

Phase 4 of the on-demand external-data layer. Unlike bathymetry and sound
speed, there is **no reliable global, no-auth point service** for seabed
geoacoustics: usSEABED / dbSEABED are sparse survey compilations (US waters,
frequently empty at an arbitrary coordinate), so a "fetch sediment at lat/lon"
call would return nothing almost everywhere. The durable, verifiable
contribution is therefore the *conversion*: turn a mean grain size (Wentworth
ϕ) or a named sediment class into a model-ready bottom.

For *European seas* a real lat/lon sediment service does exist — see
:mod:`uacpy.data.seabed` (EMODnet WFS). Elsewhere, supply a grain size or
class here.

Grain size is the natural handle. We convert ϕ to explicit
``sound_speed`` / ``density`` / ``attenuation`` (density and sound-speed
*ratios* to the overlying seawater, so fine muds come out slower than water) via
``model='hamilton'`` (default — the **low-frequency** Hamilton & Bachman (1982)
relations + Hamilton (1980) ``k_p`` attenuation) or ``model='apl-uw'`` (the
**high-frequency** APL-UW TR 9407 (1994) grain-size relations). The result is a
**half-space** bottom — usable by every model (the AT ``'grain-size'`` boundary
type is Bellhop-only and breaks Kraken/Scooter, so it is not used here).
``grain_size_phi`` is retained as informational metadata.

Sources used (the documents the coefficients were taken from):
- ``'apl-uw'``: APL-UW TR 9407 (1994), *High-Frequency Ocean Environmental
  Acoustic Models Handbook*, §IV.A.4 (Eqs. 2-5).
- ``'hamilton'``: the open-access CC-BY supplement of Fonseca, Lurton, Fezzani &
  Roche (2025), *A simplified semi-empirical model for multifrequency seafloor
  backscattering angular response (ESAB)*, which reproduces the Hamilton &
  Bachman (1982) continental-terrace table and the Hamilton (1980) ``k_p`` law.
"""

import warnings
from typing import Callable, Dict, List, Optional

import numpy as np

from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core.materials import MATERIALS, get_material, list_materials

__all__ = [
    'grain_size_to_geoacoustics',
    'bottom_from_grain_size',
    'bottom_from_class',
    'range_dependent_bottom_along',
]

# Hamilton & Bachman (1982) "Continental Terrace (Shelf and Slope)" granular
# sediments — bulk density (g/cm³) and sound-speed ratio (sediment/seawater) for
# the representative classes, referenced to seawater c_w = 1510 m/s, ρ_w = 1.030
# g/cm³. Tabulated in the open-access ESAB supplement (Fonseca et al. 2025); the
# Mz values are the Wentworth class centres of Hamilton's named classes.
_HB_REF_CW = 1510.0      # m/s   reference seawater sound speed
_HB_REF_RHOW = 1.030     # g/cm³ reference seawater density
_HB_TABLE = (
    # (Mz_phi, density_gcm3, velocity_ratio)   coarse → fine
    (0.5, 2.034, 1.201),    # coarse sand
    (2.5, 1.962, 1.152),    # fine sand
    (3.5, 1.878, 1.120),    # very fine sand
    (4.5, 1.783, 1.086),    # silty sand
    (5.0, 1.769, 1.076),    # sandy silt
    (6.0, 1.575, 1.036),    # sand-silt-clay
    (7.5, 1.489, 1.012),    # clayey silt
    (8.5, 1.480, 0.990),    # silty clay  (ratio < 1: slower than seawater)
)
_HB_PHI = np.array([r[0] for r in _HB_TABLE])
_HB_RHO = np.array([r[1] for r in _HB_TABLE])      # g/cm³ at the reference water
_HB_VRATIO = np.array([r[2] for r in _HB_TABLE])

GRAIN_SIZE_MODELS = ('hamilton', 'apl-uw')
# Valid ϕ range per model; outside it ϕ is clamped, and a UserWarning fires only
# for genuine extrapolation (≳ 1 ϕ beyond).
_MODEL_RANGE = {'hamilton': (float(_HB_PHI[0]), float(_HB_PHI[-1])),
                'apl-uw': (-1.0, 9.0)}


def _hamilton_kp(impedance: float) -> float:
    """Hamilton (1980) attenuation factor ``k_p`` vs sediment impedance.

    ``α(dB/m) = k_p · f(kHz)``; ``impedance`` is ρ·c in 10³ kg m⁻² s⁻¹. The
    piecewise fit (ESAB supplement, after Fig. 18 of Hamilton 1980) peaks
    (~0.78) in medium sand and tails to ~0.46 for coarse / ~0.07 for fine.
    """
    z = impedance
    if z < 1784.0:
        return 0.07
    if z < 2478.0:
        return 0.07 + 7.2e-5 * (z - 1784.0)
    if z < 3034.0:
        return 0.12 + 1.19e-3 * (z - 2478.0)
    if z < 3270.0:
        return 0.78 - 1.10e-3 * (z - 3034.0)
    if z < 3869.0:
        return 0.52 - 1.00e-4 * (z - 3270.0)
    return 0.46


def _hamilton_geoacoustics(phi, water_sound_speed, water_density):
    """``(cp, density, attenuation)`` from the Hamilton & Bachman ϕ-table."""
    density_ratio = float(np.interp(phi, _HB_PHI, _HB_RHO)) / _HB_REF_RHOW
    velocity_ratio = float(np.interp(phi, _HB_PHI, _HB_VRATIO))
    cp = velocity_ratio * water_sound_speed
    # k_p is calibrated on the reference-water impedance (ρ·c at Hamilton's
    # c_w/ρ_w); evaluate it there, then express α in dB/λ for the in-situ c.
    z_ref = (density_ratio * _HB_REF_RHOW) * (velocity_ratio * _HB_REF_CW)
    attenuation = _hamilton_kp(z_ref) * cp / 1000.0
    return cp, density_ratio * water_density, attenuation


# APL-UW TR 9407 (1994), §IV.A.4 "Model Input Parameters Using Grain Size"
# (valid −1 ≤ Mz ≤ 9): density ratio ρ₂/ρ₁ and sound-speed ratio c₂/c₁ as
# piecewise polynomials in Mz, plus the attenuation α₂/f (dB m⁻¹ kHz⁻¹). At
# intermediate Mz both ρ and ν run lower than Hamilton & Bachman.
def _apl_density_ratio(mz: float) -> float:
    if mz < 1.0:
        return 0.007797 * mz ** 2 - 0.17057 * mz + 2.3139
    if mz < 5.3:
        return (-0.0165406 * mz ** 3 + 0.2290201 * mz ** 2
                - 1.1069031 * mz + 3.0455)
    return -0.0012973 * mz + 1.1565


def _apl_velocity_ratio(mz: float) -> float:
    if mz < 1.0:
        return 0.002709 * mz ** 2 - 0.056452 * mz + 1.2778
    if mz < 5.3:
        return (-0.0014881 * mz ** 3 + 0.0213937 * mz ** 2
                - 0.1382798 * mz + 1.3425)
    return -0.0024324 * mz + 1.0019


def _apl_alpha_over_f(mz: float) -> float:
    """APL-UW attenuation ``α₂/f`` in dB m⁻¹ kHz⁻¹ (peaks in fine sand)."""
    if mz < 0.0:
        return 0.4556
    if mz < 2.6:
        return 0.4556 + 0.0245 * mz
    if mz < 4.5:
        return 0.1978 + 0.1245 * mz
    if mz < 6.0:
        return 8.0399 - 2.5228 * mz + 0.20098 * mz ** 2
    if mz < 9.5:
        return 0.9431 - 0.2041 * mz + 0.0117 * mz ** 2
    return 0.0601


def _apl_uw_geoacoustics(phi, water_sound_speed, water_density):
    """``(cp, density, attenuation)`` from the APL-UW TR 9407 relations."""
    cp = _apl_velocity_ratio(phi) * water_sound_speed
    # α(dB/λ) = (α₂/f)[dB/m/kHz]·c[m/s]/1000 — frequency cancels (α ∝ f).
    attenuation = _apl_alpha_over_f(phi) * cp / 1000.0
    return cp, _apl_density_ratio(phi) * water_density, attenuation


_GEOACOUSTIC_MODELS = {'hamilton': _hamilton_geoacoustics,
                       'apl-uw': _apl_uw_geoacoustics}


def grain_size_to_geoacoustics(
    grain_size_phi: float, *, model: str = 'hamilton',
    water_sound_speed: Optional[float] = None,
    water_density: Optional[float] = None,
) -> Dict[str, float]:
    """Map a mean grain size (Wentworth ϕ) to bulk geoacoustic properties.

    Returns ``{'sound_speed', 'density', 'attenuation'}`` (m/s, g/cm³,
    dB/wavelength). Sediment sound speed and density are computed as **ratios to
    the overlying seawater**, scaled by the in-situ ``water_sound_speed`` /
    ``water_density`` (so fine muds correctly come out *slower* than seawater,
    ratio < 1). Attenuation is returned in dB/wavelength, which is
    frequency-independent (``α[dB/λ] = (α/f)·c/1000``) and peaks in sand.

    Parameters
    ----------
    grain_size_phi : float
        Mean grain size on the Wentworth ϕ scale.
    model : {'hamilton', 'apl-uw'}, optional
        ``'hamilton'`` (default) — the **low-frequency** Hamilton & Bachman
        (1982) / Hamilton (1980) relations. ``'apl-uw'`` — the **high-frequency**
        APL-UW TR 9407 (1994) grain-size relations (ρ, ν polynomials + α₂/f).
    water_sound_speed, water_density : float, optional
        In-situ seawater sound speed (m/s) and density (g/cm³) the ratios are
        scaled by. ``None`` (default) uses Hamilton's reference
        (1510 m/s, 1.030 g/cm³).

    A ``UserWarning`` is emitted when ``grain_size_phi`` is well outside the
    model's valid range (ϕ is then clamped).

    Sources: APL-UW TR 9407 (1994) §IV.A.4 (``'apl-uw'``); the CC-BY ESAB
    supplement of Fonseca et al. (2025), which reproduces the Hamilton & Bachman
    (1982) table and Hamilton (1980) ``k_p`` law (``'hamilton'``).
    """
    if model not in _GEOACOUSTIC_MODELS:
        raise ConfigurationError(
            f"grain_size_to_geoacoustics: unknown model {model!r}.",
            remediation=f"Use one of {GRAIN_SIZE_MODELS}.",
        )
    if water_sound_speed is None:
        water_sound_speed = _HB_REF_CW
    if water_density is None:
        water_density = _HB_REF_RHOW
    lo, hi = _MODEL_RANGE[model]
    phi = float(np.clip(grain_size_phi, lo, hi))
    if grain_size_phi < lo - 1.0 or grain_size_phi > hi + 1.0:
        warnings.warn(
            f"grain_size_to_geoacoustics: ϕ={grain_size_phi:g} is well outside "
            f"the {model} valid range [{lo:g}, {hi:g}]; clamped.",
            UserWarning, stacklevel=2,
        )
    cp, density, attenuation = _GEOACOUSTIC_MODELS[model](
        phi, water_sound_speed, water_density)
    return {'sound_speed': cp, 'density': density, 'attenuation': attenuation}


def bottom_from_grain_size(
    grain_size_phi: float, *, roughness: float = 0.0, model: str = 'hamilton',
    water_sound_speed: Optional[float] = None,
    water_density: Optional[float] = None,
) -> BoundaryProperties:
    """Build a ``BoundaryProperties`` bottom from a mean grain size (ϕ).

    Emits a **half-space** with explicit ``sound_speed`` / ``density`` /
    ``attenuation`` (from :func:`grain_size_to_geoacoustics`) so the bottom
    works in *every* model — Bellhop, Kraken, Scooter, RAM. (The AT
    ``'grain-size'`` boundary type is Bellhop-only; Kraken/Scooter fail on it,
    so the fetched data layer never emits it.) ``grain_size_phi`` is retained
    as informational metadata.

    Parameters
    ----------
    grain_size_phi : float
        Mean grain size on the Wentworth ϕ scale.
    roughness : float, optional
        RMS interface roughness (m). Default 0.
    model, water_sound_speed, water_density
        Forwarded to :func:`grain_size_to_geoacoustics`.
    """
    g = grain_size_to_geoacoustics(
        grain_size_phi, model=model,
        water_sound_speed=water_sound_speed, water_density=water_density)
    return BoundaryProperties(
        acoustic_type='half-space',
        grain_size_phi=float(grain_size_phi),
        sound_speed=g['sound_speed'],
        density=g['density'],
        attenuation=g['attenuation'],
        roughness=roughness,
    )


def bottom_from_class(name: str, *, roughness: float = 0.0) -> BoundaryProperties:
    """Build a half-space ``BoundaryProperties`` from a named sediment class.

    ``name`` is a key of :data:`uacpy.materials.MATERIALS` (e.g. ``'sand'``,
    ``'silt'``, ``'clay'``, ``'gravel'``, ``'basalt'``).
    """
    key = str(name).lower()
    if key not in MATERIALS:
        raise ConfigurationError(
            f"bottom_from_class: unknown sediment class {name!r}.",
            remediation=f"Use one of: {', '.join(list_materials())}.",
        )
    mat = get_material(key)
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=mat['sound_speed'],
        density=mat['density'],
        attenuation=mat['attenuation'],
        shear_speed=mat.get('shear_speed', 0.0) or 0.0,
        grain_size_phi=mat.get('grain_size_phi'),
        roughness=roughness,
    )


def range_dependent_bottom_along(
    point_bottom: Callable[[float, float], BoundaryProperties],
    start, end, n_points: int, *, source_label: str,
) -> Bottom:
    """Sample a point-bottom fetcher along a geodesic → range-dependent ``Bottom``.

    ``point_bottom(lat, lon)`` returns a :class:`BoundaryProperties` or raises
    ``DataFetchError`` where the source has no coverage; such gaps hold the
    nearest covered value (forward fill, then back-fill leading gaps). Raises
    only if *no* point along the transect is covered. Shared by the EMODnet and
    local grain-size range-dependent bottom fetchers.
    """
    from uacpy.data._geo import geodesic_waypoints

    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    cp: List = []
    rho: List = []
    alpha: List = []
    cs: List = []
    alpha_s: List = []
    last = None
    for la, lo in zip(lats, lons):
        try:
            last = point_bottom(la, lo)
        except DataFetchError:
            pass
        cp.append(None if last is None else last.sound_speed)
        rho.append(None if last is None else last.density)
        alpha.append(None if last is None else last.attenuation)
        # Preserve shear so an elastic waypoint (e.g. a rock seabed) is not
        # silently flattened to a fluid half-space, matching the point fetcher.
        cs.append(None if last is None else last.shear_speed)
        alpha_s.append(None if last is None else last.shear_attenuation)

    if all(v is None for v in cp):
        raise DataFetchError(
            f"{source_label} has no seabed data anywhere along the transect.",
            remediation="Use a transect the source covers, or pass an explicit "
                        "grain size / class for a uniform bottom.",
        )
    return Bottom.from_halfspaces(
        np.asarray(ranges_m),
        sound_speed=np.asarray(_backfill_leading(cp)),
        density=np.asarray(_backfill_leading(rho)),
        attenuation=np.asarray(_backfill_leading(alpha)),
        shear_speed=np.asarray(_backfill_leading(cs)),
        shear_attenuation=np.asarray(_backfill_leading(alpha_s)),
    )


def _backfill_leading(values: List) -> List:
    """Replace leading ``None`` (before the first covered point) with it."""
    first = next(v for v in values if v is not None)
    return [first if v is None else v for v in values]
