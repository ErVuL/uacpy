"""Seafloor sediment → geoacoustic ``BoundaryProperties``.

Unlike bathymetry and sound speed, there is **no reliable global, no-auth point
service** for seabed geoacoustics: usSEABED / dbSEABED are sparse survey
compilations (US waters, frequently empty at an arbitrary coordinate), so a
"fetch sediment at lat/lon" call would return nothing almost everywhere. The
durable, verifiable contribution is therefore the *conversion*: turn a mean
grain size (Wentworth ϕ) or a named sediment class into a model-ready bottom.

For *European seas* a real lat/lon sediment service does exist — see
:mod:`uacpy.data.seabed` (EMODnet WFS). Elsewhere, supply a grain size or
class here.

The ϕ → ``sound_speed`` / ``density`` / ``attenuation`` conversion itself lives
in :mod:`uacpy.core.sediment` (``grain_size_to_geoacoustics``, re-exported here
so a bottom can be built without importing the data layer). Every
builder below returns a **half-space** bottom usable by all models — a
grain-size bottom is fluid, while a named class keeps the material's shear
speed and shear attenuation, which the fluid-only models collapse for
themselves (with a ``UserWarning``); ``grain_size_phi`` is retained as
informational metadata. There is no ``'grain-size'`` boundary type.
"""

import warnings
from typing import Callable, List, Optional

import numpy as np

from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.materials import MATERIALS, list_materials
from uacpy.core.sediment import GRAIN_SIZE_MODELS, grain_size_to_geoacoustics
from uacpy.data._geo import (
    geodesic_waypoints, run_boundary_indices, DEFAULT_MAX_TRANSECT_POINTS,
    checked_max_points,
)

__all__ = [
    'GRAIN_SIZE_MODELS',
    'grain_size_to_geoacoustics',
    'bottom_from_grain_size',
    'bottom_from_class',
    'range_dependent_bottom_along',
    'water_sound_speed_at',
]


def water_sound_speed_at(water_sound_speed, lat: float, lon: float):
    """Resolve a ``water_sound_speed`` argument at one transect waypoint.

    A float is used as given; a ``(lat, lon) -> m/s`` callable is evaluated, so
    a range-dependent water column scales every seabed column to the sound speed
    over *its* own seafloor. ``None`` keeps the Hamilton reference.
    """
    return (water_sound_speed(lat, lon) if callable(water_sound_speed)
            else water_sound_speed)


def bottom_from_grain_size(
    grain_size_phi: float, *, roughness: float = 0.0, model: str = 'hamilton',
    water_sound_speed: Optional[float] = None,
    water_density: Optional[float] = None,
) -> BoundaryProperties:
    """Build a half-space ``BoundaryProperties`` bottom from a mean grain size (ϕ).

    Thin wrapper over :meth:`BoundaryProperties.from_grain_size` — emits a
    half-space with explicit ``sound_speed`` / ``density`` / ``attenuation`` (via
    :func:`uacpy.core.sediment.grain_size_to_geoacoustics`) so the bottom works
    in *every* model. ``grain_size_phi`` is retained as informational metadata.

    Parameters
    ----------
    grain_size_phi : float
        Mean grain size on the Wentworth ϕ scale.
    roughness : float, optional
        RMS interface roughness (m). Default 0.
    model, water_sound_speed, water_density
        Forwarded to :func:`grain_size_to_geoacoustics`.
    """
    return BoundaryProperties.from_grain_size(
        grain_size_phi, model=model, roughness=roughness,
        water_sound_speed=water_sound_speed, water_density=water_density)


def bottom_from_class(name: str, *, roughness: float = 0.0,
                      elastic: bool = True) -> BoundaryProperties:
    """Build a half-space ``BoundaryProperties`` from a named sediment class.

    ``name`` is a key of :data:`uacpy.materials.MATERIALS` (e.g. ``'sand'``,
    ``'silt'``, ``'clay'``, ``'gravel'``, ``'basalt'``).

    Thin wrapper over :meth:`BoundaryProperties.from_preset`, which copies the
    preset's shear *speed* and shear *attenuation* together — they are one
    property of the material, and an elastic half-space carrying a shear speed
    but no shear loss under-predicts bottom loss in every elastic solver.
    ``elastic=True`` (the default) keeps the pair: the classes reached
    automatically are hard substrata (EMODnet Folk class 5 and DECK41 ``'rock'``
    both route to ``'limestone'``), where shear support *is* the seabed's
    acoustic identity. The fluid-only models collapse the shear themselves,
    with a ``UserWarning``; pass ``elastic=False`` to drop it here instead.
    """
    key = str(name).lower()
    if key not in MATERIALS:
        raise ConfigurationError(
            f"bottom_from_class: unknown sediment class {name!r}.",
            remediation=f"Use one of: {', '.join(list_materials())}.",
        )
    return BoundaryProperties.from_preset(key, elastic=elastic,
                                          roughness=roughness)


def range_dependent_bottom_along(
    point_bottom: Callable[[float, float], BoundaryProperties],
    start, end, n_points='auto', *, source_label: str,
    max_points=None,
) -> Bottom:
    """Sample a point-bottom fetcher along a geodesic → range-dependent ``Bottom``.

    ``point_bottom(lat, lon)`` returns a :class:`BoundaryProperties` or raises
    ``DataFetchError`` where the source has no coverage; such gaps take the
    seabed of the **nearest covered waypoint by along-track distance**, and the
    call warns with how many were filled and the worst distance. Raises only if
    *no* point along the transect is covered. Each sampled column's
    ``data_sources`` provenance and ``grain_size_phi`` are carried onto the
    corresponding column of the returned ``Bottom``, so a filled gap carries
    the record of the sample that filled it — which names where that sample was
    *fetched*, not the range it was applied at.

    With ``n_points='auto'`` the transect is probed at ``max_points`` points
    and each run of identical seabeds collapses to the probe columns
    bracketing its edges (endpoints anchored) — the ``Bottom`` reads
    nearest-node, so every reconstructed seabed transition lands within one
    probe step of the boundary the probe observed. Identity is the
    sediment's own (see :func:`_seabed_identity`), not the water-scaled
    geoacoustics. **Unlike the SSP/bathy grids, there is no analytic identity
    here**, so 'auto' calls ``point_bottom`` once per probe point: cheap for
    the local sample DBs, but up to
    ``max_points`` network calls for the live EMODnet WFS — hence bottom fetches
    default to an explicit small ``n_points`` and opt into 'auto'. An explicit
    integer samples exactly that many points (capped at ``max_points``).
    """
    if max_points is None:
        max_points = DEFAULT_MAX_TRANSECT_POINTS
    max_points = checked_max_points(max_points, source_label)
    probe_n = (max_points if n_points == 'auto'
               else max(2, min(int(n_points), max_points)))
    lats, lons, ranges_m = geodesic_waypoints(start, end, probe_n)
    props: List = []
    for la, lo in zip(lats, lons):
        try:
            props.append(point_bottom(la, lo))
        except DataFetchError:
            props.append(None)      # uncovered; filled from the nearest below

    if all(p is None for p in props):
        raise DataFetchError(
            f"{source_label} has no seabed data anywhere along the transect.",
            remediation="Use a transect the source covers, or pass an explicit "
                        "grain size / class for a uniform bottom.",
        )
    props, filled = _fill_gaps_from_nearest(props, np.asarray(ranges_m))
    if filled:
        # Every other substitution in this layer says so — the WOA23 dry-cell
        # hop, the NSIDC unobserved-cell hop, the SSP seafloor extrapolation —
        # and this one moves the seabed, which sets the reflection coefficient
        # over whatever fraction of the track it covers. Measured on a
        # 1579 km NE Atlantic transect where EMODnet covers the first 324 km
        # (21%): the remaining 1255 km all took one polygon's class, giving
        # rho*c 2.883e6 against 2.256e6 kg m^-2 s^-1 (+27.8%) for the seabed a
        # point fetch of the far end returns from the next source in the chain.
        n_filled, worst_km = filled
        warnings.warn(
            f"{source_label}: {n_filled} of {len(props)} transect waypoints "
            f"have no seabed data and were filled from the nearest covered "
            f"waypoint, up to {worst_km:.0f} km away. The seabed over those "
            f"stretches is the covered sample's, not a measurement at those "
            f"ranges, and the per-column provenance names where each sample "
            f"was fetched rather than the range it was applied at. Narrow the "
            f"transect to the covered region, or name a source that spans it.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if n_points == 'auto':
        reps = run_boundary_indices([_seabed_identity(p) for p in props])
    else:
        reps = list(range(len(props)))
    props = [props[r] for r in reps]
    rr = np.asarray(ranges_m)[reps]
    bottom = Bottom.from_halfspaces(
        rr,
        sound_speed=np.array([p.sound_speed for p in props]),
        density=np.array([p.density for p in props]),
        attenuation=np.array([p.attenuation for p in props]),
        shear_speed=np.array([p.shear_speed for p in props]),
        shear_attenuation=np.array([p.shear_attenuation for p in props]),
        roughness=np.array([p.roughness for p in props]),
    )
    # ``from_halfspaces`` emits fresh half-spaces from the geoacoustic arrays
    # alone; copy each sampled column's provenance and grain size onto its
    # rebuilt column so the transect reports the same ``data_sources``
    # (dataset + actual sample coordinates) and the same informational ϕ a
    # single-point fetch does. ϕ takes no parameter on ``from_halfspaces``
    # because it does not define geoacoustics on its own — it is set after the
    # conversion has, exactly as ``from_grain_size`` retains it.
    for column, source_props in zip(bottom.columns, props):
        column.halfspace.data_sources = tuple(
            getattr(source_props, 'data_sources', ()) or ())
        column.halfspace.grain_size_phi = source_props.grain_size_phi
    return bottom


def _seabed_identity(props: BoundaryProperties):
    """Collapse key for the ``'auto'`` reduction: the seabed's own identity.

    Where the source reports a grain size, ϕ *is* that identity — the sound
    speed, density and attenuation it yields are ratios against the overlying
    seawater, so they track the water column and vary continuously along a
    transect over a single uniform sediment. Keying on them would make every
    probe point distinct and collapse nothing. Sources that report no grain
    size (absolute crustal properties) key on the geoacoustic tuple.
    """
    if props.grain_size_phi is not None:
        return (props.grain_size_phi, props.shear_speed,
                props.shear_attenuation, props.roughness)
    return (props.sound_speed, props.density, props.attenuation,
            props.shear_speed, props.shear_attenuation, props.roughness)


def _fill_gaps_from_nearest(values: List, ranges_m):
    """Fill each uncovered waypoint from the NEAREST covered one.

    Returns ``(filled_values, report)`` where ``report`` is ``None`` when
    nothing needed filling and ``(n_filled, worst_km)`` otherwise.

    Nearest by along-track distance, not the last covered value in transect
    order: forward-filling gives a gap bracketed by coverage on both sides the
    *earlier* sample even when the later one is far closer, which is a
    direction-dependent answer to a question that has none. A leading gap is
    covered by the same rule, so no separate leading-only backfill is needed.
    """
    covered = [i for i, v in enumerate(values) if v is not None]
    if len(covered) == len(values):
        return list(values), None
    r = np.asarray(ranges_m, dtype=float)
    r_cov = r[covered]
    out = list(values)
    worst = 0.0
    for i, v in enumerate(values):
        if v is not None:
            continue
        j = int(np.argmin(np.abs(r_cov - r[i])))
        out[i] = values[covered[j]]
        worst = max(worst, abs(float(r_cov[j] - r[i])))
    return out, (len(values) - len(covered), worst / 1000.0)
