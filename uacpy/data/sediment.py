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
builder below returns a **half-space** bottom usable by all models;
``grain_size_phi`` is retained as informational metadata. There is no
``'grain-size'`` boundary type.
"""

from typing import Callable, List, Optional

import numpy as np

from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core.materials import MATERIALS, get_material, list_materials
from uacpy.core.sediment import GRAIN_SIZE_MODELS, grain_size_to_geoacoustics
from uacpy.data._geo import (
    geodesic_waypoints, run_boundary_indices, DEFAULT_MAX_TRANSECT_POINTS,
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
    start, end, n_points='auto', *, source_label: str,
    max_points=None,
) -> Bottom:
    """Sample a point-bottom fetcher along a geodesic → range-dependent ``Bottom``.

    ``point_bottom(lat, lon)`` returns a :class:`BoundaryProperties` or raises
    ``DataFetchError`` where the source has no coverage; such gaps hold the
    nearest covered value (forward fill, then back-fill leading gaps). Raises
    only if *no* point along the transect is covered. Each sampled column's
    ``data_sources`` provenance is carried onto the corresponding column of
    the returned ``Bottom`` (a forward-filled gap carries the record of the
    sample that filled it).

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
    probe_n = (int(max_points) if n_points == 'auto'
               else max(2, min(int(n_points), int(max_points))))
    lats, lons, ranges_m = geodesic_waypoints(start, end, probe_n)
    props: List = []
    last = None
    for la, lo in zip(lats, lons):
        try:
            last = point_bottom(la, lo)
        except DataFetchError:
            pass
        props.append(last)          # None until the first covered point

    if all(p is None for p in props):
        raise DataFetchError(
            f"{source_label} has no seabed data anywhere along the transect.",
            remediation="Use a transect the source covers, or pass an explicit "
                        "grain size / class for a uniform bottom.",
        )
    props = _backfill_leading(props)
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
    # ``from_halfspaces`` emits fresh half-spaces; copy each sampled column's
    # provenance onto its rebuilt column so the transect reports the same
    # ``data_sources`` (dataset + actual sample coordinates) a single-point
    # fetch does.
    for column, source_props in zip(bottom.columns, props):
        column.halfspace.data_sources = tuple(
            getattr(source_props, 'data_sources', ()) or ())
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


def _backfill_leading(values: List) -> List:
    """Replace leading ``None`` (before the first covered point) with it."""
    first = next(v for v in values if v is not None)
    return [first if v is None else v for v in values]
