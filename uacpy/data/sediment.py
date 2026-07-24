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

The ϕ → ``sound_speed`` / ``density`` / ``attenuation`` conversion itself lives
in :mod:`uacpy.core.sediment` (``grain_size_to_geoacoustics``, re-exported here
for backward compatibility) so it is usable without the data layer. Every
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
    geodesic_waypoints, run_representative_indices, DEFAULT_MAX_TRANSECT_POINTS,
)

__all__ = [
    'GRAIN_SIZE_MODELS',
    'grain_size_to_geoacoustics',
    'bottom_from_grain_size',
    'bottom_from_class',
    'range_dependent_bottom_along',
]


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
    only if *no* point along the transect is covered.

    With ``n_points='auto'`` the transect is probed at ``max_points`` points and
    consecutive identical seabeds are collapsed to one column each (endpoints
    anchored) — the seabed analogue of the SSP 'auto'. **Unlike the SSP/bathy
    grids, there is no analytic identity here**, so 'auto' calls ``point_bottom``
    once per probe point: cheap for the local sample DBs, but up to
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
        # Identity = the geoacoustic tuple; collapse consecutive equal seabeds.
        keys = [(p.sound_speed, p.density, p.attenuation,
                 p.shear_speed, p.shear_attenuation, p.roughness)
                for p in props]
        reps = run_representative_indices(keys)
    else:
        reps = list(range(len(props)))
    props = [props[r] for r in reps]
    rr = np.asarray(ranges_m)[reps]
    return Bottom.from_halfspaces(
        rr,
        sound_speed=np.array([p.sound_speed for p in props]),
        density=np.array([p.density for p in props]),
        attenuation=np.array([p.attenuation for p in props]),
        shear_speed=np.array([p.shear_speed for p in props]),
        shear_attenuation=np.array([p.shear_attenuation for p in props]),
        roughness=np.array([p.roughness for p in props]),
    )


def _backfill_leading(values: List) -> List:
    """Replace leading ``None`` (before the first covered point) with it."""
    first = next(v for v in values if v is not None)
    return [first if v is None else v for v in values]
