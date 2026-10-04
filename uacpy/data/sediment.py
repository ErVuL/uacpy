"""Seafloor sediment → geoacoustic ``BoundaryProperties``.

Unlike bathymetry and sound speed, there is **no reliable global, no-auth point
service** for seabed geoacoustics: usSEABED / dbSEABED are sparse survey
compilations (US waters, frequently empty at an arbitrary coordinate), so a
"fetch sediment at lat/lon" call would return nothing almost everywhere. The
durable, verifiable contribution is therefore the *conversion*: turn a mean
grain size (Wentworth ϕ) or a named sediment class into a model-ready bottom.

The lat/lon seabed fetchers built on this conversion live in their own
modules: EMODnet substrate for European seas (:mod:`uacpy.data.seabed` live,
:mod:`uacpy.data.emodnet_local` cached), the NCEI grain-size / DECK41 samples
(:mod:`uacpy.data.sediment_db`), AusSeabed MARS (:mod:`uacpy.data.mars`), the
Diesing deep-sea map (:mod:`uacpy.data.diesing_local`), the Graw density grid
(:mod:`uacpy.data.graw_local`) and the global pelagic rule
(:mod:`uacpy.data.pelagic`); ``fetch_environment(bottom_sources=...)`` chains
them. Where none covers a site, supply a grain size or class here.

The ϕ → ``sound_speed`` / ``density`` / ``attenuation`` conversion itself lives
in :mod:`uacpy.core.sediment` (``grain_size_to_geoacoustics`` and its inverse
``grain_size_from_density``), so a bottom can be built without importing the data
layer. Every
builder below returns a **half-space** bottom usable by all models — a
grain-size bottom is fluid, while a named class keeps the material's shear
speed and shear attenuation, which the fluid-only models collapse for
themselves (with a ``FallbackWarning``); ``grain_size_phi`` is retained as
informational metadata. There is no ``'grain-size'`` boundary type.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import numpy as np

from uacpy.core._export import ExportRecord, require_extra
from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.core.exceptions import (
    ConfigurationError, DataFetchError, FallbackWarning,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.materials import MATERIALS, list_materials
from uacpy.core.sediment import DEFAULT_GRAIN_SIZE_MODEL, GRAIN_SIZE_MODELS
from uacpy.data._cache import UnreadableCacheError
from uacpy.data.sources import DataProvenance
from uacpy.core.geo import geodesic_waypoints
from uacpy.data._geo import (
    run_boundary_indices, DEFAULT_MAX_TRANSECT_POINTS, checked_max_points,
    checked_n_points, capped_n_points,
)

__all__ = [
    'SeabedSample',
    'samples_table',
    'GRAIN_SIZE_MODELS',
    'bottom_from_grain_size',
    'bottom_from_class',
    'range_dependent_bottom_along',
    'transect_fetcher',
    'water_sound_speed_at',
]


#: The Folk classification schemes a :class:`SeabedSample` names: EMODnet's
#: five-class code (``folk_5cl``, 1-5) and Folk's full class (``'sG'`` ...).
FOLK_CLASS_SCHEMES = ('folk5', 'folk')


@dataclass(frozen=True, eq=False)
class SeabedSample(ExportRecord):
    """One seabed record at a point, as the raw seabed fetchers return it
    (:func:`~uacpy.data.fetch_emodnet_substrate`,
    :func:`~uacpy.data.fetch_emodnet_substrate_local`,
    :func:`~uacpy.data.fetch_sediment_sample`,
    :func:`~uacpy.data.fetch_seafloor_lithology`,
    :func:`~uacpy.data.fetch_mars_sediment`). Each source fills what it
    holds; the rest is ``None``.

    Attributes
    ----------
    grain_size_phi : float or None
        Mean grain size (ϕ).
    material : str or None
        A material or lithology word: a DECK41 preset (``'limestone'``,
        ``'gravel'``) or a Diesing lithology.
    folk_class : int or str or None
        The Folk class, in the scheme ``folk_class_scheme`` names: an int
        1-5 for EMODnet's ``folk_5cl``, a class string (``'sG'``) for Folk.
    folk_class_scheme : {'folk5', 'folk'} or None
        Required whenever ``folk_class`` is set.
    sample_point : tuple or None
        The sample's own ``(lat, lon)``, for a point-sample database.
    distance_km : float or None
        Its distance from the requested point.
    provenance : DataProvenance
        The dataset the record came from.
    details : mapping
        What one source adds (EMODnet's ``folk_5cl_txt`` and
        ``original_grain_size``; MARS's ``via``, the conversion that gave ϕ).
    """

    grain_size_phi: Optional[float]
    material: Optional[str]
    folk_class: Optional[Union[int, str]]
    folk_class_scheme: Optional[str]
    sample_point: Optional[Tuple[float, float]]
    distance_km: Optional[float]
    provenance: DataProvenance
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.folk_class is not None and \
                self.folk_class_scheme not in FOLK_CLASS_SCHEMES:
            raise ConfigurationError(
                f"SeabedSample: folk_class={self.folk_class!r} needs "
                f"folk_class_scheme in {FOLK_CLASS_SCHEMES}; got "
                f"{self.folk_class_scheme!r}.")
        object.__setattr__(self, 'details', dict(self.details))
        super().__post_init__()


def samples_table(*samples: SeabedSample):
    """The samples as a ``pandas.DataFrame`` (optional extra
    ``uacpy[xarray]``), one row each: ``source`` (the provenance's catalogue
    id), ``grain_size_phi``, ``material``, ``folk_class``,
    ``folk_class_scheme``, ``sample_lat``, ``sample_lon`` and
    ``distance_km``."""
    pandas = require_extra('pandas', 'samples_table')
    return pandas.DataFrame([{
        'source': s.provenance.source.id,
        'grain_size_phi': s.grain_size_phi, 'material': s.material,
        'folk_class': s.folk_class,
        'folk_class_scheme': s.folk_class_scheme,
        'sample_lat': None if s.sample_point is None else s.sample_point[0],
        'sample_lon': None if s.sample_point is None else s.sample_point[1],
        'distance_km': s.distance_km} for s in samples])


def water_sound_speed_at(water_sound_speed, lat: float, lon: float):
    """Resolve a ``water_sound_speed`` argument at one transect waypoint.

    A float is used as given; a ``(lat, lon) -> m/s`` callable is evaluated, so
    a range-dependent water column scales every seabed column to the sound speed
    over *its* own seafloor. ``None`` keeps the Hamilton reference.
    """
    return (water_sound_speed(lat, lon) if callable(water_sound_speed)
            else water_sound_speed)


def bottom_from_grain_size(
    grain_size_phi: float, *, roughness: float = 0.0, model: str = DEFAULT_GRAIN_SIZE_MODEL,
    hamilton_fit: Optional[str] = None,
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
    model, hamilton_fit, water_sound_speed, water_density
        Forwarded to
        :func:`uacpy.core.sediment.grain_size_to_geoacoustics`. ``hamilton_fit``
        picks which of Hamilton & Bachman's three fits ``'hamilton'`` uses;
        ``None`` takes the continental-terrace default, and nothing infers an
        abyssal site for the caller.
    """
    from uacpy.core.sediment import canonical_grain_size_selection
    model, hamilton_fit = canonical_grain_size_selection(
        model, hamilton_fit, who='bottom_from_grain_size')
    return BoundaryProperties.from_grain_size(
        grain_size_phi, model=model, hamilton_fit=hamilton_fit,
        roughness=roughness,
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
    with a ``FallbackWarning``; pass ``elastic=False`` to drop it here instead.

    Parameters
    ----------
    name : str
        A preset of :data:`uacpy.materials.MATERIALS`.
    roughness : float, optional
        RMS interface roughness (m). Default 0.
    elastic : bool, optional
        Keep the preset's shear speed and attenuation. Default True.
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
    call warns with how many were filled and the worst distance. Raises if
    *no* point along the transect is covered. An
    :class:`~uacpy.data._cache.UnreadableCacheError` is raised as itself: a
    cached file that cannot be read says nothing about the waypoint, and its
    remediation (re-install the dataset) is the one that applies. Each
    sampled column's ``data_sources`` provenance and ``grain_size_phi`` are carried onto the
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
    n_points = checked_n_points(n_points, source_label, allow_auto=True)
    probe_n = (max_points if n_points == 'auto'
               else capped_n_points(n_points, max_points, source_label))
    lats, lons, ranges_m = geodesic_waypoints(start, end, probe_n)
    props: List = []
    for la, lo in zip(lats, lons):
        try:
            props.append(point_bottom(la, lo))
        except UnreadableCacheError:
            raise                   # a broken cache, not a coverage gap
        except DataFetchError:
            props.append(None)      # uncovered; filled from the nearest below

    if all(p is None for p in props):
        raise DataFetchError(
            f"{source_label} has no seabed data anywhere along the transect.",
            remediation="Use a transect the source covers, or pass an explicit "
                        "grain size / class for a uniform bottom.",
        )
    props, filled = _fill_gaps_from_nearest(props, np.asarray(ranges_m))
    _warn_filled_gaps(source_label, filled, len(props))
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
        # ``BoundaryProperties.data_sources`` is a declared field that
        # ``__post_init__`` coerces to a tuple (None -> ()), so it always
        # exists here — as the bare ``grain_size_phi`` read below assumes.
        column.halfspace.data_sources = tuple(source_props.data_sources)
        column.halfspace.grain_size_phi = source_props.grain_size_phi
    return bottom


def transect_fetcher(point_fn: Callable[..., BoundaryProperties], label: str,
                     *, preload: Optional[Callable[[], object]] = None):
    """The range-dependent bottom fetcher of a half-space point fetcher.

    ``point_fn(point, **options)`` is sampled along the great circle by
    :func:`range_dependent_bottom_along`, under ``label`` in its coverage
    messages. The returned ``transect(start, end, *, n_points=6,
    max_points=None, **options)`` passes ``options`` to ``point_fn`` at every
    waypoint; ``water_sound_speed`` and ``depth`` also take a ``(lat, lon) ->
    value`` callable, evaluated per waypoint, so each column scales to the
    water over its own seafloor. ``preload`` runs once before any waypoint:
    a source whose files are read up front reports an unreadable cache as
    itself rather than as a gap at every waypoint. The point fetcher is
    ``transect.point_fetcher``, whose signature names the options.
    """
    def transect(start, end, *, n_points=6, max_points=None, **options):
        if preload is not None:
            preload()

        def at(lat, lon):
            kwargs = dict(options)
            if 'water_sound_speed' in kwargs:
                kwargs['water_sound_speed'] = water_sound_speed_at(
                    kwargs['water_sound_speed'], lat, lon)
            if callable(kwargs.get('depth')):
                kwargs['depth'] = kwargs['depth'](lat, lon)
            return point_fn((lat, lon), **kwargs)

        return range_dependent_bottom_along(
            at, start, end, n_points, source_label=label,
            max_points=max_points)

    transect.point_fetcher = point_fn
    return transect


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


def _warn_filled_gaps(source_label, filled, n_total):
    """Warn that ``filled = (n_filled, worst_km)`` of ``n_total`` transect
    waypoints took the seabed of their nearest covered neighbour; silent when
    ``filled`` is ``None`` (the report of :func:`_fill_gaps_from_nearest`)."""
    if not filled:
        return
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
        f"{source_label}: {n_filled} of {n_total} transect waypoints "
        f"have no seabed data and were filled from the nearest covered "
        f"waypoint, up to {worst_km:.0f} km away. The seabed over those "
        f"stretches is the covered sample's, not a measurement at those "
        f"ranges, and the per-column provenance names where each sample "
        f"was fetched rather than the range it was applied at. Narrow the "
        f"transect to the covered region, or name a source that spans it.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)


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
