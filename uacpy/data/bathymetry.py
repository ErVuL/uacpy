"""Bathymetry fetch — GPS coordinates → ``Environment``-ready depth.

Turns geographic coordinates into a water depth (m, positive down) ready to hand to
``Environment(bathymetry=...)`` — either a single depth for one point, or a
range-dependent ``(N, 2)`` ``[range_m, depth_m]`` transect sampled along the
great-circle path between two points.

``source='gebco'`` (the default) is cache-first, as ``fetch_environment`` is:
it samples the install-time GEBCO 2025 grid (``install.sh --data gebco``,
offline and unthrottled) and falls back to the GEBCO_2020 global grid (~450 m
resolution) served as JSON by the public OpenTopoData API
(https://www.opentopodata.org/datasets/gebco2020/) when the grid is not
installed. No API key is needed; the public host is rate-limited (≤100
locations per request, ≤1 request/s, ≤1000 requests/day). Point a
``base_url`` at a self-hosted OpenTopoData instance to lift those limits. The
other sources are ``'local'`` (the installed grid only), ``'gmrt'`` and
``'emodnet_dtm'`` (live, higher-resolution, regional coverage) — the same ids
``fetch_environment`` and every provenance record use.

Bathymetry is static in time, so these fetches take coordinates only — the
``date`` axis enters the data layer at the sound-speed stage, not here.
"""

import importlib
import json
import threading
import time
import urllib.parse
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from uacpy.core.geo import EARTH_RADIUS_M
from uacpy.core.exceptions import (
    ConfigurationError, DataFetchError, FallbackWarning, IOWarning,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.geo import (
    Coordinate, as_coordinate, normalize_lon, central_angle,
    geodesic_waypoints, EARTH_RADIUS_KM,
)
from uacpy.data._geo import (
    lon_linspace, DEFAULT_MAX_TRANSECT_POINTS, checked_max_points,
    checked_n_points, capped_n_points, require_source,
)
from uacpy.data import _cache
from uacpy.data._chain import SourceChain, SourceProvider, first_answer
from uacpy.data._http import http_get
from uacpy._log import log_message
from uacpy.core._export import ExportRecord
from uacpy.data.sources import SOURCES, DataProvenance

__all__ = ['fetch_bathy', 'fetch_bathy_transect', 'bathy_transect_plan',
           'fetch_bathy_grid', 'transect_length', 'transect_waypoints']

DEFAULT_BASE_URL = 'https://api.opentopodata.org/v1'
DEFAULT_DATASET = 'gebco2020'
#: GEBCO_2020 native grid spacing (~15 arc-seconds ≈ 0.45 km). Used only to
#: size the ``n_points='auto'`` transect — bathymetry is a *continuous*
#: (bilinearly served) field with no duplicate samples to collapse, so 'auto'
#: targets native resolution, bounded by ``max_points``.
GEBCO_NATIVE_KM = 0.45
MAX_LOCATIONS_PER_REQUEST = 100  # OpenTopoData public-host limit
MAX_GRID_REQUESTS = 100          # safety cap for fetch_bathy_grid (≤10 000 points)
#: Minimum spacing (s) between consecutive OpenTopoData calls, honouring the
#: public host's documented ≤1 request/s limit so a multi-chunk grid/transect
#: stays under the rate cap instead of bursting and tripping a 429 / IP block.
#: Only applied to the public ``DEFAULT_BASE_URL``; a self-hosted ``base_url``
#: lifts the limit (set to 0.0 there). Tests set it to 0.0 to run instantly.
OPENTOPODATA_MIN_INTERVAL_S = 1.0

# Public source names: the catalogue ids ``fetch_environment`` and every
# provenance record use, plus ``'local'`` (cached data only, no network). Each
# resolves through :data:`_BATHY_CHAIN` to the backends tried in order —
# cached first — which are what the private ``_fetch_*_backend`` functions
# take.
BATHY_SOURCES = ('gebco', 'gmrt', 'emodnet_dtm', 'local')
_BATHY_BACKENDS = ('api', 'gmrt', 'emodnet', 'local')
#: Backend token → the module serving it (``point_depth``, ``depths_along``,
#: ``region_grid``) and the grid's log label. ``'api'`` (OpenTopoData) is
#: served in this module.
_BACKEND_MODULES = {
    'local': ('gebco_local', 'GEBCO grid (local)'),
    'gmrt': ('gmrt_live', 'GMRT grid'),
    'emodnet': ('emodnet_bathy_live', 'EMODnet DTM grid'),
}

#: Live backends whose transect is one HTTP request per waypoint (GMRT
#: PointServer, EMODnet griddap). OpenTopoData batches 100 points per request.
_PER_POINT_BACKENDS = ('gmrt', 'emodnet')
#: Waypoint count above which a per-point live transect warns: past it the
#: fetch is a burst of that many sequential requests to a public service.
#: A judgement about politeness and wait time, not a service limit.
LIVE_POINT_REQUEST_WARN = 100

_last_request_monotonic = 0.0    # time.monotonic() of the last public-host call
_rate_limit_lock = threading.Lock()   # serializes the read-sleep-stamp above


def _check_source(source):
    require_source(source, _BATHY_BACKENDS, 'bathymetry backend',
                   "Use 'api', 'gmrt', 'emodnet' or 'local'.")


def _check_public_source(source):
    require_source(source, BATHY_SOURCES, 'bathymetry source',
                   "Use 'gebco' (the installed GEBCO grid, else OpenTopoData "
                   "online), 'gmrt' (GMRT multibeam, higher-res live), "
                   "'emodnet_dtm' (EMODnet DTM, ~115 m, European seas + "
                   "Caribbean), or 'local' (the installed GEBCO grid only).")


def _cache_first(source, call, *, who, chained=True):
    """``call(backend)`` for each backend ``source`` names, cached first,
    raising the most substantive error — the rule ``fetch_environment``
    resolves by. ``chained`` accepts a chain spec (``'auto'`` or a sequence
    of ids) besides one id; ``who`` is named when the spec is refused.

    The installed grid falls through to the live service only when it is
    absent or unreadable (:func:`uacpy.data._cache.is_cache_miss`); a grid that
    answered "land" or "no value" ends the chain, since the service holds the
    same dataset."""
    if not chained or (isinstance(source, str) and source != 'auto'):
        _check_public_source(source)
    sources, cache_only = _BATHY_CHAIN.resolve(source)
    answer, _attempt = first_answer(
        _BATHY_CHAIN.attempts(sources, cache_only=cache_only, who=who,
                              keyword='source='),
        lambda _source, backend: call(backend))
    return answer


def _backend_module(backend):
    """The module serving a non-``'api'`` backend token."""
    return importlib.import_module(
        f'uacpy.data.{_BACKEND_MODULES[backend][0]}')


def _network_kwargs(backend, timeout, verbose):
    """The ``timeout``/``verbose`` a backend module's fetchers take: the
    installed grid is read offline and takes neither."""
    return {} if backend == 'local' else {'timeout': timeout,
                                          'verbose': verbose}


def _fetch_bathy_backend(
    point: Coordinate,
    *,
    backend: str = 'api',
    dataset: str = DEFAULT_DATASET,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    verbose: Union[bool, str] = False,
) -> float:
    """Backend of :func:`fetch_bathy`: ``backend`` is a backend token (``'api'``, ``'gmrt'``,
    ``'emodnet'``, ``'local'``); ``fetch_environment`` resolves to these."""
    _check_source(backend)
    if backend != 'api':
        return _backend_module(backend).point_depth(
            point, **_network_kwargs(backend, timeout, verbose))
    depths = _fetch_depths(
        [as_coordinate(point)], dataset=dataset, base_url=base_url,
        timeout=timeout, verbose=verbose,
    )
    return float(depths[0])


def fetch_bathy(
    point: Coordinate,
    *,
    source: Union[str, Sequence[str]] = 'gebco',
    dataset: str = DEFAULT_DATASET,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    verbose: Union[bool, str] = False,
) -> float:
    """Water depth (m, positive down) at a single point.

    Parameters
    ----------
    point : (lat, lon)
        Latitude and longitude in decimal degrees (WGS84).
    source : str or sequence of str, optional
        The catalogue ids ``fetch_environment`` uses. ``'gebco'`` (default)
        samples the installed GEBCO 2025 grid (``install.sh --data gebco``)
        and falls back to OpenTopoData's GEBCO online when it is not
        installed — cache-first, like ``fetch_environment``; ``'gmrt'``
        queries the GMRT multibeam synthesis (higher resolution where
        surveyed, CC-BY, live); ``'emodnet_dtm'`` the EMODnet Bathymetry DTM
        (~115 m, European seas + Caribbean only, CC-BY, live); ``'local'``
        the installed GEBCO grid only (offline, no rate limit). ``'auto'``
        (EMODnet DTM, else GMRT, else GEBCO) or a sequence of ids is a
        chain: the first source that answers wins, each cached first — the
        rule ``fetch_environment(bathymetry_sources=)`` resolves by.
    dataset : str, optional
        OpenTopoData dataset name. Default ``'gebco2020'`` (``'api'`` only).
    base_url : str, optional
        OpenTopoData service root (override for a self-hosted instance).
    timeout : float, optional
        Per-request network timeout in seconds.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.

    Returns
    -------
    float
        Depth in metres, suitable for ``Environment(bathymetry=depth)``.

    Raises
    ------
    DataFetchError
        The service is unreachable/erroring, or the point is on land
        (non-negative elevation, i.e. no water column).

    The value is raw and carries no provenance;
    :func:`uacpy.data.fetch_environment` (its ``Bathymetry``) returns the
    carrier that records it in ``.data_sources``.
    """
    return _cache_first(source, lambda backend: _fetch_bathy_backend(
        point, backend=backend, dataset=dataset, base_url=base_url,
        timeout=timeout, verbose=verbose), who='fetch_bathy')


def _fetch_bathy_transect_backend(
    start: Coordinate,
    end: Coordinate,
    *,
    n_points: Union[int, str] = 50,
    max_points: int = DEFAULT_MAX_TRANSECT_POINTS,
    backend: str = 'api',
    dataset: str = DEFAULT_DATASET,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    verbose: Union[bool, str] = False,
) -> np.ndarray:
    """Backend of :func:`fetch_bathy_transect`: ``backend`` is a backend token (``'api'``, ``'gmrt'``,
    ``'emodnet'``, ``'local'``); ``fetch_environment`` resolves to these."""
    _check_source(backend)
    plan = bathy_transect_plan(start, end, n_points=n_points,
                               max_points=max_points)
    n = plan['n_points']
    lats, lons, ranges_m = plan['lats'], plan['lons'], plan['ranges_m']
    length_km = ranges_m[-1] / 1000.0
    if n_points == 'auto':
        if plan['native_points'] > max_points:
            warnings.warn(
                f"fetch_bathy_transect: native GEBCO resolution over "
                f"{length_km:.0f} km needs ~{plan['native_points']} points; "
                f"capped to max_points={max_points} "
                f"(~{length_km / (max_points - 1):.1f} km spacing). Raise "
                f"max_points, or use GMRT / a self-hosted OpenTopoData for "
                f"finer sampling.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)

    log_message(
        'bathymetry', f"sampling {n} depths (source={backend!r}) along "
        f"{ranges_m[-1] / 1000:.1f} km transect", verbose=verbose,
    )
    if backend in _PER_POINT_BACKENDS and n > LIVE_POINT_REQUEST_WARN:
        warnings.warn(
            f"fetch_bathy_transect: source {backend!r} serves one point per "
            f"request, so {n} waypoints are {n} sequential requests to a "
            f"public service (EMODnet tries a second tile where the first "
            f"has no cell). Pass a smaller n_points, or use the installed "
            f"GEBCO grid (backend='local').",
            IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if backend != 'api':
        depths = _backend_module(backend).depths_along(
            lats, lons, **_network_kwargs(backend, timeout, verbose))
    else:
        depths = _fetch_depths(
            list(zip(lats, lons)), dataset=dataset, base_url=base_url,
            timeout=timeout, verbose=verbose,
        )
    return np.column_stack([ranges_m, depths])


def fetch_bathy_transect(
    start: Coordinate,
    end: Coordinate,
    *,
    n_points: Union[int, str] = 50,
    max_points: int = DEFAULT_MAX_TRANSECT_POINTS,
    source: Union[str, Sequence[str]] = 'gebco',
    dataset: str = DEFAULT_DATASET,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    verbose: Union[bool, str] = False,
) -> np.ndarray:
    """Range-dependent bathymetry along the great-circle ``start``→``end``.

    Samples evenly spaced (in distance) along the geodesic and returns an
    ``(n, 2)`` array of ``[range_m, depth_m]`` with ``range`` measured from
    ``start`` — exactly the shape consumed by ``Environment(bathymetry=...)``
    for range-dependent runs. :func:`bathy_transect_plan` resolves ``n``.

    Parameters
    ----------
    start, end : (lat, lon)
        Endpoint coordinates in decimal degrees (WGS84).
    n_points : int or 'auto', optional
        Number of samples (≥2). Default 50. ``'auto'`` targets GEBCO native
        resolution (see :func:`bathy_transect_plan`).
    max_points : int, optional
        Ceiling on the sample count; a larger ``n_points`` (or an ``'auto'``
        native count above it) is capped to this, with a ``FallbackWarning``.
    source, dataset, base_url, timeout, verbose
        See :func:`fetch_bathy`.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 2)``: column 0 range (m), column 1 depth (m), where ``n``
        is the resolved sample count.

    Raises
    ------
    ConfigurationError
        ``n_points < 2`` or the two endpoints coincide.
    DataFetchError
        The service fails, or any sampled point falls on land.

    The value is raw and carries no provenance;
    :func:`uacpy.data.fetch_environment` (its ``Bathymetry``) returns the
    carrier that records it in ``.data_sources``.
    """
    return _cache_first(source, lambda backend: _fetch_bathy_transect_backend(
        start, end, backend=backend, n_points=n_points, max_points=max_points,
        dataset=dataset, base_url=base_url, timeout=timeout,
        verbose=verbose), who='fetch_bathy_transect')


def _point_depth(backend, point, *, timeout, verbose, **_request):
    """A :data:`_BATHY_CHAIN` point fetch: the depth at ``point`` from one
    backend."""
    return _fetch_bathy_backend(point, backend=backend, timeout=timeout,
                                verbose=verbose)


def _depths_along(backend, start, end, *, n_points, max_points, timeout,
                  verbose, **_request):
    """A :data:`_BATHY_CHAIN` transect fetch: ``[range_m, depth_m]`` rows
    along ``start`` → ``end`` from one backend."""
    return _fetch_bathy_transect_backend(
        start, end, n_points=n_points, max_points=max_points, backend=backend,
        timeout=timeout, verbose=verbose)


#: The bathymetry sources ``fetch_environment`` and the public fetchers
#: resolve through, each with its backends cached first (the installed GEBCO
#: grid before OpenTopoData). ``'auto'`` prefers the higher-resolution
#: multibeam synthesis, falling back to the global grid.
_BATHY_CHAIN = SourceChain(
    'bathymetry',
    providers=(
        SourceProvider('gebco', ('local', 'api'), _point_depth,
                       _depths_along),
        SourceProvider('gmrt', ('gmrt',), _point_depth, _depths_along),
        SourceProvider('emodnet_dtm', ('emodnet',), _point_depth,
                       _depths_along),
    ),
    auto=('emodnet_dtm', 'gmrt', 'gebco'),
)


def bathy_transect_plan(
    start: Coordinate, end: Coordinate, *,
    n_points: Union[int, str] = 'auto',
    max_points: int = DEFAULT_MAX_TRANSECT_POINTS,
) -> dict:
    """Resolve how many bathymetry samples a transect would take, and where,
    without fetching. Returns ``{'n_points', 'native_points', 'lats', 'lons',
    'ranges_m'}``; ``native_points`` is the uncapped native-resolution count,
    which is what :func:`fetch_bathy_transect` warns about when ``max_points``
    binds.

    Bathymetry is continuous, so ``'auto'`` targets GEBCO native resolution
    (``length / GEBCO_NATIVE_KM``) bounded by ``max_points`` — there is no
    duplicate-collapse step (cf. :func:`ssp_transect_plan`). This is where
    :func:`fetch_bathy_transect` resolves its sampling, so the two can never
    disagree.

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    n_points : int or 'auto', optional
        Samples, or ``'auto'`` for GEBCO native resolution. Default
        ``'auto'``.
    max_points : int, optional
        Cap on the sample count. Default 1000.
    """
    max_points = checked_max_points(max_points, 'bathy_transect_plan')
    length_km = central_angle(start, end) * EARTH_RADIUS_KM
    # +1 closes the fencepost: n samples span n-1 native-resolution intervals.
    native = int(np.ceil(length_km / GEBCO_NATIVE_KM)) + 1
    if n_points == 'auto':
        n = min(native, max_points)
    else:
        n = capped_n_points(
            checked_n_points(n_points, 'bathy_transect_plan', allow_auto=True),
            max_points, 'bathy_transect_plan')
    lats, lons, ranges_m = geodesic_waypoints(start, end, n)
    return {'n_points': int(n), 'native_points': native, 'lats': lats,
            'lons': lons, 'ranges_m': ranges_m}


@dataclass(frozen=True, eq=False)
class BathyGrid(ExportRecord):
    """Bathymetry on a regular lat/lon grid, as :func:`fetch_bathy_grid`
    returns it.

    Attributes
    ----------
    lats : ndarray
        Latitudes (decimal degrees), ascending, shape ``(n_lat,)``.
    lons : ndarray
        Longitudes (decimal degrees), eastward from the range's first end,
        shape ``(n_lon,)``.
    depths : ndarray
        Seafloor depth (m, positive down), shape ``(n_lat, n_lon)``; land
        cells are ``NaN``.
    provenance : DataProvenance or None
        The catalogue record of the dataset the grid was sampled from, which
        :func:`fetch_bathy_grid` always sets. ``None`` is user-supplied data:
        no catalogue source, so no credit is drawn.
    """

    lats: np.ndarray
    lons: np.ndarray
    depths: np.ndarray
    provenance: Optional[DataProvenance] = None

    _ARRAY_FIELDS = ('lats', 'lons', 'depths')
    _XARRAY_FIELDS = {'depth': 'depths', 'lat': 'lats', 'lon': 'lons'}

    def _payload(self):
        return {'depth': (self.depths, ('lat', 'lon'), 'm')}

    def _coords(self):
        return {'lat': (self.lats, 'degrees_north'),
                'lon': (self.lons, 'degrees_east')}


def _fetch_bathy_grid_backend(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    *,
    n_lat: int = 50,
    n_lon: int = 50,
    backend: str = 'api',
    dataset: str = DEFAULT_DATASET,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backend of :func:`fetch_bathy_grid`: ``backend`` is a backend token (``'api'``, ``'gmrt'``,
    ``'emodnet'``, ``'local'``); ``fetch_environment`` resolves to these."""
    _check_source(backend)
    n_lat = checked_n_points(n_lat, 'fetch_bathy_grid', name='n_lat')
    n_lon = checked_n_points(n_lon, 'fetch_bathy_grid', name='n_lon')
    # Ascending latitude for every source: gmrt/emodnet sort internally, so
    # sort here too and all four backends share one axis order.
    lat_range = (min(float(lat_range[0]), float(lat_range[1])),
                 max(float(lat_range[0]), float(lat_range[1])))
    # lon_range is directional on the 'api'/'local' paths — eastward from
    # lon_range[0], so (179, -179) samples the 2° dateline strip — and reversed
    # ends therefore sweep the long way round the globe.
    lon_span = (float(lon_range[1]) - float(lon_range[0])) % 360.0
    if lon_span > 180.0 and backend in ('api', 'local'):
        warnings.warn(
            f"fetch_bathy_grid: lon_range={tuple(lon_range)} spans "
            f"{lon_span:.0f}° eastward from lon_range[0]; swap the ends for "
            f"the {360.0 - lon_span:.0f}° strip west of it.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if backend != 'api':
        log_message('bathymetry', f"{_BACKEND_MODULES[backend][1]} "
                    f"{n_lat}×{n_lon} over lat{lat_range} lon{lon_range}",
                    verbose=verbose)
        return _backend_module(backend).region_grid(
            lat_range, lon_range, n_lat, n_lon,
            **_network_kwargs(backend, timeout, verbose))
    n_requests = -(-(n_lat * n_lon) // MAX_LOCATIONS_PER_REQUEST)   # ceil-div
    if n_requests > MAX_GRID_REQUESTS:
        raise ConfigurationError(
            f"fetch_bathy_grid: {n_lat}×{n_lon} = {n_lat * n_lon} points needs "
            f"{n_requests} requests (> {MAX_GRID_REQUESTS}).",
            remediation="Use a coarser grid, or a self-hosted instance via base_url=.",
        )
    lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    lons = lon_linspace(lon_range[0], lon_range[1], n_lon)   # eastward, dateline-safe
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    coords = list(zip(lat_mesh.ravel(), lon_mesh.ravel()))
    log_message('bathymetry', f"GEBCO grid {n_lat}×{n_lon} over "
                f"lat{lat_range} lon{lon_range}", verbose=verbose)
    elev = _fetch_elevations(coords, dataset=dataset, base_url=base_url,
                             timeout=timeout, verbose=verbose)
    depth = np.where(elev < 0.0, -elev, np.nan).reshape(n_lat, n_lon)
    return lats, lons, depth


def fetch_bathy_grid(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    *,
    n_lat: int = 50,
    n_lon: int = 50,
    source: str = 'gebco',
    dataset: str = DEFAULT_DATASET,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> BathyGrid:
    """Bathymetry on a regular lat/lon grid (batched GEBCO fetch).

    Samples an ``n_lat × n_lon`` grid spanning ``lat_range`` × ``lon_range`` and
    returns a :class:`BathyGrid`: ``lats``, ``lons`` and ``depths``, an
    ``(n_lat, n_lon)`` array in metres (positive down), with the ``source``
    dataset's provenance. **Land cells are ``NaN``** (so coastlines map
    cleanly) — unlike the point/transect fetchers, which raise on land.

    ``lats`` is **ascending** for every source, whatever order ``lat_range``
    was given in; ``lons`` runs eastward from ``lon_range[0]`` (an end west of
    the start crosses the antimeridian on the ``'api'``/``'local'`` paths,
    while ``'gmrt'``/``'emodnet'`` reject such a range).

    With ``source='api'`` points are fetched in chunks of ≤100 (the OpenTopoData
    per-call cap), so a default 50×50 grid is 25 requests; the public host allows
    ≤1000 requests/day at ≤1/s, so very large grids need a self-hosted
    ``base_url=`` or ``source='local'`` (the install-time GEBCO grid, which has no
    request cap).

    Parameters
    ----------
    lat_range, lon_range : (float, float)
        The grid's latitude and longitude spans (deg).
    n_lat, n_lon : int, optional
        Grid points along each axis, >= 2. Default 50.
    source : str, optional
        One catalogue id, as on :func:`fetch_bathy`: ``'gebco'`` (default),
        ``'gmrt'``, ``'emodnet_dtm'`` or ``'local'``; a chain is refused.
    dataset : str, optional
        OpenTopoData dataset name. Default ``'gebco2020'``.
    base_url : str, optional
        OpenTopoData service root (a self-hosted instance for large grids).
    timeout : float, optional
        Per-request network timeout in seconds.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.

    Raises
    ------
    ConfigurationError
        ``n_lat``/``n_lon`` < 2, or the grid would need too many requests.

    """
    lats, lons, depths = _cache_first(
        source, lambda backend: _fetch_bathy_grid_backend(
            lat_range, lon_range, backend=backend, n_lat=n_lat, n_lon=n_lon,
            dataset=dataset, base_url=base_url, timeout=timeout,
            verbose=verbose), who='fetch_bathy_grid', chained=False)
    return BathyGrid(lats=lats, lons=lons, depths=depths,
                     provenance=DataProvenance(
                         source=SOURCES['gebco' if source == 'local'
                                        else source]))


def transect_length(start: Coordinate, end: Coordinate) -> float:
    """Great-circle transect length (m) from ``start`` to ``end`` ``(lat, lon)``.

    Uses the same spherical geodesic as the range-dependent fetchers, so it
    equals the maximum range of the fetched bathymetry / SSP transect — size a
    receiver grid directly::

        L = uacpy.data.transect_length(A, B)
        rcv = uacpy.Receiver(depths=..., ranges=np.linspace(0.0, L, n))

    Returns ``0.0`` for coincident endpoints.

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    """
    return central_angle(start, end) * EARTH_RADIUS_M


def transect_waypoints(start: Coordinate, end: Coordinate, *, n_points: int
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lats, lons, ranges_m)`` of ``n_points`` evenly spaced waypoints on
    the great circle ``start`` → ``end``.

    The geodesic every transect fetcher samples, so data of your own —
    a regional grid, CTD casts, a measured seabed — lands on the same range
    axis as the fetched layers::

        lats, lons, r = uacpy.data.transect_waypoints(A, B, n_points=200)
        z = my_grid.interp(lat=xr.DataArray(lats), lon=xr.DataArray(lons))
        env = uacpy.data.fetch_environment(
            A, transect_to=B, bathymetry=np.column_stack([r, z]), ...)

    ``ranges_m`` runs from 0 at ``start`` to :func:`transect_length` at
    ``end``. ``n_points`` is an integer ``>= 2``. Coincident or antipodal
    endpoints raise :class:`~uacpy.core.exceptions.ConfigurationError`.

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    n_points : int
        Waypoints, >= 2.
    """
    n_points = checked_n_points(n_points, 'transect_waypoints')
    return geodesic_waypoints(start, end, n_points)


def _fetch_elevations(
    coords: List[Coordinate],
    *,
    dataset: str,
    base_url: str,
    timeout: float,
    verbose: Union[bool, str],
) -> np.ndarray:
    """Raw GEBCO elevations (m, positive up) for ``coords``, chunked to the
    OpenTopoData per-call cap. Land is positive; ocean negative.

    Consecutive chunks against the public host are spaced to honour its
    ≤1 request/s limit (see :data:`OPENTOPODATA_MIN_INTERVAL_S`); a self-hosted
    ``base_url`` lifts the limit and is not throttled."""
    throttled = base_url.rstrip('/') == DEFAULT_BASE_URL.rstrip('/')
    elevations: List[float] = []
    for i in range(0, len(coords), MAX_LOCATIONS_PER_REQUEST):
        if throttled:
            _rate_limit()
        chunk = coords[i:i + MAX_LOCATIONS_PER_REQUEST]
        elevations.extend(
            _request_chunk(chunk, dataset=dataset, base_url=base_url,
                           timeout=timeout, verbose=verbose)
        )
    return np.asarray(elevations, dtype=float)


def _rate_limit() -> None:
    """Block until at least ``OPENTOPODATA_MIN_INTERVAL_S`` has passed since the
    last public-host call, so chunked fetches stay under the ≤1 req/s limit.

    The lock spans the wait as well as the stamp, so concurrent callers queue
    one interval apart instead of each reading the same stale timestamp and
    firing together: eight threads at a 0.2 s interval left 0.2 ms between
    their calls unguarded, against the 1.4 s the limit asks for.

    The state is per-process, so a fan-out across :mod:`uacpy.parallel` (a
    ``ProcessPoolExecutor``) still gets one budget *per worker*. Before
    fetching in parallel, multiply ``OPENTOPODATA_MIN_INTERVAL_S`` by the
    worker count in each worker (4 workers: 4.0 s each, 1 request/s in all),
    or point ``base_url`` at a self-hosted OpenTopoData.
    """
    global _last_request_monotonic
    interval = OPENTOPODATA_MIN_INTERVAL_S
    with _rate_limit_lock:
        if interval > 0.0:
            wait = interval - (time.monotonic() - _last_request_monotonic)
            if wait > 0.0:
                time.sleep(wait)
        _last_request_monotonic = time.monotonic()


def _fetch_depths(
    coords: List[Coordinate],
    *,
    dataset: str,
    base_url: str,
    timeout: float,
    verbose: Union[bool, str],
) -> np.ndarray:
    """Resolve depths (m, positive down) for ``coords``, in order.

    Converts elevation to depth, raising if any point is on land.
    """
    elev = _fetch_elevations(coords, dataset=dataset, base_url=base_url,
                             timeout=timeout, verbose=verbose)
    on_land = elev >= 0.0
    if np.any(on_land):
        idx = np.flatnonzero(on_land)
        raise DataFetchError(
            f"{idx.size} of {elev.size} requested point(s) are on land "
            f"(non-negative GEBCO elevation); no water column there.",
            remediation="Move the coordinate(s) offshore. First on-land "
                        f"sample: index {int(idx[0])} at "
                        f"{coords[int(idx[0])][0]:.4f}, "
                        f"{coords[int(idx[0])][1]:.4f} "
                        f"(elevation {elev[idx[0]]:+.0f} m).",
        )
    return -elev


def _request_chunk(
    coords: List[Coordinate],
    *,
    dataset: str,
    base_url: str,
    timeout: float,
    verbose: Union[bool, str],
) -> List[float]:
    """One OpenTopoData call for up to ``MAX_LOCATIONS_PER_REQUEST`` points."""
    # OpenTopoData rejects out-of-range longitudes (e.g. 220) → normalize.
    locations = '|'.join(
        f"{lat:.6f},{normalize_lon(lon):.6f}" for lat, lon in coords
    )
    url = (
        f"{base_url.rstrip('/')}/{dataset}"
        f"?locations={urllib.parse.quote(locations)}"
    )
    payload = _http_get_json(url, timeout=timeout, verbose=verbose)

    if payload.get('status') != 'OK':
        raise DataFetchError(
            f"OpenTopoData returned status={payload.get('status')!r}: "
            f"{payload.get('error', 'no detail')}.",
            remediation="Check the dataset name and coordinate ranges "
                        "(lat in [-90, 90], lon in [-180, 180]).",
        )

    results = payload.get('results')
    if not isinstance(results, list) or len(results) != len(coords):
        raise DataFetchError(
            "OpenTopoData response missing or mismatched 'results' "
            f"(expected {len(coords)}, got "
            f"{len(results) if isinstance(results, list) else 'none'}).",
        )

    elevations = []
    for res, (lat, lon) in zip(results, coords):
        elev = res.get('elevation')
        if elev is None:
            raise DataFetchError(
                f"GEBCO has no data at {lat:.4f}, {lon:.4f} "
                "(null elevation).",
                remediation="Pick a coordinate inside the GEBCO grid.",
            )
        elevations.append(float(elev))
    return elevations


def _http_get_json(
    url: str, *, timeout: float, verbose: Union[bool, str],
) -> dict:
    """GET ``url`` and parse a JSON body, wrapping failures uniformly."""
    body = http_get(url, timeout=timeout, verbose=verbose, source='bathymetry')
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise DataFetchError(
            f"OpenTopoData returned a non-JSON body: {exc}.",
        ) from exc


def _refuse_a_dry_point(point, *, who, cache_only=False, timeout=30.0,
                       verbose=False):
    """Raise ``DataFetchError`` when GEBCO puts ``point`` on land.

    A dry point has no water column, so no sound speed, T/S column or seabed
    belongs to it, however near the nearest wet cell or sample stands. The
    elevation is read from the installed GEBCO grid, else (unless
    ``cache_only``) from OpenTopoData, as :func:`fetch_bathy` reads it. When
    neither answers, a ``FallbackWarning`` says the check was skipped and
    why, and the fetch goes on: it guards a fetch, it is not one.
    """
    lat, lon = as_coordinate(point)
    try:
        from uacpy.data import gebco_local
        elevation = float(gebco_local._grid().elevation(lat, lon))
    except (ConfigurationError, DataFetchError) as exc:
        if not _cache.is_cache_miss(exc):
            # The grid answered (a masked cell, say): no elevation to judge.
            return
        if cache_only:
            _dry_point_check_skipped(who, lat, lon, "no GEBCO grid is "
                                     "installed and the call is cache-only")
            return
        try:
            elevation = float(_fetch_elevations(
                [(lat, lon)], dataset=DEFAULT_DATASET,
                base_url=DEFAULT_BASE_URL, timeout=timeout,
                verbose=verbose)[0])
        except (ConfigurationError, DataFetchError) as live:
            _dry_point_check_skipped(
                who, lat, lon, f"no GEBCO grid is installed and "
                f"OpenTopoData did not answer ({live})")
            return
    if elevation >= 0.0:
        raise DataFetchError(
            f"{who}: GEBCO reports land (elevation {elevation:.0f} m) at "
            f"({lat:.4f}, {lon:.4f}); a dry point has no water column, so "
            f"nothing fetched for it from the nearest wet cell or sample "
            f"would describe it.",
            remediation="Pick a location offshore, or supply the profile or "
                        "seabed directly.")


def _dry_point_check_skipped(who, lat, lon, why):
    """The one warning that a dry-point check could not run."""
    warnings.warn(
        f"{who}: whether ({lat:.4f}, {lon:.4f}) is dry was not checked: "
        f"{why}; the fetch goes on, so a land point would get the nearest "
        f"wet cell's or sample's data.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
