"""GEBCO bathymetry fetch — GPS coordinates → ``Environment``-ready depth.

Phase 1 of the on-demand external-data layer: turn geographic coordinates
into a water depth (m, positive down) ready to hand to
``Environment(bathymetry=...)`` — either a single depth for one point, or a
range-dependent ``(N, 2)`` ``[range_m, depth_m]`` transect sampled along the
great-circle path between two points.

``source='api'`` (the default) serves depths from the GEBCO_2020 global grid
(~450 m resolution) as JSON via the public OpenTopoData API
(https://www.opentopodata.org/datasets/gebco2020/). No API key is needed;
the public host is rate-limited (≤100 locations per request, ≤1 request/s,
≤1000 requests/day). Point a ``base_url`` at a self-hosted OpenTopoData
instance to lift those limits. The other backends are ``'local'`` (the
install-time GEBCO 2025 grid, offline and unthrottled), ``'gmrt'`` and
``'emodnet'`` (live, higher-resolution, regional coverage).

Bathymetry is static in time, so these fetches take coordinates only — the
``date`` axis enters the data layer at the sound-speed stage, not here.
"""

import json
import threading
import time
import urllib.parse
import warnings
from typing import List, Tuple, Union

import numpy as np

from uacpy.core.constants import EARTH_RADIUS_M
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.data._geo import (
    Coordinate, as_coordinate, normalize_lon, lon_linspace,
    central_angle, geodesic_waypoints, EARTH_RADIUS_KM,
    DEFAULT_MAX_TRANSECT_POINTS, checked_max_points,
)
from uacpy.data._http import http_get
from uacpy._log import log_message

__all__ = ['fetch_bathy', 'fetch_bathy_transect', 'bathy_transect_plan',
           'fetch_bathy_grid', 'transect_length']

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

BATHY_SOURCES = ('api', 'gmrt', 'emodnet', 'local')

_last_request_monotonic = 0.0    # time.monotonic() of the last public-host call
_rate_limit_lock = threading.Lock()   # serializes the read-sleep-stamp above


def _check_source(source):
    if source not in BATHY_SOURCES:
        raise ConfigurationError(
            f"bathymetry source must be one of {BATHY_SOURCES}; got {source!r}.",
            remediation="Use 'api' (OpenTopoData/GEBCO), 'gmrt' (GMRT multibeam, "
                        "higher-res live), 'emodnet' (EMODnet DTM, ~115 m, "
                        "European seas + Caribbean), or 'local' (offline GEBCO "
                        "grid).",
        )


def fetch_bathy(
    point: Coordinate,
    *,
    source: str = 'api',
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
    source : {'api', 'gmrt', 'emodnet', 'local'}, optional
        ``'api'`` (default) queries OpenTopoData/GEBCO online; ``'gmrt'`` queries
        the GMRT multibeam synthesis (higher resolution where surveyed, CC-BY,
        live); ``'emodnet'`` queries the EMODnet Bathymetry DTM (~115 m,
        European seas + Caribbean only, CC-BY, live); ``'local'`` samples the
        install-time GEBCO grid (offline, no rate limit; see ``install.sh --data
        gebco``).
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
    """
    _check_source(source)
    if source == 'local':
        from uacpy.data import gebco_local
        return gebco_local.point_depth(point)
    if source == 'gmrt':
        from uacpy.data import gmrt_live
        return gmrt_live.point_depth(point, timeout=timeout, verbose=verbose)
    if source == 'emodnet':
        from uacpy.data import emodnet_bathy_live
        return emodnet_bathy_live.point_depth(point, timeout=timeout,
                                              verbose=verbose)
    depths = _fetch_depths(
        [as_coordinate(point)], dataset=dataset, base_url=base_url,
        timeout=timeout, verbose=verbose,
    )
    return float(depths[0])


def fetch_bathy_transect(
    start: Coordinate,
    end: Coordinate,
    *,
    n_points: Union[int, str] = 50,
    max_points: int = DEFAULT_MAX_TRANSECT_POINTS,
    source: str = 'api',
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
        native count above it) is capped to this, with a ``UserWarning``.
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
    """
    _check_source(source)
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
                f"(~{length_km / max(max_points, 1):.1f} km spacing). Raise "
                f"max_points, or use GMRT / a self-hosted OpenTopoData for "
                f"finer sampling.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    elif int(n_points) > int(max_points):
        warnings.warn(
            f"fetch_bathy_transect: n_points={n_points} exceeds "
            f"max_points={max_points}; sampling {max_points}.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    log_message(
        'bathymetry', f"sampling {n} GEBCO depths along "
        f"{ranges_m[-1] / 1000:.1f} km transect", verbose=verbose,
    )
    if source == 'local':
        from uacpy.data import gebco_local
        depths = gebco_local.depths_along(lats, lons)
    elif source == 'gmrt':
        from uacpy.data import gmrt_live
        depths = gmrt_live.depths_along(lats, lons, timeout=timeout, verbose=verbose)
    elif source == 'emodnet':
        from uacpy.data import emodnet_bathy_live
        depths = emodnet_bathy_live.depths_along(lats, lons, timeout=timeout,
                                                 verbose=verbose)
    else:
        depths = _fetch_depths(
            list(zip(lats, lons)), dataset=dataset, base_url=base_url,
            timeout=timeout, verbose=verbose,
        )
    return np.column_stack([ranges_m, depths])


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
    """
    max_points = checked_max_points(max_points, 'bathy_transect_plan')
    length_km = central_angle(start, end) * EARTH_RADIUS_KM
    # +1 closes the fencepost: n samples span n-1 native-resolution intervals.
    native = int(np.ceil(length_km / GEBCO_NATIVE_KM)) + 1
    if n_points == 'auto':
        n = min(native, max_points)
    else:
        msg = (f"bathy_transect_plan: n_points={n_points!r} is not a "
               f"sample count. Valid forms: an integer >= 2, or 'auto' "
               f"for GEBCO native resolution.")
        remediation = ("Pass n_points as an int (e.g. n_points=50) "
                       "or n_points='auto'.")
        try:
            n_requested = int(n_points)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(msg, remediation=remediation) from exc
        if n_requested != n_points:
            raise ConfigurationError(msg, remediation=remediation)
        if n_requested < 2:
            raise ConfigurationError(
                f"bathy_transect_plan: n_points must be >= 2, got {n_points}.",
                remediation="Pass n_points>=2 (or 'auto') to define a transect.",
            )
        n = min(n_requested, max_points)
    lats, lons, ranges_m = geodesic_waypoints(start, end, n)
    return {'n_points': int(n), 'native_points': native, 'lats': lats,
            'lons': lons, 'ranges_m': ranges_m}


def fetch_bathy_grid(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    *,
    n_lat: int = 50,
    n_lon: int = 50,
    source: str = 'api',
    dataset: str = DEFAULT_DATASET,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bathymetry on a regular lat/lon grid (batched GEBCO fetch).

    Samples an ``n_lat × n_lon`` grid spanning ``lat_range`` × ``lon_range`` and
    returns ``(lats, lons, depth)`` where ``depth`` is an ``(n_lat, n_lon)``
    array in metres (positive down). **Land cells are ``NaN``** (so coastlines
    map cleanly) — unlike the point/transect fetchers, which raise on land.

    ``lats`` is **ascending** for every source, whatever order ``lat_range``
    was given in; ``lons`` runs eastward from ``lon_range[0]`` (an end west of
    the start crosses the antimeridian on the ``'api'``/``'local'`` paths,
    while ``'gmrt'``/``'emodnet'`` reject such a range).

    With ``source='api'`` points are fetched in chunks of ≤100 (the OpenTopoData
    per-call cap), so a default 50×50 grid is 25 requests; the public host allows
    ≤1000 requests/day at ≤1/s, so very large grids need a self-hosted
    ``base_url=`` or ``source='local'`` (the install-time GEBCO grid, which has no
    request cap).

    Raises
    ------
    ConfigurationError
        ``n_lat``/``n_lon`` < 2, or the grid would need too many requests.
    """
    _check_source(source)
    if n_lat < 2 or n_lon < 2:
        raise ConfigurationError(
            f"fetch_bathy_grid: n_lat and n_lon must be >= 2; got {n_lat}, {n_lon}.")
    # Ascending latitude for every source: gmrt/emodnet sort internally, so
    # sort here too and all four backends share one axis order.
    lat_range = (min(float(lat_range[0]), float(lat_range[1])),
                 max(float(lat_range[0]), float(lat_range[1])))
    if source == 'local':
        from uacpy.data import gebco_local
        log_message('bathymetry', f"GEBCO grid (local) {n_lat}×{n_lon} over "
                    f"lat{lat_range} lon{lon_range}", verbose=verbose)
        return gebco_local.region_grid(lat_range, lon_range, n_lat, n_lon)
    if source == 'gmrt':
        from uacpy.data import gmrt_live
        log_message('bathymetry', f"GMRT grid {n_lat}×{n_lon} over "
                    f"lat{lat_range} lon{lon_range}", verbose=verbose)
        return gmrt_live.region_grid(lat_range, lon_range, n_lat, n_lon,
                                     timeout=timeout, verbose=verbose)
    if source == 'emodnet':
        from uacpy.data import emodnet_bathy_live
        log_message('bathymetry', f"EMODnet DTM grid {n_lat}×{n_lon} over "
                    f"lat{lat_range} lon{lon_range}", verbose=verbose)
        return emodnet_bathy_live.region_grid(lat_range, lon_range, n_lat, n_lon,
                                              timeout=timeout, verbose=verbose)
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


def transect_length(start: Coordinate, end: Coordinate) -> float:
    """Great-circle transect length (m) from ``start`` to ``end`` ``(lat, lon)``.

    Uses the same spherical geodesic as the range-dependent fetchers, so it
    equals the maximum range of the fetched bathymetry / SSP transect — size a
    receiver grid directly::

        L = uacpy.data.transect_length(A, B)
        rcv = uacpy.Receiver(depths=..., ranges=np.linspace(0.0, L, n))

    Returns ``0.0`` for coincident endpoints.
    """
    return central_angle(start, end) * EARTH_RADIUS_M


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
    ``ProcessPoolExecutor``) still gets one budget *per worker*. Lower
    ``OPENTOPODATA_MIN_INTERVAL_S``'s reciprocal by the worker count, or point
    ``base_url`` at a self-hosted OpenTopoData, before fetching in parallel.
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
